"""Split-based supervised score estimator for QA benchmark tasks.

This runner trains on each task's fixed train split and predicts every row in
the requested evaluation splits. It intentionally treats answer sources as
metadata only, not as behavior/target policy roles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from cce_data.estimators.density_ratio import (
    density_ratio,
    effective_sample_size,
    fit_density_ratio_classifier,
)
from cce_data.estimators.learned_embedding import (
    LearnedEmbeddingConfig,
    learn_train_only_reward_informed_embeddings,
)


DEFAULT_SPLITS_DIR = Path("data/processed/fixed_splits")
DEFAULT_OUTPUT_DIR = Path("outputs/split_score_estimator/bge_m3_learned_train_to_dev_test_v1")
DEFAULT_WEIGHTED_RUN = Path(
    "outputs/qa_judge_weighted/phase7_v2_native_5judge_command_a032025_weighted_v1"
)
METHOD_BGE_LEARNED = "split_estimator_bge_m3_learned"
METHOD_WEIGHTED_JUDGE = "weighted_llm_judge"
ESTIMATOR_NAMES = ("DM", "MIPS", "OffCEM")
TARGET_COVARIATE_FIELDS = {"id", "question", "answer"}


@dataclass(frozen=True)
class TaskSpec:
    task: str
    display_name: str


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec("ayers_askdocs_quality", "Ayer AskDocs - Quality"),
    TaskSpec("ayers_askdocs_empathy", "Ayer AskDocs - Empathy"),
    TaskSpec("counselbench_overall", "CounselBench - Overall"),
    TaskSpec(
        "counselbench_overall_leave_gemini_out",
        "CounselBench - Overall (Leave Gemini Out)",
    ),
    TaskSpec(
        "counselbench_overall_leave_gpt4_out",
        "CounselBench - Overall (Leave GPT-4 Out)",
    ),
    TaskSpec(
        "counselbench_overall_leave_llama3_out",
        "CounselBench - Overall (Leave Llama-3 Out)",
    ),
    TaskSpec(
        "rwanda_overall11_leave_gpt4o_out",
        "Rwanda Frontline - Overall 11 (Leave GPT-4o Out)",
    ),
    TaskSpec(
        "rwanda_overall11_leave_gemini2_flash_out",
        "Rwanda Frontline - Overall 11 (Leave Gemini 2 Flash Out)",
    ),
    TaskSpec(
        "rwanda_overall11_leave_o3_mini_high_out",
        "Rwanda Frontline - Overall 11 (Leave o3-mini-high Out)",
    ),
    TaskSpec(
        "rwanda_overall11_leave_deepseek_r1_out",
        "Rwanda Frontline - Overall 11 (Leave DeepSeek-R1 Out)",
    ),
    TaskSpec("mediqa_qa", "MEDIQA-QA"),
    TaskSpec("trec_liveqa", "TREC LiveQA"),
)
TASKS_BY_NAME = {spec.task: spec for spec in TASK_SPECS}
# Preserve the task set that predated Rwanda registration when ``--task`` is omitted.
# Rwanda datasets live under a separate splits directory and are selected explicitly.
DEFAULT_TASK_NAMES = tuple(
    spec.task for spec in TASK_SPECS if not spec.task.startswith("rwanda_overall11_")
)


@dataclass(frozen=True)
class EvalSplitData:
    split: str
    rows: list[dict[str, Any]]
    start: int
    stop: int


def normalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    if normalized == "validation":
        return "dev"
    if normalized not in {"train", "dev", "test"}:
        raise ValueError(f"Unsupported split {split!r}; expected train, dev, test, or validation")
    return normalized


def make_learned_config(seed: int) -> LearnedEmbeddingConfig:
    return LearnedEmbeddingConfig(
        latent_dim=64,
        hidden_dim=64,
        merge_strategy="concat-pca",
        pca_dim=128,
        max_epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-2,
        validation_fraction=0.2,
        patience=50,
        seed=seed,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=safe_json) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=safe_json) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_split(splits_dir: Path, task: str, split: str) -> list[dict[str, Any]]:
    normalized = normalize_split_name(split)
    return read_jsonl(splits_dir / task / f"{normalized}.jsonl")


def target_covariate_view(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return the only target-side fields allowed to cross the fitting boundary.

    Evaluation rows intentionally retain gold ``score`` values on disk, but fitting,
    representation learning, and density-ratio estimation need only unlabeled question
    and answer text.  Constructing this narrow view makes that contract explicit and
    prevents metadata such as ``raw_11_scores`` from reaching any fitting API.
    """

    covariates: list[dict[str, str]] = []
    for row in rows:
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        if not question.strip() or not answer.strip():
            raise ValueError(f"Missing target covariate text for {row.get('id')}")
        covariates.append(
            {
                "id": str(row.get("id", "")),
                "question": question,
                "answer": answer,
            }
        )
    return covariates


def load_eval_covariates(
    *,
    splits_dir: Path,
    task: str,
    eval_rows_by_split: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, str]]]:
    """Load score-free target covariates and fail closed on any misalignment.

    Frozen datasets may provide ``target_covariates/<task>.jsonl`` for their
    test split.  When that directory exists, the formal score-free artifact is
    mandatory and is the only test-side text passed across the fitting
    boundary.  Older split datasets without such artifacts retain the legacy
    narrow-view fallback for compatibility.
    """

    covariates_by_split: dict[str, list[dict[str, str]]] = {}
    sources: dict[str, dict[str, str]] = {}
    frozen_dir = splits_dir / "target_covariates"
    for split, eval_rows in eval_rows_by_split.items():
        expected = target_covariate_view(eval_rows)
        frozen_path = frozen_dir / f"{task}.jsonl"
        if split == "test" and frozen_dir.exists():
            if not frozen_path.is_file():
                raise FileNotFoundError(
                    f"Missing frozen target covariates for {task}: {frozen_path}"
                )
            loaded = read_jsonl(frozen_path)
            invalid_fields = [
                sorted(set(row) - TARGET_COVARIATE_FIELDS)
                for row in loaded
                if set(row) != TARGET_COVARIATE_FIELDS
            ]
            if invalid_fields:
                raise ValueError(
                    f"Frozen target covariates for {task} must contain exactly "
                    f"{sorted(TARGET_COVARIATE_FIELDS)}; invalid extra/missing fields found"
                )
            covariates = target_covariate_view(loaded)
            if covariates != expected:
                raise ValueError(
                    f"Frozen target covariates for {task} are not aligned exactly "
                    f"with {split}.jsonl by id, question, answer, and row order"
                )
            sources[split] = {
                "kind": "frozen_score_free_file",
                "path": frozen_path.as_posix(),
            }
        else:
            covariates = expected
            sources[split] = {
                "kind": "derived_score_free_view",
                "path": f"{splits_dir / task / f'{split}.jsonl'}#id,question,answer",
            }
        covariates_by_split[split] = covariates
    return covariates_by_split, sources


def score_bounds(row: dict[str, Any]) -> tuple[float, float]:
    scale = row.get("score_scale") or {}
    min_score = float(scale.get("min", 0.0))
    max_score = float(scale.get("max", 1.0))
    if max_score <= min_score:
        raise ValueError(f"Invalid score scale for {row.get('id')}: {scale}")
    return min_score, max_score


def score_to_unit(row: dict[str, Any]) -> float:
    min_score, max_score = score_bounds(row)
    return (float(row["score"]) - min_score) / (max_score - min_score)


def unit_to_native(value: float, row: dict[str, Any]) -> float:
    min_score, max_score = score_bounds(row)
    return min_score + float(value) * (max_score - min_score)


def build_features(phi_x: np.ndarray, phi_a: np.ndarray) -> np.ndarray:
    x = np.asarray(phi_x, dtype=np.float32)
    a = np.asarray(phi_a, dtype=np.float32)
    if x.ndim != 2 or a.ndim != 2:
        raise ValueError("Feature inputs must be 2D arrays")
    if len(x) != len(a):
        raise ValueError("Question and answer feature lengths differ")
    if x.shape[1] != a.shape[1]:
        raise ValueError(f"Question/answer dims differ: {x.shape[1]} vs {a.shape[1]}")
    return np.concatenate([x, a, x * a], axis=1)


def fit_score_model(features: np.ndarray, y_train_unit: np.ndarray, seed: int) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=seed,
    )
    model.fit(features, y_train_unit)
    return model


def fit_density_weights(
    *,
    x_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
    clip: float,
    density_c: float,
    calibrate: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    clf = fit_density_ratio_classifier(
        x_eval,
        x_train,
        seed=seed,
        C=density_c,
        calibrate=calibrate,
    )
    weights = density_ratio(clf, x_train, clip=clip)
    diagnostics = {
        "clip": clip,
        "density_C": density_c,
        "density_calibrate": calibrate,
        "ess_fraction": effective_sample_size(weights),
        "mean_weight": float(weights.mean()),
        "max_weight": float(weights.max()),
    }
    return weights, diagnostics


def estimate_mips(
    *,
    x_train: np.ndarray,
    x_eval: np.ndarray,
    y_train_unit: np.ndarray,
    seed: int,
    clip: float,
    density_c: float,
    calibrate: bool,
) -> dict[str, Any]:
    weights, diagnostics = fit_density_weights(
        x_train=x_train,
        x_eval=x_eval,
        seed=seed,
        clip=clip,
        density_c=density_c,
        calibrate=calibrate,
    )
    weight_sum = float(weights.sum())
    v_hat = float((weights * y_train_unit).sum() / weight_sum) if weight_sum > 0 else 0.0
    return {"v_hat_unit": v_hat, **diagnostics}


def estimate_offcem(
    *,
    x_train: np.ndarray,
    x_eval: np.ndarray,
    y_train_unit: np.ndarray,
    seed: int,
    clip: float,
    density_c: float,
    calibrate: bool,
    fitted_model: GradientBoostingRegressor | None = None,
) -> dict[str, Any]:
    model = fitted_model or fit_score_model(x_train, y_train_unit, seed)
    eval_pred = model.predict(x_eval)
    train_pred = model.predict(x_train)
    weights, diagnostics = fit_density_weights(
        x_train=x_train,
        x_eval=x_eval,
        seed=seed,
        clip=clip,
        density_c=density_c,
        calibrate=calibrate,
    )
    residual = y_train_unit - train_pred
    weight_sum = float(weights.sum())
    correction = float((weights * residual).sum() / weight_sum) if weight_sum > 0 else 0.0
    dm_term = float(eval_pred.mean())
    y_pred_unit = eval_pred + correction
    return {
        "v_hat_unit": dm_term + correction,
        "y_pred_unit": y_pred_unit,
        "dm_term_unit": dm_term,
        "correction_unit": correction,
        **diagnostics,
    }


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {"n": 0, "mae": None, "bias": None}
    errors = [float(row["bias"]) for row in predictions]
    metrics: dict[str, Any] = {
        "n": len(predictions),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
    }
    if all(
        "gold_score" in row
        and "estimated_score" in row
        and "gold_score_normalized" in row
        and "estimated_score_normalized" in row
        for row in predictions
    ):
        gold_native = np.asarray([float(row["gold_score"]) for row in predictions])
        estimate_native = np.asarray(
            [float(row["estimated_score"]) for row in predictions]
        )
        gold_normalized = np.asarray(
            [float(row["gold_score_normalized"]) for row in predictions]
        )
        estimate_normalized = np.asarray(
            [float(row["estimated_score_normalized"]) for row in predictions]
        )
        v_true_native = float(gold_native.mean())
        v_hat_native = float(estimate_native.mean())
        bias_native = v_hat_native - v_true_native
        v_true_norm = float(gold_normalized.mean())
        v_hat_norm = float(estimate_normalized.mean())
        bias_norm = v_hat_norm - v_true_norm
        metrics.update(
            {
                # OPE target-mean point-estimate metrics.
                "v_true_native": v_true_native,
                "v_hat_native": v_hat_native,
                "bias_native": bias_native,
                "mae_native": abs(bias_native),
                "v_true_norm": v_true_norm,
                "v_hat_norm": v_hat_norm,
                "bias_norm": bias_norm,
                "mae_norm": abs(bias_norm),
                "target_mean_metric_scope": "target_mean",
                # Preserve the runner's historical row-level MAE/Bias unchanged.
                "row_mae_native": metrics["mae"],
                "row_bias_native": metrics["bias"],
            }
        )
    return metrics


def make_prediction_rows(
    *,
    task: str,
    split: str,
    rows: list[dict[str, Any]],
    predicted_unit: np.ndarray,
    method: str,
) -> list[dict[str, Any]]:
    predictions = []
    clipped = np.clip(np.asarray(predicted_unit, dtype=float), 0.0, 1.0)
    for row, estimate_unit in zip(rows, clipped, strict=True):
        estimated_score = unit_to_native(float(estimate_unit), row)
        gold_score = float(row["score"])
        bias = estimated_score - gold_score
        gold_score_normalized = score_to_unit(row)
        bias_normalized = float(estimate_unit) - gold_score_normalized
        predictions.append(
            {
                "id": row.get("id"),
                "dataset": row.get("dataset", task),
                "task": task,
                "split": split,
                "question_id": row.get("question_id"),
                "answer_id": row.get("answer_id"),
                "answer_source": row.get("answer_source"),
                "gold_score": gold_score,
                "estimated_score": estimated_score,
                "judge_score": estimated_score,
                "bias": bias,
                "absolute_error": abs(bias),
                "gold_score_normalized": gold_score_normalized,
                "estimated_score_normalized": float(estimate_unit),
                "bias_normalized": bias_normalized,
                "absolute_error_normalized": abs(bias_normalized),
                "score_scale": row.get("score_scale"),
                "method": method,
            }
        )
    return predictions


def native_range_from_rows(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("Cannot infer score range from empty rows")
    min_score, max_score = score_bounds(rows[0])
    return max_score - min_score


def aggregate_metrics_from_unit_estimate(
    *,
    task: str,
    split: str,
    rows: list[dict[str, Any]],
    method: str,
    v_hat_unit: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y_true_unit = np.asarray([score_to_unit(row) for row in rows], dtype=float)
    v_true_unit = float(y_true_unit.mean())
    score_range = native_range_from_rows(rows)
    bias_unit = float(v_hat_unit - v_true_unit)
    bias_native = bias_unit * score_range
    result = {
        "task": task,
        "split": split,
        "method": method,
        "n": len(rows),
        "mae": abs(bias_native),
        "bias": bias_native,
        "v_hat_unit": float(v_hat_unit),
        "v_true_unit": v_true_unit,
        "mae_unit": abs(bias_unit),
        "bias_unit": bias_unit,
        "metric_scope": "target_mean",
    }
    if extra:
        result.update(extra)
    return result


def estimator_diagnostics_for_metrics(estimate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in estimate.items()
        if key not in {"y_pred_unit"}
    }


SUMMARY_FIELDS = [
    "task",
    "split",
    "method",
    "n",
    # Historical row-level metrics retained for backward compatibility.
    "mae",
    "bias",
    "row_mae_native",
    "row_bias_native",
    # OPE target-mean metrics on native and normalized scales.
    "v_true_native",
    "v_hat_native",
    "bias_native",
    "mae_native",
    "v_true_norm",
    "v_hat_norm",
    "bias_norm",
    "mae_norm",
    # Weight diagnostics are populated for MIPS and OffCEM.
    "ess_fraction",
    "mean_weight",
    "max_weight",
]


def summary_row_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {field: metrics.get(field) for field in SUMMARY_FIELDS}


def text_hash(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")


def embed_with_cache(
    *,
    texts: list[str],
    embed_fn: Callable[[list[str]], np.ndarray],
    cache_dir: Path,
    key: str,
    model_label: str,
    use_cache: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{slugify(key)}.npz"
    expected_hash = text_hash(texts)
    if use_cache and path.exists():
        try:
            with np.load(path, allow_pickle=False) as payload:
                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
                cached_hash = str(payload["text_hash"].item())
                cached_model = str(payload["model_label"].item())
            if (
                cached_hash == expected_hash
                and cached_model == model_label
                and embeddings.shape[0] == len(texts)
            ):
                print(f"  cache hit {path}")
                return embeddings
        except Exception as exc:
            print(f"  ignoring unreadable cache {path}: {exc}")

    print(f"  embedding {key}: n={len(texts)}")
    embeddings = np.asarray(embed_fn(texts), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(f"Embedder returned invalid shape {embeddings.shape} for {key}")
    if use_cache:
        np.savez_compressed(
            path,
            embeddings=embeddings,
            text_hash=np.array(expected_hash),
            model_label=np.array(model_label),
        )
        print(f"  saved cache {path}")
    return embeddings


def make_hash_embedder(dim: int, seed: int = 0) -> Callable[[list[str]], np.ndarray]:
    def embed(texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), dim), dtype=np.float32)
        for index, text in enumerate(texts):
            digest = hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest()
            local_seed = int.from_bytes(digest[:8], "big", signed=False)
            rng = np.random.default_rng(local_seed)
            vector = rng.normal(size=dim).astype(np.float32)
            out[index] = vector / max(float(np.linalg.norm(vector)), 1e-12)
        return out

    return embed


def make_bge_m3_embedder(
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 8,
    max_length: int = 512,
) -> Callable[[list[str]], np.ndarray]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def embed(texts: list[str]) -> np.ndarray:
        out_dim = int(getattr(model.config, "hidden_size", 1024))
        out = np.empty((len(texts), out_dim), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out[start : start + len(batch)] = pooled.float().cpu().numpy()
        return out

    return embed


def method_label(embedder: str, learned_embedding: bool) -> str:
    if embedder == "bge" and learned_embedding:
        return METHOD_BGE_LEARNED
    suffix = "learned" if learned_embedding else "base"
    return f"split_estimator_{embedder}_{suffix}"


def method_label_for_estimator(estimator_name: str, embedder: str, learned_embedding: bool) -> str:
    if estimator_name == "DM":
        return method_label(embedder, learned_embedding)
    suffix = "learned" if learned_embedding else "base"
    embedder_label = "bge_m3" if embedder == "bge" else embedder
    prefix = {"MIPS": "split_mips", "OffCEM": "split_offcem"}[estimator_name]
    return f"{prefix}_{embedder_label}_{suffix}"


def stack_eval_rows(eval_rows_by_split: dict[str, list[dict[str, Any]]]) -> list[EvalSplitData]:
    bundles: list[EvalSplitData] = []
    start = 0
    for split, rows in eval_rows_by_split.items():
        stop = start + len(rows)
        bundles.append(EvalSplitData(split=split, rows=rows, start=start, stop=stop))
        start = stop
    return bundles


def run_task(
    *,
    task_spec: TaskSpec,
    train_rows: list[dict[str, Any]],
    eval_rows_by_split: dict[str, list[dict[str, Any]]],
    eval_covariates_by_split: dict[str, list[dict[str, str]]],
    target_covariate_sources: dict[str, dict[str, str]],
    embed_fn: Callable[[list[str]], np.ndarray],
    cache_dir: Path,
    model_label: str,
    embedder_name: str,
    estimator_names: list[str],
    seed: int,
    use_cache: bool,
    learned_embedding: bool,
    mips_clip: float,
    density_c: float,
    density_calibrate: bool,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task = task_spec.task
    eval_bundles = stack_eval_rows(eval_rows_by_split)
    eval_rows = [row for bundle in eval_bundles for row in bundle.rows]
    if not train_rows:
        raise ValueError(f"No train rows for {task}")
    if not eval_rows:
        raise ValueError(f"No eval rows for {task}")

    # Keep gold target rewards outside every fitting API. ``eval_rows`` is retained
    # only for post-fit evaluation; all target-side feature construction uses the
    # separately loaded score-free covariate artifacts/views.
    eval_covariates = [
        row
        for split in eval_rows_by_split
        for row in eval_covariates_by_split[split]
    ]
    if len(eval_covariates) != len(eval_rows):
        raise ValueError(f"Target covariate/evaluation row count mismatch for {task}")

    key_prefix = f"{task}__{model_label}"
    phi_x_train = embed_with_cache(
        texts=[row["question"] for row in train_rows],
        embed_fn=embed_fn,
        cache_dir=cache_dir,
        key=f"{key_prefix}__train_question",
        model_label=model_label,
        use_cache=use_cache,
    )
    phi_a_train = embed_with_cache(
        texts=[row["answer"] for row in train_rows],
        embed_fn=embed_fn,
        cache_dir=cache_dir,
        key=f"{key_prefix}__train_answer",
        model_label=model_label,
        use_cache=use_cache,
    )
    phi_x_eval = embed_with_cache(
        texts=[row["question"] for row in eval_covariates],
        embed_fn=embed_fn,
        cache_dir=cache_dir,
        key=f"{key_prefix}__eval_{'-'.join(eval_rows_by_split)}_question",
        model_label=model_label,
        use_cache=use_cache,
    )
    phi_a_eval = embed_with_cache(
        texts=[row["answer"] for row in eval_covariates],
        embed_fn=embed_fn,
        cache_dir=cache_dir,
        key=f"{key_prefix}__eval_{'-'.join(eval_rows_by_split)}_answer",
        model_label=model_label,
        use_cache=use_cache,
    )
    y_train_unit = np.asarray([score_to_unit(row) for row in train_rows], dtype=np.float32)

    diagnostics: dict[str, Any] = {
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "train_answer_sources": sorted({str(row.get("answer_source")) for row in train_rows}),
        "eval_splits": {split: len(rows) for split, rows in eval_rows_by_split.items()},
        "answer_source_used_as_feature": False,
        "score_transform": "normalized_to_0_1",
        "target_reward_usage": "evaluation_only",
        "uses_eval_rewards_for_fitting": False,
        "target_covariate_fields_available_to_fitting": ["question", "answer"],
        "target_reward_fields_available_to_fitting": [],
        "target_covariate_sources": target_covariate_sources,
        "fitting_reward_source": "train.score_only",
    }
    if learned_embedding:
        config = make_learned_config(seed)
        (
            phi_x_train,
            phi_a_train,
            phi_x_eval,
            phi_a_eval,
            learned_diag,
        ) = learn_train_only_reward_informed_embeddings(
            phi_x_train=phi_x_train,
            phi_a_train=phi_a_train,
            phi_x_eval=phi_x_eval,
            phi_a_eval=phi_a_eval,
            y_train=y_train_unit,
            config=config,
        )
        diagnostics["learned_embedding"] = learned_diag

    x_train = build_features(phi_x_train, phi_a_train)
    x_eval = build_features(phi_x_eval, phi_a_eval)
    model = fit_score_model(x_train, y_train_unit, seed=seed)
    predicted_eval_unit = model.predict(x_eval)

    summary_rows = []
    split_metrics: dict[str, Any] = {}
    for bundle in eval_bundles:
        split_pred_unit = predicted_eval_unit[bundle.start : bundle.stop]
        split_x_eval = x_eval[bundle.start : bundle.stop]
        split_method_metrics: dict[str, Any] = {}
        if "DM" in estimator_names:
            dm_method = method_label_for_estimator("DM", embedder_name, learned_embedding)
            predictions = make_prediction_rows(
                task=task,
                split=bundle.split,
                rows=bundle.rows,
                predicted_unit=split_pred_unit,
                method=dm_method,
            )
            metrics = compute_metrics(predictions)
            metrics.update(
                {
                    "task": task,
                    "display_name": task_spec.display_name,
                    "split": bundle.split,
                    "method": dm_method,
                    "n_train": len(train_rows),
                    "metric_scope": "row_score",
                }
            )
            write_jsonl(output_dir / task / bundle.split / "predictions.jsonl", predictions)
            write_json(output_dir / task / bundle.split / "metrics.json", metrics)
            summary_rows.append(summary_row_from_metrics(metrics))
            split_method_metrics[dm_method] = metrics

        for estimator_name in estimator_names:
            if estimator_name == "DM":
                continue
            aggregate_method = method_label_for_estimator(
                estimator_name, embedder_name, learned_embedding
            )
            if estimator_name == "MIPS":
                estimate = estimate_mips(
                    x_train=x_train,
                    x_eval=split_x_eval,
                    y_train_unit=y_train_unit,
                    seed=seed,
                    clip=mips_clip,
                    density_c=density_c,
                    calibrate=density_calibrate,
                )
                predicted_unit = np.full(len(bundle.rows), estimate["v_hat_unit"], dtype=float)
            elif estimator_name == "OffCEM":
                estimate = estimate_offcem(
                    x_train=x_train,
                    x_eval=split_x_eval,
                    y_train_unit=y_train_unit,
                    seed=seed,
                    clip=mips_clip,
                    density_c=density_c,
                    calibrate=density_calibrate,
                    fitted_model=model,
                )
                predicted_unit = np.asarray(estimate["y_pred_unit"], dtype=float)
            else:
                raise ValueError(f"Unsupported estimator {estimator_name}")

            predictions = make_prediction_rows(
                task=task,
                split=bundle.split,
                rows=bundle.rows,
                predicted_unit=predicted_unit,
                method=aggregate_method,
            )
            metrics = compute_metrics(predictions)
            metrics.update(
                {
                    "display_name": task_spec.display_name,
                    "task": task,
                    "split": bundle.split,
                    "method": aggregate_method,
                    "n_train": len(train_rows),
                    "metric_scope": "row_score",
                    **estimator_diagnostics_for_metrics(estimate),
                }
            )
            method_dir = output_dir / task / bundle.split / aggregate_method
            write_jsonl(method_dir / "predictions.jsonl", predictions)
            write_json(method_dir / "metrics.json", metrics)
            summary_rows.append(summary_row_from_metrics(metrics))
            split_method_metrics[aggregate_method] = metrics
        split_metrics[bundle.split] = split_method_metrics

    diagnostics["feature_dim"] = int(x_train.shape[1])
    diagnostics["splits"] = split_metrics
    return summary_rows, diagnostics


def load_weighted_judge_summary(
    weighted_run: Path,
    *,
    metric_scope: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    path = weighted_run / "summary.csv"
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["task"]
            if row.get("calibration_n"):
                dev_bias = float(row["calibration_bias"])
                dev_mae = abs(dev_bias) if metric_scope == "target_mean" else float(row["calibration_mae"])
                rows[(task, "dev")] = {
                    "task": task,
                    "split": "dev",
                    "method": METHOD_WEIGHTED_JUDGE,
                    "n": int(float(row["calibration_n"])),
                    "mae": dev_mae,
                    "bias": dev_bias,
                }
            test_bias = float(row["bias"])
            test_mae = abs(test_bias) if metric_scope == "target_mean" else float(row["mae"])
            rows[(task, "test")] = {
                "task": task,
                "split": "test",
                "method": METHOD_WEIGHTED_JUDGE,
                "n": int(float(row["n"])),
                "mae": test_mae,
                "bias": test_bias,
            }
    return rows


def run_split_score_estimator(
    *,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    weighted_run: Path = DEFAULT_WEIGHTED_RUN,
    tasks: list[str] | None = None,
    eval_splits: list[str] | None = None,
    seed: int = 0,
    embedder: str = "bge",
    hash_dim: int = 256,
    use_cache: bool = True,
    learned_embedding: bool = True,
    estimators: list[str] | None = None,
    cache_dir: Path | None = None,
    mips_clip: float = 20.0,
    density_c: float = 0.1,
    density_calibrate: bool = True,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    model_label: str | None = None,
) -> dict[str, Any]:
    started = time.time()
    selected_tasks = [TASKS_BY_NAME[name] for name in validate_tasks(tasks)]
    selected_splits = normalize_eval_splits(eval_splits or ["dev", "test"])
    selected_estimators = validate_estimators(estimators)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_dir = cache_dir or output_dir / "embedding_cache"

    if embed_fn is None:
        if embedder == "bge":
            embed_fn = make_bge_m3_embedder()
            model_label = model_label or "BAAI_bge-m3"
        elif embedder == "hash":
            embed_fn = make_hash_embedder(hash_dim, seed=seed)
            model_label = model_label or f"hash_dim{hash_dim}"
        else:
            raise ValueError(f"Unknown embedder {embedder!r}")
    else:
        model_label = model_label or embedder

    all_summary_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for task_spec in selected_tasks:
        print()
        print("=" * 80)
        print(task_spec.display_name)
        print("=" * 80)
        train_rows = load_split(splits_dir, task_spec.task, "train")
        eval_rows_by_split = {
            split: load_split(splits_dir, task_spec.task, split)
            for split in selected_splits
        }
        eval_covariates_by_split, target_covariate_sources = load_eval_covariates(
            splits_dir=splits_dir,
            task=task_spec.task,
            eval_rows_by_split=eval_rows_by_split,
        )
        summary_rows, task_diagnostics = run_task(
            task_spec=task_spec,
            train_rows=train_rows,
            eval_rows_by_split=eval_rows_by_split,
            eval_covariates_by_split=eval_covariates_by_split,
            target_covariate_sources=target_covariate_sources,
            embed_fn=embed_fn,
            cache_dir=embedding_cache_dir,
            model_label=model_label,
            embedder_name=embedder,
            estimator_names=selected_estimators,
            seed=seed,
            use_cache=use_cache,
            learned_embedding=learned_embedding,
            mips_clip=mips_clip,
            density_c=density_c,
            density_calibrate=density_calibrate,
            output_dir=output_dir,
        )
        all_summary_rows.extend(summary_rows)
        diagnostics[task_spec.task] = task_diagnostics

    all_summary_rows.sort(key=lambda row: (row["task"], row["split"], row["method"]))
    write_csv(output_dir / "summary.csv", all_summary_rows, SUMMARY_FIELDS)

    comparison_metric_scope = "row_score"
    weighted_rows = load_weighted_judge_summary(
        weighted_run,
        metric_scope=comparison_metric_scope,
    )
    comparison_rows = list(all_summary_rows)
    for task_spec in selected_tasks:
        for split in selected_splits:
            weighted = weighted_rows.get((task_spec.task, split))
            if weighted is not None:
                comparison_rows.append(weighted)
    comparison_rows.sort(key=lambda row: (row["task"], row["split"], row["method"]))
    write_csv(output_dir / "comparison_with_weighted_judge.csv", comparison_rows, SUMMARY_FIELDS)

    manifest = {
        "created_at_unix": time.time(),
        "wall_time_seconds": time.time() - started,
        "splits_dir": splits_dir.as_posix(),
        "weighted_run": weighted_run.as_posix(),
        "output_dir": output_dir.as_posix(),
        "tasks": [spec.task for spec in selected_tasks],
        "eval_splits": selected_splits,
        "estimators": selected_estimators,
        "logged_split": "train",
        "embedder": model_label,
        "learned_embedding": None if not learned_embedding else asdict(make_learned_config(seed)),
        "methods": [
            method_label_for_estimator(name, embedder, learned_embedding)
            for name in selected_estimators
        ],
        "prediction_head": {
            "type": "GradientBoostingRegressor",
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": seed,
        },
        "mips_clip": mips_clip,
        "density_C": density_c,
        "density_calibrate": density_calibrate,
        "comparison_metric_scope": comparison_metric_scope,
        "embedding_cache_dir": embedding_cache_dir.as_posix(),
        "summary_metrics": [
            "mae",
            "bias",
            "v_true_native",
            "v_hat_native",
            "bias_native",
            "mae_native",
            "v_true_norm",
            "v_hat_norm",
            "bias_norm",
            "mae_norm",
        ],
        "metric_definitions": {
            "mae": "historical row-level mean(abs(estimated_score - gold_score)), native scale",
            "bias": "historical row-level mean(estimated_score - gold_score), native scale",
            "v_true_native": "mean target gold score on the native scale; evaluation only",
            "v_hat_native": "mean target estimated score on the native scale",
            "bias_native": "v_hat_native - v_true_native",
            "mae_native": "abs(bias_native), the target-mean point-estimate error",
            "normalized_metrics": "native values transformed by (score - min) / (max - min)",
        },
        "target_reward_isolation": {
            "usage": "evaluation_only",
            "uses_eval_rewards_for_fitting": False,
            "target_covariate_fields_available_to_fitting": ["question", "answer"],
            "target_reward_fields_available_to_fitting": [],
            "target_covariate_sources": {
                task: task_diagnostics["target_covariate_sources"]
                for task, task_diagnostics in diagnostics.items()
            },
        },
        "diagnostics": diagnostics,
    }
    write_json(output_dir / "manifest.json", manifest)
    print()
    print(f"Saved {output_dir / 'summary.csv'}")
    print(f"Saved {output_dir / 'comparison_with_weighted_judge.csv'}")
    print(f"Saved {output_dir / 'manifest.json'}")
    return manifest


def validate_tasks(tasks: list[str] | None) -> list[str]:
    if not tasks:
        return list(DEFAULT_TASK_NAMES)
    unknown = sorted(set(tasks) - set(TASKS_BY_NAME))
    if unknown:
        raise ValueError(f"Unknown task(s): {', '.join(unknown)}")
    return tasks


def validate_estimators(estimators: list[str] | None) -> list[str]:
    if not estimators:
        return ["DM"]
    unknown = sorted(set(estimators) - set(ESTIMATOR_NAMES))
    if unknown:
        raise ValueError(f"Unknown estimator(s): {', '.join(unknown)}")
    normalized = []
    for estimator_name in estimators:
        if estimator_name not in normalized:
            normalized.append(estimator_name)
    return normalized


def normalize_eval_splits(eval_splits: list[str]) -> list[str]:
    normalized = []
    for split in eval_splits:
        value = normalize_split_name(split)
        if value == "train":
            raise ValueError("Train is reserved for fitting and cannot be an eval split")
        if value not in normalized:
            normalized.append(value)
    return normalized


def safe_json(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weighted-run", type=Path, default=DEFAULT_WEIGHTED_RUN)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--eval-split", action="append", dest="eval_splits")
    parser.add_argument("--estimator", action="append", dest="estimators", choices=ESTIMATOR_NAMES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedder", choices=["bge", "hash"], default="bge")
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--learned-embedding", dest="learned_embedding", action="store_true")
    parser.add_argument("--no-learned-embedding", dest="learned_embedding", action="store_false")
    parser.add_argument("--mips-clip", type=float, default=20.0)
    parser.add_argument("--density-C", type=float, default=0.1)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.set_defaults(learned_embedding=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_split_score_estimator(
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        weighted_run=args.weighted_run,
        tasks=args.tasks,
        eval_splits=args.eval_splits,
        seed=args.seed,
        embedder=args.embedder,
        hash_dim=args.hash_dim,
        use_cache=not args.no_cache,
        learned_embedding=args.learned_embedding,
        estimators=args.estimators,
        cache_dir=args.cache_dir,
        mips_clip=args.mips_clip,
        density_c=args.density_C,
        density_calibrate=not args.no_calibrate,
    )


if __name__ == "__main__":
    main()
