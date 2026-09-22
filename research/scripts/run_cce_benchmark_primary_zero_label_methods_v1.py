#!/usr/bin/env python3
"""Run and freeze the six-arm CCE-Benchmark zero-label experiment.

This process is deliberately gold-free.  Its CLI accepts only public inputs,
the frozen method registry, an output root, a device, and the resume flag.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Sequence
import uuid

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cce_data.estimators.cce_primary_zero_label import (  # noqa: E402
    array_sha256,
    crossfit_outcome_oof,
    crossfit_sn_mips,
    fit_fixed_fold_bert_dann_pair,
    forbidden_target_keys,
    normalize_score,
    sndr_predictions,
    source_mean_predictions,
)
from cce_data.estimators.learned_embedding import (  # noqa: E402
    learn_train_only_reward_informed_embeddings,
)
from cce_data.estimators.split_score_estimator import build_features  # noqa: E402
from run_cce_benchmark_primary_dm_bge_m3_learned_gbr import (  # noqa: E402
    ExactBgeM3Encoder,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_npz,
    canonical_json_bytes,
    canonical_json_sha256,
    load_or_build_embedding_cache,
    make_learned_config,
    read_json,
    read_jsonl,
    resolve_device,
    set_all_seeds,
    sha256_file,
)


DEFAULT_PUBLIC_ROOT = REPO_ROOT / "data/processed/cce_benchmark_single_target_v2"
DEFAULT_METHOD_REGISTRY = (
    REPO_ROOT
    / "configs/cce_benchmark_primary_method_registry_m1_m3_m6_m13_m14_zero_label_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/cce_benchmark_primary_m1_m3_m6_m13_m14_zero_label_v1"
)
TASK_REGISTRY_PATH = REPO_ROOT / "configs/cce_benchmark_primary_task_registry_v2.json"
M3_CONFIG_PATH = (
    REPO_ROOT / "configs/cce_benchmark_primary_dm_bge_m3_learned_gbr_seed012_v1.json"
)
METHOD_ORDER = (
    "M1_SOURCE_MEAN",
    "M3_FORMAL_BGE_M3_DM",
    "M6_BERT_DANN_L0P1",
    "M13_SN_MIPS_BGE",
    "M14_SNDR_BGE",
    "M6_CTRL_BERT_DM_L0",
)
PRIMARY_METHODS = METHOD_ORDER[:5]
STOCHASTIC_SEEDS = (0, 1, 2)
ROW_SCHEMA = "cce-primary-zero-label-row-prediction-v1"
VALUE_SCHEMA = "cce-primary-zero-label-panel-value-v1"
FORBIDDEN_OUTPUT_PARTS = {"private", "gold"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def registry_payload_sha256(registry: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(registry, ensure_ascii=False))
    normalized["registry_payload_sha256"] = None
    return canonical_json_sha256(normalized)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"


def write_once_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"Immutable artifact already exists with different bytes: {path}")
        return
    atomic_write_bytes(path, payload)


def write_once_json(path: Path, value: Any) -> None:
    write_once_bytes(path, _json_bytes(value))


def write_once_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    write_once_bytes(path, b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def write_exclusive_json(path: Path, value: Any) -> None:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if path.exists():
            path.unlink()
        raise


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def _resolve_snapshot(model_name: str, revision: str) -> Path:
    slug = model_name.replace("/", "--")
    huggingface_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")
    )
    snapshot = huggingface_home / "hub" / f"models--{slug}" / "snapshots" / revision
    if not snapshot.is_dir() or snapshot.name != revision:
        raise FileNotFoundError(f"Frozen local model snapshot is missing: {snapshot}")
    return snapshot.resolve()


def _validate_model_asset(spec: dict[str, Any]) -> dict[str, Any]:
    snapshot = _resolve_snapshot(str(spec["model_name"]), str(spec["revision"]))
    files: dict[str, str] = {}
    for relative, expected in (spec.get("files") or {}).items():
        path = snapshot / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen model file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen model hash drift: {path}")
        files[relative] = actual
    return {
        "model_name": spec["model_name"],
        "revision": spec["revision"],
        "snapshot": snapshot.as_posix(),
        "files": files,
    }


def _framework_versions() -> dict[str, str]:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _load_fold_map(path: Path, panel_id: str) -> dict[str, int]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    mapping: dict[str, int] = {}
    for row in rows:
        if row.get("panel_id") != panel_id:
            raise RuntimeError(f"Fold panel identity drift: {path}")
        item = str(row["item_id"])
        fold = int(row["fold"])
        if item in mapping:
            raise RuntimeError(f"Duplicate fold item: {panel_id}/{item}")
        mapping[item] = fold
    if set(mapping.values()) != {0, 1, 2, 3, 4}:
        raise RuntimeError(f"Fold IDs must be exactly 0..4: {panel_id}")
    return mapping


def _task_key(task: dict[str, Any]) -> tuple[str, str]:
    return str(task["panel_id"]), str(task["primary_task_id"])


def _row_key(row: dict[str, Any]) -> tuple[str, str, int, str, str]:
    seed = -1 if row["seed"] is None else int(row["seed"])
    return (
        str(row["method_id"]),
        str(row["panel_id"]),
        seed,
        str(row["task_id"]),
        str(row["item_id"]),
    )


def _value_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    seed = -1 if row["seed"] is None else int(row["seed"])
    return str(row["method_id"]), str(row["panel_id"]), seed, str(row["task_id"])


def validate_public_contract(
    public_root: Path,
    method_registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    method_registry = read_json(method_registry_path)
    expected_payload = method_registry.get("registry_payload_sha256")
    if registry_payload_sha256(method_registry) != expected_payload:
        raise RuntimeError("Method registry canonical payload hash mismatch")
    sidecar = method_registry_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Method registry sidecar is missing: {sidecar}")
    sidecar_text = sidecar.read_text(encoding="utf-8")
    if sha256_file(method_registry_path) not in sidecar_text:
        raise RuntimeError("Method registry file hash is not bound by its sidecar")

    task_spec = method_registry["primary_task_registry"]
    task_registry_path = (REPO_ROOT / task_spec["path"]).resolve()
    if task_registry_path != TASK_REGISTRY_PATH.resolve():
        raise RuntimeError("Primary task registry path drift")
    if sha256_file(task_registry_path) != task_spec["file_sha256"]:
        raise RuntimeError("Primary task registry file hash mismatch")
    task_registry = read_json(task_registry_path)
    if registry_payload_sha256(task_registry) != task_spec["payload_sha256"]:
        raise RuntimeError("Primary task registry payload hash mismatch")

    archive = method_registry["public_data_package"]
    archive_path = (REPO_ROOT / archive["archive_path"]).resolve()
    if not archive_path.is_file() or sha256_file(archive_path) != archive["archive_sha256"]:
        raise RuntimeError("Frozen public archive is missing or has drifted")

    versions = _framework_versions()
    if versions != method_registry["runtime_contract"]["software_versions"]:
        raise RuntimeError(
            f"Frozen runtime mismatch: actual={versions} "
            f"expected={method_registry['runtime_contract']['software_versions']}"
        )

    exclusions = {
        row["panel_id"]: row for row in method_registry["scope_selection"]["excluded_panels"]
    }
    for panel_id, status in (
        ("ayers_askdocs", "EXCLUDED_NOT_APPLICABLE"),
        ("simpeval_past", "BLOCKED"),
    ):
        if exclusions.get(panel_id, {}).get("status") != status:
            raise RuntimeError(f"Excluded-panel status drift: {panel_id}")

    tasks = task_registry.get("primary_tasks")
    if not isinstance(tasks, list) or len(tasks) != 12:
        raise RuntimeError("Exactly 12 primary tasks are required")
    if len({_task_key(task) for task in tasks}) != 12:
        raise RuntimeError("Primary task registry has duplicate panel/task keys")
    selected_panels = {str(task["panel_id"]) for task in tasks}
    if selected_panels & set(exclusions):
        raise RuntimeError("An excluded panel entered the formal task list")

    fold_specs = {
        str(row["panel_id"]): row for row in method_registry["fold_contract"]["entries"]
    }
    forbidden_tokens = method_registry["target_gold_isolation"]["forbidden_test_key_tokens"]
    exceptions = method_registry["target_gold_isolation"]["forbidden_test_key_scan"][
        "exact_allowlist_exceptions"
    ]
    bundles: list[dict[str, Any]] = []
    public_hashes: dict[str, str] = {}
    train_total = 0
    test_total = 0
    for task in tasks:
        panel_id, task_id = _task_key(task)
        files = task["public_input_files"]
        resolved: dict[str, Path] = {}
        for role in ("train", "test_inputs", "metadata", "split_manifest"):
            path = (public_root / files[role]["path"]).resolve()
            if not path.is_relative_to(public_root.resolve()) or not path.is_file():
                raise RuntimeError(f"Invalid public path: {panel_id}/{role}")
            actual = sha256_file(path)
            if actual != files[role]["sha256"]:
                raise RuntimeError(f"Public file hash mismatch: {panel_id}/{role}")
            public_hashes[path.relative_to(public_root.resolve()).as_posix()] = actual
            resolved[role] = path
        train_rows = read_jsonl(resolved["train"])
        test_rows = read_jsonl(resolved["test_inputs"])
        if len(train_rows) != int(task["expected_train_rows"]):
            raise RuntimeError(f"Source row count mismatch: {panel_id}/{task_id}")
        if len(test_rows) != int(task["expected_test_rows"]):
            raise RuntimeError(f"Target row count mismatch: {panel_id}/{task_id}")
        minimum = float(task["native_scale_min"])
        maximum = float(task["native_scale_max"])
        source_items: set[str] = set()
        source_keys: set[tuple[str, str]] = set()
        for row in train_rows:
            if row.get("panel_id") != panel_id or row.get("task_id") != task_id:
                raise RuntimeError(f"Source identity drift: {panel_id}/{task_id}")
            if row.get("split") != "train" or "score" not in row:
                raise RuntimeError(f"Source schema drift: {panel_id}/{task_id}")
            score = float(row["score"])
            if not math.isfinite(score) or not minimum <= score <= maximum:
                raise RuntimeError(f"Invalid Source score: {panel_id}/{task_id}")
            item = str(row["item_id"])
            key = item, str(row["source_responder_id"])
            if key in source_keys:
                raise RuntimeError(f"Duplicate Source key: {panel_id}/{key}")
            source_keys.add(key)
            source_items.add(item)
        target_items: set[str] = set()
        target_keys: set[tuple[str, str]] = set()
        for row in test_rows:
            if row.get("panel_id") != panel_id or row.get("task_id") != task_id:
                raise RuntimeError(f"Target identity drift: {panel_id}/{task_id}")
            if row.get("split") != "test" or row.get("target_responder_id") != task[
                "target_responder_id"
            ]:
                raise RuntimeError(f"Target schema drift: {panel_id}/{task_id}")
            leaked = forbidden_target_keys(
                row,
                forbidden_tokens=forbidden_tokens,
                exact_exceptions=exceptions,
            )
            if leaked:
                raise RuntimeError(
                    f"Forbidden Target keys in {panel_id}/{task_id}: {sorted(leaked)}"
                )
            if float(row["score_scale_min"]) != minimum or float(
                row["score_scale_max"]
            ) != maximum:
                raise RuntimeError(f"Target scale drift: {panel_id}/{task_id}")
            item = str(row["item_id"])
            key = item, str(row["target_responder_id"])
            if key in target_keys:
                raise RuntimeError(f"Duplicate Target key: {panel_id}/{key}")
            target_keys.add(key)
            target_items.add(item)
        if source_items != target_items:
            raise RuntimeError(f"Source/Target item set mismatch: {panel_id}/{task_id}")

        fold_spec = fold_specs.get(panel_id)
        if fold_spec is None:
            raise RuntimeError(f"Frozen fold file is not registered: {panel_id}")
        fold_path = (public_root / fold_spec["path"]).resolve()
        if not fold_path.is_file() or sha256_file(fold_path) != fold_spec["sha256"]:
            raise RuntimeError(f"Frozen fold hash mismatch: {panel_id}")
        fold_map = _load_fold_map(fold_path, panel_id)
        if set(fold_map) != source_items or set(fold_map) != target_items:
            raise RuntimeError(f"Frozen fold item set mismatch: {panel_id}/{task_id}")
        source_folds = np.asarray([fold_map[str(row["item_id"])] for row in train_rows])
        target_folds = np.asarray([fold_map[str(row["item_id"])] for row in test_rows])
        public_hashes[fold_path.relative_to(public_root.resolve()).as_posix()] = sha256_file(
            fold_path
        )
        bundles.append(
            {
                "task": task,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "source_folds": source_folds,
                "target_folds": target_folds,
                "fold_path": fold_path,
                "input_hashes": {
                    **{role: files[role]["sha256"] for role in files},
                    "fixed_fold": fold_spec["sha256"],
                },
            }
        )
        train_total += len(train_rows)
        test_total += len(test_rows)
    expected = method_registry["primary_task_registry"]["expected_counts"]
    if train_total != expected["train_rows_per_seed"] or test_total != expected[
        "test_keys_per_seed"
    ]:
        raise RuntimeError("Global Source/Target count mismatch")
    audit = {
        "status": "PASS",
        "checked_at_utc": utc_now(),
        "method_registry_path": method_registry_path.resolve().as_posix(),
        "method_registry_file_sha256": sha256_file(method_registry_path),
        "method_registry_payload_sha256": expected_payload,
        "primary_task_registry_path": task_registry_path.as_posix(),
        "primary_task_registry_file_sha256": sha256_file(task_registry_path),
        "primary_task_registry_payload_sha256": task_spec["payload_sha256"],
        "public_archive_path": archive_path.as_posix(),
        "public_archive_sha256": archive["archive_sha256"],
        "public_root": public_root.resolve().as_posix(),
        "public_input_hashes": dict(sorted(public_hashes.items())),
        "panels": len(bundles),
        "primary_tasks": len(bundles),
        "source_rows_per_seed": train_total,
        "target_keys_per_seed": test_total,
        "excluded_panels_absent": sorted(exclusions),
        "fixed_fold_source": "public group_to_fold.csv only",
        "runtime_versions": versions,
        "private_or_gold_path_available_to_runner": False,
    }
    return method_registry, task_registry, bundles, audit


def _cell_dir(
    output_root: Path,
    method_id: str,
    task: dict[str, Any],
    seed: int | None,
) -> Path:
    seed_name = "null" if seed is None else str(seed)
    return (
        output_root
        / "methods"
        / method_id
        / str(task["panel_id"])
        / str(task["primary_task_id"])
        / f"seed={seed_name}"
    )


def _new_attempt(output_root: Path, method_id: str, task: dict[str, Any], seed: int | None) -> Path:
    attempt = (
        output_root
        / ".staging"
        / f"{method_id}__{task['panel_id']}__{task['primary_task_id']}__{seed}__{uuid.uuid4().hex}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def _commit_attempt(
    attempt: Path,
    final: Path,
    *,
    method_id: str,
    task: dict[str, Any],
    seed: int | None,
) -> None:
    if final.exists():
        raise RuntimeError(f"Completed cell overwrite is forbidden: {final}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(attempt.rglob("*")):
        if path.is_file() and path.name != "checkpoint_manifest.json":
            relative = path.relative_to(attempt).as_posix()
            files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    required = {"panel_value.json", "fit_summary.json", "diagnostics.json"}
    if not required <= set(files):
        raise RuntimeError(f"Cell is missing required files: {attempt}")
    manifest = {
        "schema_version": "cce-primary-zero-label-cell-checkpoint-manifest-v1",
        "status": "COMPLETE",
        "method_id": method_id,
        "panel_id": task["panel_id"],
        "task_id": task["primary_task_id"],
        "seed": seed,
        "committed_at_utc": utc_now(),
        "files": files,
    }
    atomic_write_json(attempt / "checkpoint_manifest.json", manifest)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(attempt, final)


def _validate_cell(final: Path, method_id: str, task: dict[str, Any], seed: int | None) -> None:
    manifest_path = final / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Completed cell has no checkpoint manifest: {final}")
    manifest = read_json(manifest_path)
    identity = (
        manifest.get("status") == "COMPLETE"
        and manifest.get("method_id") == method_id
        and manifest.get("panel_id") == task["panel_id"]
        and manifest.get("task_id") == task["primary_task_id"]
        and manifest.get("seed") == seed
    )
    if not identity:
        raise RuntimeError(f"Completed cell identity drift: {final}")
    for relative, record in manifest["files"].items():
        path = final / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"Completed cell artifact drift: {path}")


def _skip_or_block_cell(
    final: Path,
    method_id: str,
    task: dict[str, Any],
    seed: int | None,
    resume: bool,
) -> bool:
    if not final.exists():
        return False
    if not resume:
        raise RuntimeError(f"Completed cell exists but --resume was not provided: {final}")
    _validate_cell(final, method_id, task, seed)
    return True


def _row_predictions(
    method_id: str,
    task: dict[str, Any],
    test_rows: Sequence[dict[str, Any]],
    seed: int | None,
    prediction_unit: np.ndarray,
    *,
    deterministic: bool,
) -> list[dict[str, Any]]:
    minimum = float(task["native_scale_min"])
    maximum = float(task["native_scale_max"])
    unit = np.asarray(prediction_unit, dtype=np.float64).reshape(-1)
    if len(unit) != len(test_rows) or not np.all(np.isfinite(unit)):
        raise RuntimeError(f"Invalid prediction vector: {method_id}/{task['panel_id']}/{seed}")
    if np.any(unit < 0.0) or np.any(unit > 1.0):
        raise RuntimeError(f"Unclipped unit prediction reached output: {method_id}")
    native = minimum + (maximum - minimum) * unit
    return [
        {
            "schema_version": ROW_SCHEMA,
            "method_id": method_id,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "item_id": str(row["item_id"]),
            "target_responder_id": task["target_responder_id"],
            "prediction_unit": float(unit[index]),
            "prediction_native": float(native[index]),
            "is_deterministic": deterministic,
            "contains_target_reward": False,
        }
        for index, row in enumerate(test_rows)
    ]


def _panel_value(
    method_id: str,
    task: dict[str, Any],
    seed: int | None,
    value_unit: float,
    output_object: str,
) -> dict[str, Any]:
    minimum = float(task["native_scale_min"])
    maximum = float(task["native_scale_max"])
    if not math.isfinite(value_unit) or not 0.0 <= value_unit <= 1.0:
        raise RuntimeError(f"Invalid panel value: {method_id}/{task['panel_id']}/{seed}")
    return {
        "schema_version": VALUE_SCHEMA,
        "method_id": method_id,
        "panel_id": task["panel_id"],
        "task_id": task["primary_task_id"],
        "seed": seed,
        "V_hat_unit": float(value_unit),
        "V_hat_native": float(minimum + (maximum - minimum) * value_unit),
        "output_object": output_object,
        "contains_target_reward": False,
    }


def run_m1(
    output_root: Path,
    bundle: dict[str, Any],
    resume: bool,
    logger: Callable[[str], None],
) -> None:
    method_id = "M1_SOURCE_MEAN"
    task = bundle["task"]
    final = _cell_dir(output_root, method_id, task, None)
    if _skip_or_block_cell(final, method_id, task, None, resume):
        logger(f"RESUME validated {method_id}/{task['panel_id']}/{task['primary_task_id']}")
        return
    attempt = _new_attempt(output_root, method_id, task, None)
    started = time.perf_counter()
    minimum = float(task["native_scale_min"])
    maximum = float(task["native_scale_max"])
    y = normalize_score(
        np.asarray([row["score"] for row in bundle["train_rows"]]), minimum, maximum
    )
    prediction, value = source_mean_predictions(y, len(bundle["test_rows"]))
    rows = _row_predictions(
        method_id, task, bundle["test_rows"], None, prediction, deterministic=True
    )
    ingredients = [
        {
            "schema_version": "cce-primary-zero-label-bootstrap-source-v1",
            "method_id": method_id,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": None,
            "item_id": str(row["item_id"]),
            "source_responder_id": str(row["source_responder_id"]),
            "y_source_unit": float(y[index]),
            "contains_target_reward": False,
        }
        for index, row in enumerate(bundle["train_rows"])
    ]
    atomic_write_jsonl(attempt / "row_predictions.jsonl", rows)
    atomic_write_jsonl(attempt / "source_bootstrap_ingredients.jsonl", ingredients)
    atomic_write_json(
        attempt / "panel_value.json",
        _panel_value(method_id, task, None, value, "target_row_prediction_mean"),
    )
    atomic_write_json(
        attempt / "fit_summary.json",
        {
            "schema_version": "cce-primary-zero-label-fit-summary-v1",
            "status": "COMPLETE",
            "method_id": method_id,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": None,
            "source_rows": len(y),
            "target_rows": len(rows),
            "V_hat_unit": value,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    atomic_write_json(
        attempt / "diagnostics.json",
        {
            "schema_version": "cce-primary-zero-label-diagnostics-v1",
            "status": "PASS",
            "formula": "equal Source-row arithmetic mean",
            "source_input_hashes": bundle["input_hashes"],
            "source_reward_sha256": array_sha256(y),
            "target_rewards_accessed": False,
            "deterministic": True,
        },
    )
    _commit_attempt(attempt, final, method_id=method_id, task=task, seed=None)
    logger(f"DONE {method_id}/{task['panel_id']} seconds={time.perf_counter()-started:.1f}")


def _load_base_embeddings(
    output_root: Path,
    bundle: dict[str, Any],
    encoder: ExactBgeM3Encoder,
    logger: Callable[[str], None],
) -> dict[str, Any]:
    task = bundle["task"]
    cache_root = output_root / "base_embedding_cache"
    q_train, q_test, q_manifest = load_or_build_embedding_cache(
        cache_root=cache_root,
        panel_id=str(task["panel_id"]),
        task_id=str(task["primary_task_id"]),
        field="input_text",
        train_rows=bundle["train_rows"],
        test_rows=bundle["test_rows"],
        encoder=encoder,
        logger=logger,
    )
    a_train, a_test, a_manifest = load_or_build_embedding_cache(
        cache_root=cache_root,
        panel_id=str(task["panel_id"]),
        task_id=str(task["primary_task_id"]),
        field="answer_text",
        train_rows=bundle["train_rows"],
        test_rows=bundle["test_rows"],
        encoder=encoder,
        logger=logger,
    )
    return {
        "q_train": q_train,
        "q_test": q_test,
        "a_train": a_train,
        "a_test": a_test,
        "manifests": [q_manifest, a_manifest],
    }


def run_m3(
    output_root: Path,
    bundle: dict[str, Any],
    embeddings: dict[str, Any],
    seed: int,
    m3_config: dict[str, Any],
    resume: bool,
    logger: Callable[[str], None],
) -> None:
    method_id = "M3_FORMAL_BGE_M3_DM"
    task = bundle["task"]
    final = _cell_dir(output_root, method_id, task, seed)
    if _skip_or_block_cell(final, method_id, task, seed, resume):
        logger(f"RESUME validated {method_id}/{task['panel_id']}/seed={seed}")
        return
    attempt = _new_attempt(output_root, method_id, task, seed)
    started = time.perf_counter()
    seed_audit = set_all_seeds(seed, deterministic=True)
    minimum = float(task["native_scale_min"])
    maximum = float(task["native_scale_max"])
    y = normalize_score(
        np.asarray([row["score"] for row in bundle["train_rows"]]), minimum, maximum
    )
    learned_config = make_learned_config(m3_config, seed)
    z_q_train, z_a_train, z_q_test, z_a_test, learned_diagnostics = (
        learn_train_only_reward_informed_embeddings(
            phi_x_train=embeddings["q_train"],
            phi_a_train=embeddings["a_train"],
            phi_x_eval=embeddings["q_test"],
            phi_a_eval=embeddings["a_test"],
            y_train=y,
            config=learned_config,
        )
    )
    x_train = build_features(z_q_train, z_a_train)
    x_test = build_features(z_q_test, z_a_test)
    if x_train.shape[1] != 576 or x_test.shape[1] != 576:
        raise RuntimeError("M3 feature dimension drifted from 576")
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=seed,
    )
    gbr.fit(x_train, y)
    raw = np.asarray(gbr.predict(x_test), dtype=np.float64)
    prediction = np.clip(raw, 0.0, 1.0)
    rows = _row_predictions(
        method_id, task, bundle["test_rows"], seed, prediction, deterministic=False
    )
    atomic_write_jsonl(attempt / "row_predictions.jsonl", rows)
    atomic_write_json(
        attempt / "panel_value.json",
        _panel_value(
            method_id, task, seed, float(np.mean(prediction)), "target_row_prediction_mean"
        ),
    )
    atomic_write_npz(
        attempt / "learned_feature_checkpoint.npz",
        z_q_train=z_q_train,
        z_a_train=z_a_train,
        z_q_test=z_q_test,
        z_a_test=z_a_test,
    )
    joblib.dump(gbr, attempt / "gbr_checkpoint.joblib")
    diagnostics = {
        "schema_version": "cce-primary-zero-label-diagnostics-v1",
        "status": "PASS",
        "method_id": method_id,
        "seed": seed,
        "source_input_hashes": bundle["input_hashes"],
        "source_reward_sha256": array_sha256(y),
        "base_embedding_manifest_sha256": [
            canonical_json_sha256(manifest) for manifest in embeddings["manifests"]
        ],
        "learned_embedding": learned_diagnostics,
        "legacy_source_row_validation_exception": True,
        "fixed_five_fold_claim": False,
        "feature_dimension": 576,
        "gbr": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": seed,
        },
        "prediction_unit_unclipped_min": float(np.min(raw)),
        "prediction_unit_unclipped_max": float(np.max(raw)),
        "target_rewards_accessed": False,
        "seed_audit": seed_audit,
    }
    atomic_write_json(attempt / "diagnostics.json", diagnostics)
    atomic_write_json(
        attempt / "fit_summary.json",
        {
            "schema_version": "cce-primary-zero-label-fit-summary-v1",
            "status": "COMPLETE",
            "method_id": method_id,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "source_rows": len(y),
            "target_rows": len(rows),
            "V_hat_unit": float(np.mean(prediction)),
            "best_epoch": int(learned_diagnostics["best_epoch"]),
            "epochs_trained": int(learned_diagnostics["epochs_trained"]),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _commit_attempt(attempt, final, method_id=method_id, task=task, seed=seed)
    logger(
        f"DONE {method_id}/{task['panel_id']}/seed={seed} "
        f"seconds={time.perf_counter()-started:.1f}"
    )


def run_m13(
    output_root: Path,
    bundle: dict[str, Any],
    embeddings: dict[str, Any],
    seed: int,
    resume: bool,
    logger: Callable[[str], None],
) -> None:
    method_id = "M13_SN_MIPS_BGE"
    task = bundle["task"]
    final = _cell_dir(output_root, method_id, task, seed)
    if _skip_or_block_cell(final, method_id, task, seed, resume):
        logger(f"RESUME validated {method_id}/{task['panel_id']}/seed={seed}")
        return
    attempt = _new_attempt(output_root, method_id, task, seed)
    started = time.perf_counter()
    minimum = float(task["native_scale_min"])
    maximum = float(task["native_scale_max"])
    y = normalize_score(
        np.asarray([row["score"] for row in bundle["train_rows"]]), minimum, maximum
    )
    source_features = np.concatenate(
        [
            embeddings["q_train"],
            embeddings["a_train"],
            embeddings["q_train"] * embeddings["a_train"],
        ],
        axis=1,
    )
    target_features = np.concatenate(
        [
            embeddings["q_test"],
            embeddings["a_test"],
            embeddings["q_test"] * embeddings["a_test"],
        ],
        axis=1,
    )
    fitted = crossfit_sn_mips(
        source_features=source_features,
        target_features=target_features,
        source_folds=bundle["source_folds"],
        target_folds=bundle["target_folds"],
        y_source_unit=y,
        seed=seed,
    )
    source_ingredients = [
        {
            "schema_version": "cce-primary-zero-label-m13-source-ingredient-v1",
            "method_id": method_id,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "item_id": str(row["item_id"]),
            "source_responder_id": str(row["source_responder_id"]),
            "outer_fold": int(bundle["source_folds"][index]),
            "y_source_unit": float(y[index]),
            "calibrated_p_target_oof": float(fitted["source_probability_oof"][index]),
            "raw_ratio": float(fitted["source_raw_ratio_oof"][index]),
            "clipped_weight": float(fitted["source_weight_oof"][index]),
            "contains_target_reward": False,
        }
        for index, row in enumerate(bundle["train_rows"])
    ]
    target_density = [
        {
            "schema_version": "cce-primary-zero-label-m13-target-density-oof-v1",
            "method_id": method_id,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "item_id": str(row["item_id"]),
            "target_responder_id": str(row["target_responder_id"]),
            "outer_fold": int(bundle["target_folds"][index]),
            "calibrated_p_target_oof": float(fitted["target_probability_oof"][index]),
            "contains_target_reward": False,
        }
        for index, row in enumerate(bundle["test_rows"])
    ]
    atomic_write_jsonl(attempt / "source_bootstrap_ingredients.jsonl", source_ingredients)
    atomic_write_jsonl(attempt / "target_density_oof.jsonl", target_density)
    atomic_write_json(
        attempt / "panel_value.json",
        _panel_value(method_id, task, seed, fitted["v_hat_unit"], "panel_value_only"),
    )
    atomic_write_npz(
        attempt / "density_oof_checkpoint.npz",
        source_probability_oof=fitted["source_probability_oof"],
        target_probability_oof=fitted["target_probability_oof"],
        source_raw_ratio_oof=fitted["source_raw_ratio_oof"],
        source_weight_oof=fitted["source_weight_oof"],
    )
    diagnostics = {
        "schema_version": "cce-primary-zero-label-diagnostics-v1",
        "status": "POSITIVITY_WARNING" if fitted["positivity_warning"] else "PASS",
        "method_id": method_id,
        "seed": seed,
        "source_input_hashes": bundle["input_hashes"],
        "fixed_fold_file": bundle["fold_path"].as_posix(),
        "fixed_fold_sha256": bundle["input_hashes"]["fixed_fold"],
        "outer_records": fitted["outer_records"],
        "three_fold_base_fit_sha256": fitted["three_fold_base_fit_sha256"],
        "ess": fitted["ess"],
        "ess_fraction": fitted["ess_fraction"],
        "positivity_warning": fitted["positivity_warning"],
        "oof_domain_auc": fitted["oof_domain_auc"],
        "oof_domain_accuracy": fitted["oof_domain_accuracy"],
        "weight_sum": float(np.sum(fitted["source_weight_oof"])),
        "weight_min": float(np.min(fitted["source_weight_oof"])),
        "weight_max": float(np.max(fitted["source_weight_oof"])),
        "target_rewards_accessed": False,
    }
    atomic_write_json(attempt / "diagnostics.json", diagnostics)
    atomic_write_json(
        attempt / "fit_summary.json",
        {
            "schema_version": "cce-primary-zero-label-fit-summary-v1",
            "status": "COMPLETE",
            "method_id": method_id,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "source_rows": len(source_ingredients),
            "target_rows": len(target_density),
            "row_prediction_rows": 0,
            "V_hat_unit": fitted["v_hat_unit"],
            "ess": fitted["ess"],
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _commit_attempt(attempt, final, method_id=method_id, task=task, seed=seed)
    logger(
        f"DONE {method_id}/{task['panel_id']}/seed={seed} "
        f"ESS={fitted['ess']:.1f} seconds={time.perf_counter()-started:.1f}"
    )


def _read_prediction_unit(path: Path) -> np.ndarray:
    return np.asarray([float(row["prediction_unit"]) for row in read_jsonl(path)])


def run_m14(
    output_root: Path,
    bundle: dict[str, Any],
    embeddings: dict[str, Any],
    seed: int,
    resume: bool,
    logger: Callable[[str], None],
) -> None:
    method_id = "M14_SNDR_BGE"
    task = bundle["task"]
    final = _cell_dir(output_root, method_id, task, seed)
    if _skip_or_block_cell(final, method_id, task, seed, resume):
        logger(f"RESUME validated {method_id}/{task['panel_id']}/seed={seed}")
        return
    m3_dir = _cell_dir(output_root, "M3_FORMAL_BGE_M3_DM", task, seed)
    m13_dir = _cell_dir(output_root, "M13_SN_MIPS_BGE", task, seed)
    _validate_cell(m3_dir, "M3_FORMAL_BGE_M3_DM", task, seed)
    _validate_cell(m13_dir, "M13_SN_MIPS_BGE", task, seed)
    attempt = _new_attempt(output_root, method_id, task, seed)
    started = time.perf_counter()
    minimum = float(task["native_scale_min"])
    maximum = float(task["native_scale_max"])
    y = normalize_score(
        np.asarray([row["score"] for row in bundle["train_rows"]]), minimum, maximum
    )
    outcome = crossfit_outcome_oof(
        base_question=embeddings["q_train"],
        base_answer=embeddings["a_train"],
        y_source_unit=y,
        folds=bundle["source_folds"],
        seed=seed,
    )
    m13_ingredients = read_jsonl(m13_dir / "source_bootstrap_ingredients.jsonl")
    weights = np.asarray([float(row["clipped_weight"]) for row in m13_ingredients])
    if [str(row["item_id"]) for row in m13_ingredients] != [
        str(row["item_id"]) for row in bundle["train_rows"]
    ]:
        raise RuntimeError("M14/M13 Source ingredient order drift")
    m3_target = _read_prediction_unit(m3_dir / "row_predictions.jsonl")
    prediction, correction_diagnostics = sndr_predictions(
        m3_target,
        y,
        outcome["prediction_unit"],
        weights,
    )
    rows = _row_predictions(
        method_id, task, bundle["test_rows"], seed, prediction, deterministic=False
    )
    source_ingredients = [
        {
            "schema_version": "cce-primary-zero-label-m14-source-ingredient-v1",
            "method_id": method_id,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "item_id": str(row["item_id"]),
            "source_responder_id": str(row["source_responder_id"]),
            "outer_fold": int(bundle["source_folds"][index]),
            "y_source_unit": float(y[index]),
            "m_source_oof_unit": float(outcome["prediction_unit"][index]),
            "residual_oof_unit": float(y[index] - outcome["prediction_unit"][index]),
            "m13_clipped_weight": float(weights[index]),
            "contains_target_reward": False,
        }
        for index, row in enumerate(bundle["train_rows"])
    ]
    atomic_write_jsonl(attempt / "row_predictions.jsonl", rows)
    atomic_write_jsonl(attempt / "source_bootstrap_ingredients.jsonl", source_ingredients)
    atomic_write_json(
        attempt / "panel_value.json",
        _panel_value(
            method_id, task, seed, float(np.mean(prediction)), "target_row_prediction_mean"
        ),
    )
    atomic_write_npz(
        attempt / "outcome_oof_checkpoint.npz",
        source_oof_prediction_unit=outcome["prediction_unit"],
        source_weight_oof=weights,
        source_residual_oof=y - outcome["prediction_unit"],
    )
    atomic_write_json(
        attempt / "diagnostics.json",
        {
            "schema_version": "cce-primary-zero-label-diagnostics-v1",
            "status": "PASS",
            "method_id": method_id,
            "required_display_name": "SNDR-BGE",
            "offcem_claim": False,
            "seed": seed,
            "source_input_hashes": bundle["input_hashes"],
            "fixed_fold_file": bundle["fold_path"].as_posix(),
            "fixed_fold_sha256": bundle["input_hashes"]["fixed_fold"],
            "outcome_fold_records": outcome["fold_records"],
            "outcome_oof_coverage_sha256": outcome["coverage_sha256"],
            "parallel_outer_fold_processes": outcome["parallel_outer_fold_processes"],
            "m3_cell_checkpoint_manifest_sha256": sha256_file(
                m3_dir / "checkpoint_manifest.json"
            ),
            "m13_cell_checkpoint_manifest_sha256": sha256_file(
                m13_dir / "checkpoint_manifest.json"
            ),
            **correction_diagnostics,
            "row_wise_clipping_before_mean": True,
            "target_rewards_accessed": False,
        },
    )
    atomic_write_json(
        attempt / "fit_summary.json",
        {
            "schema_version": "cce-primary-zero-label-fit-summary-v1",
            "status": "COMPLETE",
            "method_id": method_id,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "source_rows": len(source_ingredients),
            "target_rows": len(rows),
            "V_hat_unit": float(np.mean(prediction)),
            **correction_diagnostics,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _commit_attempt(attempt, final, method_id=method_id, task=task, seed=seed)
    logger(
        f"DONE {method_id}/{task['panel_id']}/seed={seed} "
        f"correction={correction_diagnostics['correction_unit']:.6f} "
        f"seconds={time.perf_counter()-started:.1f}"
    )


def _write_m6_arm(
    attempt: Path,
    *,
    method_id: str,
    task: dict[str, Any],
    bundle: dict[str, Any],
    seed: int,
    fitted: dict[str, Any],
    matched_audit: dict[str, Any],
    started: float,
) -> None:
    prediction = np.asarray(fitted["target_prediction_unit"], dtype=np.float64)
    rows = _row_predictions(
        method_id, task, bundle["test_rows"], seed, prediction, deterministic=False
    )
    atomic_write_jsonl(attempt / "row_predictions.jsonl", rows)
    atomic_write_json(
        attempt / "panel_value.json",
        _panel_value(
            method_id, task, seed, float(np.mean(prediction)), "target_row_prediction_mean"
        ),
    )
    atomic_write_npz(
        attempt / "source_prediction_checkpoint.npz",
        source_prediction_unit=fitted["source_prediction_unit"],
    )
    diagnostics = {
        "schema_version": "cce-primary-zero-label-diagnostics-v1",
        "status": "PASS",
        "method_id": method_id,
        "panel_id": task["panel_id"],
        "task_id": task["primary_task_id"],
        "seed": seed,
        "source_input_hashes": bundle["input_hashes"],
        "fixed_fold_file": bundle["fold_path"].as_posix(),
        "fixed_fold_sha256": bundle["input_hashes"]["fixed_fold"],
        "matched_primary_control_audit": matched_audit,
        **fitted["diagnostics"],
    }
    atomic_write_json(attempt / "diagnostics.json", diagnostics)
    atomic_write_json(
        attempt / "fit_summary.json",
        {
            "schema_version": "cce-primary-zero-label-fit-summary-v1",
            "status": "COMPLETE",
            "method_id": method_id,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "source_rows": len(bundle["train_rows"]),
            "source_training_rows": fitted["diagnostics"]["source_train_rows"],
            "source_validation_rows": fitted["diagnostics"]["source_validation_rows"],
            "target_rows": len(rows),
            "V_hat_unit": float(np.mean(prediction)),
            "best_epoch": fitted["diagnostics"]["best_epoch"],
            "epochs_trained": fitted["diagnostics"]["epochs_trained"],
            "elapsed_seconds_including_paired_fit": time.perf_counter() - started,
        },
    )


def run_m6_pair(
    output_root: Path,
    bundle: dict[str, Any],
    bert_snapshot: Path,
    device: str,
    seed: int,
    resume: bool,
    logger: Callable[[str], None],
) -> None:
    primary_id = "M6_BERT_DANN_L0P1"
    control_id = "M6_CTRL_BERT_DM_L0"
    task = bundle["task"]
    primary_final = _cell_dir(output_root, primary_id, task, seed)
    control_final = _cell_dir(output_root, control_id, task, seed)
    states = (primary_final.exists(), control_final.exists())
    if any(states) and not all(states):
        raise RuntimeError(f"Partial M6 matched pair exists: {task['panel_id']}/seed={seed}")
    if all(states):
        if not resume:
            raise RuntimeError("M6 matched pair exists but --resume was not provided")
        _validate_cell(primary_final, primary_id, task, seed)
        _validate_cell(control_final, control_id, task, seed)
        primary_diag = read_json(primary_final / "diagnostics.json")
        control_diag = read_json(control_final / "diagnostics.json")
        if primary_diag["initial_state_sha256"] != control_diag["initial_state_sha256"]:
            raise RuntimeError("Existing M6 matched initialization digest differs")
        if primary_diag["batch_schedule"] != control_diag["batch_schedule"]:
            raise RuntimeError("Existing M6 matched batch schedule differs")
        logger(f"RESUME validated M6 pair/{task['panel_id']}/seed={seed}")
        return
    primary_attempt = _new_attempt(output_root, primary_id, task, seed)
    control_attempt = _new_attempt(output_root, control_id, task, seed)
    started = time.perf_counter()
    validation_fold = seed
    source_folds = np.asarray(bundle["source_folds"], dtype=np.int64)
    train_indices = np.flatnonzero(source_folds != validation_fold)
    validation_indices = np.flatnonzero(source_folds == validation_fold)
    minimum = float(task["native_scale_min"])
    maximum = float(task["native_scale_max"])
    y = normalize_score(
        np.asarray([row["score"] for row in bundle["train_rows"]]), minimum, maximum
    )
    logger(
        f"START M6 pair/{task['panel_id']}/seed={seed} device={device} "
        f"train={len(train_indices)} validation={len(validation_indices)}"
    )
    fitted = fit_fixed_fold_bert_dann_pair(
        questions_source=[str(row["input_text"]) for row in bundle["train_rows"]],
        answers_source=[str(row["answer_text"]) for row in bundle["train_rows"]],
        y_source_unit=y,
        source_train_indices=train_indices,
        source_validation_indices=validation_indices,
        questions_target=[str(row["input_text"]) for row in bundle["test_rows"]],
        answers_target=[str(row["answer_text"]) for row in bundle["test_rows"]],
        model_snapshot=bert_snapshot,
        device=device,
        seed=seed,
        primary_checkpoint_path=primary_attempt / "model_checkpoint.pt",
        control_checkpoint_path=control_attempt / "model_checkpoint.pt",
    )
    _write_m6_arm(
        primary_attempt,
        method_id=primary_id,
        task=task,
        bundle=bundle,
        seed=seed,
        fitted=fitted["primary"],
        matched_audit=fitted["matched_audit"],
        started=started,
    )
    _write_m6_arm(
        control_attempt,
        method_id=control_id,
        task=task,
        bundle=bundle,
        seed=seed,
        fitted=fitted["control"],
        matched_audit=fitted["matched_audit"],
        started=started,
    )
    _commit_attempt(
        primary_attempt, primary_final, method_id=primary_id, task=task, seed=seed
    )
    _commit_attempt(
        control_attempt, control_final, method_id=control_id, task=task, seed=seed
    )
    logger(
        f"DONE M6 pair/{task['panel_id']}/seed={seed} "
        f"seconds={time.perf_counter()-started:.1f}"
    )


def _collect_and_validate(
    output_root: Path,
    bundles: Sequence[dict[str, Any]],
    method_registry: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    all_values: list[dict[str, Any]] = []
    all_ingredients: list[dict[str, Any]] = []
    by_method_rows = {method: 0 for method in METHOD_ORDER}
    by_method_values = {method: 0 for method in METHOD_ORDER}
    for method_id in METHOD_ORDER:
        seeds: Sequence[int | None] = (None,) if method_id == "M1_SOURCE_MEAN" else STOCHASTIC_SEEDS
        for bundle in bundles:
            task = bundle["task"]
            for seed in seeds:
                cell = _cell_dir(output_root, method_id, task, seed)
                _validate_cell(cell, method_id, task, seed)
                value = read_json(cell / "panel_value.json")
                if value.get("contains_target_reward") is not False:
                    raise RuntimeError(f"Target-reward marker drift: {cell}")
                all_values.append(value)
                by_method_values[method_id] += 1
                row_path = cell / "row_predictions.jsonl"
                if method_id == "M13_SN_MIPS_BGE":
                    if row_path.exists():
                        raise RuntimeError("M13 must not emit Target row predictions")
                else:
                    rows = read_jsonl(row_path)
                    expected_items = [str(row["item_id"]) for row in bundle["test_rows"]]
                    if [str(row["item_id"]) for row in rows] != expected_items:
                        raise RuntimeError(f"Target prediction key order drift: {cell}")
                    recomputed = float(np.mean([float(row["prediction_unit"]) for row in rows]))
                    if not math.isclose(recomputed, float(value["V_hat_unit"]), abs_tol=1e-12):
                        raise RuntimeError(f"Panel value does not equal row mean: {cell}")
                    all_rows.extend(rows)
                    by_method_rows[method_id] += len(rows)
                ingredient_path = cell / "source_bootstrap_ingredients.jsonl"
                if ingredient_path.is_file():
                    all_ingredients.extend(read_jsonl(ingredient_path))
    expected_by_method = method_registry["output_contract"]["expected_counts"]["by_method"]
    for method_id in METHOD_ORDER:
        if by_method_rows[method_id] != expected_by_method[method_id]["row_prediction_rows"]:
            raise RuntimeError(f"Aggregate row count mismatch: {method_id}")
        if by_method_values[method_id] != expected_by_method[method_id]["panel_value_rows"]:
            raise RuntimeError(f"Aggregate value count mismatch: {method_id}")
    if len(all_rows) != 63986 or len(all_values) != 192:
        raise RuntimeError("Global aggregate count mismatch")
    primary_rows = sum(by_method_rows[method] for method in PRIMARY_METHODS)
    primary_values = sum(by_method_values[method] for method in PRIMARY_METHODS)
    if primary_rows != 49220 or primary_values != 156:
        raise RuntimeError("Five-primary-method aggregate count mismatch")
    if len({_row_key(row) for row in all_rows}) != len(all_rows):
        raise RuntimeError("Duplicate aggregate row-prediction key")
    if len({_value_key(row) for row in all_values}) != len(all_values):
        raise RuntimeError("Duplicate aggregate panel-value key")
    return (
        sorted(all_rows, key=_row_key),
        sorted(all_values, key=_value_key),
        sorted(
            all_ingredients,
            key=lambda row: (
                METHOD_ORDER.index(str(row["method_id"])),
                str(row["panel_id"]),
                -1 if row["seed"] is None else int(row["seed"]),
                str(row["item_id"]),
                str(row.get("source_responder_id", "")),
            ),
        ),
    )


def _freeze(
    output_root: Path,
    bundles: Sequence[dict[str, Any]],
    method_registry: dict[str, Any],
    input_audit: dict[str, Any],
    model_audit: dict[str, Any],
    logger: Callable[[str], None],
) -> dict[str, Any]:
    rows, values, ingredients = _collect_and_validate(output_root, bundles, method_registry)
    write_once_jsonl(output_root / "row_predictions.jsonl", rows)
    write_once_jsonl(output_root / "panel_values.jsonl", values)
    write_once_jsonl(output_root / "bootstrap_ingredients.jsonl", ingredients)
    checkpoint_rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        seeds: Sequence[int | None] = (None,) if method_id == "M1_SOURCE_MEAN" else STOCHASTIC_SEEDS
        for bundle in bundles:
            task = bundle["task"]
            for seed in seeds:
                path = _cell_dir(output_root, method_id, task, seed) / "checkpoint_manifest.json"
                checkpoint_rows.append(
                    {
                        "method_id": method_id,
                        "panel_id": task["panel_id"],
                        "task_id": task["primary_task_id"],
                        "seed": seed,
                        "path": path.relative_to(output_root).as_posix(),
                        "sha256": sha256_file(path),
                    }
                )
    write_once_json(
        output_root / "checkpoint_manifest.json",
        {
            "schema_version": "cce-primary-zero-label-root-checkpoint-manifest-v1",
            "status": "COMPLETE",
            "cell_count": len(checkpoint_rows),
            "cells": checkpoint_rows,
        },
    )
    receipt_path = output_root / "prediction_freeze_receipt.json"
    artifact_hashes: dict[str, dict[str, Any]] = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root).as_posix()
        if relative.startswith(".staging/") or relative in {
            "run.log",
            "prediction_freeze_receipt.json",
            "prediction_freeze_receipt.sha256",
        }:
            continue
        artifact_hashes[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    root_digest = hashlib.sha256(canonical_json_bytes(artifact_hashes)).hexdigest()
    receipt = {
        "schema_version": "cce-primary-zero-label-prediction-freeze-receipt-v1",
        "status": "FROZEN_PREDICTIONS / GOLD_NOT_ACCESSED",
        "frozen_at_utc": utc_now(),
        "method_registry": {
            "path": input_audit["method_registry_path"],
            "file_sha256": input_audit["method_registry_file_sha256"],
            "payload_sha256": input_audit["method_registry_payload_sha256"],
        },
        "primary_task_registry": {
            "path": input_audit["primary_task_registry_path"],
            "file_sha256": input_audit["primary_task_registry_file_sha256"],
            "payload_sha256": input_audit["primary_task_registry_payload_sha256"],
        },
        "public_inputs": input_audit["public_input_hashes"],
        "public_archive_sha256": input_audit["public_archive_sha256"],
        "model_assets": model_audit,
        "counts": {
            "row_predictions_including_control": len(rows),
            "panel_values_including_control": len(values),
            "primary_row_predictions": 49220,
            "primary_panel_values": 156,
            "checkpoint_cells": len(checkpoint_rows),
        },
        "excluded_panels": ["ayers_askdocs", "simpeval_past"],
        "target_rewards_accessed": False,
        "private_gold_accessed": False,
        "artifacts": artifact_hashes,
        "artifact_map_canonical_sha256": root_digest,
    }
    write_exclusive_json(receipt_path, receipt)
    receipt_hash = sha256_file(receipt_path)
    write_once_bytes(
        output_root / "prediction_freeze_receipt.sha256",
        f"{receipt_hash}  prediction_freeze_receipt.json\n".encode("ascii"),
    )
    logger(f"FROZEN predictions receipt_sha256={receipt_hash}")
    return receipt


def _validate_existing_freeze(output_root: Path) -> dict[str, Any]:
    receipt_path = output_root / "prediction_freeze_receipt.json"
    sidecar_path = output_root / "prediction_freeze_receipt.sha256"
    if not receipt_path.is_file() or not sidecar_path.is_file():
        raise RuntimeError("Prediction freeze is incomplete")
    expected = sidecar_path.read_text(encoding="ascii").split()[0]
    if sha256_file(receipt_path) != expected:
        raise RuntimeError("Prediction freeze receipt hash drift")
    receipt = read_json(receipt_path)
    for relative, record in receipt["artifacts"].items():
        path = output_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"Frozen artifact drift: {path}")
    actual_map_hash = hashlib.sha256(
        canonical_json_bytes(receipt["artifacts"])
    ).hexdigest()
    if actual_map_hash != receipt["artifact_map_canonical_sha256"]:
        raise RuntimeError("Freeze artifact-map hash drift")
    return receipt


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    public_root = args.public_root.resolve()
    method_registry_path = args.method_registry.resolve()
    output_root = args.output_root.resolve()
    if any(part.lower() in FORBIDDEN_OUTPUT_PARTS for part in output_root.parts):
        raise RuntimeError("Runner output path may not contain private/gold path components")
    if output_root.exists() and (output_root / "prediction_freeze_receipt.json").exists():
        if not args.resume:
            raise RuntimeError("Frozen output already exists; overwrite is forbidden")
        return _validate_existing_freeze(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise RuntimeError("Nonempty output root requires --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_root / "run.log")
    logger("PRECHECK start")
    method_registry, task_registry, bundles, input_audit = validate_public_contract(
        public_root, method_registry_path
    )
    model_audit = {
        key: _validate_model_asset(spec)
        for key, spec in method_registry["model_assets"].items()
    }
    actual_device = resolve_device(args.device)
    implementation_files = (
        Path(__file__).resolve(),
        (REPO_ROOT / "src/cce_data/estimators/cce_primary_zero_label.py").resolve(),
        (REPO_ROOT / "src/cce_data/estimators/learned_embedding.py").resolve(),
        (REPO_ROOT / "src/cce_data/estimators/split_score_estimator.py").resolve(),
        (REPO_ROOT / "src/cce_data/estimators/dann_finetuned_encoder.py").resolve(),
        (REPO_ROOT / "scripts/run_cce_benchmark_primary_dm_bge_m3_learned_gbr.py").resolve(),
    )
    implementation_manifest = {
        "schema_version": "cce-primary-zero-label-implementation-manifest-v1",
        "status": "HASH_BOUND",
        "files": {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
            for path in implementation_files
        },
        "m14_outer_execution": (
            "five isolated processes for Source panels with at least 5000 rows; "
            "execution-only parallelism with unchanged fixed-fold fits"
        ),
        "runner_imports_evaluator": False,
        "private_or_gold_path_in_runner_call_stack": False,
    }
    implementation_path = output_root / "implementation_manifest.json"
    if implementation_path.exists():
        if read_json(implementation_path) != implementation_manifest:
            raise RuntimeError("Resume implementation hash drift")
    else:
        atomic_write_json(implementation_path, implementation_manifest)
    runtime_manifest = {
        "schema_version": "cce-primary-zero-label-runtime-manifest-v1",
        "status": "RUNNING_GOLD_FREE",
        "started_at_utc": utc_now(),
        "argv_allowed_fields_only": True,
        "runner_cli_gold_free": runner_cli_is_gold_free(),
        "requested_device": args.device,
        "actual_device": actual_device,
        "runtime_versions": _framework_versions(),
        "method_registry_file_sha256": input_audit["method_registry_file_sha256"],
        "method_registry_payload_sha256": input_audit["method_registry_payload_sha256"],
        "primary_task_registry_file_sha256": input_audit[
            "primary_task_registry_file_sha256"
        ],
        "primary_task_registry_payload_sha256": input_audit[
            "primary_task_registry_payload_sha256"
        ],
        "public_root": public_root.as_posix(),
        "input_audit": input_audit,
        "model_audit": model_audit,
        "private_gold_accessed": False,
    }
    runtime_path = output_root / "runtime_manifest.json"
    if runtime_path.exists():
        if not args.resume:
            raise RuntimeError("Runtime manifest exists without --resume")
        existing = read_json(runtime_path)
        stable_keys = (
            "runner_cli_gold_free",
            "requested_device",
            "actual_device",
            "runtime_versions",
            "method_registry_file_sha256",
            "method_registry_payload_sha256",
            "primary_task_registry_file_sha256",
            "primary_task_registry_payload_sha256",
            "public_root",
            "model_audit",
            "private_gold_accessed",
        )
        if any(existing.get(key) != runtime_manifest.get(key) for key in stable_keys):
            raise RuntimeError("Resume runtime manifest drift")
    else:
        atomic_write_json(runtime_path, runtime_manifest)
    logger(f"PRECHECK pass panels={len(bundles)} device={actual_device}")

    for bundle in bundles:
        run_m1(output_root, bundle, args.resume, logger)

    m3_config = read_json(M3_CONFIG_PATH)
    m3_provenance = next(
        method for method in method_registry["methods"] if method["method_id"] == "M3_FORMAL_BGE_M3_DM"
    )["provenance_config"]
    if sha256_file(M3_CONFIG_PATH) != m3_provenance["sha256"]:
        raise RuntimeError("Frozen M3 provenance config hash drift")
    bge_spec = method_registry["model_assets"]["bge_m3"]
    bge_config = {
        "model_name": bge_spec["model_name"],
        "revision": bge_spec["revision"],
        "max_length": 512,
        "batch_size": 8,
        "pooling": "attention_mask_mean",
        "normalization": "l2",
        "base_dimension": 1024,
        "cache_shard_unique_texts": 256,
    }
    encoder = ExactBgeM3Encoder(
        config=bge_config, requested_device=actual_device, logger=logger
    )
    embedding_by_task: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for bundle in bundles:
            task = bundle["task"]
            key = _task_key(task)
            embeddings = _load_base_embeddings(output_root, bundle, encoder, logger)
            embedding_by_task[key] = embeddings
            for seed in STOCHASTIC_SEEDS:
                run_m3(
                    output_root,
                    bundle,
                    embeddings,
                    seed,
                    m3_config,
                    args.resume,
                    logger,
                )
            for seed in STOCHASTIC_SEEDS:
                run_m13(
                    output_root, bundle, embeddings, seed, args.resume, logger
                )
            for seed in STOCHASTIC_SEEDS:
                run_m14(
                    output_root, bundle, embeddings, seed, args.resume, logger
                )
    finally:
        encoder.close()

    bert_snapshot = Path(model_audit["bert_base_uncased"]["snapshot"])
    for bundle in bundles:
        for seed in STOCHASTIC_SEEDS:
            run_m6_pair(
                output_root,
                bundle,
                bert_snapshot,
                actual_device,
                seed,
                args.resume,
                logger,
            )
    receipt = _freeze(
        output_root, bundles, method_registry, input_audit, model_audit, logger
    )
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--method-registry", type=Path, default=DEFAULT_METHOD_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def runner_cli_is_gold_free() -> bool:
    source = inspect.getsource(parse_args).lower()
    forbidden = ("--private", "--gold", "private-root", "gold-path")
    return not any(token in source for token in forbidden)


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args(argv)
    receipt = run_experiment(args)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output_root": args.output_root.resolve().as_posix(),
                "counts": receipt["counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
