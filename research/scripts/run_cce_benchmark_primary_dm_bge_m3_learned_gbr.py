#!/usr/bin/env python3
"""Run the frozen 11-task CCE primary Direct-Method experiment.

This process is gold-free by construction.  Its command-line interface accepts
the public package, frozen registry, model config, and an output directory only.
It never imports evaluation code and has no private-gold or target-reward path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cce_data.estimators.learned_embedding import (  # noqa: E402
    LearnedEmbeddingConfig,
    learn_train_only_reward_informed_embeddings,
)
from cce_data.estimators.split_score_estimator import build_features  # noqa: E402


DEFAULT_PUBLIC_ROOT = REPO_ROOT / "data/processed/cce_benchmark_single_target_v1"
DEFAULT_REGISTRY = REPO_ROOT / "configs/cce_benchmark_primary_task_registry_v1.json"
DEFAULT_CONFIG = (
    REPO_ROOT / "configs/cce_benchmark_primary_dm_bge_m3_learned_gbr_seed012_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/cce_benchmark_primary_dm_bge_m3_learned_gbr_seed012_v1"
)
FORBIDDEN_TEST_FIELDS = {
    "score",
    "reward",
    "label",
    "gold",
    "rating",
    "mean_score",
    "rank",
    "score_distribution",
    "raw_rater_count",
}
EXPECTED_SEEDS = [0, 1, 2]
PREDICTION_SCHEMA = "cce-primary-dm-prediction-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def safe_json(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=safe_json,
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    payload = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in rows
    )
    atomic_write_bytes(path, payload)


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez(temporary_name, **arrays)
        with open(temporary_name, "rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        rendered = f"{utc_now()} {message}"
        print(rendered, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
            handle.flush()


def recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(recursive_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(recursive_keys(nested))
    return keys


def forbidden_test_keys(row: dict[str, Any]) -> set[str]:
    keys = recursive_keys(row)
    forbidden = keys & FORBIDDEN_TEST_FIELDS
    for key in keys:
        lowered = key.lower()
        if any(token in lowered for token in ("rater", "annotator", "language_fidelity")):
            forbidden.add(key)
        if any(
            token in lowered
            for token in ("score_std", "score_variance", "score_quantile", "score_histogram")
        ):
            forbidden.add(key)
    return forbidden


def registry_payload_sha256(registry: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(registry, ensure_ascii=False))
    normalized["registry_payload_sha256"] = None
    return canonical_json_sha256(normalized)


def validate_fixed_config(config: dict[str, Any]) -> None:
    expected = config.get("expected_counts") or {}
    if config.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("Formal seeds must be exactly [0, 1, 2]")
    if expected != {
        "dataset_families": 10,
        "panels": 11,
        "primary_tasks": 11,
        "train_rows_per_seed": 37692,
        "test_keys_per_seed": 4872,
        "task_seed_cells": 33,
        "prediction_rows": 14616,
    }:
        raise ValueError(f"Experiment count contract drift: {expected}")
    bge = config.get("bge_m3") or {}
    required_bge = {
        "model_name": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "local_files_only": True,
        "network_fallback": False,
        "fallback_encoder": None,
        "max_length": 512,
        "batch_size": 8,
        "pooling": "attention_mask_mean",
        "normalization": "l2",
        "base_dimension": 1024,
    }
    for key, value in required_bge.items():
        if bge.get(key) != value:
            raise ValueError(f"BGE-M3 contract drift at {key}: {bge.get(key)!r} != {value!r}")
    learned = config.get("learned_embedding") or {}
    required_learned = {
        "api": "learn_train_only_reward_informed_embeddings",
        "latent_dim": 64,
        "hidden_dim": 64,
        "merge_strategy": "concat-pca",
        "pca_dim": 128,
        "max_epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "dropout": 0.0,
        "validation_fraction": 0.2,
        "patience": 50,
        "normalize": True,
        "representation_dimension": 192,
    }
    for key, value in required_learned.items():
        if learned.get(key) != value:
            raise ValueError(
                f"Learned-embedding contract drift at {key}: {learned.get(key)!r} != {value!r}"
            )
    gbr = config.get("gradient_boosting_regressor") or {}
    if gbr != {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "random_state": "experiment_seed",
    }:
        raise ValueError(f"GBR contract drift: {gbr}")
    dm = config.get("direct_method") or {}
    if dm.get("feature_dimension") != 576:
        raise ValueError("Direct-Method feature dimension must be 576")
    isolation = config.get("target_gold_isolation") or {}
    required_false = (
        "runner_accepts_private_root",
        "runner_accepts_gold_path",
        "runner_reads_evaluated_predictions",
        "target_rewards_for_fitting",
        "target_rewards_for_representation",
        "target_rewards_for_validation_or_early_stopping",
        "target_rewards_for_model_or_seed_selection",
        "target_side_calibration",
    )
    if any(isolation.get(key) is not False for key in required_false):
        raise ValueError("Target-gold isolation config is not fail-closed")


def task_key(task: dict[str, Any]) -> tuple[str, str]:
    return str(task["panel_id"]), str(task["primary_task_id"])


def prediction_base_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row["dataset_family_id"]),
        str(row["dataset_id"]),
        str(row["panel_id"]),
        str(row["task_id"]),
        str(row["item_id"]),
        str(row["target_responder_id"]),
    )


def validate_public_contract(
    *,
    public_root: Path,
    registry_path: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    public_root = public_root.resolve()
    registry_path = registry_path.resolve()
    config_path = config_path.resolve()
    registry = read_json(registry_path)
    config = read_json(config_path)
    validate_fixed_config(config)
    actual_registry_hash = sha256_file(registry_path)
    if config.get("primary_task_registry_sha256") != actual_registry_hash:
        raise ValueError("Config registry SHA256 does not match the frozen registry file")
    computed_payload_hash = registry_payload_sha256(registry)
    if registry.get("registry_payload_sha256") != computed_payload_hash:
        raise ValueError("Registry canonical payload SHA256 mismatch")
    if registry.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("Registry seeds must be exactly [0, 1, 2]")
    if registry.get("expected_counts") != config.get("expected_counts"):
        raise ValueError("Registry and config expected counts differ")
    tasks = registry.get("primary_tasks")
    if not isinstance(tasks, list) or len(tasks) != 11:
        raise ValueError("Primary registry must contain exactly 11 tasks")
    if len({task_key(task) for task in tasks}) != 11:
        raise ValueError("Primary registry contains duplicate panel/task keys")
    if len({str(task["panel_id"]) for task in tasks}) != 11:
        raise ValueError("Primary registry must contain exactly 11 panels")
    if len({str(task["dataset_family_id"]) for task in tasks}) != 10:
        raise ValueError("Primary registry must contain exactly 10 dataset families")
    if any(task.get("seeds") != EXPECTED_SEEDS for task in tasks):
        raise ValueError("Every primary task must use seeds [0, 1, 2]")
    exclusions = {str(row.get("panel_id")): row for row in registry.get("exclusions", [])}
    if exclusions.get("ayers_askdocs", {}).get("status") != "EXCLUDED_NOT_APPLICABLE":
        raise ValueError("Ayers must be EXCLUDED_NOT_APPLICABLE")
    if exclusions.get("simpeval_past", {}).get("status") != "BLOCKED":
        raise ValueError("SimpEval_past must remain BLOCKED")
    if {str(task["panel_id"]) for task in tasks} & {"ayers_askdocs", "simpeval_past"}:
        raise ValueError("Excluded panels entered the primary registry")

    score_registry = read_json(public_root / "score_contract_registry.json")
    score_contracts = {
        (str(row["panel_id"]), str(row["task_id"])): row
        for row in score_registry.get("tasks", [])
    }
    benchmark_manifest = read_json(public_root / "benchmark_manifest.json")
    if benchmark_manifest.get("generalization_scope") != registry.get("generalization_scope"):
        raise ValueError("Public benchmark generalization scope drift")

    loaded: list[dict[str, Any]] = []
    public_hashes: dict[str, str] = {}
    train_total = 0
    test_total = 0
    all_secondary: set[tuple[str, str]] = set()
    all_primary: set[tuple[str, str]] = set()
    for position, task in enumerate(tasks):
        panel_id, primary_task_id = task_key(task)
        all_primary.add((panel_id, primary_task_id))
        all_secondary.update(
            (panel_id, str(value)) for value in task.get("secondary_diagnostic_tasks", [])
        )
        if task.get("inclusion_status") != "INCLUDED_PRIMARY":
            raise ValueError(f"{panel_id}/{primary_task_id}: not marked INCLUDED_PRIMARY")
        if task.get("higher_is_better") is not True:
            raise ValueError(f"{panel_id}/{primary_task_id}: selected task is not higher-is-better")
        if task.get("native_scale_max", 0) <= task.get("native_scale_min", 0):
            raise ValueError(f"{panel_id}/{primary_task_id}: invalid native scale")
        files = task.get("public_input_files") or {}
        resolved: dict[str, Path] = {}
        for role in ("train", "test_inputs", "metadata", "split_manifest"):
            spec = files.get(role) or {}
            path = (public_root / str(spec.get("path", ""))).resolve()
            if not path.is_relative_to(public_root) or not path.is_file():
                raise ValueError(f"{panel_id}/{primary_task_id}: invalid public {role} path")
            actual_hash = sha256_file(path)
            if actual_hash != spec.get("sha256"):
                raise ValueError(f"{panel_id}/{primary_task_id}: public {role} SHA256 mismatch")
            public_hashes[str(path.relative_to(public_root))] = actual_hash
            resolved[role] = path
        metadata = read_json(resolved["metadata"])
        split_manifest = read_json(resolved["split_manifest"])
        if metadata.get("status") != "READY" or split_manifest.get("status") != "READY":
            raise ValueError(f"{panel_id}: public panel is not READY")
        if metadata.get("panel_id") != panel_id or split_manifest.get("panel_id") != panel_id:
            raise ValueError(f"{panel_id}: public identity drift")
        target = str(task["target_responder_id"])
        sources = {str(value) for value in metadata.get("source_responder_ids", [])}
        if metadata.get("target_responder_id") != target or target in sources:
            raise ValueError(f"{panel_id}: target/source responder isolation failure")
        if split_manifest.get("generalization_scope") != registry.get("generalization_scope"):
            raise ValueError(f"{panel_id}: split semantics drift")
        contract = score_contracts.get((panel_id, primary_task_id))
        if contract is None:
            raise ValueError(f"{panel_id}/{primary_task_id}: score contract is absent")
        contract_checks = {
            "score_field": contract.get("score_field") == task.get("score_field"),
            "scale_min": float(contract.get("score_scale_min"))
            == float(task.get("native_scale_min")),
            "scale_max": float(contract.get("score_scale_max"))
            == float(task.get("native_scale_max")),
            "higher_is_better": contract.get("higher_is_better") is True,
            "score_provenance": contract.get("score_provenance")
            == task.get("score_provenance"),
        }
        if not all(contract_checks.values()):
            raise ValueError(
                f"{panel_id}/{primary_task_id}: score-contract mismatch: {contract_checks}"
            )
        train_rows = read_jsonl(resolved["train"])
        test_rows = read_jsonl(resolved["test_inputs"])
        if len(train_rows) != int(task["expected_train_rows"]):
            raise ValueError(f"{panel_id}/{primary_task_id}: train-row count mismatch")
        if len(test_rows) != int(task["expected_test_rows"]):
            raise ValueError(f"{panel_id}/{primary_task_id}: test-row count mismatch")
        minimum = float(task["native_scale_min"])
        maximum = float(task["native_scale_max"])
        train_keys: set[tuple[str, str]] = set()
        for row in train_rows:
            if (
                row.get("dataset_id") != task.get("dataset_id")
                or row.get("panel_id") != panel_id
                or row.get("task_id") != primary_task_id
                or row.get("split") != "train"
            ):
                raise ValueError(f"{panel_id}/{primary_task_id}: train identity drift")
            source = str(row.get("source_responder_id"))
            if source not in sources or source == target:
                raise ValueError(f"{panel_id}/{primary_task_id}: invalid train responder")
            key = (str(row.get("item_id")), source)
            if key in train_keys:
                raise ValueError(f"{panel_id}/{primary_task_id}: duplicate train key")
            train_keys.add(key)
            score = row.get("score")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ValueError(f"{panel_id}/{primary_task_id}: non-finite Source score")
            if not minimum <= float(score) <= maximum:
                raise ValueError(f"{panel_id}/{primary_task_id}: Source score outside scale")
            if row.get("score_scale_min") != task.get("native_scale_min") or row.get(
                "score_scale_max"
            ) != task.get("native_scale_max"):
                raise ValueError(f"{panel_id}/{primary_task_id}: train scale drift")
            if not str(row.get("input_text", "")).strip() or not str(
                row.get("answer_text", "")
            ).strip():
                raise ValueError(f"{panel_id}/{primary_task_id}: empty train text")
        test_keys: set[tuple[str, str]] = set()
        leaks: list[set[str]] = []
        for row in test_rows:
            if (
                row.get("dataset_id") != task.get("dataset_id")
                or row.get("panel_id") != panel_id
                or row.get("task_id") != primary_task_id
                or row.get("split") != "test"
                or row.get("target_responder_id") != target
            ):
                raise ValueError(f"{panel_id}/{primary_task_id}: test identity drift")
            key = (str(row.get("item_id")), target)
            if key in test_keys:
                raise ValueError(f"{panel_id}/{primary_task_id}: duplicate test key")
            test_keys.add(key)
            row_leaks = forbidden_test_keys(row)
            if row_leaks:
                leaks.append(row_leaks)
            if not str(row.get("input_text", "")).strip() or not str(
                row.get("answer_text", "")
            ).strip():
                raise ValueError(f"{panel_id}/{primary_task_id}: empty test text")
        if leaks:
            raise ValueError(
                f"{panel_id}/{primary_task_id}: forbidden test fields: {sorted(leaks[0])}"
            )
        train_item_ids = {item_id for item_id, _ in train_keys}
        test_item_ids = {item_id for item_id, _ in test_keys}
        if train_item_ids != test_item_ids:
            raise ValueError(f"{panel_id}/{primary_task_id}: fixed same-context item set drift")
        if train_keys & test_keys:
            raise ValueError(f"{panel_id}/{primary_task_id}: train/test full keys overlap")
        train_total += len(train_rows)
        test_total += len(test_rows)
        loaded.append(
            {
                "position": position,
                "registry": task,
                "metadata": metadata,
                "split_manifest": split_manifest,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "train_path": resolved["train"],
                "test_path": resolved["test_inputs"],
                "source_file_hashes": {
                    role: files[role]["sha256"]
                    for role in ("train", "test_inputs", "metadata", "split_manifest")
                },
            }
        )
    if all_primary & all_secondary:
        raise ValueError("A secondary diagnostic task also entered the primary run key set")
    expected = config["expected_counts"]
    if train_total != expected["train_rows_per_seed"]:
        raise ValueError(f"Total train rows {train_total} != 37692")
    if test_total != expected["test_keys_per_seed"]:
        raise ValueError(f"Total test keys {test_total} != 4872")
    audit = {
        "status": "PASS",
        "checked_at_utc": utc_now(),
        "public_root": public_root.as_posix(),
        "registry_path": registry_path.as_posix(),
        "registry_sha256": actual_registry_hash,
        "registry_payload_sha256": computed_payload_hash,
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "benchmark_manifest_sha256": sha256_file(public_root / "benchmark_manifest.json"),
        "score_contract_registry_sha256": sha256_file(
            public_root / "score_contract_registry.json"
        ),
        "dataset_family_count": len({task["dataset_family_id"] for task in tasks}),
        "panel_count": len(tasks),
        "primary_task_count": len(tasks),
        "train_rows_per_seed": train_total,
        "test_keys_per_seed": test_total,
        "task_seed_cells": len(tasks) * len(EXPECTED_SEEDS),
        "prediction_rows": test_total * len(EXPECTED_SEEDS),
        "excluded_panels_absent": ["ayers_askdocs", "simpeval_past"],
        "secondary_tasks_absent": True,
        "public_input_hashes": dict(sorted(public_hashes.items())),
        "private_or_gold_path_available_to_runner": False,
    }
    return registry, config, loaded, audit


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    return requested


def set_all_seeds(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_deterministic_warn_only": False,
    }


class ExactBgeM3Encoder:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        requested_device: str,
        logger: Callable[[str], None],
    ) -> None:
        self.config = config
        self.requested_device = requested_device
        self.device = resolve_device(requested_device)
        self.logger = logger
        self.tokenizer: Any = None
        self.model: Any = None
        self.snapshot = self._resolve_snapshot()
        self.model_metadata = self._validate_snapshot()

    def _resolve_snapshot(self) -> Path:
        revision = str(self.config["revision"])
        model_name = str(self.config["model_name"])
        if model_name != "BAAI/bge-m3":
            raise RuntimeError("Only the frozen BAAI/bge-m3 model is permitted")
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        snapshot = hf_home / "hub/models--BAAI--bge-m3/snapshots" / revision
        if not snapshot.is_dir() or snapshot.name != revision:
            raise FileNotFoundError(
                f"Exact local BGE-M3 revision is unavailable: {snapshot}; no fallback is allowed"
            )
        return snapshot.resolve()

    def _validate_snapshot(self) -> dict[str, Any]:
        required = ["config.json", "tokenizer.json", "pytorch_model.bin"]
        missing = [name for name in required if not (self.snapshot / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Exact BGE-M3 snapshot is incomplete: {missing}")
        model_config = read_json(self.snapshot / "config.json")
        if int(model_config.get("hidden_size", -1)) != int(self.config["base_dimension"]):
            raise RuntimeError("BGE-M3 snapshot hidden size is not 1024")
        weights = self.snapshot / "pytorch_model.bin"
        return {
            "model_name": self.config["model_name"],
            "resolved_revision": self.snapshot.name,
            "snapshot_path": self.snapshot.as_posix(),
            "snapshot_config_sha256": sha256_file(self.snapshot / "config.json"),
            "weights_filename": weights.name,
            "weights_size_bytes": weights.stat().st_size,
            "hidden_size": int(model_config["hidden_size"]),
            "requested_device": self.requested_device,
            "actual_device": self.device,
        }

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        self.logger(
            f"Loading exact BGE-M3 revision={self.snapshot.name} device={self.device} "
            f"torch={torch.__version__} transformers={transformers.__version__}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.snapshot, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.snapshot, local_files_only=True)
        self.model = self.model.to(self.device).eval()
        if int(getattr(self.model.config, "hidden_size", -1)) != int(
            self.config["base_dimension"]
        ):
            raise RuntimeError("Loaded BGE-M3 output dimension is not 1024")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.ensure_loaded()
        import torch

        output = np.empty(
            (len(texts), int(self.config["base_dimension"])), dtype=np.float32
        )
        batch_size = int(self.config["batch_size"])
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=int(self.config["max_length"]),
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                output[start : start + len(batch)] = pooled.float().cpu().numpy()
        norms = np.linalg.norm(output, axis=1)
        if output.shape[1] != 1024 or not np.all(np.isfinite(output)):
            raise RuntimeError("BGE-M3 returned invalid embeddings")
        if not np.allclose(norms, 1.0, atol=2e-5):
            raise RuntimeError("BGE-M3 embeddings are not L2-normalized")
        return output

    def close(self) -> None:
        if self.model is None:
            return
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception:
            pass


def framework_versions() -> dict[str, str]:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def stable_unique(values: Sequence[str]) -> tuple[list[str], np.ndarray]:
    index: dict[str, int] = {}
    unique: list[str] = []
    inverse = np.empty(len(values), dtype=np.int64)
    for position, value in enumerate(values):
        if value not in index:
            index[value] = len(unique)
            unique.append(value)
        inverse[position] = index[value]
    return unique, inverse


def load_or_build_embedding_cache(
    *,
    cache_root: Path,
    panel_id: str,
    task_id: str,
    field: str,
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    encoder: ExactBgeM3Encoder,
    logger: Callable[[str], None],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ordered_texts = [str(row[field]) for row in train_rows] + [
        str(row[field]) for row in test_rows
    ]
    unique_texts, inverse = stable_unique(ordered_texts)
    versions = framework_versions()
    contract = {
        "schema_version": "cce-bge-m3-base-cache-contract-v1",
        "panel_id": panel_id,
        "task_id": task_id,
        "field": field,
        "model_name": encoder.config["model_name"],
        "revision": encoder.config["revision"],
        "max_length": int(encoder.config["max_length"]),
        "batch_size": int(encoder.config["batch_size"]),
        "pooling": encoder.config["pooling"],
        "normalization": encoder.config["normalization"],
        "base_dimension": int(encoder.config["base_dimension"]),
        "ordered_text_sha256": sequence_sha256(ordered_texts),
        "unique_text_sha256": sequence_sha256(unique_texts),
        "ordered_row_count": len(ordered_texts),
        "unique_text_count": len(unique_texts),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "torch_version": versions["torch"],
        "transformers_version": versions["transformers"],
        "actual_device": encoder.device,
        "cache_key_uses_scores_or_rewards": False,
    }
    contract_hash = canonical_json_sha256(contract)
    field_root = cache_root / panel_id / task_id / field
    contract_path = field_root / "cache_contract.json"
    if contract_path.exists():
        existing = read_json(contract_path)
        if existing != {**contract, "cache_contract_sha256": contract_hash}:
            raise RuntimeError(f"Embedding cache contract mismatch: {contract_path}")
    else:
        atomic_write_json(
            contract_path, {**contract, "cache_contract_sha256": contract_hash}
        )
    shard_size = int(encoder.config.get("cache_shard_unique_texts", 256))
    shards: list[np.ndarray] = []
    shard_records: list[dict[str, Any]] = []
    shard_total = math.ceil(len(unique_texts) / shard_size)
    for shard_index, start in enumerate(range(0, len(unique_texts), shard_size)):
        stop = min(start + shard_size, len(unique_texts))
        shard_texts = unique_texts[start:stop]
        shard_path = field_root / "shards" / f"shard_{shard_index:05d}.npz"
        shard_text_hash = sequence_sha256(shard_texts)
        if shard_path.exists():
            with np.load(shard_path, allow_pickle=False) as payload:
                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
                stored = {
                    "start": int(payload["start"].item()),
                    "stop": int(payload["stop"].item()),
                    "texts_sha256": str(payload["texts_sha256"].item()),
                    "cache_contract_sha256": str(
                        payload["cache_contract_sha256"].item()
                    ),
                    "embeddings_sha256": str(payload["embeddings_sha256"].item()),
                }
            checks = {
                "bounds": stored["start"] == start and stored["stop"] == stop,
                "texts": stored["texts_sha256"] == shard_text_hash,
                "contract": stored["cache_contract_sha256"] == contract_hash,
                "shape": embeddings.shape == (stop - start, 1024),
                "finite": bool(np.all(np.isfinite(embeddings))),
                "l2": bool(
                    np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=2e-5)
                ),
                "array_hash": stored["embeddings_sha256"]
                == array_sha256(embeddings),
            }
            if not all(checks.values()):
                raise RuntimeError(f"Embedding cache shard validation failed: {shard_path}: {checks}")
            cache_status = "reused"
        else:
            started = time.perf_counter()
            embeddings = encoder.encode(shard_texts)
            embeddings_hash = array_sha256(embeddings)
            atomic_write_npz(
                shard_path,
                embeddings=embeddings,
                start=np.asarray(start, dtype=np.int64),
                stop=np.asarray(stop, dtype=np.int64),
                texts_sha256=np.asarray(shard_text_hash),
                cache_contract_sha256=np.asarray(contract_hash),
                embeddings_sha256=np.asarray(embeddings_hash),
            )
            cache_status = "created"
            logger(
                f"BGE cache {panel_id}/{task_id}/{field} shard "
                f"{shard_index + 1}/{shard_total} unique={stop - start} "
                f"seconds={time.perf_counter() - started:.1f}"
            )
        shard_records.append(
            {
                "shard_index": shard_index,
                "start": start,
                "stop": stop,
                "texts_sha256": shard_text_hash,
                "embeddings_sha256": array_sha256(embeddings),
                "file": str(shard_path.relative_to(cache_root.parent)),
                "file_sha256": sha256_file(shard_path),
                "status": cache_status,
            }
        )
        shards.append(embeddings)
    unique_embeddings = np.concatenate(shards, axis=0)
    if unique_embeddings.shape != (len(unique_texts), 1024):
        raise RuntimeError("Combined base embedding cache shape mismatch")
    aligned = unique_embeddings[inverse]
    train_aligned = aligned[: len(train_rows)]
    test_aligned = aligned[len(train_rows) :]
    manifest = {
        **contract,
        "cache_contract_sha256": contract_hash,
        "status": "COMPLETE",
        "completed_at_utc": utc_now(),
        "unique_embeddings_sha256": array_sha256(unique_embeddings),
        "aligned_train_embeddings_sha256": array_sha256(train_aligned),
        "aligned_test_embeddings_sha256": array_sha256(test_aligned),
        "shard_count": len(shard_records),
        "shards": shard_records,
    }
    atomic_write_json(field_root / "manifest.json", manifest)
    return train_aligned, test_aligned, manifest


def make_learned_config(config: dict[str, Any], seed: int) -> LearnedEmbeddingConfig:
    spec = config["learned_embedding"]
    return LearnedEmbeddingConfig(
        latent_dim=int(spec["latent_dim"]),
        hidden_dim=int(spec["hidden_dim"]),
        merge_strategy=str(spec["merge_strategy"]),
        pca_dim=int(spec["pca_dim"]),
        max_epochs=int(spec["max_epochs"]),
        learning_rate=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
        dropout=float(spec["dropout"]),
        validation_fraction=float(spec["validation_fraction"]),
        patience=int(spec["patience"]),
        normalize=bool(spec["normalize"]),
        seed=int(seed),
    )


def validation_index_audit(n: int, fraction: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if fraction <= 0 or n < 5:
        train_idx = np.arange(n)
        val_idx = np.asarray([], dtype=np.int64)
    else:
        n_val = min(max(int(round(n * fraction)), 1), n - 1)
        indices = rng.permutation(n)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
    return {
        "scope": "current_task_source_train_only",
        "seed": seed,
        "train_row_count": int(len(train_idx)),
        "validation_row_count": int(len(val_idx)),
        "train_indices_sha256": array_sha256(train_idx.astype(np.int64)),
        "validation_indices_sha256": array_sha256(val_idx.astype(np.int64)),
        "target_rows_in_train_or_validation": 0,
    }


def cell_id_for(task: dict[str, Any], seed: int) -> str:
    return f"{task['panel_id']}__{task['primary_task_id']}__seed_{seed}"


def cell_base_keys(
    task: dict[str, Any], test_rows: Sequence[dict[str, Any]]
) -> set[tuple[str, str, str, str, str, str]]:
    return {
        (
            str(task["dataset_family_id"]),
            str(task["dataset_id"]),
            str(task["panel_id"]),
            str(task["primary_task_id"]),
            str(row["item_id"]),
            str(task["target_responder_id"]),
        )
        for row in test_rows
    }


def fit_cell_to_attempt(
    *,
    attempt_dir: Path,
    task_bundle: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    config_hash: str,
    registry_hash: str,
    base_input_train: np.ndarray,
    base_input_test: np.ndarray,
    base_answer_train: np.ndarray,
    base_answer_test: np.ndarray,
    base_cache_manifests: Sequence[dict[str, Any]],
    actual_device: str,
    cell_log: Callable[[str], None],
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.perf_counter()
    task = task_bundle["registry"]
    train_rows = task_bundle["train_rows"]
    test_rows = task_bundle["test_rows"]
    cell_id = cell_id_for(task, seed)
    seed_audit = set_all_seeds(seed, deterministic=True)
    minimum = float(task["native_scale_min"])
    maximum = float(task["native_scale_max"])
    width = maximum - minimum
    y_native = np.asarray([float(row["score"]) for row in train_rows], dtype=np.float32)
    y_unit = (y_native - minimum) / width
    learned_config = make_learned_config(config, seed)
    validation_audit = validation_index_audit(
        len(train_rows), learned_config.validation_fraction, seed
    )
    cell_log(
        f"START {cell_id} train={len(train_rows)} test={len(test_rows)} "
        f"source_reward_sha256={array_sha256(y_unit)}"
    )
    (
        z_input_train,
        z_answer_train,
        z_input_test,
        z_answer_test,
        learned_diagnostics,
    ) = learn_train_only_reward_informed_embeddings(
        phi_x_train=base_input_train,
        phi_a_train=base_answer_train,
        phi_x_eval=base_input_test,
        phi_a_eval=base_answer_test,
        y_train=y_unit,
        config=learned_config,
    )
    representation_dims = {
        int(z_input_train.shape[1]),
        int(z_answer_train.shape[1]),
        int(z_input_test.shape[1]),
        int(z_answer_test.shape[1]),
    }
    if representation_dims != {192}:
        raise RuntimeError(f"{cell_id}: learned representation is not 192D")
    if (learned_diagnostics.get("pca") or {}).get("fit_scope") != "train_only":
        raise RuntimeError(f"{cell_id}: PCA is not Source-train-only")
    if learned_diagnostics.get("uses_eval_rewards") is not False:
        raise RuntimeError(f"{cell_id}: learned embedding reports eval-reward use")
    x_train = build_features(z_input_train, z_answer_train)
    x_test = build_features(z_input_test, z_answer_test)
    if x_train.shape[1] != 576 or x_test.shape[1] != 576:
        raise RuntimeError(f"{cell_id}: final feature dimension is not 576")
    gbr_spec = config["gradient_boosting_regressor"]
    model = GradientBoostingRegressor(
        n_estimators=int(gbr_spec["n_estimators"]),
        max_depth=int(gbr_spec["max_depth"]),
        learning_rate=float(gbr_spec["learning_rate"]),
        random_state=seed,
    )
    model.fit(x_train, y_unit)
    prediction_unit_raw = np.asarray(model.predict(x_test), dtype=np.float64)
    prediction_unit = np.clip(prediction_unit_raw, 0.0, 1.0)
    prediction_native = minimum + width * prediction_unit
    if not np.all(np.isfinite(prediction_native)):
        raise RuntimeError(f"{cell_id}: non-finite predictions")
    if not np.all((prediction_native >= minimum) & (prediction_native <= maximum)):
        raise RuntimeError(f"{cell_id}: predictions outside native scale")
    predictions: list[dict[str, Any]] = []
    for row, raw_unit, clipped_unit, native in zip(
        test_rows,
        prediction_unit_raw,
        prediction_unit,
        prediction_native,
        strict=True,
    ):
        predictions.append(
            {
                "schema_version": PREDICTION_SCHEMA,
                "dataset_family_id": task["dataset_family_id"],
                "dataset_family_name": task["dataset_family_name"],
                "dataset_id": task["dataset_id"],
                "panel_id": task["panel_id"],
                "task_id": task["primary_task_id"],
                "seed": int(seed),
                "item_id": str(row["item_id"]),
                "target_responder_id": task["target_responder_id"],
                "prediction": float(native),
                "prediction_native": float(native),
                "prediction_unit_raw": float(raw_unit),
                "prediction_unit_clipped": float(clipped_unit),
                "native_scale_min": minimum,
                "native_scale_max": maximum,
                "higher_is_better": True,
                "contains_target_reward": False,
            }
        )
    predictions.sort(key=prediction_base_key)
    expected_keys = cell_base_keys(task, test_rows)
    if {prediction_base_key(row) for row in predictions} != expected_keys:
        raise RuntimeError(f"{cell_id}: prediction keys do not match the frozen test keys")
    prediction_path = attempt_dir / "predictions.jsonl"
    atomic_write_jsonl(prediction_path, predictions)
    learned_diagnostics = dict(learned_diagnostics)
    learned_diagnostics["validation_index_audit"] = validation_audit
    learned_diagnostics["projector_fit_scope"] = "current_task_source_train_only"
    learned_diagnostics["target_base_embeddings_transform_only"] = True
    learned_diagnostics["target_base_embeddings_influence_fit"] = False
    diagnostics = {
        "schema_version": "cce-primary-dm-cell-diagnostics-v1",
        "status": "COMPLETE",
        "cell_id": cell_id,
        "dataset_family_id": task["dataset_family_id"],
        "dataset_id": task["dataset_id"],
        "panel_id": task["panel_id"],
        "task_id": task["primary_task_id"],
        "target_responder_id": task["target_responder_id"],
        "seed": seed,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "config_sha256": config_hash,
        "registry_sha256": registry_hash,
        "source_public_file_hashes": task_bundle["source_file_hashes"],
        "source_reward_sha256": array_sha256(y_unit),
        "target_reward_fields_available_to_fit": [],
        "target_rewards_accessed": False,
        "source_score_normalization": "public native scale to [0,1]",
        "base_model": "BAAI/bge-m3",
        "base_model_revision": config["bge_m3"]["revision"],
        "base_embedding_cache_contract_sha256": [
            manifest["cache_contract_sha256"] for manifest in base_cache_manifests
        ],
        "base_embedding_sha256": {
            "input_train": array_sha256(base_input_train),
            "input_test": array_sha256(base_input_test),
            "answer_train": array_sha256(base_answer_train),
            "answer_test": array_sha256(base_answer_test),
        },
        "learned_embedding_api": "learn_train_only_reward_informed_embeddings",
        "learned_embedding": learned_diagnostics,
        "representation_dimension": 192,
        "feature_definition": "[z_input,z_answer,z_input*z_answer]",
        "feature_dimension": 576,
        "feature_input_fields": ["input_text", "answer_text"],
        "identity_or_rater_metadata_features": [],
        "gbr": {
            "type": "GradientBoostingRegressor",
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": seed,
        },
        "seed_audit": seed_audit,
        "actual_device_for_base_embeddings": actual_device,
        "learned_embedding_device": "cpu",
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "prediction_min": float(prediction_native.min()),
        "prediction_max": float(prediction_native.max()),
        "prediction_mean": float(prediction_native.mean()),
        "prediction_sha256": sha256_file(prediction_path),
        "runtime_versions": framework_versions(),
    }
    diagnostics_path = attempt_dir / "diagnostics.json"
    atomic_write_json(diagnostics_path, diagnostics)
    summary = {
        "schema_version": "cce-primary-dm-fit-summary-v1",
        "status": "COMPLETE",
        "cell_id": cell_id,
        "dataset_family_id": task["dataset_family_id"],
        "dataset_id": task["dataset_id"],
        "panel_id": task["panel_id"],
        "task_id": task["primary_task_id"],
        "target_responder_id": task["target_responder_id"],
        "seed": seed,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "native_scale_min": minimum,
        "native_scale_max": maximum,
        "base_dimension": 1024,
        "representation_dimension": 192,
        "feature_dimension": 576,
        "learned_best_epoch": int(learned_diagnostics["best_epoch"]),
        "learned_epochs_trained": int(learned_diagnostics["epochs_trained"]),
        "elapsed_seconds": diagnostics["elapsed_seconds"],
        "prediction_sha256": diagnostics["prediction_sha256"],
    }
    summary_path = attempt_dir / "fit_summary.json"
    atomic_write_json(summary_path, summary)
    checkpoint = {
        "schema_version": "cce-primary-dm-cell-checkpoint-v1",
        "status": "COMPLETE",
        "cell_id": cell_id,
        "config_sha256": config_hash,
        "registry_sha256": registry_hash,
        "source_public_file_hashes": task_bundle["source_file_hashes"],
        "seed": seed,
        "prediction_rows": len(predictions),
        "prediction_sha256": sha256_file(prediction_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "fit_summary_sha256": sha256_file(summary_path),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(attempt_dir / "checkpoint.json", checkpoint)
    cell_log(
        f"DONE {cell_id} best_epoch={summary['learned_best_epoch']} "
        f"seconds={summary['elapsed_seconds']:.1f} prediction_sha256={checkpoint['prediction_sha256']}"
    )
    return checkpoint


def validate_completed_cell(
    *,
    cell_dir: Path,
    task_bundle: dict[str, Any],
    seed: int,
    config_hash: str,
    registry_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path = cell_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Cell directory exists without a complete checkpoint: {cell_dir}")
    checkpoint = read_json(checkpoint_path)
    task = task_bundle["registry"]
    checks = {
        "status": checkpoint.get("status") == "COMPLETE",
        "cell_id": checkpoint.get("cell_id") == cell_id_for(task, seed),
        "config": checkpoint.get("config_sha256") == config_hash,
        "registry": checkpoint.get("registry_sha256") == registry_hash,
        "sources": checkpoint.get("source_public_file_hashes")
        == task_bundle["source_file_hashes"],
        "seed": checkpoint.get("seed") == seed,
    }
    prediction_path = cell_dir / "predictions.jsonl"
    diagnostic_path = cell_dir / "diagnostics.json"
    summary_path = cell_dir / "fit_summary.json"
    checks.update(
        {
            "prediction_file": prediction_path.is_file(),
            "diagnostics_file": diagnostic_path.is_file(),
            "summary_file": summary_path.is_file(),
        }
    )
    if all(checks.values()):
        checks.update(
            {
                "prediction_hash": sha256_file(prediction_path)
                == checkpoint.get("prediction_sha256"),
                "diagnostics_hash": sha256_file(diagnostic_path)
                == checkpoint.get("diagnostics_sha256"),
                "summary_hash": sha256_file(summary_path)
                == checkpoint.get("fit_summary_sha256"),
            }
        )
    if not all(checks.values()):
        raise RuntimeError(f"Completed cell hash validation failed: {cell_dir}: {checks}")
    predictions = read_jsonl(prediction_path)
    if len(predictions) != int(task["expected_test_rows"]):
        raise RuntimeError(f"Completed cell row count mismatch: {cell_dir}")
    if any(int(row.get("seed", -1)) != seed for row in predictions):
        raise RuntimeError(f"Completed cell seed drift: {cell_dir}")
    if {prediction_base_key(row) for row in predictions} != cell_base_keys(
        task, task_bundle["test_rows"]
    ):
        raise RuntimeError(f"Completed cell key-set mismatch: {cell_dir}")
    if any(
        not math.isfinite(float(row["prediction"]))
        or not float(task["native_scale_min"])
        <= float(row["prediction"])
        <= float(task["native_scale_max"])
        for row in predictions
    ):
        raise RuntimeError(f"Completed cell prediction scale/finite check failed: {cell_dir}")
    return checkpoint, predictions, read_json(summary_path)


def update_run_manifest(
    path: Path,
    manifest: dict[str, Any],
    state: str,
    **updates: Any,
) -> None:
    manifest.update(updates)
    manifest["state"] = state
    manifest.setdefault("state_history", []).append({"state": state, "at_utc": utc_now()})
    atomic_write_json(path, manifest)


def initialize_or_resume_run(
    *,
    output_root: Path,
    resume: bool,
    contract_audit: dict[str, Any],
    config: dict[str, Any],
    actual_device: str,
) -> dict[str, Any]:
    run_manifest_path = output_root / "run_manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite existing output root: {output_root}; use a next-free version or --resume"
            )
        if not run_manifest_path.is_file():
            raise RuntimeError("Resume root has no run_manifest.json")
        manifest = read_json(run_manifest_path)
        if manifest.get("config_sha256") != contract_audit["config_sha256"]:
            raise RuntimeError("Resume config SHA256 mismatch")
        if manifest.get("registry_sha256") != contract_audit["registry_sha256"]:
            raise RuntimeError("Resume registry SHA256 mismatch")
        if manifest.get("public_input_hashes") != contract_audit["public_input_hashes"]:
            raise RuntimeError("Resume public input hashes mismatch")
        return manifest
    output_root.mkdir(parents=True)
    manifest = {
        "schema_version": "cce-primary-dm-run-manifest-v1",
        "experiment_id": config["experiment_id"],
        "method_name": config["method_name"],
        "scientific_scope": config["scientific_scope"],
        "started_at_utc": utc_now(),
        "config_sha256": contract_audit["config_sha256"],
        "registry_sha256": contract_audit["registry_sha256"],
        "registry_payload_sha256": contract_audit["registry_payload_sha256"],
        "public_input_hashes": contract_audit["public_input_hashes"],
        "expected_counts": config["expected_counts"],
        "seeds": config["seeds"],
        "actual_device": actual_device,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "software_versions": framework_versions(),
        "deterministic_contract": config["reproducibility"],
        "private_or_gold_path_available_to_runner": False,
        "target_reward_fields_available_to_runner": [],
        "evaluated_prediction_inputs_available_to_runner": False,
        "state_history": [],
    }
    atomic_write_json(output_root / "PUBLIC_CONTRACT_AUDIT.json", contract_audit)
    update_run_manifest(run_manifest_path, manifest, "CONTRACT_FROZEN")
    return manifest


def aggregate_checkpoint_manifest(
    *,
    output_root: Path,
    task_bundles: Sequence[dict[str, Any]],
    config_hash: str,
    registry_hash: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for bundle in task_bundles:
        task = bundle["registry"]
        for seed in EXPECTED_SEEDS:
            cell_dir = output_root / "checkpoints" / cell_id_for(task, seed)
            if not cell_dir.is_dir():
                continue
            checkpoint, _, _ = validate_completed_cell(
                cell_dir=cell_dir,
                task_bundle=bundle,
                seed=seed,
                config_hash=config_hash,
                registry_hash=registry_hash,
            )
            rows.append(
                {
                    **checkpoint,
                    "checkpoint_file": str(
                        (cell_dir / "checkpoint.json").relative_to(output_root)
                    ),
                    "checkpoint_file_sha256": sha256_file(cell_dir / "checkpoint.json"),
                }
            )
    manifest = {
        "schema_version": "cce-primary-dm-checkpoint-manifest-v1",
        "status": "COMPLETE" if len(rows) == 33 else "IN_PROGRESS",
        "completed_cells": len(rows),
        "expected_cells": 33,
        "config_sha256": config_hash,
        "registry_sha256": registry_hash,
        "cells": sorted(rows, key=lambda row: str(row["cell_id"])),
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(output_root / "checkpoint_manifest.json", manifest)
    return manifest


def combine_and_freeze_predictions(
    *,
    output_root: Path,
    task_bundles: Sequence[dict[str, Any]],
    registry: dict[str, Any],
    config: dict[str, Any],
    contract_audit: dict[str, Any],
    embedding_cache_manifest: dict[str, Any],
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    all_predictions: list[dict[str, Any]] = []
    fit_summaries: list[dict[str, Any]] = []
    base_key_sets: dict[int, set[tuple[str, str, str, str, str, str]]] = {
        seed: set() for seed in EXPECTED_SEEDS
    }
    for bundle in task_bundles:
        task = bundle["registry"]
        for seed in EXPECTED_SEEDS:
            cell_dir = output_root / "checkpoints" / cell_id_for(task, seed)
            _, predictions, summary = validate_completed_cell(
                cell_dir=cell_dir,
                task_bundle=bundle,
                seed=seed,
                config_hash=contract_audit["config_sha256"],
                registry_hash=contract_audit["registry_sha256"],
            )
            all_predictions.extend(predictions)
            fit_summaries.append(summary)
            base_key_sets[seed].update(prediction_base_key(row) for row in predictions)
    if any(len(keys) != 4872 for keys in base_key_sets.values()):
        raise RuntimeError(
            f"Per-seed target-key counts are not 4872: "
            f"{ {seed: len(keys) for seed, keys in base_key_sets.items()} }"
        )
    if not (base_key_sets[0] == base_key_sets[1] == base_key_sets[2]):
        raise RuntimeError("The three seeds do not use identical base target key sets")
    full_keys = {
        (int(row["seed"]), *prediction_base_key(row)) for row in all_predictions
    }
    if len(all_predictions) != 14616 or len(full_keys) != 14616:
        raise RuntimeError("Combined predictions are not exactly 14,616 unique keys")
    forbidden_panels = {"ayers_askdocs", "simpeval_past"}
    if {str(row["panel_id"]) for row in all_predictions} & forbidden_panels:
        raise RuntimeError("Excluded panel rows entered formal predictions")
    primary_keys = {task_key(task) for task in registry["primary_tasks"]}
    if {(str(row["panel_id"]), str(row["task_id"])) for row in all_predictions} != primary_keys:
        raise RuntimeError("Prediction task set differs from the frozen primary registry")
    all_predictions.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["panel_id"]),
            str(row["task_id"]),
            str(row["item_id"]),
            str(row["target_responder_id"]),
        )
    )
    fit_summaries.sort(
        key=lambda row: (str(row["panel_id"]), str(row["task_id"]), int(row["seed"]))
    )
    update_run_manifest(
        output_root / "run_manifest.json",
        run_manifest,
        "ALL_PREDICTIONS_COMPLETE",
        completed_cells=33,
        prediction_rows=14616,
    )
    predictions_path = output_root / "predictions.jsonl"
    fit_summary_path = output_root / "fit_summary.jsonl"
    atomic_write_jsonl(predictions_path, all_predictions)
    atomic_write_jsonl(fit_summary_path, fit_summaries)
    prediction_sha = sha256_file(predictions_path)
    fit_summary_sha = sha256_file(fit_summary_path)
    frozen_at = utc_now()
    update_run_manifest(
        output_root / "run_manifest.json",
        run_manifest,
        "PREDICTIONS_FROZEN",
        predictions_path=predictions_path.as_posix(),
        predictions_sha256=prediction_sha,
        fit_summary_path=fit_summary_path.as_posix(),
        fit_summary_sha256=fit_summary_sha,
        frozen_at_utc=frozen_at,
        embedding_cache_manifest_sha256=sha256_file(
            output_root / "embedding_cache_manifest.json"
        ),
    )
    freeze_receipt = {
        "schema_version": "cce-primary-dm-prediction-freeze-receipt-v1",
        "status": "FREEZE_RECEIPT_WRITTEN",
        "contract_frozen": True,
        "all_predictions_complete": True,
        "predictions_frozen": True,
        "private_gold_released_to_evaluator": False,
        "private_gold_opened_by_runner": False,
        "runner_private_or_gold_path_available": False,
        "predictions_path": predictions_path.as_posix(),
        "predictions_sha256": prediction_sha,
        "prediction_rows": 14616,
        "per_seed_prediction_rows": {"0": 4872, "1": 4872, "2": 4872},
        "task_seed_cells": 33,
        "base_target_key_set_identical_across_seeds": True,
        "base_target_key_set_sha256": canonical_json_sha256(
            sorted([list(key) for key in base_key_sets[0]])
        ),
        "registry_path": contract_audit["registry_path"],
        "registry_sha256": contract_audit["registry_sha256"],
        "registry_payload_sha256": contract_audit["registry_payload_sha256"],
        "config_path": contract_audit["config_path"],
        "config_sha256": contract_audit["config_sha256"],
        "public_input_hashes": contract_audit["public_input_hashes"],
        "checkpoint_manifest_sha256": sha256_file(
            output_root / "checkpoint_manifest.json"
        ),
        "embedding_cache_manifest_sha256": sha256_file(
            output_root / "embedding_cache_manifest.json"
        ),
        "fit_summary_sha256": fit_summary_sha,
        "frozen_at_utc": frozen_at,
        "receipt_written_at_utc": utc_now(),
        "state_sequence": [
            "CONTRACT_FROZEN",
            "FIT_PREDICT_SEED_0_1_2",
            "ALL_PREDICTIONS_COMPLETE",
            "PREDICTIONS_FROZEN",
            "FREEZE_RECEIPT_WRITTEN",
        ],
    }
    receipt_path = output_root / "prediction_freeze_receipt.json"
    atomic_write_json(receipt_path, freeze_receipt)
    update_run_manifest(
        output_root / "run_manifest.json",
        run_manifest,
        "FREEZE_RECEIPT_WRITTEN",
        prediction_freeze_receipt_path=receipt_path.as_posix(),
        prediction_freeze_receipt_sha256=sha256_file(receipt_path),
        completed_at_utc=utc_now(),
    )
    return freeze_receipt


def validate_existing_freeze(output_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    receipt_path = output_root / "prediction_freeze_receipt.json"
    predictions_path = output_root / "predictions.jsonl"
    if not receipt_path.is_file() or not predictions_path.is_file():
        raise RuntimeError("Run claims a frozen state but freeze artifacts are missing")
    receipt = read_json(receipt_path)
    checks = {
        "state": manifest.get("state") == "FREEZE_RECEIPT_WRITTEN",
        "receipt_status": receipt.get("status") == "FREEZE_RECEIPT_WRITTEN",
        "prediction_hash": sha256_file(predictions_path) == receipt.get("predictions_sha256"),
        "prediction_rows": len(read_jsonl(predictions_path)) == 14616,
        "runner_gold_unopened": receipt.get("private_gold_opened_by_runner") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Existing freeze validation failed: {checks}")
    return receipt


def run_experiment(
    *,
    public_root: Path,
    registry_path: Path,
    config_path: Path,
    output_root: Path,
    resume: bool,
    requested_device: str,
) -> dict[str, Any]:
    registry, config, task_bundles, contract_audit = validate_public_contract(
        public_root=public_root,
        registry_path=registry_path,
        config_path=config_path,
    )
    configured_device = str(config["bge_m3"].get("device", "auto"))
    if requested_device != "auto" and requested_device != configured_device:
        # An explicit hardware request is allowed, but the model/data contract is unchanged.
        configured_device = requested_device
    actual_device = resolve_device(configured_device)
    run_manifest = initialize_or_resume_run(
        output_root=output_root,
        resume=resume,
        contract_audit=contract_audit,
        config=config,
        actual_device=actual_device,
    )
    if run_manifest.get("state") == "FREEZE_RECEIPT_WRITTEN":
        return validate_existing_freeze(output_root, run_manifest)
    logger = RunLogger(output_root / "progress.log")
    logger(
        f"CONTRACT PASS families=10 panels=11 tasks=11 train=37692 test=4872 "
        f"cells=33 predictions=14616 device={actual_device}"
    )
    encoder = ExactBgeM3Encoder(
        config=config["bge_m3"], requested_device=configured_device, logger=logger
    )
    run_manifest["bge_m3_snapshot"] = encoder.model_metadata
    atomic_write_json(output_root / "run_manifest.json", run_manifest)
    update_run_manifest(
        output_root / "run_manifest.json",
        run_manifest,
        "BUILD_BASE_EMBEDDING_CACHE",
    )
    cache_root = output_root / "embedding_cache"
    cache_by_task: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]
    ] = {}
    cache_manifests: list[dict[str, Any]] = []
    # Build every frozen base embedding before fitting.  Shards are independently
    # hash-checked and can be resumed after interruption.
    for index, bundle in enumerate(task_bundles, 1):
        task = bundle["registry"]
        panel_id = str(task["panel_id"])
        task_id = str(task["primary_task_id"])
        logger(f"BASE_CACHE panel={panel_id} task={task_id} ({index}/11)")
        input_train, input_test, input_manifest = load_or_build_embedding_cache(
            cache_root=cache_root,
            panel_id=panel_id,
            task_id=task_id,
            field="input_text",
            train_rows=bundle["train_rows"],
            test_rows=bundle["test_rows"],
            encoder=encoder,
            logger=logger,
        )
        answer_train, answer_test, answer_manifest = load_or_build_embedding_cache(
            cache_root=cache_root,
            panel_id=panel_id,
            task_id=task_id,
            field="answer_text",
            train_rows=bundle["train_rows"],
            test_rows=bundle["test_rows"],
            encoder=encoder,
            logger=logger,
        )
        cache_by_task[(panel_id, task_id)] = (
            input_train,
            input_test,
            answer_train,
            answer_test,
            [input_manifest, answer_manifest],
        )
        cache_manifests.extend([input_manifest, answer_manifest])
    encoder.close()
    embedding_cache_manifest = {
        "schema_version": "cce-primary-bge-m3-embedding-cache-manifest-v1",
        "status": "COMPLETE",
        "model_name": config["bge_m3"]["model_name"],
        "revision": config["bge_m3"]["revision"],
        "actual_device": actual_device,
        "cache_reused_across_seeds": True,
        "cache_key_uses_scores_or_rewards": False,
        "panel_task_field_caches": cache_manifests,
        "cache_count": len(cache_manifests),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(output_root / "embedding_cache_manifest.json", embedding_cache_manifest)
    update_run_manifest(
        output_root / "run_manifest.json",
        run_manifest,
        "FIT_PREDICT_SEED_0_1_2",
        embedding_cache_manifest_sha256=sha256_file(
            output_root / "embedding_cache_manifest.json"
        ),
    )
    config_hash = contract_audit["config_sha256"]
    registry_hash = contract_audit["registry_sha256"]
    inprogress_root = output_root / "checkpoints" / ".inprogress"
    inprogress_root.mkdir(parents=True, exist_ok=True)
    for bundle in task_bundles:
        task = bundle["registry"]
        key = task_key(task)
        (
            input_train,
            input_test,
            answer_train,
            answer_test,
            task_cache_manifests,
        ) = cache_by_task[key]
        for seed in EXPECTED_SEEDS:
            cell_id = cell_id_for(task, seed)
            cell_dir = output_root / "checkpoints" / cell_id
            cell_logger = RunLogger(output_root / "logs" / f"{cell_id}.log")
            if cell_dir.exists():
                validate_completed_cell(
                    cell_dir=cell_dir,
                    task_bundle=bundle,
                    seed=seed,
                    config_hash=config_hash,
                    registry_hash=registry_hash,
                )
                cell_logger(f"RESUME_SKIP hash-validated COMPLETE cell={cell_id}")
                continue
            attempt = Path(
                tempfile.mkdtemp(prefix=f"{cell_id}__", dir=inprogress_root)
            )
            fit_cell_to_attempt(
                attempt_dir=attempt,
                task_bundle=bundle,
                seed=seed,
                config=config,
                config_hash=config_hash,
                registry_hash=registry_hash,
                base_input_train=input_train,
                base_input_test=input_test,
                base_answer_train=answer_train,
                base_answer_test=answer_test,
                base_cache_manifests=task_cache_manifests,
                actual_device=actual_device,
                cell_log=cell_logger,
            )
            if cell_dir.exists():
                raise RuntimeError(f"Refusing to overwrite completed cell: {cell_dir}")
            os.replace(attempt, cell_dir)
            validate_completed_cell(
                cell_dir=cell_dir,
                task_bundle=bundle,
                seed=seed,
                config_hash=config_hash,
                registry_hash=registry_hash,
            )
            checkpoint_manifest = aggregate_checkpoint_manifest(
                output_root=output_root,
                task_bundles=task_bundles,
                config_hash=config_hash,
                registry_hash=registry_hash,
            )
            logger(
                f"CHECKPOINT completed={checkpoint_manifest['completed_cells']}/33 cell={cell_id}"
            )
        del input_train, input_test, answer_train, answer_test
        cache_by_task.pop(key, None)
        gc.collect()
    checkpoint_manifest = aggregate_checkpoint_manifest(
        output_root=output_root,
        task_bundles=task_bundles,
        config_hash=config_hash,
        registry_hash=registry_hash,
    )
    if checkpoint_manifest["status"] != "COMPLETE":
        raise RuntimeError("Not all 33 task x seed checkpoints are complete")
    receipt = combine_and_freeze_predictions(
        output_root=output_root,
        task_bundles=task_bundles,
        registry=registry,
        config=config,
        contract_audit=contract_audit,
        embedding_cache_manifest=embedding_cache_manifest,
        run_manifest=run_manifest,
    )
    logger(
        f"FREEZE COMPLETE rows={receipt['prediction_rows']} sha256={receipt['predictions_sha256']}"
    )
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args(argv)


def runner_cli_is_gold_free() -> bool:
    source = inspect.getsource(parse_args).lower()
    forbidden = ("private-root", "private_root", "gold-path", "gold_path", "target-gold")
    return not any(token in source for token in forbidden)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not runner_cli_is_gold_free():
        raise RuntimeError("Runner CLI unexpectedly exposes a private/gold path")
    receipt = run_experiment(
        public_root=args.public_root.resolve(),
        registry_path=args.registry.resolve(),
        config_path=args.config.resolve(),
        output_root=args.output_dir.resolve(),
        resume=bool(args.resume),
        requested_device=str(args.device),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "prediction_rows": receipt["prediction_rows"],
                "predictions_sha256": receipt["predictions_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
