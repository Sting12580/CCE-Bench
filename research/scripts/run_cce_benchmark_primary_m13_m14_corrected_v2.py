#!/usr/bin/env python3
"""Run the post-gold corrected M13-v2 and M14-v2 package.

The runner is score-blind with respect to Target rows.  It reads public inputs
and hash-validated, frozen v1 nuisance artifacts only.  It has no private-gold
argument and never imports the evaluator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_cce_benchmark_primary_zero_label_methods_v1 as v1  # noqa: E402
from cce_data.estimators.cce_primary_zero_label import (  # noqa: E402
    array_sha256,
    normalize_score,
    sndr_predictions,
)
from cce_data.estimators.cce_primary_zero_label_m13_m14_v2 import (  # noqa: E402
    CorrectedM13Config,
    crossfit_reward_informed_sn_mips_v2,
)
from run_cce_benchmark_primary_dm_bge_m3_learned_gbr import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_npz,
    canonical_json_bytes,
    canonical_json_sha256,
    read_json,
    read_jsonl,
    sequence_sha256,
    sha256_file,
    stable_unique,
)


DEFAULT_PUBLIC_ROOT = REPO_ROOT / "data/processed/cce_benchmark_single_target_v2"
DEFAULT_UPSTREAM_RUN_ROOT = (
    REPO_ROOT / "outputs/cce_benchmark_primary_m1_m3_m6_m13_m14_zero_label_v1"
)
DEFAULT_METHOD_REGISTRY = (
    REPO_ROOT
    / "configs/cce_benchmark_primary_method_registry_m13_m14_corrected_v2.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/cce_benchmark_primary_m13_m14_corrected_v2"
M13_ID = "M13_V2_SN_MIPS_BGE_LEARNED576_STRICT_CF"
M14_ID = "M14_V2_SNDR_BGE_REWEIGHTED"
METHOD_ORDER = (M13_ID, M14_ID)
SEEDS = (0, 1, 2)
ROW_SCHEMA = "cce-primary-zero-label-row-prediction-v2-correction"
VALUE_SCHEMA = "cce-primary-zero-label-panel-value-v2-correction"
RECEIPT_STATUS = "FROZEN_POST_GOLD_METHOD_CORRECTION / CORRECTION_RUNNER_GOLD_FREE"
FORBIDDEN_OUTPUT_PARTS = {"private", "gold"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def registry_payload_sha256(registry: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(registry, ensure_ascii=False))
    normalized["registry_payload_sha256"] = None
    return canonical_json_sha256(normalized)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def write_once_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"Immutable artifact exists with different bytes: {path}")
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


def _cell_dir(output_root: Path, method_id: str, task: dict[str, Any], seed: int) -> Path:
    return (
        output_root
        / "methods"
        / method_id
        / str(task["panel_id"])
        / str(task["primary_task_id"])
        / f"seed={seed}"
    )


def _new_attempt(output_root: Path, method_id: str, task: dict[str, Any], seed: int) -> Path:
    path = (
        output_root
        / ".staging"
        / f"{method_id}__{task['panel_id']}__{task['primary_task_id']}__{seed}__{uuid.uuid4().hex}"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def _commit_attempt(
    attempt: Path,
    final: Path,
    *,
    method_id: str,
    task: dict[str, Any],
    seed: int,
) -> None:
    if final.exists():
        raise RuntimeError(f"Completed-cell overwrite is forbidden: {final}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(attempt.rglob("*")):
        if path.is_file() and path.name != "checkpoint_manifest.json":
            relative = path.relative_to(attempt).as_posix()
            files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    required = {"panel_value.json", "fit_summary.json", "diagnostics.json"}
    if not required <= set(files):
        raise RuntimeError(f"Corrected cell lacks required artifacts: {attempt}")
    atomic_write_json(
        attempt / "checkpoint_manifest.json",
        {
            "schema_version": "cce-primary-m13-m14-corrected-cell-manifest-v2",
            "status": "COMPLETE",
            "method_id": method_id,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "committed_at_utc": utc_now(),
            "files": files,
        },
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(attempt, final)


def _validate_cell(final: Path, method_id: str, task: dict[str, Any], seed: int) -> None:
    manifest_path = final / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Corrected cell has no manifest: {final}")
    manifest = read_json(manifest_path)
    if not (
        manifest.get("schema_version") == "cce-primary-m13-m14-corrected-cell-manifest-v2"
        and manifest.get("status") == "COMPLETE"
        and manifest.get("method_id") == method_id
        and manifest.get("panel_id") == task["panel_id"]
        and manifest.get("task_id") == task["primary_task_id"]
        and manifest.get("seed") == seed
    ):
        raise RuntimeError(f"Corrected cell identity drift: {final}")
    for relative, record in manifest["files"].items():
        path = final / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"Corrected cell artifact drift: {path}")


def _skip_or_block_cell(
    final: Path,
    method_id: str,
    task: dict[str, Any],
    seed: int,
    resume: bool,
) -> bool:
    if not final.exists():
        return False
    if not resume:
        raise RuntimeError(f"Completed corrected cell exists without --resume: {final}")
    _validate_cell(final, method_id, task, seed)
    return True


def _row_predictions(
    method_id: str,
    task: dict[str, Any],
    test_rows: Sequence[dict[str, Any]],
    seed: int,
    prediction_unit: np.ndarray,
) -> list[dict[str, Any]]:
    minimum = float(task["native_scale_min"])
    maximum = float(task["native_scale_max"])
    unit = np.asarray(prediction_unit, dtype=np.float64).reshape(-1)
    if len(unit) != len(test_rows) or not np.all(np.isfinite(unit)):
        raise RuntimeError("Corrected row prediction vector is invalid.")
    if np.any(unit < 0.0) or np.any(unit > 1.0):
        raise RuntimeError("Corrected row prediction escaped [0,1].")
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
            "is_deterministic": False,
            "contains_target_reward": False,
            "post_gold_method_correction": True,
        }
        for index, row in enumerate(test_rows)
    ]


def _panel_value(
    method_id: str,
    task: dict[str, Any],
    seed: int,
    value_unit: float,
    output_object: str,
) -> dict[str, Any]:
    minimum = float(task["native_scale_min"])
    maximum = float(task["native_scale_max"])
    if not math.isfinite(value_unit) or not 0.0 <= value_unit <= 1.0:
        raise RuntimeError("Corrected panel value is invalid.")
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
        "post_gold_method_correction": True,
    }


def _validate_registry_and_upstream(
    *,
    public_root: Path,
    upstream_run_root: Path,
    method_registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    registry = read_json(method_registry_path)
    expected_payload = registry.get("registry_payload_sha256")
    if registry_payload_sha256(registry) != expected_payload:
        raise RuntimeError("Corrected method-registry canonical payload hash mismatch.")
    sidecar = method_registry_path.with_suffix(".sha256")
    if not sidecar.is_file() or sha256_file(method_registry_path) not in sidecar.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("Corrected method-registry sidecar binding failed.")
    if registry.get("status") != "POST_GOLD_METHOD_CORRECTION / NOT_RUN":
        raise RuntimeError("Corrected registry scientific status drifted.")
    if {row["method_id"] for row in registry["methods"]} != set(METHOD_ORDER):
        raise RuntimeError("Corrected registry method IDs drifted.")

    upstream_binding = registry["upstream_v1_binding"]
    frozen_upstream_path = (REPO_ROOT / upstream_binding["run_root"]).resolve()
    if upstream_run_root.resolve() != frozen_upstream_path:
        raise RuntimeError("Upstream v1 run-root drifted from corrected registry.")
    upstream_registry_path = (
        REPO_ROOT / upstream_binding["method_registry"]["path"]
    ).resolve()
    if (
        not upstream_registry_path.is_file()
        or sha256_file(upstream_registry_path)
        != upstream_binding["method_registry"]["file_sha256"]
    ):
        raise RuntimeError("Upstream v1 registry file/hash drift.")
    upstream_registry = read_json(upstream_registry_path)
    if (
        v1.registry_payload_sha256(upstream_registry)
        != upstream_binding["method_registry"]["payload_sha256"]
    ):
        raise RuntimeError("Upstream v1 registry payload drift.")
    receipt = v1._validate_existing_freeze(upstream_run_root)  # noqa: SLF001
    receipt_path = upstream_run_root / "prediction_freeze_receipt.json"
    if (
        sha256_file(receipt_path)
        != upstream_binding["prediction_freeze_receipt"]["file_sha256"]
        or receipt["artifact_map_canonical_sha256"]
        != upstream_binding["prediction_freeze_receipt"]["artifact_map_canonical_sha256"]
    ):
        raise RuntimeError("Upstream v1 freeze receipt drift.")

    _, task_registry, bundles, public_audit = v1.validate_public_contract(
        public_root.resolve(), upstream_registry_path
    )
    task_binding = registry["primary_task_registry"]
    if (
        public_audit["primary_task_registry_file_sha256"] != task_binding["file_sha256"]
        or public_audit["primary_task_registry_payload_sha256"]
        != task_binding["payload_sha256"]
        or public_audit["source_rows_per_seed"] != 37_842
        or public_audit["target_keys_per_seed"] != 4_922
    ):
        raise RuntimeError("Corrected registry/public task binding drift.")
    return registry, task_registry, bundles, public_audit, receipt


def _validate_receipt_member(
    upstream_run_root: Path,
    upstream_receipt: dict[str, Any],
    path: Path,
) -> str:
    relative = path.resolve().relative_to(upstream_run_root.resolve()).as_posix()
    record = upstream_receipt["artifacts"].get(relative)
    if record is None:
        raise RuntimeError(f"Upstream file is not in v1 receipt: {relative}")
    actual = sha256_file(path)
    if path.stat().st_size != int(record["size_bytes"]) or actual != record["sha256"]:
        raise RuntimeError(f"Upstream receipt member drift: {relative}")
    return actual


def _load_cache_field_read_only(
    *,
    upstream_run_root: Path,
    upstream_receipt: dict[str, Any],
    bundle: dict[str, Any],
    field: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    task = bundle["task"]
    root = (
        upstream_run_root
        / "base_embedding_cache"
        / str(task["panel_id"])
        / str(task["primary_task_id"])
        / field
    )
    manifest_path = root / "manifest.json"
    contract_path = root / "cache_contract.json"
    manifest_hash = _validate_receipt_member(
        upstream_run_root, upstream_receipt, manifest_path
    )
    contract_hash = _validate_receipt_member(
        upstream_run_root, upstream_receipt, contract_path
    )
    manifest = read_json(manifest_path)
    contract = read_json(contract_path)
    ordered_texts = [str(row[field]) for row in bundle["train_rows"]] + [
        str(row[field]) for row in bundle["test_rows"]
    ]
    unique_texts, inverse = stable_unique(ordered_texts)
    checks = {
        "status": manifest.get("status") == "COMPLETE",
        "panel": manifest.get("panel_id") == task["panel_id"],
        "task": manifest.get("task_id") == task["primary_task_id"],
        "field": manifest.get("field") == field,
        "dimension": int(manifest.get("base_dimension", -1)) == 1024,
        "ordered_count": int(manifest.get("ordered_row_count", -1)) == len(ordered_texts),
        "unique_count": int(manifest.get("unique_text_count", -1)) == len(unique_texts),
        "ordered_hash": manifest.get("ordered_text_sha256") == sequence_sha256(ordered_texts),
        "unique_hash": manifest.get("unique_text_sha256") == sequence_sha256(unique_texts),
        "contract": contract.get("cache_contract_sha256")
        == manifest.get("cache_contract_sha256"),
        "reward_free_key": manifest.get("cache_key_uses_scores_or_rewards") is False,
        "upstream_cache_device": manifest.get("actual_device") == "mps",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Read-only upstream BGE cache validation failed: {checks}")
    shards: list[np.ndarray] = []
    for record in manifest["shards"]:
        path = upstream_run_root / str(record["file"])
        file_hash = _validate_receipt_member(upstream_run_root, upstream_receipt, path)
        if file_hash != record["file_sha256"]:
            raise RuntimeError("Upstream BGE shard manifest/file hash mismatch.")
        with np.load(path, allow_pickle=False) as payload:
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            stored = {
                "start": int(payload["start"].item()),
                "stop": int(payload["stop"].item()),
                "texts_sha256": str(payload["texts_sha256"].item()),
                "cache_contract_sha256": str(payload["cache_contract_sha256"].item()),
                "embeddings_sha256": str(payload["embeddings_sha256"].item()),
            }
        if not (
            stored["start"] == int(record["start"])
            and stored["stop"] == int(record["stop"])
            and stored["texts_sha256"] == record["texts_sha256"]
            and stored["cache_contract_sha256"] == manifest["cache_contract_sha256"]
            and embeddings.shape == (stored["stop"] - stored["start"], 1024)
            and np.all(np.isfinite(embeddings))
            and np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=2e-5)
            and array_sha256(embeddings) == stored["embeddings_sha256"]
            and stored["embeddings_sha256"] == record["embeddings_sha256"]
        ):
            raise RuntimeError(f"Upstream BGE shard content drift: {path}")
        shards.append(embeddings)
    unique_embeddings = np.concatenate(shards, axis=0)
    if (
        unique_embeddings.shape != (len(unique_texts), 1024)
        or array_sha256(unique_embeddings) != manifest["unique_embeddings_sha256"]
    ):
        raise RuntimeError("Upstream BGE unique-cache array drift.")
    aligned = unique_embeddings[inverse]
    train = aligned[: len(bundle["train_rows"])]
    test = aligned[len(bundle["train_rows"]) :]
    if (
        array_sha256(train) != manifest["aligned_train_embeddings_sha256"]
        or array_sha256(test) != manifest["aligned_test_embeddings_sha256"]
    ):
        raise RuntimeError("Upstream BGE aligned-cache array drift.")
    return train, test, {
        "manifest_path": manifest_path.as_posix(),
        "manifest_file_sha256": manifest_hash,
        "contract_path": contract_path.as_posix(),
        "contract_file_sha256": contract_hash,
        "cache_contract_sha256": manifest["cache_contract_sha256"],
        "aligned_train_embeddings_sha256": array_sha256(train),
        "aligned_test_embeddings_sha256": array_sha256(test),
    }


def _load_base_embeddings_read_only(
    *,
    upstream_run_root: Path,
    upstream_receipt: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    q_train, q_test, q_audit = _load_cache_field_read_only(
        upstream_run_root=upstream_run_root,
        upstream_receipt=upstream_receipt,
        bundle=bundle,
        field="input_text",
    )
    a_train, a_test, a_audit = _load_cache_field_read_only(
        upstream_run_root=upstream_run_root,
        upstream_receipt=upstream_receipt,
        bundle=bundle,
        field="answer_text",
    )
    return {
        "q_train": q_train,
        "q_test": q_test,
        "a_train": a_train,
        "a_test": a_test,
        "audits": [q_audit, a_audit],
    }


def run_m13(
    *,
    output_root: Path,
    bundle: dict[str, Any],
    embeddings: dict[str, Any],
    seed: int,
    resume: bool,
    logger: Callable[[str], None],
) -> None:
    task = bundle["task"]
    final = _cell_dir(output_root, M13_ID, task, seed)
    if _skip_or_block_cell(final, M13_ID, task, seed, resume):
        logger(f"RESUME validated {M13_ID}/{task['panel_id']}/seed={seed}")
        return
    attempt = _new_attempt(output_root, M13_ID, task, seed)
    started = time.perf_counter()
    y = normalize_score(
        np.asarray([row["score"] for row in bundle["train_rows"]]),
        float(task["native_scale_min"]),
        float(task["native_scale_max"]),
    )
    fitted = crossfit_reward_informed_sn_mips_v2(
        source_question=embeddings["q_train"],
        source_answer=embeddings["a_train"],
        target_question=embeddings["q_test"],
        target_answer=embeddings["a_test"],
        source_folds=bundle["source_folds"],
        target_folds=bundle["target_folds"],
        y_source_unit=y,
        seed=seed,
        config=CorrectedM13Config(),
        parallel_fits=5,
    )
    source_ingredients = [
        {
            "schema_version": "cce-primary-m13-v2-source-ingredient-v2",
            "method_id": M13_ID,
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
            "post_gold_method_correction": True,
        }
        for index, row in enumerate(bundle["train_rows"])
    ]
    target_density = [
        {
            "schema_version": "cce-primary-m13-v2-target-density-oof-v2",
            "method_id": M13_ID,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "item_id": str(row["item_id"]),
            "target_responder_id": str(row["target_responder_id"]),
            "outer_fold": int(bundle["target_folds"][index]),
            "calibrated_p_target_oof": float(fitted["target_probability_oof"][index]),
            "contains_target_reward": False,
            "post_gold_method_correction": True,
        }
        for index, row in enumerate(bundle["test_rows"])
    ]
    atomic_write_jsonl(attempt / "source_bootstrap_ingredients.jsonl", source_ingredients)
    atomic_write_jsonl(attempt / "target_density_oof.jsonl", target_density)
    atomic_write_json(
        attempt / "panel_value.json",
        _panel_value(M13_ID, task, seed, fitted["v_hat_unit"], "panel_value_only"),
    )
    atomic_write_npz(
        attempt / "density_oof_checkpoint.npz",
        source_probability_oof=fitted["source_probability_oof"],
        target_probability_oof=fitted["target_probability_oof"],
        source_raw_ratio_oof=fitted["source_raw_ratio_oof"],
        source_weight_oof=fitted["source_weight_oof"],
        base_decision_oof=fitted["base_decision_oof"],
        base_probability_oof=fitted["base_probability_oof"],
        calibrated_probability_oof=fitted["calibrated_probability_oof"],
    )
    if fitted["positivity_warning"]:
        status = "POSITIVITY_WARNING"
    elif fitted["near_uniform_weight_warning"]:
        status = "NEAR_UNIFORM_WEIGHT_WARNING"
    else:
        status = "PASS"
    diagnostics = {
        "schema_version": "cce-primary-m13-v2-diagnostics-v2",
        "status": status,
        "method_id": M13_ID,
        "seed": seed,
        "scientific_status": "POST_GOLD_METHOD_CORRECTION",
        "source_input_hashes": bundle["input_hashes"],
        "fixed_fold_file": bundle["fold_path"].as_posix(),
        "fixed_fold_sha256": bundle["input_hashes"]["fixed_fold"],
        "base_embedding_read_only_audits": embeddings["audits"],
        "source_reward_sha256": array_sha256(y),
        "strict_fit_count": fitted["strict_fit_count"],
        "pair_fit_records": fitted["pair_fit_records"],
        "outer_records": fitted["outer_records"],
        "coverage_sha256": fitted["coverage_sha256"],
        "parallel_fit_processes": fitted["parallel_fit_processes"],
        "base_oof_domain_auc": fitted["base_oof_domain_auc"],
        "calibrated_oof_domain_auc": fitted["calibrated_oof_domain_auc"],
        "calibrated_oof_domain_accuracy": fitted["calibrated_oof_domain_accuracy"],
        "calibrated_oof_brier": fitted["calibrated_oof_brier"],
        "calibrated_oof_log_loss": fitted["calibrated_oof_log_loss"],
        "base_decision_sd": fitted["base_decision_sd"],
        "calibrated_probability_sd": fitted["calibrated_probability_sd"],
        "source_weight_sum": float(np.sum(fitted["source_weight_oof"])),
        "source_weight_min": float(np.min(fitted["source_weight_oof"])),
        "source_weight_max": float(np.max(fitted["source_weight_oof"])),
        "source_weight_mean": float(np.mean(fitted["source_weight_oof"])),
        "source_weight_sd": fitted["source_weight_sd"],
        "source_weight_cv": fitted["source_weight_cv"],
        "source_weight_quantiles": fitted["source_weight_quantiles"],
        "source_weight_clip_fraction": fitted["source_weight_clip_fraction"],
        "ess": fitted["ess"],
        "ess_fraction": fitted["ess_fraction"],
        "positivity_warning": fitted["positivity_warning"],
        "near_uniform_weight_warning": fitted["near_uniform_weight_warning"],
        "calibration_collapse": fitted["calibration_collapse"],
        "m1_source_mean_unit": fitted["m1_source_mean_unit"],
        "m13_minus_m1_unit": fitted["m13_minus_m1_unit"],
        "target_rewards_accessed": False,
        "config": fitted["config"],
    }
    atomic_write_json(attempt / "diagnostics.json", diagnostics)
    atomic_write_json(
        attempt / "fit_summary.json",
        {
            "schema_version": "cce-primary-m13-m14-corrected-fit-summary-v2",
            "status": "COMPLETE",
            "method_id": M13_ID,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "source_rows": len(source_ingredients),
            "target_rows": len(target_density),
            "row_prediction_rows": 0,
            "V_hat_unit": fitted["v_hat_unit"],
            "M1_source_mean_unit": fitted["m1_source_mean_unit"],
            "delta_from_M1_unit": fitted["m13_minus_m1_unit"],
            "ess": fitted["ess"],
            "ess_fraction": fitted["ess_fraction"],
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _commit_attempt(attempt, final, method_id=M13_ID, task=task, seed=seed)
    logger(
        f"DONE {M13_ID}/{task['panel_id']}/seed={seed} "
        f"delta_M1={fitted['m13_minus_m1_unit']:.6g} ESS/n={fitted['ess_fraction']:.4f} "
        f"seconds={time.perf_counter()-started:.1f}"
    )


def _read_upstream_outcome_oof(
    *,
    upstream_run_root: Path,
    upstream_receipt: dict[str, Any],
    bundle: dict[str, Any],
    seed: int,
    y: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    task = bundle["task"]
    upstream_m14 = v1._cell_dir(  # noqa: SLF001
        upstream_run_root, "M14_SNDR_BGE", task, seed
    )
    v1._validate_cell(upstream_m14, "M14_SNDR_BGE", task, seed)  # noqa: SLF001
    checkpoint = upstream_m14 / "outcome_oof_checkpoint.npz"
    checkpoint_hash = _validate_receipt_member(
        upstream_run_root, upstream_receipt, checkpoint
    )
    with np.load(checkpoint, allow_pickle=False) as payload:
        if not {
            "source_oof_prediction_unit",
            "source_residual_oof",
            "source_weight_oof",
        } <= set(payload.files):
            raise RuntimeError("Upstream M14 OOF checkpoint schema drift.")
        outcome = np.asarray(payload["source_oof_prediction_unit"], dtype=np.float64)
        stored_residual = np.asarray(payload["source_residual_oof"], dtype=np.float64)
        ignored_old_weight_hash = array_sha256(
            np.asarray(payload["source_weight_oof"], dtype=np.float64)
        )
    if outcome.shape != y.shape or stored_residual.shape != y.shape:
        raise RuntimeError("Upstream outcome OOF shape drift.")
    recomputed = y - outcome
    if not np.array_equal(recomputed, stored_residual):
        raise RuntimeError("Upstream Source OOF residual does not exactly recompute.")
    ingredients_path = upstream_m14 / "source_bootstrap_ingredients.jsonl"
    ingredients_hash = _validate_receipt_member(
        upstream_run_root, upstream_receipt, ingredients_path
    )
    ingredients = read_jsonl(ingredients_path)
    if [str(row["item_id"]) for row in ingredients] != [
        str(row["item_id"]) for row in bundle["train_rows"]
    ]:
        raise RuntimeError("Upstream M14 Source ingredient ordering drift.")
    if not np.array_equal(
        np.asarray([float(row["m_source_oof_unit"]) for row in ingredients]), outcome
    ) or not np.array_equal(
        np.asarray([float(row["residual_oof_unit"]) for row in ingredients]), recomputed
    ):
        raise RuntimeError("Upstream M14 Source ingredient/outcome OOF mismatch.")
    return outcome, {
        "upstream_m14_outcome_checkpoint_path": checkpoint.as_posix(),
        "upstream_m14_outcome_checkpoint_sha256": checkpoint_hash,
        "source_oof_prediction_unit_sha256": array_sha256(outcome),
        "source_residual_oof_sha256": array_sha256(stored_residual),
        "upstream_source_ingredients_sha256": ingredients_hash,
        "old_source_weight_field_ignored": True,
        "ignored_old_source_weight_sha256": ignored_old_weight_hash,
    }


def run_m14(
    *,
    output_root: Path,
    upstream_run_root: Path,
    upstream_receipt: dict[str, Any],
    bundle: dict[str, Any],
    seed: int,
    resume: bool,
    logger: Callable[[str], None],
) -> None:
    task = bundle["task"]
    final = _cell_dir(output_root, M14_ID, task, seed)
    if _skip_or_block_cell(final, M14_ID, task, seed, resume):
        logger(f"RESUME validated {M14_ID}/{task['panel_id']}/seed={seed}")
        return
    m13_dir = _cell_dir(output_root, M13_ID, task, seed)
    _validate_cell(m13_dir, M13_ID, task, seed)
    upstream_m3 = v1._cell_dir(  # noqa: SLF001
        upstream_run_root, "M3_FORMAL_BGE_M3_DM", task, seed
    )
    v1._validate_cell(upstream_m3, "M3_FORMAL_BGE_M3_DM", task, seed)  # noqa: SLF001
    m3_path = upstream_m3 / "row_predictions.jsonl"
    m3_hash = _validate_receipt_member(upstream_run_root, upstream_receipt, m3_path)
    m3_rows = read_jsonl(m3_path)
    if [str(row["item_id"]) for row in m3_rows] != [
        str(row["item_id"]) for row in bundle["test_rows"]
    ]:
        raise RuntimeError("Upstream M3 Target row order drift.")
    target_dm = np.asarray([float(row["prediction_unit"]) for row in m3_rows])

    attempt = _new_attempt(output_root, M14_ID, task, seed)
    started = time.perf_counter()
    y = normalize_score(
        np.asarray([row["score"] for row in bundle["train_rows"]]),
        float(task["native_scale_min"]),
        float(task["native_scale_max"]),
    )
    outcome, outcome_audit = _read_upstream_outcome_oof(
        upstream_run_root=upstream_run_root,
        upstream_receipt=upstream_receipt,
        bundle=bundle,
        seed=seed,
        y=y,
    )
    m13_ingredients_path = m13_dir / "source_bootstrap_ingredients.jsonl"
    m13_ingredients = read_jsonl(m13_ingredients_path)
    if [str(row["item_id"]) for row in m13_ingredients] != [
        str(row["item_id"]) for row in bundle["train_rows"]
    ]:
        raise RuntimeError("Corrected M14/M13 Source ingredient order drift.")
    weights = np.asarray([float(row["clipped_weight"]) for row in m13_ingredients])
    prediction, correction = sndr_predictions(target_dm, y, outcome, weights)
    rows = _row_predictions(M14_ID, task, bundle["test_rows"], seed, prediction)
    source_ingredients = [
        {
            "schema_version": "cce-primary-m14-v2-source-ingredient-v2",
            "method_id": M14_ID,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "item_id": str(row["item_id"]),
            "source_responder_id": str(row["source_responder_id"]),
            "outer_fold": int(bundle["source_folds"][index]),
            "y_source_unit": float(y[index]),
            "m_source_oof_unit": float(outcome[index]),
            "residual_oof_unit": float(y[index] - outcome[index]),
            "m13_clipped_weight": float(weights[index]),
            "contains_target_reward": False,
            "post_gold_method_correction": True,
        }
        for index, row in enumerate(bundle["train_rows"])
    ]
    atomic_write_jsonl(attempt / "row_predictions.jsonl", rows)
    atomic_write_jsonl(attempt / "source_bootstrap_ingredients.jsonl", source_ingredients)
    atomic_write_json(
        attempt / "panel_value.json",
        _panel_value(
            M14_ID, task, seed, float(np.mean(prediction)), "target_row_prediction_mean"
        ),
    )
    atomic_write_npz(
        attempt / "corrected_outcome_oof_checkpoint.npz",
        source_oof_prediction_unit=outcome,
        source_weight_oof=weights,
        source_residual_oof=y - outcome,
        target_dm_unit=target_dm,
        target_sndr_prediction_unit=prediction,
    )
    diagnostics = {
        "schema_version": "cce-primary-m14-v2-diagnostics-v2",
        "status": "PASS",
        "method_id": M14_ID,
        "required_display_name": "SNDR-BGE (corrected v2)",
        "offcem_claim": False,
        "scientific_status": "POST_GOLD_METHOD_CORRECTION",
        "seed": seed,
        "source_input_hashes": bundle["input_hashes"],
        "fixed_fold_file": bundle["fold_path"].as_posix(),
        "fixed_fold_sha256": bundle["input_hashes"]["fixed_fold"],
        "upstream_m3_row_predictions_path": m3_path.as_posix(),
        "upstream_m3_row_predictions_sha256": m3_hash,
        "upstream_m3_prediction_unit_sha256": array_sha256(target_dm),
        **outcome_audit,
        "corrected_m13_cell_manifest_sha256": sha256_file(
            m13_dir / "checkpoint_manifest.json"
        ),
        "corrected_m13_source_ingredients_sha256": sha256_file(m13_ingredients_path),
        **correction,
        "row_wise_clipping_before_mean": True,
        "target_rewards_accessed": False,
    }
    atomic_write_json(attempt / "diagnostics.json", diagnostics)
    atomic_write_json(
        attempt / "fit_summary.json",
        {
            "schema_version": "cce-primary-m13-m14-corrected-fit-summary-v2",
            "status": "COMPLETE",
            "method_id": M14_ID,
            "panel_id": task["panel_id"],
            "task_id": task["primary_task_id"],
            "seed": seed,
            "source_rows": len(source_ingredients),
            "target_rows": len(rows),
            "V_hat_unit": float(np.mean(prediction)),
            **correction,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _commit_attempt(attempt, final, method_id=M14_ID, task=task, seed=seed)
    logger(
        f"DONE {M14_ID}/{task['panel_id']}/seed={seed} "
        f"correction={correction['correction_unit']:.6g} "
        f"seconds={time.perf_counter()-started:.1f}"
    )


def _row_key(row: dict[str, Any]) -> tuple[str, str, int, str, str]:
    return (
        str(row["method_id"]),
        str(row["panel_id"]),
        int(row["seed"]),
        str(row["task_id"]),
        str(row["item_id"]),
    )


def _value_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["method_id"]),
        str(row["panel_id"]),
        int(row["seed"]),
        str(row["task_id"]),
    )


def _collect_and_validate(
    output_root: Path, bundles: Sequence[dict[str, Any]]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rows: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    ingredients: list[dict[str, Any]] = []
    target_density: list[dict[str, Any]] = []
    by_method_rows = {method: 0 for method in METHOD_ORDER}
    by_method_values = {method: 0 for method in METHOD_ORDER}
    for method_id in METHOD_ORDER:
        for bundle in bundles:
            task = bundle["task"]
            for seed in SEEDS:
                cell = _cell_dir(output_root, method_id, task, seed)
                _validate_cell(cell, method_id, task, seed)
                value = read_json(cell / "panel_value.json")
                values.append(value)
                by_method_values[method_id] += 1
                row_path = cell / "row_predictions.jsonl"
                if method_id == M13_ID:
                    if row_path.exists():
                        raise RuntimeError("Corrected M13 must not emit Target row predictions.")
                    density_rows = read_jsonl(cell / "target_density_oof.jsonl")
                    expected_target_items = [
                        str(row["item_id"]) for row in bundle["test_rows"]
                    ]
                    if [str(row["item_id"]) for row in density_rows] != expected_target_items:
                        raise RuntimeError("Corrected M13 Target-density key/order drift.")
                    target_density.extend(density_rows)
                else:
                    cell_rows = read_jsonl(row_path)
                    expected_items = [str(row["item_id"]) for row in bundle["test_rows"]]
                    if [str(row["item_id"]) for row in cell_rows] != expected_items:
                        raise RuntimeError("Corrected M14 Target prediction-key drift.")
                    if not math.isclose(
                        float(np.mean([float(row["prediction_unit"]) for row in cell_rows])),
                        float(value["V_hat_unit"]),
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError("Corrected M14 row mean/panel value mismatch.")
                    rows.extend(cell_rows)
                    by_method_rows[method_id] += len(cell_rows)
                ingredient_path = cell / "source_bootstrap_ingredients.jsonl"
                cell_ingredients = read_jsonl(ingredient_path)
                if len(cell_ingredients) != len(bundle["train_rows"]):
                    raise RuntimeError("Corrected Source ingredient count drift.")
                ingredients.extend(cell_ingredients)
    if by_method_rows != {M13_ID: 0, M14_ID: 14_766}:
        raise RuntimeError(f"Corrected by-method row counts drifted: {by_method_rows}")
    if by_method_values != {M13_ID: 36, M14_ID: 36}:
        raise RuntimeError(f"Corrected by-method value counts drifted: {by_method_values}")
    if (
        len(rows) != 14_766
        or len(values) != 72
        or len(ingredients) != 227_052
        or len(target_density) != 14_766
    ):
        raise RuntimeError("Corrected global aggregate counts drifted.")
    if len({_row_key(row) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate corrected row-prediction key.")
    if len({_value_key(row) for row in values}) != len(values):
        raise RuntimeError("Duplicate corrected panel-value key.")
    if len(
        {
            (
                str(row["method_id"]),
                str(row["panel_id"]),
                str(row["task_id"]),
                int(row["seed"]),
                str(row["item_id"]),
                str(row["target_responder_id"]),
            )
            for row in target_density
        }
    ) != len(target_density):
        raise RuntimeError("Duplicate corrected Target-density diagnostic key.")
    return (
        sorted(rows, key=_row_key),
        sorted(values, key=_value_key),
        sorted(
            ingredients,
            key=lambda row: (
                METHOD_ORDER.index(str(row["method_id"])),
                str(row["panel_id"]),
                int(row["seed"]),
                str(row["item_id"]),
                str(row.get("source_responder_id", "")),
            ),
        ),
        sorted(
            target_density,
            key=lambda row: (
                str(row["panel_id"]),
                int(row["seed"]),
                str(row["task_id"]),
                str(row["item_id"]),
            ),
        ),
    )


def _freeze(
    *,
    output_root: Path,
    bundles: Sequence[dict[str, Any]],
    registry: dict[str, Any],
    method_registry_path: Path,
    public_audit: dict[str, Any],
    upstream_run_root: Path,
    upstream_receipt: dict[str, Any],
    logger: Callable[[str], None],
) -> dict[str, Any]:
    rows, values, ingredients, target_density = _collect_and_validate(output_root, bundles)
    write_once_jsonl(output_root / "row_predictions.jsonl", rows)
    write_once_jsonl(output_root / "panel_values.jsonl", values)
    write_once_jsonl(output_root / "bootstrap_ingredients.jsonl", ingredients)
    write_once_jsonl(
        output_root / "target_density_diagnostics.jsonl", target_density
    )
    cells: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        for bundle in bundles:
            task = bundle["task"]
            for seed in SEEDS:
                path = _cell_dir(output_root, method_id, task, seed) / "checkpoint_manifest.json"
                cells.append(
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
            "schema_version": "cce-primary-m13-m14-corrected-root-manifest-v2",
            "status": "COMPLETE",
            "cell_count": len(cells),
            "cells": cells,
        },
    )
    artifacts: dict[str, dict[str, Any]] = {}
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
        artifacts[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    receipt = {
        "schema_version": "cce-primary-m13-m14-corrected-freeze-receipt-v2",
        "status": RECEIPT_STATUS,
        "scientific_status": "POST_V1_GOLD_PROTOCOL_CORRECTION / RETROSPECTIVE_ON_SAME_GOLD",
        "frozen_at_utc": utc_now(),
        "method_registry": {
            "path": method_registry_path.resolve().as_posix(),
            "file_sha256": sha256_file(method_registry_path),
            "payload_sha256": registry_payload_sha256(registry),
        },
        "upstream_v1": {
            "run_root": upstream_run_root.resolve().as_posix(),
            "freeze_receipt_sha256": sha256_file(
                upstream_run_root / "prediction_freeze_receipt.json"
            ),
            "artifact_map_canonical_sha256": upstream_receipt[
                "artifact_map_canonical_sha256"
            ],
        },
        "primary_task_registry": {
            "file_sha256": public_audit["primary_task_registry_file_sha256"],
            "payload_sha256": public_audit["primary_task_registry_payload_sha256"],
        },
        "public_inputs": public_audit["public_input_hashes"],
        "public_archive_sha256": public_audit["public_archive_sha256"],
        "counts": {
            "row_predictions": len(rows),
            "panel_values": len(values),
            "bootstrap_ingredients": len(ingredients),
            "target_density_diagnostics": len(target_density),
            "checkpoint_cells": len(cells),
        },
        "excluded_panels": ["ayers_askdocs", "simpeval_past"],
        "target_rewards_accessed_by_correction_runner": False,
        "private_gold_accessed_by_correction_runner": False,
        "historical_private_gold_access_before_v2_design": True,
        "artifacts": artifacts,
        "artifact_map_canonical_sha256": hashlib.sha256(
            canonical_json_bytes(artifacts)
        ).hexdigest(),
    }
    receipt_path = output_root / "prediction_freeze_receipt.json"
    write_exclusive_json(receipt_path, receipt)
    receipt_hash = sha256_file(receipt_path)
    write_once_bytes(
        output_root / "prediction_freeze_receipt.sha256",
        f"{receipt_hash}  prediction_freeze_receipt.json\n".encode("ascii"),
    )
    logger(f"FROZEN corrected predictions receipt_sha256={receipt_hash}")
    return receipt


def _validate_existing_freeze(output_root: Path) -> dict[str, Any]:
    receipt_path = output_root / "prediction_freeze_receipt.json"
    sidecar_path = output_root / "prediction_freeze_receipt.sha256"
    if not receipt_path.is_file() or not sidecar_path.is_file():
        raise RuntimeError("Corrected freeze is incomplete.")
    expected = sidecar_path.read_text(encoding="ascii").split()[0]
    if sha256_file(receipt_path) != expected:
        raise RuntimeError("Corrected freeze receipt hash drift.")
    receipt = read_json(receipt_path)
    if receipt.get("status") != RECEIPT_STATUS:
        raise RuntimeError("Corrected freeze status drift.")
    for relative, record in receipt["artifacts"].items():
        path = output_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"Corrected frozen artifact drift: {path}")
    if hashlib.sha256(canonical_json_bytes(receipt["artifacts"])).hexdigest() != receipt[
        "artifact_map_canonical_sha256"
    ]:
        raise RuntimeError("Corrected freeze artifact-map hash drift.")
    return receipt


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    public_root = args.public_root.resolve()
    upstream_run_root = args.upstream_run_root.resolve()
    method_registry_path = args.method_registry.resolve()
    output_root = args.output_root.resolve()
    if any(part.lower() in FORBIDDEN_OUTPUT_PARTS for part in output_root.parts):
        raise RuntimeError("Correction output path may not contain private/gold components.")
    if output_root.exists() and (output_root / "prediction_freeze_receipt.json").exists():
        if not args.resume:
            raise RuntimeError("Corrected output is frozen; overwrite is forbidden.")
        return _validate_existing_freeze(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise RuntimeError("Nonempty corrected output root requires --resume.")
    output_root.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_root / "run.log")
    logger("PRECHECK corrected-v2 start")
    registry, _, bundles, public_audit, upstream_receipt = (
        _validate_registry_and_upstream(
            public_root=public_root,
            upstream_run_root=upstream_run_root,
            method_registry_path=method_registry_path,
        )
    )
    actual_device = args.device
    if actual_device != "mps":
        raise RuntimeError("Formal corrected run requires the registry-bound mps cache provenance.")
    implementation_files = (
        Path(__file__).resolve(),
        (REPO_ROOT / "src/cce_data/estimators/cce_primary_zero_label_m13_m14_v2.py").resolve(),
        (REPO_ROOT / "src/cce_data/estimators/cce_primary_zero_label.py").resolve(),
        (REPO_ROOT / "scripts/run_cce_benchmark_primary_zero_label_methods_v1.py").resolve(),
        (REPO_ROOT / "scripts/run_cce_benchmark_primary_dm_bge_m3_learned_gbr.py").resolve(),
    )
    implementation_manifest = {
        "schema_version": "cce-primary-m13-m14-corrected-implementation-manifest-v2",
        "status": "HASH_BOUND",
        "files": {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
            for path in implementation_files
        },
        "strict_representation_and_base_fits_per_cell": 15,
        "isolated_fit_processes_per_cell": 5,
        "inner_torch_and_blas_threads": 1,
        "nuisance_compute_device": "cpu",
        "requested_device_role": "validate mps-origin upstream BGE cache provenance",
        "runner_imports_evaluator": False,
        "runner_cli_private_or_gold_argument": False,
    }
    implementation_path = output_root / "implementation_manifest.json"
    if implementation_path.exists():
        if read_json(implementation_path) != implementation_manifest:
            raise RuntimeError("Corrected resume implementation hash drift.")
    else:
        atomic_write_json(implementation_path, implementation_manifest)
    runtime_path = output_root / "runtime_manifest.json"
    runtime = {
        "schema_version": "cce-primary-m13-m14-corrected-runtime-manifest-v2",
        "status": "RUNNING_CORRECTION_RUNNER_GOLD_FREE",
        "started_at_utc": utc_now(),
        "method_registry_file_sha256": sha256_file(method_registry_path),
        "method_registry_payload_sha256": registry_payload_sha256(registry),
        "upstream_v1_freeze_receipt_sha256": sha256_file(
            upstream_run_root / "prediction_freeze_receipt.json"
        ),
        "primary_task_registry_file_sha256": public_audit[
            "primary_task_registry_file_sha256"
        ],
        "primary_task_registry_payload_sha256": public_audit[
            "primary_task_registry_payload_sha256"
        ],
        "public_root": public_root.as_posix(),
        "upstream_run_root": upstream_run_root.as_posix(),
        "requested_device": args.device,
        "validated_upstream_cache_device": actual_device,
        "nuisance_compute_device": "cpu",
        "private_gold_accessed_by_correction_runner": False,
        "target_rewards_accessed_by_correction_runner": False,
        "historical_private_gold_access_before_v2_design": True,
    }
    if runtime_path.exists():
        existing = read_json(runtime_path)
        stable = {key: value for key, value in runtime.items() if key not in {"status", "started_at_utc"}}
        if any(existing.get(key) != value for key, value in stable.items()):
            raise RuntimeError("Corrected resume runtime-manifest drift.")
        runtime["started_at_utc"] = existing["started_at_utc"]
    else:
        atomic_write_json(runtime_path, runtime)
    logger(f"PRECHECK pass panels={len(bundles)} upstream_receipt={runtime['upstream_v1_freeze_receipt_sha256']}")

    for bundle in bundles:
        task = bundle["task"]
        logger(f"CACHE validate read-only {task['panel_id']}/{task['primary_task_id']}")
        embeddings = _load_base_embeddings_read_only(
            upstream_run_root=upstream_run_root,
            upstream_receipt=upstream_receipt,
            bundle=bundle,
        )
        for seed in SEEDS:
            run_m13(
                output_root=output_root,
                bundle=bundle,
                embeddings=embeddings,
                seed=seed,
                resume=args.resume,
                logger=logger,
            )
        for seed in SEEDS:
            run_m14(
                output_root=output_root,
                upstream_run_root=upstream_run_root,
                upstream_receipt=upstream_receipt,
                bundle=bundle,
                seed=seed,
                resume=args.resume,
                logger=logger,
            )
    runtime["status"] = "COMPLETE_CORRECTION_RUNNER_GOLD_FREE"
    runtime["completed_at_utc"] = utc_now()
    atomic_write_json(runtime_path, runtime)
    return _freeze(
        output_root=output_root,
        bundles=bundles,
        registry=registry,
        method_registry_path=method_registry_path,
        public_audit=public_audit,
        upstream_run_root=upstream_run_root,
        upstream_receipt=upstream_receipt,
        logger=logger,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--upstream-run-root", type=Path, default=DEFAULT_UPSTREAM_RUN_ROOT)
    parser.add_argument("--method-registry", type=Path, default=DEFAULT_METHOD_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def runner_cli_is_gold_free() -> bool:
    source = inspect.getsource(parse_args).lower()
    return not any(
        token in source for token in ("--private", "--gold", "private-root", "gold-path")
    )


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
