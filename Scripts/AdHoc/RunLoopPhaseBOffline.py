from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Module.Frontend.Frontend import IFrontend
from Module.LoopClosure import LoopClosureManager
from Module.Map import VisualMap
from Utility.Config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    with open(path, "r", encoding="utf-8") as file:
        value = json.load(file, parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_comparison_outputs(output_dir: Path) -> dict[str, Any]:
    payloads = {
        name: _load_json(output_dir / name)
        for name in (
            "loop_verification.json",
            "loop_constraints.json",
            "loop_verification_gate_enabled.json",
            "loop_verification_gate_disabled.json",
            "loop_constraints_gate_enabled.json",
            "loop_constraints_gate_disabled.json",
        )
    }
    main = payloads["loop_verification.json"]
    enabled = payloads["loop_verification_gate_enabled.json"]
    disabled = payloads["loop_verification_gate_disabled.json"]
    if main.get("comparison_enabled") is not True:
        raise ValueError("main verification is not a paired comparison result")
    run_ids = {
        main.get("comparison_run_id"),
        enabled.get("comparison_run_id"),
        disabled.get("comparison_run_id"),
    }
    if None in run_ids or len(run_ids) != 1:
        raise ValueError("verification files do not share one comparison_run_id")
    if main.get("frontend_inference_calls") != main.get("candidates_reaching_frontend"):
        raise ValueError("Frontend inference call count does not match candidates reaching Frontend")

    def paired_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        rows = payload.get("verifications")
        if not isinstance(rows, list):
            raise ValueError("verification payload has no verifications list")
        paired: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("comparison_pair_id"), str):
                raise ValueError("verification row has no comparison_pair_id")
            pair_id = row["comparison_pair_id"]
            if pair_id in paired:
                raise ValueError(f"duplicate comparison_pair_id: {pair_id}")
            paired[pair_id] = row
        return paired

    enabled_rows = paired_rows(enabled)
    disabled_rows = paired_rows(disabled)
    if set(enabled_rows) != set(disabled_rows):
        raise ValueError("gate-on and gate-off pair_id sets differ")
    primary_enabled = main.get("primary_gate_enabled")
    selected_verification = enabled if primary_enabled is True else disabled
    selected_constraints = (
        payloads["loop_constraints_gate_enabled.json"]
        if primary_enabled is True
        else payloads["loop_constraints_gate_disabled.json"]
    )
    if main.get("branch_gate_enabled") is not primary_enabled:
        raise ValueError("main verification does not belong to the configured primary branch")
    if main.get("verifications") != selected_verification.get("verifications"):
        raise ValueError("main verification rows differ from the named primary branch")
    if payloads["loop_constraints.json"] != selected_constraints:
        raise ValueError("main constraints differ from the named primary branch")

    applicable = [
        pair_id
        for pair_id in enabled_rows
        if enabled_rows[pair_id].get("comparison_applicable") is True
        and disabled_rows[pair_id].get("comparison_applicable") is True
    ]

    def diagnostic(row: dict[str, Any], key: str) -> Any:
        diagnostics = row.get("diagnostics")
        return diagnostics.get(key) if isinstance(diagnostics, dict) else None

    flow_restored = [
        pair_id
        for pair_id in applicable
        if isinstance(diagnostic(enabled_rows[pair_id], "after_flow_cov"), int)
        and isinstance(diagnostic(disabled_rows[pair_id], "after_flow_cov"), int)
        and diagnostic(disabled_rows[pair_id], "after_flow_cov")
        > diagnostic(enabled_rows[pair_id], "after_flow_cov")
    ]
    result = {
        "strict_json_valid": True,
        "comparison_run_id": next(iter(run_ids)),
        "pair_ids_unique_and_aligned": True,
        "pair_count": len(enabled_rows),
        "comparison_applicable_candidates": len(applicable),
        "frontend_inference_calls": main.get("frontend_inference_calls"),
        "candidates_reaching_frontend": main.get("candidates_reaching_frontend"),
        "gate_off_restored_flow_candidates": len(flow_restored),
        "gate_off_new_pnp_attempts": sum(
            disabled_rows[pair_id].get("pnp_attempted") is True
            and enabled_rows[pair_id].get("pnp_attempted") is not True
            for pair_id in applicable
        ),
        "gate_off_new_pnp_ransac_successes": sum(
            disabled_rows[pair_id].get("pnp_ransac_succeeded") is True
            and enabled_rows[pair_id].get("pnp_ransac_succeeded") is not True
            for pair_id in applicable
        ),
        "gate_off_new_acceptances": sum(
            disabled_rows[pair_id].get("status") == "accepted"
            and enabled_rows[pair_id].get("status") != "accepted"
            for pair_id in applicable
        ),
    }
    comparison_summary = main.get("comparison_summary")
    if not isinstance(comparison_summary, dict):
        raise ValueError("main verification has no comparison_summary")
    if comparison_summary.get("pair_count") != result["pair_count"]:
        raise ValueError("comparison_summary pair_count does not match verification rows")
    if (
        comparison_summary.get("comparison_applicable_candidates")
        != result["comparison_applicable_candidates"]
    ):
        raise ValueError("comparison_summary applicable count does not match verification rows")
    return result


def _load_visual_map(path: Path) -> VisualMap:
    with np.load(path, allow_pickle=False) as archive:
        pose_key = next(
            (key for key in ("frames//pose", "frames/pose") if key in archive.files),
            None,
        )
        if pose_key is None:
            raise ValueError(f"tensor map has no serialized frame poses: {path}")
        poses = torch.from_numpy(archive[pose_key].copy()).to(dtype=torch.float32)
    global_map = VisualMap()
    global_map.frames.data["pose"].push(poses)
    return global_map


def limit_queries(
    queries: list[dict[str, Any]], max_total_candidates: int | None
) -> list[dict[str, Any]]:
    if max_total_candidates is None:
        return queries
    if max_total_candidates <= 0:
        raise ValueError("max_total_candidates must be positive")
    remaining = max_total_candidates
    limited: list[dict[str, Any]] = []
    for query in queries:
        candidates = query.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError("query candidates must be a list")
        copied = dict(query)
        copied["candidates"] = candidates[:remaining]
        limited.append(copied)
        remaining -= len(copied["candidates"])
        if remaining == 0:
            break
    return limited


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired Phase B covariance-gate verification from an existing MAC-VO cache."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Frontend device. CPU is supported for diagnostics but is much slower than CUDA.",
    )
    parser.add_argument(
        "--primary-gate",
        choices=("enabled", "disabled"),
        default="disabled",
    )
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--max-total-candidates",
        type=int,
        help="Optional smoke-test limit; omit for the complete paired experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result_dir = args.result_dir.resolve()
    record_dir = result_dir / "loop_closure"
    config_path = (args.config or result_dir / "config.yaml").resolve()
    output_dir = (args.output_dir or result_dir / "loop_closure_phase_b_ab").resolve()
    index_path = record_dir / "index.json"
    queries_path = record_dir / "queries.json"
    map_path = result_dir / "tensor_map.npz"

    for required in (config_path, index_path, queries_path, map_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    config, _ = load_config(config_path)
    loop_config = config.Odometry.loop_closure
    frontend_config = config.Odometry.frontend
    loop_config.geometry.flow_cov_gate_enabled = args.primary_gate == "enabled"
    loop_config.geometric_verification.compare_flow_cov_gate = True
    frontend_config.args.device = args.device
    LoopClosureManager.is_valid_config(loop_config)

    index_payload = _load_json(index_path)
    queries_payload = _load_json(queries_path)
    records = index_payload.get("records")
    queries = queries_payload.get("queries")
    if not isinstance(records, list) or not isinstance(queries, list):
        raise ValueError("index.json or queries.json has an invalid Phase A payload")
    queries = limit_queries(queries, args.max_total_candidates)
    selected_candidates = sum(len(query.get("candidates", [])) for query in queries)

    configured_frontend_type = str(frontend_config.type)
    runtime_frontend_type = configured_frontend_type
    nvtx_disabled_for_cpu = False
    if args.device == "cpu":
        if configured_frontend_type == "CUDAGraph_FlowFormerCovFrontend":
            runtime_frontend_type = "FlowFormerCovFrontend"
        setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: contextlib.nullcontext())
        nvtx_disabled_for_cpu = True
    frontend = IFrontend.instantiate(runtime_frontend_type, frontend_config.args)
    manager = LoopClosureManager(loop_config)
    manager.set_output_dir(output_dir)
    if not manager.enabled or manager.output_dir is None:
        raise RuntimeError(f"loop manager could not initialize: {manager.disabled_reason}")
    manager.records = records
    manager.set_frontend(frontend)

    global_map = _load_visual_map(map_path)
    pose_before = global_map.frames.data["pose"].tensor.detach().clone()
    started = time.perf_counter()
    constraints = manager.verify_candidates(
        global_map,
        queries,
        record_dir=record_dir,
        progress_interval=args.progress_interval,
    )
    elapsed_seconds = time.perf_counter() - started
    if not torch.equal(global_map.frames.data["pose"].tensor, pose_before):
        raise RuntimeError("offline Phase B verification modified VisualMap poses")
    comparison_validation = validate_comparison_outputs(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index_path, output_dir / "source_index.json")
    shutil.copy2(queries_path, output_dir / "source_queries.json")
    manifest = {
        "schema_version": 1,
        "source_result_dir": str(result_dir),
        "source_config": str(config_path),
        "source_index_sha256": _sha256(index_path),
        "source_queries_sha256": _sha256(queries_path),
        "source_tensor_map_sha256": _sha256(map_path),
        "device": args.device,
        "configured_frontend_type": configured_frontend_type,
        "runtime_frontend_type": runtime_frontend_type,
        "nvtx_disabled_for_cpu": nvtx_disabled_for_cpu,
        "primary_gate_enabled": loop_config.geometry.flow_cov_gate_enabled,
        "comparison_enabled": True,
        "max_total_candidates": args.max_total_candidates,
        "selected_candidates": selected_candidates,
        "elapsed_seconds": elapsed_seconds,
        "primary_constraints": len(constraints),
        "pose_invariant": True,
        "comparison_validation": comparison_validation,
    }
    temporary = output_dir / "offline_run_manifest.json.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.flush()
    temporary.replace(output_dir / "offline_run_manifest.json")


if __name__ == "__main__":
    main()
