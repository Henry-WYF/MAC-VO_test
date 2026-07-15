from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Module.Frontend.Frontend import IFrontend
from Module.LoopClosure import LoopClosureManager, LoopFrameRecord
from Module.Map import VisualMap
from Utility.Config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _pose_matrix(pose: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(7)
    translation = values[:3]
    x, y, z, w = values[3:]
    norm = float(np.linalg.norm([x, y, z, w]))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("invalid pose quaternion")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def evaluate_gt_pose_proxy(
    constraints_path: Path,
    ref_poses_path: Path,
    record_dir: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate accepted PnP poses using the experiment's frozen, offline-only proxy."""
    definition = {
        "relative_pose": "inverse(T_world_current) @ T_world_candidate",
        "estimated_field": "pnp_relative_pose",
        "long_span_min_sensor_frames": 300,
        "accurate": "translation_error_m < 1 and rotation_error_deg < 5",
        "large_error": "translation_error_m > 3 or rotation_error_deg > 15",
        "remainder": "suspicious",
        "interpretation": "offline pose-error proxy; not an image-overlap or precision label",
    }
    if not ref_poses_path.is_file():
        return {"available": False, "reason": "ref_poses.npy missing", "definition": definition}
    reference = np.load(ref_poses_path, allow_pickle=False)
    if reference.ndim != 2 or reference.shape[1] != 8:
        return {"available": False, "reason": "invalid ref_poses.npy shape", "definition": definition}
    payload = _load_json(constraints_path)
    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        raise ValueError(f"constraints list missing from {constraints_path}")
    record_by_sensor = {int(row["sensor_frame_idx"]): row for row in records}
    counts = {"accurate": 0, "suspicious": 0, "large_error": 0}
    long_span_accurate = 0
    excluded = 0
    evaluated: list[dict[str, Any]] = []
    for constraint in constraints:
        candidate_idx = int(constraint["src_sensor_frame_idx"])
        current_idx = int(constraint["dst_sensor_frame_idx"])
        if not (0 <= candidate_idx < len(reference) and 0 <= current_idx < len(reference)):
            excluded += 1
            continue
        candidate_record = record_by_sensor.get(candidate_idx)
        current_record = record_by_sensor.get(current_idx)
        if candidate_record is None or current_record is None:
            excluded += 1
            continue
        try:
            candidate_ns = LoopFrameRecord.load(record_dir / str(candidate_record["file"])).frame_ns
            current_ns = LoopFrameRecord.load(record_dir / str(current_record["file"])).frame_ns
        except (OSError, KeyError, ValueError, TypeError):
            excluded += 1
            continue
        if int(reference[candidate_idx, 0]) != candidate_ns or int(reference[current_idx, 0]) != current_ns:
            excluded += 1
            continue
        world_candidate = _pose_matrix(reference[candidate_idx, 1:])
        world_current = _pose_matrix(reference[current_idx, 1:])
        gt_relative = np.linalg.inv(world_current) @ world_candidate
        estimated = _pose_matrix(constraint["pnp_relative_pose"])
        error = np.linalg.inv(gt_relative) @ estimated
        translation_error = float(np.linalg.norm(error[:3, 3]))
        cosine = float(np.clip((np.trace(error[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        rotation_error = float(np.degrees(np.arccos(cosine)))
        if translation_error < 1.0 and rotation_error < 5.0:
            label = "accurate"
        elif translation_error > 3.0 or rotation_error > 15.0:
            label = "large_error"
        else:
            label = "suspicious"
        counts[label] += 1
        span = current_idx - candidate_idx
        if label == "accurate" and span >= 300:
            long_span_accurate += 1
        evaluated.append(
            {
                "current_sensor_frame_idx": current_idx,
                "candidate_sensor_frame_idx": candidate_idx,
                "sensor_frame_span": span,
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
                "label": label,
            }
        )
    return {
        "available": True,
        "definition": definition,
        "accepted_constraints": len(constraints),
        "evaluated_constraints": len(evaluated),
        "excluded_unaligned_or_missing": excluded,
        "counts": counts,
        "long_span_accurate": long_span_accurate,
        "rows": evaluated,
    }


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
    parser.add_argument(
        "--adaptive-target-points",
        type=_positive_int,
        help="Covariance-stage minimum for gate-on; omit to reproduce the fixed threshold gate.",
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
    # Explicitly override the YAML so an omitted CLI option always means the fixed baseline.
    loop_config.geometry.flow_cov_adaptive_target_points = args.adaptive_target_points
    loop_config.geometric_verification.compare_flow_cov_gate = True
    frontend_config.args.device = args.device
    LoopClosureManager.is_valid_config(loop_config)

    index_payload = _load_json(index_path)
    queries_payload = _load_json(queries_path)
    if int(index_payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported loop index schema")
    if int(queries_payload.get("schema_version", -1)) not in {1, 2}:
        raise ValueError("unsupported Phase A queries schema")
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
    gt_pose_proxy = {
        "gate_enabled": evaluate_gt_pose_proxy(
            output_dir / "loop_constraints_gate_enabled.json",
            result_dir / "ref_poses.npy",
            record_dir,
            records,
        ),
        "gate_disabled": evaluate_gt_pose_proxy(
            output_dir / "loop_constraints_gate_disabled.json",
            result_dir / "ref_poses.npy",
            record_dir,
            records,
        ),
    }

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
        "code_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "code_source_sha256": {
            "verification": _sha256(
                Path(__file__).resolve().parents[2] / "Module/LoopClosure/Verification.py"
            ),
            "manager": _sha256(
                Path(__file__).resolve().parents[2] / "Module/LoopClosure/Manager.py"
            ),
            "offline_runner": _sha256(Path(__file__).resolve()),
            "config": _sha256(config_path),
        },
        "device": args.device,
        "configured_frontend_type": configured_frontend_type,
        "runtime_frontend_type": runtime_frontend_type,
        "nvtx_disabled_for_cpu": nvtx_disabled_for_cpu,
        "primary_gate_enabled": loop_config.geometry.flow_cov_gate_enabled,
        "comparison_enabled": True,
        "gate_enabled_selection_mode": (
            "fixed" if args.adaptive_target_points is None else "adaptive"
        ),
        "adaptive_target_points": args.adaptive_target_points,
        "max_total_candidates": args.max_total_candidates,
        "selected_candidates": selected_candidates,
        "elapsed_seconds": elapsed_seconds,
        "primary_constraints": len(constraints),
        "pose_invariant": True,
        "comparison_validation": comparison_validation,
        "gt_pose_proxy": gt_pose_proxy,
    }
    temporary = output_dir / "offline_run_manifest.json.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.flush()
    temporary.replace(output_dir / "offline_run_manifest.json")


if __name__ == "__main__":
    main()
