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
from Module.LoopClosure.VINSGeometry import (
    ORB_SLAM_HAMMING_THRESHOLD,
    ORB_SLAM_ORIENTATION_BINS,
    ORB_SLAM_ORIENTATION_WEAK_BIN_RATIO,
    ORB_SLAM_RATIO_THRESHOLD,
    run_pose_copy_pgo_comparison,
)
from Module.Map import VisualMap
from Module.Optimization.GlobalPGO import GlobalPoseGraphOptimizer
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


def summarize_engineering_admission(
    gt_proxy: dict[str, Any], pose_copy_pgo: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply the frozen, engineering-only Phase C smoke-test criteria."""
    requirements = {
        "min_gt_evaluated_edges": 3,
        "min_accurate_edges": 2,
        "min_distinct_loop_queries": 2,
        "required_gt_coverage": 1.0,
        "max_large_error_edges": 0,
        "pose_copy_pgo_executed": True,
        "pose_copy_pgo_safe": True,
        "original_pose_invariant": True,
    }
    if not gt_proxy.get("available", False):
        return {
            "eligible": False,
            "reason": "gt_proxy_unavailable",
            "requirements": requirements,
            "pose_copy_pgo": pose_copy_pgo,
        }
    evaluated = int(gt_proxy.get("evaluated_constraints", 0))
    accepted = int(gt_proxy.get("accepted_constraints", 0))
    counts = gt_proxy.get("counts", {})
    accurate = int(counts.get("accurate", 0))
    large_error = int(counts.get("large_error", 0))
    rows = gt_proxy.get("rows", [])
    pgo = pose_copy_pgo or {}
    distinct_queries = len({int(row["current_sensor_frame_idx"]) for row in rows})
    coverage = float(evaluated / accepted) if accepted > 0 else 0.0
    checks = {
        "gt_evaluated_edges": evaluated >= requirements["min_gt_evaluated_edges"],
        "accurate_edges": accurate >= requirements["min_accurate_edges"],
        "distinct_loop_queries": distinct_queries >= requirements["min_distinct_loop_queries"],
        "gt_coverage": coverage == requirements["required_gt_coverage"],
        "large_error_edges": large_error <= requirements["max_large_error_edges"],
        "pose_copy_pgo_executed": pgo.get("executed") is True,
        "pose_copy_pgo_safe": pgo.get("safe") is True,
        "original_pose_invariant": pgo.get("original_pose_invariant") is True,
    }
    return {
        "eligible": all(checks.values()),
        "reason": None if all(checks.values()) else "engineering_admission_failed",
        "requirements": requirements,
        "observed": {
            "gt_evaluated_edges": evaluated,
            "accurate_edges": accurate,
            "distinct_loop_queries": distinct_queries,
            "gt_coverage": coverage,
            "large_error_edges": large_error,
            "pose_copy_pgo_executed": pgo.get("executed"),
            "pose_copy_pgo_safe": pgo.get("safe"),
            "original_pose_invariant": pgo.get("original_pose_invariant"),
        },
        "checks": checks,
        "interpretation": "engineering smoke-test gate; not evidence of paper-level sample sufficiency",
    }


def run_vins_pose_copy_pgo(
    global_map: VisualMap,
    global_pgo_config: Any,
    fixed_path: Path,
    covariance_path: Path,
    reference_poses: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run fixed/covariance information on identical eligible edges and pose copies."""
    fixed_rows = _load_json(fixed_path).get("constraints")
    covariance_rows = _load_json(covariance_path).get("constraints")
    if not isinstance(fixed_rows, list) or not isinstance(covariance_rows, list):
        raise ValueError("pose-copy PGO inputs must contain constraint lists")
    if len(fixed_rows) != len(covariance_rows):
        raise ValueError("fixed and covariance pose-copy edge counts differ")
    if not fixed_rows:
        return {"executed": False, "safe": False, "reason": "no_pgo_comparison_eligible_edges", "edge_count": 0}

    fixed_optimizer = GlobalPoseGraphOptimizer(global_pgo_config)
    covariance_optimizer = GlobalPoseGraphOptimizer(global_pgo_config)
    fixed_optimizer.register_odometry_edges(global_map, "fixed")
    covariance_optimizer.register_odometry_edges(
        global_map, "mixed_covariance_fixed",
    )
    for fixed, covariance in zip(fixed_rows, covariance_rows):
        identity_keys = (
            "src_visual_map_idx", "dst_visual_map_idx", "src_sensor_frame_idx",
            "dst_sensor_frame_idx", "relative_pose",
        )
        if any(fixed.get(key) != covariance.get(key) for key in identity_keys):
            raise ValueError("fixed and covariance pose-copy edges are not identical")
        fixed_optimizer.add_loop_edge(
            int(fixed["src_visual_map_idx"]), int(fixed["dst_visual_map_idx"]),
            torch.tensor(fixed["relative_pose"]), torch.tensor(fixed["information"]),
        )
        covariance_optimizer.add_loop_edge(
            int(covariance["src_visual_map_idx"]), int(covariance["dst_visual_map_idx"]),
            torch.tensor(covariance["relative_pose"]), torch.tensor(covariance["information"]),
        )
    initial = global_map.frames.data["pose"].tensor.detach().clone()
    result = run_pose_copy_pgo_comparison(
        fixed_optimizer, covariance_optimizer, initial, reference_poses,
    )
    result["executed"] = True
    result["eligible_loop_edge_count"] = len(fixed_rows)
    diagnostics = covariance_optimizer.odometry_information_diagnostics
    fallback = [row for row in diagnostics if row.get("fallback") is True]
    result["odometry_information"] = {
        "branch_name": "mixed_covariance_fixed_information",
        "total_edges": len(diagnostics),
        "observation_hessian_edges": len(diagnostics) - len(fallback),
        "fallback_edges": len(fallback),
        "fallback_ratio": len(fallback) / max(len(diagnostics), 1),
        "fallback_details": fallback,
    }
    result["original_pose_invariant"] = torch.equal(
        global_map.frames.data["pose"].tensor, initial
    )
    if not result["original_pose_invariant"]:
        raise RuntimeError("pose-copy PGO modified the original VisualMap poses")
    return result


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
    phase_b5_apply = (
        isinstance(main.get("phase_b5"), dict)
        and main["phase_b5"].get("mode") == "apply"
    )
    if not phase_b5_apply and payloads["loop_constraints.json"] != selected_constraints:
        raise ValueError("main constraints differ from the named primary branch")
    if phase_b5_apply:
        cascade = _load_json(output_dir / "cascade_apply/constraints.json")
        if payloads["loop_constraints.json"] != cascade:
            raise ValueError("main constraints differ from the Phase B.5 apply branch")

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
        interp_key = next(
            (key for key in ("frames//need_interp", "frames/need_interp") if key in archive.files),
            None,
        )
        need_interp = (
            torch.from_numpy(archive[interp_key].copy()).bool()
            if interp_key is not None else torch.zeros(len(poses), dtype=torch.bool)
        )
        time_key = next(
            (key for key in ("frames//time_ns", "frames/time_ns") if key in archive.files),
            None,
        )
        time_ns = torch.from_numpy(archive[time_key].copy()).long() if time_key is not None else None
    global_map = VisualMap()
    global_map.frames.index.push(torch.arange(len(poses), dtype=torch.long))
    global_map.frames.data["pose"].push(poses)
    global_map.frames.data["need_interp"].push(need_interp)
    if time_ns is not None:
        global_map.frames.data["time_ns"].push(time_ns)
    return global_map


def _aligned_reference_poses(global_map: VisualMap, ref_poses_path: Path) -> torch.Tensor | None:
    if not ref_poses_path.is_file():
        return None
    reference = np.load(ref_poses_path, allow_pickle=False)
    times = global_map.frames.data["time_ns"].tensor.detach().cpu().numpy()
    if reference.ndim != 2 or reference.shape[1] != 8 or len(times) != len(global_map.frames):
        return None
    by_time = {int(row[0]): row[1:] for row in reference}
    if any(int(time) not in by_time for time in times):
        return None
    return torch.from_numpy(np.stack([by_time[int(time)] for time in times])).float()


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
    parser.add_argument(
        "--phase-b5-mode", choices=("disabled", "observe", "apply"), default=None,
        help="Phase B.5 mode; apply additionally requires a promoted trusted manifest.",
    )
    parser.add_argument("--phase-b5-manifest", type=Path)
    parser.add_argument("--phase-b5-absolute-median-cap", type=float)
    parser.add_argument("--phase-b5-absolute-q95-cap", type=float)
    parser.add_argument(
        "--vins-feature-source", choices=("fixed_covariance", "orb_detected"),
        help="Override the VINS-style local geometry feature source in memory.",
    )
    parser.add_argument(
        "--vins-descriptor-match-mode", choices=("vins_legacy", "orbslam"),
        help="Override the VINS-style descriptor matcher in memory.",
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
    vins_mode = bool(getattr(getattr(loop_config, "vins_geometry", None), "enabled", False))
    if args.vins_feature_source is not None:
        if not hasattr(loop_config, "vins_geometry"):
            raise ValueError("--vins-feature-source requires a vins_geometry configuration")
        loop_config.vins_geometry.feature_source = args.vins_feature_source
    if args.vins_descriptor_match_mode is not None:
        if not hasattr(loop_config, "vins_geometry"):
            raise ValueError("--vins-descriptor-match-mode requires a vins_geometry configuration")
        loop_config.vins_geometry.descriptor_match_mode = args.vins_descriptor_match_mode
    if not vins_mode:
        loop_config.geometry.flow_cov_gate_enabled = args.primary_gate == "enabled"
        # Explicitly override the YAML so an omitted CLI option always means the fixed baseline.
        loop_config.geometry.flow_cov_adaptive_target_points = args.adaptive_target_points
        loop_config.geometric_verification.compare_flow_cov_gate = True
    if hasattr(loop_config, "phase_b5"):
        if args.phase_b5_mode is not None:
            loop_config.phase_b5.enabled = args.phase_b5_mode != "disabled"
            loop_config.phase_b5.mode = args.phase_b5_mode
        if args.phase_b5_manifest is not None:
            loop_config.phase_b5.trusted_manifest = str(args.phase_b5_manifest.resolve())
            trusted_payload = _load_json(args.phase_b5_manifest.resolve())
            caps = trusted_payload.get("absolute_sanity_caps") or {}
            loop_config.phase_b5.calibration.absolute_median_log_risk_cap = caps.get("median_log_risk")
            loop_config.phase_b5.calibration.absolute_q95_log_risk_cap = caps.get("q95_log_risk")
        if args.phase_b5_absolute_median_cap is not None:
            loop_config.phase_b5.calibration.absolute_median_log_risk_cap = args.phase_b5_absolute_median_cap
        if args.phase_b5_absolute_q95_cap is not None:
            loop_config.phase_b5.calibration.absolute_q95_log_risk_cap = args.phase_b5_absolute_q95_cap
    elif args.phase_b5_mode not in {None, "disabled"} or args.phase_b5_manifest is not None:
        raise ValueError("selected configuration has no phase_b5 section")
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
    network_refinement_enabled = bool(
        vins_mode
        and getattr(
            getattr(loop_config.vins_geometry, "network_refinement", None),
            "enabled",
            False,
        )
    )
    frontend = (
        None
        if vins_mode and not network_refinement_enabled
        else IFrontend.instantiate(runtime_frontend_type, frontend_config.args)
    )
    manager = LoopClosureManager(loop_config)
    manager.set_output_dir(output_dir)
    if not manager.enabled or manager.output_dir is None:
        raise RuntimeError(f"loop manager could not initialize: {manager.disabled_reason}")
    manager.records = records
    if frontend is not None:
        manager.set_frontend(frontend)
    manager.set_match_cov_default(float(config.Odometry.args.match_cov_default))

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
    pose_copy_pgo = None
    if vins_mode:
        global_pgo_config = getattr(config.Odometry, "global_pgo", None)
        pose_copy_pgo = (
            run_vins_pose_copy_pgo(
                global_map,
                global_pgo_config,
                output_dir / "loop_constraints_pgo_fixed.json",
                output_dir / "loop_constraints_pgo_covariance.json",
                _aligned_reference_poses(global_map, result_dir / "ref_poses.npy"),
            )
            if global_pgo_config is not None else {
                "executed": False, "safe": False, "reason": "global_pgo_config_missing",
            }
        )
        pgo_target = output_dir / "pose_copy_pgo_comparison.json"
        pgo_temporary = pgo_target.with_suffix(".json.tmp")
        with open(pgo_temporary, "w", encoding="utf-8") as file:
            json.dump(pose_copy_pgo, file, indent=2, allow_nan=False)
        pgo_temporary.replace(pgo_target)
    comparison_validation = (
        {
            "vins_geometry_output": (output_dir / "loop_vins_verification.json").is_file(),
            "pose_copy_pgo_output": (output_dir / "pose_copy_pgo_comparison.json").is_file(),
        }
        if vins_mode else validate_comparison_outputs(output_dir)
    )
    gt_pose_proxy = (
        {
            "vins_geometry": evaluate_gt_pose_proxy(
                output_dir / "loop_constraints_pgo_fixed.json",
                result_dir / "ref_poses.npy", record_dir, records,
            ),
            "geometry_accepted": evaluate_gt_pose_proxy(
                output_dir / "loop_constraints.json",
                result_dir / "ref_poses.npy", record_dir, records,
            ),
        }
        if vins_mode else {
            "gate_enabled": evaluate_gt_pose_proxy(
                output_dir / "loop_constraints_gate_enabled.json", result_dir / "ref_poses.npy", record_dir, records,
            ),
            "gate_disabled": evaluate_gt_pose_proxy(
                output_dir / "loop_constraints_gate_disabled.json", result_dir / "ref_poses.npy", record_dir, records,
            ),
        }
    )
    if hasattr(loop_config, "phase_b5") and loop_config.phase_b5.mode == "apply":
        gt_pose_proxy["phase_b5_apply"] = evaluate_gt_pose_proxy(
            output_dir / "loop_constraints.json",
            result_dir / "ref_poses.npy",
            record_dir,
            records,
        )

    engineering_admission = (
        summarize_engineering_admission(gt_pose_proxy["vins_geometry"], pose_copy_pgo)
        if vins_mode else None
    )

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
            "phase_b5": _sha256(
                Path(__file__).resolve().parents[2] / "Module/LoopClosure/PhaseB5.py"
            ),
            "vins_geometry": _sha256(
                Path(__file__).resolve().parents[2] / "Module/LoopClosure/VINSGeometry.py"
            ),
            "offline_runner": _sha256(Path(__file__).resolve()),
            "config": _sha256(config_path),
        },
        "device": args.device,
        "vins_geometry_enabled": vins_mode,
        "vins_geometry_feature_source": (
            getattr(loop_config.vins_geometry, "feature_source", "fixed_covariance")
            if hasattr(loop_config, "vins_geometry") else None
        ),
        "vins_geometry_descriptor_match_mode": (
            getattr(loop_config.vins_geometry, "descriptor_match_mode", "vins_legacy")
            if hasattr(loop_config, "vins_geometry") else None
        ),
        "vins_geometry_network_refinement_enabled": network_refinement_enabled,
        "vins_geometry_descriptor_match_parameters": (
            {
                "distance_threshold_inclusive": ORB_SLAM_HAMMING_THRESHOLD,
                "ratio_threshold_strict": ORB_SLAM_RATIO_THRESHOLD,
                "orientation_filter": "simplified_orbslam_orientation_histogram",
                "orientation_histogram_bins": ORB_SLAM_ORIENTATION_BINS,
                "orientation_weak_bin_ratio": ORB_SLAM_ORIENTATION_WEAK_BIN_RATIO,
            }
            if hasattr(loop_config, "vins_geometry")
            and getattr(loop_config.vins_geometry, "descriptor_match_mode", "vins_legacy") == "orbslam"
            else {
                "distance_threshold_strict": (
                    getattr(loop_config.vins_geometry, "hamming_threshold", None)
                    if hasattr(loop_config, "vins_geometry") else None
                )
            }
        ),
        "configured_frontend_type": configured_frontend_type,
        "runtime_frontend_type": runtime_frontend_type,
        "nvtx_disabled_for_cpu": nvtx_disabled_for_cpu,
        "primary_gate_enabled": loop_config.geometry.flow_cov_gate_enabled,
        "comparison_enabled": not vins_mode,
        "gate_enabled_selection_mode": (
            "fixed" if args.adaptive_target_points is None else "adaptive"
        ),
        "adaptive_target_points": args.adaptive_target_points,
        "phase_b5_mode": (
            getattr(loop_config.phase_b5, "mode", "disabled")
            if hasattr(loop_config, "phase_b5") else "disabled"
        ),
        "phase_b5_manifest": None if args.phase_b5_manifest is None else str(args.phase_b5_manifest),
        "phase_b5_manifest_sha256": (
            None if args.phase_b5_manifest is None else _sha256(args.phase_b5_manifest.resolve())
        ),
        "phase_b5_effective_calibration": (
            None if not hasattr(loop_config, "phase_b5") else {
                "prefix_fraction": loop_config.phase_b5.calibration.prefix_fraction,
                "absolute_median_log_risk_cap": (
                    loop_config.phase_b5.calibration.absolute_median_log_risk_cap
                ),
                "absolute_q95_log_risk_cap": (
                    loop_config.phase_b5.calibration.absolute_q95_log_risk_cap
                ),
            }
        ),
        "max_total_candidates": args.max_total_candidates,
        "selected_candidates": selected_candidates,
        "elapsed_seconds": elapsed_seconds,
        "primary_constraints": len(constraints),
        "pose_invariant": True,
        "comparison_validation": comparison_validation,
        "gt_pose_proxy": gt_pose_proxy,
        "engineering_admission": engineering_admission,
        "pose_copy_pgo": pose_copy_pgo,
    }
    temporary = output_dir / "offline_run_manifest.json.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.flush()
    temporary.replace(output_dir / "offline_run_manifest.json")


if __name__ == "__main__":
    main()
