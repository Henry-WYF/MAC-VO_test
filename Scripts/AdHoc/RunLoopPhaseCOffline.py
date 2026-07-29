from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pypose as pp
import torch
import yaml

from Evaluation.MetricsSeq import evaluateATE, evaluateRPE
from Module.LoopClosure.VINSGeometry import run_pose_copy_pgo_safety
from Module.Map import VisualMap
from Module.Optimization.GlobalPGO import GlobalPoseGraphOptimizer
from Utility.Config import load_config
from Utility.Sandbox import Sandbox
from Utility.Trajectory import Trajectory


POSE_ATOL = 1e-5
POSE_RTOL = 1e-6


def _load_json(path: Path) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=reject_nonfinite)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _edge_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["src_visual_map_idx"]), int(row["dst_visual_map_idx"])


def load_phase_c_edges(phase_b_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and associate selected, comparison-eligible Phase B edges."""
    verification = _load_json(phase_b_dir / "loop_vins_verification.json")
    fixed_payload = _load_json(phase_b_dir / "loop_constraints_pgo_fixed.json")
    covariance_payload = _load_json(phase_b_dir / "loop_constraints_pgo_covariance.json")
    index_payload = _load_json(phase_b_dir / "source_index.json")

    records = index_payload.get("records")
    fixed_rows = fixed_payload.get("constraints")
    covariance_rows = covariance_payload.get("constraints")
    verification_rows = verification.get("verifications")
    if not all(isinstance(value, list) for value in (records, fixed_rows, covariance_rows, verification_rows)):
        raise ValueError("Phase B edge inputs do not contain list payloads")

    by_sensor: dict[int, dict[str, Any]] = {}
    for record in records:
        sensor_idx = int(record["sensor_frame_idx"])
        if sensor_idx in by_sensor:
            raise ValueError(f"duplicate sensor frame {sensor_idx} in source_index.json")
        by_sensor[sensor_idx] = record

    selected = [
        row for row in verification_rows
        if row.get("selected_for_query") is True
        and row.get("pgo_comparison_eligible") is True
    ]
    verification_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in selected:
        current_sensor = int(row["current_sensor_frame_idx"])
        candidate_sensor = int(row["candidate_sensor_frame_idx"])
        if current_sensor not in by_sensor or candidate_sensor not in by_sensor:
            raise ValueError("selected verification edge is absent from source_index.json")
        current = by_sensor[current_sensor]
        candidate = by_sensor[candidate_sensor]
        key = (int(candidate["visual_map_idx"]), int(current["visual_map_idx"]))
        for field, expected in (
            ("candidate_visual_map_idx", key[0]),
            ("current_visual_map_idx", key[1]),
        ):
            if row.get(field) is not None and int(row[field]) != expected:
                raise ValueError(f"verification {field} disagrees with source index for edge {key}")
        if key in verification_by_key:
            raise ValueError(f"duplicate selected verification edge {key}")
        expected_pair = f"{int(current['loop_frame_idx'])}:{int(candidate['loop_frame_idx'])}"
        if str(row.get("pair_id")) != expected_pair:
            raise ValueError(f"verification pair_id does not match source index for edge {key}")
        verification_by_key[key] = row

    def constraint_map(rows: list[dict[str, Any]], name: str) -> dict[tuple[int, int], dict[str, Any]]:
        result: dict[tuple[int, int], dict[str, Any]] = {}
        for row in rows:
            key = _edge_key(row)
            if key in result:
                raise ValueError(f"duplicate {name} edge {key}")
            src_sensor = int(row["src_sensor_frame_idx"])
            dst_sensor = int(row["dst_sensor_frame_idx"])
            if src_sensor not in by_sensor or dst_sensor not in by_sensor:
                raise ValueError(f"{name} edge sensor index is absent from source index")
            expected = (
                int(by_sensor[src_sensor]["visual_map_idx"]),
                int(by_sensor[dst_sensor]["visual_map_idx"]),
            )
            if key != expected:
                raise ValueError(f"{name} edge sensor/visual indices disagree for {key}")
            result[key] = row
        return result

    fixed_by_key = constraint_map(fixed_rows, "fixed")
    covariance_by_key = constraint_map(covariance_rows, "covariance")
    keys = set(verification_by_key)
    if not keys:
        if fixed_by_key or covariance_by_key:
            raise ValueError("Phase C zero selected verification edges disagree with fixed/covariance edges")
        return [], [], []
    if keys != set(fixed_by_key) or keys != set(covariance_by_key):
        raise ValueError("Phase C requires identical verification/fixed/covariance edge sets")

    ordered_keys = sorted(keys)
    for key in ordered_keys:
        fixed = fixed_by_key[key]
        covariance = covariance_by_key[key]
        if fixed.get("relative_pose") != covariance.get("relative_pose"):
            raise ValueError(f"fixed/covariance relative poses differ for edge {key}")
        verification_row = verification_by_key[key]
        information = verification_row.get("information_observe")
        if not isinstance(information, dict) or information.get("valid") is not True:
            raise ValueError(f"selected verification edge {key} has no valid information diagnostics")
        if information.get("used_matrix") != covariance.get("information"):
            raise ValueError(f"verification/covariance information differs for edge {key}")

    return (
        [verification_by_key[key] for key in ordered_keys],
        [fixed_by_key[key] for key in ordered_keys],
        [covariance_by_key[key] for key in ordered_keys],
    )


def load_phase_c_map(map_path: Path) -> tuple[VisualMap, torch.Tensor, torch.Tensor]:
    with np.load(map_path, allow_pickle=False) as archive:
        def array(*names: str) -> np.ndarray:
            key = next((name for name in names if name in archive.files), None)
            if key is None:
                raise ValueError(f"tensor map is missing one of {names}")
            return archive[key].copy()

        poses = torch.from_numpy(array("frames//pose", "frames/pose")).float()
        time_ns = torch.from_numpy(array("frames//time_ns", "frames/time_ns")).reshape(-1).long()
        body_to_sensor = torch.from_numpy(array("frames//T_BS", "frames/T_BS")).float()
        interp_key = next(
            (key for key in ("frames//need_interp", "frames/need_interp") if key in archive.files),
            None,
        )
        need_interp = (
            torch.from_numpy(archive[interp_key].copy()).bool()
            if interp_key is not None else torch.zeros(len(poses), dtype=torch.bool)
        )
    if not (len(poses) == len(time_ns) == len(body_to_sensor)):
        raise ValueError("tensor map pose/time/T_BS lengths differ")
    global_map = VisualMap()
    global_map.frames.index.push(torch.arange(len(poses), dtype=torch.long))
    global_map.frames.data["pose"].push(poses)
    global_map.frames.data["time_ns"].push(time_ns)
    global_map.frames.data["T_BS"].push(body_to_sensor)
    global_map.frames.data["need_interp"].push(need_interp)
    return global_map, time_ns, body_to_sensor


def sensor_to_body_timed(
    sensor_poses: torch.Tensor, body_to_sensor: torch.Tensor, timestamps: np.ndarray,
) -> np.ndarray:
    # Match the formal VO export path exactly: the sensor-to-body conjugation
    # is intentionally evaluated from the original float32 tensors.
    sensor = pp.SE3(sensor_poses.detach().cpu())
    extrinsic = pp.SE3(body_to_sensor.detach().cpu())
    body = (extrinsic @ sensor @ extrinsic.Inv()).tensor().numpy()
    return np.concatenate([np.asarray(timestamps, dtype=np.float64).reshape(-1, 1), body], axis=1)


def validate_source_trajectory(
    source_poses: np.ndarray, initial_sensor: torch.Tensor,
    body_to_sensor: torch.Tensor, time_ns: torch.Tensor,
) -> None:
    if source_poses.ndim != 2 or source_poses.shape[1] != 8 or len(source_poses) != len(initial_sensor):
        raise ValueError("source poses.npy has an invalid shape")
    converted_time = time_ns.detach().cpu().numpy().astype(source_poses[:, 0].dtype)
    if not np.array_equal(source_poses[:, 0], converted_time):
        raise ValueError("source pose timestamps do not match tensor_map timestamps")
    reconstructed = sensor_to_body_timed(
        initial_sensor, body_to_sensor, source_poses[:, 0],
    )[:, 1:]
    source_matrix = pp.SE3(torch.from_numpy(source_poses[:, 1:]).double()).matrix().numpy()
    reconstructed_matrix = pp.SE3(torch.from_numpy(reconstructed).double()).matrix().numpy()
    if not np.allclose(source_matrix, reconstructed_matrix, atol=POSE_ATOL, rtol=POSE_RTOL):
        raise ValueError("sensor-to-body reconstruction does not match source poses.npy")


def build_pgo_pair(
    global_map: VisualMap,
    config: Any,
    fixed_rows: list[dict[str, Any]],
    covariance_rows: list[dict[str, Any]],
) -> tuple[GlobalPoseGraphOptimizer, GlobalPoseGraphOptimizer]:
    fixed_optimizer = GlobalPoseGraphOptimizer(config)
    covariance_optimizer = GlobalPoseGraphOptimizer(config)
    fixed_optimizer.register_odometry_edges(global_map)
    covariance_optimizer.register_odometry_edges(global_map)
    for fixed, covariance in zip(fixed_rows, covariance_rows):
        if _edge_key(fixed) != _edge_key(covariance) or fixed["relative_pose"] != covariance["relative_pose"]:
            raise ValueError("fixed/covariance loop edge identity differs")
        for optimizer, row in ((fixed_optimizer, fixed), (covariance_optimizer, covariance)):
            optimizer.add_loop_edge(
                int(row["src_visual_map_idx"]), int(row["dst_visual_map_idx"]),
                torch.tensor(row["relative_pose"]), torch.tensor(row["information"]),
            )
    return fixed_optimizer, covariance_optimizer


def loop_residual_statistics(optimizer: Any, poses: torch.Tensor) -> dict[str, Any]:
    residuals = optimizer.compute_residuals(poses).detach().cpu()
    indices = [
        index for index, edge in enumerate(optimizer.edges)
        if getattr(edge, "edge_type", None) == "loop"
    ]
    if not indices:
        return {"count": 0, "mean": None, "rmse": None, "max": None}
    norms = torch.linalg.vector_norm(residuals[indices], dim=-1).double()
    return {
        "count": len(indices),
        "mean": float(norms.mean()),
        "rmse": float(torch.sqrt(torch.mean(norms.square()))),
        "max": float(norms.max()),
        "units": "mixed_translation_m_rotation_rad_se3_log_norm",
    }


def _atomic_save_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_evaluation_branch(
    target: Path, branch_id: str, poses: np.ndarray | None,
    result_dir: Path, metadata: dict[str, Any],
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if poses is None:
        shutil.copy2(result_dir / "poses.npy", target / "poses.npy")
    else:
        _atomic_save_numpy(target / "poses.npy", poses)
    for filename in ("ref_poses.npy", "config.yaml"):
        shutil.copy2(result_dir / filename, target / filename)
    frame_status = result_dir / "frame_status.pth"
    if frame_status.is_file():
        shutil.copy2(frame_status, target / frame_status.name)
    write_branch_metadata(target, branch_id, metadata)


def write_branch_metadata(
    target: Path, branch_id: str, metadata: dict[str, Any],
) -> None:
    payload = {"phase_c_branch": branch_id, **metadata}
    temporary = target / "metadata.yaml.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=True)
    os.replace(temporary, target / "metadata.yaml")


def formal_metrics(branch_dir: Path) -> dict[str, Any]:
    parameters = {
        "trajectory_preprocess_align_origin": True,
        "metric_align": True,
        "metric_align_origin": False,
        "correct_scale": False,
        "ate_pose_relation": "translation_part",
        "rpe_pose_relation": "full_transformation",
        "rpe_delta_frames": 1,
    }
    try:
        ground_truth, estimate = Trajectory.from_sandbox(
            Sandbox.load(branch_dir), align_time="est->gt",
        )
        ate = evaluateATE(ground_truth.data.as_evo, estimate.data.as_evo, correct_scale=False)
        rpe = evaluateRPE(ground_truth.data.as_evo, estimate.data.as_evo, correct_scale=False)
        keys = ("mean", "std", "rmse")
        return {
            "parameters": parameters,
            "ate": {key: float(ate.stats[key]) for key in keys},
            "rpe": {key: float(rpe.stats[key]) for key in keys},
            "error": None,
        }
    except Exception as error:
        return {
            "parameters": parameters,
            "ate": None,
            "rpe": None,
            "error": f"{type(error).__name__}: {error}",
        }


def metric_tolerance(value: float) -> float:
    return max(1e-9, 1e-6 * abs(float(value)))


def classify_branch(
    baseline_metrics: dict[str, Any], branch_metrics: dict[str, Any],
    baseline_residual: dict[str, Any], branch_residual: dict[str, Any],
    safe: bool,
) -> dict[str, Any]:
    def metric(payload: dict[str, Any], family: str) -> Any:
        values = payload.get(family)
        return values.get("rmse") if isinstance(values, dict) else None

    values = {
        "baseline_ate_rmse": metric(baseline_metrics, "ate"),
        "branch_ate_rmse": metric(branch_metrics, "ate"),
        "baseline_rpe_rmse": metric(baseline_metrics, "rpe"),
        "branch_rpe_rmse": metric(branch_metrics, "rpe"),
        "baseline_loop_residual_rmse": baseline_residual.get("rmse"),
        "branch_loop_residual_rmse": branch_residual.get("rmse"),
    }
    finite = all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in values.values()
    )
    if not safe or not finite:
        return {
            "classification": "failed", "online_experiment_eligible": False,
            "reason": "unsafe_or_missing_nonfinite_metric", "checks": values,
        }
    base_ate, branch_ate = float(values["baseline_ate_rmse"]), float(values["branch_ate_rmse"])
    base_rpe, branch_rpe = float(values["baseline_rpe_rmse"]), float(values["branch_rpe_rmse"])
    base_res, branch_res = float(values["baseline_loop_residual_rmse"]), float(values["branch_loop_residual_rmse"])
    ate_worse = branch_ate > base_ate + metric_tolerance(base_ate)
    rpe_worse = branch_rpe > base_rpe + metric_tolerance(base_rpe)
    residual_decreased = branch_res < base_res - metric_tolerance(base_res)
    checks = {
        **values,
        "ate_not_worse": not ate_worse,
        "rpe_not_worse": not rpe_worse,
        "loop_residual_significantly_decreased": residual_decreased,
    }
    if not ate_worse and not rpe_worse and residual_decreased:
        return {
            "classification": "positive", "online_experiment_eligible": True,
            "reason": None, "checks": checks,
        }
    if ate_worse and rpe_worse:
        return {
            "classification": "failed", "online_experiment_eligible": False,
            "reason": "ate_and_rpe_worsened", "checks": checks,
        }
    return {
        "classification": "inconclusive", "online_experiment_eligible": False,
        "reason": "mixed_or_insufficient_improvement", "checks": checks,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare no-loop, fixed, and covariance Phase C trajectories.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--phase-b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result_dir = args.result_dir.resolve()
    phase_b_dir = args.phase_b_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    required = (
        result_dir / "poses.npy", result_dir / "ref_poses.npy",
        result_dir / "config.yaml", result_dir / "tensor_map.npz",
        phase_b_dir / "loop_vins_verification.json",
        phase_b_dir / "loop_constraints_pgo_fixed.json",
        phase_b_dir / "loop_constraints_pgo_covariance.json",
        phase_b_dir / "source_index.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verification_rows, fixed_rows, covariance_rows = load_phase_c_edges(phase_b_dir)
    if not fixed_rows:
        comparison = {
            "schema_version": 1,
            "source_result_dir": str(result_dir),
            "source_phase_b_dir": str(phase_b_dir),
            "code_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "loop_edge_count": 0,
            "executed": False,
            "reason": "no_pgo_comparison_eligible_edges",
        }
        _write_json(output_dir / "phase_c_comparison.json", comparison)
        with (output_dir / "phase_c_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("branch", "classification"))
            writer.writeheader()
        print("Phase C skipped: no PGO comparison eligible loop edges")
        return
    global_map, time_ns, body_to_sensor = load_phase_c_map(result_dir / "tensor_map.npz")
    initial = global_map.frames.data["pose"].tensor.detach().clone()
    source_poses = np.load(result_dir / "poses.npy", allow_pickle=False)
    validate_source_trajectory(source_poses, initial, body_to_sensor, time_ns)

    config, _ = load_config(result_dir / "config.yaml")
    pgo_config = config.Odometry.global_pgo
    fixed_optimizer, covariance_optimizer = build_pgo_pair(
        global_map, pgo_config, fixed_rows, covariance_rows,
    )
    no_loop_residual = loop_residual_statistics(fixed_optimizer, initial)
    fixed_poses, fixed_safety = run_pose_copy_pgo_safety(fixed_optimizer, initial)
    covariance_poses, covariance_safety = run_pose_copy_pgo_safety(covariance_optimizer, initial)
    original_pose_invariant = torch.equal(global_map.frames.data["pose"].tensor, initial)
    if not original_pose_invariant:
        raise RuntimeError("Phase C modified the source VisualMap poses")

    unavailable_residual = {
        "count": len(fixed_rows), "mean": None, "rmse": None, "max": None,
        "reason": "unsafe_pgo_result",
    }
    fixed_residual = (
        loop_residual_statistics(fixed_optimizer, fixed_poses)
        if fixed_poses is not None else unavailable_residual
    )
    covariance_residual = (
        loop_residual_statistics(covariance_optimizer, covariance_poses)
        if covariance_poses is not None else unavailable_residual
    )
    common_metadata = {
        "source_result_dir": str(result_dir),
        "source_phase_b_dir": str(phase_b_dir),
        "code_commit": _git_commit(Path(__file__).resolve().parents[2]),
    }
    branches = {
        "no_loop": None,
        "fixed_information": (
            sensor_to_body_timed(fixed_poses, body_to_sensor, source_poses[:, 0])
            if fixed_poses is not None else None
        ),
        "covariance_information": (
            sensor_to_body_timed(covariance_poses, body_to_sensor, source_poses[:, 0])
            if covariance_poses is not None else None
        ),
    }
    branch_execution = {
        "no_loop": {"trajectory_source": "original", "pgo_safe": None},
        "fixed_information": {
            "trajectory_source": "optimized" if fixed_poses is not None else "original_fallback",
            "pgo_safe": bool(fixed_safety.get("safe")),
        },
        "covariance_information": {
            "trajectory_source": "optimized" if covariance_poses is not None else "original_fallback",
            "pgo_safe": bool(covariance_safety.get("safe")),
        },
    }
    branch_dirs: dict[str, Path] = {}
    for branch_id, poses in branches.items():
        branch_dir = output_dir / branch_id
        write_evaluation_branch(branch_dir, branch_id, poses, result_dir, {
            **common_metadata,
            **branch_execution[branch_id],
            "online_experiment_eligible": False,
        })
        branch_dirs[branch_id] = branch_dir

    metrics = {branch: formal_metrics(path) for branch, path in branch_dirs.items()}
    fixed_classification = classify_branch(
        metrics["no_loop"], metrics["fixed_information"],
        no_loop_residual, fixed_residual, bool(fixed_safety.get("safe")),
    )
    covariance_classification = classify_branch(
        metrics["no_loop"], metrics["covariance_information"],
        no_loop_residual, covariance_residual, bool(covariance_safety.get("safe")),
    )
    branch_classifications = {
        "no_loop": None,
        "fixed_information": fixed_classification,
        "covariance_information": covariance_classification,
    }
    for branch_id, branch_dir in branch_dirs.items():
        classification = branch_classifications[branch_id]
        write_branch_metadata(branch_dir, branch_id, {
            **common_metadata,
            **branch_execution[branch_id],
            "online_experiment_eligible": bool(
                classification and classification.get("online_experiment_eligible")
            ),
        })
    edge_diagnostics = []
    for verification_row, fixed_row in zip(verification_rows, fixed_rows):
        information = verification_row["information_observe"]
        edge_diagnostics.append({
            "edge_key": list(_edge_key(fixed_row)),
            "pair_id": verification_row["pair_id"],
            "src_sensor_frame_idx": fixed_row["src_sensor_frame_idx"],
            "dst_sensor_frame_idx": fixed_row["dst_sensor_frame_idx"],
            "rank": information["rank"],
            "raw_eigenvalues": information["raw_eigenvalues"],
            "used_eigenvalues": information["edge_eigenvalues"],
            "generalized_lambda_max": information["generalized_lambda_max"],
            "alpha": information["alpha"],
        })
    comparison = {
        "schema_version": 1,
        **common_metadata,
        "inputs": {
            "tensor_map_sha256": _sha256(result_dir / "tensor_map.npz"),
            "source_poses_sha256": _sha256(result_dir / "poses.npy"),
            "reference_poses_sha256": _sha256(result_dir / "ref_poses.npy"),
            "verification_sha256": _sha256(phase_b_dir / "loop_vins_verification.json"),
            "fixed_constraints_sha256": _sha256(phase_b_dir / "loop_constraints_pgo_fixed.json"),
            "covariance_constraints_sha256": _sha256(phase_b_dir / "loop_constraints_pgo_covariance.json"),
        },
        "policy": {
            "no_loop_registers_loop_edges": False,
            "huber_enabled": False,
            "pose_matrix_atol": POSE_ATOL,
            "pose_matrix_rtol": POSE_RTOL,
            "metric_tolerance": "max(1e-9, 1e-6 * abs(baseline))",
            "loop_residual_interpretation": "unweighted SE3 Log norm; mixed metres and radians",
        },
        "loop_edge_count": len(fixed_rows),
        "edge_diagnostics": edge_diagnostics,
        "original_pose_invariant": original_pose_invariant,
        "branches": {
            "no_loop": {
                "metrics": metrics["no_loop"],
                "loop_residual": {"before": no_loop_residual, "after": no_loop_residual},
                "max_single_pose_translation_correction_m": 0.0,
                "max_single_pose_rotation_correction_deg": 0.0,
                "max_adjacent_translation_deformation_m": 0.0,
                "max_adjacent_rotation_deformation_deg": 0.0,
                "classification": "baseline",
            },
            "fixed_information": {
                "metrics": metrics["fixed_information"],
                "loop_residual": {"before": no_loop_residual, "after": fixed_residual},
                "trajectory_is_optimized_output": fixed_poses is not None,
                "pgo": fixed_safety,
                "classification": fixed_classification,
            },
            "covariance_information": {
                "metrics": metrics["covariance_information"],
                "loop_residual": {"before": no_loop_residual, "after": covariance_residual},
                "trajectory_is_optimized_output": covariance_poses is not None,
                "pgo": covariance_safety,
                "classification": covariance_classification,
            },
        },
    }
    _write_json(output_dir / "phase_c_comparison.json", comparison)
    with (output_dir / "phase_c_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "branch", "classification", "ate_mean", "ate_std", "ate_rmse",
            "rpe_mean", "rpe_std", "rpe_rmse", "loop_residual_rmse",
            "max_pose_translation_correction_m", "max_pose_rotation_correction_deg",
            "max_adjacent_translation_deformation_m", "max_adjacent_rotation_deformation_deg",
        ))
        writer.writeheader()
        for branch in ("no_loop", "fixed_information", "covariance_information"):
            branch_payload = comparison["branches"][branch]
            pgo = branch_payload.get("pgo", branch_payload)
            classification = branch_payload["classification"]
            writer.writerow({
                "branch": branch,
                "classification": classification if isinstance(classification, str) else classification["classification"],
                "ate_mean": metrics[branch]["ate"]["mean"] if metrics[branch]["ate"] else None,
                "ate_std": metrics[branch]["ate"]["std"] if metrics[branch]["ate"] else None,
                "ate_rmse": metrics[branch]["ate"]["rmse"] if metrics[branch]["ate"] else None,
                "rpe_mean": metrics[branch]["rpe"]["mean"] if metrics[branch]["rpe"] else None,
                "rpe_std": metrics[branch]["rpe"]["std"] if metrics[branch]["rpe"] else None,
                "rpe_rmse": metrics[branch]["rpe"]["rmse"] if metrics[branch]["rpe"] else None,
                "loop_residual_rmse": branch_payload["loop_residual"]["after"]["rmse"],
                "max_pose_translation_correction_m": pgo.get("max_single_pose_translation_correction_m", 0.0),
                "max_pose_rotation_correction_deg": pgo.get("max_single_pose_rotation_correction_deg", 0.0),
                "max_adjacent_translation_deformation_m": pgo.get("max_adjacent_translation_deformation_m", 0.0),
                "max_adjacent_rotation_deformation_deg": pgo.get("max_adjacent_rotation_deformation_deg", 0.0),
            })
    print(f"Phase C comparison written to {output_dir}")
    print(f"fixed_information: {fixed_classification['classification']}")
    print(f"covariance_information: {covariance_classification['classification']}")


if __name__ == "__main__":
    main()
