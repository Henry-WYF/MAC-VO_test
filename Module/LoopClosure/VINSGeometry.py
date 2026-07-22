from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pypose as pp
import torch

from Module.Covariance.Project2to3 import Covariance_2to3_full
from Module.Optimization.GlobalPGO import make_information

from .PhaseB5 import reproj_disp_linearization, transform_information_for_inverse
from .Record import GeometryFeatureRecord, LoopFrameRecord
from .Verification import LoopConstraint


_NED_TO_CV = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
_BIT_COUNTS = np.asarray([int(index).bit_count() for index in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class FixedMatch:
    distance: int
    current_local: int
    candidate_local: int
    current_original: int
    candidate_original: int


@dataclass
class GeometryResult:
    row: dict[str, Any]
    constraint: LoopConstraint | None
    covariance_information: torch.Tensor | None


def stable_seed(current_sensor_idx: int, candidate_sensor_idx: int) -> int:
    return (int(current_sensor_idx) * 1_000_003 + int(candidate_sensor_idx)) & 0x7FFFFFFF


def fixed_point_covariance(
    pixel_uv: torch.Tensor,
    depth: torch.Tensor,
    depth_variance: torch.Tensor,
    intrinsic: torch.Tensor,
    pixel_variance: float,
) -> torch.Tensor:
    """Propagate fixed pixel quantization and same-frame depth uncertainty to NED 3D."""
    uv = pixel_uv.detach().cpu().double()
    d = depth.detach().cpu().double().reshape(-1)
    d_var = depth_variance.detach().cpu().double().reshape(-1)
    K = intrinsic.detach().cpu().double()
    if K.ndim == 3:
        K = K[0]
    variance = torch.full_like(d, float(pixel_variance))
    return Covariance_2to3_full(
        variance, torch.zeros_like(variance), variance, d_var,
        uv[:, 0], uv[:, 1], d,
        float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
    ).double()


def match_fixed_descriptors(
    current: GeometryFeatureRecord,
    candidate: GeometryFeatureRecord,
    hamming_threshold: int = 80,
) -> list[FixedMatch]:
    """VINS-style one-way nearest neighbor with deterministic train-index uniqueness."""
    current_desc = current.descriptor.detach().cpu().numpy().astype(np.uint8, copy=False)
    candidate_desc = candidate.descriptor.detach().cpu().numpy().astype(np.uint8, copy=False)
    if current_desc.ndim != 2 or candidate_desc.ndim != 2 or not len(current_desc) or not len(candidate_desc):
        return []
    proposed: list[FixedMatch] = []
    for current_local, descriptor in enumerate(current_desc):
        distances = _BIT_COUNTS[np.bitwise_xor(candidate_desc, descriptor)].sum(axis=1)
        distance = int(distances.min())
        tied = np.flatnonzero(distances == distance).tolist()
        candidate_local = min(tied, key=lambda index: int(candidate.original_index[index]))
        if distance < int(hamming_threshold):
            proposed.append(FixedMatch(
                distance=distance,
                current_local=current_local,
                candidate_local=int(candidate_local),
                current_original=int(current.original_index[current_local]),
                candidate_original=int(candidate.original_index[candidate_local]),
            ))
    proposed.sort(key=lambda item: (item.distance, item.current_original, item.candidate_original))
    used_candidate: set[int] = set()
    unique: list[FixedMatch] = []
    for item in proposed:
        if item.candidate_local in used_candidate:
            continue
        used_candidate.add(item.candidate_local)
        unique.append(item)
    return unique


def _pose_matrix(pose: pp.LieTensor | torch.Tensor) -> np.ndarray:
    value = pose if isinstance(pose, pp.LieTensor) else pp.SE3(pose)
    return value.matrix().detach().cpu().double().numpy()


def _ned_pose_to_cv(pose: pp.LieTensor | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    matrix = _pose_matrix(pose)
    rotation = _NED_TO_CV @ matrix[:3, :3] @ _NED_TO_CV.T
    translation = _NED_TO_CV @ matrix[:3, 3]
    rvec, _ = cv2.Rodrigues(rotation)
    return rvec.astype(np.float64), translation.reshape(3, 1).astype(np.float64)


def _cv_pose_to_ned(rvec: np.ndarray, tvec: np.ndarray) -> pp.LieTensor:
    rotation_cv, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    rotation = _NED_TO_CV.T @ rotation_cv @ _NED_TO_CV
    translation = _NED_TO_CV.T @ np.asarray(tvec, dtype=np.float64).reshape(3)
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, :3] = torch.from_numpy(rotation)
    matrix[:3, 3] = torch.from_numpy(translation)
    return pp.from_matrix(matrix, pp.SE3_type)


def _pose_magnitude(pose: pp.LieTensor) -> tuple[float, float]:
    matrix = pose.matrix().detach().cpu().double()
    translation = float(torch.linalg.vector_norm(matrix[:3, 3]))
    trace = float(torch.trace(matrix[:3, :3]))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) * 0.5))))
    return translation, angle


def _pose_difference(estimate: pp.LieTensor, reference: pp.LieTensor) -> tuple[float, float]:
    return _pose_magnitude(reference.Inv() @ estimate)


def _strict_information(
    pose_current_candidate: pp.LieTensor,
    candidate_points: torch.Tensor,
    candidate_covariance: torch.Tensor,
    current_uv: torch.Tensor,
    current_disparity: torch.Tensor,
    current_disparity_variance: torch.Tensor,
    K: torch.Tensor,
    baseline: torch.Tensor,
    pixel_variance: float,
    fixed_information: torch.Tensor,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    dtype = torch.float64
    point = candidate_points.to(dtype)
    point_cov = candidate_covariance.to(dtype)
    uv = current_uv.to(dtype)
    disparity = current_disparity.to(dtype)
    disparity_var = current_disparity_variance.to(dtype)
    uv_cov = torch.eye(2, dtype=dtype).unsqueeze(0).repeat(len(uv), 1, 1) * float(pixel_variance)
    residual, jacobian, covariance, transformed = reproj_disp_linearization(
        pose_current_candidate.to(dtype), point, point_cov, uv, uv_cov,
        disparity, disparity_var, K.to(dtype), baseline.to(dtype),
    )
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    diagnostics: dict[str, Any] = {
        "cross_view_covariance": "ignored_independent_approximation",
        "point_count": int(len(point)), "valid": False,
    }
    if not all(torch.isfinite(value).all() for value in (residual, jacobian, covariance, transformed)):
        diagnostics["reason"] = "nonfinite_linearization"
        return None, diagnostics
    cholesky, info = torch.linalg.cholesky_ex(covariance)
    if bool((info != 0).any()):
        diagnostics["reason"] = "residual_covariance_not_spd"
        return None, diagnostics
    whitened = torch.linalg.solve_triangular(cholesky, jacobian, upper=False).reshape(-1, 6)
    singular = torch.linalg.svdvals(whitened)
    threshold = max(float(singular.max()) * 1e-8, 1e-12) if singular.numel() else 1e-12
    rank = int((singular > threshold).sum())
    raw = whitened.T @ whitened
    raw = 0.5 * (raw + raw.T)
    eigen = torch.linalg.eigvalsh(raw)
    diagnostics.update({
        "stacked_whitened_jacobian_shape": list(whitened.shape),
        "rank": rank, "rank_threshold": threshold,
        "raw_eigenvalues": eigen.detach().cpu().tolist(),
    })
    if rank != 6 or not torch.isfinite(raw).all() or float(eigen.min()) < -1e-9:
        diagnostics["reason"] = "information_rank_or_psd"
        return None, diagnostics

    raw_edge = transform_information_for_inverse(raw, pose_current_candidate.to(dtype))
    raw_edge = 0.5 * (raw_edge + raw_edge.T)
    fixed = fixed_information.to(dtype)
    L, fixed_info = torch.linalg.cholesky_ex(fixed)
    if int(fixed_info) != 0:
        raise ValueError("Fixed loop information must be SPD")
    left = torch.linalg.solve_triangular(L, raw_edge, upper=False)
    normalized = torch.linalg.solve_triangular(L, left.T, upper=False).T
    normalized = 0.5 * (normalized + normalized.T)
    lambda_max = float(torch.linalg.eigvalsh(normalized).max())
    if not math.isfinite(lambda_max) or lambda_max <= 0.0:
        diagnostics["reason"] = "invalid_generalized_eigenvalue"
        return None, diagnostics
    alpha = min(1.0, 1.0 / lambda_max)
    used = raw_edge * alpha
    difference_eigen = torch.linalg.eigvalsh(0.5 * ((fixed - used) + (fixed - used).T))
    tolerance = max(float(torch.diagonal(fixed).abs().max()) * 1e-8, 1e-12)
    if float(difference_eigen.min()) < -tolerance:
        diagnostics["reason"] = "scaled_information_exceeds_fixed"
        return None, diagnostics
    diagnostics.update({
        "valid": True, "reason": None, "alpha": alpha,
        "generalized_lambda_max": lambda_max,
        "edge_eigenvalues": torch.linalg.eigvalsh(used).detach().cpu().tolist(),
        "fixed_minus_used_min_eigenvalue": float(difference_eigen.min()),
        "pose_direction": "T_candidate_current",
        "used_matrix": used.detach().cpu().tolist(),
    })
    return used, diagnostics


def verify_fixed_geometry(
    config: SimpleNamespace,
    query: dict[str, Any],
    candidate: dict[str, Any],
    current_frame: LoopFrameRecord,
    candidate_frame: LoopFrameRecord,
    current: GeometryFeatureRecord,
    historical: GeometryFeatureRecord,
    pose_snapshot: torch.Tensor,
    pixel_variance: float,
    fixed_information: torch.Tensor,
) -> GeometryResult:
    started = time.perf_counter()
    pair_id = f"{int(query['loop_frame_idx'])}:{int(candidate['loop_frame_idx'])}"
    row: dict[str, Any] = {
        "pair_id": pair_id, "status": "rejected", "geometry_accepted": False,
        "information_valid": False, "pgo_comparison_eligible": False,
        "current_sensor_frame_idx": int(current.sensor_frame_idx),
        "candidate_sensor_frame_idx": int(historical.sensor_frame_idx),
        "bow_score": float(candidate.get("score", 0.0)),
    }

    matches = match_fixed_descriptors(current, historical, int(config.hamming_threshold))
    row["unique_descriptor_matches"] = len(matches)
    if len(matches) < 4:
        row.update({"reject_code": "insufficient_descriptor_matches", "elapsed_ms": (time.perf_counter() - started) * 1000.0})
        return GeometryResult(row, None, None)
    current_local = torch.tensor([item.current_local for item in matches], dtype=torch.long)
    candidate_local = torch.tensor([item.candidate_local for item in matches], dtype=torch.long)
    points_ned = historical.point_camera[candidate_local].double()
    image_uv = current.pixel_uv[current_local].double()
    valid = torch.isfinite(points_ned).all(dim=1) & torch.isfinite(image_uv).all(dim=1) & (points_ned[:, 0] > 0.0)
    points_ned, image_uv = points_ned[valid], image_uv[valid]
    current_local, candidate_local = current_local[valid], candidate_local[valid]
    if len(points_ned) < 4:
        row.update({"reject_code": "insufficient_finite_positive_input", "elapsed_ms": (time.perf_counter() - started) * 1000.0})
        return GeometryResult(row, None, None)
    points_cv = points_ned[:, [1, 2, 0]].numpy().astype(np.float64)
    if not bool(np.all(points_cv[:, 2] > 0.0)):
        raise RuntimeError("NED x-positive to OpenCV z-positive conversion failed")
    K = current_frame.intrinsic.detach().cpu().double().numpy()
    if K.ndim == 3:
        K = K[0]
    T_w_current = pp.SE3(pose_snapshot[current.visual_map_idx].double())
    T_w_candidate = pp.SE3(pose_snapshot[historical.visual_map_idx].double())
    initial = T_w_current.Inv() @ T_w_candidate
    rvec, tvec = _ned_pose_to_cv(initial)
    seed = stable_seed(current.sensor_frame_idx, historical.sensor_frame_idx)
    rng_applied = hasattr(cv2, "setRNGSeed")
    if rng_applied:
        cv2.setRNGSeed(seed)
    try:
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_cv, image_uv.numpy().astype(np.float64), K, np.empty((0, 1)),
            rvec, tvec, True, int(config.iterations), float(config.reproj_error_px),
            float(config.confidence), flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as error:
        row.update({"reject_code": "pnp_exception", "reject_reason": str(error), "elapsed_ms": (time.perf_counter() - started) * 1000.0})
        return GeometryResult(row, None, None)
    row.update({
        "pnp_use_extrinsic_guess": True, "pnp_initial_pose_source": "readonly_vo_relative_pose",
        "pnp_rng_seed": seed, "pnp_rng_seed_applied": rng_applied,
    })
    if not retval or inliers is None or len(inliers) == 0:
        row.update({"reject_code": "pnp_failed", "elapsed_ms": (time.perf_counter() - started) * 1000.0})
        return GeometryResult(row, None, None)
    inlier = torch.from_numpy(np.asarray(inliers, dtype=np.int64).reshape(-1))
    if bool(((inlier < 0) | (inlier >= len(points_ned))).any()):
        row.update({"reject_code": "invalid_pnp_inliers", "elapsed_ms": (time.perf_counter() - started) * 1000.0})
        return GeometryResult(row, None, None)
    pose = _cv_pose_to_ned(rvec, tvec)
    transformed = pose.Act(points_ned[inlier])
    positive = torch.isfinite(transformed).all(dim=1) & (transformed[:, 0] > 0.0)
    inlier = inlier[positive]
    if len(inlier) < int(config.min_inliers):
        row.update({"reject_code": "insufficient_positive_inliers", "pnp_inliers": int(len(inlier)), "elapsed_ms": (time.perf_counter() - started) * 1000.0})
        return GeometryResult(row, None, None)
    projected, _ = cv2.projectPoints(points_cv[inlier.numpy()], rvec, tvec, K, np.empty((0, 1)))
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_uv[inlier].numpy(), axis=1)
    if not np.isfinite(errors).all() or not torch.isfinite(pose.tensor()).all():
        row.update({"reject_code": "nonfinite_pnp_output", "elapsed_ms": (time.perf_counter() - started) * 1000.0})
        return GeometryResult(row, None, None)
    translation, rotation = _pose_magnitude(pose)
    initial_delta_t, initial_delta_r = _pose_difference(pose, initial)
    vo_edge = T_w_candidate.Inv() @ T_w_current
    edge = pose.Inv()
    vo_delta_t, vo_delta_r = _pose_difference(edge, vo_edge)
    row.update({
        "pnp_inliers": int(len(inlier)), "mean_reprojection_error_px": float(errors.mean()),
        "median_reprojection_error_px": float(np.median(errors)), "max_reprojection_error_px": float(errors.max()),
        "relative_translation_m": translation, "relative_rotation_deg": rotation,
        "pnp_initial_to_final_translation_m": initial_delta_t,
        "pnp_initial_to_final_rotation_deg": initial_delta_r,
        "vo_difference_translation_m": vo_delta_t, "vo_difference_rotation_deg": vo_delta_r,
    })
    if translation >= float(config.max_translation_m) or rotation >= float(config.max_rotation_deg):
        row.update({"reject_code": "pose_safety_gate", "elapsed_ms": (time.perf_counter() - started) * 1000.0})
        return GeometryResult(row, None, None)

    selected_current = current_local[inlier]
    selected_candidate = candidate_local[inlier]
    disp_valid = current.disparity_valid[selected_current].bool()
    disp_valid &= torch.isfinite(current.disparity[selected_current])
    disp_valid &= torch.isfinite(current.disparity_variance[selected_current])
    disp_valid &= current.disparity_variance[selected_current] > 0.0
    info_matrix = None
    info_payload: dict[str, Any]
    if int(disp_valid.sum()) == 0:
        info_payload = {"valid": False, "reason": "no_valid_disparity_inliers", "point_count": 0}
    else:
        info_matrix, info_payload = _strict_information(
            pose,
            historical.point_camera[selected_candidate][disp_valid],
            historical.point_covariance_camera[selected_candidate][disp_valid],
            current.pixel_uv[selected_current][disp_valid],
            current.disparity[selected_current][disp_valid],
            current.disparity_variance[selected_current][disp_valid],
            current_frame.intrinsic[0] if current_frame.intrinsic.ndim == 3 else current_frame.intrinsic,
            current_frame.baseline,
            pixel_variance,
            fixed_information,
        )
    row.update({
        "status": "accepted", "reject_code": None, "geometry_accepted": True,
        "information_valid": info_matrix is not None,
        "pgo_comparison_eligible": info_matrix is not None,
        "information_observe": info_payload,
        "pnp_relative_pose": pose.tensor().detach().cpu().tolist(),
        "relative_pose": edge.tensor().detach().cpu().tolist(),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    })
    constraint = LoopConstraint(
        src_visual_map_idx=int(historical.visual_map_idx), dst_visual_map_idx=int(current.visual_map_idx),
        src_sensor_frame_idx=int(historical.sensor_frame_idx), dst_sensor_frame_idx=int(current.sensor_frame_idx),
        pnp_relative_pose=row["pnp_relative_pose"], relative_pose=row["relative_pose"],
        information=fixed_information.detach().cpu().tolist(), bow_score=float(candidate.get("score", 0.0)),
        num_flow_points=0, num_geometry_points=len(matches), num_pnp_inliers=int(len(inlier)),
        inlier_ratio=float(len(inlier) / max(len(points_ned), 1)),
        mean_reproj_error_px=float(errors.mean()), rotation_diff_deg=vo_delta_r,
        translation_diff_m=vo_delta_t, status="accepted",
    )
    return GeometryResult(row, constraint, info_matrix)


def fixed_loop_information(config: SimpleNamespace) -> torch.Tensor:
    return make_information(float(config.trans_weight), float(config.rot_weight), dtype=torch.float64)


def run_pose_copy_pgo_safety(
    optimizer: Any,
    initial_poses: torch.Tensor,
    reference_poses: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Run an existing GlobalPGO instance on a pose copy and reject unsafe output."""
    initial = initial_poses.detach().cpu().clone()
    initial_loss = optimizer.compute_loss(initial).detach().cpu()
    result: dict[str, Any] = {
        "initial_loss": float(initial_loss) if torch.isfinite(initial_loss) else None,
        "final_loss": None, "safe": False,
        "first_frame_aligned_se3_log_translation_rmse_m": None,
        "first_frame_aligned_relative_se3_log_translation_rmse_m": None,
        "first_frame_aligned_relative_se3_log_rotation_rmse_deg": None,
        "gt_metrics_reason": "ground_truth_not_provided",
    }
    if not torch.isfinite(initial_loss):
        result["reason"] = "nonfinite_initial_loss"
        return None, result
    optimized = optimizer.optimize_poses(initial).detach().cpu()
    final_loss = optimizer.compute_loss(optimized).detach().cpu()
    result["final_loss"] = float(final_loss) if torch.isfinite(final_loss) else None
    if not torch.isfinite(optimized).all() or not torch.isfinite(final_loss):
        result["reason"] = "nonfinite_optimized_result"
        return None, result
    if float(final_loss) > float(initial_loss) + 1e-9:
        result["reason"] = "loss_increased"
        return None, result
    if not torch.equal(optimized[0], initial[0]):
        result["reason"] = "first_pose_changed"
        return None, result

    initial_se3, optimized_se3 = pp.SE3(initial.double()), pp.SE3(optimized.double())
    corrections = (initial_se3.Inv() @ optimized_se3).Log().tensor()
    max_pose = float(torch.linalg.vector_norm(corrections, dim=-1).max()) if len(corrections) else 0.0
    max_translation = float(torch.linalg.vector_norm(corrections[:, :3], dim=-1).max()) if len(corrections) else 0.0
    max_rotation = math.degrees(float(torch.linalg.vector_norm(corrections[:, 3:], dim=-1).max())) if len(corrections) else 0.0
    adjacent_change = 0.0
    adjacent_translation = 0.0
    adjacent_rotation = 0.0
    if len(initial) > 1:
        initial_rel = initial_se3[:-1].Inv() @ initial_se3[1:]
        optimized_rel = optimized_se3[:-1].Inv() @ optimized_se3[1:]
        deformation = (initial_rel.Inv() @ optimized_rel).Log().tensor()
        adjacent_change = float(torch.linalg.vector_norm(deformation, dim=-1).max())
        adjacent_translation = float(torch.linalg.vector_norm(deformation[:, :3], dim=-1).max())
        adjacent_rotation = math.degrees(float(torch.linalg.vector_norm(deformation[:, 3:], dim=-1).max()))
    before_residual = optimizer.compute_residuals(initial).detach().cpu()
    after_residual = optimizer.compute_residuals(optimized).detach().cpu()
    loop_indices = [
        index for index, edge in enumerate(getattr(optimizer, "edges", []))
        if getattr(edge, "edge_type", None) == "loop"
    ]
    loop_before = before_residual[loop_indices] if loop_indices else before_residual[:0]
    loop_after = after_residual[loop_indices] if loop_indices else after_residual[:0]
    result.update({
        "safe": True, "reason": None, "max_single_pose_correction_se3_norm": max_pose,
        "max_single_pose_translation_correction_m": max_translation,
        "max_single_pose_rotation_correction_deg": max_rotation,
        "max_adjacent_deformation_se3_norm": adjacent_change,
        "max_adjacent_translation_deformation_m": adjacent_translation,
        "max_adjacent_rotation_deformation_deg": adjacent_rotation,
        "loop_residual_norm_before": float(torch.linalg.vector_norm(loop_before, dim=-1).max()) if len(loop_before) else 0.0,
        "loop_residual_norm_after": float(torch.linalg.vector_norm(loop_after, dim=-1).max()) if len(loop_after) else 0.0,
    })
    if reference_poses is not None and tuple(reference_poses.shape) == tuple(optimized.shape):
        reference = pp.SE3(reference_poses.detach().cpu().double())
        aligned = reference[0] @ optimized_se3[0].Inv() @ optimized_se3
        absolute_error = (reference.Inv() @ aligned).Log().tensor()
        result["first_frame_aligned_se3_log_translation_rmse_m"] = float(
            torch.sqrt(torch.mean(torch.linalg.vector_norm(absolute_error[:, :3], dim=-1).square()))
        )
        if len(reference) > 1:
            ref_relative = reference[:-1].Inv() @ reference[1:]
            opt_relative = aligned[:-1].Inv() @ aligned[1:]
            relative_error = (ref_relative.Inv() @ opt_relative).Log().tensor()
            result["first_frame_aligned_relative_se3_log_translation_rmse_m"] = float(
                torch.sqrt(torch.mean(torch.linalg.vector_norm(relative_error[:, :3], dim=-1).square()))
            )
            result["first_frame_aligned_relative_se3_log_rotation_rmse_deg"] = math.degrees(float(
                torch.sqrt(torch.mean(torch.linalg.vector_norm(relative_error[:, 3:], dim=-1).square()))
            ))
        result["gt_metrics_reason"] = None
    return optimized, result


def run_pose_copy_pgo_comparison(
    fixed_optimizer: Any,
    covariance_optimizer: Any,
    initial_poses: torch.Tensor,
    reference_poses: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run the two information variants only when their edge sets are identical."""
    def edge_keys(optimizer: Any) -> list[tuple[int, int, str]]:
        return [(int(edge.src), int(edge.dst), str(edge.edge_type)) for edge in optimizer.edges]

    fixed_keys = edge_keys(fixed_optimizer)
    covariance_keys = edge_keys(covariance_optimizer)
    if fixed_keys != covariance_keys:
        return {"safe": False, "reason": "pgo_edge_sets_differ", "fixed": None, "covariance": None}
    _, fixed = run_pose_copy_pgo_safety(fixed_optimizer, initial_poses, reference_poses)
    _, covariance = run_pose_copy_pgo_safety(covariance_optimizer, initial_poses, reference_poses)
    return {
        "safe": bool(fixed.get("safe")) and bool(covariance.get("safe")),
        "reason": None if bool(fixed.get("safe")) and bool(covariance.get("safe")) else "unsafe_pgo_result",
        "edge_count": len(fixed_keys),
        "fixed": fixed,
        "covariance": covariance,
    }
