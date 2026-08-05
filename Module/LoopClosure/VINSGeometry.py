from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pypose as pp
import torch

from Module.Covariance.Project2to3 import (
    Covariance_2to3_full,
    match_covariance_local_statistics,
)
from Module.Frontend.Frontend import IFrontend
from Module.Optimization.GlobalPGO import make_information
from Module.Optimization.ObservationInformation import (
    covariance_triplet_to_matrix,
    icp_linearization,
    information_diagnostics,
    robust_icp_observation_system,
    robust_observation_system,
)
from Utility.Point import pixel2point_NED

from .PhaseB5 import reproj_disp_linearization, transform_information_for_inverse
from .Record import GeometryFeatureRecord, LoopFrameRecord
from .Verification import LoopConstraint


_NED_TO_CV = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
_BIT_COUNTS = np.asarray([int(index).bit_count() for index in range(256)], dtype=np.uint8)
ORB_SLAM_HAMMING_THRESHOLD = 50
ORB_SLAM_RATIO_THRESHOLD = 0.9
ORB_SLAM_ORIENTATION_BINS = 30
ORB_SLAM_ORIENTATION_WEAK_BIN_RATIO = 0.1


def network_refinement_contract(config: SimpleNamespace | None) -> dict[str, Any]:
    residual_mode = "disp" if config is None else str(getattr(config, "residual_mode", "disp"))
    covariance = {
        "type": (
            "bilateral_match_covariance_3d_independent_sum"
            if residual_mode == "icp"
            else "reprojection_disparity_candidate_3d_plus_measurement"
        ),
        "source_covariance_type": "MatchCovariance" if residual_mode == "icp" else None,
        "kernel_size": None if residual_mode == "disp" else int(getattr(config, "kernel_size", 31)),
        "match_cov_default": None if residual_mode == "disp" else float(getattr(config, "match_cov_default", 0.25)),
        "min_depth_cov": None if residual_mode == "disp" else float(getattr(config, "min_depth_cov", 0.05)),
        "min_flow_cov": None if residual_mode == "disp" else float(getattr(config, "min_flow_cov", 0.25)),
        "candidate_position_covariance": None if residual_mode == "disp" else "fixed_0.25I",
        "cross_view_covariance": "ignored_independent_approximation",
    }
    canonical = json.dumps(covariance, sort_keys=True, separators=(",", ":"))
    return {
        "loop_residual_mode": residual_mode,
        "observation_covariance_model": covariance["type"],
        "kernel_size": covariance["kernel_size"],
        "covariance_config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "covariance_config": covariance,
    }


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
    pnp_pose_current_candidate: pp.LieTensor | None = None
    pnp_current_local: torch.Tensor | None = None
    pnp_candidate_local: torch.Tensor | None = None


def cached_orb_geometry(
    frame: LoopFrameRecord,
    pixel_variance: float,
    *,
    require_orientation: bool = False,
) -> tuple[GeometryFeatureRecord | None, str | None, dict[str, Any]]:
    """Adapt cached detected ORB points to the existing geometry verifier in memory."""
    keypoints = frame.orb_keypoints
    descriptors = frame.orb_descriptors
    diagnostics: dict[str, Any] = {
        "orb_descriptors": int(len(descriptors)) if descriptors.ndim > 0 else 0,
        "orb_pixel_covariance_mode": "fixed_match_cov_default",
        "disparity_variance_source": "derived_from_cached_depth_covariance",
    }
    valid_layout = (
        keypoints.ndim == 2 and keypoints.shape[1] >= (4 if require_orientation else 2)
        and keypoints.dtype.is_floating_point
        and descriptors.ndim == 2 and descriptors.shape[1] == 32
        and descriptors.dtype == torch.uint8
        and len(keypoints) == len(descriptors)
    )
    if not valid_layout or (len(keypoints) and not bool(torch.isfinite(keypoints[:, :2]).all())):
        return None, "invalid_orb_cache_layout", diagnostics
    if len(descriptors) == 0:
        return None, "empty_orb_descriptors", diagnostics
    depth_layout_valid = (
        frame.depth.ndim == 4 and frame.depth.shape[:2] == (1, 1)
        and frame.depth.shape[-2:] == (int(frame.height), int(frame.width))
    )
    covariance_layout_valid = (
        frame.depth_covariance is not None
        and frame.depth_covariance.ndim == 4
        and frame.depth_covariance.shape[:2] == (1, 1)
        and frame.depth_covariance.shape[-2:] == (int(frame.height), int(frame.width))
    )

    uv = keypoints[:, :2].detach().cpu().float()
    floor_uv = torch.floor(uv).long()
    inbound = (
        (floor_uv[:, 0] >= 0) & (floor_uv[:, 0] < int(frame.width))
        & (floor_uv[:, 1] >= 0) & (floor_uv[:, 1] < int(frame.height))
    )
    depth = torch.full((len(uv),), torch.nan, dtype=torch.float32)
    if depth_layout_valid:
        depth_map = frame.depth[0, 0].detach().cpu().float()
        depth[inbound] = depth_map[floor_uv[inbound, 1], floor_uv[inbound, 0]]
    depth_valid = inbound & torch.isfinite(depth) & (depth > 0.0)

    points = torch.full((len(uv), 3), torch.nan, dtype=torch.float32)
    K = frame.intrinsic[0] if frame.intrinsic.ndim == 3 else frame.intrinsic
    if bool(depth_valid.any()):
        points[depth_valid] = pixel2point_NED(uv[depth_valid], depth[depth_valid], K.float()).cpu()

    depth_variance = torch.full_like(depth, torch.nan)
    if covariance_layout_valid:
        assert frame.depth_covariance is not None
        covariance_map = frame.depth_covariance[0, 0].detach().cpu().float()
        depth_variance[inbound] = covariance_map[floor_uv[inbound, 1], floor_uv[inbound, 0]]
    covariance_valid = depth_valid & torch.isfinite(depth_variance) & (depth_variance > 0.0)
    point_covariance = torch.full((len(uv), 3, 3), torch.nan, dtype=torch.float64)
    if bool(covariance_valid.any()):
        point_covariance[covariance_valid] = fixed_point_covariance(
            uv[covariance_valid], depth[covariance_valid], depth_variance[covariance_valid],
            K, pixel_variance,
        )

    fx_baseline = float(K[0, 0]) * float(frame.baseline.reshape(-1)[0])
    disparity = torch.full_like(depth, torch.nan)
    disparity[depth_valid] = fx_baseline / depth[depth_valid]
    disparity_variance = torch.full_like(depth, torch.nan)
    disparity_variance[covariance_valid] = (
        fx_baseline / depth[covariance_valid].square()
    ).square() * depth_variance[covariance_valid]
    disparity_valid = (
        covariance_valid & torch.isfinite(disparity) & torch.isfinite(disparity_variance)
        & (disparity > 0.0) & (disparity_variance > 0.0)
    )
    diagnostics.update({
        "depth_layout_valid": depth_layout_valid,
        "depth_covariance_layout_valid": covariance_layout_valid,
        "inbound_orb_points": int(inbound.sum()),
        "valid_depth_orb_points": int(depth_valid.sum()),
        "valid_covariance_orb_points": int(covariance_valid.sum()),
    })
    record = GeometryFeatureRecord(
        sensor_frame_idx=int(frame.sensor_frame_idx), visual_map_idx=int(frame.visual_map_idx),
        loop_frame_idx=int(frame.loop_frame_idx), orb_config_sha256="cached_orb_detected",
        original_index=torch.arange(len(uv), dtype=torch.long), pixel_uv=uv,
        point_camera=points, point_covariance_camera=point_covariance,
        disparity=disparity, disparity_variance=disparity_variance,
        disparity_valid=disparity_valid, descriptor=descriptors.detach().cpu().clone(),
    )
    return record, None, diagnostics


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


def match_orbslam_descriptors(
    current: GeometryFeatureRecord,
    candidate: GeometryFeatureRecord,
    current_angles: torch.Tensor,
    candidate_angles: torch.Tensor,
) -> tuple[list[FixedMatch], dict[str, Any]]:
    """ORB-SLAM-inspired global NN matching with deterministic orientation filtering."""
    current_desc = current.descriptor.detach().cpu().numpy().astype(np.uint8, copy=False)
    candidate_desc = candidate.descriptor.detach().cpu().numpy().astype(np.uint8, copy=False)
    diagnostics: dict[str, Any] = {
        "descriptor_match_mode": "orbslam",
        "descriptors": {"current": int(len(current_desc)), "candidate": int(len(candidate_desc))},
        "distance_threshold_inclusive": ORB_SLAM_HAMMING_THRESHOLD,
        "ratio_threshold_strict": ORB_SLAM_RATIO_THRESHOLD,
        "orientation_filter": "simplified_orbslam_orientation_histogram",
        "orientation_histogram_bins": ORB_SLAM_ORIENTATION_BINS,
        "orientation_weak_bin_ratio": ORB_SLAM_ORIENTATION_WEAK_BIN_RATIO,
        "after_distance": 0,
        "after_ratio": 0,
        "after_candidate_unique": 0,
        "after_valid_orientation": 0,
        "after_orientation_histogram": 0,
        "invalid_orientation_matches": 0,
        "selected_orientation_bins": [],
    }
    if (
        current_desc.ndim != 2 or candidate_desc.ndim != 2
        or not len(current_desc) or not len(candidate_desc)
    ):
        return [], diagnostics
    if len(candidate_desc) < 2:
        diagnostics["reject_code"] = "insufficient_second_neighbor"
        return [], diagnostics

    candidate_original = candidate.original_index.detach().cpu().numpy().astype(np.int64, copy=False)
    proposed: list[FixedMatch] = []
    for current_local, descriptor in enumerate(current_desc):
        distances = _BIT_COUNTS[np.bitwise_xor(candidate_desc, descriptor)].sum(axis=1)
        order = np.lexsort((candidate_original, distances))
        best_local, second_local = int(order[0]), int(order[1])
        best_distance = int(distances[best_local])
        second_distance = int(distances[second_local])
        if best_distance > ORB_SLAM_HAMMING_THRESHOLD:
            continue
        diagnostics["after_distance"] += 1
        if not best_distance < ORB_SLAM_RATIO_THRESHOLD * second_distance:
            continue
        diagnostics["after_ratio"] += 1
        proposed.append(FixedMatch(
            distance=best_distance,
            current_local=current_local,
            candidate_local=best_local,
            current_original=int(current.original_index[current_local]),
            candidate_original=int(candidate.original_index[best_local]),
        ))

    proposed.sort(key=lambda item: (item.distance, item.current_original, item.candidate_original))
    used_candidate: set[int] = set()
    unique: list[FixedMatch] = []
    for item in proposed:
        if item.candidate_local in used_candidate:
            continue
        used_candidate.add(item.candidate_local)
        unique.append(item)
    diagnostics["after_candidate_unique"] = len(unique)

    current_angle_values = current_angles.detach().cpu().reshape(-1).double()
    candidate_angle_values = candidate_angles.detach().cpu().reshape(-1).double()
    valid_oriented: list[tuple[FixedMatch, int]] = []
    histogram = [0] * ORB_SLAM_ORIENTATION_BINS
    for item in unique:
        if (
            item.current_original < 0 or item.current_original >= len(current_angle_values)
            or item.candidate_original < 0 or item.candidate_original >= len(candidate_angle_values)
        ):
            diagnostics["invalid_orientation_matches"] += 1
            continue
        current_angle = float(current_angle_values[item.current_original])
        candidate_angle = float(candidate_angle_values[item.candidate_original])
        if not (
            math.isfinite(current_angle) and math.isfinite(candidate_angle)
            and 0.0 <= current_angle < 360.0 and 0.0 <= candidate_angle < 360.0
        ):
            diagnostics["invalid_orientation_matches"] += 1
            continue
        delta = (current_angle - candidate_angle) % 360.0
        bin_width = 360.0 / ORB_SLAM_ORIENTATION_BINS
        bin_index = (
            int(math.floor(delta / bin_width + 0.5)) % ORB_SLAM_ORIENTATION_BINS
        )
        histogram[bin_index] += 1
        valid_oriented.append((item, bin_index))
    diagnostics["after_valid_orientation"] = len(valid_oriented)
    populated = sorted(
        ((count, index) for index, count in enumerate(histogram) if count > 0),
        key=lambda value: (-value[0], value[1]),
    )[:3]
    selected_bins: list[int] = []
    if populated:
        maximum = populated[0][0]
        selected_bins = [
            index for count, index in populated
            if count >= ORB_SLAM_ORIENTATION_WEAK_BIN_RATIO * maximum
        ]
    selected = set(selected_bins)
    matches = [item for item, bin_index in valid_oriented if bin_index in selected]
    diagnostics["selected_orientation_bins"] = selected_bins
    diagnostics["after_orientation_histogram"] = len(matches)
    return matches, diagnostics


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
    if matrix.ndim == 3:
        if matrix.shape[0] != 1:
            raise ValueError(
                f"_pose_magnitude expects one pose, got batch size {matrix.shape[0]}"
            )
        matrix = matrix[0]
    if matrix.shape != (4, 4):
        raise ValueError(
            f"_pose_magnitude expects a 4x4 pose matrix, got {tuple(matrix.shape)}"
        )
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

    descriptor_match_mode = str(getattr(config, "descriptor_match_mode", "vins_legacy"))
    if descriptor_match_mode == "orbslam":
        matches, matching = match_orbslam_descriptors(
            current, historical,
            current_frame.orb_keypoints[:, 3], candidate_frame.orb_keypoints[:, 3],
        )
        row.update(matching)
        if matching.get("reject_code") == "insufficient_second_neighbor":
            row.update({
                "reject_code": "insufficient_second_neighbor",
                "unique_descriptor_matches": 0,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            })
            return GeometryResult(row, None, None)
    else:
        matches = match_fixed_descriptors(current, historical, int(config.hamming_threshold))
        row.update({
            "descriptor_match_mode": "vins_legacy",
            "descriptors": {
                "current": int(len(current.descriptor)),
                "candidate": int(len(historical.descriptor)),
            },
            "after_distance": None,
            "after_ratio": None,
            "after_candidate_unique": len(matches),
            "after_valid_orientation": None,
            "after_orientation_histogram": None,
            "orientation_filter": None,
        })
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
    row["pnp_input_points"] = int(len(points_ned))
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
    disp_valid &= torch.isfinite(historical.point_camera[selected_candidate]).all(dim=1)
    disp_valid &= torch.isfinite(historical.point_covariance_camera[selected_candidate]).all(dim=(1, 2))
    disp_valid &= torch.isfinite(current.pixel_uv[selected_current]).all(dim=1)
    row["information_valid_inliers"] = int(disp_valid.sum())
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
    return GeometryResult(
        row,
        constraint,
        info_matrix,
        pnp_pose_current_candidate=pose,
        pnp_current_local=selected_current.detach().cpu(),
        pnp_candidate_local=selected_candidate.detach().cpu(),
    )


def refine_geometry_with_network(
    result: GeometryResult,
    config: SimpleNamespace,
    frontend: IFrontend,
    current_frame: LoopFrameRecord,
    candidate_frame: LoopFrameRecord,
    current: GeometryFeatureRecord,
    historical: GeometryFeatureRecord,
    fixed_information: torch.Tensor,
    max_translation_m: float,
    max_rotation_deg: float,
) -> GeometryResult:
    """Refine one ORB/PnP-approved pair with one historical-to-current network match."""
    residual_mode = str(getattr(config, "residual_mode", "disp"))
    if residual_mode not in {"disp", "icp"}:
        raise ValueError(f"unsupported network refinement residual mode {residual_mode!r}")
    row = result.row
    refinement: dict[str, Any] = {
        "enabled": True,
        "status": "failed",
        "reason": None,
        "frontend_inference_calls": 1,
        "sampling": "floor_map_index_float_geometry",
        "flow_covariance_channels": ["uu", "vv", "uv"],
        "cross_covariance_mode": "ignored_independent_approximation",
        "huber_delta": float(getattr(config, "huber_delta", 2.795)),
        "min_points": int(getattr(config, "min_points", 6)),
        "residual_mode": residual_mode,
        "observation_covariance_model": (
            "bilateral_match_covariance_3d_independent_sum"
            if residual_mode == "icp"
            else "reprojection_disparity_candidate_3d_plus_measurement"
        ),
    }
    row["network_refinement"] = refinement
    if (
        result.constraint is None
        or result.pnp_pose_current_candidate is None
        or result.pnp_current_local is None
        or result.pnp_candidate_local is None
    ):
        refinement["reason"] = "missing_pnp_state"
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)

    try:
        depth_current, match = frontend.estimate_pair(
            candidate_frame.to_stereo_data(getattr(frontend.config, "device", "cpu")),
            current_frame.to_stereo_data(getattr(frontend.config, "device", "cpu")),
        )
    except Exception as error:
        refinement.update({
            "reason": "frontend_exception",
            "detail": f"{type(error).__name__}: {error}",
        })
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)

    flow_map = getattr(match, "flow", None)
    covariance_map = getattr(match, "cov", None)
    disparity_map = getattr(depth_current, "disparity", None)
    disparity_variance_map = getattr(depth_current, "disparity_uncertainty", None)
    current_depth_map = getattr(depth_current, "depth", None)
    if (
        flow_map is None or covariance_map is None
        or flow_map.ndim != 4 or flow_map.shape[1] != 2
        or covariance_map.ndim != 4 or covariance_map.shape[1] != 3
        or (
            residual_mode == "disp"
            and (
                disparity_map is None or disparity_variance_map is None
                or disparity_map.ndim != 4 or disparity_map.shape[1] != 1
                or disparity_variance_map.ndim != 4
                or disparity_variance_map.shape[1] != 1
            )
        )
        or (
            residual_mode == "icp"
            and (
                current_depth_map is None or current_depth_map.ndim != 4
                or current_depth_map.shape[1] != 1
            )
        )
    ):
        refinement["reason"] = "missing_or_invalid_frontend_uncertainty"
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)

    current_local = result.pnp_current_local.long()
    candidate_local = result.pnp_candidate_local.long()
    candidate_uv = historical.pixel_uv[candidate_local].detach().cpu().double()
    orb_current_uv = current.pixel_uv[current_local].detach().cpu().double()
    floor_candidate = torch.floor(candidate_uv).long()
    height, width = int(flow_map.shape[-2]), int(flow_map.shape[-1])
    candidate_inbound = (
        torch.isfinite(candidate_uv).all(dim=1)
        & (floor_candidate[:, 0] >= 0) & (floor_candidate[:, 0] < width)
        & (floor_candidate[:, 1] >= 0) & (floor_candidate[:, 1] < height)
    )
    source_indices = torch.nonzero(candidate_inbound, as_tuple=False).squeeze(1)
    if source_indices.numel() == 0:
        refinement["reason"] = "no_inbound_pnp_inlier_sources"
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)

    floor_source = floor_candidate[source_indices]
    flow_cpu = flow_map[0].detach().cpu().double()
    covariance_cpu = covariance_map[0].detach().cpu().double()
    sampled_flow = flow_cpu[:, floor_source[:, 1], floor_source[:, 0]].T
    sampled_covariance = covariance_cpu[:, floor_source[:, 1], floor_source[:, 0]].T
    network_current_uv = candidate_uv[source_indices] + sampled_flow
    floor_current = torch.floor(network_current_uv).long()
    geometry_map = disparity_map if residual_mode == "disp" else current_depth_map
    assert geometry_map is not None
    current_height, current_width = int(geometry_map.shape[-2]), int(geometry_map.shape[-1])
    current_inbound = (
        torch.isfinite(network_current_uv).all(dim=1)
        & (floor_current[:, 0] >= 0) & (floor_current[:, 0] < current_width)
        & (floor_current[:, 1] >= 0) & (floor_current[:, 1] < current_height)
    )
    mask_valid = torch.ones_like(current_inbound)
    if getattr(match, "mask", None) is not None:
        mask_cpu = match.mask[0, 0].detach().cpu().bool()
        mask_valid = mask_cpu[floor_source[:, 1], floor_source[:, 0]]
    observation_arrays: dict[str, torch.Tensor]
    intrinsic = (
        current_frame.intrinsic[0]
        if current_frame.intrinsic.ndim == 3 else current_frame.intrinsic
    ).detach().cpu().double()
    huber_delta = float(getattr(config, "huber_delta", 2.795))
    valid_count = 0
    if residual_mode == "disp":
        assert disparity_map is not None and disparity_variance_map is not None
        disparity_cpu = disparity_map[0, 0].detach().cpu().double()
        disparity_variance_cpu = disparity_variance_map[0, 0].detach().cpu().double()
        sampled_disparity = torch.full((len(source_indices),), torch.nan, dtype=torch.float64)
        sampled_disparity_variance = torch.full_like(sampled_disparity, torch.nan)
        if bool(current_inbound.any()):
            uv = floor_current[current_inbound]
            sampled_disparity[current_inbound] = disparity_cpu[uv[:, 1], uv[:, 0]]
            sampled_disparity_variance[current_inbound] = (
                disparity_variance_cpu[uv[:, 1], uv[:, 0]]
            )
        candidate_points = historical.point_camera[
            candidate_local[source_indices]
        ].detach().cpu().double()
        candidate_covariance = historical.point_covariance_camera[
            candidate_local[source_indices]
        ].detach().cpu().double()
        uv_covariance = covariance_triplet_to_matrix(sampled_covariance)
        valid = (
            current_inbound & mask_valid
            & torch.isfinite(sampled_flow).all(dim=1)
            & torch.isfinite(uv_covariance).all(dim=(1, 2))
            & torch.isfinite(sampled_disparity)
            & torch.isfinite(sampled_disparity_variance)
            & (sampled_disparity > 0.0)
            & (sampled_disparity_variance > 0.0)
            & torch.isfinite(candidate_points).all(dim=1)
            & torch.isfinite(candidate_covariance).all(dim=(1, 2))
            & (candidate_points[:, 0] > 0.0)
        )
        observation_arrays = {
            "candidate_points": candidate_points,
            "candidate_covariance": candidate_covariance,
            "network_current_uv": network_current_uv,
            "uv_covariance": uv_covariance,
            "sampled_disparity": sampled_disparity,
            "sampled_disparity_variance": sampled_disparity_variance,
        }
    else:
        kernel_size = int(getattr(config, "kernel_size", 31))
        half = kernel_size // 2
        candidate_patch_inbound = (
            (floor_source[:, 0] >= half)
            & (floor_source[:, 0] < width - half)
            & (floor_source[:, 1] >= half)
            & (floor_source[:, 1] < height - half)
        )
        current_patch_inbound = (
            current_inbound
            & (floor_current[:, 0] >= half)
            & (floor_current[:, 0] < current_width - half)
            & (floor_current[:, 1] >= half)
            & (floor_current[:, 1] < current_height - half)
        )
        valid = (
            candidate_patch_inbound & current_patch_inbound & mask_valid
            & torch.isfinite(sampled_flow).all(dim=1)
            & torch.isfinite(sampled_covariance).all(dim=1)
        )
        selected_candidate_uv = candidate_uv[source_indices][valid]
        selected_current_uv = network_current_uv[valid]
        selected_flow_covariance = sampled_covariance[valid]
        candidate_depth_map = candidate_frame.depth.detach().cpu().double()
        assert current_depth_map is not None
        current_depth_cpu = current_depth_map.detach().cpu().double()
        candidate_floor = floor_source[valid]
        selected_current_floor = floor_current[valid]
        candidate_center_depth = candidate_depth_map[
            0, 0, candidate_floor[:, 1], candidate_floor[:, 0]
        ]
        current_center_depth = current_depth_cpu[
            0, 0, selected_current_floor[:, 1], selected_current_floor[:, 0]
        ]
        candidate_center_depth_covariance = None
        if candidate_frame.depth_covariance is not None:
            cached_depth_covariance = candidate_frame.depth_covariance.detach().cpu().double()
            candidate_center_depth_covariance = cached_depth_covariance[
                0, 0, candidate_floor[:, 1], candidate_floor[:, 0]
            ]
        current_center_depth_covariance = None
        current_depth_covariance_map = getattr(depth_current, "cov", None)
        if current_depth_covariance_map is not None:
            current_depth_covariance_cpu = current_depth_covariance_map.detach().cpu().double()
            current_center_depth_covariance = current_depth_covariance_cpu[
                0, 0, selected_current_floor[:, 1], selected_current_floor[:, 0]
            ]
        candidate_intrinsic = (
            candidate_frame.intrinsic[0]
            if candidate_frame.intrinsic.ndim == 3 else candidate_frame.intrinsic
        ).detach().cpu().double()
        fixed_candidate_covariance = torch.zeros(
            (len(selected_candidate_uv), 3), dtype=torch.float64,
        )
        fixed_candidate_covariance[:, :2] = 0.25
        covariance_parameters = {
            "kernel_size": kernel_size,
            "match_cov_default": float(getattr(config, "match_cov_default", 0.25)),
            "min_position_std": float(getattr(config, "min_flow_cov", 0.25)),
            "min_depth_variance": float(getattr(config, "min_depth_cov", 0.05)),
        }
        try:
            candidate_wavg, candidate_wvar, candidate_covariance = (
                match_covariance_local_statistics(
                    kp=selected_candidate_uv,
                    depth=candidate_depth_map,
                    depth_covariance=candidate_center_depth_covariance,
                    position_covariance=fixed_candidate_covariance,
                    fx=candidate_intrinsic[0, 0], fy=candidate_intrinsic[1, 1],
                    cx=candidate_intrinsic[0, 2], cy=candidate_intrinsic[1, 2],
                    device=torch.device("cpu"), **covariance_parameters,
                )
            )
            current_wavg, current_wvar, current_covariance = (
                match_covariance_local_statistics(
                    kp=selected_current_uv,
                    depth=current_depth_cpu,
                    depth_covariance=current_center_depth_covariance,
                    position_covariance=selected_flow_covariance,
                    fx=intrinsic[0, 0], fy=intrinsic[1, 1],
                    cx=intrinsic[0, 2], cy=intrinsic[1, 2],
                    device=torch.device("cpu"), **covariance_parameters,
                )
            )
        except (IndexError, RuntimeError, ValueError) as error:
            refinement["reason"] = f"local_depth_covariance_failed:{error}"
            row.update({"information_valid": False, "pgo_comparison_eligible": False})
            return GeometryResult(row, None, None)
        candidate_points = pixel2point_NED(
            selected_candidate_uv, candidate_center_depth, candidate_intrinsic,
        ).double()
        current_points = pixel2point_NED(
            selected_current_uv, current_center_depth, intrinsic,
        ).double()
        finite_icp = (
            torch.isfinite(candidate_points).all(dim=1)
            & torch.isfinite(current_points).all(dim=1)
            & torch.isfinite(candidate_covariance).all(dim=(1, 2))
            & torch.isfinite(current_covariance).all(dim=(1, 2))
            & torch.isfinite(candidate_wavg) & torch.isfinite(candidate_wvar)
            & torch.isfinite(current_wavg) & torch.isfinite(current_wvar)
            & (candidate_center_depth > 0.0) & (current_center_depth > 0.0)
        )
        observation_arrays = {
            "candidate_points": candidate_points[finite_icp],
            "candidate_covariance": candidate_covariance[finite_icp],
            "current_points": current_points[finite_icp],
            "current_covariance": current_covariance[finite_icp],
        }
        refinement.update({
            "kernel_size": kernel_size,
            "candidate_position_covariance": "fixed_0.25I",
            "local_depth_candidate_points": int(len(candidate_points)),
            "local_depth_current_points": int(len(current_points)),
            "local_depth_valid_points": int(finite_icp.sum()),
        })
        valid_count = int(finite_icp.sum())
    refinement.update({
        "pnp_positive_inlier_sources": int(len(candidate_local)),
        "candidate_inbound": int(candidate_inbound.sum()),
        "flow_and_disparity_valid": int(valid.sum()) if residual_mode == "disp" else None,
        "flow_and_icp_valid": valid_count if residual_mode == "icp" else None,
        "match_mask_present": getattr(match, "mask", None) is not None,
    })
    flow_orb_difference = torch.linalg.vector_norm(
        network_current_uv - orb_current_uv[source_indices], dim=1,
    )
    finite_difference = flow_orb_difference[torch.isfinite(flow_orb_difference)]
    refinement["flow_vs_orb_current_uv_distance_px"] = {
        "count": int(finite_difference.numel()),
        "p50": (
            float(torch.quantile(finite_difference, 0.5))
            if finite_difference.numel() else None
        ),
        "p90": (
            float(torch.quantile(finite_difference, 0.9))
            if finite_difference.numel() else None
        ),
    }
    minimum = int(getattr(config, "min_points", 6))
    available = int(valid.sum()) if residual_mode == "disp" else valid_count
    if available < minimum:
        refinement["reason"] = "insufficient_valid_flow_refinement_points"
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)

    if residual_mode == "disp":
        observation_arrays = {
            name: value[valid] for name, value in observation_arrays.items()
        }
    baseline = current_frame.baseline.detach().cpu().double()

    def select_observations(mask: torch.Tensor) -> None:
        nonlocal observation_arrays
        observation_arrays = {
            name: value[mask] for name, value in observation_arrays.items()
        }

    def system_at(pose: pp.LieTensor):
        if residual_mode == "icp":
            return robust_icp_observation_system(
                pose,
                observation_arrays["candidate_points"],
                observation_arrays["candidate_covariance"],
                observation_arrays["current_points"],
                observation_arrays["current_covariance"],
                huber_delta,
            )
        return robust_observation_system(
            pose,
            observation_arrays["candidate_points"],
            observation_arrays["candidate_covariance"],
            observation_arrays["network_current_uv"],
            observation_arrays["uv_covariance"],
            observation_arrays["sampled_disparity"],
            observation_arrays["sampled_disparity_variance"],
            intrinsic,
            baseline,
            huber_delta,
        )

    initial_pose = result.pnp_pose_current_candidate.detach().cpu().double()
    pose = initial_pose
    try:
        if residual_mode == "icp":
            _, _, initial_covariance, _ = icp_linearization(
                pose,
                observation_arrays["candidate_points"],
                observation_arrays["candidate_covariance"],
                observation_arrays["current_points"],
                observation_arrays["current_covariance"],
            )
        else:
            _, _, initial_covariance, _ = reproj_disp_linearization(
                pose,
                observation_arrays["candidate_points"],
                observation_arrays["candidate_covariance"],
                observation_arrays["network_current_uv"],
                observation_arrays["uv_covariance"],
                observation_arrays["sampled_disparity"],
                observation_arrays["sampled_disparity_variance"],
                intrinsic,
                baseline,
            )
    except (RuntimeError, ValueError) as error:
        refinement["reason"] = f"initial_covariance_linearization_failed:{error}"
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)
    initial_covariance = 0.5 * (
        initial_covariance + initial_covariance.transpose(-1, -2)
    )
    covariance_finite = torch.isfinite(initial_covariance).all(dim=(1, 2))
    covariance_spd = torch.zeros_like(covariance_finite)
    if bool(covariance_finite.any()):
        _, cholesky_info = torch.linalg.cholesky_ex(
            initial_covariance[covariance_finite],
        )
        covariance_spd[covariance_finite] = cholesky_info == 0
    refinement.update({
        "covariance_checked_points": int(len(covariance_spd)),
        "covariance_spd_points": int(covariance_spd.sum()),
        "covariance_dropped_points": int((~covariance_spd).sum()),
    })
    if int(covariance_spd.sum()) < minimum:
        refinement["reason"] = "insufficient_spd_flow_refinement_points"
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)
    if not bool(covariance_spd.all()):
        select_observations(covariance_spd)

    try:
        initial_system = system_at(pose)
    except ValueError as error:
        refinement["reason"] = str(error)
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)
    positive = initial_system.transformed[:, 0] > 1e-6
    if int(positive.sum()) < minimum:
        refinement["reason"] = "insufficient_positive_flow_refinement_points"
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)
    if not bool(positive.all()):
        select_observations(positive)
        initial_system = system_at(pose)

    initial_cost = initial_system.robust_cost
    current_cost = initial_cost
    damping = float(getattr(config, "damping_initial", 1e-3))
    accepted_steps = 0
    rejected_trials = 0
    converged = False
    max_iterations = int(getattr(config, "max_iterations", 15))
    for _ in range(max_iterations):
        try:
            system = system_at(pose)
        except ValueError:
            break
        diagonal = torch.diagonal(system.hessian).clamp_min(1e-12)
        accepted = False
        for _trial in range(5):
            try:
                delta = torch.linalg.solve(
                    system.hessian + damping * torch.diag(diagonal),
                    -system.gradient,
                )
            except torch.linalg.LinAlgError:
                delta = torch.full((6,), torch.nan, dtype=torch.float64)
            if not torch.isfinite(delta).all():
                damping = min(damping * 10.0, 1e9)
                rejected_trials += 1
                continue
            translation_step = float(torch.linalg.vector_norm(delta[:3]))
            rotation_step = float(torch.linalg.vector_norm(delta[3:]))
            if translation_step <= 1e-6 and rotation_step <= 1e-6:
                converged = True
                accepted = True
                break
            trial_pose = pp.se3(delta).Exp() @ pose
            try:
                trial_system = system_at(trial_pose)
            except ValueError:
                trial_system = None
            if (
                trial_system is not None
                and math.isfinite(trial_system.robust_cost)
                and trial_system.robust_cost < current_cost
            ):
                pose = trial_pose
                current_cost = trial_system.robust_cost
                damping = max(damping / 3.0, 1e-9)
                accepted_steps += 1
                accepted = True
                break
            damping = min(damping * 10.0, 1e9)
            rejected_trials += 1
        if converged or not accepted:
            break

    try:
        final_system = system_at(pose)
    except ValueError as error:
        refinement["reason"] = str(error)
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)
    final_positive = (
        torch.isfinite(final_system.transformed).all(dim=1)
        & (final_system.transformed[:, 0] > 1e-6)
    )
    refinement["final_nonpositive_dropped_points"] = int((~final_positive).sum())
    if int(final_positive.sum()) < minimum:
        refinement["reason"] = "insufficient_positive_points_after_refinement"
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)
    comparison_initial_cost = initial_cost
    if not bool(final_positive.all()):
        select_observations(final_positive)
        try:
            comparison_initial_cost = system_at(initial_pose).robust_cost
            final_system = system_at(pose)
        except ValueError as error:
            refinement["reason"] = str(error)
            row.update({"information_valid": False, "pgo_comparison_eligible": False})
            return GeometryResult(row, None, None)
    tolerance = max(1e-9, 1e-9 * abs(comparison_initial_cost))
    if (
        not math.isfinite(final_system.robust_cost)
        or final_system.robust_cost > comparison_initial_cost + tolerance
        or not torch.isfinite(pose.tensor()).all()
    ):
        refinement["reason"] = "nonfinite_or_increased_refinement_cost"
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)

    refined_translation_m, refined_rotation_deg = _pose_magnitude(pose)
    pnp_to_refined_translation_m, pnp_to_refined_rotation_deg = _pose_difference(
        pose, initial_pose,
    )
    refinement.update({
        "refined_translation_m": refined_translation_m,
        "refined_rotation_deg": refined_rotation_deg,
        "pnp_to_refined_translation_m": pnp_to_refined_translation_m,
        "pnp_to_refined_rotation_deg": pnp_to_refined_rotation_deg,
        "max_translation_m": float(max_translation_m),
        "max_rotation_deg": float(max_rotation_deg),
    })
    if (
        refined_translation_m >= float(max_translation_m)
        or refined_rotation_deg >= float(max_rotation_deg)
    ):
        refinement["reason"] = "refined_pose_safety_gate"
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)

    information, information_payload = information_diagnostics(
        final_system,
        pose,
        point_count=len(observation_arrays["candidate_points"]),
    )
    if information is None:
        refinement["reason"] = information_payload.get("reason")
        refinement["information"] = information_payload
        row.update({"information_valid": False, "pgo_comparison_eligible": False})
        return GeometryResult(row, None, None)

    edge = pose.Inv()
    refinement.update({
        "status": "succeeded",
        "reason": None,
        "initial_robust_cost": comparison_initial_cost,
        "optimization_initial_robust_cost": initial_cost,
        "final_robust_cost": final_system.robust_cost,
        "accepted_steps": accepted_steps,
        "rejected_trials": rejected_trials,
        "converged": converged,
        "final_point_count": len(observation_arrays["candidate_points"]),
    })
    row.update({
        "status": "accepted",
        "reject_code": None,
        "geometry_accepted": True,
        "information_valid": True,
        "pgo_comparison_eligible": True,
        "information_observe": information_payload,
        "refined_relative_pose": edge.tensor().detach().cpu().tolist(),
        "relative_pose": edge.tensor().detach().cpu().tolist(),
    })
    constraint = replace(
        result.constraint,
        relative_pose=row["relative_pose"],
        information=fixed_information.detach().cpu().tolist(),
        num_flow_points=len(observation_arrays["candidate_points"]),
    )
    return GeometryResult(
        row,
        constraint,
        information,
        pnp_pose_current_candidate=result.pnp_pose_current_candidate,
        pnp_current_local=result.pnp_current_local,
        pnp_candidate_local=result.pnp_candidate_local,
    )


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
    solver_diagnostics = getattr(optimizer, "last_optimization_diagnostics", None)
    result["solver_diagnostics"] = solver_diagnostics
    if (
        getattr(optimizer, "solver", None) == "sparse_lm"
        and isinstance(solver_diagnostics, dict)
        and solver_diagnostics.get("safe") is False
    ):
        result["reason"] = "optimizer_reported_unsafe"
        return None, result
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
