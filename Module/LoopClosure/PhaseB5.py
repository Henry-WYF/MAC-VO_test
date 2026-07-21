from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np
import pypose as pp
import torch

from Module.Frontend.Frontend import IFrontend
from Module.Frontend.Matching import IMatcher
from Module.Frontend.StereoDepth import IStereoDepth
from Utility.Point import filterPointsInRange, pixel2point_NED
from Utility.Selection import local_minimum_nms

from .Record import LoopFrameRecord


PHASE_B5_SCHEMA_VERSION = 1


def _stats(values: torch.Tensor) -> dict[str, int | float | None]:
    finite = values.detach().cpu().double().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    result: dict[str, int | float | None] = {
        "count": int(finite.numel()), "min": None, "p25": None, "p50": None,
        "p75": None, "p90": None, "p95": None, "p99": None, "max": None,
    }
    if finite.numel() == 0:
        return result
    quantiles = torch.quantile(
        finite, torch.tensor([0.25, 0.50, 0.75, 0.90, 0.95, 0.99], dtype=torch.float64),
        interpolation="linear",
    )
    result.update({
        "min": float(finite.min()), "p25": float(quantiles[0]),
        "p50": float(quantiles[1]), "p75": float(quantiles[2]),
        "p90": float(quantiles[3]), "p95": float(quantiles[4]),
        "p99": float(quantiles[5]), "max": float(finite.max()),
    })
    return result


def tensor_signature(indices: torch.Tensor) -> str:
    array = indices.detach().cpu().to(torch.int64).contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_content_sha256(
    records: list[dict[str, Any]], record_root: Path | None,
) -> str:
    """Hash the ordered cache files, including their relative paths."""
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda value: int(value["sensor_frame_idx"])):
        relative = str(item["file"])
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if record_root is None:
            continue
        path = record_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def normalized_covariance_risk(
    covariance: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return lambda-max risk, point validity, and PSD diagnostics for CxN or BxCxHxW covariance."""
    if covariance.ndim == 4:
        if covariance.shape[0] != 1 or covariance.shape[1] != 3:
            raise ValueError(f"covariance must have shape 1x3xHxW, got {tuple(covariance.shape)}")
        uu, vv, uv = covariance[0, 0], covariance[0, 1], covariance[0, 2]
    elif covariance.ndim == 2:
        if covariance.shape[0] != 3:
            raise ValueError(f"covariance must have shape 3xN, got {tuple(covariance.shape)}")
        uu, vv, uv = covariance[0], covariance[1], covariance[2]
    else:
        raise ValueError(f"unsupported covariance shape {tuple(covariance.shape)}")

    a = uu / float(width * width)
    d = vv / float(height * height)
    b = uv / float(width * height)
    finite = torch.isfinite(a) & torch.isfinite(d) & torch.isfinite(b)
    trace = a + d
    half_diff = 0.5 * (a - d)
    radius = torch.sqrt(torch.clamp(half_diff.square() + b.square(), min=0.0))
    lambda_min = 0.5 * trace - radius
    lambda_max = 0.5 * trace + radius
    tolerance = torch.maximum(
        torch.full_like(trace, 1e-12),
        1e-6 * torch.maximum(0.5 * trace.abs(), torch.full_like(trace, 1e-12)),
    )
    small_negative = finite & (lambda_min < 0.0) & (lambda_min >= -tolerance)
    invalid_psd = finite & (lambda_min < -tolerance)
    valid = finite & ~invalid_psd
    lambda_max = torch.where(valid, torch.clamp(lambda_max, min=0.0), torch.full_like(lambda_max, torch.nan))
    return lambda_max, valid, {
        "finite_points": int(finite.sum()),
        "small_negative_eigenvalues_clamped": int(small_negative.sum()),
        "invalid_psd_points": int(invalid_psd.sum()),
        "psd_tolerance_formula": "max(1e-12,1e-6*max(abs(trace)/2,1e-12))",
    }


def _occupied_grid_cells(uv: torch.Tensor, width: int, height: int, rows: int = 8, cols: int = 8) -> int:
    if uv.numel() == 0:
        return 0
    col = torch.clamp((uv[:, 0] * cols / max(width, 1)).long(), 0, cols - 1)
    row = torch.clamp((uv[:, 1] * rows / max(height, 1)).long(), 0, rows - 1)
    return int(torch.unique(row * cols + col).numel())


def _pose_from_opencv(rvec: np.ndarray, tvec: np.ndarray) -> pp.LieTensor:
    rotation_cv, _ = cv2.Rodrigues(rvec)
    ned_to_cv = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    rotation = ned_to_cv.T @ rotation_cv @ ned_to_cv
    translation = ned_to_cv.T @ np.asarray(tvec).reshape(3)
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, :3] = torch.from_numpy(rotation)
    matrix[:3, 3] = torch.from_numpy(translation)
    return pp.from_matrix(matrix, pp.SE3_type).float()


def _stable_seed(current_sensor: int, candidate_sensor: int) -> int:
    return (int(current_sensor) * 1_000_003 + int(candidate_sensor)) & 0x7FFFFFFF


def orb_geometry_observe(
    current: LoopFrameRecord,
    historical: LoopFrameRecord,
    config: SimpleNamespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "status": "rejected", "reject_code": None, "orb_gate_pass": False,
        "essential": None, "pnp": None,
        "gate_definition": {
            "matching": "bidirectional Hamming KNN k=2, Lowe ratio, mutual",
            "essential_is_blocking": False,
            "pnp_reprojection_error_px": 3.0, "pnp_confidence": 0.999,
            "pnp_iterations": 100, "min_inliers": 20,
            "min_inlier_ratio": 0.25, "min_occupied_8x8_cells": 8,
            "min_positive_depth_ratio_over_inliers": 0.90,
            "max_mean_inlier_reprojection_error_px": 3.0,
        },
    }
    desc_h = np.ascontiguousarray(
        historical.orb_descriptors.detach().cpu().numpy().astype(np.uint8, copy=False)
    )
    desc_c = np.ascontiguousarray(
        current.orb_descriptors.detach().cpu().numpy().astype(np.uint8, copy=False)
    )
    if len(desc_h) == 0 or len(desc_c) == 0:
        result["reject_code"] = "empty_descriptors"
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return result
    if len(desc_h) < 2 or len(desc_c) < 2:
        result["reject_code"] = "insufficient_knn_neighbors"
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    ratio = float(getattr(config, "ratio", 0.8))
    forward = matcher.knnMatch(desc_h, desc_c, k=2)
    backward = matcher.knnMatch(desc_c, desc_h, k=2)
    fwd = {(m.queryIdx, m.trainIdx): float(m.distance) for pair in forward if len(pair) >= 2 for m, n in [pair[:2]] if m.distance < ratio * n.distance}
    bwd = {(m.trainIdx, m.queryIdx) for pair in backward if len(pair) >= 2 for m, n in [pair[:2]] if m.distance < ratio * n.distance}
    matches = sorted(
        [(query_idx, train_idx, distance) for (query_idx, train_idx), distance in fwd.items() if (query_idx, train_idx) in bwd],
        key=lambda item: (item[0], item[1]),
    )
    result["forward_ratio_matches"] = len(fwd)
    result["mutual_ratio_matches"] = len(matches)
    if len(matches) < 5:
        result["reject_code"] = "insufficient_orb_matches"
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    h_idx = torch.tensor([item[0] for item in matches], dtype=torch.long)
    c_idx = torch.tensor([item[1] for item in matches], dtype=torch.long)
    uv_h = historical.orb_keypoints[h_idx, :2].float()
    uv_c = current.orb_keypoints[c_idx, :2].float()
    K = (current.intrinsic[0] if current.intrinsic.ndim == 3 else current.intrinsic).cpu().numpy().astype(np.float64)
    seed = _stable_seed(current.sensor_frame_idx, historical.sensor_frame_idx)
    rng_applied = hasattr(cv2, "setRNGSeed")

    displacement = torch.linalg.vector_norm(uv_c - uv_h, dim=1)
    essential: dict[str, Any] = {
        "attempted": len(matches) >= 5, "succeeded": False,
        "median_pixel_displacement": float(torch.median(displacement)),
        "low_parallax_observation": bool(float(torch.median(displacement)) < 1.5),
        "reject_code": None,
    }
    if len(matches) >= 5:
        if rng_applied:
            cv2.setRNGSeed(seed)
        try:
            E, mask = cv2.findEssentialMat(
                uv_h.numpy().astype(np.float64), uv_c.numpy().astype(np.float64), K,
                method=cv2.RANSAC, prob=0.999, threshold=1.5,
            )
            if E is not None and mask is not None:
                recovered, _, _, pose_mask = cv2.recoverPose(
                    E, uv_h.numpy().astype(np.float64), uv_c.numpy().astype(np.float64), K, mask=mask,
                )
                essential.update({
                    "succeeded": bool(recovered > 0),
                    "ransac_inliers": int(mask.astype(bool).sum()),
                    "cheirality_inliers": int(recovered),
                    "recover_pose_cheirality_failed": bool(recovered <= 0),
                    "degenerate_or_too_few_cheirality_inliers": bool(recovered < 5),
                    "pose_mask_inliers": int(pose_mask.astype(bool).sum()) if pose_mask is not None else 0,
                })
                if recovered <= 0:
                    essential["reject_code"] = "recover_pose_cheirality_failed"
            else:
                essential["reject_code"] = "essential_matrix_failed"
        except cv2.error as error:
            essential["error"] = str(error)
            essential["reject_code"] = "essential_exception"
    result["essential"] = essential

    depth_map = historical.depth[0, 0].detach().cpu().float()
    floor_uv = torch.floor(uv_h).long()
    inbound = (
        (floor_uv[:, 0] >= 0) & (floor_uv[:, 0] < historical.width)
        & (floor_uv[:, 1] >= 0) & (floor_uv[:, 1] < historical.height)
    )
    depths = torch.full((len(matches),), torch.nan)
    depths[inbound] = depth_map[floor_uv[inbound, 1], floor_uv[inbound, 0]]
    max_depth = float(getattr(config, "max_depth", 20.0))
    valid_depth = inbound & torch.isfinite(depths) & (depths > 0.0) & (depths <= max_depth)
    result["depth_valid_matches"] = int(valid_depth.sum())
    if int(valid_depth.sum()) < 4:
        result["reject_code"] = "insufficient_orb_depth_points"
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    object_ned = pixel2point_NED(
        uv_h[valid_depth], depths[valid_depth],
        (historical.intrinsic[0] if historical.intrinsic.ndim == 3 else historical.intrinsic).float(),
    ).float()
    object_cv = object_ned.roll(shifts=-1, dims=-1).numpy().astype(np.float64)
    image = uv_c[valid_depth].numpy().astype(np.float64)
    if rng_applied:
        cv2.setRNGSeed(seed)
    try:
        succeeded, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_cv, image, K, np.empty((0, 1)), iterationsCount=100,
            reprojectionError=3.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as error:
        result["reject_code"] = "orb_pnp_exception"
        result["pnp"] = {"attempted": True, "succeeded": False, "error": str(error)}
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return result
    if not succeeded or inliers is None or len(inliers) == 0:
        result["reject_code"] = "orb_pnp_failed"
        result["pnp"] = {"attempted": True, "succeeded": False, "rng_seed": seed, "rng_seed_applied": rng_applied}
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
    projected, _ = cv2.projectPoints(object_cv[inliers], rvec, tvec, K, np.empty((0, 1)))
    errors = np.linalg.norm(projected.reshape(-1, 2) - image[inliers], axis=1)
    R_cv, _ = cv2.Rodrigues(rvec)
    transformed_cv = (R_cv @ object_cv[inliers].T).T + np.asarray(tvec).reshape(1, 3)
    positive_ratio = float(np.mean(transformed_cv[:, 2] > 0.0))
    inlier_uv = uv_c[valid_depth][torch.from_numpy(inliers)]
    coverage = _occupied_grid_cells(inlier_uv, current.width, current.height)
    count = int(len(inliers))
    ratio_value = count / max(int(valid_depth.sum()), 1)
    mean_error = float(np.mean(errors))
    gate_pass = count >= 20 and ratio_value >= 0.25 and coverage >= 8 and positive_ratio >= 0.90 and mean_error <= 3.0
    pose = _pose_from_opencv(rvec, tvec)
    result.update({
        "status": "accepted" if gate_pass else "rejected",
        "reject_code": None if gate_pass else "orb_pnp_gate_failed",
        "orb_gate_pass": gate_pass,
        "pnp": {
            "attempted": True, "succeeded": True, "input_points": int(valid_depth.sum()),
            "inliers": count, "inlier_ratio": ratio_value, "occupied_grid_cells": coverage,
            "positive_depth_ratio_inliers": positive_ratio, "mean_reprojection_error_px": mean_error,
            "rng_seed": seed, "rng_seed_applied": rng_applied,
            "T_current_candidate_orb": pose.tensor().detach().cpu().tolist(),
        },
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    })
    return result


@dataclass
class NMSSelection:
    candidate_uv: torch.Tensor
    current_uv: torch.Tensor
    depth: torch.Tensor
    covariance: torch.Tensor
    current_disparity: torch.Tensor | None
    current_disparity_covariance: torch.Tensor | None
    candidate_depth_covariance: torch.Tensor | None
    original_indices: torch.Tensor
    normalized_risk: torch.Tensor
    valid_mask: torch.Tensor


def flow_uncertainty_observe(
    match: IMatcher.Output,
    depth_current: IStereoDepth.Output,
    current: LoopFrameRecord,
    historical: LoopFrameRecord,
    pair_candidate_uv: torch.Tensor,
    pair_current_uv: torch.Tensor,
    pair_covariance: torch.Tensor | None,
    config: SimpleNamespace,
    point_risk_cap: float | None = None,
) -> tuple[dict[str, Any], NMSSelection | None]:
    result: dict[str, Any] = {
        "pair_statistics_mother_set": "sampled candidate points with inbound finite flow",
        "pair_sanity_pass": False, "pair_gate_pass": None, "nms": None,
    }
    if pair_covariance is None or match.cov is None:
        result["reject_code"] = "covariance_unavailable"
        return result, None

    risk, valid, psd = normalized_covariance_risk(pair_covariance, current.width, current.height)
    log_risk = torch.log(torch.clamp(risk, min=1e-12))
    raw_q = pair_covariance[0] + pair_covariance[1] - 2.0 * pair_covariance[2]
    normalized_q = (
        pair_covariance[0] / float(current.width * current.width)
        + pair_covariance[1] / float(current.height * current.height)
        - 2.0 * pair_covariance[2] / float(current.width * current.height)
    )
    valid_count = int(valid.sum())
    denominator = int(pair_candidate_uv.shape[0])
    valid_ratio = valid_count / max(denominator, 1)
    valid_uv = pair_current_uv[valid]
    coverage = _occupied_grid_cells(valid_uv, current.width, current.height)
    min_points = int(getattr(config, "min_valid_points", 30))
    min_ratio = float(getattr(config, "min_valid_ratio", 0.8))
    min_cells = int(getattr(config, "min_grid_cells", 8))
    sanity = valid_count >= min_points and valid_ratio >= min_ratio and coverage >= min_cells
    result.update({
        "covariance_layout_valid": True, "covariance_psd": psd,
        "pair_statistics_points": denominator, "covariance_valid_points": valid_count,
        "covariance_valid_ratio": valid_ratio, "covariance_valid_occupied_grid_cells": coverage,
        "normalized_lambda_max": _stats(risk[valid]), "log_normalized_lambda_max": _stats(log_risk[valid]),
        "frontend_q_raw": _stats(raw_q[valid]),
        "frontend_q_normalized_diagnostic": _stats(normalized_q[valid]),
        "pair_gate_statistic": "p50 of log normalized lambda-max risk",
        "pair_risk_p50": (
            None if valid_count == 0
            else float(torch.quantile(log_risk[valid].double(), 0.5, interpolation="linear"))
        ),
        "pair_sanity_pass": sanity,
    })

    dense_risk, dense_valid, dense_psd = normalized_covariance_risk(match.cov, current.width, current.height)
    q_raw = match.cov[:, 0:1] + match.cov[:, 1:2] - 2.0 * match.cov[:, 2:3]
    nms_mask = local_minimum_nms(q_raw, int(getattr(config, "nms_kernel_size", 7)), dense_valid.unsqueeze(0).unsqueeze(0))
    border = int(getattr(config, "border", 8))
    if border > 0:
        nms_mask[..., :border, :] = False
        nms_mask[..., -border:, :] = False
        nms_mask[..., :, :border] = False
        nms_mask[..., :, -border:] = False
    y, x = torch.nonzero(nms_mask[0, 0], as_tuple=True)
    candidate_uv = torch.stack([x, y], dim=1).float()
    original_indices = (y * current.width + x).to(torch.long)
    nms_risk = dense_risk[y, x]
    flow = IFrontend.retrieve_pixels(candidate_uv, match.flow)
    if flow is None:
        result["reject_code"] = "missing_flow"
        return result, None
    current_uv = candidate_uv + flow.T
    inbound = filterPointsInRange(current_uv, (0, current.width - 1), (0, current.height - 1))
    usable = inbound & torch.isfinite(flow).all(dim=0) & torch.isfinite(nms_risk)
    if match.mask is not None:
        sampled_mask = IFrontend.retrieve_pixels(candidate_uv, match.mask)
        if sampled_mask is not None:
            usable &= sampled_mask.squeeze(0).bool()
    depth = IFrontend.retrieve_pixels(candidate_uv, historical.depth.to(candidate_uv.device))
    if depth is None:
        result["reject_code"] = "missing_candidate_depth"
        return result, None
    depth = depth.squeeze(0)
    usable &= torch.isfinite(depth) & (depth > 0.0) & (depth <= float(getattr(config, "max_depth", 20.0)))
    if point_risk_cap is not None:
        usable &= nms_risk <= float(point_risk_cap)

    candidate_depth_covariance = IFrontend.retrieve_pixels(
        candidate_uv, None if historical.depth_covariance is None else historical.depth_covariance.to(candidate_uv.device)
    )
    if candidate_depth_covariance is not None:
        candidate_depth_covariance = candidate_depth_covariance.squeeze(0)
    # Preserve one entry per dense-NMS point while preventing out-of-range tensor
    # indexing.  The corresponding entries remain excluded by ``inbound`` above.
    safe_current_uv = current_uv.clone()
    safe_current_uv[:, 0].clamp_(0, current.width - 1)
    safe_current_uv[:, 1].clamp_(0, current.height - 1)
    current_disparity = IFrontend.retrieve_pixels(safe_current_uv, depth_current.disparity)
    if current_disparity is not None:
        current_disparity = current_disparity.squeeze(0)
    current_disparity_covariance = IFrontend.retrieve_pixels(
        safe_current_uv, depth_current.disparity_uncertainty
    )
    if current_disparity_covariance is not None:
        current_disparity_covariance = current_disparity_covariance.squeeze(0)

    result["nms"] = {
        "dense_psd": dense_psd, "nms_points_before_geometry": int(candidate_uv.shape[0]),
        "nms_points_after_geometry": int(usable.sum()),
        "normalized_risk": _stats(nms_risk[usable]),
        "q_raw": _stats(q_raw[0, 0, y[usable], x[usable]]),
        "point_risk_cap": point_risk_cap,
        "correspondence_count": int(usable.sum()),
        "correspondence_signature": tensor_signature(original_indices[usable]),
    }
    selection = NMSSelection(
        candidate_uv, current_uv, depth, IFrontend.retrieve_pixels(candidate_uv, match.cov),
        current_disparity, current_disparity_covariance, candidate_depth_covariance,
        original_indices, nms_risk, usable,
    )
    return result, selection


def spatial_selection_indices(
    selection: NMSSelection,
    current: LoopFrameRecord,
    config: SimpleNamespace,
) -> torch.Tensor:
    selected = torch.nonzero(selection.valid_mask, as_tuple=False).squeeze(1)
    grid_rows = int(getattr(config, "grid_rows", 8))
    grid_cols = int(getattr(config, "grid_cols", 8))
    per_cell = int(getattr(config, "max_points_per_cell", 20))
    spatially_selected: list[torch.Tensor] = []
    if selected.numel() > 0:
        # Match the frontend/candidate-depth sampling convention: spatial balance
        # is measured on the source (candidate) image where q-NMS is performed.
        uv = selection.candidate_uv[selected]
        cell_cols = torch.clamp((uv[:, 0] * grid_cols / max(current.width, 1)).long(), 0, grid_cols - 1)
        cell_rows = torch.clamp((uv[:, 1] * grid_rows / max(current.height, 1)).long(), 0, grid_rows - 1)
        cells = cell_rows * grid_cols + cell_cols
        for cell in range(grid_rows * grid_cols):
            local = selected[cells == cell]
            if local.numel() == 0:
                continue
            risks = selection.normalized_risk[local].detach().cpu().numpy()
            originals = selection.original_indices[local].detach().cpu().numpy()
            order = np.lexsort((originals, risks))[:per_cell]
            spatially_selected.append(local[torch.from_numpy(order).to(local.device)])
        selected = torch.cat(spatially_selected) if spatially_selected else selected[:0]
    max_points = int(getattr(config, "max_points", 800))
    if selected.numel() > max_points:
        risks = selection.normalized_risk[selected].detach().cpu().numpy()
        originals = selection.original_indices[selected].detach().cpu().numpy()
        order = np.lexsort((originals, risks))[:max_points]
        selected = selected[torch.from_numpy(order).to(selected.device)]
    # PnP always consumes points in their stable dense flattened-index order.
    return torch.sort(selected).values


def _run_flow_pnp(
    selection: NMSSelection,
    current: LoopFrameRecord,
    historical: LoopFrameRecord,
    config: SimpleNamespace,
) -> tuple[dict[str, Any], pp.LieTensor | None, torch.Tensor | None]:
    selected = spatial_selection_indices(selection, current, config)
    grid_rows = int(getattr(config, "grid_rows", 8))
    grid_cols = int(getattr(config, "grid_cols", 8))
    result: dict[str, Any] = {
        "pnp_attempted": False, "pnp_ransac_succeeded": False,
        "accepted": False, "input_points": int(selected.numel()),
        "occupied_grid_cells": _occupied_grid_cells(
            selection.current_uv[selected], current.width, current.height, grid_rows, grid_cols
        ),
        "correspondence_signature": tensor_signature(selection.original_indices[selected]),
    }
    min_points = int(getattr(config, "min_points", 30))
    if selected.numel() < min_points:
        result["reject_code"] = "insufficient_geometry_points"
        return result, None, None
    uv_h = selection.candidate_uv[selected]
    uv_c = selection.current_uv[selected]
    depths = selection.depth[selected]
    K_h = historical.intrinsic[0] if historical.intrinsic.ndim == 3 else historical.intrinsic
    points_ned = pixel2point_NED(uv_h, depths, K_h.to(uv_h.device).float()).float()
    points_cv = points_ned.roll(shifts=-1, dims=-1).detach().cpu().numpy().astype(np.float64)
    image = uv_c.detach().cpu().numpy().astype(np.float64)
    K = (current.intrinsic[0] if current.intrinsic.ndim == 3 else current.intrinsic).cpu().numpy().astype(np.float64)
    seed = _stable_seed(current.sensor_frame_idx, historical.sensor_frame_idx)
    rng_applied = hasattr(cv2, "setRNGSeed")
    if rng_applied:
        cv2.setRNGSeed(seed)
    result.update({"pnp_attempted": True, "rng_seed": seed, "rng_seed_applied": rng_applied})
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_cv, image, K, np.empty((0, 1)), iterationsCount=100,
        reprojectionError=3.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) == 0:
        result["reject_code"] = "pnp_failed"
        return result, None, None
    inliers_np = np.asarray(inliers, dtype=np.int64).reshape(-1)
    inlier_local = torch.from_numpy(inliers_np).to(selected.device)
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            points_cv[inliers_np], image[inliers_np], K, np.empty((0, 1)), rvec, tvec
        )
    projected, _ = cv2.projectPoints(points_cv[inliers_np], rvec, tvec, K, np.empty((0, 1)))
    error = float(np.linalg.norm(projected.reshape(-1, 2) - image[inliers_np], axis=1).mean())
    ratio = len(inliers_np) / max(len(selected), 1)
    accepted = len(inliers_np) >= 20 and ratio >= 0.25 and error <= 3.0
    result.update({
        "pnp_ransac_succeeded": True, "inliers": int(len(inliers_np)),
        "inlier_ratio": ratio, "mean_reprojection_error_px": error,
        "accepted": accepted, "reject_code": None if accepted else "pnp_gate_failed",
        "inlier_signature": tensor_signature(selection.original_indices[selected[inlier_local]]),
    })
    return result, _pose_from_opencv(rvec, tvec), selected[inlier_local]


def _skew(points: torch.Tensor) -> torch.Tensor:
    x, y, z = points.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(-1, 3, 3)


def _se3_exp(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    translation = delta[:3]
    omega = delta[3:]
    theta = torch.linalg.vector_norm(omega)
    W = _skew(omega.reshape(1, 3))[0]
    eye = torch.eye(3, dtype=delta.dtype, device=delta.device)
    if float(theta) < 1e-10:
        R = eye + W + 0.5 * (W @ W)
        V = eye + 0.5 * W + (W @ W) / 6.0
    else:
        R = eye + torch.sin(theta) / theta * W + (1.0 - torch.cos(theta)) / theta.square() * (W @ W)
        V = eye + (1.0 - torch.cos(theta)) / theta.square() * W + (theta - torch.sin(theta)) / theta.pow(3) * (W @ W)
    return R, V @ translation


def _pose_rt(pose: pp.LieTensor, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = pose.matrix().to(device=device, dtype=dtype)
    if matrix.ndim == 3:
        matrix = matrix[0]
    return matrix[:3, :3], matrix[:3, 3]


def transform_information_for_inverse(
    information: torch.Tensor, pose: pp.LieTensor
) -> torch.Tensor:
    """Convert left-tangent information on T to information on inverse(T)."""
    R, t = _pose_rt(pose, information.dtype, information.device)
    adjoint = torch.zeros((6, 6), dtype=information.dtype, device=information.device)
    adjoint[:3, :3] = R
    adjoint[:3, 3:] = _skew(t.reshape(1, 3))[0] @ R
    adjoint[3:, 3:] = R
    return adjoint.T @ information @ adjoint


def _candidate_point_covariance(
    uv: torch.Tensor, depth: torch.Tensor, depth_cov: torch.Tensor, K: torch.Tensor,
) -> torch.Tensor:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = uv[:, 0], uv[:, 1]
    J = torch.zeros((len(uv), 3, 3), dtype=uv.dtype, device=uv.device)
    J[:, 0, 2] = 1.0
    J[:, 1, 0] = depth / fx
    J[:, 1, 2] = (u - cx) / fx
    J[:, 2, 1] = depth / fy
    J[:, 2, 2] = (v - cy) / fy
    source = torch.zeros_like(J)
    source[:, 0, 0] = 1.0 / 12.0
    source[:, 1, 1] = 1.0 / 12.0
    source[:, 2, 2] = depth_cov
    return J @ source @ J.transpose(-1, -2)


def reproj_disp_linearization(
    pose: pp.LieTensor,
    candidate_points: torch.Tensor,
    candidate_covariance: torch.Tensor,
    current_uv: torch.Tensor,
    current_uv_covariance: torch.Tensor,
    current_disparity: torch.Tensor,
    current_disparity_covariance: torch.Tensor,
    K: torch.Tensor,
    baseline: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype, device = candidate_points.dtype, candidate_points.device
    R, t = _pose_rt(pose, dtype, device)
    transformed = (R @ candidate_points.T).T + t
    x, y, z = transformed.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    bf = baseline.reshape(-1)[0] * fx
    predicted = torch.stack([fx * y / x + cx, fy * z / x + cy, bf / x], dim=-1)
    observed = torch.cat([current_uv, current_disparity.unsqueeze(-1)], dim=-1)
    residual = predicted - observed
    J_proj = torch.zeros((len(x), 3, 3), dtype=dtype, device=device)
    J_proj[:, 0, 0] = -fx * y / x.square()
    J_proj[:, 0, 1] = fx / x
    J_proj[:, 1, 0] = -fy * z / x.square()
    J_proj[:, 1, 2] = fy / x
    J_proj[:, 2, 0] = -bf / x.square()
    J_motion = torch.cat([torch.eye(3, dtype=dtype, device=device).expand(len(x), -1, -1), -_skew(transformed)], dim=-1)
    jacobian = J_proj @ J_motion
    measurement = torch.zeros((len(x), 3, 3), dtype=dtype, device=device)
    measurement[:, :2, :2] = current_uv_covariance
    measurement[:, 2, 2] = current_disparity_covariance
    point_current_covariance = R.unsqueeze(0) @ candidate_covariance @ R.T.unsqueeze(0)
    residual_covariance = measurement + J_proj @ point_current_covariance @ J_proj.transpose(-1, -2)
    return residual, jacobian, residual_covariance, transformed


def _weighted_system(
    residual: torch.Tensor, jacobian: torch.Tensor, covariance: torch.Tensor,
    jitter_initial_scale: float = 1e-9, jitter_max_scale: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    mean_diag = float(torch.diagonal(covariance, dim1=-2, dim2=-1).mean().abs())
    base = max(mean_diag, 1e-12)
    jitter = jitter_initial_scale * base
    maximum = jitter_max_scale * base
    eye = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    attempts = 0
    while True:
        try:
            chol = torch.linalg.cholesky(covariance + jitter * eye)
            whitened_r = torch.linalg.solve_triangular(chol, residual.unsqueeze(-1), upper=False).squeeze(-1)
            whitened_j = torch.linalg.solve_triangular(chol, jacobian, upper=False)
            H = torch.einsum("nij,nik->jk", whitened_j, whitened_j)
            g = torch.einsum("nij,ni->j", whitened_j, whitened_r)
            cost = float(0.5 * whitened_r.square().sum())
            return H, g, cost, attempts
        except torch.linalg.LinAlgError:
            attempts += 1
            jitter *= 10.0
            if jitter > maximum:
                raise


def refine_and_information(
    selection: NMSSelection,
    inlier_indices: torch.Tensor,
    initial_pose: pp.LieTensor,
    current: LoopFrameRecord,
    historical: LoopFrameRecord,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "unavailable", "information_at_pnp_pose": None,
        "information_at_refined_pose": None,
        "lm_policy": {
            "max_iterations": 15, "damping_initial": 1e-3,
            "damping_min": 1e-9, "damping_max": 1e9,
            "reject_multiplier": 10.0, "accept_divisor": 3.0,
            "max_rejected_trials_per_iteration": 5,
            "max_total_rejected_trials": 30,
            "translation_step_tolerance_m": 1e-6,
            "rotation_step_tolerance_rad": 1e-6,
            "max_pose_jump_translation_m": 2.0,
            "max_pose_jump_rotation_deg": 10.0,
        },
        "covariance_jitter_policy": {
            "initial_scale_of_mean_diagonal": 1e-9,
            "multiplier": 10.0,
            "maximum_scale_of_mean_diagonal": 1e-3,
        },
    }
    if selection.current_disparity is None or selection.current_disparity_covariance is None or selection.candidate_depth_covariance is None:
        result["reason"] = "missing_disparity_or_depth_covariance"
        return result
    idx = inlier_indices
    uv_h, uv_c, depth = selection.candidate_uv[idx].double(), selection.current_uv[idx].double(), selection.depth[idx].double()
    flow_cov = selection.covariance[:, idx].T.double()
    current_uv_cov = torch.zeros((len(idx), 2, 2), dtype=torch.float64, device=uv_h.device)
    current_uv_cov[:, 0, 0] = flow_cov[:, 0]
    current_uv_cov[:, 1, 1] = flow_cov[:, 1]
    current_uv_cov[:, 0, 1] = current_uv_cov[:, 1, 0] = flow_cov[:, 2]
    disparity = selection.current_disparity[idx].double()
    disparity_cov = selection.current_disparity_covariance[idx].double()
    depth_cov = selection.candidate_depth_covariance[idx].double()
    candidate_K = (historical.intrinsic[0] if historical.intrinsic.ndim == 3 else historical.intrinsic).to(uv_h.device).double()
    current_K = (current.intrinsic[0] if current.intrinsic.ndim == 3 else current.intrinsic).to(uv_h.device).double()
    candidate_points = pixel2point_NED(uv_h, depth, candidate_K).double()
    candidate_cov = _candidate_point_covariance(uv_h, depth, depth_cov, candidate_K)
    finite = (
        torch.isfinite(candidate_points).all(dim=1) & torch.isfinite(candidate_cov).all(dim=(1, 2))
        & torch.isfinite(current_uv_cov).all(dim=(1, 2)) & torch.isfinite(disparity)
        & torch.isfinite(disparity_cov) & (disparity_cov >= 0.0)
        & torch.isfinite(depth_cov) & (depth_cov >= 0.0)
    )
    dropped = {"nonfinite_or_negative_covariance": int((~finite).sum())}
    if int(finite.sum()) < 4:
        result.update({"reason": "insufficient_valid_inliers", "dropped_reasons": dropped})
        return result
    uv_c, disparity, disparity_cov = uv_c[finite], disparity[finite], disparity_cov[finite]
    candidate_points, candidate_cov, current_uv_cov = candidate_points[finite], candidate_cov[finite], current_uv_cov[finite]
    original = selection.original_indices[idx][finite]
    point_signature = tensor_signature(original)

    def linearize(pose: pp.LieTensor):
        return reproj_disp_linearization(
            pose, candidate_points, candidate_cov, uv_c, current_uv_cov,
            disparity, disparity_cov, current_K, current.baseline.to(uv_h.device).double(),
        )

    residual, jacobian, covariance, transformed = linearize(initial_pose)
    positive = transformed[:, 0] > 1e-6
    dropped["nonpositive_transformed_depth_at_pnp"] = int((~positive).sum())
    if int(positive.sum()) < 4:
        result.update({"reason": "insufficient_positive_depth", "dropped_reasons": dropped})
        return result
    candidate_points, candidate_cov = candidate_points[positive], candidate_cov[positive]
    uv_c, current_uv_cov = uv_c[positive], current_uv_cov[positive]
    disparity, disparity_cov = disparity[positive], disparity_cov[positive]
    original = original[positive]
    point_signature = tensor_signature(original)
    residual, jacobian, covariance, _ = linearize(initial_pose)
    try:
        H_pnp, _, initial_cost, jitter_attempts = _weighted_system(residual, jacobian, covariance)
    except torch.linalg.LinAlgError:
        result.update({"reason": "covariance_factorization_failed", "dropped_reasons": dropped})
        return result

    def information_payload(H: torch.Tensor, pose: pp.LieTensor) -> dict[str, Any]:
        symmetric = 0.5 * (H + H.T)
        eigen = torch.linalg.eigvalsh(symmetric)
        positive_eigen = eigen[eigen > max(float(eigen.abs().max()) * 1e-10, 1e-12)]
        condition = None if positive_eigen.numel() == 0 else float(positive_eigen.max() / positive_eigen.min())
        inverse_pose_information = transform_information_for_inverse(symmetric, pose)
        trace = float(torch.trace(symmetric))
        return {
            "matrix": symmetric.detach().cpu().tolist(), "point_count": int(len(original)),
            "point_signature": point_signature, "eigenvalues": eigen.detach().cpu().tolist(),
            "rank": int(torch.linalg.matrix_rank(symmetric)), "condition_number_positive_subspace": condition,
            "matrix_per_point": (symmetric / max(len(original), 1)).detach().cpu().tolist(),
            "matrix_trace_normalized": (
                None if abs(trace) <= 1e-12 else (symmetric / trace).detach().cpu().tolist()
            ),
            "inverse_pose_matrix_via_adjoint": inverse_pose_information.detach().cpu().tolist(),
            "inverse_transform_rule": "Lambda_inverse = Ad_T^T Lambda Ad_T for left [t,r] tangent",
            "normalization": "raw_hessian_no_lm_damping",
        }

    result["information_at_pnp_pose"] = information_payload(H_pnp, initial_pose)
    R, t = _pose_rt(initial_pose, torch.float64, uv_h.device)
    damping = 1e-3
    rejected_total = 0
    accepted_steps = 0
    converged = False
    current_cost = initial_cost
    max_translation_step = 0.0
    max_rotation_step = 0.0
    for _ in range(15):
        residual, jacobian, covariance, _ = linearize(pp.from_matrix(torch.cat([torch.cat([R, t[:, None]], dim=1), torch.tensor([[0., 0., 0., 1.]], device=R.device, dtype=R.dtype)], dim=0), pp.SE3_type))
        H, g, _, _ = _weighted_system(residual, jacobian, covariance)
        accepted = False
        for _trial in range(5):
            if rejected_total >= 30:
                break
            try:
                delta = torch.linalg.solve(H + damping * torch.eye(6, dtype=H.dtype, device=H.device), -g)
            except torch.linalg.LinAlgError:
                damping = min(damping * 10.0, 1e9)
                rejected_total += 1
                continue
            translation_step = float(torch.linalg.vector_norm(delta[:3]))
            rotation_step = float(torch.linalg.vector_norm(delta[3:]))
            max_translation_step = max(max_translation_step, translation_step)
            max_rotation_step = max(max_rotation_step, rotation_step)
            if translation_step <= 1e-6 and rotation_step <= 1e-6:
                converged = True
                accepted = True
                break
            dR, dt = _se3_exp(delta)
            trial_R, trial_t = dR @ R, dR @ t + dt
            trial_pose = pp.from_matrix(torch.cat([torch.cat([trial_R, trial_t[:, None]], dim=1), torch.tensor([[0., 0., 0., 1.]], device=R.device, dtype=R.dtype)], dim=0), pp.SE3_type)
            trial_residual, trial_jacobian, trial_covariance, _ = linearize(trial_pose)
            _, _, trial_cost, _ = _weighted_system(trial_residual, trial_jacobian, trial_covariance)
            if trial_cost < current_cost:
                R, t, current_cost = trial_R, trial_t, trial_cost
                damping = max(damping / 3.0, 1e-9)
                accepted_steps += 1
                converged = translation_step <= 1e-6 and rotation_step <= 1e-6
                accepted = True
                break
            damping = min(damping * 10.0, 1e9)
            rejected_total += 1
        if converged or not accepted or rejected_total >= 30:
            break
    refined_pose = pp.from_matrix(torch.cat([torch.cat([R, t[:, None]], dim=1), torch.tensor([[0., 0., 0., 1.]], device=R.device, dtype=R.dtype)], dim=0), pp.SE3_type)
    relative_delta = (refined_pose @ initial_pose.Inv()).Log().tensor().double()
    jump_translation = float(torch.linalg.vector_norm(relative_delta[..., :3]))
    jump_rotation_deg = float(torch.linalg.vector_norm(relative_delta[..., 3:]) * 180.0 / math.pi)
    if jump_translation > 2.0 or jump_rotation_deg > 10.0:
        result.update({"reason": "refinement_pose_jump", "dropped_reasons": dropped})
        return result
    final_residual, final_jacobian, final_covariance, final_points = linearize(refined_pose)
    final_positive = final_points[:, 0] > 1e-6
    dropped["nonpositive_transformed_depth_at_refined"] = int((~final_positive).sum())
    if int(final_positive.sum()) < 4:
        result.update({"reason": "insufficient_refined_points", "dropped_reasons": dropped})
        return result
    H_refined, _, final_cost, final_jitter = _weighted_system(
        final_residual[final_positive], final_jacobian[final_positive], final_covariance[final_positive]
    )
    final_original = original[final_positive]
    original_saved, signature_saved = original, point_signature
    original = final_original
    point_signature = tensor_signature(final_original)
    result["information_at_refined_pose"] = information_payload(H_refined, refined_pose)
    original, point_signature = original_saved, signature_saved
    result.update({
        "status": "succeeded", "reason": None, "converged": converged,
        "initial_cost": initial_cost, "final_cost": final_cost,
        "accepted_steps": accepted_steps, "rejected_trials": rejected_total,
        "max_translation_step_m": max_translation_step,
        "max_rotation_step_rad": max_rotation_step,
        "pose_jump_translation_m": jump_translation,
        "pose_jump_rotation_deg": jump_rotation_deg,
        "jitter_attempts_at_pnp": jitter_attempts, "jitter_attempts_at_refined": final_jitter,
        "refined_T_current_candidate": refined_pose.tensor().detach().cpu().tolist(),
        "dropped_reasons": dropped,
    })
    return result


class PhaseB5Analyzer:
    def __init__(
        self, config: SimpleNamespace, queries: list[dict[str, Any]],
        records: list[dict[str, Any]] | None = None,
        record_root: Path | None = None,
    ) -> None:
        self.config = config
        self.mode = str(getattr(config, "mode", "disabled"))
        self.enabled = bool(getattr(config, "enabled", False)) and self.mode != "disabled"
        valid_queries = sorted(
            [q for q in queries if q.get("candidates") and all(k in q for k in ("sensor_frame_idx", "loop_frame_idx"))],
            key=lambda q: int(q["sensor_frame_idx"]),
        )
        fraction = float(getattr(getattr(config, "calibration", SimpleNamespace()), "prefix_fraction", 0.2))
        prefix_count = int(math.ceil(len(valid_queries) * fraction)) if valid_queries else 0
        self.calibration_end_sensor_frame_idx = (
            int(valid_queries[prefix_count - 1]["sensor_frame_idx"]) if prefix_count else None
        )
        self.valid_query_count = len(valid_queries)
        self.prefix_query_count = prefix_count
        self.rows: list[dict[str, Any]] = []
        self.point_risks: dict[str, list[tuple[float, torch.Tensor]]] = {
            "all_bow_candidates": [], "orb_supported_candidates": []
        }
        self.input_digest = canonical_sha256([
            [
                int(q["sensor_frame_idx"]), int(q.get("visual_map_idx", -1)),
                int(q["loop_frame_idx"]),
                [
                    [
                        int(c["sensor_frame_idx"]), int(c.get("visual_map_idx", -1)),
                        int(c["loop_frame_idx"]), float(c.get("score", 0.0)),
                    ]
                    for c in q.get("candidates", [])
                ],
            ]
            for q in valid_queries
        ])
        self.cache_index_digest = canonical_sha256([
            [int(item["sensor_frame_idx"]), int(item["loop_frame_idx"]), str(item["file"])]
            for item in sorted(records or [], key=lambda value: int(value["sensor_frame_idx"]))
        ])
        self.cache_content_digest = cache_content_sha256(records or [], record_root)
        orb_cfg = getattr(config, "orb", SimpleNamespace())
        self.orb_config_sha256 = canonical_sha256(vars(orb_cfg))
        self.phase_b5_config_sha256 = canonical_sha256({
            "calibration": vars(getattr(config, "calibration", SimpleNamespace())),
            "orb": vars(orb_cfg),
            "flow": vars(getattr(config, "flow", SimpleNamespace())),
        })
        self.manifest = self._load_manifest(getattr(config, "trusted_manifest", None))
        if self.mode == "apply" and self.manifest is None:
            raise ValueError("Phase B.5 apply requires a promoted trusted manifest")

    def _load_manifest(self, path_value: str | None) -> dict[str, Any] | None:
        if not path_value:
            return None
        path = Path(path_value)
        def reject_nonfinite(value: str) -> None:
            raise ValueError(f"non-finite JSON token {value!r} in {path}")

        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
        )
        if payload.get("schema_version") != PHASE_B5_SCHEMA_VERSION:
            raise ValueError("unsupported Phase B.5 manifest schema")
        if payload.get("input_digest") != self.input_digest:
            raise ValueError("Phase B.5 manifest input digest does not match this run")
        if payload.get("calibration_end_sensor_frame_idx") != self.calibration_end_sensor_frame_idx:
            raise ValueError("Phase B.5 manifest calibration boundary does not match this run")
        calibration_cfg = getattr(self.config, "calibration", SimpleNamespace())
        if payload.get("prefix_fraction") != float(getattr(calibration_cfg, "prefix_fraction", 0.2)):
            raise ValueError("Phase B.5 manifest calibration prefix fraction does not match this run")
        expected_exclusion = {"query_sensor_frame_idx_lte": self.calibration_end_sensor_frame_idx}
        if payload.get("evaluation_exclusion") != expected_exclusion:
            raise ValueError("Phase B.5 manifest evaluation exclusion does not match this run")
        if payload.get("orb_config_sha256") != self.orb_config_sha256:
            raise ValueError("Phase B.5 manifest ORB configuration does not match this run")
        if payload.get("phase_b5_config_sha256") != self.phase_b5_config_sha256:
            raise ValueError("Phase B.5 manifest configuration does not match this run")
        if payload.get("cache_index_digest") != self.cache_index_digest:
            raise ValueError("Phase B.5 manifest cache index does not match this run")
        if payload.get("cache_content_digest") != self.cache_content_digest:
            raise ValueError("Phase B.5 manifest cache content does not match this run")
        if self.mode == "apply":
            if payload.get("development_evaluation_pass") is not True or payload.get("frozen_evaluation_pass") is not True:
                raise ValueError("Phase B.5 apply requires passed development and frozen evaluations")
            promoted_population = payload.get("promoted_population")
            promoted = (payload.get("populations") or {}).get(promoted_population)
            if (
                promoted_population not in {"all_bow_candidates", "orb_supported_candidates"}
                or not isinstance(promoted, dict)
                or promoted.get("population") != promoted_population
                or promoted.get("trusted") is not True
                or promoted.get("pair_gate_promoted") is not True
                or promoted.get("selector_promoted") is not True
            ):
                raise ValueError("Phase B.5 apply requires a promoted Flow pair gate and selector")
            if (
                promoted_population == "orb_supported_candidates"
                and payload.get("orb_promoted") is not True
            ):
                raise ValueError("ORB-supported Phase B.5 apply requires a promoted ORB gate")
        return payload

    def manifest_population(self, population: str) -> dict[str, Any] | None:
        if self.manifest is None:
            return None
        item = (self.manifest.get("populations") or {}).get(population)
        if not isinstance(item, dict) or item.get("population") != population:
            return None
        return item

    def observe_orb(
        self, current: LoopFrameRecord, historical: LoopFrameRecord
    ) -> dict[str, Any]:
        return orb_geometry_observe(
            current, historical, getattr(self.config, "orb", SimpleNamespace())
        )

    def _in_calibration(self, query: dict[str, Any]) -> bool:
        return (
            self.calibration_end_sensor_frame_idx is not None
            and int(query["sensor_frame_idx"]) <= self.calibration_end_sensor_frame_idx
        )

    def is_in_calibration(self, query: dict[str, Any]) -> bool:
        return self._in_calibration(query)

    def record_flow_failure(
        self, query: dict[str, Any], candidate: dict[str, Any],
        orb: dict[str, Any], reject_code: str, reject_reason: str,
    ) -> dict[str, Any]:
        populations = ["all_bow_candidates"] + (
            ["orb_supported_candidates"] if orb.get("orb_gate_pass") else []
        )
        row = {
            "pair_id": f"{int(query['loop_frame_idx'])}:{int(candidate['loop_frame_idx'])}",
            "current_sensor_frame_idx": int(query["sensor_frame_idx"]),
            "candidate_sensor_frame_idx": int(candidate["sensor_frame_idx"]),
            "bow_score": float(candidate.get("score", 0.0)),
            "in_calibration_prefix": self._in_calibration(query),
            "orb": orb,
            "flow": {
                "pair_sanity_pass": False, "pair_gate_pass": None,
                "reject_code": reject_code, "reject_reason": reject_reason,
            },
            "populations": populations,
        }
        self.rows.append(row)
        return row

    def observe(
        self,
        query: dict[str, Any], candidate: dict[str, Any], current: LoopFrameRecord,
        historical: LoopFrameRecord, match: IMatcher.Output, depth_current: IStereoDepth.Output,
        pair_candidate_uv: torch.Tensor, pair_current_uv: torch.Tensor,
        pair_covariance: torch.Tensor | None,
        orb_result: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], NMSSelection | None]:
        pair_id = f"{int(query['loop_frame_idx'])}:{int(candidate['loop_frame_idx'])}"
        orb = orb_result if orb_result is not None else self.observe_orb(current, historical)
        flow, selection = flow_uncertainty_observe(
            match, depth_current, current, historical, pair_candidate_uv, pair_current_uv,
            pair_covariance, getattr(self.config, "flow", SimpleNamespace()), None,
        )
        populations = ["all_bow_candidates"] + (["orb_supported_candidates"] if orb["orb_gate_pass"] else [])
        pair_gate_by_population: dict[str, bool | None] = {}
        point_cap_by_population: dict[str, float | None] = {}
        for population in populations:
            manifest_item = self.manifest_population(population)
            pair_threshold = None if manifest_item is None else manifest_item.get("pair_threshold")
            calibration_trusted = bool(manifest_item and manifest_item.get("trusted") is True)
            pair_gate_by_population[population] = (
                None if pair_threshold is None
                else bool(
                    flow.get("pair_sanity_pass")
                    and calibration_trusted
                    and float(flow["pair_risk_p50"]) <= float(pair_threshold)
                )
            )
            point_cap_by_population[population] = (
                None if manifest_item is None else manifest_item.get("point_risk_cap")
            )
        flow["pair_gate_by_population"] = pair_gate_by_population
        flow["point_risk_cap_by_population"] = point_cap_by_population
        flow["pair_gate_pass"] = pair_gate_by_population.get(
            "orb_supported_candidates" if orb["orb_gate_pass"] else "all_bow_candidates"
        )
        in_calibration = self._in_calibration(query)
        row = {
            "pair_id": pair_id,
            "current_sensor_frame_idx": int(query["sensor_frame_idx"]),
            "candidate_sensor_frame_idx": int(candidate["sensor_frame_idx"]),
            "bow_score": float(candidate.get("score", 0.0)),
            "in_calibration_prefix": in_calibration,
            "orb": orb, "flow": flow,
            "populations": populations,
        }
        self.rows.append(row)
        if in_calibration and selection is not None and flow.get("pair_sanity_pass"):
            risks = selection.normalized_risk[selection.valid_mask].detach().cpu().float()
            pair_risk = float(flow["pair_risk_p50"])
            self.point_risks["all_bow_candidates"].append((pair_risk, risks))
            if orb["orb_gate_pass"]:
                self.point_risks["orb_supported_candidates"].append((pair_risk, risks))
        return row, selection

    def calibration_manifest(self) -> dict[str, Any]:
        calibration_cfg = getattr(self.config, "calibration", SimpleNamespace())
        minimum_queries = int(getattr(calibration_cfg, "min_queries", 20))
        populations: dict[str, Any] = {}
        for name, minimum_pairs in (
            ("all_bow_candidates", int(getattr(calibration_cfg, "min_all_bow_pairs", 100))),
            ("orb_supported_candidates", int(getattr(calibration_cfg, "min_orb_pairs", 30))),
        ):
            rows = [
                row for row in self.rows if row["in_calibration_prefix"] and name in row["populations"]
                and row["flow"].get("pair_sanity_pass") and row["flow"].get("pair_risk_p50") is not None
            ]
            values = torch.tensor([row["flow"]["pair_risk_p50"] for row in rows], dtype=torch.float64)
            median = float(torch.quantile(values, 0.5, interpolation="linear")) if values.numel() else None
            mad = (
                float(torch.quantile((values - median).abs(), 0.5, interpolation="linear"))
                if values.numel() else None
            )
            scaled_mad = None if mad is None else 1.4826 * mad
            threshold = None if median is None else median + 2.0 * scaled_mad
            risk_parts = self.point_risks[name]
            point_values = torch.cat([part for _, part in risk_parts]) if risk_parts else torch.empty(0)
            gate_point_values = torch.cat(
                [part for pair_risk, part in risk_parts if threshold is not None and pair_risk <= threshold]
            ) if any(threshold is not None and pair_risk <= threshold for pair_risk, _ in risk_parts) else torch.empty(0)
            gate_point_pairs = sum(
                threshold is not None and pair_risk <= threshold for pair_risk, _ in risk_parts
            )
            pair_enough = self.prefix_query_count >= minimum_queries and len(rows) >= minimum_pairs
            sanity_point_enough = len(risk_parts) >= 20 and point_values.numel() >= 1000
            gate_point_enough = gate_point_pairs >= 20 and gate_point_values.numel() >= 1000
            observed_point_cap = (
                float(torch.quantile(point_values, 0.95, interpolation="linear"))
                if point_values.numel() else None
            )
            observed_gate_point_cap = (
                float(torch.quantile(gate_point_values, 0.95, interpolation="linear"))
                if gate_point_values.numel() else None
            )
            point_cap = observed_point_cap if sanity_point_enough else None
            gate_point_cap = observed_gate_point_cap if gate_point_enough else None
            median_cap = getattr(calibration_cfg, "absolute_median_log_risk_cap", None)
            q95_cap = getattr(calibration_cfg, "absolute_q95_log_risk_cap", None)
            q95 = float(torch.quantile(values, 0.95, interpolation="linear")) if values.numel() else None
            absolute_configured = median_cap is not None and q95_cap is not None
            absolute_pass = bool(
                absolute_configured and median is not None and q95 is not None
                and median <= float(median_cap) and q95 <= float(q95_cap)
            )
            populations[name] = {
                "population": name, "calibration_pairs": len(rows),
                "pair_risk_median": median, "pair_risk_q95": q95,
                "pair_risk_mad": mad, "scaled_mad": scaled_mad,
                "pair_threshold": threshold,
                "point_risk_cap": point_cap,
                "point_risk_cap_sanity_population": point_cap,
                "point_risk_cap_pair_gate_population": gate_point_cap,
                "point_risk_cap_observed": observed_point_cap,
                "point_risk_cap_pair_gate_observed": observed_gate_point_cap,
                "point_risk_count": int(point_values.numel()),
                "point_risk_pairs_sanity_population": len(risk_parts),
                "point_risk_count_pair_gate_population": int(gate_point_values.numel()),
                "point_risk_pairs_pair_gate_population": int(gate_point_pairs),
                "calibration_sufficient": pair_enough,
                "point_cap_calibration_sufficient": sanity_point_enough,
                "point_cap_pair_gate_calibration_sufficient": gate_point_enough,
                "point_cap_reason": (
                    None if sanity_point_enough else "point_cap_calibration_insufficient"
                ),
                "absolute_sanity_configured": absolute_configured,
                "absolute_sanity_pass": absolute_pass,
                "trusted": bool(pair_enough and absolute_pass),
                "pair_gate_promoted": False,
                "selector_promoted": False,
                "reason": None if pair_enough and absolute_pass else (
                    "calibration_insufficient" if not pair_enough else "calibration_untrusted"
                ),
            }
        return {
            "schema_version": PHASE_B5_SCHEMA_VERSION,
            "calibration_scope": "offline_prefix_fraction",
            "prefix_fraction": float(getattr(calibration_cfg, "prefix_fraction", 0.2)),
            "valid_query_count": self.valid_query_count,
            "prefix_query_count": self.prefix_query_count,
            "calibration_end_sensor_frame_idx": self.calibration_end_sensor_frame_idx,
            "evaluation_exclusion": {
                "query_sensor_frame_idx_lte": self.calibration_end_sensor_frame_idx,
            },
            "input_digest": self.input_digest,
            "cache_index_digest": self.cache_index_digest,
            "cache_content_digest": self.cache_content_digest,
            "orb_config_sha256": self.orb_config_sha256,
            "phase_b5_config_sha256": self.phase_b5_config_sha256,
            "threshold_rule": "median(pair_log_risk_p50)+2*1.4826*MAD; MAD=0 uses median",
            "point_cap_rule": "Q95 normalized lambda-max risk on eligible calibration NMS points",
            "absolute_sanity_caps": {
                "median_log_risk": getattr(calibration_cfg, "absolute_median_log_risk_cap", None),
                "q95_log_risk": getattr(calibration_cfg, "absolute_q95_log_risk_cap", None),
            },
            "development_evaluation_pass": False,
            "frozen_evaluation_pass": False,
            "orb_promoted": False,
            "populations": populations,
        }

    def branch_payloads(self) -> dict[str, dict[str, Any]]:
        common = {
            "schema_version": PHASE_B5_SCHEMA_VERSION,
            "mode": self.mode,
            "calibration_end_sensor_frame_idx": self.calibration_end_sensor_frame_idx,
            "input_digest": self.input_digest,
            "cache_index_digest": self.cache_index_digest,
            "cache_content_digest": self.cache_content_digest,
            "phase_b5_config_sha256": self.phase_b5_config_sha256,
        }
        orb_rows = [{k: row[k] for k in ("pair_id", "current_sensor_frame_idx", "candidate_sensor_frame_idx", "in_calibration_prefix")} | {"orb": row["orb"]} for row in self.rows]
        all_rows = [row for row in self.rows if "all_bow_candidates" in row["populations"]]
        supported = [row for row in self.rows if "orb_supported_candidates" in row["populations"]]
        forced = [row for row in self.rows if (row["current_sensor_frame_idx"], row["candidate_sensor_frame_idx"]) in {(1200, 1100), (1250, 470), (1250, 480)}]
        return {
            "orb_observe/verification.json": common | {"branch_id": "orb_observe", "rows": orb_rows},
            "flow_all_bow_observe/verification.json": common | {"branch_id": "flow_all_bow_observe", "rows": all_rows},
            "flow_orb_supported_observe/verification.json": common | {"branch_id": "flow_orb_supported_observe", "rows": supported},
            "forced_control_shadow/verification.json": common | {
                "branch_id": "forced_control_shadow",
                "diagnostic_overhead_ms": 0.0,
                "diagnostic_overhead_semantics": (
                    "controls are views of already-processed BoW candidates; no extra Frontend inference"
                ),
                "rows": forced,
            },
        }
