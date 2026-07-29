from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pypose as pp
import torch

from Module.LoopClosure.PhaseB5 import (
    reproj_disp_linearization,
    transform_information_for_inverse,
)
from Module.Map import VisualMap


@dataclass
class ObservationSystem:
    hessian: torch.Tensor
    gradient: torch.Tensor
    robust_cost: float
    residual: torch.Tensor
    jacobian: torch.Tensor
    covariance: torch.Tensor
    transformed: torch.Tensor
    weights: torch.Tensor
    whitened_jacobian: torch.Tensor


def covariance_triplet_to_matrix(value: torch.Tensor) -> torch.Tensor:
    triplet = value.double()
    result = torch.zeros((len(triplet), 2, 2), dtype=triplet.dtype, device=triplet.device)
    result[:, 0, 0] = triplet[:, 0]
    result[:, 1, 1] = triplet[:, 1]
    result[:, 0, 1] = result[:, 1, 0] = triplet[:, 2]
    return result


def robust_observation_system(
    pose_current_candidate: pp.LieTensor,
    candidate_points: torch.Tensor,
    candidate_covariance: torch.Tensor,
    current_uv: torch.Tensor,
    current_uv_covariance: torch.Tensor,
    current_disparity: torch.Tensor,
    current_disparity_variance: torch.Tensor,
    intrinsic: torch.Tensor,
    baseline: torch.Tensor,
    huber_delta: float,
) -> ObservationSystem:
    residual, jacobian, covariance, transformed = reproj_disp_linearization(
        pose_current_candidate,
        candidate_points,
        candidate_covariance,
        current_uv,
        current_uv_covariance,
        current_disparity,
        current_disparity_variance,
        intrinsic,
        baseline,
    )
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    if not all(
        torch.isfinite(value).all()
        for value in (residual, jacobian, covariance, transformed)
    ):
        raise ValueError("nonfinite_observation_linearization")
    cholesky, info = torch.linalg.cholesky_ex(covariance)
    if bool((info != 0).any()):
        raise ValueError("residual_covariance_not_spd")
    whitened_r = torch.linalg.solve_triangular(
        cholesky, residual.unsqueeze(-1), upper=False,
    ).squeeze(-1)
    whitened_j = torch.linalg.solve_triangular(cholesky, jacobian, upper=False)
    norm = torch.linalg.vector_norm(whitened_r, dim=-1)
    delta = torch.as_tensor(float(huber_delta), dtype=norm.dtype, device=norm.device)
    weights = torch.where(
        norm <= delta,
        torch.ones_like(norm),
        delta / norm.clamp_min(1e-12),
    )
    weighted_j = whitened_j * weights.sqrt().reshape(-1, 1, 1)
    weighted_r = whitened_r * weights.sqrt().reshape(-1, 1)
    hessian = torch.einsum("nij,nik->jk", weighted_j, weighted_j)
    gradient = torch.einsum("nij,ni->j", weighted_j, weighted_r)
    quadratic = 0.5 * norm.square()
    robust = torch.where(
        norm <= delta,
        quadratic,
        delta * (norm - 0.5 * delta),
    )
    return ObservationSystem(
        hessian=0.5 * (hessian + hessian.T),
        gradient=gradient,
        robust_cost=float(robust.sum()),
        residual=residual,
        jacobian=jacobian,
        covariance=covariance,
        transformed=transformed,
        weights=weights,
        whitened_jacobian=weighted_j,
    )


def information_diagnostics(
    system: ObservationSystem,
    pose_current_candidate: pp.LieTensor,
    *,
    point_count: int,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    hessian = 0.5 * (system.hessian + system.hessian.T)
    singular = torch.linalg.svdvals(system.whitened_jacobian.reshape(-1, 6))
    threshold = (
        max(float(singular.max()) * 1e-8, 1e-12)
        if singular.numel() else 1e-12
    )
    rank = int((singular > threshold).sum())
    eigen = torch.linalg.eigvalsh(hessian)
    positive = eigen[eigen > max(float(eigen.abs().max()) * 1e-10, 1e-12)]
    condition = (
        None if positive.numel() == 0
        else float(positive.max() / positive.min())
    )
    trace = float(torch.trace(hessian))
    payload: dict[str, Any] = {
        "valid": False,
        "point_count": int(point_count),
        "rank": rank,
        "rank_threshold": threshold,
        "raw_eigenvalues": eigen.detach().cpu().tolist(),
        "trace": trace,
        "trace_per_point": trace / max(int(point_count), 1),
        "condition_number": condition,
        "normalization": "raw_robust_hessian_no_lm_damping",
        "cross_covariance_mode": "ignored_independent_approximation",
        "huber_weight_derivative_included": False,
    }
    tolerance = max(float(eigen.abs().max()) * 1e-9, 1e-12)
    if (
        rank != 6
        or not torch.isfinite(hessian).all()
        or float(eigen.min()) < -tolerance
    ):
        payload["reason"] = "information_rank_or_psd"
        return None, payload
    _, hessian_cholesky_info = torch.linalg.cholesky_ex(hessian)
    if int(hessian_cholesky_info) != 0:
        payload["reason"] = "information_not_strictly_spd"
        return None, payload
    edge_information = transform_information_for_inverse(
        hessian, pose_current_candidate,
    )
    edge_information = 0.5 * (edge_information + edge_information.T)
    edge_eigen = torch.linalg.eigvalsh(edge_information)
    if not torch.isfinite(edge_information).all() or float(edge_eigen.min()) < -tolerance:
        payload["reason"] = "edge_information_invalid"
        return None, payload
    _, edge_cholesky_info = torch.linalg.cholesky_ex(edge_information)
    if int(edge_cholesky_info) != 0:
        payload["reason"] = "edge_information_not_strictly_spd"
        return None, payload
    payload.update({
        "valid": True,
        "reason": None,
        "pose_direction": "T_candidate_current",
        "used_matrix": edge_information.detach().cpu().tolist(),
        "edge_eigenvalues": edge_eigen.detach().cpu().tolist(),
    })
    return edge_information, payload


def odometry_edge_information(
    global_map: VisualMap,
    src: int,
    dst: int,
    *,
    huber_delta: float,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "src": int(src),
        "dst": int(dst),
        "mode": "direct_match_observation_hessian",
        "valid": False,
    }
    frame = global_map.frames[torch.tensor([int(dst)], dtype=torch.long)]
    observations = global_map.get_frame2match(frame)
    if len(observations) == 0:
        diagnostics["reason"] = "no_frame2match_observations"
        return None, diagnostics
    frame1 = global_map.match2frame1.project(observations.index).reshape(-1)
    frame2 = global_map.match2frame2.project(observations.index).reshape(-1)
    direct = (frame1 == int(src)) & (frame2 == int(dst))
    if not bool(direct.any()):
        diagnostics["reason"] = "no_direct_match_observations"
        return None, diagnostics
    observations = observations[direct]
    points = global_map.get_match2point(observations)
    poses = pp.SE3(global_map.frames.data["pose"].tensor.double())
    T_w_candidate = poses[int(src)]
    T_w_current = poses[int(dst)]
    pose_current_candidate = T_w_current.Inv() @ T_w_candidate
    candidate_points = T_w_candidate.Inv().Act(points.data["pos_Tw"].double())
    rotation = T_w_candidate.rotation().matrix().double()
    candidate_covariance = (
        rotation.T.unsqueeze(0)
        @ points.data["cov_Tw"].double()
        @ rotation.unsqueeze(0)
    )
    current_uv = observations.data["pixel2_uv"].double()
    current_uv_covariance = covariance_triplet_to_matrix(
        observations.data["pixel2_uv_cov"],
    )
    current_disparity = observations.data["pixel2_disp"].reshape(-1).double()
    current_disparity_variance = (
        observations.data["pixel2_disp_cov"].reshape(-1).double()
    )
    finite = (
        torch.isfinite(candidate_points).all(dim=-1)
        & torch.isfinite(candidate_covariance).all(dim=(-1, -2))
        & torch.isfinite(current_uv).all(dim=-1)
        & torch.isfinite(current_uv_covariance).all(dim=(-1, -2))
        & torch.isfinite(current_disparity)
        & torch.isfinite(current_disparity_variance)
        & (current_disparity > 0.0)
        & (current_disparity_variance > 0.0)
    )
    diagnostics["direct_observation_count"] = int(len(finite))
    diagnostics["valid_observation_count"] = int(finite.sum())
    if int(finite.sum()) < 6:
        diagnostics["reason"] = "insufficient_valid_direct_observations"
        return None, diagnostics
    candidate_points = candidate_points[finite]
    candidate_covariance = candidate_covariance[finite]
    current_uv = current_uv[finite]
    current_uv_covariance = current_uv_covariance[finite]
    current_disparity = current_disparity[finite]
    current_disparity_variance = current_disparity_variance[finite]
    intrinsic = global_map.frames.data["K"][int(dst)].double()
    baseline = global_map.frames.data["baseline"][int(dst)].double()
    try:
        system = robust_observation_system(
            pose_current_candidate,
            candidate_points,
            candidate_covariance,
            current_uv,
            current_uv_covariance,
            current_disparity,
            current_disparity_variance,
            intrinsic,
            baseline,
            huber_delta,
        )
    except ValueError as error:
        diagnostics["reason"] = str(error)
        return None, diagnostics
    positive = system.transformed[:, 0] > 1e-6
    diagnostics["positive_depth_count"] = int(positive.sum())
    if int(positive.sum()) < 6:
        diagnostics["reason"] = "insufficient_positive_direct_observations"
        return None, diagnostics
    if not bool(positive.all()):
        system = robust_observation_system(
            pose_current_candidate,
            candidate_points[positive],
            candidate_covariance[positive],
            current_uv[positive],
            current_uv_covariance[positive],
            current_disparity[positive],
            current_disparity_variance[positive],
            intrinsic,
            baseline,
            huber_delta,
        )
    information, payload = information_diagnostics(
        system,
        pose_current_candidate,
        point_count=int(positive.sum()),
    )
    diagnostics.update(payload)
    return information, diagnostics
