from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pypose as pp
import torch

from Module.Frontend.Frontend import IFrontend
from Module.Optimization.GlobalPGO import make_information
from Utility.Point import filterPointsInRange, pixel2point_NED

from .Record import LoopFrameRecord
from .PhaseB5 import (
    NMSSelection,
    PhaseB5Analyzer,
    _run_flow_pnp,
    refine_and_information,
    spatial_selection_indices,
)


@dataclass(frozen=True)
class LoopConstraint:
    src_visual_map_idx: int
    dst_visual_map_idx: int
    src_sensor_frame_idx: int
    dst_sensor_frame_idx: int
    pnp_relative_pose: list[float]
    relative_pose: list[float]
    information: list[list[float]]
    bow_score: float
    num_flow_points: int
    num_geometry_points: int
    num_pnp_inliers: int
    inlier_ratio: float
    mean_reproj_error_px: float
    rotation_diff_deg: float
    translation_diff_m: float
    status: str
    reject_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationRecord:
    comparison_pair_id: str
    gate_enabled: bool
    current_sensor_frame_idx: int
    current_visual_map_idx: int
    current_loop_frame_idx: int
    candidate_sensor_frame_idx: int
    candidate_visual_map_idx: int
    candidate_loop_frame_idx: int
    bow_score: float
    status: str
    reject_code: str | None
    reject_reason: str | None
    elapsed_ms: float
    comparison_applicable: bool | None = None
    num_flow_points: int | None = None
    num_geometry_points: int | None = None
    num_pnp_inliers: int | None = None
    inlier_ratio: float | None = None
    mean_reproj_error_px: float | None = None
    rotation_diff_deg: float | None = None
    translation_diff_m: float | None = None
    pnp_relative_pose: list[float] | None = None
    relative_pose: list[float] | None = None
    pnp_attempted: bool | None = None
    pnp_ransac_succeeded: bool | None = None
    pnp_inlier_gate_passed: bool | None = None
    reprojection_gate_passed: bool | None = None
    pose_consistency_gate_passed: bool | None = None
    pnp_rng_seed: int | None = None
    pnp_rng_seed_applied: bool | None = None
    diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreparedCommon:
    candidate_uv: torch.Tensor
    current_uv: torch.Tensor
    original_indices: torch.Tensor
    depth: torch.Tensor
    depth_valid: torch.Tensor
    match_mask: torch.Tensor | None
    covariance: torch.Tensor | None
    covariance_gate_mask: torch.Tensor | None
    diagnostics: dict[str, Any]
    comparison_applicable: bool
    phase_b5_row: dict[str, Any] | None = None
    phase_b5_selection: NMSSelection | None = None


@dataclass
class CommonRejection:
    reject_code: str
    reject_reason: str
    diagnostics: dict[str, Any] | None = None
    comparison_applicable: bool | None = None


def has_required_pnp_functions() -> bool:
    return all(
        hasattr(cv2, name)
        for name in ("solvePnPRansac", "solvePnPRefineLM", "Rodrigues", "SOLVEPNP_EPNP")
    )


def _get_config(config: SimpleNamespace, section: str, key: str, default: Any) -> Any:
    owner = getattr(config, section, None)
    return getattr(owner, key, default) if owner is not None else default


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _empty_statistics() -> dict[str, int | float | None]:
    return {
        "count": 0,
        "min": None,
        "p25": None,
        "p50": None,
        "p75": None,
        "p90": None,
        "p95": None,
        "p99": None,
        "max": None,
    }


def covariance_statistics(value: torch.Tensor) -> dict[str, int | float | None]:
    finite = value.detach().cpu().float().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return _empty_statistics()
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]),
        interpolation="linear",
    )
    return {
        "count": int(finite.numel()),
        "min": float(finite.min().item()),
        "p25": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p75": float(quantiles[2].item()),
        "p90": float(quantiles[3].item()),
        "p95": float(quantiles[4].item()),
        "p99": float(quantiles[5].item()),
        "max": float(finite.max().item()),
    }


def _correspondence_signature(indices: torch.Tensor) -> str:
    normalized = indices.detach().cpu().to(torch.int64).contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _covariance_triplet_statistics(covariance: torch.Tensor) -> dict[str, Any]:
    uu, vv = covariance[0], covariance[1]
    risk = torch.maximum(uu, vv)
    return {
        "uu": covariance_statistics(uu),
        "vv": covariance_statistics(vv),
        "risk_max_uu_vv": covariance_statistics(risk),
    }


def _rotation_matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return quat


def _opencv_pose_to_ned_se3(rvec: np.ndarray, tvec: np.ndarray) -> pp.LieTensor:
    rotation_cv, _ = cv2.Rodrigues(rvec)
    translation_cv = np.asarray(tvec, dtype=np.float64).reshape(3)
    ned_to_cv = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    cv_to_ned = ned_to_cv.T
    rotation_ned = cv_to_ned @ np.asarray(rotation_cv, dtype=np.float64) @ ned_to_cv
    translation_ned = cv_to_ned @ translation_cv
    quaternion = _rotation_matrix_to_quaternion_xyzw(rotation_ned)
    return pp.SE3(torch.tensor(np.concatenate([translation_ned, quaternion]), dtype=torch.float32))


def _pose_error(measured: pp.LieTensor, reference: pp.LieTensor) -> tuple[float, float]:
    residual = (measured.Inv() @ reference).Log().tensor().detach().cpu().double()
    translation = float(torch.linalg.vector_norm(residual[..., :3]))
    rotation_deg = float(torch.linalg.vector_norm(residual[..., 3:]) * 180.0 / math.pi)
    return rotation_deg, translation


class LoopCandidateVerifier:
    def __init__(
        self,
        config: SimpleNamespace,
        frontend: IFrontend,
        phase_b5: PhaseB5Analyzer | None = None,
    ) -> None:
        self.config = config
        self.frontend = frontend
        self.phase_b5 = phase_b5
        self.phase_b5_results: dict[str, tuple[VerificationRecord, LoopConstraint | None]] = {}
        self.frontend_inference_calls = 0
        self.candidates_reaching_frontend = 0
        self._aggregate_covariance: dict[str, list[torch.Tensor]] = {
            "uu": [],
            "vv": [],
            "uv": [],
            "sampled_frontend_score": [],
        }

    @staticmethod
    def _pair_id(query: dict[str, Any], candidate: dict[str, Any]) -> str:
        return f"{int(query['loop_frame_idx'])}:{int(candidate['loop_frame_idx'])}"

    @staticmethod
    def _pnp_seed(query: dict[str, Any], candidate: dict[str, Any]) -> int:
        return (
            int(query["sensor_frame_idx"]) * 1_000_003 + int(candidate["sensor_frame_idx"])
        ) & 0x7FFFFFFF

    def aggregate_covariance_statistics(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"interpolation": "linear"}
        for name, parts in self._aggregate_covariance.items():
            values = torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)
            summary[name] = covariance_statistics(values)
        return summary

    def verify_branches(
        self,
        pose_snapshot: torch.Tensor,
        query: dict[str, Any],
        candidate: dict[str, Any],
        current: LoopFrameRecord,
        historical: LoopFrameRecord,
        gate_modes: list[bool],
    ) -> dict[bool, tuple[VerificationRecord, LoopConstraint | None]]:
        started = time.perf_counter()
        try:
            common = self._prepare_once(query, candidate, current, historical)
        except Exception as error:
            if self.phase_b5 is not None and self.phase_b5.enabled:
                pair_id = self._pair_id(query, candidate)
                if not any(row.get("pair_id") == pair_id for row in self.phase_b5.rows):
                    self.phase_b5.record_flow_failure(
                        query, candidate,
                        {
                            "status": "not_computed",
                            "reject_code": "common_preparation_exception",
                            "orb_gate_pass": False,
                        },
                        "verification_exception", str(error),
                    )
            common = CommonRejection("verification_exception", f"common preparation exception: {error}")
        if isinstance(common, CommonRejection):
            return {
                gate: (
                    self._reject(
                        query,
                        candidate,
                        gate,
                        started,
                        common.reject_code,
                        common.reject_reason,
                        comparison_applicable=common.comparison_applicable,
                        diagnostics=copy.deepcopy(common.diagnostics),
                    ),
                    None,
                )
                for gate in gate_modes
            }

        self._run_phase_b5_shadow(
            pose_snapshot, query, candidate, current, historical, common, started
        )
        results: dict[bool, tuple[VerificationRecord, LoopConstraint | None]] = {}
        for gate in gate_modes:
            try:
                results[gate] = self._verify_branch(
                    pose_snapshot, query, candidate, current, historical, common, gate, started
                )
            except Exception as error:
                results[gate] = (
                    self._reject(
                        query,
                        candidate,
                        gate,
                        started,
                        "verification_exception",
                        f"exception: {error}",
                        comparison_applicable=common.comparison_applicable,
                        diagnostics=copy.deepcopy(common.diagnostics),
                    ),
                    None,
                )
        return results

    def _run_phase_b5_shadow(
        self,
        pose_snapshot: torch.Tensor,
        query: dict[str, Any],
        candidate: dict[str, Any],
        current: LoopFrameRecord,
        historical: LoopFrameRecord,
        common: PreparedCommon,
        started: float,
    ) -> None:
        if self.phase_b5 is None or common.phase_b5_row is None:
            return
        pair_id = self._pair_id(query, candidate)
        row = common.phase_b5_row
        selection = common.phase_b5_selection
        caps = (row.get("flow") or {}).get("point_risk_cap_by_population") or {}
        row["selector_shadow_by_population"] = {}
        row["refinement_by_population"] = {}
        if selection is None:
            row["selector_shadow_unavailable_reason"] = "selection_unavailable"
            return
        for population in row.get("populations", []):
            point_cap = caps.get(population)
            if point_cap is None:
                row["selector_shadow_by_population"][population] = None
                row["refinement_by_population"][population] = None
                continue
            capped_mask = selection.valid_mask & (selection.normalized_risk <= float(point_cap))
            capped_selection = copy.copy(selection)
            capped_selection.valid_mask = capped_mask
            spatial_indices = spatial_selection_indices(
                capped_selection, current,
                getattr(self.phase_b5.config, "flow", SimpleNamespace()),
            )
            spatial_mask = torch.zeros_like(capped_mask)
            spatial_mask[spatial_indices] = True
            capped_selection.valid_mask = spatial_mask
            nms_common = PreparedCommon(
                candidate_uv=selection.candidate_uv,
                current_uv=selection.current_uv,
                original_indices=selection.original_indices,
                depth=selection.depth,
                depth_valid=spatial_mask,
                match_mask=None,
                covariance=selection.covariance,
                covariance_gate_mask=None,
                diagnostics={
                    "frontend_type": type(self.frontend).__name__,
                    "candidate_samples": int(selection.candidate_uv.shape[0]),
                    "finite_flow_points": int(spatial_mask.sum()),
                    "inbound_flow_points": int(spatial_mask.sum()),
                    "after_inbound_and_finite": int(selection.candidate_uv.shape[0]),
                    "covariance_available": True,
                    "comparison_applicable": True,
                    "covariance_statistics_mother_set": "dense_q_nms_points",
                    "quantile_interpolation": "linear",
                    "flow_cov_threshold": float("inf"),
                    "phase_b5_selector": "q_nms_normalized_lambda_max_cap",
                    "phase_b5_population": population,
                    "phase_b5_point_risk_cap": float(point_cap),
                },
                comparison_applicable=True,
            )
            try:
                verification, constraint = self._verify_branch(
                    pose_snapshot, query, candidate, current, historical,
                    nms_common, False, started,
                )
                assert verification.diagnostics is not None
                verification.diagnostics["selection_mode"] = "phase_b5_q_nms_risk_cap"
                row["selector_shadow_by_population"][population] = verification.to_dict()
                self.phase_b5_results[f"{pair_id}|{population}"] = (verification, constraint)
                row["refinement_by_population"][population] = None
                if verification.pnp_ransac_succeeded and verification.pnp_relative_pose is not None:
                    pnp_diag, pose, inlier_indices = _run_flow_pnp(
                        capped_selection, current, historical,
                        getattr(self.phase_b5.config, "flow", SimpleNamespace()),
                    )
                    row.setdefault("selector_shadow_pnp_detail_by_population", {})[population] = pnp_diag
                    if pose is not None and inlier_indices is not None:
                        row["refinement_by_population"][population] = refine_and_information(
                            capped_selection, inlier_indices, pose, current, historical
                        )
            except Exception as error:
                row["selector_shadow_by_population"][population] = {
                    "status": "rejected", "reject_code": "verification_exception",
                    "reject_reason": str(error),
                }

    def _reject(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
        gate_enabled: bool,
        started: float,
        reject_code: str,
        reason: str,
        **kwargs: Any,
    ) -> VerificationRecord:
        return VerificationRecord(
            comparison_pair_id=self._pair_id(query, candidate),
            gate_enabled=gate_enabled,
            current_sensor_frame_idx=int(query["sensor_frame_idx"]),
            current_visual_map_idx=int(query["visual_map_idx"]),
            current_loop_frame_idx=int(query["loop_frame_idx"]),
            candidate_sensor_frame_idx=int(candidate["sensor_frame_idx"]),
            candidate_visual_map_idx=int(candidate["visual_map_idx"]),
            candidate_loop_frame_idx=int(candidate["loop_frame_idx"]),
            bow_score=float(candidate.get("score", 0.0)),
            status="rejected",
            reject_code=reject_code,
            reject_reason=reason,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            **kwargs,
        )

    def _prepare_once(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
        current: LoopFrameRecord,
        historical: LoopFrameRecord,
    ) -> PreparedCommon | CommonRejection:
        current_sensor = int(query["sensor_frame_idx"])
        candidate_sensor = int(candidate["sensor_frame_idx"])
        if candidate_sensor >= current_sensor:
            if self.phase_b5 is not None and self.phase_b5.enabled:
                self.phase_b5.record_flow_failure(
                    query, candidate,
                    {"status": "not_computed", "reject_code": "common_rejection", "orb_gate_pass": False},
                    "invalid_candidate_order", "candidate is not earlier than current frame",
                )
            return CommonRejection("invalid_candidate_order", "candidate is not earlier than current frame")
        min_gap = int(getattr(self.config.geometric_verification, "min_sensor_gap", 100))
        if current_sensor - candidate_sensor < min_gap:
            if self.phase_b5 is not None and self.phase_b5.enabled:
                self.phase_b5.record_flow_failure(
                    query, candidate,
                    {"status": "not_computed", "reject_code": "common_rejection", "orb_gate_pass": False},
                    "inside_min_gap", "candidate is inside min_sensor_gap",
                )
            return CommonRejection("inside_min_gap", "candidate is inside min_sensor_gap")

        candidate_uv_cpu = self._sample_candidate_uv(historical)
        if candidate_uv_cpu.size(0) == 0:
            if self.phase_b5 is not None and self.phase_b5.enabled:
                try:
                    orb = self.phase_b5.observe_orb(current, historical)
                except Exception as error:
                    orb = {
                        "status": "rejected", "reject_code": "orb_observe_exception",
                        "reject_reason": str(error), "orb_gate_pass": False,
                    }
                self.phase_b5.record_flow_failure(
                    query, candidate, orb,
                    "no_candidate_depth_pixels", "no valid candidate depth pixels",
                )
            return CommonRejection(
                "no_candidate_depth_pixels",
                "no valid candidate depth pixels",
                diagnostics={"candidate_samples": 0},
            )

        phase_b5_orb: dict[str, Any] | None = None
        if self.phase_b5 is not None and self.phase_b5.enabled:
            try:
                phase_b5_orb = self.phase_b5.observe_orb(current, historical)
            except Exception as error:
                phase_b5_orb = {
                    "status": "rejected", "reject_code": "orb_observe_exception",
                    "reject_reason": str(error), "orb_gate_pass": False,
                }

        self.candidates_reaching_frontend += 1
        self.frontend_inference_calls += 1
        try:
            depth_current, match = self.frontend.estimate_pair(
                historical.to_stereo_data(getattr(self.frontend.config, "device", "cpu")),
                current.to_stereo_data(getattr(self.frontend.config, "device", "cpu")),
            )
        except Exception as error:
            if self.phase_b5 is not None and phase_b5_orb is not None:
                self.phase_b5.record_flow_failure(
                    query, candidate, phase_b5_orb,
                    "frontend_exception", str(error),
                )
            return CommonRejection("verification_exception", f"frontend exception: {error}")

        if getattr(match, "flow", None) is None:
            if self.phase_b5 is not None and phase_b5_orb is not None:
                self.phase_b5.record_flow_failure(
                    query, candidate, phase_b5_orb, "missing_flow", "flow is missing"
                )
            return CommonRejection(
                "missing_flow",
                "flow is missing",
                diagnostics={"candidate_samples": int(candidate_uv_cpu.size(0))},
            )

        flow_map = match.flow
        device = flow_map.device
        candidate_uv = candidate_uv_cpu.to(device=device, dtype=torch.float32)
        flow = IFrontend.retrieve_pixels(candidate_uv, flow_map)
        if flow is None:
            if self.phase_b5 is not None and phase_b5_orb is not None:
                self.phase_b5.record_flow_failure(
                    query, candidate, phase_b5_orb,
                    "missing_flow", "flow sampling failed",
                )
            return CommonRejection("missing_flow", "flow is missing")
        current_uv = candidate_uv + flow.T
        inbound = filterPointsInRange(
            current_uv,
            (0, int(current.width) - 1),
            (0, int(current.height) - 1),
        )
        finite_flow = torch.isfinite(flow).all(dim=0)
        base_mask = inbound & finite_flow
        original_indices = torch.arange(candidate_uv.size(0), device=device, dtype=torch.long)

        frontend_type = type(self.frontend).__name__
        diagnostics: dict[str, Any] = {
            "frontend_type": frontend_type,
            "uv_synthesized_zero_by_frontend_contract": frontend_type in {
                "FlowFormerCovFrontend",
                "CUDAGraph_FlowFormerCovFrontend",
            },
            "candidate_samples": int(candidate_uv.size(0)),
            "finite_flow_points": int(finite_flow.sum().item()),
            "inbound_flow_points": int(inbound.sum().item()),
            "after_inbound_and_finite": int(base_mask.sum().item()),
            "covariance_statistics_mother_set": "after_inbound_and_finite",
            "quantile_interpolation": "linear",
            "flow_cov_threshold": float(_get_config(self.config, "geometry", "max_flow_cov", 1.0)),
        }

        covariance_map = getattr(match, "cov", None)
        covariance: torch.Tensor | None = None
        covariance_gate_mask: torch.Tensor | None = None
        covariance_available = False
        if covariance_map is not None:
            if (
                covariance_map.ndim != 4
                or covariance_map.shape[0] < 1
                or covariance_map.shape[1] != 3
                or covariance_map.shape[-2:] != flow_map.shape[-2:]
            ):
                diagnostics["covariance_available"] = False
                if self.phase_b5 is not None and phase_b5_orb is not None:
                    self.phase_b5.record_flow_failure(
                        query, candidate, phase_b5_orb,
                        "invalid_covariance_layout", f"invalid covariance shape {tuple(covariance_map.shape)}",
                    )
                return CommonRejection(
                    "invalid_covariance_layout",
                    f"invalid covariance shape {tuple(covariance_map.shape)} for flow {tuple(flow_map.shape)}",
                    diagnostics=diagnostics,
                    comparison_applicable=False,
                )
            sampled_covariance = IFrontend.retrieve_pixels(candidate_uv, covariance_map)
            if sampled_covariance is None:
                diagnostics["covariance_available"] = False
                if self.phase_b5 is not None and phase_b5_orb is not None:
                    self.phase_b5.record_flow_failure(
                        query, candidate, phase_b5_orb,
                        "invalid_covariance_layout", "covariance sampling failed",
                    )
                return CommonRejection(
                    "invalid_covariance_layout",
                    "covariance sampling failed",
                    diagnostics=diagnostics,
                    comparison_applicable=False,
                )
            covariance = sampled_covariance[:, base_mask]
            covariance_available = True
            assert covariance is not None
            covariance_gate_mask = self._covariance_diagnostics(covariance, diagnostics)
        else:
            self._missing_covariance_diagnostics(diagnostics)

        comparison_applicable = covariance_available and int(base_mask.sum().item()) > 0
        diagnostics["covariance_available"] = covariance_available
        diagnostics["comparison_applicable"] = comparison_applicable

        base_candidate_uv = candidate_uv[base_mask]
        base_current_uv = current_uv[base_mask]
        base_indices = original_indices[base_mask]

        sampled_match_mask: torch.Tensor | None = None
        if getattr(match, "mask", None) is not None:
            retrieved_mask = IFrontend.retrieve_pixels(candidate_uv, match.mask)
            if retrieved_mask is not None:
                sampled_match_mask = retrieved_mask.squeeze(0).bool()[base_mask]
                diagnostics["match_mask_available"] = True
                diagnostics["match_mask_pass_points"] = int(sampled_match_mask.sum().item())
            else:
                diagnostics["match_mask_available"] = False
        else:
            diagnostics["match_mask_available"] = False

        depth_map = historical.depth.to(device=device, dtype=torch.float32)
        depth = IFrontend.retrieve_pixels(base_candidate_uv, depth_map)
        if depth is None:
            if self.phase_b5 is not None and phase_b5_orb is not None:
                self.phase_b5.record_flow_failure(
                    query, candidate, phase_b5_orb,
                    "missing_candidate_depth", "candidate depth is missing",
                )
            return CommonRejection(
                "missing_candidate_depth",
                "candidate depth is missing",
                diagnostics=diagnostics,
                comparison_applicable=comparison_applicable,
            )
        depth = depth.squeeze(0)
        depth_valid = torch.isfinite(depth) & (depth > 0.0)

        phase_b5_row: dict[str, Any] | None = None
        phase_b5_selection: NMSSelection | None = None
        if self.phase_b5 is not None and self.phase_b5.enabled:
            try:
                phase_b5_row, phase_b5_selection = self.phase_b5.observe(
                    query, candidate, current, historical, match, depth_current,
                    base_candidate_uv, base_current_uv, covariance, phase_b5_orb,
                )
                diagnostics["phase_b5_pair_id"] = phase_b5_row["pair_id"]
            except Exception as error:
                orb_for_row = phase_b5_orb or {
                    "status": "not_computed", "reject_code": "orb_unavailable",
                    "orb_gate_pass": False,
                }
                populations = ["all_bow_candidates"] + (
                    ["orb_supported_candidates"] if orb_for_row.get("orb_gate_pass") else []
                )
                phase_b5_row = {
                    "pair_id": self._pair_id(query, candidate),
                    "current_sensor_frame_idx": current_sensor,
                    "candidate_sensor_frame_idx": candidate_sensor,
                    "bow_score": float(candidate.get("score", 0.0)),
                    "status": "rejected",
                    "reject_code": "phase_b5_observe_exception",
                    "reject_reason": str(error),
                    "orb": orb_for_row,
                    "flow": {
                        "pair_sanity_pass": False, "pair_gate_pass": None,
                        "reject_code": "phase_b5_observe_exception",
                        "reject_reason": str(error),
                    },
                    "populations": populations,
                    "in_calibration_prefix": self.phase_b5.is_in_calibration(query),
                }
                self.phase_b5.rows.append(phase_b5_row)

        return PreparedCommon(
            candidate_uv=base_candidate_uv,
            current_uv=base_current_uv,
            original_indices=base_indices,
            depth=depth,
            depth_valid=depth_valid,
            match_mask=sampled_match_mask,
            covariance=covariance,
            covariance_gate_mask=covariance_gate_mask,
            diagnostics=diagnostics,
            comparison_applicable=comparison_applicable,
            phase_b5_row=phase_b5_row,
            phase_b5_selection=phase_b5_selection,
        )

    def _covariance_diagnostics(
        self, covariance: torch.Tensor, diagnostics: dict[str, Any]
    ) -> torch.Tensor:
        uu, vv, uv = covariance[0], covariance[1], covariance[2]
        threshold = float(diagnostics["flow_cov_threshold"])
        uu_finite = torch.isfinite(uu)
        vv_finite = torch.isfinite(vv)
        uv_finite = torch.isfinite(uv)
        joint_finite = uu_finite & vv_finite
        uu_pass = uu_finite & (uu <= threshold)
        vv_pass = vv_finite & (vv <= threshold)
        joint_pass = joint_finite & (uu <= threshold) & (vv <= threshold)
        score = uu + vv - 2.0 * uv
        score_finite = torch.isfinite(score)
        finite_score = score[score_finite]
        if finite_score.numel() > 0:
            reference_threshold: float | None = min(100.0, float(finite_score.median().item()) * 1.5)
            reference_pass = score_finite & (score < reference_threshold)
        else:
            reference_threshold = None
            reference_pass = torch.zeros_like(score_finite)

        diagnostics["covariance"] = {
            "layout": ["uu", "vv", "uv"],
            "uu": covariance_statistics(uu),
            "vv": covariance_statistics(vv),
            "uv": covariance_statistics(uv),
            "uu_finite": int(uu_finite.sum().item()),
            "vv_finite": int(vv_finite.sum().item()),
            "uv_finite": int(uv_finite.sum().item()),
            "uv_all_zero": (
                None
                if int(uv_finite.sum().item()) == 0
                else bool((uv[uv_finite] == 0).all().item())
            ),
            "gate_channels_joint_finite": int(joint_finite.sum().item()),
            "uu_finite_and_below_threshold": int(uu_pass.sum().item()),
            "vv_finite_and_below_threshold": int(vv_pass.sum().item()),
            "gate_channels_joint_below_threshold": int(joint_pass.sum().item()),
            "sampled_frontend_score": covariance_statistics(score),
            "frontend_score_finite": int(score_finite.sum().item()),
            "frontend_score_nonfinite": int(score.numel() - score_finite.sum().item()),
            "frontend_score_negative": int((score_finite & (score < 0)).sum().item()),
            "sampled_frontend_threshold": reference_threshold,
            "frontend_score_below_reference": int(reference_pass.sum().item()),
            "frontend_reference_kind": "sampled/no_dense_nms",
            "frontend_reference_operator": "strict_less_than",
            "negative_uu_count": int((uu_finite & (uu < 0)).sum().item()),
            "negative_vv_count": int((vv_finite & (vv < 0)).sum().item()),
        }
        diagnostics["negative_uu_count"] = diagnostics["covariance"]["negative_uu_count"]
        diagnostics["negative_vv_count"] = diagnostics["covariance"]["negative_vv_count"]
        for name, values in (("uu", uu), ("vv", vv), ("uv", uv), ("sampled_frontend_score", score)):
            finite = values[torch.isfinite(values)].detach().cpu().float()
            if finite.numel() > 0:
                self._aggregate_covariance[name].append(finite)
        return joint_pass

    @staticmethod
    def _missing_covariance_diagnostics(diagnostics: dict[str, Any]) -> None:
        diagnostics["covariance"] = {
            "layout": ["uu", "vv", "uv"],
            "uu": _empty_statistics(),
            "vv": _empty_statistics(),
            "uv": _empty_statistics(),
            "uu_finite": 0,
            "vv_finite": 0,
            "uv_finite": 0,
            "uv_all_zero": None,
            "gate_channels_joint_finite": 0,
            "uu_finite_and_below_threshold": 0,
            "vv_finite_and_below_threshold": 0,
            "gate_channels_joint_below_threshold": 0,
            "sampled_frontend_score": _empty_statistics(),
            "frontend_score_finite": 0,
            "frontend_score_nonfinite": 0,
            "frontend_score_negative": 0,
            "sampled_frontend_threshold": None,
            "frontend_score_below_reference": 0,
            "frontend_reference_kind": "sampled/no_dense_nms",
            "frontend_reference_operator": "strict_less_than",
            "negative_uu_count": 0,
            "negative_vv_count": 0,
        }
        diagnostics["negative_uu_count"] = 0
        diagnostics["negative_vv_count"] = 0

    def _covariance_selection(
        self,
        common: PreparedCommon,
        gate_enabled: bool,
        diagnostics: dict[str, Any],
    ) -> torch.Tensor:
        point_count = common.candidate_uv.size(0)
        all_points = torch.ones(
            point_count, dtype=torch.bool, device=common.candidate_uv.device
        )
        target = getattr(
            getattr(self.config, "geometry", SimpleNamespace()),
            "flow_cov_adaptive_target_points",
            None,
        )
        adaptive = gate_enabled and target is not None
        diagnostics.update(
            {
                "selection_mode": (
                    "disabled" if not gate_enabled else "adaptive" if adaptive else "fixed"
                ),
                "fixed_core_points": None,
                "adaptive_target_points": int(target) if adaptive else None,
                "adaptive_pool_points": None,
                "adaptive_added_points": None,
                "adaptive_final_points_before_match_depth": None,
                "adaptive_points_after_match_mask": None,
                "adaptive_points_after_depth": None,
                "adaptive_shortfall": None,
                "adaptive_post_filter_shortfall": None,
                "adaptive_effective_risk_cutoff": None,
                "adaptive_added_risk_statistics": None,
                "adaptive_final_risk_statistics": None,
            }
        )
        if not gate_enabled:
            diagnostics["gate_applied"] = False
            diagnostics["gate_skip_reason"] = "disabled_by_config"
            return all_points
        if common.covariance is None:
            diagnostics["gate_applied"] = False
            diagnostics["gate_skip_reason"] = "covariance_unavailable"
            return all_points

        assert common.covariance_gate_mask is not None
        core = common.covariance_gate_mask.clone()
        core_count = int(core.sum().item())
        diagnostics["gate_applied"] = True
        diagnostics["gate_skip_reason"] = None
        diagnostics["fixed_core_points"] = core_count
        if target is None:
            return core

        target = int(target)
        covariance = common.covariance
        uu, vv = covariance[0], covariance[1]
        joint_finite = torch.isfinite(uu) & torch.isfinite(vv)
        pool = joint_finite & ~core
        pool_indices = torch.nonzero(pool, as_tuple=False).squeeze(1)
        diagnostics["adaptive_pool_points"] = int(pool_indices.numel())
        need = max(target - core_count, 0)
        added_indices = pool_indices[:0]
        if need > 0 and pool_indices.numel() > 0:
            risk_cpu = torch.maximum(uu[pool_indices], vv[pool_indices]).detach().cpu().float()
            original_cpu = common.original_indices[pool_indices].detach().cpu().to(torch.int64)
            # NumPy lexsort is deterministic: risk is primary, original sampling index breaks ties.
            order = np.lexsort((original_cpu.numpy(), risk_cpu.numpy()))
            take = torch.from_numpy(order[:need]).to(device=pool_indices.device, dtype=torch.long)
            added_indices = pool_indices[take]
            core[added_indices] = True

        added_count = int(added_indices.numel())
        final_count = int(core.sum().item())
        diagnostics["adaptive_added_points"] = added_count
        diagnostics["adaptive_final_points_before_match_depth"] = final_count
        diagnostics["adaptive_shortfall"] = max(target - final_count, 0)
        if added_count > 0:
            added_covariance = covariance[:, added_indices]
            diagnostics["adaptive_effective_risk_cutoff"] = float(
                torch.maximum(added_covariance[0], added_covariance[1]).max().item()
            )
            diagnostics["adaptive_added_risk_statistics"] = _covariance_triplet_statistics(
                added_covariance
            )
        diagnostics["adaptive_final_risk_statistics"] = _covariance_triplet_statistics(
            covariance[:, core]
        )
        return core

    def _verify_branch(
        self,
        pose_snapshot: torch.Tensor,
        query: dict[str, Any],
        candidate: dict[str, Any],
        current: LoopFrameRecord,
        historical: LoopFrameRecord,
        common: PreparedCommon,
        gate_enabled: bool,
        started: float,
    ) -> tuple[VerificationRecord, LoopConstraint | None]:
        diagnostics = copy.deepcopy(common.diagnostics)
        mask = self._covariance_selection(common, gate_enabled, diagnostics)
        diagnostics["gate_requested"] = gate_enabled
        diagnostics["after_flow_cov"] = int(mask.sum().item())

        if common.match_mask is not None:
            mask &= common.match_mask
        diagnostics["after_match_mask"] = int(mask.sum().item())
        if diagnostics["selection_mode"] == "adaptive":
            diagnostics["adaptive_points_after_match_mask"] = int(mask.sum().item())

        flow_count = int(mask.sum().item())
        diagnostics["flow_points"] = flow_count
        if flow_count == 0:
            diagnostics["valid_depth_after_flow_points"] = 0
            diagnostics["geometry_points"] = 0
            if diagnostics["selection_mode"] == "adaptive":
                target = int(diagnostics["adaptive_target_points"])
                diagnostics["adaptive_points_after_depth"] = 0
                diagnostics["adaptive_post_filter_shortfall"] = target
            diagnostics["pnp_correspondence_count"] = 0
            diagnostics["pnp_correspondence_signature"] = _correspondence_signature(
                common.original_indices[:0]
            )
            diagnostics["pnp_input_points"] = 0
            return self._reject(
                query,
                candidate,
                gate_enabled,
                started,
                "no_valid_flow_correspondences",
                "no valid flow correspondences",
                comparison_applicable=common.comparison_applicable,
                pnp_attempted=False,
                diagnostics=diagnostics,
            ), None

        depth_mask = common.depth_valid[mask]
        diagnostics["valid_depth_after_flow_points"] = int(depth_mask.sum().item())
        selected = torch.nonzero(mask, as_tuple=False).squeeze(1)
        selected = selected[depth_mask]
        geometry_count = int(selected.numel())
        diagnostics["geometry_points"] = geometry_count
        if diagnostics["selection_mode"] == "adaptive":
            target = int(diagnostics["adaptive_target_points"])
            diagnostics["adaptive_points_after_depth"] = geometry_count
            diagnostics["adaptive_post_filter_shortfall"] = max(target - geometry_count, 0)
        original_indices = common.original_indices[selected]
        diagnostics["pnp_correspondence_count"] = geometry_count
        diagnostics["pnp_correspondence_signature"] = _correspondence_signature(original_indices)
        if geometry_count == 0:
            diagnostics["pnp_input_points"] = 0
            return self._reject(
                query,
                candidate,
                gate_enabled,
                started,
                "no_valid_candidate_depths",
                "no valid candidate depths after flow filtering",
                comparison_applicable=common.comparison_applicable,
                num_flow_points=flow_count,
                pnp_attempted=False,
                diagnostics=diagnostics,
            ), None

        min_points = int(_get_config(self.config, "geometry", "min_points", 80))
        if geometry_count < min_points:
            diagnostics["pnp_input_points"] = 0
            return self._reject(
                query,
                candidate,
                gate_enabled,
                started,
                "insufficient_geometry_points",
                "not enough geometry points",
                comparison_applicable=common.comparison_applicable,
                num_flow_points=flow_count,
                num_geometry_points=geometry_count,
                pnp_attempted=False,
                diagnostics=diagnostics,
            ), None

        candidate_uv = common.candidate_uv[selected]
        current_uv = common.current_uv[selected]
        depth = common.depth[selected]
        frame_K = historical.intrinsic[0] if historical.intrinsic.ndim == 3 else historical.intrinsic
        frame_K = frame_K.to(device=candidate_uv.device, dtype=torch.float32)
        points_ned = pixel2point_NED(candidate_uv, depth, frame_K).float()
        points_cv = _tensor_to_numpy(points_ned.roll(shifts=-1, dims=-1).float())
        diagnostics["pnp_input_points"] = geometry_count
        if common.covariance is not None:
            selected_covariance = common.covariance[:, selected]
            diagnostics["pnp_input_covariance_statistics"] = _covariance_triplet_statistics(
                selected_covariance
            )
        else:
            selected_covariance = None
            diagnostics["pnp_input_covariance_statistics"] = None
        diagnostics["pnp_inlier_covariance_statistics"] = None
        diagnostics["pnp_outlier_covariance_statistics"] = None
        diagnostics["pnp_inlier_original_index_count"] = None
        diagnostics["pnp_inlier_original_index_signature"] = None
        diagnostics["pnp_outlier_original_index_count"] = None
        diagnostics["pnp_outlier_original_index_signature"] = None

        seed = self._pnp_seed(query, candidate)
        rng_applied = False
        if hasattr(cv2, "setRNGSeed"):
            cv2.setRNGSeed(seed)
            rng_applied = True
        pnp = self._run_pnp(current, points_cv, current_uv)
        if isinstance(pnp, str):
            reject_code = "pnp_insufficient_points" if pnp == "not enough points for PnP" else "pnp_failed"
            return self._reject(
                query,
                candidate,
                gate_enabled,
                started,
                reject_code,
                pnp,
                comparison_applicable=common.comparison_applicable,
                num_flow_points=flow_count,
                num_geometry_points=geometry_count,
                pnp_attempted=True,
                pnp_ransac_succeeded=False,
                pnp_rng_seed=seed,
                pnp_rng_seed_applied=rng_applied,
                diagnostics=diagnostics,
            ), None

        T_current_candidate, mean_error, inliers = pnp
        inlier_count = int(len(inliers))
        inlier_local = torch.as_tensor(inliers, device=selected.device, dtype=torch.long)
        outlier_local_mask = torch.ones(geometry_count, dtype=torch.bool, device=selected.device)
        outlier_local_mask[inlier_local] = False
        outlier_local = torch.nonzero(outlier_local_mask, as_tuple=False).squeeze(1)
        diagnostics["pnp_inlier_original_index_count"] = inlier_count
        diagnostics["pnp_inlier_original_index_signature"] = _correspondence_signature(
            original_indices[inlier_local]
        )
        diagnostics["pnp_outlier_original_index_count"] = int(outlier_local.numel())
        diagnostics["pnp_outlier_original_index_signature"] = _correspondence_signature(
            original_indices[outlier_local]
        )
        if selected_covariance is not None:
            diagnostics["pnp_inlier_covariance_statistics"] = _covariance_triplet_statistics(
                selected_covariance[:, inlier_local]
            )
            diagnostics["pnp_outlier_covariance_statistics"] = _covariance_triplet_statistics(
                selected_covariance[:, outlier_local]
            )
        inlier_ratio = inlier_count / max(geometry_count, 1)
        min_inliers = int(_get_config(self.config, "pnp", "min_inliers", 50))
        min_inlier_ratio = float(_get_config(self.config, "pnp", "min_inlier_ratio", 0.25))
        inlier_gate_passed = inlier_count >= min_inliers and inlier_ratio >= min_inlier_ratio
        if not inlier_gate_passed:
            return self._reject(
                query,
                candidate,
                gate_enabled,
                started,
                "insufficient_pnp_inliers",
                "not enough PnP inliers",
                comparison_applicable=common.comparison_applicable,
                num_flow_points=flow_count,
                num_geometry_points=geometry_count,
                num_pnp_inliers=inlier_count,
                inlier_ratio=inlier_ratio,
                mean_reproj_error_px=mean_error,
                pnp_attempted=True,
                pnp_ransac_succeeded=True,
                pnp_inlier_gate_passed=False,
                pnp_rng_seed=seed,
                pnp_rng_seed_applied=rng_applied,
                diagnostics=diagnostics,
            ), None

        max_reproj = float(_get_config(self.config, "verification", "max_mean_reproj_error_px", 3.0))
        reprojection_passed = mean_error <= max_reproj
        if not reprojection_passed:
            return self._reject(
                query,
                candidate,
                gate_enabled,
                started,
                "reprojection_error",
                "mean reprojection error is too high",
                comparison_applicable=common.comparison_applicable,
                num_flow_points=flow_count,
                num_geometry_points=geometry_count,
                num_pnp_inliers=inlier_count,
                inlier_ratio=inlier_ratio,
                mean_reproj_error_px=mean_error,
                pnp_attempted=True,
                pnp_ransac_succeeded=True,
                pnp_inlier_gate_passed=True,
                reprojection_gate_passed=False,
                pnp_rng_seed=seed,
                pnp_rng_seed_applied=rng_applied,
                diagnostics=diagnostics,
            ), None

        T_edge = T_current_candidate.Inv()
        T_w_src = pp.SE3(pose_snapshot[historical.visual_map_idx])
        T_w_dst = pp.SE3(pose_snapshot[current.visual_map_idx])
        vo_edge = T_w_src.Inv() @ T_w_dst
        rotation_diff, translation_diff = _pose_error(T_edge, vo_edge)
        max_rot = float(_get_config(self.config, "verification", "max_rotation_diff_deg", 35.0))
        max_trans = float(_get_config(self.config, "verification", "max_translation_diff_m", 8.0))
        pose_passed = rotation_diff <= max_rot and translation_diff <= max_trans
        if not pose_passed:
            return self._reject(
                query,
                candidate,
                gate_enabled,
                started,
                "pose_inconsistent",
                "relative pose is inconsistent with VO trajectory",
                comparison_applicable=common.comparison_applicable,
                num_flow_points=flow_count,
                num_geometry_points=geometry_count,
                num_pnp_inliers=inlier_count,
                inlier_ratio=inlier_ratio,
                mean_reproj_error_px=mean_error,
                rotation_diff_deg=rotation_diff,
                translation_diff_m=translation_diff,
                pnp_relative_pose=T_current_candidate.tensor().detach().cpu().tolist(),
                relative_pose=T_edge.tensor().detach().cpu().tolist(),
                pnp_attempted=True,
                pnp_ransac_succeeded=True,
                pnp_inlier_gate_passed=True,
                reprojection_gate_passed=True,
                pose_consistency_gate_passed=False,
                pnp_rng_seed=seed,
                pnp_rng_seed_applied=rng_applied,
                diagnostics=diagnostics,
            ), None

        information = make_information(
            float(_get_config(self.config, "loop_information", "trans_weight", 1.0)),
            float(_get_config(self.config, "loop_information", "rot_weight", 1.0)),
            dtype=torch.float32,
        )
        constraint = LoopConstraint(
            src_visual_map_idx=int(historical.visual_map_idx),
            dst_visual_map_idx=int(current.visual_map_idx),
            src_sensor_frame_idx=int(historical.sensor_frame_idx),
            dst_sensor_frame_idx=int(current.sensor_frame_idx),
            pnp_relative_pose=T_current_candidate.tensor().detach().cpu().tolist(),
            relative_pose=T_edge.tensor().detach().cpu().tolist(),
            information=information.detach().cpu().tolist(),
            bow_score=float(candidate.get("score", 0.0)),
            num_flow_points=flow_count,
            num_geometry_points=geometry_count,
            num_pnp_inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            mean_reproj_error_px=mean_error,
            rotation_diff_deg=rotation_diff,
            translation_diff_m=translation_diff,
            status="accepted",
        )
        verification = VerificationRecord(
            comparison_pair_id=self._pair_id(query, candidate),
            gate_enabled=gate_enabled,
            current_sensor_frame_idx=int(current.sensor_frame_idx),
            current_visual_map_idx=int(current.visual_map_idx),
            current_loop_frame_idx=int(current.loop_frame_idx),
            candidate_sensor_frame_idx=int(historical.sensor_frame_idx),
            candidate_visual_map_idx=int(historical.visual_map_idx),
            candidate_loop_frame_idx=int(historical.loop_frame_idx),
            bow_score=float(candidate.get("score", 0.0)),
            status="accepted",
            reject_code=None,
            reject_reason=None,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            comparison_applicable=common.comparison_applicable,
            num_flow_points=flow_count,
            num_geometry_points=geometry_count,
            num_pnp_inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            mean_reproj_error_px=mean_error,
            rotation_diff_deg=rotation_diff,
            translation_diff_m=translation_diff,
            pnp_relative_pose=constraint.pnp_relative_pose,
            relative_pose=constraint.relative_pose,
            pnp_attempted=True,
            pnp_ransac_succeeded=True,
            pnp_inlier_gate_passed=True,
            reprojection_gate_passed=True,
            pose_consistency_gate_passed=True,
            pnp_rng_seed=seed,
            pnp_rng_seed_applied=rng_applied,
            diagnostics=diagnostics,
        )
        return verification, constraint

    def _sample_candidate_uv(self, record: LoopFrameRecord) -> torch.Tensor:
        geometry = getattr(self.config, "geometry", SimpleNamespace())
        border = int(getattr(geometry, "border", 8))
        grid_rows = int(getattr(geometry, "grid_rows", 8))
        grid_cols = int(getattr(geometry, "grid_cols", 8))
        per_cell = int(getattr(geometry, "max_points_per_cell", 20))
        max_points = int(getattr(geometry, "max_points", 800))
        max_depth = float(getattr(geometry, "max_depth", 20.0))
        max_depth_cov = float(getattr(geometry, "max_depth_cov", 0.05))

        depth = record.depth[0, 0].detach().cpu().float()
        valid = torch.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
        if record.depth_covariance is not None:
            depth_cov = record.depth_covariance[0, 0].detach().cpu().float()
            valid &= torch.isfinite(depth_cov) & (depth_cov <= max_depth_cov)
        if border > 0:
            valid[:border, :] = False
            valid[-border:, :] = False
            valid[:, :border] = False
            valid[:, -border:] = False

        height, width = depth.shape
        selected: list[torch.Tensor] = []
        y_edges = torch.linspace(0, height, grid_rows + 1).round().long()
        x_edges = torch.linspace(0, width, grid_cols + 1).round().long()
        for row in range(grid_rows):
            for col in range(grid_cols):
                y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
                x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
                ys, xs = torch.nonzero(valid[y0:y1, x0:x1], as_tuple=True)
                if len(xs) == 0:
                    continue
                coords = torch.stack([xs + x0, ys + y0], dim=1).float()
                if coords.size(0) > per_cell:
                    take = torch.linspace(0, coords.size(0) - 1, per_cell).round().long()
                    coords = coords[take]
                selected.append(coords)
        if not selected:
            return torch.empty((0, 2), dtype=torch.float32)
        points = torch.cat(selected, dim=0)
        if points.size(0) > max_points:
            take = torch.linspace(0, points.size(0) - 1, max_points).round().long()
            points = points[take]
        return points

    def _run_pnp(
        self,
        current: LoopFrameRecord,
        object_points_cv: np.ndarray,
        image_points: torch.Tensor,
    ) -> tuple[pp.LieTensor, float, np.ndarray] | str:
        pnp_cfg = getattr(self.config, "pnp", SimpleNamespace())
        if object_points_cv.shape[0] < 4:
            return "not enough points for PnP"
        image_points_np = _tensor_to_numpy(image_points.float())
        K = _tensor_to_numpy(
            current.intrinsic[0] if current.intrinsic.ndim == 3 else current.intrinsic
        ).astype(np.float64)
        dist_coeffs = np.empty((0, 1), dtype=np.float64)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points_cv.astype(np.float64),
            image_points_np.astype(np.float64),
            K,
            dist_coeffs,
            iterationsCount=int(getattr(pnp_cfg, "iterations", 100)),
            reprojectionError=float(getattr(pnp_cfg, "reproj_error_px", 3.0)),
            confidence=float(getattr(pnp_cfg, "confidence", 0.999)),
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None or len(inliers) == 0:
            return "PnP RANSAC failed"
        inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
        inlier_indices = [int(index) for index in inliers]
        if bool(getattr(pnp_cfg, "refine", True)) and hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points_cv[inlier_indices].astype(np.float64),
                image_points_np[inlier_indices].astype(np.float64),
                K,
                dist_coeffs,
                rvec,
                tvec,
            )
        projected, _ = cv2.projectPoints(
            object_points_cv[inlier_indices].astype(np.float64), rvec, tvec, K, dist_coeffs
        )
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - image_points_np[inlier_indices], axis=1
        )
        mean_error = float(np.mean(errors)) if len(errors) > 0 else math.inf
        return _opencv_pose_to_ned_se3(rvec, tvec), mean_error, inliers
