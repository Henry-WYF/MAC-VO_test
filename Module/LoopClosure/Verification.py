from __future__ import annotations

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
from Module.Map import VisualMap
from Module.Optimization.GlobalPGO import make_information
from Utility.Point import filterPointsInRange, pixel2point_NED

from .Record import LoopFrameRecord


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
    current_sensor_frame_idx: int
    current_visual_map_idx: int
    current_loop_frame_idx: int
    candidate_sensor_frame_idx: int
    candidate_visual_map_idx: int
    candidate_loop_frame_idx: int
    bow_score: float
    status: str
    reject_reason: str | None
    elapsed_ms: float
    num_flow_points: int = 0
    num_geometry_points: int = 0
    num_pnp_inliers: int = 0
    inlier_ratio: float = 0.0
    mean_reproj_error_px: float = math.inf
    rotation_diff_deg: float = math.inf
    translation_diff_m: float = math.inf
    pnp_relative_pose: list[float] | None = None
    relative_pose: list[float] | None = None
    diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def has_required_pnp_functions() -> bool:
    return all(hasattr(cv2, name) for name in ("solvePnPRansac", "solvePnPRefineLM", "Rodrigues", "SOLVEPNP_EPNP"))


def _get_config(config: SimpleNamespace, section: str, key: str, default: Any) -> Any:
    owner = getattr(config, section, None)
    return getattr(owner, key, default) if owner is not None else default


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


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
    rotation_cv = np.asarray(rotation_cv, dtype=np.float64)
    translation_cv = np.asarray(tvec, dtype=np.float64).reshape(3)

    # NED camera coordinates are [forward, right, down]. OpenCV uses [right, down, forward].
    ned_to_cv = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    cv_to_ned = ned_to_cv.T
    rotation_ned = cv_to_ned @ rotation_cv @ ned_to_cv
    translation_ned = cv_to_ned @ translation_cv
    quaternion = _rotation_matrix_to_quaternion_xyzw(rotation_ned)
    return pp.SE3(torch.tensor(np.concatenate([translation_ned, quaternion]), dtype=torch.float32))


def _pose_error(measured: pp.LieTensor, reference: pp.LieTensor) -> tuple[float, float]:
    delta = measured.Inv() @ reference
    residual = delta.Log().tensor().detach().cpu().double()
    translation = float(torch.linalg.vector_norm(residual[..., :3]))
    rotation_deg = float(torch.linalg.vector_norm(residual[..., 3:]) * 180.0 / math.pi)
    return rotation_deg, translation


class LoopCandidateVerifier:
    def __init__(self, config: SimpleNamespace, frontend: IFrontend) -> None:
        self.config = config
        self.frontend = frontend

    def verify(
        self,
        global_map: VisualMap,
        query: dict[str, Any],
        candidate: dict[str, Any],
        current: LoopFrameRecord,
        historical: LoopFrameRecord,
    ) -> tuple[VerificationRecord, LoopConstraint | None]:
        started = time.perf_counter()
        try:
            return self._verify_impl(global_map, query, candidate, current, historical, started)
        except Exception as error:
            return self._reject(query, candidate, started, f"exception: {error}"), None

    def _reject(
        self,
        query: dict[str, Any],
        candidate: dict[str, Any],
        started: float,
        reason: str,
        **kwargs: Any,
    ) -> VerificationRecord:
        return VerificationRecord(
            current_sensor_frame_idx=int(query["sensor_frame_idx"]),
            current_visual_map_idx=int(query["visual_map_idx"]),
            current_loop_frame_idx=int(query["loop_frame_idx"]),
            candidate_sensor_frame_idx=int(candidate["sensor_frame_idx"]),
            candidate_visual_map_idx=int(candidate["visual_map_idx"]),
            candidate_loop_frame_idx=int(candidate["loop_frame_idx"]),
            bow_score=float(candidate.get("score", 0.0)),
            status="rejected",
            reject_reason=reason,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            **kwargs,
        )

    def _verify_impl(
        self,
        global_map: VisualMap,
        query: dict[str, Any],
        candidate: dict[str, Any],
        current: LoopFrameRecord,
        historical: LoopFrameRecord,
        started: float,
    ) -> tuple[VerificationRecord, LoopConstraint | None]:
        current_sensor = int(query["sensor_frame_idx"])
        candidate_sensor = int(candidate["sensor_frame_idx"])
        if candidate_sensor >= current_sensor:
            return self._reject(query, candidate, started, "candidate is not earlier than current frame"), None
        min_gap = int(getattr(self.config.geometric_verification, "min_sensor_gap", 100))
        if current_sensor - candidate_sensor < min_gap:
            return self._reject(query, candidate, started, "candidate is inside min_sensor_gap"), None

        depth_current, match = self.frontend.estimate_pair(
            historical.to_stereo_data(getattr(self.frontend.config, "device", "cpu")),
            current.to_stereo_data(getattr(self.frontend.config, "device", "cpu")),
        )
        del depth_current  # Current-frame depth is not the source of PnP 3D points.

        candidate_uv = self._sample_candidate_uv(historical)
        if candidate_uv.size(0) == 0:
            return self._reject(
                query, candidate, started, "no valid candidate depth pixels",
                diagnostics={"candidate_samples": 0},
            ), None

        prepared = self._prepare_correspondences(historical, current, match, candidate_uv)
        if isinstance(prepared[0], str):
            reason, diagnostics = prepared
            return self._reject(query, candidate, started, reason, diagnostics=diagnostics), None
        candidate_uv, current_uv, points_ned, points_cv, flow_count, diagnostics = prepared
        min_points = int(_get_config(self.config, "geometry", "min_points", 80))
        if points_cv.shape[0] < min_points:
            return self._reject(
                query, candidate, started, "not enough geometry points",
                num_flow_points=flow_count, num_geometry_points=int(points_cv.shape[0]),
                diagnostics=diagnostics,
            ), None

        pnp = self._run_pnp(current, points_cv, current_uv)
        if isinstance(pnp, str):
            return self._reject(
                query, candidate, started, pnp,
                num_flow_points=flow_count, num_geometry_points=int(points_cv.shape[0]),
                diagnostics=diagnostics,
            ), None
        T_current_candidate, mean_error, inliers = pnp
        inlier_count = int(len(inliers))
        inlier_ratio = inlier_count / max(int(points_cv.shape[0]), 1)
        min_inliers = int(_get_config(self.config, "pnp", "min_inliers", 50))
        min_inlier_ratio = float(_get_config(self.config, "pnp", "min_inlier_ratio", 0.25))
        if inlier_count < min_inliers or inlier_ratio < min_inlier_ratio:
            return self._reject(
                query, candidate, started, "not enough PnP inliers",
                num_flow_points=flow_count, num_geometry_points=int(points_cv.shape[0]),
                num_pnp_inliers=inlier_count, inlier_ratio=inlier_ratio, mean_reproj_error_px=mean_error,
                diagnostics=diagnostics,
            ), None

        max_reproj = float(_get_config(self.config, "verification", "max_mean_reproj_error_px", 3.0))
        if mean_error > max_reproj:
            return self._reject(
                query, candidate, started, "mean reprojection error is too high",
                num_flow_points=flow_count, num_geometry_points=int(points_cv.shape[0]),
                num_pnp_inliers=inlier_count, inlier_ratio=inlier_ratio, mean_reproj_error_px=mean_error,
                diagnostics=diagnostics,
            ), None

        T_edge = T_current_candidate.Inv()
        T_w_src = pp.SE3(global_map.frames.data["pose"][historical.visual_map_idx])
        T_w_dst = pp.SE3(global_map.frames.data["pose"][current.visual_map_idx])
        vo_edge = T_w_src.Inv() @ T_w_dst
        rotation_diff, translation_diff = _pose_error(T_edge, vo_edge)
        max_rot = float(_get_config(self.config, "verification", "max_rotation_diff_deg", 35.0))
        max_trans = float(_get_config(self.config, "verification", "max_translation_diff_m", 8.0))
        if rotation_diff > max_rot or translation_diff > max_trans:
            return self._reject(
                query, candidate, started, "relative pose is inconsistent with VO trajectory",
                num_flow_points=flow_count, num_geometry_points=int(points_cv.shape[0]),
                num_pnp_inliers=inlier_count, inlier_ratio=inlier_ratio, mean_reproj_error_px=mean_error,
                rotation_diff_deg=rotation_diff, translation_diff_m=translation_diff,
                pnp_relative_pose=T_current_candidate.tensor().detach().cpu().tolist(),
                relative_pose=T_edge.tensor().detach().cpu().tolist(),
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
            num_geometry_points=int(points_cv.shape[0]),
            num_pnp_inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            mean_reproj_error_px=mean_error,
            rotation_diff_deg=rotation_diff,
            translation_diff_m=translation_diff,
            status="accepted",
            reject_reason=None,
        )
        verification = VerificationRecord(
            current_sensor_frame_idx=int(current.sensor_frame_idx),
            current_visual_map_idx=int(current.visual_map_idx),
            current_loop_frame_idx=int(current.loop_frame_idx),
            candidate_sensor_frame_idx=int(historical.sensor_frame_idx),
            candidate_visual_map_idx=int(historical.visual_map_idx),
            candidate_loop_frame_idx=int(historical.loop_frame_idx),
            bow_score=float(candidate.get("score", 0.0)),
            status="accepted",
            reject_reason=None,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            num_flow_points=flow_count,
            num_geometry_points=int(points_cv.shape[0]),
            num_pnp_inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            mean_reproj_error_px=mean_error,
            rotation_diff_deg=rotation_diff,
            translation_diff_m=translation_diff,
            pnp_relative_pose=constraint.pnp_relative_pose,
            relative_pose=constraint.relative_pose,
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
        if len(selected) == 0:
            return torch.empty((0, 2), dtype=torch.float32)
        points = torch.cat(selected, dim=0)
        if points.size(0) > max_points:
            take = torch.linspace(0, points.size(0) - 1, max_points).round().long()
            points = points[take]
        return points

    def _prepare_correspondences(
        self,
        historical: LoopFrameRecord,
        current: LoopFrameRecord,
        match: Any,
        candidate_uv_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, int, dict[str, Any]] | tuple[str, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "candidate_samples": int(candidate_uv_cpu.size(0)),
            "flow_cov_threshold": float(_get_config(self.config, "geometry", "max_flow_cov", 1.0)),
        }
        device = match.flow.device
        candidate_uv = candidate_uv_cpu.to(device=device, dtype=torch.float32)
        flow = IFrontend.retrieve_pixels(candidate_uv, match.flow)
        if flow is None:
            return "flow is missing", diagnostics
        current_uv = candidate_uv + flow.T
        inbound = filterPointsInRange(
            current_uv,
            (0, int(current.width) - 1),
            (0, int(current.height) - 1),
        )
        finite_flow = torch.isfinite(flow).all(dim=0)
        diagnostics["finite_flow_points"] = int(finite_flow.sum().item())
        diagnostics["inbound_flow_points"] = int(inbound.sum().item())
        mask = inbound & finite_flow
        diagnostics["after_inbound_and_finite"] = int(mask.sum().item())
        flow_cov = IFrontend.retrieve_pixels(candidate_uv, match.cov)
        if flow_cov is not None:
            max_flow_cov = float(_get_config(self.config, "geometry", "max_flow_cov", 1.0))
            cov_finite = torch.isfinite(flow_cov[:2]).all(dim=0)
            cov_pass = (flow_cov[:2] <= max_flow_cov).all(dim=0)
            diagnostics["flow_cov_available"] = True
            diagnostics["flow_cov_finite_points"] = int(cov_finite.sum().item())
            diagnostics["flow_cov_below_threshold_points"] = int(cov_pass.sum().item())
            mask &= cov_finite
            mask &= cov_pass
            diagnostics["after_flow_cov"] = int(mask.sum().item())
        else:
            diagnostics["flow_cov_available"] = False
            diagnostics["after_flow_cov"] = int(mask.sum().item())
        if match.mask is not None:
            match_mask = IFrontend.retrieve_pixels(candidate_uv, match.mask)
            if match_mask is not None:
                valid_match_mask = match_mask.squeeze(0).bool()
                diagnostics["match_mask_available"] = True
                diagnostics["match_mask_pass_points"] = int(valid_match_mask.sum().item())
                mask &= valid_match_mask
                diagnostics["after_match_mask"] = int(mask.sum().item())
            else:
                diagnostics["match_mask_available"] = False
                diagnostics["after_match_mask"] = int(mask.sum().item())
        else:
            diagnostics["match_mask_available"] = False
            diagnostics["after_match_mask"] = int(mask.sum().item())

        candidate_uv = candidate_uv[mask]
        current_uv = current_uv[mask]
        flow_count = int(candidate_uv.size(0))
        diagnostics["flow_points"] = flow_count
        if flow_count == 0:
            return "no valid flow correspondences", diagnostics

        depth_map = historical.depth.to(device=device, dtype=torch.float32)
        depth = IFrontend.retrieve_pixels(candidate_uv, depth_map)
        if depth is None:
            return "candidate depth is missing", diagnostics
        depth = depth.squeeze(0)
        depth_mask = torch.isfinite(depth) & (depth > 0.0)
        diagnostics["valid_depth_after_flow_points"] = int(depth_mask.sum().item())
        candidate_uv = candidate_uv[depth_mask]
        current_uv = current_uv[depth_mask]
        depth = depth[depth_mask]
        diagnostics["geometry_points"] = int(candidate_uv.size(0))
        if candidate_uv.size(0) == 0:
            return "no valid candidate depths after flow filtering", diagnostics

        K = historical.intrinsic.to(device=device, dtype=torch.float32)
        frame_K = K[0] if K.ndim == 3 else K
        points_ned = pixel2point_NED(candidate_uv, depth, frame_K).float()
        points_cv = points_ned.roll(shifts=-1, dims=-1)
        return candidate_uv.cpu(), current_uv.cpu(), points_ned.cpu(), _tensor_to_numpy(points_cv.float()), flow_count, diagnostics

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
        K = _tensor_to_numpy(current.intrinsic[0] if current.intrinsic.ndim == 3 else current.intrinsic).astype(np.float64)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points_cv.astype(np.float64),
            image_points_np.astype(np.float64),
            K,
            None,
            iterationsCount=int(getattr(pnp_cfg, "iterations", 100)),
            reprojectionError=float(getattr(pnp_cfg, "reproj_error_px", 3.0)),
            confidence=float(getattr(pnp_cfg, "confidence", 0.999)),
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None or len(inliers) == 0:
            return "PnP RANSAC failed"
        inliers = inliers.reshape(-1)
        if bool(getattr(pnp_cfg, "refine", True)) and hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points_cv[inliers].astype(np.float64),
                image_points_np[inliers].astype(np.float64),
                K,
                None,
                rvec,
                tvec,
            )
        projected, _ = cv2.projectPoints(object_points_cv[inliers].astype(np.float64), rvec, tvec, K, None)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points_np[inliers], axis=1)
        mean_error = float(np.mean(errors)) if len(errors) > 0 else math.inf
        return _opencv_pose_to_ned_se3(rvec, tvec), mean_error, inliers
