from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import numpy as np
import torch
import pypose as pp

from Module.Map import VisualMap
from Utility.Extensions import ConfigTestable

from .Graph import PoseGraphEdge, as_se3, compute_edge_residual, make_information
from .Graph import relative_pose as compute_relative_pose


class GlobalPoseGraphOptimizer(ConfigTestable):
    """
    Minimal global pose graph optimizer for keyframe poses.

    Phase 1 intentionally has no place recognition. Odometry edges are built from
    the current trajectory, so a graph without loop/non-local edges is usually a no-op.
    """

    ZERO_LOSS_TOL = 1e-8

    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.enabled: bool = bool(config.enabled)
        self.optimize_on_terminate: bool = bool(config.optimize_on_terminate)
        self.max_iterations: int = int(config.max_iterations)
        self.trans_weight: float = float(config.trans_weight)
        self.rot_weight: float = float(config.rot_weight)
        self.device = torch.device(config.device)
        self.include_interp_frames: bool = bool(config.include_interp_frames)
        self.solver: str = str(getattr(config, "solver", "lbfgs"))
        self.loop_huber_delta: float = float(
            getattr(config, "loop_huber_delta", 3.548)
        )
        self.observation_huber_delta: float = float(
            getattr(config, "observation_huber_delta", 2.795)
        )
        self.observation_residual_mode: str = str(
            getattr(config, "observation_residual_mode", "disp")
        )
        sparse = getattr(config, "sparse_lm", SimpleNamespace())
        self.sparse_initial_damping = float(getattr(sparse, "initial_damping", 1e-3))
        self.sparse_min_damping = float(getattr(sparse, "min_damping", 1e-9))
        self.sparse_max_damping = float(getattr(sparse, "max_damping", 1e9))
        self.sparse_damping_up = float(getattr(sparse, "damping_up", 10.0))
        self.sparse_damping_down = float(getattr(sparse, "damping_down", 3.0))
        self.sparse_max_trials = int(getattr(sparse, "max_trials", 5))
        self.sparse_jacobian_epsilon = float(getattr(sparse, "jacobian_epsilon", 1e-6))

        self.edges: list[PoseGraphEdge] = []
        self._edge_keys: set[tuple[int, int, str]] = set()
        self._num_frames: int | None = None
        self._latest_poses: torch.Tensor | None = None
        self.odometry_information_diagnostics: list[dict] = []
        self.last_optimization_diagnostics: dict = {
            "solver": self.solver,
            "safe": False,
            "trajectory_source": "not_run",
        }

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        specification = {
            "enabled": lambda b: isinstance(b, bool),
            "optimize_on_terminate": lambda b: isinstance(b, bool),
            "max_iterations": lambda n: isinstance(n, int) and n >= 0,
            "trans_weight": lambda v: isinstance(v, (float, int)) and v > 0.0,
            "rot_weight": lambda v: isinstance(v, (float, int)) and v > 0.0,
            "device": lambda v: isinstance(v, str) and (v == "cpu" or "cuda" in v),
            "include_interp_frames": lambda b: isinstance(b, bool),
        }
        optional = {
            "solver": lambda value: value in {"lbfgs", "sparse_lm"},
            "loop_huber_delta": lambda value: isinstance(value, (int, float))
            and not isinstance(value, bool) and value > 0.0,
            "observation_huber_delta": lambda value: isinstance(value, (int, float))
            and not isinstance(value, bool) and value > 0.0,
            "observation_residual_mode": lambda value: value in {"disp", "icp"},
            "sparse_lm": lambda value: isinstance(value, SimpleNamespace),
        }
        for key, predicate in optional.items():
            if hasattr(config, key):
                specification[key] = predicate
        cls._enforce_config_spec(config, specification)
        if hasattr(config, "sparse_lm"):
            cls._enforce_config_spec(config.sparse_lm, {
                "initial_damping": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0,
                "min_damping": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0,
                "max_damping": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0,
                "damping_up": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1.0,
                "damping_down": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1.0,
                "max_trials": lambda value: isinstance(value, int) and not isinstance(value, bool) and value > 0,
                "jacobian_epsilon": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0,
            })

    def default_information(self) -> torch.Tensor:
        return make_information(self.trans_weight, self.rot_weight, self.device)

    def _poses_from_map(self, global_map: VisualMap) -> torch.Tensor:
        poses = global_map.frames.data["pose"][:]
        self._num_frames = int(poses.size(0))
        self._latest_poses = poses.detach().clone()
        return poses.to(device=self.device, dtype=torch.double)

    def _valid_frame_indices(self, global_map: VisualMap) -> list[int]:
        num_frames = len(global_map.frames)
        self._num_frames = num_frames
        if self.include_interp_frames or "need_interp" not in global_map.frames.data:
            return list(range(num_frames))

        need_interp = global_map.frames.data["need_interp"][:num_frames].bool()
        return torch.arange(num_frames, dtype=torch.long)[~need_interp].tolist()

    def _normalize_information(self, information: torch.Tensor | None) -> torch.Tensor:
        if information is None:
            return self.default_information()
        if information.shape != (6, 6):
            raise ValueError(f"Information matrix must have shape (6, 6), got {tuple(information.shape)}")
        return information.to(device=self.device, dtype=torch.double)

    def _make_edge(
        self,
        src: int,
        dst: int,
        rel_pose: pp.LieTensor | torch.Tensor,
        information: torch.Tensor | None,
        edge_type: str,
    ) -> PoseGraphEdge:
        if edge_type not in {"odometry", "loop"}:
            raise ValueError(f"Unsupported edge type '{edge_type}'")
        info = self._normalize_information(information)
        return PoseGraphEdge(
            src=int(src),
            dst=int(dst),
            relative_pose=as_se3(rel_pose, self.device, torch.double),
            information=info,
            edge_type=edge_type,
        )

    def _validate_frame_index(self, frame_idx: int) -> None:
        if frame_idx < 0:
            raise IndexError(f"Frame index must be non-negative, got {frame_idx}")
        if self._num_frames is not None and frame_idx >= self._num_frames:
            raise IndexError(f"Frame index {frame_idx} is out of range for {self._num_frames} frames")

    def _add_edge(self, edge: PoseGraphEdge, replace: bool = False) -> None:
        key = (edge.src, edge.dst, edge.edge_type)
        if key in self._edge_keys:
            if not replace:
                return
            for idx, existing in enumerate(self.edges):
                if (existing.src, existing.dst, existing.edge_type) == key:
                    self.edges[idx] = edge
                    return
        self.edges.append(edge)
        self._edge_keys.add(key)

    def register_odometry_edges(
        self,
        global_map: VisualMap,
        information_mode: str = "fixed",
    ) -> None:
        if information_mode not in {"fixed", "mixed_covariance_fixed"}:
            raise ValueError(f"unsupported odometry information mode {information_mode!r}")
        poses = self._poses_from_map(global_map)
        valid_indices = self._valid_frame_indices(global_map)
        if len(valid_indices) < 2:
            return

        pose_se3 = pp.SE3(poses)
        self.odometry_information_diagnostics = []
        for src, dst in zip(valid_indices[:-1], valid_indices[1:]):
            rel_pose = compute_relative_pose(pose_se3[src], pose_se3[dst])
            information = None
            diagnostics = {
                "src": int(src), "dst": int(dst),
                "requested_mode": information_mode,
                "used_mode": "fixed_information",
                "fallback": False,
                "reason": None,
            }
            if information_mode == "mixed_covariance_fixed":
                from Module.Optimization.ObservationInformation import (
                    odometry_edge_information,
                )
                try:
                    information, observed = odometry_edge_information(
                        global_map,
                        src,
                        dst,
                        huber_delta=self.observation_huber_delta,
                        residual_mode=self.observation_residual_mode,
                    )
                except Exception as error:
                    information = None
                    observed = {
                        "valid": False,
                        "reason": (
                            "direct_observation_lookup_failed:"
                            f"{type(error).__name__}"
                        ),
                    }
                diagnostics.update(observed)
                if information is None:
                    diagnostics.update({
                        "used_mode": "fixed_information",
                        "fallback": True,
                        "fallback_reason": observed.get("reason"),
                    })
                else:
                    diagnostics.update({
                        "used_mode": "observation_hessian",
                        "fallback": False,
                    })
            self.odometry_information_diagnostics.append(diagnostics)
            self._add_edge(
                self._make_edge(src, dst, rel_pose, information, "odometry"),
                replace=False,
            )

    def add_loop_edge(
        self,
        src: int,
        dst: int,
        relative_pose: pp.LieTensor | torch.Tensor,
        information: torch.Tensor | None = None,
    ) -> None:
        src, dst = int(src), int(dst)
        self._validate_frame_index(src)
        self._validate_frame_index(dst)
        edge = self._make_edge(src, dst, relative_pose, information, "loop")
        self._add_edge(edge, replace=True)

    def compute_residuals(self, poses: torch.Tensor | pp.LieTensor | None = None) -> torch.Tensor:
        if poses is None:
            if self._latest_poses is None:
                raise ValueError("No poses were provided and no map poses have been registered yet.")
            poses = self._latest_poses

        if len(self.edges) == 0:
            return torch.empty((0, 6), device=self.device, dtype=torch.double)

        pose_seq = as_se3(poses, self.device, torch.double)
        residuals = [compute_edge_residual(edge, pose_seq) for edge in self.edges]
        return torch.stack(residuals, dim=0).to(device=self.device, dtype=torch.double)

    def compute_loss(self, poses: torch.Tensor | pp.LieTensor | None = None) -> torch.Tensor:
        residuals = self.compute_residuals(poses)
        if residuals.numel() == 0:
            return torch.zeros((), device=self.device, dtype=torch.double)

        losses = []
        for residual, edge in zip(residuals, self.edges):
            info = edge.information.to(device=residual.device, dtype=residual.dtype)
            squared = residual.unsqueeze(0) @ info @ residual.unsqueeze(-1)
            if self.solver == "sparse_lm" and edge.edge_type == "loop":
                norm = squared.reshape(()).clamp_min(0.0).sqrt()
                delta = torch.as_tensor(
                    self.loop_huber_delta,
                    device=norm.device,
                    dtype=norm.dtype,
                )
                losses.append(torch.where(
                    norm <= delta,
                    0.5 * norm.square(),
                    delta * (norm - 0.5 * delta),
                ))
            else:
                losses.append(
                    0.5 * squared.reshape(())
                    if self.solver == "sparse_lm" else squared
                )
        return torch.stack(losses).sum()

    def _apply_pose_delta(self, base_poses: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if base_poses.size(0) <= 1:
            return base_poses
        updated_tail = pp.se3(delta).Exp() @ pp.SE3(base_poses[1:])
        return torch.cat([base_poses[:1], updated_tail.tensor()], dim=0)

    def optimize_poses(self, initial_poses: torch.Tensor) -> torch.Tensor:
        if self.solver == "sparse_lm":
            return self._optimize_sparse_lm(initial_poses)
        base_poses = initial_poses.to(device=self.device, dtype=torch.double).detach()
        if base_poses.size(0) <= 1 or len(self.edges) == 0 or self.max_iterations == 0:
            return base_poses.detach().cpu().float()

        initial_loss = self.compute_loss(base_poses).detach()
        if not torch.isfinite(initial_loss):
            raise RuntimeError(f"Initial global PGO loss is not finite: {initial_loss.item()}")
        if initial_loss.item() <= self.ZERO_LOSS_TOL:
            return base_poses.detach().cpu().float()

        delta = torch.zeros((base_poses.size(0) - 1, 6), device=self.device, dtype=torch.double, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [delta],
            max_iter=self.max_iterations,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            current_poses = self._apply_pose_delta(base_poses, delta)
            loss = self.compute_loss(current_poses)
            loss.backward()
            return loss

        optimizer.step(closure)
        optimized = self._apply_pose_delta(base_poses, delta.detach()).detach()
        optimized[0] = base_poses[0]
        return optimized.cpu().float()

    def _edge_local_residuals(
        self,
        source: pp.LieTensor,
        destination: pp.LieTensor,
        measurement: pp.LieTensor,
    ) -> torch.Tensor:
        return (measurement.Inv() @ (source.Inv() @ destination)).Log().tensor()

    def _sparse_linearization(
        self,
        poses: torch.Tensor,
    ) -> tuple[object, np.ndarray, float]:
        try:
            import scipy.sparse as sparse
        except ImportError as error:
            raise RuntimeError("sparse_lm requires scipy") from error
        pose = pp.SE3(poses)
        src_index = torch.tensor([edge.src for edge in self.edges], dtype=torch.long)
        dst_index = torch.tensor([edge.dst for edge in self.edges], dtype=torch.long)
        source = pose[src_index]
        destination = pose[dst_index]
        measurement = pp.SE3(torch.stack([
            edge.relative_pose.tensor().detach().cpu().double()
            for edge in self.edges
        ]))
        residual = self._edge_local_residuals(source, destination, measurement)
        epsilon = self.sparse_jacobian_epsilon
        jacobian_source = torch.empty(
            (len(self.edges), 6, 6), dtype=torch.float64,
        )
        jacobian_destination = torch.empty_like(jacobian_source)
        for axis in range(6):
            perturbation = torch.zeros((len(self.edges), 6), dtype=torch.float64)
            perturbation[:, axis] = epsilon
            positive = pp.se3(perturbation).Exp()
            negative = pp.se3(-perturbation).Exp()
            jacobian_source[:, :, axis] = (
                self._edge_local_residuals(
                    positive @ source, destination, measurement,
                )
                - self._edge_local_residuals(
                    negative @ source, destination, measurement,
                )
            ) / (2.0 * epsilon)
            jacobian_destination[:, :, axis] = (
                self._edge_local_residuals(
                    source, positive @ destination, measurement,
                )
                - self._edge_local_residuals(
                    source, negative @ destination, measurement,
                )
            ) / (2.0 * epsilon)

        information = torch.stack([
            edge.information.detach().cpu().double() for edge in self.edges
        ])
        cholesky, info = torch.linalg.cholesky_ex(information)
        if bool((info != 0).any()):
            raise RuntimeError("pose graph information is not SPD")
        whitener = cholesky.transpose(-1, -2)
        whitened_residual = (whitener @ residual.unsqueeze(-1)).squeeze(-1)
        whitened_source = whitener @ jacobian_source
        whitened_destination = whitener @ jacobian_destination
        norms = torch.linalg.vector_norm(whitened_residual, dim=-1)
        weights = torch.ones_like(norms)
        robust_cost = torch.zeros((), dtype=torch.float64)
        for index, edge in enumerate(self.edges):
            norm = norms[index]
            if edge.edge_type == "loop":
                delta = self.loop_huber_delta
                weights[index] = min(1.0, delta / max(float(norm), 1e-12))
                robust_cost = robust_cost + (
                    0.5 * norm.square()
                    if float(norm) <= delta
                    else delta * (norm - 0.5 * delta)
                )
            else:
                robust_cost = robust_cost + 0.5 * norm.square()
        square_root_weight = weights.sqrt().reshape(-1, 1)
        whitened_residual = whitened_residual * square_root_weight
        whitened_source = whitened_source * square_root_weight.unsqueeze(-1)
        whitened_destination = whitened_destination * square_root_weight.unsqueeze(-1)

        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        for edge_index, (src, dst) in enumerate(zip(src_index.tolist(), dst_index.tolist())):
            rows = np.repeat(
                np.arange(edge_index * 6, edge_index * 6 + 6, dtype=np.int64),
                6,
            )
            for frame_index, block in (
                (src, whitened_source[edge_index]),
                (dst, whitened_destination[edge_index]),
            ):
                if frame_index == 0:
                    continue
                columns = np.tile(
                    np.arange(
                        (frame_index - 1) * 6,
                        frame_index * 6,
                        dtype=np.int64,
                    ),
                    6,
                )
                row_parts.append(rows)
                column_parts.append(columns)
                value_parts.append(block.detach().cpu().numpy().reshape(-1))
        dimension = max(int(poses.shape[0]) - 1, 0) * 6
        if value_parts:
            matrix = sparse.coo_matrix(
                (
                    np.concatenate(value_parts),
                    (np.concatenate(row_parts), np.concatenate(column_parts)),
                ),
                shape=(len(self.edges) * 6, dimension),
            ).tocsr()
        else:
            matrix = sparse.csr_matrix((len(self.edges) * 6, dimension))
        return (
            matrix,
            whitened_residual.detach().cpu().numpy().reshape(-1),
            float(robust_cost),
        )

    def _optimize_sparse_lm(self, initial_poses: torch.Tensor) -> torch.Tensor:
        if self.device.type != "cpu":
            raise ValueError("sparse_lm currently requires device=cpu")
        try:
            import scipy.sparse as sparse
            from scipy.sparse.linalg import MatrixRankWarning, spsolve
        except ImportError as error:
            raise RuntimeError("sparse_lm requires scipy") from error
        base = initial_poses.detach().cpu().double()
        current = base.clone()
        diagnostics = {
            "solver": "sparse_lm",
            "safe": False,
            "trajectory_source": "original_fallback",
            "iterations": 0,
            "accepted_steps": 0,
            "rejected_trials": 0,
            "initial_robust_loss": None,
            "final_robust_loss": None,
            "convergence_reason": None,
        }
        self.last_optimization_diagnostics = diagnostics
        if current.size(0) <= 1 or len(self.edges) == 0 or self.max_iterations == 0:
            diagnostics.update({
                "safe": True,
                "trajectory_source": "optimized",
                "initial_robust_loss": 0.0,
                "final_robust_loss": 0.0,
                "convergence_reason": "nothing_to_optimize",
            })
            return current.float()
        try:
            jacobian, residual, current_loss = self._sparse_linearization(current)
        except (RuntimeError, ValueError) as error:
            diagnostics["convergence_reason"] = str(error)
            return base.float()
        diagnostics["initial_robust_loss"] = current_loss
        if not math.isfinite(current_loss):
            diagnostics["convergence_reason"] = "nonfinite_initial_loss"
            return base.float()
        damping = self.sparse_initial_damping
        convergence_reason = "max_iterations"
        for iteration in range(self.max_iterations):
            diagnostics["iterations"] = iteration + 1
            hessian = (jacobian.T @ jacobian).tocsr()
            gradient = jacobian.T @ residual
            diagonal = np.maximum(hessian.diagonal(), 1e-12)
            accepted = False
            previous_loss = current_loss
            for _trial in range(self.sparse_max_trials):
                damped = hessian + sparse.diags(damping * diagonal)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", MatrixRankWarning)
                        delta = np.asarray(spsolve(damped, -gradient)).reshape(-1)
                except (MatrixRankWarning, RuntimeError, ValueError):
                    delta = np.full_like(gradient, np.nan)
                if not np.isfinite(delta).all():
                    damping = min(
                        damping * self.sparse_damping_up,
                        self.sparse_max_damping,
                    )
                    diagnostics["rejected_trials"] += 1
                    continue
                delta_tensor = torch.from_numpy(delta).reshape(-1, 6).double()
                trial = current.clone()
                trial[1:] = (
                    pp.se3(delta_tensor).Exp() @ pp.SE3(current[1:])
                ).tensor()
                try:
                    trial_jacobian, trial_residual, trial_loss = (
                        self._sparse_linearization(trial)
                    )
                except (RuntimeError, ValueError):
                    trial_loss = math.inf
                if math.isfinite(trial_loss) and trial_loss < current_loss:
                    current = trial
                    current_loss = trial_loss
                    jacobian, residual = trial_jacobian, trial_residual
                    damping = max(
                        damping / self.sparse_damping_down,
                        self.sparse_min_damping,
                    )
                    diagnostics["accepted_steps"] += 1
                    accepted = True
                    translation_step = float(
                        torch.linalg.vector_norm(delta_tensor[:, :3], dim=-1).max()
                    )
                    rotation_step = float(
                        torch.linalg.vector_norm(delta_tensor[:, 3:], dim=-1).max()
                    )
                    if translation_step <= 1e-6 and rotation_step <= 1e-6:
                        convergence_reason = "step_tolerance"
                    elif (
                        previous_loss - current_loss
                        <= max(abs(previous_loss), 1.0) * 1e-9
                    ):
                        convergence_reason = "relative_loss_tolerance"
                    break
                damping = min(
                    damping * self.sparse_damping_up,
                    self.sparse_max_damping,
                )
                diagnostics["rejected_trials"] += 1
            if not accepted:
                convergence_reason = "no_decreasing_trial"
                break
            if convergence_reason in {"step_tolerance", "relative_loss_tolerance"}:
                break
        diagnostics.update({
            "safe": (
                torch.isfinite(current).all().item()
                and torch.equal(current[0], base[0])
                and current_loss <= float(diagnostics["initial_robust_loss"]) + 1e-9
            ),
            "final_robust_loss": current_loss,
            "final_damping": damping,
            "convergence_reason": convergence_reason,
        })
        if not diagnostics["safe"]:
            return base.float()
        diagnostics["trajectory_source"] = "optimized"
        return current.float()

    def optimize_global(self, global_map: VisualMap) -> torch.Tensor:
        poses = self._poses_from_map(global_map)
        return self.optimize_poses(poses)

    def write_back(self, global_map: VisualMap, optimized_poses: torch.Tensor) -> None:
        num_frames = len(global_map.frames)
        current = global_map.frames.data["pose"][:num_frames]
        if optimized_poses.shape != current.shape:
            raise ValueError(
                f"Optimized poses shape {tuple(optimized_poses.shape)} does not match map pose shape {tuple(current.shape)}"
            )
        global_map.frames.data["pose"][:num_frames] = optimized_poses.to(device=current.device, dtype=current.dtype)

    def run_on_terminate(self, global_map: VisualMap) -> None:
        if not self.enabled or not self.optimize_on_terminate:
            return
        self.register_odometry_edges(global_map)
        optimized_poses = self.optimize_global(global_map)
        self.write_back(global_map, optimized_poses)
