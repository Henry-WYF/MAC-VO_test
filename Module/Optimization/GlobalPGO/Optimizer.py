from __future__ import annotations

from types import SimpleNamespace

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

        self.edges: list[PoseGraphEdge] = []
        self._edge_keys: set[tuple[int, int, str]] = set()
        self._num_frames: int | None = None
        self._latest_poses: torch.Tensor | None = None

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config, {
            "enabled": lambda b: isinstance(b, bool),
            "optimize_on_terminate": lambda b: isinstance(b, bool),
            "max_iterations": lambda n: isinstance(n, int) and n >= 0,
            "trans_weight": lambda v: isinstance(v, (float, int)) and v > 0.0,
            "rot_weight": lambda v: isinstance(v, (float, int)) and v > 0.0,
            "device": lambda v: isinstance(v, str) and (v == "cpu" or "cuda" in v),
            "include_interp_frames": lambda b: isinstance(b, bool),
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

    def register_odometry_edges(self, global_map: VisualMap) -> None:
        poses = self._poses_from_map(global_map)
        valid_indices = self._valid_frame_indices(global_map)
        if len(valid_indices) < 2:
            return

        pose_se3 = pp.SE3(poses)
        for src, dst in zip(valid_indices[:-1], valid_indices[1:]):
            rel_pose = compute_relative_pose(pose_se3[src], pose_se3[dst])
            self._add_edge(
                self._make_edge(src, dst, rel_pose, None, "odometry"),
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
            losses.append(residual.unsqueeze(0) @ info @ residual.unsqueeze(-1))
        return torch.stack(losses).sum()

    def _apply_pose_delta(self, base_poses: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if base_poses.size(0) <= 1:
            return base_poses
        updated_tail = pp.se3(delta).Exp() @ pp.SE3(base_poses[1:])
        return torch.cat([base_poses[:1], updated_tail.tensor()], dim=0)

    def optimize_poses(self, initial_poses: torch.Tensor) -> torch.Tensor:
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
