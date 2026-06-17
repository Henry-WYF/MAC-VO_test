from __future__ import annotations

from dataclasses import dataclass

import torch
import pypose as pp


@dataclass
class PoseGraphEdge:
    src: int
    dst: int
    relative_pose: pp.LieTensor
    information: torch.Tensor
    edge_type: str


def make_information(
    trans_weight: float,
    rot_weight: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.double,
) -> torch.Tensor:
    weights = torch.tensor(
        [trans_weight, trans_weight, trans_weight, rot_weight, rot_weight, rot_weight],
        device=device,
        dtype=dtype,
    )
    return torch.diag(weights)


def as_se3(
    value: pp.LieTensor | torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> pp.LieTensor:
    if isinstance(value, pp.LieTensor):
        pose = value
    else:
        tensor = value
        if tensor.ndim == 2 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.shape[-1] != 7:
            raise ValueError(f"SE3 pose must have last dimension 7, got shape {tuple(tensor.shape)}")
        pose = pp.SE3(tensor)
    if device is not None and dtype is not None:
        pose = pose.to(device=device, dtype=dtype)
    elif device is not None:
        pose = pose.to(device)
    elif dtype is not None:
        pose = pose.to(dtype=dtype)
    return pose


def relative_pose(T_w_i: pp.LieTensor | torch.Tensor, T_w_j: pp.LieTensor | torch.Tensor) -> pp.LieTensor:
    return as_se3(T_w_i).Inv() @ as_se3(T_w_j)


def compute_edge_residual(edge: PoseGraphEdge, poses: pp.LieTensor | torch.Tensor) -> torch.Tensor:
    pose_seq = as_se3(poses, edge.information.device, edge.information.dtype)
    if edge.src < 0 or edge.dst < 0 or edge.src >= pose_seq.shape[0] or edge.dst >= pose_seq.shape[0]:
        raise IndexError(
            f"Pose graph edge ({edge.src}, {edge.dst}) is out of range for {pose_seq.shape[0]} poses"
        )

    T_w_i = pose_seq[edge.src]
    T_w_j = pose_seq[edge.dst]
    T_i_j = edge.relative_pose.to(device=edge.information.device, dtype=edge.information.dtype)
    return (T_i_j.Inv() @ (T_w_i.Inv() @ T_w_j)).Log().tensor()
