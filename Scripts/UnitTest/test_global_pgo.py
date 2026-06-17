from types import SimpleNamespace

import pytest
import torch

pp = pytest.importorskip("pypose")

from Module.Optimization.GlobalPGO import GlobalPoseGraphOptimizer, relative_pose


class FakeFrames:
    def __init__(self, poses: torch.Tensor, need_interp: torch.Tensor | None = None) -> None:
        self.data = {"pose": poses.clone()}
        if need_interp is not None:
            self.data["need_interp"] = need_interp.clone()

    def __len__(self) -> int:
        return int(self.data["pose"].size(0))


class FakeMap:
    def __init__(self, poses: torch.Tensor, need_interp: torch.Tensor | None = None) -> None:
        self.frames = FakeFrames(poses, need_interp)


def make_config(**overrides) -> SimpleNamespace:
    values = {
        "enabled": True,
        "optimize_on_terminate": True,
        "max_iterations": 25,
        "trans_weight": 100.0,
        "rot_weight": 100.0,
        "device": "cpu",
        "include_interp_frames": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_pose_chain(num_poses: int) -> torch.Tensor:
    poses = torch.zeros((num_poses, 7), dtype=torch.float32)
    poses[:, 0] = torch.arange(num_poses, dtype=torch.float32)
    poses[:, 6] = 1.0
    return poses


def translation_error(poses: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (poses[:, :3] - target[:, :3]).norm(dim=-1).mean()


def test_register_odometry_edges_and_consistent_loss_zero() -> None:
    poses = make_pose_chain(5)
    optimizer = GlobalPoseGraphOptimizer(make_config())

    optimizer.register_odometry_edges(FakeMap(poses))

    assert [(edge.src, edge.dst, edge.edge_type) for edge in optimizer.edges] == [
        (0, 1, "odometry"),
        (1, 2, "odometry"),
        (2, 3, "odometry"),
        (3, 4, "odometry"),
    ]
    assert optimizer.compute_loss(poses).item() < 1e-8

    optimizer.register_odometry_edges(FakeMap(poses))
    assert len(optimizer.edges) == 4


def test_include_interp_frames_changes_registered_edges() -> None:
    poses = make_pose_chain(3)
    need_interp = torch.tensor([False, True, False])

    skip_interp = GlobalPoseGraphOptimizer(make_config(include_interp_frames=False))
    skip_interp.register_odometry_edges(FakeMap(poses, need_interp))
    assert [(edge.src, edge.dst) for edge in skip_interp.edges] == [(0, 2)]

    include_interp = GlobalPoseGraphOptimizer(make_config(include_interp_frames=True))
    include_interp.register_odometry_edges(FakeMap(poses, need_interp))
    assert [(edge.src, edge.dst) for edge in include_interp.edges] == [(0, 1), (1, 2)]


def test_loop_edge_participates_in_loss_and_validates_indices() -> None:
    poses = make_pose_chain(5)
    perturbed = poses.clone()
    perturbed[-1, 0] += 0.5

    optimizer = GlobalPoseGraphOptimizer(make_config())
    optimizer.register_odometry_edges(FakeMap(poses))
    loss_without_loop = optimizer.compute_loss(perturbed)

    loop_relative = relative_pose(pp.SE3(poses[0]), pp.SE3(poses[-1]))
    optimizer.add_loop_edge(0, 4, loop_relative)
    loss_with_loop = optimizer.compute_loss(perturbed)

    assert loss_with_loop > loss_without_loop

    with pytest.raises(IndexError):
        optimizer.add_loop_edge(-1, 4, loop_relative)
    with pytest.raises(IndexError):
        optimizer.add_loop_edge(0, 5, loop_relative)


def test_optimize_poses_reduces_loss_and_keeps_first_frame_fixed() -> None:
    target = make_pose_chain(5)
    perturbed = target.clone()
    perturbed[1:, 0] += torch.tensor([0.2, -0.3, 0.4, -0.2])

    optimizer = GlobalPoseGraphOptimizer(make_config(max_iterations=30))
    optimizer.register_odometry_edges(FakeMap(target))
    optimizer.add_loop_edge(0, 4, relative_pose(pp.SE3(target[0]), pp.SE3(target[4])))

    loss_before = optimizer.compute_loss(perturbed)
    optimized = optimizer.optimize_poses(perturbed)
    loss_after = optimizer.compute_loss(optimized)

    assert loss_after < loss_before
    assert translation_error(optimized, target) < translation_error(perturbed, target)
    assert torch.allclose(optimized[0], perturbed[0])


def test_zero_residual_graph_returns_initial_poses() -> None:
    poses = make_pose_chain(5)
    optimizer = GlobalPoseGraphOptimizer(make_config(max_iterations=30))
    optimizer.register_odometry_edges(FakeMap(poses))

    optimized = optimizer.optimize_poses(poses)

    assert torch.allclose(optimized, poses)


def test_write_back_preserves_pose_shape_and_dtype() -> None:
    poses = make_pose_chain(4)
    global_map = FakeMap(poses)
    optimizer = GlobalPoseGraphOptimizer(make_config())
    optimized = poses.double()
    optimized[:, 0] += 1.0

    optimizer.write_back(global_map, optimized)

    assert global_map.frames.data["pose"].shape == poses.shape
    assert global_map.frames.data["pose"].dtype == torch.float32
    assert torch.allclose(global_map.frames.data["pose"], optimized.float())
