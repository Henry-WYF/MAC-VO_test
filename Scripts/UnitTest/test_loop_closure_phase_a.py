from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypose as pp
import pytest
import torch

from DataLoader import StereoData, StereoFrame
from Module.Frontend.StereoDepth import IStereoDepth
from Module.LoopClosure import BinaryVocabulary, CausalBoWDatabase, LoopClosureManager, LoopFrameRecord, ORBPlaceRecognizer
from Module.Optimization.GlobalPGO import PoseGraphEdge, compute_edge_residual, make_information, relative_pose


def make_stereo(sensor_idx: int, height: int = 96, width: int = 128) -> StereoFrame:
    generator = torch.Generator().manual_seed(sensor_idx + 7)
    image = torch.rand((1, 3, height, width), generator=generator)
    stereo = StereoData(
        T_BS=pp.identity_SE3(1), K=torch.eye(3).unsqueeze(0), baseline=torch.tensor([0.2]),
        time_ns=[sensor_idx * 1_000_000], height=height, width=width,
        imageL=image, imageR=image.flip(-1), gt_flow=None, flow_mask=None, gt_depth=None,
    )
    return StereoFrame(idx=[sensor_idx], time_ns=[sensor_idx * 1_000_000], gt_pose=None, stereo=stereo)


def make_depth(height: int = 96, width: int = 128) -> IStereoDepth.Output:
    depth = torch.ones((1, 1, height, width), dtype=torch.float32)
    covariance = torch.full_like(depth, 0.1)
    return IStereoDepth.Output(depth=depth, cov=covariance)


def make_config(vocabulary_path: Path, **overrides) -> SimpleNamespace:
    values = dict(
        enabled=True, vocabulary_path=str(vocabulary_path), keyframe_stride_sensor_frames=10,
        temporal_exclusion_sensor_frames=50, cache_failure_policy="disable_loop",
        orb_nfeatures=1000, orb_scale_factor=1.2, orb_nlevels=8, top_k=10,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_loop_frame_roundtrip_reconstructs_stereo_data(tmp_path: Path) -> None:
    frame = make_stereo(3)
    record = LoopFrameRecord(
        sensor_frame_idx=3, visual_map_idx=2, loop_frame_idx=0, frame_ns=frame.stereo.frame_ns,
        height=frame.stereo.height, width=frame.stereo.width, image_left=frame.stereo.imageL,
        image_right=frame.stereo.imageR, intrinsic=frame.stereo.K, baseline=frame.stereo.baseline,
        body_to_sensor=frame.stereo.T_BS.tensor(), depth=make_depth().depth,
        depth_covariance=make_depth().cov, registered_pose=pp.identity_SE3(1).tensor(),
        orb_keypoints=torch.empty((0, 7)), orb_descriptors=torch.empty((0, 32), dtype=torch.uint8),
        bow_vector=None,
    )
    path = tmp_path / "frame.pt"
    record.save(path)
    loaded = LoopFrameRecord.load(path)
    reconstructed = loaded.to_stereo_data("cpu")
    assert reconstructed.imageL.shape == (1, 3, 96, 128)
    assert reconstructed.imageR.shape == reconstructed.imageL.shape
    assert reconstructed.K.shape == (1, 3, 3)
    assert reconstructed.T_BS.shape == (1, 7)
    assert torch.equal(reconstructed.imageL, frame.stereo.imageL)
    assert torch.equal(loaded.depth_covariance, record.depth_covariance)


def test_orb_extraction_is_repeatable(tmp_path: Path) -> None:
    recognizer = ORBPlaceRecognizer(tmp_path / "missing.npz")
    frame = make_stereo(4)
    points_a, descriptors_a = recognizer.extract(frame.stereo.imageL)
    points_b, descriptors_b = recognizer.extract(frame.stereo.imageL)
    assert torch.equal(points_a, points_b)
    assert torch.equal(descriptors_a, descriptors_b)


def test_vocabulary_save_load_and_similarity(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    base = rng.integers(0, 256, size=(24, 32), dtype=np.uint8)
    documents = [base, base.copy(), np.bitwise_xor(base, np.uint8(255))]
    vocabulary = BinaryVocabulary.train(documents, branch=2, depth=2, max_iterations=5)
    path = tmp_path / "vocabulary.npz"
    vocabulary.save(path)
    loaded = BinaryVocabulary.load(path)
    assert loaded.checksum() == vocabulary.checksum()
    assert float(np.dot(loaded.transform(base), loaded.transform(base.copy()))) == pytest.approx(1.0)


def test_causal_database_excludes_recent_and_future_frames() -> None:
    database = CausalBoWDatabase(temporal_exclusion_sensor_frames=50)
    vector = torch.tensor([1.0, 0.0])
    database.add(0, 0, 0, vector)
    database.add(80, 8, 1, vector)
    results, _ = database.query(100, vector, top_k=10)
    assert [result.sensor_frame_idx for result in results] == [0]
    database.add(120, 12, 2, vector)
    with pytest.raises(AssertionError):
        database.query(110, vector, top_k=10)


def test_manager_registration_stride_and_cache_failure_policy(tmp_path: Path) -> None:
    manager = LoopClosureManager(make_config(tmp_path / "missing.npz"))
    manager.set_output_dir(tmp_path / "cache")
    pose = pp.identity_SE3(1).tensor()
    assert manager.register_loop_frame(make_stereo(0), make_depth(), 0, pose)
    assert not manager.register_loop_frame(make_stereo(5), make_depth(), 1, pose)
    assert manager.register_loop_frame(make_stereo(10), make_depth(), 2, pose)
    assert [item["sensor_frame_idx"] for item in manager.records] == [0, 10]

    failing = LoopClosureManager(make_config(tmp_path / "missing.npz"))
    assert not failing.register_loop_frame(make_stereo(0), make_depth(), 0, pose)
    assert not failing.enabled
    assert "cache failure" in str(failing.disabled_reason)


def test_loop_constraint_relative_pose_direction_matches_global_pgo() -> None:
    src = pp.SE3(torch.tensor([1.0, -0.5, 0.2, 0.0, 0.0, 0.258819, 0.965926]))
    dst = pp.SE3(torch.tensor([-0.3, 2.0, 1.1, 0.0, 0.573576, 0.0, 0.819152]))
    poses = torch.stack([src.tensor(), dst.tensor()], dim=0)
    information = make_information(1.0, 1.0, dtype=torch.float64)

    edge_relative = relative_pose(src, dst)
    edge = PoseGraphEdge(0, 1, edge_relative, information, "loop")
    assert torch.linalg.vector_norm(compute_edge_residual(edge, poses)).item() < 1e-6

    pnp_direction = edge_relative.Inv()
    wrong_edge = PoseGraphEdge(0, 1, pnp_direction, information, "loop")
    assert torch.linalg.vector_norm(compute_edge_residual(wrong_edge, poses)).item() > 1e-3
