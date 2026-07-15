from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypose as pp
import pytest
import torch

from DataLoader import StereoData, StereoFrame
from Module.Frontend.StereoDepth import IStereoDepth
from Module.LoopClosure import (
    BinaryVocabulary,
    CausalBoWDatabase,
    CausalRetrievalController,
    FrameIdentity,
    LoopClosureManager,
    LoopFrameRecord,
    ORBPlaceRecognizer,
)
from Module.Optimization.GlobalPGO import PoseGraphEdge, compute_edge_residual, make_information, relative_pose
from Scripts.LoopClosure.evaluate_loop_retrieval import common_query_ids, compute_metrics


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


class _DescriptorBackend:
    def __init__(self) -> None:
        self.entries: list[np.ndarray] = []

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    def add(self, descriptors) -> int:
        entry_id = len(self.entries)
        self.entries.append(np.asarray(descriptors).copy())
        return entry_id

    def query(self, descriptors) -> list[tuple[int, float]]:
        value = int(np.asarray(descriptors)[0, 0])
        return [(entry_id, 1.0 if int(candidate[0, 0]) == value else 0.1) for entry_id, candidate in enumerate(self.entries)]

    def metadata(self) -> dict:
        return {"type": "test"}


def _descriptors(value: int) -> torch.Tensor:
    return torch.full((1, 32), value, dtype=torch.uint8)


def test_causal_controller_excludes_time_before_backend_query() -> None:
    backend = _DescriptorBackend()
    controller = CausalRetrievalController(backend, 50, top_k=10, min_score=None)
    first = controller.query_then_enqueue(FrameIdentity(0, 0, 0), _descriptors(1))
    recent = controller.query_then_enqueue(FrameIdentity(40, 1, 1), _descriptors(2))
    current = controller.query_then_enqueue(FrameIdentity(60, 2, 2), _descriptors(1))
    assert first.eligible_database_entries == 0
    assert recent.eligible_database_entries == 0
    assert current.eligible_database_entries == 1
    assert [item.sensor_frame_idx for item in current.candidates] == [0]
    assert backend.entry_count == 1


def test_causal_controller_empty_descriptors_are_never_candidates() -> None:
    backend = _DescriptorBackend()
    controller = CausalRetrievalController(backend, 10, top_k=10, min_score=None)
    empty = torch.empty((0, 32), dtype=torch.uint8)
    result = controller.query_then_enqueue(FrameIdentity(0, 0, 0), empty)
    later = controller.query_then_enqueue(FrameIdentity(20, 1, 1), _descriptors(3))
    assert result.empty_descriptors
    assert result.query_time_ms is None
    assert later.eligible_database_entries == 0
    assert backend.entry_count == 0


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


def test_missing_vocabulary_disables_retrieval_but_preserves_cache(tmp_path: Path) -> None:
    manager = LoopClosureManager(make_config(tmp_path / "missing.npz"))
    manager.set_output_dir(tmp_path / "cache")
    pose = pp.identity_SE3(1).tensor()
    assert manager.register_loop_frame(make_stereo(0), make_depth(), 0, pose)
    assert manager.detect_all() == []
    assert manager.cache_enabled
    assert not manager.retrieval_enabled
    assert not manager.geometry_enabled
    assert manager.enabled


def test_manager_writes_schema_two_from_cached_descriptors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = LoopClosureManager(make_config(tmp_path / "unused.npz"))
    cache = tmp_path / "cache"
    manager.set_output_dir(cache)
    pose = pp.identity_SE3(1).tensor()
    assert manager.register_loop_frame(make_stereo(0), make_depth(), 0, pose)
    assert manager.register_loop_frame(make_stereo(60), make_depth(), 1, pose)
    assert all(not item["has_bow"] for item in manager.records)
    backend = _DescriptorBackend()
    monkeypatch.setattr(manager, "_make_backend", lambda: backend)
    queries = manager.detect_all()
    payload = __import__("json").loads((cache / "queries.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["recognizer"]["type"] == "test"
    assert len(queries) == 2
    assert queries[1]["eligible_database_entries"] == 1
    assert queries[1]["returned_candidates"] == 1


def test_phase_a_query_failure_preserves_cache_and_disables_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = LoopClosureManager(make_config(tmp_path / "unused.npz"))
    manager.set_output_dir(tmp_path / "cache")
    pose = pp.identity_SE3(1).tensor()
    assert manager.register_loop_frame(make_stereo(0), make_depth(), 0, pose)
    backend = _DescriptorBackend()
    monkeypatch.setattr(backend, "query", lambda _: (_ for _ in ()).throw(RuntimeError("query failed")))
    monkeypatch.setattr(manager, "_make_backend", lambda: backend)
    assert manager.detect_all() == []
    assert manager.cache_enabled
    assert not manager.retrieval_enabled
    assert not manager.geometry_enabled


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan")])
def test_bow_min_score_validation_rejects_invalid_values(tmp_path: Path, value) -> None:
    with pytest.raises(ValueError):
        LoopClosureManager.is_valid_config(make_config(tmp_path / "unused.npz", bow_min_score=value))


def test_dbow2_native_binding_with_tiny_text_vocabulary(tmp_path: Path) -> None:
    native = pytest.importorskip("macvo_dbow2")
    zero = " ".join(["0"] * 32)
    full = " ".join(["255"] * 32)
    vocabulary = tmp_path / "tiny_orbvoc.txt"
    vocabulary.write_text(
        f"2 1 0 0\n0 1 {zero} 1.0\n0 1 {full} 1.0",
        encoding="utf-8",
    )
    database = native.OrbDatabase(str(vocabulary))
    descriptors = np.zeros((8, 32), dtype=np.uint8)
    entry_id = database.add(descriptors)
    results = database.query(descriptors)
    assert entry_id == 0
    assert results[0][0] == 0
    assert results[0][1] > 0.0
    assert database.metadata()["branching"] == 2
    assert native.orb_slam3_commit == "0df83dde1c85c7ab91a0d47de7a29685d046f637"


def test_false_candidate_average_uses_common_fixed_query_set() -> None:
    custom = {
        10: {
            "loop_frame_idx": 10, "empty_descriptors": False,
            "eligible_database_entries": 3,
            "candidates": [{"loop_frame_idx": 1, "sensor_frame_idx": 0, "score": 0.9}],
        },
        11: {
            "loop_frame_idx": 11, "empty_descriptors": False,
            "eligible_database_entries": 4, "candidates": [],
        },
    }
    dbow2 = {
        10: {
            "loop_frame_idx": 10, "empty_descriptors": False,
            "eligible_database_entries": 3, "candidates": [],
        },
        11: {
            "loop_frame_idx": 11, "empty_descriptors": False,
            "eligible_database_entries": 4, "candidates": [],
        },
    }
    labels = {"10:1": "false", "10:2": "true"}
    fixed = common_query_ids(custom, dbow2, labels)
    assert fixed == [10, 11]
    custom_metrics = compute_metrics(custom, fixed, labels, None)
    dbow2_metrics = compute_metrics(dbow2, fixed, labels, None)
    assert custom_metrics["false_candidates_total"] == 1
    assert custom_metrics["false_candidates_per_fixed_query"] == pytest.approx(0.5)
    assert dbow2_metrics["false_candidates_total"] == 0
    assert dbow2_metrics["false_candidates_per_fixed_query"] == pytest.approx(0.0)
    assert dbow2_metrics["no_candidate_queries"] == 2


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
