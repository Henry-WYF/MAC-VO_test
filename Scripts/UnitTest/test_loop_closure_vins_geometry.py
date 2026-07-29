from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pypose as pp
import pytest
import torch

from Module.LoopClosure.Recognizer import ORBFeatureExtractor
from Module.LoopClosure.Record import GeometryFeatureRecord
from Module.LoopClosure.Record import LoopFrameRecord
from Module.LoopClosure.VINSGeometry import (
    GeometryResult,
    _cv_pose_to_ned,
    _ned_pose_to_cv,
    _strict_information,
    cached_orb_geometry,
    fixed_point_covariance,
    match_fixed_descriptors,
    match_orbslam_descriptors,
    refine_geometry_with_network,
    run_pose_copy_pgo_comparison,
    run_pose_copy_pgo_safety,
    verify_fixed_geometry,
)
from Module.LoopClosure.Verification import LoopConstraint
from Scripts.AdHoc.RunLoopPhaseBOffline import (
    run_vins_pose_copy_pgo,
    summarize_engineering_admission,
)
from Module.Map import VisualMap
from Module.Optimization.ObservationInformation import (
    information_diagnostics,
    robust_observation_system,
)
from Utility.Point import pixel2point_NED


def geometry_record(descriptors: torch.Tensor, original: torch.Tensor | None = None) -> GeometryFeatureRecord:
    count = len(descriptors)
    original = torch.arange(count) if original is None else original
    return GeometryFeatureRecord(
        sensor_frame_idx=10, visual_map_idx=1, loop_frame_idx=1, orb_config_sha256="abc",
        original_index=original, pixel_uv=torch.zeros((count, 2)),
        point_camera=torch.ones((count, 3)),
        point_covariance_camera=torch.eye(3).repeat(count, 1, 1),
        disparity=torch.ones(count), disparity_variance=torch.ones(count),
        disparity_valid=torch.ones(count, dtype=torch.bool), descriptor=descriptors,
    )


def loop_frame(sensor: int, visual: int, loop: int, K: torch.Tensor) -> LoopFrameRecord:
    return LoopFrameRecord(
        sensor_frame_idx=sensor, visual_map_idx=visual, loop_frame_idx=loop,
        frame_ns=sensor, height=80, width=100, image_left=torch.zeros((1, 3, 80, 100)),
        image_right=torch.zeros((1, 3, 80, 100)), intrinsic=K, baseline=torch.tensor([0.2]),
        body_to_sensor=pp.identity_SE3(1).tensor(), depth=torch.ones((1, 1, 80, 100)),
        depth_covariance=torch.ones((1, 1, 80, 100)), registered_pose=pp.identity_SE3(1).tensor(),
        orb_keypoints=torch.empty((0, 7)), orb_descriptors=torch.empty((0, 32), dtype=torch.uint8),
        bow_vector=None,
    )


def pnp_geometry(count: int, sensor: int, visual: int, loop: int, translated_x: float) -> GeometryFeatureRecord:
    x = torch.linspace(2.0, 5.0, count)
    y = torch.linspace(-0.7, 0.7, count)
    z = torch.sin(torch.linspace(0.0, 3.0, count)) * 0.4
    points = torch.stack([x, y, z], dim=-1)
    current_x = x + translated_x
    pixels = torch.stack([100.0 * y / current_x + 50.0, 100.0 * z / current_x + 40.0], dim=-1)
    descriptors = torch.zeros((count, 32), dtype=torch.uint8)
    descriptors[:, 0] = torch.arange(count, dtype=torch.uint8)
    return GeometryFeatureRecord(
        sensor_frame_idx=sensor, visual_map_idx=visual, loop_frame_idx=loop,
        orb_config_sha256="abc", original_index=torch.arange(count), pixel_uv=pixels,
        point_camera=points, point_covariance_camera=torch.eye(3).repeat(count, 1, 1) * 1e-3,
        disparity=20.0 / current_x, disparity_variance=torch.full((count,), 0.1),
        disparity_valid=torch.ones(count, dtype=torch.bool), descriptor=descriptors,
    )


def refinement_geometry(sensor: int, visual: int, loop: int) -> GeometryFeatureRecord:
    pixel = torch.tensor([
        [30., 25.], [40., 25.], [50., 25.], [60., 25.],
        [35., 35.], [45., 35.], [55., 35.], [65., 35.],
    ])
    depth = torch.linspace(2.0, 4.8, len(pixel))
    point = torch.stack([
        depth,
        (pixel[:, 0] - 50.0) * depth / 100.0,
        (pixel[:, 1] - 40.0) * depth / 100.0,
    ], dim=-1)
    return GeometryFeatureRecord(
        sensor_frame_idx=sensor,
        visual_map_idx=visual,
        loop_frame_idx=loop,
        orb_config_sha256="abc",
        original_index=torch.arange(len(pixel)),
        pixel_uv=pixel,
        point_camera=point,
        point_covariance_camera=(
            torch.eye(3).repeat(len(pixel), 1, 1) * 1e-3
        ),
        disparity=20.0 / depth,
        disparity_variance=torch.full((len(pixel),), 0.1),
        disparity_valid=torch.ones(len(pixel), dtype=torch.bool),
        descriptor=torch.zeros((len(pixel), 32), dtype=torch.uint8),
    )


class SyntheticRefinementFrontend:
    def __init__(
        self,
        geometry: GeometryFeatureRecord,
        *,
        invalid_covariance_index: int | None = None,
    ) -> None:
        self.config = SimpleNamespace(device="cpu")
        self.calls = 0
        self.flow = torch.zeros((1, 2, 80, 100))
        self.covariance = torch.zeros((1, 3, 80, 100))
        self.covariance[:, :2] = 0.25
        self.disparity = torch.ones((1, 1, 80, 100))
        self.disparity_variance = torch.full((1, 1, 80, 100), 0.1)
        for index, (uv, disparity) in enumerate(
            zip(geometry.pixel_uv, geometry.disparity)
        ):
            u, v = torch.floor(uv).long()
            self.disparity[0, 0, v, u] = disparity
            if invalid_covariance_index == index:
                self.covariance[0, 0, v, u] = -1e6

    def estimate_pair(self, _historical, _current):
        self.calls += 1
        depth = SimpleNamespace(
            disparity=self.disparity,
            disparity_uncertainty=self.disparity_variance,
        )
        match = SimpleNamespace(
            flow=self.flow,
            cov=self.covariance,
            mask=None,
        )
        return depth, match


def refinement_input() -> tuple[
    GeometryResult,
    LoopFrameRecord,
    LoopFrameRecord,
    GeometryFeatureRecord,
    GeometryFeatureRecord,
]:
    K = torch.tensor(
        [[[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]]],
    )
    current = refinement_geometry(20, 2, 2)
    historical = refinement_geometry(10, 1, 1)
    current_frame = loop_frame(20, 2, 2, K)
    historical_frame = loop_frame(10, 1, 1, K)
    identity = pp.identity_SE3(1).double()
    constraint = LoopConstraint(
        src_visual_map_idx=1,
        dst_visual_map_idx=2,
        src_sensor_frame_idx=10,
        dst_sensor_frame_idx=20,
        pnp_relative_pose=identity.tensor().tolist(),
        relative_pose=identity.tensor().tolist(),
        information=(torch.eye(6) * 100.0).tolist(),
        bow_score=1.0,
        num_flow_points=0,
        num_geometry_points=len(current.pixel_uv),
        num_pnp_inliers=len(current.pixel_uv),
        inlier_ratio=1.0,
        mean_reproj_error_px=0.0,
        rotation_diff_deg=0.0,
        translation_diff_m=0.0,
        status="accepted",
    )
    result = GeometryResult(
        row={"geometry_accepted": True},
        constraint=constraint,
        covariance_information=None,
        pnp_pose_current_candidate=identity,
        pnp_current_local=torch.arange(len(current.pixel_uv)),
        pnp_candidate_local=torch.arange(len(current.pixel_uv)),
    )
    return result, current_frame, historical_frame, current, historical


def test_geometry_sidecar_first_write_is_atomic_and_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "geometry_features_v1" / "loop_000001.pt"
    first = geometry_record(torch.zeros((2, 32), dtype=torch.uint8))
    second = geometry_record(torch.ones((2, 32), dtype=torch.uint8))
    assert first.save_if_absent(path) is True
    assert second.save_if_absent(path) is False
    loaded = GeometryFeatureRecord.load(path)
    assert torch.equal(loaded.descriptor, first.descriptor)
    assert not path.with_suffix(".pt.tmp").exists()


def test_one_way_hamming_is_strict_and_candidate_unique() -> None:
    candidate = geometry_record(torch.stack([
        torch.zeros(32, dtype=torch.uint8),
        torch.full((32,), 255, dtype=torch.uint8),
    ]), torch.tensor([20, 10]))
    current_desc = torch.stack([
        torch.zeros(32, dtype=torch.uint8),
        torch.cat([torch.tensor([1], dtype=torch.uint8), torch.zeros(31, dtype=torch.uint8)]),
        torch.full((32,), 255, dtype=torch.uint8),
    ])
    current = geometry_record(current_desc, torch.tensor([3, 2, 1]))
    matches = match_fixed_descriptors(current, candidate, 80)
    # Equal-distance matches are ordered by stable original indices, not local storage order.
    assert [(item.current_local, item.candidate_local) for item in matches] == [(2, 1), (0, 0)]
    exactly_80 = geometry_record(torch.cat([
        torch.full((1, 10), 255, dtype=torch.uint8), torch.zeros((1, 22), dtype=torch.uint8)
    ], dim=1))
    assert match_fixed_descriptors(exactly_80, geometry_record(torch.zeros((1, 32), dtype=torch.uint8)), 80) == []


def bit_descriptor(*bits: int) -> torch.Tensor:
    descriptor = torch.zeros(32, dtype=torch.uint8)
    for bit in bits:
        descriptor[bit // 8] |= 1 << (bit % 8)
    return descriptor


def test_orbslam_distance_and_ratio_boundaries() -> None:
    current = geometry_record(torch.zeros((1, 32), dtype=torch.uint8))
    accepted_candidate = geometry_record(torch.stack([
        bit_descriptor(*range(50)), bit_descriptor(*range(100)),
    ]))
    accepted, diagnostics = match_orbslam_descriptors(
        current, accepted_candidate, torch.tensor([0.0]), torch.tensor([0.0, 0.0]),
    )
    assert len(accepted) == 1
    assert diagnostics["after_distance"] == diagnostics["after_ratio"] == 1

    rejected_candidate = geometry_record(torch.stack([
        bit_descriptor(*range(51)), bit_descriptor(*range(100)),
    ]))
    rejected, diagnostics = match_orbslam_descriptors(
        current, rejected_candidate, torch.tensor([0.0]), torch.tensor([0.0, 0.0]),
    )
    assert rejected == []
    assert diagnostics["after_distance"] == 0

    ratio_boundary = geometry_record(torch.stack([
        bit_descriptor(*range(45)), bit_descriptor(*range(50)),
    ]))
    rejected, diagnostics = match_orbslam_descriptors(
        current, ratio_boundary, torch.tensor([0.0]), torch.tensor([0.0, 0.0]),
    )
    assert rejected == []
    assert diagnostics["after_distance"] == 1
    assert diagnostics["after_ratio"] == 0


def test_orbslam_candidate_unique_and_orientation_filter_are_deterministic() -> None:
    descriptors = torch.stack([bit_descriptor(index) for index in range(13)])
    current = geometry_record(descriptors.clone(), torch.arange(13))
    candidate = geometry_record(descriptors.clone(), torch.arange(13))
    current_angles = torch.tensor([0.0] * 11 + [12.0, 24.0])
    matches, diagnostics = match_orbslam_descriptors(
        current, candidate, current_angles, torch.zeros(13),
    )
    assert diagnostics["after_candidate_unique"] == 13
    assert diagnostics["after_valid_orientation"] == 13
    assert diagnostics["selected_orientation_bins"] == [0]
    assert diagnostics["after_orientation_histogram"] == 11
    assert [item.current_original for item in matches] == list(range(11))

    duplicate_current = geometry_record(torch.stack([
        bit_descriptor(0), bit_descriptor(0, 1),
    ]), torch.tensor([5, 3]))
    duplicate_candidate = geometry_record(torch.stack([
        bit_descriptor(0), bit_descriptor(*range(20, 80)),
    ]), torch.tensor([9, 8]))
    unique, diagnostics = match_orbslam_descriptors(
        duplicate_current, duplicate_candidate,
        torch.zeros(6), torch.zeros(10),
    )
    assert diagnostics["after_ratio"] == 2
    assert diagnostics["after_candidate_unique"] == 1
    assert unique[0].current_original == 5


def test_orbslam_orientation_wrap_half_bin_and_invalid_point_filter() -> None:
    descriptors = torch.stack([bit_descriptor(index) for index in range(3)])
    record = geometry_record(descriptors)
    matches, diagnostics = match_orbslam_descriptors(
        record, record,
        torch.tensor([359.0, 6.0, -1.0]), torch.tensor([5.0, 0.0, 0.0]),
    )
    assert diagnostics["invalid_orientation_matches"] == 1
    assert diagnostics["after_valid_orientation"] == 2
    assert diagnostics["selected_orientation_bins"] == [0, 1]
    assert len(matches) == 2

    four = geometry_record(torch.stack([bit_descriptor(index) for index in range(4)]))
    _, tied = match_orbslam_descriptors(
        four, four, torch.tensor([0.0, 12.0, 24.0, 36.0]), torch.zeros(4),
    )
    assert tied["selected_orientation_bins"] == [0, 1, 2]
    assert tied["after_orientation_histogram"] == 3


def test_orbslam_requires_a_second_candidate_descriptor() -> None:
    matches, diagnostics = match_orbslam_descriptors(
        geometry_record(torch.zeros((1, 32), dtype=torch.uint8)),
        geometry_record(torch.zeros((1, 32), dtype=torch.uint8)),
        torch.tensor([0.0]), torch.tensor([0.0]),
    )
    assert matches == []
    assert diagnostics["reject_code"] == "insufficient_second_neighbor"


def test_fixed_point_orb_compute_restores_class_ids_without_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = ORBFeatureExtractor()

    class FakeORB:
        def compute(self, image: np.ndarray, keypoints: list[cv2.KeyPoint]):
            return [keypoints[2], keypoints[0]], np.stack([
                np.full(32, 2, dtype=np.uint8), np.zeros(32, dtype=np.uint8),
            ])

    extractor.orb = FakeORB()  # type: ignore[assignment]
    image = torch.zeros((3, 40, 40))
    indices, descriptors = extractor.compute_at(image, torch.tensor([[10., 10.], [15., 15.], [20., 20.]]))
    assert indices.tolist() == [2, 0]
    assert descriptors[:, 0].tolist() == [2, 0]


def test_ned_opencv_pose_conversion_round_trip() -> None:
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    pose = pp.from_matrix(matrix, pp.SE3_type)
    rvec, tvec = _ned_pose_to_cv(pose)
    restored = _cv_pose_to_ned(rvec, tvec)
    assert torch.allclose(restored.matrix(), matrix, atol=1e-10)


def test_fixed_point_covariance_uses_pixel_and_same_frame_depth_variance() -> None:
    covariance = fixed_point_covariance(
        torch.tensor([[50.0, 40.0]]), torch.tensor([2.0]), torch.tensor([0.04]),
        torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]),
        0.25,
    )
    expected = torch.diag(torch.tensor([0.04, 1.01e-4, 1.01e-4], dtype=torch.float64))
    assert torch.allclose(covariance[0], expected, atol=1e-9)


def test_cached_orb_geometry_floors_only_depth_lookup() -> None:
    K = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]])
    frame = loop_frame(20, 1, 1, K)
    frame.orb_keypoints = torch.tensor([[10.75, 20.25, 31.0, 10.0, 1.0, 0.0, -1.0]])
    frame.orb_descriptors = torch.zeros((1, 32), dtype=torch.uint8)
    frame.depth.fill_(4.0)
    frame.depth[0, 0, 20, 10] = 2.0
    assert frame.depth_covariance is not None
    frame.depth_covariance.fill_(0.04)
    record, error, diagnostics = cached_orb_geometry(frame, 0.25)
    assert error is None and record is not None
    expected = pixel2point_NED(record.pixel_uv, torch.tensor([2.0]), K[0])
    assert torch.allclose(record.point_camera, expected)
    assert torch.equal(record.pixel_uv, torch.tensor([[10.75, 20.25]]))
    assert diagnostics["inbound_orb_points"] == 1
    assert diagnostics["valid_depth_orb_points"] == 1
    assert torch.allclose(record.disparity, torch.tensor([10.0]))
    assert torch.allclose(record.disparity_variance, torch.tensor([1.0]))


def test_cached_orb_geometry_keeps_2d_and_3d_when_covariance_is_missing() -> None:
    K = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]])
    frame = loop_frame(20, 1, 1, K)
    frame.orb_keypoints = torch.tensor([[10.5, 20.5, 31.0, 10.0, 1.0, 0.0, -1.0]])
    frame.orb_descriptors = torch.zeros((1, 32), dtype=torch.uint8)
    frame.depth.fill_(2.0)
    frame.depth_covariance = None
    record, error, diagnostics = cached_orb_geometry(frame, 0.25)
    assert error is None and record is not None
    assert torch.isfinite(record.pixel_uv).all()
    assert torch.isfinite(record.point_camera).all()
    assert not torch.isfinite(record.point_covariance_camera).any()
    assert record.disparity_valid.tolist() == [False]
    assert diagnostics["depth_covariance_layout_valid"] is False


def test_cached_orb_geometry_requires_strictly_positive_depth_variance() -> None:
    K = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]])
    frame = loop_frame(20, 1, 1, K)
    frame.orb_keypoints = torch.tensor([[10.5, 20.5, 31.0, 10.0, 1.0, 0.0, -1.0]])
    frame.orb_descriptors = torch.zeros((1, 32), dtype=torch.uint8)
    assert frame.depth_covariance is not None
    frame.depth_covariance.zero_()
    record, error, _ = cached_orb_geometry(frame, 0.25)
    assert error is None and record is not None
    assert torch.isfinite(record.point_camera).all()
    assert not torch.isfinite(record.point_covariance_camera).any()
    assert record.disparity_valid.tolist() == [False]


@pytest.mark.parametrize(
    "keypoints,descriptors,expected",
    [
        (torch.zeros((1, 1)), torch.zeros((1, 32), dtype=torch.uint8), "invalid_orb_cache_layout"),
        (torch.zeros((1, 7)), torch.zeros((2, 32), dtype=torch.uint8), "invalid_orb_cache_layout"),
        (torch.zeros((1, 7)), torch.zeros((1, 32)), "invalid_orb_cache_layout"),
        (torch.empty((0, 7)), torch.empty((0, 32), dtype=torch.uint8), "empty_orb_descriptors"),
    ],
)
def test_cached_orb_geometry_rejects_invalid_cache_layout(
    keypoints: torch.Tensor, descriptors: torch.Tensor, expected: str,
) -> None:
    K = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]])
    frame = loop_frame(20, 1, 1, K)
    frame.orb_keypoints = keypoints
    frame.orb_descriptors = descriptors
    record, error, _ = cached_orb_geometry(frame, 0.25)
    assert record is None
    assert error == expected


def test_cached_orb_geometry_requires_angle_column_only_for_orbslam() -> None:
    K = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]])
    frame = loop_frame(20, 1, 1, K)
    frame.orb_keypoints = torch.tensor([[10.0, 20.0]])
    frame.orb_descriptors = torch.zeros((1, 32), dtype=torch.uint8)
    legacy, legacy_error, _ = cached_orb_geometry(frame, 0.25)
    strict, strict_error, _ = cached_orb_geometry(frame, 0.25, require_orientation=True)
    assert legacy is not None and legacy_error is None
    assert strict is None and strict_error == "invalid_orb_cache_layout"


def test_invalid_candidate_covariance_does_not_reject_pnp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    candidate_geometry = pnp_geometry(26, 10, 0, 0, 0.0)
    current_geometry = pnp_geometry(26, 20, 1, 1, 1.0)
    candidate_geometry.point_covariance_camera[:] = torch.nan

    def fake_solve(*args, **kwargs):
        return True, args[4], args[5], np.arange(26, dtype=np.int32).reshape(-1, 1)

    monkeypatch.setattr(cv2, "solvePnPRansac", fake_solve)
    config = SimpleNamespace(
        hamming_threshold=80, iterations=100, reproj_error_px=10.0, confidence=0.99,
        min_inliers=26, max_translation_m=20.0, max_rotation_deg=30.0,
    )
    result = verify_fixed_geometry(
        config, {"loop_frame_idx": 1},
        {"loop_frame_idx": 0, "sensor_frame_idx": 10, "score": 1.0},
        loop_frame(20, 1, 1, K), loop_frame(10, 0, 0, K),
        current_geometry, candidate_geometry, pp.identity_SE3(2).tensor(),
        0.25, torch.eye(6, dtype=torch.float64),
    )
    assert result.constraint is not None
    assert result.row["geometry_accepted"] is True
    assert result.row["information_valid"] is False
    assert result.row["pgo_comparison_eligible"] is False
    assert result.row["information_valid_inliers"] == 0


@pytest.mark.parametrize(
    "retval,returned_inliers,expected_reject",
    [(False, 0, "pnp_failed"), (True, 25, "insufficient_positive_inliers"), (True, 26, None)],
)
def test_verify_fixed_geometry_uses_vo_guess_direction_and_26_inlier_boundary(
    monkeypatch: pytest.MonkeyPatch,
    retval: bool,
    returned_inliers: int,
    expected_reject: str | None,
) -> None:
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    candidate_geometry = pnp_geometry(26, 10, 0, 0, 0.0)
    current_geometry = pnp_geometry(26, 20, 1, 1, 1.0)
    poses = pp.identity_SE3(2).tensor()
    poses[0, 0] = 1.0
    captured: dict[str, object] = {}

    def fake_solve(*args, **kwargs):
        captured["rvec"] = np.asarray(args[4]).copy()
        captured["tvec"] = np.asarray(args[5]).copy()
        captured["use_guess"] = args[6]
        inliers = (
            None if returned_inliers == 0
            else np.arange(returned_inliers, dtype=np.int32).reshape(-1, 1)
        )
        return retval, args[4], args[5], inliers

    monkeypatch.setattr(cv2, "solvePnPRansac", fake_solve)
    config = SimpleNamespace(
        hamming_threshold=80, iterations=100, reproj_error_px=10.0, confidence=0.99,
        min_inliers=26, max_translation_m=20.0, max_rotation_deg=30.0,
    )
    result = verify_fixed_geometry(
        config,
        {"loop_frame_idx": 1},
        {"loop_frame_idx": 0, "sensor_frame_idx": 10, "score": 1.0},
        loop_frame(20, 1, 1, K), loop_frame(10, 0, 0, K),
        current_geometry, candidate_geometry, poses, 0.25, torch.eye(6, dtype=torch.float64),
    )
    assert captured["use_guess"] is True
    assert np.allclose(np.asarray(captured["tvec"]).reshape(3), [0.0, 0.0, 1.0])
    assert (result.constraint is not None) is (expected_reject is None)
    if expected_reject is None:
        expected_edge = (pp.SE3(poses[1]).Inv() @ pp.SE3(poses[0])).Inv()
        assert torch.allclose(
            pp.SE3(torch.tensor(result.constraint.relative_pose)).matrix(),
            expected_edge.matrix(), atol=1e-6,
        )
    else:
        assert result.row["reject_code"] == expected_reject


def test_disp_information_checks_stacked_rank_and_only_downweights() -> None:
    points = torch.tensor([
        [2.0, -0.5, -0.3], [2.2, 0.4, -0.2], [2.5, -0.3, 0.4],
        [3.0, 0.5, 0.3], [3.5, -0.6, 0.2], [4.0, 0.2, -0.4],
    ], dtype=torch.float64)
    K = torch.tensor([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]], dtype=torch.float64)
    baseline = torch.tensor([0.2], dtype=torch.float64)
    uv = torch.stack([100. * points[:, 1] / points[:, 0] + 50., 100. * points[:, 2] / points[:, 0] + 40.], dim=-1)
    disparity = 20. / points[:, 0]
    used, diagnostics = _strict_information(
        pp.identity_SE3(1).double(), points, torch.eye(3).repeat(len(points), 1, 1).double() * 1e-3,
        uv, disparity, torch.ones(len(points), dtype=torch.float64) * 0.1,
        K, baseline, 0.25, torch.eye(6, dtype=torch.float64),
    )
    assert diagnostics["rank"] == 6
    assert diagnostics["valid"] is True
    assert used is not None
    assert torch.linalg.eigvalsh(torch.eye(6, dtype=torch.float64) - used).min() >= -1e-8

    _, single = _strict_information(
        pp.identity_SE3(1).double(), points[:1], torch.eye(3).unsqueeze(0).double() * 1e-3,
        uv[:1], disparity[:1], torch.ones(1, dtype=torch.float64) * 0.1,
        K, baseline, 0.25, torch.eye(6, dtype=torch.float64),
    )
    assert single["rank"] <= 3
    assert single["valid"] is False


def test_robust_observation_information_uses_covariance_sum_and_is_full_rank() -> None:
    points = torch.tensor([
        [2.0, -0.5, -0.3], [2.2, 0.4, -0.2], [2.5, -0.3, 0.4],
        [3.0, 0.5, 0.3], [3.5, -0.6, 0.2], [4.0, 0.2, -0.4],
    ], dtype=torch.float64)
    K = torch.tensor(
        [[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]],
        dtype=torch.float64,
    )
    uv = torch.stack([
        100. * points[:, 1] / points[:, 0] + 50.,
        100. * points[:, 2] / points[:, 0] + 40.,
    ], dim=-1)
    system = robust_observation_system(
        pp.identity_SE3(1).double(),
        points,
        torch.eye(3, dtype=torch.float64).repeat(len(points), 1, 1) * 1e-3,
        uv,
        torch.eye(2, dtype=torch.float64).repeat(len(points), 1, 1) * 0.25,
        20.0 / points[:, 0],
        torch.full((len(points),), 0.1, dtype=torch.float64),
        K,
        torch.tensor([0.2], dtype=torch.float64),
        2.795,
    )
    # Measurement and projected candidate covariance are added, never subtracted.
    assert torch.all(
        torch.diagonal(system.covariance, dim1=-2, dim2=-1)
        > torch.tensor([0.25, 0.25, 0.1], dtype=torch.float64)
    )
    information, diagnostics = information_diagnostics(
        system, pp.identity_SE3(1).double(), point_count=len(points),
    )
    assert information is not None
    assert diagnostics["rank"] == 6
    assert diagnostics["normalization"] == "raw_robust_hessian_no_lm_damping"


def test_network_refinement_succeeds_with_one_frontend_inference() -> None:
    result, current_frame, historical_frame, current, historical = refinement_input()
    frontend = SyntheticRefinementFrontend(historical)
    refined = refine_geometry_with_network(
        result,
        SimpleNamespace(
            min_points=6,
            huber_delta=2.795,
            max_iterations=3,
            damping_initial=1e-3,
        ),
        frontend,
        current_frame,
        historical_frame,
        current,
        historical,
        torch.eye(6, dtype=torch.float64) * 100.0,
        20.0,
        180.0,
    )
    assert frontend.calls == 1
    assert refined.constraint is not None
    assert refined.covariance_information is not None
    assert refined.row["network_refinement"]["status"] == "succeeded"
    assert refined.row["network_refinement"]["covariance_dropped_points"] == 0


def test_network_refinement_drops_one_non_spd_covariance_point() -> None:
    result, current_frame, historical_frame, current, historical = refinement_input()
    frontend = SyntheticRefinementFrontend(
        historical, invalid_covariance_index=0,
    )
    refined = refine_geometry_with_network(
        result,
        SimpleNamespace(
            min_points=6,
            huber_delta=2.795,
            max_iterations=3,
            damping_initial=1e-3,
        ),
        frontend,
        current_frame,
        historical_frame,
        current,
        historical,
        torch.eye(6, dtype=torch.float64) * 100.0,
        20.0,
        180.0,
    )
    assert frontend.calls == 1
    assert refined.constraint is not None
    assert refined.covariance_information is not None
    assert refined.row["network_refinement"]["covariance_dropped_points"] == 1
    assert refined.row["network_refinement"]["final_point_count"] == 7


def test_pose_copy_safety_rejects_loss_increase() -> None:
    class FakeOptimizer:
        def compute_loss(self, poses: torch.Tensor) -> torch.Tensor:
            return torch.tensor(2.0 if float(poses[1, 4]) != 0.0 else 1.0)

        def optimize_poses(self, poses: torch.Tensor) -> torch.Tensor:
            result = poses.clone()
            result[1, 4] = 1.0
            return result

        def compute_residuals(self, poses: torch.Tensor) -> torch.Tensor:
            return torch.zeros((1, 6))

    poses = pp.identity_SE3(2).tensor()
    optimized, diagnostics = run_pose_copy_pgo_safety(FakeOptimizer(), poses)
    assert optimized is None
    assert diagnostics["reason"] == "loss_increased"


def test_pose_copy_safety_does_not_apply_sparse_status_to_lbfgs() -> None:
    class FakeLBFGSOptimizer:
        solver = "lbfgs"
        last_optimization_diagnostics = {
            "solver": "lbfgs",
            "safe": False,
            "trajectory_source": "not_run",
        }
        edges: list = []

        def compute_loss(self, _poses: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.0)

        def optimize_poses(self, poses: torch.Tensor) -> torch.Tensor:
            return poses.clone()

        def compute_residuals(self, _poses: torch.Tensor) -> torch.Tensor:
            return torch.empty((0, 6))

    poses = pp.identity_SE3(2).tensor()
    optimized, diagnostics = run_pose_copy_pgo_safety(
        FakeLBFGSOptimizer(), poses,
    )
    assert optimized is not None
    assert diagnostics["safe"] is True


def test_pose_copy_comparison_requires_identical_edge_sets() -> None:
    fixed = SimpleNamespace(edges=[SimpleNamespace(src=0, dst=1, edge_type="odometry")])
    covariance = SimpleNamespace(edges=[SimpleNamespace(src=0, dst=2, edge_type="odometry")])
    result = run_pose_copy_pgo_comparison(
        fixed, covariance, pp.identity_SE3(3).tensor()
    )
    assert result["safe"] is False
    assert result["reason"] == "pgo_edge_sets_differ"


def test_offline_pose_copy_pgo_runs_identical_eligible_edge_files(tmp_path: Path) -> None:
    global_map = VisualMap()
    poses = pp.identity_SE3(3).tensor()
    poses[1, 0] = 1.0
    poses[2, 0] = 2.0
    global_map.frames.index.push(torch.arange(3))
    global_map.frames.data["pose"].push(poses)
    global_map.frames.data["need_interp"].push(torch.zeros(3, dtype=torch.bool))
    constraint = {
        "src_visual_map_idx": 0, "dst_visual_map_idx": 2,
        "src_sensor_frame_idx": 0, "dst_sensor_frame_idx": 2,
        "relative_pose": (pp.SE3(poses[0]).Inv() @ pp.SE3(poses[2])).tensor().tolist(),
        "information": torch.eye(6).tolist(),
    }
    fixed_path = tmp_path / "fixed.json"
    covariance_path = tmp_path / "covariance.json"
    fixed_path.write_text(json.dumps({"constraints": [constraint]}), encoding="utf-8")
    covariance_path.write_text(json.dumps({"constraints": [constraint]}), encoding="utf-8")
    config = SimpleNamespace(
        enabled=False, optimize_on_terminate=False, max_iterations=5,
        trans_weight=1.0, rot_weight=1.0, device="cpu", include_interp_frames=True,
    )
    before = global_map.frames.data["pose"].tensor.clone()
    result = run_vins_pose_copy_pgo(global_map, config, fixed_path, covariance_path)
    assert result["executed"] is True
    assert result["safe"] is True
    assert result["eligible_loop_edge_count"] == 1
    assert torch.equal(global_map.frames.data["pose"].tensor, before)


def test_engineering_admission_requires_safe_executed_pose_copy_pgo() -> None:
    gt_proxy = {
        "available": True, "accepted_constraints": 3, "evaluated_constraints": 3,
        "counts": {"accurate": 3, "suspicious": 0, "large_error": 0},
        "rows": [
            {"current_sensor_frame_idx": 100},
            {"current_sensor_frame_idx": 200},
            {"current_sensor_frame_idx": 200},
        ],
    }
    unsafe = summarize_engineering_admission(
        gt_proxy, {"executed": True, "safe": False, "original_pose_invariant": True},
    )
    assert unsafe["eligible"] is False
    safe = summarize_engineering_admission(
        gt_proxy, {"executed": True, "safe": True, "original_pose_invariant": True},
    )
    assert safe["eligible"] is True
