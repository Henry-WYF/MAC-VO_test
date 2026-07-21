from __future__ import annotations

from types import SimpleNamespace

import pypose as pp
import pytest
import torch
import torch.nn.functional as F

from Module.LoopClosure.PhaseB5 import (
    PhaseB5Analyzer,
    _candidate_point_covariance,
    _se3_exp,
    _weighted_system,
    normalized_covariance_risk,
    orb_geometry_observe,
    reproj_disp_linearization,
    transform_information_for_inverse,
)
from Utility.Selection import local_minimum_nms


def phase_b5_config() -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        mode="observe",
        trusted_manifest=None,
        calibration=SimpleNamespace(
            prefix_fraction=0.2,
            min_queries=20,
            min_all_bow_pairs=100,
            min_orb_pairs=30,
            absolute_median_log_risk_cap=None,
            absolute_q95_log_risk_cap=None,
        ),
        orb=SimpleNamespace(ratio=0.8, max_depth=20.0),
        flow=SimpleNamespace(
            min_valid_points=30, min_valid_ratio=0.8, min_grid_cells=8,
            nms_kernel_size=7, border=8, min_points=30, max_points=800,
            max_depth=20.0, grid_rows=8, grid_cols=8, max_points_per_cell=20,
        ),
    )


def test_nms_masks_invalid_covariance_before_local_minimum() -> None:
    quality = torch.full((1, 1, 5, 5), 10.0)
    quality[0, 0, 2, 2] = -100.0
    quality[0, 0, 2, 3] = 1.0
    valid = torch.ones_like(quality, dtype=torch.bool)
    valid[0, 0, 2, 2] = False
    unmasked = local_minimum_nms(quality, 3)
    masked = local_minimum_nms(quality, 3, valid)
    assert unmasked[0, 0, 2, 3].item() is False
    assert masked[0, 0, 2, 3].item() is True
    assert masked[0, 0, 2, 2].item() is False


def test_nms_helper_preserves_legacy_frontend_result_without_mask() -> None:
    quality = torch.tensor(
        [[[[4.0, 3.0, 2.0], [5.0, float("nan"), 1.0], [6.0, 7.0, 8.0]]]]
    )
    eroded = -F.max_pool2d(-quality, kernel_size=3, stride=1, padding=1)
    legacy = (quality == eroded) & ~quality.isnan()
    assert torch.equal(local_minimum_nms(quality, 3), legacy)


def test_normalized_lambda_max_detects_correlated_high_risk() -> None:
    covariance = torch.tensor([[100.0], [100.0], [99.0]])
    risk, valid, diagnostics = normalized_covariance_risk(covariance, 1, 1)
    assert valid.tolist() == [True]
    assert risk.item() == pytest.approx(199.0)
    assert diagnostics["invalid_psd_points"] == 0


def test_psd_tolerance_clamps_only_small_negative_eigenvalue() -> None:
    covariance = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0000001, 2.0]])
    _, valid, diagnostics = normalized_covariance_risk(covariance, 1, 1)
    assert valid.tolist() == [True, False]
    assert diagnostics["small_negative_eigenvalues_clamped"] == 1
    assert diagnostics["invalid_psd_points"] == 1


def test_candidate_uv_depth_covariance_propagation() -> None:
    uv = torch.tensor([[5.0, 5.0]], dtype=torch.float64)
    depth = torch.tensor([2.0], dtype=torch.float64)
    depth_cov = torch.tensor([0.25], dtype=torch.float64)
    K = torch.tensor([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    covariance = _candidate_point_covariance(uv, depth, depth_cov, K)
    assert covariance[0, 0, 0].item() == pytest.approx(0.25)
    assert covariance[0, 1, 1].item() == pytest.approx((2.0 / 10.0) ** 2 / 12.0)
    assert covariance[0, 2, 2].item() == pytest.approx((2.0 / 10.0) ** 2 / 12.0)


def test_reprojection_disparity_zero_residual_and_finite_jacobian() -> None:
    point = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    point_cov = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 0.01
    uv = torch.tensor([[5.0, 5.0]], dtype=torch.float64)
    uv_cov = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    disparity = torch.tensor([1.0], dtype=torch.float64)
    disparity_cov = torch.tensor([0.5], dtype=torch.float64)
    K = torch.tensor([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    residual, jacobian, covariance, _ = reproj_disp_linearization(
        pp.identity_SE3(1).double(), point, point_cov, uv, uv_cov,
        disparity, disparity_cov, K, torch.tensor([0.2], dtype=torch.float64),
    )
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-12)
    assert jacobian.shape == (1, 3, 6)
    assert torch.isfinite(jacobian).all()
    # Disparity uncertainty is already a variance and must not be squared.
    assert covariance[0, 2, 2].item() >= 0.5


def test_reprojection_disparity_jacobian_matches_finite_difference() -> None:
    point = torch.tensor([[2.0, 0.2, -0.1]], dtype=torch.float64)
    point_cov = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 0.01
    K = torch.tensor([[10.0, 0.0, 5.0], [0.0, 11.0, 4.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    uv = torch.tensor([[6.0, 3.45]], dtype=torch.float64)
    disparity = torch.tensor([1.0], dtype=torch.float64)
    uv_cov = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    disparity_cov = torch.tensor([0.5], dtype=torch.float64)
    identity = pp.identity_SE3(1).double()
    residual, jacobian, _, _ = reproj_disp_linearization(
        identity, point, point_cov, uv, uv_cov, disparity,
        disparity_cov, K, torch.tensor([0.2], dtype=torch.float64),
    )
    epsilon = 1e-6
    numerical = torch.empty((3, 6), dtype=torch.float64)
    for column in range(6):
        delta = torch.zeros(6, dtype=torch.float64)
        delta[column] = epsilon
        R, t = _se3_exp(delta)
        matrix = torch.eye(4, dtype=torch.float64)
        matrix[:3, :3], matrix[:3, 3] = R, t
        perturbed = pp.from_matrix(matrix, pp.SE3_type)
        changed, _, _, _ = reproj_disp_linearization(
            perturbed, point, point_cov, uv, uv_cov, disparity,
            disparity_cov, K, torch.tensor([0.2], dtype=torch.float64),
        )
        numerical[:, column] = (changed[0] - residual[0]) / epsilon
    assert torch.allclose(jacobian[0], numerical, atol=2e-4, rtol=2e-4)


def test_larger_residual_covariance_does_not_strengthen_information() -> None:
    residual = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float64)
    jacobian = torch.arange(18, dtype=torch.float64).reshape(1, 3, 6) / 10.0
    covariance = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    first, _, _, _ = _weighted_system(residual, jacobian, covariance)
    second, _, _, _ = _weighted_system(residual, jacobian, covariance * 2.0)
    assert torch.allclose(first, first.T, atol=1e-12)
    assert torch.linalg.eigvalsh(first).min().item() >= -1e-9
    assert int(torch.linalg.matrix_rank(first)) < 6
    assert torch.linalg.eigvalsh(first - second).min().item() >= -1e-9


def test_information_inverse_adjoint_round_trip() -> None:
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, 3] = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    pose = pp.from_matrix(matrix, pp.SE3_type)
    information = torch.diag(torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64))
    inverse = transform_information_for_inverse(information, pose)
    restored = transform_information_for_inverse(inverse, pose.Inv())
    assert torch.allclose(restored, information, atol=1e-9, rtol=1e-9)


def test_calibration_prefix_is_shared_and_does_not_expand() -> None:
    queries = [
        {
            "sensor_frame_idx": index * 10,
            "visual_map_idx": index,
            "loop_frame_idx": index,
            "candidates": [{
                "sensor_frame_idx": max(index - 1, 0) * 10,
                "visual_map_idx": max(index - 1, 0),
                "loop_frame_idx": max(index - 1, 0),
                "score": 0.5,
            }],
        }
        for index in range(10)
    ]
    analyzer = PhaseB5Analyzer(phase_b5_config(), queries, [])
    assert analyzer.prefix_query_count == 2
    assert analyzer.calibration_end_sensor_frame_idx == 10
    manifest = analyzer.calibration_manifest()
    assert manifest["populations"]["all_bow_candidates"]["reason"] == "calibration_insufficient"
    assert manifest["populations"]["orb_supported_candidates"]["reason"] == "calibration_insufficient"


def test_pair_calibration_and_point_cap_sufficiency_are_independent() -> None:
    config = phase_b5_config()
    config.calibration.min_queries = 1
    config.calibration.min_all_bow_pairs = 1
    config.calibration.absolute_median_log_risk_cap = 10.0
    config.calibration.absolute_q95_log_risk_cap = 10.0
    queries = [
        {
            "sensor_frame_idx": index * 10,
            "visual_map_idx": index,
            "loop_frame_idx": index,
            "candidates": [{
                "sensor_frame_idx": max(index - 1, 0) * 10,
                "visual_map_idx": max(index - 1, 0),
                "loop_frame_idx": max(index - 1, 0),
                "score": 0.5,
            }],
        }
        for index in range(5)
    ]
    analyzer = PhaseB5Analyzer(config, queries, [])
    analyzer.rows.append({
        "in_calibration_prefix": True,
        "populations": ["all_bow_candidates"],
        "flow": {"pair_sanity_pass": True, "pair_risk_p50": 0.5},
    })
    analyzer.point_risks["all_bow_candidates"].append(
        (0.5, torch.ones(10, dtype=torch.float32))
    )
    population = analyzer.calibration_manifest()["populations"]["all_bow_candidates"]
    assert population["trusted"] is True
    assert population["point_cap_calibration_sufficient"] is False
    assert population["point_risk_cap"] is None
    assert population["point_cap_reason"] == "point_cap_calibration_insufficient"


def test_orb_empty_and_single_descriptor_are_stable_rejections() -> None:
    base = {
        "sensor_frame_idx": 10,
        "orb_keypoints": torch.empty((0, 7)),
        "orb_descriptors": torch.empty((0, 32), dtype=torch.uint8),
    }
    empty = SimpleNamespace(**base)
    assert orb_geometry_observe(empty, empty, SimpleNamespace(ratio=0.8, max_depth=20.0))["reject_code"] == "empty_descriptors"
    single = SimpleNamespace(
        sensor_frame_idx=10,
        orb_keypoints=torch.zeros((1, 7)),
        orb_descriptors=torch.zeros((1, 32), dtype=torch.uint8),
    )
    assert orb_geometry_observe(single, single, SimpleNamespace(ratio=0.8, max_depth=20.0))["reject_code"] == "insufficient_knn_neighbors"


def test_apply_requires_promoted_manifest() -> None:
    config = phase_b5_config()
    config.mode = "apply"
    with pytest.raises(ValueError, match="promoted trusted manifest"):
        PhaseB5Analyzer(config, [], [])
