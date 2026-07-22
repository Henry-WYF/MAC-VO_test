from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pypose as pp
import pytest
import torch

from Module.Frontend.Matching import IMatcher
from Module.Frontend.StereoDepth import IStereoDepth
from Module.LoopClosure import LoopClosureManager, LoopFrameRecord
from Module.LoopClosure.Verification import (
    LoopCandidateVerifier,
    _correspondence_signature,
    covariance_statistics,
)
from Module.Map import VisualMap
from Module.Map.Template import FrameNode
from Scripts.AdHoc.RunLoopPhaseBOffline import (
    evaluate_gt_pose_proxy,
    limit_queries,
    validate_comparison_outputs,
)


_MISSING = object()


def make_config(
    tmp_path: Path,
    *,
    gate: bool | None = False,
    compare: bool | None = True,
    adaptive_target: int | None | object = _MISSING,
) -> SimpleNamespace:
    geometric = SimpleNamespace(enabled=True, max_candidates_to_verify=10, min_sensor_gap=50)
    geometry = SimpleNamespace(
        min_points=10,
        max_points=16,
        max_depth=20.0,
        max_depth_cov=1.0,
        max_flow_cov=100.0,
        border=0,
        grid_rows=2,
        grid_cols=2,
        max_points_per_cell=4,
    )
    if compare is not None:
        geometric.compare_flow_cov_gate = compare
    if gate is not None:
        geometry.flow_cov_gate_enabled = gate
    if adaptive_target is not _MISSING:
        geometry.flow_cov_adaptive_target_points = adaptive_target
    return SimpleNamespace(
        enabled=True,
        vocabulary_path=str(tmp_path / "missing.npz"),
        keyframe_stride_sensor_frames=10,
        temporal_exclusion_sensor_frames=50,
        cache_failure_policy="disable_loop",
        orb_nfeatures=1000,
        orb_scale_factor=1.2,
        orb_nlevels=8,
        top_k=10,
        geometric_verification=geometric,
        geometry=geometry,
        pnp=SimpleNamespace(
            reproj_error_px=3.0,
            confidence=0.999,
            iterations=100,
            min_inliers=4,
            min_inlier_ratio=0.25,
            refine=True,
        ),
        verification=SimpleNamespace(
            max_mean_reproj_error_px=3.0,
            max_rotation_diff_deg=35.0,
            max_translation_diff_m=8.0,
        ),
        loop_information=SimpleNamespace(trans_weight=1.0, rot_weight=1.0),
    )


def vins_geometry_config(
    feature_source: str | None = None,
    descriptor_match_mode: str | None = None,
) -> SimpleNamespace:
    config = SimpleNamespace(
        enabled=True, max_candidates=10, hamming_threshold=80, iterations=100,
        reproj_error_px=10.0, confidence=0.99, min_inliers=26,
        max_translation_m=20.0, max_rotation_deg=30.0,
    )
    if feature_source is not None:
        config.feature_source = feature_source
    if descriptor_match_mode is not None:
        config.descriptor_match_mode = descriptor_match_mode
    return config


def test_vins_feature_source_is_optional_and_controls_sidecar(tmp_path: Path) -> None:
    legacy = make_config(tmp_path)
    legacy.vins_geometry = vins_geometry_config()
    LoopClosureManager.is_valid_config(legacy)
    legacy_manager = LoopClosureManager(legacy)
    assert legacy_manager.requires_geometry_sidecar is True

    detected = make_config(tmp_path)
    detected.vins_geometry = vins_geometry_config("orb_detected")
    LoopClosureManager.is_valid_config(detected)
    detected_manager = LoopClosureManager(detected)
    assert detected_manager.requires_geometry_sidecar is False

    invalid = make_config(tmp_path)
    invalid.vins_geometry = vins_geometry_config("unknown")
    with pytest.raises(ValueError):
        LoopClosureManager.is_valid_config(invalid)

    orbslam = make_config(tmp_path)
    orbslam.vins_geometry = vins_geometry_config("orb_detected", "orbslam")
    LoopClosureManager.is_valid_config(orbslam)

    # Mirror the offline CLI sequence: start from the YAML's valid ORB-SLAM
    # combination, override only the feature source, then validate the effective config.
    incompatible = make_config(tmp_path)
    incompatible.vins_geometry = vins_geometry_config("orb_detected", "orbslam")
    incompatible.vins_geometry.feature_source = "fixed_covariance"
    with pytest.raises(ValueError):
        LoopClosureManager.is_valid_config(incompatible)

    fixed_baseline = make_config(tmp_path)
    fixed_baseline.vins_geometry = vins_geometry_config("fixed_covariance", "vins_legacy")
    LoopClosureManager.is_valid_config(fixed_baseline)

    invalid_matcher = make_config(tmp_path)
    invalid_matcher.vins_geometry = vins_geometry_config("orb_detected", "unknown")
    with pytest.raises(ValueError):
        LoopClosureManager.is_valid_config(invalid_matcher)


def make_record(sensor_idx: int, visual_idx: int, loop_idx: int, height: int = 12, width: int = 12) -> LoopFrameRecord:
    image = torch.zeros((1, 3, height, width), dtype=torch.float32)
    intrinsic = torch.tensor(
        [[[8.0, 0.0, width / 2], [0.0, 8.0, height / 2], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    return LoopFrameRecord(
        sensor_frame_idx=sensor_idx,
        visual_map_idx=visual_idx,
        loop_frame_idx=loop_idx,
        frame_ns=sensor_idx * 1_000_000,
        height=height,
        width=width,
        image_left=image,
        image_right=image.clone(),
        intrinsic=intrinsic,
        baseline=torch.tensor([0.2]),
        body_to_sensor=pp.identity_SE3(1).tensor(),
        depth=torch.ones((1, 1, height, width), dtype=torch.float32),
        depth_covariance=torch.full((1, 1, height, width), 0.01, dtype=torch.float32),
        registered_pose=pp.identity_SE3(1).tensor(),
        orb_keypoints=torch.empty((0, 7)),
        orb_descriptors=torch.empty((0, 32), dtype=torch.uint8),
        bow_vector=None,
    )


class FakeFrontend:
    def __init__(self, output: IMatcher.Output) -> None:
        self.output = output
        self.config = SimpleNamespace(device="cpu")
        self.calls = 0

    def estimate_pair(self, historical, current):
        self.calls += 1
        depth = torch.ones((1, 1, historical.height, historical.width), dtype=torch.float32)
        return IStereoDepth.Output(depth=depth, cov=torch.ones_like(depth)), self.output


def make_match(
    covariance: torch.Tensor | None,
    height: int = 12,
    width: int = 12,
    mask: torch.Tensor | None = None,
) -> IMatcher.Output:
    return IMatcher.Output(
        flow=torch.zeros((1, 2, height, width), dtype=torch.float32),
        cov=covariance,
        mask=mask,
    )


def query_and_candidate() -> tuple[dict, dict]:
    query = {"sensor_frame_idx": 200, "visual_map_idx": 1, "loop_frame_idx": 1}
    candidate = {"sensor_frame_idx": 100, "visual_map_idx": 0, "loop_frame_idx": 0, "score": 0.8}
    return query, candidate


def fixed_points() -> torch.Tensor:
    return torch.tensor([[2.0, 2.0], [4.0, 2.0], [2.0, 4.0], [4.0, 4.0]])


def covariance_map() -> torch.Tensor:
    covariance = torch.ones((1, 3, 12, 12), dtype=torch.float32)
    points = fixed_points().long()
    covariance[0, 0, points[:, 1], points[:, 0]] = torch.tensor([100.0, 150.0, 50.0, float("nan")])
    covariance[0, 1, points[:, 1], points[:, 0]] = torch.tensor([100.0, 50.0, 150.0, 50.0])
    covariance[0, 2, points[:, 1], points[:, 0]] = 0.0
    return covariance


def run_pair(verifier: LoopCandidateVerifier, monkeypatch: pytest.MonkeyPatch, gates: list[bool]):
    current = make_record(200, 1, 1)
    historical = make_record(100, 0, 0)
    monkeypatch.setattr(verifier, "_sample_candidate_uv", lambda _: fixed_points())
    query, candidate = query_and_candidate()
    poses = torch.cat([pp.identity_SE3(1).tensor(), pp.identity_SE3(1).tensor()], dim=0)
    return verifier.verify_branches(poses, query, candidate, current, historical, gates)


def test_covariance_statistics_are_strict_and_linear() -> None:
    stats = covariance_statistics(torch.tensor([1.0, 2.0, float("nan"), float("inf")]))
    assert stats["count"] == 2
    assert stats["p50"] == pytest.approx(1.5)
    empty = covariance_statistics(torch.tensor([float("nan")]))
    assert empty["count"] == 0
    assert all(empty[key] is None for key in empty if key != "count")


def test_single_frontend_call_feeds_gate_on_and_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frontend = FakeFrontend(make_match(covariance_map()))
    verifier = LoopCandidateVerifier(make_config(tmp_path), frontend)  # type: ignore[arg-type]
    results = run_pair(verifier, monkeypatch, [True, False])

    on, off = results[True][0], results[False][0]
    assert frontend.calls == 1
    assert verifier.frontend_inference_calls == verifier.candidates_reaching_frontend == 1
    assert on.comparison_pair_id == off.comparison_pair_id == "1:0"
    assert on.comparison_applicable is off.comparison_applicable is True
    assert on.diagnostics["after_flow_cov"] == 1  # type: ignore[index]
    assert off.diagnostics["after_flow_cov"] == 4  # type: ignore[index]
    assert on.diagnostics["pnp_correspondence_count"] == 1  # type: ignore[index]
    assert off.diagnostics["pnp_correspondence_count"] == 4  # type: ignore[index]
    assert on.diagnostics["pnp_correspondence_signature"] != off.diagnostics["pnp_correspondence_signature"]  # type: ignore[index]
    cov_diag = off.diagnostics["covariance"]  # type: ignore[index]
    assert cov_diag["uv_all_zero"] is True
    assert cov_diag["frontend_score_nonfinite"] == 1
    assert cov_diag["sampled_frontend_threshold"] == 100.0
    assert cov_diag["frontend_score_below_reference"] == 0


def test_adaptive_gate_supplements_by_risk_and_preserves_original_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verifier = LoopCandidateVerifier(
        make_config(tmp_path, adaptive_target=3),
        FakeFrontend(make_match(covariance_map())),
    )  # type: ignore[arg-type]
    record = run_pair(verifier, monkeypatch, [True])[True][0]
    diagnostics = record.diagnostics
    assert diagnostics is not None
    assert diagnostics["selection_mode"] == "adaptive"
    assert diagnostics["fixed_core_points"] == 1
    assert diagnostics["adaptive_pool_points"] == 2
    assert diagnostics["adaptive_added_points"] == 2
    assert diagnostics["adaptive_final_points_before_match_depth"] == 3
    assert diagnostics["adaptive_points_after_match_mask"] == 3
    assert diagnostics["adaptive_points_after_depth"] == 3
    assert diagnostics["adaptive_shortfall"] == 0
    assert diagnostics["adaptive_post_filter_shortfall"] == 0
    assert diagnostics["adaptive_effective_risk_cutoff"] == 150.0
    assert diagnostics["pnp_correspondence_signature"] == _correspondence_signature(
        torch.tensor([0, 1, 2])
    )


def test_adaptive_shortfall_and_null_target_legacy_equivalence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = LoopCandidateVerifier(
        make_config(tmp_path), FakeFrontend(make_match(covariance_map()))
    )  # type: ignore[arg-type]
    explicit_null = LoopCandidateVerifier(
        make_config(tmp_path, adaptive_target=None), FakeFrontend(make_match(covariance_map()))
    )  # type: ignore[arg-type]
    adaptive = LoopCandidateVerifier(
        make_config(tmp_path, adaptive_target=5), FakeFrontend(make_match(covariance_map()))
    )  # type: ignore[arg-type]
    fixed_row = run_pair(missing, monkeypatch, [True])[True][0]
    null_row = run_pair(explicit_null, monkeypatch, [True])[True][0]
    adaptive_row = run_pair(adaptive, monkeypatch, [True])[True][0]
    assert fixed_row.diagnostics["pnp_correspondence_signature"] == null_row.diagnostics[  # type: ignore[index]
        "pnp_correspondence_signature"
    ]
    assert fixed_row.diagnostics["after_flow_cov"] == null_row.diagnostics["after_flow_cov"] == 1  # type: ignore[index]
    assert adaptive_row.diagnostics["adaptive_final_points_before_match_depth"] == 3  # type: ignore[index]
    assert adaptive_row.diagnostics["adaptive_shortfall"] == 2  # type: ignore[index]
    assert adaptive_row.diagnostics["adaptive_post_filter_shortfall"] == 2  # type: ignore[index]


def test_adaptive_target_is_before_match_mask_and_depth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    match_mask = torch.ones((1, 1, 12, 12), dtype=torch.bool)
    points = fixed_points().long()
    match_mask[0, 0, points[2:, 1], points[2:, 0]] = False
    verifier = LoopCandidateVerifier(
        make_config(tmp_path, adaptive_target=3),
        FakeFrontend(make_match(covariance_map(), mask=match_mask)),
    )  # type: ignore[arg-type]
    record = run_pair(verifier, monkeypatch, [True])[True][0]
    diagnostics = record.diagnostics
    assert diagnostics is not None
    assert diagnostics["adaptive_final_points_before_match_depth"] == 3
    assert diagnostics["adaptive_points_after_match_mask"] == 2
    assert diagnostics["adaptive_points_after_depth"] == 2
    assert diagnostics["pnp_input_points"] == 0
    assert diagnostics["adaptive_post_filter_shortfall"] == 1


def test_adaptive_target_config_rejects_bool_and_nonpositive(tmp_path: Path) -> None:
    LoopClosureManager.is_valid_config(make_config(tmp_path, adaptive_target=1))
    LoopClosureManager.is_valid_config(make_config(tmp_path, adaptive_target=None))
    with pytest.raises(ValueError):
        LoopClosureManager.is_valid_config(make_config(tmp_path, adaptive_target=True))
    with pytest.raises(ValueError):
        LoopClosureManager.is_valid_config(make_config(tmp_path, adaptive_target=0))


def test_pnp_covariance_statistics_map_inliers_to_original_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path, adaptive_target=3)
    config.geometry.min_points = 3
    config.pnp.min_inliers = 2
    verifier = LoopCandidateVerifier(
        config, FakeFrontend(make_match(covariance_map()))
    )  # type: ignore[arg-type]
    monkeypatch.setattr(
        verifier,
        "_run_pnp",
        lambda *args: (pp.identity_SE3(1), 0.1, np.asarray([0, 2], dtype=np.int64)),
    )
    record = run_pair(verifier, monkeypatch, [True])[True][0]
    diagnostics = record.diagnostics
    assert record.status == "accepted"
    assert diagnostics is not None
    assert diagnostics["pnp_inlier_original_index_count"] == 2
    assert diagnostics["pnp_inlier_original_index_signature"] == _correspondence_signature(
        torch.tensor([0, 2])
    )
    assert diagnostics["pnp_outlier_original_index_signature"] == _correspondence_signature(
        torch.tensor([1])
    )
    assert diagnostics["pnp_input_covariance_statistics"]["risk_max_uu_vv"]["count"] == 3
    assert diagnostics["pnp_inlier_covariance_statistics"]["risk_max_uu_vv"]["count"] == 2
    assert diagnostics["pnp_outlier_covariance_statistics"]["risk_max_uu_vv"]["count"] == 1


def test_pnp_failure_does_not_call_all_inputs_outliers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path, adaptive_target=3)
    config.geometry.min_points = 3
    verifier = LoopCandidateVerifier(
        config, FakeFrontend(make_match(covariance_map()))
    )  # type: ignore[arg-type]
    monkeypatch.setattr(verifier, "_run_pnp", lambda *args: "PnP RANSAC failed")
    record = run_pair(verifier, monkeypatch, [True])[True][0]
    diagnostics = record.diagnostics
    assert diagnostics is not None
    assert diagnostics["pnp_input_covariance_statistics"]["risk_max_uu_vv"]["count"] == 3
    assert diagnostics["pnp_inlier_covariance_statistics"] is None
    assert diagnostics["pnp_outlier_covariance_statistics"] is None


def test_frozen_gt_proxy_uses_sensor_rows_and_timestamp_alignment(tmp_path: Path) -> None:
    record_dir = tmp_path / "loop_closure"
    frames = record_dir / "frames"
    frames.mkdir(parents=True)
    candidate = make_record(0, 0, 0)
    current = make_record(300, 300, 30)
    candidate.save(frames / "candidate.pt")
    current.save(frames / "current.pt")
    records = [
        {"sensor_frame_idx": 0, "file": "frames/candidate.pt"},
        {"sensor_frame_idx": 300, "file": "frames/current.pt"},
    ]
    reference = np.zeros((301, 8), dtype=np.float64)
    reference[:, 7] = 1.0
    reference[0, 0] = candidate.frame_ns
    reference[300, 0] = current.frame_ns
    np.save(tmp_path / "ref_poses.npy", reference)
    constraints = {
        "schema_version": 1,
        "constraints": [
            {
                "src_sensor_frame_idx": 0,
                "dst_sensor_frame_idx": 300,
                "pnp_relative_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            }
        ],
    }
    constraints_path = tmp_path / "constraints.json"
    constraints_path.write_text(json.dumps(constraints), encoding="utf-8")
    result = evaluate_gt_pose_proxy(
        constraints_path, tmp_path / "ref_poses.npy", record_dir, records
    )
    assert result["evaluated_constraints"] == 1
    assert result["counts"]["accurate"] == 1
    assert result["long_span_accurate"] == 1

    reference[300, 0] += 1
    np.save(tmp_path / "ref_poses.npy", reference)
    unaligned = evaluate_gt_pose_proxy(
        constraints_path, tmp_path / "ref_poses.npy", record_dir, records
    )
    assert unaligned["evaluated_constraints"] == 0
    assert unaligned["excluded_unaligned_or_missing"] == 1


def test_signature_uses_original_indices_and_seed_is_stable(tmp_path: Path) -> None:
    assert _correspondence_signature(torch.tensor([0, 2])) != _correspondence_signature(torch.tensor([0, 1]))
    verifier = LoopCandidateVerifier(make_config(tmp_path), FakeFrontend(make_match(None)))  # type: ignore[arg-type]
    query, candidate = query_and_candidate()
    seed = verifier._pnp_seed(query, candidate)
    assert seed == verifier._pnp_seed(dict(query), dict(candidate))
    assert 0 <= seed <= 0x7FFFFFFF


def test_uv_diagnostics_do_not_change_fixed_gate(tmp_path: Path) -> None:
    verifier = LoopCandidateVerifier(make_config(tmp_path), FakeFrontend(make_match(None)))  # type: ignore[arg-type]
    diagnostics: dict[str, Any] = {"flow_cov_threshold": 1.0}
    gate_mask = verifier._covariance_diagnostics(
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.5, float("nan")]]),
        diagnostics,
    )
    assert gate_mask.tolist() == [True, True]
    covariance = diagnostics["covariance"]
    assert covariance["uv_finite"] == 1
    assert covariance["uv_all_zero"] is False
    assert covariance["frontend_score_finite"] == 1
    assert covariance["frontend_score_nonfinite"] == 1


def test_gate_removing_all_flow_marks_pnp_not_attempted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    covariance = torch.full((1, 3, 12, 12), 150.0, dtype=torch.float32)
    covariance[:, 2] = 0.0
    verifier = LoopCandidateVerifier(
        make_config(tmp_path), FakeFrontend(make_match(covariance))  # type: ignore[arg-type]
    )
    results = run_pair(verifier, monkeypatch, [True, False])
    gate_on = results[True][0]
    assert gate_on.reject_code == "no_valid_flow_correspondences"
    assert gate_on.pnp_attempted is False
    assert gate_on.pnp_ransac_succeeded is None


def test_aggregate_quantiles_merge_raw_samples(tmp_path: Path) -> None:
    verifier = LoopCandidateVerifier(make_config(tmp_path), FakeFrontend(make_match(None)))  # type: ignore[arg-type]
    verifier._aggregate_covariance["uu"] = [torch.tensor([0.0, 100.0]), torch.tensor([1.0])]
    stats = verifier.aggregate_covariance_statistics()["uu"]
    assert stats["count"] == 3
    assert stats["p50"] == pytest.approx(1.0)


def test_branch_exception_does_not_cancel_other_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verifier = LoopCandidateVerifier(
        make_config(tmp_path), FakeFrontend(make_match(covariance_map()))  # type: ignore[arg-type]
    )
    monkeypatch.setattr(verifier, "_sample_candidate_uv", lambda _: fixed_points())
    original = verifier._verify_branch

    def fail_gate_on(*args, **kwargs):
        gate_enabled = args[-2]
        if gate_enabled:
            raise RuntimeError("gate-on test failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(verifier, "_verify_branch", fail_gate_on)
    query, candidate = query_and_candidate()
    results = verifier.verify_branches(
        torch.cat([pp.identity_SE3(1).tensor(), pp.identity_SE3(1).tensor()], dim=0),
        query,
        candidate,
        make_record(200, 1, 1),
        make_record(100, 0, 0),
        [True, False],
    )
    assert results[True][0].reject_code == "verification_exception"
    assert results[False][0].reject_code == "insufficient_geometry_points"


def test_covariance_none_is_compatible_but_not_comparable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    verifier = LoopCandidateVerifier(make_config(tmp_path), FakeFrontend(make_match(None)))  # type: ignore[arg-type]
    results = run_pair(verifier, monkeypatch, [True, False])
    for gate in (True, False):
        row = results[gate][0]
        assert row.comparison_applicable is False
        assert row.diagnostics["covariance_available"] is False  # type: ignore[index]
        assert row.diagnostics["gate_applied"] is False  # type: ignore[index]
        assert row.diagnostics["after_flow_cov"] == row.diagnostics["after_inbound_and_finite"]  # type: ignore[index]


def test_malformed_covariance_rejects_both_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    malformed = torch.ones((1, 2, 12, 12), dtype=torch.float32)
    match = make_match(None)
    match.cov = malformed
    verifier = LoopCandidateVerifier(make_config(tmp_path), FakeFrontend(match))  # type: ignore[arg-type]
    results = run_pair(verifier, monkeypatch, [True, False])
    assert {results[gate][0].reject_code for gate in (True, False)} == {"invalid_covariance_layout"}
    assert results[True][0].comparison_pair_id == results[False][0].comparison_pair_id


def test_common_rejection_is_paired_without_frontend_call(tmp_path: Path) -> None:
    frontend = FakeFrontend(make_match(covariance_map()))
    verifier = LoopCandidateVerifier(make_config(tmp_path), frontend)  # type: ignore[arg-type]
    current = make_record(100, 1, 1)
    historical = make_record(200, 0, 0)
    query = {"sensor_frame_idx": 100, "visual_map_idx": 1, "loop_frame_idx": 1}
    candidate = {"sensor_frame_idx": 200, "visual_map_idx": 0, "loop_frame_idx": 0}
    poses = torch.cat([pp.identity_SE3(1).tensor(), pp.identity_SE3(1).tensor()], dim=0)
    results = verifier.verify_branches(poses, query, candidate, current, historical, [True, False])
    assert frontend.calls == 0
    assert [results[gate][0].reject_code for gate in (True, False)] == [
        "invalid_candidate_order",
        "invalid_candidate_order",
    ]
    assert results[True][0].comparison_pair_id == results[False][0].comparison_pair_id == "1:0"


def test_no_candidate_depth_pixels_is_paired_before_frontend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frontend = FakeFrontend(make_match(covariance_map()))
    verifier = LoopCandidateVerifier(make_config(tmp_path), frontend)  # type: ignore[arg-type]
    monkeypatch.setattr(verifier, "_sample_candidate_uv", lambda _: torch.empty((0, 2)))
    query, candidate = query_and_candidate()
    results = verifier.verify_branches(
        torch.cat([pp.identity_SE3(1).tensor(), pp.identity_SE3(1).tensor()], dim=0),
        query,
        candidate,
        make_record(200, 1, 1),
        make_record(100, 0, 0),
        [True, False],
    )
    assert frontend.calls == 0
    assert verifier.frontend_inference_calls == verifier.candidates_reaching_frontend == 0
    assert [results[gate][0].reject_code for gate in (True, False)] == [
        "no_candidate_depth_pixels",
        "no_candidate_depth_pixels",
    ]


def test_optional_config_defaults_and_bool_numeric_rejection(tmp_path: Path) -> None:
    old_config = make_config(tmp_path, gate=None, compare=None)
    LoopClosureManager.is_valid_config(old_config)
    assert LoopClosureManager.primary_flow_cov_gate_enabled(old_config) is True
    assert LoopClosureManager.flow_cov_comparison_enabled(old_config) is False

    new_config = make_config(tmp_path, gate=False, compare=True)
    LoopClosureManager.is_valid_config(new_config)
    new_config.geometry.max_flow_cov = True
    with pytest.raises(ValueError):
        LoopClosureManager.is_valid_config(new_config)


def test_comparison_summary_excludes_non_applicable_pairs() -> None:
    applicable_on = {
        "comparison_pair_id": "1:0",
        "comparison_applicable": True,
        "status": "rejected",
        "pnp_attempted": False,
        "pnp_ransac_succeeded": None,
        "diagnostics": {"after_flow_cov": 0},
    }
    applicable_off = {
        **applicable_on,
        "status": "accepted",
        "pnp_attempted": True,
        "pnp_ransac_succeeded": True,
        "diagnostics": {"after_flow_cov": 4},
    }
    unavailable = {
        "comparison_pair_id": "2:0",
        "comparison_applicable": False,
        "status": "rejected",
        "pnp_attempted": None,
        "pnp_ransac_succeeded": None,
        "diagnostics": {},
    }
    summary = LoopClosureManager._comparison_summary(
        [applicable_on, unavailable], [applicable_off, dict(unavailable)]
    )
    assert summary["pair_count"] == 2
    assert summary["comparison_applicable_candidates"] == 1
    assert summary["after_flow_cov_compared_candidates"] == 1
    assert summary["after_flow_cov_changed_candidates"] == 1
    assert summary["pnp_ransac_result_compared_candidates"] == 0
    assert summary["pnp_ransac_result_changed_candidates"] == 0
    assert summary["accepted_result_changed_candidates"] == 1


def test_offline_query_limit_preserves_order_and_total() -> None:
    queries = [
        {"loop_frame_idx": 0, "candidates": []},
        {"loop_frame_idx": 1, "candidates": [{"id": 0}, {"id": 1}]},
        {"loop_frame_idx": 2, "candidates": [{"id": 2}, {"id": 3}]},
    ]
    limited = limit_queries(queries, 3)
    assert [query["loop_frame_idx"] for query in limited] == [0, 1, 2]
    assert [candidate["id"] for query in limited for candidate in query["candidates"]] == [0, 1, 2]
    assert queries[2]["candidates"] == [{"id": 2}, {"id": 3}]


def make_visual_map() -> VisualMap:
    graph = VisualMap()
    for _ in range(2):
        graph.frames.push(FrameNode.init({
            "pose": pp.identity_SE3(1),
            "T_BS": pp.identity_SE3(1),
            "need_interp": torch.tensor([False]),
            "time_ns": torch.tensor([0], dtype=torch.long),
            "K": torch.eye(3).unsqueeze(0),
            "baseline": torch.tensor([0.2]),
        }))
    return graph


def test_manager_writes_paired_strict_outputs_and_preserves_pose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path, gate=False, compare=True)
    manager = LoopClosureManager(config)
    output_dir = tmp_path / "loop_closure"
    record_dir = tmp_path / "cached_loop_closure"
    manager.set_output_dir(output_dir)
    current = make_record(200, 1, 1)
    historical = make_record(100, 0, 0)
    frame_dir = record_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    historical.save(frame_dir / "historical.pt")
    current.save(frame_dir / "current.pt")
    manager.records = [
        {"loop_frame_idx": 0, "sensor_frame_idx": 100, "file": "frames/historical.pt"},
        {"loop_frame_idx": 1, "sensor_frame_idx": 200, "file": "frames/current.pt"},
    ]
    frontend = FakeFrontend(make_match(covariance_map()))
    manager.set_frontend(frontend)  # type: ignore[arg-type]
    monkeypatch.setattr(LoopCandidateVerifier, "_sample_candidate_uv", lambda self, _: fixed_points())
    graph = make_visual_map()
    pose_before = graph.frames.data["pose"].tensor.clone()
    query, candidate = query_and_candidate()

    constraints = manager.verify_candidates(
        graph,
        [{**query, "candidates": [candidate]}],
        record_dir=record_dir,
    )
    assert constraints == []
    assert frontend.calls == 1
    assert torch.equal(graph.frames.data["pose"].tensor, pose_before)

    expected = {
        "loop_verification.json",
        "loop_constraints.json",
        "loop_verification_gate_enabled.json",
        "loop_verification_gate_disabled.json",
        "loop_constraints_gate_enabled.json",
        "loop_constraints_gate_disabled.json",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    primary = json.loads((output_dir / "loop_verification.json").read_text(encoding="utf-8"))
    enabled = json.loads((output_dir / "loop_verification_gate_enabled.json").read_text(encoding="utf-8"))
    disabled = json.loads((output_dir / "loop_verification_gate_disabled.json").read_text(encoding="utf-8"))
    assert primary["schema_version"] == 3
    assert primary["branch_gate_enabled"] is False
    assert primary["frontend_inference_calls"] == primary["candidates_reaching_frontend"] == 1
    assert primary["comparison_summary"]["pair_count"] == 1
    assert primary["comparison_summary"]["comparison_applicable_candidates"] == 1
    assert primary["comparison_summary"]["after_flow_cov_changed_candidates"] == 1
    assert primary["comparison_summary"]["changed_metrics_mother_set"] == "comparison_applicable_candidates"
    assert enabled["comparison_run_id"] == disabled["comparison_run_id"] == primary["comparison_run_id"]
    assert "Infinity" not in (output_dir / "loop_verification.json").read_text(encoding="utf-8")
    validation = validate_comparison_outputs(output_dir)
    assert validation["pair_count"] == 1
    assert validation["pair_ids_unique_and_aligned"] is True
    assert validation["gate_off_restored_flow_candidates"] == 1


def test_compare_disabled_only_writes_primary_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = make_config(tmp_path, gate=True, compare=False)
    manager = LoopClosureManager(config)
    output_dir = tmp_path / "loop_closure"
    manager.set_output_dir(output_dir)
    current = make_record(200, 1, 1)
    historical = make_record(100, 0, 0)
    historical.save(output_dir / "historical.pt")
    current.save(output_dir / "current.pt")
    manager.records = [
        {"loop_frame_idx": 0, "sensor_frame_idx": 100, "file": "historical.pt"},
        {"loop_frame_idx": 1, "sensor_frame_idx": 200, "file": "current.pt"},
    ]
    manager.set_frontend(FakeFrontend(make_match(covariance_map())))  # type: ignore[arg-type]
    monkeypatch.setattr(LoopCandidateVerifier, "_sample_candidate_uv", lambda self, _: fixed_points())
    query, candidate = query_and_candidate()
    manager.verify_candidates(make_visual_map(), [{**query, "candidates": [candidate]}])
    files = {path.name for path in output_dir.iterdir()}
    assert "loop_verification.json" in files
    assert "loop_constraints.json" in files
    assert not any("gate_enabled" in name or "gate_disabled" in name for name in files)
    payload = json.loads((output_dir / "loop_verification.json").read_text(encoding="utf-8"))
    assert payload["comparison_enabled"] is False
    assert "comparison_summary" not in payload


def test_phase_b5_observe_writes_isolated_branches_without_extra_frontend_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path, gate=False, compare=True)
    config.bow_min_score = SimpleNamespace()
    config.phase_b5 = SimpleNamespace(
        enabled=True,
        mode="observe",
        trusted_manifest=SimpleNamespace(),
        calibration=SimpleNamespace(
            prefix_fraction=0.2, min_queries=20, min_all_bow_pairs=100,
            min_orb_pairs=30, absolute_median_log_risk_cap=SimpleNamespace(),
            absolute_q95_log_risk_cap=SimpleNamespace(),
        ),
        orb=SimpleNamespace(ratio=0.8, max_depth=20.0),
        flow=SimpleNamespace(
            min_valid_points=4, min_valid_ratio=0.0, min_grid_cells=1,
            nms_kernel_size=3, border=0, min_points=4, max_points=16,
            max_depth=20.0, grid_rows=2, grid_cols=2, max_points_per_cell=4,
        ),
    )
    LoopClosureManager.is_valid_config(config)
    manager = LoopClosureManager(config)
    assert manager.config.bow_min_score is None
    assert manager.config.phase_b5.trusted_manifest is None
    assert manager.config.phase_b5.calibration.absolute_median_log_risk_cap is None
    assert manager.config.phase_b5.calibration.absolute_q95_log_risk_cap is None
    output_dir = tmp_path / "loop_closure"
    record_dir = tmp_path / "cache"
    manager.set_output_dir(output_dir)
    current = make_record(200, 1, 1)
    historical = make_record(100, 0, 0)
    (record_dir / "frames").mkdir(parents=True)
    historical.save(record_dir / "frames/historical.pt")
    current.save(record_dir / "frames/current.pt")
    manager.records = [
        {"loop_frame_idx": 0, "sensor_frame_idx": 100, "file": "frames/historical.pt"},
        {"loop_frame_idx": 1, "sensor_frame_idx": 200, "file": "frames/current.pt"},
    ]
    frontend = FakeFrontend(make_match(covariance_map()))
    manager.set_frontend(frontend)  # type: ignore[arg-type]
    monkeypatch.setattr(LoopCandidateVerifier, "_sample_candidate_uv", lambda self, _: fixed_points())
    query, candidate = query_and_candidate()
    manager.verify_candidates(
        make_visual_map(), [{**query, "candidates": [candidate]}], record_dir=record_dir
    )
    assert frontend.calls == 1
    assert (output_dir / "orb_observe/verification.json").is_file()
    assert (output_dir / "flow_all_bow_observe/verification.json").is_file()
    assert (output_dir / "flow_orb_supported_observe/verification.json").is_file()
    assert (output_dir / "forced_control_shadow/verification.json").is_file()
    orb_branch = json.loads(
        (output_dir / "orb_observe/verification.json").read_text(encoding="utf-8")
    )
    assert len(orb_branch["rows"]) == 1
    assert orb_branch["rows"][0]["pair_id"] == "1:0"
    calibration = json.loads(
        (output_dir / "phase_b5_calibration_manifest.json").read_text(encoding="utf-8")
    )
    assert calibration["prefix_query_count"] == 1
    assert calibration["populations"]["all_bow_candidates"]["trusted"] is False
    main = json.loads((output_dir / "loop_verification.json").read_text(encoding="utf-8"))
    assert main["phase_b5"]["mode"] == "observe"
    assert main["phase_b5"]["pose_invariant"] is True
