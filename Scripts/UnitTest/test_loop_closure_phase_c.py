from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypose as pp
import pytest
import torch

from Module.LoopClosure.VINSGeometry import network_refinement_contract
from Scripts.AdHoc import RunLoopPhaseCOffline as phase_c


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _phase_b_inputs(root: Path, edge_count: int = 4) -> None:
    records = []
    verifications = []
    fixed = []
    covariance = []
    identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    fixed_information = np.eye(6).tolist()
    covariance_information = (np.eye(6) * 0.5).tolist()
    for offset in range(edge_count):
        candidate_sensor = 100 + offset
        current_sensor = 200 + offset
        candidate_visual = 10 + offset
        current_visual = 20 + offset
        candidate_loop = 30 + offset
        current_loop = 40 + offset
        records.extend((
            {
                "sensor_frame_idx": candidate_sensor,
                "visual_map_idx": candidate_visual,
                "loop_frame_idx": candidate_loop,
            },
            {
                "sensor_frame_idx": current_sensor,
                "visual_map_idx": current_visual,
                "loop_frame_idx": current_loop,
            },
        ))
        verifications.append({
            "pair_id": f"{current_loop}:{candidate_loop}",
            "current_sensor_frame_idx": current_sensor,
            "candidate_sensor_frame_idx": candidate_sensor,
            "selected_for_query": True,
            "pgo_comparison_eligible": True,
            "information_observe": {
                "valid": True,
                "rank": 6,
                "raw_eigenvalues": [1.0] * 6,
                "edge_eigenvalues": [0.5] * 6,
                "generalized_lambda_max": 2.0,
                "alpha": 0.5,
                "used_matrix": covariance_information,
            },
        })
        base = {
            "src_visual_map_idx": candidate_visual,
            "dst_visual_map_idx": current_visual,
            "src_sensor_frame_idx": candidate_sensor,
            "dst_sensor_frame_idx": current_sensor,
            "relative_pose": identity,
        }
        fixed.append({**base, "information": fixed_information})
        covariance.append({**base, "information": covariance_information})

    verifications.append({
        "pair_id": "999:998",
        "selected_for_query": False,
        "pgo_comparison_eligible": True,
    })
    _write_json(root / "source_index.json", {"records": records})
    _write_json(root / "loop_vins_verification.json", {"verifications": verifications})
    _write_json(root / "loop_constraints_pgo_fixed.json", {"constraints": fixed})
    _write_json(root / "loop_constraints_pgo_covariance.json", {"constraints": covariance})


def test_phase_c_edge_association_filters_and_fixes_direction(tmp_path: Path) -> None:
    _phase_b_inputs(tmp_path, edge_count=3)
    verification, fixed, covariance = phase_c.load_phase_c_edges(tmp_path)
    assert len(verification) == len(fixed) == len(covariance) == 3
    assert [phase_c._edge_key(row) for row in fixed] == [
        (10, 20), (11, 21), (12, 22),
    ]
    assert [phase_c._edge_key(row) for row in fixed] == [
        phase_c._edge_key(row) for row in covariance
    ]


def test_phase_c_edge_association_allows_zero_edges(tmp_path: Path) -> None:
    _phase_b_inputs(tmp_path, edge_count=0)
    assert phase_c.load_phase_c_edges(tmp_path) == ([], [], [])


def test_phase_c_edge_association_rejects_duplicate_selected_edge(tmp_path: Path) -> None:
    _phase_b_inputs(tmp_path)
    path = tmp_path / "loop_vins_verification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["verifications"].append(dict(payload["verifications"][0]))
    _write_json(path, payload)
    with pytest.raises(ValueError, match="duplicate selected verification edge"):
        phase_c.load_phase_c_edges(tmp_path)


def test_phase_c_edge_association_rejects_relative_pose_difference(tmp_path: Path) -> None:
    _phase_b_inputs(tmp_path)
    path = tmp_path / "loop_constraints_pgo_covariance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["constraints"][0]["relative_pose"][0] = 1.0
    _write_json(path, payload)
    with pytest.raises(ValueError, match="relative poses differ"):
        phase_c.load_phase_c_edges(tmp_path)


def test_phase_c_rejects_disp_or_mismatched_icp_manifest(tmp_path: Path) -> None:
    refinement = SimpleNamespace(
        residual_mode="icp", kernel_size=31, match_cov_default=0.25,
        min_depth_cov=0.05, min_flow_cov=0.25,
    )
    contract = network_refinement_contract(refinement)
    for name in (
        "offline_run_manifest.json", "loop_vins_verification.json",
        "loop_constraints_pgo_fixed.json", "loop_constraints_pgo_covariance.json",
    ):
        _write_json(tmp_path / name, {
            **contract,
            "vo_graph_type": "icp",
            "odometry_residual_mode": "icp",
        })
    config = SimpleNamespace(Odometry=SimpleNamespace(
        optimizer=SimpleNamespace(args=SimpleNamespace(graph_type="icp")),
        global_pgo=SimpleNamespace(observation_residual_mode="icp"),
        loop_closure=SimpleNamespace(vins_geometry=SimpleNamespace(
            network_refinement=refinement,
        )),
        cov=SimpleNamespace(obs=SimpleNamespace(type="MatchCovariance", args=SimpleNamespace(
            kernel_size=31, match_cov_default=0.25,
            min_depth_cov=0.05, min_flow_cov=0.25,
        ))),
    ))
    observed = phase_c.validate_icp_phase_b_contract(tmp_path, config)
    assert observed["loop_residual_mode"] == "icp"
    payload = json.loads((tmp_path / "loop_constraints_pgo_fixed.json").read_text())
    payload["loop_residual_mode"] = "disp"
    _write_json(tmp_path / "loop_constraints_pgo_fixed.json", payload)
    with pytest.raises(ValueError, match="residual modes disagree"):
        phase_c.validate_icp_phase_b_contract(tmp_path, config)


def test_phase_c_rejects_icp_with_disp_odometry_mode(tmp_path: Path) -> None:
    refinement = SimpleNamespace(
        residual_mode="icp", kernel_size=31, match_cov_default=0.25,
        min_depth_cov=0.05, min_flow_cov=0.25,
    )
    contract = network_refinement_contract(refinement)
    for name in (
        "offline_run_manifest.json", "loop_vins_verification.json",
        "loop_constraints_pgo_fixed.json", "loop_constraints_pgo_covariance.json",
    ):
        _write_json(tmp_path / name, {
            **contract,
            "vo_graph_type": "icp",
            "odometry_residual_mode": "disp",
        })
    config = SimpleNamespace(Odometry=SimpleNamespace(
        optimizer=SimpleNamespace(args=SimpleNamespace(graph_type="icp")),
        global_pgo=SimpleNamespace(observation_residual_mode="disp"),
        loop_closure=SimpleNamespace(vins_geometry=SimpleNamespace(
            network_refinement=refinement,
        )),
        cov=SimpleNamespace(obs=SimpleNamespace(type="MatchCovariance", args=SimpleNamespace(
            kernel_size=31, match_cov_default=0.25,
            min_depth_cov=0.05, min_flow_cov=0.25,
        ))),
    ))
    with pytest.raises(ValueError, match="observation_residual_mode disagrees"):
        phase_c.validate_phase_b_contract(tmp_path, config)


def test_phase_c_accepts_legacy_disp_ablation(tmp_path: Path) -> None:
    for name in (
        "offline_run_manifest.json", "loop_vins_verification.json",
        "loop_constraints_pgo_fixed.json", "loop_constraints_pgo_covariance.json",
    ):
        _write_json(tmp_path / name, {})
    config = SimpleNamespace(Odometry=SimpleNamespace(
        optimizer=SimpleNamespace(args=SimpleNamespace(graph_type="disp")),
    ))
    observed = phase_c.validate_phase_b_contract(tmp_path, config)
    assert observed == {
        "vo_graph_type": "disp",
        "loop_residual_mode": "disp",
        "odometry_residual_mode": "disp",
        "observation_covariance_model": "legacy_reprojection_disparity",
        "kernel_size": None,
        "covariance_config_sha256": None,
        "contract_validation": "legacy_disp_source_config",
    }


def test_phase_c_rejects_partial_legacy_disp_metadata(tmp_path: Path) -> None:
    for name in (
        "offline_run_manifest.json", "loop_vins_verification.json",
        "loop_constraints_pgo_fixed.json", "loop_constraints_pgo_covariance.json",
    ):
        _write_json(tmp_path / name, {})
    payload = {"covariance_config_sha256": "partial-new-contract"}
    _write_json(tmp_path / "loop_constraints_pgo_fixed.json", payload)
    config = SimpleNamespace(Odometry=SimpleNamespace(
        optimizer=SimpleNamespace(args=SimpleNamespace(graph_type="disp")),
    ))
    with pytest.raises(ValueError, match="partial contract metadata"):
        phase_c.validate_phase_b_contract(tmp_path, config)


def test_phase_c_accepts_explicit_disp_ablation(tmp_path: Path) -> None:
    contract = network_refinement_contract(SimpleNamespace(residual_mode="disp"))
    for name in (
        "offline_run_manifest.json", "loop_vins_verification.json",
        "loop_constraints_pgo_fixed.json", "loop_constraints_pgo_covariance.json",
    ):
        _write_json(tmp_path / name, {
            **contract,
            "vo_graph_type": "disp",
            "odometry_residual_mode": "disp",
        })
    config = SimpleNamespace(Odometry=SimpleNamespace(
        optimizer=SimpleNamespace(args=SimpleNamespace(graph_type="disp")),
        global_pgo=SimpleNamespace(observation_residual_mode="disp"),
    ))
    observed = phase_c.validate_phase_b_contract(tmp_path, config)
    assert observed["loop_residual_mode"] == "disp"
    assert observed["contract_validation"] == "explicit_disp"


def _metrics(ate: float, rpe: float) -> dict:
    return {
        "ate": {"rmse": ate},
        "rpe": {"rmse": rpe},
    }


def test_phase_c_classification_boundaries() -> None:
    baseline = _metrics(1.0, 2.0)
    residual_before = {"rmse": 3.0}
    positive = phase_c.classify_branch(
        baseline, _metrics(1.0, 2.0), residual_before, {"rmse": 2.0}, True,
    )
    assert positive["classification"] == "positive"
    assert positive["online_experiment_eligible"] is True

    inconclusive = phase_c.classify_branch(
        baseline, _metrics(0.9, 2.0), residual_before, {"rmse": 3.0}, True,
    )
    assert inconclusive["classification"] == "inconclusive"
    assert inconclusive["online_experiment_eligible"] is False

    failed = phase_c.classify_branch(
        baseline, _metrics(1.1, 2.1), residual_before, {"rmse": 2.0}, True,
    )
    assert failed["classification"] == "failed"
    missing = phase_c.classify_branch(
        baseline, {"ate": None, "rpe": None}, residual_before, {"rmse": 2.0}, True,
    )
    assert missing["classification"] == "failed"


def test_sensor_to_body_validation_accepts_quaternion_sign(tmp_path: Path) -> None:
    sensor = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    extrinsic = sensor.clone()
    timestamps = torch.tensor([123], dtype=torch.long)
    source = np.array([[123.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -1.0]])
    phase_c.validate_source_trajectory(source, sensor, extrinsic, timestamps)


def test_sensor_to_body_matches_float32_export_with_rotated_extrinsic() -> None:
    sensor = torch.tensor([
        [0.1, -0.2, 0.3, 0.92736185, 125000.125, -98000.25, 76500.5],
        [-0.3, 0.2, 0.1, 0.92736185, -225000.5, 198000.75, -176500.25],
    ], dtype=torch.float32)
    body_to_sensor = torch.tensor([
        [0.70710677, 0.0, 0.0, 0.70710677, 1.25, -2.5, 3.75],
        [0.70710677, 0.0, 0.0, 0.70710677, 1.25, -2.5, 3.75],
    ], dtype=torch.float32)
    timestamps = torch.tensor([1000, 2000], dtype=torch.long)
    expected = (
        pp.SE3(body_to_sensor) @ pp.SE3(sensor) @ pp.SE3(body_to_sensor).Inv()
    ).tensor().numpy()
    reconstructed = phase_c.sensor_to_body_timed(sensor, body_to_sensor, timestamps.numpy())
    assert np.array_equal(reconstructed[:, 1:], expected)
    source = np.concatenate([timestamps.numpy().reshape(-1, 1), expected], axis=1)
    phase_c.validate_source_trajectory(source, sensor, body_to_sensor, timestamps)


def test_no_loop_branch_copies_source_pose_and_optional_status(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "no_loop"
    source.mkdir()
    poses = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    np.save(source / "poses.npy", poses)
    np.save(source / "ref_poses.npy", poses)
    (source / "config.yaml").write_text("Project: test\n", encoding="utf-8")
    (source / "frame_status.pth").write_bytes(b"status")
    phase_c.write_evaluation_branch(target, "no_loop", None, source, {})
    assert (target / "poses.npy").read_bytes() == (source / "poses.npy").read_bytes()
    assert (target / "frame_status.pth").read_bytes() == b"status"
    assert np.array_equal(np.load(target / "poses.npy"), poses)


def test_formal_metrics_uses_frozen_parameters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = SimpleNamespace(data=SimpleNamespace(as_evo=object()))
    monkeypatch.setattr(phase_c.Sandbox, "load", lambda path: object())
    monkeypatch.setattr(
        phase_c.Trajectory, "from_sandbox",
        lambda box, align_time: (fake, fake),
    )
    calls = []

    def metric(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stats={"mean": 1.0, "std": 2.0, "rmse": 3.0})

    monkeypatch.setattr(phase_c, "evaluateATE", metric)
    monkeypatch.setattr(phase_c, "evaluateRPE", metric)
    result = phase_c.formal_metrics(tmp_path)
    assert calls == [{"correct_scale": False}, {"correct_scale": False}]
    assert result["parameters"] == {
        "trajectory_preprocess_align_origin": True,
        "metric_align": True,
        "metric_align_origin": False,
        "correct_scale": False,
        "ate_pose_relation": "translation_part",
        "rpe_pose_relation": "full_transformation",
        "rpe_delta_frames": 1,
    }
    assert result["ate"]["rmse"] == 3.0


def test_branch_metadata_marks_original_fallback(tmp_path: Path) -> None:
    phase_c.write_branch_metadata(tmp_path, "fixed_information", {
        "trajectory_source": "original_fallback",
        "pgo_safe": False,
        "online_experiment_eligible": False,
    })
    payload = (tmp_path / "metadata.yaml").read_text(encoding="utf-8")
    assert "trajectory_source: original_fallback" in payload
    assert "pgo_safe: false" in payload
    assert "online_experiment_eligible: false" in payload
