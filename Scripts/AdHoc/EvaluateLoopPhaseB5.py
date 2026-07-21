from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON token {value!r} in {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
    )
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _labels(path: Path) -> dict[str, str]:
    payload = _load(path)
    if payload.get("frozen") is not True:
        raise ValueError("Phase B.5 evaluation requires frozen labels")
    if payload.get("pair_match_policy") != "exact loop_frame_idx pair":
        raise ValueError("Phase B.5 requires exact loop-frame pair labels")
    return {str(item["pair_id"]): str(item["label"]) for item in payload.get("items", [])}


def _complete_query_ids(rows: list[dict[str, Any]], labels: dict[str, str], end_sensor: int) -> list[int]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row["current_sensor_frame_idx"]) <= end_sensor:
            continue
        grouped.setdefault(int(str(row["pair_id"]).split(":", 1)[0]), []).append(row)
    return sorted(query for query, items in grouped.items() if items and all(str(item["pair_id"]) in labels for item in items))


def gate_metrics(
    rows: list[dict[str, Any]], labels: dict[str, str], end_sensor: int,
    pass_fn,
) -> dict[str, Any]:
    query_ids = _complete_query_ids(rows, labels, end_sensor)
    selected = [
        row for row in rows
        if int(str(row["pair_id"]).split(":", 1)[0]) in query_ids
        and labels.get(str(row["pair_id"])) in {"true", "false"}
    ]
    true_rows = [row for row in selected if labels[str(row["pair_id"])] == "true"]
    false_rows = [row for row in selected if labels[str(row["pair_id"])] == "false"]
    positive_queries = {int(str(row["pair_id"]).split(":", 1)[0]) for row in true_rows}
    survived_true = [row for row in true_rows if pass_fn(row)]
    survived_false = [row for row in false_rows if pass_fn(row)]
    survived_positive_queries = {
        int(str(row["pair_id"]).split(":", 1)[0]) for row in survived_true
    }
    false_before_per_query = _ratio(len(false_rows), len(query_ids))
    false_after_per_query = _ratio(len(survived_false), len(query_ids))
    false_reduction = (
        None if false_before_per_query in {None, 0.0} or false_after_per_query is None
        else 1.0 - false_after_per_query / false_before_per_query
    )
    true_survival = _ratio(len(survived_true), len(true_rows))
    positive_survival = _ratio(len(survived_positive_queries), len(positive_queries))
    sufficient = len(positive_queries) >= 10 and len(true_rows) >= 20 and len(false_rows) >= 20
    checks = {
        "evaluation_set_sufficient": sufficient,
        "positive_query_survival_drop_lte_0_05": None if positive_survival is None else 1.0 - positive_survival <= 0.05,
        "true_pair_survival_gte_0_90": None if true_survival is None else true_survival >= 0.90,
        "reviewed_false_per_query_reduction_gte_0_30": None if false_reduction is None else false_reduction >= 0.30,
    }
    return {
        "fixed_post_prefix_query_count": len(query_ids),
        "positive_query_count": len(positive_queries),
        "reviewed_true_pairs": len(true_rows), "reviewed_false_pairs": len(false_rows),
        "survived_true_pairs": len(survived_true), "survived_false_pairs": len(survived_false),
        "positive_query_survival_rate": positive_survival,
        "true_pair_survival_rate": true_survival,
        "false_candidates_per_query_before": false_before_per_query,
        "false_candidates_per_query_after": false_after_per_query,
        "false_candidates_per_query_reduction": false_reduction,
        "checks": checks,
        "promotion_metrics_pass": sufficient and all(value is True for value in checks.values()),
    }


def selector_metrics(
    rows: list[dict[str, Any]], labels: dict[str, str], end_sensor: int, population: str,
    legacy_by_pair: dict[str, dict[str, Any]], pose_error_fn, role: str,
) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if population in row.get("populations", [])
        and ((row.get("flow") or {}).get("pair_gate_by_population") or {}).get(population) is True
    ]
    query_ids = _complete_query_ids(eligible, labels, end_sensor)
    true_rows = [
        row for row in eligible
        if int(str(row["pair_id"]).split(":", 1)[0]) in query_ids
        and labels.get(str(row["pair_id"])) == "true"
    ]
    false_rows = [
        row for row in eligible
        if int(str(row["pair_id"]).split(":", 1)[0]) in query_ids
        and labels.get(str(row["pair_id"])) == "false"
    ]
    positive_queries = {
        int(str(row["pair_id"]).split(":", 1)[0]) for row in true_rows
    }
    shadows = [(row.get("selector_shadow_by_population") or {}).get(population) for row in true_rows]
    false_shadows = [(row.get("selector_shadow_by_population") or {}).get(population) for row in false_rows]
    unavailable = sum(item is None for item in shadows + false_shadows)
    true_pnp_attempted = sum(bool(item and item.get("pnp_attempted")) for item in shadows)
    true_pnp_succeeded = sum(bool(item and item.get("pnp_ransac_succeeded")) for item in shadows)
    false_accepted = sum(bool(item and item.get("status") == "accepted") for item in false_shadows)
    legacy_true_attempted = sum(legacy_by_pair.get(str(row["pair_id"]), {}).get("pnp_attempted") is True for row in true_rows)
    legacy_true_succeeded = sum(legacy_by_pair.get(str(row["pair_id"]), {}).get("pnp_ransac_succeeded") is True for row in true_rows)
    legacy_false_accepted = sum(legacy_by_pair.get(str(row["pair_id"]), {}).get("status") == "accepted" for row in false_rows)
    paired_errors: list[tuple[float, float, float, float]] = []
    legacy_large = 0
    shadow_large = 0
    controls: dict[str, Any] = {}
    for row in true_rows + false_rows:
        pair = str(row["pair_id"])
        legacy = legacy_by_pair.get(pair, {})
        shadow = (row.get("selector_shadow_by_population") or {}).get(population) or {}
        legacy_error = pose_error_fn(legacy, row)
        shadow_error = pose_error_fn(shadow, row)
        if legacy.get("status") == "accepted" and legacy_error is not None:
            legacy_large += legacy_error[0] > 3.0 or legacy_error[1] > 15.0
        if shadow.get("status") == "accepted" and shadow_error is not None:
            shadow_large += shadow_error[0] > 3.0 or shadow_error[1] > 15.0
        if labels.get(pair) == "true" and legacy_error is not None and shadow_error is not None:
            paired_errors.append((legacy_error[0], legacy_error[1], shadow_error[0], shadow_error[1]))
        sensor_pair = f"{int(row['current_sensor_frame_idx'])}:{int(row['candidate_sensor_frame_idx'])}"
        if sensor_pair in {"1200:1100", "1250:470", "1250:480"}:
            controls[sensor_pair] = {
                "legacy_status": legacy.get("status"), "shadow_status": shadow.get("status"),
                "legacy_error": legacy_error, "shadow_error": shadow_error,
            }
    def median_at(index: int) -> float | None:
        if not paired_errors:
            return None
        values = sorted(item[index] for item in paired_errors)
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else 0.5 * (values[middle - 1] + values[middle])
    legacy_t, legacy_r, shadow_t, shadow_r = (median_at(i) for i in range(4))
    reach_not_lower = true_pnp_attempted >= legacy_true_attempted
    success_not_lower = true_pnp_succeeded >= legacy_true_succeeded
    pose_noninferior = (
        None if None in {legacy_t, legacy_r, shadow_t, shadow_r}
        else shadow_t <= legacy_t + 0.10 and shadow_r <= legacy_r + 1.0
    )
    control_true = controls.get("1200:1100")
    control_true_pass = None
    if control_true and control_true["legacy_error"] is not None and control_true["shadow_error"] is not None:
        control_true_pass = (
            control_true["shadow_error"][0] <= control_true["legacy_error"][0] + 0.10
            and control_true["shadow_error"][1] <= control_true["legacy_error"][1] + 1.0
        )
    false_controls_pass = all(
        not (controls.get(pair, {}).get("legacy_status") != "accepted" and controls.get(pair, {}).get("shadow_status") == "accepted")
        for pair in ("1250:470", "1250:480")
    )
    checks = {
        "evaluation_set_sufficient": (
            len(positive_queries) >= 10 and len(true_rows) >= 20 and len(false_rows) >= 20
        ),
        "true_pnp_attempt_rate_not_lower": reach_not_lower,
        "true_pnp_success_rate_not_lower": success_not_lower,
        "paired_true_median_pose_noninferior": pose_noninferior,
        "reviewed_false_accepted_not_increased": false_accepted <= legacy_false_accepted,
        "accepted_large_error_not_increased": shadow_large <= legacy_large,
    }
    if role == "development":
        checks["control_1200_1100_noninferior"] = control_true_pass
        checks["false_controls_not_restored"] = false_controls_pass
    decision = all(value is True for value in checks.values()) and unavailable == 0
    information_available = 0
    information_invalid = 0
    information_rank_deficient = 0
    information_high_condition = 0
    for row in true_rows + false_rows:
        refinement = (row.get("refinement_by_population") or {}).get(population) or {}
        information = refinement.get("information_at_pnp_pose") or {}
        matrix = information.get("matrix")
        if matrix is None:
            continue
        import numpy as np
        value = np.asarray(matrix, dtype=np.float64)
        information_available += 1
        symmetric = np.allclose(value, value.T, atol=1e-8, rtol=1e-8)
        eigen = np.linalg.eigvalsh(0.5 * (value + value.T)) if value.shape == (6, 6) else np.asarray([-1.0])
        if value.shape != (6, 6) or not np.isfinite(value).all() or not symmetric or float(eigen.min()) < -1e-8:
            information_invalid += 1
        if int(information.get("rank", 0)) < 6:
            information_rank_deficient += 1
        condition = information.get("condition_number_positive_subspace")
        if condition is None or float(condition) > 1e12:
            information_high_condition += 1
    return {
        "population": population,
        "fixed_post_prefix_query_count": len(query_ids),
        "positive_query_count": len(positive_queries),
        "reviewed_true_pairs": len(true_rows), "reviewed_false_pairs": len(false_rows),
        "shadow_unavailable_pairs": unavailable,
        "true_pnp_attempted": true_pnp_attempted,
        "true_pnp_succeeded": true_pnp_succeeded,
        "reviewed_false_accepted": false_accepted,
        "legacy_true_pnp_attempted": legacy_true_attempted,
        "legacy_true_pnp_succeeded": legacy_true_succeeded,
        "legacy_reviewed_false_accepted": legacy_false_accepted,
        "paired_pose_error_count": len(paired_errors),
        "median_translation_error_m": {"legacy": legacy_t, "shadow": shadow_t},
        "median_rotation_error_deg": {"legacy": legacy_r, "shadow": shadow_r},
        "accepted_large_error": {"legacy": legacy_large, "shadow": shadow_large},
        "controls": controls, "checks": checks,
        "information_diagnostics": {
            "available": information_available,
            "invalid_symmetric_or_psd": information_invalid,
            "rank_deficient": information_rank_deficient,
            "condition_number_gt_1e12_or_null": information_high_condition,
            "mathematics_pass": information_available >= 20 and information_invalid == 0,
            "numerical_conditioning_pass": (
                information_available >= 20
                and information_rank_deficient == 0
                and information_high_condition == 0
            ),
            "scale_calibration_pass": False,
        },
        "promotion_decision": decision,
        "reason": None if decision else "one or more pre-registered selector checks failed or were unavailable",
    }


def _pose_matrix(pose: list[float]) -> list[list[float]]:
    import numpy as np
    value = np.asarray(pose, dtype=np.float64)
    translation, quaternion = value[:3], value[3:]
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    x, y, z, w = quaternion
    rotation = np.asarray([
        [1 - 2*(y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix.tolist()


def _pose_error_factory(ref_poses_path: Path, index_path: Path, record_dir: Path):
    import numpy as np
    from Module.LoopClosure.Record import LoopFrameRecord
    reference = np.load(ref_poses_path, allow_pickle=False)
    index = _load(index_path)
    metadata = {int(item["sensor_frame_idx"]): item for item in index.get("records", [])}
    def pose_error(verification: dict[str, Any], row: dict[str, Any]) -> tuple[float, float] | None:
        pose = verification.get("pnp_relative_pose")
        current_idx = int(row["current_sensor_frame_idx"])
        candidate_idx = int(row["candidate_sensor_frame_idx"])
        if pose is None or not (0 <= candidate_idx < len(reference) and 0 <= current_idx < len(reference)):
            return None
        try:
            current_record = LoopFrameRecord.load(record_dir / metadata[current_idx]["file"])
            candidate_record = LoopFrameRecord.load(record_dir / metadata[candidate_idx]["file"])
        except (KeyError, OSError, ValueError, TypeError):
            return None
        if int(reference[current_idx, 0]) != current_record.frame_ns or int(reference[candidate_idx, 0]) != candidate_record.frame_ns:
            return None
        world_current = np.asarray(_pose_matrix(reference[current_idx, 1:].tolist()))
        world_candidate = np.asarray(_pose_matrix(reference[candidate_idx, 1:].tolist()))
        gt = np.linalg.inv(world_current) @ world_candidate
        estimated = np.asarray(_pose_matrix(pose))
        error = np.linalg.inv(gt) @ estimated
        translation = float(np.linalg.norm(error[:3, 3]))
        cosine = float(np.clip((np.trace(error[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        return translation, float(np.degrees(np.arccos(cosine)))
    return pose_error


def evaluate(
    branch_path: Path, manifest_path: Path, labels_path: Path,
    legacy_path: Path, ref_poses_path: Path, index_path: Path,
    record_dir: Path, role: str, output: Path,
) -> None:
    branch = _load(branch_path)
    manifest = _load(manifest_path)
    labels = _labels(labels_path)
    if int(branch.get("schema_version", -1)) != 1 or int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported Phase B.5 schema")
    if branch.get("branch_id") != "flow_all_bow_observe":
        raise ValueError(
            "evaluation requires flow_all_bow_observe so ORB and both Flow populations "
            "share the same BoW candidate universe"
        )
    for digest_key in (
        "input_digest", "cache_index_digest", "cache_content_digest",
        "phase_b5_config_sha256",
    ):
        if branch.get(digest_key) != manifest.get(digest_key):
            raise ValueError(f"branch and calibration manifest {digest_key} values differ")
    end_sensor = manifest.get("calibration_end_sensor_frame_idx")
    if not isinstance(end_sensor, int):
        raise ValueError("calibration boundary is unavailable")
    rows = branch.get("rows")
    if not isinstance(rows, list):
        raise ValueError("branch contains no rows")
    legacy_payload = _load(legacy_path)
    legacy_rows = legacy_payload.get("verifications")
    if not isinstance(legacy_rows, list):
        raise ValueError("legacy verification contains no rows")
    legacy_by_pair = {str(row["comparison_pair_id"]): row for row in legacy_rows}
    pose_error_fn = _pose_error_factory(ref_poses_path, index_path, record_dir)
    orb = gate_metrics(rows, labels, end_sensor, lambda row: bool((row.get("orb") or {}).get("orb_gate_pass")))
    flow: dict[str, Any] = {}
    selectors: dict[str, Any] = {}
    for population in ("all_bow_candidates", "orb_supported_candidates"):
        population_rows = [row for row in rows if population in row.get("populations", [])]
        flow[population] = gate_metrics(
            population_rows, labels, end_sensor,
            lambda row, p=population: bool(((row.get("flow") or {}).get("pair_gate_by_population") or {}).get(p)),
        )
        legacy_large = 0
        survived_large = 0
        for row in population_rows:
            if int(row["current_sensor_frame_idx"]) <= end_sensor:
                continue
            legacy = legacy_by_pair.get(str(row["pair_id"]), {})
            if legacy.get("status") != "accepted":
                continue
            error = pose_error_fn(legacy, row)
            if error is None or not (error[0] > 3.0 or error[1] > 15.0):
                continue
            legacy_large += 1
            if bool(((row.get("flow") or {}).get("pair_gate_by_population") or {}).get(population)):
                survived_large += 1
        flow[population]["accepted_large_error"] = {
            "input_population": legacy_large, "after_pair_gate": survived_large,
        }
        flow[population]["checks"]["accepted_large_error_not_increased"] = survived_large <= legacy_large
        flow[population]["promotion_metrics_pass"] = (
            flow[population]["checks"]["evaluation_set_sufficient"] is True
            and all(value is True for value in flow[population]["checks"].values())
        )
        if population == "orb_supported_candidates" and flow[population]["reviewed_false_pairs"] < 20:
            flow[population]["incremental_uncertainty_evidence_insufficient"] = True
            flow[population]["promotion_metrics_pass"] = False
        selectors[population] = selector_metrics(
            rows, labels, end_sensor, population, legacy_by_pair, pose_error_fn, role
        )
    promotable_populations = [
        population for population in ("all_bow_candidates", "orb_supported_candidates")
        if flow[population].get("promotion_metrics_pass") is True
        and selectors[population].get("promotion_decision") is True
    ]
    information_math_pass = any(
        (selectors[population].get("information_diagnostics") or {}).get("mathematics_pass") is True
        for population in promotable_populations
    )
    information_scale_pass = any(
        (selectors[population].get("information_diagnostics") or {}).get("scale_calibration_pass") is True
        for population in promotable_populations
    )
    information_numerical_pass = any(
        (selectors[population].get("information_diagnostics") or {}).get(
            "numerical_conditioning_pass"
        ) is True
        for population in promotable_populations
    )
    phase_c_ready_populations = [
        population for population in promotable_populations
        if (selectors[population].get("information_diagnostics") or {}).get("mathematics_pass") is True
        and (selectors[population].get("information_diagnostics") or {}).get(
            "numerical_conditioning_pass"
        ) is True
        and (selectors[population].get("information_diagnostics") or {}).get(
            "scale_calibration_pass"
        ) is True
    ]
    output_integrity = (
        int(legacy_payload.get("schema_version", -1)) == 3
        and legacy_payload.get("frontend_inference_calls")
        == legacy_payload.get("candidates_reaching_frontend")
        and legacy_payload.get("phase_b5", {}).get("pose_invariant") is True
        and branch.get("comparison_run_id") == manifest.get("comparison_run_id")
    )
    phase_c_checks = {
        "flow_pair_gate_and_selector_promotable": bool(promotable_populations),
        "information_mathematics_pass": information_math_pass,
        "information_numerical_conditioning_pass": information_numerical_pass,
        "information_scale_calibration_pass": information_scale_pass,
        "same_population_passes_all_flow_and_information_checks": bool(
            phase_c_ready_populations
        ),
        "pose_single_inference_strict_output_pass": output_integrity,
    }
    phase_c_admission = all(value is True for value in phase_c_checks.values())
    payload = {
        "schema_version": 1,
        "recall_scope": "post-prefix frozen reviewed overlap evaluation set",
        "pair_match_policy": "exact loop_frame_idx pair",
        "evaluation_role": role,
        "branch_sha256": _sha256(branch_path), "manifest_sha256": _sha256(manifest_path),
        "labels_sha256": _sha256(labels_path),
        "legacy_verification_sha256": _sha256(legacy_path),
        "ref_poses_sha256": _sha256(ref_poses_path),
        "cache_index_sha256": _sha256(index_path),
        "calibration_end_sensor_frame_idx": end_sensor,
        "orb_gate": orb, "flow_pair_gate": flow, "selector_shadow": selectors,
        "gt_large_error_check": {
            population: flow[population]["accepted_large_error"] for population in flow
        },
        "promotable_flow_populations": promotable_populations,
        "phase_c_candidate_populations": phase_c_ready_populations,
        "phase_c_checks": phase_c_checks,
        "phase_c_admission": phase_c_admission,
        "phase_c_reason": (
            None if phase_c_admission
            else "one or more pre-registered Phase C admission checks failed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def promote_pair(
    manifest_path: Path, development_path: Path, frozen_path: Path,
    population: str, output: Path,
) -> None:
    manifest = _load(manifest_path)
    development = _load(development_path)
    frozen = _load(frozen_path)
    manifest_digest = _sha256(manifest_path)
    if manifest_digest not in {development.get("manifest_sha256"), frozen.get("manifest_sha256")}:
        raise ValueError("target manifest is not the calibration source of either evaluation")
    if development.get("evaluation_role") != "development" or frozen.get("evaluation_role") != "frozen":
        raise ValueError("promotion inputs must be development and frozen evaluations")
    if population not in {"all_bow_candidates", "orb_supported_candidates"}:
        raise ValueError("unsupported population")
    population_manifest = (manifest.get("populations") or {}).get(population)
    if not isinstance(population_manifest, dict) or population_manifest.get("trusted") is not True:
        raise ValueError("target population calibration is not trusted")
    dev_pair = ((development.get("flow_pair_gate") or {}).get(population) or {})
    frozen_pair = ((frozen.get("flow_pair_gate") or {}).get(population) or {})
    if dev_pair.get("promotion_metrics_pass") is not True or frozen_pair.get("promotion_metrics_pass") is not True:
        raise ValueError("Flow pair gate must pass development and frozen evaluations")
    dev_orb = (development.get("orb_gate") or {}).get("promotion_metrics_pass") is True
    frozen_orb = (frozen.get("orb_gate") or {}).get("promotion_metrics_pass") is True
    if population == "orb_supported_candidates" and not (dev_orb and frozen_orb):
        raise ValueError("ORB-supported Flow population cannot be promoted before the ORB gate")
    point_cap = population_manifest.get("point_risk_cap_pair_gate_population")
    point_count = int(population_manifest.get("point_risk_count_pair_gate_population", 0))
    point_pairs = int(population_manifest.get("point_risk_pairs_pair_gate_population", 0))
    if point_cap is None or point_count < 1000 or point_pairs < 20:
        raise ValueError("promoted pair-gate point population is insufficient")
    promoted = json.loads(json.dumps(manifest))
    promoted["promoted_population"] = population
    promoted["pair_development_evaluation_pass"] = True
    promoted["pair_frozen_evaluation_pass"] = True
    promoted["pair_development_evaluation_sha256"] = _sha256(development_path)
    promoted["pair_frozen_evaluation_sha256"] = _sha256(frozen_path)
    promoted["orb_promoted"] = bool(population == "orb_supported_candidates")
    promoted_population = promoted["populations"][population]
    promoted_population["pair_gate_promoted"] = True
    promoted_population["selector_promoted"] = False
    promoted_population["point_risk_cap"] = point_cap
    promoted["promotion_stage"] = "pair_gate_only"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(promoted, indent=2, allow_nan=False), encoding="utf-8")


def promote(
    manifest_path: Path, development_path: Path, frozen_path: Path,
    population: str, output: Path,
) -> None:
    manifest = _load(manifest_path)
    development = _load(development_path)
    frozen = _load(frozen_path)
    manifest_digest = _sha256(manifest_path)
    if manifest_digest not in {development.get("manifest_sha256"), frozen.get("manifest_sha256")}:
        raise ValueError("target manifest is not the calibration source of either evaluation")
    if development.get("evaluation_role") != "development" or frozen.get("evaluation_role") != "frozen":
        raise ValueError("promotion inputs must be development and frozen evaluations")
    if population not in {"all_bow_candidates", "orb_supported_candidates"}:
        raise ValueError("unsupported population")
    population_manifest = (manifest.get("populations") or {}).get(population)
    if (
        not isinstance(population_manifest, dict)
        or population_manifest.get("trusted") is not True
        or population_manifest.get("pair_gate_promoted") is not True
    ):
        raise ValueError("target population pair gate has not been promoted")

    def dataset_pass(payload: dict[str, Any]) -> bool:
        pair = ((payload.get("flow_pair_gate") or {}).get(population) or {})
        selector = ((payload.get("selector_shadow") or {}).get(population) or {})
        return pair.get("promotion_metrics_pass") is True and selector.get("promotion_decision") is True

    development_pass = dataset_pass(development)
    frozen_pass = dataset_pass(frozen)
    if not development_pass or not frozen_pass:
        raise ValueError("development and frozen evaluations must both pass before promotion")
    point_cap = population_manifest.get("point_risk_cap")
    point_count = int(population_manifest.get("point_risk_count_pair_gate_population", 0))
    point_pairs = int(population_manifest.get("point_risk_pairs_pair_gate_population", 0))
    if point_cap is None or point_count < 1000 or point_pairs < 20:
        raise ValueError("promoted pair-gate point population is insufficient")
    promoted = json.loads(json.dumps(manifest))
    promoted["development_evaluation_pass"] = True
    promoted["frozen_evaluation_pass"] = True
    promoted["development_evaluation_sha256"] = _sha256(development_path)
    promoted["frozen_evaluation_sha256"] = _sha256(frozen_path)
    promoted["promoted_population"] = population
    promoted_population = promoted["populations"][population]
    promoted_population["pair_gate_promoted"] = True
    promoted_population["selector_promoted"] = True
    dev_orb = (development.get("orb_gate") or {}).get("promotion_metrics_pass") is True
    frozen_orb = (frozen.get("orb_gate") or {}).get("promotion_metrics_pass") is True
    if population == "orb_supported_candidates" and not (dev_orb and frozen_orb):
        raise ValueError("ORB-supported Flow population cannot be promoted before the ORB gate")
    promoted["orb_promoted"] = bool(dev_orb and frozen_orb and population == "orb_supported_candidates")
    promoted["promotion_policy"] = "pre-registered Phase B.5 development and frozen evaluation v1"
    promoted["promotion_stage"] = "pair_gate_and_selector"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(promoted, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Phase B.5 ORB/Flow gates.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--branch", required=True, type=Path)
    evaluation.add_argument("--manifest", required=True, type=Path)
    evaluation.add_argument("--labels", required=True, type=Path)
    evaluation.add_argument("--legacy-verification", required=True, type=Path)
    evaluation.add_argument("--ref-poses", required=True, type=Path)
    evaluation.add_argument("--cache-index", required=True, type=Path)
    evaluation.add_argument("--record-dir", required=True, type=Path)
    evaluation.add_argument("--role", required=True, choices=("development", "frozen"))
    evaluation.add_argument("--output", required=True, type=Path)
    promotion = subparsers.add_parser("promote")
    promotion.add_argument("--manifest", required=True, type=Path)
    promotion.add_argument("--development-eval", required=True, type=Path)
    promotion.add_argument("--frozen-eval", required=True, type=Path)
    promotion.add_argument("--population", required=True)
    promotion.add_argument("--output", required=True, type=Path)
    pair_promotion = subparsers.add_parser("promote-pair")
    pair_promotion.add_argument("--manifest", required=True, type=Path)
    pair_promotion.add_argument("--development-eval", required=True, type=Path)
    pair_promotion.add_argument("--frozen-eval", required=True, type=Path)
    pair_promotion.add_argument("--population", required=True)
    pair_promotion.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(
            args.branch, args.manifest, args.labels, args.legacy_verification,
            args.ref_poses, args.cache_index, args.record_dir, args.role, args.output,
        )
    elif args.command == "promote-pair":
        promote_pair(
            args.manifest, args.development_eval, args.frozen_eval,
            args.population, args.output,
        )
    else:
        promote(
            args.manifest, args.development_eval, args.frozen_eval,
            args.population, args.output,
        )


if __name__ == "__main__":
    main()
