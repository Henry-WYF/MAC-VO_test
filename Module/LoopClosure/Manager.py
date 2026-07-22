from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import cv2
import torch

from DataLoader import StereoFrame
from Module.Frontend.Frontend import IFrontend
from Module.Frontend.StereoDepth import IStereoDepth
from Module.Map import VisualMap
from Utility.Config import is_config_null, optional_config_value
from Utility.Extensions import ConfigTestable
from Utility.PrettyPrint import Logger

from .Recognizer import (
    CausalRetrievalController,
    CustomBinaryBackend,
    DBoW2ORBBackend,
    FrameIdentity,
    ORBFeatureExtractor,
    PlaceRecognitionBackend,
)
from .Record import GeometryFeatureRecord, LoopFrameRecord
from .PhaseB5 import PhaseB5Analyzer
from .VINSGeometry import (
    GeometryResult,
    cached_orb_geometry,
    fixed_loop_information,
    fixed_point_covariance,
    verify_fixed_geometry,
)
from .Verification import LoopCandidateVerifier, LoopConstraint, has_required_pnp_functions


def _is_int(value: Any, predicate=lambda _: True) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and predicate(value)


def _is_number(value: Any, predicate=lambda _: True) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and predicate(value)


def _validate_section(
    config: SimpleNamespace,
    required: dict[str, Any],
    optional: dict[str, Any] | None = None,
) -> None:
    optional = optional or {}
    allowed = set(required) | set(optional)
    excessive = set(vars(config)) - allowed
    if excessive:
        raise KeyError(f"Excessive Keys: {excessive} from {sorted(allowed)}")
    for key, predicate in required.items():
        if not hasattr(config, key):
            raise KeyError(f"Config does not match specification! (expect to have key {key} but did not found)")
        if not predicate(getattr(config, key)):
            raise ValueError(f"Config does not match specification! ({key}={getattr(config, key)!r})")
    for key, predicate in optional.items():
        if hasattr(config, key) and not predicate(getattr(config, key)):
            raise ValueError(f"Config does not match specification! ({key}={getattr(config, key)!r})")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


class LoopClosureManager(ConfigTestable):
    def __init__(self, config: SimpleNamespace) -> None:
        # Utility.Config represents YAML null as an empty namespace.  Normalize
        # only optional scalar fields; empty component ``args:`` sections retain
        # their historical namespace behavior.
        if hasattr(config, "bow_min_score"):
            config.bow_min_score = optional_config_value(config.bow_min_score)
        phase_b5 = getattr(config, "phase_b5", None)
        if isinstance(phase_b5, SimpleNamespace):
            if hasattr(phase_b5, "trusted_manifest"):
                phase_b5.trusted_manifest = optional_config_value(phase_b5.trusted_manifest)
            calibration = getattr(phase_b5, "calibration", None)
            if isinstance(calibration, SimpleNamespace):
                for key in (
                    "absolute_median_log_risk_cap", "absolute_q95_log_risk_cap",
                ):
                    if hasattr(calibration, key):
                        setattr(calibration, key, optional_config_value(getattr(calibration, key)))
        self.config = config
        self.enabled = bool(config.enabled)
        self.cache_enabled = bool(config.enabled)
        self.retrieval_enabled = bool(config.enabled)
        self.disabled_reason: str | None = None
        self.output_dir: Path | None = None
        self.records: list[dict[str, Any]] = []
        self.last_registered_sensor_idx: int | None = None
        self.extractor: ORBFeatureExtractor | None = None
        self.frontend: IFrontend | None = None
        self.match_cov_default: float | None = None
        self.vins_geometry_enabled = bool(
            config.enabled and getattr(getattr(config, "vins_geometry", None), "enabled", False)
        )
        self.geometry_enabled = bool(
            config.enabled
            and getattr(getattr(config, "geometric_verification", None), "enabled", False)
        )
        if self.enabled:
            try:
                self.extractor = ORBFeatureExtractor(
                    config.orb_nfeatures, config.orb_scale_factor, config.orb_nlevels,
                )
            except Exception as error:
                self._disable_all(f"Failed to initialize ORB feature extractor: {error}")

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        base_spec: dict[str, Any] = {
            "enabled": lambda value: isinstance(value, bool),
            "vocabulary_path": lambda value: isinstance(value, str),
            "keyframe_stride_sensor_frames": lambda value: _is_int(value, lambda item: item > 0),
            "temporal_exclusion_sensor_frames": lambda value: _is_int(value, lambda item: item >= 0),
            "cache_failure_policy": lambda value: value in {"disable_loop", "raise"},
            "orb_nfeatures": lambda value: _is_int(value, lambda item: item > 0),
            "orb_scale_factor": lambda value: _is_number(value, lambda item: item > 1.0),
            "orb_nlevels": lambda value: _is_int(value, lambda item: item > 0),
            "top_k": lambda value: _is_int(value, lambda item: item > 0),
        }
        cls._enforce_config_spec(config, base_spec, allow_excessive_cfg=True)
        optional_base = {
            "recognizer_type": lambda value: value in {"custom_binary", "dbow2_orb"},
            "bow_min_score": lambda value: is_config_null(value) or _is_number(
                value, lambda item: math.isfinite(float(item)) and 0.0 <= float(item) <= 1.0
            ),
        }
        for key, predicate in optional_base.items():
            if hasattr(config, key) and not predicate(getattr(config, key)):
                raise ValueError(f"Config does not match specification! ({key}={getattr(config, key)!r})")
        excessive = set(vars(config)) - (
            set(base_spec) | set(optional_base)
            | {"geometric_verification", "geometry", "pnp", "verification", "loop_information", "phase_b5", "vins_geometry"}
        )
        if excessive:
            raise KeyError(f"Excessive Keys: {excessive} from {list(base_spec)}")
        if hasattr(config, "geometric_verification"):
            _validate_section(
                config.geometric_verification,
                {
                    "enabled": lambda value: isinstance(value, bool),
                    "max_candidates_to_verify": lambda value: _is_int(value, lambda item: item > 0),
                    "min_sensor_gap": lambda value: _is_int(value, lambda item: item >= 0),
                },
                {"compare_flow_cov_gate": lambda value: isinstance(value, bool)},
            )
            _validate_section(
                config.geometry,
                {
                    "min_points": lambda value: _is_int(value, lambda item: item >= 4),
                    "max_points": lambda value: _is_int(value, lambda item: item > 0),
                    "max_depth": lambda value: _is_number(value, lambda item: item > 0.0),
                    "max_depth_cov": lambda value: _is_number(value, lambda item: item >= 0.0),
                    "max_flow_cov": lambda value: _is_number(value, lambda item: item >= 0.0),
                    "border": lambda value: _is_int(value, lambda item: item >= 0),
                    "grid_rows": lambda value: _is_int(value, lambda item: item > 0),
                    "grid_cols": lambda value: _is_int(value, lambda item: item > 0),
                    "max_points_per_cell": lambda value: _is_int(value, lambda item: item > 0),
                },
                {
                    "flow_cov_gate_enabled": lambda value: isinstance(value, bool),
                    "flow_cov_adaptive_target_points": lambda value: value is None
                    or _is_int(value, lambda item: item > 0),
                },
            )
            _validate_section(config.pnp, {
                "reproj_error_px": lambda value: _is_number(value, lambda item: item > 0.0),
                "confidence": lambda value: _is_number(value, lambda item: 0.0 < item < 1.0),
                "iterations": lambda value: _is_int(value, lambda item: item > 0),
                "min_inliers": lambda value: _is_int(value, lambda item: item >= 4),
                "min_inlier_ratio": lambda value: _is_number(value, lambda item: 0.0 <= item <= 1.0),
                "refine": lambda value: isinstance(value, bool),
            })
            _validate_section(config.verification, {
                "max_mean_reproj_error_px": lambda value: _is_number(value, lambda item: item > 0.0),
                "max_rotation_diff_deg": lambda value: _is_number(value, lambda item: item >= 0.0),
                "max_translation_diff_m": lambda value: _is_number(value, lambda item: item >= 0.0),
            })
            _validate_section(config.loop_information, {
                "trans_weight": lambda value: _is_number(value, lambda item: item > 0.0),
                "rot_weight": lambda value: _is_number(value, lambda item: item > 0.0),
            })
        if hasattr(config, "phase_b5"):
            _validate_section(
                config.phase_b5,
                {
                    "enabled": lambda value: isinstance(value, bool),
                    "mode": lambda value: value in {"disabled", "observe", "apply"},
                    "calibration": lambda value: isinstance(value, SimpleNamespace),
                    "orb": lambda value: isinstance(value, SimpleNamespace),
                    "flow": lambda value: isinstance(value, SimpleNamespace),
                },
                {"trusted_manifest": lambda value: is_config_null(value) or isinstance(value, str)},
            )
            _validate_section(config.phase_b5.calibration, {
                "prefix_fraction": lambda value: _is_number(value, lambda item: 0.0 < item < 1.0),
                "min_queries": lambda value: _is_int(value, lambda item: item > 0),
                "min_all_bow_pairs": lambda value: _is_int(value, lambda item: item > 0),
                "min_orb_pairs": lambda value: _is_int(value, lambda item: item > 0),
                "absolute_median_log_risk_cap": lambda value: is_config_null(value) or _is_number(value, math.isfinite),
                "absolute_q95_log_risk_cap": lambda value: is_config_null(value) or _is_number(value, math.isfinite),
            })
            _validate_section(config.phase_b5.orb, {
                "ratio": lambda value: _is_number(value, lambda item: 0.0 < item < 1.0),
                "max_depth": lambda value: _is_number(value, lambda item: item > 0.0),
            })
            _validate_section(config.phase_b5.flow, {
                "min_valid_points": lambda value: _is_int(value, lambda item: item >= 4),
                "min_valid_ratio": lambda value: _is_number(value, lambda item: 0.0 <= item <= 1.0),
                "min_grid_cells": lambda value: _is_int(value, lambda item: item > 0),
                "nms_kernel_size": lambda value: _is_int(value, lambda item: item > 0 and item % 2 == 1),
                "border": lambda value: _is_int(value, lambda item: item >= 0),
                "min_points": lambda value: _is_int(value, lambda item: item >= 4),
                "max_points": lambda value: _is_int(value, lambda item: item > 0),
                "max_depth": lambda value: _is_number(value, lambda item: item > 0.0),
                "grid_rows": lambda value: _is_int(value, lambda item: item > 0),
                "grid_cols": lambda value: _is_int(value, lambda item: item > 0),
                "max_points_per_cell": lambda value: _is_int(value, lambda item: item > 0),
            })
        if hasattr(config, "vins_geometry"):
            _validate_section(config.vins_geometry, {
                "enabled": lambda value: isinstance(value, bool),
                "max_candidates": lambda value: _is_int(
                    value, lambda item: 0 < item <= int(config.top_k)
                ),
                "hamming_threshold": lambda value: _is_int(value, lambda item: 0 < item <= 256),
                "iterations": lambda value: _is_int(value, lambda item: item > 0),
                "reproj_error_px": lambda value: _is_number(value, lambda item: item > 0.0),
                "confidence": lambda value: _is_number(value, lambda item: 0.0 < item < 1.0),
                "min_inliers": lambda value: _is_int(value, lambda item: item >= 4),
                "max_translation_m": lambda value: _is_number(value, lambda item: item > 0.0),
                "max_rotation_deg": lambda value: _is_number(value, lambda item: 0.0 < item <= 180.0),
            }, {
                "feature_source": lambda value: value in {"fixed_covariance", "orb_detected"},
                "descriptor_match_mode": lambda value: value in {"vins_legacy", "orbslam"},
            })
            feature_source = getattr(config.vins_geometry, "feature_source", "fixed_covariance")
            descriptor_match_mode = getattr(
                config.vins_geometry, "descriptor_match_mode", "vins_legacy"
            )
            if descriptor_match_mode == "orbslam" and feature_source != "orb_detected":
                raise ValueError("descriptor_match_mode=orbslam requires feature_source=orb_detected")

    def set_frontend(self, frontend: IFrontend) -> None:
        self.frontend = frontend

    @staticmethod
    def primary_flow_cov_gate_enabled(config: SimpleNamespace) -> bool:
        return bool(getattr(config.geometry, "flow_cov_gate_enabled", True))

    @staticmethod
    def flow_cov_comparison_enabled(config: SimpleNamespace) -> bool:
        return bool(getattr(config.geometric_verification, "compare_flow_cov_gate", False))

    def set_output_dir(self, output_dir: Path) -> None:
        if not self.cache_enabled:
            return
        try:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            self.output_dir = None
            self._handle_cache_failure(error)

    def _disable_all(self, reason: str) -> None:
        self.enabled = False
        self.cache_enabled = False
        self.retrieval_enabled = False
        self.geometry_enabled = False
        self.disabled_reason = reason
        Logger.write("error", f"Loop closure disabled: {reason}")

    def _disable_retrieval(self, reason: str) -> None:
        self.retrieval_enabled = False
        self.geometry_enabled = False
        self.disabled_reason = reason
        Logger.write("error", f"Loop retrieval disabled; frame caching remains enabled: {reason}")

    def disable_geometry(self, reason: str) -> None:
        self.geometry_enabled = False
        self.disabled_reason = reason
        Logger.write("error", f"Loop geometric verification disabled: {reason}")

    def set_match_cov_default(self, value: float) -> None:
        self.match_cov_default = float(value)

    @property
    def requires_geometry_sidecar(self) -> bool:
        return (
            self.vins_geometry_enabled
            and getattr(self.config.vins_geometry, "feature_source", "fixed_covariance")
            == "fixed_covariance"
        )

    def geometry_sidecar_path(self, loop_frame_idx: int, root: Path | None = None) -> Path:
        base = self.output_dir if root is None else Path(root)
        if base is None:
            raise RuntimeError("loop cache output directory was not configured")
        return base / "geometry_features_v1" / f"loop_{int(loop_frame_idx):06d}.pt"

    def cache_geometry_features(
        self,
        frame: StereoFrame,
        visual_map_idx: int,
        *,
        original_index: torch.Tensor,
        pixel_uv: torch.Tensor,
        point_camera: torch.Tensor,
        depth: torch.Tensor,
        depth_variance: torch.Tensor,
        disparity: torch.Tensor,
        disparity_variance: torch.Tensor,
    ) -> bool:
        if not self.cache_enabled or not self.requires_geometry_sidecar:
            return False
        if self.output_dir is None or self.extractor is None or self.match_cov_default is None:
            return False
        metadata = next((
            item for item in self.records
            if int(item["sensor_frame_idx"]) == int(frame.frame_idx)
            and int(item["visual_map_idx"]) == int(visual_map_idx)
        ), None)
        if metadata is None:
            return False
        target = self.geometry_sidecar_path(int(metadata["loop_frame_idx"]))
        if target.exists():
            return False
        try:
            returned, descriptors = self.extractor.compute_at(frame.stereo.imageL, pixel_uv)
            selected = returned.long()
            disp = disparity.reshape(-1)
            disp_var = disparity_variance.reshape(-1)
            def take(value: torch.Tensor) -> torch.Tensor:
                return value[selected.to(value.device)]
            selected_disp, selected_disp_var = take(disp), take(disp_var)
            point_covariance = fixed_point_covariance(
                pixel_uv, depth, depth_variance, frame.stereo.K, self.match_cov_default,
            )
            record = GeometryFeatureRecord(
                sensor_frame_idx=int(frame.frame_idx), visual_map_idx=int(visual_map_idx),
                loop_frame_idx=int(metadata["loop_frame_idx"]),
                orb_config_sha256=self.extractor.config_sha256(),
                original_index=take(original_index.reshape(-1)), pixel_uv=take(pixel_uv),
                point_camera=take(point_camera), point_covariance_camera=take(point_covariance),
                disparity=selected_disp, disparity_variance=selected_disp_var,
                disparity_valid=torch.isfinite(selected_disp) & torch.isfinite(selected_disp_var)
                & (selected_disp > 0.0) & (selected_disp_var > 0.0),
                descriptor=descriptors,
            )
            return record.save_if_absent(target)
        except Exception as error:
            self._disable_all(f"geometry sidecar failure: {error}")
            return False

    def _handle_cache_failure(self, error: Exception) -> None:
        if self.config.cache_failure_policy == "raise":
            raise error
        self._disable_all(f"cache failure: {error}")

    def _write_index(self) -> None:
        assert self.output_dir is not None
        target = self.output_dir / "index.json"
        temporary = target.with_suffix(".json.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as file:
                json.dump({"schema_version": 1, "records": self.records}, file, indent=2)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def should_register(self, sensor_frame_idx: int) -> bool:
        if not self.cache_enabled:
            return False
        if self.last_registered_sensor_idx is None:
            return True
        return int(sensor_frame_idx) - self.last_registered_sensor_idx >= int(self.config.keyframe_stride_sensor_frames)

    def register_loop_frame(self, frame: StereoFrame, depth: IStereoDepth.Output, visual_map_idx: int, registered_pose: torch.Tensor) -> bool:
        if not self.should_register(frame.frame_idx):
            return False
        if self.output_dir is None:
            self._handle_cache_failure(RuntimeError("loop cache output directory was not configured"))
            return False
        assert self.extractor is not None
        try:
            retrieval_started = time.perf_counter()
            keypoints, descriptors = self.extractor.extract(frame.stereo.imageL)
            retrieval_build_ms = (time.perf_counter() - retrieval_started) * 1000.0
            loop_idx = len(self.records)
            record = LoopFrameRecord(
                sensor_frame_idx=int(frame.frame_idx), visual_map_idx=int(visual_map_idx), loop_frame_idx=loop_idx,
                frame_ns=int(frame.stereo.frame_ns), height=int(frame.stereo.height), width=int(frame.stereo.width),
                image_left=frame.stereo.imageL, image_right=frame.stereo.imageR, intrinsic=frame.stereo.K,
                baseline=frame.stereo.baseline, body_to_sensor=frame.stereo.T_BS.tensor(), depth=depth.depth,
                depth_covariance=depth.cov, registered_pose=registered_pose, orb_keypoints=keypoints,
                orb_descriptors=descriptors, bow_vector=None,
            )
            relative_path = Path("frames", f"loop_{loop_idx:06d}_sensor_{frame.frame_idx:09d}.pt")
            record.save(self.output_dir / relative_path)
            self.records.append({
                "sensor_frame_idx": int(frame.frame_idx), "visual_map_idx": int(visual_map_idx),
                "loop_frame_idx": loop_idx, "file": relative_path.as_posix(),
                "orb_features": int(len(descriptors)), "has_bow": False,
                "feature_extraction_time_ms": retrieval_build_ms,
                "retrieval_build_time_ms": retrieval_build_ms,
            })
            self._write_index()
            self.last_registered_sensor_idx = int(frame.frame_idx)
            return True
        except Exception as error:
            self._handle_cache_failure(error)
            return False

    def _make_backend(self) -> PlaceRecognitionBackend:
        recognizer_type = str(getattr(self.config, "recognizer_type", "custom_binary"))
        if recognizer_type == "custom_binary":
            return CustomBinaryBackend(
                self.config.vocabulary_path,
                self.config.orb_nfeatures,
                self.config.orb_scale_factor,
                self.config.orb_nlevels,
            )
        if recognizer_type == "dbow2_orb":
            return DBoW2ORBBackend(self.config.vocabulary_path)
        raise ValueError(f"Unsupported loop recognizer type: {recognizer_type}")

    def detect_all(
        self,
        record_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        if not self.retrieval_enabled or self.output_dir is None:
            return []
        record_root = self.output_dir if record_dir is None else Path(record_dir)
        result_root = self.output_dir if output_dir is None else Path(output_dir)
        try:
            result_root.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            self._disable_retrieval(f"failed to prepare Phase A output directory: {error}")
            return []
        try:
            backend = self._make_backend()
        except Exception as error:
            self._disable_retrieval(f"failed to initialize place-recognition backend: {error}")
            return []

        controller = CausalRetrievalController(
            backend,
            self.config.temporal_exclusion_sensor_frames,
            self.config.top_k,
            getattr(self.config, "bow_min_score", None),
        )
        queries: list[dict[str, Any]] = []
        try:
            for metadata in sorted(self.records, key=lambda item: item["sensor_frame_idx"]):
                record = LoopFrameRecord.load(record_root / metadata["file"])
                result = controller.query_then_enqueue(
                    FrameIdentity(record.sensor_frame_idx, record.visual_map_idx, record.loop_frame_idx),
                    record.orb_descriptors,
                )
                assert all(candidate.sensor_frame_idx < record.sensor_frame_idx for candidate in result.candidates)
                queries.append({
                    "sensor_frame_idx": record.sensor_frame_idx,
                    "visual_map_idx": record.visual_map_idx,
                    "loop_frame_idx": record.loop_frame_idx,
                    "orb_features": int(len(record.orb_descriptors)),
                    "feature_extraction_time_ms": float(metadata.get(
                        "feature_extraction_time_ms", metadata.get("retrieval_build_time_ms", 0.0)
                    )),
                    "empty_descriptors": result.empty_descriptors,
                    "eligible_database_entries": result.eligible_database_entries,
                    "retrieved_before_score_filter": result.retrieved_before_score_filter,
                    "after_score_filter": result.after_score_filter,
                    "returned_candidates": len(result.candidates),
                    "query_time_ms": result.query_time_ms,
                    "candidates": [candidate.__dict__ for candidate in result.candidates],
                })
        except Exception as error:
            self._disable_retrieval(f"Phase A query failed: {error}")
            return []

        try:
            backend_metadata = backend.metadata()
        except Exception as error:
            self._disable_retrieval(f"failed to collect Phase A backend metadata: {error}")
            return []
        payload = {
            "schema_version": 2,
            "temporal_exclusion_unit": "sensor_frame_idx",
            "temporal_exclusion": int(self.config.temporal_exclusion_sensor_frames),
            "top_k": int(self.config.top_k),
            "bow_min_score": getattr(self.config, "bow_min_score", None),
            "recognizer": backend_metadata,
            "summary": {
                "query_count": len(queries),
                "queries_with_candidates": sum(bool(query["candidates"]) for query in queries),
                "returned_candidate_count": sum(len(query["candidates"]) for query in queries),
            },
            "queries": queries,
        }
        target = result_root / "queries.json"
        temporary = target.with_suffix(".json.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as file:
                json.dump(_json_safe(payload), file, indent=2, allow_nan=False)
            os.replace(temporary, target)
        except Exception as error:
            self._disable_retrieval(f"failed to write Phase A output: {error}")
            return []
        finally:
            if temporary.exists():
                temporary.unlink()
        return queries


    def _record_by_loop_idx(self) -> dict[int, dict[str, Any]]:
        return {int(record["loop_frame_idx"]): record for record in self.records}

    @staticmethod
    def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        funnel_keys = (
            "candidate_samples",
            "after_inbound_and_finite",
            "after_flow_cov",
            "after_match_mask",
            "valid_depth_after_flow_points",
            "geometry_points",
            "pnp_input_points",
            "pnp_inliers",
        )
        point_totals = {key: 0 for key in funnel_keys}
        candidate_stage_counts = {key: 0 for key in funnel_keys}
        covariance_keys = (
            "uu_finite",
            "vv_finite",
            "uv_finite",
            "gate_channels_joint_finite",
            "uu_finite_and_below_threshold",
            "vv_finite_and_below_threshold",
            "gate_channels_joint_below_threshold",
            "frontend_score_finite",
            "frontend_score_nonfinite",
            "frontend_score_negative",
            "frontend_score_below_reference",
        )
        covariance_totals = {key: 0 for key in covariance_keys}
        reject_codes: Counter[str] = Counter()
        for row in rows:
            if row.get("reject_code") is not None:
                reject_codes[str(row["reject_code"])] += 1
            diagnostics = row.get("diagnostics") or {}
            values = {
                "candidate_samples": diagnostics.get("candidate_samples"),
                "after_inbound_and_finite": diagnostics.get("after_inbound_and_finite"),
                "after_flow_cov": diagnostics.get("after_flow_cov"),
                "after_match_mask": diagnostics.get("after_match_mask"),
                "valid_depth_after_flow_points": diagnostics.get("valid_depth_after_flow_points"),
                "geometry_points": diagnostics.get("geometry_points", row.get("num_geometry_points")),
                "pnp_input_points": diagnostics.get("pnp_input_points"),
                "pnp_inliers": row.get("num_pnp_inliers"),
            }
            for key, value in values.items():
                if value is None:
                    continue
                count = int(value)
                point_totals[key] += count
                if count > 0:
                    candidate_stage_counts[key] += 1
            covariance = diagnostics.get("covariance") or {}
            for key in covariance_keys:
                value = covariance.get(key)
                if value is not None:
                    covariance_totals[key] += int(value)
        return {
            "candidate_count": len(rows),
            "accepted_candidates": sum(row.get("status") == "accepted" for row in rows),
            "comparison_applicable_candidates": sum(
                row.get("comparison_applicable") is True for row in rows
            ),
            "covariance_available_candidates": sum(
                (row.get("diagnostics") or {}).get("covariance_available") is True
                for row in rows
            ),
            "reject_code_counts": dict(sorted(reject_codes.items())),
            "funnel_point_totals": point_totals,
            "funnel_candidate_counts": candidate_stage_counts,
            "covariance_diagnostic_mother_set": "after_inbound_and_finite",
            "covariance_diagnostic_point_totals": covariance_totals,
            "pnp_attempted_candidates": sum(row.get("pnp_attempted") is True for row in rows),
            "pnp_ransac_succeeded_candidates": sum(row.get("pnp_ransac_succeeded") is True for row in rows),
            "pnp_inlier_gate_passed_candidates": sum(row.get("pnp_inlier_gate_passed") is True for row in rows),
            "reprojection_gate_passed_candidates": sum(row.get("reprojection_gate_passed") is True for row in rows),
            "pose_consistency_gate_passed_candidates": sum(row.get("pose_consistency_gate_passed") is True for row in rows),
        }

    @staticmethod
    def _comparison_summary(
        gate_enabled_rows: list[dict[str, Any]], gate_disabled_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        enabled = {str(row["comparison_pair_id"]): row for row in gate_enabled_rows}
        disabled = {str(row["comparison_pair_id"]): row for row in gate_disabled_rows}
        if len(enabled) != len(gate_enabled_rows) or len(disabled) != len(gate_disabled_rows):
            raise RuntimeError("duplicate comparison_pair_id in Phase B verification")
        if set(enabled) != set(disabled):
            raise RuntimeError("gate-on and gate-off pair_id sets differ")

        applicable = [
            pair_id
            for pair_id in enabled
            if enabled[pair_id].get("comparison_applicable") is True
            and disabled[pair_id].get("comparison_applicable") is True
        ]

        def diagnostic(row: dict[str, Any], key: str) -> Any:
            return (row.get("diagnostics") or {}).get(key)

        after_flow_pairs = [
            pair_id
            for pair_id in applicable
            if diagnostic(enabled[pair_id], "after_flow_cov") is not None
            and diagnostic(disabled[pair_id], "after_flow_cov") is not None
        ]
        pnp_attempted_pairs = [
            pair_id
            for pair_id in applicable
            if enabled[pair_id].get("pnp_attempted") is not None
            and disabled[pair_id].get("pnp_attempted") is not None
        ]
        pnp_ransac_pairs = [
            pair_id
            for pair_id in applicable
            if enabled[pair_id].get("pnp_ransac_succeeded") is not None
            and disabled[pair_id].get("pnp_ransac_succeeded") is not None
        ]

        return {
            "pair_count": len(enabled),
            "pair_ids_unique_and_aligned": True,
            "changed_metrics_mother_set": "comparison_applicable_candidates",
            "comparison_applicable_candidates": len(applicable),
            "after_flow_cov_compared_candidates": len(after_flow_pairs),
            "after_flow_cov_changed_candidates": sum(
                diagnostic(enabled[pair_id], "after_flow_cov")
                != diagnostic(disabled[pair_id], "after_flow_cov")
                for pair_id in after_flow_pairs
            ),
            "pnp_attempted_compared_candidates": len(pnp_attempted_pairs),
            "pnp_attempted_changed_candidates": sum(
                enabled[pair_id].get("pnp_attempted") != disabled[pair_id].get("pnp_attempted")
                for pair_id in pnp_attempted_pairs
            ),
            "pnp_ransac_result_compared_candidates": len(pnp_ransac_pairs),
            "pnp_ransac_result_changed_candidates": sum(
                enabled[pair_id].get("pnp_ransac_succeeded")
                != disabled[pair_id].get("pnp_ransac_succeeded")
                for pair_id in pnp_ransac_pairs
            ),
            "accepted_result_changed_candidates": sum(
                (enabled[pair_id].get("status") == "accepted")
                != (disabled[pair_id].get("status") == "accepted")
                for pair_id in applicable
            ),
        }

    def _write_json_bundle(
        self,
        payloads: list[tuple[str, dict[str, Any]]],
        comparison_run_id: str,
    ) -> None:
        assert self.output_dir is not None
        temporary_paths: list[Path] = []
        try:
            for filename, payload in payloads:
                target = self.output_dir / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f"{target.name}.{comparison_run_id}.tmp")
                with open(temporary, "w", encoding="utf-8") as file:
                    json.dump(_json_safe(payload), file, indent=2, allow_nan=False)
                    file.flush()
                    os.fsync(file.fileno())
                temporary_paths.append(temporary)
            for (filename, _), temporary in zip(payloads, temporary_paths, strict=True):
                os.replace(temporary, self.output_dir / filename)
        finally:
            for temporary in temporary_paths:
                if temporary.exists():
                    temporary.unlink()

    def verify_candidates(
        self,
        global_map: VisualMap,
        queries: list[dict[str, Any]],
        *,
        record_dir: Path | None = None,
        progress_interval: int | None = None,
    ) -> list[LoopConstraint]:
        if not self.enabled or not self.geometry_enabled:
            return []
        if not queries or not any(query.get("candidates") for query in queries):
            Logger.write("info", "Skip loop geometric verification because Phase A returned no candidates.")
            return []
        if self.output_dir is None:
            Logger.write("warn", "Skip loop geometric verification because output directory is unavailable.")
            return []
        if self.vins_geometry_enabled:
            return self._verify_vins_candidates(global_map, queries, record_dir=record_dir)
        if self.frontend is None:
            Logger.write("error", "Skip loop geometric verification because Frontend was not injected.")
            return []
        if not has_required_pnp_functions():
            Logger.write("error", "Skip loop geometric verification because OpenCV PnP/calib3d functions are unavailable.")
            return []

        record_metadata = self._record_by_loop_idx()
        record_root = self.output_dir if record_dir is None else Path(record_dir)
        if not record_root.is_dir():
            raise RuntimeError(f"loop-frame record directory does not exist: {record_root}")
        phase_b5 = None
        phase_b5_config = getattr(self.config, "phase_b5", None)
        if bool(getattr(phase_b5_config, "enabled", False)) and getattr(phase_b5_config, "mode", "disabled") != "disabled":
            phase_b5 = PhaseB5Analyzer(
                self.config.phase_b5, queries, self.records, record_root
            )
        verifier = LoopCandidateVerifier(self.config, self.frontend, phase_b5)
        primary_gate_enabled = self.primary_flow_cov_gate_enabled(self.config)
        comparison_enabled = self.flow_cov_comparison_enabled(self.config)
        gate_modes = [True, False] if comparison_enabled else [primary_gate_enabled]
        verification_rows: dict[bool, list[dict[str, Any]]] = {gate: [] for gate in gate_modes}
        constraints: dict[bool, list[LoopConstraint]] = {gate: [] for gate in gate_modes}
        max_candidates = int(self.config.geometric_verification.max_candidates_to_verify)
        total_candidates = sum(
            min(len(query.get("candidates", [])), max_candidates) for query in queries
        )
        processed_candidates = 0
        poses = global_map.frames.data["pose"].tensor
        pose_snapshot = poses.detach().clone()
        snapshot_guard = pose_snapshot.clone()

        for query in queries:
            current_meta = record_metadata.get(int(query["loop_frame_idx"]))
            if current_meta is None:
                raise RuntimeError(f"missing loop-frame metadata for query {query['loop_frame_idx']}")
            current = LoopFrameRecord.load(record_root / current_meta["file"])
            for candidate in query.get("candidates", [])[:max_candidates]:
                historical_meta = record_metadata.get(int(candidate["loop_frame_idx"]))
                if historical_meta is None:
                    raise RuntimeError(
                        f"missing loop-frame metadata for candidate {candidate['loop_frame_idx']}"
                    )
                historical = LoopFrameRecord.load(record_root / historical_meta["file"])
                branch_results = verifier.verify_branches(
                    pose_snapshot, query, candidate, current, historical, gate_modes
                )
                for gate in gate_modes:
                    verification, constraint = branch_results[gate]
                    verification_rows[gate].append(verification.to_dict())
                    if constraint is not None:
                        constraints[gate].append(constraint)
                processed_candidates += 1
                if (
                    progress_interval is not None
                    and progress_interval > 0
                    and (
                        processed_candidates % progress_interval == 0
                        or processed_candidates == total_candidates
                    )
                ):
                    Logger.write(
                        "info",
                        f"Loop Phase B verification progress: {processed_candidates} / {total_candidates}",
                    )

        if not torch.equal(poses, pose_snapshot):
            raise RuntimeError("Phase B modified VisualMap poses")
        if not torch.equal(pose_snapshot, snapshot_guard):
            raise RuntimeError("Phase B modified its pose snapshot")

        comparison_run_id = uuid.uuid4().hex
        covariance_summary = verifier.aggregate_covariance_statistics()
        comparison_summary = None
        if comparison_enabled:
            comparison_summary = self._comparison_summary(
                verification_rows[True], verification_rows[False]
            )

        def verification_payload(gate: bool) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "schema_version": 3,
                "comparison_run_id": comparison_run_id,
                "primary_gate_enabled": primary_gate_enabled,
                "comparison_enabled": comparison_enabled,
                "branch_gate_enabled": gate,
                "frontend_type": type(self.frontend).__name__,
                "max_candidates_to_verify": max_candidates,
                "frontend_inference_calls": verifier.frontend_inference_calls,
                "candidates_reaching_frontend": verifier.candidates_reaching_frontend,
                "covariance_statistics": covariance_summary,
                "summary": self._summarize_rows(verification_rows[gate]),
                "verifications": verification_rows[gate],
                "constraints_trust_policy": "trust only after completion marker and pair validation",
            }
            if comparison_summary is not None:
                payload["comparison_summary"] = comparison_summary
            return payload

        def constraint_payload(gate: bool) -> dict[str, Any]:
            return {
                "schema_version": 1,
                "pose_direction": "relative_pose = T_candidate_current = inverse(T_current_candidate)",
                "constraints": [constraint.to_dict() for constraint in constraints[gate]],
            }

        payloads: list[tuple[str, dict[str, Any]]] = []
        if comparison_enabled:
            payloads.extend([
                ("loop_constraints_gate_enabled.json", constraint_payload(True)),
                ("loop_constraints_gate_disabled.json", constraint_payload(False)),
                ("loop_verification_gate_enabled.json", verification_payload(True)),
                ("loop_verification_gate_disabled.json", verification_payload(False)),
            ])
        payloads.append(("loop_constraints.json", constraint_payload(primary_gate_enabled)))
        phase_b5_primary_constraints: list[LoopConstraint] | None = None
        if phase_b5 is not None:
            # A replay with a trusted/intermediate manifest must publish the exact
            # effective thresholds and promotion stage that produced its branches.
            calibration_manifest = (
                phase_b5.calibration_manifest()
                if phase_b5.manifest is None
                else json.loads(json.dumps(phase_b5.manifest))
            )
            calibration_manifest["comparison_run_id"] = comparison_run_id
            payloads.append(("phase_b5_calibration_manifest.json", calibration_manifest))
            for filename, payload in phase_b5.branch_payloads().items():
                payload["comparison_run_id"] = comparison_run_id
                payloads.append((filename, payload))
            if phase_b5.mode == "apply":
                phase_b5_primary_constraints = []
                cascade_rows: list[dict[str, Any]] = []
                orb_promoted = bool(phase_b5.manifest and phase_b5.manifest.get("orb_promoted") is True)
                promoted_population = str(
                    (phase_b5.manifest or {}).get("promoted_population", "all_bow_candidates")
                )
                for row in phase_b5.rows:
                    pair_id = str(row["pair_id"])
                    sensor_pair = (
                        int(row["current_sensor_frame_idx"]),
                        int(row["candidate_sensor_frame_idx"]),
                    )
                    forced_control = sensor_pair in {
                        (1200, 1100), (1250, 470), (1250, 480),
                    }
                    orb_pass = bool((row.get("orb") or {}).get("orb_gate_pass"))
                    preferred_population = promoted_population
                    pair_pass = bool(
                        ((row.get("flow") or {}).get("pair_gate_by_population") or {}).get(preferred_population)
                    )
                    if preferred_population == "orb_supported_candidates" and (not orb_promoted or not orb_pass):
                        pair_pass = False
                    selector = (row.get("selector_shadow_by_population") or {}).get(preferred_population) or {}
                    result = verifier.phase_b5_results.get(f"{pair_id}|{preferred_population}")
                    accepted = bool(
                        not forced_control
                        and pair_pass
                        and selector.get("status") == "accepted"
                        and result is not None
                        and result[1] is not None
                    )
                    cascade_rows.append({
                        "pair_id": pair_id,
                        "population": preferred_population,
                        "forced_control_shadow": forced_control,
                        "pair_gate_pass": pair_pass,
                        "selector_accepted": selector.get("status") == "accepted",
                        "accepted": accepted,
                    })
                    if accepted:
                        assert result is not None and result[1] is not None
                        phase_b5_primary_constraints.append(result[1])
                payloads.extend([
                    ("cascade_apply/verification.json", {
                        "schema_version": 1,
                        "branch_id": "cascade_apply",
                        "comparison_run_id": comparison_run_id,
                        "rows": cascade_rows,
                    }),
                    ("cascade_apply/constraints.json", {
                        "schema_version": 1,
                        "pose_direction": "relative_pose = T_candidate_current = inverse(T_current_candidate)",
                        "constraints": [item.to_dict() for item in phase_b5_primary_constraints],
                    }),
                ])
                # In apply mode the standard constraint output and return value both
                # belong to the promoted Phase B.5 branch. Named gate files remain
                # legacy diagnostics.
                for index, (filename, _) in enumerate(payloads):
                    if filename == "loop_constraints.json":
                        payloads[index] = ("loop_constraints.json", {
                            "schema_version": 1,
                            "pose_direction": "relative_pose = T_candidate_current = inverse(T_current_candidate)",
                            "constraints": [item.to_dict() for item in phase_b5_primary_constraints],
                        })
                        break
        # The main verification file is replaced last and acts as the run-completion marker.
        main_verification = verification_payload(primary_gate_enabled)
        if phase_b5 is not None:
            main_verification["phase_b5"] = {
                "enabled": True,
                "mode": phase_b5.mode,
                "schema_version": 1,
                "pose_invariant": True,
                "calibration_manifest": "phase_b5_calibration_manifest.json",
                "branch_ids": [
                    "orb_observe", "flow_all_bow_observe",
                    "flow_orb_supported_observe", "forced_control_shadow",
                ] + (["cascade_apply"] if phase_b5.mode == "apply" else []),
            }
            main_verification["primary_pipeline"] = (
                "phase_b5_apply" if phase_b5.mode == "apply" else "legacy_phase_b"
            )
            main_verification["phase_b5"]["primary_constraint_count"] = (
                None if phase_b5_primary_constraints is None
                else len(phase_b5_primary_constraints)
            )
        payloads.append(("loop_verification.json", main_verification))
        self._write_json_bundle(payloads, comparison_run_id)

        primary_rows = verification_rows[primary_gate_enabled]
        primary_constraints = constraints[primary_gate_enabled]
        returned_constraints = (
            primary_constraints if phase_b5_primary_constraints is None
            else phase_b5_primary_constraints
        )
        pipeline_name = "legacy Phase B" if phase_b5_primary_constraints is None else "Phase B.5 apply"
        Logger.write(
            "info",
            f"{pipeline_name} accepted {len(returned_constraints)} / {len(primary_rows)} candidates.",
        )
        return returned_constraints

    def _load_geometry_sidecar(
        self, metadata: dict[str, Any], record_root: Path,
    ) -> GeometryFeatureRecord | None:
        path = self.geometry_sidecar_path(int(metadata["loop_frame_idx"]), record_root)
        if not path.is_file() or self.extractor is None:
            return None
        try:
            record = GeometryFeatureRecord.load(path)
        except Exception:
            return None
        valid = (
            int(record.sensor_frame_idx) == int(metadata["sensor_frame_idx"])
            and int(record.visual_map_idx) == int(metadata["visual_map_idx"])
            and int(record.loop_frame_idx) == int(metadata["loop_frame_idx"])
            and record.orb_config_sha256 == self.extractor.config_sha256()
        )
        return record if valid else None

    def _verify_vins_candidates(
        self,
        global_map: VisualMap,
        queries: list[dict[str, Any]],
        *,
        record_dir: Path | None = None,
    ) -> list[LoopConstraint]:
        if self.output_dir is None or self.match_cov_default is None:
            Logger.write("warn", "Skip VINS-style loop verification because cache or pixel variance is unavailable.")
            return []
        if not all(hasattr(cv2, name) for name in ("solvePnPRansac", "Rodrigues", "SOLVEPNP_ITERATIVE")):
            Logger.write("warn", "Skip VINS-style loop verification because OpenCV PnP is unavailable.")
            return []
        record_root = self.output_dir if record_dir is None else Path(record_dir)
        metadata_by_loop = self._record_by_loop_idx()
        poses = global_map.frames.data["pose"].tensor
        pose_snapshot = poses.detach().clone()
        fixed_information = fixed_loop_information(self.config.loop_information)
        rows: list[dict[str, Any]] = []
        selected_constraints: list[LoopConstraint] = []
        pgo_fixed_constraints: list[dict[str, Any]] = []
        pgo_covariance_constraints: list[dict[str, Any]] = []
        config = self.config.vins_geometry
        feature_source = str(getattr(config, "feature_source", "fixed_covariance"))
        descriptor_match_mode = str(getattr(config, "descriptor_match_mode", "vins_legacy"))
        geometry_cache: dict[int, tuple[GeometryFeatureRecord | None, str | None, dict[str, Any]]] = {}

        def load_geometry(
            metadata: dict[str, Any], frame: LoopFrameRecord,
        ) -> tuple[GeometryFeatureRecord | None, str | None, dict[str, Any]]:
            loop_idx = int(metadata["loop_frame_idx"])
            if loop_idx in geometry_cache:
                return geometry_cache[loop_idx]
            if feature_source == "orb_detected":
                loaded = cached_orb_geometry(
                    frame, self.match_cov_default,
                    require_orientation=descriptor_match_mode == "orbslam",
                )
            else:
                record = self._load_geometry_sidecar(metadata, record_root)
                loaded = (
                    record,
                    None if record is not None else "geometry_sidecar_unavailable",
                    {},
                )
            geometry_cache[loop_idx] = loaded
            return loaded

        for query in queries:
            current_meta = metadata_by_loop.get(int(query["loop_frame_idx"]))
            if current_meta is None:
                continue
            current_frame = LoopFrameRecord.load(record_root / current_meta["file"])
            current_geometry, current_error, current_diagnostics = load_geometry(
                current_meta, current_frame,
            )
            candidates = sorted(
                query.get("candidates", []),
                key=lambda item: (-float(item.get("score", 0.0)), int(item["sensor_frame_idx"])),
            )[:int(config.max_candidates)]
            accepted: list[tuple[int, GeometryResult]] = []
            for candidate in candidates:
                candidate_meta = metadata_by_loop.get(int(candidate["loop_frame_idx"]))
                pair_id = f"{int(query['loop_frame_idx'])}:{int(candidate['loop_frame_idx'])}"
                if current_geometry is None or candidate_meta is None:
                    rows.append({
                        "pair_id": pair_id, "status": "rejected",
                        "reject_code": current_error or "geometry_cache_metadata_unavailable",
                        "feature_source": feature_source,
                        "descriptor_match_mode": descriptor_match_mode,
                        "current_orb": current_diagnostics,
                        "geometry_accepted": False,
                        "information_valid": False, "pgo_comparison_eligible": False,
                    })
                    continue
                candidate_frame = LoopFrameRecord.load(record_root / candidate_meta["file"])
                candidate_geometry, candidate_error, candidate_diagnostics = load_geometry(
                    candidate_meta, candidate_frame,
                )
                if candidate_geometry is None:
                    rows.append({
                        "pair_id": pair_id, "status": "rejected",
                        "reject_code": candidate_error or "geometry_sidecar_unavailable",
                        "feature_source": feature_source,
                        "descriptor_match_mode": descriptor_match_mode,
                        "current_orb": current_diagnostics,
                        "candidate_orb": candidate_diagnostics,
                        "geometry_accepted": False,
                        "information_valid": False, "pgo_comparison_eligible": False,
                    })
                    continue
                try:
                    result = verify_fixed_geometry(
                        config, query, candidate, current_frame, candidate_frame,
                        current_geometry, candidate_geometry, pose_snapshot,
                        self.match_cov_default, fixed_information,
                    )
                except Exception as error:
                    rows.append({
                        "pair_id": pair_id, "status": "rejected",
                        "reject_code": "verification_exception", "reject_reason": str(error),
                        "feature_source": feature_source,
                        "descriptor_match_mode": descriptor_match_mode,
                        "current_orb": current_diagnostics,
                        "candidate_orb": candidate_diagnostics,
                        "geometry_accepted": False, "information_valid": False,
                        "pgo_comparison_eligible": False,
                    })
                    continue
                result.row.update({
                    "feature_source": feature_source,
                    "descriptor_match_mode": descriptor_match_mode,
                    "current_orb": current_diagnostics,
                    "candidate_orb": candidate_diagnostics,
                })
                rows.append(result.row)
                if result.constraint is not None:
                    result.row["selected_for_query"] = False
                    accepted.append((int(candidate["sensor_frame_idx"]), result))
            if accepted:
                _, chosen = min(accepted, key=lambda item: item[0])
                chosen.row["selected_for_query"] = True
                assert chosen.constraint is not None
                selected_constraints.append(chosen.constraint)
                if chosen.covariance_information is not None:
                    fixed_payload = chosen.constraint.to_dict()
                    covariance_payload = dict(fixed_payload)
                    covariance_payload["information"] = (
                        chosen.covariance_information.detach().cpu().tolist()
                    )
                    pgo_fixed_constraints.append(fixed_payload)
                    pgo_covariance_constraints.append(covariance_payload)

        if not torch.equal(poses, pose_snapshot):
            raise RuntimeError("VINS-style loop verification modified VisualMap poses")
        payload = {
            "schema_version": 1,
            "mode": "vins_geometry_disp_information_observe",
            "feature_source": feature_source,
            "descriptor_match_mode": descriptor_match_mode,
            "summary": {
                "attempted_pairs": len(rows),
                "geometry_accepted_pairs": sum(row.get("geometry_accepted") is True for row in rows),
                "information_valid_pairs": sum(row.get("information_valid") is True for row in rows),
                "selected_constraints": len(selected_constraints),
                "selected_pgo_comparison_edges": sum(
                    row.get("selected_for_query") is True
                    and row.get("pgo_comparison_eligible") is True
                    for row in rows
                ),
            },
            "verifications": rows,
        }
        constraints_payload = {
            "schema_version": 1,
            "feature_source": feature_source,
            "descriptor_match_mode": descriptor_match_mode,
            "pose_direction": "relative_pose = T_candidate_current = inverse(T_current_candidate)",
            "information_policy": "fixed_information; covariance information is observe-only",
            "constraints": [constraint.to_dict() for constraint in selected_constraints],
        }
        pgo_fixed_payload = {
            "schema_version": 1,
            "feature_source": feature_source,
            "descriptor_match_mode": descriptor_match_mode,
            "information_policy": "fixed_information",
            "constraints": pgo_fixed_constraints,
        }
        pgo_covariance_payload = {
            "schema_version": 1,
            "feature_source": feature_source,
            "descriptor_match_mode": descriptor_match_mode,
            "information_policy": "disp_covariance_information_observe",
            "constraints": pgo_covariance_constraints,
        }
        self._write_json_bundle([
            ("loop_constraints.json", constraints_payload),
            ("loop_constraints_pgo_fixed.json", pgo_fixed_payload),
            ("loop_constraints_pgo_covariance.json", pgo_covariance_payload),
            ("loop_vins_verification.json", payload),
        ], uuid.uuid4().hex)
        Logger.write("info", f"VINS-style geometry accepted {len(selected_constraints)} loop constraints.")
        return selected_constraints
