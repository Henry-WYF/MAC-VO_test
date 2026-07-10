from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from DataLoader import StereoFrame
from Module.Frontend.Frontend import IFrontend
from Module.Frontend.StereoDepth import IStereoDepth
from Module.Map import VisualMap
from Utility.Extensions import ConfigTestable
from Utility.PrettyPrint import Logger

from .Recognizer import CausalBoWDatabase, ORBPlaceRecognizer
from .Record import LoopFrameRecord
from .Verification import LoopCandidateVerifier, LoopConstraint, has_required_pnp_functions


class LoopClosureManager(ConfigTestable):
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self.disabled_reason: str | None = None
        self.output_dir: Path | None = None
        self.records: list[dict[str, Any]] = []
        self.last_registered_sensor_idx: int | None = None
        self.recognizer: ORBPlaceRecognizer | None = None
        self.frontend: IFrontend | None = None
        self.geometry_enabled = bool(getattr(getattr(config, "geometric_verification", None), "enabled", False))
        if self.enabled:
            try:
                self.recognizer = ORBPlaceRecognizer(
                    config.vocabulary_path, config.orb_nfeatures,
                    config.orb_scale_factor, config.orb_nlevels,
                )
                if self.recognizer.vocabulary is None:
                    Logger.write("warn", f"Loop vocabulary '{config.vocabulary_path}' is missing. Frames will be cached, but BoW query is disabled.")
            except Exception as error:
                self._disable(f"Failed to initialize ORB place recognizer: {error}")

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        base_spec = {
            "enabled": lambda value: isinstance(value, bool),
            "vocabulary_path": lambda value: isinstance(value, str),
            "keyframe_stride_sensor_frames": lambda value: isinstance(value, int) and value > 0,
            "temporal_exclusion_sensor_frames": lambda value: isinstance(value, int) and value >= 0,
            "cache_failure_policy": lambda value: value in {"disable_loop", "raise"},
            "orb_nfeatures": lambda value: isinstance(value, int) and value > 0,
            "orb_scale_factor": lambda value: isinstance(value, (int, float)) and value > 1.0,
            "orb_nlevels": lambda value: isinstance(value, int) and value > 0,
            "top_k": lambda value: isinstance(value, int) and value > 0,
        }
        cls._enforce_config_spec(config, base_spec, allow_excessive_cfg=True)
        excessive = set(vars(config)) - (set(base_spec) | {"geometric_verification", "geometry", "pnp", "verification", "loop_information"})
        if excessive:
            raise KeyError(f"Excessive Keys: {excessive} from {list(base_spec)}")
        if hasattr(config, "geometric_verification"):
            cls._enforce_config_spec(config.geometric_verification, {
                "enabled": lambda value: isinstance(value, bool),
                "max_candidates_to_verify": lambda value: isinstance(value, int) and value > 0,
                "min_sensor_gap": lambda value: isinstance(value, int) and value >= 0,
            })
            cls._enforce_config_spec(config.geometry, {
                "min_points": lambda value: isinstance(value, int) and value >= 4,
                "max_points": lambda value: isinstance(value, int) and value > 0,
                "max_depth": lambda value: isinstance(value, (int, float)) and value > 0.0,
                "max_depth_cov": lambda value: isinstance(value, (int, float)) and value >= 0.0,
                "max_flow_cov": lambda value: isinstance(value, (int, float)) and value >= 0.0,
                "border": lambda value: isinstance(value, int) and value >= 0,
                "grid_rows": lambda value: isinstance(value, int) and value > 0,
                "grid_cols": lambda value: isinstance(value, int) and value > 0,
                "max_points_per_cell": lambda value: isinstance(value, int) and value > 0,
            })
            cls._enforce_config_spec(config.pnp, {
                "reproj_error_px": lambda value: isinstance(value, (int, float)) and value > 0.0,
                "confidence": lambda value: isinstance(value, (int, float)) and 0.0 < value < 1.0,
                "iterations": lambda value: isinstance(value, int) and value > 0,
                "min_inliers": lambda value: isinstance(value, int) and value >= 4,
                "min_inlier_ratio": lambda value: isinstance(value, (int, float)) and 0.0 <= value <= 1.0,
                "refine": lambda value: isinstance(value, bool),
            })
            cls._enforce_config_spec(config.verification, {
                "max_mean_reproj_error_px": lambda value: isinstance(value, (int, float)) and value > 0.0,
                "max_rotation_diff_deg": lambda value: isinstance(value, (int, float)) and value >= 0.0,
                "max_translation_diff_m": lambda value: isinstance(value, (int, float)) and value >= 0.0,
            })
            cls._enforce_config_spec(config.loop_information, {
                "trans_weight": lambda value: isinstance(value, (int, float)) and value > 0.0,
                "rot_weight": lambda value: isinstance(value, (int, float)) and value > 0.0,
            })

    def set_frontend(self, frontend: IFrontend) -> None:
        self.frontend = frontend

    def set_output_dir(self, output_dir: Path) -> None:
        if not self.enabled:
            return
        try:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            self.output_dir = None
            self._handle_cache_failure(error)

    def _disable(self, reason: str) -> None:
        self.enabled = False
        self.disabled_reason = reason
        Logger.write("error", f"Loop closure disabled: {reason}")

    def _handle_cache_failure(self, error: Exception) -> None:
        if self.config.cache_failure_policy == "raise":
            raise error
        self._disable(f"cache failure: {error}")

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
        if not self.enabled:
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
        assert self.recognizer is not None
        try:
            retrieval_started = time.perf_counter()
            keypoints, descriptors = self.recognizer.extract(frame.stereo.imageL)
            bow = self.recognizer.make_bow(descriptors)
            retrieval_build_ms = (time.perf_counter() - retrieval_started) * 1000.0
            loop_idx = len(self.records)
            record = LoopFrameRecord(
                sensor_frame_idx=int(frame.frame_idx), visual_map_idx=int(visual_map_idx), loop_frame_idx=loop_idx,
                frame_ns=int(frame.stereo.frame_ns), height=int(frame.stereo.height), width=int(frame.stereo.width),
                image_left=frame.stereo.imageL, image_right=frame.stereo.imageR, intrinsic=frame.stereo.K,
                baseline=frame.stereo.baseline, body_to_sensor=frame.stereo.T_BS.tensor(), depth=depth.depth,
                depth_covariance=depth.cov, registered_pose=registered_pose, orb_keypoints=keypoints,
                orb_descriptors=descriptors, bow_vector=bow,
            )
            relative_path = Path("frames", f"loop_{loop_idx:06d}_sensor_{frame.frame_idx:09d}.pt")
            record.save(self.output_dir / relative_path)
            self.records.append({
                "sensor_frame_idx": int(frame.frame_idx), "visual_map_idx": int(visual_map_idx),
                "loop_frame_idx": loop_idx, "file": relative_path.as_posix(),
                "orb_features": int(len(descriptors)), "has_bow": bow is not None,
                "retrieval_build_time_ms": retrieval_build_ms,
            })
            self._write_index()
            self.last_registered_sensor_idx = int(frame.frame_idx)
            return True
        except Exception as error:
            self._handle_cache_failure(error)
            return False

    def detect_all(self) -> list[dict[str, Any]]:
        if not self.enabled or self.output_dir is None or self.recognizer is None:
            return []
        if self.recognizer.vocabulary is None:
            Logger.write("warn", "Skip causal BoW query because no vocabulary was loaded.")
            return []
        database = CausalBoWDatabase(self.config.temporal_exclusion_sensor_frames)
        queries: list[dict[str, Any]] = []
        for metadata in sorted(self.records, key=lambda item: item["sensor_frame_idx"]):
            record = LoopFrameRecord.load(self.output_dir / metadata["file"])
            if record.bow_vector is None:
                continue
            candidates, elapsed_ms = database.query(record.sensor_frame_idx, record.bow_vector, self.config.top_k)
            assert all(candidate.sensor_frame_idx < record.sensor_frame_idx for candidate in candidates)
            queries.append({
                "sensor_frame_idx": record.sensor_frame_idx, "visual_map_idx": record.visual_map_idx,
                "loop_frame_idx": record.loop_frame_idx, "orb_features": int(len(record.orb_descriptors)),
                "retrieval_build_time_ms": float(metadata.get("retrieval_build_time_ms", 0.0)),
                "query_time_ms": elapsed_ms, "candidates": [candidate.__dict__ for candidate in candidates],
            })
            database.add(record.sensor_frame_idx, record.visual_map_idx, record.loop_frame_idx, record.bow_vector)
        with open(self.output_dir / "queries.json", "w", encoding="utf-8") as file:
            json.dump({
                "schema_version": 1, "temporal_exclusion_unit": "sensor_frame_idx",
                "temporal_exclusion": int(self.config.temporal_exclusion_sensor_frames),
                "top_k": int(self.config.top_k), "queries": queries,
            }, file, indent=2)
        return queries


    def _record_by_loop_idx(self) -> dict[int, dict[str, Any]]:
        return {int(record["loop_frame_idx"]): record for record in self.records}

    def verify_candidates(self, global_map: VisualMap, queries: list[dict[str, Any]]) -> list[LoopConstraint]:
        if not self.enabled or not self.geometry_enabled:
            return []
        if self.output_dir is None:
            Logger.write("warn", "Skip loop geometric verification because output directory is unavailable.")
            return []
        if self.frontend is None:
            Logger.write("error", "Skip loop geometric verification because Frontend was not injected.")
            return []
        if not has_required_pnp_functions():
            Logger.write("error", "Skip loop geometric verification because OpenCV PnP/calib3d functions are unavailable.")
            return []

        record_metadata = self._record_by_loop_idx()
        verifier = LoopCandidateVerifier(self.config, self.frontend)
        verification_rows: list[dict[str, Any]] = []
        constraints: list[LoopConstraint] = []
        max_candidates = int(self.config.geometric_verification.max_candidates_to_verify)

        for query in queries:
            current_meta = record_metadata.get(int(query["loop_frame_idx"]))
            if current_meta is None:
                continue
            current = LoopFrameRecord.load(self.output_dir / current_meta["file"])
            for candidate in query.get("candidates", [])[:max_candidates]:
                historical_meta = record_metadata.get(int(candidate["loop_frame_idx"]))
                if historical_meta is None:
                    continue
                historical = LoopFrameRecord.load(self.output_dir / historical_meta["file"])
                verification, constraint = verifier.verify(global_map, query, candidate, current, historical)
                verification_rows.append(verification.to_dict())
                if constraint is not None:
                    constraints.append(constraint)

        with open(self.output_dir / "loop_verification.json", "w", encoding="utf-8") as file:
            json.dump({
                "schema_version": 1,
                "max_candidates_to_verify": max_candidates,
                "verifications": verification_rows,
            }, file, indent=2)
        with open(self.output_dir / "loop_constraints.json", "w", encoding="utf-8") as file:
            json.dump({
                "schema_version": 1,
                "pose_direction": "relative_pose = T_candidate_current = inverse(T_current_candidate)",
                "constraints": [constraint.to_dict() for constraint in constraints],
            }, file, indent=2)
        Logger.write("info", f"Loop geometric verification accepted {len(constraints)} / {len(verification_rows)} candidates.")
        return constraints
