from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypose as pp
import torch

from DataLoader import StereoData


@dataclass
class LoopFrameRecord:
    sensor_frame_idx: int
    visual_map_idx: int
    loop_frame_idx: int
    frame_ns: int
    height: int
    width: int
    image_left: torch.Tensor
    image_right: torch.Tensor
    intrinsic: torch.Tensor
    baseline: torch.Tensor
    body_to_sensor: torch.Tensor
    depth: torch.Tensor
    depth_covariance: torch.Tensor | None
    registered_pose: torch.Tensor
    orb_keypoints: torch.Tensor
    orb_descriptors: torch.Tensor
    bow_vector: torch.Tensor | None

    SCHEMA_VERSION = 1

    @staticmethod
    def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
        return value.detach().cpu().clone()

    def to_stereo_data(self, device: torch.device | str = "cpu") -> StereoData:
        target = torch.device(device)
        return StereoData(
            T_BS=pp.SE3(self.body_to_sensor.to(device=target, dtype=torch.float32)),
            K=self.intrinsic.to(device=target, dtype=torch.float32),
            baseline=self.baseline.to(device=target, dtype=torch.float32),
            time_ns=[int(self.frame_ns)],
            height=int(self.height),
            width=int(self.width),
            imageL=self.image_left.to(device=target),
            imageR=self.image_right.to(device=target),
            gt_flow=None,
            flow_mask=None,
            gt_depth=None,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "sensor_frame_idx": int(self.sensor_frame_idx),
            "visual_map_idx": int(self.visual_map_idx),
            "loop_frame_idx": int(self.loop_frame_idx),
            "frame_ns": int(self.frame_ns),
            "height": int(self.height),
            "width": int(self.width),
            "image_left": self._cpu_clone(self.image_left),
            "image_right": self._cpu_clone(self.image_right),
            "intrinsic": self._cpu_clone(self.intrinsic),
            "baseline": self._cpu_clone(self.baseline),
            "body_to_sensor": self._cpu_clone(self.body_to_sensor),
            "depth": self._cpu_clone(self.depth).float(),
            "depth_covariance": None if self.depth_covariance is None else self._cpu_clone(self.depth_covariance).float(),
            "registered_pose": self._cpu_clone(self.registered_pose),
            "orb_keypoints": self._cpu_clone(self.orb_keypoints),
            "orb_descriptors": self._cpu_clone(self.orb_descriptors),
            "bow_vector": None if self.bow_vector is None else self._cpu_clone(self.bow_vector).float(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LoopFrameRecord":
        version = int(payload.get("schema_version", -1))
        if version != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported loop-frame schema version {version}")
        fields = dict(payload)
        fields.pop("schema_version")
        return cls(**fields)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(self.to_payload(), temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: Path) -> "LoopFrameRecord":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch versions before weights_only was introduced.
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Loop-frame cache {path} does not contain a dictionary")
        return cls.from_payload(payload)


@dataclass
class GeometryFeatureRecord:
    """Optional fixed VO features used only by loop geometric verification."""

    sensor_frame_idx: int
    visual_map_idx: int
    loop_frame_idx: int
    orb_config_sha256: str
    original_index: torch.Tensor
    pixel_uv: torch.Tensor
    point_camera: torch.Tensor
    point_covariance_camera: torch.Tensor
    disparity: torch.Tensor
    disparity_variance: torch.Tensor
    disparity_valid: torch.Tensor
    descriptor: torch.Tensor

    SCHEMA_VERSION = 1

    def to_payload(self) -> dict[str, Any]:
        tensor_fields = (
            "original_index", "pixel_uv", "point_camera", "point_covariance_camera",
            "disparity", "disparity_variance", "disparity_valid", "descriptor",
        )
        payload: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "sensor_frame_idx": int(self.sensor_frame_idx),
            "visual_map_idx": int(self.visual_map_idx),
            "loop_frame_idx": int(self.loop_frame_idx),
            "orb_config_sha256": str(self.orb_config_sha256),
        }
        for name in tensor_fields:
            payload[name] = LoopFrameRecord._cpu_clone(getattr(self, name))
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GeometryFeatureRecord":
        version = int(payload.get("schema_version", -1))
        if version != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported geometry-feature schema version {version}")
        fields = dict(payload)
        fields.pop("schema_version")
        record = cls(**fields)
        count = int(record.original_index.numel())
        if any(len(getattr(record, name)) != count for name in (
            "pixel_uv", "point_camera", "point_covariance_camera", "disparity",
            "disparity_variance", "disparity_valid", "descriptor",
        )):
            raise ValueError("Geometry-feature fields do not have a common length")
        if record.descriptor.ndim != 2 or record.descriptor.shape[1] != 32:
            raise ValueError(f"Invalid ORB descriptor shape {tuple(record.descriptor.shape)}")
        return record

    def save_if_absent(self, path: Path) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return False
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(self.to_payload(), temporary)
            if path.exists():
                return False
            os.replace(temporary, path)
            return True
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: Path) -> "GeometryFeatureRecord":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Geometry-feature sidecar {path} does not contain a dictionary")
        return cls.from_payload(payload)
