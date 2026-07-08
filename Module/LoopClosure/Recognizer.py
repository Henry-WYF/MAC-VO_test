from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .Vocabulary import BinaryVocabulary


@dataclass(frozen=True)
class RetrievalResult:
    sensor_frame_idx: int
    visual_map_idx: int
    loop_frame_idx: int
    score: float
    sensor_frame_gap: int


class ORBPlaceRecognizer:
    def __init__(self, vocabulary_path: str | Path, nfeatures: int = 1000, scale_factor: float = 1.2, nlevels: int = 8) -> None:
        if not hasattr(cv2, "ORB_create"):
            raise RuntimeError("This OpenCV build does not provide cv2.ORB_create")
        self.orb = cv2.ORB_create(nfeatures=int(nfeatures), scaleFactor=float(scale_factor), nlevels=int(nlevels))
        path = Path(vocabulary_path)
        self.vocabulary = BinaryVocabulary.load(path) if path.is_file() else None
        self.vocabulary_path = path
        if self.vocabulary is not None:
            expected = (int(nfeatures), float(scale_factor), int(nlevels))
            actual = (
                self.vocabulary.orb_nfeatures,
                self.vocabulary.orb_scale_factor,
                self.vocabulary.orb_nlevels,
            )
            if actual != expected:
                raise ValueError(f"ORB vocabulary parameters {actual} do not match runtime parameters {expected}")

    @staticmethod
    def rgb_tensor_to_gray(image: torch.Tensor) -> np.ndarray:
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(f"Expected an unbatched RGB tensor, got {tuple(image.shape)}")
            image = image[0]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected RGB tensor shape (3,H,W), got {tuple(image.shape)}")
        rgb = image.detach().float().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

    def extract(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keypoints, descriptors = self.orb.detectAndCompute(self.rgb_tensor_to_gray(image), None)
        if descriptors is None or len(keypoints) == 0:
            return torch.empty((0, 7), dtype=torch.float32), torch.empty((0, 32), dtype=torch.uint8)
        serialized = torch.tensor([
            [kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, float(kp.octave), float(kp.class_id)]
            for kp in keypoints
        ], dtype=torch.float32)
        return serialized, torch.from_numpy(np.ascontiguousarray(descriptors)).to(torch.uint8)

    def make_bow(self, descriptors: torch.Tensor) -> torch.Tensor | None:
        if self.vocabulary is None or len(descriptors) == 0:
            return None
        vector = torch.from_numpy(self.vocabulary.transform(descriptors.detach().cpu().numpy()))
        return vector if torch.linalg.vector_norm(vector).item() > 0.0 else None


class CausalBoWDatabase:
    def __init__(self, temporal_exclusion_sensor_frames: int) -> None:
        self.temporal_exclusion = int(temporal_exclusion_sensor_frames)
        self._entries: list[tuple[int, int, int, np.ndarray]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, sensor_frame_idx: int, visual_map_idx: int, loop_frame_idx: int, bow_vector: torch.Tensor | np.ndarray) -> None:
        vector = np.asarray(torch.as_tensor(bow_vector).cpu(), dtype=np.float32)
        self._entries.append((int(sensor_frame_idx), int(visual_map_idx), int(loop_frame_idx), vector.copy()))

    def query(self, sensor_frame_idx: int, bow_vector: torch.Tensor | np.ndarray, top_k: int) -> tuple[list[RetrievalResult], float]:
        started = time.perf_counter()
        query_vector = np.asarray(torch.as_tensor(bow_vector).cpu(), dtype=np.float32)
        results: list[RetrievalResult] = []
        for candidate_sensor, candidate_map, candidate_loop, candidate_vector in self._entries:
            if candidate_sensor >= sensor_frame_idx:
                raise AssertionError(f"Causal database contains future frame {candidate_sensor} for query {sensor_frame_idx}")
            gap = int(sensor_frame_idx) - candidate_sensor
            if gap < self.temporal_exclusion:
                continue
            results.append(RetrievalResult(candidate_sensor, candidate_map, candidate_loop, float(np.dot(query_vector, candidate_vector)), gap))
        results.sort(key=lambda item: (-item.score, item.sensor_frame_idx))
        return results[:int(top_k)], (time.perf_counter() - started) * 1000.0
