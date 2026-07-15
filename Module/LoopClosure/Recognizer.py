from __future__ import annotations

import hashlib
import importlib
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


@dataclass(frozen=True)
class FrameIdentity:
    sensor_frame_idx: int
    visual_map_idx: int
    loop_frame_idx: int


@dataclass(frozen=True)
class CausalQueryResult:
    candidates: list[RetrievalResult]
    query_time_ms: float | None
    eligible_database_entries: int
    retrieved_before_score_filter: int
    after_score_filter: int
    empty_descriptors: bool


def _descriptor_array(descriptors: torch.Tensor | np.ndarray) -> np.ndarray:
    result = np.asarray(torch.as_tensor(descriptors).detach().cpu(), dtype=np.uint8)
    if result.ndim != 2 or result.shape[1] != 32:
        raise ValueError(f"ORB descriptors must have shape (N,32), got {result.shape}")
    return np.ascontiguousarray(result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ORBFeatureExtractor:
    def __init__(self, nfeatures: int = 1000, scale_factor: float = 1.2, nlevels: int = 8) -> None:
        if not hasattr(cv2, "ORB_create"):
            raise RuntimeError("This OpenCV build does not provide cv2.ORB_create")
        self.orb = cv2.ORB_create(
            nfeatures=int(nfeatures), scaleFactor=float(scale_factor), nlevels=int(nlevels)
        )

    @staticmethod
    def rgb_tensor_to_gray(image: torch.Tensor) -> np.ndarray:
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(f"Expected an unbatched RGB tensor, got {tuple(image.shape)}")
            image = image[0]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected RGB tensor shape (3,H,W), got {tuple(image.shape)}")
        rgb = (
            image.detach().float().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)
            .permute(1, 2, 0).contiguous().numpy()
        )
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


class ORBPlaceRecognizer(ORBFeatureExtractor):
    """Backward-compatible custom-vocabulary recognizer used by older callers/tests."""

    def __init__(self, vocabulary_path: str | Path, nfeatures: int = 1000, scale_factor: float = 1.2, nlevels: int = 8) -> None:
        super().__init__(nfeatures, scale_factor, nlevels)
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

    def make_bow(self, descriptors: torch.Tensor) -> torch.Tensor | None:
        if self.vocabulary is None or len(descriptors) == 0:
            return None
        vector = torch.from_numpy(self.vocabulary.transform(_descriptor_array(descriptors)))
        return vector if torch.linalg.vector_norm(vector).item() > 0.0 else None


class PlaceRecognitionBackend(Protocol):
    @property
    def entry_count(self) -> int: ...
    def add(self, descriptors: torch.Tensor | np.ndarray) -> int | None: ...
    def query(self, descriptors: torch.Tensor | np.ndarray) -> list[tuple[int, float]]: ...
    def metadata(self) -> dict[str, Any]: ...


class CustomBinaryBackend:
    def __init__(
        self, vocabulary_path: str | Path, nfeatures: int = 1000,
        scale_factor: float = 1.2, nlevels: int = 8,
    ) -> None:
        path = Path(vocabulary_path)
        if not path.is_file():
            raise FileNotFoundError(f"custom loop vocabulary is missing: {path}")
        self.path = path
        self.vocabulary = BinaryVocabulary.load(path)
        expected = (int(nfeatures), float(scale_factor), int(nlevels))
        actual = (
            self.vocabulary.orb_nfeatures,
            self.vocabulary.orb_scale_factor,
            self.vocabulary.orb_nlevels,
        )
        if actual != expected:
            raise ValueError(f"ORB vocabulary parameters {actual} do not match runtime parameters {expected}")
        self._entries: list[np.ndarray] = []

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def _transform(self, descriptors: torch.Tensor | np.ndarray) -> np.ndarray | None:
        array = _descriptor_array(descriptors)
        if len(array) == 0:
            return None
        vector = np.asarray(self.vocabulary.transform(array), dtype=np.float32)
        return vector if float(np.linalg.norm(vector)) > 0.0 else None

    def add(self, descriptors: torch.Tensor | np.ndarray) -> int | None:
        vector = self._transform(descriptors)
        if vector is None:
            return None
        entry_id = len(self._entries)
        self._entries.append(vector.copy())
        return entry_id

    def query(self, descriptors: torch.Tensor | np.ndarray) -> list[tuple[int, float]]:
        vector = self._transform(descriptors)
        if vector is None:
            return []
        return [(entry_id, float(np.dot(vector, candidate))) for entry_id, candidate in enumerate(self._entries)]

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "custom_binary",
            "vocabulary_path": str(self.path),
            "vocabulary_sha256": _sha256(self.path),
            "vocabulary_checksum": self.vocabulary.checksum(),
            "words": self.vocabulary.num_words,
            "zero_score_entries_omitted": False,
        }


class DBoW2ORBBackend:
    ORB_SLAM3_COMMIT = "0df83dde1c85c7ab91a0d47de7a29685d046f637"

    def __init__(self, vocabulary_path: str | Path) -> None:
        path = Path(vocabulary_path)
        if not path.is_file():
            raise FileNotFoundError(f"ORBvoc text vocabulary is missing: {path}")
        native = importlib.import_module("macvo_dbow2")
        self.path = path
        self._database = native.OrbDatabase(str(path))
        self._native_commit = str(getattr(native, "orb_slam3_commit", "unknown"))

    @property
    def entry_count(self) -> int:
        return int(self._database.size())

    def add(self, descriptors: torch.Tensor | np.ndarray) -> int | None:
        array = _descriptor_array(descriptors)
        if len(array) == 0:
            return None
        return int(self._database.add(array))

    def query(self, descriptors: torch.Tensor | np.ndarray) -> list[tuple[int, float]]:
        array = _descriptor_array(descriptors)
        if len(array) == 0:
            return []
        return [(int(entry_id), float(score)) for entry_id, score in self._database.query(array)]

    def metadata(self) -> dict[str, Any]:
        result = dict(self._database.metadata())
        result.update({
            "type": "dbow2_orb",
            "vocabulary_path": str(self.path),
            "vocabulary_sha256": _sha256(self.path),
            "orb_slam3_commit": self._native_commit,
            "vocabulary_training_source": "not_authoritatively_verified",
            "zero_score_entries_omitted": True,
        })
        return result


class CausalRetrievalController:
    def __init__(
        self, backend: PlaceRecognitionBackend, temporal_exclusion_sensor_frames: int,
        top_k: int, min_score: float | None,
    ) -> None:
        self.backend = backend
        self.temporal_exclusion = int(temporal_exclusion_sensor_frames)
        self.top_k = int(top_k)
        self.min_score = None if min_score is None else float(min_score)
        self._pending: deque[tuple[FrameIdentity, np.ndarray]] = deque()
        self._entry_identity: dict[int, FrameIdentity] = {}

    def _promote_eligible(self, current_sensor_frame_idx: int) -> None:
        while self._pending:
            identity, descriptors = self._pending[0]
            if current_sensor_frame_idx - identity.sensor_frame_idx < self.temporal_exclusion:
                break
            self._pending.popleft()
            entry_id = self.backend.add(descriptors)
            if entry_id is None:
                continue
            if entry_id in self._entry_identity:
                raise RuntimeError(f"duplicate place-database entry id {entry_id}")
            self._entry_identity[entry_id] = identity
            if self.backend.entry_count != len(self._entry_identity):
                raise RuntimeError("backend entry count diverged from the causal identity map")

    def query_then_enqueue(
        self, identity: FrameIdentity, descriptors: torch.Tensor | np.ndarray,
    ) -> CausalQueryResult:
        if self._pending and self._pending[-1][0].sensor_frame_idx >= identity.sensor_frame_idx:
            raise AssertionError("loop frames must be processed in strictly increasing sensor-frame order")
        if self._entry_identity:
            newest = max(item.sensor_frame_idx for item in self._entry_identity.values())
            if newest >= identity.sensor_frame_idx:
                raise AssertionError("causal database contains a future frame")

        self._promote_eligible(identity.sensor_frame_idx)
        eligible = len(self._entry_identity)
        array = _descriptor_array(descriptors)
        if len(array) == 0:
            return CausalQueryResult([], None, eligible, 0, 0, True)

        started = time.perf_counter()
        raw = self.backend.query(array)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        retrieved = len(raw)
        filtered: list[RetrievalResult] = []
        for entry_id, score in raw:
            if not math.isfinite(score):
                continue
            if self.min_score is not None and score < self.min_score:
                continue
            candidate = self._entry_identity.get(int(entry_id))
            if candidate is None:
                raise RuntimeError(f"backend returned unknown entry id {entry_id}")
            gap = identity.sensor_frame_idx - candidate.sensor_frame_idx
            if gap < self.temporal_exclusion:
                raise AssertionError("backend returned a temporally ineligible candidate")
            filtered.append(RetrievalResult(
                candidate.sensor_frame_idx, candidate.visual_map_idx, candidate.loop_frame_idx,
                float(score), int(gap),
            ))
        filtered.sort(key=lambda item: (-item.score, item.sensor_frame_idx))
        after_score_filter = len(filtered)
        self._pending.append((identity, array.copy()))
        return CausalQueryResult(
            filtered[:self.top_k], elapsed_ms, eligible, retrieved,
            after_score_filter, False,
        )


class CausalBoWDatabase:
    """Legacy dense-vector database retained for old callers and regression tests."""

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
            results.append(RetrievalResult(
                candidate_sensor, candidate_map, candidate_loop,
                float(np.dot(query_vector, candidate_vector)), gap,
            ))
        results.sort(key=lambda item: (-item.score, item.sensor_frame_idx))
        return results[:int(top_k)], (time.perf_counter() - started) * 1000.0
