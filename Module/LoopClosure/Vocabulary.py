from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np


_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def _hamming_distance(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if data.size == 0 or centers.size == 0:
        return np.empty((len(data), len(centers)), dtype=np.int32)
    result = np.empty((len(data), len(centers)), dtype=np.int32)
    for start in range(0, len(data), 4096):
        batch = data[start:start + 4096]
        xor = np.bitwise_xor(batch[:, None, :], centers[None, :, :])
        result[start:start + len(batch)] = _POPCOUNT[xor].sum(axis=2)
    return result


def _binary_center(data: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(data, axis=1)
    # Strict majority; ties deterministically resolve to zero.
    majority = (bits.sum(axis=0) * 2 > len(data)).astype(np.uint8)
    return np.packbits(majority)


class BinaryVocabulary:
    FORMAT_VERSION = 1

    def __init__(
        self,
        centers: np.ndarray,
        children: np.ndarray,
        word_ids: np.ndarray,
        idf: np.ndarray,
        branch: int,
        depth: int,
        seed: int,
        orb_nfeatures: int = 1000,
        orb_scale_factor: float = 1.2,
        orb_nlevels: int = 8,
    ) -> None:
        self.centers = np.asarray(centers, dtype=np.uint8)
        self.children = np.asarray(children, dtype=np.int32)
        self.word_ids = np.asarray(word_ids, dtype=np.int32)
        self.idf = np.asarray(idf, dtype=np.float32)
        self.branch = int(branch)
        self.depth = int(depth)
        self.seed = int(seed)
        self.orb_nfeatures = int(orb_nfeatures)
        self.orb_scale_factor = float(orb_scale_factor)
        self.orb_nlevels = int(orb_nlevels)
        if self.centers.ndim != 2 or self.centers.shape[1] != 32:
            raise ValueError(f"ORB vocabulary centers must have shape (N, 32), got {self.centers.shape}")
        if self.children.shape != (len(self.centers), self.branch):
            raise ValueError("Vocabulary child table has an invalid shape")

    @property
    def num_words(self) -> int:
        return int(len(self.idf))

    @staticmethod
    def _cluster(data: np.ndarray, count: int, max_iterations: int) -> tuple[np.ndarray, np.ndarray]:
        count = min(int(count), len(data))
        centers = [data[0].copy()]
        while len(centers) < count:
            distance = _hamming_distance(data, np.stack(centers)).min(axis=1)
            centers.append(data[int(np.argmax(distance))].copy())
        center_array = np.stack(centers)

        labels = np.zeros(len(data), dtype=np.int32)
        for _ in range(max_iterations):
            distance = _hamming_distance(data, center_array)
            new_labels = distance.argmin(axis=1).astype(np.int32)
            new_centers = center_array.copy()
            min_distance = distance[np.arange(len(data)), new_labels]
            reserved: set[int] = set()
            for cluster_idx in range(count):
                members = data[new_labels == cluster_idx]
                if len(members) > 0:
                    new_centers[cluster_idx] = _binary_center(members)
                else:
                    for candidate in np.argsort(-min_distance, kind="stable"):
                        if int(candidate) not in reserved:
                            reserved.add(int(candidate))
                            new_centers[cluster_idx] = data[int(candidate)]
                            break
            if np.array_equal(new_labels, labels) and np.array_equal(new_centers, center_array):
                labels = new_labels
                break
            labels, center_array = new_labels, new_centers
        return labels, center_array

    @classmethod
    def train(
        cls,
        documents: Sequence[np.ndarray],
        branch: int = 8,
        depth: int = 4,
        seed: int = 42,
        max_iterations: int = 30,
        max_descriptors: int = 1_000_000,
        orb_nfeatures: int = 1000,
        orb_scale_factor: float = 1.2,
        orb_nlevels: int = 8,
    ) -> "BinaryVocabulary":
        valid_documents = [np.asarray(doc, dtype=np.uint8) for doc in documents if doc is not None and len(doc) > 0]
        if len(valid_documents) == 0:
            raise ValueError("Cannot train a vocabulary without ORB descriptors")
        if any(doc.ndim != 2 or doc.shape[1] != 32 for doc in valid_documents):
            raise ValueError("Every ORB descriptor document must have shape (N, 32)")

        all_descriptors = np.concatenate(valid_documents, axis=0)
        if len(all_descriptors) > max_descriptors:
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(len(all_descriptors), max_descriptors, replace=False))
            training = all_descriptors[selected]
        else:
            training = all_descriptors

        centers: list[np.ndarray] = []
        child_rows: list[list[int]] = []
        words: list[int] = []
        next_word = 0

        def build_node(data: np.ndarray, level: int) -> int:
            nonlocal next_word
            node_idx = len(centers)
            centers.append(_binary_center(data))
            child_rows.append([-1] * branch)
            words.append(-1)
            if level >= depth or len(data) <= 1:
                words[node_idx] = next_word
                next_word += 1
                return node_idx

            labels, child_centers = cls._cluster(data, branch, max_iterations)
            for cluster_idx in range(len(child_centers)):
                members = data[labels == cluster_idx]
                if len(members) == 0:
                    continue
                child_idx = build_node(members, level + 1)
                centers[child_idx] = child_centers[cluster_idx]
                child_rows[node_idx][cluster_idx] = child_idx
            if all(child < 0 for child in child_rows[node_idx]):
                words[node_idx] = next_word
                next_word += 1
            return node_idx

        build_node(training, 0)
        provisional = cls(
            np.stack(centers), np.asarray(child_rows), np.asarray(words),
            np.ones(next_word, dtype=np.float32), branch, depth, seed,
            orb_nfeatures, orb_scale_factor, orb_nlevels,
        )
        document_frequency = np.zeros(next_word, dtype=np.int64)
        for document in valid_documents:
            document_frequency[np.unique(provisional.quantize(document))] += 1
        idf = np.log((len(valid_documents) + 1.0) / (document_frequency + 1.0)) + 1.0
        provisional.idf = idf.astype(np.float32)
        return provisional

    def quantize(self, descriptors: np.ndarray) -> np.ndarray:
        descriptors = np.asarray(descriptors, dtype=np.uint8)
        if descriptors.ndim != 2 or descriptors.shape[1] != 32:
            raise ValueError(f"ORB descriptors must have shape (N, 32), got {descriptors.shape}")
        output = np.empty(len(descriptors), dtype=np.int32)
        for descriptor_idx, descriptor in enumerate(descriptors):
            node = 0
            while self.word_ids[node] < 0:
                child_indices = self.children[node]
                child_indices = child_indices[child_indices >= 0]
                if len(child_indices) == 0:
                    raise RuntimeError(f"Vocabulary node {node} has neither a word nor children")
                distances = _hamming_distance(descriptor[None, :], self.centers[child_indices])[0]
                node = int(child_indices[int(np.argmin(distances))])
            output[descriptor_idx] = self.word_ids[node]
        return output

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        vector = np.zeros(self.num_words, dtype=np.float32)
        if descriptors is None or len(descriptors) == 0:
            return vector
        words = self.quantize(descriptors)
        vector += np.bincount(words, minlength=self.num_words).astype(np.float32)
        vector /= max(float(vector.sum()), 1.0)
        vector *= self.idf
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for value in (self.centers, self.children, self.word_ids, self.idf):
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(json.dumps({
            "branch": self.branch, "depth": self.depth, "seed": self.seed,
            "orb_nfeatures": self.orb_nfeatures,
            "orb_scale_factor": self.orb_scale_factor,
            "orb_nlevels": self.orb_nlevels,
        }, sort_keys=True).encode())
        return digest.hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps({
            "format_version": self.FORMAT_VERSION,
            "branch": self.branch,
            "depth": self.depth,
            "seed": self.seed,
            "num_words": self.num_words,
            "orb_nfeatures": self.orb_nfeatures,
            "orb_scale_factor": self.orb_scale_factor,
            "orb_nlevels": self.orb_nlevels,
            "checksum": self.checksum(),
        }, sort_keys=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(temporary, "wb") as file:
                np.savez_compressed(
                    file,
                    centers=self.centers,
                    children=self.children,
                    word_ids=self.word_ids,
                    idf=self.idf,
                    metadata=np.asarray(metadata),
                )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: Path) -> "BinaryVocabulary":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if int(metadata["format_version"]) != cls.FORMAT_VERSION:
                raise ValueError(f"Unsupported vocabulary format {metadata['format_version']}")
            vocabulary = cls(
                data["centers"], data["children"], data["word_ids"], data["idf"],
                metadata["branch"], metadata["depth"], metadata["seed"],
                metadata["orb_nfeatures"], metadata["orb_scale_factor"], metadata["orb_nlevels"],
            )
        if vocabulary.checksum() != metadata["checksum"]:
            raise ValueError(f"Vocabulary checksum mismatch for {path}")
        return vocabulary
