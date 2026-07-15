from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

import macvo_dbow2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_descriptors(cache: Path) -> tuple[np.ndarray, Path, Path]:
    index_path = cache / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for metadata in index.get("records", []):
        record_path = cache / metadata["file"]
        try:
            payload = torch.load(record_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(record_path, map_location="cpu")
        descriptors = np.asarray(payload["orb_descriptors"], dtype=np.uint8)
        if descriptors.ndim == 2 and descriptors.shape[1] == 32 and len(descriptors) > 0:
            return np.ascontiguousarray(descriptors), index_path, record_path
    raise RuntimeError(f"no non-empty ORB descriptor cache was found under {cache}")


def loaded_opencv_libraries() -> list[str]:
    paths: set[str] = set()
    with open("/proc/self/maps", "r", encoding="utf-8") as stream:
        for line in stream:
            path = line.rstrip().split()[-1]
            if "libopencv_" in path and path.startswith("/"):
                paths.add(os.path.realpath(path))
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MAC-VO DBoW2 Docker build gate.")
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path, help="loop_closure cache containing index.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocabulary-archive", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for required in (args.vocabulary, args.vocabulary_archive, args.cache / "index.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    descriptors, index_path, record_path = load_descriptors(args.cache)
    first_line = args.vocabulary.open("r", encoding="utf-8").readline().strip().split()
    if len(first_line) != 4:
        raise RuntimeError(f"unexpected ORBvoc header: {first_line}")

    build_info = cv2.getBuildInformation()
    build_info_path = args.output / "opencv_build_information.txt"
    build_info_path.write_text(build_info, encoding="utf-8")

    rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    database = macvo_dbow2.OrbDatabase(str(args.vocabulary))
    load_seconds = time.perf_counter() - started
    rss_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    entry_id = int(database.add(descriptors))
    results = database.query(descriptors)
    if not results or int(results[0][0]) != entry_id or float(results[0][1]) <= 0.0:
        raise RuntimeError(f"DBoW2 self-query failed: {results[:3]}")

    ldd = subprocess.run(
        ["ldd", str(Path(macvo_dbow2.__file__).resolve())],
        check=True, text=True, capture_output=True,
    ).stdout
    (args.output / "dbow2_extension_ldd.txt").write_text(ldd, encoding="utf-8")
    opencv_distributions = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if str(dist.metadata.get("Name", "")).lower().startswith("opencv-")
    )
    if opencv_distributions:
        raise RuntimeError(f"pip OpenCV distributions are installed: {opencv_distributions}")

    native_metadata = dict(database.metadata())
    expected_metadata = {
        "branching": int(first_line[0]),
        "depth": int(first_line[1]),
        "scoring": int(first_line[2]),
        "weighting": int(first_line[3]),
    }
    if any(int(native_metadata[key]) != value for key, value in expected_metadata.items()):
        raise RuntimeError(
            f"loaded vocabulary metadata {native_metadata} does not match header {expected_metadata}"
        )
    opencv_libraries = loaded_opencv_libraries()
    opencv_library_roots = sorted({str(Path(path).parent) for path in opencv_libraries})
    if len(opencv_library_roots) != 1:
        raise RuntimeError(f"multiple OpenCV library roots were loaded: {opencv_library_roots}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "cv2_file": str(Path(cv2.__file__).resolve()),
        "dbow2_extension_file": str(Path(macvo_dbow2.__file__).resolve()),
        "loaded_opencv_libraries": opencv_libraries,
        "loaded_opencv_library_roots": opencv_library_roots,
        "opencv_build_information_sha256": sha256(build_info_path),
        "vocabulary_path": str(args.vocabulary.resolve()),
        "vocabulary_sha256": sha256(args.vocabulary),
        "vocabulary_archive_sha256": sha256(args.vocabulary_archive),
        "vocabulary_header": {
            "branching": int(first_line[0]),
            "depth": int(first_line[1]),
            "scoring": int(first_line[2]),
            "weighting": int(first_line[3]),
        },
        "native_metadata": native_metadata,
        "orb_slam3_commit": str(macvo_dbow2.orb_slam3_commit),
        "load_seconds": load_seconds,
        "max_rss_before_kib": int(rss_before_kib),
        "max_rss_after_kib": int(rss_after_kib),
        "max_rss_delta_kib": int(max(rss_after_kib - rss_before_kib, 0)),
        "cache_index_sha256": sha256(index_path),
        "cache_record": str(record_path.resolve()),
        "cache_record_sha256": sha256(record_path),
        "descriptor_count": int(len(descriptors)),
        "self_query_score": float(results[0][1]),
        "pip_opencv_distributions": opencv_distributions,
    }
    target = args.output / "dbow2_docker_gate.json"
    target.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
