from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from Module.LoopClosure import LoopClosureManager
from Utility.Config import load_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def score(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("score must be finite and in [0,1]")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a MAC-VO loop cache through one Phase A backend without rerunning VO."
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backend", required=True, choices=("custom_binary", "dbow2_orb"))
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--min-score", type=score)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    record_dir = result_dir / "loop_closure"
    index_path = record_dir / "index.json"
    config_path = (args.config or result_dir / "config.yaml").resolve()
    output_dir = args.output_dir.resolve()
    vocabulary = args.vocabulary.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Phase A output directory is not empty: {output_dir}")
    for required in (index_path, config_path, vocabulary):
        if not required.is_file():
            raise FileNotFoundError(required)

    config, _ = load_config(config_path)
    loop_config = config.Odometry.loop_closure
    loop_config.recognizer_type = args.backend
    loop_config.vocabulary_path = str(vocabulary)
    loop_config.bow_min_score = args.min_score
    LoopClosureManager.is_valid_config(loop_config)

    index_payload = load_json(index_path)
    records = index_payload.get("records")
    if index_payload.get("schema_version") != 1 or not isinstance(records, list):
        raise ValueError("unsupported or malformed loop index")

    manager = LoopClosureManager(loop_config)
    manager.set_output_dir(output_dir)
    if not manager.cache_enabled or manager.output_dir is None:
        raise RuntimeError(f"loop manager could not initialize: {manager.disabled_reason}")
    manager.records = records
    started = time.perf_counter()
    queries = manager.detect_all(record_dir=record_dir, output_dir=output_dir)
    elapsed_seconds = time.perf_counter() - started
    if not manager.retrieval_enabled:
        raise RuntimeError(manager.disabled_reason or "Phase A retrieval was disabled")

    shutil.copy2(index_path, output_dir / "source_index.json")
    manifest = {
        "schema_version": 1,
        "source_result_dir": str(result_dir),
        "source_config": str(config_path),
        "source_index_sha256": sha256(index_path),
        "backend": args.backend,
        "vocabulary": str(vocabulary),
        "vocabulary_sha256": sha256(vocabulary),
        "bow_min_score": args.min_score,
        "query_count": len(queries),
        "returned_candidate_count": sum(len(query["candidates"]) for query in queries),
        "elapsed_seconds": elapsed_seconds,
        "code_commit": git_commit(Path(__file__).resolve().parents[2]),
        "code_source_sha256": {
            "manager": sha256(Path(__file__).resolve().parents[2] / "Module/LoopClosure/Manager.py"),
            "recognizer": sha256(Path(__file__).resolve().parents[2] / "Module/LoopClosure/Recognizer.py"),
        },
    }
    (output_dir / "phase_a_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
