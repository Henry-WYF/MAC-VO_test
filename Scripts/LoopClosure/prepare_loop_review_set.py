from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PROTOCOL_VERSION = "macvo-loop-overlap-review-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def pose(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(7)
    translation = values[:3]
    quaternion = values[3:]
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(values).all() or norm == 0.0:
        raise ValueError("invalid GT pose")
    return translation, quaternion / norm


def pair_geometry(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float]:
    lhs_t, lhs_q = pose(lhs)
    rhs_t, rhs_q = pose(rhs)
    distance = float(np.linalg.norm(lhs_t - rhs_t))
    cosine = float(np.clip(abs(np.dot(lhs_q, rhs_q)), 0.0, 1.0))
    angle_deg = math.degrees(2.0 * math.acos(cosine))
    return distance, angle_deg


def data_from_record(path: Path) -> tuple[np.ndarray, int]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    image = torch.as_tensor(payload["image_left"])
    if image.ndim == 4:
        image = image[0]
    rgb = image.float().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR), int(payload["frame_ns"])


def pair_ids(queries: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for query in queries.get("queries", []):
        current = int(query["loop_frame_idx"])
        for candidate in query.get("candidates", []):
            result.add(f"{current}:{int(candidate['loop_frame_idx'])}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a score-blind loop-overlap review set.")
    parser.add_argument("--cache", required=True, type=Path, help="loop_closure directory")
    parser.add_argument("--gt-poses", required=True, type=Path)
    parser.add_argument("--gt-index-space", required=True, choices=("sensor_frame_idx", "loop_frame_idx"))
    parser.add_argument("--custom-queries", required=True, type=Path)
    parser.add_argument("--dbow2-queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--temporal-exclusion", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = args.cache.resolve()
    output = args.output.resolve()
    images_dir = output / "review_images"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"review output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    index_path = cache / "index.json"
    index = load_json(index_path)
    records = sorted(index.get("records", []), key=lambda item: int(item["sensor_frame_idx"]))
    if index.get("schema_version") != 1 or not records:
        raise ValueError("invalid loop cache index")
    gt_data = np.load(args.gt_poses)
    if gt_data.ndim != 2 or gt_data.shape[1] not in {7, 8}:
        raise ValueError(f"expected GT poses with shape (N,7) or (N,8), got {gt_data.shape}")
    gt_has_timestamps = gt_data.shape[1] == 8
    gt_poses = gt_data[:, 1:] if gt_has_timestamps else gt_data

    by_loop = {int(item["loop_frame_idx"]): item for item in records}
    image_cache: dict[int, np.ndarray] = {}
    frame_ns_by_loop: dict[int, int] = {}
    for loop_idx, metadata in by_loop.items():
        image_cache[loop_idx], frame_ns_by_loop[loop_idx] = data_from_record(cache / metadata["file"])
    raw_union = pair_ids(load_json(args.custom_queries)) | pair_ids(load_json(args.dbow2_queries))
    review: dict[str, dict[str, Any]] = {}
    for current in records:
        current_sensor = int(current["sensor_frame_idx"])
        current_loop = int(current["loop_frame_idx"])
        current_gt_idx = current_sensor if args.gt_index_space == "sensor_frame_idx" else current_loop
        for candidate in records:
            candidate_sensor = int(candidate["sensor_frame_idx"])
            candidate_loop = int(candidate["loop_frame_idx"])
            if current_sensor - candidate_sensor < args.temporal_exclusion:
                continue
            pair_id = f"{current_loop}:{candidate_loop}"
            gt_proxy = "invalid_gt"
            distance = None
            angle = None
            candidate_gt_idx = candidate_sensor if args.gt_index_space == "sensor_frame_idx" else candidate_loop
            timestamp_aligned = False
            if 0 <= current_gt_idx < len(gt_poses) and 0 <= candidate_gt_idx < len(gt_poses):
                if not gt_has_timestamps:
                    timestamp_aligned = True
                elif np.isfinite(gt_data[[current_gt_idx, candidate_gt_idx], 0]).all():
                    timestamp_aligned = (
                        int(gt_data[current_gt_idx, 0]) == frame_ns_by_loop[current_loop]
                        and int(gt_data[candidate_gt_idx, 0]) == frame_ns_by_loop[candidate_loop]
                    )
            if timestamp_aligned:
                try:
                    distance, angle = pair_geometry(gt_poses[current_gt_idx], gt_poses[candidate_gt_idx])
                    if distance <= 1.0 and angle <= 30.0:
                        gt_proxy = "gt_near_strong"
                    elif distance <= 3.0 and angle <= 60.0:
                        gt_proxy = "gt_near_boundary"
                    else:
                        gt_proxy = "gt_far_proxy"
                except ValueError:
                    pass
            if gt_proxy in {"gt_near_strong", "gt_near_boundary"} or pair_id in raw_union:
                review[pair_id] = {
                    "pair_id": pair_id,
                    "query_loop_frame_idx": current_loop,
                    "candidate_loop_frame_idx": candidate_loop,
                    "query_sensor_frame_idx": current_sensor,
                    "candidate_sensor_frame_idx": candidate_sensor,
                    "gt_proxy": gt_proxy,
                    "gt_translation_distance_m": distance,
                    "gt_rotation_difference_deg": angle,
                    "label": None,
                    "review_note": "",
                }

    sealed_proxy = {
        pair: {
            "gt_proxy": item.pop("gt_proxy"),
            "gt_translation_distance_m": item.pop("gt_translation_distance_m"),
            "gt_rotation_difference_deg": item.pop("gt_rotation_difference_deg"),
        }
        for pair, item in review.items()
    }
    sealed_proxy_path = output / "sealed_gt_proxy.json"
    sealed_proxy_path.write_text(
        json.dumps(sealed_proxy, indent=2, allow_nan=False), encoding="utf-8"
    )
    items = list(review.values())
    random.Random(args.seed).shuffle(items)
    record_hashes: list[str] = []
    record_hash_cache: dict[Path, str] = {}

    def cached_sha256(path: Path) -> str:
        if path not in record_hash_cache:
            record_hash_cache[path] = sha256(path)
        return record_hash_cache[path]

    for ordinal, item in enumerate(items):
        current_meta = by_loop[item["query_loop_frame_idx"]]
        candidate_meta = by_loop[item["candidate_loop_frame_idx"]]
        current_path = cache / current_meta["file"]
        candidate_path = cache / candidate_meta["file"]
        current_image = image_cache[item["query_loop_frame_idx"]]
        candidate_image = image_cache[item["candidate_loop_frame_idx"]]
        if current_image.shape[:2] != candidate_image.shape[:2]:
            candidate_image = cv2.resize(candidate_image, (current_image.shape[1], current_image.shape[0]))
        canvas = np.concatenate([candidate_image, current_image], axis=1)
        cv2.putText(
            canvas, f"pair {item['pair_id']}  candidate | query", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
        )
        image_name = f"review_{ordinal:06d}.jpg"
        if not cv2.imwrite(str(images_dir / image_name), canvas):
            raise RuntimeError(f"failed to write {image_name}")
        item["review_image"] = f"review_images/{image_name}"
        record_hashes.extend([
            f"{current_meta['file']}:{cached_sha256(current_path)}",
            f"{candidate_meta['file']}:{cached_sha256(candidate_path)}",
        ])

    pair_digest = hashlib.sha256(
        "\n".join(sorted(review)).encode("utf-8")
    ).hexdigest()
    cache_digest = hashlib.sha256(
        "\n".join(sorted(set(record_hashes))).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "review_protocol_version": PROTOCOL_VERSION,
        "review_seed": args.seed,
        "label_values": ["true", "false", "uncertain"],
        "blind_fields_removed": ["backend", "rank", "score"],
        "pair_match_policy": "exact loop_frame_idx pair",
        "recall_scope": "frozen reviewed overlap evaluation set",
        "temporal_exclusion_sensor_frames": args.temporal_exclusion,
        "gt_index_space": args.gt_index_space,
        "gt_pose_format": "timestamp_ns+translation_xyz+quaternion_xyzw" if gt_has_timestamps else "translation_xyz+quaternion_xyzw",
        "gt_thresholds": {
            "strong": {"max_translation_m": 1.0, "max_rotation_deg": 30.0},
            "boundary": {"max_translation_m": 3.0, "max_rotation_deg": 60.0},
            "gt_far_proxy_is_automatic_false": False,
        },
        "input_sha256": {
            "index": sha256(index_path),
            "gt_poses": sha256(args.gt_poses),
            "custom_raw_queries": sha256(args.custom_queries),
            "dbow2_raw_queries": sha256(args.dbow2_queries),
            "review_pair_ids": pair_digest,
            "reviewed_cache_records_manifest": cache_digest,
            "sealed_gt_proxy": sha256(sealed_proxy_path),
        },
        "items": items,
    }
    target = output / "loop_review_items.json"
    target.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Prepared {len(items)} blinded pairs at {target}")


if __name__ == "__main__":
    main()
