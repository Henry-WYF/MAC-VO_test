from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Module.LoopClosure import BinaryVocabulary, LoopFrameRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase-A hierarchical ORB vocabulary from independent loop caches.")
    parser.add_argument("cache", nargs="+", type=Path, help="Loop-cache directories containing index.json files.")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz vocabulary file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-descriptors", type=int, default=1_000_000)
    parser.add_argument("--max-iterations", type=int, default=30)
    return parser.parse_args()


def load_documents(cache_dirs: list[Path]) -> list:
    documents = []
    for cache_dir in cache_dirs:
        with open(cache_dir / "index.json", "r", encoding="utf-8") as file:
            index = json.load(file)
        for metadata in index["records"]:
            record = LoopFrameRecord.load(cache_dir / metadata["file"])
            if len(record.orb_descriptors) > 0:
                documents.append(record.orb_descriptors.cpu().numpy())
    return documents


def main() -> None:
    args = parse_args()
    documents = load_documents(args.cache)
    vocabulary = BinaryVocabulary.train(
        documents,
        branch=8,
        depth=4,
        seed=args.seed,
        max_iterations=args.max_iterations,
        max_descriptors=args.max_descriptors,
    )
    vocabulary.save(args.output)
    print(json.dumps({
        "documents": len(documents),
        "words": vocabulary.num_words,
        "checksum": vocabulary.checksum(),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
