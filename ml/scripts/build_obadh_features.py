#!/usr/bin/env python3
"""Build Obadh feature JSONL for a Dakshina Bengali lexicon split."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obadh_ml.data.dakshina import iter_bn_lexicon
from obadh_ml.features.runner import (
    ensure_feature_binary,
    extract_feature_batch,
    repo_root_from_ml_package,
)

EXAMPLE_SCHEMA = "obadh.ml.example.v0"


def batched(iterator, size: int):
    while batch := list(islice(iterator, size)):
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dakshina-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()

    repo_root = repo_root_from_ml_package()
    binary = args.binary or ensure_feature_binary(repo_root, release=args.release)
    examples = iter_bn_lexicon(args.dakshina_root, args.split)
    if args.limit is not None:
        examples = islice(examples, args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for batch in batched(examples, args.batch_size):
            feature_docs = extract_feature_batch(binary, [example.latin for example in batch])
            for example, features in zip(batch, feature_docs, strict=True):
                row = {
                    "schema": EXAMPLE_SCHEMA,
                    "source": example.source,
                    "latin": example.latin,
                    "target": example.target,
                    "weight": example.weight,
                    "features": features,
                }
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")


if __name__ == "__main__":
    main()
