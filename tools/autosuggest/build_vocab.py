#!/usr/bin/env python3
"""Build a bounded next-word vocabulary from the autosuggest sentence corpus."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from tools.autosuggest.common import SPECIAL_TOKENS, iter_sentence_rows, save_manifest


def build_vocab(
    corpus_dir: Path,
    output: Path,
    vocab_size: int,
    min_frequency: int,
) -> dict:
    counts: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()

    for row in iter_sentence_rows(corpus_dir):
        counts.update(row.tokens)
        source_rows[row.source] += 1
        source_tokens[row.source] += len(row.tokens)

    reserved = len(SPECIAL_TOKENS)
    if vocab_size <= reserved:
        raise ValueError(f"vocab_size must be greater than {reserved}")

    words = [
        (word, frequency)
        for word, frequency in counts.items()
        if frequency >= min_frequency and word not in SPECIAL_TOKENS
    ]
    words.sort(key=lambda item: (-item[1], item[0]))
    words = words[: vocab_size - reserved]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "token", "frequency"])
        for index, token in enumerate(SPECIAL_TOKENS):
            writer.writerow([index, token, 0])
        for offset, (word, frequency) in enumerate(words, start=reserved):
            writer.writerow([offset, word, frequency])

    total_tokens = sum(counts.values())
    covered_tokens = sum(frequency for _, frequency in words)
    report = {
        "corpus_dir": str(corpus_dir),
        "output": str(output),
        "vocab_size": len(words) + reserved,
        "requested_vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "unique_tokens": len(counts),
        "total_tokens": total_tokens,
        "covered_tokens": covered_tokens,
        "coverage": covered_tokens / total_tokens if total_tokens else 0.0,
        "source_rows": dict(source_rows),
        "source_tokens": dict(source_tokens),
    }
    save_manifest(output.with_suffix(".manifest.json"), **report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/autosuggest/corpus"))
    parser.add_argument("--output", type=Path, default=Path("data/autosuggest/models/ngram/vocab.tsv"))
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--min-frequency", type=int, default=3)
    args = parser.parse_args()

    print(build_vocab(args.corpus_dir, args.output, args.vocab_size, args.min_frequency))


if __name__ == "__main__":
    main()
