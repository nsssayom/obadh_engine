#!/usr/bin/env python3
"""Encode the autosuggest corpus into binary arrays for random-access training."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from .common import BOS_ID, UNK_ID, iter_sentence_rows, load_vocab, save_manifest


TOKEN_STRUCT = struct.Struct("<I")
OFFSET_STRUCT = struct.Struct("<QQ")


def prepare_dataset(
    corpus_dir: Path,
    vocab_path: Path,
    output_dir: Path,
    min_encoded_tokens: int,
    max_unk_ratio: float,
) -> dict:
    _, vocab = load_vocab(vocab_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / "tokens.u32"
    offsets_path = output_dir / "sentences.u64"

    sentence_count = 0
    token_count = 0
    target_positions = 0
    skipped_short = 0
    skipped_unk_heavy = 0
    source_sentences: dict[str, int] = {}

    with tokens_path.open("wb") as tokens_handle, offsets_path.open("wb") as offsets_handle:
        for row in iter_sentence_rows(corpus_dir):
            encoded = [BOS_ID]
            unk_count = 0
            for token in row.tokens:
                token_id = vocab.get(token, UNK_ID)
                if token_id == UNK_ID:
                    unk_count += 1
                encoded.append(token_id)

            if len(encoded) < min_encoded_tokens:
                skipped_short += 1
                continue
            if row.tokens and unk_count / len(row.tokens) > max_unk_ratio:
                skipped_unk_heavy += 1
                continue

            start = token_count
            for token_id in encoded:
                tokens_handle.write(TOKEN_STRUCT.pack(token_id))
            token_count += len(encoded)
            offsets_handle.write(OFFSET_STRUCT.pack(start, token_count))
            sentence_count += 1
            target_positions += len(encoded) - 1
            source_sentences[row.source] = source_sentences.get(row.source, 0) + 1

    report = {
        "corpus_dir": str(corpus_dir),
        "vocab_path": str(vocab_path),
        "output_dir": str(output_dir),
        "tokens_path": str(tokens_path),
        "offsets_path": str(offsets_path),
        "vocab_size": len(vocab),
        "sentence_count": sentence_count,
        "token_count": token_count,
        "target_positions": target_positions,
        "min_encoded_tokens": min_encoded_tokens,
        "max_unk_ratio": max_unk_ratio,
        "skipped_short": skipped_short,
        "skipped_unk_heavy": skipped_unk_heavy,
        "source_sentences": source_sentences,
    }
    save_manifest(output_dir / "manifest.json", **report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/autosuggest/corpus"))
    parser.add_argument("--vocab", type=Path, default=Path("data/autosuggest/models/neural/vocab.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/autosuggest/models/neural/dataset"))
    parser.add_argument("--min-encoded-tokens", type=int, default=3)
    parser.add_argument("--max-unk-ratio", type=float, default=0.5)
    args = parser.parse_args()

    print(
        prepare_dataset(
            args.corpus_dir,
            args.vocab,
            args.output_dir,
            args.min_encoded_tokens,
            args.max_unk_ratio,
        )
    )


if __name__ == "__main__":
    main()
