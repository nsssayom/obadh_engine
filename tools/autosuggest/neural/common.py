#!/usr/bin/env python3
"""Shared utilities for the neural autosuggest pipeline."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


PAD = "<pad>"
BOS = "<bos>"
UNK = "<unk>"
SPECIAL_TOKENS = (PAD, BOS, UNK)
PAD_ID = 0
BOS_ID = 1
UNK_ID = 2


@dataclass(frozen=True)
class CorpusRow:
    source: str
    document_id: str
    sentence_id: int
    tokens: list[str]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sentence_paths(corpus_dir: Path) -> list[Path]:
    return sorted((corpus_dir / "sentences").glob("*.tsv.gz"))


def iter_sentence_rows(corpus_dir: Path) -> Iterator[CorpusRow]:
    for path in sentence_paths(corpus_dir):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                tokens = row["tokens"].split(" ") if row.get("tokens") else []
                if not tokens:
                    continue
                yield CorpusRow(
                    source=row["source"],
                    document_id=row["document_id"],
                    sentence_id=int(row["sentence_id"]),
                    tokens=tokens,
                )


def load_vocab(path: Path) -> tuple[list[str], dict[str, int]]:
    words: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            words.append(row["token"])

    ids = {word: index for index, word in enumerate(words)}
    for index, token in enumerate(SPECIAL_TOKENS):
        if ids.get(token) != index:
            raise ValueError(f"vocab must reserve {token} at id {index}")
    return words, ids


def save_manifest(path: Path, **values: object) -> None:
    serializable = {}
    for key, value in values.items():
        if hasattr(value, "__dataclass_fields__"):
            serializable[key] = asdict(value)
        else:
            serializable[key] = value
    write_json(path, serializable)
