"""Readers for the Google Dakshina transliteration dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DAKSHINA_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"


@dataclass(frozen=True)
class TransliterationExample:
    latin: str
    target: str
    weight: int
    source: str


def bn_lexicon_path(root: Path, split: str) -> Path:
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be one of: train, dev, test")
    return root / "bn" / "lexicons" / f"bn.translit.sampled.{split}.tsv"


def read_lexicon_tsv(path: Path, *, source: str | None = None) -> Iterator[TransliterationExample]:
    row_source = source or path.name
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            columns = line.split("\t")
            if len(columns) < 2:
                raise ValueError(f"{path}:{line_number}: expected at least 2 TSV columns")

            target, latin = columns[0], columns[1]
            weight = int(columns[2]) if len(columns) >= 3 and columns[2] else 1
            yield TransliterationExample(
                latin=latin,
                target=target,
                weight=weight,
                source=row_source,
            )


def iter_bn_lexicon(root: Path, split: str) -> Iterator[TransliterationExample]:
    path = bn_lexicon_path(root, split)
    yield from read_lexicon_tsv(path, source=f"dakshina.bn.lexicon.{split}")
