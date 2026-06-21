from __future__ import annotations

import csv
import dataclasses
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator


@dataclasses.dataclass(frozen=True)
class LexiconRow:
    word: str
    frequency: int


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip().lstrip("\ufeff"))


def is_joiner(ch: str) -> bool:
    return ch in ("\u200c", "\u200d")


def is_bangla_base_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x0985 <= code <= 0x098C
        or 0x098F <= code <= 0x0990
        or 0x0993 <= code <= 0x09A8
        or 0x09AA <= code <= 0x09B0
        or code == 0x09B2
        or 0x09B6 <= code <= 0x09B9
        or code == 0x09CE
        or 0x09DC <= code <= 0x09DD
        or 0x09DF <= code <= 0x09E1
    )


def is_bangla_word_char(ch: str) -> bool:
    code = ord(ch)
    return (
        is_bangla_base_char(ch)
        or 0x0981 <= code <= 0x0983
        or code == 0x09BC
        or 0x09BE <= code <= 0x09C4
        or 0x09C7 <= code <= 0x09C8
        or 0x09CB <= code <= 0x09CD
        or code == 0x09D7
        or 0x09E2 <= code <= 0x09E3
        or code == 0x09FE
    )


def is_bengali_block_word_char(ch: str) -> bool:
    code = ord(ch)
    return is_bangla_word_char(ch) or 0x09F0 <= code <= 0x09F1


def is_bangla_token_char(ch: str) -> bool:
    return is_bengali_block_word_char(ch) or is_joiner(ch)


def is_bangla_lexicon_word(word: str) -> bool:
    has_base = False
    previous_joiner = False
    previous_hasant = False

    for index, ch in enumerate(word):
        if is_joiner(ch):
            if index == 0 or previous_joiner:
                return False
            previous_joiner = True
            continue

        if not is_bangla_word_char(ch):
            return False
        if not has_base and not is_bangla_base_char(ch):
            return False
        if previous_hasant and not is_bangla_base_char(ch):
            return False
        if previous_hasant and ch == "\u09cd":
            return False

        previous_joiner = False
        previous_hasant = ch == "\u09cd"
        has_base = has_base or is_bangla_base_char(ch)

    return has_base and not previous_joiner and not previous_hasant


def iter_bangla_tokens(text: str) -> Iterator[str]:
    token: list[str] = []

    for ch in text:
        if is_bangla_token_char(ch):
            token.append(ch)
            continue

        if token:
            word = normalize("".join(token))
            if is_bangla_lexicon_word(word):
                yield word
            token.clear()

    if token:
        word = normalize("".join(token))
        if is_bangla_lexicon_word(word):
            yield word


def has_repeated_dependent_mark(word: str) -> bool:
    repeated_marks = (
        "\u09be\u09be",
        "\u09bf\u09bf",
        "\u09c0\u09c0",
        "\u09c1\u09c1",
        "\u09c2\u09c2",
        "\u09c7\u09c7",
        "\u09c8\u09c8",
        "\u09cb\u09cb",
        "\u09cc\u09cc",
    )
    return any(mark in word for mark in repeated_marks)


def has_obvious_corruption_shape(word: str) -> bool:
    if has_repeated_dependent_mark(word):
        return True
    if word.startswith(("অঅ", "আআ", "ইই", "উউ", "এএ", "ওও")):
        return True
    suspicious_fragments = (
        "ক্সজ",
        "জন্দ্রিয়া",
        "হাে",
        "ুূ",
        "ূু",
        "িী",
        "ীি",
    )
    return any(fragment in word for fragment in suspicious_fragments)


def read_lexicon_tsv(path: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) > 2:
                continue
            word = normalize(columns[0])
            frequency = 1
            if len(columns) == 2 and columns[1].strip():
                try:
                    frequency = int(columns[1].strip())
                except ValueError:
                    continue
            if word:
                rows[word] = rows.get(word, 0) + frequency
    return rows


def write_lexicon(path: Path, rows: Iterable[LexiconRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: (-row.frequency, row.word))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for row in sorted_rows:
            writer.writerow([row.word, row.frequency])
