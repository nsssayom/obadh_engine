#!/usr/bin/env python3
"""Curate Bangla autocorrect lexicon sources before FST merge.

Raw corpus TSVs remain intact. This script emits production-facing curated
sources plus a quarantine report. The policy is intentionally conservative:
cross-source evidence protects rare words, while unsupported low-frequency rows
must have a clean Bengali word shape to survive.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

from bangla_lexicon_utils import (
    LexiconRow,
    has_obvious_corruption_shape,
    is_bangla_lexicon_word,
    read_lexicon_tsv,
    write_lexicon,
)


@dataclasses.dataclass(frozen=True)
class CuratedRow:
    word: str
    frequency: int
    source: str
    decision: str
    reason: str


def has_clean_semantic_shape(word: str, source: str, frequency: int) -> bool:
    if has_obvious_corruption_shape(word):
        return False
    if "\u200c" in word or "\u200d" in word:
        return False

    length = len(word)
    if source == "epub":
        return (frequency == 2 and length <= 14) or (frequency == 1 and length <= 10)
    if source in {"wiki", "news"}:
        return frequency == 2 and length <= 12
    return False


def curate_source(
    source_name: str,
    rows: dict[str, int],
    other_sources: set[str],
    loanwords: set[str],
) -> tuple[list[LexiconRow], list[CuratedRow]]:
    curated: list[LexiconRow] = []
    quarantine: list[CuratedRow] = []

    for word, frequency in rows.items():
        if not is_bangla_lexicon_word(word):
            quarantine.append(
                CuratedRow(word, frequency, source_name, "drop", "invalid_bangla_token")
            )
            continue
        if has_obvious_corruption_shape(word):
            quarantine.append(
                CuratedRow(word, frequency, source_name, "drop", "obvious_corruption_shape")
            )
            continue

        if frequency > 2:
            curated.append(LexiconRow(word, frequency))
            continue
        if word in other_sources:
            curated.append(LexiconRow(word, frequency))
            continue
        if word in loanwords:
            curated.append(LexiconRow(word, frequency))
            continue
        if has_clean_semantic_shape(word, source_name, frequency):
            curated.append(LexiconRow(word, frequency))
            continue

        quarantine.append(
            CuratedRow(word, frequency, source_name, "quarantine", "unsupported_low_frequency")
        )

    return curated, quarantine


def write_quarantine(path: Path, rows: list[CuratedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["word", "frequency", "source", "decision", "reason"])
        for row in sorted(rows, key=lambda row: (row.source, row.reason, row.frequency, row.word)):
            writer.writerow([row.word, row.frequency, row.source, row.decision, row.reason])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epub", required=True, type=Path)
    parser.add_argument("--wiki", required=True, type=Path)
    parser.add_argument("--news", required=True, type=Path)
    parser.add_argument("--loan", required=True, type=Path)
    parser.add_argument("--epub-output", required=True, type=Path)
    parser.add_argument("--wiki-output", required=True, type=Path)
    parser.add_argument("--news-output", required=True, type=Path)
    parser.add_argument("--quarantine-output", required=True, type=Path)
    args = parser.parse_args()

    epub = read_lexicon_tsv(args.epub)
    wiki = read_lexicon_tsv(args.wiki)
    news = read_lexicon_tsv(args.news)
    loan = read_lexicon_tsv(args.loan)

    source_sets = {
        "epub": set(epub),
        "wiki": set(wiki),
        "news": set(news),
    }

    epub_curated, epub_quarantine = curate_source(
        "epub", epub, source_sets["wiki"] | source_sets["news"], set(loan)
    )
    wiki_curated, wiki_quarantine = curate_source(
        "wiki", wiki, source_sets["epub"] | source_sets["news"], set(loan)
    )
    news_curated, news_quarantine = curate_source(
        "news", news, source_sets["epub"] | source_sets["wiki"], set(loan)
    )

    write_lexicon(args.epub_output, epub_curated)
    write_lexicon(args.wiki_output, wiki_curated)
    write_lexicon(args.news_output, news_curated)
    write_quarantine(
        args.quarantine_output,
        epub_quarantine + wiki_quarantine + news_quarantine,
    )

    print(
        {
            "epub_input": len(epub),
            "epub_output": len(epub_curated),
            "wiki_input": len(wiki),
            "wiki_output": len(wiki_curated),
            "news_input": len(news),
            "news_output": len(news_curated),
            "quarantine_rows": len(epub_quarantine)
            + len(wiki_quarantine)
            + len(news_quarantine),
        }
    )


if __name__ == "__main__":
    main()
