#!/usr/bin/env python3
"""Extract a Bangla word-frequency TSV from the Furcifer newspaper JSON array."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any, Iterable

from bangla_lexicon_utils import LexiconRow, iter_bangla_tokens, write_lexicon


DEFAULT_FIELDS = ("title", "content", "category_bn", "tag")


def require_ijson():
    try:
        import ijson  # type: ignore
    except ImportError as error:
        raise SystemExit(
            "extract_news_json_lexicon.py requires ijson for streaming large JSON arrays"
        ) from error
    return ijson


def iter_text_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                yield item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--field", action="append", dest="fields")
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args()

    fields = tuple(args.fields) if args.fields else DEFAULT_FIELDS
    ijson = require_ijson()
    counts: collections.Counter[str] = collections.Counter()
    records = 0
    text_values = 0
    token_count = 0

    with args.input.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            records += 1
            if not isinstance(item, dict):
                continue

            for field in fields:
                for text in iter_text_values(item.get(field)):
                    text_values += 1
                    for token in iter_bangla_tokens(text):
                        counts[token] += 1
                        token_count += 1

            if args.max_records is not None and records >= args.max_records:
                break
            if args.progress_every > 0 and records % args.progress_every == 0:
                print(
                    {
                        "records": records,
                        "unique_words": len(counts),
                        "tokens": token_count,
                    },
                    file=sys.stderr,
                )

    rows = [
        LexiconRow(word, frequency)
        for word, frequency in counts.items()
        if frequency >= args.min_frequency
    ]
    write_lexicon(args.output, rows)

    print(
        {
            "input": str(args.input),
            "output": str(args.output),
            "records": records,
            "text_values": text_values,
            "tokens": token_count,
            "unique_words": len(counts),
            "emitted_rows": len(rows),
            "min_frequency": args.min_frequency,
            "fields": fields,
        }
    )


if __name__ == "__main__":
    main()
