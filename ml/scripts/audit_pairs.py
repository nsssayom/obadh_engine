#!/usr/bin/env python3
"""Audit candidate Bengali transliteration pairs before corpus admission."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obadh_ml.data.audit import (
    AuditConfig,
    PairRecord,
    audit_records,
    result_to_json,
    summary_to_json,
    write_audit_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=["csv", "tsv", "jsonl"], required=True)
    parser.add_argument("--latin-column", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--mode", choices=["word", "sentence"], default="word")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path)
    parser.add_argument("--max-latin-chars", type=int, default=64)
    parser.add_argument("--max-target-chars", type=int, default=64)
    parser.add_argument("--max-length-ratio", type=float, default=4.0)
    parser.add_argument("--min-bengali-letter-ratio", type=float, default=0.65)
    parser.add_argument("--allow-target-ascii", action="store_true")
    parser.add_argument("--allow-sentence-punctuation", action="store_true")
    args = parser.parse_args()

    config = AuditConfig(
        mode=args.mode,
        max_latin_chars=args.max_latin_chars,
        max_target_chars=args.max_target_chars,
        max_length_ratio=args.max_length_ratio,
        min_bengali_letter_ratio=args.min_bengali_letter_ratio,
        max_target_ascii_alpha_ratio=1.0 if args.allow_target_ascii else 0.0,
        allow_sentence_punctuation=args.allow_sentence_punctuation,
    )
    records = list(read_records(args))
    results, summary = audit_records(records, config)

    write_audit_report(args.report, summary, results)
    if args.accepted_output:
        write_accepted_jsonl(args.accepted_output, results)

    print(json.dumps(summary_to_json(summary), ensure_ascii=False, separators=(",", ":")))


def read_records(args: argparse.Namespace) -> Iterator[PairRecord]:
    if args.format == "jsonl":
        yield from read_jsonl(args.input, args.source_id, args.latin_column, args.target_column)
    else:
        delimiter = "\t" if args.format == "tsv" else ","
        yield from read_delimited(
            args.input,
            args.source_id,
            args.latin_column,
            args.target_column,
            delimiter,
        )


def read_jsonl(
    path: Path,
    source_id: str,
    latin_column: str,
    target_column: str,
) -> Iterator[PairRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            yield make_record(source_id, str(line_number), row, latin_column, target_column)


def read_delimited(
    path: Path,
    source_id: str,
    latin_column: str,
    target_column: str,
    delimiter: str,
) -> Iterator[PairRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for line_number, row in enumerate(reader, start=2):
            yield make_record(source_id, str(line_number), row, latin_column, target_column)


def make_record(
    source_id: str,
    row_id: str,
    row: dict[str, Any],
    latin_column: str,
    target_column: str,
) -> PairRecord:
    try:
        latin = str(row[latin_column])
        target = str(row[target_column])
    except KeyError as error:
        raise KeyError(f"missing column {error.args[0]!r}") from error

    metadata = {
        key: value
        for key, value in row.items()
        if key not in {latin_column, target_column}
    }
    return PairRecord(
        source_id=source_id,
        row_id=row_id,
        latin=latin,
        target=target,
        metadata=metadata,
    )


def write_accepted_jsonl(path: Path, results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            if result.accepted:
                handle.write(
                    json.dumps(
                        result_to_json(result),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")


if __name__ == "__main__":
    main()
