#!/usr/bin/env python3
"""Build Obadh's compact next-word n-gram autosuggest artifact."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
import struct
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from tools.autosuggest.common import BOS_ID, PAD_ID, UNK_ID, load_vocab, sentence_paths


MAGIC = b"OBAUTOSUGLM_V1\0\0"
VERSION = 1
U32 = struct.Struct("<I")
I32 = struct.Struct("<i")


class MemoryCounts:
    def __init__(self) -> None:
        self.unigrams: Counter[int] = Counter()
        self.bigrams: dict[int, Counter[int]] = defaultdict(Counter)
        self.trigrams: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)

    def observe(self, encoded: list[int]) -> None:
        for index in range(1, len(encoded)):
            target = encoded[index]
            if not is_target_id(target):
                continue
            self.unigrams[target] += 1
            previous = encoded[index - 1]
            if is_context_id(previous):
                self.bigrams[previous][target] += 1
            if index >= 2:
                previous2 = encoded[index - 2]
                if is_context_id(previous2) and is_context_id(previous):
                    self.trigrams[(previous2, previous)][target] += 1

    def finalize(self) -> None:
        return

    def rows(
        self,
        max_candidates_per_prefix: int,
        min_count: int,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int, list[tuple[int, int]]]], list[tuple[int, int, int, list[tuple[int, int]]]]]:
        unigrams = sorted(
            self.unigrams.items(),
            key=lambda item: (-item[1], item[0]),
        )
        bigrams = [
            (prefix, 0, top_candidates(counter, max_candidates_per_prefix, min_count))
            for prefix, counter in sorted(self.bigrams.items())
        ]
        trigrams = [
            (prefix1, prefix2, 0, top_candidates(counter, max_candidates_per_prefix, min_count))
            for (prefix1, prefix2), counter in sorted(self.trigrams.items())
        ]
        return (
            unigrams,
            [(prefix, total, candidates) for prefix, total, candidates in bigrams if candidates],
            [
                (prefix1, prefix2, total, candidates)
                for prefix1, prefix2, total, candidates in trigrams
                if candidates
            ],
        )


class SqliteCounts:
    def __init__(self, path: Path, batch_size: int, reset: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if reset and path.exists():
            path.unlink()
        self.path = path
        self.batch_size = batch_size
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        if not reset:
            self._verify_existing()
            self.unigram_batch = Counter()
            self.bigram_batch = Counter()
            self.trigram_batch = Counter()
            self.pending = 0
            return
        self.connection.execute(
            "CREATE TABLE unigrams(token INTEGER PRIMARY KEY, count INTEGER NOT NULL)"
        )
        self.connection.execute(
            """
            CREATE TABLE bigrams(
              prefix INTEGER NOT NULL,
              token INTEGER NOT NULL,
              count INTEGER NOT NULL,
              PRIMARY KEY(prefix, token)
            ) WITHOUT ROWID
            """
        )
        self.connection.execute(
            """
            CREATE TABLE trigrams(
              prefix1 INTEGER NOT NULL,
              prefix2 INTEGER NOT NULL,
              token INTEGER NOT NULL,
              count INTEGER NOT NULL,
              PRIMARY KEY(prefix1, prefix2, token)
            ) WITHOUT ROWID
            """
        )
        self.unigram_batch: Counter[int] = Counter()
        self.bigram_batch: Counter[tuple[int, int]] = Counter()
        self.trigram_batch: Counter[tuple[int, int, int]] = Counter()
        self.pending = 0

    def _verify_existing(self) -> None:
        required = {"unigrams", "bigrams", "trigrams"}
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        existing = {row[0] for row in rows}
        missing = required - existing
        if missing:
            raise ValueError(f"existing SQLite count DB is missing tables: {sorted(missing)}")

    def observe(self, encoded: list[int]) -> None:
        for index in range(1, len(encoded)):
            target = encoded[index]
            if not is_target_id(target):
                continue
            self.unigram_batch[target] += 1
            self.pending += 1

            previous = encoded[index - 1]
            if is_context_id(previous):
                self.bigram_batch[(previous, target)] += 1
                self.pending += 1
            if index >= 2:
                previous2 = encoded[index - 2]
                if is_context_id(previous2) and is_context_id(previous):
                    self.trigram_batch[(previous2, previous, target)] += 1
                    self.pending += 1

            if self.pending >= self.batch_size:
                self.flush()

    def flush(self) -> None:
        if self.unigram_batch:
            self.connection.executemany(
                """
                INSERT INTO unigrams(token, count) VALUES(?, ?)
                ON CONFLICT(token) DO UPDATE SET count = count + excluded.count
                """,
                self.unigram_batch.items(),
            )
            self.unigram_batch.clear()
        if self.bigram_batch:
            self.connection.executemany(
                """
                INSERT INTO bigrams(prefix, token, count) VALUES(?, ?, ?)
                ON CONFLICT(prefix, token) DO UPDATE SET count = count + excluded.count
                """,
                ((prefix, token, count) for (prefix, token), count in self.bigram_batch.items()),
            )
            self.bigram_batch.clear()
        if self.trigram_batch:
            self.connection.executemany(
                """
                INSERT INTO trigrams(prefix1, prefix2, token, count) VALUES(?, ?, ?, ?)
                ON CONFLICT(prefix1, prefix2, token) DO UPDATE SET count = count + excluded.count
                """,
                (
                    (prefix1, prefix2, token, count)
                    for (prefix1, prefix2, token), count in self.trigram_batch.items()
                ),
            )
            self.trigram_batch.clear()
        self.connection.commit()
        self.pending = 0

    def finalize(self) -> None:
        self.flush()
        self.connection.execute("CREATE INDEX IF NOT EXISTS bigram_rank ON bigrams(prefix, count DESC, token)")
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS trigram_rank ON trigrams(prefix1, prefix2, count DESC, token)"
        )
        self.connection.commit()

    def rows(
        self,
        max_candidates_per_prefix: int,
        min_count: int,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int, list[tuple[int, int]]]], list[tuple[int, int, int, list[tuple[int, int]]]]]:
        unigrams = list(
            self.connection.execute(
                "SELECT token, count FROM unigrams ORDER BY count DESC, token ASC"
            )
        )
        bigrams = grouped_bigram_rows(
            self.connection.execute(
                """
                SELECT prefix, token, count
                FROM (
                  SELECT prefix, token, count,
                         ROW_NUMBER() OVER (
                           PARTITION BY prefix ORDER BY count DESC, token ASC
                         ) AS rank
                  FROM bigrams
                  WHERE count >= ?
                )
                WHERE rank <= ?
                ORDER BY prefix ASC, rank ASC
                """,
                (min_count, max_candidates_per_prefix),
            )
        )
        trigrams = grouped_trigram_rows(
            self.connection.execute(
                """
                SELECT prefix1, prefix2, token, count
                FROM (
                  SELECT prefix1, prefix2, token, count,
                         ROW_NUMBER() OVER (
                           PARTITION BY prefix1, prefix2 ORDER BY count DESC, token ASC
                         ) AS rank
                  FROM trigrams
                  WHERE count >= ?
                )
                WHERE rank <= ?
                ORDER BY prefix1 ASC, prefix2 ASC, rank ASC
                """,
                (min_count, max_candidates_per_prefix),
            )
        )
        return unigrams, bigrams, trigrams


def build_ngram_lm(
    corpus_dir: Path,
    vocab_path: Path,
    output: Path,
    backend: str,
    sqlite_path: Path,
    sources: set[str] | None,
    max_sentences: int | None,
    max_sentences_per_source: int | None,
    reuse_sqlite: bool,
    log_every_sentences: int,
    max_candidates_per_prefix: int,
    unigram_size: int,
    min_count: int,
    batch_size: int,
) -> dict:
    if reuse_sqlite and backend != "sqlite":
        raise ValueError("--reuse-sqlite requires --backend sqlite")
    words, vocab = load_vocab(vocab_path)
    counts = (
        MemoryCounts()
        if backend == "memory"
        else SqliteCounts(sqlite_path, batch_size, reset=not reuse_sqlite)
    )
    observed_sentences = 0
    observed_tokens = 0
    source_sentences: Counter[str] = Counter()
    started_at = time.monotonic()

    if not reuse_sqlite:
        for source, tokens in iter_limited_sentence_tokens(
            corpus_dir,
            sources=sources,
            max_sentences=max_sentences,
            max_sentences_per_source=max_sentences_per_source,
        ):
            encoded = encode_tokens(tokens, vocab)
            if len(encoded) < 2:
                continue
            counts.observe(encoded)
            observed_sentences += 1
            observed_tokens += len(encoded) - 1
            source_sentences[source] += 1
            if log_every_sentences > 0 and observed_sentences % log_every_sentences == 0:
                elapsed = time.monotonic() - started_at
                print(
                    json.dumps(
                        {
                            "event": "autosuggest_ngram_build_progress",
                            "sentences": observed_sentences,
                            "tokens": observed_tokens,
                            "elapsed_seconds": round(elapsed, 3),
                            "source_sentences": dict(source_sentences),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    counts.finalize()

    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(counts, SqliteCounts):
        export_report = encode_sqlite_artifact(
            words=words,
            counts=counts,
            output=output,
            max_candidates_per_prefix=max_candidates_per_prefix,
            min_count=min_count,
            unigram_size=unigram_size,
        )
    else:
        unigrams, bigrams, trigrams = counts.rows(max_candidates_per_prefix, min_count)
        unigrams = unigrams[:unigram_size]
        artifact = encode_artifact(words, unigrams, bigrams, trigrams)
        output.write_bytes(artifact)
        export_report = {
            "unigram_count": len(unigrams),
            "bigram_rows": len(bigrams),
            "trigram_rows": len(trigrams),
            "candidate_rows": sum(len(row[2]) for row in bigrams)
            + sum(len(row[3]) for row in trigrams),
            "artifact_bytes": len(artifact),
        }

    report = {
        "artifact": "obadh-autosuggest-ngram",
        "version": VERSION,
        "format": "bounded trigram/bigram/unigram binary",
        "corpus_dir": str(corpus_dir),
        "vocab_path": str(vocab_path),
        "output": str(output),
        "backend": backend,
        "sqlite_path": str(sqlite_path) if backend == "sqlite" else None,
        "reuse_sqlite": reuse_sqlite,
        "sources": sorted(sources) if sources else None,
        "max_sentences": max_sentences,
        "max_sentences_per_source": max_sentences_per_source,
        "observed_sentences": observed_sentences,
        "observed_tokens": observed_tokens,
        "source_sentences": dict(source_sentences),
        "vocab_size": len(words),
        "unigram_count": export_report["unigram_count"],
        "bigram_rows": export_report["bigram_rows"],
        "trigram_rows": export_report["trigram_rows"],
        "candidate_rows": export_report["candidate_rows"],
        "max_candidates_per_prefix": max_candidates_per_prefix,
        "min_count": min_count,
        "artifact_bytes": export_report["artifact_bytes"],
    }
    manifest_path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def iter_limited_sentence_tokens(
    corpus_dir: Path,
    sources: set[str] | None,
    max_sentences: int | None,
    max_sentences_per_source: int | None,
) -> Iterator[tuple[str, list[str]]]:
    emitted_total = 0
    emitted_by_source: Counter[str] = Counter()
    pending_sources = set(sources) if sources else None

    for path in sentence_paths(corpus_dir):
        source_hint = path.stem.split(".")[0]
        if sources is not None and source_hint not in sources:
            continue
        if (
            max_sentences_per_source is not None
            and emitted_by_source[source_hint] >= max_sentences_per_source
        ):
            continue

        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                source = row["source"]
                if sources is not None and source not in sources:
                    continue
                if (
                    max_sentences_per_source is not None
                    and emitted_by_source[source] >= max_sentences_per_source
                ):
                    if pending_sources is not None:
                        pending_sources.discard(source)
                    break

                tokens = row["tokens"].split(" ") if row.get("tokens") else []
                if not tokens:
                    continue

                yield source, tokens
                emitted_total += 1
                emitted_by_source[source] += 1

                if max_sentences is not None and emitted_total >= max_sentences:
                    return
                if (
                    pending_sources is not None
                    and max_sentences_per_source is not None
                    and emitted_by_source[source] >= max_sentences_per_source
                ):
                    pending_sources.discard(source)
                    if not pending_sources:
                        return
                    break


def encode_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    return [BOS_ID, *(vocab.get(token, UNK_ID) for token in tokens)]


def is_target_id(token_id: int) -> bool:
    return token_id > UNK_ID


def is_context_id(token_id: int) -> bool:
    return token_id != PAD_ID and token_id != UNK_ID


def top_candidates(counter: Counter[int], limit: int, min_count: int) -> list[tuple[int, int]]:
    return [
        (token, count)
        for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_count
    ][:limit]


def grouped_bigram_rows(rows: Iterable[tuple[int, int, int]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    grouped: list[tuple[int, int, list[tuple[int, int]]]] = []
    current_prefix: int | None = None
    candidates: list[tuple[int, int]] = []
    for prefix, token, count in rows:
        if current_prefix is not None and prefix != current_prefix:
            grouped.append((current_prefix, 0, candidates))
            candidates = []
        current_prefix = prefix
        candidates.append((token, count))
    if current_prefix is not None:
        grouped.append((current_prefix, 0, candidates))
    return grouped


def grouped_trigram_rows(
    rows: Iterable[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, list[tuple[int, int]]]]:
    grouped: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    current_prefix: tuple[int, int] | None = None
    candidates: list[tuple[int, int]] = []
    for prefix1, prefix2, token, count in rows:
        prefix = (prefix1, prefix2)
        if current_prefix is not None and prefix != current_prefix:
            grouped.append((current_prefix[0], current_prefix[1], 0, candidates))
            candidates = []
        current_prefix = prefix
        candidates.append((token, count))
    if current_prefix is not None:
        grouped.append((current_prefix[0], current_prefix[1], 0, candidates))
    return grouped


def encode_artifact(
    words: list[str],
    unigrams: list[tuple[int, int]],
    bigrams: list[tuple[int, int, list[tuple[int, int]]]],
    trigrams: list[tuple[int, int, int, list[tuple[int, int]]]],
) -> bytes:
    token_bytes = bytearray()
    id_records: list[tuple[int, int]] = []
    for word in words:
        encoded = word.encode("utf-8")
        id_records.append((len(token_bytes), len(encoded)))
        token_bytes.extend(encoded)

    token_index = [
        (token_bytes[offset : offset + length], offset, length, token_id)
        for token_id, (offset, length) in enumerate(id_records)
    ]
    token_index.sort(key=lambda item: bytes(item[0]))

    candidate_records: list[tuple[int, int, int]] = []
    bigram_rows: list[tuple[int, int, int]] = []
    for prefix, _, candidates in bigrams:
        start = len(candidate_records)
        for token_id, count in candidates:
            candidate_records.append((token_id, count, score_from_count(count)))
        bigram_rows.append((prefix, start, len(candidates)))

    trigram_rows: list[tuple[int, int, int, int]] = []
    for prefix1, prefix2, _, candidates in trigrams:
        start = len(candidate_records)
        for token_id, count in candidates:
            candidate_records.append((token_id, count, score_from_count(count)))
        trigram_rows.append((prefix1, prefix2, start, len(candidates)))

    artifact = bytearray()
    artifact.extend(MAGIC)
    write_u32(artifact, VERSION)
    write_u32(artifact, len(words))
    write_u32(artifact, len(token_index))
    write_u32(artifact, len(unigrams))
    write_u32(artifact, len(bigram_rows))
    write_u32(artifact, len(trigram_rows))
    write_u32(artifact, len(candidate_records))
    write_u32(artifact, len(token_bytes))
    write_u32(artifact, 0)

    for offset, length in id_records:
        write_u32(artifact, offset)
        write_u32(artifact, length)
    for _, offset, length, token_id in token_index:
        write_u32(artifact, offset)
        write_u32(artifact, length)
        write_u32(artifact, token_id)
    for token_id, count in unigrams:
        write_u32(artifact, token_id)
        write_u32(artifact, count)
        write_i32(artifact, score_from_count(count))
    for prefix, start, length in bigram_rows:
        write_u32(artifact, prefix)
        write_u32(artifact, start)
        write_u32(artifact, length)
    for prefix1, prefix2, start, length in trigram_rows:
        write_u32(artifact, prefix1)
        write_u32(artifact, prefix2)
        write_u32(artifact, start)
        write_u32(artifact, length)
    for token_id, count, score in candidate_records:
        write_u32(artifact, token_id)
        write_u32(artifact, count)
        write_i32(artifact, score)
    artifact.extend(token_bytes)
    return bytes(artifact)


def encode_sqlite_artifact(
    words: list[str],
    counts: SqliteCounts,
    output: Path,
    max_candidates_per_prefix: int,
    min_count: int,
    unigram_size: int,
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"{output.name}.", dir=output.parent) as temp_dir:
        temp_path = Path(temp_dir)
        id_tokens_path = temp_path / "id_tokens.bin"
        token_index_path = temp_path / "token_index.bin"
        unigrams_path = temp_path / "unigrams.bin"
        bigram_rows_path = temp_path / "bigram_rows.bin"
        trigram_rows_path = temp_path / "trigram_rows.bin"
        candidates_path = temp_path / "candidates.bin"
        token_bytes_path = temp_path / "token_bytes.bin"

        token_bytes_len = write_token_sections(
            words,
            id_tokens_path,
            token_index_path,
            token_bytes_path,
        )
        unigram_count = write_unigram_section(
            counts.connection,
            unigrams_path,
            unigram_size,
        )
        candidate_count = 0
        bigram_rows, candidate_count = write_bigram_sections(
            counts.connection,
            bigram_rows_path,
            candidates_path,
            max_candidates_per_prefix,
            min_count,
            candidate_count,
        )
        trigram_rows, candidate_count = write_trigram_sections(
            counts.connection,
            trigram_rows_path,
            candidates_path,
            max_candidates_per_prefix,
            min_count,
            candidate_count,
        )

        with output.open("wb") as handle:
            write_header(
                handle,
                vocab_size=len(words),
                token_index_count=len(words),
                unigram_count=unigram_count,
                bigram_rows=bigram_rows,
                trigram_rows=trigram_rows,
                candidate_count=candidate_count,
                token_bytes_len=token_bytes_len,
            )
            append_file(handle, id_tokens_path)
            append_file(handle, token_index_path)
            append_file(handle, unigrams_path)
            append_file(handle, bigram_rows_path)
            append_file(handle, trigram_rows_path)
            append_file(handle, candidates_path)
            append_file(handle, token_bytes_path)

    return {
        "unigram_count": unigram_count,
        "bigram_rows": bigram_rows,
        "trigram_rows": trigram_rows,
        "candidate_rows": candidate_count,
        "artifact_bytes": output.stat().st_size,
    }


def write_token_sections(
    words: list[str],
    id_tokens_path: Path,
    token_index_path: Path,
    token_bytes_path: Path,
) -> int:
    id_records: list[tuple[int, int]] = []
    token_index: list[tuple[bytes, int, int, int]] = []
    token_bytes_len = 0

    with token_bytes_path.open("wb") as token_bytes_handle:
        for token_id, word in enumerate(words):
            encoded = word.encode("utf-8")
            offset = token_bytes_len
            length = len(encoded)
            id_records.append((offset, length))
            token_index.append((encoded, offset, length, token_id))
            token_bytes_handle.write(encoded)
            token_bytes_len += length

    with id_tokens_path.open("wb") as id_tokens_handle:
        for offset, length in id_records:
            write_u32_to_file(id_tokens_handle, offset)
            write_u32_to_file(id_tokens_handle, length)

    token_index.sort(key=lambda item: item[0])
    with token_index_path.open("wb") as token_index_handle:
        for _, offset, length, token_id in token_index:
            write_u32_to_file(token_index_handle, offset)
            write_u32_to_file(token_index_handle, length)
            write_u32_to_file(token_index_handle, token_id)

    return token_bytes_len


def write_unigram_section(
    connection: sqlite3.Connection,
    output: Path,
    unigram_size: int,
) -> int:
    count = 0
    with output.open("wb") as handle:
        for token_id, frequency in connection.execute(
            "SELECT token, count FROM unigrams ORDER BY count DESC, token ASC LIMIT ?",
            (unigram_size,),
        ):
            write_candidate_record(handle, token_id, frequency)
            count += 1
    return count


def write_bigram_sections(
    connection: sqlite3.Connection,
    row_output: Path,
    candidate_output: Path,
    max_candidates_per_prefix: int,
    min_count: int,
    candidate_start: int,
) -> tuple[int, int]:
    query = """
        SELECT prefix, token, count
        FROM (
          SELECT prefix, token, count,
                 ROW_NUMBER() OVER (
                   PARTITION BY prefix ORDER BY count DESC, token ASC
                 ) AS rank
          FROM bigrams
          WHERE count >= ?
        )
        WHERE rank <= ?
        ORDER BY prefix ASC, rank ASC
    """
    rows = connection.execute(query, (min_count, max_candidates_per_prefix))
    with row_output.open("wb") as row_handle, candidate_output.open("ab") as candidate_handle:
        return write_grouped_bigram_stream(rows, row_handle, candidate_handle, candidate_start)


def write_trigram_sections(
    connection: sqlite3.Connection,
    row_output: Path,
    candidate_output: Path,
    max_candidates_per_prefix: int,
    min_count: int,
    candidate_start: int,
) -> tuple[int, int]:
    query = """
        SELECT prefix1, prefix2, token, count
        FROM (
          SELECT prefix1, prefix2, token, count,
                 ROW_NUMBER() OVER (
                   PARTITION BY prefix1, prefix2 ORDER BY count DESC, token ASC
                 ) AS rank
          FROM trigrams
          WHERE count >= ?
        )
        WHERE rank <= ?
        ORDER BY prefix1 ASC, prefix2 ASC, rank ASC
    """
    rows = connection.execute(query, (min_count, max_candidates_per_prefix))
    with row_output.open("wb") as row_handle, candidate_output.open("ab") as candidate_handle:
        return write_grouped_trigram_stream(rows, row_handle, candidate_handle, candidate_start)


def write_grouped_bigram_stream(
    rows: Iterable[tuple[int, int, int]],
    row_handle,
    candidate_handle,
    candidate_count: int,
) -> tuple[int, int]:
    row_count = 0
    current_prefix: int | None = None
    row_start = candidate_count
    row_len = 0

    for prefix, token_id, count in rows:
        if current_prefix is not None and prefix != current_prefix:
            write_bigram_row(row_handle, current_prefix, row_start, row_len)
            row_count += 1
            row_start = candidate_count
            row_len = 0
        current_prefix = prefix
        write_candidate_record(candidate_handle, token_id, count)
        candidate_count += 1
        row_len += 1

    if current_prefix is not None:
        write_bigram_row(row_handle, current_prefix, row_start, row_len)
        row_count += 1

    return row_count, candidate_count


def write_grouped_trigram_stream(
    rows: Iterable[tuple[int, int, int, int]],
    row_handle,
    candidate_handle,
    candidate_count: int,
) -> tuple[int, int]:
    row_count = 0
    current_prefix: tuple[int, int] | None = None
    row_start = candidate_count
    row_len = 0

    for prefix1, prefix2, token_id, count in rows:
        prefix = (prefix1, prefix2)
        if current_prefix is not None and prefix != current_prefix:
            write_trigram_row(row_handle, current_prefix[0], current_prefix[1], row_start, row_len)
            row_count += 1
            row_start = candidate_count
            row_len = 0
        current_prefix = prefix
        write_candidate_record(candidate_handle, token_id, count)
        candidate_count += 1
        row_len += 1

    if current_prefix is not None:
        write_trigram_row(row_handle, current_prefix[0], current_prefix[1], row_start, row_len)
        row_count += 1

    return row_count, candidate_count


def write_header(
    handle,
    vocab_size: int,
    token_index_count: int,
    unigram_count: int,
    bigram_rows: int,
    trigram_rows: int,
    candidate_count: int,
    token_bytes_len: int,
) -> None:
    handle.write(MAGIC)
    write_u32_to_file(handle, VERSION)
    write_u32_to_file(handle, vocab_size)
    write_u32_to_file(handle, token_index_count)
    write_u32_to_file(handle, unigram_count)
    write_u32_to_file(handle, bigram_rows)
    write_u32_to_file(handle, trigram_rows)
    write_u32_to_file(handle, candidate_count)
    write_u32_to_file(handle, token_bytes_len)
    write_u32_to_file(handle, 0)


def write_bigram_row(handle, prefix: int, start: int, length: int) -> None:
    write_u32_to_file(handle, prefix)
    write_u32_to_file(handle, start)
    write_u32_to_file(handle, length)


def write_trigram_row(handle, prefix1: int, prefix2: int, start: int, length: int) -> None:
    write_u32_to_file(handle, prefix1)
    write_u32_to_file(handle, prefix2)
    write_u32_to_file(handle, start)
    write_u32_to_file(handle, length)


def write_candidate_record(handle, token_id: int, count: int) -> None:
    write_u32_to_file(handle, token_id)
    write_u32_to_file(handle, count)
    write_i32_to_file(handle, score_from_count(count))


def append_file(output_handle, path: Path) -> None:
    with path.open("rb") as input_handle:
        while chunk := input_handle.read(1024 * 1024):
            output_handle.write(chunk)


def score_from_count(count: int) -> int:
    return min(count, 2_147_483_647)


def write_u32(output: bytearray, value: int) -> None:
    output.extend(U32.pack(value))


def write_i32(output: bytearray, value: int) -> None:
    output.extend(I32.pack(value))


def write_u32_to_file(output, value: int) -> None:
    output.write(U32.pack(value))


def write_i32_to_file(output, value: int) -> None:
    output.write(I32.pack(value))


def manifest_path(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/autosuggest/corpus"))
    parser.add_argument("--vocab", type=Path, default=Path("data/autosuggest/models/ngram/vocab.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/autosuggest/models/ngram/autosuggest-ngram.bin"))
    parser.add_argument("--backend", choices=("sqlite", "memory"), default="sqlite")
    parser.add_argument("--sqlite-path", type=Path, default=Path("data/autosuggest/models/ngram/autosuggest-ngram.sqlite"))
    parser.add_argument("--reuse-sqlite", action="store_true")
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--max-sentences-per-source", type=int)
    parser.add_argument("--log-every-sentences", type=int, default=250_000)
    parser.add_argument("--max-candidates-per-prefix", type=int, default=8)
    parser.add_argument("--unigram-size", type=int, default=2048)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=500_000)
    args = parser.parse_args()

    report = build_ngram_lm(
        corpus_dir=args.corpus_dir,
        vocab_path=args.vocab,
        output=args.output,
        backend=args.backend,
        sqlite_path=args.sqlite_path,
        sources=set(args.sources) if args.sources else None,
        max_sentences=args.max_sentences,
        max_sentences_per_source=args.max_sentences_per_source,
        reuse_sqlite=args.reuse_sqlite,
        log_every_sentences=args.log_every_sentences,
        max_candidates_per_prefix=args.max_candidates_per_prefix,
        unigram_size=args.unigram_size,
        min_count=args.min_count,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
