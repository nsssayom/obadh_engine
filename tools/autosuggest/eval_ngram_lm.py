#!/usr/bin/env python3
"""Evaluate Obadh n-gram autosuggest artifacts on held-out corpus rows."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from tools.autosuggest.common import BOS_ID, PAD_ID, UNK_ID, sentence_paths


MAGIC = b"OBAUTOSUGLM_V1\0\0"
HEADER = struct.Struct("<16sIIIIIIIII")
U32 = struct.Struct("<I")
I32 = struct.Struct("<i")
ID_TOKEN_RECORD_LEN = 8
TOKEN_INDEX_RECORD_LEN = 12
CANDIDATE_RECORD_LEN = 12
BIGRAM_ROW_LEN = 12
TRIGRAM_ROW_LEN = 16


@dataclass(frozen=True)
class Candidate:
    token_id: int
    count: int
    score: int


def candidate_sort_key(candidate: Candidate) -> tuple[int, int, int]:
    return (-candidate.score, -candidate.count, candidate.token_id)


class NgramLm:
    def __init__(self, path: Path) -> None:
        self.bytes = path.read_bytes()
        (
            magic,
            self.version,
            self.vocab_size,
            self.token_index_count,
            self.unigram_count,
            self.bigram_row_count,
            self.trigram_row_count,
            self.candidate_count,
            self.token_bytes_len,
            _reserved,
        ) = HEADER.unpack_from(self.bytes, 0)
        if magic != MAGIC:
            raise ValueError(f"invalid artifact magic in {path}")
        if self.version != 1:
            raise ValueError(f"unsupported artifact version {self.version}")

        offset = HEADER.size
        self.id_tokens_offset = offset
        offset += self.vocab_size * ID_TOKEN_RECORD_LEN
        self.token_index_offset = offset
        offset += self.token_index_count * TOKEN_INDEX_RECORD_LEN
        self.unigrams_offset = offset
        offset += self.unigram_count * CANDIDATE_RECORD_LEN
        self.bigram_rows_offset = offset
        offset += self.bigram_row_count * BIGRAM_ROW_LEN
        self.trigram_rows_offset = offset
        offset += self.trigram_row_count * TRIGRAM_ROW_LEN
        self.candidates_offset = offset
        offset += self.candidate_count * CANDIDATE_RECORD_LEN
        self.token_bytes_offset = offset
        offset += self.token_bytes_len
        if offset != len(self.bytes):
            raise ValueError("artifact section sizes do not match file length")

        self.token_to_id = self._load_token_lookup()
        self.score_mode = self._detect_score_mode()

    def _load_token_lookup(self) -> dict[str, int]:
        lookup: dict[str, int] = {}
        for index in range(self.token_index_count):
            offset = self.token_index_offset + index * TOKEN_INDEX_RECORD_LEN
            token_offset = read_u32(self.bytes, offset)
            token_len = read_u32(self.bytes, offset + 4)
            token_id = read_u32(self.bytes, offset + 8)
            lookup[self._token_text(token_offset, token_len)] = token_id
        return lookup

    def token_id(self, token: str) -> int:
        return self.token_to_id.get(token, UNK_ID)

    def suggest_ids(self, context_ids: list[int], limit: int) -> list[int]:
        recent: list[int] = []
        for token_id in context_ids:
            if token_id > UNK_ID:
                if len(recent) == 2:
                    recent.pop(0)
                recent.append(token_id)
            elif token_id in (PAD_ID, BOS_ID, UNK_ID):
                recent.clear()

        if self.score_mode == "backoff":
            return self._suggest_ids_backoff(recent, limit)

        output: list[Candidate] = []

        if len(recent) == 2:
            row = self._find_trigram_row(recent[0], recent[1])
            if row:
                self._merge(row[0], row[1], limit, output)
        if recent:
            row = self._find_bigram_row(recent[-1])
            if row:
                self._merge(row[0], row[1], limit, output)
        for index in range(self.unigram_count):
            offset = self.unigrams_offset + index * CANDIDATE_RECORD_LEN
            if self._merge_candidate(self._candidate_at(offset), limit, output):
                break
        return [candidate.token_id for candidate in output]

    def _suggest_ids_backoff(self, recent: list[int], limit: int) -> list[int]:
        output: list[int] = []
        seen: set[int] = set()

        if len(recent) == 2:
            row = self._find_trigram_row(recent[0], recent[1])
            if row:
                self._append(row[0], row[1], limit, seen, output)
        if len(output) < limit and recent:
            row = self._find_bigram_row(recent[-1])
            if row:
                self._append(row[0], row[1], limit, seen, output)
        if len(output) < limit:
            for index in range(self.unigram_count):
                if len(output) >= limit:
                    break
                offset = self.unigrams_offset + index * CANDIDATE_RECORD_LEN
                token_id = read_u32(self.bytes, offset)
                if token_id > UNK_ID and token_id not in seen:
                    seen.add(token_id)
                    output.append(token_id)
        return output

    def _find_bigram_row(self, prefix: int) -> tuple[int, int] | None:
        low = 0
        high = self.bigram_row_count
        while low < high:
            mid = low + (high - low) // 2
            offset = self.bigram_rows_offset + mid * BIGRAM_ROW_LEN
            row_prefix = read_u32(self.bytes, offset)
            if row_prefix < prefix:
                low = mid + 1
            elif row_prefix > prefix:
                high = mid
            else:
                return read_u32(self.bytes, offset + 4), read_u32(self.bytes, offset + 8)
        return None

    def _find_trigram_row(self, prefix1: int, prefix2: int) -> tuple[int, int] | None:
        low = 0
        high = self.trigram_row_count
        target = (prefix1, prefix2)
        while low < high:
            mid = low + (high - low) // 2
            offset = self.trigram_rows_offset + mid * TRIGRAM_ROW_LEN
            row = (read_u32(self.bytes, offset), read_u32(self.bytes, offset + 4))
            if row < target:
                low = mid + 1
            elif row > target:
                high = mid
            else:
                return read_u32(self.bytes, offset + 8), read_u32(self.bytes, offset + 12)
        return None

    def _merge(
        self,
        start: int,
        length: int,
        limit: int,
        output: list[Candidate],
    ) -> None:
        for index in range(start, start + length):
            offset = self.candidates_offset + index * CANDIDATE_RECORD_LEN
            if self._merge_candidate(self._candidate_at(offset), limit, output):
                break

    def _append(
        self,
        start: int,
        length: int,
        limit: int,
        seen: set[int],
        output: list[int],
    ) -> None:
        for index in range(start, start + length):
            if len(output) >= limit:
                break
            offset = self.candidates_offset + index * CANDIDATE_RECORD_LEN
            token_id = read_u32(self.bytes, offset)
            if token_id > UNK_ID and token_id not in seen:
                seen.add(token_id)
                output.append(token_id)

    def _merge_candidate(
        self,
        candidate: Candidate,
        limit: int,
        output: list[Candidate],
    ) -> bool:
        if candidate.token_id <= UNK_ID:
            return False

        existing_index = next(
            (
                index
                for index, item in enumerate(output)
                if item.token_id == candidate.token_id
            ),
            None,
        )
        if existing_index is not None:
            if candidate_sort_key(candidate) < candidate_sort_key(output[existing_index]):
                output.pop(existing_index)
            elif len(output) >= limit and candidate_sort_key(candidate) >= candidate_sort_key(output[-1]):
                return True
            else:
                return False
        elif len(output) >= limit and candidate_sort_key(candidate) >= candidate_sort_key(output[-1]):
            return True
        elif len(output) >= limit:
            output.pop()

        insert_at = len(output)
        for index, item in enumerate(output):
            if candidate_sort_key(candidate) < candidate_sort_key(item):
                insert_at = index
                break
        output.insert(insert_at, candidate)
        return False

    def _candidate_at(self, offset: int) -> Candidate:
        return Candidate(
            token_id=read_u32(self.bytes, offset),
            count=read_u32(self.bytes, offset + 4),
            score=read_i32(self.bytes, offset + 8),
        )

    def _detect_score_mode(self) -> str:
        sample_len = min(self.unigram_count, 8)
        if sample_len == 0:
            return "interpolated"
        for index in range(sample_len):
            offset = self.unigrams_offset + index * CANDIDATE_RECORD_LEN
            count = read_u32(self.bytes, offset + 4)
            score = read_i32(self.bytes, offset + 8)
            if score < 0 or score != count:
                return "interpolated"
        return "backoff"

    def _token_text(self, offset: int, length: int) -> str:
        start = self.token_bytes_offset + offset
        return self.bytes[start : start + length].decode("utf-8")


def evaluate(
    model: Path,
    corpus_dir: Path,
    sources: set[str] | None,
    skip_sentences_per_source: int,
    max_sentences_per_source: int | None,
    max_targets: int | None,
    top_k: int,
) -> dict:
    lm = NgramLm(model)
    total = 0
    skipped_unknown_target = 0
    hits = Counter()
    reciprocal_rank_sum = 0.0
    per_source_total = Counter()
    per_source_top1 = Counter()
    per_source_top5 = Counter()

    for source, tokens in iter_eval_sentence_tokens(
        corpus_dir,
        sources=sources,
        skip_sentences_per_source=skip_sentences_per_source,
        max_sentences_per_source=max_sentences_per_source,
    ):
        encoded = [BOS_ID, *(lm.token_id(token) for token in tokens)]
        for index in range(1, len(encoded)):
            target = encoded[index]
            if target <= UNK_ID:
                skipped_unknown_target += 1
                continue
            candidates = lm.suggest_ids(encoded[:index], top_k)
            total += 1
            per_source_total[source] += 1
            try:
                rank = candidates.index(target) + 1
            except ValueError:
                rank = 0
            if rank:
                reciprocal_rank_sum += 1.0 / rank
            for k in (1, 3, 5, top_k):
                if rank and rank <= min(k, top_k):
                    hits[k] += 1
            if rank == 1:
                per_source_top1[source] += 1
            if rank and rank <= min(5, top_k):
                per_source_top5[source] += 1
            if max_targets is not None and total >= max_targets:
                return report(
                    lm,
                    total,
                    skipped_unknown_target,
                    hits,
                    reciprocal_rank_sum,
                    per_source_total,
                    per_source_top1,
                    per_source_top5,
                    top_k,
                )

    return report(
        lm,
        total,
        skipped_unknown_target,
        hits,
        reciprocal_rank_sum,
        per_source_total,
        per_source_top1,
        per_source_top5,
        top_k,
    )


def report(
    lm: NgramLm,
    total: int,
    skipped_unknown_target: int,
    hits: Counter,
    reciprocal_rank_sum: float,
    per_source_total: Counter,
    per_source_top1: Counter,
    per_source_top5: Counter,
    top_k: int,
) -> dict:
    def ratio(value: int, denominator: int = total) -> float:
        return value / denominator if denominator else 0.0

    return {
        "artifact": {
            "bytes": len(lm.bytes),
            "vocab_size": lm.vocab_size,
            "unigram_count": lm.unigram_count,
            "bigram_rows": lm.bigram_row_count,
            "trigram_rows": lm.trigram_row_count,
            "candidate_rows": lm.candidate_count,
        },
        "eligible_targets": total,
        "skipped_unknown_targets": skipped_unknown_target,
        "top1": ratio(hits[1]),
        "top3": ratio(hits[3]),
        "top5": ratio(hits[5]),
        f"top{top_k}": ratio(hits[top_k]),
        "mrr": reciprocal_rank_sum / total if total else 0.0,
        "per_source": {
            source: {
                "eligible_targets": per_source_total[source],
                "top1": ratio(per_source_top1[source], per_source_total[source]),
                "top5": ratio(per_source_top5[source], per_source_total[source]),
            }
            for source in sorted(per_source_total)
        },
    }


def iter_eval_sentence_tokens(
    corpus_dir: Path,
    sources: set[str] | None,
    skip_sentences_per_source: int,
    max_sentences_per_source: int | None,
) -> Iterator[tuple[str, list[str]]]:
    emitted_by_source: Counter[str] = Counter()
    seen_by_source: Counter[str] = Counter()
    pending_sources = set(sources) if sources else None

    for path in sentence_paths(corpus_dir):
        source_hint = path.stem.split(".")[0]
        if sources is not None and source_hint not in sources:
            continue
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                source = row["source"]
                if sources is not None and source not in sources:
                    continue
                seen_by_source[source] += 1
                if seen_by_source[source] <= skip_sentences_per_source:
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
                emitted_by_source[source] += 1
                yield source, tokens
                if (
                    pending_sources is not None
                    and max_sentences_per_source is not None
                    and emitted_by_source[source] >= max_sentences_per_source
                ):
                    pending_sources.discard(source)
                    if not pending_sources:
                        return
                    break


def read_u32(bytes_: bytes, offset: int) -> int:
    return U32.unpack_from(bytes_, offset)[0]


def read_i32(bytes_: bytes, offset: int) -> int:
    return I32.unpack_from(bytes_, offset)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/autosuggest/corpus"))
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--skip-sentences-per-source", type=int, default=0)
    parser.add_argument("--max-sentences-per-source", type=int)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    result = evaluate(
        model=args.model,
        corpus_dir=args.corpus_dir,
        sources=set(args.sources) if args.sources else None,
        skip_sentences_per_source=args.skip_sentences_per_source,
        max_sentences_per_source=args.max_sentences_per_source,
        max_targets=args.max_targets,
        top_k=args.top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
