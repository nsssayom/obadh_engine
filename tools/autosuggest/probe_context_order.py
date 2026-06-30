#!/usr/bin/env python3
"""Probe autosuggest quality for bounded n-gram context lengths.

This is an offline measurement tool. It intentionally does not define a runtime
artifact format; it answers whether a longer context is worth promoting into one.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from tools.autosuggest.common import BOS_ID, UNK_ID, load_vocab, sentence_paths


@dataclass
class ContextOrderCounts:
    unigram: Counter[int] = field(default_factory=Counter)
    contexts: dict[int, dict[tuple[int, ...], Counter[int]]] = field(default_factory=dict)

    @classmethod
    def with_max_context(cls, max_context: int) -> "ContextOrderCounts":
        return cls(
            contexts={
                order: defaultdict(Counter)
                for order in range(1, max_context + 1)
            }
        )

    def observe(self, encoded: list[int], max_context: int) -> None:
        recent: list[int] = []
        for token_id in encoded[1:]:
            if token_id <= UNK_ID:
                recent.clear()
                continue

            self.unigram[token_id] += 1
            usable = min(max_context, len(recent))
            for order in range(1, usable + 1):
                self.contexts[order][tuple(recent[-order:])][token_id] += 1

            recent.append(token_id)
            if len(recent) > max_context:
                recent.pop(0)


@dataclass(frozen=True)
class CandidateTable:
    unigrams: list[int]
    contexts: dict[int, dict[tuple[int, ...], list[int]]]
    context_rows: dict[int, int]
    candidate_rows: dict[int, int]

    @classmethod
    def from_counts(
        cls,
        counts: ContextOrderCounts,
        max_context: int,
        max_candidates_per_prefix: int,
        min_count: int,
        unigram_size: int,
    ) -> "CandidateTable":
        contexts: dict[int, dict[tuple[int, ...], list[int]]] = {}
        context_rows: dict[int, int] = {}
        candidate_rows: dict[int, int] = {}

        for order in range(1, max_context + 1):
            rows: dict[tuple[int, ...], list[int]] = {}
            row_candidates = 0
            for context, counter in counts.contexts[order].items():
                candidates = top_candidate_ids(counter, max_candidates_per_prefix, min_count)
                if not candidates:
                    continue
                rows[context] = candidates
                row_candidates += len(candidates)
            contexts[order] = rows
            context_rows[order] = len(rows)
            candidate_rows[order] = row_candidates

        return cls(
            unigrams=[
                token_id
                for token_id, _ in sorted(
                    counts.unigram.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:unigram_size]
            ],
            contexts=contexts,
            context_rows=context_rows,
            candidate_rows=candidate_rows,
        )

    def suggest(self, recent: list[int], max_context: int, limit: int) -> list[int]:
        output: list[int] = []
        seen: set[int] = set()

        usable = min(max_context, len(recent))
        for order in range(usable, 0, -1):
            candidates = self.contexts[order].get(tuple(recent[-order:]))
            if not candidates:
                continue
            append_unique(candidates, limit, seen, output)
            if len(output) >= limit:
                return output

        append_unique(self.unigrams, limit, seen, output)
        return output


def probe(
    corpus_dir: Path,
    vocab_path: Path,
    sources: set[str] | None,
    train_sentences_per_source: int,
    eval_sentences_per_source: int,
    max_context: int,
    max_candidates_per_prefix: int,
    min_count: int,
    unigram_size: int,
    top_k: int,
    log_every_sentences: int,
) -> dict:
    words, vocab = load_vocab(vocab_path)
    counts = ContextOrderCounts.with_max_context(max_context)
    started_at = time.monotonic()
    train_sentences = Counter()
    train_tokens = Counter()

    for source, tokens in iter_sentence_split(
        corpus_dir,
        sources=sources,
        start_per_source=0,
        limit_per_source=train_sentences_per_source,
    ):
        encoded = encode_tokens(tokens, vocab)
        counts.observe(encoded, max_context)
        train_sentences[source] += 1
        train_tokens[source] += max(0, len(encoded) - 1)
        total_train_sentences = train_sentences.total()
        if log_every_sentences > 0 and total_train_sentences % log_every_sentences == 0:
            log_progress("train", started_at, total_train_sentences, train_sentences)

    table = CandidateTable.from_counts(
        counts,
        max_context=max_context,
        max_candidates_per_prefix=max_candidates_per_prefix,
        min_count=min_count,
        unigram_size=unigram_size,
    )

    profiles = {
        order: EvaluationStats()
        for order in range(1, max_context + 1)
    }
    skipped_unknown_targets = 0
    eval_sentences = Counter()
    eval_tokens = Counter()

    for source, tokens in iter_sentence_split(
        corpus_dir,
        sources=sources,
        start_per_source=train_sentences_per_source,
        limit_per_source=eval_sentences_per_source,
    ):
        recent: list[int] = []
        eval_sentences[source] += 1
        encoded = encode_tokens(tokens, vocab)
        eval_tokens[source] += max(0, len(encoded) - 1)

        for target in encoded[1:]:
            if target <= UNK_ID:
                skipped_unknown_targets += 1
                recent.clear()
                continue

            for order, stats in profiles.items():
                candidates = table.suggest(recent, max_context=order, limit=top_k)
                stats.observe(source, target, candidates, top_k)

            recent.append(target)
            if len(recent) > max_context:
                recent.pop(0)

    return {
        "probe": "autosuggest_context_order",
        "corpus_dir": str(corpus_dir),
        "vocab_path": str(vocab_path),
        "sources": sorted(sources) if sources else None,
        "vocab_size": len(words),
        "train_sentences_per_source": train_sentences_per_source,
        "eval_sentences_per_source": eval_sentences_per_source,
        "max_context": max_context,
        "max_candidates_per_prefix": max_candidates_per_prefix,
        "min_count": min_count,
        "unigram_size": unigram_size,
        "top_k": top_k,
        "train": {
            "sentences": dict(sorted(train_sentences.items())),
            "tokens": dict(sorted(train_tokens.items())),
        },
        "eval": {
            "sentences": dict(sorted(eval_sentences.items())),
            "tokens": dict(sorted(eval_tokens.items())),
            "skipped_unknown_targets": skipped_unknown_targets,
        },
        "tables": {
            str(order): {
                "context_rows": table.context_rows[order],
                "candidate_rows": table.candidate_rows[order],
            }
            for order in range(1, max_context + 1)
        },
        "profiles": {
            str(order): stats.report(top_k)
            for order, stats in profiles.items()
        },
    }


@dataclass
class EvaluationStats:
    eligible_targets: int = 0
    reciprocal_rank_sum: float = 0.0
    hits: Counter[int] = field(default_factory=Counter)
    per_source_total: Counter[str] = field(default_factory=Counter)
    per_source_top1: Counter[str] = field(default_factory=Counter)
    per_source_top5: Counter[str] = field(default_factory=Counter)

    def observe(self, source: str, target: int, candidates: list[int], top_k: int) -> None:
        self.eligible_targets += 1
        self.per_source_total[source] += 1
        try:
            rank = candidates.index(target) + 1
        except ValueError:
            rank = 0

        if rank:
            self.reciprocal_rank_sum += 1.0 / rank
        for k in (1, 3, 5, top_k):
            if rank and rank <= min(k, top_k):
                self.hits[k] += 1
        if rank == 1:
            self.per_source_top1[source] += 1
        if rank and rank <= min(5, top_k):
            self.per_source_top5[source] += 1

    def report(self, top_k: int) -> dict:
        def ratio(value: int, denominator: int = self.eligible_targets) -> float:
            return value / denominator if denominator else 0.0

        return {
            "eligible_targets": self.eligible_targets,
            "top1": ratio(self.hits[1]),
            "top3": ratio(self.hits[3]),
            "top5": ratio(self.hits[5]),
            f"top{top_k}": ratio(self.hits[top_k]),
            "mrr": (
                self.reciprocal_rank_sum / self.eligible_targets
                if self.eligible_targets
                else 0.0
            ),
            "per_source": {
                source: {
                    "eligible_targets": self.per_source_total[source],
                    "top1": ratio(self.per_source_top1[source], self.per_source_total[source]),
                    "top5": ratio(self.per_source_top5[source], self.per_source_total[source]),
                }
                for source in sorted(self.per_source_total)
            },
        }


def iter_sentence_split(
    corpus_dir: Path,
    sources: set[str] | None,
    start_per_source: int,
    limit_per_source: int,
) -> Iterator[tuple[str, list[str]]]:
    seen_by_source: Counter[str] = Counter()
    emitted_by_source: Counter[str] = Counter()
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
                if seen_by_source[source] <= start_per_source:
                    continue
                if emitted_by_source[source] >= limit_per_source:
                    if pending_sources is not None:
                        pending_sources.discard(source)
                    break

                tokens = row["tokens"].split(" ") if row.get("tokens") else []
                if not tokens:
                    continue

                emitted_by_source[source] += 1
                yield source, tokens

                if emitted_by_source[source] >= limit_per_source:
                    if pending_sources is not None:
                        pending_sources.discard(source)
                        if not pending_sources:
                            return
                    break


def encode_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    return [BOS_ID, *(vocab.get(token, UNK_ID) for token in tokens)]


def top_candidate_ids(counter: Counter[int], limit: int, min_count: int) -> list[int]:
    return [
        token_id
        for token_id, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_count
    ][:limit]


def append_unique(
    candidates: list[int],
    limit: int,
    seen: set[int],
    output: list[int],
) -> None:
    for token_id in candidates:
        if len(output) >= limit:
            break
        if token_id <= UNK_ID or token_id in seen:
            continue
        seen.add(token_id)
        output.append(token_id)


def log_progress(
    phase: str,
    started_at: float,
    sentences: int,
    by_source: Counter[str],
) -> None:
    print(
        json.dumps(
            {
                "event": "autosuggest_context_order_probe_progress",
                "phase": phase,
                "sentences": sentences,
                "source_sentences": dict(sorted(by_source.items())),
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/autosuggest/corpus"))
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("data/autosuggest/models/ngram/vocab.tsv"),
    )
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--train-sentences-per-source", type=int, default=100_000)
    parser.add_argument("--eval-sentences-per-source", type=int, default=25_000)
    parser.add_argument("--max-context", type=int, default=3)
    parser.add_argument("--max-candidates-per-prefix", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--unigram-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--log-every-sentences", type=int, default=50_000)
    args = parser.parse_args()

    if args.max_context < 1:
        raise SystemExit("--max-context must be at least 1")
    if args.max_candidates_per_prefix < 1:
        raise SystemExit("--max-candidates-per-prefix must be at least 1")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")

    result = probe(
        corpus_dir=args.corpus_dir,
        vocab_path=args.vocab_path,
        sources=set(args.sources) if args.sources else None,
        train_sentences_per_source=args.train_sentences_per_source,
        eval_sentences_per_source=args.eval_sentences_per_source,
        max_context=args.max_context,
        max_candidates_per_prefix=args.max_candidates_per_prefix,
        min_count=args.min_count,
        unigram_size=args.unigram_size,
        top_k=args.top_k,
        log_every_sentences=args.log_every_sentences,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
