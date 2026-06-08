"""Dataset admission checks for Bengali transliteration pairs."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

AUDIT_SCHEMA_VERSION = "obadh.ml.dataset_audit.v0"

ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
URL_OR_EMAIL_RE = re.compile(r"(https?://|www\.|@[\w.-]+|[\w.-]+@[\w.-]+)", re.IGNORECASE)
MARKUP_RE = re.compile(r"<[^>]+>|&[A-Za-z]+;")
CODELIKE_RE = re.compile(r"(\b[a-zA-Z_][\w-]*\.[a-zA-Z0-9]{1,5}\b|[{}<>]=?|::|//)")
REPEATED_PUNCT_RE = re.compile(r"([!?.,।])\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")

COMMON_ALLOWED_TARGET_ASCII = set(" \t\n\r0123456789.,!?;:'\"()[]{}-/\\|@#$%&*+=_~`")
FOREIGN_BENGALI_BLOCK_LETTERS = {
    "\u09f0": "assamese_ra",
    "\u09f1": "assamese_wa",
}


@dataclass(frozen=True)
class AuditConfig:
    mode: str = "word"
    max_latin_chars: int = 64
    max_target_chars: int = 64
    max_length_ratio: float = 4.0
    min_bengali_letter_ratio: float = 0.65
    max_target_ascii_alpha_ratio: float = 0.0
    max_latin_native_ratio: float = 0.0
    allow_digits: bool = True
    allow_sentence_punctuation: bool = False
    warn_on_domain_noise: bool = True


@dataclass(frozen=True)
class PairRecord:
    source_id: str
    row_id: str
    latin: str
    target: str
    split: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class PairAuditResult:
    source_id: str
    row_id: str
    accepted: bool
    latin: str
    target: str
    normalized_latin: str
    normalized_target: str
    issues: list[AuditIssue]
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class AuditSummary:
    schema: str
    source_id: str
    mode: str
    total_rows: int
    accepted_rows: int
    rejected_rows: int
    issue_counts: Mapping[str, int]
    severity_counts: Mapping[str, int]
    duplicate_latin_targets: int
    conflicting_latin_labels: int

    @property
    def acceptance_rate(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.accepted_rows / self.total_rows


def audit_records(
    records: Iterable[PairRecord],
    config: AuditConfig,
) -> tuple[list[PairAuditResult], AuditSummary]:
    results = [audit_pair(record, config) for record in records]
    issue_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    accepted_rows = 0

    latin_to_targets: dict[str, set[str]] = {}
    pair_counts: Counter[tuple[str, str]] = Counter()

    for result in results:
        if result.accepted:
            accepted_rows += 1
        for issue in result.issues:
            issue_counts[issue.code] += 1
            severity_counts[issue.severity] += 1
        if result.normalized_latin and result.normalized_target:
            latin_to_targets.setdefault(result.normalized_latin.casefold(), set()).add(
                result.normalized_target
            )
            pair_counts[(result.normalized_latin.casefold(), result.normalized_target)] += 1

    duplicate_latin_targets = sum(count - 1 for count in pair_counts.values() if count > 1)
    conflicting_latin_labels = sum(1 for targets in latin_to_targets.values() if len(targets) > 1)

    summary = AuditSummary(
        schema=AUDIT_SCHEMA_VERSION,
        source_id=results[0].source_id if results else "unknown",
        mode=config.mode,
        total_rows=len(results),
        accepted_rows=accepted_rows,
        rejected_rows=len(results) - accepted_rows,
        issue_counts=dict(sorted(issue_counts.items())),
        severity_counts=dict(sorted(severity_counts.items())),
        duplicate_latin_targets=duplicate_latin_targets,
        conflicting_latin_labels=conflicting_latin_labels,
    )
    return results, summary


def audit_pair(record: PairRecord, config: AuditConfig) -> PairAuditResult:
    if config.mode not in {"word", "sentence"}:
        raise ValueError("audit mode must be 'word' or 'sentence'")

    normalized_latin = normalize(record.latin)
    normalized_target = normalize(record.target)
    issues: list[AuditIssue] = []

    add_required_text_issues(issues, "latin", normalized_latin)
    add_required_text_issues(issues, "target", normalized_target)
    add_latin_issues(issues, normalized_latin, config)
    add_target_issues(issues, normalized_target, config)
    add_pair_shape_issues(issues, normalized_latin, normalized_target, config)

    metrics = pair_metrics(normalized_latin, normalized_target)
    accepted = not any(issue.severity == "reject" for issue in issues)

    return PairAuditResult(
        source_id=record.source_id,
        row_id=record.row_id,
        accepted=accepted,
        latin=record.latin,
        target=record.target,
        normalized_latin=normalized_latin,
        normalized_target=normalized_target,
        issues=issues,
        metrics=metrics,
    )


def normalize(text: str) -> str:
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", text).strip())


def add_required_text_issues(issues: list[AuditIssue], field_name: str, text: str) -> None:
    if not text:
        issues.append(reject(f"{field_name}_empty", f"{field_name} text is empty"))
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in text):
        issues.append(
            reject(
                f"{field_name}_control_char",
                f"{field_name} contains control characters",
            )
        )


def add_latin_issues(issues: list[AuditIssue], latin: str, config: AuditConfig) -> None:
    if len(latin) > config.max_latin_chars:
        issues.append(reject("latin_too_long", "latin text exceeds configured length limit"))

    if not ASCII_ALPHA_RE.search(latin):
        issues.append(reject("latin_has_no_ascii_letters", "latin text has no ASCII letters"))

    native_count = sum(is_bengali_block(char) for char in latin)
    if ratio(native_count, len(latin)) > config.max_latin_native_ratio:
        issues.append(
            reject(
                "latin_contains_native_script",
                "latin side contains Bengali-script text",
            )
        )

    other_letters = [
        char
        for char in latin
        if char.isalpha() and not char.isascii() and not is_bengali_block(char)
    ]
    if other_letters:
        issues.append(
            reject(
                "latin_contains_foreign_letters",
                "latin side contains non-ASCII letters",
            )
        )

    unexpected = [
        char
        for char in latin
        if not char.isascii()
        and not is_bengali_block(char)
        and unicodedata.category(char)[0] != "P"
    ]
    if unexpected:
        issues.append(
            warn(
                "latin_non_ascii_noise",
                "latin side contains non-ASCII symbols or marks",
            )
        )

    if URL_OR_EMAIL_RE.search(latin):
        issues.append(
            warn_or_reject(
                config,
                "latin_url_or_email",
                "latin side contains URL/email-like text",
            )
        )
    if MARKUP_RE.search(latin):
        issues.append(reject("latin_markup", "latin side contains markup-like text"))
    if CODELIKE_RE.search(latin):
        issues.append(
            warn_or_reject(
                config,
                "latin_codelike",
                "latin side contains code/file-like text",
            )
        )
    if REPEATED_PUNCT_RE.search(latin):
        issues.append(
            warn("latin_repeated_punctuation", "latin side contains repeated punctuation")
        )


def add_target_issues(issues: list[AuditIssue], target: str, config: AuditConfig) -> None:
    if len(target) > config.max_target_chars:
        issues.append(reject("target_too_long", "target text exceeds configured length limit"))

    bengali_letters = sum(is_bengali_letter(char) for char in target)
    target_letters = sum(char.isalpha() for char in target)
    if bengali_letters == 0:
        issues.append(reject("target_has_no_bengali_letters", "target has no Bengali letters"))
    if target_letters and ratio(bengali_letters, target_letters) < config.min_bengali_letter_ratio:
        issues.append(
            reject(
                "target_not_bengali_dominant",
                "target is not Bengali-letter dominant",
            )
        )

    ascii_alpha = sum(char.isascii() and char.isalpha() for char in target)
    if ratio(ascii_alpha, max(1, len(target))) > config.max_target_ascii_alpha_ratio:
        issues.append(reject("target_contains_ascii_letters", "target contains Latin letters"))

    foreign_block_letters = [
        f"{FOREIGN_BENGALI_BLOCK_LETTERS[char]}:{char}"
        for char in target
        if char in FOREIGN_BENGALI_BLOCK_LETTERS
    ]
    if foreign_block_letters:
        issues.append(
            reject(
                "target_contains_non_bangla_bengali_block_letters",
                "target contains non-Bangla Bengali-block letters",
            )
        )

    disallowed_ascii = [
        char for char in target if char.isascii() and char not in COMMON_ALLOWED_TARGET_ASCII
    ]
    if disallowed_ascii:
        issues.append(
            reject("target_contains_disallowed_ascii", "target contains disallowed ASCII")
        )

    if not config.allow_digits and any(char.isdigit() for char in target):
        issues.append(reject("target_contains_digits", "target contains digits"))
    if URL_OR_EMAIL_RE.search(target):
        issues.append(
            warn_or_reject(
                config,
                "target_url_or_email",
                "target contains URL/email-like text",
            )
        )
    if MARKUP_RE.search(target):
        issues.append(reject("target_markup", "target contains markup-like text"))
    if CODELIKE_RE.search(target):
        issues.append(
            warn_or_reject(
                config,
                "target_codelike",
                "target contains code/file-like text",
            )
        )
    if REPEATED_PUNCT_RE.search(target):
        issues.append(warn("target_repeated_punctuation", "target contains repeated punctuation"))


def add_pair_shape_issues(
    issues: list[AuditIssue],
    latin: str,
    target: str,
    config: AuditConfig,
) -> None:
    latin_words = word_count(latin)
    target_words = word_count(target)

    if config.mode == "word":
        if latin_words != 1 or target_words != 1:
            issues.append(
                reject(
                    "not_word_pair",
                    "word-model rows must contain one source and one target token",
                )
            )
        if any(is_sentence_punctuation(char) for char in latin + target):
            issues.append(
                reject(
                    "word_pair_contains_sentence_punctuation",
                    "word-model rows cannot contain sentence punctuation",
                )
            )

    if config.mode == "sentence" and not config.allow_sentence_punctuation:
        if any(is_sentence_punctuation(char) for char in latin + target):
            issues.append(warn("sentence_punctuation", "sentence row contains punctuation"))

    if latin and target:
        length_ratio = max(len(latin), len(target)) / max(1, min(len(latin), len(target)))
        if length_ratio > config.max_length_ratio:
            issues.append(reject("length_ratio_outlier", "latin/target length ratio is an outlier"))


def pair_metrics(latin: str, target: str) -> dict[str, Any]:
    return {
        "latin_chars": len(latin),
        "target_chars": len(target),
        "latin_words": word_count(latin),
        "target_words": word_count(target),
        "target_bengali_letters": sum(is_bengali_letter(char) for char in target),
        "target_ascii_letters": sum(char.isascii() and char.isalpha() for char in target),
        "latin_native_chars": sum(is_bengali_block(char) for char in latin),
    }


def is_bengali_block(char: str) -> bool:
    return "\u0980" <= char <= "\u09ff"


def is_bengali_letter(char: str) -> bool:
    return is_bengali_block(char) and unicodedata.category(char).startswith("L")


def is_sentence_punctuation(char: str) -> bool:
    return char in ".!?।॥"


def word_count(text: str) -> int:
    return len([part for part in text.split(" ") if part])


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def reject(code: str, message: str) -> AuditIssue:
    return AuditIssue(code=code, severity="reject", message=message)


def warn(code: str, message: str) -> AuditIssue:
    return AuditIssue(code=code, severity="warn", message=message)


def warn_or_reject(config: AuditConfig, code: str, message: str) -> AuditIssue:
    if config.mode == "word":
        return reject(code, message)
    return warn(code, message) if config.warn_on_domain_noise else reject(code, message)


def result_to_json(result: PairAuditResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["issues"] = [asdict(issue) for issue in result.issues]
    return payload


def summary_to_json(summary: AuditSummary) -> dict[str, Any]:
    payload = asdict(summary)
    payload["acceptance_rate"] = summary.acceptance_rate
    return payload


def write_audit_report(path: Path, summary: AuditSummary, results: list[PairAuditResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary_to_json(summary),
        "rejected_examples": [
            result_to_json(result)
            for result in results
            if not result.accepted
        ][:100],
        "warning_examples": [
            result_to_json(result)
            for result in results
            if result.accepted and result.issues
        ][:100],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
