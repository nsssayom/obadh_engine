"""Corpus admission helpers built on source readers and audit results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obadh_ml.data.audit import (
    AuditConfig,
    PairAuditResult,
    audit_records,
    summary_to_json,
    write_audit_report,
)
from obadh_ml.data.sources import LoadedSource, SourceSpec, file_sha256, load_source, parse_bool

CORPUS_MANIFEST_SCHEMA = "obadh.ml.corpus_manifest.v0"
CORPUS_SUMMARY_SCHEMA = "obadh.ml.corpus_admission.v0"
ADMITTED_PAIR_SCHEMA = "obadh.ml.admitted_pair.v0"


@dataclass(frozen=True)
class CorpusSourceAdmission:
    source_id: str
    kind: str
    split: str | None
    mode: str
    admit: bool
    input_files: list[dict[str, Any]]
    audit_report: str
    accepted_output: str | None
    summary: Mapping[str, Any]


def admit_source(
    spec: SourceSpec,
    output_dir: Path,
    *,
    corpus_id: str,
    audit_config: AuditConfig,
) -> CorpusSourceAdmission:
    loaded = load_source(spec)
    results, summary = audit_records(loaded.records, audit_config, source_id=spec.source_id)

    reports_dir = output_dir / "reports"
    accepted_dir = output_dir / "accepted"
    source_key = source_file_stem(spec)
    report_path = reports_dir / f"{source_key}.audit.json"
    write_audit_report(report_path, summary, results)

    accepted_path: Path | None = None
    if spec.admit:
        accepted_path = accepted_dir / f"{corpus_id}.{source_key}.accepted.jsonl"
        write_accepted_pairs(accepted_path, results)

    return CorpusSourceAdmission(
        source_id=spec.source_id,
        kind=spec.kind,
        split=spec.split,
        mode=spec.mode,
        admit=spec.admit,
        input_files=input_file_metadata(loaded),
        audit_report=str(report_path),
        accepted_output=str(accepted_path) if accepted_path else None,
        summary=summary_to_json(summary),
    )


def write_accepted_pairs(path: Path, results: list[PairAuditResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            if result.accepted:
                handle.write(json.dumps(admitted_pair_json(result), ensure_ascii=False))
                handle.write("\n")


def admitted_pair_json(result: PairAuditResult) -> dict[str, Any]:
    return {
        "schema": ADMITTED_PAIR_SCHEMA,
        "source_id": result.source_id,
        "row_id": result.row_id,
        "split": result.split,
        "latin": result.normalized_latin,
        "target": result.normalized_target,
        "original_latin": result.latin,
        "original_target": result.target,
        "metadata": dict(result.metadata),
        "audit": {
            "accepted": result.accepted,
            "issues": [issue.code for issue in result.issues],
            "metrics": dict(result.metrics),
        },
    }


def audit_config_from_json(payload: Mapping[str, Any], *, mode: str) -> AuditConfig:
    audit_payload = payload.get("audit", {})
    if not isinstance(audit_payload, Mapping):
        audit_payload = {}

    return AuditConfig(
        mode=mode,
        max_latin_chars=int(audit_payload.get("max_latin_chars", 64)),
        max_target_chars=int(audit_payload.get("max_target_chars", 64)),
        max_length_ratio=float(audit_payload.get("max_length_ratio", 4.0)),
        min_bengali_letter_ratio=float(audit_payload.get("min_bengali_letter_ratio", 0.65)),
        max_target_ascii_alpha_ratio=float(
            audit_payload.get("max_target_ascii_alpha_ratio", 0.0)
        ),
        max_latin_native_ratio=float(audit_payload.get("max_latin_native_ratio", 0.0)),
        allow_digits=parse_bool(audit_payload.get("allow_digits", True), field_name="allow_digits"),
        allow_sentence_punctuation=parse_bool(
            audit_payload.get("allow_sentence_punctuation", False),
            field_name="allow_sentence_punctuation",
        ),
        warn_on_domain_noise=parse_bool(
            audit_payload.get("warn_on_domain_noise", True),
            field_name="warn_on_domain_noise",
        ),
    )


def write_corpus_summary(
    path: Path,
    *,
    corpus_id: str,
    admissions: list[CorpusSourceAdmission],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CORPUS_SUMMARY_SCHEMA,
        "corpus_id": corpus_id,
        "source_count": len(admissions),
        "admitted_sources": sum(1 for admission in admissions if admission.admit),
        "total_rows": sum(int(admission.summary["total_rows"]) for admission in admissions),
        "accepted_rows": sum(
            int(admission.summary["accepted_rows"])
            for admission in admissions
            if admission.admit
        ),
        "sources": [admission_json(admission) for admission in admissions],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def admission_json(admission: CorpusSourceAdmission) -> dict[str, Any]:
    return {
        "source_id": admission.source_id,
        "kind": admission.kind,
        "split": admission.split,
        "mode": admission.mode,
        "admit": admission.admit,
        "input_files": admission.input_files,
        "audit_report": admission.audit_report,
        "accepted_output": admission.accepted_output,
        "summary": dict(admission.summary),
    }


def input_file_metadata(loaded: LoadedSource) -> list[dict[str, Any]]:
    files = []
    for path in loaded.files:
        files.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return files


def source_file_stem(spec: SourceSpec) -> str:
    parts = [spec.source_id]
    if spec.split:
        parts.append(spec.split)
    return ".".join(sanitize_stem_part(part) for part in parts)


def sanitize_stem_part(part: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in part)
