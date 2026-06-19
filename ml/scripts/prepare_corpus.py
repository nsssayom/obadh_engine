#!/usr/bin/env python3
"""Prepare audited corpus shards from a source manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obadh_ml.data.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    admit_source,
    audit_config_from_json,
    write_corpus_summary,
)
from obadh_ml.data.sources import source_spec_from_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    require_manifest(manifest)
    corpus_id = str(manifest["corpus_id"])

    admissions = []
    for source_payload in manifest["sources"]:
        spec = source_spec_from_json(source_payload)
        audit_config = audit_config_from_json(source_payload, mode=spec.mode)
        admissions.append(
            admit_source(
                spec,
                args.output_dir,
                corpus_id=corpus_id,
                audit_config=audit_config,
            )
        )

    summary_path = args.output_dir / f"{corpus_id}.admission.json"
    write_corpus_summary(summary_path, corpus_id=corpus_id, admissions=admissions)
    print(summary_path)


def require_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != CORPUS_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported corpus manifest schema: {manifest.get('schema')!r}")
    if not manifest.get("corpus_id"):
        raise ValueError("manifest requires corpus_id")
    if not isinstance(manifest.get("sources"), list):
        raise ValueError("manifest requires a sources list")


if __name__ == "__main__":
    main()
