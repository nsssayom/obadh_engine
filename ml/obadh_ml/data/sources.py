"""Source-specific readers normalized to Obadh pair records."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obadh_ml.data.audit import PairRecord
from obadh_ml.data.dakshina import bn_lexicon_path, iter_bn_lexicon

EXCLUDED_SOURCE_IDS = {"aksharantar"}
SUPPORTED_SOURCE_KINDS = {
    "dakshina_bn",
    "banglatlit",
    "sknahin",
    "generic_pairs",
    "kaggle_manual",
}


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    kind: str
    path: Path
    split: str | None
    mode: str
    admit: bool
    fmt: str | None = None
    latin_column: str | None = None
    target_column: str | None = None
    weight_column: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LoadedSource:
    spec: SourceSpec
    records: list[PairRecord]
    files: list[Path]


def source_spec_from_json(payload: Mapping[str, Any]) -> SourceSpec:
    source_id = str(payload["id"])
    kind = str(payload["kind"])
    assert_source_allowed(source_id, kind)

    split = optional_str(payload.get("split"))
    return SourceSpec(
        source_id=source_id,
        kind=kind,
        path=Path(str(payload["path"])),
        split=split,
        mode=str(payload.get("mode", "word")),
        admit=parse_bool(payload.get("admit", default_admit(kind, split)), field_name="admit"),
        fmt=optional_str(payload.get("format")),
        latin_column=optional_str(payload.get("latin_column")),
        target_column=optional_str(payload.get("target_column")),
        weight_column=optional_str(payload.get("weight_column")),
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else None,
    )


def load_source(spec: SourceSpec) -> LoadedSource:
    assert_source_allowed(spec.source_id, spec.kind)
    if spec.kind == "dakshina_bn":
        return load_dakshina_bn(spec)
    if spec.kind == "banglatlit":
        return load_columnar_source(
            spec,
            default_latin_column="text_transliterated",
            default_target_column="text_bengali",
        )
    if spec.kind == "sknahin":
        return load_columnar_source(spec, default_latin_column="rm", default_target_column="bn")
    if spec.kind in {"generic_pairs", "kaggle_manual"}:
        return load_columnar_source(spec, default_latin_column=None, default_target_column=None)
    raise ValueError(f"unsupported source kind: {spec.kind}")


def assert_source_allowed(source_id: str, kind: str) -> None:
    normalized_id = source_id.casefold()
    normalized_kind = kind.casefold()
    if normalized_id in EXCLUDED_SOURCE_IDS or normalized_kind in EXCLUDED_SOURCE_IDS:
        raise ValueError(f"excluded source is not allowed: {source_id}/{kind}")
    if kind not in SUPPORTED_SOURCE_KINDS:
        raise ValueError(f"unsupported source kind: {kind}")


def default_admit(kind: str, split: str | None) -> bool:
    if kind == "dakshina_bn" and split in {"dev", "test"}:
        return False
    return True


def load_dakshina_bn(spec: SourceSpec) -> LoadedSource:
    if spec.split is None:
        raise ValueError("dakshina_bn source requires split")
    if spec.admit and spec.split in {"dev", "test"}:
        raise ValueError("Dakshina dev/test splits are benchmark-only and cannot be admitted")

    file_path = bn_lexicon_path(spec.path, spec.split)
    records = [
        PairRecord(
            source_id=spec.source_id,
            row_id=str(index),
            latin=example.latin,
            target=example.target,
            split=spec.split,
            metadata=source_metadata(spec, {"weight": example.weight, "source": example.source}),
        )
        for index, example in enumerate(iter_bn_lexicon(spec.path, spec.split), start=1)
    ]
    return LoadedSource(spec=spec, records=records, files=[file_path])


def load_columnar_source(
    spec: SourceSpec,
    *,
    default_latin_column: str | None,
    default_target_column: str | None,
) -> LoadedSource:
    latin_column = spec.latin_column or default_latin_column
    target_column = spec.target_column or default_target_column
    if latin_column is None or target_column is None:
        raise ValueError(f"{spec.kind} source requires latin_column and target_column")

    records = []
    for index, row in enumerate(read_rows(spec.path, spec.fmt), start=1):
        if latin_column not in row or target_column not in row:
            raise KeyError(
                f"{spec.path}: missing required columns {latin_column!r}/{target_column!r}"
            )
        if row.get(latin_column) is None or row.get(target_column) is None:
            continue
        records.append(
            PairRecord(
                source_id=spec.source_id,
                row_id=str(row.get("id", index)),
                latin=str(row[latin_column]),
                target=str(row[target_column]),
                split=spec.split,
                metadata=source_metadata(
                    spec,
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {latin_column, target_column}
                    },
                ),
            )
        )
    return LoadedSource(spec=spec, records=records, files=[spec.path])


def read_rows(path: Path, fmt: str | None) -> Iterator[dict[str, Any]]:
    resolved_format = (fmt or infer_format(path)).lower()
    if resolved_format == "csv":
        yield from read_delimited(path, ",")
    elif resolved_format == "tsv":
        yield from read_delimited(path, "\t")
    elif resolved_format == "jsonl":
        yield from read_jsonl(path)
    elif resolved_format == "json":
        yield from read_json(path)
    elif resolved_format == "parquet":
        yield from read_parquet(path)
    else:
        raise ValueError(f"unsupported data format: {resolved_format}")


def read_delimited(path: Path, delimiter: str) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        yield from reader


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_json(path: Path) -> Iterator[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        for row in payload["data"]:
            if isinstance(row, dict):
                yield row
    else:
        raise ValueError("JSON source must be a list of objects or an object with a data list")


def read_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("reading parquet sources requires pandas and pyarrow") from error

    dataframe = pd.read_parquet(path)
    for row in dataframe.to_dict(orient="records"):
        yield row


def infer_format(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    if path.suffix.lower() in {".csv", ".tsv", ".json", ".parquet"}:
        return path.suffix.lower().lstrip(".")
    raise ValueError(f"cannot infer data format from path: {path}")


def source_metadata(spec: SourceSpec, row_metadata: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if spec.metadata:
        metadata.update(spec.metadata)
    metadata.update(row_metadata)
    return metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")
