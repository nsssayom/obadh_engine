#!/usr/bin/env python3
"""Verify the packaged production autosuggest generator end to end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools.autosuggest.common import write_json


DEFAULT_GENERATOR_PREFIX = Path(
    "data/autosuggest/models/neural/autosuggest-generator-gru256-topk128-c64-balanced"
)
DEFAULT_REPORT = Path("target/autosuggest-full-vocab-topk128-static64-gru256-balanced-combined-report.json")
DEFAULT_NGRAM = Path("data/autosuggest/models/ngram/autosuggest-ngram-c64.bin")
DEFAULT_NGRAM_MANIFEST = Path("data/autosuggest/models/ngram/autosuggest-ngram-c64.manifest.json")
DEFAULT_CONTEXTS = ("আমি বাংলা", "আজ রবীন্দ্রনাথ", "সে এখন")
MAX_GENERATOR_SESSION_HEAP_BYTES = 64 * 1024
MAX_AUTOSUGGEST_RUNTIME_SESSION_HEAP_BYTES = 64 * 1024
MAX_AUTOSUGGEST_RUNTIME_SESSION_HEAP_LIMIT_BYTES = 256 * 1024
MAX_PERSONAL_SESSION_SNAPSHOT_LIMIT_BYTES = 256 * 1024


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    prefix = args.generator_prefix
    manifest = args.manifest or prefix.with_suffix(".manifest.json")
    onnx = args.onnx or prefix.with_suffix(".onnx")
    quantized_onnx = args.quantized_onnx or prefix.with_suffix(".int8.onnx")
    coreml = args.coreml or prefix.with_suffix(".mlpackage")

    package_command = [
        sys.executable,
        "-m",
        "tools.autosuggest.package_scorer",
        "--ngram",
        str(args.ngram),
        "--ngram-manifest",
        str(args.ngram_manifest),
        "--scorer-report",
        str(args.scorer_report),
        "--onnx",
        str(onnx),
        "--quantized-onnx",
        str(quantized_onnx),
        "--coreml",
        str(coreml),
        "--output",
        str(manifest),
        "--scored-union-profile",
        args.scored_union_profile,
        "--check",
    ]
    run(package_command, repo_root)
    manifest_doc = read_json_file(repo_root / manifest)
    quality_gate = quality_gate_from_manifest(manifest_doc)
    enforce_quality_gate(quality_gate)

    cargo_base = ["cargo", "run", "--quiet"]
    if args.release:
        cargo_base.append("--release")
    cargo_base += ["--bin", "obadh-autosuggest", "--"]

    validate = run_json(
        cargo_base
        + [
            "validate-generator",
            "--model",
            str(args.ngram),
            "--manifest",
            str(manifest),
            "--asset-root",
            str(args.asset_root),
            "--pretty",
        ],
        repo_root,
    )

    bench = run_json(
        cargo_base
        + [
            "bench",
            "--model",
            str(args.ngram),
            "--generator-manifest",
            str(manifest),
            "--mode",
            "generator-scored-union-session-personal",
            "--iterations",
            str(args.iterations),
            "--pretty",
            *context_args(args.context),
        ],
        repo_root,
    )
    runtime_budget = runtime_budget_from_bench(bench)
    enforce_bench_budget(
        bench,
        runtime_budget,
        args.max_handoff_us,
        args.max_heap_bytes,
        args.max_runtime_session_heap_bytes,
        args.max_runtime_session_heap_limit_bytes,
        args.max_personal_session_snapshot_limit_bytes,
    )

    report = {
        "package_manifest": str(manifest),
        "release_build": args.release,
        "package_check": "ok",
        "quality_gate": quality_gate,
        "compatibility": validate["compatibility"],
        "assets": validate.get("assets"),
        "handoff_benchmark": {
            "iterations": bench["iterations"],
            "mean_us": bench["mean_us"],
            "mode": bench["mode"],
            "generator_heap_bytes": bench["generator_heap_bytes"],
            "generator_heap_limit_bytes": bench["generator_heap_limit_bytes"],
            "generator_top_k_output": bench["generator_top_k_output"],
            "generator_candidate_pool": bench["generator_candidate_pool"],
            "generator_visible_candidates": bench["generator_visible_candidates"],
            "personal_heap_bytes": bench.get("personal_heap_bytes"),
            "personal_heap_limit_bytes": bench.get("personal_heap_limit_bytes"),
            "personal_session_heap_bytes": bench.get("personal_session_heap_bytes"),
            "personal_session_heap_limit_bytes": bench.get(
                "personal_session_heap_limit_bytes"
            ),
            "personal_snapshot_bytes": bench.get("personal_snapshot_bytes"),
            "personal_snapshot_limit_bytes": bench.get("personal_snapshot_limit_bytes"),
            "personal_session_snapshot_bytes": bench.get("personal_session_snapshot_bytes"),
            "personal_session_snapshot_limit_bytes": bench.get(
                "personal_session_snapshot_limit_bytes"
            ),
            "runtime_session_heap_bytes": runtime_budget["runtime_session_heap_bytes"],
            "runtime_session_heap_limit_bytes": runtime_budget[
                "runtime_session_heap_limit_bytes"
            ],
        },
        "budgets": {
            "max_handoff_us": args.max_handoff_us,
            "max_generator_heap_bytes": args.max_heap_bytes,
            "max_runtime_session_heap_bytes": args.max_runtime_session_heap_bytes,
            "max_runtime_session_heap_limit_bytes": args.max_runtime_session_heap_limit_bytes,
            "max_personal_session_snapshot_limit_bytes": (
                args.max_personal_session_snapshot_limit_bytes
            ),
        },
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--ngram", type=Path, default=DEFAULT_NGRAM)
    parser.add_argument("--ngram-manifest", type=Path, default=DEFAULT_NGRAM_MANIFEST)
    parser.add_argument("--scorer-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--generator-prefix", type=Path, default=DEFAULT_GENERATOR_PREFIX)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--quantized-onnx", type=Path)
    parser.add_argument("--coreml", type=Path)
    parser.add_argument("--asset-root", type=Path, default=Path("."))
    parser.add_argument("--scored-union-profile", default="balanced_by_mrr")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument(
        "--max-handoff-us",
        type=float,
        default=250.0,
        help="Maximum Rust handoff and scored-union latency for this package.",
    )
    parser.add_argument(
        "--max-heap-bytes",
        type=int,
        default=MAX_GENERATOR_SESSION_HEAP_BYTES,
        help="Maximum generator session heap bytes.",
    )
    parser.add_argument(
        "--max-runtime-session-heap-bytes",
        type=int,
        default=MAX_AUTOSUGGEST_RUNTIME_SESSION_HEAP_BYTES,
        help="Maximum measured generator plus personal heap for one active session.",
    )
    parser.add_argument(
        "--max-runtime-session-heap-limit-bytes",
        type=int,
        default=MAX_AUTOSUGGEST_RUNTIME_SESSION_HEAP_LIMIT_BYTES,
        help="Maximum configured generator plus personal heap limit for one active session.",
    )
    parser.add_argument(
        "--max-personal-session-snapshot-limit-bytes",
        type=int,
        default=MAX_PERSONAL_SESSION_SNAPSHOT_LIMIT_BYTES,
        help="Maximum configured compact personal snapshot size for one active session.",
    )
    parser.add_argument(
        "--release",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Rust validation/bench through cargo --release.",
    )
    parser.add_argument("--context", action="append", default=list(DEFAULT_CONTEXTS))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def context_args(contexts: list[str]) -> list[str]:
    args: list[str] = []
    for context in contexts:
        args.extend(["--context", context])
    return args


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_json(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = run(command, cwd)
    return json.loads(completed.stdout)


def read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def quality_gate_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    quality = manifest["quality"]
    selected = quality["selected_scored_union"]
    source_reports = manifest.get("source_reports", {})
    sources = {}
    for name, source in sorted(selected["eval_per_source"].items()):
        sources[name] = {
            "eligible_targets": int(source["eligible_targets"]),
            "top5_all_targets": float(source["top5_all_targets"]),
            "mrr_all_targets": float(source["mrr_all_targets"]),
            "top5_gain_vs_static": float(source["top5_all_target_gain_vs_static"]),
            "mrr_gain_vs_static": float(source["mrr_all_target_gain_vs_static"]),
        }
    return {
        "profile": selected.get("selection"),
        "selection_source": selected.get("selection_source"),
        "accepted_for_packaging": bool(selected.get("accepted_for_packaging", False)),
        "accepted_for_packaging_all_eval_sources": bool(
            selected.get("accepted_for_packaging_all_eval_sources", False)
        ),
        "heldout_targets": int(quality["heldout_targets"]),
        "eligible_targets": int(quality["eligible_targets"]),
        "static_top1_all_targets": float(quality["static_pool"]["top1_all_targets"]),
        "static_top5_all_targets": float(quality["static_pool"]["top5_all_targets"]),
        "static_mrr_all_targets": float(quality["static_pool"]["mrr_all_targets"]),
        "selected_top1_all_targets": float(selected["top1_all_targets"]),
        "selected_top5_all_targets": float(selected["top5_all_targets"]),
        "selected_mrr_all_targets": float(selected["mrr_all_targets"]),
        "selected_top5_gain_vs_static": float(selected["top5_all_target_gain_vs_static"]),
        "selected_mrr_gain_vs_static": float(selected["mrr_all_target_gain_vs_static"]),
        "neural_recall_all_targets": float(quality["neural_recall_all_targets"]),
        "union_recall_all_targets": float(quality["union_recall_all_targets"]),
        "union_recall_all_target_gain": float(quality["union_recall_all_target_gain"]),
        "per_source": sources,
        "source_report": source_reports.get("generator_export"),
    }


def enforce_quality_gate(gate: dict[str, Any]) -> None:
    if gate["selection_source"] != "split_eval":
        raise SystemExit("generator quality gate must use a split-evaluated policy")
    if not gate["accepted_for_packaging"]:
        raise SystemExit("generator quality gate is not accepted for packaging")
    if not gate["accepted_for_packaging_all_eval_sources"]:
        raise SystemExit("generator quality gate is not accepted for every eval source")
    if gate["selected_top5_gain_vs_static"] <= 0:
        raise SystemExit("generator quality gate does not improve top-5 over static")
    if gate["selected_mrr_gain_vs_static"] <= 0:
        raise SystemExit("generator quality gate does not improve MRR over static")
    if gate["union_recall_all_target_gain"] <= 0:
        raise SystemExit("generator quality gate does not improve union recall")
    if not gate["per_source"]:
        raise SystemExit("generator quality gate is missing per-source metrics")
    for source_name, source in gate["per_source"].items():
        if source["top5_gain_vs_static"] <= 0 or source["mrr_gain_vs_static"] <= 0:
            raise SystemExit(
                f"generator quality gate regresses source {source_name}"
            )


def runtime_budget_from_bench(bench: dict[str, Any]) -> dict[str, int]:
    if bench["mode"] != "generator_scored_union_session_personal":
        raise SystemExit(
            "production generator verification must use generator_scored_union_session_personal"
        )
    generator_heap = required_int(bench, "generator_heap_bytes")
    generator_heap_limit = required_int(bench, "generator_heap_limit_bytes")
    personal_heap = required_int(bench, "personal_session_heap_bytes")
    personal_heap_limit = required_int(bench, "personal_session_heap_limit_bytes")
    personal_snapshot_limit = required_int(
        bench,
        "personal_session_snapshot_limit_bytes",
    )
    return {
        "runtime_session_heap_bytes": generator_heap + personal_heap,
        "runtime_session_heap_limit_bytes": generator_heap_limit + personal_heap_limit,
        "personal_session_snapshot_limit_bytes": personal_snapshot_limit,
    }


def required_int(bench: dict[str, Any], field: str) -> int:
    if field not in bench:
        raise SystemExit(f"bench report is missing {field}")
    return int(bench[field])


def enforce_bench_budget(
    bench: dict[str, Any],
    runtime_budget: dict[str, int],
    max_handoff_us: float,
    max_generator_heap_bytes: int,
    max_runtime_session_heap_bytes: int,
    max_runtime_session_heap_limit_bytes: int,
    max_personal_session_snapshot_limit_bytes: int,
) -> None:
    mean_us = float(bench["mean_us"])
    generator_heap_bytes = int(bench["generator_heap_bytes"])
    if mean_us <= 0 or mean_us > max_handoff_us:
        raise SystemExit(
            f"generator handoff mean {mean_us:.3f} us exceeds budget {max_handoff_us:.3f} us"
        )
    if generator_heap_bytes > max_generator_heap_bytes:
        raise SystemExit(
            "generator heap "
            f"{generator_heap_bytes} bytes exceeds budget {max_generator_heap_bytes} bytes"
        )
    runtime_heap_bytes = runtime_budget["runtime_session_heap_bytes"]
    if runtime_heap_bytes > max_runtime_session_heap_bytes:
        raise SystemExit(
            "runtime session heap "
            f"{runtime_heap_bytes} bytes exceeds budget {max_runtime_session_heap_bytes} bytes"
        )
    runtime_heap_limit_bytes = runtime_budget["runtime_session_heap_limit_bytes"]
    if runtime_heap_limit_bytes > max_runtime_session_heap_limit_bytes:
        raise SystemExit(
            "runtime session heap limit "
            f"{runtime_heap_limit_bytes} bytes exceeds budget "
            f"{max_runtime_session_heap_limit_bytes} bytes"
        )
    snapshot_limit_bytes = runtime_budget["personal_session_snapshot_limit_bytes"]
    if snapshot_limit_bytes > max_personal_session_snapshot_limit_bytes:
        raise SystemExit(
            "personal session snapshot limit "
            f"{snapshot_limit_bytes} bytes exceeds budget "
            f"{max_personal_session_snapshot_limit_bytes} bytes"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
