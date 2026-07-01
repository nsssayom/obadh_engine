#!/usr/bin/env python3
"""Build deployment manifests for autosuggest neural model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

from tools.autosuggest.common import read_json, write_json


FNV64_OFFSET = 0xCBF29CE484222325
FNV64_PRIME = 0x00000100000001B3

MAX_GENERATOR_PARAMETERS = 10_000_000
MAX_GENERATOR_TOP_K_OUTPUT = 128
MAX_GENERATOR_CANDIDATE_POOL = 64
MAX_GENERATOR_ONNX_BYTES = 70 * 1024 * 1024
MAX_GENERATOR_QUANTIZED_ONNX_BYTES = 20 * 1024 * 1024
MAX_GENERATOR_COREML_BYTES = 20 * 1024 * 1024
MAX_GENERATOR_GRAPH_US_PER_ITEM = 1_000.0
MIN_GENERATOR_ELIGIBLE_TARGETS = 10_000
MIN_GENERATOR_SOURCE_ELIGIBLE_TARGETS = 1_000


def main() -> None:
    args = parse_args()
    ngram_manifest = read_json(args.ngram_manifest)
    scorer_report = read_json(args.scorer_report)

    ngram_bytes = args.ngram.read_bytes()
    ngram_fingerprint = fnv1a64_hex(ngram_bytes)
    if len(ngram_bytes) != ngram_manifest["artifact_bytes"]:
        raise SystemExit(
            f"ngram byte mismatch: manifest={ngram_manifest['artifact_bytes']} actual={len(ngram_bytes)}"
        )
    if ngram_fingerprint != ngram_manifest["artifact_fingerprint"]:
        raise SystemExit(
            "ngram fingerprint mismatch: "
            f"manifest={ngram_manifest['artifact_fingerprint']} actual={ngram_fingerprint}"
        )

    report_artifact = Path(scorer_report["artifact"])
    if report_artifact.is_file():
        report_bytes = report_artifact.read_bytes()
        report_fingerprint = fnv1a64_hex(report_bytes)
        if len(report_bytes) != len(ngram_bytes) or report_fingerprint != ngram_fingerprint:
            raise SystemExit(
                "scorer report was exported against a different ngram artifact: "
                f"report={report_artifact} package={args.ngram}"
            )
    elif report_artifact.name != args.ngram.name:
        raise SystemExit(
            f"scorer report was exported against {report_artifact}, not {args.ngram}"
        )

    model = scorer_report["model"]
    export = scorer_report["export"]
    quality = scorer_report["quality"]
    benchmark = scorer_report["benchmark"]
    verification = scorer_report["verification"]
    export_kind = str(export.get("kind", "candidate-scorer"))
    context_window = int(model["context_window"])

    onnx_path = args.onnx or Path(required_export_field(export, "onnx"))
    quantized_onnx_path = args.quantized_onnx or Path(
        required_export_field(export, "quantized_onnx")
    )
    coreml_path = args.coreml or Path(required_export_field(export, "coreml"))
    onnx = checked_file(onnx_path, int(required_export_field(export, "onnx_bytes")))
    quantized_onnx = checked_file(
        quantized_onnx_path,
        int(required_export_field(export, "quantized_onnx_bytes")),
    )
    coreml = checked_package(coreml_path, int(required_export_field(export, "coreml_bytes")))

    if export_kind in ("full-vocab-topk", "full-vocab-topk-scorer"):
        combined_generator = export_kind == "full-vocab-topk-scorer"
        top_k_output = int(export["top_k_output"])
        selected_quality_name = select_topk_quality_name(quality)
        selected_quality = quality[selected_quality_name]
        selected_merged_quality_name = selected_quality_name.replace("_topk", "_merged")
        selected_merged_quality = select_merged_quality_profile(
            quality,
            selected_merged_quality_name,
            visible_candidates=args.visible_candidates,
            locked_static_prefix=args.locked_prefix,
        )
        reference_scored_union_name = selected_quality_name.replace(
            "_topk",
            "_scored_union",
        )
        reference_scored_union = select_scored_union_quality(
            quality,
            reference_scored_union_name,
        )
        split_scored_union_name = f"{reference_scored_union_name}_split"
        split_scored_union = select_scored_union_split_quality(
            quality,
            split_scored_union_name,
        )
        if combined_generator and split_scored_union is None:
            raise SystemExit(
                "combined generator reports must include split-evaluated scored-union "
                f"quality: {split_scored_union_name}"
            )
        selected_scored_union, selected_scored_union_source = (
            select_scored_union_package_policy(
                reference_scored_union,
                split_scored_union,
                args.scored_union_profile,
            )
        )
        runtime_contract = {
            "token_id_dtype": "uint32",
            "onnx_input_dtype": "int64",
            "coreml_input_dtype": "int32",
            "scores_dtype": "float32",
            "batch_size": int(export["fixed_batch"]),
            "context_ids_shape": [int(export["fixed_batch"]), context_window],
            "token_ids_shape": [int(export["fixed_batch"]), top_k_output],
            "scores_shape": [int(export["fixed_batch"]), top_k_output],
            "pad_id": 0,
            "bos_id": 1,
            "unk_id": 2,
            "visible_candidates": args.visible_candidates,
        }
        if combined_generator:
            runtime_contract["candidate_ids_shape"] = [
                int(export["fixed_batch"]),
                int(export["pool_k"]),
            ]
            runtime_contract["candidate_scores_shape"] = [
                int(export["fixed_batch"]),
                int(export["pool_k"]),
            ]
            runtime_contract["scored_union_policy"] = {
                "locked_static_prefix": selected_scored_union["locked_static_prefix"],
                "static_bonus": selected_scored_union["static_bonus"],
                "static_rank_penalty": selected_scored_union["static_rank_penalty"],
                "generated_penalty": selected_scored_union["generated_penalty"],
                "overlap_bonus": selected_scored_union.get("overlap_bonus", 0.0),
                "generated_rank_penalty": selected_scored_union.get(
                    "generated_rank_penalty",
                    0.0,
                ),
                "static_log_count_scale": selected_scored_union.get(
                    "static_log_count_scale",
                    0.0,
                ),
                "static_source_bonus": selected_scored_union.get(
                    "static_source_bonus",
                    0.0,
                ),
            }
        generator = {
            "checkpoint": scorer_report["checkpoint"],
            "architecture": model["architecture"],
            "context_window": context_window,
            "embedding_dim": model["embedding_dim"],
            "hidden_dim": model["hidden_dim"],
            "parameter_count": model["parameter_count"],
            "top_k_output": top_k_output,
            "onnx": onnx,
            "quantized_onnx": quantized_onnx,
            "coreml": coreml,
            "coreml_target": export["coreml_target"],
            "coreml_precision": export["coreml_precision"],
            "coreml_compute_unit": export["coreml_compute_unit"],
            "export_kind": export_kind,
        }
        if combined_generator:
            generator["pool_k"] = int(export["pool_k"])
        quality_report = {
            "heldout_targets": quality["static_pool"]["total_targets"],
            "eligible_targets": quality["static_pool"]["eligible_targets"],
            "static_pool": pick_metrics(quality["static_pool"]),
            "selected_topk": pick_metrics(selected_quality),
            "selected_merged_visible": {
                **pick_metrics(selected_merged_quality),
                "visible_candidates": selected_merged_quality["visible_candidates"],
                "locked_static_prefix": selected_merged_quality["locked_static_prefix"],
                "top5_all_target_gain_vs_static": selected_merged_quality[
                    "top5_all_target_gain_vs_static"
                ],
                "top10_all_target_gain_vs_static": selected_merged_quality[
                    "top10_all_target_gain_vs_static"
                ],
                "mrr_all_target_gain_vs_static": selected_merged_quality[
                    "mrr_all_target_gain_vs_static"
                ],
            },
            "reference_scored_union": reference_scored_union,
            "selected_scored_union": {
                **selected_scored_union,
                "selection": args.scored_union_profile,
                "candidate_scorer": selected_scored_union.get(
                    "candidate_scorer",
                    reference_scored_union["candidate_scorer"],
                ),
                "selection_source": selected_scored_union_source,
            },
            "static_pool_recall_all_targets": quality["static_pool"][
                "pool_recall_all_targets"
            ],
            "neural_recall_all_targets": selected_quality["neural_recall_all_targets"],
            "union_recall_all_targets": selected_quality["union_recall_all_targets"],
            "union_recall_all_target_gain": selected_quality[
                "absolute_union_gain_all_targets"
            ],
        }
        if split_scored_union is not None:
            quality_report["split_scored_union"] = split_scored_union
        source_reports = {
            "generator_export": str(args.scorer_report),
            "ngram_manifest": str(args.ngram_manifest),
            "selected_quality": selected_quality_name,
            "selected_merged_quality": selected_merged_quality_name,
            "reference_scored_union_quality": reference_scored_union_name,
            "selected_scored_union_profile": args.scored_union_profile,
            "selected_scored_union_source": selected_scored_union_source,
        }
        if split_scored_union is not None:
            source_reports["split_scored_union_quality"] = split_scored_union_name
        manifest = {
            "artifact": "obadh-autosuggest-generator-package",
            "version": 1,
            "runtime_role": "next_word_candidate_generate",
            "runtime_contract": runtime_contract,
            "ngram": {
                "path": str(args.ngram),
                "manifest": str(args.ngram_manifest),
                "bytes": len(ngram_bytes),
                "artifact_fingerprint": ngram_fingerprint,
                "vocab_size": ngram_manifest["vocab_size"],
                "vocab_fingerprint": ngram_manifest["vocab_fingerprint"],
                "max_candidates_per_prefix": ngram_manifest["max_candidates_per_prefix"],
                "candidate_rows": ngram_manifest["candidate_rows"],
                "candidate_record_len": ngram_manifest["candidate_record_len"],
            },
            "generator": generator,
            "quality": quality_report,
            "verification": verification,
            "benchmark": {
                "onnx_mean_us_per_item": benchmark["onnx"]["mean_us_per_item"],
                "quantized_onnx_mean_us_per_item": benchmark["quantized_onnx"][
                    "mean_us_per_item"
                ],
                "coreml_mean_us_per_item": benchmark["coreml"]["mean_us_per_item"],
            },
            "source_reports": source_reports,
        }
        validate_generator_manifest(manifest)
        write_or_check_manifest(args.output, manifest, args.check)
        return

    if export_kind != "candidate-scorer":
        raise SystemExit(f"unsupported export kind: {export_kind}")

    pool_k = int(export["pool_k"])
    selected_quality = select_quality_profile(
        quality["quantized_rerank_locked_first"],
        rank_penalty=args.rank_penalty,
        locked_prefix=args.locked_prefix,
    )

    manifest = {
        "artifact": "obadh-autosuggest-scorer-package",
        "version": 1,
        "runtime_role": "next_word_candidate_rerank",
        "runtime_contract": {
            "token_id_dtype": "uint32",
            "onnx_input_dtype": "int64",
            "coreml_input_dtype": "int32",
            "scores_dtype": "float32",
            "batch_size": int(export["fixed_batch"]),
            "context_ids_shape": [int(export["fixed_batch"]), context_window],
            "candidate_ids_shape": [int(export["fixed_batch"]), pool_k],
            "scores_shape": [int(export["fixed_batch"]), pool_k],
            "pad_id": 0,
            "bos_id": 1,
            "unk_id": 2,
            "locked_prefix": args.locked_prefix,
            "rank_penalty": args.rank_penalty,
            "visible_candidates": args.visible_candidates,
        },
        "ngram": {
            "path": str(args.ngram),
            "manifest": str(args.ngram_manifest),
            "bytes": len(ngram_bytes),
            "artifact_fingerprint": ngram_fingerprint,
            "vocab_size": ngram_manifest["vocab_size"],
            "vocab_fingerprint": ngram_manifest["vocab_fingerprint"],
            "max_candidates_per_prefix": ngram_manifest["max_candidates_per_prefix"],
            "candidate_rows": ngram_manifest["candidate_rows"],
            "candidate_record_len": ngram_manifest["candidate_record_len"],
        },
        "scorer": {
            "checkpoint": scorer_report["checkpoint"],
            "architecture": model["architecture"],
            "context_window": context_window,
            "embedding_dim": model["embedding_dim"],
            "hidden_dim": model["hidden_dim"],
            "parameter_count": model["parameter_count"],
            "pool_k": pool_k,
            "onnx": onnx,
            "quantized_onnx": quantized_onnx,
            "coreml": coreml,
            "coreml_target": export["coreml_target"],
            "coreml_precision": export["coreml_precision"],
            "coreml_compute_unit": export["coreml_compute_unit"],
        },
        "quality": {
            "heldout_targets": quality["static_pool"]["total_targets"],
            "eligible_targets": quality["static_pool"]["eligible_targets"],
            "static_pool": pick_metrics(quality["static_pool"]),
            "selected_quantized_locked_first": pick_metrics(selected_quality),
            "top5_all_target_gain": selected_quality["top5_all_targets"]
            - quality["static_pool"]["top5_all_targets"],
            "top10_all_target_gain": selected_quality["top10_all_targets"]
            - quality["static_pool"]["top10_all_targets"],
            "pool_recall_all_targets": quality["static_pool"]["pool_recall_all_targets"],
        },
        "verification": verification,
        "benchmark": {
            "onnx_mean_us_per_item": benchmark["onnx"]["mean_us_per_item"],
            "quantized_onnx_mean_us_per_item": benchmark["quantized_onnx"]["mean_us_per_item"],
            "coreml_mean_us_per_item": benchmark["coreml"]["mean_us_per_item"],
        },
        "source_reports": {
            "scorer_export": str(args.scorer_report),
            "ngram_manifest": str(args.ngram_manifest),
        },
    }
    validate_manifest(manifest)
    write_or_check_manifest(args.output, manifest, args.check)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ngram",
        type=Path,
        default=Path("target/autosuggest-full-ngram-4gram-count-c64-b8t8f14-compact.bin"),
    )
    parser.add_argument(
        "--ngram-manifest",
        type=Path,
        default=Path("target/autosuggest-full-ngram-4gram-count-c64-b8t8f14-compact.manifest.json"),
    )
    parser.add_argument(
        "--scorer-report",
        type=Path,
        default=Path("target/autosuggest-next-word-lm-gru256-c64-9epoch-export-coreml-report.json"),
        help=(
            "Export report from export_next_word_lm.py. Supports candidate-scorer, "
            "full-vocab-topk, and full-vocab-topk-scorer reports."
        ),
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Packaged fp32 ONNX path to verify and record. Defaults to the export report path.",
    )
    parser.add_argument(
        "--quantized-onnx",
        type=Path,
        default=None,
        help="Packaged INT8 ONNX path to verify and record. Defaults to the export report path.",
    )
    parser.add_argument(
        "--coreml",
        type=Path,
        default=None,
        help="Packaged Core ML mlpackage path to verify and record. Defaults to the export report path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("target/autosuggest-c64-gru256-candidate-scorer.manifest.json"),
    )
    parser.add_argument("--rank-penalty", type=float, default=0.25)
    parser.add_argument("--locked-prefix", type=int, default=1)
    parser.add_argument("--visible-candidates", type=int, default=5)
    parser.add_argument(
        "--scored-union-profile",
        choices=("best_by_mrr", "best_by_top5", "balanced_by_mrr", "balanced_by_top5"),
        default="best_by_mrr",
        help="Scored-union profile to package as the deployable generator policy.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that --output already matches the generated manifest.",
    )
    return parser.parse_args()


def fnv1a64_hex(data: bytes) -> str:
    value = FNV64_OFFSET
    for byte in data:
        value = ((value ^ byte) * FNV64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    return f"{value or 1:016x}"


def checked_file(path: Path, expected_bytes: int) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing model file: {path}")
    actual = path.stat().st_size
    if actual != expected_bytes:
        raise SystemExit(f"model byte mismatch for {path}: report={expected_bytes} actual={actual}")
    return {
        "path": str(path),
        "bytes": actual,
        "sha256": sha256_path(path),
    }


def checked_package(path: Path, expected_bytes: int) -> dict[str, Any]:
    if not path.is_dir():
        raise SystemExit(f"missing Core ML package directory: {path}")
    actual = package_size(path)
    if actual != expected_bytes:
        raise SystemExit(f"Core ML package byte mismatch for {path}: report={expected_bytes} actual={actual}")
    return {
        "path": str(path),
        "bytes": actual,
        "sha256": sha256_tree(path),
    }


def write_or_check_manifest(output: Path, manifest: dict[str, Any], check: bool) -> None:
    if check:
        existing = read_json(output)
        if existing != manifest:
            raise SystemExit(f"deployment manifest is stale: {output}")
    else:
        write_json(output, manifest)


def package_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def select_quality_profile(profiles: list[dict[str, Any]], rank_penalty: float, locked_prefix: int) -> dict[str, Any]:
    lock_first = locked_prefix > 0
    for profile in profiles:
        if profile.get("lock_first") is lock_first and abs(float(profile["rank_penalty"]) - rank_penalty) < 1e-9:
            return profile
    raise SystemExit(
        f"missing quality profile for lock_first={lock_first} rank_penalty={rank_penalty}"
    )


def select_topk_quality_name(quality: dict[str, Any]) -> str:
    for name in ("coreml_topk", "quantized_topk", "onnx_topk"):
        if name in quality:
            return name
    raise SystemExit("full-vocab top-k report has no top-k quality profile")


def select_merged_quality_profile(
    quality: dict[str, Any],
    profile_name: str,
    visible_candidates: int,
    locked_static_prefix: int,
) -> dict[str, Any]:
    profiles = quality.get(profile_name)
    if not profiles:
        raise SystemExit(f"full-vocab top-k report has no merged quality profile: {profile_name}")
    for profile in profiles:
        if (
            int(profile["visible_candidates"]) == visible_candidates
            and int(profile["locked_static_prefix"]) == locked_static_prefix
        ):
            return profile
    raise SystemExit(
        "missing merged quality profile for "
        f"visible_candidates={visible_candidates} locked_static_prefix={locked_static_prefix}"
    )


def select_scored_union_quality(quality: dict[str, Any], profile_name: str) -> dict[str, Any]:
    profile = quality.get(profile_name)
    if not profile:
        raise SystemExit(f"full-vocab top-k report has no scored union profile: {profile_name}")
    return {
        "candidate_scorer": profile["candidate_scorer"],
        "union_candidate_count_max": profile["union_candidate_count_max"],
        "union_candidate_count_mean": profile["union_candidate_count_mean"],
        "best_by_top5": pick_scored_union_metrics(profile["best_by_top5"]),
        "best_by_mrr": pick_scored_union_metrics(profile["best_by_mrr"]),
    }


def select_scored_union_split_quality(
    quality: dict[str, Any],
    profile_name: str,
) -> dict[str, Any] | None:
    profile = quality.get(profile_name)
    if not profile:
        return None
    if not profile.get("enabled", False):
        raise SystemExit(
            f"scored union split profile is disabled: {profile_name} "
            f"({profile.get('reason', 'no reason recorded')})"
        )
    output = {
        "candidate_scorer": profile["candidate_scorer"],
        "union_candidate_count_max": profile["union_candidate_count_max"],
        "union_candidate_count_mean": profile["union_candidate_count_mean"],
        "selection_eval_mod": profile["selection_eval_mod"],
        "selection_eval_remainder": profile["selection_eval_remainder"],
        "selection_size": profile["selection_size"],
        "eval_size": profile["eval_size"],
        "selection_static_pool": profile["selection_static_pool"],
        "eval_static_pool": profile["eval_static_pool"],
        "best_by_top5": pick_split_scored_union_metrics(profile["selected_by_top5"]),
        "best_by_mrr": pick_split_scored_union_metrics(profile["selected_by_mrr"]),
    }
    if "selected_by_balanced_top5" in profile:
        output["balanced_by_top5"] = pick_split_scored_union_metrics(
            profile["selected_by_balanced_top5"]
        )
    if "selected_by_balanced_mrr" in profile:
        output["balanced_by_mrr"] = pick_split_scored_union_metrics(
            profile["selected_by_balanced_mrr"]
        )
    return output


def select_scored_union_package_policy(
    reference_scored_union: dict[str, Any],
    split_scored_union: dict[str, Any] | None,
    profile_name: str,
) -> tuple[dict[str, Any], str]:
    if split_scored_union is None:
        if profile_name not in reference_scored_union:
            raise SystemExit(f"scored union profile is not available: {profile_name}")
        return reference_scored_union[profile_name], "same_sample_grid"
    if profile_name not in split_scored_union:
        raise SystemExit(f"split-evaluated scored union profile is not available: {profile_name}")
    selected = split_scored_union[profile_name]
    if not selected.get("accepted_for_packaging", False):
        raise SystemExit(
            "split-evaluated scored union does not improve both top-5 and MRR for "
            f"{profile_name}"
        )
    if profile_name.startswith("balanced_by") and not selected.get(
        "accepted_for_packaging_all_eval_sources",
        False,
    ):
        raise SystemExit(
            "source-balanced scored union still regresses at least one eval source for "
            f"{profile_name}"
        )
    return selected, "split_eval"


def required_export_field(export: dict[str, Any], field: str) -> Any:
    if field not in export:
        raise SystemExit(
            f"scorer report is missing export.{field}; deployment packaging requires "
            "a full export with ONNX, quantized ONNX, and Core ML artifacts"
        )
    return export[field]


def pick_split_scored_union_metrics(selection: dict[str, Any]) -> dict[str, Any]:
    selected = pick_scored_union_metrics(selection["eval_profile"])
    selected["accepted_for_packaging"] = bool(selection["accepted_for_packaging"])
    if "accepted_for_packaging_all_eval_sources" in selection:
        selected["accepted_for_packaging_all_eval_sources"] = bool(
            selection["accepted_for_packaging_all_eval_sources"]
        )
    if "eval_per_source" in selection:
        selected["eval_per_source"] = selection["eval_per_source"]
    selected["selection_top5_all_targets"] = selection["selection_profile"][
        "top5_all_targets"
    ]
    selected["selection_mrr_all_targets"] = selection["selection_profile"][
        "mrr_all_targets"
    ]
    selected["selection_top5_all_target_gain_vs_static"] = selection[
        "selection_profile"
    ]["top5_all_target_gain_vs_static"]
    selected["selection_mrr_all_target_gain_vs_static"] = selection[
        "selection_profile"
    ]["mrr_all_target_gain_vs_static"]
    return selected


def pick_scored_union_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        **pick_metrics(profile),
        "locked_static_prefix": profile["locked_static_prefix"],
        "static_bonus": profile["static_bonus"],
        "static_rank_penalty": profile["static_rank_penalty"],
        "generated_penalty": profile["generated_penalty"],
        "overlap_bonus": profile.get("overlap_bonus", 0.0),
        "generated_rank_penalty": profile.get("generated_rank_penalty", 0.0),
        "static_log_count_scale": profile.get("static_log_count_scale", 0.0),
        "static_source_bonus": profile.get("static_source_bonus", 0.0),
        "top5_all_target_gain_vs_static": profile["top5_all_target_gain_vs_static"],
        "top10_all_target_gain_vs_static": profile["top10_all_target_gain_vs_static"],
        "mrr_all_target_gain_vs_static": profile["mrr_all_target_gain_vs_static"],
    }


def pick_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "top1_all_targets": profile["top1_all_targets"],
        "top5_all_targets": profile["top5_all_targets"],
        "top10_all_targets": profile["top10_all_targets"],
        "mrr_all_targets": profile["mrr_all_targets"],
    }


def require_positive_bounded_us(field: str, value: object) -> None:
    measured = float(value)
    if (
        not math.isfinite(measured)
        or measured <= 0
        or measured > MAX_GENERATOR_GRAPH_US_PER_ITEM
    ):
        raise SystemExit(
            f"{field} must be finite, positive, and <= {MAX_GENERATOR_GRAPH_US_PER_ITEM}"
        )


def validate_quality_metrics(field: str, metrics: dict[str, Any]) -> None:
    top1 = float(metrics["top1_all_targets"])
    top5 = float(metrics["top5_all_targets"])
    top10 = float(metrics["top10_all_targets"])
    mrr = float(metrics["mrr_all_targets"])
    if not all(is_probability(value) for value in (top1, top5, top10, mrr)):
        raise SystemExit(f"{field} metrics must be finite probabilities")
    if top1 > top5 or top5 > top10:
        raise SystemExit(f"{field} top-k metrics must be monotonic")
    if mrr < top1 or mrr > top5:
        raise SystemExit(f"{field} MRR must stay between top-1 and top-5")


def validate_source_balanced_quality(selected_scored_union: dict[str, Any]) -> None:
    per_source = selected_scored_union.get("eval_per_source")
    if not per_source:
        raise SystemExit("selected scored union must include per-source eval metrics")
    for source_name, source in sorted(per_source.items()):
        if int(source["eligible_targets"]) < MIN_GENERATOR_SOURCE_ELIGIBLE_TARGETS:
            raise SystemExit(
                f"{source_name} source gate is too small: "
                f"{source['eligible_targets']} < {MIN_GENERATOR_SOURCE_ELIGIBLE_TARGETS}"
            )
        validate_quality_metrics(f"{source_name} source metrics", source)
        top5_gain = float(source["top5_all_target_gain_vs_static"])
        mrr_gain = float(source["mrr_all_target_gain_vs_static"])
        if not math.isfinite(top5_gain) or not math.isfinite(mrr_gain):
            raise SystemExit(f"{source_name} source gains must be finite")
        if top5_gain <= 0 or mrr_gain <= 0:
            raise SystemExit(f"{source_name} source top-5 and MRR gains must be positive")


def is_probability(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


def validate_generator_manifest(manifest: dict[str, Any]) -> None:
    contract = manifest["runtime_contract"]
    generator = manifest["generator"]
    quality = manifest["quality"]
    benchmark = manifest["benchmark"]
    if manifest["artifact"] != "obadh-autosuggest-generator-package":
        raise SystemExit("generator artifact kind is invalid")
    if manifest["runtime_role"] != "next_word_candidate_generate":
        raise SystemExit("generator runtime role is invalid")
    if contract["token_id_dtype"] != "uint32":
        raise SystemExit("token ID dtype must be uint32")
    if contract["onnx_input_dtype"] != "int64":
        raise SystemExit("ONNX input dtype must be int64")
    if contract["coreml_input_dtype"] != "int32":
        raise SystemExit("Core ML input dtype must be int32")
    if contract["scores_dtype"] != "float32":
        raise SystemExit("score dtype must be float32")
    if contract["batch_size"] != 1:
        raise SystemExit("generator batch size must be fixed to 1")
    if contract["context_ids_shape"][1] != generator["context_window"]:
        raise SystemExit("context shape does not match generator context window")
    if contract["token_ids_shape"][1] != generator["top_k_output"]:
        raise SystemExit("token output shape does not match generator top-k")
    if contract["scores_shape"][1] != generator["top_k_output"]:
        raise SystemExit("score output shape does not match generator top-k")
    if generator["architecture"] != "gru":
        raise SystemExit("production generator architecture must be gru")
    if generator["coreml_target"] != "ios17":
        raise SystemExit("production generator Core ML target must be ios17")
    if generator["coreml_precision"] != "float16":
        raise SystemExit("production generator Core ML precision must be float16")
    if generator["coreml_compute_unit"] != "cpu_and_ne":
        raise SystemExit("production generator Core ML compute unit must be cpu_and_ne")
    if int(generator["parameter_count"]) > MAX_GENERATOR_PARAMETERS:
        raise SystemExit(
            f"generator parameter count exceeds mobile budget {MAX_GENERATOR_PARAMETERS}"
        )
    if int(generator["top_k_output"]) > MAX_GENERATOR_TOP_K_OUTPUT:
        raise SystemExit(
            f"generator top-k exceeds mobile budget {MAX_GENERATOR_TOP_K_OUTPUT}"
        )
    if int(generator["onnx"]["bytes"]) > MAX_GENERATOR_ONNX_BYTES:
        raise SystemExit(f"fp32 ONNX exceeds mobile budget {MAX_GENERATOR_ONNX_BYTES}")
    if int(generator["quantized_onnx"]["bytes"]) > MAX_GENERATOR_QUANTIZED_ONNX_BYTES:
        raise SystemExit(
            f"INT8 ONNX exceeds mobile budget {MAX_GENERATOR_QUANTIZED_ONNX_BYTES}"
        )
    if int(generator["coreml"]["bytes"]) > MAX_GENERATOR_COREML_BYTES:
        raise SystemExit(f"Core ML package exceeds mobile budget {MAX_GENERATOR_COREML_BYTES}")
    require_positive_bounded_us(
        "benchmark.onnx_mean_us_per_item",
        benchmark["onnx_mean_us_per_item"],
    )
    require_positive_bounded_us(
        "benchmark.quantized_onnx_mean_us_per_item",
        benchmark["quantized_onnx_mean_us_per_item"],
    )
    require_positive_bounded_us(
        "benchmark.coreml_mean_us_per_item",
        benchmark["coreml_mean_us_per_item"],
    )
    if int(quality["eligible_targets"]) < MIN_GENERATOR_ELIGIBLE_TARGETS:
        raise SystemExit(
            "generator held-out gate is too small: "
            f"{quality['eligible_targets']} < {MIN_GENERATOR_ELIGIBLE_TARGETS}"
        )
    validate_quality_metrics("quality.static_pool", quality["static_pool"])
    validate_quality_metrics("quality.selected_topk", quality["selected_topk"])
    if "candidate_ids_shape" in contract or "candidate_scores_shape" in contract:
        if "candidate_ids_shape" not in contract or "candidate_scores_shape" not in contract:
            raise SystemExit("combined generator must include both candidate input and score shapes")
        if "pool_k" not in generator:
            raise SystemExit("combined generator manifest is missing pool_k")
        if contract["candidate_ids_shape"][0] != contract["batch_size"]:
            raise SystemExit("candidate input batch shape does not match fixed batch")
        if contract["candidate_scores_shape"][0] != contract["batch_size"]:
            raise SystemExit("candidate score batch shape does not match fixed batch")
        if contract["candidate_ids_shape"][1] != generator["pool_k"]:
            raise SystemExit("candidate input shape does not match generator pool")
        if contract["candidate_scores_shape"][1] != generator["pool_k"]:
            raise SystemExit("candidate score shape does not match generator pool")
        if int(generator["pool_k"]) > MAX_GENERATOR_CANDIDATE_POOL:
            raise SystemExit(
                f"generator candidate pool exceeds mobile budget {MAX_GENERATOR_CANDIDATE_POOL}"
            )
        if generator.get("export_kind") != "full-vocab-topk-scorer":
            raise SystemExit("combined generator export kind must be full-vocab-topk-scorer")
        if "scored_union_policy" not in contract:
            raise SystemExit("combined generator manifest is missing scored_union_policy")
        selected_scored_union = quality.get("selected_scored_union")
        if not selected_scored_union:
            raise SystemExit("combined generator manifest is missing selected scored union quality")
        validate_quality_metrics("quality.selected_scored_union", selected_scored_union)
        policy = contract["scored_union_policy"]
        for key in (
            "locked_static_prefix",
            "static_bonus",
            "static_rank_penalty",
            "generated_penalty",
            "overlap_bonus",
            "generated_rank_penalty",
            "static_log_count_scale",
            "static_source_bonus",
        ):
            if key not in policy:
                raise SystemExit(f"scored union policy is missing {key}")
            if key not in selected_scored_union:
                raise SystemExit(f"selected scored union quality is missing {key}")
            if abs(float(policy[key]) - float(selected_scored_union[key])) > 1e-6:
                raise SystemExit(f"scored union policy does not match selected quality for {key}")
        if selected_scored_union["top5_all_target_gain_vs_static"] <= 0:
            raise SystemExit("selected scored union does not improve top-5")
        if selected_scored_union["mrr_all_target_gain_vs_static"] <= 0:
            raise SystemExit("selected scored union does not improve MRR")
        if not selected_scored_union.get("accepted_for_packaging", False):
            raise SystemExit("selected scored union is not accepted for packaging")
        if not selected_scored_union.get("accepted_for_packaging_all_eval_sources", False):
            raise SystemExit("selected scored union fails source-balanced packaging")
        validate_source_balanced_quality(selected_scored_union)
        if "split_scored_union" in quality:
            if selected_scored_union.get("selection_source") != "split_eval":
                raise SystemExit("split-scored manifest must package the split-eval policy")
            if not selected_scored_union.get("accepted_for_packaging", False):
                raise SystemExit("selected split-eval scored union is not packageable")
    if contract["visible_candidates"] > generator["top_k_output"]:
        raise SystemExit("visible candidate count exceeds generated top-k")
    if quality["union_recall_all_target_gain"] <= 0:
        raise SystemExit("selected generator does not improve union recall")
    merged = quality.get("selected_merged_visible")
    if not merged:
        raise SystemExit("selected generator manifest is missing visible merge quality")
    validate_quality_metrics("quality.selected_merged_visible", merged)
    if merged["visible_candidates"] != contract["visible_candidates"]:
        raise SystemExit("visible merge quality does not match runtime visible candidate count")
    scored_union = quality.get("reference_scored_union")
    if not scored_union:
        raise SystemExit("selected generator manifest is missing scored union reference quality")
    if scored_union["best_by_top5"]["top5_all_target_gain_vs_static"] <= 0:
        raise SystemExit("scored union top-5 reference does not improve top-5")
    if scored_union["best_by_mrr"]["mrr_all_target_gain_vs_static"] <= 0:
        raise SystemExit("scored union MRR reference does not improve MRR")


def validate_manifest(manifest: dict[str, Any]) -> None:
    contract = manifest["runtime_contract"]
    if contract["token_id_dtype"] != "uint32":
        raise SystemExit("token ID dtype must be uint32")
    if contract["onnx_input_dtype"] != "int64":
        raise SystemExit("ONNX input dtype must be int64")
    if contract["coreml_input_dtype"] != "int32":
        raise SystemExit("Core ML input dtype must be int32")
    if contract["scores_dtype"] != "float32":
        raise SystemExit("score dtype must be float32")
    if contract["context_ids_shape"][1] != manifest["scorer"]["context_window"]:
        raise SystemExit("context shape does not match scorer context window")
    if contract["candidate_ids_shape"][1] != manifest["scorer"]["pool_k"]:
        raise SystemExit("candidate shape does not match scorer pool")
    if manifest["ngram"]["max_candidates_per_prefix"] < manifest["scorer"]["pool_k"]:
        raise SystemExit("ngram pool is smaller than scorer pool")
    if contract["visible_candidates"] > manifest["scorer"]["pool_k"]:
        raise SystemExit("visible candidate count exceeds scorer pool")
    if manifest["quality"]["selected_quantized_locked_first"]["top5_all_targets"] <= manifest["quality"]["static_pool"]["top5_all_targets"]:
        raise SystemExit("selected scorer does not improve top-5 all-target accuracy")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
