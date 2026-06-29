#!/usr/bin/env python3
"""Export the trained autosuggest model to portable inference formats."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import save_manifest
from .model import AutosuggestModelConfig, NextWordTransformer


def export_onnx(checkpoint_path: Path, output_path: Path, opset: int) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = AutosuggestModelConfig(**checkpoint["config"])
    model = NextWordTransformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.zeros((1, config.context_length), dtype=torch.long)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "format": "onnx",
        "opset": opset,
        "config": config.to_dict(),
        "bytes": output_path.stat().st_size,
    }
    save_manifest(output_path.with_suffix(".manifest.json"), **report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("data/autosuggest/models/neural/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/autosuggest/models/neural/autosuggest.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    print(export_onnx(args.checkpoint, args.output, args.opset))


if __name__ == "__main__":
    main()
