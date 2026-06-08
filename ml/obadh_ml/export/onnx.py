"""ONNX export helper for the first Obadh BiGRU-CTC model."""

from __future__ import annotations

from pathlib import Path

import torch

from obadh_ml.models.bigru_ctc import ObadhBiGruCtc


def export_bigru_ctc_onnx(model: ObadhBiGruCtc, output_path: Path, *, max_time_steps: int = 96) -> None:
    model.eval()
    sample = torch.zeros((1, max_time_steps), dtype=torch.long)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        str(output_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "time"},
            "logits": {0: "batch", 1: "time"},
        },
        opset_version=17,
        dynamo=True,
    )
