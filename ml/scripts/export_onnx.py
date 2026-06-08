#!/usr/bin/env python3
"""Export a trained Obadh BiGRU-CTC checkpoint to ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-time-steps", type=int, default=96)
    args = parser.parse_args()

    import torch

    from obadh_ml.export.onnx import export_bigru_ctc_onnx
    from obadh_ml.models.bigru_ctc import BigruCtcConfig, ObadhBiGruCtc

    config = BigruCtcConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    model = ObadhBiGruCtc(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    export_bigru_ctc_onnx(model, args.output, max_time_steps=args.max_time_steps)


if __name__ == "__main__":
    main()
