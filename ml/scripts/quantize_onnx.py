#!/usr/bin/env python3
"""Apply ONNX Runtime dynamic INT8 quantization."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(args.input), str(args.output), weight_type=QuantType.QInt8)


if __name__ == "__main__":
    main()
