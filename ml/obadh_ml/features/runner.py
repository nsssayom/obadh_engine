"""Batched bridge to the Rust `obadh-ml-features` binary."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def repo_root_from_ml_package() -> Path:
    return Path(__file__).resolve().parents[3]


def feature_binary_path(repo_root: Path, *, release: bool) -> Path:
    profile = "release" if release else "debug"
    return repo_root / "target" / profile / "obadh-ml-features"


def ensure_feature_binary(repo_root: Path, *, release: bool) -> Path:
    binary = feature_binary_path(repo_root, release=release)
    if binary.exists():
        return binary

    command = ["cargo", "build", "--bin", "obadh-ml-features"]
    if release:
        command.insert(2, "--release")
    subprocess.run(command, cwd=repo_root, check=True)
    return binary


def extract_feature_batch(binary: Path, inputs: Iterable[str]) -> list[dict[str, Any]]:
    input_list = list(inputs)
    if not input_list:
        return []

    process_input = "".join(f"{item}\n" for item in input_list)
    completed = subprocess.run(
        [str(binary)],
        input=process_input,
        text=True,
        capture_output=True,
        check=True,
    )

    lines = [line for line in completed.stdout.splitlines() if line]
    if len(lines) != len(input_list):
        raise RuntimeError(
            f"feature binary returned {len(lines)} rows for {len(input_list)} inputs"
        )

    return [json.loads(line) for line in lines]
