#!/usr/bin/env python3
"""Download the Dakshina dataset into ignored local storage."""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obadh_ml.data.dakshina import DAKSHINA_URL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("ml/data/raw"))
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "dakshina_dataset_v1.0.tar"

    if not archive_path.exists():
        urllib.request.urlretrieve(DAKSHINA_URL, archive_path)

    if args.extract:
        with tarfile.open(archive_path) as archive:
            safe_extract(archive, args.output_dir)


def safe_extract(archive: tarfile.TarFile, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    for member in archive.getmembers():
        member_path = (output_root / member.name).resolve()
        if not member_path.is_relative_to(output_root):
            raise RuntimeError(f"refusing to extract unsafe tar member: {member.name}")
    archive.extractall(output_root)


if __name__ == "__main__":
    main()
