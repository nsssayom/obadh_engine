#!/usr/bin/env python3
"""PyTorch datasets for Obadh autosuggest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import PAD_ID


@dataclass(frozen=True)
class EncodedDatasetInfo:
    token_count: int
    sentence_count: int
    target_positions: int
    vocab_size: int


class RandomContextDataset(Dataset):
    """Random-access next-word examples from encoded sentence arrays."""

    def __init__(
        self,
        dataset_dir: Path,
        context_length: int,
        sample_count: int,
        seed: int,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.context_length = context_length
        self.sample_count = sample_count
        self.seed = seed
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        self.info = EncodedDatasetInfo(
            token_count=int(manifest["token_count"]),
            sentence_count=int(manifest["sentence_count"]),
            target_positions=int(manifest["target_positions"]),
            vocab_size=int(manifest["vocab_size"]),
        )
        self.tokens = np.memmap(
            dataset_dir / "tokens.u32",
            dtype="<u4",
            mode="r",
            shape=(self.info.token_count,),
        )
        self.offsets = np.memmap(
            dataset_dir / "sentences.u64",
            dtype="<u8",
            mode="r",
            shape=(self.info.sentence_count, 2),
        )

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        while True:
            sentence_index = int(rng.integers(0, self.info.sentence_count))
            start, end = self.offsets[sentence_index]
            start = int(start)
            end = int(end)
            if end - start >= 2:
                break

        target_position = int(rng.integers(start + 1, end))
        context_start = max(start, target_position - self.context_length)
        context = self.tokens[context_start:target_position].astype(np.int64)
        if len(context) < self.context_length:
            padded = np.full((self.context_length,), PAD_ID, dtype=np.int64)
            padded[-len(context) :] = context
            context = padded

        target = np.int64(self.tokens[target_position])
        return torch.from_numpy(np.asarray(context, dtype=np.int64)), torch.tensor(target)
