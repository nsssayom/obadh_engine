"""A tiny BiGRU + CTC model for word-level Bengali sequence prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BigruCtcConfig:
    input_vocab_size: int
    output_vocab_size: int
    embedding_dim: int = 48
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    padding_id: int = 0
    blank_id: int = 0


class ObadhBiGruCtc(nn.Module):
    """Small export-friendly sequence model.

    The forward pass intentionally avoids packed sequences. Padding is handled
    through CTC input lengths during training and by decoder masking during
    inference. This keeps ONNX/Core ML export simpler and predictable.
    """

    def __init__(self, config: BigruCtcConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.input_vocab_size,
            config.embedding_dim,
            padding_idx=config.padding_id,
        )
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        direction_factor = 2 if config.bidirectional else 1
        self.classifier = nn.Linear(config.hidden_size * direction_factor, config.output_vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        embeddings = self.embedding(input_ids)
        sequence, _ = self.gru(embeddings)
        return self.classifier(sequence)

    def log_probs_for_ctc(self, input_ids: Tensor) -> Tensor:
        logits = self.forward(input_ids)
        return F.log_softmax(logits, dim=-1).transpose(0, 1)
