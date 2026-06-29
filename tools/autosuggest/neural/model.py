#!/usr/bin/env python3
"""Compact neural next-word model for Obadh autosuggest."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AutosuggestModelConfig:
    vocab_size: int
    context_length: int = 16
    embedding_dim: int = 192
    layers: int = 2
    heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    pad_id: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class NextWordTransformer(nn.Module):
    """Small causal Transformer that predicts the next Bengali word ID."""

    def __init__(self, config: AutosuggestModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_id,
        )
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_bias)
        with torch.no_grad():
            self.token_embedding.weight[self.config.pad_id].fill_(0.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if not torch.jit.is_tracing() and seq_len != self.config.context_length:
            raise ValueError(f"expected context length {self.config.context_length}, got {seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        padding_mask = input_ids.eq(self.config.pad_id)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        last = self.norm(encoded[:, -1, :])
        return F.linear(last, self.token_embedding.weight, self.output_bias)

    @torch.no_grad()
    def suggest(self, input_ids: torch.Tensor, top_k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids)
        return torch.topk(logits, k=top_k, dim=-1)
