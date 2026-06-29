#!/usr/bin/env python3
"""Run local autosuggest inference from Bengali context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import BOS_ID, PAD_ID, UNK_ID, load_vocab
from .data import RandomContextDataset
from .model import AutosuggestModelConfig, NextWordTransformer
from .train import accuracy_at_k


def encode_context(text: str, vocab: dict[str, int], context_length: int) -> np.ndarray:
    tokens = [token for token in text.strip().split() if token]
    ids = [BOS_ID]
    ids.extend(vocab.get(token, UNK_ID) for token in tokens)
    ids = ids[-context_length:]
    if len(ids) < context_length:
        ids = [PAD_ID] * (context_length - len(ids)) + ids
    return np.asarray(ids, dtype=np.int64)


def top_predictions(logits: np.ndarray, words: list[str], top_k: int) -> list[dict]:
    blocked = {PAD_ID, BOS_ID, UNK_ID}
    order = np.argsort(logits)[::-1]
    rows = []
    for token_id in order:
        token_id = int(token_id)
        if token_id in blocked:
            continue
        rows.append(
            {
                "id": token_id,
                "token": words[token_id],
                "score": float(logits[token_id]),
            }
        )
        if len(rows) >= top_k:
            break
    return rows


def predict_torch(checkpoint_path: Path, vocab_path: Path, context: str, top_k: int) -> dict:
    words, vocab = load_vocab(vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = AutosuggestModelConfig(**checkpoint["config"])
    model = NextWordTransformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    encoded = encode_context(context, vocab, config.context_length)
    with torch.no_grad():
        logits = model(torch.from_numpy(encoded).unsqueeze(0)).squeeze(0).numpy()

    return {
        "backend": "torch",
        "checkpoint": str(checkpoint_path),
        "context": context,
        "encoded": encoded.tolist(),
        "predictions": top_predictions(logits, words, top_k),
    }


def predict_onnx(model_path: Path, vocab_path: Path, context: str, top_k: int) -> dict:
    import onnxruntime as ort

    words, vocab = load_vocab(vocab_path)
    manifest_path = model_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context_length = int(manifest["config"]["context_length"])
    encoded = encode_context(context, vocab, context_length)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], {"input_ids": encoded.reshape(1, -1)})[0][0]

    return {
        "backend": "onnx",
        "model": str(model_path),
        "context": context,
        "encoded": encoded.tolist(),
        "predictions": top_predictions(logits, words, top_k),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=Path, default=Path("data/autosuggest/models/neural/vocab.tsv"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--context")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--eval-dataset-dir", type=Path)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=99_991)
    args = parser.parse_args()

    if bool(args.checkpoint) == bool(args.onnx):
        raise SystemExit("pass exactly one of --checkpoint or --onnx")

    if args.eval_dataset_dir:
        if not args.checkpoint:
            raise SystemExit("evaluation currently requires --checkpoint")
        print(
            json.dumps(
                evaluate_torch(
                    args.checkpoint,
                    args.vocab,
                    args.eval_dataset_dir,
                    args.eval_batches,
                    args.batch_size,
                    args.seed,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.checkpoint:
        if args.context is None:
            raise SystemExit("--context is required for prediction")
        result = predict_torch(args.checkpoint, args.vocab, args.context, args.top_k)
    else:
        if args.context is None:
            raise SystemExit("--context is required for prediction")
        result = predict_onnx(args.onnx, args.vocab, args.context, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def evaluate_torch(
    checkpoint_path: Path,
    vocab_path: Path,
    dataset_dir: Path,
    eval_batches: int,
    batch_size: int,
    seed: int,
) -> dict:
    words, _ = load_vocab(vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = AutosuggestModelConfig(**checkpoint["config"])
    model = NextWordTransformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = RandomContextDataset(
        dataset_dir,
        context_length=config.context_length,
        sample_count=eval_batches * batch_size,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
    total_loss = 0.0
    total = 0
    top1 = top3 = top5 = top10 = 0.0
    with torch.no_grad():
        for context, target in loader:
            logits = model(context)
            loss = loss_fn(logits, target)
            batch = target.numel()
            total += batch
            total_loss += loss.item() * batch
            top1 += accuracy_at_k(logits, target, 1) * batch
            top3 += accuracy_at_k(logits, target, 3) * batch
            top5 += accuracy_at_k(logits, target, 5) * batch
            top10 += accuracy_at_k(logits, target, 10) * batch

    return {
        "checkpoint": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "vocab": str(vocab_path),
        "vocab_size": len(words),
        "examples": total,
        "loss": total_loss / total if total else 0.0,
        "top1": top1 / total if total else 0.0,
        "top3": top3 / total if total else 0.0,
        "top5": top5 / total if total else 0.0,
        "top10": top10 / total if total else 0.0,
    }


if __name__ == "__main__":
    main()
