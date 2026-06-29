#!/usr/bin/env python3
"""Train the neural Obadh autosuggest model."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .common import PAD_ID, load_vocab, save_manifest
from .data import RandomContextDataset
from .model import AutosuggestModelConfig, NextWordTransformer


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def accuracy_at_k(logits: torch.Tensor, target: torch.Tensor, k: int) -> float:
    top = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1).indices
    return top.eq(target.unsqueeze(-1)).any(dim=-1).float().mean().item()


def run_eval(
    model: NextWordTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    top1 = 0.0
    top3 = 0.0
    top5 = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    with torch.no_grad():
        for batch_index, (context, target) in enumerate(loader):
            if batch_index >= max_batches:
                break
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(context)
            loss = loss_fn(logits, target)
            batch = target.numel()
            total_examples += batch
            total_loss += loss.item() * batch
            top1 += accuracy_at_k(logits, target, 1) * batch
            top3 += accuracy_at_k(logits, target, 3) * batch
            top5 += accuracy_at_k(logits, target, 5) * batch

    if total_examples == 0:
        return {"loss": 0.0, "perplexity": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0}
    loss = total_loss / total_examples
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "top1": top1 / total_examples,
        "top3": top3 / total_examples,
        "top5": top5 / total_examples,
    }


def train(args: argparse.Namespace) -> dict:
    words, _ = load_vocab(args.vocab)
    device = pick_device(args.device)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        config = AutosuggestModelConfig(**resume_checkpoint["config"])
        if config.vocab_size != len(words):
            raise ValueError(
                f"resume vocab size {config.vocab_size} does not match {len(words)}"
            )
    else:
        config = AutosuggestModelConfig(
            vocab_size=len(words),
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            layers=args.layers,
            heads=args.heads,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            pad_id=PAD_ID,
        )

    eval_data = RandomContextDataset(
        args.dataset_dir,
        context_length=config.context_length,
        sample_count=args.eval_batches * args.batch_size,
        seed=args.seed + 9_999_999,
    )
    eval_loader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = NextWordTransformer(config).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history = list(resume_checkpoint.get("history", [])) if resume_checkpoint else []
    best_top5 = max((item.get("top5", -1.0) for item in history), default=-1.0)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_data = RandomContextDataset(
            args.dataset_dir,
            context_length=config.context_length,
            sample_count=args.steps_per_epoch * args.batch_size,
            seed=args.seed + epoch * 1_000_003,
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        model.train()
        epoch_loss = 0.0
        epoch_examples = 0
        for step, (context, target) in enumerate(train_loader, start=1):
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(context)
            loss = loss_fn(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch = target.numel()
            epoch_examples += batch
            epoch_loss += loss.item() * batch
            if args.log_every > 0 and step % args.log_every == 0:
                print(
                    {
                        "epoch": epoch,
                        "step": step,
                        "loss": epoch_loss / max(1, epoch_examples),
                        "device": device.type,
                    },
                    flush=True,
                )

        metrics = run_eval(model, eval_loader, device, args.eval_batches)
        metrics["train_loss"] = epoch_loss / max(1, epoch_examples)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(metrics, flush=True)

        checkpoint = {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "vocab": str(args.vocab),
            "metrics": metrics,
            "history": history,
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if metrics["top5"] > best_top5:
            best_top5 = metrics["top5"]
            torch.save(checkpoint, args.output_dir / "best.pt")

    report = {
        "output_dir": str(args.output_dir),
        "dataset_dir": str(args.dataset_dir),
        "vocab": str(args.vocab),
        "device": device.type,
        "torch_version": torch.__version__,
        "config": config.to_dict(),
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "elapsed_seconds": time.time() - start_time,
        "history": history,
        "best_top5": best_top5,
    }
    save_manifest(args.output_dir / "train_manifest.json", **report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/autosuggest/models/neural/dataset"))
    parser.add_argument("--vocab", type=Path, default=Path("data/autosuggest/models/neural/vocab.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/autosuggest/models/neural/checkpoints"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=25_000)
    parser.add_argument("--eval-batches", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    print(train(args))


if __name__ == "__main__":
    main()
