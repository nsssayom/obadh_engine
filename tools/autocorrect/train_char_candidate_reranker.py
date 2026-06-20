#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn


FEATURE_DIM = 11
PAD = 0
UNK = 1


class CharCandidateRanker(nn.Module):
    def __init__(self, source_vocab_size, candidate_vocab_size, embed_size, hidden_size):
        super().__init__()
        self.source_embedding = nn.Embedding(source_vocab_size, embed_size, padding_idx=PAD)
        self.candidate_embedding = nn.Embedding(candidate_vocab_size, embed_size, padding_idx=PAD)
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM + embed_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, numeric, source_tokens, candidate_tokens, valid_mask):
        source_mask = (source_tokens != PAD).unsqueeze(-1)
        source_emb = self.source_embedding(source_tokens)
        source_repr = (source_emb * source_mask).sum(dim=1) / source_mask.sum(dim=1).clamp_min(1)

        candidate_mask = (candidate_tokens != PAD).unsqueeze(-1)
        candidate_emb = self.candidate_embedding(candidate_tokens)
        candidate_repr = (candidate_emb * candidate_mask).sum(dim=2) / candidate_mask.sum(dim=2).clamp_min(1)

        source_pair = source_repr.unsqueeze(1).expand_as(candidate_repr)
        pair = torch.cat(
            [
                numeric,
                source_pair,
                candidate_repr,
                source_pair * candidate_repr,
                (source_pair - candidate_repr).abs(),
            ],
            dim=-1,
        )
        return self.net(pair).squeeze(-1).masked_fill(~valid_mask, -1.0e9)


def load_records(paths):
    rows = 0
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                record = json.loads(line)
                label_indexes = [i for i, c in enumerate(record["candidates"]) if c["label"]]
                if len(label_indexes) != 1:
                    continue
                candidates = [
                    {
                        "text": candidate["text"],
                        "features": [float(v) for v in candidate["features"]]
                        + [float(candidate["score"])],
                    }
                    for candidate in record["candidates"]
                ]
                records.append(
                    {
                        "source": record["source"],
                        "candidates": candidates,
                        "label": label_indexes[0],
                    }
                )
    return rows, records


def build_vocab(records, field, max_size):
    counts = Counter()
    if field == "source":
        for record in records:
            counts.update(record["source"])
    else:
        for record in records:
            for candidate in record["candidates"]:
                counts.update(candidate["text"])
    chars = [char for char, _ in counts.most_common(max_size - 2)]
    return {char: index + 2 for index, char in enumerate(chars)}


def encode(text, vocab):
    return [vocab.get(char, UNK) for char in text] or [UNK]


def feature_stats(records):
    count = 0
    total = torch.zeros(FEATURE_DIM)
    total_sq = torch.zeros(FEATURE_DIM)
    for record in records:
        tensor = torch.tensor([c["features"] for c in record["candidates"]], dtype=torch.float32)
        count += tensor.shape[0]
        total += tensor.sum(dim=0)
        total_sq += (tensor * tensor).sum(dim=0)
    mean = total / max(count, 1)
    variance = (total_sq / max(count, 1)) - (mean * mean)
    return mean, variance.clamp_min(1e-6).sqrt()


def prepare_records(records, source_vocab, candidate_vocab, mean, std):
    mean = mean.tolist()
    std = std.tolist()
    for record in records:
        record["source_tensor"] = torch.tensor(
            encode(record["source"], source_vocab), dtype=torch.long
        )
        numeric = torch.empty((len(record["candidates"]), FEATURE_DIM), dtype=torch.float32)
        encoded_candidates = []
        for candidate in record["candidates"]:
            encoded = encode(candidate["text"], candidate_vocab)
            encoded_candidates.append(encoded)
            candidate["features"] = [
                (value - mean[index]) / std[index]
                for index, value in enumerate(candidate["features"])
            ]
        max_candidate_len = max(len(tokens) for tokens in encoded_candidates)
        candidate_tokens = torch.zeros(
            (len(record["candidates"]), max_candidate_len), dtype=torch.long
        )
        for index, (candidate, encoded) in enumerate(
            zip(record["candidates"], encoded_candidates)
        ):
            numeric[index] = torch.tensor(candidate["features"], dtype=torch.float32)
            candidate_tokens[index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        record["numeric_tensor"] = numeric
        record["candidate_tokens_tensor"] = candidate_tokens
        del record["candidates"]
        del record["source"]


def batches(records, batch_size, shuffle):
    indexes = list(range(len(records)))
    if shuffle:
        random.shuffle(indexes)
    for start in range(0, len(indexes), batch_size):
        chunk = [records[index] for index in indexes[start : start + batch_size]]
        max_candidates = max(record["numeric_tensor"].shape[0] for record in chunk)
        max_source_len = max(record["source_tensor"].shape[0] for record in chunk)
        max_candidate_len = max(
            record["candidate_tokens_tensor"].shape[1] for record in chunk
        )
        numeric = torch.zeros((len(chunk), max_candidates, FEATURE_DIM), dtype=torch.float32)
        source_tokens = torch.zeros((len(chunk), max_source_len), dtype=torch.long)
        candidate_tokens = torch.zeros(
            (len(chunk), max_candidates, max_candidate_len), dtype=torch.long
        )
        valid_mask = torch.zeros((len(chunk), max_candidates), dtype=torch.bool)
        labels = torch.empty((len(chunk),), dtype=torch.long)

        for row, record in enumerate(chunk):
            source = record["source_tensor"]
            candidate_features = record["numeric_tensor"]
            candidate_token_rows = record["candidate_tokens_tensor"]
            source_tokens[row, : source.shape[0]] = source
            candidate_count = candidate_features.shape[0]
            candidate_token_len = candidate_token_rows.shape[1]
            numeric[row, :candidate_count] = candidate_features
            candidate_tokens[row, :candidate_count, :candidate_token_len] = candidate_token_rows
            valid_mask[row, :candidate_count] = True
            labels[row] = record["label"]
        yield numeric, source_tokens, candidate_tokens, valid_mask, labels


def original_metrics(records):
    top1 = sum(1 for record in records if record["label"] == 0)
    reciprocal = sum(1.0 / (record["label"] + 1) for record in records)
    count = len(records)
    return {
        "top1_correct": top1,
        "top1_accuracy_given_present": top1 / count if count else 0.0,
        "mean_reciprocal_rank_given_present": reciprocal / count if count else 0.0,
    }


def evaluate(model, records, batch_size, device):
    model.eval()
    top1 = 0
    reciprocal = 0.0
    with torch.no_grad():
        for numeric, source, candidate, valid, labels in batches(records, batch_size, False):
            scores = model(
                numeric.to(device),
                source.to(device),
                candidate.to(device),
                valid.to(device),
            )
            labels = labels.to(device)
            order = scores.argsort(dim=1, descending=True)
            top1 += (order[:, 0] == labels).sum().item()
            for row in range(order.shape[0]):
                rank = (order[row] == labels[row]).nonzero(as_tuple=False)[0, 0].item() + 1
                reciprocal += 1.0 / rank
    count = len(records)
    return {
        "target_present_rows": count,
        "top1_correct": top1,
        "top1_accuracy_given_present": top1 / count if count else 0.0,
        "mean_reciprocal_rank_given_present": reciprocal / count if count else 0.0,
    }


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, nargs="+")
    parser.add_argument("--dev", required=True, nargs="+")
    parser.add_argument("--test", nargs="*")
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embed-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--source-vocab-size", type=int, default=96)
    parser.add_argument("--candidate-vocab-size", type=int, default=192)
    parser.add_argument("--replacement-min-score", type=float)
    parser.add_argument("--replacement-min-margin", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if (args.replacement_min_score is None) != (args.replacement_min_margin is None):
        parser.error("--replacement-min-score and --replacement-min-margin must be provided together")
    replacement_policy = None
    if args.replacement_min_score is not None:
        if args.replacement_min_margin < 0:
            parser.error("--replacement-min-margin must be non-negative")
        replacement_policy = {
            "min_score": args.replacement_min_score,
            "min_margin": args.replacement_min_margin,
        }

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    train_rows, train_records = load_records(args.train)
    dev_rows, dev_records = load_records(args.dev)
    test_rows, test_records = load_records(args.test or [])
    source_vocab = build_vocab(train_records, "source", args.source_vocab_size)
    candidate_vocab = build_vocab(train_records, "candidate", args.candidate_vocab_size)
    mean, std = feature_stats(train_records)
    prepare_records(train_records, source_vocab, candidate_vocab, mean, std)
    prepare_records(dev_records, source_vocab, candidate_vocab, mean, std)
    prepare_records(test_records, source_vocab, candidate_vocab, mean, std)
    model = CharCandidateRanker(
        len(source_vocab) + 2, len(candidate_vocab) + 2, args.embed_size, args.hidden_size
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_dev_mrr = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for numeric, source, candidate, valid, labels in batches(
            train_records, args.batch_size, True
        ):
            scores = model(
                numeric.to(device),
                source.to(device),
                candidate.to(device),
                valid.to(device),
            )
            loss = loss_fn(scores, labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        dev_metrics = evaluate(model, dev_records, args.batch_size, device)
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(batch_count, 1),
                "dev_top1_accuracy_given_present": dev_metrics["top1_accuracy_given_present"],
                "dev_mrr_given_present": dev_metrics["mean_reciprocal_rank_given_present"],
            }
        )
        print(
            f"epoch {epoch}/{args.epochs} loss={history[-1]['loss']:.6f} "
            f"dev_top1={history[-1]['dev_top1_accuracy_given_present']:.6f} "
            f"dev_mrr={history[-1]['dev_mrr_given_present']:.6f}",
            flush=True,
        )
        if dev_metrics["mean_reciprocal_rank_given_present"] > best_dev_mrr:
            best_dev_mrr = dev_metrics["mean_reciprocal_rank_given_present"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    artifact = {
        "kind": "obadh_char_candidate_reranker",
        "source_vocab": source_vocab,
        "candidate_vocab": candidate_vocab,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
    }
    if replacement_policy is not None:
        artifact["replacement_policy"] = replacement_policy
    torch.save(artifact, args.output)
    if args.json_output:
        state = artifact["state_dict"]
        json_artifact = {
            "kind": artifact["kind"],
            "source_vocab": artifact["source_vocab"],
            "candidate_vocab": artifact["candidate_vocab"],
            "feature_mean": artifact["feature_mean"],
            "feature_std": artifact["feature_std"],
            "source_embedding": state["source_embedding.weight"].tolist(),
            "candidate_embedding": state["candidate_embedding.weight"].tolist(),
            "layer0_weight": state["net.0.weight"].tolist(),
            "layer0_bias": state["net.0.bias"].tolist(),
            "layer2_weight": state["net.2.weight"].squeeze(0).tolist(),
            "layer2_bias": float(state["net.2.bias"].squeeze(0).item()),
        }
        if replacement_policy is not None:
            json_artifact["replacement_policy"] = replacement_policy
        Path(args.json_output).write_text(
            json.dumps(json_artifact, ensure_ascii=False), encoding="utf-8"
        )

    report = {
        "device": str(device),
        "train_rows": train_rows,
        "trainable_rows": len(train_records),
        "dev_rows": dev_rows,
        "dev_target_present_rows": len(dev_records),
        "test_rows": test_rows,
        "test_target_present_rows": len(test_records),
        "output": args.output,
        "json_output": args.json_output,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "embed_size": args.embed_size,
        "hidden_size": args.hidden_size,
        "source_vocab_size": len(source_vocab) + 2,
        "candidate_vocab_size": len(candidate_vocab) + 2,
        "replacement_policy": replacement_policy,
        "original_dev": original_metrics(dev_records),
        "reranked_dev": evaluate(model, dev_records, args.batch_size, device),
        "history": history,
    }
    if test_records:
        report["original_test"] = original_metrics(test_records)
        report["reranked_test"] = evaluate(model, test_records, args.batch_size, device)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
