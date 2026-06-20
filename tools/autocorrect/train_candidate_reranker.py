#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn


FEATURE_DIM = 11


class CandidateRanker(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def load_records(paths):
    rows = 0
    target_present = 0
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                record = json.loads(line)
                candidates = record["candidates"]
                label_indexes = [i for i, c in enumerate(candidates) if c["label"]]
                if len(label_indexes) != 1:
                    continue
                target_present += 1
                features = [
                    [float(value) for value in c["features"]] + [float(c["score"])]
                    for c in candidates
                ]
                records.append((features, label_indexes[0]))
    return rows, target_present, records


def feature_stats(records):
    count = 0
    total = torch.zeros(FEATURE_DIM)
    total_sq = torch.zeros(FEATURE_DIM)
    for features, _ in records:
        tensor = torch.tensor(features, dtype=torch.float32)
        count += tensor.shape[0]
        total += tensor.sum(dim=0)
        total_sq += (tensor * tensor).sum(dim=0)
    mean = total / max(count, 1)
    variance = (total_sq / max(count, 1)) - (mean * mean)
    std = variance.clamp_min(1e-6).sqrt()
    return mean, std


def batches(records, batch_size, shuffle):
    indexes = list(range(len(records)))
    if shuffle:
        random.shuffle(indexes)
    for start in range(0, len(indexes), batch_size):
        chunk = [records[index] for index in indexes[start : start + batch_size]]
        max_len = max(len(features) for features, _ in chunk)
        features = torch.zeros((len(chunk), max_len, FEATURE_DIM), dtype=torch.float32)
        mask = torch.zeros((len(chunk), max_len), dtype=torch.bool)
        labels = torch.empty((len(chunk),), dtype=torch.long)
        for row, (row_features, label) in enumerate(chunk):
            row_tensor = torch.tensor(row_features, dtype=torch.float32)
            features[row, : row_tensor.shape[0]] = row_tensor
            mask[row, : row_tensor.shape[0]] = True
            labels[row] = label
        yield features, mask, labels


def evaluate(model, records, mean, std, device, batch_size):
    model.eval()
    top1 = 0
    reciprocal_sum = 0.0
    with torch.no_grad():
        for features, mask, labels in batches(records, batch_size, shuffle=False):
            features = features.to(device)
            features = (features - mean) / std
            mask = mask.to(device)
            labels = labels.to(device)
            scores = model(features).masked_fill(~mask, -1.0e9)
            order = scores.argsort(dim=1, descending=True)
            top1 += (order[:, 0] == labels).sum().item()
            for row in range(order.shape[0]):
                rank = (order[row] == labels[row]).nonzero(as_tuple=False)[0, 0].item() + 1
                reciprocal_sum += 1.0 / rank
    count = len(records)
    return {
        "target_present_rows": count,
        "top1_correct": top1,
        "top1_accuracy_given_present": top1 / count if count else 0.0,
        "mean_reciprocal_rank_given_present": reciprocal_sum / count if count else 0.0,
    }


def original_metrics(records):
    top1 = 0
    reciprocal_sum = 0.0
    for _, label in records:
        top1 += int(label == 0)
        reciprocal_sum += 1.0 / (label + 1)
    count = len(records)
    return {
        "top1_correct": top1,
        "top1_accuracy_given_present": top1 / count if count else 0.0,
        "mean_reciprocal_rank_given_present": reciprocal_sum / count if count else 0.0,
    }


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, nargs="+")
    parser.add_argument("--dev", required=True, nargs="+")
    parser.add_argument("--test", nargs="*")
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    train_rows, train_present, train_records = load_records(args.train)
    dev_rows, dev_present, dev_records = load_records(args.dev)
    test_rows, test_present, test_records = load_records(args.test or [])
    mean, std = feature_stats(train_records)
    model = CandidateRanker(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    mean_device = mean.to(device)
    std_device = std.to(device)
    best_state = None
    best_dev_mrr = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for features, mask, labels in batches(train_records, args.batch_size, shuffle=True):
            features = features.to(device)
            features = (features - mean_device) / std_device
            mask = mask.to(device)
            labels = labels.to(device)
            scores = model(features).masked_fill(~mask, -1.0e9)
            loss = loss_fn(scores, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        dev_metrics = evaluate(model, dev_records, mean_device, std_device, device, args.batch_size)
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(batch_count, 1),
                "dev_top1_accuracy_given_present": dev_metrics[
                    "top1_accuracy_given_present"
                ],
                "dev_mrr_given_present": dev_metrics[
                    "mean_reciprocal_rank_given_present"
                ],
            }
        )
        if dev_metrics["mean_reciprocal_rank_given_present"] > best_dev_mrr:
            best_dev_mrr = dev_metrics["mean_reciprocal_rank_given_present"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    artifact = {
        "kind": "obadh_candidate_mlp_reranker",
        "feature_dim": FEATURE_DIM,
        "hidden_size": args.hidden_size,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
    }
    torch.save(artifact, args.output)
    if args.json_output:
        state = artifact["state_dict"]
        json_artifact = {
            "kind": artifact["kind"],
            "feature_dim": artifact["feature_dim"],
            "hidden_size": artifact["hidden_size"],
            "feature_mean": artifact["feature_mean"],
            "feature_std": artifact["feature_std"],
            "layer0_weight": state["net.0.weight"].tolist(),
            "layer0_bias": state["net.0.bias"].tolist(),
            "layer2_weight": state["net.2.weight"].squeeze(0).tolist(),
            "layer2_bias": float(state["net.2.bias"].squeeze(0).item()),
        }
        Path(args.json_output).write_text(
            json.dumps(json_artifact, ensure_ascii=False), encoding="utf-8"
        )

    report = {
        "device": str(device),
        "train_rows": train_rows,
        "train_target_present_rows": train_present,
        "trainable_rows": len(train_records),
        "dev_rows": dev_rows,
        "dev_target_present_rows": dev_present,
        "test_rows": test_rows,
        "test_target_present_rows": test_present,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "output": args.output,
        "json_output": args.json_output,
        "original_dev": original_metrics(dev_records),
        "reranked_dev": evaluate(model, dev_records, mean_device, std_device, device, args.batch_size),
        "history": history,
    }
    if test_records:
        report["original_test"] = original_metrics(test_records)
        report["reranked_test"] = evaluate(
            model, test_records, mean_device, std_device, device, args.batch_size
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
