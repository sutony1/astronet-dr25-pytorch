#!/usr/bin/env python3
"""Train AstroNet on exactly one CUDA device and write reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from astronet_dr25.dataset import ViewDataset
from astronet_dr25.metrics import binary_metrics
from astronet_dr25.model import AstroNet, OFFICIAL_PARAMETER_COUNT


def predict(model, loader, device):
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    row_ids: list[int] = []
    with torch.no_grad():
        for local, global_view, label, row_id in loader:
            logits = model(local.to(device), global_view.to(device))
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(label.numpy().astype(int).tolist())
            row_ids.extend(row_id.numpy().astype(int).tolist())
    return np.asarray(row_ids), np.asarray(labels), np.asarray(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        print(
            f"Visible CUDA devices={torch.cuda.device_count()}; training will still use only {device}",
            flush=True,
        )

    index = pd.read_parquet(args.view_root / "view_index.parquet")
    loaders = {}
    for split in ("train", "validation", "test"):
        dataset = ViewDataset(index[index["split"] == split], args.view_root)
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    model = AstroNet().to(device)
    if model.parameter_count != OFFICIAL_PARAMETER_COUNT:
        raise RuntimeError(f"Unexpected parameter count: {model.parameter_count}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    history: list[dict[str, object]] = []
    best_pr_auc = -1.0
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        examples = 0
        for local, global_view, label, _ in loaders["train"]:
            local = local.to(device, non_blocking=True)
            global_view = global_view.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            # Official augmentation: random horizontal reflection.
            flip = torch.rand(len(label), device=device) < 0.5
            local[flip] = torch.flip(local[flip], dims=[1])
            global_view[flip] = torch.flip(global_view[flip], dims=[1])
            optimizer.zero_grad(set_to_none=True)
            logits = model(local, global_view)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(label)
            examples += len(label)

        _, val_labels, val_scores = predict(model, loaders["validation"], device)
        val_metrics = binary_metrics(val_labels, val_scores)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(examples, 1),
            "validation": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        current_pr_auc = val_metrics["pr_auc"] if val_metrics["pr_auc"] is not None else -1.0
        if current_pr_auc > best_pr_auc:
            best_pr_auc = current_pr_auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "parameter_count": model.parameter_count,
                    "seed": args.seed,
                },
                args.output / "best.pt",
            )

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    row_ids, test_labels, test_scores = predict(model, loaders["test"], device)
    test_metrics = binary_metrics(test_labels, test_scores)
    predictions = pd.DataFrame(
        {"row_id": row_ids, "label": test_labels, "astronet_score": test_scores}
    ).merge(index, on=["row_id", "label"], how="left")
    predictions.to_csv(args.output / "test_predictions.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "visible_cuda_devices": torch.cuda.device_count(),
        "parameter_count": model.parameter_count,
        "best_epoch": int(checkpoint["epoch"]),
        "test": test_metrics,
        "history": history,
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
