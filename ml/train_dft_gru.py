#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a GRU to predict the future best DFT beam for each RU.

The same GRU is shared by all RUs.  Every input is expressed relative to its
RU, so the learned relationship is geometry-invariant:

* UE direction: sine/cosine of RU direction in the UE body frame
* UE motion: velocity toward and tangential to the RU, plus yaw rate
* RU-UE distance
* currently applied DFT beam bin
* current measured beam quality
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

from ml.dft_dataset import FEATURE_NAMES, build_sequences, train_validation_split
from ml.dft_gru import DFTBeamGRU, require_torch, resolve_device


def evaluate(model, loader, criterion, device, num_beams):
    torch = require_torch()
    model.eval()
    total_loss = total = correct = top2 = 0.0
    confusion = np.zeros((num_beams, num_beams), dtype=np.int64)
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            batch = y.numel()
            total_loss += float(loss) * batch
            total += batch
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum())
            k = min(2, num_beams)
            top2 += int((logits.topk(k, dim=1).indices == y[:, None]).any(dim=1).sum())
            for actual, guessed in zip(y.cpu().numpy(), pred.cpu().numpy()):
                confusion[int(actual), int(guessed)] += 1
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "top2_accuracy": top2 / max(1, total),
        "count": int(total),
        "confusion": confusion.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", nargs="+", help="measurement folders or parents containing dft_measurements.csv")
    ap.add_argument("--output", default="models/dft_beam_gru.pt")
    ap.add_argument("--sequence-length", type=int, default=40)
    ap.add_argument("--horizon", type=float, default=0.25, help="prediction horizon [s]")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-gap", type=float, default=0.03)
    ap.add_argument("--max-label-delay", type=float, default=15.0)
    ap.add_argument("--hidden-size", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--validation-fraction", type=float, default=0.2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    try:
        torch = require_torch()
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 2
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    print("[DATA] loading and aligning measurements/sweeps ...")
    dataset = build_sequences(
        a.data,
        sequence_length=a.sequence_length,
        horizon_s=a.horizon,
        stride=a.stride,
        max_gap_s=a.max_gap,
        max_label_delay_s=a.max_label_delay,
    )
    train_idx, val_idx = train_validation_split(dataset, a.validation_fraction, a.seed)
    mean = dataset.x[train_idx].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = dataset.x[train_idx].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    x = ((dataset.x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)

    TensorDataset = torch.utils.data.TensorDataset
    DataLoader = torch.utils.data.DataLoader
    train_set = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(dataset.y[train_idx]))
    val_set = TensorDataset(torch.from_numpy(x[val_idx]), torch.from_numpy(dataset.y[val_idx]))
    generator = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=a.batch_size, shuffle=False)

    num_beams = len(dataset.codebook["beams"])
    counts = np.bincount(dataset.y[train_idx], minlength=num_beams).astype(np.float32)
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = np.sqrt(counts[present].sum() / counts[present])
    if present.any():
        weights[present] /= weights[present].mean()

    device = resolve_device(a.device)
    config = {
        "input_size": len(FEATURE_NAMES),
        "hidden_size": a.hidden_size,
        "num_layers": a.layers,
        "num_beams": num_beams,
        "dropout": a.dropout,
    }
    model = DFTBeamGRU(**config).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    print(
        f"[TRAIN] samples train={len(train_set)} val={len(val_set)} beams={num_beams} "
        f"device={device} classes={counts.astype(int).tolist()}"
    )

    best_state = None
    best_metrics = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        running_loss = seen = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss) * yb.numel()
            seen += yb.numel()
        metrics = evaluate(model, val_loader, criterion, device, num_beams)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / max(1, seen)
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(metrics)
        scheduler.step(metrics["loss"])
        print(
            f"[E{epoch:03d}] train={metrics['train_loss']:.4f} val={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']*100:5.1f}% top2={metrics['top2_accuracy']*100:5.1f}% "
            f"lr={metrics['learning_rate']:.2e}"
        )
        if best_metrics is None or metrics["loss"] < best_metrics["loss"] - 1e-5:
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_metrics = dict(metrics)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= a.patience:
                print(f"[STOP] no validation improvement for {a.patience} epochs")
                break

    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "fr3-dft-gru-v1",
        "model_state": best_state,
        "model_config": config,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "sequence_length": a.sequence_length,
        "prediction_horizon_s": a.horizon,
        "label_source": dataset.label_source,
        "codebook": dataset.codebook,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "class_counts": counts.astype(int).tolist(),
        "train_samples": int(train_idx.size),
        "validation_samples": int(val_idx.size),
        "seed": a.seed,
    }
    torch.save(checkpoint, output)
    report = {
        key: value for key, value in checkpoint.items()
        if key not in {"model_state"}
    }
    report["elapsed_s"] = time.time() - started
    report["history"] = history
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[DONE] {output} · best epoch {best_epoch} · "
        f"val acc {best_metrics['accuracy']*100:.1f}% · report {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
