#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build GRU sequences from synchronized DFT measurement folders."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


FEATURE_NAMES = (
    "ue_direction_sin",
    "ue_direction_cos",
    "velocity_toward_mps",
    "velocity_tangential_mps",
    "yaw_rate_dps",
    "distance_m",
    "current_beam_norm",
    "current_quality_db",
)


@dataclass
class SequenceDataset:
    x: np.ndarray
    y: np.ndarray
    trial_ids: np.ndarray
    anchor_times: np.ndarray
    ru_ids: np.ndarray
    codebook: dict
    label_source: str
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __len__(self):
        return int(self.y.size)


def _float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value, default=-1) -> int:
    number = _float(value, default)
    return int(number) if np.isfinite(number) else int(default)


def discover_trials(paths: Iterable[str | Path]) -> list[Path]:
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_file() and path.name == "dft_measurements.csv":
            found.add(path.parent)
        elif path.is_dir() and (path / "dft_measurements.csv").exists():
            found.add(path)
        elif path.is_dir():
            found.update(p.parent for p in path.rglob("dft_measurements.csv"))
    return sorted(found)


def _load_sweeps(path: Path, ru: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.empty(0, np.float64), np.empty(0, np.int64)
    times, labels = [], []
    with open(path, newline="", encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            if "valid" in row and row["valid"] not in ("1", "true", "True"):
                continue
            t = _float(row.get("completed_host_time"))
            label = _int(row.get(f"ru{ru}_best_beam"), -1)
            if np.isfinite(t) and label >= 0:
                times.append(t)
                labels.append(label)
    if not times:
        return np.empty(0, np.float64), np.empty(0, np.int64)
    order = np.argsort(times)
    return np.asarray(times, np.float64)[order], np.asarray(labels, np.int64)[order]


def _read_measurements(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def build_sequences(
    paths: Iterable[str | Path],
    sequence_length: int = 40,
    horizon_s: float = 0.25,
    stride: int = 5,
    max_gap_s: float = 0.03,
    max_label_delay_s: float = 15.0,
    exclude_phases: tuple[str, ...] = ("settle",),
) -> SequenceDataset:
    """Create per-RU examples from geometry labels or measured sweep labels.

    Geometry-mode rows already contain a beam computed at the extrapolated
    future UE position and are preferred.  For legacy/sweep trials, an anchor
    at time ``t`` receives the first completed sweep label at or after
    ``t+horizon_s``.  Invalid pose, settling samples, NaNs, and CSI gaps are
    rejected.
    """
    if sequence_length < 2 or stride < 1:
        raise ValueError("sequence_length >= 2 and stride >= 1 are required")
    trials = discover_trials(paths)
    if not trials:
        raise FileNotFoundError("no dft_measurements.csv found")

    examples, targets, ids, anchors, ru_ids = [], [], [], [], []
    label_sources: set[str] = set()
    reference_codebook = None
    for trial_no, trial in enumerate(trials):
        meta_path = trial / "metadata.json"
        cb_path = trial / "dft_codebook.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            codebook = metadata.get("codebook")
        else:
            metadata = {}
            codebook = None
        if codebook is None and cb_path.exists():
            codebook = json.loads(cb_path.read_text(encoding="utf-8"))
        if not codebook:
            raise ValueError(f"{trial}: codebook metadata is missing")
        signature = [(int(b["dft_bin"]), round(float(b["angle_deg"]), 6)) for b in codebook["beams"]]
        if reference_codebook is None:
            reference_codebook = codebook
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError(f"{trial}: DFT codebook differs from earlier trials")

        rows = _read_measurements(trial / "dft_measurements.csv")
        sweep_path = trial / "dft_sweeps.csv"
        if not rows:
            continue
        times = np.asarray([_float(row.get("host_rx_time")) for row in rows], np.float64)
        valid_base = np.asarray([
            row.get("pose_valid", "0") == "1" and row.get("phase", "") not in exclude_phases
            for row in rows
        ], dtype=bool)

        for ru in (1, 2, 3):
            geometry_labels = np.asarray(
                [_int(row.get(f"ru{ru}_geometry_beam"), -1) for row in rows],
                dtype=np.int64,
            )
            use_geometry = np.any(geometry_labels >= 0)
            if use_geometry:
                geometry_horizon = float(metadata.get(
                    "geometry_label_lead_s", metadata.get("geometry_lead_s", horizon_s)
                ))
                if abs(geometry_horizon - float(horizon_s)) > 1e-6:
                    raise ValueError(
                        f"{trial}: geometry labels use {geometry_horizon:g}s lead, "
                        f"but requested training horizon is {horizon_s:g}s"
                    )
            sweep_times, sweep_labels = _load_sweeps(sweep_path, ru)
            if not use_geometry and sweep_times.size == 0:
                continue
            columns = [f"ru{ru}_{name}" for name in FEATURE_NAMES]
            features = np.asarray(
                [[_float(row.get(column)) for column in columns] for row in rows],
                dtype=np.float32,
            )
            valid = valid_base & np.isfinite(features).all(axis=1) & np.isfinite(times)
            for end in range(sequence_length - 1, len(rows), stride):
                start = end - sequence_length + 1
                if not valid[start : end + 1].all():
                    continue
                if np.any(np.diff(times[start : end + 1]) > max_gap_s):
                    continue
                if use_geometry:
                    label = int(geometry_labels[end])
                    label_source = "future_geometry"
                else:
                    target_time = times[end] + float(horizon_s)
                    label_pos = int(np.searchsorted(sweep_times, target_time, side="left"))
                    if label_pos >= sweep_times.size:
                        continue
                    if sweep_times[label_pos] - target_time > max_label_delay_s:
                        continue
                    label = int(sweep_labels[label_pos])
                    label_source = "measured_sweep"
                if not 0 <= label < len(codebook["beams"]):
                    continue
                examples.append(features[start : end + 1].copy())
                targets.append(label)
                ids.append(trial_no)
                anchors.append(times[end])
                ru_ids.append(ru)
                label_sources.add(label_source)

    if not examples:
        raise ValueError(
            "no valid sequences were built; check geometry labels or collect complete sweeps and verify pose/CSI timestamps"
        )
    return SequenceDataset(
        x=np.stack(examples).astype(np.float32),
        y=np.asarray(targets, np.int64),
        trial_ids=np.asarray(ids, np.int32),
        anchor_times=np.asarray(anchors, np.float64),
        ru_ids=np.asarray(ru_ids, np.int8),
        codebook=reference_codebook,
        label_source="+".join(sorted(label_sources)),
    )


def train_validation_split(
    dataset: SequenceDataset,
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole trials when possible, otherwise use a chronological split."""
    if not 0.05 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0.05, 0.5)")
    trials = np.unique(dataset.trial_ids)
    if trials.size >= 2:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(trials)
        n_val = min(trials.size - 1, max(1, int(round(trials.size * validation_fraction))))
        val_trials = set(int(v) for v in shuffled[:n_val])
        val_mask = np.asarray([int(v) in val_trials for v in dataset.trial_ids])
    else:
        split_time = float(np.quantile(dataset.anchor_times, 1.0 - validation_fraction))
        val_mask = dataset.anchor_times >= split_time
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("train/validation split is empty")
    return train_idx, val_idx
