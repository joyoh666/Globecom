#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyTorch GRU used for DFT beam classification and online inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # Collection/sweep mode must work without PyTorch.
    torch = None
    nn = None


def require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for GRU training/inference. Install the build appropriate "
            "for this host, then run ml/train_dft_gru.py again."
        )
    return torch


if nn is not None:
    class DFTBeamGRU(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_beams: int,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_size = int(input_size)
            self.hidden_size = int(hidden_size)
            self.num_layers = int(num_layers)
            self.num_beams = int(num_beams)
            self.gru = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=float(dropout) if self.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_size, self.num_beams),
            )

        def forward(self, x):
            output, _ = self.gru(x)
            return self.head(output[:, -1])
else:
    class DFTBeamGRU:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_torch()


def resolve_device(requested: str = "auto") -> str:
    t = require_torch()
    if requested != "auto":
        return requested
    if t.cuda.is_available():
        return "cuda"
    if hasattr(t.backends, "mps") and t.backends.mps.is_available():
        return "mps"
    return "cpu"


class BeamPredictor:
    """Load a training checkpoint and classify one sequence per RU."""

    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        t = require_torch()
        self.device = resolve_device(device)
        path = Path(checkpoint)
        try:
            saved = t.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            saved = t.load(path, map_location=self.device)
        if saved.get("schema") != "fr3-dft-gru-v1":
            raise ValueError(f"unsupported checkpoint schema in {path}")
        self.feature_names = tuple(saved["feature_names"])
        self.sequence_length = int(saved["sequence_length"])
        self.codebook = saved["codebook"]
        config = saved["model_config"]
        self.model = DFTBeamGRU(**config).to(self.device)
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()
        self.mean = np.asarray(saved["feature_mean"], dtype=np.float32)
        self.std = np.asarray(saved["feature_std"], dtype=np.float32)
        if self.mean.shape != (config["input_size"],) or self.std.shape != self.mean.shape:
            raise ValueError("checkpoint feature normalization has the wrong shape")

    def assert_codebook(self, runtime: dict) -> None:
        saved_beams, runtime_beams = self.codebook.get("beams", []), runtime.get("beams", [])
        if len(saved_beams) != len(runtime_beams):
            raise ValueError(
                f"DFT codebook size differs: model={len(saved_beams)} runtime={len(runtime_beams)}"
            )
        for left, right in zip(saved_beams, runtime_beams):
            if int(left["dft_bin"]) != int(right["dft_bin"]):
                raise ValueError("DFT bin ordering differs between model and runtime")

    def predict(self, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(beam_class, probability)`` for input ``(B,S,F)``."""
        t = require_torch()
        x = np.asarray(sequences, dtype=np.float32)
        expected = (self.sequence_length, self.mean.size)
        if x.ndim != 3 or x.shape[1:] != expected:
            raise ValueError(f"expected input (B,{expected[0]},{expected[1]}), got {x.shape}")
        x = (x - self.mean[None, None, :]) / self.std[None, None, :]
        with t.inference_mode():
            logits = self.model(t.from_numpy(x).to(self.device))
            probability = t.softmax(logits, dim=-1)
            confidence, classes = probability.max(dim=-1)
        return classes.cpu().numpy(), confidence.cpu().numpy()

