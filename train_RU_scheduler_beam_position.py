#!/usr/bin/env python3
"""Train a position-aware GRU Double-DQN RU scheduler for beamformed CSI.

This is the beam-aware successor to ``train_RU_scheduler.py``.  The original
scheduler observes channel magnitude only.  This version observes

* the *effective* per-RU CSI power gain after beamforming,
  ``|h_eff|^2 = |h^H f|^2``;
* UE position in the experiment coordinate frame; and
* UE heading encoded as sine/cosine (no angle wrap discontinuity).

The script accepts either synchronized ``dft_measurements.csv`` files produced
by ``runtime/dft_beam_measurement.py`` or NPZ files.  In the CSV, ``ruN_h_re``
and ``ruN_h_im`` were measured while ``ruN_beam`` was active, so they already
are scalar effective CSI.  An NPZ may instead contain raw array CSI and the
applied beamforming vectors; in that case this script explicitly evaluates
``h^H f`` before building scheduler observations and rewards.

NPZ schema
----------
Required geometry:

``ue_position_m``
    ``(N,L,2|3)`` or ``(L,2|3)`` float array.
``ue_yaw_rad`` / ``ue_yaw_deg`` / ``ue_direction_xy``
    Heading as ``(N,L)``, ``(L,)``, or unit vectors ``(...,2)``.

Required channel data, one of:

``effective_csi``
    Complex ``(N,L,R)`` or ``(L,R)`` array, measured after applying the beam.
``channel_csi`` and ``beamforming_vectors``
    Complex raw CSI ``(N,L,R,A)`` and broadcast-compatible beam weights.  The
    weights are normalized to unit norm so all RU actions use equal total TX
    power.  The effective downlink channel is ``sum(conj(h) * f, axis=-1)``.

Optional scalar metadata: ``dt`` (default 0.005 s).  Train/validation/test data
are passed separately to prevent trial leakage, for example::

    python train_RU_scheduler_beam_position.py \
      --train-data logs/dft_train_1 logs/dft_train_2 \
      --validation-data logs/dft_validation \
      --test-data logs/dft_test \
      --save-dir checkpoints_ru_scheduler_beam_position

The RU action space is the three single-RU actions followed by the three
two-RU combinations.  Multi-RU reward uses equal total transmit power and the
expected coherent gain after effective-CSI phase precoding.  Its residual
phase-error standard deviation is configurable.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "PyTorch is required for RU scheduler training. Install requirements-dft.txt first."
    ) from exc


DEFAULT_DT = 0.005
NUM_RUS = 3
ACTION_STATES: Tuple[Tuple[int, ...], ...] = (
    (0,), (1,), (2,), (0, 1), (0, 2), (1, 2),
)
ACTION_NAMES = tuple("+".join(f"RU{ru + 1}" for ru in state) for state in ACTION_STATES)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available")
    return device


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return float(end)
    fraction = min(max(step, 0) / float(duration), 1.0)
    return float(start + fraction * (end - start))


def count_trainable_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def effective_csi_from_channel(
    channel_csi: np.ndarray,
    beamforming_vectors: np.ndarray,
    normalize_beams: bool = True,
) -> np.ndarray:
    """Return downlink effective CSI ``h^H f`` for every RU and time.

    ``channel_csi`` must end in ``(num_rus, num_antennas)``.  Beam weights may
    have the same shape or any shape broadcast-compatible with it.  Unit-norm
    weights make RU and multi-RU actions comparable under a fixed TX budget.
    """
    channel = np.asarray(channel_csi)
    weights = np.asarray(beamforming_vectors)
    if channel.ndim < 2:
        raise ValueError(f"channel_csi must end in (num_rus,num_antennas), got {channel.shape}")
    if channel.shape[-2] != NUM_RUS:
        raise ValueError(f"expected {NUM_RUS} RUs in channel_csi, got shape {channel.shape}")
    if weights.shape[-1] != channel.shape[-1]:
        raise ValueError(
            f"beam antenna dimension {weights.shape[-1]} != channel dimension {channel.shape[-1]}"
        )
    try:
        weights = np.broadcast_to(weights, channel.shape)
    except ValueError as exc:
        raise ValueError(
            f"beamforming_vectors shape {np.shape(beamforming_vectors)} is not broadcast-compatible "
            f"with channel_csi shape {channel.shape}"
        ) from exc
    weights = weights.astype(np.complex128, copy=False)
    if normalize_beams:
        norms = np.linalg.norm(weights, axis=-1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
            raise ValueError("beamforming_vectors contain a zero or non-finite vector")
        weights = weights / norms
    effective = np.sum(np.conj(channel.astype(np.complex128, copy=False)) * weights, axis=-1)
    if np.any(~np.isfinite(effective.real)) or np.any(~np.isfinite(effective.imag)):
        raise ValueError("computed effective CSI contains NaN or infinity")
    return effective.astype(np.complex64)


def effective_gain_db(effective_csi: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    """Power gain ``10 log10(|h_eff|^2)`` with a finite lower floor."""
    power = np.abs(np.asarray(effective_csi, dtype=np.complex128)) ** 2
    floor_power = 10.0 ** (float(floor_db) / 10.0)
    return (10.0 * np.log10(np.maximum(power, floor_power))).astype(np.float32)


def expected_action_power(
    effective_csi: np.ndarray,
    state: Sequence[int],
    mode: str = "phase-aligned-expected",
    phase_error_std_rad: float = math.pi / 8.0,
) -> float:
    """Average received power for a fixed-total-power RU action.

    ``phase-aligned-expected`` assumes predicted effective-CSI phase precoding.
    Independent zero-mean Gaussian phase errors give a cross-term retention of
    ``exp(-sigma^2)``.  ``measured-phase`` uses the measured complex phases,
    while ``noncoherent`` discards cross terms.
    """
    g = np.asarray(effective_csi, dtype=np.complex128)
    if g.ndim != 2 or g.shape[1] != NUM_RUS:
        raise ValueError(f"effective_csi segment must be (ticks,{NUM_RUS}), got {g.shape}")
    selected = g[:, list(state)]
    count = selected.shape[1]
    if count < 1:
        raise ValueError("an RU action cannot be empty")
    if mode == "measured-phase":
        power_t = np.abs(np.sum(selected, axis=1) / math.sqrt(count)) ** 2
    elif mode == "noncoherent":
        power_t = np.sum(np.abs(selected) ** 2, axis=1) / count
    elif mode == "phase-aligned-expected":
        amplitude = np.abs(selected)
        coherent = np.sum(amplitude**2, axis=1)
        retention = math.exp(-(float(phase_error_std_rad) ** 2))
        for left, right in itertools.combinations(range(count), 2):
            coherent += 2.0 * retention * amplitude[:, left] * amplitude[:, right]
        power_t = coherent / count
    else:
        raise ValueError(f"unknown combining mode: {mode}")
    return float(np.mean(np.maximum(power_t, 0.0)))


@dataclass(frozen=True)
class Episode:
    effective_csi: np.ndarray  # (L,3), complex; already includes the active f
    position_m: np.ndarray  # (L,2|3)
    heading_xy: np.ndarray  # (L,2), [cos(yaw), sin(yaw)]
    dt: float
    source: str

    @property
    def length(self) -> int:
        return int(self.effective_csi.shape[0])


def _as_batched(array: np.ndarray, single_ndim: int, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == single_ndim:
        return value[None, ...]
    if value.ndim == single_ndim + 1:
        return value
    raise ValueError(f"{name} must have {single_ndim} or {single_ndim + 1} dimensions, got {value.shape}")


def _heading_from_npz(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "ue_direction_xy" in data:
        direction = _as_batched(np.asarray(data["ue_direction_xy"], np.float32), 2, "ue_direction_xy")
    elif "ue_yaw_rad" in data or "ue_yaw_deg" in data:
        key = "ue_yaw_rad" if "ue_yaw_rad" in data else "ue_yaw_deg"
        yaw = _as_batched(np.asarray(data[key], np.float64), 1, key)
        if key.endswith("_deg"):
            yaw = np.deg2rad(yaw)
        direction = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1).astype(np.float32)
    else:
        raise KeyError("NPZ needs ue_yaw_rad, ue_yaw_deg, or ue_direction_xy")
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 1e-8):
        raise ValueError("UE heading contains a zero or non-finite direction")
    return (direction / norm).astype(np.float32)


def _windows_from_arrays(
    effective: np.ndarray,
    position: np.ndarray,
    heading: np.ndarray,
    dt: float,
    source: str,
    window_ticks: int,
    stride_ticks: int,
) -> list[Episode]:
    if effective.shape[0] != position.shape[0] or effective.shape[0] != heading.shape[0]:
        raise ValueError(
            f"time length mismatch in {source}: CSI={effective.shape}, position={position.shape}, "
            f"heading={heading.shape}"
        )
    if effective.ndim != 2 or effective.shape[1] != NUM_RUS:
        raise ValueError(f"effective CSI in {source} must be (L,{NUM_RUS}), got {effective.shape}")
    if position.ndim != 2 or position.shape[1] not in (2, 3):
        raise ValueError(f"position in {source} must be (L,2|3), got {position.shape}")
    if heading.shape != (effective.shape[0], 2):
        raise ValueError(f"heading in {source} must be (L,2), got {heading.shape}")
    finite = (
        np.isfinite(effective.real).all(axis=1)
        & np.isfinite(effective.imag).all(axis=1)
        & np.isfinite(position).all(axis=1)
        & np.isfinite(heading).all(axis=1)
    )
    episodes: list[Episode] = []
    start = 0
    while start < len(finite):
        while start < len(finite) and not finite[start]:
            start += 1
        end = start
        while end < len(finite) and finite[end]:
            end += 1
        for window_start in range(start, end - window_ticks + 1, stride_ticks):
            window_end = window_start + window_ticks
            episodes.append(Episode(
                effective_csi=np.asarray(effective[window_start:window_end], np.complex64).copy(),
                position_m=np.asarray(position[window_start:window_end], np.float32).copy(),
                heading_xy=np.asarray(heading[window_start:window_end], np.float32).copy(),
                dt=float(dt),
                source=source,
            ))
        start = max(end, start + 1)
    return episodes


def _load_npz_episodes(
    path: Path,
    window_sec: float,
    stride_sec: float,
) -> list[Episode]:
    with np.load(path, allow_pickle=False) as data:
        dt = float(data["dt"]) if "dt" in data else DEFAULT_DT
        if dt <= 0.0:
            raise ValueError(f"{path}: dt must be positive")
        if "effective_csi" in data:
            effective = _as_batched(np.asarray(data["effective_csi"]), 2, "effective_csi")
        elif "channel_csi" in data and "beamforming_vectors" in data:
            channel = _as_batched(np.asarray(data["channel_csi"]), 3, "channel_csi")
            effective = effective_csi_from_channel(channel, np.asarray(data["beamforming_vectors"]))
        else:
            raise KeyError(
                f"{path}: need effective_csi, or channel_csi plus beamforming_vectors; "
                "raw X/channel gain is intentionally not accepted"
            )
        position = _as_batched(np.asarray(data["ue_position_m"], np.float32), 2, "ue_position_m")
        heading = _heading_from_npz(data)

    if not (effective.shape[:2] == position.shape[:2] == heading.shape[:2]):
        raise ValueError(
            f"{path}: batch/time mismatch: CSI={effective.shape}, position={position.shape}, "
            f"heading={heading.shape}"
        )
    window_ticks = int(round(window_sec / dt))
    stride_ticks = int(round(stride_sec / dt))
    if window_ticks < 2 or stride_ticks < 1:
        raise ValueError("window/stride are too short for the dataset dt")
    episodes: list[Episode] = []
    for index in range(effective.shape[0]):
        episodes.extend(_windows_from_arrays(
            effective[index], position[index], heading[index], dt,
            f"{path.resolve()}#{index}", window_ticks, stride_ticks,
        ))
    return episodes


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _load_csv_episodes(
    path: Path,
    window_sec: float,
    stride_sec: float,
    max_gap_s: float,
) -> list[Episode]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        return []
    time_s = np.asarray([_float(row, "host_rx_time") for row in rows], np.float64)
    finite_time = time_s[np.isfinite(time_s)]
    if finite_time.size < 2:
        raise ValueError(f"{path}: not enough finite host_rx_time values")
    dt = float(np.median(np.diff(finite_time)))
    if dt <= 0.0:
        raise ValueError(f"{path}: non-positive measured dt")

    effective = np.empty((len(rows), NUM_RUS), np.complex64)
    for ru in range(1, NUM_RUS + 1):
        effective[:, ru - 1] = np.asarray([
            complex(_float(row, f"ru{ru}_h_re"), _float(row, f"ru{ru}_h_im"))
            for row in rows
        ], np.complex64)
    position = np.asarray([
        [_float(row, "ue_x_m"), _float(row, "ue_y_m")]
        for row in rows
    ], np.float32)
    yaw_rad = np.deg2rad(np.asarray([_float(row, "ue_yaw_deg") for row in rows], np.float64))
    heading = np.stack([np.cos(yaw_rad), np.sin(yaw_rad)], axis=-1).astype(np.float32)
    valid = np.asarray([
        row.get("pose_valid", "1").lower() in {"1", "true"}
        and row.get("phase", "").lower() not in {"settle", "waiting"}
        for row in rows
    ], dtype=bool)
    valid &= np.isfinite(time_s)
    gaps = np.r_[False, np.diff(time_s) > float(max_gap_s)]
    valid[gaps] = False
    effective[~valid] = np.nan + 1j * np.nan
    position[~valid] = np.nan
    heading[~valid] = np.nan

    window_ticks = int(round(window_sec / dt))
    stride_ticks = int(round(stride_sec / dt))
    return _windows_from_arrays(
        effective, position, heading, dt, str(path.resolve()), window_ticks, stride_ticks
    )


def discover_inputs(paths: Iterable[str | Path]) -> list[Path]:
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".npz":
            found.add(path)
        elif path.is_file() and path.name == "dft_measurements.csv":
            found.add(path)
        elif path.is_dir() and (path / "dft_measurements.csv").is_file():
            found.add(path / "dft_measurements.csv")
        elif path.is_dir():
            found.update(path.rglob("dft_measurements.csv"))
            found.update(path.glob("*.npz"))
        else:
            raise FileNotFoundError(path)
    return sorted(found)


class BeamPositionStore:
    """Fixed-duration beamformed-CSI/pose episodes for one dataset split."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        window_sec: float,
        stride_sec: float,
        max_gap_s: float,
    ):
        self.input_paths = discover_inputs(paths)
        if not self.input_paths:
            raise FileNotFoundError("no dft_measurements.csv or NPZ input was found")
        episodes: list[Episode] = []
        for path in self.input_paths:
            if path.suffix.lower() == ".npz":
                episodes.extend(_load_npz_episodes(path, window_sec, stride_sec))
            else:
                episodes.extend(_load_csv_episodes(path, window_sec, stride_sec, max_gap_s))
        if not episodes:
            raise ValueError("no valid fixed-duration episodes could be built")
        reference = episodes[0]
        for episode in episodes[1:]:
            if episode.length != reference.length:
                raise ValueError("all episodes in a split must have the same length")
            if episode.position_m.shape[1] != reference.position_m.shape[1]:
                raise ValueError("all episodes must use the same position dimension")
            if not math.isclose(episode.dt, reference.dt, rel_tol=0.0, abs_tol=reference.dt * 0.02):
                raise ValueError(f"mixed sample intervals: {reference.dt} and {episode.dt}")
        self.episodes = episodes
        self.dt = reference.dt
        self.length = reference.length
        self.position_dim = int(reference.position_m.shape[1])

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> Episode:
        return self.episodes[int(index)]


def validate_split_compatibility(stores: Dict[str, BeamPositionStore]) -> None:
    train = stores["train"]
    seen: dict[Path, str] = {}
    for split, store in stores.items():
        if store.length != train.length:
            raise ValueError(f"{split} episode length {store.length} != train {train.length}")
        if store.position_dim != train.position_dim:
            raise ValueError(f"{split} position dimension differs from train")
        if not math.isclose(store.dt, train.dt, rel_tol=0.0, abs_tol=train.dt * 0.02):
            raise ValueError(f"{split} dt={store.dt}, train dt={train.dt}")
        for path in store.input_paths:
            if path in seen:
                raise ValueError(f"data leakage: {path} is present in both {seen[path]} and {split}")
            seen[path] = split


@dataclass(frozen=True)
class FeatureStats:
    gain_mean_db: float
    gain_std_db: float
    position_mean_m: Tuple[float, ...]
    position_std_m: Tuple[float, ...]


def compute_feature_stats(store: BeamPositionStore, gain_floor_db: float) -> FeatureStats:
    gain_sum = gain_sq = 0.0
    gain_count = 0
    pos_sum = np.zeros(store.position_dim, np.float64)
    pos_sq = np.zeros(store.position_dim, np.float64)
    pos_count = 0
    for episode in store.episodes:
        gain = effective_gain_db(episode.effective_csi, gain_floor_db).astype(np.float64)
        gain_sum += float(gain.sum())
        gain_sq += float(np.square(gain).sum())
        gain_count += gain.size
        pos = episode.position_m.astype(np.float64)
        pos_sum += pos.sum(axis=0)
        pos_sq += np.square(pos).sum(axis=0)
        pos_count += pos.shape[0]
    gain_mean = gain_sum / max(gain_count, 1)
    gain_var = gain_sq / max(gain_count, 1) - gain_mean**2
    position_mean = pos_sum / max(pos_count, 1)
    position_var = pos_sq / max(pos_count, 1) - position_mean**2
    return FeatureStats(
        gain_mean_db=float(gain_mean),
        gain_std_db=float(max(math.sqrt(max(gain_var, 0.0)), 1e-3)),
        position_mean_m=tuple(float(value) for value in position_mean),
        position_std_m=tuple(float(max(math.sqrt(max(value, 0.0)), 1e-3)) for value in position_var),
    )


def estimate_noise_power(
    store: BeamPositionStore,
    seg_len_ticks: int,
    target_snr_db: float,
) -> float:
    """Choose noise so the median strongest single RU has target SNR."""
    references = []
    for episode in store.episodes:
        for start in range(0, episode.length - seg_len_ticks + 1, seg_len_ticks):
            power = np.mean(np.abs(episode.effective_csi[start:start + seg_len_ticks]) ** 2, axis=0)
            references.append(float(np.max(power)))
    reference_power = float(np.median(references))
    if not math.isfinite(reference_power) or reference_power <= 0.0:
        raise ValueError("cannot estimate noise power from zero/non-finite effective CSI")
    return reference_power / (10.0 ** (float(target_snr_db) / 10.0))


@dataclass
class EnvConfig:
    seg_len_ticks: int
    obs_hist_bins: int = 3
    observe_all_rus: bool = True
    gain_floor_db: float = -120.0
    handover_penalty: float = 0.05
    two_ru_penalty: float = 0.0
    reward_offset: float = 3.0
    noise_power: float = 1e-6
    combining_mode: str = "phase-aligned-expected"
    phase_error_std_rad: float = math.pi / 8.0
    max_episode_steps: int = 0


class BeamPositionEnv:
    """Scheduler environment using beamformed effective CSI and UE geometry."""

    def __init__(
        self,
        store: BeamPositionStore,
        episode_indices: Sequence[int],
        cfg: EnvConfig,
        stats: FeatureStats,
    ):
        self.store = store
        self.episode_indices = list(map(int, episode_indices))
        if not self.episode_indices:
            raise ValueError("episode_indices cannot be empty")
        self.cfg = cfg
        self.stats = stats
        self.n_actions = len(ACTION_STATES)
        # Per history bin: gain z-score, gain visibility, position z-score,
        # heading cos/sin.  Append current connection one-hot.
        self.features_per_bin = 2 * NUM_RUS + store.position_dim + 2
        self.obs_dim = self.features_per_bin * cfg.obs_hist_bins + self.n_actions
        self.feature_names = self._feature_names()
        self._episode: Optional[Episode] = None
        self._seg_k = 0
        self._n_segs = 0
        self._steps = 0
        self._max_steps = 0
        self._conn_state = 0
        self._conn_hist: list[int] = []

    def _feature_names(self) -> list[str]:
        axes = ["x", "y", "z"][: self.store.position_dim]
        names = []
        for history in range(self.cfg.obs_hist_bins):
            prefix = f"hist{history}_"
            names += [prefix + f"effective_gain_db_z_ru{ru}" for ru in range(1, NUM_RUS + 1)]
            names += [prefix + f"gain_visible_ru{ru}" for ru in range(1, NUM_RUS + 1)]
            names += [prefix + f"ue_{axis}_m_z" for axis in axes]
            names += [prefix + "ue_heading_cos", prefix + "ue_heading_sin"]
        names += [f"current_connection_{name.lower()}" for name in ACTION_NAMES]
        return names

    def reset(self, episode_index: Optional[int] = None) -> np.ndarray:
        if episode_index is None:
            episode_index = random.choice(self.episode_indices)
        if int(episode_index) not in self.episode_indices:
            raise ValueError("episode_index is not part of this environment")
        episode = self.store[int(episode_index)]
        if episode.length % self.cfg.seg_len_ticks != 0:
            raise ValueError(
                f"episode length {episode.length} is not divisible by seg_len_ticks={self.cfg.seg_len_ticks}"
            )
        self._episode = episode
        self._seg_k = 0
        self._n_segs = episode.length // self.cfg.seg_len_ticks
        if self._n_segs < 2:
            raise ValueError("an episode needs at least two scheduling segments")
        self._steps = 0
        self._conn_state = int(np.random.randint(self.n_actions))
        self._conn_hist = [self._conn_state]
        full_steps = self._n_segs - 1
        self._max_steps = full_steps if self.cfg.max_episode_steps <= 0 else min(
            full_steps, self.cfg.max_episode_steps
        )
        return self._make_obs()

    def _segment_slice(self, segment: int) -> slice:
        start = int(segment) * self.cfg.seg_len_ticks
        return slice(start, start + self.cfg.seg_len_ticks)

    def _segment_features(self, segment: int, connected_action: int) -> np.ndarray:
        assert self._episode is not None
        sl = self._segment_slice(segment)
        gain_db = effective_gain_db(self._episode.effective_csi[sl], self.cfg.gain_floor_db).mean(axis=0)
        gain_z = (gain_db - self.stats.gain_mean_db) / self.stats.gain_std_db
        visible = np.ones(NUM_RUS, np.float32)
        if not self.cfg.observe_all_rus:
            allowed = set(ACTION_STATES[int(connected_action)])
            visible = np.asarray([float(ru in allowed) for ru in range(NUM_RUS)], np.float32)
            gain_z = gain_z * visible
        position = self._episode.position_m[sl].mean(axis=0)
        position_z = (
            position - np.asarray(self.stats.position_mean_m, np.float32)
        ) / np.asarray(self.stats.position_std_m, np.float32)
        heading = self._episode.heading_xy[sl].mean(axis=0)
        heading /= max(float(np.linalg.norm(heading)), 1e-8)
        return np.concatenate([
            gain_z.astype(np.float32), visible, position_z.astype(np.float32), heading.astype(np.float32)
        ])

    def _make_obs(self) -> np.ndarray:
        bins = []
        for bin_index in range(self.cfg.obs_hist_bins):
            segment = self._seg_k - (self.cfg.obs_hist_bins - 1 - bin_index)
            segment = max(0, min(segment, self._n_segs - 1))
            history_index = max(0, min(segment, len(self._conn_hist) - 1))
            bins.append(self._segment_features(segment, self._conn_hist[history_index]))
        connection = np.zeros(self.n_actions, np.float32)
        connection[self._conn_state] = 1.0
        observation = np.concatenate(bins + [connection]).astype(np.float32)
        if observation.shape != (self.obs_dim,):
            raise RuntimeError(f"internal observation shape mismatch: {observation.shape}")
        return observation

    def _state_rates(self, segment: int) -> np.ndarray:
        assert self._episode is not None
        effective = self._episode.effective_csi[self._segment_slice(segment)]
        power = np.asarray([
            expected_action_power(
                effective,
                state,
                mode=self.cfg.combining_mode,
                phase_error_std_rad=self.cfg.phase_error_std_rad,
            )
            for state in ACTION_STATES
        ], np.float64)
        return np.log2(1.0 + power / self.cfg.noise_power).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        action = int(np.clip(int(action), 0, self.n_actions - 1))
        connection_before = self._conn_state
        reward_segment = self._seg_k + 1
        state_rates = self._state_rates(reward_segment)
        handover = int(action != connection_before)
        two_rus = int(len(ACTION_STATES[action]) == 2)
        reward = (
            float(state_rates[action])
            - self.cfg.handover_penalty * handover
            - self.cfg.two_ru_penalty * two_rus
            - self.cfg.reward_offset
        )
        self._conn_state = action
        self._seg_k += 1
        self._steps += 1
        if len(self._conn_hist) <= self._seg_k:
            self._conn_hist.append(action)
        done = self._steps >= self._max_steps
        observation = self._make_obs()
        info = {
            "source": self._episode.source if self._episode else "",
            "segment": self._seg_k,
            "reward_segment": reward_segment,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "selected_state_rate": float(state_rates[action]),
            "best_state_rate": float(np.max(state_rates)),
            "is_best_action": int(action == int(np.argmax(state_rates))),
            "handover_attempt": handover,
            "connection_before": connection_before,
            "connection_after": action,
        }
        return observation, reward, done, info


class VecEnv:
    def __init__(self, envs: Sequence[BeamPositionEnv]):
        if not envs:
            raise ValueError("envs cannot be empty")
        self.envs = list(envs)
        self.n = len(self.envs)
        self.obs_dim = self.envs[0].obs_dim
        self.n_actions = self.envs[0].n_actions

    def reset(self) -> np.ndarray:
        return np.stack([environment.reset() for environment in self.envs])

    def step(self, actions: np.ndarray):
        observations, rewards, dones, infos = [], [], [], []
        for environment, action in zip(self.envs, actions):
            observation, reward, done, info = environment.step(int(action))
            if done:
                observation = environment.reset()
            info["episode_reset"] = bool(done)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(observations).astype(np.float32),
            np.asarray(rewards, np.float32),
            np.asarray(dones, np.bool_),
            infos,
        )


class GRUQNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.n_actions = int(n_actions)
        self.gru = nn.GRUCell(self.obs_dim, self.hidden_size)
        self.q_head = nn.Linear(self.hidden_size, self.n_actions)
        self.h0 = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)

    def init_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.h0.to(device).unsqueeze(0).expand(batch, -1).clone()

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor):
        next_hidden = self.gru(observation, hidden)
        return self.q_head(next_hidden), next_hidden


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, hidden_size: int):
        self.capacity = int(capacity)
        self.observation = np.zeros((capacity, obs_dim), np.float32)
        self.hidden = np.zeros((capacity, hidden_size), np.float32)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.next_observation = np.zeros((capacity, obs_dim), np.float32)
        self.next_hidden = np.zeros((capacity, hidden_size), np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self, observation, hidden, action, reward, next_observation, next_hidden, done
    ) -> None:
        count = int(len(action))
        indices = (np.arange(count) + self.position) % self.capacity
        self.observation[indices] = observation
        self.hidden[indices] = hidden
        self.action[indices] = action
        self.reward[indices] = reward
        self.next_observation[indices] = next_observation
        self.next_hidden[indices] = next_hidden
        self.done[indices] = done
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = np.random.randint(0, self.size, size=int(batch_size))
        arrays = (
            self.observation[indices], self.hidden[indices], self.action[indices],
            self.reward[indices], self.next_observation[indices],
            self.next_hidden[indices], self.done[indices],
        )
        return tuple(torch.from_numpy(array).to(device) for array in arrays)


@dataclass
class DQNConfig:
    total_env_steps: int
    num_envs: int
    gamma: float
    learning_rate: float
    batch_size: int
    replay_size: int
    learning_starts: int
    train_every: int
    gradient_steps: int
    target_update_interval: int
    max_grad_norm: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int


@torch.no_grad()
def evaluate(
    q_network: GRUQNetwork,
    store: BeamPositionStore,
    env_config: EnvConfig,
    stats: FeatureStats,
    device: torch.device,
    maximum_episodes: int,
    seed: int,
) -> dict[str, float]:
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    q_network.eval()
    try:
        indices = list(range(len(store)))
        random.shuffle(indices)
        if maximum_episodes > 0:
            indices = indices[:maximum_episodes]
        rewards, best_ratios, handovers = [], [], []
        for index in indices:
            environment = BeamPositionEnv(store, [index], env_config, stats)
            observation = environment.reset(index)
            hidden = q_network.init_hidden(1, device)
            total_reward = best = handover = steps = 0.0
            done = False
            while not done:
                tensor = torch.from_numpy(observation).to(device).view(1, -1)
                q_values, hidden = q_network(tensor, hidden)
                action = int(q_values.argmax(dim=-1).item())
                observation, reward, done, info = environment.step(action)
                total_reward += reward
                best += info["is_best_action"]
                handover += info["handover_attempt"]
                steps += 1
            rewards.append(total_reward / max(steps, 1))
            best_ratios.append(best / max(steps, 1))
            handovers.append(handover / max(steps, 1))
        return {
            "reward_mean": float(np.mean(rewards)),
            "best_action_ratio": float(np.mean(best_ratios)),
            "handover_attempt_per_step": float(np.mean(handovers)),
            "n_episodes": len(indices),
        }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def checkpoint(
    q_network: GRUQNetwork,
    target_network: GRUQNetwork,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    env_config: EnvConfig,
    dqn_config: DQNConfig,
    stats: FeatureStats,
    train_store: BeamPositionStore,
    feature_names: Sequence[str],
    args: argparse.Namespace,
    validation_metrics: Optional[dict],
) -> dict:
    return {
        "schema": "fr3-beam-position-ru-ddqn-v1",
        "global_step": int(global_step),
        "q_net": cpu_state_dict(q_network),
        "target_net": cpu_state_dict(target_network),
        "optimizer": optimizer.state_dict(),
        "env_cfg": asdict(env_config),
        "dqn_cfg": asdict(dqn_config),
        "feature_stats": asdict(stats),
        "feature_names": list(feature_names),
        "validation_metrics": validation_metrics,
        "args": vars(args),
        "data_meta": {
            "input_csi": "effective_csi_after_beamforming",
            "effective_csi_equation": "h_eff = h^H f",
            "effective_gain_feature": "10*log10(|h_eff|^2)",
            "position_dim": train_store.position_dim,
            "num_rus": NUM_RUS,
            "action_states": [list(state) for state in ACTION_STATES],
            "action_names": list(ACTION_NAMES),
            "dt": train_store.dt,
            "window_length": train_store.length,
            "training_sources": [str(path) for path in train_store.input_paths],
        },
        "model_meta": {
            "obs_dim": q_network.obs_dim,
            "hidden_size": q_network.hidden_size,
            "n_actions": q_network.n_actions,
            "n_params": count_trainable_params(q_network),
            "algorithm": "double_dqn_gru_beam_position",
        },
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--validation-data", nargs="+", required=True)
    parser.add_argument("--test-data", nargs="+", required=True)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints_ru_scheduler_beam_position"))
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--window-stride-sec", type=float, default=0.5)
    parser.add_argument("--max-gap-sec", type=float, default=0.03)
    parser.add_argument("--segment-ms", type=float, default=100.0)
    parser.add_argument("--obs-history-bins", type=int, default=3)
    parser.add_argument(
        "--observe-all-rus", action=argparse.BooleanOptionalAction, default=True,
        help="all FDM pilot gains are visible; disable if inactive RU pilots are unavailable",
    )
    parser.add_argument("--gain-floor-db", type=float, default=-120.0)
    parser.add_argument("--handover-penalty", type=float, default=0.05)
    parser.add_argument("--two-ru-penalty", type=float, default=0.0)
    parser.add_argument("--reward-offset", type=float, default=3.0)
    parser.add_argument("--noise-power", type=float, default=None)
    parser.add_argument(
        "--reward-snr-db", type=float, default=10.0,
        help="used only to auto-calibrate noise when --noise-power is omitted",
    )
    parser.add_argument(
        "--combining-mode",
        choices=("phase-aligned-expected", "measured-phase", "noncoherent"),
        default="phase-aligned-expected",
    )
    parser.add_argument("--phase-error-std-deg", type=float, default=22.5)
    parser.add_argument("--max-episode-steps", type=int, default=0)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eval-seed", type=int, default=456)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--total-env-steps", type=int, default=1_500_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=2_000)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=150_000)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--validation-max-episodes", type=int, default=200)
    parser.add_argument("--test-max-episodes", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> dict:
    args = parse_args(argv)
    if args.window_sec <= 0.0 or args.window_stride_sec <= 0.0:
        raise ValueError("window durations must be positive")
    if args.obs_history_bins < 1:
        raise ValueError("obs-history-bins must be positive")
    seed_all(args.seed)
    device = resolve_device(args.device)
    print(f"[DEVICE] {device}")

    stores = {
        "train": BeamPositionStore(args.train_data, args.window_sec, args.window_stride_sec, args.max_gap_sec),
        "validation": BeamPositionStore(
            args.validation_data, args.window_sec, args.window_stride_sec, args.max_gap_sec
        ),
        "test": BeamPositionStore(args.test_data, args.window_sec, args.window_stride_sec, args.max_gap_sec),
    }
    validate_split_compatibility(stores)
    train_store = stores["train"]
    segment_seconds = args.segment_ms / 1000.0
    segment_ticks = int(round(segment_seconds / train_store.dt))
    if segment_ticks <= 0 or train_store.length % segment_ticks != 0:
        raise ValueError(
            f"window length {train_store.length} must be divisible by segment ticks {segment_ticks}"
        )
    stats = compute_feature_stats(train_store, args.gain_floor_db)
    noise_power = (
        float(args.noise_power)
        if args.noise_power is not None
        else estimate_noise_power(train_store, segment_ticks, args.reward_snr_db)
    )
    if noise_power <= 0.0 or not math.isfinite(noise_power):
        raise ValueError("noise power must be finite and positive")
    for split, store in stores.items():
        print(
            f"[DATA] {split:10s} episodes={len(store):5d} length={store.length} "
            f"dt={store.dt:.6f}s sources={len(store.input_paths)}"
        )
    print(
        f"[INPUT] h_eff=h^Hf · gain=10log10(|h_eff|^2), position={train_store.position_dim}D, "
        f"heading=sin/cos, noise={noise_power:.3e}"
    )

    env_config = EnvConfig(
        seg_len_ticks=segment_ticks,
        obs_hist_bins=args.obs_history_bins,
        observe_all_rus=args.observe_all_rus,
        gain_floor_db=args.gain_floor_db,
        handover_penalty=args.handover_penalty,
        two_ru_penalty=args.two_ru_penalty,
        reward_offset=args.reward_offset,
        noise_power=noise_power,
        combining_mode=args.combining_mode,
        phase_error_std_rad=math.radians(args.phase_error_std_deg),
        max_episode_steps=args.max_episode_steps,
    )
    dqn_config = DQNConfig(
        total_env_steps=args.total_env_steps,
        num_envs=args.num_envs,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        learning_starts=args.learning_starts,
        train_every=args.train_every,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        max_grad_norm=args.max_grad_norm,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    training_indices = list(range(len(train_store)))
    vector_env = VecEnv([
        BeamPositionEnv(train_store, training_indices, env_config, stats)
        for _ in range(dqn_config.num_envs)
    ])
    observation = vector_env.reset()
    feature_names = vector_env.envs[0].feature_names

    q_network = GRUQNetwork(vector_env.obs_dim, args.hidden_size, vector_env.n_actions).to(device)
    target_network = GRUQNetwork(vector_env.obs_dim, args.hidden_size, vector_env.n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    for parameter in target_network.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=dqn_config.learning_rate, eps=1e-5)
    replay = ReplayBuffer(dqn_config.replay_size, vector_env.obs_dim, q_network.hidden_size)
    print(
        f"[MODEL] obs={q_network.obs_dim} hidden={q_network.hidden_size} "
        f"actions={q_network.n_actions} params={count_trainable_params(q_network)}"
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    hidden = q_network.init_hidden(dqn_config.num_envs, device).cpu().numpy()
    recent_reward, recent_best, recent_handover = (deque(maxlen=100) for _ in range(3))
    recent_loss = deque(maxlen=200)
    running_reward = np.zeros(dqn_config.num_envs, np.float64)
    running_best = np.zeros(dqn_config.num_envs, np.float64)
    running_handover = np.zeros(dqn_config.num_envs, np.float64)
    running_steps = np.zeros(dqn_config.num_envs, np.int64)
    global_step = environment_iterations = last_target_sync = 0
    next_log, next_eval, next_save = args.log_every, args.eval_every, args.save_every
    best_validation_reward = -math.inf
    best_state = None
    best_metrics = None
    best_step = 0
    started = time.time()

    while global_step < dqn_config.total_env_steps:
        q_network.train()
        epsilon = linear_schedule(
            global_step, dqn_config.epsilon_start, dqn_config.epsilon_end,
            dqn_config.epsilon_decay_steps,
        )
        obs_tensor = torch.from_numpy(observation).to(device)
        hidden_tensor = torch.from_numpy(hidden).to(device)
        with torch.no_grad():
            q_values, next_hidden_tensor = q_network(obs_tensor, hidden_tensor)
        greedy = q_values.argmax(dim=-1).cpu().numpy()
        random_actions = np.random.randint(0, vector_env.n_actions, dqn_config.num_envs)
        actions = np.where(np.random.rand(dqn_config.num_envs) < epsilon, random_actions, greedy)
        next_observation, rewards, dones, infos = vector_env.step(actions)
        next_hidden = next_hidden_tensor.cpu().numpy().astype(np.float32)
        next_hidden[dones] = 0.0

        for index, info in enumerate(infos):
            running_reward[index] += rewards[index]
            running_best[index] += info["is_best_action"]
            running_handover[index] += info["handover_attempt"]
            running_steps[index] += 1
            if dones[index]:
                length = max(int(running_steps[index]), 1)
                recent_reward.append(float(running_reward[index] / length))
                recent_best.append(float(running_best[index] / length))
                recent_handover.append(float(running_handover[index] / length))
                running_reward[index] = running_best[index] = running_handover[index] = 0.0
                running_steps[index] = 0

        replay.add_batch(
            observation, hidden, actions, rewards, next_observation, next_hidden, dones.astype(np.float32)
        )
        observation, hidden = next_observation, next_hidden
        environment_iterations += 1
        global_step += dqn_config.num_envs

        if (
            global_step >= dqn_config.learning_starts
            and len(replay) >= dqn_config.batch_size
            and environment_iterations % dqn_config.train_every == 0
        ):
            for _ in range(dqn_config.gradient_steps):
                obs_b, hidden_b, action_b, reward_b, next_obs_b, next_hidden_b, done_b = replay.sample(
                    dqn_config.batch_size, device
                )
                q_all, _ = q_network(obs_b, hidden_b)
                q_selected = q_all.gather(1, action_b[:, None]).squeeze(1)
                with torch.no_grad():
                    online_next, _ = q_network(next_obs_b, next_hidden_b)
                    next_actions = online_next.argmax(dim=-1, keepdim=True)
                    target_next, _ = target_network(next_obs_b, next_hidden_b)
                    bootstrap = target_next.gather(1, next_actions).squeeze(1)
                    target = reward_b + dqn_config.gamma * (1.0 - done_b) * bootstrap
                loss = F.smooth_l1_loss(q_selected, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), dqn_config.max_grad_norm)
                optimizer.step()
                recent_loss.append(float(loss.item()))

        if global_step - last_target_sync >= dqn_config.target_update_interval:
            target_network.load_state_dict(q_network.state_dict())
            last_target_sync = global_step

        mean = lambda values: float(np.mean(values)) if values else float("nan")
        if args.log_every > 0 and global_step >= next_log:
            print(
                f"[TRAIN {global_step:9d}] eps={epsilon:.3f} loss={mean(recent_loss):.4f} "
                f"reward={mean(recent_reward):.4f} best={mean(recent_best):.3f} "
                f"handover={mean(recent_handover):.3f} elapsed={(time.time()-started)/60:.1f}min"
            )
            next_log += args.log_every

        if args.eval_every > 0 and global_step >= next_eval:
            metrics = evaluate(
                q_network, stores["validation"], env_config, stats, device,
                args.validation_max_episodes, args.eval_seed,
            )
            print(f"[VALIDATION {global_step:9d}] {json.dumps(metrics, ensure_ascii=False)}")
            if metrics["reward_mean"] > best_validation_reward:
                best_validation_reward = metrics["reward_mean"]
                best_state = cpu_state_dict(q_network)
                best_metrics = metrics
                best_step = global_step
                torch.save(
                    checkpoint(
                        q_network, target_network, optimizer, global_step, env_config,
                        dqn_config, stats, train_store, feature_names, args, metrics,
                    ),
                    args.save_dir / "best_validation.pt",
                )
            next_eval += args.eval_every

        if args.save_every > 0 and global_step >= next_save:
            torch.save(
                checkpoint(
                    q_network, target_network, optimizer, global_step, env_config,
                    dqn_config, stats, train_store, feature_names, args, best_metrics,
                ),
                args.save_dir / f"scheduler_step_{global_step:09d}.pt",
            )
            next_save += args.save_every

    if best_state is None:
        best_metrics = evaluate(
            q_network, stores["validation"], env_config, stats, device,
            args.validation_max_episodes, args.eval_seed,
        )
        best_state = cpu_state_dict(q_network)
        best_step = global_step
    q_network.load_state_dict(best_state)
    test_metrics = evaluate(
        q_network, stores["test"], env_config, stats, device,
        args.test_max_episodes, args.eval_seed + 1,
    )
    result = {"best_validation_step": best_step, "validation": best_metrics, "test": test_metrics}
    final = checkpoint(
        q_network, target_network, optimizer, best_step, env_config, dqn_config,
        stats, train_store, feature_names, args, best_metrics,
    )
    final["test_metrics"] = test_metrics
    torch.save(final, args.save_dir / "final_best_model.pt")
    (args.save_dir / "final_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[TEST] {json.dumps(test_metrics, ensure_ascii=False)}")
    print(f"[DONE] {args.save_dir.resolve()}")
    return result


if __name__ == "__main__":
    main()
