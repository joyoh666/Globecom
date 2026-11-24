#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Far-field LOS simulation for position-based 2-D DFT beam selection.

This module models a *fully controllable* rectangular UPA.  It is deliberately
separate from :mod:`beam.dft_codebook`, which represents the current prototype
hardware's four-element azimuth control axis.

Conventions
-----------
``position_m`` is ``[x, y, z]`` in a common Cartesian frame.  RU azimuth is
counter-clockwise from +x and elevation is positive upward.  The local UPA
axes are horizontal-right (``u``) and vertical-up (``v``); boresight is normal
to the panel.  Beam weights and channels have unit-norm steering-vector
conventions, so a perfectly matched 4x4 beam has a 16x (12.04 dB) power gain
over one isotropic element.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np


C_M_PER_S = 299_792_458.0


def power_db(value: float, floor: float = 1e-300) -> float:
    """Convert a non-negative linear power quantity to dB."""
    return 10.0 * math.log10(max(float(value), floor))


def ratio_db(numerator: float, denominator: float) -> float:
    """Power ratio in dB."""
    return power_db(float(numerator) / max(float(denominator), 1e-300))


@dataclass(frozen=True)
class UPADFTBeam:
    index: int
    horizontal_bin: int
    vertical_bin: int
    u: float
    v: float
    azimuth_deg: float
    elevation_deg: float
    off_boresight_deg: float


class UPADFTCodebook:
    """Oversampled 2-D DFT codebook for a rectangular UPA."""

    def __init__(
        self,
        rows: int,
        columns: int,
        oversampling: int,
        rf_hz: float,
        element_spacing_m: float,
        max_abs_azimuth_deg: float = 60.0,
        max_abs_elevation_deg: float = 60.0,
    ):
        if rows < 2 or columns < 2:
            raise ValueError("rows and columns must both be at least 2")
        if oversampling < 1:
            raise ValueError("oversampling must be positive")
        if rf_hz <= 0.0 or element_spacing_m <= 0.0:
            raise ValueError("rf_hz and element_spacing_m must be positive")
        if not 0.0 < max_abs_azimuth_deg < 90.0:
            raise ValueError("max_abs_azimuth_deg must be in (0, 90)")
        if not 0.0 < max_abs_elevation_deg < 90.0:
            raise ValueError("max_abs_elevation_deg must be in (0, 90)")

        self.rows = int(rows)
        self.columns = int(columns)
        self.oversampling = int(oversampling)
        self.rf_hz = float(rf_hz)
        self.element_spacing_m = float(element_spacing_m)
        self.wavelength_m = C_M_PER_S / self.rf_hz
        self.spacing_lambda = self.element_spacing_m / self.wavelength_m
        self.max_abs_azimuth_deg = float(max_abs_azimuth_deg)
        self.max_abs_elevation_deg = float(max_abs_elevation_deg)
        self.horizontal_dft_size = self.columns * self.oversampling
        self.vertical_dft_size = self.rows * self.oversampling

        beams: list[UPADFTBeam] = []
        for kv in self._shifted_bins(self.vertical_dft_size):
            v = kv / (self.vertical_dft_size * self.spacing_lambda)
            for ku in self._shifted_bins(self.horizontal_dft_size):
                u = ku / (self.horizontal_dft_size * self.spacing_lambda)
                radial_sq = u * u + v * v
                if radial_sq >= 1.0:
                    continue
                forward = math.sqrt(max(0.0, 1.0 - radial_sq))
                azimuth = math.degrees(math.atan2(u, forward))
                elevation = math.degrees(math.asin(v))
                if abs(azimuth) > self.max_abs_azimuth_deg + 1e-12:
                    continue
                if abs(elevation) > self.max_abs_elevation_deg + 1e-12:
                    continue
                beams.append(
                    UPADFTBeam(
                        index=len(beams),
                        horizontal_bin=int(ku),
                        vertical_bin=int(kv),
                        u=float(u),
                        v=float(v),
                        azimuth_deg=float(azimuth),
                        elevation_deg=float(elevation),
                        off_boresight_deg=math.degrees(math.acos(forward)),
                    )
                )
        if not beams:
            raise ValueError("configured steering sector produced no DFT beams")
        self.beams = tuple(beams)

        horizontal = (np.arange(self.columns, dtype=float) - (self.columns - 1.0) / 2.0)
        vertical = (np.arange(self.rows, dtype=float) - (self.rows - 1.0) / 2.0)
        xx, yy = np.meshgrid(horizontal, vertical)
        self._horizontal_positions_m = xx.reshape(-1) * self.element_spacing_m
        self._vertical_positions_m = yy.reshape(-1) * self.element_spacing_m

    @staticmethod
    def _shifted_bins(size: int) -> range:
        return range(-(size // 2), (size + 1) // 2)

    @property
    def num_elements(self) -> int:
        return self.rows * self.columns

    def __len__(self) -> int:
        return len(self.beams)

    def __iter__(self) -> Iterable[UPADFTBeam]:
        return iter(self.beams)

    def __getitem__(self, index: int) -> UPADFTBeam:
        return self.beams[int(index)]

    def steering_vector(self, u: float, v: float) -> np.ndarray:
        """Return a unit-norm UPA steering vector for direction cosines u/v."""
        if float(u) ** 2 + float(v) ** 2 >= 1.0 + 1e-12:
            raise ValueError("u/v direction cosines are outside the visible region")
        phase = (2.0 * math.pi / self.wavelength_m) * (
            self._horizontal_positions_m * float(u) + self._vertical_positions_m * float(v)
        )
        return np.exp(1j * phase) / math.sqrt(self.num_elements)

    def weights(self, beam_or_index: UPADFTBeam | int) -> np.ndarray:
        beam = beam_or_index if isinstance(beam_or_index, UPADFTBeam) else self[int(beam_or_index)]
        return self.steering_vector(beam.u, beam.v)

    def nearest_direction(self, u: float, v: float) -> UPADFTBeam:
        """Select the DFT sample nearest in spatial-frequency coordinates."""
        return min(self.beams, key=lambda beam: (beam.u - float(u)) ** 2 + (beam.v - float(v)) ** 2)

    def oracle_beam(self, true_steering: np.ndarray) -> tuple[UPADFTBeam, float]:
        """Return the codebook beam that maximizes array overlap power."""
        scores = [abs(np.vdot(true_steering, self.weights(beam))) ** 2 for beam in self.beams]
        index = int(np.argmax(scores))
        return self.beams[index], float(scores[index])

    def to_dict(self) -> dict:
        return {
            "type": "oversampled_2d_dft_upa",
            "rows": self.rows,
            "columns": self.columns,
            "num_elements": self.num_elements,
            "oversampling": self.oversampling,
            "rf_hz": self.rf_hz,
            "wavelength_m": self.wavelength_m,
            "element_spacing_m": self.element_spacing_m,
            "spacing_lambda": self.spacing_lambda,
            "beam_count": len(self),
            "beams": [asdict(beam) for beam in self.beams],
        }


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return value / norm


def ru_local_direction(ru: dict, ue_position_m: Sequence[float]) -> dict[str, float]:
    """Convert the RU-to-UE direction into the panel's local coordinates."""
    ru_position = np.asarray(ru["position_m"], dtype=float)
    ue_position = np.asarray(ue_position_m, dtype=float)
    if ru_position.shape != (3,) or ue_position.shape != (3,):
        raise ValueError("RU and UE positions must be [x, y, z]")
    delta = ue_position - ru_position
    distance = float(np.linalg.norm(delta))
    if distance <= 0.0:
        raise ValueError("RU and UE positions must be different")
    direction = delta / distance

    azimuth = math.radians(float(ru.get("boresight_azimuth_deg", 0.0)))
    elevation = math.radians(float(ru.get("boresight_elevation_deg", 0.0)))
    boresight = np.asarray(
        [math.cos(elevation) * math.cos(azimuth),
         math.cos(elevation) * math.sin(azimuth),
         math.sin(elevation)],
        dtype=float,
    )
    right = np.asarray([-math.sin(azimuth), math.cos(azimuth), 0.0], dtype=float)
    up = _unit(np.cross(boresight, right))

    u = float(np.dot(direction, right))
    v = float(np.dot(direction, up))
    forward = float(np.dot(direction, boresight))
    if forward <= 0.0:
        raise ValueError(f"UE is behind RU {ru.get('id', '?')} panel (forward cosine={forward:.3f})")
    return {
        "distance_m": distance,
        "u": u,
        "v": v,
        "forward": forward,
        "azimuth_deg": math.degrees(math.atan2(u, forward)),
        "elevation_deg": math.degrees(math.atan2(v, math.hypot(u, forward))),
        "off_boresight_deg": math.degrees(math.acos(max(-1.0, min(1.0, forward)))),
    }


def free_space_power_gain(distance_m: float, wavelength_m: float) -> float:
    """Friis free-space power gain between isotropic antennas."""
    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive")
    return (float(wavelength_m) / (4.0 * math.pi * float(distance_m))) ** 2


def phase_only_precoder(
    predicted_effective_channels: Sequence[complex],
    reference_index: int = 0,
) -> np.ndarray:
    """Return unit-magnitude RU phases that align predicted effective CSI.

    If ``g_hat[r]`` is the predicted scalar channel after applying RU ``r``'s
    selected DFT beam, this returns ``exp(j*(angle(g_hat[ref])-angle(g_hat[r])))``.
    The common reference phase does not affect received power and keeps the
    reference RU's additional phase equal to zero.
    """
    predicted = np.asarray(predicted_effective_channels, dtype=np.complex128)
    if predicted.ndim != 1 or predicted.size == 0:
        raise ValueError("predicted_effective_channels must be a non-empty 1-D sequence")
    if not 0 <= int(reference_index) < predicted.size:
        raise ValueError("reference_index is outside predicted_effective_channels")
    if not np.all(np.isfinite(predicted.real)) or not np.all(np.isfinite(predicted.imag)):
        raise ValueError("predicted effective CSI must be finite")
    if np.any(np.abs(predicted) <= 1e-15):
        raise ValueError("predicted effective CSI magnitude must be non-zero")
    phases = np.angle(predicted)
    return np.exp(1j * (phases[int(reference_index)] - phases))


def simulate_position(
    rus: Sequence[dict],
    ue_position_m: Sequence[float],
    codebook: UPADFTCodebook,
    predicted_effective_channels: Sequence[complex] | None = None,
) -> dict:
    """Simulate location-selected beams for a single-antenna UE.

    The channel is single-path LOS and uses free-space loss.  All network gains
    use total transmit power 1: equal-power results allocate ``1 / num_rus`` to
    each RU, while ``network_mrt_gain`` may also optimize power allocation.

    ``predicted_csi_precoded_gain`` is the proposed method: select each DFT beam
    from geometry, predict its scalar effective CSI, and apply one additional
    common phase per RU before all RUs transmit the same symbol.  If explicit
    predictions are omitted, the simulator uses the true effective CSI plus
    each RU's configurable ``predicted_csi_phase_error_deg``.  The direct sum,
    perfect phase alignment, and global MRT remain comparison points.
    """
    if not rus:
        raise ValueError("at least one RU is required")

    per_ru: list[dict] = []
    selected_channels: list[complex] = []
    oracle_channels: list[complex] = []
    mrt_gains: list[float] = []
    configured_prediction_errors_deg: list[float] = []

    wave_number = 2.0 * math.pi / codebook.wavelength_m
    for ordinal, ru in enumerate(rus, start=1):
        local = ru_local_direction(ru, ue_position_m)
        true_steering = codebook.steering_vector(local["u"], local["v"])
        path_gain = free_space_power_gain(local["distance_m"], codebook.wavelength_m)
        # h^H w is the scalar channel.  The common propagation phase is kept so
        # that the no-inter-RU-phase-compensation result is also observable.
        channel = (
            math.sqrt(codebook.num_elements * path_gain)
            * np.exp(1j * wave_number * local["distance_m"])
            * true_steering
        )

        selected = codebook.nearest_direction(local["u"], local["v"])
        tx_phase_offset_deg = float(ru.get("tx_phase_offset_deg", 0.0))
        tx_phase_offset = np.exp(1j * math.radians(tx_phase_offset_deg))
        selected_scalar = complex(np.vdot(channel, codebook.weights(selected)) * tx_phase_offset)
        selected_gain = abs(selected_scalar) ** 2
        phase_correction_deg = -math.degrees(math.atan2(selected_scalar.imag, selected_scalar.real))
        phase_correction_deg = (phase_correction_deg + 180.0) % 360.0 - 180.0

        oracle, oracle_overlap = codebook.oracle_beam(true_steering)
        oracle_scalar = complex(np.vdot(channel, codebook.weights(oracle)) * tx_phase_offset)
        oracle_gain = abs(oracle_scalar) ** 2
        mrt_gain = float(np.vdot(channel, channel).real)

        selected_channels.append(selected_scalar)
        oracle_channels.append(oracle_scalar)
        mrt_gains.append(mrt_gain)
        configured_prediction_errors_deg.append(
            float(ru.get("predicted_csi_phase_error_deg", 0.0))
        )
        per_ru.append(
            {
                "id": ru.get("id", ordinal),
                **local,
                "tx_phase_offset_deg": tx_phase_offset_deg,
                "free_space_path_gain": path_gain,
                "free_space_path_gain_db": power_db(path_gain),
                "selected_beam": asdict(selected),
                "codebook_oracle_beam": asdict(oracle),
                "selected_matches_codebook_oracle": selected.index == oracle.index,
                "selected_channel_real": selected_scalar.real,
                "selected_channel_imag": selected_scalar.imag,
                "selected_channel_magnitude": abs(selected_scalar),
                "inter_ru_phase_correction_deg": phase_correction_deg,
                "selected_gain": selected_gain,
                "selected_gain_db": power_db(selected_gain),
                "codebook_oracle_gain": oracle_gain,
                "codebook_oracle_gain_db": power_db(oracle_gain),
                "mrt_gain": mrt_gain,
                "mrt_gain_db": power_db(mrt_gain),
                "selected_loss_vs_codebook_oracle_db": ratio_db(selected_gain, oracle_gain),
                "selected_loss_vs_mrt_db": ratio_db(selected_gain, mrt_gain),
                "selected_power_retained_vs_mrt": selected_gain / mrt_gain,
                "oracle_overlap": oracle_overlap,
            }
        )

    count = len(per_ru)
    selected_array = np.asarray(selected_channels, dtype=np.complex128)
    oracle_array = np.asarray(oracle_channels, dtype=np.complex128)
    selected_powers = np.abs(selected_array) ** 2
    oracle_powers = np.abs(oracle_array) ** 2
    mrt_array = np.asarray(mrt_gains, dtype=float)

    if predicted_effective_channels is None:
        configured_errors_rad = np.radians(np.asarray(configured_prediction_errors_deg))
        predicted_array = np.abs(selected_array) * np.exp(
            1j * (np.angle(selected_array) + configured_errors_rad)
        )
        prediction_source = "true_effective_csi_plus_configured_phase_error"
    else:
        predicted_array = np.asarray(predicted_effective_channels, dtype=np.complex128)
        if predicted_array.shape != selected_array.shape:
            raise ValueError(
                "predicted_effective_channels length must equal the number of RUs"
            )
        prediction_source = "provided_effective_csi"

    predicted_phase_precoders = phase_only_precoder(predicted_array)
    predicted_precoded_channels = selected_array * predicted_phase_precoders
    predicted_precoded_gain = float(
        abs(np.sum(predicted_precoded_channels) / math.sqrt(count)) ** 2
    )
    predicted_phase_errors_deg = np.degrees(
        np.angle(predicted_array * np.conj(selected_array))
    )
    predicted_phase_errors_deg[np.abs(predicted_phase_errors_deg) < 1e-12] = 0.0
    for row, predicted, phase_error, precoder in zip(
        per_ru, predicted_array, predicted_phase_errors_deg, predicted_phase_precoders
    ):
        row.update({
            "predicted_effective_csi_real": float(predicted.real),
            "predicted_effective_csi_imag": float(predicted.imag),
            "predicted_effective_csi_magnitude": float(abs(predicted)),
            "predicted_effective_csi_phase_deg": math.degrees(
                math.atan2(predicted.imag, predicted.real)
            ),
            "predicted_csi_phase_error_deg": float(phase_error),
            "predicted_phase_precoder_real": float(precoder.real),
            "predicted_phase_precoder_imag": float(precoder.imag),
            "predicted_phase_precoder_deg": math.degrees(
                math.atan2(precoder.imag, precoder.real)
            ),
        })

    selected_raw = float(abs(np.sum(selected_array) / math.sqrt(count)) ** 2)
    selected_noncoherent = float(np.sum(selected_powers) / count)
    selected_coherent = float(np.sum(np.abs(selected_array)) ** 2 / count)
    oracle_coherent = float(np.sum(np.abs(oracle_array)) ** 2 / count)
    continuous_equal_power = float(np.sum(np.sqrt(mrt_array)) ** 2 / count)
    network_mrt = float(np.sum(mrt_array))
    best_single_selected = float(np.max(selected_powers))

    network = {
        "ue_antennas": 1,
        "receive_combining": "none_over_the_air_complex_superposition",
        "transmit_precoding": "position_dft_plus_predicted_effective_csi_phase",
        "effective_csi_prediction_source": prediction_source,
        "phase_reference_ru": per_ru[0]["id"],
        "total_tx_power_normalized": 1.0,
        "power_per_ru_for_equal_power_results": 1.0 / count,
        "direct_superposition_gain": selected_raw,
        "direct_superposition_gain_db": power_db(selected_raw),
        "direct_superposition_channel_magnitude": math.sqrt(selected_raw),
        "selected_raw_gain": selected_raw,
        "selected_raw_gain_db": power_db(selected_raw),
        "best_single_ru_gain": best_single_selected,
        "best_single_ru_gain_db": power_db(best_single_selected),
        "selected_noncoherent_gain": selected_noncoherent,
        "selected_noncoherent_gain_db": power_db(selected_noncoherent),
        "selected_coherent_equal_power_gain": selected_coherent,
        "selected_coherent_equal_power_gain_db": power_db(selected_coherent),
        "selected_coherent_channel_magnitude": math.sqrt(selected_coherent),
        "predicted_csi_precoded_gain": predicted_precoded_gain,
        "predicted_csi_precoded_gain_db": power_db(predicted_precoded_gain),
        "predicted_csi_precoded_channel_magnitude": math.sqrt(predicted_precoded_gain),
        "codebook_oracle_coherent_equal_power_gain": oracle_coherent,
        "codebook_oracle_coherent_equal_power_gain_db": power_db(oracle_coherent),
        "continuous_beam_coherent_equal_power_gain": continuous_equal_power,
        "continuous_beam_coherent_equal_power_gain_db": power_db(continuous_equal_power),
        "network_mrt_gain": network_mrt,
        "network_mrt_gain_db": power_db(network_mrt),
        "network_mrt_channel_magnitude": math.sqrt(network_mrt),
        "coherent_combining_gain_vs_noncoherent_db": ratio_db(selected_coherent, selected_noncoherent),
        "coherent_combining_gain_vs_best_single_ru_db": ratio_db(selected_coherent, best_single_selected),
        "inter_ru_phase_alignment_gain_db": ratio_db(selected_coherent, selected_raw),
        "predicted_csi_precoding_gain_vs_direct_db": ratio_db(
            predicted_precoded_gain, selected_raw
        ),
        "predicted_csi_precoding_loss_vs_perfect_phase_db": ratio_db(
            predicted_precoded_gain, selected_coherent
        ),
        "predicted_csi_precoding_loss_vs_network_mrt_db": ratio_db(
            predicted_precoded_gain, network_mrt
        ),
        "predicted_csi_precoding_power_retained_vs_network_mrt": (
            predicted_precoded_gain / network_mrt
        ),
        "direct_loss_vs_best_single_ru_db": ratio_db(selected_raw, best_single_selected),
        "direct_loss_vs_phase_aligned_dft_db": ratio_db(selected_raw, selected_coherent),
        "direct_loss_vs_network_mrt_db": ratio_db(selected_raw, network_mrt),
        "direct_power_retained_vs_network_mrt": selected_raw / network_mrt,
        "phase_aligned_position_dft_loss_vs_codebook_oracle_db": ratio_db(
            selected_coherent, oracle_coherent
        ),
        "phase_aligned_dft_loss_vs_continuous_equal_power_db": ratio_db(
            selected_coherent, continuous_equal_power
        ),
        "phase_aligned_dft_loss_vs_network_mrt_db": ratio_db(selected_coherent, network_mrt),
        "phase_aligned_dft_power_retained_vs_network_mrt": selected_coherent / network_mrt,
    }
    return {
        "model": {
            "channel": "far_field_single_path_los_friis",
            "frequency_hz": codebook.rf_hz,
            "wavelength_m": codebook.wavelength_m,
            "upa_rows": codebook.rows,
            "upa_columns": codebook.columns,
            "upa_elements": codebook.num_elements,
            "ue_antennas": 1,
            "codebook_oversampling": codebook.oversampling,
            "codebook_beams": len(codebook),
        },
        "ue_position_m": [float(value) for value in ue_position_m],
        "per_ru": per_ru,
        "network": network,
    }
