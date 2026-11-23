#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Azimuth DFT codebook utilities for the FR3 4x4 TX panel.

The physical antenna panel is 4x4, but each BBox Lite FR3 exposes four
independently controlled RF channels.  The prototype uses two BBoxes per RU as
two coherent 1x4 horizontal active subarrays on that panel.  Consequently the
controllable DFT dimension is the four-column azimuth axis; the inter-BBox
splitter/path offset is handled separately by ``dphi_deg`` calibration.

For an ``M`` element controllable axis and an oversampled DFT size
``B=M*oversampling``, DFT bin ``k`` has progressive phase ``2*pi*k/B``.  Its
physical steering angle is

    sin(theta_k) = k / (B * d_lambda)

where ``d_lambda`` is the element spacing in wavelengths.  Only bins inside
the configured steering sector are retained.  This is preferable to sampling
angles uniformly: every entry remains an actual (possibly oversampled) DFT
beam and maps one-to-one to a class learned by the GRU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


C_M_PER_S = 3.0e8


@dataclass(frozen=True)
class DFTBeam:
    """One usable DFT beam.

    ``class_index`` is the contiguous classifier label, while ``dft_bin`` is
    the signed bin in the full oversampled DFT dictionary.
    """

    class_index: int
    dft_bin: int
    angle_deg: float
    phase_step_deg: float
    spatial_sine: float


class DFTCodebook:
    def __init__(
        self,
        num_elements: int,
        oversampling: int,
        rf_hz: float,
        element_spacing_m: float,
        max_abs_angle_deg: float,
        array_geometry: dict | None = None,
    ):
        if num_elements < 2:
            raise ValueError("num_elements must be at least 2")
        if oversampling < 1:
            raise ValueError("oversampling must be at least 1")
        if rf_hz <= 0 or element_spacing_m <= 0:
            raise ValueError("rf_hz and element_spacing_m must be positive")
        if not 0.0 < max_abs_angle_deg < 90.0:
            raise ValueError("max_abs_angle_deg must be in (0, 90)")

        self.num_elements = int(num_elements)
        self.oversampling = int(oversampling)
        self.rf_hz = float(rf_hz)
        self.element_spacing_m = float(element_spacing_m)
        self.max_abs_angle_deg = float(max_abs_angle_deg)
        self.array_geometry = dict(array_geometry or {})
        self.dft_size = self.num_elements * self.oversampling
        self.spacing_lambda = self.element_spacing_m / (C_M_PER_S / self.rf_hz)

        # Use the FFT-shifted integer bins.  For an even DFT size the positive
        # Nyquist bin does not exist, matching numpy.fft.fftshift ordering.
        bins = range(-self.dft_size // 2, (self.dft_size + 1) // 2)
        entries: list[DFTBeam] = []
        for k in bins:
            spatial_sine = k / (self.dft_size * self.spacing_lambda)
            if abs(spatial_sine) > 1.0:
                continue
            angle = math.degrees(math.asin(spatial_sine))
            if abs(angle) > self.max_abs_angle_deg + 1e-9:
                continue
            entries.append(
                DFTBeam(
                    class_index=len(entries),
                    dft_bin=int(k),
                    angle_deg=float(angle),
                    phase_step_deg=float(360.0 * k / self.dft_size),
                    spatial_sine=float(spatial_sine),
                )
            )
        if not entries or not any(b.dft_bin == 0 for b in entries):
            raise ValueError("configured sector produced no broadside DFT beam")
        self.beams = tuple(entries)

    @classmethod
    def from_config(cls, cfg: dict) -> "DFTCodebook":
        dcfg = cfg["dft_beam"]
        topology = cfg.get("radio_topology", {})
        scan_axis = str(dcfg.get("scan_axis", "azimuth")).lower()
        if scan_axis != "azimuth":
            raise ValueError(
                "this two-BBox prototype supports azimuth DFT only; "
                "independent 2-D DFT requires all 16 TX elements to be phase-controlled"
            )
        physical = topology.get("tx_physical_shape", [4, 4])
        active = topology.get("tx_active_shape", [2, 4])
        axis_elements = int(dcfg.get("axis_elements", active[1]))
        return cls(
            num_elements=axis_elements,
            oversampling=int(dcfg.get("oversampling", 4)),
            rf_hz=float(cfg["freq"]["rf_ghz"]) * 1e9,
            element_spacing_m=float(dcfg.get("element_spacing_mm", 18.5)) * 1e-3,
            max_abs_angle_deg=float(dcfg.get("max_abs_angle_deg", cfg["beam"]["max_deg"])),
            array_geometry={
                "tx_physical_shape": [int(physical[0]), int(physical[1])],
                "tx_active_shape": [int(active[0]), int(active[1])],
                "bbox_per_ru": int(topology.get("bbox_per_ru", 2)),
                "channels_per_bbox": int(topology.get("channels_per_bbox", 4)),
                "bbox_subarray": str(topology.get("bbox_subarray", "1x4_horizontal")),
                "scan_axis": scan_axis,
                "elevation_policy": str(dcfg.get("elevation_policy", "fixed_broadside")),
                "rx_antennas": int(topology.get("rx_antennas", 1)),
            },
        )

    def __len__(self) -> int:
        return len(self.beams)

    def __iter__(self) -> Iterable[DFTBeam]:
        return iter(self.beams)

    def __getitem__(self, class_index: int) -> DFTBeam:
        return self.beams[int(class_index)]

    @property
    def broadside_index(self) -> int:
        return next(b.class_index for b in self.beams if b.dft_bin == 0)

    @property
    def max_abs_bin(self) -> int:
        return max(1, max(abs(b.dft_bin) for b in self.beams))

    def nearest_angle(self, angle_deg: float) -> DFTBeam:
        return min(self.beams, key=lambda b: abs(b.angle_deg - float(angle_deg)))

    def normalized_bin(self, class_index: int) -> float:
        return self[class_index].dft_bin / float(self.max_abs_bin)

    def weights(self, class_index: int) -> list[complex]:
        """Unit-norm complex DFT steering weights for documentation/simulation."""
        b = self[class_index]
        scale = math.sqrt(self.num_elements)
        return [
            complex(
                math.cos(2.0 * math.pi * n * b.dft_bin / self.dft_size),
                math.sin(2.0 * math.pi * n * b.dft_bin / self.dft_size),
            )
            / scale
            for n in range(self.num_elements)
        ]

    def to_dict(self) -> dict:
        return {
            "type": "oversampled_dft_ula",
            "num_elements": self.num_elements,
            "oversampling": self.oversampling,
            "dft_size": self.dft_size,
            "rf_hz": self.rf_hz,
            "element_spacing_m": self.element_spacing_m,
            "spacing_lambda": self.spacing_lambda,
            "max_abs_angle_deg": self.max_abs_angle_deg,
            "array_geometry": self.array_geometry,
            "beams": [asdict(b) for b in self.beams],
        }

    def assert_compatible(self, other: dict, angle_tol_deg: float = 1e-3) -> None:
        saved = other.get("beams", [])
        if len(saved) != len(self.beams):
            raise ValueError(f"codebook size mismatch: model={len(saved)} runtime={len(self.beams)}")
        for now, old in zip(self.beams, saved):
            if int(old["dft_bin"]) != now.dft_bin or abs(float(old["angle_deg"]) - now.angle_deg) > angle_tol_deg:
                raise ValueError("model and runtime DFT codebooks do not match")
