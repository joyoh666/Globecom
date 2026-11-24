#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate position-selected 4x4 UPA DFT beams for three LOS RUs."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path

from beam.upa_dft_sim import UPADFTCodebook, simulate_position


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "position_beamforming_test.json"
DEFAULT_PLOT = ROOT / "plots" / "position_beamforming_result.svg"


def build_codebook(cfg: dict) -> UPADFTCodebook:
    upa = cfg["upa"]
    dft = cfg["codebook"]
    return UPADFTCodebook(
        rows=int(upa["rows"]),
        columns=int(upa["columns"]),
        oversampling=int(dft.get("oversampling", 1)),
        rf_hz=float(cfg["frequency_hz"]),
        element_spacing_m=float(upa["element_spacing_m"]),
        max_abs_azimuth_deg=float(dft.get("max_abs_azimuth_deg", 60.0)),
        max_abs_elevation_deg=float(dft.get("max_abs_elevation_deg", 60.0)),
    )


def print_report(result: dict) -> None:
    model = result["model"]
    print(
        f"LOS {model['frequency_hz'] / 1e9:.3f} GHz | "
        f"{model['upa_rows']}x{model['upa_columns']} UPA | "
        f"DFT oversampling={model['codebook_oversampling']} "
        f"({model['codebook_beams']} visible beams)"
    )
    print(f"UE position [m]: {result['ue_position_m']}")
    print()
    header = (
        "RU", "dist[m]", "true az/el[deg]", "selected (ku,kv)",
        "beam az/el[deg]", "|h_eff|", "CSI precoder", "gain[dB]", "vs MRT[dB]", "oracle?",
    )
    print(
        f"{header[0]:>3} {header[1]:>8} {header[2]:>19} {header[3]:>18} "
        f"{header[4]:>19} {header[5]:>11} {header[6]:>11} {header[7]:>10} "
        f"{header[8]:>11} {header[9]:>8}"
    )
    for row in result["per_ru"]:
        beam = row["selected_beam"]
        print(
            f"{str(row['id']):>3} {row['distance_m']:8.3f} "
            f"{row['azimuth_deg']:8.2f}/{row['elevation_deg']:7.2f} "
            f"({beam['horizontal_bin']:3d},{beam['vertical_bin']:3d}) "
            f"{beam['azimuth_deg']:8.2f}/{beam['elevation_deg']:7.2f} "
            f"{row['selected_channel_magnitude']:11.4e} "
            f"{row['predicted_phase_precoder_deg']:10.2f}° "
            f"{row['selected_gain_db']:10.2f} "
            f"{row['selected_loss_vs_mrt_db']:11.2f} "
            f"{str(row['selected_matches_codebook_oracle']):>8}"
        )

    net = result["network"]
    print()
    print("Single-antenna UE, fixed total TX power = 1 (equal RU power = 1/3)")
    print(f"  position DFT: direct complex sum       : {net['direct_superposition_gain_db']:8.2f} dB")
    print(f"  + predicted-CSI coherent precoding     : {net['predicted_csi_precoded_gain_db']:8.2f} dB")
    print(f"  best single RU, full total power       : {net['best_single_ru_gain_db']:8.2f} dB")
    print(f"  random-phase long-term average [ref]   : {net['selected_noncoherent_gain_db']:8.2f} dB")
    print(f"  phase-aligned DFT [upper bound]        : {net['selected_coherent_equal_power_gain_db']:8.2f} dB")
    print(f"  global full-CSI MRT [upper bound]      : {net['network_mrt_gain_db']:8.2f} dB")
    print()
    print(f"Predicted-CSI precoding vs direct sum    : {net['predicted_csi_precoding_gain_vs_direct_db']:+.2f} dB")
    print(f"Prediction loss vs perfect phase bound   : {net['predicted_csi_precoding_loss_vs_perfect_phase_db']:+.2f} dB")
    print(f"Predicted-CSI precoding vs global MRT    : {net['predicted_csi_precoding_loss_vs_network_mrt_db']:+.2f} dB")
    print(
        "Predicted-CSI power retained vs MRT      : "
        f"{100.0 * net['predicted_csi_precoding_power_retained_vs_network_mrt']:.2f}%"
    )


def write_csv(path: Path, result: dict) -> None:
    fields = [
        "id", "distance_m", "u", "v", "azimuth_deg", "elevation_deg",
        "selected_horizontal_bin", "selected_vertical_bin",
        "selected_beam_azimuth_deg", "selected_beam_elevation_deg",
        "tx_phase_offset_deg",
        "selected_channel_real", "selected_channel_imag", "selected_channel_magnitude",
        "inter_ru_phase_correction_deg",
        "predicted_effective_csi_real", "predicted_effective_csi_imag",
        "predicted_effective_csi_magnitude", "predicted_effective_csi_phase_deg",
        "predicted_csi_phase_error_deg",
        "predicted_phase_precoder_real", "predicted_phase_precoder_imag",
        "predicted_phase_precoder_deg",
        "selected_gain_db", "codebook_oracle_gain_db", "mrt_gain_db",
        "selected_loss_vs_codebook_oracle_db", "selected_loss_vs_mrt_db",
        "selected_matches_codebook_oracle",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for item in result["per_ru"]:
            selected = item["selected_beam"]
            writer.writerow({
                "id": item["id"],
                "distance_m": item["distance_m"],
                "u": item["u"],
                "v": item["v"],
                "azimuth_deg": item["azimuth_deg"],
                "elevation_deg": item["elevation_deg"],
                "selected_horizontal_bin": selected["horizontal_bin"],
                "selected_vertical_bin": selected["vertical_bin"],
                "selected_beam_azimuth_deg": selected["azimuth_deg"],
                "selected_beam_elevation_deg": selected["elevation_deg"],
                "tx_phase_offset_deg": item["tx_phase_offset_deg"],
                "selected_channel_real": item["selected_channel_real"],
                "selected_channel_imag": item["selected_channel_imag"],
                "selected_channel_magnitude": item["selected_channel_magnitude"],
                "inter_ru_phase_correction_deg": item["inter_ru_phase_correction_deg"],
                "predicted_effective_csi_real": item["predicted_effective_csi_real"],
                "predicted_effective_csi_imag": item["predicted_effective_csi_imag"],
                "predicted_effective_csi_magnitude": item["predicted_effective_csi_magnitude"],
                "predicted_effective_csi_phase_deg": item["predicted_effective_csi_phase_deg"],
                "predicted_csi_phase_error_deg": item["predicted_csi_phase_error_deg"],
                "predicted_phase_precoder_real": item["predicted_phase_precoder_real"],
                "predicted_phase_precoder_imag": item["predicted_phase_precoder_imag"],
                "predicted_phase_precoder_deg": item["predicted_phase_precoder_deg"],
                "selected_gain_db": item["selected_gain_db"],
                "codebook_oracle_gain_db": item["codebook_oracle_gain_db"],
                "mrt_gain_db": item["mrt_gain_db"],
                "selected_loss_vs_codebook_oracle_db": item["selected_loss_vs_codebook_oracle_db"],
                "selected_loss_vs_mrt_db": item["selected_loss_vs_mrt_db"],
                "selected_matches_codebook_oracle": item["selected_matches_codebook_oracle"],
            })


def _bar_panel(
    parts: list[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    categories: list[str],
    series: list[tuple[str, str, list[float]]],
) -> None:
    """Append one grouped dB bar panel to a standalone SVG."""
    values = [value for _, _, rows in series for value in rows]
    lower = 2.0 * math.floor((min(values) - 1.0) / 2.0)
    upper = 2.0 * math.ceil((max(values) + 1.0) / 2.0)
    if upper <= lower:
        upper = lower + 2.0

    plot_left, plot_right = x + 70.0, x + width - 18.0
    plot_top, plot_bottom = y + 58.0, y + height - 78.0
    plot_width, plot_height = plot_right - plot_left, plot_bottom - plot_top

    def y_of(value: float) -> float:
        return plot_top + (upper - value) * plot_height / (upper - lower)

    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" '
        'fill="#ffffff" stroke="#d8dee9"/>'
    )
    parts.append(
        f'<text x="{x + 20}" y="{y + 31}" class="panel-title">{escape(title)}</text>'
    )
    for tick_no in range(6):
        value = lower + (upper - lower) * tick_no / 5.0
        tick_y = y_of(value)
        parts.append(
            f'<line x1="{plot_left}" y1="{tick_y:.2f}" x2="{plot_right}" y2="{tick_y:.2f}" '
            'stroke="#e5e9f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{plot_left - 10}" y="{tick_y + 4:.2f}" text-anchor="end" '
            f'class="axis">{value:.1f}</text>'
        )
    parts.append(
        f'<text x="{x + 17}" y="{plot_top + plot_height / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 {x + 17} {plot_top + plot_height / 2:.2f})" class="axis-title">Gain [dB]</text>'
    )

    group_width = plot_width / len(categories)
    available = group_width * 0.72
    bar_width = min(46.0, available / len(series))
    for category_index, category in enumerate(categories):
        center = plot_left + group_width * (category_index + 0.5)
        start = center - bar_width * len(series) / 2.0
        for series_index, (_, color, rows) in enumerate(series):
            value = rows[category_index]
            bar_x = start + series_index * bar_width
            bar_y = y_of(value)
            bar_height = max(0.0, plot_bottom - bar_y)
            parts.append(
                f'<rect x="{bar_x + 2:.2f}" y="{bar_y:.2f}" width="{bar_width - 4:.2f}" '
                f'height="{bar_height:.2f}" fill="{color}"/>'
            )
            # Position DFT and codebook oracle are normally identical in this
            # LOS test.  Avoid printing the same value twice on adjacent bars.
            if len(series) == 1 or series[series_index][0] != "Codebook oracle":
                parts.append(
                    f'<text x="{bar_x + bar_width / 2:.2f}" y="{bar_y - 7.0:.2f}" '
                    f'text-anchor="middle" class="value">{value:.2f}</text>'
                )
        parts.append(
            f'<text x="{center:.2f}" y="{plot_bottom + 25:.2f}" text-anchor="middle" '
            f'class="axis-label">{escape(category)}</text>'
        )

    legend_width = sum(25.0 + len(label) * 7.0 for label, _, _ in series)
    legend_x = x + (width - legend_width) / 2.0
    legend_y = y + height - 22.0
    for label, color, _ in series:
        parts.append(
            f'<rect x="{legend_x:.2f}" y="{legend_y - 10:.2f}" width="12" height="12" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 18:.2f}" y="{legend_y:.2f}" class="legend">{escape(label)}</text>'
        )
        legend_x += 25.0 + len(label) * 7.0


def write_svg(path: Path, result: dict) -> None:
    """Write a dependency-free SVG summary of per-RU and network gains."""
    per_ru = result["per_ru"]
    network = result["network"]
    width, height = 1200, 700
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<title>Position DFT plus predicted-CSI coherent precoding</title>',
        '<desc>Per-RU beam gains and predicted effective-CSI phase precoding for a single-antenna UE, compared with direct transmission and upper bounds.</desc>',
        '<style>',
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #20242b; }',
        '.title { font-size: 24px; font-weight: 600; }',
        '.subtitle { font-size: 14px; fill: #596273; }',
        '.panel-title { font-size: 17px; font-weight: 600; }',
        '.axis, .legend { font-size: 12px; fill: #596273; }',
        '.axis-title, .axis-label { font-size: 13px; }',
        '.value { font-size: 12px; font-weight: 600; }',
        '.metric { font-size: 15px; }',
        '</style>',
        '<rect width="1200" height="700" fill="#f7f9fc"/>',
        '<text x="54" y="42" class="title">Position DFT + predicted-CSI coherent precoding</text>',
        (
            f'<text x="54" y="67" class="subtitle">7.4 GHz · 3 RU · 4×4 UPA · UE '
            f'{escape(str(result["ue_position_m"]))} m · prediction phase error '
            f'{escape(str([round(item["predicted_csi_phase_error_deg"], 2) for item in per_ru]))}°</text>'
        ),
    ]
    _bar_panel(
        parts,
        x=50,
        y=90,
        width=545,
        height=490,
        title="Per-RU effective channel gain",
        categories=[f"RU {item['id']}" for item in per_ru],
        series=[
            ("Position DFT", "#4c78a8", [item["selected_gain_db"] for item in per_ru]),
            ("Codebook oracle", "#f28e2b", [item["codebook_oracle_gain_db"] for item in per_ru]),
            ("Full-CSI MRT", "#59a14f", [item["mrt_gain_db"] for item in per_ru]),
        ],
    )
    _bar_panel(
        parts,
        x=620,
        y=90,
        width=530,
        height=490,
        title="Single-antenna UE: over-the-air coherent sum",
        categories=["Best RU", "Direct DFT", "Pred.-CSI", "Perfect phase", "Global MRT"],
        series=[
            (
                "Channel gain",
                "#4c78a8",
                [
                    network["best_single_ru_gain_db"],
                    network["direct_superposition_gain_db"],
                    network["predicted_csi_precoded_gain_db"],
                    network["selected_coherent_equal_power_gain_db"],
                    network["network_mrt_gain_db"],
                ],
            )
        ],
    )
    parts.extend([
        (
            f'<text x="60" y="626" class="metric">Predicted CSI vs direct: '
            f'<tspan font-weight="600">{network["predicted_csi_precoding_gain_vs_direct_db"]:+.2f} dB</tspan></text>'
        ),
        (
            f'<text x="410" y="626" class="metric">Prediction loss vs perfect phase: '
            f'<tspan font-weight="600">{network["predicted_csi_precoding_loss_vs_perfect_phase_db"]:+.2f} dB</tspan></text>'
        ),
        (
            f'<text x="800" y="626" class="metric">Predicted-CSI power vs MRT: '
            f'<tspan font-weight="600">{100.0 * network["predicted_csi_precoding_power_retained_vs_network_mrt"]:.2f}%</tspan></text>'
        ),
        '<text x="60" y="663" class="subtitle">One predicted common phase is applied per RU; the UE still has one antenna and performs no receive combining.</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--ue", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="override UE position in metres",
    )
    parser.add_argument("--json", type=Path, help="write complete result JSON")
    parser.add_argument("--csv", type=Path, help="write per-RU result CSV")
    parser.add_argument(
        "--phase-errors-deg", nargs="+", type=float,
        help="override effective-CSI phase prediction error for each RU, in degrees",
    )
    parser.add_argument(
        "--plot", nargs="?", type=Path, const=DEFAULT_PLOT,
        help=f"write an SVG plot (default path when omitted: {DEFAULT_PLOT})",
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    ue_position = args.ue if args.ue is not None else cfg["ue_position_m"]
    rus = [dict(ru) for ru in cfg["rus"]]
    if args.phase_errors_deg is not None:
        if len(args.phase_errors_deg) != len(rus):
            parser.error(
                f"--phase-errors-deg requires exactly {len(rus)} values "
                f"(received {len(args.phase_errors_deg)})"
            )
        for ru, error_deg in zip(rus, args.phase_errors_deg):
            ru["predicted_csi_phase_error_deg"] = error_deg
    result = simulate_position(rus, ue_position, build_codebook(cfg))
    print_report(result)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON: {args.json}")
    if args.csv:
        write_csv(args.csv, result)
        print(f"CSV : {args.csv}")
    if args.plot:
        write_svg(args.plot, result)
        print(f"Plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
