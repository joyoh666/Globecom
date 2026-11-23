#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DFT-beam measurement, sweep labeling, and GRU beam inference.

This process intentionally sits beside ``DMIMO_3TX_Live_Rx.py``.  The latter
continues to generate three FDM pilots and estimate H0/H1/H2 every 5 ms.  This
process receives that CSI, receives TurtleBot telemetry, controls the six TX
BBoxes with an oversampled DFT codebook, and stores one synchronized table.

Normal collection and beam selection (no sweep)::

    python -m runtime.dft_beam_measurement --mode geometry --dry-run

``sweep`` mode remains available for calibration and measured-oracle checks.

Online prediction after training::

    python -m runtime.dft_beam_measurement --mode predict \
        --model models/dft_beam_gru.pt

CSI input is the existing ``<dii6f`` UDP packet.  Configure the RF process with
``LIVE_UDP=127.0.0.1:5010``.  Pose input accepts both the new
``turtlebot_telemetry.py`` schema and the legacy ``hy_robot_proc.py`` JSON.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import signal
import socket
import struct
import threading
import time
from typing import Any

import numpy as np

from beam import geometry as geo
from beam.dft_codebook import DFTCodebook
from tmytek.tmy_ctl import TMY, addrs_of, cfg_gain, chmap_of


CSI_PACKET = struct.Struct("<dii6f")
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


def quality_db(h: complex) -> float:
    return float(20.0 * math.log10(max(abs(h), 1e-12)))


def rotate_xy(x: float, y: float, yaw_deg: float) -> tuple[float, float]:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return c * x + s * y, -s * x + c * y


class Inputs:
    """Background UDP receivers with a loss-bounded CSI queue."""

    def __init__(self, csi_addr: tuple[str, int], pose_addr: tuple[str, int], max_queue: int = 8192):
        self.stop = threading.Event()
        self.csi: queue.Queue[tuple[float, float, int, int, np.ndarray]] = queue.Queue(maxsize=max_queue)
        self.pose_lock = threading.Lock()
        self.poses: deque[dict[str, Any]] = deque(maxlen=512)
        self.fallback_start: tuple[float, float, float] | None = None
        self.csi_packets = 0
        self.csi_dropped = 0
        self.pose_packets = 0
        # Bind before starting threads so a port conflict/permission error is
        # reported to the caller instead of leaving a silently dead receiver.
        csi_sock = self._bound(csi_addr)
        try:
            pose_sock = self._bound(pose_addr)
        except Exception:
            csi_sock.close()
            raise
        self.threads = [
            threading.Thread(target=self._csi_loop, args=(csi_sock,), daemon=True),
            threading.Thread(target=self._pose_loop, args=(pose_sock,), daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    @staticmethod
    def _bound(addr):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(addr)
        sock.settimeout(0.2)
        return sock

    def _csi_loop(self, sock):
        try:
            while not self.stop.is_set():
                try:
                    data, _ = sock.recvfrom(256)
                except socket.timeout:
                    continue
                if len(data) != CSI_PACKET.size:
                    continue
                recv_time = time.time()
                t_usrp, frame, half, *v = CSI_PACKET.unpack(data)
                h = np.array(
                    [v[0] + 1j * v[1], v[2] + 1j * v[3], v[4] + 1j * v[5]],
                    dtype=np.complex64,
                )
                item = (recv_time, float(t_usrp), int(frame), int(half), h)
                try:
                    self.csi.put_nowait(item)
                except queue.Full:
                    self.csi_dropped += 1
                    try:
                        self.csi.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.csi.put_nowait(item)
                    except queue.Full:
                        pass
                self.csi_packets += 1
        finally:
            sock.close()

    def _pose_loop(self, sock):
        try:
            while not self.stop.is_set():
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                recv_time = time.time()
                try:
                    pose = json.loads(data.decode("utf-8"))
                    x, y = float(pose["x"]), float(pose["y"])
                    yaw = float(pose["yaw_deg"])
                except (ValueError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if self.fallback_start is None:
                    self.fallback_start = (x, y, yaw)
                pose["_host_rx_time"] = recv_time
                with self.pose_lock:
                    self.poses.append(pose)
                self.pose_packets += 1
        finally:
            sock.close()

    def latest_pose(self, at_time: float) -> dict[str, Any] | None:
        with self.pose_lock:
            return next(
                (pose for pose in reversed(self.poses) if pose["_host_rx_time"] <= at_time),
                None,
            )

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=0.5)


def pose_in_start_frame(pose: dict[str, Any] | None, fallback_start, now: float, max_age_s: float) -> dict[str, float] | None:
    if pose is None or not 0.0 <= now - float(pose.get("_host_rx_time", 0.0)) <= max_age_s:
        return None
    x, y, yaw = float(pose["x"]), float(pose["y"]), float(pose["yaw_deg"])
    if all(k in pose for k in ("local_x", "local_y", "local_yaw_deg")):
        px, py = float(pose["local_x"]), float(pose["local_y"])
        yaw_s = float(pose["local_yaw_deg"])
        vx = float(pose.get("local_vx", pose.get("vx", 0.0)))
        vy = float(pose.get("local_vy", pose.get("vy", 0.0)))
    else:
        sx = float(pose.get("sx", fallback_start[0] if fallback_start else x))
        sy = float(pose.get("sy", fallback_start[1] if fallback_start else y))
        syaw = float(pose.get("syaw", fallback_start[2] if fallback_start else yaw))
        px, py = geo.odom_to_start(x, y, sx, sy, syaw)
        yaw_s = geo.wrap_deg(yaw - syaw)
        vx, vy = rotate_xy(float(pose.get("vx", 0.0)), float(pose.get("vy", 0.0)), syaw)
    yaw_rate = float(pose.get("yaw_rate_dps", pose.get("vz", 0.0)))
    if not all(math.isfinite(value) for value in (px, py, yaw_s, vx, vy, yaw_rate)):
        return None
    return {
        "x": px,
        "y": py,
        "yaw_deg": yaw_s,
        "vx": vx,
        "vy": vy,
        # Legacy hy_robot_proc called angular velocity "vz".
        "yaw_rate_dps": yaw_rate,
        "age_ms": (now - float(pose["_host_rx_time"])) * 1e3,
        "source_ts": float(pose.get("ts", float("nan"))),
    }


def ru_features(
    ru: dict,
    pose: dict[str, float],
    beam_class: int,
    q_db: float,
    codebook: DFTCodebook,
) -> tuple[dict[str, float], np.ndarray]:
    dx, dy = float(ru["x"]) - pose["x"], float(ru["y"]) - pose["y"]
    distance = max(math.hypot(dx, dy), 1e-9)
    ux, uy = dx / distance, dy / distance
    bearing = math.degrees(math.atan2(dy, dx))
    rel = geo.wrap_deg(bearing - pose["yaw_deg"])
    rel_rad = math.radians(rel)
    toward = pose["vx"] * ux + pose["vy"] * uy
    tangential = -pose["vx"] * uy + pose["vy"] * ux
    values = {
        "ue_direction_sin": math.sin(rel_rad),
        "ue_direction_cos": math.cos(rel_rad),
        "velocity_toward_mps": toward,
        "velocity_tangential_mps": tangential,
        "yaw_rate_dps": pose["yaw_rate_dps"],
        "distance_m": distance,
        "current_beam_norm": codebook.normalized_bin(beam_class),
        "current_quality_db": q_db,
        "ue_to_ru_deg": rel,
        "ru_bearing_deg": bearing,
    }
    return values, np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)


class BeamHardware:
    def __init__(self, cfg: dict, codebook: DFTCodebook, dry_run: bool):
        self.cfg, self.codebook = cfg, codebook
        self.rus = {int(r["ru"]): r for r in cfg["geometry"]["rus"]}
        self.boxes = cfg["tmytek"]["bbox"]
        topology = cfg.get("radio_topology", {})
        expected_boxes = int(topology.get("bbox_per_ru", 2))
        expected_channels = int(topology.get("channels_per_bbox", 4))
        if int(topology.get("rx_antennas", 1)) != 1:
            raise ValueError("this measurement path expects the UE to use one receive antenna")
        for ru in self.rus:
            ru_boxes = [box for box in self.boxes if int(box["ru"]) == ru]
            if len(ru_boxes) != expected_boxes:
                raise ValueError(f"RU{ru}: expected {expected_boxes} BBoxes, found {len(ru_boxes)}")
            for box in ru_boxes:
                if len(chmap_of(box)) != expected_channels:
                    raise ValueError(
                        f"RU{ru} {box['sn']}: expected {expected_channels} active channels, "
                        f"found {len(chmap_of(box))}"
                    )
        self.tmy = TMY(cfg["tmytek"].get("working_root", "."), dry_run=dry_run)
        self.tmy.open([b["sn"] for b in self.boxes], addrs_of(self.boxes))
        self.gains = {}
        for box in self.boxes:
            dr = self.tmy.setup_bbox(box["sn"], cfg["freq"]["rf_ghz"], mode="TX", box=box)
            self.gains[box["sn"]] = cfg_gain(cfg["tmytek"], dr, "tx")
        self.current = {ru: codebook.broadside_index for ru in self.rus}

    def apply(self, classes: dict[int, int]) -> float:
        started = time.perf_counter()
        for ru, class_index in classes.items():
            beam = self.codebook[class_index]
            theta, phi = geo.to_theta_phi(beam.angle_deg, self.rus[ru].get("phi_sign", 1))
            for box in self.boxes:
                if int(box["ru"]) == ru:
                    self.tmy.set_beam(
                        box["sn"], self.gains[box["sn"]], theta, phi, box.get("dphi_deg", 0.0)
                    )
            self.current[ru] = int(class_index)
        return (time.perf_counter() - started) * 1e3

    def close(self):
        self.tmy.close()


def geometry_beam_targets(
    rus: dict[int, dict], pose: dict[str, float], codebook: DFTCodebook, lead_s: float
) -> tuple[dict[int, int], dict[int, float]]:
    """Quantize the LOS steering angle at an extrapolated future UE position."""
    px = pose["x"] + pose["vx"] * float(lead_s)
    py = pose["y"] + pose["vy"] * float(lead_s)
    classes, steering = {}, {}
    for ru, ru_cfg in rus.items():
        angle = geo.steer_deg(ru_cfg, px, py)
        steering[ru] = float(angle)
        classes[ru] = codebook.nearest_angle(angle).class_index
    return classes, steering


class GeometryController:
    """Position/motion based DFT selection without an online beam sweep."""

    def __init__(self, hw: BeamHardware, cfg: dict):
        self.hw, self.cb = hw, hw.codebook
        dcfg = cfg["dft_beam"]
        self.apply_lead_s = float(dcfg.get("geometry_apply_lead_s", 0.08))
        self.label_lead_s = float(dcfg.get("prediction_horizon_s", 0.25))
        self.period_s = float(dcfg.get("geometry_update_period_s", 0.1))
        self.settle_s = float(dcfg.get("geometry_settle_s", 0.05))
        self.last_apply_ms = 0.0
        self.ready_at = float("inf")
        self.next_update = 0.0

    def update(self, pose: dict[str, float]) -> bool:
        now = time.monotonic()
        if now < self.next_update:
            return False
        self.next_update = now + self.period_s
        applied, _ = geometry_beam_targets(
            self.hw.rus, pose, self.cb, self.apply_lead_s + pose["age_ms"] / 1e3
        )
        initial = math.isinf(self.ready_at)
        changes = {ru: beam for ru, beam in applied.items() if initial or beam != self.hw.current[ru]}
        if not changes:
            return False
        self.last_apply_ms = self.hw.apply(changes)
        self.ready_at = time.time() + self.settle_s
        return True


class SweepController:
    def __init__(self, hw: BeamHardware, cfg: dict, writer: csv.DictWriter, file_obj):
        self.hw, self.cb, self.writer, self.file_obj = hw, hw.codebook, writer, file_obj
        dcfg = cfg["dft_beam"]
        self.settle_s = float(dcfg["settle_s"])
        self.measure_s = float(dcfg["measure_s"])
        self.hold_s = float(dcfg["hold_s"])
        self.cycle = 0
        self.scan_pos = 0
        self.phase = "waiting"
        self.deadline = float("inf")
        self.buffers = {ru: [] for ru in hw.rus}
        self.scores = {ru: {} for ru in hw.rus}
        self.latest_best = {ru: None for ru in hw.rus}
        # Keep a safe broadside beam while USRP sync and robot pose come up.
        self.last_apply_ms = self.hw.apply({ru: self.cb.broadside_index for ru in hw.rus})

    def start(self):
        """Start only after a synchronized CSI sample and a valid pose exist."""
        if self.phase != "waiting":
            return
        self.scan_pos = 0
        self.last_apply_ms = self.hw.apply({ru: 0 for ru in self.hw.rus})
        self.phase = "settle"
        self.deadline = time.monotonic() + self.settle_s

    def observe(self, qualities: list[float]):
        if self.phase == "measure":
            for ru, value in zip(sorted(self.hw.rus), qualities):
                if math.isfinite(value):
                    self.buffers[ru].append(value)

    def advance(self):
        if self.phase == "waiting":
            return
        now_mono = time.monotonic()
        if now_mono < self.deadline:
            return
        if self.phase == "settle":
            self.phase = "measure"
            self.buffers = {ru: [] for ru in self.hw.rus}
            self.deadline = now_mono + self.measure_s
            return
        if self.phase == "measure":
            class_index = self.scan_pos
            for ru in self.hw.rus:
                vals = self.buffers[ru]
                self.scores[ru][class_index] = float(np.median(vals)) if vals else float("-inf")
            self.scan_pos += 1
            if self.scan_pos < len(self.cb):
                self.last_apply_ms = self.hw.apply({ru: self.scan_pos for ru in self.hw.rus})
                self.phase = "settle"
                self.deadline = time.monotonic() + self.settle_s
                return

            best = {
                ru: max(self.scores[ru], key=self.scores[ru].get)
                for ru in self.hw.rus
            }
            valid_cycle = all(
                len(self.scores[ru]) == len(self.cb)
                and all(math.isfinite(self.scores[ru].get(i, float("nan"))) for i in range(len(self.cb)))
                for ru in self.hw.rus
            )
            if valid_cycle:
                self.latest_best = dict(best)
            completed = time.time()
            row: dict[str, Any] = {
                "cycle": self.cycle,
                "completed_host_time": completed,
                "valid": int(valid_cycle),
                "num_beams": len(self.cb),
                "settle_s": self.settle_s,
                "measure_s": self.measure_s,
            }
            for ru in sorted(self.hw.rus):
                row[f"ru{ru}_best_beam"] = best[ru]
                row[f"ru{ru}_best_dft_bin"] = self.cb[best[ru]].dft_bin
                row[f"ru{ru}_best_angle_deg"] = self.cb[best[ru]].angle_deg
                row[f"ru{ru}_best_quality_db"] = self.scores[ru][best[ru]]
                row[f"ru{ru}_scores_json"] = json.dumps(
                    [self.scores[ru].get(i, float("nan")) for i in range(len(self.cb))],
                    separators=(",", ":"),
                )
            self.writer.writerow(row)
            self.file_obj.flush()
            print(
                f"[SWEEP {self.cycle}{'' if valid_cycle else ' INVALID'}] "
                + " · ".join(
                    f"RU{ru}=b{best[ru]}({self.cb[best[ru]].angle_deg:+.1f}deg, "
                    f"{self.scores[ru][best[ru]]:+.1f}dB)"
                    for ru in sorted(self.hw.rus)
                )
            )
            hold = best if valid_cycle else {ru: self.cb.broadside_index for ru in self.hw.rus}
            self.last_apply_ms = self.hw.apply(hold)
            self.phase = "hold"
            self.deadline = time.monotonic() + self.hold_s
            return

        # hold -> next sweep
        self.cycle += 1
        self.scan_pos = 0
        self.scores = {ru: {} for ru in self.hw.rus}
        self.last_apply_ms = self.hw.apply({ru: 0 for ru in self.hw.rus})
        self.phase = "settle"
        self.deadline = time.monotonic() + self.settle_s


def address(text: str) -> tuple[str, int]:
    host, port = text.rsplit(":", 1)
    return host, int(port)


def measurement_header(rus) -> list[str]:
    fields = [
        "host_rx_time", "csi_usrp_time", "frame_idx", "half", "phase", "sweep_cycle",
        "pose_valid", "pose_source_time", "pose_age_ms", "ue_x_m", "ue_y_m", "ue_yaw_deg",
        "ue_vx_mps", "ue_vy_mps", "ue_yaw_rate_dps", "beam_apply_ms",
    ]
    for ru in sorted(rus):
        fields += [
            f"ru{ru}_h_re", f"ru{ru}_h_im", f"ru{ru}_quality_db",
            f"ru{ru}_beam", f"ru{ru}_dft_bin", f"ru{ru}_beam_angle_deg",
            f"ru{ru}_geometry_beam", f"ru{ru}_geometry_steer_deg",
            f"ru{ru}_oracle_beam", f"ru{ru}_predicted_beam", f"ru{ru}_prediction_confidence",
            f"ru{ru}_ue_to_ru_deg", f"ru{ru}_ru_bearing_deg",
        ] + [f"ru{ru}_{name}" for name in FEATURE_NAMES]
    return fields


def sweep_header(rus) -> list[str]:
    fields = ["cycle", "completed_host_time", "valid", "num_beams", "settle_s", "measure_s"]
    for ru in sorted(rus):
        fields += [
            f"ru{ru}_best_beam", f"ru{ru}_best_dft_bin", f"ru{ru}_best_angle_deg",
            f"ru{ru}_best_quality_db", f"ru{ru}_scores_json",
        ]
    return fields


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(geo.DEFAULT_CONFIG))
    ap.add_argument("--mode", choices=("geometry", "sweep", "predict", "fixed"), default="geometry")
    ap.add_argument("--model", help="GRU checkpoint; required in predict mode")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    ap.add_argument("--csi-udp", default="0.0.0.0:5010")
    ap.add_argument("--pose-udp", default="0.0.0.0:5011")
    ap.add_argument("--output", help="output directory (default logs/dft_TIMESTAMP)")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds after setup; 0 runs until signal")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fixed-beam", type=int, help="class index; default broadside")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    a = ap.parse_args()
    if a.mode == "predict" and not a.model:
        ap.error("--model is required in predict mode")

    cfg = geo.load_config(a.config)
    cb = DFTCodebook.from_config(cfg)
    rus = {int(r["ru"]): r for r in cfg["geometry"]["rus"]}
    out = Path(a.output) if a.output else geo.CODE_ROOT / "logs" / f"dft_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dft_codebook.json").write_text(json.dumps(cb.to_dict(), indent=2), encoding="utf-8")

    predictor = None
    histories = {ru: deque() for ru in rus}
    predicted = {ru: None for ru in rus}
    confidence = {ru: float("nan") for ru in rus}
    if a.mode == "predict":
        try:
            from ml.dft_gru import BeamPredictor
            predictor = BeamPredictor(a.model, device=a.device)
        except (ImportError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"cannot load GRU predictor: {exc}") from exc
        if tuple(predictor.feature_names) != FEATURE_NAMES:
            raise SystemExit(
                f"model feature schema {predictor.feature_names} does not match runtime {FEATURE_NAMES}"
            )
        predictor.assert_codebook(cb.to_dict())
        sequence_length = predictor.sequence_length
        histories = {ru: deque(maxlen=sequence_length) for ru in rus}
    else:
        sequence_length = int(cfg["dft_beam"].get("sequence_length", 40))

    print(
        f"[DFT] {len(cb)} beams "
        + ", ".join(f"b{b.class_index}:k{b.dft_bin}/{b.angle_deg:+.1f}deg" for b in cb)
    )
    ag = cb.array_geometry
    print(
        f"[ARRAY] TX physical {ag.get('tx_physical_shape', [4, 4])} · "
        f"active {ag.get('tx_active_shape', [2, 4])} via "
        f"{ag.get('bbox_per_ru', 2)}x BBox/RU · scan={ag.get('scan_axis', 'azimuth')} · "
        f"RX antennas={ag.get('rx_antennas', 1)}"
    )
    hw = BeamHardware(cfg, cb, a.dry_run)
    inputs = Inputs(address(a.csi_udp), address(a.pose_udp))
    stop = threading.Event()

    def request_stop(_signum=None, _frame=None):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    meas_file = open(out / "dft_measurements.csv", "w", newline="", encoding="utf-8")
    meas_writer = csv.DictWriter(meas_file, fieldnames=measurement_header(rus), extrasaction="ignore")
    meas_writer.writeheader()
    sweep_file = open(out / "dft_sweeps.csv", "w", newline="", encoding="utf-8")
    sweep_writer = csv.DictWriter(sweep_file, fieldnames=sweep_header(rus), extrasaction="ignore")
    sweep_writer.writeheader()

    sweep_controller = None
    geometry_controller = None
    last_apply_ms = 0.0
    phase = a.mode
    if a.mode == "sweep":
        sweep_controller = SweepController(hw, cfg, sweep_writer, sweep_file)
    elif a.mode == "geometry":
        geometry_controller = GeometryController(hw, cfg)
        last_apply_ms = geometry_controller.last_apply_ms
    else:
        initial = cb.broadside_index if a.fixed_beam is None else int(a.fixed_beam)
        if not 0 <= initial < len(cb):
            raise SystemExit(f"--fixed-beam must be 0..{len(cb)-1}")
        last_apply_ms = hw.apply({ru: initial for ru in rus})

    alpha = float(cfg["dft_beam"].get("quality_ema_alpha", 0.2))
    pose_max_age = float(cfg["dft_beam"].get("pose_max_age_s", 0.5))
    predict_period = float(cfg["dft_beam"].get("predict_period_s", 0.1))
    ema_q = {ru: None for ru in rus}
    start_mono = time.monotonic()
    next_predict = start_mono
    rows = 0
    geometry_skipped = 0
    try:
        while not stop.is_set():
            if a.duration > 0 and time.monotonic() - start_mono >= a.duration:
                break
            if sweep_controller is not None:
                sweep_controller.advance()
                phase = sweep_controller.phase
                last_apply_ms = sweep_controller.last_apply_ms
            if geometry_controller is not None:
                now = time.time()
                control_pose = pose_in_start_frame(
                    inputs.latest_pose(now), inputs.fallback_start, now, pose_max_age
                )
                if control_pose is not None and geometry_controller.update(control_pose):
                    last_apply_ms = geometry_controller.last_apply_ms
                    ema_q = dict.fromkeys(rus)
            try:
                host_time, usrp_time, frame, half, h = inputs.csi.get(timeout=0.05)
            except queue.Empty:
                continue

            pose = pose_in_start_frame(inputs.latest_pose(host_time), inputs.fallback_start, host_time, pose_max_age)
            geometry_labels, geometry_steering = {}, {}
            if geometry_controller is not None:
                if pose is None or host_time < geometry_controller.ready_at:
                    geometry_skipped += 1
                    continue
                geometry_labels, geometry_steering = geometry_beam_targets(
                    rus, pose, cb, geometry_controller.label_lead_s + pose["age_ms"] / 1e3
                )

            raw_q = [quality_db(complex(v)) for v in h]
            for ru, value in zip(sorted(rus), raw_q):
                ema_q[ru] = value if ema_q[ru] is None else alpha * value + (1.0 - alpha) * ema_q[ru]
            smooth_q = [float(ema_q[ru]) for ru in sorted(rus)]
            if sweep_controller is not None:
                sweep_controller.observe(raw_q)

            if sweep_controller is not None and sweep_controller.phase == "waiting" and pose is not None:
                sweep_controller.start()
                phase = sweep_controller.phase
                last_apply_ms = sweep_controller.last_apply_ms
            row: dict[str, Any] = {
                "host_rx_time": host_time,
                "csi_usrp_time": usrp_time,
                "frame_idx": frame,
                "half": half,
                "phase": phase,
                "sweep_cycle": sweep_controller.cycle if sweep_controller else -1,
                "pose_valid": int(pose is not None),
                "beam_apply_ms": last_apply_ms,
            }
            if pose is not None:
                row.update({
                    "pose_source_time": pose["source_ts"], "pose_age_ms": pose["age_ms"],
                    "ue_x_m": pose["x"], "ue_y_m": pose["y"], "ue_yaw_deg": pose["yaw_deg"],
                    "ue_vx_mps": pose["vx"], "ue_vy_mps": pose["vy"],
                    "ue_yaw_rate_dps": pose["yaw_rate_dps"],
                })

            batch_features = {}
            for idx, ru in enumerate(sorted(rus)):
                beam_class = hw.current[ru]
                beam = cb[beam_class]
                prefix = f"ru{ru}_"
                row.update({
                    prefix + "h_re": float(h[idx].real), prefix + "h_im": float(h[idx].imag),
                    prefix + "quality_db": smooth_q[idx], prefix + "beam": beam_class,
                    prefix + "dft_bin": beam.dft_bin, prefix + "beam_angle_deg": beam.angle_deg,
                    prefix + "geometry_beam": geometry_labels.get(ru, ""),
                    prefix + "geometry_steer_deg": geometry_steering.get(ru, ""),
                    prefix + "oracle_beam": sweep_controller.latest_best[ru] if sweep_controller else "",
                    prefix + "predicted_beam": predicted[ru] if predicted[ru] is not None else "",
                    prefix + "prediction_confidence": confidence[ru],
                })
                if pose is not None:
                    values, feature = ru_features(rus[ru], pose, beam_class, smooth_q[idx], cb)
                    row[prefix + "ue_to_ru_deg"] = values["ue_to_ru_deg"]
                    row[prefix + "ru_bearing_deg"] = values["ru_bearing_deg"]
                    for name in FEATURE_NAMES:
                        row[prefix + name] = values[name]
                    batch_features[ru] = feature
                    if predictor is not None:
                        histories[ru].append(feature)

            if predictor is not None and time.monotonic() >= next_predict:
                next_predict = time.monotonic() + predict_period
                if all(len(histories[ru]) >= sequence_length for ru in rus):
                    x = np.stack([np.stack(histories[ru]) for ru in sorted(rus)])
                    classes, conf = predictor.predict(x)
                    changes = {}
                    for ru, cls, prob in zip(sorted(rus), classes, conf):
                        predicted[ru], confidence[ru] = int(cls), float(prob)
                        if prob >= a.min_confidence and int(cls) != hw.current[ru]:
                            changes[ru] = int(cls)
                    if changes:
                        last_apply_ms = hw.apply(changes)

            meas_writer.writerow(row)
            rows += 1
            if rows % 200 == 0:
                meas_file.flush()
                print(
                    f"[DFT] rows={rows} CSI={inputs.csi_packets} pose={inputs.pose_packets} "
                    f"dropped={inputs.csi_dropped} phase={phase}",
                    flush=True,
                )
    finally:
        inputs.close()
        hw.close()
        meas_file.flush(); meas_file.close()
        sweep_file.flush(); sweep_file.close()
        metadata = {
            "schema": "fr3-dft-measurement-v1",
            "created": datetime.now().isoformat(),
            "mode": a.mode,
            "config_path": str(Path(a.config).resolve()),
            "feature_names": list(FEATURE_NAMES),
            "codebook": cb.to_dict(),
            "radio_topology": cfg.get("radio_topology", {}),
            "label_source": "future_geometry" if a.mode == "geometry" else "measured_sweep" if a.mode == "sweep" else None,
            "geometry_apply_lead_s": float(cfg["dft_beam"].get("geometry_apply_lead_s", 0.08)),
            "geometry_settle_s": geometry_controller.settle_s if geometry_controller else None,
            "geometry_skipped_csi": geometry_skipped,
            "geometry_label_lead_s": float(cfg["dft_beam"].get("prediction_horizon_s", 0.25)),
            "rows": rows,
            "csi_packets": inputs.csi_packets,
            "csi_dropped": inputs.csi_dropped,
            "pose_packets": inputs.pose_packets,
            "dry_run": bool(a.dry_run or hw.tmy.dry),
            "model": str(Path(a.model).resolve()) if a.model else None,
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[DFT] saved {rows} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
