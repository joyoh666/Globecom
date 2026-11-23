#!/usr/bin/env python3
"""Send synthetic FDM CSI and TurtleBot telemetry for DFT pipeline checks."""
from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import time

import numpy as np


PACK = struct.Struct("<dii6f")


def addr(value: str) -> tuple[str, int]:
    host, port = value.rsplit(":", 1)
    return host, int(port)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csi-udp", default="127.0.0.1:5010")
    ap.add_argument("--pose-udp", default="127.0.0.1:5011")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--pose-rate", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    csi_addr, pose_addr = addr(a.csi_udp), addr(a.pose_udp)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rng = np.random.default_rng(a.seed)
    count = max(1, int(a.duration * a.rate))
    pose_stride = max(1, int(round(a.rate / a.pose_rate)))
    started = time.perf_counter()
    for index in range(count):
        due = started + index / a.rate
        delay = due - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        t = index / a.rate
        x = 0.15 * t / max(a.duration, 1e-6)
        y = 0.08 * math.sin(2.0 * math.pi * t / max(a.duration, 1.0))
        yaw = 12.0 * math.sin(2.0 * math.pi * t / max(a.duration * 0.8, 1.0))
        phase = 2.0 * math.pi * np.asarray([0.4, 0.7, 1.1]) * t
        amp = np.asarray([0.10, 0.08, 0.06]) * (1.0 + 0.15 * np.sin(phase * 0.2 + np.arange(3)))
        h = amp * np.exp(1j * phase)
        h += (rng.normal(size=3) + 1j * rng.normal(size=3)) * 5e-4
        sock.sendto(
            PACK.pack(t, index // 2, index % 2, *[float(v) for z in h for v in (z.real, z.imag)]),
            csi_addr,
        )
        if index % pose_stride == 0:
            pose = {
                "schema": "fr3-turtlebot-telemetry-v1-sim",
                "seq": index // pose_stride,
                "ts": time.time(),
                "x": x,
                "y": y,
                "yaw_deg": yaw,
                "vx": 0.15 / max(a.duration, 1e-6),
                "vy": 0.08 * (2.0 * math.pi / max(a.duration, 1.0)) * math.cos(2.0 * math.pi * t / max(a.duration, 1.0)),
                "yaw_rate_dps": 12.0 * (2.0 * math.pi / max(a.duration * 0.8, 1.0)) * math.cos(2.0 * math.pi * t / max(a.duration * 0.8, 1.0)),
                "sx": 0.0,
                "sy": 0.0,
                "syaw": 0.0,
            }
            sock.sendto(json.dumps(pose, separators=(",", ":")).encode("utf-8"), pose_addr)
    sock.close()
    print(f"[DFT-SIM] sent CSI={count} pose~={math.ceil(count/pose_stride)} to {csi_addr}/{pose_addr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

