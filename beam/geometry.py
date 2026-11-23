# -*- coding: utf-8 -*-
"""[GC 1117] 빔 추적 기하. 좌표 = '이동 시작 자세' 기준 (출발점 원점, 처음 바라본 방향 +x, 왼쪽 +y, m)."""
from __future__ import annotations

import json
import math
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CODE_ROOT / "config" / "demo_config.json"


def load_config(path=None) -> dict:
    return json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def odom_to_start(x: float, y: float, sx: float, sy: float, syaw_deg: float) -> tuple[float, float]:
    """odom 좌표 → 시작 자세 좌표."""
    c, s = math.cos(math.radians(syaw_deg)), math.sin(math.radians(syaw_deg))
    dx, dy = x - sx, y - sy
    return c * dx + s * dy, -s * dx + c * dy


def steer_deg(ru: dict, px: float, py: float) -> float:
    """RU 정면(boresight) 기준으로 로봇이 있는 방향 [deg], +는 RU 에서 봤을 때 반시계."""
    ang = math.degrees(math.atan2(py - ru["y"], px - ru["x"]))
    return wrap_deg(ang - ru["boresight_deg"])


def to_theta_phi(steer: float, phi_sign: int = 1) -> tuple[float, float]:
    """수평 1×4 줄: theta = |조향각|, phi = 0 또는 180 (방향). phi_sign 은 첫날 실측으로 맞춘다 [모름]."""
    s = steer * (1 if phi_sign >= 0 else -1)
    return abs(s), (0.0 if s >= 0 else 180.0)


def ru_centroid_dir_deg(rus: list[dict], px: float, py: float) -> float:
    """로봇에서 본 세 RU 의 가운데 방향 (단위벡터 평균) [deg, 시작 자세 좌표]."""
    cx = sum(math.cos(math.atan2(r["y"] - py, r["x"] - px)) for r in rus)
    cy = sum(math.sin(math.atan2(r["y"] - py, r["x"] - px)) for r in rus)
    return math.degrees(math.atan2(cy, cx))


def rx_steer_deg(rx: dict, rus: list[dict], px: float, py: float, yaw_start_deg: float) -> float:
    """[GC FR3] UE 배열 조향각 (배열 정면 기준, +는 반시계). yaw_start = 시작 자세 기준 로봇 방향.
    fixed: 로봇이 돌면 같이 돈다 · centroid: 로봇 회전을 상쇄해 RU 무리 쪽을 본다."""
    if rx.get("policy", "fixed") == "fixed":
        return float(rx.get("fixed_deg", 0.0))
    face = yaw_start_deg + float(rx.get("mount_yaw_deg", 0.0))
    return wrap_deg(ru_centroid_dir_deg(rus, px, py) - face)
