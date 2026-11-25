# -*- coding: utf-8 -*-
"""[GC 1124] GRU 입력용 빔 각도 특징 — 학습(build/train)과 실시간(대시보드)이 같은 식을 쓰도록 한 곳에 둔다.

빔 각도 = RU 정면 기준 조향각 [deg] (beam_tracker 가 BBox 에 넣는 값과 같은 정의, ±max_deg 로 자름).
특징   = [sin θ, cos θ]  (θ rad) — 각도 2개를 RU 마다 GRU 입력 끝에 붙인다.

각도의 출처는 둘:
  ① 빔 추적을 켜고 모은 데이터 → beam_tracker 로그(logs/beam_*.csv)의 실제 적용 각도
  ② 빔 없이 모은 데이터(2.2 GHz ISO, 10-26 S66~S75) → 로봇 위치에서 '추적기가 넣었을 각도'를 계산
     (빔이 없었으므로 이때 각도는 '로봇이 RU 에서 어느 방향에 있나' 라는 기하 정보다)
"""
from __future__ import annotations

import math

import numpy as np

from . import geometry as geo

BEAM_FEAT_DIM = 2


def ru_steer_deg(rus: list[dict], px: float, py: float, max_deg: float) -> np.ndarray:
    """시작 자세 좌표 (px, py) 에 대한 RU 1·2·3 조향각 [deg] (3,)."""
    out = np.zeros(3, np.float32)
    for r in rus:
        d = geo.steer_deg(r, px, py)
        out[int(r["ru"]) - 1] = max(-max_deg, min(max_deg, d))
    return out


def steer_track(rus: list[dict], px: np.ndarray, py: np.ndarray, max_deg: float) -> np.ndarray:
    """위치 열 (T,) → 조향각 (T, 3)."""
    return np.stack([ru_steer_deg(rus, float(x), float(y), max_deg) for x, y in zip(px, py)]).astype(np.float32)


def angle_feats(deg: np.ndarray) -> np.ndarray:
    """각도 [deg] (...,) → 특징 (..., 2) = [sin, cos]."""
    r = np.deg2rad(np.asarray(deg, np.float32))
    return np.stack([np.sin(r), np.cos(r)], axis=-1).astype(np.float32)


def step_mean_deg(deg_ticks: np.ndarray, taps: int) -> np.ndarray:
    """5 ms 틱 각도 (L, ...) → GRU 스텝(taps 틱) 평균 (L//taps, ...)."""
    deg_ticks = np.asarray(deg_ticks, np.float32)
    S = deg_ticks.shape[0] // taps
    return deg_ticks[: S * taps].reshape((S, taps) + deg_ticks.shape[1:]).mean(axis=1)


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def odom_to_start(x, y, sx, sy, syaw_deg):
    """배열판 geo.odom_to_start."""
    c, s = math.cos(math.radians(syaw_deg)), math.sin(math.radians(syaw_deg))
    dx, dy = np.asarray(x) - sx, np.asarray(y) - sy
    return c * dx + s * dy, -s * dx + c * dy
