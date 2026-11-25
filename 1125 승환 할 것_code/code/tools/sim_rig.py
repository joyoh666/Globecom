# -*- coding: utf-8 -*-
"""[GC FR3 1117] 장비 없이 FR3 도구(fr3_check · calib_ru)를 시험하는 가짜 장비.
TMY 와 CsiRx 를 한 객체로 흉내 낸다: BBox 켜고 끄기·이득·dphi 가 CSI 에 반영되고, 수신 포화도 넣었다.
숨은 정답: RU 안 B 쪽 위상 offset(true_dphi), 위상기 부호(sign), 포화 시작 이득.  ⚠ 모형은 단순화 [추론]."""
from __future__ import annotations

import time

import numpy as np


class SimRig:
    dry = True

    def __init__(self, cfg, seed=0, sign=-1, sat_backoff_db=14.0, cfo_hz=0.0):
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.boxes = {b["sn"]: b for b in cfg["tmytek"]["bbox"]}
        self.on = {sn: True for sn in self.boxes}
        self.gain = {sn: 49.0 for sn in self.boxes}
        self.dphi = {sn: 0.0 for sn in self.boxes}
        self.dr = {sn: [18.0, 49.0] for sn in self.boxes}
        self.sign = sign                                   # 위상기 부호 규약 (모름 → 교정이 맞혀야 함)
        # 숨은 정답: B 쪽 경로가 A 보다 이만큼 앞선다 (스플리터·케이블·BBox 차)
        self.true_off = {sn: (float(self.rng.uniform(-180, 180)) if b["row"] == "lower" else 0.0)
                         for sn, b in self.boxes.items()}
        self.box_amp = {sn: float(self.rng.uniform(0.8, 1.2)) for sn in self.boxes}
        self.ru_ch = {1: 1.0 * np.exp(1j * 0.3), 2: 0.7 * np.exp(1j * 2.0), 3: 0.5 * np.exp(-1j * 1.1)}
        self.sat_in = len(self.boxes) * 10 ** ((49.0 - sat_backoff_db) / 20)   # 6대가 이 backoff 보다 세면 수신이 눌린다
        self.cfo = cfo_hz
        self.t = 0.0
        self.types = {}

    # --- TMY 흉내 ---
    def open(self, sns=None):
        self.types = {sn: "sim" for sn in (sns or [])}
        return self.types

    def setup_bbox(self, sn, freq, kit="4x4", mode="TX", box=None):
        return self.dr.get(sn, [18.0, 49.0])

    def max_gain(self, sn, mode="TX"):
        return self.dr.get(sn)

    def set_beam(self, sn, gain_db, theta, phi, dphi_deg=0.0):
        if sn in self.boxes:
            self.gain[sn] = float(gain_db)
            self.dphi[sn] = float(dphi_deg)

    def switch_channels(self, sn, off):
        if sn in self.boxes:
            self.on[sn] = not off

    def close(self):
        pass

    # --- CsiRx 흉내 ---
    count = 1

    def collect(self, seconds, settle=0.0):
        n = max(1, int(seconds / 0.005))
        t = self.t + np.arange(n) * 0.005
        self.t = t[-1] + 0.005
        common = np.exp(1j * (self.rng.uniform(-np.pi, np.pi) + 2 * np.pi * self.cfo * t))   # 캡처마다 공통 위상
        tot_amp = 0.0
        h = np.zeros((n, 3), complex)
        for ru in (1, 2, 3):
            s = 0j
            for sn, b in self.boxes.items():
                if b["ru"] != ru or not self.on[sn]:
                    continue
                a = 10 ** (self.gain[sn] / 20) * self.box_amp[sn]
                s += a * np.exp(1j * np.radians(self.true_off[sn] + self.sign * self.dphi[sn]))
                tot_amp += a
            h[:, ru - 1] = s * self.ru_ch[ru]
        comp = 1.0 / np.sqrt(1.0 + (tot_amp / self.sat_in) ** 2)       # 포화: 출력이 sat 근처에서 평평해짐
        noise = (self.rng.normal(size=(n, 3)) + 1j * self.rng.normal(size=(n, 3))) * 2.0
        return t, (h * comp + noise) * common[:, None] * 1e-3

    def close_rx(self):
        pass
