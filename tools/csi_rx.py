# -*- coding: utf-8 -*-
"""[GC FR3 1117] 5 ms CSI UDP 수신기 (runtime/DMIMO_3TX_Live_Rx.py 의 '<dii6f' 패킷).
측정 스크립트를 LIVE_UDP="127.0.0.1:5005,127.0.0.1:5009" 로 돌리면 대시보드(5005)와 이 도구(5009)가 같이 받는다."""
from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np

FMT = struct.Struct("<dii6f")


class CsiRx:
    def __init__(self, addr="0.0.0.0:5009"):
        h, p = addr.rsplit(":", 1)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((h, int(p)))
        self.sock.settimeout(0.2)
        self.lock = threading.Lock()
        self.buf: list[tuple[float, float, np.ndarray]] = []     # (t_os, t_usrp, H[3])
        self.count = 0
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) != FMT.size:
                continue
            t, _f, _half, *v = FMT.unpack(data)
            h = np.array([complex(v[0], v[1]), complex(v[2], v[3]), complex(v[4], v[5])], dtype=np.complex128)
            with self.lock:
                self.buf.append((time.time(), t, h))
                self.count += 1
                if len(self.buf) > 20000:
                    del self.buf[:5000]

    def collect(self, seconds: float, settle: float = 0.0):
        """settle 초 버리고 seconds 동안 모은 (t_usrp[N], H[N,3])."""
        if settle > 0:
            time.sleep(settle)
        t0 = time.time()
        time.sleep(seconds)
        with self.lock:
            rows = [r for r in self.buf if r[0] >= t0]
        if not rows:
            return np.zeros(0), np.zeros((0, 3), complex)
        return np.array([r[1] for r in rows]), np.array([r[2] for r in rows])

    def close(self):
        self._stop.set()
        self.sock.close()


def wait_for_csi(rx: CsiRx, timeout=10.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if rx.count > 0:
            return True
        time.sleep(0.2)
    return False


def phase_rate_hz(t: np.ndarray, h: np.ndarray) -> float:
    """한 TX 의 위상 기울기 [Hz] — UD/USRP 기준이 잠겼으면 ≈0 (CFO 잔차)."""
    ok = np.isfinite(h) & (np.abs(h) > 0)
    if ok.sum() < 10:
        return float("nan")
    ph = np.unwrap(np.angle(h[ok]))
    tt = t[ok] - t[ok][0]
    if tt[-1] <= 0:
        return float("nan")
    return float(np.polyfit(tt, ph, 1)[0] / (2 * np.pi))


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-15)
