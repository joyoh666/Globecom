# -*- coding: utf-8 -*-
"""
[GC 1117] 로봇 위치를 따라 RU 3개의 빔 각도를 실시간으로 바꾼다.

    python -m beam.beam_tracker                 # 추적 (장비 없으면 자동 dry-run)
    python -m beam.beam_tracker --fixed         # 비교용: 정면(0°) 고정
    python -m beam.beam_tracker --dry-run       # TMYTEK 장비 없이 각도 계산·송출만

입력: 로봇 위치 UDP 5007 (runtime/hy_robot_proc.py 가 보냄: x, y, yaw, vx, vy, sx, sy, syaw)
출력: ① BBox 6대에 setBeamAngle (RU 안 B 쪽은 dphi 보정)  ② 빔 상태 UDP 5008 → 대시보드  ③ logs/beam_YYYYmmdd_HHMMSS.csv
      ④ [FR3] UE 4×4 UPA (RX BBox) 빔: config geometry.rx.policy — fixed(배열 정면) | centroid(로봇 회전을 상쇄해 세 RU 가운데 쪽)
위치 예측: 속도로 lead_s 만큼 앞을 외삽 (로봇이 느려 50~100 ms 면 충분, 경로를 몰라도 됨 → random 모드에서도 같다).
흔들림 방지: step_deg 이상 바뀔 때만 다시 설정.

⚠ 빔을 바꾸면 UE 가 재는 채널도 바뀐다 → GRU 는 추적을 켠 상태로 모은 데이터로 다시 학습해야 정확하다.
"""
from __future__ import annotations

import argparse
import csv
import json
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

from . import geometry as geo
from tmytek.tmy_ctl import TMY, addrs_of, cfg_gain


class PoseRx:
    def __init__(self, addr):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(addr)
        self.sock.settimeout(0.2)
        self.last = None
        self.start = None
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
                p = json.loads(data.decode("utf-8"))
            except socket.timeout:
                continue
            except Exception:                              # noqa: BLE001
                continue
            with self.lock:
                if "sx" in p:
                    self.start = (float(p["sx"]), float(p["sy"]), float(p["syaw"]))
                elif self.start is None:                   # 시작 자세를 아직 못 받았으면 첫 위치를 원점으로
                    self.start = (float(p["x"]), float(p["y"]), float(p.get("yaw_deg", 0.0)))
                p["_rx"] = time.time()
                self.last = p

    def get(self):
        with self.lock:
            return self.last, self.start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(geo.DEFAULT_CONFIG))
    ap.add_argument("--pose-udp", default="0.0.0.0:5007")
    ap.add_argument("--beam-udp", default="127.0.0.1:5008", help="대시보드로 보내는 빔 상태")
    ap.add_argument("--fixed", action="store_true", help="추적 끄고 정면 고정 (비교용)")
    ap.add_argument("--fixed-deg", type=float, nargs=3, default=None, help="RU1 RU2 RU3 고정 각도 [deg]")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-setup", action="store_true", help="(FR3 에서는 무시) 그룹·주파수는 TLKCore 안 상태라 프로세스마다 다시 만든다")
    ap.add_argument("--no-rx", action="store_true", help="UE(RX BBox) 빔은 건드리지 않는다")
    ap.add_argument("--log-dir", default=str(geo.CODE_ROOT / "logs"))
    a = ap.parse_args()

    cfg = geo.load_config(a.config)
    bcfg, rus = cfg["beam"], {r["ru"]: r for r in cfg["geometry"]["rus"]}
    t = cfg["tmytek"]
    boxes = t["bbox"]
    rxboxes = [] if a.no_rx else t.get("rx_bbox", [])
    rxcfg = cfg["geometry"].get("rx", {"policy": "fixed"})
    tmy = TMY(t.get("working_root", "."), dry_run=a.dry_run)
    tmy.open([b["sn"] for b in boxes + rxboxes], addrs_of(boxes + rxboxes))
    gains = {}
    if a.no_setup:
        print("[BEAM] --no-setup 은 FR3 에서 무시한다 (그룹·주파수를 이 프로세스에서 다시 만든다. 장치에는 모드만 다시 보낸다)")
    for b, mode in [(b, "TX") for b in boxes] + [(b, "RX") for b in rxboxes]:
        dr = tmy.setup_bbox(b["sn"], cfg["freq"]["rf_ghz"], mode=mode, box=b)
        gains[b["sn"]] = cfg_gain(t, dr, mode.lower())

    h, p = a.pose_udp.rsplit(":", 1)
    pose = PoseRx((h, int(p)))
    bh, bp = a.beam_udp.rsplit(":", 1)
    out_sock, out_addr = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (bh, int(bp))

    Path(a.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(a.log_dir) / f"beam_{datetime.now():%Y%m%d_%H%M%S}.csv"
    logf = open(log_path, "w", newline="", encoding="utf-8")
    lw = csv.writer(logf)
    lw.writerow(["t_os", "mode", "px", "py", "yaw_s", "ru1_deg", "ru2_deg", "ru3_deg", "rx_deg", "applied", "apply_ms"])
    print(f"[BEAM] {'FIXED' if a.fixed else 'TRACK'} · {bcfg['rate_hz']} Hz · lead {bcfg['lead_s']} s · "
          f"UE {'off' if not rxboxes else rxcfg.get('policy', 'fixed')} · log {log_path}")

    applied = {ru: None for ru in rus}
    applied["rx"] = None
    period = 1.0 / float(bcfg["rate_hz"])
    next_t = time.monotonic()

    def apply(ru, deg):
        theta, phi = geo.to_theta_phi(deg, rus[ru].get("phi_sign", 1))
        for b in boxes:
            if b["ru"] == ru:
                tmy.set_beam(b["sn"], gains[b["sn"]], theta, phi, b.get("dphi_deg", 0.0))

    def apply_rx(deg):
        theta, phi = geo.to_theta_phi(deg, rxcfg.get("phi_sign", 1))
        for b in rxboxes:
            tmy.set_beam(b["sn"], gains[b["sn"]], theta, phi, b.get("dphi_deg", 0.0))

    try:
        while True:
            last, start = pose.get()
            px = py = float("nan")
            yaw_s = float("nan")
            if a.fixed or last is None or start is None:
                target = {ru: (a.fixed_deg[ru - 1] if a.fixed_deg else 0.0) for ru in rus}
                target["rx"] = float(rxcfg.get("fixed_deg", 0.0))
            else:
                x = float(last["x"]) + float(last.get("vx", 0.0)) * bcfg["lead_s"]
                y = float(last["y"]) + float(last.get("vy", 0.0)) * bcfg["lead_s"]
                px, py = geo.odom_to_start(x, y, *start)
                target = {}
                for ru, r in rus.items():
                    d = geo.steer_deg(r, px, py)
                    target[ru] = max(-bcfg["max_deg"], min(bcfg["max_deg"], d))
                yaw_s = geo.wrap_deg(float(last.get("yaw_deg", 0.0)) - start[2])     # 시작 자세 기준 로봇 방향
                d = geo.rx_steer_deg(rxcfg, list(rus.values()), px, py, yaw_s)
                target["rx"] = max(-bcfg["max_deg"], min(bcfg["max_deg"], d))
            t_apply = time.perf_counter()
            changed = []
            for ru, deg in target.items():
                if ru == "rx" and not rxboxes:
                    continue
                if applied[ru] is None or abs(deg - applied[ru]) >= bcfg["step_deg"]:
                    apply_rx(deg) if ru == "rx" else apply(ru, deg)
                    applied[ru] = deg
                    changed.append(ru)
            apply_ms = (time.perf_counter() - t_apply) * 1e3
            msg = {"t": time.time(), "mode": "fixed" if a.fixed else "track", "px": px, "py": py,
                   "deg": [applied[1], applied[2], applied[3]], "changed": changed, "apply_ms": apply_ms,
                   "dry": tmy.dry, "rx_deg": applied["rx"], "yaw_s": yaw_s,
                   "rx_mount_deg": float(rxcfg.get("mount_yaw_deg", 0.0))}
            try:
                out_sock.sendto(json.dumps(msg).encode("utf-8"), out_addr)
            except OSError:
                pass
            lw.writerow([msg["t"], msg["mode"], px, py, yaw_s, *msg["deg"], applied["rx"], "/".join(map(str, changed)), round(apply_ms, 2)])
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        print("\n[BEAM] 종료")
    finally:
        logf.close()
        tmy.close()


if __name__ == "__main__":
    main()
