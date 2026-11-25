# -*- coding: utf-8 -*-
"""
[GC 1117] 장비 없이 리허설: 저장된 trial 의 5 ms CSI 와 로봇 위치를 실시간 속도로 UDP 에 다시 보낸다.
RX 스크립트·로봇 자식과 같은 패킷이라 대시보드·빔 추적기를 그대로 시험할 수 있다.

    python live_demo/replay_sender.py <trial 폴더> [폴더 ...] [--loop] [--speed 1] [--restart-each]
    (trial 폴더 = usrp_sparse_h.npz · turtlebot_states.csv · scenario_marker.json 이 있는 곳.
     배치 폴더를 주면 그 안의 trial 을 차례로)

  CSI  → --udp       (기본 127.0.0.1:5005)
  위치 → --pose-udp  (기본 127.0.0.1:5006,127.0.0.1:5007  = 대시보드 + 빔 추적기)
  --restart-each : trial 마다 frame_idx 를 0 부터 (배치 모드처럼 스트림이 끊겼다 다시 시작하는 것을 흉내)
"""
import argparse
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np

FMT = struct.Struct("<dii6f")


def expand(paths):
    out = []
    for p in map(Path, paths):
        if (p / "usrp_sparse_h.npz").exists():
            out.append(p)
        else:
            out += sorted(q.parent for q in p.glob("*/usrp_sparse_h.npz"))
    return out


def load_run(run_dir: Path, full: bool):
    z = np.load(run_dir / "usrp_sparse_h.npz")
    c, t = z["csi"], z["csi_t_usrp"]
    mk = run_dir / "scenario_marker.json"
    m = json.loads(mk.read_text(encoding="utf-8")) if mk.exists() else None
    if m and not full:
        sel = (t >= m["t0_usrp_est"]) & (t < m["t1_usrp_est"])
        c, t = c[sel], t[sel]
    poses = np.zeros((0, 7))
    pc = run_dir / "turtlebot_states.csv"
    if pc.exists():
        try:
            poses = np.loadtxt(pc, delimiter=",", skiprows=1, ndmin=2)       # t_os, x, y, yaw_deg, vx, vy, vz
        except ValueError:
            poses = np.zeros((0, 7))
    off = (m["t0_usrp_est"] - m["t0_wall"]) if m else (t[0] - poses[0, 0] if len(poses) and len(t) else 0.0)
    if len(poses):
        poses = poses.copy()
        poses[:, 0] += off                                                  # host 시각 → USRP 시각
    start = None
    if len(poses) and len(t):
        k = int(np.searchsorted(poses[:, 0], t[0]))
        k = min(k, len(poses) - 1)
        start = poses[k, 1:4]
    return c, t, poses, start


def send_runs(runs, udp="127.0.0.1:5005", pose_udp="127.0.0.1:5006,127.0.0.1:5007", speed=1.0, loop=False,
              restart_each=False, full=False, motion="replay", stop=None):
    host, port = udp.rsplit(":", 1)
    addr = (host, int(port))
    paddrs = [(h, int(p)) for h, p in (x.strip().rsplit(":", 1) for x in pose_udp.split(",") if x.strip())]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    key = 0
    while True:
        for rd in runs:
            c, t, poses, start = load_run(Path(rd), full)
            if restart_each:
                key = 0
            print(f"[REPLAY] {Path(rd).name}  {len(c)} windows ({len(c) * 0.005:.1f} s) · poses {len(poses)} x{speed}")
            t_start = time.perf_counter()
            pi = 0
            for i in range(len(c)):
                if stop is not None and stop.is_set():
                    return
                due = t_start + i * 0.005 / speed
                while True:
                    dt = due - time.perf_counter()
                    if dt <= 0:
                        break
                    time.sleep(min(dt, 0.002))
                h = c[i]
                s.sendto(FMT.pack(float(t[i]), key // 2, key % 2, *[float(v) for x in h for v in (x.real, x.imag)]), addr)
                key += 1
                while pi < len(poses) and poses[pi, 0] <= t[i]:
                    r = poses[pi]
                    msg = {"ts": float(r[0]), "x": float(r[1]), "y": float(r[2]), "yaw_deg": float(r[3]),
                           "vx": float(r[4]), "vy": float(r[5]), "vz": float(r[6]), "phase": "run", "motion": motion}
                    if start is not None:
                        msg.update({"sx": float(start[0]), "sy": float(start[1]), "syaw": float(start[2])})
                    data = json.dumps(msg).encode("utf-8")
                    for pa in paddrs:
                        s.sendto(data, pa)
                    pi += 1
        if not loop:
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--udp", default="127.0.0.1:5005")
    ap.add_argument("--pose-udp", default="127.0.0.1:5006,127.0.0.1:5007")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--restart-each", action="store_true")
    ap.add_argument("--full", action="store_true", help="시나리오 창(t0..t1) 밖의 pre/post-roll 도 보낸다")
    a = ap.parse_args()
    runs = expand(a.runs)
    if not runs:
        raise SystemExit("trial 폴더를 못 찾았다")
    try:
        send_runs(runs, a.udp, a.pose_udp, a.speed, a.loop, a.restart_each, a.full)
    except KeyboardInterrupt:
        pass
