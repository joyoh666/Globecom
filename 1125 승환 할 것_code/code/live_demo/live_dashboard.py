#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[GC 1117] Globecom real-time dashboard: CSI -> GRU channel prediction -> DQN RU scheduling, plus robot
position and beam angles. Same equations as dataset_dashboard.OfflinePipeline (Summit 11-10), run online.

Inputs (UDP):
  5005  CSI   struct '<dii6f' = t_usrp, frame_idx, half, Re/Im H0..H2   (runtime/DMIMO_3TX_Live_Rx.py, every 5 ms)
  5006  pose  JSON {x, y, yaw_deg, vx, vy, sx, sy, syaw, motion}         (runtime/hy_robot_proc.py)
  5008  beam  JSON {mode, deg[3], px, py}                                (beam/beam_tracker.py, optional)

    python live_demo/live_dashboard.py                      # live
    python live_demo/live_dashboard.py --headless 30        # no GUI, print metrics after 30 s (test)
    python live_demo/live_dashboard.py --snapshot out.png --snapshot-sec 20

Stream restarts (frame_idx jumps back) reset the GRU context and the scheduler.
Scale: first --auto-scale seconds set the CSI scale to the training power (1e-2), or give --scale.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
import hj_common as hc  # noqa: E402
from beam import geometry as geo  # noqa: E402

PACK = struct.Struct("<dii6f")
TARGET_POWER = 1e-2


class LivePipeline:
    """Online version of OfflinePipeline. All indices are 5 ms ticks."""

    def __init__(self, a, predictor: hc.NumpyPredictor, dqn: hc.NumpyDQN):
        self.a, self.pred, self.dqn = a, predictor, dqn
        self.taps, self.K = predictor.taps, predictor.K
        self.seg = int(dqn.cfg["seg_len_ticks"])
        self.hist_bins = int(dqn.cfg["obs_hist_bins"])
        self.phase_std = float(dqn.cfg["inter_ru_phase_noise_std"])
        self.noise_power = float(dqn.cfg["noise_power"])
        self.snr_db = float(dqn.cfg.get("reward_snr_db", 10.0))
        self.N = int(a.max_minutes * 60 / hc.DT)
        self.lock = threading.Lock()
        self.scale = a.scale if a.scale > 0 else None
        self.restarts = 0
        self.t_proc = deque(maxlen=200)
        self.beam_now = np.zeros(3, np.float32)            # [GC 1124] 지금 적용된 RU 빔 각도 [deg] (UDP 5008, 없으면 0°)
        self.reset()

    def reset(self):
        N = self.N
        self.raw = np.zeros((N, 3), np.complex64)
        self.x = np.zeros((N, 3), np.complex64)
        self.ang = np.zeros((N, 3), np.float32)            # [GC 1124] 틱별 빔 각도 (각도 입력 GRU 용)
        self.t_usrp = np.full(N, np.nan)
        self.real = np.zeros(N, bool)
        self.n = 0
        self.last_key = None
        self.gaps = self.dups = 0
        self.pred_at = np.full((self.K, N, 3), np.nan + 0j, np.complex64)
        self.err_hist = []                                 # (tick, k, model_err, power, lr_err, kf_err)
        self.steps_done = self.segs_done = self.skipped = 0
        self.ru_gain, self.state_rate, self.q_hist = [], [], []
        self.serve = [self.a.init_state] if self.dqn.timing == "next-segment" else [self.a.init_state] * 2
        self.dqn.reset()
        self.rng = np.random.RandomState(0)

    # ---------------- receive ----------------
    def push(self, frame_idx: int, half: int, t: float, h: np.ndarray):
        key = 2 * frame_idx + half
        with self.lock:
            if self.last_key is not None and key <= self.last_key:
                if self.last_key - key < 400:
                    self.dups += 1
                    return
                print("[UDP ] stream restarted (frame_idx went back) -> reset GRU context and scheduler")
                self.restarts += 1
                self.reset()
            miss = 0 if self.last_key is None else key - self.last_key - 1
            if miss > 200:                                 # long outage: do not fill, start a new context
                print(f"[UDP ] {miss} windows missing -> reset")
                self.restarts += 1
                self.reset()
                miss = 0
            for _ in range(max(0, miss)):
                self._append(self.raw[self.n - 1] if self.n else h, np.nan, False)
            self.gaps += max(0, miss)
            self._append(h, t, True)
            self.last_key = key

    def _append(self, h, t, real):
        if self.n >= self.N:
            raise SystemExit("--max-minutes exceeded")
        i = self.n
        self.raw[i], self.t_usrp[i], self.real[i] = h, t, real
        self.ang[i] = self.beam_now
        if self.scale is not None:
            self.x[i] = h * self.scale
        self.n += 1
        if self.scale is None and self.n >= int(self.a.auto_scale / hc.DT):
            p = float(np.mean(np.abs(self.raw[:self.n]) ** 2))
            self.scale = math.sqrt(TARGET_POWER / max(p, 1e-30))
            self.x[:self.n] = self.raw[:self.n] * self.scale
            print(f"[SCALE] auto {self.scale:.3f} (raw mean |h|^2 {p:.3e})")

    # ---------------- process (worker thread) ----------------
    def process(self):
        n = self.n if self.scale is not None else 0        # caller holds self.lock
        while self.segs_done < n // self.seg:              # scheduler first (cheap)
            self._schedule_segment(self.segs_done)
            self.segs_done += 1
        last = n // self.taps
        if last - self.steps_done > self.a.max_lag_steps:  # predictor fell behind: skip to now
            self.skipped += last - 1 - self.steps_done
            self.steps_done = last - 1
        while self.steps_done < last:
            t0 = time.perf_counter()
            self._predict_step(self.steps_done)
            self.t_proc.append((time.perf_counter() - t0) * 1e3)
            self.steps_done += 1

    def _predict_step(self, s: int):
        tp, K = self.taps, self.K
        end = (s + 1) * tp
        for k in range(K):
            if s - (k + 1) < 0:
                continue
            pred = self.pred_at[k, s * tp:end]
            if np.isnan(pred.real).any():
                continue
            gt = self.x[s * tp:end]
            gt0 = self.x[(s - k - 1) * tp:(s - k) * tp].T
            lr = hc.lr_predict(gt0, (k + 1) * tp)[:, k * tp:].T
            kf = hc.kf_predict(gt0, (k + 1) * tp)[:, k * tp:].T
            self.err_hist.append((end, k, float(np.mean(np.abs(pred - gt) ** 2)), float(np.mean(np.abs(gt) ** 2)),
                                  float(np.mean(np.abs(lr - gt) ** 2)), float(np.mean(np.abs(kf - gt) ** 2))))
        context = min(self.a.context_steps, s + 1)
        if context < self.a.min_context_steps:
            return
        bdeg = self.ang[end - context * tp:end].T if self.pred.beam_dim else None
        future = self.pred.predict_complex(self.x[end - context * tp:end].T, bdeg).reshape(3, K, tp)
        for k in range(K):
            a_, b_ = end + k * tp, end + (k + 1) * tp
            if b_ <= self.N:
                self.pred_at[k, a_:b_] = future[:, k].T

    def _schedule_segment(self, c: int):
        a_, b_ = c * self.seg, (c + 1) * self.seg
        gains = np.abs(self.x[a_:b_])
        self.ru_gain.append(hc.ru_gain_segment(gains))
        self.state_rate.append(hc.all_state_power_segment(gains, self.phase_std, self.noise_power, self.snr_db, self.rng))
        if self.dqn.timing == "next-segment":
            seg_k, conn = c, self.serve[c]
        else:
            seg_k, conn = c + 1, self.serve[c + 1]
        action, q = self.dqn.step(hc.make_obs(self.ru_gain, self.serve, seg_k, conn, self.hist_bins))
        self.serve.append(action)
        self.q_hist.append(q)

    # ---------------- metrics ----------------
    def nmse_db(self, last_ticks: int) -> dict:
        n = self.n
        rows = [r for r in self.err_hist[-5000:] if r[0] > n - last_ticks]
        out = {}
        for name, idx in (("GRU", 2), ("LR", 4), ("KF", 5)):
            vals = []
            for k in range(self.K):
                sel = [r for r in rows if r[1] == k]
                num, den = sum(r[idx] for r in sel), sum(r[3] for r in sel)
                vals.append(10 * np.log10(num / den) if den > 0 and num > 0 else np.nan)
            out[name] = vals
        return out

    def rate_summary(self, last_segs: int) -> dict:
        cnt = len(self.state_rate)
        lo = max(0, cnt - last_segs)
        if cnt <= lo:
            return {}
        rates = np.asarray(self.state_rate[lo:cnt])
        sel = np.asarray(self.serve[lo:cnt], dtype=np.int64)[: len(rates)]     # same indexing as OfflinePipeline
        rates = rates[: len(sel)]
        if len(sel) == 0:
            return {}
        return {"DQN": float(rates[np.arange(len(sel)), sel].mean()), "Oracle": float(rates.max(1).mean()),
                "Random": float(rates.mean()), "best_ratio": float(np.mean(sel == rates.argmax(1))),
                "handover": float(np.mean(np.diff(sel) != 0)) if len(sel) > 1 else 0.0}

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        n = self.n
        np.savez_compressed(out_dir / "live_pipeline.npz", raw=self.raw[:n], x=self.x[:n], t_usrp=self.t_usrp[:n],
                            real=self.real[:n], pred_at=self.pred_at[:, :n], serve=np.array(self.serve),
                            state_rate=np.array(self.state_rate), q=np.array(self.q_hist),
                            err_hist=np.array(self.err_hist), scale=self.scale or np.nan)


class JsonRx:
    def __init__(self, addr, maxlen=3000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(addr)
        self.sock.settimeout(0.2)
        self.buf = deque(maxlen=maxlen)
        self.count = 0
        self.lock = threading.Lock()

    def loop(self, stop):
        while not stop.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
                p = json.loads(data.decode("utf-8"))
            except socket.timeout:
                continue
            except Exception:                              # noqa: BLE001
                continue
            p["_rx"] = time.time()
            with self.lock:
                self.buf.append(p)
                self.count += 1

    def snapshot(self):
        with self.lock:
            return list(self.buf)


def csi_loop(pipe: LivePipeline, addr, stop, stats):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(addr)
    s.settimeout(0.2)
    while not stop.is_set():
        try:
            data, _ = s.recvfrom(256)
        except socket.timeout:
            continue
        if len(data) != PACK.size:
            continue
        t, fi, half, *v = PACK.unpack(data)
        h = np.array([v[0] + 1j * v[1], v[2] + 1j * v[3], v[4] + 1j * v[5]], np.complex64)
        pipe.push(fi, half, t, h)
        stats["csi"] += 1
        stats["last"] = time.time()


def beam_follow(pipe: LivePipeline, beam_rx, stop):
    """[GC 1124] beam_tracker 가 보낸 마지막 각도를 파이프라인에 넣는다 (각도 입력 GRU 가 CSI 틱마다 기록)."""
    while not stop.is_set():
        snap = beam_rx.snapshot()
        if snap:
            deg = snap[-1].get("deg") or []
            if len(deg) == 3 and all(d is not None for d in deg):
                pipe.beam_now = np.asarray(deg, np.float32)
        time.sleep(0.02)


def worker_loop(pipe: LivePipeline, stop):
    """Holds the pipeline lock for one pass so a stream reset cannot swap arrays mid-prediction.
    An exception is printed and the loop continues (a silent dead worker would freeze the dashboard)."""
    while not stop.is_set():
        try:
            with pipe.lock:
                pipe.process()
        except Exception as e:                             # noqa: BLE001
            print(f"[WORK] error {e!r} -> reset")
            with pipe.lock:
                pipe.reset()
        time.sleep(0.004)


def pose_start_frame(poses):
    """pose JSON list -> (xs, ys) in the motion-start frame."""
    xs, ys = [], []
    for p in poses:
        if "sx" in p:
            x, y = geo.odom_to_start(p["x"], p["y"], p["sx"], p["sy"], p["syaw"])
        else:
            x, y = p["x"], p["y"]
        xs.append(x)
        ys.append(y)
    return xs, ys


def run_gui(a, pipe: LivePipeline, pose_rx, beam_rx, cfg, stats, stop):
    import matplotlib
    if a.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title("D-MIMO 3-RU live — prediction · scheduling · beam") if hasattr(fig.canvas, "manager") and fig.canvas.manager else None
    gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 2.2, 1.6], height_ratios=[1.3, 1.0, 0.8], hspace=0.38, wspace=0.28)
    ax_mag = fig.add_subplot(gs[0, :2])
    ax_ph = fig.add_subplot(gs[1, :2])
    ax_sch = fig.add_subplot(gs[2, :2])
    ax_map = fig.add_subplot(gs[0:2, 2])
    ax_txt = fig.add_subplot(gs[2, 2])
    ax_txt.axis("off")
    cols = ["#ff9f43", "#3ec5ff", "#7bed9f"]
    kshow = pipe.K - 1
    horizon_ms = (kshow + 1) * pipe.taps * hc.DT * 1e3
    l_mag = [ax_mag.plot([], [], color=c, lw=1.2, label=f"RU{i+1}")[0] for i, c in enumerate(cols)]
    l_pmag = [ax_mag.plot([], [], color=c, lw=1.0, ls="--", alpha=0.8)[0] for c in cols]
    l_ph = [ax_ph.plot([], [], color=cols[i], lw=1.0, label=f"RU{i+1}-RU1")[0] for i in (1, 2)]
    l_pph = [ax_ph.plot([], [], color=cols[i], lw=0.9, ls="--", alpha=0.8)[0] for i in (1, 2)]
    l_sel, = ax_sch.step([], [], where="post", color="#ffd166", lw=1.6, label="DQN")
    l_orc, = ax_sch.step([], [], where="post", color="#aaaaaa", lw=0.9, ls=":", label="best (oracle)")
    ax_mag.set_ylabel("|H| [dB]")
    ax_mag.set_title(f"RU channel magnitude — solid: measured · dashed: GRU +{horizon_ms:.0f} ms prediction", fontsize=10)
    ax_mag.legend(loc="upper left", ncol=3, fontsize=8)
    ax_ph.set_ylabel("relative phase [deg]")
    ax_ph.set_ylim(-180, 180)
    ax_ph.legend(loc="upper left", ncol=2, fontsize=8)
    ax_sch.set_yticks(range(6))
    ax_sch.set_yticklabels(hc.STATE_NAMES, fontsize=8)
    ax_sch.set_ylim(-0.5, 5.5)
    ax_sch.set_xlabel("time [s]")
    ax_sch.legend(loc="upper left", ncol=2, fontsize=8)
    for ax in (ax_mag, ax_ph, ax_sch):
        ax.grid(alpha=0.2)

    rus = cfg["geometry"]["rus"]
    ax_map.set_title("UE position & beams (start frame)", fontsize=10)
    ax_map.set_aspect("equal")
    allx = [r["x"] for r in rus] + [0.0]
    ally = [r["y"] for r in rus] + [0.0]
    ax_map.set_xlim(min(allx) - 0.8, max(allx) + 0.8)
    ax_map.set_ylim(min(ally) - 0.8, max(ally) + 0.8)
    ax_map.add_patch(plt.Circle((0, 0), 0.5, fill=False, ls=":", color="#888888"))
    ru_txt = []
    for i, r in enumerate(rus):
        ax_map.plot(r["x"], r["y"], "s", color=cols[i], ms=10)
        ru_txt.append(ax_map.text(r["x"], r["y"] + 0.18, f"RU{r['ru']}", color=cols[i], ha="center", fontsize=9))
    beam_lines = [ax_map.plot([], [], color=cols[i], lw=2.0, alpha=0.8)[0] for i in range(3)]
    trail, = ax_map.plot([], [], color="#dddddd", lw=1.0, alpha=0.7)
    dot, = ax_map.plot([], [], "o", color="#ffffff", ms=8)
    rx_line, = ax_map.plot([], [], color="#ffd400", lw=2.0, ls="--", alpha=0.9)     # [FR3] UE 4×4 UPA 수신 빔
    txt = ax_txt.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=8.5)

    def db(v):
        return 20 * np.log10(np.abs(v) + 1e-12)

    def update(_):
        with pipe.lock:
            n = pipe.n
            w = int(a.window_sec / hc.DT)
            lo = max(0, n - w)
            x = pipe.x[lo:n].copy()
            pa = pipe.pred_at[kshow, lo:n].copy()
            serve = list(pipe.serve)
            rates = list(pipe.state_rate)
        if n == 0:
            txt.set_text(f"waiting for CSI on UDP {a.udp} ...")
            return []
        t = np.arange(lo, n) * hc.DT
        for i in range(3):
            l_mag[i].set_data(t, db(x[:, i]))
            l_pmag[i].set_data(t, db(pa[:, i]))
        rel = [np.angle(x[:, i] * np.conj(x[:, 0]), deg=True) for i in (1, 2)]
        prel = [np.angle(pa[:, i] * np.conj(pa[:, 0]), deg=True) for i in (1, 2)]
        for j in range(2):
            l_ph[j].set_data(t, rel[j])
            l_pph[j].set_data(t, prel[j])
        fin = np.isfinite(db(x)).all(1)
        if fin.any():
            yy = db(x[fin])
            ax_mag.set_ylim(np.nanmin(yy) - 3, np.nanmax(yy) + 3)
        for ax in (ax_mag, ax_ph, ax_sch):
            ax.set_xlim(max(0, t[-1] - a.window_sec), max(a.window_sec, t[-1]))
        seg_t = np.arange(len(rates)) * pipe.seg * hc.DT
        sel = serve[:len(rates)]
        if len(sel):
            l_sel.set_data(seg_t[:len(sel)], sel)
            l_orc.set_data(seg_t, [int(np.argmax(r)) for r in rates])

        poses = pose_rx.snapshot() if pose_rx else []
        if poses:
            xs, ys = pose_start_frame(poses[-600:])
            trail.set_data(xs, ys)
            dot.set_data([xs[-1]], [ys[-1]])
        beams = beam_rx.snapshot() if beam_rx else []
        bmode = "-"
        if beams:
            b = beams[-1]
            bmode = b.get("mode", "-") + (" (dry)" if b.get("dry") else "")
            for i, r in enumerate(rus):
                d = b["deg"][i]
                if d is None:
                    continue
                ang = math.radians(r["boresight_deg"] + d)
                beam_lines[i].set_data([r["x"], r["x"] + 1.2 * math.cos(ang)], [r["y"], r["y"] + 1.2 * math.sin(ang)])
                ru_txt[i].set_text(f"RU{r['ru']} {d:+.0f}°")
            if b.get("rx_deg") is not None and poses and np.isfinite(b.get("yaw_s", float("nan"))):
                ang = math.radians(b["yaw_s"] + b.get("rx_mount_deg", 0.0) + b["rx_deg"])
                x0, y0 = xs[-1], ys[-1]
                rx_line.set_data([x0, x0 + 0.8 * math.cos(ang)], [y0, y0 + 0.8 * math.sin(ang)])
                bmode += f" · UE {b['rx_deg']:+.0f}°"

        nm = pipe.nmse_db(int(a.stat_sec / hc.DT))
        rs = pipe.rate_summary(int(a.stat_sec / (pipe.seg * hc.DT)))
        tproc = np.mean(pipe.t_proc) if pipe.t_proc else float("nan")
        cur = hc.STATE_NAMES[serve[-1]] if serve else "-"
        lines = [
            f"serving now      : {cur}",
            f"NMSE +{horizon_ms:.0f} ms [dB] GRU {nm['GRU'][kshow]:6.1f}  LR {nm['LR'][kshow]:6.1f}  KF {nm['KF'][kshow]:6.1f}",
            f"NMSE +25 ms  [dB] GRU {nm['GRU'][0]:6.1f}  LR {nm['LR'][0]:6.1f}  KF {nm['KF'][0]:6.1f}",
        ]
        if rs:
            lines += [f"rate [b/s/Hz]    DQN {rs['DQN']:.2f} · oracle {rs['Oracle']:.2f} · random {rs['Random']:.2f}",
                      f"best-pick {100*rs['best_ratio']:.0f} % · handover {100*rs['handover']:.0f} %"]
        lines += [f"beam: {bmode}",
                  f"CSI pkts {stats['csi']} · gaps {pipe.gaps} · restarts {pipe.restarts} · skipped {pipe.skipped}",
                  f"GRU {tproc:.1f} ms / 25 ms · scale {pipe.scale if pipe.scale else 0:.2f}"]
        txt.set_text("\n".join(lines))
        return []

    if a.snapshot:
        time.sleep(a.snapshot_sec)
        update(0)
        fig.savefig(a.snapshot, dpi=110, facecolor=fig.get_facecolor())
        print(f"[SNAP] {a.snapshot}")
        stop.set()
        return
    _anim = FuncAnimation(fig, update, interval=1000 / a.fps, cache_frame_data=False)  # noqa: F841
    plt.show()
    stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--udp", default="0.0.0.0:5005")
    ap.add_argument("--pose-udp", default="0.0.0.0:5006")
    ap.add_argument("--beam-udp", default="0.0.0.0:5008")
    ap.add_argument("--config", default=str(geo.DEFAULT_CONFIG))
    ap.add_argument("--predictor-model", default=str(CODE_ROOT / "models" / "best_validation_nmse.pt"))
    ap.add_argument("--dqn-model", default=str(CODE_ROOT / "models" / "final_best_model.pt"))
    ap.add_argument("--auto-scale", type=float, default=2.0, help="seconds of CSI used to set the scale")
    ap.add_argument("--scale", type=float, default=0.0, help="fixed scale (overrides --auto-scale)")
    ap.add_argument("--context-steps", type=int, default=40)
    ap.add_argument("--min-context-steps", type=int, default=8)
    ap.add_argument("--max-lag-steps", type=int, default=2)
    ap.add_argument("--init-state", type=int, default=0)
    ap.add_argument("--window-sec", type=float, default=6.0)
    ap.add_argument("--stat-sec", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--headless", type=float, default=0.0, help="no GUI: run N seconds and print metrics")
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--snapshot-sec", type=float, default=20.0)
    ap.add_argument("--save-dir", default=str(CODE_ROOT / "logs"))
    a = ap.parse_args()

    pipe = LivePipeline(a, hc.NumpyPredictor(a.predictor_model), hc.NumpyDQN(a.dqn_model))
    cfg = geo.load_config(a.config)
    stop = threading.Event()
    stats = {"csi": 0, "last": None}
    h, p = a.udp.rsplit(":", 1)
    threading.Thread(target=csi_loop, args=(pipe, (h, int(p)), stop, stats), daemon=True).start()
    threading.Thread(target=worker_loop, args=(pipe, stop), daemon=True).start()
    pose_rx = beam_rx = None
    for spec, name in ((a.pose_udp, "pose"), (a.beam_udp, "beam")):
        if not spec:
            continue
        hh, pp = spec.rsplit(":", 1)
        rx = JsonRx((hh, int(pp)))
        threading.Thread(target=rx.loop, args=(stop,), daemon=True).start()
        if name == "pose":
            pose_rx = rx
        else:
            beam_rx = rx
            threading.Thread(target=beam_follow, args=(pipe, rx, stop), daemon=True).start()
    if pipe.pred.beam_dim:
        print(f"[LIVE] 각도 입력 GRU (beam_feat_dim {pipe.pred.beam_dim}) — 빔 상태 UDP {a.beam_udp} 가 없으면 0° 로 본다")
    print(f"[LIVE] CSI {a.udp} · pose {a.pose_udp} · beam {a.beam_udp} · GRU taps {pipe.taps} K {pipe.K} · "
          f"DQN seg {pipe.seg} ticks ({pipe.dqn.timing})")

    try:
        if a.headless > 0:
            time.sleep(a.headless)
            nm = pipe.nmse_db(int(a.stat_sec / hc.DT))
            rs = pipe.rate_summary(int(a.stat_sec / (pipe.seg * hc.DT)))
            print(json.dumps({"ticks": pipe.n, "csi_pkts": stats["csi"], "gaps": pipe.gaps, "restarts": pipe.restarts,
                              "skipped": pipe.skipped, "gru_ms": float(np.mean(pipe.t_proc)) if pipe.t_proc else None,
                              "scale": pipe.scale, "nmse_db": nm, "rates": rs,
                              "pose_pkts": pose_rx.count if pose_rx else 0, "beam_pkts": beam_rx.count if beam_rx else 0},
                             indent=1, default=lambda v: None if (isinstance(v, float) and not math.isfinite(v)) else v))
        else:
            run_gui(a, pipe, pose_rx, beam_rx, cfg, stats, stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if pipe.n:
            d = Path(a.save_dir) / f"dashboard_{datetime.now():%Y%m%d_%H%M%S}"
            pipe.save(d)
            print(f"[SAVE] {d}")


if __name__ == "__main__":
    main()
