# -*- coding: utf-8 -*-
"""
[GC 1124] 터틀봇(Create3) odometry 위치가 얼마나 맞는지 줄자로 잰다 → 빔 각도 오차로 바꿔 본다.

로봇 쪽 PC (ROS2 · turtlebot4 bringup 이 떠 있는 곳) 에서, code/ 폴더에서:

  A. 점 찍기 (manual) — 로봇을 손/텔레옵으로 표시한 점들에 옮기며 그때마다 Enter
     python -m tools.odom_check manual
     python -m tools.odom_check manual --points "1,0 1,1 0,1 0,0"     # 줄자 점을 미리 정해 두면 입력 생략

  B. 데모 이동 그대로 (facing) — DURATION 초 데모 이동 후 멈춘 자리를 줄자로 재서 입력
     python -m tools.odom_check facing --duration 60 --seed 72

  C. 저장한 결과 다시 계산 (RU 자리를 바꿨을 때 · 로봇 없이)
     python -m tools.odom_check analyze logs/odom_check_YYYYmmdd_HHMMSS.json [--config ...]

좌표 = beam_tracker 와 같은 '시작 자세' 좌표: 출발 자리 원점, 로봇이 처음 바라본 방향 +x, 왼쪽 +y (m).
  → 바닥에 출발 자리(로봇 가운데)와 +x 방향을 테이프로 표시하고 시작한다.
줄자 값은 로봇 가운데(바퀴 축 가운데) 기준으로 잰다.

결과: 점마다 위치 오차 [cm] · 방향 오차 [deg] · RU 1·2·3 빔 각도 오차 [deg] (config 의 RU 자리 기준)
      + 판정: 빔폭 반폭(기본 12.5°) 대비 → logs/odom_check_*.json · *.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from beam import geometry as geo  # noqa: E402


# ---------------------------------------------------------------- 계산 (로봇 없이도 쓰는 부분)
def evaluate(points: list[dict], cfg: dict, half_beam_deg: float) -> dict:
    """points: [{name, odom_x, odom_y, odom_yaw, true_x, true_y, true_yaw?}] (시작 자세 좌표) → 오차 표와 요약."""
    rus = cfg["geometry"]["rus"]
    rows = []
    for p in points:
        ex, ey = p["odom_x"] - p["true_x"], p["odom_y"] - p["true_y"]
        r = {"name": p["name"], "true_x": p["true_x"], "true_y": p["true_y"],
             "odom_x": round(p["odom_x"], 3), "odom_y": round(p["odom_y"], 3),
             "pos_err_cm": round(100 * math.hypot(ex, ey), 1),
             "yaw_err_deg": (round(geo.wrap_deg(p["odom_yaw"] - p["true_yaw"]), 1)
                             if p.get("true_yaw") is not None else None)}
        for ru in rus:
            d = geo.wrap_deg(geo.steer_deg(ru, p["odom_x"], p["odom_y"]) - geo.steer_deg(ru, p["true_x"], p["true_y"]))
            r[f"ru{ru['ru']}_beam_err_deg"] = round(d, 2)
        rows.append(r)
    beam_errs = [abs(r[f"ru{ru['ru']}_beam_err_deg"]) for r in rows for ru in rus]
    pos = [r["pos_err_cm"] for r in rows]
    yaws = [abs(r["yaw_err_deg"]) for r in rows if r["yaw_err_deg"] is not None]
    worst = max(beam_errs) if beam_errs else float("nan")
    summary = {"n_points": len(rows), "pos_err_cm_max": max(pos) if pos else None,
               "pos_err_cm_mean": round(sum(pos) / len(pos), 1) if pos else None,
               "yaw_err_deg_max": max(yaws) if yaws else None,
               "beam_err_deg_max": round(worst, 2), "half_beam_deg": half_beam_deg,
               "verdict": ("OK — 빔 반폭의 1/3 안" if worst <= half_beam_deg / 3 else
                           "주의 — 반폭 안이지만 1/3 넘음" if worst <= half_beam_deg else
                           "문제 — 빔 반폭을 넘는다 (위치 보정 필요)")}
    return {"rows": rows, "summary": summary}


def print_report(res: dict):
    rows, s = res["rows"], res["summary"]
    if not rows:
        print("점이 없다")
        return
    keys = [k for k in rows[0] if k.endswith("_beam_err_deg")]
    print("\n점        줄자(x,y)       odom(x,y)        위치오차  방향오차  " + "  ".join(k[:3].upper() for k in keys))
    for r in rows:
        yaw = "   -  " if r["yaw_err_deg"] is None else f"{r['yaw_err_deg']:+6.1f}°"
        print(f"{r['name']:<8} ({r['true_x']:+.2f},{r['true_y']:+.2f})  ({r['odom_x']:+.3f},{r['odom_y']:+.3f})  "
              f"{r['pos_err_cm']:6.1f} cm  {yaw}  " + "  ".join(f"{r[k]:+5.1f}°" for k in keys))
    print(f"\n최대 위치 오차 {s['pos_err_cm_max']} cm (평균 {s['pos_err_cm_mean']}) · 최대 방향 오차 {s['yaw_err_deg_max']}° · "
          f"최대 빔 각도 오차 {s['beam_err_deg_max']}° (반폭 {s['half_beam_deg']}°)\n→ {s['verdict']}")


def save(res: dict, meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"odom_check_{datetime.now():%Y%m%d_%H%M%S}"
    stem.with_suffix(".json").write_text(json.dumps({**meta, **res}, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res["rows"][0]))
        w.writeheader()
        w.writerows(res["rows"])
    print(f"[SAVE] {stem}.json · .csv")
    return stem


# ---------------------------------------------------------------- 로봇 쪽
class OdomLogger:
    """odom → base_link 를 20 Hz 로 계속 기록 (시작 자세 좌표로도 바꿔 둔다)."""

    def __init__(self, node, hz=20.0):
        self.node, self.dt = node, 1.0 / hz
        self.rows, self.start = [], None
        self.stop_ev = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop_ev.is_set():
            p = self.node.get_pose()
            if p is not None:
                self.rows.append((time.time(), *p))
            time.sleep(self.dt)

    def now(self, avg_s=0.5):
        """정지 상태에서 avg_s 초 평균 (x, y, yaw)."""
        time.sleep(avg_s)
        t_end = time.time()
        sel = [r for r in self.rows if r[0] >= t_end - avg_s] or self.rows[-1:]
        x = sum(r[1] for r in sel) / len(sel)
        y = sum(r[2] for r in sel) / len(sel)
        yaw = math.degrees(math.atan2(sum(math.sin(math.radians(r[3])) for r in sel),
                                      sum(math.cos(math.radians(r[3])) for r in sel)))
        return x, y, yaw

    def to_start(self, x, y, yaw):
        sx, sy, syaw = self.start
        px, py = geo.odom_to_start(x, y, sx, sy, syaw)
        return px, py, geo.wrap_deg(yaw - syaw)


def ask_xy(prompt: str, default=None):
    while True:
        s = input(prompt).strip()
        if not s and default is not None:
            return default
        try:
            v = [float(t) for t in s.replace(",", " ").split()]
            if len(v) in (2, 3):
                return v
        except ValueError:
            pass
        print("  'x y' 또는 'x y yaw' (m, m, deg) 로 입력")


def start_robot():
    import rclpy
    from runtime.turtle_control import TB4AdvancedControl
    rclpy.init()
    node = TB4AdvancedControl("odom_check_node", pose_frame="odom")
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    t0 = time.monotonic()
    while node.get_pose() is None:
        if time.monotonic() - t0 > 30:
            raise SystemExit("odom → base_link TF 를 30 s 동안 못 받았다 (bringup · ROS_DOMAIN_ID 확인)")
        time.sleep(0.1)
    return node


def run_manual(a, cfg):
    node = start_robot()
    log = OdomLogger(node)
    preset = [[float(v) for v in p.split(",")] for p in a.points.split()] if a.points else []
    input("로봇을 출발 자리(원점)에 +x 를 보게 놓고 Enter … ")
    log.start = log.now()
    print(f"  시작 자세 odom = ({log.start[0]:.3f}, {log.start[1]:.3f}) yaw {log.start[2]:.1f}°")
    pts, k = [], 0
    while True:
        tgt = preset[k] if k < len(preset) else None
        hint = f" [줄자 점 {tgt[0]:+.2f},{tgt[1]:+.2f}]" if tgt else ""
        s = input(f"\n점 {k + 1}{hint}: 로봇을 옮긴 뒤 Enter (끝내려면 q) … ").strip().lower()
        if s == "q":
            break
        px, py, pyaw = log.to_start(*log.now())
        print(f"  odom (시작 좌표) = ({px:+.3f}, {py:+.3f}) yaw {pyaw:+.1f}°")
        v = ask_xy(f"  줄자 x y [yaw] (m){' — Enter 면 ' + str(tgt) if tgt else ''}: ", tgt)
        pts.append({"name": f"P{k + 1}", "odom_x": px, "odom_y": py, "odom_yaw": pyaw,
                    "true_x": v[0], "true_y": v[1], "true_yaw": v[2] if len(v) == 3 else None})
        print_report(evaluate(pts[-1:], cfg, a.half_beam))
        k += 1
    log.stop_ev.set()
    return pts, log.rows, log.start


def run_facing(a, cfg):
    import os
    from runtime.motion import run_motion
    node = start_robot()
    log = OdomLogger(node)
    input(f"로봇을 출발 자리(원점)에 +x 를 보게 놓고 Enter → {a.duration:.0f} s facing 이동 (seed {a.seed}) … ")
    os.environ.setdefault("FACE_DEV_DEG", str(a.face_dev))
    out = CODE_ROOT / "logs" / f"odom_check_motion_{datetime.now():%Y%m%d_%H%M%S}"
    ev = run_motion(node, "facing", 72, a.duration, out, countdown=2.0, seed=a.seed,
                    on_start_pose=lambda x, y, yaw: setattr(log, "start", (x, y, yaw)))
    node.stop()
    time.sleep(1.0)
    px, py, pyaw = log.to_start(*log.now())
    print(f"\n이동 끝 ({'중단: ' + str(ev.get('abort_reason')) if ev.get('aborted') else '정상'}). "
          f"odom (시작 좌표) = ({px:+.3f}, {py:+.3f}) yaw {pyaw:+.1f}°")
    print("출발 표시에서 로봇 가운데까지 줄자로: x = 출발 방향으로 앞(+)/뒤(−), y = 왼쪽(+)/오른쪽(−). 방향(yaw)은 각도기로 재면 같이.")
    v = ask_xy("  줄자 x y [yaw]: ")
    pts = [{"name": "END", "odom_x": px, "odom_y": py, "odom_yaw": pyaw,
            "true_x": v[0], "true_y": v[1], "true_yaw": v[2] if len(v) == 3 else None}]
    log.stop_ev.set()
    path_len = sum(math.hypot(b[1] - a_[1], b[2] - a_[2]) for a_, b in zip(log.rows, log.rows[1:]))
    print(f"  odom 상 이동 거리 {path_len:.2f} m")
    return pts, log.rows, log.start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["manual", "facing", "analyze"])
    ap.add_argument("file", nargs="?", help="analyze: 저장한 odom_check_*.json")
    ap.add_argument("--config", default=str(geo.DEFAULT_CONFIG))
    ap.add_argument("--points", default="", help='manual: 줄자 점 미리 지정 "x,y x,y …" (m)')
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=72)
    ap.add_argument("--face-dev", type=float, default=25.0)
    ap.add_argument("--half-beam", type=float, default=12.5, help="판정 기준 빔 반폭 [deg] (빔폭 25~30° → 12.5)")
    ap.add_argument("--out", default=str(CODE_ROOT / "logs"))
    a = ap.parse_args()
    cfg = geo.load_config(a.config)

    if a.mode == "analyze":
        if not a.file:
            raise SystemExit("analyze 는 json 경로가 필요하다")
        d = json.loads(Path(a.file).read_text(encoding="utf-8"))
        pts = d["points"]
        res = evaluate(pts, cfg, a.half_beam)
        print_report(res)
        return
    pts, trace, start = (run_manual if a.mode == "manual" else run_facing)(a, cfg)
    if not pts:
        print("점이 없어 저장 안 함")
        return
    res = evaluate(pts, cfg, a.half_beam)
    print_report(res)
    stem = save(res, {"mode": a.mode, "config": a.config, "points": pts, "start_pose_odom": start,
                      "seed": a.seed if a.mode == "facing" else None,
                      "duration_s": a.duration if a.mode == "facing" else None}, Path(a.out))
    with open(f"{stem}_trace.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_os", "odom_x", "odom_y", "odom_yaw_deg"])
        w.writerows(trace)
    print(f"그림: python tools/odom_plot.py {stem}.json")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                        # noqa: BLE001
        pass
    main()
