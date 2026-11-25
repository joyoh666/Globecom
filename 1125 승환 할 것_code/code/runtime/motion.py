# -*- coding: utf-8 -*-
"""
[GC 1117] Globecom 데모용 로봇 이동 — 세 가지 모드를 같은 제어기로 돌린다.

  MOTION=scenario  미리 정해진 75개 경로 중 하나(SCENARIO_ID)를 DURATION 초 동안 계속 돈다
                   (경유점 목록을 끝까지 가면 처음으로 되돌아 반복).
  MOTION=random    출발점 반경 안에서 경유점을 그때그때 무작위로 뽑아 DURATION 초 동안 돈다.
                   속도·다리 길이·회전은 학습 데이터(75개 경로)와 같은 범위 → GRU 입력 분포 유지.
  MOTION=facing    [GC 1118 FR3] 로봇이 늘 RU 쪽을 바라본 채 전진·후진으로 반경 안을 오간다 (MOTION_SEED 로 재현, 기본 72).
                   로봇 위 4×4 수신 배열이 RU 를 늘 앞에 둔다 (scenario 경로는 사방을 봐서 RU 가 ±40° 안에 드는 시간이 14~28 %).
                   방향 = config 의 RU 셋 가운데 방향(출발 자세 기준) ± FACE_DEV_DEG(기본 25°). 후진 다리는 최대 0.56 m.
  MOTION=manual    로봇을 움직이지 않는다. 다른 터미널에서 사람이 키보드로 몬다
                   (scripts/run_turtlebot4_teleop.sh). 위치 기록·송출만 한다.

제어기·안전 반경은 turtlebotscenario15s75episodes.py 의 것을 그대로 쓴다 (전진만, 0.47 m 에서 안쪽으로, 0.50 m 에서 중단).
저장 파일 이름과 열은 run_scenario() 와 같다 → RX 스크립트의 후처리를 그대로 쓴다.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import turtlebotscenario15s75episodes as ts

MODES = ("scenario", "random", "manual", "facing")
CONFIG = Path(__file__).resolve().parents[1] / "config" / "demo_config.json"


def face_dir_deg() -> float:
    """출발 자세 기준 RU 셋의 가운데 방향 [deg]. FACE_DEG 환경변수가 있으면 그 값."""
    import os
    if os.environ.get("FACE_DEG", "").strip():
        return float(os.environ["FACE_DEG"])
    try:
        rus = json.loads(CONFIG.read_text(encoding="utf-8"))["geometry"]["rus"]
        cx = sum(math.cos(math.atan2(r["y"], r["x"])) for r in rus)
        cy = sum(math.sin(math.atan2(r["y"], r["x"])) for r in rus)
        return math.degrees(math.atan2(cy, cx))
    except Exception:                                      # noqa: BLE001
        return 0.0


class FacingWaypoints:
    """RU 쪽(face)을 바라본 채 다니는 경유점. 다리 방향이 face±dev(전진) 또는 face+180±dev(후진) 안에만 있다."""

    def __init__(self, seed: Optional[int], face_deg: float, dev_deg: float = 25.0,
                 r_max=ts.WAYPOINT_RADIUS_MAX_M - 0.03, leg_min=0.30, leg_max=0.56):
        self.rng = np.random.default_rng(72 if seed is None else seed)
        self.face, self.dev = math.radians(face_deg), math.radians(dev_deg)
        self.r_max, self.leg_min, self.leg_max = r_max, leg_min, leg_max
        self.prev = (0.0, 0.0)
        self.fwd = False                                   # 다음 다리는 번갈아 (전진 → 후진 → …) 를 기본으로

    def next(self) -> ts.Waypoint:
        px, py = self.prev
        self.fwd = not self.fwd
        for attempt in range(800):
            fwd = self.fwd if attempt < 400 else bool(self.rng.integers(2))
            d = self.face + (0.0 if fwd else math.pi) + float(self.rng.uniform(-self.dev, self.dev))
            leg = float(self.rng.uniform(self.leg_min, self.leg_max))
            cx, cy = px + leg * math.cos(d), py + leg * math.sin(d)
            if math.hypot(cx, cy) <= self.r_max:
                break
        else:                                              # 막히면 원점으로 (방향이 콘 밖일 수 있음)
            cx, cy, fwd = 0.0, 0.0, True
        self.prev = (cx, cy)
        wp = ts.Waypoint(cx, cy, float(self.rng.uniform(ts.V_CRUISE_MIN, ts.V_CRUISE_MAX)))
        return wp


def _publish(node, v: float, w: float) -> None:
    """후진 허용 속도 명령 (ts.publish_velocity 는 후진을 막는다). facing 모드에서만 쓴다."""
    msg = ts.TwistStamped()
    msg.header.frame_id = "base_link"
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.twist.linear.x = float(v)
    msg.twist.angular.z = float(w)
    node.pub.publish(msg)


def facing_controller(pose, target, face_rad: float, cruise_v: float, align_mode: bool):
    """다리 방향이 face 쪽이면 전진, 반대면 후진 — 로봇 머리는 늘 face 근처. 돌리기·몰기는 ts 제어기와 같은 상수."""
    x, y, yaw_deg = pose
    dx, dy = target[0] - x, target[1] - y
    dist = math.hypot(dx, dy)
    if dist <= ts.WAYPOINT_TOL_M:
        return 0.0, 0.0, dist, 0.0, False, "REACHED"
    leg = math.atan2(dy, dx)
    fwd = abs(ts.wrap_pi(leg - face_rad)) <= math.pi / 2
    want = leg if fwd else ts.wrap_pi(leg + math.pi)       # 후진이면 머리는 다리 반대 방향
    err = ts.wrap_pi(want - math.radians(yaw_deg))
    if align_mode:
        if abs(err) <= math.radians(ts.ALIGN_EXIT_DEG):
            align_mode = False
    elif abs(err) >= math.radians(ts.ALIGN_ENTER_DEG):
        align_mode = True
    if align_mode:
        return 0.0, math.copysign(ts.W_ALIGN, err), dist, err, True, "ALIGN"
    v = max(ts.V_APPROACH_MIN, cruise_v * max(0.58, math.cos(err)) * min(1.0, max(0.45, dist / 0.22)))
    w = max(-ts.W_DRIVE_MAX, min(ts.W_DRIVE_MAX, ts.K_W_DRIVE * err))
    return (v if fwd else -v), w, dist, err, False, ("DRIVE" if fwd else "REVERSE")


class RandomWaypoints:
    """반경 안 무작위 경유점. 학습 경로 생성기와 같은 범위의 다리 길이·회전·속도."""

    def __init__(self, seed: Optional[int], r_min=ts.WAYPOINT_RADIUS_MIN_M, r_max=ts.WAYPOINT_RADIUS_MAX_M,
                 leg_min=0.30, leg_max=0.56, max_turn_deg=120.0):
        self.rng = np.random.default_rng(seed)
        self.r_min, self.r_max = r_min, r_max
        self.leg_min, self.leg_max = leg_min, leg_max
        self.max_turn = math.radians(max_turn_deg)
        self.prev = (0.0, 0.0)
        self.heading = 0.0

    def next(self) -> ts.Waypoint:
        px, py = self.prev
        for _ in range(500):
            leg = float(self.rng.uniform(self.leg_min, self.leg_max))
            turn = float(self.rng.uniform(-self.max_turn, self.max_turn))
            hd = self.heading + turn
            cx, cy = px + leg * math.cos(hd), py + leg * math.sin(hd)
            if math.hypot(cx, cy) <= self.r_max:
                break
        else:                                              # 막히면 원점 쪽으로
            hd = math.atan2(-py, -px)
            leg = min(self.leg_max, math.hypot(px, py))
            cx, cy = px + leg * math.cos(hd), py + leg * math.sin(hd)
        self.prev, self.heading = (cx, cy), hd
        speed = float(self.rng.uniform(ts.V_CRUISE_MIN, ts.V_CRUISE_MAX))
        return ts.Waypoint(cx, cy, speed)


def run_motion(node, mode: str, scenario_id: int, duration: float, out_dir: Path,
               countdown: float = 2.0, start_callback=None, rate_hz: float = ts.CONTROL_HZ,
               seed: Optional[int] = None, on_start_pose=None) -> dict:
    if mode not in MODES:
        raise ValueError(f"MOTION must be one of {MODES}, got {mode!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import os
    route = ts.SCENARIOS[scenario_id].waypoints if mode == "scenario" else None
    face_deg = face_dir_deg()
    face_dev = float(os.environ.get("FACE_DEV_DEG", "25"))
    rnd = (RandomWaypoints(seed) if mode == "random"
           else FacingWaypoints(seed, face_deg, face_dev) if mode == "facing" else None)
    definition = {"mode": mode, "duration_s": duration, "seed": seed,
                  "face_deg": face_deg if mode == "facing" else None, "face_dev_deg": face_dev if mode == "facing" else None,
                  "scenario": ts.scenario_definition_dict(ts.SCENARIOS[scenario_id]) if mode == "scenario" else None}
    (out_dir / "scenario_definition.json").write_text(json.dumps(definition, indent=2, default=str), encoding="utf-8")

    start_pose = ts.wait_for_pose(node)
    sx, sy, syaw = start_pose
    if on_start_pose is not None:
        on_start_pose(sx, sy, syaw)
    print(f"[READY] MOTION={mode} · {duration:.1f} s · start=({sx:.3f}, {sy:.3f}) yaw={syaw:.1f} deg")

    if countdown > 0:
        deadline = time.monotonic() + countdown
        while time.monotonic() < deadline:
            if mode != "manual":
                ts.publish_velocity(node, 0.0, 0.0)
            time.sleep(0.05)

    t0_wall, t0_mono = time.time(), time.monotonic()
    callback_data = start_callback(t0_wall, t0_mono) if start_callback else None
    marker = {"scenario_id": scenario_id, "motion": mode, "t0_wall": t0_wall, "t0_monotonic": t0_mono,
              "duration_s": duration, "pose_frame": node.pose_frame,
              "start_pose": {"frame_id": node.pose_frame, "x": sx, "y": sy, "yaw_deg": syaw},
              "callback_data": callback_data}
    (out_dir / "scenario_event.json").write_text(json.dumps(marker, indent=2, default=str), encoding="utf-8")

    rows, generated = [], []
    period = 1.0 / float(rate_hz)
    next_tick = t0_mono
    wp_idx, reached, align_mode = 0, 0, True
    aborted, abort_reason = False, None

    def target(idx):
        if mode == "scenario":
            return route[idx % len(route)]
        while len(generated) <= idx:
            generated.append(rnd.next())
        return generated[idx]

    try:
        while True:
            now = time.monotonic()
            t_rel = now - t0_mono
            if t_rel >= duration:
                break
            pose = node.get_pose()
            if pose is None:
                if mode != "manual":
                    ts.publish_velocity(node, 0.0, 0.0)
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
                continue
            x, y, yaw_deg = pose
            radius = math.hypot(x - sx, y - sy)
            v_cmd = w_cmd = 0.0
            mode_used, dist, herr = "MANUAL", float("nan"), float("nan")
            wp = None
            tgt = (float("nan"), float("nan"))

            if mode != "manual":
                if radius >= ts.HARD_RADIUS_M:
                    aborted, abort_reason = True, f"hard_radius_exceeded:{radius:.3f}m"
                    print(f"[SAFETY] Hard radius exceeded ({radius:.3f} m). Aborting.")
                    break
                wp = target(wp_idx)
                tgt = ts.local_to_pose_frame(wp.x, wp.y, sx, sy, syaw)
                if mode == "facing":
                    face_rad = math.radians(syaw + face_deg)
                    if radius >= ts.SOFT_RADIUS_M:
                        tgt = (sx, sy)
                        v_cmd, w_cmd, dist, herr, align_mode, inner = facing_controller(pose, tgt, face_rad, 0.10, align_mode)
                        mode_used = "RECOVER_" + inner
                    else:
                        v_cmd, w_cmd, dist, herr, align_mode, mode_used = facing_controller(pose, tgt, face_rad, wp.speed, align_mode)
                        if mode_used == "REACHED":
                            reached += 1
                            wp_idx += 1
                            wp = target(wp_idx)
                            tgt = ts.local_to_pose_frame(wp.x, wp.y, sx, sy, syaw)
                            v_cmd, w_cmd, dist, herr, align_mode, mode_used = facing_controller(
                                pose, tgt, face_rad, wp.speed, True)
                    _publish(node, v_cmd, w_cmd)
                elif radius >= ts.SOFT_RADIUS_M:
                    tgt = (sx, sy)
                    v_cmd, w_cmd, dist, herr, align_mode, inner = ts._forward_drive_controller(
                        pose, tgt, cruise_v=0.10, align_mode=align_mode)
                    mode_used = "RECOVER_" + inner
                else:
                    v_cmd, w_cmd, dist, herr, align_mode, inner = ts._forward_drive_controller(
                        pose, tgt, cruise_v=wp.speed, align_mode=align_mode)
                    mode_used = inner
                    if inner == "REACHED":
                        reached += 1
                        wp_idx += 1
                        wp = target(wp_idx)
                        tgt = ts.local_to_pose_frame(wp.x, wp.y, sx, sy, syaw)
                        v_cmd, w_cmd, dist, herr, align_mode, mode_used = ts._forward_drive_controller(
                            pose, tgt, cruise_v=wp.speed, align_mode=True)
                if mode != "facing":
                    ts.publish_velocity(node, v_cmd, w_cmd)

            rows.append({
                "t_rel": t_rel, "t_wall": time.time(), "waypoint_idx": wp_idx, "mode_used": mode_used,
                "x": x, "y": y, "yaw_deg": yaw_deg, "radius_from_start_m": radius, "pose_frame": node.pose_frame,
                "target_frame_x": tgt[0], "target_frame_y": tgt[1], "target_map_x": tgt[0], "target_map_y": tgt[1],
                "target_local_x": wp.x if wp else float("nan"), "target_local_y": wp.y if wp else float("nan"),
                "distance_to_target_m": dist, "heading_error_rad": herr,
                "cmd_linear_x": v_cmd if mode == "facing" else max(0.0, v_cmd), "cmd_angular_z": w_cmd,
            })
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        aborted, abort_reason = True, "keyboard_interrupt"
    finally:
        if mode != "manual":
            ts.stop_robot(node)

    csv_path = out_dir / "turtle_trajectory.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if generated:
        (out_dir / ("facing_waypoints.json" if mode == "facing" else "random_waypoints.json")).write_text(
            json.dumps([{"x": g.x, "y": g.y, "speed": g.speed} for g in generated], indent=2), encoding="utf-8")
    summary = {**marker, "t_end_wall": time.time(), "actual_elapsed_s": time.monotonic() - t0_mono,
               "aborted": aborted, "abort_reason": abort_reason, "num_logged_points": len(rows),
               "waypoints_reached": reached,
               "max_radius_actual_m": max((r["radius_from_start_m"] for r in rows), default=float("nan")),
               "trajectory_csv": str(csv_path)}
    (out_dir / "scenario_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[DONE] MOTION={mode} elapsed={summary['actual_elapsed_s']:.1f} s reached={reached} aborted={aborted}")
    return summary
