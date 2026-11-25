# -*- coding: utf-8 -*-
"""
[GC 1117] Globecom 판: 이동은 runtime/motion.py (MOTION=scenario|random|manual|facing, DURATION 초), 위치는 LIVE_STATUS_UDP 여러 곳으로.

TurtleBot control child process used by DMIMO_3TX_Live_Rx.py.

The robot pose comes from Create 3 odometry (odom -> base_link), so neither SLAM
nor Nav2 localization is required. Every run verifies that the robot is on the
powered charger, undocks, executes one scenario, and docks again.

Parent protocol (one line per event):
  @@DOCK_READY <json>       charger connection confirmed
  @@UNDOCKED <json>         undock action and dock sensor confirmed
  @@READY                   odometry and robot are ready; parent may send GO
  @@T0 <wall> <monotonic>   first scenario velocity command marker
  @@DONE <json>             scenario and return-to-charger both succeeded
  @@ERROR <message>         failure; the next batch item must not start
"""
import csv
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import rclpy

from .turtle_control import TB4AdvancedControl
from .motion import run_motion  # [GC 1117] scenario(한 경로 계속) / random(무작위 경유점) / manual(사람이 조종)


def say(msg):
    print(msg, flush=True)


def _result_error(label, result):
    return RuntimeError(f"{label} 실패: {result.get('reason')} (result={result})")


def _shutdown_node(node):
    try:
        node.stop()
    except Exception:  # noqa: BLE001 - best-effort safety stop
        pass
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:  # noqa: BLE001 - process exits immediately afterwards
        pass


def main():
    sid = int(sys.argv[1])
    out = Path(sys.argv[2])
    pre = float(sys.argv[3])
    log_hz = float(os.environ.get("ROBOT_LOG_HZ", "5"))
    lag = float(os.environ.get("POSE_QUERY_LAG_SEC", "0.10"))
    wait_pose_s = float(os.environ.get("WAIT_POSE_TIMEOUT_SEC", "30"))
    dock_status_timeout = float(os.environ.get("DOCK_STATUS_TIMEOUT_SEC", "10"))
    dock_timeout = float(os.environ.get("DOCK_TIMEOUT_SEC", "60"))
    undock_timeout = float(os.environ.get("UNDOCK_TIMEOUT_SEC", "30"))
    allow_dock_failure = os.environ.get("ALLOW_DOCK_FAILURE", "1") == "1"
    out.mkdir(parents=True, exist_ok=True)

    # [GC 1117] 위치를 여러 곳에 보낸다: 대시보드(5006) + 빔 추적기(5007)
    status_udp = os.environ.get("LIVE_STATUS_UDP", "127.0.0.1:5006,127.0.0.1:5007")
    status_addrs = []
    for item in status_udp.split(","):
        if item.strip():
            h_, p_ = item.strip().rsplit(":", 1)
            status_addrs.append((h_, int(p_)))
    status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    motion = os.environ.get("MOTION", "scenario").strip().lower()
    duration = float(os.environ.get("DURATION", "60"))
    seed_text = os.environ.get("MOTION_SEED", "").strip()
    motion_seed = int(seed_text) if seed_text else None
    start_ref = {}

    def emit_status(tp, x, y, yaw_deg, vx, vy, vz, phase="run"):
        payload = {
            "ts": float(tp),
            "x": float(x),
            "y": float(y),
            "yaw_deg": float(yaw_deg),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "phase": phase,
            "sid": sid,
            "motion": motion,
            **start_ref,                                     # sx, sy, syaw: 이동 시작 자세 (빔 추적 좌표 원점)
        }
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        for addr in status_addrs:
            try:
                status_sock.sendto(data, addr)
            except OSError:
                pass

    rclpy.init()
    node = TB4AdvancedControl("robot_proc_node", pose_frame="odom")
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    t = time.monotonic()
    while node.get_pose() is None:
        if time.monotonic() - t > wait_pose_s:
            say(
                "@@ERROR odom -> base_link TF를 못 받았습니다. "
                "turtlebot4_bringup/ROS_DOMAIN_ID/네트워크를 확인하세요."
            )
            _shutdown_node(node)
            return 2
        time.sleep(0.1)

    initial_dock = {"success": True, "action": "skip_dock", "reason": None, "already_docked": True, "elapsed_s": 0.0, "dock_status": {"is_docked": True, "dock_visible": False, "received_at_monotonic": time.monotonic()}}
    say("@@DOCK_READY " + json.dumps(initial_dock, default=str))

    undock_result = {"success": True, "action": "skip_undock", "reason": None, "already_docked": True, "elapsed_s": 0.0, "dock_status": {"is_docked": False, "dock_visible": False, "received_at_monotonic": time.monotonic()}}
    say("@@UNDOCKED " + json.dumps(undock_result, default=str))
    say("@@READY")

    if sys.stdin.readline().strip() != "GO":
        say(f"@@ERROR 부모가 GO 를 안 보냈다 (skip_dock_mode={True})")
        _shutdown_node(node)
        return 3

    rows = []
    stop = threading.Event()

    def logger():
        dt, prev = 1.0 / log_hz, None
        while not stop.is_set():
            tp = time.time() - lag
            try:
                p = node.get_pose(tp)
            except TypeError:
                p = None
            if p is None:
                p = node.get_pose()
            if p is None:
                time.sleep(dt)
                continue
            x, y, z = p
            if prev is None:
                vx = vy = vz = 0.0
            else:
                t0, x0, y0, z0 = prev
                d = max(1e-6, tp - t0)
                vx, vy, vz = (x - x0) / d, (y - y0) / d, (z - z0) / d
            prev = (tp, x, y, z)
            rows.append((tp, x, y, z, vx, vy, vz))
            emit_status(tp, x, y, z, vx, vy, vz, phase="run")
            time.sleep(dt)

    lt = threading.Thread(target=logger, daemon=True)
    lt.start()

    def cb(t0_wall, t0_mono):
        say(f"@@T0 {t0_wall!r} {t0_mono!r}")
        return {"t0_wall": t0_wall, "t0_monotonic": t0_mono}

    ev = None
    run_error = None
    try:
        ev = run_motion(
            node,
            motion,
            sid,
            duration,
            out,
            countdown=pre,
            start_callback=cb,
            rate_hz=20.0,
            seed=motion_seed,
            on_start_pose=lambda x_, y_, yaw_: start_ref.update({"sx": x_, "sy": y_, "syaw": yaw_}),
        )
        if ev.get("aborted"):
            run_error = RuntimeError(f"시나리오가 중단됨: {ev.get('abort_reason')}")
    except Exception as exc:  # noqa: BLE001 - always attempt docking before reporting
        run_error = exc

    return_dock = {"success": True, "action": "skip_return_dock", "reason": None, "already_docked": False, "elapsed_s": 0.0, "dock_status": {"is_docked": False, "dock_visible": False, "received_at_monotonic": time.monotonic()}}
    say("[DOCK] 충전기 복귀 단계 비활성화 (skip_return_dock)")

    if ev is not None:
        ev["pose_source"] = "create3_odometry"
        ev["dock_cycle"] = {
            "initial_dock": initial_dock,
            "undock": undock_result,
            "return_dock": return_dock,
        }
        with open(out / "scenario_summary.json", "w", encoding="utf-8") as f:
            json.dump(ev, f, indent=2, default=str)

    with open(out / "dock_cycle.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "initial_dock": initial_dock,
                "undock": undock_result,
                "return_dock": return_dock,
            },
            f,
            indent=2,
            default=str,
        )

    stop.set()
    lt.join(timeout=1.0)
    with open(out / "robot_states_raw.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_os", "x", "y", "z", "vx", "vy", "vz"])
        writer.writerows(rows)

    exit_code = 0
    if run_error is not None:
        say(f"@@ERROR {run_error!r}")
        exit_code = 1
    elif not return_dock["success"]:
        say(f"@@ERROR {_result_error('충전기 복귀', return_dock)!r}")
        exit_code = 1
    elif ev is None:
        say("@@ERROR 시나리오 결과가 없습니다")
        exit_code = 1
    else:
        say("@@DONE " + json.dumps(ev, default=str))

    _shutdown_node(node)
    time.sleep(0.2)
    os._exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
