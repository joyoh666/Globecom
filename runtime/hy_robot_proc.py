# -*- coding: utf-8 -*-
"""
[GC 1117] Globecom 판: 이동은 runtime/motion.py (MOTION=scenario|random|manual|facing, DURATION 초), 위치는 LIVE_STATUS_UDP 여러 곳으로.

TurtleBot control child process used by DMIMO_3TX_Live_Rx.py.

The robot pose comes from Create 3 odometry (odom -> base_link), so neither SLAM
nor Nav2 localization is required.

Parent protocol (one line per event):
  @@READY                   odometry is ready; parent may send GO
  @@T0 <wall> <monotonic>   first scenario velocity command marker
  @@MEASURED <json>        forward motion ended; child waits for RETURN
  @@DONE <json>            scenario and return completed
  @@ERROR <message>        failure; the next batch item must not start
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
from .motion import retrace_motion, run_motion  # [GC 1117] scenario(한 경로 계속) / random(무작위 경유점) / manual(사람이 조종)


def say(msg):
    print(msg, flush=True)


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
    return_timeout = float(os.environ.get("RETURN_TIMEOUT_SEC", str(3 * duration + 30)))
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

    say("@@READY")

    if sys.stdin.readline().strip() != "GO":
        say("@@ERROR 부모가 GO 를 안 보냈습니다")
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
    except Exception as exc:
        run_error = exc

    stop.set()
    lt.join(timeout=1.0)
    with open(out / "robot_states_raw.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_os", "x", "y", "z", "vx", "vy", "vz"])
        writer.writerows(rows)

    exit_code = 0
    try:
        if run_error is not None:
            raise run_error
        ev["pose_source"] = "create3_odometry"
        say("@@MEASURED " + json.dumps(ev, default=str))
        if sys.stdin.readline().strip() != "RETURN":
            raise RuntimeError("부모가 복귀를 승인하지 않았습니다")
        ev["return_path"] = ({"status": "skipped_manual"} if motion == "manual"
                             else retrace_motion(node, ev, return_timeout))
        say("@@DONE " + json.dumps(ev, default=str))
    except (Exception, KeyboardInterrupt) as exc:
        if ev is not None:
            ev["return_path"] = {"status": "failed", "error": repr(exc)}
        say(f"@@ERROR {exc!r}")
        exit_code = 1
    finally:
        if ev is not None:
            with open(out / "scenario_summary.json", "w", encoding="utf-8") as f:
                json.dump(ev, f, indent=2, default=str)

    _shutdown_node(node)
    time.sleep(0.2)
    os._exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
