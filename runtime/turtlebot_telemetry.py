#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Send TurtleBot 4 pose and motion telemetry to the measurement host.

Run this on the TurtleBot Raspberry Pi (or any ROS 2 machine that can see the
Create 3 TF tree):

    python -m runtime.turtlebot_telemetry --host 192.168.0.10 --port 5011

The receiver should use ``host_rx_time`` for RF alignment because the robot
and host wall clocks need not be identical.  NTP/chrony is still recommended.
The JSON schema is backward compatible with ``hy_robot_proc.py`` and also adds
start-frame pose, translational velocity, motion heading, and yaw rate.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time


SCHEMA = "fr3-turtlebot-telemetry-v1"


def wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def odom_to_start(x: float, y: float, sx: float, sy: float, syaw_deg: float) -> tuple[float, float]:
    c, s = math.cos(math.radians(syaw_deg)), math.sin(math.radians(syaw_deg))
    dx, dy = x - sx, y - sy
    return c * dx + s * dy, -s * dx + c * dy


def rotate_to_start(vx: float, vy: float, syaw_deg: float) -> tuple[float, float]:
    c, s = math.cos(math.radians(syaw_deg)), math.sin(math.radians(syaw_deg))
    return c * vx + s * vy, -s * vx + c * vy


def parse_destinations(hosts: str, port: int) -> list[tuple[str, int]]:
    result = []
    for item in hosts.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            h, p = item.rsplit(":", 1)
            result.append((h, int(p)))
        else:
            result.append((item, int(port)))
    if not result:
        raise ValueError("at least one --host destination is required")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, help="host IP; comma-separated IP[:port] values are allowed")
    ap.add_argument("--port", type=int, default=5011)
    ap.add_argument("--rate", type=float, default=20.0, help="telemetry rate [Hz]")
    ap.add_argument("--pose-frame", default="odom")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--node-name", default="fr3_turtlebot_telemetry")
    ap.add_argument("--print-every", type=int, default=100, help="print one status every N packets; 0 disables")
    a = ap.parse_args()
    if a.rate <= 0:
        ap.error("--rate must be positive")
    destinations = parse_destinations(a.host, a.port)

    try:
        import rclpy
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformListener
    except ImportError as exc:
        print(f"ROS 2 Python packages are required: {exc}", file=sys.stderr)
        return 2

    class TelemetryNode(Node):
        def __init__(self):
            super().__init__(a.node_name)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        def pose(self):
            try:
                tr = self.tf_buffer.lookup_transform(
                    a.pose_frame,
                    a.base_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
            except Exception:
                return None
            q = tr.transform.rotation
            yaw = math.degrees(
                math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                )
            )
            stamp = tr.header.stamp
            ros_time_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            return (
                float(tr.transform.translation.x),
                float(tr.transform.translation.y),
                float(yaw),
                ros_time_s,
            )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rclpy.init()
    node = TelemetryNode()
    period = 1.0 / a.rate
    next_tick = time.monotonic()
    seq = 0
    start = None
    previous = None
    print(f"[TB-TX] {a.pose_frame}->{a.base_frame} at {a.rate:g} Hz -> {destinations}")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=min(0.02, period))
            pose = node.pose()
            now_wall, now_mono = time.time(), time.monotonic()
            # Re-reading the same TF would alternate false stops with velocity spikes.
            if pose is not None and previous is not None and pose[3] == previous[0]:
                pose = None
            if pose is not None:
                x, y, yaw_deg, ros_time_s = pose
                if start is None:
                    start = (x, y, yaw_deg)
                sx, sy, syaw = start
                local_x, local_y = odom_to_start(x, y, sx, sy, syaw)

                vx = vy = yaw_rate = 0.0
                if previous is not None:
                    prev_t, prev_x, prev_y, prev_yaw = previous
                    dt = ros_time_s - prev_t
                    if 1e-3 <= dt <= 1.0:
                        vx = (x - prev_x) / dt
                        vy = (y - prev_y) / dt
                        yaw_rate = wrap_deg(yaw_deg - prev_yaw) / dt
                previous = (ros_time_s, x, y, yaw_deg)
                local_vx, local_vy = rotate_to_start(vx, vy, syaw)
                speed = math.hypot(local_vx, local_vy)
                movement_heading = (
                    math.degrees(math.atan2(local_vy, local_vx)) if speed > 1e-4 else float("nan")
                )
                payload = {
                    "schema": SCHEMA,
                    "seq": seq,
                    "ts": now_wall,
                    "robot_monotonic_s": now_mono,
                    "ros_time_s": ros_time_s,
                    "frame_id": a.pose_frame,
                    "x": x,
                    "y": y,
                    "yaw_deg": yaw_deg,
                    "vx": vx,
                    "vy": vy,
                    "speed_mps": speed,
                    "movement_heading_deg": movement_heading,
                    "yaw_rate_dps": yaw_rate,
                    "sx": sx,
                    "sy": sy,
                    "syaw": syaw,
                    "local_x": local_x,
                    "local_y": local_y,
                    "local_yaw_deg": wrap_deg(yaw_deg - syaw),
                    "local_vx": local_vx,
                    "local_vy": local_vy,
                }
                data = json.dumps(payload, separators=(",", ":"), allow_nan=True).encode("utf-8")
                for addr in destinations:
                    try:
                        sock.sendto(data, addr)
                    except OSError as exc:
                        node.get_logger().warning(f"UDP send to {addr} failed: {exc}")
                seq += 1
                if a.print_every > 0 and seq % a.print_every == 0:
                    print(
                        f"[TB-TX] #{seq} local=({local_x:+.3f},{local_y:+.3f}) "
                        f"yaw={wrap_deg(yaw_deg-syaw):+.1f} speed={speed:.3f} m/s"
                    )
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -period:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

