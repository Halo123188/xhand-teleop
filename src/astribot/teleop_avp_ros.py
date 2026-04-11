#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astribot Teleoperation — ROS 2 Client (Astribot SDK).

Connects to the Vuer server (teleop_avp.py) via websocket.
Receives hand/head tracking, computes arm/head commands,
and drives the robot via the Astribot SDK (set_cartesian_pose).

Requirements:
    Ubuntu 22.04+ with ROS 2 Humble or above
    ROS_DOMAIN_ID=25 (robot default)
    astribot_sdk_ros2 installed (supports x86 and ARM)

Usage:
    # Clone the ROS2 SDK if not already present:
    #   git clone <astribot_sdk_ros2_url> /home/astribot/fortyfive/astribot-sdk-ros2
    source /home/astribot/fortyfive/astribot-sdk-ros2/env.sh
    python3 src/astribot/teleop_avp_ros.py --server_uri ws://<vuer-ip>:8012
"""

import io
import sys
import time
import asyncio
import threading
from collections import deque
from pathlib import Path

from PIL import Image as PILImage

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from vuer.client import VuerClient
from vuer.events import ClientEvent

# ---- Astribot ROS2 SDK ----
_SDK_ROOT = Path("/home/astribot/fortyfive/astribot-sdk-ros2")
if str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))
from core.astribot_api.astribot_client import Astribot


# ---------------------------------------------------------------------------
# Quaternion math (XYZW format, self-contained)
# ---------------------------------------------------------------------------

def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])

def from_3js(matrix):
    return np.array(matrix).reshape(4, 4).T

# Axis remap: VR (Y-up, -Z forward) → robot (same as position delta: x→x, -z→y, y→z)
_P = np.array([[1,  0,  0],
               [0,  0, -1],
               [0,  1,  0]], dtype=float)

def rot_to_quat(R):
    """3x3 rotation matrix → quaternion [x, y, z, w] (Shepperd method)."""
    R = np.asarray(R, dtype=float)
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])

def wrist_quat(mat4x4):
    """Extract wrist orientation from a 4x4 wrist matrix, remapped to robot frame."""
    R_robot = _P @ mat4x4[:3, :3] @ _P.T
    return [float(v) for v in rot_to_quat(R_robot)]


# ---------------------------------------------------------------------------
# WristTracker — always-on arm tracking with safety
# ---------------------------------------------------------------------------

def quat_to_rot(q):
    """Quaternion [x, y, z, w] → 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


class WristTracker:
    """Maps VR wrist pose to robot arm pose (relative delta from calibration).

    Position: delta from first VR frame anchored to robot's current position.
    Orientation: delta rotation from first VR frame applied to robot's current quaternion.
    Safety: velocity clamping, workspace bounds, dead zone, EMA smoothing.
    """

    def __init__(self, max_velocity=0.005, dead_zone=0.002,
                 smoothing_alpha=0.3, workspace_radius=0.8):
        self.max_velocity = max_velocity
        self.dead_zone = dead_zone
        self.alpha = smoothing_alpha
        self.radius = workspace_radius
        self._base_hand_pos = None
        self._base_hand_rot = None      # VR rotation at calibration (3x3)
        self._base_robot_pos = None
        self._base_robot_quat = None    # robot quaternion at calibration [x,y,z,w]
        self._home = None
        self._smooth = None

    def reset(self):
        self._base_hand_pos = None

    def update(self, wrist_matrix, robot_pose_7):
        """
        Args:
            wrist_matrix: 4x4 VR wrist transform
            robot_pose_7: [x, y, z, qx, qy, qz, qw] — robot arm's current/initial pose

        Returns:
            (position [x,y,z], quaternion [qx,qy,qz,qw]) or None on first frame.
        """
        if wrist_matrix is None or robot_pose_7 is None:
            return None

        pos = wrist_matrix[:3, 3]
        robot_pos = np.array(robot_pose_7[:3])
        robot_quat = np.array(robot_pose_7[3:7])   # [qx, qy, qz, qw]

        if self._base_hand_pos is None:
            self._base_hand_pos = pos.copy()
            self._base_hand_rot = wrist_matrix[:3, :3].copy()
            self._base_robot_pos = robot_pos.copy()
            self._base_robot_quat = robot_quat.copy()
            self._home = robot_pos.copy()
            self._smooth = robot_pos.copy()
            print(f"[WristTracker] Calibrated. VR base pos={pos.tolist()} robot home={robot_pos.tolist()}")
            return None

        # --- Position: delta in VR space remapped to robot frame ---
        d = pos - self._base_hand_pos
        target = self._base_robot_pos + np.array([d[0], -d[2], d[1]])

        # Dead zone
        if np.linalg.norm(target - self._smooth) < self.dead_zone:
            pos_out = list(self._smooth)
        else:
            # Velocity clamp
            delta = target - self._smooth
            dist = np.linalg.norm(delta)
            if dist > self.max_velocity:
                target = self._smooth + delta * (self.max_velocity / dist)

            # Workspace bounds
            offset = target - self._home
            r = np.linalg.norm(offset)
            if r > self.radius:
                target = self._home + offset * (self.radius / r)

            # EMA smoothing
            self._smooth = self.alpha * target + (1 - self.alpha) * self._smooth
            pos_out = list(self._smooth)

        # --- Orientation: delta from calibration VR rotation, applied to robot base quat ---
        R_vr_delta = wrist_matrix[:3, :3] @ self._base_hand_rot.T
        R_delta_robot = _P @ R_vr_delta @ _P.T
        q_delta = rot_to_quat(R_delta_robot)
        # Apply delta on top of the robot's initial quaternion
        q_out = quat_multiply(q_delta, self._base_robot_quat)

        return pos_out, [float(v) for v in q_out]


# ---------------------------------------------------------------------------
# HeadTracker — relative pitch/yaw from WebXR head matrix
# ---------------------------------------------------------------------------

class HeadTracker:
    """Tracks head pitch/yaw relative to calibration frame.

    Orientation output is applied as a delta on top of the robot's current head quaternion.
    """

    def __init__(self):
        self._init_pitch = None
        self._init_yaw = None
        self._init_vr_rot = None
        self._base_robot_quat = None  # robot head quaternion at calibration [x,y,z,w]
        self.pitch = 0.0
        self.yaw = 0.0
        self.ready = False
        self._current_quat = None

    def calibrate(self, robot_head_quat):
        """Call once with the robot's actual current head quaternion [x,y,z,w]."""
        self._base_robot_quat = np.array(robot_head_quat)

    def update(self, mat):
        pitch_raw = np.arccos(np.clip(np.dot(mat[:3,1], [0,1,0]), -1, 1)) - np.pi/2
        yaw_raw = np.arctan2(mat[:3,2][0], mat[:3,2][2])
        if self._init_pitch is None:
            self._init_pitch, self._init_yaw = pitch_raw, yaw_raw
            self._init_vr_rot = mat[:3, :3].copy()
            print(f"[HeadTrack] Baseline: pitch={pitch_raw:.3f}, yaw={yaw_raw:.3f}")
        self.pitch = pitch_raw - self._init_pitch
        self.yaw = yaw_raw - self._init_yaw

        # Orientation: delta from initial VR head rotation applied to robot's initial head quat
        if self._base_robot_quat is not None and self._init_vr_rot is not None:
            R_vr_delta = mat[:3, :3] @ self._init_vr_rot.T
            R_delta_robot = _P @ R_vr_delta @ _P.T
            q_delta = rot_to_quat(R_delta_robot)
            self._current_quat = quat_multiply(q_delta, self._base_robot_quat)

        self.ready = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Astribot ROS2 client for teleop_avp (Astribot SDK)")
    p.add_argument("--server_uri", default="ws://localhost:8012")
    p.add_argument("--freq", type=int, default=250)
    p.add_argument("--max_velocity", type=float, default=0.005)
    p.add_argument("--workspace_radius", type=float, default=0.8)
    p.add_argument("--camera_topic", default="/astribot_camera/head_stereo/color_compress/compressed")
    p.add_argument("--camera_fps", type=int, default=10)
    args = p.parse_args()

    # ---- ROS 2 init ----
    rclpy.init()
    node = rclpy.create_node("teleop_avp_ros2")
    print("ROS 2 node ready")

    # Spin the node in a background thread so subscriptions stay alive
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ---- Astribot SDK ----
    print("Initializing Astribot SDK...")
    astribot = Astribot(freq=float(args.freq))
    print(f"SDK ready (alive={astribot.is_alive})")

    # Move robot to home pose before starting teleoperation
    print("Moving to home pose...")
    result = astribot.move_to_home()
    print(f"Move to home result: {result}")

    # Get home poses from SDK (world frame) — used as reference for all tracking deltas
    arm_names = [astribot.arm_left_name, astribot.arm_right_name]
    init_poses = astribot.get_desired_cartesian_pose(names=arm_names, frame="world")
    LEFT_ARM_HOME  = init_poses[0]   # [x, y, z, qx, qy, qz, qw]
    RIGHT_ARM_HOME = init_poses[1]
    print(f"[SDK] Left  arm home: {LEFT_ARM_HOME}")
    print(f"[SDK] Right arm home: {RIGHT_ARM_HOME}")

    # Head home
    head_init = astribot.get_desired_cartesian_pose(names=[astribot.head_name], frame="world")
    HEAD_HOME_POSE = head_init[0]  # [x, y, z, qx, qy, qz, qw]
    print(f"[SDK] Head  home: {HEAD_HOME_POSE}")

    # ---- Camera subscriber ----
    # Stereo image is 3200x1200; crop left half (1600x1200) for the HUD.
    frame_queue = deque(maxlen=2)
    _warn_t = [0.0]

    def on_camera_image(msg):
        try:
            img = PILImage.open(io.BytesIO(bytes(msg.data)))
            w, h = img.size
            left_half = img.crop((0, 0, w // 2, h))
            buf = io.BytesIO()
            left_half.save(buf, format="JPEG", quality=80)
            frame_queue.append(buf.getvalue())
        except Exception as e:
            now = time.monotonic()
            if now - _warn_t[0] > 5.0:
                node.get_logger().warning(f"[camera] decode error: {e}")
                _warn_t[0] = now

    camera_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    node.create_subscription(CompressedImage, args.camera_topic, on_camera_image, camera_qos)

    # ---- Trackers ----
    rh = WristTracker(max_velocity=args.max_velocity, workspace_radius=args.workspace_radius)
    lh = WristTracker(max_velocity=args.max_velocity, workspace_radius=args.workspace_radius)
    head = HeadTracker()
    head.calibrate(HEAD_HOME_POSE[3:7])  # seed head tracker with robot's current orientation

    # ---- Shared tracking state ----
    lock = threading.Lock()
    tracking = {"r_poses": None, "l_poses": None, "head_mat": None,
                "_r_count": 0, "_l_count": 0}

    # ---- Control loop (250 Hz) ----
    def control_loop():
        interval = 1.0 / args.freq
        _loop_n = 0
        _cmd_n = 0
        while rclpy.ok():
            with lock:
                r_poses = tracking["r_poses"]
                l_poses = tracking["l_poses"]
                head_mat = tracking["head_mat"]
                r_count = tracking["_r_count"]
                l_count = tracking["_l_count"]

            r_mat = np.array(r_poses[0:16]).reshape(4,4).T if r_poses and len(r_poses) >= 16 else None
            l_mat = np.array(l_poses[0:16]).reshape(4,4).T if l_poses and len(l_poses) >= 16 else None

            # Diagnose: every 250 ticks (~1 s) print tracking status
            if _loop_n % 250 == 0:
                print(
                    f"[DIAG @{_loop_n}] events: right={r_count} left={l_count} "
                    f"| r_mat={'ok' if r_mat is not None else 'NONE'} "
                    f"l_mat={'ok' if l_mat is not None else 'NONE'} "
                    f"head={'ok' if head_mat is not None else 'NONE'} "
                    f"| rh.calibrated={rh._base_hand_pos is not None} "
                    f"lh.calibrated={lh._base_hand_pos is not None} "
                    f"| cmd_count={_cmd_n}"
                )
            _loop_n += 1

            # WristTracker: full 7-element pose [x,y,z,qx,qy,qz,qw] as home seed
            r_result = rh.update(r_mat, RIGHT_ARM_HOME)
            l_result = lh.update(l_mat, LEFT_ARM_HOME)

            if head_mat is not None:
                head.update(from_3js(head_mat))

            # Build SDK command lists
            sdk_names = []
            sdk_poses = []

            if r_result is not None:
                r_pos, r_quat = r_result
                sdk_names.append(astribot.arm_right_name)
                sdk_poses.append(r_pos + r_quat)   # [x, y, z, qx, qy, qz, qw]

            if l_result is not None:
                l_pos, l_quat = l_result
                sdk_names.append(astribot.arm_left_name)
                sdk_poses.append(l_pos + l_quat)

            if head.ready and head._current_quat is not None:
                head_pose = list(HEAD_HOME_POSE[:3]) + [float(v) for v in head._current_quat]
                sdk_names.append(astribot.head_name)
                sdk_poses.append(head_pose)

            if sdk_names:
                astribot.set_cartesian_pose(sdk_names, sdk_poses,
                                            control_way="filter", add_default_torso=True)
                _cmd_n += 1
                if _cmd_n == 1 or _cmd_n % 500 == 0:
                    r_str = f"{r_result[0]}" if r_result else "—"
                    l_str = f"{l_result[0]}" if l_result else "—"
                    print(f"[CMD #{_cmd_n}] names={sdk_names} | R={r_str} | L={l_str}")

            time.sleep(interval)

    threading.Thread(target=control_loop, daemon=True).start()

    # ---- Websocket bridge ----
    async def bridge():
        while rclpy.ok():
            try:
                print(f"Connecting to {args.server_uri}...")
                async with VuerClient(uri=args.server_uri) as client:
                    print("Connected!")

                    async def recv_tracking():
                        _recv_n = 0
                        while rclpy.ok():
                            event = await client.recv()
                            if event is None:
                                continue
                            etype = getattr(event, "etype", None)
                            # ServerEvent serializes as 'data'; ClientEvent as 'value'
                            payload = getattr(event, "data", None) or getattr(event, "value", None)

                            # Print every event type on the first few, then throttled
                            _recv_n += 1
                            if _recv_n <= 10 or _recv_n % 200 == 0:
                                print(f"[RECV #{_recv_n}] etype={etype!r} payload_type={type(payload).__name__} "
                                      f"keys={list(payload.keys()) if isinstance(payload, dict) else '?'}")

                            if not isinstance(payload, dict):
                                continue
                            with lock:
                                if etype == "HAND_TRACKING":
                                    if payload.get("right"):
                                        tracking["r_poses"] = payload["right"]
                                        tracking["_r_count"] += 1
                                        if tracking["_r_count"] == 1:
                                            print(f"[RECV] First RIGHT hand tracking! poses len={len(payload['right'])}")
                                    if payload.get("left"):
                                        tracking["l_poses"] = payload["left"]
                                        tracking["_l_count"] += 1
                                        if tracking["_l_count"] == 1:
                                            print(f"[RECV] First LEFT hand tracking! poses len={len(payload['left'])}")
                                elif etype == "HEAD_TRACKING":
                                    if payload.get("matrix"):
                                        tracking["head_mat"] = payload["matrix"]

                    async def send_camera():
                        interval = 1.0 / max(args.camera_fps, 1)
                        while rclpy.ok():
                            if frame_queue:
                                await client.send(ClientEvent(
                                    etype="CAMERA_FRAME", value={"jpeg": frame_queue[-1]}))
                            await asyncio.sleep(interval)

                    await asyncio.gather(send_camera(), recv_tracking())
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Disconnected: {e}, retrying in 2s...")
                await asyncio.sleep(2.0)

    try:
        asyncio.run(bridge())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
