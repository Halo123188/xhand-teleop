#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astribot Teleoperation -- Vuer + WebRTC camera.

Single-process teleop: Vuer server for AVP (hand/head tracking via websocket),
robot control via Astribot SDK, camera feed (head_rgbd + torso_rgbd stacked)
streamed over WebRTC H264.

Usage:
    source /home/astribot/fortyfive/astribot_sdk_aarch64/env.sh
    ngrok http 8012   # in another terminal
    .venv/bin/python src/astribot/teleop_avp_ros.py --server_url https://XXXX.ngrok.app
"""

import sys
import time
import asyncio
import argparse
import threading
from pathlib import Path

import cv2
import numpy as np
import rclpy.node
import astribot_ros_middleware

# Raise aiortc's hardcoded H264 bitrate ceiling (3 Mbps default is too low
# for sharp 720p). Must be patched before vuer/aiortc instantiates the encoder.
import aiortc.codecs.h264 as _h264
_h264.MIN_BITRATE = 3_000_000
_h264.DEFAULT_BITRATE = 3_000_000
_h264.MAX_BITRATE = 3_000_000

from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, Hands, Head, WebRTCVideoPlane

# Patch rclpy.node.Node.create_publisher to accept 'queue_size' (ROS1 compat).
# The compiled Cython module robotics_library_base.so passes queue_size as a
# keyword argument, but rclpy's Node only accepts qos_profile.
_orig_create_publisher = rclpy.node.Node.create_publisher

def _patched_create_publisher(self, msg_type, topic, qos_profile=10, **kwargs):
    if "queue_size" in kwargs:
        qos_profile = kwargs.pop("queue_size")
    return _orig_create_publisher(self, msg_type, topic, qos_profile, **kwargs)

rclpy.node.Node.create_publisher = _patched_create_publisher

_SDK_ROOT = Path("/home/astribot/fortyfive/astribot_sdk_aarch64")
if str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))
from astribot_sdk.core.astribot_api.astribot_client import Astribot

CAMERAS = ["head_rgbd", "torso_rgbd"]


# ---------------------------------------------------------------------------
# Math helpers
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


# ---------------------------------------------------------------------------
# Head tracking — pitch/yaw decomposition (2-DOF, no roll)
# ---------------------------------------------------------------------------

def head_quat_from_yz(pitch, yaw):
    """Build quaternion [x,y,z,w] from head pitch (nod) and yaw (turn).

    Rotation order: Ry(yaw) * Rz(pitch), matching the robot's 2-DOF head.
    """
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([
        -sy * sp,           # x
        sy * cp,            # y
        cy * sp,            # z
        cy * cp,            # w
    ])


def make_head_tracker(alpha=0.9, pitch_limit=0.8, yaw_limit=0.8,
                      warmup_frames=30):
    """Head tracker using pitch/yaw angles (2-DOF).

    Args:
        alpha: EMA smoothing factor per frame (0=frozen, 1=instant)
        pitch_limit: max pitch deviation from baseline (radians, ~46 deg)
        yaw_limit: max yaw deviation from baseline (radians, ~46 deg)
        warmup_frames: number of initial frames to skip before capturing baseline
    """
    return dict(
        init_pitch=None,
        init_yaw=None,
        smooth_pitch=0.0,
        smooth_yaw=0.0,
        current_quat=None,
        ready=False,
        alpha=alpha,
        pitch_limit=pitch_limit,
        yaw_limit=yaw_limit,
        last_update=None,
        warmup_frames=warmup_frames,
        frame_count=0,
    )


def _extract_pitch_yaw(mat):
    """Extract pitch and yaw from a VR 4x4 head matrix.

    VR frame: Y-up, -Z forward.
    Pitch: elevation angle of the gaze direction (positive = looking up).
    Yaw: heading of the gaze direction on the XZ ground plane (turning).
    """
    forward = -mat[:3, 2]
    pitch = np.arcsin(np.clip(forward[1], -1, 1))
    yaw = np.arctan2(forward[0], -forward[2])
    return pitch, yaw


def update_head(t, mat):
    pitch, yaw = _extract_pitch_yaw(mat)
    t["frame_count"] += 1

    # Skip initial frames before tracking data stabilises
    if t["frame_count"] <= t["warmup_frames"]:
        return

    if t["init_pitch"] is None:
        t["init_pitch"] = pitch
        t["init_yaw"] = yaw
        print(f"[Head] Baseline: pitch={pitch:.3f} yaw={yaw:.3f}")
        return

    # Delta from baseline, clamped to safety limits
    dp = np.clip(pitch - t["init_pitch"], -t["pitch_limit"], t["pitch_limit"])
    dy = np.clip(yaw - t["init_yaw"], -t["yaw_limit"], t["yaw_limit"])

    # EMA smoothing
    t["smooth_pitch"] = t["alpha"] * dp + (1 - t["alpha"]) * t["smooth_pitch"]
    t["smooth_yaw"] = t["alpha"] * dy + (1 - t["alpha"]) * t["smooth_yaw"]

    # Robot head convention: Ry = pitch (nod), Rz = yaw (turn), both inverted
    # relative to VR. VR yaw drives the function's "pitch" slot and VR pitch
    # drives the function's "yaw" slot, both with sign flips.
    t["current_quat"] = head_quat_from_yz(-t["smooth_yaw"], -t["smooth_pitch"])
    t["ready"] = True
    t["last_update"] = time.monotonic()


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def activate_camera(astribot, name):
    astribot.activate_camera()
    for _ in range(10):
        if astribot.get_cameras_info().get(name, {}).get("activate"):
            print(f"[CAM] {name} active")
            return True
        time.sleep(1.0)
    print(f"[CAM] {name} activation timeout")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Astribot teleop + WebRTC camera")
    p.add_argument("--server_url", required=True, help="ngrok HTTPS URL")
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--freq", type=int, default=100, help="Control loop Hz")
    p.add_argument("--camera_fps", type=int, default=30)
    p.add_argument("--camera_width", type=int, default=1280, help="Tile width")
    p.add_argument("--camera_height", type=int, default=720, help="Tile height")
    p.add_argument("--camera_bitrate", type=int, default=3_000_000,
                   help="WebRTC max bitrate in bps")
    args = p.parse_args()

    # ---- SDK init ----
    astribot = Astribot(freq=float(args.freq))
    print(f"SDK ready (alive={astribot.is_alive})")
    astribot.move_to_home()

    HEAD_HOME = astribot.get_desired_cartesian_pose(
        names=[astribot.head_name], frame="world")[0]
    print(f"[SDK] HEAD_HOME={HEAD_HOME}")

    # ---- Cameras ----
    for c in CAMERAS:
        activate_camera(astribot, c)

    latest_frames = {c: None for c in CAMERAS}
    recv_count = [0]

    def on_image(topic_name, msg, width, height, array):
        if array is None or msg.format.lower() != "jpeg":
            return
        parts = topic_name.split('/')
        cam = parts[2] if len(parts) > 2 else None
        if cam in latest_frames:
            latest_frames[cam] = array
            recv_count[0] += 1

    for c in CAMERAS:
        astribot.register_image_callback(c, "color", on_image, need_decode=True)

    # ---- Vuer + WebRTC ----
    app = Vuer(host="0.0.0.0", port=args.port)
    tw, th = args.camera_width, args.camera_height
    stream = app.create_webrtc_stream(
        "robot-camera", codec="H264",
        max_bitrate=args.camera_bitrate,
        max_framerate=args.camera_fps, resolution=(tw, th * 2))

    def camera_sender():
        stream.wait_ready_sync(timeout=30)
        print("[CAM] WebRTC ready")
        interval = 1.0 / max(args.camera_fps, 1)
        last, n = None, 0

        def _fit(f):
            if f is None:
                return np.zeros((th, tw, 3), dtype=np.uint8)
            if f.shape[0] == th and f.shape[1] == tw:
                return f
            return cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)

        while True:
            hf, tf = latest_frames["head_rgbd"], latest_frames["torso_rgbd"]
            if hf is not None or tf is not None:
                last = np.vstack((_fit(hf), _fit(tf)))
            if last is not None:
                stream.push_frame(last)
                n += 1
                if n == 1 or n % 300 == 0:
                    print(f"[CAM] #{n} recv={recv_count[0]}")
            time.sleep(interval)

    threading.Thread(target=camera_sender, daemon=True, name="CamSender").start()

    # ---- Trackers ----
    ht = make_head_tracker()
    head_home_pos = list(HEAD_HOME[:3])
    head_home_quat = np.array(HEAD_HOME[3:7])

    # Stop sending head commands if AVP data is older than this
    HEAD_STALE_TIMEOUT = 0.5  # seconds

    # ---- Shared state ----
    lock = threading.Lock()
    tracking = {"head": None}

    @app.add_handler("HEAD_MOVE")
    async def on_head_evt(event, session):
        m = event.value.get("matrix")
        if m is not None:
            with lock:
                tracking["head"] = m

    # ---- Control loop (head only) ----
    def control_loop():
        rate = astribot_ros_middleware.Rate(args.freq)
        n, cmd_n = 0, 0
        while True:
            with lock:
                hm = tracking["head"]

            if n % 500 == 0:
                stale = ""
                if ht["last_update"] is not None:
                    age = time.monotonic() - ht["last_update"]
                    stale = f" age={age:.2f}s"
                print(f"[DIAG @{n}] head={'ok' if ht['ready'] else 'NONE'}"
                      f" p={ht['smooth_pitch']:.3f} y={ht['smooth_yaw']:.3f}"
                      f"{stale} cmds={cmd_n}")
            n += 1

            if hm is not None:
                update_head(ht, np.array(hm).reshape(4, 4).T)

            # Skip if head data is stale (AVP disconnected / tracking lost)
            if ht["last_update"] is not None and \
               (time.monotonic() - ht["last_update"]) > HEAD_STALE_TIMEOUT:
                rate.sleep()
                continue

            if ht["ready"] and ht["current_quat"] is not None:
                final_quat = quat_multiply(ht["current_quat"], head_home_quat)
                head_pose = head_home_pos + [float(v) for v in final_quat]
            else:
                head_pose = list(HEAD_HOME)

            astribot.set_cartesian_pose(
                [astribot.head_name], [head_pose],
                control_way="direct", add_default_torso=True)
            cmd_n += 1
            rate.sleep()

    threading.Thread(target=control_loop, daemon=True, name="CtrlLoop").start()

    # ---- AVP browser session ----
    @app.spawn(start=True, client="browser")
    async def main_session(session: VuerSession):
        session.set @ DefaultScene(
            Hands(stream=True, fps=30, key="hands"),
            Head(stream=True, fps=60, key="head"),
            WebRTCVideoPlane(
                src="/webrtc/offer/robot-camera",
                distanceToCamera=3, height=1.5,
                key="webrtc-camera",
            ),
        )
        print("Vuer ready -- open ngrok URL on Apple Vision Pro")
        while True:
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    main()
