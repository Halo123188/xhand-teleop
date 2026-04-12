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

from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, Hands, Head, WebRTCVideoPlane

_SDK_ROOT = Path("/home/astribot/fortyfive/astribot_sdk_aarch64")
if str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))
from astribot_sdk.core.astribot_api.astribot_client import Astribot

# Axis remap: VR (Y-up, -Z forward) -> robot frame
_P = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
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


def rot_to_quat(R):
    """3x3 rotation matrix -> quaternion [x, y, z, w]."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([(R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s])


def delta_quat(R_now, R_base, base_robot_quat):
    """Compute robot quat from VR rotation delta applied to robot base quat."""
    R_delta = _P @ (R_now @ R_base.T) @ _P.T
    return quat_multiply(rot_to_quat(R_delta), base_robot_quat)


# ---------------------------------------------------------------------------
# Wrist tracking (dict-based, no class)
# ---------------------------------------------------------------------------

def make_wrist_tracker(max_vel=0.005, dead_zone=0.002, alpha=0.3, radius=0.8):
    return dict(max_vel=max_vel, dead_zone=dead_zone, alpha=alpha, radius=radius,
                base_pos=None, base_rot=None, robot_pos=None, robot_quat=None,
                home=None, smooth=None)


def update_wrist(t, wrist_mat, robot_pose_7):
    """Returns (pos_list, quat_list) or None on first frame / missing data."""
    if wrist_mat is None or robot_pose_7 is None:
        return None
    pos = wrist_mat[:3, 3]
    rp, rq = np.array(robot_pose_7[:3]), np.array(robot_pose_7[3:7])

    if t["base_pos"] is None:
        t["base_pos"], t["base_rot"] = pos.copy(), wrist_mat[:3, :3].copy()
        t["robot_pos"], t["robot_quat"] = rp.copy(), rq.copy()
        t["home"], t["smooth"] = rp.copy(), rp.copy()
        print(f"[Wrist] Calibrated. home={rp.tolist()}")
        return None

    # Position delta remapped to robot frame
    d = pos - t["base_pos"]
    target = t["robot_pos"] + np.array([d[0], -d[2], d[1]])

    if np.linalg.norm(target - t["smooth"]) < t["dead_zone"]:
        pos_out = t["smooth"]
    else:
        delta = target - t["smooth"]
        dist = np.linalg.norm(delta)
        if dist > t["max_vel"]:
            target = t["smooth"] + delta * (t["max_vel"] / dist)
        off = target - t["home"]
        r = np.linalg.norm(off)
        if r > t["radius"]:
            target = t["home"] + off * (t["radius"] / r)
        t["smooth"] = t["alpha"] * target + (1 - t["alpha"]) * t["smooth"]
        pos_out = t["smooth"]

    q_out = delta_quat(wrist_mat[:3, :3], t["base_rot"], t["robot_quat"])
    return list(pos_out), [float(v) for v in q_out]


# ---------------------------------------------------------------------------
# Head tracking (dict-based, no class)
# ---------------------------------------------------------------------------

def make_head_tracker(robot_head_quat):
    return dict(init_rot=None, robot_quat=np.array(robot_head_quat),
                current_quat=None, ready=False)


def update_head(t, mat):
    if t["init_rot"] is None:
        t["init_rot"] = mat[:3, :3].copy()
        print(f"[Head] Baseline captured")
    t["current_quat"] = delta_quat(mat[:3, :3], t["init_rot"], t["robot_quat"])
    t["ready"] = True


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
    p.add_argument("--freq", type=int, default=250, help="Control loop Hz")
    p.add_argument("--max_velocity", type=float, default=0.005)
    p.add_argument("--workspace_radius", type=float, default=0.8)
    p.add_argument("--camera_fps", type=int, default=30)
    p.add_argument("--camera_width", type=int, default=640, help="Tile width")
    p.add_argument("--camera_height", type=int, default=480, help="Tile height")
    args = p.parse_args()

    # ---- SDK init ----
    astribot = Astribot(freq=float(args.freq))
    print(f"SDK ready (alive={astribot.is_alive})")
    astribot.move_to_home()

    arm_names = [astribot.arm_left_name, astribot.arm_right_name]
    arm_homes = astribot.get_desired_cartesian_pose(names=arm_names, frame="world")
    LEFT_HOME, RIGHT_HOME = arm_homes[0], arm_homes[1]
    HEAD_HOME = astribot.get_desired_cartesian_pose(
        names=[astribot.head_name], frame="world")[0]
    print(f"[SDK] L={LEFT_HOME}  R={RIGHT_HOME}  H={HEAD_HOME}")

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
        max_framerate=args.camera_fps, resolution=(tw, th * 2))

    def camera_sender():
        stream.wait_ready_sync(timeout=30)
        print("[CAM] WebRTC ready")
        interval = 1.0 / max(args.camera_fps, 1)
        last, n = None, 0
        while True:
            hf, tf = latest_frames["head_rgbd"], latest_frames["torso_rgbd"]
            if hf is not None or tf is not None:
                top = cv2.resize(hf, (tw, th), interpolation=cv2.INTER_NEAREST) \
                    if hf is not None else np.zeros((th, tw, 3), dtype=np.uint8)
                bot = cv2.resize(tf, (tw, th), interpolation=cv2.INTER_NEAREST) \
                    if tf is not None else np.zeros((th, tw, 3), dtype=np.uint8)
                last = np.vstack((top, bot))
            if last is not None:
                stream.push_frame(last)
                n += 1
                if n == 1 or n % 300 == 0:
                    print(f"[CAM] #{n} recv={recv_count[0]}")
            time.sleep(interval)

    threading.Thread(target=camera_sender, daemon=True, name="CamSender").start()

    # ---- Trackers ----
    rh = make_wrist_tracker(max_vel=args.max_velocity, radius=args.workspace_radius)
    lh = make_wrist_tracker(max_vel=args.max_velocity, radius=args.workspace_radius)
    ht = make_head_tracker(HEAD_HOME[3:7])

    # ---- Shared state ----
    lock = threading.Lock()
    tracking = {"r": None, "l": None, "head": None, "rc": 0, "lc": 0}

    @app.add_handler("HAND_MOVE")
    async def on_hand(event, session):
        d = event.value
        with lock:
            if d.get("right"):
                tracking["r"] = d["right"]
                tracking["rc"] += 1
                if tracking["rc"] == 1:
                    print("[RECV] First RIGHT hand data")
            if d.get("left"):
                tracking["l"] = d["left"]
                tracking["lc"] += 1
                if tracking["lc"] == 1:
                    print("[RECV] First LEFT hand data")

    @app.add_handler("HEAD_MOVE")
    async def on_head_evt(event, session):
        m = event.value.get("matrix")
        if m is not None:
            with lock:
                tracking["head"] = m

    # ---- Control loop ----
    def control_loop():
        dt = 1.0 / args.freq
        n, cmd_n = 0, 0
        while True:
            with lock:
                rp, lp, hm = tracking["r"], tracking["l"], tracking["head"]

            rm = np.array(rp[:16]).reshape(4, 4).T if rp and len(rp) >= 16 else None
            lm = np.array(lp[:16]).reshape(4, 4).T if lp and len(lp) >= 16 else None

            if n % 500 == 0:
                print(f"[DIAG @{n}] R={tracking['rc']} L={tracking['lc']} cmds={cmd_n}")
            n += 1

            rr = update_wrist(rh, rm, RIGHT_HOME)
            lr = update_wrist(lh, lm, LEFT_HOME)
            if hm is not None:
                update_head(ht, np.array(hm).reshape(4, 4).T)

            names, poses = [], []
            if rr:
                names.append(astribot.arm_right_name)
                poses.append(rr[0] + rr[1])
            if lr:
                names.append(astribot.arm_left_name)
                poses.append(lr[0] + lr[1])
            if ht["ready"] and ht["current_quat"] is not None:
                names.append(astribot.head_name)
                poses.append(list(HEAD_HOME[:3]) + [float(v) for v in ht["current_quat"]])

            if names:
                astribot.set_cartesian_pose(names, poses,
                                            control_way="filter", add_default_torso=True)
                cmd_n += 1
            time.sleep(dt)

    threading.Thread(target=control_loop, daemon=True, name="CtrlLoop").start()

    # ---- AVP browser session ----
    @app.spawn(start=True, client="browser")
    async def main_session(session: VuerSession):
        session.set @ DefaultScene(
            Hands(stream=True, fps=30, key="hands"),
            Head(stream=True, fps=30, key="head"),
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
