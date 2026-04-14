#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astribot Teleoperation -- Vuer + WebRTC camera.

Single-process teleop via Apple Vision Pro:
  - Vuer hosts the WebXR session; AVP streams Head + Hands tracking events.
  - Head 2-DOF (pitch/yaw) mapped to Astribot head pose.
  - Both wrists 6-DOF mapped to left/right arm end-effectors (relative to
    arm home). Torso follows via SDK WBC.
  - Dual camera feed (head_rgbd + torso_rgbd stacked) streamed back over
    WebRTC H264 as an AVP HUD.

Smoothing strategy: thin upstream EMA + safety clips on velocity and
workspace, real smoothing is handled by the SDK filter (control_way=
'filter' with configurable filter_scale).

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


def quat_slerp(q1, q2, t):
    """Spherical linear interpolation between two quats [x, y, z, w]."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        out = q1 + t * (q2 - q1)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    return (np.sin((1.0 - t) * theta) * q1 + np.sin(t * theta) * q2) / s


def quat_angle(q1, q2):
    """Angle (rad) between two unit quaternions [x, y, z, w]."""
    dot = abs(float(np.dot(q1, q2)))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def mat_to_quat(m):
    """3x3 rotation matrix -> quaternion [x, y, z, w]."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


# VR frame (Y-up, -Z forward, +X user-right) -> Robot world (Z-up, +X forward,
# +Y robot-left). Applied as p_robot = VR_TO_ROBOT @ p_vr, and for rotation
# deltas as R_robot = VR_TO_ROBOT @ R_vr @ VR_TO_ROBOT.T. Tune signs if the
# mapping is off for your setup.
VR_TO_ROBOT = np.array([
    [ 0,  0, -1],
    [-1,  0,  0],
    [ 0,  1,  0],
], dtype=float)


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
# Wrist tracking -- 6-DOF pose delta from VR wrist matrix
# ---------------------------------------------------------------------------

def make_wrist_tracker(pos_alpha=0.5, pos_limit=0.5,
                        max_lin_vel=0.5, pos_deadband=0.002,
                        rot_alpha=0.15, max_ang_vel=1.5,
                        rot_deadband=0.02, warmup_frames=30):
    """Per-hand tracker for wrist pose (position + orientation delta).

    Args:
        pos_alpha: EMA smoothing factor for position (0=frozen, 1=instant)
        pos_limit: max per-axis position deviation from baseline (meters)
        max_lin_vel: max Cartesian speed of the commanded target (m/s);
            caps the per-step jump to avoid IK joint-velocity blow-ups
        pos_deadband: VR wrist position change below this (meters) is
            treated as noise and skipped -- kills hold-still jitter
        rot_alpha: EMA smoothing factor for orientation (via slerp)
        max_ang_vel: max angular speed of the commanded orientation
            (rad/s); caps per-step quaternion jumps to keep wrist joint
            velocities under the SDK limit
        rot_deadband: angle (rad) below which wrist rotation changes are
            treated as noise and skipped
        warmup_frames: initial frames to skip before capturing baseline
    """
    ident_q = np.array([0.0, 0.0, 0.0, 1.0])
    return dict(
        init_vr_pos=None,
        init_vr_R=None,
        smooth_pos=np.zeros(3),
        smooth_quat=ident_q.copy(),
        current_pos=None,
        current_quat=None,
        last_out_pos=None,
        last_out_quat=ident_q.copy(),
        ready=False,
        last_update=None,
        pos_alpha=pos_alpha,
        pos_limit=pos_limit,
        max_lin_vel=max_lin_vel,
        pos_deadband=pos_deadband,
        rot_alpha=rot_alpha,
        max_ang_vel=max_ang_vel,
        rot_deadband=rot_deadband,
        warmup_frames=warmup_frames,
        frame_count=0,
    )


def update_wrist(t, mat):
    """Update wrist tracker state from a 4x4 VR wrist matrix."""
    t["frame_count"] += 1
    if t["frame_count"] <= t["warmup_frames"]:
        return

    vr_pos = mat[:3, 3]
    vr_R = mat[:3, :3]

    if t["init_vr_pos"] is None:
        t["init_vr_pos"] = vr_pos.copy()
        t["init_vr_R"] = vr_R.copy()
        print(f"[Wrist] Baseline: pos={vr_pos}")
        return

    now = time.monotonic()

    # Delta from baseline, mapped into robot frame. EMA + velocity clip run
    # every control cycle (100Hz) even when the AVP data hasn't changed, so
    # the output interpolates smoothly between the ~30Hz AVP samples.
    dpos_vr = vr_pos - t["init_vr_pos"]
    dR_vr = vr_R @ t["init_vr_R"].T
    dpos_robot = VR_TO_ROBOT @ dpos_vr
    dR_robot = VR_TO_ROBOT @ dR_vr @ VR_TO_ROBOT.T

    # ---- Position: absolute clip -> EMA -> per-step velocity clip ----
    dpos_robot = np.clip(dpos_robot, -t["pos_limit"], t["pos_limit"])
    t["smooth_pos"] = (t["pos_alpha"] * dpos_robot
                       + (1 - t["pos_alpha"]) * t["smooth_pos"])

    target_pos = t["smooth_pos"].copy()
    if t["last_out_pos"] is None:
        t["last_out_pos"] = target_pos.copy()
    if t["last_update"] is not None:
        dt = max(now - t["last_update"], 1e-3)
        step_limit = t["max_lin_vel"] * dt
        step = target_pos - t["last_out_pos"]
        norm = np.linalg.norm(step)
        if norm > step_limit:
            target_pos = t["last_out_pos"] + step * (step_limit / norm)

    # Output-side dead-band: if the commanded step is tiny, hold the last
    # commanded pos so residual noise doesn't make the end-effector shimmer
    # when the user is trying to stay still.
    if np.linalg.norm(target_pos - t["last_out_pos"]) < t["pos_deadband"]:
        target_pos = t["last_out_pos"]
    t["last_out_pos"] = target_pos

    # ---- Rotation: slerp EMA -> per-step angular velocity clip ----
    target_q_raw = mat_to_quat(dR_robot)
    t["smooth_quat"] = quat_slerp(t["smooth_quat"], target_q_raw,
                                  t["rot_alpha"])

    target_quat = t["smooth_quat"].copy()
    if t["last_update"] is not None:
        dt = max(now - t["last_update"], 1e-3)
        ang_limit = t["max_ang_vel"] * dt
        ang_step = quat_angle(t["last_out_quat"], target_quat)
        if ang_step > ang_limit:
            frac = ang_limit / ang_step
            target_quat = quat_slerp(t["last_out_quat"], target_quat, frac)

    # Output-side rotation dead-band.
    if quat_angle(target_quat, t["last_out_quat"]) < t["rot_deadband"]:
        target_quat = t["last_out_quat"]
    t["last_out_quat"] = target_quat

    t["current_pos"] = target_pos
    t["current_quat"] = target_quat
    t["ready"] = True
    t["last_update"] = now


def extract_wrist_mat(hand_payload):
    """Pull the 4x4 wrist matrix out of a Vuer HAND_MOVE hand payload.

    Vuer's Hands component sends a flat 25 * 16 sequence of floats, wrist
    being joint index 0. Each 4x4 is Three.js column-major, so we transpose
    after reshape. Occasionally individual frames are malformed (bytes in
    place of floats); swallow those and return None.
    """
    if not hand_payload:
        return None
    try:
        arr = np.asarray(hand_payload, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 16:
        return None
    return arr[:16].reshape(4, 4).T.astype(float)


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
    p.add_argument("--pos_only", action="store_true",
                   help="Debug: track wrist position only, freeze orientation "
                        "at arm home (useful for verifying axis mapping)")
    p.add_argument("--wrist_pos_alpha", type=float, default=1.0,
                   help="EMA smoothing for wrist position (0=frozen, "
                        "1=instant). 1.0 disables EMA -- rely on SDK "
                        "control_way='filter' for smoothing instead.")
    p.add_argument("--wrist_pos_limit", type=float, default=0.5,
                   help="Max wrist position deviation from baseline (meters)")
    p.add_argument("--wrist_max_vel", type=float, default=0.3,
                   help="Max Cartesian speed of wrist target (m/s). Caps "
                        "per-step jumps to prevent joint velocity blow-ups.")
    p.add_argument("--wrist_deadband", type=float, default=0.002,
                   help="Ignore VR wrist position changes below this "
                        "(meters) -- kills hold-still jitter.")
    p.add_argument("--wrist_rot_alpha", type=float, default=1.0,
                   help="EMA (slerp) smoothing for wrist orientation. "
                        "1.0 disables EMA -- rely on SDK filter instead.")
    p.add_argument("--wrist_max_ang_vel", type=float, default=3.0,
                   help="Max angular speed of wrist target (rad/s). "
                        "Lower = safer for wrist joint velocity limits.")
    p.add_argument("--wrist_rot_deadband", type=float, default=0.02,
                   help="Ignore VR wrist rotation changes below this "
                        "(rad) -- kills hold-still orientation jitter.")
    p.add_argument("--filter_scale", type=float, default=0.7,
                   help="SDK filter scale (when control_way='filter'). "
                        "Lower = smoother but laggier. Example doc uses "
                        "0.1 (slow); 0.5-1.0 feels more responsive.")
    args = p.parse_args()

    # ---- SDK init ----
    astribot = Astribot(freq=float(args.freq))
    print(f"SDK ready (alive={astribot.is_alive})")
    astribot.set_filter_parameters(args.filter_scale, 0.5)
    print(f"[SDK] filter_scale={args.filter_scale}")
    astribot.move_to_home()

    HEAD_HOME = astribot.get_desired_cartesian_pose(
        names=[astribot.head_name], frame="world")[0]
    print(f"[SDK] HEAD_HOME={HEAD_HOME}")

    ARM_HOME = astribot.get_desired_cartesian_pose(
        names=[astribot.arm_left_name, astribot.arm_right_name], frame="world")
    LEFT_HOME, RIGHT_HOME = ARM_HOME[0], ARM_HOME[1]
    print(f"[SDK] LEFT_HOME ={LEFT_HOME}")
    print(f"[SDK] RIGHT_HOME={RIGHT_HOME}")
    if args.pos_only:
        print("[DEBUG] --pos_only: wrist orientation frozen at arm home")

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

    _wrist_kw = dict(
        pos_alpha=args.wrist_pos_alpha,
        pos_limit=args.wrist_pos_limit,
        max_lin_vel=args.wrist_max_vel,
        pos_deadband=args.wrist_deadband,
        rot_alpha=args.wrist_rot_alpha,
        max_ang_vel=args.wrist_max_ang_vel,
        rot_deadband=args.wrist_rot_deadband,
    )
    lt = make_wrist_tracker(**_wrist_kw)
    rt = make_wrist_tracker(**_wrist_kw)
    left_home_pos = np.array(LEFT_HOME[:3])
    left_home_quat = np.array(LEFT_HOME[3:7])
    right_home_pos = np.array(RIGHT_HOME[:3])
    right_home_quat = np.array(RIGHT_HOME[3:7])

    # Stop sending commands if AVP data is older than this
    HEAD_STALE_TIMEOUT = 0.5   # seconds
    WRIST_STALE_TIMEOUT = 0.5  # seconds

    # ---- Shared state ----
    lock = threading.Lock()
    tracking = {"head": None, "left_wrist": None, "right_wrist": None}

    @app.add_handler("HEAD_MOVE")
    async def on_head_evt(event, session):
        m = event.value.get("matrix")
        if m is not None:
            with lock:
                tracking["head"] = m

    @app.add_handler("HAND_MOVE")
    async def on_hand_evt(event, session):
        lh = event.value.get("left")
        rh = event.value.get("right")
        with lock:
            if lh:
                tracking["left_wrist"] = lh
            if rh:
                tracking["right_wrist"] = rh

    def _compose_arm_pose(wt, home_pos, home_quat, home_full):
        """Apply wrist tracker delta to the arm's home pose."""
        if not wt["ready"] or wt["current_pos"] is None:
            return list(home_full)
        if wt["last_update"] is not None and \
           (time.monotonic() - wt["last_update"]) > WRIST_STALE_TIMEOUT:
            return list(home_full)
        pos = home_pos + wt["current_pos"]
        if args.pos_only:
            quat = home_quat
        else:
            quat = quat_multiply(wt["current_quat"], home_quat)
        return [float(v) for v in pos] + [float(v) for v in quat]

    # ---- Control loop (head + both arms) ----
    def control_loop():
        rate = astribot_ros_middleware.Rate(args.freq)
        n, cmd_n = 0, 0
        while True:
            with lock:
                hm = tracking["head"]
                lw = tracking["left_wrist"]
                rw = tracking["right_wrist"]

            if hm is not None:
                update_head(ht, np.array(hm).reshape(4, 4).T)
            if lw is not None:
                lm = extract_wrist_mat(lw)
                if lm is not None:
                    update_wrist(lt, lm)
            if rw is not None:
                rm = extract_wrist_mat(rw)
                if rm is not None:
                    update_wrist(rt, rm)

            if n % 500 == 0:
                stale = ""
                if ht["last_update"] is not None:
                    age = time.monotonic() - ht["last_update"]
                    stale = f" age={age:.2f}s"
                print(f"[DIAG @{n}] head={'ok' if ht['ready'] else 'NONE'}"
                      f" L={'ok' if lt['ready'] else '-'}"
                      f" R={'ok' if rt['ready'] else '-'}"
                      f" p={ht['smooth_pitch']:.3f} y={ht['smooth_yaw']:.3f}"
                      f"{stale} cmds={cmd_n}")
            n += 1

            # Fail-safe: if head data is stale (AVP disconnected), skip
            # sending anything so the robot holds its last commanded pose.
            if ht["last_update"] is not None and \
               (time.monotonic() - ht["last_update"]) > HEAD_STALE_TIMEOUT:
                rate.sleep()
                continue

            if ht["ready"] and ht["current_quat"] is not None:
                final_quat = quat_multiply(ht["current_quat"], head_home_quat)
                head_pose = head_home_pos + [float(v) for v in final_quat]
            else:
                head_pose = list(HEAD_HOME)

            left_pose = _compose_arm_pose(lt, left_home_pos, left_home_quat,
                                          LEFT_HOME)
            right_pose = _compose_arm_pose(rt, right_home_pos, right_home_quat,
                                           RIGHT_HOME)

            astribot.set_cartesian_pose(
                [astribot.head_name,
                 astribot.arm_left_name,
                 astribot.arm_right_name],
                [head_pose, left_pose, right_pose],
                control_way="filter", add_default_torso=True,
                use_wbc=True)
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
