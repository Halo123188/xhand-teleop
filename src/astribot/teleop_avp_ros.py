"""
Astribot Teleoperation — ROS Client.

Connects to the Vuer server (teleop_avp.py) via websocket.
Receives hand/head tracking, computes arm/head/torso commands,
drives the Astribot, and sends camera frames back for HUD.

Usage:
    python -m scripts.teleop_avp_ros --server_uri ws://<vuer-ip>:8012
"""

import io
import time
import asyncio
import threading
from collections import deque
from functools import partial

import numpy as np
from PIL import Image as PILImage
import rospy
from core.astribot_api.astribot_client import Astribot
from vuer.client import VuerClient
from vuer.events import ClientEvent


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

def get_head_quat_yz(angle_y, angle_z):
    q0 = np.array([-0.707, 0, 0, 0.707])
    qy = np.array([0, np.sin(angle_y/2), 0, np.cos(angle_y/2)])
    qz = np.array([0, 0, np.sin(angle_z/2), np.cos(angle_z/2)])
    return quat_multiply(q0, quat_multiply(qz, qy))

def get_torso_quat_y(angle):
    return quat_multiply([0,0,0,1], [0, np.sin(angle/2), 0, np.cos(angle/2)])

def from_3js(matrix):
    return np.array(matrix).reshape(4, 4).T


# ---------------------------------------------------------------------------
# WristTracker — always-on arm tracking with safety
# ---------------------------------------------------------------------------

class WristTracker:
    """Maps VR wrist position to robot arm position (relative delta).

    Safety: velocity clamping, workspace bounds, dead zone, EMA smoothing.
    """

    def __init__(self, max_velocity=0.005, dead_zone=0.002,
                 smoothing_alpha=0.3, workspace_radius=0.8):
        self.max_velocity = max_velocity
        self.dead_zone = dead_zone
        self.alpha = smoothing_alpha
        self.radius = workspace_radius
        self._base_hand_pos = None
        self._base_robot_pos = None
        self._home = None
        self._smooth = None

    def reset(self):
        self._base_hand_pos = None

    def update(self, wrist_matrix, robot_pose):
        if wrist_matrix is None or robot_pose is None:
            return None

        pos = wrist_matrix[:3, 3]
        robot_pos = np.array(robot_pose[:3])

        if self._base_hand_pos is None:
            self._base_hand_pos = pos.copy()
            self._base_robot_pos = robot_pos.copy()
            self._home = robot_pos.copy()
            self._smooth = robot_pos.copy()
            return None

        # VR Y-up to robot: (x,y,z) -> (x,-z,y)
        d = pos - self._base_hand_pos
        target = self._base_robot_pos + np.array([d[0], -d[2], d[1]])

        # Dead zone
        if np.linalg.norm(target - self._smooth) < self.dead_zone:
            return list(self._smooth)

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
        return list(self._smooth)


# ---------------------------------------------------------------------------
# HeadTracker — relative pitch/yaw from WebXR head matrix
# ---------------------------------------------------------------------------

class HeadTracker:
    def __init__(self):
        self._init_pitch = None
        self._init_yaw = None
        self.pitch = 0.0
        self.yaw = 0.0
        self.ready = False

    def update(self, mat):
        pitch_raw = np.arccos(np.clip(np.dot(mat[:3,1], [0,1,0]), -1, 1)) - np.pi/2
        yaw_raw = np.arctan2(mat[:3,2][0], mat[:3,2][2])
        if self._init_pitch is None:
            self._init_pitch, self._init_yaw = pitch_raw, yaw_raw
            print(f"[HeadTrack] Baseline: pitch={pitch_raw:.3f}, yaw={yaw_raw:.3f}")
        self.pitch = pitch_raw - self._init_pitch
        self.yaw = yaw_raw - self._init_yaw
        self.ready = True


# ---------------------------------------------------------------------------
# Camera callback
# ---------------------------------------------------------------------------

def ros_image_callback(queue, _name, _topic, msg, _w, _h, array):
    if msg.format.lower() == "jpeg":
        if len(queue) >= queue.maxlen:
            queue.popleft()
        queue.append(array)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Astribot ROS client for teleop_avp")
    p.add_argument("--server_uri", default="ws://localhost:8012")
    p.add_argument("--freq", type=int, default=250)
    p.add_argument("--head_pitch", type=int, default=45)
    p.add_argument("--filter_scale", type=float, default=0.02)
    p.add_argument("--gripper_filter_scale", type=float, default=0.1)
    p.add_argument("--camera_name", default="head_rgbd")
    p.add_argument("--camera_fps", type=int, default=10)
    p.add_argument("--max_velocity", type=float, default=0.005)
    p.add_argument("--workspace_radius", type=float, default=0.8)
    args = p.parse_args()

    # ---- Robot init ----
    astribot = Astribot(freq=args.freq)
    astribot.set_filter_parameters(0.02, 0.1)
    astribot.move_to_home()

    init_pose = astribot.get_desired_cartesian_pose(
        names=[astribot.head_name, astribot.torso_name])
    head_pitch_rad = np.deg2rad(args.head_pitch)
    head_cmd_init = tuple(init_pose[0][:3]) + tuple(
        get_head_quat_yz(angle_z=head_pitch_rad, angle_y=0.0))
    astribot.move_cartesian_pose(
        [astribot.head_name], [head_cmd_init], duration=1, use_wbc=True)
    astribot.set_filter_parameters(args.filter_scale, args.gripper_filter_scale)
    print("Astribot ready")

    # ---- Camera ----
    frame_queue = deque(maxlen=2)
    astribot.activate_camera()
    for _ in range(10):
        if astribot.get_cameras_info().get(args.camera_name, {}).get("activate"):
            break
        time.sleep(1)
    astribot.register_image_callback(
        args.camera_name, "color", partial(ros_image_callback, frame_queue, None),
        need_decode=True)

    # ---- Trackers ----
    rh = WristTracker(max_velocity=args.max_velocity, workspace_radius=args.workspace_radius)
    lh = WristTracker(max_velocity=args.max_velocity, workspace_radius=args.workspace_radius)
    head = HeadTracker()

    # ---- Shared tracking state ----
    lock = threading.Lock()
    tracking = {"r_poses": None, "l_poses": None, "head_mat": None}

    # ---- Control loop (250Hz) ----
    def control_loop():
        rate = rospy.Rate(args.freq)
        while not rospy.is_shutdown():
            with lock:
                r_poses, l_poses = tracking["r_poses"], tracking["l_poses"]
                head_mat = tracking["head_mat"]

            poses = astribot.get_current_cartesian_pose(frame=astribot.world_frame_name)
            right_arm, left_arm = list(poses[-3]), list(poses[-5])

            # Wrist matrices (joint 0 = wrist, 16 floats per joint)
            r_mat = np.array(r_poses[0:16]).reshape(4,4).T if r_poses and len(r_poses) >= 16 else None
            l_mat = np.array(l_poses[0:16]).reshape(4,4).T if l_poses and len(l_poses) >= 16 else None

            r_pos = rh.update(r_mat, right_arm)
            l_pos = lh.update(l_mat, left_arm)

            if head_mat is not None:
                head.update(from_3js(head_mat))

            names, cmds = [], []

            if r_pos is not None:
                names.append(astribot.arm_right_name)
                cmds.append(r_pos + right_arm[3:])  # position + current orientation

            if l_pos is not None:
                names.append(astribot.arm_left_name)
                cmds.append(l_pos + left_arm[3:])

            if head.ready:
                hp = float(np.clip(head.pitch, -0.8, 0.8))
                hy = float(np.clip(head.yaw, -0.8, 0.8))
                names.append(astribot.head_name)
                cmds.append(tuple(init_pose[0][:3]) + tuple(
                    get_head_quat_yz(angle_y=-hy, angle_z=head_pitch_rad + hp)))
                names.append(astribot.torso_name)
                cmds.append(tuple(init_pose[1][:3]) + tuple(get_torso_quat_y(0.0)))

            if names:
                astribot.set_cartesian_pose(names, cmds, control_way="filter", use_wbc=True)
            rate.sleep()

    threading.Thread(target=control_loop, daemon=True).start()

    # ---- Websocket bridge ----
    async def bridge():
        while not rospy.is_shutdown():
            try:
                print(f"Connecting to {args.server_uri}...")
                async with VuerClient(uri=args.server_uri) as client:
                    print("Connected!")

                    async def send_camera():
                        interval = 1.0 / max(args.camera_fps, 1)
                        while not rospy.is_shutdown():
                            if len(frame_queue) > 0:
                                img = PILImage.fromarray(frame_queue[-1][..., ::-1]).resize((640, 480))
                                buf = io.BytesIO()
                                img.save(buf, format="JPEG", quality=75)
                                await client.send(ClientEvent(
                                    etype="CAMERA_FRAME", value={"jpeg": buf.getvalue()}))
                            await asyncio.sleep(interval)

                    async def recv_tracking():
                        while not rospy.is_shutdown():
                            event = await client.recv()
                            if event is None:
                                continue
                            etype = getattr(event, "etype", None)
                            value = getattr(event, "value", None)
                            if not isinstance(value, dict):
                                continue
                            with lock:
                                if etype == "HAND_TRACKING":
                                    if value.get("right"):
                                        tracking["r_poses"] = value["right"]
                                    if value.get("left"):
                                        tracking["l_poses"] = value["left"]
                                elif etype == "HEAD_TRACKING":
                                    if value.get("matrix"):
                                        tracking["head_mat"] = value["matrix"]

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


if __name__ == "__main__":
    main()
