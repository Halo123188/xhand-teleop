
import os
import time
import logging
import numpy as np
import killport
from types import SimpleNamespace
from asyncio import sleep
from functools import partial
from multiprocessing import Process
from queue import Empty, Full
from pathlib import Path
from collections import deque

from params_proto import proto

# Set logging level to WARNING to suppress verbose gesture controller logs
logging.basicConfig(level=logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')

from vuer_envs_astribot.utils.hand_pose_utils import SimplePinchTracker
from vuer_envs_astribot.utils.pose_utils import (
    wxyz_pose_to_xyzw_pose,
    xyzw_pose_to_wxyz_pose,
    from_3js,
    get_head_yaw_relative_to_hips,
    get_hips_pitch,
    pose7d_to_pose9d,
)
from vuer_envs_astribot.utils.rotation_utils import get_head_quat_yz, get_torso_quat_y
from vuer_envs_astribot.utils.control_utils import GestureController
from vuer_envs_astribot.utils.video_utils import ros_image_callback
from vuer_envs_astribot.utils.teleop_utils import TimingStats, SharedState, TrajectoryRecorder, gaze_to_hud_pixel
from vuer_envs_astribot.utils.dexhand_utils import (
    DEXHAND_AVAILABLE,
    dexhand_control_loop,
)
from vuer import Vuer, VuerSession
from vuer.events import ClientEvent
from vuer.schemas import Box, Hands, Html, Octahedron, group, span, Body

import vuer_vision_eye_tracking
from vuer_vision_eye_tracking.tools.camera import CameraManager
from vuer_vision_eye_tracking.tools.threads import EyeImageThread
from vuer_vision_eye_tracking.eye_config import Camera as EyeCameraConfig

from vuer_vision_py import VuerVision
from vuer_vision_py.schemas import Hands as VuerVisionHands, Head as VuerVisionHead

ROSPY_AVAILABLE = False
try:
    import rospy
    ROSPY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: rospy not available: {e}")

ASTRIBOT_AVAILABLE = False
if ROSPY_AVAILABLE:
    try:
        from core.astribot_api.astribot_client import Astribot
        ASTRIBOT_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Astribot modules not available: {e}")


@proto.prefix
class Dexhand:
    """DexHand robot hand configuration."""
    enabled: bool = False  # Enable DexHand robot hand teleoperation
    right: bool = True  # Enable right DexHand
    left: bool = True  # Enable left DexHand
    host: str = "localhost"  # DexHand server host
    port: int = 8765  # DexHand server port
    no_server: bool = False  # Run DexHand in standalone mode (no server)
    control_rate: float = 240.0  # DexHand control rate in Hz
    retargeting_mode: str = "mujoco"  # DexHand retargeting mode


@proto.prefix
class ScreenPred:
    """Screen prediction / eye gaze configuration."""
    enabled: bool = False  # Enable gaze prediction (requires eye camera + model weights)
    weight: str = str(Path(vuer_vision_eye_tracking.__file__).parent.parent.parent / "weights" / "track.pt" )  # TrackNet model weights path (auto-detected from package if None)
    sim_weight: str = str(Path(vuer_vision_eye_tracking.__file__).parent.parent.parent / "weights" / "sim.pt" )  # SimNet model weights path (auto-detected from package if None)
    grid: str = "16,16"  # Grid size for classification (width,height)
    regress: bool = True  # Use regression mode instead of classification
    thr: float = 0.025  # SimNet similarity threshold


@proto.prefix
class Hud:
    """Head-Up Display configuration."""
    enabled: bool = True  # Enable robot camera HUD on Vision Pro
    camera: str = "head_rgbd"  # Camera to use for HUD feed
    fps: int = 10  # HUD frame rate
    plane_width: int = 960  # HUD plane width in SwiftUI points (must match Vision Pro side)
    plane_height: int = 540  # HUD plane height in SwiftUI points (must match Vision Pro side)
    gaze_scale: float = 1.0  # Scale yellow dot displacement from center (>1 = outward, <1 = inward)


def teleop_thread(shared_state: SharedState, use_vuer_vision=False, avp_ip=None, enable_gesture_control=True,
                   enable_hud=False, hud_camera="head_rgbd", hud_fps=10, screen_pred_cfg=None):
    """
    Teleoperation thread that handles hand tracking input.

    Args:
        shared_state: SharedState instance for fast IPC
        use_vuer_vision: If True, use VuerVision backend; if False, use Vuer backend
        avp_ip: IP address for Apple Vision Pro (required if use_vuer_vision=True)
        enable_gesture_control: If True, enable gesture-based controls
        enable_hud: If True, send robot camera frames to Vision Pro as HUD
        hud_camera: Camera name for HUD feed
        hud_fps: Target FPS for HUD
        screen_pred_cfg: Screen prediction config dict for eye tracking gaze prediction
    """
    if use_vuer_vision:
        app = VuerVision(
            avp_ip=avp_ip or '172.20.10.4',
            target_event_rate=60.0,
            enable_rate_limiting=False,
            tracking_cfg={"enabled": True, "rgb_depth": False},  # Must be enabled for HAND_MOVE events to fire
            screen_pred_cfg=screen_pred_cfg or {"enabled": False},
        )

        # Initialize gesture controller for VuerVision backend
        gesture_controller = None
        first_hand_data_received = False
        initial_head_pitch = [None]  # Baseline head pitch from first HEAD_MOVE event
        initial_head_yaw = [None]    # Baseline head yaw from first HEAD_MOVE event

        # Timing diagnostics for hand tracking data
        last_hand_data_time = [None]  # Use list to allow modification in nested function
        hand_data_intervals = []
        hand_data_count = [0]

        @app.add_handler("HAND_MOVE")
        async def handle_hand_movement(event, session):
            """Handle hand movement events from VuerVision"""
            nonlocal first_hand_data_received
            hand_data = event.value

            # Track timing of hand data arrival
            now = time.perf_counter()
            now_wall = time.time()  # Wall clock for comparing with frontend timestamp
            if last_hand_data_time[0] is not None:
                interval_ms = (now - last_hand_data_time[0]) * 1000
                hand_data_intervals.append(interval_ms)

                # Warn if big gap (>50ms = <20Hz)
                if interval_ms > 50:
                    # Check if frontend sent a timestamp
                    frontend_ts = hand_data.get("timestamp") or hand_data.get("ts") or hand_data.get("time")
                    if frontend_ts:
                        # Calculate network delay (frontend timestamp vs our receive time)
                        if frontend_ts > 1e12:  # milliseconds
                            frontend_ts_sec = frontend_ts / 1000
                        else:  # seconds
                            frontend_ts_sec = frontend_ts
                        network_delay_ms = (now_wall - frontend_ts_sec) * 1000
                        print(f"⚠️ [VuerVision] Gap: {interval_ms:.1f}ms | Network delay: {network_delay_ms:.1f}ms | Frontend ts: {frontend_ts}")
                    # else:
                    #     print(f"⚠️ [VuerVision] Hand data gap: {interval_ms:.1f}ms (no frontend timestamp)")

                # Print stats every 500 samples
                hand_data_count[0] += 1
                if hand_data_count[0] % 500 == 0:
                    recent = hand_data_intervals[-500:]
                    avg = sum(recent) / len(recent)
                    max_gap = max(recent)
                    min_gap = min(recent)
                    rate = 1000 / avg if avg > 0 else 0
                    print(f"[HandTrack] {rate:.0f}Hz avg:{avg:.1f}ms min:{min_gap:.1f}ms max:{max_gap:.1f}ms")
            last_hand_data_time[0] = now

            # Signal when first hand data is received
            if not first_hand_data_received and (hand_data.get("right") or hand_data.get("left")):
                first_hand_data_received = True
                print("✅ [VuerVision] First hand tracking data received!")
                # Log available keys to check for timestamp field
                print(f"   Available keys in hand_data: {list(hand_data.keys())}")

            if hand_data.get("right") and hand_data.get("rightState"):
                right_poses = hand_data["right"]
                right_state = hand_data["rightState"]

                # Write to shared memory (fast) - DexHand now uses this too
                shared_state.set_right_hand(right_poses, right_state)

            if hand_data.get("left") and hand_data.get("leftState"):
                left_poses = hand_data["left"]
                left_state = hand_data["leftState"]

                # Write to shared memory (fast) - DexHand now uses this too
                shared_state.set_left_hand(left_poses, left_state)

            # Update gesture controller with hand data
            if gesture_controller is not None:
                gesture_controller.update(hand_data)

        def quaternion_to_euler(qx, qy, qz, qw):
            """
            Convert quaternion to Euler angles (roll, pitch, yaw) in radians.

            Args:
                qx, qy, qz, qw: Quaternion components

            Returns:
                (roll, pitch, yaw) in radians
                - Roll: rotation around X-axis (left/right tilt)
                - Pitch: rotation around Y-axis (up/down tilt)
                - Yaw: rotation around Z-axis (left/right turn)
            """
            import math
            # Roll (X-axis rotation)
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            # Pitch (Y-axis rotation)
            sinp = 2 * (qw * qy - qz * qx)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
            else:
                pitch = math.asin(sinp)

            # Yaw (Z-axis rotation)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            return roll, pitch, yaw

        @app.add_handler("HEAD_MOVE")
        async def handle_head_movement(event, session):
            """Handle head movement events from VuerVision (Apple Vision Pro)."""
            head_data = event.value

            # Check if head data is available
            if head_data.get('matrix') is not None:
                # Store full head matrix for recording
                shared_state.set_head_matrix(head_data['matrix'])

                # Access rotation (quaternion) from the head data
                rotation = head_data.get('rotation')
                if rotation:
                    qx, qy, qz, qw = rotation

                    # Convert quaternion to Euler angles (roll, pitch, yaw)
                    roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)

                    head_pitch_raw = -roll
                    head_yaw_raw = pitch

                    # Capture initial head orientation as baseline (first HEAD_MOVE event)
                    if initial_head_pitch[0] is None:
                        initial_head_pitch[0] = head_pitch_raw
                        initial_head_yaw[0] = head_yaw_raw
                        print(f"[HeadTrack] Initial head baseline captured: pitch={head_pitch_raw:.3f}, yaw={head_yaw_raw:.3f}")

                    # Store only the relative change from initial orientation
                    # Note: For VuerVision we don't have hip tracking, so hips_pitch stays at 0
                    shared_state.head_pitch.value = head_pitch_raw - initial_head_pitch[0]
                    shared_state.head_yaw.value = head_yaw_raw - initial_head_yaw[0]
                    shared_state.body_data_ready.value = True

        @app.add_handler("GAZE_MOVE")
        async def handle_gaze_movement(event, session):
            """Handle gaze/eye tracking events from VuerVision (Apple Vision Pro)."""
            gaze_data = event.value
            if gaze_data.get("x") is not None and gaze_data.get("y") is not None:
                shared_state.set_gaze(gaze_data["x"], gaze_data["y"])

        @app.spawn(start=True)
        async def main(session):
            nonlocal gesture_controller
            session.upsert @ VuerVisionHands(
                stream=True,
                scale=1,
                key="hands",
            )

            # Register head tracking for Apple Vision Pro
            session.upsert @ VuerVisionHead(
                stream=True,
                key="head",
            )
            print("Head tracking enabled for VuerVision (Apple Vision Pro)")

            # Initialize gesture controller with custom callbacks if enabled
            if enable_gesture_control:
                print("Initializing gesture controller for VuerVision...")
                print("  🤏 Thumb-Index pinch (both hands): Toggle Pause/Resume")
                print("  ✌️  Thumb-Middle pinch (both hands): Reset")
                print("  👋 Thumb-Ring pinch (both hands): Delete trajectory")

                # Custom callbacks for our teleop system
                def on_toggle():
                    """Toggle pause/resume"""
                    current_paused = shared_state.paused.value
                    shared_state.paused.value = not current_paused
                    status = "PAUSED" if not current_paused else "RESUMED"
                    print(f"🎮 Gesture control: {status}")

                def on_reset():
                    """Reset pose and save trajectory"""
                    print("🔄 Gesture control: Reset (saving trajectory)")
                    shared_state.paused.value = True
                    shared_state.do_reset.value = True
                    shared_state.save_logs.value = True
                    # Reset head baseline so it re-calibrates on next episode
                    initial_head_pitch[0] = None
                    initial_head_yaw[0] = None

                def on_delete():
                    """Reset pose and delete trajectory"""
                    print("🗑️  Gesture control: Delete trajectory")
                    shared_state.paused.value = True
                    shared_state.do_reset.value = True
                    shared_state.save_logs.value = False
                    # Reset head baseline so it re-calibrates on next episode
                    initial_head_pitch[0] = None
                    initial_head_yaw[0] = None

                gesture_controller = GestureController(
                    on_thumb_index=on_toggle,
                    on_thumb_middle=on_reset,
                    on_thumb_ring=on_delete,
                    debounce_time=1.0,
                    hold_time=1.0,
                )
                print("✅ Gesture controller initialized for VuerVision")

            # Start HUD sender thread to push robot camera frames to Vision Pro
            if enable_hud and use_vuer_vision:
                import threading
                import io
                import cv2
                from PIL import Image as PILImage

                # Compute physical image-content size on the HUD plane.
                # Swift renders the sent image with .aspectRatio(.fit) inside
                # the HUD frame (TrackingConfig.hudPlaneWidth x hudPlaneHeight).
                # Hud.plane_width/height should represent the physical size
                # of the SwiftUI frame in millimetres (1 value = 1 mm).
                _img_send_w, _img_send_h = 640, 480
                _frame_w_mm = float(Hud.plane_width)   # physical frame width (mm)
                _frame_h_mm = float(Hud.plane_height)  # physical frame height (mm)
                _img_ratio = _img_send_w / _img_send_h
                _frame_ratio = _frame_w_mm / _frame_h_mm
                if _img_ratio < _frame_ratio:
                    # height-constrained (letterboxed sides)
                    _content_h_mm = _frame_h_mm
                    _content_w_mm = _frame_h_mm * _img_ratio
                else:
                    # width-constrained (letterboxed top/bottom)
                    _content_w_mm = _frame_w_mm
                    _content_h_mm = _frame_w_mm / _img_ratio
                _hud_width_m = _content_w_mm / 1000.0
                _hud_height_m = _content_h_mm / 1000.0
                _gaze_scale = float(Hud.gaze_scale)
                print(f"[HUD] Image content physical size: {_hud_width_m:.3f}m x {_hud_height_m:.3f}m "
                      f"(frame {_frame_w_mm:.0f}x{_frame_h_mm:.0f}mm, image {_img_send_w}x{_img_send_h})")

                def hud_sender():
                    hud_frame_count = 0
                    connected_logged = False
                    wait_count = 0
                    while True:
                        try:
                            frame = shared_state.hud_image_queue.get(timeout=1.0)
                            streamer = app.get_avp_streamer()
                            if not (streamer and streamer.protocol.is_connected()):
                                wait_count += 1
                                if wait_count % 5 == 1:  # Print every 5 attempts
                                    print(f"[HUD] Waiting for streamer... streamer={streamer is not None}, "
                                          f"connected={streamer.protocol.is_connected() if streamer else 'N/A'} "
                                          f"(attempt {wait_count})")
                                continue
                            if not connected_logged:
                                print("[HUD] Streamer connected, sending frames")
                                connected_logged = True
                            img = PILImage.fromarray(frame).resize((_img_send_w, _img_send_h))

                            # Draw yellow verification dot at projected gaze position.
                            # Read directly from the streamer attribute (same process,
                            # no IPC) so we always get the latest prediction.
                            gaze = getattr(streamer, 'latest_gaze', None)
                            if gaze is not None:
                                result = gaze_to_hud_pixel(
                                    gaze[0], gaze[1], _img_send_w, _img_send_h,
                                    hud_width_m=_hud_width_m,
                                    hud_height_m=_hud_height_m,
                                    scale=_gaze_scale,
                                )
                                if result is not None:
                                    px, py = int(result[0]), int(result[1])
                                    img_arr = np.array(img)
                                    cv2.circle(img_arr, (px, py), 4, (255, 255, 0), -1)
                                    img = PILImage.fromarray(img_arr)

                            buf = io.BytesIO()
                            img.save(buf, format='JPEG', quality=80)
                            jpeg_bytes = buf.getvalue()
                            streamer.protocol.send_camera_feed(jpeg_bytes, _img_send_w, _img_send_h, hud_camera)
                            hud_frame_count += 1
                            if hud_frame_count % 30 == 1:
                                print(f"[HUD] Sent frame #{hud_frame_count} ({len(jpeg_bytes)} bytes)")
                        except Empty:
                            continue
                        except Exception as e:
                            import traceback
                            print(f"[HUD] Error: {e}")
                            traceback.print_exc()

                hud_thread = threading.Thread(target=hud_sender, daemon=True)
                hud_thread.start()
                print(f"✅ HUD sender thread started (camera: {hud_camera}, fps: {hud_fps})")

            # Start eye camera capture thread for screen prediction (gaze dot)
            # NOTE: Heavy imports (torch, cv2) are deferred to the background thread
            # to avoid GIL contention that blocks the AVP streamer's socket connection.
            if screen_pred_cfg and screen_pred_cfg.get("enabled", False):
                import threading

                def _start_eye_camera():
                    """Open eye camera and start capture loop (runs in background thread)."""
                    # Wait for the AVP streamer to connect FIRST, then import heavy modules.
                    # Importing torch/cv2 holds the GIL and blocks the streamer's socket.connect().
                    print("[ScreenPred] Waiting for AVP streamer to connect...", flush=True)
                    streamer = None
                    for i in range(300):  # Wait up to 30 seconds
                        streamer = app.get_avp_streamer()
                        if streamer is not None:
                            print(f"[ScreenPred] ✅ Got streamer after {(i+1)*0.1:.1f}s", flush=True)
                            break
                        if i % 50 == 0 and i > 0:
                            print(f"[ScreenPred] Still waiting for streamer... ({(i+1)*0.1:.1f}s)", flush=True)
                        time.sleep(0.1)

                    if streamer is None:
                        print("❌ [ScreenPred] AVP streamer not ready after 30s, eye camera not started", flush=True)
                        return

                    # Forward each gaze prediction to shared_state so the HUD
                    # overlay (and trajectory recorder) can access the values.
                    if hasattr(streamer, 'prediction_callback'):
                        streamer.prediction_callback = lambda x, y: shared_state.set_gaze(x, y)
                        print("[ScreenPred] prediction_callback → shared_state.set_gaze installed", flush=True)

                    print(f"[ScreenPred] Opening eye camera: path={EyeCameraConfig.path}, "
                          f"size={EyeCameraConfig.width}x{EyeCameraConfig.height}", flush=True)
                    eye_camera = CameraManager(
                        camera_path=EyeCameraConfig.path,
                        width=EyeCameraConfig.width,
                        height=EyeCameraConfig.height,
                    )
                    try:
                        eye_camera.open()
                        print(f"✅ [ScreenPred] Eye camera opened successfully", flush=True)
                    except RuntimeError as e:
                        print(f"❌ [ScreenPred] Failed to open eye camera: {e}", flush=True)
                        print("   Set USB_CAMERA_PATH env var or check camera connection")
                        return

                    print(f"[ScreenPred] Streamer ready, creating EyeImageThread...", flush=True)
                    eye_events = {"save": threading.Event(), "stop": threading.Event(), "template": threading.Event()}
                    try:
                        eye_thread_worker = EyeImageThread(
                            save_path=Path("/tmp/eye_images"),
                            streamer=streamer,
                            events_dict=eye_events,
                            camera_manager=eye_camera,
                        )
                        print("✅ [ScreenPred] Eye camera capture started (feeding prediction pipeline)", flush=True)
                        eye_thread_worker.run()  # Blocking — runs in this background thread
                        print("[ScreenPred] EyeImageThread.run() returned (unexpected)", flush=True)
                    except Exception as e:
                        import traceback
                        print(f"❌ [ScreenPred] EyeImageThread error: {e}", flush=True)
                        traceback.print_exc()

                threading.Thread(target=_start_eye_camera, daemon=True, name="EyeCameraThread").start()
                print("[ScreenPred] Eye camera thread launching in background...", flush=True)

            while True:
                await sleep(1.0)
    else:
        # Vuer backend for desktop/browser
        app = Vuer(host="0.0.0.0", port=8013)
        app.cert = os.path.expanduser('~/cert.pem')
        app.key = os.path.expanduser('~/key.pem')

        # Initialize gesture controller if enabled
        gesture_controller = None
        first_hand_data_received = False

        # Timing diagnostics for hand tracking data
        last_hand_data_time = [None]  # Use list to allow modification in nested function
        hand_data_intervals = []
        hand_data_count = [0]

        @app.add_handler("HAND_MOVE")
        async def handle_hand_movement(event, session):
            """Handle hand movement events and compute mocap point and control value"""
            nonlocal first_hand_data_received
            hand_data = event.value

            # Track timing of hand data arrival
            now = time.perf_counter()
            if last_hand_data_time[0] is not None:
                interval_ms = (now - last_hand_data_time[0]) * 1000
                hand_data_intervals.append(interval_ms)

                # Warn if big gap (>50ms = <20Hz)
                if interval_ms > 50:
                    print(f"⚠️ [Vuer] Hand data gap: {interval_ms:.1f}ms")

                # Print stats every 500 samples
                hand_data_count[0] += 1
                if hand_data_count[0] % 500 == 0:
                    recent = hand_data_intervals[-500:]
                    avg = sum(recent) / len(recent)
                    max_gap = max(recent)
                    min_gap = min(recent)
                    rate = 1000 / avg if avg > 0 else 0
                    print(f"[HandTrack] {rate:.0f}Hz avg:{avg:.1f}ms min:{min_gap:.1f}ms max:{max_gap:.1f}ms")
            last_hand_data_time[0] = now

            # Signal when first hand data is received
            if not first_hand_data_received and (hand_data.get("right") or hand_data.get("left")):
                first_hand_data_received = True
                print("✅ [Vuer] First hand tracking data received!")

            if hand_data.get("right") and hand_data.get("rightState"):
                right_poses = hand_data["right"]
                right_state = hand_data["rightState"]

                # Write to shared memory (fast) - DexHand now uses this too
                shared_state.set_right_hand(right_poses, right_state)

            if hand_data.get("left") and hand_data.get("leftState"):
                left_poses = hand_data["left"]
                left_state = hand_data["leftState"]

                # Write to shared memory (fast) - DexHand now uses this too
                shared_state.set_left_hand(left_poses, left_state)

            # Update gesture controller with hand data
            if gesture_controller is not None:
                gesture_controller.update(hand_data)

        @app.add_handler("BODY_TRACKING_MOVE")
        async def handle_body_tracking(event, session):
            """Handle body tracking events from Quest 3 for head/torso control.

            Note: This handler may not receive data if body tracking is not available
            on the VR device (e.g., Quest 3 supports it, but it may not always be active).
            """
            if "head" not in event.value or "hips" not in event.value:
                return

            # Store full head matrix for recording
            shared_state.set_head_matrix(event.value["head"]["matrix"])

            head = from_3js(event.value["head"]["matrix"])
            hips = from_3js(event.value["hips"]["matrix"])

            # Calculate head pitch from local y-axis angle to global y-axis
            y_axis = head[:3, 1]  # Second column
            global_y = np.array([0, 1, 0])
            angle_y = np.arccos(np.clip(np.dot(y_axis, global_y), -1.0, 1.0))

            shared_state.head_pitch.value = angle_y - np.pi/2
            shared_state.head_yaw.value = get_head_yaw_relative_to_hips(head_mat=head, hips_mat=hips)
            shared_state.hips_pitch.value = get_hips_pitch(head_mat=head, hips_mat=hips)
            shared_state.body_data_ready.value = True

        box_state = "#23aaff"

        async def handle_reset(log_trajectory: bool, button_key: str, proxy: VuerSession):
            nonlocal box_state

            box_state = "#FFA500"

            shared_state.do_reset.value = True
            shared_state.save_logs.value = log_trajectory

            await sleep(3.0)
            box_state = "#54f963"

        @app.add_handler("ON_CLICK")
        async def on_click(event: ClientEvent, proxy: VuerSession):
            key = event.value["key"]
            if key == "reset-button":
                await handle_reset(log_trajectory=True, button_key=key, proxy=proxy)
            elif key == "delete-button":
                await handle_reset(log_trajectory=False, button_key=key, proxy=proxy)
            elif key == "pause-button":
                print("PAUSE button clicked")
                shared_state.paused.value = not shared_state.paused.value

        @app.spawn(start=True)
        async def main(session):
            nonlocal box_state, gesture_controller
            session.upsert @ Hands(
                stream=True,
                scale=1,
                key="hands",
            )

            # Add body tracking for Quest 3 (head/torso control)
            # Note: Body tracking may not be available on all VR devices or vuer versions
            session.upsert(
                Body(
                    stream=True,
                    key="bodies",
                ),
                to="bgChildren",
            )

            # Initialize gesture controller with custom callbacks if enabled
            if enable_gesture_control:
                print("Initializing gesture controller...")
                print("  🤏 Thumb-Index pinch (both hands): Toggle Pause/Resume")
                print("  ✌️  Thumb-Middle pinch (both hands): Reset")
                print("  👋 Thumb-Ring pinch (both hands): Delete trajectory")

                # Custom callbacks for our teleop system
                def on_toggle():
                    """Toggle pause/resume"""
                    current_paused = shared_state.paused.value
                    shared_state.paused.value = not current_paused
                    status = "PAUSED" if not current_paused else "RESUMED"
                    print(f"🎮 Gesture control: {status}")

                def on_reset():
                    """Reset pose and save trajectory"""
                    print("🔄 Gesture control: Reset (saving trajectory)")
                    shared_state.paused.value = True
                    shared_state.do_reset.value = True
                    shared_state.save_logs.value = True

                def on_delete():
                    """Reset pose and delete trajectory"""
                    print("🗑️  Gesture control: Delete trajectory")
                    shared_state.paused.value = True
                    shared_state.do_reset.value = True
                    shared_state.save_logs.value = False

                gesture_controller = GestureController(
                    on_thumb_index=on_toggle,
                    on_thumb_middle=on_reset,
                    on_thumb_ring=on_delete,
                    debounce_time=1.0,
                    hold_time=1.0,
                )
                print("✅ Gesture controller initialized")

            _box_state = None
            while True:
                if _box_state and _box_state == box_state:
                    await sleep(0.016)
                    continue

                _box_state = box_state

                session.upsert @ group(
                    Html(
                        span("reset pose"),
                        key="reset-label",
                        style={"top": 30, "width": 700, "fontSize": 20},
                    ),
                    Box(
                        args=[0.25, 0.25, 0.25],
                        key="reset-button",
                        material={"color": box_state},
                    ),
                    key="reset-button",
                    position=[2, 2, -1],
                )  # type: ignore

                session.upsert @ group(
                    Html(
                        span("delete traj"),
                        key="delete-label",
                        style={"top": 30, "width": 150, "fontSize": 20},
                    ),
                    Octahedron(
                        args=[0.15, 0],
                        key="delete-button",
                        material={"color": box_state},
                    ),
                    key="delete-button",
                    position=[2, 2, 0],
                )  # type: ignore

                session.upsert @ group(
                    Html(
                        span("pause"),
                        key="pause-label",
                        style={"top": 30, "width": 700, "fontSize": 20},
                    ),
                    Box(
                        args=[0.25, 0.25, 0.25],
                        key="pause-button",
                        material={"color": "red"},
                    ),
                    key="pause-button",
                    position=[2, 2, 1],
                )  # type: ignore

                await sleep(1.0)


def activate_camera(astribot, target_camera):
    astribot.activate_camera()

    cameras_stat = astribot.get_cameras_info()
    if cameras_stat[target_camera]["activate"] != True:
        total_seconds = 10
        print(f"Waiting for camera module activate for {total_seconds} seconds ", end="", flush=True)

        for _ in range(total_seconds):
            cameras_stat = astribot.get_cameras_info()
            if cameras_stat[target_camera]["activate"] == True:
                break
            print(".", end="", flush=True)

        print("\n Waiting end!")

    # get cameras activate state
    cameras_stat = astribot.get_cameras_info()
    print(f"cameras status: {cameras_stat}")

    if cameras_stat[target_camera]["activate"] != True:
        print(f"Can not activate camera {target_camera}")
        os._exit(1)


def process_pose(pose_xyzw, hand_poses, hand_state, tracker):
    if hand_poses is None or hand_state is None:
        return None, None

    wxyz_pose = xyzw_pose_to_wxyz_pose(pose_xyzw)
    data = tracker.update(
        hand_poses=hand_poses,
        hand_state=hand_state,
        lab_pose=wxyz_pose,
        align_mujoco_frame=True,
    )

    target = wxyz_pose_to_xyzw_pose(np.concatenate([data["position"], data["quaternion"]]))
    ctrl = [data["control"] * 100]

    return target.tolist(), ctrl


def astribot_control_loop(shared_state: SharedState, *, freq, enable_astribot, experiment_name,
                          local_path, log_images, filter_scale, gripper_filter_scale,
                          head_pitch, camera_names, log_freq, enable_eye_tracking, backend):
    """
    Astribot arm control loop - handles robot arm teleoperation independently.

    Args:
        shared_state: SharedState instance for fast IPC
        Remaining params are passed as keyword arguments from main().
        Prefix class params (Dexhand, Hud) are accessed directly.
    """
    if not ASTRIBOT_AVAILABLE or not enable_astribot:
        print("Astribot control loop disabled")
        return

    # Initialize trajectory recorder (egocentric-compatible format)
    session_name = experiment_name or f"session_{time.strftime('%Y%m%d-%H%M%S')}"
    recorder = TrajectoryRecorder(root_path=local_path, session_name=session_name, log_freq=log_freq)

    gripper_control = not Dexhand.enabled

    astribot = Astribot(freq=freq)

    astribot.set_filter_parameters(0.02, 0.1)

    # Create frame queues for each camera
    frame_queues = {}
    for camera_name in camera_names:
        frame_queues[camera_name] = deque(maxlen=2)

    if log_images:
        # Create ONE unified callback that will be shared by all cameras
        # (SDK bug: overwrites callback each time, so we use the same object)
        unified_callback = partial(ros_image_callback, frame_queues, None)  # None = will extract from topic

        # Activate and register all cameras with the SAME callback object
        for camera_name in camera_names:
            activate_camera(astribot, target_camera=camera_name)
            # Register same callback for each camera (creates ROS subscriber for each topic)
            astribot.register_image_callback(camera_name, "color", unified_callback, need_decode=True)

    # Start a dedicated HUD feeder thread so camera frames are sent to the
    # Vision Pro as soon as the camera is active, even before the control
    # loop unpauses (i.e. before entering immersive space).
    if Hud.enabled and log_images:
        import threading

        hud_camera_key = Hud.camera
        hud_interval = 1.0 / max(Hud.fps, 1)

        def _hud_feeder():
            while True:
                if hud_camera_key in frame_queues and len(frame_queues[hud_camera_key]) > 0:
                    try:
                        while not shared_state.hud_image_queue.empty():
                            shared_state.hud_image_queue.get_nowait()
                        shared_state.hud_image_queue.put_nowait(frame_queues[hud_camera_key][-1].copy())
                    except Exception:
                        pass
                time.sleep(hud_interval)

        threading.Thread(target=_hud_feeder, daemon=True, name="HudFeeder").start()
        print(f"[HUD] Feeder thread started (camera: {hud_camera_key}, fps: {Hud.fps})")

    astribot.move_to_home()
    init_pose = astribot.get_desired_cartesian_pose(names=[astribot.head_name, astribot.torso_name])

    head_pitch_rad = np.deg2rad(head_pitch)
    head_quat = get_head_quat_yz(angle_z=head_pitch_rad, angle_y=0.0)
    head_command = tuple(init_pose[0][:3]) + tuple(head_quat)
    astribot.move_cartesian_pose([astribot.head_name], [head_command], duration=1, use_wbc=True)

    astribot.set_filter_parameters(filter_scale, gripper_filter_scale)
    rate = rospy.Rate(freq)

    # Use vuer_vision_mode for coordinate transformation when using Apple Vision Pro
    use_vuer_vision = backend in ("vuer_vision", "vuer-vision")
    rh_tracker = SimplePinchTracker(hand="right", vuer_vision_mode=use_vuer_vision)
    lh_tracker = SimplePinchTracker(hand="left", vuer_vision_mode=use_vuer_vision)

    step = 0
    assert freq % log_freq == 0, "log_freq must evenly divide freq"
    decimation = freq // log_freq

    # Track episode state
    episode_active = False

    # Timing stats
    stats = TimingStats("Astribot", print_interval=500)

    while not rospy.is_shutdown():
        stats.tick()
        if shared_state.do_reset.value:
            # End current episode if active
            if episode_active:
                recorder.end_episode(save=shared_state.save_logs.value)
                episode_active = False
                shared_state.save_logs.value = False

            step = 0

            print("Resetting to home position")
            astribot.move_to_home(duration=0.75)
            astribot.move_cartesian_pose([astribot.head_name], [head_command], duration=1.0, use_wbc=True)
            time.sleep(1.5)
            # Reset trackers to prevent jumping back to old position
            rh_tracker.reset()
            lh_tracker.reset()

            shared_state.do_reset.value = False
            continue

        if shared_state.paused.value:
            continue

        # Start a new episode if not already active
        if not episode_active:
            recorder.start_episode()
            episode_active = True

        # Update pose references when tracker is inactive (before activation)
        t0 = time.perf_counter()
        right_pose = np.array(astribot.get_current_cartesian_pose(frame=astribot.world_frame_name)[-3])  # arm right
        left_pose = np.array(astribot.get_current_cartesian_pose(frame=astribot.world_frame_name)[-5])  # arm left
        stats.record("get_pose", time.perf_counter() - t0)

        # Read from shared memory (fast)
        t0 = time.perf_counter()
        r_poses, r_state = shared_state.get_right_hand()
        l_poses, l_state = shared_state.get_left_hand()
        stats.record("ipc_read", time.perf_counter() - t0)

        t0 = time.perf_counter()
        right_target, right_ctrl = process_pose(right_pose, r_poses, r_state, rh_tracker)
        left_target, left_ctrl = process_pose(left_pose, l_poses, l_state, lh_tracker)
        stats.record("process_pose", time.perf_counter() - t0)

        names = []
        command_list = []

        if right_target is None and right_ctrl is None and left_target is None and left_ctrl is None:
            print("No data")
            time.sleep(1.0)
            continue

        if right_target is not None and right_ctrl is not None:
            names.extend([astribot.arm_right_name])
            command_list.append(right_target)

            # Update gripper value when active, otherwise use last value
            if gripper_control:
                names.extend([astribot.effector_right_name])
                command_list.append(right_ctrl)

        if left_target is not None and left_ctrl is not None:
            names.extend([astribot.arm_left_name])
            command_list.append(left_target)

            # Update gripper value when active, otherwise use last value
            if gripper_control:
                names.extend([astribot.effector_left_name])
                command_list.append(left_ctrl)

        # Add head and torso control if body tracking data is available (Quest 3 only, Vuer backend)
        # Note: body_data_ready may be False if body tracking is not available/active on the VR device
        body_head_pitch = None
        body_head_yaw = None
        body_hips_pitch = None
        if shared_state.body_data_ready.value:
            body_head_pitch = shared_state.head_pitch.value
            body_head_yaw = shared_state.head_yaw.value
            body_hips_pitch = shared_state.hips_pitch.value

            # Clip values for safety
            body_head_pitch = np.clip(body_head_pitch, -0.8, 0.8)
            body_head_yaw = np.clip(body_head_yaw, -0.8, 0.8)
            body_hips_pitch = np.clip(body_hips_pitch, -0.8, 0.8)

            # Compute head command from body tracking (delta added to initial head pitch)
            # Scale yaw by 2x so small head turns produce larger robot head turns
            body_head_quat = get_head_quat_yz(angle_y=-body_head_yaw, angle_z=head_pitch_rad + body_head_pitch)
            body_head_command = tuple(init_pose[0][:3]) + tuple(body_head_quat)

            # Compute torso command from body tracking
            torso_quat = get_torso_quat_y(body_hips_pitch)
            torso_command = tuple(init_pose[1][:3]) + tuple(torso_quat)

            # Add head and torso to command list
            names.extend([astribot.head_name, astribot.torso_name])
            command_list.extend([body_head_command, torso_command])

        # Record data at ctrl_freq rate
        if step % decimation == 0:
            timestamp_ms = int(time.time() * 1000)

            # Convert 7D poses to 9D SO3 format (xyz + 6D rotation)
            right_arm_pose_9d = pose7d_to_pose9d(right_target)
            left_arm_pose_9d = pose7d_to_pose9d(left_target)

            # Get head angles and torso pose
            if shared_state.body_data_ready.value:
                record_head_pitch = body_head_pitch
                record_head_yaw = body_head_yaw
                torso_pose_9d = pose7d_to_pose9d(list(torso_command))
            else:
                record_head_pitch = head_pitch_rad  # Initial head pitch from config
                record_head_yaw = 0.0
                torso_pose_9d = pose7d_to_pose9d(list(init_pose[1]))  # Initial torso pose

            # Record robot actions
            recorder.record_robot_action(
                timestamp_ms=timestamp_ms,
                right_arm_pose=right_arm_pose_9d,
                left_arm_pose=left_arm_pose_9d,
                right_gripper=right_ctrl[0] if right_ctrl else None,
                left_gripper=left_ctrl[0] if left_ctrl else None,
                head_pitch=record_head_pitch,
                head_yaw=record_head_yaw,
                torso_pose=torso_pose_9d
            )

            # Record camera images
            if log_images:
                for camera_name in camera_names:
                    if len(frame_queues[camera_name]) > 0:
                        frame = frame_queues[camera_name][-1]
                        recorder.record_camera_image(camera_name, timestamp_ms, frame.copy())

            # Record eye tracking data
            if enable_eye_tracking:
                gaze_x, gaze_y = shared_state.get_gaze()
                if gaze_x is not None:
                    recorder.record_eye_tracking(timestamp_ms, gaze_x, gaze_y)

        if names:
            t0 = time.perf_counter()
            astribot.set_cartesian_pose(names, command_list, control_way="filter", use_wbc=True)
            stats.record("set_pose", time.perf_counter() - t0)

        step += 1
        rate.sleep()


@proto.cli
def main(
    freq: int = 250,  # Main loop frequency in Hz
    enable_hand_tracking: bool = True,  # Enable VR hand tracking
    enable_astribot: bool = True,  # Enable Astribot arm teleoperation
    enable_gesture_control: bool = True,  # Enable gesture-based controls
    enable_eye_tracking: bool = True,  # Enable eye tracking data recording
    backend: str = "vuer",  # Backend: "vuer" (browser) or "vuer_vision" (Apple Vision Pro)
    avp_ip: str = None,  # Apple Vision Pro IP address (required for vuer_vision backend)
    experiment_name: str = None,  # Session name for trajectory recording
    local_path: str = "/home/yanbinghan/datasets/dreamlake/teleoperation",  # Root path for trajectory data
    log_images: bool = True,  # Log camera images
    filter_scale: float = 0.02,  # Filter scale for arm control
    gripper_filter_scale: float = 0.1,  # Filter scale for gripper control
    head_pitch: int = 45,  # Initial head pitch in degrees
    camera_names: str = "head_rgbd,left_wrist_rgbd,right_wrist_rgbd",  # Comma-separated camera names  "head_rgbd,left_wrist_rgbd,right_wrist_rgbd"
    log_freq: int = 50,  # Logging/recording frequency in Hz (must divide freq evenly)
):
    """Teleoperate Astribot with gripper and optional DexHand using VR hand tracking."""

    # Parse camera_names from comma-separated string to list
    camera_names_list = [name.strip() for name in camera_names.split(",")]

    # Build dexhand config for child process (prefix classes aren't available in spawned processes)
    dexhand_config = SimpleNamespace(
        enable_dexhand=Dexhand.enabled,
        enable_dexhand_right=Dexhand.right,
        enable_dexhand_left=Dexhand.left,
        dexhand_host=Dexhand.host,
        dexhand_port=Dexhand.port,
        dexhand_no_server=Dexhand.no_server,
        dexhand_control_rate=Dexhand.control_rate,
        dexhand_retargeting_mode=Dexhand.retargeting_mode,
    )

    enable_dexhand = Dexhand.enabled and DEXHAND_AVAILABLE
    enable_astribot = enable_astribot and ASTRIBOT_AVAILABLE
    use_vuer_vision = backend in ("vuer_vision", "vuer-vision")

    print("=" * 60)
    print(f"  Hand Tracking: {'✓ Avail' if enable_hand_tracking else '✗ Disabled'}")
    if enable_hand_tracking:
        print(f"    Backend: {backend}" + (f" (AVP IP: {avp_ip})" if use_vuer_vision else ""))
    print(f"  DexHand:     {'✓ Avail' if enable_dexhand else ('✗ Not installed' if not DEXHAND_AVAILABLE else '✗ Disabled')}")
    print(f"  Astribot:    {'✓ Avail' if enable_astribot else ('✗ Not installed' if not ASTRIBOT_AVAILABLE else '✗ Disabled')}")
    print(f"  Screen Pred: {'✓ Enabled' if ScreenPred.enabled else '✗ Disabled'}")
    if enable_dexhand and not Dexhand.no_server:
        # Print DexHand server instructions if enabled
        hands_to_enable = []
        if Dexhand.left:
            hands_to_enable.append("left")
        if Dexhand.right:
            hands_to_enable.append("right")
        hands_str = " ".join(hands_to_enable)
        print("\n📋 DEXHAND SERVER SETUP:")
        print("  Start a SINGLE server for BOTH hands (they share one USB CAN device):")
        print(f"    python scripts/teleop/dexhand_server.py --hand_types={hands_str} --port={Dexhand.port}")

    processes = []
    shared_state = SharedState()  # Create shared memory state (fast IPC)

    try:
        print("=" * 60)
        if enable_hand_tracking:
            print(f"Starting hand tracking process ({backend} backend)...")
            # Build screen prediction config
            if not use_vuer_vision:
                killport.kill_ports(ports=[8013])
            screen_pred_cfg = {
                "enabled": ScreenPred.enabled,
                "weight_path": ScreenPred.weight,
                "sim_weight_path": ScreenPred.sim_weight,
                "grid": tuple([int(x.strip()) for x in ScreenPred.grid.split(",")]),
                "regress": ScreenPred.regress,
                "thr": ScreenPred.thr,
                "save_labeled_frame": False,
                "use_pos_enc": False,
                "pupil_augment": False,
            }
            p_hand_tracking = Process(target=teleop_thread, args=(shared_state, use_vuer_vision, avp_ip, enable_gesture_control, Hud.enabled, Hud.camera, Hud.fps, screen_pred_cfg))
            p_hand_tracking.start()
            processes.append(p_hand_tracking)
        else:
            print("\nHand tracking disabled")

        if enable_dexhand:
            print("Starting DexHand control process...")
            p_dexhand = Process(target=dexhand_control_loop, args=(shared_state, dexhand_config))
            p_dexhand.start()
            processes.append(p_dexhand)
        else:
            if not DEXHAND_AVAILABLE:
                print("DexHand control skipped (yanbing-hand-example not installed)")
            else:
                print("DexHand control disabled by user")

        if enable_astribot:
            print("Starting Astribot control process...")
            astribot_control_loop(shared_state,
                                  freq=freq, enable_astribot=enable_astribot,
                                  experiment_name=experiment_name, local_path=local_path,
                                  log_images=log_images, filter_scale=filter_scale,
                                  gripper_filter_scale=gripper_filter_scale, head_pitch=head_pitch,
                                  camera_names=camera_names_list, log_freq=log_freq,
                                  enable_eye_tracking=enable_eye_tracking, backend=backend)
        else:
            if not ASTRIBOT_AVAILABLE:
                print("Astribot control skipped (rospy or astribot_api not installed)")
            else:
                print("Astribot control disabled by user")

            # When Astribot is unavailable but HUD is enabled, generate fake red frames.
            if Hud.enabled and use_vuer_vision:
                print("Starting fake camera feed (red)...")
                frame = np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8)
                interval = 1.0 / max(Hud.fps, 1)
                while True:
                    while not shared_state.hud_image_queue.empty():
                        try:
                            shared_state.hud_image_queue.get_nowait()
                        except Exception:
                            break
                    try:
                        shared_state.hud_image_queue.put_nowait(frame)
                    except Full:
                        pass
                    time.sleep(interval)

            for p in processes:
                p.join()

    except KeyboardInterrupt:
        print("\nCtrl+C received, shutting down gracefully...")
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=3)
            if p.is_alive():
                p.kill()
        print("All processes stopped.")
        os._exit(0)


if __name__ == "__main__":
    main()
