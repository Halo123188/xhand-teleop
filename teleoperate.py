"""
XHAND Teleoperation via Apple Vision Pro.

Vision Pro (ARKit) -> Vuer HAND_MOVE -> update MuJoCo mocap bodies
    -> mj_step (weld constraints solve IK) -> extract 12 joint qpos
    -> XHandBridge -> EtherCAT -> XHAND hardware

Usage:
    # Dry run (no hardware, MuJoCo viewer only):
    python collect_demo.py --dry_run

    # Real hardware:
    python collect_demo.py --ethercat_interface enp3s0

    # With cloudflare tunnel for Vision Pro:
    python collect_demo.py --dry_run --cert ~/cert.pem --key ~/key.pem
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from params_proto import proto
from vuer import Vuer, VuerSession
from vuer.events import ClientEvent
from vuer.schemas import Hands

from xhand_bridge import (
    XHAND_AVAILABLE,
    XHAND_IMPORT_ERROR,
    XHandBridge,
    XHandBridgeConfig,
    clamp_to_limits,
)

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_MJCF_PATH = str((_HERE / "xhand_right_teleop.xml").resolve())


# Vuer hand landmarks (25 total, from ARKit via Vuer)
# Each landmark is a 4x4 column-major matrix = 16 floats, so 25 * 16 = 400 floats
HAND_LANDMARKS = [
    "wrist",
    "thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip",
    "index-finger-metacarpal", "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate", "index-finger-phalanx-distal", "index-finger-tip",
    "middle-finger-metacarpal", "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate", "middle-finger-phalanx-distal", "middle-finger-tip",
    "ring-finger-metacarpal", "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate", "ring-finger-phalanx-distal", "ring-finger-tip",
    "pinky-finger-metacarpal", "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate", "pinky-finger-phalanx-distal", "pinky-finger-tip",
]

# Which landmarks we track -> mocap body names in the MJCF
# Maps landmark name -> mocap body name
TRACKED_LANDMARKS = {
    "wrist": "right-wrist",
    "thumb-tip": "right-thumb-tip",
    "index-finger-tip": "right-index-finger-tip",
    "middle-finger-tip": "right-middle-finger-tip",
    "ring-finger-tip": "right-ring-finger-tip",
    "pinky-finger-tip": "right-pinky-finger-tip",
}

# Fingertip landmarks -> XHAND site names (for computing robot finger lengths)
FINGERTIP_SITES = {
    "thumb-tip": "xhand_right_thumb_tip_site",
    "index-finger-tip": "xhand_right_index_finger_tip_site",
    "middle-finger-tip": "xhand_right_middle_finger_tip_site",
    "ring-finger-tip": "xhand_right_ring_finger_tip_site",
    "pinky-finger-tip": "xhand_right_pinky_finger_tip_site",
}

# Vuer (three.js, Y-up) -> MuJoCo (Z-up) rotation
_R_VUER_TO_MUJOCO = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def extract_landmark_se3(poses_flat, landmark_name):
    """Extract a 4x4 SE3 matrix from the flat Vuer poses array.

    Vuer sends 25 landmarks * 16 floats (4x4 column-major) = 400 floats.

    Returns:
        4x4 numpy array in Vuer (Y-up) coordinates.
    """
    idx = HAND_LANDMARKS.index(landmark_name)
    mat = np.array(poses_flat[16 * idx : 16 * idx + 16]).reshape(4, 4).T
    return mat


def vuer_to_mujoco(mat4x4):
    """Convert a 4x4 SE3 matrix from Vuer (Y-up) to MuJoCo (Z-up) coordinates."""
    T = np.eye(4)
    T[:3, :3] = _R_VUER_TO_MUJOCO
    return T @ mat4x4


def compute_xhand_finger_lengths(mj_model, mj_data):
    """Compute XHAND wrist-to-fingertip distances from the model's default pose.

    Returns:
        dict mapping landmark name -> distance (meters).
    """
    # Use site positions from current (default) pose
    wrist_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "xhand_right_wrist_site")
    wrist_pos = mj_data.site_xpos[wrist_site_id].copy()

    lengths = {}
    for landmark_name, site_name in FINGERTIP_SITES.items():
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            continue
        tip_pos = mj_data.site_xpos[site_id]
        lengths[landmark_name] = np.linalg.norm(tip_pos - wrist_pos)

    return lengths


def compute_human_finger_lengths(poses_flat):
    """Compute human wrist-to-fingertip distances from a single frame of VR data.

    Returns:
        dict mapping landmark name -> distance (meters) in MuJoCo coords.
    """
    wrist_mat_mj = vuer_to_mujoco(extract_landmark_se3(poses_flat, "wrist"))
    wrist_pos = wrist_mat_mj[:3, 3]

    lengths = {}
    for landmark_name in FINGERTIP_SITES:
        mat_mj = vuer_to_mujoco(extract_landmark_se3(poses_flat, landmark_name))
        tip_pos = mat_mj[:3, 3]
        lengths[landmark_name] = np.linalg.norm(tip_pos - wrist_pos)
    return lengths


def update_mocap_bodies(mj_model, mj_data, poses_flat, finger_scale_factors):
    """Update MuJoCo mocap bodies from Vuer hand landmark data.

    Fingertip positions are scaled by a constant per-finger factor so that
    any human hand maps to full XHAND range of motion while preserving flexion.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        poses_flat: Flat array of 400 floats from Vuer HAND_MOVE event.
        finger_scale_factors: dict mapping landmark name -> scale factor
            (xhand_length / human_length), computed once at calibration.
    """
    # Get wrist position in MuJoCo coords (used as reference for scaling)
    wrist_mat_mj = vuer_to_mujoco(extract_landmark_se3(poses_flat, "wrist"))
    wrist_pos_mj = wrist_mat_mj[:3, 3]

    for landmark_name, mocap_body_name in TRACKED_LANDMARKS.items():
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
        if body_id < 0:
            continue
        mocap_id = mj_model.body_mocapid[body_id]
        if mocap_id < 0:
            continue

        # Extract SE3 from Vuer data and convert to MuJoCo coords
        mat_vuer = extract_landmark_se3(poses_flat, landmark_name)
        mat_mj = vuer_to_mujoco(mat_vuer)
        pos = mat_mj[:3, 3]

        # Scale fingertip positions relative to wrist using constant factor
        if landmark_name != "wrist" and landmark_name in finger_scale_factors:
            offset = pos - wrist_pos_mj
            pos = wrist_pos_mj + offset * finger_scale_factors[landmark_name]

        # Set position
        mj_data.mocap_pos[mocap_id] = pos

        # Set quaternion (scipy gives [x,y,z,w] but MuJoCo wants [w,x,y,z])
        quat = Rotation.from_matrix(mat_mj[:3, :3]).as_quat(scalar_first=True)
        mj_data.mocap_quat[mocap_id] = quat


def run(args):
    # -- MuJoCo --
    logger.info("Loading MuJoCo model: %s", args.mjcf)
    mj_model = mujoco.MjModel.from_xml_path(args.mjcf)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    # Compute XHAND finger lengths from default pose (for auto-scaling)
    xhand_finger_lengths = compute_xhand_finger_lengths(mj_model, mj_data)
    logger.info("XHAND finger lengths (m): %s",
                {k: f"{v:.4f}" for k, v in xhand_finger_lengths.items()})

    # -- XHAND bridge --
    bridge_config = XHandBridgeConfig(
        ethercat_interface=args.ethercat_interface,
        kp=args.kp,
        tor_max=args.tor_max,
        smoothing_window=args.smoothing_window,
        command_rate_hz=args.command_rate,
        dry_run=args.dry_run,
    )
    bridge = XHandBridge(bridge_config, mj_model)

    # -- Vuer --
    Vuer.cors = "*"
    app = Vuer(host=args.host, port=args.port, static_root=str(_HERE))
    if args.cert:
        app.cert = os.path.expanduser(args.cert)
    if args.key:
        app.key = os.path.expanduser(args.key)

    # Shared state: latest hand poses from VR
    latest_hand_poses = {"right": None}

    hand_frame_count = {"n": 0}

    @app.add_handler("HAND_MOVE")
    async def on_hand_move(event: ClientEvent, session: VuerSession):
        """Store latest VR hand data (processed in main loop)."""
        hand_data = event.value
        if not hand_data:
            return
        right = hand_data.get("right")
        if right and len(right) >= 400:
            latest_hand_poses["right"] = right
            hand_frame_count["n"] += 1
            if hand_frame_count["n"] % 100 == 1:
                logger.info("[VR] HAND_MOVE received (frame %d, %d floats)",
                            hand_frame_count["n"], len(right))

    @app.spawn(start=True)
    async def main_loop(session: VuerSession):
        try:
            # Enable hand tracking
            session.upsert @ Hands(stream=True, scale=1, key="hands")

            # Start bridge
            try:
                await bridge.start()
            except Exception as e:
                logger.error("Failed to start XHAND bridge: %s", e)
                logger.warning("Continuing without hardware — MuJoCo simulation only")
                # Force dry_run so bridge.send_qpos doesn't fail
                bridge._config.dry_run = True
                bridge._connected = True
                bridge._send_task = asyncio.create_task(bridge._sender_loop())

            # Try to launch MuJoCo viewer (optional — requires mjpython on macOS)
            viewer = None
            try:
                viewer = mujoco.viewer.launch_passive(
                    model=mj_model, data=mj_data,
                    show_left_ui=False, show_right_ui=False,
                )
                mujoco.mjv_defaultFreeCamera(mj_model, viewer.cam)
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
                logger.info("MuJoCo viewer launched")
            except Exception as e:
                logger.warning("MuJoCo viewer unavailable (%s), running headless", e)

            dt = 1.0 / args.control_rate
            frame_count = 0
            # Track max observed human finger lengths (continuously updated)
            max_human_lengths = {}
            finger_scale_factors = {}

            logger.info("Teleoperation running at %.0f Hz. Ctrl+C to stop.", args.control_rate)

            try:
                while True:
                    if viewer is not None and not viewer.is_running():
                        break

                    # 1. Update mocap bodies from latest VR hand data
                    poses = latest_hand_poses["right"]
                    if poses is not None:
                        # Update max human finger lengths if we see longer ones
                        human_lengths = compute_human_finger_lengths(poses)
                        updated = False
                        for name, length in human_lengths.items():
                            if name in xhand_finger_lengths and length > max_human_lengths.get(name, 0):
                                max_human_lengths[name] = length
                                finger_scale_factors[name] = xhand_finger_lengths[name] / length
                                updated = True
                        if updated and frame_count % 50 == 0:
                            logger.info("Scale factors: %s",
                                        {k: f"{v:.3f}" for k, v in finger_scale_factors.items()})

                        update_mocap_bodies(mj_model, mj_data, poses, finger_scale_factors)

                    # 2. Step MuJoCo (weld constraints solve IK)
                    mujoco.mj_step(mj_model, mj_data)

                    # 3. Extract qpos and send to XHAND
                    await bridge.send_qpos(mj_data.qpos)

                    # 4. Sync viewer if available
                    if viewer is not None:
                        viewer.sync()

                    # 5. Log stats periodically
                    frame_count += 1
                    if frame_count % 200 == 0:
                        stats = bridge.stats
                        # Log mocap positions
                        for lname, bname in TRACKED_LANDMARKS.items():
                            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
                            if bid >= 0:
                                mid = mj_model.body_mocapid[bid]
                                if mid >= 0:
                                    p = mj_data.mocap_pos[mid]
                                    logger.info("[MOCAP] %s: pos=[%.4f, %.4f, %.4f]",
                                                lname, p[0], p[1], p[2])

                        # Log all 12 XHAND joint qpos
                        if bridge._mapper is not None:
                            raw_joints = bridge._mapper.map(mj_data.qpos)
                            logger.info("[QPOS] raw 12 joints (rad): %s",
                                        np.array2string(raw_joints, precision=4, suppress_small=True))
                            clamped = clamp_to_limits(raw_joints)
                            logger.info("[QPOS] clamped joints (rad): %s",
                                        np.array2string(clamped, precision=4, suppress_small=True))
                            # Check if any joints are changing
                            logger.info("[QPOS] joint range: min=%.4f max=%.4f sum=%.4f",
                                        raw_joints.min(), raw_joints.max(), raw_joints.sum())

                        logger.info(
                            "[BRIDGE] Frame %d | VR frames=%d | sent=%d dropped=%d errors=%d | connected=%s",
                            frame_count, hand_frame_count["n"],
                            stats["sent"], stats["dropped"], stats["errors"],
                            bridge.connected,
                        )

                    await asyncio.sleep(dt)
            finally:
                logger.info("Shutting down...")
                if viewer is not None:
                    viewer.close()
                await bridge.close()
        except Exception as e:
            logger.exception("main_loop crashed: %s", e)
            raise


@proto.cli
def main(
    # Vuer / networking
    port: int = 8012,                       # Vuer server port
    host: str = "0.0.0.0",                  # Vuer server host
    cert: str = None,                       # TLS cert path
    key: str = None,                        # TLS key path
    # MuJoCo
    mjcf: str = _MJCF_PATH,                # Path to MJCF model
    control_rate: float = 50.0,             # Control loop rate in Hz
    # XHAND
    dry_run: bool = False,                  # No hardware (MuJoCo only)
    ethercat_interface: str = "enp3s0",     # EtherCAT NIC
    kp: float = 100.0,                      # PD position gain
    tor_max: float = 50.0,                  # Max torque (0-100)
    smoothing_window: int = 5,              # Moving avg window
    command_rate: float = 50.0,             # XHAND command rate Hz
):
    """XHAND Teleoperation via Apple Vision Pro."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build a simple namespace so run() can access args.xxx
    import types
    args = types.SimpleNamespace(**{k: v for k, v in locals().items()})

    print("=" * 50)
    print("XHAND Teleoperation")
    print("=" * 50)
    print(f"  MuJoCo:    {args.mjcf}")
    print(f"  Vuer:      {args.host}:{args.port}")
    print(f"  Hardware:  {'DRY RUN' if args.dry_run else f'EtherCAT ({args.ethercat_interface})'}")
    print(f"  Rate:      {args.control_rate} Hz")
    print(f"  Smoothing: {args.smoothing_window}-frame moving average")
    print("=" * 50)

    if not XHAND_AVAILABLE and not args.dry_run:
        print(f"\nWARNING: xhand_control SDK not found: {XHAND_IMPORT_ERROR}")
        print("Use --dry_run for testing without hardware.\n")

    if not Path(args.mjcf).exists():
        print(f"ERROR: MJCF not found: {args.mjcf}")
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
