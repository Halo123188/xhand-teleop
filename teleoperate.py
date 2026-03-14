"""
XHAND Right-Hand Teleoperation via Apple Vision Pro.

Vision Pro (ARKit) -> Vuer HAND_MOVE -> update MuJoCo mocap bodies
    -> mj_step (weld constraints solve IK) -> extract 12 joint qpos
    -> XHandBridge -> EtherCAT -> XHAND hardware

Usage:
    # Dry run (no hardware, MuJoCo viewer only):
    python teleoperate.py --dry_run

    # Real hardware:
    python teleoperate.py --ethercat_interface enp3s0

    # With cloudflare tunnel for Vision Pro:
    python teleoperate.py --dry_run --cert ~/cert.pem --key ~/key.pem
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

from params_proto import proto
from vuer import Vuer, VuerSession
from vuer.events import ClientEvent
from vuer.schemas import Hands

from xhand_bridge import (
    XHAND_AVAILABLE,
    XHAND_IMPORT_ERROR,
    XHAND_RIGHT_JOINT_NAMES,
    XHandBridge,
    XHandBridgeConfig,
)
from teleop_utils import (
    TRACKED_LANDMARKS_RIGHT,
    FINGERTIP_SITES_RIGHT,
    compute_xhand_finger_lengths,
    compute_human_finger_lengths,
    update_mocap_bodies,
    update_finger_scale_factors,
)

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_MJCF_PATH = str((_HERE / "xhand_right_teleop.xml").resolve())


def run(args):
    # -- MuJoCo --
    logger.info("Loading MuJoCo model: %s", args.mjcf)
    mj_model = mujoco.MjModel.from_xml_path(args.mjcf)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    # Compute XHAND finger lengths from default pose (for auto-scaling)
    xhand_finger_lengths = compute_xhand_finger_lengths(
        mj_model, mj_data, "xhand_right_wrist_site", FINGERTIP_SITES_RIGHT)
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
    bridge = XHandBridge(bridge_config, mj_model, joint_names=XHAND_RIGHT_JOINT_NAMES)

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
                bridge._config.dry_run = True
                bridge._connected = True
                bridge._send_task = asyncio.create_task(bridge._sender_loop())

            # Try to launch MuJoCo viewer (tracking right wrist)
            viewer = None
            try:
                viewer = mujoco.viewer.launch_passive(
                    model=mj_model, data=mj_data,
                    show_left_ui=False, show_right_ui=False,
                )
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_BODY, "xhand_right_base")
                viewer.cam.distance = 0.5
                viewer.cam.elevation = -20
                viewer.cam.azimuth = 180
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
                logger.info("MuJoCo viewer launched (tracking right wrist)")
            except Exception as e:
                logger.warning("MuJoCo viewer unavailable (%s), running headless", e)

            dt = 1.0 / args.control_rate
            frame_count = 0
            max_human_lengths = {}
            finger_scale_factors = {}

            logger.info("Teleoperation running at %.0f Hz (right hand). Ctrl+C to stop.",
                        args.control_rate)

            try:
                while True:
                    if viewer is not None and not viewer.is_running():
                        break

                    # 1. Update mocap bodies from latest VR hand data
                    poses = latest_hand_poses["right"]
                    if poses is not None:
                        human_lengths = compute_human_finger_lengths(poses, FINGERTIP_SITES_RIGHT)
                        updated = update_finger_scale_factors(
                            human_lengths, xhand_finger_lengths,
                            max_human_lengths, finger_scale_factors)
                        if updated and frame_count % 50 == 0:
                            logger.info("Scale factors: %s",
                                        {k: f"{v:.3f}" for k, v in finger_scale_factors.items()})
                        update_mocap_bodies(mj_model, mj_data, poses,
                                            finger_scale_factors, TRACKED_LANDMARKS_RIGHT)

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
    """XHAND Right-Hand Teleoperation via Apple Vision Pro."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import types
    args = types.SimpleNamespace(**{k: v for k, v in locals().items()})

    print("=" * 50)
    print("XHAND Teleoperation (Right Hand)")
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
