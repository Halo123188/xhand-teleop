"""
XHAND Right-Hand Teleoperation via Apple Vision Pro + Vuer MuJoCo WASM.

Architecture:
    Browser (Vision Pro):
        Hand tracking -> mocap bodies (auto-mapped by name) -> MuJoCo WASM IK
    Server (this script):
        ON_MUJOCO_FRAME (qpos) -> JointMapper (12 joints) -> XHandBridge -> hardware

Usage:
    python -m xhand_teleop.teleoperate --dry_run --server_url https://YOUR.ngrok.app
"""

import asyncio
import sys
from pathlib import Path

import mujoco
import numpy as np
from params_proto import proto
from vuer import Vuer, VuerSession
from vuer.events import ClientEvent
from vuer.schemas import DefaultScene, Hands, MuJoCo

from xhand_teleop.bridge import (
    XHAND_AVAILABLE, XHAND_IMPORT_ERROR, XHAND_RIGHT_JOINT_NAMES,
    XHandBridge, XHandBridgeConfig,
)

_ROOT = Path(__file__).resolve().parent.parent.parent
_MJCF_DIR = _ROOT / "xhand_mjcf"
_MJCF_PATH = str((_MJCF_DIR / "xhand_right_teleop.xml").resolve())

_RIGHT_MESHES = [
    "right_hand_link", "right_hand_ee_link", "right_hand_back_link",
    "right_hand_thumb_bend_link", "right_hand_thumb_rota_link1",
    "right_hand_thumb_rotaback_link1", "right_hand_thumb_rota_link2",
    "right_hand_thumb_rotaback_link2", "right_hand_thumb_rota_tip",
    "right_hand_index_bend_link", "right_hand_index_rota_link1",
    "right_hand_index_rotaback_link1", "right_hand_index_rota_link2",
    "right_hand_index_rotaback_link2", "right_hand_index_rota_tip",
    "right_hand_mid_link1", "right_hand_midback_link1",
    "right_hand_mid_link2", "right_hand_midback_link2", "right_hand_mid_tip",
    "right_hand_ring_link1", "right_hand_ringback_link1",
    "right_hand_ring_link2", "right_hand_ringback_link2", "right_hand_ring_tip",
    "right_hand_pinky_link1", "right_hand_pinkyback_link1",
    "right_hand_pinky_link2", "right_hand_pinkyback_link2", "right_hand_pinky_tip",
]


def run(args):
    mj_model = mujoco.MjModel.from_xml_path(args.mjcf)

    bridge = XHandBridge(
        XHandBridgeConfig(
            ethercat_interface=args.ethercat_interface,
            kp=args.kp, tor_max=args.tor_max,
            smoothing_window=args.smoothing_window,
            command_rate_hz=args.command_rate,
            dry_run=args.dry_run,
        ),
        mj_model, joint_names=XHAND_RIGHT_JOINT_NAMES,
    )

    app = Vuer(host="0.0.0.0", port=args.port, static_root=str(_ROOT))

    prefix = args.server_url.rstrip("/") + "/workspace/"
    asset_urls = [prefix + "xhand_mjcf/xhand_right_teleop.xml"] + [
        prefix + f"xhand_mjcf/meshes/right/{m}.STL" for m in _RIGHT_MESHES
    ]

    frame_n = {"n": 0}

    @app.add_handler("ON_MUJOCO_FRAME")
    async def on_frame(event: ClientEvent, session: VuerSession):
        kf = (event.value or {}).get("keyFrame")
        if not kf or kf.get("qpos") is None:
            return
        qpos = np.array(kf["qpos"], dtype=np.float64)
        frame_n["n"] += 1
        if frame_n["n"] % 100 == 1:
            print(f"[frame {frame_n['n']}] qpos({len(qpos)}): {qpos[:7]}...")
        await bridge.send_qpos(qpos)

    @app.spawn(start=True)
    async def main_loop(session: VuerSession):
        try:
            await bridge.start()
        except Exception:
            bridge._config.dry_run = True
            bridge._connected = True
            bridge._send_task = asyncio.create_task(bridge._sender_loop())

        session.set @ DefaultScene()
        session.upsert @ Hands(stream=True, key="hands", scale=args.hand_scale)
        await asyncio.sleep(1.0)
        session.upsert @ MuJoCo(
            key="xhand-right",
            src=prefix + "xhand_mjcf/xhand_right_teleop.xml",
            assets=asset_urls,
            useLights=True,
            useMocap=True,
            timeout=1000000,
        )

        try:
            while True:
                await asyncio.sleep(1.0)
        finally:
            await bridge.close()


@proto.cli
def main(
    port: int = 8012,
    mjcf: str = _MJCF_PATH,
    server_url: str = None,
    dry_run: bool = False,
    hand_scale: float = 1.25,
    ethercat_interface: str = "enp3s0",
    kp: float = 100.0, tor_max: float = 50.0,
    smoothing_window: int = 5, command_rate: float = 50.0,
):
    """XHAND right-hand teleoperation via Apple Vision Pro."""
    import types
    args = types.SimpleNamespace(**{k: v for k, v in locals().items()})

    if not XHAND_AVAILABLE and not args.dry_run:
        print(f"WARNING: xhand_control SDK not found: {XHAND_IMPORT_ERROR}")
    if not Path(args.mjcf).exists():
        sys.exit(f"ERROR: MJCF not found: {args.mjcf}")

    run(args)


if __name__ == "__main__":
    main()
