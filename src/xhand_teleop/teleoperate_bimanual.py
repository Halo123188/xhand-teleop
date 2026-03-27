"""
XHAND Bimanual Teleoperation via Apple Vision Pro + Vuer MuJoCo WASM.

Architecture:
    Browser (Vision Pro):
        Hand tracking -> mocap bodies (auto-mapped by name) -> MuJoCo WASM IK
    Server (this script):
        ON_MUJOCO_FRAME (qpos) -> JointMapper (12 joints per hand) -> XHandProcessBridge -> hardware

Usage:
    python -m xhand_teleop.teleoperate_bimanual --dry_run --server_url https://YOUR.ngrok.app
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
    XHAND_AVAILABLE, XHAND_IMPORT_ERROR,
    XHAND_LEFT_JOINT_NAMES, XHAND_RIGHT_JOINT_NAMES,
    XHandBridgeConfig, XHandProcessBridge,
)

_ROOT = Path(__file__).resolve().parent.parent.parent
_ASSETS_DIR = _ROOT / "assets"
_MJCF_DIR = _ASSETS_DIR / "xhand_mjcf"
_MJCF_PATH = str((_MJCF_DIR / "xhand_bimanual_teleop.xml").resolve())


def _collect_asset_urls(prefix, mjcf_name, mesh_dirs):
    """Glob STL files from mesh directories and build asset URL list."""
    urls = [prefix + f"xhand_mjcf/{mjcf_name}"]
    for mesh_dir in mesh_dirs:
        for stl in sorted((_MJCF_DIR / mesh_dir).glob("*.STL")):
            urls.append(prefix + f"xhand_mjcf/{mesh_dir}/{stl.name}")
    return urls


def run(args):
    mj_model = mujoco.MjModel.from_xml_path(args.mjcf)

    base_config = dict(
        kp=args.kp, tor_max=args.tor_max,
        smoothing_window=args.smoothing_window,
        command_rate_hz=args.command_rate,
        dry_run=args.dry_run,
    )

    bridge_right = XHandProcessBridge(
        XHandBridgeConfig(ethercat_interface=args.ethercat_interface_right, **base_config),
        mj_model, joint_names=XHAND_RIGHT_JOINT_NAMES, side="right",
    )
    bridge_left = XHandProcessBridge(
        XHandBridgeConfig(ethercat_interface=args.ethercat_interface_left, **base_config),
        mj_model, joint_names=XHAND_LEFT_JOINT_NAMES, side="left",
    )
    for side, bridge in [("right", bridge_right), ("left", bridge_left)]:
        try:
            bridge.connect_and_start()
        except Exception as e:
            print(f"WARNING: XHAND {side} init FAILED: {e} — falling back to DRY RUN")
            bridge._config.dry_run = True
            bridge.connect_and_start()

    app = Vuer(host="0.0.0.0", port=args.port, workspace=str(_ASSETS_DIR))

    prefix = args.server_url.rstrip("/") + "/workspace/"
    asset_urls = _collect_asset_urls(
        prefix, "xhand_bimanual_teleop.xml", ["meshes/right", "meshes/left"],
    )

    frame_n = {"n": 0}

    @app.add_handler("ON_MUJOCO_FRAME")
    async def on_frame(event: ClientEvent, session: VuerSession):
        kf = (event.value or {}).get("keyFrame")
        if not kf or kf.get("qpos") is None:
            return
        qpos = np.array(kf["qpos"], dtype=np.float64)
        expected = mj_model.nq
        if len(qpos) != expected:
            if frame_n["n"] == 0:
                print(f"WARNING: qpos size {len(qpos)} != expected {expected}, waiting for correct model...")
            frame_n["n"] += 1
            return
        frame_n["n"] += 1
        if frame_n["n"] % 100 == 1:
            print(f"[frame {frame_n['n']}] qpos({len(qpos)}): {qpos[:7]}...")
        bridge_right.send_qpos(qpos)
        bridge_left.send_qpos(qpos)

    @app.spawn(start=True)
    async def main_loop(session: VuerSession):
        session.set @ DefaultScene()
        session.upsert @ Hands(stream=True, key="hands", scale=args.hand_scale)
        await asyncio.sleep(1.0)
        session.upsert @ MuJoCo(
            key="xhand-bimanual",
            src=prefix + "xhand_mjcf/xhand_bimanual_teleop.xml",
            assets=asset_urls,
            useLights=True,
            useMocap=True,
            timeout=1000000,
        )

        try:
            while True:
                await asyncio.sleep(1.0)
        finally:
            await bridge_right.close()
            await bridge_left.close()


@proto.cli
def main(
    port: int = 8012,
    mjcf: str = _MJCF_PATH,
    server_url: str = None,
    dry_run: bool = False,
    hand_scale: float = 1.25,
    ethercat_interface_right: str = "enp3s0",
    ethercat_interface_left: str = "enp4s0",
    kp: float = 100.0, tor_max: float = 50.0,
    smoothing_window: int = 2, command_rate: float = 100.0,
):
    """XHAND bimanual teleoperation via Apple Vision Pro."""
    import types
    args = types.SimpleNamespace(**{k: v for k, v in locals().items()})

    if not XHAND_AVAILABLE and not args.dry_run:
        print(f"WARNING: xhand_control SDK not found: {XHAND_IMPORT_ERROR}")
    if not Path(args.mjcf).exists():
        sys.exit(f"ERROR: MJCF not found: {args.mjcf}")

    run(args)


if __name__ == "__main__":
    main()
