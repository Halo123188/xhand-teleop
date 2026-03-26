"""Lightweight VR teleop demo collection using Vuer + MuJoCo.

Usage (from the repo root):

    python collect_demo.py \
        --scene-dir scenes/three_block_dual_sharpa \
        --entry-file three_block_dual_sharpa.mjcf.xml \
        --actuators duo \
        --hands-scale 1.5 \
        --show-lights
"""

import asyncio
import os
import sys
import argparse
import json
from asyncio import sleep
from datetime import datetime
from os.path import join
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import yaml
import zarr

from pynput import keyboard

from utils import collect_asset_paths

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_num_list(s):
    return [*map(int, s.split(","))]


def frames_to_arrays(trajectory: list[dict]) -> dict[str, np.ndarray]:
    """Convert a trajectory (list of frame dicts) to arrays keyed by field name."""
    if not trajectory:
        return {}
    keys = trajectory[0].keys()
    return {key: np.array([f[key] for f in trajectory]) for key in keys}


def load_xml_keyframe(xml_path: str | Path, key_name: str = "robot/init_state") -> dict:
    """Load a named MuJoCo keyframe from an MJCF file."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    result: dict = {}
    keyframe_elem = root.find("keyframe")
    if keyframe_elem is None:
        return result
    for key_elem in keyframe_elem.findall("key"):
        if key_elem.get("name") == key_name:
            for attr in ["qpos", "qvel", "mocap_pos", "mocap_quat", "ctrl"]:
                val = key_elem.get(attr)
                if val:
                    result[attr] = np.fromstring(val, sep=" ")
            break
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect VR demos via Vuer + MuJoCo")

    parser.add_argument(
        "--scene-dir", type=str, dest="scene_dir", required=True,
        help="Directory containing the MJCF XML and its assets/ folder",
    )
    parser.add_argument(
        "--entry-file", type=str, dest="entry_file", default=None,
        help="MJCF file name inside --scene-dir (default: first *.mjcf.xml found)",
    )
    parser.add_argument(
        "--assets", type=str, default="assets",
        help="Assets sub-folder name inside --scene-dir (default: assets)",
    )
    parser.add_argument("--vuer-port", type=int, dest="vuer_port", default=int(os.environ.get("PORT", 8012)))
    parser.add_argument(
        "--tunnel-url", type=str, dest="tunnel_url",
        default=os.environ.get("TUNNEL_URL", os.environ.get("NGROK_URL")),
        help="Public tunnel URL (ngrok or cloudflared). Also settable via TUNNEL_URL or NGROK_URL env var.",
    )
    parser.add_argument(
        "--cloudflared", action="store_true",
        help="Auto-launch a free cloudflared tunnel (no account needed)",
    )
    parser.add_argument(
        "--asset-prefix", type=str, dest="asset_prefix", default=None,
        help="Override the full asset URL prefix (advanced; normally derived from --tunnel-url)",
    )
    parser.add_argument("--localhost", action="store_true", help="Use localhost instead of a tunnel")
    parser.add_argument("--data-dir", type=str, dest="data_dir", default="data", help="Directory for saved trajectories")
    parser.add_argument(
        "--frame-keys", type=str, dest="frame_keys",
        default="mocap_pos mocap_quat qpos qvel site_xpos site_xmat ctrl sensordata",
    )
    parser.add_argument("--clip-head", type=int, dest="clip_head", default=10, help="Frames to clip from trajectory start")
    parser.add_argument("--visible-groups", type=str, dest="visible_groups", default="0,1,2")
    parser.add_argument("--show-lights", action="store_true", dest="show_lights")
    parser.add_argument("--actuators", type=str, choices=["mono", "duo", "none"], default="mono")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reset-time", type=float, dest="reset_time", default=1.0)
    parser.add_argument("--Vuer.cors", dest="Vuer_cors", default=None)
    parser.add_argument("--hands-scale", type=float, dest="hands_scale", default=1.0)
    parser.add_argument(
        "--frame-file", type=str, dest="frame_file", default=None,
        help="Path to a .frame.yaml to use as initial keyframe (optional)",
    )

    return parser


def parse_cli_args():
    parser = build_arg_parser()
    args = parser.parse_args()

    args.scene_dir = os.path.abspath(args.scene_dir)

    # Auto-detect entry file
    if args.entry_file is None:
        candidates = sorted(Path(args.scene_dir).glob("*.mjcf.xml"))
        if not candidates:
            parser.error(f"No *.mjcf.xml found in {args.scene_dir}; pass --entry-file explicitly.")
        args.entry_file = candidates[0].name
        print(f"[auto] Using entry file: {args.entry_file}")

    # cloudflared auto-launch will be handled in main() after parse,
    # so just mark that we need to resolve the prefix later.
    args._needs_tunnel_resolve = False

    if args.localhost:
        args.asset_prefix = f"http://localhost:{args.vuer_port}/static"
    elif args.asset_prefix is not None:
        pass  # explicit override
    elif args.tunnel_url:
        args.asset_prefix = f"{args.tunnel_url.rstrip('/')}/static"
    elif args.cloudflared:
        # Will be resolved after cloudflared starts
        args._needs_tunnel_resolve = True
        args.asset_prefix = None
    else:
        parser.error(
            "Specify a tunnel: --tunnel-url URL, --cloudflared, or --localhost"
        )

    if args.asset_prefix:
        args.src = f"{args.asset_prefix}/{args.entry_file}"
    args.visible_groups = parse_num_list(args.visible_groups) if isinstance(args.visible_groups, str) else args.visible_groups
    args.init_keyframe = {}

    # Derive scene name from directory name
    args.name = Path(args.scene_dir).name

    return args


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------

keyboard_queue: asyncio.Queue = asyncio.Queue()


def setup_keyboard_listener():
    def on_press(key):
        try:
            if key.char in ("a", "b", "c"):
                keyboard_queue.put_nowait(key.char)
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _launch_cloudflared(port: int) -> str:
    """Launch cloudflared tunnel and return the public URL."""
    import subprocess, re, time, signal, atexit

    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Ensure tunnel is killed when the script exits
    def _cleanup():
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    atexit.register(_cleanup)

    # Wait for cloudflared to print the public URL
    url = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        match = re.search(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)", line)
        if match:
            url = match.group(1)
            break

    if url is None:
        proc.terminate()
        raise RuntimeError(
            "Could not start cloudflared tunnel. "
            "Install it with: brew install cloudflared  (or see https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)"
        )

    print(f"[cloudflared] Tunnel running at {url}")
    return url


def main():
    args = parse_cli_args()

    # Auto-launch cloudflared tunnel if requested
    if args.cloudflared or args._needs_tunnel_resolve:
        tunnel_url = _launch_cloudflared(args.vuer_port)
        args.tunnel_url = tunnel_url
        args.asset_prefix = f"{tunnel_url}/static"
        args.src = f"{args.asset_prefix}/{args.entry_file}"

    from vuer import Vuer, VuerSession
    from vuer.events import ClientEvent
    from vuer.schemas import (
        Box, HandActuator, Html, MuJoCo, Octahedron, Sphere, group, span, Hands,
    )

    if getattr(args, "Vuer_cors", None):
        Vuer.cors = args.Vuer_cors

    # Change into the scene directory so Vuer serves assets correctly
    os.chdir(args.scene_dir)
    vuer = Vuer(static_root=".", port=args.vuer_port)

    # Collect asset paths from the XML
    assets = collect_asset_paths(args.entry_file)
    args.asset_paths = [join(args.asset_prefix, args.assets, a) for a in assets]
    print(f"Found {len(assets)} assets in {args.entry_file}")
    if args.verbose:
        for p in args.asset_paths:
            print(f"  {p}")

    # Load initial keyframe from .frame.yaml if provided
    if args.frame_file and Path(args.frame_file).exists():
        with open(args.frame_file) as f:
            keyframes = yaml.load(f.read(), Loader=yaml.FullLoader)
        if keyframes:
            args.init_keyframe = {
                k: np.array(v)
                for k, v in keyframes[-1].items()
                if k in ("qpos", "qvel", "mocap_pos", "mocap_quat", "ctrl")
            }
            print(f"[init] Loaded keyframe from {args.frame_file}")

    # Set up data output directory
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(args.data_dir, args.name, session_stamp)
    frames_dir = os.path.join(run_dir, "frames_zarr")
    os.makedirs(frames_dir, exist_ok=True)

    # Copy the XML into the run directory for provenance
    xml_source = Path(args.entry_file)
    if xml_source.exists():
        (Path(run_dir) / xml_source.name).write_bytes(xml_source.read_bytes())

    if args.localhost:
        editor_url = f"http://localhost:{args.vuer_port}/editor"
    else:
        editor_url = f"{args.tunnel_url.rstrip('/')}/editor"
    print(f"\nOpen on headset: {editor_url}")
    print(f"Data will be saved to: {os.path.abspath(run_dir)}\n")

    # ----- MuJoCo component builder -----

    def _get_mujoco_model(
        *,
        mode: Literal["mono", "duo", "none"],
        visible=None,
        show_lights=None,
    ):
        actuators = []
        if mode in ("mono", "duo"):
            actuators.append(
                HandActuator(
                    key="pinch-on-squeeze", cond="right-squeeze",
                    value="right:thumb-tip,right:index-finger-tip",
                    offset=0.10, scale=-12, low=0, high=1, ctrlId=-1,
                ),
            )
        if mode == "duo":
            actuators.append(
                HandActuator(
                    key="left-pinch-on-squeeze", cond="left-squeeze",
                    value="left:thumb-tip,left:index-finger-tip",
                    offset=0.10, scale=-12, low=0, high=1, ctrlId=-2,
                ),
            )

        return MuJoCo(
            *actuators,
            key="default-sim",
            src=args.src,
            assets=args.asset_paths,
            frameKeys=args.frame_keys,
            pause=True,
            useLights=show_lights,
            visible=visible,
            mocapHandleSize=0.05,
            mocapHandleWireframe=True,
            fps=50,
            useDrag=False,
            **args.init_keyframe,
        )

    # ----- State -----

    IS_LOADED = False

    @vuer.add_handler("ON_CONTRIB_LOAD")
    async def on_contrib_load(event: ClientEvent, proxy: VuerSession):
        nonlocal IS_LOADED
        IS_LOADED = True
        print("ON_CONTRIB_LOAD event", event.value)

    box_state = "#23aaff"
    demo_counter = 0
    frame_stack: list[dict] = []
    b_press: list = []
    c_press: list = []

    # ----- Reset / save -----

    async def handle_reset(log_trajectory: bool, proxy: VuerSession):
        nonlocal demo_counter, frame_stack, b_press, c_press, box_state

        if log_trajectory:
            demo_counter += 1
            ep_path = os.path.join(frames_dir, f"ep_{demo_counter:05d}.zarr")

            trajectory = frame_stack[args.clip_head:]
            trajectory_arrays = frames_to_arrays(trajectory)

            root = zarr.open(ep_path, mode="w")
            root.attrs["command"] = " ".join(sys.argv)
            root.attrs["trajectory_length"] = len(trajectory)

            traj_grp = root.create_group("trajectory")
            for key, array in trajectory_arrays.items():
                chunk_shape = (min(500, len(array)),) + array.shape[1:]
                traj_grp.create_dataset(key, shape=array.shape, chunks=chunk_shape, dtype=array.dtype)
                traj_grp[key][:] = array

            root.create_dataset("b_press", data=np.array(b_press, dtype=np.float64))
            root.create_dataset("c_press", data=np.array(c_press, dtype=np.float64))

            print(f"Saved ep_{demo_counter:05d} ({len(trajectory)} frames) -> {ep_path}")

        # Reload initial keyframe from .frame.yaml if present
        frame_file = args.frame_file
        if frame_file and Path(frame_file).exists():
            with open(frame_file) as f:
                keyframes = yaml.load(f.read(), Loader=yaml.FullLoader)
            if keyframes:
                args.init_keyframe = {
                    k: np.array(v)
                    for k, v in keyframes[-1].items()
                    if k in ("qpos", "qvel", "mocap_pos", "mocap_quat", "ctrl")
                }

        proxy.upsert @ _get_mujoco_model(
            mode=args.actuators, visible=args.visible_groups, show_lights=args.show_lights,
        )

        box_state = "#FFA500"
        await sleep(args.reset_time + 0.1)
        box_state = "#54f963"

        frame_stack.clear()
        b_press.clear()
        c_press.clear()

    # ----- Event handlers -----

    @vuer.add_handler("ON_CLICK")
    async def on_click(event: ClientEvent, proxy: VuerSession):
        key = event.value["key"]
        if key == "reset-button":
            await handle_reset(log_trajectory=True, proxy=proxy)
        elif key == "delete-button":
            await handle_reset(log_trajectory=False, proxy=proxy)
        elif key == "record-button":
            yaml_str = yaml.dump([last_frame], default_flow_style=False)
            record_path = args.name + ".frame.yaml"
            with open(record_path, "a") as f:
                f.write(yaml_str)
            print(f"Saved init keyframe to {record_path}")
        print(f"Clicked: {key}  N: {demo_counter}")

    @vuer.add_handler("CAMERA_MOVE")
    async def on_camera_move(event: ClientEvent, proxy: VuerSession):
        if event.key != "ego":
            return
        mat = event.value["camera"]["matrix"]
        if frame_stack:
            frame_stack[-1]["camera_matrix"] = mat

    last_frame = None

    @vuer.add_handler("ON_MUJOCO_FRAME")
    async def on_frame(event: ClientEvent, proxy: VuerSession):
        nonlocal last_frame
        last_frame = event.value["keyFrame"]
        frame_stack.append(last_frame | {"ts": np.int64(event.ts.timestamp())})

    # ----- Keyboard listener -----

    klistener = setup_keyboard_listener()
    print("Keyboard: 'a' = save & reset | 'b'/'c' = record timestamp")

    # ----- Main loop -----

    @vuer.spawn(start=True)
    async def run(proxy: VuerSession):
        nonlocal IS_LOADED, box_state

        IS_LOADED = False

        async def process_keyboard():
            while True:
                try:
                    key = keyboard_queue.get_nowait()
                    if key == "a":
                        print("Key 'a': saving trajectory and resetting")
                        await handle_reset(log_trajectory=True, proxy=proxy)
                    elif key == "b":
                        ts = np.int64(datetime.utcnow().timestamp())
                        b_press.append(ts)
                        print(f"Key 'b': recorded timestamp {ts}")
                    elif key == "c":
                        ts = np.int64(datetime.utcnow().timestamp())
                        c_press.append(ts)
                        print(f"Key 'c': recorded timestamp {ts}")
                except asyncio.QueueEmpty:
                    pass
                await sleep(0.05)

        asyncio.create_task(process_keyboard())

        proxy.upsert @ Hands(stream=True, key="hands", scale=args.hands_scale)

        t0 = perf_counter() + 5.0
        while not IS_LOADED and perf_counter() < t0:
            print("\rWaiting for module load...", end="")
            await sleep(1.0)
        print()

        proxy.upsert @ _get_mujoco_model(
            mode=args.actuators, visible=args.visible_groups, show_lights=args.show_lights,
        )
        await sleep(1.0)

        _box_state = None
        while True:
            if _box_state and _box_state == box_state:
                await sleep(0.016)
                continue
            _box_state = box_state
            await sleep(0.016)

            proxy.upsert @ group(
                Html(span("reset pose"), key="reset-label", style={"top": 30, "width": 700, "fontSize": 20}),
                Box(args=[0.25, 0.25, 0.25], key="reset-button", material={"color": box_state}),
                key="reset-button", position=[-0.4, 1.4, -1],
            )
            proxy.upsert @ group(
                Html(span("delete traj"), key="delete-label", style={"top": 30, "width": 150, "fontSize": 20}),
                Octahedron(args=[0.15, 0], key="delete-button", material={"color": box_state}),
                key="delete-button", position=[0, 1.4, -1],
            )
            proxy.upsert @ group(
                Html(span("record initial pose"), key="record-label", style={"top": 30, "width": 150, "fontSize": 20}),
                Sphere(args=[0.1, 32, 16], key="record-button", material={"color": "red"}),
                key="record-button", position=[0.4, 1.4, -1],
            )
            proxy.upsert @ group(
                Html(span(f"Trajectory: {demo_counter}"), key="traj-label", style={"top": 30, "width": 150, "fontSize": 20}),
                key="traj-label", position=[-0.1, 1.9, -1],
            )
            await sleep(0.016)


if __name__ == "__main__":
    main()