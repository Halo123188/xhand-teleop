"""
Astribot Teleoperation — Vuer Server (Python 3.10+, no ROS).

Runs on any machine. Apple Vision Pro connects via Safari + ngrok.
Communicates with the ROS side via Vuer websocket (teleop_avp_ros.py).

Data flow:
    AVP (Safari/WebXR)          This script (Vuer server)         ROS client
    ──────────────────          ─────────────────────────         ──────────────
    HAND_MOVE events ────────►  Forward as ServerEvent   ──────►  WristTracker
    HEAD_MOVE events ────────►  Forward as ServerEvent   ──────►  head/torso cmd
                                                                   ▼
                                ImageBackground (HUD)    ◄──────  CAMERA_FRAME
                                                                   (JPEG bytes)

Usage:
    # Terminal 1: ngrok
    ngrok http 8012

    # Terminal 2: Vuer server (this script, Python 3.10+)
    python -m scripts.teleop_avp --server_url https://XXXX.ngrok.app

    # Terminal 3: ROS client (on robot machine, ROS Python)
    python -m scripts.teleop_avp_ros --server_uri ws://<vuer-ip>:8012
"""

import base64
import logging
import time
from asyncio import sleep

from vuer import Vuer, VuerSession
from vuer.events import ClientEvent, ServerEvent
from vuer.schemas import Scene, Hands, Head, ImageBackground

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")


# Custom ServerEvents for relaying tracking data to VuerClient (ROS side)

class HandTrackingEvent(ServerEvent):
    etype = "HAND_TRACKING"

class HeadTrackingEvent(ServerEvent):
    etype = "HEAD_TRACKING"


def main(
    port: int = 8012,
    server_url: str = None,
    camera_fps: int = 10,
):
    """Vuer relay server for Astribot teleoperation via Apple Vision Pro."""

    if server_url is None:
        import sys
        sys.exit("ERROR: --server_url is required (your ngrok HTTPS URL)")

    app = Vuer(host="0.0.0.0", port=port)

    # Camera frame from ROS client (written by handler, read by HUD loop)
    camera_state = {"jpeg": None}

    # -- Forward AVP tracking to ROS client via ServerEvent broadcast --

    @app.add_handler("HAND_MOVE")
    async def on_hand_move(event: ClientEvent, session: VuerSession):
        # Forward hand data to all websocket clients (including ROS VuerClient)
        session.send @ HandTrackingEvent(value=event.value)

    @app.add_handler("HEAD_MOVE")
    async def on_head_move(event: ClientEvent, session: VuerSession):
        matrix = event.value.get("matrix")
        if matrix is not None:
            session.send @ HeadTrackingEvent(value={"matrix": matrix})

    # -- Receive camera frames from ROS client --

    @app.add_handler("CAMERA_FRAME")
    async def on_camera_frame(event: ClientEvent, session: VuerSession):
        camera_state["jpeg"] = event.value.get("jpeg")

    # -- Main session --

    @app.spawn(start=True)
    async def main_loop(session: VuerSession):
        session.set @ Scene()
        session.upsert @ Hands(stream=True, key="hands", scale=1)
        session.upsert @ Head(stream=True, key="head_tracking", fps=30)

        await sleep(2.0)
        print("Vuer server ready — waiting for AVP and ROS client...")

        interval = 1.0 / max(camera_fps, 1)
        while True:
            # Display camera from robot as HUD
            jpeg = camera_state["jpeg"]
            if jpeg is not None and isinstance(jpeg, bytes):
                data_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
                session.upsert(
                    ImageBackground(
                        data_uri,
                        format="jpeg",
                        quality=75,
                        key="robot-camera",
                        interpolate=True,
                        fixed=True,
                        distanceToCamera=1,
                        position=[0, 0, -3],
                    ),
                    to="bgChildren",
                )

            await sleep(interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vuer server for Astribot AVP teleoperation")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--server_url", type=str, required=True,
                        help="ngrok HTTPS URL")
    parser.add_argument("--camera_fps", type=int, default=10)
    args = parser.parse_args()
    main(port=args.port, server_url=args.server_url, camera_fps=args.camera_fps)
