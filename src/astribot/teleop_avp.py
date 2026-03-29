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
    python src/astribot/teleop_avp.py --server_url https://XXXX.ngrok.app

    # Terminal 3: ROS client (on robot machine, ROS Python)
    python src/astribot/teleop_avp_ros.py --server_uri ws://localhost:8012
"""

import logging
from asyncio import sleep

from vuer import Vuer, VuerSession
from vuer.events import ClientEvent, ServerEvent
from vuer.schemas import DefaultScene, Hands, Head, ImageBackground

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

    # Camera frame from ROS client (written by handler, read by browser HUD loop)
    camera_state = {"jpeg": None}

    # Track connected Python (ROS) client session IDs
    python_sessions: set = set()

    # ---- Python client session (ROS side) ----
    # Just keeps the connection alive; no scene setup needed.

    @app.spawn(client="python")
    async def python_loop(session: VuerSession):
        ws_id = session.CURRENT_WS_ID
        python_sessions.add(ws_id)
        print(f"ROS client connected: {ws_id}")
        try:
            while True:
                await sleep(1.0)
        finally:
            python_sessions.discard(ws_id)
            print(f"ROS client disconnected: {ws_id}")

    # ---- Forward AVP tracking → all connected Python clients ----

    @app.add_handler("HAND_MOVE")
    async def on_hand_move(event: ClientEvent, session: VuerSession):
        ev = HandTrackingEvent(event.value)
        for ws_id in list(python_sessions):
            await session.vuer.send(ws_id=ws_id, event=ev)

    @app.add_handler("HEAD_MOVE")
    async def on_head_move(event: ClientEvent, session: VuerSession):
        matrix = event.value.get("matrix")
        if matrix is not None:
            ev = HeadTrackingEvent({"matrix": matrix})
            for ws_id in list(python_sessions):
                await session.vuer.send(ws_id=ws_id, event=ev)

    # ---- Receive camera frames from ROS client ----

    @app.add_handler("CAMERA_FRAME")
    async def on_camera_frame(event: ClientEvent, session: VuerSession):
        jpeg = event.value.get("jpeg")
        if jpeg is not None:
            camera_state["jpeg"] = jpeg

    # ---- Browser (AVP) session ----

    @app.spawn(start=True, client="browser")
    async def main_loop(session: VuerSession):
        print(f"[browser] client connected: {session.CURRENT_WS_ID}")
        session.set @ DefaultScene(
            Hands(stream=True, fps=30, key="hands"),
            Head(stream=True, fps=30, key="head_tracking"),
        )

        await sleep(1.0)
        print("Vuer server ready — waiting for AVP and ROS client...")

        interval = 1.0 / max(camera_fps, 1)
        while True:
            jpeg = camera_state["jpeg"]
            if jpeg is not None:
                session.upsert(
                    ImageBackground(
                        src=jpeg,
                        format="jpeg",
                        key="robot-camera",
                        interpolate=True,
                        distanceToCamera=16,
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
