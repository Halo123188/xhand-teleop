# XHAND Teleop

Teleoperate XHAND dexterous hands via Apple Vision Pro using browser-side MuJoCo WASM for real-time IK.

## Architecture

```
Apple Vision Pro (ARKit hand tracking)
    -> Vuer Hands component (WebXR)
    -> Browser MuJoCo WASM (mocap bodies + weld constraints solve IK)
    -> ON_MUJOCO_FRAME event (WebSocket)
    -> Server: XHandBridge (joint mapping + smoothing + safety clamping)
    -> EtherCAT -> XHAND hardware
```

## Setup

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

Requires the `xhand_control` SDK for hardware control. Without it, use `--dry_run` for visualization-only mode.

## Usage

### Right hand only

```bash
# Start ngrok tunnel
ngrok http 8012

# Run teleoperation (dry run)
python -m xhand_teleop.teleoperate --dry_run --server_url https://YOUR.ngrok.app

# Run with hardware
python -m xhand_teleop.teleoperate --server_url https://YOUR.ngrok.app --ethercat_interface enp3s0
```

### Both hands

```bash
python -m xhand_teleop.teleoperate_bimanual --dry_run --server_url https://YOUR.ngrok.app

# With hardware (separate EtherCAT NICs per hand)
python -m xhand_teleop.teleoperate_bimanual \
    --server_url https://YOUR.ngrok.app \
    --ethercat_interface_right enp3s0 \
    --ethercat_interface_left enp4s0
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--server_url` | (required) | Public URL (ngrok) for asset serving |
| `--dry_run` | `False` | Visualization only, no hardware |
| `--hand_scale` | `1.25` | Scale hand landmarks to match XHAND finger lengths |
| `--kp` | `100.0` | PD position gain |
| `--tor_max` | `50.0` | Max torque (0-100) |
| `--smoothing_window` | `5` | Moving average window for joint smoothing |
| `--command_rate` | `50.0` | Hardware command rate (Hz) |

## Astribot Teleoperation

Besides the XHAND hands, this repo can teleoperate a full **Astribot** robot
(dual arms + 2-DOF head, with the torso following via whole-body control) from
an Apple Vision Pro. The code lives in `src/astribot/` and is independent of the
XHAND pipeline — it drives the robot through the Astribot SDK, not EtherCAT.

### Architecture

```
Apple Vision Pro (ARKit head + hand tracking, WebXR)
    -> Vuer server hosted in-process (teleop_avp_ros.py)
    -> HEAD_MOVE / HAND_MOVE events (WebSocket)
        -> Head tracker   (pitch/yaw, 2-DOF, relative to baseline)
        -> Wrist trackers (both hands, 6-DOF delta from arm home)
        -> Astribot.set_cartesian_pose(head + both arms, torso via WBC)
    <- head_rgbd + torso_rgbd stacked, streamed back over WebRTC H264 as an AVP HUD
```

`teleop_avp_ros.py` is a single, self-contained process: it hosts the Vuer/WebXR
session, runs the control loop against the SDK, and pushes the camera HUD back to
the headset. (An earlier two-process relay variant, `teleop_avp.py`, forwards
tracking to a separate ROS client and is kept for reference.)

### Usage

Run on the robot (aarch64), which has the Astribot SDK and cameras:

```bash
# Source the Astribot SDK environment
source /home/astribot/fortyfive/astribot_sdk_aarch64/env.sh

# Expose the Vuer port so the Vision Pro can reach it (separate terminal)
ngrok http 8012

# Start teleop (drives the real robot — no dry-run mode)
.venv/bin/python src/astribot/teleop_avp_ros.py --server_url https://YOUR.ngrok.app
```

Then open the ngrok URL in Safari on the Apple Vision Pro and enter the immersive
session. The robot moves to home, captures a head/wrist baseline over the first
~30 frames, and then follows your head and both wrists. The stacked head+torso
camera feed shows up as a floating HUD in the headset.

Requirements: the aarch64 Astribot SDK (`astribot_sdk_aarch64`), ROS 2
(`rclpy` + `astribot_ros_middleware`), and a Vuer build with WebRTC support
(`aiortc`, `opencv`).

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--server_url` | (required) | Public ngrok HTTPS URL the AVP connects to |
| `--port` | `8012` | Vuer server port |
| `--freq` | `100` | Control loop rate (Hz) |
| `--filter_scale` | `0.7` | SDK command filter. Lower = smoother but laggier; 0.5–1.0 is responsive |

### How it works

1. **Vuer** hosts the WebXR session; the AVP streams `HEAD_MOVE` and `HAND_MOVE`
   events over WebSocket.
2. **Head** pitch/yaw are extracted from the VR head matrix as a delta from a
   captured baseline, EMA-smoothed and clamped (~46° each), then mapped to the
   robot's 2-DOF head.
3. **Wrists** are mapped 6-DOF as a pose delta from each arm's home pose (via the
   `VR_TO_ROBOT` frame remap), with absolute workspace clips, per-step velocity
   clips, and dead-bands. Final smoothing is delegated to the SDK filter.
4. Head + both arm poses are sent every cycle via
   `astribot.set_cartesian_pose(..., control_way="filter", add_default_torso=True, use_wbc=True)`;
   the torso follows through whole-body control.
5. A **fail-safe** holds the last commanded pose when AVP tracking goes stale
   (>0.5 s), so the robot never chases stale targets.
6. `head_rgbd` and `torso_rgbd` frames are stacked and streamed back to the
   headset as a **WebRTC H264** HUD.

## Project Structure

```
xhand-teleop/
├── src/
│   ├── xhand_teleop/              # Python package
│   │   ├── bridge.py              # XHandBridge: joint mapping, smoothing, EtherCAT
│   │   ├── teleoperate.py         # Right-hand teleoperation
│   │   └── teleoperate_bimanual.py # Bimanual teleoperation
│   ├── astribot/                  # Astribot arm + head teleoperation (AVP)
│   │   ├── teleop_avp_ros.py      # Single-process teleop + WebRTC camera HUD
│   │   └── teleop_avp.py          # Legacy Vuer relay server (no ROS)
│   └── scripts/                   # Development tools
│       ├── convert_urdf_to_mjcf.py # URDF -> MJCF converter
│       └── collect_demo.py        # VR demo collection
├── assets/                        # All servable assets
│   ├── xhand_mjcf/               # MuJoCo scene files + meshes
│   │   ├── meshes/
│   │   │   ├── right/            # Right hand STL files
│   │   │   └── left/             # Left hand STL files
│   │   ├── xhand_right_teleop.xml
│   │   ├── xhand_left_teleop.xml
│   │   └── xhand_bimanual_teleop.xml
│   └── xhand_urdf/               # Original XHAND URDF + meshes
└── pyproject.toml
```

## How It Works

1. **Vuer** serves a web app that the Vision Pro connects to via ngrok
2. **Hand tracking** data from ARKit populates mocap bodies in browser-side MuJoCo
3. **Weld constraints** between mocap bodies and XHAND fingertip sites drive IK
4. **MuJoCo WASM** solves joint positions in real-time in the browser
5. **`ON_MUJOCO_FRAME`** events send solved `qpos` to the Python server
6. **XHandBridge** maps MuJoCo joints to XHAND's 12 DOF, applies smoothing and safety limits, then sends commands via EtherCAT
