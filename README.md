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

## Project Structure

```
xhand-teleop/
├── src/
│   ├── xhand_teleop/              # Python package
│   │   ├── bridge.py              # XHandBridge: joint mapping, smoothing, EtherCAT
│   │   ├── teleoperate.py         # Right-hand teleoperation
│   │   └── teleoperate_bimanual.py # Bimanual teleoperation
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
