"""
XHAND Hardware Bridge for MuJoCo Teleoperation.

Bridges MuJoCo joint positions (qpos) to the real XHAND hardware via the
XHAND Python SDK (EtherCAT). Includes smoothing filter, joint safety
clamping, and async executor for non-blocking comms.

Architecture:
    MuJoCo qpos (19) -> JointMapper (pick 12) -> JointSmoother (5-frame MA)
    -> safety clamp -> asyncio.Queue(maxsize=1) -> executor thread -> xhand.send_command()
"""

import asyncio
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import the XHAND SDK
XHAND_AVAILABLE = False
XHAND_IMPORT_ERROR = None
try:
    import xhand_controller as xhand
    from xhand_controller import HandCommand_t
    XHAND_AVAILABLE = True
except ImportError:
    try:
        import xhand_control as xhand
        from xhand_control import HandCommand_t
        XHAND_AVAILABLE = True
    except ImportError as e:
        XHAND_IMPORT_ERROR = str(e)


# ---------------------------------------------------------------------------
# XHAND joint specification (limits from XHAND1_URDF_ver 1.3)
# ---------------------------------------------------------------------------

NUM_XHAND_JOINTS = 12

# (min_rad, max_rad) for each of the 12 joints, indexed by XHAND ID
XHAND_JOINT_LIMITS = [
    (0.0,    1.832),   # 0  thumb_bend
    (-0.698, 1.57),    # 1  thumb_rota1
    (0.0,    1.57),    # 2  thumb_rota2
    (-0.174, 0.174),   # 3  index_bend
    (0.0,    1.919),   # 4  index_joint1
    (0.0,    1.919),   # 5  index_joint2
    (0.0,    1.919),   # 6  mid_joint1
    (0.0,    1.919),   # 7  mid_joint2
    (0.0,    1.919),   # 8  ring_joint1
    (0.0,    1.919),   # 9  ring_joint2
    (0.0,    1.919),   # 10 pinky_joint1
    (0.0,    1.919),   # 11 pinky_joint2
]

# XHAND URDF joint names (1:1 with hardware IDs 0-11)
XHAND_URDF_JOINT_NAMES = [
    "right_hand_thumb_bend_joint",   # 0
    "right_hand_thumb_rota_joint1",  # 1
    "right_hand_thumb_rota_joint2",  # 2
    "right_hand_index_bend_joint",   # 3
    "right_hand_index_joint1",       # 4
    "right_hand_index_joint2",       # 5
    "right_hand_mid_joint1",         # 6
    "right_hand_mid_joint2",         # 7
    "right_hand_ring_joint1",        # 8
    "right_hand_ring_joint2",        # 9
    "right_hand_pinky_joint1",       # 10
    "right_hand_pinky_joint2",       # 11
]


# ---------------------------------------------------------------------------
# MuJoCo-to-XHAND joint mapping
# ---------------------------------------------------------------------------

class JointMapper:
    """Extracts 12 XHAND joint positions from full MuJoCo qpos array."""

    def __init__(self, mj_model, joint_names: list[str] | None = None):
        import mujoco

        if joint_names is None:
            joint_names = XHAND_URDF_JOINT_NAMES

        self._qpos_addrs: list[int] = []
        for jname in joint_names:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                raise ValueError(f"Joint '{jname}' not found in MuJoCo model")
            self._qpos_addrs.append(mj_model.jnt_qposadr[jid])

        logger.info("JointMapper: %d joints resolved", len(self._qpos_addrs))

    def map(self, qpos: np.ndarray) -> np.ndarray:
        raw = np.array([qpos[addr] for addr in self._qpos_addrs], dtype=np.float64)
        # MuJoCo's weld-constraint IK can push joint2 values negative
        # (distal phalanx bending backward). Take abs so hardware gets
        # the correct curl magnitude.
        raw = np.abs(raw)
        return raw


# ---------------------------------------------------------------------------
# Smoothing filter
# ---------------------------------------------------------------------------

class JointSmoother:
    """N-frame moving average filter for joint positions."""

    def __init__(self, num_joints: int = NUM_XHAND_JOINTS, window_size: int = 5):
        self._buffers: list[deque] = [deque(maxlen=window_size) for _ in range(num_joints)]

    def smooth(self, positions: np.ndarray) -> np.ndarray:
        smoothed = np.zeros_like(positions)
        for i, val in enumerate(positions):
            self._buffers[i].append(val)
            smoothed[i] = np.mean(self._buffers[i])
        return smoothed

    def reset(self):
        for buf in self._buffers:
            buf.clear()


# ---------------------------------------------------------------------------
# Safety clamping
# ---------------------------------------------------------------------------

def clamp_to_limits(positions: np.ndarray) -> np.ndarray:
    clamped = positions.copy()
    for i, (lo, hi) in enumerate(XHAND_JOINT_LIMITS):
        clamped[i] = np.clip(clamped[i], lo, hi)
    return clamped


# ---------------------------------------------------------------------------
# XHAND Bridge
# ---------------------------------------------------------------------------

@dataclass
class XHandBridgeConfig:
    ethercat_interface: str = "enp3s0"
    kp: float = 100.0
    tor_max: float = 50.0
    smoothing_window: int = 5
    command_rate_hz: float = 50.0
    dry_run: bool = False


class XHandBridge:
    """Async bridge from MuJoCo simulation to XHAND hardware.

    Usage:
        bridge = XHandBridge(config, mj_model)
        await bridge.start()
        await bridge.send_qpos(qpos)   # call each control loop tick
        await bridge.close()           # sends zero torque, then disconnects
    """

    def __init__(self, config: XHandBridgeConfig, mj_model=None):
        self._config = config
        self._mapper = JointMapper(mj_model) if mj_model is not None else None
        self._smoother = JointSmoother(NUM_XHAND_JOINTS, config.smoothing_window)
        self._device_id: Optional[int] = None
        self._connected = False
        self._stopped = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xhand")
        self._cmd_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._send_task: Optional[asyncio.Task] = None
        self._last_send_time = 0.0
        self._min_send_interval = 1.0 / config.command_rate_hz
        self._stats = {"sent": 0, "dropped": 0, "errors": 0}

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    # -- lifecycle --

    async def start(self):
        if self._config.dry_run:
            logger.info("XHAND bridge: DRY RUN mode")
            self._connected = True
        else:
            if not XHAND_AVAILABLE:
                raise RuntimeError(
                    f"xhand_control SDK not available: {XHAND_IMPORT_ERROR}\n"
                    "Install with: cd xhand_control_sdk_py && sudo pip3 install ."
                )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._connect_sync)

        self._send_task = asyncio.create_task(self._sender_loop())
        logger.info("XHAND bridge started (device_id=%s)", self._device_id)

    def _connect_sync(self):
        iface = self._config.ethercat_interface
        logger.info("Opening EtherCAT on '%s'...", iface)
        xhand.open_ethercat(iface)

        hand_ids = xhand.list_hands_id()
        if not hand_ids:
            raise RuntimeError("No XHAND devices found!")
        self._device_id = hand_ids[0]
        logger.info("XHAND connected: device_id=%d, available=%s", self._device_id, hand_ids)
        self._connected = True

    async def close(self):
        logger.info("Closing XHAND bridge...")
        self._stopped = True

        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass

        if self._connected and not self._config.dry_run:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(self._executor, self._zero_torque_sync)
                await loop.run_in_executor(self._executor, xhand.close_device)
            except Exception as e:
                logger.error("Error during XHAND close: %s", e)

        self._executor.shutdown(wait=False)
        self._connected = False
        logger.info("XHAND bridge closed. Stats: %s", self._stats)

    # -- command pipeline --

    async def send_qpos(self, qpos: np.ndarray):
        """Map, smooth, clamp, and queue a MuJoCo qpos for sending."""
        if self._stopped or not self._connected:
            return

        if self._mapper is not None:
            raw = self._mapper.map(qpos)
        else:
            raw = np.asarray(qpos[:NUM_XHAND_JOINTS], dtype=np.float64)

        smoothed = self._smoother.smooth(raw)
        clamped = clamp_to_limits(smoothed)

        if self._stats["sent"] % 200 == 0:
            logger.info("[BRIDGE] raw:      %s", np.array2string(raw, precision=4, suppress_small=True))
            logger.info("[BRIDGE] smoothed: %s", np.array2string(smoothed, precision=4, suppress_small=True))
            logger.info("[BRIDGE] clamped:  %s", np.array2string(clamped, precision=4, suppress_small=True))

        # Drop stale, keep only freshest command
        try:
            self._cmd_queue.get_nowait()
            self._stats["dropped"] += 1
        except asyncio.QueueEmpty:
            pass
        try:
            self._cmd_queue.put_nowait(clamped)
        except asyncio.QueueFull:
            self._stats["dropped"] += 1

    async def _sender_loop(self):
        loop = asyncio.get_event_loop()
        while not self._stopped:
            try:
                positions = await asyncio.wait_for(self._cmd_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            # Rate limit
            now = time.monotonic()
            wait = self._min_send_interval - (now - self._last_send_time)
            if wait > 0:
                await asyncio.sleep(wait)

            try:
                await loop.run_in_executor(self._executor, self._send_sync, positions)
                self._stats["sent"] += 1
                self._last_send_time = time.monotonic()
            except Exception as e:
                self._stats["errors"] += 1
                logger.error("XHAND send error: %s", e)

    # -- hardware calls (run on executor thread) --

    def _send_sync(self, positions: np.ndarray):
        if self._config.dry_run:
            if self._stats["sent"] % 200 == 0:
                logger.info("[HW-DRY] would send: %s",
                            np.array2string(positions, precision=4, suppress_small=True))
            return
        # Convert radians to degrees — XHAND SDK expects degrees
        positions_deg = np.degrees(positions)
        cmd = HandCommand_t()
        for i in range(NUM_XHAND_JOINTS):
            cmd.finger_command[i].id = i
            cmd.finger_command[i].position = float(positions_deg[i])
            cmd.finger_command[i].kp = self._config.kp
            cmd.finger_command[i].tor_max = self._config.tor_max
            cmd.finger_command[i].mode = 3  # PD position control
        if self._stats["sent"] % 200 == 0:
            logger.info("[HW] sending to device %d: positions_deg=%s kp=%.1f tor_max=%.1f mode=3",
                        self._device_id,
                        np.array2string(positions_deg, precision=2, suppress_small=True),
                        self._config.kp, self._config.tor_max)
        result = xhand.send_command(self._device_id, cmd)
        if self._stats["sent"] % 200 == 0:
            logger.info("[HW] send_command returned: %s", result)

    def _zero_torque_sync(self):
        cmd = HandCommand_t()
        for i in range(NUM_XHAND_JOINTS):
            cmd.finger_command[i].id = i
            cmd.finger_command[i].position = 0.0
            cmd.finger_command[i].kp = 0.0
            cmd.finger_command[i].tor_max = 0.0
            cmd.finger_command[i].mode = 0
        xhand.send_command(self._device_id, cmd)
        logger.warning("Zero torque sent to device %d", self._device_id)
