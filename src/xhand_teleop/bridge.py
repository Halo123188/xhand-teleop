"""
XHAND Hardware Bridge for MuJoCo Teleoperation.

Bridges MuJoCo joint positions (qpos) to the real XHAND hardware via the
XHAND Python SDK (EtherCAT). Includes smoothing filter, joint safety
clamping, and async executor for non-blocking comms.

Architecture:
    MuJoCo qpos (19) -> JointMapper (pick 12) -> JointSmoother (5-frame MA)
    -> safety clamp -> Queue(maxsize=1) -> sender thread -> xhand.send_command()

XHandBridge        – single-hand, in-process (thread-based sender).
XHandProcessBridge – one hand per subprocess; use for bimanual with separate
                     EtherCAT adapters to avoid SDK global-state conflicts.
"""

import ctypes
import logging
import multiprocessing as mp
import queue as thread_queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import the XHAND SDK
XHAND_AVAILABLE = False
XHAND_IMPORT_ERROR = None
XHandControl = None
HandCommand_t = None
try:
    from xhand_controller.xhand_control import XHandControl, HandCommand_t
    XHAND_AVAILABLE = True
except ImportError:
    try:
        import xhand_control
        XHandControl = xhand_control.XHandControl
        HandCommand_t = xhand_control.HandCommand_t
        XHAND_AVAILABLE = True
    except (ImportError, AttributeError) as e:
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
XHAND_RIGHT_JOINT_NAMES = [
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

XHAND_LEFT_JOINT_NAMES = [
    "left_hand_thumb_bend_joint",    # 0
    "left_hand_thumb_rota_joint1",   # 1
    "left_hand_thumb_rota_joint2",   # 2
    "left_hand_index_bend_joint",    # 3
    "left_hand_index_joint1",        # 4
    "left_hand_index_joint2",        # 5
    "left_hand_mid_joint1",          # 6
    "left_hand_mid_joint2",          # 7
    "left_hand_ring_joint1",         # 8
    "left_hand_ring_joint2",         # 9
    "left_hand_pinky_joint1",        # 10
    "left_hand_pinky_joint2",        # 11
]


# ---------------------------------------------------------------------------
# MuJoCo-to-XHAND joint mapping
# ---------------------------------------------------------------------------

class JointMapper:
    """Extracts 12 XHAND joint positions from full MuJoCo qpos array."""

    def __init__(self, mj_model, joint_names: list[str] | None = None):
        import mujoco

        if joint_names is None:
            joint_names = XHAND_RIGHT_JOINT_NAMES

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
    """Single-hand bridge from MuJoCo to XHAND hardware (in-process).

    Usage:
        bridge = XHandBridge(config, mj_model)
        bridge.connect_and_start()
        bridge.send_qpos(qpos)          # call each control loop tick
        await bridge.close()            # stops sender thread
    """

    def __init__(self, config: XHandBridgeConfig, mj_model=None, joint_names=None):
        self._config = config
        self._mapper = JointMapper(mj_model, joint_names) if mj_model is not None else None
        self._smoother = JointSmoother(NUM_XHAND_JOINTS, config.smoothing_window)
        self._device: Optional[object] = None
        self._device_id: Optional[int] = None
        self._connected = False
        self._stopped = False
        self._cmd_queue: thread_queue.Queue = thread_queue.Queue(maxsize=1)
        self._send_thread: Optional[threading.Thread] = None
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

    def connect_and_start(self):
        """Synchronous init: connect hardware + start sender thread.

        Call this BEFORE the asyncio event loop is busy (e.g. before Vuer starts).
        """
        self._stopped = False

        if self._config.dry_run:
            logger.info("XHAND bridge: DRY RUN mode")
            self._connected = True
        else:
            if not XHAND_AVAILABLE:
                raise RuntimeError(
                    f"xhand_control SDK not available: {XHAND_IMPORT_ERROR}\n"
                    "Install with: cd xhand_control_sdk_py && sudo pip3 install ."
                )
            if not self._connected:
                self._connect_sync()

        # Start dedicated sender thread (not on asyncio event loop)
        if self._send_thread is None or not self._send_thread.is_alive():
            self._send_thread = threading.Thread(
                target=self._sender_loop_thread, daemon=True, name="xhand-sender"
            )
            self._send_thread.start()
        logger.info("XHAND bridge started (device_id=%s)", self._device_id)

    async def start(self):
        """Async wrapper — delegates to connect_and_start."""
        self.connect_and_start()

    def _connect_sync(self, timeout: float = 5.0):
        iface = self._config.ethercat_interface
        logger.info("Opening EtherCAT on '%s'...", iface)
        self._device = XHandControl()

        # Run open_ethercat in a thread with timeout — the SDK can hang
        result = [None]
        error = [None]

        def _open():
            try:
                result[0] = self._device.open_ethercat(iface)
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_open, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            raise RuntimeError(
                f"EtherCAT open_ethercat('{iface}') timed out after {timeout}s — "
                f"check cable/device on this interface"
            )
        if error[0] is not None:
            raise RuntimeError(f"EtherCAT open failed: {error[0]}")
        rsp = result[0]
        if rsp.error_code != 0:
            raise RuntimeError(f"Failed to open EtherCAT: {rsp.error_message}")

        hand_ids = self._device.list_hands_id()
        if not hand_ids:
            raise RuntimeError("No XHAND devices found!")
        self._device_id = hand_ids[0]
        logger.info("XHAND connected: device_id=%d, available=%s", self._device_id, hand_ids)
        self._connected = True

    async def close(self, shutdown_hardware: bool = False):
        logger.info("Closing XHAND bridge sender loop...")
        self._stopped = True

        if self._send_thread and self._send_thread.is_alive():
            self._send_thread.join(timeout=1.0)

        if shutdown_hardware and self._connected and not self._config.dry_run:
            try:
                self._zero_torque_sync()
                self._device.close_device()
            except Exception as e:
                logger.error("Error during XHAND close: %s", e)
            self._connected = False

        logger.info("XHAND bridge closed. Stats: %s", self._stats)

    # -- command pipeline --

    def send_qpos(self, qpos: np.ndarray):
        """Map, smooth, clamp, and queue a MuJoCo qpos for sending.

        This is intentionally a plain (non-async) method so it executes
        immediately when called from the event loop, without yielding
        control back to asyncio.
        """
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
        except thread_queue.Empty:
            pass
        try:
            self._cmd_queue.put_nowait(clamped)
        except thread_queue.Full:
            self._stats["dropped"] += 1

    def _sender_loop_thread(self):
        """Dedicated sender thread — not on the asyncio event loop."""
        logger.info("[SENDER] sender thread started")
        while not self._stopped:
            t0 = time.monotonic()
            try:
                positions = self._cmd_queue.get(timeout=0.005)
            except thread_queue.Empty:
                continue
            t_queue = time.monotonic()

            # Rate limit
            now = time.monotonic()
            wait = self._min_send_interval - (now - self._last_send_time)
            if wait > 0:
                time.sleep(wait)
            t_rate = time.monotonic()

            try:
                self._send_sync(positions)
                self._stats["sent"] += 1
                self._last_send_time = time.monotonic()
                t_hw = time.monotonic()
                if self._stats["sent"] % 50 == 0:
                    logger.info(
                        "[TIMING] queue=%.1fms rate_wait=%.1fms hw=%.1fms total=%.1fms | sent=%d dropped=%d",
                        (t_queue - t0) * 1000, (t_rate - t_queue) * 1000,
                        (t_hw - t_rate) * 1000, (t_hw - t0) * 1000,
                        self._stats["sent"], self._stats["dropped"],
                    )
            except Exception as e:
                self._stats["errors"] += 1
                logger.error("XHAND send error: %s", e)
        logger.info("[SENDER] sender thread exited")

    # -- hardware calls (run on executor thread) --

    def _send_sync(self, positions: np.ndarray):
        if self._config.dry_run:
            if self._stats["sent"] % 200 == 0:
                logger.info("[HW-DRY] would send: %s",
                            np.array2string(positions, precision=4, suppress_small=True))
            return
        # XHAND SDK expects radians (positions are already in radians from MuJoCo)
        cmd = HandCommand_t()
        for i in range(NUM_XHAND_JOINTS):
            cmd.finger_command[i].id = i
            cmd.finger_command[i].position = float(positions[i])
            cmd.finger_command[i].kp = int(self._config.kp)
            cmd.finger_command[i].tor_max = int(self._config.tor_max)
            cmd.finger_command[i].mode = 3  # PD position control
        if self._stats["sent"] % 200 == 0:
            logger.info("[HW] sending to device %d: positions_rad=%s kp=%d tor_max=%d mode=3",
                        self._device_id,
                        np.array2string(positions, precision=4, suppress_small=True),
                        int(self._config.kp), int(self._config.tor_max))
        result = self._device.send_command(self._device_id, cmd)
        if self._stats["sent"] % 200 == 0:
            logger.info("[HW] send_command returned: error_code=%s", result.error_code)

    def _zero_torque_sync(self):
        cmd = HandCommand_t()
        for i in range(NUM_XHAND_JOINTS):
            cmd.finger_command[i].id = i
            cmd.finger_command[i].position = 0.0
            cmd.finger_command[i].kp = 0
            cmd.finger_command[i].tor_max = 0
            cmd.finger_command[i].mode = 0
        self._device.send_command(self._device_id, cmd)
        logger.warning("Zero torque sent to device %d", self._device_id)


# ---------------------------------------------------------------------------
# Process-based bridge for separate EtherCAT adapters
# ---------------------------------------------------------------------------

def _hand_worker(
    side: str,
    iface: str,
    shm_positions: mp.Array,
    shm_seq: mp.Value,
    stop_flag: mp.Value,
    ready_flag: mp.Value,
    error_msg: mp.Array,
    kp: float,
    tor_max: float,
    smoothing_window: int,
    command_rate_hz: float,
):
    """Subprocess entry point: opens EtherCAT on *iface* and sends commands.

    Reads 12 joint positions from *shm_positions* whenever *shm_seq* changes.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent handles Ctrl-C

    log = logging.getLogger(f"xhand.{side}")
    logging.basicConfig(level=logging.INFO, format=f"[{side}] %(message)s")

    # Import SDK inside subprocess (fresh global state)
    try:
        from xhand_controller.xhand_control import XHandControl, HandCommand_t
    except ImportError:
        try:
            import xhand_control
            XHandControl = xhand_control.XHandControl
            HandCommand_t = xhand_control.HandCommand_t
        except (ImportError, AttributeError) as e:
            error_msg.value = f"SDK import failed: {e}".encode()[:255]
            return

    # Open EtherCAT
    device = XHandControl()
    try:
        rsp = device.open_ethercat(iface)
        if rsp.error_code != 0:
            error_msg.value = f"open_ethercat failed: {rsp.error_message}".encode()[:255]
            return
    except Exception as e:
        error_msg.value = f"open_ethercat exception: {e}".encode()[:255]
        return

    hand_ids = device.list_hands_id()
    if not hand_ids:
        error_msg.value = b"No hands found on bus"
        return
    device_id = hand_ids[0]
    log.info("Connected: device_id=%d on %s", device_id, iface)

    ready_flag.value = 1

    smoother = JointSmoother(NUM_XHAND_JOINTS, smoothing_window)
    min_interval = 1.0 / command_rate_hz
    last_send = 0.0
    last_seq = -1
    sent = 0

    while stop_flag.value == 0:
        cur_seq = shm_seq.value
        if cur_seq == last_seq:
            time.sleep(0.001)
            continue
        last_seq = cur_seq

        # Read positions from shared memory
        with shm_positions.get_lock():
            positions = np.array(shm_positions[:], dtype=np.float64)

        smoothed = smoother.smooth(positions)
        clamped = clamp_to_limits(smoothed)

        # Rate limit
        now = time.monotonic()
        wait = min_interval - (now - last_send)
        if wait > 0:
            time.sleep(wait)

        cmd = HandCommand_t()
        for i in range(NUM_XHAND_JOINTS):
            cmd.finger_command[i].id = i
            cmd.finger_command[i].position = float(clamped[i])
            cmd.finger_command[i].kp = int(kp)
            cmd.finger_command[i].tor_max = int(tor_max)
            cmd.finger_command[i].mode = 3

        try:
            device.send_command(device_id, cmd)
            sent += 1
            last_send = time.monotonic()
            if sent % 200 == 0:
                log.info("sent=%d pos=%s", sent, np.array2string(clamped, precision=3))
        except Exception as e:
            log.error("send_command error: %s", e)

    # Shutdown: zero torque
    log.info("Stopping, sending zero torque...")
    cmd = HandCommand_t()
    for i in range(NUM_XHAND_JOINTS):
        cmd.finger_command[i].id = i
        cmd.finger_command[i].position = 0.0
        cmd.finger_command[i].kp = 0
        cmd.finger_command[i].tor_max = 0
        cmd.finger_command[i].mode = 0
    try:
        device.send_command(device_id, cmd)
    except Exception:
        pass
    try:
        device.close_device()
    except Exception:
        pass
    log.info("Worker exited (sent=%d)", sent)


class XHandProcessBridge:
    """Bridge that runs EtherCAT in a separate process.

    Provides the same send_qpos() / close() interface as XHandBridge,
    but internally writes mapped joint positions to shared memory that
    a child process reads.
    """

    def __init__(self, config: XHandBridgeConfig, mj_model=None, joint_names=None, side="hand"):
        self._config = config
        self._side = side
        self._mapper = JointMapper(mj_model, joint_names) if mj_model is not None else None
        # Shared memory: 12 doubles for positions, 1 int for sequence counter
        self._shm_positions = mp.Array(ctypes.c_double, NUM_XHAND_JOINTS)
        self._shm_seq = mp.Value(ctypes.c_long, 0)
        self._stop_flag = mp.Value(ctypes.c_int, 0)
        self._ready_flag = mp.Value(ctypes.c_int, 0)
        self._error_msg = mp.Array(ctypes.c_char, 256)
        self._process: Optional[mp.Process] = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect_and_start(self, timeout: float = 10.0):
        if self._config.dry_run:
            self._connected = True
            print(f"[BRIDGE-{self._side}] DRY RUN mode (process bridge)")
            return

        self._process = mp.Process(
            target=_hand_worker,
            args=(
                self._side,
                self._config.ethercat_interface,
                self._shm_positions,
                self._shm_seq,
                self._stop_flag,
                self._ready_flag,
                self._error_msg,
                self._config.kp,
                self._config.tor_max,
                self._config.smoothing_window,
                self._config.command_rate_hz,
            ),
            daemon=True,
            name=f"xhand-{self._side}",
        )
        self._process.start()

        # Wait for child to signal ready or fail
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._ready_flag.value == 1:
                self._connected = True
                print(f"[BRIDGE-{self._side}] Process connected on {self._config.ethercat_interface}")
                return
            if not self._process.is_alive():
                err = self._error_msg.value.decode().strip('\x00')
                raise RuntimeError(f"XHAND {self._side} worker died: {err}")
            time.sleep(0.1)

        raise RuntimeError(f"XHAND {self._side} worker timed out after {timeout}s")

    def send_qpos(self, qpos: np.ndarray):
        if not self._connected:
            return

        if self._config.dry_run:
            return

        if self._mapper is not None:
            raw = self._mapper.map(qpos)
        else:
            raw = np.asarray(qpos[:NUM_XHAND_JOINTS], dtype=np.float64)

        with self._shm_positions.get_lock():
            self._shm_positions[:] = raw.tolist()
        self._shm_seq.value += 1

    async def close(self, shutdown_hardware: bool = True):
        if self._process and self._process.is_alive():
            self._stop_flag.value = 1
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._connected = False
        print(f"[BRIDGE-{self._side}] Process bridge closed")
