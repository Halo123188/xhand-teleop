"""
Shared utilities for XHAND teleoperation.

Contains Vuer hand landmark parsing, coordinate transforms,
finger length computation, and mocap body updates.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# Vuer hand landmarks (25 total, from ARKit via Vuer)
# Each landmark is a 4x4 column-major matrix = 16 floats, so 25 * 16 = 400 floats
HAND_LANDMARKS = [
    "wrist",
    "thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip",
    "index-finger-metacarpal", "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate", "index-finger-phalanx-distal", "index-finger-tip",
    "middle-finger-metacarpal", "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate", "middle-finger-phalanx-distal", "middle-finger-tip",
    "ring-finger-metacarpal", "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate", "ring-finger-phalanx-distal", "ring-finger-tip",
    "pinky-finger-metacarpal", "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate", "pinky-finger-phalanx-distal", "pinky-finger-tip",
]

# Which landmarks we track -> mocap body names (per side)
TRACKED_LANDMARKS_RIGHT = {
    "wrist": "right-wrist",
    "thumb-tip": "right-thumb-tip",
    "index-finger-tip": "right-index-finger-tip",
    "middle-finger-tip": "right-middle-finger-tip",
    "ring-finger-tip": "right-ring-finger-tip",
    "pinky-finger-tip": "right-pinky-finger-tip",
}

TRACKED_LANDMARKS_LEFT = {
    "wrist": "left-wrist",
    "thumb-tip": "left-thumb-tip",
    "index-finger-tip": "left-index-finger-tip",
    "middle-finger-tip": "left-middle-finger-tip",
    "ring-finger-tip": "left-ring-finger-tip",
    "pinky-finger-tip": "left-pinky-finger-tip",
}

# Fingertip landmarks -> XHAND site names (for computing robot finger lengths)
FINGERTIP_SITES_RIGHT = {
    "thumb-tip": "xhand_right_thumb_tip_site",
    "index-finger-tip": "xhand_right_index_finger_tip_site",
    "middle-finger-tip": "xhand_right_middle_finger_tip_site",
    "ring-finger-tip": "xhand_right_ring_finger_tip_site",
    "pinky-finger-tip": "xhand_right_pinky_finger_tip_site",
}

FINGERTIP_SITES_LEFT = {
    "thumb-tip": "xhand_left_thumb_tip_site",
    "index-finger-tip": "xhand_left_index_finger_tip_site",
    "middle-finger-tip": "xhand_left_middle_finger_tip_site",
    "ring-finger-tip": "xhand_left_ring_finger_tip_site",
    "pinky-finger-tip": "xhand_left_pinky_finger_tip_site",
}

# Vuer (three.js, Y-up) -> MuJoCo (Z-up) rotation
_R_VUER_TO_MUJOCO = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def extract_landmark_se3(poses_flat, landmark_name):
    """Extract a 4x4 SE3 matrix from the flat Vuer poses array.

    Vuer sends 25 landmarks * 16 floats (4x4 column-major) = 400 floats.

    Returns:
        4x4 numpy array in Vuer (Y-up) coordinates.
    """
    idx = HAND_LANDMARKS.index(landmark_name)
    mat = np.array(poses_flat[16 * idx : 16 * idx + 16]).reshape(4, 4).T
    return mat


def vuer_to_mujoco(mat4x4):
    """Convert a 4x4 SE3 matrix from Vuer (Y-up) to MuJoCo (Z-up) coordinates."""
    T = np.eye(4)
    T[:3, :3] = _R_VUER_TO_MUJOCO
    return T @ mat4x4


def compute_xhand_finger_lengths(mj_model, mj_data, wrist_site_name, fingertip_sites):
    """Compute XHAND wrist-to-fingertip distances from the model's default pose.

    Returns:
        dict mapping landmark name -> distance (meters).
    """
    wrist_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, wrist_site_name)
    wrist_pos = mj_data.site_xpos[wrist_site_id].copy()

    lengths = {}
    for landmark_name, site_name in fingertip_sites.items():
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            continue
        tip_pos = mj_data.site_xpos[site_id]
        lengths[landmark_name] = np.linalg.norm(tip_pos - wrist_pos)

    return lengths


def compute_human_finger_lengths(poses_flat, fingertip_sites):
    """Compute human wrist-to-fingertip distances from a single frame of VR data.

    Returns:
        dict mapping landmark name -> distance (meters) in MuJoCo coords.
    """
    wrist_mat_mj = vuer_to_mujoco(extract_landmark_se3(poses_flat, "wrist"))
    wrist_pos = wrist_mat_mj[:3, 3]

    lengths = {}
    for landmark_name in fingertip_sites:
        mat_mj = vuer_to_mujoco(extract_landmark_se3(poses_flat, landmark_name))
        tip_pos = mat_mj[:3, 3]
        lengths[landmark_name] = np.linalg.norm(tip_pos - wrist_pos)
    return lengths


def update_mocap_bodies(mj_model, mj_data, poses_flat, finger_scale_factors, tracked_landmarks):
    """Update MuJoCo mocap bodies from Vuer hand landmark data.

    Fingertip positions are scaled by a constant per-finger factor so that
    any human hand maps to full XHAND range of motion while preserving flexion.
    """
    wrist_mat_mj = vuer_to_mujoco(extract_landmark_se3(poses_flat, "wrist"))
    wrist_pos_mj = wrist_mat_mj[:3, 3]

    for landmark_name, mocap_body_name in tracked_landmarks.items():
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
        if body_id < 0:
            continue
        mocap_id = mj_model.body_mocapid[body_id]
        if mocap_id < 0:
            continue

        mat_vuer = extract_landmark_se3(poses_flat, landmark_name)
        mat_mj = vuer_to_mujoco(mat_vuer)
        pos = mat_mj[:3, 3]

        if landmark_name != "wrist" and landmark_name in finger_scale_factors:
            offset = pos - wrist_pos_mj
            pos = wrist_pos_mj + offset * finger_scale_factors[landmark_name]

        mj_data.mocap_pos[mocap_id] = pos

        quat = Rotation.from_matrix(mat_mj[:3, :3]).as_quat(scalar_first=True)
        mj_data.mocap_quat[mocap_id] = quat


def update_finger_scale_factors(human_lengths, xhand_finger_lengths, max_human_lengths, finger_scale_factors):
    """Update max human finger lengths and scale factors if longer fingers are seen.

    Mutates max_human_lengths and finger_scale_factors in place.
    Returns True if any factor was updated.
    """
    updated = False
    for name, length in human_lengths.items():
        if name in xhand_finger_lengths and length > max_human_lengths.get(name, 0):
            max_human_lengths[name] = length
            finger_scale_factors[name] = xhand_finger_lengths[name] / length
            updated = True
    return updated
