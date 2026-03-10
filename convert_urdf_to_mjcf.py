"""
Convert XHAND URDF to a MuJoCo MJCF teleop scene.

Loads the XHAND URDF, converts it to MJCF, then wraps it with:
- Mocap bodies for VR fingertip tracking
- Weld equality constraints to drive the hand from mocap targets
- Proper solver/contact settings for teleoperation

Usage:
    python convert_urdf_to_mjcf.py
    # Outputs: xhand_right_teleop.xml
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

_HERE = Path(__file__).parent
URDF_PATH = _HERE / "XHAND1_URDF_ver 1.3" / "xhand1_right" / "urdf" / "xhand_right.urdf"
MESH_DIR = _HERE / "XHAND1_URDF_ver 1.3" / "xhand1_right" / "meshes"
OUTPUT_PATH = _HERE / "xhand_right_teleop.xml"


def patch_urdf_text(urdf_text: str) -> str:
    """Fix URDF issues that prevent MuJoCo loading."""
    # Fix package:// URIs to plain filenames
    urdf_text = urdf_text.replace('filename="package://xhand_right/meshes/', 'filename="')

    MIN_INERTIA = "0.00000001"

    # Fix inertia: diagonal to minimum, off-diagonal to zero
    def fix_inertia_attr(m):
        tag, val_str = m.group(1), m.group(2)
        val = float(val_str)
        if tag in ("ixx", "iyy", "izz"):
            if abs(val) < 1e-7:
                return f'{tag}="{MIN_INERTIA}"'
        else:
            if abs(val) < 1e-7:
                return f'{tag}="0"'
        return m.group(0)

    urdf_text = re.sub(r'(ixx|ixy|ixz|iyy|iyz|izz)="([^"]+)"', fix_inertia_attr, urdf_text)

    # Fix tiny masses
    urdf_text = re.sub(
        r'<mass\s+value="([^"]+)"\s*/>',
        lambda m: m.group(0) if float(m.group(1)) >= 1e-4
        else '<mass value="0.0001" />',
        urdf_text,
    )
    return urdf_text


def load_urdf_via_wrapper():
    """Load URDF via an MJCF wrapper to control compiler settings.

    Uses fusestatic="false" so MuJoCo preserves all bodies (including
    fixed-joint tip links and the hand base body).

    Returns:
        (model, data, temp_files_to_cleanup)
    """
    patched_urdf_text = patch_urdf_text(URDF_PATH.read_text())

    # Write patched URDF next to meshes
    patched_urdf = MESH_DIR / "_xhand_right_patched.urdf"
    patched_urdf.write_text(patched_urdf_text)

    # Create an MJCF wrapper that includes the URDF with fusestatic="false"
    # MuJoCo's <include> doesn't support URDF, so we use <compiler> + direct load
    # But actually, we can't include URDF in MJCF. Instead, we load the URDF
    # directly with the fusestatic flag set via a wrapper MJCF that overrides compiler.
    #
    # Workaround: compile the URDF with fusestatic=false by setting it globally.
    # MuJoCo doesn't support this via URDF directly, so we:
    # 1. Load URDF → get compiled XML
    # 2. Add fusestatic="false" and re-parse

    # Step 1: Load URDF
    model = mujoco.MjModel.from_xml_path(str(patched_urdf))
    data = mujoco.MjData(model)

    # Step 2: Save as MJCF, modify compiler, reload
    compiled_path = MESH_DIR / "_xhand_compiled.xml"
    mujoco.mj_saveLastXML(str(compiled_path), model)

    # Parse and add fusestatic="false"
    tree = ET.parse(compiled_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("fusestatic", "false")

    # Rewrite and reload
    tree.write(str(compiled_path), xml_declaration=True, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(compiled_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    return model, data, [patched_urdf, compiled_path]


def main():
    print(f"Loading URDF: {URDF_PATH}")

    model, data, temp_files = load_urdf_via_wrapper()

    try:
        # Print body info
        print(f"\nModel: {model.njnt} joints, {model.nq} qpos, {model.nbody} bodies")
        print("\nBodies:")
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            print(f"  [{i}] {name}  pos={data.xpos[i]}")

        print("\nJoints:")
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            jtype = model.jnt_type[i]
            type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
            print(f"  [{i}] {name} ({type_names.get(jtype, jtype)}, qpos[{model.jnt_qposadr[i]}])")

        # Identify tip bodies (last body in each finger chain)
        # The wrist is handled separately via the wrapper body.
        tip_body_candidates = {
            "right-thumb-tip": "right_hand_thumb_rota_tip",
            "right-index-finger-tip": "right_hand_index_rota_tip",
            "right-middle-finger-tip": "right_hand_mid_tip",
            "right-ring-finger-tip": "right_hand_ring_tip",
            "right-pinky-finger-tip": "right_hand_pinky_tip",
        }

        # Fallback: use the last link in the actuated chain if tip not found
        tip_body_fallbacks = {
            "right-thumb-tip": "right_hand_thumb_rota_link2",
            "right-index-finger-tip": "right_hand_index_rota_link2",
            "right-middle-finger-tip": "right_hand_mid_link2",
            "right-ring-finger-tip": "right_hand_ring_link2",
            "right-pinky-finger-tip": "right_hand_pinky_link2",
        }

        # Resolve which bodies actually exist
        tip_bodies = {}
        for mocap_name in tip_body_candidates:
            primary = tip_body_candidates[mocap_name]
            fallback = tip_body_fallbacks[mocap_name]

            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, primary)
            if bid >= 0:
                tip_bodies[mocap_name] = primary
            else:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fallback)
                if bid >= 0:
                    tip_bodies[mocap_name] = fallback
                    print(f"  Note: using fallback '{fallback}' for {mocap_name}")
                else:
                    print(f"  WARNING: no body found for {mocap_name}")

        # Get tip positions for mocap initialization
        print("\nTip body positions:")
        mocap_init = {}
        for mocap_name, body_name in tip_bodies.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            pos = data.xpos[bid].copy()
            quat = data.xquat[bid].copy()
            mocap_init[mocap_name] = (pos, quat)
            print(f"  {mocap_name} -> {body_name}: pos={pos}")

        # Build the teleop scene
        print("\nBuilding teleop scene...")
        mesh_rel = MESH_DIR.relative_to(_HERE)

        # Save current model as base MJCF
        base_path = _HERE / "_xhand_base.xml"
        mujoco.mj_saveLastXML(str(base_path), model)

        tree = ET.parse(base_path)
        root = tree.getroot()

        # -- Compiler --
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        compiler.set("angle", "radian")
        compiler.set("autolimits", "true")
        compiler.set("fusestatic", "false")
        compiler.set("meshdir", str(mesh_rel))
        compiler.set("assetdir", str(mesh_rel))

        # -- Solver --
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("iterations", "200")
        option.set("tolerance", "1e-10")
        option.set("solver", "Newton")
        option.set("integrator", "implicit")

        # -- Size --
        size = root.find("size")
        if size is None:
            size = ET.SubElement(root, "size")
        size.set("njmax", "1000")
        size.set("nconmax", "1000")

        # -- Default --
        default = root.find("default")
        if default is None:
            default = ET.SubElement(root, "default")
        jd = default.find("joint")
        if jd is None:
            jd = ET.SubElement(default, "joint")
        jd.set("damping", "0.1")

        # -- Visual --
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        hl = visual.find("headlight")
        if hl is None:
            hl = ET.SubElement(visual, "headlight")
        hl.set("diffuse", "0.6 0.6 0.6")
        hl.set("ambient", "0.3 0.3 0.3")
        hl.set("specular", "0 0 0")

        worldbody = root.find("worldbody")

        # -- Lights --
        ET.SubElement(worldbody, "light", {
            "name": "key", "pos": "0.5 0.5 1.0", "dir": "-1 -0.5 -0.5",
            "cutoff": "45", "diffuse": "0.6 0.6 0.6", "directional": "true",
        })
        ET.SubElement(worldbody, "light", {
            "name": "fill", "pos": "-0.5 -0.5 1.0", "dir": "0.8 0.5 -0.6",
            "cutoff": "60", "diffuse": "0.3 0.3 0.3", "directional": "true",
        })

        # -- Cameras --
        ET.SubElement(worldbody, "camera", {
            "name": "front", "pos": "0 -0.5 0.3", "xyaxes": "1 0 0 0 0.5 1", "fovy": "60",
        })

        # -- Freejoint + wrist site --
        # The URDF root link (right_hand_link) gets merged into world by MuJoCo.
        # We need to wrap all finger chain bodies in a new body with a freejoint.
        hand_body = worldbody.find(".//body[@name='right_hand_link']")
        if hand_body is None:
            # Root link was merged into world. Create a wrapper body.
            wrapper = ET.SubElement(worldbody, "body")
            wrapper.set("name", "xhand_right_base")
            wrapper.set("pos", "0 0 0")

            fj = ET.SubElement(wrapper, "joint")
            fj.set("name", "xhand_right_floating_base")
            fj.set("type", "free")

            s = ET.SubElement(wrapper, "site")
            s.set("name", "xhand_right_wrist_site")
            s.set("size", "0.01")
            s.set("rgba", "0 1 1 0.5")

            # Move all finger chain bodies under the wrapper
            bodies_to_move = []
            for child in list(worldbody):
                if child.tag == "body" and child.get("name", "").startswith("right_hand_"):
                    bodies_to_move.append(child)
            for b in bodies_to_move:
                worldbody.remove(b)
                wrapper.append(b)

            # Also move any geoms that belong to the hand base
            geoms_to_move = []
            for child in list(worldbody):
                if child.tag == "geom":
                    geoms_to_move.append(child)
            for g in geoms_to_move:
                worldbody.remove(g)
                wrapper.insert(0, g)  # before the child bodies

            print("  Created wrapper body 'xhand_right_base' with freejoint")
        else:
            fj = ET.SubElement(hand_body, "joint")
            fj.set("name", "xhand_right_floating_base")
            fj.set("type", "free")
            s = ET.SubElement(hand_body, "site")
            s.set("name", "xhand_right_wrist_site")
            s.set("size", "0.01")
            s.set("rgba", "0 1 1 0.5")

        # -- Sites on tip bodies --
        tip_site_names = {}
        for mocap_name, body_name in tip_bodies.items():
            if mocap_name == "right-wrist":
                continue  # wrist site already added above
            site_name = f"xhand_{mocap_name.replace('-', '_')}_site"
            tip_site_names[mocap_name] = site_name

            elem = worldbody.find(f".//body[@name='{body_name}']")
            if elem is not None:
                s = ET.SubElement(elem, "site")
                s.set("name", site_name)
                s.set("size", "0.005")
                s.set("rgba", "1 0 0 1")
            else:
                print(f"  WARNING: body '{body_name}' not found in scene")

        # -- Mocap bodies --
        # Wrist mocap (at origin, since hand base was at world origin)
        wrist_mb = ET.SubElement(worldbody, "body")
        wrist_mb.set("mocap", "true")
        wrist_mb.set("name", "right-wrist")
        wrist_mb.set("pos", "0 0 0")
        s = ET.SubElement(wrist_mb, "site")
        s.set("name", "right-wrist-mocap-site")
        s.set("rgba", "0 0 1 1.0")
        s.set("size", ".01")

        # Fingertip mocap bodies
        for mocap_name, (pos, quat) in mocap_init.items():
            mb = ET.SubElement(worldbody, "body")
            mb.set("mocap", "true")
            mb.set("name", mocap_name)
            mb.set("pos", f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
            mb.set("quat", f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}")
            s = ET.SubElement(mb, "site")
            s.set("name", f"{mocap_name}-mocap-site")
            s.set("rgba", "0 0 1 1.0")
            s.set("size", ".01")

        # -- Equality constraints (weld) --
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")

        # Wrist weld
        w = ET.SubElement(equality, "weld")
        w.set("site1", "xhand_right_wrist_site")
        w.set("site2", "right-wrist-mocap-site")

        # Fingertip welds
        for mocap_name, site_name in tip_site_names.items():
            w = ET.SubElement(equality, "weld")
            w.set("site1", site_name)
            w.set("site2", f"{mocap_name}-mocap-site")

        # Write output
        ET.indent(tree, space="  ")
        tree.write(str(OUTPUT_PATH), xml_declaration=True, encoding="utf-8")
        print(f"\nWritten to: {OUTPUT_PATH}")

        # Cleanup intermediate
        base_path.unlink(missing_ok=True)

        # Verify
        print("\nVerifying...")
        vmodel = mujoco.MjModel.from_xml_path(str(OUTPUT_PATH))
        vdata = mujoco.MjData(vmodel)
        mujoco.mj_forward(vmodel, vdata)
        print(f"  OK: {vmodel.njnt} joints, {vmodel.nq} qpos, "
              f"{vmodel.nbody} bodies, {vmodel.nmocap} mocap bodies")

        # Print the 1:1 joint mapping
        print("\n  XHAND joints (1:1 mapping, no averaging needed):")
        hinge_idx = 0
        for i in range(vmodel.njnt):
            name = mujoco.mj_id2name(vmodel, mujoco.mjtObj.mjOBJ_JOINT, i)
            if vmodel.jnt_type[i] == 3:  # hinge
                qaddr = vmodel.jnt_qposadr[i]
                print(f"    XHAND[{hinge_idx}] {name} -> qpos[{qaddr}]")
                hinge_idx += 1

    finally:
        for f in temp_files:
            Path(f).unlink(missing_ok=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
