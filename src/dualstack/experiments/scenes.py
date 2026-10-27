"""Scenes use the existing dual Panda model, with explicit contact pairs."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from dualstack.envs.scene import build_scene
from .registry import ROOT


def camera(world, name, eye, target, fovy=45):
    z = np.asarray(eye) - target
    z = z / np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(world, "camera", name=name, pos=" ".join(map(str, eye)),
                  xyaxes=" ".join(map(str, np.r_[x, y])), fovy=str(fovy))


def scene(config):
    c = json.loads((ROOT / "configs/default.json").read_text())
    root = ET.fromstring(build_scene(c))
    root.set("model", config["experiment"])
    root.find("option").set("timestep", str(config["dt"]))
    root.find("option").set("iterations", "100")
    root.find("visual/global").attrib.update(offwidth="1920", offheight="1080")
    ET.SubElement(root.find("visual"), "quality", shadowsize="2048", offsamples="4")
    world = root.find("worldbody")
    for name in ("block_a", "block_b"):
        world.remove(world.find(f"body[@name='{name}']"))
    ET.SubElement(world, "light", pos="1 1 3", dir="-.3 -.3 -1", diffuse=".6 .6 .6", castshadow="false")
    contact = root.find("contact")
    exp = config["experiment"]
    target_pos = [.15, 0, .83]
    body = ET.SubElement(world, "body", name="sample", pos=".15 0 .78")
    if exp != "04_button":
        ET.SubElement(body, "freejoint", name="sample_joint")
    attrs = dict(name="sample_geom", rgba=".92 .18 .08 1", mass=".10", condim="3")
    if exp == "01_friction":
        body.set("pos", "-.15 0 .78")
        ET.SubElement(body, "geom", type="box", size=".03 .03 .03", **attrs)
        ET.SubElement(contact, "pair", geom1="sample_geom", geom2="table", condim="3",
                      friction=f"{config['mu']} {config['mu']} 0 0 0", solref=".006 1", solimp=".95 .95 .001")
        camera(world, "film_detail", [.25, -.64, .86], [0.0, 0, .78], 39)
    elif exp == "02_bounce":
        body.set("pos", ".15 0 1.18")
        ET.SubElement(body, "geom", type="sphere", size=".035", **attrs)
        # Direct negative form: stiffness fixed, damping varied independently.
        ET.SubElement(contact, "pair", geom1="sample_geom", geom2="table", condim="3",
                      friction=".1 .1 0 0 0", solref=f"-{config['contact_stiffness']} -{config['contact_damping']}", solimp=".99 .99 .001")
        target_pos = [.15, 0, .97]
        camera(world, "film_detail", [.48, -.83, 1.0], target_pos, 36)
    elif exp == "03_stack":
        body.set("pos", f"{.15+config['offset_m']} 0 .845")
        ET.SubElement(body, "geom", type="box", size=".025 .025 .025", **attrs)
        base = ET.SubElement(world, "body", name="support", pos=".15 0 .785")
        ET.SubElement(base, "geom", name="support_geom", type="box", size=".035 .035 .035", rgba=".08 .3 .88 1")
        ET.SubElement(contact, "pair", geom1="sample_geom", geom2="support_geom", condim="3",
                      friction="1.5 1.5 0 0 0", solref=".006 1", solimp=".95 .95 .001")
        ET.SubElement(contact, "pair", geom1="sample_geom", geom2="table", condim="3",
                      friction="1.5 1.5 0 0 0", solref=".006 1")
        camera(world, "film_detail", [.46, -.46, 1.00], [.17, 0, .81], 28)
    else:
        body.set("pos", ".15 0 .88")
        ET.SubElement(body, "joint", name="sample_joint", type="slide", axis="0 0 1",
                      limited="true", range="-.09 .025", stiffness=str(config["stiffness"]),
                      damping=str(config["damping"]), springref="0")
        ET.SubElement(body, "geom", type="cylinder", size=".038 .014", **attrs)
        ET.SubElement(body, "site", name="button_top", pos="0 0 .014", size=".002", rgba="0 1 0 1")
        hand = world.find(".//body[@name='left_hand']")
        ET.SubElement(hand, "geom", name="press_tool", type="capsule", fromto="0 0 .10 0 0 .13", size=".010", mass=".005", rgba=".15 .7 .4 1", contype="0", conaffinity="0")
        ET.SubElement(contact, "pair", geom1="press_tool", geom2="sample_geom", condim="3", friction=".5 .5 0 0 0", solref=".008 1")
        ET.SubElement(world, "geom", name="button_housing", type="cylinder", size=".052 .025", pos=".15 0 .775", rgba=".12 .2 .3 1")
        # Decorative spring guides; no collisions and no dynamic force contribution.
        for x in (.12, .18):
            ET.SubElement(world, "geom", type="capsule", fromto=f"{x} 0 .80 {x} 0 .86", size=".002", rgba=".7 .7 .7 1", contype="0", conaffinity="0", group="2")
        camera(world, "film_detail", [.40, -.38, .97], [.15, 0, .84], 29)
    camera(world, "film_overview", [1.1, -1.8, 1.9], target_pos, 43)
    return root


def initialize(model, data, config):
    for side in ("left", "right"):
        for i, q in enumerate([0, 0, 0, -1.57079, 0, 1.57079, -.7853, .04, .04]):
            name = f"{side}_joint{i+1}" if i < 7 else f"{side}_finger_joint{i-6}"
            data.qpos[model.joint(name).qposadr[0]] = q
        for i, q in enumerate([0, 0, 0, -1.57079, 0, 1.57079, -.7853, 255]):
            data.ctrl[model.actuator(f"{side}_actuator{i+1}").id] = q
    joint = model.joint("sample_joint")
    if config["experiment"] == "04_button":
        # Static equilibrium m*g + k*q = 0 (axis points up).
        data.qpos[joint.qposadr[0]] = -.1 * 9.81 / config["stiffness"]
    mujoco.mj_forward(model, data)
    if config["experiment"] == "01_friction":
        # Establish support at rest, then apply velocity only once at reset.
        for _ in range(round(.15 / config["dt"])):
            mujoco.mj_step(model, data)
        data.qvel[joint.dofadr[0]:joint.dofadr[0]+6] = [0.7, 0, 0, 0, 0, 0]
        data.time = 0
        mujoco.mj_forward(model, data)
