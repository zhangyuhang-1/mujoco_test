import json
import xml.etree.ElementTree as ET
import numpy as np
from dualstack.envs.scene import build_scene
from dualstack.experiments.scenes import camera
from dualstack.experiments.registry import ROOT
from .config import initial_conditions


def make_scene(config):
    if config["experiment"] in ("08_balance","09_support","10_hidden","11_pour"):
        from .visual_scenes import make_visual_scene
        return make_visual_scene(config)
    init=initial_conditions(config)
    base=json.loads((ROOT/"configs/default.json").read_text())
    root=ET.fromstring(build_scene(base)); root.set("model",config["experiment"])
    root.find("option").attrib.update(timestep=str(config["dt"]),iterations="100")
    root.find("visual/global").attrib.update(offwidth="1920",offheight="1080")
    ET.SubElement(root.find("visual"),"quality",offsamples="4",shadowsize="2048")
    world=root.find("worldbody"); contacts=root.find("contact")
    for name in ("block_a","block_b"): world.remove(world.find(f"body[@name='{name}']"))
    ET.SubElement(world,"light",pos="1 1 3",dir="-.3 -.3 -1",diffuse=".6 .6 .6",castshadow="false")
    p=init["position"]; yaw=init["yaw"]
    body=ET.SubElement(world,"body",name="sample",pos=" ".join(map(str,p)),quat=f"{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}")
    ET.SubElement(body,"freejoint",name="sample_joint")
    def geom(parent,name,**kwargs):
        return ET.SubElement(parent,"geom",name=name,**{k:str(v) for k,v in kwargs.items()})
    def pair(a,b,mu=1.5,rolling=None):
        ET.SubElement(contacts,"pair",geom1=a,geom2=b,condim="6" if rolling is not None else "3",
            friction=f"{mu} {mu} .001 {rolling or 0} {rolling or 0}",solref=".006 1",solimp=".95 .95 .001")
    task=config["experiment"]
    if task in ("06_eccentric","07_tray"):
        # Same position-servo interface and +/-100 N actuator limits. The
        # original kp=100 N/m was insufficient for these heavier objects.
        kp=config.get("gripper_kp",500.)
        for side in ("left","right"):
            actuator=root.find(f"actuator/general[@name='{side}_actuator8']")
            actuator.set("gainprm",f"{.04*kp/255} 0 0")
            actuator.set("biasprm",f"0 {-kp} -20")
    if task=="05_push":
        geom(body,"sample_geom",type="box",size=".025 .025 .025",mass=config["mass"],rgba=".9 .13 .05 1")
        hand=world.find(".//body[@name='left_hand']")
        geom(hand,"push_tool",type="sphere",pos="0 0 .125",size=".012",mass=".005",rgba=".15 .7 .4 1",contype="0",conaffinity="0")
        pair("push_tool","sample_geom",.8); pair("sample_geom","table",config["mu"])
        x,y,_=init["goal"]; half=config["target_half_size"]
        geom(world,"target_zone",type="box",size=f"{half} {half} .0003",pos=f"{x} {y} .7505",rgba=".08 .8 .3 .35",contype="0",conaffinity="0",group="2")
    elif task=="06_eccentric":
        geom(body,"sample_geom",type="box",size=".10 .018 .018",mass=config["bar_mass"],rgba=".85 .16 .08 1")
        offset=config["weight_side"]*config["weight_offset"]
        geom(body,"counterweight",type="box",size=".019 .020 .012",pos=f"{offset} 0 .018",mass=config["weight_mass"],rgba=".97 .72 .05 1")
        goal=init["goal"]
        geom(world,"platform",type="box",size=f"{config['platform_half_x']} .045 .035",pos=f"{goal[0]} {goal[1]} .785",rgba=".07 .3 .8 1")
        pair("sample_geom","platform");pair("sample_geom","table");pair("counterweight","table");pair("counterweight","platform")
    else:
        geom(body,"sample_geom",type="box",size=".14 .14 .006",mass=".18",rgba=".12 .4 .72 1")
        for axis in (0,1):
            for sign in (-1,1):
                pos=[0,0,.020];pos[axis]=sign*.135
                size=[.14,.14,.020];size[axis]=.005
                geom(body,f"rim_{axis}_{sign}",type="box",size=" ".join(map(str,size)),pos=" ".join(map(str,pos)),mass=".01",rgba=".1 .3 .6 1")
        for side,sign in (("left",-1),("right",1)):
            geom(body,side+"_handle",type="box",size=".045 .014 .012",pos=f"0 {sign*.185} .034",mass=".02",friction="2 .01 .001",rgba=".25 .75 .32 1")
            for x in (-.032,.032):
                geom(body,f"{side}_connector_{x}",type="capsule",fromto=f"{x} {sign*.13} .01 {x} {sign*.185} .034",size=".006",mass=".005",rgba=".25 .75 .32 1")
        ball=ET.SubElement(world,"body",name="ball",pos=f"{init['ball_xy'][0]} {init['ball_xy'][1]} .780")
        ET.SubElement(ball,"freejoint",name="ball_joint")
        geom(ball,"ball_geom",type="sphere",size=".018",mass=".025",rgba="1 .55 .05 1")
        for name in ("sample_geom",*[f"rim_{a}_{s}" for a in (0,1) for s in (-1,1)]): pair("ball_geom",name,.25,init["rolling_friction"])
        gx,gy,gz=init["goal"]
        geom(world,"tray_goal",type="box",size=".145 .145 .001",pos=f"{gx} {gy} {gz-.006}",rgba=".1 .8 .3 .12",contype="0",conaffinity="0",group="2")
    center=[.1,-.30,.83] if task!="07_tray" else [.05,0,.84]
    camera(world,"film_overview",[1.1,-1.7,1.9],center,43)
    camera(world,"film_detail",[.50,-.85,1.20] if task!="07_tray" else [.8,-1.0,1.4],center,40)
    return root,init
