import xml.etree.ElementTree as ET
import numpy as np
from dualstack.experiments.scenes import camera


def make_visual_scene(config):
    from .scenes import make_scene
    task=config["experiment"]
    c=dict(config,experiment="06_eccentric" if task in ("08_balance","10_hidden") else "07_tray",difficulty="easy",rolling_friction=.002,ball_jitter=0.)
    root,_=make_scene(c);root.set("model",task)
    w=root.find("worldbody");sample=w.find("body[@name='sample']");contact=root.find("contact")
    def geom(parent,name,**kw):return ET.SubElement(parent,"geom",name=name,**{k:str(v) for k,v in kw.items()})
    def pair(a,b,mu=1.5,roll=None):
        ET.SubElement(contact,"pair",geom1=a,geom2=b,condim="6" if roll is not None else "4",friction=f"{mu} {mu} .02 {roll or .0001} {roll or .0001}",solref=".006 1",solimp=".95 .95 .001")
    if task in ("08_balance","10_hidden"):
        sample.find("geom[@name='sample_geom']").set("size",".16 .018 .018")
        platform=w.find("geom[@name='platform']")
        top=.90 if task=="08_balance" else .96
        platform.set("pos",f".20 -.28 {(top+.75)/2}")
        platform.set("size",f"{config['platform_half_x']} .045 {(top-.75)/2}")
        x=config.get("initial_x",0.);y=-.38 if task=="08_balance" else -.34
        z=.768 if task=="08_balance" else .91
        yaw=config.get("yaw",0.)
        sample.set("pos",f"{x} {y} {z}");sample.set("quat",f"{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}")
        ballast=sample.find("geom[@name='counterweight']")
        if task=="10_hidden":
            sample.find("geom[@name='sample_geom']").set("rgba",".55 .62 .70 1")
            ballast.attrib.update(size=".018 .014 .014",pos=f"{config['weight_side']*config['weight_offset']} 0 0",rgba="0 0 0 0",contype="0",conaffinity="0",group="5")
            for p in list(contact.findall("pair")):
                if "counterweight" in (p.get("geom1"),p.get("geom2")):contact.remove(p)
            geom(w,"observation_pedestal",type="cylinder",size=".012 .07",pos=f"{x} {y} .82",rgba=".25 .3 .35 1")
            pivot=ET.SubElement(w,"body",name="fixture_pivot",pos=f"{x} {y} {z}")
            ET.SubElement(pivot,"joint",name="observation_hinge",type="hinge",axis="0 1 0",range="-.18 .18",limited="true",damping=".20",stiffness="0")
            geom(pivot,"fixture_pin",type="cylinder",size=".009 .026",quat=".70710678 .70710678 0 0",mass=".005",rgba=".2 .25 .3 1",contype="0",conaffinity="0")
            ET.SubElement(root.find("equality"),"weld",name="observation_fixture",body1="sample",body2="fixture_pivot",solref=".004 1",solimp=".95 .99 .001")
        init=dict(position=[x,y,z],yaw=yaw,goal=[.20,-.28,top],platform_top=top)
        center=[.08,-.32,.88 if task=="08_balance" else .96]
    elif task=="09_support":
        for b in list(w.findall("body")):
            if b.get("name") in ("sample","ball"):w.remove(b)
        for p in list(contact.findall("pair")):contact.remove(p)
        zone=w.find("geom[@name='tray_goal']")
        if zone is not None:w.remove(zone)
        sample=ET.SubElement(w,"body",name="sample",pos=".05 -.16 .80")
        ET.SubElement(sample,"freejoint",name="sample_joint")
        geom(sample,"sample_geom",type="box",size=".022 .026 .05",mass=".06",rgba=".9 .20 .07 1")
        geom(sample,"support_handle",type="box",size=".06 .014 .014",pos="-.07 0 0",mass=".01",rgba=".2 .75 .3 1",friction="2 .02 .001")
        geom(w,"right_support",type="box",size=".022 .026 .05",pos=".05 .16 .80",rgba=".24 .40 .70 1")
        beam=ET.SubElement(w,"body",name="beam",pos=".05 0 .86")
        ET.SubElement(beam,"freejoint",name="beam_free")
        geom(beam,"beam_geom",type="box",size=".025 .22 .01",mass=".14",rgba=".55 .60 .67 1",friction="2 .02 .001",condim="4")
        payload=ET.SubElement(w,"body",name="payload",pos=f".05 {config.get('load_offset',.09)} .895")
        ET.SubElement(payload,"freejoint",name="payload_free")
        geom(payload,"payload_geom",type="box",size=".025 .025 .025",mass=".05",rgba="1 .65 .05 1")
        pair("beam_geom","sample_geom");pair("beam_geom","right_support");pair("payload_geom","beam_geom",.8)
        pair("sample_geom","table");pair("support_handle","table")
        geom(w,"support_goal",type="box",pos=".155 -.40 .7505",size=".105 .065 .0005",rgba=".1 .8 .3 .3",contype="0",conaffinity="0",group="2")
        init=dict(position=[.05,-.16,.80],yaw=0.,goal=[.20,-.40,.80],beam_initial=[.05,0,.86],load_offset=config.get("load_offset",.09))
        center=[.06,-.06,.84]
    else:
        # Outlet on +Y, offset from the central right handle. Tilting about X
        # uses differential height of the already validated Y-separated grasps.
        wall=sample.find("geom[@name='rim_1_1']");sample.remove(wall)
        for p in list(contact.findall("pair")):
            if "rim_1_1" in (p.get("geom1"),p.get("geom2")):contact.remove(p)
        for name,cx,half in (("exit_post_left",-.05,.09),("exit_post_right",.125,.015)):
            geom(sample,name,type="box",size=f"{half} .005 .012",pos=f"{cx} .135 .018",mass=".008",rgba=".95 .75 .05 1")
            pair("ball_geom",name,.25,.0002)
        for p in contact.findall("pair"):
            p.set("friction",".25 .25 .001 .0002 .0002")
        for g in sample.findall("geom"):
            if (g.get("name") or "").startswith("rim_"):
                size=list(map(float,g.get("size").split()));size[2]=.012;g.set("size"," ".join(map(str,size)))
                pos=list(map(float,g.get("pos").split()));pos[2]=.018;g.set("pos"," ".join(map(str,pos)))
        ball=w.find("body[@name='ball']");ball.set("pos",".075 -.045 .780")
        x=.075+config.get("receiver_shift",0.);y=.40
        for name in ("tray_goal",):
            element=w.find(f"geom[@name='{name}']")
            if element is not None:w.remove(element)
        receiver=ET.SubElement(w,"body",name="receiver",pos=f"{x} {y} .79")
        geom(receiver,"receiver_floor",type="box",size=".075 .065 .006",rgba=".15 .65 .45 1")
        for axis in (0,1):
            for sign in (-1,1):
                pos=[0,0,.05];pos[axis]=sign*(.075 if axis==0 else .065)
                size=[.080,.070,.05];size[axis]=.005
                geom(receiver,f"receiver_wall_{axis}_{sign}",type="box",size=" ".join(map(str,size)),pos=" ".join(map(str,pos)),rgba=".15 .65 .45 1")
        for g in receiver.findall("geom"):pair("ball_geom",g.get("name"),.5,.002)
        init=dict(position=[0.,0.,.756],yaw=0.,goal=[x,y,.79],receiver=[x,y,.79],receiver_top=.89,
            handle_offsets={side:[0,sign*.185,.034] for side,sign in (("left",-1),("right",1))},exit_axis=1,exit_center=.075,exit_half_width=.035)
        center=[.15,0,.85]
    # Dedicated filming and vision cameras; leave the old observation cameras intact.
    for name in ("film_overview","film_detail"):
        old=w.find(f"camera[@name='{name}']")
        if old is not None:w.remove(old)
    camera(w,"film_overview",[1.05,-1.65,1.85],center,42)
    eye=[.48,-.92,1.14] if task in ("08_balance","10_hidden") else ([.55,-.75,1.20] if task=="09_support" else [.75,-.92,1.30])
    if task=="11_pour":
        camera(w,"film_detail",[.50,-.30,1.90],[.04,.20,.85],35)
    else:camera(w,"film_detail",eye,center,36)
    camera(w,"vision_overview",eye,center,45)
    return root,init
