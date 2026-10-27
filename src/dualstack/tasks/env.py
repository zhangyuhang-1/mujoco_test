import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from dualstack.control.ik import DualIK
from dualstack.experiments.scenes import initialize
from .scenes import make_scene


class Oracle:
    """Privileged physical state. Only oracle experts may access this object."""
    def __init__(self,env): self.env=env
    def state(self):
        e=self.env;m,d=e.model,e.data
        R=d.body("sample").xmat.reshape(3,3).copy()
        result=dict(position=d.body("sample").xpos.copy(),rotation=R,com=d.body("sample").xipos.copy(),
            com_local=m.body("sample").ipos.copy(),goal=np.array(e.initial["goal"]),velocity=e.velocity("sample"),
            grasp=e.grasps(),half_size=m.geom("sample_geom").size.copy(),time=float(d.time),robot_poses={s:e.pose(s) for s in ("left","right")})
        if e.task in ("07_tray","11_pour"):
            ball=d.body("ball").xpos.copy()
            result.update(ball_position=ball,ball_relative=R.T@(ball-result["position"]),ball_velocity=e.velocity("ball"),
                          handles={s:result["position"]+R@np.array([0,y,.034]) for s,y in (("left",-.185),("right",.185))})
            if "handle_offsets" in e.initial:
                result["handles"]={s:result["position"]+R@np.array(offset) for s,offset in e.initial["handle_offsets"].items()}
        if e.task=="09_support":
            result.update(beam_position=d.body("beam").xpos.copy(),beam_rotation=d.body("beam").xmat.reshape(3,3).copy(),beam_grasp=e.grasp_body("beam","right"),beam_velocity=e.velocity("beam"),payload_position=d.body("payload").xpos.copy())
        if e.task=="10_hidden":
            result.update(fixture_active=bool(d.eq_active[m.equality("observation_fixture").id]),hinge_angle=float(d.qpos[m.joint("observation_hinge").qposadr[0]]))
        return result
    def release_fixture(self):
        e=self.env
        if e.task!="10_hidden" or not e.grasp_body("sample","left"):raise RuntimeError("Fixture release requires confirmed physical grasp")
        before=e.data.qpos.copy();velocity=e.data.qvel.copy()
        e.data.eq_active[e.model.equality("observation_fixture").id]=0
        mujoco.mj_forward(e.model,e.data)
        e.fixture_release=dict(time_s=float(e.data.time),qpos_jump=float(np.max(abs(e.data.qpos-before))),qvel_jump=float(np.max(abs(e.data.qvel-velocity))),purpose="Observation fixture release, not robot attachment")


class TaskEnv:
    """Same reset/step/observe and 16D EE action contract as StackEnv."""
    def __init__(self,config):
        self.config=dict(config);self.task=config["experiment"]
        ratio=config.get("control_dt",.02)/config["dt"]
        if config["dt"]<=0 or ratio<=0 or abs(ratio-round(ratio))>1e-8:
            raise ValueError("control_dt must be a positive integer multiple of dt")
        self.tree,self.initial=make_scene(config)
        self.xml=ET.tostring(self.tree,encoding="unicode")
        self.model=mujoco.MjModel.from_xml_string(self.xml)
        self.data=mujoco.MjData(self.model)
        self.controller=DualIK(self.model,config.get("control_dt",.02))
        self.oracle=Oracle(self);self.renderer=None;self.recorder=None
    def reset(self,seed=None,options=None):
        changed=(seed is not None and seed!=self.config["seed"]) or (options and "random" in options and options["random"]!=self.config.get("random"))
        if changed:
            new=dict(self.config)
            if seed is not None:new["seed"]=seed
            if options and "random" in options:new["random"]=options["random"]
            self.close();self.__init__(new)
        mujoco.mj_resetData(self.model,self.data)
        initialize(self.model,self.data,self.config)
        for _ in range(round(self.config.get("warmup_s",.2)/self.config["dt"])): mujoco.mj_step(self.model,self.data)
        self.data.time=0.;mujoco.mj_forward(self.model,self.data)
        self.last_action=np.r_[self.pose("left"),1.,self.pose("right"),1.]
        self.controller.reset(self.data)
        if self.task in ("08_balance","09_support","10_hidden","11_pour"):
            from .visual_eval import VisualDetector as Detector
        else:
            from .evaluation import Detector
        self.detector=Detector(self)
        self.diagnostics=self.detector.update(0.)
        self.done=False;self.control_steps=0
        return self.observe(),self.diagnostics
    def pose(self,side):
        site=self.data.site(side+"_ee");quat=np.empty(4)
        mujoco.mju_mat2Quat(quat,site.xmat)
        return np.r_[site.xpos,quat]
    def velocity(self,body):
        out=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.model.body(body).id,out,0)
        return out
    def contacts(self):
        m,d=self.model,self.data;out=[]
        for i,c in enumerate(d.contact):
            force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
            if force[0]<=1e-6:continue
            names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or "" for g in (c.geom1,c.geom2)]
            bodies=[int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
            out.append((names,bodies,float(force[0])))
        return out
    def grasps(self):
        result={s:set() for s in ("left","right")};sample=self.model.body("sample").id
        for names,bodies,force in self.contacts():
            if sample not in bodies:continue
            other=bodies[1] if bodies[0]==sample else bodies[0]
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,other) or ""
            for side in result:
                for finger in ("left_finger","right_finger"):
                    if name==side+"_"+finger:result[side].add(finger)
        return {s:len(v)==2 for s,v in result.items()}
    def grasp_body(self,body,side):
        bid=self.model.body(body).id;fingers=set()
        for names,bodies,force in self.contacts():
            if bid not in bodies:continue
            other=bodies[1] if bodies[0]==bid else bodies[0]
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,other) or ""
            for finger in ("left_finger","right_finger"):
                if name==side+"_"+finger:fingers.add(finger)
        return len(fingers)==2
    def model_observation(self):
        from .visual_config import INSTRUCTIONS
        return dict(images=self.render() if self.config.get("images") else {},instruction=self.config.get("instruction",INSTRUCTIONS.get(self.task,self.config["task_title"])))
    def observe(self):
        if self.config.get("vision_only"):return self.model_observation()
        robot={}
        for side in ("left","right"):
            m,d=self.model,self.data
            robot[side]=dict(qpos=np.array([d.qpos[m.joint(f"{side}_joint{i}").qposadr[0]] for i in range(1,8)]),
                qvel=np.array([d.qvel[m.joint(f"{side}_joint{i}").dofadr[0]] for i in range(1,8)]),ee_pose=self.pose(side),
                gripper_width=float(sum(d.qpos[m.joint(f"{side}_finger_joint{i}").qposadr[0]] for i in (1,2))))
        return dict(robot=robot,images=self.render() if self.config.get("images") else {},task=self.config["task_title"],time=float(self.data.time))
    def step(self,action):
        if self.done:raise RuntimeError("Episode ended")
        a=np.asarray(action,float)
        if a.shape!=(16,) or not np.all(np.isfinite(a)):raise ValueError("Expected finite 16D action")
        for i in (0,8):
            if abs(np.linalg.norm(a[i+3:i+7])-1)>.001 or not 0<=a[i+7]<=1:raise ValueError("Invalid quaternion/gripper")
            if np.any(a[i:i+3]<[-.65,-.85,.755]) or np.any(a[i:i+3]>[.55,.85,1.7]):raise ValueError("Unreachable/safe-workspace target")
        self.controller.apply(self.data,a);self.last_action=a.copy();self.control_steps+=1
        for _ in range(round(self.config.get("control_dt",.02)/self.config["dt"])):
            mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
            self.diagnostics=self.detector.update(self.config["dt"])
            if self.recorder:self.recorder.sample()
            if self.diagnostics["reason"]!="running":self.done=True;break
        info=dict(self.diagnostics)
        return self.observe(),float(info["success"]),self.done and info["reason"]!="timeout",info["reason"]=="timeout",info
    def render(self):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=256,width=256)
        images={}
        for camera in (("vision_overview",) if self.config.get("vision_only") else ("overview","left_wrist","right_wrist")):
            self.renderer.update_scene(self.data,camera=camera);images[camera]=self.renderer.render().copy()
        return images
    def close(self):
        if self.renderer:self.renderer.close();self.renderer=None
