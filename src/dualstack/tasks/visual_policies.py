"""Oracle experts for image/instruction-only interfaces. Not visual policies."""
import numpy as np
import mujoco
from .policies import Expert,TaskFailure,down_quaternion,yaw_of


def quat(rotation):
    q=np.empty(4);mujoco.mju_mat2Quat(q,np.asarray(rotation).reshape(9));return q
def rx(a):
    c,s=np.cos(a),np.sin(a);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def ry(a):
    c,s=np.cos(a),np.sin(a);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def slerp(a,b,u):
    dot=np.dot(a,b)
    if dot<0:b=-b;dot=-dot
    if dot>.9995:q=a+u*(b-a);return q/np.linalg.norm(q)
    angle=np.arccos(np.clip(dot,-1,1));return (np.sin((1-u)*angle)*a+np.sin(u*angle)*b)/np.sin(angle)


class VisualExpert(Expert):
    def move(self,positions,quats,grips,duration,elapsed):
        u=np.clip(elapsed/max(duration,1e-6),0,1);u=u*u*(3-2*u)
        for side,target in positions.items():
            i=0 if side=="left" else 8
            self.action[i:i+3]=self.from_action[i:i+3]+u*(np.asarray(target)-self.from_action[i:i+3])
            self.action[i+3:i+7]=slerp(self.from_action[i+3:i+7],quats[side],u)
            self.action[i+7]=grips[side]
        return elapsed>=duration


class BalanceOracle(VisualExpert):
    def reset(self):
        super().reset();s=self.oracle.state();self.initial=s["position"].copy();R=s["rotation"]
        local=np.array([np.clip(s["com_local"][0],-.065,.065),0,0])
        self.grasp_offset=R@local;self.quat=quat(R@rx(np.pi))
        desired=s["goal"][:2]-(R@s["com_local"])[:2] if self.env.config["strategy"]=="com" else s["goal"][:2]
        self.place=np.r_[desired,self.env.initial["platform_top"]+.018+.0015]
        self.offset=self.place[:2]-s["goal"][:2]
        grab=self.initial+self.grasp_offset;put=self.place+self.grasp_offset
        high=max(self.initial[2]+.25,self.env.initial["platform_top"]+.15)
        self.waypoints=[("approach",grab+[0,0,.17],1.,1.8),("descend",grab+[0,0,.002],1.,1.5),
            ("close",grab+[0,0,.002],0.,1.),("lift",np.r_[grab[:2],high],0.,2.),
            ("transfer",np.r_[put[:2],high],0.,2.),("lower",put,0.,2.),("release",put,1.,1.),
            ("withdraw",put+[0,0,.20],1.,1.5),("hold",put+[0,0,.20],1.,5.)]
        self.index=0;self.enter("approach",0.)
    def act(self,observation):
        s=self.oracle.state();t=s["time"];name,pos,grip,duration=self.waypoints[self.index]
        if self.move({"left":pos},{"left":self.quat},{"left":grip},duration,t-self.stage_time) and self.index<len(self.waypoints)-1:
            if name=="close" and not s["grasp"]["left"]:raise TaskFailure("grasp_not_confirmed")
            if name=="lift" and s["position"][2]<self.initial[2]+.12:raise TaskFailure("lift_not_confirmed")
            self.index+=1;self.enter(self.waypoints[self.index][0],t)
        return self.action.copy()


class HiddenOracle(VisualExpert):
    def reset(self):
        super().reset();self.index=0;self.enter("observe_natural_tilt",0.);self.waypoints=None
    def act(self,observation):
        s=self.oracle.state();t=s["time"]
        if self.phase=="observe_natural_tilt":
            if t<1.:return self.action.copy()
            if abs(s["hinge_angle"])<.12 or np.linalg.norm(s["velocity"][:3])>.05:
                if t>3.:raise TaskFailure("tilt_not_observable_or_not_settled")
                return self.action.copy()
            self.initial=s["position"].copy();R=s["rotation"].copy();local=np.array([s["com_local"][0],0,0])
            self.grasp_offset=R@local;tilted=quat(R@rx(np.pi));level=down_quaternion(0.)
            grab=self.initial+self.grasp_offset
            self.place=np.r_[s["goal"][:2]-s["com_local"][:2],self.env.initial["platform_top"]+.0195]
            self.offset=self.place[:2]-s["goal"][:2];put=self.place+local
            high=max(self.initial[2]+.20,self.env.initial["platform_top"]+.16)
            self.waypoints=[("approach",grab+[0,0,.15],tilted,1.,1.8),("descend",grab+[0,0,.001],tilted,1.,1.5),
                ("close",grab+[0,0,.001],tilted,0.,1.),("fixture_release",grab+[0,0,.001],tilted,0.,.20),
                ("lift",np.r_[grab[:2],high],tilted,0.,1.8),("level",np.r_[self.initial[:2]+local[:2],high],level,0.,1.4),
                ("transfer",np.r_[put[:2],high],level,0.,2.),("lower",put,level,0.,2.),("release",put,level,1.,1.),
                ("withdraw",put+[0,0,.18],level,1.,1.4),("hold",put+[0,0,.18],level,1.,4.)]
            self.enter("approach",t)
        name,pos,q,grip,duration=self.waypoints[self.index]
        if self.move({"left":pos},{"left":q},{"left":grip},duration,t-self.stage_time) and self.index<len(self.waypoints)-1:
            if name=="close":
                if not s["grasp"]["left"]:raise TaskFailure("fixture_grasp_not_confirmed")
                self.oracle.release_fixture()
            if name=="lift" and s["position"][2]<self.initial[2]+.10:raise TaskFailure("fixture_lift_failed")
            self.index+=1;self.enter(self.waypoints[self.index][0],t)
        return self.action.copy()


class SupportOracle(VisualExpert):
    def reset(self):
        super().reset();s=self.oracle.state();beam=s["beam_position"].copy();victim=s["position"].copy();handle=victim+[-.07,0,0]
        self.right_target=beam+np.array([0,0,.001]);rq=down_quaternion(np.pi/2);lq=down_quaternion(0.)
        self.stages=[]
        if self.env.config["strategy"]=="backup":
            self.stages += [("backup_approach",{"right":self.right_target+[0,0,.14]},{"right":rq},{"right":1.},1.8),
                ("backup_descend",{"right":self.right_target},{"right":rq},{"right":1.},1.5),
                ("backup_close",{"right":self.right_target},{"right":rq},{"right":0.},1.)]
        pulled=handle+[-.15,0,0]
        # Pull using the side handle, then regrasp near the column COM on the
        # table. This avoids a long sustained torsional load during transport.
        regrasp=victim+[-.16,0,.005];goal=s["goal"]+[-.01,0,.005]
        for name,target,grip,duration in [("target_approach",handle+[0,0,.16],1.,1.5),("target_descend",handle,1.,1.4),
            ("target_close",handle,0.,1.),("extract",pulled,0.,2.2),
            ("handle_release",pulled,1.,.5),("regrasp_approach",regrasp+[0,0,.15],1.,1.),
            ("regrasp_descend",regrasp,1.,1.),("regrasp_close",regrasp,0.,.8),("lift",regrasp+[0,0,.13],0.,1.),
            ("transfer",goal+[0,0,.13],0.,1.8),("lower",goal,0.,1.4),("release",goal,1.,.8),
            ("withdraw",goal+[0,0,.18],1.,1.2),("hold",goal+[0,0,.18],1.,4.)]:
            self.stages.append((name,{"left":target},{"left":lq},{"left":grip},duration))
        if self.env.config["strategy"]=="direct":
            # Negative control removes and lifts the support immediately. It
            # must not accidentally replace it with the approaching regrasp hand.
            self.stages=self.stages[:4]
            self.stages.append(("lift_removed_support",{"left":pulled+[0,0,.18]},{"left":lq},{"left":0.},2.))
            self.stages.append(("hold",{"left":pulled+[0,0,.18]},{"left":lq},{"left":0.},8.))
        self.index=0;self.enter(self.stages[0][0],0.)
    def act(self,observation):
        s=self.oracle.state();t=s["time"];name,pos,q,grip,duration=self.stages[self.index]
        if self.move(pos,q,grip,duration,t-self.stage_time) and self.index<len(self.stages)-1:
            if name=="backup_close":
                stable=s["beam_grasp"] and np.arccos(np.clip(s["beam_rotation"][2,2],-1,1))<.06 and np.linalg.norm(s["beam_velocity"])<.06
                if not stable:
                    if t-self.stage_time>2.:raise TaskFailure("backup_support_not_stable")
                    return self.action.copy()
            if name in ("target_close","regrasp_close") and not s["grasp"]["left"]:raise TaskFailure("support_grasp_not_confirmed")
            self.index+=1;self.enter(self.stages[self.index][0],t)
        return self.action.copy()


class PourOracle(VisualExpert):
    def reset(self):
        super().reset();s=self.oracle.state();self.initial=s["position"].copy()
        self.hand_quats={"left":down_quaternion(0.),"right":down_quaternion(0.)}
        self.offsets={side:np.array(offset) for side,offset in self.env.initial["handle_offsets"].items()}
        self.receiver=s["goal"].copy();self.pour_position=np.array([self.receiver[0]-.075,self.receiver[1]-.175,.91])
        self.pitch=0.;self.roll=0.;self.enter("approach",0.)
    def act(self,observation):
        s=self.oracle.state();t=s["time"];elapsed=t-self.stage_time;q=self.hand_quats
        if self.phase in ("approach","descend","close","lift","align"):
            if self.phase=="approach":base=self.initial+[0,0,.15];grip=1.;duration=2.;next_phase="descend"
            elif self.phase=="descend":base=self.initial+[0,0,.002];grip=1.;duration=1.6;next_phase="close"
            elif self.phase=="close":base=self.initial+[0,0,.002];grip=0.;duration=1.;next_phase="lift"
            elif self.phase=="lift":base=np.r_[self.initial[:2],.95];grip=0.;duration=2.;next_phase="align"
            else:base=self.pour_position;grip=0.;duration=2.8;next_phase="pour"
            if self.move({side:base+offset for side,offset in self.offsets.items()},q,{side:grip for side in self.offsets},duration,elapsed):
                if self.phase=="close" and not all(s["grasp"].values()):raise TaskFailure("box_grasp_failed")
                self.enter(next_phase,t)
        else:
            local=s["ball_relative"];relv=s["ball_velocity"][3:]-s["velocity"][3:]
            if local[1]<.10 and self.pitch<.24:self.pitch+=.055*self.env.config["control_dt"]
            target_roll=np.clip(-.45*(local[0]-.075)-.12*relv[0],-.025,.025)
            self.roll+=np.clip(target_roll-self.roll,-.001,.001)
            R=rx(-self.pitch)@ry(self.roll)
            # Keep the actual outlet close above the receiver rim throughout tilt.
            position=self.pour_position.copy();position[2]=self.env.initial["receiver_top"]+.015+.14*np.sin(self.pitch)-.006*np.cos(self.pitch)
            for side,offset in self.offsets.items():
                base=np.empty(9);mujoco.mju_quat2Mat(base,self.hand_quats[side])
                i=0 if side=="left" else 8;self.action[i:i+8]=np.r_[position+R@offset,quat(R@base.reshape(3,3)),0.]
        return self.action.copy()
