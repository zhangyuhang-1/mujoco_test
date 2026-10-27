"""Privileged oracle experts. Ordinary observations never contain object truth."""
import numpy as np


class TaskFailure(RuntimeError): pass


def down_quaternion(yaw):return np.array([0.,np.cos(yaw/2),np.sin(yaw/2),0.])
def yaw_of(rotation):return float(np.arctan2(rotation[1,0],rotation[0,0]))


class Expert:
    reads_privileged_state=True
    reads_friction=False
    def __init__(self,env):self.env=env;self.oracle=env.oracle;self.reset()
    def reset(self):
        self.action=self.env.last_action.copy();self.events=[];self.phase="initial";self.stage_time=0.;self.from_action=self.action.copy()
    def enter(self,phase,t):
        self.phase=phase;self.stage_time=t;self.from_action=self.action.copy()
        self.events.append(dict(time_s=float(t),phase=phase))
    def move(self,positions,quats,grips,duration,elapsed):
        u=np.clip(elapsed/max(duration,1e-6),0,1);u=u*u*(3-2*u)
        for side,target in positions.items():
            i=0 if side=="left" else 8
            self.action[i:i+3]=self.from_action[i:i+3]+u*(np.asarray(target)-self.from_action[i:i+3])
            self.action[i+3:i+7]=quats[side];self.action[i+7]=grips[side]
        return elapsed>=duration
    def close(self):pass


class PushOracle(Expert):
    """Probe response -> feedback speed, bounded corrections; never reads mu."""
    def reset(self):
        super().reset();self.pushes=0;self.overshoots=0;self.probe_response=0.;self.push_speed=.035
        self.direction=None;self.enter("approach",0.);self.plan()
    def plan(self):
        s=self.oracle.state();delta=s["goal"][:2]-s["position"][:2];distance=np.linalg.norm(delta)
        if distance<1e-5:self.direction=np.array([1.,0.])
        else:self.direction=delta/distance
        self.behind=np.r_[s["position"][:2]-.065*self.direction,.94]
        self.quat=down_quaternion(0.)
    def act(self,observation):
        t=observation["time"];elapsed=t-self.stage_time;s=self.oracle.state()
        self.action[7]=0.
        if self.phase=="approach":
            if self.move({"left":self.behind},{"left":self.quat},{"left":0.},1.4,elapsed):self.enter("descend",t)
        elif self.phase=="descend":
            if self.move({"left":np.r_[self.behind[:2],.797]},{"left":self.quat},{"left":0.},1.,elapsed):
                self.probe_start=s["position"].copy();self.probe_tool=self.action[:3].copy();self.enter("probe",t)
        elif self.phase=="probe":
            if self.move({"left":self.probe_tool+np.r_[self.direction*.045,0.]},{"left":self.quat},{"left":0.},1.5,elapsed):
                self.probe_response=float(np.dot(s["position"][:2]-self.probe_start[:2],self.direction))
                self.push_speed=float(np.clip(.018+.6*self.probe_response,.022,.045));self.pushes+=1
                self.command=self.action[:3].copy();self.enter("push_feedback",t)
        elif self.phase=="push_feedback":
            error=s["goal"][:2]-s["position"][:2]
            along=float(np.dot(error,self.direction));cross=error-along*self.direction
            if np.linalg.norm(error)<.010 or along<-.008:
                if along<-.008:self.overshoots+=1
                self.retreat=self.action[:3]-np.r_[self.direction*.07,0.]+[0,0,.12]
                self.enter("withdraw",t)
            elif elapsed>12.:
                self.retreat=self.action[:3]+[0,0,.15];self.enter("withdraw",t)
            else:
                dt=self.env.config.get("control_dt",.02)
                self.command[:2]+=self.direction*min(self.push_speed,max(.008,along*.8))*dt+np.clip(cross,-.01,.01)*dt
                self.action[:3]=self.command
        elif self.phase=="withdraw":
            if self.move({"left":self.retreat},{"left":self.quat},{"left":0.},.8,elapsed):self.enter("observe",t)
        elif self.phase=="observe":
            if elapsed>.8:
                if np.linalg.norm(s["goal"][:2]-s["position"][:2])<.018:self.enter("hold",t)
                elif self.pushes>=self.env.config["max_pushes"]:raise TaskFailure("correction_budget_exhausted")
                else:self.plan();self.enter("approach",t)
        return self.action.copy()


class PlacementOracle(Expert):
    def reset(self):
        super().reset();s=self.oracle.state();self.initial=s["position"].copy();self.quat=down_quaternion(yaw_of(s["rotation"]))
        # COM-aware desired geometric center, not geometric-center alignment.
        self.place=np.r_[s["goal"][:2]-(s["rotation"]@s["com_local"])[:2],.82+.018+.0015]
        self.offset=self.place[:2]-s["goal"][:2];self.grasp_confirmed=False;self.lift_confirmed=False
        # Near-COM grasp removes the gravity torque that made a geometric-center
        # grasp rotate out of the narrow fingertip contact patches. Clip before
        # the visible weight structure to retain finger clearance.
        self.grasp_offset=s["rotation"]@np.array([np.clip(s["com_local"][0],-.025,.025),0,0])
        grab=self.initial+self.grasp_offset;put=self.place+self.grasp_offset
        self.waypoints=[("approach",grab+[0,0,.17],1.,1.8),
            ("descend",grab+[0,0,.002],1.,1.5),("close",grab+[0,0,.002],0.,1.0),
            ("lift",grab+[0,0,.19],0.,1.8),("transfer",np.r_[put[:2],self.initial[2]+.19],0.,2.2),
            ("lower",put,0.,2.0),("release",put,1.,1.),("withdraw",put+[0,0,.20],1.,1.6),
            ("hold",put+[0,0,.20],1.,5.)]
        self.index=0;self.enter(self.waypoints[0][0],0.)
    def act(self,observation):
        t=observation["time"];name,pos,grip,duration=self.waypoints[self.index]
        finished=self.move({"left":pos},{"left":self.quat},{"left":grip},duration,t-self.stage_time)
        if finished and self.index<len(self.waypoints)-1:
            s=self.oracle.state()
            if name=="close":
                if not s["grasp"]["left"]:raise TaskFailure("grasp_not_confirmed")
                self.grasp_confirmed=True
            if name=="lift":
                if s["position"][2]<self.initial[2]+.12:raise TaskFailure("lift_not_confirmed")
                self.lift_confirmed=True
            self.index+=1;self.enter(self.waypoints[self.index][0],t)
        return self.action.copy()


class TrayOracle(Expert):
    def reset(self):
        super().reset();s=self.oracle.state();self.initial=s["position"].copy();self.rotation=s["rotation"].copy()
        self.quat=down_quaternion(yaw_of(self.rotation));self.hand_offsets={side:self.rotation@np.array([0,sign*.185,.034]) for side,sign in (("left",-1),("right",1))}
        self.goal=s["goal"].copy();self.progress=0.;self.paused_seconds=0.;self.repairs=0
        self.enter("approach",0.)
    def act(self,observation):
        t=observation["time"];elapsed=t-self.stage_time;s=self.oracle.state();q={side:self.quat for side in ("left","right")}
        offsets=self.hand_offsets
        if self.phase=="approach":
            if self.move({side:self.initial+offset+[0,0,.15] for side,offset in offsets.items()},q,{"left":1.,"right":1.},2.,elapsed):self.enter("descend",t)
        elif self.phase=="descend":
            if self.move({side:self.initial+offset+[0,0,.002] for side,offset in offsets.items()},q,{"left":1.,"right":1.},1.6,elapsed):self.enter("close",t)
        elif self.phase=="close":
            if self.move({side:self.initial+offset+[0,0,.002] for side,offset in offsets.items()},q,{"left":0.,"right":0.},1.,elapsed):
                if not all(s["grasp"].values()):
                    if self.repairs>=1:raise TaskFailure("bilateral_tray_grasp_not_confirmed")
                    self.repairs+=1;self.enter("close",t)
                else:self.enter("lift",t)
        elif self.phase=="lift":
            lifted=np.r_[self.initial[:2],self.goal[2]]
            if self.move({side:lifted+offset for side,offset in offsets.items()},q,{"left":0.,"right":0.},2.,elapsed):
                if s["position"][2]<self.initial[2]+.07:raise TaskFailure("tray_lift_failed")
                self.transport_start=lifted;self.progress=0.;self.enter("transport",t)
        elif self.phase=="transport":
            rel=s["ball_relative"];relv=s["ball_velocity"][3:]-s["velocity"][3:]
            scale=.35 if np.linalg.norm(rel[:2])>.07 or np.linalg.norm(relv)>.10 else 1.
            if np.linalg.norm(rel[:2])>.095:scale=0.;self.paused_seconds+=self.env.config["control_dt"]
            if self.paused_seconds>3.:raise TaskFailure("ball_feedback_pause_budget")
            self.progress+=scale*self.env.config["control_dt"]/4.
            u=min(1.,self.progress)
            if self.env.config["difficulty"]=="medium":
                middle=(self.transport_start+self.goal)/2+np.array([0,-.035,0])
                a,b,v=(self.transport_start,middle,u*2) if u<.5 else (middle,self.goal,(u-.5)*2)
            else:a,b,v=self.transport_start,self.goal,u
            v=v*v*(3-2*v);position=a+(b-a)*v
            for side,offset in offsets.items():
                i=0 if side=="left" else 8;self.action[i:i+8]=np.r_[position+offset,self.quat,0.]
            if u>=1:self.enter("hold",t)
        elif self.phase=="hold":
            for side,offset in offsets.items():
                i=0 if side=="left" else 8;self.action[i:i+8]=np.r_[self.goal+offset,self.quat,0.]
        return self.action.copy()
