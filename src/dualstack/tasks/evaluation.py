"""Success detector: no policy phases, waypoint clocks, or policy success flags."""
import numpy as np
import mujoco


class Detector:
    def __init__(self,env):
        self.e=env;self.hold=0.;self.max_height=env.data.body("sample").xpos[2]
        self.start_height=self.max_height;self.grasp_seen=False;self.push_seen=False
        self.max_ball_offset=0.;self.max_tilt=0.;self.max_tracking=0.
        self.tray_grasp_seen=False;self.grasp_loss_time=0.;self.grasp_loss_events=0
    def update(self,dt):
        e=self.e;m,d=e.model,e.data;t=float(d.time)
        sample=d.body("sample");pos=sample.xpos.copy();R=sample.xmat.reshape(3,3)
        velocity=e.velocity("sample");speed=float(np.linalg.norm(velocity[3:]));omega=float(np.linalg.norm(velocity[:3]))
        tilt=float(np.arccos(np.clip(R[2,2],-1,1)));com=sample.xipos.copy();goal=np.array(e.initial["goal"])
        self.max_height=max(self.max_height,float(pos[2]));self.max_tilt=max(self.max_tilt,tilt)
        contacts=e.contacts();sid=m.body("sample").id;grasps=e.grasps()
        self.grasp_seen|=grasps["left"]
        support_table=support_platform=0.;robot_contact=False;push_contact=False
        for names,bodies,force in contacts:
            if sid not in bodies:continue
            support_table+=force if "table" in names else 0.
            support_platform+=force if "platform" in names else 0.
            other=bodies[1] if bodies[0]==sid else bodies[0]
            other_name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(other)) or ""
            # Menagerie fingertip collision geoms are often unnamed. Classify
            # robot contact by body identity, never only by geom names.
            robot_contact|=other_name.startswith(("left_","right_")) or "push_tool" in names
            push_contact|="push_tool" in names
        self.push_seen|=push_contact
        widths=[e.observe_width(s) if hasattr(e,"observe_width") else sum(d.qpos[m.joint(f"{s}_finger_joint{i}").qposadr[0]] for i in (1,2)) for s in ("left","right")]
        clear=not robot_contact and np.linalg.norm(e.pose("left")[:3]-pos)>.10
        reason="running"
        extras={}
        if e.task=="05_push":
            footprint=np.abs(R[:2,:])@np.array([.025,.025,.025])
            error=float(np.linalg.norm(pos[:2]-goal[:2]))
            inside=bool(np.all(abs(pos[:2]-goal[:2])+footprint<e.config["target_half_size"]-.002))
            conditions=dict(inside=inside,quiet=speed<.012 and omega<.12,supported=support_table>.4*m.body_mass[sid]*9.81,
                withdrawn=clear,pushed=self.push_seen,on_table=abs(pos[2]-.775)<.008)
            if self.max_height>self.start_height+.015:reason="lifted_not_pushed"
            extras.update(goal_error_m=error,inside=inside,push_contact=int(push_contact))
        elif e.task=="06_eccentric":
            error=float(np.linalg.norm(com[:2]-goal[:2]))
            inside=bool(abs(com[0]-goal[0])<e.config["platform_half_x"]-.003 and abs(com[1]-goal[1])<.045-.003)
            lifted=self.max_height>self.start_height+.10
            conditions=dict(com_supported=inside,quiet=speed<.012 and omega<.10,upright=tilt<.10,
                platform=support_platform>.4*m.body_mass[sid]*9.81,open=widths[0]>.065,withdrawn=clear,lifted=lifted,grasped=self.grasp_seen)
            if lifted and pos[2]<.80:reason="tipped" if tilt>.4 else "slid_off"
            extras.update(goal_error_m=error,inside=inside,push_contact=0,com_x_m=float(com[0]),com_y_m=float(com[1]),
                geometric_x_m=float(pos[0]),com_offset_x_m=float((R.T@(com-pos))[0]),lifted=int(lifted),
                com_margin_x_m=float(e.config["platform_half_x"]-abs(com[0]-goal[0])),com_margin_y_m=float(.045-abs(com[1]-goal[1])))
        else:
            error=float(np.linalg.norm(pos-goal));ball=d.body("ball");rel=R.T@(ball.xpos-pos)
            bv=e.velocity("ball");ball_speed=float(np.linalg.norm(bv[3:]));relative_speed=float(np.linalg.norm(bv[3:]-velocity[3:]))
            inside=bool(np.all(abs(rel[:2])<.117) and .01<rel[2]<.09)
            offset=float(np.linalg.norm(rel[:2]));self.max_ball_offset=max(self.max_ball_offset,offset)
            lifted=self.max_height>self.start_height+.07
            self.tray_grasp_seen|=all(grasps.values())
            if lifted and self.tray_grasp_seen and not all(grasps.values()):
                if self.grasp_loss_time==0:self.grasp_loss_events+=1
                self.grasp_loss_time+=dt
            else:self.grasp_loss_time=0.
            cy,sy=np.cos(e.initial["yaw"]),np.sin(e.initial["yaw"])
            target_R=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
            orientation_error=float(np.arccos(np.clip((np.trace(target_R.T@R)-1)/2,-1,1)))
            alignment=[np.linalg.norm(e.pose(s)[:3]-(pos+R@np.array([0,sign*.185,.034]))) for s,sign in (("left",-1),("right",1))]
            tracking=[np.linalg.norm(e.pose(s)[:3]-e.last_action[i:i+3]) for s,i in (("left",0),("right",8))]
            self.max_tracking=max(self.max_tracking,*tracking)
            conditions=dict(at_goal=error<.018,level=tilt<.08,orientation=orientation_error<.10,quiet=speed<.015 and omega<.10 and ball_speed<.035,
                ball_inside=inside,left_grasp=grasps["left"],right_grasp=grasps["right"],lifted=lifted,tracking=max(alignment)<.025)
            if np.any(abs(rel[:2])>.165) or rel[2]<-.02:reason="ball_dropped"
            if lifted and pos[2]<.77:reason="tray_dropped"
            if reason=="running" and self.grasp_loss_time>.15:reason="grasp_lost"
            extras.update(goal_error_m=error,inside=inside,push_contact=0,ball_rel_x_m=float(rel[0]),ball_rel_y_m=float(rel[1]),ball_rel_z_m=float(rel[2]),
                ball_speed_m_s=ball_speed,ball_relative_speed_m_s=relative_speed,left_tracking_error_m=float(tracking[0]),right_tracking_error_m=float(tracking[1]),
                left_grip_alignment_m=float(alignment[0]),right_grip_alignment_m=float(alignment[1]),tray_orientation_error_rad=orientation_error,
                grasp_loss_events=self.grasp_loss_events)
        self.hold=self.hold+dt if all(conditions.values()) and reason=="running" else 0.
        success=self.hold>=.6-1e-9
        if success:reason="success"
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():reason="numerical_error";success=False
        elif pos[2]<.70:reason="dropped";success=False
        elif not(-.65<pos[0]<.7 and -.75<pos[1]<.75):reason="out_of_bounds";success=False
        elif reason=="running" and t>=e.config["duration"]-1e-8:reason="timeout"
        return dict(success=bool(success),reason=reason,hold_time_s=float(self.hold),conditions={k:bool(v) for k,v in conditions.items()},speed_m_s=speed,
            omega_rad_s=omega,tilt_rad=tilt,grasp_left=bool(grasps["left"]),grasp_right=bool(grasps["right"]),
            support_force_N=float(support_table+support_platform),robot_contact=bool(robot_contact),**extras)
