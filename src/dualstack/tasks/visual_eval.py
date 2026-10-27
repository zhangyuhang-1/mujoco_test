"""Physical outcome checks, independent of expert stages and strategy labels."""
import numpy as np
import mujoco
from .evaluation import Detector


class VisualDetector(Detector):
    def __init__(self,e):
        super().__init__(e);self.max_beam_tilt=0.;self.max_observed_tilt=0.;self.exit_seen=False;self.wrong_exit=False
        self.last_ball_local=None;self.last_fixture=True;self.fixture_grasp_valid=True
        self.collapse_seen=False;self.collapse_time=None
        self.observation_hold=0.;self.observation_settled=False
    def update(self,dt):
        e=self.e;m,d=e.model,e.data;task=e.task;t=float(d.time)
        pos=d.body("sample").xpos.copy();com=d.body("sample").xipos.copy();R=d.body("sample").xmat.reshape(3,3)
        v=e.velocity("sample");speed=float(np.linalg.norm(v[3:]));omega=float(np.linalg.norm(v[:3]));tilt=float(np.arccos(np.clip(R[2,2],-1,1)))
        self.max_height=max(self.max_height,float(pos[2]));self.max_tilt=max(self.max_tilt,tilt)
        goal=np.array(e.initial["goal"]);grasps=e.grasps();self.grasp_seen|=grasps["left"]
        sid=m.body("sample").id;robot=False;table_force=platform_force=0.;payload_force=receiver_force=0.
        for names,bodies,force in e.contacts():
            if sid in bodies:
                other=bodies[1] if bodies[0]==sid else bodies[0]
                robot|=(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,other) or "").startswith(("left_","right_"))
                table_force+=force if "table" in names else 0.
                platform_force+=force if "platform" in names else 0.
            if set(names)=={"payload_geom","beam_geom"}:payload_force+=force
            if "ball_geom" in names and "receiver_floor" in names:receiver_force+=force
        width=float(sum(d.qpos[m.joint(f"left_finger_joint{i}").qposadr[0]] for i in (1,2)))
        clear=not robot and np.linalg.norm(e.pose("left")[:3]-pos)>.10
        reason="running";extra={};error=float(np.linalg.norm(com[:2]-goal[:2]))
        if task in ("08_balance","10_hidden"):
            mx=e.config["platform_half_x"]-abs(com[0]-goal[0]);my=.045-abs(com[1]-goal[1])
            conditions=dict(com_inside=mx>.003 and my>.003,platform_contact=platform_force>.35*m.body_mass[sid]*9.81,
                quiet=speed<.012 and omega<.10,level=tilt<.10,open=width>.065,withdrawn=clear,
                lifted=self.max_height>self.start_height+.10,grasp_seen=self.grasp_seen,not_table_supported=table_force<.02)
            if task=="10_hidden":
                angle=float(d.qpos[m.joint("observation_hinge").qposadr[0]])
                active=bool(d.eq_active[m.equality("observation_fixture").id])
                if active:
                    self.max_observed_tilt=max(self.max_observed_tilt,abs(angle))
                    self.observation_hold=self.observation_hold+dt if abs(angle)>.12 and omega<.05 else 0.
                    self.observation_settled|=self.observation_hold>=.3-1e-9
                if self.last_fixture and not active:self.fixture_grasp_valid=self.grasp_seen
                self.last_fixture=active
                conditions.update(observed_tilt=self.max_observed_tilt>.12,observed_settled=self.observation_settled,fixture_released=not active,release_after_grasp=self.fixture_grasp_valid)
                extra.update(hinge_angle_rad=angle,fixture_active=int(active),observed_tilt_rad=self.max_observed_tilt)
            # Let a genuine tip complete instead of freezing at its first frame.
            self.collapse_seen|=tilt>.35 and self.max_height>self.start_height+.10
            if self.collapse_seen and self.collapse_time is None:self.collapse_time=t
            if self.collapse_seen and t-self.collapse_time>1.2 and (self.max_tilt>.6 or pos[2]<goal[2]-.04) and width>.065:reason="tipped"
            extra.update(com_margin_x_m=float(mx),com_margin_y_m=float(my),support_table_N=table_force,
                com_projection_offset_x_m=float(com[0]-goal[0]))
        elif task=="09_support":
            beam=d.body("beam");load=d.body("payload");br=beam.xmat.reshape(3,3)
            bt=float(np.arccos(np.clip(br[2,2],-1,1)));self.max_beam_tilt=max(self.max_beam_tilt,bt)
            bv=e.velocity("beam");lv=e.velocity("payload")
            # Includes the visible handle, not only the support column.
            inside=bool(.055<pos[0]-.13 and pos[0]+.022<.255 and abs(pos[1]+.40)<.033)
            backup=e.grasp_body("beam","right")
            conditions=dict(target_inside=inside,target_quiet=speed<.015,open=width>.065,target_released=clear,
                beam_upright=bt<.08,beam_height=beam.xpos[2]>.84,payload_height=load.xpos[2]>.875,
                payload_supported=payload_force>.15,beam_quiet=np.linalg.norm(bv)<.03,load_quiet=np.linalg.norm(lv)<.035,backup_contact=backup)
            error=float(np.linalg.norm(pos[:2]-goal[:2]))
            if bt>.7 and (load.xpos[2]<.85 or beam.xpos[2]<.81):reason="structure_collapse"
            extra.update(beam_tilt_rad=bt,beam_height_m=float(beam.xpos[2]),payload_height_m=float(load.xpos[2]),backup_grasp=int(backup))
        else:
            ball=d.body("ball");local=R.T@(ball.xpos-pos);bv=e.velocity("ball");bs=float(np.linalg.norm(bv[3:]))
            if self.last_ball_local is not None and self.last_ball_local[1]<=.14<local[1]:
                if abs(local[0]-.075)<.035 and -.02<local[2]<.085:self.exit_seen=True
                else:self.wrong_exit=True
            if local[1]<-.17 or abs(local[0])>.17:self.wrong_exit=True
            self.last_ball_local=local.copy();self.max_ball_offset=max(self.max_ball_offset,float(np.linalg.norm(local[:2])))
            # Account for 5 mm wall half-thickness, not just outer dimensions.
            # Ten micrometres permits solver-scale contact penetration only.
            radius=float(m.geom("ball_geom").size[0]);tol=1e-5
            inner_x=.075-.005-radius+tol;inner_y=.065-.005-radius+tol
            inside=bool(abs(ball.xpos[0]-goal[0])<inner_x and abs(ball.xpos[1]-goal[1])<inner_y and .79<ball.xpos[2]<.90)
            conditions=dict(correct_exit=self.exit_seen and not self.wrong_exit,ball_in_receiver=inside,receiver_floor_contact=receiver_force>.03,
                ball_quiet=bs<.025,left_grasp=grasps["left"],right_grasp=grasps["right"],box_held=pos[2]>.84,
                box_controlled=tilt<.42 and np.linalg.norm(v)<.045)
            error=float(np.linalg.norm(ball.xpos[:2]-goal[:2]))
            if self.wrong_exit and ball.xpos[2]<.80:reason="wrong_exit"
            elif self.exit_seen and not inside and ball.xpos[2]<.78:reason="not_caught"
            if self.max_height>self.start_height+.08 and pos[2]<.77:reason="box_dropped"
            extra.update(ball_rel_x_m=float(local[0]),ball_rel_y_m=float(local[1]),ball_rel_z_m=float(local[2]),ball_speed_m_s=bs,
                correct_exit=int(self.exit_seen and not self.wrong_exit),receiver_inside=int(inside),receiver_contact_N=receiver_force)
        self.hold=self.hold+dt if reason=="running" and all(conditions.values()) else 0.
        success=self.hold>=.6-1e-9
        if success:reason="success"
        elif reason=="running" and t>=e.config["duration"]-1e-8:
            reason="not_exited" if task=="11_pour" and not self.exit_seen else "timeout"
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():reason="numerical_error";success=False
        return dict(success=bool(success),reason=reason,hold_time_s=float(self.hold),conditions={k:bool(x) for k,x in conditions.items()},
            speed_m_s=speed,omega_rad_s=omega,tilt_rad=tilt,goal_error_m=error,inside=bool(all(conditions.values())),
            grasp_left=grasps["left"],grasp_right=grasps["right"],robot_contact=bool(robot),support_force_N=table_force+platform_force,**extra)
