"""Task adapter for the existing state recorder, package and offline renderer."""
import csv
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from dualstack.experiments.physics import STATE,COLUMNS,measure
from dualstack.experiments.registry import dump
from dualstack.policies.base import load_policy
from .env import TaskEnv
from .policies import TaskFailure

EXTRA=["goal_error_m","inside","hold_time_s","grasp_left","grasp_right","robot_contact","support_force_N",
       "com_x_m","com_y_m","com_z_m","com_margin_x_m","com_margin_y_m","ball_rel_x_m","ball_rel_y_m","ball_rel_z_m","ball_speed_m_s","ball_relative_speed_m_s",
       "left_tracking_error_m","right_tracking_error_m","left_grip_alignment_m","right_grip_alignment_m","tray_orientation_error_rad","grasp_loss_events","phase_id","push_count","overshoot_count","push_speed_command_m_s",
       *[f"{side}_ee_{axis}_m" for side in ("left","right") for axis in "xyz"],
       *[f"{side}_target_{axis}_m" for side in ("left","right") for axis in "xyz"],"table_mu",
       "beam_tilt_rad","beam_height_m","payload_height_m","backup_grasp","hinge_angle_rad","fixture_active","observed_tilt_rad",
       "correct_exit","receiver_inside","receiver_contact_N","support_table_N","com_projection_offset_x_m"]


class Recorder:
    def __init__(self,env,policy,record):
        self.env=env;self.policy=policy;self.record=record;self.rows=[];self.states=[];self.contacts=[];self.poses=[]
        self.phases={};self.state=np.empty(mujoco.mj_stateSize(env.model,STATE));self.table_friction=[]
        self.body_names=["sample"]+env.config.get("extra_bodies",(["ball"] if env.task=="07_tray" else []))
    def sample(self):
        e,p=self.env,self.policy;m,d=e.model,e.data
        row,_=measure(m,d,e.config);info=e.diagnostics;com=d.body("sample").xipos
        phase=getattr(p,"phase","model")
        if phase not in self.phases:self.phases[phase]=len(self.phases)
        values=dict(info,com_x_m=com[0],com_y_m=com[1],com_z_m=com[2],phase_id=self.phases[phase],
            push_count=getattr(p,"pushes",0),overshoot_count=getattr(p,"overshoots",0),push_speed_command_m_s=getattr(p,"push_speed",0.))
        for si,side in enumerate(("left","right")):
            for k,axis in enumerate("xyz"):
                values[f"{side}_ee_{axis}_m"]=e.pose(side)[k]
                values[f"{side}_target_{axis}_m"]=e.last_action[si*8+k]
        target_bodies={m.body(n).id for n in self.body_names}
        values["table_mu"]=0.
        for i,c in enumerate(d.contact):
            if not target_bodies.intersection([int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2)]):continue
            f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
            self.contacts.append([len(self.rows),int(c.geom1),int(c.geom2),float(c.dist),*c.pos.tolist(),*f.tolist(),*c.friction.tolist()])
            names={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)}
            if names=={"sample_geom","table"}:
                values["table_mu"]=float(c.friction[0]);self.table_friction.append(float(c.friction[0]))
        self.rows.append(row+[float(values.get(k,0.)) for k in EXTRA])
        if self.record:
            mujoco.mj_getState(m,d,self.state,STATE);self.states.append(self.state.copy())
            self.poses.append([np.r_[d.body(n).xpos,d.body(n).xquat] for n in self.body_names])


def run(config,output,record=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);dump(output/"config.json",config)
    env=TaskEnv(config);env.reset();policy=load_policy(config["policy"],env)
    (output/"scene.xml").write_text(env.xml)
    recorder=Recorder(env,policy,record);env.recorder=recorder;recorder.sample()
    started=time.perf_counter();failure=None
    try:
        obs=env.observe()
        while not env.done:
            if time.perf_counter()-started>config.get("group_budget_s",120):raise TimeoutError("Task wall-clock budget")
            try:action=policy.act(obs)
            except TaskFailure as e:failure=str(e);break
            obs,_,_,_,_=env.step(action)
    finally:env.close();policy.close()
    seconds=time.perf_counter()-started
    array=np.array(recorder.rows);fields=COLUMNS+EXTRA;get=lambda key:array[:,fields.index(key)]
    info=env.diagnostics;reason=failure or info["reason"]
    result=dict(execution="completed",success=bool(info["success"] and failure is None),outcome=reason,
        final_error_m=float(info["goal_error_m"]),com_support_error_m=float(info["goal_error_m"]),
        final_tilt_rad=float(info["tilt_rad"]),max_tilt_rad=float(env.detector.max_tilt),hold_time_s=float(info["hold_time_s"]),
        completion_time_s=float(env.data.time),physics_seconds=seconds,state_signature=STATE,state_size=len(recorder.state),
        sample_interval_s=config["dt"],samples=len(array),mujoco_version=mujoco.__version__,
        max_penetration_m=float(get("max_penetration_m").max()),model_mutations_during_run=False,
        warnings={str(i):int(w.number) for i,w in enumerate(env.data.warning) if w.number},
        initial_conditions={key:float(array[0,i]) for i,key in enumerate(fields)},scene_initialization=env.initial,
        detector_conditions=info["conditions"],oracle=bool(getattr(policy,"reads_privileged_state",False)),ordinary_observation_contains_object_truth=False,
        reads_friction=bool(getattr(policy,"reads_friction",False)),push_count=getattr(policy,"pushes",0),
        overshoot_count=getattr(policy,"overshoots",0),probe_response_m=getattr(policy,"probe_response",None),
        adaptive_push_speed_m_s=getattr(policy,"push_speed",None),
        effective_table_mu_range=[min(recorder.table_friction),max(recorder.table_friction)] if recorder.table_friction else None,
        max_ball_relative_offset_m=env.detector.max_ball_offset,max_tracking_error_m=env.detector.max_tracking,
        actual_object_mass_kg=float(env.model.body_mass[env.model.body("sample").id]),com_local_m=env.model.body("sample").ipos.tolist(),
        inertia_diagonal_kg_m2=env.model.body("sample").inertia.tolist(),
        placement_offset_m=getattr(policy,"offset",np.zeros(2)).tolist(),
        maximum_object_lift_m=float(env.detector.max_height-env.detector.start_height),
        ball_feedback_pause_s=getattr(policy,"paused_seconds",0.))
    result.update(geometry_center_error_m=float(np.linalg.norm(env.data.body("sample").xpos[:2]-np.asarray(env.initial["goal"][:2]))),
        max_stable_hold_s=float(get("hold_time_s").max()),com_margin_x_m=info.get("com_margin_x_m"),com_margin_y_m=info.get("com_margin_y_m"))
    result.update(grasp_loss_events=env.detector.grasp_loss_events,tray_orientation_error_rad=info.get("tray_orientation_error_rad"))
    result.update(model_input_keys=["images","instruction"] if config.get("vision_only") else ["robot","images","task","time"],
        fixture_release=getattr(env,"fixture_release",None),max_beam_tilt_rad=getattr(env.detector,"max_beam_tilt",None),
        observed_tilt_rad=getattr(env.detector,"max_observed_tilt",None),correct_exit=getattr(env.detector,"exit_seen",None),
        observed_settled=getattr(env.detector,"observation_settled",None),
        strategy=config.get("strategy"),expected_success=config.get("expected_success"))
    result["expected_outcome_observed"]=(result["success"]==config["expected_success"]) if "expected_success" in config else None
    if config.get("vision_only") and not config.get("expected_success",True):
        result["expected_outcome_observed"]=reason=={"08_balance":"tipped","09_support":"structure_collapse"}.get(config["experiment"],"not_applicable")
    bid=env.model.body("sample").id
    principal_rotation=np.empty(9);mujoco.mju_quat2Mat(principal_rotation,env.model.body_iquat[bid])
    principal_rotation=principal_rotation.reshape(3,3)
    result.update(inertia_frame_quaternion=env.model.body_iquat[bid].tolist(),
        inertia_tensor_body_kg_m2=(principal_rotation@np.diag(env.model.body_inertia[bid])@principal_rotation.T).tolist(),
        grasp_offset_world_m=getattr(policy,"grasp_offset",np.zeros(3)).tolist(),
        gripper_force_limits_N={side:env.model.actuator_forcerange[env.model.actuator(side+"_actuator8").id].tolist() for side in ("left","right")})
    with (output/"data.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(fields);writer.writerows(array)
    np.savez_compressed(output/"contact_history.npz",contacts=np.asarray(recorder.contacts,dtype=float).reshape(-1,18),
        columns=np.array(["sample","geom1","geom2","distance","px","py","pz","normal","tangent1","tangent2","torque_n","torque_t1","torque_t2","mu1","mu2","torsion","roll1","roll2"]))
    if record:
        poses=np.asarray(recorder.poses)
        np.savez_compressed(output/"trajectory.npz",states=np.asarray(recorder.states),times=get("time_s"),pose=poses[:,0,:],
            body_poses=poses,body_names=np.array(recorder.body_names),phase_ids=get("phase_id"),phase_names=np.array(list(recorder.phases)),
            state_signature=STATE,body_name="sample",qpos_size=env.model.nq,qvel_size=env.model.nv)
    dump(output/"events.json",dict(phases=recorder.phases,transitions=getattr(policy,"events",[]),controlled_failure=failure,
        privileged_interfaces=["env.oracle.state()"] if getattr(policy,"reads_privileged_state",False) else [],reads_friction=bool(getattr(policy,"reads_friction",False))))
    dump(output/"metrics.json",result)
    return result
