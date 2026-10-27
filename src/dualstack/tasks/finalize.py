"""Derived-data checks and final export using the shared artifact pipeline.

No physics is advanced. Raw trajectory states and contact forces are immutable.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from dualstack.experiments.registry import ROOT,TASKS,dump
from dualstack.experiments.artifacts import package,zip_package,validate_zip
from .reporting import aggregate,report
from .env import TaskEnv
from .evaluation import Detector


def enrich(group):
    path=group["path"];metrics=group["metrics"];config=group["config"]
    original=(path/"data.csv").read_bytes();before_hash=hashlib.sha256(original).hexdigest()
    a=np.genfromtxt(path/"data.csv",delimiter=",",names=True)
    columns={name:a[name] for name in a.dtype.names}
    m=mujoco.MjModel.from_xml_path(str(path/"scene.xml"));sid=m.body("sample").id
    mass=float(m.body_mass[sid]);principal=Rotation.from_quat(m.body_iquat[sid][[1,2,3,0]]).as_matrix()
    inertia=principal@np.diag(m.body_inertia[sid])@principal.T
    rot=Rotation.from_quat(np.column_stack([columns[k] for k in ("qx","qy","qz","qw")])).as_matrix()
    positions=np.column_stack([columns[k] for k in ("x_m","y_m","z_m")])
    predicted_com=positions+np.einsum("nij,j->ni",rot,m.body_ipos[sid])
    measured_com=np.column_stack([columns[k] for k in ("com_x_m","com_y_m","com_z_m")])
    com_error=float(np.max(abs(predicted_com-measured_com)))
    if com_error>1e-9:raise AssertionError(f"Recorded COM does not match compiled mass distribution: {path}")
    omega=np.column_stack([columns[k] for k in ("wx_rad_s","wy_rad_s","wz_rad_s")])
    omega_body=np.einsum("nji,nj->ni",rot,omega)
    linear=np.column_stack([columns[k] for k in ("vx_m_s","vy_m_s","vz_m_s")])
    energy=.5*mass*np.sum(linear**2,axis=1)+.5*np.einsum("ni,ij,nj->n",omega_body,inertia,omega_body)
    energy_error=float(np.max(abs(energy-columns["kinetic_energy_J"])))
    columns["kinetic_energy_J"]=energy
    with np.load(path/"contact_history.npz",allow_pickle=False) as z:contacts=z["contacts"]
    robot_contact=np.zeros(len(a),bool)
    physical_push_seen=False
    if len(contacts):
        gids=contacts[:,1:3].astype(int);bodies=m.geom_bodyid[gids]
        body_is_robot=np.array([(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or "").startswith(("left_","right_")) for i in range(m.nbody)])
        geom_is_tool=np.array([(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or "")=="push_tool" for i in range(m.ngeom)])
        mask=((bodies[:,0]==sid)&(body_is_robot[bodies[:,1]]|geom_is_tool[gids[:,1]]))|((bodies[:,1]==sid)&(body_is_robot[bodies[:,0]]|geom_is_tool[gids[:,0]]))
        mask &= contacts[:,7]>1e-6
        robot_contact[contacts[mask,0].astype(int)]=True
        physical_push_seen=bool(np.any(((bodies[:,0]==sid)&geom_is_tool[gids[:,1]]|((bodies[:,1]==sid)&geom_is_tool[gids[:,0]]))&(contacts[:,7]>1e-6)))
    contact_changes=int(np.count_nonzero(robot_contact!=columns["robot_contact"].astype(bool)))
    columns["robot_contact"]=robot_contact.astype(float)
    metrics.update(actual_object_mass_kg=mass,com_local_m=m.body_ipos[sid].tolist(),
        inertia_diagonal_kg_m2=m.body_inertia[sid].tolist(),inertia_frame_quaternion=m.body_iquat[sid].tolist(),inertia_tensor_body_kg_m2=inertia.tolist())
    if config["experiment"]=="06_eccentric":
        goal=metrics["scene_initialization"]["goal"]
        columns["com_margin_x_m"]=config["platform_half_x"]-abs(columns["com_x_m"]-goal[0])
        columns["com_margin_y_m"] = .045-abs(columns["com_y_m"]-goal[1])
        metrics.update(com_margin_x_m=float(columns["com_margin_x_m"][-1]),com_margin_y_m=float(columns["com_margin_y_m"][-1]))
    if config["experiment"]=="07_tray":
        lifted=columns["z_m"]>columns["z_m"][0]+.07
        metrics["max_transport_tracking_error_m"]=max(float(columns[k][lifted].max()) if np.any(lifted) else 0. for k in ("left_tracking_error_m","right_tracking_error_m"))
    if metrics["success"]:
        with np.load(path/"trajectory.npz",allow_pickle=False) as z:last=z["states"][-1];sig=int(z["state_signature"])
        d=mujoco.MjData(m);mujoco.mj_setState(m,d,last,sig);mujoco.mj_forward(m,d)
        e=TaskEnv.__new__(TaskEnv);e.model=m;e.data=d;e.task=config["experiment"];e.config=config;e.initial=metrics["scene_initialization"]
        e.last_action=np.r_[e.pose("left"),0.,e.pose("right"),0.]
        for si,side in enumerate(("left","right")):
            e.last_action[si*8:si*8+3]=[columns[f"{side}_target_{axis}_m"][-1] for axis in "xyz"]
        detector=Detector(e);detector.start_height=float(columns["z_m"][0]);detector.max_height=detector.start_height+metrics["maximum_object_lift_m"]
        detector.grasp_seen=bool(np.any(columns["grasp_left"]));detector.push_seen=physical_push_seen
        snapshot=detector.update(.6)
        if not all(snapshot["conditions"].values()):raise AssertionError(f"Successful episode fails independent final-state recheck: {path}: {snapshot['conditions']}")
        if e.task!="07_tray":
            tail=columns["time_s"]>=columns["time_s"][-1]-.598
            if np.any(robot_contact[tail]):raise AssertionError("Successful release has robot contact during final hold")
        metrics["final_conditions_rechecked"]=snapshot["conditions"]
    if energy_error>1e-10 or contact_changes or set(columns)!=set(a.dtype.names):
        archive=ROOT/"outputs/tasks_dev/recorded_csv_before_derived_corrections"/config["experiment"]/config["name"]
        archive.mkdir(parents=True,exist_ok=True)
        if not (archive/"data.csv").exists():(archive/"data.csv").write_bytes(original)
        with (path/"data.csv").open("w",newline="") as f:
            writer=csv.writer(f);writer.writerow(columns);writer.writerows(np.column_stack(list(columns.values())))
    metrics["derived_data_review"]=dict(energy_max_correction_J=energy_error,robot_contact_samples_corrected=contact_changes,
        rationale="Use principal inertia frame for kinetic energy; classify unnamed fingertip contacts by body identity",
        original_csv_sha256=before_hash,compiled_com_reconstruction_max_error_m=com_error,
        simulation_states_modified=False,historical_forces_modified=False)
    dump(path/"metrics.json",metrics)
    return metrics["derived_data_review"]


def finalize(root,env):
    from dualstack.experiments.runner import load_groups,summarize
    root=Path(root).resolve();reviews={}
    for exp in TASKS:
        path=root/exp;manifest=json.loads((path/"manifest.json").read_text())
        if manifest["status"]=="running":raise RuntimeError(f"Wait for {exp} exports to finish")
        groups=load_groups(path,manifest);reviews[exp]={}
        for g in groups:
            reviews[exp][g["name"]]=enrich(g)
            for entry in manifest["groups"]:
                if entry["name"]==g["name"]:entry["metrics"]=g["metrics"]
        configs=json.loads((path/"evaluation_protocol.json").read_text())["configs"]
        stats=aggregate(groups,configs);manifest["task_evaluation"]=stats
        manifest["outcome_status"]="partial" if stats["evaluation_successes"]<stats["evaluation_attempted"] or any(not g["metrics"]["success"] for g in groups if g["config"]["split"]!="validation") else "passed"
        complete=len(groups)==len(configs) and manifest.get("curves",{}).get("status")=="passed" and manifest.get("timestep_comparison",{}).get("process",{}).get("status")=="passed"
        manifest["artifact_status"]="passed" if complete else "partial"
        pkg=package(exp,groups,path);zip_package(pkg,path/"replay.zip")
        validation,seconds=validate_zip(path/"replay.zip",path/"zip_validation",env,timeout=240)
        shutil.copy2(path/"zip_validation/validation.json",pkg/"validation.json")
        metadata=json.loads((pkg/"metadata.json").read_text());metadata.update(validation_status="passed",gui="回放数据和加载已验证，本地 GUI 待验证")
        dump(pkg/"metadata.json",metadata);zip_package(pkg,path/"replay.zip")
        manifest["replay"]=dict(status="passed",zip=str(path/"replay.zip"),validation=validation,validation_seconds=seconds)
        if validation["offscreen_status"]!="passed":manifest["artifact_status"]="partial"
        if any(v.get("status")!="passed" for v in manifest["videos"].values() if isinstance(v,dict)):manifest["artifact_status"]="partial"
        manifest["status"]="passed" if manifest["outcome_status"]==manifest["artifact_status"]=="passed" else "partial"
        dump(path/"manifest.json",manifest);dump(path/"evaluation.json",stats)
        summarize(path,manifest,groups);report(path,manifest,groups)
        print(exp,manifest["status"],stats["evaluation_successes"],"/",stats["evaluation_attempted"],flush=True)
    dump(root/"verification/derived_data_review.json",reviews)
