import numpy as np
import mujoco
import pytest
from dualstack.tasks.config import configurations,SEEDS
from dualstack.tasks.env import TaskEnv
from dualstack.tasks.policies import PlacementOracle,PushOracle


def config(task):return configurations(task,"fixed")[1]


def test_predeclared_seeds_and_ordinary_observation_separation():
    assert SEEDS==list(range(10))
    for task in ("05_push","06_eccentric","07_tray"):
        configs=configurations(task,"evaluation")
        assert len(configs)==30
        assert [c["seed"] for c in configs[:10]]==SEEDS
    e=TaskEnv(config("05_push"));obs,_=e.reset()
    assert set(obs)=={"robot","images","task","time"}
    assert not {"mu","position","com","object","goal"}.intersection(obs)
    assert "position" in e.oracle.state()


@pytest.mark.parametrize("side",[-1,1])
def test_mass_distribution_and_rotated_com_placement(side):
    c=dict(config("06_eccentric"),weight_side=side,random=True,seed=4)
    e=TaskEnv(c);e.reset();s=e.oracle.state();p=PlacementOracle(e)
    expected=side*c["weight_offset"]*c["weight_mass"]/(c["bar_mass"]+c["weight_mass"])
    assert s["com_local"][0]==pytest.approx(expected)
    mass=e.model.body_mass[e.model.body("sample").id]
    assert mass==pytest.approx(c["bar_mass"]+c["weight_mass"])
    inertia=e.model.body_inertia[e.model.body("sample").id]
    assert np.all(inertia>0) and np.max(inertia)<np.sum(inertia)-np.max(inertia)
    np.testing.assert_allclose(p.place[:2]+(s["rotation"]@s["com_local"])[:2],s["goal"][:2],atol=1e-12)


def move_sample(e,position,velocity=None):
    j=e.model.joint("sample_joint");q=j.qposadr[0];v=j.dofadr[0]
    e.data.qpos[q:q+3]=position
    e.data.qvel[v:v+6]=np.zeros(6) if velocity is None else velocity
    mujoco.mj_forward(e.model,e.data)


def synthetic_contacts(e,table=True,platform=False,held=False):
    sid=e.model.body("sample").id
    contacts=[]
    if table:contacts.append((["sample_geom","table"],[sid,0],10.))
    if platform:contacts.append((["sample_geom","platform"],[sid,0],10.))
    if held:
        for finger in ("left_finger","right_finger"):
            contacts.append((["sample_geom","left_"+finger],[sid,e.model.body("left_"+finger).id],10.))
    e.contacts=lambda:contacts


def test_push_rejects_transit_floating_and_contact_with_pusher():
    e=TaskEnv(config("05_push"));e.reset();goal=e.initial["goal"]
    e.detector.push_seen=True
    move_sample(e,[*goal[:2],.775],[.3,0,0,0,0,0]);synthetic_contacts(e)
    assert not e.detector.update(1.)["success"]
    move_sample(e,[*goal[:2],.90]);synthetic_contacts(e,table=False)
    assert not e.detector.update(1.)["success"]
    move_sample(e,[*goal[:2],.775]);synthetic_contacts(e,held=True)
    assert not e.detector.update(1.)["success"]


def test_placement_rejects_held_floating_and_falling():
    e=TaskEnv(config("06_eccentric"));e.reset();s=e.oracle.state();e.detector.grasp_seen=True;e.detector.max_height=1.1
    position=np.r_[s["goal"][:2]-(s["rotation"]@s["com_local"])[:2],.838]
    move_sample(e,position);synthetic_contacts(e,table=False,platform=True,held=True)
    assert not e.detector.update(1.)["success"]
    move_sample(e,position+[0,0,.2]);synthetic_contacts(e,table=False)
    assert not e.detector.update(1.)["success"]
    move_sample(e,position,[0,0,-.5,0,0,0]);synthetic_contacts(e,table=False,platform=True)
    assert not e.detector.update(1.)["success"]


def test_tray_requires_both_real_grasps_and_not_only_target_pose():
    e=TaskEnv(config("07_tray"));e.reset();goal=e.initial["goal"]
    move_sample(e,goal);e.detector.max_height=goal[2]
    e.grasps=lambda:{"left":True,"right":False}
    result=e.detector.update(1.)
    assert not result["success"] and not result["conditions"]["right_grasp"]


def test_no_object_weld_or_hidden_push_walls():
    import xml.etree.ElementTree as ET
    for task in ("05_push","06_eccentric","07_tray"):
        e=TaskEnv(config(task));root=ET.fromstring(e.xml)
        for element in root.findall("equality/*"):
            assert element.tag!="weld"
            assert "sample" not in str(element.attrib)
        if task=="05_push":
            zone=root.find("worldbody/geom[@name='target_zone']")
            assert zone.get("contype")=="0" and zone.get("conaffinity")=="0"


def test_push_overshoot_replans_opposite_side_without_reading_mu():
    e=TaskEnv(config("05_push"));e.reset();p=PushOracle(e);p.phase="push_feedback";p.direction=np.array([1.,0.]);p.command=p.action[:3].copy()
    goal=np.array(e.initial["goal"])
    move_sample(e,[goal[0]+.05,goal[1],.775])
    p.act(e.observe());assert p.overshoots==1 and p.phase=="withdraw"
    p.plan();assert p.direction[0]<0 and not p.reads_friction


def test_physics_variation_changes_only_rolling_contact_parameter():
    from dualstack.tasks.config import initial_conditions
    a=configurations("07_tray","evaluation","easy")[0]
    b=configurations("07_tray","evaluation","physics_variation")[0]
    ia,ib=initial_conditions(a),initial_conditions(b)
    assert ia.pop("rolling_friction")!=ib.pop("rolling_friction")
    assert ia==ib


def test_seeded_reset_and_control_timestep_validation():
    c=dict(config("05_push"),random=True)
    e=TaskEnv(c);e.reset(seed=7);state=e.data.qpos.copy()
    e.reset(seed=7);np.testing.assert_array_equal(e.data.qpos,state)
    with pytest.raises(ValueError):TaskEnv(dict(c,dt=.003))


def test_unnamed_fingertip_geoms_still_count_as_robot_contact():
    e=TaskEnv(config("06_eccentric"));e.reset()
    sid=e.model.body("sample").id;finger=e.model.body("left_left_finger").id
    e.contacts=lambda:[(["sample_geom",""],[sid,finger],1.)]
    assert e.detector.update(0.)["robot_contact"]


def test_failed_processes_are_included_in_all_evaluation_denominators():
    from dualstack.tasks.reporting import aggregate
    result=aggregate([],configurations("07_tray","evaluation"))
    assert result["evaluation_attempted"]==30 and result["evaluation_successes"]==0
    assert result["failure_reasons"]=={"execution_failed":30}
    assert sum(r["attempted"] for r in result["rolling_variation"].values())==10


def test_minimal_non_oracle_policy_uses_same_interface_without_phase_or_events(tmp_path):
    from dualstack.tasks.recording import run
    c=dict(config("05_push"),duration=.04,policy="dualstack.policies.model_policy:ModelPolicy")
    result=run(c,tmp_path,record=False)
    assert result["execution"]=="completed" and result["outcome"]=="timeout"
    assert not result["oracle"] and not result["success"]
