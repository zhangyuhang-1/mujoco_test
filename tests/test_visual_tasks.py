"""Input isolation, real mass geometry, fixture state and detector counterexamples."""
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from dualstack.tasks.visual_config import visual_cases
from dualstack.tasks.env import TaskEnv
from dualstack.tasks.visual_policies import BalanceOracle
from dualstack.experiments.physics import STATE


def config(task):return visual_cases(task,"fixed")[0]


def test_image_instruction_interface_and_hidden_mass_visual_observability():
    pictures=[];angles=[];instructions=[]
    for side in (-1,1):
        e=TaskEnv(dict(config("10_hidden"),weight_side=side,images=True))
        try:
            obs,_=e.reset()
            assert set(obs)=={"images","instruction"}
            assert set(obs["images"])=={"vision_overview"}
            rgb=obs["images"]["vision_overview"]
            assert rgb.shape==(256,256,3) and rgb.dtype==np.uint8
            pictures.append(rgb.copy());instructions.append(obs["instruction"])
            for _ in range(500):mujoco.mj_step(e.model,e.data)
            angles.append(e.oracle.state()["hinge_angle"])
        finally:e.close()
    np.testing.assert_array_equal(pictures[0],pictures[1])
    assert instructions[0]==instructions[1]
    assert angles[0]<-.12 and angles[1]>.12


@pytest.mark.parametrize("side",[-1,1])
def test_balance_actual_com_outside_geometric_alignment_and_corrected_margin(side):
    e=TaskEnv(dict(config("08_balance"),weight_side=side));e.reset()
    try:
        s=e.oracle.state();p=BalanceOracle(e)
        assert s["com_local"][0]==pytest.approx(side*.0625)
        assert abs(s["com_local"][0])-e.config["platform_half_x"]>.045
        assert e.model.geom("sample_geom").size[0]==pytest.approx(.16)
        assert e.model.geom("platform").size[0]==pytest.approx(.012)
        np.testing.assert_allclose(p.place[:2]+(s["rotation"]@s["com_local"])[:2],s["goal"][:2],atol=1e-12)
        assert e.initial["platform_top"]-.75>.14
        inertia=e.model.body("sample").inertia
        assert np.all(inertia>0) and 2*max(inertia)<sum(inertia)
    finally:e.close()


def test_fixture_release_requires_grasp_and_eq_active_survives_recording():
    e=TaskEnv(config("10_hidden"));e.reset()
    try:
        with pytest.raises(RuntimeError,match="physical grasp"):e.oracle.release_fixture()
        eq=e.model.equality("observation_fixture").id
        before=e.data.qpos.copy();v=e.data.qvel.copy()
        # Unit-test the guarded release using a confirmed-grasp stub; actual
        # bilateral contacts are checked separately in the recorded rollouts.
        e.grasp_body=lambda body,side:True
        e.oracle.release_fixture()
        np.testing.assert_array_equal(before,e.data.qpos);np.testing.assert_array_equal(v,e.data.qvel)
        assert not e.data.eq_active[eq]
        state=np.empty(mujoco.mj_stateSize(e.model,STATE));mujoco.mj_getState(e.model,e.data,state,STATE)
        e.data.eq_active[eq]=1;mujoco.mj_setState(e.model,e.data,state,STATE)
        assert not e.data.eq_active[eq]
    finally:e.close()


@pytest.mark.parametrize("task",["08_balance","09_support","10_hidden","11_pour"])
def test_no_robot_object_attachment_and_no_success_at_initialization(task):
    e=TaskEnv(config(task));_,info=e.reset()
    try:
        root=ET.fromstring(e.xml)
        welds=root.findall("equality/weld")
        assert len(welds)==int(task=="10_hidden")
        if welds:
            assert welds[0].get("body1")=="sample" and welds[0].get("body2")=="fixture_pivot"
        assert not root.findall(".//body[@mocap='true']")
        assert not info["success"]
        assert set(e.observe())=={"images","instruction"}
    finally:e.close()


def test_balance_detector_rejects_held_floating_and_transient_falling():
    e=TaskEnv(config("08_balance"));e.reset()
    try:
        goal=np.array(e.initial["goal"]);j=e.model.joint("sample_joint");q=j.qposadr[0];v=j.dofadr[0]
        e.data.qpos[q:q+3]=np.r_[goal[:2]-e.model.body("sample").ipos[:2],goal[2]+.018]
        e.data.qpos[q+3:q+7]=[1,0,0,0];e.data.qvel[v:v+6]=0;mujoco.mj_forward(e.model,e.data)
        e.detector.max_height=1.2;e.detector.grasp_seen=True
        sid=e.model.body("sample").id;finger=e.model.body("left_left_finger").id
        e.contacts=lambda:[(["sample_geom","platform"],[sid,0],3.),(["sample_geom",""],[sid,finger],1.)]
        assert not e.detector.update(1.)["success"]
        e.contacts=lambda:[]
        assert not e.detector.update(1.)["success"]
        e.contacts=lambda:[(["sample_geom","platform"],[sid,0],3.)]
        e.data.qvel[v+2]=-.5;mujoco.mj_forward(e.model,e.data)
        assert not e.detector.update(1.)["success"]
    finally:e.close()


def test_receiver_uses_inner_wall_faces_and_full_ball_radius():
    e=TaskEnv(config("11_pour"));e.reset()
    try:
        j=e.model.joint("ball_joint");q=j.qposadr[0];goal=np.array(e.initial["goal"])
        for offset,inside in ((.044,False),(.041,True)):
            e.data.qpos[q:q+3]=goal+[0,offset,.024];mujoco.mj_forward(e.model,e.data)
            info=e.detector.update(1.)
            assert info["conditions"]["ball_in_receiver"]==inside
            assert not info["success"]  # No verified outlet crossing or dual grasp.
    finally:e.close()
