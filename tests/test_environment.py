import copy
import json
import mujoco
import numpy as np
import pytest
from dualstack.envs.stack import StackEnv
from dualstack.evaluation.success import measure,update
from dualstack.sensors.cameras import camera_info,project,pixel_depth_to_world,export_cameras
from dualstack.data.episodes import inspect_episode
from dualstack.perception.base import EmptyPerception,Detection,ThirdPartyPerception

@pytest.fixture(scope='module')
def env():
    e=StackEnv()
    yield e
    e.close()

def test_lifecycle_observation_and_action_validation(env):
    obs,info=env.reset(3)
    assert set(obs)=={'robot','images','task','time'}
    assert 'a' not in obs and 'scene' not in obs
    assert all(len(obs['robot'][s]['qpos'])==7 for s in ['left','right'])
    obs,reward,terminated,truncated,info=env.step(env.last_action)
    assert obs['time']==pytest.approx(.02)
    assert not terminated and not truncated and reward==0
    for action in [np.zeros(15),np.full(16,np.nan),np.zeros(16)]:
        with pytest.raises(ValueError): env.step(action)

def test_seed_reproducibility(env):
    env.reset(19,{'random':True})
    state=env.data.qpos.copy(); scene=copy.deepcopy(env.scene)
    env.reset(4,{'random':True})
    env.reset(19,{'random':True})
    np.testing.assert_array_equal(state,env.data.qpos)
    assert scene==env.scene

def test_sampling_and_initial_contacts(env):
    for seed in range(50):
        env.reset(seed,{'random':True})
        a,b=env.expert_state().values()
        assert np.linalg.norm(a[:2]-b[:2])>.12
        for p in [a,b]: assert -.12<p[0]<.28 and -.5<p[1]<-.18
        ga,gb=env.model.geom('a_geom').id,env.model.geom('b_geom').id
        for c in env.data.contact:
            if ga in (c.geom1,c.geom2) or gb in (c.geom1,c.geom2):
                names=[mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,int(env.model.geom_bodyid[g])) or '' for g in [c.geom1,c.geom2]]
                assert not any('left_' in n or 'right_' in n for n in names)

def initialize_a(env,offset_z=.06,quat=(1,0,0,0)):
    """Synthetic INITIAL states for checker tests only; never used by policies."""
    env.reset(0)
    index=env.model.joint('a_free').qposadr[0]
    b=env.expert_state()['b']
    env.data.qpos[index:index+7]=[*b[:2],b[2]+offset_z,*quat]
    env.data.qvel[:]=0
    mujoco.mj_forward(env.model,env.data)

def test_success_requires_support_and_duration(env):
    initialize_a(env)
    for _ in range(100): mujoco.mj_step(env.model,env.data)
    env.data.time=0
    for _ in range(24):
        reward,terminated,truncated,info=update(env)
        assert not info['success']
    _,terminated,_,info=update(env)
    assert terminated and info['success'] and info['support_force']==pytest.approx(.5886,rel=.05)

def test_floating_and_tilted_are_rejected(env):
    initialize_a(env,offset_z=.09)
    conditions,_=measure(env)
    assert not conditions['supported']
    assert not update(env)[3]['success']
    initialize_a(env,quat=(np.cos(np.pi/8),np.sin(np.pi/8),0,0))
    assert not measure(env)[0]['upright']

def test_closed_gripper_and_transient_are_rejected(env):
    initialize_a(env)
    for _ in range(100): mujoco.mj_step(env.model,env.data)
    for _ in range(20): update(env)
    assert 0<env.held<.5
    for finger in [1,2]: env.data.qpos[env.model.joint(f'left_finger_joint{finger}').qposadr]=.02
    mujoco.mj_forward(env.model,env.data)
    conditions,_=measure(env)
    assert not conditions['released']
    assert not update(env)[3]['success'] and env.held==0
    initialize_a(env,offset_z=.12)
    assert not update(env)[3]['success'] and env.held==0

def test_timeout_and_fall(env):
    env.reset(0)
    env.data.time=env.config['max_time']
    _,terminated,truncated,info=update(env)
    assert truncated and not terminated and info['reason']=='timeout'
    env.data.qpos[env.model.joint('a_free').qposadr[0]+2]=.4
    mujoco.mj_forward(env.model,env.data)
    _,terminated,truncated,info=update(env)
    assert terminated and not truncated and info['reason']=='out_of_workspace'

def test_camera_projection_and_images(env,tmp_path):
    env.reset(0)
    calibration=camera_info(env)
    for info in calibration.values():
        point=env.expert_state()['a']
        uv,z=project([point],info)
        np.testing.assert_allclose(pixel_depth_to_world(uv[0],z[0],info),point,atol=1e-10)
    results=export_cameras(env,tmp_path)
    for name,result in results.items():
        assert result['image_std']>10
        for check in result['objects'].values(): assert check['within_projected_corners_2px'] is not False
    assert results['overview']['objects']['a']['visible_pixels']>0
    assert results['left_wrist']['objects']['a']['visible_pixels']>0
    assert results['right_wrist']['objects']['a']['visible_pixels']>0

def test_rejection_sampler_bounded(env):
    original=copy.deepcopy(env.config)
    try:
        env.config.update(a_xy=[0,-.35],b_xy=[0,-.35],random_xy=[0,0],sample_attempts=3)
        with pytest.raises(ValueError,match='exhausted'): env.reset(0,{'random':True})
    finally: env.config=original

def test_actually_held_object_not_success(env):
    from dualstack.envs.scene import ROOT
    import h5py
    path=ROOT/'outputs/release/episode.h5'
    if not path.exists(): path=ROOT/'outputs/demo/episode.h5'
    if not path.exists(): pytest.skip('Requires actual demonstration')
    # Restore a recorded physical grasp as an INITIAL checker test state.
    env.reset(0)
    with h5py.File(path,'r') as f:
        index=int(np.argmin(abs(f['observations/time'][:]-10.3)))
        env.data.qpos[:]=f['privileged/qpos'][index]
        env.data.qvel[:]=f['privileged/qvel'][index]
    mujoco.mj_forward(env.model,env.data)
    assert env.expert_grasp('left')
    assert not measure(env)[0]['released']
    assert not update(env)[3]['success']

def test_perception_empty_and_invalid():
    assert EmptyPerception().predict({}, {}, {})==[]
    with pytest.raises(ValueError): Detection('x','a',(5,0,1,2),1).validate()
    adapter=ThirdPartyPerception(lambda images:[{'camera':'bad','object_id':'a','bbox_xyxy':(0,0,5,5),'confidence':.5}])
    assert adapter.predict({}, {}, {})==[] and adapter.errors

def test_image_observation_and_camera_rename(env):
    config=copy.deepcopy(env.config)
    config['images']=True
    config['cameras']['overview']['name']='custom_overview'
    e=StackEnv(config)
    try:
        obs,_=e.reset(0)
        assert set(obs['images'])=={'custom_overview','left_wrist','right_wrist'}
        assert all(img.shape==(256,256,3) and img.dtype==np.uint8 for img in obs['images'].values())
    finally: e.close()

def test_invalid_data_report(tmp_path):
    import h5py
    path=tmp_path/'broken.h5'
    with h5py.File(path,'w') as f: f.create_dataset('actions',data=np.zeros((2,16)))
    assert not inspect_episode(path)['valid']

def test_actual_demo_integrity():
    from dualstack.envs.scene import ROOT
    path=ROOT/'outputs/release/episode.h5'
    if not path.exists(): path=ROOT/'outputs/demo/episode.h5'
    if not path.exists(): pytest.skip('Run scripts/collect.py to generate integration artifact')
    report=inspect_episode(path)
    assert report['valid'],report['errors']
    assert report['final_info']['success']
