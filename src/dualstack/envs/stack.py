import json
import pathlib
import mujoco
import numpy as np
from dualstack.envs.scene import ROOT, build_scene
from dualstack.control.ik import DualIK
from dualstack.evaluation.success import update

class StackEnv:
    def __init__(self, config=None):
        self.config = json.loads((ROOT/'configs/default.json').read_text()) if config is None else config
        if self.config['active_arm'] not in ['left','right']:
            raise ValueError('active_arm must be left or right')
        ratio=self.config['control_dt']/self.config['physics_dt']
        if self.config['control_dt']<=0 or self.config['physics_dt']<=0 or not np.isclose(ratio,round(ratio)):
            raise ValueError('control_dt must be a positive integer multiple of physics_dt')
        self.model = mujoco.MjModel.from_xml_string(build_scene(self.config))
        self.camera_names = [value.get('name',role) for role,value in self.config['cameras'].items()]
        self.data = mujoco.MjData(self.model)
        self.controller = DualIK(self.model, self.config['control_dt'])
        self.renderer = None
        self.steps = 0
        self.held = 0.
        self.done = False

    def pose(self, side):
        site = self.model.site(side+'_ee').id
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site])
        return np.r_[self.data.site_xpos[site], quat]

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.seed = seed
        rng = np.random.default_rng(seed)
        random = (options or {}).get('random', self.config['random'])
        a, b = np.array(self.config['a_xy']), np.array(self.config['b_xy'])
        if random:
            for _ in range(self.config['sample_attempts']):
                aa = a+rng.uniform(-np.array(self.config['random_xy']), self.config['random_xy'])
                bb = b+rng.uniform(-np.array(self.config['random_xy']), self.config['random_xy'])
                if np.linalg.norm(aa-bb) > .12 and all(-.12 < p[0] < .28 and -.5 < p[1] < -.18 for p in [aa,bb]):
                    a,b = aa,bb
                    break
            else:
                raise ValueError('Scene rejection sampling exhausted')
        yaw = rng.uniform(-self.config['random_yaw'], self.config['random_yaw']) if random else 0.
        a_joint = self.model.joint('a_free').qposadr[0]
        self.data.qpos[a_joint:a_joint+7] = [*a,self.config['table_top']+self.config['a_half_size'],np.cos(yaw/2),0,0,np.sin(yaw/2)]
        self.model.body('block_b').pos[:] = [*b,self.config['table_top']+self.config['b_half_size']]
        for side in ['left','right']:
            for j,q in enumerate([0,0,0,-1.57079,0,1.57079,-.7853,.04,.04]):
                name = f'{side}_joint{j+1}' if j<7 else f'{side}_finger_joint{j-6}'
                self.data.qpos[self.model.joint(name).qposadr] = q
            for j,q in enumerate([0,0,0,-1.57079,0,1.57079,-.7853,255]):
                self.data.ctrl[self.model.actuator(f'{side}_actuator{j+1}').id] = q
        mujoco.mj_forward(self.model,self.data)
        for _ in range(100):
            mujoco.mj_step(self.model,self.data)
        self.data.time = 0.
        mujoco.mj_forward(self.model,self.data)
        self.steps,self.held = 0,0.
        self.done=False
        self.controller.reset(self.data)
        self.scene = {'a_xy': a.tolist(), 'b_xy': b.tolist(), 'a_yaw': float(yaw), 'random': bool(random)}
        self.last_action = np.r_[self.pose('left'),1.,self.pose('right'),1.]
        return self.observe(), self.info()

    def expert_state(self):
        return {name: self.data.body('block_'+name).xpos.copy() for name in ['a','b']}

    def expert_grasp(self,side):
        """Privileged two-sided finger contact, used only by the oracle expert."""
        ga=self.model.geom('a_geom').id
        contacts=set()
        for i in range(self.data.ncon):
            c=self.data.contact[i]
            if ga in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==ga else c.geom1
                body=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[other]))
                if body in (side+'_left_finger',side+'_right_finger'): contacts.add(body)
        return len(contacts)==2

    def observe(self):
        robot = {}
        for side in ['left','right']:
            joints = [self.model.joint(f'{side}_joint{j+1}') for j in range(7)]
            robot[side] = {'qpos': np.array([self.data.qpos[j.qposadr[0]] for j in joints]), 'qvel': np.array([self.data.qvel[j.dofadr[0]] for j in joints]), 'ee_pose': self.pose(side), 'gripper_width': sum(self.data.qpos[self.model.joint(f'{side}_finger_joint{i}').qposadr[0]] for i in [1,2])}
        return {'robot': robot, 'images': self.render() if self.config['images'] else {}, 'task': 'Place red block A on blue block B and release.', 'time': self.data.time}

    def info(self):
        a,b = self.expert_state().values()
        return {'success': False, 'reason': 'running', 'sim_time': float(self.data.time), 'steps': self.steps, 'xy_error': float(np.linalg.norm(a[:2]-b[:2])), 'height_error': float(abs(a[2]-b[2]-self.config['a_half_size']-self.config['b_half_size'])), 'hold_time': self.held, 'seed': self.seed, 'scene': self.scene}

    def step(self, action):
        if self.done: raise RuntimeError('Episode ended; call reset before step')
        action = np.asarray(action, dtype=float)
        if action.shape != (16,) or not np.all(np.isfinite(action)):
            raise ValueError('Action must be finite shape (16,)')
        for i in [0,8]:
            norm = np.linalg.norm(action[i+3:i+7])
            if abs(norm-1)>1e-3 or not 0 <= action[i+7] <= 1:
                raise ValueError('Expected unit wxyz quaternion and gripper in [0,1]')
            if np.any(action[i:i+3] < [-.65,-.85,.76]) or np.any(action[i:i+3] > [.5,.85,1.7]):
                raise ValueError('EE target outside configured safe workspace')
        self.controller.apply(self.data,action)
        for _ in range(round(self.config['control_dt']/self.config['physics_dt'])):
            mujoco.mj_step(self.model,self.data)
        # Synchronize derived transforms/contact forces with the final integrated qpos.
        mujoco.mj_forward(self.model,self.data)
        self.last_action = action.copy()
        self.steps += 1
        reward,terminated,truncated,info=update(self)
        self.done=terminated or truncated
        return self.observe(),reward,terminated,truncated,info

    def render(self, camera=None):
        if self.renderer is None:
            h,w = self.config['resolution']
            self.renderer = mujoco.Renderer(self.model,height=h,width=w)
        images = {}
        for name in ([camera] if camera else self.camera_names):
            self.renderer.update_scene(self.data,camera=name)
            images[name] = self.renderer.render().copy()
        return images[camera] if camera else images

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
