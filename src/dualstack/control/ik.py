import mink
import mujoco
import numpy as np

class DualIK:
    """Mink differential IK -> Menagerie joint position servos, never object state."""
    def __init__(self, model, dt):
        self.model, self.dt = model, dt
        self.configuration = mink.Configuration(model)
        self.tasks = [mink.FrameTask(frame_name=s+'_ee', frame_type='site', position_cost=1., orientation_cost=1., lm_damping=1.) for s in ['left', 'right']]
        self.posture = mink.PostureTask(model, cost=1e-3)
        self.limits = [mink.ConfigurationLimit(model)]

    def reset(self, data):
        self.configuration.update(data.qpos)
        self.posture.set_target_from_configuration(self.configuration)

    def apply(self, data, action):
        self.configuration.update(data.qpos)
        for i, task in enumerate(self.tasks):
            a = action[i*8:(i+1)*8]
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(a[3:7]), a[:3]))
        for _ in range(3):
            velocity = mink.solve_ik(self.configuration, self.tasks+[self.posture], self.dt, solver='daqp', damping=1e-5, limits=self.limits)
            self.configuration.integrate_inplace(velocity, self.dt)
        for i, side in enumerate(['left', 'right']):
            for j in range(7):
                joint = self.model.joint(f'{side}_joint{j+1}')
                data.ctrl[self.model.actuator(f'{side}_actuator{j+1}').id] = self.configuration.q[joint.qposadr[0]]
            data.ctrl[self.model.actuator(side+'_actuator8').id] = 255*action[i*8+7]
