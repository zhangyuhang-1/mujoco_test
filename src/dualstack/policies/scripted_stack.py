import numpy as np

class ScriptedStack:
    """Oracle expert; reads ground truth only through the separate expert API."""
    def __init__(self, env):
        self.env=env
        self.reset()

    def reset(self):
        state = self.env.expert_state()
        a,b = state['a'],state['b']
        self.a_initial_z=float(a[2])
        self.grasp_confirmed=False
        self.lift_confirmed=False
        self.initial = self.env.last_action.copy()
        self.arm = 0 if self.env.config['active_arm']=='left' else 1
        self.start = self.initial[self.arm*8:self.arm*8+3].copy()
        self.quat = np.array([0.,1.,0.,0.])
        place_height=self.env.config['a_half_size']+self.env.config['b_half_size']+.003
        self.waypoints = [
            (2.0, a+[0,0,.17], 1., 'approach'),
            (1.8, a+[0,0,.002], 1., 'descend'),
            (1.0, a+[0,0,.002], 0., 'close'),
            (1.8, a+[0,0,.20], 0., 'lift'),
            (2.0, np.r_[b[:2],a[2]+.20], 0., 'transfer'),
            (1.8, b+[0,0,place_height], 0., 'place'),
            (1.0, b+[0,0,place_height], 1., 'release'),
            (1.8, b+[0,0,.23], 1., 'retreat'),
            (5.0, b+[0,0,.23], 1., 'wait')]
        self.phase='approach'

    def act(self, observation):
        t=observation['time']
        if t>=4.8 and not self.grasp_confirmed:
            if not self.env.expert_grasp(self.env.config['active_arm']):
                raise RuntimeError('grasp_failure: no opposing finger contacts')
            self.grasp_confirmed=True
        if t>=6.6 and not self.lift_confirmed:
            if self.env.expert_state()['a'][2]<self.a_initial_z+.12:
                raise RuntimeError('lift_failure: object was not lifted')
            self.lift_confirmed=True
        previous=self.start
        for duration,target,grip,phase in self.waypoints:
            if t < duration:
                ratio=np.clip(t/duration,0,1)
                ratio=ratio*ratio*(3-2*ratio)
                position=previous+(target-previous)*ratio
                self.phase=phase
                break
            t-=duration
            previous=target
        else:
            position,grip=target,1.
            self.phase='wait'
        action=self.initial.copy()
        i=self.arm*8
        action[i:i+8]=np.r_[position,self.quat,grip]
        return action

    def close(self):
        pass
