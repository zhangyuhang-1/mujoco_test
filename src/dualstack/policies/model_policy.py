import numpy as np

class ModelPolicy:
    """Runnable holding policy showing the shared action interface; no task oracle."""
    def __init__(self, env): self.reset()
    def reset(self): self.steps=0
    def act(self,observation):
        self.steps+=1
        # Replace this with model inference. 2D boxes require depth/pose first.
        return np.concatenate([np.r_[observation['robot'][s]['ee_pose'],1.] for s in ['left','right']])
    def close(self): pass
