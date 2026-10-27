"""A deliberate failure policy for verifying evaluation exception diagnostics."""
class FailingPolicy:
    def __init__(self,env): pass
    def reset(self): pass
    def act(self,observation): raise RuntimeError('intentional inference exception for evaluation verification')
    def close(self): pass
