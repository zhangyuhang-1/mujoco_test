import importlib
from typing import Protocol

class Policy(Protocol):
    def reset(self): ...
    def act(self, observation): ...
    def close(self): ...

def load_policy(spec, env):
    module, name = spec.split(':')
    return getattr(importlib.import_module(module), name)(env)
