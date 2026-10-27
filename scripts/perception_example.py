import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dataclasses import asdict
from dualstack.envs.stack import StackEnv
from dualstack.sensors.cameras import camera_info
from dualstack.perception.base import EmptyPerception,ThirdPartyPerception
from dualstack.perception.oracle import OraclePerception

if __name__=='__main__':
    env=StackEnv()
    try:
        obs,_=env.reset(0)
        images=env.render(); calibration=camera_info(env)
        for model in [EmptyPerception(),ThirdPartyPerception(lambda images: []),OraclePerception(env)]:
            model.reset()
            print(type(model).__name__,[asdict(d) for d in model.predict(images,calibration,obs['robot'])])
    finally: env.close()
