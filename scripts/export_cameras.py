import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import json
from dualstack.envs.stack import StackEnv
from dualstack.envs.scene import ROOT
from dualstack.sensors.cameras import export_cameras

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(ROOT/'configs/default.json'))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--random',action='store_true')
    p.add_argument('--output',default='outputs/cameras')
    a=p.parse_args()
    env=StackEnv(json.loads(Path(a.config).read_text(encoding='utf-8')))
    try:
        env.reset(a.seed,{'random':a.random})
        print(json.dumps(export_cameras(env,a.output),indent=2))
    finally: env.close()

if __name__=='__main__': main()
