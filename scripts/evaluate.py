import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import json
from dualstack.envs.scene import ROOT
from dualstack.evaluation.runner import evaluate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(ROOT/'configs/default.json'))
    p.add_argument('--protocol',default=str(ROOT/'configs/evaluation.json'))
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--episodes',type=int)
    p.add_argument('--policy')
    p.add_argument('--fixed',action='store_true')
    p.add_argument('--failure-video',action='store_true')
    p.add_argument('--output',default='outputs/evaluation')
    a=p.parse_args()
    c=json.loads(Path(a.config).read_text(encoding='utf-8'))
    protocol=json.loads(Path(a.protocol).read_text(encoding='utf-8'))
    seeds=a.seeds if a.seeds is not None else protocol['seeds']
    if a.episodes:
        if a.episodes>len(seeds): p.error('--episodes exceeds number of supplied seeds')
        seeds=seeds[:a.episodes]
    if a.policy: c['policy']=a.policy
    evaluate(c,seeds,a.output,random=not a.fixed,save_failure_video=a.failure_video)

if __name__=='__main__': main()
