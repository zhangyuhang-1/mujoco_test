import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import json
from dualstack.envs.stack import StackEnv
from dualstack.envs.scene import ROOT
from dualstack.policies.base import load_policy
from dualstack.data.episodes import EpisodeWriter,inspect_episode,replay_video

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(ROOT/'configs/default.json'))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--random',action='store_true')
    p.add_argument('--policy')
    p.add_argument('--image-stride',type=int,default=5)
    p.add_argument('--output',default='outputs/demo/episode.h5')
    p.add_argument('--video',action='store_true')
    a=p.parse_args()
    config=json.loads(Path(a.config).read_text(encoding='utf-8'))
    if a.policy: config['policy']=a.policy
    env=StackEnv(config); writer=None; policy=None; complete=False
    try:
        obs,info=env.reset(a.seed,{'random':a.random})
        policy=load_policy(config['policy'],env)
        writer=EpisodeWriter(a.output,env,a.image_stride)
        writer.observation(obs)
        while True:
            action=policy.act(obs)
            obs,reward,terminated,truncated,info=env.step(action)
            writer.transition(action,reward,terminated,truncated,info)
            writer.observation(obs,force_image=terminated or truncated)
            if terminated or truncated: break
        complete=True
        print(json.dumps(info,indent=2))
    finally:
        if writer: writer.close(complete)
        if policy: policy.close()
        env.close()
    result=inspect_episode(a.output)
    Path(a.output).with_suffix('.check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['valid']: raise RuntimeError(result['errors'])
    if a.video: print(replay_video(a.output,Path(a.output).with_suffix('.mp4'),config['cameras']['overview'].get('name','overview')))

if __name__=='__main__': main()
