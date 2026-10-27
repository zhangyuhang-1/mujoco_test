import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import json
import imageio.v2 as imageio
import time
from dualstack.envs.scene import ROOT
from dualstack.envs.stack import StackEnv
from dualstack.policies.base import load_policy

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs/stage1')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--random',action='store_true')
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--config',default=str(ROOT/'configs/default.json'))
    parser.add_argument('--policy')
    parser.add_argument('--images',action='store_true',help='Include all enabled camera images in policy observations')
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.policy: config['policy']=args.policy
    if args.images: config['images']=True
    env=StackEnv(config)
    writer=viewer=policy=None
    try:
        obs,info=env.reset(args.seed,{'random':args.random})
        policy=load_policy(config['policy'],env)
        if args.viewer:
            import mujoco.viewer
            viewer=mujoco.viewer.launch_passive(env.model,env.data)
            viewer.cam.lookat[:]=[0,0,1]
            viewer.cam.distance=2.5
            viewer.cam.azimuth=135
            viewer.cam.elevation=-30
        writer=imageio.get_writer(out/'overview.mp4',fps=10) if args.video else None
        last=''
        for step in range(round(config['max_time']/config['control_dt'])+1):
            started=time.perf_counter()
            action=policy.act(obs)
            obs,reward,terminated,truncated,info=env.step(action)
            phase=getattr(policy,'phase','model')
            if last!=phase:
                print(phase,round(obs['time'],2),env.expert_state(),flush=True)
                if args.video:
                    for name,img in env.render().items(): imageio.imwrite(out/(phase+'_'+name+'.png'),img)
                last=phase
            if writer and step%5==0: writer.append_data(env.render(config['cameras']['overview'].get('name','overview')))
            if viewer:
                if not viewer.is_running(): break
                viewer.sync()
                time.sleep(max(0,config['control_dt']-(time.perf_counter()-started)))
            if terminated or truncated: break
        if args.video:
            for name,img in env.render().items(): imageio.imwrite(out/('final_'+name+'.png'),img)
        info['expert_grasp_confirmed']=getattr(policy,'grasp_confirmed',False)
        info['expert_lift_confirmed']=getattr(policy,'lift_confirmed',False)
        (out/'result.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
        print(info)
    finally:
        if writer: writer.close()
        if viewer: viewer.close()
        if policy: policy.close()
        env.close()

if __name__=='__main__': main()
