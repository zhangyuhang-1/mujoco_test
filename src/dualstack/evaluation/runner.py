import csv
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from dualstack.envs.stack import StackEnv
from dualstack.policies.base import load_policy

def evaluate(config, seeds, output, random=True, save_failure_video=False):
    if not seeds: raise ValueError('At least one seed is required')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/'protocol.json').write_text(json.dumps({'seeds':seeds,'config':config,'random':random},indent=2),encoding='utf-8')
    rows=[]
    env=StackEnv(config)
    try:
        for seed in seeds:
            obs,info=env.reset(seed,{'random':random})
            policy=load_policy(config['policy'],env)
            policy.reset()
            inference=0.
            video_frames=[]
            try:
                while True:
                    if save_failure_video and env.steps%5==0: video_frames.append(env.render(config['cameras']['overview'].get('name','overview')))
                    start=time.perf_counter()
                    try: action=policy.act(obs)
                    except Exception as exc:
                        inference+=time.perf_counter()-start
                        info.update(success=False,reason='inference_exception',exception=repr(exc))
                        break
                    inference+=time.perf_counter()-start
                    obs,_,terminated,truncated,info=env.step(action)
                    if terminated or truncated: break
            except Exception as exc:
                info.update(success=False,reason='execution_exception',exception=repr(exc))
            finally: policy.close()
            if save_failure_video and not info['success']:
                import imageio.v2 as imageio
                with imageio.get_writer(output/f'failure_seed_{seed}.mp4',fps=1/(config['control_dt']*5)) as writer:
                    for frame in video_frames: writer.append_data(frame)
            row={**info,'inference_seconds':inference}
            rows.append(row)
            (output/f'seed_{seed}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
            print(f"seed={seed} success={info['success']} reason={info['reason']} t={info['sim_time']:.2f}",flush=True)
    finally: env.close()
    successful=[r['sim_time'] for r in rows if r['success']]
    summary={'episodes':len(rows),'successes':len(successful),'success_rate':len(successful)/len(rows),'mean_success_sim_time':float(np.mean(successful)) if successful else None,'failure_reasons':dict(Counter(r['reason'] for r in rows if not r['success'])),'total_inference_seconds':sum(r['inference_seconds'] for r in rows)}
    (output/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (output/'episodes.csv').open('w',newline='',encoding='utf-8') as stream:
        keys=['seed','success','reason','sim_time','steps','xy_error','height_error','inference_seconds']
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(summary,flush=True)
    return summary
