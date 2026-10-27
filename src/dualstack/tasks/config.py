import numpy as np

TASKS = {
    "05_push": dict(title="Push into target", parameter="mu", values=[.15,.35,.65],duration=35.,dt=.002,metric="final_error_m"),
    "06_eccentric": dict(title="Eccentric object placement",parameter="weight_mass",values=[.04,.08,.16],duration=24.,dt=.002,metric="com_support_error_m"),
    "07_tray": dict(title="Bimanual tray transport",parameter="difficulty",values=["easy","medium","physics_variation"],duration=28.,dt=.002,metric="final_error_m"),
}
from .visual_config import VISUAL_TASKS
TASKS.update(VISUAL_TASKS)
SEEDS=list(range(10))  # Declared before evaluation; every seed is run, including failures.


def configurations(task,suite="all",difficulty="all"):
    if task in VISUAL_TASKS:
        from .visual_config import visual_cases
        return visual_cases(task,suite)
    spec=TASKS[task]
    variants=spec["values"] if difficulty=="all" or task!="07_tray" else [difficulty]
    result=[]
    for value in variants:
        tests=([("fixed",0)] if suite in ("fixed","all") else [])+([("evaluation",s) for s in SEEDS] if suite in ("evaluation","all") else [])
        for split,seed in tests:
            common=dict(experiment=task,dt=spec["dt"],duration=spec["duration"],control_dt=.02,
                seed=seed,split=split,random=split=="evaluation",group_budget_s=120,
                task_title=spec["title"],images=False,policy=f"dualstack.tasks.policies:{ {'05_push':'PushOracle','06_eccentric':'PlacementOracle','07_tray':'TrayOracle'}[task]}",
                record_interval_s=spec["dt"])
            common[spec["parameter"]]=value
            if task=="05_push": common.update(mass=.10,target_half_size=.055,max_pushes=4,reads_friction=False)
            if task=="06_eccentric": common.update(weight_side=1 if split=="fixed" or seed%2==0 else -1,bar_mass=.12,weight_offset=.055,platform_half_x=.028,gripper_kp=500.)
            if task=="07_tray": common.update(rolling_friction=([.0001,.002,.01][seed%3] if value=="physics_variation" else .002),ball_jitter=0. if value!="medium" else .008,gripper_kp=500.)
            common["name"]=f"{str(value).replace('.','p')}_{split}_{seed:02}"
            result.append(common)
    if task!="05_push":
        result.sort(key=lambda c:0 if c["split"]=="fixed" else 1)
    if suite=="all" and result:
        # An explicitly labeled short-budget counterexample, never counted in
        # evaluation success rates. Guarantees an honest incomplete-task replay
        # even when all conservative evaluation seeds succeed.
        negative=dict(result[0],name="validation_short_budget",split="validation",random=False,
                      duration={"05_push":2.6,"06_eccentric":6.,"07_tray":7.}[task],
                      validation_purpose="Incomplete/held/transit must not count as success; excluded from evaluation rate")
        result.append(negative)
    return result


def initial_conditions(config):
    rng=np.random.default_rng(config["seed"])
    random=config.get("random",False)
    task=config["experiment"]
    jitter=lambda size:rng.uniform(-size,size) if random else 0.
    if task=="05_push":
        return dict(position=[jitter(.015),-.32+jitter(.015),.775],yaw=jitter(.08),
                    goal=[.17+jitter(.015),-.27+jitter(.012),.751])
    if task=="06_eccentric":
        return dict(position=[jitter(.012),-.38+jitter(.012),.768],yaw=jitter(.08),goal=[.19,-.29,.82])
    difficulty=config["difficulty"]
    roll=config["rolling_friction"]
    return dict(position=[0.,0.,.756],yaw=0.,goal=[.09 if difficulty!="medium" else .15,.0 if difficulty!="medium" else .04,.88],
                ball_xy=[jitter(config["ball_jitter"]),jitter(config["ball_jitter"])],rolling_friction=roll)
