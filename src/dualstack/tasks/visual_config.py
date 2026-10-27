"""Small, predeclared visual-only physical reasoning demonstrations."""
VISUAL_TASKS={
    "08_balance":dict(title="Long eccentric bar: geometry vs COM",parameter="strategy",values=["com","geometry"],duration=19.,dt=.002,metric="com_support_error_m"),
    "09_support":dict(title="Remove load-bearing support",parameter="strategy",values=["backup","direct"],duration=24.,dt=.002,metric="final_error_m"),
    "10_hidden":dict(title="Observe hidden ballast by natural tilt",parameter="weight_side",values=[1,-1],duration=25.,dt=.002,metric="com_support_error_m"),
    "11_pour":dict(title="Bimanual tilt and pour ball",parameter="receiver_shift",values=[0.,.015],duration=28.,dt=.002,metric="final_error_m"),
}
INSTRUCTIONS={"08_balance":"Place the long object stably on the narrow platform and release it.",
    "09_support":"Remove the marked support into the green region while keeping the beam and load stable.",
    "10_hidden":"Observe the natural tilt, then place the object stably on the narrow platform and release it.",
    "11_pour":"Tilt the box so its ball exits through the yellow opening into the green receiver."}


def visual_cases(task,suite="all"):
    s=VISUAL_TASKS[task]
    common=dict(experiment=task,dt=s["dt"],duration=s["duration"],control_dt=.02,seed=0,split="fixed",random=False,
        group_budget_s=180,task_title=s["title"],images=False,vision_only=True,no_package=True,
        gripper_kp=700.,bar_mass=.12,weight_mass=.20,weight_offset=.10,weight_side=1,platform_half_x=.012,
        record_interval_s=s["dt"],policy="dualstack.tasks.visual_policies:"+{"08_balance":"BalanceOracle","09_support":"SupportOracle","10_hidden":"HiddenOracle","11_pour":"PourOracle"}[task])
    result=[]
    if task=="08_balance":
        for side in (1,-1):
            for strategy in ("com","geometry"):
                result.append(dict(common,name=f"{'right' if side==1 else 'left'}_{strategy}",strategy=strategy,weight_side=side,expected_success=strategy=="com"))
        if suite!="fixed":
            for strategy in ("com","geometry"):
                result.append(dict(common,name=f"yaw_variant_{strategy}",split="variant",strategy=strategy,yaw=.04,initial_x=.008,expected_success=strategy=="com"))
    elif task=="09_support":
        for strategy in ("backup","direct"):
            result.append(dict(common,name=strategy,strategy=strategy,extra_bodies=["beam","payload"],expected_success=strategy=="backup"))
        if suite!="fixed":result.append(dict(common,name="backup_offset",strategy="backup",split="variant",load_offset=.075,extra_bodies=["beam","payload"],expected_success=True))
    elif task=="10_hidden":
        for side in (1,-1):result.append(dict(common,name="hidden_right" if side==1 else "hidden_left",weight_side=side,platform_half_x=.015,warmup_s=0.,extra_bodies=["fixture_pivot"],expected_success=True))
    else:
        for shift in ([0.] if suite=="fixed" else [0.,.015]):
            result.append(dict(common,name="receiver_fixed" if shift==0 else "receiver_shifted",receiver_shift=shift,extra_bodies=["ball"],expected_success=True))
    return result
