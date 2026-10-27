import collections
import csv
import json
from dualstack.experiments.registry import dump


def aggregate(groups,configs):
    by_name={g["name"]:g for g in groups};strata={};failures=collections.Counter()
    for config in configs:
        if config["split"]!="evaluation":continue
        key=str(config.get("mu",config.get("weight_mass",config.get("difficulty"))))
        row=strata.setdefault(key,dict(attempted=0,successes=0,seeds=[],failures={}))
        row["attempted"]+=1;row["seeds"].append(config["seed"])
        g=by_name.get(config["name"])
        outcome=g["metrics"]["outcome"] if g else "execution_failed"
        if g and g["metrics"]["success"]:row["successes"]+=1
        else:failures[outcome]+=1;row["failures"][outcome]=row["failures"].get(outcome,0)+1
    for row in strata.values():row["success_rate"]=row["successes"]/row["attempted"]
    count=sum(r["attempted"] for r in strata.values());success=sum(r["successes"] for r in strata.values())
    roll={}
    for c in configs:
        if c["experiment"]=="07_tray" and c["split"]=="evaluation" and c["difficulty"]=="physics_variation":
            row=roll.setdefault(str(c["rolling_friction"]),dict(attempted=0,successes=0,seeds=[]))
            g=by_name.get(c["name"])
            row["attempted"]+=1;row["successes"]+=int(bool(g and g["metrics"]["success"]));row["seeds"].append(c["seed"])
    from .visual_config import VISUAL_TASKS
    contrasts={}
    if configs and configs[0]["experiment"] in VISUAL_TASKS:
        positive=[c for c in configs if c["expected_success"]];negative=[c for c in configs if not c["expected_success"]]
        contrasts=dict(positive_attempts=len(positive),positive_successes=sum(bool(by_name.get(c["name"],{}).get("metrics",{}).get("success")) for c in positive),
            negative_attempts=len(negative),negative_contrasts_observed=sum(bool(by_name.get(c["name"],{}).get("metrics",{}).get("expected_outcome_observed")) for c in negative))
    return dict(contrasts=contrasts,fixed_success_seen=any(g["config"]["split"]=="fixed" and g["metrics"]["success"] for g in groups),
        scheduled_episodes=len(configs),recorded_episodes=len(groups),evaluation_attempted=count,evaluation_successes=success,
        success_rate=success/count if count else None,strata=strata,rolling_variation=roll,failure_reasons=dict(failures),
        note="All predeclared seeds included; simulation task failure is distinct from a subprocess error")


def select_demos(groups):
    selected=[g for g in groups if g["config"].get("split")=="fixed"]
    failed=next((g for g in groups if not g["metrics"].get("success")),None)
    if failed and failed not in selected:selected.append(failed)
    if not any(g["metrics"].get("success") for g in selected):
        successful=next((g for g in groups if g["metrics"].get("success")),None)
        if successful and successful not in selected:selected.append(successful)
    return selected[:5] or groups[:1]


def report(path,manifest,groups):
    stats=manifest["task_evaluation"]
    with (path/"REPORT.md").open("a",encoding="utf-8") as f:
        f.write("\n\n## Robot evaluation (oracle)\n\n")
        if stats.get("contrasts"):
            cs=stats['contrasts']
            f.write(f"Predeclared positive scenarios: {cs['positive_successes']}/{cs['positive_attempts']} succeeded; physical negative contrasts observed: {cs['negative_contrasts_observed']}/{cs['negative_attempts']}. These are small deterministic scenarios, not a random-seed generalization estimate.\n\n")
        else:f.write(f"Fixed success observed: {stats['fixed_success_seen']}; predeclared evaluation: {stats['evaluation_successes']}/{stats['evaluation_attempted']}.\n\n")
        f.write("| Parameter/difficulty | Seeds attempted | Successes | Rate | Failure reasons |\n|---|---:|---:|---:|---|\n")
        for key,r in stats["strata"].items():f.write(f"| {key} | {r['attempted']} | {r['successes']} | {r['success_rate']:.1%} | {r['failures']} |\n")
        from .visual_config import VISUAL_TASKS
        if manifest["experiment"] in VISUAL_TASKS:
            f.write("\nImage/instruction-only model interface; this run uses explicitly privileged oracle experts. Full per-step numerical histories (including negative controls) remain in runs/. No replay package or ZIP generated. Hidden-mass observation uses one explicitly releasable fixture weld, disabled only after verified real grasp. No robot-to-object binding or runtime object pose overwrite.\n")
            f.write("\n| Case | Task success | Actual outcome | Expected contrast observed |\n|---|---|---|---|\n")
            for g in groups:f.write(f"| {g['name']} | {g['metrics']['success']} | {g['metrics']['outcome']} | {g['metrics']['expected_outcome_observed']} |\n")
        else:
            f.write("\nPrivileged state is accessible only through env.oracle; standard observations exclude object state and friction. Full episodes (including failures) are in replay.zip. Videos show fixed cases and one representative failure if present. No robot-to-object weld, mocap binding or runtime object pose overwrite is used; the original gripper joint equality is retained.\n")
        f.write(f"\nArtifact status: {manifest.get('artifact_status','see manifest')}; task outcome coverage: {manifest.get('outcome_status','see evaluation')}. Diagnostic validation timeouts are excluded from evaluation rates.\n")
    with (path/"evaluation.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["case","split","seed","success","reason","final_error_m","time_s","push_count","overshoot_count","max_ball_offset_m","max_tracking_error_m"])
        recorded={g["name"]:g for g in groups}
        configs=json.loads((path/"evaluation_protocol.json").read_text())["configs"]
        for c in configs:
            g=recorded.get(c["name"])
            if not g:
                w.writerow([c["name"],c["split"],c["seed"],False,"execution_failed",None,None,None,None,None,None]);continue
            c,m=g["config"],g["metrics"]
            w.writerow([g["name"],c["split"],c["seed"],m["success"],m["outcome"],m["final_error_m"],m["completion_time_s"],m["push_count"],m["overshoot_count"],m["max_ball_relative_offset_m"],m["max_tracking_error_m"]])
