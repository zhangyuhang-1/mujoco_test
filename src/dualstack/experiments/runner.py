import argparse
import csv
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np
import mujoco
from .registry import ROOT, REGISTRY, TASKS, cases, dump
from dualstack.tasks.visual_config import VISUAL_TASKS

ENTRY = ROOT / "scripts/experiments.py"
ACTIVE_CHILDREN = set()


def terminate_children(signum, frame):
    import signal
    for pid in list(ACTIVE_CHILDREN):
        try: os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    raise SystemExit(128+signum)


def environment():
    result = dict(python=sys.executable, python_version=platform.python_version(), mujoco=mujoco.__version__,
                  platform=platform.platform(), DISPLAY=os.environ.get("DISPLAY"), packages={},
                  physics_import_backend="disable",video_backend="osmesa",backend=os.environ.get("MUJOCO_GL", "disable"))
    for name in ("numpy", "scipy", "mink", "Pillow", "imageio", "imageio-ffmpeg"):
        result["packages"][name] = importlib.metadata.version(name)
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], capture_output=True, text=True, timeout=8)
        result["gpu_query"] = dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as e:
        result["gpu_query"] = dict(error=repr(e))
    usage = shutil.disk_usage(ROOT)
    result["disk_free_bytes"] = usage.free
    result["no_global_environment_modification"] = True
    return result


def child_env(backend="osmesa"):
    env = os.environ.copy()
    env.update(MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend if backend in ("osmesa", "egl") else "",
               MESA_SHADER_CACHE_DISABLE="true", LP_NUM_THREADS="4", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    return env


def child(command, logfile, timeout, backend="osmesa"):
    logfile = Path(logfile); logfile.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with logfile.open("w") as log:
            # One process group so FFmpeg cannot survive a timed-out renderer.
            p = subprocess.Popen([sys.executable, str(ENTRY), *map(str, command)], cwd=ROOT, env=child_env(backend),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            ACTIVE_CHILDREN.add(p.pid)
            try:
                code = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
                ACTIVE_CHILDREN.discard(p.pid)
                return dict(status="failed", reason="timeout", timeout_s=timeout, log=str(logfile), seconds=time.perf_counter()-started)
            ACTIVE_CHILDREN.discard(p.pid)
        return dict(status="passed" if code == 0 else "failed", returncode=code, log=str(logfile), seconds=time.perf_counter()-started)
    except Exception as e:
        return dict(status="failed", reason=repr(e), log=str(logfile), seconds=time.perf_counter()-started)


def render_with_fallback(root, output, view, fps, budget, backend="osmesa"):
    attempts = []
    for width, height in ((1920,1080), (1280,720), (640,360), (320,180)):
        path = output / f"{view}.mp4"
        detail = output / f"{view}_render.json"
        detail.unlink(missing_ok=True)
        result = child(["_render", "--package", root, "--output", path, "--view", view, "--width", width, "--height", height, "--fps", fps],
                       output / f"{view}_{width}.log", budget, backend)
        if detail.exists():
            data = json.loads(detail.read_text())
            # A stale previous-level record must not mark the current attempt passed.
            if data.get("resolution") == [width,height]: result["details"] = data
        result["resolution"] = [width,height]
        attempts.append(result)
        if result["status"] == "passed":
            return dict(result, attempts=attempts.copy(), degraded=width<1920)
    # Encoding might be unavailable even with working OpenGL; independent image fallback.
    fallback = child(["_render", "--package", root, "--output", output / f"{view}_png", "--view", view, "--width", 320, "--height", 180, "--fps", 10, "--png"],
                     output / f"{view}_png.log", budget, backend)
    attempts.append(fallback)
    if fallback["status"] == "passed":
        (output / "ENCODE_PNG.md").write_text(f"PNG fallback at 10 fps. On your local machine: ffmpeg -framerate 10 -i {view}_png/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {view}.mp4\n")
    return dict(status="partial" if fallback["status"] == "passed" else "failed", attempts=attempts, degraded=True)


def load_groups(path, manifest):
    result = []
    for group in manifest["groups"]:
        if group["status"] != "passed": continue
        p = path / "runs" / group["name"]
        result.append(dict(name=group["name"], path=p, config=json.loads((p/"config.json").read_text()),
                           metrics=json.loads((p/"metrics.json").read_text())))
    return result


def trend(experiment, groups):
    if experiment in TASKS:
        successes=sum(bool(g["metrics"].get("success")) for g in groups)
        return dict(status="measured",description="Oracle task success; failures retained without seed selection",successes=successes,recorded_episodes=len(groups),independent_from_execution_status=True)
    if len(groups) != len(cases(experiment)):
        return dict(status="not_evaluated", reason="Incomplete groups")
    metrics = [g["metrics"] for g in groups]
    if experiment == "01_friction":
        vals = [m["sliding_distance_m"] for m in metrics]
        matches = all(vals[i] > vals[i+1] for i in range(len(vals)-1))
        detail = "Sliding distance decreases with effective friction"
    elif experiment == "02_bounce":
        vals = [m["first_rebound_height_m"] for m in metrics]
        matches = all(v is not None for v in vals) and all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
        detail = "Rebound height decreases with direct contact damping; solref damping is NOT restitution"
    elif experiment == "03_stack":
        vals = [m["outcome"] for m in metrics]
        matches = vals[0] == "stable" and vals[-1] in ("tipped", "slid_off")
        detail = "Small offset stable; large offset loses support, measured from motion"
    else:
        vals = [m["outcome"] for m in metrics]
        matches = all(abs(m["final_equilibrium_error_m"]) < .001 for m in metrics)
        detail = "Final position approaches gravity equilibrium; finite-budget oscillation/settling retained in data"
    return dict(status="matches" if matches else "does_not_match", description=detail, measured=vals,
                independent_from_execution_status=True)


def summarize(path, manifest, groups):
    dump(path / "summary.json", dict(experiment=manifest["experiment"], status=manifest["status"],
         trend=manifest.get("trend"), timestep_comparison=manifest.get("timestep_comparison"),
         groups=[dict(name=g["name"], parameters=g["config"], metrics=g["metrics"]) for g in groups],
         replay=manifest.get("replay"), videos=manifest.get("videos")))
    lines = [f"# {manifest['experiment']} — {REGISTRY[manifest['experiment']]['title']}", "", f"Execution/output status: **{manifest['status']}**", "",
             "Numerical outcomes are independent from execution status and expected trend.", "", "| Case | Parameter | Observed outcome | Main metric | Max penetration (m) |", "|---|---|---|---|---|"]
    from .artifacts import label
    for g in groups:
        metric = REGISTRY[manifest["experiment"]]["metric"]
        lines.append(f"| {g['name']} | {label(g['config'])} | {g['metrics']['outcome']} | {g['metrics'].get(metric)} | {g['metrics']['max_penetration_m']:.6g} |")
    replay_note = "回放数据和加载已验证，本地 GUI 待验证。" if manifest.get("replay",{}).get("status")=="passed" else "回放验证尚未通过或已主动关闭；本地 GUI 待验证。请查看 manifest 的 replay 状态。"
    lines += ["", "## Trend check", "", json.dumps(manifest.get("trend"), ensure_ascii=False), "", "## Two-timestep check", "", "```json", json.dumps(manifest.get("timestep_comparison"), indent=2), "```", "", "## Outputs and limitations", "", "- `replay.zip`: portable XML/assets/full trajectory/player; server ZIP extraction + all-pose/offscreen validation recorded in manifest.", "- "+replay_note, "- `runs/*/data.csv`, `contact_history.npz`, `trajectory.npz`: historical numerical data at every physics step (trajectory only if requested).", "- `curves.png`: comparison plots. `videos/`: overview, detail, and bounce slow motion with actual encoded resolution/fps in JSON.", "- `manifest.json`: failures, fallback attempts, budgets, timings, software versions, precise paths.", "- Replayed poses do not guarantee exact historical contact force reconstruction; use recorded force history.", "- Numerical soft-contact simulations are demonstrations, not experimental validation of real material parameters."]
    if manifest.get("button_control"):
        lines += ["", "## Button control selection", "", "```json", json.dumps(manifest["button_control"], indent=2), "```"]
    if manifest["experiment"] in VISUAL_TASKS:
        lines=[line for line in lines if "replay.zip" not in line and "GUI" not in line]
        lines += ["", "本次仅视频交付：未建立回放包，未生成 ZIP。服务器逐物理步记录保留在 runs/，用于离线渲染及数值审查。",
                  "模型普通观测仅 images 和 instruction；本次脚本专家通过独立 oracle 接口读取真值。model_input_examples/ 为无叠字原始相机输入示例。",
                  "expected_outcome_observed 依据真实 success/tipped/structure_collapse 判断；负对照的任务成功仍为 false。"]
        lines += ["", "限制：本次夹指抓持期间软接触最大穿透约5–7 mm，数值步长比较不等于真实硬接触精度验证。"]
    (path / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def experiment(args):
    exp = args.experiment
    path = args.output.resolve()
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / "manifest.json"
    configs = cases(exp,suite=args.suite,difficulty=args.difficulty)
    if args.images:
        for config in configs:config["images"]=True
    if exp in TASKS and args.policy:
        for config in configs:config["policy"]=args.policy
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        same_cases=[g["name"] for g in manifest["groups"]]==[c["name"] for c in configs]
        same_configs=True
        if exp in TASKS and same_cases:
            for config in configs:
                saved=path/"runs"/config["name"]/"config.json"
                previous=json.loads(saved.read_text()) if saved.exists() else {}
                if {k:v for k,v in previous.items() if k!="group_budget_s"}!={k:v for k,v in config.items() if k!="group_budget_s"}:
                    same_configs=False;break
        exported_partial=manifest.get("artifact_status")=="passed" and manifest.get("outcome_status")=="partial"
        if (manifest["status"] == "passed" or exported_partial) and same_cases and same_configs and not args.refresh_analysis:
            print(f"Resume: {exp} exports complete; task outcome {manifest.get('outcome_status',manifest['status'])}", flush=True); return
        previous={g["name"]:g for g in manifest["groups"]}
        manifest["groups"]=[previous.get(c["name"],dict(name=c["name"],status="pending")) for c in configs]
    else:
        manifest = dict(experiment=exp, status="pending", groups=[dict(name=c["name"], status="pending") for c in configs], videos={}, replay={},
                        budgets=dict(group_s=args.group_budget, render_attempt_s=args.render_budget, experiment_s=args.experiment_budget), environment=environment())
    manifest.update(suite=args.suite,difficulty=args.difficulty)
    if exp in TASKS:
        from dualstack.tasks.config import SEEDS
        dump(path/"evaluation_protocol.json",dict(seeds=sorted({c['seed'] for c in configs}) if exp in VISUAL_TASKS else SEEDS,declared_before_rollouts=True,configs=configs,success_detector="dualstack.tasks.visual_eval.VisualDetector" if exp in VISUAL_TASKS else "dualstack.tasks.evaluation.Detector",oracle=True,reads_friction=False))
    manifest["status"] = "running"; dump(manifest_path, manifest)
    missing = [name for name in ("mj_getState", "mj_setState", "mj_stateSize", "mj_contactForce") if not hasattr(mujoco, name)]
    if missing:
        manifest.update(status="skipped", reason=f"Required APIs unavailable; no reliable recording substitute: {missing}")
        for group in manifest["groups"]:
            group.update(status="skipped", reason=manifest["reason"])
        dump(manifest_path, manifest)
        return
    # Prefer actual arm contact; only one bounded diagnostic attempt, then deterministic known force fallback.
    if exp == "04_button":
        probe_dir = path / "robot_press_probe"
        probe = dict(configs[1], name="robot_probe", press_mode="robot", duration=4.5, release_time_s=3.3, group_budget_s=args.group_budget)
        dump(probe_dir / "input.json", probe)
        p = child(["_physics", "--config", probe_dir / "input.json", "--output", probe_dir], probe_dir / "run.log", args.group_budget)
        usable = False
        reason = "Robot diagnostic process failed"
        if p["status"] == "passed":
            from .artifacts import read_csv
            a = read_csv(probe_dir / "data.csv")
            eq = -.981/probe["stiffness"]
            contact = a["contact_count"] > 0
            usable = bool(np.max(eq-a["joint_q_m"]) > .005 and np.any(contact) and not np.any(contact[a["time_s"] > 3.5]))
            reason = "Robot contact did not produce >=5 mm reliable press followed by full separation" if not usable else "Measured arm contact, depression and separation passed diagnostic"
        manifest["button_control"] = dict(mode="robot" if usable else "known_external_force", probe=p, reason=reason,
                                           fallback=not usable, max_diagnostic_attempts=1)
        for c in configs:
            c.update(press_mode="robot" if usable else "force", release_time_s=3.3 if usable else 1., duration=6.0 if usable else 4.0,
                     simplification_reason=None if usable else reason)
        dump(manifest_path, manifest)
    for i, c in enumerate(configs):
        g = manifest["groups"][i]
        c["group_budget_s"] = args.group_budget
        out = path / "runs" / c["name"]
        if args.resume and g["status"] == "passed" and (out / "config.json").exists():
            previous = json.loads((out / "config.json").read_text())
            if {k:v for k,v in previous.items() if k!="group_budget_s"} == {k:v for k,v in c.items() if k!="group_budget_s"}:
                continue
            g["previous_configuration"] = previous
        g["status"] = "running"; dump(manifest_path, manifest)
        dump(out / "input.json", c)
        command = ["_physics", "--config", out / "input.json", "--output", out]
        if c.get("images"):command.append("--images")
        if args.no_trajectory: command += ["--no-trajectory"]
        result = child(command, out / "run.log", args.group_budget)
        g.update(result)
        if result["status"] == "passed":
            g["metrics"] = json.loads((out / "metrics.json").read_text())
        dump(manifest_path, manifest)
        print(f"{exp}/{c['name']}: {g['status']} {g.get('metrics',{}).get('outcome','')}", flush=True)
    groups = load_groups(path, manifest)
    if not groups:
        manifest.update(status="failed", reason="All parameter groups failed")
        dump(manifest_path, manifest); return
    # Recompute derived metrics from immutable numerical history on resume too.
    from .physics import metrics
    for group in groups:
        if exp in TASKS:continue
        rows = np.genfromtxt(group["path"] / "data.csv", delimiter=",", skip_header=1)
        group["metrics"].update(metrics(rows, group["config"]))
        dump(group["path"] / "metrics.json", group["metrics"])
        for entry in manifest["groups"]:
            if entry["name"] == group["name"]: entry["metrics"] = group["metrics"]
    # Representative group, same physical duration/control profile and a halved time step.
    representative = next((g for g in groups if g["metrics"].get("success") and g["config"].get("split")=="fixed"),groups[len(groups)//2])
    refined = dict(representative["config"], dt=representative["config"]["dt"]/2, name="timestep_half")
    check_dir = path / "timestep_check"
    dump(check_dir / "input.json", refined)
    check = child(["_physics", "--config", check_dir / "input.json", "--output", check_dir, "--no-trajectory"]+(["--images"] if refined.get("images") else []), check_dir / "run.log", args.group_budget)
    comparison = dict(process=check, case=representative["name"], dt_s=[representative["config"]["dt"], refined["dt"]])
    if check["status"] == "passed":
        other = json.loads((check_dir / "metrics.json").read_text())
        key = REGISTRY[exp]["metric"]
        a, b = representative["metrics"].get(key), other.get(key)
        comparison.update(metric=key, values=[a,b], absolute_difference=abs(a-b) if a is not None and b is not None else None,
                          outcomes=[representative["metrics"]["outcome"], other["outcome"]],
                          max_penetration_m=[representative["metrics"]["max_penetration_m"], other["max_penetration_m"]],
                          initial_conditions=[representative["metrics"]["initial_conditions"], other["initial_conditions"]],
                          interpretation="Sensitivity comparison, not proof of physical truth or framewise equality")
    manifest["timestep_comparison"] = comparison
    manifest["trend"] = trend(exp, groups)
    from .artifacts import plot, package, zip_package, validate_zip, label
    failures = len(groups) != len(configs) or check["status"] != "passed"
    demo_groups=groups
    if exp in TASKS:
        from dualstack.tasks.reporting import aggregate,select_demos
        manifest["task_evaluation"]=aggregate(groups,configs)
        failures |= not any(g["metrics"].get("success") for g in groups)
        if any(c["split"]=="fixed" for c in configs):
            failures |= not manifest["task_evaluation"]["fixed_success_seen"]
        demo_groups=select_demos(groups)
        dump(path/"evaluation.json",manifest["task_evaluation"])
    try:
        plot(exp, demo_groups, path / "curves.png")
        manifest["curves"] = dict(status="passed", path=str(path / "curves.png"))
    except Exception as e:
        manifest["curves"] = dict(status="failed", reason=repr(e))
        failures = True
    if args.no_package or exp in VISUAL_TASKS:
        manifest["replay"]=dict(status="skipped",reason="Video-only delivery: no replay package or ZIP requested")
    elif not args.no_trajectory:
        try:
            pkg = package(exp, groups, path)
            zip_path = path / "replay.zip"
            zip_package(pkg, zip_path)
            validation, seconds = validate_zip(zip_path, path / "zip_validation", child_env(), timeout=args.group_budget)
            shutil.copy2(path / "zip_validation/validation.json", pkg / "validation.json")
            meta=json.loads((pkg / "metadata.json").read_text())
            meta.update(validation_status="passed",gui="回放数据和加载已验证，本地 GUI 待验证")
            dump(pkg / "metadata.json",meta)
            zip_package(pkg, zip_path)
            manifest["replay"] = dict(status="passed", zip=str(zip_path), validation=validation, validation_seconds=seconds)
            failures |= validation.get("offscreen_status") == "failed"
        except Exception as e:
            manifest["replay"] = dict(status="failed", reason=repr(e), traceback=traceback.format_exc())
            failures = True
        dump(manifest_path, manifest)
    else:
        pkg = None
        manifest["replay"] = dict(status="skipped", reason="Explicit --no-trajectory option")
    if not args.no_trajectory:
        dump(path/"recording_source/metadata.json",dict(experiment=exp,groups=[dict(name=g["name"],label=label(g["config"]),config=g["config"],metrics=g["metrics"],scene=str(g["path"]/"scene.xml"),trajectory=str(g["path"]/"trajectory.npz")) for g in groups]))
        if exp in VISUAL_TASKS:
            observation_export=child(["_observations","--package",path/"recording_source","--output",path/"model_input_examples"],path/"observation_export.log",args.group_budget)
            manifest["model_input_examples"]=observation_export
            failures |= observation_export["status"]!="passed"
    if not args.no_video:
        # Independent package construction for offline rendering, even if replay validation failed.
        if args.no_trajectory:
            manifest["videos"] = dict(status="skipped", reason="--no-trajectory requires --no-video")
            failures = True
        else:
            # Video is independent of ZIP export. Absolute original paths are fine
            # for the server-only render source, never used in the portable ZIP.
            pkg = path / "render_source"
            dump(pkg / "metadata.json", dict(experiment=exp, groups=[dict(name=g["name"], label=label(g["config"]),
                 config=g["config"], metrics=g["metrics"], scene=str(g["path"] / "scene.xml"),
                 trajectory=str(g["path"] / "trajectory.npz")) for g in demo_groups]))
            for view in ["overview", "detail"] + (["slowmotion"] if exp=="02_bounce" else []):
                fps = 60 if exp=="02_bounce" else 30
                result = render_with_fallback(pkg, path / "videos", view, fps, args.render_budget)
                manifest["videos"][view] = result
                failures |= result["status"] != "passed" or result.get("degraded",False)
                dump(manifest_path, manifest)
                print(f"{exp} video {view}: {result['status']} {result.get('resolution')}", flush=True)
    else:
        manifest["videos"] = dict(status="skipped", reason="Explicit --no-video/--quick option")
    manifest["artifact_status"]="partial" if failures else "passed"
    if exp in TASKS:
        outcome_partial=any(not g["metrics"].get("expected_outcome_observed") for g in groups) if exp in VISUAL_TASKS else any(not g["metrics"].get("success") for g in groups if g["config"]["split"]!="validation")
        manifest["outcome_status"]="partial" if outcome_partial else "passed"
        manifest["status"]="partial" if failures or outcome_partial else "passed"
    else:
        manifest["status"] = "partial" if failures else "passed"
    manifest["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dump(manifest_path, manifest)
    summarize(path, manifest, groups)
    if exp in TASKS:
        from dualstack.tasks.reporting import report
        report(path,manifest,groups)


def batch(args):
    root = args.output.resolve()
    # Keep every mutation within the authorized project folder.
    if not root.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Output must be inside {ROOT}")
    root.mkdir(parents=True, exist_ok=True)
    # Preserve the original `run all` contract for the four physics experiments.
    selected = [e for e in REGISTRY if e not in TASKS] if args.experiment == "all" else ([e for e in TASKS if e not in VISUAL_TASKS] if args.experiment=="tasks" else (list(VISUAL_TASKS) if args.experiment=="visual" else [args.experiment]))
    manifest = dict(status="running", created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    environment=environment(), experiments={e:dict(status="pending") for e in selected})
    dump(root / "manifest.json", manifest)
    for exp in selected:
        out = root / exp
        manifest["experiments"][exp] = dict(status="running", output=str(out))
        dump(root / "manifest.json", manifest)
        cmd = ["_experiment", exp, "--output", out, "--group-budget", args.group_budget,
               "--render-budget", args.render_budget, "--experiment-budget", args.experiment_budget,"--suite",args.suite,"--difficulty",args.difficulty]
        if args.resume: cmd.append("--resume")
        if args.refresh_analysis: cmd.append("--refresh-analysis")
        if args.no_video or args.quick: cmd.append("--no-video")
        if args.no_trajectory: cmd.append("--no-trajectory")
        if args.no_package:cmd.append("--no-package")
        if args.images:cmd.append("--images")
        if args.policy:cmd.extend(["--policy",args.policy])
        p = child(cmd, out / "experiment.log", args.experiment_budget)
        if (out / "manifest.json").exists():
            detail = json.loads((out / "manifest.json").read_text())
            state = detail["status"] if p["status"] == "passed" else "failed"
            if p["status"] != "passed":
                detail.update(status="failed", termination=p)
                for group in detail.get("groups",[]):
                    if group["status"]=="running": group.update(status="failed",reason="Experiment process terminated",termination=p)
                    elif group["status"]=="pending": group.update(status="skipped",reason="Experiment terminated before this group could execute")
                dump(out / "manifest.json", detail)
        else:
            state = "failed"
        manifest["experiments"][exp] = dict(status=state, process=p, output=str(out), report=str(out / "REPORT.md"))
        dump(root / "manifest.json", manifest)
        print(f"{exp}: {state} -> {out}", flush=True)
    states = [s["status"] for s in manifest["experiments"].values()]
    manifest["status"] = "passed" if all(s=="passed" for s in states) else ("failed" if all(s=="failed" for s in states) else "partial")
    dump(root / "manifest.json", manifest)
    lines = ["# Physical experiments — exported results", "", f"Batch status: {manifest['status']}", "", "| Experiment | Status | Report | Replay |", "|---|---|---|---|"]
    for exp, item in manifest["experiments"].items():
        lines.append(f"| {exp} | {item['status']} | [{exp}/REPORT.md]({exp}/REPORT.md) | [{exp}/replay.zip]({exp}/replay.zip) |")
    rows=[]
    for exp in selected:
        detail_path=root / exp / "manifest.json"
        if not detail_path.exists(): continue
        detail=json.loads(detail_path.read_text())
        lines += ["", f"## {exp}: {REGISTRY[exp]['title']}", "", "| Case / parameter | Observed result | Main metric |", "|---|---|---|"]
        from .artifacts import label
        for group in detail.get("groups",[]):
            if group["status"]!="passed":
                lines.append(f"| {group['name']} | {group['status']} | see manifest |")
                continue
            config=json.loads((root/exp/"runs"/group["name"]/"config.json").read_text())
            met=group["metrics"]; key=REGISTRY[exp]["metric"]
            lines.append(f"| {group['name']}: {label(config)} | {met['outcome']} | {key} = {met.get(key)} |")
            rows.append([exp,group["name"],label(config),met["outcome"],key,met.get(key),met["max_penetration_m"],met.get("actual_release_time_s"),met["physics_seconds"]])
        lines += ["", f"[曲线]({exp}/curves.png) · [逐组报告]({exp}/REPORT.md) · [三维回放 ZIP]({exp}/replay.zip)", ""]
        for view, item in detail.get("videos",{}).items():
            if isinstance(item,dict):
                encoded=item.get("details",{})
                lines.append(f"- [{view} 视频]({exp}/videos/{view}.mp4): {item.get('status')}, {encoded.get('resolution')}, {encoded.get('fps')} fps; degraded={item.get('degraded')}")
        check=detail.get("timestep_comparison",{})
        lines += ["", f"步长对照 {check.get('dt_s')}: {check.get('metric')} = {check.get('values')}; 最大穿透 {check.get('max_penetration_m')} m。",
                  f"趋势检查：{detail.get('trend',{}).get('status')}；与运行是否成功分开判断。"]
        if exp=="04_button":
            lines += ["", f"按钮按压方式：{detail.get('button_control',{}).get('mode')}；{detail.get('button_control',{}).get('reason')}。实际释放时刻来自历史接触力。"]
        if exp in VISUAL_TASKS:
            cs=detail.get("task_evaluation",{}).get("contrasts",{})
            lines += ["",f"正向场景成功 {cs.get('positive_successes')}/{cs.get('positive_attempts')}；真实负对照符合预期 {cs.get('negative_contrasts_observed')}/{cs.get('negative_attempts')}（负对照不算任务成功）。"]
    with (root/"SUMMARY.csv").open("w",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["experiment","case","parameter","outcome","metric","value","max_penetration_m","actual_release_time_s","physics_seconds"])
        writer.writerows(rows)
    guide=ROOT/("docs/visual_tasks.md" if all(e in VISUAL_TASKS for e in selected) else ("docs/robot_tasks.md" if any(e in TASKS for e in selected) else "docs/physics_experiments.md"))
    shutil.copy2(guide,root/"START_HERE.md")
    lines += ["", "## 本地打开", "", "Windows: extract an experiment's replay.zip, install replay_requirements.txt in Python 3.11, run python replay.py.",
              "", "本地 GUI 待验证。服务器回放数据/加载是否已验证，见每项 manifest 中的 replay 状态；不声称已打开 Windows GUI。",
              "", "[完整操作和官方参考](START_HERE.md)。每组历史记录在 runs/*/data.csv 和 contact_history.npz。",
              "数值物理、图形渲染、编码提交和结束等待分别计时，见 manifest 中 physics_seconds 和 videos/*_render.json。",
              "", "正常运行的降级/错误见各项 manifest；编码器故障注入测试单独在 verification/fallback_encoding.json，不属于正常实验失败。"]
    verification=root/"verification/verification.json"
    if verification.exists():
        check=json.loads(verification.read_text())
        lines += ["", f"测试：{check.get('tests')} passed；[测试日志](verification/pytest.log)。",
                  "[最终 ZIP/视频完整解码验收](verification/final_audit.json) · [编码故障注入](verification/fallback_encoding.json)。"]
        if (root/"verification/bounce_initial_soft_contact").exists():
            lines.append("初版软接触诊断保留于 verification/bounce_initial_soft_contact/，属于开发诊断，不是最终对照结果。")
        if (root/"verification/grasp_slip_initial").exists():
            lines.append("[初版真实抓取滑脱回放](verification/grasp_slip_initial/replay.zip)及视频保留，属于开发诊断，不计入最终评测。")
    if all(e in VISUAL_TASKS for e in selected):
        for exp in selected:
            lines=[line.replace(f"[{exp}/replay.zip]({exp}/replay.zip)","未生成（本次仅视频）") for line in lines]
        lines=[line.split(" · [三维回放 ZIP]")[0] for line in lines]
        lines=[line for line in lines if not any(term in line for term in ("## 本地打开","Windows: extract","本地 GUI","最终 ZIP/视频","初版真实抓取滑脱回放","编码器故障注入测试"))]
        lines += ["", "## 观看视频", "", "下载各任务 videos/overview.mp4 和 detail.mp4，在本地播放器观看。本次未生成回放目录或压缩包。",
                  "模型输入只含原始相机 RGB 与任务指令；本次策略均为显式 oracle 脚本，未训练视觉模型。",
                  "正负对照按实际动力学结果核验；预期倾倒并不计作正策略成功。少量固定变体不能代表广泛泛化能力。"]
        lines += ["", "夹指软接触最大穿透约5–7 mm，已在各组指标与步长对照如实保留。", "", "[视频全帧解码与记录状态重新加载验收](verification/final_audit.json) · [测试记录](verification/pytest.log)。"]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    import signal
    signal.signal(signal.SIGTERM, terminate_children)
    signal.signal(signal.SIGINT, terminate_children)
    p = argparse.ArgumentParser(description="Reproducible physical experiments and portable replay")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    for name in ("run", "_experiment"):
        q = sub.add_parser(name)
        q.add_argument("experiment", choices=["all","tasks","visual", *REGISTRY] if name=="run" else list(REGISTRY),help="all: original physics; tasks: previous three robot tasks; visual: four visual reasoning demos")
        q.add_argument("--output", type=Path, default=ROOT / "deliverables" / "physics")
        q.add_argument("--no-video", action="store_true"); q.add_argument("--no-trajectory", action="store_true")
        q.add_argument("--no-package",action="store_true",help="Record states and render videos without building replay packages or ZIP files")
        q.add_argument("--images",action="store_true",help="Render raw camera observations for policies; oracle evaluation otherwise omits unused camera rendering")
        q.add_argument("--resume", action="store_true"); q.add_argument("--quick", action="store_true")
        q.add_argument("--suite",choices=["all","fixed","evaluation"],default="all")
        q.add_argument("--difficulty",choices=["all","easy","medium","physics_variation"],default="all")
        q.add_argument("--policy",help="Robot task policy as module:Class; defaults to the oracle expert")
        q.add_argument("--refresh-analysis", action="store_true", help="With --resume recompute metrics/packages/videos even for passed experiments")
        q.add_argument("--group-budget", type=float, default=120)
        q.add_argument("--render-budget", type=float, default=900)
        q.add_argument("--experiment-budget", type=float, default=3600)
    q = sub.add_parser("_physics"); q.add_argument("--config", type=Path, required=True); q.add_argument("--output", type=Path, required=True); q.add_argument("--no-trajectory", action="store_true");q.add_argument("--images",action="store_true")
    q = sub.add_parser("_observations");q.add_argument("--package",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
    for name in ("render", "_render"):
        q = sub.add_parser(name); q.add_argument("--package", type=Path, required=True); q.add_argument("--output", type=Path, required=True)
        q.add_argument("--view", choices=["overview", "detail", "slowmotion"], default="detail")
        q.add_argument("--width", type=int, default=1920); q.add_argument("--height", type=int, default=1080); q.add_argument("--fps", type=int, default=30); q.add_argument("--png", action="store_true")
    q = sub.add_parser("pack"); q.add_argument("directory", type=Path)
    q = sub.add_parser("validate"); q.add_argument("directory", type=Path)
    q = sub.add_parser("bundle"); q.add_argument("directory", type=Path, default=ROOT / "deliverables/physics", nargs="?")
    q = sub.add_parser("audit"); q.add_argument("directory", type=Path, default=ROOT / "deliverables/physics", nargs="?")
    q = sub.add_parser("finalize-tasks"); q.add_argument("directory",type=Path,default=ROOT/"deliverables/robot_tasks",nargs="?")
    args = p.parse_args()
    if args.action=="run" and (args.experiment in VISUAL_TASKS or args.experiment=="visual") and args.output==ROOT/"deliverables/physics":
        args.output=ROOT/"deliverables/vision_tasks"
    if args.action=="run" and (args.experiment in TASKS or args.experiment=="tasks") and args.output==ROOT/"deliverables/physics":
        args.output=ROOT/"deliverables/robot_tasks"
    for attr in ("output", "package", "directory"):
        candidate = getattr(args, attr, None)
        if candidate is not None and not Path(candidate).resolve().is_relative_to(ROOT.resolve()):
            p.error(f"{attr} must stay inside the authorized project directory: {ROOT}")
    # Render defaults only apply to these child processes, not any global environment.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")
    if args.action == "list":
        for key, value in REGISTRY.items(): print(key, value["title"], "implemented; headless + portable replay")
    elif args.action == "run":
        if args.no_trajectory and not (args.no_video or args.quick): p.error("--no-trajectory requires --no-video or --quick")
        batch(args)
    elif args.action == "_experiment": experiment(args)
    elif args.action == "_physics":
        from .physics import run
        result = run(json.loads(args.config.read_text()), args.output, not args.no_trajectory)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    elif args.action in ("render", "_render"):
        # Late import is essential: PyOpenGL reads its backend at import time.
        from .video import render
        render(args.package, args.output, args.view, args.width, args.height, args.fps, args.png)
    elif args.action=="_observations":
        from .video import raw_observations
        print(raw_observations(args.package,args.output))
    elif args.action == "pack":
        from .artifacts import zip_package
        zip_package(args.directory / "replay_package", args.directory / "replay.zip")
    elif args.action == "validate":
        from .artifacts import validate_zip
        print(validate_zip(args.directory / "replay.zip", args.directory / "zip_validation", child_env())[0])
    elif args.action == "bundle":
        destination=args.directory / ("robot_tasks_results.zip" if args.directory.name=="robot_tasks" else "physics_results.zip")
        with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for file in sorted(args.directory.rglob("*")):
                if not file.is_file() or file==destination: continue
                relative=file.relative_to(args.directory)
                if str(relative)=="START_HERE.md":continue
                if any(part in ("replay_package","zip_validation","render_source","__pycache__") for part in relative.parts): continue
                if file.suffix in (".xml", ".npz") and file.name not in ("contact_history.npz",): continue
                z.write(file,relative)
            z.write(ROOT / ("docs/robot_tasks.md" if args.directory.name=="robot_tasks" else "docs/physics_experiments.md"),"START_HERE.md")
        print(destination)
    elif args.action == "audit":
        from .audit import audit
        print(json.dumps(audit(args.directory,child_env()),ensure_ascii=False))
    elif args.action == "finalize-tasks":
        from dualstack.tasks.finalize import finalize
        finalize(args.directory,child_env())
