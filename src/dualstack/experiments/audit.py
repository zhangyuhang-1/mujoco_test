"""Validate deliverables, including full video decode and ZIP relocation."""
import json
from pathlib import Path
import subprocess
import time
import imageio_ffmpeg
import numpy as np
from .registry import ROOT, REGISTRY, dump
from .artifacts import validate_zip

REFERENCES = [
    "https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters",
    "https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint",
    "https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-getstate",
    "https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce",
    "https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer",
    "https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/viewer.py",
]


def audit(root, env):
    root = Path(root).resolve()
    started=time.perf_counter()
    report=dict(status="running", experiments={}, errors=[], gui="回放数据和加载已验证，本地 GUI 待验证")
    manifest_path=root/"manifest.json"
    selected=list(json.loads(manifest_path.read_text()).get("experiments",{})) if manifest_path.exists() else [name for name in REGISTRY if (root/name/"manifest.json").exists()]
    for exp in selected:
        path=root/exp
        try:
            manifest=json.loads((path/"manifest.json").read_text())
            item=dict(status=manifest["status"], task_status=manifest["status"],groups=len(manifest["groups"]), videos={}, historical_records=[])
            report["experiments"][exp]=item
            if manifest["status"] not in ("passed","partial"): raise AssertionError(f"Experiment state is {manifest['status']}")
            manifest["official_references"]=REFERENCES
            manifest["documentation_version_note"]="Versioned 3.17.0 pages inaccessible; installed 3.17.0 APIs and compiled scenes checked; stable official docs used with explicit caveat."
            dump(path/"manifest.json",manifest)
            for group in manifest["groups"]:
                if group["status"]!="passed": raise AssertionError(f"Group not passed: {group['name']}")
                a=np.genfromtxt(path/"runs"/group["name"]/"data.csv",delimiter=",",names=True)
                if len(a)!=group["metrics"]["samples"]: raise AssertionError("CSV sample length mismatch")
                if not np.all(np.isfinite(a.view(float).reshape(len(a),-1))): raise AssertionError("Nonfinite CSV data")
                if group["metrics"]["warnings"]: raise AssertionError(f"MuJoCo warnings {group['metrics']['warnings']}")
                item["historical_records"].append(dict(name=group["name"],samples=len(a),duration_s=float(a["time_s"][-1]),
                    sampling_interval_s=group["metrics"]["sample_interval_s"],max_penetration_m=group["metrics"]["max_penetration_m"]))
            if manifest.get("replay",{}).get("status")=="skipped" and (path/"recording_source/metadata.json").exists():
                started_load=time.perf_counter()
                validation=subprocess.run([env.get("PYTHON",__import__('sys').executable),str(ROOT/"src/dualstack/experiments/replay_standalone.py"),"--root",str(path/"recording_source"),"--validate","--offscreen"],env=env,capture_output=True,text=True,timeout=180)
                (path/"recording_validation.log").write_text(validation.stdout+validation.stderr)
                if validation.returncode:raise AssertionError(f"Raw recording validation failed: {validation.stderr[-1000:]}")
                item["recording_validation"]=json.loads((path/"recording_source/validation.json").read_text())
                item["recording_validation_seconds"]=time.perf_counter()-started_load
                item["replay_package"]="not generated, as requested"
            else:
                validation,seconds=validate_zip(path/"replay.zip",path/"zip_validation",env)
                item["zip_validation"]=validation; item["zip_validation_seconds"]=seconds
            for view,result in manifest["videos"].items():
                video=path/"videos"/(view+".mp4")
                if result["status"]!="passed": raise AssertionError(f"Video {view} not passed")
                reader=imageio_ffmpeg.read_frames(str(video)); header=next(reader); reader.close()
                expected=result["details"]
                if list(header["size"])!=expected["resolution"] or abs(header["fps"]-expected["fps"])>.1:
                    raise AssertionError("Video dimensions/fps mismatch")
                decoded=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-v","error","-nostats","-i",str(video),
                    "-map","0:v:0","-threads","2","-progress","pipe:1","-f","null","-"],capture_output=True,text=True,timeout=120)
                if decoded.returncode: raise AssertionError(f"Video full decode failed: {decoded.stderr}")
                frames=[int(line.split("=",1)[1]) for line in decoded.stdout.splitlines() if line.startswith("frame=")]
                if not frames or frames[-1]!=expected["frames"]: raise AssertionError("Decoded video frame count mismatch")
                item["videos"][view]=dict(status="passed",size=header["size"],fps=header["fps"],decoded_frames=frames[-1],
                    duration_s=header["duration"],bytes=video.stat().st_size)
            item["status"]="passed"
            print(f"Audited {exp}: recorded states + {len(item['videos'])} complete video decodes",flush=True)
        except Exception as e:
            report["errors"].append(dict(experiment=exp,error=repr(e)))
            report["experiments"].setdefault(exp,{})["status"]="failed"
    report["status"]="passed" if not report["errors"] else "failed"
    report["seconds"]=time.perf_counter()-started
    report["official_references"]=REFERENCES
    dump(root/"verification/final_audit.json",report)
    if report["errors"]: raise RuntimeError(report["errors"])
    return report
