import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from .registry import ROOT, REGISTRY, dump


def label(config):
    if config.get("vision_only"):
        return f"{config['name']} | oracle expert | " + (f"COM side={config['weight_side']:+d}" if config['experiment'] in ('08_balance','10_hidden') else config.get('strategy','physical tilt'))
    if "task_title" in config:
        exp=config["experiment"]
        if exp=="05_push":value=f"mu={config['mu']:g}"
        elif exp=="06_eccentric":value=f"weight={config['weight_mass']:g}kg side={config['weight_side']:+d}"
        else:value=f"{config['difficulty']} rolling={config['rolling_friction']:g} seed={config['seed']}"
        return f"{value} | {config['split']} {config['seed']}"
    if config["experiment"] == "04_button":
        return f"{config.get('family', 'probe')}: k={config['stiffness']:g} N/m, c={config['damping']:g} N.s/m"
    key = REGISTRY[config["experiment"]]["parameter"]
    return f"{key}={config[key]:g}"


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def plot(experiment, groups, output):
    """Portable raster plots using already-installed Pillow; no training dependencies."""
    specs = {
        "01_friction": [("time_s", "vx_m_s", "Horizontal velocity (m/s)"), ("time_s", "distance", "Sliding distance (m)"), ("time_s", "kinetic_energy_J", "Kinetic energy (J)"), ("time_s", "normal_force_N", "Normal force (N)")],
        "02_bounce": [("time_s", "z_m", "Ball center height (m)"), ("time_s", "vz_m_s", "Vertical velocity (m/s)"), ("time_s", "normal_force_N", "Contact normal force (N)"), ("time_s", "max_penetration_m", "Contact penetration (m)")],
        "03_stack": [("time_s", "x_m", "COM x (m)"), ("time_s", "z_m", "COM height (m)"), ("time_s", "tilt_rad", "Tilt (rad)"), ("time_s", "wy_rad_s", "Angular velocity y (rad/s)")],
        "04_button": [("time_s", "depression", "Downward displacement from gravity equilibrium (m)"), ("time_s", "joint_v_m_s", "Button velocity (m/s)"), ("time_s", "forces", "Spring / damper / gravity / applied force (N)"), ("joint_q_m", "spring_force_N", "Spring force vs joint position (N vs m)")],
        "05_push": [("time_s","goal_error_m","Target error (m)"),("time_s","vx_m_s","Block x velocity (m/s)"),("time_s","push_speed_command_m_s","Probe-adapted pushing speed (m/s)"),("time_s","left_ee_x_m","Actual left EE x (m)")],
        "06_eccentric": [("time_s","x_m","Geometric center x (m)"),("time_s","com_x_m","True center of mass x (m)"),("time_s","goal_error_m","COM projection error (m)"),("time_s","z_m","Object height (m)")],
        "07_tray": [("time_s","ball_rel_x_m","Ball relative tray x (m)"),("time_s","ball_speed_m_s","Ball world speed (m/s)"),("time_s","tilt_rad","Tray tilt (rad)"),("time_s","left_tracking_error_m","Left grip tracking error (m)")],
        "08_balance":[("time_s","com_margin_x_m","COM support margin (m)"),("time_s","tilt_rad","Bar tilt (rad)"),("time_s","z_m","Bar height (m)"),("time_s","robot_contact","Robot contact (boolean)")],
        "09_support":[("time_s","beam_tilt_rad","Beam tilt (rad)"),("time_s","beam_height_m","Beam height (m)"),("time_s","payload_height_m","Payload height (m)"),("time_s","backup_grasp","Physical backup grip")],
        "10_hidden":[("time_s","hinge_angle_rad","Natural observation tilt (rad)"),("time_s","fixture_active","Observation clamp active"),("time_s","com_margin_x_m","COM support margin (m)"),("time_s","z_m","Object height (m)")],
        "11_pour":[("time_s","tilt_rad","Box tilt (rad)"),("time_s","ball_rel_y_m","Ball local y toward exit (m)"),("time_s","receiver_inside","Ball inside receiver"),("time_s","receiver_contact_N","Receiver normal force (N)")],
    }
    image = Image.new("RGB", (1800, 1240), "#f5f6f8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    draw.text((45, 16), f"{experiment}: {REGISTRY[experiment]['title']} | real MuJoCo trajectories", fill="#17202a", font=title)
    colors = ["#1261a0", "#d55e00", "#009e73", "#8a49a8", "#cc8c00", "#2f3437"]
    arrays = []
    for group in groups:
        a = read_csv(group["path"] / "data.csv")
        arrays.append((group, a))
    for panel, (xkey, ykey, caption) in enumerate(specs[experiment]):
        x0, y0 = 95+(panel % 2)*890, 130+(panel//2)*480
        w, h = 735, 340
        series = []
        for gi, (group, a) in enumerate(arrays):
            if ykey == "distance":
                y = np.r_[0, np.cumsum(np.hypot(np.diff(a["x_m"]), np.diff(a["y_m"])))]
            elif ykey == "depression":
                y = group["metrics"]["equilibrium_q_m"]-a["joint_q_m"]
            elif ykey == "forces":
                # One representative configuration; all groups remain available in CSV.
                if gi:
                    continue
                for fi, name in enumerate(("spring_force_N", "damper_force_N", "gravity_force_N", "applied_force_N")):
                    series.append((a[xkey], a[name], colors[fi], name))
                continue
            else:
                y = a[ykey]
            series.append((a[xkey], y, colors[gi % len(colors)], group["name"]))
        xmin = min(float(s[0].min()) for s in series); xmax = max(float(s[0].max()) for s in series)
        ymin = min(float(s[1].min()) for s in series); ymax = max(float(s[1].max()) for s in series)
        if xmax-xmin < 1e-10: xmax = xmin+1
        pad = max((ymax-ymin)*.06, 1e-6); ymin -= pad; ymax += pad
        draw.rectangle((x0, y0, x0+w, y0+h), fill="white", outline="#7c8791", width=2)
        draw.text((x0, y0-34), caption, fill="#17202a", font=font)
        for tick in range(6):
            f = tick/5
            yy = y0+h-h*f; xx = x0+w*f
            draw.line((x0, yy, x0+w, yy), fill="#e0e3e6")
            draw.text((x0-84, yy-10), f"{ymin+f*(ymax-ymin):.3g}", fill="#17202a", font=font)
            draw.text((xx-16, y0+h+10), f"{xmin+f*(xmax-xmin):.3g}", fill="#17202a", font=font)
        draw.text((x0+w//2-40, y0+h+36), "time (s)" if xkey == "time_s" else "joint position (m)", fill="#17202a", font=font)
        for xx, yy, color, name in series:
            # Keep every extremum; plotting every saved sample is inexpensive here.
            points = list(zip((x0+(xx-xmin)/(xmax-xmin)*w).tolist(), (y0+h-(yy-ymin)/(ymax-ymin)*h).tolist()))
            draw.line(points, fill=color, width=3)
        if ykey == "forces":
            draw.text((x0+5, y0+5), " | ".join(s[3].replace("_force_N", "") for s in series), fill="#17202a", font=font)
    for i, (group, _) in enumerate(arrays):
        draw.text((65+(i%2)*890, 1095+(i//2)*35), label(group["config"])+" | "+group["metrics"]["outcome"], fill=colors[i % len(colors)], font=font)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


REPLAY_README = """# Recorded MuJoCo 3D replay / 三维轨迹回放

在 Windows 解压整个 ZIP，保持目录结构。安装 Python 3.11，然后在解压目录：

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r replay_requirements.txt
.\\.venv\\Scripts\\python.exe replay.py
# 指定参数组或依次播放
.\\.venv\\Scripts\\python.exe replay.py --case 2
.\\.venv\\Scripts\\python.exe replay.py --all
```

不要把 NPZ 轨迹拖进 simulate.exe；入口是 python replay.py。
这是记录状态的三维回放，不是可以任意修改后继续演化的在线仿真。
播放器不调用 mj_step，不运行原控制策略。历史接触力见 data/ 内 CSV 和 contact_history.npz，
不能由恢复位置后重新计算的接触力冒充历史力。

## 操作

- Space 暂停/继续；R 重播；+/- 倍速（0.125–8）；逗号/句号逐记录帧。
- 左/右箭头跳转 -/+1 秒；Home 起点；End 终点。
- 数字 1–9 选择组；[ / ] 上一组/下一组（会重开 viewer）。
- C 在特写、总览、自由视角之间循环。自由视角时鼠标左键旋转、右键平移、滚轮缩放；
  MuJoCo 的 Shift 修饰键可改变旋转/平移方向。
- 默认按原仿真时间播放，末帧保持，直到关闭或重播；--all 依次播放后保持最后一组末帧。
- 屏幕显示组参数、仿真时间、倍速和播放状态。

metadata.json 记录版本、状态字段、采样步长、各组模型和结果。所有物理步都记录，
回放只能恢复已采样状态，不重建相邻样本之间的碰撞细节；视频可能以更低帧率采样。
弹簧导向杆属于无碰撞装饰；实验中没有动态改变模型参数。无需额外插件。

服务器验证入口：python replay.py --validate --offscreen（先配置离屏渲染后端）。
本地 GUI 待验证。服务器数据和加载验证状态以 validation.json 为准（若导出验证成功）。
"""


def package(experiment, groups, output):
    output = Path(output)
    root = output / "replay_package"
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("assets", "scenes", "trajectories", "data"):
        (root / directory).mkdir(exist_ok=True)
    meta = dict(experiment=experiment, mujoco_version=mujoco.__version__, python="3.11",
                state_signature="mjSTATE_INTEGRATION", state_signature_value=16383,
                state_fields=[k for k,v in mujoco.mjtState.__members__.items() if k.startswith("mjSTATE_") and int(v)>0 and int(v)&(int(v)-1)==0],
                model_mutations_during_run=False, plugins=[],
                historical_forces="data/*_contacts.npz and *_data.csv; NOT reconstructed by replay",
                gui="本地 GUI 待验证；服务器验证结果见 validation.json", validation_status="pending", groups=[])
    for i, group in enumerate(groups):
        name = group["name"]
        tree = ET.parse(group["path"] / "scene.xml")
        for el in tree.iter():
            if "file" not in el.attrib:
                continue
            src = Path(el.get("file"))
            if not src.is_absolute():
                src = (group["path"] / src).resolve()
            dst = root / "assets" / src.name
            if not dst.exists(): shutil.copy2(src, dst)
            el.set("file", "../assets/"+src.name)
        tree.write(root / "scenes" / f"{name}.xml", encoding="unicode")
        if i == 0:
            for el in tree.iter():
                if "file" in el.attrib: el.set("file", el.get("file").replace("../assets/", "assets/"))
            tree.write(root / "scene.xml", encoding="unicode")
        shutil.copy2(group["path"] / "trajectory.npz", root / "trajectories" / f"{name}.npz")
        for source, suffix in (("data.csv", "data.csv"), ("metrics.json", "metrics.json"), ("config.json", "config.json"), ("contact_history.npz", "contacts.npz")):
            shutil.copy2(group["path"] / source, root / "data" / f"{name}_{suffix}")
        if (group["path"] / "events.json").exists():
            shutil.copy2(group["path"] / "events.json",root / "data" / f"{name}_events.json")
        meta["groups"].append(dict(name=name, label=label(group["config"]), config=group["config"], metrics=group["metrics"],
                                  scene=f"scenes/{name}.xml", trajectory=f"trajectories/{name}.npz"))
    for name in ("LICENSE", "SOURCE.json"):
        shutil.copy2(ROOT / "assets/franka_emika_panda" / name, root / "assets" / ("PANDA_"+name))
    shutil.copy2(Path(__file__).with_name("replay_standalone.py"), root / "replay.py")
    (root / "replay_requirements.txt").write_text(f"mujoco=={mujoco.__version__}\nnumpy==2.4.6\n")
    (root / "README.md").write_text(REPLAY_README, encoding="utf-8")
    dump(root / "metadata.json", meta)
    return root


def zip_package(root, destination):
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                z.write(file, file.relative_to(root))


def validate_zip(zip_path, destination, env, timeout=120):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(destination)
    started = time.perf_counter()
    # New interpreter with cwd=unpacked package; no project imports or asset cache.
    command=[sys.executable,str(destination/"replay.py"),"--root",str(destination),"--validate"]
    numerical_env=dict(env,MUJOCO_GL="disable")
    with (destination / "validation_numeric.log").open("w") as log:
        result = subprocess.run(command,cwd=destination,env=numerical_env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Exported ZIP numerical validation failed: {destination/'validation_numeric.log'}")
    numerical=json.loads((destination/"validation.json").read_text())
    try:
        with (destination/"validation.log").open("w") as log:
            visual=subprocess.run([*command,"--offscreen"],cwd=destination,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if visual.returncode: raise RuntimeError(f"Offscreen validation exit={visual.returncode}; see validation.log")
        report=json.loads((destination/"validation.json").read_text())
        report.update(numerical_status="passed",offscreen_status="passed")
    except Exception as e:
        report=dict(numerical,numerical_status="passed",offscreen_status="failed",offscreen_error=repr(e))
    dump(destination/"validation.json",report)
    return report,time.perf_counter()-started
