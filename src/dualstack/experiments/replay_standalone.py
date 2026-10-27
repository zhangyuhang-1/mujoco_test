"""Portable recorded-state viewer. Intentionally depends only on numpy + mujoco."""
import argparse
import json
import os
from pathlib import Path
import queue
import time
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent


def load(root, group):
    m = mujoco.MjModel.from_xml_path(str(root / group["scene"]))
    d = mujoco.MjData(m)
    with np.load(root / group["trajectory"], allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    sig = int(data["state_signature"])
    if data["states"].shape[1] != mujoco.mj_stateSize(m, sig):
        raise ValueError("State length mismatch. Install the exact MuJoCo version in replay_requirements.txt")
    if np.any(np.diff(data["times"]) <= 0):
        raise ValueError("Trajectory timestamps must increase strictly")
    return m, d, data, sig


def restore(m, d, data, sig, index):
    mujoco.mj_setState(m, d, data["states"][index], sig)
    mujoco.mj_forward(m, d)  # Derived transforms only; NO mj_step during replay.


def validate(root, offscreen=False):
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    results = []
    for group in meta["groups"]:
        m, d, data, sig = load(root, group)
        errors = []
        # All poses, not only final frame. Additional visual checks at key times.
        for i in range(len(data["times"])):
            restore(m, d, data, sig, i)
            actual = np.r_[d.body("sample").xpos, d.body("sample").xquat]
            errors.append(float(np.max(abs(actual-data["pose"][i]))))
            if "body_poses" in data:
                for bi,name in enumerate(data["body_names"]):
                    other=np.r_[d.body(str(name)).xpos,d.body(str(name)).xquat]
                    errors[-1]=max(errors[-1],float(np.max(abs(other-data["body_poses"][i,bi]))))
        maximum = max(errors)
        if maximum > 1e-9:
            raise AssertionError(f"Pose roundtrip failed: {group['name']} error={maximum}")
        result = dict(name=group["name"], frames=len(errors), max_pose_error=maximum,
                      state_size=data["states"].shape[1], physics_steps_during_replay=0)
        if offscreen:
            with mujoco.Renderer(m, height=180, width=320) as renderer:
                checks = []
                indices = {0, len(errors)//2, len(errors)-1}
                for key in ("first_impact_time_s", "first_rebound_peak_time_s", "stop_time_s"):
                    if group.get("metrics", {}).get(key) is not None:
                        indices.add(int(np.argmin(abs(data["times"]-group["metrics"][key]))))
                for i in sorted(indices):
                    restore(m, d, data, sig, i)
                    renderer.update_scene(d, camera="film_detail")
                    image = renderer.render().copy()
                    if image.std() < 2:
                        raise AssertionError("Blank replay rendering")
                    checks.append(dict(frame=i, time=float(d.time), image_std=float(image.std())))
                result["offscreen_checks"] = checks
        results.append(result)
    report = dict(status="passed", version=mujoco.__version__, groups=results,
                  gui="回放数据和加载已验证，本地 GUI 待验证")
    (root / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def play(root, selected=0, sequential=False):
    import mujoco.viewer
    import glfw
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    groups = meta["groups"]
    commands = queue.SimpleQueue()
    index_group = selected % len(groups)
    speed, paused = 1., False
    while True:
        group = groups[index_group]
        m, d, data, sig = load(root, group)
        times = data["times"]
        cursor, index, camera = float(times[0]), 0, 0
        restore(m, d, data, sig, index)
        print(f"Case {index_group+1}/{len(groups)}: {group['name']} {group['config']}")
        change = None
        with mujoco.viewer.launch_passive(m, d, key_callback=commands.put) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = m.camera("film_detail").id
            previous = time.perf_counter()
            while viewer.is_running():
                now = time.perf_counter()
                elapsed, previous = now-previous, now
                while not commands.empty():
                    key = commands.get()
                    if key == glfw.KEY_SPACE:
                        paused = not paused
                    elif key == glfw.KEY_R:
                        cursor, paused = float(times[0]), False
                    elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                        speed = min(8., speed*2)
                    elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                        speed = max(.125, speed/2)
                    elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
                        cursor += -1 if key == glfw.KEY_LEFT else 1
                    elif key in (glfw.KEY_COMMA, glfw.KEY_PERIOD):
                        paused = True
                        index = np.clip(index + (-1 if key == glfw.KEY_COMMA else 1), 0, len(times)-1)
                        cursor = float(times[index])
                    elif key == glfw.KEY_HOME:
                        cursor = float(times[0])
                    elif key == glfw.KEY_END:
                        cursor, paused = float(times[-1]), True
                    elif key == glfw.KEY_C:
                        camera = (camera+1) % 3
                        with viewer.lock():
                            if camera == 2:
                                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                                viewer.cam.lookat[:] = [.15, 0, .86]
                                viewer.cam.distance = .8
                                viewer.cam.azimuth = 110
                                viewer.cam.elevation = -22
                            else:
                                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                                viewer.cam.fixedcamid = m.camera("film_detail" if camera == 0 else "film_overview").id
                    elif key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET):
                        change = (index_group + (-1 if key == glfw.KEY_LEFT_BRACKET else 1)) % len(groups)
                    elif glfw.KEY_1 <= key <= glfw.KEY_9:
                        change = (key-glfw.KEY_1) % len(groups)
                if change is not None:
                    break
                if not paused:
                    cursor += elapsed*speed
                cursor = float(np.clip(cursor, times[0], times[-1]))
                index = max(0, int(np.searchsorted(times, cursor, side="right")-1))
                restore(m, d, data, sig, index)
                if hasattr(viewer, "set_texts"):
                    viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f"{meta['experiment']} | {group['name']} | {speed:g}x | {'PAUSED' if paused else 'PLAY'}",
                        f"t={times[index]:.3f}s / {times[-1]:.3f}s\n{group['label']}\nSpace pause | R replay | +/- speed | , . step | arrows seek\n[ ] cases | C camera (third mode: free mouse orbit)\nRecorded trajectory; historical forces in data files")])
                viewer.sync()
                if cursor >= times[-1] and sequential and index_group+1 < len(groups):
                    time.sleep(.5)
                    change = index_group+1
                    break
                time.sleep(.005)
        if change is None:
            return
        index_group = change
        while not commands.empty():
            commands.get()


def main():
    p = argparse.ArgumentParser(description="Recorded 3D trajectory replay; does not rerun a controller")
    p.add_argument("--root", type=Path, default=HERE)
    p.add_argument("--case", type=int, default=1, help="1-based case")
    p.add_argument("--all", action="store_true", help="Play cases sequentially")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--offscreen", action="store_true")
    args = p.parse_args()
    if args.validate:
        validate(args.root, args.offscreen)
    else:
        if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            p.error("No desktop display. Copy this package to your Windows computer and run python replay.py there.")
        play(args.root, args.case-1, args.all)


if __name__ == "__main__":
    main()
