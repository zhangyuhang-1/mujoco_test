"""Offline video renderer. This module never advances physics."""
import json
from pathlib import Path
import time
import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .registry import dump
from .replay_standalone import load, restore


def raw_observations(root,destination):
    """Native camera RGB + instruction only; human video captions are excluded."""
    from dualstack.tasks.visual_config import INSTRUCTIONS
    root,destination=Path(root),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/"metadata.json").read_text());examples=[]
    for ei,g in enumerate(meta["groups"]):
        m,d,data,sig=load(root,g)
        indices=sorted({0,int(np.argmin(abs(data["times"]-1.))),len(data["times"])-1})
        with mujoco.Renderer(m,height=256,width=256) as renderer:
            for j,i in enumerate(indices):
                restore(m,d,data,sig,i);renderer.update_scene(d,camera="vision_overview")
                image=renderer.render().copy();name=f"episode_{ei:03}_frame_{j:02}.png";imageio.imwrite(destination/name,image)
                examples.append(dict(images={"vision_overview":name},instruction=g["config"].get("instruction",INSTRUCTIONS.get(meta["experiment"],g["config"]["task_title"]))))
    dump(destination/"model_inputs.json",examples)
    dump(destination/"interface.json",dict(keys=["images","instruction"],resolution=[256,256],rgb=True,overlays=False,
        excludes=["joint_state","time","object_pose","mass","com","contact_force","wrist_torque"],note="Human annotated videos are not model input."))
    return len(examples)


def render(root, destination, view, width=1920, height=1080, fps=30, png=False):
    root, destination = Path(root), Path(destination)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded_path = destination.with_name(destination.stem+".inprogress"+destination.suffix)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    size = max(10, round(width/70))
    font = ImageFont.truetype(font_path, size)
    small = ImageFont.truetype(font_path, max(9, round(size*.8)))
    writer = None
    frames = 0
    timings = dict(render_seconds=0., overlay_seconds=0., encoding_submission_seconds=0., encoder_finalize_seconds=0.)
    started = time.perf_counter()
    stage = "context"
    try:
        for gi, group in enumerate(meta["groups"]):
            m, d, data, sig = load(root, group)
            # Correct buffer size at construction; never upscale low-resolution renders.
            m.vis.global_.offwidth = width; m.vis.global_.offheight = height
            renderer = None
            try:
                renderer = mujoco.Renderer(m, height=height, width=width)
                times = data["times"]
                slow = view == "slowmotion"
                first = group["metrics"].get("first_impact_time_s")
                if slow:
                    if first is None: continue
                    begin, end = max(0, first-.09), min(float(times[-1]), first+.20)
                    speed = .25
                else:
                    begin, end, speed = float(times[0]), float(times[-1]), 1.
                sample_times = np.arange(begin, end+1e-8, speed/fps)
                sequence = np.r_[np.repeat(begin, max(1, round(.3*fps))), sample_times, np.repeat(end, max(1, round(.5*fps)))]
                camera = "film_overview" if view == "overview" else "film_detail"
                if png:
                    destination.mkdir(parents=True, exist_ok=True)
                elif writer is None:
                    stage = "encoding"
                    writer = imageio.get_writer(str(encoded_path), fps=fps, codec="libx264", quality=8,
                                               macro_block_size=1, pixelformat="yuv420p",
                                               ffmpeg_params=["-preset", "fast", "-threads", "2", "-movflags", "+faststart"])
                for requested_time in sequence:
                    i = max(0, int(np.searchsorted(times, requested_time, side="right")-1))
                    stage = "rendering"
                    t0 = time.perf_counter()
                    restore(m, d, data, sig, i)
                    renderer.update_scene(d, camera=camera)
                    image = renderer.render().copy()
                    timings["render_seconds"] += time.perf_counter()-t0
                    t0 = time.perf_counter()
                    im = Image.fromarray(image); draw = ImageDraw.Draw(im)
                    draw.rectangle((0, 0, width, size*4), fill="#101d30")
                    draw.text((size//2, size//3), f"{meta['experiment']} | {group['name']} | {view} | {width}x{height} {fps}fps", font=font, fill="white")
                    draw.text((size//2, size*1.7), group["label"]+f" | t={times[i]:.3f} s | playback={speed:g}x", font=font, fill="#8be6ed")
                    draw.rectangle((0, height-size*3, width, height), fill="#101d30")
                    k = group["metrics"]
                    if "task_title" in group["config"]:
                        value=f"error={k['final_error_m']:.4f} m | stable={k['hold_time_s']:.2f} s | oracle expert"
                        if "phase_names" in data:value+=f" | {data['phase_names'][int(data['phase_ids'][i])]}"
                    elif meta["experiment"] == "01_friction":
                        stop = f"{k['stop_time_s']:.3f}" if k['stop_time_s'] is not None else "not reached"
                        value = f"distance={k['sliding_distance_m']:.4f} m | stop={stop} s"
                        # Velocity comes from the stored state through objectVelocity, without stepping.
                        vel = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body("sample").id, vel, 0)
                        value += f" | speed={np.linalg.norm(vel[3:]):.3f} m/s"
                    elif meta["experiment"] == "02_bounce":
                        value = f"rebound={k['first_rebound_height_m']:.4f} m | peak normal force={k['peak_contact_force_N']:.2f} N"
                    elif meta["experiment"] == "03_stack":
                        value = f"offset={group['config']['offset_m']:.3f} m | final tilt={k['final_tilt_rad']:.3f} rad"
                    else:
                        q = d.qpos[m.joint("sample_joint").qposadr[0]]
                        value = f"depression={(k['equilibrium_q_m']-q)*1000:.2f} mm | press={k['press_mode']} | release={k.get('actual_release_time_s',group['config'].get('release_time_s',1.)):.3f} s"
                    draw.text((size//2, height-size*2.75), f"Result: {k['outcome']} | {value}", font=small, fill="white")
                    draw.text((size//2, height-size*1.25), "Recorded physics | history CSV/NPZ contains forces | video rendering does not advance physics", font=small, fill="#a9b9cf")
                    frame = np.asarray(im)
                    timings["overlay_seconds"] += time.perf_counter()-t0
                    t0 = time.perf_counter(); stage = "encoding"
                    if png:
                        imageio.imwrite(destination / f"frame_{frames:06d}.png", frame)
                    else:
                        writer.append_data(frame)
                    timings["encoding_submission_seconds"] += time.perf_counter()-t0
                    if frames == 0:
                        im.save(destination.parent / f"{view}_preview.jpg", quality=92)
                    frames += 1
            finally:
                if renderer is not None: renderer.close()
        if writer is not None:
            t0 = time.perf_counter(); writer.close(); writer = None
            timings["encoder_finalize_seconds"] = time.perf_counter()-t0
        if not frames:
            raise RuntimeError("No frames generated")
        if not png:
            reader = imageio_ffmpeg.read_frames(str(encoded_path))
            encoded = next(reader); reader.close()
            if tuple(encoded["size"]) != (width, height) or abs(encoded["fps"]-fps) > .1:
                raise AssertionError(f"Encoded dimensions/fps mismatch: {encoded}")
            encoded_path.replace(destination)
        status = dict(status="passed", path=str(destination), resolution=[width, height], fps=fps, frames=frames,
                      playback_speed=.25 if view=="slowmotion" else 1., stage="complete", physics_steps=0,
                      wall_seconds=time.perf_counter()-started, **timings)
        dump(destination.parent / f"{view}_render.json", status)
        print(json.dumps(status), flush=True)
        return status
    except Exception as e:
        dump(destination.parent / f"{view}_render.json", dict(status="failed", stage=stage, error=repr(e), resolution=[width,height], fps=fps, frames=frames, **timings))
        raise
    finally:
        if writer is not None:
            writer.close()
