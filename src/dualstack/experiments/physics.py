import csv
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from .registry import dump,TASKS
from .scenes import scene, initialize

STATE = int(mujoco.mjtState.mjSTATE_INTEGRATION)
COLUMNS = ["time_s", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz", "vx_m_s", "vy_m_s", "vz_m_s", "wx_rad_s", "wy_rad_s", "wz_rad_s", "normal_force_N", "max_penetration_m", "contact_count", "support_contacts", "effective_mu", "joint_q_m", "joint_v_m_s", "applied_force_N", "spring_force_N", "damper_force_N", "gravity_force_N", "tilt_rad", "kinetic_energy_J"]


def measure(m, d, config):
    bid = m.body("sample").id
    gid = m.geom("sample_geom").id
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
    force = penetration = count = support = 0
    mu = 0.0
    contacts = []
    for i, c in enumerate(d.contact):
        if gid not in (c.geom1, c.geom2):
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        count += 1
        force += max(0, f[0])
        penetration = max(penetration, -c.dist)
        mu = float(c.friction[0])
        other = c.geom2 if c.geom1 == gid else c.geom1
        support += int(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(other)) in ("table", "support_geom"))
        contacts.append([int(c.geom1), int(c.geom2), float(c.dist), *c.pos.tolist(), *f.tolist(), *c.friction.tolist()])
    joint = m.joint("sample_joint")
    button = config["experiment"] == "04_button"
    q = float(d.qpos[joint.qposadr[0]]) if button else 0.
    v = float(d.qvel[joint.dofadr[0]]) if button else 0.
    spring = -config["stiffness"] * q if button else 0.
    damper = -config["damping"] * v if button else 0.
    gravity = -float(m.body_mass[bid]) * 9.81 if button else 0.
    tilt = float(np.arccos(np.clip(d.xmat[bid, 8], -1, 1)))
    inertia = m.body_inertia[bid]
    # Translational energy is exact; rotational term is evaluated in body coordinates.
    body_omega = d.ximat[bid].reshape(3, 3).T @ vel[:3]
    energy = .5 * m.body_mass[bid] * np.dot(vel[3:], vel[3:]) + .5 * np.dot(inertia, body_omega**2)
    row = [float(d.time), *d.xpos[bid].tolist(), *d.xquat[bid].tolist(), *vel[3:].tolist(), *vel[:3].tolist(), force, penetration, count, support, mu, q, v, float(d.qfrc_applied[joint.dofadr[0]]) if button else 0., spring, damper, gravity, tilt, float(energy)]
    return row, contacts


def _first_hold(t, mask, duration, start=0):
    beginning = None
    for i in range(start, len(t)):
        if mask[i]:
            if beginning is None:
                beginning = i
            if t[i] - t[beginning] >= duration - 1e-8:
                return float(t[beginning])
        else:
            beginning = None
    return None


def metrics(rows, config):
    a = np.asarray(rows)
    get = lambda n: a[:, COLUMNS.index(n)]
    t, z = get("time_s"), get("z_m")
    result = dict(execution="completed", expected_trend="not_evaluated", max_penetration_m=float(get("max_penetration_m").max()),
                  initial_conditions={k: float(get(k)[0]) for k in COLUMNS if k not in ("effective_mu",)},
                  final_position_m=a[-1, 1:4].tolist(), final_tilt_rad=float(get("tilt_rad")[-1]), max_tilt_rad=float(get("tilt_rad").max()))
    exp = config["experiment"]
    if exp == "01_friction":
        speed = np.hypot(get("vx_m_s"), get("vy_m_s"))
        stop = _first_hold(t, (speed < config["stop_speed_m_s"]) & (get("support_contacts") > 0), config["stop_hold_s"])
        bounds = bool(np.any(np.abs(get("x_m") - .05) > .75) or np.any(z < .74))
        observed = get("effective_mu")[get("contact_count") > 0]
        distance = float(np.sum(np.linalg.norm(np.diff(a[:, 1:3], axis=0), axis=1)))
        ideal = float(speed[0]**2 / (2 * config["mu"] * 9.81))
        result.update(outcome="out_of_bounds" if bounds else ("stopped" if stop is not None else "not_stopped"),
                      sliding_distance_m=distance, stop_time_s=stop, ideal_distance_m=ideal,
                      ideal_relative_error=(distance-ideal)/ideal,
                      effective_mu_range=[float(observed.min()), float(observed.max())] if len(observed) else None,
                      initial_kinetic_energy_J=float(get("kinetic_energy_J")[0]), final_kinetic_energy_J=float(get("kinetic_energy_J")[-1]))
    elif exp == "02_bounce":
        contact = get("support_contacts") > 0
        hits = np.flatnonzero(contact)
        result.update(outcome="no_collision", first_impact_time_s=None, first_rebound_height_m=None,
                      pre_impact_vz_m_s=None, post_impact_vz_m_s=None, measured_velocity_ratio=None)
        if len(hits):
            hit = int(hits[0])
            releases = np.flatnonzero(~contact[hit+1:]) + hit + 1
            release = int(releases[0]) if len(releases) else None
            vz = get("vz_m_s")
            # First free-flight apex: positive to nonpositive vertical velocity after separation.
            apex = None
            if release is not None and vz[release] > 0.001:
                candidates = np.flatnonzero((vz[release:-1] > 0) & (vz[release+1:] <= 0)) + release
                apex = int(candidates[0]) if len(candidates) else None
            result.update(outcome="rebounded" if apex is not None else "no_resolved_rebound", first_impact_time_s=float(t[hit]),
                          first_rebound_peak_time_s=float(t[apex]) if apex is not None else None,
                          first_rebound_height_m=float(z[apex]-.785) if apex is not None else 0.,
                          pre_impact_vz_m_s=float(vz[max(0, hit-1)]), post_impact_vz_m_s=float(vz[release]) if release is not None else None,
                          measured_velocity_ratio=float(max(0, vz[release])/abs(vz[max(0, hit-1)])) if release is not None else None,
                          peak_contact_force_N=float(get("normal_force_N").max()))
    elif exp == "03_stack":
        speed = np.linalg.norm(a[:, 8:11], axis=1)
        omega = np.linalg.norm(a[:, 11:14], axis=1)
        on_base = (get("support_contacts") > 0) & (z > .83)
        quiet = (speed < .008) & (omega < .03) & (get("tilt_rad") < .06) & on_base
        stable_time = _first_hold(t, quiet, .8)
        # Classification uses measured motion, tilt and support, never the input offset alone.
        fell = bool(np.any(z < .815))
        tipped = bool(np.any(get("tilt_rad") > .35))
        result.update(outcome="tipped" if tipped and fell else ("slid_off" if fell else ("stable" if stable_time is not None and np.all(quiet[t > t[-1]-.5]) else "timeout_undetermined")),
                      stable_time_s=stable_time, support_fraction=float(on_base.mean()), initial_offset_m=float(a[0, 1]-.15))
    else:
        q, v = get("joint_q_m"), get("joint_v_m_s")
        planned_release = config.get("release_time_s", 1.0)
        # Arm withdrawal is not the same event as loss of contact. Use the
        # measured last nonzero contact force for spring response metrics.
        if config.get("press_mode") == "robot":
            loaded = np.flatnonzero(get("normal_force_N") > 1e-5)
            release_index = min(len(t)-1, int(loaded[-1])+1) if len(loaded) else 0
            release = float(t[release_index])
        else:
            release = planned_release
        eq = -.1*9.81/config["stiffness"]
        after = np.flatnonzero(t >= release)
        tolerance = .0005
        ok = (abs(q-eq) < tolerance) & (abs(v) < .003)
        settling = None
        # Require no later violation as well as >=.4 s remaining.
        for i in after:
            if t[-1]-t[i] >= .4 and np.all(ok[i:]):
                settling = float(t[i]-release)
                break
        peaks = [i for i in after[:-1] if i > 0 and (q[i]-q[i-1])*(q[i+1]-q[i]) <= 0 and abs(q[i]-eq) > .00005]
        result.update(outcome="settled" if settling is not None else "not_settled_in_budget", equilibrium_q_m=eq,
                      final_equilibrium_error_m=float(q[-1]-eq), max_depression_m=float(eq-q.min()),
                      settling_time_after_release_s=settling,
                      actual_release_time_s=release, planned_withdrawal_complete_s=planned_release,
                      damping_ratio=config["damping"]/(2*np.sqrt(.1*config["stiffness"])),
                      undamped_natural_frequency_hz=float(np.sqrt(config["stiffness"]/.1)/(2*np.pi)),
                      oscillation_peaks=[dict(time_s=float(t[i]), displacement_from_equilibrium_m=float(q[i]-eq)) for i in peaks],
                      applied_force_peak_N=float(abs(get("applied_force_N")).max()), peak_contact_force_N=float(get("normal_force_N").max()),
                      gravity_N=-.981, force_identity="F_spring=-k*q; F_damper=-c*qdot; F_gravity=-m*g (up is positive)",
                      press_mode=config.get("press_mode", "force"), simplification_reason=config.get("simplification_reason"))
    return result


def run(config, output, record=True):
    if config["experiment"] in TASKS:
        from dualstack.tasks.recording import run as run_task
        return run_task(config,output,record)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "config.json", config)
    root = scene(config)
    xml = ET.tostring(root, encoding="unicode")
    (output / "scene.xml").write_text(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    initialize(m, d, config)
    state = np.empty(mujoco.mj_stateSize(m, STATE))
    j = m.joint("sample_joint")
    controller = None
    robot_mode = config.get("press_mode") == "robot"
    if robot_mode:
        from dualstack.control.ik import DualIK
        controller = DualIK(m, .02)
        controller.reset(d)
        action = []
        for side in ("left", "right"):
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, d.site(side+"_ee").xmat)
            action.extend([*d.site(side+"_ee").xpos, *quat, 0.])
        action = np.array(action)
        initial = action[:3].copy()
    rows, states, contact_records = [], [], []
    started = time.perf_counter()
    n = round(config["duration"] / config["dt"])
    for step in range(n+1):
        t = step * config["dt"]
        if config["experiment"] == "04_button":
            if robot_mode:
                if step % max(1, round(.02/config["dt"])) == 0:
                    if t < 1.5:
                        a, b, u = initial, np.array([.15, 0, 1.02]), t/1.5
                    elif t < 2.7:
                        a, b, u = np.array([.15, 0, 1.02]), np.array([.15, 0, .872]), (t-1.5)/1.2
                    else:
                        a, b, u = np.array([.15, 0, .872]), np.array([.15, 0, 1.08]), min(1, (t-2.7)/.6)
                    u = np.clip(u, 0, 1); u = u*u*(3-2*u)
                    action[:8] = np.r_[a+(b-a)*u, [0, 1, 0, 0], 0.]
                    controller.apply(d, action)
            else:
                # Identical known force history across both families, no position forcing.
                ramp = min(t/.35, 1.) if t < .8 else max(0., (1.-t)/.2)
                d.qfrc_applied[j.dofadr[0]] = -3.0 * ramp
        mujoco.mj_forward(m, d)
        row, contacts = measure(m, d, config)
        rows.append(row)
        if record:
            mujoco.mj_getState(m, d, state, STATE)
            states.append(state.copy())
        contact_records.extend([[step, *c] for c in contacts])
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise FloatingPointError("Nonfinite physical state")
        if time.perf_counter()-started > config.get("group_budget_s", 120):
            raise TimeoutError("Physics group budget exhausted")
        if step < n:
            mujoco.mj_step(m, d)
    seconds = time.perf_counter()-started
    result = metrics(rows, config)
    result.update(physics_seconds=seconds, state_signature=STATE, state_size=len(state),
                  sample_interval_s=config["dt"], samples=len(rows), mujoco_version=mujoco.__version__,
                  model_mutations_during_run=False, warnings={str(i): int(w.number) for i, w in enumerate(d.warning) if w.number})
    with (output / "data.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(COLUMNS); writer.writerows(rows)
    # Historical contact frames and forces are separate from replay state; contact frame is local.
    np.savez_compressed(output / "contact_history.npz", contacts=np.asarray(contact_records, dtype=float).reshape(-1, 18),
                        columns=np.array(["sample", "geom1", "geom2", "distance", "px", "py", "pz", "normal", "tangent1", "tangent2", "torque_n", "torque_t1", "torque_t2", "mu1", "mu2", "torsion", "roll1", "roll2"]))
    if record:
        np.savez_compressed(output / "trajectory.npz", states=np.asarray(states), times=np.asarray(rows)[:, 0],
                            pose=np.asarray(rows)[:, 1:8], state_signature=STATE, body_name="sample",
                            qpos_size=m.nq, qvel_size=m.nv)
    dump(output / "metrics.json", result)
    return result
