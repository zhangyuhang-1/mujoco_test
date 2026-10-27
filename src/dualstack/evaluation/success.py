import mujoco
import numpy as np

def measure(env):
    """Independent geometry/contact/dynamics check; no access to policy state."""
    m,d,c = env.model,env.data,env.config['success']
    a,b = m.body('block_a').id,m.body('block_b').id
    ga,gb = m.geom('a_geom').id,m.geom('b_geom').id
    pos,base = d.xpos[a],d.xpos[b]
    ar,br = d.xmat[a].reshape(3,3),d.xmat[b].reshape(3,3)
    local = br.T@(pos-base)
    half_a,half_b = m.geom_size[ga],m.geom_size[gb]
    # Entire A footprint must fit the support face (including yaw).
    footprint = np.abs((br.T@ar)[:2,:])@half_a
    inside = bool(np.all(np.abs(local[:2])+footprint < half_b[:2]-c['xy_margin']))
    height_error = abs(local[2]-half_a[2]-half_b[2])
    tilt = np.arccos(np.clip(ar[2,2],-1,1))
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,a,velocity,0)
    support_force = 0.
    finger_contact = False
    for i in range(d.ncon):
        contact = d.contact[i]
        pair = {contact.geom1,contact.geom2}
        if pair == {ga,gb}:
            force = np.zeros(6)
            mujoco.mj_contactForce(m,d,i,force)
            support_force += max(force[0],0)
        if ga in pair:
            other = contact.geom2 if contact.geom1==ga else contact.geom1
            body_name = mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[other])) or ''
            finger_contact |= 'finger' in body_name
    clear = True
    for side in ['left','right']:
        width = sum(d.qpos[m.joint(f'{side}_finger_joint{i}').qposadr[0]] for i in [1,2])
        clear &= bool(width>c['open_width'] and np.linalg.norm(env.pose(side)[:3]-pos)>c['gripper_clearance'])
    conditions = {'inside':inside, 'height':bool(height_error<c['height_tolerance']), 'upright':bool(tilt<c['tilt_radians']), 'slow':bool(np.linalg.norm(velocity[3:])<c['linear_speed'] and np.linalg.norm(velocity[:3])<c['angular_speed']), 'released':bool(clear and not finger_contact), 'supported':bool(support_force>m.body_mass[a]*9.81*.4)}
    return conditions, {'height_error':float(height_error), 'xy_error':float(np.linalg.norm(local[:2])), 'support_force':float(support_force), 'tilt':float(tilt), 'linear_speed':float(np.linalg.norm(velocity[3:])), 'angular_speed':float(np.linalg.norm(velocity[:3]))}

def update(env):
    conditions,diagnostics = measure(env)
    env.held = env.held+env.config['control_dt'] if all(conditions.values()) else 0.
    success = env.held >= env.config['success']['hold_seconds']
    reason = 'success' if success else 'running'
    a = env.data.body('block_a').xpos
    if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
        reason = 'numerical_error'
    elif a[2]<.72 or not (-.7<a[0]<.8 and -.8<a[1]<.8):
        reason = 'out_of_workspace'
    terminated = reason!='running'
    truncated = not terminated and env.data.time+1e-9 >= env.config['max_time']
    if truncated: reason='timeout'
    info=env.info()
    info.update(diagnostics,success=bool(success),reason=reason,hold_time=env.held,conditions=conditions)
    return float(success),terminated,truncated,info
