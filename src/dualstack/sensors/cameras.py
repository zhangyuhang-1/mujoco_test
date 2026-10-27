import itertools
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np

def camera_info(env):
    """OpenCV optical frame: +x right, +y down, +z forward. Pixels u,v."""
    h,w=env.config['resolution']
    result={}
    for name in env.camera_names:
        i=env.model.camera(name).id
        f=.5*h/np.tan(np.deg2rad(env.model.cam_fovy[i])/2)
        k=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]])
        # MuJoCo camera local +y up, looks along -z.
        rotation=env.data.cam_xmat[i].reshape(3,3)@np.diag([1,-1,-1])
        world_from_cv=np.eye(4)
        world_from_cv[:3,:3]=rotation
        world_from_cv[:3,3]=env.data.cam_xpos[i]
        result[name]={'K':k.tolist(),'T_world_from_cv':world_from_cv.tolist(),'T_cv_from_world':np.linalg.inv(world_from_cv).tolist(),'fovy_degrees':float(env.model.cam_fovy[i]),'resolution_hw':[h,w],'color':'RGB','dtype':'uint8','origin':'top-left','sim_time':float(env.data.time)}
    return result

def project(points, info):
    points=np.asarray(points).reshape(-1,3)
    transform=np.array(info['T_cv_from_world'])
    cv=points@transform[:3,:3].T+transform[:3,3]
    homogeneous=cv@np.array(info['K']).T
    return homogeneous[:,:2]/homogeneous[:,2:3],cv[:,2]

def pixel_depth_to_world(uv, depth, info):
    """Depth is optical z in meters, not Euclidean range."""
    xyz=np.linalg.solve(np.array(info['K']),np.r_[uv,1.])*depth
    t=np.array(info['T_world_from_cv'])
    return t[:3,:3]@xyz+t[:3,3]

def block_bbox(env, name, info):
    gid=env.model.geom(name+'_geom').id
    corners=np.array(list(itertools.product([-1,1],repeat=3)))*env.model.geom_size[gid]
    points=corners@env.data.geom_xmat[gid].reshape(3,3).T+env.data.geom_xpos[gid]
    uv,z=project(points,info)
    if np.any(z<=0): return None
    return [*uv.min(axis=0),*uv.max(axis=0)]

def export_cameras(env, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    images=env.render()
    calibration=camera_info(env)
    validation={}
    for name,img in images.items():
        imageio.imwrite(output/(name+'.png'),img)
        env.renderer.update_scene(env.data,camera=name)
        env.renderer.enable_segmentation_rendering()
        seg=env.renderer.render().copy()
        env.renderer.disable_segmentation_rendering()
        checks={}
        for block in ['a','b']:
            gid=env.model.geom(block+'_geom').id
            mask=(seg[:,:,0]==gid)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
            bbox=block_bbox(env,block,calibration[name])
            if mask.any() and bbox is not None:
                v,u=np.where(mask)
                observed=[int(u.min()),int(v.min()),int(u.max()),int(v.max())]
                contained=bool(u.min()>=bbox[0]-2 and v.min()>=bbox[1]-2 and u.max()<=bbox[2]+2 and v.max()<=bbox[3]+2)
                checks[block]={'visible_pixels':int(mask.sum()),'projected_bbox':bbox,'segmentation_bbox':observed,'within_projected_corners_2px':contained}
            else: checks[block]={'visible_pixels':int(mask.sum()),'projected_bbox':bbox,'within_projected_corners_2px':None}
        validation[name]={'image_std':float(img.std()),'objects':checks}
    (output/'calibration.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
    (output/'projection_check.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    return validation
