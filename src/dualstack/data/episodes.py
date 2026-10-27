import importlib.metadata
import json
import platform
import sys
import hashlib
from pathlib import Path
import h5py
import imageio.v2 as imageio
import numpy as np
from dualstack.sensors.cameras import camera_info

def dependencies():
    names=['mujoco','mink','daqp','qpsolvers','numpy','scipy','h5py','imageio','imageio-ffmpeg','pillow']
    return {name:importlib.metadata.version(name) for name in names}

def robot_vector(obs):
    return np.concatenate([np.r_[obs['robot'][side]['qpos'],obs['robot'][side]['qvel'],obs['robot'][side]['ee_pose'],obs['robot'][side]['gripper_width']] for side in ['left','right']])

class EpisodeWriter:
    def __init__(self,path,env,image_stride=5):
        if image_stride<1: raise ValueError('image_stride must be >=1')
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=h5py.File(path,'w')
        self.env=env
        self.stride=image_stride
        self.file.attrs['format']='dualstack-v1'
        self.file.attrs['metadata']=json.dumps({'config':env.config,'scene':env.scene,'seed':env.seed,'dependencies':dependencies(),'python':sys.executable,'python_version':platform.python_version(),'platform':platform.platform(),'image_stride':image_stride,'alignment':'obs[t], actions[t], obs[t+1]; images indexed by observation_index','camera_convention':'OpenCV +x right +y down +z forward; T_world_from_cv','robot_layout':'left then right; each qpos[7],qvel[7],ee xyz+wxyz[7],width[1]'})
        from dualstack.envs.scene import ROOT,build_scene
        self.file.attrs['model_source_manifest']=(ROOT/'assets/franka_emika_panda/SOURCE.json').read_text(encoding='utf-8')
        self.file.attrs['scene_xml_sha256']=hashlib.sha256(build_scene(env.config).encode()).hexdigest()
        self.obs_index=0
        self.finished=False

    def append(self,name,value):
        value=np.asarray(value)
        if name not in self.file:
            self.file.create_dataset(name,shape=(0,*value.shape),maxshape=(None,*value.shape),dtype=value.dtype,compression='gzip' if value.ndim>=2 else None,chunks=True)
        ds=self.file[name]; ds.resize(ds.shape[0]+1,axis=0); ds[-1]=value

    def observation(self,obs,force_image=False):
        self.append('observations/robot',robot_vector(obs))
        self.append('observations/time',np.float64(obs['time']))
        self.append('privileged/qpos',self.env.data.qpos.copy())
        self.append('privileged/qvel',self.env.data.qvel.copy())
        self.append('privileged/blocks',np.array([np.r_[self.env.data.body('block_'+s).xpos,self.env.data.body('block_'+s).xquat] for s in ['a','b']]))
        self.file['privileged'].attrs['privileged']=True
        if self.obs_index%self.stride==0 or force_image:
            images=self.env.render()
            calibration=camera_info(self.env)
            self.append('images/observation_index',np.int64(self.obs_index))
            for name,img in images.items():
                self.append('images/'+name,img)
                self.append('cameras/'+name+'/K',calibration[name]['K'])
                self.append('cameras/'+name+'/T_world_from_cv',calibration[name]['T_world_from_cv'])
        self.obs_index+=1

    def transition(self,action,reward,terminated,truncated,info):
        for key,value in {'actions':action,'rewards':np.float64(reward),'terminated':np.bool_(terminated),'truncated':np.bool_(truncated),'success':np.bool_(info['success']),'transition_time':np.float64(info['sim_time'])}.items(): self.append(key,value)
        self.file.attrs['final_info']=json.dumps(info)
        self.append('diagnostics/hold_time',np.float64(info['hold_time']))
        self.append('diagnostics/support_force',np.float64(info['support_force']))
        if 'diagnostics/info_json' not in self.file:
            self.file.create_dataset('diagnostics/info_json',shape=(0,),maxshape=(None,),dtype=h5py.string_dtype('utf-8'))
        ds=self.file['diagnostics/info_json']; ds.resize(ds.shape[0]+1,axis=0); ds[-1]=json.dumps(info)

    def close(self,complete=False):
        self.file.attrs['complete']=complete
        self.file.close()

def _inspect_episode(path):
    errors=[]
    with h5py.File(path,'r') as f:
        t=len(f['actions']); meta=json.loads(f.attrs['metadata'])
        if f.attrs.get('format')!='dualstack-v1': errors.append('unknown format')
        if not f.attrs.get('complete',False): errors.append('incomplete recording')
        for name in ['observations/robot','observations/time','privileged/qpos','privileged/qvel','privileged/blocks']:
            if len(f[name])!=t+1: errors.append(name+' must have T+1 rows')
        for name in ['rewards','terminated','truncated','success','transition_time']:
            if len(f[name])!=t: errors.append(name+' must have T rows')
        if f['actions'].shape!=(t,16): errors.append('action shape')
        if f['observations/robot'].shape!=(t+1,44): errors.append('robot shape')
        if not np.allclose(f['observations/time'][1:],f['transition_time'][:],atol=1e-10): errors.append('transition time alignment')
        if not np.allclose(np.diff(f['observations/time'][:]),meta['config']['control_dt'],atol=1e-8): errors.append('control interval')
        if not bool(f['terminated'][-1] or f['truncated'][-1]): errors.append('missing terminal transition')
        if np.any(f['terminated'][:-1]) or np.any(f['truncated'][:-1]): errors.append('early terminal transition')
        if np.any(f['terminated'][:]&f['truncated'][:]): errors.append('terminated and truncated overlap')
        indices=f['images/observation_index'][:]
        if indices[0]!=0 or indices[-1]!=t or np.any(np.diff(indices)<=0): errors.append('image indices')
        shapes={}
        h,w=meta['config']['resolution']
        for role,value in meta['config']['cameras'].items():
            camera=value.get('name',role)
            ds=f['images/'+camera]
            shapes[camera]=list(ds.shape)
            if ds.shape!=(len(indices),h,w,3) or ds.dtype!=np.uint8: errors.append(camera+' image format')
            for part,shape in [('K',(3,3)),('T_world_from_cv',(4,4))]:
                if f[f'cameras/{camera}/{part}'].shape!=(len(indices),*shape): errors.append(camera+' calibration alignment')
        for key in ['actions','observations/robot','privileged/qpos','privileged/qvel']:
            if not np.isfinite(f[key][:]).all(): errors.append(key+' nonfinite values')
        final=json.loads(f.attrs['final_info'])
        if bool(f['success'][-1])!=final['success']: errors.append('final success mismatch')
        result={'valid':not errors,'errors':errors,'transitions':t,'observations':t+1,'image_frames':len(indices),'image_shapes':shapes,'final_info':final,'metadata':meta}
    return result

def inspect_episode(path):
    try:
        return _inspect_episode(path)
    except (OSError,KeyError,ValueError,IndexError,TypeError) as exc:
        return {'valid':False,'errors':[f'{type(exc).__name__}: {exc}']}

def replay_video(path,output,camera='overview'):
    with h5py.File(path,'r') as f:
        meta=json.loads(f.attrs['metadata'])
        time=f['observations/time'][:]
        indices=f['images/observation_index'][:]
        # Exact frame timestamps are saved alongside video; last frame may be shorter.
        fps=1/(meta['config']['control_dt']*meta['image_stride'])
        output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
        with imageio.get_writer(output,fps=fps) as writer:
            for frame in f['images/'+camera]: writer.append_data(frame)
        output.with_suffix('.timestamps.json').write_text(json.dumps(time[indices].tolist()),encoding='utf-8')
        return {'frames':len(indices),'fps':fps,'output':str(output)}
