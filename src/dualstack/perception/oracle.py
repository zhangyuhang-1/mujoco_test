from dualstack.perception.base import Detection
from dualstack.sensors.cameras import block_bbox

class OraclePerception:
    """PRIVILEGED: geometric projection, no appearance inference or occlusion reasoning."""
    def __init__(self, env): self.env=env
    def reset(self): pass
    def predict(self, images, camera_info, robot_state):
        detections=[]
        for camera in images:
            h,w=images[camera].shape[:2]
            for block in ['a','b']:
                bbox=block_bbox(self.env,block,camera_info[camera])
                if bbox is None: continue
                bbox=[max(0,bbox[0]),max(0,bbox[1]),min(w-1,bbox[2]),min(h-1,bbox[3])]
                if bbox[2]<=bbox[0] or bbox[3]<=bbox[1]: continue
                body=self.env.data.body('block_'+block)
                detections.append(Detection(camera,block,tuple(bbox),1.,tuple(body.xpos),tuple(body.xquat),'world',True).validate())
        return detections
