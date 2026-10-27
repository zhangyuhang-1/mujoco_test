from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass
class Detection:
    camera: str
    object_id: str
    bbox_xyxy: tuple[float,float,float,float]
    confidence: float
    position: tuple[float,float,float] | None = None
    quaternion_wxyz: tuple[float,float,float,float] | None = None
    frame: str = 'image_pixels'
    privileged: bool = False

    def validate(self):
        if not self.camera or not self.object_id: raise ValueError('Missing camera/object ID')
        box=np.asarray(self.bbox_xyxy)
        if box.shape!=(4,) or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]: raise ValueError('Invalid xyxy box')
        if not np.isfinite(self.confidence) or not 0<=self.confidence<=1: raise ValueError('Invalid confidence')
        if self.position is not None and (len(self.position)!=3 or not np.isfinite(self.position).all()): raise ValueError('Invalid position')
        if self.quaternion_wxyz is not None and (len(self.quaternion_wxyz)!=4 or not np.isfinite(self.quaternion_wxyz).all() or not np.isclose(np.linalg.norm(self.quaternion_wxyz),1,atol=1e-3)): raise ValueError('Invalid quaternion')
        if self.position is not None and self.frame=='image_pixels': raise ValueError('3D output requires explicit coordinate frame')
        return self

class PerceptionModel(Protocol):
    def reset(self): ...
    def predict(self, images, camera_info, robot_state) -> list[Detection]: ...

class EmptyPerception:
    def reset(self): pass
    def predict(self, images, camera_info, robot_state): return []

class ThirdPartyPerception:
    """Inject any callable backend; invalid detections are dropped and diagnosed."""
    def __init__(self, backend):
        self.backend=backend
        self.errors=[]
    def reset(self): self.errors=[]
    def predict(self,images,camera_info,robot_state):
        self.errors=[]; result=[]
        for item in self.backend(images) or []:
            try:
                detection=Detection(**item).validate()
                if detection.camera not in images: raise ValueError('Unknown camera')
                result.append(detection)
            except (ValueError,TypeError) as exc: self.errors.append(str(exc))
        return result
