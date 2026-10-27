import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import platform
from dualstack.data.episodes import dependencies
from dualstack.envs.stack import StackEnv

if __name__=='__main__':
    env=StackEnv()
    try:
        env.reset(0)
        images=env.render()
        from OpenGL import GL
        gpu={k:GL.glGetString(v).decode() for k,v in {'vendor':GL.GL_VENDOR,'renderer':GL.GL_RENDERER,'version':GL.GL_VERSION}.items()}
        result={'python':sys.executable,'base_prefix':sys.base_prefix,'version':sys.version,'platform':platform.platform(),'dependencies':dependencies(),'opengl':gpu,'image_shapes':{k:list(v.shape) for k,v in images.items()}}
        output=Path('outputs/environment.json'); output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
    finally: env.close()
