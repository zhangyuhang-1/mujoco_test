"""CLI entry point; imports no renderer before backend configuration."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    # Numerical workers and orchestration must import MuJoCo even if there is
    # no usable OpenGL library/context. Only video workers enable rendering.
    if (len(sys.argv)>1 and sys.argv[1] in ("render", "_render","_observations")) or "--images" in sys.argv:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
    else:
        os.environ["MUJOCO_GL"]="disable"
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")
    os.environ.setdefault("LP_NUM_THREADS", "4")
    from dualstack.experiments.runner import main
    main()
