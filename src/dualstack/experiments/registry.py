from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = {
    "01_friction": {"title": "Sliding friction", "parameter": "mu", "values": [0.12, 0.25, 0.5], "duration": 1.6, "dt": 0.002, "metric": "sliding_distance_m"},
    "02_bounce": {"title": "Drop and rebound", "parameter": "contact_damping", "values": [100.0, 1000.0, 3200.0], "duration": 1.6, "dt": 0.0002, "metric": "first_rebound_height_m"},
    "03_stack": {"title": "Offset stacking", "parameter": "offset_m", "values": [0.0, 0.033, 0.055], "duration": 2.6, "dt": 0.002, "metric": "final_tilt_rad"},
    "04_button": {"title": "Spring button", "duration": 4.0, "dt": 0.002, "metric": "settling_time_after_release_s"},
}
from dualstack.tasks.config import TASKS
REGISTRY.update(TASKS)


def cases(experiment, quick=False, suite="all", difficulty="all"):
    if experiment in TASKS:
        from dualstack.tasks.config import configurations
        return configurations(experiment,suite,difficulty)
    spec = REGISTRY[experiment]
    common = dict(experiment=experiment, dt=spec["dt"], duration=spec["duration"], seed=0,
                  group_budget_s=120, stop_speed_m_s=0.01, stop_hold_s=0.15)
    if experiment == "02_bounce":
        common["contact_stiffness"] = 2560000.0
    if experiment == "04_button":
        result = [dict(common, name=f"A_damping_{c:g}", family="A", stiffness=160.0, damping=c) for c in (0.4, 4.0, 12.0)]
        result += [dict(common, name=f"B_stiffness_{k:g}", family="B", stiffness=k, damping=2.0) for k in (80.0, 160.0, 320.0)]
    else:
        result = [dict(common, name=f"case_{i+1}", **{spec["parameter"]: value}) for i, value in enumerate(spec["values"])]
    return result


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)
