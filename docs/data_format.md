# HDF5 demonstration format

`scripts/collect.py` writes `dualstack-v1` files. For an episode with `T` transitions, `observations/*` and privileged state have `T+1` rows; actions, rewards, terminal flags and transition times have `T` rows. Row `t` means observation_t, action_t, then observation_{t+1}. Image datasets are indexed by `images/observation_index` and may be decimated by `--image-stride`; the terminal observation is always stored.

Fields include `observations/robot` `(T+1,44)`, `observations/time`, `actions` `(T,16)`, rewards, terminated, truncated, success, `transition_time`, privileged `qpos/qvel/blocks`, and `images/{overview,left_wrist,right_wrist}` `(N,256,256,3)`. Calibration is stored per image frame in `cameras/<name>/K` and `T_world_from_cv`. The HDF5 metadata stores seed, sampled scene, full config, camera convention, Python executable and dependency versions. `privileged/*` is explicitly marked oracle data and must not be used as a visual policy input.

`diagnostics/info_json` saves the complete info dictionary for each transition, while `diagnostics/hold_time` and `support_force` are numerical columns. The file also stores the bundled robot source manifest and scene XML hash. All robot/action data is saved at 50 Hz; `--image-stride` controls image frequency (default 5 = 10 Hz) while retaining all transitions.

`scripts/inspect_data.py` checks row alignment, dimensions, finite values, terminal semantics, image indexing, calibration alignment and final success. `scripts/replay.py` creates a video from saved image frames. This is video replay, not physics re-simulation. Re-simulation would depend on MuJoCo, controller and dependency versions and can diverge numerically; this project stores those versions for that reason.
