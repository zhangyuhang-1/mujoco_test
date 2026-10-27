# References and licensing

The runtime uses the official Python MuJoCo bindings and the Panda MJCF from [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda). The model is released under Apache-2.0; its original `LICENSE`, `README.md`, all referenced meshes, and `SOURCE.json` are kept in `assets/franka_emika_panda/`. The downloaded source bytes are identified by SHA256 in `SOURCE.json`.

The control layer uses [Mink](https://github.com/kevinzakka/mink), Apache-2.0, pinned to 0.0.13. It converts operational-space end-effector targets to joint position commands; this project does not copy Mink source. The installed dependency versions are recorded in every HDF5 episode.

The design review consulted these upstream sources and stores the retrieved copies in `docs/upstream/`:

- [robosuite TwoArmLift](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/two_arm_lift.py), [Stack](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/stack.py), and [OSC controller](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/controllers/parts/arm/osc.py). These informed the two-arm layout, staged waypoints, independent contact checks, and 256x256 RGB camera convention. Robosuite is MIT-licensed; no robosuite code is imported at runtime.
- [ACT scripted policy](https://github.com/tonyzhaozh/act/blob/main/scripted_policy.py) and [simulation environment](https://github.com/tonyzhaozh/act/blob/main/sim_env.py). These informed finite waypoint interpolation and synchronized camera observations. ACT is MIT-licensed; no ACT code is imported at runtime.
- [Mink README](https://github.com/kevinzakka/mink) and [MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html). The hosted MuJoCo page returned 403. The official documentation source was subsequently retrieved from [MuJoCo doc/python.rst](https://github.com/google-deepmind/mujoco/blob/main/doc/python.rst), and used to check passive viewer ownership, `sync()`, and rendering. `docs/upstream/research_downloads.json` records retrieval results. The requested v0.0.13 Mink dual-Panda example URL returned 404; it was not used or claimed as reused code. Installed Mink 0.0.13 APIs were inspected directly and tested.

Upstream reference snapshots retain their accompanying `robosuite_LICENSE`, `act_LICENSE`, and `mink_LICENSE` in the same directory. Only one robot resource subtree is bundled; complete upstream repositories were not copied.

The task intentionally fixes B as a static body in version 1. A is a dynamic free body and moves only through MuJoCo contacts and Panda actuator control. Changing B to a free joint requires changing only the scene body and reset/success configuration.
