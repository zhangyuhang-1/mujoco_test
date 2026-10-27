# Design and coordinate conventions

`StackEnv` is a Gymnasium-style equivalent: `reset(seed, options) -> (observation, info)`, `step(action) -> (observation, reward, terminated, truncated, info)`, `render(camera=None)`, and `close()`. There is no Gymnasium dependency so the project remains small on Windows.

The model is two fixed-base Franka Emika Panda arms from Menagerie. The active arm is `left` by default. The right arm is driven by its home posture task and is not used by the scripted policy. The XML uses position actuators from the Panda model; Mink computes joint targets and writes only actuator controls, never object pose or velocity.

The action is a 16-vector `[left_xyz(3), left_quat_wxyz(4), left_gripper(1), right_xyz(3), right_quat_wxyz(4), right_gripper(1)]`. Positions are meters in the MuJoCo world frame; quaternions are unit `wxyz`; gripper is normalized open `[0,1]` where 1 is open. The control interval is 20 ms and contains 10 physics steps of 2 ms. Invalid shapes, nonfinite values, non-unit quaternions, and unsafe positions raise `ValueError`.

The policy observation contains RGB images only when `images=true`, both arm `qpos/qvel`, gripper width, end-effector poses, time, and a task sentence. Object state is available only through `expert_state()` to the explicitly labelled oracle scripted expert and to the independent evaluator. It is never inserted in the normal observation.

The success checker requires the A footprint inside B, correct support height, upright orientation, low body velocity, open grippers away from A, and measured A-B contact force. The conditions must hold for `hold_seconds`; success is not inferred from the policy's state name. `terminated` marks success or explicit failure; timeout is `truncated`.

Camera images are RGB `uint8`, shape `(H,W,3)`, top-left origin, from one MuJoCo state. The overview camera covers the table and both arms. The two wrist cameras are attached to the hand bodies. Calibration exports `K`, `T_world_from_cv`, and its inverse. The optical convention is +x right, +y down, +z forward; MuJoCo's camera +y-up/-z-forward frame is converted explicitly. `projection_check.json` compares projected corners with segmentation pixels.
