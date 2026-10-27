# Validation record

Validated in the workspace Python 3.11.17 virtual environment based on the user's existing `D:\conda_envs\mujoco` Conda environment. No global environment was modified. `outputs/environment.json` records all primary versions and actual OpenGL vendor/renderer.

The fixed scene was implemented and run first. The physical motion lifted A from z=0.775 m to approximately 0.970 m and placed it at z=0.84498 m. A-B support force was 0.5886 N for a 0.06 kg cube; the final planar error was 0.000735 m. Success required 0.5 s of sustained conditions and occurred at simulation time 12.92 s.

Only after this fixed-scene validation did randomized evaluation begin. The seed list `0,1,2,3,4,5,6,7,8,9` was saved in `configs/evaluation.json` before randomized runs. All ten succeeded, with no seed replacement, threshold loosening, or random range reduction. A is randomized by up to 25 mm in each planar axis and yaw up to 0.12 rad; B is randomized within the same planar limits and remains fixed. First and final regression results are saved separately under `outputs/evaluation/` and `outputs/evaluation_release/`.

Three synchronized RGB camera images were exported and inspected. Wrist camera mounting was adjusted to remove self-occlusion; final projection checks compare corner projections against MuJoCo geom segmentation. Both blocks are visible in all three initial views and each bounding box comparison passes within 2 pixels. Wrist occlusion during physical grasp is expected and is visible in the saved recording.

The final HDF5 demonstration at `outputs/release/episode.h5` stores 646 transitions, 647 observations and 131 image sets. Calibration is stored at every image timestamp, together with the source model manifest and scene XML SHA256. The data checker verifies alignment, flags, shapes, finite values and terminal success. Video playback uses recorded frames and writes exact timestamps beside the MP4. `outputs/release/montage.jpg` shows eight stages from the actual saved frames.

The interactive Windows viewer was actually launched and completed the task. OpenGL selected `AMD Radeon(TM) Graphics` on this run and all exports were valid; the NVIDIA GPU selection was not changed. `outputs/viewer_verified/result.json` records the successful viewer run and oracle grasp/lift confirmations.

Failure paths were exercised separately: the holding-policy template timed out, yielding `truncated` with `reason=timeout` and a failure video; the deliberate exception policy yielded `reason=inference_exception`, recorded separately from physical failures. These diagnostic runs are not included in the ten-seed scripted-expert success rate.

Fourteen necessary tests pass. They cover reset/step/close, deterministic seeds, 50 valid random scenes and absence of initial robot-object contacts, bounded rejection sampling, image shapes, renamed cameras, optical-coordinate projection, empty/invalid perception, recorded real grasp rejection, floating/tilted/closed-gripper/transient counterexamples, timeout/fall semantics, malformed data, and saved demonstration integrity. Exact test output is exported as JUnit XML under `outputs/tests_final.xml`. A concurrent read attempted while the same demo file was being written initially encountered Windows HDF5 file locking; rerunning after the writer closed passed. No file locking was disabled.

Unverified: using the right arm to execute the task, larger randomization ranges, other OS/GPU driver combinations, learned perception/control models. B is intentionally static. Oracle geometric boxes do not model occlusion and must not be treated as a learned detection result. The project uses world targets with joint limits and workspace checks, but does not claim a general-purpose collision-free planner.
