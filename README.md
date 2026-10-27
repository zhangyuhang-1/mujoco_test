# MuJoCo 双 Panda 方块堆叠 benchmark

GitHub 仓库包含源代码、配置、测试、文档和运行所需的 Panda 模型资产。`outputs/`、`deliverables/`、虚拟环境及认证配置不纳入版本管理；下文的结果报告和视频链接需要运行实验生成后在本地查看。

新增四项视觉物理推理演示：修正偏心放置正负对照、移除承重支撑、隐藏配重倾斜观察与放置、双臂倾倒取球。
本次只交付视频和数值结果，不生成回放压缩包。见[运行与设计说明](docs/visual_tasks.md)、[实际结果](deliverables/vision_tasks/REPORT.md)。

```bash
python scripts/experiments.py run visual --no-package --resume
```

新增机器人任务：**推块停进目标区、偏心物体稳定放置、双臂托盘运球**。
继续复用同一个实验记录/评测/回放/视频导出框架，说明见
[机器人任务文档](docs/robot_tasks.md)，结果见
[机器人任务交付](deliverables/robot_tasks/REPORT.md)。

```bash
python scripts/experiments.py run tasks --resume
```

`run all` 仍表示原有四个基础物理实验；`run tasks` 表示这三个新任务。

新增四项物理实验（滑动摩擦、落球回弹、偏移堆叠、弹簧按钮），支持无桌面服务器、
数据/曲线导出、高清视频和 Windows 三维回放 ZIP。完整操作见
[物理实验说明](docs/physics_experiments.md)，已生成结果见
[交付汇总](deliverables/physics/REPORT.md)。原堆叠 demo 保留不变。

```bash
conda activate /data_lewm/envs/mujoco_test
python scripts/experiments.py list
python scripts/experiments.py run all --resume
```

这是一个可运行的第一阶段项目：固定桌面上有两只 Panda 和平行夹爪，左臂通过 Mink 逆运动学真实抓取动态方块 A，将它放到固定底座 B 上并释放；右臂保持安全姿态。B 固定是第一版简化，A 从未被瞬移、焊接或强制附着。成功由独立接触/几何/速度检测器判定。

## 安装

Windows PowerShell（项目目录）执行：

```powershell
D:\conda_envs\mujoco\python.exe -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -e .
# 如需完整传递依赖锁定：
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

当前已验证环境：`.venv\Scripts\python.exe`，Python 3.11.17，Windows 10，MuJoCo 3.17.0，Mink 0.0.13，NumPy 2.4.6，SciPy 1.17.0，HDF5 3.15.1，imageio 2.37.2。依赖锁定在 `pyproject.toml`。

## 运行

```powershell
# 固定场景闭环，保存三路阶段图像和 overview 视频
.\.venv\Scripts\python.exe scripts/demo.py --video

# 交互式 MuJoCo 查看器（Windows 使用 GLFW，不配置 Linux EGL）
.\.venv\Scripts\python.exe scripts/demo.py --viewer

# 导出同一仿真状态的 overview / left_wrist / right_wrist、标定和投影检查
.\.venv\Scripts\python.exe scripts/export_cameras.py

# 采集 HDF5 演示、完整性检查和视频回放
.\.venv\Scripts\python.exe scripts/collect.py --video
.\.venv\Scripts\python.exe scripts/inspect_data.py outputs/demo/episode.h5
.\.venv\Scripts\python.exe scripts/replay.py outputs/demo/episode.h5

# 10 个预先声明的随机种子，不挑选成功样本
.\.venv\Scripts\python.exe scripts/evaluate.py

# 随机 demo / 自选种子 / 策略输入图像
.\.venv\Scripts\python.exe scripts/demo.py --random --seed 4 --images --video

# 感知接口调用示例
.\.venv\Scripts\python.exe scripts/perception_example.py

# 必要测试（环境、复现、采样、相机、检测器、成功反例、数据）
.\.venv\Scripts\python.exe -m pytest -q
```

## 代码入口

- `src/dualstack/envs/scene.py` 生成双 Panda、桌面、A/B 和三相机 MJCF；`envs/stack.py` 提供环境接口。
- `src/dualstack/control/ik.py` 是 Mink 控制封装；动作定义和限制见 `docs/design.md`。
- `src/dualstack/policies/scripted_stack.py` 是可替换的 oracle 航点专家。替换动作策略只需实现 `reset/act/close` 并通过 `--policy module:Class` 传入。
- `src/dualstack/evaluation/success.py` 是独立成功检测器；阈值在 `configs/default.json` 的 `success` 中。
- `src/dualstack/perception/base.py`、`perception/oracle.py` 定义空检测、第三方模板和明确标记的特权 oracle。第三方 2D 框必须再经过深度/标定转换，不能直接作为世界抓取点。
- `src/dualstack/data/episodes.py` 定义采集、完整性检查和回放。

随机范围在 `configs/default.json` 的 `random_xy`、`random_yaw` 和 `sample_attempts`；固定 A/B 位置在 `a_xy`、`b_xy`。采样采用有限拒绝次数，失败会显式抛错。

相机角色固定为 `overview/left_wrist/right_wrist`，各角色的 `name/pos/xyaxes/fovy`、全局 `resolution` 均可配置。`images=false` 是低开销 oracle 评测模式；视觉策略应设 `images=true` 或使用 demo 的 `--images`。相机导出和演示采集始终会渲染三路图像。调试 viewer 独立于策略输入；视频默认使用 overview。动作是 16 维（左后右，每臂世界坐标 xyz、wxyz 单位四元数、0 关闭/1 张开），控制 50 Hz，物理 500 Hz。

自定义策略示例：`.\.venv\Scripts\python.exe scripts/demo.py --policy dualstack.policies.model_policy:ModelPolicy`。该模板会保持当前姿态，不执行堆叠；它不读取物体真值。感知模型接入 `ThirdPartyPerception(backend)`，backend 返回 Detection 字段字典；空结果返回 `[]`，无效结果跳过并写入 `.errors`。从深度到世界坐标的入口是 `sensors/cameras.py:pixel_depth_to_world`。

## 已验证结果与限制

固定场景 seed 0 实测成功：12.92 s，A-B 平面误差约 0.735 mm，高度误差约 0.018 mm，支撑力 0.5886 N，条件持续 0.5 s；最终视频位于 `outputs/release/episode.mp4`，HDF5 在同目录。预声明 seeds 0–9 全部成功（10/10），首次结果位于 `outputs/evaluation/`，最终回归结果在 `outputs/evaluation_release/`。相机导出三路 256x256 RGB，投影检查通过；HDF5 演示包含 646 个动作、647 个观测、131 个同步图像帧，检查通过。

交互 viewer 已实际执行成功（`outputs/viewer_verified/result.json`）。本次 OpenGL 实际使用 AMD Radeon 核显，渲染正常；完整环境报告在 `outputs/environment.json`。若你的机器再次黑屏，可在 Windows 图形设置中为本项目的 `.venv\\Scripts\\python.exe` 指定 NVIDIA 高性能 GPU；本项目不修改系统显卡设置，也不设置 Linux EGL。

未验证右臂作为执行臂的抓取任务、其他操作系统或更大随机范围。右臂动作接口和 `active_arm` 选择已预留，当前摆放和验证针对左臂。未实现训练大型感知/学习模型、动力学随机化和 B 可移动版本；已提供模型策略与感知接口模板。联网调研中 GitHub API 限流、MuJoCo 文档网页 403 和部分示例 404 已记录；已成功读取官方文档源码，不影响运行验证。

完整设计、坐标约定、数据字段、来源许可证及验证过程分别见 `docs/design.md`、`docs/data_format.md`、`docs/references.md`、`docs/validation.md`。
