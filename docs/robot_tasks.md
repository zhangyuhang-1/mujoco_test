# 三个机器人任务（复用实验导出框架）

代码均在项目内。原有 `scripts/demo.py`、四个基础物理实验及对应交付目录保留。
新任务复用同一个 `scripts/experiments.py`、子进程/预算/resume、全状态记录格式、
资源相对路径导出、独立 ZIP 验证、交互回放、离屏视频和逐级降级系统。

## 服务器入口

```bash
conda activate /data_lewm/envs/mujoco_test
cd /home/haichao/zhangyuhang/mujoco_test
python scripts/experiments.py list
# 固定场景先验证；名称依次为 05_push、06_eccentric、07_tray
python scripts/experiments.py run 05_push --suite fixed --output outputs/tasks_push_fixed
# 固定场景 + 评测前预声明的 seeds 0–9，每个参数/难度都逐个运行
python scripts/experiments.py run 05_push
python scripts/experiments.py run 06_eccentric
python scripts/experiments.py run 07_tray
# 新任务批量运行（不会重跑旧实验）
python scripts/experiments.py run tasks --resume
# 仅数值/轨迹/回放验证，不渲染高清视频
python scripts/experiments.py run tasks --quick --output outputs/tasks_quick
# 只评测；--suite fixed / evaluation / all
python scripts/experiments.py run 07_tray --suite evaluation --difficulty medium --output outputs/tray_medium_eval
# 调整预算，单位为墙钟秒
python scripts/experiments.py run tasks --group-budget 120 --render-budget 900 --experiment-budget 3600
# 再次离线录像、打包、验收、统一下载包
python scripts/experiments.py render --package deliverables/robot_tasks/05_push/replay_package \
 --output deliverables/robot_tasks/05_push/videos/detail_again.mp4 --view detail --fps 30
python scripts/experiments.py pack deliverables/robot_tasks/05_push
python scripts/experiments.py audit deliverables/robot_tasks
python scripts/experiments.py bundle deliverables/robot_tasks
```

默认新任务输出至 `deliverables/robot_tasks/`。每项包含 `evaluation_protocol.json`（在运行前写入种子和全部配置）、
`evaluation.json/csv`、`runs/*/{config.json,data.csv,trajectory.npz,contact_history.npz,metrics.json,events.json}`、
曲线、报告、回放 ZIP 和视频。`--resume` 可以从固定场景扩展到全部场景；新增组会运行，
配置一致的已完成组保留。`--refresh-analysis` 可强制重新生成报告/回放/视频。

## 接口和代码

- `src/dualstack/tasks/config.py`：三档参数、三个难度、预声明种子、保守随机范围。
- `scenes.py`：复用双 Panda 和桌面，仅添加任务刚体、平台、把手和非碰撞目标标记。
- `env.py:TaskEnv`：`reset/step/observe/close`，16 维动作与原 `StackEnv` 一致。
- `policies.py:PushOracle/PlacementOracle/TrayOracle`：`reset/act/close` 脚本专家。
- `evaluation.py:Detector`：独立几何、接触、速度、夹爪状态及持续稳定检测，不读取专家阶段。
- `recording.py`：接入已有 `mjSTATE_INTEGRATION` 格式，每个物理步记录；多刚体位姿额外保存。
- `reporting.py`：按预声明参数/种子统计成功率和失败原因，选取演示样本。

普通观测只有 `robot/images/task/time`，不含物体位姿、COM、球真值或摩擦系数。
oracle 专家通过 `env.oracle.state()` 单独读取真值；`info` 中的评测结果不传给策略。
策略可由配置中的 `module:Class` 替换。默认 `images=false` 是 oracle 数值模式；
视觉模型接入时应启用图像，并在环境进程配置可用离屏上下文。暂未接入学习模型。

后续替换策略可以使用 `--policy module:Class`，例如创建自己的策略类后运行
`python scripts/experiments.py run 05_push --policy my_policies:Policy --output outputs/push_model_eval`。
策略只实现 `reset/act/close`，通过普通观测生成原 16 维动作，评测器和导出不变。
更改策略代码后应使用新的输出目录或不加 `--resume`，避免把旧轨迹当作新策略结果。

## 推块

自由方块质量默认 0.10 kg，半边长 0.025 m；桌面滑动摩擦 μ=0.15/0.35/0.65，显式 pair
保证有效摩擦生效，质量可在配置中修改。目标正方形半边长 0.055 m，无碰撞、无围墙/限位。
方块位置约 x=0±0.015 m、y=-0.32±0.015 m，目标约 x=0.17±0.015 m、y=-0.27±0.012 m，
方块 yaw 在 ±0.08 rad 内。固定场景不使用随机扰动。

左手安装可见的小球形推头，只通过显式接触对推块，不夹起物体。专家先接近、下降，
试探推进 0.045 m，观察方块实际位移，再计算后续慢推速度；不读取 μ。
推进时观察目标误差，撤离后观察滑动；超调时从反侧重新规划，最多 4 次推动。
所有实际速度、试探响应、推动次数、超调、末端目标和实际轨迹均记录。

检测方块完整足迹进入目标区（含旋转足迹）、速度 <0.012 m/s、角速度 <0.12 rad/s、
桌面支撑、机械臂无接触且撤离 >0.10 m，连续 0.6 s 才成功。
方块离桌面超过 15 mm 会以 `lifted_not_pushed` 失败，另外记录掉落、越界和超时。

## 偏心放置

长条边长 0.20×0.036×0.036 m，质量 0.12 kg；黄色配重与长条同一刚体，
几何体质量 0.04/0.08/0.16 kg，局部 x 偏移 ±0.055 m。MuJoCo 从几何体质量推断总质量、COM、
主惯量和惯性系朝向，记录编译后的实际值；没有可松动关节或运行时质量修改。
平台半宽 x=0.028 m、y=0.045 m，顶面高 0.82 m。

偶数种子配重在右、奇数种子在左，位置小范围变化、yaw ±0.08 rad。
专家在可抓握的长条中部真实抓取，用实际旋转 R 和局部 COM 计算
`p_geometry.xy = p_platform.xy - (R @ COM_local).xy`。
夹爪姿态同初始 yaw，完成接近、双指接触确认、真实抬升、搬运、慢放、释放和撤离。
抓取点尽量接近 COM，并限制到配重外形之外的可夹持区域。
这两个抓持任务的夹爪位置伺服 kp=500 N/m，仍使用原执行器的 ±100 N 力上限；
原堆叠任务的参数没有更改。初版 kp=100 N/m 导致长条滑脱的轨迹保留在
`outputs/tasks_dev/place_fixed_v1`、`place_fixed_v2`，最终调试不使用固定连接。
不要求平台与几何中心对齐；配重左右交换时放置偏移反向。

成功需要真实抓持/抬升历史、平台支撑、COM 投影在支撑区内、姿态合理、低速，
夹爪张开 >65 mm、无机器人接触且撤离 >0.10 m，连续 0.6 s。
悬空、正在掉落或仍被夹住不计成功。另记滑落、翻倒、掉落和超时。

## 双臂托盘

托盘为自由刚体，绿色把手与托盘同一刚体，球为独立自由刚体。双指真实接触抓持把手；
没有机器人—物体 weld、mocap、强制附着或逐帧物体位姿赋值。
托盘半尺寸约 0.14 m，把手局部位置 `[0, ±0.185, 0.034]` m，球半径 0.018 m。
目标手位置按 `p_hand = p_tray + R_tray @ p_handle_local` 生成。
双手同步平滑接近、闭合、双侧双指接触确认、抬升、运输、稳定保持。
根据球相对位置和速度减速/暂停；暂停和抓持重试都有预算，动作超出安全工作区明确失败。

- easy：直线运输约 0.09 m，较高但可见的低围边，固定球初始位置。
- medium：目标 x≈0.15 m、y≈0.04 m，一次转弯，小范围球位置变化；与 easy 分开统计。
- physics_variation：保持 easy 几何/路径/球初始位置，仅改变滚动摩擦 0.0001/0.002/0.01。

easy 的初始条件有意固定，因此 10 个种子是确定场景的重复验证，不应解释成 10 种随机场景的泛化统计。
medium 才增加球初始位置的随机变化；physics_variation 固定其他条件，按预声明种子循环分配滚动摩擦。
当前平移/暂停专家对极低滚动摩擦的恢复能力有限；若达到暂停预算就明确失败，绝不计为成功。

检测托盘到目标的距离、倾角、速度、球是否仍在有效内部区以及球速度、两侧双指抓持和
把手跟踪误差，连续稳定 0.6 s 才成功。无需松手放回桌面。
抓持不可靠时保留失败过程，标记部分完成，绝不改成固定连接冒称成功。

## 记录、输出及物理准确性

延用四个实验的资源相对路径 ZIP、最小 `replay.py`、暂停/重播/倍速/单步/时间跳转/自由视角。
回放只 `mj_setState + mj_forward`，不运行控制策略或继续推进物理；托盘和球都逐帧检查位姿。
历史力单独存储，不能把恢复位置后重新算出的力当作原接触历史。
视频默认原生 1080p/30 fps，总览和特写；优先展示固定场景和一个代表性失败（若存在），
全部评测轨迹都在包内，不从成功种子中挑选统计结果。失败/降级规则与既有框架相同。
完整 suite 另附明确标注的 `validation_short_budget` 反例：按相同策略执行，
但在接近/抓持/抬升阶段达到短时间预算；必须判为未完成。
它不计入评测成功率，用来展示“仍在执行、仍被抓持、尚未稳定”不会误判为成功。
自然评测中的失败也全部保留，不用这个反例替代或掩盖实际失败。

每项代表场景另用 dt=0.001 s 对照主步长 0.002 s，报告独立检测结果、主要误差、穿透和初始条件。
这属于步长敏感性检查，不代表真实机器人精度或完整数值收敛证明。

最终数据复核按刚体身份识别无名指尖碰撞几何体的机器人接触，并使用主惯性系计算复合刚体转动能；
这些是由原接触历史、位姿和速度重新计算的派生列修正，不修改仿真状态或历史接触力。
修正前 CSV 保留在 `outputs/tasks_dev/recorded_csv_before_derived_corrections/`，记录见
`verification/derived_data_review.json`。所有成功末帧及末段无接触要求均另行复核。

## 官方参考和版本

实际环境 MuJoCo 3.17.0，Python 3.11。版本化 3.17.0 文档仍无法通过当前工具访问；
采用官方 stable 文档和官方 Panda 参考代码，并在安装版本编译和运行验证，没有冒称读取了不可达页面。

- [接触、摩擦混合、抓持滑移排查](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)
- [刚体质量和惯量自动推断](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-inertiafromgeom)
- [惯性坐标系](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-inertial)
- [刚体速度接口](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-objectvelocity)
- [官方 Panda 夹爪、接触几何及执行器](https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/panda.xml)

## Windows 打开

下载某项的 `replay.zip`，完整解压；Python 3.11 环境执行：

```powershell
python -m pip install -r replay_requirements.txt
python replay.py
```

`--case N` 选组，`--all` 依次播放；C 切到自由视角，Space 暂停，R 重播，+/- 倍速，
左右箭头跳转，逗号/句号单步，[ / ] 换组。服务器仅验证加载、全状态及离屏图像；
**回放数据和加载已验证，本地 GUI 待验证**，实际状态以各项 manifest 和 audit 报告为准。
