# 四项物理实验：运行、下载和本地三维回放

原有 `scripts/demo.py` 和堆叠策略保持不变。新增入口是 `scripts/experiments.py`。
实验输出默认集中在项目的 `deliverables/physics/`，不修改上级目录和全局环境。

## 最简闭环

服务器（SSH，无需 DISPLAY）：

```bash
conda activate /data_lewm/envs/mujoco_test
cd /home/haichao/zhangyuhang/mujoco_test
python scripts/experiments.py run all
```

下载任意实验的 `replay.zip`。例如在本地 Windows PowerShell：

```powershell
scp haichao@你的服务器:/home/haichao/zhangyuhang/mujoco_test/deliverables/physics/01_friction/replay.zip .
Expand-Archive replay.zip -DestinationPath friction_replay
cd friction_replay
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r replay_requirements.txt
.\.venv\Scripts\python.exe replay.py
```

每个 ZIP 独立包含 XML、所有 Panda 网格和许可证、所有参数组轨迹、CSV 和历史接触力、
最小播放器和固定版本依赖。播放器只要求 MuJoCo 和 NumPy（其余由 MuJoCo 的包依赖安装），
不要求 Mink、pytest、训练框架或服务器项目。不要把 NPZ 拖入 simulate.exe。
服务器验证了 ZIP 解压后独立新进程的资源加载、每一记录帧位姿及关键时刻离屏图像。
**回放数据和加载已验证，本地 GUI 待验证。**

## 入口

```bash
# 注册表和支持状态
python scripts/experiments.py list
# 单项（其余编号：02_bounce、03_stack、04_button）
python scripts/experiments.py run 01_friction
# 全部；各项由独立子进程执行；每一参数组也有独立进程
python scripts/experiments.py run all
# 快速数值和回放检查，完整物理时长但不渲染视频
python scripts/experiments.py run all --quick --output outputs/physics_quick
# 仅数值，无轨迹和视频
python scripts/experiments.py run all --no-video --no-trajectory --output outputs/physics_numeric
# 继续未完成项，已通过项不重算
python scripts/experiments.py run all --resume
# 推导指标更新后，保留相同配置的物理轨迹，重新分析/打包/录像
python scripts/experiments.py run 04_button --resume --refresh-analysis
# 配置预算，单位为墙钟秒
python scripts/experiments.py run all --group-budget 120 --render-budget 900 --experiment-budget 3600
# 仅从现有轨迹重新渲染；相机不属于原策略观测协议
python scripts/experiments.py render --package deliverables/physics/01_friction/replay_package \
  --output deliverables/physics/01_friction/videos/detail_again.mp4 --view detail --width 1920 --height 1080 --fps 30
# 重新打包和从 ZIP 验证（不会启动桌面窗口）
python scripts/experiments.py pack deliverables/physics/01_friction
python scripts/experiments.py validate deliverables/physics/01_friction
# 最终验收：所有 CSV、ZIP 解压后新进程逐帧位姿/离屏图像、所有视频全片解码和帧数
python scripts/experiments.py audit deliverables/physics
# 汇总 ZIP：包含四项回放 ZIP、视频、数值数据、曲线、日志和报告，排除重复解压资产
python scripts/experiments.py bundle deliverables/physics
# 测试包含原有 demo 回归
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -m pytest -q --basetemp=outputs/pytest_tmp
```

`--no-video` 可保留轨迹；`--no-trajectory` 需要同时 `--no-video`，因为离屏视频使用记录状态。
`--quick` 不缩短物理过程、不减少参数组，只省略最耗时的高清视频。
参数定义见 `src/dualstack/experiments/registry.py`，每组实际配置和初始条件在
`runs/<case>/config.json` 和 `metrics.json`。如需精确重跑某一导出配置：

```bash
python scripts/experiments.py _physics --config deliverables/physics/01_friction/runs/case_1/config.json \
  --output outputs/friction_exact_repeat
```

## 本地操作

`python replay.py --case 2` 选择第二组；`--all` 依次播放全部组。

| 按键 | 操作 |
|---|---|
| Space | 暂停 / 继续 |
| R | 从头重播 |
| + / - | 0.125–8 倍速 |
| 逗号 / 句号 | 暂停并逐记录帧后退 / 前进 |
| 左 / 右箭头 | 跳转 -/+ 1 秒 |
| Home / End | 起点 / 终点 |
| 1–9、[、] | 选组、上一组、下一组（重开窗口） |
| C | 特写 → 总览 → 自由视角 |

自由视角下鼠标左键旋转，右键平移，滚轮缩放；Shift 为 MuJoCo 默认修饰键。
末帧保持直到关闭或重播。当前组参数、时间、播放状态在 viewer 内叠加显示。
回放不运行控制策略、不调用 `mj_step`，不能任意改物体后继续仿真。

## 实验设计与判据

所有实验复用原桌面和双 Panda。未执行实验的手臂用原关节位置执行器保持安全姿态。
种子固定 0；当前实验没有随机扰动。模型参数在每组加载时确定，运行中不改变。
仅 reset 设置初始位置、姿态和速度；运行过程使用 `mj_step`、关节执行器和正常广义外力。

### 01：滑动摩擦

质量 0.10 kg、边长 0.06 m 的箱形方块；初始水平速度 0.7 m/s；滑动摩擦
μ=0.12、0.25、0.50，其他条件相同。先静置建立支撑，再在 reset 结束一次性赋初速度。
使用显式 `<pair>` 指定两个切向摩擦分量，避免自动 geom 混合的 max 规则使低摩擦设置失效。
实际有效值从 `mjData.contact.friction` 逐步采集。

速度 <0.01 m/s 且有桌面支撑连续 0.15 s 才算停止；另外记录未停止、越界。
距离按累计水平路程计算。理想参考是 v0²/(2μg)，忽略软接触初始瞬态、微小转动、
离散积分和速度阈值。记录能量而非仅凭视觉断言耗散。默认步长 0.002 s，代表组另跑 0.001 s。

### 02：落球回弹

半径 0.035 m、质量 0.10 kg；球心初始高度 1.18 m、零初速度；桌面高 0.75 m。
显式接触对使用 direct `solref=(-2560000, -damping)`，damping=100、1000、3200。
固定 impedance=0.99、刚度相关参数、摩擦和质量，只改变阻尼项。direct 参数属于
MuJoCo 约束参考加速度的参数化，不能把这些数字当作实测材料弹簧 N/m 或恢复系数。
低阻尼到高阻尼趋势来自真实轨迹，不强行要求每组回弹。

首次接触自动识别；离开接触后竖直速度由正变负判定首次自由飞行峰值。
报告球底离桌面的回弹净高度，以及碰前/脱离后速度、测得速度比、法向力、最大穿透。
没有可分辨回弹为有效物理结果。默认步长 0.0002 s，代表组另跑 0.0001 s。
每物理步记录，视频 60 fps；慢放从同一轨迹取首次碰撞前后片段，0.25 倍速，未改变步长。

### 03：偏移堆叠

保留原尺寸：上块边长 0.05 m，下块边长 0.07 m，底座固定。
初始 COM 水平偏移 0、0.033、0.055 m，涵盖中心、接近支撑边界和边界外。
上块自由，reset 后自然演化，μ=1.5，接触参数一致；不要求机械臂放置。

分类仅使用记录轨迹：有底座支撑、低速、低角速度、小倾角连续 0.8 s 且末段仍满足
才判 stable；失去高度支撑且显著倾斜为 tipped；掉落但无显著倾斜为 slid_off；
其余 timeout_undetermined。分类不根据输入 offset 写死。记录完整四元数、角速度和接触。
未添加参与碰撞的“重心标记”。默认步长 0.002 s，代表组另跑 0.001 s。

### 04：弹簧按钮

竖直 slide 关节，按钮质量 0.10 kg，有限行程 [-0.09,0.025] m。
A：k=160 N/m，c=0.4、4、12 N·s/m；B：c=2 N·s/m，k=80、160、320 N/m。
按组只改变目标参数；重力平衡 q_eq=-mg/k，初始状态按相应重力平衡设置。

先运行有时限的机械臂按压诊断：左手增加物理接触压头，IK 经关节执行器缓慢接近、
按下、撤离；检查实际接触、按下至少 5 mm 及随后分离。
不满足时整项统一降级到已知外力，不对不同组混用两种输入；原因和诊断轨迹保留。
已知外力：0–0.35 s 线性增加至 -3 N，保持至 0.8 s，1 s 时撤除。撤力后不修改位置。
压头没有默认碰撞，仅与按钮显式接触；灰色弹簧导向杆仅为无碰撞装饰。

记录弹簧力 -kq、阻尼力 -c*qdot、重力 -mg、已知外力和历史接触力，不能混称“按钮力”。
机械臂模式的释放时刻取记录中最后非零接触法向力之后的第一个采样点；
不把机械臂完全撤离的 3.3 s 航点时间当作接触力消失时刻。
弹簧力—位移图不包含阻尼和重力；另绘各分力随时间图。回弹峰值相对重力平衡识别。
稳定要求位移误差 <0.5 mm、速度 <3 mm/s，后续全部保持且至少留出 0.4 s。
默认步长 0.002 s，代表组另跑 0.001 s。有限预算内未稳定不属于执行失败。

## 记录、视频和失败处理

- `mj_getState/mj_setState` 使用 **mjSTATE_INTEGRATION**，保存 time/qpos/qvel/act/history/
  warmstart/ctrl/qfrc_applied/xfrc_applied/eq_active/mocap/userdata/plugin 等签名中的字段；
  每个模型的实际状态长度、版本和采样间隔随包导出。没有运行中参数修改或外部插件。
- `trajectory.npz` 每个物理步保存状态、时间和物体位姿；球碰撞没有降采样。
  这些样本不能恢复未记录的连续时间细节。viewer 以保存时间选择前一个记录帧，不做物理推进。
- 历史力、接触点和有效摩擦另存 `contact_history.npz`，法向/切向力在 MuJoCo 接触局部坐标系；
  不声称只恢复位置可精确重建历史接触力。CSV 保存每步汇总量。
- 数值执行、打包验证和录像解耦；录像使用专用 film_overview/film_detail 相机，
  不改原 demo 的策略相机协议。原生离屏缓冲 1920×1080，4 倍多重采样。
  数值进程导入 MuJoCo 时显式禁用 OpenGL；ZIP 验证先在无 GL 新进程检查数据，
  再独立尝试离屏图像检查。缺少渲染库/上下文不能阻止基础数值仿真和轨迹导出。
- 每视角按 **1080p → 720p → 640×360 → 320×180** 独立进程重试，记录每级错误并释放上下文。
  编码仍失败则尝试 320×180 PNG 序列；完全无上下文时不无限重试，保留数据和回放包。
- 报告渲染、文字叠加、编码提交和编码结束等待的独立耗时。FFmpeg 异步工作，编码提交时间
  包含管道等待，不等价于编码器纯 CPU 耗时；不会把整段录像耗时当成物理计算耗时。
- 每项/每参数组独立进程及预算；超时终止子进程和编码进程。状态为 pending、running、passed、
  partial、skipped、failed。非预期趋势与执行状态分开；跌落和不回弹不是失败。
  缺少必要 API 且无可靠替代才 skipped；Python 异常 failed；部分产物失败 partial。
- 运行器只做隔离、记录、降级和继续，不自动改代码。开发修复和测试记录见交付报告。
- `--resume` 跳过完整 passed 实验及已通过参数组，partial 实验重新尝试缺失输出。

## 官方参考与版本核对

安装和验证版本：MuJoCo **3.17.0**、Python **3.11.17**，其余依赖写入 manifest。
请求 `https://mujoco.readthedocs.io/en/3.17.0/...` 的版本化文档未能通过当前网络工具取得，
因此没有将其他版本网页冒称为 3.17.0 文档。采用下列官方 stable 文档、官方源码/示例，
并在已安装版本上编译场景、检查 API/枚举、验证接触参数、状态回放和步长敏感性：

- [官方接触参数、混合规则与 direct solref](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)
- [官方建模源码](https://github.com/google-deepmind/mujoco/blob/main/doc/modeling.rst)
- [关节、springref、stiffness、damping](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
- [mj_stateSize / mj_getState / mj_setState](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-getstate)
- [mj_contactForce](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce)
- [Python passive viewer 与 key_callback 示例](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)
- [官方 Python viewer 实现](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/viewer.py)

物理步长对照仅报告敏感性和穿透，不把视觉平滑或两条近似曲线称为真实物理精度证明。

## 开发验证中的修正

首次完整摩擦闭环发现视频成功后 manifest 尝试列表存在循环引用；已修正并添加降级序列的
JSON 序列化测试。初版落球使用 stiffness=160000、dt=0.001 s，测得最大穿透约 6.4 mm；
这批原始结果完整保留在 `outputs/physics_dev/bounce_initial_soft_contact/`（包括原回放 ZIP）。
为降低穿透，最终三组统一提高刚度并减小步长，记录新的完整对照；不是按趋势选择性删样本。
按钮的稳定时间先按撤离航点计算，随后改为从历史接触力识别真实释放点，重新分析导出。
