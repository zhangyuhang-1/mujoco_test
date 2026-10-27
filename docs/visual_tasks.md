# 四项视觉物理推理任务（本次仅视频交付）

在项目目录使用现有环境运行：

```bash
conda activate /data_lewm/envs/mujoco_test
python scripts/experiments.py run visual --no-package --resume
# 单项；固定场景；只计算并记录、不导视频
python scripts/experiments.py run 08_balance --suite fixed --no-package
python scripts/experiments.py run 09_support --quick --no-package
python scripts/experiments.py run 10_hidden --no-package
python scripts/experiments.py run 11_pour --no-package
```

顺序为修正偏心放置、移除支撑、观察隐藏配重、双臂倾倒。旧 `run all`、`run tasks` 与堆叠 demo 入口保持原义。新任务自动禁止构建回放包/ZIP，无需执行 pack 或 bundle。结果集中在 `deliverables/vision_tasks/`；每项 `videos/overview.mp4` 和 `detail.mp4` 是下载观看入口，不需要本地安装 MuJoCo。

`--suite fixed` 只跑基础条件，默认 `all` 追加预先声明的小变体。参数定义在 `src/dualstack/tasks/visual_config.py` 的 `visual_cases`，不挑选随机成功种子。这次是少量确定性场景验证，不能宣称完成十种子泛化评测。每参数组独立进程、默认预算 120 秒；每项独立进程、默认预算 3600 秒；单次渲染预算 900 秒，可用 `--group-budget/--render-budget/--experiment-budget` 调整。

## 输入、动作与 oracle 边界

`TaskEnv.observe()` 对这四项严格返回 `images` 与 `instruction`。`--images` 开启原始 256×256 RGB 相机 `vision_overview`；不传关节状态、物体位姿、时间、质量、重心、力或力矩。指令不含隐藏配重方向。专家评测默认关闭未使用的相机绘制以节省资源，`images` 此时为空；每项另导出 `model_input_examples/`，包含真实记录状态下无叠字的相机图像及恰好两字段的示例输入。

所有本次脚本均明确为 **oracle**：`visual_policies.py` 通过独立 `env.oracle.state()` 读取真值，不宣称纯视觉策略。底层沿用 Mink 差分 IK 和关节位置伺服，反馈只在控制器内部。未来视觉策略使用相同 `reset/act/close` 接口，`act` 仅接收普通观测，例如：

```bash
python scripts/experiments.py run 10_hidden --suite fixed --images \
  --policy your_module:YourVisualPolicy --no-package --group-budget 600
```

动作仍为16维，两臂依次世界 xyz、wxyz 四元数、夹爪开合，50 Hz；物理默认 0.002 秒。不要将 `step` 返回的评测 `info`、离线 CSV、带参数叠字的演示视频或 oracle 接口输入视觉模型。人类查看的视频允许显示对照参数。

## 场景与真实动力学

统一场景入口 `visual_scenes.py`，脚本 `visual_policies.py`，独立成功检测 `visual_eval.py`。继续使用原任务记录、绘图、子进程隔离、离线渲染和降级系统；没有另建重复的导出框架。

| ID | 几何/真实质量 | 策略与对照 | 独立验收 |
|---|---|---|---|
| 08_balance | 长条32×3.6×3.6 cm、0.12 kg；可见配重0.20 kg，x=±10 cm；总质量0.32 kg，真实 COM x=±6.25 cm；平台宽2.4 cm、顶部0.90 m | 左/右配重各做几何中心对齐及 COM 对齐；追加 yaw=0.04 rad、初始 x+8 mm | 自由平台接触、COM 支撑余量>3 mm、低速度/倾角、夹爪张开且撤离、排除桌面额外支撑，保持0.6 s。错误放置必须真实倾倒并继续记录1.2 s |
| 09_support | 横梁44 cm、0.14 kg；支撑间距32 cm；载荷0.05 kg；目标红色支撑及绿色把手为同一自由刚体 | 正确顺序右臂先真实夹持横梁，左臂移走支撑、近 COM 换抓并放入绿色区；负对照直接移除并抬离；载荷偏移9→7.5 cm变体 | 目标含把手完整进入区域、释放撤离；横梁/载荷保持高度及低速；真实备用抓持，保持0.6 s。允许右臂继续托举作为终态 |
| 10_hidden | 与08同长；配重完全藏于相同外壳中，外观与指令不泄漏左右；平台宽3 cm、顶部0.96 m | 先在中央有限位铰链夹具上自然倾斜，再真实抓持、释放夹具、提取、调平、按 COM 放置；左右各一例 | 观察到>0.12 rad倾斜并低角速度保持0.3 s、抓持后夹具释放、自由平台接触、释放撤离并稳定0.6 s |
| 11_pour | 28 cm浅盒、低围边；半径1.8 cm自由球；+Y壁唯一黄色出口宽7 cm，中心x=7.5 cm；接收器15×13 cm | 沿用Y两侧把手真实夹持；双臂差动高度，倾角速度0.055 rad/s、上限0.24 rad；侧向球反馈作小角度校正；接收器x横移1.5 cm变体 | 球实际跨过规定出口，接触接收器底、完全在有效内部且低速0.6 s；球半径及墙厚参与内部区域检测（10 μm数值接触容差）；同时盒体受控、两臂真实抓持 |

所有刚性质量分布与惯量由 MuJoCo 根据多几何体质量合成，编译后的 COM、主惯量与惯量坐标系写入每组 `metrics.json`。08的几何中心对齐时 COM 比平台边界外偏约5.05 cm；这只是设计检查，最终结论来自仿真中真实倾倒。平台顶部比桌面高15 cm，正确释放时不依靠桌面或夹爪支撑。

隐藏配重观察装置只有一条明确命名 `observation_fixture` 的 sample→fixture_pivot weld，表示物体安装于真实重力驱动铰链夹具，不连接机器人。确认双侧夹指真实接触后，仅将 `data.eq_active` 置零表示夹具解锁；不改 qpos/qvel。释放时跳变量写入指标。铰链限位为软约束，运动峰值可略超过名义±0.18 rad。所有搬运由接触和执行器完成。初始化、记录恢复之外不逐帧写物体位姿，也不设置球出射速度。

## 输出、重渲染与验证

每项保留 `runs/<case>/config.json, scene.xml, data.csv, metrics.json, events.json, contact_history.npz, trajectory.npz`，以及曲线 `curves.png`、报告、清单。状态采用 mjSTATE_INTEGRATION（含运行时 eq_active、控制、外力等）；记录每个0.002秒物理步，接触历史独立保存。录像只恢复状态并 mj_forward，不推进物理；不能由恢复位置声称精确重算历史力。

```bash
# 仅从已保存轨迹重新导出原生1080p，保留物理时间
python scripts/experiments.py render \
  --package deliverables/vision_tasks/11_pour/render_source \
  --output deliverables/vision_tasks/11_pour/videos/detail.mp4 \
  --view detail --width 1920 --height 1080 --fps 30
# 原始无叠字模型输入示例（不提供真值数据）
python scripts/experiments.py _observations \
  --package deliverables/vision_tasks/10_hidden/recording_source \
  --output deliverables/vision_tasks/10_hidden/model_input_examples
# 最终完整数据、状态重新加载与视频全帧解码检查
python scripts/experiments.py audit deliverables/vision_tasks
```

视频均原生目标1920×1080、30 fps、4倍离屏抗锯齿；每组开始暂停0.3秒、结束暂停0.5秒。独立电影相机不修改模型观测相机。失败按既有1920×1080→1280×720→640×360→320×180顺序重建上下文渲染，仍不能编码则尝试PNG序列；每轮错误与实际分辨率写入 manifest。物理完成与视频完成分开判断。数值步长0.002/0.001秒对照保持同一控制周期和初始条件，记录敏感性，不据此宣称真实物理精度。

服务器没有 DISPLAY，不启动桌面查看器。既有 OSMesa 离屏渲染可用；没有修改全局环境或安装管理员软件。当前 Python/MuJoCo/依赖与 GPU/硬盘检查见根 `manifest.json` 的 environment；数值、渲染、叠字及编码分别计时。

下载 Windows 示例（本地 PowerShell，将用户名/服务器/目录替换为实际值）：

```powershell
scp -r haichao@SERVER:/home/haichao/zhangyuhang/mujoco_test/deliverables/vision_tasks ./vision_tasks
```

文件较多时可仅下载四项 `videos` 和 `REPORT.md`、`curves.png`，直接播放 MP4。本次不生成压缩包或本地三维回放交付物。

## 官方依据与限制

当前实际安装 MuJoCo 3.17.0；对应版本网页访问失败，查阅官方 stable 页面，并用已安装 API 和编译场景核对：

- [刚体质量/惯量](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-inertial)：多几何体合成质量及惯量，不用仅视觉上的配重。
- [接触与摩擦混合](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)：保留正常接触、滑动/扭转/滚动摩擦；明确pair参数。
- [weld 夹具语义](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld) 与 [mjData](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata)：eq_active 可运行时启停，此处仅用于有记录的观察夹具释放。
- [状态保存](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-getstate) 与 [mj_forward](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-forward)：保存含夹具状态的完整积分状态，离线画面不重跑控制或物理。
- [官方 Python renderer 示例](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/renderer.py)：离屏缓冲和原生渲染尺寸。

隐藏质量推断、支撑替代顺序及双臂反馈以后适合替换为视觉模型；本次提供的是 oracle 上限与物理可行性验证。软接触/有限限位、保守固定姿态和少量变体仍是局限，不代表任意物体、相机或质量分布均可泛化。
