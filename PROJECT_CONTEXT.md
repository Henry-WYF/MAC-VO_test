# MAC-VO 工作上下文与交接记录

> 类型：持续维护的项目活文档，用于新 Agent 快速接手和记录已冻结决策。
>
> 最后更新：2026-07-21。
>
> 证据标签：`[代码已核验]`、`[论文信息]`、`[实验观察]`、`[设计计划]`、`[待验证]`、`[已废弃]`。

## 0. 60 秒接手摘要

| 项目 | 当前状态 |
|---|---|
| 活动代码库 | `MAC-VO-test` |
| GitHub | `https://github.com/Henry-WYF/MAC-VO_test.git` |
| 当前分支 / HEAD | `agent-a-codex/loop-geometric-verification` / `00ac6b2` |
| 稳定主线 | `origin/main` / `7b65afe`；已含本地 GlobalPGO，并非纯上游 MAC-VO |
| 当前阶段 | Phase A/B 已实现；Phase B.5 只读实现已落地，待服务器 pytest 与离线 observe/calibration/evaluation 验证 |
| 当前后端 | 地点检索继续使用 `custom_binary`；DBoW2 仅保留为消融后端 |
| 当前唯一优先事项 | 完成不确定性感知的回环帧对 veto、点选择、`PnP → reprojection+disparity` 验证和 6×6 information 诊断 |
| 当前边界 | Phase B.5 不修改 VO 轨迹、不接入 GlobalPGO；通过准入条件后才实施 Phase C |
| 禁止参考 | `MAC-VO-test_loop`，它是已废弃的早期回环尝试 |

新 Agent 修改前必须先执行 `git status --short --branch`。当前工作区含用户所有的未提交改动；不得擅自恢复、覆盖或一并提交。

## 1. 研究目标与事实来源

`[论文信息]` 上游项目论文为 **MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry**（ICRA 2025）。

- 预印本：<https://arxiv.org/abs/2409.09479>
- 官方主页：<https://mac-vo.github.io/>

长期目标不是简单添加回环检测器，而是构建：

```text
MAC-VO 顺序里程计
→ 地点识别
→ 不确定性感知的跨时刻几何验证
→ 带 6×6 information 的非相邻回环边
→ robust GlobalPGO
→ 将累计漂移分摊到整条轨迹
```

论文叙事候选为：**面向学习式双目视觉里程计的不确定性感知回环约束与全局优化**。本项目暂不引入紧耦合 IMU。

事实优先级：当前分支源码与测试 > 本次运行保存的配置/结果 > 论文与官方资料 > 设计文档 > 历史讨论。无法确认的内容必须标为 `[待验证]`。

## 2. 原系统与本地扩展

`[代码已核验]` MAC-VO 是学习式双目 VO，不是原生完整 SLAM。默认链路为：

```text
双目预处理
→ FlowFormerCov 输出视差、相邻帧光流及 covariance
→ covariance-aware 选点和 2D→3D covariance 传播
→ VisualMap 注册
→ TwoFrame_PGO 优化当前帧
→ 后处理和轨迹输出
```

| 位置 | 职责 |
|---|---|
| `MACVO.py` | 配置、运行、保存与评估入口 |
| `Odometry/MACVO.py` | 前端、选点、地图、TwoFrame 和终止流程编排 |
| `Module/Frontend/` | 深度、光流和学习式 covariance |
| `Module/Covariance/Project2to3.py` | 像素/视差不确定性向 3D 点 covariance 传播 |
| `Module/Optimization/TwoFramePGO/` | `icp/reproj/disp` 局部位姿优化；默认使用 `disp` |
| `Module/LoopClosure/` | 本地 Phase A/B 回环扩展 |
| `Module/Optimization/GlobalPGO/` | 本地全局位姿图扩展，不属于原论文实现 |

`TwoFrame_PGO` 默认只优化当前帧，不能消化历史非相邻回环。GlobalPGO 已具备节点、顺序边和 `add_loop_edge()` 接口；没有可靠回环边时基本是 no-op。

## 3. 阶段、状态与低侵入边界

| 阶段 | 工作 | 改变轨迹 | 状态 |
|---|---|---:|---|
| A | ORB-BoW 缓存与因果历史检索 | 否 | 已实现；`custom_binary` 保留，DBoW2 未晋升 |
| B | FlowFormerCov 跨时刻匹配、PnP、诊断与 `LoopConstraint` | 否 | 已实现 |
| B.5 | 自适应帧对 veto、点选择、`reproj+disp` 精化、6×6 information 诊断 | 否 | 当前唯一开发阶段 |
| C | 回环边接入 GlobalPGO、Huber、安全检查和轨迹写回 | 是 | 未开始 |
| D | 多数据集、消融、性能和论文评估 | 视配置 | 未开始 |

低侵入原则：

- 允许配置启停、低频缓存、`terminate()` 离线检索/验证和独立离线重放。
- 不改变原 `run_pair`、TwoFrame 优化和地图注册语义。
- Phase A/B/B.5 不写回 `VisualMap.pose`，不调用 `GlobalPGO.add_loop_edge()`。
- 回环模块失败只禁用回环；原 VO 必须继续完成。
- CUDA 完整实验在远程服务器 Docker 中运行；本地负责代码、CPU/可运行 pytest 和结果分析。

## 4. 已锁定的坐标与接口契约

- ORB 仅用于地点识别，不参与原 MAC-VO 前端位姿计算。
- BoW 离线数据库按 `sensor_frame_idx` 从空状态回放；先查历史、后插当前，禁止未来帧泄漏。
- Phase B 调用 `estimate_pair(candidate, current)`；`match.flow` 表示 `candidate→current`。
- PnP 的 3D 来自 `candidate_record.depth`，2D 来自 flow 得到的当前帧像素。
- PnP 输出 `T_current_candidate`。GlobalPGO 约定 `src=candidate`、`dst=current`，边测量保存为 `inverse(T_current_candidate)`。
- 几何估计和 GlobalPGO 使用左目 sensor pose，不混用 body pose。
- 每个候选的诊断索引必须始终引用公共 `candidate_uv` 原始采样索引。
- Phase B/B.5 前后 `VisualMap.frames.data["pose"]` 必须逐位不变。

## 5. 已冻结实验结论

### 5.1 Phase B covariance gate

`[实验观察]` `07_15_043519` 全序列包含 128 个缓存帧、1135 个自然候选。固定门限 `uu/vv<=100` 在 597380 个入界有限点中仅保留 3992 点（约 0.67%），28 个候选到达 PnP、接受 21 条；关闭门限后 1126 个候选到达 PnP、接受 28 条。

结论：固定 `100` 是 PnP 覆盖率的直接阻断项，但 covariance 排序仍有质量作用；不能永久关闭门限。

### 5.2 强制补点路线已淘汰

`[实验观察]` fixed/null、adaptive 50/100/200 同源实验分别接受 `21/48/43/32` 条。三种补点档均提高 PnP 覆盖，同时增加大误差约束；补点 cutoff 进入很高 risk 区间。

`[已废弃]` 不再以“达到固定点数”为主要目标，不继续为 50/100/200 调参，也不将 gate-off 作为最终方案。保留已有实现仅供回归和消融。

已知案例：

- `1200↔1100` 是真实回环，但 PnP 位姿误差较大，说明“真实重叠”和“相对位姿估计正确”必须分开评价。
- `1250↔470/480` 是重复纹理误检；帧对整体 covariance 很高，但少量低 covariance 点仍能形成 PnP 内点，说明只做点级筛选不足。
- 同一序列诊断中，帧对整体 uncertainty veto 能拒绝部分已知假回环，但真假分布仍有重叠；不能把单一固定分位数阈值直接设为通用默认值。

### 5.3 DBoW2 冻结结论

`[代码已核验]` 提交 `00ac6b2` 增加可选 DBoW2/ORBvoc 后端与 Docker ABI gate；默认仍为 `custom_binary`。服务器 gate 成功。

`[实验观察]` 在当前 OpenCV ORB 描述子与 abf001 冻结审核集上，custom/DBoW2 的 Query Recall@10 分别为 `0.9211/0.8026`；DBoW2 虽降低 false candidates/query，但未保持 recall，故不晋升、不冻结阈值、不进入 MH05。该结论不能外推为 DBoW2 本身无效；论文发布前 AI 盲审标签仍需人工复核。

## 6. 当前冻结技术路线：Phase B.5

```text
custom_binary BoW 候选
├─ ORB 局部匹配 + ORB PnP：传统几何基线/候选 gate 假设
└─ 每候选单次 FlowFormerCov 推理
   → 帧对整体 uncertainty gate
   → q-NMS + population 专属 risk cap 点选择
   → Flow PnP
   → observe-only reprojection+disparity 精化与 6×6 information
```

`[设计计划]` ORB 与 Flow 必须先对全部 BoW 候选并行 observe、独立晋升；ORB 失败不得阻断 Flow 假设验证。只有 ORB 与其对应 Flow population 均晋升后才形成串联 cascade。`all_bow_candidates` 与 `orb_supported_candidates` 共享同一时间 calibration prefix，但门限、point cap、manifest 和输出物理隔离。固定控制样本始终运行影子 Flow，不进入正式约束。

实施顺序冻结为：B5a 并行数据 → B5b prefix 无标签校准 → B5c post-prefix 冻结评价 → B5d selector shadow A/B → B5e 精化/information observe。任何 `apply` 都要求开发集与冻结集通过预注册指标和可信 manifest。

`[待验证]` 当前工作区已实现上述 B5a–B5e、严格分支输出、两阶段 manifest 晋升及离线评价入口；尚未在服务器 CUDA/Docker 环境执行，因此不得将其表述为实验通过或 Phase C 已准入。

### 6.1 帧对级自适应 uncertainty veto

- 母集为现有最多 800 个候选深度采样点中入界且 flow 有限的点；不使用 covariance gate 或 PnP 内点后的分布。
- `Σn=diag(1/W,1/H)Σdiag(1/W,1/H)`；pair risk 为对称化 `Σn` 的 `λmax`，正式第一版仅使用 `median(log(max(risk,1e-12)))`。
- 最早 20% 有效查询确定唯一 `calibration_end_sensor_frame_idx`，prefix 不进入正式评价且不得因 population 样本不足向后扩展。
- 门限为 prefix 内 pair-risk p50 的 `median + 2×1.4826×MAD`，并要求有效点≥30、有效率≥0.8、8×8覆盖≥8以及开发集冻结的绝对 calibration sanity cap。
- calibration 不读标签；post-prefix 冻结审核集才评价。样本不足或开发/冻结集任一未通过时保持 observe。

### 6.2 点级选择

- pair 统计点与 dense NMS 点是两个独立集合；NMS 点原始索引为 `v×W+u`。
- 点级 score 使用前端原始 `q=uu+vv-2uv`；pair risk 仍使用归一化 `λmax`，二者不得混用。
- 非法 covariance 在 NMS 前置为 `+inf`；公共 helper 在原前端不传 mask，保持历史行为。
- point cap 为对应 population calibration NMS risk 的 Q95；不强制补点，按 8×8 网格稳定选择。只有同次推理的 legacy/shadow A/B 通过预注册召回、误检、GT 位姿代理和控制样本条件后才能 apply。

### 6.3 几何残差与双深度

- PnP 继续作为稳健初值和外点剔除器。
- 主精化路线采用原 MAC-VO `disp` 思想：每点残差为当前帧二维重投影误差 `[u,v]` 加当前帧视差误差 `[disp]`。
- 当前帧 depth/disparity 因而参与验证，可检查 PnP 是否得到度量深度支持；这比第一版新增独立 3D–3D RANSAC 更低侵入。
- 配对 3D–3D 刚体配准只作为可选离线诊断。若与 `reproj+disp` 无互补证据，不进入最终在线链路。

### 6.4 covariance 与 6×6 information

- 候选 3D 点 covariance 是候选像素与视差/深度不确定性通过相机模型传播得到的 3×3 covariance。
- 当前观测包含 flow 的 2×2 covariance 与当前 disparity covariance。
- 严格的 `[u,v,disp]` 残差 covariance 还须包含候选 3D 点 covariance 经投影 Jacobian 的贡献；把候选 3D 点当作精确常量只可作为消融近似。
- 局部相对位姿以 `SE(3)` 为优化量；从加权正规方程获得 `Λ_loop≈Σ JᵀΣ_r⁻¹J`，作为未来回环边的 6×6 information。
- 稠密 flow 点高度相关，原始 Hessian 不能直接视为已标定 information。必须记录有效样本数、特征值、秩和条件数，并研究空间降采样/相关性修正、总体尺度标定和特征值上下限。
- 位姿或边方向求逆时，covariance/information 必须按 `SE(3)` Adjoint 变换，不能原样复制。

## 7. Phase C 准入与后端原则

进入 Phase C 前必须满足：

1. 至少一个 Flow population 的 pair gate 在开发集和未参与调参的冻结集通过量化晋升条件；ORB 仅独立晋升后才可组成 cascade。
2. 对应 q-NMS/risk-cap selector 通过同源 shadow A/B；真实回环 PnP 不下降，reviewed false 和 GT large-error 约束不增加。
3. `1200↔1100` 与 `1250↔470/480` 等控制案例有可解释结果，但不得替代全体统计。
4. 6×6 information 对称、半正定、方向正确、数值可观且尺度经过离线标定；退化时能拒绝或降权。
5. `LoopConstraint` 索引、坐标系和方向通过 synthetic residual test。
6. Phase B.5 前后 pose 不变量、严格 JSON、分支隔离、单候选单次 Frontend 和可信 manifest 校验全部通过。

Phase C 才允许：

- 将主分支 `LoopConstraint` 注册进 GlobalPGO；
- 对白化/马氏 SE(3) 残差使用 Huber；
- 添加优化前后安全检查和轨迹写回。

information 与 Huber 不冲突：前者描述正常条件下六个方向的统计置信度，后者限制异常大残差边的影响。Huber 不能替代 Phase B.5 的错误回环拒绝；若仍存在低残差假回环，再评估 switchable constraints/DCS。

## 8. 测试、输出与运行约束

- 主要测试方式保持为 `pytest` 与服务器离线/完整 VO；不新增独立测试框架。
- 验证失败的未计算指标写 `null`，所有 JSON 使用严格序列化。
- 每次运行保存 branch、commit、展开配置、代码/输入摘要和独立结果目录。
- 缓存、模型和运行结果不提交 GitHub；只提交代码、配置、测试和小型文档。
- 正式完整 VO 只在 Phase B.5 准入通过且准备验证 Phase C 时运行。

## 9. Git 与新 Agent 阅读清单

协作规则见 [`AGENT_SYNC_GUIDE.md`](./AGENT_SYNC_GUIDE.md)。不得使用 `reset --hard`、`checkout -- .`、`clean -fd`、强制推送或覆盖其他 Agent 改动。只暂存和提交当前任务相关文件。

推荐阅读顺序：

1. 本文件。
2. `Module/LoopClosure/Verification.py` 与 `Manager.py`。
3. `Module/Covariance/Project2to3.py`。
4. `Module/Optimization/TwoFramePGO/Graphs.py` 与 `Optimizer.py`。
5. `Module/Optimization/GlobalPGO/`。
6. `Config/Experiment/MACVO/MACVO_Performant_LoopPhaseB.yaml`。
7. `07_15_043519` 的 `queries.json`、`loop_verification.json` 与各档对比结果。

交接必须记录：branch/commit、改动文件、完成内容、验证、已知问题、对当前路线的影响、下一项唯一优先事项，以及本文件是否更新。
