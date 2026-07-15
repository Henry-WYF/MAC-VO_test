# MAC-VO 工作上下文与交接记录

> 类型：持续维护的项目“活文档”，用于新 Agent 接手、跨对话恢复上下文和记录关键决策。
>
> 最后更新：2026-07-15。
>
> 证据标签：`[代码已核验]`、`[论文信息]`、`[实验观察]`、`[设计计划]`、`[待验证]`、`[已废弃]`。

## 0. 60 秒接手摘要

| 项目 | 当前状态 |
|---|---|
| 活动代码库 | `MAC-VO-test` |
| GitHub | `https://github.com/Henry-WYF/MAC-VO_test.git` |
| 当前分支 / HEAD | `agent-a-codex/loop-geometric-verification` / `645ae11` |
| 稳定主线 | `origin/main` / `7b65afe`；已包含本地 GlobalPGO，不是纯上游 MAC-VO |
| 当前阶段 | 阶段 B 的同源 A/B 协方差门控诊断和本地 CPU 验证已完成；正式 CUDA 完整实验待用户在服务器运行，阶段 C 尚未接入 |
| 当前唯一优先事项 | 用户在服务器 CUDA 环境复用 `07_11_011209` 缓存运行一次完整同源 A/B 序列实验 |
| 当前结论 | 阶段 B 尚未达到进入阶段 C 的条件 |
| 禁止参考 | `MAC-VO-test_loop`，它是已废弃的早期回环尝试 |

新 Agent 开始修改前必须先执行 `git status --short --branch`。审阅本文件时，工作区还存在用户所有的未提交改动（包括删除的“修改反馈.txt”以及未跟踪文档）；不得擅自恢复、覆盖或一并提交。

## 1. 原项目、论文与研究背景

### 1.1 MAC-VO 论文信息

`[论文信息]` MAC-VO 发表在 **ICRA 2025**，论文题目为：

> **MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry**

检索入口：

- 论文预印本：<https://arxiv.org/abs/2409.09479>
- 官方项目主页：<https://mac-vo.github.io/>
- 建议检索词：`MAC-VO ICRA 2025 metrics-aware covariance stereo visual odometry`

论文和官方资料用于理解作者动机、消融与算法选择；本地实现行为仍必须以当前分支的代码、配置和测试为准。

### 1.2 总体研究计划

目标不是简单“给 MAC-VO 加一个回环检测器”，而是在保持其度量感知不确定性优势的前提下形成完整闭环：

```text
MAC-VO 顺序里程计
→ 地点识别与几何验证
→ 非相邻回环相对位姿约束
→ Global Pose Graph Optimization
→ 将累计漂移分摊到整条轨迹
→ 进一步研究 covariance-aware 回环权重
```

原始计划先完成最小 GlobalPGO，因为回环检测只能发现约束，不能自行修改轨迹；第一版后端只需节点、顺序边、回环边和固定信息矩阵。基础闭环稳定后，再将 MAC-VO covariance、几何验证质量和内点数量融合为不确定性感知权重。

长期论文叙事为：**面向学习式双目视觉里程计的不确定性感知回环约束与全局后端优化**。第一阶段不引入紧耦合 IMU，以避免时间同步、外参、bias、重力初始化、预积分和滑窗优化导致研究范围失控。

原始总体计划文件：[`计划建议.txt`](C:\Users\lenovo\Desktop\code\SLAM\MAC-VO\计划建议.txt)。该文件记录最初动机；本文件记录其当前实施状态。

## 2. 代码范围与事实来源

| 路径/材料 | 定位 |
|---|---|
| `MAC-VO-main` | 原始项目目录，供上游基线对照 |
| `MAC-VO-test` | 唯一活动实现，包含 GlobalPGO 与阶段 A/B 回环扩展 |
| GitHub `main` | 当前稳定集成线，已经包含 GlobalPGO；不要等同于纯上游代码 |
| `MAC-VO-test_loop` | `[已废弃]`，不得作为设计或实现参考 |
| `ARCHITECTURE.md` | 历史线索，不是事实来源；内容必须重新由代码核验 |
| `MACVO_SYSTEM_GUIDE.md` | 辅助系统导读，仍应与当前代码交叉核验 |
| `AGENT_SYNC_GUIDE.md` | Git/GitHub 与多 Agent 协作的权威规则 |

事实优先级：当前分支源码与测试 > 本次运行保存的配置/结果 > 论文与官方资料 > 设计文档 > 历史讨论。无法确认的内容必须标为 `[待验证]`，不得写成代码事实。

## 3. 多 Agent Git/GitHub 协作（P0）

### 3.1 身份、远端与分支

三名 Agent 共用仓库 `Henry-WYF/MAC-VO_test`，通过分支名前缀区分：

| Agent | 分支前缀 |
|---|---|
| A1：电脑 A 的 Codex | `agent-a-codex/` |
| A2：电脑 A 的 ClaudeCode | `agent-a-claude/` |
| B1：电脑 B 的 Codex | `agent-b-codex/` |

当前已知提交链：

| 分支/提交 | 内容 |
|---|---|
| `main` / `7b65afe` | GlobalPGO 修复、测试和稳定主线 |
| `agent-a-codex/loop-place-recognition` / `d464bcc` | 阶段 A：ORB-BoW 地点召回 |
| `agent-a-codex/loop-geometric-verification` / `648ff73` | 阶段 B：Flow/PnP 几何验证 |
| 同上 / `645ae11` | 阶段 B 诊断日志增强，当前 HEAD |

### 3.2 每次工作的标准流程

新任务默认从最新 `main` 创建分支；若用户明确要求继续当前阶段分支，则先拉取该远端分支。任何情况下都先保护工作区：

```bash
git status --short --branch
git diff
git remote -v
git fetch origin
```

从稳定主线开始新任务：

```bash
git checkout main
git pull --ff-only origin main
git checkout -b agent-name/task-name
```

保存阶段性成果：

```bash
git status
git diff
git add <仅与当前任务相关的文件>
git commit -m "清晰、具体的任务摘要"
git push -u origin agent-name/task-name
```

同步主线变化：

```bash
git fetch origin
git checkout agent-name/task-name
git merge origin/main
```

冲突必须人工检查并运行相关测试；不得用 `reset --hard`、`checkout -- .`、`clean -fd`、强制推送或覆盖其他 Agent 改动来消除冲突。功能分支通过 PR 合入 `main`，不要直接在 `main` 开发。

交接必须包含：分支、提交、改动文件、完成内容、验证、已知问题、下一步，并同步更新本文件。完整规则见 [`AGENT_SYNC_GUIDE.md`](./AGENT_SYNC_GUIDE.md)。

### 3.3 GitHub 网络代理

用户确认电脑 A 的项目代理端口为：

```text
127.0.0.1:7898
```

执行 `fetch/pull/push` 前先检查：

```powershell
Test-NetConnection 127.0.0.1 -Port 7898
git config --show-origin --get-regexp "^(http|https)\..*proxy|^(http|https)\.proxy"
```

需要仓库级代理时：

```powershell
git config --local http.proxy  http://127.0.0.1:7898
git config --local https.proxy http://127.0.0.1:7898
git ls-remote origin
```

`[实验观察]` 2026-07-13 审阅时，当前仓库级配置仍指向 `7897`，全局 Git 配置指向 `33210`，而 `7898` 未监听。因此下一次联网前必须先确认代理程序实际端口并统一配置，不能假设文档值已经在本机生效。`127.0.0.1` 只代表当前机器；电脑 B 必须检查自己的代理，不能直接复用电脑 A 的回环地址。

## 4. MAC-VO 原始系统概览

`[代码已核验]` MAC-VO 是学习式双目视觉里程计，不是原生完整 SLAM。默认主链路为：

```text
双目帧预处理
→ FlowFormerCov 估计当前帧双目视差、相邻帧光流及其协方差
→ covariance-aware 选点、深度与 2D→3D 协方差传播
→ VisualMap 注册帧、匹配和地图点
→ TwoFrame_PGO 局部优化当前帧
→ 后处理与轨迹/张量地图输出
```

主要入口和职责：

| 位置 | 职责 |
|---|---|
| `MACVO.py` | 加载实验/数据配置、创建系统、运行、保存与评估 |
| `Odometry/MACVO.py` | 编排前端、选点、地图注册、TwoFrame 和终止流程 |
| `Module/Frontend/` | 深度、光流和学习式协方差 |
| `Module/Map/` | `VisualMap` 的帧、匹配、点和索引关系 |
| `Module/Optimization/TwoFramePGO/` | 当前帧局部几何优化 |
| `Module/Optimization/GlobalPGO/` | 本地新增的全局位姿图扩展，不属于原论文实现 |

默认 `TwoFrame_PGO` 只优化当前帧，不能消化历史帧间的非相邻回环边。项目几何使用左目 sensor pose；最终输出涉及的 body/sensor 外参转换不得提前混入回环边或 GlobalPGO。

## 5. 本地扩展、阶段计划与低侵入边界

### 5.1 GlobalPGO

`[代码已核验]` `GlobalPoseGraphOptimizer` 在 `terminate()` 阶段可选运行：节点为有效帧，顺序边来自相邻位姿，首帧固定，并预留 `add_loop_edge()`。当图中只有由同一初始轨迹生成的顺序边时，初始残差接近零，因此优化近似 no-op；它必须获得可靠回环边才可能修正漂移。

### 5.2 阶段 A–D

| 阶段 | 工作 | 主要产物 | 改变轨迹 | 状态 |
|---|---|---|---|---|
| A | ORB-BoW、低频缓存、因果历史检索 | `queries.json`、缓存帧 | 否 | 已实现并完成首轮正确性检查 |
| B | FlowFormerCov 跨时刻匹配、候选深度、PnP、几何验证 | `loop_verification.json`、`LoopConstraint` | 否 | 已实现，当前正在诊断 |
| C | 可信回环边接入 GlobalPGO、Huber、安全检查和写回 | 优化轨迹 | 是 | 未开始 |
| D | 参数/模块消融、多数据集与性能分析 | 论文表格和曲线 | 视配置 | 未开始 |

### 5.3 低侵入约束

允许：配置启停；初始化时注入 Frontend；成功跟踪且非插值帧的低频缓存；`terminate()` 离线检索与验证。

禁止：改变原始 `run_pair` 跟踪、TwoFrame 优化和地图注册语义；阶段 A/B 写回 `VisualMap.pose`；阶段 B 调用 `GlobalPGO.add_loop_edge()`。回环缓存或验证故障默认只禁用回环，原 VO 必须继续完成。

## 6. 已锁定的数据与坐标契约

- 阶段 A 的 ORB 仅用于地点识别，不参与原 MAC-VO 前端位姿计算。
- 离线 BoW 数据库从空状态按 `sensor_frame_idx` 回放；先查询历史，再插入当前帧，禁止未来帧泄漏。
- 阶段 B 调用 `estimate_pair(candidate, current)`，`match.flow` 表示 `candidate→current`；返回的 `depth_current` 不是候选帧 3D 来源。
- PnP 3D 使用 `candidate_record.depth` 与其深度协方差，2D 使用光流得到的当前帧像素。
- PnP 输出 `T_current_candidate`；未来 GlobalPGO 中 `src=candidate`、`dst=current`，边测量保存为 `inverse(T_current_candidate)`。
- 几何估计和 GlobalPGO 均使用左目 sensor pose，不混用 body pose。
- 阶段 B 只生成日志和 `LoopConstraint`，任何运行后 `VisualMap.pose` 都应保持不变。

当前研究主线继续解决 FlowFormerCov 跨时刻匹配与协方差问题。VINS 式 BRIEF/ORB 描述子匹配加 PnP 可作为未来对照基线，但暂不替代主链路。

## 7. 最新实验：`07_11_011209`

本机可读结果目录：`C:\Users\13479\Desktop\code\MAC-VO-main\MAC-VO-test_1\07_11_011209`。

使用阶段 B 诊断配置；其中 `max_flow_cov=100`、`geometry.min_points=30`、`pnp.min_inliers=20`，比原计划的后两项验收门限更宽松，不能直接视为最终配置。

`[实验观察]`

- 1274 帧全部成功跟踪，`need_interp=0`；每 10 帧缓存，共 128 个回环帧。
- 阶段 A 产生 1135 个历史候选，没有当前帧或未来帧泄漏；BoW 查询均值约 0.357 ms。
- 阶段 B 验证全部 1135 个候选，接受 21 个（1.85%），耗时约 8.9 分钟。
- 1092 个候选因 `no valid flow correspondences` 被拒绝：有限且未越界的 flow 存在，但没有点通过 `flow_cov<=100` 门限。
- 人工候选 `200→710` 是 BoW 第 1 名却在 flow covariance gate 处失败；`170→750` 进入第 4 名但 PnP RANSAC 失败；另有人工候选没有进入 top-10。
- 以“间隔≥100、GT距离≤8 m、GT姿态差≤30°”作为代理标签时，含真回环查询的 top-10 查询级召回约 33.3%；21 条通过约束中仅 4 条满足该代理条件。该标签不包含真实图像重叠，只能作为风险提示，不能直接等同最终 precision。
- 缓存约 1.21 GiB；`loop_verification.json` 含非严格 JSON 的 `Infinity`；本次元数据的 Git 版本为 `NOT_AVAILABLE`。
- `tensor_map.npz` 存在远距离地图点和极大协方差离群值；它是次级地图质量问题，与当前回环 flow gate 问题分开处理。

人工重点回环对：`168–748`、`161–779`、`162–811`、`174–701`、`200–706`。由于每 10 帧缓存，验证时应检查对应的邻近注册帧，而不是要求精确帧号全部存在。

### 7.1 2026-07-15：Phase B 同源 A/B 诊断更新

`[代码已核验]` 已实现每候选单次 Frontend 推理的 gate-on/off 配对诊断；两分支共享 flow、covariance、mask、采样点、深度、pose snapshot 和 PnP 参数，只改变 `uu/vv <= max_flow_cov` 过滤。输出包含稳定配对、漏斗与分位数、PnP 阶段状态、严格 JSON 和 pose 只读保护；旧配置默认行为不变，Phase B 不接入 GlobalPGO。离线入口可复用既有缓存与 `queries.json`，无需重跑 VO。

`[代码已核验]` Phase B、Phase A、GlobalPGO 和配置测试共 `45 passed`；全仓排除 `local/trt` 后为 `111 passed, 11 deselected`。本轮相关文件的 `pyright` 为 0 错误；全仓仍有 27 项位于未修改文件的历史或缺依赖错误。

`[实验观察]` 本机已完成 10 个自然 BoW 候选的 CPU 端到端烟雾测试：10 个候选仅调用 10 次 Frontend；7403 个入界有限点在 gate-on 下全部被过滤、gate-off 下全部恢复；gate-off 的 10 个候选均成功执行 PnP RANSAC，其中 9 个未通过内点门限、1 个未通过 VO 位姿一致性门限，最终接受 0 个。严格 JSON、A/B 配对和 pose 不变量均通过。该测试只验证实现链路并表明 covariance gate 是直接过滤阶段，不能证明 covariance 预测错误或 gate-off 约束正确。

`[待验证]` 正式 1135 候选 CUDA 配对实验由用户在服务器运行；完成后仅将最终漏斗、有限结论和必要复现信息补充到本文件。在此之前不调整 selector、PnP/位姿门限或最终 covariance 阈值，也不进入 Phase C。

## 8. 当前问题、下一步与进入阶段 C 的条件

### P0：FlowFormerCov 协方差门控

本地实现、回归测试和 CPU 烟雾测试已完成。下一步仅由用户在服务器完成 1135 候选 CUDA 同源 A/B 实验，核验配对、单次推理、pose 不变量和严格 JSON，并报告各级漏斗。完整结果写回本文档后结束本轮；此前不调整其他门限、selector 或 GlobalPGO。

### P1：召回与验证质量

阶段 A 的 top-10 仍会漏掉部分人工回环；但当前先排除 flow gate 的阻断，再统一评估 `top_k`、ORB/BoW 参数和几何通过率，避免同时改变过多变量。

### P1：可复现性与输出格式

- 将验证失败的无穷指标序列化为 `null`，保证严格 JSON。
- 每次运行保存 Git branch、commit、完整展开配置和结果目录。
- 不把缓存、模型或运行结果提交到 GitHub；只提交代码、配置、测试和小型文档。

### 阶段 C 准入条件

人工真回环能稳定产生足够 3D–2D 对应与 PnP 内点；明显错误候选被几何门控拒绝；LoopConstraint 的索引、坐标系和方向通过 synthetic residual test；阶段 B 运行前后轨迹完全不变。满足这些条件后，再制定 GlobalPGO 回环边注册、Huber、安全检查和写回方案。

## 9. 新 Agent 阅读与交接清单

推荐阅读顺序：

1. 本文件：研究目标、当前状态和阻断项。
2. [`AGENT_SYNC_GUIDE.md`](./AGENT_SYNC_GUIDE.md)：分支、提交、拉取、代理与交接规范。
3. `MACVO.py`、`Odometry/MACVO.py`：真实入口与终止调用顺序。
4. `Module/LoopClosure/`：阶段 A/B 当前实现。
5. `Module/Optimization/GlobalPGO/`：阶段 C 将使用的后端接口。
6. `Config/Experiment/MACVO/MACVO_Performant_LoopPhaseB.yaml`：当前诊断配置。
7. 最新结果目录的 `queries.json`、`loop_verification.json`、`loop_constraints.json`。

交接记录模板：

```text
Branch / commit:
Changed files:
What was done:
Evidence / tests:
Known issues:
Impact on current plan:
Next single priority:
PROJECT_CONTEXT.md updated: yes/no
```

本文件只记录已压缩的结论、证据、影响和下一步，不粘贴聊天全文。出现以下节点必须更新：阶段或路线改变；关键接口/坐标约定改变；重要提交；有效实验；新阻断项被确认或关闭；跨 Agent 交接。
