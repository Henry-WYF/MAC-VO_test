# MAC-VO 项目上下文

> 供新 Agent 快速接手。最后更新：2026-07-23。

## 1. 当前状态

| 项目 | 状态 |
|---|---|
| 仓库 / 分支 | `MAC-VO-test` / `agent-a-codex/loop-geometric-verification` |
| Phase C 基线提交 | `7ca8970` |
| 地点识别 | 默认 `custom_binary`；DBoW2 保留为可切换后端，但当前审核集未晋升 |
| 正式局部验证 | 缓存 ORB-to-ORB + ORB-SLAM 风格描述子筛选 + PnP |
| Phase B | 工程准入已通过；原 `VisualMap.pose` 保持不变 |
| Phase C | 离线比较 no-loop、fixed information、covariance information |

修改前必须执行 `git status --short --branch`。工作区可能包含用户改动，不得恢复、覆盖或一并提交。CUDA 完整实验在远程服务器 Docker 中运行。

## 2. 原 MAC-VO 与低侵入边界

原系统主链：

```text
FlowFormerCov 双目/相邻帧估计
→ covariance-aware 选点
→ 2D→3D covariance 传播
→ VisualMap
→ TwoFrame PGO
```

回环是可插拔扩展：

- 禁用回环时，不提取额外几何描述子、不写 sidecar、不执行回环验证。
- 不修改原 `KeypointSelector`、Frontend、TwoFramePGO 和 GlobalPGO 主逻辑。
- 回环缓存或验证失败只能禁用回环，原 VO 必须继续。
- Phase A/B/C 离线实验不得修改原 `VisualMap.pose`。
- 新路线优先复用缓存和现有接口，避免增加与结论无关的门限和模块。

## 3. 当前正式回环路线

```text
BoW 候选（诊断配置取 top-10）
→ 两侧缓存的独立 ORB 点
→ Hamming ≤ 50
→ best < 0.9 × second-best
→ candidate 索引唯一
→ 简化 ORB-SLAM 方向直方图
→ 带只读 VO 初值的 PnP RANSAC
→ 每个 query 从通过者中选择历史最早候选
→ PnP 内点的 disparity information observe
→ pose 副本 GlobalPGO
```

关键契约：

- BoW 与局部验证可复用同一缓存 ORB，但职责独立。
- ORB 深度采样使用 `floor(uv)` 索引；反投影与 PnP 保留浮点 `uv`。
- PnP 使用 `candidate 3D → current 2D`，输出 `T_current_candidate`；回环边保存逆方向 `T_candidate_current`。
- MAC-VO NED 相机前向轴为 `x`，OpenCV 相机前向轴为 `z`。
- 当前参数：100 次、10 px、confidence 0.99、正深度内点至少 25。
- 绝对旋转上限已放宽到 180°，不再阻断正常大角度真实回环；VO 初值差只记录。
- 不采用 mutual、内点率、空间覆盖、Essential、旧 pair-risk/calibration gate。

旧 fixed-point 局部验证与旧 Phase B.5 已完成 CUDA observe，但未晋升；仅保留用于复现和消融，不再作为正式主路线。

## 4. Information 契约

每个 PnP 内点使用 `[u,v,disp]` 残差：

```text
Sigma_r = Sigma_current
        + J_X R Sigma_candidate_3D R^T J_X^T
```

- ORB 像素 covariance 使用 `match_cov_default`；disparity 使用同帧深度方差换算。
- current depth covariance 不重复计权；Hamming 距离不作为 covariance。
- `Sigma_r` 对称化后严格 Cholesky，不使用 jitter。
- 白化 `3×6` Jacobian 堆叠后必须满秩。
- raw Hessian 在 `T_current_candidate` 上计算，再以 Adjoint 转到最终边方向。
- covariance information 通过广义特征值缩放，只允许相对 fixed information 降权。
- information 无效边同时退出 fixed/covariance pose-copy PGO，保证两分支边集完全一致。

第一轮 information 为 observe-only，不写入在线回环边。

## 5. 已冻结实验结论

- 固定 flow covariance 门限 `uu/vv<=100` 是旧 Flow PnP 覆盖率的直接阻断项；固定补点会增加大误差约束，路线已淘汰。
- DBoW2 在当前 OpenCV ORB 与 abf001 审核集上 Recall@10 低于 custom，因此未晋升；不代表 DBoW2 本身无效。
- ORB-to-ORB 严格匹配在 `1200→1100` 恢复了 25 个 PnP 内点并形成有效 information。
- 提交 `7ca8970` 的 Phase B 结果：5 对几何通过、5 对 information 有效、最终选择 4 条回环边。
- 4 条最终边的 GT 位姿代理均为 `accurate`，`large-error=0`；fixed/covariance pose-copy PGO 均 `safe=true`，Phase B 工程准入通过。
- 当前 4 条边均约为 100–110 帧跨度，`long_span_accurate=0`。这只是工程 smoke test，不是论文级稳定性证据。

## 6. Phase C 当前任务

离线、只读比较：

```text
no-loop：原轨迹，不注册 loop edge、不运行 PGO
fixed：原轨迹 + odometry edges + 4 条 fixed-information loop edges
covariance：相同轨迹、相同 edges，仅替换为 covariance information
```

要求：

- 三方边集、相对位姿和索引严格一致；fixed/covariance 只允许 information 不同。
- no-loop 的回环边只用于 residual 诊断，不进入优化图。
- 正式 ATE/RPE 复用 `Evaluation/MetricsSeq.py` 的 evo 全局对齐口径。
- 同时报最大位姿修正、相邻变形、loop residual，以及各自优化前后 loss。
- 本轮不采用 Huber，不重新运行 VO，不修改 GlobalPGO，不在线写回 pose。
- 只有安全检查通过、ATE/RPE 均不恶化且 loop residual 显著下降的分支才标为 `positive`；其余为 `inconclusive` 或 `failed`。

主要入口：`Scripts/AdHoc/RunLoopPhaseCOffline.py`。

## 7. 测试与 GT 口径

只使用 `pytest` 与服务器离线/完整 VO 测试。

- GT translation error：`||t_est - t_gt||`。
- GT rotation error：SO(3) geodesic。
- accurate：平移 `<1 m` 且旋转 `<5°`。
- large-error：平移 `>3 m` 或旋转 `>15°`。
- Phase B 工程准入：GT 可评价边≥3、accurate≥2、不同 query≥2、GT 覆盖率100%、large-error=0，并且 pose-copy PGO 安全。

主要代码入口：`Odometry/MACVO.py`、`Module/LoopClosure/Manager.py`、`Module/LoopClosure/VINSGeometry.py`。禁止参考已废弃的 `MAC-VO-test_loop`。
