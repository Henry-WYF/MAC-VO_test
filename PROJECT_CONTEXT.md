# MAC-VO 项目上下文

> 供新 Agent 快速接手。最后更新：2026-07-30。

## 1. 当前状态

| 项目 | 状态 |
|---|---|
| 仓库 / 分支 | `MAC-VO-test` / `agent-a-codex/loop-geometric-verification` |
| 已同步基线 | `1972da7`（testb 最新 DBoW2/通用 Phase C） |
| 地点识别 | 默认 `dbow2_orb` + `ORBvoc.txt`；`custom_binary` 仅作消融 |
| 正式局部验证 | ORB-to-ORB + ORB-SLAM 风格筛选 + PnP + 最终候选 Flow LM 精化 |
| 回环缓存 | 实验配置每 5 个 sensor frame 缓存一次 |
| 全局优化 | CPU 稀疏 LM；fixed 与 mixed covariance/fixed information 对比 |
| 当前状态 | 新整合路线已实现，等待服务器 Docker CUDA/离线验证 |

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
- 不修改原 `KeypointSelector`、Frontend 和 TwoFramePGO 主逻辑；GlobalPGO
  保留原 LBFGS 默认路径，新增可选 `sparse_lm`。
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
→ 只对最终候选执行一次 historical→current FlowFormer 推理
→ PnP 正深度内点位置采样 flow/covariance 与 current disparity
→ Huber 加权 `[u,v,disp]` 两帧 LM 精化
→ 最终鲁棒 Hessian 生成回环边 information
→ pose 副本稀疏 Global LM
```

关键契约：

- BoW 与局部验证可复用同一缓存 ORB，但职责独立。
- ORB 深度采样使用 `floor(uv)` 索引；反投影与 PnP 保留浮点 `uv`。
- PnP 使用 `candidate 3D → current 2D`，输出 `T_current_candidate`；回环边保存逆方向 `T_candidate_current`。
- MAC-VO NED 相机前向轴为 `x`，OpenCV 相机前向轴为 `z`。
- 当前参数：100 次、10 px、confidence 0.99、正深度内点至少 25。
- 绝对旋转上限已放宽到 180°，不再阻断正常大角度真实回环；VO 初值差只记录。
- 不采用 mutual、内点率、空间覆盖、Essential、旧 pair-risk/calibration gate。
- Flow 仅在 ORB/PnP 最终候选上运行一次；失败则拒绝该边，不回退 PnP 位姿。
- `match.mask=None` 表示无附加 mask；稠密图仅用 `floor(uv)` 索引，几何保留浮点坐标。
- 记录 Flow 与 ORB 对同一 current 像素预测差的 p50/p90，只诊断、不设门限。

旧 fixed-point 局部验证与旧 Phase B.5 已完成 CUDA observe，但未晋升；仅保留用于复现和消融，不再作为正式主路线。

## 4. 精化与 Information 契约

每个 PnP 内点使用 `[u,v,disp]` 残差：

```text
Sigma_r = Sigma_current
        + J_X R Sigma_candidate_3D R^T J_X^T
```

- 回环 current `[u,v]` covariance 使用本次长跨度 FlowFormer 的
  `uu,vv,uv`；current disparity/variance 也来自本次 `estimate_pair`。
- candidate 3D/covariance 始终来自 historical 缓存深度及其 covariance，
  不误用本次返回的 current depth。
- current depth covariance 不重复计权；Hamming 距离不作为 covariance。
- `Sigma_r` 对称化后严格 Cholesky，不使用 jitter。
- 局部 Huber 使用白化残差范数，`delta=2.795`；最终 information 为终点
  重新线性化的鲁棒 Hessian，不含 LM 阻尼，也不对 Huber 权重求导。
- 非有限/非 SPD residual covariance 与精化后非正深度观测按点剔除；
  剩余点不足或整体 Jacobian 不满秩时才拒绝整条边。
- 精化后重新执行与 PnP 相同的平移/完整旋转安全门，并记录
  PnP→精化位姿变化。
- 白化 `3×6` Jacobian 堆叠后必须满秩。
- raw Hessian 在 `T_current_candidate` 上计算，再以 Adjoint 转到最终边方向。
- 回环 raw Hessian 不再做旧广义特征值上限缩放；逐边输出点数、trace、
  `trace/point_count`、秩、条件数与特征值。

顺序边：

- 仅当 `match2frame1=src && match2frame2=dst` 时使用对应 MatchObs 构造同一
  `[u,v,disp]` 鲁棒观测 Hessian。
- 缺少直接观测、数值非法或退化时显式回退 `100I`，不删除边，保证图连通。
- 因而该实验分支必须称为 `mixed covariance/fixed information`，并报告
  observation-Hessian 边数、fallback 数量、比例和原因。

## 5. 已冻结实验结论

- 固定 flow covariance 门限 `uu/vv<=100` 是旧 Flow PnP 覆盖率的直接阻断项；固定补点会增加大误差约束，路线已淘汰。
- DBoW2 在早期 abf001 人工审核集上 Recall@10 低于 custom，但该标签未经最终
  确认；为提高跨数据集泛化性，当前正式默认已改为标准 ORBvoc/DBoW2。
- ORB-to-ORB 严格匹配在 `1200→1100` 恢复了 25 个 PnP 内点并形成有效 information。
- 提交 `7ca8970` 的 Phase B 结果：5 对几何通过、5 对 information 有效、最终选择 4 条回环边。
- 4 条最终边的 GT 位姿代理均为 `accurate`，`large-error=0`；fixed/covariance pose-copy PGO 均 `safe=true`，Phase B 工程准入通过。
- 当前 4 条边均约为 100–110 帧跨度，`long_span_accurate=0`。这只是工程 smoke test，不是论文级稳定性证据。
- K09 stride-5/DBoW2（提交 `2d66a27`）完成 1591 帧，319 个缓存帧；
  5 次最终候选网络精化全部成功并形成 5 条可比较回环边。
- 该轮 Phase C：no-loop ATE RMSE `9.91183`，fixed `2.46161`，
  mixed `2.31736`；两种 PGO 均安全，但 RPE 轻微上升，均归为
  `inconclusive`。
- 该轮顺序边 observation-Hessian 因离线 map loader 未恢复 MatchObs/Point/edge
  而 100% 回退 `100I`；因此 mixed 结果尚不能解释为顺序边 covariance
  生效。加载器已修复，需复用同一完整 VO 结果重跑 Phase B/C。

## 6. 当前全局优化与实验任务

CPU 稀疏 LM：

```text
首帧固定
→ 对 src/dst 左扰动做中心差分 Jacobian
→ 稀疏 J、H=JᵀJ
→ H + λ·diag(max(diag(H), eps))
→ scipy.sparse.linalg.spsolve
→ 仅接受鲁棒损失下降的步
```

- loop edge 使用 Huber `delta=3.548`；odometry observation Hessian 已在局部
  观测层使用 Huber，不再在位姿残差层重复加核。
- 非有限解、秩警告、无下降步均拒绝；失败输出原轨迹 fallback，禁止晋升。
- 原 LBFGS 路径仍是缺少 `solver` 字段时的兼容默认值。

离线比较：

```text
no-loop：原轨迹，不注册 loop edge
fixed：odometry 100I + loop 100I
mixed covariance/fixed：direct odometry observation Hessian
                        + missing-edge 100I fallback
                        + refined loop raw Hessian
```

- fixed/mixed 使用完全相同的精化回环边集，只允许 information 不同。
- no-loop 的回环边只用于 residual 诊断，不进入优化图。
- 正式 ATE/RPE 复用 `Evaluation/MetricsSeq.py` 的 evo 全局对齐口径。
- 同时报最大位姿修正、相邻变形、loop residual、优化前后鲁棒 loss 和
  顺序边 information fallback 覆盖率。
- Phase B/C 离线仍不得修改原 `VisualMap.pose`。

主要入口：`Scripts/AdHoc/RunLoopPhaseBOffline.py` 与
`Scripts/AdHoc/RunLoopPhaseCOffline.py`。

## 7. 测试与 GT 口径

只使用 `pytest` 与服务器离线/完整 VO 测试。

- GT translation error：`||t_est - t_gt||`。
- GT rotation error：SO(3) geodesic。
- accurate：平移 `<1 m` 且旋转 `<5°`。
- large-error：平移 `>3 m` 或旋转 `>15°`。
- Phase B 工程准入：GT 可评价边≥3、accurate≥2、不同 query≥2、GT 覆盖率100%、large-error=0，并且 pose-copy PGO 安全。

主要代码入口：`Odometry/MACVO.py`、`Module/LoopClosure/Manager.py`、`Module/LoopClosure/VINSGeometry.py`。禁止参考已废弃的 `MAC-VO-test_loop`。
