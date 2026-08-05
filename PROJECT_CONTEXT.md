# MAC-VO 项目上下文

> 供新 Agent 快速接手。最后更新：2026-08-06。

## 1. 当前状态

| 项目 | 状态 |
|---|---|
| 仓库 / 分支 | `MAC-VO-test` / `agent-a-codex/loop-geometric-verification` |
| 已同步基线 | `8524e0b`（已推送当前 GitHub 分支） |
| 地点识别 | 默认 `dbow2_orb` + `ORBvoc.txt`；`custom_binary` 仅作消融 |
| 正式局部验证 | ORB-to-ORB + ORB-SLAM 风格筛选 + PnP + 双侧 3D covariance ICP 精化 |
| 回环缓存 | 实验配置每 5 个 sensor frame 缓存一次 |
| 全局优化 | CPU 稀疏 LM；固定信息矩阵与观测 Hessian 信息矩阵对比 |
| 当前状态 | 新 ICP covariance 路线已实现并通过静态检查；等待服务器 CUDA/离线验证 |

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
→ PnP 正深度内点位置采样 flow/covariance 与双侧深度
→ 双侧 31×31 局部深度不确定性校正及完整 3D covariance
→ Huber 加权 3D ICP 两帧 LM 精化
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
- Flow 仅在 ORB/PnP 最终候选上运行一次；candidate 3D 来自历史缓存，current
  3D 来自本次 `estimate_pair` 返回的当前深度；失败则拒绝该边，不回退 PnP 位姿。
- `match.mask=None` 表示无附加 mask；稠密图仅用 `floor(uv)` 索引，几何保留浮点坐标。
- 记录 Flow 与 ORB 对同一 current 像素预测差的 p50/p90，只诊断、不设门限。

旧 fixed-point 局部验证与旧 Phase B.5 已完成 CUDA observe，但未晋升；仅保留用于复现和消融，不再作为正式主路线。

## 4. 回环相对位姿精化与信息矩阵契约

正式模式为双侧 3D ICP：

```text
r_i = T_current_candidate · P_candidate,i - P_current,i
Sigma_r,i = R Sigma_candidate,i R^T + Sigma_current,i
```

- 双侧 3D 点坐标使用中心像素的中心深度；3D covariance 内部严格复用原
  `MatchCovariance` 的 31×31 局部统计，使用 `wavg_depth` 与 `wvar_depth`。
- candidate 位置 covariance 固定为 `0.25I` 并进入局部统计分支；current
  位置 covariance 使用本次长跨度 FlowFormer 的 `uu,vv,uv`。
- 提供位置 covariance 时，局部统计使用深度空间方差，不再使用中心
  `depth_cov`；完整邻域越界点直接删除，不补点。
- 双侧 covariance 相加采用
  `cross_view_covariance=ignored_independent_approximation`，论文必须明确该近似。
- 每轮 LM 按当前旋转重新计算 `Sigma_r`；Jacobian 只对 ICP 残差求导，不对
  covariance、白化矩阵或 Huber 权重求导。
- `Sigma_r` 对称化后严格 Cholesky，不使用 jitter；Hamming 距离不参与权重。
- 局部 Huber 使用白化残差范数，`delta=2.795`；最终信息矩阵为终点
  重新线性化得到的鲁棒加权高斯–牛顿近似 Hessian，不含 LM 阻尼，
  也不对 Huber 权重求导。
- 非有限/非 SPD residual covariance 与精化后非正深度观测按点剔除；
  剩余点不足或整体 Jacobian 不满秩时才拒绝整条边。
- 精化后重新执行与 PnP 相同的平移/完整旋转安全门，并记录
  PnP→精化位姿变化。
- 白化 `3×6` Jacobian 堆叠后必须满秩。
- raw Hessian 在 `T_current_candidate` 上计算，再以 Adjoint 转到最终边方向。
- 回环原始 Hessian 不再做旧广义特征值上限缩放；逐边输出点数、trace、
  `trace/point_count`、秩、条件数与特征值。

顺序边：

- 仅当 `match2frame1=src && match2frame2=dst` 时使用对应 MatchObs 构造同一
  双侧 3D ICP 鲁棒观测 Hessian：`pixel1_uv/pixel1_d/obs1_covTc` 对应
  candidate，`pixel2_uv/pixel2_d/obs2_covTc` 对应 current；不得用世界地图点
  替代双侧相机观测。
- 缺少直接观测、数值非法或退化时显式回退 `100I`，不删除边，保证图连通。
- 存在回退时，该实验分支称为“混合信息矩阵分支”，并报告观测 Hessian
  边数、fallback 数量、比例和原因；fallback=0 时称为“统一观测 Hessian
  信息矩阵分支”，不再使用 `mixed`。

## 5. 已冻结实验结论

以下多序列 Phase C 数字来自旧 `disp` 相对位姿精化/观测 Hessian 路线，保留
为消融基线，不能作为新 ICP 正式路线的结果：

- 固定 flow covariance 门限 `uu/vv<=100` 是旧 Flow PnP 覆盖率的直接阻断项；固定补点会增加大误差约束，路线已淘汰。
- DBoW2 在早期 abf001 人工审核集上 Recall@10 低于 custom，但该标签未经最终
  确认；为提高跨数据集泛化性，当前正式默认已改为标准 ORBvoc/DBoW2。
- ORB-to-ORB 严格匹配在 `1200→1100` 恢复了 25 个 PnP 内点并形成有效 information。
- 提交 `7ca8970` 的 Phase B 结果：5 对几何通过、5 对 information 有效、最终选择 4 条回环边。
- 4 条最终边的 GT 位姿代理均为 `accurate`，`large-error=0`；fixed/covariance pose-copy PGO 均 `safe=true`，Phase B 工程准入通过。
- 当前 4 条边均约为 100–110 帧跨度，`long_span_accurate=0`。这只是工程 smoke test，不是论文级稳定性证据。
- K09 stride-5/DBoW2 完成 1591 帧，319 个缓存帧；
  5 次最终候选网络精化全部成功并形成 5 条可比较回环边。
- 离线加载器修复后，K09 的 1590 条顺序边全部成功构造观测 Hessian，
  fallback=0。Phase C：no-loop ATE RMSE `9.91183`、fixed `2.46161`、
  观测 Hessian分支 `2.34212`；后者 RPE `0.016092`，略高于 no-loop
  `0.016051`，分类为 `inconclusive`。
- abf001 stride-5/DBoW2 构造 8 条合格回环边，1273 条顺序边全部使用
  观测 Hessian、fallback=0。no-loop/fixed/观测 Hessian分支 ATE RMSE 分别为
  `1.8764/1.8261/1.6016 m`；观测 Hessian分支 RPE `0.009905`，优于
  no-loop 的 `0.010445`，分类为 `positive`。
- KITTI 02 的全局稀疏 LM 对迭代上限敏感；观测 Hessian分支在上限 2000
  的实验中于第 1149 次按 `relative_loss_tolerance` 收敛，ATE `18.5138 m`、
  RPE `0.02745`，分类保持 `positive`。LM50→100→200→400→1000 的收益逐步
  饱和，不能仅以“达到迭代上限”判断算法未产生有效优化。

提交 `8524e0b` 已完成正式 ICP 路线实现，但尚未运行完整 ICP VO、Phase B 和
Phase C。服务器结果产生前，不得宣称 ICP 优于 disp。

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
- 正式配置的全局稀疏 LM `max_iterations=2000`；回环帧对 ICP 局部 LM
  `max_iterations=15`，两者不可混淆。

离线比较：

```text
no-loop：原轨迹，不注册 loop edge
fixed：odometry 100I + loop 100I
observation-Hessian：direct odometry ICP observation Hessian
                    + missing-edge 100I fallback（若存在）
                    + refined loop ICP robust Hessian
```

- fixed/observation-Hessian 使用完全相同的精化回环边集，只允许信息矩阵不同。
- no-loop 的回环边只用于 residual 诊断，不进入优化图。
- 正式 ATE/RPE 复用 `Evaluation/MetricsSeq.py` 的 evo 全局对齐口径。
- 同时报最大位姿修正、相邻变形、loop residual、优化前后鲁棒 loss 和
  顺序边 information fallback 覆盖率。
- Phase B/C 离线仍不得修改原 `VisualMap.pose`。
- Phase B manifest 必须记录 `vo_graph_type`、`loop_residual_mode`、
  `odometry_residual_mode`、covariance 模型、kernel 与配置摘要。正式 ICP 要求
  三种 residual mode 均为 `icp`；旧 disp 只作为显式或完整 legacy 消融读取，
  部分合约元数据或模式混用必须拒绝。

主要入口：`Scripts/AdHoc/RunLoopPhaseBOffline.py` 与
`Scripts/AdHoc/RunLoopPhaseCOffline.py`。

## 7. 论文与代码术语

- 整套扩展称为“回环闭合与全局优化模块”，不能称为“重定位模块”。重定位
  特指跟踪丢失后在已有地图中恢复相机绝对位姿。
- DBoW2 阶段称为“视觉地点识别与回环候选检索”；其输出只是候选，不是
  已确认回环约束。
- ORB 匹配与 PnP-RANSAC 称为“回环几何验证与初始相对位姿估计”。
- PnP 后的 FlowFormer+ICP LM 称为“不确定性加权回环相对位姿精化”，不称为
  “回环局部 PGO”。这里只有一个相对位姿变量，不是多节点位姿图。
- 原代码 `TwoFramePGO` 在论文中称为“相邻帧相对位姿优化”；代码类名不改。
- “不确定性”是概念；具体随机量称为“协方差矩阵”；PGO 边权重称为
  “信息矩阵”。`H=J^T W J` 称为“鲁棒加权高斯–牛顿近似 Hessian”，
  不是网络直接输出的六维协方差。
- 全局阶段称为“全局位姿图优化（Global PGO）”；“全局一致性优化”用于
  描述目标或效果，不作为具体求解器名称。

## 8. 测试与 GT 口径

只使用 `pytest` 与服务器离线/完整 VO 测试。

- GT translation error：`||t_est - t_gt||`。
- GT rotation error：SO(3) geodesic。
- accurate：平移 `<1 m` 且旋转 `<5°`。
- large-error：平移 `>3 m` 或旋转 `>15°`。
- Phase B 工程准入：GT 可评价边≥3、accurate≥2、不同 query≥2、GT 覆盖率100%、large-error=0，并且 pose-copy PGO 安全。

主要代码入口：`Odometry/MACVO.py`、`Module/LoopClosure/Manager.py`、`Module/LoopClosure/VINSGeometry.py`。禁止参考已废弃的 `MAC-VO-test_loop`。
