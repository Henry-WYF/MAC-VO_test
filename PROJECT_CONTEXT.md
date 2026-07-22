# MAC-VO 项目上下文

> 用于新 Agent 快速接手。最后更新：2026-07-22。

## 1. 当前状态

| 项目 | 状态 |
|---|---|
| 仓库 / 分支 | `MAC-VO-test` / `agent-a-codex/loop-geometric-verification` |
| 当前基线提交 | `9589eba`（实现改动后需重新记录实际 HEAD） |
| 地点识别 | 默认 `custom_binary`；DBoW2 保留为消融后端 |
| Phase A/B | 已实现；不写回 VO pose |
| 旧 Phase B.5 | 已完成 1135 候选 CUDA observe，但 calibration 未获信任，现退出正式主路线 |
| 当前任务 | VINS 风格固定低方差点几何验证与 `disp` information observe |
| Phase C | 仅在工程准入通过后，才允许在 pose 副本上测试 GlobalPGO |

修改前必须执行 `git status --short --branch`。工作区可能包含用户改动，不得恢复、覆盖或一并提交。CUDA 完整实验在远程服务器 Docker 中运行。

## 2. 原 MAC-VO 与低侵入边界

原系统主链为：

```text
FlowFormerCov 双目/相邻帧估计
→ covariance-aware 选点
→ 2D→3D covariance 传播
→ VisualMap
→ TwoFrame PGO
```

回环是可插拔扩展：

- 禁用回环时，不提取几何描述子、不写 sidecar、不执行回环验证。
- 不修改原 `KeypointSelector`、Frontend、TwoFramePGO 和 GlobalPGO 主逻辑。
- 回环缓存或验证失败只能禁用回环，原 VO 必须继续。
- Phase A/B 和离线 pose-copy PGO 不得修改原 `VisualMap.pose`。

## 3. 当前正式回环路线

```text
BoW 高分候选
→ 取最高分前 10（当前诊断配置，最终数量待控制样本实验确定）
→ 两侧 MAC-VO 最终低方差固定点 ORB descriptor-only 匹配
→ 单向 Hamming < 80，candidate 索引唯一
→ 带只读 VO 初值的 PnP RANSAC
→ 从通过者中选择历史最早候选
→ PnP 内点的 reprojection+disparity information observe
→ pose 副本 GlobalPGO 安全比较
```

关键契约：

- 独立 ORB 检测点只用于 BoW；几何验证只使用 OutlierFilter 后的固定 VO 点。
- 几何特征保存于可选 `geometry_features_v1` sidecar，主 loop-frame 缓存 schema 不变。
- sidecar 的 candidate 3D covariance 仅由固定像素方差和该帧 depth covariance 传播，不复用相邻帧 flow covariance。
- PnP 使用 `candidate 3D → current 2D`，输出 `T_current_candidate`；回环边保存其逆 `T_candidate_current`。
- MAC-VO NED 相机前向轴为 `x`；OpenCV 相机前向轴为 `z`。
- PnP 固定为 `SOLVEPNP_ITERATIVE`、100 次、10 px、confidence 0.99、正深度内点至少 26。
- 不使用 ratio、mutual、内点率、空间覆盖、Essential、pair-risk 或 calibration gate。
- 相对平移 `<20 m`、完整 SO(3) 角 `<30°` 是可配置安全门控；与 VO 的差异只记录。

## 4. Information 契约

每个 PnP 内点使用 `[u,v,disp]` 残差：

```text
Sigma_r = Sigma_current
        + J_X R Sigma_candidate_3D R^T J_X^T
```

- current `[u,v]` 复用 `match_cov_default` 方差，disparity 使用双目前端方差。
- current depth covariance 不重复计权；Hamming 距离不作为 covariance。
- `Sigma_r` 对称化后严格 Cholesky，不使用 jitter。
- 所有白化 `3×6` Jacobian 堆叠后必须满秩；raw Hessian 在 `T_current_candidate` 上计算，再通过 Adjoint 转到最终边方向。
- covariance information 只能相对固定 information 降权；第一轮仅 observe。
- information 无效的边不得进入固定/covariance pose-copy PGO 对比。
- 离线入口从同一组 `selected_for_query && pgo_comparison_eligible` 边构造固定/covariance 两套 pose-copy PGO；在线 GlobalPGO 保持禁用。

## 5. 已冻结实验结论

- 固定 flow covariance 门限 `uu/vv<=100` 只保留约 0.67% 入界有限点，是旧 Flow PnP 覆盖率的直接阻断项。
- adaptive 50/100/200 虽提高 PnP 覆盖，也增加大误差约束，固定补点路线已淘汰。
- `1200↔1100` 是真实回环但旧 PnP 位姿误差较大；`1250↔470/480` 是重复纹理误检。
- DBoW2 在当前 OpenCV ORB/abf001 审核集上未保持 custom 的 Recall@10，因此未晋升；该结论不代表 DBoW2 本身无效。
- 旧 B5 all-BoW calibration 有 195 对但 `trusted=false`；ORB-supported calibration 为 0。

## 6. 工程准入与测试

只使用 `pytest` 与服务器离线/完整 VO：

- GT误差：`||t_est-t_gt||` 与 SO(3) geodesic。
- accurate：平移 `<1 m` 且旋转 `<5°`。
- large-error：平移 `>3 m` 或旋转 `>15°`。
- 工程准入：GT可评价边≥3、accurate边≥2、不同query≥2、GT覆盖率100%、accepted large-error为0。
- pose-copy PGO 必须保证 pose/loss 有限、最终 loss 不增、首帧固定，并报告首帧对齐诊断、最大修正、相邻变形和 loop residual；正式 ATE/RPE 仍统一使用 `Evaluation/MetricsSeq.py` 的 evo 全局对齐口径。

主要代码入口：`Odometry/MACVO.py`、`Module/LoopClosure/Manager.py`、`Module/LoopClosure/VINSGeometry.py`。禁止参考已废弃的 `MAC-VO-test_loop`。
