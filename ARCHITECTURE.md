# MAC-VO 项目架构与数据流

本文档将 MAC-VO 项目的代码框架串成一条线：每个文件做什么、运行时数据如何流通、添加新功能需要改动哪些文件。不使用 ASCII 艺术流程图。

---

## Part A：项目总览与文件职责

> 按数据流自下而上的顺序分层。每层内的文件按"接口在前、实现在后"排列。

### 数据集层 — `DataLoader/`

| 文件 | 一句职责 | 关键导出 |
|---|---|---|
| [Interface.py](DataLoader/Interface.py) | 定义所有传感器数据容器：`StereoData`(单帧双目)、`IMUData`(IMU 批量)、`StereoFrame`(带时间戳的双目帧)、`StereoInertialFrame`(双目+IMU 帧)、`DataFramePair`(帧对) | `StereoData`, `StereoFrame`, `StereoInertialFrame`, `IMUData`, `DataFrame`, `Collatable` |
| [SequenceBase.py](DataLoader/SequenceBase.py) | 可迭代序列的抽象基类：支持 clip(切片)、preload(预加载到内存)、transform(数据变换链) | `SequenceBase`, `PreloadedSequence`, `TransformSequence`, `smart_transform` |
| [Transform.py](DataLoader/Transform.py) | `IDataTransform` 接口：定义单帧变换（resize、normalize 等），`smart_transform` 用它来构建变换链 | `IDataTransform` |

### 前端层 — `Module/Frontend/`

| 文件 | 一句职责 | 关键导出 |
|---|---|---|
| [StereoDepth.py](Module/Frontend/StereoDepth.py) | `IStereoDepth` 接口：给定双目图像 → 输出稠密深度图 + 深度协方差；包含 `disparity_to_depth()` / `disparity_to_depth_cov()` 转换工具 | `IStereoDepth`, `IStereoDepth.Output` |
| [Matching.py](Module/Frontend/Matching.py) | `IMatcher` 接口：给定两帧左目图像 → 输出稠密光流 + 光流协方差；`IMatcher.Output` 包含 `flow`(B,2,H,W)、`cov`(B,3,H,W)、`mask` | `IMatcher`, `IMatcher.Output` |
| [Frontend.py](Module/Frontend/Frontend.py) | `IFrontend` 接口：将深度和匹配联合为一次调用；`FlowFormerCovFrontend` 是核心实现——通过 B=2 批次（stereo→disparity, temporal→flow）在一次网络前向中同时输出深度和光流（含协方差） | `IFrontend`, `FlowFormerCovFrontend`, `FrontendCompose` |

### 滤波层 — `Module/`

| 文件 | 一句职责 | 关键导出 |
|---|---|---|
| [KeypointSelector.py](Module/KeypointSelector.py) | `IKeypointSelector` 接口：在 frame0 上选择 N 个关键点；`CovAwareSelector` 是核心算法——用 depth_cov + flow_cov 构建 quality map，经 NMS + 多层 mask 后随机采样 | `IKeypointSelector`, `CovAwareSelector`, `MappingPointSelector` |
| [OutlierFilter.py](Module/OutlierFilter.py) | `IObservationFilter` 接口：剔除异常观测；`FilterCompose` 按 AND 组合多个子滤波器；具体滤波器包括协方差异常检查、深度范围检查、前方景深 2σ 检查 | `IObservationFilter`, `FilterCompose`, `CovarianceSanityFilter`, `SimpleDepthFilter`, `LikelyFrontOfCamFilter` |
| [Covariance/Project2to3.py](Module/Covariance/Project2to3.py) | `ICovariance2to3` 接口：将 2D 像素不确定性投影为 3×3 空间协方差矩阵；`MatchCovariance` 是核心实现——在局部深度 patch 上用 flow 协方差构建 2D 高斯核，计算加权平均深度和加权深度方差，最后通过 `Covariance_2to3_full()` 解析传播到 3D | `ICovariance2to3`, `MatchCovariance`, `Covariance_2to3_full`, `Covariance_2to3_diag` |
| [MotionModel.py](Module/MotionModel.py) | `IMotionModel` 接口：为每帧提供初始位姿估计；`TartanMotionNet` 用网络回归相对 SE3 后复合到上一帧位姿；`GTMotionwithNoise` 用真值加噪；`StaticMotionModel` 返回上一帧位姿 | `IMotionModel`, `TartanMotionNet`, `GTMotionwithNoise` |
| [KeyframeSelector.py](Module/KeyframeSelector.py) | `IKeyframeSelector` 接口：判断当前帧是否关键帧；`AllKeyframe` 每帧都是关键帧，`UniformKeyframe` 按固定间隔选取 | `IKeyframeSelector`, `AllKeyframe`, `UniformKeyframe` |
| [MapProcessor.py](Module/MapProcessor.py) | `IMapProcessor` 接口：对 VisualMap 做后处理；`PoseInterpolate` 对 lost-track 帧做 SE3 线性插值，`MotionInterpolate` 对相对位姿插值后累积积重建绝对位姿 | `IMapProcessor`, `PoseInterpolate`, `MotionInterpolate` |

### 因子图存储层 — `Module/Map/`

| 文件 | 一句职责 | 关键导出 |
|---|---|---|
| [Template.py](Module/Map/Template.py) | 定义 `TensorBundle` 数据结构和三种 Feature 的字面量类型：`FrameFeature`(位姿、内参、时间戳)、`MatchingFeature`(像素坐标、深度、视差、协方差)、`PointFeature`(世界坐标、协方差、颜色) | `TensorBundle`, `FrameFeature`, `MatchingFeature`, `PointFeature`, `FrameNode`, `MatchObs`, `PointNode` |
| [Graph.py](Module/Map/Graph.py) | 定义 `AutoScalingBundle`(自动扩容的张量容器)和 6 种有向边类型：`Scaling_DenseEdge_Multi`(1→N 密集)、`Scaling_SparseEdge_Multi`(1→N 稀疏)、`Scaling_SingleEdge`(1→1)；所有边通过 `push()` / `add()` / `set()` / `get()` 操作 | `AutoScalingBundle`, `Scaling_DenseEdge_Multi`, `Scaling_SparseEdge_Multi`, `Scaling_SingleEdge` |
| [VisualMap.py](Module/Map/VisualMap.py) | 全局因子图：3 个节点存储(`FrameStore`、`MatchStore`、`PointStore`) + 6 种有向边(`frame2match`、`match2point`、`point2match`、`match2frame1`、`match2frame2`、`frame2map`)。提供所有 get_* 查询方法、序列化/反序列化 | `VisualMap`, `FrameStore`, `MatchStore`, `PointStore` |

### 优化层 — `Module/Optimization/`

| 文件 | 一句职责 | 关键导出 |
|---|---|---|
| [Interface.py](Module/Optimization/Interface.py) | `IOptimizer` 是优化器基类，支持顺序/并行两种模式：顺序模式在主线程阻塞运行，并行模式通过 `multiprocessing.Pipe` 将优化投递到子进程，让前端可以趁优化期间处理下一帧 | `IOptimizer`, `IOptimizerParallelWorker`, `move_dataclass_to_local` |
| [TwoFramePGO/Graphs.py](Module/Optimization/TwoFramePGO/Graphs.py) | 定义 `GraphInput`/`GraphOutput` 数据类 + 6 种因子图类（ICP/Reproj/ReprojDisp × Autodiff/Analytic），每种图负责前向残差计算 + 协方差提供 + 雅可比构建 | `GraphInput`, `GraphOutput`, `ICP_TwoframePGO`, `Reproj_TwoFramePGO`, `ReprojDisp_TwoFramePGO`，以及对应的 `Analytic_*` 变体 |
| [TwoFramePGO/Optimizer.py](Module/Optimization/TwoFramePGO/Optimizer.py) | `TwoFrame_PGO` 类：`get_graph_data()` 从 VisualMap 提取 GraphInput → `_optimize()` 以 Σ⁻¹ 为信息矩阵运行 LM-PGO → `write_graph_data()` 将优化后位姿写回 VisualMap | `TwoFrame_PGO`, `Local_TwoFrame_PGO` |
| [PyposeOptimizers.py](Module/Optimization/PyposeOptimizers.py) | 为 pypose 做了薄封装：`FactorGraph`(因子图基类)、`AnalyticModule`(解析雅可比)、`LM_analytic`(接收 weight=Σ⁻¹ 的 LM 优化器) | `FactorGraph`, `AnalyticModule`, `LM_analytic` |

### 网络模型层 — `Module/Network/`

| 文件 | 一句职责 | 关键导出 |
|---|---|---|
| [FlowFormerCov/flownet.py](Module/Network/FlowFormerCov/flownet.py) | `FlowFormerCov` 主网络：修改版 FlowFormer，输出稠密光流 + 每个像素的 2×2 协方差矩阵 | `FlowFormerCov` |
| [FlowFormerCov/covhead.py](Module/Network/FlowFormerCov/covhead.py) | 协方差预测头：`CovHead` + `CovUpdateBlock`(GRU 残差更新) + `MemoryCovDecoder`(从匹配特征解码协方差) | `CovHead`, `CovUpdateBlock`, `MemoryCovDecoder` |
| [CovHead.py](Module/Network/CovHead.py) | 独立的协方差头（非 FlowFormer 专用） | `CovarianceHead` |
| [PWCNet/RAFTCov.py](Module/Network/PWCNet/RAFTCov.py) | `PWCFeature` + `RAFTFlowCovNet`：PWCNet 特征金字塔提取 + RAFT GRU 迭代求精 + 协方差头 | `PWCFeature`, `RAFTFlowCovNet` |
| [PWCNet/pwc/](Module/Network/PWCNet/pwc/) | 标准 PWC-Net 实现（相关层、光流解码器） | `PWCModel` |
| [PWCNet/pwc_cov/](Module/Network/PWCNet/pwc_cov/) | PWC-Net 协方差变体（注意力模块、GRU 单元） | — |
| [StereoCov/](Module/Network/StereoCov/) | 立体匹配网络 + 协方差分支 | — |
| [TartanVOStereo/](Module/Network/TartanVOStereo/) | TartanVO 双目位姿估计 + 光流网络 | — |

### 主控层 — `Odometry/`

| 文件 | 一句职责 | 关键导出 |
|---|---|---|
| [Interface.py](Odometry/Interface.py) | `IOdometry` 基类：`receive_frames()` — 遍历序列 → `run(frame)` → 终止 → 保存 poses.npy 和 tensor_map.npz | `IOdometry` |
| [MACVO.py](Odometry/MACVO.py) | MAC-VO 主算法：`initialize()` 处理首帧 → `run_pair()` 串联全部 13 步 pipeline → `push_keyframe()` 注册非关键帧 → `terminate()` 做后处理和清理 | `MACVO` |

### 工具层 — `Utility/`

| 文件 | 一句职责 | 关键导出 |
|---|---|---|
| [Config.py](Utility/Config.py) | YAML 配置加载系统：支持 `!include`（文件引用）和 `!flatten_seq`（序列展平）自定义标签；`asNamespace()` 将嵌套 dict 递归转为 SimpleNamespace 以支持属性访问 | `load_config`, `build_dynamic_config`, `asNamespace` |
| [Extensions.py](Utility/Extensions.py) | `ConfigTestableSubclass` 混入类：所有模块基类通过它获得 `instantiate(type, config)` 工厂方法和 `is_valid_config()` 验证钩子 | `ConfigTestableSubclass` |
| [Point.py](Utility/Point.py) | 坐标工具：`pixel2point_NED`(像素→相机 3D，输出 EDN 坐标经 roll 转为 NED)、`point2pixel_NED`(相机 3D→像素)、`filterPointsInRange`(过滤超出图像范围的点) | `pixel2point_NED`, `point2pixel_NED`, `filterPointsInRange` |
| [Timer.py](Utility/Timer.py) | GPU/CPU 计时装饰器：`@Timer.cpu_timeit` / `@Timer.gpu_timeit` | `Timer` |
| [PrettyPrint.py](Utility/PrettyPrint.py) | 日志系统：`Logger.write(level, msg)` + `ColoredTqdm` 进度条 | `Logger`, `ColoredTqdm` |
| [Trajectory.py](Utility/Trajectory.py) | 轨迹工具：SE3 轨迹的对齐、缩放、坐标系转换，对接 evo 库 | `Trajectory` |
| [Sandbox.py](Utility/Sandbox.py) | 实验沙盒：管理单次实验的输入配置、输出结果、中间文件的目录结构 | `Sandbox` |
| [Plot.py](Utility/Plot.py) | 可视化工具：轨迹对比图、颜色映射 | `getColor`, `plot_trajectories` |
| [Utils.py](Utility/Utils.py) | 杂项工具 | — |
| [Math.py](Utility/Math.py) | 数学工具：`gaussian_mixture_mean_var`(高斯混合的均值/方差)、`gaussain_full_kernels`(从 2×2 协方差构建 2D 高斯核) | `gaussian_mixture_mean_var`, `gaussain_full_kernels` |

### 评估层 — `Evaluation/`

| 文件 | 一句职责 | 关键导出 |
|---|---|---|
| [EvalSeq.py](Evaluation/EvalSeq.py) | 批量评估：对多个实验空间计算 ATE/RTE/ROE/RPE 指标，支持 scale 校正，输出表格和 CSV | `EvaluateSequences`, `EvaluateSequencesAvg` |
| [MetricsSeq.py](Evaluation/MetricsSeq.py) | 封装 evo 库的轨迹评估指标 | `evaluateATE`, `evaluateRTE`, `evaluateROE`, `evaluateRPE` |

---

## Part B：线性流水线（MACVO.run_pair 逐步追溯）

> 每一步标注源文件行号，说明输入/输出 tensor shape，以及调用链。

### 步骤 0：帧进入

```
MACVO.run(frame) → 首帧 initialize() 否则 run_pair(prev_keyframe[0], frame1)
  输入: frame → StereoFrame { stereo: StereoData, gt_pose: SE3 | None, idx: list[int] }
```

### 步骤 1：首帧初始化 [MACVO.py:158-172]

```
initialize(frame0)
  ├─ IFrontend.estimate_depth(frame0.stereo)
  │   输入: StereoData { imageL: Float32[1,3,H,W], imageR: Float32[1,3,H,W] }
  │   输出: IStereoDepth.Output { depth: Float32[1,1,H,W], cov: Float32[1,1,H,W]|None,
  │                               disparity: Float32[1,1,H,W]|None, ... }
  ├─ IMotionModel.predict(frame0, flow=None, depth=depth0.depth)
  │   输出: pp.SE3 → .unsqueeze(0) → Float32[1,7]  世界坐标系位姿
  ├─ VisualMap.frames.push(FrameNode)  
  │   入: { pose[1,7], T_BS[1,7], need_interp[1]bool, time_ns[1]int64, K[1,3,3], baseline[1] }
  │   出: frame_idx[1]int64
  └─ 设置 self.prev_keyframe = (frame0, int(frame_idx), depth0)
```

### 步骤 2：关键帧判断 [MACVO.py:194]

```
IKeyframeSelector.isKeyframe(frame1) → bool
  AllKeyframe      → 总是 True
  UniformKeyframe  → frame_idx % freq == 0
若 False → push_keyframe(frame1, est_pose, need_interp=True) 然后 return (不优化)
```

### 步骤 3：前端联合估计 [MACVO.py:199]

```
depth1, match01 = IFrontend.estimate_pair(frame0.stereo, frame1.stereo)

[FlowFormerCovFrontend 路径]:
  input_A = cat([frame_t2.imageL, frame_t1.imageL]) → Float32[2,3,H,W]
  input_B = cat([frame_t2.imageR, frame_t2.imageL]) → Float32[2,3,H,W]
  est_flow, est_cov = model.inference(input_A, input_B)
    est_flow → Float32[2,2,H,W],  est_cov → Float32[2,3,H,W]

  拆分:
  [0] → disparity → inference_2_depth() → depth1: IStereoDepth.Output
    disparity = flow[0,:1].abs() → Float32[1,1,H,W]
    depth = (baseline * fx) / disparity → Float32[1,1,H,W]
    depth_cov = (baseline*fx)² * disp_cov / disparity^4 (一阶误差传播)
  [1] → flow → inference_2_match() → match01: IMatcher.Output
    flow: Float32[1,2,H,W],  cov: Float32[1,3,H,W]
```

### 步骤 4：上一轮优化回写 [MACVO.py:204]

```
IOptimizer.write_map(graph)  # 将上一帧的优化结果写入
  → TwoFrame_PGO.write_graph_data(result, global_map)
  → global_map.frames.data["pose"][result.frame_idx] = optimized_pose
影响: VisualMap.frames 中上一优化帧的 pose 被更新
```

### 步骤 5：运动模型预测 [MACVO.py:210-211]

```
IMotionModel.update(pp.SE3(graph.frames.data["pose"][prev_kf_idx]))
est_pose = IMotionModel.predict(frame1, match01.flow, depth1.depth)
  输入: match01.flow → Float32[1,2,H,W],  depth1.depth → Float32[1,1,H,W]
  输出: pp.SE3 → .unsqueeze(0) → Float32[1,7]  # frame1 初始位姿
```

### 步骤 6：关键点选择 [MACVO.py:214]

```
kp0_uv = IKeypointSelector.select_point(frame0.stereo, self.num_point, depth0, depth1, match01)
  输入: depth0, depth1 (IStereoDepth.Output), match01 (IMatcher.Output), num_point int
  输出: Float32[N,2]  # (u, v) 像素坐标，N ≤ num_point

[CovAwareSelector 算法]:
  quality = depth_cov0 + depth_cov1                → Float32[1,1,H,W]
  若 match01.cov 存在: quality *= (σ_uu + σ_vv - 2σ_uv) → 协方差越低=质量越高
  NMS: -max_pool2d(-quality, kernel_size)         → 找局部极大值（质量好的区域）
  多层 mask AND: border_mask + depth_mask + depth_cov_mask + flow_cov_mask
  最终: torch.nonzero(mask) → 随机子集截断到 numPoint
```

### 步骤 7：光流跟踪到 frame1 [MACVO.py:215-222]

```
kp1_uv = kp0_uv + IFrontend.retrieve_pixels(kp0_uv, match01.flow).T
  retrieve_pixels: 从 Float32[1,2,H,W] 中按 kp0_uv[N,2] 索引 → Float32[2,N]
  .T → Float32[N,2]
  kp1_uv = kp0_uv + 光流位移 → Float32[N,2]

inbound_mask = filterPointsInRange(kp1_uv, (edge_w, W-edge_w), (edge_v, H-edge_v)) → bool[N]
kp0_uv = kp0_uv[inbound_mask]; kp1_uv = kp1_uv[inbound_mask]  → Float32[≤N,2]
```

### 步骤 8：深度和协方差采样 [MACVO.py:225-249]

```
对每个关键点从稠密图上采样:

kp0_d          = retrieve_pixels(kp0_uv, depth0.depth).squeeze(0)   → Float32[≤N,]
kp0_sigma_dd   = retrieve_pixels(kp0_uv, depth0.cov).squeeze(0)     → Float32[≤N,]|None
kp0_disparity  = retrieve_pixels(kp0_uv, depth0.disparity)           → Float32[1,≤N]|None
kp1_d          = retrieve_pixels(kp1_uv, depth1.depth).squeeze(0)   → Float32[≤N,]
kp1_sigma_dd   = retrieve_pixels(kp1_uv, depth1.cov).squeeze(0)     → Float32[≤N,]|None

kp0_sigma_uv = ones(≤N, 3) * match_cov_default;  → Float32[≤N,3]
kp0_sigma_uv[..., 2] = 0  # kp0 是选定像素，无 uv 协方差（只有量化噪声）
kp1_sigma_uv = retrieve_pixels(kp0_uv, match01.cov).T              → Float32[≤N,3]|None
```

### 步骤 9：2D→3D 投影 + 协方差传播 [MACVO.py:257-259]

```
pos0_Tc = pixel2point_NED(kp0_uv, kp0_d, frame0.stereo.frame_K)
  调用 pypose.pixel2point(pixels[N,2], depths[N,], K[3,3])
  然后 .roll(shifts=1, dims=-1)  # EDN→NED: (east,down,north)→(north,east,down)
  输出: Float32[≤N,3]  相机坐标系下的 3D 点 (NED)

pos0_covTc = ICovariance2to3.estimate(frame0.stereo, kp0_uv, depth0, kp0_sigma_dd, kp0_sigma_uv)
pos1_covTc = ICovariance2to3.estimate(frame1.stereo, kp1_uv, depth1, kp1_sigma_dd, kp1_sigma_uv)
  输出: Float64[≤N,3,3]

[MatchCovariance 算法]:
  1. 从 flow_cov [N,3] 的 (σ_uu, σ_vv, σ_uv) 构建 2×2 协方差矩阵 → 2D 高斯核 [N,K,K]
  2. 在每个关键点周围提取 depth_est.depth 的 K×K patch → Float32[N,K,K]
  3. 加权平均深度 wavg_depth = Σ(kernel * patch)         → Float32[N,]
  4. 加权深度方差 wvar_depth = Σ(kernel * (patch - wavg)²) → Float32[N,]
  5. Covariance_2to3_full(σ_uu, σ_uv, σ_vv, wvar_depth, u, v, wavg_depth, fx, fy, cx, cy)
     - σ_xx = ((u-cx)²·σ_dd + d²·σ_uu + σ_uu·σ_dd) / fx²
     - σ_yy = ((v-cy)²·σ_dd + d²·σ_vv + σ_vv·σ_dd) / fy²
     - σ_zz = σ_dd
     - σ_xy = ((u-cx)(v-cy)·σ_dd + (d²+σ_dd)·σ_uv) / (fx·fy)
     - σ_xz = σ_dd·(u-cx) / fx
     - σ_yz = σ_dd·(v-cy) / fy
     → Float64[N,3,3] 相机坐标系下的 3D 协方差
```

### 步骤 10：外点过滤 [MACVO.py:263-287]

```
match_obs = MatchObs.init({ pixel1_uv, pixel2_uv, pixel1_d, pixel2_d,
  pixel1_disp, pixel2_disp, pixel1_disp_cov, pixel2_disp_cov,
  pixel1_d_cov, pixel2_d_cov, pixel1_uv_cov, pixel2_uv_cov,
  obs1_covTc, obs2_covTc })  # 所有字段 shape 为 [N,] 或 [N,2/3/3x3]

mask = IObservationFilter.filter(match_obs, device("cpu")) → bool[≤N]
  FilterCompose: AND 串联
    CovarianceSanityFilter  → 剔除 NaN/Inf 协方差
    SimpleDepthFilter       → 剔除深度超出 [min_depth, max_depth]
    LikelyFrontOfCamFilter  → 剔除 depth - 2σ < 0（可能在相机后方）

match_obs = match_obs[mask] → 剩余 M ≤ N 条有效观测
```

### 步骤 11：因子图注册 [MACVO.py:289-309]

```
prev_pose = pp.SE3(graph.frames.data["pose"][prev_kf_idx])    → SE3
prev_rot  = prev_pose.rotation().matrix().repeat((M,1,1))     → Float64[M,3,3]

# 点
pos_Tw = pp.SE3.Act(prev_pose, pos0_Tc)[..., :3]               → Float32[M,3]  世界坐标
cov_Tw = R @ pos0_covTc @ R^T                                   → Float64[M,3,3]  世界坐标协方差
point_idx = graph.points.push(PointNode{pos_Tw, cov_Tw, color}) → int64[M,]

# 帧
frame_idx = graph.frames.push(FrameNode{pose, T_BS, K, baseline, need_interp, time_ns}) → int64[1,]

# 匹配
match_idx = graph.match.push(match_obs) → int64[M,]

# 6 条边
graph.point2match.add(point_idx, match_idx)                 # Point→Match (稀疏多)
graph.match2point.set(match_idx, point_idx)                 # Match→Point (一对一)
graph.frame2match.add(prev_kf_idx, start, M)                # Frame0→Match (密集多)
graph.frame2match.add(frame_idx, start, M)                  # Frame1→Match (密集多)
graph.match2frame1.set(match_idx, prev_kf_idx.item())       # Match→Frame0
graph.match2frame2.set(match_idx, frame_idx.item())         # Match→Frame1
```

### 步骤 12：掉跟踪检查 [MACVO.py:320-324]

```
if M < min_num_point (默认 10):
    标记 graph.frames.data["need_interp"][frame_idx] = True
    return  # 跳过优化，该帧交由 MapRefiner 后处理
```

### 步骤 13：启动优化 [MACVO.py:326-328]

```
graph_data = Optimizer.get_graph_data(graph, frame_idx)
  → 从 VisualMap 提取 GraphInput { frame_idx, from_idx, init_motion[1,7],
      baseline[1,], observations[M,], points[M,], K[3,3], edges_index[M,], device }

Optimizer.start_optimize(graph_data)
  顺序模式: _optimize(context, graph_data) → 阻塞, optim_result 存入 self.optimize_res
  并行模式: Pipe.send(graph_data) → 子进程运行优化 → 非阻塞, write_map() 时才接收

[_optimize 内部]:
  graph = pose_graph_class(graph_data).to(device, torch.double)
  while 未收敛:
    weight = block_diag(*pinverse(graph.covariance_array()))  # Σ⁻¹ 作为信息矩阵
    loss = LM_optimizer.step(input=(), weight=weight)
  return graph.write_back() → GraphOutput { motion[1,7], frame_idx, from_idx }
```

### 步骤 14：稠密建图点（可选） [MACVO.py:331-354]

```
若 self.mapping == True:
  map0_uv = MappointSelector.select_point(frame0, 2000, depth0, depth1, match01) → int64[M',2]
  同样的 2D→3D 投影 + 协方差传播
  写入 graph.map_points + graph.frame2map 边
```

### 步骤 15：终止 [MACVO.py:391-397]

```
terminate():
  Optimizer.write_map(graph)           # 最后写入待处理结果
  Optimizer.terminate()                # 关闭子进程
  MapRefiner.elaborate_map(graph.frames)  # 插值 need_interp 帧的位姿

receive_frames() 结尾:
  global_map = get_map() → VisualMap
  body_poses = T_BS @ sensor_poses @ T_BS.Inv() → 转为机体坐标系
  np.save("poses.npy", body_poses)           # Nx8: [time_ns, tx, ty, tz, qw, qx, qy, qz]
  np.savez_compressed("tensor_map.npz", **global_map.serialize())
```

---

## Part C：调用链树

```
IOdometry.receive_frames(sequence, saveto)             [Odometry/Interface.py]
 └─ for each StereoFrame:
     ├─ MACVO.run(frame)                                [Odometry/MACVO.py]
     │   ├─ [首帧] initialize(frame0)                   [MACVO.py:158]
     │   │   ├─ IFrontend.estimate_depth(frame0.stereo)  [Frontend.py]
     │   │   │   ├─ [FlowFormerCov] model.inference()
     │   │   │   └─ disparity → depth + depth_cov (公式传播)
     │   │   ├─ IMotionModel.predict(frame0, None, depth)
     │   │   ├─ VisualMap.frames.push(FrameNode)
     │   │   └─ OutlierFilter.set_meta(frame0.stereo)
     │   │
     │   └─ [后续帧] run_pair(frame0, frame1)            [MACVO.py:174]
     │       ├─ [1] IKeyframeSelector.isKeyframe(frame1) → bool
     │       ├─ [2] IFrontend.estimate_pair(frame0.stereo, frame1.stereo)
     │       │       输出: depth1, match01 (含 cov)
     │       ├─ [3] IOptimizer.write_map(graph)  # 上轮结果回写
     │       ├─ [4] IMotionModel.update(pose) + predict()
     │       ├─ [5] IKeypointSelector.select_point(..., depth0, depth1, match01)
     │       ├─ [6] retrieve_pixels(kp0_uv, flow) → kp1_uv; filterPointsInRange()
     │       ├─ [7] retrieve_pixels(kp_uv, depth/cov maps) → 逐点深度/方差
     │       ├─ [8] pixel2point_NED() → pos_Tc; ICovariance2to3.estimate() → cov_Tc
     │       ├─ [9] IObservationFilter.filter(match_obs) → mask
     │       ├─ [10] 因子图注册:
     │       │        VisualMap.points.push(PointNode)   [pos_Tw, cov_Tw, color]
     │       │        VisualMap.frames.push(FrameNode)   [pose, K, T_BS, ...]
     │       │        VisualMap.match.push(MatchObs)     [pixel_uv, depth, cov, ...]
     │       │        graph.point2match.add() + graph.match2point.set()
     │       │        graph.frame2match.add() + graph.match2frame1/2.set()
     │       ├─ [11] 掉跟踪检查: M < min_point → need_interp=True, return
     │       ├─ [12] get_graph_data() → start_optimize() → LM-PGO(Σ⁻¹)
     │       └─ [13] [可选] MappointSelector → 稠密建图点
     │
     └─ terminate()                                      [MACVO.py:391]
         ├─ IOptimizer.write_map()  # 最后一次回写
         ├─ IOptimizer.terminate()  # 关闭子进程
         └─ IMapProcessor.elaborate_map(frames)  # 插值位姿

  保存:
  ├─ global_map = get_map()
  ├─ body_poses = T_BS @ sensor_poses @ T_BS.Inv()
  ├─ np.save("poses.npy", body_poses)  # N×8
  └─ np.savez_compressed("tensor_map.npz", **serialize(global_map))
```

---

## Part D：扩展指南 — 添加新功能需要改动哪些文件

> 如果你想添加功能 X，你需要看 Y 文件、做 Z 改动。

### D.1 添加回环检测

**改动范围**：只需要在优化层和因子图层做扩展，前端/滤波/协方差层完全不受影响。

```
新增文件:
  Module/LoopClosure.py                    # ILoopDetector 接口 + DBoW/NetVLAD 实现
    - ILoopDetector.detect(frame) → LoopCandidate[]
    - 每帧提取描述子，与新帧比较找到回环候选

需要改动的现有文件:
  Module/Map/Template.py                   # 新增 LoopFeature 字面量组
    + LoopFeature = Literal["from_frame", "to_frame", "rel_pose", "cov", "score"]
    + LoopObs = TensorBundle[LoopFeature]  # 类型别名

  Module/Map/Graph.py                      # 新增回环边类型
    + Scaling_DenseEdge_Multi: loop2frame  # 1 个回环匹配 → N 个相关帧
    + Scaling_SingleEdge:      frame2loop  # 1 帧 → 1 条回环边

  Module/Map/VisualMap.py                  # 新增回环节点存储 + 边
    + LoopStore       → AutoScalingBundle[LoopFeature]
    + loop2frame      → Scaling_DenseEdge_Multi
    + frame2loop      → Scaling_SingleEdge

  Module/Optimization/TwoFramePGO/Graphs.py  # 新增回环因子图
    + Loop_TwoFramePGO(AnalyticModule)  # Sim(3) 或 SE3 回环约束
    - 残差: T_cur @ p - T_loop @ p'     # 相对位姿约束
    - 协方差: 回环匹配置信度
    - 雅可比: (1, 7) SE3 解析雅可比，参考现有 Analytic_ICP_TwoframePGO 的 build_jacobian()

  Module/Optimization/TwoFramePGO/Optimizer.py  # 扩展 GraphInput
    + GraphInput.loop_obs → LoopObs | None
    + get_graph_data() 中提取回环边数据

  Odometry/MACVO.py                        # 在关键帧处调用回环检测
    + 步骤 4.5 之后: self.LoopDetector.detect(frame1) → candidates[]
    + 若发现回环: 注册 LoopObs 到 VisualMap
    + 回环帧的优化: 可能需要重新优化多个历史帧

完全不需要改动的文件:
  - Module/Frontend/*       (前端输出不依赖回环)
  - Module/KeypointSelector.py  (关键点选择不变)
  - Module/Covariance/*     (协方差投影不变)
  - Module/OutlierFilter.py (filter 链不变)
  - Module/KeyframeSelector.py (关键帧判断不变)
  - Module/MotionModel.py   (运动模型不变)
  - Module/MapProcessor.py  (后处理不变)
  - DataLoader/*            (数据格式不变)
```

### D.2 添加 IMU 紧耦合

**改动范围**：影响面较大，贯穿数据层 → 前端层 → 因子图层 → 优化层。因为 IMU 数据已由 `StereoInertialFrame` 携带，数据层改动较小。

```
新增文件:
  Module/IMUPreintegration.py               # IMU 预积分模块
    - preintegrate(acc, gyro, dt) → (ΔR, Δv, Δp, Σ)  # 两帧间的 IMU 预积分
    - 实现: On-Manifold IMU Preintegration (Forster 2015)
    - 预积分结果: 相对旋转(3,3) + 相对平移(3,) + 速度变化(3,) + 协方差(9,9)

需要改动的现有文件:
  Module/Map/Template.py                    # FrameFeature 扩展
    + "vel"    → Float32[N,3]  # 机体速度 (世界坐标系)
    + "ba"     → Float32[N,3]  # 加速度计偏置
    + "bg"     → Float32[N,3]  # 陀螺仪偏置

  Module/Map/Graph.py                       # 无需新增边类型（IMU 已有 frame 间关系）
    # IMU 预积分是帧间约束，复用已有的 frame 间顺序关系

  Module/Map/VisualMap.py                   # FrameStore 扩展
    + frames.data["vel"]   → Float32[N,3]
    + frames.data["ba"]    → Float32[N,3]
    + frames.data["bg"]    → Float32[N,3]

  Module/Optimization/TwoFramePGO/Graphs.py  # GraphInput 扩展 + 新增 IMU 因子图
    + GraphInput.imu_preinteg  → IMUPreintegration | None
    + IMU_TwoFramePGO  # ICP + IMU 联合优化
    - 视觉残差: T·p_cam - p_world            (3×1, 来自已有 ICP)
    - IMU 残差:  preinteg - (T1⁻¹·T2)       (15×1, 旋转3+速度3+位置3+偏置6)
    - 总残差: cat(visual, imu)               (3M+15,)
    - 协方差: block_diag(Σ_visual, Σ_imu)    (3M+15, 3M+15)
    - 信息矩阵: pinv(total_cov)              block_diag(Σ_visu⁻¹, Σ_imu⁻¹)

  Module/Optimization/TwoFramePGO/Optimizer.py  # get_graph_data() 扩展
    + 提取相邻帧的 IMU 数据 → 预积分
    + GraphInput 中传入 preintegrated 结果

  Module/MotionModel.py                     # 实现 IMU 运动模型
    + IMUMotionModel: 用 IMU 预积分替代纯视觉预测
    - predict(frame, flow, depth):
        imu = frame.imu  # 获取当前帧的 IMU 数据
        return prev_pose @ preintegrate(imu)  # 预积分累积的位姿变化

  Odometry/MACVO.py                         # 主循环集成 IMU
    + initialize(): 初始化 IMU 偏置、速度
    + run_pair() 步骤 5 之前:
        imu_data = frame1.imu  # IMUData { acc[N,3], gyro[N,3], time_ns[N,1] }
        preinteg = IMUPreintegrator(imu_data, dt)  → (ΔR, Δv, Δp, Σ_imu)
    + 将 preinteg 传入 Optimizer
    + push_keyframe(): 写入速度/偏置到 FrameStore

  DataLoader/Interface.py                   # 已有 IMUData，确认无误
    - StereoInertialFrame 已定义 imu: IMUData 字段（无需修改）
    - 确认数据集序列返回 StereoInertialFrame 而非 StereoFrame

  Module/MapProcessor.py                    # 可选：偏置平滑
    + BiasSmoothingProcessor: 对 ba, bg 做后处理平滑

不需要改动的文件:
  - Module/Frontend/*       (前端输出不依赖 IMU)
  - Module/KeypointSelector.py  (关键点选择不变)
  - Module/Covariance/*     (协方差投影不变)
  - Module/OutlierFilter.py (filter 链不变)
  - Module/KeyframeSelector.py (关键帧判断不变)
  - Evaluation/*            (评估工具不变)
```

### D.3 共同注意事项

1. **遵循插件架构**：新模块实现对应接口（`ILoopDetector`、`IMUPreintegrator`），通过 YAML 配置的 `type` 字段动态加载。在 `Config/Experiment/` 下新建实验配置文件。

2. **优化器的并行模式**：`IOptimizer` 的并行子进程通过 `move_dataclass_to_local()` 将数据深拷贝到子进程内存。如果新增的 `GraphInput` 字段包含 tensor，确保它们在序列化路径中被正确处理（`tensor_safe_asdict()`）。

3. **VisualMap 的序列化**：`serialize()` / `deserialize()` 方法按字段名遍历。新增字段会自动被包含，但需要确保无自定义序列化逻辑遗漏。

4. **坐标系一致性**：所有世界坐标使用 NED（+x 北，+y 东，+z 下）。IMU 读数通常是机体坐标系，需通过 `T_BS`（body→sensor）转换。新加的 IMU 预积分必须在机体坐标系下计算，保持与视觉位姿的坐标系对齐。

5. **VisualMap 是 monotonic 增长的**：`AutoScalingBundle` 只增不删。回环优化如果修正了历史帧位姿，需要通过 overwrite（`frames.data["pose"][idx] = new_pose`）而非 push 来实现。

---

## Part E：建议阅读路线

> 按"被依赖者优先"排列。先读不依赖其他模块的基础层，再读调用它们的上层模块，最后读串起所有的编排层。

### 第 1 层：基础数据结构 — 理解"数据长什么样"

| 顺序 | 文件 | 重点看 | 花费 |
|---|---|---|---|
| 1 | [DataLoader/Interface.py](DataLoader/Interface.py) | `StereoData`、`StereoFrame`、`IMUData`、`DataFrame`、`Collatable.collate()` 的字段和 shape | ~15min |
| 2 | [Module/Map/Template.py](Module/Map/Template.py) | `TensorBundle` 的 push/get/set 操作、`FrameFeature` / `MatchingFeature` / `PointFeature` 字面量定义了哪些字段 | ~10min |

### 第 2 层：配置系统与插件机制 — 理解"模块怎么被加载的"

| 顺序 | 文件 | 重点看 | 花费 |
|---|---|---|---|
| 3 | [Utility/Extensions.py](Utility/Extensions.py) | `ConfigTestableSubclass.instantiate(type, config)` — 这就是所有模块通 YAML 动态切换的秘密 | ~10min |
| 4 | [Utility/Config.py](Utility/Config.py) | `!include` 标签、`!flatten_seq` 标签、`asNamespace()` 递归转换 | ~10min |

### 第 3 层：因子图存储 — 理解"数据是怎么组织和查询的"

| 顺序 | 文件 | 重点看 | 花费 |
|---|---|---|---|
| 5 | [Module/Map/Graph.py](Module/Map/Graph.py) | `AutoScalingBundle` 的内部 buffer 扩容策略；三种边类型 `Scaling_DenseEdge_Multi` / `Scaling_SparseEdge_Multi` / `Scaling_SingleEdge` 各适合什么场景 | ~20min |
| 6 | [Module/Map/VisualMap.py](Module/Map/VisualMap.py) | 三个 Store + 六条边的初始化；每个 `get_*` 方法返回什么；`serialize()` / `deserialize()` 如何遍历字段 | ~25min |

### 第 4 层：接口定义 — 理解"每个模块的契约是什么"

| 顺序 | 文件 | 重点看 | 花费 |
|---|---|---|---|
| 7 | [Module/Frontend/StereoDepth.py](Module/Frontend/StereoDepth.py) | `IStereoDepth.estimate()` 接口签名；`IStereoDepth.Output` 包含哪些字段；`disparity_to_depth()` 是如何从视差 → 深度的 | ~15min |
| 8 | [Module/Frontend/Matching.py](Module/Frontend/Matching.py) | `IMatcher.estimate()` 接口签名；`IMatcher.Output` 的 `flow`(2 通道) 和 `cov`(3 通道) 分别代表什么 | ~10min |
| 9 | [Module/Frontend/Frontend.py](Module/Frontend/Frontend.py) | `IFrontend.estimate_pair()` 如何通过一次性拼接两个 batch 条目来节省一次网络前向——这是关键性能优化 | ~20min |
| 10 | [Module/KeypointSelector.py](Module/KeypointSelector.py) | `CovAwareSelector.select_point()` — quality 图构建、NMS 局部最大值抑制、多层 mask 交运算、随机采样截断到此数的四步全貌 | ~15min |
| 11 | [Module/Covariance/Project2to3.py](Module/Covariance/Project2to3.py) | `MatchCovariance.estimate()` — 这是 MAC-VO 根据论文第三节创的新点：在局部深度 patch 上用 flow 协方差构建 2D 高斯核 → 加权均值+方差 → `Covariance_2to3_full()` 六条公式解析传播到 3D | ~20min |
| 12 | [Module/OutlierFilter.py](Module/OutlierFilter.py) | `FilterCompose` 组合模式（AND 串联多个子 filter）；三个具体 filter 各检查什么条件 | ~10min |
| 13 | [Module/MotionModel.py](Module/MotionModel.py) | `IMotionModel.predict()` — 初始位姿从哪来？几种实现：网络回归、真值加噪、静止假设 | ~10min |
| 14 | [Module/KeyframeSelector.py](Module/KeyframeSelector.py) | `IKeyframeSelector.isKeyframe()` — 很简单的接口，一眼看完 | ~5min |
| 15 | [Module/MapProcessor.py](Module/MapProcessor.py) | `IMapProcessor.elaborate_map()` — `PoseInterpolate` 对 lost-track 帧做 SE3 插值 | ~10min |

### 第 5 层：优化引擎 — 理解"因子图怎么变成优化后的位姿"

| 顺序 | 文件 | 重点看 | 花费 |
|---|---|---|---|
| 16 | [Module/Optimization/PyposeOptimizers.py](Module/Optimization/PyposeOptimizers.py) | `FactorGraph` 基类、`LM_analytic` — 替 pypose 做的薄封装。重点看 `weight` 参数如何参与 LM 步长计算 | ~15min |
| 17 | [Module/Optimization/TwoFramePGO/Graphs.py](Module/Optimization/TwoFramePGO/Graphs.py) | 六种因子图类。每类的 `forward()`(残差)、`covariance_array()`/`__information_matrix()`(协方差 → 信息矩阵)、`build_jacobian()`(解析雅可比)。理解 ICP(3D-3D 点距离) vs Reproj(2D 重投影误差) vs ReprojDisp(+ 视差约束) 的区别 | ~30min |
| 18 | [Module/Optimization/TwoFramePGO/Optimizer.py](Module/Optimization/TwoFramePGO/Optimizer.py) | `get_graph_data()` → `_optimize()` → `write_graph_data()` 三步流程；LM 主循环中 Σ⁻¹ 如何被构建为 `block_diag` 权重矩阵；`GraphInput` 和 `GraphOutput` 的结构 | ~20min |
| 19 | [Module/Optimization/Interface.py](Module/Optimization/Interface.py) | 顺序/并行双模式，`multiprocessing.Pipe` 如何让前端和优化走异步流水线 | ~15min |

### 第 6 层：主循环与入口 — 理解"一切怎么串起来"

| 顺序 | 文件 | 重点看 | 花费 |
|---|---|---|---|
| 20 | [Odometry/Interface.py](Odometry/Interface.py) | `receive_frames()` — 数据集遍历 + `run(frame)` + 终止后保存 `poses.npy` + `tensor_map.npz` | ~10min |
| 21 | [MACVO.py](MACVO.py) | `initialize()` + `run_pair()`(15 步) + `terminate()`。对照 ARCHITECTURE.md Part B 一起看，把每一步的变量名和类型在代码中找出来 | ~30min |

### 第 7 层：工具与评估 — 按需翻阅

| 顺序 | 文件 | 何时看 |
|---|---|---|
| 22 | [Utility/Point.py](Utility/Point.py) | `pixel2point_NED` 被 run_pair 频繁调用时回来看 |
| 23 | [Utility/Math.py](Utility/Math.py) | 想理解 `gaussian_mixture_mean_var` / `gaussain_full_kernels` 的数学细节时看 |
| 24 | [Utility/Timer.py](Utility/Timer.py) | 做性能分析时看 GPU/CPU 计时装饰器 |
| 25 | [Utility/Trajectory.py](Utility/Trajectory.py) + [Utility/Sandbox.py](Utility/Sandbox.py) | 需要管理实验输入输出时看 |
| 26 | [Evaluation/EvalSeq.py](Evaluation/EvalSeq.py) + [Evaluation/MetricsSeq.py](Evaluation/MetricsSeq.py) | 需要评估 ATE/RTE/ROE/RPE 指标时看 |

### 网络模型层 — 跳过，除非要改网络架构

`Module/Network/*` 下的所有文件对外面的 `FlowFormerCovFrontend` 只暴露一个 `inference(input_A, input_B) → (flow, cov)` 调用。如果不需要修改网络架构本身，不用细看。

### 建议的阅读节奏

- **第 1-2 层**（文件 1-4）：花 ~45 分钟过一遍，主要是记住类型名和字段名
- **第 3 层**（文件 5-6）：花 ~45 分钟仔细读，这是整个系统的"持久化层"
- **第 4 层**（文件 7-15）：每个文件 ~10-20 分钟，重点是接口的抽象方法签名 + 核心实现的算法逻辑
- **第 5 层**（文件 16-19）：最复杂的一层。建议先读 PyposeOptimizers(了解 LM 如何收 weight)，再读 Graphs(六种因子图的残差和协方差)，然后读 Optimizer(三步骤流程)，最后读 Interface(并行模式)
- **第 6 层**（文件 20-21）：有了前面 5 层的铺垫，MACVO.py 读起来会非常顺畅——每一步调用的模块和它们的数据流你都已经熟悉了

**总计约 4-5 小时**（不含网络模型层）。
