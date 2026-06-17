import typing as T
from .Graph import TensorBundle, AutoScalingBundle

# 定义因子图中各节点存储的特征字段（字段名 + 形状/类型约束）

# FrameNode: 关键帧节点，存储相机位姿、内参、基线、时间戳
FrameFeature = T.Literal[
    "K",            # Nx3x3 , dtype=float32
    "baseline",     # Nx1   , dtype=float32
    "pose",         # Nx7   , dtype=float32, pose of sensor under world frame.
    "T_BS",         # Nx7   , dtype=float32, body-to-sensor SE3 transformation.
    "need_interp",  # Nx1   , dtype=bool
    "time_ns"       # Nx1   , dtype=long
]

# MatchObs: 帧间匹配观测，存储两个像素坐标、深度、视差及其协方差
MatchingFeature = T.Literal[
    "pixel1_uv",    # Nx2   , dtype=float32
    "pixel1_d",     # Nx1   , dtype=float32
    "pixel2_uv",    # Nx2   , dtype=float32
    "pixel2_d",     # Nx1   , dtype=float32
    "pixel1_disp",  # Nx1   , dtype=float32
    "pixel2_disp",  # Nx1   , dtype=float32
    "pixel1_uv_cov",# Nx3   , dtype=float32, (\sigma_uu, \sigma_vv, \sigma_uv)
    "pixel2_uv_cov",# Nx3   , dtype=float32, (\sigma_uu, \sigma_vv, \sigma_uv)
    "pixel1_d_cov" ,# Nx1   , dtype=float32
    "pixel2_d_cov" ,# Nx1   , dtype=float32
    "pixel1_disp_cov",    # Nx1   , dtype=float32
    "pixel2_disp_cov",    # Nx1   , dtype=float32
    "obs1_covTc",   # Nx3x3 , dtype=float64
    "obs2_covTc",   # Nx3x3 , dtype=float64
]

# PointNode: 3D 路标点，存储世界坐标位置、协方差矩阵、颜色
PointFeature = T.Literal[
    "pos_Tw",       # Nx3   , dtype=float32
    "cov_Tw",       # Nx3x3 , dtype=float64
    "color" ,       # Nx3   , dtype=uint8
]


# 类型别名：区分"单条记录"（TensorBundle，用于查询/切片）和"存储容器"（AutoScalingBundle，用于累积数据）

FrameNode    = TensorBundle[FrameFeature]       # 单个 / 一小组关键帧
FrameStore   = AutoScalingBundle[FrameFeature]  # 全局关键帧存储（自动扩容）

MatchObs     = TensorBundle[MatchingFeature]    # 单批帧间匹配观测
MatchStore   = AutoScalingBundle[MatchingFeature]  # 全局匹配存储（自动扩容）

PointNode    = TensorBundle[PointFeature]       # 单批 3D 路标点
PointStore   = AutoScalingBundle[PointFeature]  # 全局路标点存储（自动扩容）
