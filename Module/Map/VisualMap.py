import typing as T
import torch
import numpy as np
from typing_extensions import Self

from Utility.Extensions import AutoScalingTensor
from .Graph import Scaling_DenseEdge_Multi, Scaling_SparseEdge_Multi, Scaling_SingleEdge

# 因子图存储的节点/边类型定义
from .Template   import (
    FrameStore, MatchStore , PointStore,
    FrameNode , MatchObs, PointNode ,
)

class VisualMap:
    """
    MAC-VO 的全局因子图（Factor Graph）数据结构，是系统的核心存储单元。

    维护三种节点（Frame/Point/Match）及其之间的有向边关系，表示双目视觉里程计中的：
      - **FrameNode**:   关键帧，存储相机位姿、内参、基线、时间戳
      - **MatchObs**:    帧间匹配观测，存储像素坐标、深度、视差及其协方差
      - **PointNode**:   3D 路标点，存储世界坐标下的位置和协方差

    边拓扑关系（图结构）：
      - frame2match:   Frame → Match  (一对多，记录每帧产生了哪些匹配)
      - match2frame1/2: Match → Frame  (一对一，记录每条匹配关联的两个帧)
      - match2point:   Match → Point   (一对一，记录每条匹配对应的 3D 点)
      - point2match:   Point → Match   (一对多，记录每个 3D 点被哪些匹配观测到)
      - frame2map:     Frame → Point   (一对多，记录每帧产生的稠密地图点)

    所有存储使用 AutoScalingTensor，会根据数据量自动扩容。
    """
    def __init__(self) -> None:
        self.init_size: T.Final[int]  = 1024
        self.max_pt_obs: T.Final[int] = 5
        self.max_frame_range: T.Final[int] = 2
        
        self.frames = FrameStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "K"          : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
                "baseline"   : AutoScalingTensor((self.init_size,     ), grow_on=0, dtype=torch.float32),
                "pose"       : AutoScalingTensor((self.init_size, 7   ), grow_on=0, dtype=torch.float32),
                "T_BS"       : AutoScalingTensor((self.init_size, 7   ), grow_on=0, dtype=torch.float32),
                "need_interp": AutoScalingTensor((self.init_size,     ), grow_on=0, dtype=torch.bool),
                "time_ns"    : AutoScalingTensor((self.init_size,     ), grow_on=0, dtype=torch.long)
            }
            # 此处 data 是python中的 字典 数据结构，在数据聚合上类似于C++中定义一个结构体/类，
            # 但是是动态结构，类似于维护一个动态数组
        )
        
        self.points = PointStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "pos_Tw" : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "cov_Tw" : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "color"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.uint8)
            }
        )
        
        self.map_points = PointStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "pos_Tw" : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "cov_Tw" : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "color"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.uint8)
            }
        )

        self.match = MatchStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "pixel1_uv"      : AutoScalingTensor((self.init_size, 2   ), grow_on=0, dtype=torch.float32),
                "pixel2_uv"      : AutoScalingTensor((self.init_size, 2   ), grow_on=0, dtype=torch.float32),
                "pixel1_d"       : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_d"       : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel1_disp"    : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_disp"    : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel1_disp_cov": AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_disp_cov": AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "obs1_covTc"     : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "obs2_covTc"     : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "pixel1_uv_cov"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "pixel2_uv_cov"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "pixel1_d_cov"   : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_d_cov"   : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32)
            }
        )

        self.frame2match  = Scaling_DenseEdge_Multi(self.init_size, self.max_frame_range)
        self.frame2map    = Scaling_DenseEdge_Multi(self.init_size, self.max_frame_range)
        self.match2frame1 = Scaling_SingleEdge(self.init_size)
        self.match2frame2 = Scaling_SingleEdge(self.init_size)
        self.match2point  = Scaling_SingleEdge(self.init_size)
        self.point2match  = Scaling_SparseEdge_Multi(self.init_size, self.max_pt_obs)
        
        self.frames.register_edge(self.frame2map)
        self.frames.register_edge(self.frame2match)
        self.points.register_edge(self.point2match)
        self.match.register_edge(self.match2point)
        self.match.register_edge(self.match2frame1)
        self.match.register_edge(self.match2frame2)
        

    def get_frame2match(self, frame: FrameNode) -> MatchObs:
        """遍历 frame2match 边，返回该帧产生的所有 MatchObs 观测"""
        return self.match[self.frame2match.project(frame.index)]

    def get_match2point(self, match: MatchObs) -> PointNode:
        """遍历 match2point 边，返回该匹配对应的 3D 路标点"""
        return self.points[self.match2point.project(match.index)]

    def get_point2match(self, point: PointNode) -> MatchObs:
        """遍历 point2match 边，返回观测到该 3D 点的所有 MatchObs（最多 max_pt_obs 条）"""
        return self.match[self.point2match.project(point.index)]

    def get_match2frame1(self, match: MatchObs) -> FrameNode:
        """遍历 match2frame1 边，返回该匹配中 frame1 的帧节点"""
        return self.frames[self.match2frame1.project(match.index)]

    def get_match2frame2(self, match: MatchObs) -> FrameNode:
        """遍历 match2frame2 边，返回该匹配中 frame2 的帧节点"""
        return self.frames[self.match2frame2.project(match.index)]

    def get_frame2map(self, frame: FrameNode) -> PointNode:
        """遍历 frame2map 边，返回该帧产生的稠密地图点（仅在 mapping 模式下有效）"""
        return self.map_points[self.frame2map.project(frame.index)]

    def serialize(self) -> dict[str, np.ndarray]:
        """将整个因子图序列化为 numpy 数组字典，用于保存到 .npz 文件"""
        return (
            self.frames.serialize("frames/")
          | self.points.serialize("points/")
          | self.match.serialize("match/")
          | self.frame2match.serialize("edge/frame2match")
          | self.point2match.serialize("edge/point2match")
          | self.match2point.serialize("edge/match2point")
          | self.match2frame1.serialize("edge/match2frame1")
          | self.match2frame2.serialize("edge/match2frame2")
          | self.frame2map.serialize("edge/frame2map")
        )
    
    @classmethod
    def deserialize(cls, value: dict[str, np.ndarray]) -> Self:
        """从 serialize() 输出的 numpy 数组字典重建 VisualMap 对象"""
        map = cls()
        map.frames = map.frames.deserialize("frames/", value)
        map.match  = map.match.deserialize("match/", value)
        map.points = map.points.deserialize("points/", value)
        
        map.frame2match  = map.frame2match.deserialize("edge/frame2match", value)
        map.point2match  = map.point2match.deserialize("edge/point2match", value)
        map.match2point  = map.match2point .deserialize("edge/match2point", value)
        map.match2frame1 = map.match2frame1.deserialize("edge/match2frame1", value)
        map.match2frame2 = map.match2frame2.deserialize("edge/match2frame2", value)
        map.frame2map    = map.frame2map.deserialize("edge/frame2map", value)
        return map

    def __repr__(self) -> str:
        return f"VisualMap(#frame={len(self.frames)}, #point={len(self.points)}, #map={len(self.map_points)})"
