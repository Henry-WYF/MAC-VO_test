import torch
import pypose as pp
import numpy  as np

import typing as T
from typing_extensions import Self
from dataclasses import dataclass
from itertools import chain

# 定义了一个泛型类型变量 Tp，它可被任何类型替换
Tp = T.TypeVar("Tp")
# 表示一个可调用对象，它接收一个 Tp 类型的序列（列表/元组等），返回单个 Tp 类型的值
CollateFn = T.Callable[[T.Sequence[Tp],], Tp]

@dataclass(kw_only=True)
class Collatable:

    # 关于 类变量ClassVar
    # 类变量 定义在类体中、方法之外，属于类本身，所有实例共享同一份数据； 
    # 实例变量通常在 __init__ 等实例方法中通过 self.变量名 定义，属于各自实例，互不影响
    collate_handlers: T.ClassVar[dict[str, CollateFn]] = dict()
    # 此处 collate_handlers 用于存储特定属性的自定义 collate 函数，允许用户为不同类型的数据指定不同的批处理方式
    # 如 对于 tensor 类型数据，不希望使用默认的拼接方式，可以在 collate_handlers 中为该属性指定一个新的 collate_fn 来覆盖默认行为
    #    对于特定属性（如 height 和 width），保留默认值（即 batch[0] ）
    
    # 关于 cls 与 self
    # cls 是 类方法 的第一个参数，代表类本身
    # self 是 实例方法 的第一个参数，代表类的实例
    @classmethod
    def collate(cls, batch: T.Sequence[Self]) -> Self:
        """
        A default collate function that will handle torch.Tensor, pp.LieTensor and
        np.array automatically. You can perform more customized collate by one of the following methods:
        
        1. Overriding the collate method
        
        2. Setting the class attribute `collate_handlers` to a dictionary that maps the attribute name to the collate function corresponding to that field.
        
        """
        data_dict = dict()

        # 遍历 batch 中第一个元素的属性（通过 __dict__.items() 获取），
        # 根据属性类型选择合适的 collate 函数进行批量拼接处理
        for key, value in batch[0].__dict__.items():
            if key in cls.collate_handlers:
                collate_fn = cls.collate_handlers[key]
            
            # lambda 匿名函数，相较于定义一个普通函数更简洁，适用于简单的函数逻辑
            # 关于维度：# dim=0 表示轴向，此处的轴向指时间轴 B，即将多个样本沿着时间轴拼接成一个批次
            elif isinstance(value, torch.Tensor):
                collate_fn = lambda seq: torch.cat(seq, dim=0)
            # LieTensor 是 pypose 中表示李群元素的张量类，通常用于表示位姿等几何变换
            # torch.cat 拼接；torch.stack 叠加；区别在于 cat 沿着指定维度拼接，而 stack 则在新维度上叠加
            elif isinstance(value, pp.LieTensor):
                collate_fn = lambda seq: torch.stack(seq, dim=0)
            elif isinstance(value, np.ndarray):
                collate_fn = lambda seq: np.concatenate(seq, axis=0)
            elif isinstance(value, list):
                collate_fn = lambda seq: list(chain.from_iterable(seq))
            elif isinstance(value, Collatable):
                collate_fn = value.collate
            elif value is None:
                collate_fn = lambda seq: None
            else:
                raise ValueError(f"Unsupported data type {type(value)}, you need to overrider the collate method.")
            data_dict[key] = cls._collate([getattr(x, key) for x in batch], collate_fn)
        return cls(**data_dict)
    
    @staticmethod
    def _collate(batch: T.Sequence[Tp | None], collate_fn: CollateFn) -> Tp | None:
        if any([x is None for x in batch]): return None
        return collate_fn([x for x in batch if x is not None])


@dataclass(kw_only=True)
class StereoData(Collatable):
    # Transformation from body frame to sensor frame
    T_BS: pp.LieTensor      # torch.float32, pp.SE3 of shape Bx7
    K   : torch.Tensor      # torch.float32 of shape Bx3x3
    baseline: torch.Tensor   # Baseline (m) between left and right camera, len(list) = B
    time_ns : list[int]     # Time (ns) of data received, len(list) = B
    height: int             # H
    width : int             # W
    
    # @property 装饰器将方法 frame_ns 转换为属性（实例/类？变量），
    # 使得可以通过 stereo_data.frame_ns 直接访问，而不需要调用 stereo_data.frame_ns()，提供了更简洁的接口
    @property
    def frame_ns(self) -> int:
        # 异常判断，条件为假时触发 AssertionError，并输出指定的错误信息
        assert len(self.time_ns) == 1, "Can only use frame_ns on unbatched data."
        return self.time_ns[0]
    @property
    def frame_ms(self) -> float: return self.frame_ns / 1000.
    @property
    def frame_baseline(self) -> float:
        assert self.baseline.size(0) == 1, "Can only use frame_baseline on unbatched data"
        return self.baseline.item()
    @property
    def frame_K(self) -> torch.Tensor:
        assert self.K.size(0) == 1, "Can only use frame_K on unbatched data"
        return self.K[0]
    
    @property
    def time_ms(self) -> list[float]: return [t / 1000. for t in self.time_ns]

    # 内参矩阵K的快捷访问属性
    @property
    def fx(self) -> float:
        assert self.K.size(0) == 1, "Can only use property shortcut on unbatched data"
        return self.K[0, 0, 0].item()
    @property
    def fy(self) -> float:
        assert self.K.size(0) == 1, "Can only use property shortcut on unbatched data"
        return self.K[0, 1, 1].item()
    @property
    def cx(self) -> float:
        assert self.K.size(0) == 1, "Can only use property shortcut on unbatched data"
        return self.K[0, 0, 2].item()
    @property
    def cy(self) -> float:
        assert self.K.size(0) == 1, "Can only use property shortcut on unbatched data"
        return self.K[0, 1, 2].item()
    
    # Sensor Data
    imageL: torch.Tensor    # torch.float32 of shape Bx3xHxW
    imageR: torch.Tensor    # torch.float32 of shape Bx3xHxW
    
    # Label & Ground Truth
    gt_flow  : torch.Tensor | None = None    # torch.float32 of shape Bx2xHxW 
    flow_mask: torch.Tensor | None = None    # torch.bool    of shape Bx1xHxW
    gt_depth : torch.Tensor | None = None    # torch.float32 of shape Bx1xHxW 
    
    collate_handlers = {
        "height": lambda batch: batch[0],
        "width" : lambda batch: batch[0],
    }


@dataclass(kw_only=True)
class IMUData(Collatable):
    """
    (N) IMU measurements from a certain period of time
    """
    # Transformation from body frame to sensor frame
    T_BS: pp.LieTensor          # torch.float32, pp.SE3 of shape Bx7
    time_ns: torch.Tensor       # torch.int64 of shape BxNx1
    gravity: list[float]        # gravity constant
    
    @property
    def time_delta(self) -> torch.Tensor: return self.time_ns[:, 1:] - self.time_ns[:, :-1]
    @property
    def time_ms(self) -> torch.Tensor   : return self.time_ns.double() / 1000.
    @property
    def frame_gravity(self) -> float:
        assert len(self.gravity) == 1, "frame_gravity can only be used on unbatched data"
        return self.gravity[0]
    
    # acc: Raw acceleration of IMU body frame with gravity added
    acc   : torch.Tensor                # torch.float32 of shape BxNx3
    # gyro: Angular rate of the IMU body frame
    gyro  : torch.Tensor                # torch.float32 of shape BxNx3


@dataclass(kw_only=True)
class AttitudeData(Collatable):
    # Transformation from body frame to sensor frame
    T_BS: pp.LieTensor          # torch.float32, pp.SE3 of shape Bx7
    time_ns: torch.Tensor       # torch.int64 of shape BxNx1
    gravity: list[float]        # gravity constant
    
    @property
    def time_delta(self) -> torch.Tensor: return self.time_ns[:, 1:] - self.time_ns[:, :-1]
    @property
    def time_ms(self) -> torch.Tensor   : return self.time_ns.double() / 1000.
    @property
    def frame_gravity(self) -> float:
        assert len(self.gravity) == 1, "frame_gravity can only be used on unbatched data"
        return self.gravity[0]
    
    # Ground truth velocity, position and rotation
    gt_vel: torch.Tensor      # torch.float32 of shape BxNx3
    gt_pos: torch.Tensor      # torch.float32 of shape BxNx3
    gt_rot: pp.LieTensor      # torch.float32 of shape BxNx4, pp.SO3 rotation.
    
    # Initial condition for IMU preintegration
    init_vel: torch.Tensor      # torch.float32 of shape Bx1x3
    init_pos: torch.Tensor      # torch.float32 of shape Bx1x3
    init_rot: pp.LieTensor      # torch.float32 of shape Bx1x4, pp.SO3 rotation.


@dataclass(kw_only=True)
class DataFrame(Collatable):
    """所有传感器帧的基类：包含帧索引、时间戳和可选的 ground truth 位姿"""
    idx: list[int]
    gt_pose  : pp.LieTensor | None = None   # pp.SE3 of shape Bx7, ground truth 机体位姿

    time_ns  : list[int]                     # 时间戳（纳秒）
    
    @property
    def frame_idx(self) -> int:
        assert len(self.idx) == 1, "frame_idx property is only valid on unbatched data"
        return self.idx[0]

    @property
    def frame_time_ns(self) -> int:
        assert len(self.time_ns) == 1, "frame_time_ns property is only valid on unbatched data"
        return self.time_ns[0]

# 定义了一个泛型类型变量 T_Data，它被限制为 DataFrame 类或其子类的实例
T_Data = T.TypeVar("T_Data", bound=DataFrame)

@dataclass(kw_only=True)
# T.Generic[T_Data] 泛型基类，允许 DataFramePair 类在实例化时指定 T_Data 的具体类型（如 StereoFrame 或 StereoInertialFrame）
class DataFramePair(DataFrame, T.Generic[T_Data]):
    """帧对：同时持有 cur（当前帧）和 nxt（下一帧），用于帧间光流估计"""
    cur : T_Data
    nxt : T_Data

@dataclass(kw_only=True)
class StereoFrame(DataFrame):
    """双目帧：包含一帧双目图像数据 (StereoData)"""
    stereo   : StereoData

@dataclass(kw_only=True)
class StereoInertialFrame(StereoFrame):
    """双目+IMU 帧：在双目帧基础上附加 IMU 数据"""
    imu        : IMUData
    gt_attitude: AttitudeData | None = None
