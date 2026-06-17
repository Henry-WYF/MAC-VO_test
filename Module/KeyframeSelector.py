import typing as T
from abc import ABC, abstractmethod
from types import SimpleNamespace

from Utility.Extensions import ConfigTestableSubclass
from DataLoader import StereoFrame, T_Data


class IKeyframeSelector(ABC, T.Generic[T_Data], ConfigTestableSubclass):
    """
    Keyframe selector - decide whether a frame is considered as "keyframe" for backend optimization.
    
    In current implementation of MAC-VO, non-keyframes will be linearly interpolated on termination of system.
    """
    def __init__(self, config: SimpleNamespace):
        self.config = config
    
    @abstractmethod
    def isKeyframe(self, frame: T_Data) -> bool: ...


class AllKeyframe(IKeyframeSelector[StereoFrame]):
    """每帧都是关键帧——不跳帧，精度最高但计算量最大"""
    def isKeyframe(self, frame: StereoFrame) -> bool:
        return True

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {})


class UniformKeyframe(IKeyframeSelector[StereoFrame]):
    """均匀关键帧选择：每 keyframe_freq 帧选一帧作为关键帧，其余帧位姿由插值生成"""
    def isKeyframe(self, frame: StereoFrame) -> bool:
        return (frame.frame_idx % self.config.keyframe_freq) == 0

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "keyframe_freq": lambda freq: isinstance(freq, int) and (freq >= 1)
        })
