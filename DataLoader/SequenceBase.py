from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Callable, final, Generator, Final
from typing_extensions import Self

import numpy as np
import multiprocessing as mp
# torch.utils.data.IterableDataset：将序列暴露为可迭代数据集
# 相比于映射式数据集 Dataset，更适于流式数据加载，支持动态生成数据、无限长度序列等场景
from torch.utils.data import IterableDataset
from concurrent.futures import ThreadPoolExecutor

from Utility.Extensions import ConfigTestableSubclass
from Utility.PrettyPrint import Logger
from Utility.Config import build_dynamic_config
from .Interface import T_Data
from .Transform import IDataTransform

# 数据层抽象    SequenceBase 序列基类
# T_Data 类型变量，帧数据类型 DataFrame 及其子类
class SequenceBase(IterableDataset[T_Data], ABC, ConfigTestableSubclass):
    @classmethod
    def name(cls) -> str:
        """
        Assign a short name for the dataset class. By default will be the class name.
        Overwrite this function if you want to create a more readable name used in `name` field in config.
        """
        return cls.__name__

    @abstractmethod
    def __getitem__(self, local_index: int) -> T_Data: ...

    # No need to read further ### Implementation details below ############
    def __init__(self, length: int) -> None:
        super().__init__()
        self.origin_length: Final[int] = length
        self.indices = np.arange(0, length, 1)
    
    def get_index(self, local_index: int) -> int:
        """
        SequenceBase class supports masking / sampling of sequences.
        The 'actual index' refer to the index in the original sequence,
        In contrast with the 'logical index' (index after mask is applied) used by the user.
        """
        return self.indices[local_index].item()

    @final
    def clip(self, start_idx: int | None = None, end_idx: int | None = None, step: int | None = None) -> Self:
        """截取序列的子区间（类似列表切片），返回 self 支持链式调用"""
        self.indices = self.indices[start_idx:end_idx:step]
        return self

    def preload(self) -> "PreloadedSequence[T_Data]":
        """将整个序列预加载到内存中，使用多线程加速 I/O，避免运行时磁盘读取瓶颈"""
        return PreloadedSequence(self)

    def transform(self, actions: list[Callable[[T_Data,], T_Data]] | Callable[[T_Data,], T_Data]) -> "TransformSequence[T_Data] | Self":
        """对序列的每一帧应用数据变换（如缩放、裁剪、归一化），返回新的 TransformSequence"""
        if isinstance(actions, list) and len(actions) == 0: return self
        return TransformSequence(self, actions)

    def __len__(self) -> int:
        return self.indices.size
    
    def __iter__(self) -> Generator[T_Data, None, None]:
        for idx in range(len(self)): yield self[idx]

    def __repr__(self) -> str:
        return f"{self.name()}(orig_len={self.origin_length}, clip_len={len(self)})"
    
    # 变量 T_Data ,由 interface.py 定义，继承自 Collatable类，具有 collate 方法，支持批处理拼接
    @staticmethod
    def collate_fn(batch: list[T_Data]) -> T_Data:
        """
        Collate function for DataLoader.
        """
        return batch[0].collate(batch)
    
    @staticmethod
    def config_dict2ns(cfg: SimpleNamespace | dict[str, Any]) -> SimpleNamespace:
        if isinstance(cfg, SimpleNamespace): return cfg
        return build_dynamic_config(cfg)[0]


class PreloadedSequence(SequenceBase[T_Data]):
    """预加载序列包装器：将原序列的全部帧一次性加载到内存 frame buffer 中，__getitem__ 直接从 buffer 读取"""
    def __init__(self, generic_seq: SequenceBase[T_Data]):
        self.sequence = generic_seq
        
        Logger.write("info", f"Preloading {self.sequence}")
        with ThreadPoolExecutor(max_workers=2 * mp.cpu_count()) as exc:
            frames = list(exc.map(self.sequence.__getitem__, [_ for _ in range(len(self.sequence))]))
        self._framebuffer = frames
        super().__init__(len(self._framebuffer))

    def __getitem__(self, local_index: int) -> T_Data:
        index = self.get_index(local_index)
        return self._framebuffer[index]
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        raise KeyError("This sequence class should never be called in config directly. It is meant to be"
                       "implicitly created by .preload() method.")


class TransformSequence(SequenceBase[T_Data]):
    """变换序列包装器：在 __getitem__ 时依次应用数据变换链（如缩放→裁剪→归一化），惰性执行"""
    def __init__(self, original_seq: SequenceBase[T_Data],
                 actions: list[Callable[[T_Data,], T_Data]] | Callable[[T_Data,], T_Data]) -> None:
        super().__init__(len(original_seq))
        self.original_seq = original_seq
        self.actions: list[Callable[[T_Data,], T_Data]] = []
        if isinstance(actions, list):
            self.actions = actions
        else:
            self.actions = [actions]
    
    def __getitem__(self, local_index: int) -> T_Data:
        frame = self.original_seq[local_index]
        for action in self.actions: frame = action(frame)
        return frame
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        raise KeyError("This sequence class should never be called in config directly. It is meant to be"
                       "implicitly created by .transform(...) method.")


def smart_transform(seq: SequenceBase[T_Data], trans_cfg: SimpleNamespace | dict[str, Any] | list) -> SequenceBase[T_Data]:
    # 将配置字典转换为 SimpleNamespace 对象，方便属性访问（类访问）
    if isinstance(trans_cfg, dict):
        trans_cfg = build_dynamic_config(trans_cfg)[0]
    elif isinstance(trans_cfg, list):
        trans_cfg = [
            tcfg if isinstance(tcfg, SimpleNamespace) else build_dynamic_config(tcfg)[0]
            for tcfg in trans_cfg
        ]
    
    if isinstance(trans_cfg, list):
        transform_cfg = trans_cfg
    else:
        seq_type = seq.name()
        if not hasattr(trans_cfg, seq_type): return seq
        transform_cfg = getattr(trans_cfg, seq_type)
    
    Logger.write("info", "Using data transformation:\n- " + '\n- '.join([str(x) for x in transform_cfg]))
    trans_fn = [
        IDataTransform.instantiate(tcfg.type, tcfg.args)
        for tcfg in transform_cfg
    ]
    return seq.transform(trans_fn)
