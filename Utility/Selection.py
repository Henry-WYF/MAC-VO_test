from __future__ import annotations

import torch
import torch.nn.functional as F


def local_minimum_nms(
    quality_map: torch.Tensor,
    kernel_size: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return local minima; an optional mask excludes invalid values before pooling."""
    if quality_map.ndim == 3:
        quality_map = quality_map.unsqueeze(1)
    if quality_map.ndim != 4 or quality_map.shape[1] != 1:
        raise ValueError(f"quality map must have shape Bx1xHxW, got {tuple(quality_map.shape)}")
    if kernel_size <= 0 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    # The no-mask path intentionally matches the historical selector exactly,
    # including max-pooling behavior around NaNs.
    if valid_mask is None:
        eroded = -F.max_pool2d(-quality_map, kernel_size, stride=1, padding=kernel_size // 2)
        return (quality_map == eroded) & ~quality_map.isnan()
    valid = ~quality_map.isnan() & valid_mask.bool()
    q_for_nms = torch.where(valid, quality_map, torch.full_like(quality_map, torch.inf))
    eroded = -F.max_pool2d(-q_for_nms, kernel_size, stride=1, padding=kernel_size // 2)
    return (q_for_nms == eroded) & valid
