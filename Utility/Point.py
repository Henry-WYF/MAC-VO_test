import torch
import pypose as pp

# 坐标系说明：
#   NED (World): +x=North, +y=East, +z=Down
#   EDN (pypose): 默认输出 EDN 坐标 → 通过 roll/shift 转为 NED

def filterPointsInRange(pts1:torch.Tensor, u_range: tuple[int, int], v_range: tuple[int, int]) -> torch.Tensor:
    """过滤超出图像范围的关键点，返回布尔掩码 (N,)，True 表示在图像范围内"""
    u_min, u_max = u_range
    v_min, v_max = v_range

    u_selector = torch.logical_and(pts1[..., 0] < u_max, pts1[..., 0] > u_min)
    v_selector = torch.logical_and(pts1[..., 1] < v_max, pts1[..., 1] > v_min)
    selector = torch.logical_and(u_selector, v_selector)

    return selector

def pixel2point_NED(pixels: torch.Tensor, depths: torch.Tensor, intrinsics: torch.Tensor):
    # pp.pixel2point will output points in EDN coordinate, we will convert it to NED coord.
    return pp.pixel2point(pixels, depths, intrinsics).roll(shifts=1, dims=-1)

def point2pixel_NED(points: torch.Tensor, intrinsics: torch.Tensor):
    # pp.pixel2point will output points in EDN coordinate, we will convert it to NED coord.
    return pp.point2pixel(points.roll(shifts=-1, dims=-1), intrinsics)
