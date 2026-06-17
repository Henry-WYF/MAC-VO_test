"""
双帧位姿图优化（Two-frame Pose Graph Optimization）的因子图定义。

本模块实现了 6 种因子图类型，覆盖两种实现方式 × 三种残差模型：

  **实现方式：**
    - Autodiff（自动微分）：由 PyTorch autograd 计算雅可比
    - Analytic（解析雅可比）：手动推导雅可比公式，避免计算图开销

  **残差模型：**
    - ICP（3D-3D 点云配准）：
        R = (T * p_cam) - p_world
        即估计位姿将当前帧的 3D 点变换到世界坐标，与地图点位置对齐
    - Reprojection（2D-3D 重投影）：
        R = proj(T⁻¹ * p_world) - kp_uv
        即将地图点反投影到当前帧像素平面，与观测像素对齐
    - Disparity（视差约束 = 重投影 + 深度）：
        R = [proj_error; disparity_error]
        在重投影基础上增加视差/深度维度的约束，利用双目信息

选择逻辑在 Optimizer.py 的 init_context() 中根据 config.graph_type 和 config.autodiff 分派。
"""

import torch
import pypose as pp
import typing as T
from dataclasses import dataclass

from Module.Map import MatchObs, PointNode
from Utility.Point import pixel2point_NED, point2pixel_NED
from ..PyposeOptimizers import AnalyticModule, FactorGraph


@dataclass
class GraphInput:
    """
    优化器的输入数据结构，包含双帧 PGO 所需的所有信息。

    字段说明：
      - frame_idx: 待优化帧的索引
      - from_idx:  参考帧的索引
      - init_motion: 待优化帧的初始位姿（SE3）
      - baseline: 双目基线
      - observations: 帧间匹配观测（MatchObs）
      - points: 对应的 3D 地图点（PointNode）
      - images_intrinsic: 相机内参 K (3x3)
      - edges_index: 边索引，用于多帧组合优化时选择对应的观测子集
      - device: 计算设备
    """
    frame_idx         : torch.Tensor
    from_idx          : torch.Tensor
    init_motion       : pp.LieTensor
    baseline          : torch.Tensor
    observations      : MatchObs
    points            : PointNode
    images_intrinsic  : torch.Tensor
    edges_index       : torch.Tensor
    device            : str


@dataclass
class GraphOutput:
    """
    优化器的输出数据结构。

      - motion: 优化后的位姿（SE3 的 7 维参数化）
      - from_idx / frame_idx: 用于 write_back 定位回写位置
    """
    motion   : torch.Tensor
    from_idx : torch.Tensor
    frame_idx: torch.Tensor


############## 优化因子图定义

class ICP_TwoframePGO(FactorGraph):
    """
    ICP（Iterative Closest Point）因子图 —— 3D-3D 点云对齐。

    残差：R = T * p_cam - p_world
    其中 p_cam 是当前帧相机坐标系下的 3D 点（由像素反投影得到），
    p_world 是地图点的世界坐标，T 是待优化帧的位姿。

    协方差传播：Σ_res = R * Σ_obs_cam * Rᵀ + Σ_point_world
    即观测协方差（相机系）经旋转 + 地图点协方差（世界系）。
    """
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        self.device                = graph_data.device
        self.init_motion           = graph_data.init_motion
        self.from_idx              = graph_data.from_idx
        self.frame_idx             = graph_data.frame_idx

        self.pose2opt       = pp.Parameter(pp.SE3(self.init_motion))
        self.edges_index    = graph_data.edges_index

        self.pts = graph_data.points
        self.obs = graph_data.observations

        # 相机内参
        self.register_buffer("K", graph_data.images_intrinsic)
        # 将 frame2 的像素坐标反投影到相机坐标系下的 3D 点
        self.register_buffer("points_Tc",
            pixel2point_NED(self.obs.data["pixel2_uv"], self.obs.data["pixel2_d"].squeeze(-1), graph_data.images_intrinsic)
        )
        self.points_Tc: torch.Tensor
        # 地图点在世界坐标系下的位置和协方差
        self.register_buffer("points_Tw", self.pts.data["pos_Tw"])
        # frame2 观测在相机坐标系下的协方差（由 Covariance2to3 模块计算）
        self.register_buffer("obs_covTc", self.obs.data["obs2_covTc"])
        self.register_buffer("pts_covTw", self.pts.data["cov_Tw"])


    def forward(self) -> torch.Tensor:
        """计算 ICP 残差：transform(points_cam) - points_world"""
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        return frame_pose.Act(self.points_Tc) - self.points_Tw

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        """
        计算每个残差的 3x3 协方差矩阵。

        Σ_res = R * Σ_obs_cam * Rᵀ + Σ_point_world
        即将相机坐标系下的观测协方差旋转到世界坐标系，再加上地图点自身的协方差。
        """
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        R  = frame_pose.rotation().matrix()
        RT = R.transpose(-2, -1)
        return (R @ self.obs_covTc @ RT) + self.pts_covTw # type: ignore

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        """将优化后的位姿写入输出结构"""
        return GraphOutput(motion=self.pose2opt, frame_idx=self.frame_idx, from_idx=self.from_idx)


class Reproj_TwoFramePGO(FactorGraph):
    """
    重投影（Reprojection）因子图 —— 2D-3D 像素对齐。

    残差：R = proj(T⁻¹ * p_world, K) - kp2_uv
    即将世界坐标系下的地图点反投影到当前帧，计算与观测像素坐标的差异。

    协方差：Σ_res = Σ_kp2（观测像素坐标的 2x2 协方差矩阵）。
    """
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        self.from_idx : torch.Tensor = graph_data.from_idx
        self.frame_idx: torch.Tensor = graph_data.frame_idx
        self.init_motion:  pp.LieTensor = graph_data.init_motion

        self.pose2opt       = pp.Parameter(pp.SE3(self.init_motion))
        self.edges_index    = graph_data.edges_index

        self.pts     = graph_data.points
        self.obs     = graph_data.observations

        self.pos_Tc: torch.Tensor
        self.pos_Tw: torch.Tensor
        self.K: torch.Tensor
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("pos_Tw" , self.pts.data["pos_Tw"])
        self.register_buffer("cov_Tw" , self.pts.data["cov_Tw"])
        self.register_buffer("kp2"    , self.obs.data["pixel2_uv"])

        # 构建 frame2 像素坐标的 2x2 协方差矩阵 [σ_uu, σ_uv; σ_uv, σ_vv]
        N = self.obs.data["pixel2_uv_cov"].size(0)
        cov_kp2 = torch.empty((N, 2, 2))
        cov_kp2[:, 0, 0] = self.obs.data["pixel2_uv_cov"][:, 0]
        cov_kp2[:, 1, 1] = self.obs.data["pixel2_uv_cov"][:, 1]
        cov_kp2[:, 0, 1] = self.obs.data["pixel2_uv_cov"][:, 2]
        cov_kp2[:, 1, 0] = self.obs.data["pixel2_uv_cov"][:, 2]
        self.register_buffer("cov_kp2", cov_kp2)

    def forward(self) -> torch.Tensor:
        """计算重投影残差：proj(inv_transform(p_world), K) - kp2"""
        self.pos_Tc = self.pose2opt.Inv().Act(self.pos_Tw)
        return point2pixel_NED(self.pos_Tc, self.K) - self.kp2

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        """返回观测像素坐标的 2x2 协方差矩阵作为每个残差的协方差"""
        return T.cast(torch.Tensor, self.cov_kp2)

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        with torch.no_grad():
            return GraphOutput(motion=self.pose2opt, frame_idx=self.frame_idx, from_idx=self.from_idx)


class ReprojDisp_TwoFramePGO(Reproj_TwoFramePGO):
    """
    视差约束（Reprojection + Disparity）因子图。

    在重投影残差（2 维）的基础上增加视差/深度约束（1 维），形成 3 维残差。
    残差 = [proj(T⁻¹ * p_world) - kp2, (bf/x) - disparity]
    其中 x 是地图点在相机坐标系下的 X 坐标（深度方向），bf 是基线×焦距。

    协方差：3x3 块对角矩阵 [Σ_kp2, 0; 0, σ_disp²]
    """
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)
        self.register_buffer("baseline", graph_data.baseline)
        self.baseline: torch.Tensor
        self.register_buffer("kp2_disparity", graph_data.observations.data["pixel2_disp"])

        cov_kp2 = T.cast(torch.Tensor, self.cov_kp2)

        # 构建 3x3 协方差矩阵：前 2x2 为像素协方差，最后 1x1 为视差协方差
        N = cov_kp2.size(0)
        cov = torch.zeros((N, 3, 3))
        cov[:, :2, :2] = cov_kp2
        cov[:, 2, 2] = graph_data.observations.data["pixel2_disp_cov"].squeeze(-1)
        self.register_buffer("cov", cov)

    def forward(self) -> torch.Tensor:
        """计算组合残差：[重投影误差 (2维); 视差误差 (1维)]"""
        self.pos_Tc = self.pose2opt.Inv() * self.pos_Tw
        K = T.cast(torch.Tensor, self.K)
        bl = T.cast(torch.Tensor, self.baseline)

        reproj_err = point2pixel_NED(self.pos_Tc, K) - T.cast(torch.Tensor, self.kp2)
        # 视差 = (基线 * 焦距) / 深度(X)  →  误差 = 预测视差 - 观测视差
        depth_err = (self.pos_Tc[:, 0:1].reciprocal() * (K[0, 0] * bl)) - self.kp2_disparity
        return torch.cat((reproj_err, depth_err), dim=-1)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        return T.cast(torch.Tensor, self.cov)


############## 解析雅可比版本（避免 autograd 开销）

class Analytic_ICP_TwoframePGO(ICP_TwoframePGO, AnalyticModule):
    """
    ICP 因子的解析雅可比版本。

    雅可比矩阵 J = ∂R/∂ξ ∈ R^(3E × 7)，对 SE3 的参数化求导：
      J = [I_3x3 | -[p_transformed]×]
    其中 [·]× 是叉积反对称矩阵（skew-symmetric matrix）。
    """
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        """构建 ICP 残差的解析雅可比矩阵 J = [I | -[T*p]×]"""
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        R = frame_pose.rotation().matrix()
        p = self.points_Tc
        E = p.shape[0]

        J = torch.zeros((E, 3, 7), device=p.device, dtype=p.dtype)

        # 平移部分：∂(T·p)/∂t = I
        I3 = torch.eye(3, device=p.device, dtype=p.dtype).unsqueeze(0)
        J[..., 0:3] = I3
        # 旋转部分：∂(T·p)/∂ω = -[T·p]×
        J[..., 3:6] = -pp.vec2skew(frame_pose.Act(p))

        # 展平为 (3E, 7) 形状供 LM 优化器使用
        return J.view(-1, 7)


class Analytic_Reproj_TwoFramePGO(Reproj_TwoFramePGO, AnalyticModule):
    """
    重投影因子的解析雅可比版本。

    使用链式法则：J = J_proj @ J_Tinv_p
      - J_proj:     ∂(proj)/∂p_cam ∈ R^(2×3)，像素坐标对相机坐标系的导数
      - J_Tinv_p:   ∂(T⁻¹·p_world)/∂ξ ∈ R^(3×7)，反投影点对 SE3 参数的导数
    """
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        assert self.pos_Tc is not None, "pos_Tc not found, need to call forward() before building jacobian."
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        assert self.K[0, 1] == 0, "K[0, 1] non-zero is currently not supported"

        x, y, z = self.pos_Tc[:, 0], self.pos_Tc[:, 1], self.pos_Tc[:, 2]
        x_square = x ** 2
        # J_proj: 投影函数 (u= fx*y/x + cx,  v= fy*z/x + cy) 对相机坐标 (x,y,z) 的雅可比
        J_homoKS = torch.zeros(self.pos_Tc.shape[0], 2, 3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_homoKS[:, 0, 0] = -fx * y / x_square  # ∂u/∂x
        J_homoKS[:, 0, 1] = fx / x               # ∂u/∂y
        J_homoKS[:, 1, 0] = -fy * z / x_square  # ∂v/∂x
        J_homoKS[:, 1, 2] = fy / x               # ∂v/∂z

        R = self.pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        # J_Tinv_p: T⁻¹·p_world 对 pose 参数的导数
        # 最后 2 列（6:7）在 pypose SE3 参数化中无用，故保留为 0
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)
        J_Tinv_p[..., :3] = -R_T              # ∂(T⁻¹·p)/∂t = -Rᵀ
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)  # ∂(T⁻¹·p)/∂ω = Rᵀ·[p_w]×

        J = (J_homoKS @ J_Tinv_p).view(-1, 7)
        return J


class Analytic_ReprojDisp_TwoFramePGO(ReprojDisp_TwoFramePGO, AnalyticModule):
    """
    视差约束因子的解析雅可比版本。

    在 Analytic_Reproj 的基础上增加视差维度的雅可比：
      J = [J_reproj (2×7); J_disp (1×7)]
    其中 J_disp = ∂(bf/x)/∂ξ = -(bf/x²) · ∂x/∂ξ
    """
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        assert self.pos_Tc is not None, "pos_Tc not found, need to call forward() before building jacobian."
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        assert self.K[0, 1] == 0, "K[0, 1] non-zero is currently not supported"

        x, y, z = self.pos_Tc[:, 0], self.pos_Tc[:, 1], self.pos_Tc[:, 2]
        x_square = x ** 2
        # J_proj: 同 Analytic_Reproj
        J_homoKS = torch.zeros(self.pos_Tc.shape[0], 2, 3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_homoKS[:, 0, 0] = -fx * y / x_square
        J_homoKS[:, 0, 1] = fx / x
        J_homoKS[:, 1, 0] = -fy * z / x_square
        J_homoKS[:, 1, 2] = fy / x
        R = self.pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        # J_Tinv_p: 同 Analytic_Reproj
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)
        J_Tinv_p[..., :3] = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)
        J_reproj = (J_homoKS @ J_Tinv_p)
        # J_disp: 视差 (bf/x) 对 pose 的导数 = -(bf/x²) * ∂x/∂ξ
        J_disp = (-(self.baseline * fx) / x_square).view(-1, 1, 1) * J_Tinv_p[:, 0:1, :]
        J = torch.cat((J_reproj, J_disp), dim=1).view(-1, 7)
        return J
