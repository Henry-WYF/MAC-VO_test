import torch
import torch.nn as nn

from ..FlowFormer.core.gru import SepConvGRU
from ..FlowFormer.core.decoder import MemoryDecoder, initialize_flow


class CovHead(nn.Module):
    """协方差预测头：通过 4 层卷积 (ReLU+Conv) 将隐藏特征映射为 2 通道对数方差 (log-variance) 输出。

    输出 2 通道分别对应 σ²_u 和 σ²_v（光流 u、v 方向的对数方差）。
    """
    def __init__(self, input_dim=128, hidden_dim=256):
        super(CovHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim // 4, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2(self.relu(self.conv1(x)))
        x = self.conv4(self.relu(self.conv3(x)))
        return x


class CovUpdateBlock(nn.Module):
    """协方差更新模块：结合 GRU 迭代更新和 CovHead 预测，输出协方差的增量 delta_cov。

    内部包含：
      - SepConvGRU: 对协方差隐藏状态做门控更新
      - CovHead: 从隐藏状态生成 delta_cov（对数方差空间）
      - mask: 生成上采样掩码（用于后续光流/协方差上采样）
    """
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.gru = SepConvGRU(
            hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim
        )
        self.cov_head = CovHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, covs_net, inp_cat):
        covs_net = self.gru(covs_net, inp_cat)
        delta_covs = self.cov_head(covs_net)
        mask = 0.25 * self.mask(covs_net)
        return covs_net, delta_covs, mask


class MemoryCovDecoder(MemoryDecoder):
    """扩展的 MemoryDecoder：在标准 FlowFormer 解码器的基础上增加协方差解码分支。

    与 MemoryDecoder 共享 flow_net 和 GRU 的 inp_cat 输入，通过独立的 CovUpdateBlock
    并行预测协方差的迭代更新。两个分支共享编码特征但不共享更新权重。
    """
    def __init__(self, cfg, decoder_dtype: torch.dtype):
        super(MemoryCovDecoder, self).__init__(cfg)
        self.cov_update = CovUpdateBlock(self.cfg, hidden_dim=128)
        
        self.decoder_dtype = decoder_dtype
        
        self.delta              = self.delta.to(dtype=self.decoder_dtype)
        self.att                = self.att.to(dtype=self.decoder_dtype)
        self.decoder_layer      = self.decoder_layer.to(dtype=self.decoder_dtype)
        self.flow_token_encoder = self.flow_token_encoder.to(dtype=self.decoder_dtype)
        self.update_block       = self.update_block.to(dtype=self.decoder_dtype)
        self.cov_update         = self.cov_update.to(dtype=self.decoder_dtype)

    def forward(self, cost_memory, context, cost_maps):  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        memory: [B*H1*W1, H2'*W2', C]
        context: [B, D, H1, W1]
        """
        cost_memory = cost_memory.to(dtype=self.decoder_dtype)
        
        flow_coords0, flow_coords1 = initialize_flow(context)        
        cov_coords0 , cov_coords1  = flow_coords0, flow_coords1.clone()
        
        flow_predictions = []
        cov_predictions = []

        context   = self.proj(context)
        flow_net, flow_inp  = torch.split(context, [128, 128], dim=1)
        flow_net = flow_net.tanh().to(dtype=self.decoder_dtype)
        fcov_net = flow_net.clone().to(dtype=self.decoder_dtype)
        flow_inp = flow_inp.relu()
        
        flow_inp  = flow_inp.to(dtype=self.decoder_dtype)
        attention = self.att(flow_inp)

        size = flow_net.shape
        key, value = None, None

        for _ in range(self.depth):
            flow_coords1      = flow_coords1.detach()
            bf16_flow_coords1 = flow_coords1.to(dtype=self.decoder_dtype)
            flow              = (flow_coords1 - flow_coords0).to(dtype=self.decoder_dtype)
            
            with torch.cuda.nvtx.range("Encode Flow Token"):
                # NOTE: This module MUST run in fp32 precision
                cost_forward = self.encode_flow_token(cost_maps, flow_coords1)
                cost_forward = cost_forward.to(dtype=self.decoder_dtype)
            
            with torch.cuda.nvtx.range("CNN Encoder"):
                query = self.flow_token_encoder(cost_forward)
                query = query.permute(0, 2, 3, 1).view(size[0] * size[2] * size[3], 1, self.dim)
            
            with torch.cuda.nvtx.range("Cross Attention"):
                cost_global, key, value = self.decoder_layer(
                    query, key, value, cost_memory, bf16_flow_coords1, size, self.cfg.query_latent_dim
                )
                corr = torch.cat([cost_global, cost_forward], dim=1)

            with torch.cuda.nvtx.range("GMA Update Block"):
                motion_feat        = self.update_block.encoder(flow, corr)
                motion_feat_global = self.update_block.aggregator(attention, motion_feat)
            
            inp_cat = torch.cat([flow_inp, motion_feat, motion_feat_global], dim=1)
            
            with torch.cuda.nvtx.range("Flow Update Block"):
                flow_net    = self.update_block.gru(flow_net, inp_cat)
                delta_flow  = self.update_block.flow_head(flow_net)
                up_mask     = self.update_block.mask(flow_net)
                
            with torch.cuda.nvtx.range("Cov Update Block"):
                fcov_net, delta_cov, cov_mask = self.cov_update(fcov_net, inp_cat)    
            
            with torch.cuda.nvtx.range("Flow Upsample"):
                # NOTE: This module MUST run in fp32 precision.
                delta_flow = delta_flow.to(dtype=torch.float32)
                up_mask    = 0.25 * up_mask.to(dtype=torch.float32)
                
                flow_coords1 = flow_coords1 + delta_flow
                flow_up      = self.upsample_flow(flow_coords1 - flow_coords0, up_mask)
                flow_predictions.append(flow_up)
            
            with torch.cuda.nvtx.range("Cov Upsample"):
                # NOTE: This module MUST run in fp32 precision.
                delta_cov = delta_cov.to(dtype=torch.float32)
                cov_mask  = cov_mask.to(dtype=torch.float32) 
                
                cov_coords1 = cov_coords1 + delta_cov
                cov_up = self.upsample_flow(cov_coords1 - cov_coords0, cov_mask)
                cov_predictions.append(cov_up)
            
        if self.training:
            return flow_predictions, cov_predictions
        else:
            return (flow_predictions[-1], flow_coords1 - flow_coords0), (cov_predictions[-1], cov_coords1 - cov_coords0)
