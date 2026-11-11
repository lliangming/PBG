import torch
import torch.nn as nn
import torch.nn.functional as F


class TDCR_Module(nn.Module):
    """
    自适应多帧时序差分重构模块
    输入: [B, T, C, H, W]
    输出: [B, C, H, W] (增强后的运动特征)
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 时序特征投影层：处理每一路差分特征
        self.diff_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        # 时序注意力分配：决定哪一帧的差分对当前分割贡献最大
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B*T_history, C, 1, 1]
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),  # 输出单一权重
            nn.Sigmoid()
        )

        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x_seq):
        """
        x_seq: [B, T, C, H, W]
        """
        B, T, C, H, W = x_seq.shape

        x_curr = x_seq[:, -1, ...]  # 当前帧 [B, C, H, W]
        x_history = x_seq[:, :-1, ...]  # 历史帧 [B, T-1, C, H, W]
        T_hist = T - 1

        # 1. 计算差分序列 (Broadcasting)
        # 将 x_curr 扩展维度以便相减：[B, 1, C, H, W]
        diffs = x_history - x_curr.unsqueeze(1)  # [B, T-1, C, H, W]

        # 2. 空间特征提取
        # 将 B 和 T 合并，并行处理所有差分图
        diffs = diffs.view(B * T_hist, C, H, W)
        diff_feats = self.diff_conv(diffs)  # [B*T-1, C, H, W]

        # 3. 计算时序注意力权重
        # 动态决定哪些历史帧提供的运动信息更可靠
        weights = self.temporal_attention(diff_feats)  # [B*T-1, 1, 1, 1]

        # 4. 加权融合
        weighted_diffs = diff_feats * weights
        weighted_diffs = weighted_diffs.view(B, T_hist, C, H, W)

        # 对时间维度求和/平均，还原为 [B, C, H, W]
        motion_feat = torch.sum(weighted_diffs, dim=1)

        return self.out_proj(motion_feat)
