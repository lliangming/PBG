import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MLP(nn.Module):
    """Linear embedding used by SegFormer-style decode heads."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


@HEADS.register_module()
class GasPhysEdgeHead(BaseDecodeHead):
    """Edge-aware anisotropic diffusion head for infrared gas segmentation."""

    def __init__(self,
                 feature_strides,
                 edge_channels=64,
                 edge_gate_scale=1.0,
                 edge_refine_scale=0.25,
                 diffusion_iters=2,
                 diffusion_rate=0.15,
                 edge_loss_weight=0.1,
                 edge_pos_weight=4.0,
                 **kwargs):
        super(GasPhysEdgeHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.edge_gate_scale = edge_gate_scale
        self.edge_refine_scale = edge_refine_scale
        self.diffusion_iters = diffusion_iters
        self.diffusion_rate = diffusion_rate
        self.edge_loss_weight = edge_loss_weight
        self.edge_pos_weight = edge_pos_weight
        self.visualize_debug = False

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = (
            self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.edge_pred = nn.Sequential(
            ConvModule(
                in_channels=embedding_dim,
                out_channels=edge_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg),
            nn.Conv2d(edge_channels, 1, kernel_size=1)
        )

        self.edge_refine = ConvModule(
            in_channels=embedding_dim + 1,
            out_channels=embedding_dim,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg)

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.reset_edge_parameters()

    def reset_edge_parameters(self):
        nn.init.constant_(self.edge_pred[-1].weight, 0.0)
        nn.init.constant_(self.edge_pred[-1].bias, -2.0)
        self.edge_debug = {}

    def _feature_edge_prior(self, feat):
        gray = feat.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
        edge_scale = edge_mag.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return torch.sigmoid(edge_mag / edge_scale - 1.0)

    def _anisotropic_diffusion(self, logits, edge_gate):
        if self.diffusion_iters <= 0 or self.diffusion_rate <= 0:
            return logits

        conductance = (1.0 - edge_gate).clamp(0.0, 1.0)
        for _ in range(self.diffusion_iters):
            smoothed = F.avg_pool2d(logits, kernel_size=3, stride=1, padding=1)
            logits = logits + self.diffusion_rate * conductance * (
                smoothed - logits)
        return logits

    def forward_with_edge(self, inputs):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n = c4.shape[0]

        c4_embed = self.linear_c4(c4).permute(0, 2, 1).reshape(
            n, -1, c4.shape[2], c4.shape[3])
        c4_embed = resize(
            c4_embed, size=c1.size()[2:], mode='bilinear',
            align_corners=False)

        c3_embed = self.linear_c3(c3).permute(0, 2, 1).reshape(
            n, -1, c3.shape[2], c3.shape[3])
        c3_embed = resize(
            c3_embed, size=c1.size()[2:], mode='bilinear',
            align_corners=False)

        c2_embed = self.linear_c2(c2).permute(0, 2, 1).reshape(
            n, -1, c2.shape[2], c2.shape[3])
        c2_embed = resize(
            c2_embed, size=c1.size()[2:], mode='bilinear',
            align_corners=False)

        c1_embed = self.linear_c1(c1).permute(0, 2, 1).reshape(
            n, -1, c1.shape[2], c1.shape[3])

        fused = self.linear_fuse(
            torch.cat([c4_embed, c3_embed, c2_embed, c1_embed], dim=1))

        edge_logit = self.edge_pred(fused)
        edge_prior = self._feature_edge_prior(c1_embed)
        edge_gate = torch.sigmoid(
            edge_logit + self.edge_gate_scale * (edge_prior - 0.5))
        self.edge_debug = dict(
            edge_logit=edge_logit.detach(),
            edge_gate=edge_gate.detach(),
            edge_prior=edge_prior.detach())

        logits_before_bgfr = (
            self.linear_pred(fused)
            if self.visualize_debug and not self.training else None)

        edge_feat = self.edge_refine(torch.cat([fused, edge_gate], dim=1))
        fused = fused + self.edge_refine_scale * edge_gate * edge_feat

        if self.dropout is not None:
            fused = self.dropout(fused)
        logits = self.linear_pred(fused)
        logits_before_bld = logits
        logits = self._anisotropic_diffusion(logits, edge_gate)
        if self.visualize_debug and not self.training:
            self.edge_debug.update(
                logits_before_bgfr=logits_before_bgfr.detach(),
                logits_before_bld=logits_before_bld.detach(),
                logits_after_bld=logits.detach())
        return logits, edge_logit

    def forward(self, inputs):
        logits, _ = self.forward_with_edge(inputs)
        return logits

    def _edge_target(self, gt_semantic_seg):
        if gt_semantic_seg.dim() == 5:
            seg_label = gt_semantic_seg[:, -1]
        else:
            seg_label = gt_semantic_seg
        if seg_label.dim() == 4 and seg_label.size(1) == 1:
            seg_label = seg_label.squeeze(1)

        valid = seg_label != self.ignore_index
        gas_mask = (seg_label == 1).float().unsqueeze(1)
        dilated = F.max_pool2d(gas_mask, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool2d(
            1.0 - gas_mask, kernel_size=3, stride=1, padding=1)
        edge = (dilated - eroded).clamp_min(0.0)
        return edge, valid.float().unsqueeze(1)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits, edge_logit = self.forward_with_edge(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)

        if self.edge_loss_weight > 0:
            edge_target, valid_mask = self._edge_target(gt_semantic_seg)
            edge_logit = resize(
                input=edge_logit,
                size=edge_target.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            pos_weight = edge_logit.new_tensor(self.edge_pos_weight)
            edge_loss = F.binary_cross_entropy_with_logits(
                edge_logit,
                edge_target,
                reduction='none',
                pos_weight=pos_weight)
            edge_loss = (edge_loss * valid_mask).sum() / (
                valid_mask.sum() + 1e-6)
            losses['loss_edge'] = self.edge_loss_weight * edge_loss

        return losses
