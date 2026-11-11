# ---------------------------------------------------------------
# Fused Temporal SegFormer Backbone
# Early fusion: temporal features injected as extra input channels
# to the MIT backbone, removing the need for parallel auxiliary
# paths and cross-attention fusion modules.
#
# Architecture:
#   Input [B, T, 3, H, W]
#     -> Temporal Encoder (temporal_proj + TDCR + DeformableAlign)
#     -> Concat with last-frame RGB
#     -> MIT backbone (patch_embed1 with expanded in_chans)
#     -> 4-scale features
# ---------------------------------------------------------------
import torch

import torch.nn as nn

import math

import warnings

from timm.models.layers import trunc_normal_

from mmseg.utils import get_root_logger

from .Temporal import TDCR_Module

from .gasmit import Block, OverlapPatchEmbed

class DeformableAlign(nn.Module):
    """Motion-aware spatial alignment via deformable convolution.

    A small offset predictor estimates 2D sampling offsets, and
    DeformConv2d warps the TDCR motion features to align with the
    current frame's spatial coordinates.

    DCN weights are zero-initialized so temporal features contribute
    nothing early in training and gradually activate as the offset
    predictor learns useful alignments.

    Falls back to a residual conv block if mmcv.ops is unavailable.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.use_deform = self._check_deform_available()

        if self.use_deform:
            self.offset_conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 2 * 3 * 3, 3, 1, 1),
            )

            from mmcv.ops import DeformConv2d
            self.deform_conv = DeformConv2d(
                channels, channels, kernel_size=3, padding=1, bias=False
            )
            self.reset_deform_parameters()
        else:
            warnings.warn(
                'mmcv.ops.DeformConv2d not available. '
                'Falling back to residual conv block for DeformableAlign.'
            )
            self.refine = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _check_deform_available():
        try:
            from mmcv.ops import DeformConv2d
            return True
        except (ImportError, ModuleNotFoundError, AttributeError):
            return False

    def forward(self, x):
        if self.use_deform:
            offset = self.offset_conv(x)
            x = self.deform_conv(x, offset)
            return x
        else:
            return self.refine(x) + x

    def reset_deform_parameters(self):
        if not self.use_deform:
            return
        nn.init.constant_(self.offset_conv[-1].weight, 0.0)
        if self.offset_conv[-1].bias is not None:
            nn.init.constant_(self.offset_conv[-1].bias, 0.0)
        nn.init.constant_(self.deform_conv.weight, 0.0)


class FusedMIT(nn.Module):
    """Fused Temporal SegFormer backbone.

    Temporal features from T frames are extracted via per-frame
    projection, TDCR motion encoding, and deformable alignment.
    They are then concatenated with the last frame's RGB channels
    and fed into a standard MIT backbone through an expanded
    patch embedding layer.

    This eliminates the parallel auxiliary paths and expensive
    cross-attention fusion modules of TriPathMIT while preserving
    temporal awareness.

    Params (B1, temporal_dim=64): ~14.8M  (vs ~24.5M TriPathMIT B1)
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=(64, 128, 320, 512),
                 num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1),
                 temporal_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        temporal_dim = temporal_dim or embed_dims[0]
        self.temporal_dim = temporal_dim

        # ---- Temporal Encoder ----
        self.temporal_proj = nn.Sequential(
            nn.Conv2d(in_chans, temporal_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(temporal_dim),
            nn.ReLU(inplace=True)
        )
        self.tdc_encoder = TDCR_Module(temporal_dim)
        self.deform_align = DeformableAlign(temporal_dim)

        # ---- Expanded patch embedding (RGB + temporal channels) ----
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4,
            in_chans=in_chans + temporal_dim,
            embed_dim=embed_dims[0]
        )

        # ---- Standard patch embeddings for stages 2-4 ----
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2,
            in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )

        # ---- MIT Transformer blocks ----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)
        self.deform_align.reset_deform_parameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        """Load pretrained weights with expanded patch_embed1 handling.

        Pretrained MIT checkpoints have patch_embed1.proj.weight of shape
        [E, 3, 7, 7]. Our model expects [E, 3+C_t, 7, 7].

        Strategy:
        - RGB channels (0:3): copied directly from pretrained
        - Extra channels (3:): initialized with the channel-wise mean
          of the RGB weights (preserves spatial filter scale)
        - All other matching weights: loaded as-is
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            checkpoint = torch.load(pretrained, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)

            # Handle expanded patch_embed1
            key = 'patch_embed1.proj.weight'
            if key in state_dict:
                old_w = state_dict[key]
                new_w = self.patch_embed1.proj.weight
                if old_w.shape != new_w.shape:
                    with torch.no_grad():
                        # Copy RGB channels
                        new_w[:, :3].copy_(old_w)
                        # Extra channels: mean of RGB weights
                        new_w[:, 3:].copy_(old_w.mean(dim=1, keepdim=True))
                    del state_dict[key]
                    logger.info(
                        f'FusedMIT: expanded patch_embed1.proj from '
                        f'{list(old_w.shape)} to {list(new_w.shape)} '
                        f'(extra channels init from RGB mean)'
                    )

            # Load matching weights
            own_state = self.state_dict()
            loaded = 0
            skipped = 0
            for name, param in state_dict.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded += 1
                else:
                    skipped += 1
            logger.info(
                f'FusedMIT: loaded {loaded} params, skipped {skipped} '
                f'(shape mismatch or not in model)'
            )
