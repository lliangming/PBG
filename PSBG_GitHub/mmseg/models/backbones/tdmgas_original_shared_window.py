from functools import partial

import torch
import torch.nn as nn

from mmseg.models.builder import BACKBONES
from .fused_mit import FusedMIT
from .TemporalPhysicsOriginalShared import PhysicsGuidedTDCR_Module


class OriginalSharedWindowFusedMIT(FusedMIT):
    """Original TDMGas layers with shared-history routing and single-frame bypass.

    All frames except the last supply both SM's mean reference and PITE's
    historical inputs. The dataset supplies the temporal spacing; the model
    does not sample the clip again. Parameter names/shapes match TDMGas.
    """

    def __init__(self, num_frames=3, memory_gate_scale=1.0,
                 physics_cfg=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(num_frames, int) or num_frames < 1:
            raise ValueError('num_frames must be a positive integer')
        self.num_frames = num_frames
        self.memory_gate_scale = float(memory_gate_scale)
        in_chans = kwargs.get('in_chans', 3)
        hidden_dim = max(self.temporal_dim // 4, 16)
        scene_in_chans = in_chans * 4

        # Keep the original SM projections, gate and initialization order.
        self.scene_residual_proj = nn.Sequential(
            nn.Conv2d(scene_in_chans, self.temporal_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.temporal_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.temporal_dim, self.temporal_dim, 1))
        self.scene_gate = nn.Sequential(
            nn.Conv2d(scene_in_chans, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid())
        self.scene_residual_proj.apply(self._init_weights)
        self.scene_gate.apply(self._init_weights)
        self.reset_scene_parameters()

        self.tdc_encoder = PhysicsGuidedTDCR_Module(
            self.temporal_dim, **(physics_cfg or {}))
        self.tdc_encoder.apply(self._init_weights)
        self.tdc_encoder.reset_physics_parameters()

    def reset_scene_parameters(self):
        nn.init.constant_(self.scene_residual_proj[-1].weight, 0.0)
        if self.scene_residual_proj[-1].bias is not None:
            nn.init.constant_(self.scene_residual_proj[-1].bias, 0.0)

    def forward_features(self, x, spte_enabled=None):
        self.tdc_encoder.visual_debug = {}
        self.tdc_encoder._last_physics_error = None
        if x.dim() == 4:
            x = x.unsqueeze(1)
        batch, time, channels, height, width = x.shape
        current = x[:, -1]
        aligned_feat = x.new_zeros(batch, self.temporal_dim, height, width)
        if spte_enabled is None:
            spte_enabled = torch.ones(batch, device=x.device, dtype=torch.bool)
        else:
            spte_enabled = spte_enabled.to(device=x.device, dtype=torch.bool)

        if self.num_frames > 1 and time >= self.num_frames and spte_enabled.any():
            query = x[spte_enabled, -self.num_frames:]
            active_batch = query.shape[0]
            features = self.temporal_proj(query.reshape(
                -1, channels, height, width))
            features = features.reshape(
                active_batch, self.num_frames, self.temporal_dim, height, width)
            motion_feat = self.tdc_encoder(features)
            temporal_feat = self.deform_align(motion_feat)

            scene_current = query[:, -1]
            prototype = query[:, :-1].mean(dim=1)
            residual = scene_current - prototype
            scene_input = torch.cat([
                scene_current, prototype, residual.abs(), residual
            ], dim=1)
            scene_feat = self.scene_residual_proj(scene_input)
            scene_gate = self.scene_gate(scene_input)
            aligned_feat[spte_enabled] = temporal_feat + (
                self.memory_gate_scale * scene_gate * scene_feat)

        # Original MIT stages; no extra fusion projection or auxiliary loss.
        x = torch.cat([current, aligned_feat], dim=1)
        outputs = []
        for stage in range(1, 5):
            x, height, width = getattr(self, f'patch_embed{stage}')(x)
            for block in getattr(self, f'block{stage}'):
                x = block(x, height, width)
            x = getattr(self, f'norm{stage}')(x)
            x = x.reshape(batch, height, width, -1).permute(
                0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs

    def forward(self, x, spte_enabled=None):
        return self.forward_features(x, spte_enabled=spte_enabled)


@BACKBONES.register_module()
class tdmgas_original_shared_window_b1(OriginalSharedWindowFusedMIT):
    def __init__(self, **kwargs):
        drop_path_rate = kwargs.pop('drop_path_rate', 0.1)
        num_frames = kwargs.pop('num_frames', 3)
        memory_gate_scale = kwargs.pop('memory_gate_scale', 1.0)
        physics_cfg = kwargs.pop('physics_cfg', None)
        temporal_dim = kwargs.pop('temporal_dim', None)
        base_kwargs = dict(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            num_frames=num_frames,
            memory_gate_scale=memory_gate_scale,
            physics_cfg=physics_cfg)
        if temporal_dim is not None:
            base_kwargs['temporal_dim'] = temporal_dim
        super().__init__(**base_kwargs)
