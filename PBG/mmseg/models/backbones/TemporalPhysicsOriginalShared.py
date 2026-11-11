import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGuidedTDCR_Module(nn.Module):
    """Physics-guided temporal difference reconstruction.

    The module keeps the original TDCR interface:
        input:  [B, T, C, H, W]
        output: [B, C, H, W]

    It adds a lightweight latent advection-diffusion step on top of frame
    differences. The physical fields are learned from feature pairs, while the
    update itself uses fixed spatial gradient and Laplacian operators.
    """

    def __init__(self,
                 channels,
                 dt=1.0,
                 velocity_scale=1.0,
                 diffusion_scale=0.25,
                 source_scale=0.10,
                 conf_scale=2.0,
                 gate_temporal_features=True):
        super().__init__()
        self.channels = channels
        self.dt = dt
        self.velocity_scale = velocity_scale
        self.diffusion_scale = diffusion_scale
        self.source_scale = source_scale
        self.conf_scale = conf_scale
        self.gate_temporal_features = gate_temporal_features

        self.diff_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        hidden_channels = max(channels // 2, 16)
        self.physics_param = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels + 3, 1)
        )

        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 1), 1, 1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Conv2d(channels, channels, 1)

        grad_x = torch.tensor(
            [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
            dtype=torch.float32).view(1, 1, 3, 3)
        grad_y = torch.tensor(
            [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
            dtype=torch.float32).view(1, 1, 3, 3)
        laplace = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('grad_x_kernel', grad_x)
        self.register_buffer('grad_y_kernel', grad_y)
        self.register_buffer('laplace_kernel', laplace)
        self.visualize_debug = False
        self.visual_debug = {}
        self._last_physics_error = None
        self.reset_physics_parameters()

    def reset_physics_parameters(self):
        nn.init.constant_(self.physics_param[-1].weight, 0.0)
        nn.init.constant_(self.physics_param[-1].bias, 0.0)
        with torch.no_grad():
            self.physics_param[-1].bias[2].fill_(-2.0)

    def _depthwise_filter(self, x, kernel):
        channels = x.shape[1]
        weight = kernel.expand(channels, 1, 3, 3)
        return F.conv2d(x, weight, padding=1, groups=channels)

    def _physics_update(self, hist, curr):
        observed_delta = curr - hist
        pair = torch.cat([hist, curr, observed_delta], dim=1)
        params = self.physics_param(pair)

        velocity = torch.tanh(params[:, :2]) * self.velocity_scale
        diffusivity = torch.sigmoid(params[:, 2:3]) * self.diffusion_scale
        source = torch.tanh(params[:, 3:]) * self.source_scale

        grad_x = self._depthwise_filter(hist, self.grad_x_kernel)
        grad_y = self._depthwise_filter(hist, self.grad_y_kernel)
        laplace = self._depthwise_filter(hist, self.laplace_kernel)

        advection = velocity[:, 0:1] * grad_x + velocity[:, 1:2] * grad_y
        pde_delta = self.dt * (-advection + diffusivity * laplace + source)
        pred_curr = hist + pde_delta

        residual = pred_curr - curr
        residual_mag = residual.abs().mean(dim=1, keepdim=True)
        if self.visualize_debug:
            self._last_physics_error = residual_mag.detach()
        residual_norm = residual_mag / (
            residual_mag.mean(dim=(2, 3), keepdim=True) + 1e-6)
        physics_conf = torch.sigmoid(self.conf_scale * (1.0 - residual_norm))

        guided_delta = observed_delta + physics_conf * pde_delta
        return guided_delta, physics_conf

    def forward(self, x_seq):
        self.visual_debug = {}
        self._last_physics_error = None
        B, T, C, H, W = x_seq.shape
        if T <= 1:
            return x_seq.new_zeros(B, C, H, W)

        curr = x_seq[:, -1, ...]
        history = x_seq[:, :-1, ...]
        t_hist = T - 1

        curr_rep = curr.unsqueeze(1).expand_as(history)
        hist_flat = history.reshape(B * t_hist, C, H, W)
        curr_flat = curr_rep.reshape(B * t_hist, C, H, W)

        guided_delta, physics_conf = self._physics_update(hist_flat, curr_flat)
        diff_feats = self.diff_conv(guided_delta)

        learned_weight = self.temporal_attention(diff_feats)
        if self.visualize_debug:
            self.visual_debug = dict(
                confidence=physics_conf.detach().view(B, t_hist, 1, H, W),
                temporal_weight=learned_weight.detach().view(B, t_hist),
                physical_error=self._last_physics_error.view(
                    B, t_hist, 1, H, W))
        weighted = diff_feats * learned_weight
        if self.gate_temporal_features:
            weighted = weighted * physics_conf
        weighted = weighted.view(B, t_hist, C, H, W)
        motion_feat = torch.sum(weighted, dim=1)

        return self.out_proj(motion_feat)
