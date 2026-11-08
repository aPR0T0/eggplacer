"""Denoiser + diffusion utilities.

Backbone: ChipDiffusion's AttGNN ("small": width 64, 2 hybrid GATv2 + global-attention blocks,
4 heads), vendored unmodified in third_party/chipdiffusion. Conditioning follows ChipDiffusion:
node features (sizes + flags), edge features = pin offsets of both ends, fixed-node mask as input,
sinusoidal timestep encoding fed through FiLM.

Diffusion (EGGPlacer App. C): cosine schedule, T = 200, epsilon prediction, masked MSE,
deterministic DDIM (eta = 0).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_TP = Path(__file__).resolve().parents[1] / "third_party" / "chipdiffusion"
if str(_TP) not in sys.path:
    sys.path.insert(0, str(_TP))

try:
    from networks.gnn import AttGNN  # noqa: E402  (ChipDiffusion backbone, fetched into third_party/)
    import pos_encoding  # noqa: E402
except ModuleNotFoundError as e:  # pragma: no cover
    raise ImportError("ChipDiffusion backbone not found: run `bash scripts/fetch_chipdiffusion.sh` "
                      "(it is fetched from upstream because that repository has no licence)") from e

NODE_FEATS = 5   # w, h (normalised to the canvas), bottom flag, edge-part flag, log1p(pins)/4
EDGE_FEATS = 4   # (dx, dy) of source pin, (dx, dy) of sink pin, normalised

SMALL = dict(hidden_size=64, hidden_node_features=[192, 192], attention_node_features=[64, 64],
             layers_per_block=2, input_encoding_dim=32, dropout=0.0, num_heads=4, mlp_num_layers=2,
             mlp_size_factor=4, ff_num_layers=2, ff_size_factor=1, att_implementation="default",
             dir_att_input=True, conv_params=dict(layer_type="gat", heads=4, concat=True))


class GraphCond:
    """Minimal stand-in for a torch_geometric Data object (what AttGNN reads)."""

    def __init__(self, x, edge_index, edge_attr, is_fixed):
        self.x, self.edge_index, self.edge_attr, self.is_fixed = x, edge_index, edge_attr, is_fixed

    def __getitem__(self, k):
        return getattr(self, k)

    def __contains__(self, k):
        return hasattr(self, k)

    def to(self, dev):
        return GraphCond(self.x.to(dev), self.edge_index.to(dev), self.edge_attr.to(dev), self.is_fixed.to(dev))


def cosine_alpha_bar(T=200, s=0.008):
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    beta = np.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    return np.cumprod(1 - beta)  # alpha_bar_t for t = 1..T


class Denoiser(nn.Module):
    def __init__(self, T=200, t_dim=32, size=None, device="cpu"):
        super().__init__()
        p = dict(SMALL); p.update(size or {})
        self.T = T
        self.net = AttGNN(in_node_features=2, out_node_features=2, cond_node_features=NODE_FEATS,
                          edge_features=EDGE_FEATS, t_encoding_dim=t_dim, mask_key="is_fixed",
                          device="cuda" if str(device).startswith("cuda") else "cpu", **p)
        self.register_buffer("t_enc", pos_encoding.get_positional_encodings(T + 1, t_dim).float())
        ab = torch.tensor(cosine_alpha_bar(T), dtype=torch.float32)
        self.register_buffer("alpha_bar", ab)

    def forward(self, x, cond: GraphCond, t):
        """x (B,V,2) noisy coords, t (B,) int in 1..T -> eps (B,V,2)."""
        return self.net(x, cond, self.t_enc[t]).view(*x.shape)

    def loss(self, x0, cond: GraphCond, B=16, t=None):
        """masked eps-MSE on one circuit with B noise draws. x0 (V,2)."""
        V = x0.shape[0]
        dev = x0.device
        if t is None:
            t = torch.randint(1, self.T + 1, (B,), device=dev)
        ab = self.alpha_bar[t - 1].view(B, 1, 1)
        eps = torch.randn(B, V, 2, device=dev)
        x = ab.sqrt() * x0[None] + (1 - ab).sqrt() * eps          # Eq. (14): z_t = sqrt(ab_t) z_0 + sqrt(1-ab_t) eps
        fixed = cond.is_fixed.view(1, V, 1)
        x = torch.where(fixed, x0[None].expand(B, V, 2), x)
        pred = self(x, cond, t)                                      # eps_theta(z_t, c, t)
        m = (~fixed).expand(B, V, 2).float()
        return ((pred - eps) ** 2 * m).sum() / m.sum().clamp(min=1)  # Eq. (15): E || eps - eps_theta ||^2 (movable parts)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)
