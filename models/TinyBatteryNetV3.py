"""
TinyBatteryNet v3 — ICA-Physics-Informed Battery Life Predictor
================================================================

Architecture overview
---------------------
  Input  : [B, L, C, T]   L=100 cycles, C=3 channels (V=0/I=1/Q=2), T=300 time-points
  Mask   : [B, L]          1=visible cycle, 0=padded
  Output : [B, 1]          normalised EOL

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Stage 1 — Multi-Scale DSConv Pyramid + SE  [from v1/v2]                │
  │  [B, L, C, T] → [B, L, d]                                               │
  ├──────────────────────────────────────────────────────────────────────────┤
  │  Stage 2 — ICA Feature Extractor  [NEW v3]                               │
  │  raw [B*L, C, T] → per-cycle IC/DVA stats → [B, L, d_ic]               │
  │  • dQ/dV  (IC)  : mean, soft-peak, std, center-of-gravity               │
  │  • dV/dQ  (DVA) : mean, std                                              │
  │  Fused with Stage-1 embedding via concat→linear                          │
  ├──────────────────────────────────────────────────────────────────────────┤
  │  Stage 3 — Dual Degradation Residual Injection  [NEW v3]                 │
  │  [B, L, d] → [B, L, d]                                                  │
  │  • absolute  delta:  f[t] − f[0]                (from v2 RDRI)           │
  │  • normalised delta: (f[t]−f[0]) / (|f[0]|+ε)  (new — scale-invariant) │
  │  Two separate projections injected as residual + LayerNorm               │
  ├──────────────────────────────────────────────────────────────────────────┤
  │  Stage 4 — Sinusoidal PE + CLS Token  [from v2]                          │
  │  [B, L, d] → [B, L+1, d]                                                │
  ├──────────────────────────────────────────────────────────────────────────┤
  │  Stage 5 — Pre-LN Cross-Cycle Transformer  [from v2]                     │
  │  [B, L+1, d] → [B, d]   (CLS token output)                              │
  ├──────────────────────────────────────────────────────────────────────────┤
  │  Stage 6 — Regression Head                                               │
  │  [B, d] → [B, 1]                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

Key innovations vs v2
---------------------
  1. ICA Feature Extractor (ICA-FE)  [Novel]
       Computes incremental capacity IC = dQ/dV and differential voltage
       DVA = dV/dQ per cycle from raw V and Q channels (no extra params for
       signal extraction).  Extracts 6 physics-meaningful statistics:
         • IC mean        — overall capacity delivery rate
         • IC soft-peak   — differentiable peak height (log-sum-exp)
         • IC std         — phase-transition sharpness (peaks narrow with aging)
         • IC CoG         — center-of-gravity position (shifts left as Li inventory
                            decreases — a direct signature of capacity loss)
         • DVA mean       — voltage operating window indicator
         • DVA std        — spread of voltage response
       IC peak shift and collapse are the electrochemical gold standard for
       battery state-of-health (Dubarry et al. 2012; Riviere et al. 2019).
       Cost: ~1.3 K extra parameters.

  2. Dual Degradation Residual Injection (DDRI)  [Novel, extends v2 RDRI]
       v2 injected only absolute feature delta: δ_abs = f[t] − f[0].
       v3 adds a chemistry-normalised delta:
           δ_norm = δ_abs / (|f[0]| + ε)
       which converts the signal into a unitless ratio, making it comparable
       across chemistries (a 10 % capacity drop looks identical whether the
       absolute capacity is 1 Ah or 5 Ah).  Two separate linear projections
       W_abs, W_norm are summed as a residual before LayerNorm.
       Cost: ~8.3 K extra parameters (one additional projection vs v2).

Retained from v2
-----------------
  Multi-Scale DSConv Pyramid, SE recalibration, sinusoidal PE with learnable
  amplitude, learnable CLS token, Pre-LN TransformerEncoderLayer.

Parameter count: ~65 K at d_model=64  (INT8 ≈ 65 KB — fits STM32H7 Flash)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _make_sinusoidal_pe(L: int, d: int) -> torch.Tensor:
    position = torch.arange(L, dtype=torch.float).unsqueeze(1)
    half_d   = d // 2
    div_term = torch.exp(
        torch.arange(0, half_d, dtype=torch.float) * (-math.log(10000.0) / half_d)
    )
    pe          = torch.zeros(L, d)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _DSConvBlock(nn.Module):
    """Depthwise-separable 1-D conv: DW → PW → BN → GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size,
                            padding='same', groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.bn(self.pw(self.dw(x))))


class _MultiScalePyramid(nn.Module):
    """Three parallel DS-Conv1D branches (k = 15 / 31 / 61)."""

    def __init__(self, in_ch: int, scale_ch: int,
                 kernels: tuple = (15, 31, 61), pool_out: int = 5) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [_DSConvBlock(in_ch, scale_ch, k) for k in kernels]
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.pool(b(x)) for b in self.branches], dim=1)


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation recalibration on flat [N, C] feature vectors."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        r = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, r, bias=False), nn.GELU(),
            nn.Linear(r, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class _ICABlock(nn.Module):
    """
    Incremental Capacity Analysis (ICA) Feature Extractor.

    Computes IC = dQ/dV and DVA = dV/dQ from raw V (ch 0) and Q (ch 2),
    then extracts 6 physics-meaningful statistics per cycle:

      [IC_mean, IC_peak, IC_std, IC_CoG, DVA_mean, DVA_std]

    where:
      IC_mean  — overall capacity delivery rate per unit voltage
      IC_peak  — differentiable peak height via log-sum-exp
      IC_std   — sharpness of phase transitions (narrows with aging)
      IC_CoG   — soft center-of-gravity of IC curve
                 (shifts toward lower voltages as Li inventory is lost)
      DVA_mean — mean differential voltage (operating window indicator)
      DVA_std  — spread of voltage response (resistance signature)

    Both IC and DVA values are bounded through tanh to avoid numerical
    instability near dV ≈ 0 or dQ ≈ 0, while preserving the sign and
    relative magnitude of the electrochemical signal.
    """

    _N_STATS = 6

    def __init__(self, out_dim: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps     = eps
        self.proj    = nn.Linear(self._N_STATS, out_dim, bias=False)
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw : [N, C, T]  — raw cycle curves; V=ch0, Q=ch2
        Returns: [N, out_dim]
        """
        v = x_raw[:, 0, :]   # voltage  [N, T]
        q = x_raw[:, 2, :]   # capacity [N, T]

        dv = v[:, 1:] - v[:, :-1]   # [N, T-1]
        dq = q[:, 1:] - q[:, :-1]   # [N, T-1]

        # Bounded IC and DVA via tanh
        ic  = torch.tanh(dq / dv.abs().clamp(min=self.eps))   # [N, T-1]
        dva = torch.tanh(dv / dq.abs().clamp(min=self.eps))   # [N, T-1]

        ic_mean = ic.mean(dim=-1, keepdim=True)                                  # [N,1]
        ic_peak = (torch.logsumexp(ic, dim=-1, keepdim=True)
                   - math.log(ic.shape[-1]))                                     # [N,1]
        ic_std  = ic.std(dim=-1, keepdim=True).clamp(min=1e-6)                  # [N,1]

        # Center-of-gravity: soft-argmax of IC curve (differentiable peak position)
        T_m1  = ic.shape[-1]
        t_idx = torch.linspace(0.0, 1.0, T_m1, device=ic.device).unsqueeze(0)  # [1,T-1]
        ic_cog = (t_idx * F.softmax(ic * 5.0, dim=-1)).sum(dim=-1, keepdim=True) # [N,1]

        dva_mean = dva.mean(dim=-1, keepdim=True)                                # [N,1]
        dva_std  = dva.std(dim=-1, keepdim=True).clamp(min=1e-6)                # [N,1]

        stats = torch.cat(
            [ic_mean, ic_peak, ic_std, ic_cog, dva_mean, dva_std], dim=-1
        )   # [N, 6]
        return self.norm(self.proj(stats))   # [N, out_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """TinyBatteryNet v3 — see module docstring for full architecture."""

    _N_CHANNELS: int = 3   # V, I, Q

    def __init__(self, configs) -> None:
        super().__init__()

        T: int = configs.charge_discharge_length   # 300
        L: int = configs.early_cycle_threshold     # 100
        d: int = configs.d_model                   # 64

        _SCALES   = (15, 31, 61)
        _SCALE_CH = max(d // 4, 8)
        _POOL_OUT = 5
        _FLAT_DIM = len(_SCALES) * _SCALE_CH * _POOL_OUT   # 240 @ d=64
        _ICA_DIM  = max(d // 4, 8)                         #  16 @ d=64

        drop    = getattr(configs, 'dropout', 0.1)
        n_heads = getattr(configs, 'n_heads', 4)
        while d % n_heads != 0:
            n_heads = max(1, n_heads - 1)
        d_ffn = getattr(configs, 'd_ff', d * 2)

        self.charge_discharge_length = T
        self.early_cycle_threshold   = L
        self.d                       = d

        # ── Stage 1: Multi-Scale DSConv Pyramid + SE ─────────────────────
        self.intra_pyramid = _MultiScalePyramid(
            self._N_CHANNELS, _SCALE_CH, _SCALES, _POOL_OUT,
        )
        self.intra_proj = nn.Sequential(
            nn.Linear(_FLAT_DIM, d), nn.GELU(), nn.LayerNorm(d),
        )
        self.se         = _SEBlock(d)
        self.intra_drop = nn.Dropout(drop)

        # ── Stage 2: ICA Feature Extractor + fusion ───────────────────────
        self.ica_block = _ICABlock(out_dim=_ICA_DIM)
        self.ica_fuse  = nn.Sequential(
            nn.Linear(d + _ICA_DIM, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        # ── Stage 3: Dual Degradation Residual Injection (DDRI) ──────────
        # W_abs  projects the absolute feature delta  (f[t] − f[0])
        # W_norm projects the normalised feature delta (δ_abs / |f[0]|)
        # Both injected as a combined residual → LayerNorm.
        self.ddri_abs_proj  = nn.Linear(d, d, bias=False)
        self.ddri_norm_proj = nn.Linear(d, d, bias=False)
        self.ddri_ln        = nn.LayerNorm(d)

        # ── Stage 4: Sinusoidal PE + CLS token ───────────────────────────
        self.register_buffer('_pe', _make_sinusoidal_pe(L, d))
        self.pe_scale  = nn.Parameter(torch.tensor(0.1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))

        # ── Stage 5: Pre-LN Cross-Cycle Transformer ───────────────────────
        self.cycle_transformer = nn.TransformerEncoderLayer(
            d_model         = d,
            nhead           = n_heads,
            dim_feedforward = d_ffn,
            dropout         = drop,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,
        )

        # ── Stage 6: Regression Head ──────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 1),
        )

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        cycle_curve_data: torch.Tensor,   # [B, L, C, T]
        curve_attn_mask:  torch.Tensor,   # [B, L]
        return_embedding: bool = False,
    ):
        B, L, C, T = cycle_curve_data.shape
        d = self.d

        x_raw = cycle_curve_data.view(B * L, C, T)   # [B*L, C, T]

        # Stage 1: multi-scale conv features
        x = self.intra_pyramid(x_raw).flatten(1)      # [B*L, flat_dim]
        x = self.intra_proj(x)                        # [B*L, d]
        x = self.se(x)
        x = self.intra_drop(x)
        x = x.view(B, L, d)                           # [B, L, d]

        # Stage 2: ICA features — fuse electrochemical stats with conv features
        ic_feat = self.ica_block(x_raw).view(B, L, -1)            # [B, L, d_ic]
        x = self.ica_fuse(torch.cat([x, ic_feat], dim=-1))        # [B, L, d]

        # Stage 3: DDRI — absolute + chemistry-normalised residual injection
        anchor     = x[:, 0:1, :].detach()                            # [B, 1, d]
        delta_abs  = x - anchor                                        # [B, L, d]
        delta_norm = delta_abs / anchor.abs().clamp(min=1e-6)         # [B, L, d]
        x = self.ddri_ln(
            x
            + self.ddri_abs_proj(delta_abs)
            + self.ddri_norm_proj(delta_norm)
        )

        # Zero-out padded cycles (after DDRI — valid deltas are already encoded)
        x = x * curve_attn_mask.unsqueeze(-1)                         # [B, L, d]

        # Stage 4: sinusoidal PE + CLS token
        x   = x + self.pe_scale * self._pe.unsqueeze(0)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)                              # [B, L+1, d]

        pad_mask  = (1.0 - curve_attn_mask).bool()
        cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)            # [B, L+1]

        # Stage 5: cross-cycle Transformer
        x = self.cycle_transformer(x, src_key_padding_mask=full_mask)

        # Stage 6: predict from CLS token
        cls_out = x[:, 0, :]                                          # [B, d]
        out     = self.head(cls_out)                                  # [B, 1]

        if return_embedding:
            return out, cls_out
        return out
