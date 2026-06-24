"""
TinyBatteryNet v2 — Chemistry-Adaptive Cross-Cycle Transformer
==============================================================

Designed for:
  • STM32 microcontroller deployment (< 60 K parameters)
  • High accuracy competitive with CP-model benchmarks across all chemistries
  • Publication-quality innovations targeting cross-chemistry generalisation

Architecture overview
---------------------
  Input  : [B, L, C, T]   L=100 cycles, C=3 channels (V/I/Q), T=300 time-points
  Mask   : [B, L]          1=visible cycle, 0=padded
  Output : [B, 1]          normalised EOL (inverse-transform to get cycle count)

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Stage 1 — Multi-Scale Depthwise-Separable Feature Pyramid               │
  │  [B, L, C, T] → [B, L, d_model]                                         │
  │  • 3 parallel DS-Conv1D branches (k=15, 31, 61, padding='same')          │
  │  • AdaptiveAvgPool1d(pool_out=5) per branch                              │
  │  • Concat → flatten → Linear → GELU → LayerNorm                          │
  │  • Squeeze-and-Excitation (SE) channel recalibration                     │
  ├──────────────────────────────────────────────────────────────────────────┤
  │  Stage 2 — Relative Degradation Residual Injection  [NEW v2]             │
  │  [B, L, d] → [B, L, d]                                                  │
  │  • delta[t] = f[t] − f[0] in feature space                              │
  │  • x ← LayerNorm(x + W_rel · delta[t])                                  │
  │  • Chemistry-scale-invariant: relative degradation is universal          │
  ├──────────────────────────────────────────────────────────────────────────┤
  │  Stage 3 — Sinusoidal PE + CLS Token  [NEW v2]                           │
  │  [B, L, d] → [B, L+1, d]                                                │
  │  • Sinusoidal PE with learnable amplitude (init=0.1)                     │
  │  • Learnable CLS token prepended for global trajectory aggregation       │
  ├──────────────────────────────────────────────────────────────────────────┤
  │  Stage 4 — CLS-Token Cross-Cycle Transformer  [NEW v2, replaces GRU]    │
  │  [B, L+1, d] → [B, d]                                                   │
  │  • Pre-LN TransformerEncoderLayer (n_heads, ffn=2d)                     │
  │  • Masked attention — cycle 100 can attend directly to cycle 1           │
  │  • CLS token output = global trajectory summary                          │
  ├──────────────────────────────────────────────────────────────────────────┤
  │  Stage 5 — Regression Head                                               │
  │  [B, d] → [B, 1]                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

Key innovations for journal publication
-----------------------------------------
  1. Relative Degradation Residual Injection (RDRI)  [Novel]
       After extracting per-cycle embeddings, computes the feature-space
       delta between each cycle and the first observed cycle, then injects
       it as a residual:  x <- LN(x + W_rel*(f[t]-f[0])).
       Creates a chemistry-scale-invariant relative degradation signal
       without any explicit chemistry label or normalisation statistics.

  2. CLS-Token Cross-Cycle Transformer  [Novel for battery lifetime prediction]
       Replaces the sequential GRU with a single masked Transformer encoder
       layer + learnable CLS aggregation token.  Unlike GRU, self-attention
       allows cycle 100 to attend directly to cycle 1 (no 99-step bottleneck).
       CLS token, having no position, learns a global trajectory summary.

  3. Multi-Scale DS-Conv Feature Pyramid + SE  [Preserved from v1]
       MobileNet-style branches capture fine (k=15), mid (k=31), and coarse
       (k=61) temporal features at 8-9x lower parameter cost than Conv2D.

  4. Sinusoidal PE with Learnable Amplitude
       Fixed sinusoidal frequencies scaled by a single trainable scalar
       (init=0.1).  If positional information is unhelpful, the gradient
       drives pe_scale toward 0 without harming other features.

  5. STM32 Deployment Readiness
       Float32: ~56 K x 4B = 224 KB (fits STM32H7 2 MB Flash)
       INT8:    ~56 K x 1B =  56 KB (fits STM32F4 512 KB Flash)

Parameter count (d_model=64, scale_ch=16, n_heads=4)  ~56 K total
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _make_sinusoidal_pe(L: int, d: int) -> torch.Tensor:
    """Return [L, d] sinusoidal positional encoding (no gradient)."""
    position  = torch.arange(L, dtype=torch.float).unsqueeze(1)
    half_d    = d // 2
    div_term  = torch.exp(
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
    """Depthwise-separable 1-D conv: DW -> PW -> BN -> GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.dw = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            padding='same', groups=in_ch, bias=False,
        )
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.bn(self.pw(self.dw(x))))


class _MultiScalePyramid(nn.Module):
    """Three parallel DS-Conv1D branches at different temporal scales."""

    def __init__(
        self,
        in_ch:    int,
        scale_ch: int,
        kernels:  tuple = (15, 31, 61),
        pool_out: int   = 5,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [_DSConvBlock(in_ch, scale_ch, k) for k in kernels]
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.pool(b(x)) for b in self.branches], dim=1)


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel recalibration on flat [N, C] vectors."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        r = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, r,        bias=False),
            nn.GELU(),
            nn.Linear(r,        channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """TinyBatteryNet v2 — see module docstring for full description."""

    _N_CHANNELS: int = 3   # V, I, Q

    def __init__(self, configs) -> None:
        super().__init__()

        T:  int = configs.charge_discharge_length   # 300
        L:  int = configs.early_cycle_threshold     # 100
        d:  int = configs.d_model                   # 64

        _SCALES   = (15, 31, 61)
        _SCALE_CH = max(d // 4, 8)
        _POOL_OUT = 5
        _FLAT_DIM = len(_SCALES) * _SCALE_CH * _POOL_OUT   # 240 at d=64

        drop    = getattr(configs, 'dropout', 0.1)
        n_heads = getattr(configs, 'n_heads', 4)
        while d % n_heads != 0:
            n_heads = max(1, n_heads - 1)

        self.charge_discharge_length = T
        self.early_cycle_threshold   = L
        self.d                       = d

        # Stage 1 — multi-scale intra-cycle feature extraction
        self.intra_pyramid = _MultiScalePyramid(
            self._N_CHANNELS, _SCALE_CH, _SCALES, _POOL_OUT,
        )
        self.intra_proj = nn.Sequential(
            nn.Linear(_FLAT_DIM, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.se         = _SEBlock(d)
        self.intra_drop = nn.Dropout(drop)

        # Stage 2 — Relative Degradation Residual Injection (RDRI)
        # Encodes feature-space delta relative to the first observed cycle.
        # Using bias=False avoids a constant offset that could mask the signal.
        self.rdri_proj = nn.Linear(d, d, bias=False)
        self.rdri_norm = nn.LayerNorm(d)

        # Stage 3 — Positional Encoding + CLS token
        self.register_buffer('_pe', _make_sinusoidal_pe(L, d))   # [L, d]
        self.pe_scale  = nn.Parameter(torch.tensor(0.3))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))

        # Stage 4 — Cross-Cycle Transformer (Pre-LN, batch_first)
        self.cycle_transformer = nn.TransformerEncoderLayer(
            d_model         = d,
            nhead           = n_heads,
            dim_feedforward = d * 2,
            dropout         = drop,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,
        )

        # Stage 5 — Regression Head
        # Applied directly to the CLS token (global trajectory summary).
        # Keeping this simple matches the original model that achieved ZN-coin 0.427.
        # The fusion head (CLS+mean+last) was added later as an "upgrade" but
        # degraded Zn-ion performance — extra capacity + dropout noise hurt on the
        # small ZN-coin dataset.
        self.head_norm = nn.LayerNorm(d)
        self.head      = nn.Linear(d, 1)

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

    def enable_mc_dropout(self) -> None:
        """Set all Dropout layers to training mode for MC-Dropout inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    # ─────────────────────────────────────────────────────────────────────
    def _forward_core(
        self,
        cycle_curve_data: torch.Tensor,
        curve_attn_mask:  torch.Tensor,
    ):
        B, L, C, T = cycle_curve_data.shape
        d = self.d

        # Stage 1: intra-cycle features
        x = cycle_curve_data.view(B * L, C, T)
        x = self.intra_pyramid(x)
        x = x.flatten(1)
        x = self.intra_proj(x)          # [B*L, d]
        x = self.se(x)
        x = self.intra_drop(x)
        x = x.view(B, L, d)             # [B, L, d]

        # Stage 2: RDRI — chemistry-scale-invariant relative residual
        # Use mean of first min(3, L) cycles as reference — more robust than
        # cycle 0 alone, which can be noisy during formation/activation.
        k_ref = min(3, L)
        ref   = x[:, :k_ref, :].mean(dim=1, keepdim=True).detach()  # [B, 1, d]
        delta = x - ref                                               # [B, L, d]
        x     = self.rdri_norm(x + self.rdri_proj(delta))

        # Zero-out padded cycles (after RDRI to keep valid deltas for real cycles)
        x = x * curve_attn_mask.unsqueeze(-1)     # [B, L, d]

        # Stage 3: sinusoidal PE + CLS token
        x   = x + self.pe_scale * self._pe.unsqueeze(0)  # [B, L, d]
        cls = self.cls_token.expand(B, -1, -1)            # [B, 1, d]
        x   = torch.cat([cls, x], dim=1)                  # [B, L+1, d]

        # Key-padding mask: True = ignore.  CLS is always visible.
        pad_mask  = (1.0 - curve_attn_mask).bool()
        cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)   # [B, L+1]

        # Stage 4: cross-cycle Transformer with causal mask for cycle tokens.
        # Each cycle token at position t (sequence index t+1) may attend only to
        # the CLS token (index 0) and previous cycle tokens (indices 1..t).
        # CLS token stays fully bidirectional — it needs global trajectory context.
        # This matches the physical constraint that degradation at cycle t is only
        # influenced by cycles 0..t-1, not future cycles.
        L_full = x.shape[1]   # L + 1  (CLS prepended)
        causal = torch.triu(
            torch.ones(L_full, L_full, dtype=torch.bool, device=x.device), diagonal=1
        )
        causal[0, :] = False   # CLS row:  can attend to all positions
        causal[:, 0] = False   # CLS col:  all tokens can attend to CLS
        x = self.cycle_transformer(x, src_mask=causal, src_key_padding_mask=full_mask)

        # Stage 5: Regression on CLS token
        embedding = self.head_norm(x[:, 0, :])   # [B, d]
        out       = self.head(embedding)          # [B, 1]
        return out, embedding

    def forward(
        self,
        cycle_curve_data: torch.Tensor,
        curve_attn_mask:  torch.Tensor,
        return_embedding: bool = False,
    ):
        """
        Args:
            cycle_curve_data : [B, L, num_variables, fixed_length_of_curve]
            curve_attn_mask  : [B, L]  float, 1=visible 0=padded
            return_embedding : if True, also return [B, d] CLS embedding
        Returns:
            [B, 1]  or  ([B, 1], [B, d])
        """
        out, embedding = self._forward_core(cycle_curve_data, curve_attn_mask)
        if return_embedding:
            return out, embedding
        return out
