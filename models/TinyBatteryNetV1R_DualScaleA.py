"""
TinyBatteryNetV1R_DualScaleA — k={15, 31} only (no k=61 branch).
EXP-C ablation: two-scale variant A.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class _DSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size, padding='same', groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        return F.gelu(self.bn(self.pw(self.dw(x))))


class _MultiScalePyramid(nn.Module):
    def __init__(self, in_ch, scale_ch, kernels=(15, 31), pool_out=5):
        super().__init__()
        self.branches = nn.ModuleList([_DSConvBlock(in_ch, scale_ch, k) for k in kernels])
        self.pool = nn.AdaptiveAvgPool1d(pool_out)
    def forward(self, x):
        return torch.cat([self.pool(b(x)) for b in self.branches], dim=1)


class _SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        r = max(channels // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(channels, r, bias=False), nn.GELU(),
                                 nn.Linear(r, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x)


class Model(nn.Module):
    _N_CHANNELS = 3
    def __init__(self, configs):
        super().__init__()
        d = configs.d_model
        _SCALES = (15, 31)        # ← two scales only
        _SCALE_CH = max(d // 4, 8)
        _POOL_OUT = 5
        _FLAT_DIM = len(_SCALES) * _SCALE_CH * _POOL_OUT
        _GRU_LAYERS = max(getattr(configs, 'lstm_layers', 1), 1)
        drop = getattr(configs, 'dropout', 0.1)
        self.intra_pyramid = _MultiScalePyramid(self._N_CHANNELS, _SCALE_CH, _SCALES, _POOL_OUT)
        self.intra_proj = nn.Sequential(nn.Linear(_FLAT_DIM, d), nn.GELU(), nn.LayerNorm(d))
        self.se = _SEBlock(d, 4)
        self.intra_drop = nn.Dropout(drop)
        self.cycle_gate = nn.Linear(d, 1)
        self.gru = nn.GRU(d, d, num_layers=_GRU_LAYERS, batch_first=True,
                          dropout=drop if _GRU_LAYERS > 1 else 0.0)
        self.inter_norm = nn.LayerNorm(d)
        self.inter_drop = nn.Dropout(drop)
        self.head = nn.Linear(d, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); \
                (nn.init.zeros_(m.bias) if m.bias is not None else None)
            elif isinstance(m, nn.Conv1d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        for n, p in self.gru.named_parameters():
            if 'weight_ih' in n: nn.init.xavier_uniform_(p)
            elif 'weight_hh' in n: nn.init.orthogonal_(p)
            elif 'bias' in n: nn.init.zeros_(p)

    def forward(self, cycle_curve_data, curve_attn_mask, return_embedding=False):
        B, L, C, T = cycle_curve_data.shape; d = self.gru.hidden_size
        x = cycle_curve_data.view(B*L, C, T)
        x = self.intra_pyramid(x); x = x.flatten(1); x = self.intra_proj(x)
        x = self.se(x); x = self.intra_drop(x); x = x.view(B, L, d)
        gate = torch.sigmoid(self.cycle_gate(x)) * curve_attn_mask.unsqueeze(-1)
        x = x * gate
        lengths = curve_attn_mask.sum(1).long().cpu().clamp(min=1)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        gru_out, _ = self.gru(packed); gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)
        idx = (lengths - 1).clamp(0).to(gru_out.device).view(B, 1, 1).expand(B, 1, d)
        emb = self.inter_norm(self.inter_drop(gru_out.gather(1, idx).squeeze(1)))
        out = self.head(emb)
        return (out, emb) if return_embedding else out
