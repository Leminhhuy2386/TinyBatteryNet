import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class _DSConvBlock(nn.Module):
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
    def __init__(
        self,
        in_ch: int,
        scale_ch: int,
        kernels: tuple = (15, 31, 61),
        pool_out: int = 5,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [_DSConvBlock(in_ch, scale_ch, k) for k in kernels]
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.pool(branch(x)) for branch in self.branches]
        return torch.cat(feats, dim=1)


class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.GELU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class Model(nn.Module):
    _N_CHANNELS: int = 3

    def __init__(self, configs) -> None:
        super().__init__()

        d: int = configs.d_model
        _SCALES = (15, 31, 61)
        _SCALE_CH = max(d // 4, 8)
        _POOL_OUT = 5
        _FLAT_DIM = len(_SCALES) * _SCALE_CH * _POOL_OUT
        _SE_REDUCE = 4
        _GRU_LAYERS = max(getattr(configs, 'lstm_layers', 1), 1)
        drop = getattr(configs, 'dropout', 0.1)

        self.intra_pyramid = _MultiScalePyramid(
            self._N_CHANNELS, _SCALE_CH, _SCALES, _POOL_OUT,
        )
        self.intra_proj = nn.Sequential(
            nn.Linear(_FLAT_DIM, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.se = _SEBlock(d, _SE_REDUCE)
        self.intra_drop = nn.Dropout(drop)

        self.cycle_gate = nn.Linear(d, 1)

        self.gru = nn.GRU(
            d, d,
            num_layers=_GRU_LAYERS,
            batch_first=True,
            dropout=drop if _GRU_LAYERS > 1 else 0.0,
        )
        self.inter_norm = nn.LayerNorm(d)
        self.inter_drop = nn.Dropout(drop)

        self.head = nn.Linear(d, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for name, p in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def enable_mc_dropout(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _forward_core(self, cycle_curve_data: torch.Tensor, curve_attn_mask: torch.Tensor):
        B, L, C, T = cycle_curve_data.shape
        d = self.gru.hidden_size

        x = cycle_curve_data.view(B * L, C, T)
        x = self.intra_pyramid(x)
        x = x.flatten(1)
        x = self.intra_proj(x)
        x = self.se(x)
        x = self.intra_drop(x)
        x = x.view(B, L, d)

        gate = torch.sigmoid(self.cycle_gate(x))
        mask = curve_attn_mask.unsqueeze(-1)
        gate = gate * mask
        x = x * gate

        lengths = curve_attn_mask.sum(dim=1).long().cpu().clamp(min=1)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        gru_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)

        idx = (lengths - 1).clamp(min=0).to(gru_out.device)
        idx = idx.view(B, 1, 1).expand(B, 1, d)
        embedding = gru_out.gather(1, idx).squeeze(1)

        embedding = self.inter_norm(embedding)
        embedding = self.inter_drop(embedding)
        out = self.head(embedding)
        return out, embedding

    def forward(self, cycle_curve_data: torch.Tensor, curve_attn_mask: torch.Tensor, return_embedding: bool = False):
        out, embedding = self._forward_core(cycle_curve_data, curve_attn_mask)
        if return_embedding:
            return out, embedding
        return out
