import math
import torch
import torch.nn as nn


def sinusoidal_positional_encoding(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(seq_len, device=device).unsqueeze(1)          # [T, 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [T, d_model]


class ResidualEncoder(nn.Module):
    """
    Transformer encoder producing time-varying residual z(t).
    Input:  MFCCs [Batch, T, 30]
    Output: z(t)  [Batch, T, D_z]
    """

    def __init__(
        self,
        n_mfcc: int = 30,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_z: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm_in = nn.LayerNorm(n_mfcc)
        self.input_proj = nn.Linear(n_mfcc, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, d_z)

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        """mfcc: [B, T, 30]  →  z: [B, T, D_z]"""
        x = self.input_proj(self.norm_in(mfcc))                      # [B, T, d_model]
        pe = sinusoidal_positional_encoding(x.shape[1], self.d_model, x.device)
        x = x + pe.unsqueeze(0)
        x = self.transformer(x)                                        # [B, T, d_model]
        return self.output_proj(x)                                     # [B, T, D_z]


class GlobalTimbreEncoder(nn.Module):
    """
    Transformer encoder with CLS token producing a global timbre vector.
    Input:  MFCCs [Batch, T, 30]
    Output: h_t   [Batch, D_t]
    """

    def __init__(
        self,
        n_mfcc: int = 30,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_t: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm_in = nn.LayerNorm(n_mfcc)
        self.input_proj = nn.Linear(n_mfcc, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, d_t)

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        """mfcc: [B, T, 30]  →  h_t: [B, D_t]"""
        B, T, _ = mfcc.shape
        x = self.input_proj(self.norm_in(mfcc))                      # [B, T, d_model]

        pe = sinusoidal_positional_encoding(T, self.d_model, x.device)
        x = x + pe.unsqueeze(0)

        # prepend CLS; CLS gets no positional encoding (position is implicit)
        cls = self.cls_token.expand(B, -1, -1)                       # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)                               # [B, T+1, d_model]

        x = self.transformer(x)                                       # [B, T+1, d_model]
        cls_out = x[:, 0, :]                                          # [B, d_model]
        return self.output_proj(cls_out)                              # [B, D_t]