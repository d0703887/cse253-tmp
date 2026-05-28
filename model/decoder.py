import math
import torch
import torch.nn as nn

from .encoders import sinusoidal_positional_encoding


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 3) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n_layers):
        in_d = in_dim if i == 0 else hidden_dim
        out_d = out_dim if i == n_layers - 1 else hidden_dim
        layers.append(nn.Linear(in_d, out_d))
        if i < n_layers - 1:
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def modified_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """DDSP modified sigmoid: maps x → (0, 2] with log-10 curvature."""
    return 2.0 * torch.sigmoid(x) ** math.log(10) + 1e-7


class DDSPDecoder(nn.Module):
    """
    Fuses f0(t), loudness(t), z(t) and global timbre h_t to predict
    synthesizer parameters.

    Outputs:
        harmonic_params: [B, T, 101]   (global amp + 100 harmonic amps)
        noise_params:    [B, T, 65]    (noise filter magnitudes)
    """

    def __init__(
        self,
        d_z: int = 16,
        d_t: int = 256,
        mlp_units: int = 512,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        n_harmonics: int = 100,
        n_noise_magnitudes: int = 65,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.mlp_units = mlp_units

        # Independent 3-layer MLPs for each input stream
        self.f0_mlp = build_mlp(1, mlp_units, mlp_units)
        self.loud_mlp = build_mlp(1, mlp_units, mlp_units)
        self.z_mlp = build_mlp(d_z, mlp_units, mlp_units)

        # Project concatenated features → d_model
        fused_dim = mlp_units * 3 + d_t
        self.fuse_proj = nn.Linear(fused_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Skip-connection MLP before output heads
        skip_dim = d_model + mlp_units * 2   # transformer out + f0 + loudness embeddings
        self.pre_head_mlp = nn.Sequential(
            nn.Linear(skip_dim, mlp_units),
            nn.ReLU(),
        )

        self.harmonic_head = nn.Linear(mlp_units, 1 + n_harmonics)
        self.noise_head = nn.Linear(mlp_units, n_noise_magnitudes)

        # Start with small noise and larger harmonic amplitude so harmonics
        # are audible from the first step rather than being buried under noise.
        nn.init.constant_(self.noise_head.bias, -2.0)      # modified_sigmoid(-2) ≈ 0.016
        nn.init.constant_(self.harmonic_head.bias[0], 2.0) # modified_sigmoid(+2) ≈ 1.5 → global_amp

    def forward(
        self,
        f0: torch.Tensor,           # [B, T, 1]
        loudness: torch.Tensor,     # [B, T, 1]
        z: torch.Tensor,            # [B, T, D_z]
        h_t: torch.Tensor,          # [B, D_t]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = f0.shape

        f0_emb = self.f0_mlp(f0)                                    # [B, T, mlp_units]
        loud_emb = self.loud_mlp(loudness)                           # [B, T, mlp_units]
        z_emb = self.z_mlp(z)                                        # [B, T, mlp_units]

        h_t_exp = h_t.unsqueeze(1).expand(-1, T, -1)                # [B, T, D_t]

        fused = torch.cat([f0_emb, loud_emb, z_emb, h_t_exp], dim=-1)  # [B, T, fused_dim]
        x = self.fuse_proj(fused)                                    # [B, T, d_model]

        pe = sinusoidal_positional_encoding(T, self.d_model, x.device)
        x = x + pe.unsqueeze(0)
        x = self.transformer(x)                                      # [B, T, d_model]

        # skip connection with f0 and loudness embeddings
        x = torch.cat([x, f0_emb, loud_emb], dim=-1)                # [B, T, skip_dim]
        x = self.pre_head_mlp(x)                                     # [B, T, mlp_units]

        harmonic_raw = self.harmonic_head(x)                           # [B, T, 101]
        harmonic_params = torch.cat([
            modified_sigmoid(harmonic_raw[:, :, :1]),  # global amp:  (0, 2]
            harmonic_raw[:, :, 1:],                    # harm dist:   raw logits → softmax in synthesizer
        ], dim=-1)                                                     # [B, T, 101]
        noise_params = modified_sigmoid(self.noise_head(x))           # [B, T, 65]

        return harmonic_params, noise_params
