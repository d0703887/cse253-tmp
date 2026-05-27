import torch
import torch.nn as nn

from .encoders import ResidualEncoder, GlobalTimbreEncoder
from .grl import InstrumentClassifier
from .decoder import DDSPDecoder
from .synthesizer import DDSPSynthesizer
from config import ModelConfig


class TimbreTransferModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.residual_encoder = ResidualEncoder(
            n_mfcc=cfg.n_mfcc,
            d_model=cfg.residual_d_model,
            n_layers=cfg.residual_n_layers,
            n_heads=cfg.residual_n_heads,
            d_z=cfg.residual_d_z,
        )
        self.timbre_encoder = GlobalTimbreEncoder(
            n_mfcc=cfg.n_mfcc,
            d_model=cfg.timbre_d_model,
            n_layers=cfg.timbre_n_layers,
            n_heads=cfg.timbre_n_heads,
            d_t=cfg.timbre_d_t,
        )
        self.instrument_classifier = InstrumentClassifier(
            d_z=cfg.residual_d_z,
            n_instruments=cfg.n_instruments,
        )
        self.decoder = DDSPDecoder(
            d_z=cfg.residual_d_z,
            d_t=cfg.timbre_d_t,
            mlp_units=cfg.decoder_mlp_units,
            d_model=cfg.decoder_d_model,
            n_layers=cfg.decoder_n_layers,
            n_heads=cfg.decoder_n_heads,
            n_harmonics=cfg.n_harmonics,
            n_noise_magnitudes=cfg.n_noise_magnitudes,
        )
        self.synthesizer = DDSPSynthesizer(
            sample_rate=cfg.sample_rate,
            audio_length=cfg.audio_length,
            n_harmonics=cfg.n_harmonics,
            n_noise_magnitudes=cfg.n_noise_magnitudes,
        )

    def encode(self, mfcc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (z, h_t)."""
        z = self.residual_encoder(mfcc)
        h_t = self.timbre_encoder(mfcc)
        return z, h_t

    def decode_and_synthesize(
        self,
        f0: torch.Tensor,
        loudness: torch.Tensor,
        z: torch.Tensor,
        h_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (audio, harmonic_params, noise_params)."""
        harmonic_params, noise_params = self.decoder(f0, loudness, z, h_t)
        audio = self.synthesizer(harmonic_params, noise_params, f0)
        return audio, harmonic_params, noise_params

    def forward(
        self,
        mfcc: torch.Tensor,         # [B, T, 30]
        f0: torch.Tensor,           # [B, T, 1]
        loudness: torch.Tensor,     # [B, T, 1]
        grl_lambda: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        z, h_t = self.encode(mfcc)
        audio, harmonic_params, noise_params = self.decode_and_synthesize(f0, loudness, z, h_t)
        classifier_logits = self.instrument_classifier(z, lambda_=grl_lambda)
        return {
            "audio": audio,
            "z": z,
            "h_t": h_t,
            "harmonic_params": harmonic_params,
            "noise_params": noise_params,
            "classifier_logits": classifier_logits,
        }

    def transfer(
        self,
        source_mfcc: torch.Tensor,
        source_f0: torch.Tensor,
        source_loudness: torch.Tensor,
        target_mfcc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Timbre transfer: encode source content, encode target timbre,
        decode and synthesize.

        Returns audio [B, audio_length].
        """
        z, _ = self.encode(source_mfcc)
        _, h_t = self.encode(target_mfcc)
        audio, _, _ = self.decode_and_synthesize(source_f0, source_loudness, z, h_t)
        return audio
