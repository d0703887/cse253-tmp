"""Quick forward-pass shape smoke test (no NSynth data needed)."""
import torch
from config import ModelConfig
from model import TimbreTransferModel

def test():
    cfg = ModelConfig()
    cfg.n_instruments = 10
    model = TimbreTransferModel(cfg)
    model.eval()

    B, T = 2, cfg.n_frames
    mfcc     = torch.randn(B, T, cfg.n_mfcc)
    f0       = torch.rand(B, T, 1) * 440 + 80    # 80-520 Hz
    loudness = torch.randn(B, T, 1)

    with torch.no_grad():
        out = model(mfcc, f0, loudness, grl_lambda=0.5)

    assert out["audio"].shape           == (B, cfg.audio_length),          f"audio: {out['audio'].shape}"
    assert out["z"].shape               == (B, T, cfg.residual_d_z),        f"z: {out['z'].shape}"
    assert out["h_t"].shape             == (B, cfg.timbre_d_t),             f"h_t: {out['h_t'].shape}"
    assert out["harmonic_params"].shape == (B, T, 1 + cfg.n_harmonics),     f"harm: {out['harmonic_params'].shape}"
    assert out["noise_params"].shape    == (B, T, cfg.n_noise_magnitudes),  f"noise: {out['noise_params'].shape}"
    assert out["classifier_logits"].shape == (B, cfg.n_instruments),        f"logits: {out['classifier_logits'].shape}"

    # Timbre transfer
    src_mfcc = torch.randn(B, T, cfg.n_mfcc)
    tgt_mfcc = torch.randn(B, T, cfg.n_mfcc)
    with torch.no_grad():
        transferred = model.transfer(src_mfcc, f0, loudness, tgt_mfcc)
    assert transferred.shape == (B, cfg.audio_length), f"transfer: {transferred.shape}"

    print("All shape assertions passed.")
    print(f"  audio:            {out['audio'].shape}")
    print(f"  z:                {out['z'].shape}")
    print(f"  h_t:              {out['h_t'].shape}")
    print(f"  harmonic_params:  {out['harmonic_params'].shape}")
    print(f"  noise_params:     {out['noise_params'].shape}")
    print(f"  classifier_logits:{out['classifier_logits'].shape}")

if __name__ == "__main__":
    test()
