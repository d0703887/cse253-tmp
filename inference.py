"""
Inference & timbre transfer.

Usage:
    python inference.py \
        --checkpoint checkpoints/step_0010000.pt \
        --source source.wav \
        --target target.wav \
        --output output.wav
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from config import ModelConfig
from dataset import compute_loudness
from losses import MultiScaleSpectrogramLoss
from model import TimbreTransferModel


def load_audio(path: str, sample_rate: int, audio_length: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(0)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] < audio_length:
        waveform = torch.nn.functional.pad(waveform, (0, audio_length - waveform.shape[0]))
    else:
        waveform = waveform[:audio_length]
    return waveform


def extract_features(waveform: torch.Tensor, cfg: ModelConfig) -> dict:
    """Extract MFCC, f0 (CREPE), and loudness — identical pipeline to preprocess.py."""
    import torchaudio.transforms as T
    import torchcrepe

    hop = cfg.audio_length // cfg.frame_rate
    n_frames = cfg.audio_length // hop

    mfcc_transform = T.MFCC(
        sample_rate=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        melkwargs={
            "n_fft": hop * 4,
            "hop_length": hop,
            "n_mels": 128,
            "f_min": 20.0,
            "f_max": cfg.sample_rate / 2.0,
        },
    )
    mfcc = mfcc_transform(waveform)[:, :n_frames]
    if mfcc.shape[1] < n_frames:
        mfcc = F.pad(mfcc, (0, n_frames - mfcc.shape[1]))
    mfcc = mfcc.T  # [T, n_mfcc]

    loudness = compute_loudness(waveform, cfg.sample_rate, hop)[:n_frames]
    if loudness.shape[0] < n_frames:
        loudness = F.pad(loudness, (0, 0, 0, n_frames - loudness.shape[0]))

    f0, _ = torchcrepe.predict(
        waveform.unsqueeze(0),
        cfg.sample_rate,
        hop_length=hop,
        fmin=50.0,
        fmax=2000.0,
        model="tiny",
        return_periodicity=True,
        decoder=torchcrepe.decode.viterbi,
    )
    f0 = f0.squeeze(0)[:n_frames]
    if f0.shape[0] < n_frames:
        f0 = F.pad(f0, (0, n_frames - f0.shape[0]))
    f0 = f0.unsqueeze(-1)  # [T, 1]

    return {"mfcc": mfcc, "f0": f0, "loudness": loudness}


@torch.no_grad()
def transfer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg: ModelConfig = ckpt["model_cfg"]
    instrument_to_idx: dict = ckpt["instrument_to_idx"]
    model_cfg.n_instruments = len(instrument_to_idx)

    model = TimbreTransferModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    src_wav = load_audio(args.source, model_cfg.sample_rate, model_cfg.audio_length)
    src_feats = extract_features(src_wav, model_cfg)

    def to_batch(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).to(device)

    if args.mode == "reconstruct":
        outputs = model(
            to_batch(src_feats["mfcc"]),
            to_batch(src_feats["f0"]),
            to_batch(src_feats["loudness"]),
            grl_lambda=0.0,
        )
        output_audio = outputs["audio"].squeeze(0).cpu()  # [audio_length]

        # ── evaluation metrics ────────────────────────────────────────────────
        print("\n── Reconstruction metrics ──")

        # Multi-scale spectral loss (same config as training)
        mss = MultiScaleSpectrogramLoss(fft_sizes=model_cfg.fft_sizes, overlap=model_cfg.overlap)
        spectral_loss = mss(src_wav.unsqueeze(0), output_audio.unsqueeze(0)).item()
        print(f"  Multi-scale spectral loss : {spectral_loss:.4f}")

        # f0 and loudness of the reconstructed audio
        print("  Extracting features from reconstructed audio (CREPE)...")
        recon_feats = extract_features(output_audio, model_cfg)

        f0_l1 = F.l1_loss(recon_feats["f0"], src_feats["f0"]).item()
        loudness_l1 = F.l1_loss(recon_feats["loudness"], src_feats["loudness"]).item()
        print(f"  f0 L1                     : {f0_l1:.4f} Hz")
        print(f"  Loudness L1               : {loudness_l1:.4f} dB")
        print()
        # ─────────────────────────────────────────────────────────────────────

    else:
        if args.target is None:
            raise ValueError("--target is required for transfer mode")
        tgt_wav = load_audio(args.target, model_cfg.sample_rate, model_cfg.audio_length)
        tgt_feats = extract_features(tgt_wav, model_cfg)
        output_audio = model.transfer(
            source_mfcc=to_batch(src_feats["mfcc"]),
            source_f0=to_batch(src_feats["f0"]),
            source_loudness=to_batch(src_feats["loudness"]),
            target_mfcc=to_batch(tgt_feats["mfcc"]),
        ).squeeze(0).cpu()  # [audio_length]

    output_audio = output_audio / (output_audio.abs().max() + 1e-8)  # peak normalise
    torchaudio.save(args.output, output_audio.unsqueeze(0), model_cfg.sample_rate)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="Source instrument audio")
    parser.add_argument("--target", type=str, default=None, help="Target instrument audio (transfer mode only)")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--mode", type=str, default="transfer", choices=["transfer", "reconstruct"],
                        help="transfer: swap timbre; reconstruct: encode+decode source only")
    args = parser.parse_args()
    transfer(args)
