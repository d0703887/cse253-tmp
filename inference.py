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
import torchaudio

from config import ModelConfig
from dataset import NSynthDataset
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


def extract_features(waveform: torch.Tensor, dataset: NSynthDataset) -> dict:
    n_frames = dataset.audio_length // dataset.hop
    import torchaudio.transforms as T
    mfcc_transform = T.MFCC(
        sample_rate=dataset.sample_rate,
        n_mfcc=dataset.n_mfcc,
        melkwargs={
            "n_fft": dataset.hop * 4,    # 1024
            "hop_length": dataset.hop,   # 256
            "n_mels": 128,
            "f_min": 20.0,
            "f_max": dataset.sample_rate / 2.0,
        },
    )
    mfcc = mfcc_transform(waveform)[:, :n_frames]
    if mfcc.shape[1] < n_frames:
        mfcc = torch.nn.functional.pad(mfcc, (0, n_frames - mfcc.shape[1]))
    mfcc = mfcc.T

    from dataset import compute_loudness
    loudness = compute_loudness(waveform, dataset.sample_rate, dataset.hop)[:n_frames]
    if loudness.shape[0] < n_frames:
        loudness = torch.nn.functional.pad(loudness, (0, 0, 0, n_frames - loudness.shape[0]))

    f0 = dataset._autocorr_f0(waveform, n_frames)

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

    dummy_dataset = NSynthDataset.__new__(NSynthDataset)
    dummy_dataset.sample_rate = model_cfg.sample_rate
    dummy_dataset.audio_length = model_cfg.audio_length
    dummy_dataset.frame_rate = model_cfg.frame_rate
    dummy_dataset.n_mfcc = model_cfg.n_mfcc
    dummy_dataset.hop = model_cfg.audio_length // model_cfg.frame_rate

    src_wav = load_audio(args.source, model_cfg.sample_rate, model_cfg.audio_length)
    src_feats = extract_features(src_wav, dummy_dataset)

    def to_batch(t):
        return t.unsqueeze(0).to(device)

    if args.mode == "reconstruct":
        outputs = model(
            to_batch(src_feats["mfcc"]),
            to_batch(src_feats["f0"]),
            to_batch(src_feats["loudness"]),
            grl_lambda=0.0,
        )
        output_audio = outputs["audio"]  # [1, audio_length]
    else:
        if args.target is None:
            raise ValueError("--target is required for transfer mode")
        tgt_wav = load_audio(args.target, model_cfg.sample_rate, model_cfg.audio_length)
        tgt_feats = extract_features(tgt_wav, dummy_dataset)
        output_audio = model.transfer(
            source_mfcc=to_batch(src_feats["mfcc"]),
            source_f0=to_batch(src_feats["f0"]),
            source_loudness=to_batch(src_feats["loudness"]),
            target_mfcc=to_batch(tgt_feats["mfcc"]),
        )  # [1, audio_length]

    output_audio = output_audio.squeeze(0).cpu()
    output_audio = output_audio / (output_audio.abs().max() + 1e-8)   # peak normalise
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
