"""
NSynth dataset loader.

Expects the NSynth dataset in JSON + WAV format:
  <data_dir>/
    examples.json
    audio/
      *.wav

Each example has keys: note_str, instrument, pitch, qualities, etc.
We pre-compute MFCCs, f0 (CREPE), and loudness on-the-fly or from cache.
"""

import json
import os
import math
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


def compute_loudness(audio: torch.Tensor, sample_rate: int, hop_size: int) -> torch.Tensor:
    """
    A-weighted power spectrum loudness, log-scaled, mean-centered.
    Returns [T, 1].
    """
    n_fft = hop_size * 4
    window = torch.hann_window(n_fft)
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_size, win_length=n_fft,
                      window=window, return_complex=True)   # [F, T]
    magnitude = stft.abs()

    # Approximate A-weighting: weight by frequency bin index (simplified)
    freqs = torch.linspace(0, sample_rate / 2, magnitude.shape[0])
    a_weight = _a_weighting(freqs).to(magnitude.device)     # [F]
    power = (magnitude ** 2) * a_weight.unsqueeze(1)

    loudness = 10.0 * torch.log10(power.sum(0) + 1e-8)     # [T]
    loudness = loudness - loudness.mean()
    return loudness.unsqueeze(-1)                            # [T, 1]


def _a_weighting(freqs: torch.Tensor) -> torch.Tensor:
    """Approximate A-weighting curve."""
    f2 = freqs ** 2
    f4 = freqs ** 4
    ra = (12194 ** 2 * f4) / (
        (f2 + 20.6 ** 2)
        * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194 ** 2)
    )
    return ra + 1e-8


class NSynthDataset(Dataset):
    """
    Loads NSynth audio and returns pre-processed features.

    If use_crepe=False, f0 is estimated via autocorrelation (fast fallback).
    Set use_crepe=True and install torchcrepe for production use.
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        audio_length: int = 64000,
        frame_rate: int = 250,
        n_mfcc: int = 30,
        use_crepe: bool = False,
        instrument_to_idx: dict | None = None,
        instrument_families: list[str] | None = None,
        cache_dir: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.frame_rate = frame_rate
        self.n_mfcc = n_mfcc
        self.use_crepe = use_crepe
        # hop_size = audio_length / n_frames = 64000 / 250 = 256 samples
        self.hop = audio_length // frame_rate  # 256
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self._use_cache = self.cache_dir.exists()

        with open(self.data_dir / "examples.json") as f:
            examples = json.load(f)

        # Filter by instrument family and MIDI pitch range (matching DDSP paper)
        keep_families = set(instrument_families) if instrument_families is not None else None
        examples = {
            k: v for k, v in examples.items()
            if (keep_families is None or v["instrument_family_str"] in keep_families)
            and 24 < v["pitch"] < 84
        }
        print(f"Filtered to families={keep_families}, pitch 25–83: {len(examples)} examples")

        self.keys = list(examples.keys())
        self.metadata = examples

        # Build instrument family label mapping
        if instrument_to_idx is None:
            families = sorted({v["instrument_family"] for v in examples.values()})
            self.instrument_to_idx = {fam: i for i, fam in enumerate(families)}
        else:
            self.instrument_to_idx = instrument_to_idx

        # MFCC transform — hop=256 → n_fft=1024 → T=250 frames for 4s audio
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": self.hop * 4,    # 1024
                "hop_length": self.hop,   # 256
                "n_mels": 128,
                "f_min": 20.0,
                "f_max": sample_rate / 2.0,
            },
        )

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        key = self.keys[idx]
        meta = self.metadata[key]

        # Always load raw audio (needed as reconstruction target)
        audio_path = self.data_dir / "audio" / f"{key}.wav"
        waveform, sr = torchaudio.load(str(audio_path))
        waveform = waveform.mean(0)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] < self.audio_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.audio_length - waveform.shape[0]))
        else:
            waveform = waveform[: self.audio_length]

        # Load pre-computed features from cache, or compute on the fly
        cache_path = self.cache_dir / f"{key}.pt"
        if self._use_cache and cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            mfcc = cached["mfcc"]           # [T, n_mfcc]
            f0 = cached["f0"]               # [T, 1]
            loudness = cached["loudness"]   # [T, 1]
        else:
            n_frames = self.audio_length // self.hop
            mfcc = self.mfcc_transform(waveform)[:, :n_frames].T
            if mfcc.shape[0] < n_frames:
                mfcc = torch.nn.functional.pad(mfcc, (0, 0, 0, n_frames - mfcc.shape[0]))
            loudness = compute_loudness(waveform, self.sample_rate, self.hop)[:n_frames]
            if loudness.shape[0] < n_frames:
                loudness = torch.nn.functional.pad(loudness, (0, 0, 0, n_frames - loudness.shape[0]))
            f0 = self._crepe_f0(waveform, n_frames) if self.use_crepe else self._autocorr_f0(waveform, n_frames)

        instrument_label = torch.tensor(
            self.instrument_to_idx[meta["instrument_family"]], dtype=torch.long
        )

        return {
            "audio": waveform,
            "mfcc": mfcc,
            "f0": f0,
            "loudness": loudness,
            "instrument_label": instrument_label,
        }

    def _autocorr_f0(self, waveform: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Fast autocorrelation-based f0 estimation (frame-by-frame)."""
        f0_list = []
        for i in range(n_frames):
            start = i * self.hop
            end = start + self.hop * 4
            frame = waveform[start:end]
            if frame.shape[0] < self.hop * 4:
                frame = torch.nn.functional.pad(frame, (0, self.hop * 4 - frame.shape[0]))
            corr = torch.nn.functional.conv1d(
                frame.unsqueeze(0).unsqueeze(0),
                frame.flip(0).unsqueeze(0).unsqueeze(0),
                padding=frame.shape[0] - 1,
            ).squeeze()
            mid = corr.shape[0] // 2
            corr = corr[mid:]
            # Search for peak in plausible f0 range [50, 2000] Hz
            min_lag = int(self.sample_rate / 2000)
            max_lag = int(self.sample_rate / 50)
            search = corr[min_lag:max_lag]
            if search.numel() == 0 or search.max() <= 0:
                f0_list.append(0.0)
            else:
                lag = search.argmax().item() + min_lag
                f0_list.append(self.sample_rate / lag)
        f0 = torch.tensor(f0_list, dtype=torch.float32).unsqueeze(-1)   # [T, 1]
        return f0

    def _crepe_f0(self, waveform: torch.Tensor, n_frames: int) -> torch.Tensor:
        try:
            import torchcrepe
            f0, _ = torchcrepe.predict(
                waveform.unsqueeze(0),
                self.sample_rate,
                hop_length=self.hop,
                fmin=50.0,
                fmax=2000.0,
                model="tiny",
                return_periodicity=True,
                decoder=torchcrepe.decode.viterbi,
            )
            f0 = f0.squeeze(0)[:n_frames]
            if f0.shape[0] < n_frames:
                f0 = torch.nn.functional.pad(f0, (0, n_frames - f0.shape[0]))
            return f0.unsqueeze(-1)                                       # [T, 1]
        except ImportError:
            return self._autocorr_f0(waveform, n_frames)
