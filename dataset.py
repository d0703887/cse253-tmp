"""
NSynth dataset loader.

Expects the NSynth dataset in JSON + WAV format:
  <data_dir>/
    examples.json
    audio/
      *.wav
  cache/
    *.pt   (pre-computed by preprocess.py)

All features (MFCC, f0, loudness) must be pre-computed via preprocess.py.
"""

import json
import math
from pathlib import Path

import torch
import torchaudio
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
    Loads NSynth audio and cached features (MFCC, f0, loudness).
    All features must be pre-computed via preprocess.py before training.
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        audio_length: int = 64000,
        frame_rate: int = 250,
        n_mfcc: int = 30,
        instrument_to_idx: dict | None = None,
        instrument_families: list[str] | None = None,
        cache_dir: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.frame_rate = frame_rate
        self.n_mfcc = n_mfcc
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

        # MFCC transform kept for reference (not used during training)
        import torchaudio.transforms as T
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

        # Load raw audio (reconstruction target)
        audio_path = self.data_dir / "audio" / f"{key}.wav"
        waveform, sr = torchaudio.load(str(audio_path))
        waveform = waveform.mean(0)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] < self.audio_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.audio_length - waveform.shape[0]))
        else:
            waveform = waveform[: self.audio_length]

        # Load pre-computed features — must exist, run preprocess.py if missing
        cache_path = self.cache_dir / f"{key}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache missing for '{key}'. Run preprocess.py on {self.data_dir} first."
            )
        cached = torch.load(cache_path, weights_only=True)
        mfcc     = cached["mfcc"]       # [T, n_mfcc]
        f0       = cached["f0"]         # [T, 1]
        loudness = cached["loudness"]   # [T, 1]

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
