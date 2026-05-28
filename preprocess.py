"""
Pre-compute and cache MFCC, f0, and loudness for every NSynth example.

Usage:
    python preprocess.py --data_dir data/nsynth-valid
    python preprocess.py --data_dir data/nsynth-train --use_crepe --num_workers 8

Output: <data_dir>/cache/<key>.pt  containing {mfcc, f0, loudness} tensors.
Re-running skips already-processed files, so it is safe to resume.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm.auto import tqdm

from config import ModelConfig, TrainConfig
from dataset import compute_loudness, _a_weighting


# ── worker (must be a module-level function for multiprocessing) ──────────────

def _process_one(args_tuple):
    (key, audio_path, cache_path, sample_rate, audio_length,
     hop, n_mfcc) = args_tuple

    if cache_path.exists():
        return key, "skip"

    try:
        waveform, sr = torchaudio.load(str(audio_path))
        waveform = waveform.mean(0)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        if waveform.shape[0] < audio_length:
            waveform = torch.nn.functional.pad(waveform, (0, audio_length - waveform.shape[0]))
        else:
            waveform = waveform[:audio_length]

        n_frames = audio_length // hop   # 250

        # MFCC
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": hop * 4,
                "hop_length": hop,
                "n_mels": 128,
                "f_min": 20.0,
                "f_max": sample_rate / 2.0,
            },
        )
        mfcc = mfcc_transform(waveform)[:, :n_frames]   # [n_mfcc, T]
        if mfcc.shape[1] < n_frames:
            mfcc = torch.nn.functional.pad(mfcc, (0, n_frames - mfcc.shape[1]))
        mfcc = mfcc.T                                    # [T, n_mfcc]

        # Loudness
        loudness = compute_loudness(waveform, sample_rate, hop)[:n_frames]  # [T, 1]
        if loudness.shape[0] < n_frames:
            loudness = torch.nn.functional.pad(loudness, (0, 0, 0, n_frames - loudness.shape[0]))

        # f0
        import torchcrepe
        f0, _ = torchcrepe.predict(
            waveform.unsqueeze(0),
            sample_rate,
            hop_length=hop,
            fmin=50.0,
            fmax=2000.0,
            model="tiny",
            return_periodicity=True,
            decoder=torchcrepe.decode.viterbi,
        )
        f0 = f0.squeeze(0)[:n_frames]
        if f0.shape[0] < n_frames:
            f0 = torch.nn.functional.pad(f0, (0, n_frames - f0.shape[0]))
        f0 = f0.unsqueeze(-1)

        torch.save({"mfcc": mfcc, "f0": f0, "loudness": loudness}, cache_path)
        return key, "ok"

    except Exception as e:
        return key, f"error: {e}"


def _autocorr_f0(waveform, sample_rate, hop, n_frames):
    f0_list = []
    for i in range(n_frames):
        start = i * hop
        frame = waveform[start: start + hop * 4]
        if frame.shape[0] < hop * 4:
            frame = torch.nn.functional.pad(frame, (0, hop * 4 - frame.shape[0]))
        corr = torch.nn.functional.conv1d(
            frame.unsqueeze(0).unsqueeze(0),
            frame.flip(0).unsqueeze(0).unsqueeze(0),
            padding=frame.shape[0] - 1,
        ).squeeze()
        mid = corr.shape[0] // 2
        corr = corr[mid:]
        min_lag = int(sample_rate / 2000)
        max_lag = int(sample_rate / 50)
        search = corr[min_lag:max_lag]
        if search.numel() == 0 or search.max() <= 0:
            f0_list.append(0.0)
        else:
            lag = search.argmax().item() + min_lag
            f0_list.append(sample_rate / lag)
    return torch.tensor(f0_list, dtype=torch.float32).unsqueeze(-1)  # [T, 1]


# ── main ──────────────────────────────────────────────────────────────────────

def preprocess(args):
    data_dir = Path(args.data_dir)
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    with open(data_dir / "examples.json") as f:
        examples = json.load(f)

    # Read settings from config
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    families = set(args.families.split(",") if args.families else train_cfg.instrument_families)

    # Parse optional per-family source filter, e.g. "bass:acoustic,guitar:acoustic,keyboard:electronic"
    source_map: dict[str, str] = {}
    if args.source_map:
        for pair in args.source_map.split(","):
            fam, src = pair.strip().split(":")
            source_map[fam.strip()] = src.strip()

    def _passes_source(v: dict) -> bool:
        if not source_map:
            return True
        required = source_map.get(v["instrument_family_str"])
        return required is None or v.get("instrument_source_str") == required

    examples = {
        k: v for k, v in examples.items()
        if v["instrument_family_str"] in families
        and 24 < v["pitch"] < 84
        and _passes_source(v)
    }
    src_info = f", sources: {source_map}" if source_map else ""
    print(f"Families: {families}{src_info}, pitch 25–83 — {len(examples)} examples selected.")

    sample_rate = model_cfg.sample_rate
    audio_length = model_cfg.audio_length
    hop = audio_length // model_cfg.frame_rate
    n_mfcc = model_cfg.n_mfcc

    tasks = [
        (
            key,
            data_dir / "audio" / f"{key}.wav",
            cache_dir / f"{key}.pt",
            sample_rate, audio_length, hop, n_mfcc,
        )
        for key in examples
    ]

    already_done = sum(1 for t in tasks if t[2].exists())
    print(f"{len(tasks)} examples total, {already_done} already cached, "
          f"{len(tasks) - already_done} to process.")

    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_process_one, tasks),
                total=len(tasks),
                desc="Preprocessing",
                unit="file",
            ))
    else:
        results = [_process_one(t) for t in tqdm(tasks, desc="Preprocessing", unit="file")]

    errors = [(k, msg) for k, msg in results if msg.startswith("error")]
    print(f"Done. {len(tasks) - len(errors)} succeeded, {len(errors)} failed.")
    for k, msg in errors:
        print(f"  FAILED {k}: {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--families", type=str, default=None,
        help="Comma-separated families to preprocess (overrides config.py)"
    )
    parser.add_argument(
        "--source_map", type=str, default=None,
        help='Per-family source filter, e.g. "bass:acoustic,guitar:acoustic,keyboard:electronic"'
    )
    args = parser.parse_args()
    preprocess(args)



