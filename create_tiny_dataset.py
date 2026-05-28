"""
Copy the filtered NSynth subset (families + pitch range + per-family cap) to tiny folders.

Usage:
    python create_tiny_dataset.py

Reads all settings from config.py.
Output:
    data/nsynth-train-tiny/  (examples.json + audio/ + cache/)
    data/nsynth-valid-tiny/  (examples.json + audio/ + cache/)
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from config import ModelConfig, TrainConfig


def subsample_per_family(examples: dict, max_per_family: int, seed: int) -> dict:
    """Randomly sample up to max_per_family examples per instrument family."""
    by_family = defaultdict(list)
    for k, v in examples.items():
        by_family[v["instrument_family_str"]].append(k)

    rng = random.Random(seed)
    kept = {}
    for family, keys in sorted(by_family.items()):
        sampled = rng.sample(keys, min(len(keys), max_per_family))
        for k in sampled:
            kept[k] = examples[k]
        print(f"  {family}: {len(keys)} available → {len(sampled)} kept")

    return kept


def copy_split(src_dir: Path, dst_dir: Path, train_cfg: TrainConfig):
    src_json = src_dir / "examples.json"
    if not src_json.exists():
        print(f"Skipping {src_dir} — examples.json not found.")
        return

    with open(src_json) as f:
        examples = json.load(f)

    families = set(train_cfg.instrument_families)

    # Step 1: filter by family, source type, and pitch
    source_map = train_cfg.instrument_source_map
    filtered = {
        k: v for k, v in examples.items()
        if v["instrument_family_str"] in families
        and 24 < v["pitch"] < 84
        and v.get("instrument_source_str") == source_map.get(v["instrument_family_str"])
    }
    print(f"\n{src_dir.name}: {len(examples)} total → {len(filtered)} after family+pitch+source filter")

    # Step 2: cap per family
    if train_cfg.max_per_family is not None:
        filtered = subsample_per_family(filtered, train_cfg.max_per_family, train_cfg.dataset_seed)
    print(f"  → {len(filtered)} examples after {train_cfg.max_per_family}/family cap")

    # Step 3: copy files
    dst_audio_dir = dst_dir / "audio"
    dst_audio_dir.mkdir(parents=True, exist_ok=True)

    src_cache_dir = src_dir / "cache"
    dst_cache_dir = dst_dir / "cache"
    has_cache = src_cache_dir.exists()
    if has_cache:
        dst_cache_dir.mkdir(parents=True, exist_ok=True)

    missing_wav, missing_cache = [], []
    for key in tqdm(filtered, desc=f"Copying {dst_dir.name}", unit="file"):
        src_wav = src_dir / "audio" / f"{key}.wav"
        dst_wav = dst_audio_dir / f"{key}.wav"
        if not dst_wav.exists():
            if src_wav.exists():
                shutil.copy2(src_wav, dst_wav)
            else:
                missing_wav.append(key)

        if has_cache:
            src_pt = src_cache_dir / f"{key}.pt"
            dst_pt = dst_cache_dir / f"{key}.pt"
            if not dst_pt.exists():
                if src_pt.exists():
                    shutil.copy2(src_pt, dst_pt)
                else:
                    missing_cache.append(key)

    with open(dst_dir / "examples.json", "w") as f:
        json.dump(filtered, f, indent=4)

    print(f"  Wrote {dst_dir / 'examples.json'}")
    if missing_wav:
        print(f"  WARNING: {len(missing_wav)} audio files not found — skipped.")
    if missing_cache:
        print(f"  NOTE: {len(missing_cache)} cache files missing — run preprocess.py to generate.")


def main():
    train_cfg = TrainConfig()
    print(f"Families: {train_cfg.instrument_families}")
    print(f"Sources: {train_cfg.instrument_source_map}")
    print(f"Pitch range: 25–83")
    print(f"Cap: {train_cfg.max_per_family} per family (seed={train_cfg.dataset_seed})")

    base = Path("data")
    splits = [
        (base / "nsynth-train", base / "nsynth-train-tiny"),
        (base / "nsynth-valid", base / "nsynth-valid-tiny"),
    ]

    for src, dst in splits:
        copy_split(src, dst, train_cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
