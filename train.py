"""
Training script for the Timbre Transfer model on NSynth.

Usage:
    python train.py --data_dir data/nsynth-train --val_dir data/nsynth-valid
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import ModelConfig, TrainConfig
from dataset import NSynthDataset
from model import TimbreTransferModel
from losses import TotalLoss


def grl_lambda_schedule(step: int, max_lambda: float, ramp_steps: int) -> float:
    """Ramp lambda from 0 → max_lambda following the original GRL paper schedule."""
    progress = min(step / ramp_steps, 1.0)
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    if args.data_dir:
        train_cfg.nsynth_data_dir = args.data_dir

    families = args.families.split(",") if args.families else train_cfg.instrument_families

    # Build datasets
    train_dataset = NSynthDataset(
        data_dir=args.data_dir,
        sample_rate=model_cfg.sample_rate,
        audio_length=model_cfg.audio_length,
        frame_rate=model_cfg.frame_rate,
        n_mfcc=model_cfg.n_mfcc,
        use_crepe=args.use_crepe,
        instrument_families=families,
    )
    model_cfg.n_instruments = len(train_dataset.instrument_to_idx)
    print(f"Dataset: {len(train_dataset)} examples, {model_cfg.n_instruments} instruments")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    val_loader = None
    if args.val_dir:
        val_dataset = NSynthDataset(
            data_dir=args.val_dir,
            sample_rate=model_cfg.sample_rate,
            audio_length=model_cfg.audio_length,
            frame_rate=model_cfg.frame_rate,
            n_mfcc=model_cfg.n_mfcc,
            use_crepe=args.use_crepe,
            instrument_to_idx=train_dataset.instrument_to_idx,
            instrument_families=families,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = TimbreTransferModel(model_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    # Cosine annealing with linear warmup
    def lr_lambda(step):
        if step < train_cfg.warmup_steps:
            return step / max(1, train_cfg.warmup_steps)
        total_steps = len(train_loader) * train_cfg.n_epochs
        progress = (step - train_cfg.warmup_steps) / max(1, total_steps - train_cfg.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = TotalLoss(
        fft_sizes=model_cfg.fft_sizes,
        overlap=model_cfg.overlap,
        grl_loss_weight=model_cfg.grl_loss_weight,
    )

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    epoch_bar = tqdm(range(train_cfg.n_epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_totals = {"total": 0.0, "reconstruction": 0.0, "classifier": 0.0}
        n_batches = 0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False, position=0)
        for batch in batch_bar:
            audio = batch["audio"].to(device)
            mfcc = batch["mfcc"].to(device)
            f0 = batch["f0"].to(device)
            loudness = batch["loudness"].to(device)
            instrument_labels = batch["instrument_label"].to(device)

            lam = grl_lambda_schedule(global_step, model_cfg.grl_lambda_max, train_cfg.grl_ramp_steps)

            outputs = model(mfcc, f0, loudness, grl_lambda=lam)
            losses = criterion(
                target_audio=audio,
                predicted_audio=outputs["audio"],
                classifier_logits=outputs["classifier_logits"],
                instrument_labels=instrument_labels,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            for k in train_totals:
                train_totals[k] += losses[k].item()
            n_batches += 1

            if global_step % train_cfg.save_interval == 0:
                ckpt_path = checkpoint_dir / f"step_{global_step:07d}.pt"
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "model_cfg": model_cfg,
                    "instrument_to_idx": train_dataset.instrument_to_idx,
                }, ckpt_path)
                tqdm.write(f"Saved checkpoint: {ckpt_path}")

        # Epoch-end summary
        tr = {k: v / max(n_batches, 1) for k, v in train_totals.items()}
        summary = (
            f"Epoch {epoch+1:>3} | "
            f"Train — loss: {tr['total']:.4f}  recon: {tr['reconstruction']:.4f}  cls: {tr['classifier']:.4f}"
        )
        if val_loader is not None:
            val = _validate(model, criterion, val_loader, device, epoch)
            summary += (
                f"  ||  Val — loss: {val['total']:.4f}  recon: {val['recon']:.4f}  cls: {val['classifier']:.4f}"
            )
        tqdm.write(summary)

    print("Training complete.")


@torch.no_grad()
def _validate(model, criterion, val_loader, device, epoch) -> dict:
    model.eval()
    totals = {"total": 0.0, "reconstruction": 0.0, "classifier": 0.0}
    n = 0
    for batch in tqdm(val_loader, desc="Validation", unit="batch", leave=False):
        audio = batch["audio"].to(device)
        mfcc = batch["mfcc"].to(device)
        f0 = batch["f0"].to(device)
        loudness = batch["loudness"].to(device)
        instrument_labels = batch["instrument_label"].to(device)
        outputs = model(mfcc, f0, loudness, grl_lambda=0.0)
        losses = criterion(
            target_audio=audio,
            predicted_audio=outputs["audio"],
            classifier_logits=outputs["classifier_logits"],
            instrument_labels=instrument_labels,
        )
        for k in totals:
            totals[k] += losses[k].item()
        n += 1

    metrics = {k: v / max(n, 1) for k, v in totals.items()}
    metrics["recon"] = metrics.pop("reconstruction")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to NSynth train split")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to NSynth valid split")
    parser.add_argument("--use_crepe", action="store_true", help="Use CREPE for f0 extraction")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--families", type=str, default=None,
        help="Comma-separated instrument families (overrides config.py). e.g. keyboard,guitar,bass,string"
    )
    args = parser.parse_args()
    train(args)
