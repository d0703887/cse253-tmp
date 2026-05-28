"""
Probing classifier: verifies disentanglement by training a fresh MLP on
frozen z(t) features (no GRL) and measuring how well it recovers instrument identity.

Interpretation:
  - Accuracy ≈ 1/n_classes (random)  → z(t) contains no instrument information  ✓
  - Accuracy >> 1/n_classes           → z(t) still leaks instrument identity     ✗

Usage:
    python probe.py --checkpoint checkpoints/step_0010000.pt \
                    --data_dir data/nsynth-train-tiny \
                    --val_dir  data/nsynth-valid-tiny
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import ModelConfig, TrainConfig
from dataset import NSynthDataset
from model import TimbreTransferModel


class ProbeClassifier(nn.Module):
    def __init__(self, d_z: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.mean(dim=1))   # mean-pool over time → [B, n_classes]


def run_probe(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model, freeze encoder
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg: ModelConfig = ckpt["model_cfg"]
    instrument_to_idx: dict = ckpt["instrument_to_idx"]
    model_cfg.n_instruments = len(instrument_to_idx)

    model = TimbreTransferModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    train_cfg = TrainConfig()
    families = train_cfg.instrument_families

    train_dataset = NSynthDataset(
        data_dir=args.data_dir,
        sample_rate=model_cfg.sample_rate,
        audio_length=model_cfg.audio_length,
        frame_rate=model_cfg.frame_rate,
        n_mfcc=model_cfg.n_mfcc,
        instrument_to_idx=instrument_to_idx,
        instrument_families=families,
    )
    val_dataset = NSynthDataset(
        data_dir=args.val_dir,
        sample_rate=model_cfg.sample_rate,
        audio_length=model_cfg.audio_length,
        frame_rate=model_cfg.frame_rate,
        n_mfcc=model_cfg.n_mfcc,
        instrument_to_idx=instrument_to_idx,
        instrument_families=families,
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2)

    n_classes = len(instrument_to_idx)
    probe = ProbeClassifier(model_cfg.residual_d_z, n_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Probe: {n_classes} classes, random baseline = {1/n_classes:.2%}")
    print(f"Training probe for {args.epochs} epochs on frozen z(t)...\n")

    for epoch in range(args.epochs):
        probe.train()
        for batch in tqdm(train_loader, desc=f"Probe epoch {epoch+1}", leave=False):
            mfcc   = batch["mfcc"].to(device)
            labels = batch["instrument_label"].to(device)
            with torch.no_grad():
                z = model.residual_encoder(mfcc)   # [B, T, D_z]
            logits = probe(z)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation accuracy
        probe.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                mfcc   = batch["mfcc"].to(device)
                labels = batch["instrument_label"].to(device)
                z = model.residual_encoder(mfcc)
                preds = probe(z).argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = correct / max(total, 1)
        print(f"Epoch {epoch+1:>2} | Probe val accuracy: {acc:.2%}  (random = {1/n_classes:.2%})")

    print("\nConclusion:")
    if acc <= 1/n_classes + 0.05:
        print(f"  z(t) contains NO instrument information (acc {acc:.2%} ≈ random {1/n_classes:.2%}) ✓")
    else:
        print(f"  z(t) still leaks instrument identity (acc {acc:.2%} >> random {1/n_classes:.2%}) ✗")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--val_dir",    type=str, required=True)
    parser.add_argument("--epochs",     type=int, default=20)
    args = parser.parse_args()
    run_probe(args)
