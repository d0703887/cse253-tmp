import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSpectrogramLoss(nn.Module):
    """
    L1 loss on magnitude and log-magnitude spectrograms at multiple FFT scales.
    FFT sizes: (2048, 1024, 512, 256, 128, 64) with 75% overlap.
    """

    def __init__(self, fft_sizes=(2048, 1024, 512, 256, 128, 64), overlap=0.75):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.overlap = overlap

    def forward(self, target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target, predicted: [B, audio_length]
        Returns:
            scalar loss
        """
        total = torch.tensor(0.0, device=target.device)
        for n_fft in self.fft_sizes:
            hop_length = int(n_fft * (1.0 - self.overlap))
            window = torch.hann_window(n_fft, device=target.device)

            def stft_mag(x):
                s = torch.stft(
                    x,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    window=window,
                    return_complex=True,
                    pad_mode="reflect",
                )
                return s.abs()  # [B, F, T]

            tgt_mag = stft_mag(target)
            pred_mag = stft_mag(predicted)

            # Magnitude L1
            total = total + F.l1_loss(pred_mag, tgt_mag)
            # Log-magnitude L1
            total = total + F.l1_loss(
                torch.log(pred_mag + 1e-7),
                torch.log(tgt_mag + 1e-7),
            )

        return total / len(self.fft_sizes)


class TotalLoss(nn.Module):
    def __init__(self, fft_sizes=(2048, 1024, 512, 256, 128, 64), overlap=0.75,
                 grl_loss_weight: float = 1.0):
        super().__init__()
        self.recon_loss = MultiScaleSpectrogramLoss(fft_sizes, overlap)
        self.grl_loss_weight = grl_loss_weight

    def forward(
        self,
        target_audio: torch.Tensor,
        predicted_audio: torch.Tensor,
        classifier_logits: torch.Tensor,
        instrument_labels: torch.Tensor,
        # grl_lambda: float,
    ) -> dict[str, torch.Tensor]:
        l_recon = self.recon_loss(target_audio, predicted_audio)
        l_cls = F.cross_entropy(classifier_logits, instrument_labels)
        # # GRL handles the adversarial gradient automatically; we minimise l_cls
        # # here normally — the GRL negates the gradient flowing into the encoder.
        # total = l_recon + grl_lambda * self.grl_loss_weight * l_cls

        # [FIX] Do not scale by grl_lambda here!
        # The classifier needs unscaled gradients to train a strong adversary.
        # The GradientReversalLayer handles the lambda scaling on the backward pass
        # specifically for the gradients flowing into the encoder.
        total = l_recon + self.grl_loss_weight * l_cls
        return {"total": total, "reconstruction": l_recon, "classifier": l_cls}
