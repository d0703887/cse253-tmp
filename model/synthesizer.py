import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_to_audio(x: torch.Tensor, audio_length: int) -> torch.Tensor:
    """Linearly upsample [B, T, C] → [B, audio_length, C]."""
    x = x.permute(0, 2, 1)                          # [B, C, T]
    x = F.interpolate(x, size=audio_length, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)                        # [B, audio_length, C]


class AdditiveSynthesizer(nn.Module):
    """
    Converts harmonic parameters + f0(t) into a harmonic waveform.
    """

    def __init__(self, sample_rate: int = 16000, audio_length: int = 64000, n_harmonics: int = 100):
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.n_harmonics = n_harmonics
        harmonic_nums = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        self.register_buffer("harmonic_nums", harmonic_nums)

    def forward(self, harmonic_params: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        B = f0.shape[0]

        global_amp = harmonic_params[:, :, :1]
        harm_dist = harmonic_params[:, :, 1:]
        harm_dist = F.softmax(harm_dist, dim=-1)

        # Per-harmonic amplitudes
        harm_amps = global_amp * harm_dist

        # Upsample to audio rate
        f0_up = upsample_to_audio(f0, self.audio_length)
        harm_amps_up = upsample_to_audio(harm_amps, self.audio_length)

        freqs = f0_up * self.harmonic_nums.view(1, 1, -1)

        # [FIX 1] Mask amplitudes above Nyquist, not frequencies
        nyquist = self.sample_rate / 2.0
        nyquist_mask = (freqs < nyquist).float()
        harm_amps_up = harm_amps_up * nyquist_mask

        # [FIX 3] Cast to float64 for cumulative sum to prevent phase drift
        dt = 1.0 / self.sample_rate
        phase_increments = 2.0 * torch.pi * freqs * dt
        phases = torch.cumsum(phase_increments.double(), dim=1).float()

        # Sum sinusoids
        sinusoids = harm_amps_up * torch.sin(phases)
        return sinusoids.sum(dim=-1)


class SubtractiveSynthesizer(nn.Module):
    """
    Converts noise filter magnitudes into a shaped noise waveform.
    """

    def __init__(self, sample_rate: int = 16000, audio_length: int = 64000, n_magnitudes: int = 65):
        super().__init__()
        self.audio_length = audio_length
        self.n_magnitudes = n_magnitudes
        self.filter_length = (n_magnitudes - 1) * 2  # 128

    def forward(self, noise_params: torch.Tensor) -> torch.Tensor:
        B, T, _ = noise_params.shape

        # White noise source
        noise = torch.randn(B, self.audio_length, device=noise_params.device)

        hop = self.filter_length // 2  # 64
        window = torch.hann_window(self.filter_length, device=noise_params.device)

        noise_stft = torch.stft(
            noise,
            n_fft=self.filter_length,
            hop_length=hop,
            win_length=self.filter_length,
            window=window,
            return_complex=True,
            pad_mode="reflect",
        )  # [B, 65, n_frames_stft]

        n_frames_stft = noise_stft.shape[2]

        # [FIX 2] Interpolate directly from frame sequence T to n_frames_stft
        mags_t = noise_params.permute(0, 2, 1)  # [B, 65, T]
        mags_resampled = F.interpolate(
            mags_t, size=n_frames_stft, mode="linear", align_corners=False
        )  # [B, 65, n_frames_stft]

        # Multiply in frequency domain
        filtered_stft = noise_stft * mags_resampled

        # ISTFT back to waveform
        output = torch.istft(
            filtered_stft,
            n_fft=self.filter_length,
            hop_length=hop,
            win_length=self.filter_length,
            window=window,
            length=self.audio_length,
        )

        return output


class DDSPSynthesizer(nn.Module):
    """Combines additive + subtractive synthesizers."""

    def __init__(self, sample_rate: int = 16000, audio_length: int = 64000,
                 n_harmonics: int = 100, n_noise_magnitudes: int = 65):
        super().__init__()
        self.additive = AdditiveSynthesizer(sample_rate, audio_length, n_harmonics)
        self.subtractive = SubtractiveSynthesizer(sample_rate, audio_length, n_noise_magnitudes)

    def forward(
        self,
        harmonic_params: torch.Tensor,   # [B, T, 101]
        noise_params: torch.Tensor,      # [B, T, 65]
        f0: torch.Tensor,                # [B, T, 1]
    ) -> torch.Tensor:
        """Returns audio [B, audio_length]."""
        harmonic_audio = self.additive(harmonic_params, f0)
        noise_audio = self.subtractive(noise_params)
        return harmonic_audio + noise_audio
