import dataclasses


@dataclasses.dataclass
class ModelConfig:
    # Audio
    sample_rate: int = 16000
    audio_length: int = 64000          # 4 seconds
    frame_rate: int = 250
    n_frames: int = 250
    hop_size: int = 256

    # MFCC input
    n_mfcc: int = 30

    # Residual encoder
    residual_d_model: int = 256
    residual_n_layers: int = 4
    residual_n_heads: int = 8
    residual_d_z: int = 16             # D_z

    # Timbre encoder
    timbre_d_model: int = 256
    timbre_n_layers: int = 4
    timbre_n_heads: int = 8
    timbre_d_t: int = 256              # D_t

    # GRL
    grl_lambda_max: float = 0.1
    n_instruments: int = 11            # NSynth instrument families; overridden at runtime from dataset

    # Decoder MLP hidden dim
    decoder_mlp_units: int = 512

    # Decoder transformer
    decoder_d_model: int = 512
    decoder_n_layers: int = 4
    decoder_n_heads: int = 8

    # Synthesizer
    n_harmonics: int = 100
    n_noise_magnitudes: int = 65

    # Loss
    reconstruction_loss_weight: float = 1.0
    grl_loss_weight: float = 1.0

    # Multi-scale spectrogram FFT sizes
    fft_sizes: tuple = (2048, 1024, 512, 256, 128, 64)
    overlap: float = 0.75


@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    n_epochs: int = 100
    warmup_steps: int = 1000
    grl_ramp_steps: int = 10000       # steps to ramp lambda from 0 to lambda_max
    grad_clip: float = 1.0
    log_interval: int = 100
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    nsynth_data_dir: str = "data/nsynth"
    instrument_families: list = dataclasses.field(
        default_factory=lambda: ["bass", "keyboard", "guitar"]
    )
    max_per_family: int = 1500      # max examples per instrument family (None = no limit)
    dataset_seed: int = 42          # random seed for family subsampling
