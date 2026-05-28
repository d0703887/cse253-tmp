import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_,) = ctx.saved_tensors
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = torch.tensor(self.lambda_, dtype=x.dtype, device=x.device)
        return _GradientReversalFunction.apply(x, lam)


class InstrumentClassifier(nn.Module):
    """Temporal pooling → LayerNorm → 2-layer MLP → instrument logits."""

    def __init__(self, d_z: int, n_instruments: int, hidden_dim: int = 256):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.norm = nn.LayerNorm(d_z)
        self.mlp = nn.Sequential(
            nn.Linear(d_z, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_instruments),
        )

    def forward(self, z: torch.Tensor, lambda_: float | None = None) -> torch.Tensor:
        """
        Args:
            z: [Batch, T, D_z]
            lambda_: override GRL lambda for this forward pass
        Returns:
            logits: [Batch, N_instruments]
        """
        if lambda_ is not None:
            self.grl.set_lambda(lambda_)
        x = self.grl(z)
        x = x.mean(dim=1)       # temporal mean pooling → [Batch, D_z]
        x = self.norm(x)        # prevent logit explosion during adversarial training
        return self.mlp(x)