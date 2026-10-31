import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    """Identity forwards, gradient scaled by -lambda backwards."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x) # using view_as makes sure autograd doesn't skip the node


    @staticmethod
    def backward(ctx, grad_output):
        # Return one gradient for each input of forward.
        # lambda doesn't need a gradient so it's None.
        return grad_output * -ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)



class FamilyDiscriminator(nn.Module):
    """Predicts protein family from the pooled graph embedding.

    The gradient reversal lives inside forward, so the training loop only
    has to pass the current lambda.
    """

    def __init__(self, in_dim: int, n_families: int, hidden: int = 128,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_families)
        )


    def forward(self, z: torch.Tensor, lambd: float) -> torch.Tensor:
        z_reversed = grad_reverse(z, lambd)
        logits = self.mlp(z_reversed)
        return logits
