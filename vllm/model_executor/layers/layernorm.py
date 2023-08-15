"""Custom normalization layers."""
import torch
import torch.nn as nn

# from vllm import layernorm_ops
from vllm.amdSupport import RMSNorma


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        # layernorm_ops.rms_norm(
        #     out,
        #     x,
        #     self.weight.data,
        #     self.variance_epsilon,
        # )
        # ref = RMSNorma(self.hidden_size).cuda()
        ref = RMSNorma(self.weight.data, self.variance_epsilon).to(x.dtype).cuda()
        out = ref(x)
        return out
