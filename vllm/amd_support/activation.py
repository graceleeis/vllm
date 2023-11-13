import torch
import torch.nn.functional as F


def ref_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(chunks=2, dim=1)
    return F.silu(x1) * x2
