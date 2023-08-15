import torch
import torch.nn as nn


class RMSNorma(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        weight = torch.empty(hidden_size)
        weight.uniform_(-1e-3, 1e-3)
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1,
                                                               keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        if self.weight.dtype in [torch.half, torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states

