from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotate_fn = rotate_neox if is_neox_style else rotate_gptj
    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbeddingNeox(nn.Module):
    """Reference implementation of the GPT-NeoX style rotary embedding."""

    def __init__(
        self,
        dim: int,
        is_neox_style: bool,
        max_position_embeddings: int = 8192,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.rotary_dim = dim
        self.is_neox_style = is_neox_style
        self.max_position_embeddings = max_position_embeddings

        # Create cos and sin embeddings.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq.float())
        if is_neox_style:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.repeat_interleave(freqs, 2, -1)
        cos = emb.cos().to(dtype=inv_freq.dtype)
        sin = emb.sin().to(dtype=inv_freq.dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,  # [num_tokens]
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]

        query_rot = query_rot.transpose(0, 1)
        key_rot = key_rot.transpose(0, 1)
        cos = F.embedding(positions, self.cos_cached)
        sin = F.embedding(positions, self.sin_cached)
        query_rot, key_rot = apply_rope(query_rot, key_rot, cos, sin, self.is_neox_style)
        query_rot = query_rot.transpose(0, 1).contiguous()
        key_rot = key_rot.transpose(0, 1).contiguous()

        query = torch.cat((query_rot, query_pass), dim=-1)
        key = torch.cat((key_rot, key_pass), dim=-1)

        # Output query/key shape: [num_tokens, num_tokens, head_size]
        return query, key
