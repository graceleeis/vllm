import random
from typing import List, Optional

import torch
# from xformers import ops as xops
# from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

# from vllm import attention_ops

MAX_SEQ_LEN = 4096
TEST_SEED = 0


def masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query * scale
    attn = torch.einsum('qhd,khd->hqk', query, key)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum('hqk,khd->qhd', attn, value)
    return out


def single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_size**0.5)
        out = masked_attention(q, keys, values, scale)
        out = out.view(num_heads, head_size)
        output[i].copy_(out, non_blocking=True)


def multi_query_kv_attention(
    cu_seq_lens: List[int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    head_size = query.shape[-1]
    scale = 1.0 / (head_size**0.5)

    num_seqs = len(cu_seq_lens) - 1
    ref_outputs = []
    for i in range(num_seqs):
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        seq_len = end_idx - start_idx

        # Create attention mask.
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype),
                               diagonal=1)
        attn_mask = attn_mask * torch.finfo(dtype).min
        attn_mask = attn_mask.to(dtype=dtype, device='cuda')

        ref_output = masked_attention(
            query[start_idx:end_idx],
            key[start_idx:end_idx],
            value[start_idx:end_idx],
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)
    ref_output = torch.cat(ref_outputs, dim=0)
    return ref_output


def multi_query_cached_kv_attention(
    cu_query_lens: List[int],
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    scale = 1.0 / (head_size**0.5)

    num_queries = len(cu_query_lens) - 1
    ref_outputs = []
    for i in range(num_queries):
        start_idx = cu_query_lens[i]
        end_idx = cu_query_lens[i + 1]
        query_len = end_idx - start_idx
        context_len = int(context_lens[i])
        block_table = block_tables[i]

        # Create attention mask
        attn_mask = torch.triu(torch.ones(query_len, context_len),
                               diagonal=context_len - query_len + 1) * -1e5
        attn_mask = attn_mask.to(dtype=dtype, device='cuda')

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        ref_output = masked_attention(
            query[start_idx:end_idx],
            keys,
            values,
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)
    ref_output = torch.cat(ref_outputs, dim=0)
    return ref_output

