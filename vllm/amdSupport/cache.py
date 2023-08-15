import random

import torch


def reshape_and_cache(
    num_tokens,
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
) -> None:

    for i in range(num_tokens):
        reshaped_key = key.reshape(num_tokens, num_heads, head_size // x, x)
        block_idx = torch.div(slot_mapping[i],
                              block_size,
                              rounding_mode='floor')
        block_offset = slot_mapping[i] % block_size
        key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        value_cache[block_idx, :, :, block_offset] = value[i]

    return cloned_key_cache, cloned_value_cache

