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

    block_size = value_cache.shape[-1]
    for i in range(num_tokens):
        reshaped_key = key.reshape(num_tokens, key_cache.shape[1], key_cache.shape[2], key_cache.shape[-1])
        block_idx = torch.div(slot_mapping[i],
                              block_size,
                              rounding_mode='floor')
        block_offset = slot_mapping[i] % block_size
        key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        value_cache[block_idx, :, :, block_offset] = value[i]

    return key_cache, value_cache


def copy_blocks(
    key_caches,
    value_caches,
    block_mapping,
) -> None:
    # Create the KV cache.
    cloned_key_caches = []
    for key_cache in key_caches:
        cloned_key_caches.append(key_cache.clone())

    cloned_value_caches = []
    for value_cache in value_caches:
        cloned_value_caches.append(value_cache.clone())


    # Reference implementation.
    for src, dsts in block_mapping.items():
        for dst in dsts:
            for key_cache, cloned_key_cache in zip(key_caches,
                                                   cloned_key_caches):
                cloned_key_cache[dst] = cloned_key_cache[src]
            for value_cache, cloned_value_cache in zip(value_caches,
                                                       cloned_value_caches):
                cloned_value_cache[dst] = cloned_value_cache[src]

    return cloned_key_cache, cloned_value_cache