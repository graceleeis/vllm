#include "hip/hip_runtime.h"
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include "hip/hip_fp16.h"
#include "hip/hip_bfloat16.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  hipMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(
      src_device.index() == dst_device.index(),
      "src and dst must be on the same GPU");
    memcpy_type = hipMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = hipMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = hipMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  char *src_ptr = (char *)src.data_ptr();
  char *dst_ptr = (char *)dst.data_ptr();

  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    int64_t dst_block_number = pair.second;
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    hipMemcpyAsync(
      dst_ptr + dst_offset,
      src_ptr + src_offset,
      block_size_in_bytes,
      memcpy_type,
      stream);
  }
}

namespace vllm {

template<typename scalar_t>
__global__ void reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,     // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,   // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
  const int* __restrict__ slot_mapping, // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  const int token_idx = blockIdx.x;
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int src_key_idx = token_idx * key_stride + i;
    const int src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                            + head_idx * (head_size / x) * block_size * x
                            + x_idx * block_size * x
                            + block_offset * x
                            + x_offset;
    const int tgt_value_idx = block_idx * num_heads * head_size * block_size
                              + head_idx * head_size * block_size
                              + head_offset * block_size
                              + block_offset;
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

} // namespace vllm

void reshape_and_cache(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping)  // [num_tokens]
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    key.scalar_type(),
    "reshape_and_cache_kernel",
    [&] {
      vllm::reshape_and_cache_kernel<scalar_t><<<grid, block, 0>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x);
    });
}
