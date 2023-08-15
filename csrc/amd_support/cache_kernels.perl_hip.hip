#include "hip/hip_runtime.h"
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <stdio.h>
#include "hip/hip_fp16.h"

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

  printf("xr Size of pointer: %zu",sizeof(src.data_ptr()));
  char *src_ptr = (char *)src.data_ptr();
  char *dst_ptr = (char *)dst.data_ptr();


  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
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