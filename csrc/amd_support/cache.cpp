#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
    torch::Tensor& src,
    torch::Tensor& dst,
    const std::map<int64_t, int64_t>& block_mapping);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "swap_blocks",
        &swap_blocks,
        "Swap in (out) the cache blocks from src to dst");
}