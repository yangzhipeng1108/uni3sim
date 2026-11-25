/*
 * MetaX C500 PyTorch custom-op memory-task benchmark.
 *
 * The operator is deliberately not a production operator. One lane per
 * hardware wave performs one ordinary global load, one dependent integer add,
 * and one ordinary global store. The C++ dispatcher and device launch use the
 * same torch.ops boundary as vLLM custom operators, while the body remains an
 * operator-independent runtime mechanism.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>


namespace {

constexpr int kWaveSize = 64;


__global__ void custom_op_empty_task() {}


__global__ void custom_op_memory_dependency_task(
        const std::uint32_t* input, std::uint32_t* output) {
    const int lane = threadIdx.x & (kWaveSize - 1);
    if (lane != 0) {
        return;
    }
    const int waves_per_block = (blockDim.x + kWaveSize - 1) / kWaveSize;
    const int wave = threadIdx.x / kWaveSize;
    const int flattened_block = blockIdx.y * gridDim.x + blockIdx.x;
    const int index = flattened_block * waves_per_block + wave;
    output[index] = input[index] + 1U;
}


void launch_custom_op_empty_task(
        const torch::Tensor& anchor,
        std::int64_t grid_x, std::int64_t grid_y, std::int64_t block_size) {
    TORCH_CHECK(anchor.is_cuda(),
                "anchor must be a CUDA-compatible tensor");
    TORCH_CHECK(grid_x > 0 && grid_y > 0,
                "grid dimensions must be positive");
    TORCH_CHECK(block_size > 0 && block_size <= 1024,
                "block size must be in [1, 1024]");
    TORCH_CHECK(block_size % kWaveSize == 0,
                "block size must contain whole wave64 groups");
    const c10::cuda::CUDAGuard device_guard(anchor.device());
    const auto stream = at::cuda::getCurrentCUDAStream(anchor.get_device());
    custom_op_empty_task<<<
        dim3(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), 1),
        dim3(static_cast<unsigned>(block_size), 1, 1), 0, stream>>>();
}


void launch_custom_op_memory_dependency_task(
        const torch::Tensor& input, torch::Tensor output,
        std::int64_t grid_x, std::int64_t grid_y, std::int64_t block_size) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA-compatible tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA-compatible tensor");
    TORCH_CHECK(input.scalar_type() == torch::kUInt32,
                "input must have dtype uint32");
    TORCH_CHECK(output.scalar_type() == torch::kUInt32,
                "output must have dtype uint32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(grid_x > 0 && grid_y > 0,
                "grid dimensions must be positive");
    TORCH_CHECK(block_size > 0 && block_size <= 1024,
                "block size must be in [1, 1024]");
    TORCH_CHECK(block_size % kWaveSize == 0,
                "block size must contain whole wave64 groups");
    const std::int64_t required_elements = (
        grid_x * grid_y * ((block_size + kWaveSize - 1) / kWaveSize));
    TORCH_CHECK(input.numel() >= required_elements,
                "input allocation is smaller than the launch demand");
    TORCH_CHECK(output.numel() >= required_elements,
                "output allocation is smaller than the launch demand");
    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    custom_op_memory_dependency_task<<<
        dim3(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), 1),
        dim3(static_cast<unsigned>(block_size), 1, 1), 0, stream>>>(
            input.data_ptr<std::uint32_t>(),
            output.data_ptr<std::uint32_t>());
}

}  // namespace


TORCH_LIBRARY(midlayer_microbench, library) {
    library.def(
        "custom_op_empty_task(Tensor anchor, int grid_x, int grid_y, "
        "int block_size) -> ()");
    library.def(
        "custom_op_memory_dependency_task(Tensor input, Tensor(a!) output, "
        "int grid_x, int grid_y, int block_size) -> ()");
}


TORCH_LIBRARY_IMPL(midlayer_microbench, CUDA, library) {
    library.impl(
        "custom_op_empty_task",
        &launch_custom_op_empty_task);
    library.impl(
        "custom_op_memory_dependency_task",
        &launch_custom_op_memory_dependency_task);
}
