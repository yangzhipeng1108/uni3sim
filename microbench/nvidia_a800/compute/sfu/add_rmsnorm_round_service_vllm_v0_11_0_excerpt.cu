/*
 * Verbatim semantic excerpt from vLLM 0.11.0.
 * Repository: https://github.com/vllm-project/vllm.git
 * Commit: b8b302cde434df8c9289a2b465406b47ebab1c2d
 * Paths and full-file SHA256:
 * csrc/layernorm_kernels.cu fb3b6102d6c0efdf66ed7a2030d0bebf6a25aa46e07fe7274ec545f39e200409
 * csrc/type_convert.cuh 1c7cc528f08f14eb4fcb7cee50255d4437db8301335412af3bb86b3a248ad5e2
 *
 * This file is a static source pin and is not compiled.
 */

template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[strided_id] = temp;
  }
}

/* Launch excerpt from fused_add_rms_norm. */
const int max_block_size = (num_tokens < 256) ? 1024 : 256;
dim3 grid(num_tokens);
dim3 block(std::min(hidden_size, max_block_size));
constexpr int vector_width = 8;
bool offsets_are_multiple_of_vector_width =
    hidden_size % vector_width == 0 && input_stride % vector_width == 0;
if (ptrs_are_aligned && offsets_are_multiple_of_vector_width) {
  LAUNCH_FUSED_ADD_RMS_NORM(8);
} else {
  LAUNCH_FUSED_ADD_RMS_NORM(0);
}

/* Exact BF16 vec8 operation bodies from csrc/type_convert.cuh. */
__device__ _f16Vec& operator+=(const _f16Vec<scalar_t, width>& other) {
  if constexpr (width % 2 == 0) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      T2 temp{data[i], data[i + 1]};
      temp += T2{other.data[i], other.data[i + 1]};
      data[i] = temp.x;
      data[i + 1] = temp.y;
    }
  }
  return *this;
}

__device__ _f16Vec& operator*=(const _f16Vec<scalar_t, width>& other) {
  if constexpr (width % 2 == 0) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      T2 temp{data[i], data[i + 1]};
      temp *= T2{other.data[i], other.data[i + 1]};
      data[i] = temp.x;
      data[i + 1] = temp.y;
    }
  }
  return *this;
}

__device__ _f16Vec& operator*=(const float scale) {
  if constexpr (width % 2 == 0) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
      temp_f.x *= scale;
      temp_f.y *= scale;
      T2 temp = Converter::convert(temp_f);
      data[i] = temp.x;
      data[i + 1] = temp.y;
    }
  }
  return *this;
}

__device__ float sum_squares() const {
  float result = 0.0f;
  if constexpr (width % 2 == 0) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      float2 z = Converter::convert(T2{data[i], data[i + 1]});
      result += z.x * z.x + z.y * z.y;
    }
  }
  return result;
}
