"""K3 ordinary prefill state assignment from vLLM0.30 and Torch2.13.

kda.py writes recurrent_state[state_indices] = final_state.to(pool.dtype).
For its FP32 pool/final state, to() aliases its input. TensorAdvancedIndexing
make_info widens INT32 indices to Long, then IndexKernel.cu index_put uses
index_elementwise_kernel<128,4>. Its gather-only vectorized fast path cannot
serve this scatter. A gapped pool block stride is an INT64 indexed offset,
outside TensorIterator's restrided, zero-row-stride output extent.

Official v2.13.0 Copy.cu/CUDALoops.cuh independently bind the separate
INT32-to-INT64 dynamic-cast kernel to128 threads/four values. No2.8 cubin
register counts or any SM80 fact are transferred. Deterministic/sort,
accumulation, other state dtypes and split iterators remain unbound.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_positive_integer, require_shape


def select_kda_final_state_scatter_recipe(shape, dtype, attributes, hardware):
    """Return widening, when necessary, and the complete row assignment.

    rows counts actual final states/requests, not query tokens or chunks.
    state_indices and the pool's real row stride are mandatory caller facts.
    The KDA source's inner [head,value,key] row and final-state tensor are
    contiguous, while arena allocation may put gaps between pool rows.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 state scatter requires H200 HWSpec")
    require_dtype(dtype, ("fp32",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] not in (
                "vllm_0_30_kimi_k3_triton_fresh_prefill",
                "vllm_0_30_kimi_k3_triton_context_prefill",
                "vllm_0_30_kimi_k3_triton_mixed",
                "vllm_0_30_kimi_k3_flashkda_fresh_prefill",
                "vllm_0_30_kimi_k3_flashkda_context_prefill",
                "vllm_0_30_kimi_k3_flashkda_mixed")
            or attributes["stage"] != "final_state_scatter"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_state_inner_layout"] != "heads_value_key_contiguous"
            or attributes["source_final_state_layout"] != "contiguous_distinct_fp32"
            or attributes["source_deterministic_algorithms"] is not False):
        raise NotImplementedError("K3 state scatter requires its exact0.30/Torch2.13 unsorted FP32 source")
    rows, heads, dim, pool_slots = require_shape(shape, ("rows", "heads", "head_dim", "state_pool_slots"))
    if heads not in (6, 12, 24, 32, 48, 96) or dim != 128:
        raise NotImplementedError("K3 state scatter binds128-wide source heads")
    state_row = heads * dim * dim
    row_stride = require_positive_integer(attributes, "source_state_pool_row_stride_elements")
    if row_stride < state_row:
        raise ValueError("K3 state pool rows must not overlap")
    if 4 * ((pool_slots - 1) * row_stride + state_row - 1) >= 2**63:
        raise NotImplementedError("K3 state pool indexed offsets exceed signed int64")
    stride_bytes = 4 * row_stride
    if int(float(stride_bytes)) != stride_bytes:
        raise NotImplementedError("K3 state stride cannot be represented exactly in launch parameters")
    indices = attributes["state_indices"]
    if (not isinstance(indices, (tuple, list)) or len(indices) != rows
            or any(type(index) is not int or not 0 < index < pool_slots for index in indices)
            or len(set(indices)) != rows):
        raise ValueError("K3 prefill requires one distinct positive live state ID per final state")
    index_dtype = attributes["source_state_indices_dtype"]
    if index_dtype not in ("int32", "int64"):
        raise NotImplementedError("K3 state assignment requires actual INT32/INT64 indices")
    if index_dtype == "int32" and any(index >= 2**31 for index in indices):
        raise ValueError("K3 INT32 state ID does not fit its declared input dtype")
    # IndexKernel checks the iterator's source/output/index byte offsets.
    # The pool's selected-row offset is instead added separately in int64;
    # large or gapped pools alone must not trigger a fake split iterator.
    elements = rows * state_row
    if elements >= 2**31 or 4 * (elements - 1) >= 2**31 - 1 or 8 * (rows - 1) >= 2**31 - 1:
        raise NotImplementedError("K3 state scatter split iterator requires every physical submission")
    scatter = LaunchConfig("pytorch_2_13_sm90_kda_fp32_index_put",
        (ceil_div(elements, 512), 1, 1), (128, 1, 1), parameters={
            "dynamic_shared_bytes": 0.0,
            "source_indexed_row_stride_bytes": float(stride_bytes),
            "source_values_per_thread": 4.0})
    if index_dtype == "int64":
        return (("final_state_index_put", scatter),)
    widen = LaunchConfig("pytorch_2_13_sm90_int32_to_int64",
        (ceil_div(rows, 512), 1, 1), (128, 1, 1),
        parameters={"dynamic_shared_bytes": 0.0, "source_values_per_thread": 4.0})
    return (("state_ids_long", widen), ("final_state_index_put", scatter))
