"""vLLM0.30's complete fused vocab-parallel CUDA lookup on H200.

VocabParallelEmbedding caches use_fused_embedding for CUDA/unquantized/TP>1.
_custom_ops.vocab_parallel_embedding allocates a fresh partial output and
csrc/libtorch_stable/vocab_parallel_embedding_kernels.cu masks,shifts and
copies orzeroes each row in one kernel. The following group all-reduce is
a separate caller operation,not a fabricated native collective launch.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def select_vllm030_vocab_parallel_embedding_launch(shape, dtype, attributes, hardware):
    """Select the entire native local lookup,not the outer TP collective.

    Pointer alignment is an actual address fact capped at16 bytes;source
    dispatch ORs the two addresses and row_bytes. Contiguity/fresh allocation
    alone doesnot establish that fact. IDs retain their input INT32/INT64
    dtype;there is no long cast,mask tensor,standalone fill orindex_select.
    Out-of-shard rows are fully zeroed in this same kernel. Whether a row
    reads weights depends on its actual ID andis not inferred from shape.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("fused vocab embedding requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16", "fp32"))
    vocab, hidden = require_shape(shape, ("source_rows", "hidden"))
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("fused vocab embedding requires non-negative physical token rows")
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_vocab_parallel_embedding_cuda"
            or type(attributes["source_tp_size"]) is not int or attributes["source_tp_size"] <= 1
            or attributes["source_unquantized_embedding_method"] is not True
            or attributes["source_use_fused_embedding"] is not True
            or attributes["source_input_layout"] != "contiguous_ids_and_row_major_weight"
            or attributes["source_output_layout"] != "fresh_contiguous_rows"):
        raise NotImplementedError("fused embedding requires exact0.30 CUDA unquantized TP>1 caller facts")
    indices = attributes["source_shard_indices"]
    names = ("org_vocab_start_index", "org_vocab_end_index", "num_org_vocab_padding",
             "added_vocab_start_index", "added_vocab_end_index", "num_added_vocab_padding")
    values = tuple(indices[name] for name in names)
    if any(type(value) is not int or not 0 <= value < 2**63 for value in values):
        raise ValueError("fused embedding shard ranges/padding must be actual non-negative signed integers")
    org_start, org_end, org_pad, added_start, added_end, added_pad = values
    if (org_end < org_start or added_end < added_start
            or org_end - org_start + org_pad + added_end - added_start + added_pad != vocab
            or (org_start < org_end and added_start < added_end
                and max(org_start, added_start) < min(org_end, added_end))):
        raise ValueError("fused embedding actual shard ranges/padding must match the allocated weight rows")
    index_dtype = attributes["source_input_ids_dtype"]
    if index_dtype not in ("int32", "int64"):
        raise NotImplementedError("fused embedding consumes actual INT32/INT64 IDs without conversion")
    width = 4 if dtype == "fp32" else 2
    row_bytes = hidden * width
    if max(vocab, rows) * row_bytes >= 2**63:
        raise NotImplementedError("fused embedding storage exceeds source signed address extents")
    if rows == 0:
        return None
    alignments = (attributes["source_weight_address_alignment"], attributes["source_output_address_alignment"])
    if any(type(value) is not int or value not in (1, 2, 4, 8, 16) for value in alignments):
        raise ValueError("fused embedding needs actual address alignments capped at16 bytes")
    vector = min(alignments)
    while row_bytes % vector:
        vector //= 2
    vecs = row_bytes // vector
    return LaunchConfig(f"vllm_0_30_sm90_vocab_parallel_embedding_{index_dtype}_vec{vector}",
        (min(ceil_div(vecs, 256), 256), min(rows, 4 * int(hardware.num_compute_units)), 1),
        (256, 1, 1), parameters={"dynamic_shared_bytes": 0.0})
