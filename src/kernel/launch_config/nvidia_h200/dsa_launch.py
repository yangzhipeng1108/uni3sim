"""H200 source facts for individual DSA indexer sub-kernels.

vLLM0.30 indexer_topk.py selects cooperative_topk on SM90 for at most64
rows, aligned FP32 row stride and k512/1024/2048. cooperative_topk.cu fixes
the cluster launch; cooperative_topk.cuh and topk_histogram_4096.cuh define
its dynamic shared allocation. Registers/static shared require compilation.
This component is not the whole indexer: DeepGEMM logits and auxiliary
pack/quantization calls must remain in the caller's complete source sequence.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def _require_topk_source(dtype, attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 DSA TopK source requires H200 HWSpec")
    require_dtype(dtype, ("fp32",))
    if attributes["model_execution_version"] != "vllm_0_30_0":
        raise NotImplementedError("DSA TopK source binds only vLLM0.30")
    inner_stride = attributes["source_logits_inner_stride"]
    if type(inner_stride) is not int or inner_stride != 1:
        raise NotImplementedError("DSA TopK requires explicit unit inner logits stride")


def _cooperative_smem_size(stages, passes):
    """sizeof(SmemFused<stages,passes>), fixed-width CUDA POD field layout.

    Fields are barrier, counter_gt, counter_eq, MatchBin, warp_sum,
    histogram/Tie union, and score_buffer. Explicit alignas determines layout;
    no pointer, host-long or device instruction ABI enters this structure.
    The same sizes were independently checked with a small CPU C++ program.
    """
    offset = passes * stages * 8
    fields = ((4, 128), (4, 128), (16, 128), (32 * 4, 4),
              (max(1024 * 4, 2048 * 8), 8), (stages * 16384 * 4, 128))
    for size, alignment in fields:
        offset = ceil_div(offset, alignment) * alignment + size
    return ceil_div(offset, 128) * 128


def select_dsa_cooperative_topk_launch(shape, dtype, attributes, hardware):
    """Return one physical TopK launch with no aggregate execution_variant.

    The caller must map this into the full indexer program; it cannot attach
    this launch to a whole SPARSE_ATTN_INDEXER KernelSpec and omit logits or
    helper calls. An unknown/non-cooperative route raises rather than using
    another device package or inventing occupancy for persistent_topk.
    """
    _require_topk_source(dtype, attributes, hardware)
    if attributes["indexer_topk_backend"] not in ("auto", "cooperative"):
        raise NotImplementedError("requested DSA TopK backend is not the cooperative source")
    rows, stride, topk = require_shape(shape, ("rows", "logits_stride", "topk"))
    if rows > 64 or stride % 4 or topk not in (512, 1024, 2048):
        raise NotImplementedError("DSA TopK route requires persistent/per-row or another explicit backend")
    if stride >= 2**32:
        raise NotImplementedError("DSA TopK row stride must fit its uint32 host parameter")
    # SM90 never selects CS16: cooperative_topk.cu gates that branch on major10.
    cluster = 8 if rows <= 8 else 4 if rows <= 33 else 2
    output_shared = 2048 * 4 + 128
    if cluster == 8:
        shared = _cooperative_smem_size(2, 1) + output_shared
    else:
        shared = max(_cooperative_smem_size(2, 2), _cooperative_smem_size(3, 1)) + output_shared
    return LaunchConfig(f"vllm_0_30_sm90_dsa_cooperative_topk_cs{cluster}_k{topk}",
        (rows, cluster, 1), (1024, 1, 1), cluster=(1, cluster, 1),
        parameters={"dynamic_shared_bytes": float(shared)})


def select_dsa_filtered_topk_launch(shape, dtype, attributes, hardware):
    """H200 persistent entry's one-CTA-per-row FilteredTopK route.

    topk.cu selects this route for rows>32 on a device with at least128KiB
    opt-in shared memory, which H200 supplies. persistent_topk.cuh fixes
    block1024 and128KiB dynamic shared; vector width is gcd(row_stride,4).
    The smaller persistent route queries compiled-kernel occupancy and
    submits a workspace memset, and is deliberately not approximated here.
    """
    _require_topk_source(dtype, attributes, hardware)
    backend = attributes["indexer_topk_backend"]
    if backend not in ("auto", "persistent"):
        raise NotImplementedError("FilteredTopK requires the persistent source entry")
    rows, stride, topk = require_shape(shape, ("rows", "logits_stride", "topk"))
    if topk not in (512, 1024, 2048) or rows <= 32:
        raise NotImplementedError("FilteredTopK requires rows>32 and a compiled TopK specialization")
    if max(rows, stride) >= 2**32:
        raise NotImplementedError("FilteredTopK rows and stride must fit uint32 host parameters")
    if backend == "auto" and rows <= 64 and stride % 4 == 0:
        raise NotImplementedError("auto selects cooperative TopK for this source domain")
    vector = 4 if stride % 4 == 0 else 2 if stride % 2 == 0 else 1
    return LaunchConfig(f"vllm_0_30_sm90_dsa_filtered_topk_vec{vector}_k{topk}",
        (rows, 1, 1), (1024, 1, 1),
        parameters={"dynamic_shared_bytes": 131072.0})


def select_dsa_decode_topk_launch(shape, dtype, attributes, hardware):
    """Resolve the bound decode TopK component, never the full indexer."""
    _require_topk_source(dtype, attributes, hardware)
    rows, stride, topk = require_shape(shape, ("rows", "logits_stride", "topk"))
    backend = attributes["indexer_topk_backend"]
    if backend == "cooperative" or (backend == "auto" and rows <= 64
            and stride % 4 == 0 and topk in (512, 1024, 2048)):
        return select_dsa_cooperative_topk_launch(shape, dtype, attributes, hardware)
    if backend in ("auto", "persistent"):
        return select_dsa_filtered_topk_launch(shape, dtype, attributes, hardware)
    raise NotImplementedError("DSA TopK backend has no bound H200 source component")


def select_dsa_prefill_topk_launches(shape, dtype, attributes, hardware):
    """sampler.cu's complete per-chunk prefill TopK launch sequence.

    The first12288 rows use the insertion specialization, subsequent rows
    use a second radix specialization with source-fixed offset12288. This
    boundary is row count, not each row's context length. Return both
    physical launches when required, with no whole-indexer variant.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("DSA prefill TopK source requires H200 HWSpec")
    require_dtype(dtype, ("fp32",))
    if attributes["model_execution_version"] != "vllm_0_30_0":
        raise NotImplementedError("DSA prefill TopK source binds only vLLM0.30")
    rows, topk = require_shape(shape, ("rows", "topk"))
    if rows >= 2**31 or topk not in (512, 1024, 2048):
        raise NotImplementedError("DSA prefill TopK binds int32 rows and512/1024/2048-index source domains")
    launches = [LaunchConfig("vllm_0_30_sm90_dsa_prefill_topk_insertion",
        (min(rows, 12288), 1, 1), (512, 1, 1),
        parameters={"dynamic_shared_bytes": float(topk * 4)})]
    if rows > 12288:
        launches.append(LaunchConfig("vllm_0_30_sm90_dsa_prefill_topk_radix_offset12288",
            (rows - 12288, 1, 1), (512, 1, 1),
            parameters={"dynamic_shared_bytes": float(topk * 4)}))
    return tuple(launches)


def select_dsa_prefill_cache_gather_launch(shape, dtype, attributes, hardware):
    """cache_kernels.cu::cp_gather_indexer_k_quant_cache,128B K+4B scale.

    rows is the actual destination K row count for one source chunk, which
    includes cached context and differs from the query's TopK row count.
    Skip/alias decisions belong to the complete caller's chunk metadata.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("DSA prefill cache gather requires H200 HWSpec")
    require_dtype(dtype, ("uint8",))
    if attributes["model_execution_version"] != "vllm_0_30_0":
        raise NotImplementedError("DSA cache gather source binds only vLLM0.30")
    rows, head_dim = require_shape(shape, ("rows", "head_dim"))
    quant_block = attributes["quant_block_size"]
    if head_dim != 128 or rows >= 2**31 or type(quant_block) is not int or quant_block != 128:
        raise NotImplementedError("DSA cache gather requires int32 rows and128-value scale blocks")
    block_y = 1 if rows < 32 else 2 if rows < 64 else 4 if rows < 128 else 8 if rows < 256 else 16 if rows < 512 else 32
    return LaunchConfig(f"vllm_0_30_sm90_dsa_cache_gather_y{block_y}",
        (ceil_div(rows, block_y), 1, 1), (8, block_y, 1),
        parameters={"dynamic_shared_bytes": 0.0})


def dsa_cooperative_topk_workspace_bytes(shape):
    """Actual indexer_topk.py workspace allocation, including unused tail.

    The C++ minimum is rows * max(topk,1024) * sizeof(Tie), but the Python
    caller reserves the fixed RADIX_TOPK_WORKSPACE_SIZE1MiB arena. Preserve
    that allocation rather than pricing the minimum as the actual lifetime.
    """
    rows, stride, topk = require_shape(shape, ("rows", "logits_stride", "topk"))
    if rows > 64 or stride % 4 or stride >= 2**32 or topk not in (512, 1024, 2048):
        raise NotImplementedError("cooperative TopK workspace requires its complete source domain")
    return 1024 * 1024


def select_dsa_decode_indexer_recipe(shape, dtype, attributes, hardware):
    """Native contiguous logits->TopK body, with explicit metadata ownership.

    Input Q is already quantized, weights prepared, and KV cache updated.
    Decode packing, candidate-block masking and varlen fallbacks are outside
    this source domain. Returning the full stage tuple does not bind an
    aggregate KernelSpec launch or claim source-aware stage pricing.
    """
    from src.kernel.launch_config.nvidia_h200.dsa_logits_launch import (
        select_dsa_decode_logits_launch, select_dsa_decode_metadata_launch)
    if (attributes["source_indexer_input_stage"] != "quantized_q_weights_and_cache_ready"
            or attributes["source_decode_layout"] != "native_contiguous"
            or attributes["source_candidate_blocks"] != "none"):
        raise NotImplementedError("DSA decode recipe requires its explicit native input and no candidate-mask stages")
    batch, next_n, capacity, topk = require_shape(shape, ("batch", "next_n", "max_model_len", "topk"))
    logits = select_dsa_decode_logits_launch(shape, dtype, attributes, hardware)
    # attention.hpp owns allocation alignment and unit inner stride.
    topk_shape = {"rows": batch * next_n, "logits_stride": ceil_div(capacity, 256) * 256, "topk": topk}
    topk_attrs = {**attributes, "source_logits_inner_stride": 1}
    selected = select_dsa_decode_topk_launch(topk_shape, "fp32", topk_attrs, hardware)
    metadata = attributes["source_schedule_metadata"]
    if metadata == "build_current_context":
        setup = (("schedule_metadata", select_dsa_decode_metadata_launch(shape, "int32", attributes, hardware)),)
    elif metadata == "provided_current_context":
        setup = ()
    else:
        raise NotImplementedError("DSA decode requires explicit current-context metadata ownership")
    return (*setup, ("logits", logits), ("topk", selected))


def select_dsa_prefill_indexer_chunk_recipe(shape, dtype, attributes, hardware):
    """One actual nonempty prefill chunk, including optional cache gather."""
    from src.kernel.launch_config.nvidia_h200.dsa_logits_launch import select_dsa_prefill_logits_launch
    if (attributes["source_indexer_input_stage"] != "quantized_q_weights_and_cache_ready"
            or attributes["source_candidate_blocks"] != "none"):
        raise NotImplementedError("DSA prefill recipe requires its actual prepared input and no candidate-mask stages")
    rows, kv_rows, dim, topk = require_shape(shape, ("rows", "kv_rows", "head_dim", "topk"))
    skip = attributes["source_skip_kv_gather"]
    if type(skip) is not bool:
        raise TypeError("DSA prefill gather skip must be an explicit source fact")
    logits = select_dsa_prefill_logits_launch(shape, dtype, attributes, hardware)
    if skip:
        if attributes["source_gather_input_state"] != "materialized_k_quant_and_scale_ready":
            raise ValueError("skipped gather requires the actual materialized K and scale producer")
        setup = ()
    else:
        gather = select_dsa_prefill_cache_gather_launch({"rows": kv_rows, "head_dim": dim},
            "uint8", attributes, hardware)
        setup = (("cache_gather", gather),)
    selected = select_dsa_prefill_topk_launches({"rows": rows, "topk": topk}, "fp32", attributes, hardware)
    return (*setup, ("logits", logits), *(("topk", launch) for launch in selected))


def select_dsa_prepared_indexer_recipe(shape, dtype, attributes, hardware):
    """Source indexer body after Q/cache preparation and upstream clearing.

    The fused V3.2 producer clears the same live top-k region. This entry
    requires that explicit ownership, preserving the caller's skip-clear
    branch. Metadata supplies every prefill chunk; the selector never
    creates chunks, chooses cache pages, or assumes a materialized gather.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("DSA prepared indexer requires H200 HWSpec")
    require_dtype(dtype, ("fp8",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_sparse_attn_indexer"
            or attributes["source_parallel_mode"] != "pcp1_dcp1"
            or attributes["source_dense_mha_metadata_layer"] != "none"
            or attributes["skip_topk_buffer_clear"] is not True
            or attributes["source_topk_clear_state"] != "upstream_fused_norm_rope_ready"):
        raise NotImplementedError("DSA prepared body requires its exact source and upstream top-k clear")
    rows, heads, dim, topk = require_shape(shape, ("rows", "num_heads", "head_dim", "topk"))
    if heads not in (32, 64) or dim != 128 or topk not in (512, 1024, 2048):
        raise NotImplementedError("DSA prepared body requires its FP8 head128 and native top-k domain")
    decode = shape["num_decode_tokens"]
    prefill = shape["num_prefill_tokens"]
    if (type(decode) is not int or type(prefill) is not int
            or min(decode, prefill) < 0 or decode + prefill != rows):
        raise ValueError("DSA prepared token counts must partition the actual source rows")
    chunks = attributes["source_prefill_chunks"]
    if not isinstance(chunks, (tuple, list)):
        raise TypeError("DSA source prefill chunks must be explicit metadata records")
    stages, cursor = [], decode
    for index, chunk in enumerate(chunks):
        start, end, kv_rows = (chunk[field] for field in ("token_start", "token_end", "kv_rows"))
        if (any(type(value) is not int for value in (start, end, kv_rows))
                or start != cursor or not start < end <= rows or kv_rows <= 0):
            raise ValueError("DSA chunks must partition the actual nonempty prefill suffix")
        cursor = end
        chunk_shape = {"rows": end - start, "kv_rows": kv_rows,
            "num_heads": heads, "head_dim": dim, "topk": topk}
        chunk_attrs = {**attributes, "source_schedule_metadata": "none", "source_kv_scale_dtype": "fp32",
            "source_skip_kv_gather": chunk["skip_kv_gather"],
            "source_gather_input_state": chunk["gather_input_state"], "quant_block_size": 128}
        recipe = select_dsa_prefill_indexer_chunk_recipe(chunk_shape, dtype, chunk_attrs, hardware)
        stages.extend((f"prefill_{index}_{step}_{role}", launch)
            for step, (role, launch) in enumerate(recipe))
    if cursor != rows:
        raise ValueError("DSA chunks do not cover the actual prefill suffix")
    if decode:
        batch, next_n = require_shape(shape, ("batch", "next_n"))
        if batch * next_n != decode:
            raise ValueError("DSA native decode shape disagrees with actual token count")
        stages.extend(("decode_" + role, launch) for role, launch in
            select_dsa_decode_indexer_recipe(shape, dtype, attributes, hardware))
    return tuple(stages)


def require_dsa_cooperative_topk_cluster(recipe):
    """Return the cooperative cluster fact or reject another selected backend."""
    launches = dict(recipe)
    if "decode_topk" in launches:
        launch = launches["decode_topk"]
    elif "topk" in launches:
        launch = launches["topk"]
    else:
        raise ValueError("DSA cooperative source work requires a decode TopK launch")
    if (launch.cluster is None or not launch.algorithm.startswith(
            "vllm_0_30_sm90_dsa_cooperative_topk_cs")):
        raise NotImplementedError(
            "DSA cooperative source work supports cooperative TopK only; "
            f"selected {launch.algorithm} requires a separate service model")
    cluster = launch.cluster[1]
    if type(cluster) is not int or cluster not in (2, 4, 8):
        raise ValueError("DSA cooperative TopK cluster must be one of 2, 4 or 8")
    return cluster


def dsa_decode_topk_source_route(recipe):
    """Return the launch-bound decode TopK route facts for the service model.

    The vLLM0.30 auto chain selects the cooperative kernel only for its
    aligned <=64-row domain and otherwise the persistent entry's
    FilteredTopK (one CTA per row, block 1024, 128 KiB dynamic shared,
    vector width gcd(row_stride, 4)). The service model prices the
    selected route; it never re-selects a backend. A route without a
    bound service model is rejected, not approximated.
    """
    launches = dict(recipe)
    if "decode_topk" in launches:
        launch = launches["decode_topk"]
    elif "topk" in launches:
        launch = launches["topk"]
    else:
        raise ValueError("DSA source work requires a decode TopK launch")
    algorithm = launch.algorithm
    if algorithm.startswith("vllm_0_30_sm90_dsa_cooperative_topk_cs"):
        cluster = launch.cluster[1] if launch.cluster is not None else None
        if type(cluster) is not int or cluster not in (2, 4, 8):
            raise ValueError("cooperative TopK cluster must be one of 2, 4 or 8")
        return {"route": "cooperative", "cluster": cluster, "vector_width": None,
            "block_threads": 1024, "dynamic_shared_bytes": None, "algorithm": algorithm}
    if algorithm.startswith("vllm_0_30_sm90_dsa_filtered_topk_vec"):
        if launch.cluster is not None:
            raise ValueError("FilteredTopK is a one-CTA-per-row launch without clusters")
        vector = int(algorithm.split("_vec")[1].split("_k")[0])
        if vector not in (1, 2, 4):
            raise ValueError("FilteredTopK vector width must be 1, 2 or 4")
        shared = launch.parameters["dynamic_shared_bytes"]
        if shared != 131072.0:
            raise ValueError("FilteredTopK dynamic shared memory must be its 128 KiB source constant")
        return {"route": "filtered_persistent", "cluster": None, "vector_width": vector,
            "block_threads": 1024, "dynamic_shared_bytes": int(shared), "algorithm": algorithm}
    raise NotImplementedError(
        f"decode TopK launch {algorithm} has no bound service model")
