"""Conditional vLLM0.30 SM90 Marlin auto dispatch and modular expert body.

ops.cu queries every shape/cache-valid candidate, not just the selected
kernel. ``candidate_resources`` is a transient caller argument keyed by
the complete template identity; None means an actual MarlinDefault result,
otherwise (numRegs, sharedSizeBytes) comes from that candidate's compiler/
device query. This module neither supplies nor persists those facts. A
recompiled candidate is not automatically the deployed binary. There is
no public KernelSpec admission, native-provider claim or timing price here.
"""

from collections.abc import Mapping
from dataclasses import replace

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def _source(dtype, attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Marlin source requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_marlin_a16_nozp_auto"
            or attributes["source_thread_config"] != "auto"
            or attributes["source_has_zero_points"] is not False):
        raise NotImplementedError("Marlin requires exact0.30 A16/no-ZP automatic source")
    weight, scale, group = (attributes["source_weight_type"], attributes["source_scale_dtype"],
                            attributes["source_group_size"])
    if type(group) is not int:
        raise TypeError("Marlin group size must be an actual integer")
    if weight == "float4_e2m1f":
        if dtype != "bf16" or scale != "float8_e8m0fnu" or group != 32:
            raise NotImplementedError("SM90 MXFP4 binds BF16/E8M0/group32 source")
    elif weight == "uint4b8":
        if scale != dtype or group not in (-1, 32, 64, 128):
            raise NotImplementedError("SM90 GPTQ INT4 binds matching A/C/scales and source groups")
    else:
        raise NotImplementedError("Marlin weight type is outside this no-ZP four-bit source")
    if (attributes["source_num_compute_units"] != int(hardware.num_compute_units)
            or type(attributes["source_num_compute_units"]) is not int
            or attributes["source_max_shared_memory_bytes"] != hardware.max_smem_per_block_bytes
            or type(attributes["source_max_shared_memory_bytes"]) is not int):
        raise ValueError("Marlin actual device SM/shared queries must agree with H200 HWSpec")


def marlin_candidate_identity(dtype, weight, scale, block_m, group, thread_k, thread_n, threads):
    """All get_marlin_kernel arguments, including source flags and stages."""
    return (dtype, weight, dtype, scale, ceil_div(block_m, 16), thread_n // 16,
            thread_k // 16, block_m == 8, False, -1 if group == -1 else group // 16,
            threads, False, 4)


def _cache_bytes(block_m, group, thread_k, thread_n):
    # get_kernel_cache_size intentionally uses16 rows even for the M8 MMA.
    rows = ceil_div(block_m, 16) * 16
    a = 4 * rows * thread_k * 2
    b = 4 * (thread_k * thread_n // 8) * 4
    reduction = rows * (thread_n + 8) * 2
    tmp = max(b, reduction, min(b, reduction) + thread_n * 2)
    scales = (1 if group == -1 else ceil_div(thread_k, group)) * thread_n * 2 * 4
    return tmp + a + scales + rows * 16


def select_marlin_auto_launch(shape, dtype, attributes, hardware, candidate_resources):
    """Replay determine_exec_config, requiring all eligible candidate facts.

    Identity order follows get_marlin_kernel, not the generated Marlin
    template's printed order. No selected-kernel-only receipt can establish
    automatic selection. Synthetic facts are useful for testing this source
    rule, but do not constitute a production resource binding.
    """
    _source(dtype, attributes, hardware)
    m, n, k, topk, block_m = require_shape(shape, ("m", "n", "k", "topk", "block_m"))
    if block_m not in (8, 16, 32, 48, 64):
        raise NotImplementedError("Marlin source M block must be8/16/32/48/64")
    group = attributes["source_group_size"]
    if (max(m, n, k, topk) >= 2**31 or n * m * topk * 4 >= 2**31
            or (group != -1 and (k % group or k // group <= 1))):
        raise NotImplementedError("Marlin source requires signed host extents and actual multi-group scales")
    candidates = (((128, 128, 256), (64, 128, 128), (128, 64, 128)) if block_m <= 16
                  else ((64, 256, 256), (64, 128, 128), (128, 64, 128)))
    shared, sms = attributes["source_max_shared_memory_bytes"], attributes["source_num_compute_units"]
    eligible = []
    for tk, tn, threads in candidates:
        cache = _cache_bytes(block_m, group, tk, tn)
        if k % tk == 0 and n % tn == 0 and cache <= shared - 512:
            identity = marlin_candidate_identity(dtype, attributes["source_weight_type"],
                attributes["source_scale_dtype"], block_m, group, tk, tn, threads)
            eligible.append((identity, tk, tn, threads, cache))
    if not isinstance(candidate_resources, Mapping) or set(candidate_resources) != {row[0] for row in eligible}:
        raise ValueError("Marlin requires exact complete shape/cache-valid candidate resources")
    selected = None
    best = 0
    for identity, tk, tn, threads, cache in eligible:
        resource = candidate_resources[identity]
        if resource is None:
            continue
        if (not isinstance(resource, (tuple, list)) or len(resource) != 2
                or any(type(value) is not int for value in resource)
                or not 0 <= resource[0] <= 255 or not 0 <= resource[1] <= shared):
            raise ValueError("Marlin candidate needs actual numRegs/sharedSizeBytes or explicit unavailability")
        registers, static = resource
        allow = min((255 * 1024) // (max(registers, 1) * threads * 4), shared // (cache + 1536))
        allow = max(min(allow, 4 if block_m <= 16 else 2), 1)
        work = (n // tn) * m * topk * 4
        if work < sms * allow:
            allow = max(work // sms, 1)
        if allow > best:
            best, selected = allow, (tk, tn, threads, cache, registers, static)
    if selected is None:
        raise NotImplementedError("Marlin has no available legal SM90 candidate")
    tk, tn, threads, cache, registers, static = selected
    dynamic = shared if best == 1 else shared // best - 1024
    if cache > dynamic or static + dynamic > shared:
        raise ValueError("Marlin selected launch cannot satisfy actual shared-memory request")
    return LaunchConfig(f"vllm_0_30_sm90_marlin_{dtype}_{attributes['source_weight_type']}_g{group}_m{block_m}_n{tn}_k{tk}",
        (sms * best, 1, 1), (threads, 1, 1),
        tiling=TilingSpec(tile_shape={"m": block_m, "n": tn, "k": tk}, pipeline_depth=4),
        parameters={"registers_per_thread": registers, "static_shared_bytes": static,
                    "dynamic_shared_bytes": dynamic, "source_cache_bytes": cache,
                    "source_blocks_per_sm": best})


def select_marlin_modular_expert_recipe(shape, dtype, attributes, hardware, w1_resources, w2_resources):
    """Complete prepared no-LoRA MarlinExperts.apply body, after routing.

    This selected modular caller supplies the two activation arenas/output
    but omits the locks workspace: _fused_marlin_moe allocates and zeroes
    sms*4 INT32 locks on every call. Views, FP32 C_tmp new_empty and arena
    reuse are not copies/fills. W1 and W2 each auto-select from their own
    full candidate facts. No outer EP/TP collective is invented here.
    """
    from src.kernel.launch_config.nvidia_h200.moe_aux_launch import (
        select_source_moe_aux_launch, select_situ_launch, select_clamped_swiglu_launch)
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_fill_launch

    _source(dtype, attributes, hardware)
    tokens, experts, local, topk, hidden, intermediate = require_shape(shape,
        ("tokens", "experts", "local_experts", "topk", "hidden", "intermediate"))
    if (attributes["source_expert_caller"] != "vllm_0_30_modular_marlin_no_lora"
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_prepared_parameters"] != "contiguous_marlin_repacked_int32_and_scales"
            or attributes["source_topk_id_dtype"] != "int32"
            or attributes["source_topk_weight_dtype"] != "fp32"
            or attributes["source_input_quant_dtype"] != "none"
            or attributes["source_has_bias"] is not False
            or attributes["source_apply_router_weight_on_input"] is not False):
        raise NotImplementedError("Marlin recipe requires exact prepared modular A16 caller/operands")
    mapping = attributes["source_has_expert_map"]
    if type(mapping) is not bool or local > experts or topk > experts or (not mapping and local != experts):
        raise ValueError("Marlin actual local/global experts and map disagree")
    if hidden % 64 or intermediate % 64 or ceil_div(experts, 32) * 32 >= 1024 or tokens * topk >= 2**31:
        raise NotImplementedError("Marlin recipe needs unpadded64-wide expert matrices and source alignment extent")
    # global_num_experts is passed by this modular caller, so the Python
    # source first estimates local valid rows, including the EP-off case.
    estimated = ceil_div(tokens * local, experts)
    block_m = 64
    for block_m in (8, 16, 32, 48, 64):
        if estimated * topk / local / block_m < 0.9:
            break
    capacity = tokens * topk + experts * (block_m - 1)
    if tokens * topk < experts:
        capacity = min(capacity, tokens * topk * block_m)
    if capacity >= 2**31:
        raise NotImplementedError("Marlin sorted-token allocation exceeds source signed indices")
    stages = []
    if tokens * topk < 1024 and experts <= 64:
        shared = ((max(experts, 32) + 1) * experts + experts + 1) * 4
        stages.append(("moe_align_small_batch", LaunchConfig("vllm_0_30_sm90_marlin_align_small",
            (1, 1, 1), (256 + max(experts, 32), 1, 1), parameters={"dynamic_shared_bytes": shared})))
    else:
        stages.extend((("moe_align", LaunchConfig("vllm_0_30_sm90_marlin_align",
            (2, 1, 1), (1024, 1, 1), parameters={"dynamic_shared_bytes": ceil_div(experts, 32) * 32 * 4})),
            ("moe_count_and_sort", LaunchConfig("vllm_0_30_sm90_marlin_sort",
                (1, min(ceil_div(tokens * topk, 256), 65535), 1), (256, 1, 1),
                parameters={"dynamic_shared_bytes": 0}))))
    stages.append(("locks_zero", select_torch213_fill_launch(
        {"element_count": attributes["source_num_compute_units"] * 4}, "int32", attributes, hardware)))
    w1 = select_marlin_auto_launch({"m": tokens, "n": 2 * intermediate, "k": hidden,
        "topk": topk, "block_m": block_m}, dtype, attributes, hardware, w1_resources)
    w2 = select_marlin_auto_launch({"m": tokens * topk, "n": hidden, "k": intermediate,
        "topk": 1, "block_m": block_m}, dtype, attributes, hardware, w2_resources)
    # C_tmp is uninitialized storage, not a full read/write or a zero fill.
    # M8 doubles this allocated extent after the source min expression.
    def with_allocation(launch, n):
        scratch = min(n * capacity, attributes["source_num_compute_units"] * 4 * block_m * 256)
        return replace(launch, parameters={**launch.parameters,
            "source_sorted_token_capacity": capacity,
            "source_fp32_scratch_allocation_bytes": scratch * (2 if block_m == 8 else 1) * 4})
    w1, w2 = with_allocation(w1, 2 * intermediate), with_allocation(w2, hidden)
    activation = attributes["source_activation"]
    act_shape = {**shape, "rows": tokens * topk}
    if activation == "silu":
        if attributes["source_clamp_limit"] is not None:
            raise ValueError("plain Marlin SiLU requires the actual absent clamp config")
        act = select_source_moe_aux_launch(shape, dtype, "moe_act_and_mul", attributes, hardware)
    elif activation == "situ":
        act = select_situ_launch(act_shape, dtype,
            {**attributes, "source_component": "vllm_0_30_apply_moe_activation_situ",
             "source_valid_counts_layout": "none"}, hardware)
    elif activation == "silu_clamped":
        act = select_clamped_swiglu_launch(act_shape, dtype,
            {**attributes, "source_component": "vllm_0_30_apply_moe_activation",
             "source_topk_ids": "contiguous_int32",
             "source_expert_map": "contiguous_int32" if mapping else "none"}, hardware)
    else:
        raise NotImplementedError("Marlin activation requires its complete selected gated source")
    reduce = select_source_moe_aux_launch(shape, dtype, "moe_sum", attributes, hardware)
    if mapping:
        # Marlin's contiguous aligned intermediates meet the pad-aware
        # vector path. TopK16 uses the dynamic loop; no separate mask fill.
        suffix = f"top{topk}" if topk in (1, 2, 4, 6, 8, 9) else "dynamic"
        reduce = LaunchConfig("vllm_0_30_sm90_moe_sum_pad_aware_vec8_" + suffix,
            reduce.grid, reduce.block, parameters={"dynamic_shared_bytes": 0})
    return (*stages, ("w1", w1), ("activation", act), ("w2", w2), ("moe_sum", reduce))


def select_marlin_prepared_expert_launch_recipe(shape, dtype, attributes, hardware):
    """Conditional caller bridge; compiler candidates remain explicit inputs.

    The identity is a caller declaration, not verification of a loaded
    library. Synthetic test declarations exercise selection only. Missing
    candidates/identity never become zero-resource or default candidates.
    """
    if ("source_candidate_compiler_identity" not in attributes
            or not isinstance(attributes["source_candidate_compiler_identity"], str)
            or not attributes["source_candidate_compiler_identity"].strip()):
        raise ValueError("Marlin source requires an explicit candidate compiler identity declaration")
    if "source_candidate_resources" not in attributes:
        raise ValueError("Marlin source requires explicit complete W1/W2 candidate resources")
    resources = attributes["source_candidate_resources"]
    if not isinstance(resources, Mapping) or set(resources) != {"w1", "w2"}:
        raise ValueError("Marlin source requires separate complete W1/W2 candidate mappings")
    if attributes["source_router_state"] != "prepared_topk_weights_and_ids":
        raise ValueError("Marlin expert sidecar cannot repeat its external router")
    return select_marlin_modular_expert_recipe(shape, dtype, attributes, hardware,
        resources["w1"], resources["w2"])
