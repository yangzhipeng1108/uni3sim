"""Compiled SM90 facts for the fixed vLLM0.11 library, without GPU execution.

The software artifact identity has one existing owner:
microbench/nvidia_a800/references/vllm_rmsnorm_v0_11_0_sm80.json,
installed_artifacts/0. Only that shared-library identity is reused here.
Resource values below were read independently from its SM90 code with NVIDIA
cuobjdump12.9.82, not from that owner's SM80 fields. Regeneration command:
cuobjdump --dump-resource-usage --gpu-architecture sm_90 <fixed _C.abi3.so>.
The static-resource receipt is in this Goal's vllm011_sm90_resources.json.
These facts do not describe the vLLM0.30 kernel dataset/compiler.
"""

from src.kernel.launch_config.selection_primitives import require_dtype, require_shape


def select_legacy_vector_resources(shape, dtype, variant, attributes):
    """Return existing LaunchConfig parameter names for a bound0.11 program.

    The public KernelSpec resource-field bridge remains a root integration
    patch. Callers must keep these facts separate from0.30 resources until
    that shared contract accepts them; no predictor/rate is defined here.
    """
    require_dtype(dtype, ("bf16", "fp16"))
    if attributes["model_execution_version"] != "vllm_0_11_0":
        raise NotImplementedError("compiled SM90 resources bind only vLLM0.11")
    layouts = {"contiguous_aligned_16"}
    if variant in ("rmsnorm", "silu"):
        layouts.add("contiguous")
    elif variant in ("rope", "kv_cache", "kvcache", "paged_kv_cache"):
        layouts.update(("contiguous", "unit_final_stride"))
    if attributes["source_input_layout"] not in layouts:
        raise NotImplementedError("compiled SM90 resource selection needs its source layout")
    if variant in ("rmsnorm", "add_rmsnorm"):
        hidden, = require_shape(shape, ("hidden",))
        registers = 32 if variant == "add_rmsnorm" and hidden % 8 == 0 else 22
        shared = 1192
    elif variant == "silu":
        registers, shared = 22, 0
    elif variant == "rope":
        is_neox = attributes["is_neox"]
        if type(is_neox) is not bool:
            raise TypeError("compiled RoPE resources require a boolean source style")
        registers, shared = (32 if is_neox else 31), 0
    elif variant in ("kv_cache", "kvcache", "paged_kv_cache"):
        if attributes["kv_cache_dtype"] != "auto":
            raise NotImplementedError("compiled cache resources require the matching-storage template")
        registers, shared = 56, 0
    else:
        raise NotImplementedError("compiled SM90 source template is not bound for this variant")
    return {"registers_per_thread": float(registers), "static_shared_bytes": float(shared),
            "dynamic_shared_bytes": 0.0}


def select_legacy_moe_aux_resources(shape, dtype, variant, attributes):
    """Independent0.11 SM90 _moe_C resources, awaiting the root ABI bridge.

    Software identity is owned by
    microbench/nvidia_a800/references/pytorch_moe_reduce_vllm_0_11_0_torch_2_8_0.json,
    installed_artifacts/1 (the unique _moe_C.abi3.so entry). SM90 resources
    were read directly with the same cuobjdump command as above, rather than
    consuming that reference's SM80 resource fields. Regenerable evidence:
    experiment_logs/h200_launch_rules_20261104/vllm011_moe_sm90_resources.json.
    Dynamic shared is independently specified by0.11 CUDA host wrappers.
    Torch reduction and Triton main resources are not part of this library.
    """
    from src.kernel.launch_config.selection_primitives import ceil_div

    require_dtype(dtype, ("bf16", "fp32") if variant == "moe_topk" else ("bf16",))
    if attributes["model_execution_version"] != "vllm_0_11_0":
        raise NotImplementedError("compiled SM90 MoE resources bind only vLLM0.11")
    tokens, experts, topk = require_shape(shape, ("tokens", "experts", "topk"))
    if topk > experts:
        raise ValueError("topk exceeds global experts")
    dynamic = 0
    if variant == "moe_topk":
        if (attributes["routing_input_dtype"] != "fp32"
                or attributes["source_routing"] != "softmax"
                or attributes["source_topk_id_dtype"] != "int32"):
            raise NotImplementedError("compiled topk binds FP32 logits and INT32 selected IDs")
        # Each entry identifies an actual compiled EXPERTS specialization.
        registers_by_experts = {1: 30, 2: 32, 4: 30, 8: 30, 16: 30,
            32: 32, 64: 32, 128: 32, 192: 32, 256: 32, 320: 32,
            384: 32, 448: 32, 512: 39, 576: 40}
        if experts not in registers_by_experts:
            raise NotImplementedError("compiled fused topk expert specialization is not bound")
        registers, shared = registers_by_experts[experts], 0
    elif variant in ("moe_align_small_batch", "moe_align", "moe_count_and_sort"):
        if attributes["source_topk_id_dtype"] != "int32":
            raise NotImplementedError("compiled alignment requires INT32 routed IDs")
        if ceil_div(experts, 32) * 32 >= 1024:
            raise ValueError("compiled alignment requires padded experts below1024")
        small = tokens * topk < 1024 and experts <= 64
        if (variant == "moe_align_small_batch") != small:
            raise ValueError("compiled alignment helper disagrees with its source branch")
        if small:
            registers, shared = 28, 1024
            threads = max(experts, 32)
            dynamic = ((threads + 1) * experts + experts + 1) * 4
        elif variant == "moe_align":
            registers, shared = 45, 5280
            dynamic = ceil_div(experts, 32) * 32 * 4
        else:
            registers, shared = 16, 1024
    elif variant == "moe_sum":
        if topk not in (2, 3, 4):
            raise NotImplementedError("Torch reduction is not compiled into the MoE CUDA sum template")
        if (attributes["source_input_layout"] != "contiguous_aligned_16"
                or type(attributes["work_fraction"]) not in (int, float)
                or attributes["work_fraction"] != 1.0):
            raise NotImplementedError("compiled CUDA sum binds complete contiguous operands")
        registers, shared = {2: 20, 3: 22, 4: 32}[topk], 1024
    elif variant == "moe_act_and_mul":
        # The activation is in _C, whose independent SM90 entry is owned above.
        return select_legacy_vector_resources(shape, dtype, "silu", attributes)
    else:
        raise NotImplementedError("compiled SM90 resources are not bound for this MoE stage")
    return {"registers_per_thread": float(registers), "static_shared_bytes": float(shared),
            "dynamic_shared_bytes": float(dynamic)}
