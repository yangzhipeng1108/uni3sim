"""Complete prepared vLLM0.30 all-greedy rejection wrapper on H200.

RejectionSampler.forward first gathers/samples bonus logits,then gathers,
casts and conditionally clones target logits. Its bonus sampler always
requests logits,even when outer logprobs are absent. Raw mode therefore
retains two independent FP32 casts of16-bit bonus logits. The core fills
the full output,argmaxes every draft row,and verifies one request/CTA.
Acceptance is data-dependent;no accepted count or duration is inferred.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import require_dtype, require_shape
from src.kernel.launch_config.nvidia_h200.sampler_launch import select_torch213_argmax_recipe
from src.kernel.launch_config.nvidia_h200.torch_vector_launch import (
    select_torch213_fill_launch, select_torch213_integer_cast_launch,
    select_torch213_large_vocab_logsoftmax_launch, select_torch213_logits_fp32_cast_launch,
    select_torch213_row_index_recipe)


def _greedy_rejection_kernel_launch(batch, limit, hardware):
    """Only the natural all-greedy verification kernel, after its producers."""
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("greedy rejection source requires H200 HWSpec")
    if (type(batch) is not int or batch <= 0 or type(limit) is not int
            or not 0 <= limit <= 128 or batch * (limit + 1) >= 2**31):
        raise ValueError("greedy rejection requires bounded INT32 output rows and max_spec_len0..128")
    # _rejection_greedy_sample returns grid(batch), SYNTHETIC_MODE=False.
    # Its actual bonus input is INT32, despite the INT64 warmup fixture.
    # Non-autotuned Triton defaults to four warps; acceptance changes writes,
    # not the grid. This declares no registers/shared or typed issue price.
    return LaunchConfig("vllm_0_30_sm90_rejection_greedy_int32_bonus",
        (batch, 1, 1), (128, 1, 1),
        parameters={"source_max_spec_len": float(limit), "source_synthetic_mode": 0.0})


def select_greedy_rejection_wrapper_recipe(shape, dtype, attributes, hardware):
    """Every submission after target-model logits and actual metadata exist.

    Includes raw/processed sampler modes,heterogeneous draft counts andzero
    draft requests. Index tensors are actual caller facts,not reconstructed
    from shape. Metadata construction/proposer/model execution and outer
    logprob gathering are separate owners. Runtime stream_memset descriptors
    remain rejected by shared MEMCPY until its owner admits that service.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig

    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("greedy rejection source requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16", "fp32"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_legacy_greedy_rejection_wrapper"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["source_all_greedy"] is not True
            or attributes["source_all_random"] is not False
            or attributes["source_synthetic_mode"] is not False
            or attributes["source_max_num_logprobs"] is not None
            or attributes["source_logprob_token_ids"] != ()
            or attributes["source_allowed_token_ids_mask"] != "none"
            or attributes["source_bad_words"] != ()
            or attributes["source_non_argmax_processors"] != ()
            or attributes["source_no_penalties"] is not True
            or attributes["source_thinking_budget_holder"] != "none"
            or attributes["source_draft_and_cu_dtype"] != "int32_contiguous"):
        raise NotImplementedError("greedy rejection requires exact0.30/Torch2.13 ordinary all-greedy metadata without extra processing")
    batch, vocab, logits_rows = require_shape(shape, ("batch", "vocab", "logits_rows"))
    limit = shape["max_spec_len"]
    counts = attributes["num_draft_tokens"]
    if (type(limit) is not int or not 0 <= limit <= 128
            or not isinstance(counts, (tuple, list)) or len(counts) != batch
            or any(type(count) is not int or not 0 <= count <= limit for count in counts)):
        raise ValueError("rejection requires actual per-request draft counts within the source max_spec_len0..128")
    cumulative, total = [], 0
    for count in counts:
        total += count
        cumulative.append(total)
    if (tuple(attributes["cu_num_draft_tokens"]) != tuple(cumulative)
            or any(type(value) is not int for value in attributes["cu_num_draft_tokens"])):
        raise ValueError("rejection cumulative metadata must exactly match actual per-request draft counts")
    if logits_rows < total + batch or total >= 2**31:
        raise ValueError("rejection source logits do not cover all target/bonus rows")
    mode = attributes["source_logprobs_mode"]
    if mode not in ("raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits"):
        raise ValueError("rejection requires the actual sampler logprobs mode")
    raw = mode in ("raw_logprobs", "raw_logits")
    stages = []
    def append(role, recipe):
        stages.extend((role + "_" + name, submission) for name, submission in recipe)
    def index(role, rows, indices):
        append(role, select_torch213_row_index_recipe({"rows": rows, "source_rows": logits_rows, "columns": vocab},
            dtype, {**attributes, "source_row_indices": indices}, hardware))
    def fp32_cast(role, rows):
        stages.append((role, select_torch213_logits_fp32_cast_launch({"element_count": rows * vocab}, dtype, attributes, hardware)))
    def clone(role, rows):
        stages.append((role, PhysicalTransferConfig(algorithm="stream_copy", bytes=rows * vocab * 4, direction="d2d", link="dram")))
    # The original wrapper's first advanced index is always a fresh tensor.
    index("bonus_logits", batch, attributes["bonus_logits_indices"])
    if raw:
        if dtype == "fp32":
            clone("raw_bonus_logits_clone", batch)
        else:
            fp32_cast("raw_bonus_logits_fp32_cast", batch)
    if dtype != "fp32":
        fp32_cast("bonus_sampler_logits_fp32_cast", batch)
    append("bonus", select_torch213_argmax_recipe({"rows": batch, "vocab": vocab}, "fp32", attributes, hardware))
    # Sampler.forward uses the override for raw capture,but does not forward
    # it to self.sample. An original processed_logprobs mode therefore still
    # computes log_softmax here,even though the wrapper requested logits.
    if mode == "processed_logprobs":
        stages.append(("bonus_processed_logsoftmax", select_torch213_large_vocab_logsoftmax_launch(
            {"rows": batch, "vocab": vocab}, "fp32", attributes, hardware)))
    stages.append(("bonus_ids_int32", select_torch213_integer_cast_launch({"element_count": batch}, "int64",
        {**attributes, "source_output_dtype": "int32"}, hardware)))
    # INT64 long()/view and final unsqueeze only alias these sampler outputs.
    index("target_logits", total, attributes["target_logits_indices"])
    if total:
        if dtype != "fp32":
            fp32_cast("target_logits_fp32_cast", total)
        if raw:
            clone("raw_target_logits_clone", total)
    stages.append(("output_placeholder_fill", select_torch213_fill_launch(
        {"element_count": batch * (limit + 1)}, "int32", attributes, hardware)))
    if total:
        append("target", select_torch213_argmax_recipe({"rows": total, "vocab": vocab}, "fp32", attributes, hardware))
    stages.append(("rejection_greedy", _greedy_rejection_kernel_launch(batch, limit, hardware)))
    return tuple(stages)
