"""Static operator evidence for independent source comparison, without gates."""

from collections.abc import Mapping
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from math import isfinite
import json
import hashlib
import re
import shlex
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
H200_HARDWARE_CONFIG = REPOSITORY_ROOT / "configs/hardware/nvidia_h200.yaml"

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor, predict
from src.kernel.kernel_spec import KernelKind
from src.kernel.predictors.gemm_block_fp8 import is_block_fp8_gemm
from src.kernel.predictors.fused_moe import dsv4_marlin_moe_source_work, dsv4_marlin_moe_roofline_work, dsv4_marlin_moe_scratch_work
from src.kernel.predictors.sparse_attn_indexer import dsv4_indexer_logical_work
from src.kernel.predictors.kv_cache_update import (
    dsv4_finish_logical_work, dsv4_swa_insert_tensor_objects,
    dsv4_partial_state_tensor_objects, dsv4_finish_partial_state_input,
)
from src.kernel.predictors.sparse_attn import dsv4_prefill_gather_logical_work, dsv4_attention_source_inputs, dsv4_decode_index_materialization_work, dsv4_decode_attention_index_inputs
from src.kernel.predictors.vector import (
    mhc_prenorm_logical_work, mhc_prenorm_roofline_work, mhc_stage_logical_work,
    dsv4_indexer_q_rope_quant_logical_work, dsv4_indexer_q_roofline_work,
    clamped_silu_logical_work, rmsnorm_logical_work,
    dsv4_moe_route_logical_work, dsv4_route_source_work, dsv32_moe_route_logical_work,
    dsv4_short_context_topk_source_work,
    dsv4_route_tensor_objects,
    kimi_moe_route_logical_work, vocab_parallel_embedding_logical_work,
    dsv32_draft_sampling_logical_work, vllm_logits_row_selection_logical_work,
    fp8_group_quant_source_work,
    vllm_logits_row_selection_roofline_bytes,
    vllm_logits_fp32_cast_logical_work, vllm_logits_gather_reorder_logical_work,
    vllm_plain_greedy_sampling_logical_work, vllm_plain_greedy_roofline_work,
    vllm_rejection_logits_materialization_logical_work,
    DSV32_DRAFT_MATERIALIZATION_VARIANTS, dsv32_draft_materialization_roofline_bytes,
    DSV32_DRAFT_INT32_BOOKKEEPING_VARIANTS, dsv32_draft_int32_bookkeeping_roofline_work)
from src.serving.deployment.deployment_config import (
    CollectiveConfig, DeploymentConfig, ParallelConfig, PipelineP2PConfig, WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.kernel_lowering.execution_timeline import build_execution_timeline
from src.serving.schedule.schedule_step import ScheduledRequest, ScheduleStep
from src.serving.kv_cache.model_residency import (
    dsv4_non_routed_loader_payload_bytes, glm_block_fp8_loader_payload_bytes,
    glm_block_fp8_constructor_buffer_bytes,
    glm_block_fp8_postload_parameter_bytes, GLM_SM90_DEEP_GEMM_STORAGE_SOURCE,
    glm_block_fp8_deepgemm_workspace_bytes,
    glm_block_fp8_deepgemm_call_allocation_bytes,
    dsv4_routed_loader_payload_bytes, kimi_k25_outer_loader_payload_bytes,
    model_rank_residencies)
from src.serving.workload.traced_sampling import (
    glm_gpu_v2_mtp_storage_work, glm_gpu_v2_prepare_decode_work,
    glm_gpu_v2_slot_mapping_work, glm_gpu_v2_metadata_refresh_plan,
    glm_gpu_v2_sparse_backend_metadata_plan)
from src.serving.workload.model_config import (
    load_model_config, GLM_BLOCK_FP8_CONFIG_SHA256S, is_validated_qwen3_moe_ep2_model_scope)
from src.serving.workload.runtime_inputs import (ExpertRoutingInput, RuntimeInputs, V32PaddedMTPInputs,
    vllm_0_30_source_model_inputs)
from src.serving.workload.traced_models import build_symbolic_workload


def _json_value(value):
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"operator evidence cannot encode {type(value).__name__}")


def _snapshot():
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "uncommitted_changes": subprocess.check_output(
            ["git", "status", "--short"], text=True).splitlines(),
    }


def write_kimi_k3_nope_prefill_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-prefill")


def write_kimi_k3_nope_decode_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-decode")


def write_kimi_k3_nope_mixed_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-mixed")


def write_kimi_k3_kda_mixed_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "kda-mixed")


def write_kimi_k3_mixed_greedy_step_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "mixed-greedy-step")


def write_kimi_k3_kda_context_prefill_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "kda-context-prefill")


def write_kimi_k3_nope_context_prefill_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-context-prefill")


def write_kimi_k3_context_prefill_greedy_step_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "context-prefill-greedy-step")


def write_kimi_k3_nope_context_continuation_components(model_path, source_root, output, *, backend_supports_out):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-context-continuation",
                                      backend_supports_out=backend_supports_out)


def write_kimi_k3_context_continuation_greedy_step_components(model_path, source_root, output, *, backend_supports_out):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "context-continuation-greedy-step",
                                      backend_supports_out=backend_supports_out)


def write_kimi_k3_native_latent_output_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "native-latent-output")


def write_kimi_k3_marlin_routed_output_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "marlin-routed-output")


def write_kimi_k3_native_marlin_moe_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "native-marlin-moe")


def write_kimi_k3_nope_prefill_layer_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-prefill-layer")


def write_kimi_k3_kda_fresh_prefill_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "kda-fresh-prefill")


def write_kimi_k3_dense_kda_prefill_layer_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "dense-kda-prefill-layer")


def write_kimi_k3_fresh_prefill_decoder_stack_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "fresh-prefill-decoder-stack")


def write_kimi_k3_fresh_prefill_logits_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "fresh-prefill-logits")


def write_kimi_k3_fresh_prefill_forward_logits_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "fresh-prefill-forward-logits")


def write_kimi_k3_single_decode_forward_logits_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "single-decode-forward-logits")


def write_kimi_k3_fresh_prefill_greedy_step_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "fresh-prefill-greedy-step")


def write_kimi_k3_single_decode_greedy_step_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "single-decode-greedy-step")


def write_kimi_k3_situ_offline_compile(model_path, source_root, output):
    """Compile the unchanged activation TU; publish conditional static facts only.

    No GPU context/module, measured duration or production parameter is
    created. The task-local SDK supplies real Torch headers. The local
    CUDA12.2/SM90 experiment is explicitly not an observed runtime build.
    """
    import ast
    from src.serving.backend.vllm_scheduler_process import vllm030_source_identity

    model = load_model_config(model_path)
    if model.kimi_k3_semantics is None:
        raise ValueError("SiTU compiler probe requires the exact K3 config")
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    identity = vllm030_source_identity(source_root)
    directory = output.with_suffix("")
    directory.mkdir(parents=True, exist_ok=True)
    sdk = Path(".venv-vllm030/bin/python").absolute()
    # Inspect the installed module's location without importing Torch/CUDA.
    torch_root = Path(subprocess.check_output([
        str(sdk), "-c", "import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec('torch').origin).parent)"
    ], text=True).strip())
    module = ast.parse((torch_root / "utils/cpp_extension.py").read_text())
    flags = None
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "COMMON_NVCC_FLAGS"
                                               for target in node.targets):
            flags = ast.literal_eval(node.value)
    if flags is None or not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
        raise ValueError("Torch COMMON_NVCC_FLAGS is not a literal flag list")
    cmake = (source_root / "cmake/utils.cmake").read_text()
    removed = ("-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
               "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__")
    if "if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)" not in cmake or any(flag not in cmake for flag in removed):
        raise ValueError("verified source does not bind the CUDA>=12.0 Torch-flag removal")
    nvcc = Path("/usr/local/cuda-12.2/bin/nvcc")
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "release 12.2," not in version:
        raise ValueError("this offline source experiment requires the declared CUDA12.2 toolchain")
    command = [str(nvcc), "--cubin", "-std=c++20", "-arch=sm_90", "-O3",
               "-DTORCH_TARGET_VERSION=0x020B000000000000ULL", "-DUSE_CUDA", "-DENABLE_FP8",
               *(flag for flag in flags if flag not in removed),
               f"-I{source_root / 'csrc'}", f"-I{torch_root / 'include'}",
               f"-I{torch_root / 'include/torch/csrc/api/include'}", "-o", str(directory / "activation.cubin"),
               str(source_root / "csrc/libtorch_stable/activation_kernels.cu")]
    with (directory / "compile.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    metadata = {"snapshot": _snapshot(), "source_identity": identity, "command": command,
                "compiler_version": version, "torch_headers": str(torch_root / "include"),
                "removed_flags": removed, "source_transform": None, "exit_code": result.returncode,
                "GPU_module_loaded_or_run": False, "actual_runtime_build_bound": False}
    (directory / "command.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if result.returncode != 0:
        raise RuntimeError(f"unchanged activation TU compilation failed; see {directory / 'compile.log'}")
    dump = subprocess.check_output([str(nvcc.parent / "cuobjdump"), "--dump-sass", str(directory / "activation.cubin")], text=True)
    (directory / "activation.sass").write_text(dump)
    instances = []
    for part in dump.split("Function : ")[1:]:
        symbol, body = part.split("\n", 1)
        if not symbol.startswith("_ZN4vllm19situ_and_mul_kernel"):
            continue
        opcodes = Counter(re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\s", body))
        instances.append({"symbol": symbol.strip(), "static_opcode_counts": dict(opcodes),
                          "magnitude_branch_compares": [line.strip() for line in body.splitlines()
                                                       if "FSETP.GE" in line and "0.60000002384185791016" in line]})
    bf16 = [instance for instance in instances if "BFloat16" in instance["symbol"]]
    if len(bf16) != 1 or not bf16[0]["magnitude_branch_compares"]:
        raise ValueError("offline BF16 ordinary SiTU magnitude-branch evidence not identified")
    metadata.update(component_count=len(instances), instances=instances,
        scope="local_CUDA12.2_SM90_default_math_complete_activation_TU_static_compiler_facts",
        dynamic_instruction_counts_inferred=False, provider_accepted=False, runtime_accepted=False,
        conditional_scalar_examples=[
            {"gate": 0, "up": 0, "tanh_arguments": [0, 0, 0], "magnitude_ge_0_6": [False, False, False]},
            {"gate": 4, "up": 25, "tanh_arguments": [1, 2, 1], "magnitude_ge_0_6": [True, True, True]}],
        scalar_example_scope="constructed finite BF16-representable source predicate examples; not checkpoint/model/GPU outputs",
        production_parameters_changed=False,
        conclusion="shape and three tanhf calls do not fix dynamic SFU demand; preserve unpriced boundary",
        limitations=["compiler flags derived for this local toolchain, not an observed installed vLLM runtime",
                     "static instruction counts include loops/alternate paths and are not per-element dynamic counts",
                     "input magnitude activity and actual toolchain/provider binding remain unobserved",
                     "no latency or issue rate inferred; no GPU module loaded or executed"])
    output.write_text(json.dumps(_json_value(metadata), indent=2) + "\n")
    return metadata


def _write_kimi_k3_source_chain(model_path, source_root, output, component, *, backend_supports_out=None):
    """One new source branch; do not rerun previously published components."""
    from src.serving.backend.vllm_scheduler_process import vllm030_source_identity
    from src.serving.workload.kimi_k3 import (
        vllm030_kimi_k3_nope_fresh_prefill_workload, vllm030_kimi_k3_nope_decode_workload,
        vllm030_kimi_k3_nope_mixed_workload, vllm030_kimi_k3_triton_mixed_workload,
        vllm030_kimi_k3_mixed_forward_logits_workload, vllm030_kimi_k3_triton_context_prefill_workload,
        vllm030_kimi_k3_nope_context_prefill_workload, vllm030_kimi_k3_context_prefill_forward_logits_workload,
        vllm030_kimi_k3_tp1_native_latent_output_workload,
        vllm030_kimi_k3_marlin_routed_output_workload,
        vllm030_kimi_k3_native_marlin_moe_workload, vllm030_kimi_k3_nope_prefill_layer_workload,
        vllm030_kimi_k3_triton_fresh_prefill_workload, vllm030_kimi_k3_dense_fresh_prefill_layer_workload,
        vllm030_kimi_k3_fresh_prefill_decoder_stack_workload, vllm030_kimi_k3_fresh_prefill_logits_workload,
        vllm030_kimi_k3_fresh_prefill_forward_logits_workload,
        vllm030_kimi_k3_single_decode_forward_logits_workload, vllm030_kimi_k3_append_plain_greedy_sampler)
    from src.kernel.predictors.vector import (
        kimi_k3_nope_prefill_logical_work, kimi_k3_nope_decode_logical_work, kimi_k3_nope_mixed_logical_work,
        kimi_k3_nope_context_logical_work,
        kimi_k3_moe_route_logical_work,
        kimi_k3_situ_logical_work, kimi_k3_attention_residual_logical_work,
        kimi_k3_kda_prefill_logical_work, kimi_k3_kda_mixed_logical_work, kimi_k3_kda_context_prefill_logical_work,
        kimi_k3_kda_decode_logical_work, vllm_embedding_id_widen_logical_work,
        vllm_plain_greedy_sampling_logical_work)
    from src.kernel.predictors.fused_moe import kimi_k3_marlin_expert_logical_work
    from src.kernel.predictors.gemm import kimi_k3_nope_bmm_logical_work
    from src.kernel.predictors.flash_attn import kimi_k3_nope_context_attention_logical_work

    model = load_model_config(model_path)
    identity = vllm030_source_identity(Path(source_root))
    hardware = HWSpec(H200_HARDWARE_CONFIG)
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (WorkerPoolConfig("main", "colocated", "nvidia_h200"),)),
        num_layers=model.num_hidden_layers, num_experts=model.num_experts)
    if component in ("single-decode-forward-logits", "single-decode-greedy-step"):
        kinds = model.kimi_k3_semantics.layer_attention_types
        scenario = {"input_ids": (1,), "sequence_length": 513, "state_pool_slots": 2,
                    "state_indices_by_layer": {layer: 1 for layer, kind in enumerate(kinds) if kind == "kda"},
                    "slot_mapping_by_layer": {layer: (512,) for layer, kind in enumerate(kinds) if kind == "mla_nope"},
                    "cache_capacity_tokens": 1024, "shared_experts_stream_token_threshold": 256,
                    "aux_streams_available": True, "pool_name": "main", "rank": 0}
        step_id = "source-k3-single-decode-stack"
        declared = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0)
        routes = {layer: declared.resolve(
            step_id=step_id, layer_index=layer, moe_invocation_id=f"{step_id}:layer-{layer}:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=1, num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
                  for layer in range(1, model.num_hidden_layers)}
        graph = vllm030_kimi_k3_single_decode_forward_logits_workload(model, routing_by_layer=routes, **scenario)
        scope = "TP1_BF16_single_request_context513_embedding_93_layer_decode_and_logits_source_candidate"
        limitations = [
            "input token and every route/cache address are declared source inputs, not model or observed output",
            "one continuation token; independent logical per-layer cache namespaces, not physical shared-pool allocations",
            "explicit ordinary auxiliary streams; native CUDA streams/events and compiler/provider unbound",
            "router/down join precedes external shared/routed overlap; shared completion waits before latent transform",
            "NoPE short frontend retains QKV/gate slices, absorption,576 query/cache and V up-projection",
            "all92 histograms retain seeded-assumption provenance, not checkpoint logits",
            "sampler, metadata/index producers, normal capacity and normalized Serving bridge remain outside this graph"]
    elif component in ("fresh-prefill-decoder-stack", "fresh-prefill-forward-logits", "fresh-prefill-greedy-step"):
        kinds = model.kimi_k3_semantics.layer_attention_types
        scenario = {"query_start_loc": (0, 512), "state_pool_slots": 2,
                    "state_indices_by_layer": {layer: 1 for layer, kind in enumerate(kinds) if kind == "kda"},
                    "slot_mapping_by_layer": {layer: tuple(range(512)) for layer, kind in enumerate(kinds) if kind == "mla_nope"},
                    "cache_capacity_tokens": 512, "shared_experts_stream_token_threshold": 256,
                    "pool_name": "main", "rank": 0}
        step_id = "source-k3-fresh-prefill-stack"
        declared = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0)
        routes = {layer: declared.resolve(
            step_id=step_id, layer_index=layer, moe_invocation_id=f"{step_id}:layer-{layer}:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=512, num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
                  for layer in range(1, model.num_hidden_layers)}
        if component == "fresh-prefill-decoder-stack":
            graph = vllm030_kimi_k3_fresh_prefill_decoder_stack_workload(model, routing_by_layer=routes, **scenario)
        else:
            graph = vllm030_kimi_k3_fresh_prefill_forward_logits_workload(
                model, input_ids=tuple(range(512)), routing_by_layer=routes,
                **{key: value for key, value in scenario.items() if key != "query_start_loc"})
            scenario["input_ids"] = tuple(range(512))
        scope = "TP1_T512_fresh_93_layer_hybrid_decoder_stack_plus_output_AttnRes_source_candidate"
        limitations = [
            "input embedding hidden states external; excludes row selection, final norm, lm_head and sampler",
            "separate logical per-layer cache namespaces; declared slots are not actual shared native allocations",
            "all92 histograms retain seeded-assumption provenance, not checkpoint-computed logits or observed IDs",
            "one fresh full64-chunk sequence; no context/decode/mixed, MTP, auxiliary capture or distributed execution",
            "explicit Triton KDA/standard Marlin source candidates; actual backend/provider dispatch unbound",
            "chunk, SiTU, route, expert aggregate and NoPE sigmoid gate remain unpriced",
            "source node/aggregate counts are not GPU launch counts; source graph is not a complete model"]
        if component in ("fresh-prefill-forward-logits", "fresh-prefill-greedy-step"):
            scope = "TP1_T512_fresh_embedding_93_layer_stack_and_selected_row_logits_source_candidate"
            limitations[0] = "includes INT32-to-INT64 embedding IDs, BF16 embedding output and selected-row norm/head; excludes sampler"
    elif component == "fresh-prefill-logits":
        scenario = {"query_start_loc": (0, 512), "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_fresh_prefill_logits_workload(model, **scenario)
        scope = "TP1_T512_single_fresh_sequence_row_selection_final_RMSNorm_and_BF16_lm_head"
        limitations = [
            "external stack pre-norm hidden output; last-row selection precedes final norm and head",
            "one INT32 query boundary without speculation, prompt logprobs, pooling or overrides",
            "sampler, index producer/H2D and embedding input conversion are outside this component",
            "row selection logical-copy Roofline does not establish actual library/compiler launch",
            "generic analytical norm/head do not bind actual H200 provider"]
    elif component == "nope-decode":
        scenario = {"layer_index": 3, "slot_mapping": (512,), "kv_lens": (513,), "cache_capacity_tokens": 1024,
                    "cache_dtype": "bf16", "aux_stream_available": True, "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_nope_decode_workload(model, **scenario)
        scope = "TP1_BF16_NoPE_MLA_single_decode_sequence513_source_candidate"
        limitations = [
            "declared cache slots, sequence lengths and cache contents, not native Scheduler/metadata execution",
            "model source owns a shared auxiliary stream; T<512 slices QKV/gate with an output-gate join",
            "short _unquantized_gemm low-latency-vs-F.linear dispatch remains unbound",
            "BMM absorbs all Q rows; 576-wide query concat/cache insert precedes latent attention",
            "W_UV BMM writes the attention output directly, without a separate copy",
            "decode concat CUDA source uses3 cooperating warps per row at T<=64 and1 above64",
            "canonical analytical MLA decoder/split selector is not actual0.30 backend/provider execution",
            "compiled BF16 sigmoid output gate retains an unpriced materialization/compiler boundary"]
    elif component == "mixed-greedy-step":
        kinds = model.kimi_k3_semantics.layer_attention_types
        scenario = {"input_ids": (1, *range(512)), "decode_kv_len": 513, "state_pool_slots": 3,
                    "state_indices_by_layer": {layer: (1, 2) for layer, kind in enumerate(kinds) if kind == "kda"},
                    "slot_mapping_by_layer": {layer: (512, *range(1024, 1536))
                                              for layer, kind in enumerate(kinds) if kind == "mla_nope"},
                    "cache_capacity_tokens": 2048, "shared_experts_stream_token_threshold": 256,
                    "aux_streams_available": True, "pool_name": "main", "rank": 0}
        step_id = "source-k3-mixed-greedy"
        declared = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0)
        routes = {layer: declared.resolve(
            step_id=step_id, layer_index=layer, moe_invocation_id=f"{step_id}:layer-{layer}:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=513, num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
                  for layer in range(1, model.num_hidden_layers)}
        graph = vllm030_kimi_k3_mixed_forward_logits_workload(model, routing_by_layer=routes, **scenario)
        scope = "TP1_embedding_93_layer_mixed_one_decode_one_fresh512_and_two_selected_row_logits_source_candidate"
        limitations = [
            "declared input IDs and per-layer cache slots, not native Scheduler/metadata or computed cache values",
            "all92 expert histograms retain seeded-assumption provenance, not checkpoint logits or observed IDs",
            "KDA uses mixed chunk for decode1 and fresh512; NoPE shares frontend and runs prefill first",
            "row0 androw512 selected before two-row final norm/head; excludes sampler",
            "ordinary aux streams declared; no overlap discount or actual provider/backend execution",
            "chunk, SiTU, route, expert aggregate and NoPE output gate remain unpriced",
            "no context prefill, speculation, MTP, distributed or normal H200 capacity acceptance"]
    elif component in ("context-prefill-greedy-step", "context-continuation-greedy-step"):
        kinds = model.kimi_k3_semantics.layer_attention_types
        scenario = {"input_ids": tuple(range(512)), "context_length": 1024, "state_pool_slots": 2,
                    "state_indices_by_layer": {layer: 1 for layer, kind in enumerate(kinds) if kind == "kda"},
                    "slot_mapping_by_layer": {layer: tuple(range(1024, 1536))
                                              for layer, kind in enumerate(kinds) if kind == "mla_nope"},
                    "context_block_tables_by_layer": {layer: tuple(range(16))
                                                      for layer, kind in enumerate(kinds) if kind == "mla_nope"},
                    "block_size": 64, "cache_capacity_tokens": 2048, "shared_experts_stream_token_threshold": 256,
                    "aux_streams_available": True, "prefill_source": "declared_fa3_bf16", "pool_name": "main", "rank": 0}
        if component == "context-continuation-greedy-step":
            if type(backend_supports_out) is not bool:
                raise ValueError("context continuation report requires explicit backend supports_out")
            scenario.update(context_chunk_lengths=(512, 512), backend_supports_out=backend_supports_out)
        step_id = "source-k3-context-greedy"
        declared = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0)
        routes = {layer: declared.resolve(
            step_id=step_id, layer_index=layer, moe_invocation_id=f"{step_id}:layer-{layer}:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=512, num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
                  for layer in range(1, model.num_hidden_layers)}
        graph = vllm030_kimi_k3_context_prefill_forward_logits_workload(model, routing_by_layer=routes, **scenario)
        scope = "TP1_embedding_93_layer_initialized_context1024_new512_and_selected_row_logits_source_candidate"
        limitations = [
            "declared prefix cache/state contents, slots and input IDs, no native Scheduler context trace",
            "all92 route histograms retain seeded-assumption provenance, not checkpoint logits or observed IDs",
            "KDA reads existing conv/FP32 recurrent state; every NoPE layer gathers historical latent and merges two LSE partials",
            "single complete prefix chunk covers all new rows; no continuation chunks, empty slices, speculation or DCP",
            "canonical analytical FA3 reuse is explicit for new tokens, noncausal context backend remains unbound",
            "last query row511 selected before final norm/head; excludes sampler",
            "chunk/SiTU/route/expert/context attention/merge/output gate remain unpriced",
            "CPU MLA disables chunked prefill; source scenario is not actual Scheduler/normal capacity/provider acceptance"]
        if component == "context-continuation-greedy-step":
            scope += "__two512_context_chunks_explicit_supports_out"
            limitations.append("continuation merges in-place BF16 partial and FP32 LSE; first chunk copies depend on declared supports_out")
            limitations[3] = "one prefix request with two512 chunks; no empty slices, speculation or DCP"
    elif component in ("nope-context-prefill", "nope-context-continuation"):
        scenario = {"layer_index": 3, "slot_mapping": tuple(range(1024, 1536)), "context_length": 1024,
                    "context_block_table": tuple(range(16)), "block_size": 64, "cache_capacity_tokens": 2048,
                    "cache_dtype": "bf16", "prefill_source": "declared_fa3_bf16", "pool_name": "main", "rank": 0}
        if component == "nope-context-continuation":
            if type(backend_supports_out) is not bool:
                raise ValueError("context continuation report requires explicit backend supports_out")
            scenario.update(context_chunk_lengths=(512, 512), backend_supports_out=backend_supports_out)
        graph = vllm030_kimi_k3_nope_context_prefill_workload(model, **scenario)
        scope = "TP1_BF16_NoPE_one_complete_prefix1024_chunk_and_new512_source_candidate"
        limitations = [
            "declared cache blocks/slots/contents and source backend profile, not native metadata or observed provider",
            "one complete prefix chunk starts0 and covers all queries; no continuation/empty slices/DCP",
            "new causal FA3 analytical source includes FP32 LSE output in the same call",
            "context gather BF16-to-BF16 has no scale/token_to_seq read; latent cast aliases BF16 workspace",
            "history KV-B and key concat preserve strided V view; no context accumulator or output copy",
            "noncausal context attention exports tensor math/typed roles, backend softmax/tile/split remains unpriced",
            "merge LSE value branches/per-pack compiler work and BF16 output gate remain unpriced",
            "CPU platform disables MLA chunked prefill, so this source scenario is not a Scheduler context trace"]
        if component == "nope-context-continuation":
            scope += "__two512_context_chunks_explicit_supports_out"
            limitations[1] = "one prefix request split into two512 chunks, second a continuation; no empty slices/DCP"
            limitations[4] = "strided V aliases history KV-B; first output copy omitted only when declared supports_out"
            limitations.append("first chunk copies FP32 LSE; continuation merges in-place output+LSE, final suffix merge does not write LSE")
    elif component == "kda-context-prefill":
        scenario = {"layer_index": 0, "query_start_loc": (0, 128), "state_index": 1,
                    "state_pool_slots": 2, "has_initial_state": True,
                    "conv_state_dtype": "bf16", "recurrent_state_dtype": "fp32", "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_triton_context_prefill_workload(model, **scenario)
        scope = "TP1_Triton_KDA_one_initialized128_token_context_extension_source_candidate"
        limitations = [
            "declared initial state/index and full64-chunk query; prior length/cache values external, not a Scheduler trace",
            "three conv_fn calls read prior3 history values per channel; full query needs no short-history shift",
            "gather reads one FP32 pool row and writes one dense row; chunk final state scatters into the same pool slot",
            "multi-kernel chunk internal math/traffic remains unpriced, with no recurrence substitution",
            "not a complete hybrid decoder context graph, actual backend/provider or normal capacity acceptance"]
    elif component == "kda-mixed":
        scenario = {"layer_index": 0, "query_start_loc": (0, 1, 513), "state_indices": (1, 2),
                    "state_pool_slots": 3, "has_initial_state": (True, False),
                    "conv_state_dtype": "bf16", "recurrent_state_dtype": "fp32", "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_triton_mixed_workload(model, **scenario)
        scope = "TP1_Triton_KDA_one_initialized_decode_one_fresh512_shared_chunk_caller_source_candidate"
        limitations = [
            "declared state indices/flags and initial contents, not native metadata or numeric cache execution",
            "any prefill selects three conv_fn calls and chunk for all non-speculative rows, including decode1",
            "decode reads prior3 convolution values and rereads2 shifted values per channel; fresh has no old reads",
            "gather reads one FP32 state row and writes two dense rows, fresh zeroed; scatter writes both pools",
            "nine chunk tiles include one short decode segment; internal masked work/traffic remain unpriced",
            "no pure-decode recurrence substitution, speculation, prefix checkpoint or actual backend/provider binding"]
    elif component == "nope-mixed":
        scenario = {"layer_index": 3, "query_start_loc": (0, 1, 513), "decode_kv_len": 513,
                    "slot_mapping": (512, *range(1024, 1536)), "cache_capacity_tokens": 2048,
                    "cache_dtype": "bf16", "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_nope_mixed_workload(model, **scenario)
        scope = "TP1_BF16_NoPE_MLA_one_decode513_one_fresh512_shared_frontend_source_candidate"
        limitations = [
            "declared decode-first slots and length, no native metadata/Scheduler or cached tensor execution",
            "T513 merged frontend/norm/QB and output gate/O-projection run once across both requests",
            "fresh prefill executes before decode absorption despite decode-first token order",
            "KV-B covers prefill512; absorption and W_UV cover decode1 only",
            "two fused epilogues insert disjoint cache slices; attention writes disjoint output slices without copy",
            "canonical analytical MLA selectors are not actual0.30 backend/provider execution",
            "compiled BF16 sigmoid output gate retains an unpriced materialization/compiler boundary"]
    elif component == "nope-prefill":
        scenario = {"layer_index": 3, "slot_mapping": tuple(range(512)), "cache_capacity_tokens": 512,
                    "cache_dtype": "bf16", "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_nope_fresh_prefill_workload(model, **scenario)
        scope = "single_fresh_NoPE_MLA_T512_BF16_TP1_source_branch"
        limitations = [
            "merged frontend proven for T>=512; no short-input auxiliary stream dispatch claim",
            "one fresh sequence, no context chunks, MTP, DCP or graph padding",
            "canonical FA3 analytical selector reuse is not vLLM0.30 backend/provider execution",
            "compiled sigmoid gate preserves BF16 tensor boundary; intermediate storage remains unbound"]
    elif component == "native-latent-output":
        scenario = {"layer_index": 1, "rows": 8, "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_tp1_native_latent_output_workload(model, **scenario)
        scope = "TP1_native_MoERunner_output_tail_external_routed_latent_and_shared_output"
        limitations = [
            "external already combined BF16 routed latent and shared expert outputs; not a complete MoE call",
            "TP1 native runner only; no MegaMoE beta-add, TP tail tier, expert dispatch or routing assertion",
            "RMSNorm follows combined routed latent; ordinary up projection then distinct shared add",
            "all source-selected generic algorithms are analytical; actual backend/provider remains unbound"]
    elif component == "marlin-routed-output":
        scenario = {"layer_index": 1, "rows": 8, "pool_name": "main", "rank": 0}
        step_id = "source-k3-marlin-output"
        routing = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0).resolve(
            step_id=step_id, layer_index=1, moe_invocation_id=f"{step_id}:layer-1:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=scenario["rows"], num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
        graph = vllm030_kimi_k3_marlin_routed_output_workload(model, routing=routing, **scenario)
        scope = "TP1_Marlin_W4A16_SiTU_experts_plus_native_latent_output_tail_source_candidate"
        limitations = [
            "explicit seeded routing assumption, not computed from checkpoint logits or observed expert IDs",
            "external BF16 down-projected latent, selected IDs/weights and BF16 shared MLP output",
            "source aggregate includes alignment/W13/SiTU/weighted_W2/topk_sum; not one physical launch",
            "MXFP4 active weight payload/scales and live tensor math do not establish physical padded/scratch traffic",
            "SiTU compiler services, alignment/scratch and actual H200 backend/provider remain unbound"]
    elif component in ("kda-fresh-prefill", "dense-kda-prefill-layer"):
        scenario = {"query_start_loc": (0, 128), "state_index": 1, "state_pool_slots": 2,
                    "has_initial_state": False, "conv_state_dtype": "bf16", "recurrent_state_dtype": "fp32",
                    "pool_name": "main", "rank": 0}
        if component == "kda-fresh-prefill":
            scenario["layer_index"] = 0
            graph = vllm030_kimi_k3_triton_fresh_prefill_workload(model, **scenario)
            scope = "TP1_T128_fresh_single_sequence_Triton_KDA_source_candidate"
        else:
            graph = vllm030_kimi_k3_dense_fresh_prefill_layer_workload(model, **scenario)
            scope = "TP1_T128_fresh_KDA_dense_decoder_layer0_full_source_candidate"
        limitations = [
            "explicit Triton fresh full64-chunk candidate; no auto backend or model execution assertion",
            "one positive declared state slot, BF16 conv/FP32 recurrent cache; no prefix/MTP/RecoverSSM",
            "three Q/K/V convolutions, FP32 dense initial zeros, chunk result pool scatter and BF16 core copy retained",
            "chunk_h is BF16, final recurrent state is FP32; output overwrites post-conv V only after its earlier uses",
            "chunk internal buffer storage is not a complete arithmetic/traffic census or an assigned latency",
            "source state/index roles and copy projection do not bind actual library/compiler launches"]
    elif component in ("native-marlin-moe", "nope-prefill-layer"):
        layer_index = 1 if component == "native-marlin-moe" else 3
        scenario = {"layer_index": layer_index, "rows": 512, "pool_name": "main", "rank": 0,
                    "shared_experts_stream_token_threshold": 256}
        if component == "nope-prefill-layer":
            scenario.update(slot_mapping=tuple(range(scenario["rows"])), cache_capacity_tokens=512, cache_dtype="bf16")
        step_id = f"source-k3-{component}"
        routing = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0).resolve(
            step_id=step_id, layer_index=layer_index, moe_invocation_id=f"{step_id}:layer-{layer_index}:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=scenario["rows"], num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
        if component == "native-marlin-moe":
            graph = vllm030_kimi_k3_native_marlin_moe_workload(model, routing=routing, **scenario)
            scope = "TP1_T512_native_Marlin_MoE_original_hidden_to_output_source_candidate"
        else:
            graph = vllm030_kimi_k3_nope_prefill_layer_workload(
                model, routing=routing, **{key: value for key, value in scenario.items() if key != "rows"})
            scope = "TP1_T512_fresh_NoPE_decoder_layer3_full_source_candidate"
        limitations = [
            "explicit Marlin standard modular/no-DP-EP source program, not auto backend dispatch",
            "T512 exceeds model down-stream threshold256 and declared shared-stream threshold256",
            "original hidden input shared by router/down/shared gate; gate emits FP32 logits from BF16 A/B",
            "canonical seeded histogram remains an explicit assumption, not derived checkpoint logits",
            "shared SiTU, grouped top16 and expert aggregate remain unpriced; no generic activation substitution",
            "source call/aggregate counts are not a census of actual GPU launches",
            "NoPE layer case uses externally supplied AttnRes prefix/delta/block state and fresh BF16 cache slots"]
    else:
        raise ValueError("unknown K3 source component chain")
    if component in ("fresh-prefill-greedy-step", "single-decode-greedy-step", "mixed-greedy-step", "context-prefill-greedy-step", "context-continuation-greedy-step"):
        limitations = [line.replace("excludes sampler", "explicit plain greedy sampler appended below")
                       .replace("sampler, metadata/index producers", "metadata/index producers") for line in limitations]
        scenario["sampling_source"] = "vllm_0_30_plain_greedy_bf16"
        graph = vllm030_kimi_k3_append_plain_greedy_sampler(model, graph, sampling_source=scenario["sampling_source"])
        scope += "__plain_greedy_no_processors_penalties_masks_logprobs_or_thinking_budget"
        limitations.append("sampler profile is declared; no sampled values or native SamplingMetadata execution; argmax compiler/ID services unpriced")
    plan = lower_workload(graph, deployment, {"nvidia_h200": hardware})
    calls = []
    for call in plan.calls:
        spec = call.kernel_spec
        record = {"call_id": call.call_id, "dependencies": call.dependencies,
                  "kind": spec.kind.value, "variant": spec.variant, "dtype": spec.dtype,
                  "shape": dict(spec.shape), "attributes": dict(spec.attributes),
                  "execution_support": spec.execution_support, "actual_provider_bound": False,
                  "analytical_algorithm": None if spec.launch is None else spec.launch.algorithm}
        if spec.variant == "kimi_k3_nope_prefill":
            record["source_work"] = kimi_k3_nope_prefill_logical_work(spec)
        elif spec.variant == "kimi_k3_nope_decode":
            record["source_work"] = kimi_k3_nope_decode_logical_work(spec)
        elif spec.variant == "kimi_k3_nope_mixed":
            record["source_work"] = kimi_k3_nope_mixed_logical_work(spec)
        elif spec.variant == "kimi_k3_nope_context":
            record["source_work"] = kimi_k3_nope_context_logical_work(spec)
        elif spec.variant == "kimi_k3_nope_context_attention":
            record["source_work"] = kimi_k3_nope_context_attention_logical_work(spec)
        elif spec.variant == "kimi_k3_nope_bmm":
            record["source_work"] = kimi_k3_nope_bmm_logical_work(spec)
        elif spec.variant == "kimi_k3_marlin_experts":
            record["source_work"] = kimi_k3_marlin_expert_logical_work(spec)
        elif spec.variant == "kimi_k3_moe_route":
            record["source_work"] = kimi_k3_moe_route_logical_work(spec)
        elif spec.variant == "kimi_k3_situ":
            record["source_work"] = kimi_k3_situ_logical_work(spec)
        elif spec.variant == "kimi_k3_attn_res":
            record["source_work"] = kimi_k3_attention_residual_logical_work(spec)
        elif spec.variant == "kimi_k3_kda_prefill":
            record["source_work"] = kimi_k3_kda_prefill_logical_work(spec)
        elif spec.variant == "kimi_k3_kda_context_prefill":
            record["source_work"] = kimi_k3_kda_context_prefill_logical_work(spec)
        elif spec.variant == "kimi_k3_kda_mixed":
            record["source_work"] = kimi_k3_kda_mixed_logical_work(spec)
        elif spec.variant == "kimi_k3_kda_decode":
            record["source_work"] = kimi_k3_kda_decode_logical_work(spec)
        elif spec.variant == "vllm_logits_row_select":
            record["source_work"] = vllm_logits_row_selection_logical_work(spec)
        elif spec.variant == "vllm_embedding_id_widen":
            record["source_work"] = vllm_embedding_id_widen_logical_work(spec)
        elif spec.variant == "vllm_plain_greedy_sample":
            record["source_work"] = vllm_plain_greedy_sampling_logical_work(spec)
        elif spec.variant == "vllm_logits_fp32_cast":
            record["source_work"] = vllm_logits_fp32_cast_logical_work(spec)
        if (spec.variant in ("kimi_k3_marlin_experts", "kimi_k3_moe_route", "kimi_k3_situ", "kimi_k3_nope_context_attention")
                or (spec.variant == "kimi_k3_nope_context" and spec.attributes["stage"] in ("merge", "context_partial_merge", "output_gate"))
                or (spec.variant in ("kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill") and spec.attributes["stage"] == "chunk")
                or (spec.variant in ("kimi_k3_nope_prefill", "kimi_k3_nope_decode", "kimi_k3_nope_mixed")
                    and spec.attributes["stage"] == "output_gate")
                or (spec.variant == "vllm_plain_greedy_sample" and spec.attributes["stage"] == "argmax")):
            try:
                predict(spec, hardware, method="roofline")
            except NotImplementedError as error:
                record.update(status="unpriced", roofline_seconds=None, reason=str(error))
            else:
                raise RuntimeError("K3 unbound SiTU/gate source cannot silently inherit a generic activation price")
        else:
            record.update(status="priced", roofline_seconds=predict(spec, hardware, method="roofline").t_total_s)
        calls.append(record)
    result = {
        "snapshot": _snapshot(), "source_identity": identity,
        "model_id": model.model_id, "model_revision": model.model_revision,
        "config_sha256": model.config_sha256,
        "scope": scope,
        "source_nodes": tuple({"node_id": node.node_id, "kind": node.kind.value,
                               "variant": node.variant, "dtype": node.dtype,
                               "shape": dict(node.shape), "attributes": dict(node.attributes)}
                              for node in graph.nodes),
        "source_edges": tuple({"source": edge.source, "destination": edge.destination, "kind": edge.kind}
                              for edge in graph.edges),
        "scenario": scenario, "calls": calls, "component_count": len(calls),
        "priced_calls": sum(call["status"] == "priced" for call in calls),
        "complete_moe_body_source_candidate": component in ("native-marlin-moe", "nope-prefill-layer"),
        "complete_decoder_layer_source_candidate": component in ("nope-prefill-layer", "dense-kda-prefill-layer"),
        "complete_decoder_stack_source_candidate": component in (
            "fresh-prefill-decoder-stack", "fresh-prefill-forward-logits", "single-decode-forward-logits",
            "fresh-prefill-greedy-step", "single-decode-greedy-step", "mixed-greedy-step", "context-prefill-greedy-step", "context-continuation-greedy-step"),
        "complete_forward_logits_source_candidate": component in (
            "fresh-prefill-forward-logits", "single-decode-forward-logits", "fresh-prefill-greedy-step", "single-decode-greedy-step", "mixed-greedy-step", "context-prefill-greedy-step", "context-continuation-greedy-step"),
        "complete_plain_greedy_model_source_candidate": component in ("fresh-prefill-greedy-step", "single-decode-greedy-step", "mixed-greedy-step", "context-prefill-greedy-step", "context-continuation-greedy-step"),
        "unpriced_calls": sum(call["status"] == "unpriced" for call in calls),
        "complete_model_workload": False, "runtime_accepted": False,
        "normal_capacity_accepted": False, "independent_reference_accepted": False,
        "limitations": limitations + [
            "full K3 prefill/decode/mixed hybrid decoder and normal Serving bridge remain incomplete"],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_value(result), indent=2) + "\n")
    return result


def write_kimi_k3_source_components(model_path, source_root, output):
    """Evaluate independent source components, never a complete K3 graph."""
    from src.serving.backend.vllm_scheduler_process import vllm030_source_identity
    from src.serving.workload.kimi_k3 import (
        vllm030_kimi_k3_attention_residual_node, vllm030_kimi_k3_ordinary_mlp_workload,
        vllm030_kimi_k3_triton_decode_workload, vllm030_kimi_k3_dense_decode_layer_workload)
    from src.serving.workload.shape_inference import infer_gemm_dimensions
    from src.serving.workload.workload_graph import WorkloadGraph
    from src.kernel.predictors.vector import (
        kimi_k3_attention_residual_logical_work, kimi_k3_situ_logical_work, kimi_k3_kda_decode_logical_work)

    model = load_model_config(model_path)
    if model.attention_layout != "hybrid_kda_mla":
        raise ValueError("K3 source components require an exact hybrid config")
    identity = vllm030_source_identity(Path(source_root))
    hardware = HWSpec(H200_HARDWARE_CONFIG)
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (WorkerPoolConfig("main", "colocated", "nvidia_h200"),)),
        num_layers=model.num_hidden_layers, num_experts=model.num_experts)
    positions = (
        "in_proj_qkvgfab", "f_b_proj", "fused_qkv_a_g_proj", "q_b_proj", "kv_b_proj", "o_proj",
        "gate_up_proj", "down_proj", "shared_experts.gate_up_proj", "shared_experts.down_proj",
        "experts.w13_weight", "experts.w2_weight", "routed_expert_down_proj", "router", "lm_head")
    dimensions = []
    for tp in (1, 8):
        for position in positions:
            dimensions.append({"tp": tp, "position": position, "rows": 8,
                               "m_n_k": infer_gemm_dimensions(model, position, 8, tp_size=tp)})
        for tier in ((None,) if tp == 1 else ("ALLREDUCE_OVERLAP", "COLUMN_PARALLEL")):
            dimensions.append({"tp": tp, "position": "routed_expert_up_proj", "rows": 8,
                               "explicit_source_tail_tier": tier,
                               "m_n_k": infer_gemm_dimensions(model, "routed_expert_up_proj", 8,
                                                               tp_size=tp, kimi_k3_tail_tier=tier)})
    stages = [(stage, layer) for layer in range(model.num_hidden_layers)
              for stage in ("pre_attention", "post_attention")] + [("output", None)]
    components = []
    # These are independent operator evaluations with externally supplied
    # tensors. No edges or combined model latency are invented between them.
    for rows in (1, 8, 256):
        for stage, layer in stages:
            node = vllm030_kimi_k3_attention_residual_node(
                model, rows=rows, stage=stage, layer_index=layer, pool_name="main", rank=0)
            plan = lower_workload(WorkloadGraph([node]), deployment, {"nvidia_h200": hardware})
            if len(plan.calls) != 1:
                raise RuntimeError("K3 residual component must retain one source call")
            spec = plan.calls[0].kernel_spec
            work = kimi_k3_attention_residual_logical_work(spec)
            prediction = predict(spec, hardware, method="roofline")
            components.append({"stage": stage, "layer_index": layer, "rows": rows,
                               "shape": dict(spec.shape), "attributes": dict(spec.attributes),
                               "source_work": work, "roofline_seconds": prediction.t_total_s,
                               "execution_support": spec.execution_support, "launch": None})
    mlp_cases = []
    for rows in (1, 8, 256):
        for role, layer in (("dense", 0), ("shared", model.num_dense_layers)):
            graph = vllm030_kimi_k3_ordinary_mlp_workload(
                model, rows=rows, layer_index=layer, role=role, pool_name="main", rank=0)
            plan = lower_workload(graph, deployment, {"nvidia_h200": hardware})
            calls = []
            for call in plan.calls:
                spec = call.kernel_spec
                record = {"call_id": call.call_id, "dependencies": call.dependencies,
                          "kind": spec.kind.value, "variant": spec.variant,
                          "dtype": spec.dtype, "shape": dict(spec.shape),
                          "execution_support": spec.execution_support,
                          "launch_policy": "analytical_launch", "actual_provider_bound": False}
                if spec.variant == "kimi_k3_situ":
                    work = kimi_k3_situ_logical_work(spec)
                    try:
                        predict(spec, hardware, method="roofline")
                    except NotImplementedError as error:
                        record.update(status="unpriced", roofline_seconds=None, reason=str(error), source_work=work)
                    else:
                        raise RuntimeError("K3 source tanhf cannot be silently priced")
                else:
                    prediction = predict(spec, hardware, method="roofline")
                    record.update(status="priced", roofline_seconds=prediction.t_total_s)
                calls.append(record)
            mlp_cases.append({"role": role, "layer_index": layer, "rows": rows,
                              "source_branch": "ordinary_TP1_KimiMLP_not_provider_dispatch",
                              "calls": calls, "unpriced_calls": sum(call["status"] == "unpriced" for call in calls)})
    def source_calls(graph):
        plan = lower_workload(graph, deployment, {"nvidia_h200": hardware})
        records = []
        for call in plan.calls:
            spec = call.kernel_spec
            record = {"call_id": call.call_id, "dependencies": call.dependencies,
                      "kind": spec.kind.value, "variant": spec.variant, "dtype": spec.dtype,
                      "shape": dict(spec.shape), "attributes": dict(spec.attributes),
                      "execution_support": spec.execution_support, "actual_provider_bound": False}
            if spec.variant == "kimi_k3_kda_decode":
                record["source_work"] = kimi_k3_kda_decode_logical_work(spec)
            elif spec.variant == "kimi_k3_attn_res":
                record["source_work"] = kimi_k3_attention_residual_logical_work(spec)
            elif spec.variant == "kimi_k3_situ":
                record["source_work"] = kimi_k3_situ_logical_work(spec)
            if spec.variant == "kimi_k3_situ":
                try:
                    predict(spec, hardware, method="roofline")
                except NotImplementedError as error:
                    record.update(status="unpriced", roofline_seconds=None, reason=str(error))
                else:
                    raise RuntimeError("K3 SiTU cannot inherit KDA source pricing")
            else:
                prediction = predict(spec, hardware, method="roofline")
                record.update(status="priced", roofline_seconds=prediction.t_total_s)
            records.append(record)
        return records

    kda_cases, first_layer_cases = [], []
    for rows in (1, 8):
        scenario = {"state_indices": tuple(range(1, rows + 1)), "state_pool_slots": rows + 1,
                    "conv_state_dtype": "bf16", "recurrent_state_dtype": "fp32", "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_triton_decode_workload(model, layer_index=0, **scenario)
        calls = source_calls(graph)
        kda_cases.append({"rows": rows, "layer_index": 0, "scenario": scenario, "calls": calls,
                          "phase": "decode", "source_branch": "eager_triton_TP1_not_observed_dispatch"})
        graph = vllm030_kimi_k3_dense_decode_layer_workload(model, **scenario)
        calls = source_calls(graph)
        for phase in ("decode1", "decode2"):
            first_layer_cases.append({"rows": rows, "layer_index": 0, "phase": phase,
                                      "scenario": scenario, "calls": calls,
                                      "unpriced_calls": sum(call["status"] == "unpriced" for call in calls),
                                      "state_policy": "same declared positive slots across phases, no actual cache values or Scheduler trace",
                                      "full_model": False, "runtime_accepted": False})
    result = {
        "snapshot": _snapshot(), "source_identity": identity,
        "model_id": model.model_id, "model_revision": model.model_revision,
        "config_sha256": model.config_sha256,
        "scope": "K3 source components plus ordinary TP1 Triton KDA decode and first dense decode-layer chains",
        "projection_dimensions": dimensions, "attention_residual_components": components,
        "ordinary_mlp_cases": mlp_cases,
        "ordinary_kda_decode_cases": kda_cases, "first_dense_decode_layer_cases": first_layer_cases,
        "component_count": len(components), "complete_model_workload": False,
        "runtime_accepted": False, "normal_capacity_accepted": False,
        "independent_reference_accepted": False,
        "limitations": [
            "PP1/SP1 BF16 residual components use externally supplied source tensors",
            "no KDA prefill/chunk/mixed or NoPE gated MLA decoder DAG, latent SiTU MoE or normal Serving bridge",
            "projection dimensions precede backend expert packing and do not price latent beta=1 C reads",
            "declared latent tail tier is not inferred from tokens or bound to runtime dispatch",
            "source launch metadata and logical bytes are not observed compiler/provider/DRAM evidence"],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_value(result), indent=2) + "\n")
    return result


def _weight_residency_case(args):
    (model_path, tp, pp, vision_tensor_parallel_mode, model_execution_version,
     mtp_mode, constructor_buffer_limits, source_fp8_backend, source_moe_workspace_tokens) = args
    model = load_model_config(model_path)
    if constructor_buffer_limits is not None and model.config_sha256 not in GLM_BLOCK_FP8_CONFIG_SHA256S:
        raise ValueError("constructor buffer source limits currently require exact GLM block-FP8")
    if source_fp8_backend is not None and model.config_sha256 not in GLM_BLOCK_FP8_CONFIG_SHA256S:
        raise ValueError("post-load FP8 source backend currently requires exact GLM-5.3")
    if model.model_type != "glm_moe_dsa" and (
            model_execution_version is not None or mtp_mode is not None):
        raise ValueError("versioned MTP residency diagnostic currently requires GLM")
    capacity = HWSpec(H200_HARDWARE_CONFIG).resource_by_role("dram").size_bytes
    if model.model_type == "kimi_k2":
        if vision_tensor_parallel_mode is None:
            raise ValueError("Kimi outer diagnostic requires explicit vision tensor parallel mode")
        parameters = kimi_k25_outer_loader_payload_bytes(
            model, ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
            vision_tensor_parallel_mode=vision_tensor_parallel_mode)
        return {
            "payload_scope": "kimi_active_outer_registered_parameters",
            "tp": tp, "pp": pp, "physical_hbm_bytes": capacity,
            "vision_tensor_parallel_mode": vision_tensor_parallel_mode,
            "outer_loader_parameter_bytes_per_rank": parameters,
            "outer_constructor_parameter_payload_bytes_per_rank": sum(parameters.values()),
            "normal_kv_capacity_accepted": False,
            "actual_outer_loader_or_runtime_binding": False,
            "scope": "explicit enabled-tower/no-offload BF16 parameter constructor; no text/KV/reserve accounting",
            "limitations": [
                "raw vt_* config metadata does not override VisionConfig constructor defaults",
                "parent config replaces projector mm_hidden_size equal to vision width with text width",
                "CompressedTensorsConfig is ignored by the outer tower/projector",
                "source vision mode is conditional; actual language_model_only/vision_chunk skip, TP mode and offload not bound",
                "nonpersistent time/rotary buffers, text weights, post-load aliases, scratch/reserves and allocator excluded",
            ],
        }
    if vision_tensor_parallel_mode is not None:
        raise ValueError("explicit vision tensor parallel mode applies only to Kimi outer diagnostic")
    if model.model_type == "deepseek_v4":
        payload = dsv4_routed_loader_payload_bytes(
            model, ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp))
        packed = payload["layers"] * (
            payload["w13_packed_bytes_per_layer"] + payload["w2_packed_bytes_per_layer"])
        scales = payload["layers"] * (
            payload["w13_ue8m0_bytes_per_layer"] + payload["w2_ue8m0_bytes_per_layer"])
        other_parameters = dsv4_non_routed_loader_payload_bytes(
            model, ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
            attention_backend="flashmla")
        return {
            "tp": tp, "pp": pp, "physical_hbm_bytes": capacity,
            "routed_loader_payload_per_rank": payload,
            "routed_packed_bytes_per_rank": packed,
            "routed_loader_scale_bytes_per_rank": scales,
            "non_routed_loader_parameter_bytes_per_rank": other_parameters,
            "non_routed_loader_bytes_per_rank": sum(other_parameters.values()),
            "constructor_parameter_payload_bytes_per_rank": packed + scales + sum(other_parameters.values()),
            "attention_backend_source_scenario": "flashmla",
            "actual_attention_backend_bound": False,
            "packed_payload_exceeds_physical_capacity": packed > capacity,
            "normal_kv_capacity_accepted": False,
            "scope": "conditional V4 EP-off aligned Marlin/FlashMLA source constructor parameter payload only; not post-load runtime residency",
            "limitations": [
                "non-MegaMoE FusedMoEFactory with Marlin is a source scenario, not observed provider selection",
                "registered source parameters are counted; actual checkpoint/loader/storage are not bound",
                "post-load conversion/aliases, rotary/topk/broadcast/scratch/workspaces are excluded",
                "target-off only; DSpark draft identity/weights are not included",
                "reserve/profile/allocator/Scheduler and actual available-KV remain unbound",
            ],
        }
    if model.model_type != "glm_moe_dsa":
        raise NotImplementedError("this Goal weight diagnostic requires exact GLM or V4 config")
    if model.config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S:
        if model_execution_version != "vllm_0_30_0" or mtp_mode != "off":
            raise ValueError("GLM block-FP8 payload requires explicit vLLM 0.30 and target-off")
        parallel = ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp)
        registered = glm_block_fp8_loader_payload_bytes(
            model, parallel, include_mla_absorption=False)
        absorbed = glm_block_fp8_loader_payload_bytes(
            model, parallel, include_mla_absorption=True)
        additions = {name: size for name, size in absorbed.items() if name not in registered}
        buffers = (None if constructor_buffer_limits is None else
                   glm_block_fp8_constructor_buffer_bytes(
                       model, parallel, max_batched_tokens=constructor_buffer_limits[0],
                       max_requests=constructor_buffer_limits[1]))
        postload = (None if source_fp8_backend is None else
                    glm_block_fp8_postload_parameter_bytes(
                        model, parallel, hardware_id="nvidia_h200",
                        linear_backend="deep_gemm", routed_backend="deep_gemm",
                        include_mla_absorption=True))
        workspace = (None if source_moe_workspace_tokens is None else
                     glm_block_fp8_deepgemm_workspace_bytes(
                         model, parallel, hardware_id="nvidia_h200",
                         tokens=source_moe_workspace_tokens))
        call_allocations = (None if source_moe_workspace_tokens is None else
                            glm_block_fp8_deepgemm_call_allocation_bytes(
                                model, parallel, hardware_id="nvidia_h200",
                                tokens=source_moe_workspace_tokens))
        return {
            "payload_scope": "glm_fp8_source_parameters",
            "tp": tp, "pp": pp, "physical_hbm_bytes": capacity,
            "registered_parameter_bytes_per_rank": registered,
            "registered_parameter_payload_bytes_per_rank": sum(registered.values()),
            "mla_absorption_parameter_bytes_per_rank": additions,
            "source_payload_with_mla_absorption_bytes_per_rank": sum(absorbed.values()),
            "constructor_buffer_source_limits": constructor_buffer_limits,
            "constructor_buffer_bytes_per_rank": buffers,
            "constructor_buffer_total_bytes_per_rank": None if buffers is None else sum(buffers.values()),
            "source_fp8_postload_backend": source_fp8_backend,
            "postload_parameter_bytes_per_rank": postload,
            "postload_parameter_payload_bytes_per_rank": None if postload is None else sum(postload.values()),
            "postload_source_binding": None if postload is None else GLM_SM90_DEEP_GEMM_STORAGE_SOURCE,
            "moe_workspace_source_request_bytes_per_rank": workspace,
            "moe_workspace_source_tokens": source_moe_workspace_tokens,
            "moe_call_allocation_bytes_per_rank": call_allocations,
            "moe_call_allocation_source": (None if call_allocations is None else {
                "source_binding_owner": "src.serving.backend.vllm_scheduler_process.vllm030_allocator_source_identity",
                "symbols": (
                    "fused_moe/modular_kernel.py:FusedMoEKernel.apply/_fused_experts",
                    "fused_moe/prepare_finalize/no_dp_ep.py:_quantize_input",
                    "fused_moe/utils.py:_fp8_quantize/count_expert_num_tokens",
                    "fused_moe/deep_gemm_utils.py:deepgemm_moe_permute",
                    "quantization/utils/fp8_utils.py:per_token_group_quant_fp8/silu_mul_per_token_group_quant_fp8_colmajor"),
                "scope": "declared unchanged vLLM0.30 source scenario, not loaded CUDA objects or provider selection",
            }),
            "moe_call_allocation_scope": (None if call_allocations is None else
                "unique tensor payloads outside the shared workspace for one declared call; "
                "sum is an allocation envelope, not a simultaneous live peak or total worker reserve"),
            "moe_workspace_source_request_scope": (None if workspace is None else
                "declared unchunked M=source-moe-workspace-tokens, BF16, no CPU expert counts, EP-off, default SM90 alignment, one lane/ubatch; shared by layers; not total worker workspace or activation peak"),
            "registered_payload_exceeds_physical_capacity": sum(registered.values()) > capacity,
            "model_execution_version": model_execution_version, "mtp_mode": "off",
            "normal_kv_capacity_accepted": False, "actual_loader_or_runtime_binding": False,
            "scope": "PP1 unreplicated embedding, unpadded FP8 constructors and ordinary CUDA/CP1 BF16 MLA absorption; source payload only",
            "limitations": [
                ("Backend FP8 weight/scale relayout, padding and persistent conversion objects remain unbound"
                 if postload is None else
                 "SM90 DeepGEMM registered parameter storage counted under explicit source profile; actual provider, other backend layouts and kernel persistent objects remain unbound"),
                ("Rotary/topk/index-group and other persistent buffers are excluded" if buffers is None else
                 "Explicit eager BF16 default-RoPE/topk/index-group source buffers counted; other backend/runtime persistent storage remains unbound"),
                "MTP, other activation/dispatch/library scratch, reserves, actual loading and allocator-ready residency are excluded",
                "Declared MLA absorption branch is not an observed provider or execution",
                "Normal capacity entry remains rejected; this is not available-KV evidence",
            ],
        }
    ranks = model_rank_residencies(
        model, ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
        worker_role="colocated", include_mtp=mtp_mode != "off",
        model_execution_version=model_execution_version)
    return {
        "payload_scope": "glm_parameter_source_projection",
        "tp": tp, "pp": pp, "physical_hbm_bytes": capacity,
        "maximum_owner_weight_bytes": max(rank.weight_resident_bytes for rank in ranks),
        "rank_residencies": [asdict(rank) for rank in ranks],
        "model_execution_version": model_execution_version,
        "mtp_mode": "native" if mtp_mode is None else mtp_mode,
        "normal_kv_capacity_accepted": False,
        "actual_loader_or_runtime_binding": False,
        "scope": ("GLM vLLM 0.30 BF16 parameter source projection with MTP vocabulary aliases; unreplicated embedding source scenario, not normal runtime capacity"
                  if model_execution_version == "vllm_0_30_0" else
                  "existing independent draft-vocabulary inventory; v0.30 aliases and runtime reserves not bound"),
    }


def write_vllm030_model_identity_probe(model_path, source_root, output, *, draft_path=None):
    """Bind official target/draft metadata through genuine guarded CPU APIs.

    ModelConfig inspects the native registry without constructing a model or
    loading weights. This is identity/mode evidence, not a provider, Scheduler,
    capacity, full source graph or runtime acceptance result.
    """
    import tempfile
    from src.serving.backend.vllm_scheduler_process import (
        _install_cpu_device_guard, _verify_no_accelerator_access,
        vllm030_allocator_source_identity,
    )
    from src.serving.workload.model_config import _load_model_json

    model = load_model_config(model_path)
    if any(not isinstance(getattr(model, name), str) or not getattr(model, name) for name in (
            "model_id", "model_revision", "config_sha256", "model_source_url")):
        raise ValueError("model identity probe requires registered exact checkpoint provenance")
    identity = vllm030_allocator_source_identity(Path(source_root))
    guard = _install_cpu_device_guard()
    from vllm.config import ModelConfig as NativeModelConfig, ParallelConfig as NativeParallelConfig, SpeculativeConfig
    from vllm.model_executor.models import ModelRegistry

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def registry_binding(native):
        if len(native.architectures) != 1:
            raise ValueError("identity probe requires one explicit native architecture")
        architecture = native.architectures[0]
        if architecture not in ModelRegistry.models:
            raise ValueError("identity probe rejects a Transformers fallback or unregistered architecture")
        entry = ModelRegistry.models[architecture]
        return {"architecture": architecture, "module": entry.module_name, "class": entry.class_name,
                "inspected_architecture": native._model_info.architecture}

    with tempfile.TemporaryDirectory(prefix="model-identity-", dir=output.parent) as directory:
        target_dir = Path(directory) / "target"
        target_dir.mkdir()
        target_dir.joinpath("config.json").write_bytes(Path(model_path).read_bytes())
        target = NativeModelConfig(model=str(target_dir), skip_tokenizer_init=True, enforce_eager=True,
                                   dtype="bfloat16", max_model_len=128, runner="generate")
        target_binding = registry_binding(target)
        layers = target.hf_text_config.num_nextn_predict_layers
        if layers != model.num_mtp_layers:
            raise ValueError("actual native MTP layer count disagrees with the exact target config")
        native_on = None
        if layers == 0:
            if draft_path is not None:
                raise ValueError("zero-MTP target identity probe does not accept an invented draft")
        else:
            if model.model_type == "glm_moe_dsa":
                if draft_path is not None:
                    raise ValueError("GLM native MTP uses the target checkpoint configuration")
                method, draft_dir, tokens = "mtp", target_dir, 1
                draft_identity = {name: getattr(model, name) for name in (
                    "model_id", "model_revision", "config_sha256", "model_source_url")}
            elif model.model_type == "deepseek_v4":
                if draft_path is None:
                    raise ValueError("V4 native mode identity requires the exact official DSpark config")
                raw, sha, source_sha, canonical_sha, draft_id, revision, url, draft_scope = _load_model_json(str(draft_path))
                if (draft_id != model.model_id + "-DSpark" or draft_scope != "text"
                        or raw["model_type"] != "deepseek_v4"
                        or raw["architectures"] != ["DeepseekV4ForCausalLM"]):
                    raise ValueError("V4 DSpark draft identity does not match the exact target")
                method, tokens = "dspark", raw["dspark_block_size"]
                draft_dir = Path(directory) / "draft"
                draft_dir.mkdir()
                draft_dir.joinpath("config.json").write_bytes(Path(draft_path).read_bytes())
                draft_identity = dict(model_id=draft_id, model_revision=revision, config_sha256=sha,
                                      source_config_sha256=source_sha, canonical_config_sha256=canonical_sha,
                                      model_source_url=url)
            else:
                raise NotImplementedError("native mode identity is not bound for this target family")
            speculative = SpeculativeConfig(
                model=str(draft_dir), method=method, num_speculative_tokens=tokens,
                target_model_config=target,
                target_parallel_config=NativeParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1))
            draft = speculative.draft_model_config
            if draft is None:
                raise ValueError("native speculative config did not construct draft metadata")
            native_on = dict(method=speculative.method, registry=registry_binding(draft),
                             draft_identity=draft_identity, n_predict=draft.hf_config.n_predict,
                             num_speculative_tokens=speculative.num_speculative_tokens,
                             max_num_new_slots_for_drafting=speculative.max_num_new_slots_for_drafting,
                             parallel_drafting=speculative.parallel_drafting,
                             use_dspark=speculative.use_dspark())
        result = dict(snapshot=_snapshot(), source_identity=identity,
                      model_identity={name: getattr(model, name) for name in (
                          "model_id", "model_revision", "config_sha256", "model_source_url")},
                      target_registry=target_binding, configured_mtp_layers=layers,
                      applicable_modes=("off",) if layers == 0 else ("off", "native"),
                      native_mode=native_on, platform="guarded CPU metadata APIs; not H200 provider",
                      model_instantiated=False, weights_loaded=False, scheduler_executed=False,
                      normal_capacity_accepted=False, runtime_acceptance=False)
    _verify_no_accelerator_access(guard)
    output.write_text(json.dumps(result, indent=2) + "\n")


def write_model_weight_residency_diagnostic(
        model_path, layouts, output, *, vision_tensor_parallel_mode=None,
        model_execution_version=None, mtp_mode=None, constructor_buffer_limits=None,
        source_fp8_backend=None, source_moe_workspace_tokens=None):
    """Estimate which layouts can proceed to normal-capacity validation.

    Consume GLM residency, V4 parameters or explicit active Kimi outer parameters.
    No second formula, reserve assumption, debug block count or success gate.
    """
    if not layouts:
        raise ValueError("residency diagnostic requires at least one layout")
    if mtp_mode not in (None, "off", "native"):
        raise ValueError("unknown residency MTP mode")
    if source_fp8_backend not in (None, "deep_gemm_sm90"):
        raise ValueError("unsupported post-load FP8 source backend")
    if source_moe_workspace_tokens is not None:
        if type(source_moe_workspace_tokens) is not int or source_moe_workspace_tokens <= 0:
            raise ValueError("MoE workspace source tokens must be a positive integer")
        if source_fp8_backend != "deep_gemm_sm90":
            raise ValueError("MoE workspace source tokens require the explicit DeepGEMM SM90 profile")
    if constructor_buffer_limits is not None and (
            not isinstance(constructor_buffer_limits, tuple) or len(constructor_buffer_limits) != 2
            or any(type(value) is not int or value <= 0 for value in constructor_buffer_limits)):
        raise ValueError("constructor buffer limits require explicit positive batch-token and request limits")
    snapshot = _snapshot()
    with ProcessPoolExecutor(max_workers=len(layouts)) as executor:
        cases = list(executor.map(
            _weight_residency_case, [(model_path, tp, pp, vision_tensor_parallel_mode,
                                     model_execution_version, mtp_mode, constructor_buffer_limits,
                                     source_fp8_backend, source_moe_workspace_tokens)
                                    for tp, pp in layouts]))
    model_identity = load_model_config(model_path)
    payload = {"snapshot": snapshot, "model_config_path": model_path,
               "model_identity": {name: getattr(model_identity, name) for name in (
                   "model_id", "model_revision", "config_sha256", "model_source_url")},
               "status": "diagnostic_not_normal_capacity_acceptance", "cases": cases}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fp8_runner_source_links(steps):
    """Wire production source state ports; never infer a numeric runner state."""
    links, previous = [], {}
    for step in steps:
        cohorts = {}
        for node in step["nodes"]:
            attributes = node["attributes"]
            if "source_linear_io" not in attributes:
                continue
            history = attributes["source_linear_io"]["conditional_sdk_history"]
            rank = node["rank"]
            if rank not in cohorts:
                cohorts[rank] = []
            cohorts[rank].append(history)
        for rank in sorted(cohorts):
            history = cohorts[rank][0]
            if any(item != history for item in cohorts[rank]) or len(cohorts[rank]) != history["cohort_calls"]:
                raise ValueError("FP8 source state ports must describe the complete rank/model-call cohort")
            if history["invocation"] != step["phase"] or history["rank"] != rank:
                raise ValueError("FP8 source runner state cannot cross invocation or rank")
            links.append({"invocation": step["phase"], "rank": rank,
                "input_port": history["input_port"], "output_port": history["output_port"],
                "previous_output_port": previous[rank] if rank in previous else None,
                "initial_parameter": (None if rank in previous else
                    {"name": f"rank-{rank}:initial-sdk-H", "domain": history["entry_domain"], "value": None})})
            previous[rank] = history["output_port"]
    return links


def _compressor_state_source_links(steps):
    """Track request lineage, keeping physical admission/CoW events unknown."""
    writers = [node for step in steps for node in step["nodes"] if node["variant"] == "dsv4_compressor_state_cache"]
    if not writers:
        return []
    declared = ["source_cache_transition" in node["attributes"] for node in writers]
    if not all(declared):
        if any(declared):
            raise ValueError("compressor source lineage cannot omit some writer transitions")
        return None
    links, history, cohort = [], {}, None
    for step in steps:
        identities = [(node["rank"], node["attributes"]["source_dataflow"]["state_owner"]["layer"],
                       node["attributes"]["source_dataflow"]["state_owner"]["component"])
                      for node in step["nodes"] if node["variant"] == "dsv4_compressor_state_cache"]
        if len(set(identities)) != len(identities) or not identities:
            raise ValueError("compressor source lineage must cover each model cache exactly once")
        if cohort is None:
            cohort = set(identities)
        elif set(identities) != cohort:
            raise ValueError("compressor source lineage cannot drop or replace a model cache between steps")
        for node in sorted((node for node in step["nodes"] if node["variant"] == "dsv4_compressor_state_cache"), key=lambda node: node["node_id"]):
            transition = node["attributes"]["source_cache_transition"]
            cache = transition["cache"]
            if (transition["program"] != "vllm_0_30_compressor_partial_state_lineage" or transition["state_dtype"] != "fp32"
                    or type(cache["rank"]) is not int or cache["rank"] != node["rank"]
                    or type(cache["layer"]) is not int or cache["layer"] < 0
                    or node["dtype"] != "fp32" or transition["compress_ratio"] not in (4, 128)
                    or transition["compress_ratio"] != node["attributes"]["source_dataflow"]["compress_ratio"]
                    or cache != {"rank": node["rank"], **node["attributes"]["source_dataflow"]["state_owner"]}
                    or transition["input_port"] != node["node_id"] + ":state-entry"
                    or transition["output_port"] != node["node_id"] + ":state-exit"
                    or not node["node_id"].startswith(step["phase"] + ":")
                    or transition["physical_table_transition"] is not None or transition["prefix_cache_hit_observation"] is not None
                    or transition["actual_copy_completion_and_retained_content_bound"]):
                raise ValueError("compressor source lineage requires typed source ports, not unobserved physical/copy claims")
            updates = transition["request_updates"]
            ids = [update["request_id"] for update in updates]
            if len(set(ids)) != len(ids) or not ids or any(not isinstance(identity, str) or not identity for identity in ids):
                raise ValueError("compressor source lineage requires unique nonempty model requests")
            cursor = 0
            for update in updates:
                old, new = update["input_prefix"], update["output_prefix"]
                if (type(old) is not int or type(new) is not int or not 0 <= old < new
                        or update["write_positions"] != [old, new] or update["model_token_rows"] != [cursor, cursor + new - old]):
                    raise ValueError("compressor source lineage request spans do not cover its new model rows")
                cursor += new - old
            if cursor != node["shape"]["seq_len"]:
                raise ValueError("compressor source lineage must cover every writer row")
            identity = cache["rank"], cache["layer"], cache["component"]
            if identity not in history:
                history[identity] = {"exit": None, "requests": {}}
            prior = history[identity]
            requests = []
            for update in updates:
                request_id = update["request_id"]
                if request_id in prior["requests"]:
                    prefix = prior["requests"][request_id]
                    if prefix != update["input_prefix"]:
                        raise ValueError("compressor source lineage cannot skip or replay request prefixes")
                    initial = None
                else:
                    prefix = None
                    initial = {"prefix": [0, update["input_prefix"]], "values": None,
                        "prefix_cache_hit_observation": None, "admission_copy_transition": None}
                requests.append({"request_id": request_id, "previous_output_prefix": prefix,
                    "input_prefix": update["input_prefix"], "output_prefix": update["output_prefix"], "initial_context": initial})
                prior["requests"][request_id] = update["output_prefix"]
            links.append({"invocation": step["phase"], "source_node_id": node["node_id"], "cache": cache,
                "input_port": transition["input_port"], "output_port": transition["output_port"],
                "previous_output_port": prior["exit"], "requests": requests,
                "between_step_physical_transition": None, "retained_content_or_copy_completion_proven": False})
            prior["exit"] = transition["output_port"]
    return links


def normalize_model_operator_probe_source(model_path, target_sampling_source):
    """Select an existing logical scenario from an explicit versioned sampler.

    Model identity alone does not select a backend. Constructor history,
    phase IDs, topology and typed native inputs remain separately mandatory
    at the existing source-input and shared-forward boundaries. Sampler-less
    historical dense probes keep their LM-head-only default unchanged.
    """
    model = load_model_config(model_path)
    source = None
    if (model.model_type in ("qwen3", "llama") and not model.num_experts
            and model.compute_dtype == model.weight_dtype
            and target_sampling_source in (("vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16")
                if model.compute_dtype == "bf16" else ("vllm_0_30_plain_greedy_fp16",))
            and (model.compute_dtype == "bf16" or (model.compute_dtype == "fp16" and model.model_type == "qwen3"))):
        source = "cuda_eager_inplace"
    elif (model.model_type == "glm_moe_dsa" and model.compute_dtype == "bf16"
            and target_sampling_source in ("vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16",
                "vllm_0_30_rejection_raw_logits_bf16")):
        if model.weight_dtype == "bf16":
            source = "vllm_0_30_v32_bf16_sync_explicit_allreduce"
        elif model.weight_dtype == "fp8":
            source = "vllm_0_30_v32_fp8_sync_explicit_allreduce"
    elif (model.model_type == "kimi_k2" and model.compute_dtype == "bf16"
            and model.weight_dtype == "mixed_int4_bf16"
            and target_sampling_source in ("vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16")):
        source = "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
    elif (model.model_type == "deepseek_v3" and model.compute_dtype == "bf16" and model.weight_dtype == "fp8"
            and target_sampling_source == "vllm_0_27_1_plain_greedy_bf16"):
        source = "vllm_0_27_1_deepseek_v3_sync_fp8"
    if source is None:
        return None, None
    return source, {
        "entrypoint": "static_probe_cli_versioned_sampler",
        "source_scenario": source,
        "target_sampling_source": target_sampling_source,
        "source_projection_only": True,
        "actual_provider_runtime_bound": False,
        "token_values_inferred": False,
    }


def build_model_operator_probe_source_inputs(*, symbolic_forward_source, embedding_token_ids_by_phase,
        source_max_model_len, source_max_batched_tokens, target_sampling_source,
        workspace_initial_capacity_bytes=None, glm_indexer_selection_source=None, glm_prefill_attention_source=None,
        source_max_requests=None, mtp_draft_sampling_source=None, model_source_inputs_by_phase=None):
    """Normalize caller-declared target/native source inputs for the shared probe.

    No token values, constructor limits, workspace history, source backend or
    cardinality declaration is inferred from a model/config or a duration.
    The existing RuntimeInputs boundary owns their complete validation.
    """
    glm_sources = ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
    dsv3_source = symbolic_forward_source == "vllm_0_27_1_deepseek_v3_sync_fp8"
    if symbolic_forward_source == "cuda_eager_inplace":
        if (target_sampling_source not in (None, "vllm_0_30_plain_sampler_bf16", "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_greedy_fp16")
                or not isinstance(embedding_token_ids_by_phase, Mapping) or not embedding_token_ids_by_phase):
            raise ValueError("dense source probe requires explicit phase input ports and LM-head-only,0.30 FP32 cast or plain greedy source")
        if any(value is not None for value in (source_max_model_len, source_max_batched_tokens,
                workspace_initial_capacity_bytes, glm_indexer_selection_source, glm_prefill_attention_source,
                source_max_requests, mtp_draft_sampling_source, model_source_inputs_by_phase)):
            raise ValueError("dense source probe does not consume GLM/Kimi constructor or native draft inputs")
        return {phase: RuntimeInputs(ExpertRoutingInput.not_applicable(), "paged_varlen", "uniform",
            model_execution_version="vllm_0_30_0", symbolic_forward_source=symbolic_forward_source,
            target_sampling_source=target_sampling_source, embedding_token_ids=values)
            for phase, values in embedding_token_ids_by_phase.items()}
    if symbolic_forward_source not in (*glm_sources, "vllm_0_30_k2_6_int4_sync_explicit_allreduce",
            "vllm_0_27_1_deepseek_v3_sync_fp8"):
        raise ValueError("ordinary source probe requires its explicit GLM/K2.6/DeepSeek-V3 forward source")
    if not isinstance(embedding_token_ids_by_phase, Mapping) or not embedding_token_ids_by_phase:
        raise ValueError("ordinary source probe requires explicit phase token IDs")
    if source_max_model_len is None or source_max_batched_tokens is None:
        raise ValueError("ordinary source probe requires explicit context and batch constructor limits")
    native = model_source_inputs_by_phase is not None
    if native:
        if (symbolic_forward_source not in glm_sources or not isinstance(model_source_inputs_by_phase, Mapping)
                or set(model_source_inputs_by_phase) != set(embedding_token_ids_by_phase)
                or any(not isinstance(state, V32PaddedMTPInputs) for state in model_source_inputs_by_phase.values())
                or mtp_draft_sampling_source != "vllm_0_30_legacy_full_logits_greedy"
                or type(source_max_requests) is not int or source_max_requests < 1
                or glm_prefill_attention_source != "force_sparse_mqa"):
            raise ValueError("GLM native source requires typed inputs for every phase, full-logits greedy, request capacity and sparse prefill source")
        if target_sampling_source != "vllm_0_30_rejection_raw_logits_bf16":
            raise ValueError("GLM native source requires its raw-logits rejection target chain")
    elif source_max_requests is not None or mtp_draft_sampling_source is not None:
        raise ValueError("native source fields require explicit per-phase V32 inputs")
    elif target_sampling_source not in ((None, "vllm_0_27_1_plain_greedy_bf16") if dsv3_source else
            (None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16")):
        raise ValueError("ordinary source probe requires logits-only, a versioned FP32 cast or an explicit plain greedy target chain")
    if symbolic_forward_source in glm_sources:
        if workspace_initial_capacity_bytes is None or glm_indexer_selection_source is None:
            raise ValueError("GLM source probe requires explicit rank workspace history and indexer cardinality source")
    elif any(value is not None for value in (workspace_initial_capacity_bytes,
            glm_indexer_selection_source, glm_prefill_attention_source)):
        raise ValueError("K2.6/DeepSeek-V3 source probe does not consume GLM workspace/indexer options")
    return {phase: RuntimeInputs(
        ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0),
        "mla", "uniform", model_execution_version="vllm_0_27_1" if dsv3_source else "vllm_0_30_0",
        symbolic_forward_source=symbolic_forward_source, target_sampling_source=target_sampling_source,
        symbolic_source_max_model_len=source_max_model_len, symbolic_source_max_batched_tokens=source_max_batched_tokens,
        symbolic_source_max_requests=source_max_requests, mtp_draft_sampling_source=mtp_draft_sampling_source,
        model_source_inputs=None if not native else model_source_inputs_by_phase[phase],
        symbolic_workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,
        glm_indexer_selection_source=glm_indexer_selection_source, glm_prefill_attention_source=glm_prefill_attention_source,
        embedding_token_ids=values) for phase, values in embedding_token_ids_by_phase.items()}


def _operator_probe_phase_requests(prefill_tokens, mtp_mode, *, decode_prefix=False):
    """One declared request fixture, shared by source inputs and execution."""
    if mtp_mode == "native":
        def request(request_id, phase, input_tokens, output_tokens, computed, accepted, verified):
            return ScheduledRequest(
                request_id, phase, input_tokens, output_tokens, computed,
                raw_model_output_tokens=accepted + 1,
                mtp_proposed_tokens=verified, mtp_verified_tokens=verified,
                mtp_accepted_tokens=accepted, mtp_proposer_tokens=2,
                mtp_next_proposed_tokens=2, mtp_proposer_mode="padded",
                mtp_acceptance_source_kind="explicit_assumption",
                mtp_acceptance_source_id="static_explicit_accept_reject_array")

        return (
            ("prefill", (request("request", "prefill", prefill_tokens, 0, 0, 0, 0),)),
            ("decode1", (request("request", "decode", 0, 2, prefill_tokens, 1, 1),)),
            ("decode2", (request("request", "decode", 0, 2, prefill_tokens + 2, 0, 1),)),
            ("mixed", (request("new-request", "prefill", prefill_tokens, 0, 0, 0, 0),
                       request("request", "decode", 0, 2, prefill_tokens + 3, 0, 1))),
        )
    if mtp_mode == "off":
        mixed = (ScheduledRequest("new-request", "prefill", prefill_tokens, 0, 0),
            ScheduledRequest("request", "decode", 0, 1, prefill_tokens + 2))
        return (
            ("prefill", (ScheduledRequest("request", "prefill", prefill_tokens, 0, 0),)),
            ("decode1", (ScheduledRequest("request", "decode", 0, 1, prefill_tokens),)),
            ("decode2", (ScheduledRequest("request", "decode", 0, 1, prefill_tokens + 1),)),
            ("mixed", tuple(sorted(mixed, key=lambda request: request.scheduled_tokens > 1)) if decode_prefix else mixed),
        )
    raise ValueError("unknown static MTP mode")


def _declared_glm_padded_probe_state(step, target_ids, *, source_id):
    """Independent IDs in incoming-request and source-packed row orders."""
    draft_counts = {request.mtp_proposer_tokens for request in step.requests}
    if len(draft_counts) != 1 or next(iter(draft_counts)) < 1:
        raise ValueError("synthetic padded probe needs one explicit positive proposer depth")
    depth = next(iter(draft_counts))
    packed, packed_ids, _ = vllm_0_30_source_model_inputs(step, embedding_token_ids=target_ids,
        prompt_lengths=None, num_speculative_tokens=depth)
    first_draft = [*packed_ids[1:], 97]
    cursor = 0
    for index, request in enumerate(packed.requests):
        retained = cursor + request.scheduled_tokens - request.mtp_verified_tokens + request.mtp_accepted_tokens - 1
        first_draft[retained] = 59 + index
        cursor += request.scheduled_tokens
    return V32PaddedMTPInputs(request_ids=tuple(request.request_id for request in step.requests),
        source_kind="declared_synthetic", source_id=source_id,
        draft_input_token_ids=(tuple(first_draft), *(tuple(127 + draft + index for index in range(len(step.requests)))
            for draft in range(depth - 1))),
        discarded_rows=(False,) * len(step.requests), block_size_tokens=16,
        parallel_drafting=False, use_local_argmax_reduction=False, probabilistic=False)


def _default_glm_native_probe_source(model, parallel, phase_requests, *, sampler, draft_sampler):
    """Declare standalone padded phases, never sampled IDs or runtime state."""
    if (model.model_type != "glm_moe_dsa" or parallel.tensor_parallel_size not in (1, 8)
            or parallel.pipeline_parallel_size != 1 or parallel.enable_expert_parallel):
        return None
    if sampler not in (None, "vllm_0_30_rejection_raw_logits_bf16"):
        raise ValueError("default GLM native fixture requires its raw-logits rejection source")
    if draft_sampler not in (None, "vllm_0_30_legacy_full_logits_greedy"):
        raise ValueError("default GLM native fixture requires its full-logits greedy draft source")
    ports = {phase: tuple(range(1, 1 + sum(request.scheduled_tokens for request in requests)))
        for phase, requests in phase_requests}
    states = {phase: _declared_glm_padded_probe_state(ScheduleStep(phase, "main", requests), ports[phase],
        source_id="default-static-independent-padded-input-scenario") for phase, requests in phase_requests}
    # Include every physical recurrent pass even for requests whose next
    # proposals will not be consumed. This is a declared fixture cap.
    context = max(request.computed_tokens + request.scheduled_tokens + len(states[phase].draft_input_token_ids) - 1
        for phase, requests in phase_requests for request in requests)
    batch = max(sum(request.scheduled_tokens for request in requests) for _, requests in phase_requests)
    requests_cap = max(len(requests) for _, requests in phase_requests)
    source = "vllm_0_30_v32_" + model.weight_dtype + "_sync_explicit_allreduce"
    inputs = build_model_operator_probe_source_inputs(symbolic_forward_source=source,
        embedding_token_ids_by_phase=ports, source_max_model_len=context, source_max_batched_tokens=batch,
        source_max_requests=requests_cap, target_sampling_source="vllm_0_30_rejection_raw_logits_bf16",
        workspace_initial_capacity_bytes=(0,) * parallel.worker_count,
        glm_indexer_selection_source="declared_causal_topk", glm_prefill_attention_source="force_sparse_mqa",
        mtp_draft_sampling_source="vllm_0_30_legacy_full_logits_greedy", model_source_inputs_by_phase=states)
    receipt = {"entrypoint": "default_static_glm_native_fixture", "source_scenario": source,
        "synthetic_fixture": True, "historical_default_equivalence_claimed": False,
        "phase_scope": "independent standalone phases, not linked sampled outputs or runtime buffer history",
        "token_input_scope": "declared sequential target IDs; packed global shift plus independent retained/recurrent draft IDs",
        "source_constructor_inputs": {"max_model_len": context, "max_batched_tokens": batch,
            "max_requests": requests_cap, "workspace_initial_capacity_bytes": (0,) * parallel.worker_count},
        "target_sampling_source": "vllm_0_30_rejection_raw_logits_bf16",
        "mtp_draft_sampling_source": "vllm_0_30_legacy_full_logits_greedy",
        "requested_target_sampling_source": sampler, "requested_draft_sampling_source": draft_sampler,
        "sampler_scope": "explicit synthetic rejection/full-logits-greedy source scenario; not observed dispatch",
        "source_projection_only": True, "actual_provider_runtime_bound": False, "normal_capacity_accepted": False}
    return inputs, receipt


def _default_glm4_target_probe_source(model, parallel, phase_requests, *, token_ports, sampler):
    """Correct the old bare GLM4 probe's0.30/MLA mix with its0.11 source."""
    if (model.model_type != "glm4_moe" or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1 or parallel.enable_expert_parallel):
        return None
    if sampler not in (None, "vllm_0_11_plain_sampler_bf16", "vllm_0_11_plain_greedy_bf16"):
        raise ValueError("GLM4 probe requires an accurate0.11 output source")
    synthetic_ids = token_ports is None
    if synthetic_ids:
        token_ports = {phase: tuple(index % model.vocab_size for index in range(
            sum(request.scheduled_tokens for request in requests))) for phase, requests in phase_requests}
    routing = ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0)
    inputs = {phase: RuntimeInputs(routing, "paged_varlen", "uniform", model_execution_version="vllm_0_11_0",
        symbolic_forward_source="vllm_0_11_glm4_sync", target_sampling_source=sampler,
        embedding_token_ids=ids) for phase, ids in token_ports.items()}
    return inputs, {"entrypoint": "default_static_glm4_target_fixture", "synthetic_fixture": True,
        "source_scenario": "H200_vLLM0.11_BF16_GLM4_TP1_PP1_eager",
        "model_source_profile": "vllm_0_11_glm4_sync",
        "source_correction": "old generic probe declared vLLM0.30/MLA; GLM4 uses0.11 GQA paged-varlen source",
        "historical_default_equivalence_claimed": False,
        "token_input_scope": "declared sequential IDs modulo vocabulary per phase" if synthetic_ids else "caller-provided IDs",
        "target_sampling_source": sampler, "source_projection_only": True,
        "actual_provider_runtime_bound": False, "normal_capacity_accepted": False}


def _default_target_probe_source(model, parallel, phase_requests, *, token_ports, sampler,
        v4_routing_source, v4_max_model_len, v4_max_batched_tokens, v4_indexer_selection,
        v4_use_deep_gemm):
    """Declare a reproducible synthetic static fixture, never runtime facts.

    The old bare probe supplied no constructor history or token values.
    These explicit choices define the new virtual source scenario; they
    are not an equivalence claim about the historical default or a provider.
    """
    tp, pp = parallel.tensor_parallel_size, parallel.pipeline_parallel_size
    glm = model.model_type == "glm_moe_dsa"
    kimi = model.model_type == "kimi_k2" and model.model_id == "moonshotai/Kimi-K2.6"
    v3, v4 = model.model_type == "deepseek_v3", model.model_type == "deepseek_v4"
    if not (glm or kimi or v3 or v4):
        return None
    if ((glm or kimi or v3) and tp not in (1, 8) or v4 and tp not in (4, 8)
            or parallel.enable_expert_parallel and (not (glm or kimi) or tp != 8)
            or pp != 1 and not (glm and tp == 8 and not parallel.enable_expert_parallel)):
        return None
    expected_sampler = "vllm_0_27_1_plain_greedy_bf16" if v3 else "vllm_0_30_plain_greedy_bf16"
    if sampler not in ((None, expected_sampler) if v3 else
            (None, expected_sampler, "vllm_0_30_plain_sampler_bf16")):
        return None
    if v4 and v4_routing_source not in (None, "vllm_0_30_modular_cuda_top6_pure_text", "vllm_0_30_triton_top6_pure_text"):
        return None
    original_order = {phase: tuple(request.request_id for request in requests) for phase, requests in phase_requests}
    original_ports = token_ports
    if v3:
        repacked = []
        token_ports = None if original_ports is None else {}
        for phase, requests in phase_requests:
            ordered = tuple(sorted(requests, key=lambda request: request.scheduled_tokens > 1))
            repacked.append((phase, ordered))
            if original_ports is not None:
                cursor, parts = 0, {}
                for request in requests:
                    parts[request.request_id] = original_ports[phase][cursor:cursor + request.scheduled_tokens]
                    cursor += request.scheduled_tokens
                token_ports[phase] = tuple(token for request in ordered for token in parts[request.request_id])
        phase_requests = tuple(repacked)
    max_context = max(request.computed_tokens + request.scheduled_tokens
        for _, requests in phase_requests for request in requests)
    max_rows = max(sum(request.scheduled_tokens for request in requests) for _, requests in phase_requests)
    if token_ports is None:
        token_ports = {phase: tuple(index % model.vocab_size for index in range(
            sum(request.scheduled_tokens for request in requests))) for phase, requests in phase_requests}
    source = ("vllm_0_30_v32_bf16_sync_explicit_allreduce" if glm and model.weight_dtype == "bf16" else
        "vllm_0_30_v32_fp8_sync_explicit_allreduce" if glm else
        "vllm_0_30_k2_6_int4_sync_explicit_allreduce" if kimi else
        "vllm_0_27_1_deepseek_v3_sync_fp8" if v3 else "vllm_0_30_v4_sync_fp8_ds_mla")
    if v4:
        context = max_context if v4_max_model_len is None else v4_max_model_len
        batch = max_rows if v4_max_batched_tokens is None else v4_max_batched_tokens
        branch = False if v4_use_deep_gemm is None else v4_use_deep_gemm
        routing = ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0)
        inputs = {phase: RuntimeInputs(routing, "mla", "uniform", model_execution_version="vllm_0_30_0",
            symbolic_forward_source=source, target_sampling_source=sampler, embedding_token_ids=tuple(ids),
            symbolic_workspace_initial_capacity_bytes=(0,) * parallel.worker_count,
            v4_eager_metadata_source="vllm_0_30_eager_dict_all_valid",
            v4_routing_source="vllm_0_30_modular_cuda_top6_pure_text" if v4_routing_source is None else v4_routing_source,
            v4_source_max_model_len=context, v4_source_max_batched_tokens=batch,
            v4_source_use_deep_gemm=branch,
            v4_indexer_selection_source="declared_causal_topk" if v4_indexer_selection is None else v4_indexer_selection)
            for phase, ids in token_ports.items()}
    else:
        inputs = build_model_operator_probe_source_inputs(symbolic_forward_source=source,
            embedding_token_ids_by_phase=token_ports, source_max_model_len=max_context,
            source_max_batched_tokens=max_rows, target_sampling_source=sampler,
            workspace_initial_capacity_bytes=(0,) * parallel.worker_count if glm else None,
            glm_indexer_selection_source="declared_causal_topk" if glm else None)
    receipt = {"entrypoint": "default_static_target_fixture", "source_scenario": source,
        "synthetic_fixture": True, "historical_default_equivalence_claimed": False,
        "constructor_scope": "fixture minimum context/batch extents; not observed runtime configuration",
        "token_input_scope": "caller-provided IDs preserved per request" if original_ports is not None else
            "declared sequential IDs modulo vocabulary per phase; not model outputs or observed tokens",
        "phase_request_order_before": original_order,
        "phase_request_order_after": {phase: tuple(request.request_id for request in requests)
            for phase, requests in phase_requests},
        "source_constructor_inputs": {phase: {name: vars(runtime)[name] for name in (
            "symbolic_source_max_model_len", "symbolic_source_max_batched_tokens",
            "symbolic_workspace_initial_capacity_bytes", "glm_indexer_selection_source",
            "v4_source_max_model_len", "v4_source_max_batched_tokens", "v4_eager_metadata_source",
            "v4_routing_source", "v4_source_use_deep_gemm", "v4_indexer_selection_source")}
            for phase, runtime in inputs.items()},
        "workspace_scope": "declared cold source workspace per physical rank" if glm or v4 else "not selected by this fixture",
        "target_sampling_source": sampler, "source_projection_only": True,
        "actual_provider_runtime_bound": False, "normal_capacity_accepted": False}
    return phase_requests, inputs, receipt


def build_model_operator_probe(
    model_path: str,
    tensor_parallel_size: int,
    enable_expert_parallel: bool,
    *,
    mtp_mode: str = "off",
    prefill_tokens: int = 8,
    v4_eager_metadata: bool = False,
    v4_routing_source: str | None = None,
    pipeline_parallel_size: int = 1,
    pipeline_link: str | None = None,
    pipeline_layer_partition: str | None = None,
    source_graph_only: bool = False,
    mtp_draft_sampling_source: str | None = None,
    target_sampling_source: str | None = None,
    embedding_token_ids_by_phase: Mapping[str, tuple[int, ...]] | None = None,
    v4_source_max_model_len: int | None = None,
    v4_source_max_batched_tokens: int | None = None,
    v4_indexer_selection_source: str | None = None,
    v4_source_use_deep_gemm: bool | None = None,
    source_phase_requests: tuple[tuple[str, tuple[ScheduledRequest, ...]], ...] | None = None,
    source_runtime_inputs_by_phase: Mapping[str, RuntimeInputs] | None = None,
) -> dict[str, object]:
    """Run full layers for prefill, two decode steps, and a mixed step.

    This emits modeled facts, not independent reference expectations or a
    success gate. Native GLM-on uses explicit acceptance/rejection inputs;
    real Scheduler execution and final source alignment remain unverified.
    """
    if isinstance(prefill_tokens, bool) or not isinstance(prefill_tokens, int) or prefill_tokens < 1:
        raise ValueError("static probe prefill_tokens must be a positive integer")
    hardware = HWSpec(H200_HARDWARE_CONFIG)
    model = load_model_config(model_path)
    if not isinstance(source_graph_only, bool):
        raise TypeError("source_graph_only must be boolean")
    if source_graph_only and (model.model_type != "deepseek_v4" or mtp_mode != "off"):
        raise NotImplementedError("this provenance-only graph capture requires V4 target-off")
    if type(pipeline_parallel_size) is not int or pipeline_parallel_size not in (1, 2):
        raise ValueError("static source probe supports PP1 or GLM target-off PP2")
    if pipeline_parallel_size == 2:
        if (model.model_type != "glm_moe_dsa" or mtp_mode != "off" or pipeline_link != "nvlink"
                or pipeline_layer_partition != "38,40"):
            raise NotImplementedError("PP2 source probe requires GLM target-off, explicit NVLink scenario and partition 38,40")
    elif pipeline_link is not None or pipeline_layer_partition is not None:
        raise ValueError("pipeline link applies only to PP2 source probe")
    if not isinstance(v4_eager_metadata, bool):
        raise TypeError("v4_eager_metadata must be boolean")
    if v4_eager_metadata and model.model_type != "deepseek_v4":
        raise ValueError("V4 eager source metadata requires a V4 model")
    if mtp_mode not in ("off", "native"):
        raise ValueError("unknown static MTP mode")
    if mtp_mode == "native" and model.model_type != "glm_moe_dsa":
        raise NotImplementedError("native static probe currently requires exact GLM V32 config")
    if mtp_draft_sampling_source is not None and mtp_mode != "native":
        raise ValueError("draft sampling source requires native probe")
    if target_sampling_source in ("vllm_0_30_plain_sampler_bf16", "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_greedy_fp16") and mtp_mode != "off":
        raise ValueError("plain target sampling source requires MTP-off probe")
    if target_sampling_source == "vllm_0_30_rejection_raw_logits_bf16" and mtp_mode != "native":
        raise ValueError("rejection target sampling source requires native probe")
    if source_runtime_inputs_by_phase is not None:
        if not isinstance(source_runtime_inputs_by_phase, Mapping):
            raise TypeError("source runtime inputs require a phase mapping")
        if (v4_eager_metadata or v4_routing_source is not None
                or mtp_draft_sampling_source is not None or target_sampling_source is not None
                or embedding_token_ids_by_phase is not None
                or v4_source_max_model_len is not None or v4_source_max_batched_tokens is not None
                or v4_indexer_selection_source is not None
                or v4_source_use_deep_gemm is not None):
            raise ValueError("source runtime inputs conflict with independent legacy runtime arguments")
    collectives = tuple(
        CollectiveConfig(
            operation, "ring", "nvlink", "ring", "main-collective-fabric",
            "analytical_roofline_v1", None)
        for operation in ("all_reduce", "all_gather"))
    parallel = ParallelConfig(
        tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size,
        enable_expert_parallel=enable_expert_parallel)
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main", "colocated", "nvidia_h200", parallel,
                ranks=tuple(range(parallel.worker_count)),
                collectives=collectives, pipeline_p2p=(
                    (PipelineP2PConfig(0, 1, "send_recv", pipeline_link, "point_to_point",
                        "explicit_source_pipeline_fabric", "analytical_roofline_v1"),)
                    if pipeline_parallel_size == 2 else ())),)),
        num_layers=model.num_hidden_layers, num_experts=model.num_experts,
        pipeline_layer_partition=(38, 40) if pipeline_parallel_size == 2 else None)
    routing = ExpertRoutingInput.seeded_assumption(
        "operator_support_static_seeded_routing_v1", 0)
    runtime = RuntimeInputs(
        routing, "paged_varlen" if model.model_type == "glm4_moe" else "mla", "uniform",
        model_execution_version="vllm_0_11_0" if model.model_type == "glm4_moe" else "vllm_0_30_0",
        symbolic_forward_source="vllm_0_11_glm4_sync" if model.model_type == "glm4_moe" else None,
        v4_eager_metadata_source="vllm_0_30_eager_dict_all_valid" if v4_eager_metadata else None,
        v4_source_max_model_len=v4_source_max_model_len,
        v4_source_max_batched_tokens=v4_source_max_batched_tokens,
        v4_source_use_deep_gemm=v4_source_use_deep_gemm,
        v4_routing_source=v4_routing_source)
    if mtp_draft_sampling_source is not None:
        runtime = replace(runtime, mtp_draft_sampling_source=mtp_draft_sampling_source)
    if target_sampling_source is not None:
        runtime = replace(runtime, target_sampling_source=target_sampling_source)
    pp_weight_projection = None
    if pipeline_parallel_size == 2:
        rank_weights = model_rank_residencies(model, parallel, worker_role="colocated", include_mtp=False)
        pp_weight_projection = {
            "scope": "existing owner target-off static weight projection; not normal allocator evidence",
            "maximum_rank_weight_bytes": max(rank.weight_resident_bytes for rank in rank_weights),
            "physical_hbm_spec_bytes": hardware.resource_by_role("dram").size_bytes,
            "rank_residencies": [asdict(rank) for rank in rank_weights],
            "reserve_activation_cache_runtime_allocation_bound": False,
        }
    phase_requests = _operator_probe_phase_requests(prefill_tokens, mtp_mode)
    if (source_phase_requests is None and source_runtime_inputs_by_phase is not None
            and source_runtime_inputs_by_phase and all(isinstance(inputs, RuntimeInputs)
                and inputs.symbolic_forward_source == "vllm_0_27_1_deepseek_v3_sync_fp8"
                for inputs in source_runtime_inputs_by_phase.values())):
        # This static fixture declares model-input order, not a Scheduler or
        # persistent InputBatch reorder observation. V3's common source
        # requires query-length-one requests before longer prefill rows.
        # Supplied phase IDs are packed in this declared fixture order.
        phase_requests = _operator_probe_phase_requests(prefill_tokens, mtp_mode, decode_prefix=True)
    if source_phase_requests is not None:
        if (mtp_mode != "off" or (source_runtime_inputs_by_phase is None
                and (model.model_type != "deepseek_v4" or not v4_eager_metadata))):
            raise ValueError("explicit source phase requests require target-off and complete phase RuntimeInputs or V4 eager metadata")
        if (not isinstance(source_phase_requests, tuple) or not source_phase_requests
                or any(not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
                       or not isinstance(item[1], tuple) or not item[1]
                       or any(not isinstance(request, ScheduledRequest) for request in item[1]) for item in source_phase_requests)
                or len({item[0] for item in source_phase_requests}) != len(source_phase_requests)):
            raise ValueError("source phase requests require unique named nonempty ScheduledRequest tuples")
        phase_requests = source_phase_requests
    if embedding_token_ids_by_phase is not None:
        if not isinstance(embedding_token_ids_by_phase, Mapping):
            raise TypeError("embedding tokens require a phase mapping")
        if mtp_mode != "off":
            raise NotImplementedError("explicit embedding tokens require a lowered target-off source probe")
        if set(embedding_token_ids_by_phase) != {phase for phase, _ in phase_requests}:
            raise ValueError("embedding tokens require exactly the declared source probe phases")
    source_normalization = None
    if v4_indexer_selection_source is not None and v4_source_max_batched_tokens is None:
        raise ValueError("indexer source requires the explicit virtual batch constructor cap")
    if v4_source_max_batched_tokens is not None:
        if (model.model_type != "deepseek_v4" or model.compute_dtype != "bf16" or model.weight_dtype != "fp8"
                or tensor_parallel_size not in (4, 8) or pipeline_parallel_size != 1 or enable_expert_parallel
                or mtp_mode != "off" or not v4_eager_metadata
                or v4_routing_source not in ("vllm_0_30_modular_cuda_top6_pure_text", "vllm_0_30_triton_top6_pure_text")
                or embedding_token_ids_by_phase is None or v4_source_max_model_len is None
                or v4_source_use_deep_gemm is None or target_sampling_source not in (
                    None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16")):
            raise ValueError("explicit V4 batch capacity requires the complete target-off eager source facts")
        # The persistent top-k buffer uses the caller's constructor capacity,
        # never a capacity inferred from these static scheduled rows.
        source_runtime_inputs_by_phase = {phase: replace(runtime,
            symbolic_forward_source="vllm_0_30_v4_sync_fp8_ds_mla",
            v4_indexer_selection_source=v4_indexer_selection_source,
            embedding_token_ids=embedding_token_ids_by_phase[phase]) for phase, _ in phase_requests}
        source_normalization = {"entrypoint": "historical_v4_probe_arguments",
            "source_scenario": f"H200_vLLM0.30_FP8_V4_TP{tensor_parallel_size}_PP1_eager",
            "target_sampling_source": target_sampling_source, "source_projection_only": True,
            "actual_provider_runtime_bound": False, "token_values_inferred": False}
    ordinary_dense = model.model_type in ("qwen3", "llama") and not model.num_experts
    ordinary_moe = model.model_type == "qwen3_moe" and model.num_experts in (2, 128)
    dense_tp2 = ordinary_dense and tensor_parallel_size == 2
    moe_ep2 = (ordinary_moe and tensor_parallel_size == 2 and enable_expert_parallel
        and is_validated_qwen3_moe_ep2_model_scope(model))
    moe_tp2 = ordinary_moe and tensor_parallel_size == 2 and not enable_expert_parallel
    if (source_runtime_inputs_by_phase is None and (ordinary_dense or ordinary_moe)
            and model.compute_dtype == model.weight_dtype and model.compute_dtype in ("bf16", "fp16")
            and (model.compute_dtype == "bf16" or model.model_type == "qwen3")
            and (tensor_parallel_size == 1 or dense_tp2 or moe_ep2 or moe_tp2) and pipeline_parallel_size == 1
            and (not enable_expert_parallel or moe_ep2)
            and mtp_mode == "off" and target_sampling_source in ((None, "vllm_0_30_plain_greedy_bf16",
                "vllm_0_30_plain_sampler_bf16") if model.compute_dtype == "bf16" else
                (None, "vllm_0_30_plain_greedy_fp16"))
            and not source_graph_only and not v4_eager_metadata
            and all(value is None for value in (v4_routing_source, v4_source_max_model_len,
                v4_source_use_deep_gemm, mtp_draft_sampling_source))):
        # Preserve the historical probe's H200/0.30 source scenario and
        # its explicitly chosen LM-head-only or greedy scope. No sampler preference, token
        # value, provider or constructor state is inferred from a checkpoint.
        ports = ({phase: None for phase, _ in phase_requests} if embedding_token_ids_by_phase is None
            else embedding_token_ids_by_phase)
        source_runtime_inputs_by_phase = build_model_operator_probe_source_inputs(
            symbolic_forward_source="cuda_eager_inplace", embedding_token_ids_by_phase=ports,
            source_max_model_len=None, source_max_batched_tokens=None, target_sampling_source=target_sampling_source)
        if ordinary_moe:
            # Reuse the probe's declared routing, never derive an expert
            # histogram or token values from the checkpoint/config.
            source_runtime_inputs_by_phase = {phase: replace(inputs, expert_routing=routing)
                for phase, inputs in source_runtime_inputs_by_phase.items()}
        source_normalization = {"entrypoint": "historical_moe_probe_arguments" if ordinary_moe else "historical_dense_probe_arguments",
            "source_scenario": ("H200_vLLM0.30_BF16_Qwen3_MoE_TP2_EP2_PP1_eager" if moe_ep2
                else "H200_vLLM0.30_BF16_Qwen3_MoE_TP2_EP1_PP1_eager" if moe_tp2
                else "H200_vLLM0.30_BF16_Qwen3_MoE_TP1_PP1_eager" if ordinary_moe
                else f"H200_vLLM0.30_{model.compute_dtype.upper()}_dense_TP{tensor_parallel_size}_PP1_eager"),
            "target_sampling_source": target_sampling_source, "source_projection_only": True,
            "actual_provider_runtime_bound": False, "token_values_inferred": False}
    if source_runtime_inputs_by_phase is None and mtp_mode == "off":
        fixture = _default_target_probe_source(model, parallel, phase_requests,
            token_ports=embedding_token_ids_by_phase, sampler=target_sampling_source,
            v4_routing_source=v4_routing_source, v4_max_model_len=v4_source_max_model_len,
            v4_max_batched_tokens=v4_source_max_batched_tokens, v4_indexer_selection=v4_indexer_selection_source,
            v4_use_deep_gemm=v4_source_use_deep_gemm)
        if fixture is not None:
            phase_requests, source_runtime_inputs_by_phase, source_normalization = fixture
        elif model.model_type == "glm4_moe":
            if any(value is not None for value in (v4_routing_source, v4_source_max_model_len,
                    v4_source_max_batched_tokens, v4_indexer_selection_source, v4_source_use_deep_gemm)):
                raise ValueError("GLM4 probe does not consume V4 source inputs")
            fixture = _default_glm4_target_probe_source(model, parallel, phase_requests,
                token_ports=embedding_token_ids_by_phase, sampler=target_sampling_source)
            if fixture is not None:
                source_runtime_inputs_by_phase, source_normalization = fixture
    if source_runtime_inputs_by_phase is None and mtp_mode == "native":
        if any(value is not None for value in (v4_routing_source, v4_source_max_model_len,
                v4_source_max_batched_tokens, v4_indexer_selection_source, v4_source_use_deep_gemm)):
            raise ValueError("GLM native fixture does not consume V4 source options")
        fixture = _default_glm_native_probe_source(model, parallel, phase_requests,
            sampler=target_sampling_source, draft_sampler=mtp_draft_sampling_source)
        if fixture is not None:
            source_runtime_inputs_by_phase, source_normalization = fixture
    if source_runtime_inputs_by_phase is None:
        raise NotImplementedError(
            "static probe arguments do not select a maintained common source: "
            f"model={model.model_type}, TP={tensor_parallel_size}, PP={pipeline_parallel_size}, "
            f"EP={enable_expert_parallel}, MTP={mtp_mode}, sampler={target_sampling_source!r}")
    if set(source_runtime_inputs_by_phase) != {phase for phase, _ in phase_requests}:
        raise ValueError("source runtime inputs require exactly the declared probe phases")
    from src.serving.workload.traced_models import bind_ordinary_forward_source
    normalized_inputs = {}
    for phase, requests in phase_requests:
        supplied = source_runtime_inputs_by_phase[phase]
        if not isinstance(supplied, RuntimeInputs):
            raise TypeError("every source phase requires RuntimeInputs")
        supplied = bind_ordinary_forward_source(model, deployment, supplied)
        if supplied.symbolic_forward_source is None:
            raise ValueError("every source phase requires complete facts for a shared symbolic forward source")
        normalized_inputs[phase] = supplied
        # Unknown IDs are real virtual input ports. The common source
        # builder owns their exact applicability; TP/EP fused embedding
        # and other value-dependent domains still require token values.
        if (supplied.embedding_token_ids is not None
                and len(supplied.embedding_token_ids) != sum(request.scheduled_tokens for request in requests)):
            raise ValueError("source embedding token IDs must cover every scheduled row")
    if any(normalized_inputs[phase] is not supplied for phase, supplied in source_runtime_inputs_by_phase.items()):
        source_normalization = {"entrypoint": "programmatic_phase_runtime_inputs",
            "source_projection_only": True, "actual_provider_runtime_bound": False,
            "token_values_inferred": False, "sampler_source_inferred": False}
    source_runtime_inputs_by_phase = normalized_inputs

    def runtime_field(name):
        values = tuple(vars(source_runtime_inputs_by_phase[phase])[name] for phase, _ in phase_requests)
        return values[0] if all(value == values[0] for value in values) else None

    static_lm_head_only = (mtp_mode == "off" and source_runtime_inputs_by_phase is not None and all(
        source_runtime_inputs_by_phase[phase].target_sampling_source is None for phase, _ in phase_requests))
    static_cast_only = (mtp_mode == "off" and source_runtime_inputs_by_phase is not None and all(
        source_runtime_inputs_by_phase[phase].target_sampling_source in (
            "vllm_0_11_plain_sampler_bf16", "vllm_0_30_plain_sampler_bf16")
        for phase, _ in phase_requests))
    static_output_scope = ("lm_head_only_no_sampler" if static_lm_head_only else
        "fp32_logits_only" if static_cast_only else None)
    results = []
    for phase, requests in phase_requests:
        phase_runtime = source_runtime_inputs_by_phase[phase]
        step = ScheduleStep(phase, "main", requests, attention_kv_layout=phase_runtime.attention_kv_layout)
        source_storage, source_graph = None, None

        def observe_source(source, observed_graph):
            nonlocal source_storage, source_graph
            from src.serving.kv_cache.activation_memory import symbolic_forward_storage_report
            if source_storage is not None:
                raise RuntimeError("static probe executed its source forward more than once")
            source_storage = symbolic_forward_storage_report(source, observed_graph,
                source_profile=phase_runtime.symbolic_forward_source)
            source_graph = observed_graph

        graph = build_symbolic_workload(step, model, deployment, launch_policy="analytical_launch",
            kv_cache_events=(), runtime_inputs=phase_runtime, hardware_by_id={"nvidia_h200": hardware},
            trace_observer=observe_source)
        if source_storage is None or source_graph is not graph:
            raise RuntimeError("static probe lost the storage facts of its actual source graph")
        if source_graph_only:
            results.append({
                "phase": phase,
                "source_runtime_inputs": asdict(phase_runtime),
                "source_activation_memory": source_storage,
                "scheduled_requests": [asdict(request) for request in requests],
                "nodes": [{
                    "node_id": node.node_id, "rank": node.rank,
                    "kind": node.kind.value, "variant": node.variant,
                    "shape": node.shape, "dtype": node.dtype,
                    "attributes": node.attributes,
                } for node in graph.nodes],
                "edges": [asdict(edge) for edge in graph.edges],
                "block_repeats": [asdict(repeat) for repeat in graph.block_repeats],
            })
            continue
        plan = lower_workload(graph, deployment, {"nvidia_h200": hardware})
        calls = []
        priced_predictions = {}
        for call in plan.calls:
            kernel = call.kernel_spec
            try:
                result = predict(kernel, hardware, method="roofline")
            except NotImplementedError as error:
                prediction = {"method": "roofline", "seconds": None, "status": "unpriced", "reason": str(error)}
            else:
                prediction = {"method": "roofline", "seconds": result.t_total_s,
                    "status": "priced", "scope": "logical_roofline_only_no_native_launch_acceptance"}
            resource = {"status": "public_resource_input_unavailable", "demand": None, "limitations": ()}
            try:
                if kernel.execution_support == "unsupported":
                    resource = {
                        "status": "launch_selector_unavailable",
                        "demand": None, "limitations": (kernel.unsupported_reason,),
                    }
                    if kernel.kind is KernelKind.VECTOR and kernel.variant in (
                            "mhc_prenorm", "mhc_prenorm_broadcast"):
                        work = mhc_prenorm_roofline_work(kernel)
                        resource["logical_scope"] = work['arithmetic_contract']
                        resource.update({"logical_work": work["logical_work"],
                                         "demand": work["demand"], "logical_bytes": work["logical_bytes"],
                                         "limitations": work["limitations"]})
                        resource["status"] = "source_logical_work_launch_unbound"
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "dsv4_indexer_q_rope_quant":
                        work = dsv4_indexer_q_roofline_work(kernel)
                        resource["logical_scope"] = work['scope']
                        resource.update({"logical_work": work["logical_work"], "logical_bytes": work["logical_bytes"],
                                         "roofline_source_programs": {**work["source_programs"],
                                                                      "all_source_branches_memory_dominated": True}})
                        resource["status"] = "source_logical_work_launch_unbound"
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "vllm_logits_gather_reorder":
                        work = vllm_logits_gather_reorder_logical_work(kernel)
                        resource["logical_scope"] = work['scope']
                        resource.update({"status": "source_logical_work_launch_unbound", "logical_work": work,
                            "logical_bytes": sum(operand["bytes"] for operand in work["operands"]), "demand": work["demand"]})
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant in (
                            "vllm_logits_fp32_cast", "vllm_rejection_logits_materialize"):
                        work = (vllm_logits_fp32_cast_logical_work(kernel)
                                if kernel.variant == "vllm_logits_fp32_cast"
                                else vllm_rejection_logits_materialization_logical_work(kernel))
                        resource["logical_scope"] = 'logical_roofline_only_no_launch_binding'
                        resource.update({"status": "source_logical_work_launch_unbound",
                                         "logical_work": work,
                                         "logical_bytes": sum(operand["bytes"] for operand in work["operands"]),
                                         "limitations": work["limitations"]})
                        if kernel.variant == "vllm_logits_fp32_cast":
                            resource["demand"] = work["demand"]
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "vllm_plain_greedy_sample":
                        work = vllm_plain_greedy_roofline_work(kernel)
                        resource["logical_scope"] = work['arithmetic_contract']
                        resource.update({"status": "source_logical_work_launch_unbound",
                                         "logical_work": work["logical_work"], "demand": work["demand"],
                                         "logical_bytes": work["logical_bytes"],
                                         "limitations": work["limitations"]})
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant in (
                            "dsv32_draft_argmax", "dsv32_draft_id_cast", "dsv32_draft_row_select", "dsv32_draft_buffer_copy", "dsv32_draft_token_stack", "dsv32_draft_id_scatter", "dsv32_draft_prepare_next_ids", "dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback", "dsv32_draft_step_metadata"):
                        resource["logical_work"] = dsv32_draft_sampling_logical_work(kernel)
                        resource["status"] = "source_logical_work_launch_unbound"
                        if kernel.variant in DSV32_DRAFT_MATERIALIZATION_VARIANTS:
                            resource["logical_scope"] = 'logical_roofline_only_no_launch_binding'
                            resource["logical_bytes"] = dsv32_draft_materialization_roofline_bytes(kernel)
                            resource["limitations"] = (
                                "typed_logical_materialization_bytes_not_observed_dram_traffic",
                                "zero_float_work_not_zero_integer_address_control_or_conversion_services",
                                "actual_library_launch_compiler_provider_and_runtime_unbound")
                        elif (kernel.variant in DSV32_DRAFT_INT32_BOOKKEEPING_VARIANTS
                              and (kernel.attributes["index_preparation_mode"] == "padded_spec"
                                   or (kernel.variant == "dsv32_draft_prepare_sample_indices"
                                       and kernel.attributes["index_preparation_mode"] == "ordinary_query_end"))):
                            logical_bytes, int32_ops = dsv32_draft_int32_bookkeeping_roofline_work(kernel)
                            resource["logical_scope"] = 'logical_roofline_only_no_launch_binding'
                            resource["logical_bytes"] = logical_bytes
                            resource["logical_int32_operations"] = int32_ops
                            resource["limitations"] = (
                                "declared_metadata_source_mode_not_observed_scheduler_output",
                                "source_int32_arithmetic_comparison_select_not_compiled_instruction_count",
                                "logical_role_bytes_not_observed_dram_or_cache_traffic",
                                "address_control_library_launch_compiler_and_runtime_services_unbound")
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "vllm_logits_row_select":
                        resource["logical_work"] = vllm_logits_row_selection_logical_work(kernel)
                        resource["demand"] = resource["logical_work"]["demand"]
                        resource["status"] = "source_logical_work_launch_unbound"
                        if kernel.attributes["index_source"] != "unbound_spec_metadata":
                            resource["logical_scope"] = 'logical_roofline_only_no_launch_binding'
                            resource["logical_bytes"] = vllm_logits_row_selection_roofline_bytes(kernel)
                            resource["limitations"] = (
                                "typed_logical_copy_and_index_bytes_not_observed_dram_traffic",
                                "zero_mathematical_flops_not_zero_address_or_control_instructions",
                                "actual_provider_library_launch_compiler_and_runtime_unbound")
                    elif (kernel.kind is KernelKind.VECTOR and kernel.variant == "fp8_group_quant"
                          and kernel.attributes["source_program"] is not None):
                        work = fp8_group_quant_source_work(kernel)
                        resource.update({"status": "conditional_source_arithmetic_byte_baseline", "logical_work": work,
                            "logical_bytes": work["logical_bytes"], "source_read_write_bytes": work["source_read_write_bytes"],
                            "all_source_programs_memory_dominated": True})
                        resource["logical_scope"] = 'common ideal arithmetic/byte Roofline for all source alternatives; no provider, scale-mode or native issue binding'
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "vocab_parallel_embedding":
                        work = vocab_parallel_embedding_logical_work(kernel)
                        resource["logical_work"] = work
                        resource["status"] = "source_logical_work_launch_unbound"
                        if work["owned_token_rows"] is not None:
                            resource["logical_scope"] = 'logical_roofline_only_no_launch_binding'
                            resource["logical_bytes"] = sum(operand["bytes"] for operand in work["operands"])
                            resource["limitations"] = work["limitations"]
                        else:
                            resource["limitations"] = (*resource["limitations"], 'scheduled_token_ownership_unbound')
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "dsv32_moe_route":
                        resource["logical_work"] = dsv32_moe_route_logical_work(kernel)
                        resource["status"] = "source_logical_work_launch_unbound"
                        resource["limitations"] = (*resource["limitations"], 'grouped_topk_selection_and_nonlinear_logical_service_unbound')
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "kimi_moe_route":
                        resource["logical_work"] = kimi_moe_route_logical_work(kernel)
                        resource["status"] = "source_logical_work_launch_unbound"
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "dsv4_moe_route":
                        resource["logical_work"] = dsv4_moe_route_logical_work(kernel)
                        resource["status"] = "source_logical_work_launch_unbound"
                elif kernel.execution_support == "supported":
                    pass
                else:
                    raise ValueError("unknown kernel execution support")
                if (kernel.kind is KernelKind.SPARSE_ATTN_INDEXER
                      and kernel.variant == "dsa_index_reuse"):
                    resource = {
                        "status": "logical_buffer_alias_no_kernel_work",
                        "demand": None, "limitations": (),
                    }
                elif kernel.execution_support == "unsupported":
                    pass  # Logical pricing and physical launch support are separate contracts.
                elif kernel.kind in (
                        KernelKind.COMMUNICATION, KernelKind.MEMCPY,
                        KernelKind.KV_OFFLOAD):
                    resource = {
                        "status": "transfer_uses_own_physical_contract",
                        "demand": None, "limitations": (),
                    }
                elif kernel.kind is KernelKind.GEMM and is_block_fp8_gemm(kernel):
                    from src.kernel.predictors.gemm_block_fp8 import block_fp8_gemm_roofline_work
                    work = block_fp8_gemm_roofline_work(kernel)
                    resource = {
                        "status": "block_fp8_template_exposes_no_resource_input",
                        "demand": None,
                        "logical_work": work,
                        "logical_bytes": work["logical_bytes"],
                        "limitations": ("public_resource_input_unavailable",),
                    }
                else:
                    inp, logical_bytes = create_predictor(
                        kernel.kind).resource_input(kernel, hardware)
                    resource = {
                        "status": "public_source_resource_input",
                        "demand": inp.demand_per_resource,
                        "logical_bytes": logical_bytes,
                        "modeled_dram_bytes": {
                            access: sum(node.bytes for path in inp.memory_paths
                                        if path.access_kind == access for node in path.nodes
                                        if node.resource_role == "dram")
                            for access in ("read", "write")},
                        "modeled_l2_bytes": {
                            access: sum(node.bytes for path in inp.memory_paths
                                        if path.access_kind == access for node in path.nodes
                                        if node.resource_role == "l2_cache")
                            for access in ("read", "write")},
                        "limitations": inp.limitations,
                    }
                    if kernel.kind is KernelKind.SPARSE_ATTN_INDEXER and kernel.variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode"):
                        resource["indexer_logical_work"] = dsv4_indexer_logical_work(kernel)
                        resource["logical_scope"] = resource['indexer_logical_work']['scope']
                    if kernel.kind is KernelKind.SPARSE_ATTN and kernel.variant in ("dsv4_sparse_prefill", "dsv4_sparse_decode"):
                        resource["attention_inputs"] = dsv4_attention_source_inputs(kernel)
                        if kernel.variant == "dsv4_sparse_decode":
                            resource["decode_attention_index_inputs"] = dsv4_decode_attention_index_inputs(kernel)
                    if kernel.kind is KernelKind.VECTOR and kernel.variant == "silu" and kernel.attributes["swiglu_limit"] is not None:
                        resource["logical_work"] = clamped_silu_logical_work(kernel)
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "dsv4_short_context_topk":
                        resource["logical_work"] = dsv4_short_context_topk_source_work(kernel)
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "dsv4_moe_route":
                        resource["route_source_work"] = dsv4_route_source_work(kernel)
                        resource["route_tensor_objects"] = dsv4_route_tensor_objects(kernel)
                    elif (model.model_type == "deepseek_v4" and kernel.kind is KernelKind.VECTOR
                          and kernel.variant == "rmsnorm" and kernel.dtype == "bf16"):
                        resource["logical_work"] = rmsnorm_logical_work(kernel)
                    elif kernel.kind is KernelKind.SPARSE_ATTN and kernel.variant == "dsv4_sparse_prefill":
                        resource["logical_work"] = dsv4_prefill_gather_logical_work(kernel)
                        resource["logical_scope"] = 'logical_common_work_and_causal_candidate_capacity_projection_not_provider_or_latency_bound'
                    elif (kernel.kind is KernelKind.SPARSE_ATTN and kernel.variant == "dsv4_sparse_decode"
                          and kernel.shape["compress_ratio"] == 4):
                        resource["decode_index_materialization"] = dsv4_decode_index_materialization_work(kernel)
                        resource["logical_scope"] = 'logical_c4_index_materialization_and_attention_projection_not_complete_source_or_provider_service'
                    elif kernel.kind is KernelKind.KV_CACHE and kernel.variant == "dsv4_qnorm_rope_kv_insert":
                        resource["swa_insert_objects"] = dsv4_swa_insert_tensor_objects(kernel)
                    elif kernel.kind is KernelKind.KV_CACHE and kernel.variant == "dsv4_compressed_state_cache":
                        resource["finish_work"] = dsv4_finish_logical_work(kernel)
                        resource["partial_state_input"] = dsv4_finish_partial_state_input(kernel)
                    elif kernel.kind is KernelKind.KV_CACHE and kernel.variant == "dsv4_compressor_state_cache":
                        resource["partial_state_objects"] = dsv4_partial_state_tensor_objects(kernel)
                    elif (model.model_type == "deepseek_v4" and kernel.kind is KernelKind.FUSED_MOE
                          and kernel.variant == "routed_experts" and kernel.attributes["expert_parallel_size"] == 1
                          and kernel.attributes["swiglu_limit"] is not None):
                        resource["logical_work"] = dsv4_marlin_moe_source_work(kernel)
                        resource["marlin_roofline"] = dsv4_marlin_moe_roofline_work(kernel, hardware)
                        resource["marlin_scratch"] = dsv4_marlin_moe_scratch_work(kernel, hardware)
                        resource["logical_scope"] = resource['marlin_roofline']['scope']
                if kernel.kind is KernelKind.VECTOR and kernel.variant in (
                        "mhc_prenorm", "mhc_prenorm_broadcast", "mhc_post_prenorm",
                        "mhc_pre_norm", "mhc_pre_norm_broadcast", "mhc_post", "hc_head"):
                    resource["mhc_stage_work"] = mhc_stage_logical_work(kernel)
            except NotImplementedError as error:
                resource["limitations"] = (*resource["limitations"], str(error))
            if prediction["status"] == "priced":
                if "logical_scope" in resource:
                    prediction["scope"] = resource["logical_scope"]
                if resource["status"] == "logical_buffer_alias_no_kernel_work":
                    if result.t_total_s != 0:
                        raise ValueError("logical buffer alias must have zero public prediction duration")
                    prediction["status"] = "logical_alias"
            launch = kernel.launch
            if prediction["status"] in ("priced", "logical_alias"):
                priced_predictions[call.call_id] = result
            calls.append({
                "call_id": call.call_id, "rank": call.rank,
                "source_node_ids": call.source_node_ids,
                "dependencies": call.dependencies,
                "kind": kernel.kind.value, "variant": kernel.variant,
                "shape": kernel.shape, "dtype": kernel.dtype,
                "attributes": kernel.attributes,
                "execution_support": kernel.execution_support,
                "transfer": None if kernel.transfer is None else asdict(kernel.transfer),
                "launch": None if launch is None else {
                    "algorithm": launch.algorithm, "grid": launch.grid,
                    "block": launch.block,
                    "execution_variant": launch.execution_variant,
                    "parameters": launch.parameters,
                },
                "resource_input": resource,
                "prediction": prediction,
            })
        logical_timeline = None
        if len(priced_predictions) == len(plan.calls):
            timeline = build_execution_timeline(plan, priced_predictions)
            logical_timeline = {
                "duration_s": timeline.duration_s,
                "predicted_device_time_s": timeline.predicted_device_time_s,
                "call_count": len(plan.calls),
                "scope": "logical_roofline_kernel_DAG_no_host_dispatch_or_runtime_acceptance",
            }
        complete_source_chain = source_runtime_inputs_by_phase is not None and (
            phase_runtime.target_sampling_source == "vllm_0_30_plain_greedy_bf16"
            or (phase_runtime.symbolic_forward_source == "vllm_0_27_1_deepseek_v3_sync_fp8"
                and phase_runtime.target_sampling_source == "vllm_0_27_1_plain_greedy_bf16")
            or (phase_runtime.symbolic_forward_source == "vllm_0_11_glm4_sync"
                and phase_runtime.target_sampling_source == "vllm_0_11_plain_greedy_bf16")
            or (mtp_mode == "native" and isinstance(phase_runtime.model_source_inputs, V32PaddedMTPInputs)
                and phase_runtime.mtp_draft_sampling_source == "vllm_0_30_legacy_full_logits_greedy"
                and phase_runtime.target_sampling_source == "vllm_0_30_rejection_raw_logits_bf16"))
        results.append({
            "phase": phase,
            "source_runtime_inputs": asdict(phase_runtime),
            "source_activation_memory": source_storage,
            "scheduled_requests": [
                {**asdict(request), "positions": tuple(range(
                    request.computed_tokens,
                    request.computed_tokens + request.scheduled_tokens))}
                for request in requests],
            "logical_nodes": len(graph.nodes), "kernel_calls": len(calls),
            "unpriced_calls": sum(
                call["prediction"]["status"] == "unpriced" for call in calls),
            # Logical pricing and native launch support are independent.
            # The old native-proposer projection still omits source work;
            # an explicitly complete virtual target does not become partial
            # merely because its mathematical kernels have no native launch.
            "roofline_end_to_end": "source_projection_only" if pipeline_parallel_size == 2 else "partial" if (
                not complete_source_chain) or any(
                call["prediction"]["status"] == "unpriced" for call in calls
            ) else "priced",
            "modeled_calls_pricing": "partial" if any(
                call["prediction"]["status"] == "unpriced" for call in calls
            ) else "priced",
            "logical_roofline_timeline": logical_timeline,
            "nodes": [{
                "node_id": node.node_id, "rank": node.rank,
                "kind": node.kind.value, "variant": node.variant,
                "shape": node.shape, "dtype": node.dtype,
                "attributes": node.attributes,
                "predecessors": graph.predecessors(node.node_id),
            } for node in graph.nodes],
            "calls": calls,
        })
    if source_graph_only:
        return _json_value({
            "snapshot": _snapshot(), "model_config_path": model_path,
            "probe_workload_path": "symbolic_forward",
            "source_normalization": source_normalization,
            "static_output_scope": static_output_scope,
            "source_runtime_inputs_by_phase": {phase: asdict(source_runtime_inputs_by_phase[phase]) for phase, _ in phase_requests},
            "model_id": model.model_id, "model_revision": model.model_revision,
            "model_execution_version": runtime_field("model_execution_version"),
            "v4_eager_metadata_source": runtime_field("v4_eager_metadata_source"),
            "v4_source_max_model_len": runtime_field("v4_source_max_model_len"),
            "v4_source_use_deep_gemm": runtime_field("v4_source_use_deep_gemm"),
            "fp8_runner_source_links": _fp8_runner_source_links(results),
            "compressor_state_source_links": _compressor_state_source_links(results),
            "source_phase_requests_supplied": source_phase_requests is not None,
            "v4_routing_source": runtime_field("v4_routing_source"),
            "deployment": asdict(parallel), "mtp": "off",
            "scope": "workload source provenance graph only; no lowering, launch selection or prediction",
            "lowering_executed": False, "prediction_executed": False,
            "actual_runtime_or_loaded_weight_binding": False,
            "normal_kv_capacity_accepted": False,
            "steps": results,
        })
    return _json_value({
        "snapshot": _snapshot(), "model_config_path": model_path,
        "probe_workload_path": "symbolic_forward",
        "source_normalization": source_normalization,
        "static_output_scope": static_output_scope,
        "source_runtime_inputs_by_phase": {phase: asdict(source_runtime_inputs_by_phase[phase]) for phase, _ in phase_requests},
        "model_id": model.model_id, "model_revision": model.model_revision,
        "hardware_id": "nvidia_h200", "method": "roofline",
        "mtp": "on" if mtp_mode == "native" else "off",
        "prefill_tokens": prefill_tokens,
        "v4_eager_metadata_source": runtime_field("v4_eager_metadata_source"),
        "v4_source_max_model_len": runtime_field("v4_source_max_model_len"),
        "v4_source_use_deep_gemm": runtime_field("v4_source_use_deep_gemm"),
        "fp8_runner_source_links": _fp8_runner_source_links(results),
        "compressor_state_source_links": _compressor_state_source_links(results),
        "source_phase_requests_supplied": source_phase_requests is not None,
        "v4_routing_source": runtime_field("v4_routing_source"),
        "v4_routing_requirements": {
            "scope": "conditional routing source projection; hash priced only with explicit modular CUDA scenario",
            "assumptions": "modular CUDA; contiguous FP32 logits/bias; normalized top6; INT32 output IDs; no VL/padding; hash scenario also assumes INT32 input IDs/table without conversion",
            "actual_provider_compiler_runtime_bound": False,
            "normal_kv_capacity_accepted": False,
        } if runtime_field("v4_routing_source") is not None else None,
        "model_execution_version": runtime_field("model_execution_version"),
        "mtp_draft_sampling_source": runtime_field("mtp_draft_sampling_source"),
        "target_sampling_source": runtime_field("target_sampling_source"),
        "embedding_token_ids_by_phase": {phase: source_runtime_inputs_by_phase[phase].embedding_token_ids for phase, _ in phase_requests},
        "embedding_token_input_scope": (source_normalization["token_input_scope"]
            if source_normalization is not None and "token_input_scope" in source_normalization else
            "explicit caller source inputs in scheduled row order; not observed model samples or runtime trace"),
        "target_sampling_requirements": ({
            "scope": "declared all-greedy source chain; not observed SamplingMetadata or sampler/model execution",
            "requires": "no logprobs/logprob-token-ids, active logits processors, penalties, masks or thinking-budget mutation",
            "stages": ("BF16_to_FP32_logits", "FP32_argmax_INT64_ID", "INT64_to_INT32_output"),
            "aliases": ("sampled.long()", "output.unsqueeze(-1)"),
            "excluded_after_greedy_return": ("temperature", "min_p", "top_k_top_p", "random_sampling"),
            "actual_provider_runtime_bound": False,
        } if runtime_field("target_sampling_source") in (
            "vllm_0_11_plain_greedy_bf16", "vllm_0_30_plain_greedy_bf16", "vllm_0_27_1_plain_greedy_bf16") else None),
        "native_weight_source_requirements": ({
            "scope": "explicit legacy full-logits source projection, not observed proposer dispatch",
            "decoder_weights": "exact per-module FP8/BF16 rules at the appended MTP layer prefix",
            "merge_projection": "BF16 nn.Linear outside quantizer selection",
            "draft_head": "llm_base_proposer._maybe_share_lm_head aliases target lm_head, preserving its BF16 exclusion",
            "actual_loading_provider_and_runtime_bound": False,
        } if mtp_mode == "native" and model.config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S else None),
        "pipeline_source_requirements": {
            "VLLM_PP_LAYER_PARTITION": pipeline_layer_partition,
            "pipeline_link_scenario": pipeline_link,
            "physical_topology_observed": False,
            "normal_kv_capacity_accepted": False,
            "scheduler_runtime_accepted": False,
            "transport_semantics_accepted": False if pipeline_parallel_size == 2 else None,
            "modeled_boundary_reduction_order": (
                "receiver_stage_before_input_norm" if pipeline_parallel_size == 2 else None),
            "modeled_tp_partial_producer_scope": (
                "All modeled rank-local source math materialized on every TP rank, "
                "including replicated indexer/projections and TP-sharded attention/MLP/head; "
                "actual compiler/runtime/transport bindings remain incomplete"
                if pipeline_parallel_size == 2 else
                "The same complete virtual target program runs on each physical rank; "
                "logical pricing is recorded per call, independently of native launch support; "
                "actual compiler/runtime/provider and normal allocator remain unbound"),
            "kimi_ep_source_predicates": (
                {"dp_size": 1, "pcp_size": 1, "sp_size": 1,
                 "use_sequence_parallel_moe": False,
                 "use_all2all_kernels": False, "do_naive_dispatch_combine": False,
                 "source_symbol": "FusedMoEParallelConfig.use_all2all_kernels / MoERunner.do_naive_dispatch_combine",
                 "actual_backend_selection": None,
                 "scope": "explicit_non_sequence_parallel_source_scenario_not_observed_runtime"}
                if model.model_type == "kimi_k2" and tensor_parallel_size == 8
                and enable_expert_parallel else None),
            "v4_ffn_source_predicates": (
                {"use_mega_moe": False, "use_sequence_parallel": False,
                 "enable_expert_parallel": False, "ep_rank": 0,
                 "factory_expert_ownership": "all_experts_per_tensor_rank",
                 "factory_intermediate_per_rank": model.moe_intermediate_size // tensor_parallel_size,
                 "source_symbol": "DeepseekV4MoE._init_fused_moe_experts / FusedMoEParallelConfig.make",
                 "actual_backend_selection": None,
                 "parent_metadata_and_loaded_weight_binding_validated": False,
                 "scope": "non_MegaMoE_EP_off_source_branch_not_observed_runtime"}
                if model.model_type == "deepseek_v4" and tensor_parallel_size > 1
                and not enable_expert_parallel else None),
            "transport_semantics_gap": (
                "V32 leaves MLP hidden_states TP-partial (reduce_results=False); "
                "GPUWorker default TP shard/send/all-gather requires replicated inputs. "
                "No actual override or compiler transformation preserving partials is bound. "
                "The two payloads and wire bytes describe the source-default branch only; "
                "its priced calls do not prove a numerically valid PP model."
                if pipeline_parallel_size == 2 else None),
        },
        "pipeline_weight_projection": pp_weight_projection,
        "scope": "static modeled evidence; requires independent comparison",
        "on_mode_status": ("priced_declared_logical_source_chain" if source_runtime_inputs_by_phase is not None
            and all(step["roofline_end_to_end"] == "priced" for step in results) else "partial_source_projection")
            if mtp_mode == "native" else "not_executed_by_this_probe",
        "draft_architecture": "DeepseekV32MTPModel" if mtp_mode == "native" else None,
        "native_source_gaps": (
            ("exact_0_30_compiler_launch_binding_unavailable",
             "target_fusion_projection_actual_launch_and_slot_mapping_unvalidated",
             "independent_declared_draft_inputs_not_checkpoint_samples_or_runtime_buffer_observations",
             "actual_slot_mapping_and_normal_capacity_not_validated",
             "draft_backend_independent_autoselection_and_group_metadata_build_unbound",
             "runtime_draft_sampler_dispatch_not_bound") if mtp_mode == "native" else ()),
        "routing": "explicit_per_phase_runtime_inputs_not_observed",
        "deployment": asdict(parallel),
        "steps": results,
    })


def _glm_official_fp8_source_inputs(tp, mode):
    """Explicit synthetic comparison inputs; never model-generated outputs.

    Every phase is standalone, including empty workspace history. Padded
    first-pass stale/anchor IDs and recurrent argmax IDs are external scenario
    values; only the source's known non-anchor shift is preserved here.
    """
    routing = ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0)
    runtimes = {}
    for phase, requests in _operator_probe_phase_requests(8, mode):
        rows = sum(request.scheduled_tokens for request in requests)
        target_ids = tuple(range(1, rows + 1))
        state = None
        if mode == "native":
            state = _declared_glm_padded_probe_state(ScheduleStep(phase, "main", requests), target_ids,
                source_id="official-family-independent-padded-input-scenario")
        runtimes[phase] = RuntimeInputs(routing, "mla", "uniform", model_execution_version="vllm_0_30_0",
            symbolic_forward_source="vllm_0_30_v32_fp8_sync_explicit_allreduce",
            target_sampling_source="vllm_0_30_rejection_raw_logits_bf16" if mode == "native" else "vllm_0_30_plain_greedy_bf16",
            mtp_draft_sampling_source="vllm_0_30_legacy_full_logits_greedy" if mode == "native" else None,
            symbolic_source_max_model_len=128, symbolic_source_max_batched_tokens=128,
            symbolic_source_max_requests=2 if mode == "native" else None,
            symbolic_workspace_initial_capacity_bytes=(0,) * tp,
            glm_indexer_selection_source="declared_causal_topk",
            glm_prefill_attention_source="force_sparse_mqa" if mode == "native" else None,
            embedding_token_ids=target_ids, model_source_inputs=state)
    return runtimes


def _glm_official_fp8_probe(arguments):
    path, tp, mode = arguments
    return build_model_operator_probe(path, tp, False, mtp_mode=mode,
        source_runtime_inputs_by_phase=_glm_official_fp8_source_inputs(tp, mode))


def write_glm_official_fp8_family_comparison(output, *, tensor_parallel_size, mtp_mode):
    """Compare the official pair under identical declared source workloads."""
    identity_keys = frozenset((
        "model_id", "model_revision", "config_sha256", "source_config_sha256",
        "canonical_config_sha256", "extracted_config_sha256", "model_source_url", "model_source_revision"))

    def normalized(value):
        if isinstance(value, Mapping):
            return {key: normalized(item) for key, item in value.items() if key not in identity_keys}
        if isinstance(value, (tuple, list)):
            return [normalized(item) for item in value]
        return value

    paths = ("configs/model/glm-5.2-fp8.json", "configs/model/glm-5.3.json")
    models = tuple(load_model_config(path) for path in paths)
    with ProcessPoolExecutor(max_workers=2) as executor:
        probes = tuple(executor.map(_glm_official_fp8_probe,
                                    ((path, tensor_parallel_size, mtp_mode) for path in paths)))
    phases = []
    for first, second in zip(probes[0]["steps"], probes[1]["steps"], strict=True):
        phases.append({
            "phase": first["phase"], "logical_nodes": first["logical_nodes"], "kernel_calls": first["kernel_calls"],
            "unpriced_calls": first["unpriced_calls"], "roofline_end_to_end": first["roofline_end_to_end"],
            "graph_equal": normalized(first["nodes"]) == normalized(second["nodes"]),
            "lowered_calls_and_predictions_equal": normalized(first["calls"]) == normalized(second["calls"]),
            "scheduled_inputs_equal": first["scheduled_requests"] == second["scheduled_requests"],
            "unpriced_counts_equal": first["unpriced_calls"] == second["unpriced_calls"],
        })
    parallel = ParallelConfig(tensor_parallel_size=tensor_parallel_size)
    capacity = []
    for model in models:
        terms = {
            "loader_with_mla": glm_block_fp8_loader_payload_bytes(model, parallel, include_mla_absorption=True),
            "postload_with_mla": glm_block_fp8_postload_parameter_bytes(
                model, parallel, hardware_id="nvidia_h200", linear_backend="deep_gemm", routed_backend="deep_gemm",
                include_mla_absorption=True),
            "constructor_buffers": glm_block_fp8_constructor_buffer_bytes(model, parallel, max_batched_tokens=128, max_requests=2),
            "shared_workspace": glm_block_fp8_deepgemm_workspace_bytes(model, parallel, hardware_id="nvidia_h200", tokens=128),
            "kv_bytes_per_token_per_rank": model.kv_bytes_per_token_per_parallel_rank(tensor_parallel_size, 1),
        }
        capacity.append(terms)
    checks = {
        "separate_checkpoint_identities": models[0].model_id != models[1].model_id and models[0].config_sha256 != models[1].config_sha256,
        "shared_assumption_semantics": models[0].expert_routing_assumption_semantics_sha256 == models[1].expert_routing_assumption_semantics_sha256,
        "source_capacity_terms_equal": capacity[0] == capacity[1],
        "all_graphs_equal": all(phase["graph_equal"] for phase in phases),
        "all_lowering_and_prediction_records_equal": all(phase["lowered_calls_and_predictions_equal"] for phase in phases),
        "all_scheduled_inputs_equal": all(phase["scheduled_inputs_equal"] for phase in phases),
        "all_unpriced_counts_equal": all(phase["unpriced_counts_equal"] for phase in phases),
    }
    result = {
        "snapshot": _snapshot(), "status": "source_comparison_only_not_acceptance", "checks": checks,
        "failed": sum(not value for value in checks.values()), "tensor_parallel_size": tensor_parallel_size, "mtp_mode": mtp_mode,
        "hardware_id": "nvidia_h200", "source_execution": "vllm_0_30_0_eager_PP1_SP1_EP_off",
        "probe_workload_paths": [probe["probe_workload_path"] for probe in probes],
        "source_workload_assumptions": {
            "profile": "standalone_shared_virtual_target_frames",
            "max_model_len": 128, "max_batched_tokens": 128,
            "workspace_entry_bytes_per_rank": [0] * tensor_parallel_size,
            "workspace_scope": "declared fresh workspace per independent phase, not observed history or normal capacity",
            "token_ids": "explicit synthetic sequential IDs per phase, not checkpoint outputs",
            "indexer": "declared causal top-k cardinality",
            "sampler": "vllm_0_30_rejection_raw_logits_bf16" if mtp_mode == "native" else "vllm_0_30_plain_greedy_bf16",
            "padded_draft_inputs": "independent declared synthetic/stale IDs; no numerical draft/sampler execution" if mtp_mode == "native" else None,
            "prefill_attention_source": "force_sparse_mqa" if mtp_mode == "native" else None},
        "source_capacity_inventory_mtp_mode": "off",
        "identities": [{key: getattr(model, key) for key in ("model_id", "model_revision", "config_sha256", "model_source_url")} for model in models],
        "phase_comparisons": phases,
        "source_capacity_terms": [{"model_id": model.model_id, "terms": terms} for model, terms in zip(models, capacity, strict=True)],
        "runtime_acceptance": False, "normal_capacity_accepted": False,
        "limitations": [
            "same explicitly declared routing assumption; real replay/profile retains checkpoint binding",
            "capacity terms cover the same target-off source inventory/profile, not complete native or allocator-ready residency",
            "target-off logical pricing does not establish native-MTP migration, actual provider/runtime or normal capacity",
            "old derived-local per-tensor GLM5.2-FP8 artifacts are excluded from official acceptance",
            "main agent owns final acceptance; no old snapshot receipt inheritance"],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_value(result), ensure_ascii=False, indent=2) + "\n")
    return result


def require_logical_roofline_forward(payload):
    """Require a fully priced four-phase forward, never native acceptance."""
    if "static_output_scope" in payload and payload["static_output_scope"] == "lm_head_only_no_sampler":
        raise ValueError("static LM-head-only probe omits the sampler and cannot establish a complete logical request forward")
    if "static_output_scope" in payload and payload["static_output_scope"] == "fp32_logits_only":
        raise ValueError("static FP32 logits cast omits sampling and cannot establish a complete logical request forward")
    if ("source_runtime_inputs_by_phase" in payload and payload["source_runtime_inputs_by_phase"] is not None
            and any(runtime["target_sampling_source"] in (None, "vllm_0_11_plain_sampler_bf16", "vllm_0_30_plain_sampler_bf16")
                for runtime in payload["source_runtime_inputs_by_phase"].values())):
        raise ValueError("static logits-only source omits the sampler and cannot establish a complete logical request forward")
    if payload["mtp"] not in ("off", "on"):
        raise ValueError("complete logical forward check requires target-off or an explicit complete GLM MTP source")
    if payload["mtp"] == "on":
        if ("probe_workload_path" not in payload or payload["probe_workload_path"] != "symbolic_forward"
                or "source_runtime_inputs_by_phase" not in payload or payload["source_runtime_inputs_by_phase"] is None):
            raise ValueError("complete GLM MTP logical forward requires explicit per-phase common source inputs")
        for step in payload["steps"]:
            runtime = payload["source_runtime_inputs_by_phase"][step["phase"]]
            if (runtime["symbolic_forward_source"] not in (
                    "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
                    or runtime["mtp_draft_sampling_source"] != "vllm_0_30_legacy_full_logits_greedy"
                    or runtime["target_sampling_source"] != "vllm_0_30_rejection_raw_logits_bf16"
                    or runtime["model_source_inputs"] is None or step["roofline_end_to_end"] != "priced"):
                raise ValueError("complete GLM MTP logical forward requires its paired typed target/proposer source chain")
    if tuple(step["phase"] for step in payload["steps"]) != ("prefill", "decode1", "decode2", "mixed"):
        raise ValueError("complete logical forward requires all four phases")
    for step in payload["steps"]:
        calls = step["calls"]
        if (type(step["kernel_calls"]) is not int or step["kernel_calls"] != len(calls)
                or not calls or len({call["call_id"] for call in calls}) != len(calls)):
            raise ValueError("complete logical forward call coverage is inconsistent")
        if step["roofline_end_to_end"] == "source_projection_only":
            raise ValueError("PP source projection cannot establish a complete logical forward")
        if (type(step["unpriced_calls"]) is not int or step["unpriced_calls"] != 0
                or step["modeled_calls_pricing"] != "priced"):
            raise ValueError("complete logical forward still has unpriced calls")
        for call in calls:
            prediction = call["prediction"]
            if prediction["method"] != "roofline" or prediction["status"] not in ("priced", "logical_alias"):
                raise ValueError("complete logical forward requires every call to have a Roofline price")
            seconds = prediction["seconds"]
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not isfinite(seconds) or seconds < 0):
                raise ValueError("complete logical forward has an invalid call duration")
        timeline = step["logical_roofline_timeline"]
        if timeline is None:
            raise ValueError("complete logical forward lacks a complete positive DAG timeline")
        duration = timeline["duration_s"]
        if (type(timeline["call_count"]) is not int or timeline["call_count"] != len(calls)
                or isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not isfinite(duration) or duration <= 0):
            raise ValueError("complete logical forward lacks a complete positive DAG timeline")


def logical_roofline_summary(payload):
    """Compact user result from a completely priced logical execution DAG.

    Device durations come from the production timeline. Rank-local kernel
    durations are never summed to produce a model latency. This report does
    not upgrade source-only inputs to Scheduler, capacity or GPU acceptance.
    """
    require_logical_roofline_forward(payload)
    result = {field: payload[field] for field in (
        "snapshot", "model_config_path", "model_id", "model_revision", "hardware_id", "method",
        "mtp", "prefill_tokens", "deployment", "model_execution_version", "routing",
        "target_sampling_source", "target_sampling_requirements", "embedding_token_ids_by_phase",
        "embedding_token_input_scope", "v4_eager_metadata_source", "v4_source_max_model_len",
        "v4_source_use_deep_gemm", "v4_routing_source", "v4_routing_requirements")}
    if "source_runtime_inputs_by_phase" in payload:
        result["source_runtime_inputs_by_phase"] = payload["source_runtime_inputs_by_phase"]
        result["probe_workload_path"] = payload["probe_workload_path"]
    if "source_normalization" in payload:
        result["source_normalization"] = payload["source_normalization"]
    if "static_output_scope" in payload:
        result["static_output_scope"] = payload["static_output_scope"]
    result["scope"] = "static logical Roofline; declared inputs and routing, not model generation or Serving"
    result["steps"] = []
    for step in payload["steps"]:
        timeline = step["logical_roofline_timeline"]
        result["steps"].append({
            **{field: step[field] for field in ("phase", "scheduled_requests", "logical_nodes", "kernel_calls",
                "unpriced_calls", "roofline_end_to_end", "modeled_calls_pricing")},
            "logical_roofline_timeline": timeline,
            "latency_ms": timeline["duration_s"] * 1000,
            "native_unsupported_calls": sum(call["execution_support"] == "unsupported" for call in step["calls"]),
        })
    result["limitations"] = [
        "FLOPs/bytes baseline; instruction/control, compiler, cache and physical traffic costs are not fully modeled.",
        "Source/operator alignment remains incomplete; fully priced calls do not establish complete source coverage.",
        "No real Scheduler, normal KV capacity, QPS16, model-generated tokens or hardware accuracy accepted.",
        "Phase durations are individual model steps; mixed is a separate batch and must not be added to single-request latency.",
    ]
    return result


def write_model_operator_probe(
    model_path: str,
    tensor_parallel_size: int,
    enable_expert_parallel: bool,
    output: Path,
    *,
    mtp_mode: str = "off",
    prefill_tokens: int = 8,
    v4_eager_metadata: bool = False,
    v4_routing_source: str | None = None,
    pipeline_parallel_size: int = 1,
    pipeline_link: str | None = None,
    pipeline_layer_partition: str | None = None,
    source_graph_only: bool = False,
    mtp_draft_sampling_source: str | None = None,
    target_sampling_source: str | None = None,
    embedding_token_ids_by_phase: Mapping[str, tuple[int, ...]] | None = None,
    v4_source_max_model_len: int | None = None,
    v4_source_max_batched_tokens: int | None = None,
    v4_indexer_selection_source: str | None = None,
    v4_source_use_deep_gemm: bool | None = None,
    summary_only: bool = False,
    source_runtime_inputs_by_phase: Mapping[str, RuntimeInputs] | None = None,
    source_normalization: Mapping[str, object] | None = None,
) -> None:
    if type(summary_only) is not bool:
        raise TypeError("summary_only must be boolean")
    if source_normalization is not None:
        if not isinstance(source_normalization, Mapping) or "target_sampling_source" not in source_normalization:
            raise ValueError("source normalization requires the versioned sampler derivation")
        source, expected = normalize_model_operator_probe_source(model_path, source_normalization["target_sampling_source"])
        if (expected is None or dict(source_normalization) != expected
                or not isinstance(source_runtime_inputs_by_phase, Mapping) or not source_runtime_inputs_by_phase
                or any(not isinstance(runtime, RuntimeInputs) or runtime.symbolic_forward_source != source
                    or runtime.target_sampling_source != source_normalization["target_sampling_source"]
                    for runtime in source_runtime_inputs_by_phase.values())):
            raise ValueError("source normalization must match every declared phase and cannot promote provider acceptance")
    payload = build_model_operator_probe(
        model_path, tensor_parallel_size, enable_expert_parallel,
        mtp_mode=mtp_mode, prefill_tokens=prefill_tokens,
        v4_eager_metadata=v4_eager_metadata, v4_routing_source=v4_routing_source,
        v4_source_max_model_len=v4_source_max_model_len,
        v4_source_max_batched_tokens=v4_source_max_batched_tokens,
        v4_indexer_selection_source=v4_indexer_selection_source,
        v4_source_use_deep_gemm=v4_source_use_deep_gemm,
        pipeline_parallel_size=pipeline_parallel_size,
        pipeline_link=pipeline_link, pipeline_layer_partition=pipeline_layer_partition,
        source_graph_only=source_graph_only,
        mtp_draft_sampling_source=mtp_draft_sampling_source,
        target_sampling_source=target_sampling_source,
        embedding_token_ids_by_phase=embedding_token_ids_by_phase,
        source_runtime_inputs_by_phase=source_runtime_inputs_by_phase)
    if source_normalization is not None:
        if payload["source_normalization"] is None:
            payload["source_normalization"] = dict(source_normalization)
        else:
            payload["source_normalization"]["cli_source_derivation"] = dict(source_normalization)
    if summary_only:
        payload = logical_roofline_summary(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_glm_gpu_v2_storage_probe(model_path, tensor_parallel_size, output, *, prefill_context_parallel_size):
    """Export the production owner's unpriced source roles for an N3 case.

    The explicit acceptance array is the three-request source conservation
    scenario, not a measured acceptance probability. No model forward,
    launch selector, Scheduler, allocation or prediction executes here.
    """
    model = load_model_config(model_path)
    if tensor_parallel_size not in (1, 8) or type(tensor_parallel_size) is not int:
        raise NotImplementedError("GLM storage source probe covers TP1/TP8 only")
    requests = tuple(ScheduledRequest(
        f"request-{index}", "decode", 0, 4, 64,
        raw_model_output_tokens=accepted + 1,
        mtp_proposed_tokens=3, mtp_verified_tokens=3,
        mtp_accepted_tokens=accepted, mtp_proposer_tokens=3,
        mtp_next_proposed_tokens=3, mtp_proposer_mode="padded",
        mtp_acceptance_source_kind="explicit_assumption",
        mtp_acceptance_source_id="three_request_source_conservation_array")
        for index, accepted in enumerate((0, 2, 3)))
    runtime = RuntimeInputs(
        ExpertRoutingInput.seeded_assumption("unused_storage_only_routing", 0),
        "mla", "uniform", model_execution_version="vllm_0_30_0",
        mtp_proposer_execution_source="vllm_0_30_gpu_v2")
    work = glm_gpu_v2_mtp_storage_work(model, requests, runtime,
        prefill_context_parallel_size=prefill_context_parallel_size)
    # Separate explicit state cases, not metadata observed for the above
    # scheduled requests. Production owner derives all states/role bytes.
    prepare_scenario = {
        "target_seq_lens": (13, 16, 8), "num_rejected": (2, 0, 3),
        "positions": (10, 15, 4), "sample_src_positions": (11, 41, 90),
        "max_num_requests": 8, "max_model_len": 16, "proposal_depth": 3,
        "num_rejected_dtype": "int32",
    }
    prepare_cases = []
    for advance in (True, False):
        scenario = {**prepare_scenario, "advance_draft_positions": advance}
        prepare_cases.append({"input_scenario": scenario,
                              "work": glm_gpu_v2_prepare_decode_work(**scenario)})
    slot_scenario = {
        "idx_mapping": (0, 1, 2), "query_start_loc": (0, 1, 2, 3),
        "positions": (11, 15, 5),
        "block_tables_by_group": (
            ((10, 11, 12, 13), (20, 21, 22, 23), (30, 31, 32, 33)),
            ((40, 41, 42, 43), (50, 51, 52, 53), (60, 61, 62, 63))),
        "kv_block_sizes": (16, 16), "kernel_block_sizes": (4, 8),
        "mapping_enabled": (True, False), "slot_capacity": 16,
        "returned_tokens": 4, "cp_size": 1,
    }
    payload = {
        "snapshot": _snapshot(), "model_config_path": model_path,
        "model_type": model.model_type, "tensor_parallel_size": tensor_parallel_size,
        "prefill_context_parallel_size": prefill_context_parallel_size,
        "status": "source_storage_subset_not_full_native_or_priced_model",
        "independent_acceptance": False, "new_gpu_execution": False,
        "normal_gpu_v2_workload_supported": False,
        "scheduled_requests": [asdict(request) for request in requests],
        "logical_work": work,
        "prepare_decode_source_scenarios": prepare_cases,
        "slot_mapping_source_scenario": {
            "input_scenario": slot_scenario,
            "work": glm_gpu_v2_slot_mapping_work(**slot_scenario)},
        "metadata_refresh_source_scenarios": tuple(glm_gpu_v2_metadata_refresh_plan(
            proposal_depth=work["proposal_depth"], metadata_update_supported_by_groups=capabilities)
            for capabilities in ((True, True), (True, False))),
        "sparse_backend_metadata_source_constraints": tuple(glm_gpu_v2_sparse_backend_metadata_plan(
            proposal_depth=work["proposal_depth"], backend_name=backend)
            for backend in ("FLASH_ATTN_MLA_SPARSE", "FLASHMLA_SPARSE", "FLASHINFER_MLA_SPARSE_SM90")),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _rescore_formal_layer_shapes(comparison):
    """Replay external shape obligations on current facts; create no expectations."""
    records = []
    required = passed = failed = 0
    for record in comparison["records"]:
        if not record["checks"]:
            records.append({"model": record["model"], "status": record["status"],
                            "reason": record["reason"]})
            continue
        probe = json.loads(Path(record["artifact"]).read_text(encoding="utf-8"))
        phase_calls = {step["phase"]: step["calls"] for step in probe["steps"]}
        checks = []
        for obligation in record["checks"]:
            call_id = (f"{obligation['phase']}:layer-{obligation['layer']}"
                       f"{obligation['operation']}")
            actual = [call for call in phase_calls[obligation["phase"]]
                      if call["rank"] == record["rank"] and call["kind"] == "GEMM" and (
                          call["call_id"] == call_id
                          or call["call_id"].startswith(call_id + ":"))]
            expected = obligation["expected_shape"]
            ok = len(actual) == 1 and set(expected) <= set(actual[0]["shape"])
            if ok:
                ok = all(actual[0]["shape"][key] == value
                         for key, value in expected.items())
            checks.append({"call_id": call_id, "pass": ok})
        count = sum(check["pass"] for check in checks)
        records.append({"model": record["model"], "artifact": record["artifact"],
                        "snapshot": probe["snapshot"], "rank": record["rank"],
                        "passed": count, "failed": len(checks) - count,
                        "checks": checks})
        required += len(checks)
        passed += count
        failed += len(checks) - count
    if required != comparison["totals"]["required_shape_checks"]:
        raise ValueError("formal layer shape denominator changed during replay")
    return {"totals": {"required_shape_checks": required, "passed": passed, "failed": failed},
            "records": records, "independent_acceptance": False,
            "scope": "replay of existing external rank0 shape expectations; no new expectation, complete launch or runtime acceptance"}


def _rescore_v4_compressor_bytes(comparison):
    """Compare current facts to the external source object's byte lower bound."""
    records = []
    passed = failed = 0
    for record in comparison["records"]:
        probe = json.loads(Path(record["artifact"]).read_text(encoding="utf-8"))
        calls = {call["call_id"]: call for step in probe["steps"] for call in step["calls"]}
        checks = []
        for obligation in record["checks"]:
            call_id = (f"{obligation['phase']}:layer-{obligation['layer']}:"
                       f"{obligation['operation']}")
            call = calls[call_id]
            if obligation["operation"] in ("compressor-kv-score", "indexer-compressor-kv-score"):
                ok = (call["dtype"] == "bf16" and call["shape"]["dtype_bytes"] == 2
                      and call["attributes"]["output_dtype"] == "fp32"
                      and call["resource_input"]["logical_bytes"] >=
                      obligation["expected_unique_tensor_bytes"]["minimum_sum"])
            elif obligation["operation"] in ("compressor-state-update", "indexer-compressor-state-update"):
                expected_bytes = sum(obligation["expected_source_role_bytes_all_valid_projection"].values())
                arithmetic = obligation["expected_arithmetic"]
                demand = call["resource_input"]["demand"]
                ok = (call["dtype"] == "fp32" and call["shape"]["dtype_bytes"] == 4
                      and call["resource_input"]["logical_bytes"] == expected_bytes
                      and demand["compute_fma"] == arithmetic["compute_fma"]
                      and demand["compute_alu"] == arithmetic["score_plus_APE_additions"] + arithmetic["position_modulo"])
            else:
                raise ValueError("unknown formal V4 compressor operation")
            checks.append({"call_id": call_id, "pass": ok})
        count = sum(check["pass"] for check in checks)
        passed += count
        failed += len(checks) - count
        records.append({"model": record["model"], "snapshot": probe["snapshot"],
                        "passed": count, "failed": len(checks) - count, "checks": checks})
    if passed + failed != comparison["totals"]["checks"]:
        raise ValueError("formal V4 compressor denominator changed during replay")
    return {"records": records, "totals": {"checks": passed + failed, "passed": passed, "failed": failed},
            "independent_acceptance": False,
            "scope": "existing external source byte lower bound and dtype assertions only; actual launch, local services and full W3 remain unvalidated"}


def _validate_grouped_topk_finite_precision_receipt(document):
    """Validate source witness fields before publishing any checkpoint files."""
    message = "formal source review finite-precision receipt"
    for key in ("accepted_scope", "source_reference_qualification"):
        if key not in document or not isinstance(document[key], str) or not document[key].strip():
            raise ValueError(f"{message} requires nonempty {key}")
    for key in ("examples", "findings", "limits"):
        if key not in document or not isinstance(document[key], list) or not document[key]:
            raise ValueError(f"{message} requires nonempty {key} list")
    if any(not isinstance(value, str) or not value.strip() for value in document["limits"]):
        raise ValueError(f"{message} limits must be nonempty strings")
    for finding in document["findings"]:
        if not isinstance(finding, dict) or any(
                key not in finding or not isinstance(finding[key], str) or not finding[key].strip()
                for key in ("item", "required_action")):
            raise ValueError(f"{message} findings require item and required_action strings")
    for example in document["examples"]:
        if not isinstance(example, dict):
            raise ValueError(f"{message} examples must be objects")
        for key in ("experts", "topk"):
            if key not in example or type(example[key]) is not int or example[key] <= 0:
                raise ValueError(f"{message} example requires positive {key}")
        if example["topk"] > example["experts"]:
            raise ValueError(f"{message} topk cannot exceed experts")
        for key in ("logits_all_zero", "same_value"):
            if key not in example or type(example[key]) is not bool:
                raise ValueError(f"{message} example requires boolean {key}")
        for key in ("correction_bias_all", "selection_score_fp32", "source_block_unbiased",
                    "source_warp_recovered_unbiased", "block_weight_each_fp32", "warp_weight_each_fp32"):
            if key not in example or type(example[key]) not in (int, float) or not isfinite(example[key]):
                raise ValueError(f"{message} example requires finite numeric {key}")
        for key in ("condition", "dispatch_transition"):
            if key not in example or not isinstance(example[key], str) or not example[key].strip():
                raise ValueError(f"{message} example requires nonempty {key}")
        if example["same_value"] is not (example["block_weight_each_fp32"] == example["warp_weight_each_fp32"]):
            raise ValueError(f"{message} same_value contradicts stored weights")
    if not any(example["same_value"] is False and
               example["block_weight_each_fp32"] != example["warp_weight_each_fp32"]
               for example in document["examples"]):
        raise ValueError(f"{message} requires a differing-output source witness")


_MHC_SM90_REVIEW_FILENAMES = (
    "mhc_prenorm_sm90_review.json", "mhc_prenorm_sm90_host_binary_review.json",
    "mhc_prenorm_sm90_descriptor_review.json", "mhc_prenorm_sm90_model_delta_review.json",
    "mhc_prenorm_sm90_tma_review.json", "mhc_prenorm_sm90_tma_model_delta_review.json",
    "mhc_prenorm_sm90_zero_stage_review.json", "mhc_prenorm_sm90_zero_stage_model_delta_review.json",
    "operator_sm90_formal_publication_review.json")


def _validate_mhc_sm90_review(filename, document):
    """Validate local-only publication boundaries, never replay its formulas."""
    if document["no_GPU_or_download_or_push"] is not True:
        raise ValueError("formal SM90 source receipt requires its offline boundary")
    for key in ("actual_provider_bound", "pricing_or_launch_assignment_accepted"):
        if key in document and document[key] is not False:
            raise ValueError("formal SM90 source receipt cannot promote local provider/pricing")
    if filename in ("mhc_prenorm_sm90_review.json", "mhc_prenorm_sm90_tma_review.json",
                    "mhc_prenorm_sm90_zero_stage_review.json"):
        snapshots = (document["candidate_start"], document["candidate_end"])
        scope = document["accepted_scope"]
        direct_key = ("direct_scopes_passed" if filename == "mhc_prenorm_sm90_zero_stage_review.json"
                      else "own_direct_scopes_passed")
        counts = (document[direct_key], document["candidate_failures"],
                  document["new_counter_categories"])
        limits = document["limits"]
    elif filename in ("mhc_prenorm_sm90_host_binary_review.json",
                      "mhc_prenorm_sm90_descriptor_review.json"):
        host = filename == "mhc_prenorm_sm90_host_binary_review.json"
        snapshots = (document["publication_snapshot"] if host else
                     document["observed_publication_head"],)
        scope = [document["source_basis"] if host else document["accepted_basis"]]
        counts = (document["new_counter_categories"],)
        limits = document["not_accepted"] if host else document["limits"]
        if not isinstance(document["conditionally_accepted"], dict) or not document["conditionally_accepted"]:
            raise ValueError("formal SM90 source receipt requires conditional facts")
        if (document["actual_provider_bound"] is not False or
                document["pricing_or_launch_assignment_accepted"] is not False):
            raise ValueError("formal SM90 source receipt cannot promote local provider/pricing")
    elif filename in ("mhc_prenorm_sm90_model_delta_review.json",
                      "mhc_prenorm_sm90_tma_model_delta_review.json",
                      "mhc_prenorm_sm90_zero_stage_model_delta_review.json"):
        snapshots = tuple(document[key]["commit"] for key in ("before_snapshot", "candidate_snapshot"))
        if filename == "mhc_prenorm_sm90_model_delta_review.json":
            count_key, calls_key = "conditional_metadata_checks", "conditional_calls"
            scope = [document["coverage_change"]]
        elif filename == "mhc_prenorm_sm90_tma_model_delta_review.json":
            count_key, calls_key = "metadata_checks", "conditional_source_calls"
            scope = document["limits"]
        else:
            count_key = calls_key = "nonempty_partition_checks"
            scope = [document["scope"]]
        counts = (document[count_key],)
        limits = document["limits"]
        records = document["records"]
        if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
            raise ValueError("formal SM90 source receipt requires original phase records")
        if any(not isinstance(item[calls_key], list) for item in records):
            raise ValueError("formal SM90 source receipt requires original conditional calls")
        if document[count_key] != sum(len(item[calls_key]) for item in records):
            raise ValueError("formal SM90 source receipt conditional count disagrees")
        if filename == "mhc_prenorm_sm90_model_delta_review.json" and any(
                item["runtime_or_provider_accepted"] is not False for item in records):
            raise ValueError("formal SM90 source receipt cannot promote phase runtime/provider")
        for record in records:
            for call in record[calls_key]:
                if filename == "mhc_prenorm_sm90_tma_model_delta_review.json":
                    metadata = call["expected_source_copy_metadata"]
                    if (metadata["actual_instruction_count_bound"] is not False or
                            metadata["dram_bytes_bound"] is not False):
                        raise ValueError("formal SM90 source receipt cannot promote source calls to traffic/instructions")
                elif filename == "mhc_prenorm_sm90_zero_stage_model_delta_review.json":
                    partition = call["K_stage_partition"]
                    if not isinstance(partition, list) or not partition or any(
                            type(value) is not int or value <= 0 for value in partition):
                        raise ValueError("formal SM90 source receipt requires nonempty K partitions")
        if type(document["all_other_artifact_content_equal"]) is not bool:
            raise ValueError("formal SM90 source receipt requires original equality flag")
    elif filename == "operator_sm90_formal_publication_review.json":
        snapshots = (document["publication_snapshot"]["commit"],)
        scope = [document["accepted_scope"]]
        counts = (document["receipts_checked"], document["lost_fields_or_relabels"])
        limits = document["limits"]
        records = document["records"]
        if not isinstance(records, list) or document["receipts_checked"] != len(records):
            raise ValueError("formal SM90 source receipt publication count disagrees")
        if any(type(record["full_original_receipt_equal"]) is not bool for record in records):
            raise ValueError("formal SM90 source receipt requires original publication flags")
        if document["lost_fields_or_relabels"] == 0 and any(
                not record["full_original_receipt_equal"] for record in records):
            raise ValueError("formal SM90 source receipt cannot hide publication losses")
    else:
        raise ValueError("unknown formal SM90 source receipt")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{8,40}", value) is None for value in snapshots):
        raise ValueError("formal SM90 source receipt requires original snapshot identifiers")
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("formal SM90 source receipt requires nonnegative integer counts")
    if any(not isinstance(values, list) or not values or
           any(not isinstance(value, str) or not value for value in values) for values in (scope, limits)):
        raise ValueError("formal SM90 source receipt requires original scope and limitations")


def _validate_target_fused_embedding_review(document):
    """Keep source-only acceptance tied to the publisher's actual records."""
    message = "formal target embedding receipt"
    if document["no_GPU_or_download_or_push"] is not True:
        raise ValueError(f"{message} requires its offline boundary")
    if (not isinstance(document["source_reference"], str) or not document["source_reference"].strip()
            or not isinstance(document["remaining"], dict) or not document["remaining"]):
        raise ValueError(f"{message} requires its source reference and remaining limits")
    for field in ("direct_scopes_passed", "model_phases", "same_case_removed_calls", "candidate_failures"):
        if type(document[field]) is not int or document[field] < 0:
            raise ValueError(f"{message} requires nonnegative integer counts")
    snapshots = list(document["implementation_snapshots"])
    if not snapshots or not isinstance(document["direct_heads"], dict) or not document["direct_heads"]:
        raise ValueError(f"{message} requires original snapshot bindings")
    if set(document["direct_heads"]) != set(document["direct_dirty"]):
        raise ValueError(f"{message} requires matching head/dirty observation scopes")
    for scope, observation in document["direct_heads"].items():
        for boundary in ("before", "after"):
            snapshots.append(observation[boundary])
            changes = document["direct_dirty"][scope][boundary]
            if not isinstance(changes, list) or any(not isinstance(change, str) for change in changes):
                raise ValueError(f"{message} requires original dirty-file observations")
    records = document["records"]
    if not isinstance(records, list) or not records or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{message} requires original model records")
    for record in records:
        snapshots.extend(record[key]["commit"] for key in ("before_snapshot", "after_snapshot"))
        if not isinstance(record["phases"], list) or not record["phases"]:
            raise ValueError(f"{message} requires original phase records")
        for phase in record["phases"]:
            if any(not isinstance(phase[key], list) or any(not isinstance(value, str) for value in phase[key])
                   for key in ("removed", "changed")):
                raise ValueError(f"{message} requires original call-delta lists")
            if (type(phase["unpriced"]) is not int or phase["unpriced"] < 0
                    or (phase["unpriced"] > 0 and phase["roofline_end_to_end"] != "partial")):
                raise ValueError(f"{message} cannot promote an unpriced phase")
    if (document["model_phases"] != sum(len(record["phases"]) for record in records)
            or document["same_case_removed_calls"] != sum(
                len(phase["removed"]) for record in records for phase in record["phases"])):
        raise ValueError(f"{message} counts disagree with original records")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{8,40}", value) is None for value in snapshots):
        raise ValueError(f"{message} requires original snapshot identifiers")
    if (not isinstance(document["limits"], list) or not document["limits"]
            or any(not isinstance(value, str) or not value for value in document["limits"])
            or not isinstance(document["resolved_pending_embedding"], str)
            or not document["resolved_pending_embedding"].strip()):
        raise ValueError(f"{message} requires its original scope and limitations")


def _validate_residual_norm_review(filename, document):
    """Preserve independent residual receipts without upgrading their scope."""
    message = "formal residual norm receipt"
    if (document["no_GPU_or_download_or_push"] is not True
            or document["all_other_call_and_step_content_equal"] is not True):
        raise ValueError(f"{message} requires original offline/delta qualifications")
    if (not isinstance(document["source_reference"], str) or not document["source_reference"]
            or not isinstance(document["snapshot_qualification"], str) or not document["snapshot_qualification"]
            or not isinstance(document["remaining"], dict) or not document["remaining"]
            or not isinstance(document["remaining"]["item"], str) or not document["remaining"]["item"].strip()
            or not isinstance(document["limits"], list) or not document["limits"]
            or any(not isinstance(limit, str) or not limit for limit in document["limits"])):
        raise ValueError(f"{message} requires source, snapshot and remaining limitations")
    snapshots = [document["implementation"]]

    def snapshot(value):
        snapshots.append(value["commit"])
        dirty(value["uncommitted_changes"])

    def dirty(values):
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{message} requires original dirty observations")

    heads, changes = document["direct_heads"], document["direct_dirty"]
    if not isinstance(heads, dict) or not heads or not isinstance(changes, dict) or set(heads) != set(changes):
        raise ValueError(f"{message} requires matching direct head/dirty observations")
    records = document["records"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"{message} requires original phase records")
    if filename == "glm_residual_norm_review.json":
        snapshots.append(document["reporting_snapshot"])
        for scope, boundaries in heads.items():
            # The initial direct observation has only a captured before.
            # Its missing after must stay missing, as qualified by the receipt.
            if not isinstance(boundaries, dict) or "before" not in boundaries or set(boundaries) != set(changes[scope]):
                raise ValueError(f"{message} requires captured direct boundaries")
            for boundary, head in boundaries.items():
                snapshots.append(head)
                dirty(changes[scope][boundary])
        phases = []
        for record in records:
            snapshot(record["before_snapshot"])
            snapshot(record["candidate_snapshot"])
            if not isinstance(record["phases"], list) or not record["phases"]:
                raise ValueError(f"{message} requires original phase records")
            phases.extend(record["phases"])
        phase_count, changed_field = document["model_phase_count"], "changed_norms"
    elif filename == "kimi_residual_norm_review.json":
        for boundary, head in heads.items():
            snapshots.append(head)
            dirty(changes[boundary])
        for key in ("artifact_actual_snapshot", "before_snapshot", "after_snapshot"):
            snapshot(document[key])
        phases = records
        phase_count, changed_field = document["model_phases"], "changed_norm_calls"
    else:
        raise ValueError(f"{message} has an unknown source")
    counts = [document["direct_scopes_passed"], document["candidate_failures"],
              document["changed_norm_calls"], phase_count]
    for phase in phases:
        counts.extend(phase[key] for key in ("calls", changed_field, "added_logical_bytes", "unpriced"))
        if phase["roofline_end_to_end"] != "partial":
            raise ValueError(f"{message} cannot promote partial source phases")
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError(f"{message} requires nonnegative integer counts")
    if phase_count != len(phases) or document["changed_norm_calls"] != sum(phase[changed_field] for phase in phases):
        raise ValueError(f"{message} counts disagree with original records")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{8,40}", value) is None for value in snapshots):
        raise ValueError(f"{message} requires original snapshot identifiers")


def _validate_target_sampling_review(document, delta):
    """Validate the original sampler receipt and its linked phase deltas."""
    message = "formal target sampling receipt"
    if (document["no_GPU_or_download_or_push"] is not True
            or delta["no_GPU_or_download_or_push"] is not True
            or delta["all_other_artifact_content_equal"] is not True):
        raise ValueError(f"{message} requires original offline/source qualifications")
    if (not isinstance(document["source_reference"], str) or not document["source_reference"]
            or document["source_reference"] != delta["source_reference"]):
        raise ValueError(f"{message} requires a consistent source reference")
    for values in (document["accepted_scope"], document["limits"], delta["limits"]):
        if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{message} requires original scope and limits")
    if (not isinstance(document["snapshot_qualification"], str) or not document["snapshot_qualification"]
            or not isinstance(document["remaining"]["item"], str) or not document["remaining"]["item"]):
        raise ValueError(f"{message} requires snapshot and remaining qualifications")
    snapshots = list(document["implementations"])
    if not snapshots or set(document["direct_heads"]) != {"before", "after"} or set(document["direct_dirty"]) != {"before", "after"}:
        raise ValueError(f"{message} requires captured direct observations")
    for boundary in ("before", "after"):
        snapshots.append(document["direct_heads"][boundary])
        changes = document["direct_dirty"][boundary]
        if not isinstance(changes, list) or any(not isinstance(change, str) for change in changes):
            raise ValueError(f"{message} requires original dirty observations")
    checks = document["AST_scope_checks"]
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"{message} requires original helper binding checks")
    for check in checks:
        snapshots.extend((check["implementation"], check["direct_snapshot"]))
        if (check["function_AST_unchanged"] is not True
                or check["implementation"] not in document["implementations"]
                or check["direct_snapshot"] not in document["direct_heads"].values()):
            raise ValueError(f"{message} cannot inherit changed helper acceptance")
    counts = [document[key] for key in ("direct_scopes_passed", "candidate_failures", "new_counter_categories")]
    records = delta["records"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"{message} requires original model records")
    phases = []
    for record in records:
        for key in ("before_snapshot", "candidate_snapshot"):
            snapshots.append(record[key]["commit"])
            changes = record[key]["uncommitted_changes"]
            if not isinstance(changes, list) or any(not isinstance(change, str) for change in changes):
                raise ValueError(f"{message} requires original artifact dirty state")
        if not isinstance(record["phases"], list) or not record["phases"]:
            raise ValueError(f"{message} requires original phase records")
        phases.extend(record["phases"])
    projections = 0
    for phase in phases:
        counts.extend(phase[key] for key in ("before_calls", "after_calls", "added_materializations",
                                            "logical_bytes", "before_unpriced", "after_unpriced"))
        indices = phase["newly_bound_INT64_hidden_selection"]
        if not isinstance(indices, list) or any(not isinstance(value, str) for value in indices):
            raise ValueError(f"{message} requires original conditional projection IDs")
        projections += len(indices)
        if phase["roofline_end_to_end"] != "partial":
            raise ValueError(f"{message} cannot promote partial phases")
        if phase["after_calls"] - phase["before_calls"] != phase["added_materializations"]:
            raise ValueError(f"{message} call counts disagree")
    counts.extend(delta[key] for key in ("model_phases", "added_materializations", "new_declared_INT64_hidden_projections"))
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError(f"{message} requires nonnegative integer counts")
    if (delta["model_phases"] != len(phases)
            or delta["added_materializations"] != sum(phase["added_materializations"] for phase in phases)
            or delta["new_declared_INT64_hidden_projections"] != projections):
        raise ValueError(f"{message} totals disagree with original phases")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{8,40}", value) is None for value in snapshots):
        raise ValueError(f"{message} requires original snapshot identifiers")


def _formal_source_delta_reviews(reference: Path):
    """Optional published source receipts, preserving each original binding.

    These documents are independent reference output, not current-tree or
    runtime acceptance. Counts and scopes are copied; no predictor replay
    or inferred snapshot equivalence is performed here.
    """
    filenames = (
        "glm_greedy_sampling_comparison.json", "glm_legacy_materialization_comparison.json",
        "glm_parameter_storage_review.json", "target_logits_row_review.json",
        "glm_first_input_review.json", "glm_padded_preparation_review.json",
        "glm_legacy_step_metadata_published_review.json", "glm_initial_slot_mapping_review.json",
        "grouped_topk_offline_compile_review.json", "grouped_topk_full_translation_unit_review.json",
        "grouped_topk_host_registration_review.json", "grouped_topk_finite_precision_reference.json",
        "operator_finite_precision_publication_review.json", "operator_model_roofline_paths_review.json",
        "glm_target_fused_embedding_review.json", "glm_residual_norm_review.json", "kimi_residual_norm_review.json",
        "target_sampling_materialization_review.json",
        *_MHC_SM90_REVIEW_FILENAMES)
    # Optional fields belong to the published per-document receipt schema;
    # absent metadata stays absent and cannot become current-head acceptance.
    metadata_fields = (
        "candidate", "snapshot", "accepted_scope", "scope", "source_scope", "binding_limit", "limits", "limitations",
        "before_head", "before_dirty", "after_head", "after_dirty", "test_start", "test_end_observation",
        "after_observation", "after_observation_is_not_test_end_snapshot",
        "helper_same_as_test_start_committed_base", "branch_qualification",
        "current_source_branches_match_committed740", "direct_summary",
        "direct_checks_reused", "passing_tests_repeated", "final_candidate_or_goal_accepted",
        "full_module_compile", "device_subset_compile", "kernels", "independent_static_observations",
        "result", "command", "source_transform", "header_source", "production_metadata_change",
        "historical_failure_qualification", "source_receipts", "independent_checks", "ELF_needed_libraries",
        "own_reader_corrections", "reused_worker_host_observation", "independently_loaded_library", "closes",
        "source_reference_qualification", "examples", "findings",
        "candidate_observation", "source_metadata_implementation", "receipt_validation_implementation",
        "source_reference", "metadata_checks", "published_receipt", "source_scenarios_preserved",
        "own_reader_correction", "remaining", "publication", "reader_correction")
    count_fields = ("checks", "failed", "immutable_comparison_checks", "immutable_comparison_failed",
                    "artifact_checks", "artifact_failed", "model_layout_mode_records", "phase_records",
                    "recount_mismatches")
    records = []
    for filename in filenames:
        path = reference / filename
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise TypeError("formal source review must be a JSON object")
        if filename == "target_sampling_materialization_review.json":
            delta_path = Path(document["model_delta_review"])
            delta = json.loads(delta_path.read_text(encoding="utf-8"))
            _validate_target_sampling_review(document, delta)
            records.append({"reference_artifact": str(path.resolve()), "receipt": document,
                            "model_delta_receipt": {"reference_artifact": str(delta_path.resolve()), "receipt": delta},
                            "snapshot_policy": "original receipt binding only; no current-head or runtime inheritance"})
            continue
        if filename in ("glm_residual_norm_review.json", "kimi_residual_norm_review.json"):
            _validate_residual_norm_review(filename, document)
            records.append({"reference_artifact": str(path.resolve()), "receipt": document,
                            "snapshot_policy": "original receipt binding only; no current-head or runtime inheritance"})
            continue
        if filename == "glm_target_fused_embedding_review.json":
            _validate_target_fused_embedding_review(document)
            records.append({"reference_artifact": str(path.resolve()), "receipt": document,
                            "snapshot_policy": "original receipt binding only; no current-head or runtime inheritance"})
            continue
        if filename in _MHC_SM90_REVIEW_FILENAMES:
            _validate_mhc_sm90_review(filename, document)
            # Preserve the whole original receipt, including conditional
            # boxes, excluded bindings and original per-call evidence.
            records.append({"reference_artifact": str(path.resolve()), "receipt": document,
                            "snapshot_policy": "original receipt binding only; no current-head or runtime inheritance"})
            continue
        if filename == "grouped_topk_finite_precision_reference.json":
            _validate_grouped_topk_finite_precision_receipt(document)
        if filename == "operator_finite_precision_publication_review.json":
            candidate = document["candidate_observation"]
            if not isinstance(candidate, str) or re.fullmatch(r"[0-9a-f]{40}", candidate) is None:
                raise ValueError("finite-precision publication requires original candidate observation")
            checks = document["metadata_checks"]
            if (not isinstance(checks, list) or len(checks) != 2 or
                    any(not isinstance(check, dict) or type(check["block_path"]) is not bool
                        or not isinstance(check["metadata"], dict) for check in checks) or
                    {check["block_path"] for check in checks} != {True, False}):
                raise ValueError("finite-precision publication requires both metadata branches")
            published = document["published_receipt"]
            _validate_grouped_topk_finite_precision_receipt(published["receipt"])
            scenarios = document["source_scenarios_preserved"]
            if type(scenarios) is not int or scenarios != len(published["receipt"]["examples"]):
                raise ValueError("finite-precision publication scenario count disagrees with receipt")
        if filename == "operator_model_roofline_paths_review.json":
            source_records = document["records"]
            if not isinstance(source_records, list) or any(
                    not isinstance(record, dict) or not isinstance(record["phases"], list)
                    for record in source_records):
                raise ValueError("source-path publication requires original record/phase lists")
            if (document["model_layout_mode_records"] != len(source_records) or
                    document["phase_records"] != sum(len(record["phases"]) for record in source_records)):
                raise ValueError("source-path publication count disagrees with original records")
        receipt = {key: document[key] for key in metadata_fields if key in document}
        if not any(key in receipt for key in ("accepted_scope", "scope", "source_scope")):
            raise ValueError("formal source review requires its original acceptance scope")
        if filename == "grouped_topk_host_registration_review.json":
            checks = document["independent_checks"]
            if not isinstance(checks, dict) or any(type(value) is not bool for value in checks.values()):
                raise ValueError("formal source review host checks must be a boolean record map")
            receipt["checks"] = len(checks)
            receipt["failed"] = sum(not value for value in checks.values())
        for key in count_fields:
            if key in document:
                count = document[key]
                if filename == "grouped_topk_full_translation_unit_review.json" and key == "checks":
                    if not isinstance(count, list):
                        raise ValueError("formal source review full-TU checks must be a record list")
                    if any(not isinstance(record, dict) or type(record["pass"]) is not bool for record in count):
                        raise ValueError("formal source review full-TU check requires a boolean pass")
                    if sum(not record["pass"] for record in count) != document["failed"]:
                        raise ValueError("formal source review full-TU failed count disagrees with records")
                    receipt["check_records"] = count
                    receipt[key] = len(count)
                    continue
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"formal source review {key} must be a nonnegative integer")
                receipt[key] = count
        if "records" in document:
            if filename == "operator_model_roofline_paths_review.json":
                # Preserve each original phase flag, family and snapshot;
                # source_projection_only cannot be normalized to partial.
                receipt["records"] = document["records"]
            else:
                receipt["records"] = [{key: record[key] for key in
                                        ("snapshot", "artifact", "path", "tp", "failed") if key in record}
                                       for record in document["records"]]
        records.append({"reference_artifact": str(path.resolve()), "receipt": receipt,
                        "snapshot_policy": "original receipt binding only; no current-head or runtime inheritance"})
    return records


def _static_roofline_path_coverage(probe, artifact):
    """Read source-case pricing coverage without predicting or accepting it."""
    phases = []
    for step in probe["steps"]:
        calls = step["calls"]
        unpriced = []
        for call in calls:
            status = call["prediction"]["status"]
            if status == "unpriced":
                if call["prediction"]["seconds"] is not None:
                    raise ValueError("unpriced source call cannot carry predicted seconds")
                unpriced.append(call)
            elif status not in ("priced", "logical_alias"):
                raise ValueError("unknown source-case pricing status")
        if (type(step["kernel_calls"]) is not int or step["kernel_calls"] != len(calls)
                or type(step["unpriced_calls"]) is not int or step["unpriced_calls"] != len(unpriced)):
            raise ValueError("source-case call coverage disagrees with original artifact")
        phases.append({"phase": step["phase"], "calls": len(calls),
            "unpriced_calls": len(unpriced),
            "unpriced_families": dict(Counter(call["variant"] for call in unpriced)),
            "first_unpriced_call": None if not unpriced else unpriced[0]["call_id"],
            "roofline_end_to_end": step["roofline_end_to_end"]})
    return {"artifact": str(artifact), "snapshot": probe["snapshot"],
            "model_id": probe["model_id"], "model_revision": probe["model_revision"],
            "mtp": probe["mtp"], "deployment": probe["deployment"], "phases": phases,
            "runtime_or_normal_kv_accepted": False,
            "scope": "source-case coverage only; original identity, provider, routing and runtime qualifications retained"}


def _operator_probe_reproduction(probe, artifact):
    """Spell existing CLI inputs only when they reproduce every source fact.

    A historical artifact with missing constructor state or a programmatic
    input outside the existing CLI is retained as evidence, without an
    executable-looking command which would silently select the old graph.
    """
    def unavailable(reason):
        return {"command": None, "status": "inputs_not_representable", "reason": reason}

    if ("source_runtime_inputs_by_phase" not in probe
            or probe["source_runtime_inputs_by_phase"] is None):
        return unavailable("artifact has no complete phase RuntimeInputs; source/constructor inputs cannot be inferred")
    phases = probe["source_runtime_inputs_by_phase"]
    if not isinstance(phases, dict) or not phases:
        return unavailable("artifact has no named source phases")
    first = next(iter(phases.values()))
    source = first["symbolic_forward_source"]
    sampler = first["target_sampling_source"]
    mode = {"off": "off", "on": "native"}[probe["mtp"]]
    expected_requests = _operator_probe_phase_requests(probe["prefill_tokens"], mode,
        decode_prefix=source == "vllm_0_27_1_deepseek_v3_sync_fp8")
    expected_rows = {phase: _json_value([{**asdict(request), "positions": tuple(range(
        request.computed_tokens, request.computed_tokens + request.scheduled_tokens))}
        for request in requests]) for phase, requests in expected_requests}
    if (set(phases) != set(expected_rows) or len(probe["steps"]) != len(expected_rows)
            or any(step["phase"] not in expected_rows
                or step["scheduled_requests"] != expected_rows[step["phase"]] for step in probe["steps"])):
        return unavailable("custom phase requests are not expressible by the existing probe CLI fixture")
    ports = {phase: values["embedding_token_ids"] for phase, values in phases.items()}
    if any(ids is None for ids in ports.values()) and not all(ids is None for ids in ports.values()):
        return unavailable("mixed known/unknown phase token ports are not expressible by the existing CLI")
    command = [sys.executable, "scripts/evaluation/serving/report_model_operator_probe.py",
        "--model", str(probe["model_config_path"]), "--tensor-parallel-size",
        str(probe["deployment"]["tensor_parallel_size"]), "--prefill-tokens", str(probe["prefill_tokens"]),
        "--mtp-mode", mode, "--output", str(artifact)]
    if probe["deployment"]["enable_expert_parallel"]:
        command.append("--enable-expert-parallel")
    if sampler is not None:
        command += ["--target-sampling-source", sampler]
    pp = probe["deployment"]["pipeline_parallel_size"]
    if pp != 1:
        pipeline = probe["pipeline_source_requirements"]
        command += ["--pipeline-parallel-size", str(pp), "--pipeline-link", pipeline["pipeline_link_scenario"],
            "--pipeline-layer-partition", pipeline["VLLM_PP_LAYER_PARTITION"]]
    glm_sources = ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
    if source == "vllm_0_11_glm4_sync":
        if mode != "off" or pp != 1:
            return unavailable("GLM4 probe CLI declares only its target-off TP1 source fixture")
        if any(ids is None for ids in ports.values()):
            return unavailable("GLM4 source CLI cannot reproduce unknown token values with a synthetic-ID fixture")
        fixture = _default_glm4_target_probe_source(load_model_config(probe["model_config_path"]),
            ParallelConfig(tensor_parallel_size=probe["deployment"]["tensor_parallel_size"],
                enable_expert_parallel=probe["deployment"]["enable_expert_parallel"]),
            expected_requests, token_ports={phase: None if ids is None else tuple(ids) for phase, ids in ports.items()},
            sampler=sampler)
        if fixture is None:
            return unavailable("GLM4 probe CLI cannot represent this parallel source fixture")
        expected, _ = fixture
    elif source == "vllm_0_30_v4_sync_fp8_ds_mla":
        if mode != "off" or first["v4_eager_metadata_source"] != "vllm_0_30_eager_dict_all_valid":
            return unavailable("V4 CLI requires the existing target-off eager dict source")
        command += ["--v4-eager-metadata", "--v4-routing-source", first["v4_routing_source"],
            "--v4-source-max-model-len", str(first["v4_source_max_model_len"]),
            "--v4-source-max-batched-tokens", str(first["v4_source_max_batched_tokens"])]
        branch = first["v4_source_use_deep_gemm"]
        if type(branch) is not bool:
            return unavailable("V4 artifact has no declared DeepGEMM process branch")
        command.append("--v4-source-use-deep-gemm" if branch else "--no-v4-source-use-deep-gemm")
        if first["v4_indexer_selection_source"] is not None:
            command += ["--v4-indexer-selection-source", first["v4_indexer_selection_source"]]
        routing = ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0)
        expected = {phase: RuntimeInputs(routing, "mla", "uniform", model_execution_version="vllm_0_30_0",
            symbolic_forward_source=source, target_sampling_source=sampler,
            embedding_token_ids=None if ids is None else tuple(ids),
            v4_eager_metadata_source=first["v4_eager_metadata_source"],
            v4_routing_source=first["v4_routing_source"], v4_source_max_model_len=first["v4_source_max_model_len"],
            v4_source_max_batched_tokens=first["v4_source_max_batched_tokens"],
            v4_source_use_deep_gemm=branch, v4_indexer_selection_source=first["v4_indexer_selection_source"])
            for phase, ids in ports.items()}
    elif source in ("cuda_eager_inplace", *glm_sources, "vllm_0_30_k2_6_int4_sync_explicit_allreduce",
            "vllm_0_27_1_deepseek_v3_sync_fp8"):
        if source != "cuda_eager_inplace" and sampler is None:
            return unavailable("existing ordinary probe CLI cannot combine explicit constructor inputs with a sampler-less source")
        if source == "cuda_eager_inplace" and sampler not in (
                None, "vllm_0_30_plain_sampler_bf16", "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_greedy_fp16"):
            return unavailable("existing ordinary probe CLI cannot spell this model/sampler version")
        state = None
        if mode == "native":
            if source not in glm_sources:
                return unavailable("existing native CLI accepts only explicit GLM padded inputs")
            from .simulation_configuration import decode_typed_step_source_json
            state = {phase: decode_typed_step_source_json(json.dumps(values["model_source_inputs"]),
                V32PaddedMTPInputs) for phase, values in phases.items()}
            for phase, values in phases.items():
                command += ["--glm-mtp-phase-source-input", phase, json.dumps(values["model_source_inputs"])]
        if source != "cuda_eager_inplace":
            for flag, name in (("--source-max-model-len", "symbolic_source_max_model_len"),
                    ("--source-max-batched-tokens", "symbolic_source_max_batched_tokens"),
                    ("--source-max-requests", "symbolic_source_max_requests"),
                    ("--mtp-draft-sampling-source", "mtp_draft_sampling_source"),
                    ("--glm-indexer-selection-source", "glm_indexer_selection_source"),
                    ("--glm-prefill-attention-source", "glm_prefill_attention_source")):
                if first[name] is not None:
                    command += [flag, str(first[name])]
            if first["symbolic_workspace_initial_capacity_bytes"] is not None:
                command += ["--source-workspace-initial-capacity-bytes",
                    *(str(value) for value in first["symbolic_workspace_initial_capacity_bytes"])]
        expected = build_model_operator_probe_source_inputs(symbolic_forward_source=source,
            embedding_token_ids_by_phase={phase: None if ids is None else tuple(ids) for phase, ids in ports.items()},
            source_max_model_len=first["symbolic_source_max_model_len"],
            source_max_batched_tokens=first["symbolic_source_max_batched_tokens"], target_sampling_source=sampler,
            workspace_initial_capacity_bytes=(None if first["symbolic_workspace_initial_capacity_bytes"] is None
                else tuple(first["symbolic_workspace_initial_capacity_bytes"])),
            glm_indexer_selection_source=first["glm_indexer_selection_source"],
            glm_prefill_attention_source=first["glm_prefill_attention_source"],
            source_max_requests=first["symbolic_source_max_requests"],
            mtp_draft_sampling_source=first["mtp_draft_sampling_source"], model_source_inputs_by_phase=state)
        model = load_model_config(probe["model_config_path"])
        if source == "cuda_eager_inplace" and model.num_experts:
            routing = ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0)
            expected = {phase: replace(value, expert_routing=routing) for phase, value in expected.items()}
    else:
        return unavailable(f"source {source!r} has no existing probe CLI spelling")
    expected = {phase: _json_value(asdict(value)) for phase, value in expected.items()}
    differences = sorted({name for phase, values in expected.items()
        for name in set(values) | set(phases[phase])
        if name not in values or name not in phases[phase] or phases[phase][name] != values[name]})
    if differences:
        return unavailable("existing CLI cannot preserve phase RuntimeInputs fields: " + ", ".join(differences))
    if all(ids is not None for ids in ports.values()):
        token_file = Path(artifact).with_name(Path(artifact).stem + "_embedding_token_ids.json")
        token_file.write_text(json.dumps(ports, indent=2) + "\n", encoding="utf-8")
        command += ["--embedding-token-ids", str(token_file)]
    return {"command": shlex.join(command), "status": "complete_source_inputs", "reason": None}


def write_operator_support_checkpoint(
        output: Path, reference: Path, tests_log: Path, recheck_log: Path | None,
        kernel_results: Path, *, current_probes: tuple[Path, ...] = (),
        source_storage_probes: tuple[Path, ...] = (),
        weight_diagnostics: tuple[Path, ...] = (),
        source_graph_probes: tuple[Path, ...] = ()) -> None:
    """Regenerate the Goal checkpoint from static facts and external review.

    A static priced call is not source alignment or real Scheduler evidence.
    This checkpoint deliberately leaves every full-scope gate open; it does
    not turn implementation smoke or missing evidence into acceptance.
    """
    # Validate optional published receipts before any derived output writes.
    formal_source_reviews = _formal_source_delta_reviews(reference)
    contract_path = reference / "source_contract.json"
    review_path = reference / "review.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    glm_capacity_path = reference / "glm_capacity_reference.json"
    glm_capacity = json.loads(glm_capacity_path.read_text(encoding="utf-8"))
    for case in glm_capacity["cases"]:
        if case["normal_capacity_accepted"] is not False:
            raise ValueError("GLM routed-weight lower bounds are not normal KV acceptance")
    glm_capacity_record = {"artifact": str(glm_capacity_path), **glm_capacity}
    weight_records = []
    for artifact in weight_diagnostics:
        diagnostic = json.loads(artifact.read_text(encoding="utf-8"))
        if diagnostic["status"] != "diagnostic_not_normal_capacity_acceptance":
            raise ValueError("weight diagnostic cannot claim normal capacity acceptance")
        for case in diagnostic["cases"]:
            if case["normal_kv_capacity_accepted"] is not False:
                raise ValueError("constructor parameter inventory is not normal capacity evidence")
            # payload_scope is an optional diagnostic metadata field. Older
            # V4 records keep their original schema/snapshot and normalize None.
            scope = case["payload_scope"] if "payload_scope" in case else None
            if scope not in (None, "kimi_active_outer_registered_parameters", "glm_parameter_source_projection"):
                raise ValueError("unsupported constructor parameter diagnostic scope")
            case["payload_scope"] = scope
            if scope == "kimi_active_outer_registered_parameters" and case["actual_outer_loader_or_runtime_binding"] is not False:
                raise ValueError("Kimi constructor source scenario cannot claim actual loader/runtime binding")
            if scope == "glm_parameter_source_projection" and case["actual_loader_or_runtime_binding"] is not False:
                raise ValueError("GLM parameter source projection cannot claim actual loader/runtime binding")
        weight_records.append({"artifact": str(artifact), **diagnostic})
    graph_records = []
    for artifact in source_graph_probes:
        graph = json.loads(artifact.read_text(encoding="utf-8"))
        for flag in ("lowering_executed", "prediction_executed",
                     "actual_runtime_or_loaded_weight_binding", "normal_kv_capacity_accepted"):
            if graph[flag] is not False:
                raise ValueError("source graph evidence must exclude lowering/pricing/runtime/capacity acceptance")
        graph_records.append({
            "artifact": str(artifact), "snapshot": graph["snapshot"],
            "model_id": graph["model_id"], "deployment": graph["deployment"],
            "v4_eager_metadata_source": graph["v4_eager_metadata_source"],
            "scope": graph["scope"],
            "phases": [{"phase": step["phase"], "nodes": len(step["nodes"]),
                        "edges": len(step["edges"])} for step in graph["steps"]],
        })
    layer_comparison_path = reference / "layer_stack_comparison.json"
    layer_comparison = json.loads(layer_comparison_path.read_text(encoding="utf-8"))
    layer_rescore = _rescore_formal_layer_shapes(layer_comparison)
    layer_rescore["expectation_reference"] = str(layer_comparison_path)
    layer_rescore_path = output / "layer_stack_rescore.json"
    layer_rescore_path.write_text(json.dumps(layer_rescore, indent=2) + "\n", encoding="utf-8")
    v4_comparison_path = reference / "v4_compressor_comparison.json"
    v4_comparison = json.loads(v4_comparison_path.read_text(encoding="utf-8"))
    v4_rescore = _rescore_v4_compressor_bytes(v4_comparison)
    v4_rescore["expectation_reference"] = str(v4_comparison_path)
    v4_rescore_path = output / "v4_compressor_rescore.json"
    v4_rescore_path.write_text(json.dumps(v4_rescore, indent=2) + "\n", encoding="utf-8")
    native_comparison_path = reference / "native_mtp_comparison.json"
    native_comparison = json.loads(native_comparison_path.read_text(encoding="utf-8"))
    kernel_matrix = json.loads(kernel_results.read_text(encoding="utf-8"))
    tests_summary = tests_log.read_text(encoding="utf-8").splitlines()[-1]
    if recheck_log is None:
        if not re.fullmatch(r"\d+ passed in .*", tests_summary):
            raise ValueError("a failing or incomplete test run requires explicit recheck evidence")
        recheck_summary = "Full maintained suite passed this round; no failed-test reruns"
    else:
        recheck_summary = recheck_log.read_text(encoding="utf-8").splitlines()[-1]
    if contract["new_gpu_execution"] is not False:
        raise ValueError("checkpoint expects an offline source reference")
    models = ("kimi-k2.6", "kimi-k3", "glm-5.2", "glm-5.2-fp8", "glm-5.3",
              "deepseek-v4-flash", "deepseek-v4-pro")
    alignment = []
    rows = []
    native_rows = []
    native_records = []
    for model in models:
        relative_config = Path("configs/model") / f"{model}.json"
        config = REPOSITORY_ROOT / relative_config
        if not config.is_file():
            alignment.append({"model": model, "status": "missing_exact_config"})
            rows.append(f"| {model} | Not run | Exact checkpoint/config missing |")
            continue
        artifact = output / f"{model}_static_probe.json"
        if not artifact.is_file():
            alignment.append({"model": model, "status": "no_complete_model_probe"})
            rows.append(f"| {model} | Full model not run | Source components do not establish full static support |")
            continue
        probe = json.loads(artifact.read_text(encoding="utf-8"))
        current_model = load_model_config(str(config))
        if ("model_id" not in probe or "model_revision" not in probe
                or probe["model_id"] != current_model.model_id
                or probe["model_revision"] != current_model.model_revision):
            alignment.append({"model": model, "status": "historical_identity_excluded", "artifact": str(artifact)})
            rows.append(f"| {model} | Historical-identity results excluded | Not acceptance of the current official checkpoint |")
            continue
        if Path(probe["model_config_path"]) != relative_config or probe["mtp"] != "off":
            raise ValueError("static probe model/mode binding disagrees")
        phases = []
        for step in probe["steps"]:
            phases.append({
                "phase": step["phase"], "calls": step["kernel_calls"],
                "unpriced_calls": step["unpriced_calls"],
                "roofline_end_to_end": step["roofline_end_to_end"],
                "variant_counts": dict(Counter(call["variant"] for call in step["calls"])),
            })
        reproduction = _operator_probe_reproduction(probe, artifact)
        alignment.append({
            "model": model, "status": "modeled_facts_await_independent_comparison",
            "artifact": str(artifact), "snapshot": probe["snapshot"], "phases": phases,
            "command": reproduction["command"], "reproduction": reproduction,
        })
        counts = "/".join(str(phase["calls"]) for phase in phases)
        gaps = "/".join(str(phase["unpriced_calls"]) for phase in phases)
        rows.append(f"| {model} | {counts} calls; {gaps} unpriced | MTP-off static facts, pending independent alignment |")
    for name in ("glm-5.2_mtp_tp1", "glm-5.2_mtp"):
        artifact = output / f"{name}_static_probe.json"
        if not artifact.is_file():
            native_records.append({"artifact": str(artifact), "status": "not_executed"})
            native_rows.append(f"| {name} | Not run | Native-on evidence missing |")
            continue
        probe = json.loads(artifact.read_text(encoding="utf-8"))
        if (probe["mtp"] != "on" or probe["draft_architecture"] != "DeepseekV32MTPModel"
                or Path(probe["model_config_path"]) != Path("configs/model/glm-5.2.json")):
            raise ValueError("native GLM probe identity/mode disagrees")
        phases = [{"phase": step["phase"], "calls": step["kernel_calls"],
                   "unpriced_calls": step["unpriced_calls"]} for step in probe["steps"]]
        native_records.append({"artifact": str(artifact), "snapshot": probe["snapshot"],
                               "status": probe["on_mode_status"], "phases": phases,
                               "gaps": probe["native_source_gaps"]})
        counts = "/".join(str(phase["calls"]) for phase in phases)
        native_rows.append(f"| {name} | {counts} calls | Partial: compiler/runtime, actual target fused launch/slots, proposer metadata, and normal capacity remain unaccepted |")
    current_records = []
    current_rows = []
    current_pipeline_scopes = []
    existing_paths = {}
    for artifact in current_probes:
        probe = json.loads(artifact.read_text(encoding="utf-8"))
        # The last supplied artifact for each layout/mode retains its own
        # source scenario and snapshot; it is never current-head acceptance.
        path_key = (probe["model_id"], probe["mtp"],
                    probe["deployment"]["tensor_parallel_size"],
                    probe["deployment"]["pipeline_parallel_size"])
        existing_paths[path_key] = _static_roofline_path_coverage(probe, artifact)
        phases = [{"phase": step["phase"], "calls": step["kernel_calls"],
                   "unpriced_calls": step["unpriced_calls"],
                   "draft_sampling_stage_counts": dict(Counter(
                       call["variant"] for call in step["calls"]
                       if call["variant"] in ("dsv32_draft_argmax", "dsv32_draft_id_cast",
                                              "dsv32_draft_row_select", "dsv32_draft_buffer_copy", "dsv32_draft_token_stack", "dsv32_draft_id_scatter", "dsv32_draft_prepare_next_ids", "dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback", "dsv32_draft_step_metadata"))),
                   "draft_materialization_logical_roofline_calls": sum(
                       call["variant"] in DSV32_DRAFT_MATERIALIZATION_VARIANTS
                       and call["prediction"]["status"] == "priced"
                       and "scope" in call["prediction"]
                       and call["prediction"]["scope"] == "logical_roofline_only_no_launch_binding"
                       for call in step["calls"]),
                   "draft_int32_logical_roofline_calls": sum(
                       call["variant"] in DSV32_DRAFT_INT32_BOOKKEEPING_VARIANTS
                       and call["prediction"]["status"] == "priced"
                       and "scope" in call["prediction"]
                       and call["prediction"]["scope"] == "logical_roofline_only_no_launch_binding"
                       for call in step["calls"]),
                   "target_logits_selection_calls": sum(
                       call["variant"] == "vllm_logits_row_select" for call in step["calls"]),
                   "target_sampler_cast_calls": sum(
                       call["variant"] == "vllm_logits_fp32_cast" for call in step["calls"]),
                   "target_sampler_cast_logical_bytes": sum(
                       call["resource_input"]["logical_bytes"] for call in step["calls"]
                       if call["variant"] == "vllm_logits_fp32_cast"),
                   "rejection_materialization_stage_counts": dict(Counter(
                       call["attributes"]["materialization_stage"] for call in step["calls"]
                       if call["variant"] == "vllm_rejection_logits_materialize")),
                   "rejection_materialization_logical_bytes": sum(
                       call["resource_input"]["logical_bytes"] for call in step["calls"]
                       if call["variant"] == "vllm_rejection_logits_materialize"),
                   "target_logits_logical_roofline_calls": sum(
                       call["variant"] == "vllm_logits_row_select"
                       and call["prediction"]["status"] == "priced"
                       and "scope" in call["prediction"]
                       and call["prediction"]["scope"] == "logical_roofline_only_no_launch_binding"
                       for call in step["calls"]),
                   "target_logits_selection_output_bytes": sum(
                       operand["bytes"] for call in step["calls"]
                       if call["variant"] == "vllm_logits_row_select"
                       for operand in call["resource_input"]["logical_work"]["operands"]
                       if operand["role"] == "sample_hidden_states"),
                   "target_logits_selection_index_sources": dict(Counter(
                       call["attributes"]["index_source"] for call in step["calls"]
                       if call["variant"] == "vllm_logits_row_select")),
                   "target_fused_embed_norm_calls": sum(
                       call["variant"] == "dsv32_embed_norm"
                       and call["prediction"]["status"] == "priced"
                       for call in step["calls"]),
                   "residual_add_norm_calls": sum(
                       call["variant"] == "add_rmsnorm" for call in step["calls"]),
                   "residual_add_norm_logical_bytes": sum(
                       call["resource_input"]["logical_bytes"] for call in step["calls"]
                       if call["variant"] == "add_rmsnorm"),
                   "source_vocab_embedding_calls": sum(
                       call["variant"] == "vocab_parallel_embedding" for call in step["calls"]),
                   "source_vocab_embedding_partial_output_bytes": sum(
                       operand["bytes"] for call in step["calls"]
                       if call["variant"] == "vocab_parallel_embedding"
                       for operand in call["resource_input"]["logical_work"]["operands"]
                       if operand["role"] == "partial_output"),
                   "unpriced_calls_with_logical_work": sum(
                       call["prediction"]["status"] == "unpriced"
                       and "logical_work" in call["resource_input"]
                       for call in step["calls"]),
                   "roofline_end_to_end": step["roofline_end_to_end"],
                   "scheduled_requests": step["scheduled_requests"],
                   "rank0_c128_compress": [{
                       "call_id": call["call_id"], "shape": call["shape"],
                       "dtype": call["dtype"], "demand": call["resource_input"]["demand"],
                       "dependencies": call["dependencies"],
                   } for call in step["calls"] if call["rank"] == 0
                   and call["variant"] == "dsv4_compressed_state_cache"
                   and call["attributes"]["finish_stage"] == "compress"]}
                  for step in probe["steps"]]
        current_records.append({"artifact": str(artifact), "snapshot": probe["snapshot"],
                                "model": probe["model_id"], "mtp": probe["mtp"],
                                "deployment": probe["deployment"], "phases": phases,
                                "scope": "modeled source-layer facts; independent acceptance not inherited"})
        if "mtp_draft_sampling_source" in probe:
            current_records[-1]["mtp_draft_sampling_source"] = probe["mtp_draft_sampling_source"]
        if "target_sampling_source" in probe:
            current_records[-1]["target_sampling_source"] = probe["target_sampling_source"]
        if "v4_eager_metadata_source" in probe:
            current_records[-1]["v4_eager_metadata_source"] = probe["v4_eager_metadata_source"]
        if "v4_routing_source" in probe:
            current_records[-1]["v4_routing_source"] = probe["v4_routing_source"]
            current_records[-1]["v4_routing_requirements"] = probe["v4_routing_requirements"]
        if "pipeline_source_requirements" in probe:
            current_records[-1]["pipeline_source_requirements"] = probe["pipeline_source_requirements"]
            current_records[-1]["pipeline_weight_projection"] = probe["pipeline_weight_projection"]
            rank_scope = probe["pipeline_source_requirements"]["modeled_tp_partial_producer_scope"]
            if rank_scope is not None:
                current_pipeline_scopes.append(
                    f"{artifact.name} ({probe['snapshot']['commit'][:8]}) source-compute coverage declaration: {rank_scope}.")
        counts = "/".join(str(phase["calls"]) for phase in phases)
        gaps = "/".join(str(phase["unpriced_calls"]) for phase in phases)
        logical_work = "/".join(str(phase["unpriced_calls_with_logical_work"]) for phase in phases)
        current_rows.append(f"| {artifact.name} | {probe['snapshot']['commit'][:8]} | {counts} calls; {gaps} unpriced; {logical_work} with logical math/storage quantities |")
        if probe["model_id"] in ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro"):
            current_rows[-1] = current_rows[-1][:-1] + (
                f"; V4 metadata source={probe['v4_eager_metadata_source']} |")
            if "v4_routing_source" in probe and probe["v4_routing_source"] is not None:
                current_rows[-1] = current_rows[-1][:-1] + (
                    f"; routing source={probe['v4_routing_source']} (actual provider unbound) |")
    source_storage_records = []
    for artifact in source_storage_probes:
        probe = json.loads(artifact.read_text(encoding="utf-8"))
        if (probe["status"] != "source_storage_subset_not_full_native_or_priced_model"
                or probe["independent_acceptance"] is not False
                or probe["new_gpu_execution"] is not False
                or probe["normal_gpu_v2_workload_supported"] is not False):
            raise ValueError("source storage subset cannot claim native/runtime acceptance")
        # uniform_updates is the schema-defined optional source-work part;
        # old storage-only snapshots keep their original subset scope.
        work = probe["logical_work"]
        updates = work["uniform_updates"] if "uniform_updates" in work else ()
        if any(op["launch"] is not None or op["seconds"] is not None or op["demand"] is not None
               for op in (*work["operations"], *updates)):
            raise ValueError("source storage subset cannot assign launch, demand or duration")
        prepare_scenarios = (probe["prepare_decode_source_scenarios"]
                             if "prepare_decode_source_scenarios" in probe else ())
        if any(case["work"]["launch"] is not None or case["work"]["seconds"] is not None
               or case["work"]["demand"] is not None for case in prepare_scenarios):
            raise ValueError("prepare decode source scenarios cannot assign launch, demand or duration")
        slot_work = (probe["slot_mapping_source_scenario"]["work"]
                     if "slot_mapping_source_scenario" in probe else None)
        if slot_work is not None and any(slot_work[key] is not None for key in ("launch", "seconds", "demand")):
            raise ValueError("slot mapping source scenario cannot assign launch, demand or duration")
        refresh_plans = (probe["metadata_refresh_source_scenarios"]
                         if "metadata_refresh_source_scenarios" in probe else ())
        backend_plans = (probe["sparse_backend_metadata_source_constraints"]
                         if "sparse_backend_metadata_source_constraints" in probe else ())
        if any(event[key] is not None for plan in (*refresh_plans, *backend_plans) for event in plan["source_events"]
               for key in ("launch", "seconds", "demand")):
            raise ValueError("metadata refresh source events cannot assign launch, demand or duration")
        source_storage_records.append({"artifact": str(artifact), **probe})
    actuals = (
        "Four locally bound configs support source-math projection. V4 comes from an existing H200 campaign, without verified HF revision/checkpoint identity. K2.7/K3/GLM5.3 exact configs and DSpark draft-field/weight bindings are missing.",
        "Scoped V4 QK, finish, eager short-context, and gather reviews exist. Formal scoped passes: C128 visible state 73788e5a; FFN gate/shared clamp 3c354e6b; unpriced router stage 0f4183d2; inverse RoPE/padded scales 30cb98ac (3328/3328 and 6 direct); routed-core clamp 40bd4b28/9e008651 (2496 and 2 direct); Marlin W2 weight multiply d07fb71b (1072 and 2 direct). GPU-v2 storage42, metadata40, and refresh110 passed within their scopes. PP2 reduction placement 92c3dfee passed 52/52 and 2 direct. GLM PP1 all-TP/native math graph f1479578 has independent scoped acceptance, with routing still unpriced. The finite CUDA route source program retains conditional/compiler-service gaps and inherits no actual provider/runtime acceptance. PP2 default partial transport/actual launch remains incomplete. Original snapshots are retained.",
        "Kimi d7e4356a explicitly separates the v0.30 BF16 A/B/C plain GateLinear from its grouped sigmoid top8 producer, leaving 480 unpriced routes per phase. Older Kimi artifacts retain original snapshots and historical 0-unpriced labels, without establishing complete current pricing. GLM V32 GateLinear separates BF16 A/B and FP32 C, while grouped routing remains unpriced. V4 first-layer/large-T DeepGEMM prenorm remains unpriced. Actual backend binding is incomplete; supplementary artifacts retain full phase denominators.",
        "Real v0.30 Scheduler lacks a valid NamespaceTool dependency. Kimi full-parent-model skip/loaded residency is unbound; normal capacity rejects text_config-only accounting. GLM BF16 TP1/PP1 and TP8/PP1 target routed weights alone exceed full-card HBM, so off/native source graphs lack normal capacity. TP8/PP2 is not excluded by that lower bound, but its explicit 38,40+NVLink source candidate retains unpriced routing. Default TP-partial hidden shard/all-gather does not prove numerical correctness; actual transport overrides/compiler transformations are unbound. Scheduler/native-on still reject; physical topology/normal allocator/QPS16 and V4 normal residency remain unaccepted.",
        "GLM V32 has TP1/TP8 native-on static facts and a GPU-v2 source storage/state subset: one hidden gather and temporary/front copy; intermediate steps update IDs/hidden/positions/seq_lens; final steps store draft tokens only. Independent prepare_decode and CP1 slot scenarios retain typed state, full capacity padding, disabled behavior, and SIMD-tail reads. Eager fused/ordinary events follow explicit group capabilities and retain dependencies. They remain unpriced pending independent review; actual backend capability is unbound. Full GPU-v2, INT32 library lowering, sampler/slot/runtime remain unsupported/unbound; V4 DSpark remains incomplete.",
        "Independent source review 3a2f725d found no mismatch in 1904 fusion calls; independent reference memory rescoring passed 5024/5024. Formal files bind their own snapshots; full-target independent_acceptance=false."
        + ("; ordinary/rejection sampler materialization, conditional INT64 hidden projection, and scenario rejection boundaries have formal source-only acceptance. Routing/native state service, complete sampler, actual metadata/provider/runtime/normal capacity, and complete alignment remain unaccepted."
           if any(Path(record["reference_artifact"]).name == "target_sampling_materialization_review.json"
                  for record in formal_source_reviews) else ""),
        f"Maintained tests: {tests_summary}; direct-change/failed-item and dependency rechecks: {recheck_summary}; Kernel matrix results are in logs. Serving QPS16 and native-on runtime validation remain missing. Historical full suites do not establish coverage of new production commits.",
    )
    gates = [{"gate": f"W{index}", "target": "GOAL.md Hard acceptance gates",
              "actual": actual, "status": "Fail"}
             for index, actual in enumerate(actuals, 1)]
    for filename, data in (
            ("operator_alignment.json", {"independent_acceptance": False,
                "source_contract": str(contract_path), "review": str(review_path),
                "review_acceptance": review["independent_acceptance"], "records": alignment,
                "native_records": native_records,
                "independent_glm_capacity_evidence": glm_capacity_record,
                "formal_layer_comparison": {
                    "path": str(layer_comparison_path),
                    "totals": layer_comparison["totals"],
                    "snapshot_binding": "each record retains its own reviewed snapshot; no inheritance across source deltas",
                },
                "current_v4_compressor_replay": {"path": str(v4_rescore_path),
                                                "totals": v4_rescore["totals"]},
                "current_layer_shape_replay": {"path": str(layer_rescore_path),
                                               "totals": layer_rescore["totals"]},
                "formal_native_comparison": {
                    "path": str(native_comparison_path),
                    "totals": native_comparison["totals"],
                    "snapshot_binding": "each record retains its own reviewed snapshot; no inheritance across source deltas",
                }}),
            ("gate.json", {"status": "Goal incomplete — checkpoint", "gates": gates})):
        (output / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    handoff = {
        "status": "Goal incomplete — checkpoint", "report_generation_snapshot": _snapshot(),
        "static_evidence": alignment,
        "current_static_evidence": current_records,
        "native_static_evidence": native_records,
        "source_storage_evidence": source_storage_records,
        "weight_diagnostic_evidence": weight_records,
        "source_graph_evidence": graph_records,
        "independent_glm_capacity_evidence": glm_capacity_record,
        "formal_source_delta_reviews": formal_source_reviews,
        "validation": {
            "tests_log": str(tests_log), "tests_terminal_summary": tests_summary,
            "recheck_log": None if recheck_log is None else str(recheck_log),
            "recheck_terminal_summary": recheck_summary,
            "kernel_results": str(kernel_results),
            "kernel_status": {key: value["status"] for key, value in kernel_matrix["kernel_matrix"].items()},
            "scope": "historical snapshots in log filenames retained; recheck runs only failed items and direct dependencies",
        },
        "failed_gates": [gate["gate"] for gate in gates],
        "blockers": [actuals[index] for index in (0, 3, 4)],
        "next_steps": [
            "Retain formal residual-norm and target sampler materialization receipts at their original snapshots, including dirty and source-only qualifications. Reuse all prior scopes without reopening unchanged findings; complete sampler/native/runtime and full W6 remain unaccepted.",
            "Preserve original vLLM0.30 PP baseline and source_projection_only. Source rank math, receiver reduction and BF16 payloads do not prove TP-partial preservation in GPUWorker's default transport. Do not assume an upstream patch/override or change wire bytes to claim acceptance; normal Serving remains fail-closed.",
            "Known typed materializations already have logical Roofline pricing without exact launches. Ordinary sampling closes only its mandatory BF16-to-FP32 stage. The explicit metadata-present raw rejection scenario additionally preserves bonus/target INT32 selectors, two distinct bonus casts and target FP32 clone. Optional outer logprobs, processors, sampling, rejection and output-ID conversion remain uncovered; conditional prefill index binding is not actual Scheduler evidence. TP-sharded embedding point reads need real valid IDs or per-rank owned rows; pointer alignment/grid/block are separate launch facts.",
            "Grouped routing retains source math and conditional CUDA12.2 SM90 EX2/RCP/shared-scratch evidence, plus finite-precision branch limitations. Point service pricing still needs defensible semantics for transcendental/packed64 work; do not invent a provider, infer instruction counts from static sites, use production durations or tune parameters.",
            "Keep precise external blockers with the main agent: missing accurate configs/DSpark assets, compatible NamespaceTool SDK and normal residency/allocator facts. Reuse bounded local searches and successful local Torch2.11 compilation evidence. Do not repeat unchanged asset/environment audits; continue on concrete source deltas, new inputs or an approved implementation path.",
        ],
        "push_status": "not pushed; no remote changes",
        "reproduce_checkpoint_command": shlex.join([
            sys.executable, "scripts/evaluation/serving/report_operator_support_checkpoint.py",
            "--output", str(output), "--reference", str(reference),
            "--tests-log", str(tests_log),
            *(["--recheck-log", str(recheck_log)] if recheck_log is not None else []),
            "--kernel-results", str(kernel_results)]),
    }
    if current_probes:
        handoff["reproduce_checkpoint_command"] += " " + shlex.join([
            argument for artifact in current_probes
            for argument in ("--current-probe", str(artifact))])
    if source_storage_probes:
        handoff["reproduce_checkpoint_command"] += " " + shlex.join([
            argument for artifact in source_storage_probes
            for argument in ("--source-storage-probe", str(artifact))])
    if weight_diagnostics:
        handoff["reproduce_checkpoint_command"] += " " + shlex.join([
            argument for artifact in weight_diagnostics
            for argument in ("--weight-diagnostic", str(artifact))])
    if source_graph_probes:
        handoff["reproduce_checkpoint_command"] += " " + shlex.join([
            argument for artifact in source_graph_probes
            for argument in ("--source-graph-probe", str(artifact))])
    (output / "cli_handoff.json").write_text(json.dumps(handoff, indent=2) + "\n", encoding="utf-8")
    status = {
        "status": "goal_checkpoint", "goal_complete": False,
        "commit": handoff["report_generation_snapshot"]["commit"],
        "snapshot": handoff["report_generation_snapshot"],
        "working_on": "W1-W7 remain incomplete. Existing variants now preserve GLM/Kimi residual norm reads/writes and memo semantics. Explicit ordinary BF16-to-FP32 and metadata-present raw rejection sampler materializations have logical Roofline pricing without launches. Rejection bonus/target INT32 selectors, independent casts and FP32 clone retain distinct roles; neither scenario covers complete sampling or actual metadata. Source scenarios, prior formal receipts, actual provider/compiler/runtime, PP transport and normal capacity retain separate acceptance. Main owns linear integration; unchanged asset/environment audits are not a next action.",
        "failed_gates": handoff["failed_gates"],
        "validation": handoff["validation"],
        "current_static_evidence": current_records,
        "gate_checkpoint": gates,
        "source_storage_evidence": source_storage_records,
        "weight_diagnostic_evidence": weight_records,
        "source_graph_evidence": graph_records,
        "independent_glm_capacity_evidence": glm_capacity_record,
        "formal_source_delta_reviews": formal_source_reviews,
        "external_inputs_missing": ["actual H200 provider/capability and normal memory/reserve evidence", "V4 DSpark proposer/native runtime and dynamic state/sampling bindings"],
        "limitations": [actuals[index] for index in (1, 2, 3, 4, 5)],
        "ongoing_commands": [], "writes_stopped": False,
        "next_steps": handoff["next_steps"],
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    text = ["# Goal incomplete — checkpoint", "",
        "The target remains seven models on H200 + vLLM 0.30, text-only eager, with shared MTP-off and native MTP-on paths. These are local offline static facts without GPU execution; priced does not establish real source-launch alignment.", "",
        f"GPU-v2 source storage-only evidence: {[record['artifact'] for record in source_storage_records]}. Records cover producer/temporary/copy/feedback roles, typed logical bytes, and dependencies; launch/demand/seconds are None. They do not enter full model-call denominators or establish INT32 public lowering or normal native/Serving support. New prepare_decode_source_scenarios are separate declared state scenarios rather than actual scheduled_requests metadata in the same file. Depth1 returns before producers; otherwise target seq subtracts rejected counts, sampling keys always increment by 1, and only advance=True advances/saturates positions/seq_lens. Extra CTAs write query_start_loc and zero seq_lens tails over full max_num_requests, including eager capacity writes. Prepare reads DraftModelSpeculator INT64 draft_tokens, distinct from sampler-return dtype. Rejection-count dtype is explicit; actual producer, Triton warps, slot/page/cache, and runtime remain unbound. The state/role subset received formal scoped review at 43bb0de3, without actual native/runtime acceptance.", "",
        f"Checkpoint generation snapshot: {handoff['report_generation_snapshot']['commit']}. Goal incomplete; writes_stopped=false. Platform Goal status is managed independently by the scheduler; this report does not infer active/usageLimited. Supplementary evidence binds its own generation snapshot without replacing historical artifacts.", "",
        "slot_mapping_source_scenario is an explicit CP1 request/block-table scenario, not actual allocator input for scheduled_requests. INT64 slot writes span group×full capacity; returning num_tokens_padded slices cannot remove tail writes. mapping_enabled=False still reads block0 before writing PAD_ID. The 1024-element SIMD tail has position mask=0 but no valid-row mask on block-number loads; tail block0 source objects are listed separately. Shared group inputs count distinct objects only, without transaction/DRAM/local-service inference. Layer slot views add no copy; slot mapping, model KV insertion, rejected rollback, and page ownership remain separate. Zero groups launch no kernel; CP>1 is explicitly unsupported. Actual Triton/metadata/group/lifecycle remain unbound. Formal reference confirmed d6bf2ca4 terminal-round native rejection and off-control 2 direct checks, retaining that snapshot; prepare/slot subsets passed scoped 40/40 and 6 direct at 43bb0de3.", "",
        "metadata_refresh_source_scenarios record GLM advancing, non-dummy NONE/eager uniform-loop control flow. Capabilities are explicit source scenarios, actual_attention_backend_binding=None; eager does not imply fused behavior. If all attention groups support updates, metadata is built once, then each non-final input update refreshes slots and updates groups in order. With an unsupported group, slots/layer views/metadata rebuild before every forward. Depth1 returns before uniform events; the final update writes draft tokens only without another slot refresh. Current-step fill, model forward, sampler, input update, slot, and host/view/backend events receive no launch/demand/seconds and count neither as kernels nor complete native pricing. EPLB/DP/prefill/cache lifecycles remain out of scope.", "",
        "sparse_backend_metadata_source_constraints further restrict branches. CUDA SM90 sparse candidates FLASH_ATTN_MLA_SPARSE, FLASHMLA_SPARSE, and FLASHINFER_MLA_SPARSE_SM90 inherit AttentionMetadataBuilder.supports_draft_decode_metadata_update=False through SparseMLACommonMetadataBuilder without overriding it. Any draft attention group using an unchanged v0.30 builder forces the ordinary advancing loop, regardless of other group capabilities/counts. These are explicit backend-source conditions: backend_selection_observed=False and actual_group_membership=None. They neither select an observed backend nor change normal GPU-v2 unsupported boundaries. All-True generic scenarios check the shared-loop predicate only, rather than the SM90 GLM branch. V4 SparseSWA/True capabilities are not applied to GLM. Source constraints passed scoped review 3cd6235a; actual backend selection remains unbound.", "",
        "Formal reference added GPU-v2 metadata 40/40 plus 6 direct checks at 43bb0de3, and refresh 110/110 with generic3direct/sparse1direct at ae4fd782/3cd6235a or dependency-reuse notes in direct logs. Sparse-builder False constraints were independently verified. Acceptance covers typed roles and source host/readiness events, rather than CUDA data edges, actual kernel count, streams/overlap, backend selection, or native pricing. Host ordering between updaters does not establish physical data dependencies. Original snapshots are retained. PP2 reduction placement 92c3dfee passed scoped 52/52 and 2 direct; subsequent producer changes inherit no whole-tree acceptance.", "",
        "Existing Kimi-K2.6 d07fb71b H200 TP8/PP1/EP8 static artifacts are reusable: prefill/decode1/decode2/mixed contain 3681/3622/3562/3864 modeled calls and 0/0/0/0 unpriced calls. This does not establish actual launch/Scheduler acceptance. Independent routed-capacity lower bounds and owner TP8 static weights do not exclude H200 TP8/PP1, but reserves/profile/allocator are unbound; observed available-KV or normal blocks cannot be derived. Existing Kimi TP8/PP2/EP8 production rows bind A800 only; this round retains the smaller TP8/PP1 candidate rather than adding H200 PP2. The compatible v0.30 Scheduler still lacks OpenAI NamespaceTool SDK; downloading, stubbing, or using v0.11/debug blocks cannot fill W4. Exact K2.7/K3/GLM5.3 and DSpark configs remain external-input gaps.", "",
        "Normal-capacity boundary 06c7398b passed independent scoped review. A complete KimiK25 config text_config supplies text-submodel parameters only; it does not prove vision_tower/mm_projector skipping or offload. No pixel_values and ordinary image/video limit0 do not imply vision_chunk skip; this custom source modality defaults to 999. resolve_analytical_kv_capacity_plan rejects non-$ extraction before residency/native allocation, preserving existing complete-root-model paths and source-only text-math diagnostics. Kimi active-outer constructor ledger 8573f835 passed independent source review of all 335 unique keys/bytes. Buffers/post-load/text/KV/reserves are excluded; actual mode/skip/offload/loader remain unbound. No assumed boolean bypasses normal capacity. Direct boundary/root-capacity/pure-PP dependencies 3passed/3.67s do not accept real v0.30 normal capacity.", "",
        "| Current supplementary static artifact | Snapshot | prefill/decode1/decode2/mixed |",
        "|---|---|---|", *current_rows, "",
        "Supplementary V4 artifacts declaring v4_eager_metadata_source=vllm_0_30_eager_dict_all_valid specify eager dictionary metadata with all scheduled rows valid, rather than observed runner state. C4 short contexts dispatch by max_seq_len//4<=TOPK, retaining projections/compressors while filling the full INT32 topk buffer once and omitting indexer Q/scoring. Prefill gathers the original cache union once into one BF16 workspace; query K/V role-use is separate from that write footprint. Physical read reuse and dequant/helper service remain unbound. Default unbound scenarios retain original scoring projections; this is neither INT32 materialized gather nor full native acceptance.", "",
        "| Model | prefill/decode1/decode2/mixed | Scope and gaps |", "|---|---|---|", *rows, "",
        "| Native-on facts | prefill/decode1/decode2/mixed | Scope and gaps |", "|---|---|---|", *native_rows, "",
        "| Gate | Target | Actual | Status |", "|---|---|---|---|",
        *(f"| {gate['gate']} | See GOAL hard gate | {gate['actual']} | Fail |" for gate in gates), "",
        "V4 mHC separates broadcast prenorm, post/prenorm, BIG_FUSE, post, and head. Regular post/pre uses two stages for T≤32 and three for T>32; first-layer broadcast input is H, not a 4H ordinary GEMM. Calls without DeepGEMM binding retain complete dependencies and seconds=null. Unpriced items cannot sum to complete end-to-end latency. Priced items describe source math and logical unique bytes; compiler reductions, repeated reads, and actual DRAM residency require evidence. Midlayer explicitly rejects these mHC stages.", "",
        "New local source input: [bounded DeepGEMM location record](deepgemm_bounded_source_inputs.json). The inspected DeepGEMM package VERSION is 0.1.2, base package 0.1.0; this is not the selected vLLM provider version. Package headers include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh, common/math.cuh, and mma/sm90.cuh support a conditional source program: BF16 A converts from shared fragments to FP32 registers, FP32 B/C, TF32 WGMMA, M64/K64, 128 math threads; 24 outputs need at least N32 in a full single-N-tile. Full-grid padded fragments, square/accumulation work, and 512 subwarp additions per CTA are output separately. FP32 Torch math reference, TF32 source program, and actual execution remain separate.", "",
        "Opaque TMA descriptor bounds/OOB fill, padded backing, selected N, TMA/total block threads, and stages remain unbound. Read-only binary symbols include tf32_hc_prenorm_gemm host/FFI entries, but build path /deepgemm/csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp has no matching local file. vLLM utils/deep_gemm.py prefers external then vendored code; headers and five-argument interfaces do not prove version/host-config/compiler/provider equivalence. No import/load, GPU use, download, or shared-environment change occurred. conditional_sm90_source_program demand/seconds are None; prenorm remains unsupported. Direct tests cover square/accumulation and synthetic M/K padding; formal review of the new snapshot remains pending.", "",
        "Further bounded host-binary evidence: [raw commands and local interpretation](deepgemm_sm90_host_binary_probe/result.json). Read-only review checked the complete local 0.1.2 ELF hyperconnection host function, launch-config helper, and two rodata sections. SM90 checks N≤32, N%8=0, K%64=0; JIT uses M64/K64, math128/TMA128; grid.x=ceil(M/64)*splits, block=(256,1,1), dynamic shared launch reserve 232448B. N24 arithmetic gives blockN32, swizzle128, template-used shared204992B, stages12; template usage and launch reserve remain separate. This local binary ABI/JIT/launch-construction review is neither formal operator-reference acceptance nor actual launch. Actual vLLM settings remain unbound; launch/demand/seconds stay None. TMA descriptor fill, selected provider, missing source provenance, compiler/device, occupancy, numerical equivalence, and time remain incomplete. H200 HWSpec Tensor Core variants are fp16/bf16/fp8 only; BF16 rates cannot price TF32, and hardware rates are unchanged.", "",
        "Bounded read-only review of the same binary descriptor helper confirms preserved global bounds and oobFill0/NONE selecting documented CUDA zero fill. A encodes global[K,M]/box[64,64]. B uses FP32 storage and TFLOAT32 tensor-map dtype; at N24, global[K,24]/box[32,32]. The earlier B box[64,32] interpretation was corrected: swizzle128/4B overwrites the inner box extent, so caller K64 cannot be applied directly. Under default multicast1, tma_copy.cuh uses one A atom64 and two B atom32 per K64. tma_load_source_calls records source calls separately from compiled instructions and DRAM transactions/bytes, without timing. M-tail and N24→N32 zero fill hold only under this local descriptor condition; generic header descriptor fields remain None. Synthetic K32 checks device-template arithmetic only and is unreachable under host K%64 ABI. D split1/multi-split descriptors were checked; device-store OOB behavior remains undeclared.", "",
        "V4 sparse QK was independently corrected to head_dim=512 including RoPE, shared by Serving and Kernel evaluation. GLM native inputs use source work for two separate RMSNorm operations plus concat. TP1 with full vocabulary combines embedding gather; TP8 with sharded vocabulary gathers/all-reduces first. Triton blocks explicitly project local 3.4.0 CUDAOptions defaults, without claiming actual v0.30 compiler binding. Native artifacts retain accepted/rejected arrays and all source gaps; they do not establish normal capacity or real Scheduler completion.", "",
        "Explicit GLM v0.30 performs one Q absorption over all T rows before sparse MHA/MLA dispatch. Prefill/mixed retain this call; decode does not duplicate it. Legacy row counts remain. Source version is part of the layer-template memo signature. The decode-q-down-proj ID is retained for historical comparisons, with all_model_rows declaring scope. Formal layer_stack comparison retains its older snapshot; local implementation/new artifacts do not inherit 5024-check independent acceptance.", "",
        "GLM target uses Roofline-only source projections dsv32_fused_norm_rope and dsv32_fused_q, replacing separate norm/rotary/cache launches absorbed in explicit v0.30. Norm includes two RMSNorms, index LayerNorm/RoPE/FP8 cache, materialized BF16 K/MLA cache, and full-indexer topk sentinel reset (shared does not clear topk). Fused Q includes BF16 Query RoPE, FP8 index Query, and FP32 scaled index weights. H200 SM90 does not satisfy CuTeDSL SM100. Fused-Q Triton num_warps=1 is source-backed; norm block128 still projects local Triton3.4 defaults rather than required Triton3.7.1. Index LayerNorm gamma/beta are explicit FP32; fused-Q literal ledger includes a separate 4B FP32 Q scale. Both retain index_rope_interleave=True and branch-changing BF16 cache/quantize_mqa=False contracts. Padded rows project valid local slots; actual slot/owner/scoring skip, compiler, and runtime remain unaccepted. Unique bytes are not observed DRAM/cache hits.", "",
        f"Reproducible replay of existing formal-reference expectations on current artifacts: {layer_rescore['totals']}; see layer_stack_rescore.json. No expectations or denominators were added. Independent review confirmed original 156 checks at 92fe1b04, then 1904 fusion calls at 3a2f725d and formal comparison in-memory rescoring. Scoped conclusions, formal-reference file updates, and complete source launch/runtime acceptance remain separate.", "",
        f"V4 compressor independent reference adds matching state obligations: original 612 projection plus new 612 state checks, 1224 total. Rescored independent expectations: {v4_rescore['totals']}; see v4_compressor_rescore.json. BF16 A/B remain 2B; GEMM output_dtype=fp32 makes C 4B, while default output_dtype=None normalizes at the schema boundary. Mixed output is Roofline-only; actual injected launch, Midlayer, and PipeWeave reject it. Only known global A/B/C and Tensor FLOPs are priced; missing local SMEM/epilogue service is not observed zero. Downstream updates read FP32 projection and half-width FP32 APE scores, INT64 positions/slots, and write FP32 state. APE reuse across tokens, slot-address/control, and actual valid slots retain limitations. First-layer DeepGEMM remains unpriced.", "",
        f"Formal layer_stack checks: {layer_comparison['totals']}; native local checks: {native_comparison['totals']}. Each original comparison retains its own snapshot; current artifacts do not automatically inherit passes from different snapshots.", "",
        "The V4 FFN candidate follows formal v4_ffn_reference.json, upstream nvidia/model.py GateLinear(out_dtype=float32), MoERunner gate calls, and SiluAndMulWithClamp. Hash replaces ID search only; every layer retains gate projection. Plain mixed-output GEMM prices BF16 A/B and FP32 logits. Shared activation retains swiglu_limit: upper-clip gate, two-sided clip up, then SiLU/multiply. Clamp supports Roofline logical math/unique bytes only; Midlayer/PipeWeave reject it. Independent router ID/weight producers remain unpriced with source math and typed-role reads. Actual provider, modular/monolithic fusion placement, GateLinear/activation backend, and compiled launch remain unbound, rather than observed zero. New snapshots require separate acceptance. C128 attention uses per-query (position+1)//128 states in explicit eager-metadata source scenarios; default remains a candidate upper bound. Accepted finish scopes are not reopened.", "",
        "Formal reference passed scoped FFN 2496/2496 plus 3direct+2clamp at 3c354e6b and C128 visible-state 3direct at 73788e5a; later commits inherit no whole-tree acceptance. The output-projection candidate follows common/ops/fused_inv_rope_fp8_quant.py: inverse RoPE per local head, log2/exp2 per live quant group, FP32 scales aligned to 16/4=4 rows, explicit zero stores for padding. H200 grid=(aligned_T,local_heads), block32. Input, RoPE/quant math, and FP8 output count live rows; FP32 scale writes count aligned rows. Unique/role storage remains distinct from compiler instructions, partner/repeated cache reads, transactions, and DeepGEMM einsum service. Midlayer rejects the unbound detailed model.", "",
        "Formal reference passed scoped output projection 3328/3328 with original 4 counterexamples+2 boundary checks at 30cb98ac, without accepting actual instructions/traffic/capacity/native behavior. Unpriced route producer 0f4183d2 passed 2496 source-role/dependency checks and 2direct, without provider or modular/monolithic placement acceptance. Routed-core clamp follows quantization/mxfp4.py, fused_moe/oracle/mxfp4.py make_mxfp4_moe_quant_config, and fused_moe/activation.py. Swiglu_limit becomes gemm1_clamp_limit; SILU upper-clips gate and two-sided clips up before activation, alpha=1/beta=0. Workload/lowering retain constraints. Roofline counts three clamps, SiLU/multiply ordinary math, exp/reciprocal per live routed row, preserving BF16 materialization and Tensor GEMM. Padded GEMM rows imply no extra activation work. Default None preserves unclamped/INT4 aggregates; explicit clamp is MXFP4-only and rejected by Midlayer/PipeWeave. Actual backend/compiled instructions/local service remain unbound.", "",
        "Formal scoped acceptance covers routed-core clamp 2496/2496 and 2direct at 40bd4b28/9e008651, and Marlin weighted output 1072/1072 and 2direct at d07fb71b, without subsequent whole-tree/runtime acceptance. The separate default weighted-output math gap was corrected: marlin_moe.py sets W2 mul_topk_weights=True; marlin_template.h store_result applies __hmul2 to valid BF16 rows before moe_sum/torch.sum top-k reduction. Selected-weight multiplication counts live routed rows×hidden using existing aggregate ALU logical-operation semantics, rather than padded GEMM rows. Scalar math counts establish neither packed-instruction throughput nor actual backend binding. Tensor FLOPs/storage roles remain unchanged; INT4/MXFP4 share this source branch.", "",
        f"Independent GLM capacity lower bound: {glm_capacity_path}. BF16 target routed parameters alone exceed configured full H200 capacity at TP1/PP1 and TP8/PP1; draft/other weights/cache/reserves can only increase occupancy. Thus f1479578 TP8/PP1 off/native all-rank source graphs are math evidence, not normal Serving cases. TP8/PP2 below uses reference default partition 39,39 (routed layers 36,39) and is not excluded; it is not capacity accounting for explicit 38,40 source candidates. Full post-load residency, transport, and normal allocator remain unaccepted. The table reads independent reference data directly without copying production/reference formulas.", "",
        "| GLM BF16 layout | Routed layers/stage | Routed target lower bound/rank B | Configured HBM B | Excess B | Independent conclusion |",
        "|---|---|---:|---:|---:|---|",
        *[f"| TP{case['tp']}/PP{case['pp']} | {case['target_moe_layers_per_stage']} | {case['rank_routed_target_weight_lower_bound_bytes']} | {case['configured_H200_physical_bytes']} | {case['excess_bytes_if_positive']} | {case['conclusion']} |" for case in glm_capacity["cases"]], "",
        "Independent read-only source-math review accepted GLM f1479578 TP8/PP1 off/native: all-rank actors, EP-off all-expert I/TP, same-rank two-hop topk alias reuse, single group collectives, and memo replay. Formal glm_native_tp_direct.log records 2passed4.13s within that scope/snapshot, not normal capacity, complete alignment, or actual provider/compiler/runtime acceptance. Conditional finite CUDA source program 3e6ca754 also passed scoped review: sigmoid uses tanhf; T<=1024 writes two FP32 shared-score copies, while T>1024 rereads selected bias and subtracts it. Global role rereads remain separate from unique footprint; uint64 local-sort comparisons/warp reductions do not become hardware instructions or rates. Nonfinite branches and actual tanhf/compare/reduction compiled service lack evidence. Demand/seconds remain None, routing unpriced, and source counts do not establish end-to-end Roofline acceptance.", "",
        "Independent read-only review accepted explicit Kimi d7e4356a vLLM0.30 Gate/routing within scope. Exact config does not require FP32 router output: GateLinear.out_dtype=None uses BF16 F.linear. Plain GEMM prices projection; separate kimi_moe_route reads BF16 logits/FP32 bias and emits FP32 weights/INT32 IDs. Same-rank gate→route→existing INT4 core dependencies remain. CUDA shim forwards BF16 scores; preprocess internally promotes to FP32. Torch-branch BF16 scoring/weights, FP32 bias selection, and final export remain separate without inventing FP32 logits copies. Conditional 384/8 CUDA launch is grid=T/block256/shared3072B for T<=1024, otherwise grid=min(ceil(T/8),2048)/block256/shared0; neither is assigned to KernelSpec. Each phase has 60 routes/rank, 480 unpriced total. TP8 four-phase calls change from 66d9ab0f 8420/8788/8728/9884 to 8900/9268/9208/10364 solely by 480 required producers. Gate SFU is 0 on both sides; no removed SFU overcount is claimed. T127 likewise has 480 routes. Compiler/provider/fusion/PDL/runtime/nonfinite/physical traffic remain unbound; complete Roofline fails. Direct dependencies passed 6 in kimi_v30_routing_failed_recheck.log; failed cases ultimately passed 3/5.74s in kimi_v30_routing_dtype_failed_recheck.log. Original failures and prior source/parameter-ledger snapshots remain.", "",
        "Independent source-work review accepted final RMS fragment reduction in V4 mHC 545ea688. Both regular/broadcast TileLang sources reduce gcd(1024,H) FP32 elements after per-position squares/local accumulation; T*(gcd(1024,H)-1) logical additions follow existing Sinkhorn aggregate ALU semantics. Existing collapse, square/accumulation, two RMSNorm multiplications, and Sinkhorn rounds remain. Compiler padding/shuffle/reduction instructions are unbound. Original 5 positive dependencies passed; 2 new fragment tests failed only because test replacement omitted elements alias. Only those 2 were rerun after correction: v4_mhc_rms_fragment_failed_recheck.log, 2passed2.57s. Production math changes triggered the five-hardware 545ea688 offline Kernel matrix, all PASS. Canonical matrices do not accept new mHC hardware/full Serving behavior. First-layer mhc_pre_broadcast_tilelang forces use_tilelang_fallback=False; missing actual DeepGEMM cannot be replaced by Torch reference or ordinary GEMM launch/rates.", "",
        "Same-scenario 8eab4256→545ea688 V4 T8 modular comparison: Flash 344 norms add ALU 2815296/351912/351912/3167208 across four phases; Pro 976 norms add 7987584/998448/998448/8986032. Calls, FMA/SFU, and logical bytes match. Flash retains 4 and Pro 8 unpriced DeepGEMM prenorms; complete static end-to-end acceptance is absent. Actual launch/compiler/provider/normal capacity/DSpark remain unbound. Duration/errors never selected the reduction formula.", "",
        "V4 routed loader payloads: deepseek-v4-flash_routed_loader_payload.json and deepseek-v4-pro_routed_loader_payload.json retain their generation snapshots. The explicit non-MegaMoE/EP-off/PP1 source scenario has aligned Marlin H256/I128: every rank retains all experts with I/TP sharding, packed uint8 uses 1B per two FP4 values, and UE8M0 scales use 1B per 32 input values. Only Mxfp4MoEMethod.create_weights routed payload is counted, not post-load residency. Pro TP1 packed weights alone exceed H200 capacity. FlashTP4/ProTP8 are not excluded by this subset, without normal-capacity acceptance. Non-routed weights, transformed scales/padding/aliases/workspace, DSpark draft, runtime reserves/profile/allocator, and provider remain incomplete. Production model_rank_residencies still rejects V4; no available-KV/blocks are exported.", "",
        "Explicit V4 v0.30 EP-off uses non-MegaMoE FusedMoEFactory: TP remains, ep_rank=0, every rank owns all experts with I/TP and TP-sharded shared MLP. Modeled rank-local inputs, compressor/indexer, attention/head instantiate per rank; each collective remains one group call. Outer V4 n_local_experts metadata, loaded weights/provider, and runtime remain unbound. Layer-template signatures distinguish layer0 broadcast/later mhc_post_pre and hash/ordinary routing; two memo graphs match uncached nodes/edges. Memo fixes and static metadata scenarios remain separate: old d07 T8 declares eager-valid metadata, while the new default is None. C4 default retains indexer-wq-b and phase-specific score-topk; eager short context fills once. Call-count differences across these scenarios cannot be attributed to memo or compared as identical metadata. Both scenarios/snapshots remain; route producers and first broadcast prenorm stay explicitly unpriced.", "",
        "Current V4 source records use actual v0.30 attn/ffn registration modules, compressor.fused_wkv_wgate weights, and DeepseekV4Attention/Indexer/MLP call sites; mHC points to actual TileLang functions. Private layer-template weights explicitly use routed=mxfp4, shared=fp8. Only source labels and memo identities change; no formulas/rates/launch facts are added. Old f24 artifacts retain original snapshots/labels without current-source binding acceptance. Source names alone do not bind instances, checkpoint addresses, loader mappings, compiler, or backend.", "",
        "GLM BF16 TP8/PP2 target-off candidates explicitly use VLLM_PP_LAYER_PARTITION=38,40. Default 39,39 starts stage2 at a shared-indexer layer, whose guard rejects missing local full-indexer input. V32 vLLM0.30 forwards hidden_states/residual; PP scatter/send/receive/all-gather count two BF16 tensors. The candidate removes stage0 layer37 terminal MoE all-reduce and reduces hidden at receiving stage1 before layer38 input norm, which separately waits for residual gather. Stage-tail blocks are explicit rather than memo/repeated old reduction tails. EP1 follows FusedMoEParallelConfig EP-off/TP rules: all experts and router counts per rank, I/TP intermediate, ep_rank=0, TP-sharded shared MLP, lane readiness linked to matching partial producers. Modeled non-MoE math also instantiates per TP rank with local indexer/projection, attention/dense/head shapes and same-rank topk/compute dependencies. A collective remains one group call waiting for all rank producers. Artifact-specific snapshot declarations define coverage; historical labels remain. Actual CUDA producer/data-edge/fusion/provider are unbound, and default partial-transport risk remains. V32 MLP reduce_results=False retains TP-partial hidden; GPUWorker default sharded send/all-gather assumes replicated inputs and loses coordinates of differing rank partials that later all-reduce cannot restore. Correct reduction order does not prove transport correctness; actual overrides/compiler transformations are unbound. Scope is graph construction/static emission only, without Scheduler acceptance; PP2 native-on terminal verification rejects. NVLink is explicit topology, not an observed 16-card machine. Normal capacity/reserves/cache/runtime remain unverified; all W gates fail.", "",
        *[line for scope in current_pipeline_scopes for line in (scope, "")],
        f"Independent reference: {contract_path}; review: {review_path}. operator_alignment.json records current facts and denominators only; it neither creates its own reference nor claims independent acceptance.", "",
        "Every static JSON retains exact commit and dirty state. Artifacts from different snapshots keep original labels and are not current-HEAD results. Commands, tests, Kernel matrices, and external-gap handling are in cli_handoff.json. v0.11/debug-block Serving cannot replace W4; classic MTP cannot replace DSpark.",
    ]
    v4_weight_records = [record for record in weight_records
                         if all(case["payload_scope"] is None for case in record["cases"])]
    kimi_weight_records = [record for record in weight_records
                           if all(case["payload_scope"] == "kimi_active_outer_registered_parameters"
                                  for case in record["cases"])]
    glm_weight_records = [record for record in weight_records
                          if all(case["payload_scope"] == "glm_parameter_source_projection"
                                 for case in record["cases"])]
    if len(v4_weight_records) + len(kimi_weight_records) + len(glm_weight_records) != len(weight_records):
        raise ValueError("constructor diagnostic cannot mix source payload scopes")
    if glm_weight_records:
        text.extend(["", "## GLM source parameter ledger (normal capacity unaccepted)", "",
            "Full-indexer k_norm uses vLLM LayerNorm with FP32 weight/bias. The 38,40 stages have 11/10 copies, adding 5632/5120B per TP rank over the old BF16 ledger. Default separate draft vocabulary remains; explicit vllm_0_30_0 BF16 shares MTP LM head per legacy/GPU-v2 source, also sharing embedding at PP1 and retaining draft embedding at PP>1. Three draft RMSNorms plus eh_proj/decoder remain separate. Input embedding assumes VLLM_REPLICATE_EMBED off. Parameter aliases and occupancy below HBM do not establish loader/native/normal-KV acceptance. PP2 transport, reserves/profile/allocator/Scheduler gaps remain.", "",
            "| Artifact / snapshot | TP | PP | MTP inventory | Version | Stage ranges | Maximum parameter B/rank | Physical HBM B |",
            "|---|---:|---:|---|---|---|---:|---:|"])
        for record in glm_weight_records:
            for case in record["cases"]:
                ranges = sorted({(rank["layer_start"], rank["layer_end"])
                                 for rank in case["rank_residencies"]})
                text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {case['tp']} | {case['pp']} | {case['mtp_mode']} | {case['model_execution_version']} | {ranges} | {case['maximum_owner_weight_bytes']} | {case['physical_hbm_bytes']} |")
        text.extend(["", "89fe70ba corrected affine storage; glm_indexer_affine_direct.log: 4passed3.47s. fe8c6e8c added explicit version API/diagnostics; glm_v30_vocab_alias_direct.log initially 3failed3passed3.41s because the guard used V4-only expert_dtype. Existing GLM weight_dtype validation corrected this; only 3 failed aliases reran, 3passed3.40s in glm_v30_vocab_alias_failed_recheck.log, reusing 3 positive dependencies. Review also found missing runtime-binding flag validation; true or missing fields now fail closed, glm_parameter_binding_boundary_direct.log: 2passed3.42s. Kernel demand/launch remain unchanged, reusing 545ea688 five-hardware matrix. No new full suite or real Scheduler/QPS16 ran. Read-only review verified both proposer sharing sources; the new candidate awaits independent snapshot review, without whole-tree inheritance.", ""])
    embedding_records = [record for record in current_records
                         if any(phase["source_vocab_embedding_calls"] for phase in record["phases"])]
    if embedding_records:
        text.extend(["", "## CUDA vocabulary-embedding source branch (unpriced)", "",
            "2172e36a corrected BF16 GLM vLLM0.30 TP>1 target/native embedding. CUDA with UnquantizedEmbeddingMethod selects fused VocabParallelEmbedding. Target, legacy proposer, and GPU-v2 pass INT32 IDs; scalar widening inside the kernel allocates no INT64 input copy. 91eefd21 applies the same source to existing Kimi-K2.6/V4 H2000.30 branches: Kimi compressed-tensors targets Linear only, with no matching VocabParallelEmbedding scheme; V4 Fp8Config.get_quant_method returns None for input embedding. Both retain unquantized BF16. Each rank reads owned weight rows only but writes full BF16[T,H] partial outputs (zeros for nonowners), then one group all-reduce. Logical GATHER/embedding interfaces remain; special lowering emits VECTOR/vocab_parallel_embedding rather than borrowing generic gather flat launches or materialized_output=False pricing. TP1 native retains folded embed_eh_norm.", "",
            "vocab_parallel_embedding_kernels.cu fixes block256, grid_x=min(ceil((row_bytes/vector_bytes)/256),256), grid_y=min(T,4*SM_count). Pointer/row alignment selects 16/8/4/2/1B vectors; H6144 BF16 conditional grid_x values are 3/6/12/24/48. Actual IDs, owner counts/rank, alignment, compiler, and allocator remain unbound. KernelSpec stays unsupported with launch=None and demand/seconds=None. Source work retains full output, INT32 inputs, and conditional weight-read bounds, without treating role bytes as observed DRAM/cache hits.", "",
            "| Artifact / snapshot | Embedding calls (prefill/decode1/decode2/mixed) | Required BF16 partial-output writes B (same order) |",
            "|---|---|---|"])
        for record in embedding_records:
            counts = "/".join(str(phase["source_vocab_embedding_calls"]) for phase in record["phases"])
            writes = "/".join(str(phase["source_vocab_embedding_partial_output_bytes"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} | {writes} |")
        text.extend(["", "Same-scenario f1479578 PP1 off/native and 3e6ca754 PP2 off→2172e36a preserve every call ID/dependency and nonembedding prediction. Per-phase replacement counts are 8/24/8 generic gathers, with unpriced counts 600→608, 616→640, 600→608. Read-only review independently confirmed source counterexamples; candidate implementation review stopped at reviewer tool limits, so formal reference remains pending without inherited broad acceptance. glm_vocab_embedding_source_direct.log: 5passed6.95s; PP2 dependency: 1passed7.00s; 2172e36a five-hardware offline Kernel matrix: all PASS, 36.7s. No GPU, real Scheduler/QPS16, or new full suite ran.", ""])
        text.extend(["Same-scenario 91eefd21 comparison against Kimi d7e4356a and V4 modular545ea688 preserves call IDs/dependencies and nonembedding predictions. Three models replace 8/4/8 generic gathers per phase, changing unpriced 480→488, 4→8, 8→16. kimi_v4_vocab_embedding_direct.log: 8passed7.64s, including 3 Kimi phases, FlashTP4/ProTP8 all ranks, and original routing dependencies. Routing math/independent scopes remain. Only Serving source dispatch expanded; predictor changes were docstrings only, with unchanged pricing helpers/KernelSpec/launch formulas. 2172e36a Kernel matrix is reused; no new full-suite/real-Scheduler/QPS16/GPU validation occurred. Full-parent residency, V4 post-load, and DSpark gaps remain; new-snapshot acceptance awaits primary/reference review.", ""])
    target_selection_records = [record for record in current_records
                                if any(phase["target_logits_selection_calls"] for phase in record["phases"])]
    if target_selection_records:
        text.extend(["", "## vLLM target logits row selection and logical Roofline scope", "",
            "At the final PP stage, vLLM0.30 GPUModelRunner.execute_model selects hidden_states[logits_indices] before compute_logits. Its independent materialized BF16 output cannot be omitted merely because LM head reads fewer rows. Ordinary query_start_loc is INT32; speculative cu_num_tokens comes from NumPy default INT64 cumsum and retains INT64 through _calc_spec_decode_metadata/async_tensor_h2d. Explicit eager/H2000.30 source scenarios select verified-draft and bonus rows from scheduled-request tails. Target selection includes unverified draft rows rather than proposer post-rejection retained-row rules.", "",
            "MTP-disabled source scenarios use ordinary INT32; verified-draft scenarios use speculative INT64. MTP-active with zero verified tokens cannot determine whether Scheduler dictionaries are empty, so index dtype/bytes remain None with 4–8B/row bounds. These are schedule source scenarios, not observed SchedulerOutput. Each rank reads selected BF16 rows and writes complete sample_hidden_states; PP2 creates same-rank norm/select/head edges only at the final stage. Known index width permits unified predict(...,method=roofline) pricing of two BF16 rows plus index-role bytes without grid/block; unknown width rejects point estimates. Launch=None and execution_support=unsupported remain; Midlayer/Oracle/PipeWeave and detailed resource_input reject. Address/control, actual DRAM/cache, provider/library/compiler, and normal runtime remain unaccepted. prediction.scope=logical_roofline_only_no_launch_binding cannot pass full phases or W gates. Historical artifacts retain unpriced states.", "",
            "| Artifact / snapshot | Selection calls (prefill/decode1/decode2/mixed) | Materialized BF16 output B (same order) | Index source (same order) | Logical Roofline-only calls (same order) |",
            "|---|---|---|---|---|"])
        for record in target_selection_records:
            phases = record["phases"]
            counts = "/".join(str(phase["target_logits_selection_calls"]) for phase in phases)
            output_bytes = "/".join(str(phase["target_logits_selection_output_bytes"]) for phase in phases)
            index_sources = "/".join(str(phase["target_logits_selection_index_sources"]) for phase in phases)
            roofline_counts = "/".join(str(phase["target_logits_logical_roofline_calls"]) for phase in phases)
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} | {output_bytes} | {index_sources} | {roofline_counts} |")
        text.extend(["", "First-stage boundary: nonfinal GLM V32 PP stages return unreduced hidden_states/residual. Ordinary eager GPUWorker supplies empty all_gather_tensors; GroupCoordinator selects sharded send/receive/all-gather when numel divides TP. Per-tensor False overrides exist, but GPUWorker SP sets only residual and V32 nonfinal stages reject SP. The missing transport fact is matching sender/receiver hidden_states overrides or a proven equivalent reduction-placement transformation; API existence does not establish selection. Per-rank embedding weight reads depend on owned_token_rows. Known IDs/owner counts suffice for logical bytes without pointer alignment; neither 0..T bounds nor TP averages are exact reads.", ""])
        text.extend(["", "target_logits_selection_glm_direct.log: 9passed7.11s covering 3 index-storage branches, fail-closed boundaries, TP1/TP8 target/native dependencies, and memo. target_logits_selection_kimi_v4_direct.log: 7passed7.58s covering Kimi 3 phases, all-rank Flash/Pro producers, and original routing dependencies. target_logits_selection_pp2_direct.log: 1passed6.93s, selecting only final-stage ranks8–15. New source math adds known copy roles/dtype lower bounds without production-duration tuning. New-snapshot formal acceptance awaits primary coordination; older scoped passes do not inherit.", ""])
    fused_embed_records = [record for record in current_records
                         if any(phase["target_fused_embed_norm_calls"] for phase in record["phases"])]
    if fused_embed_records:
        text.extend(["", "## GLM TP1 embedding source logical Roofline", "",
            "Correction to 64471903: GLM V32 target has_full_vocab_on_rank=True at TP1/unquantized, so Model.forward calls fused_embed_norm and bypasses VocabParallelEmbedding.forward. Fused target reads INT32 IDs, BF16 weight rows, and first-layer gamma, writing BF16 residual and normalized attention input. Logical operand bytes are 4T+6TH+2H; gamma objects and repeated CTA reads remain separate. No global INT64 conversion or separate layer0 input norm. Declared source grid=(T,), num_warps=min(32,max(4,next_power_of_2(H)//512)), rather than observed local runtime/compiler settings. Valid full-vocabulary IDs need no TP ownership distribution; native TP1 draft retains fused_embed_eh_norm. Historical 64471903 artifacts/reviews cover leaf embedding only, not GLM model dispatch.", "",
            "| Artifact / snapshot | Target fused calls(prefill/decode1/decode2/mixed) |", "|---|---|"])
        for record in fused_embed_records:
            counts = "/".join(str(phase["target_fused_embed_norm_calls"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} |")
        text.append("")
    residual_norm_records = [record for record in current_records
                             if record["model"] in ("zai-org/GLM-5.2", "moonshotai/Kimi-K2.6")
                             and any(phase["residual_add_norm_calls"] for phase in record["phases"])]
    if residual_norm_records:
        text.extend(["", "## GLM / Kimi residual-add RMSNorm", "",
            "Later V32 input norms, every post-attention norm, and final norm consume hidden_states/residual through fused_allreduce_rms_norm, updating residual and normalized output. TP1 calls norm(hidden,residual); TP>1 actual FlashInfer fusion versus explicit all-reduce fallback remains unbound. cc0c4218 reuses add_rmsnorm and adds memo semantic keys without predictor/rate changes. First-layer no-residual branches remain plain or embedding-fused; first MTP decoder residual=None remains plain, MTP post-attention becomes add, existing head add remains. Kimi-K2.6 constructs DeepseekV2ForCausalLM; its later input/post-attention/final norms also consume residual. 88f488aa applies the same existing add variant while retaining V2 source symbols, without assuming V32 all-reduce fusion or changing Kimi routing/EP/embedding. Tables list recorded variants/public logical bytes at original snapshots; old artifacts inherit no new semantics. PP-preserving transport and full-phase unpriced gaps remain.", "",
            "| Artifact / snapshot | add_rmsnorm calls (prefill/decode1/decode2/mixed) | Logical bytes (same order) |",
            "|---|---|---|"])
        for record in residual_norm_records:
            counts = "/".join(str(phase["residual_add_norm_calls"]) for phase in record["phases"])
            byte_counts = "/".join(str(phase["residual_add_norm_logical_bytes"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} | {byte_counts} |")
        text.append("")
    target_sampling_records = [record for record in current_records
                               if "target_sampling_source" in record
                               and record["target_sampling_source"] == "vllm_0_30_plain_sampler_bf16"]
    if target_sampling_records:
        text.extend(["", "## Required FP32 materialization in the ordinary target sampler", "",
            "Explicit target_sampling_source=vllm_0_30_plain_sampler_bf16 covers only ordinary Sampler.forward logits.to(torch.float32) before apply_logits_processors. It reads BF16 logits after LogitsProcessor TP gather/original-vocabulary slicing and writes independent FP32 logits, 6×rows×vocab logical bytes. Each final-stage TP rank retains its producer. No library launch is required by this projection; execution_support=unsupported remains, without claiming conversion instructions, physical DRAM, or downstream sampler completeness. Extra raw_logits/logprobs materialization, penalties/temperature/top-k/top-p, sampling, and output-ID conversion remain uncovered. MTP rejection bonus/target indexing, clone, and cast are separate programs; this scenario rejects speculative requests. Default model-only behavior remains, without provider/normal-capacity/real-Scheduler/W-gate promotion.", "",
            "| Artifact / snapshot | Cast calls (prefill/decode1/decode2/mixed) | Logical bytes (same order) |",
            "|---|---|---|"])
        for record in target_sampling_records:
            counts = "/".join(str(phase["target_sampler_cast_calls"]) for phase in record["phases"])
            byte_counts = "/".join(str(phase["target_sampler_cast_logical_bytes"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} | {byte_counts} |")
        text.append("")
    rejection_records = [record for record in current_records
                         if "target_sampling_source" in record
                         and record["target_sampling_source"] == "vllm_0_30_rejection_raw_logits_bf16"]
    if rejection_records:
        text.extend(["", "## Conditional materialization producers in rejection sampling", "",
            "Explicit target_sampling_source=vllm_0_30_rejection_raw_logits_bf16 declares present SpecDecodeMetadata, BF16 full-vocabulary logits, and nonprocessed logprobs, rather than observed Scheduler metadata/sampler settings. Bonus rows equal full batch requests; target rows sum mtp_verified_tokens, without accepted-count reduction. RejectionSampler indexes BF16 bonus/target separately. Sampler max_num_logprobs=-1/raw_logits override forces an independent raw FP32 bonus cast; the working cast still reads BF16 bonus and cannot alias FP32 to remove it. Target casts independently then clones FP32 working logits to protect raw logits from later in-place changes.", "",
            "Metadata bonus/target indices use cumsum_dtype=np.int32 and read 4B per selected row. Earlier hidden logits_indices use scheduled-token INT64 cumulative arrays; these roles are distinct. In this explicit metadata-present scenario, hidden selection declares speculative INT64 even at verified=0. Default unbound scenarios retain 4–8B bounds. Zero target rows omit only selections/casts/clones without logical reads/writes, without implying zero library launch.", "",
            "Source-role totals per final-stage TP rank are bonus_rows×(16×vocab+4) + target_rows×(18×vocab+4) bytes, counting typed materialization only. Bonus sampling, processors/constraints, rejection, optional outer logprobs, metadata H2D, library/compiler/provider, actual streams, and normal capacity remain uncovered. New calls retain launch=None and execution_support=unsupported, supporting logical Roofline only. Complete phases remain partial.", "",
            "| Artifact / snapshot | Stage counts (prefill/decode1/decode2/mixed) | Logical bytes (same order) |",
            "|---|---|---|"])
        for record in rejection_records:
            counts = "/".join(str(phase["rejection_materialization_stage_counts"]) for phase in record["phases"])
            byte_counts = "/".join(str(phase["rejection_materialization_logical_bytes"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} | {byte_counts} |")
        text.append("")
    sampling_records = [record for record in current_records
                        if "mtp_draft_sampling_source" in record
                        and record["mtp_draft_sampling_source"] is not None]
    if sampling_records:
        text.extend(["", "## GLM legacy greedy source scenario (partial logical Roofline)", "",
            "Explicit mtp_draft_sampling_source=vllm_0_30_legacy_full_logits_greedy selects full-logits greedy source behavior. Default BF16 LM-head output and TP full-vocabulary gather feed argmax INT64 IDs; .int() produces INT32 only before the next round. First-pass query-end minus rejected counts selects INT32 row indices. Sample/recurrent hidden each materialize BF16 output even when sharing a source tensor; positions materialize INT64. Later [:batch_size] are views without repeated indexing. Before advancing, input_ids/hidden_states copy to buffer fronts. Head depends on sample output; next-round norm depends on hidden copy.", "",
            "7aa185a1 artifacts cover argmax/cast only; 87c0977b adds hidden/position selection and buffer copies. Later topk compaction artifacts retain separate snapshots. Legacy propose runs compact_topk_indices before depth1 early return. With model.index_share_for_mtp_iteration enabled, full INT32[B,index_topk] gathers into a temporary then copies to front, retaining two reads/two writes. Next-pass shared indexer depends on that front producer. Depth1 still has both topk stages; only multiple rounds add position/recurrent-hidden selection and input-ID/hidden copies. Disabling sharing removes topk compaction. GPU-v2 depth>1 storage conclusions retain their separate control semantics.", "",
            "First-pass set_inputs_first_pass without extra slots shifts INT32 IDs across T-1 rows, scatters each request INT32 next ID through INT32 token_indices, then copies full INT64 positions and BF16 target post-norm hidden. CPU/padded prepare_next_token_ids outputs are INT32. Conditional padded-prepare roles exist; target sampling/backup H2D service remain unpriced. T1 omits empty shift only, retaining scatter/position/hidden work. The scenario requires the whole scheduled batch in the proposer and rejects partial subsets. method=mtp, parallel_drafting=False, pass_hidden=True, and no M-RoPE/aux-hidden/padded-disable/extra-slots are explicit conditions, not actual SpeculativeConfig evidence.", "",
            "First set_forward_context calls _get_slot_mapping and fully copies common target INT64 slots into separate proposer storage. Capture-disabled/no-extra-input-slot scenarios add no tail PAD fill. Later _get_slot_mapping without input returns shared per-draft-layer buffer slices/dict views, without repeated copies or invented host/view GPU kernels. Advancing metadata writes slot buffers; first copy and later views remain separate. _create_draft_vllm_config independently selects speculative attention backend, autoselecting at None rather than inheriting target backend. build_per_group_and_layer_attn_metadata invokes build_for_drafting per actual group; layers share only their group object. Base build defaults common_prefix_len=0,fast_build=True. Concrete builders/backends may add GPU work; unbound groups/providers prevent observed-zero or complete-launch claims.", "",
            "Target Sampler/RejectionSampler outputs are INT32. Padded prepare-next conditionally reads sampled/backup IDs by discard/validity and always writes INT32 next IDs/valid counts. Unknown data branches retain conditional read bounds rather than guessed discard bits. At least one ID read per row gives global-byte bounds 13B to 13+4*sampled_width B, separate from parent backup H2D4B. BLOCK_SIZE_TOKENS=next_power_of_2(width), grid=B, warps unbound; no launch assigned. Prepare-sample-indices distinguishes ordinary query-end minus one, padded rejected-row arithmetic, and unbound metadata. Current/previous padded cu-prefix elements share one INT32 object without duplicate unique footprint. This producer feeds initial scatter and hidden/position/topk selections; counts/selected indices remain schedule-source assumptions.", "",
            "Before multi-round advancement, seq_lens-=num_rejected executes once. With unbound metadata, presence and three INT32 read/write roles remain conditional; explicitly present padded rollback still executes when rejected=0. Each eagle_step_slot_mapping_metadata advance reads INT64 position plus INT32 block-ID/seq_len, 16B per real request, and writes INT64 position/slot plus INT32 seq_len, 20B. Dummy rows write 8B slots only. First input positions come from materialized selection; later inputs are output-buffer views. Alias changes unique extent without removing read/write roles. Exceeded caps set position=0,seq_len=1,slot=PAD while still reading block0. State helpers require explicit cap/block/table/values and reject missing/invalid inputs; actual_allocator_binding=False. Nominal artifact KV lengths do not bind real runtime caps; actual slot/page/control data remain unaccepted.", "",
            "Final torch.stack(draft_token_ids_list,dim=1) reads separate INT64[B] results from every pass and writes materialized INT64[B,depth], retaining all argmax dependencies. No final INT32 feedback cast is added. Depth1 returns a view without stack writes. Role bytes/argmax comparison lower bounds prove neither DRAM/cache nor PyTorch/library launch. New calls remain unsupported with launch/demand/seconds=None; full static results remain partial. Exact SpeculativeConfig/index-sharing overrides, local/probabilistic/GPU-v2 branches, heterogeneous vocabulary, watermarker/estimator, head dtype/scale/soft-cap overrides, target sampler/backup H2D, and actual slot/metadata/allocator bindings remain missing.", "",
            "Known typed copy/row-select/scatter/INT64-to-INT32 cast/final stack use unified Roofline logical read/write bytes without grid/block. Only pure-materialization mathematical floating-point work is zero; integer conversion, address/control, and compiled service are not claimed zero. Launch=None and execution_support=unsupported remain. Argmax comparison trees, conditional preparation, and slot/state updates are outside this closure. Historical artifacts retain unpriced states; native phases remain partial. PP1 source graphs cannot bypass excluded normal-capacity boundaries.", "",
            "Known ordinary/padded index preparation and explicitly present padded rollback additionally price typed-role bytes and source INT32 subtract/add/compare/select. Padded preparation includes tl.where selection and cannot omit it. Existing H200 compute_alu INT32 capacity is reused without rate changes. Unknown metadata, absent rollback, and slot/state remain rejected, without actual launch/compiler/native acceptance.", "",
            "| Artifact / snapshot | Source | Source-stage counts per phase | Pure-materialization logical Roofline calls per phase | INT32 bookkeeping logical calls per phase |",
            "|---|---|---|---|---|"])
        for record in sampling_records:
            counts = "/".join(str(phase["draft_sampling_stage_counts"]) for phase in record["phases"])
            logical_counts = "/".join(str(phase["draft_materialization_logical_roofline_calls"]) for phase in record["phases"])
            int32_counts = "/".join(str(phase["draft_int32_logical_roofline_calls"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {record['mtp_draft_sampling_source']} | {counts} | {logical_counts} | {int32_counts} |")
        text.extend(["", "glm_legacy_greedy_source_direct.log: 2failed3passed7.49s, from two ScheduledRequest fixtures with raw_model_output_tokens=None. Corrected failed cases passed 2/6.93s in glm_legacy_greedy_source_failed_recheck.log, reusing original 3 positive dependencies. Both 7aa185a1 four-phase native probes completed with 80/664 unpriced calls per phase at TP1/TP8. Five-hardware offline Kernel matrix all PASS. Later selection/copy checks and latest matrix/probes follow validation and table snapshots. New-snapshot formal review awaits primary coordination, without inherited older acceptance. Real Scheduler/QPS16, normal KV, GPU, and final full suite were not run.", ""])
        text.extend(["glm_legacy_feedback_direct.log: 1failed4passed7.36s. Four positive checks cover TP1/TP8 mixed rejected-row graphs/memo, depth1, and typed sampler contracts. A normalized-shape dtype edit triggered dtype_bytes before the intended boundary. After correction, glm_legacy_feedback_failed_recheck.log still had 1failed3.26s because the supported fixture lacked LaunchConfig and hit a public guard first. Adding complete synthetic launch yielded 1passed3.58s in glm_legacy_feedback_boundary_recheck.log. Only fixtures changed, preserving 4 positive results without relaxing contracts or removing counterexamples.", ""])
        text.extend(["glm_legacy_topk_compaction_direct.log: 4passed7.25s covering TP1/TP8 mixed rejection, next-round same-rank front producers, memo, full INT32 temporary/copy bytes, depth1, and sharing disabled. KernelSpec reuses source-only selection/copy variants, without new production KernelKind or generic INT32 lowering. This source subset remains separate from library/runtime gates.", ""])
        text.extend(["glm_legacy_stack_direct.log: 4passed7.16s, covering all TP1/TP8 pass dependencies, no depth1 materialization, and three-pass typed-byte boundaries. Adds only source-only VECTOR/dsv32_draft_token_stack, without new KernelKind or pricing unknown library launches.", ""])
        text.extend(["glm_legacy_first_inputs_direct.log: 4passed7.37s, covering typed-role bytes, indexed-write boundaries, TP1/TP8 mixed-rejection (7,8) scatter, full target-norm dependencies, memo, empty T1 shift, and partial-proposer rejection. Source-only VECTOR/dsv32_draft_id_scatter reuses copy variants; no new KernelKind/generic INT32 lowering or library-launch timing.", ""])
        text.extend(["glm_legacy_padded_prepare_direct.log: 4passed7.26s, covering conditional reads, INT32 outputs, unbound SIMD width/warps, distinct unique/repeated cu-prefix reads, and TP1/TP8 producer edges. After adding total 13B lower and 13+4*width B upper bounds, only helper/report boundaries reran in glm_legacy_padded_prepare_bounds_report_direct.log: 3passed4.23s, reusing 3 graph positives. Backup H2D is role-only; actual sampler/slot/allocator/compiler remain unbound. Unobserved metadata cannot establish ordinary/padded actual launches.", ""])
        text.extend(["glm_legacy_step_metadata_direct.log: 5passed7.32s, covering typed mixed-role bytes, padding/position aliases, cap reset/block0 source state, invalid/missing input rejection, TP1/TP8 edges, and no depth1 update. Three-pass position-buffer reuse, single rollback, and report boundaries passed 3/3.52s in glm_legacy_step_alias_report_direct.log. Older GPU-v2 state scopes retain original semantics without inheritance/rewrite. Source-only VECTOR/dsv32_draft_seq_rollback/step_metadata remain unpriced, without new KernelKind/normal allocator support. Host metadata reconstruction/CPU upper bounds, actual maxlen/blocks/slot/backend, compiler, and runtime remain missing.", ""])
        text.extend(["glm_legacy_initial_slot_copy_direct.log: 4passed7.52s, covering full INT64 reads/writes, first-round-only TP1/TP8 mixed/single-depth behavior, and memo. Reuses source-only copy without new KernelKind or generic INT64 lowering; actual slots/library launch remain unbound. Independent source-backend selection and host-metadata gaps remain, without inherited target/GPU-v2 provider acceptance.", ""])
    if kimi_weight_records:
        text.extend(["", "## Kimi active-outer constructor parameters (normal capacity unaccepted)", "",
            "Explicit enabled-tower/no-offload source scenario counts every registered Parameter as BF16; no image input does not remove the tower. VisionConfig defaults/projector width normalization come from vLLM0.30 source. Patch/norm/projector replicate; vision weights shard by explicit weights/data scenarios, retaining full parameters when heads do not divide. This outer ledger excludes text weights, buffers, post-load, reserves, and allocator. Normal KV still rejects text_config; no available-KV/blocks are derived.", "",
            "| Artifact / snapshot | TP | Vision source mode | Outer parameter paths | Outer registered B/rank |",
            "|---|---:|---|---:|---:|"])
        for record in kimi_weight_records:
            for case in record["cases"]:
                text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {case['tp']} | {case['vision_tensor_parallel_mode']} | {len(case['outer_loader_parameter_bytes_per_rank'])} | {case['outer_constructor_parameter_payload_bytes_per_rank']} |")
    if v4_weight_records:
        text.extend(["", "## V4 source constructor parameter ledger (normal capacity unaccepted)", "",
            "These records count registered parameters only under explicit FlashMLA/Marlin, EP-off/PP1, default BF16 dtype. Post-load transformations, aliases, buffers, scratch, and reserves are excluded. Actual checkpoint/loader/provider are unbound; totals below physical capacity do not establish normal available-KV. Production V4 capacity still rejects.", "",
            "| Artifact / snapshot | TP | Routed packed B | Loader scales B | Non-routed B | Constructor total B | Physical HBM B |",
            "|---|---:|---:|---:|---:|---:|---:|"])
        for record in v4_weight_records:
            for case in record["cases"]:
                text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {case['tp']} | {case['routed_packed_bytes_per_rank']} | {case['routed_loader_scale_bytes_per_rank']} | {case['non_routed_loader_bytes_per_rank']} | {case['constructor_parameter_payload_bytes_per_rank']} | {case['physical_hbm_bytes']} |")
        text.extend(["", "Independent scoped review accepted 646a9575 non-routed inventory: Flash1330/Pro1894 unique registered Parameter paths match source. Post-load storage is excluded. First-layer finalize_mhc_broadcast_weights separately allocates FP32 sum output. DeepGemmFp8BlockScaledMMKernel uses deepgemm_post_process_weight_scale_block to convert UE8M0 to FP32, then delegates transform_sf_into_required_layout. The wrapper does not determine extension-allocated final layout/storage/aliases. Constructor totals cannot replace final residency or derive normal KV blocks/available-KV.", ""])
    if formal_source_reviews:
        text.extend(["", "## Newly delivered scoped source reviews from formal reference", "",
            "The table reads original checks, scopes, and binding limits from independent reference publications, without rerunning passed items, changing snapshots, or treating dirty direct observations as clean whole-tree acceptance. Earlier pending/tool-limited statements remain historical context; this table records newly published scoped conclusions. Current HEAD, actual compiler/provider/runtime, normal capacity, and W1–W7 remain incomplete.", "",
            "| Reference receipt | Original counts | Original scope | Binding limitations |",
            "|---|---|---|---|"])
        for record in formal_source_reviews:
            receipt = record["receipt"]
            name = Path(record["reference_artifact"]).name
            if name == "target_sampling_materialization_review.json":
                delta = record["model_delta_receipt"]["receipt"]
                counts = "; ".join(f"{key}={receipt[key]}" for key in ("direct_scopes_passed", "candidate_failures"))
                counts += "; " + "; ".join(f"{key}={delta[key]}" for key in (
                    "model_phases", "added_materializations", "new_declared_INT64_hidden_projections"))
                binding = (f"implementations={receipt['implementations']}; {receipt['snapshot_qualification']}; "
                           f"direct_heads={receipt['direct_heads']}; direct_dirty={receipt['direct_dirty']}")
                text.append(f"| [{name}]({record['reference_artifact']}) | {counts} | {'; '.join(receipt['accepted_scope'])} | {binding}; {'; '.join(receipt['limits'])} |")
                continue
            if name in ("glm_residual_norm_review.json", "kimi_residual_norm_review.json"):
                counts = "; ".join(f"{key}={receipt[key]}" for key in (
                    "direct_scopes_passed", "model_phase_count", "model_phases",
                    "changed_norm_calls", "candidate_failures") if key in receipt)
                binding = (f"implementation={receipt['implementation']}; {receipt['snapshot_qualification']}; "
                           f"direct_heads={receipt['direct_heads']}; direct_dirty={receipt['direct_dirty']}")
                text.append(f"| [{name}]({record['reference_artifact']}) | {counts} | {receipt['remaining']['item']} | {binding}; {'; '.join(receipt['limits'])} |")
                continue
            if name == "glm_target_fused_embedding_review.json":
                counts = "; ".join(f"{key}={receipt[key]}" for key in (
                    "direct_scopes_passed", "model_phases", "same_case_removed_calls", "candidate_failures"))
                binding = f"implementations={receipt['implementation_snapshots']}; direct_heads={receipt['direct_heads']}"
                text.append(f"| [{name}]({record['reference_artifact']}) | {counts} | {receipt['resolved_pending_embedding']} | {binding}; {'; '.join(receipt['limits'])} |")
                continue
            if name in _MHC_SM90_REVIEW_FILENAMES:
                if name in ("mhc_prenorm_sm90_review.json", "mhc_prenorm_sm90_tma_review.json",
                            "mhc_prenorm_sm90_zero_stage_review.json"):
                    scope = "; ".join(receipt["accepted_scope"])
                    binding = f"candidate_start={receipt['candidate_start']}; candidate_end={receipt['candidate_end']}"
                    limits = receipt["limits"]
                elif name == "mhc_prenorm_sm90_host_binary_review.json":
                    scope = receipt["source_basis"]
                    binding = f"publication_snapshot={receipt['publication_snapshot']}"
                    limits = receipt["not_accepted"]
                elif name == "mhc_prenorm_sm90_descriptor_review.json":
                    scope = receipt["accepted_basis"]
                    binding = f"observed_publication_head={receipt['observed_publication_head']}"
                    limits = receipt["limits"]
                elif name == "operator_sm90_formal_publication_review.json":
                    scope = receipt["accepted_scope"]
                    binding = f"publication_snapshot={receipt['publication_snapshot']['commit']}"
                    limits = receipt["limits"]
                else:
                    if name == "mhc_prenorm_sm90_model_delta_review.json":
                        scope = receipt["coverage_change"]
                    elif name == "mhc_prenorm_sm90_tma_model_delta_review.json":
                        scope = "; ".join(receipt["limits"])
                    else:
                        scope = receipt["scope"]
                    binding = f"original candidate_snapshot={receipt['candidate_snapshot']['commit']}"
                    limits = receipt["limits"]
                counts = "; ".join(f"{key}={receipt[key]}" for key in
                    ("own_direct_scopes_passed", "direct_scopes_passed", "candidate_failures",
                     "conditional_metadata_checks", "metadata_checks", "nonempty_partition_checks",
                     "receipts_checked", "lost_fields_or_relabels", "new_counter_categories") if key in receipt)
                path = record["reference_artifact"]
                text.append(f"| [{name}]({path}) | {counts} | {scope} | {binding}; {'; '.join(limits)} |")
                continue
            scope = (receipt["accepted_scope"] if "accepted_scope" in receipt else
                     receipt["scope"] if "scope" in receipt else receipt["source_scope"])
            counts = "; ".join(f"{key}={receipt[key]}" for key in
                               ("checks", "failed", "immutable_comparison_checks", "immutable_comparison_failed",
                                "artifact_checks", "artifact_failed", "model_layout_mode_records",
                                "phase_records", "recount_mismatches") if key in receipt)
            if "binding_limit" in receipt:
                binding = receipt["binding_limit"]
            elif "candidate" in receipt:
                binding = f"candidate={receipt['candidate']}; original artifact scope only"
            elif "snapshot" in receipt:
                binding = f"snapshot={receipt['snapshot']['commit']}; parameter/source scope only"
            elif "before_head" in receipt:
                binding = "; ".join(f"{key}={receipt[key]}" for key in
                                    ("before_head", "before_dirty", "after_head", "after_dirty")
                                    if key in receipt)
            elif "device_subset_compile" in receipt:
                compiled = receipt["device_subset_compile"]
                binding = (f"full module exit={receipt['full_module_compile']['exit_code']}; "
                           f"device subset exit={compiled['exit_code']}; "
                           f"GPU run={compiled['GPU_module_loaded_or_run']}; "
                           f"actual runtime build bound={compiled['actual_vLLM_runtime_build_bound']}")
            elif "production_metadata_change" in receipt:
                metadata = receipt["production_metadata_change"]
                binding = f"source metadata candidate={metadata['candidate']}; {metadata['scope']}; original compile scope only"
            elif "reused_worker_host_observation" in receipt:
                observation = receipt["reused_worker_host_observation"]
                binding = (f"worker observation reused; independent load={receipt['independently_loaded_library']}; "
                           f"actual runtime bound={observation['actual_vLLM_runtime_build_bound']}")
            elif "source_reference_qualification" in receipt:
                binding = receipt["source_reference_qualification"]
            elif "candidate_observation" in receipt:
                binding = f"candidate={receipt['candidate_observation']}; metadata/receipt scope only"
            else:
                binding = ", ".join(sorted({item["snapshot"]["commit"] for item in receipt["records"]}))
            path = record["reference_artifact"]
            text.append(f"| [{Path(path).name}]({path}) | {counts} | {scope} | {binding} |")
        for record in formal_source_reviews:
            receipt = record["receipt"]
            if "device_subset_compile" in receipt:
                text.extend(["", "Offline compiled resources and static instruction sites (retaining the local toolchain/template conditions above):", "",
                    "| Device instance | Named score arrays B | Compiler static shared B | EX2 / RCP / TANH sites |",
                    "|---|---:|---:|---|"])
                for kernel in receipt["kernels"]:
                    text.append(f"| {kernel['demangled']} | {kernel['named_score_array_bytes_in_source']} | "
                                f"{kernel['compiler_reported_static_shared_bytes']} | "
                                f"{kernel['MUFU_EX2_sites']} / {kernel['MUFU_RCP_sites']} / {kernel['MUFU_TANH_sites']} |")
                text.extend(["", "Sites are not dynamic execution counts. Named score arrays exclude compiler/cooperative-groups scratch and cannot stand for total static shared memory or actual-provider occupancy. Source logical work separates named_score_array_bytes_per_cta from unknown static_shared_bytes_per_cta=None. Routing remains unpriced with launch=None; local cubins establish no H200 service rates, provider, or latency acceptance.", ""])
        text.extend(["", "Bounded review of major unpriced families: GLM TP8 native has 616 grouped routes among 808 calls; Kimi has 480 routes among 496; Flash modular12 comprises 4 each embedding/prenorm/target-selection. grouped_topk_kernels.cu contains tanhf and packed64 comparison/reduction, compiled through _moe_C_stable_libtorch and VLLM_GPU_FLAGS, with common NVCC flags from PyTorch. CMake use_fast_math settings belong to named Kimi kernels, so they cannot bind grouped tanhf to one SFU instruction or assign INT32 ALU throughput to uint64 comparison/packing/reduction. Missing actual compiler/provider evidence keeps routing unpriced; full/domain production duration or errors never tune parameters.", ""])
    if graph_records:
        text.extend(["", "## V4 source graphs (without lowering or pricing)", "",
            "New graphs retain module_path/source_symbol/shape/dtype/attributes and full edges for every phase/TP rank, supplying new-snapshot source-comparison inputs to formal reference only. Node counts are not lowered kernel-call counts. No latency is computed; older Roofline artifacts retain their labels.", "",
            "| Artifact / snapshot | Phase | Workload nodes | Edges |",
            "|---|---|---:|---:|"])
        for record in graph_records:
            for phase in record["phases"]:
                text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {phase['phase']} | {phase['nodes']} | {phase['edges']} |")
        text.extend(["", "Independent scoped review accepted source graphs 66ad4c85: three indexer-source fixes cover Flash336/Pro960 matching nodes without stale MLA/LightningIndexer or self_attn/mlp labels. All-phase/rank mHC/FFN/attention/head provenance passed. Sparse QK retains total dimension512 including RoPE across Flash860/Pro2440 nodes. Passed pricing/launch/GPU tests were not rerun. Formal reference Kimi TP/EP2695/2695 binds 66d9ab0f; V4 TP/memo3772/3772 binds f24b7c56. Math/ownership/source scopes do not accept actual provider, loaded weights, runtime, allocator, or full alignment.", ""])
    assessment_path = output / "unpriced_source_work_assessment.json"
    if assessment_path.is_file():
        assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
        if (assessment["actual_runtime_bound"] is not False
                or assessment["production_pricing_changed"] is not False):
            raise ValueError("bounded source assessment cannot assert runtime or pricing acceptance")
        text.extend(["", "## One bounded source assessment of remaining unpriced work", "",
            f"[Full fields and source locations]({assessment_path.resolve()}). This is worker analytical judgment, not new independent acceptance or latency prediction. No passed tests, compilation, environment checks, or asset probes were repeated.", "",
            "| Scope | Source-determined facts | Unresolved point values/dynamic fields | Decision this round |",
            "|---|---|---|---|"])
        for family in assessment["families"]:
            text.append(f"| {family['family']} | {family['source_determined']} | {family['not_derivable']} | {family['decision']} |")
        text.extend(["", assessment["request_default_qualification"], "", assessment["pp_ruling"], ""])
    local_inputs_path = output / "bounded_local_inputs.json"
    if local_inputs_path.is_file():
        local_inputs = json.loads(local_inputs_path.read_text(encoding="utf-8"))
        text.extend(["", "## Historical bounded local-input checks (not current asset status)", "",
            f"Results and reproduction commands: [bounded_local_inputs.json]({local_inputs_path.resolve()}). Checks covered declared upstream/models, HF cache, and known Conda environments only; they do not establish machine-wide asset absence.", "",
            "| Search root | Present | config.json | Draft JSON |", "|---|---|---|---|"])
        for search in local_inputs["searches"]:
            text.append(f"| {search['root']} | {search['exists']} | {search['config_paths']} | {search['draft_paths']} |")
        text.extend(["", f"Offline environment: {local_inputs['offline_environment']}; package versions/paths: {local_inputs['offline_versions']}.",
            f"Actual NamespaceTool import result: {local_inputs['NamespaceTool_import']}.",
            "Original inspection records are retained. Exact configs for all seven targets are now available; K2.7-Code is outside scope. Actual provider, normal capacity, and native dispatch require separate acceptance; historical missing-config conclusions no longer apply.", ""])
        if "torch_sdk_environment_records" in local_inputs:
            text.extend(["Actual Torch SDKs in known Conda environments and base:", "",
                "| SDK metadata | Version fields | ScalarType.h | CUDA stream / BLAS declarations |",
                "|---|---|---|---|"])
            for sdk in local_inputs["torch_sdk_environment_records"]:
                text.append(f"| {sdk['metadata']} | {sdk['fields']} | {sdk['ScalarType_header_present']} | "
                            f"{sdk['aoti_cuda_stream_declaration']} / {sdk['cuda_blas_handle_declaration']} |")
            text.extend(["", local_inputs["CUDA_API_scope_correction"], ""])
    module_result_path = output / "grouped_topk_cuda122_torch211_full_module" / "result.json"
    if module_result_path.is_file():
        module_result = json.loads(module_result_path.read_text(encoding="utf-8"))
        text.extend(["", "## Full-module compilation check with newly located local SDK", "",
            f"[Commands and results]({module_result_path.resolve()}); [compile log]({(module_result_path.parent / 'compile.log').resolve()}).",
            f"Torch header source: {module_result['torch_header_source']}; source_transform={module_result['source_transform']}; exit_code={module_result['exit_code']}; GPU run={module_result['GPU_module_loaded_or_run']}; actual runtime build bound={module_result['actual_vLLM_runtime_build_bound']}.",
            "The older offline command omitted USE_CUDA, hiding two CUDA API declarations guarded by that macro; this does not prove Torch2.11 lacks them. Actual SDK headers provide both. vLLM CMakeLists.txt defines USE_CUDA and Torch2.11 stable ABI for _moe_C_stable_libtorch. Python Torch2.13 requirements and C++ Torch2.11 ABI targets are separate. Old failure logs remain evidence of missing command flags, rather than an SDK missing-API blocker.", ""])
    for directory in ("grouped_topk_cuda122_torch211_use_cuda_full_module",
                      "grouped_topk_cuda122_torch211_cmake_definitions",
                      "grouped_topk_cuda122_torch211_cmake_flags"):
        corrected_result_path = output / directory / "result.json"
        if corrected_result_path.is_file():
            corrected = json.loads(corrected_result_path.read_text(encoding="utf-8"))
            text.extend(["", f"Corrected full CUDA translation unit: [commands and results]({corrected_result_path.resolve()}); "
                f"[Compile log]({(corrected_result_path.parent / 'compile.log').resolve()}).",
                f"exit_code={corrected['exit_code']}; cubin_present={corrected['cubin_present']}; "
                f"source_transform={corrected['source_transform']}; GPU run={corrected['GPU_module_loaded_or_run']}; "
                f"actual runtime build bound={corrected['actual_vLLM_runtime_build_bound']}.",
                "Offline compilation evidence is limited to local CUDA12.2/SM90, actual Torch2.11 headers, and explicit flags. No vLLM extension was linked/imported and no H200 provider selected/run. It establishes no dynamic count, rate, traffic, or duration acceptance; formal scopes of the four older device instances do not expand.", ""])
            if "CUDA12_CMake_flag_filter_source" in corrected:
                text.extend([corrected["CUDA12_CMake_flag_filter_source"],
                    "For CUDA>=12, vLLM explicitly removes four half/BF16-disable macros from Torch COMMON_NVCC_FLAGS. Earlier failure with unfiltered flags was a command-control error, not missing source/SDK. Flags are neither guessed from failures nor latency; target definitions/filtering come from CMake source.", ""])
    complete_instances_path = output / "grouped_topk_cuda122_torch211_cmake_flags" / "selected_instances.json"
    if complete_instances_path.is_file():
        complete_instances = json.loads(complete_instances_path.read_text(encoding="utf-8"))
        text.extend(["", f"Four matching full-translation-unit instances: [static output and reproduction commands]({complete_instances_path.resolve()}).",
            "| Device instance | Named score arrays B | Full-TU shared B | EX2 / RCP / TANH sites |",
            "|---|---:|---:|---|"])
        for kernel in complete_instances["records"]:
            text.append(f"| {kernel['demangled']} | {kernel['named_score_array_bytes_in_source']} | "
                        f"{kernel['resource_usage']['SHARED']} | {kernel['MUFU_EX2_static_sites']} / "
                        f"{kernel['MUFU_RCP_static_sites']} / {kernel['MUFU_TANH_static_sites']} |")
        accepted_tu = any(Path(record["reference_artifact"]).name == "grouped_topk_full_translation_unit_review.json"
                          and record["receipt"]["failed"] == 0 and record["receipt"]["checks"] > 0
                          for record in formal_source_reviews)
        qualification = ("Formal reference accepted full-TU static output within scope; original scope and four checks are in the receipt table. Actual provider/dynamic service and full models remain unaccepted."
                         if accepted_tu else "New full-TU static output has not received independent acceptance.")
        text.extend(["", "Two warp instances in the local full TU report shared=1024B versus 0B in the older isolated subset; block instances remain GLM3072B/Kimi4096B. Compiled resources must retain TU/flag conditions; subset values cannot transfer to complete modules or actual providers. "
                     + qualification + " Source logical-work total shared remains None; all routing service remains unpriced.", ""])
    host_probe_path = output / "grouped_topk_torch211_host_registration_probe" / "host_load_result.json"
    if host_probe_path.is_file():
        host_probe = json.loads(host_probe_path.read_text(encoding="utf-8"))
        text.extend(["", "## Torch2.11 host-linkage and registration boundary", "",
            f"[Link results and scope]({Path(host_probe['link_result']).resolve()}); "
            f"[Host-load observations and reproduction commands]({host_probe_path.resolve()}). load_exit_code={host_probe['load_exit_code']}.",
            "Original grouped_topk CUDA TU/host wrappers are unchanged. Only vendor grouped_topk schema/TORCH_BOX registration statements are selected into an isolated diagnostic namespace. This is not a complete vLLM extension and does not replace OpenAI SDK or actual-provider evidence.", ""])
        if host_probe["load_exit_code"] == 0:
            observation = host_probe["observation"]
            text.extend([f"Actual Torch: {observation['torch_version']} / {observation['torch_source']}.",
                f"schema: `{observation['schema']}`.",
                f"CUDA dispatch registered={observation['CUDA_dispatch_registration']}; CPU dispatch registered={observation['CPU_dispatch_registration']}; "
                f"CUDA_VISIBLE_DEVICES='{observation['CUDA_VISIBLE_DEVICES']}'; PyTorch CUDA initialized before/after="
                f"{observation['pytorch_cuda_initialized_before']}/{observation['pytorch_cuda_initialized_after']}.",
                f"tensor/operator invoked={observation['tensor_created_or_operator_invoked']}; GPU kernel invoked={observation['GPU_kernel_invoked']}; "
                f"actual vLLM runtime bound={observation['actual_vLLM_runtime_build_bound']}.",
                "Confirms single-operator host ABI linkage/registration with the actual local SDK only. No CUDA kernel is called and no numerical/traffic/demand/rate/duration evidence is produced. W1–W7/full-model acceptance do not expand.", ""])
            host_review = [record for record in formal_source_reviews if
                           Path(record["reference_artifact"]).name == "grouped_topk_host_registration_review.json"]
            if host_review:
                receipt = host_review[0]["receipt"]
                text.extend([f"Formal reference checks/failures: {receipt['checks']}/{receipt['failed']}; "
                    f"independently reloaded library={receipt['independently_loaded_library']}. Original scope: {receipt['accepted_scope']}.",
                    "Checks cover source selection, ELF/dependencies/symbols, and existing raw observations. Their denominator is neither GPU executions nor model cases; no full-vLLM/provider/runtime/latency acceptance is inherited.", ""])
        else:
            text.extend([f"Host loading failed: {host_probe['load_error']}. This failure remains SDK-boundary evidence.", ""])
    finite_precision_reviews = [record for record in formal_source_reviews if
        Path(record["reference_artifact"]).name == "grouped_topk_finite_precision_reference.json"]
    if finite_precision_reviews:
        receipt = finite_precision_reviews[0]["receipt"]
        text.extend(["", "## Grouped-topk finite-precision boundary (source counterexample)", "",
            f"Formal reference: {finite_precision_reviews[0]['reference_artifact']}. Original scope: {receipt['accepted_scope']}.",
            f"Acceptance qualification: {receipt['source_reference_qualification']}.",
            "The source block reads retained FP32 unbiased scores; the warp subtracts bias from rounded biased scores. A nonempty selection mask prevents uniform fallback even if restored sum is zero. The table reads reviewed receipt arithmetic directly without recomputation; it does not establish an actual checkpoint/GPU defect.", "",
            "| E / K | Logits all zero | Bias | Selection FP32 | Block unbiased | Warp recovered | Block weight each | Warp weight each | Same value |",
            "|---|---|---|---|---|---|---|---|---|"])
        for example in receipt["examples"]:
            text.append(f"| {example['experts']} / {example['topk']} | {example['logits_all_zero']} | {example['correction_bias_all']} | {example['selection_score_fp32']} | {example['source_block_unbiased']} | {example['source_warp_recovered_unbiased']} | {example['block_weight_each_fp32']} | {example['warp_weight_each_fp32']} | {example['same_value']} |")
        text.append("")
        for example in receipt["examples"]:
            text.extend([f"Condition: {example['condition']}. Dispatch qualification: {example['dispatch_transition']}.", ""])
        for finding in receipt["findings"]:
            text.extend([f"Reference finding: {finding['item']}.Required action: {finding['required_action']}.", ""])
        text.extend(["GLM/Kimi conditional source contracts explicitly retain different restoration, selection validity, uniform fallback, and epsilon-normalization semantics. Actual checkpoint bias domains, selected providers, and native values remain unbound. CPU Torch branches, offline compilation, and host registration cannot replace them. Operation/byte counts, launch assignments, HWSpec rates, demand/time remain unchanged; routing stays unpriced and W1–W7 fail.", ""])
        text.extend([f"Original limitation: {limit}" for limit in receipt["limits"]])
    publication_reviews = [record for record in formal_source_reviews if
        Path(record["reference_artifact"]).name == "operator_finite_precision_publication_review.json"]
    if publication_reviews:
        receipt = publication_reviews[0]["receipt"]
        text.extend(["", "", "## Scoped formal-candidate review of finite-precision qualification", "",
            f"Formal reference: [publication review]({publication_reviews[0]['reference_artifact']}); "
            f"original candidate={receipt['candidate_observation']}, metadata branches={len(receipt['metadata_checks'])}, "
            f"original source scenarios retained={receipt['source_scenarios_preserved']}, failed={receipt['failed']}.",
            f"Original accepted scope: {receipt['accepted_scope']}.",
            "Confirms source numerical qualifications and receipt publication only. Metadata branch/scenario counts are not new hardware/model/numerical acceptance denominators. W6/full target remain unaccepted; actual provider/runtime/pricing acceptance does not automatically transfer to later HEAD.", ""])
    if existing_paths:
        path_records = list(existing_paths.values())
        coverage_path = output / "existing_model_roofline_paths.json"
        coverage_path.write_text(json.dumps({
            "records": path_records, "goal_complete": False,
            "selection": "last explicitly supplied artifact per model/layout/mode; original source scenario and snapshot only",
            "actual_runtime_or_normal_capacity_accepted": False}, indent=2) + "\n", encoding="utf-8")
        text.extend(["", "## Minimal-path assessment for models with existing configs", "",
            f"[Per-phase unpriced families and first blocking call]({coverage_path.resolve()}). Existing static artifacts are reused without rerunning passed cases. Rows retain original snapshots, config revisions, and source scenarios. Unpriced counts are not latency errors and cannot tune parameters.",
            "| Artifact / original snapshot | Model / TP / PP / MTP | Unpriced calls (prefill/decode1/decode2/mixed) | First-phase gap |",
            "|---|---|---|---|"])
        for record in path_records:
            deployment = record["deployment"]
            phases = record["phases"]
            counts = "/".join(str(phase["unpriced_calls"]) for phase in phases)
            artifact = Path(record["artifact"])
            text.append(f"| [{artifact.name}]({artifact.resolve()}) / {record['snapshot']['commit'][:8]} | {record['model_id']} / {deployment['tensor_parallel_size']} / {deployment['pipeline_parallel_size']} / {record['mtp']} | {counts} | {phases[0]['unpriced_families']} |")
        text.extend(["", "Kimi-K2.6 and GLM-5.2 have exact configs and implemented source paths. Local blockers are grouped routing plus embedding/target indexing and native state service, rather than missing configs for three other models. Independent routed-weight bounds exclude GLM BF16 TP8/PP1 and TP1/PP1 normal capacity. PP2 is not excluded, but remains a source projection with unaccepted TP-partial transport.",
            "V4 Flash is the smaller conditional source case, but config revision remains existing-h200-campaign-config-revision-unverified rather than exact checkpoint identity. First-layer DeepGEMM prenorm and embedding/target indexing lack source-to-actual-launch/provider binding; reducing tokens cannot remove required producers. V4 Pro retains these gaps plus post-load residency/DSpark gaps.",
            "Source-supported mathematical FLOPs, typed bytes, and conditional launches remain valid, but no row establishes complete real Roofline/normal Serving closure. Real v0.30 Scheduler also retains the documented NamespaceTool SDK gap. Config, source scenario, compiler/host registration, actual provider, and normal-KV acceptance remain separate.", ""])
        path_publications = [record for record in formal_source_reviews if
            Path(record["reference_artifact"]).name == "operator_model_roofline_paths_review.json"]
        if path_publications:
            receipt = path_publications[0]["receipt"]
            text.extend([f"Formal independent publication-statistics review: [original receipt]({path_publications[0]['reference_artifact']}); "
                f"layout/mode records={receipt['model_layout_mode_records']}, phase records={receipt['phase_records']}, "
                f"recount mismatches={receipt['recount_mismatches']}.",
                f"original scope: {receipt['accepted_scope']}. Every record retains snapshot and original source_projection_only/partial flags. No operator pricing, actual runtime, normal KV, or full-Goal acceptance is added.", ""])
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
