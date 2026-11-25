"""NVIDIA H200 kernel-dataset runner for weight-only-quantized (W4A16) MoE.

Measures, under the canonical ``nsys_nvtx_kernel_median`` protocol
(``nsys_campaign_runner.py``), the routed-expert computation that vLLM 0.24.0
executes on SM90 for two 4-bit weight-only checkpoint formats:

* ``int4_g32`` -- compressed-tensors ``pack-quantized`` symmetric INT4, group
  32 (Kimi-K2.6 routed experts).  vLLM 0.24 selects
  ``CompressedTensorsWNA16MarlinMoEMethod`` (``check_moe_marlin_supports_layer``
  holds) and the WNA16 oracle picks ``WNA16MoEBackend.MARLIN``
  (``FLASHINFER_TRTLLM`` needs SM100), i.e. ``MarlinExperts`` with scalar type
  ``uint4b8``.
* ``mxfp4`` -- MXFP4 experts (E2M1 values, UE8M0 scale per 32 along K).  On
  SM90 both DeepSeek-V4 (``DeepseekV4FP8Config`` -> ``Mxfp4MoEMethod`` ->
  ``select_deepseek_v4_mxfp4_moe_backend``: TRTLLM-MXFP8 and DeepGEMM-MXFP4
  need SM100, so ``MARLIN``) and compressed-tensors ``mxfp4-pack-quantized``
  (``CompressedTensorsW4A4Mxfp4MoEMethod``: ``CutlassExpertsMxfp4`` is not
  supported on capability 90, so ``MarlinExperts``) run ``MarlinExperts`` with
  scalar type ``float4_e2m1f`` and BF16 activations.

Weights are synthetic (seed 42) and converted OUTSIDE the NVTX range with the
exact vLLM post-load helpers (``convert_to_wna16_moe_kernel_format`` with
``WNA16MoEBackend.MARLIN`` -> ``gptq_marlin_moe_repack`` /
``marlin_moe_permute_scales``; ``convert_weight_to_mxfp4_moe_kernel_format``
with ``Mxfp4MoeBackend.MARLIN`` -> ``prepare_moe_mxfp4_layer_for_marlin``).
Inside the NVTX range there is exactly one ``fused_marlin_moe`` call with the
arguments ``MarlinExperts.apply`` passes (preallocated modular-kernel
workspaces swapped as in ``MarlinExperts.workspace_shapes``, ``ops.moe_sum``,
``apply_moe_activation``, no Marlin workspace argument so the per-call
``marlin_make_workspace_new`` zero-fill is included as in deployment).
Routing (top-k selection) and the modular-kernel prepare/finalize are outside
the range, matching ``fused_moe_runner.py`` which times ``fused_experts``.

Expert parallelism: ``--ep_size``/``--ep_rank`` give rank ``ep_rank`` the
contiguous local experts ``[ep_rank*E/ep, (ep_rank+1)*E/ep)`` (vLLM linear
placement) and pass the global->local ``expert_map``; every rank routes all
``num_tokens`` tokens over the global experts (vLLM EP with
all-gather/reduce-scatter dispatch).

vLLM 0.30.0 (``installed_vllm_stack() == "vllm030"``), explicit branch with
the chosen path recorded as ``wq_moe_routing.call_path``:

* Selection is unchanged on SM90: compressed-tensors INT4 g32 ->
  ``CompressedTensorsWNA16MoEMethod`` -> ``select_wna16_moe_backend`` ->
  MARLIN (FLASHINFER_TRTLLM needs SM100, RDNA3 gfx1100); DeepSeek-V4
  ``Mxfp4MoEMethod`` -> ``select_deepseek_v4_mxfp4_moe_backend`` -> MARLIN
  (TRTLLM-MXFP8 / DeepGEMM-MXFP4 need SM100); compressed-tensors
  ``mxfp4-pack-quantized`` (Kimi-K3) -> ``CompressedTensorsW4A4Mxfp4MoEMethod``
  -> ``MarlinExperts`` (``CutlassExpertsMxfp4`` needs SM100..SM11x).  The
  runner re-checks the higher-priority experts classes'
  ``_supports_current_device()`` and fails if any is True.
* ``convert_to_wna16_moe_kernel_format`` lost the g_idx arguments (no act
  order); ``fused_marlin_moe`` lost ``g_idx*/sort_indices*/is_k_full/
  clamp_limit/gemm1_alpha/gemm1_beta`` and takes ``activation_config``
  (``ApplyMoEActivationConfig``: clamp limit, SiTU betas), the activation
  callback ``activation(act, out, in, *, topk_ids, expert_map)`` of
  ``FusedMoEExperts.activation`` and ``moe_sum(in, out, topk_ids,
  expert_map)`` of ``MarlinExperts.moe_sum``; the Marlin M-block heuristic
  uses ``M = ceil(M * E_local / E_global)``.
* ``--activation situ`` (0.30 only; 0.24 has no ``MoEActivation.SITU``):
  Kimi-K3's own expert activation ``torch.ops._C.situ_and_mul(out, in,
  activation_situ_beta, activation_situ_linear_beta)``; its weights are
  converted with the K3 method's in-place ``prepare_moe_fp4_layer_for_marlin``.
"""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_h200_capture import (
    add_h200_capture_arguments,
    installed_vllm_stack,
    open_h200_capture_session,
)

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation, apply_moe_activation)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    fused_marlin_moe)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend, convert_to_wna16_moe_kernel_format)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend, convert_weight_to_mxfp4_moe_kernel_format)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_layer, get_marlin_input_dtype,
    marlin_moe_intermediate_size)
from vllm.scalar_type import scalar_types

VLLM_STACK = installed_vllm_stack()
if VLLM_STACK == "vllm030":
    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig)
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        check_moe_marlin_supports_config, marlin_moe_padded_intermediate)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_fp4_layer_for_marlin)
elif VLLM_STACK != "vllm024":
    raise AssertionError(f"unhandled vLLM stack {VLLM_STACK}")

from dataset_pipeline.kernel_dataset.nvidia_a800_event_capture import (
    build_nvidia_a800_moe_topk_ids,
)


INPUT_SEED = 42
MXFP4_BLOCK = 32
# UE8M0 exponents 2^-9 .. 2^0; 255 is the UE8M0 NaN encoding.
MXFP4_SCALE_EXPONENT_RANGE = (118, 128)
WEIGHT_FORMATS = ("int4_g32", "mxfp4")
ACTIVATIONS = ("silu", "situ")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"H200 WQ-MoE {name} must be a positive integer")


def _local_expert_map(num_experts, ep_size, ep_rank, device):
    if num_experts % ep_size != 0:
        raise ValueError("H200 WQ-MoE num_experts must divide by ep_size")
    if not 0 <= ep_rank < ep_size:
        raise ValueError("H200 WQ-MoE ep_rank must be in [0, ep_size)")
    local = num_experts // ep_size
    first = ep_rank * local
    if ep_size == 1:
        return local, first, None
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32,
                            device=device)
    expert_map[first:first + local] = torch.arange(
        local, dtype=torch.int32, device=device)
    return local, first, expert_map


def _routing(num_tokens, num_experts, top_k, expected_token_counts, device):
    """Seeded softmax/top-k routing (fused_moe_runner.py construction); an
    explicit tok_per_expert histogram replaces the ids deterministically."""
    router_logits = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1)
    if expected_token_counts is not None:
        topk_ids = torch.tensor(
            build_nvidia_a800_moe_topk_ids(
                token_counts=expected_token_counts, num_tokens=num_tokens,
                num_experts=num_experts, top_k=top_k,
                context="H200 WQ-MoE canonical routing"),
            dtype=torch.int32, device=device)
    # MarlinExperts requires float32 router weights.
    return topk_weights.to(torch.float32).contiguous(), \
        topk_ids.to(torch.int32).contiguous()


def _int4_g32_weights(local_experts, hidden, inter, group_size, device):
    """compressed-tensors pack-quantized symmetric INT4 in the Marlin layout
    ``create_weights`` allocates (is_transposed: (E, K/8, N) int32 and
    (E, K/group, N) BF16 scales), then vLLM's Marlin post-load conversion."""
    pack = 8
    info = torch.iinfo(torch.int32)
    w13 = torch.randint(info.min, info.max, (local_experts, hidden // pack,
                        2 * inter), dtype=torch.int32, device=device)
    w2 = torch.randint(info.min, info.max, (local_experts, inter // pack,
                       hidden), dtype=torch.int32, device=device)
    w13_scale = (torch.rand(local_experts, hidden // group_size, 2 * inter,
                            device=device) * 1e-2 + 1e-3).to(torch.bfloat16)
    w2_scale = (torch.rand(local_experts, inter // group_size, hidden,
                           device=device) * 1e-2 + 1e-3).to(torch.bfloat16)
    w13_g_idx = torch.empty(local_experts, hidden, dtype=torch.int32,
                            device=device)
    w2_g_idx = torch.empty(local_experts, inter, dtype=torch.int32,
                           device=device)
    layer = types.SimpleNamespace(
        intermediate_size_per_partition=inter,
        num_groups_w13=hidden // group_size,
        num_groups_w2=inter // group_size)
    quant_args = QuantizationArgs(
        num_bits=4, type="int", symmetric=True, group_size=group_size,
        strategy="group", actorder=None, dynamic=False)
    if VLLM_STACK == "vllm024":
        (w13, w2, w13_scale, w2_scale, w13_g_idx, w2_g_idx, w13_sort, w2_sort,
         w13_zp, w2_zp, w13_gscale, w2_gscale, _, _) = \
            convert_to_wna16_moe_kernel_format(
                backend=WNA16MoEBackend.MARLIN, layer=layer,
                quant_config=quant_args, input_dtype=None,
                w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale,
                w13_g_idx=w13_g_idx, w2_g_idx=w2_g_idx)
    else:
        # vLLM 0.30: no act-order g_idx / sort indices anywhere.
        del w13_g_idx, w2_g_idx
        (w13, w2, w13_scale, w2_scale, w13_zp, w2_zp, w13_gscale, w2_gscale,
         w13_bias, w2_bias) = convert_to_wna16_moe_kernel_format(
            backend=WNA16MoEBackend.MARLIN, layer=layer,
            quant_config=quant_args, input_dtype=None,
            w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale)
        if w13_bias is not None or w2_bias is not None:
            raise RuntimeError("H200 WQ-MoE INT4 conversion produced a bias")
        w13_g_idx = w2_g_idx = w13_sort = w2_sort = None
    for value, name in ((w13_zp, "w13_zp"), (w2_zp, "w2_zp"),
                        (w13_gscale, "w13_input_global_scale"),
                        (w2_gscale, "w2_input_global_scale")):
        if value is not None:
            raise RuntimeError(f"H200 WQ-MoE INT4 conversion produced {name}")
    return {
        "w1": w13, "w2": w2, "w1_scale": w13_scale, "w2_scale": w2_scale,
        "g_idx1": w13_g_idx, "g_idx2": w2_g_idx,
        "sort_indices1": w13_sort, "sort_indices2": w2_sort,
        "quant_type_id": scalar_types.uint4b8.id,
        "quant_type": "uint4b8",
    }


def _mxfp4_weights(local_experts, hidden, inter, device,
                   k3_compressed_tensors=False):
    """Checkpoint layout of ``Mxfp4MoEMethod.create_weights`` (uint8 packed
    E2M1 pairs along K, uint8 UE8M0 scales per 32 along K), then vLLM's
    Marlin post-load conversion (``k3_compressed_tensors``: the in-place
    ``prepare_moe_fp4_layer_for_marlin`` of
    ``CompressedTensorsW4A4Mxfp4MoEMethod``, vLLM 0.30 only)."""
    w13 = torch.randint(0, 256, (local_experts, 2 * inter, hidden // 2),
                        dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (local_experts, hidden, inter // 2),
                       dtype=torch.uint8, device=device)
    low, high = MXFP4_SCALE_EXPONENT_RANGE
    w13_scale = torch.randint(low, high, (local_experts, 2 * inter,
                              hidden // MXFP4_BLOCK), dtype=torch.uint8,
                              device=device)
    w2_scale = torch.randint(low, high, (local_experts, hidden,
                             inter // MXFP4_BLOCK), dtype=torch.uint8,
                             device=device)
    if k3_compressed_tensors:
        if VLLM_STACK != "vllm030":
            raise RuntimeError("K3 compressed-tensors MXFP4 path is vLLM 0.30")
        layer = types.SimpleNamespace(
            params_dtype=torch.bfloat16,
            moe_config=types.SimpleNamespace(
                num_local_experts=local_experts, hidden_dim=hidden,
                intermediate_size_per_partition=inter),
            w13_weight=w13, w2_weight=w2, w13_weight_scale=w13_scale,
            w2_weight_scale=w2_scale)
        prepare_moe_fp4_layer_for_marlin(layer)
        w13, w2 = layer.w13_weight.data, layer.w2_weight.data
        w13_scale = layer.w13_weight_scale.data
        w2_scale = layer.w2_weight_scale.data
    else:
        layer = types.SimpleNamespace(params_dtype=torch.bfloat16)
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = \
            convert_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=Mxfp4MoeBackend.MARLIN, layer=layer,
                w13_weight=w13, w2_weight=w2, w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale, w13_bias=None, w2_bias=None)
        if w13_bias is not None or w2_bias is not None:
            raise RuntimeError("H200 WQ-MoE MXFP4 conversion produced a bias")
    return {
        "w1": w13, "w2": w2, "w1_scale": w13_scale, "w2_scale": w2_scale,
        "g_idx1": None, "g_idx2": None,
        "sort_indices1": None, "sort_indices2": None,
        "quant_type_id": scalar_types.float4_e2m1f.id,
        "quant_type": "float4_e2m1f",
    }


def _marlin_block_size_m(num_tokens, top_k, local_experts, global_experts):
    """``fused_marlin_moe`` M-block heuristic (BF16 input).  vLLM 0.30 first
    scales M to the estimated tokens per rank, ceil(M * E_local / E_global),
    since ``MarlinExperts.apply`` passes ``global_num_experts``."""
    if VLLM_STACK == "vllm024":
        m = num_tokens
    else:
        m = math.ceil(num_tokens * local_experts / global_experts)
    for block_size_m in (8, 16, 32, 48, 64):
        if m * top_k / local_experts / block_size_m < 0.9:
            break
    return block_size_m


def _sm90_selection_check(weight_format, activation):
    """vLLM 0.30: re-check that every experts class the SM90 oracle ranks
    above Marlin is unsupported on this device (fail loud otherwise)."""
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        MarlinExperts)
    if weight_format == "int4_g32":
        from vllm.model_executor.layers.fused_moe.experts.trtllm_mxint4_moe \
            import TrtLlmMxint4ExpertsMonolithic
        above = {"TrtLlmMxint4ExpertsMonolithic":
                 TrtLlmMxint4ExpertsMonolithic}
        method = ("CompressedTensorsWNA16MoEMethod -> select_wna16_moe_backend"
                  " -> WNA16MoEBackend.MARLIN")
    elif activation == "situ":
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            CutlassExpertsMxfp4)
        above = {"CutlassExpertsMxfp4": CutlassExpertsMxfp4}
        method = ("CompressedTensorsW4A4Mxfp4MoEMethod (Kimi-K3 "
                  "mxfp4-pack-quantized) -> MarlinExperts")
    else:
        from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            DeepGemmFP4Experts)
        from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe \
            import TrtLlmMxfp4ExpertsModular, TrtLlmMxfp4ExpertsMonolithic
        above = {"TrtLlmMxfp4ExpertsMonolithic": TrtLlmMxfp4ExpertsMonolithic,
                 "TrtLlmMxfp4ExpertsModular": TrtLlmMxfp4ExpertsModular,
                 "DeepGemmFP4Experts": DeepGemmFP4Experts}
        method = ("Mxfp4MoEMethod -> select_deepseek_v4_mxfp4_moe_backend -> "
                  "Mxfp4MoeBackend.MARLIN")
    supported = {name: bool(cls._supports_current_device())
                 for name, cls in above.items()}
    if any(supported.values()):
        raise RuntimeError(f"vLLM 0.30 would not pick Marlin here: {supported}")
    if not MarlinExperts._supports_current_device():
        raise RuntimeError("MarlinExperts unsupported on this device")
    return {"quant_method_path": method, "experts_class": "MarlinExperts",
            "higher_priority_supports_current_device": supported}


def run_wq_moe(args, capture, device="cuda"):
    for name in ("num_tokens", "num_experts", "top_k", "hidden_size",
                 "intermediate_size", "group_size", "ep_size"):
        _positive_int(getattr(args, name), name)
    if args.pre_nvtx_warmup_iterations < 0:
        raise ValueError("H200 WQ-MoE pre-NVTX warmup must be non-negative")
    if get_marlin_input_dtype() is not None:
        raise RuntimeError("H200 WQ-MoE requires BF16 Marlin activations "
                           "(VLLM_MARLIN_INPUT_DTYPE unset)")
    hidden, inter = args.hidden_size, args.intermediate_size
    if args.activation == "situ":
        if VLLM_STACK != "vllm030":
            raise ValueError("SiTU MoE activation needs vLLM 0.30 "
                             "(vLLM 0.24 has no MoEActivation.SITU)")
        if args.weight_format != "mxfp4" or args.swiglu_limit is not None:
            raise ValueError("SiTU is the Kimi-K3 MXFP4 path (no swiglu limit)")
        if (args.activation_situ_beta is None
                or args.activation_situ_linear_beta is None):
            raise ValueError("SiTU needs --activation-situ-beta and "
                             "--activation-situ-linear-beta")
    elif (args.activation_situ_beta is not None
          or args.activation_situ_linear_beta is not None):
        raise ValueError("SiTU betas given for a SiLU case")
    if args.weight_format == "int4_g32":
        if args.group_size != 32:
            raise ValueError("H200 WQ-MoE int4_g32 requires --group_size 32")
        if args.swiglu_limit is not None:
            raise ValueError("H200 WQ-MoE int4_g32 has no swiglu limit")
        if VLLM_STACK == "vllm024":
            # check_moe_marlin_supports_layer (vLLM 0.24 marlin_utils).
            layer_probe = types.SimpleNamespace(
                hidden_size=hidden, apply_router_weight_on_input=False,
                moe_config=types.SimpleNamespace(
                    intermediate_size_per_partition_unpadded=inter))
            if not check_moe_marlin_supports_layer(layer_probe,
                                                   args.group_size):
                raise ValueError("H200 WQ-MoE shape is not Marlin-supported; "
                                 "vLLM would use the Triton WNA16 path")
        else:
            # CompressedTensorsWNA16MoEMethod: allow_tile_padding=True; the
            # runner refuses shapes vLLM would tile-pad instead of padding.
            probe = types.SimpleNamespace(
                hidden_dim=hidden,
                intermediate_size_per_partition_unpadded=inter)
            if not check_moe_marlin_supports_config(
                    probe, args.group_size, allow_tile_padding=True):
                raise ValueError("H200 WQ-MoE shape is not Marlin-supported")
            if marlin_moe_padded_intermediate(inter, args.group_size) != inter:
                raise ValueError("vLLM 0.30 would tile-pad this intermediate")
    elif args.weight_format == "mxfp4":
        if args.group_size != MXFP4_BLOCK:
            raise ValueError("H200 WQ-MoE mxfp4 requires --group_size 32")
        # mxfp4_round_up_hidden_size_and_intermediate_size (MARLIN, CUDA):
        # the runner refuses shapes vLLM would pad instead of padding them.
        if hidden % 256 != 0 or inter % 128 != 0:
            raise ValueError("H200 WQ-MoE mxfp4 Marlin needs hidden%256==0 "
                             "and intermediate%128==0 (no padding here)")
    else:
        raise ValueError(f"unknown weight format {args.weight_format}")

    expected_token_counts = (
        None if args.expected_tok_per_expert_json is None
        else json.loads(args.expected_tok_per_expert_json))

    torch.manual_seed(INPUT_SEED)
    torch.cuda.manual_seed(INPUT_SEED)
    hidden_states = torch.randn(args.num_tokens, hidden, dtype=torch.bfloat16,
                                device=device)
    topk_weights, topk_ids = _routing(
        args.num_tokens, args.num_experts, args.top_k, expected_token_counts,
        device)
    local_experts, first_local, expert_map = _local_expert_map(
        args.num_experts, args.ep_size, args.ep_rank, device)
    if args.weight_format == "int4_g32":
        weights = _int4_g32_weights(local_experts, hidden, inter,
                                    args.group_size, device)
    else:
        weights = _mxfp4_weights(
            local_experts, hidden, inter, device,
            k3_compressed_tensors=args.activation == "situ")
    marlin_n = marlin_moe_intermediate_size(weights["w1"], weights["w2"])
    if marlin_n != inter:
        raise RuntimeError(f"Marlin intermediate {marlin_n} != {inter}")

    tok_per_expert = torch.bincount(
        topk_ids.flatten(), minlength=args.num_experts).tolist()
    if expected_token_counts is not None \
            and tok_per_expert != expected_token_counts:
        raise ValueError("H200 WQ-MoE routing disagrees with canonical Attr")
    block_size_m = _marlin_block_size_m(args.num_tokens, args.top_k,
                                        local_experts, args.num_experts)
    _, align_expert_ids, num_post_padded = moe_align_block_size(
        topk_ids, block_size_m, args.num_experts, expert_map,
        ignore_invalid_experts=True)
    padded = int(num_post_padded.item())
    active_blocks = int((align_expert_ids[:padded // block_size_m] >= 0)
                        .sum().item())
    local_counts = tok_per_expert[first_local:first_local + local_experts]
    if VLLM_STACK == "vllm024":
        if args.swiglu_limit is None:
            activation_text = "silu"
        else:
            activation_text = "silu_and_mul_with_clamp"
        call_path = {
            "vllm_stack": VLLM_STACK,
            "call": "fused_marlin_moe(g_idx/sort_indices, is_k_full, "
                    "clamp_limit, activation_func=apply_moe_activation, "
                    "moe_sum=ops.moe_sum(in, out))",
            "activation_kernel": ("torch.ops._C.silu_and_mul"
                                  if args.swiglu_limit is None else
                                  "torch.ops._C.silu_and_mul_with_clamp"),
        }
    else:
        if args.activation == "situ":
            activation_text = "situ"
            act_kernel = "torch.ops._C.situ_and_mul(beta, linear_beta)"
        elif args.swiglu_limit is None:
            activation_text = "silu"
            act_kernel = "torch.ops._C.silu_and_mul"
        else:
            activation_text = "silu_and_mul_with_clamp"
            act_kernel = ("torch.ops._C.silu_and_mul_with_clamp"
                          if expert_map is None else
                          "swiglu_limit_func -> Triton "
                          "_swiglu_limit_pad_aware_kernel (EP expert_map)")
        call_path = {
            "vllm_stack": VLLM_STACK,
            "call": "fused_marlin_moe(activation_config=ApplyMoEActivation"
                    "Config, activation_func=FusedMoEExperts.activation, "
                    "moe_sum=MarlinExperts.moe_sum(in, out, topk_ids, "
                    "expert_map))",
            "activation_kernel": act_kernel,
            "weight_conversion": (
                "prepare_moe_fp4_layer_for_marlin (in place)"
                if args.activation == "situ" else
                "convert_to_wna16_moe_kernel_format(MARLIN)"
                if args.weight_format == "int4_g32" else
                "convert_weight_to_mxfp4_moe_kernel_format(MARLIN)"),
            **_sm90_selection_check(args.weight_format, args.activation),
        }
    capture.add_launch_evidence("wq_moe_routing", {
        "weight_format": args.weight_format,
        "marlin_scalar_type": weights["quant_type"],
        "num_tokens": args.num_tokens,
        "num_experts": args.num_experts,
        "local_experts": local_experts,
        "ep_size": args.ep_size,
        "ep_rank": args.ep_rank,
        "first_local_expert": first_local,
        "top_k": args.top_k,
        "hidden_size": hidden,
        "intermediate_size_per_partition": inter,
        "group_size": args.group_size,
        "activation": activation_text,
        "swiglu_limit": args.swiglu_limit,
        "activation_situ_beta": args.activation_situ_beta,
        "activation_situ_linear_beta": args.activation_situ_linear_beta,
        "call_path": call_path,
        "routing": "seeded_softmax_topk" if expected_token_counts is None
        else "explicit_tok_per_expert",
        "tok_per_expert": tok_per_expert,
        "local_tok_per_expert": local_counts,
        "local_routed_pairs": sum(local_counts),
        "marlin_block_size_m": block_size_m,
        "num_tokens_post_padded": padded,
        "active_m_blocks": active_blocks,
    })

    num_pairs = args.num_tokens * args.top_k
    # MarlinExperts.workspace_shapes (workspaces swapped in apply()).
    workspace13 = torch.empty((num_pairs, max(inter, hidden)),
                              dtype=torch.bfloat16, device=device)
    workspace2 = torch.empty((num_pairs * max(2 * inter, hidden),),
                             dtype=torch.bfloat16, device=device)
    output = torch.empty((args.num_tokens, hidden), dtype=torch.bfloat16,
                         device=device)

    def moe_sum(moe_output, moe_out):
        ops.moe_sum(moe_output, moe_out)

    def operation_v024():
        with torch.no_grad():
            fused_marlin_moe(
                hidden_states=hidden_states, w1=weights["w1"],
                w2=weights["w2"], bias1=None, bias2=None,
                w1_scale=weights["w1_scale"], w2_scale=weights["w2_scale"],
                topk_weights=topk_weights, topk_ids=topk_ids,
                global_scale1=None, global_scale2=None,
                input_global_scale1=None, input_global_scale2=None,
                w1_zeros=None, w2_zeros=None,
                quant_type_id=weights["quant_type_id"],
                apply_router_weight_on_input=False,
                global_num_experts=args.num_experts,
                activation=MoEActivation.SILU,
                activation_func=apply_moe_activation, moe_sum=moe_sum,
                expert_map=expert_map, output=output,
                intermediate_cache13=workspace2,
                intermediate_cache2=workspace13,
                g_idx1=weights["g_idx1"], g_idx2=weights["g_idx2"],
                sort_indices1=weights["sort_indices1"],
                sort_indices2=weights["sort_indices2"],
                is_k_full=True, input_dtype=None,
                clamp_limit=args.swiglu_limit, gemm1_alpha=1.0,
                gemm1_beta=0.0)

    if VLLM_STACK == "vllm024":
        operation = operation_v024
    else:
        # ApplyMoEActivationConfig.from_configs: clamp from swiglu_limit,
        # alpha 1 / beta 0, SiTU betas from the model config.
        moe_activation = (MoEActivation.SITU if args.activation == "situ"
                          else MoEActivation.SILU)
        activation_config = ApplyMoEActivationConfig(
            clamp_limit=args.swiglu_limit, alpha=1.0, beta=0.0,
            activation_situ_beta=args.activation_situ_beta,
            activation_situ_linear_beta=args.activation_situ_linear_beta)

        def activation_func(activation, act_out, act_in, *, topk_ids=None,
                            expert_map=None):
            # FusedMoEExperts.activation (valid_token_counts unset).
            apply_moe_activation(activation, act_out, act_in,
                                 activation_config=activation_config,
                                 topk_ids=topk_ids, expert_map=expert_map,
                                 valid_token_counts=None)

        def moe_sum_v030(moe_input, moe_output, sum_topk_ids,
                         sum_expert_map):
            # MarlinExperts.moe_sum.
            if sum_expert_map is not None:
                ops.moe_sum(moe_input, moe_output, sum_topk_ids,
                            sum_expert_map)
            else:
                ops.moe_sum(moe_input, moe_output)

        def operation():
            with torch.no_grad():
                fused_marlin_moe(
                    hidden_states=hidden_states, w1=weights["w1"],
                    w2=weights["w2"], bias1=None, bias2=None,
                    w1_scale=weights["w1_scale"],
                    w2_scale=weights["w2_scale"],
                    topk_weights=topk_weights, topk_ids=topk_ids,
                    global_scale1=None, global_scale2=None,
                    input_global_scale1=None, input_global_scale2=None,
                    w1_zeros=None, w2_zeros=None,
                    quant_type_id=weights["quant_type_id"],
                    apply_router_weight_on_input=False,
                    global_num_experts=args.num_experts,
                    activation=moe_activation,
                    activation_func=activation_func,
                    activation_config=activation_config,
                    moe_sum=moe_sum_v030, expert_map=expert_map,
                    output=output, intermediate_cache13=workspace2,
                    intermediate_cache2=workspace13, input_dtype=None)

    torch.cuda.synchronize()
    capture.power_warmup(torch)
    for _ in range(args.pre_nvtx_warmup_iterations):
        operation()
    torch.cuda.synchronize()
    capture.before_nvtx()
    torch.cuda.nvtx.range_push(args.nvtx_range_name)
    operation()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    capture.after_nvtx()
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("H200 WQ-MoE output is not finite")
    capture.add_launch_evidence("environment", {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_stack": VLLM_STACK,
        "input_seed": INPUT_SEED,
        "public_op": "vllm.model_executor.layers.fused_moe.experts."
                     "marlin_moe.fused_marlin_moe",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_format", choices=WEIGHT_FORMATS,
                        required=True)
    parser.add_argument("--num_tokens", type=int, required=True)
    parser.add_argument("--num_experts", type=int, required=True,
                        help="global routed experts")
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--intermediate_size", type=int, required=True,
                        help="per-rank expert intermediate size")
    parser.add_argument("--group_size", type=int, required=True)
    parser.add_argument("--ep_size", type=int, required=True)
    parser.add_argument("--ep_rank", type=int, required=True)
    parser.add_argument("--swiglu_limit", type=float, default=None)
    parser.add_argument("--activation", choices=ACTIVATIONS, default="silu")
    parser.add_argument("--activation-situ-beta", type=float, default=None)
    parser.add_argument("--activation-situ-linear-beta", type=float,
                        default=None)
    parser.add_argument("--nvtx_range_name", type=str, required=True)
    parser.add_argument("--pre-nvtx-warmup-iterations", type=int, default=0)
    parser.add_argument("--expected-tok-per-expert-json")
    add_h200_capture_arguments(parser)
    args = parser.parse_args()
    if not args.nvtx_range_name:
        raise ValueError("H200 WQ-MoE runner requires an NVTX name")
    capture = open_h200_capture_session(args, args.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)
    run_wq_moe(args, capture)
    capture.finish()


if __name__ == "__main__":
    main()
