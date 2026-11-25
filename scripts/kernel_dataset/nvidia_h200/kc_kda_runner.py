"""NVIDIA H200 Kimi Delta Attention (KDA) kernel runner.

Two executed stacks, selected per case with ``--op``:

``vllm_*`` ops -- vLLM 0.24.0 ``KimiGatedDeltaNetAttention._forward``
(``vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py``, the
``KimiLinearForCausalLM`` KDA layer; vLLM 0.24 has no Kimi-K3 model class):

* ``vllm_conv_prefill``: the three ``causal_conv1d_fn`` calls (q, k, v) of the
  prefill branch, channel-last ``x`` (``proj.transpose(0, 1)``), FP32 conv
  weights ``(H*D, width)``, SD conv-state pool viewed as
  ``(num_lines, 3*H*D, width-1) -> chunk(3)``, ``has_initial_state``,
  ``cache_indices`` and the GDN metadata-builder conv metadata
  (``compute_causal_conv1d_metadata``) built before the NVTX range.
* ``vllm_conv_decode``: the three ``causal_conv1d_update`` calls (q, k, v) of
  the decode branch (``validate_data=True``).
* ``vllm_chunk``: ``chunk_kda_with_fused_gate(q, k, v, raw_g, beta, A_log,
  g_bias=dt_bias, initial_state, output_final_state=True,
  use_qk_l2norm_in_kernel=True, cu_seqlens)`` -- l2norm, fused
  gate+chunk-cumsum, intra-chunk KKT, solve_tril, WY recompute, chunk state
  recurrence and output kernels.  ``initial_state`` is gathered from the
  state pool before the NVTX range (``recurrent_state[indices].contiguous()``).
* ``vllm_decode``: ``fused_kda_gate`` followed by ``fused_recurrent_kda`` over
  the in-place FP32 state pool with ``ssm_state_indices`` and ``cu_seqlens``
  (exactly the decode branch), q_len 1 only: vLLM 0.24 KDA has no
  spec-decode path (multi-token verification would need the GDN-style 2-D
  per-token ``ssm_state_indices`` + ``num_accepted_tokens`` layout).
* ``vllm_o_norm``: ``rms_norm_gated(..., activation="sigmoid")`` of
  ``FusedRMSNormGated`` (``o_norm``), per-head RMSNorm over ``head_dim``.

``fla_*`` ops -- the Kimi-K3 reference KDA call of the checkpoint's
``modeling_kimi_linear.py`` (``fla.ops.kda``) with the K3 gate semantics that
vLLM 0.24 kernels do not implement (``safe_gate=True``,
``lower_bound=gate_lower_bound``, ``use_beta_sigmoid_in_kernel=True``,
``transpose_state_layout=True``); executed from flash-linear-attention 0.5.2
on ``sys.path`` via ``--fla-pydeps`` (pure-Python/Triton package; torch and
Triton stay those of the vLLM 0.24 environment):

* ``fla_chunk``: ``chunk_kda(..., use_gate_in_kernel=True, safe_gate=True,
  lower_bound, cu_seqlens)`` (prefill, varlen, B=1 packed).
* ``fla_recurrent``: ``fused_recurrent_kda(..., use_gate_in_kernel=True,
  lower_bound)`` over ``[B, q_len, H, D]`` (the HF decode call).

Shapes are per tensor-parallel rank: ``--num-heads`` is the local KDA head
count and ``--head-dim`` the KDA head dimension (q/k/v share it).  Inputs are
seeded random tensors; one op call (the sequence above) inside one NVTX
range after the H200 power warm-up and WARMUP=5 untimed calls.

vLLM 0.30.0 (``installed_vllm_stack() == "vllm030"``; the branch taken is
recorded as ``kda_case.vllm_stack`` / ``kda_case.call_path``).  The
``vllm_*`` ops keep their names and NVTX boundaries but run the 0.30
``KimiGatedDeltaNetAttention._forward`` calls
(``model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py``, kernels from
``vllm/models/kimi_k3/nvidia/ops/third_party/kda``).  0.30 feeds the layer
one merged projection ``in_proj_qkvgfab`` (KimiLinear layout
``[3*H*D | H | D]`` = mixed_qkv | beta | f_a), so mixed_qkv and the raw beta
logits are row-strided views of it, and beta's sigmoid moved into the kernels:

* ``vllm_conv_prefill``: 3x ``causal_conv1d_fn`` on the strided q/k/v column
  slices of mixed_qkv, packed conv weight ``(3*H*D, width)`` split per q/k/v,
  packed SD conv-state pool viewed ``(lines, 3*H*D, width-1).split``;
* ``vllm_conv_decode``: ONE packed ``causal_conv1d_update`` over all
  ``3*H*D`` channels (``out=`` a separate buffer, ``validate_data=True``);
* ``vllm_chunk``: ``chunk_kda_with_fused_gate(q, k, v, raw_g, raw_beta,
  A_log[H], g_bias, lower_bound=None, initial_state, output_final_state,
  use_qk_l2norm_in_kernel, cu_seqlens)``; ``initial_state`` comes from
  vLLM's ``gather_initial_states`` before the NVTX range (same boundary as
  the 0.24 op, which gathered outside the range too);
* ``vllm_decode``: ``fused_recurrent_kda_packed_decode(mixed_qkv =
  packed conv output, raw_g, raw_beta, A_log, dt_bias, lower_bound=None,
  initial_state=pool, state_indices)`` -- gate, beta sigmoid and recurrence
  in one kernel (0.24: ``fused_kda_gate`` + ``fused_recurrent_kda``);
* ``vllm_o_norm``: ``core_attn_out.copy_(FusedRMSNormGated.forward_cuda
  (core_attn_out, g2))`` (= ``rms_norm_gated(..., "sigmoid")``, the Triton
  path the 0.24 op also used; with Inductor's default ``custom_ops=none`` the
  CustomOp would dispatch ``forward_native`` -- recorded as
  ``custom_op_default_dispatch``).

``k3_*`` ops (vLLM 0.30 only) run ``KimiK3DeltaAttention._forward``
(``vllm/models/kimi_k3/nvidia/kda.py``) for one batch type, from the merged
K3 projection ``[3P | P (output gate) | D (f_a) | H (beta) | pad to 16]``
(full-rank gate), through the output norm:

* ``k3_prefill_flashkda`` / ``k3_prefill_triton``: 3x ``causal_conv1d_fn`` +
  ``gather_initial_states`` + FlashKDA ``torch.ops._flashkda_C.fwd``
  (vLLM's ``_flashkda_prefill``) or ``chunk_kda_with_fused_gate(lower_bound)``
  + ``recurrent_state[indices] = final_state`` (+ the out copy of the Triton
  path) + ``o_norm``;
* ``k3_decode_native``: one ``ops.fused_kda_decode`` (conv update, gate,
  recurrence and gated RMSNorm fused; width-major fp32 conv weight and fp32
  norm weight copies as the layer keeps them);
* ``k3_decode_triton``: packed ``causal_conv1d_update`` +
  ``fused_recurrent_kda_packed_decode(lower_bound)`` + out copy + ``o_norm``;
* ``k3_spec_decode`` (``--q-len`` = 1 + num_spec): ``causal_conv1d_update``
  (``num_accepted_tokens``, ``query_start_loc``, conv state length
  ``width-1+num_spec``) + ``fused_recurrent_kda`` (2-D ``ssm_state_indices``,
  ``num_accepted_tokens``) + ``o_norm``.

Every ``k3_*`` case evaluates vLLM's own ``resolve_kda_prefill_backend`` /
``resolve_kda_decode_backend`` (``auto``) on the H200 and records the result
next to the executed path (``vllm_auto_backend``, ``matches_vllm_auto``).
The FlashKDA checkpoint branch (``mamba_cache_mode="align"``) is not covered.
"""

import argparse
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_h200_capture import (
    add_h200_capture_arguments,
    installed_vllm_stack,
    open_h200_capture_session,
)

import torch

WARMUP = 5
INPUT_SEED = 42
VLLM_OPS = ("vllm_conv_prefill", "vllm_conv_decode", "vllm_chunk",
            "vllm_decode", "vllm_o_norm")
FLA_OPS = ("fla_chunk", "fla_recurrent")
K3_OPS = ("k3_prefill_flashkda", "k3_prefill_triton", "k3_decode_native",
          "k3_decode_triton", "k3_spec_decode")
PREFILL_OPS = ("vllm_conv_prefill", "vllm_chunk", "fla_chunk",
               "k3_prefill_flashkda", "k3_prefill_triton")
O_NORM_EPS = 1e-5


def _measure(fn, nvtx_range_name, capture):
    torch.cuda.synchronize()
    capture.power_warmup(torch)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    capture.before_nvtx()
    torch.cuda.nvtx.range_push(nvtx_range_name)
    fn()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    capture.after_nvtx()


def _cu(lengths, device):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32, device=device)


def _state_lines(batch, device):
    """Scattered state-pool line per request in a pool of 2*batch+1 lines."""
    num_lines = 2 * batch + 1
    lines = torch.randperm(num_lines - 1, device=device)[:batch] + 1
    return num_lines, lines.to(torch.int32)


def _tensor_facts(tensors):
    return {name: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                   "stride": list(t.stride())}
            for name, t in tensors.items()}


def _pad16(width):
    return width + (-width % 16)


def _build_vllm030(args, evidence, device):
    """vLLM 0.30 KimiLinear (``vllm_*``) and Kimi-K3 (``k3_*``) KDA calls."""
    from einops import rearrange
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
    from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
        gather_initial_states,
    )
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda,
        fused_recurrent_kda_packed_decode,
    )
    from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    if FLA_CHUNK_SIZE != args.chunk_size:
        raise ValueError(f"vLLM KDA chunk size is {FLA_CHUNK_SIZE}, "
                         f"plan says {args.chunk_size}")
    layout = get_conv_state_layout()
    if layout != "SD":
        raise RuntimeError(f"runner builds the SD conv-state layout, vLLM uses {layout}")
    H, D, W = args.num_heads, args.head_dim, args.conv_kernel
    B, Q = args.batch, args.q_len
    T, P = B * Q, H * D
    bf16, fp32 = torch.bfloat16, torch.float32
    is_k3 = args.op in K3_OPS
    lower_bound = args.gate_lower_bound if is_k3 else None
    num_spec = Q - 1 if args.op == "k3_spec_decode" else 0
    state_len = W - 1 + num_spec

    # Merged in_proj_qkvgfab output (KimiGatedDeltaNetAttention /
    # KimiK3DeltaAttention.__init__ + forward split).
    if is_k3:
        width = _pad16(4 * P + D + H)
        proj = torch.randn(T, width, dtype=bf16, device=device)
        mixed_qkv = proj[:, :3 * P]
        g2 = proj[:, 3 * P:4 * P].view(T, H, D)
        beta = proj[:, 4 * P + D:4 * P + D + H].unsqueeze(0)
        proj_layout = (f"K3 full-rank gate [3P | P gate | D f_a | H beta | "
                       f"pad {width - (4 * P + D + H)}] width {width}")
    else:
        width = 3 * P + H + D
        proj = torch.randn(T, width, dtype=bf16, device=device)
        mixed_qkv = proj[:, :3 * P]
        beta = proj[:, 3 * P:3 * P + H].unsqueeze(0)
        g2 = torch.randn(T, H, D, dtype=bf16, device=device)
        proj_layout = f"KimiLinear [3P | H beta | D f_a] width {width}"
    g1 = rearrange(torch.randn(T, P, dtype=bf16, device=device),
                   "n (h d) -> 1 n h d", d=D)
    conv_weight = torch.randn(3 * P, 1, W, dtype=fp32, device=device) * 0.1
    conv_weights = conv_weight.view(3 * P, W)
    A_log = torch.log(torch.empty(H, dtype=fp32, device=device).uniform_(1, 16))
    dt_bias = torch.randn(P, dtype=fp32, device=device) * 0.1
    norm_weight = torch.ones(D, dtype=bf16, device=device)
    core_attn_out = torch.empty(1, T, H, D, dtype=bf16, device=device)

    if num_spec:
        num_lines = 2 * B * Q + 1
        lines2d = (torch.randperm(num_lines - 1, device=device)[:B * Q] + 1
                   ).to(torch.int32).view(B, Q)
        lines = lines2d[:, 0].contiguous()
    else:
        num_lines, lines = _state_lines(B, device)
    conv_pool = torch.randn(num_lines, state_len, 3 * P, dtype=bf16, device=device)
    conv_state = conv_pool.transpose(-1, -2)
    q_conv_state, k_conv_state, v_conv_state = conv_state.split(P, dim=-2)
    q_conv_weight, k_conv_weight, v_conv_weight = conv_weights.split(P, dim=0)
    pool = torch.randn(num_lines, H, D, D, dtype=fp32, device=device) * 0.01
    has_initial_state = torch.full((B,), bool(args.has_initial_state),
                                   dtype=torch.bool, device=device)
    qsl_cpu = torch.tensor([Q * i for i in range(B + 1)], dtype=torch.int32)
    query_start_loc = qsl_cpu.to(device)
    q_lens = [Q] * B
    evidence.update({
        "q_lens": q_lens,
        "has_initial_state_per_request": [bool(args.has_initial_state)] * B,
        "kv_context": "none: fixed-size recurrent + conv state per request",
        "chunk_size": args.chunk_size,
        "state_pool_lines": num_lines,
        "state_lines": (lines2d.tolist() if num_spec else lines.tolist()),
        "recurrent_state_per_request": {"shape": [H, D, D], "dtype": "float32"},
        "conv_state_per_request": {"shape": [state_len, 3 * P], "layout": layout,
                                   "dtype": "bfloat16"},
        "num_chunks_per_request": [-(-q // args.chunk_size) for q in q_lens],
        "projection_layout": proj_layout,
        "query_start_loc": qsl_cpu.tolist(),
        "gate_lower_bound": lower_bound,
        "num_spec": num_spec,
        "o_norm": {"eps": O_NORM_EPS, "activation": "sigmoid",
                   "executed": "FusedRMSNormGated.forward_cuda = rms_norm_gated (Triton)",
                   "custom_op_default_dispatch": _o_norm_default_dispatch()},
    })

    def prefill_conv():
        metadata = SimpleNamespace(**dict(zip(
            ("nums_dict", "batch_ptr", "token_chunk_offset_ptr"),
            conv_meta)))
        outs = []
        for x, state, weight in zip(mixed_qkv.split(P, dim=-1),
                                    (q_conv_state, k_conv_state, v_conv_state),
                                    (q_conv_weight, k_conv_weight, v_conv_weight)):
            outs.append(causal_conv1d_fn(
                x.transpose(0, 1), weight, None, activation="silu",
                conv_states=state, has_initial_state=has_initial_state,
                cache_indices=lines, query_start_loc=query_start_loc,
                metadata=metadata).transpose(0, 1))
        return outs

    conv_meta = None
    if args.op in ("vllm_conv_prefill", "k3_prefill_flashkda", "k3_prefill_triton"):
        conv_meta = compute_causal_conv1d_metadata(qsl_cpu, device=torch.device(device))
        evidence["conv_metadata"] = {
            str(bm): {"nums": v["nums"].tolist(), "tot": int(v["tot"]),
                      "mlist_len": int(v["mlist_len"])}
            for bm, v in conv_meta[0].items()}

    def o_norm():
        core_attn_out.copy_(rms_norm_gated(core_attn_out, g2, norm_weight, None,
                                           "sigmoid", eps=O_NORM_EPS))

    op = args.op
    tensors = {"mixed_qkv": mixed_qkv, "raw_beta": beta, "raw_g": g1,
               "conv_weights": conv_weights, "conv_state": conv_state,
               "A_log": A_log, "dt_bias": dt_bias}
    if op == "vllm_conv_prefill":
        fn = prefill_conv
        evidence["call"] = ("3x causal_conv1d_fn(x=mixed_qkv[:, q|k|v].T (row-strided), "
                            "packed fp32 weight split, packed SD pool split, metadata)")
        evidence["public_op"] = ["vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn"] * 3
        tensors.update(cache_indices=lines, has_initial_state=has_initial_state)
    elif op == "vllm_conv_decode":
        conv_out = torch.empty(mixed_qkv.shape, dtype=bf16, device=device)

        def fn():
            causal_conv1d_update(mixed_qkv, conv_state, conv_weights, None,
                                 activation="silu", conv_state_indices=lines,
                                 validate_data=True, out=conv_out)
        evidence["call"] = ("1x packed causal_conv1d_update(mixed_qkv [B, 3HD] row-strided, "
                            "conv_state [lines, 3HD, w-1], conv_state_indices, "
                            "validate_data=True, out=separate buffer)")
        evidence["public_op"] = ["vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update"]
        tensors.update(conv_state_indices=lines, out=conv_out)
    elif op == "vllm_chunk":
        q, k, v = (torch.randn(1, T, H, D, dtype=bf16, device=device) for _ in range(3))
        if not args.has_initial_state:
            pool[lines.long()] = 0
        initial_state = gather_initial_states(pool, lines, has_initial_state)

        def fn():
            chunk_kda_with_fused_gate(
                q=q, k=k, v=v, raw_g=g1, raw_beta=beta, A_log=A_log,
                g_bias=dt_bias, lower_bound=None, initial_state=initial_state,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc)
        evidence["call"] = ("chunk_kda_with_fused_gate(raw_g, raw_beta (strided bf16 logits), "
                            "A_log[H], g_bias, lower_bound=None, use_qk_l2norm_in_kernel=True, "
                            "output_final_state=True, cu_seqlens); initial_state from "
                            "gather_initial_states outside the NVTX range")
        evidence["public_op"] = ["vllm.models.kimi_k3.nvidia.ops.third_party.kda.chunk_kda_with_fused_gate"]
        evidence["cu_seqlens"] = qsl_cpu.tolist()
        tensors.update(q=q, k=k, v=v, initial_state=initial_state, cu_seqlens=query_start_loc)
    elif op == "vllm_decode":
        packed = torch.randn(T, 3 * P, dtype=bf16, device=device)

        def fn():
            fused_recurrent_kda_packed_decode(
                mixed_qkv=packed, raw_g=g1, raw_beta=beta, A_log=A_log,
                dt_bias=dt_bias, lower_bound=None, initial_state=pool,
                state_indices=lines)
        evidence["call"] = ("fused_recurrent_kda_packed_decode(mixed_qkv=packed conv out, "
                            "raw_g, raw_beta, A_log, dt_bias, lower_bound=None, "
                            "initial_state=pool (in place), state_indices)")
        evidence["public_op"] = ["vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent_kda_packed_decode"]
        tensors = {"mixed_qkv": packed, "raw_beta": beta, "raw_g": g1, "A_log": A_log,
                   "dt_bias": dt_bias, "state_pool": pool, "state_indices": lines}
    elif op == "vllm_o_norm":
        fn = o_norm
        evidence["call"] = ("core_attn_out.copy_(FusedRMSNormGated.forward_cuda(core_attn_out, g2)) "
                            "= rms_norm_gated(activation='sigmoid')")
        evidence["public_op"] = ["vllm.third_party.flash_linear_attention.ops.kda.rms_norm_gated"]
        tensors = {"x": core_attn_out, "gate": g2, "weight": norm_weight}
    else:
        from vllm.models.kimi_k3.nvidia import kda as k3_kda
        auto_prefill = k3_kda.resolve_kda_prefill_backend(
            "auto", D, bf16, fp32, lower_bound)
        auto_decode = k3_kda.resolve_kda_decode_backend(
            "auto", H, D, W, num_spec, bf16, bf16, fp32)
        executed = {"k3_prefill_flashkda": "flashkda", "k3_prefill_triton": "triton",
                    "k3_decode_native": "native", "k3_decode_triton": "triton",
                    "k3_spec_decode": "spec_triton"}[op]
        if op.startswith("k3_prefill"):
            auto = auto_prefill
        elif op == "k3_spec_decode":
            # Spec decodes always take the Triton spec branch; the resolver
            # (num_spec > 0) must then refuse the fused native decode.
            if auto_decode != "triton":
                raise RuntimeError(f"num_spec={num_spec} resolved {auto_decode}")
            auto = "spec_triton"
        else:
            auto = auto_decode
        if op == "k3_decode_native" and auto != "native":
            raise ValueError(f"vLLM 0.30 does not run the native fused KDA decode "
                             f"at {H} heads/rank (auto -> {auto})")
        evidence["vllm_auto_backend"] = {"prefill": auto_prefill,
                                         "decode": auto_decode, "for_op": auto}
        evidence["executed_backend"] = executed
        evidence["matches_vllm_auto"] = auto == executed
        evidence["kda_prefill_backend_config"] = "additional_config.kda_prefill_backend"
        evidence["kda_decode_backend_config"] = "additional_config.kda_decode_backend"
        if op in ("k3_prefill_flashkda", "k3_prefill_triton"):
            if not args.has_initial_state:
                pool[lines.long()] = 0
            if op == "k3_prefill_flashkda":
                import vllm._flashkda_C  # noqa: F401
                ws_bytes = int(torch.ops._flashkda_C.get_workspace_size(T, H, B))
                workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=device)
                final_buf = torch.empty(B, H, D, D, dtype=fp32, device=device)
                evidence["flashkda_workspace"] = {
                    "bytes": ws_bytes, "sized_for": {"T": T, "H": H, "N": B},
                    "note": "vLLM sizes it for max_num_batched_tokens/max_num_seqs"}
                tensors.update(workspace=workspace, final_state=final_buf)

            def fn():
                q_ns, k_ns, v_ns = (rearrange(x, "n (h d) -> 1 n h d", d=D)
                                    for x in prefill_conv())
                initial_state = gather_initial_states(pool, lines, has_initial_state)
                if op == "k3_prefill_flashkda":
                    out = core_attn_out[:, :T]
                    _, last = k3_kda._flashkda_prefill(
                        q=q_ns, k=k_ns, v=v_ns, g=g1, beta=beta, A_log=A_log,
                        dt_bias=dt_bias, lower_bound=lower_bound,
                        initial_state=initial_state, cu_seqlens=query_start_loc,
                        out=out, final_state=final_buf[:initial_state.shape[0]],
                        workspace=workspace)
                    pool[lines] = last.to(pool.dtype)
                else:
                    res, last = chunk_kda_with_fused_gate(
                        q=q_ns, k=k_ns, v=v_ns, raw_g=g1, raw_beta=beta,
                        A_log=A_log, g_bias=dt_bias, lower_bound=lower_bound,
                        initial_state=initial_state, output_final_state=True,
                        use_qk_l2norm_in_kernel=True, cu_seqlens=query_start_loc,
                        out=None)
                    pool[lines] = last.to(pool.dtype)
                    core_attn_out[0, :T] = res[0, :T]
                o_norm()
            core = ("FlashKDA torch.ops._flashkda_C.fwd via kda._flashkda_prefill"
                    if op == "k3_prefill_flashkda" else
                    "chunk_kda_with_fused_gate(lower_bound, use_qk_l2norm_in_kernel)")
            evidence["call"] = ("3x causal_conv1d_fn + gather_initial_states + " + core +
                                " + recurrent_state[indices] = final_state" +
                                ("" if op == "k3_prefill_flashkda" else " + out copy") +
                                " + o_norm")
            evidence["public_op"] = [
                "vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn"] * 3 + [
                "vllm.model_executor.layers.mamba.ops.gather_initial_states.gather_initial_states",
                ("vllm.models.kimi_k3.nvidia.kda._flashkda_prefill"
                 if op == "k3_prefill_flashkda" else
                 "vllm.models.kimi_k3.nvidia.ops.third_party.kda.chunk_kda_with_fused_gate"),
                "vllm.third_party.flash_linear_attention.ops.kda.rms_norm_gated"]
            evidence["cu_seqlens"] = qsl_cpu.tolist()
            evidence["checkpoint"] = "none (mamba_cache_mode != align)"
            tensors.update(cache_indices=lines, has_initial_state=has_initial_state,
                           output_gate=g2, state_pool=pool)
        elif op == "k3_decode_native":
            decode_conv1d_weight = torch.stack([
                w.transpose(0, 1) for w in conv_weights.split(P, dim=0)]).contiguous()
            decode_norm_weight = norm_weight.to(fp32)

            def fn():
                ops.fused_kda_decode(
                    x=mixed_qkv, weight=decode_conv1d_weight, bias=None,
                    conv_state=conv_state, raw_g=g1, raw_beta=beta, A_log=A_log,
                    dt_bias=dt_bias, state_indices=lines[:T], state=pool,
                    out=core_attn_out[:, :T], lower_bound=lower_bound,
                    output_gate=g2[:T], norm_weight=decode_norm_weight,
                    norm_eps=O_NORM_EPS)
            evidence["call"] = ("ops.fused_kda_decode(x=mixed_qkv (strided), weight=[3, w, HD] "
                                "fp32, conv_state, raw_g, raw_beta, A_log, dt_bias, "
                                "state_indices, state=pool, lower_bound, output_gate, "
                                "norm_weight fp32, norm_eps)")
            evidence["public_op"] = ["vllm._custom_ops.fused_kda_decode"]
            tensors.update(decode_conv1d_weight=decode_conv1d_weight,
                           state_pool=pool, state_indices=lines, output_gate=g2)
        elif op == "k3_decode_triton":
            conv_out = torch.empty(mixed_qkv.shape, dtype=bf16, device=device)

            def fn():
                packed = causal_conv1d_update(
                    mixed_qkv, conv_state, conv_weights, None, activation="silu",
                    conv_state_indices=lines, validate_data=True, out=conv_out)
                res, _ = fused_recurrent_kda_packed_decode(
                    mixed_qkv=packed, raw_g=g1, raw_beta=beta, A_log=A_log,
                    dt_bias=dt_bias, lower_bound=lower_bound, initial_state=pool,
                    state_indices=lines, out=None)
                core_attn_out[0, :T] = res[0, :T]
                o_norm()
            evidence["call"] = ("causal_conv1d_update (packed, validate_data=True) + "
                                "fused_recurrent_kda_packed_decode(lower_bound) + out copy + o_norm")
            evidence["public_op"] = [
                "vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update",
                "vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent_kda_packed_decode",
                "vllm.third_party.flash_linear_attention.ops.kda.rms_norm_gated"]
            tensors.update(state_pool=pool, state_indices=lines, output_gate=g2)
        elif op == "k3_spec_decode":
            num_accepted = torch.randint(1, Q + 1, (B,), dtype=torch.int32, device=device)
            spec_conv_out = torch.empty(mixed_qkv.shape, dtype=bf16, device=device)
            evidence["num_accepted_tokens"] = num_accepted.tolist()
            evidence["spec_state_indices_shape"] = [B, Q]

            def fn():
                conv = causal_conv1d_update(
                    mixed_qkv, conv_state, conv_weights, None, activation="silu",
                    conv_state_indices=lines2d[:, 0][:B],
                    num_accepted_tokens=num_accepted,
                    query_start_loc=query_start_loc, max_query_len=Q,
                    validate_data=False, out=spec_conv_out)
                q_s, k_s, v_s = (rearrange(x, "n (h d) -> 1 n h d", d=D)
                                 for x in conv.split(P, dim=-1))
                fused_recurrent_kda(
                    q=q_s, k=k_s, v=v_s, raw_g=g1, raw_beta=beta, A_log=A_log,
                    dt_bias=dt_bias, lower_bound=lower_bound, initial_state=pool,
                    cu_seqlens=query_start_loc[:B + 1], ssm_state_indices=lines2d,
                    num_accepted_tokens=num_accepted, out=core_attn_out[:, :T])
                o_norm()
            evidence["call"] = ("causal_conv1d_update(num_accepted_tokens, query_start_loc, "
                                "max_query_len=1+num_spec) + fused_recurrent_kda(2-D "
                                "ssm_state_indices, num_accepted_tokens, lower_bound) + o_norm")
            evidence["public_op"] = [
                "vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update",
                "vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent_kda",
                "vllm.third_party.flash_linear_attention.ops.kda.rms_norm_gated"]
            evidence["cu_seqlens"] = qsl_cpu.tolist()
            tensors.update(state_pool=pool, ssm_state_indices=lines2d,
                           num_accepted_tokens=num_accepted, output_gate=g2)
        else:
            raise ValueError(f"unknown vLLM 0.30 KDA op {op!r}")
    evidence["call_path"] = (
        "vllm030 KimiK3DeltaAttention._forward (models/kimi_k3/nvidia/kda.py)"
        if is_k3 else
        "vllm030 KimiGatedDeltaNetAttention._forward "
        "(model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py)")
    return fn, tensors


def _o_norm_default_dispatch():
    """Forward method FusedRMSNormGated picks under a default VllmConfig."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
    config = VllmConfig()
    with set_current_vllm_config(config):
        enabled = bool(FusedRMSNormGated.enabled())
    return {"vllm_config": "VllmConfig() default",
            "custom_ops": list(config.compilation_config.custom_ops),
            "enabled": enabled,
            "dispatch": "forward_cuda" if enabled else "forward_native"}


def run_kda(args, capture):
    device = "cuda"
    H, D, W = args.num_heads, args.head_dim, args.conv_kernel
    B, Q = args.batch, args.q_len
    if min(H, D, W, B, Q) <= 0:
        raise ValueError("KDA workload dimensions must be positive")
    stack = installed_vllm_stack()
    if args.op in K3_OPS and stack != "vllm030":
        raise ValueError(f"{args.op} is the vLLM 0.30 Kimi-K3 path; "
                         f"installed stack is {stack}")
    if args.op in ("k3_decode_native", "k3_decode_triton") and Q != 1:
        raise ValueError(f"{args.op} is the non-speculative q_len=1 decode")
    if args.op == "k3_spec_decode" and Q < 2:
        raise ValueError("k3_spec_decode needs q_len = 1 + num_spec >= 2")
    if args.op in ("vllm_conv_decode", "vllm_decode") and Q != 1:
        # vLLM 0.24 KDA has no spec-decode path; fused_recurrent_kda with
        # 1-D ssm_state_indices reads one index per token (IMA for q_len>1).
        raise ValueError(f"{args.op} is the vLLM 0.24 q_len=1 decode path")
    if args.has_initial_state not in (0, 1):
        raise ValueError("--has-initial-state must be 0 or 1")
    if args.op not in PREFILL_OPS and args.has_initial_state != 1:
        raise ValueError(f"{args.op} always reads the recurrent/conv state")
    T = B * Q
    dim = H * D
    bf16 = torch.bfloat16
    torch.manual_seed(INPUT_SEED)
    torch.cuda.manual_seed(INPUT_SEED)
    evidence = {"op": args.op, "num_heads": H, "head_dim": D,
                "conv_kernel": W, "batch": B, "q_len": Q, "num_tokens": T,
                "vllm_stack": stack}

    if stack == "vllm030" and args.op in (*VLLM_OPS, *K3_OPS):
        if args.chunk_size != 64:
            raise ValueError("vLLM 0.30 KDA chunk size is 64")
        fn, tensors = _build_vllm030(args, evidence, device)
        with torch.inference_mode():
            _measure(fn, args.nvtx_range_name, capture)
        evidence["tensors"] = _tensor_facts(tensors)
        capture.add_launch_evidence("kda_case", evidence)
        capture.add_launch_evidence("environment", _environment(stack))
        return

    if args.op in VLLM_OPS:
        from vllm.model_executor.layers.fla.ops.kda import (
            chunk_kda_with_fused_gate,
            fused_kda_gate,
            fused_recurrent_kda,
            rms_norm_gated,
        )
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn,
            causal_conv1d_update,
        )
        from vllm.v1.attention.backends.utils import (
            compute_causal_conv1d_metadata,
        )
        from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
        if FLA_CHUNK_SIZE != args.chunk_size:
            raise ValueError(f"vLLM KDA chunk size is {FLA_CHUNK_SIZE}, "
                             f"plan says {args.chunk_size}")
        evidence["stack"] = "vllm"
        evidence["vllm_version"] = importlib.metadata.version("vllm")
        evidence["call_path"] = ("vllm024 KimiGatedDeltaNetAttention._forward "
                                 "(model_executor/layers/fla/ops/kda)")
    else:
        if args.fla_pydeps is None:
            raise ValueError(f"{args.op} requires --fla-pydeps")
        pydeps = Path(args.fla_pydeps)
        if not (pydeps / "fla" / "ops" / "kda").is_dir():
            raise FileNotFoundError(f"fla.ops.kda not found under {pydeps}")
        sys.path.insert(0, str(pydeps))
        import fla
        from fla.ops.kda import chunk_kda, fused_recurrent_kda as fla_recurrent_kda
        evidence["stack"] = "flash-linear-attention"
        evidence["fla_version"] = fla.__version__
        evidence["fla_path"] = str(Path(fla.__file__).resolve().parent)
        evidence["gate_lower_bound"] = args.gate_lower_bound
        evidence["call_path"] = "flash-linear-attention reference (Kimi-K3 HF modeling)"

    q_lens = [Q] * B
    num_lines, lines = _state_lines(B, device)
    evidence.update({
        "q_lens": q_lens,
        "has_initial_state_per_request": [bool(args.has_initial_state)] * B,
        "kv_context": "none: fixed-size recurrent + conv state per request",
        "chunk_size": args.chunk_size,
        "state_pool_lines": num_lines,
        "state_lines": lines.tolist(),
        "recurrent_state_per_request": {"shape": [H, D, D], "dtype": "float32"},
        "conv_state_per_request": {"shape": [W - 1, 3 * dim], "layout": "SD",
                                   "dtype": "bfloat16"},
        "num_chunks_per_request": [-(-q // args.chunk_size) for q in q_lens],
    })
    tensors = {}

    if args.op == "vllm_conv_prefill":
        qkv = [torch.randn(T, dim, dtype=bf16, device=device) for _ in range(3)]
        weights = [torch.randn(dim, W, dtype=torch.float32, device=device) * 0.1
                   for _ in range(3)]
        conv_pool = torch.randn(num_lines, W - 1, 3 * dim, dtype=bf16, device=device)
        conv_views = conv_pool.transpose(-1, -2).chunk(3, dim=-2)
        qsl_cpu = torch.tensor([0] + [Q * (i + 1) for i in range(B)], dtype=torch.int32)
        query_start_loc = qsl_cpu.to(device)
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            qsl_cpu, device=torch.device(device))
        metadata = SimpleNamespace(nums_dict=nums_dict, batch_ptr=batch_ptr,
                                   token_chunk_offset_ptr=token_chunk_offset_ptr)
        has_initial_state = torch.full((B,), bool(args.has_initial_state),
                                       dtype=torch.bool, device=device)
        xs = [x.transpose(0, 1) for x in qkv]

        def fn():
            for x, w, state in zip(xs, weights, conv_views):
                causal_conv1d_fn(x, w, None, activation="silu",
                                 conv_states=state,
                                 has_initial_state=has_initial_state,
                                 cache_indices=lines,
                                 query_start_loc=query_start_loc,
                                 metadata=metadata).transpose(0, 1)
        evidence["call"] = ("3x causal_conv1d_fn(x=proj.T, weight fp32, "
                            "conv_states SD pool view, metadata)")
        evidence["public_op"] = ["vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn"] * 3
        evidence["conv_metadata"] = {
            str(bm): {"nums": v["nums"].tolist(), "tot": int(v["tot"]),
                      "mlist_len": int(v["mlist_len"])}
            for bm, v in nums_dict.items()}
        evidence["query_start_loc"] = qsl_cpu.tolist()
        tensors = {"x_q": xs[0], "weight_q": weights[0], "conv_state_q": conv_views[0],
                   "conv_pool": conv_pool, "cache_indices": lines,
                   "has_initial_state": has_initial_state}
        evidence["has_initial_state"] = bool(args.has_initial_state)

    elif args.op == "vllm_conv_decode":
        qkv = [torch.randn(B, dim, dtype=bf16, device=device) for _ in range(3)]
        weights = [torch.randn(dim, W, dtype=torch.float32, device=device) * 0.1
                   for _ in range(3)]
        conv_pool = torch.randn(num_lines, W - 1, 3 * dim, dtype=bf16, device=device)
        conv_views = conv_pool.transpose(-1, -2).chunk(3, dim=-2)

        def fn():
            for x, w, state in zip(qkv, weights, conv_views):
                causal_conv1d_update(x, state, w, None, activation="silu",
                                     conv_state_indices=lines,
                                     validate_data=True)
        evidence["call"] = "3x causal_conv1d_update(conv_state_indices, validate_data=True)"
        evidence["public_op"] = ["vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update"] * 3
        tensors = {"x_q": qkv[0], "weight_q": weights[0], "conv_state_q": conv_views[0],
                   "conv_pool": conv_pool, "conv_state_indices": lines}

    elif args.op == "vllm_chunk":
        q, k, v, raw_g = (torch.randn(1, T, H, D, dtype=bf16, device=device)
                          for _ in range(4))
        beta = torch.rand(1, T, H, dtype=torch.float32, device=device)
        A_log = torch.log(torch.empty(1, 1, H, 1, dtype=torch.float32,
                                      device=device).uniform_(1, 16))
        dt_bias = torch.randn(dim, dtype=torch.float32, device=device) * 0.1
        pool = torch.randn(num_lines, H, D, D, dtype=torch.float32, device=device) * 0.01
        if not args.has_initial_state:
            pool[lines.long()] = 0
        initial_state = pool[lines.long()].contiguous()
        cu_seqlens = _cu(q_lens, device)

        def fn():
            chunk_kda_with_fused_gate(
                q=q, k=k, v=v, raw_g=raw_g, beta=beta, A_log=A_log,
                g_bias=dt_bias, initial_state=initial_state,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens)
        evidence["call"] = ("chunk_kda_with_fused_gate(use_qk_l2norm_in_kernel=True, "
                            "output_final_state=True, cu_seqlens)")
        evidence["public_op"] = ["vllm.model_executor.layers.fla.ops.kda.chunk_kda_with_fused_gate"]
        evidence["cu_seqlens"] = cu_seqlens.tolist()
        tensors = {"q": q, "k": k, "v": v, "raw_g": raw_g, "beta": beta,
                   "A_log": A_log, "dt_bias": dt_bias, "initial_state": initial_state,
                   "cu_seqlens": cu_seqlens}
        evidence["has_initial_state"] = bool(args.has_initial_state)

    elif args.op == "vllm_decode":
        q, k, v = (torch.randn(1, T, H, D, dtype=bf16, device=device) for _ in range(3))
        g1 = torch.randn(T, dim, dtype=bf16, device=device)
        beta = torch.rand(1, T, H, dtype=torch.float32, device=device)
        A_log = torch.log(torch.empty(1, 1, H, 1, dtype=torch.float32,
                                      device=device).uniform_(1, 16))
        dt_bias = torch.randn(dim, dtype=torch.float32, device=device) * 0.1
        pool = torch.randn(num_lines, H, D, D, dtype=torch.float32, device=device) * 0.01
        cu_seqlens = _cu(q_lens, device)

        def fn():
            g = fused_kda_gate(g1, A_log, D, g_bias=dt_bias).unsqueeze(0)
            fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta,
                                initial_state=pool,
                                use_qk_l2norm_in_kernel=True,
                                cu_seqlens=cu_seqlens,
                                ssm_state_indices=lines)
        evidence["call"] = ("fused_kda_gate(g_bias=dt_bias) + fused_recurrent_kda("
                            "initial_state=pool (in place), cu_seqlens, ssm_state_indices)")
        evidence["public_op"] = ["vllm.model_executor.layers.fla.ops.kda.fused_kda_gate",
                                 "vllm.model_executor.layers.fla.ops.kda.fused_recurrent_kda"]
        evidence["cu_seqlens"] = cu_seqlens.tolist()
        tensors = {"q": q, "k": k, "v": v, "g1_raw": g1, "beta": beta, "A_log": A_log,
                   "dt_bias": dt_bias, "state_pool": pool, "cu_seqlens": cu_seqlens,
                   "ssm_state_indices": lines}

    elif args.op == "vllm_o_norm":
        x = torch.randn(1, T, H, D, dtype=bf16, device=device)
        gate = torch.randn(T, H, D, dtype=bf16, device=device)
        weight = torch.ones(D, dtype=bf16, device=device)

        def fn():
            rms_norm_gated(x, gate, weight, None, "sigmoid", eps=1e-5)
        evidence["call"] = "rms_norm_gated(activation='sigmoid') (FusedRMSNormGated o_norm)"
        evidence["public_op"] = ["vllm.model_executor.layers.fla.ops.kda.rms_norm_gated"]
        tensors = {"x": x, "gate": gate, "weight": weight}

    elif args.op == "fla_chunk":
        q, k, v, g = (torch.randn(1, T, H, D, dtype=bf16, device=device)
                      for _ in range(4))
        beta = torch.randn(1, T, H, dtype=torch.float32, device=device)
        A_log = torch.log(torch.empty(H, dtype=torch.float32,
                                      device=device).uniform_(1, 16))
        dt_bias = torch.randn(dim, dtype=torch.float32, device=device) * 0.1
        if args.has_initial_state:
            initial_state = torch.randn(B, H, D, D, dtype=torch.float32,
                                        device=device) * 0.01
        else:
            initial_state = torch.zeros(B, H, D, D, dtype=torch.float32, device=device)
        cu_seqlens = _cu(q_lens, device).to(torch.int64)

        def fn():
            chunk_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log,
                      dt_bias=dt_bias, initial_state=initial_state,
                      output_final_state=True, use_qk_l2norm_in_kernel=True,
                      use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                      safe_gate=True, lower_bound=args.gate_lower_bound,
                      transpose_state_layout=True, cu_seqlens=cu_seqlens,
                      chunk_size=args.chunk_size)
        evidence["call"] = ("fla chunk_kda(use_gate_in_kernel, use_beta_sigmoid_in_kernel, "
                            "safe_gate=True, lower_bound, transpose_state_layout=True, cu_seqlens)")
        evidence["public_op"] = ["fla.ops.kda.chunk_kda"]
        evidence["cu_seqlens"] = cu_seqlens.tolist()
        tensors = {"q": q, "k": k, "v": v, "g_raw": g, "beta_logits": beta,
                   "A_log": A_log, "dt_bias": dt_bias, "initial_state": initial_state,
                   "cu_seqlens": cu_seqlens}
        evidence["has_initial_state"] = bool(args.has_initial_state)

    elif args.op == "fla_recurrent":
        q, k, v, g = (torch.randn(B, Q, H, D, dtype=bf16, device=device)
                      for _ in range(4))
        beta = torch.randn(B, Q, H, dtype=torch.float32, device=device)
        A_log = torch.log(torch.empty(H, dtype=torch.float32,
                                      device=device).uniform_(1, 16))
        dt_bias = torch.randn(dim, dtype=torch.float32, device=device) * 0.1
        initial_state = torch.randn(B, H, D, D, dtype=torch.float32, device=device) * 0.01

        def fn():
            fla_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log,
                              dt_bias=dt_bias, initial_state=initial_state,
                              output_final_state=True,
                              use_qk_l2norm_in_kernel=True,
                              use_gate_in_kernel=True,
                              use_beta_sigmoid_in_kernel=True,
                              lower_bound=args.gate_lower_bound,
                              transpose_state_layout=True)
        evidence["call"] = ("fla fused_recurrent_kda(use_gate_in_kernel, "
                            "use_beta_sigmoid_in_kernel, lower_bound, "
                            "transpose_state_layout=True)")
        evidence["public_op"] = ["fla.ops.kda.fused_recurrent_kda"]
        tensors = {"q": q, "k": k, "v": v, "g_raw": g, "beta_logits": beta,
                   "A_log": A_log, "dt_bias": dt_bias, "initial_state": initial_state}
    else:
        raise ValueError(f"unknown KDA op {args.op!r}")

    with torch.inference_mode():
        _measure(fn, args.nvtx_range_name, capture)
    evidence["tensors"] = _tensor_facts(tensors)
    capture.add_launch_evidence("kda_case", evidence)
    capture.add_launch_evidence("environment", _environment(stack))


def _environment(stack):
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "triton_version": importlib.metadata.version("triton"),
        "vllm_stack": stack,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True, choices=[*VLLM_OPS, *FLA_OPS, *K3_OPS])
    p.add_argument("--num-heads", type=int, required=True)
    p.add_argument("--head-dim", type=int, required=True)
    p.add_argument("--conv-kernel", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--q-len", type=int, required=True)
    p.add_argument("--has-initial-state", type=int, required=True)
    p.add_argument("--chunk-size", type=int, required=True)
    p.add_argument("--gate-lower-bound", type=float, default=None)
    p.add_argument("--fla-pydeps", default=None)
    p.add_argument("--nvtx_range_name", required=True)
    add_h200_capture_arguments(p)
    a = p.parse_args()
    if a.op in (*FLA_OPS, *K3_OPS) and a.gate_lower_bound is None:
        raise ValueError(f"{a.op} requires --gate-lower-bound")
    if a.op in VLLM_OPS and a.gate_lower_bound is not None:
        raise ValueError("vLLM 0.24 KDA kernels have no lower-bound gate")
    capture = open_h200_capture_session(a, a.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)
    run_kda(a, capture)
    capture.finish()
    print("KDA done")


if __name__ == "__main__":
    main()
