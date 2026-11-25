#!/usr/bin/env python3
"""Measure Host runtime-prepare service outside Device execution.

The benchmark uses short asynchronous call chains.  Device synchronization is
outside the timed interval.  A one-call Torch profiler pass records the number
of physical Device activities; the independently measured eager enqueue cost
is then removed from the fitted Host service slope.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import torch


CHAIN_LENGTHS = (1, 2, 4, 8)
WARMUPS = 8


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linear_slope(xs, ys):
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / denominator


def _cpu_model() -> str:
    names = {
        line.partition(":")[2].strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        if line.startswith("model name")
    }
    if len(names) != 1 or not next(iter(names)):
        raise RuntimeError("Host runtime-prepare capture needs one CPU model identity")
    return next(iter(names))


def _cpu_product(model: str) -> str:
    # Linux often appends an advertised clock to the same vendor/SKU.
    return model.casefold().split(" cpu @", 1)[0].strip()


def _profile_device_activities(call):
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities) as prof:
        call()
        torch.cuda.synchronize()
    device_events = [
        event for event in prof.events()
        if str(event.device_type) == "DeviceType.CUDA"
        and not event.name.startswith("## Call CompiledFxGraph ")
    ]
    return {
        "count": len(device_events),
        "names": [event.name for event in device_events],
    }


def _measure(call, trials, eager_enqueue_ns):
    for _ in range(WARMUPS):
        call()
    torch.cuda.synchronize()
    device_activities = _profile_device_activities(call)
    samples = []
    for trial in range(trials):
        lengths = CHAIN_LENGTHS if trial % 2 == 0 else tuple(reversed(CHAIN_LENGTHS))
        for chain_length in lengths:
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            for _ in range(chain_length):
                call()
            ended = time.perf_counter_ns()
            torch.cuda.synchronize()
            samples.append({
                "trial": trial + 1,
                "chain_length": chain_length,
                "host_interval_ns": ended - started,
            })
    median_total_by_chain = {
        chain_length: statistics.median(
            sample["host_interval_ns"] for sample in samples
            if sample["chain_length"] == chain_length
        )
        for chain_length in CHAIN_LENGTHS
    }
    host_service_ns = _linear_slope(
        CHAIN_LENGTHS,
        tuple(median_total_by_chain[length] for length in CHAIN_LENGTHS),
    )
    submit_ns = device_activities["count"] * eager_enqueue_ns
    runtime_prepare_ns = host_service_ns - submit_ns
    if not math.isfinite(runtime_prepare_ns) or runtime_prepare_ns < 0:
        raise RuntimeError(
            "Host runtime-prepare slope is smaller than independently "
            "measured eager enqueue; recapture instead of clipping")
    return {
        "physical_device_activity_count": device_activities["count"],
        "physical_device_activity_names": device_activities["names"],
        "median_total_ns_by_chain_length": median_total_by_chain,
        "fitted_host_service_ns_per_call": host_service_ns,
        "eager_enqueue_ns_removed_per_call": submit_ns,
        "runtime_prepare_ns_per_call": runtime_prepare_ns,
        "samples": samples,
    }


def _nvidia_vllm_series():
    """Return the installed CUDA vLLM release series; others fail loud.

    The same wrappers are measured on both series; only their import
    locations and the MoE wrapper's inplace keyword differ.
    """

    import vllm

    version = vllm.__version__
    series = ".".join(version.split("+", 1)[0].split(".")[:2])
    if series in ("0.11", "0.30"):
        return series
    raise RuntimeError(
        f"unsupported installed vLLM {version!r} for NVIDIA runtime prepare: "
        "only the 0.11.x and 0.30.x wrapper locations are known")


def _make_cases(hardware_id):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    keepalive = []

    gemm_left = torch.randn((32, 512), device=device, dtype=dtype)
    gemm_right = torch.randn((512, 1536), device=device, dtype=dtype)

    def unquantized_gemm():
        keepalive[:] = [torch.mm(gemm_left, gemm_right)]

    tokens, heads, head_dim = 32, 2, 128
    key = torch.randn((tokens, heads, head_dim), device=device, dtype=dtype)
    value = torch.randn_like(key)
    key_cache = torch.empty((8, 16, heads, head_dim), device=device, dtype=dtype)
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.arange(tokens, device=device, dtype=torch.int64)
    scale = torch.tensor(1.0, device=device, dtype=torch.float32)

    nvidia_series = None
    if hardware_id in ("nvidia_a800", "nvidia_h200"):
        nvidia_series = _nvidia_vllm_series()
    if nvidia_series == "0.11":
        # vllm 0.11 attention/utils/fa_utils.py:11-15 binds
        # _custom_ops.reshape_and_cache_flash and vllm_flash_attn on CUDA.
        from vllm.attention.utils.fa_utils import reshape_and_cache_flash
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_experts,
            fused_topk,
        )
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    elif nvidia_series == "0.30":
        # vllm 0.30 moved the CUDA binding of the same custom op to
        # v1/attention/backends/fa_utils.py:22-28 (_custom_ops.py:2622) and
        # fused_topk to fused_moe/router/fused_topk_router.py:80; the
        # softmax top-k path issues the same topk_softmax op.
        from vllm.v1.attention.backends.fa_utils import (
            reshape_and_cache_flash,
        )
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_experts,
        )
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
            fused_topk,
        )
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    elif nvidia_series is not None:
        raise RuntimeError(f"unhandled NVIDIA vLLM series {nvidia_series!r}")
    elif hardware_id == "metax_c500":
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
            fused_topk,
        )
        from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
            FUSED_MOE_UNQUANTIZED_CONFIG,
            fused_experts,
        )
        # The installed MetaX plugin reads the new vLLM field while its
        # bundled unquantized config predates that field.  Supplying the
        # source-defined false value restores the unquantized path used by
        # the benchmark without changing the installed environment.
        quant_config_type = type(FUSED_MOE_UNQUANTIZED_CONFIG)
        if not hasattr(quant_config_type, "use_int4_w4a8"):
            setattr(
                quant_config_type,
                "use_int4_w4a8",
                property(lambda self: False),
            )
        from vllm_metax.v1.attention.backends.flash_attn import (
            flash_attn_varlen_func,
        )
        from vllm_metax.v1.attention.backends.fa_utils import (
            reshape_and_cache_flash,
        )
    else:
        raise ValueError(f"unsupported hardware: {hardware_id}")

    def reshape_cache():
        reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping,
            "auto", scale, scale,
        )

    query = torch.randn((tokens, 8, head_dim), device=device, dtype=dtype)
    attn_key = torch.randn((tokens, heads, head_dim), device=device, dtype=dtype)
    attn_value = torch.randn_like(attn_key)
    cu = torch.tensor((0, tokens), device=device, dtype=torch.int32)

    def fused_attention():
        if hardware_id in ("nvidia_a800", "nvidia_h200"):
            result = flash_attn_varlen_func(
                query, attn_key, attn_value, tokens, cu, tokens,
                cu_seqlens_k=cu, causal=True,
            )
        else:
            result = flash_attn_varlen_func(
                query, attn_key, attn_value, cu, cu, tokens, tokens,
                causal=True,
            )
        keepalive[:] = [result]

    hidden = torch.randn((tokens, 512), device=device, dtype=dtype)
    # Qwen3-30B-A3B routes with fused_topk (128 experts, topk=8, softmax +
    # renormalize); grouped_topk is the DeepSeek-style path and never runs
    # for this model on either stack.
    gating_hidden = torch.randn((tokens, 2048), device=device, dtype=dtype)
    gating_logits = torch.randn(
        (tokens, 128), device=device, dtype=torch.float32)

    def moe_gating():
        keepalive[:] = list(fused_topk(gating_hidden, gating_logits, 8, True))

    experts, intermediate = 8, 512
    w1 = torch.randn((experts, intermediate * 2, 512), device=device, dtype=dtype)
    w2 = torch.randn((experts, 512, intermediate), device=device, dtype=dtype)
    topk_weights = torch.full((tokens, 4), 0.25, device=device, dtype=torch.float32)
    topk_ids = torch.arange(tokens * 4, device=device, dtype=torch.int64).reshape(tokens, 4) % experts

    def moe_wrapper():
        # vllm 0.30 fused_moe.py:1593-1634 has no inplace keyword: its only
        # op (fused_experts, registered at :1537) is the out-of-place path
        # that 0.11 selects with inplace=False.
        kwargs = {} if nvidia_series == "0.30" else {"inplace": False}
        if hardware_id == "metax_c500":
            kwargs["quant_config"] = FUSED_MOE_UNQUANTIZED_CONFIG
        keepalive[:] = [fused_experts(
            hidden, w1, w2, topk_weights, topk_ids, **kwargs,
        )]

    stream_event = torch.cuda.Event()

    def stream_record_event():
        torch.cuda.current_stream().record_event(stream_event)

    compiled_left = torch.ones((4096,), device=device)
    compiled_right = torch.empty_like(compiled_left)

    def compiled_body(left, right):
        return torch.add(left, 1.0, out=right)

    compiled = torch.compile(compiled_body, dynamic=True)

    def compiled_fx_entry():
        keepalive[:] = [compiled(compiled_left, compiled_right)]

    @torch.library.custom_op(
        "host_runtime_prepare::custom_wrapper",
        mutates_args=(),
    )
    def custom_wrapper(left: torch.Tensor) -> torch.Tensor:
        return left + 1.0

    def compiled_fx_moe_wrapper_control():
        keepalive[:] = [custom_wrapper(compiled_left)]

    return {
        "compiled_fx_backend_entry": compiled_fx_entry,
        "compiled_fx_moe_wrapper_control": compiled_fx_moe_wrapper_control,
        "fused_attention_source_wrapper_prepare": fused_attention,
        "moe_gating_prepare": moe_gating,
        "moe_unquant_mlp_wrapper_prepare": moe_wrapper,
        "reshape_cache_prepare": reshape_cache,
        "stream_record_event_prepare": stream_record_event,
        "unquantized_gemm_prepare": unquantized_gemm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware-id", choices=("nvidia_a800", "nvidia_h200", "metax_c500"),
        required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--eager-enqueue-us", type=float, required=True)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--expected-host-cpu-model")
    parser.add_argument("--expected-vllm-version")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.eager_enqueue_us) or args.eager_enqueue_us <= 0:
        raise ValueError("--eager-enqueue-us must be finite and positive")
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    host_cpu_model = _cpu_model()
    if (args.expected_host_cpu_model is not None
            and (not args.expected_host_cpu_model.strip()
                 or _cpu_product(host_cpu_model) != _cpu_product(
                     args.expected_host_cpu_model))):
        raise RuntimeError(
            f"Host CPU {host_cpu_model!r} differs from expected "
            f"{args.expected_host_cpu_model!r}")
    installed_vllm_version = None
    if args.hardware_id in ("nvidia_a800", "nvidia_h200"):
        import vllm
        installed_vllm_version = vllm.__version__
    if args.expected_vllm_version is not None:
        if installed_vllm_version != args.expected_vllm_version:
            raise RuntimeError(
                f"vLLM {installed_vllm_version!r} differs from expected "
                f"{args.expected_vllm_version!r}")
    torch_device = torch.cuda.get_device_name(args.device_id)
    if args.hardware_id == "nvidia_h200" and torch_device != "NVIDIA H200":
        raise RuntimeError(
            f"nvidia_h200 runtime-prepare capture requires NVIDIA H200, got {torch_device!r}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.cuda.set_device(args.device_id)
    cases = _make_cases(args.hardware_id)
    results = {}
    for name, call in cases.items():
        results[name] = _measure(
            call, args.trials, args.eager_enqueue_us * 1000.0,
        )
        print(name, results[name]["runtime_prepare_ns_per_call"] / 1000.0,
              flush=True)
    source = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "hardware_id": args.hardware_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": platform.node(),
            "host_cpu_model": host_cpu_model,
            "python": sys.version,
            "torch": torch.__version__,
            "device": torch_device,
            "vllm_version": installed_vllm_version,
        },
        "source": {"path": str(source), "sha256": _sha256(source)},
        "contract": {
            "boundary": "short_async_call_chain_slope_minus_measured_physical_enqueue",
            "chain_lengths": list(CHAIN_LENGTHS),
            "warmups": WARMUPS,
            "trials": args.trials,
            "eager_enqueue_us": args.eager_enqueue_us,
        },
        "results": results,
    }
    _write_json(args.output_dir / "summary.json", payload)


if __name__ == "__main__":
    main()
