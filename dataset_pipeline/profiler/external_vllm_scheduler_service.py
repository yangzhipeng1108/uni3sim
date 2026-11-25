"""Measure a container's installed vLLM Scheduler without model execution.

This runner is deliberately version-scoped evidence: it imports the vLLM
package installed in the target container instead of Midlayer's pinned 0.11
adapter.  Its output must therefore never be consumed by a Forward contract
for a different vLLM version.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
from statistics import median
import tempfile
import time


# The benchmark measures Python Scheduler service only.  Force vLLM's CPU
# platform before any vLLM import so no model runner or accelerator context is
# constructed on a shared device host.


def install_cpu_platform() -> object:
    """Prevent vLLM's platform discovery from creating a CUDA context."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("importing PyTorch initialized CUDA before the probe")
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    import vllm.platforms as platforms
    from vllm.platforms.cpu import CpuPlatform

    platforms._current_platform = CpuPlatform()
    if not platforms.current_platform.is_cpu():
        raise RuntimeError("external Scheduler probe could not select CpuPlatform")
    return torch


def service_quantile(samples: list[int], numerator: int, denominator: int) -> int:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("Scheduler service samples cannot be empty")
    return ordered[(len(ordered) - 1) * numerator // denominator]


def service_summary(samples: list[int]) -> dict[str, int]:
    return {
        "count": len(samples),
        "min_ns": min(samples),
        "median_ns": int(median(samples)),
        "p95_ns": service_quantile(samples, 95, 100),
        "max_ns": max(samples),
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--max-batched-tokens", type=int, default=512)
    parser.add_argument("--max-requests", type=int, default=16)
    parser.add_argument("--num-kv-blocks", type=int, default=4096)
    parser.add_argument("--kv-block-size-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--platform", choices=("cpu", "native"), default="cpu")
    parser.add_argument(
        "--enable-prefix-caching", action="store_true",
        help="match a capture that scheduled with prefix caching enabled")
    parser.add_argument(
        "--log-stats", action="store_true",
        help="match a capture whose engine collected Scheduler stats; "
             "make_stats() runs inside update_from_output and the prefix "
             "cache counters inside schedule()")
    parser.add_argument(
        "--admission-batch-sizes", type=int, nargs="+", default=[],
        help="cycle these new-request admission batch sizes; each batch is "
             "admitted only while the Scheduler is idle so its first step is a "
             "pure-prefill step at that scheduled request count")
    parser.add_argument(
        "--from-raw", type=Path,
        help="write the trace from an existing raw candidate instead of "
             "running a new measurement")
    parser.add_argument(
        "--trace-root", type=Path,
        help="write per-step records as <root>/<hardware>/<model>/<case>/"
             "scheduler_steps.jsonl.zst plus manifest.json")
    parser.add_argument("--trace-model-id")
    parser.add_argument("--trace-case-id")
    parser.add_argument("--trace-target-qps", type=float)
    parser.add_argument(
        "--trace-phases", nargs="+", choices=("prefill", "mixed", "decode"),
        default=["prefill", "mixed", "decode"],
        help="phases written to the trace; the raw output always keeps every "
             "measured step")
    return parser


# Release series whose KVCacheTensor takes (size, shared_by).
_SHARED_BY_SERIES = ("0.10", "0.11", "0.18", "0.24")


def vllm_series(version: str) -> str:
    """Return the supported vLLM release series; any other version fails.

    The KV cache configuration ABI differs between the supported series, so
    the probe dispatches on the installed release explicitly instead of
    probing constructor signatures.
    """

    release = version.split("+", 1)[0].split(".")
    series = ".".join(release[:2])
    if series in _SHARED_BY_SERIES or series == "0.30":
        return series
    raise RuntimeError(
        f"unsupported installed vLLM {version!r}: this runner supports the "
        f"{', '.join(_SHARED_BY_SERIES)} and 0.30 KVCacheTensor ABIs only")


def kv_cache_tensor(series: str, KVCacheTensor, layer_name: str,
                    num_blocks: int, page_size_bytes: int):
    """One dedicated KV allocation per attention layer, in either ABI."""

    if series in _SHARED_BY_SERIES:
        # size + shared_by in every series before 0.30 this runner has run on
        # (A800 0.11, C500 0.10.2 and 0.24, 910B 0.18; e.g. vllm 0.11
        # v1/kv_cache_interface.py:339-344, 0.24 v1/kv_cache_interface.py:858-859).
        return KVCacheTensor(
            size=num_blocks * page_size_bytes,
            shared_by=[layer_name],
        )
    if series == "0.30":
        # vllm 0.30 v1/kv_cache_interface.py:1361-1382 replaced shared_by
        # with an explicit placement; a single layer-outermost layer is
        # exactly what v1/core/kv_cache_utils.py:1704-1712 builds for one
        # dedicated per-layer tensor (offset 0 in its own allocation).
        return KVCacheTensor(
            size=num_blocks * page_size_bytes,
            layers=[layer_name],
            layer_stride=page_size_bytes * num_blocks,
            block_stride=page_size_bytes,
        )
    raise RuntimeError(f"unsupported vLLM series {series!r}")


def build_scheduler(args: argparse.Namespace):
    """Construct one native CPU scheduler across the vLLM v1 ABI variants."""

    if args.platform == "cpu":
        torch = install_cpu_platform()
    else:
        import torch
    import vllm
    series = vllm_series(vllm.__version__)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    with tempfile.TemporaryDirectory(prefix="midlayer-scheduler-profile-") as directory:
        model_directory = Path(directory)
        model_directory.joinpath("config.json").write_bytes(
            args.model_config.resolve().read_bytes())
        engine_arguments = {
            "model": str(model_directory),
            "dtype": "bfloat16",
            "max_model_len": args.max_model_len,
            "enforce_eager": True,
            "trust_remote_code": False,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": args.enable_prefix_caching,
            "max_num_batched_tokens": args.max_batched_tokens,
            "max_num_seqs": args.max_requests,
            "block_size": args.kv_block_size_tokens,
            "load_format": "auto",
            "skip_tokenizer_init": True,
        }
        if "async_scheduling" in inspect.signature(EngineArgs).parameters:
            engine_arguments["async_scheduling"] = False
        engine_args = EngineArgs(**engine_arguments)
        vllm_config = engine_args.create_engine_config()
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    layer_start, layer_end = model_config.get_layers_start_end_indices(
        parallel_config)
    layer_names = [
        f"model.layers.{index}.self_attn.attn"
        for index in range(layer_start, layer_end)
    ]
    cache_spec = FullAttentionSpec(
        block_size=args.kv_block_size_tokens,
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_size=model_config.get_head_size(),
        dtype=model_config.dtype,
    )
    page_size_bytes = cache_spec.page_size_bytes
    kv_cache_config = KVCacheConfig(
        num_blocks=args.num_kv_blocks,
        kv_cache_tensors=[
            kv_cache_tensor(series, KVCacheTensor, layer_name,
                            args.num_kv_blocks, page_size_bytes)
            for layer_name in layer_names
        ],
        kv_cache_groups=[KVCacheGroupSpec(
            layer_names=layer_names,
            kv_cache_spec=cache_spec,
        )],
    )
    vllm_config.cache_config.num_gpu_blocks = args.num_kv_blocks
    vllm_config.cache_config.num_gpu_blocks_override = args.num_kv_blocks
    scheduler_parameters = inspect.signature(Scheduler).parameters
    scheduler_arguments = {
        "vllm_config": vllm_config,
        "kv_cache_config": kv_cache_config,
        "structured_output_manager": StructuredOutputManager(vllm_config),
    }
    if "block_size" in scheduler_parameters:
        scheduler_arguments["block_size"] = args.kv_block_size_tokens
    if "hash_block_size" in scheduler_parameters:
        scheduler_arguments["hash_block_size"] = args.kv_block_size_tokens
    if args.log_stats:
        if "log_stats" not in scheduler_parameters:
            raise RuntimeError(
                "installed vLLM Scheduler does not accept log_stats")
        scheduler_arguments["log_stats"] = True
    scheduler = Scheduler(**scheduler_arguments)
    return scheduler, vllm.__version__, torch


def block_hasher(block_size: int):
    """Build the engine-core block hasher used when prefix caching is on.

    vLLM hashes a request's full blocks when the engine core admits it, not
    inside schedule().  Attaching the same hasher here keeps the hashing cost
    outside the timed region while letting schedule() run its real prefix
    cache lookup and block caching, exactly as in a captured deployment.
    """

    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher,
        init_none_hash,
    )
    try:
        from vllm.utils.hashing import get_hash_fn_by_name
    except ImportError:
        from vllm.utils import get_hash_fn_by_name

    hash_function = get_hash_fn_by_name("sha256")
    init_none_hash(hash_function)
    return get_request_block_hasher(block_size, hash_function)


_PROMPT_TOKEN_MODULUS = 30000


def prompt_token_ids(request_ordinal: int, prompt_tokens: int) -> list[int]:
    """Give every request a distinct prompt of the same length.

    Identical prompts would make the prefix cache hit on every request after
    the first, which no ShareGPT capture does; the block allocation work in
    schedule() would then be measured on an unrepresentative cache-hit path.
    The modulus stays below the vocabulary of every profiled model.
    """

    base = (request_ordinal + 1) * 1_000_003
    return [(base + index) % _PROMPT_TOKEN_MODULUS for index in range(prompt_tokens)]


def scheduler_request(request_id: str, request_ordinal: int, prompt_tokens: int,
                      output_tokens: int, block_hasher=None):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    arguments = {
        "request_id": request_id,
        "prompt_token_ids": prompt_token_ids(request_ordinal, prompt_tokens),
        "sampling_params": SamplingParams(max_tokens=output_tokens),
        "pooling_params": None,
        "arrival_time": 0.0,
    }
    parameters = inspect.signature(Request).parameters
    if "eos_token_id" in parameters:
        arguments["eos_token_id"] = None
    if block_hasher is not None:
        if "block_hasher" not in parameters:
            raise RuntimeError(
                "installed vLLM Request does not accept a block hasher; "
                "prefix caching cannot be measured with this version")
        arguments["block_hasher"] = block_hasher
    return Request(**arguments)


def runner_output(scheduler_output):
    from vllm.v1.outputs import ModelRunnerOutput

    request_ids = list(scheduler_output.num_scheduled_tokens)
    arguments = {
        "req_ids": request_ids,
        "req_id_to_index": {request_id: index for index, request_id
                              in enumerate(request_ids)},
        "sampled_token_ids": [[42] for _ in request_ids],
        "logprobs": None,
        "prompt_logprobs_dict": {},
        "pooler_output": None,
        "kv_connector_output": None,
    }
    parameters = inspect.signature(ModelRunnerOutput).parameters
    return ModelRunnerOutput(**{
        name: value for name, value in arguments.items()
        if name in parameters
    })


def step_phase(prefill_request_count: int, decode_request_count: int) -> str:
    """Classify one step exactly like the Scheduler Host profile builder."""

    if prefill_request_count > 0 and decode_request_count == 0:
        return "prefill"
    if decode_request_count > 0 and prefill_request_count == 0:
        return "decode"
    if prefill_request_count > 0 and decode_request_count > 0:
        return "mixed"
    raise ValueError("Scheduler step has no scheduled request phase")


def measure_steps(scheduler, args) -> list[dict[str, int | str]]:
    """Time schedule()/update_from_output() per step and classify the state.

    A request is in prefill for one step when its computed token count before
    that schedule() call is still short of its prompt, which is the same rule
    the Serving bridge and the ShareGPT trace adapter apply.
    """

    admission_batch_sizes = tuple(args.admission_batch_sizes) or (args.max_requests,)
    hasher = (
        block_hasher(args.kv_block_size_tokens)
        if args.enable_prefix_caching else None)
    records: list[dict[str, int | str]] = []
    request_index = 0
    batch_index = 0
    while len(records) < args.sample_count:
        if not scheduler.has_unfinished_requests():
            batch_size = admission_batch_sizes[batch_index % len(admission_batch_sizes)]
            batch_index += 1
            for offset in range(batch_size):
                scheduler.add_request(scheduler_request(
                    f"scheduler-microbench-{request_index + offset}",
                    request_index + offset,
                    args.prompt_tokens,
                    args.output_tokens,
                    block_hasher=hasher,
                ))
            request_index += batch_size
        computed_before = {
            request_id: request.num_computed_tokens
            for request_id, request in scheduler.requests.items()
        }
        prompt_tokens = {
            request_id: request.num_prompt_tokens
            for request_id, request in scheduler.requests.items()
        }
        schedule_start_ns = time.perf_counter_ns()
        scheduler_output = scheduler.schedule()
        schedule_end_ns = time.perf_counter_ns()
        prefill_request_count = 0
        decode_request_count = 0
        for request_id in scheduler_output.num_scheduled_tokens:
            if computed_before[request_id] < prompt_tokens[request_id]:
                prefill_request_count += 1
            else:
                decode_request_count += 1
        model_runner_output = runner_output(scheduler_output)
        update_start_ns = time.perf_counter_ns()
        scheduler.update_from_output(scheduler_output, model_runner_output)
        update_end_ns = time.perf_counter_ns()
        scheduled_request_count = prefill_request_count + decode_request_count
        if scheduled_request_count == 0:
            raise RuntimeError(
                "Scheduler microbenchmark produced a step with no scheduled "
                "request; the control workload cannot make progress")
        records.append({
            "step_index": len(records),
            "scheduled_request_count": scheduled_request_count,
            "prefill_request_count": prefill_request_count,
            "decode_request_count": decode_request_count,
            "admission_batch_size": admission_batch_sizes[
                (batch_index - 1) % len(admission_batch_sizes)],
            "phase": step_phase(prefill_request_count, decode_request_count),
            "schedule_start_ns": schedule_start_ns,
            "schedule_end_ns": schedule_end_ns,
            "scheduler_update_start_ns": update_start_ns,
            "scheduler_update_end_ns": update_end_ns,
        })
    return records


def write_trace(records, args, version: str, scheduler_constructor: str, *,
                source_payload=None, source_reference=None) -> dict[str, str]:
    """Emit builder-shaped per-step records under a dataset-root layout."""

    import zstandard
    import re

    # Preserve only this measurement's supplied provenance. Never consult a
    # current checkout or borrow identity from another capture to fill gaps.
    source_commit = None
    source_evidence = None
    if source_payload is not None:
        scheduler = source_payload["scheduler"]
        if scheduler["vllm_version"] != version or scheduler["scheduler_constructor"] != scheduler_constructor:
            raise ValueError("raw Scheduler runtime/constructor provenance disagrees")
        identities = [scheduler]
        if "source_identity" in scheduler and scheduler["source_identity"] is not None:
            if not isinstance(scheduler["source_identity"], dict):
                raise ValueError("raw source_identity must be an object")
            identities.append(scheduler["source_identity"])
        if "software" in source_payload and source_payload["software"]["vllm"] != version:
            raise ValueError("raw software and Scheduler versions disagree")
        if "measurement_source" in scheduler and scheduler["measurement_source"] is not None:
            measured = scheduler["measurement_source"]
            required = {"source_identity", "scheduler_class", "scheduler_constructor",
                        "scheduler_source_files", "worker_pid", "control_workload"}
            if not isinstance(measured, dict) or not required.issubset(measured):
                raise ValueError("same-measurement Scheduler source binding is incomplete")
            identity = measured["source_identity"]
            if (not isinstance(identity, dict) or identity["version"] != version
                    or measured["scheduler_constructor"] != scheduler_constructor
                    or scheduler_constructor.rpartition(".")[2] != measured["scheduler_class"]
                    or type(measured["worker_pid"]) is not int or measured["worker_pid"] <= 0):
                raise ValueError("same-measurement Scheduler source/class binding disagrees")
            identities.append(identity)
            files = measured["scheduler_source_files"]
            if not isinstance(files, list) or not files:
                raise ValueError("actual Scheduler source files are required")
            classes = []
            for entry in files:
                relative = Path(entry["relative_path"])
                if (relative.is_absolute() or ".." in relative.parts
                        or Path(identity["source_root"]) / relative != Path(entry["path"])
                        or not isinstance(entry["sha256"], str)
                        or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None):
                    raise ValueError("actual Scheduler source file binding is invalid")
                classes.append(entry["class"])
            if len(set(classes)) != len(classes) or scheduler_constructor not in classes:
                raise ValueError("actual Scheduler class is absent or duplicated in source binding")
            control = source_payload["measurement"]["control_workload"]
            if measured["control_workload"]["platform"] not in ("cpu", "native"):
                raise ValueError("same-measurement control platform is invalid")
            for name in ("enable_prefix_caching", "log_stats"):
                if type(measured["control_workload"][name]) is not bool or type(control[name]) is not bool:
                    raise ValueError("same-measurement control flags must be boolean")
            admission = measured["control_workload"]["admission_batch_sizes"]
            if not isinstance(admission, list) or not admission or any(type(value) is not int or value <= 0 for value in admission):
                raise ValueError("same-measurement admission counts are invalid")
            for name in ("platform", "enable_prefix_caching", "log_stats", "admission_batch_sizes"):
                if measured["control_workload"][name] != control[name]:
                    raise ValueError("same-measurement control workload provenance disagrees")
            samples = source_payload["samples_ns"]
            if any(len(samples[name]) != len(records)
                   or any(type(value) is not int or value <= 0 for value in samples[name])
                   for name in ("schedule", "update_from_output")):
                raise ValueError("same-measurement samples and step counts disagree")
            for index, record in enumerate(records):
                names = ("schedule_start_ns", "schedule_end_ns", "scheduler_update_start_ns", "scheduler_update_end_ns",
                         "scheduled_request_count", "prefill_request_count", "decode_request_count", "step_index")
                if any(type(record[name]) is not int or record[name] < 0 for name in names):
                    raise ValueError("same-measurement step timestamps/counts must be nonnegative integers")
                if (record["step_index"] != index
                        or record["scheduled_request_count"] != record["prefill_request_count"] + record["decode_request_count"]
                        or record["schedule_end_ns"] - record["schedule_start_ns"] != samples["schedule"][index]
                        or record["scheduler_update_end_ns"] - record["scheduler_update_start_ns"] != samples["update_from_output"][index]
                        or samples["schedule"][index] <= 0 or samples["update_from_output"][index] <= 0):
                    raise ValueError("same-measurement step/sample conservation disagrees")
        commits = set()
        for identity in identities:
            if "version" in identity and identity["version"] != version:
                raise ValueError("source identity version disagrees with raw runtime")
            if "source_commit" in identity and identity["source_commit"] is not None:
                commit = identity["source_commit"]
                if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
                    raise ValueError("source commit must be an observed full Git SHA or unknown")
                commits.add(commit)
        if len(commits) > 1:
            raise ValueError("same-measurement source commit bindings disagree")
        if commits:
            source_commit = next(iter(commits))
        source_evidence = {"scheduler": {name: scheduler[name] for name in
            ("vllm_version", "scheduler_constructor", "source_commit", "source_identity", "measurement_source") if name in scheduler}}
        for name in ("host", "software"):
            if name in source_payload:
                source_evidence[name] = source_payload[name]
        if source_reference is not None:
            if (set(source_reference) != {"path", "sha256"}
                    or re.fullmatch(r"[0-9a-f]{64}", source_reference["sha256"]) is None):
                raise ValueError("raw source content address is malformed")
            source_evidence["raw_artifact"] = source_reference

    for name in ("trace_model_id", "trace_case_id", "trace_target_qps"):
        if getattr(args, name) is None:
            raise ValueError(
                f"--{name.replace('_', '-')} is required with --trace-root")
    if args.trace_target_qps <= 0:
        raise ValueError("--trace-target-qps must be positive")
    selected = [
        record for record in records if record["phase"] in args.trace_phases]
    if not selected:
        raise ValueError("no measured step matches --trace-phases")
    directory = (args.trace_root / args.hardware_id / args.trace_model_id
                 / args.trace_case_id)
    directory.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps({
            name: value for name, value in record.items()
            if name != "phase"
        }, sort_keys=True) + "\n"
        for record in selected
    ).encode("utf-8")
    trace_path = directory / "scheduler_steps.jsonl.zst"
    trace_path.write_bytes(zstandard.ZstdCompressor().compress(payload))
    manifest = {
        "schema_version": 1,
        "identity": {
            "hardware_id": args.hardware_id,
            "model_id": args.trace_model_id,
            "case_id": args.trace_case_id,
            "source_kind": "scheduler_host_microbenchmark",
        },
        "runtime": {
            "runtime_version": version,
            "source_commit": source_commit,
            "target_qps": args.trace_target_qps,
        },
        "measurement": {
            "boundary": "native_scheduler_schedule_and_output_update_only",
            "scheduler_constructor": scheduler_constructor,
            "emitted_phases": sorted(args.trace_phases),
            "measured_step_count": len(records),
            "emitted_step_count": len(selected),
            "control_workload": {
                "platform": args.platform,
                "enable_prefix_caching": args.enable_prefix_caching,
                "log_stats": args.log_stats,
                "admission_batch_sizes": list(
                    args.admission_batch_sizes) or [args.max_requests],
                "max_batched_tokens": args.max_batched_tokens,
                "max_requests": args.max_requests,
                "num_kv_blocks": args.num_kv_blocks,
                "kv_block_size_tokens": args.kv_block_size_tokens,
                "max_model_len": args.max_model_len,
                "prompt_tokens": args.prompt_tokens,
                "output_tokens": args.output_tokens,
            },
        },
    }
    if source_evidence is not None:
        manifest["measurement"]["source_evidence"] = source_evidence
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "trace_path": str(trace_path),
        "manifest_path": str(manifest_path),
    }


def write_trace_from_raw(args) -> None:
    """Emit the dataset-root trace from a raw candidate produced elsewhere.

    Container images that lack zstandard still produce the raw per-step JSON,
    so the trace is written from that same record set rather than from a
    second measurement implementation.
    """

    raw_bytes = args.from_raw.read_bytes()
    payload = json.loads(raw_bytes)
    if payload["hardware_id"] != args.hardware_id:
        raise ValueError("raw candidate hardware disagrees with --hardware-id")
    records = []
    for record in payload["steps"]:
        phase = step_phase(record["prefill_request_count"], record["decode_request_count"])
        if "phase" in record and record["phase"] != phase:
            raise ValueError("raw Scheduler phase disagrees with its actual request counts")
        records.append({**record, "phase": phase})
    control = payload["measurement"]["control_workload"]
    for name in ("platform", "enable_prefix_caching", "log_stats",
                 "admission_batch_sizes",
                 "max_batched_tokens", "max_requests", "num_kv_blocks",
                 "kv_block_size_tokens", "max_model_len", "prompt_tokens",
                 "output_tokens"):
        setattr(args, name, control[name])
    written = write_trace(
        records, args, payload["scheduler"]["vllm_version"],
        payload["scheduler"]["scheduler_constructor"], source_payload=payload,
        source_reference={"path": str(args.from_raw.resolve()), "sha256": hashlib.sha256(raw_bytes).hexdigest()})
    print(json.dumps(written, indent=2, sort_keys=True))


def main() -> None:
    args = argument_parser().parse_args()
    if args.from_raw is not None:
        if args.trace_root is None:
            raise ValueError("--from-raw requires --trace-root")
        write_trace_from_raw(args)
        return
    for name in (
            "sample_count", "max_batched_tokens", "max_requests",
            "num_kv_blocks", "kv_block_size_tokens", "max_model_len",
            "prompt_tokens", "output_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.prompt_tokens > args.max_batched_tokens:
        raise ValueError("--prompt-tokens cannot exceed --max-batched-tokens")
    for batch_size in args.admission_batch_sizes:
        if batch_size < 1 or batch_size > args.max_requests:
            raise ValueError(
                "--admission-batch-sizes entries must be within --max-requests")
        if batch_size * args.prompt_tokens > args.max_batched_tokens:
            raise ValueError(
                "--admission-batch-sizes entry cannot prefill in one step "
                "within --max-batched-tokens")

    scheduler, version, torch = build_scheduler(args)
    if args.platform == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("external Scheduler microbenchmark initialized CUDA")
    records = measure_steps(scheduler, args)
    schedule_samples = [
        record["schedule_end_ns"] - record["schedule_start_ns"]
        for record in records
    ]
    update_samples = [
        record["scheduler_update_end_ns"] - record["scheduler_update_start_ns"]
        for record in records
    ]
    by_state: dict[str, dict[str, list[int]]] = {}
    for record, schedule_ns, update_ns in zip(
            records, schedule_samples, update_samples):
        key = f"{record['phase']}:{record['scheduled_request_count']}"
        if key not in by_state:
            by_state[key] = {"schedule": [], "update": []}
        by_state[key]["schedule"].append(schedule_ns)
        by_state[key]["update"].append(update_ns)

    payload = {
        "schema_version": 2,
        "hardware_id": args.hardware_id,
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "scheduler": {
            "vllm_version": version,
            "scheduler_constructor": str(inspect.signature(type(scheduler).__init__)),
            "model_config_sha256": hashlib.sha256(
                args.model_config.resolve().read_bytes()).hexdigest(),
        },
        "measurement": {
            "boundary": "native_scheduler_schedule_and_output_update_only",
            "control_workload": {
                "platform": args.platform,
                "enable_prefix_caching": args.enable_prefix_caching,
                "log_stats": args.log_stats,
                "admission_batch_sizes": list(
                    args.admission_batch_sizes) or [args.max_requests],
                "sample_count": args.sample_count,
                "max_batched_tokens": args.max_batched_tokens,
                "max_requests": args.max_requests,
                "num_kv_blocks": args.num_kv_blocks,
                "kv_block_size_tokens": args.kv_block_size_tokens,
                "max_model_len": args.max_model_len,
                "prompt_tokens": args.prompt_tokens,
                "output_tokens": args.output_tokens,
            },
        },
        "service_ns": {
            "schedule": service_summary(schedule_samples),
            "update_from_output": service_summary(update_samples),
        },
        "service_ns_by_state": {
            key: {
                "schedule": service_summary(values["schedule"]),
                "update_from_output": service_summary(values["update"]),
            }
            for key, values in sorted(by_state.items())
        },
        "steps": records,
    }
    if args.trace_root is not None:
        payload["trace"] = write_trace(
            records, args, version,
            str(inspect.signature(type(scheduler).__init__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
