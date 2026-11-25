"""Isolated CPU-only process for the pinned vLLM v1 Scheduler.

The PyPI GPU wheel performs platform discovery while importing its config
stack.  The scheduler itself is CPU control-plane code, so this module starts
it in a dedicated process, pins the audited wheel's Python sources, and
installs device-access tripwires before importing vLLM.  No vLLM module is
imported in the simulator process.
"""

from __future__ import annotations

from collections import deque
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
from multiprocessing.connection import Connection
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, replace
from types import ModuleType
from typing import Any
import weakref
from src.serving.deployment.deployment_config import worker_role_operator_group

from src.serving.backend.mtp_acceptance_replay import (
    COLOCATED_MTP_REPLAY_ADAPTER_CONTRACT,
    PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT,
)


VLLM_SCHEDULER_VERSION = "0.11.0"
VLLM_SCHEDULER_SOURCE_COMMIT = "b8b302cde434df8c9289a2b465406b47ebab1c2d"
VLLM030_SOURCE_COMMIT = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
# Unique Python-package binding for the unchanged tagged 0.30 allocator SDK.
# Captured with _package_python_manifest at the source root below; this is
# source identity, not evidence of H200/provider or normal capacity execution.
_VLLM030_ALLOCATOR_PACKAGE_MANIFEST_SHA256 = (
    "c0955783f251f7419d61aa5ae8fbc8b78babc586deab407c322c439c6733ad6f"
)
_STARTUP_TIMEOUT_S = 60.0
# Cold torch/vLLM0.30 imports can exceed one minute during CPU-only parallel
# evaluations. Bound constructor waiting separately; RPC/shutdown deadlines
# and every physical scheduler/model control remain unchanged.
_VLLM030_STARTUP_TIMEOUT_S = 300.0
# A CPU production-trace point can spend several minutes in one scheduler
# command while draining a long-output request.  The previous 60-second RPC
# deadline terminated otherwise healthy runs before their aggregate result was
# persisted.  Keep startup/shutdown deadlines tight, but allow the control
# command to complete within a bounded ten-minute window.
_COMMAND_TIMEOUT_S = 60.0
_PRODUCTION_TRACE_COMMAND_TIMEOUT_S = 600.0
_SHUTDOWN_TIMEOUT_S = 5.0
_DEVICE_AUDIT_COMMAND_INTERVAL = 256
_WORKER_ENVIRONMENT_FLAG = "MIDLAYER_VLLM_SCHEDULER_WORKER"
_PINNED_PACKAGE_PYTHON_FILE_COUNT = 945
_PINNED_PACKAGE_PYTHON_MANIFEST_SHA256 = (
    "49a6f51a5e375a3eabb2dee2042c05efa78d7b03c89235787932c94cc0904bdb"
)

# These files in the installed 0.11.0 wheel are byte-identical to the source
# at VLLM_SCHEDULER_SOURCE_COMMIT.  Pinning the control-plane and bootstrap
# surfaces makes upstream API drift fail before any scheduler state is used.
_PINNED_SOURCE_HASHES = (
    ("v1/core/sched/scheduler.py",
     "357ac414a5b5da8c3423984dd1fdde860584a0d417c30ac2ee25874ea9e5a04c"),
    ("v1/core/sched/async_scheduler.py",
     "f982d4add72c8b695003a2a7721c531f9d4381ee82129a148a5e440ab27896d7"),
    ("v1/core/kv_cache_utils.py",
     "0824beae09a3a8faeb54e8e0605d1661045e34a33a2026de537e6849a4c78ee4"),
    ("v1/kv_cache_interface.py",
     "bf0d73db3442b5bfe96c558c36cd5601e800aea9a1508c989ffe4b54e8d5ce98"),
    ("v1/request.py",
     "be457706757fe6734c5a1bd9f3261722276ec27616dac996d6b1c33cacdfa5ec"),
    ("v1/outputs.py",
     "b859de3c3e057d5c38ce050b35b2bb5b222a4050dbd83f43c3cd1f3b15018124"),
    ("engine/arg_utils.py",
     "2238ade872b5f7478330fa9ca1214389520e08991582090e3e03e2feabcf95ac"),
    ("platforms/__init__.py",
     "48bf2526618f5f435394dfd4ba52cf21ee71acfd1d0eb4bf45ba3c784727dcc7"),
    ("platforms/cpu.py",
     "0a71e9e78a5feb4435884fac762b2718730c2463c961a88200df4ed586d35cd2"),
    ("config/speculative.py",
     "dc025a9d3002e80737046b1efdbc4a70a5f0bc5d119104b3117c13ec033a87ed"),
    ("v1/spec_decode/eagle.py",
     "a2857fd3a3a6c261892b3b25655b513428399a0d3a763d77fcbd52d7105b6ebf"),
    ("v1/worker/gpu_model_runner.py",
     "ea6b7bfe4082349e80042df4801c7ef9e18c80d1c3879eeb6ac6e9c476303574"),
    ("v1/engine/core.py",
     "4573563d926cafe0f1e11ebb25e862ce3bc722af9a24425dec791b9d43cfbedd"),
    ("model_executor/models/glm4_moe.py",
     "7fa520422bdefc0751ba0175a0959300caad7128576de0802b5500cbc70d66e0"),
    ("model_executor/models/glm4_moe_mtp.py",
     "44d2e20dd4b27cb7e5d1b7f1a803b51e1b50be1fbeab92f09b4d62e0a2347472"),
    ("model_executor/models/qwen3.py",
     "ebcfa01f530ecd7c123cdddc6085d015dd809cb0a0189887a09f8a17a11f00ec"),
    ("model_executor/models/qwen2.py",
     "ae57d84ad2a2cbc986538ce221d082f4662e169413988ddaaf4a48539dee1bff"),
    ("model_executor/models/deepseek_v2.py",
     "f9483594c8ccf5af3a67ab6eebd0d55ffee8db7fc79f9a0f30f6db8448dc67ed"),
    ("model_executor/layers/mla.py",
     "b509f0230356f851ef752070627382c541ba2757b7e9c6e68add3367162368b1"),
    ("model_executor/models/llama.py",
     "135cc5e748672057e1528b7f21c7a767c56d2ea4438b73ad60e3cfa3743d8613"),
    ("model_executor/layers/linear.py",
     "b2cc3cb18d161976a957f0b70af302291baa66aaed99229497674f266e1aac4d"),
    ("model_executor/layers/vocab_parallel_embedding.py",
     "f74cbf9882ae59449ce4713cd3ac24b77bdf5a585770399c5968b281f591a6e7"),
    ("model_executor/layers/activation.py",
     "5a514c782e1f5dedc3266a892a50003d102c4ab5e4aa36a769befce2c072ee68"),
    ("model_executor/layers/rotary_embedding/__init__.py",
     "c2649d32e466a5593e254c5d00188e2fdd570d18fec7d39af86dcb8805debb85"),
    ("model_executor/layers/logits_processor.py",
     "3dc2eb8526b0ec49aae393022db488d9220be3fa507893d1d1bff9442529b56e"),
    ("attention/layer.py",
     "83136ff53e104322f5ab1425c19038d75c4ea8b06085b627af13211a39bc2653"),
    ("model_executor/layers/layernorm.py",
     "878ae09c09a54a3565c740a99324946b99e60810ee8297be9850951759bbe022"),
)


def vllm_scheduler_source_identity() -> dict[str, object]:
    """Return the immutable source/package identity of the pinned runtime."""

    return {
        "version": VLLM_SCHEDULER_VERSION,
        "source_commit": VLLM_SCHEDULER_SOURCE_COMMIT,
        "source_hashes": _PINNED_SOURCE_HASHES,
        "package_python_file_count": _PINNED_PACKAGE_PYTHON_FILE_COUNT,
        "package_python_manifest_sha256": (
            _PINNED_PACKAGE_PYTHON_MANIFEST_SHA256),
    }


def vllm030_allocator_source_identity(source_root: Path | None = None) -> dict[str, str]:
    """Return the binding; a supplied root also verifies the actual SDK."""
    if source_root is not None:
        vllm030_source_identity(source_root)
        _, manifest = _package_python_manifest(Path(source_root) / "vllm")
        if manifest != _VLLM030_ALLOCATOR_PACKAGE_MANIFEST_SHA256:
            raise RuntimeError("0.30 allocator Python package differs from its pinned source binding")
    return {"version": "0.30.0", "source_commit": VLLM030_SOURCE_COMMIT,
            "package_python_manifest_sha256": _VLLM030_ALLOCATOR_PACKAGE_MANIFEST_SHA256}


def vllm030_source_identity(source_root: Path) -> dict[str, str]:
    """Verify the real unchanged 0.30 checkout and the imported package root.

    This is a separate source-only SDK identity. It does not relabel the
    production process or the 0.11 runtime/allocator manifest above.
    """
    source_root = Path(source_root).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    tag = subprocess.check_output(
        ["git", "-C", str(source_root), "describe", "--tags", "--exact-match", "HEAD"],
        text=True).strip()
    if commit != VLLM030_SOURCE_COMMIT or tag != "v0.30.0":
        raise RuntimeError("vLLM 0.30 source identity disagrees with the pinned checkout")
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
        text=True)
    if dirty:
        raise RuntimeError("vLLM 0.30 source checkout must be unchanged")
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("cannot locate the real vLLM 0.30 package")
    locations = tuple(Path(location).resolve() for location in spec.submodule_search_locations)
    if locations != (source_root / "vllm",):
        raise RuntimeError("imported vLLM does not resolve to the pinned 0.30 checkout")
    return {"version": "0.30.0", "source_commit": commit,
            "source_tag": tag, "source_root": str(source_root)}


@dataclass(frozen=True)
class VLLMSchedulerRuntimeInfo:
    version: str
    # Official tag used to audit the installed Python-source manifest below.
    source_commit: str
    source_hashes: tuple[tuple[str, str], ...]
    package_python_file_count: int
    package_python_manifest_sha256: str
    platform: str
    model_config_sha256: str
    requested_dtype: str
    effective_dtype: str
    requested_tensor_parallel_size: int
    effective_tensor_parallel_size: int
    requested_pipeline_parallel_size: int
    effective_pipeline_parallel_size: int
    requested_expert_parallel: bool
    effective_expert_parallel: bool
    # These fields identify the synthetic Scheduler KV-geometry
    # representative. They do not identify the central control-plane rank,
    # which is validated separately as pipeline stage 0.
    scheduler_pipeline_stage_rank: int
    scheduler_pipeline_stage_layer_start: int
    scheduler_pipeline_stage_layer_end: int
    scheduler_kv_cache_num_heads_per_rank: int
    scheduler_kv_cache_page_size_bytes: int
    scheduler_kv_cache_tensor_owners: tuple[tuple[str, ...], ...]
    device_guard: str
    bootstrap: str
    requested_chunked_prefill: bool
    effective_chunked_prefill: bool
    requested_prefix_caching: bool
    effective_prefix_caching: bool
    requested_async_scheduling: bool
    effective_async_scheduling: bool
    prefix_oracle_mode: str
    speculative_method: str
    mtp_num_speculative_tokens: int
    mtp_acceptance_policy: str
    mtp_fixed_accepted_tokens: int
    mtp_acceptance_source_id: str
    mtp_acceptance_replay_path: str | None
    mtp_acceptance_replay_sha256: str | None
    mtp_acceptance_replay_artifact_id: str | None
    mtp_acceptance_replay_step_count: int | None
    mtp_acceptance_assumption_sha256: str | None
    mtp_request_stream_sha256: str | None
    mtp_padded_drafter_batch: bool
    mtp_draft_model_type: str
    mtp_num_mtp_layers: int
    mtp_draft_max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int
    requested_gpu_memory_utilization: float
    effective_gpu_memory_utilization: float
    max_num_partial_prefills: int
    max_long_partial_prefills: int
    long_prefill_token_threshold: int
    kv_block_size_tokens: int
    num_kv_blocks: int
    scheduler_policy: str
    num_lookahead_slots: int
    attention_kv_layout: str
    backend_role: str
    kv_handoff_mode: str
    # Review-06 R1 artifact rows for DeepSeek-V4 lifecycle cache groups:
    # (kind, lifecycle, block_tokens, page_bytes, window_tokens|None,
    # layer_count, charged_per_token, raw_per_token); None for other models.
    dsv4_capacity_components: tuple[tuple[str, str, int, int, int | None,
                                          int, str, str], ...] | None = None


def vllm_scheduler_runtime_sha256(
    runtime: VLLMSchedulerRuntimeInfo | dict,
) -> str:
    """Hash requested/effective Scheduler facts, excluding process identity.

    The isolated 0.30 owner exports its native runtime dictionary. Its worker
    PID is a lifecycle observation, not a scheduling/configuration fact; all
    other fields remain bound, including native allocation and device guards.
    """

    if isinstance(runtime, VLLMSchedulerRuntimeInfo):
        facts = asdict(runtime)
    elif isinstance(runtime, dict):
        required = {"version", "worker_pid", "source_identity", "native_allocation", "device_guard",
            "model_executed", "weights_loaded", "gpu_worker_processes_started", "effective_chunked_prefill",
            "requested_max_num_batched_tokens", "effective_max_num_batched_tokens", "effective_cache_block_size"}
        if (required - set(runtime) or runtime["version"] != "0.30.0"
                or type(runtime["worker_pid"]) is not int or runtime["worker_pid"] <= 0
                or runtime["model_executed"] is not False or runtime["weights_loaded"] is not False
                or runtime["gpu_worker_processes_started"] != 0):
            raise ValueError("0.30 Scheduler runtime hashing requires its complete guarded CPU facts")
        facts = dict(runtime)
        facts.pop("worker_pid")
    else:
        raise TypeError(
            "vLLM Scheduler runtime hashing requires "
            "VLLMSchedulerRuntimeInfo or isolated CPU0.30 runtime facts")
    payload = {
        "schema_version": 1,
        "vllm_scheduler_runtime": facts,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class VLLMSchedulerProcess:
    """Synchronous RPC facade over one isolated native vLLM Scheduler."""

    def __init__(self, backend_config: Any, scheduler_config: Any) -> None:
        configured_path = Path(scheduler_config.model).resolve()
        temporary_model_directory = None
        if configured_path.is_file():
            if configured_path.suffix != ".json":
                raise ValueError(
                    "vLLM scheduler model config file must use .json")
            temporary_model_directory = tempfile.TemporaryDirectory(
                prefix="midlayer-vllm-model-config-")
            model_path = Path(temporary_model_directory.name)
            (model_path / "config.json").write_bytes(
                configured_path.read_bytes())
        else:
            model_path = configured_path
        config_path = model_path / "config.json"
        if not model_path.is_dir() or not config_path.is_file():
            raise ValueError(
                "vLLM scheduler model must be a JSON config file or a local "
                "directory containing config.json")
        scheduler_config = replace(scheduler_config, model=str(model_path))
        self.scheduler_config = scheduler_config

        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_scheduler_worker,
            args=(child_connection, backend_config, scheduler_config),
            name=f"midlayer-vllm-scheduler-{backend_config.pool_name}",
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process
        self._temporary_model_directory = temporary_model_directory
        self._closed = False
        self._finalizer = weakref.finalize(
            self,
            _finalize_worker,
            parent_connection,
            process,
            temporary_model_directory,
        )
        try:
            response = self._receive(_STARTUP_TIMEOUT_S)
            self.runtime_info = self._parse_ready(response)
        except BaseException:
            self._force_close()
            raise

    def add_request(self, request: Any) -> None:
        self._rpc(("add_request", request))

    def add_requests(self, requests: tuple[Any, ...]) -> None:
        self._rpc(("add_requests", requests))

    def plan_step(self, virtual_time_s: float) -> Any:
        return self._rpc(("plan_step", virtual_time_s))

    def enable_modeled_simple_cpu_offload(self) -> None:
        self._rpc(("enable_modeled_simple_cpu_offload",))

    def take_modeled_simple_cpu_releases(self) -> tuple[Any, ...]:
        return self._rpc(("take_modeled_simple_cpu_releases",))

    def probe_simple_cpu_connector(self, requests: tuple[Any, ...]) -> Any:
        return self._rpc(("probe_simple_cpu_connector", requests))

    def drain_steps(self, max_steps: int) -> Any:
        return self._rpc(("drain_steps", max_steps))

    def complete_step(
        self,
        step: Any,
        virtual_time_s: float,
        *,
        execute_model: bool,
        execute_transfers: bool,
        sample_tokens: bool,
    ) -> float:
        return self._rpc((
            "complete_step",
            step.schedule.step_id,
            virtual_time_s,
            execute_model,
            execute_transfers,
            sample_tokens,
        ))

    def measure_scheduler_service(
        self,
        *,
        sample_count: int,
        requests_per_batch: int,
        prompt_tokens: int,
        output_tokens: int,
        include_step_records: bool = False,
    ) -> dict[str, tuple[int, ...]]:
        if type(include_step_records) is not bool:
            raise TypeError("include_step_records must be boolean")
        command = ("measure_scheduler_service", sample_count, requests_per_batch,
            prompt_tokens, output_tokens)
        if not include_step_records:
            return self._rpc(command)
        response = self._rpc((*command, True))
        if not isinstance(response, dict) or set(response) != {"samples", "source"}:
            raise TypeError("Scheduler measurement source envelope is malformed")
        self.runtime_info["scheduler_measurement_source"] = response["source"]
        return response["samples"]

    def has_work(self) -> bool:
        result = self._rpc(("has_work",))
        if not isinstance(result, bool):
            raise TypeError("vLLM scheduler process has_work must return boolean")
        return result

    def request_queue_snapshot(self) -> dict:
        from src.serving.backend.vllm import _validated_request_queue_snapshot
        return _validated_request_queue_snapshot(self._rpc(("request_queue_snapshot",)))

    def retain_pd_kv(self, request_id):
        return self._rpc(("retain_pd_kv", request_id))

    def release_pd_kv(self, request_id):
        return self._rpc(("release_pd_kv", request_id))

    def reserve_pd_destination(self, request):
        return self._rpc(("reserve_pd_destination", request))

    def try_reserve_pd_destination(self, request):
        return self._rpc(("try_reserve_pd_destination", request))

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._rpc(("close",))
        finally:
            self._force_close()

    def abort_observation(self) -> None:
        """Explicitly discard a censored CPU observation, never certify drain."""
        if self._closed:
            return
        try:
            self._rpc(("abort_observation",))
        finally:
            self._force_close()

    def _rpc(self, command: tuple[Any, ...]) -> Any:
        if self._closed:
            raise RuntimeError("vLLM scheduler process is closed")
        if not self._process.is_alive():
            self._force_close()
            raise RuntimeError("vLLM scheduler process exited unexpectedly")
        try:
            self._connection.send(command)
            timeout_s = (
                _PRODUCTION_TRACE_COMMAND_TIMEOUT_S
                if command and command[0] == "drain_steps"
                else _COMMAND_TIMEOUT_S)
            response = self._receive(timeout_s)
            return self._parse_response(response)
        except BaseException:
            self._force_close()
            raise

    def _receive(self, timeout_s: float) -> tuple[Any, ...]:
        if not self._connection.poll(timeout_s):
            raise TimeoutError(
                f"vLLM scheduler process did not respond within {timeout_s}s")
        response = self._connection.recv()
        if not isinstance(response, tuple) or not response:
            raise RuntimeError("vLLM scheduler process returned an invalid response")
        return response

    def _parse_ready(
        self,
        response: tuple[Any, ...],
    ) -> VLLMSchedulerRuntimeInfo:
        if response[0] == "ready":
            if len(response) != 2 or not isinstance(
                    response[1], VLLMSchedulerRuntimeInfo):
                raise RuntimeError(
                    "vLLM scheduler process returned invalid runtime provenance")
            return response[1]
        self._raise_worker_error(response)
        raise AssertionError("worker error must raise")

    def _parse_response(self, response: tuple[Any, ...]) -> Any:
        if response[0] == "ok":
            if len(response) != 2:
                raise RuntimeError(
                    "vLLM scheduler process returned malformed success")
            return response[1]
        self._raise_worker_error(response)
        raise AssertionError("worker error must raise")

    @staticmethod
    def _raise_worker_error(response: tuple[Any, ...]) -> None:
        if response[0] != "error" or len(response) != 4:
            raise RuntimeError(
                "vLLM scheduler process returned an unknown response")
        error_type, message, worker_traceback = response[1:]
        raise RuntimeError(
            f"vLLM scheduler worker failed with {error_type}: {message}\n"
            f"{worker_traceback}")

    def _force_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._finalizer.detach()
        _finalize_worker(
            self._connection,
            self._process,
            self._temporary_model_directory,
        )


class _NVMLTripwire(ModuleType):
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"vLLM scheduler attempted NVML access: {name}")


class _SchedulerPopenWorker:
    """Process lifecycle interface for an explicitly selected Python SDK."""

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def terminate(self) -> None:
        self.process.terminate()

    def join(self, timeout: float) -> None:
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return


class VLLM030SchedulerProcess(VLLMSchedulerProcess):
    """Real 0.30 CPU control plane in an isolated SDK, under declared budgets.

    Does not derive normal available-KV, create GPU workers or execute a model.
    RPC feedback uses explicit synthetic length-only tokens for simulation.
    """

    def __init__(self, scheduler_config: Any, *, python_executable: Path,
                 source_root: Path, pool_name: str, max_batched_tokens: int,
                 max_requests: int, kv_block_size_tokens: int,
                 available_memory_by_rank: tuple[int, ...],
                 supported_layouts_by_rank: tuple[tuple[str, ...], ...], role: str = "colocated") -> None:
        from src.serving.backend.vllm import VLLMSchedulerConfig
        if (not isinstance(scheduler_config, VLLMSchedulerConfig)
                or scheduler_config.scheduler_version != "0.30.0"):
            raise ValueError("isolated 0.30 process requires explicit 0.30 Scheduler config")
        if scheduler_config.context_parallel_size != 1 or scheduler_config.data_parallel_size != 1:
            raise NotImplementedError("native CPU0.30 process requires CP1; use VLLMBackend's explicit PCP cache projection")
        if (scheduler_config.max_model_len is None
                or (scheduler_config.async_scheduling and (role != "colocated"
                    or scheduler_config.tensor_parallel_size not in (1, 2)
                    or scheduler_config.pipeline_parallel_size != 1
                    or (scheduler_config.enable_expert_parallel and scheduler_config.tensor_parallel_size != 2)
                    or scheduler_config.mtp_num_speculative_tokens != 0))
                or (scheduler_config.enable_prefix_caching
                    and scheduler_config.prefix_cache_mode != "trace_no_hit"
                    and scheduler_config.simple_cpu_offload_bytes_per_rank is None)
                or scheduler_config.dtype not in ("bfloat16", "float16")):
            raise NotImplementedError("isolated 0.30 path requires explicit length, declared 16-bit dtype, and ordinary TP1/TP2 for async, with only bounded trace_no_hit prefix caching")
        if scheduler_config.mtp_num_speculative_tokens:
            from src.serving.workload.model_config import load_model_config, is_glm_mtp_tensor_model_scope
            mtp_model = load_model_config(scheduler_config.model)
            if scheduler_config.speculative_method == "dspark":
                from src.serving.workload.model_config import is_deepseek_v4_parallel_model_scope
                draft = load_model_config(scheduler_config.dspark_draft_model)
                if (not draft.is_dspark_draft or draft.model_id != mtp_model.model_id + "-DSpark"
                        or role != "colocated" or not is_deepseek_v4_parallel_model_scope(mtp_model,
                            scheduler_config.tensor_parallel_size, scheduler_config.pipeline_parallel_size,
                            expert_parallel=scheduler_config.enable_expert_parallel)):
                    raise NotImplementedError("DSpark CPU Scheduler requires its exact V4 target/draft and legal TP/PP geometry")
            elif (mtp_model.model_type != "glm_moe_dsa" or mtp_model.mtp_architecture != "glm_moe_dsa_mtp"
                    or mtp_model.num_mtp_layers != 1 or scheduler_config.mtp_num_speculative_tokens < 1
                    or not is_glm_mtp_tensor_model_scope(mtp_model, scheduler_config.tensor_parallel_size,
                        expert_parallel=scheduler_config.enable_expert_parallel)
                    or scheduler_config.pipeline_parallel_size > mtp_model.num_hidden_layers
                    or role not in ("colocated", "decode")
                    or (role == "decode" and scheduler_config.pipeline_parallel_size != 1)
                    or scheduler_config.mtp_acceptance_policy not in ("fixed_count", "maximum_entropy")):
                raise NotImplementedError("isolated0.30 MTP requires exact GLM sequential whole-shard TP/nonempty PP or PD decode/PP1 and explicit acceptance")
        if not isinstance(pool_name, str) or not pool_name:
            raise ValueError("0.30 process requires a non-empty pool name")
        if role not in ("colocated", "prefill", "decode") and worker_role_operator_group(role) != "attention":
            raise ValueError("0.30 process requires a whole-model or AF attention Scheduler role")
        for value in (max_batched_tokens, max_requests, kv_block_size_tokens):
            if type(value) is not int or value <= 0:
                raise ValueError("0.30 requested Scheduler limits must be positive integers")
        python_executable = Path(python_executable).absolute()
        if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
            raise ValueError("0.30 process requires an explicit executable SDK Python")
        source_root = Path(source_root).resolve()
        configured = Path(scheduler_config.model).resolve()
        config_file = configured if configured.is_file() else configured / "config.json"
        if not config_file.is_file() or config_file.suffix != ".json":
            raise ValueError("0.30 process requires a real local model config.json")
        startup_request = (scheduler_config, python_executable, source_root, pool_name,
            max_batched_tokens, max_requests, kv_block_size_tokens, available_memory_by_rank,
            supported_layouts_by_rank, role)
        startup_model = hashlib.sha256(config_file.read_bytes()).hexdigest()
        if scheduler_config.enable_expert_parallel:
            from src.serving.workload.model_config import load_model_config
            from src.serving.kv_cache.vllm_allocation import vllm030_supports_expert_parallel_cache
            if not vllm030_supports_expert_parallel_cache(
                    load_model_config(config_file), scheduler_config.tensor_parallel_size):
                raise NotImplementedError("0.30 CPU expert placement requires Qwen3MoE EP2 or GLM/K2.6 EP8 cache scope")
        from src.serving.workload.model_config import load_model_config
        from src.serving.parallelism.pp import pipeline_layer_ranges, pipeline_layer_ranges_for_model
        source_model = load_model_config(config_file)
        from src.serving.backend.vllm import _validate_vllm030_dtype_scope
        _validate_vllm030_dtype_scope(source_model, scheduler_config, role)
        declared_partition = scheduler_config.pipeline_layer_partition
        pipeline_partition_value = None
        if declared_partition is not None:
            pipeline_layer_ranges(source_model.num_hidden_layers, scheduler_config.pipeline_parallel_size, declared_partition)
            pipeline_partition_value = ",".join(map(str, declared_partition))
        elif source_model.model_type == "glm_moe_dsa" and scheduler_config.pipeline_parallel_size > 1:
            ranges = pipeline_layer_ranges_for_model(source_model, scheduler_config.pipeline_parallel_size)
            pipeline_partition_value = ",".join(str(end - first) for first, end in ranges)
        temporary = tempfile.TemporaryDirectory(prefix="midlayer-vllm030-model-")
        model_root = Path(temporary.name)
        (model_root / "config.json").write_bytes(config_file.read_bytes())
        if scheduler_config.speculative_method == "dspark":
            draft_path = Path(scheduler_config.dspark_draft_model).resolve()
            draft_file = draft_path if draft_path.is_file() else draft_path / "config.json"
            if not draft_file.is_file() or draft_file.suffix != ".json":
                temporary.cleanup()
                raise ValueError("DSpark CPU Scheduler requires a real local draft config.json")
            draft_root = model_root / "draft"
            draft_root.mkdir()
            (draft_root / "config.json").write_bytes(draft_file.read_bytes())
            scheduler_config = replace(scheduler_config, dspark_draft_model=str(draft_root))
        scheduler_config = replace(scheduler_config, model=str(model_root))
        parent, child = multiprocessing.get_context("spawn").Pipe(duplex=True)
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES="", VLLM_TARGET_DEVICE="cpu", VLLM_PLUGINS="",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
            VLLM_NO_USAGE_STATS="1", VLLM_CACHE_ROOT=str(model_root / "cache"),
            PYTHONPATH=str(source_root) + os.pathsep + str(Path(__file__).resolve().parents[3]),
        )
        if scheduler_config.dspark_draft_model is not None:
            environment["VLLM_USE_V2_MODEL_RUNNER"]="1"
        if "VLLM_PP_LAYER_PARTITION" in environment:
            del environment["VLLM_PP_LAYER_PARTITION"]
        if pipeline_partition_value is not None:
            environment["VLLM_PP_LAYER_PARTITION"] = pipeline_partition_value
        entry = ("import sys; sys.path.insert(0, sys.argv[2]); "
                 "from src.serving.backend.vllm_scheduler_process "
                 "import _vllm030_pipe_worker; _vllm030_pipe_worker(int(sys.argv[1]))")
        try:
            process = subprocess.Popen(
                [str(python_executable), "-c", entry, str(child.fileno()), str(Path(__file__).resolve().parents[3])],
                env=environment, pass_fds=(child.fileno(),))
        except BaseException:
            parent.close()
            child.close()
            temporary.cleanup()
            raise
        child.close()
        self._connection = parent
        self._process = _SchedulerPopenWorker(process)
        self._temporary_model_directory = temporary
        self._closed = False
        self._finalizer = weakref.finalize(
            self, _finalize_worker, parent, self._process, temporary)
        try:
            parent.send((scheduler_config, source_root, pool_name, max_batched_tokens,
                         max_requests, kv_block_size_tokens, available_memory_by_rank,
                         supported_layouts_by_rank, role))
            self.runtime_info = self._parse_ready(self._receive(_VLLM030_STARTUP_TIMEOUT_S))
            self._startup_request = startup_request
            self._startup_model_sha256 = startup_model
            self._startup_available = True
        except BaseException:
            self._force_close()
            raise

    @property
    def scheduler_config(self):
        """Validated declared configuration, retained independently of RPC use.

        The SDK owns its temporary model copy; its parent still exposes the
        same immutable scheduling/Host contract as the older process owner.
        Reading this does not consume or reopen an allocator handoff.
        """
        return self._startup_request[0]

    def _rpc(self, command):
        self._startup_available = False
        return super()._rpc(command)

    def claim_initial_state(self, scheduler_config, *, python_executable, source_root, pool_name,
            max_batched_tokens, max_requests, kv_block_size_tokens, available_memory_by_rank,
            supported_layouts_by_rank, role):
        """Transfer the allocator's untouched native Scheduler to its backend.

        The same startup request and model bytes must still apply, and no RPC
        may have admitted a request or advanced native state in the meantime.
        """
        if (self._closed or not self._startup_available or not self._process.is_alive()
                or any(type(value) is not int or value <= 0
                    for value in (max_batched_tokens, max_requests, kv_block_size_tokens))
                or type(available_memory_by_rank) is not tuple
                or any(type(value) is not int or value <= 0 for value in available_memory_by_rank)
                or type(supported_layouts_by_rank) is not tuple
                or any(type(row) is not tuple or any(type(value) is not str for value in row)
                    for row in supported_layouts_by_rank)):
            raise ValueError("allocator Scheduler handoff requires untouched live native state and exact limits")
        request = (scheduler_config, Path(python_executable).absolute(), Path(source_root).resolve(), pool_name,
            max_batched_tokens, max_requests, kv_block_size_tokens, available_memory_by_rank,
            supported_layouts_by_rank, role)
        configured = Path(scheduler_config.model).resolve()
        config_file = configured if configured.is_file() else configured / "config.json"
        if (request != self._startup_request
                or hashlib.sha256(config_file.read_bytes()).hexdigest() != self._startup_model_sha256):
            raise ValueError("allocator Scheduler handoff differs from its original native configuration")
        self._startup_available = False

    def _parse_ready(self, response: tuple[Any, ...]) -> dict[str, Any]:
        from src.serving.schedule.backend_protocol import BackendConfig
        from src.serving.kv_cache.vllm_allocation import VLLMKVAllocation
        if response[0] != "ready":
            self._raise_worker_error(response)
        if (len(response) != 3 or not isinstance(response[1], BackendConfig)
                or not isinstance(response[2], dict)
                or not {"version", "runtime_acceptance", "normal_capacity_accepted",
                        "native_allocation"}.issubset(response[2])
                or response[2]["version"] != "0.30.0"
                or response[2]["runtime_acceptance"] is not False
                or response[2]["normal_capacity_accepted"] is not False
                or not isinstance(response[2]["native_allocation"], dict)):
            raise RuntimeError("0.30 worker returned invalid declared-scope runtime provenance")
        allocation = VLLMKVAllocation(**response[2]["native_allocation"])
        if (allocation.version != "0.30.0"
                or allocation.num_kv_blocks != response[1].num_kv_blocks
                or allocation.kv_block_size_tokens != response[1].kv_block_size_tokens):
            raise RuntimeError("0.30 worker allocation disagrees with its effective backend config")
        self.backend_config = response[1]
        return response[2]


def _package_python_manifest(package_root: Path) -> tuple[int, str]:
    source_paths = sorted(
        package_root.rglob("*.py"),
        key=lambda path: path.relative_to(package_root).as_posix(),
    )
    manifest = hashlib.sha256()
    for source_path in source_paths:
        relative_path = source_path.relative_to(package_root).as_posix()
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        manifest.update(relative_path.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(source_hash.encode("ascii"))
        manifest.update(b"\n")
    return len(source_paths), manifest.hexdigest()


def _verify_pinned_vllm_source() -> tuple[Path, str, str]:
    version = importlib.metadata.version("vllm")
    if version != VLLM_SCHEDULER_VERSION:
        raise RuntimeError(
            "unsupported vLLM scheduler version: "
            f"expected {VLLM_SCHEDULER_VERSION}, got {version}")
    package_spec = importlib.util.find_spec("vllm")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise RuntimeError("cannot locate the installed vLLM package")
    locations = tuple(package_spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(
            f"vLLM package must have one source location, got {locations!r}")
    package_root = Path(locations[0]).resolve()
    package_file_count, package_manifest = _package_python_manifest(
        package_root)
    if package_file_count != _PINNED_PACKAGE_PYTHON_FILE_COUNT:
        raise RuntimeError(
            "installed vLLM Python source count differs from the audited "
            f"wheel: expected {_PINNED_PACKAGE_PYTHON_FILE_COUNT}, "
            f"got {package_file_count}")
    if package_manifest != _PINNED_PACKAGE_PYTHON_MANIFEST_SHA256:
        raise RuntimeError(
            "installed vLLM Python source manifest differs from the audited "
            f"wheel: expected {_PINNED_PACKAGE_PYTHON_MANIFEST_SHA256}, "
            f"got {package_manifest}")
    for relative_path, expected_hash in _PINNED_SOURCE_HASHES:
        source_path = package_root / relative_path
        if not source_path.is_file():
            raise RuntimeError(f"pinned vLLM source file is missing: {source_path}")
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                "installed vLLM source differs from audited commit "
                f"{VLLM_SCHEDULER_SOURCE_COMMIT}: {relative_path} "
                f"expected {expected_hash}, got {actual_hash}")
    return package_root, version, package_manifest


def _loaded_accelerator_libraries() -> frozenset[str]:
    libraries: set[str] = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        if "libcuda.so" not in line and "libnvidia-ml.so" not in line:
            continue
        fields = line.split()
        if len(fields) < 6:
            raise RuntimeError(
                "cannot identify accelerator library mapping: " + line)
        libraries.add(fields[-1])
    return frozenset(libraries)


def _install_cpu_device_guard() -> frozenset[str]:
    import torch

    # The CUDA-enabled PyTorch wheel links libcuda during ``import torch`` even
    # though it has not queried a device or initialized a CUDA context.  Record
    # that import-only baseline and reject every later accelerator-library
    # mapping.  Context state, device descriptors, torch APIs, and NVML remain
    # guarded independently below.
    baseline_libraries = _loaded_accelerator_libraries()
    unexpected_baseline = tuple(
        path for path in sorted(baseline_libraries)
        if "libcuda.so" not in Path(path).name
    )
    if unexpected_baseline:
        raise RuntimeError(
            "PyTorch import loaded an unexpected NVIDIA device library: "
            + ", ".join(unexpected_baseline))
    if torch.cuda.is_initialized():
        raise RuntimeError("PyTorch import initialized a CUDA context")

    nvml = _NVMLTripwire("pynvml")
    nvml.__file__ = "<midlayer-nvml-tripwire>"
    sys.modules["pynvml"] = nvml
    bundled_nvml = _NVMLTripwire("vllm.third_party.pynvml")
    bundled_nvml.__file__ = "<midlayer-vllm-nvml-tripwire>"
    bundled_nvml.__package__ = "vllm.third_party"
    sys.modules["vllm.third_party.pynvml"] = bundled_nvml

    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

    def forbidden(name: str):
        def blocked(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise RuntimeError(
                f"vLLM scheduler attempted CUDA access: torch.cuda.{name}")
        return blocked

    for name in (
        "init",
        "_lazy_init",
        "current_device",
        "set_device",
        "synchronize",
        "get_device_capability",
        "get_device_name",
        "get_device_properties",
        "mem_get_info",
        "memory_allocated",
        "memory_reserved",
    ):
        if hasattr(torch.cuda, name):
            setattr(torch.cuda, name, forbidden(name))

    import vllm.platforms as platforms
    from vllm.platforms.cpu import CpuPlatform

    platforms._current_platform = CpuPlatform()
    return baseline_libraries


def _runtime_info(
    version: str,
    package_manifest: str,
    scheduler_config: Any,
    backend_config: Any,
    vllm_config: Any,
    scheduler: Any,
) -> VLLMSchedulerRuntimeInfo:
    config_path = Path(scheduler_config.model) / "config.json"
    effective_scheduler = vllm_config.scheduler_config
    effective_cache = vllm_config.cache_config
    raw_effective_dtype = str(vllm_config.model_config.dtype)
    if raw_effective_dtype == "torch.bfloat16":
        effective_dtype = "bfloat16"
    elif raw_effective_dtype == "torch.float16":
        effective_dtype = "float16"
    else:
        raise RuntimeError(
            "unexpected effective vLLM model dtype: "
            f"{raw_effective_dtype!r}")
    if effective_dtype != scheduler_config.dtype:
        raise RuntimeError(
            "effective vLLM model dtype disagrees with the requested dtype: "
            f"requested={scheduler_config.dtype!r}, "
            f"effective={effective_dtype!r}")
    if scheduler_config.mtp_num_speculative_tokens == 0:
        speculative_method = "disabled"
        mtp_draft_model_type = "disabled"
        mtp_num_mtp_layers = 0
        mtp_draft_max_model_len = 0
    else:
        speculative_method = "mtp"
        speculative_config = vllm_config.speculative_config
        if speculative_config is None:
            raise RuntimeError("effective MTP config disappeared")
        draft_model_config = speculative_config.draft_model_config
        if draft_model_config is None:
            raise RuntimeError("effective MTP draft model disappeared")
        draft_hf_config = draft_model_config.hf_config
        mtp_draft_model_type = draft_hf_config.model_type
        mtp_num_mtp_layers = draft_hf_config.num_nextn_predict_layers
        draft_max_model_len = draft_model_config.max_model_len
        if draft_max_model_len is None:
            draft_max_model_len = vllm_config.model_config.max_model_len
        if (isinstance(draft_max_model_len, bool)
                or not isinstance(draft_max_model_len, int)
                or draft_max_model_len <= 0):
            raise RuntimeError(
                "effective MTP draft max model length must be positive")
        mtp_draft_max_model_len = draft_max_model_len
    if backend_config.is_prefill_only or backend_config.is_decode_only:
        kv_handoff_mode = "metadata_only_private_destination_kv"
    else:
        kv_handoff_mode = "not_applicable"

    parallel = vllm_config.parallel_config
    control_plane_stage_rank = (
        parallel.rank // parallel.tensor_parallel_size
    ) % parallel.pipeline_parallel_size
    if control_plane_stage_rank != 0:
        raise RuntimeError(
            "the central Scheduler KV config must describe pipeline stage 0")
    scheduler_only_mtp = (
        mtp_draft_model_type == "glm_moe_dsa_mtp")
    if (scheduler_config.mtp_num_speculative_tokens
            and not scheduler_only_mtp):
        stage_rank = 0
        layer_start = 0
        layer_end = None
    else:
        stage_rank = (
            vllm_config.kv_geometry_representative_pipeline_stage_rank)
        layer_start = vllm_config.kv_geometry_representative_layer_start
        layer_end = vllm_config.kv_geometry_representative_layer_end
        if (isinstance(stage_rank, bool) or not isinstance(stage_rank, int)
                or not 0 <= stage_rank < parallel.pipeline_parallel_size):
            raise RuntimeError(
                "Scheduler KV-geometry representative stage is invalid")
        if (isinstance(layer_start, bool) or not isinstance(layer_start, int)
                or isinstance(layer_end, bool) or not isinstance(layer_end, int)
                or not 0 <= layer_start < layer_end):
            raise RuntimeError(
                "Scheduler KV-geometry representative layer range is invalid")
    scheduler_kv_cache = scheduler.kv_cache_config
    hf_model_type = json.loads(config_path.read_bytes())["model_type"]
    dsv4_capacity_components = None
    if hf_model_type == "deepseek_v4":
        # Review-06 R1: DeepSeek-V4 runs one Scheduler group per cache
        # lifecycle class; validate the constructed groups component by
        # component against the ModelConfig table and carry the exact
        # charged-vs-raw per-token rates into the runtime artifact.
        from src.serving.workload.model_config import load_model_config

        dsv4_model = load_model_config(scheduler_config.model)
        expected_groups = dsv4_model.dsv4_scheduler_cache_groups(
            backend_config.kv_block_size_tokens)
        from vllm.v1.kv_cache_interface import (
            MLAAttentionSpec,
            SlidingWindowSpec,
        )
        if len(scheduler_kv_cache.kv_cache_groups) != len(expected_groups):
            raise RuntimeError(
                "DeepSeek-V4 Scheduler KV group count disagrees with the "
                f"ModelConfig cache groups: "
                f"{len(scheduler_kv_cache.kv_cache_groups)} vs "
                f"{len(expected_groups)}")
        attention_kv_layout = "mla"
        scheduler_kv_owners = tuple(
            tuple(tensor.shared_by)
            for tensor in scheduler_kv_cache.kv_cache_tensors)
        raw_rate_by_layer_kind = {}
        for layer_index in range(dsv4_model.num_hidden_layers):
            for kind, bytes_per_token, _window in (
                    dsv4_model.dsv4_layer_cache_components(layer_index)):
                raw_rate_by_layer_kind[(layer_index, kind)] = (
                    bytes_per_token)
        expected_owners = []
        component_rows = []
        tensors_by_owner = {}
        for tensor in scheduler_kv_cache.kv_cache_tensors:
            if len(tensor.shared_by) != 1:
                raise RuntimeError(
                    "Scheduler KV tensors must have one layer owner")
            tensors_by_owner[tensor.shared_by[0]] = tensor.size
        for group, group_spec_zip in zip(
                expected_groups,
                scheduler_kv_cache.kv_cache_groups,
                strict=True):
            group_spec = group_spec_zip.kv_cache_spec
            if group.lifecycle == "sliding_window":
                if not isinstance(group_spec, SlidingWindowSpec):
                    raise RuntimeError(
                        f"DeepSeek-V4 group {group.kind!r} must use a "
                        "SlidingWindow spec")
                if group_spec.sliding_window != group.window_tokens:
                    raise RuntimeError(
                        f"DeepSeek-V4 group {group.kind!r} window "
                        f"disagrees: scheduler={group_spec.sliding_window}, "
                        f"model={group.window_tokens}")
            elif group.lifecycle == "stream":
                if type(group_spec) is not MLAAttentionSpec:
                    raise RuntimeError(
                        f"DeepSeek-V4 group {group.kind!r} must use an "
                        "MLAAttention spec")
            else:
                raise AssertionError(
                    f"unhandled DeepSeek-V4 lifecycle: {group.lifecycle!r}")
            if group_spec.block_size != group.block_tokens:
                raise RuntimeError(
                    f"DeepSeek-V4 group {group.kind!r} block size "
                    f"disagrees: scheduler={group_spec.block_size}, "
                    f"model={group.block_tokens}")
            if group_spec.page_size_bytes != group.page_bytes:
                raise RuntimeError(
                    f"DeepSeek-V4 group {group.kind!r} page bytes "
                    f"disagree: scheduler={group_spec.page_size_bytes}, "
                    f"model={group.page_bytes}")
            expected_names = [
                f"model.layers.{index}.self_attn.attn:{group.kind}"
                for index in group.layer_indices
            ]
            if list(group_spec_zip.layer_names) != expected_names:
                raise RuntimeError(
                    f"DeepSeek-V4 group {group.kind!r} layer ownership "
                    "disagrees with the ModelConfig cache inventory")
            for name in expected_names:
                expected_owners.append((name,))
                expected_size = (
                    backend_config.num_kv_blocks * group.page_bytes)
                if name not in tensors_by_owner:
                    raise RuntimeError(
                        f"DeepSeek-V4 Scheduler KV tensor owner is missing: {name!r}")
                if tensors_by_owner[name] != expected_size:
                    raise RuntimeError(
                        f"DeepSeek-V4 tensor {name!r} size disagrees: "
                        f"scheduler={tensors_by_owner[name]}, "
                        f"expected={expected_size}")
            component_kind = group.kind.removesuffix("_r4").removesuffix(
                "_r128")
            raw_per_token = raw_rate_by_layer_kind[(
                group.layer_indices[0], component_kind)]
            component_rows.append((
                group.kind,
                group.lifecycle,
                group.block_tokens,
                group.page_bytes,
                group.window_tokens,
                len(group.layer_indices),
                str(group.per_token_bytes),
                str(raw_per_token),
            ))
        if scheduler_kv_owners != tuple(expected_owners):
            raise RuntimeError(
                "DeepSeek-V4 Scheduler KV tensor owners disagree with the "
                "per-component cache groups")
        dsv4_capacity_components = tuple(component_rows)
        model_kv_bytes_per_token = (
            dsv4_model.kv_bytes_per_token_per_parallel_rank(
                scheduler_config.tensor_parallel_size,
                scheduler_config.pipeline_parallel_size))
        expected_kv_bytes_per_token = (
            -(-model_kv_bytes_per_token.numerator
              // model_kv_bytes_per_token.denominator))
        if backend_config.kv_bytes_per_token != expected_kv_bytes_per_token:
            raise RuntimeError(
                "BackendConfig KV bytes disagree with the ceiled DeepSeek-V4 "
                "per-rank attention-cache capacity: "
                f"{backend_config.kv_bytes_per_token} vs "
                f"{expected_kv_bytes_per_token}")
        scheduler_kv_spec = None
    else:
        if len(scheduler_kv_cache.kv_cache_groups) != 1:
            raise NotImplementedError(
                "runtime provenance currently requires one Scheduler KV group")
        scheduler_kv_spec = scheduler_kv_cache.kv_cache_groups[0].kv_cache_spec
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MLAAttentionSpec,
        )
        if isinstance(scheduler_kv_spec, MLAAttentionSpec):
            attention_kv_layout = "mla"
        elif type(scheduler_kv_spec) is FullAttentionSpec:
            attention_kv_layout = "paged_varlen"
        else:
            raise NotImplementedError(
                "runtime provenance does not support Scheduler KV spec "
                f"{type(scheduler_kv_spec).__name__}")
        scheduler_kv_owners = tuple(
            tuple(tensor.shared_by)
            for tensor in scheduler_kv_cache.kv_cache_tensors)
        if any(len(owners) != 1 for owners in scheduler_kv_owners):
            raise RuntimeError(
                "Scheduler KV tensors must have one layer owner")
        if layer_end is None:
            layer_end = len(scheduler_kv_owners)
        expected_scheduler_kv_owners = tuple(
            (f"model.layers.{layer_index}.self_attn.attn",)
            for layer_index in range(layer_start, layer_end)
        )
        if scheduler_kv_owners != expected_scheduler_kv_owners:
            raise RuntimeError(
                "Scheduler KV tensor owners disagree with the KV-geometry "
                "representative pipeline stage")
        block_size = backend_config.kv_block_size_tokens
        page_size = scheduler_kv_spec.page_size_bytes
        if page_size % block_size:
            raise RuntimeError(
                "vLLM Scheduler KV block size must divide page bytes for "
                "rank-local capacity accounting: "
                f"block_size_tokens={block_size}, page_size_bytes={page_size}")
        total_scheduler_kv_bytes = sum(
            tensor.size for tensor in scheduler_kv_cache.kv_cache_tensors)
        capacity_denominator = backend_config.num_kv_blocks * block_size
        if total_scheduler_kv_bytes % capacity_denominator:
            raise RuntimeError(
                "Scheduler KV tensor bytes must divide by its block "
                "capacity")
        expected_kv_bytes_per_token = (
            total_scheduler_kv_bytes // capacity_denominator)
        if backend_config.kv_bytes_per_token != expected_kv_bytes_per_token:
            raise RuntimeError(
                "BackendConfig KV bytes per token disagree with the pinned "
                "vLLM rank-local KV ownership: "
                f"{backend_config.kv_bytes_per_token} vs "
                f"{expected_kv_bytes_per_token}")

    replay = scheduler_config.mtp_acceptance_replay
    assumption = scheduler_config.mtp_acceptance_assumption
    runtime = VLLMSchedulerRuntimeInfo(
        version=version,
        source_commit=VLLM_SCHEDULER_SOURCE_COMMIT,
        source_hashes=_PINNED_SOURCE_HASHES,
        package_python_file_count=_PINNED_PACKAGE_PYTHON_FILE_COUNT,
        package_python_manifest_sha256=package_manifest,
        platform="cpu",
        model_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        requested_dtype=scheduler_config.dtype,
        effective_dtype=effective_dtype,
        requested_tensor_parallel_size=(
            scheduler_config.tensor_parallel_size),
        effective_tensor_parallel_size=(
            vllm_config.parallel_config.tensor_parallel_size),
        requested_pipeline_parallel_size=(
            scheduler_config.pipeline_parallel_size),
        effective_pipeline_parallel_size=(
            vllm_config.parallel_config.pipeline_parallel_size),
        requested_expert_parallel=(
            scheduler_config.enable_expert_parallel),
        effective_expert_parallel=(
            vllm_config.parallel_config.enable_expert_parallel),
        scheduler_pipeline_stage_rank=stage_rank,
        scheduler_pipeline_stage_layer_start=layer_start,
        scheduler_pipeline_stage_layer_end=layer_end,
        scheduler_kv_cache_num_heads_per_rank=(
            1
            if scheduler_kv_spec is None
            else scheduler_kv_spec.num_kv_heads),
        # For DeepSeek-V4 lifecycle groups this is the summed page bytes of
        # one block across every cache group; the per-component geometry is
        # carried by dsv4_capacity_components below.
        scheduler_kv_cache_page_size_bytes=(
            sum(row[3] for row in dsv4_capacity_components)
            if dsv4_capacity_components is not None
            else scheduler_kv_spec.page_size_bytes),
        scheduler_kv_cache_tensor_owners=scheduler_kv_owners,
        device_guard="python_tripwire+cuda_state+procfs",
        bootstrap=(
            "version_pinned_scheduler_only_mtp_cpu"
            if scheduler_only_mtp
            else (
                "version_pinned_engine_args_mtp_cpu"
                if scheduler_config.mtp_num_speculative_tokens
                else "version_pinned_scheduler_only_cpu")
        ),
        requested_chunked_prefill=scheduler_config.enable_chunked_prefill,
        effective_chunked_prefill=(
            effective_scheduler.chunked_prefill_enabled),
        requested_prefix_caching=scheduler_config.enable_prefix_caching,
        effective_prefix_caching=effective_cache.enable_prefix_caching,
        requested_async_scheduling=scheduler_config.async_scheduling,
        effective_async_scheduling=effective_scheduler.async_scheduling,
        prefix_oracle_mode=(
            "native_token_ids_v1" if scheduler_config.prefix_cache_mode == "token_ids"
            else "trace_no_hit_distinct_first_block_v1"
            if scheduler_config.prefix_cache_mode == "trace_no_hit"
            else "trace_length_private_nonshared_v1"),
        speculative_method=speculative_method,
        mtp_num_speculative_tokens=(
            scheduler_config.mtp_num_speculative_tokens),
        mtp_acceptance_policy=scheduler_config.mtp_acceptance_policy,
        mtp_fixed_accepted_tokens=(
            scheduler_config.mtp_fixed_accepted_tokens),
        mtp_acceptance_source_id=(
            scheduler_config.mtp_acceptance_source_id),
        mtp_acceptance_replay_path=(
            None if replay is None else replay.source_path),
        mtp_acceptance_replay_sha256=(
            None if replay is None else replay.source_sha256),
        mtp_acceptance_replay_artifact_id=(
            None if replay is None else replay.artifact_id),
        mtp_acceptance_replay_step_count=(
            None if replay is None else len(replay.steps)),
        mtp_acceptance_assumption_sha256=(
            None
            if assumption is None
            else assumption.policy_descriptor_sha256),
        mtp_request_stream_sha256=scheduler_config.request_stream_sha256,
        mtp_padded_drafter_batch=(
            scheduler_config.mtp_num_speculative_tokens > 0),
        mtp_draft_model_type=mtp_draft_model_type,
        mtp_num_mtp_layers=mtp_num_mtp_layers,
        mtp_draft_max_model_len=mtp_draft_max_model_len,
        max_num_batched_tokens=effective_scheduler.max_num_batched_tokens,
        max_num_seqs=effective_scheduler.max_num_seqs,
        max_model_len=effective_scheduler.max_model_len,
        requested_gpu_memory_utilization=(
            scheduler_config.gpu_memory_utilization),
        effective_gpu_memory_utilization=(
            effective_cache.gpu_memory_utilization),
        max_num_partial_prefills=(
            effective_scheduler.max_num_partial_prefills),
        max_long_partial_prefills=(
            effective_scheduler.max_long_partial_prefills),
        long_prefill_token_threshold=(
            effective_scheduler.long_prefill_token_threshold),
        kv_block_size_tokens=effective_cache.block_size,
        num_kv_blocks=backend_config.num_kv_blocks,
        scheduler_policy=effective_scheduler.policy,
        num_lookahead_slots=effective_scheduler.num_lookahead_slots,
        attention_kv_layout=attention_kv_layout,
        backend_role=backend_config.role,
        kv_handoff_mode=kv_handoff_mode,
        dsv4_capacity_components=dsv4_capacity_components,
    )
    _validate_mtp_acceptance_replay_runtime(
        scheduler_config, backend_config, runtime)
    return runtime


def _validate_mtp_acceptance_replay_runtime(
    scheduler_config: Any,
    backend_config: Any,
    runtime: VLLMSchedulerRuntimeInfo,
) -> None:
    replay = scheduler_config.mtp_acceptance_replay
    if replay is None:
        return
    provenance = replay.provenance
    context = replay.scheduler_context
    actual_provenance = (
        runtime.version,
        runtime.source_commit,
        runtime.package_python_manifest_sha256,
        runtime.model_config_sha256,
    )
    expected_provenance = (
        provenance.vllm_version,
        provenance.vllm_source_commit,
        provenance.vllm_package_manifest_sha256,
        provenance.model_config_sha256,
    )
    if actual_provenance != expected_provenance:
        raise RuntimeError(
            "MTP acceptance replay provenance disagrees with the effective "
            f"vLLM runtime: expected={expected_provenance}, "
            f"actual={actual_provenance}")
    actual_common_context = (
        backend_config.pool_name,
        runtime.backend_role,
        runtime.scheduler_policy,
        runtime.max_num_batched_tokens,
        runtime.max_num_seqs,
        runtime.num_kv_blocks,
        runtime.kv_block_size_tokens,
        runtime.max_model_len,
        runtime.effective_dtype,
        runtime.mtp_num_speculative_tokens,
        runtime.mtp_draft_model_type,
        runtime.mtp_num_mtp_layers,
        runtime.mtp_draft_max_model_len,
        runtime.mtp_padded_drafter_batch,
        runtime.prefix_oracle_mode,
        runtime.attention_kv_layout,
        runtime.effective_chunked_prefill,
        runtime.max_num_partial_prefills,
        runtime.max_long_partial_prefills,
        runtime.long_prefill_token_threshold,
    )
    expected_common_context = (
        context.pool_name,
        context.backend_role,
        context.scheduler_policy,
        context.max_batched_tokens,
        context.max_requests,
        context.num_kv_blocks,
        context.kv_block_size_tokens,
        context.max_model_len,
        context.dtype,
        context.num_speculative_tokens,
        context.draft_model_type,
        context.num_mtp_layers,
        context.draft_max_model_len,
        context.padded_drafter_batch,
        context.prefix_oracle_mode,
        context.attention_kv_layout,
        context.chunked_prefill,
        context.max_num_partial_prefills,
        context.max_long_partial_prefills,
        context.long_prefill_token_threshold,
    )
    if context.adapter_contract == COLOCATED_MTP_REPLAY_ADAPTER_CONTRACT:
        actual_identity = (
            COLOCATED_MTP_REPLAY_ADAPTER_CONTRACT,
            scheduler_config.request_stream_sha256,
        )
        expected_identity = (
            context.adapter_contract,
            context.request_stream_sha256,
        )
    elif context.adapter_contract == PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT:
        actual_identity = (
            PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT,
            scheduler_config.request_stream_sha256,
            scheduler_config.decode_admission_contract_sha256,
        )
        expected_identity = (
            context.adapter_contract,
            context.ingress_request_stream_sha256,
            context.decode_admission_contract_sha256,
        )
    else:
        raise RuntimeError("unknown MTP replay adapter contract")
    actual_context = actual_identity + actual_common_context
    expected_context = expected_identity + expected_common_context
    if actual_context != expected_context:
        raise RuntimeError(
            "MTP acceptance replay Scheduler context disagrees with the "
            f"effective runtime: expected={expected_context}, "
            f"actual={actual_context}")


def _verify_no_accelerator_access(
    baseline_libraries: frozenset[str],
) -> None:
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "vLLM scheduler initialized a CUDA context")
    fd_root = Path("/proc/self/fd")
    for descriptor in fd_root.iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            # Iterating /proc/self/fd temporarily creates a descriptor for the
            # directory itself.  It can close before readlink; a closed FD
            # cannot retain an accelerator device handle.
            target = None
        if target is not None and (
                target.startswith("/dev/nvidia")
                or target.startswith("/dev/dri")):
            raise RuntimeError(
                "vLLM scheduler opened an accelerator device: "
                f"{descriptor.name} -> {target}")
    unexpected_libraries = tuple(sorted(
        _loaded_accelerator_libraries() - baseline_libraries
    ))
    if unexpected_libraries:
        raise RuntimeError(
            "vLLM scheduler loaded an NVIDIA device library after the "
            "PyTorch import baseline: " + ", ".join(unexpected_libraries))


def _verify_no_cuda_context() -> None:
    """Cheap per-command guard between complete procfs audits."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("vLLM scheduler initialized a CUDA context")


def _scheduler_worker(
    connection: Connection,
    backend_config: Any,
    scheduler_config: Any,
) -> None:
    try:
        _, version, package_manifest = _verify_pinned_vllm_source()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_CPU_KVCACHE_SPACE"] = "1"
        os.environ[_WORKER_ENVIRONMENT_FLAG] = "1"
        baseline_libraries = _install_cpu_device_guard()

        from src.serving.backend.vllm import _NativeVLLMControlPlane

        control_plane = _NativeVLLMControlPlane(
            backend_config,
            scheduler_config,
        )
        _verify_no_accelerator_access(baseline_libraries)
        connection.send((
            "ready",
            _runtime_info(
                version,
                package_manifest,
                scheduler_config,
                backend_config,
                control_plane.vllm_config,
                control_plane.scheduler,
            ),
        ))
        _serve_scheduler_commands(connection, control_plane, baseline_libraries)
    except EOFError:
        return
    except BaseException as exc:
        try:
            connection.send((
                "error",
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            ))
        except (BrokenPipeError, EOFError, OSError) as report_error:
            raise RuntimeError(
                "vLLM scheduler worker could not report its failure"
            ) from report_error
        raise
    finally:
        connection.close()


def _restore_vllm030_fp8_mla_scheduler_controls(native, requested, model, *, max_batched_tokens: int) -> bool:
    """Keep declared GPU scheduling controls in the guarded CPU-only bridge.

    CpuPlatform disables MLA chunking for CPU *model execution*. This process
    constructs only cache metadata and the real CPU Scheduler, like its async
    override below. Restore only the registered FP8 GLM source domain; later
    validation still checks the actual Scheduler against the declaration.
    Neither the model/provider nor normal-capacity acceptance is changed.
    """
    from src.serving.workload.model_config import GLM_BLOCK_FP8_CONFIG_SHA256S

    if (model.config_sha256 not in GLM_BLOCK_FP8_CONFIG_SHA256S
            or model.model_type != "glm_moe_dsa" or model.weight_dtype != "fp8"
            or not requested.enable_chunked_prefill):
        return False
    if type(max_batched_tokens) is not int or max_batched_tokens <= 0:
        raise ValueError("target Scheduler batch-token budget must be positive")
    changed = (not native.scheduler_config.enable_chunked_prefill
        or native.scheduler_config.max_num_batched_tokens != max_batched_tokens)
    native.scheduler_config.enable_chunked_prefill = True
    # CpuPlatform also expands the token budget to max_model_len. Restore
    # the explicitly declared GPU Scheduler budget before constructing it.
    native.scheduler_config.max_num_batched_tokens = max_batched_tokens
    return changed


def _vllm030_pipe_worker(fd: int) -> None:
    """Private SDK process entry; trusted local IPC, no model execution."""
    connection = Connection(fd)
    try:
        startup = connection.recv()
        if not isinstance(startup, tuple) or len(startup) != 9:
            raise ValueError("invalid isolated 0.30 startup packet")
        (requested, source_root, pool_name, max_batch, max_requests, block_size,
         budgets, layouts, role) = startup
        if role not in ("colocated", "prefill", "decode") and worker_role_operator_group(role) != "attention":
            raise ValueError("invalid isolated 0.30 Scheduler role")
        identity = vllm030_source_identity(source_root)
        baseline = _install_cpu_device_guard()
        from src.serving.backend.vllm import VLLMBackend
        from src.serving.kv_cache.vllm_allocation import allocate_vllm030_model_cache, vllm030_worker_cache_specs
        from src.serving.parallelism.parallel_config import ParallelConfig as MidlayerParallel
        from src.serving.schedule.backend_protocol import BackendConfig
        from src.serving.workload.model_config import load_model_config
        from vllm.config import (
            CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig, VllmConfig,
        )
        from vllm.config.kv_transfer import KVTransferConfig
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.structured_output import StructuredOutputManager
        model = load_model_config(requested.model)
        target_model = ModelConfig(
                model=requested.model, skip_tokenizer_init=True, enforce_eager=True,
                dtype=requested.dtype, max_model_len=requested.max_model_len,
                trust_remote_code=requested.trust_remote_code, runner="generate")
        target_parallel = ParallelConfig(tensor_parallel_size=requested.tensor_parallel_size,
            pipeline_parallel_size=requested.pipeline_parallel_size, enable_expert_parallel=requested.enable_expert_parallel)
        speculative = (None if not requested.mtp_num_speculative_tokens else SpeculativeConfig(
            method=requested.speculative_method, num_speculative_tokens=requested.mtp_num_speculative_tokens,
            **({"model": requested.dspark_draft_model} if requested.speculative_method == "dspark" else {}),
            target_model_config=target_model, target_parallel_config=target_parallel,
            parallel_drafting=False, disable_padded_drafter_batch=False,
            use_local_argmax_reduction=False, draft_sample_method="greedy"))
        cache_config = CacheConfig(
            block_size=block_size, gpu_memory_utilization=requested.gpu_memory_utilization,
            cache_dtype="fp8_ds_mla" if model.model_type == "deepseek_v4" else "auto",
            enable_prefix_caching=requested.enable_prefix_caching,
            **({"mamba_block_size": requested.max_model_len} if model.model_id == "moonshotai/Kimi-K3" else {}))
        simple_cpu_bytes = requested.simple_cpu_offload_bytes_per_rank
        if simple_cpu_bytes is None:
            kv_transfer = None
        else:
            kv_transfer = KVTransferConfig(
                kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both",
                kv_buffer_device="cpu", kv_load_failure_policy="fail",
                kv_connector_extra_config={
                    "cpu_bytes_to_use": simple_cpu_bytes * target_parallel.world_size,
                    "cpu_bytes_to_use_per_rank": simple_cpu_bytes,
                    "lazy_offload": False,
                    "kv_offload_backend": "cpu",
                })
        native = VllmConfig(
            model_config=target_model,
            scheduler_config=SchedulerConfig(
                max_model_len=requested.max_model_len, is_encoder_decoder=False,
                max_num_batched_tokens=max_batch, max_num_seqs=max_requests,
                enable_chunked_prefill=requested.enable_chunked_prefill,
                async_scheduling=requested.async_scheduling),
            cache_config=cache_config,
            device_config=DeviceConfig(device="cpu"),
            parallel_config=target_parallel,
            speculative_config=None if requested.speculative_method == "dspark" else speculative,
            kv_transfer_config=kv_transfer,
        )
        if requested.speculative_method == "dspark":
            # The CPU SDK cannot construct the full V2/Triton runner config.
            # Attach the independently native-validated SpeculativeConfig only
            # for allocator/Scheduler control-plane projection, as in its SDK
            # reference test. This does not bind a runner or model execution.
            native.speculative_config = speculative
        if requested.async_scheduling:
            # CpuPlatform disables async because CPU model execution does not
            # need it. This guarded process executes no model; restore the
            # target GPU Scheduler control-plane mode before construction.
            native.scheduler_config.async_scheduling = True
        _restore_vllm030_fp8_mla_scheduler_controls(native, requested, model, max_batched_tokens=max_batch)
        workers, scheduler_cache = allocate_vllm030_model_cache(
            native, model, MidlayerParallel(tensor_parallel_size=requested.tensor_parallel_size,
                pipeline_parallel_size=requested.pipeline_parallel_size,
                enable_expert_parallel=requested.enable_expert_parallel),
            budgets, source_root=source_root, supported_layouts_by_rank=layouts)
        native.cache_config.num_gpu_blocks = scheduler_cache.num_blocks
        scheduler_block_size = (requested.max_model_len if model.model_id == "moonshotai/Kimi-K3"
                                else native.cache_config.block_size)
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        scheduler_class = AsyncScheduler if requested.async_scheduling else Scheduler
        scheduler = scheduler_class(native, scheduler_cache, StructuredOutputManager(native),
                              block_size=scheduler_block_size)
        # A plain FullAttentionSpec is per layer even when its group owns
        # many layers. UniformTypeKVCacheSpecs already sums its members.
        # Count source leaves once rather than treating both group forms alike.
        all_worker_specs = vllm030_worker_cache_specs(native, model)
        page_bytes = max(sum(spec.page_size_bytes for spec in specs.values()) for specs in all_worker_specs)
        if model.model_type == "deepseek_v4":
            rate = model.kv_bytes_per_token_per_parallel_rank(requested.tensor_parallel_size, 1)
            reporting_bytes_per_token = -(-rate.numerator // rate.denominator)
        elif model.model_id == "moonshotai/Kimi-K3":
            reporting_bytes_per_token = 0
        else:
            reporting_bytes_per_token = page_bytes // scheduler_block_size
        effective = BackendConfig(
            pool_name, role, scheduler.max_num_scheduled_tokens,
            native.scheduler_config.max_num_seqs, scheduler_cache.num_blocks,
            kv_block_size_tokens=scheduler_block_size,
            kv_bytes_per_token=reporting_bytes_per_token,
            scheduler_pipeline_depth=2 if requested.async_scheduling else 1)
        backend = VLLMBackend.from_vllm030_scheduler(
            effective, requested, scheduler, source_root=source_root,
            worker_kv_configs=tuple(workers))
        control = backend._control_plane
        control.runtime_info.update(
            source_identity=identity, worker_pid=os.getpid(),
            requested_async_scheduling=requested.async_scheduling,
            effective_async_scheduling=native.scheduler_config.async_scheduling,
            scheduler_class=type(scheduler).__name__,
            bootstrap="explicit SDK Python subprocess; guarded CpuPlatform",
            device_guard="CpuPlatform forced; CUDA availability hidden; CUDA/NVML APIs blocked",
            requested_max_num_batched_tokens=max_batch, requested_cache_block_size=block_size,
            available_memory_by_rank=budgets, supported_layouts_by_rank=layouts,
            capacity_input_source="caller-declared byte budgets, not measured/normal available KV",
            layout_capabilities_source="caller declaration, not worker report",
            gpu_worker_processes_started=0, weights_loaded=False, model_executed=False,
            full_speculative_vllm_config_constructed=(requested.speculative_method != "dspark"),
            explicit_speculative_control_plane_projection=(requested.speculative_method == "dspark"),
            kv_connector=(None if simple_cpu_bytes is None else "SimpleCPUOffloadConnector"),
            kv_connector_role=(None if simple_cpu_bytes is None else "kv_both"),
            simple_cpu_offload_bytes_per_rank=simple_cpu_bytes,
            simple_cpu_offload_mode=(None if simple_cpu_bytes is None else "eager_cpu"),
        )
        _verify_no_accelerator_access(baseline)
        connection.send(("ready", effective, control.runtime_info))
        _serve_scheduler_commands(connection, control, baseline)
    except EOFError:
        return
    except BaseException as exc:
        connection.send(("error", type(exc).__name__, str(exc), traceback.format_exc()))
        raise
    finally:
        connection.close()


def _serve_scheduler_commands(connection: Connection, control_plane: Any,
                              baseline_libraries: frozenset[str]) -> None:
    pending_steps: deque[Any] = deque()
    commands_since_device_audit = 0
    while True:
        command = connection.recv()
        if not isinstance(command, tuple) or not command:
            raise TypeError("vLLM scheduler command must be a non-empty tuple")
        name = command[0]
        if name == "add_request":
            if len(command) != 2:
                raise ValueError("add_request command has invalid arity")
            control_plane.add_request(command[1])
            result = None
        elif name == "add_requests":
            if len(command) != 2 or not isinstance(command[1], tuple):
                raise ValueError("add_requests command is invalid")
            control_plane.add_requests(command[1])
            result = None
        elif name == "request_queue_snapshot":
            if len(command) != 1:
                raise ValueError("request_queue_snapshot command has invalid arity")
            result = control_plane.request_queue_snapshot()
        elif name == "plan_step":
            if len(command) != 2:
                raise ValueError("plan_step command has invalid arity")
            result = control_plane.plan_step(command[1])
            if result is not None:
                pending_steps.append(result)
        elif name == "enable_modeled_simple_cpu_offload":
            if len(command) != 1 or pending_steps:
                raise ValueError("SimpleCPU modeling requires an unused Scheduler")
            control_plane.enable_modeled_simple_cpu_offload()
            result = None
        elif name == "take_modeled_simple_cpu_releases":
            if len(command) != 1 or pending_steps:
                raise ValueError("SimpleCPU release query requires no pending steps")
            result = control_plane.take_modeled_simple_cpu_releases()
        elif name == "probe_simple_cpu_connector":
            if len(command) != 2 or pending_steps:
                raise ValueError("SimpleCPU probe requires requests and no pending steps")
            result = control_plane.probe_simple_cpu_connector(command[1])
        elif name == "drain_steps":
            if len(command) != 2:
                raise ValueError("drain_steps command has invalid arity")
            result = control_plane.drain_steps(command[1])
            steps_by_id = {step.schedule.step_id: step
                for step in (*pending_steps, *result[0])}
            pending_ids = control_plane.pending_step_ids()
            if len(steps_by_id) != len(pending_steps) + len(result[0]):
                raise RuntimeError("vLLM drain returned a duplicate step ID")
            pending_steps.clear()
            pending_steps.extend(steps_by_id[step_id] for step_id in pending_ids)
        elif name == "complete_step":
            if len(command) != 6:
                raise ValueError("complete_step command has invalid arity")
            if not pending_steps:
                raise RuntimeError("no pending vLLM step to complete")
            if pending_steps[0].schedule.step_id != command[1]:
                raise ValueError(
                    "vLLM completion step ID does not match the oldest "
                    "pending step")
            result = control_plane.complete_step(
                pending_steps[0],
                command[2],
                execute_model=command[3],
                execute_transfers=command[4],
                sample_tokens=command[5],
            )
            pending_steps.popleft()
        elif name in ("retain_pd_kv", "release_pd_kv", "reserve_pd_destination", "try_reserve_pd_destination"):
            if len(command) != 2:
                raise ValueError("PD cache command has invalid arity")
            if name == "retain_pd_kv":
                result = control_plane.retain_pd_kv(command[1])
            elif name == "release_pd_kv":
                result = control_plane.release_pd_kv(command[1])
            elif name == "reserve_pd_destination":
                result = control_plane.reserve_pd_destination(command[1])
            else:
                result = control_plane.try_reserve_pd_destination(command[1])
        elif name == "has_work":
            if len(command) != 1:
                raise ValueError("has_work command has invalid arity")
            result = control_plane.has_work()
        elif name == "measure_scheduler_service":
            if len(command) not in (5, 6):
                raise ValueError(
                    "Scheduler service measurement command has invalid arity")
            if pending_steps:
                raise RuntimeError(
                    "Scheduler service measurement requires no pending steps")
            arguments = dict(sample_count=command[1], requests_per_batch=command[2],
                prompt_tokens=command[3], output_tokens=command[4])
            include_step_records = len(command) == 6
            if include_step_records:
                if command[5] is not True or control_plane.scheduler_config.scheduler_version != "0.30.0":
                    raise ValueError("source-bound Scheduler step measurements require explicit0.30 opt-in")
                import inspect
                scheduler_type = type(control_plane.scheduler)
                source_root = Path(control_plane.runtime_info["source_identity"]["source_root"]).resolve()
                source_files = []
                for parent in scheduler_type.__mro__:
                    if not parent.__module__.startswith("vllm.v1.core.sched."):
                        continue
                    source_file = inspect.getsourcefile(parent)
                    if source_file is None:
                        raise RuntimeError("actual Scheduler class has no source file")
                    source_path = Path(source_file).resolve()
                    relative = source_path.relative_to(source_root)
                    source_files.append({"class": parent.__module__ + "." + parent.__name__,
                        "path": str(source_path), "relative_path": str(relative),
                        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()})
                if not source_files:
                    raise RuntimeError("actual Scheduler class source is outside the pinned checkout")
                source = {"source_identity": control_plane.runtime_info["source_identity"],
                    "scheduler_class": scheduler_type.__name__,
                    "scheduler_constructor": scheduler_type.__module__ + "." + scheduler_type.__name__,
                    "scheduler_source_files": source_files, "worker_pid": os.getpid(),
                    "control_workload": {"platform": "cpu",
                        "enable_prefix_caching": control_plane.scheduler.cache_config.enable_prefix_caching,
                        "log_stats": control_plane.scheduler.log_stats,
                        "admission_batch_sizes": [command[2]]}}
                arguments["include_step_records"] = True
            result = control_plane.measure_scheduler_service(**arguments)
            if include_step_records:
                for entry in source_files:
                    if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
                        raise RuntimeError("Scheduler source changed during the measurement")
                result = {"samples": result, "source": source}
        elif name == "abort_observation":
            if len(command) != 1:
                raise ValueError("abort_observation command has invalid arity")
            _verify_no_accelerator_access(baseline_libraries)
            connection.send(("ok", None))
            break
        elif name == "close":
            if len(command) != 1:
                raise ValueError("close command has invalid arity")
            if pending_steps:
                raise RuntimeError(
                    "cannot close vLLM Scheduler with pending steps")
            if control_plane.scheduler_config.scheduler_version == "0.30.0":
                control_plane.close()
            else:
                control_plane.assert_mtp_contract_exhausted()
            _verify_no_accelerator_access(baseline_libraries)
            connection.send(("ok", None))
            break
        else:
            raise ValueError(f"unknown vLLM scheduler command: {name!r}")
        commands_since_device_audit += 1
        if commands_since_device_audit == _DEVICE_AUDIT_COMMAND_INTERVAL:
            _verify_no_accelerator_access(baseline_libraries)
            commands_since_device_audit = 0
        else:
            _verify_no_cuda_context()
        connection.send(("ok", result))


def _finalize_worker(
    connection: Connection,
    process: Any,
    temporary_model_directory: tempfile.TemporaryDirectory[str] | None,
) -> None:
    if process.is_alive():
        try:
            connection.send(("close",))
        except (BrokenPipeError, EOFError, OSError):
            process.terminate()
    try:
        connection.close()
    finally:
        process.join(_SHUTDOWN_TIMEOUT_S)
        if process.is_alive():
            process.terminate()
            process.join(_SHUTDOWN_TIMEOUT_S)
        if temporary_model_directory is not None:
            temporary_model_directory.cleanup()
