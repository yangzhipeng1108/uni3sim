"""Real SGLang scheduler construction via Dummy+Patch (no GPU device).

Two construction modes share one patching core:

* Non-spec: ``ServerArgs(speculative_algorithm=None)`` — the scheduler
  runs ordinary prefill/decode with the real RadixCache and
  PrefillAdder budget logic.

* EAGLE/NEXTN (MTP): ``ServerArgs(speculative_algorithm="EAGLE",
  speculative_draft_model_path=<dummy>)`` — a real ``EAGLEWorker`` is
  constructed (it loads the same tiny Llama as the draft).  The draft
  worker is wrapped so that its model loading is skipped: the worker
  object is real (its allocation/verify bookkeeping runs), only
  ``ModelRunner.load_model`` is patched to keep tensors on meta device.
  Per-step verify results are fabricated with acceptance lengths drawn
  from an explicit ``MTPAcceptanceAssumption`` (zero fitting: the
  acceptance distribution is declared, never regressed from durations).

All SGLang imports are lazy; midlayer stays importable without SGLang.

Global patch lifecycle
-----------------------
Every monkey patch this module installs is recorded in
:func:`_push_global_patch` and restored by :func:`restore_global_patches`.
The restoration runs on the normal exit path
(:func:`destroy_native_sglang_scheduler`) and on the exception path
(``try/finally`` inside :func:`build_native_sglang_scheduler`), so a
failed EAGLE construction never leaks patches into the process.  The
caller-visible contract is::

    scheduler = build_native_sglang_scheduler(...)   # pushes patches
    try:
        ...                                          # use scheduler
    finally:
        destroy_native_sglang_scheduler(scheduler)     # restores + teardown

After destroy, a fresh ordinary or spec construction observes pristine
sglang classes (validated by tests).
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.serving.backend.mtp_acceptance_assumption import (
    MTPAcceptanceAssumption,
)


_DUMMY_MODEL_PATH: Optional[str] = None

# (module_object, attribute_name, original_value) in installation order.
_GLOBAL_PATCH_STACK: list[tuple[Any, str, Any]] = []


class _OwnedEnvPatch:
    """One owned environment mutation with explicit restore semantics.

    ``os.environ`` has no module attribute to snapshot, so environment
    entries travel through this record instead of the (module, name,
    original) triples the attribute patch stack uses.  A variable this
    process owned (set here, possibly overwriting a prior value) is
    deleted on restore — a later construction re-owns it.  A variable
    that predates the first construction (inherited from the parent
    environment) is restored to that original value instead of being
    deleted, so a foreign setting is never destroyed.
    """

    __slots__ = ("name", "prior", "had_prior")

    def __init__(self, name: str):
        self.name = name
        self.had_prior = name in os.environ
        self.prior = os.environ[name] if self.had_prior else None


_ENV_PATCHES: list[_OwnedEnvPatch] = []


def _own_environment_variable(name: str, value: str) -> None:
    """Record ownership of ``os.environ[name] = value`` and apply it.

    The first call for ``name`` snapshots the pre-construction state;
    later calls only update the value (the original snapshot is kept so
    restore returns to the true pre-construction state, not to an
    intermediate construction's value).
    """
    if not any(entry.name == name for entry in _ENV_PATCHES):
        _ENV_PATCHES.append(_OwnedEnvPatch(name))
    os.environ[name] = value


def restore_owned_environment() -> None:
    """Undo every environment variable this module owns (LIFO by name)."""
    while _ENV_PATCHES:
        entry = _ENV_PATCHES.pop()
        if entry.had_prior:
            os.environ[entry.name] = entry.prior
        else:
            del os.environ[entry.name]


def _push_global_patch(module: Any, name: str, replacement: Any) -> None:
    """Install ``module.name = replacement`` and record the original."""
    original = getattr(module, name)
    _GLOBAL_PATCH_STACK.append((module, name, original))
    setattr(module, name, replacement)


def restore_global_patches() -> None:
    """Undo every patch recorded by :func:`_push_global_patch` (LIFO)."""
    while _GLOBAL_PATCH_STACK:
        module, name, original = _GLOBAL_PATCH_STACK.pop()
        setattr(module, name, original)
    restore_owned_environment()


def _ensure_dummy_model() -> str:
    """Create (once) a tiny 1-layer Llama that satisfies load_model()."""
    global _DUMMY_MODEL_PATH
    if _DUMMY_MODEL_PATH is not None:
        return _DUMMY_MODEL_PATH
    import json
    import tempfile
    from pathlib import Path

    path = Path(tempfile.gettempdir()) / "sglang_test_model"
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "vocab_size": 100,
        "max_position_embeddings": 512,
        "torch_dtype": "float16",
    }
    (path / "config.json").write_text(json.dumps(config))
    if not (path / "model.safetensors").exists():
        # One tensor: model.embed_tokens.weight (100x64 fp16).  The real
        # safetensors writer is the only supported path; a failure here
        # propagates to the construction exception handler (no fabricated
        # fallback blob).
        import torch
        from safetensors.torch import save_file

        save_file({"model.embed_tokens.weight":
                   torch.zeros(100, 64, dtype=torch.float16)},
                  str(path / "model.safetensors"))
    _DUMMY_MODEL_PATH = str(path)
    return _DUMMY_MODEL_PATH


def _patch_allocator(allocator: Any, size: int) -> None:
    """Resize a TokenToKVPoolAllocator (or hybrid wrapper) to ``size``."""
    if hasattr(allocator, "full_attn_allocator"):
        _patch_allocator(allocator.full_attn_allocator, size)
        if hasattr(allocator, "swa_attn_allocator"):
            _patch_allocator(allocator.swa_attn_allocator, size)
        allocator.clear()
        return
    allocator.size = size
    if hasattr(allocator, "num_pages"):
        allocator.num_pages = size // allocator.page_size
    allocator.clear()


def _noop_sender() -> Any:
    class _NoopSender:
        def send_output(self, *args, **kwargs):
            return None
    return _NoopSender()


def _patch_scheduler_process_io(scheduler_cls: Any) -> None:
    """Use the bridge's in-process endpoints from construction onward.

    The bridge drives batches directly and never runs SGLang's IPC loop.
    Replacing sockets only after construction leaves ZeroMQ contexts for
    cyclic GC, whose default infinite linger can block teardown forever.
    The real server watchdog also assumes GPU forwards and retains the
    scheduler through an unbounded daemon thread. Neither service belongs
    to this in-process, virtual-time scheduler.
    """
    def _init_process_io(self, server_args, port_args):
        self.idle_sleeper = None
        self.recv_from_tokenizer = None
        self.recv_from_rpc = None
        self.send_to_tokenizer = _noop_sender()
        self.send_to_detokenizer = _noop_sender()

    def _watchdog_unused(self):
        return None

    _push_global_patch(scheduler_cls, "init_sockets", _init_process_io)
    _push_global_patch(scheduler_cls, "watchdog_thread", _watchdog_unused)


def _patch_eagle_cpu_weight_sharing() -> None:
    """Keep Llama's shared Parameters without CUDA allocator/sync calls.

    SGLang 0.5.4 LlamaForCausalLM.set_embed_and_head assigns these two
    Parameters, then unconditionally flushes/synchronizes CUDA. The dummy
    target and draft use CPU tensors, whose assignments are synchronous.
    """
    from sglang.srt.models.llama import LlamaForCausalLM

    def _set_embed_and_head_cpu(self, embed, head):
        if embed.device.type != "cpu" or head.device.type != "cpu":
            raise ValueError("CPU SGLang bridge requires CPU embedding/head weights")
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    _push_global_patch(LlamaForCausalLM, "set_embed_and_head", _set_embed_and_head_cpu)


def _patch_schedule_policy_clip() -> None:
    """Keep admission from reserving output budgets the dummy cannot hold.

    ``schedule_policy.CLIP_MAX_NEW_TOKENS`` is read at import time, so the
    env knob is useless once sglang is loaded.  Each running request's
    estimated decode reservation (``min(remaining_output, CLIP) *
    new_token_ratio``) lands in ``rem_total_token_offset``; with the
    module default of 4096 the first admission already exceeds the
    patched pool and the PrefillAdder reports NO_TOKEN, starving the
    schedule.  Clipping to a token keeps the reservation mechanism (it
    still scales with batch size and ``new_token_ratio``) while
    reflecting that the replayed bridge fabricates generation anyway.
    """
    from sglang.srt.managers import schedule_policy

    _push_global_patch(schedule_policy, "CLIP_MAX_NEW_TOKENS", 64)


def _destroy_distributed_state() -> None:
    """Tear down TP/distributed globals so another Scheduler can build."""
    import gc

    from sglang.srt.distributed import parallel_state

    parallel_state.destroy_model_parallel()
    # 0.5.4's destroy_model_parallel only clears TP/PP; reset the MoE
    # group globals too so initialize_model_parallel can run again.
    parallel_state._MOE_EP = None
    parallel_state._MOE_TP = None
    parallel_state.destroy_distributed_environment()
    gc.collect()


def _patch_eagle_draft_worker(eagle_worker: Any) -> None:
    """Skip the draft model's real weight load; keep the worker real.

    The EAGLEWorker object must exist (scheduler consults it for
    spec counters), but its draft ModelRunner must not load weights.
    We replace ``load_model`` on the *class* of the draft runner so
    the instance keeps every other attribute.
    """
    runner = eagle_worker.model_runner
    cls = type(runner)

    def _load_model_skipped(self):
        self.model = None
        return

    _push_global_patch(cls, "load_model", _load_model_skipped)


def _patch_eagle_draft_alloc_len(scheduler: Any) -> None:
    """Own the ``ALLOC_LEN_PER_DECODE`` constant for the construction cycle.

    The 0.5.4 non-v2 EAGLE merge path reads this class constant at
    ``EagleDraftInput`` construction time; worker-v2 normally sets it, but
    the Dummy+Patch bridge drives the non-v2 processor.  Setting it here
    (instead of inside ``complete_step``) keeps every global mutation
    inside the tracked construction cycle, so destroy restores the class
    to its pristine None for the next ordinary or spec construction.
    """
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    draft_num = scheduler.server_args.speculative_num_draft_tokens
    _push_global_patch(EagleDraftInput, "ALLOC_LEN_PER_DECODE", draft_num)


def _patch_kv_copy_warmup() -> None:
    """Skip the triton KV-copy warmup on CPU engines.

    ``MHATokenToKVPool._init_kv_copy_and_warmup`` launches a triton
    kernel unconditionally when speculative decoding is enabled, and
    triton rejects CPU tensors.  The warmup only validates the copy
    pipeline; skipping it leaves the allocation logic untouched.
    """
    from sglang.srt.mem_cache import memory_pool

    def _warmup_skipped(self):
        self._kv_copy_config = None

    _push_global_patch(
        memory_pool.MHATokenToKVPool,
        "_init_kv_copy_and_warmup",
        _warmup_skipped,
    )


def _patch_triton_dispatch() -> None:
    """Route triton JIT helpers to their torch fallbacks on CPU engines.

    0.5.4 dispatches on the *configured* attention backend name, not the
    device: with ``attention_backend="triton"`` selected only to satisfy
    the EAGLE factory, ``write_cache_indices`` and ``get_last_loc`` would
    launch triton kernels against CPU tensors.  Force the torch-native
    paths; the allocation semantics are identical.
    """
    from sglang.srt.mem_cache import common

    def _write_cache_indices_torch(
        out_cache_loc,
        req_pool_indices_tensor,
        req_pool_indices_cpu,
        prefix_lens_tensor,
        prefix_lens_cpu,
        seq_lens_tensor,
        seq_lens_cpu,
        extend_lens_tensor,
        extend_lens_cpu,
        prefix_tensors,
        req_to_token_pool,
    ):
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()
            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)), prefix_tensors[i])
            req_to_token_pool.write(
                (req_idx, slice(prefix_len, seq_len)),
                out_cache_loc[pt: pt + extend_len])
            pt += extend_len

    _push_global_patch(common, "write_cache_indices", _write_cache_indices_torch)
    _push_global_patch(common, "get_last_loc", common.get_last_loc_torch)


def _patch_spec_utils_assignment() -> None:
    """Replace sglang's triton slot-assignment with a torch equivalent.

    ``speculative.spec_utils.assign_req_to_token_pool_func`` branches on
    an import-time CUDA probe (true on GPU hosts even when the scheduler
    runs on CPU tensors) and launches a triton kernel.  The torch path
    below implements the same row-wise scatter used by the kernel.
    Out-of-range windows are rejected: silently clamping would corrupt
    the req-to-token table or lose evidence about allocator behaviour.
    """
    from sglang.srt.speculative import spec_utils

    def _assign_torch(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        batch_size,
    ):
        del batch_size
        pool_len = req_to_token.shape[1]
        remaining = out_cache_loc
        for index in range(req_pool_indices.shape[0]):
            req_index = int(req_pool_indices[index].item())
            start = int(start_offset[index].item())
            end = int(end_offset[index].item())
            if not 0 <= start < pool_len or not 0 <= end <= pool_len:
                raise ValueError(
                    "assign_req_to_token_pool window out of range: "
                    f"req={req_index} [{start}, {end}) pool_len={pool_len}")
            length = end - start
            if length <= 0:
                continue
            slots = remaining[:length].to(req_to_token.dtype)
            req_to_token[req_index, start:end] = slots
            remaining = remaining[length:]

    _push_global_patch(spec_utils, "assign_req_to_token_pool_func", _assign_torch)
    # eagle_info imports the symbol by name at its own import time, so
    # repatching spec_utils alone leaves an earlier import holding the
    # triton-dispatching original.  Rebind the name on every module that
    # may already have imported it.
    from sglang.srt.speculative import eagle_info

    _push_global_patch(
        eagle_info, "assign_req_to_token_pool_func", _assign_torch)


def patch_scheduler_capacities(
    scheduler: Any,
    *,
    max_total_num_tokens: int,
    max_running_requests: int,
    chunked_prefill_size: int,
    max_model_len: int,
    max_output_headroom: int = 4096,
) -> None:
    """Give the dummy-model scheduler the target model's token budget."""
    worker = scheduler.tp_worker
    runner = worker.model_runner

    _patch_allocator(runner.token_to_kv_pool_allocator, max_total_num_tokens)

    req_pool = runner.req_to_token_pool
    req_pool.size = max_running_requests
    req_pool.free_slots = list(range(1, max_running_requests + 1))
    # The dummy model's context (512) sizes the req_to_token rows; real
    # prompts/sequences exceed it.  Rows are fixed-stride per request, so
    # oversized rows only waste pool memory while keeping every indexed
    # write in range.
    table_len = max_model_len + max_output_headroom + chunked_prefill_size
    if req_pool.req_to_token.shape[1] < table_len:
        import torch as _torch

        req_pool.req_to_token = _torch.zeros(
            (req_pool.req_to_token.shape[0], table_len),
            dtype=req_pool.req_to_token.dtype,
            device=req_pool.req_to_token.device,
        )
    req_pool.max_context_len = table_len

    scheduler.max_total_num_tokens = max_total_num_tokens
    scheduler.max_prefill_tokens = chunked_prefill_size
    scheduler.max_running_requests = max_running_requests
    worker.max_total_num_tokens = max_total_num_tokens
    worker.max_prefill_tokens = chunked_prefill_size
    worker.max_running_requests = max_running_requests
    # The admission budget multiplies every request's total-token estimate
    # by ``new_token_ratio`` (init 0.7, floored at 0.4 after retractions).
    # That factor guards real decode-token consumption, which the replayed
    # bridge fabricates; pinning it to 1 keeps the PrefillAdder from
    # rejecting oversized prompts the patched pool can actually hold.
    scheduler.init_new_token_ratio = 1.0
    scheduler.min_new_token_ratio = 1.0
    scheduler.new_token_ratio = 1.0
    # init_req_max_new_tokens clamps against the worker's pool-derived
    # max_req_len (min(context_len, max_total_num_tokens) - 1); the dummy
    # model's tiny pool would truncate every real request's output budget.
    worker.max_req_len = max_model_len + max_output_headroom
    worker.max_req_input_len = max_model_len
    scheduler.max_req_len = worker.max_req_len
    scheduler.max_req_input_len = worker.max_req_input_len

    scheduler.send_to_detokenizer = _noop_sender()
    scheduler.send_to_tokenizer = _noop_sender()
    scheduler.recv_from_tokenizer = None
    scheduler.recv_from_rpc = None

    draft = getattr(scheduler, "draft_worker", None)
    if draft is not None:
        draft.max_total_num_tokens = max_total_num_tokens
        draft.max_running_requests = max_running_requests


def build_native_sglang_scheduler(
    *,
    max_total_num_tokens: int,
    max_running_requests: int,
    chunked_prefill_size: int,
    max_model_len: int,
    mtp_num_speculative_tokens: int = 0,
    mtp_acceptance_assumption: Optional[MTPAcceptanceAssumption] = None,
) -> Any:
    """Construct a real SGLang Scheduler patched to the target capacity.

    Returns the scheduler; the caller drives it through
    ``SGLangBackend.from_scheduler`` and **must** pair the call with
    :func:`destroy_native_sglang_scheduler` (try/finally) so global
    patches and distributed state are torn down on both exit paths.
    When ``mtp_num_speculative_tokens`` is positive, the scheduler runs
    EAGLE (NEXTN) with a real EAGLEWorker whose draft weights are
    skipped.
    """
    if mtp_num_speculative_tokens > 0 and mtp_acceptance_assumption is None:
        raise ValueError("MTP requires an explicit MTPAcceptanceAssumption")
    try:
        _own_environment_variable("SGLANG_USE_CPU_ENGINE", "1")
        from sglang.srt.server_args import PortArgs, ServerArgs

        model_path = _ensure_dummy_model()
        kwargs: dict = dict(
            model_path=model_path,
            device="cpu",
            dtype="float16",
            trust_remote_code=False,
            skip_tokenizer_init=True,
            disable_radix_cache=False,
            chunked_prefill_size=chunked_prefill_size,
            max_total_tokens=max_total_num_tokens,
            max_running_requests=max_running_requests,
            max_prefill_tokens=chunked_prefill_size,
            attention_backend="torch_native",
            cuda_graph_max_bs=1,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            stream_interval=10**9,
            enable_metrics=False,
            log_level="warning",
        )
        if mtp_num_speculative_tokens > 0:
            _patch_kv_copy_warmup()
            _patch_triton_dispatch()
            _patch_spec_utils_assignment()
            _patch_eagle_cpu_weight_sharing()
            kwargs.update(
                speculative_algorithm="EAGLE",
                speculative_draft_model_path=model_path,
                speculative_num_steps=mtp_num_speculative_tokens,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=mtp_num_speculative_tokens + 1,
                # The draft attention backend only runs during real draft
                # forwards; our bridge fabricates verify inputs instead, so
                # the triton backend is never exercised at runtime.  Setting
                # it satisfies the EAGLE factory (torch_native has no entry).
                attention_backend="triton",
                decode_attention_backend="triton",
            )
        _patch_schedule_policy_clip()
        server_args = ServerArgs(**kwargs)
        port_args = PortArgs.init_new(server_args)
        from sglang.srt.managers.scheduler import Scheduler

        _patch_scheduler_process_io(Scheduler)
        # SGLang borrows vLLM custom-op classes while loading its CPU dummy
        # model. Give those classes an explicit CPU/native configuration
        # instead of their default GPU-device autodetection. The public
        # context restores any caller-owned vLLM configuration on both paths.
        from vllm.config import (
            CompilationConfig,
            DeviceConfig,
            VllmConfig,
            get_cached_compilation_config,
            set_current_vllm_config,
        )

        cpu_config = VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            compilation_config=CompilationConfig(custom_ops=["none"]),
        )
        with set_current_vllm_config(cpu_config):
            # vLLM 0.11 clears this cache on context exit, not entry; a
            # caller may already have cached its own compilation config.
            get_cached_compilation_config.cache_clear()
            scheduler = Scheduler(
                server_args,
                port_args,
                gpu_id=0,
                tp_rank=0,
                moe_ep_rank=0,
                pp_rank=0,
                dp_rank=None,
            )
        if mtp_num_speculative_tokens > 0:
            _patch_eagle_draft_worker(scheduler.draft_worker)
            _patch_eagle_draft_alloc_len(scheduler)
        patch_scheduler_capacities(
            scheduler,
            max_total_num_tokens=max_total_num_tokens,
            max_running_requests=max_running_requests,
            chunked_prefill_size=chunked_prefill_size,
            max_model_len=max_model_len,
            # Covers the largest per-request output budget the engine may
            # attach on top of the model context; the worker's constructor
            # derives max_req_len from the dummy pool instead.
            max_output_headroom=max(4096, chunked_prefill_size),
        )
    except BaseException:
        # A failed construction must not leak monkey patches or partial
        # distributed groups into the process.
        restore_global_patches()
        _destroy_distributed_state()
        raise
    return scheduler


def destroy_native_sglang_scheduler(scheduler: Any) -> None:
    """Destroy a scheduler built by :func:`build_native_sglang_scheduler`.

    Restores every monkey patch installed during construction (in
    reverse order) and tears down the distributed state, on both the
    normal and the exception path of the caller's finally block.
    """
    try:
        del scheduler
    finally:
        restore_global_patches()
        _destroy_distributed_state()
