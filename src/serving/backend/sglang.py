"""SGLang scheduler/RadixCache bridge with a short-circuited model runner.

SGLang does not expose a worker-free ``Scheduler`` constructor: constructing
one also constructs ``TpModelWorker`` and loads the model.  Consequently this
module deliberately does not offer a misleading ``from_model`` API.  The
production bridge is :meth:`SGLangBackend.from_scheduler`, which takes an
already initialized, idle SGLang scheduler, reuses its scheduling and cache
state, and replaces only model execution/sampling with a CPU dummy result.

All SGLang imports are lazy.  Midlayer therefore remains importable when
SGLang is not installed, and embedding applications can still supply an
already-normalized control plane through the regular constructor.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.request_gen.request_stream import RequestSpec
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.schedule_step import BackendStep, ScheduleStep, ScheduledRequest


def _sglang_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"SGLang {field} must be an integer")
    return value


def _sglang_request_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("SGLang request ID must be a non-empty string")
    return value


class SGLangBackend:
    """Framework-independent facade over an SGLang control plane."""

    def __init__(self, config: BackendConfig, control_plane: Any) -> None:
        self.config = config
        self._control_plane = control_plane

    @classmethod
    def from_scheduler(
        cls,
        config: BackendConfig,
        scheduler: Any,
        *,
        dummy_token_id: int = 42,
        mtp_acceptance_assumption: Any = None,
        price_verify_steps: bool = True,
    ) -> "SGLangBackend":
        """Bridge an initialized SGLang ``Scheduler`` without running a model.

        The scheduler must be idle and must not have its event loop running.
        SGLang overlap scheduling and native disaggregation modes are rejected
        because their output pipelines require worker-owned GPU state or a
        live KV-transfer process.  Colocated, non-overlap generation uses the
        real ``ScheduleBatch`` and RadixCache decisions; EAGLE/NEXTN (MTP)
        runs with fabricated verify results whose acceptance lengths follow
        the explicit ``mtp_acceptance_assumption``.

        ``price_verify_steps`` selects how speculative decode steps reach
        the pricing layer.  ``True`` (the only supported production mode)
        prices the declared target-verify work through the ordinary dense
        forward on the scheduled draft-token stream, attaching an explicit
        synthetic-token embedding stream whose source label declares the
        fixture identity.  ``False`` was a legacy probe-only mode that
        attempted to withhold the priced view; it is not expressible in
        the engine contract without either violating token conservation
        or zeroing MTP evidence, and it is no longer accepted — callers
        that cannot price MTP-enabled rows must report the combination as
        unsupported instead.  Verify rows are never downgraded to
        ordinary decode and MTP evidence is never zeroed.
        """

        if isinstance(dummy_token_id, bool) or not isinstance(
                dummy_token_id, int):
            raise TypeError("SGLang dummy_token_id must be an integer")
        if not isinstance(price_verify_steps, bool):
            raise TypeError("SGLang price_verify_steps must be a boolean")
        return cls(
            config,
            _NativeSGLangControlPlane(
                config,
                scheduler,
                dummy_token_id=dummy_token_id,
                mtp_acceptance_assumption=mtp_acceptance_assumption,
                price_verify_steps=price_verify_steps,
            ),
        )

    def add_request(self, request: RequestSpec) -> None:
        if request.output_tokens == 0:
            raise NotImplementedError(
                "SGLang zero-output request completion is not implemented")
        if request.prefix_hit_tokens != 0:
            raise NotImplementedError(
                "SGLang trace-authoritative prefix injection is not implemented")
        self._control_plane.add_request(request)

    def plan_step(self, virtual_time_s: float):
        return self._control_plane.plan_step(virtual_time_s)

    def complete_step(self, step: BackendStep, virtual_time_s: float) -> float:
        self._control_plane.complete_step(
            step,
            virtual_time_s,
            execute_model=False,
            execute_transfers=False,
            sample_tokens=False,
        )

    def has_work(self) -> bool:
        result = self._control_plane.has_work()
        if not isinstance(result, bool):
            raise TypeError("SGLang control_plane.has_work() must return boolean")
        return result

    def close(self) -> None:
        if not hasattr(self._control_plane, "close"):
            return
        self._control_plane.close()


class _NativeSGLangControlPlane:
    def __init__(
        self,
        config: BackendConfig,
        scheduler: Any,
        *,
        dummy_token_id: int,
        mtp_acceptance_assumption: Any = None,
        price_verify_steps: bool = True,
    ) -> float:
        _validate_native_scheduler(scheduler)
        if not isinstance(price_verify_steps, bool):
            raise TypeError("SGLang price_verify_steps must be a boolean")
        if not price_verify_steps:
            raise NotImplementedError(
                "SGLangBackend no longer supports price_verify_steps=False: "
                "withholding the priced view while preserving real "
                "accepted/emitted progress is not expressible in the engine "
                "contract (a nonterminal step must emit its full raw model "
                "output, and raw output is tied to the accepted count). "
                "Callers that cannot price MTP-enabled rows must report the "
                "combination as unsupported instead.")
        self.config = config
        self.scheduler = scheduler
        self.dummy_token_id = dummy_token_id
        self._mtp_assumption = mtp_acceptance_assumption
        self._price_verify_steps = price_verify_steps
        self._requests: dict[str, RequestSpec] = {}
        self._native_requests: dict[str, Any] = {}
        self._first_schedule: set[str] = set()
        self._pending: tuple[BackendStep, Any] | None = None
        self._step_index = 0
        self._pending_accepts: list[int] | None = None

    def add_request(self, request: RequestSpec) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"duplicate SGLang request ID {request.request_id!r}")
        if request.output_tokens == 0:
            raise NotImplementedError(
                "SGLang zero-output request completion is not implemented")
        if request.prefix_hit_tokens != 0:
            raise NotImplementedError(
                "SGLang trace-authoritative prefix injection is not implemented")
        output_limit = 1 if self.config.is_prefill_only else request.output_tokens
        native = _make_native_request(
            self.scheduler,
            request,
            dummy_token_id=self.dummy_token_id,
            max_output_tokens=output_limit,
        )
        self.scheduler._add_request_to_queue(native)
        if native.finished():
            raise RuntimeError(
                f"SGLang rejected request {request.request_id!r} during admission"
            )
        queued_limit = self.scheduler.max_queued_requests
        if queued_limit is not None and native not in self.scheduler.waiting_queue:
            raise RuntimeError(
                f"SGLang dropped request {request.request_id!r}: waiting-queue "
                f"limit {queued_limit} reached"
            )
        self._requests[request.request_id] = request
        self._native_requests[request.request_id] = native

    def plan_step(self, virtual_time_s: float) -> BackendStep | None:
        if self._pending is not None:
            raise RuntimeError("SGLang step must be completed before planning another step")
        if not self.has_work():
            return None

        batch = self.scheduler.get_next_batch_to_run()
        if batch is None:
            # A finished request lingers in the native running batch until
            # the next scheduler pass filters it (its ``finished()`` flag
            # also keeps ``has_work`` true), so a None batch here means the
            # scheduler could not even form a filtered decode batch — i.e.
            # genuine scheduler state corruption, not a closed-loop turn
            # boundary (the engine consumes completions before planning
            # again, so a drained pool reports has_work()==False above).
            raise RuntimeError("SGLang Scheduler returned no batch while work remains")
        self.scheduler.cur_batch = batch
        is_spec_decode = (
            not self.scheduler.spec_algorithm.is_none()
            and batch.forward_mode.is_decode()
        )
        if is_spec_decode:
            self._prepare_spec_decode_batch(batch)
            # TARGET_VERIFY is an extend-family mode; the extend vectors
            # now hold per-draft-token entries (prefix_lens were advanced
            # by the allocation), so the decode normalization below must
            # read them from the requests instead.
            accepts = self._draw_verify_accepts(batch)
            requests, generated_tokens, finished_request_ids = (
                _normalize_native_verify_batch(
                    batch,
                    accepts,
                    acceptance_source_kind=(
                        "explicit_assumption"
                        if self._mtp_assumption is not None
                        else "replay_observation"),
                    acceptance_source_id=(
                        self._mtp_assumption.source_id
                        if self._mtp_assumption is not None
                        else "chain_default"),
                ))
            scheduled_ids = {r.request_id for r in requests}
            finished_request_ids = [
                rid for rid in finished_request_ids if rid in scheduled_ids]
            # The speculative target verify's declared work and
            # accepted/emitted progress always reach the native Scheduler
            # (``spec_accepted_tokens`` / ``accept_lens`` below).  The
            # engine-facing rows keep the full MTP metadata (raw model
            # output equals accepted+1, verified equals proposed equals
            # accepted, padded proposer mode, explicit acceptance source).
            # The dense0.11 eager source used by the replay has no
            # MTP-enabled program, so the priced view cannot integrate
            # these rows without silently degrading them to ordinary
            # decode (root-forbidden).  Hold the step as an explicit
            # unpriced integration boundary instead: the engine sees a
            # transfer-only step, the native Scheduler keeps its real
            # accepted/emitted progress, and the run surfaces as partial
            # (never a complete-performance claim).
            unfinished = [
                request for request in requests
                if request.request_id not in finished_request_ids]
            if unfinished:
                # The dense0.11 eager source has no MTP-enabled program, so
                # the priced view cannot integrate these rows without
                # silently degrading them to ordinary decode (root-forbidden).
                # Surface the combination as explicitly unsupported at the
                # backend boundary instead: no fabricated transfer/store
                # marker, no zeroed MTP evidence, no ordinary-decode
                # downgrade.  Callers that need MTP-priced runs must supply
                # a pricing stack with an MTP-enabled source.
                raise NotImplementedError(
                    "SGLangBackend: speculative verify rows require an "
                    "MTP-enabled pricing source; the dense0.11 eager "
                    "forward would silently degrade them to ordinary "
                    "decode. Report this run as unsupported (or price "
                    "through an MTP-capable source).")
            # Every verify row finished its output budget this step: the
            # engine can observe the truthful terminal rows (emitted equals
            # raw on finished requests is allowed).
            embedding_token_ids = tuple(
                self.dummy_token_id
                for request in requests
                for _ in range(request.output_tokens))
            embedding_token_source = (
                "sglang_declared_synthetic_fixture_token_ids")
            model_source_inputs = None
        else:
            requests, generated_tokens, finished_request_ids = (
                _normalize_native_batch(batch)
            )
            embedding_token_ids, embedding_token_source = (
                self._embedding_stream_for_batch(batch))
            model_source_inputs = None
        cache_events = []
        for scheduled in requests:
            request_id = scheduled.request_id
            if request_id in self._first_schedule:
                continue
            self._first_schedule.add(request_id)
            native = self._native_requests[request_id]
            request = self._requests[request_id]
            cache_events.extend(self._initial_cache_events(native, request))

        step = BackendStep(
            schedule=ScheduleStep(
                step_id=f"{self.config.pool_name}:sglang:{self._step_index}",
                pool_name=self.config.pool_name,
                requests=tuple(requests),
                finished_request_ids=tuple(finished_request_ids),
            ),
            kv_cache_events=tuple(cache_events),
            generated_tokens=generated_tokens,
            embedding_token_ids=embedding_token_ids,
            embedding_token_source=embedding_token_source,
            model_source_inputs=model_source_inputs,
        )
        self._pending = (step, batch)
        self._pending_accepts = accepts if is_spec_decode else None
        self._step_index += 1
        return step

    def _v32_padded_mtp_state(
        self, batch: Any, requests: list[Any],
    ) -> Any:
        """Typed padded-MTP source state for V32-family pricing.

        The fabricated verify batch uses the chain topology (topk=1): one
        draft pass proposes every speculative token for every request, so
        the V32 packed row layout is ``(all-rows, retained-row)`` — the
        first element covers ``token_count`` ids and the tail elements
        cover one id per request each.  The bridge's dummy token stands
        in for the draft model's sampled ids (acceptance was already
        drawn in ``_draw_verify_accepts``).
        """
        from src.serving.workload.runtime_inputs import V32PaddedMTPInputs

        request_ids = tuple(r.request_id for r in requests)
        token_count = sum(r.input_tokens + r.output_tokens for r in requests)
        # ``draft_num`` packed passes cover the padded batch: the first
        # element spans all scheduled rows, each tail element spans one
        # retained row per request (gpu_model_runner.propose_draft_token_ids
        # layout, mirrored in the GLM V32 source).
        draft_num = self.scheduler.server_args.speculative_num_draft_tokens
        draft_ids = (
            tuple([self.dummy_token_id] * token_count),
            *([tuple([self.dummy_token_id] * len(requests))]
              * (draft_num - 1)),
        )
        return V32PaddedMTPInputs(
            request_ids=request_ids,
            source_kind="explicit_assumption",
            source_id=(
                self._mtp_assumption.source_id
                if self._mtp_assumption is not None
                else "chain_default"),
            draft_input_token_ids=draft_ids,
            discarded_rows=(False,) * len(requests),
            block_size_tokens=1,
            parallel_drafting=False,
            use_local_argmax_reduction=False,
            probabilistic=False,
        )

    def _embedding_stream_for_batch(
        self, batch: Any,
    ) -> tuple[tuple[int, ...] | None, str | None]:
        """Explicit scheduled token ids for symbolic-forward pricing paths.

        Returns ``(ids, source)`` when every scheduled request carries its
        real token content (the Dummy+Patch bridge never invents tokens);
        ``(None, None)`` when the batch is decode-shaped (per-step decode
        ids are fabricated by the data plane, not part of the request
        stream) so shape-only sources price it without a value fallback.
        """
        forward_mode = batch.forward_mode
        if not hasattr(forward_mode, "is_extend") or not forward_mode.is_extend():
            return None, None
        ids: list[int] = []
        for req in batch.reqs:
            # ``fill_ids``/``prefix_indices`` are real Req fields; a decode
            # batch that lacks them is not a fresh-extend shape and must
            # fail closed instead of silently degrading to a value-less row.
            fill = req.fill_ids
            prefix = req.prefix_indices
            if fill is None or prefix is None:
                return None, None
            ids.extend(int(t) for t in fill[len(prefix):])
        if not ids:
            return None, None
        return tuple(ids), "sglang_explicit_request_token_ids"

    def _prepare_spec_decode_batch(self, batch: Any) -> None:
        """Install a fabricated EagleVerifyInput on a decode batch.

        Mirrors what ``EAGLEWorker.forward_batch_speculative_generation``
        does before the target verify forward: allocate the draft-token
        slots through the real allocator and attach the verify spec_info.
        The acceptance pattern is a fixed chain (bonus + first draft
        accepted, rest rejected), giving ``num_draft_tokens - 1`` accepted
        tokens per verify for the chain topology with topk=1.
        """
        import torch

        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        draft_num = self.scheduler.server_args.speculative_num_draft_tokens
        bs = batch.batch_size()
        device = "cpu"
        draft_tokens = []
        retrieve_index = []
        retrieve_next_token = []
        retrieve_next_sibling = []
        offset = 0
        for _ in batch.reqs:
            draft_tokens.extend([self.dummy_token_id] * draft_num)
            retrieve_index.append(list(range(offset, offset + draft_num)))
            retrieve_next_token.append(
                [-1] + list(range(offset + 1, offset + draft_num)))
            retrieve_next_sibling.append([-1] * draft_num)
            offset += draft_num
        seq_lens_cpu = batch.seq_lens_cpu
        spec_info = EagleVerifyInput(
            draft_token=torch.tensor(draft_tokens, dtype=torch.long, device=device),
            custom_mask=torch.ones(len(draft_tokens), dtype=torch.bool, device=device),
            positions=torch.zeros(len(draft_tokens), dtype=torch.long, device=device),
            retrive_index=torch.tensor(
                retrieve_index, dtype=torch.int32, device=device),
            retrive_next_token=torch.tensor(
                retrieve_next_token, dtype=torch.int32, device=device),
            retrive_next_sibling=torch.tensor(
                retrieve_next_sibling, dtype=torch.int32, device=device),
            retrive_cum_len=torch.arange(
                0, (bs + 1) * draft_num, draft_num, dtype=torch.int32, device=device),
            spec_steps=self.scheduler.server_args.speculative_num_steps,
            topk=1,
            draft_token_num=draft_num,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=int(seq_lens_cpu.sum().item())
            if seq_lens_cpu is not None else 0,
            seq_lens_cpu=seq_lens_cpu,
        )
        batch.spec_info = spec_info
        spec_info.prepare_for_verify(batch, self.scheduler.page_size)
        batch.forward_mode = ForwardMode.TARGET_VERIFY

    def _draw_verify_accepts(self, batch: Any) -> list[int]:
        """Draw per-request accepted draft-token counts for one verify.

        Falls back to the chain default (draft_num - 1) when no explicit
        ``MTPAcceptanceAssumption`` is installed; otherwise draws through
        the assumption's deterministic per-(step, request) sampler.
        """
        draft_num = self.scheduler.server_args.speculative_num_draft_tokens
        proposal_depth = draft_num - 1
        accepts: list[int] = []
        assumption = self._mtp_assumption
        # Stash the raw draws before output-budget clamping so the
        # completion path can record them as the zero-fitting acceptance
        # evidence on the native requests.
        self._pending_raw_accepts: dict[str, int] = {}
        for req in batch.reqs:
            request_id = _sglang_request_id(req.rid)
            if assumption is not None:
                accepted = assumption.accepted_count(
                    step_id=f"{self.config.pool_name}:sglang:{self._step_index}",
                    request_id=request_id,
                    proposed_tokens=proposal_depth,
                )
                accept = accepted + 1  # bonus token is always verified
            else:
                raise ValueError(
                    "SGLang speculative decode requires an explicit "
                    "MTPAcceptanceAssumption; no implicit chain_default "
                    "acceptance is allowed")
            self._pending_raw_accepts[request_id] = accept
            native = self._native_requests[request_id]
            remaining = (native.sampling_params.max_new_tokens
                         - len(native.output_ids))
            accept = max(0, min(accept, remaining))
            accepts.append(min(accept, draft_num))
        return accepts

    def complete_step(
        self,
        step: BackendStep,
        virtual_time_s: float,
        *,
        execute_model: bool,
        execute_transfers: bool,
        sample_tokens: bool,
    ) -> None:
        del virtual_time_s
        if execute_model or execute_transfers or sample_tokens:
            raise ValueError(
                "the SGLang simulator bridge never executes a data plane")
        if self._pending is None or self._pending[0] is not step:
            raise ValueError("completion does not match the pending SGLang step")

        _, batch = self._pending
        was_spec_decode = (
            not self.scheduler.spec_algorithm.is_none()
            and batch.spec_info is not None
        )
        if was_spec_decode:
            accepts = self._pending_accepts
            if accepts is None or len(accepts) != len(batch.reqs):
                raise RuntimeError(
                    "SGLang verify completion lost its per-request accepts")
            # Run the fabricated-verify bookkeeping that the real
            # ``EagleVerifyInput.verify`` performs inside the worker
            # (SGLang 0.5.4 non-v2 eagle leaves verify side-effects to
            # the worker, so with enable_overlap=False the scheduler
            # path consumes them directly):
            #   * extend req.output_ids by the drawn accept count,
            #   * free the KV slots of rejected draft tokens,
            #   * advance batch.seq_lens/seq_lens_cpu,
            #   * compact batch.out_cache_loc to the accepted slots.
            # This must happen before process_batch_result so the result
            # processor only sees finished-cache / stream-output work.
            self._apply_verify_side_effects(batch, accepts)
            # Extend req.output_ids here (mirroring the worker-side
            # ``EagleVerifyInput.verify`` tail): the non-v2 eagle decode
            # processor only appends its own terminal token for
            # *v2-eagle* batches, so the accepted draft tokens must land
            # on output_ids here for both the priced and unpriced paths.
            for req, accept in zip(batch.reqs, accepts):
                if accept:
                    req.output_ids.extend([self.dummy_token_id] * accept)
                    req.check_finished(accept)
            # get_next_batch_to_run merges last_batch into the next batch
            # via ``running_batch.merge_batch`` which calls
            # ``spec_info.merge_batch``; EagleVerifyInput has no merge
            # (the worker replaces it with EagleDraftInput).  Swap in a
            # minimal draft-state object that satisfies the merge/filter
            # contract: an empty EagleDraftInput (page_size=1 uses only
            # allocate_lens/future_indices).
            from sglang.srt.model_executor.forward_batch_info import (
                CaptureHiddenMode,
            )
            from sglang.srt.speculative.eagle_info import EagleDraftInput
            import torch as _torch
            # ALLOC_LEN_PER_DECODE is owned by the construction cycle
            # (sglang_native._patch_eagle_draft_alloc_len) and restored by
            # destroy; this path only reads it.
            draft_input = EagleDraftInput(
                hidden_states=_torch.zeros(
                    (batch.batch_size(), 1), dtype=_torch.float16),
                verified_id=_torch.full(
                    (batch.batch_size(), 1), self.dummy_token_id,
                    dtype=_torch.long),
                accept_length=_torch.ones(
                    batch.batch_size(), dtype=_torch.int32),
                accept_length_cpu=[1] * batch.batch_size(),
                seq_lens_for_draft_extend=batch.seq_lens,
                seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu,
                req_pool_indices_for_draft_extend=batch.req_pool_indices,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            batch.spec_info = draft_input
            # Spec completions must NOT merge back into the running batch
            # through last_batch: their requests were already extended by
            # the fabricated verify, and the running batch still holds the
            # same Req objects — merging would double-count every request
            # in the next decode batch.
            self.scheduler.last_batch = None
            result = _fake_spec_generation_batch_result(
                batch,
                self.dummy_token_id,
                step.generated_tokens,
            )
            result = self._attach_spec_result_fields(batch, result, accepts)
            # run_batch hands the raw next-token tensor to the batch so
            # filter_batch (retraction / finishing) can slice it on the
            # next scheduler pass.
            batch.output_ids = result.next_token_ids
        else:
            result = _fake_generation_batch_result(
                batch,
                self.dummy_token_id,
                step.generated_tokens,
            )
        self._pending_accepts = None
        self.scheduler.process_batch_result(batch, result)
        # These assignments mirror ``Scheduler.event_loop_normal`` around
        # process_batch_result.  The next call uses last_batch to merge/filter
        # native running state before selecting another batch.
        self.scheduler.forward_ct += 1
        if batch.forward_mode.is_decode():
            self.scheduler.forward_ct_decode += 1
        if was_spec_decode:
            # Speculative decode batches run entirely inside the worker in
            # 0.5.4 (their reqs were extended by the fabricated verify),
            # so there is no filtered last_batch to merge back — keeping
            # it would double-count every request in the running batch.
            # The batch is dropped entirely; the running batch retains
            # whatever draft state it already had.
            self.scheduler.last_batch = None
            batch.spec_info = None
            # The accepted counts were already extended onto req.output_ids
            # before process_batch_result (mirroring the worker-side
            # ``EagleVerifyInput.verify`` tail).
            pass
        else:
            self.scheduler.last_batch = batch
        self.scheduler.cur_batch = None
        self._pending = None
        # Requests that hit their output budget this step are finished from
        # the bridge's point of view, but they still occupy the native
        # running batch until the next scheduler pass filters them.  Drop
        # them from the native queue bookkeeping right away so has_work()
        # mirrors the native scheduler's own view; the engine completes the
        # scheduled step through its completion-update path either way.
        for request_id in tuple(self._native_requests):
            if self._native_requests[request_id].finished():
                del self._native_requests[request_id]
        return 0.0

    def _apply_verify_side_effects(
        self, batch: Any, accepts: list[int],
    ) -> None:
        """Apply the verify side-effects the worker would normally perform.

        Mirrors the tail of ``EagleVerifyInput.verify`` (chain topology,
        page_size=1, no finished requests in this batch): extend
        ``req.output_ids``, free the rejected draft-token slots, write
        the accepted slots into the req-to-token table, advance
        ``batch.seq_lens``/``batch.seq_lens_cpu`` and compact
        ``batch.out_cache_loc`` to the accepted slots.
        """
        import torch

        draft_num = self.scheduler.server_args.speculative_num_draft_tokens
        if len(accepts) != len(batch.reqs):
            raise RuntimeError("SGLang verify accepts do not match batch")
        accept_lens = torch.tensor(accepts, dtype=torch.int64)
        accept_lens_cpu = accept_lens.clone()

        evict_mask = torch.ones(
            batch.out_cache_loc.shape[0], dtype=torch.bool)
        offset = 0
        for index, req in enumerate(batch.reqs):
            for accepted_index in range(offset, offset + accepts[index]):
                evict_mask[accepted_index] = False
            offset += draft_num
        self.scheduler.token_to_kv_pool_allocator.free(
            batch.out_cache_loc[evict_mask])

        # 0.5.4's get_next_batch_to_run advances seq_lens by +1 inside
        # prepare_for_decode *before* returning the batch (the spec
        # early-return skips only allocation, not the increment).  The
        # table rows for the draft tokens were therefore already written
        # at [seq_len, seq_len + draft_num) by prepare_for_verify; the
        # accepted slots beyond that window start at seq_len + draft_num.
        bs = batch.batch_size()
        extended = torch.tensor(
            [accepts[i] + 1 - draft_num for i in range(bs)],
            dtype=torch.int64)
        _assign_req_to_token_pool_torch(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens + draft_num,
            batch.seq_lens + draft_num + extended.to(batch.seq_lens.dtype),
            batch.out_cache_loc[~evict_mask],
            bs,
        )
        batch.out_cache_loc = batch.out_cache_loc[~evict_mask]
        batch.seq_lens.add_(accept_lens.to(batch.seq_lens.dtype) + 1)
        batch.seq_lens_cpu.add_(accept_lens_cpu + 1)
        # Record the assumption-drawn accept lengths on the native
        # requests (mirrors ``EagleVerifyInput.verify``): the replay
        # log reads them per verify step as the zero-fitting
        # acceptance evidence.  ``req.output_ids`` extension happens in
        # ``complete_step`` (before ``process_batch_result``) because the
        # non-v2 eagle decode processor does not append output ids for
        # this batch.
        raw_accepts = getattr(self, "_pending_raw_accepts", None)
        self._pending_raw_accepts = None
        if raw_accepts:
            for req in batch.reqs:
                drawn = raw_accepts.get(_sglang_request_id(req.rid))
                if drawn is None:
                    continue
                req.spec_verify_ct = getattr(req, "spec_verify_ct", 0) + 1
                req.spec_accepted_tokens = (
                    getattr(req, "spec_accepted_tokens", 0)
                    + max(0, drawn - 1))

    def _attach_spec_result_fields(
        self, batch: Any, result: Any, accepts: list[int],
    ) -> Any:
        """Give a fabricated verify result the v2-eagle tensor fields.

        ``process_batch_result_decode`` consumes ``accept_lens`` (tokens
        accepted per request, including the bonus) and ``allocate_lens``
        (total slots allocated per request) when the batch is v2-eagle;
        ``_resolve_spec_overlap_token_ids`` slices ``next_token_ids`` in
        ``stride = speculative_num_draft_tokens`` chunks per request.
        """
        import torch

        draft_num = self.scheduler.server_args.speculative_num_draft_tokens
        accept_lens = [min(a, draft_num) for a in accepts]
        allocate_lens = [draft_num] * len(batch.reqs)
        result.accept_lens = torch.tensor(accept_lens, dtype=torch.int32)
        result.allocate_lens = torch.tensor(allocate_lens, dtype=torch.int32)
        result.num_accepted_tokens = sum(accept_lens) - len(batch.reqs)
        flat = list(result.next_token_ids)
        if len(flat) < len(batch.reqs) * draft_num:
            flat = flat + [self.dummy_token_id] * (
                len(batch.reqs) * draft_num - len(flat))
        result.next_token_ids = torch.tensor(
            flat, dtype=torch.long)
        return result

    def has_work(self) -> bool:
        # Only requests the native scheduler can still schedule count: a
        # request that hit its output budget stays in the native running
        # batch (it is filtered on the next scheduler pass), but planning
        # another step for it is impossible — no batch can be formed.
        # The engine still completes every in-flight step through its own
        # completion-update path, so draining semantics are unchanged.
        return self._pending is not None or any(
            not native.finished()
            for native in self._native_requests.values()
        )

    def _initial_cache_events(
        self,
        native_request: Any,
        request: RequestSpec,
    ) -> list[KVCacheEvent]:
        prompt_tokens = request.input_tokens
        tiered_fields = (
            "cached_tokens_device", "cached_tokens_host",
            "cached_tokens_storage",
        )
        has_tiered = [hasattr(native_request, name) for name in tiered_fields]
        if all(has_tiered):
            device = _sglang_integer(
                native_request.cached_tokens_device, "device cached tokens")
            host = _sglang_integer(
                native_request.cached_tokens_host, "host cached tokens")
            storage = _sglang_integer(
                native_request.cached_tokens_storage, "storage cached tokens")
            if not hasattr(native_request, "cached_tokens"):
                raise TypeError("SGLang tiered request lacks cached_tokens total")
            cached_total = _sglang_integer(
                native_request.cached_tokens, "total cached tokens")
            if min(device, host, storage, cached_total) < 0:
                raise ValueError("SGLang cached-token counts cannot be negative")
            if device + host + storage != cached_total:
                raise ValueError(
                    "SGLang tiered cached-token counts disagree with cached_tokens")
        elif any(has_tiered):
            raise TypeError(
                "SGLang request exposes an incomplete tiered cache contract")
        elif hasattr(native_request, "cached_tokens"):
            device = _sglang_integer(
                native_request.cached_tokens, "device cached tokens")
            if device < 0:
                raise ValueError("SGLang cached-token count cannot be negative")
            host = 0
            storage = 0
        else:
            raise TypeError("SGLang request lacks cached-token accounting")
        if device + host + storage > prompt_tokens:
            raise ValueError(
                "SGLang cached-token count exceeds the request prompt length")
        if request.prefix_hit_tokens not in (0, device + host + storage):
            raise ValueError(
                "SGLang trace-authoritative prefix_hit_tokens disagrees with "
                "the scheduler-computed cached tokens; token-identity replay "
                "must derive prefix hits from the real RadixCache")

        events = []
        offset = 0
        for tier, count in (("hbm", device), ("host", host), ("storage", storage)):
            if count <= 0:
                continue
            events.append(self._cache_event(
                request.request_id,
                KVCacheAction.HIT,
                tier,
                offset,
                offset + count,
            ))
            offset += count
        if offset < prompt_tokens:
            events.append(self._cache_event(
                request.request_id,
                KVCacheAction.MISS,
                "hbm",
                offset,
                prompt_tokens,
            ))
        return events

    def _cache_event(
        self,
        request_id: str,
        action: KVCacheAction,
        tier: str,
        start: int,
        end: int,
    ) -> KVCacheEvent:
        return KVCacheEvent(
            request_id=request_id,
            action=action,
            tier=tier,
            token_start=start,
            token_end=end,
            bytes=(end - start) * self.config.kv_bytes_per_token,
        )


def _validate_native_scheduler(scheduler: Any) -> None:
    required_methods = (
        "get_next_batch_to_run",
        "process_batch_result",
        "_add_request_to_queue",
        "init_req_max_new_tokens",
    )
    missing_methods = [
        name for name in required_methods
        if not hasattr(scheduler, name) or not callable(getattr(scheduler, name))
    ]
    if missing_methods:
        raise TypeError(
            "SGLang scheduler is missing native control-plane methods: "
            + ", ".join(missing_methods)
        )
    required_fields = (
        "enable_overlap", "pp_size", "spec_algorithm",
        "disaggregation_mode", "model_config", "tokenizer",
    )
    missing_fields = [
        name for name in required_fields if not hasattr(scheduler, name)
    ]
    if missing_fields:
        raise TypeError(
            "SGLang scheduler is missing native control-plane fields: "
            + ", ".join(missing_fields)
        )
    if not isinstance(scheduler.enable_overlap, bool):
        raise TypeError("SGLang enable_overlap must be boolean")
    if scheduler.enable_overlap:
        raise ValueError("SGLang overlap scheduling requires worker-owned GPU results")
    if not isinstance(scheduler.pp_size, int) or scheduler.pp_size != 1:
        raise ValueError("SGLang pipeline-parallel result relaying is not supported")
    spec_algorithm = scheduler.spec_algorithm
    if not hasattr(spec_algorithm, "is_none") or not callable(
            spec_algorithm.is_none):
        raise TypeError("SGLang spec_algorithm lacks is_none()")
    spec_is_none = spec_algorithm.is_none()
    if not isinstance(spec_is_none, bool):
        raise TypeError("SGLang spec_algorithm.is_none() must return boolean")
    if not spec_is_none:
        # EAGLE/NEXTN (MTP) is supported via the native construction path;
        # the draft worker runs for allocation/verify bookkeeping only.
        if not (spec_algorithm.is_eagle() or spec_algorithm.name == "NEXTN"):
            raise ValueError(
                "SGLang speculative decoding other than EAGLE/NEXTN requires "
                "a live draft worker")
        if getattr(scheduler, "draft_worker", None) is None:
            raise ValueError("SGLang EAGLE/NEXTN requires a draft worker")
    disaggregation_mode = scheduler.disaggregation_mode
    if disaggregation_mode is not None:
        if not hasattr(disaggregation_mode, "value"):
            raise TypeError("SGLang disaggregation_mode lacks enum value")
        if not isinstance(disaggregation_mode.value, str):
            raise TypeError("SGLang disaggregation_mode.value must be a string")
        mode = disaggregation_mode.value.lower()
        if mode not in ("", "null", "none"):
            raise ValueError(
                "SGLang native disaggregation requires live KV-transfer processes"
            )


def _make_native_request(
    scheduler: Any,
    request: RequestSpec,
    *,
    dummy_token_id: int,
    max_output_tokens: int,
):
    max_output_tokens = _sglang_integer(
        max_output_tokens, "max_output_tokens")
    if max_output_tokens <= 0:
        raise ValueError("SGLang max_output_tokens must be positive")
    try:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "SGLangBackend.from_scheduler requires an SGLang installation"
        ) from exc

    sampling_params = SamplingParams(
        max_new_tokens=max_output_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    # The tokenizer pipeline calls normalize() before admission so
    # stop-string state exists; replay that here (tokenizer may be None
    # under skip_tokenizer_init).
    sampling_params.normalize(scheduler.tokenizer)
    # Mark every replayed request as retract-safe: the Dummy+Patch bridge
    # never runs a real prefill forward, so the "prefill must not be
    # interrupted" guarantee sglang gives non-retractable requests does
    # not apply.  Without this, requests past the retraction threshold
    # make ``init_next_round_input`` refuse to expose their prefix match
    # (extend_input_len stays 0), and the PrefillAdder sees a zero-token
    # admission it must reject — stalling the schedule with work pending.
    if hasattr(sampling_params, "retractable"):
        sampling_params.retractable = True
    elif hasattr(sampling_params, "retract_safe"):
        sampling_params.retract_safe = True
    model_config = scheduler.model_config
    if model_config is None or not hasattr(model_config, "vocab_size"):
        raise TypeError("SGLang scheduler model_config lacks vocab_size")
    vocab_size = _sglang_integer(model_config.vocab_size, "model vocab_size")
    if vocab_size <= 0:
        raise ValueError("SGLang model vocab_size must be positive")
    dummy_token_id = _sglang_integer(dummy_token_id, "dummy_token_id")
    if not 0 <= dummy_token_id < vocab_size:
        raise ValueError(
            "SGLang dummy_token_id must be within the model vocabulary")
    if request.prompt_token_ids is not None:
        input_ids = [
            _sglang_integer(token, "prompt token id") for token in request.prompt_token_ids
        ]
        if len(input_ids) != request.input_tokens:
            raise ValueError(
                f"SGLang prompt_token_ids length {len(input_ids)} does not "
                f"match request input_tokens {request.input_tokens}")
        for token in input_ids:
            if not 0 <= token < vocab_size:
                raise ValueError(
                    "SGLang prompt token id must be within the model vocabulary")
    else:
        # Unknown prompt identity must not be promoted to known cache
        # content: fabricating a repeated dummy id would make unrelated
        # requests share a false prefix in the RadixCache.  A recorded
        # prompt dropped by the controller (over-length turn) surfaces as
        # a prompt-less stub with ``input_tokens == 0``; anything else is
        # a controller contract violation and fails closed.
        if request.input_tokens == 0:
            input_ids = []
        else:
            raise ValueError(
                "SGLang request lacks prompt_token_ids; refusing to "
                "fabricate prompt identity for unknown request content")
    native = Req(
        request.request_id,
        "",
        input_ids,
        sampling_params,
        eos_token_ids=set(),
        vocab_size=vocab_size,
        disagg_mode=scheduler.disaggregation_mode,
    )
    native.tokenizer = scheduler.tokenizer
    scheduler.init_req_max_new_tokens(native)
    native_max_output_tokens = _sglang_integer(
        native.sampling_params.max_new_tokens,
        "native max_new_tokens",
    )
    if native_max_output_tokens < max_output_tokens:
        raise ValueError(
            "SGLang scheduler capacity truncated max_new_tokens from "
            f"{max_output_tokens} to {native.sampling_params.max_new_tokens}"
        )
    # ``_add_request_to_queue`` silently drops requests once the waiting
    # queue passes ``max_queued_requests`` (abort-on-queued-limit).  A
    # silently-dropped request would surface later as a starvation error
    # with no mention of the drop, so fail closed at admission instead.
    # ``None`` means unlimited (the sglang default).
    queued_limit = scheduler.max_queued_requests
    if queued_limit is not None:
        if isinstance(queued_limit, bool) or not isinstance(queued_limit, int):
            raise TypeError("SGLang scheduler max_queued_requests must be int/None")
        waiting_count = len(scheduler.waiting_queue)
        if queued_limit <= 0:
            raise ValueError(
                "SGLang scheduler max_queued_requests is non-positive: "
                f"{queued_limit}")
        if waiting_count >= queued_limit:
            raise RuntimeError(
                f"SGLang scheduler waiting-queue limit {queued_limit} would "
                f"drop request {request.request_id!r} (queued={waiting_count})"
            )
    return native


def _assign_req_to_token_pool_torch(
    req_pool_indices: Any,
    req_to_token: Any,
    start_offset: Any,
    end_offset: Any,
    out_cache_loc: Any,
    batch_size: int,
) -> None:
    """Torch-native equivalent of sglang's triton slot-assignment kernel.

    ``sglang.srt.speculative.spec_utils.assign_req_to_token_pool_func``
    launches a triton kernel whenever the *import-time* CUDA probe is
    true, even when every tensor lives on CPU (our Dummy+Patch bridge).
    Both offsets and locations are 1-D here (chain topology); scatter
    each request's accepted slots into its req_to_token row.
    """
    del batch_size
    pool_len = req_to_token.shape[1]
    for index in range(req_pool_indices.shape[0]):
        req_index = int(req_pool_indices[index].item())
        start = int(start_offset[index].item())
        end = int(end_offset[index].item())
        if not 0 <= start < pool_len or not 0 <= end <= pool_len:
            raise ValueError(
                "SGLang verify slot-assignment window out of range: "
                f"req={req_index} [{start}, {end}) pool_len={pool_len}")
        length = end - start
        if length <= 0:
            continue
        if length > out_cache_loc.shape[0]:
            raise RuntimeError(
                "SGLang verify slot-assignment mismatch: "
                f"request row needs {length} slots but only "
                f"{out_cache_loc.shape[0]} accepted locations remain")
        slots = out_cache_loc[:length].to(req_to_token.dtype)
        req_to_token[req_index, start:end] = slots
        out_cache_loc = out_cache_loc[length:]


def _normalize_native_verify_batch(
    batch: Any,
    accepts: list[int],
    *,
    acceptance_source_kind: str,
    acceptance_source_id: str,
) -> tuple[list[ScheduledRequest], dict[str, int], list[str]]:
    """Normalize a decode batch after ``prepare_for_verify`` rewrote it.

    ``EagleVerifyInput.prepare_for_verify`` turns the decode batch into a
    TARGET_VERIFY batch: ``input_ids``/``extend_lens`` describe the draft
    tokens (bs * draft_num entries) and ``prefix_lens`` were advanced by
    the allocation.  The engine contract must still see one decode entry
    per request with its pre-verify computed-token count.

    Verify rows deliberately carry no speculative forward work: the dense
    shape-only pricing source cannot consume speculative metadata, so
    ``raw_model_output_tokens`` alone reports the accepted-token count
    (for G3 acceptance bookkeeping) and every MTP metadata field stays
    disabled.  The acceptance-dependent cost lands in the decode average.
    """
    reqs = list(batch.reqs)
    if len(accepts) != len(reqs):
        raise RuntimeError("SGLang verify accepts do not match batch requests")
    seq_lens_cpu = batch.seq_lens_cpu
    if seq_lens_cpu is None:
        raise RuntimeError("SGLang verify batch lacks seq_lens_cpu")
    seq_lens = [int(x) for x in seq_lens_cpu.tolist()]

    draft_num = None
    spec_info = batch.spec_info
    if spec_info is not None and hasattr(spec_info, "draft_token_num"):
        draft_num = _sglang_integer(spec_info.draft_token_num, "draft_token_num")
    if draft_num is None or draft_num < 1:
        raise RuntimeError("SGLang verify batch lacks a draft-token count")

    scheduled = []
    generated_tokens: dict[str, int] = {}
    finished_request_ids = []
    for index, req in enumerate(reqs):
        request_id = _sglang_request_id(req.rid)
        # prepare_for_decode (spec path) did NOT advance seq_lens, and
        # prepare_for_verify allocated draft_num slots at the current
        # sequence tail; the pre-verify computed length is seq_lens - 1.
        computed_tokens = seq_lens[index] - 1
        if computed_tokens < 0:
            raise ValueError(
                f"SGLang decode request {request_id!r} has negative "
                "computed-token count")
        # Verify rows carry the full declared target-verify work: the
        # accepted token count is the raw model output (the worker-side
        # verify produced that many logits rows), ``output_tokens`` is the
        # per-request verify row count (one terminal/bonus row plus the
        # padded draft-token rows that the dense forward prices), and every
        # MTP field stays enabled — the declared draft depth rides in the
        # proposer metadata and the acceptance evidence stays attached to
        # the row.  Proposed/verified keep the full draft_num width even
        # when the drawn acceptance is zero: a verify step that rejects
        # every draft token still proposed and verified draft_num tokens
        # (the dense forward priced those rows); only the accepted prefix
        # shrinks.  Zeroing them here would erase the real verify work
        # from the engine-facing row.
        accept = accepts[index]
        accepted = max(0, accept - 1)
        scheduled.append(ScheduledRequest(
            request_id=request_id,
            phase="decode",
            input_tokens=0,
            output_tokens=draft_num,
            computed_tokens=computed_tokens,
            raw_model_output_tokens=accept,
            mtp_proposed_tokens=draft_num - 1,
            mtp_verified_tokens=draft_num - 1,
            mtp_accepted_tokens=accepted,
            mtp_proposer_tokens=1,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind=acceptance_source_kind,
            mtp_acceptance_source_id=acceptance_source_id,
        ))
        generated_tokens[request_id] = accept
        max_new_tokens = _sglang_integer(
            req.sampling_params.max_new_tokens, "max_new_tokens")
        if max_new_tokens <= 0:
            raise ValueError("SGLang max_new_tokens must be positive")
        if len(req.output_ids) + accept >= max_new_tokens:
            finished_request_ids.append(request_id)

    return scheduled, generated_tokens, finished_request_ids


def _normalize_native_batch(
    batch: Any,
) -> tuple[list[ScheduledRequest], dict[str, int], list[str]]:
    """Normalize a real SGLang ``ScheduleBatch`` without importing SGLang."""

    forward_mode = batch.forward_mode
    if not hasattr(forward_mode, "is_decode") or not callable(
            forward_mode.is_decode):
        raise RuntimeError("SGLang forward mode lacks is_decode()")
    if not hasattr(forward_mode, "is_extend") or not callable(
            forward_mode.is_extend):
        raise RuntimeError("SGLang forward mode lacks is_extend()")
    is_decode = forward_mode.is_decode()
    is_extend = forward_mode.is_extend()
    if not isinstance(is_decode, bool) or not isinstance(is_extend, bool):
        raise TypeError("SGLang forward-mode predicates must return booleans")
    if not is_decode and not is_extend:
        raise RuntimeError(
            f"unsupported SGLang forward mode for scheduler simulation: "
            f"{forward_mode}"
        )

    reqs = list(batch.reqs)
    if is_decode:
        prefix_lens = [0] * len(reqs)
        extend_lens = [0] * len(reqs)
        decoding_reqs = {id(req) for req in reqs}
    elif is_extend:
        for field in ("prefix_lens", "extend_lens"):
            if not hasattr(batch, field):
                raise RuntimeError(
                    f"SGLang extend batch lacks required {field}")
        prefix_lens = list(batch.prefix_lens)
        extend_lens = list(batch.extend_lens)
        decoding_reqs = {
            id(req) for req in (batch.decoding_reqs or ())}
    else:
        raise RuntimeError(f"unsupported SGLang forward mode: {forward_mode}")
    if len(prefix_lens) != len(reqs) or len(extend_lens) != len(reqs):
        raise RuntimeError("SGLang ScheduleBatch token vectors do not match its requests")

    scheduled = []
    generated_tokens: dict[str, int] = {}
    finished_request_ids = []
    for index, req in enumerate(reqs):
        request_id = _sglang_request_id(req.rid)
        request_is_decode = is_decode or id(req) in decoding_reqs
        if request_is_decode:
            phase = "decode"
            input_tokens = 0
            output_tokens = 1
            computed_tokens = (
                len(req.origin_input_ids) + len(req.output_ids) - 1)
            if computed_tokens < 0:
                raise ValueError(
                    f"SGLang decode request {request_id!r} has negative "
                    "computed-token count")
            generates_token = True
        else:
            phase = "prefill"
            input_tokens = _sglang_integer(
                extend_lens[index], "extend token count")
            output_tokens = 0
            computed_tokens = _sglang_integer(
                prefix_lens[index], "prefix token count")
            if not hasattr(req, "is_chunked"):
                raise RuntimeError(
                    f"SGLang request {request_id!r} lacks is_chunked")
            is_chunked = _sglang_integer(req.is_chunked, "is_chunked flag")
            if is_chunked not in (0, 1):
                raise ValueError("SGLang is_chunked flag must be 0 or 1")
            generates_token = is_chunked == 0
        if input_tokens + output_tokens <= 0:
            raise RuntimeError(
                f"SGLang scheduled zero tokens for request {request_id!r}"
            )
        scheduled.append(ScheduledRequest(
            request_id=request_id,
            phase=phase,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            computed_tokens=computed_tokens,
            raw_model_output_tokens=int(generates_token),
        ))
        generated_tokens[request_id] = 0

        if generates_token:
            generated_tokens[request_id] = 1
            max_new_tokens = _sglang_integer(
                req.sampling_params.max_new_tokens, "max_new_tokens")
            if max_new_tokens <= 0:
                raise ValueError("SGLang max_new_tokens must be positive")
            if len(req.output_ids) + 1 >= max_new_tokens:
                finished_request_ids.append(request_id)

    return scheduled, generated_tokens, finished_request_ids


def _fake_spec_generation_batch_result(
    batch: Any,
    dummy_token_id: int,
    generated_tokens: dict[str, int],
):
    """Fabricated verify result: one terminal token per request.

    The accepted-token extension already happened in
    ``_apply_verify_side_effects`` (mirroring ``EagleVerifyInput.verify``);
    the prefill-path processor only needs a single next-token id per
    request to append and to check finish conditions.
    """
    try:
        import torch
        from sglang.srt.managers.utils import GenerationBatchResult
    except ImportError as exc:
        raise RuntimeError(
            "SGLangBackend.from_scheduler requires an SGLang installation"
        ) from exc

    token_ids = []
    for req in batch.reqs:
        request_id = _sglang_request_id(req.rid)
        generated = _sglang_integer(
            generated_tokens[request_id], "generated token count")
        if generated < 0:
            raise ValueError("SGLang generated token count cannot be negative")
        token_ids.append(dummy_token_id)
    for field in ("extend_lens", "extend_logprob_start_lens"):
        if not hasattr(batch, field):
            raise TypeError(f"SGLang batch lacks {field}")
    return GenerationBatchResult(
        next_token_ids=torch.tensor(token_ids, dtype=torch.long, device="cpu"),
        extend_input_len_per_req=(
            list(batch.extend_lens) if batch.extend_lens is not None
            else None
        ),
        extend_logprob_start_len_per_req=(
            list(batch.extend_logprob_start_lens)
            if batch.extend_logprob_start_lens is not None
            else None
        ),
        can_run_cuda_graph=False,
    )


def _fake_generation_batch_result(
    batch: Any,
    dummy_token_id: int,
    generated_tokens: dict[str, int],
):
    try:
        import torch
        from sglang.srt.managers.utils import GenerationBatchResult
    except ImportError as exc:
        raise RuntimeError(
            "SGLangBackend.from_scheduler requires an SGLang installation"
        ) from exc

    token_ids = []
    for req in batch.reqs:
        request_id = _sglang_request_id(req.rid)
        generated = _sglang_integer(
            generated_tokens[request_id], "generated token count")
        if generated not in (0, 1):
            raise ValueError(
                "non-speculative SGLang steps must generate zero or one token")
        token_ids.append(dummy_token_id)
    for field in ("extend_lens", "extend_logprob_start_lens"):
        if not hasattr(batch, field):
            raise TypeError(f"SGLang batch lacks {field}")
    result = GenerationBatchResult(
        next_token_ids=torch.tensor(token_ids, dtype=torch.long, device="cpu"),
        extend_input_len_per_req=(
            list(batch.extend_lens) if batch.extend_lens is not None
            else None
        ),
        extend_logprob_start_len_per_req=(
            list(batch.extend_logprob_start_lens)
            if batch.extend_logprob_start_lens is not None
            else None
        ),
        can_run_cuda_graph=False,
    )
    # ``filter_batch`` on the next scheduler pass slices ``batch.output_ids``
    # with the keep-indices; a real decode run leaves the sampled ids there
    # (see ``run_batch``).  Mirror that so finishing/retraction filters see
    # a well-formed tensor instead of None.
    batch.output_ids = result.next_token_ids
    return result
