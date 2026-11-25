"""AgentX multi-turn session controller for the Midlayer Serving simulator.

This module implements the InferenceX AgentX ``--scenario
inferencex-agentx-mvp`` closed-loop client contract as a drop-in
replacement for ``ClosedLoopClients``:

* a fixed population of live-session lanes (``--concurrency``), each with
  at most one outstanding request;
* every lane starts at a seeded 25-75% point of its recorded trajectory;
* the next turn of a session is injected at the *simulated completion
  instant* of the previous turn plus the recorded driver idle gap
  (``t_k - (t_{k-1} + api_time_{k-1})``), capped at the configured
  ``--think-cap-s``;
* each lane replays the main-agent stream of its trace in recorded order
  (every top-level ``n``/``s`` request); subagent turns are currently not
  fanned out (their duration shows up as main-stream idle gap), matching
  the kept semantics of the existing ``agentx_session_replay.py`` prototype;
* when a trajectory ends, the lane continues with the next spare trace
  from turn 0 (session rollover);
* block-exact prefix structure is preserved: prompt = one 64-token
  synthetic block per hash id, seeded by trace, instance and hash id, so
  turn k+1 reuses exactly the blocks it shares with turn k and nothing
  crosses sessions or instances.  This drives the native token-identity
  prefix cache of the pinned vLLM Scheduler;
* per-session cache-bust: synthetic token blocks are namespaced by
  ``(trace, instance)``, so a new session instance of the same trace
  starts cold, exactly as the InferenceX driver busts the per-session
  cache.

The controller pre-builds the full finite turn chain per lane at
construction time (recording the exact number of turns per lane, with
rollover to spare traces), so the native Scheduler sees a static stream
identity.  ``notify_finished`` returns the successor ``RequestSpec``
stamped at ``finish_time_s + idle_gap``; the single-pool engine injects
it into the pending arrival queue at that engine-clock instant.

The workload container type is the versioned
``src.serving.request_gen.agentx_workload.AgentxWorkload`` schema; the
dataset_pipeline owner builds it from the raw corpus
(``agentx_session_workload/2``) and the controller only consumes the
normalized form, keeping the src owner free of dataset_pipeline imports.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Deque, Dict, Optional, Tuple

from src.serving.request_gen.agentx_workload import (
    BLOCK_TOKENS,
    AgentxSubagentSession,
    AgentxTrace,
    AgentxTurn,
    AgentxWorkload,
    hash_ids_at,
)
from src.serving.request_gen.closed_loop import ClosedLoopClients
from src.serving.request_gen.request_stream import RequestSpec


# Default synthetic block token id range [1000, 100000), matching the
# prototype's plain-vocabulary range of the DSR1 tokenizer and avoiding the
# special-id range that the native Scheduler reserves.  Controllers may
# override it (e.g. clamp to a small test-fixture vocabulary).
TOKEN_ID_RANGE = (1000, 100000)
REQUEST_ID_PREFIX = "agentx"


class AgentxPrompts:
    """Synthetic token blocks; identical across servers for the same nonce.

    Block content is a deterministic function of (trace, instance, hash_id)
    so that block-exact prefix reuse drives the native prefix cache and a
    new session instance of the same trace starts cold (per-session
    cache-bust).
    """

    def __init__(self, nonce: str, token_id_range: Tuple[int, int] = TOKEN_ID_RANGE) -> None:
        if not isinstance(nonce, str) or not nonce:
            raise ValueError("AgentX prompts nonce must be a non-empty string")
        lo, hi = token_id_range
        if (isinstance(lo, bool) or isinstance(hi, bool) or not isinstance(lo, int)
                or not isinstance(hi, int) or lo < 1 or hi <= lo):
            raise ValueError(
                f"AgentX token id range must be 1 <= lo < hi, got {token_id_range}")
        if hi > TOKEN_ID_RANGE[1]:
            raise ValueError(
                f"AgentX token id range upper bound {hi} exceeds the "
                f"prototype vocabulary {TOKEN_ID_RANGE[1]}")
        self._nonce = nonce
        self._token_id_range = (lo, hi)
        self._blocks: Dict[Tuple[str, int, int], Tuple[int, ...]] = {}

    def block(self, trace_id: str, instance: int, hash_id: int) -> Tuple[int, ...]:
        key = (trace_id, instance, hash_id)
        if key not in self._blocks:
            seed = hashlib.blake2b(
                f"{self._nonce}|{trace_id}|{instance}|{hash_id}".encode(),
                digest_size=8).digest()
            # xorshift64* — deterministic across runs and platforms
            state = int.from_bytes(seed, "little") or 0x9E3779B97F4A7C15
            lo, hi = self._token_id_range
            span = hi - lo
            tokens = []
            for _ in range(BLOCK_TOKENS):
                state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
                state ^= (state >> 7)
                state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
                tokens.append(lo + state % span)
            self._blocks[key] = tuple(tokens)
        return self._blocks[key]

    def prompt(self, trace_id: str, instance: int, hash_ids: Tuple[int, ...]) -> Tuple[int, ...]:
        out: list[int] = []
        for hid in hash_ids:
            out.extend(self.block(trace_id, instance, hid))
        return tuple(out)

    def forget(self, trace_id: str, instance: int) -> None:
        for key in [k for k in self._blocks if k[0] == trace_id and k[1] == instance]:
            del self._blocks[key]


@dataclass(frozen=True)
class AgentxSessionControllerConfig:
    """Frozen parameters of one AgentX simulation run."""
    workload: AgentxWorkload
    concurrency: int
    nonce: str
    think_cap_s: float
    max_turns: Optional[int] = None
    vocab_size: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    replay_subagents: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.workload, AgentxWorkload):
            raise TypeError("AgentX controller config requires an AgentxWorkload")
        if (isinstance(self.concurrency, bool) or not isinstance(self.concurrency, int)
                or self.concurrency < 1):
            raise ValueError("AgentX concurrency must be a positive integer")
        if self.concurrency > len(self.workload.lanes):
            raise ValueError(
                f"AgentX concurrency {self.concurrency} exceeds the planned "
                f"{len(self.workload.lanes)} lanes")
        if not isinstance(self.nonce, str) or not self.nonce:
            raise ValueError("AgentX nonce must be a non-empty string")
        if not isinstance(self.think_cap_s, (int, float)) or not math.isfinite(self.think_cap_s):
            raise TypeError("think_cap_s must be finite")
        if self.think_cap_s <= 0:
            raise ValueError("think_cap_s must be positive")
        if self.max_turns is not None:
            if (isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int)
                    or self.max_turns < 1):
                raise TypeError("max_turns must be a positive integer or None")
        if self.vocab_size is not None:
            if (isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int)
                    or self.vocab_size < 2):
                raise TypeError("vocab_size must be an integer >= 2 or None")
        if self.max_prompt_tokens is not None:
            if (isinstance(self.max_prompt_tokens, bool) or not isinstance(self.max_prompt_tokens, int)
                    or self.max_prompt_tokens < BLOCK_TOKENS):
                raise TypeError(
                    f"max_prompt_tokens must be an integer >= {BLOCK_TOKENS} or None")
        if self.max_output_tokens is not None:
            if (isinstance(self.max_output_tokens, bool) or not isinstance(self.max_output_tokens, int)
                    or self.max_output_tokens < 1):
                raise TypeError("max_output_tokens must be a positive integer or None")
        if not isinstance(self.replay_subagents, bool):
            raise TypeError("replay_subagents must be a boolean")


class AgentxSessionController(ClosedLoopClients):
    """AgentX closed-loop session controller.

    Implements the same ``notify_finished`` contract as
    ``ClosedLoopClients`` so the single-pool engine can inject successor
    turns at the simulated completion instant plus the recorded external
    idle gap.  The full finite turn chain per lane is built at construction
    time so the native Scheduler backend can bind a static stream identity
    (prefix-cache and MTP admission contracts require it).
    """

    def __init__(
        self,
        config: AgentxSessionControllerConfig,
    ) -> None:
        if not isinstance(config, AgentxSessionControllerConfig):
            raise TypeError("AgentxSessionController requires AgentxSessionControllerConfig")
        # Satisfy the ClosedLoopClients structural contract without running
        # its canonical-chain initializer: the AgentX controller owns its
        # lane chains (built below) and reimplements the protocol.
        self._canonical = ()
        self._client_count = config.concurrency
        self._arrival_transform = None
        self._config = config
        # Clamp the synthetic token range to the deployment vocabulary when
        # known, so the same controller can drive a small test fixture (whose
        # prompts must stay inside its tiny vocabulary) and a full model
        # (default range preserved).  Keeps ids >= 1 to avoid special ids.
        token_id_range = TOKEN_ID_RANGE
        if config.vocab_size is not None:
            token_id_range = (1, min(config.vocab_size, TOKEN_ID_RANGE[1]))
        self._prompts = AgentxPrompts(config.nonce, token_id_range=token_id_range)
        # Optional per-turn prompt clamp: the turn's block-exact tail is
        # dropped (and the recorded in-token count with it) so the prompt
        # fits a deployment whose max_model_len is smaller than the corpus
        # window.  Block-aligned, so prefix structure is preserved.
        self._max_prompt_blocks: Optional[int] = (
            None if config.max_prompt_tokens is None
            else config.max_prompt_tokens // BLOCK_TOKENS)
        # Per-lane spare queues are not used: lanes share the spare pool in
        # workload order, rolling over as their starting trajectories
        # exhaust.  The chain follows the recorded trajectory across
        # rollovers until the turn cap or the pool runs dry.
        self._spares: Deque[str] = deque(config.workload.spares)
        self._global_turn_index = 0
        self._truncated_outputs: Dict[str, bool] = {}
        # Pre-build the per-lane turn chains (injected dynamically).
        self._requests: Dict[str, RequestSpec] = {}
        self._lane_of_request: Dict[str, int] = {}
        self._lane_states: Dict[int, Dict[str, Any]] = {}
        self._chain_order: Dict[int, list[str]] = {}
        # The full issued stream (initial + injected successors); the
        # controller folds every injected successor into it inside
        # ``notify_finished`` (single or batch), so the stream stays
        # complete no matter which closed-loop entry point the engine uses.
        self.metrics_stream: list = []
        # Exact recorded turn behind every issued request: built at request
        # construction (rolls over to spare traces), so gap/in accessors
        # never re-derive it from chain position.
        self._turn_of_request: Dict[str, AgentxTurn] = {}
        # Full fan-out bookkeeping: per finished parent turn, the subagent
        # sessions still waiting to start (one entry per recorded session)
        # plus each session's remaining in-session turns; and parent /
        # session identity per injected child request.
        self._pending_sessions: Dict[str, list] = {}
        self._session_turns: Dict[str, list] = {}
        self._session_of_request: Dict[str, tuple] = {}
        self._subagent_requests_total = 0
        self._subagent_completed_total = 0
        self._build_stream()
        if self._config.replay_subagents:
            self._subagent_requests_total = sum(
                sum(session.num_turns for session in turn.subagent_sessions)
                for turn in self._turn_of_request.values())

    # -- stream construction ----------------------------------------------

    def _new_lane_state(self, lane: int, trace_id: str, start_turn: int) -> Dict[str, Any]:
        self._config.workload.traces[trace_id]
        return {
            "trace_id": trace_id,
            "instance": 0,
            "next_index": start_turn,
            "turns_completed": 0,
            "pending": None,
            "closed": False,
        }

    def _next_spare(self, lane: int) -> str:
        if not self._spares:
            raise RuntimeError(f"AgentX spare traces exhausted for lane {lane}")
        return self._spares.popleft()

    def _build_stream(self) -> None:
        for plan in self._config.workload.lanes[:self._config.concurrency]:
            state = self._new_lane_state(plan.lane, plan.trace_id, plan.start_turn)
            self._lane_states[plan.lane] = state
            chain: list[str] = []
            # Each lane always has exactly one turn in flight; build the
            # chain of successors until the turn cap or the end of the
            # lane's starting trace.  Rollover to spare traces stops the
            # chain (spares are a runtime concept, not part of the static
            # stream identity).
            while True:
                if self._config.max_turns is not None and len(chain) >= self._config.max_turns:
                    break
                request = self._successor_for(plan.lane, state)
                if request is None:
                    break
                self._requests[request.request_id] = request
                self._lane_of_request[request.request_id] = plan.lane
                chain.append(request.request_id)
                # Advance: the successor becomes the pending turn.
                state["pending"] = None
                # Follow the recorded trajectory across spare rollovers;
                # the chain stops only at the turn cap.
                if state["next_index"] >= len(
                        self._config.workload.traces[state["trace_id"]].turns):
                    if not self._spares:
                        break
            state["pending"] = (chain[0], 0) if chain else None
            state["next_index"] = 0
            self._chain_order[plan.lane] = chain

    def _output_ids(self, count: int) -> Tuple[int, ...]:
        """Synthetic decoded-token ids for one turn's completion.

        The native token-identity prefix cache requires complete identity
        (prompt *and* output ids) per request.  Content is irrelevant for
        pricing — only lengths matter — so ids are a short deterministic
        sequence inside the same clamped vocabulary.
        """
        if count < 1:
            # Zero-output turns still need *some* identity to satisfy the
            # cache contract; a single id keeps validation happy and is
            # semantically inert (the engine prices zero generated tokens).
            count = 1
        lo, hi = self._prompts._token_id_range
        # Deterministic ids, distinct from typical prompt blocks.
        return tuple(lo + (i * 9973 + 31) % (hi - lo) for i in range(count))

    @property
    def max_prompt_blocks(self) -> Optional[int]:
        return self._max_prompt_blocks

    def lane_turn_info(self, lane: int, position: int) -> Dict[str, Any]:
        """Diagnostics: the recorded turn and effective sizes for one chain slot."""
        chain = self._chain_order[lane]
        request = self._requests[chain[position]]
        turn = self._turn_of_request[request.request_id]
        return {
            "request_id": request.request_id,
            "turn_in": turn.in_tokens,
            "turn_out": turn.out_tokens,
            "turn_subagent_fanout": turn.subagent_fanout,
            "effective_input": request.input_tokens,
            "effective_output": request.output_tokens,
            "full_blocks": (turn.in_tokens // BLOCK_TOKENS),
            "clamped_blocks": self._max_prompt_blocks,
            "truncated_output": self._truncated_outputs[request.request_id],
        }

    def recorded_prompt_tokens(self, request_id: str) -> int:
        """The corpus-recorded prompt length behind a controller request.

        The corpus itself drops turns whose ``in + out`` exceeds the served
        ``max_model_len`` (the workload was built with that filter); live
        lanes only contain turns that survived it, so the recorded length
        is the real replayed context for every issued request.
        """
        if request_id not in self._requests:
            raise ValueError(f"unknown AgentX request {request_id!r}")
        return self._turn_of_request[request_id].in_tokens

    @property
    def replayed_context_turns(self) -> int:
        """Turns the workload carries whose recorded ``in`` exceeds its
        ``max_model_len`` (the corpus filter applied at its own window, so
        these turns replay as prompt-less stubs)."""
        count = 0
        for lane_plan in self._config.workload.lanes[:self._config.concurrency]:
            trace = self._config.workload.traces[lane_plan.trace_id]
            for position in range(lane_plan.start_turn, len(trace.turns)):
                if trace.turns[position].in_tokens > self._config.workload.max_model_len:
                    count += 1
        return count

    def _successor_for(self, lane: int, state: Dict[str, Any]) -> Optional[RequestSpec]:
        if state["closed"]:
            return None
        trace: AgentxTrace = self._config.workload.traces[state["trace_id"]]
        next_index = state["next_index"]
        if next_index >= len(trace.turns):
            # Trajectory exhausted: roll over to the next spare trace.
            self._prompts.forget(state["trace_id"], state["instance"])
            new_trace = self._next_spare(lane)
            state.update(self._new_lane_state(lane, new_trace, 0))
            trace = self._config.workload.traces[state["trace_id"]]
            next_index = 0
        turn = trace.turns[next_index]
        hash_ids = tuple(hash_ids_at(trace.turns, next_index))
        n_prompt_blocks = len(hash_ids)
        clamped_prompt = False
        dropped_turn = False
        if self._max_prompt_blocks is not None and n_prompt_blocks > self._max_prompt_blocks:
            # Deployment prompt clamp: keep the block-exact tail that fits.
            # The recorded in-token count is dropped with the blocks.
            hash_ids = hash_ids[n_prompt_blocks - self._max_prompt_blocks:]
            n_prompt_blocks = self._max_prompt_blocks
            clamped_prompt = True
        if turn.in_tokens > self._config.workload.max_model_len:
            # The corpus filter (in + out <= max_model_len) did not apply at
            # the corpus' own 256K window; replay drops the turn at the
            # input boundary exactly like the driver drops an over-context
            # session.  The scheduled stub carries no prompt content.
            dropped_turn = True
            hash_ids = ()
            n_prompt_blocks = 0
            clamped_prompt = False
        prompt = self._prompts.prompt(state["trace_id"], state["instance"], hash_ids)
        input_tokens = turn.in_tokens
        if dropped_turn:
            input_tokens = 0
        elif len(prompt) != input_tokens:
            if not clamped_prompt:
                raise RuntimeError(
                    f"AgentX lane {lane} turn {next_index}: prompt {len(prompt)} != in {input_tokens}")
            input_tokens = len(prompt)
        request_id = f"{REQUEST_ID_PREFIX}-l{lane}-{self._global_turn_index}"
        self._global_turn_index += 1
        output_tokens = turn.out_tokens
        self._truncated_outputs[request_id] = False
        if (self._config.max_output_tokens is not None
                and output_tokens > self._config.max_output_tokens):
            output_tokens = self._config.max_output_tokens
            self._truncated_outputs[request_id] = True
        request = RequestSpec(
            arrival_time_s=0.0,  # stamped at injection time
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prefix_hit_tokens=0,
            prompt_token_ids=prompt if prompt else None,
            output_token_ids=(self._output_ids(output_tokens)
                if output_tokens > 0 else None),
        )
        state["pending"] = (request_id, next_index)
        state["next_index"] = next_index + 1
        self._turn_of_request[request_id] = turn
        if self._config.replay_subagents:
            # Queue the recorded subagent sessions for execution after the
            # parent finishes; the parent/child dependency is enforced by
            # successors_of (sessions inject only at the parent's finish
            # instant), sessions of one parent run concurrently with each
            # other, and each session's own turns chain in recorded order.
            self._pending_sessions[request_id] = list(turn.subagent_sessions)
        return request

    def _session_turn_request(self, session: AgentxSubagentSession,
                              turn_index: int,
                              hash_ids: tuple[int, ...],
                              input_tokens: int | None = None) -> RequestSpec:
        """Build one in-session child turn under the session's identity.

        The prompt is the block-exact rebuild of the session's recorded
        hash-id prefix under its ``agent_id`` namespace (trace-local
        identity, cross-session isolation preserved); consecutive turns of
        one session share the session's prefix cache exactly like the
        recorded driver run.  ``hash_ids`` is the full prefix for this
        turn; ``input_tokens`` is the recorded in size (used when the
        session history was rebuilt from delta form).
        """
        turn = session.turns[turn_index]
        if input_tokens is None:
            input_tokens = turn.in_tokens
        if input_tokens > self._config.workload.max_model_len:
            prompt: tuple[int, ...] = ()
            input_tokens = 0
        else:
            prompt = self._prompts.prompt(session.agent_id, 0, hash_ids)
        if len(prompt) != input_tokens and input_tokens:
            raise RuntimeError(
                f"AgentX session {session.agent_id} turn {turn_index}: "
                f"prompt {len(prompt)} != in {input_tokens}")
        request_id = f"{REQUEST_ID_PREFIX}-{session.agent_id}-{self._global_turn_index}"
        self._global_turn_index += 1
        output_tokens = turn.out_tokens
        self._truncated_outputs[request_id] = False
        if (self._config.max_output_tokens is not None
                and output_tokens > self._config.max_output_tokens):
            output_tokens = self._config.max_output_tokens
            self._truncated_outputs[request_id] = True
        return RequestSpec(
            arrival_time_s=0.0,  # stamped at injection time
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prefix_hit_tokens=0,
            prompt_token_ids=prompt if prompt else None,
            output_token_ids=self._output_ids(output_tokens),
        )

    # -- ClosedLoopClients-compatible protocol ----------------------------

    def on_controller_step(
        self, step: Any, completion_time_s: float,
    ) -> list[RequestSpec]:
        """Per-step fan-out view for the thin CLI (full-fan-out mode).

        Returns the successors for this step's finished requests, stamped
        at their causal instants.  The engine consumes fan-out through the
        ``ClosedLoopClients.successors_of`` completion contract, so this
        hook is only a convenience view for callers that want the whole
        step's successor set in one call; in main-only mode it returns an
        empty list and keeps the metrics-stream fold-in behaviour.
        """

        finished = [(rid, completion_time_s) for rid in step.finished_request_ids]
        if self._config.replay_subagents:
            successors: list[RequestSpec] = []
            for request_id, finish_time_s in finished:
                successors.extend(self.successors_of(request_id, finish_time_s))
            return successors
        for request_id in step.finished_request_ids:
            successor = self.peek_successor(request_id)
            if successor is not None:
                self.metrics_stream.append(successor)
        return []

    @property
    def client_count(self) -> int:
        return self._config.concurrency

    @property
    def requests_per_client(self) -> int:
        if self._config.max_turns is not None:
            return self._config.max_turns
        # Unbounded lanes vary by trace length; report the minimum.
        return min(
            len(self._config.workload.traces[plan.trace_id].turns) - plan.start_turn
            for plan in self._config.workload.lanes[:self._config.concurrency]
        )

    def initial_requests(self) -> Tuple[RequestSpec, ...]:
        """The first turn of every live lane, all arriving at t=0."""
        initial = []
        for lane in sorted(self._lane_states):
            state = self._lane_states[lane]
            if state["pending"] is not None:
                request_id = state["pending"][0]
                initial.append(self._requests[request_id])
        self.metrics_stream.extend(initial)
        return tuple(initial)

    def peek_successor(self, request_id: str) -> Optional[RequestSpec]:
        """The successor turn a finished request *will* inject (or None).

        Non-mutating companion of ``notify_finished``: lets a caller record
        the full issued stream (e.g. for metrics token conservation) before
        the engine injects the successor at the finish instant.
        """
        if request_id not in self._requests:
            raise ValueError(f"unknown AgentX request {request_id!r}")
        lane = self._lane_of_request[request_id]
        state = self._lane_states[lane]
        if state["pending"] is None or state["pending"][0] != request_id:
            raise RuntimeError(
                f"AgentX lane {lane} finished unknown request {request_id!r}")
        chain = self._chain_order[lane]
        position = chain.index(request_id)
        if position + 1 >= len(chain):
            return None
        return self._requests[chain[position + 1]]

    def _resume_lane_after_children(
        self, parent_id: str, finish_time_s: float,
    ) -> Tuple[RequestSpec, ...]:
        """Fan-in: after a parent's last subagent session completes, resume
        the parent's lane with its next main turn at exactly this instant."""

        lane = self._lane_of_request[parent_id]
        state = self._lane_states[lane]
        chain = self._chain_order[lane]
        position = chain.index(parent_id)
        if position + 1 >= len(chain):
            state["pending"] = None
            return ()
        successor_id = chain[position + 1]
        successor = self._requests[successor_id]
        state["pending"] = (successor_id, position + 1)
        stamped = replace(successor, arrival_time_s=finish_time_s)
        self.metrics_stream.append(stamped)
        return (stamped,)

    def successors_of(
        self,
        request_id: str,
        finish_time_s: float,
    ) -> Tuple[RequestSpec, ...]:
        """Every successor a finish injects, stamped at its causal instant.

        Implements the ``ClosedLoopClients.successors_of`` completion
        contract: the engine calls this exactly once per finished request
        and appends every returned successor to its closed-loop arrival
        queue, which admits each at its true arrival instant.

        * In-session child turn: the session's next turn is injected at
          ``finish_time_s + min(recorded gap, think cap)`` (source-derived
          in-session sequencing); the session's last turn completes the
          session and decrements the parent's outstanding session count.
        * Last session of a parent: the parent's lane resumes with its
          next main turn at exactly that instant (fan-in dependency).
        * Main turn with recorded sessions (full fan-out): every session's
          first turn is returned at the parent's finish instant, so
          different agents run concurrently while each agent keeps its own
          recorded turn sequence (parent trigger, per-agent sequencing).
        * Ordinary main turn: the next chain turn arrives at
          ``finish_time_s + min(recorded gap, think cap)``.
        """

        if request_id not in self._requests:
            raise ValueError(f"unknown AgentX request {request_id!r}")
        if request_id in self._session_of_request:
            session_key = self._session_of_request.pop(request_id)
            parent_id, agent_id = session_key
            turns = self._session_turns[session_key]
            self._subagent_completed_total += 1
            if turns:
                # Next in-session turn: recorded gap, same agent identity.
                hash_ids, next_turn = turns[0]
                session = next(
                    session for session in self._pending_sessions[parent_id]
                    if session.agent_id == agent_id)
                request = self._session_turn_request(
                    session, 0, hash_ids, input_tokens=next_turn.in_tokens)
                turns[0:] = turns[1:]
                self._requests[request.request_id] = request
                self._lane_of_request[request.request_id] = \
                    self._lane_of_request[request_id]
                self._session_of_request[request.request_id] = session_key
                self.metrics_stream.append(request)
                gap = min(next_turn.gap_s, self._config.think_cap_s)
                return (replace(request, arrival_time_s=finish_time_s + gap),)
            turns[0:] = []
            # Session complete: drop it from the parent's outstanding set.
            sessions = self._pending_sessions[parent_id]
            sessions[:] = [s for s in sessions if s.agent_id != agent_id]
            if sessions:
                return ()
            return self._resume_lane_after_children(parent_id, finish_time_s)
        lane = self._lane_of_request[request_id]
        state = self._lane_states[lane]
        if state["pending"] is None or state["pending"][0] != request_id:
            raise RuntimeError(
                f"AgentX lane {lane} finished unknown request {request_id!r}")
        state["pending"] = None
        state["turns_completed"] += 1
        if self._config.replay_subagents:
            sessions = self._pending_sessions[request_id]
            if sessions:
                first_turns: list[RequestSpec] = []
                for session in sessions:
                    session_key = (request_id, session.agent_id)
                    # Seed each session's delta-form hash-id history with
                    # its recorded first turn; later turns rebuild the
                    # prefix from (lcp, new) exactly like main turns.
                    turns: list[tuple[tuple[int, ...], object]] = []
                    prev_ids: list[int] = []
                    for turn in session.turns:
                        ids = tuple(
                            prev_ids[:turn.lcp_blocks]) + tuple(turn.new_hash_ids)
                        turns.append((ids, turn))
                        prev_ids = list(ids)
                    self._session_turns[session_key] = turns
                    hash_ids, _first = turns[0]
                    child = self._session_turn_request(session, 0, hash_ids)
                    # Keep the seeded list at one entry per *remaining*
                    # in-session turn; turns[0] is always the next one to
                    # run, so the just-issued first turn is dropped here
                    # and the session's recorded gaps stay attached to
                    # their own turns.
                    turns[0:] = turns[1:]
                    self._requests[child.request_id] = child
                    self._lane_of_request[child.request_id] = lane
                    self._session_of_request[child.request_id] = session_key
                    self.metrics_stream.append(child)
                    first_turns.append(child)
                state["pending"] = (request_id, -1)
                return tuple(
                    replace(child, arrival_time_s=finish_time_s)
                    for child in first_turns)
        chain = self._chain_order[lane]
        if state["pending"] is not None and state["pending"][1] == -1:
            # Lane resume after the last child: advance to the chain
            # successor of the parent that owned the children.
            position = chain.index(request_id)
            if position + 1 >= len(chain):
                state["pending"] = None
                return ()
            successor_id = chain[position + 1]
            successor = self._requests[successor_id]
            state["pending"] = (successor_id, position + 1)
            gap = min(self._turn_of_request[successor_id].gap_s,
                      self._config.think_cap_s)
            stamped = replace(successor, arrival_time_s=finish_time_s + gap)
            self.metrics_stream.append(stamped)
            return (stamped,)
        try:
            position = chain.index(request_id)
        except ValueError:
            raise RuntimeError(
                f"AgentX lane {lane} request {request_id!r} not in chain") from None
        if position + 1 >= len(chain):
            return ()
        successor_id = chain[position + 1]
        successor = self._requests[successor_id]
        state["pending"] = (successor_id, position + 1)
        # Stamp the arrival at the finish instant plus the recorded gap.
        gap = min(self._turn_of_request[successor_id].gap_s,
                  self._config.think_cap_s)
        stamped = replace(successor, arrival_time_s=finish_time_s + gap)
        self.metrics_stream.append(stamped)
        return (stamped,)

    def notify_finished(
        self,
        request_id: str,
        finish_time_s: float,
    ) -> Optional[RequestSpec]:
        """Inherited single-successor adapter over ``successors_of``.

        Kept for the ``ClosedLoopClients`` contract: returns the first
        successor (or ``None``).  The single-pool engine consumes
        ``successors_of`` directly, so fan-out siblings are never dropped.
        """

        successors = self.successors_of(request_id, finish_time_s)
        if not successors:
            return None
        return successors[0]

    # -- introspection ----------------------------------------------------
    @property
    def issued_count(self) -> int:
        return self._global_turn_index

    @property
    def completed_count(self) -> int:
        return sum(s["turns_completed"] for s in self._lane_states.values())

    @property
    def truncated_output_turns(self) -> int:
        return sum(self._truncated_outputs.values())

    @property
    def subagent_fanout_total(self) -> int:
        """Total recorded subagent requests across all issued main turns.

        Replay remains main-agent-only (subagent duration already sits in
        the main-stream idle gap); this count keeps the driver contract's
        fan-out visible so main-only and full-fan-out profiles stay
        distinguishable per GOAL.md I2/I3.
        """
        return sum(turn.subagent_fanout for turn in self._turn_of_request.values())

    @property
    def subagent_requests_total(self) -> int:
        """Recorded child requests scheduled under full-fan-out replay."""
        return self._subagent_requests_total

    @property
    def subagent_completed_total(self) -> int:
        """Child requests that finished under full-fan-out replay."""
        return self._subagent_completed_total
