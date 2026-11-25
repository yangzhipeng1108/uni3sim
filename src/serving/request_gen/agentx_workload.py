"""Normalized AgentX session workload contract for the src Serving owner.

This module is the src-side owner of the AgentX multi-turn workload
container: it defines the versioned turn/trace/lane/workload schema the
session controller consumes, plus the rebuild helper that turns a turn's
delta-form hash ids into its full block-aligned prefix.

The dataset_pipeline owner parses and validates the raw external corpus
(``semianalysisai/cc-traces-weka-062126`` ``traces.jsonl``) and *converts*
its records into instances of these containers at the owner boundary, so
the controller never imports dataset_pipeline code.  The on-disk
``agentx_session_workload/2`` JSON schema is defined by the dataset owner;
these containers are the normalized in-memory contract both owners share.

Nothing in this module prices GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

BLOCK_TOKENS = 64            # corpus block_size; in == 64 * len(hash_ids)
MIN_KEPT_TURNS = 20          # the corpus' own session filter (>= 20 requests)

MAIN_TURN_TYPES = ("n", "s")
ALL_TURN_TYPES = ("n", "s", "subagent")


@dataclass(frozen=True)
class AgentxSubagentTurn:
    """One recorded turn inside a subagent (child) conversation.

    Carries the same delta-form prefix fields as a main turn plus the
    source-recorded ``gap_s`` (the wait before this child turn starts,
    derived from the corpus' ``t``/``api_time`` fields).  Children of one
    ``AgentxSubagentSession`` execute strictly in sequence: turn ``i+1`` is
    injected only when turn ``i`` finishes, stamped at
    ``finish + gap_s``.  Different sessions fan out concurrently at the
    parent turn's finish instant (the parent's trigger).
    """

    in_tokens: int
    out_tokens: int
    gap_s: float
    lcp_blocks: int
    new_hash_ids: tuple[int, ...]
    model: str

    def __post_init__(self) -> None:
        for name in ("in_tokens", "out_tokens", "lcp_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"AgentX subagent turn {name} must be an integer")
        if self.in_tokens < BLOCK_TOKENS or self.in_tokens % BLOCK_TOKENS:
            raise ValueError(
                f"AgentX subagent turn in_tokens must be a positive multiple of {BLOCK_TOKENS}")
        if self.out_tokens < 0:
            raise ValueError("AgentX subagent turn out_tokens cannot be negative")
        if not isinstance(self.gap_s, (int, float)) or not math.isfinite(self.gap_s):
            raise TypeError("AgentX subagent turn gap_s must be finite")
        if self.gap_s < 0:
            raise ValueError("AgentX subagent turn gap_s cannot be negative")
        if self.lcp_blocks < 0:
            raise ValueError("AgentX subagent turn lcp_blocks cannot be negative")
        ids = tuple(int(h) for h in self.new_hash_ids)
        if self.in_tokens != BLOCK_TOKENS * (self.lcp_blocks + len(ids)):
            raise ValueError(
                f"AgentX subagent turn in_tokens {self.in_tokens} disagrees with "
                f"{BLOCK_TOKENS} x ({self.lcp_blocks} lcp + {len(ids)} new) hash ids")
        if any(h < 0 for h in ids):
            raise ValueError("AgentX subagent turn hash ids must be non-negative")
        if not isinstance(self.model, str) or not self.model:
            raise TypeError("AgentX subagent turn model must be a non-empty string")
        object.__setattr__(self, "new_hash_ids", ids)


@dataclass(frozen=True)
class AgentxSubagentSession:
    """One recorded subagent conversation: the parent's fan-out unit.

    ``agent_id`` is the corpus' child identity; ``turns`` is the child's
    own recorded sequence (source ``t``/``api_time`` gaps preserved).  The
    session starts when the parent turn finishes; within the session,
    turns chain with their recorded gaps, and the parent's lane resumes
    only after the session's last turn completes (fan-in).
    """

    agent_id: str
    model: str
    turns: tuple[AgentxSubagentTurn, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise ValueError("AgentX subagent session agent_id must be a non-empty string")
        if not isinstance(self.model, str) or not self.model:
            raise TypeError("AgentX subagent session model must be a non-empty string")
        turns = tuple(self.turns)
        if not turns:
            raise ValueError("AgentX subagent session requires at least one turn")
        if any(not isinstance(turn, AgentxSubagentTurn) for turn in turns):
            raise TypeError("AgentX subagent session turns must be AgentxSubagentTurn objects")
        object.__setattr__(self, "turns", turns)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def total_tokens(self) -> int:
        return sum(turn.in_tokens + turn.out_tokens for turn in self.turns)


@dataclass(frozen=True)
class AgentxTurn:
    """One main-agent turn in delta form.

    ``subagent_fanout`` records how many subagent requests this turn fanned
    out in the corpus; ``subagent_requests`` carries the child identities
    themselves (full-fan-out replay).  Subagent durations remain inside the
    main-stream idle gap in main-only mode; the count and identities keep
    the driver contract's fan-out observable so main-only replay and full
    fan-out stay distinguishable in the workload statistics.
    """
    in_tokens: int
    out_tokens: int
    gap_s: float
    lcp_blocks: int
    new_hash_ids: tuple[int, ...]
    model: str
    subagent_fanout: int = 0
    subagent_sessions: tuple[AgentxSubagentSession, ...] = ()

    def __post_init__(self) -> None:
        for name in ("in_tokens", "out_tokens", "lcp_blocks", "subagent_fanout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"AgentX turn {name} must be an integer")
        if self.subagent_fanout < 0:
            raise ValueError("AgentX turn subagent_fanout cannot be negative")
        sessions = tuple(self.subagent_sessions)
        if any(not isinstance(s, AgentxSubagentSession) for s in sessions):
            raise TypeError("AgentX turn subagent_sessions must be AgentxSubagentSession objects")
        if sessions and len(sessions) != self.subagent_fanout:
            raise ValueError(
                f"AgentX turn fanout {self.subagent_fanout} disagrees with "
                f"{len(sessions)} recorded subagent sessions")
        object.__setattr__(self, "subagent_sessions", sessions)
        if self.in_tokens < BLOCK_TOKENS or self.in_tokens % BLOCK_TOKENS:
            raise ValueError(
                f"AgentX turn in_tokens must be a positive multiple of {BLOCK_TOKENS}")
        if self.out_tokens < 0:
            raise ValueError("AgentX turn out_tokens cannot be negative")
        if not isinstance(self.gap_s, (int, float)) or not math.isfinite(self.gap_s):
            raise TypeError("AgentX turn gap_s must be finite")
        if self.gap_s < 0:
            raise ValueError("AgentX turn gap_s cannot be negative")
        if self.lcp_blocks < 0:
            raise ValueError("AgentX turn lcp_blocks cannot be negative")
        ids = tuple(int(h) for h in self.new_hash_ids)
        if self.in_tokens != BLOCK_TOKENS * len(ids):
            # The corpus is block-aligned: in == 64 * len(hash_ids).
            # Tolerate unaligned test-fixture turns but reject anything
            # claiming fewer hash ids than whole blocks cover.
            if self.in_tokens < BLOCK_TOKENS * len(ids):
                raise ValueError(
                    f"AgentX turn in_tokens {self.in_tokens} disagrees with "
                    f"{BLOCK_TOKENS} x {len(ids)} hash ids")
        if any(h < 0 for h in ids):
            raise ValueError("AgentX turn hash ids must be non-negative")
        object.__setattr__(self, "new_hash_ids", ids)
        if not isinstance(self.model, str) or not self.model:
            raise TypeError("AgentX turn model must be a non-empty string")


@dataclass(frozen=True)
class AgentxTrace:
    """One normalized AgentX trace: main-agent turns in recorded order."""
    trace_id: str
    models: tuple[str, ...]
    turns: tuple[AgentxTurn, ...]
    dropped_over_max_len: int

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id:
            raise ValueError("AgentX trace id must be a non-empty string")
        models = tuple(self.models)
        if any(not isinstance(m, str) or not m for m in models):
            raise TypeError("AgentX trace models must be non-empty strings")
        object.__setattr__(self, "models", models)
        turns = tuple(self.turns)
        if not turns:
            raise ValueError("AgentX trace requires at least one turn")
        if any(not isinstance(t, AgentxTurn) for t in turns):
            raise TypeError("AgentX trace turns must be AgentxTurn objects")
        object.__setattr__(self, "turns", turns)
        if isinstance(self.dropped_over_max_len, bool) or not isinstance(self.dropped_over_max_len, int):
            raise TypeError("AgentX trace dropped count must be an integer")
        if self.dropped_over_max_len < 0:
            raise ValueError("AgentX trace dropped count cannot be negative")


@dataclass(frozen=True)
class AgentxLanePlan:
    """One live-session lane: which trace, where it starts, how many turns."""
    lane: int
    trace_id: str
    start_turn: int
    start_ratio: float
    num_turns: int

    def __post_init__(self) -> None:
        if isinstance(self.lane, bool) or not isinstance(self.lane, int) or self.lane < 0:
            raise TypeError("lane index must be a non-negative integer")
        if not isinstance(self.trace_id, str) or not self.trace_id:
            raise ValueError("lane trace_id must be a non-empty string")
        for name in ("start_turn", "num_turns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"lane {name} must be a non-negative integer")
        if not isinstance(self.start_ratio, (int, float)) or not math.isfinite(self.start_ratio):
            raise TypeError("lane start_ratio must be finite")
        if not 0.0 <= self.start_ratio <= 1.0:
            raise ValueError("lane start_ratio must be in [0, 1]")


@dataclass(frozen=True)
class AgentxWorkload:
    """The frozen, hash-bound AgentX session workload.

    ``source_path``/``source_sha256`` record the raw corpus identity the
    dataset owner validated; lanes and spares reference keys of ``traces``.
    """
    source_path: Any  # pathlib.Path; kept as Any to avoid hard import cost
    source_sha256: str
    max_model_len: int
    seed: int
    lanes: tuple[AgentxLanePlan, ...]
    spares: tuple[str, ...]
    start_min_ratio: float
    start_max_ratio: float
    think_cap_s: float
    traces: Mapping[str, AgentxTrace]
    corpus_stats: Mapping[str, int]

    def __post_init__(self) -> None:
        from pathlib import Path
        if not isinstance(self.source_path, Path):
            raise TypeError("workload source path must be a Path")
        if (not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64):
            raise ValueError("workload source sha256 must be a 64-char hex")
        for name in ("max_model_len", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"workload {name} must be an integer")
        if self.max_model_len < 1:
            raise ValueError("workload max_model_len must be positive")
        lanes = tuple(self.lanes)
        if not lanes:
            raise ValueError("workload requires at least one lane")
        if any(not isinstance(l, AgentxLanePlan) for l in lanes):
            raise TypeError("workload lanes must be AgentxLanePlan objects")
        if len({l.lane for l in lanes}) != len(lanes):
            raise ValueError("workload lane indices must be unique")
        object.__setattr__(self, "lanes", lanes)
        spares = tuple(self.spares)
        if any(not isinstance(t, str) or not t for t in spares):
            raise TypeError("workload spares must be non-empty strings")
        object.__setattr__(self, "spares", spares)
        for name in ("start_min_ratio", "start_max_ratio", "think_cap_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise TypeError(f"workload {name} must be finite")
        if not 0.0 <= self.start_min_ratio <= self.start_max_ratio <= 1.0:
            raise ValueError("workload start ratios must satisfy 0 <= min <= max <= 1")
        if self.think_cap_s <= 0:
            raise ValueError("workload think_cap_s must be positive")
        traces = dict(self.traces)
        if not traces:
            raise ValueError("workload requires at least one trace")
        object.__setattr__(self, "traces", traces)
        used = {l.trace_id for l in lanes} | set(spares)
        if not used <= set(traces):
            raise ValueError("workload lanes and spares reference unknown traces")
        corpus_stats = dict(self.corpus_stats)
        for name in ("traces", "no_api_time_traces", "too_few_turns",
                     "dropped_turns_over_max_len", "eligible_traces"):
            if name not in corpus_stats:
                raise ValueError(f"corpus stat {name} is required")
            value = corpus_stats[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"corpus stat {name} must be a non-negative integer")
        object.__setattr__(self, "corpus_stats", corpus_stats)


    def turn_prompt_tokens(self, index: int) -> int:
        """The full recorded prompt length of turn ``index``."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("turn index must be an integer")
        if not 0 <= index < len(self.turns):
            raise ValueError("turn index out of range")
        return self.turns[index].in_tokens


def hash_ids_at(turns: tuple[AgentxTurn, ...], index: int) -> list[int]:
    """Rebuild the full hash-id prefix of ``turns[index]``.

    Turn 0 owns its full prefix; every later turn shares ``lcp_blocks``
    leading blocks with its predecessor and appends ``new_hash_ids``.
    """
    if not isinstance(turns, (list, tuple)) or any(
            not isinstance(turn, AgentxTurn) for turn in turns):
        raise TypeError("hash_ids_at requires a sequence of AgentxTurn")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("hash_ids_at index must be an integer")
    if not 0 <= index < len(turns):
        raise ValueError("hash_ids_at index out of range")
    ids: list[int] = []
    for position, turn in enumerate(turns[:index + 1]):
        ids = ids[:turn.lcp_blocks] + list(turn.new_hash_ids)
    return ids
