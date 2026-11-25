"""InferenceX AgentX ``traces.jsonl`` corpus reader and session extractor.

This module parses the external AgentX agentic-coding corpus (the
SemiAnalysis ``cc-traces-weka-062126`` ``traces.jsonl`` file) into a
normalized, hash-bound workload description.  It is the dataset_pipeline
owner for the AgentX input contract: it validates the corpus schema, block
size, hash-id scope and per-request invariants, and emits a deterministic,
seeded lane plan compatible with the session controller in
``src.serving.request_gen.agentx_sessions``.

The workload schema it produces (``agentx_session_workload/2``) is a
*dataset* schema: it carries the source identity (HF dataset name, file,
sha256), the frozen driver parameters, and per-lane turn sequences in
delta form.  Production simulation consumes the normalized containers
defined by the src owner (``src.serving.request_gen.agentx_workload``);
this module builds instances of those containers at the owner boundary.
Nothing in this module prices GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from src.serving.request_gen.agentx_workload import (
    BLOCK_TOKENS,
    MIN_KEPT_TURNS,
    AgentxLanePlan,
    AgentxSubagentSession,
    AgentxSubagentTurn,
    AgentxTrace,
    AgentxTurn,
    AgentxWorkload,
    hash_ids_at,
)

WORKLOAD_SCHEMA = "agentx_session_workload/3"

MAIN_TURN_TYPES = ("n", "s")
ALL_TURN_TYPES = ("n", "s", "subagent")

AGENTX_HF_DATASET = "semianalysisai/cc-traces-weka-062126"
AGENTX_TRACES_FILE = "traces.jsonl"
# sha256 of the traces.jsonl we built against; bound at workload build time.
AGENTX_TRACES_SHA256 = "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276"


@dataclass(frozen=True)
class AgentxCorpus:
    """The parsed corpus plus its source identity."""
    source_path: Path
    source_sha256: str
    max_model_len: int
    num_traces: int
    traces: Mapping[str, AgentxTrace]
    no_api_time_traces: int
    too_few_turns_traces: int
    dropped_turns_over_max_len: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            raise TypeError("AgentX corpus source path must be a Path")
        if (not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.source_sha256)):
            raise ValueError("AgentX corpus source sha256 must be a 64-char lowercase hex")
        if isinstance(self.max_model_len, bool) or not isinstance(self.max_model_len, int) or self.max_model_len < 1:
            raise TypeError("AgentX corpus max_model_len must be a positive integer")
        for name in ("num_traces", "no_api_time_traces", "too_few_turns_traces",
                     "dropped_turns_over_max_len"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"AgentX corpus {name} must be a non-negative integer")
        traces = dict(self.traces)
        if not traces:
            raise ValueError("AgentX corpus requires at least one trace")
        if set(traces) != set(self.traces):
            raise ValueError("AgentX corpus traces lost identity")
        object.__setattr__(self, "traces", traces)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source(path: Path) -> tuple[str, Any]:
    """Content hash of a raw traces source plus its decoded text stream.

    Accepts a plain ``traces.jsonl`` or a zstd-compressed
    ``traces.jsonl.zst`` (the pinned-raw storage form; decompression is
    streamed, so the 1.7 GiB corpus never lands on disk uncompressed).
    The returned hash is always the SHA256 of the *uncompressed* bytes,
    which is what ``AGENTX_TRACES_SHA256`` binds.
    """
    import io
    if path.suffix == ".zst":
        import zstandard
        digest = hashlib.sha256()
        with path.open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                for chunk in iter(lambda: reader.read(1 << 24), b""):
                    digest.update(chunk)
        raw = path.open("rb")
        return digest.hexdigest(), io.TextIOWrapper(
            zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
    return sha256_file(path), None


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def _parse_subagent_children(trace_id: str, record: Mapping[str, Any],
                             max_model_len: int) -> list[AgentxSubagentSession]:
    """Parse one recorded subagent into its own sequenced session.

    The corpus nests the subagent's recorded request sequence (each with
    source ``t``/``api_time``); the session preserves that internal
    ordering and derives each in-session gap from the source timestamps
    (``next.t - (prev.t + prev.api_time)``, clamped at zero).  Turns whose
    ``in + out`` exceeds the served window are dropped exactly like main
    turns; a session with no surviving turn leaves no executable identity
    (the parent's fan-out simply excludes it, like a dropped main turn).

    Cross-session concurrency is the source-supported fan-out unit: the
    parent turn's finish starts every session of its fan-out set at the
    same instant, while each session's own turns chain with the recorded
    gaps.  No edges are invented: sessions never interleave their own
    turns, and the parent lane resumes only after the session set drains.
    """
    agent_id = record["agent_id"]
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError(f"trace {trace_id}: subagent agent_id must be a non-empty string")
    nested = record["requests"]
    if not isinstance(nested, (list, tuple)) or not nested:
        raise TypeError(f"trace {trace_id}: subagent {agent_id} requests must be a non-empty sequence")
    turns: list[AgentxSubagentTurn] = []
    prev_ids: list[int] = []
    prev_end: float | None = None
    model: str | None = None
    for index, child in enumerate(nested):
        if not isinstance(child, Mapping):
            raise TypeError(f"trace {trace_id}: subagent {agent_id} request must be an object")
        child_type = child["type"]
        if child_type not in MAIN_TURN_TYPES:
            raise ValueError(
                f"trace {trace_id}: subagent {agent_id} request type {child_type!r}")
        in_tokens = _require_int(child["in"], f"trace {trace_id} subagent {agent_id}: in")
        out_tokens = _require_int(child["out"], f"trace {trace_id} subagent {agent_id}: out")
        if in_tokens + out_tokens > max_model_len:
            prev_end = None
            continue
        ids = child["hash_ids"]
        if not isinstance(ids, (list, tuple)):
            raise TypeError(f"trace {trace_id}: subagent {agent_id} hash_ids must be a sequence")
        ids = [int(h) for h in ids]
        if in_tokens != BLOCK_TOKENS * len(ids):
            raise ValueError(
                f"trace {trace_id} subagent {agent_id}: in {in_tokens} != "
                f"{BLOCK_TOKENS} x {len(ids)} hash ids")
        t = child["t"]
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise ValueError(f"trace {trace_id}: subagent {agent_id} t must be finite and non-negative")
        gap = 0.0 if prev_end is None else max(0.0, t - prev_end)
        lcp = 0
        for a, b in zip(prev_ids, ids):
            if a != b:
                break
            lcp += 1
        turn_model = child["model"]
        if model is None:
            model = turn_model
        turns.append(AgentxSubagentTurn(
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            gap_s=round(gap, 3),
            lcp_blocks=lcp,
            new_hash_ids=tuple(ids[lcp:]),
            model=turn_model,
        ))
        prev_ids = ids
        if "api_time" in child:
            api_time = child["api_time"]
            if not isinstance(api_time, (int, float)) or not math.isfinite(api_time) or api_time < 0:
                raise ValueError(f"trace {trace_id}: subagent {agent_id} api_time must be finite and non-negative")
            prev_end = float(t) + float(api_time)
        else:
            prev_end = None
    if not turns:
        return []
    return [AgentxSubagentSession(
        agent_id=agent_id,
        model=model if model is not None else "unknown",
        turns=tuple(turns),
    )]


def _parse_main_turns(trace_id: str, requests: Iterable[Mapping[str, Any]],
                      max_model_len: int) -> tuple[tuple[AgentxTurn, ...], int, int]:
    """Kept main-agent turns in delta form, the dropped count, and the number
    of kept turns (other than the last) without ``api_time``.

    Subagent records are captured as real child sessions
    (``AgentxSubagentSession`` per recorded ``agent_id``, preserving the
    child's own turn sequence and source-derived in-session gaps).  Replay
    stays main-only by default; full fan-out mode executes the sessions so
    the parent trigger, per-agent sequencing and cross-agent concurrency
    are exercised (GOAL.md I2).  The fan-out set attaches to the *next
    kept* main turn, matching the corpus' ordering (the subagent records
    sit between the triggering main turn and its successor).
    """
    turns: list[AgentxTurn] = []
    dropped = 0
    no_api_time = 0
    prev_ids: list[int] = []
    prev_end: float | None = None
    pending_subagents: list[AgentxSubagentSession] = []
    for req in requests:
        req_type = req["type"]
        if req_type not in ALL_TURN_TYPES:
            raise ValueError(f"trace {trace_id}: request type {req_type!r}")
        if req_type == "subagent":
            pending_subagents.extend(_parse_subagent_children(trace_id, req, max_model_len))
            continue
        in_tokens = _require_int(req["in"], f"trace {trace_id} t={req['t']}: in")
        out_tokens = _require_int(req["out"], f"trace {trace_id} t={req['t']}: out")
        ids = req["hash_ids"]
        if not isinstance(ids, (list, tuple)):
            raise TypeError(f"trace {trace_id}: hash_ids must be a sequence")
        ids = [int(h) for h in ids]
        if in_tokens != BLOCK_TOKENS * len(ids):
            raise ValueError(
                f"trace {trace_id} t={req['t']}: in {in_tokens} != "
                f"{BLOCK_TOKENS} x {len(ids)} hash ids")
        if in_tokens + out_tokens > max_model_len:
            dropped += 1
            pending_subagents = []
            continue
        t = req["t"]
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise ValueError(f"trace {trace_id}: t must be finite and non-negative")
        if turns and prev_end is None:
            # The previous kept turn cannot anchor this gap.
            no_api_time += 1
        lcp = 0
        for a, b in zip(prev_ids, ids):
            if a != b:
                break
            lcp += 1
        gap = 0.0 if not turns or prev_end is None else max(0.0, t - prev_end)
        turns.append(AgentxTurn(
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            gap_s=round(gap, 3),
            lcp_blocks=lcp,
            new_hash_ids=tuple(ids[lcp:]),
            model=req["model"],
            subagent_fanout=len(pending_subagents),
            subagent_sessions=tuple(pending_subagents),
        ))
        pending_subagents = []
        prev_ids = ids
        # api_time is optional in the corpus schema (WekaNormalRequest); a
        # trace that needs a missing one is rejected by load_corpus().
        if "api_time" in req:
            api_time = req["api_time"]
            if not isinstance(api_time, (int, float)) or not math.isfinite(api_time) or api_time < 0:
                raise ValueError(f"trace {trace_id}: api_time must be finite and non-negative")
            prev_end = float(t) + float(api_time)
        else:
            prev_end = None
    return tuple(turns), dropped, no_api_time


def load_corpus(traces_path: str | Path, *, max_model_len: int) -> AgentxCorpus:
    """Load and validate the raw ``traces.jsonl`` corpus.

    ``max_model_len`` is the served model context length; turns whose
    ``in + out`` exceed it are dropped, exactly as the corpus' own
    ``_256k`` variant filters at 256K.
    """
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 1:
        raise ValueError("max_model_len must be a positive integer")
    source = Path(traces_path)
    if not source.is_file():
        raise ValueError(f"AgentX traces source is not a file: {source}")
    sha, decompressed = sha256_source(source)
    if sha != AGENTX_TRACES_SHA256:
        raise ValueError(
            f"{source} sha256 {sha} != expected {AGENTX_TRACES_SHA256}")
    stats = {"traces": 0, "no_api_time_traces": 0,
             "too_few_turns": 0, "dropped_turns_over_max_len": 0}
    traces: dict[str, AgentxTrace] = {}
    with (decompressed if decompressed is not None else source.open()) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"trace line {line_number} is not valid JSON") from error
            if not isinstance(record, Mapping):
                raise TypeError(f"trace line {line_number} must be an object")
            stats["traces"] += 1
            trace_id = record["id"]
            if not isinstance(trace_id, str) or not trace_id:
                raise ValueError(f"trace line {line_number}: id must be a non-empty string")
            block_size = record["block_size"]
            if block_size != BLOCK_TOKENS:
                raise ValueError(
                    f"trace {trace_id}: block_size {block_size} != {BLOCK_TOKENS}")
            hash_id_scope = record["hash_id_scope"]
            if hash_id_scope != "local":
                raise ValueError(
                    f"trace {trace_id}: hash_id_scope {hash_id_scope} != 'local'")
            models = record["models"]
            if not isinstance(models, (list, tuple)):
                raise TypeError(f"trace {trace_id}: models must be a sequence")
            requests = record["requests"]
            if not isinstance(requests, (list, tuple)):
                raise TypeError(f"trace {trace_id}: requests must be a sequence")
            turns, dropped, no_api_time = _parse_main_turns(trace_id, requests, max_model_len)
            stats["dropped_turns_over_max_len"] += dropped
            if no_api_time:
                stats["no_api_time_traces"] += 1
                continue
            if len(turns) < MIN_KEPT_TURNS:
                stats["too_few_turns"] += 1
                continue
            traces[trace_id] = AgentxTrace(
                trace_id=trace_id,
                models=tuple(str(m) for m in models),
                turns=turns,
                dropped_over_max_len=dropped,
            )
    return AgentxCorpus(
        source_path=source,
        source_sha256=sha,
        max_model_len=max_model_len,
        num_traces=stats["traces"],
        traces=traces,
        no_api_time_traces=stats["no_api_time_traces"],
        too_few_turns_traces=stats["too_few_turns"],
        dropped_turns_over_max_len=stats["dropped_turns_over_max_len"],
    )


# ----------------------------------------------------------------- workload


def build_workload(
    corpus: AgentxCorpus,
    *,
    seed: int,
    lanes: int,
    spares: int,
    start_min_ratio: float,
    start_max_ratio: float,
    think_cap_s: float,
    prefer_subagent_traces: bool = False,
) -> AgentxWorkload:
    """Deterministically plan live lanes and spare trajectories.

    ``prefer_subagent_traces`` orders lane candidates by descending
    recorded subagent fan-out (then the plain seeded shuffle) so short
    fan-out replays exercise real parent/child dependencies without
    walking whole trajectories; the spare pool follows the same order.
    Turn windows, models, ratios and the think cap are unchanged.
    """
    """Deterministically pick the lane traces and seeded 25-75% start points.

    The trace order is shuffled with ``seed`` and lanes take the first
    ``lanes`` traces; spares follow.  Each lane starts at a seeded uniform
    point in ``[start_min_ratio, start_max_ratio]`` of its trajectory,
    matching the InferenceX AIPerf ``--scenario inferencex-agentx-mvp``
    driver contract.  ``max_model_len`` is the value the corpus was loaded
    with; it is carried through to the workload unchanged.
    """
    if not isinstance(corpus, AgentxCorpus):
        raise TypeError("build_workload requires an AgentxCorpus")
    for name in ("seed", "lanes", "spares"):
        value = locals()[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if lanes < 1:
        raise ValueError("lanes must be positive")
    if spares < 0:
        raise ValueError("spares cannot be negative")
    for name in ("start_min_ratio", "start_max_ratio", "think_cap_s"):
        value = locals()[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TypeError(f"{name} must be finite")
    if not 0.0 <= start_min_ratio <= start_max_ratio <= 1.0:
        raise ValueError("start ratios must satisfy 0 <= min <= max <= 1")
    if think_cap_s <= 0:
        raise ValueError("think_cap_s must be positive")
    order = sorted(corpus.traces)
    if prefer_subagent_traces:
        def _fanout(trace_id: str) -> int:
            return sum(
                turn.subagent_fanout
                for turn in corpus.traces[trace_id].turns)
        order.sort(key=_fanout, reverse=True)
    rng = random.Random(seed)
    rng.shuffle(order)
    need = lanes + spares
    if len(order) < need:
        raise ValueError(f"only {len(order)} eligible traces, need {need}")
    lane_plans = []
    for lane, trace_id in enumerate(order[:lanes]):
        n = len(corpus.traces[trace_id].turns)
        ratio = rng.uniform(start_min_ratio, start_max_ratio)
        lane_plans.append(AgentxLanePlan(
            lane=lane,
            trace_id=trace_id,
            start_turn=int(ratio * n),
            start_ratio=round(ratio, 4),
            num_turns=n,
        ))
    used = [plan.trace_id for plan in lane_plans] + order[lanes:need]
    return AgentxWorkload(
        source_path=corpus.source_path,
        source_sha256=corpus.source_sha256,
        max_model_len=corpus.max_model_len,
        seed=seed,
        lanes=tuple(lane_plans),
        spares=tuple(order[lanes:need]),
        start_min_ratio=start_min_ratio,
        start_max_ratio=start_max_ratio,
        think_cap_s=think_cap_s,
        traces={t: corpus.traces[t] for t in used},
        corpus_stats={
            "traces": corpus.num_traces,
            "no_api_time_traces": corpus.no_api_time_traces,
            "too_few_turns": corpus.too_few_turns_traces,
            "dropped_turns_over_max_len": corpus.dropped_turns_over_max_len,
            "eligible_traces": len(order),
        },
    )


def describe_workload(workload: AgentxWorkload) -> dict[str, Any]:
    """Summarize the frozen workload: start contexts and gap/out distributions."""
    if not isinstance(workload, AgentxWorkload):
        raise TypeError("describe_workload requires an AgentxWorkload")
    starts = []
    for plan in workload.lanes:
        trace = workload.traces[plan.trace_id]
        starts.append(trace.turns[plan.start_turn].in_tokens)
    gaps = sorted(turn.gap_s for trace in workload.traces.values() for turn in trace.turns[1:])
    outs = sorted(turn.out_tokens for trace in workload.traces.values() for turn in trace.turns)
    fans = sorted(turn.subagent_fanout for trace in workload.traces.values() for turn in trace.turns)
    new_tok = sorted(
        BLOCK_TOKENS * len(turn.new_hash_ids)
        for trace in workload.traces.values() for turn in trace.turns[1:])

    def pct(values: list[float], q: float) -> float:
        return values[min(len(values) - 1, int(q * len(values)))]

    return {
        "lane_start_context_tokens": starts,
        "lane_start_context_sum_first_n": {n: sum(starts[:n])
                                           for n in (4, 8, 12, 16) if n <= len(starts)},
        "gap_s_p50_p90_p99": [pct(gaps, 0.5), pct(gaps, 0.9), pct(gaps, 0.99)],
        "gap_fraction_above_cap": round(
            sum(g > workload.think_cap_s for g in gaps) / len(gaps), 4),
        "out_p50_p90_p99_max": [pct(outs, 0.5), pct(outs, 0.9), pct(outs, 0.99), outs[-1]],
        "subagent_fanout_p50_p90_max": [pct(fans, 0.5), pct(fans, 0.9), fans[-1]],
        "new_tokens_per_turn_p50_p90": [pct(new_tok, 0.5), pct(new_tok, 0.9)],
    }


# ------------------------------------------------------------- serialization


def workload_to_dict(workload: AgentxWorkload) -> dict[str, Any]:
    """Serialize a workload to the versioned JSON dict (schema agentx_session_workload/2)."""
    if not isinstance(workload, AgentxWorkload):
        raise TypeError("workload_to_dict requires an AgentxWorkload")
    return {
        "schema": WORKLOAD_SCHEMA,
        "source": {
            "hf_dataset": AGENTX_HF_DATASET,
            "file": AGENTX_TRACES_FILE,
            "sha256": workload.source_sha256,
            "path": str(workload.source_path),
        },
        "params": {
            "max_model_len": workload.max_model_len,
            "seed": workload.seed,
            "lanes": len(workload.lanes),
            "spares": len(workload.spares),
            "start_min_ratio": workload.start_min_ratio,
            "start_max_ratio": workload.start_max_ratio,
            "block_tokens": BLOCK_TOKENS,
            "min_kept_turns": MIN_KEPT_TURNS,
            "think_cap_s": workload.think_cap_s,
        },
        "corpus_stats": dict(workload.corpus_stats),
        "lanes": [
            {
                "lane": plan.lane,
                "trace": plan.trace_id,
                "start": plan.start_turn,
                "start_ratio": plan.start_ratio,
                "turns": plan.num_turns,
            }
            for plan in workload.lanes
        ],
        "spares": list(workload.spares),
        "traces": {
            trace_id: {
                "models": list(trace.models),
                "turns": [
                    {
                        "in": turn.in_tokens,
                        "out": turn.out_tokens,
                        "gap_s": turn.gap_s,
                        "lcp": turn.lcp_blocks,
                        "new": list(turn.new_hash_ids),
                        "model": turn.model,
                        "sub": turn.subagent_fanout,
                        "subs": [
                            {
                                "agent": session.agent_id,
                                "model": session.model,
                                "turns": [
                                    {
                                        "in": child.in_tokens,
                                        "out": child.out_tokens,
                                        "gap_s": child.gap_s,
                                        "lcp": child.lcp_blocks,
                                        "new": list(child.new_hash_ids),
                                        "model": child.model,
                                    }
                                    for child in session.turns
                                ],
                            }
                            for session in turn.subagent_sessions
                        ],
                    }
                    for turn in trace.turns
                ],
                "dropped_over_max_len": trace.dropped_over_max_len,
            }
            for trace_id, trace in workload.traces.items()
        },
    }


def save_workload(workload: AgentxWorkload, path: str | Path) -> None:
    if not isinstance(workload, AgentxWorkload):
        raise TypeError("save_workload requires an AgentxWorkload")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(workload_to_dict(workload), indent=1) + "\n")


def load_workload(path: str | Path) -> AgentxWorkload:
    """Load a saved workload JSON (schema ``agentx_session_workload/2``)."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"AgentX workload is not a file: {source}")
    try:
        record = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"AgentX workload is not valid JSON: {source}") from error
    if not isinstance(record, Mapping):
        raise TypeError("AgentX workload must be a JSON object")
    schema = record["schema"]
    if schema != WORKLOAD_SCHEMA:
        raise ValueError(
            f"unexpected AgentX workload schema {schema!r}")
    source_info = record["source"]
    if not isinstance(source_info, Mapping):
        raise TypeError("AgentX workload source must be an object")
    source_sha = source_info["sha256"]
    params = record["params"]
    if not isinstance(params, Mapping):
        raise TypeError("AgentX workload params must be an object")
    lanes_raw = record["lanes"]
    if not isinstance(lanes_raw, (list, tuple)):
        raise TypeError("AgentX workload lanes must be a sequence")
    lane_plans = []
    for lane_raw in lanes_raw:
        if not isinstance(lane_raw, Mapping):
            raise TypeError("AgentX workload lane must be an object")
        lane_plans.append(AgentxLanePlan(
            lane=lane_raw["lane"],
            trace_id=lane_raw["trace"],
            start_turn=lane_raw["start"],
            start_ratio=lane_raw["start_ratio"],
            num_turns=lane_raw["turns"],
        ))
    spares_raw = record["spares"]
    if not isinstance(spares_raw, (list, tuple)):
        raise TypeError("AgentX workload spares must be a sequence")
    traces_raw = record["traces"]
    if not isinstance(traces_raw, Mapping):
        raise TypeError("AgentX workload traces must be an object")
    traces: dict[str, AgentxTrace] = {}
    for trace_id, trace_raw in traces_raw.items():
        if not isinstance(trace_raw, Mapping):
            raise TypeError("AgentX workload trace must be an object")
        models = trace_raw["models"]
        if not isinstance(models, (list, tuple)):
            raise TypeError("AgentX workload trace models must be a sequence")
        turns_raw = trace_raw["turns"]
        if not isinstance(turns_raw, (list, tuple)):
            raise TypeError("AgentX workload trace turns must be a sequence")
        turns = []
        for turn_raw in turns_raw:
            if not isinstance(turn_raw, Mapping):
                raise TypeError("AgentX workload turn must be an object")
            if "subs" in turn_raw:
                subs_raw = turn_raw["subs"]
                if not isinstance(subs_raw, (list, tuple)):
                    raise TypeError("AgentX workload turn subs must be a sequence")
                sessions = []
                for sub_raw in subs_raw:
                    if not isinstance(sub_raw, Mapping):
                        raise TypeError("AgentX workload session must be an object")
                    turns_raw = sub_raw["turns"]
                    if not isinstance(turns_raw, (list, tuple)):
                        raise TypeError("AgentX workload session turns must be a sequence")
                    session_turns = tuple(AgentxSubagentTurn(
                        in_tokens=child_raw["in"],
                        out_tokens=child_raw["out"],
                        gap_s=child_raw["gap_s"],
                        lcp_blocks=child_raw["lcp"],
                        new_hash_ids=tuple(child_raw["new"]),
                        model=child_raw["model"],
                    ) for child_raw in turns_raw)
                    sessions.append(AgentxSubagentSession(
                        agent_id=sub_raw["agent"],
                        model=sub_raw["model"],
                        turns=session_turns,
                    ))
                subs: tuple[AgentxSubagentSession, ...] = tuple(sessions)
            else:
                subs = ()
            turns.append(AgentxTurn(
                in_tokens=turn_raw["in"],
                out_tokens=turn_raw["out"],
                gap_s=turn_raw["gap_s"],
                lcp_blocks=turn_raw["lcp"],
                new_hash_ids=tuple(turn_raw["new"]),
                model=turn_raw["model"],
                subagent_fanout=turn_raw["sub"],
                subagent_sessions=subs,
            ))
        dropped_raw = trace_raw["dropped_over_max_len"]
        if isinstance(dropped_raw, bool) or not isinstance(dropped_raw, int):
            raise TypeError("AgentX workload dropped count must be an integer")
        traces[trace_id] = AgentxTrace(
            trace_id=trace_id,
            models=tuple(models),
            turns=tuple(turns),
            dropped_over_max_len=dropped_raw,
        )
    corpus_stats = record["corpus_stats"]
    if not isinstance(corpus_stats, Mapping):
        raise TypeError("AgentX workload corpus_stats must be an object")
    return AgentxWorkload(
        source_path=Path(source_info["path"]),
        source_sha256=source_sha,
        max_model_len=params["max_model_len"],
        seed=params["seed"],
        lanes=tuple(lane_plans),
        spares=tuple(spares_raw),
        start_min_ratio=params["start_min_ratio"],
        start_max_ratio=params["start_max_ratio"],
        think_cap_s=params["think_cap_s"],
        traces=traces,
        corpus_stats=corpus_stats,
    )


__all__ = [
    "AGENTX_HF_DATASET",
    "AGENTX_TRACES_FILE",
    "AGENTX_TRACES_SHA256",
    "ALL_TURN_TYPES",
    "AgentxCorpus",
    "AgentxLanePlan",
    "AgentxTrace",
    "AgentxTurn",
    "AgentxWorkload",
    "BLOCK_TOKENS",
    "MAIN_TURN_TYPES",
    "MIN_KEPT_TURNS",
    "WORKLOAD_SCHEMA",
    "build_workload",
    "describe_workload",
    "hash_ids_at",
    "load_corpus",
    "load_workload",
    "save_workload",
    "sha256_file",
    "workload_to_dict",
]
