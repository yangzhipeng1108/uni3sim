#!/usr/bin/env python3
"""AgentX-derived multi-turn session replay (KV host-offload group, DSR1 TP8).

Workload source: the InferenceX AgentX corpus (``agentic-coding`` scenario;
HF ``semianalysisai/cc-traces-weka-062126`` ``traces.jsonl``, 393 Claude Code
sessions; each request carries ``in``/``out`` token counts, 64-token KV block
``hash_ids`` scoped per trace, and ``t``/``api_time`` timestamps).  InferenceX
replays it with the SemiAnalysis AIPerf fork (``--scenario
inferencex-agentx-mvp``: closed loop, ``--concurrency`` = live session lanes,
each lane starts at a random 25-75% point of its trajectory, recorded idle
gaps honoured up to 300 s, ``max_tokens`` = recorded ``out`` with
``ignore_eos``, a per-session cache-bust marker, 3600 s per point, subagents
fanned out).  This client keeps the parts that decide KV reuse and drops what
does not fit a 40-60 min 8-GPU budget:

* kept: main-agent stream of each trace in recorded order (every top-level
  ``n``/``s`` request), block-exact prefix structure (prompt = one 64-token
  synthetic block per hash id, seeded by trace, instance and hash id, so turn
  k+1 reuses exactly the blocks it shares with turn k and nothing crosses
  sessions), ``max_tokens`` = recorded ``out`` with ``ignore_eos``, closed-loop
  lanes starting at a seeded 25-75% point, next trace from turn 0 when a
  trajectory ends, idle gap before a turn = ``t_k - (t_{k-1} + api_time_{k-1})``
  of the kept turns (capped, ``--think-cap-s``).
* changed: requests with ``in + out`` above the server's max model length are
  dropped (the corpus' ``_256k`` variant does the same at 256K; DSR1 serves
  163,840), subagents are not replayed (their duration shows up as main-stream
  idle gap), duration is ``--warmup-s`` + ``--measure-s`` instead of 3600 s, no
  chat template (token-id prompts on ``/v1/completions``).

``build`` writes the workload JSON (deterministic from the corpus file, its
sha256 and the seed); ``replay`` runs it and writes ``requests.jsonl``,
``summary.json`` and three ``/metrics`` snapshots (start, measure start, end).
``build --dry-run``-style checking is simply ``build`` (CPU only); ``replay
--dry-run`` prints the lane plan and the first request of every lane without
sending anything.

Evaluation-only measurements; no timing feeds any model parameter.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import urllib.request

BLOCK_TOKENS = 64            # corpus block_size; in == 64 * len(hash_ids)
TOKEN_ID_RANGE = (1000, 100000)   # plain vocabulary ids of the DSR1 tokenizer
MIN_KEPT_TURNS = 20          # the corpus' own session filter (>= 20 requests)
WORKLOAD_SCHEMA = "agentx_session_workload/1"


# ------------------------------------------------------------------- build

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _main_turns(trace: dict, max_model_len: int) -> tuple:
    """Kept main-agent turns in delta form, the dropped count, and the number
    of kept turns (other than the last) without ``api_time``."""
    if trace["block_size"] != BLOCK_TOKENS or trace["hash_id_scope"] != "local":
        raise ValueError(f"trace {trace['id']}: block_size {trace['block_size']} "
                         f"scope {trace['hash_id_scope']}")
    turns, dropped, no_api_time = [], 0, 0
    prev_ids, prev_end = [], None
    for req in trace["requests"]:
        if req["type"] == "subagent":
            continue
        if req["type"] not in ("n", "s"):
            raise ValueError(f"trace {trace['id']}: request type {req['type']!r}")
        ids = req["hash_ids"]
        if req["in"] != BLOCK_TOKENS * len(ids):
            raise ValueError(f"trace {trace['id']} t={req['t']}: in {req['in']} != "
                             f"64 x {len(ids)} hash ids")
        if req["in"] + req["out"] > max_model_len:
            dropped += 1
            continue
        if turns and prev_end is None:
            no_api_time += 1  # the previous kept turn cannot anchor this gap
        lcp = 0
        for a, b in zip(prev_ids, ids):
            if a != b:
                break
            lcp += 1
        gap = 0.0 if not turns or prev_end is None else max(0.0, req["t"] - prev_end)
        turns.append({"in": req["in"], "out": req["out"], "gap_s": round(gap, 3),
                      "lcp": lcp, "new": ids[lcp:], "model": req["model"]})
        prev_ids = ids
        # api_time is optional in the corpus schema (WekaNormalRequest); a
        # trace that needs a missing one is rejected by build().
        prev_end = req["t"] + req["api_time"] if "api_time" in req else None
    return turns, dropped, no_api_time


def build(args) -> None:
    source = Path(args.traces)
    sha = _sha256_file(source)
    if sha != args.expect_sha256:
        raise SystemExit(f"{source} sha256 {sha} != expected {args.expect_sha256}")
    eligible, stats = {}, {"traces": 0, "no_api_time_traces": 0,
                           "too_few_turns": 0, "dropped_turns_over_max_len": 0}
    with source.open() as handle:
        for line in handle:
            trace = json.loads(line)
            stats["traces"] += 1
            turns, dropped, no_api_time = _main_turns(trace, args.max_model_len)
            stats["dropped_turns_over_max_len"] += dropped
            if no_api_time:
                stats["no_api_time_traces"] += 1
                continue
            if len(turns) < MIN_KEPT_TURNS:
                stats["too_few_turns"] += 1
                continue
            eligible[trace["id"]] = turns
    order = sorted(eligible)
    rng = random.Random(args.seed)
    rng.shuffle(order)
    need = args.lanes + args.spares
    if len(order) < need:
        raise SystemExit(f"only {len(order)} eligible traces, need {need}")
    lanes = []
    for lane, trace_id in enumerate(order[:args.lanes]):
        n = len(eligible[trace_id])
        ratio = rng.uniform(args.start_min_ratio, args.start_max_ratio)
        lanes.append({"lane": lane, "trace": trace_id, "start": int(ratio * n),
                      "start_ratio": round(ratio, 4), "turns": n})
    spares = order[args.lanes:need]
    used = [lane["trace"] for lane in lanes] + spares
    workload = {
        "schema": WORKLOAD_SCHEMA,
        "source": {"hf_dataset": "semianalysisai/cc-traces-weka-062126",
                   "file": "traces.jsonl", "sha256": sha, "path": str(source)},
        "params": {"max_model_len": args.max_model_len, "seed": args.seed,
                   "lanes": args.lanes, "spares": args.spares,
                   "start_min_ratio": args.start_min_ratio,
                   "start_max_ratio": args.start_max_ratio,
                   "block_tokens": BLOCK_TOKENS, "min_kept_turns": MIN_KEPT_TURNS},
        "corpus_stats": dict(stats, eligible_traces=len(order)),
        "lanes": lanes, "spares": spares,
        "traces": {t: eligible[t] for t in used}}
    Path(args.output).write_text(json.dumps(workload, separators=(",", ":")) + "\n")
    print(json.dumps({"output": args.output, "corpus_stats": workload["corpus_stats"],
                      "lanes": lanes}, indent=1))
    _describe(workload, args.think_cap_s)


def _describe(workload: dict, think_cap_s: float) -> None:
    """Context sizes at each lane start and the recorded gap distribution."""
    starts = []
    for lane in workload["lanes"]:
        turns = workload["traces"][lane["trace"]]
        starts.append(turns[lane["start"]]["in"])
    gaps = sorted(t["gap_s"] for tr in workload["traces"].values() for t in tr[1:])
    outs = sorted(t["out"] for tr in workload["traces"].values() for t in tr)
    new_tok = sorted(BLOCK_TOKENS * len(t["new"])
                     for tr in workload["traces"].values() for t in tr[1:])

    def pct(values, q):
        return values[min(len(values) - 1, int(q * len(values)))]
    print(json.dumps({
        "lane_start_context_tokens": starts,
        "lane_start_context_sum_first_n": {n: sum(starts[:n])
                                           for n in (4, 8, 12, 16) if n <= len(starts)},
        "gap_s_p50_p90_p99": [pct(gaps, 0.5), pct(gaps, 0.9), pct(gaps, 0.99)],
        "gap_fraction_above_cap": round(sum(g > think_cap_s for g in gaps) / len(gaps), 4),
        "out_p50_p90_p99_max": [pct(outs, 0.5), pct(outs, 0.9), pct(outs, 0.99), outs[-1]],
        "new_tokens_per_turn_p50_p90": [pct(new_tok, 0.5), pct(new_tok, 0.9)]},
        indent=1))


# ------------------------------------------------------------------ replay

class Prompts:
    """Synthetic token blocks; identical across servers for the same nonce."""

    def __init__(self, nonce: str):
        self.nonce = nonce
        self.blocks = {}

    def block(self, trace: str, instance: int, hid: int) -> list:
        key = (trace, instance, hid)
        if key not in self.blocks:
            import numpy as np  # vLLM env; kept out of `build`
            seed = hashlib.blake2b(f"{self.nonce}|{trace}|{instance}|{hid}".encode(),
                                   digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(seed, "little"))
            lo, hi = TOKEN_ID_RANGE
            self.blocks[key] = rng.integers(lo, hi, BLOCK_TOKENS).tolist()
        return self.blocks[key]

    def prompt(self, trace: str, instance: int, hash_ids: list) -> list:
        out = []
        for hid in hash_ids:
            out.extend(self.block(trace, instance, hid))
        return out

    def forget(self, trace: str, instance: int) -> None:
        for key in [k for k in self.blocks if k[0] == trace and k[1] == instance]:
            del self.blocks[key]


def _hash_ids_at(turns: list, index: int) -> list:
    ids = []
    for turn in turns[:index + 1]:
        ids = ids[:turn["lcp"]] + turn["new"]
    return ids


def _percentiles(values: list) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)

    def q(p):
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
    return {"n": len(s), "mean": sum(s) / len(s), "p50": q(0.5), "p90": q(0.9),
            "p99": q(0.99), "max": s[-1]}


def _scrape(url: str, path: Path) -> float:
    with urllib.request.urlopen(url, timeout=30) as r:
        path.write_bytes(r.read())
    return time.time()


class Replay:
    def __init__(self, args, workload: dict):
        self.args = args
        self.w = workload
        self.prompts = Prompts(args.nonce)
        self.spares = list(workload["spares"])
        self.records = []
        self.url = f"http://127.0.0.1:{args.port}/v1/completions"

    def next_spare(self, lane: int) -> str:
        if not self.spares:
            raise RuntimeError(f"lane {lane}: spare traces exhausted")
        return self.spares.pop(0)

    async def request(self, session, lane: int, trace: str, instance: int,
                      index: int, turn: dict, hash_ids: list, seen: set,
                      phase_of) -> None:
        prompt = self.prompts.prompt(trace, instance, hash_ids)
        if len(prompt) != turn["in"]:
            raise RuntimeError(f"{trace}#{index}: prompt {len(prompt)} != in {turn['in']}")
        # Leading blocks this session instance already sent in any earlier
        # turn (side calls interleave, so not only the previous turn).
        reusable, chain, prefix_known = 0, 0, True
        for hid in hash_ids:
            chain = hash((chain, hid))
            if prefix_known and chain in seen:
                reusable += 1
            else:
                prefix_known = False
                seen.add(chain)
        payload = {"model": self.args.served_model, "prompt": prompt,
                   "max_tokens": max(1, turn["out"]), "ignore_eos": True,
                   "temperature": 0, "stream": True,
                   "stream_options": {"include_usage": True}}
        data = json.dumps(payload).encode()
        rec = {"lane": lane, "trace": trace, "instance": instance, "turn": index,
               "prompt_tokens": turn["in"], "max_tokens": payload["max_tokens"],
               "reusable_prefix_tokens": BLOCK_TOKENS * reusable,
               "t_send": time.time()}
        rec["phase"] = phase_of(rec["t_send"])
        t_first = None
        usage = metrics = None
        chunks = 0
        try:
            async with session.post(self.url, data=data,
                                    headers={"Content-Type": "application/json"}) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:500]
                    raise RuntimeError(f"HTTP {resp.status}: {body}")
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk["choices"]:
                        chunks += 1
                        if t_first is None:
                            t_first = time.time()
                    if "usage" in chunk and chunk["usage"] is not None:
                        usage = chunk["usage"]
                        # present with --enable-per-request-metrics
                        metrics = chunk["metrics"]
            rec["t_end"] = time.time()
            if usage is None or t_first is None:
                raise RuntimeError("stream ended without usage or first token")
            if usage["completion_tokens"] != payload["max_tokens"]:
                raise RuntimeError(f"completion_tokens {usage['completion_tokens']} != "
                                   f"max_tokens {payload['max_tokens']}")
            if usage["prompt_tokens"] != turn["in"]:
                raise RuntimeError(f"server prompt_tokens {usage['prompt_tokens']} != "
                                   f"{turn['in']}")
            details = usage["prompt_tokens_details"]
            if details is None:
                raise RuntimeError("no prompt_tokens_details: server lacks "
                                   "--enable-prompt-tokens-details")
            rec.update({"status": "ok", "completion_tokens": usage["completion_tokens"],
                        "cached_tokens": details["cached_tokens"], "chunks": chunks,
                        "ttft_s": t_first - rec["t_send"],
                        "e2e_s": rec["t_end"] - rec["t_send"],
                        "tpot_s": ((rec["t_end"] - t_first) / (usage["completion_tokens"] - 1)
                                   if usage["completion_tokens"] > 1 else None),
                        "server_metrics": metrics})
        except asyncio.CancelledError:
            rec.update({"status": "cancelled", "t_end": time.time(),
                        "ttft_s": None if t_first is None else t_first - rec["t_send"],
                        "chunks": chunks})
            self.records.append(rec)
            raise
        except Exception as exc:  # recorded, then re-raised: the replay fails
            rec.update({"status": "error", "error": repr(exc)[:800], "t_end": time.time()})
            self.records.append(rec)
            raise
        self.records.append(rec)

    async def lane(self, session, lane: int, t_stop: float, phase_of) -> None:
        plan = self.w["lanes"][lane]
        trace, start, instance = plan["trace"], plan["start"], 0
        while True:
            turns = self.w["traces"][trace]
            prev_ids = _hash_ids_at(turns, start - 1) if start > 0 else []
            seen = set()  # block-prefix chains sent by this trajectory
            for index in range(start, len(turns)):
                turn = turns[index]
                if index > start:
                    gap = min(turn["gap_s"], self.args.think_cap_s)
                    if time.time() + gap >= t_stop:
                        return
                    await asyncio.sleep(gap)
                if time.time() >= t_stop:
                    return
                hash_ids = prev_ids[:turn["lcp"]] + turn["new"]
                await self.request(session, lane, trace, instance, index, turn,
                                   hash_ids, seen, phase_of)
                prev_ids = hash_ids
            self.prompts.forget(trace, instance)
            trace, start, instance = self.next_spare(lane), 0, instance + 1

    async def run(self, out_dir: Path) -> dict:
        import aiohttp
        args = self.args
        metrics_url = f"http://127.0.0.1:{args.port}/metrics"
        t0 = _scrape(metrics_url, out_dir / "metrics_start.prom")
        t_measure = t0 + args.warmup_s
        t_stop = t_measure + args.measure_s
        marks = {"t_start": t0, "t_measure": t_measure, "t_stop": t_stop}

        def phase_of(t):
            return "warmup" if t < t_measure else "measure"

        async def measure_marker():
            await asyncio.sleep(max(0.0, t_measure - time.time()))
            marks["t_measure_scrape"] = await asyncio.to_thread(
                _scrape, metrics_url, out_dir / "metrics_measure_start.prom")

        timeout = aiohttp.ClientTimeout(total=None, sock_read=args.sock_read_timeout_s)
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            marker = asyncio.create_task(measure_marker())
            tasks = [asyncio.create_task(self.lane(session, i, t_stop, phase_of))
                     for i in range(args.concurrency)]
            done, pending = await asyncio.wait(
                tasks, timeout=max(0.0, t_stop - time.time()) + args.grace_s,
                return_when=asyncio.FIRST_EXCEPTION)
            failed = [t for t in done if t.exception() is not None]
            marks["lanes_cancelled_after_grace"] = 0 if failed else len(pending)
            for task in list(pending) + ([marker] if failed else []):
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if failed:
                await asyncio.gather(marker, return_exceptions=True)
                raise failed[0].exception()
            await marker
        marks["t_end"] = time.time()
        marks["t_end_scrape"] = _scrape(metrics_url, out_dir / "metrics_end.prom")
        return marks

    def summary(self, marks: dict) -> dict:
        measured = [r for r in self.records if r["phase"] == "measure"]
        ok = [r for r in measured if r["status"] == "ok"]
        cancelled = [r for r in measured if r["status"] == "cancelled"]

        def by(key, rows):
            return _percentiles([r[key] for r in rows if r[key] is not None])
        span = self.args.measure_s
        return {
            "marks": marks, "concurrency": self.args.concurrency,
            "requests": {"total": len(self.records), "measure": len(measured),
                         "measure_ok": len(ok), "measure_cancelled": len(cancelled),
                         "warmup": len(self.records) - len(measured)},
            "ttft_s": by("ttft_s", ok), "tpot_s": by("tpot_s", ok), "e2e_s": by("e2e_s", ok),
            "prompt_tokens": by("prompt_tokens", ok),
            "cached_tokens_per_request": by("cached_tokens", ok),
            "measure_ok_tokens": {
                "prompt": sum(r["prompt_tokens"] for r in ok),
                "cached": sum(r["cached_tokens"] for r in ok),
                "reusable_prefix": sum(r["reusable_prefix_tokens"] for r in ok),
                "completion": sum(r["completion_tokens"] for r in ok)},
            "output_tokens_per_s_measure_ok": sum(r["completion_tokens"] for r in ok) / span,
            "lanes_trajectories": sorted({(r["lane"], r["trace"], r["instance"])
                                          for r in self.records})}


def replay(args) -> None:
    workload = json.loads(Path(args.workload).read_text())
    if workload["schema"] != WORKLOAD_SCHEMA:
        raise SystemExit(f"unexpected workload schema {workload['schema']}")
    if args.concurrency > len(workload["lanes"]):
        raise SystemExit(f"C={args.concurrency} > {len(workload['lanes'])} planned lanes")
    if workload["params"]["max_model_len"] != args.max_model_len:
        raise SystemExit(f"workload built for max_model_len "
                         f"{workload['params']['max_model_len']}, server {args.max_model_len}")
    if args.dry_run:
        total = 0
        for plan in workload["lanes"][:args.concurrency]:
            turns = workload["traces"][plan["trace"]]
            ids = _hash_ids_at(turns, plan["start"])
            if BLOCK_TOKENS * len(ids) != turns[plan["start"]]["in"]:
                raise SystemExit(f"lane {plan['lane']}: hash ids do not rebuild the prompt")
            total += turns[plan["start"]]["in"]
            print(f"[lane {plan['lane']}] trace {plan['trace']} start turn "
                  f"{plan['start']}/{plan['turns']} context {turns[plan['start']]['in']} "
                  f"out {turns[plan['start']]['out']} blocks {len(ids)}")
        print(f"[lanes] start context sum {total} tokens")
        print(f"[replay] C={args.concurrency} warmup {args.warmup_s}s measure "
              f"{args.measure_s}s grace {args.grace_s}s think cap {args.think_cap_s}s "
              f"-> POST {args.port}/v1/completions stream ignore_eos temperature 0")
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    runner = Replay(args, workload)
    try:
        marks = asyncio.run(runner.run(out_dir))
    finally:  # keep every request record, also when the replay fails
        with (out_dir / "requests.jsonl").open("w") as handle:
            for rec in sorted(runner.records, key=lambda r: r["t_send"]):
                handle.write(json.dumps(rec) + "\n")
    summary = runner.summary(marks)
    summary["argv"] = sys.argv
    summary["workload"] = {"path": args.workload, "source": workload["source"],
                           "params": workload["params"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("requests", "ttft_s", "tpot_s",
                                              "measure_ok_tokens")}, indent=1))
    if summary["requests"]["measure_ok"] == 0:
        raise SystemExit("no measured request completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--traces", required=True)
    b.add_argument("--expect-sha256", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--max-model-len", type=int, required=True)
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--lanes", type=int, required=True)
    b.add_argument("--spares", type=int, required=True)
    b.add_argument("--start-min-ratio", type=float, required=True)
    b.add_argument("--start-max-ratio", type=float, required=True)
    b.add_argument("--think-cap-s", type=float, required=True,
                   help="only for the printed gap statistics")
    r = sub.add_parser("replay")
    r.add_argument("--workload", required=True)
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--served-model", required=True)
    r.add_argument("--max-model-len", type=int, required=True)
    r.add_argument("--concurrency", type=int, required=True)
    r.add_argument("--nonce", required=True,
                   help="cache-bust seed of the synthetic tokens (same across arms)")
    r.add_argument("--warmup-s", type=float, required=True)
    r.add_argument("--measure-s", type=float, required=True)
    r.add_argument("--grace-s", type=float, required=True)
    r.add_argument("--think-cap-s", type=float, required=True)
    r.add_argument("--sock-read-timeout-s", type=float, required=True)
    r.add_argument("--out-dir")
    r.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cmd == "build":
        build(args)
    else:
        if not args.dry_run and not args.out_dir:
            parser.error("replay needs --out-dir")
        replay(args)


if __name__ == "__main__":
    main()
