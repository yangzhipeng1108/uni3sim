"""Delivery-region probe, run alongside a Serving capture.

The region is the one the serving trace bounds from the step ending in
EngineCore to the client seeing the token.  Three processes reproduce it as the
deployment runs it: one builds the step's EngineCoreOutputs and encodes them
with the runtime's MsgpackEncoder over ZeroMQ, the API server decodes on one
event loop and runs the deployment's own chain (OutputProcessor.process_outputs
-> RequestOutputCollector -> the per-request generate task -> the completion
stream generator), and a third runs the client the capture itself uses, aiohttp
with vLLM's own StreamedResponseHandler, every stream on one event loop.

Why this is a published runner and not an evidence harness any more: the host
delivery path drifts between days by far more than it varies within a session.
Measured on the same protocol, A800's marginal term ran 52.250 / 47.970 /
71.846 us/output on the publication day and 41.128 / 42.130 / 41.498 later --
about 25% apart -- while five back-to-back repeats inside one session spread
only 3%.  Decomposed, the cross-session spread is 23.68% true drift against
3.03% measurement noise; on C500 it is 4.82% against 3.83%, on 910B 5.85%
against 1.27%.

So a probe value measured days away from a capture cannot be compared with that
capture: the difference is dominated by machine state, not by the quantity being
measured.  Running the probe immediately before and after the capture puts both
on the same machine state, and the drift cancels in the comparison.  That is the
point of wiring it into the capture CLI rather than leaving it to be run by hand.

Estimators are unchanged and deliberately so: of the three tried, the two-point
slope and its extrapolated intercept were the most repeatable (least squares
over all fanouts and taking n=1 directly were both worse, because the noise is
concentrated at the low fanouts that those two lean on).

Each fanout's whole setup is repeated and the median taken: the per-fanout
value is already a median over its rounds, so the spread between invocations
lives in the setup -- spawning the processes, binding the socket, where the
scheduler puts them -- and only repeating the setup reaches it.  Measured over
five repeats per card, that takes the marginal term's within-session CV from
3.03% to 1.14% on A800, 3.83% to 1.25% on C500 and 1.27% to 0.81% on 910B,
leaving the single-output term unchanged (the difference there, 0.94x to 1.27x,
is inside the sampling uncertainty of a CV estimated from four points, about
41%).  That is what a bracket needs: it compares its two halves inside one
session, so within-session precision is exactly its resolving power.

The first setup of an invocation is measured and discarded.  It runs
consistently high -- as the inflated first idle half of every paired cycle, as
C500's first value in the five-repeat series, and in this probe's own smoke
runs -- and it would otherwise land on the low fanouts the intercept is built
from.  Repeat counts should be odd, because an even count makes the median
average its two middle values and lets a single outlier through.

Fanout budget is deliberately uniform.  Concentrating it on the two fanouts the
estimator actually reads was tried and rejected: it improved the intercept about
3x on the two quiet hosts but degraded the slope 1.5-1.6x on them and failed
outright on A800.  The fanouts that never enter the estimator are not waste --
they ramp the system up to the large fanouts, and removing or thinning them
destabilises the two that do.

The published delivery parameters were produced with one setup per fanout and
are NOT re-measured here: their limiting uncertainty is day-to-day drift, not
setup noise, so repeating the setup would tighten them by 1.01x on two of three
cards.  Pass 1 as the repeat count to reproduce that protocol exactly.

Usage: vllm_delivery_region_runner.py MODEL FANOUTS ROUNDS [REPEATS] [JSON_OUT] [HISTORY_ROUNDS]
"""

import asyncio, inspect, json, os, socket as sk, statistics, subprocess, sys, threading, time
from pathlib import Path
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")


def _handler_class():
    from vllm.benchmarks.lib.endpoint_request_func import StreamedResponseHandler
    return StreamedResponseHandler


if "--engine" in sys.argv:
    import zmq, time as _t
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
    from vllm.v1.serial_utils import MsgpackEncoder
    addr = os.environ["MID_OUT"]; F = int(os.environ["MID_F"])
    R = int(os.environ["MID_R"]); T = int(os.environ["MID_T"])
    enc = MsgpackEncoder(); c = zmq.Context(io_threads=1)
    s = c.socket(zmq.PUSH); s.connect(addr); _t.sleep(1.2)
    for _ in range(R):
        o = EngineCoreOutputs(engine_index=0, outputs=[
            EngineCoreOutput(request_id=f"cmpl-request-{i:012d}-0",
                             new_token_ids=[T], finish_reason=None)
            for i in range(F)])
        t0 = _t.perf_counter_ns(); p = enc.encode(o)
        fr = list(p) if isinstance(p, (list, tuple)) else [p]
        s.send_multipart([t0.to_bytes(8, "little")] + fr); _t.sleep(0.04)
    s.send_multipart([b""]); s.close(linger=2000); c.term(); sys.exit(0)

if "--client" in sys.argv:
    import aiohttp
    H = _handler_class(); PORT = int(os.environ["MID_PORT"])
    F = int(os.environ["MID_F"]); R = int(os.environ["MID_R"])
    async def one(sess, i, out):
        h = H()
        async with sess.post(f"http://127.0.0.1:{PORT}/v1/completions",
                             json={"model": "probe", "prompt": "x", "stream": True}) as r:
            async for raw in r.content.iter_any():
                raw = raw.strip()
                if not raw:
                    continue
                for msg in h.add_chunk(raw):
                    if msg.startswith(":"):
                        continue
                    chunk = msg.removeprefix("data: ")
                    if chunk == "[DONE]":
                        return
                    d = json.loads(chunk)
                    if d.get("choices"):
                        out.append(time.perf_counter_ns())
                    if len(out) >= R:
                        return
    async def main():
        conn = aiohttp.TCPConnector(limit=0)
        outs = [[] for _ in range(F)]
        async with aiohttp.ClientSession(connector=conn) as sess:
            await asyncio.gather(*[one(sess, i, outs[i]) for i in range(F)])
        print(json.dumps(outs))
    asyncio.run(main())
    sys.exit(0)

import uvicorn, zmq, zmq.asyncio
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route
from transformers import AutoTokenizer
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.serial_utils import MsgpackDecoder


def _resp_classes():
    for mod in ("vllm.entrypoints.openai.completion.protocol",
                "vllm.entrypoints.openai.protocol"):
        try:
            m = __import__(mod, fromlist=["CompletionStreamResponse"])
        except ImportError:
            continue
        if hasattr(m, "CompletionStreamResponse"):
            return m.CompletionStreamResponse, m.CompletionResponseStreamChoice
    raise SystemExit("streaming response classes absent")


MODEL = sys.argv[1]; FANOUTS = [int(x) for x in sys.argv[2].split(",")]
R = int(sys.argv[3]); WARM = int(sys.argv[6]) if len(sys.argv) > 6 else 5
if WARM < 5:
    raise SystemExit("HISTORY_ROUNDS must be at least five")
# Repeats are specified per fanout ("2:14,64:7") because the two-point
# estimator reads only two of the measured fanouts and they do not deserve
# equal budget.  ic = v2 - (v64-v2)/62, so d(ic)/d(v2) = 1.016 against
# d(ic)/d(v64) = -0.016: the intercept is v2 alone, 63x over.  The slope has
# equal leverage on both, so its budget follows the absolute spread, which is
# comparable on A800 and v64-dominated on C500.  A plain integer keeps the
# uniform behaviour for comparison runs.
_spec = sys.argv[4] if len(sys.argv) > 4 else "3"
if ":" in _spec:
    REPEAT_BY_FANOUT = {int(k): int(v) for k, v in
                        (p.split(":") for p in _spec.split(","))}
    REPEATS = None
else:
    REPEAT_BY_FANOUT = {}
    REPEATS = int(_spec)
CompletionStreamResponse, CompletionResponseStreamChoice = _resp_classes()
tok = AutoTokenizer.from_pretrained(MODEL)
ids = list(tok("delivery probe")["input_ids"])
TOKEN = int(ids[0])
dec = MsgpackDecoder(EngineCoreOutputs)


def mkreq(i):
    kw = dict(request_id=f"cmpl-request-{i:012d}-0", prompt_token_ids=list(ids),
              sampling_params=SamplingParams(max_tokens=4096,
                                             output_kind=RequestOutputKind.DELTA),
              eos_token_id=tok.eos_token_id, arrival_time=time.time(),
              mm_features=None, lora_request=None, cache_salt=None,
              data_parallel_rank=None, client_index=0, current_wave=0, priority=0,
              trace_headers=None, prompt_embeds=None, mm_hashes=None, mm_inputs=None,
              mm_placeholders=None, pooling_params=None,
              external_req_id=f"cmpl-request-{i:012d}-0")
    sig = set(inspect.signature(EngineCoreRequest).parameters)
    return EngineCoreRequest(**{k: v for k, v in kw.items() if k in sig})


collectors = {}
ready = threading.Semaphore(0)
loopbox = []


async def completions(request):
    if not loopbox:
        loopbox.append(asyncio.get_running_loop())
    i = len(collectors)
    # 0.18.0 and 0.24.0 added a request_id argument to the collector; the call
    # is built from the signature so a further change shows up rather than
    # silently taking a different path.
    coll_params = list(inspect.signature(RequestOutputCollector.__init__).parameters)
    coll_kw = {}
    if "request_id" in coll_params:
        coll_kw["request_id"] = f"cmpl-request-{i:012d}-0"
    coll = RequestOutputCollector(RequestOutputKind.DELTA, **coll_kw)
    collectors[i] = coll
    ready.release()

    async def generate():
        """Stands in for AsyncLLM.generate: awaits the collector, yields out."""
        while True:
            out = await coll.get()
            if out is None:
                return
            yield out

    async def body():
        """Stands in for completion_stream_generator: frames what it yields."""
        async for out in generate():
            text = out.outputs[0].text if out.outputs else " "
            ch = CompletionStreamResponse(
                id=f"cmpl-request-{i:012d}-0", created=1758000000, model="probe",
                choices=[CompletionResponseStreamChoice(
                    index=0, text=text or " ", logprobs=None, finish_reason=None)])
            yield ("data: " + ch.model_dump_json(exclude_unset=True) + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


app = Starlette(routes=[Route("/v1/completions", completions, methods=["POST"])])
with sk.socket() as s:
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
threading.Thread(target=srv.run, daemon=True).start()
while not srv.started:
    time.sleep(0.01)

res = {}
_warmed = []
for F in FANOUTS:
    setups = []
    _k = REPEAT_BY_FANOUT.get(F, REPEATS)
    if _k is not None and not _warmed:
        _k += 1        # one extra, consumed by the discarded warm-up
    if _k is None:
        raise SystemExit(f'no repeat count given for fanout {F}')
    for _rep in range(_k):
        collectors.clear()
        # OutputProcessor construction differs across runtimes; both forms tried.
        proc = None
        for build in (lambda: OutputProcessor(tok, log_stats=False),
                      lambda: OutputProcessor(tok, False)):
            try:
                proc = build(); break
            except TypeError:
                continue
        if proc is None:
            raise SystemExit("OutputProcessor construction failed")
        env = dict(os.environ); env.update(MID_PORT=str(port), MID_F=str(F), MID_R=str(R + WARM))
        cli = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--client"],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(F):
            if not ready.acquire(timeout=120):
                raise RuntimeError("streams did not open")
        if not loopbox:
            raise RuntimeError("server loop was not captured")
        # add_request's signature moves across runtimes (0.11.0 / 0.18.0 / 0.24.0),
        # so the call is built from the signature rather than assumed positional.
        add_sig = list(inspect.signature(proc.add_request).parameters)
        for i in range(F):
            kw = {"prompt": None, "parent_req": None, "request_index": 0,
                  "queue": collectors[i]}
            proc.add_request(mkreq(i), **{k: v for k, v in kw.items() if k in add_sig})
        addr = f"ipc:///tmp/mid-d7-{os.getpid()}-{F}-{_rep}"
        env2 = dict(os.environ); env2.update(MID_OUT=addr, MID_F=str(F),
                                             MID_R=str(R + WARM), MID_T=str(TOKEN))
        ctx = zmq.asyncio.Context(io_threads=1); rx = ctx.socket(zmq.PULL); rx.bind(addr)
        eng = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--engine"],
                               env=env2, stdout=subprocess.DEVNULL)
        sent = []

        async def pump():
            while True:
                fr = await rx.recv_multipart()
                if len(fr) < 2:
                    return
                sent.append(int.from_bytes(fr[0], "little"))
                proc.process_outputs(dec.decode(fr[1:]).outputs)

        fut = asyncio.run_coroutine_threadsafe(pump(), loopbox[0])
        # Budget scales with the work asked for: a fanout opens F streams and
        # each round sleeps, and a loaded host stretches both.  A fixed 180s
        # was exceeded at F=64 on a host at load 10, which failed the whole
        # invocation -- and with the probe wired into every capture, that is a
        # condition to absorb rather than to fail on.
        _budget = (R + WARM) * 0.25 + 6.0 * F + 240
        fut.result(timeout=_budget)
        eng.wait(timeout=max(60, _budget / 4))
        for coll in list(collectors.values()):
            coll.put(None)
        out, err = cli.communicate(timeout=_budget)
        got = json.loads(out.strip().splitlines()[-1])
        n = min(min(len(v) for v in got), len(sent))
        last = [max(got[i][r] for i in range(F)) - sent[r] for r in range(WARM, n)]
        _value = statistics.median(last) / 1e6
        if not _warmed:
            # The first setup of an invocation is consistently high: the same
            # warm-up shows up as C500's first marginal in the five-repeat
            # series and as the inflated first idle half of every paired cycle.
            # Here it would land on v2, the point the intercept is made of, so
            # it is measured and discarded rather than left to bias the budget
            # that was deliberately concentrated there.
            _warmed.append(_value)
            print("  n=%3d  warm-up setup %.4f ms (discarded)" % (F, _value),
                  flush=True)
            continue
        setups.append(_value)
    res[F] = statistics.median(setups)
    print("  n=%3d  %8.4f ms  (median of %d setups: %s)"
          % (F, res[F], len(setups), " ".join("%.4f" % s for s in setups)),
          flush=True)

def derive_terms(per_fanout):
    """Single-output and marginal terms from the per-fanout medians.

    Both are host costs of carrying tokens out, so both are positive by
    construction.  A non-positive value means the measurement did not resolve
    the region -- too short a lever between the two fanouts the estimator
    reads, or a disturbance large enough to invert them -- and must not reach a
    caller that would treat it as a delivery cost.  This is checked here rather
    than at the print, because the JSON artifact makes the value something a
    downstream reader consumes rather than a line in a log.
    """

    fit = sorted((k, v) for k, v in per_fanout.items() if k >= 2)
    if len(fit) < 2:
        raise SystemExit("need at least two fanouts >= 2 to derive the terms")
    slope = (fit[-1][1] - fit[0][1]) / (fit[-1][0] - fit[0][0])
    intercept = fit[0][1] - slope * (fit[0][0] - 1)
    if intercept <= 0 or slope <= 0:
        raise SystemExit(
            "delivery region not resolved: single %.4f ms, marginal %.3f us/out; "
            "per-fanout medians %s"
            % (intercept, slope * 1000,
               {k: round(v, 4) for k, v in sorted(per_fanout.items())}))
    return intercept, slope


ic, sl = derive_terms(res)
print("RESULT single %.4f ms  marginal %.3f us/out" % (ic, sl * 1000))
if len(sys.argv) > 5:
    import datetime
    import platform
    Path(sys.argv[5]).write_text(json.dumps({
        "schema_version": 1,
        "measurement_boundary": "host_only_no_device_no_model",
        "captured_at": datetime.datetime.now().astimezone().isoformat(),
        "host": {"node": platform.node(), "cpu_count": os.cpu_count()},
        "tokenizer_path": MODEL,
        "protocol": {"fanouts": FANOUTS, "rounds": R, "warmup": WARM},
        "per_fanout_ms": {str(k): v for k, v in sorted(res.items())},
        "first_token_delivery_s": ic / 1e3,
        "output_delivery_per_output_s": sl / 1e3,
    }, indent=2, sort_keys=True) + "\n")
    print("WROTE %s" % sys.argv[5])

