"""Real-serving TP8 capture on 8x H200 with vLLM 0.30 (dummy weights).

The caller supplies --runtime-python; child server scripts receive the same
executable through MIDLAYER_H200_RUNTIME_PYTHON.

Shared orchestrator behind the per-model entry points, which only pick a
``ModelProfile``:

* ``nvidia_h200_glm_5_2_tp8_capture_runner.py``: GLM-5.2-FP8, MTP off.
* ``nvidia_h200_deepseek_r1_tp8_capture_runner.py``: DeepSeek-R1-0528 FP8, MTP off / on.

Modeled on the DSV3-L32 TP4 H200 captures (``nvidia_h200_nsys_pure_decode_capture.sh``,
``nvidia_h200_nsys_prefill_capture.sh``, and the remote ``sweep_closedloop.sh``):

* ``smoke``: start the server (same flags as the real run, no nsys), check the
  startup facts (KV capacity, attention / MoE / all-reduce backends, routing
  simulation active), send one completion request, measure server start ->
  ready and request -> response, shut down, verify all eight GPUs are free.
  Also warms the DeepGEMM / FlashInfer / Triton JIT caches used by ``run``.
* ``run --phases closedloop``: InferenceX-style 8K/1K closed loop without
  nsys: ``vllm bench serve`` random dataset, input 7373 +-11.1% (6554..8192),
  output 922 +-11.1% (820..1024), ``--ignore-eos``, ``--max-concurrency C``,
  ``10*C`` measured prompts after ``2*C`` warm-ups, per C in ``--concurrency``.
* ``run --phases nsys``: one server under ``nsys --capture-range=cudaProfilerApi
  --cuda-graph-trace=node`` (vLLM ``--profiler-config '{"profiler": "cuda"}'``,
  windows through ``/start_profile`` / ``/stop_profile``).  Per C one
  pure-decode window (exactly C requests, captured after all C prefills have
  produced their first token -- checked on the TTFT histogram count -- and
  before the shortest request, >= 820 output tokens, can finish) and one
  prefill-only window (C closed-loop clients, fixed 7373-token prompts, one
  output token, window starts 3 s into the main run).  Windows are 1.5 s.
  ``windows.txt`` uses the DSV3 ``<label> start|stop <time>`` format so
  ``summarize_nvidia_h200_nsys_decode_windows.py`` can read it; ``windows.json``
  adds unix times, metric deltas (purity checks) and clock flags.

* ``run --phases offload`` (profiles with an ``OffloadGroup``; runs alone,
  MTP off): KV host-offload group.  One server per ``--offload-arms`` entry
  (prefix caching on, long ``--max-model-len``; arms differ only in the KV
  connector), each replaying an AgentX-derived multi-turn workload
  (``agentx_session_replay.py``) at every ``--concurrency`` value (session
  lanes).  Per point: full ``/metrics`` before / at measure start / at end,
  GPU vs host (KV connector) prefix hits, offload transfer bytes, per-request
  TTFT/TPOT/E2E and cached tokens, clock flags over the measured window.  Fails
  when the arm's connector is not the one configured (server.log) or when an
  offload arm has no host prefix hit in any point.

Profiles with an MTP option start one server per ``--mtp`` value and phase;
the order is every closed loop first (ground truth), then every nsys phase.

Every GPU use must hold the eight whole-GPU locks; launch through
``microbench/nvidia_h200/kcampaign/remote/with_gpu_locks.sh 0,1,2,3,4,5,6,7
<wait> -- <entry point> ...``.  The script verifies that it inherited the
lock fds (with_gpu_locks.sh opens them as fds 200..207) and that no compute
process is on any GPU before it starts.  SM clocks cannot be locked: an
``nvidia-smi`` clock log is taken for the whole job and every closed-loop C
and every window reports its busy samples at a clock other than 1980 MHz
(flag only; nothing is dropped).

``--dry-run`` prints every command and output path without launching any
process, touching a GPU or calling nvidia-smi.

Evaluation-only measurements; no timing feeds any model parameter.
"""

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CLOCK_SUMMARIZER = (ROOT / "microbench" / "nvidia_h200" / "kcampaign" / "remote"
                    / "summarize_clock_log.py")

GPUS = tuple(range(8))
WRAPPER_FIRST_FD = 200  # with_gpu_locks.sh opens gpu locks at fds 200, 201, ...
EXPECTED_SM_MHZ = 1980
EXPECTED_DEVICE = "NVIDIA H200"

ISL_MEAN = 7373
OSL_MEAN = 922
RANGE_RATIO = "0.1111111111"  # InferenceX [0.8*L, L] as vLLM's symmetric ratio
MAX_REQUEST_TOKENS = 8192 + 1024  # longest input + longest output
PROMPTS_PER_CLIENT = 10
WARMUPS_PER_CLIENT = 2
WINDOW_S = 1.5
PREFILL_WINDOW_DELAY_S = 3.0
PREFILL_MIN_PROMPTS = 400
SEED = 0

STARTUP_TIMEOUT_S = 2400
DECODE_GATE_TIMEOUT_S = 900
PREFILL_GATE_TIMEOUT_S = 900
IDLE_TIMEOUT_S = 600
STOP_PROFILE_TIMEOUT_S = 1800
# Last window: nsys (--capture-range-end=repeat-shutdown:N) flushes the report
# and then stops the engine, so /stop_profile may never be answered while
# uvicorn waits for that very connection before exiting (2026-09-27 10:31 hang).
# Give up on the reply after this long, then wait for nsys to exit.
LAST_STOP_PROFILE_TIMEOUT_S = 300
LAST_WINDOW_EXIT_TIMEOUT_S = 600
NSYS_EXIT_TIMEOUT_S = 2400
CLOSEDLOOP_TIMEOUT_S = 3600
GPU_RELEASE_TIMEOUT_S = 180

CLOCK_QUERY = ("timestamp,index,clocks.sm,clocks.mem,utilization.gpu,"
               "power.draw,temperature.gpu,clocks_event_reasons.active")
CLOCK_PERIOD_MS = 200

BACKEND_PATTERNS = (
    r"Initializing a V1 LLM engine",
    r"attention backend",
    r"prefill backend",
    r"all-reduce backends",
    r"FlashInfer Allreduce",
    r"Fp8 MoE backend",
    r"Selected \w+ for ",
    r"BREAKABLE_CUDAGRAPH",
    r"[Cc]udagraph mode",
    r"Simulating MoE routing",
    r"KV cache size",
    r"Graph capturing finished",
    r"init engine",
)
LOG_PREFIX = re.compile(r"^\([^)]*\)\s+[A-Z]+\s+\S+\s+\S+\s+\[[^\]]+\]\s+(.*)$")
METRICS = ("vllm:num_requests_running", "vllm:num_requests_waiting",
           "vllm:time_to_first_token_seconds_count",
           "vllm:request_success_total", "vllm:num_preemptions_total",
           "vllm:kv_cache_usage_perc")
# vLLM 0.30 vllm/v1/spec_decode/metrics.py counters (Prometheus adds _total).
SPEC_METRICS = ("vllm:spec_decode_num_drafts_total",
                "vllm:spec_decode_num_draft_tokens_total",
                "vllm:spec_decode_num_accepted_tokens_total")


@dataclass(frozen=True)
class OffloadArm:
    """One server configuration of the KV host-offload group."""
    serve_args: tuple          # extra `vllm serve` arguments
    use_simple_kv_offload: str # VLLM_USE_SIMPLE_KV_OFFLOAD ("0" | "1")
    connector: str             # expected "Creating v1 connector with name: <X>", or ""
    activation_patterns: tuple # server.log regexes that must all match
    host_loads_required: bool  # fail when no host (external) prefix hit happens


@dataclass(frozen=True)
class OffloadGroup:
    """KV host-offload group (AgentX-derived multi-turn replay, MTP off)."""
    serve_sh: Path
    max_model_len: int
    arms: dict                 # arm name -> OffloadArm, in the default run order
    replay_script: Path
    warmup_s: float
    measure_s: float
    grace_s: float
    think_cap_s: float
    sock_read_timeout_s: float
    server_extra_patterns: tuple


@dataclass(frozen=True)
class ModelProfile:
    """Everything that differs between the models an entry point captures."""
    key: str
    log_tag: str               # log prefix and output-directory stem
    doc: str                   # argparse description (the entry point's __doc__)
    serve_sh: Path
    model_dir: Path
    served_model: str
    port: int
    expected_config: dict      # config.json fields that must match exactly
    kv_cache_dtypes: tuple
    kv_cache_dtype_help: str
    env_prefix: str            # <prefix>_KV_CACHE_DTYPE / _PORT [/ _GPU_MEMORY_UTILIZATION]
    gpu_memory_utilization_arg: bool
    speculative_config_mtp_on: dict   # {} -> the profile has no --mtp option
    separate_nsys_concurrency: bool   # --nsys-concurrency instead of --concurrency
    bench_sampling_args: tuple        # appended to every vllm bench serve call
    extra_backend_patterns: tuple
    extended_record: bool             # model profile / MTP / selection summary in records
    sigterm_cleanup: bool             # SIGTERM -> SystemExit, so the server is stopped
    offload: object                   # OffloadGroup, or None: no `run --phases offload`


P = None  # the active ModelProfile, set once by main()


# ----------------------------------------------------------------- helpers

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")


def _log(message: str) -> None:
    print(f"[{P.log_tag} {datetime.now():%F %T}] {message}", flush=True)


def _save(record: dict, path: Path) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")


def _scripts() -> dict:
    return {p.name: _sha256(p) for p in (Path(sys.argv[0]).resolve(),
                                         Path(__file__).resolve(), P.serve_sh)}


def _has_mtp() -> bool:
    return bool(P.speculative_config_mtp_on)


def _num_spec_tokens(mtp) -> int:
    if mtp is None or mtp == "off":
        return 0
    if mtp == "on":
        return P.speculative_config_mtp_on["num_speculative_tokens"]
    raise ValueError(f"unknown mtp setting {mtp!r}")


def _serve_args(mtp) -> list:
    if mtp is None or mtp == "off":
        return []
    if mtp == "on":
        return ["--speculative-config",
                json.dumps(P.speculative_config_mtp_on, separators=(",", ":"))]
    raise ValueError(f"unknown mtp setting {mtp!r}")


def _metric_names(mtp) -> tuple:
    return METRICS + (SPEC_METRICS if _num_spec_tokens(mtp) else ())


def _dir_suffix(mtp) -> str:
    return "" if mtp is None else f"mtp{mtp}_"


def _record_extra(args, mtp) -> dict:
    if not P.extended_record:
        return {}
    extra = {"model_profile": P.key, "mtp": mtp,
             "speculative_config": (P.speculative_config_mtp_on if mtp == "on"
                                    else None),
             "bench_sampling_args": list(P.bench_sampling_args)}
    if P.gpu_memory_utilization_arg:
        extra["gpu_memory_utilization"] = args.gpu_memory_utilization
    return extra


# Live child process groups by role, so a failure never leaves a server on
# the GPUs after with_gpu_locks.sh has released the locks.
_LIVE = {}


def _register(role: str, proc) -> None:
    if role in _LIVE:
        raise RuntimeError(f"a {role} process is already live")
    _LIVE[role] = proc


def _unregister(role: str) -> None:
    del _LIVE[role]


def _cleanup() -> None:
    for role in ("bench", "server", "clock"):
        if role not in _LIVE:
            continue
        proc = _LIVE[role]
        if proc.poll() is None:
            _log(f"cleanup after failure: stopping {role} (pgid {proc.pid})")
            if role == "server":
                _stop_server(proc, 120)
            elif role == "bench":
                _stop_bench(proc)
            else:
                _stop_clock_log(proc)
        del _LIVE[role]


class Plan:
    """Either executes or, with --dry-run, only prints what it would do."""

    def __init__(self, dry_run: bool):
        self.dry_run = dry_run

    def show(self, label: str, cmd: list, env_extra: dict) -> None:
        env_text = " ".join(f"{k}={v}" for k, v in sorted(env_extra.items()))
        print(f"[{label}] {env_text} {' '.join(str(c) for c in cmd)}".strip(),
              flush=True)


def _base_env() -> dict:
    env = dict(os.environ)
    env.update({
        "HOME": str(MICRO / "home"), "XDG_CACHE_HOME": str(B / "cache" / "xdg"),
        "HF_HOME": str(B / "cache" / "hf"), "HF_HUB_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1",
    })
    return env


def _server_env_extra(args) -> dict:
    extra = {"MIDLAYER_H200_PUBLIC_ROOT": str(B),
             "MIDLAYER_H200_RUNTIME_PYTHON": str(args.runtime_python),
             "VLLM_MOE_ROUTING_SIMULATION_STRATEGY": args.routing,
             f"{P.env_prefix}_KV_CACHE_DTYPE": args.kv_cache_dtype,
             f"{P.env_prefix}_PORT": str(P.port)}
    if P.gpu_memory_utilization_arg:
        extra[f"{P.env_prefix}_GPU_MEMORY_UTILIZATION"] = args.gpu_memory_utilization
    return extra


def _client_env_extra() -> dict:
    # The load generator and probes never need a GPU.
    return {"CUDA_VISIBLE_DEVICES": ""}


def _smi(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], check=True,
                          capture_output=True, text=True).stdout


# ---------------------------------------------------------------- preflight

def _check_model_dir() -> dict:
    config = json.loads((P.model_dir / "config.json").read_text())
    for key, want in P.expected_config.items():
        if config[key] != want:
            raise RuntimeError(f"unexpected {key} {config[key]}")
    quant = config["quantization_config"]
    if (quant["quant_method"], quant["weight_block_size"]) != ("fp8", [128, 128]):
        raise RuntimeError(f"unexpected quantization_config {quant}")
    weights = sorted(p.name for p in P.model_dir.glob("*.safetensors"))
    if weights:
        raise RuntimeError(f"config-only dir unexpectedly has weights {weights}")
    return {"path": str(P.model_dir),
            "config_sha256": _sha256(P.model_dir / "config.json"),
            "tokenizer_sha256": _sha256(P.model_dir / "tokenizer.json"),
            "model_type": config["model_type"],
            "num_hidden_layers": config["num_hidden_layers"],
            "quantization_config": {k: quant[k] for k in
                                    ("quant_method", "fmt", "activation_scheme",
                                     "weight_block_size")}}


def _assert_holding_gpu_locks() -> dict:
    held = {}
    for offset, gpu in enumerate(GPUS):
        fd = WRAPPER_FIRST_FD + offset
        fd_path = Path(f"/proc/self/fd/{fd}")
        want = (LOCK_DIR / f"gpu{gpu}.proc.lock").resolve()
        if not fd_path.exists():
            raise RuntimeError(
                f"fd {fd} (gpu{gpu} lock) not inherited: launch through "
                "with_gpu_locks.sh 0,1,2,3,4,5,6,7 <wait> -- ...")
        target = Path(os.readlink(fd_path)).resolve()
        if target != want:
            raise RuntimeError(f"fd {fd} is {target}, expected {want}")
        held[str(gpu)] = str(target)
    return held


def _preflight(out_dir: Path) -> dict:
    rows = [line.split(", ") for line in _smi(
        "--query-gpu=index,name,uuid,pci.bus_id,driver_version,clocks.max.sm,"
        "clocks.sm,memory.total", "--format=csv,noheader,nounits").splitlines()]
    indices = tuple(int(r[0]) for r in rows)
    if indices != GPUS:
        raise RuntimeError(f"expected GPUs {GPUS}, nvidia-smi shows {indices}")
    for r in rows:
        if r[1] != EXPECTED_DEVICE:
            raise RuntimeError(f"GPU {r[0]} is {r[1]!r}, expected {EXPECTED_DEVICE}")
    apps = _smi("--query-compute-apps=gpu_bus_id,pid,used_memory",
                "--format=csv,noheader").strip()
    if apps:
        raise RuntimeError(f"compute processes already on the GPUs:\n{apps}")
    probe = socket.socket()
    port_busy = probe.connect_ex(("127.0.0.1", P.port)) == 0
    probe.close()
    if port_busy:
        raise RuntimeError(f"port {P.port} already in use")
    (out_dir / "topo.txt").write_text(_smi("topo", "-m"))
    (out_dir / "clocks_before.txt").write_text(_smi("-q", "-d", "CLOCK"))
    gpus = [{"index": int(r[0]), "name": r[1], "uuid": r[2], "pci_bus_id": r[3],
             "driver_version": r[4], "clocks_max_sm_mhz": int(r[5]),
             "clocks_sm_mhz_idle": int(r[6]), "memory_total_mib": int(r[7])}
            for r in rows]
    max_off = [g["index"] for g in gpus if g["clocks_max_sm_mhz"] != EXPECTED_SM_MHZ]
    if max_off:
        _log(f"FLAG: max SM clock != {EXPECTED_SM_MHZ} MHz on GPUs {max_off}")
    return {"gpus": gpus, "max_sm_clock_not_1980": max_off,
            "gpu_locks_inherited": _assert_holding_gpu_locks()}


def _software_versions() -> dict:
    code = ("import importlib.metadata as m, json; print(json.dumps({p: m.version(p) "
            "for p in ('vllm', 'torch', 'flashinfer-python', 'nvidia-nccl-cu12', "
            "'transformers', 'triton')}))")
    env = _base_env()
    env.update(_client_env_extra())
    versions = json.loads(subprocess.run(
        [str(PY), "-c", code], check=True, capture_output=True, text=True,
        env=env).stdout)
    versions["nsys"] = subprocess.run([str(NSYS), "--version"], check=True,
                                      capture_output=True, text=True).stdout.strip()
    versions["python_orchestrator"] = platform.python_version()
    return versions


def _begin_record(args, mode: str, mtp, extra: dict) -> dict:
    """Common record head (same keys and order for every mode)."""
    record = {"schema_version": 1, "mode": mode, "argv": sys.argv,
              "source_commit": args.source_commit, "routing": args.routing,
              "kv_cache_dtype": args.kv_cache_dtype, "port": P.port}
    record.update(_record_extra(args, mtp))
    record.update(extra)
    return record


def _prepare_real_run(record: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    record["model"] = _check_model_dir()
    record["preflight"] = _preflight(out_dir)
    record["software"] = _software_versions()
    record["scripts"] = _scripts()


# ------------------------------------------------------------- clock log

def _start_clock_log(plan: Plan, out_dir: Path):
    cmd = ["nvidia-smi", f"--query-gpu={CLOCK_QUERY}", "--format=csv,noheader,nounits",
           "-lms", str(CLOCK_PERIOD_MS)]
    plan.show("clock-log", cmd + [">", str(out_dir / "clocks.csv")], {})
    if plan.dry_run:
        return None
    handle = (out_dir / "clocks.csv").open("w")
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                            start_new_session=True)
    _register("clock", proc)
    return proc


def _stop_clock_log(proc) -> None:
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=30)
    if "clock" in _LIVE:
        _unregister("clock")


def _clock_samples(out_dir: Path) -> list:
    samples = []
    with (out_dir / "clocks.csv").open() as handle:
        for row in csv.reader(handle):
            if len(row) != 8:
                continue  # a partial last line while nvidia-smi is writing
            stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            samples.append({"t": stamp.timestamp(), "gpu": int(row[1]),
                            "sm_mhz": int(row[2]), "util": int(row[4])})
    return samples


def _clock_flags(out_dir: Path, t0: float, t1: float) -> dict:
    inside = [s for s in _clock_samples(out_dir) if t0 <= s["t"] <= t1]
    per_gpu = {}
    for gpu in GPUS:
        mine = [s for s in inside if s["gpu"] == gpu]
        busy = [s for s in mine if s["util"] > 0]
        per_gpu[str(gpu)] = {
            "samples": len(mine), "busy": len(busy),
            "busy_non_1980": sum(1 for s in busy if s["sm_mhz"] != EXPECTED_SM_MHZ),
            "busy_sm_mhz": sorted({s["sm_mhz"] for s in busy})}
    flagged = sorted(g for g, v in per_gpu.items() if v["busy_non_1980"])
    return {"interval_unix": [t0, t1], "per_gpu": per_gpu,
            "flag_non_1980": bool(flagged), "flagged_gpus": flagged}


def _summarize_clock_log(out_dir: Path) -> None:
    subprocess.run([sys.executable, str(CLOCK_SUMMARIZER),
                    "--clock-log", str(out_dir / "clocks.csv"),
                    "--gpus", ",".join(str(g) for g in GPUS),
                    "--output", str(out_dir / "clock_summary.json")], check=True)


# ------------------------------------------------------------------ server

def _server_cmd(out_dir: Path, nsys_windows, mtp) -> list:
    serve = ["bash", str(P.serve_sh), *_serve_args(mtp)]
    if nsys_windows is None:
        return serve
    return [str(NSYS), "profile", "-t", "cuda,nvtx", "--cuda-graph-trace=node",
            "--sample=none", "--cpuctxsw=none", "--capture-range=cudaProfilerApi",
            f"--capture-range-end=repeat-shutdown:{nsys_windows}",
            "--trace-fork-before-exec=true", "-o", str(out_dir / "window_%n"),
            "--force-overwrite", "true", *serve,
            "--profiler-config", '{"profiler": "cuda"}']


def _start_server(plan: Plan, args, out_dir: Path, nsys_windows, mtp):
    cmd = _server_cmd(out_dir, nsys_windows, mtp)
    extra = _server_env_extra(args)
    plan.show("server", cmd + [">", str(out_dir / "server.log")], extra)
    if plan.dry_run:
        return None, None
    env = os.environ.copy()
    env.update(extra)
    handle = (out_dir / "server.log").open("w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    _register("server", proc)
    return proc, t0


def _wait_startup(proc, out_dir: Path) -> None:
    deadline = time.time() + STARTUP_TIMEOUT_S
    log = out_dir / "server.log"
    while "Application startup complete" not in log.read_text(errors="replace"):
        if proc.poll() is not None:
            tail = "".join(log.read_text(errors="replace").splitlines(True)[-40:])
            raise RuntimeError(f"server exited with {proc.returncode}:\n{tail}")
        if time.time() > deadline:
            raise RuntimeError(f"server not ready after {STARTUP_TIMEOUT_S}s")
        time.sleep(5)


def _selection(text: str, kv_tokens: int, required: int, args,
               memory_total_mib: int) -> dict:
    """Structured startup selection read from the vLLM 0.30 server log."""
    def found(pattern, source=text):
        return sorted(set(re.findall(pattern, source)))

    engine = "\n".join(line for line in text.splitlines()
                       if "Initializing a V1 LLM engine" in line)
    kv_dtype = found(r"kv_cache_dtype=([\w.]+),", engine)
    if len(kv_dtype) != 1:
        raise RuntimeError(f"expected one engine kv_cache_dtype, found {kv_dtype}")
    attention = found(r"Using (\w+) attention backend out of potential backends")
    if not attention:
        raise RuntimeError("no 'Using <X> attention backend' line in server.log")
    available = [float(v) for v in found(r"Available KV cache memory: ([\d.]+) GiB")]
    if not available:
        raise RuntimeError("no 'Available KV cache memory' line in server.log")
    selection = {
        "engine_kv_cache_dtype": kv_dtype[0],
        "attention_backends": attention,
        "mla_prefill_backends": found(r"Using (\w+) MLA prefill backend"),
        "fp8_moe_backends": found(r"Using (\w+) Fp8 MoE backend"),
        "all_reduce_backends": found(
            r"Using (\[[^\]]*\]) all-reduce backends \(in dispatch order\) "
            r"for group '([^']+)'"),
        "model_loading_gib": [float(v) for v in
                              found(r"Model loading took ([\d.]+) GiB")],
        "available_kv_cache_gib": available,
        "cuda_graph_pool_gib_actual": [float(v) for v in found(
            r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)")],
    }
    if P.gpu_memory_utilization_arg:
        # Utilization at which the KV pool would just hold `required` tokens,
        # assuming everything else stays as measured in this start.
        bytes_per_token = min(available) * 2**30 / kv_tokens
        selection["gpu_memory_utilization_for_required_kv"] = round(
            float(args.gpu_memory_utilization)
            + (required - kv_tokens) * bytes_per_token / (memory_total_mib * 2**20), 4)
    return selection


def _startup_facts(out_dir: Path, max_concurrency: int, mtp, args,
                   memory_total_mib: int) -> dict:
    required = max_concurrency * (MAX_REQUEST_TOKENS + _num_spec_tokens(mtp))
    return _startup_facts_for(out_dir, required, args, memory_total_mib, ())


def _startup_facts_for(out_dir: Path, required: int, args, memory_total_mib: int,
                       more_patterns: tuple) -> dict:
    text = (out_dir / "server.log").read_text(errors="replace")
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
    if not kv:
        raise RuntimeError("no 'GPU KV cache size' line in server.log")
    kv_tokens = min(int(v.replace(",", "")) for v in kv)
    lines = {}
    patterns = BACKEND_PATTERNS + P.extra_backend_patterns + more_patterns
    for line in text.splitlines():
        for pattern in patterns:
            if re.search(pattern, line):
                match = LOG_PREFIX.match(line)
                message = match.group(1) if match else line
                if message not in lines:
                    lines[message] = line
                break
    messages = list(lines)
    if not any("Simulating MoE routing" in m for m in messages):
        raise RuntimeError("routing simulation not active (no 'Simulating MoE "
                           "routing' line); VLLM_MOE_ROUTING_SIMULATION_STRATEGY lost")
    if not any("attention backend" in m for m in messages):
        raise RuntimeError("no attention backend selection line in server.log")
    facts = {"kv_cache_tokens": kv_tokens,
             "kv_tokens_required_for_max_c": required,
             "kv_capacity_ok": kv_tokens >= required,
             "backend_lines": messages}
    if P.extended_record:
        facts["selection"] = _selection(text, kv_tokens, required, args,
                                        memory_total_mib)
    return facts


def _min_memory_total_mib(record: dict) -> int:
    return min(g["memory_total_mib"] for g in record["preflight"]["gpus"])


def _stop_server(proc, graceful_s: int) -> dict:
    """SIGINT the server's process group, escalate, then check the GPUs."""
    steps = []
    for sig, wait_s in ((signal.SIGINT, graceful_s), (signal.SIGTERM, 60),
                        (signal.SIGKILL, 30)):
        if proc.poll() is not None:
            break
        os.killpg(proc.pid, sig)
        steps.append(signal.Signals(sig).name)
        deadline = time.time() + wait_s
        while proc.poll() is None and time.time() < deadline:
            time.sleep(1)
    if proc.poll() is None:
        raise RuntimeError(f"server process group {proc.pid} survived SIGKILL")
    if "server" in _LIVE:
        _unregister("server")
    _wait_gpus_free()
    return {"signals_sent": steps, "returncode": proc.returncode}


def _wait_gpus_free() -> None:
    deadline = time.time() + GPU_RELEASE_TIMEOUT_S
    while True:
        apps = _smi("--query-compute-apps=gpu_bus_id,pid,used_memory",
                    "--format=csv,noheader").strip()
        if not apps:
            return
        if time.time() > deadline:
            raise RuntimeError(f"GPUs still busy after server stop:\n{apps}")
        time.sleep(3)


# ----------------------------------------------------------- HTTP / metrics

def _metrics(names: tuple) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{P.port}/metrics", timeout=10) as r:
        text = r.read().decode()
    sums = {name: 0.0 for name in names}
    seen = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[{ ]", line, maxsplit=1)[0]
        if name in sums:
            sums[name] += float(line.rsplit(" ", 1)[1])
            seen.add(name)
    missing = [n for n in names if n not in seen]
    if missing:
        raise RuntimeError(f"metrics missing from /metrics: {missing}")
    return sums


def _spec_delta(before: dict, after: dict) -> dict:
    """Acceptance over an interval from the vLLM spec-decode counters."""
    drafts = after[SPEC_METRICS[0]] - before[SPEC_METRICS[0]]
    draft_tokens = after[SPEC_METRICS[1]] - before[SPEC_METRICS[1]]
    accepted = after[SPEC_METRICS[2]] - before[SPEC_METRICS[2]]
    if drafts <= 0:
        raise RuntimeError(f"MTP on but no spec-decode drafts counted: {after}")
    return {"drafts": drafts, "draft_tokens": draft_tokens, "accepted_tokens": accepted,
            "mean_acceptance_length": 1 + accepted / drafts,
            "per_draft_token_acceptance": accepted / draft_tokens}


def _post(path: str, timeout_s: int, payload=None) -> bytes:
    data = b"" if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{P.port}{path}", data=data, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_s) as r:
        if r.status != 200:
            raise RuntimeError(f"POST {path} returned {r.status}")
        return r.read()


def _wait_idle(names: tuple) -> None:
    deadline = time.time() + IDLE_TIMEOUT_S
    while True:
        m = _metrics(names)
        if m["vllm:num_requests_running"] == 0 and m["vllm:num_requests_waiting"] == 0:
            return
        if time.time() > deadline:
            raise RuntimeError(f"server not idle after {IDLE_TIMEOUT_S}s: {m}")
        time.sleep(1)


# ---------------------------------------------------------------- load gen

def _bench_cmd(concurrency: int, input_len: int, output_len: int, ratio: str,
               num_prompts: int, warmups: int, result_dir=None, result_name=None) -> list:
    cmd = [str(PY), "-m", "vllm.entrypoints.cli.main", "bench", "serve",
           "--backend", "vllm", "--model", P.served_model, "--tokenizer", str(P.model_dir),
           "--port", str(P.port), "--dataset-name", "random",
           "--random-input-len", str(input_len), "--random-output-len", str(output_len),
           "--random-range-ratio", ratio, "--num-prompts", str(num_prompts),
           "--max-concurrency", str(concurrency), "--num-warmups", str(warmups),
           "--ignore-eos", "--seed", str(SEED),
           # 0.30 default, pinned: no initial test request, so the decode gate
           # counts exactly the C measured first tokens.
           "--ready-check-timeout-sec", "0", *P.bench_sampling_args]
    if result_dir is not None:
        cmd += ["--percentile-metrics", "ttft,tpot,itl,e2el",
                "--metric-percentiles", "50,90,99", "--save-result", "--save-detailed",
                "--result-dir", str(result_dir), "--result-filename", result_name]
    return cmd


def _start_bench(plan: Plan, label: str, cmd: list, log_path: Path):
    plan.show(label, cmd + [">", str(log_path)], _client_env_extra())
    if plan.dry_run:
        return None
    env = _base_env()
    env.update(_client_env_extra())
    proc = subprocess.Popen(cmd, stdout=log_path.open("w"), stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    _register("bench", proc)
    return proc


def _stop_bench(proc) -> int:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        deadline = time.time() + 60
        while proc.poll() is None and time.time() < deadline:
            time.sleep(1)
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    if "bench" in _LIVE:
        _unregister("bench")
    return proc.returncode


# ------------------------------------------------------------------- smoke

def smoke(args, plan: Plan, mtp) -> None:
    out_dir = MICRO / P.log_tag / f"smoke_{_dir_suffix(mtp)}{args.tag}"
    _log(f"smoke -> {out_dir}")
    record = _begin_record(args, "smoke", mtp, {})
    if plan.dry_run:
        _check_model_dir()
        _start_clock_log(plan, out_dir)
        _start_server(plan, args, out_dir, None, mtp)
        print(f"[request] POST http://127.0.0.1:{P.port}/v1/completions "
              f"model={P.served_model} prompt='Hello' max_tokens=16 ignore_eos")
        return
    _prepare_real_run(record, out_dir)
    names = _metric_names(mtp)
    _save(record, out_dir / "smoke.json")
    clock = _start_clock_log(plan, out_dir)
    proc, t0 = _start_server(plan, args, out_dir, None, mtp)
    record["server_cmd"] = _server_cmd(out_dir, None, mtp)
    _wait_startup(proc, out_dir)
    record["startup_s"] = time.time() - t0
    record["startup"] = _startup_facts(out_dir, max(args.concurrency), mtp, args,
                                       _min_memory_total_mib(record))
    _save(record, out_dir / "smoke.json")
    before = _metrics(names) if _num_spec_tokens(mtp) else None
    t1 = time.time()
    body = json.loads(_post("/v1/completions", 600, {
        "model": P.served_model, "prompt": "Hello", "max_tokens": 16,
        "ignore_eos": True}))
    record["request_s"] = time.time() - t1
    record["completion_tokens"] = body["usage"]["completion_tokens"]
    if record["completion_tokens"] != 16:
        raise RuntimeError(f"expected 16 completion tokens, got {body['usage']}")
    if _num_spec_tokens(mtp):
        record["spec_decode"] = _spec_delta(before, _metrics(names))
    record["server_to_first_response_s"] = time.time() - t0
    record["stop"] = _stop_server(proc, 180)
    _stop_clock_log(clock)
    (out_dir / "clocks_after.txt").write_text(_smi("-q", "-d", "CLOCK"))
    _summarize_clock_log(out_dir)
    record["status"] = ("ok" if record["startup"]["kv_capacity_ok"]
                        else "kv_capacity_insufficient")
    _save(record, out_dir / "smoke.json")
    _log(f"smoke status={record['status']} startup={record['startup_s']:.0f}s "
         f"request={record['request_s']:.1f}s "
         f"kv_tokens={record['startup']['kv_cache_tokens']:,} "
         f"(need {record['startup']['kv_tokens_required_for_max_c']:,})")
    if P.extended_record:
        print("    selection", json.dumps(record["startup"]["selection"]))
        if _num_spec_tokens(mtp):
            print("    spec_decode", json.dumps(record["spec_decode"]))
    for line in record["startup"]["backend_lines"]:
        print("   ", line[:300])
    if not record["startup"]["kv_capacity_ok"]:
        raise SystemExit(f"KV capacity below C={max(args.concurrency)} x "
                         f"{MAX_REQUEST_TOKENS + _num_spec_tokens(mtp)} tokens")


# ---------------------------------------------------------------- closedloop

def closedloop(args, plan: Plan, mtp) -> None:
    out_dir = (MICRO / "closedloop"
               / f"{P.log_tag}_8k1k_{_dir_suffix(mtp)}{args.routing}_{args.tag}")
    _log(f"closedloop -> {out_dir}")
    record = _begin_record(args, "closedloop", mtp, {
        "protocol": {"input_len": ISL_MEAN, "output_len": OSL_MEAN,
                     "range_ratio": RANGE_RATIO,
                     "prompts_per_client": PROMPTS_PER_CLIENT,
                     "warmups_per_client": WARMUPS_PER_CLIENT,
                     "ignore_eos": True, "seed": SEED},
        "concurrency": {}})
    if plan.dry_run:
        _check_model_dir()
        _start_clock_log(plan, out_dir)
        _start_server(plan, args, out_dir, None, mtp)
        for c in args.concurrency:
            _start_bench(plan, f"closedloop C={c}", _bench_cmd(
                c, ISL_MEAN, OSL_MEAN, RANGE_RATIO, PROMPTS_PER_CLIENT * c,
                WARMUPS_PER_CLIENT * c, out_dir, f"c{c}.json"), out_dir / f"c{c}.log")
        return
    _prepare_real_run(record, out_dir)
    names = _metric_names(mtp)
    clock = _start_clock_log(plan, out_dir)
    proc, t0 = _start_server(plan, args, out_dir, None, mtp)
    record["server_cmd"] = _server_cmd(out_dir, None, mtp)
    _save(record, out_dir / "run.json")
    _wait_startup(proc, out_dir)
    record["startup_s"] = time.time() - t0
    record["startup"] = _startup_facts(out_dir, max(args.concurrency), mtp, args,
                                       _min_memory_total_mib(record))
    _save(record, out_dir / "run.json")
    if not record["startup"]["kv_capacity_ok"]:
        _stop_server(proc, 180)
        raise RuntimeError(f"KV cache {record['startup']['kv_cache_tokens']} tokens < "
                           f"{record['startup']['kv_tokens_required_for_max_c']}")
    for c in args.concurrency:
        _wait_idle(names)
        before = _metrics(names)
        cmd = _bench_cmd(c, ISL_MEAN, OSL_MEAN, RANGE_RATIO, PROMPTS_PER_CLIENT * c,
                         WARMUPS_PER_CLIENT * c, out_dir, f"c{c}.json")
        c0 = time.time()
        bench = _start_bench(plan, f"closedloop C={c}", cmd, out_dir / f"c{c}.log")
        bench.wait(timeout=CLOSEDLOOP_TIMEOUT_S)
        _unregister("bench")
        c1 = time.time()
        if bench.returncode != 0:
            raise RuntimeError(f"closed loop C={c} failed ({bench.returncode}); "
                               f"see {out_dir / f'c{c}.log'}")
        result = json.loads((out_dir / f"c{c}.json").read_text())
        if result["completed"] != PROMPTS_PER_CLIENT * c:
            raise RuntimeError(f"C={c}: completed {result['completed']} of "
                               f"{PROMPTS_PER_CLIENT * c}")
        after = _metrics(names)
        record["concurrency"][str(c)] = {
            "cmd": cmd, "t_start_unix": c0, "t_end_unix": c1,
            "completed": result["completed"],
            "preemptions": after["vllm:num_preemptions_total"]
            - before["vllm:num_preemptions_total"]}
        if _num_spec_tokens(mtp):
            # Includes the 2*C warm-up requests (same server interval).
            record["concurrency"][str(c)]["spec_decode"] = _spec_delta(before, after)
        _save(record, out_dir / "run.json")
        _log(f"closedloop C={c} done in {c1 - c0:.0f}s; "
             f"preemptions {record['concurrency'][str(c)]['preemptions']:.0f}")
    record["stop"] = _stop_server(proc, 180)
    _stop_clock_log(clock)
    for c, entry in record["concurrency"].items():
        entry["clock"] = _clock_flags(out_dir, entry["t_start_unix"], entry["t_end_unix"])
        _log(f"closedloop C={c} clock flag {entry['clock']['flag_non_1980']} "
             f"{entry['clock']['flagged_gpus']}")
    (out_dir / "clocks_after.txt").write_text(_smi("-q", "-d", "CLOCK"))
    _summarize_clock_log(out_dir)
    record["status"] = "complete"
    _save(record, out_dir / "run.json")


# --------------------------------------------------------------------- nsys

def _window(out_dir: Path, label: str, windows: list, names: tuple,
            last: bool = False) -> dict:
    before = _metrics(names)
    t0 = time.time()
    with (out_dir / "windows.txt").open("a") as handle:
        handle.write(f"{label} start {_now()}\n")
    _post("/start_profile", 120)
    time.sleep(WINDOW_S)
    t_stop_call = time.time()
    stop_answered = True
    if last:
        # nsys stops the server after this range; its reply may never come.
        try:
            _post("/stop_profile", LAST_STOP_PROFILE_TIMEOUT_S)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
            stop_answered = False
            _log(f"{label}: /stop_profile not answered ({type(error).__name__}); "
                 "nsys is shutting the server down after the last range")
    else:
        _post("/stop_profile", STOP_PROFILE_TIMEOUT_S)
    t1 = time.time()
    with (out_dir / "windows.txt").open("a") as handle:
        handle.write(f"{label} stop {_now()}\n")
    after = _metrics(names) if stop_answered else None
    entry = {"label": label, "window_index": len(windows) + 1,
             "t_start_unix": t0, "t_stop_call_unix": t_stop_call,
             "t_stop_return_unix": t1, "stop_profile_answered": stop_answered,
             "metrics_before": before, "metrics_after": after}
    windows.append(entry)
    return entry


def nsys_windows(args, plan: Plan, mtp) -> None:
    concurrency = args.nsys_concurrency
    out_dir = (MICRO / "nsys"
               / f"{P.log_tag}_windows_{_dir_suffix(mtp)}{args.routing}_{args.tag}")
    n_windows = 2 * len(concurrency)
    _log(f"nsys -> {out_dir} ({n_windows} windows)")
    record = _begin_record(args, "nsys", mtp, {
        "window_s": WINDOW_S, "protocol": {
            "decode": {"input_len": ISL_MEAN, "output_len": OSL_MEAN,
                       "range_ratio": RANGE_RATIO, "num_prompts": "C",
                       "gate": "running==C, waiting==0, TTFT count +C "
                               "(--ready-check-timeout-sec 0: no test prompt)"},
            "prefill": {"input_len": ISL_MEAN, "output_len": 1,
                        "range_ratio": "0",
                        "delay_after_main_run_s": PREFILL_WINDOW_DELAY_S}},
        "windows": []})
    if plan.dry_run:
        _check_model_dir()
        _start_clock_log(plan, out_dir)
        _start_server(plan, args, out_dir, n_windows, mtp)
        for c in concurrency:
            _start_bench(plan, f"decode C={c}", _bench_cmd(
                c, ISL_MEAN, OSL_MEAN, RANGE_RATIO, c, 0), out_dir / f"load_decode_c{c}.log")
            print(f"[window decode_C={c}] gate running=={c} waiting==0 "
                  f"ttft_count+={c}; POST /start_profile; sleep {WINDOW_S}; "
                  "POST /stop_profile; stop load; wait idle")
        for c in concurrency:
            _start_bench(plan, f"prefill C={c}", _bench_cmd(
                c, ISL_MEAN, 1, "0", max(PREFILL_MIN_PROMPTS, 8 * c), 0),
                out_dir / f"load_prefill_c{c}.log")
            print(f"[window prefill_C={c}] after 'Starting main benchmark run' + "
                  f"{PREFILL_WINDOW_DELAY_S}s: POST /start_profile; sleep {WINDOW_S}; "
                  "POST /stop_profile; stop load; wait idle")
        return
    _prepare_real_run(record, out_dir)
    names = _metric_names(mtp)
    clock = _start_clock_log(plan, out_dir)
    proc, t0 = _start_server(plan, args, out_dir, n_windows, mtp)
    record["server_cmd"] = _server_cmd(out_dir, n_windows, mtp)
    _save(record, out_dir / "run.json")
    _wait_startup(proc, out_dir)
    record["startup_s"] = time.time() - t0
    record["startup"] = _startup_facts(out_dir, max(concurrency), mtp, args,
                                       _min_memory_total_mib(record))
    _save(record, out_dir / "run.json")
    if not record["startup"]["kv_capacity_ok"]:
        _stop_server(proc, 180)
        raise RuntimeError("KV capacity insufficient for max C; see run.json")
    windows = record["windows"]

    for c in concurrency:
        _wait_idle(names)
        base = _metrics(names)
        cmd = _bench_cmd(c, ISL_MEAN, OSL_MEAN, RANGE_RATIO, c, 0)
        bench = _start_bench(plan, f"decode C={c}", cmd, out_dir / f"load_decode_c{c}.log")
        deadline = time.time() + DECODE_GATE_TIMEOUT_S
        while True:
            if bench.poll() is not None:
                raise RuntimeError(f"decode load C={c} exited {bench.returncode} "
                                   "before the window")
            m = _metrics(names)
            if (m["vllm:num_requests_running"] == c
                    and m["vllm:num_requests_waiting"] == 0
                    and m["vllm:time_to_first_token_seconds_count"]
                    - base["vllm:time_to_first_token_seconds_count"] >= c):
                break
            if time.time() > deadline:
                raise RuntimeError(f"decode gate C={c} not reached: {m}")
            time.sleep(0.2)
        entry = _window(out_dir, f"decode_C={c}", windows, names)
        entry.update({"phase": "decode", "concurrency": c, "load_cmd": cmd})
        b, a = entry["metrics_before"], entry["metrics_after"]
        entry["pure_decode"] = (
            a["vllm:time_to_first_token_seconds_count"]
            == b["vllm:time_to_first_token_seconds_count"]
            and a["vllm:request_success_total"] == b["vllm:request_success_total"]
            and a["vllm:num_preemptions_total"] == b["vllm:num_preemptions_total"])
        if _num_spec_tokens(mtp):
            entry["spec_decode"] = _spec_delta(b, a)
        entry["load_returncode"] = _stop_bench(bench)
        _save(record, out_dir / "run.json")
        _log(f"window {entry['window_index']} decode C={c}: pure={entry['pure_decode']} "
             f"flush={entry['t_stop_return_unix'] - entry['t_stop_call_unix']:.0f}s")

    for c in concurrency:
        _wait_idle(names)
        log_path = out_dir / f"load_prefill_c{c}.log"
        cmd = _bench_cmd(c, ISL_MEAN, 1, "0", max(PREFILL_MIN_PROMPTS, 8 * c), 0)
        bench = _start_bench(plan, f"prefill C={c}", cmd, log_path)
        deadline = time.time() + PREFILL_GATE_TIMEOUT_S
        while "Starting main benchmark run" not in log_path.read_text(errors="replace"):
            if bench.poll() is not None:
                raise RuntimeError(f"prefill load C={c} exited {bench.returncode} "
                                   "before the main run")
            if time.time() > deadline:
                raise RuntimeError(f"prefill load C={c} never started its main run")
            time.sleep(0.2)
        time.sleep(PREFILL_WINDOW_DELAY_S)
        if bench.poll() is not None:
            raise RuntimeError(f"prefill load C={c} finished before the window")
        entry = _window(out_dir, f"prefill_C={c}", windows, names,
                        last=(c == concurrency[-1]))
        entry.update({"phase": "prefill", "concurrency": c, "load_cmd": cmd})
        entry["load_still_running_at_stop"] = bench.poll() is None
        entry["load_returncode"] = _stop_bench(bench)
        _save(record, out_dir / "run.json")
        _log(f"window {entry['window_index']} prefill C={c}: "
             f"flush={entry['t_stop_return_unix'] - entry['t_stop_call_unix']:.0f}s")

    # repeat-shutdown:N makes nsys stop the server after the last range.
    deadline = time.time() + (NSYS_EXIT_TIMEOUT_S if windows[-1]["stop_profile_answered"]
                              else LAST_WINDOW_EXIT_TIMEOUT_S)
    while proc.poll() is None and time.time() < deadline:
        time.sleep(5)
    if proc.poll() is None:
        record["stop"] = _stop_server(proc, 600)
        record["nsys_self_shutdown"] = False
    else:
        _unregister("server")
        _wait_gpus_free()
        record["nsys_self_shutdown"] = True
        record["stop"] = {"returncode": proc.returncode}
    _stop_clock_log(clock)
    for entry in windows:
        entry["clock"] = _clock_flags(out_dir, entry["t_start_unix"],
                                      entry["t_stop_call_unix"])
        _log(f"window {entry['window_index']} {entry['label']} clock flag "
             f"{entry['clock']['flag_non_1980']} {entry['clock']['flagged_gpus']}")
    (out_dir / "clocks_after.txt").write_text(_smi("-q", "-d", "CLOCK"))
    _summarize_clock_log(out_dir)
    reps = sorted(p.name for p in out_dir.glob("window_*.nsys-rep"))
    record["reports"] = reps
    (out_dir / "windows.json").write_text(json.dumps(windows, indent=2) + "\n")
    if len(reps) != n_windows:
        record["status"] = "report_count_mismatch"
        _save(record, out_dir / "run.json")
        raise RuntimeError(f"expected {n_windows} nsys reports, found {reps}")
    record["status"] = "complete"
    _save(record, out_dir / "run.json")


# ------------------------------------------------------------------ offload

# Prometheus series read per interval (vLLM 0.30 v1/metrics/loggers.py; the
# prometheus client appends _total to counters).  GPU prefix cache = local
# hits; "external" = hits served by the KV connector, i.e. host (CPU) hits.
OFFLOAD_SUMS = {
    "prompt_tokens": ("vllm:prompt_tokens_total", ()),
    "gpu_prefix_queries": ("vllm:prefix_cache_queries_total", ()),
    "gpu_prefix_hits": ("vllm:prefix_cache_hits_total", ()),
    "host_prefix_queries": ("vllm:external_prefix_cache_queries_total", ()),
    "host_prefix_hits": ("vllm:external_prefix_cache_hits_total", ()),
    "tokens_local_compute": ("vllm:prompt_tokens_by_source_total",
                             (("source", "local_compute"),)),
    "tokens_local_cache_hit": ("vllm:prompt_tokens_by_source_total",
                               (("source", "local_cache_hit"),)),
    "tokens_external_kv_transfer": ("vllm:prompt_tokens_by_source_total",
                                    (("source", "external_kv_transfer"),)),
    "preemptions": ("vllm:num_preemptions_total", ()),
    "requests_ok": ("vllm:request_success_total", ()),
    "generation_tokens": ("vllm:generation_tokens_total", ()),
}
# OffloadingConnector (CPUOffloadingSpec) transfer counters; the transfer_type
# series are created at startup (offloading/metrics.py OffloadPromMetrics),
# the flat vllm:kv_offload_{load,store}_* ones only after the first transfer.
NATIVE_OFFLOAD_SUMS = {
    "host_to_gpu_bytes": ("vllm:kv_offload_total_bytes_total",
                          (("transfer_type", "CPU_to_GPU"),)),
    "gpu_to_host_bytes": ("vllm:kv_offload_total_bytes_total",
                          (("transfer_type", "GPU_to_CPU"),)),
    "host_to_gpu_time_s": ("vllm:kv_offload_total_time_total",
                           (("transfer_type", "CPU_to_GPU"),)),
    "gpu_to_host_time_s": ("vllm:kv_offload_total_time_total",
                           (("transfer_type", "GPU_to_CPU"),)),
}
PROM_LINE = re.compile(r'^([A-Za-z_:][\w:]*)(\{(?:[^"}]|"(?:[^"\\]|\\.)*")*\})?\s+(\S+)$')
PROM_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def _prom_parse(text: str) -> list:
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROM_LINE.match(line)
        if not match:
            raise RuntimeError(f"unparsable /metrics line {line!r}")
        labels = dict(PROM_LABEL.findall(match.group(2) or ""))
        rows.append((match.group(1), labels, float(match.group(3))))
    return rows


def _prom_sums(rows: list, table: dict) -> dict:
    out = {}
    for key, (name, want) in table.items():
        values = [v for n, labels, v in rows
                  if n == name and all(k in labels and labels[k] == x
                                       for k, x in want)]
        if not values:
            raise RuntimeError(f"/metrics has no {name} {dict(want)} series")
        out[key] = sum(values)
    return out


def _offload_tables(arm: str) -> dict:
    table = dict(OFFLOAD_SUMS)
    if arm == "native":
        table.update(NATIVE_OFFLOAD_SUMS)
    return table


def _offload_delta(before_path: Path, after_path: Path, arm: str) -> dict:
    table = _offload_tables(arm)
    before = _prom_sums(_prom_parse(before_path.read_text()), table)
    after = _prom_sums(_prom_parse(after_path.read_text()), table)
    delta = {k: after[k] - before[k] for k in table}
    delta["gpu_prefix_hit_rate"] = (delta["gpu_prefix_hits"] / delta["gpu_prefix_queries"]
                                    if delta["gpu_prefix_queries"] else None)
    delta["host_prefix_hit_rate"] = (delta["host_prefix_hits"] / delta["host_prefix_queries"]
                                     if delta["host_prefix_queries"] else None)
    return delta


def _scrape_metrics(path: Path) -> None:
    with urllib.request.urlopen(f"http://127.0.0.1:{P.port}/metrics", timeout=30) as r:
        path.write_bytes(r.read())


def _offload_out_dir(args, arm: str) -> Path:
    return MICRO / "offload" / f"{P.log_tag}_agentx_{arm}_{args.routing}_{args.tag}"


def _offload_env_extra(args, arm: str) -> dict:
    extra = _server_env_extra(args)
    extra[f"{P.env_prefix}_MAX_MODEL_LEN"] = str(P.offload.max_model_len)
    extra["VLLM_USE_SIMPLE_KV_OFFLOAD"] = P.offload.arms[arm].use_simple_kv_offload
    return extra


def _offload_server_cmd(arm: str) -> list:
    return ["bash", str(P.offload.serve_sh), *P.offload.arms[arm].serve_args]


def _replay_cmd(args, c: int, out_dir) -> list:
    g = P.offload
    cmd = [str(PY), str(g.replay_script), "replay", "--workload", args.offload_workload,
           "--port", str(P.port), "--served-model", P.served_model,
           "--max-model-len", str(g.max_model_len), "--concurrency", str(c),
           "--nonce", f"agentx|{args.tag}|C{c}", "--warmup-s", str(g.warmup_s),
           "--measure-s", str(g.measure_s), "--grace-s", str(g.grace_s),
           "--think-cap-s", str(g.think_cap_s),
           "--sock-read-timeout-s", str(g.sock_read_timeout_s)]
    return cmd + (["--dry-run"] if out_dir is None else ["--out-dir", str(out_dir)])


def _offload_activation(out_dir: Path, arm: str) -> dict:
    """Connector / prefix-cache facts from server.log; raise if not as configured."""
    text = (out_dir / "server.log").read_text(errors="replace")
    spec = P.offload.arms[arm]
    engine = [line for line in text.splitlines() if "Initializing a V1 LLM engine" in line]
    if not engine or not all("enable_prefix_caching=True" in line for line in engine):
        raise RuntimeError("engine config does not show enable_prefix_caching=True")
    connectors = sorted(set(re.findall(r"Creating v1 connector with name: (\w+)", text)))
    want = [spec.connector] if spec.connector else []
    if connectors != want:
        raise RuntimeError(f"arm {arm}: KV connectors {connectors}, expected {want}")
    facts = {"kv_connectors": connectors, "activation_lines": {}}
    for pattern in spec.activation_patterns:
        found = re.findall(pattern, text)
        if not found:
            raise RuntimeError(f"arm {arm}: no server.log line matches {pattern!r}")
        facts["activation_lines"][pattern] = sorted(set(found))
    if re.search(r"disabling CPU offload", text):
        raise RuntimeError(f"arm {arm}: server disabled CPU offload")
    if re.search(r"cudaHostRegister failed", text):
        raise RuntimeError(f"arm {arm}: host KV buffer not pinned (cudaHostRegister failed)")
    return facts


def _offload_checks(arm: str, delta: dict) -> dict:
    """Mechanism checks of one measured interval (flags; the arm-level
    host-load requirement is enforced after the last point)."""
    spec = P.offload.arms[arm]
    checks = {"gpu_prefix_hits_positive": delta["gpu_prefix_hits"] > 0}
    if spec.connector:
        checks["host_prefix_hits_positive"] = delta["host_prefix_hits"] > 0
    else:
        checks["no_host_queries"] = delta["host_prefix_queries"] == 0
    if arm == "native":
        checks["gpu_to_host_bytes_positive"] = delta["gpu_to_host_bytes"] > 0
        checks["host_to_gpu_bytes_positive"] = delta["host_to_gpu_bytes"] > 0
    return checks


def _check_offload_prereqs(arm: str) -> dict:
    if ("PYTORCH_CUDA_ALLOC_CONF" in os.environ
            and "expandable_segments:True" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]):
        raise RuntimeError("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True rejects "
                           "KV connectors in vLLM 0.30")
    shm = os.statvfs("/dev/shm")
    free_gib = shm.f_bavail * shm.f_frsize / 2**30
    size = P.offload.arms[arm].serve_args
    facts = {"dev_shm_free_gib": round(free_gib, 1)}
    if "--kv-offloading-size" in size:
        gib = float(size[size.index("--kv-offloading-size") + 1])
        facts["kv_offloading_size_gib"] = gib
        # OffloadingConnector maps /dev/shm/vllm_offload_<engine>.mmap of this size.
        if P.offload.arms[arm].use_simple_kv_offload == "0" and free_gib < gib:
            raise RuntimeError(f"/dev/shm has {free_gib:.0f} GiB free < {gib} GiB")
    return facts


def _offload_arm(args, plan: Plan, arm: str) -> dict:
    g = P.offload
    out_dir = _offload_out_dir(args, arm)
    _log(f"offload arm {arm} -> {out_dir}")
    record = _begin_record(args, "offload", "off", {
        "arm": arm, "offload_group": {
            "serve_sh": g.serve_sh.name, "max_model_len": g.max_model_len,
            "arm_serve_args": list(g.arms[arm].serve_args),
            "use_simple_kv_offload": g.arms[arm].use_simple_kv_offload,
            "expected_connector": g.arms[arm].connector,
            "workload": args.offload_workload,
            "warmup_s": g.warmup_s, "measure_s": g.measure_s, "grace_s": g.grace_s,
            "think_cap_s": g.think_cap_s},
        "concurrency": {}})
    if plan.dry_run:
        _check_model_dir()
        _start_clock_log(plan, out_dir)
        plan.show("server", _offload_server_cmd(arm) + [">", str(out_dir / "server.log")],
                  _offload_env_extra(args, arm))
        print(f"[checks] server.log: engine enable_prefix_caching=True; connectors == "
              f"{[g.arms[arm].connector] if g.arms[arm].connector else []}; "
              f"lines {list(g.arms[arm].activation_patterns)}")
        for c in args.concurrency:
            cmd = _replay_cmd(args, c, out_dir / f"c{c}")
            plan.show(f"offload {arm} C={c}", cmd + [">", str(out_dir / f"c{c}.log")],
                      _client_env_extra())
            env = _base_env()
            env.update(_client_env_extra())
            subprocess.run([sys.executable, *_replay_cmd(args, c, None)[1:]], check=True,
                           env=env)
        return record
    _prepare_real_run(record, out_dir)
    record["scripts"][g.serve_sh.name] = _sha256(g.serve_sh)
    record["scripts"][g.replay_script.name] = _sha256(g.replay_script)
    record["workload_sha256"] = _sha256(Path(args.offload_workload))
    record["prereqs"] = _check_offload_prereqs(arm)
    names = _metric_names("off")
    clock = _start_clock_log(plan, out_dir)
    extra = _offload_env_extra(args, arm)
    cmd = _offload_server_cmd(arm)
    plan.show("server", cmd + [">", str(out_dir / "server.log")], extra)
    env = os.environ.copy()
    env.update(extra)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=(out_dir / "server.log").open("w"),
                            stderr=subprocess.STDOUT, env=env, start_new_session=True)
    _register("server", proc)
    record["server_cmd"] = cmd
    record["server_env_extra"] = extra
    _save(record, out_dir / "run.json")
    _wait_startup(proc, out_dir)
    record["startup_s"] = time.time() - t0
    record["startup"] = _startup_facts_for(out_dir, g.max_model_len, args,
                                           _min_memory_total_mib(record),
                                           g.server_extra_patterns)
    record["activation"] = _offload_activation(out_dir, arm)
    _save(record, out_dir / "run.json")
    _log(f"offload arm {arm}: kv_tokens={record['startup']['kv_cache_tokens']:,} "
         f"connectors={record['activation']['kv_connectors']} "
         f"startup={record['startup_s']:.0f}s")
    host_hits = 0.0
    for c in args.concurrency:
        _wait_idle(names)
        c_dir = out_dir / f"c{c}"
        _scrape_metrics(out_dir / f"metrics_c{c}_before.prom")
        cmd = _replay_cmd(args, c, c_dir)
        c0 = time.time()
        bench = _start_bench(plan, f"offload {arm} C={c}", cmd, out_dir / f"c{c}.log")
        bench.wait(timeout=g.warmup_s + g.measure_s + g.grace_s + 900)
        _unregister("bench")
        c1 = time.time()
        _scrape_metrics(out_dir / f"metrics_c{c}_after.prom")
        if bench.returncode != 0:
            raise RuntimeError(f"offload {arm} C={c} replay failed ({bench.returncode}); "
                               f"see {out_dir / f'c{c}.log'}")
        summary = json.loads((c_dir / "summary.json").read_text())
        marks = summary["marks"]
        measured = _offload_delta(c_dir / "metrics_measure_start.prom",
                                  c_dir / "metrics_end.prom", arm)
        whole = _offload_delta(out_dir / f"metrics_c{c}_before.prom",
                               out_dir / f"metrics_c{c}_after.prom", arm)
        entry = {"cmd": cmd, "t_start_unix": c0, "t_end_unix": c1, "marks": marks,
                 "requests": summary["requests"], "ttft_s": summary["ttft_s"],
                 "tpot_s": summary["tpot_s"], "e2e_s": summary["e2e_s"],
                 "measure_ok_tokens": summary["measure_ok_tokens"],
                 "metrics_measure_window": measured, "metrics_whole_point": whole,
                 "checks": _offload_checks(arm, measured)}
        host_hits += whole["host_prefix_hits"]
        record["concurrency"][str(c)] = entry
        _save(record, out_dir / "run.json")
        _log(f"offload {arm} C={c}: {summary['requests']['measure_ok']} measured ok, "
             f"TTFT p50 {summary['ttft_s']['p50'] if summary['ttft_s']['n'] else None} s, "
             f"GPU hits {measured['gpu_prefix_hits']:.0f} host hits "
             f"{measured['host_prefix_hits']:.0f} preemptions {measured['preemptions']:.0f} "
             f"checks {entry['checks']}")
    record["stop"] = _stop_server(proc, 180)
    _stop_clock_log(clock)
    for c, entry in record["concurrency"].items():
        entry["clock"] = _clock_flags(out_dir, entry["marks"]["t_measure"],
                                      entry["marks"]["t_stop"])
        _log(f"offload {arm} C={c} clock flag {entry['clock']['flag_non_1980']} "
             f"{entry['clock']['flagged_gpus']}")
    (out_dir / "clocks_after.txt").write_text(_smi("-q", "-d", "CLOCK"))
    _summarize_clock_log(out_dir)
    if g.arms[arm].host_loads_required and host_hits <= 0:
        record["status"] = "no_host_loads"
        _save(record, out_dir / "run.json")
        raise RuntimeError(f"arm {arm}: offload connector active but no host prefix "
                           "hit (external_prefix_cache_hits) in any point")
    record["status"] = "complete"
    _save(record, out_dir / "run.json")
    return record


def offload_group(args, plan: Plan) -> None:
    workload = json.loads(Path(args.offload_workload).read_text())
    if workload["params"]["max_model_len"] != P.offload.max_model_len:
        raise RuntimeError(f"workload max_model_len {workload['params']['max_model_len']}"
                           f" != {P.offload.max_model_len}")
    if max(args.concurrency) > len(workload["lanes"]):
        raise RuntimeError(f"workload plans {len(workload['lanes'])} lanes < "
                           f"C={max(args.concurrency)}")
    for arm in args.offload_arms:
        _offload_arm(args, plan, arm)


# --------------------------------------------------------------------- main

def _raise_on_sigterm(signum, frame) -> None:
    raise SystemExit(f"stopped by signal {signum}")


def _gpu_memory_utilization(text: str) -> str:
    value = float(text)
    if not 0.5 < value < 1.0:
        raise argparse.ArgumentTypeError(f"{text} is not in (0.5, 1.0)")
    return text


def main(profile: ModelProfile) -> None:
    global P, B, MICRO, PY, NSYS, LOCK_DIR
    P = profile
    parser = argparse.ArgumentParser(
        description=P.doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("smoke", "run"))
    parser.add_argument(
        "--public-root",
        type=Path,
        default=(Path(os.environ["MIDLAYER_H200_PUBLIC_ROOT"])
                 if "MIDLAYER_H200_PUBLIC_ROOT" in os.environ else None),
        help="H200 dependency/output root; may use MIDLAYER_H200_PUBLIC_ROOT",
    )
    parser.add_argument("--runtime-python", required=True, type=Path,
                        help="Python executable for the selected vLLM runtime")
    parser.add_argument("--tag", required=True,
                        help="output suffix, e.g. 20260927_1230")
    parser.add_argument("--source-commit", required=True,
                        help="repository commit the deployed snapshot came from")
    parser.add_argument("--routing", required=True,
                        choices=("uniform_random", "normal_routing"),
                        help="VLLM_MOE_ROUTING_SIMULATION_STRATEGY (vLLM 0.30 "
                             "RoutingSimulator strategies)")
    parser.add_argument("--kv-cache-dtype", required=True,
                        choices=P.kv_cache_dtypes, help=P.kv_cache_dtype_help)
    if P.gpu_memory_utilization_arg:
        parser.add_argument("--gpu-memory-utilization", required=True,
                            type=_gpu_memory_utilization,
                            help="vllm serve --gpu-memory-utilization")
    if _has_mtp():
        parser.add_argument("--mtp", nargs="+", required=True, choices=("off", "on"),
                            help="server configurations, run in this order within "
                                 "each phase; on = --speculative-config "
                                 + json.dumps(P.speculative_config_mtp_on))
    parser.add_argument("--concurrency", type=int, nargs="+", required=True)
    if P.separate_nsys_concurrency:
        parser.add_argument("--nsys-concurrency", type=int, nargs="+",
                            help="C values of the nsys windows (run --phases nsys)")
    if P.offload is None:
        parser.add_argument("--phases", nargs="+", choices=("closedloop", "nsys"),
                            help="run mode only; order is always closedloop then nsys")
    else:
        parser.add_argument("--phases", nargs="+", choices=("closedloop", "nsys", "offload"),
                            help="run mode only; order is always closedloop then nsys; "
                                 "offload (KV host-offload group, MTP off, --concurrency = "
                                 "AgentX session lanes) runs alone")
        parser.add_argument("--offload-arms", nargs="+", choices=tuple(P.offload.arms),
                            help="offload group server configurations, in this order")
        parser.add_argument("--offload-workload",
                            help="agentx_session_replay.py build output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.public_root is None:
        parser.error("--public-root or MIDLAYER_H200_PUBLIC_ROOT is required")
    B = args.public_root
    P = replace(
        P,
        model_dir=(P.model_dir if P.model_dir.is_absolute()
                   else B / P.model_dir),
    )
    MICRO = B / "h200_microbench"
    PY = args.runtime_python
    NSYS = B / "tools" / "nsys_2025.6.1" / "bin" / "nsys"
    LOCK_DIR = MICRO / "kcampaign" / "runs"
    if P.offload is not None:
        wants_offload = args.mode == "run" and "offload" in args.phases
        if wants_offload and len(args.phases) != 1:
            parser.error("--phases offload runs alone")
        if wants_offload != (args.offload_arms is not None) or \
                wants_offload != (args.offload_workload is not None):
            parser.error("--offload-arms and --offload-workload are required with, and "
                         "only with, run --phases offload")
        if wants_offload and args.mtp != ["off"]:
            parser.error("the offload group is MTP off: pass --mtp off")
        if wants_offload and len(set(args.offload_arms)) != len(args.offload_arms):
            parser.error(f"duplicate --offload-arms {args.offload_arms}")
    if args.mode == "run" and not args.phases:
        parser.error("run needs --phases")
    if args.mode == "smoke" and args.phases:
        parser.error("smoke takes no --phases")
    if P.separate_nsys_concurrency:
        wants_nsys = args.mode == "run" and "nsys" in args.phases
        if wants_nsys != (args.nsys_concurrency is not None):
            parser.error("--nsys-concurrency is required with, and only with, "
                         "run --phases nsys")
    else:
        args.nsys_concurrency = args.concurrency
    mtp_settings = [None]
    if _has_mtp():
        if len(set(args.mtp)) != len(args.mtp):
            parser.error(f"duplicate --mtp values {args.mtp}")
        mtp_settings = args.mtp
    plan = Plan(args.dry_run)
    if P.sigterm_cleanup:
        # `kill <runner pid>` then runs the finally below, which stops the
        # server; otherwise the server would outlive the released GPU locks.
        signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        if args.mode == "smoke":
            for mtp in mtp_settings:
                smoke(args, plan, mtp)
            return
        if "offload" in args.phases:
            offload_group(args, plan)
            return
        if "closedloop" in args.phases:
            for mtp in mtp_settings:
                closedloop(args, plan, mtp)
        if "nsys" in args.phases:
            for mtp in mtp_settings:
                nsys_windows(args, plan, mtp)
    finally:
        # Re-raises whatever failed; only guarantees no orphaned GPU process.
        _cleanup()
