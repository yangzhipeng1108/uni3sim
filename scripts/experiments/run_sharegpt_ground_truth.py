#!/usr/bin/env python3
"""CLI for canonical ShareGPT production Serving ground truth.

The host delivery path drifts between days by much more than it varies inside
one session: on the same protocol A800's marginal delivery term measured about
25% apart on two different days, while five back-to-back repeats within a
session spread only 3%.  Decomposed, the cross-session spread is 23.68% true
drift against 3.03% measurement noise (C500 4.82% against 3.83%, 910B 5.85%
against 1.27%).  A probe value measured days away from a capture therefore
cannot be compared against that capture -- the difference is dominated by
machine state rather than by the quantity being measured.

So the delivery-region probe runs here as a routine step, once immediately
before the capture and once immediately after, writing both results next to the
capture output.  Bracketing the capture puts probe and capture on the same
machine state so the drift cancels in the comparison, and it also exposes any
drift that happened *during* the capture as the spread between the two.
"""

from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.sharegpt_ground_truth import main

PROBE = (ROOT / "microbench" / "nvidia_a800" / "runtime" / "host_request_path"
         / "vllm_delivery_region_runner.py")
PROBE_FANOUTS = "1,2,4,8,16,32,64"
PROBE_ROUNDS = "40"
# The probe reads its fourth argument as the per-fanout repeat count and only
# writes the JSON artifact when a fifth argument is present.
PROBE_REPEATS = "3"


def _probe_plan(argv: list[str]) -> tuple[str, Path] | None:
    """Model and output directory for the probe, or None when it is off."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--output")
    parser.add_argument("--no-host-path-probe", action="store_true")
    known, _ = parser.parse_known_args(argv)
    if known.no_host_path_probe:
        return None
    if known.model is None or known.output is None:
        # The capture itself requires both; let its own parser report that
        # rather than guessing here.
        return None
    return known.model, Path(known.output).resolve().parent


def _run_probe(model: str, destination: Path, when: str) -> None:
    target = destination / f"host_request_path_probe_{when}.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite probe artifact: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, str(PROBE), model, PROBE_FANOUTS, PROBE_ROUNDS,
         PROBE_REPEATS, str(target)],
        capture_output=True, text=True)
    if completed.returncode == 0 and target.exists():
        print(f"[host-path-probe] {when}: {target}")
        return
    # Either half may fail without the capture being at fault: the probe is a
    # three-process measurement and a busy host can push it past its internal
    # timeouts, which is an expected condition rather than a sign of trouble.
    # The bracket makes a later comparison trustworthy; it is not part of what
    # makes the capture itself valid, so it must never block one.  The failure
    # is written into the artifact instead, which states that this half is
    # missing and why -- an explicit record, not a silent fallback.
    target.write_text(
        '{\n  "schema_version": 1,\n  "status": "probe_failed",\n'
        f'  "returncode": {completed.returncode},\n'
        f'  "stderr_tail": {completed.stderr.strip()[-2000:]!r}\n''}\n'
        .replace("'", '"'))
    print(f"[host-path-probe] {when}: FAILED, recorded at {target}", file=sys.stderr)


def run(argv: list[str]) -> int:
    plan = _probe_plan(argv)
    if plan is None:
        return main()
    model, destination = plan
    _run_probe(model, destination, "before")
    try:
        return main()
    finally:
        _run_probe(model, destination, "after")


if __name__ == "__main__":
    # The probe plan reads the raw arguments so it can see its own opt-out
    # flag; the capture's parser must not, so it is stripped from sys.argv,
    # which is what main() parses.
    _raw = sys.argv[1:]
    sys.argv = [sys.argv[0]] + [a for a in _raw if a != "--no-host-path-probe"]
    raise SystemExit(run(_raw))
