"""Capture NCU counters for three pinned H200 dataset cases on Modal.

Each case first runs one NSYS window with the canonical runner and must match
the existing canonical kernel names and launch geometry.  NCU durations are
raw profiler output only and never replace the canonical NSYS timing.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import modal

_MODULE_PATH = Path(__file__).resolve()
ROOT = (Path("/opt/midlayer") if (len(_MODULE_PATH.parents) < 4
                                or Path("/opt/midlayer/scripts").is_dir())
        else _MODULE_PATH.parents[3])
sys.path.insert(0, str(ROOT))

from scripts.kernel_dataset.nvidia_h200.missing39_modal_runner import (
    CAMPAIGN, IMAGE, _check_identity, _sha256, _stream_files)


APP = modal.App("midlayer-h200-ncu-subset-20261030")
REMOTE = Path("/opt/midlayer")


def _dimensions(text: str) -> list[int]:
    return [int(part.strip()) for part in text.strip("()").split(",")]


def _verify_launch(case: dict, window_path: Path) -> None:
    window = json.loads(window_path.read_text())
    if not window["telemetry"]["clock_accepted"]:
        raise ValueError(f"{case['case_id']} preflight clock rejected")
    actual = [[kernel["name"], kernel["grid"], kernel["block"]]
              for kernel in window["launches"]]
    expected = [[name, _dimensions(grid), _dimensions(block)]
                for name, grid, block in case["canonical_launch_signature"]]
    if actual != expected:
        raise ValueError(f"{case['case_id']} launch differs from canonical")


@APP.function(gpu="H200", image=IMAGE, timeout=100,
              max_containers=1, single_use_containers=True)
def capture(plan: dict, case_id: str, attempt: str):
    identity = _check_identity()
    yield {"kind": "identity", "data": identity}
    with tempfile.TemporaryDirectory(prefix="midlayer-ncu-subset-", dir="/tmp") as tmp:
        root = Path(tmp)
        sent = set()
        for case in (item for item in plan["cases"]
                     if item["case_id"] == case_id):
            family = case["family"]
            target = root / "ncu_raw" / attempt / case_id
            target.mkdir(parents=True)
            (target / "metric_plan.json").write_text(
                json.dumps(plan, indent=2) + "\n")
            plan_path = target / "preflight_plan.json"
            plan_path.write_text(json.dumps({
                "schema_version": 1, "hardware_id": "nvidia_h200",
                "campaign": "ncu_preflight_" + case_id,
                "case_count": 1, "plans": [case["plan"]],
            }, indent=2) + "\n")
            preflight = target / "preflight"
            preflight_cmd = [
                sys.executable,
                str(REMOTE / "scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py"),
                "--plans", str(plan_path), "--output-dir", str(preflight),
                "--window-ids", "1", "--concurrency", "1",
                "--physical-gpu", "0", "--nsys", shutil.which("nsys"),
                "--python", sys.executable, "--gpu-lock", str(root / "gpu.lock"),
            ]
            preflight_run = subprocess.run(
                preflight_cmd, capture_output=True, text=True, timeout=90)
            (target / "preflight_command.json").write_text(json.dumps({
                "command": preflight_cmd, "returncode": preflight_run.returncode,
                "stdout": preflight_run.stdout, "stderr": preflight_run.stderr,
            }, indent=2) + "\n")
            yield from _stream_files(root, sent)
            if preflight_run.returncode != 0:
                yield {"kind": "stopped", "case_id": case_id,
                       "reason": "preflight_runner_failed"}
                return
            window_path = preflight / "cases" / case_id / "window_01.launch.json"
            if not window_path.is_file():
                yield {"kind": "stopped", "case_id": case_id,
                       "reason": "preflight_window_missing"}
                return
            try:
                _verify_launch(case, window_path)
            except ValueError:
                yield {"kind": "stopped", "case_id": case_id,
                       "reason": "launch_mismatch"}
                return
            marker = case["plan"]["marker"]
            metrics = plan["traffic_metrics"] + plan["service_metrics"]
            ncu_cmd = [
                shutil.which("ncu"), "--nvtx", "--nvtx-include", marker + "/",
                "--clock-control", "none", "--cache-control", "none",
                "--csv", "--page", "raw", "--metrics", ",".join(metrics),
                "--export", str(target / "profile"),
                sys.executable, str(REMOTE / case["plan"]["runner_relpath"]),
                *case["plan"]["runner_args"],
                "--h200-gpu-lock", str(root / "gpu.lock"),
                "--h200-telemetry-output", str(target / "ncu_telemetry.json"),
                "--h200-physical-gpu", "0",
            ]
            result = subprocess.run(ncu_cmd, capture_output=True,
                                    timeout=180)
            (target / "ncu_stdout.csv").write_bytes(result.stdout)
            (target / "ncu_stderr.txt").write_bytes(result.stderr)
            (target / "ncu_command.json").write_text(json.dumps({
                "command": ncu_cmd, "returncode": result.returncode,
                "identity": identity,
            }, indent=2) + "\n")
            profile = target / "profile.ncu-rep"
            if result.returncode == 0 and profile.is_file():
                import zstandard
                compressed = target / "profile.ncu-rep.zst"
                with profile.open("rb") as source, compressed.open("wb") as dest:
                    zstandard.ZstdCompressor(level=3).copy_stream(source, dest)
                profile.unlink()
            yield from _stream_files(root, sent)
            if result.returncode != 0:
                yield {"kind": "stopped", "case_id": case_id,
                       "reason": "ncu_failed"}
                return
            if not (target / "profile.ncu-rep.zst").is_file():
                yield {"kind": "stopped", "case_id": case_id,
                       "reason": "ncu_profile_missing"}
                return
            yield {"kind": "case_complete", "case_id": case_id}
        yield {"kind": "complete"}


@APP.local_entrypoint()
def main(case_id: str, attempt: str = "attempt_01",
         plan_name: str = "ncu_plan.json") -> None:
    if plan_name not in ("ncu_plan.json", "ncu_cycles_plan.json"):
        raise ValueError(f"unknown prepared NCU plan: {plan_name}")
    plan = json.loads((CAMPAIGN / plan_name).read_text())
    if plan["status"] != "candidate_not_executed":
        raise ValueError("NCU plan is not a prepared candidate")
    if case_id not in [case["case_id"] for case in plan["cases"]]:
        raise ValueError(f"unknown prepared NCU case: {case_id}")
    output = CAMPAIGN / "ncu_raw" / attempt / case_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite NCU raw: {output}")
    output.mkdir(parents=True)
    open_files = {}
    for event in capture.remote_gen(plan, case_id, attempt):
        kind = event["kind"]
        if kind == "identity":
            (output / "identity.json").write_text(
                json.dumps(event["data"], indent=2) + "\n")
        elif kind == "file_start":
            relative = Path(event["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"invalid remote path: {relative}")
            target = CAMPAIGN / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            open_files[event["path"]] = (target.open("xb"), event["sha256"],
                                          event["size"])
        elif kind == "file_chunk":
            open_files[event["path"]][0].write(event["data"])
        elif kind == "file_end":
            handle, digest, size = open_files.pop(event["path"])
            handle.close()
            target = CAMPAIGN / event["path"]
            if target.stat().st_size != size or _sha256(target) != digest:
                raise ValueError(f"transferred NCU raw failed hash check: {target}")
        elif kind in ("case_complete", "stopped", "complete"):
            (output / "transfer_status.json").write_text(
                json.dumps(event, indent=2) + "\n")
            print(json.dumps(event))
        else:
            raise ValueError(f"unknown remote event: {kind}")
    if open_files:
        raise ValueError(f"incomplete NCU transfer: {sorted(open_files)}")
