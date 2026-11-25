"""Record independent worker Host and stream event spans before perturbing a run."""

import json
import os
from pathlib import Path
import time

import torch
import torch_npu  # noqa: F401
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


_trace_dir = Path(os.environ["MID_TP2_TINY_TRACE_DIR"])
_original = NPUModelRunner.execute_model
_spans = []
_written = False


def _timed_execute(self, *args, **kwargs):
    global _written
    stream = torch.npu.current_stream()
    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    begin.record(stream)
    host_start_ns = time.monotonic_ns()
    try:
        return _original(self, *args, **kwargs)
    finally:
        host_end_ns = time.monotonic_ns()
        end.record(stream)
        end_done_at_return = end.query()
        _spans.append((begin, end, host_start_ns, host_end_ns,
                       end_done_at_return))
        if len(_spans) == 6 and not _written:
            # The first five spans predate this synchronization. It is used
            # only to read completed event times, and the sixth is discarded.
            torch.npu.synchronize()
            rows = [
                {
                    "host_start_ns": start,
                    "host_end_ns": finish,
                    "host_execute_ms": (finish - start) / 1e6,
                    "device_event_ms": first.elapsed_time(last),
                    "end_done_at_return": done,
                }
                for first, last, start, finish, done in _spans[:5]
            ]
            (_trace_dir / f"worker_{os.getpid()}.json").write_text(
                json.dumps({"schema_version": 1, "pid": os.getpid(),
                            "device": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                            "spans": rows}, indent=2) + "\n")
            _written = True


NPUModelRunner.execute_model = _timed_execute
