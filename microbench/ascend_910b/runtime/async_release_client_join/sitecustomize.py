"""Trace the actual vLLM 0.18 EngineCore batch queue for a tiny source probe."""

import json
import os
import time

from vllm.v1.engine.core import EngineCore


_ORIGINAL_STEP = EngineCore.step_with_batch_queue
_TRACE_PATH = os.path.join(
    os.environ["MIDLAYER_ASCEND_CONTAINER_TASK_ROOT"], "queue_events.jsonl")


def _traced_step(self):
    queue_before = self.batch_queue
    before = [] if queue_before is None else [
        {"future_id": id(item[0]), "done": item[0].done()}
        for item in queue_before
    ]
    result = _ORIGINAL_STEP(self)
    queue_after = self.batch_queue
    after = [] if queue_after is None else [
        {"future_id": id(item[0]), "done": item[0].done()}
        for item in queue_after
    ]
    record = {
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        "batch_queue_size": self.batch_queue_size,
        "before": before,
        "after": after,
        "returned_output": result[0] is not None,
        "model_executed": result[1],
    }
    fd = os.open(_TRACE_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(record, separators=(",", ":")) + "\n").encode())
    finally:
        os.close(fd)
    return result


EngineCore.step_with_batch_queue = _traced_step
