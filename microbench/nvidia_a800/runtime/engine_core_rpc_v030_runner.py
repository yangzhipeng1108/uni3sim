"""CPU-only vLLM0.30 two-worker RPC/Future transport mechanism.

Actual MessageQueue, collective_rpc, FutureWrapper and WorkerProc dispatch/
output methods handle synthetic typed messages. There is no Scheduler run,
model, GPU, kernel, measured Serving input or production duration.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import gc
import gzip
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import queue
import threading
import time
from types import MethodType, SimpleNamespace

# Process-local device visibility, also inherited by spawned CPU workers.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from vllm.distributed.device_communicators import shm_broadcast
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.outputs import ModelRunnerOutput
from vllm.utils.gc_utils import freeze_gc_heap


COUNTS = (1, 8, 32, 128)
PHASES = ('fresh64', 'cached64', 'cached1024')
DEPTHS = (1, 2)
WARMUP, SAMPLES = 12, 63


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def payload(count, phase):
    ids = [f'synthetic-{i:012}' for i in range(count)]
    cached = CachedRequestData.make_empty()
    fresh = []
    if phase == 'fresh64':
        for i, name in enumerate(ids):
            tokens = [(j + i) % 257 + 1 for j in range(64)]
            fresh.append(NewRequestData(req_id=name, prompt_token_ids=tokens,
                prefill_token_ids=tokens, mm_features=[],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
                pooling_params=None, block_ids=([i * 4 + j for j in range(4)],),
                num_computed_tokens=0, lora_request=None))
        scheduled = dict.fromkeys(ids, 64)
    elif phase in ('cached64', 'cached1024'):
        computed = {'cached64': 64, 'cached1024': 1024}[phase]
        cached.req_ids = ids
        cached.new_block_ids = [None for _ in ids]
        cached.num_computed_tokens = [computed for _ in ids]
        cached.num_output_tokens = [1 for _ in ids]
        scheduled = dict.fromkeys(ids, 1)
    else:
        raise ValueError('unknown synthetic phase')
    result = SchedulerOutput(scheduled_new_reqs=fresh, scheduled_cached_reqs=cached,
        num_scheduled_tokens=scheduled, total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0], finished_req_ids=set(), free_encoder_mm_hashes=[],
        preempted_req_ids=set(), new_block_ids_to_zero=([i for i in range(count * 4)] if fresh else []))
    assert len(result.num_scheduled_tokens) == count and result.total_num_scheduled_tokens <= 8192
    return result


class DummyWorker:
    """Ready CPU output, so no model/Device delay enters transport samples."""
    def __init__(self):
        self.ids = ()
        self.execute_calls = 0
        self.sample_calls = 0
        self.rows = 0

    def execute_model(self, schedule):
        self.ids = tuple(schedule.num_scheduled_tokens)
        self.execute_calls += 1
        self.rows += len(self.ids)
        return None

    def sample_tokens(self, grammar):
        received = time.perf_counter_ns()
        assert grammar is None and self.ids
        ids = list(self.ids)
        output = ModelRunnerOutput(req_ids=ids, req_id_to_index={key: i for i, key in enumerate(ids)},
            sampled_token_ids=[[i] for i in range(len(ids))], prompt_logprobs_dict={})
        output._probe_worker_received_ns = received
        output._probe_worker_ready_ns = time.perf_counter_ns()
        self.sample_calls += 1
        return output


def worker(input_handle, pipe, rank, freeze_startup):
    incoming = shm_broadcast.MessageQueue.create_from_handle(input_handle, rank)
    outgoing = shm_broadcast.MessageQueue(1, 1)
    actor = SimpleNamespace(worker=DummyWorker(), rank=rank, use_async_scheduling=True,
        worker_response_mq=outgoing, async_output_queue=queue.Queue())
    actor.enqueue_output = MethodType(WorkerProc.enqueue_output, actor)
    actor.handle_output = MethodType(WorkerProc.handle_output, actor)
    stop = object()

    def emit():
        while True:
            value = actor.async_output_queue.get()
            if value is stop:
                return
            actor.enqueue_output(value)

    thread = threading.Thread(target=emit, name='synthetic-ready-output')
    thread.start()
    pipe.send(outgoing.export_handle())
    incoming.wait_until_ready()
    outgoing.wait_until_ready()
    if freeze_startup:
        freeze_gc_heap()
    pipe.send({'rank': rank, 'ready': True, 'startup_heap_frozen': freeze_startup})
    while True:
        message = incoming.dequeue(indefinite=True)
        if message[0] == 'probe_stop':
            break
        WorkerProc._execute_worker_rpc(actor, message)
    actor.async_output_queue.put(stop)
    thread.join(timeout=10)
    assert not thread.is_alive()
    pipe.send({'rank': rank, 'execute_calls': actor.worker.execute_calls,
        'sample_calls': actor.worker.sample_calls, 'rows': actor.worker.rows,
        'output_thread_stopped': True})
    pipe.close()
    incoming.shutdown()
    outgoing.shutdown()
    del actor.worker_response_mq
    del incoming, outgoing, actor
    gc.collect()


def execute(output, observe_waits, freeze_startup):
    context = multiprocessing.get_context('spawn')
    incoming = shm_broadcast.MessageQueue(2, 2)
    children, pipes, outgoing = [], [], []
    for rank in (0, 1):
        parent, child = context.Pipe()
        process = context.Process(target=worker, args=(incoming.export_handle(), child, rank, freeze_startup))
        process.start()
        child.close()
        children.append(process)
        pipes.append(parent)
    for parent in pipes:
        assert parent.poll(45), 'worker did not publish its queue handle'
        outgoing.append(shm_broadcast.MessageQueue.create_from_handle(parent.recv(), 0))
    incoming.wait_until_ready()
    for mq in outgoing:
        mq.wait_until_ready()
    for parent in pipes:
        assert parent.poll(45), 'worker did not finish its startup state'
        assert parent.recv()['startup_heap_frozen'] is freeze_startup
    executor = SimpleNamespace(is_failed=False, rpc_broadcast_mq=incoming,
        response_mqs=outgoing, output_rank=0, futures_queue=deque(),
        kv_output_aggregator=None, ec_output_aggregator=None)
    executor.collective_rpc = MethodType(MultiprocExecutor.collective_rpc, executor)
    if freeze_startup:
        freeze_gc_heap()
    cells, expected_calls, expected_rows = [], 0, 0
    write_waits, gc_events = [], []
    original_acquire = shm_broadcast.MessageQueue.acquire_write

    @contextmanager
    def observed_acquire(mq, *args, **kwargs):
        assert mq is incoming
        started = time.perf_counter_ns()
        with original_acquire(mq, *args, **kwargs) as buffer:
            acquired = time.perf_counter_ns()
            write_waits.append({'start_ns': started, 'acquired_ns': acquired})
            yield buffer

    def collection_event(phase, info):
        gc_events.append({'phase': phase, 'time_ns': time.perf_counter_ns(),
            'generation': info['generation']})

    if observe_waits:
        shm_broadcast.MessageQueue.acquire_write = observed_acquire
        gc.callbacks.append(collection_event)
    try:
        for repeat in range(2):
            grid = [(phase, count, depth) for phase in PHASES for count in COUNTS for depth in DEPTHS]
            for phase, count, depth in (grid if repeat == 0 else list(reversed(grid))):
                schedule = payload(count, phase)
                samples = []
                for trial in range(WARMUP + SAMPLES):
                    frames, started = [], time.perf_counter_ns()
                    for _ in range(depth):
                        t0 = time.perf_counter_ns()
                        execute_future = MultiprocExecutor.execute_model(executor, schedule, non_block=True)
                        t1 = time.perf_counter_ns()
                        sample_future = MultiprocExecutor.sample_tokens(executor, None, non_block=True)
                        t2 = time.perf_counter_ns()
                        assert execute_future is not sample_future
                        frames.append((execute_future, sample_future, t0, t1, t2))
                    records = []
                    for execute_future, sample_future, t0, t1, t2 in frames:
                        result = sample_future.result()
                        end = time.perf_counter_ns()
                        assert execute_future.done() and execute_future.result() is None
                        assert result.req_ids == list(schedule.num_scheduled_tokens)
                        assert result.sampled_token_ids == [[i] for i in range(count)]
                        assert t0 <= result._probe_worker_received_ns <= result._probe_worker_ready_ns <= end
                        records.append({'execute_enqueue_ns': t1 - t0, 'sample_enqueue_ns': t2 - t1,
                            'pair_enqueue_ns': t2 - t0, 'pair_roundtrip_ns': end - t0,
                            'ready_to_receive_ns': end - result._probe_worker_ready_ns,
                            'dummy_output_construct_ns': result._probe_worker_ready_ns - result._probe_worker_received_ns,
                            'submit_start_ns': t0, 'execute_submitted_ns': t1,
                            'sample_submitted_ns': t2, 'response_received_ns': end})
                    assert not executor.futures_queue
                    expected_calls += depth
                    expected_rows += count * depth
                    samples.append({'trial': trial, 'warmup': trial < WARMUP,
                        'group_wall_ns': time.perf_counter_ns() - started, 'frames': records})
                cell = {'repeat': repeat, 'phase': phase, 'request_count': count, 'depth': depth,
                    'scheduled_tokens': schedule.total_num_scheduled_tokens,
                    'synthetic_pickle_bytes': len(pickle.dumps(('execute_model', (schedule,), {}, 0), protocol=pickle.HIGHEST_PROTOCOL)),
                    'samples': samples}
                cells.append(cell)
                write(output / 'partial_samples.json', cells)
                print(json.dumps({key: cell[key] for key in ('repeat', 'phase', 'request_count', 'depth')}), flush=True)
    finally:
        incoming.enqueue(('probe_stop', (), {}, 0))
        worker_receipts = []
        for parent, process in zip(pipes, children, strict=True):
            assert parent.poll(15), 'worker did not return its completion counters'
            worker_receipts.append(parent.recv())
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                raise RuntimeError('CPU transport worker failed to stop')
            assert process.exitcode == 0
            parent.close()
        write(output / 'worker_receipts.json', worker_receipts)
        if observe_waits:
            shm_broadcast.MessageQueue.acquire_write = original_acquire
            gc.callbacks.remove(collection_event)
            write(output / 'parent_waits_and_gc.json', {'write_waits': write_waits, 'gc_events': gc_events})
        incoming.shutdown()
        for mq in outgoing:
            mq.shutdown()
        del executor.rpc_broadcast_mq
        del executor.response_mqs
        del incoming, outgoing, executor
        gc.collect()
    assert len(cells) == 48
    assert all(row['execute_calls'] == row['sample_calls'] == expected_calls
        and row['rows'] == expected_rows for row in worker_receipts)
    return {'status': 'complete_independent_rpc_control', 'cells': cells,
        'worker_receipts': worker_receipts, 'batches_per_worker': expected_calls,
        'request_rows_per_worker': expected_rows, 'rpc_calls_per_worker': expected_calls * 2,
        'source_future_identities_per_batch': 2, 'model_or_gpu_executed': False,
        'production_duration_consumed': False, 'cpu_affinity_changed': False,
        'parent_wait_gc_observer_enabled': observe_waits,
        'startup_heap_frozen': freeze_startup}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--observe-waits', action='store_true')
    parser.add_argument('--freeze-startup', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = {'runner_sha256': sha(Path(__file__)),
        'source_sha256': {str(Path(inspect.getfile(obj)).resolve()): sha(Path(inspect.getfile(obj)).resolve())
            for obj in (shm_broadcast.MessageQueue, MultiprocExecutor, SchedulerOutput, ModelRunnerOutput, SamplingParams, freeze_gc_heap)},
        'spinloop_extension_enabled': shm_broadcast.SPINLOOP_EXT_ENABLED,
        'CPU_visible_affinity': sorted(os.sched_getaffinity(0)),
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'parent_wait_gc_observer_enabled': args.observe_waits,
        'startup_heap_frozen': args.freeze_startup,
        'queue': {'readers': 2, 'local_readers': 2, 'max_chunk_bytes': 24 * 1024 * 1024, 'max_chunks': 10},
        'grid': {'counts': COUNTS, 'phases': PHASES, 'depths': DEPTHS,
            'repeats': 2, 'warmup': WARMUP, 'samples': SAMPLES},
        'scope': 'Independent typed transport/ready-output control; no Scheduler execution, model, GPU or Serving trace',
        'source_methods': ['MultiprocExecutor.execute_model/sample_tokens/collective_rpc',
            'FutureWrapper.result', 'WorkerProc._execute_worker_rpc/handle_output/enqueue_output']}
    write(args.output / 'source.json', source)
    (args.output / 'runner_source.py.gz').write_bytes(gzip.compress(Path(__file__).read_bytes(), mtime=0))
    for phase in PHASES:
        for count in COUNTS:
            schedule = payload(count, phase)
            restored = pickle.loads(pickle.dumps(schedule, protocol=pickle.HIGHEST_PROTOCOL))
            assert restored.num_scheduled_tokens == schedule.num_scheduled_tokens
    if args.fixture:
        write(args.output / 'fixture.json', {'status': 'typed_payload_roundtrip_passed', 'payloads': 12})
        return
    started = time.perf_counter()
    try:
        result = execute(args.output, args.observe_waits, args.freeze_startup)
        result['wall_s'] = time.perf_counter() - started
        write(args.output / 'samples.json', result)
    except BaseException as error:
        write(args.output / 'failure.json', {'status': 'failed', 'wall_s': time.perf_counter() - started,
            'exception_type': type(error).__name__, 'exception': str(error)})
        raise


if __name__ == '__main__':
    main()
