"""Collection-only Ascend trace bridge for an isolated native profiler copy.

CPU module scopes and torch_to_npu flows establish ownership. Only Ascend
Hardware task intervals contribute latency, with a union per invocation.
"""
from collections import defaultdict
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import uuid


def _ns(value):
    scaled = Decimal(str(value)) * 1000
    if not scaled.is_finite() or scaled != scaled.to_integral_value():
        raise ValueError("NPU timestamp must have exact nanosecond precision")
    return int(scaled)


def _union(intervals):
    spans = sorted(intervals)
    if not spans:
        return 0
    left, right = spans[0]
    if left >= right:
        raise ValueError("nonpositive NPU task")
    total = 0
    for start, end in spans[1:]:
        if start >= end:
            raise ValueError("nonpositive NPU task")
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left


def parse_trace(events):
    """Produce native summary entries while requiring every device task owner."""
    devices = {e['pid'] for e in events if e['ph'] == 'M'
               and e['name'] == 'process_name' and e['args']['name'] == 'Ascend Hardware'}
    if len(devices) != 1:
        raise ValueError("expected one Ascend Hardware process")
    tasks = [e for e in events if e['ph'] == 'X' and e['pid'] in devices]
    modules = []
    for event in events:
        if event['ph'] == 'X' and event['name'].startswith('nn.Module: '):
            match = re.fullmatch(r'nn.Module: (.+)_[0-9]+', event['name'])
            if match is None:
                raise ValueError("unknown NPU module label")
            modules.append({'name': match[1], 'event': event,
                            'start': _ns(event['ts']),
                            'end': _ns(event['ts']) + _ns(event['dur']),
                            'children': [], 'tasks': []})
    roots = []
    for module in modules:
        candidates = [parent for parent in modules if parent is not module
                      and parent['event']['pid'] == module['event']['pid']
                      and parent['event']['tid'] == module['event']['tid']
                      and parent['start'] <= module['start'] and module['end'] <= parent['end']]
        candidates.sort(key=lambda m: m['end'] - m['start'])
        if candidates:
            if len(candidates) > 1 and candidates[0]['end'] - candidates[0]['start'] == candidates[1]['end'] - candidates[1]['start']:
                raise ValueError("ambiguous NPU module parent")
            if candidates[0]['end'] - candidates[0]['start'] <= module['end'] - module['start']:
                raise ValueError("identical NPU module intervals")
            candidates[0]['children'].append(module)
        else:
            roots.append(module)
    flows = defaultdict(dict)
    for event in events:
        if 'cat' in event and event['cat'] == 'async_npu' and event['name'] == 'torch_to_npu':
            phase = event['ph']
            if phase not in ('s', 'f') or phase in flows[event['id']]:
                raise ValueError("duplicate or invalid torch_to_npu flow")
            flows[event['id']][phase] = event
    by_endpoint = defaultdict(list)
    for flow in flows.values():
        if set(flow) != {'s', 'f'}:
            raise ValueError("incomplete torch_to_npu flow")
        end = flow['f']
        by_endpoint[(end['pid'], end['tid'], _ns(end['ts']))].append(flow['s'])
    identities = set()
    excluded_control, excluded_outside = [], []
    for task in tasks:
        task_type = task['args']['Task Type']
        if task_type in ('EVENT_RECORD', 'EVENT_WAIT'):
            excluded_control.append(task)
            continue
        if task_type not in ('KERNEL_AICORE', 'KERNEL_AIVEC', 'KERNEL_MIX_AIC', 'PCIE_DMA_SQE', 'SDMA_SQE'):
            raise ValueError(f'unsupported NPU task type: {task_type}')
        start = _ns(task['ts'])
        identity = (task['pid'], task['tid'], start)
        if identity in identities:
            raise ValueError("duplicate NPU device task endpoint")
        identities.add(identity)
        launches = by_endpoint[identity]
        if len(launches) != 1:
            raise ValueError("NPU task has no unique native launch flow")
        launch, = launches
        candidates = [m for m in modules if m['event']['pid'] == launch['pid']
                      and m['event']['tid'] == launch['tid']
                      and m['start'] <= _ns(launch['ts']) < m['end']]
        candidates.sort(key=lambda m: m['end'] - m['start'])
        # Like the native vLLM module tree, runner setup outside nn.Module
        # calls is outside the layerwise measurement boundary. Retain its
        # identity explicitly, rather than assigning it to a nearby module.
        if not candidates:
            excluded_outside.append(task)
            continue
        if len(candidates) > 1 and candidates[0]['end'] - candidates[0]['start'] == candidates[1]['end'] - candidates[1]['start']:
            raise ValueError("NPU launch has no unique innermost module")
        candidates[0]['tasks'].append(task)
    if not tasks:
        raise ValueError("no native NPU device tasks")
    summary = {}

    def collect(module, path=()):
        descendants = list(module['tasks'])
        path = path + (module['name'],)
        for child in module['children']:
            descendants.extend(collect(child, path))
        if not descendants:
            return descendants
        active = _union([(_ns(t['ts']), _ns(t['ts']) + _ns(t['dur'])) for t in descendants]) / 1000
        if path not in summary:
            summary[path] = {'entry': {'name': module['name'], 'cuda_time_us': 0.0,
                                      'pct_cuda_time': 0.0, 'invocations': 0}, 'children': []}
        entry = summary[path]['entry']
        entry['cuda_time_us'] += active
        entry['invocations'] += 1
        for task in module['tasks']:
            leaf_path = path + (task['name'],)
            if leaf_path not in summary:
                summary[leaf_path] = {'entry': {'name': task['name'], 'cuda_time_us': 0.0,
                                               'pct_cuda_time': 0.0, 'invocations': 0}, 'children': []}
            summary[leaf_path]['entry']['cuda_time_us'] += _ns(task['dur']) / 1000
            summary[leaf_path]['entry']['invocations'] += 1
        return descendants

    retained = [t for root in roots for t in collect(root)]
    if len(retained) + len(excluded_control) + len(excluded_outside) != len(tasks):
        raise ValueError("NPU module tree changed device task coverage")
    total = sum(value['entry']['cuda_time_us'] for path, value in summary.items() if len(path) == 1)
    tree = []
    for path, value in sorted(summary.items()):
        value['entry']['pct_cuda_time'] = value['entry']['cuda_time_us'] / total * 100
        if len(path) == 1:
            tree.append(value)
        else:
            summary[path[:-1]]['children'].append(value)
    return {'summary_stats': tree, 'device_task_count': len(tasks),
            'retained_device_tasks': len(retained),
            'excluded_runtime_control': excluded_control,
            'excluded_outside_module_calls': excluded_outside,
            'measurement_protocol': 'npu-active-union-dummy-kv-query-v1'}


class layerwise_profile:
    """Native-profile-shaped context backed by real torch_npu device traces."""
    def __enter__(self):
        import torch_npu.profiler as p
        self.output = Path(os.environ['LLMSIM_RAW_OUTPUT']) / 'npu-traces' / uuid.uuid4().hex
        self.profiler = p.profile(activities=[p.ProfilerActivity.CPU,p.ProfilerActivity.NPU],
            record_shapes=True, with_stack=True, with_modules=True,
            on_trace_ready=p.tensorboard_trace_handler(str(self.output)))
        self.profiler.__enter__()
        return self

    def __exit__(self, kind, value, traceback):
        self.profiler.__exit__(kind, value, traceback)
        if kind is not None:
            return False
        traces = list(self.output.rglob('trace_view.json'))
        if len(traces) != 1:
            raise ValueError("NPU profiler did not produce exactly one device trace")
        self.stats = parse_trace(json.loads(traces[0].read_text()))
        (self.output / 'measurement.json').write_text(json.dumps(self.stats, indent=2))
        self.results = self
        return False

    def convert_stats_to_dict(self):
        return self.stats


def initialize_dummy_cache(runner, batch):
    """BF16 FullAttentionSpec cache pages, initialized outside timed forwards."""
    import torch
    from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    requests = batch.scheduled_new_reqs
    if not requests or not any(r.num_computed_tokens > 0 for r in requests):
        return
    config = runner.kv_cache_config
    context = runner.compilation_config.static_forward_context
    if runner.vllm_config.cache_config.cache_dtype != 'auto':
        raise ValueError('Ascend collection bridge supports BF16 KV only')
    with torch.inference_mode():
        for index, group in enumerate(config.kv_cache_groups):
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec) or spec.dtype != torch.bfloat16:
                raise TypeError('unsupported NPU KV spec')
            pages = [p for r in requests for p in r.block_ids[index]]
            if not pages or pages != list(range(len(pages))) or len(pages) > config.num_blocks:
                raise ValueError('NPU KV pages must be contiguous and in bounds')
            for name in group.layer_names:
                layer = context[name]
                shape = layer.get_attn_backend().get_kv_cache_shape(-1,spec.block_size,spec.num_kv_heads,spec.head_size)
                if shape != (2,-1,spec.block_size,spec.num_kv_heads,spec.head_size):
                    raise ValueError('unsupported Ascend backend KV layout')
                cache, = layer.kv_cache  # vLLM virtual-engine wrapper
                if not isinstance(cache, (list, tuple)) or len(cache) != 2:
                    raise ValueError('expected native Ascend key/value cache pair')
                for tensor in cache:
                    if (tensor.shape[0] != config.num_blocks or tensor.dtype != spec.dtype
                            or tensor.numel() != config.num_blocks * spec.block_size * spec.num_kv_heads * spec.head_size):
                        raise ValueError('NPU physical KV pages mismatch')
                    initialize_single_dummy_weight(tensor.narrow(0,0,len(pages)))
    torch.npu.synchronize()


_BLOCK16_CONFIGURED = False


def _preserve_validated_block16(config):
    """Retain the requested geometry for the two validated BF16 dense models.

    The vendor policy otherwise replaces 16 with 128 when chunking is enabled.
    Native attention/cache operations and metadata builders remain unchanged.
    """
    import torch

    model = config.model_config
    if (config.cache_config.block_size != 16
            or config.cache_config.cache_dtype != 'auto'
            or model.dtype != torch.bfloat16
            or model.hf_text_config.model_type not in ('qwen3', 'llama')
            or model.get_head_size() != 128
            or model.hf_text_config.num_attention_heads != 32
            or model.hf_text_config.num_key_value_heads != 8
            or config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1):
        raise ValueError('block16 collection capability is not validated for this configuration')


def configure_block16():
    """Opt-in private-process capability adapter, guarded by real operator probes.

    Both FIA and paged-attention must have succeeded at block16 and block128
    against the CPU reference. The exact vendor sizing functions are bound by
    source fingerprints. No device implementation or measured time is changed.
    """
    global _BLOCK16_CONFIGURED
    if _BLOCK16_CONFIGURED:
        return
    import hashlib
    import inspect
    import torch
    import torch_npu
    import vllm
    import vllm_ascend.platform as platform
    import vllm_ascend.utils as utils
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

    evidence_path = Path(os.environ['LLMSIM_BLOCK16_EVIDENCE'])
    evidence = json.loads(evidence_path.read_text())
    if evidence['schema_version'] != 1:
        raise ValueError('unsupported block capability evidence')
    versions = {'torch': str(torch.__version__), 'torch_npu': str(torch_npu.__version__),
                'vllm': str(vllm.__version__)}
    if evidence['versions'] != versions:
        raise ValueError('operator capability evidence comes from a different runtime')
    observed = set()
    for item in evidence['probes']:
        path = Path(item['path'])
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError('operator capability evidence content changed')
        result = json.loads(payload)
        command = result['command']
        operator = command[command.index('--operator') + 1] if '--operator' in command else 'fia'
        if (result['status'] != 'completed' or result['exit_code'] != 0
                or result['hardware'] != 'ascend_910b'
                or result['kind'] != 'native_operator_capability_no_timing_table'
                or result['torch'] != versions['torch']
                or result['torch_npu'] != versions['torch_npu']
                or result['cache_shape'][1] != result['block_size']
                or item['operator'] != operator):
            raise ValueError('invalid block capability probe')
        observed.add((item['operator'], result['block_size']))
    if observed != {('fia', 16), ('fia', 128), ('paged', 16), ('paged', 128)} or len(evidence['probes']) != 4:
        raise ValueError('both native attention interfaces require two block-size controls')
    original_sources = {
        'refresh_block_size': hashlib.sha256(inspect.getsource(utils.refresh_block_size).encode()).hexdigest(),
        'get_supported_kernel_block_sizes': hashlib.sha256(inspect.getsource(
            AscendAttentionBackend.get_supported_kernel_block_sizes).encode()).hexdigest()}
    if original_sources != evidence['native_sources'] or platform.refresh_block_size is not utils.refresh_block_size:
        raise ValueError('native block sizing implementation changed')
    if AscendAttentionBackend.get_supported_kernel_block_sizes() != [128]:
        raise ValueError('unexpected native advertised block sizes')
    # This is a collector-private adapter. The same native operators receive
    # actual block16 caches, rather than relabelled block128 measurements.
    utils.refresh_block_size = _preserve_validated_block16
    platform.refresh_block_size = _preserve_validated_block16
    AscendAttentionBackend.get_supported_kernel_block_sizes = staticmethod(lambda: [16])
    _BLOCK16_CONFIGURED = True
    output = Path(os.environ['LLMSIM_RAW_OUTPUT']) / f'block16-adapter-{os.getpid()}.json'
    output.write_text(json.dumps({'adapter': 'validated-native-block16-capability-v1',
        'evidence': str(evidence_path), 'evidence_sha256': hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        'original_sources': original_sources, 'versions': versions,
        'requested_block_size': 16, 'native_advertised_block_sizes': [128],
        'collector_advertised_block_sizes': [16],
        'kernel_implementation': 'unchanged native FIA, paged-attention and cache operations'}, indent=2)+'\n')
