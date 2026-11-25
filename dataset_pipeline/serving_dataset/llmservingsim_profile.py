"""Versioned collection-only adapters for a private LLMServingSim execution copy."""

from pathlib import Path
import json
import subprocess


def require_capture_owner(owner: str) -> str:
    """Require an explicit task identity before starting hardware collection."""
    if not isinstance(owner, str) or not owner or owner != owner.strip():
        raise ValueError("capture owner must be a non-empty task identity without surrounding whitespace")
    return owner


def validate_resume_identity(recorded: dict, owner: str, hardware: str) -> None:
    """Prevent checkpoints belonging to another task or hardware from reuse."""
    require_capture_owner(owner)
    if recorded["owner"] != owner or recorded["hardware"] != hardware:
        raise ValueError("cannot resume another task or hardware")


def prepare_metax_catalog(source: Path, destination: Path) -> None:
    """Bind MetaX subclasses to the same native logical roles and parents.

    The three classes inherit RMSNorm, RotaryEmbedding and SiluAndMul in
    vllm_metax 0.24.0 customized/ops. No operators or timing rules change.
    Architecture YAML content participates in native acquisition identity.
    """
    import yaml

    destination.mkdir(parents=True, exist_ok=True)
    for architecture in ("qwen3", "llama"):
        document = yaml.safe_load((source / f"{architecture}.yaml").read_text())
        catalog = document["catalog"]["dense"]
        expected = {"layernorm": "RMSNorm", "final_layernorm": "RMSNorm",
                    "act_fn": "SiluAndMul"}
        if architecture == "qwen3":
            expected["qk_norm"] = "RMSNorm"
        for role, native in expected.items():
            if catalog[role]["vllm"] != native:
                raise ValueError(f"unexpected {architecture}/{role} class")
            catalog[role]["vllm"] = [native, f"Maca{native}"]
        rotary = catalog["rotary_emb"]["vllm"]
        if rotary == "RotaryEmbedding":
            classes = [rotary]
        elif rotary == ["Llama3RotaryEmbedding", "RotaryEmbedding"]:
            classes = list(rotary)
        else:
            raise ValueError("unexpected native rotary classes")
        catalog["rotary_emb"]["vllm"] = classes + ["MacaRotaryEmbedding"]
        (destination / f"{architecture}.yaml").write_text(yaml.safe_dump(document, sort_keys=False))


def prepare_legacy_kineto(source: Path, destination: Path) -> None:
    """Use logical/OS CPU thread IDs and native linked correlations.

    Torch 2.8 and MetaX Torch 2.10 omit activity_type(). Event times, device
    unions, strict ownership and event-coverage guards stay unchanged.
    """
    replacements = [
            ("""        activity = event.activity_type()
        if activity == 'python_function':
            python_resources[(event.start_ns(), event.end_ns(),
                              event.start_thread_id())].add(event.device_resource_id())
        elif activity in ('cuda_runtime', 'cuda_driver'):
            launches[event.correlation_id()].append(event)""",
             """        # Torch 2.8 operator events use resource 0, while CUPTI records
        # carry the OS thread and linked operator correlation. Only those
        # linked CPU activities establish the logical-to-OS thread mapping.
        # No event interval, ownership requirement or latency is changed.
        if event.linked_correlation_id() != 0:
            python_resources[event.start_thread_id()].add(event.device_resource_id())
            launches[event.correlation_id()].append(event)"""),
            ("resources = python_resources[(start, start + duration, event.start_tid)]",
             "resources = python_resources[event.start_tid]"),
    ]
    original = source.read_text()
    for before, after in replacements:
        if original.count(before) != 1:
            raise ValueError("unexpected upstream Kineto ownership fragment")
        original = original.replace(before, after, 1)
    destination.write_text(original)


def prepare_vllm_011_profiler(runtime: Path, source_manifest: Path, evidence: Path) -> None:
    """Map native V2 request plumbing to the installed vLLM 0.11 V1 contract.

    Query/history tensors, native kernels, measurement and lookup remain unchanged.
    The original external checkout is never modified.
    """
    runtime, evidence = runtime.resolve(), evidence.resolve()
    with source_manifest.open() as stream:
        source = json.load(stream)
    revision = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    if revision != source["commit"]:
        raise ValueError("profiler execution copy differs from the source binding")
    if "experiment_logs/evaluation/llmservingsim" not in str(runtime):
        raise ValueError("profiler patch requires the task execution copy")
    hooks = runtime / "profiler/core/hooks"
    changes = {
        "batch.py": [
            ("                prefill_token_ids=[1] * total_len,\n", ""),
            ("    scheduler_output = SchedulerOutput.make_empty()",
             "    from .midlayer_vllm_011 import empty_scheduler_output\n    scheduler_output = empty_scheduler_output()"),
        ],
        "history.py": [
            ("from .dummy_cache import initialize_dummy_cache",
             "from .midlayer_vllm_011 import initialize_dummy_cache"),
            ("requests.append(replace(request, prompt_token_ids=tokens,\n                                prefill_token_ids=tokens))",
             "requests.append(replace(request, prompt_token_ids=tokens))"),
            ("        batch = type(template).make_empty()",
             "        from .midlayer_vllm_011 import empty_scheduler_output\n        batch = empty_scheduler_output()"),
            ("            prompt_token_ids=list(old.prompt_token_ids),\n            prefill_token_ids=list(old.prefill_token_ids))",
             "            prompt_token_ids=list(old.prompt_token_ids))"),
        ],
    }
    prepared = {}
    for name, replacements in changes.items():
        original = subprocess.check_output(["git", "-C", str(runtime), "show", f"HEAD:profiler/core/hooks/{name}"], text=True)
        for before, after in replacements:
            if original.count(before) != 1:
                raise ValueError(f"unexpected upstream profiler fragment: {name}")
            original = original.replace(before, after, 1)
        prepared[name] = original
    for name, content in prepared.items():
        (hooks / name).write_text(content)
    (hooks / "midlayer_vllm_011.py").write_text('''"""Exact vLLM 0.11 scheduler-output construction for collection only."""
from importlib.metadata import version
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

def empty_scheduler_output():
    if version("vllm") != "0.11.0":
        raise ValueError("this collection adapter requires vLLM 0.11.0")
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(req_ids=[], resumed_from_preemption=[],
            new_token_ids=[], new_block_ids=[], num_computed_tokens=[]),
        num_scheduled_tokens={}, total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
        num_common_prefix_blocks=[], finished_req_ids=set(), free_encoder_mm_hashes=[],
        structured_output_request_ids={}, grammar_bitmask=None)

def initialize_dummy_cache(runner, batch):
    import torch
    from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    requests = batch.scheduled_new_reqs
    if not requests or not any(request.num_computed_tokens > 0 for request in requests):
        return
    if runner.vllm_config.cache_config.cache_dtype != "auto":
        raise ValueError("vLLM 0.11 adapter supports the approved BF16 KV only")
    config = runner.kv_cache_config
    context = runner.compilation_config.static_forward_context
    with torch.inference_mode():
        for index, group in enumerate(config.kv_cache_groups):
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec) or spec.dtype != torch.bfloat16:
                raise TypeError("unsupported profiling cache spec")
            pages = [page for request in requests for page in request.block_ids[index]]
            if not pages or pages != list(range(len(pages))) or len(pages) > config.num_blocks:
                raise ValueError("dummy KV pages must be contiguous and in bounds")
            for name in group.layer_names:
                layer = context[name]
                cache, = layer.kv_cache
                shape = layer.get_attn_backend().get_kv_cache_shape(
                    -1, spec.block_size, spec.num_kv_heads, spec.head_size,
                    cache_dtype_str="auto")
                axes = [axis for axis, size in enumerate(shape) if size == -1]
                if len(axes) != 1 or cache.shape[axes[0]] != config.num_blocks:
                    raise ValueError("cannot identify native cache block axis")
                view = cache.narrow(axes[0], 0, len(pages))
                module = torch.nn.Module()
                module.register_buffer("cache", view)
                initialize_dummy_weights(module)
    torch.cuda.synchronize()
''')
    measurement = subprocess.check_output(["git", "-C", str(runtime), "show", "HEAD:profiler/core/measurement.py"], text=True)
    fragment = "'sampler_shim.py', 'history.py', 'dummy_cache.py')]"
    if measurement.count(fragment) != 1:
        raise ValueError("unexpected upstream measurement identity")
    (runtime / "profiler/core/measurement.py").write_text(measurement.replace(fragment,
        "'sampler_shim.py', 'history.py', 'dummy_cache.py', 'midlayer_vllm_011.py')]", 1))
    evidence.mkdir(parents=True, exist_ok=True)
    original_ownership = evidence / "activity_ownership.original.py"
    original_ownership.write_text(subprocess.check_output(["git", "-C", str(runtime), "show",
        "HEAD:profiler/core/hooks/activity_ownership.py"], text=True))
    prepare_legacy_kineto(original_ownership, hooks / "activity_ownership.py")
    diff = subprocess.check_output(["git", "-C", str(runtime), "diff", "--", "profiler"], text=True)
    (evidence / "profiler-vllm-011.patch").write_text(diff)
    (evidence / "midlayer_vllm_011.py").write_text((hooks / "midlayer_vllm_011.py").read_text())


def prepare_ascend_profiler(source: Path, destination: Path, *, block16_capability: bool = False) -> None:
    """Generate an isolated Ascend collection patch from the bound native tree."""
    changes = {
        'core/engine.py': [
            ('torch.cuda.empty_cache()', 'torch.npu.empty_cache()'),
            ('    return RuntimeLimits(\n',
             '    if (args is not None and args.block_size is not None\n'
             '            and cfg.cache_config.block_size != args.block_size):\n'
             '        raise ValueError(f"Requested block_size={args.block_size}, "\n'
             '                         f"native Ascend engine resolved {cfg.cache_config.block_size}; "\n'
             '                         "target geometry cannot be preserved")\n'
             '    return RuntimeLimits(\n')],
        'core/hooks/history.py': [
            ('from .dummy_cache import initialize_dummy_cache', 'from .midlayer_npu import initialize_dummy_cache'),
            ('torch.cuda.synchronize()', 'torch.npu.synchronize()')],
        'core/hooks/extension.py': [
            ('from vllm.profiler.layerwise_profile import layerwise_profile',
             'from .midlayer_npu import layerwise_profile'),
            ('        from .cuda_timing import from_vllm\n', ''),
            ('        measured, _ = from_vllm(hook.results)', '        measured = hook.results')],
        'core/measurement.py': [
            ("PROTOCOL = 'cuda-active-union-dummy-kv-query-v4'", "PROTOCOL = 'npu-active-union-dummy-kv-query-v1'"),
            ("'sampler_shim.py', 'history.py', 'dummy_cache.py')]",
             "'sampler_shim.py', 'history.py', 'dummy_cache.py', 'midlayer_npu.py')]")],
        'core/writer.py': [('torch.cuda.is_available()', 'torch.npu.is_available()'),
                           ('torch.cuda.get_device_name(0)', 'torch.npu.get_device_name(0)')],
    }
    for relative, replacements in changes.items():
        original = (source / relative).read_text()
        for before, after in replacements:
            count = original.count(before)
            # Two cleanup calls and two worker entry points are intentional.
            expected = 2 if (relative, before) in (
                ('core/engine.py', 'torch.cuda.empty_cache()'),
                ('core/hooks/extension.py', 'from vllm.profiler.layerwise_profile import layerwise_profile')) else 1
            if count != expected:
                raise ValueError(f'unexpected Ascend adaptation fragment: {relative}')
            original = original.replace(before, after)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if block16_capability and relative in ('core/engine.py', 'core/hooks/extension.py'):
            marker = 'from __future__ import annotations\n'
            if original.count(marker) != 1:
                raise ValueError('unexpected native module import boundary')
            original = original.replace(marker, marker +
                '\nfrom profiler.core.hooks.midlayer_npu import configure_block16\nconfigure_block16()\n', 1)
        target.write_text(original)
    bridge = Path(__file__).with_name('llmservingsim_npu.py')
    (destination / 'core/hooks/midlayer_npu.py').write_bytes(bridge.read_bytes())
    categories = (source / 'core/categories.py').read_text()
    before = '''    # 16 .. 63 — step of 4. Transition regime.
    pts.extend(range(16, min(64, max_tokens + 1), 4))
    # 64 .. max — step of 16. Longer-chunk regime.
    pts.extend(range(64, max_tokens + 1, 16))'''
    after = '''    # Collection-only Ascend pilot grid: retain native 1..15, then
    # fixed powers of two. Native interpolation and domain limits stay intact.
    # This bounds per-shot CANN trace parsing cost; no error data select points.
    point = 16
    while point <= max_tokens:
        pts.append(point)
        point *= 2'''
    if categories.count(before) != 1:
        raise ValueError('unexpected native token grid')
    (destination / 'core/categories.py').write_text(categories.replace(before,after,1))


def prepare_ascend_catalog(source: Path, destination: Path) -> None:
    """Expand verified vllm_ascend 0.18 subclasses without changing roles."""
    import yaml

    aliases = {'RMSNorm': 'AscendRMSNorm', 'RotaryEmbedding': 'AscendRotaryEmbedding',
               'QKVParallelLinear': 'AscendQKVParallelLinear',
               'RowParallelLinear': 'AscendRowParallelLinear',
               'MergedColumnParallelLinear': 'AscendMergedColumnParallelLinear',
               'SiluAndMul': 'AscendSiluAndMul', 'LogitsProcessor': 'AscendLogitsProcessor',
               'VocabParallelEmbedding': 'AscendVocabParallelEmbedding', 'Sampler': 'AscendSampler'}
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('qwen3', 'llama'):
        document = yaml.safe_load((source / f'{name}.yaml').read_text())
        for group in document['catalog'].values():
            for entry in group.values():
                original = entry['vllm']
                names = [original] if isinstance(original, str) else list(original)
                expanded = names + [aliases[n] for n in names if n in aliases]
                entry['vllm'] = expanded if expanded != names else original
        (destination / f'{name}.yaml').write_text(yaml.safe_dump(document,sort_keys=False))


def write_nvidia_hardware_metadata(hardware: str, device_uuid: str, output: Path) -> None:
    """Read one physical GPU's NVML specification without running any kernels.

    Torch 2.8 lacks the clock/bus properties used by native probe_spec().
    NVML's maximum memory clock and bus width give the same DDR peak-byte
    formula; no workload duration or interconnect measurement is involved.
    """
    import re
    import pynvml
    import yaml

    expected = {"nvidia_a800": "A800", "nvidia_h200": "H200"}[hardware]
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(device_uuid)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if uuid != device_uuid or re.search(rf"\b{expected}\b", name) is None:
            raise ValueError("NVML device identity differs from the reserved hardware")
        observed = {"uuid": uuid, "gpu_name": name,
                    "physical_device": pynvml.nvmlDeviceGetIndex(handle),
                    "memory_total_bytes": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                    "memory_clock_mhz": pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM),
                    "memory_bus_width_bit": pynvml.nvmlDeviceGetMemoryBusWidth(handle),
                    "driver_version": pynvml.nvmlSystemGetDriverVersion(),
                    "nvml_version": pynvml.nvmlSystemGetNVMLVersion()}
    finally:
        pynvml.nvmlShutdown()
    for field in ("memory_total_bytes", "memory_clock_mhz", "memory_bus_width_bit"):
        if type(observed[field]) is not int or observed[field] <= 0:
            raise ValueError(f"invalid NVML specification field: {field}")
    (output / "nvml-specification.json").write_text(json.dumps(observed, indent=2) + "\n")
    spec = {"gpu_name": name, "memory_total_gib": observed["memory_total_bytes"] / 1024**3,
            "memory_clock_mhz": observed["memory_clock_mhz"],
            "memory_bus_width_bit": observed["memory_bus_width_bit"],
            "memory_bw_gbps": observed["memory_clock_mhz"] * 1e6 * observed["memory_bus_width_bit"] / 8 * 2 / 1e9}
    document = {"hardware": hardware, "written_by": "NVML specification query; no device benchmark",
                "source": {"path": str((output / "nvml-specification.json").resolve()),
                           "bandwidth_derivation": "max memory clock MHz * 1e6 * bus bits / 8 * 2 / 1e9; same DDR formula as native probe_spec"},
                "spec": spec, "measured": {"interconnect": None,
                    "interconnect_unavailable": "not measured; only one GPU assigned; TP1 target uses no interconnect"},
                "defaults": {}}
    path = output / "perf" / hardware / "hardware.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False))


def write_ascend_hardware_metadata(config: Path, output: Path) -> None:
    """Translate bound INI-derived specification fields; never claim a link measurement."""
    import hashlib
    import yaml

    document = yaml.safe_load(config.read_text())
    if document['hardware_id'] != 'ascend_910b':
        raise ValueError('Ascend metadata requires its matching hardware specification')
    dram, = [c for c in document['components'] if c['name'] == 'dram']
    if dram['role'] != 'dram':
        raise ValueError('unexpected hardware memory component role')
    bytes_per_s = dram['count'] * dram['throughput_per_clk'] * document['clock_domains'][dram['clock_domain']]
    hardware = {'hardware':document['hardware_id'], 'written_by':'bound specification conversion; no device benchmark',
                'source':{'path':str(config.resolve()),'sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
                          'fields':'components[role=dram] and its clock_domains entry; vendor CANN platform INI'},
                'spec':{'gpu_name':'Ascend 910B2','memory_total_gib':dram['size_bytes']/1024**3,
                        'memory_bw_gbps':bytes_per_s/1e9},
                'measured':{'interconnect':None,'interconnect_unavailable':'not measured; TP1 target uses no interconnect'},
                'defaults':{}}
    path = output / 'perf/ascend_910b/hardware.yaml'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(hardware,sort_keys=False))
