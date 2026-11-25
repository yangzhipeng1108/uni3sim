"""Compare one completed native startup with production capacity owners on CPU."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_native_record(record):
    for name in ('pid','counter_epoch'):
        value=record[name]
        _require(type(value)is int and value>=0,'Native PID/epoch must be nonnegative integers')
    for value in record['native_bytes'].values():
        _require(type(value)is int and value>=0,'Native memory byte fields must be nonnegative integers')
    values=record['native_bytes']
    _require(values['allocated_bytes.all.current']<=values['allocated_bytes.all.peak']<=values['reserved_bytes.all.peak']<=values['total_bytes'],'Native allocator bounds are inconsistent')
    _require(values['reserved_bytes.all.current']<=values['reserved_bytes.all.peak'] and values['free_bytes']<=values['total_bytes'],'Native reserved/free bytes are inconsistent')


def _native_budget_floor(number):
    value=number['value']
    _require(type(value) in (int,float) and math.isfinite(value) and value>=0,
        'Native budget must be a finite nonnegative number')
    _require(number['float_hex']==(value.hex() if type(value) is float else None),
        'Native budget hexadecimal identity is inconsistent')
    return int(value)


def _physical_startup_usage(warm, requested, visible, physical, *, profile_budget_only):
    values=warm['native_bytes']
    used=values['total_bytes']-values['free_bytes']
    _require(values['total_bytes']==visible and 0<=used<=visible<=physical,
        'Native startup usage must fit runtime-visible physical memory')
    _require(values['allocated_bytes.all.peak']<=values['reserved_bytes.all.peak']<=visible,
        'Native allocator peak must fit runtime-visible physical memory')
    _require(0<requested<visible,'Native profile budget must fit runtime-visible physical memory')
    if not profile_budget_only:
        _require(used<requested,'Ascend warmup usage must remain below requested memory')
    return used


def _source_profile_components(source, *, cuda):
    if not cuda:
        return (source['model_load_allocation_diagnostic']['source_model_constructor_buffer_bytes'],
            source['source_comparison_metric']['bytes'])
    prediction=source['prediction']
    _require(prediction['source_program']=='cuda_eager_inplace_fa_no_metadata_flashinfer_0_4_1_top_k_first',
        'A800 comparison requires the declared CUDA source profile')
    _require(prediction['actual_provider_bound'] is False and prediction['complete_worker_memory_accepted'] is False,
        'Source profile must preserve its provider and complete-worker limits')
    constructor=0
    for row in prediction['persistent_source_objects']:
        name=row['allocation_id']
        _require(type(row['bytes']) is int and row['bytes']>=0,'Source object bytes must be nonnegative integers')
        if name=='model.rope-cache':
            constructor+=row['bytes']
        else:
            _require(name.startswith('profile.'),'Unexpected source model constructor buffer role')
    _require(constructor>0 and sum(row['allocation_id']=='model.rope-cache' for row in prediction['persistent_source_objects'])==1,
        'Source profile requires one declared model RoPE cache')
    measurement=prediction['profile_storage_measurements']
    _require(measurement['retained_source_growth_bytes']==0,
        'A800 source comparison requires zero retained profile growth')
    _require(type(measurement['transient_source_headroom_bytes']) is int and measurement['transient_source_headroom_bytes']>0,
        'Source transient profile headroom must be positive integer bytes')
    return constructor,measurement['transient_source_headroom_bytes']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('model','hardware-config','runtime-policy','startup-raw','source-report','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args(argv)
    out=args.output.resolve().parent
    inputs={p.resolve()for p in (args.model,args.hardware_config,args.runtime_policy,args.source_report)}
    targets={args.output.resolve(),out/'observed_activation_reference_policy.json',out/'analytical_activation_policy.json'}
    raw=args.startup_raw.resolve()
    if inputs & targets or len(targets)!=3 or any(p==raw or raw in p.parents for p in targets):
        raise ValueError('Generated reports and policies must preserve input files')
    out.mkdir(parents=True,exist_ok=True)
    from src.kernel.hardware_spec import HWSpec
    from src.serving.kv_cache.kv_capacity import (CapacityEvidence,RuntimeMemoryReserve,load_runtime_memory_policy,
        project_native_model_loading_policy,resolve_analytical_kv_capacity_plan,runtime_memory_policy_payload)
    from src.serving.kv_cache.model_residency import model_rank_residencies
    from src.serving.parallelism.parallel_config import ParallelConfig
    from src.serving.workload.model_config import load_model_config
    raw=args.startup_raw.resolve()
    receipt=json.loads((raw/'collection_receipt.json').read_text())
    _require(receipt['collected_required_phases'] is True and receipt['missing_phases'] == [], "Native startup contract mismatch: receipt['collected_required_phases'] is True and receipt['missing_phases'] == []")
    manifests=json.loads((raw/'executed_bindings.json').read_text())
    _require(bool(manifests['modules']),'Executed module bindings cannot be empty')
    version=manifests['software_versions']['vllm'].split('+',1)[0]
    cuda=version=='0.11.0'
    _require(cuda or (version=='0.18.0' and manifests['software_versions']['vllm_ascend']=='0.18.0'),
        'Native startup comparison requires bound CUDA0.11 or Ascend0.18 source family')
    eligible=[]
    required={'worker_init.end','model_load.begin','model_load.end','profile.begin','profile.end','available_KV.end','KV_allocate.end','all_capture.begin','all_capture.end','startup_warmup.end'}
    for path in raw.glob('memory.rank-process-*.jsonl'):
        rows=[json.loads(s)for s in path.read_text().splitlines()]
        native=[r for r in rows if r['record_type']=='native_startup_memory_bytes']
        for row in rows:
            if row['record_type']=='installed_source_binding':
                _require(row['module_ast_sha256']==manifests['modules'][row['module']]['module_ast_sha256'],'Observed module AST disagrees with executed bindings')
                if cuda:
                    for field in ('file_sha256','source_file'):
                        _require(row[field]==manifests['modules'][row['module']][field],
                            'Observed installed source disagrees with executed bindings')
        for record in native:
            _require(record['source_bindings']==manifests['identity'],'Native record binding identity is inconsistent')
        phases={r['phase']:r for r in native}
        for phase in required & set(phases):
            _require(sum(r['phase']==phase for r in native)==1,'Required phase must occur exactly once per worker')
        modules={r['module']for r in rows if r['record_type']=='installed_source_binding'}
        if required<=set(phases) and set(manifests['modules'])<=modules:eligible.append((path,phases))
    _require(len(eligible)==1,'Exactly one complete worker PID is required')
    path,phases=eligible[0]
    for record in phases.values():
        _check_native_record(record)
        _require(record['software_versions']==manifests['software_versions'],'Native record software versions disagree with executed bindings')
    _require(len(eligible) == 1 and len({p['pid'] for p in phases.values()}) == 1, "Native startup contract mismatch: len(eligible) == 1 and len({p['pid'] for p in phases.values()}) == 1")
    _require(len({phases[k]['counter_epoch'] for k in ('profile.begin', 'profile.end', 'KV_allocate.end', 'all_capture.end', 'startup_warmup.end')}) == 1, "Native startup contract mismatch: len({phases[k]['counter_epoch'] for k in ('profile.begin', 'profile.end', 'KV_allocate.end', 'all_capture.end', 'startup_warmup.end')}) == 1")
    identity=json.loads((raw/'capture_identity.json').read_text())
    model_path=args.model.resolve()
    _require(identity['config_sha256'] == hashlib.sha256(model_path.read_bytes()).hexdigest(), "Native startup contract mismatch: identity['config_sha256'] == hashlib.sha256(model_path.read_bytes()).hexdigest()")
    model=load_model_config(model_path);parallel=ParallelConfig()
    residency,=model_rank_residencies(model,parallel,worker_role='colocated',include_mtp=False)
    policy=load_runtime_memory_policy(args.runtime_policy).policy_for_pool('main')
    hardware_path=args.hardware_config.resolve();hardware=HWSpec(str(hardware_path))
    if (hardware.hardware_id!=('nvidia_a800' if cuda else 'ascend_910b') or model.attention_layout!='standard_gqa'
            or model.model_type not in('qwen3','llama') or model.num_experts!=0
            or model.compute_dtype!='bf16' or model.weight_dtype!='bf16'
            or (policy.binding.tensor_parallel_size,policy.binding.pipeline_parallel_size,policy.binding.expert_parallel_size)!=(1,1,1)):
        raise NotImplementedError('Native startup comparison requires the bound hardware BF16 GQA TP1 PP1 EP1')
    physical=hardware.resource_by_role('dram').size_bytes
    profile=phases['profile.end'];initial=phases['worker_init.end'];kv=phases['KV_allocate.end'];warm=phases['startup_warmup.end']
    binding=kv['resolved_runner_kv_binding'] if cuda else kv['actual_runner_KV_binding']
    visible=initial['native_bytes']['total_bytes'];block_size=16 if cuda else 128
    if cuda:
        _require(initial['configured_block_size_tokens']==block_size,'Unexpected native CUDA block size')
        capture=identity['capture_scope']
        _require((capture['hardware'],capture['vllm'],capture['dtype'],capture['tensor_parallel_size'],
            capture['pipeline_parallel_size'],capture['speculative_or_mtp_tokens'],capture['block_size_tokens'])
            ==('nvidia_a800','0.11.0','bfloat16',1,1,0,16),'Native CUDA capture identity mismatch')
        for record in phases.values():
            values=record['native_bytes']
            _require(values['total_bytes']==visible and values['total_bytes']-values['free_bytes']<=visible
                and values['reserved_bytes.all.peak']<=visible,'CUDA counters exceed runtime-visible physical memory')
            _require(record['config_sha256']==identity['config_sha256'] and record['capture_scope']==capture,
                'Native CUDA record identity mismatch')
    else:
        _require((kv['resolved_cache_block_size_tokens'],kv['runner_block_size_tokens'])==(128,128),
            'Native Ascend runner block size mismatch')
    _require(len(binding['groups'])==1 and binding['groups'][0]['block_size_tokens']==block_size,
        'Native KV group block size mismatch')
    group=binding['groups'][0]
    _require(len(group['layer_names']) == model.num_hidden_layers, "Native startup contract mismatch: len(group['layer_names']) == model.num_hidden_layers")
    exact_load=profile['weights_memory_bytes']
    observed_activation=profile['torch_peak_increase_bytes'] if cuda else profile['torch_peak_increase_bytes']['value']
    nontorch=profile['non_torch_increase_bytes'] if cuda else profile['non_torch_increase_bytes']['value']
    _require(exact_load == phases['model_load.end']['native_bytes']['allocated_bytes.all.current'] - phases['model_load.begin']['native_bytes']['allocated_bytes.all.current'], "Native startup contract mismatch: exact_load == phases['model_load.end']['native_bytes']['allocated_bytes.all.current'] - phases['model_load.begin']['native_bytes']['allocated_bytes.all.current']")
    for value in (exact_load,observed_activation,nontorch):
        _require(type(value)is int,'Profile/load counters require exact integer bytes')
    _require(observed_activation>0,'Observed activation increase must be positive for this comparison')
    requested=initial['requested_memory']
    requested_floor=_native_budget_floor(requested)
    available_return=phases['available_KV.end']['available_kv_return_bytes' if cuda else 'available_KV_return_bytes']
    observed_nonkv=profile['non_kv_cache_memory_bytes'] if cuda else profile['non_KV_cache_memory_bytes']['value']
    if cuda:
        _require(_native_budget_floor(phases['available_KV.end']['available_kv_cache_memory'])
            ==available_return,'Native available KV floor mismatch')
    _require(observed_nonkv == exact_load + observed_activation + nontorch, 'Native startup contract mismatch: observed_nonkv == exact_load + observed_activation + nontorch')
    evidence=CapacityEvidence('runtime_configuration',str(path)+': same PID native counters',hashlib.sha256(path.read_bytes()).hexdigest())
    source=json.loads(args.source_report.read_text())
    if (Path(source['model']).resolve()!=model_path or source['model_execution_version']!=('vllm_0_11_0' if cuda else 'vllm_0_18_0')
            or source['tokens']!=policy.binding.max_batched_tokens or source['max_requests']!=policy.binding.max_requests):
        raise ValueError('Analytical source report disagrees with bound model/runtime capacities')
    if (initial['max_num_batched_tokens'],initial['max_num_seqs'])!=(policy.binding.max_batched_tokens,policy.binding.max_requests):
        raise ValueError('Native startup dimensions disagree with policy binding')
    constructor,source_activation=_source_profile_components(source,cuda=cuda)
    source_evidence=CapacityEvidence('framework_source',
        str(args.source_report.resolve())+(': prediction.profile_storage_measurements.transient_source_headroom_bytes' if cuda else ': analytical source_comparison_metric')+'; lifetime_policy='
        +str(source['source_lifetime_policy'])+'; source_program='+source['prediction']['source_program']+'; declared_compilation_options='
        +json.dumps(source['source_compilation_options'],sort_keys=True),
        hashlib.sha256(args.source_report.read_bytes()).hexdigest())
    variants={}
    for name,activation in (('observed_activation_reference',observed_activation),('analytical_activation',source_activation)):
        activation_evidence=evidence if name=='observed_activation_reference'else source_evidence
        reserves=tuple(RuntimeMemoryReserve(r.kind,nontorch if r.kind=='framework'else activation,
            evidence if r.kind=='framework'else activation_evidence)
            if r.kind in('framework','activation')else r for r in policy.non_kv_reserves)
        base=replace(policy,binding=replace(policy.binding,kv_block_size_tokens=block_size),non_kv_reserves=reserves,
            runtime_visible_hbm_bytes=visible,runtime_visible_hbm_provenance=evidence)
        candidate,diagnostic=project_native_model_loading_policy(base,pure_parameter_bytes=residency.weight_resident_bytes,
            known_constructor_buffer_bytes=constructor,observed_model_loading_allocated_increase_bytes=exact_load,
            observed_non_torch_increase_bytes=nontorch,framework_coverage='non_torch_only',model_loading_provenance=evidence,non_torch_provenance=evidence)
        policy_path=out/(name+'_policy.json')
        policy_path.write_text(json.dumps(runtime_memory_policy_payload((candidate,)),indent=2,allow_nan=False)+'\n')
        candidate=load_runtime_memory_policy(policy_path).policy_for_pool('main')
        plan=resolve_analytical_kv_capacity_plan('main',hardware=hardware,hardware_config_sha256=policy.binding.hardware_config_sha256,
            model=model,parallelism=parallel,deployment='colocated',worker_role='colocated',kv_block_size_tokens=block_size,
            max_model_len=policy.binding.max_model_len,max_batched_tokens=policy.binding.max_batched_tokens,max_requests=policy.binding.max_requests,policy=candidate,scheduler_version='0.11.0')
        budget,=plan.rank_budgets
        _require(budget.physical_hbm_bytes == physical and budget.utilized_hbm_bytes == (requested_floor if cuda else requested['value']), "Native startup contract mismatch: budget.physical_hbm_bytes == physical and budget.utilized_hbm_bytes == (requested_floor if cuda else requested['value'])")
        _require(plan.native_allocation.kv_page_size_bytes_per_layer == group['page_size_bytes'], "Native startup contract mismatch: plan.native_allocation.kv_page_size_bytes_per_layer == group['page_size_bytes']")
        if name=='observed_activation_reference':
            _require(budget.available_kv_bytes == available_return, "Native startup contract mismatch: budget.available_kv_bytes == available_return")
            _require(plan.num_kv_blocks == binding['num_blocks'], "Native startup contract mismatch: plan.num_kv_blocks == binding['num_blocks']")
            _require(plan.native_allocation.worker_allocated_tensor_bytes == (sum((t['size_bytes'] for t in binding['tensors'])),), "Native startup contract mismatch: plan.native_allocation.worker_allocated_tensor_bytes == (sum((t['size_bytes'] for t in binding['tensors'])),)")
        variants[name]={'policy':str(policy_path),'budget':asdict(budget),'blocks':plan.num_kv_blocks,
            'page_bytes_per_layer':plan.native_allocation.kv_page_size_bytes_per_layer,
            'allocated_bytes':list(plan.native_allocation.worker_allocated_tensor_bytes),'diagnostic':diagnostic,
            'allocator_version':plan.native_allocation.version,'scope':('CPU pinned0.11 allocator storage contract compared with actual0.11 CUDA GQA storage; source/provider completeness remains unaccepted' if cuda else 'CPU pinned0.11 allocator storage contract compared with actual0.18 GQA group; not0.18 Scheduler/provider acceptance')}
    activation_error=abs(source_activation/observed_activation-1)
    _require(activation_error <= 0.1, 'Native startup contract mismatch: activation_error <= 0.1')
    native_used=_physical_startup_usage(warm,requested['value'],visible,physical,profile_budget_only=cuda)
    if cuda:
        actual_storage=phases['runner_KV_initialize.end']['actual_storage']
        _require(actual_storage['measurement']=='untyped_storage_metadata_no_device_operation'
            and actual_storage['unique_storage_bytes']==sum(t['size_bytes'] for t in binding['tensors'])
            and actual_storage['unique_storage_count']==len(binding['tensors'])
            and actual_storage['runner_tensor_count']==len(binding['tensors']),
            'Actual CUDA KV storage disagrees with resolved binding')
    repository=Path(__file__).resolve().parents[3]
    result={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip(),
        'repository_status':subprocess.check_output(['git','status','--porcelain'],cwd=repository,text=True).splitlines(),
        'production_root':str(repository),'raw':str(raw),'pid':warm['pid'],'software_versions':warm['software_versions'],
        'all_required_phases_same_process':True,'profile_through_warmup_same_epoch':True,'source_activation_bytes':source_activation,'observed_activation_bytes':observed_activation,
        'source_lifetime_policy':source['source_lifetime_policy'],'source_prediction_scope':source['scope'],
        'source_compilation_options':source['source_compilation_options'],
        'analytical_activation_evidence':asdict(source_evidence),'actual_compiler_plan_bound':False,
        'source_uncovered_categories':source['prediction']['uncovered'],'activation_error_fraction':activation_error,'observed_blocks':binding['num_blocks'],'effective_block_size_tokens':block_size,
        'input_policy_block_size_tokens':policy.binding.kv_block_size_tokens,
        'input_policy_matches_native_KV_geometry':policy.binding.kv_block_size_tokens==block_size,
        'input_policy_capacity_acceptance':False,
        'native_visible_total_bytes':visible,'nominal_physical_hbm_bytes':physical,'warmup_native_used_bytes':native_used,
        'warmup_native_counters':warm['native_bytes'],'variants':variants,
        'normal_startup_observation_and_budget_reconciliation_passed':True,
        'native_profile_budget_and_allocator_reconciliation_accepted':True,
        'source_activation_within_ten_percent_accepted':True,
        'complete_worker_memory_accepted':False,'actual_provider_bound':False,
        'native_requested_memory':requested,'native_requested_memory_floor_bytes':requested_floor,
        'observed_non_kv_profile_bytes':observed_nonkv,
        'input_policy_reserves':runtime_memory_policy_payload((policy,)),
        'input_policy_reserves_promoted':False,
        'post_profile_footprint':{
            'profile_end_native_used_bytes':visible-profile['native_bytes']['free_bytes'],
            'capture_native_used_net_increase_bytes':phases['all_capture.begin']['native_bytes']['free_bytes']-phases['all_capture.end']['native_bytes']['free_bytes'],
            'capture_delta_semantics':'observed net mem_get_info difference; not a full transient peak or named allocation',
            'warmup_requested_budget_overrun_bytes':max(0,native_used-requested['value']),
            'warmup_visible_utilization_fraction':native_used/visible,
            'warmup_physical_headroom_bytes':visible-native_used},
        'native_budget_interpretation':('vLLM0.11 utilization is the profile-time KV budget; capture and warmup may consume remaining visible headroom' if cuda else 'Ascend0.18 observed warmup remains below requested budget'),
        'native_capture_identity':identity,
        'source_program':source['prediction']['source_program'],
        'analytical_blocks_difference_from_native':variants['analytical_activation']['blocks']-binding['num_blocks'],
        'global_normal_capacity_accepted':False,'scope':'One bound real normal startup and budget cross-check. Input-policy KV geometry, other models/hardware and unseen dynamic/library reserves remain separately unaccepted.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('variants','warmup_native_counters')},indent=2))
    return 0

if __name__=='__main__':main()
