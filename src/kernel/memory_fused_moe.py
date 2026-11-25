"""Grouped expert interfaces with explicit routed and formatted layout limits."""

from src.kernel.prediction_cache import runtime_lru_cache
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from math import prod
from src.kernel.kernel_spec import KernelKind,KernelSpec
from src.kernel.model_contract import SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity,
    MemoryAccess, MemoryAccessProgram, MemoryAllocation, Operation, Ordering,
    ProgramEvent, Semantics, Span, TensorView, Typed, Unmapped, region_end, unique_by,
    LogicalSelectionView, TileAxis, TileAccessTemplate, TileDependency, TileProgram,
    LogicalCoordinateBinding,LogicalCoordinateTerm,TileIndexBinding,TileIndexTerm,
)

_SOURCE_WORK = ContextVar('fused_moe_source_work')


def _count_sort_source_algorithm(kernel):
    if kernel.hardware_id == 'nvidia_h200':
        from src.kernel.launch_config.nvidia_h200.moe_aux_launch import COUNT_SORT_VALID_GLOBAL_ALGORITHM
        if kernel.launch.algorithm == COUNT_SORT_VALID_GLOBAL_ALGORITHM:
            routed = kernel.shape['tokens'] * kernel.shape['topk']
            return (kernel.launch.grid == (1, min(-(-routed // 256), 65535), 1)
                and kernel.launch.block == (256, 1, 1) and kernel.launch.tiling is None)
    return kernel.launch.algorithm in ('moe_count_and_sort', 'observed_moe_aux:moe_count_and_sort')


def has_moe_count_sort_source(kernel,hardware):
    from src.kernel.launch_config.select_launch_config import moe_count_sort_source_program
    return (kernel.kind is KernelKind.VECTOR and kernel.variant=='moe_count_and_sort'
        and kernel.execution_support=='supported' and hardware.kernel_memory_model=='gpu_mshr'
        and kernel.hardware_id==hardware.hardware_id
        and _count_sort_source_algorithm(kernel)
        and moe_count_sort_source_program(kernel.hardware_id)=='cuda_moe_count_sort_int32_unmasked')


def standalone_moe_count_sort_bindings(kernel, *, call_id,binding_id, device_id="logical-local-device", rank=0, stream="offline"):
    from src.kernel.memory_access_builder import ElementwiseBindings
    routed=kernel.shape['tokens']*kernel.shape['topk'];experts=kernel.shape['experts']
    allocations,views,events,ready=[],[],[],[]
    for role,shape in (('topk_ids',(routed,)),('cumsum',(experts+1,)),('sorted_token_ids',(routed,))):
        identity=f'{binding_id}.{role}';unknown=role=='sorted_token_ids'
        allocation=MemoryAllocation(identity,device_id,'logical-storage',None if unknown else shape[0]*4,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=(LogicalSelectionView(role,identity,shape,'int32',4,f'{binding_id}.actual_sorted_slots',None)
            if unknown else TensorView(role,identity,0,shape,(4,),'int32',4))
        allocations.append(allocation);views.append(view);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if not unknown:
            event=ProgramEvent(f'{identity}.initialized',EventKind.INITIALIZED,identity,view.region(),'moe-count-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),('topk_ids','cumsum'),
        ('cumsum','sorted_token_ids'),tuple(ready),rank,stream,'moe-count-visible')


def _moe_count_sort_tiles(views,ready,scope,visibility,evidence,*,cumsum_dependencies=None,topk_unmapped=False):
    from src.kernel.memory_access import TileValueIndex,TileAtomicAdd
    routed=views['topk_ids'].shape[0];selected=[]
    for role in ('cumsum','sorted_token_ids'):
        parent=views[role]
        selected.append(LogicalSelectionView(f'{role}.selected',parent.allocation_id,(routed,),'int32',4,
            f'{scope.scope_id}.{role}.from_loaded_expert' if role=='cumsum' else f'{scope.scope_id}.sorted_slots.from_atomic_previous',role))
    index=TileValueIndex('expert','loaded_scalar');rank=TileValueIndex('increment','atomic_previous')
    dep=lambda name,kind:TileDependency(name,kind,())
    accesses=(
        TileAccessTemplate('expert','topk_ids',Operation.READ,('route',),('route',),(ready['topk_ids'],),(),evidence,unmapped=topk_unmapped),
        TileAccessTemplate('counter_read','cumsum.selected',Operation.READ,('route',),('route',),
            (ready['cumsum'],) if cumsum_dependencies is None else (),
            (dep('expert',EventKind.READ_READY),*(cumsum_dependencies if cumsum_dependencies is not None else ())),
            evidence,unmapped=True,value_index=index),
        TileAccessTemplate('increment','cumsum.selected',Operation.WRITE,('route',),('route',),(),
            (dep('expert',EventKind.READ_READY),dep('counter_read',EventKind.READ_READY)),evidence,unmapped=True,
            value_index=index,atomic_add=TileAtomicAdd('counter_read',1)),
        TileAccessTemplate('scatter','sorted_token_ids.selected',Operation.WRITE,('route',),('route',),(),
            (dep('increment',EventKind.WRITE_VISIBLE),),evidence,unmapped=True,value_index=rank))
    family=TileProgram(scope.scope_id,scope,(TileAxis('route',routed,1),),accesses,visibility,evidence)
    return tuple(selected),family


def build_moe_count_sort_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_moe_count_sort_source(kernel,hardware):raise ValueError('This launch is not the CUDA routed-token count-and-sort interface')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('Count-and-sort requires actual typed IDs/cumsum/slot bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Count-and-sort exports logical route tiles')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(('topk_ids','cumsum'),('cumsum','sorted_token_ids'),2):
        raise ValueError('Count-and-sort roles/readiness differ')
    views=unique_by(bindings.views,'view_id');routed=kernel.shape['tokens']*kernel.shape['topk'];experts=kernel.shape['experts']
    if set(views)!={'topk_ids','cumsum','sorted_token_ids'} or views['topk_ids'].shape!=(routed,) or views['cumsum'].shape!=(experts+1,):
        raise ValueError('Count-and-sort input extents differ from actual kernel')
    scope=AccessScope(program_id,('cuda_moe_count_sort',str(kernel.shape['tokens']),str(kernel.shape['topk']),str(experts)),
        bindings.call_id,(),bindings.rank,bindings.stream,'logical-moe-count-sort')
    evidence=('vLLM CUDA _count_and_sort_expert_tokens: load expert ID, atomicAdd(cumsum[expert],1), scatter route ordinal at returned previous counter.',
        'The public unmasked/no-expert-map TopK interface supplies valid INT32 global expert IDs and precomputed padded offsets.',
        'Atomic updates to equal runtime keys are linearizable; no actor order, concrete rank, block padding capacity or address is fabricated.',
        'MetaX kernelHistogram1D is a different program and is not admitted by this CUDA source.')
    selected,family=_moe_count_sort_tiles(views,dict(zip(bindings.read_views,bindings.input_ready,strict=True)),scope,bindings.visibility_domain,evidence)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Runtime expert keys and atomic returned ranks remain symbolic; a consumer needing concrete values/order must bind them.',
            'Sorted slot selections count actual routed writes, not padded tensor capacity. Expert-map/token-mask/invalid-ID paths require their own source.',
            'The inherited aggregate price includes an expert-extent term; logical requests are separate from this retained projection.')),
        bindings.allocations,(*bindings.views,*selected),(),bindings.events,(),(family,))
    moe_count_sort_source_demands(program,expected_kernel=kernel);return program


def moe_count_sort_source_demands(program, *, expected_kernel=None):
    from src.kernel.memory_path_solver import logical_requested_bytes
    from src.kernel.launch_config.select_launch_config import moe_count_sort_source_program
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:raise ValueError('Count-and-sort requires its unique route source')
    family,=program.tile_programs;header=family.scope.op_ids
    if len(header)!=4 or header[0]!='cuda_moe_count_sort' or family.scope.agent!='logical-moe-count-sort':raise ValueError('Count-and-sort source identity differs')
    tokens,topk,experts=map(int,header[1:]);routed=tokens*topk
    if min(tokens,topk,experts)<=0:raise ValueError('Count-and-sort requires positive routed/expert domains')
    if expected_kernel is not None and (expected_kernel.kind is not KernelKind.VECTOR or expected_kernel.variant!='moe_count_and_sort'
            or not _count_sort_source_algorithm(expected_kernel)
            or moe_count_sort_source_program(expected_kernel.hardware_id)!='cuda_moe_count_sort_int32_unmasked'
            or expected_kernel.hardware_id!=program.hardware_id or (tokens,topk,experts)!=tuple(expected_kernel.shape[name] for name in ('tokens','topk','experts'))):
        raise ValueError('Count-and-sort source differs from its actual kernel')
    views=unique_by(program.views,'view_id');parents={role:views[role] for role in ('topk_ids','cumsum','sorted_token_ids')}
    if len({view.allocation_id for view in parents.values()})!=3:raise ValueError('Count-and-sort input/counter/output objects must be distinct')
    for role,shape in (('topk_ids',(routed,)),('cumsum',(experts+1,)),('sorted_token_ids',(routed,))):
        view=parents[role]
        if (view.shape,view.dtype,view.element_bytes)!=(shape,'int32',4):raise ValueError('Count-and-sort requires its exact INT32 operand ABI')
        if role!='sorted_token_ids' and (not isinstance(view,TensorView) or view.strides_bytes!=(4,)):
            raise ValueError('Count-and-sort ID/counter input layout differs')
    templates=unique_by(family.accesses,'access_id');ready={}
    for role,name in (('topk_ids','expert'),('cumsum','counter_read')):
        if len(templates[name].input_ready)!=1:raise ValueError('Count-and-sort actual input readiness differs')
        ready[role]=templates[name].input_ready[0]
    selected,expected=_moe_count_sort_tiles(parents,ready,family.scope,family.visibility_domain,family.evidence_refs)
    if family!=expected or set(views)!={*parents,*(view.view_id for view in selected)} or any(views[view.view_id]!=view for view in selected):
        raise ValueError('Count-and-sort lost actual expert/counter atomic/returned-rank scatter dataflow')
    read=logical_requested_bytes(program,Operation.READ);write=logical_requested_bytes(program,Operation.WRITE)
    return {'tokens':tokens,'topk':topk,'experts':experts,'routed_items':routed,'read_bytes':read,'write_bytes':write,
        'counter_rmw_read_bytes':routed*4,'counter_rmw_write_bytes':routed*4,'actual_sorted_slot_write_bytes':routed*4,
        'legacy_aggregate_read_price_bytes':read+write+experts*2*parents['cumsum'].element_bytes}


@contextmanager
def fused_moe_source_work(kernel, work):
    token = _SOURCE_WORK.set((kernel,work))
    try:
        yield
    finally:
        _SOURCE_WORK.reset(token)


def _bound_work(kernel):
    context = copy_context()
    if _SOURCE_WORK not in context:
        return None
    current,work = context[_SOURCE_WORK]
    if current is not kernel:
        raise ValueError('grouped MoE source work crossed a kernel boundary')
    return work


@dataclass(frozen=True)
class FusedMoEBindings(Typed):
    binding_id: str
    call_id: str
    allocations: tuple[MemoryAllocation, ...]
    views: tuple[TensorView, ...]
    events: tuple[ProgramEvent, ...]
    query_view: str
    weight_view: str
    output_view: str
    routing: str
    query_ready: EventRef
    weight_ready: EventRef
    routing_ready: EventRef
    rank: int
    stream: str
    visibility_domain: str
    formatted_weight: str | None = None
    formatted_ready: tuple[EventRef, ...] = ()


def has_fused_moe_source(kernel, hardware):
    return (kernel.kind is KernelKind.FUSED_MOE and kernel.variant in ('w1','w2')
            and kernel.execution_support == 'supported' and hardware.kernel_memory_model in ('gpu_mshr','npu_tile')
            and (kernel.attributes['expert_parallel_size'] == 1 or hardware.kernel_memory_model == 'gpu_mshr')
            and ('routed_weight_dtype' not in kernel.attributes or kernel.attributes['routed_weight_dtype'] == 'bf16'))


def fused_moe_projection(kernel):
    work = _bound_work(kernel)
    if work is not None:
        return work['projection']
    tokens,topk,hidden,intermediate = (kernel.shape[name] for name in ('tokens','topk','hidden','intermediate'))
    if kernel.variant == 'w1':
        return 'w1',tokens,hidden,2*intermediate
    if kernel.variant == 'w2':
        return 'w2',tokens*topk,intermediate,hidden
    raise ValueError('grouped MoE projection requires w1/w2')


def fused_moe_routing(kernel):
    work = _bound_work(kernel)
    if work is not None:
        return work['routing']
    global_counts,local_counts = kernel.attributes['tokens_per_expert'],kernel.attributes['local_tokens_per_expert']
    if len(global_counts) != kernel.shape['experts'] or len(local_counts) != kernel.shape['experts']:
        raise ValueError('grouped MoE requires one global/local count per expert')
    return tuple(global_counts),tuple(local_counts)


def _layouts(kernel, hardware):
    _,rows,reduction,columns = fused_moe_projection(kernel)
    if hardware.kernel_memory_model == 'npu_tile':
        _,counts = fused_moe_routing(kernel)
        return (('query',(sum(counts),reduction)),('weight',(kernel.shape['experts'],reduction,columns)),
                ('output',(sum(counts),columns)))
    return (('query',(rows,reduction)),('weight',(len(kernel.attributes['owned_experts']),columns,reduction)),
            ('output',(kernel.shape['tokens'],kernel.shape['topk'],columns)))


def _routing_region(kernel, binding_id, *, packed=False):
    _,counts = fused_moe_routing(kernel)
    if packed:
        # Values may be counts or cumulative offsets in the native recipes;
        # the current KernelSpec does not bind that group-list mode.
        return Unmapped((len(counts),),'int64',8,f'{binding_id}.group_list.mode_unbound.{counts}')
    return Unmapped((sum(counts),),'int32',4,
        f'{binding_id}.valid_routed_rows.{counts}.owned{kernel.attributes["owned_experts"]}'
        f'.ep{kernel.attributes["expert_parallel_size"]}.expert_rank{kernel.attributes["expert_rank"]}')


def native_grouped_router_weight_traffic(kernel):
    work=_bound_work(kernel)
    if work is None or 'router_weight_ingress_bytes' not in work:return None
    return work['router_weight_ingress_bytes'],work['router_weight_compulsory_bytes']


def native_grouped_moe_interface(kernel):
    algorithms=('triton_grouped_gemm:w1','triton_grouped_gemm:w2','triton_fused_moe_w1','triton_fused_moe_w2')
    return (kernel.kind is KernelKind.FUSED_MOE and kernel.variant in ('w1','w2')
        and kernel.execution_support=='supported' and kernel.dtype=='bf16'
        and kernel.hardware_id in ('metax_c500','nvidia_a800')
        and ('routed_weight_dtype' not in kernel.attributes or kernel.attributes['routed_weight_dtype']=='bf16')
        and kernel.launch.algorithm in algorithms
        and (kernel.hardware_id=='nvidia_a800' or SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER in kernel.launch.parameters))


def has_native_grouped_moe_controls(kernel,hardware):
    return has_fused_moe_source(kernel,hardware) and hardware.kernel_memory_model=='gpu_mshr' and native_grouped_moe_interface(kernel)


def has_native_grouped_moe_analytical_work(kernel,hardware):
    if has_native_grouped_moe_controls(kernel,hardware):return True
    from src.kernel.memory_fused_moe_naive import has_h200_bf16_controls,h200_bf16_naive
    # Existing H200 aligned sources use this same callee and demand. Keep
    # source/caller ABI admission unchanged; token-wise naive work is separate.
    return (hardware.kernel_memory_model=='gpu_mshr' and has_h200_bf16_controls(kernel)
        and not h200_bf16_naive(kernel))


def _native_moe_numeric_geometry(kernel):
    _,_,reduction,columns=fused_moe_projection(kernel)
    counts,local=fused_moe_routing(kernel)
    owned=tuple(kernel.attributes['owned_experts'])
    owned_set=frozenset(owned)
    if local!=tuple(count if expert in owned_set else 0 for expert,count in enumerate(counts)):
        raise ValueError('Native aligned control source requires exact global/local EP placement')
    bm,bn,bk=(kernel.launch.tiling.tile_shape[name] for name in ('m','n','k'))
    tokens,experts,topk=(kernel.shape[name] for name in ('tokens','experts','topk'))
    # Only the prefix consumed by active blocks is known from routing.
    # Wrapper allocation bounds and naive/align selection are separate facts.
    capacity=sum(-(-count//bm) for count in counts)*bm
    ntiles=-(-columns//bn);grid_m,remainder=divmod(kernel.launch.grid_size,ntiles)
    if SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER in kernel.launch.parameters:
        marker=kernel.launch.parameters[SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER]
    elif kernel.hardware_id in ('nvidia_a800','nvidia_h200'):marker=int(kernel.variant=='w2')
    else:raise ValueError('Native grouped source is missing its explicit W2 program fact')
    if type(marker) not in (int,float) or marker not in (0,1) or bool(marker)!=(kernel.variant=='w2') or remainder:
        raise ValueError('Native grouped source requires actual W2 program and divisible launch grid')
    geometry=(tokens,experts,topk,reduction,columns,bm,bn,bk,grid_m,capacity,counts)
    if kernel.attributes['expert_parallel_size']==1 and owned!=tuple(range(experts)):
        raise ValueError('Native single-rank source requires its complete expert domain')
    _validate_native_geometry_values(geometry,owned,kernel.variant,int(marker))
    return geometry,owned,kernel.variant,int(marker)


def _native_moe_header(kernel):
    geometry,owned,variant,marker=_native_moe_numeric_geometry(kernel)
    *dimensions,counts=geometry
    header=('native_grouped_moe',variant,*(str(value) for value in dimensions),','.join(map(str,counts)),str(marker))
    return header if kernel.attributes['expert_parallel_size']==1 else (*header,','.join(map(str,owned)))


def _native_moe_owned(header):
    owned=tuple(range(int(header[3]))) if len(header)==14 else tuple(map(int,header[14].split(',')))
    if not owned or tuple(sorted(set(owned)))!=owned or min(owned)<0 or max(owned)>=int(header[3]):
        raise ValueError('Native grouped source requires valid compact EP placement')
    return owned


def _native_moe_geometry(header):
    if len(header) not in (14,15) or header[:1]!=('native_grouped_moe',) or header[1] not in ('w1','w2'):
        raise ValueError('Native grouped source requires exact geometry/header')
    t,e,topk,k,n,bm,bn,bk,grid_m,capacity=map(int,header[2:12]);counts=tuple(map(int,header[12].split(',')))
    geometry=(t,e,topk,k,n,bm,bn,bk,grid_m,capacity,counts)
    if header[13]!=('1' if header[1]=='w2' else '0'):
        raise ValueError('Native grouped W2 program marker differs')
    _validate_native_geometry_values(geometry,_native_moe_owned(header),header[1],int(header[13]))
    return geometry


def _validate_native_geometry_values(geometry,owned,variant,marker):
    t,e,topk,k,n,bm,bn,bk,grid_m,capacity,counts=geometry
    expected=sum(-(-count//bm) for count in counts)*bm
    if (min(t,e,topk,k,n,bm,bn,bk,grid_m,capacity)<=0 or len(counts)!=e or min(counts)<0
            or sum(counts)!=t*topk or capacity!=expected or marker!=int(variant=='w2')
            or sum(-(-count//bm) for count in counts)>grid_m):
        raise ValueError('Native grouped logical prefix/routing/launch mismatch')
    if not owned or tuple(sorted(set(owned)))!=owned or min(owned)<0 or max(owned)>=e:
        raise ValueError('Native grouped source requires valid compact EP placement')


def _native_moe_source_geometry(source):
    if isinstance(source,KernelSpec):return _native_moe_numeric_geometry(source)
    return _native_moe_geometry(source),_native_moe_owned(source),source[1],int(source[13])


def _native_moe_control_records(geometry,marker):
    t,e,topk,k,n,bm,bn,bk,grid_m,capacity,counts=geometry
    return (('sorted_token_ids',(capacity,),'int32',4),('expert_ids',(-(-capacity//bm),),'int32',4),
        ('num_tokens_post_padded',(1,),'int32',4),*((('router_weight',(t,topk),'fp32',4),) if marker else ()))


def _native_moe_control_layouts(source):
    geometry,owned,variant,marker=_native_moe_source_geometry(source)
    return _native_moe_control_records(geometry,marker)


def _native_moe_operand_layouts(source):
    geometry,owned,variant,marker=_native_moe_source_geometry(source)
    t,e,topk,k,n,bm,bn,bk,grid_m,capacity,counts=geometry
    return (('query',((t if variant=='w1' else t*topk),k),'bf16',2),
        ('weight',(len(owned),n,k),'bf16',2),('output',(t,topk,n),'bf16',2),*_native_moe_control_records(geometry,marker))


def _native_moe_parent_views(header,views):
    layouts=_native_moe_operand_layouts(header)
    parents={role:views[role] for role,*_ in layouts}
    if len({v.allocation_id for v in parents.values()})!=len(parents):raise ValueError('Native grouped control parents must remain distinct')
    for role,shape,dtype,width in layouts:
        v=parents[role];strides=[];size=width
        for extent in reversed(shape):strides.append(size);size*=extent
        if (v.shape,v.dtype,v.element_bytes,v.offset_bytes,v.strides_bytes)!=(shape,dtype,width,0,tuple(reversed(strides))):raise ValueError('Native grouped typed parent layout differs')
    return parents


def _native_moe_readiness(bindings,views):
    ready={'query':bindings.query_ready,'weight':bindings.weight_ready,'sorted_token_ids':bindings.routing_ready}
    for role in ('expert_ids','num_tokens_post_padded',*(('router_weight',) if 'router_weight' in views else ())):
        parent=views[role]
        events=[event for event in bindings.events if event.allocation_id==parent.allocation_id
            and event.kind in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE,EventKind.LOGICAL_AVAILABLE)
            and (event.region==parent.region() or event.kind is EventKind.LOGICAL_AVAILABLE
                and isinstance(event.region,Unmapped)
                and (event.region.logical_shape,event.region.dtype,event.region.element_bytes)==(parent.shape,parent.dtype,parent.element_bytes))]
        if len(events)!=1:raise ValueError('Native grouped controls require actual full typed parent availability')
        ready[role]=EventRef(events[0].event_id,events[0].kind)
    return ready


def _native_moe_recipe(header,parents,ready,scope,visibility):
    return _cached_native_moe_recipe(header,tuple(sorted(parents.items())),
        tuple(sorted(ready.items())),scope,visibility)


@runtime_lru_cache(maxsize=512)
def _native_moe_view_template(shape,dtype,width,offset,strides,coordinate_map):
    if strides is None:
        return LogicalSelectionView('view','allocation',shape,dtype,width,
            'selection','parent',coordinate_map)
    return TensorView('view','allocation',offset,shape,strides,dtype,width)


@runtime_lru_cache(maxsize=256)
def _native_moe_access_template(access_id,operation,iteration_axes,view_axes,
        input_ready,dependencies,evidence_refs,*,unmapped=False,constant_value=None):
    return TileAccessTemplate(access_id,access_id,operation,iteration_axes,view_axes,
        input_ready,dependencies,evidence_refs,unmapped=unmapped,constant_value=constant_value)


def _native_moe_access(access_id,view_id,*args,**kwargs):
    from src.kernel.memory_access import _replace_source_identity
    template=_native_moe_access_template(access_id,*args,**kwargs)
    return (template if template.view_id==view_id else
        _replace_source_identity(template,view_id=view_id))


@runtime_lru_cache(maxsize=128)
def _native_moe_axis(name,extent):
    return TileAxis(name,extent,1)


@runtime_lru_cache(maxsize=8)
def _cached_native_moe_recipe(header,parent_items,ready_items,scope,visibility):
    # Complete immutable arguments include every actual caller, operand and
    # readiness reference. Only repeated construction of that exact recipe
    # is shared; a new frame still constructs and validates its own source.
    parents,ready=dict(parent_items),dict(ready_items)
    from dataclasses import replace
    from src.kernel.memory_access import _replace_source_identity
    t,e,topk,k,n,bm,bn,bk,grid_m,capacity,counts=_native_moe_geometry(header)
    owned=_native_moe_owned(header)
    selected=[];families=[];m_offset=0
    evidence=('Pinned common BF16 fused_moe_kernel aligned interface: padded-count early return precedes expert/route loads; selected sorted IDs guard payload rows.',
        'Every valid logical output dot consumes its actual routed input and compact expert weight; no physical token/permutation address is fabricated.',
        'W2 loads FP32 topk_weights_ptr after the K loop once per valid row and output-N program; every store consumes its own weight.',
        'Configured grid includes inactive M programs; those only read padded-count. Native issues, cache and DRAM are separate projections.')
    if len(header)==15:
        evidence+=('A remapped nonlocal expert ID (-1) writes zero to every valid routed output and returns without matrix or router-weight reads; local expert slots follow the declared compact contiguous ownership.',)
    def view(role,label,shape,offset=0,strides=None,opaque=False,parent_role=None,coordinate_map=()):
        parent=parents[role if parent_role is None else parent_role];identity=f'{scope.scope_id}.{label}.{role}'
        # Geometry is immutable and checked once per exact template. Only
        # opaque identities change; the complete actual program below still
        # validates allocation bounds, parent maps and producer causality.
        template=_native_moe_view_template(shape,parent.dtype,parent.element_bytes,
            0 if opaque else parent.offset_bytes+offset,None if opaque else strides,coordinate_map)
        child=(_replace_source_identity(template,view_id=identity,allocation_id=parent.allocation_id,
                selection_id=f'{identity}.selected',parent_view_id=parent.view_id)
            if opaque else _replace_source_identity(template,view_id=identity,allocation_id=parent.allocation_id))
        selected.append(child);return child
    dep=lambda name,kind=EventKind.READ_READY,reduce=(),bind=():TileDependency(name,kind,reduce,bind)
    count_dep=(dep('num_tokens_post_padded'),)
    route_dep=(dep('route',bind=(TileIndexBinding('slot',(TileIndexTerm('row',1),)),)),dep('expert_ids'))
    dot_deps=(dep('input'),dep('weight'))
    router_deps=(dep('input',reduce=('column',)),dep('weight',reduce=('column',)))
    weighted_dot_deps=(*dot_deps,dep('router_weight'))
    nfull,ntail=divmod(n,bn);ngroups=([('nfull',nfull,bn,0)] if nfull else [])+([('ntail',1,ntail,nfull*bn)] if ntail else [])
    for expert,count in enumerate(counts):
        mfull,mtail=divmod(count,bm)
        mgroups=([('mfull',mfull,bm,0)] if mfull else [])+([('mtail',1,mtail,mfull)] if mtail else [])
        for ml,mc,rows,mfirst in mgroups:
            for nl,nc,cols,nfirst in ngroups:
                label=f'e{expert}.{ml}.{nl}';identity=f'{scope.scope_id}.{label}'
                expert_id=view('expert_ids',label,(mc,),opaque=True,coordinate_map=(
                    LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),m_offset+mfirst),))
                route=view('sorted_token_ids',label,(mc,bm),opaque=True,coordinate_map=(
                    LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,bm),LogicalCoordinateTerm(1,1)),bm*(m_offset+mfirst)),))
                if expert in owned:
                    query=view('query',label,(mc,rows,k),opaque=True)
                    weight=view('weight',label,(nc,cols,k),owned.index(expert)*n*k*2+nfirst*k*2,(bn*k*2,k*2,2))
                output=view('output',label,(mc,rows,nc,cols),opaque=True)
                axes=(_native_moe_axis('m',mc),_native_moe_axis('n',nc),_native_moe_axis('slot',bm),_native_moe_axis('row',rows),_native_moe_axis('column',cols))
                actor=('m','n');dot=(*actor,'row','column')
                accesses=[
                    _native_moe_access('num_tokens_post_padded','num_tokens_post_padded',Operation.READ,actor,(None,),(ready['num_tokens_post_padded'],),(),evidence),
                    _native_moe_access('expert_ids',expert_id.view_id,Operation.READ,actor,('m',),(ready['expert_ids'],),count_dep,evidence,unmapped=True),
                    _native_moe_access('route',route.view_id,Operation.READ,(*actor,'slot'),('m','slot'),(ready['sorted_token_ids'],),count_dep,evidence,unmapped=True)]
                if expert not in owned:
                    from src.kernel.memory_access import ConstantValue
                    accesses.append(_native_moe_access('output',output.view_id,Operation.WRITE,dot,('m','row','n','column'),(),
                        route_dep,evidence,unmapped=True,constant_value=ConstantValue.ZERO))
                    local=replace(scope,scope_id=identity,op_ids=(*header,str(expert),ml,nl),agent='native-grouped-moe-remote-zero')
                    families.append(TileProgram(identity,local,axes,tuple(accesses),visibility,evidence))
                    continue
                accesses.extend((
                    _native_moe_access('input',query.view_id,Operation.READ,dot,('m','row',None),(ready['query'],),route_dep,evidence,unmapped=True),
                    _native_moe_access('weight',weight.view_id,Operation.READ,dot,('n','column',None),(ready['weight'],),route_dep,evidence)))
                output_deps=dot_deps
                if header[13]=='1':
                    routed_weight=view('router_weight',label,(mc,rows),opaque=True)
                    accesses.append(_native_moe_access('router_weight',routed_weight.view_id,Operation.READ,(*actor,'row'),('m','row'),(ready['router_weight'],),
                        router_deps,evidence,unmapped=True))
                    output_deps=weighted_dot_deps
                accesses.append(_native_moe_access('output',output.view_id,Operation.WRITE,dot,('m','row','n','column'),(),output_deps,evidence,unmapped=True))
                local=replace(scope,scope_id=identity,op_ids=(*header,str(expert),ml,nl),agent='native-grouped-moe-control-dots')
                families.append(TileProgram(identity,local,axes,tuple(accesses),visibility,evidence))
        m_offset+=-(-count//bm)
    inactive=grid_m-m_offset
    if inactive:
        identity=f'{scope.scope_id}.inactive';local=replace(scope,scope_id=identity,op_ids=(*header,'inactive'),agent='native-grouped-moe-early-return')
        families.append(TileProgram(identity,local,(TileAxis('m',inactive,1),TileAxis('n',-(-n//bn),1)),
            (TileAccessTemplate('num_tokens_post_padded','num_tokens_post_padded',Operation.READ,('m','n'),(None,),(ready['num_tokens_post_padded'],),(),evidence),),visibility,evidence))
    return tuple(selected),tuple(families)


def build_native_grouped_moe_source(kernel,hardware,bindings,views, *, program_id):
    header=_native_moe_header(kernel);ready=_native_moe_readiness(bindings,views)
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'native-grouped-moe')
    selected,families=_native_moe_recipe(header,views,ready,scope,bindings.visibility_domain)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Native scalar/expert/sorted-ID inputs and optional W2 FP32 router weights are explicit, including padded IDs and count-only early returns.',
            'Sorted/expert views describe only the required padded logical prefix; standalone allocation bounds and physical addresses remain unknown.',
            'Full-K logical output dots retain unknown routed input/output coordinates; native staged K-panel issue and physical footprints are separate.',
            'Existing compulsory-interface and traffic prices are retained; newly exposed control requests are not extra DRAM prices.')),
        bindings.allocations,(*bindings.views,*selected),(scope,),bindings.events,(),families)
    native_grouped_moe_demands(program)
    return program


def native_grouped_moe_demands(program):
    # All returned values are immutable scalar/tuple work. Give each consumer
    # its own dictionary while sharing validation of the complete program.
    return dict(_native_grouped_moe_demands(program))


@runtime_lru_cache(maxsize=8)
def _native_grouped_moe_demands(program):
    if program.hardware_id=='nvidia_h200' and program.scopes and program.scopes[0].op_ids[-1]=='naive':
        from src.kernel.memory_fused_moe_naive import h200_naive_demands
        return h200_naive_demands(program)
    from src.kernel.memory_access import _tile_template_requested_bytes
    program.__post_init__()
    if not program.tile_programs:raise ValueError('Native grouped source requires actual tiles and producer causality')
    scope,=program.scopes;header=scope.op_ids
    t,e,topk,k,n,bm,bn,bk,grid_m,capacity,counts=_native_moe_geometry(header)
    if program.hardware_id not in ('metax_c500','nvidia_a800','nvidia_h200'):raise ValueError('Native grouped control source hardware ABI differs')
    owned=_native_moe_owned(header);local_counts=tuple(count if expert in owned else 0 for expert,count in enumerate(counts))
    views=unique_by(program.views,'view_id');parents=_native_moe_parent_views(header,views)
    ready={}
    for family in program.tile_programs:
        for access in family.accesses:
            if access.operation is not Operation.READ:continue
            role={'input':'query','route':'sorted_token_ids'}[access.access_id] if access.access_id in ('input','route') else access.access_id
            if len(access.input_ready)!=1 or role in ready and ready[role]!=access.input_ready[0]:raise ValueError('Native grouped input availability differs')
            ready[role]=access.input_ready[0]
    expected_ready={'sorted_token_ids','expert_ids','num_tokens_post_padded'}
    if sum(local_counts):
        expected_ready.update(('query','weight'))
        if header[13]=='1':expected_ready.add('router_weight')
    if set(ready)!=expected_ready:
        raise ValueError('Native grouped source lost actual local input producer causality')
    selected,families=_native_moe_recipe(header,parents,ready,scope,program.tile_programs[0].visibility_domain)
    if program.tile_programs!=families or set(views)!={*parents,*(v.view_id for v in selected)} or any(views[v.view_id]!=v for v in selected):
        raise ValueError('Native grouped source requires exact controls, selected rows, routing/input/weight producer causality and every actual router READ_READY')
    events=unique_by(program.events,'event_id')
    for role,ref in ready.items():
        event=events[ref.node_id]
        parent=parents[role]
        logical=(ref.kind is EventKind.LOGICAL_AVAILABLE and isinstance(event.region,Unmapped)
            and (event.region.logical_shape,event.region.dtype,event.region.element_bytes)==(parent.shape,parent.dtype,parent.element_bytes))
        if (ref.kind not in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE,EventKind.LOGICAL_AVAILABLE)
                or event.allocation_id!=parent.allocation_id or not (logical or event.region==parent.region())):
            raise ValueError('Native grouped controls require actual full parent readiness')
    logical={role:0 for role in ('input','weight','output','route','expert_ids','num_tokens_post_padded','router_weight')}
    for family in program.tile_programs:
        axes=unique_by(family.axes,'axis_id')
        for access in family.accesses:logical[access.access_id]+=_tile_template_requested_bytes(views,axes,family,access)
    work=_native_moe_prediction_demand(header)
    if logical['router_weight']!=work['router_weight_ingress_bytes']:
        raise ValueError('Native grouped router-weight iteration demand differs')
    return {**work,
        'logical_requested_read_bytes':sum(value for role,value in logical.items() if role!='output'),
        'logical_requested_write_bytes':logical['output'],'control_read_bytes':sum(logical[role] for role in ('route','expert_ids','num_tokens_post_padded'))}



def _native_moe_prediction_demand(header):
    """Compulsory routed matrices and per-N-tile W2 weight ingress.

    These are the existing analytical model's complete memory inputs. The
    unpriced count/scan/dot-request detail remains available in graph export.
    """
    geometry,owned,variant,marker=_native_moe_source_geometry(header)
    t,e,topk,k,n,bm,bn,bk,grid_m,capacity,counts=geometry
    owned_set=frozenset(owned)
    local=tuple(count if expert in owned_set else 0 for expert,count in enumerate(counts))
    routes=sum(local);router=routes*4 if marker else 0
    return {'projection':(variant,t if variant=='w1' else t*topk,k,n),'routing':(counts,local),
        'read_bytes':routes*4+routes*k*2+sum(c>0 for c in local)*n*k*2,'write_bytes':routes*n*2,
        'router_weight_ingress_bytes':router*-(-n//bn),'router_weight_compulsory_bytes':router}


def native_grouped_moe_prediction_work(kernel,hardware,bindings):
    """Validate actual operands/metadata and evaluate the routed demand."""
    from src.kernel.memory_access import validate_storage_bindings
    if kernel.hardware_id!=hardware.hardware_id or not has_native_grouped_moe_analytical_work(kernel,hardware):
        raise ValueError('Native grouped analytical work requires its existing source ABI')
    from src.kernel.memory_access_builder import MemoryInvocation
    if isinstance(bindings,MemoryInvocation):
        bindings.assert_boundary_unchanged()
        if kernel!=bindings.kernel:raise ValueError('Grouped invocation crossed its source kernel')
        validate_native_moe_invocation(bindings)
        return _native_moe_prediction_demand(bindings.operand_source)
    views=_validate_gpu_projection_parents(kernel,hardware,bindings)
    bindings.__post_init__()
    header=_native_moe_header(kernel);work=_native_moe_prediction_demand(header)
    parents=_native_moe_parent_views(header,views)
    if set(views)!=set(parents):
        raise ValueError('Native grouped source contains an undeclared operand selection')
    validate_storage_bindings(bindings.allocations,bindings.views,bindings.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual grouped-MoE operand/control boundary; private tile execution is not materialized.',)))
    ready=_native_moe_readiness(bindings,parents);events=unique_by(bindings.events,'event_id')
    consumed={'sorted_token_ids','expert_ids','num_tokens_post_padded'}
    if sum(work['routing'][1]):
        consumed.update(('query','weight'))
        if header[13]=='1':consumed.add('router_weight')
    for role in consumed:
        ref=ready[role];parent=parents[role]
        if ref.node_id not in events:raise ValueError('Native grouped input readiness is missing')
        event=events[ref.node_id]
        logical=(ref.kind is EventKind.LOGICAL_AVAILABLE and isinstance(event.region,Unmapped)
            and (event.region.logical_shape,event.region.dtype,event.region.element_bytes)==(parent.shape,parent.dtype,parent.element_bytes))
        if (event.kind is not ref.kind or ref.allow_skipped_read
                or ref.kind not in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE,EventKind.LOGICAL_AVAILABLE)
                or event.allocation_id!=parent.allocation_id or event.visibility_domain!=bindings.visibility_domain
                or event.view_id is not None and event.view_id!=parent.view_id
                or not (logical or event.region==parent.region())):
            raise ValueError('Native grouped controls require actual full parent readiness')
    return work


def native_moe_operand_records(kernel,*,binding_id,device_id):
    """Declare the existing GPU callee's actual parents, without events."""
    from src.kernel.memory_fused_moe_naive import has_h200_bf16_controls
    h200=has_h200_bf16_controls(kernel)
    layouts=_native_moe_operand_layouts(kernel)
    allocations=[];views=[]
    for role,shape,dtype,width in layouts:
        identity=binding_id+('.routing' if role=='sorted_token_ids' else '.'+role)
        strides=[];size=width
        for extent in reversed(shape):strides.append(size);size*=extent
        domain=('logical-native-sorted-token-ids' if role=='sorted_token_ids' else
            'logical-native-moe-control' if role in ('expert_ids','num_tokens_post_padded','router_weight') else 'logical-storage')
        if h200:
            domain='logical-native-moe-control' if role in ('sorted_token_ids','expert_ids') else 'logical-storage'
        allocations.append(MemoryAllocation(identity,device_id,domain,
            None if role in ('sorted_token_ids','expert_ids') else size,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None))
        views.append(TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width))
    return kernel,tuple(allocations),tuple(views)


def validate_native_moe_invocation(invocation):
    from src.kernel.memory_access import check_type
    from src.kernel.memory_fused_moe_naive import has_h200_bf16_controls,h200_bf16_naive
    source=(native_grouped_moe_interface(invocation.kernel)
        or has_h200_bf16_controls(invocation.kernel) and not h200_bf16_naive(invocation.kernel))
    if not source or invocation.operand_source!=invocation.kernel:
        raise ValueError('Grouped invocation differs from its declared source rule')
    if type(invocation.allocations) is not tuple or type(invocation.views) is not tuple:
        raise TypeError('Grouped source operands must be immutable')
    for parent in invocation.allocations:
        check_type(parent,MemoryAllocation);parent.__post_init__()
    for view in invocation.views:
        check_type(view,TensorView);view.__post_init__()
    views=unique_by(invocation.views,'view_id');parents=unique_by(invocation.allocations,'allocation_id')
    expected=_native_moe_parent_views(invocation.operand_source,views)
    if set(views)!=set(expected) or {view.allocation_id for view in views.values()}!=set(parents):
        raise ValueError('Grouped invocation contains undeclared storage')
    if len({parent.device_id for parent in parents.values()})!=1:
        raise ValueError('Grouped invocation crossed actual devices')
    for view in views.values():
        parent=parents[view.allocation_id]
        # The source layout check above proved a dense, zero-offset parent.
        if parent.freed_event is not None or parent.size_bytes is not None and prod(view.shape)*view.element_bytes>parent.size_bytes:
            raise ValueError('Grouped invocation escapes its actual storage lifetime or capacity')
    names=native_moe_event_names(invocation)
    if len(set(names))!=len(names) or set(names)&(set(parents)|set(views)) or set(parents)&set(views):
        raise ValueError('Grouped invocation has colliding operand/event identities')


def _native_moe_ready_identity(invocation,role):
    from src.kernel.memory_fused_moe_naive import has_h200_bf16_controls
    virtual=invocation.binding_id+('.routing' if role=='sorted_token_ids' else '.'+role)
    suffix=('.available' if has_h200_bf16_controls(invocation.kernel) else
        '.initialized' if role in ('query','weight') else '.native_available')
    identity=virtual+suffix
    if invocation.scope_initialized:identity=invocation.binding_id+'.'+identity
    return identity


def native_moe_input_ready(invocation,role):
    if role not in invocation.read_views:raise ValueError('Unknown grouped input role')
    view=next(view for view in invocation.views if view.view_id==role)
    parent=next(parent for parent in invocation.allocations if parent.allocation_id==view.allocation_id)
    logical=role in ('sorted_token_ids','expert_ids') or role=='query' and parent.size_bytes is None
    return EventRef(_native_moe_ready_identity(invocation,role),EventKind.LOGICAL_AVAILABLE if logical else EventKind.INITIALIZED)


def native_moe_event_names(invocation):
    return tuple(parent.allocated_event for parent in invocation.allocations)+tuple(
        _native_moe_ready_identity(invocation,role) for role in invocation.read_views)


def materialize_native_moe_invocation(invocation):
    """Explicit canonical projection of the same declared actual operands."""
    parents=unique_by(invocation.allocations,'allocation_id');views=unique_by(invocation.views,'view_id')
    events=[]
    for role in ('query','weight','output','sorted_token_ids','expert_ids','num_tokens_post_padded',
            *(('router_weight',) if 'router_weight' in views else ())):
        view=views[role];parent=parents[view.allocation_id]
        events.append(ProgramEvent(parent.allocated_event,EventKind.ALLOCATED,parent.allocation_id,None,None,()))
        if role=='output':continue
        ready=native_moe_input_ready(invocation,role)
        virtual=invocation.binding_id+('.routing' if role=='sorted_token_ids' else '.'+role)
        region=(Unmapped(view.shape,view.dtype,view.element_bytes,virtual+'.required_logical_prefix')
            if role in ('sorted_token_ids','expert_ids') else
            Unmapped(view.shape,view.dtype,view.element_bytes,invocation.binding_id+'.native_routed_query')
            if role=='query' and parent.size_bytes is None else view.region())
        events.append(ProgramEvent(ready.node_id,ready.kind,parent.allocation_id,region,invocation.visibility_domain,
            (EventRef(parent.allocated_event,EventKind.ALLOCATED),)))
    return FusedMoEBindings(invocation.binding_id,invocation.call_id,invocation.allocations,invocation.views,
        tuple(events),'query','weight','output',views['sorted_token_ids'].allocation_id,
        native_moe_input_ready(invocation,'query'),native_moe_input_ready(invocation,'weight'),
        native_moe_input_ready(invocation,'sorted_token_ids'),invocation.rank,invocation.stream,invocation.visibility_domain)


def standalone_fused_moe_bindings(kernel, hardware, *, call_id, binding_id):
    if kernel.hardware_id=='nvidia_h200':
        from src.kernel.memory_fused_moe_naive import has_h200_bf16_controls,standalone_h200_bf16_bindings
        if has_h200_bf16_controls(kernel):
            return standalone_h200_bf16_bindings(kernel,hardware,call_id=call_id,binding_id=binding_id)
    allocations,views,events,ready = [],[],[],[]
    width = kernel.shape['dtype_bytes']
    for role,shape in _layouts(kernel,hardware):
        identity = f'{binding_id}.{role}'
        strides,size = [],width
        for axis in reversed(shape):
            strides.append(size)
            size *= axis
        view = TensorView(role,identity,0,shape,tuple(reversed(strides)),kernel.dtype,width)
        storage = ('logical-nd-descriptor' if role == 'weight' and hardware.kernel_memory_model == 'npu_tile'
                   else 'logical-storage')
        allocation = MemoryAllocation(identity,'logical-local-device',storage,size,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation)
        views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role != 'output':
            event = ProgramEvent(f'{identity}.initialized',EventKind.INITIALIZED,identity,view.region(),'moe-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event)
            ready.append(EventRef(event.event_id,event.kind))
    routing = f'{binding_id}.routing'
    allocations.append(MemoryAllocation(routing,'logical-local-device','logical-routing-storage',None,
        AddressLevel.OBJECT,None,None,f'{routing}.allocated',None))
    events.append(ProgramEvent(f'{routing}.allocated',EventKind.ALLOCATED,routing,None,None,()))
    routing_ready = EventRef(f'{routing}.allocated',EventKind.ALLOCATED)
    _,counts = fused_moe_routing(kernel)
    if sum(counts) or hardware.kernel_memory_model == 'npu_tile':
        available = ProgramEvent(f'{routing}.available',EventKind.LOGICAL_AVAILABLE,routing,
            _routing_region(kernel,binding_id,packed=hardware.kernel_memory_model == 'npu_tile'),'moe-visible',
            (routing_ready,))
        events.append(available)
        routing_ready = EventRef(available.event_id,available.kind)
    if has_native_grouped_moe_controls(kernel,hardware):
        header=_native_moe_header(kernel)
        for role,shape,dtype,element_bytes in _native_moe_control_layouts(header):
            identity=routing if role=='sorted_token_ids' else f'{binding_id}.{role}'
            unknown=role in ('sorted_token_ids','expert_ids')
            strides=[];size=element_bytes
            for extent in reversed(shape):strides.append(size);size*=extent
            if role!='sorted_token_ids':
                allocation=MemoryAllocation(identity,'logical-local-device','logical-native-moe-control',None if unknown else size,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
                allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
            view=TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,element_bytes);views.append(view)
            event_id=f'{identity}.native_available'
            if role=='sorted_token_ids':
                from dataclasses import replace
                allocations=[replace(allocation,memory_domain='logical-native-sorted-token-ids') if allocation.allocation_id==routing else allocation for allocation in allocations]
                if routing_ready.kind is not EventKind.ALLOCATED:
                    events=[event for event in events if event.event_id!=routing_ready.node_id]
                routing_ready=EventRef(event_id,EventKind.LOGICAL_AVAILABLE)
            region=Unmapped(shape,dtype,element_bytes,f'{identity}.required_logical_prefix') if unknown else view.region()
            events.append(ProgramEvent(event_id,EventKind.LOGICAL_AVAILABLE if unknown else EventKind.INITIALIZED,identity,region,'moe-visible',
                (EventRef(f'{identity}.allocated',EventKind.ALLOCATED),)))
    formatted,formatted_ready = None,[]
    if hardware.kernel_memory_model == 'npu_tile':
        formatted = f'{binding_id}.formatted_weight'
        allocation = MemoryAllocation(formatted,'logical-local-device','logical-formatted-weight',None,
            AddressLevel.OBJECT,None,None,f'{formatted}.allocated',None)
        allocations.append(allocation)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,formatted,None,None,()))
        _,counts = fused_moe_routing(kernel)
        for expert,count in enumerate(counts):
            if count:
                event = ProgramEvent(f'{formatted}.{expert}.available',EventKind.LOGICAL_AVAILABLE,formatted,
                    _formatted_region(kernel,binding_id,expert),'moe-visible',
                    (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
                events.append(event)
                formatted_ready.append(EventRef(event.event_id,event.kind))
    return FusedMoEBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),'query','weight','output',
        routing,*ready,routing_ready,0,'offline','moe-visible',formatted,tuple(formatted_ready))


def _validate_gpu_projection_parents(kernel,hardware,bindings):
    if not isinstance(bindings,FusedMoEBindings):
        raise TypeError('grouped MoE requires typed query/weights/routing/output bindings')
    views = unique_by(bindings.views,'view_id')
    query,weight,output = (views[name] for name in (bindings.query_view,bindings.weight_view,bindings.output_view))
    if bindings.formatted_weight is not None or bindings.formatted_ready:
        raise ValueError('GPU grouped projection cannot consume NPU formatted-weight bindings')
    for view,(_,shape) in zip((query,weight,output),_layouts(kernel,hardware),strict=True):
        strides,size = [],kernel.shape['dtype_bytes']
        for axis in reversed(shape):
            strides.append(size)
            size *= axis
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes) != (shape,tuple(reversed(strides)),kernel.dtype,kernel.shape['dtype_bytes']):
            raise ValueError('grouped MoE operand shape/storage mismatch')
    if len({query.allocation_id,weight.allocation_id,output.allocation_id,bindings.routing}) != 4:
        raise NotImplementedError('grouped MoE overlap requires native in-place/lifecycle evidence')
    readiness = unique_by(bindings.events,'event_id')
    # Without the selected coordinates, a resolved query subset cannot
    # authorize an arbitrary opaque routed read. Require the whole parent.
    if bindings.query_ready.kind not in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE):
        raise ValueError('grouped MoE selected input requires complete query parent readiness')
    parent_ready = readiness[bindings.query_ready.node_id]
    covered = (parent_ready.region == query.region() or isinstance(parent_ready.region,Span)
               and parent_ready.region.offset_bytes <= query.offset_bytes
               and region_end(parent_ready.region) >= region_end(query.region()))
    if parent_ready.allocation_id != query.allocation_id or not covered:
        raise ValueError('grouped MoE selected input requires complete query parent readiness')
    return views


def build_fused_moe_source(kernel, hardware, bindings, granularity, *, program_id):
    if not has_fused_moe_source(kernel,hardware):
        raise NotImplementedError('grouped MoE source domain requires supported nonquantized projection')
    if hardware.kernel_memory_model == 'npu_tile':
        return build_packed_fused_moe_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if granularity not in (Granularity.OP, Granularity.KERNEL, Granularity.TILE):
        raise ValueError('grouped MoE exports logical route/dot tiles only')
    views=_validate_gpu_projection_parents(kernel,hardware,bindings)
    if kernel.hardware_id=='nvidia_h200':
        from src.kernel.memory_fused_moe_naive import has_h200_bf16_controls,build_h200_bf16_source
        if has_h200_bf16_controls(kernel):
            return build_h200_bf16_source(kernel,hardware,bindings,views,program_id=program_id)
    if has_native_grouped_moe_controls(kernel,hardware):
        return build_native_grouped_moe_source(kernel,hardware,bindings,views,program_id=program_id)
    _,counts = fused_moe_routing(kernel)
    if sum(counts):
        return build_grouped_moe_tiles(kernel, hardware, bindings, views, program_id=program_id)
    # No valid local matrix work. Preserve global EP placement and caller
    # parents, without fabricating empty tensor accesses or native zero traffic.
    global_counts, _ = fused_moe_routing(kernel)
    descriptor = ('fused_moe_ep', kernel.variant,
        str(kernel.attributes['expert_parallel_size']), str(kernel.attributes['expert_rank']),
        ','.join(map(str,global_counts)), ','.join(map(str,kernel.attributes['owned_experts'])),
        bindings.query_view, bindings.weight_view, bindings.output_view)
    scope = AccessScope(f'{program_id}.ep', descriptor, bindings.call_id, (),
        bindings.rank, bindings.stream, 'logical-rank-local-projection')
    return MemoryAccessProgram('midlayer.memory-access.v1', program_id, hardware.hardware_id, bindings.binding_id,
        Coverage(granularity, Semantics.LOGICAL, Ordering.UNORDERED, False, (
            'No valid local route emits mathematical matrix work; this does not imply zero native metadata or wrapper traffic.',
            'Global placement and actual parent objects are retained; routing padding, scratch and native zero-fill remain partial.')),
        bindings.allocations, bindings.views, (scope,), bindings.events, ())


def build_grouped_moe_tiles(kernel, hardware, bindings, views, *, program_id):
    """Expert-local route ordinals keep unknown token/permutation mappings."""
    query, weight, output = (views[name] for name in (bindings.query_view, bindings.weight_view, bindings.output_view))
    _, _, reduction, columns = fused_moe_projection(kernel)
    global_counts, counts = fused_moe_routing(kernel)
    owned = kernel.attributes['owned_experts']
    root_region = _routing_region(kernel, bindings.binding_id)
    routing = LogicalSelectionView(f'{program_id}.routing', bindings.routing, root_region.logical_shape,
        'int32', 4, root_region.selection_id, None)
    views[routing.view_id] = routing
    families, scopes = [], []
    if kernel.attributes['expert_parallel_size'] != 1:
        scopes.append(AccessScope(f'{program_id}.ep', ('fused_moe_ep', kernel.variant,
            str(kernel.attributes['expert_parallel_size']), str(kernel.attributes['expert_rank']),
            ','.join(map(str,global_counts)), ','.join(map(str,owned))), bindings.call_id, (),
            bindings.rank, bindings.stream, 'logical-rank-local-projection'))
    evidence = ('vLLM grouped experts: each valid expert-local route computes an output dot from its selected query row and compact local expert weight',
        'Expert-local route ordinal is a logical coordinate; routing values, sorted-token positions and physical input/output rows remain unknown',
        'Repeated selections of one query token do not create capacity, distinct storage or an affine parent offset')
    for expert, count in enumerate(counts):
        if not count:
            continue
        identity = f'{program_id}.{expert}'
        selected = []
        for role, parent, shape in (('route', routing, (count,)), ('input', query, (count,reduction)), ('output', output, (count,columns))):
            selection = LogicalSelectionView(f'{identity}.{role}_selection', parent.allocation_id, shape,
                parent.dtype, parent.element_bytes, f'{bindings.binding_id}.{kernel.variant}.expert{expert}.{role}.valid_route_ordinal',
                parent.view_id)
            views[selection.view_id] = selection
            selected.append(selection)
        route, selected_input, selected_output = selected
        weight_slice = TensorView(f'{identity}.weight', weight.allocation_id,
            weight.offset_bytes + owned.index(expert)*columns*reduction*weight.element_bytes,
            (columns,reduction), (reduction*weight.element_bytes,weight.element_bytes), weight.dtype, weight.element_bytes)
        views[weight_slice.view_id] = weight_slice
        route_ready = (TileDependency('route', EventKind.READ_READY, ()),)
        accesses = (
            TileAccessTemplate('route', route.view_id, Operation.READ, ('row',), ('row',),
                (bindings.routing_ready,), (), evidence, unmapped=True),
            TileAccessTemplate('input', selected_input.view_id, Operation.READ, ('row','column'), ('row',None),
                (bindings.query_ready,), route_ready, evidence, unmapped=True),
            TileAccessTemplate('weight', weight_slice.view_id, Operation.READ, ('row','column'), ('column',None),
                (bindings.weight_ready,), route_ready, evidence),
            TileAccessTemplate('output', selected_output.view_id, Operation.WRITE, ('row','column'), ('row','column'), (), (
                TileDependency('input',EventKind.READ_READY,()), TileDependency('weight',EventKind.READ_READY,())), evidence, unmapped=True))
        scope = AccessScope(identity, ('fused_moe_tiles', kernel.variant, str(expert), query.view_id, weight.view_id,
            output.view_id, routing.view_id), bindings.call_id, (), bindings.rank, bindings.stream, 'logical-routed-expert-dots')
        families.append(TileProgram(identity, scope, (TileAxis('row',count,1),TileAxis('column',columns,1)),
            accesses, bindings.visibility_domain, evidence))
    program = MemoryAccessProgram('midlayer.memory-access.v1', program_id, hardware.hardware_id, bindings.binding_id,
        Coverage(Granularity.TILE, Semantics.LOGICAL, Ordering.PARTIAL, False, (
            'Expert-local route ordinals and full-K output dots are compressed; input/output permutation, selected token values and physical footprint remain unresolved.',
            'Valid routing descriptor only; native padded sorted IDs, expert IDs, counters, expert-map metadata, scratch and wrapper lifetimes remain partial.',
            'Logical rereads do not infer DRAM traffic or repeated-token distinct capacity; existing interface/launch/cache projections remain separate.')),
        bindings.allocations, tuple(views.values()), tuple(scopes), bindings.events, (), tuple(families))
    from src.kernel.memory_path_solver import fused_moe_source_demands
    fused_moe_source_demands(program)
    return program

def _formatted_region(kernel, binding_id, expert):
    _,_,reduction,columns = fused_moe_projection(kernel)
    return Unmapped((reduction,columns),kernel.dtype,kernel.shape['dtype_bytes'],
        f'{binding_id}.{kernel.variant}.expert{expert}.formatted_weight')


def build_packed_fused_moe_source(kernel, hardware, bindings, granularity, *, program_id):
    if not isinstance(bindings,FusedMoEBindings):
        raise TypeError('packed grouped MoE requires typed input/ND descriptor/formatted weight/output bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise NotImplementedError('packed grouped MoE exposes logical output dots; native issue order remains unknown')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    query,weight,output = (views[name] for name in (bindings.query_view,bindings.weight_view,bindings.output_view))
    for view,(_,shape) in zip((query,weight,output),_layouts(kernel,hardware),strict=True):
        strides,size = [],kernel.shape['dtype_bytes']
        for axis in reversed(shape):
            strides.append(size)
            size *= axis
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes) != (shape,tuple(reversed(strides)),kernel.dtype,kernel.shape['dtype_bytes']):
            raise ValueError('packed grouped MoE parent shape/storage mismatch')
    _,counts = fused_moe_routing(kernel)
    if bindings.formatted_weight is None or len(bindings.formatted_ready) != sum(count > 0 for count in counts):
        raise ValueError('packed grouped MoE requires each active formatted-weight availability')
    if len({query.allocation_id,weight.allocation_id,output.allocation_id,bindings.routing,bindings.formatted_weight}) != 5:
        raise NotImplementedError('ND descriptor and formatted weight must retain separate storage identity')
    route = _routing_region(kernel,bindings.binding_id,packed=True)
    routing = LogicalSelectionView(f'{program_id}.group_list',bindings.routing,route.logical_shape,
        route.dtype,route.element_bytes,route.selection_id,None)
    selected = [routing]
    for expert,count in enumerate(counts):
        if count:
            region = _formatted_region(kernel,bindings.binding_id,expert)
            selected.append(LogicalSelectionView(f'{program_id}.{expert}.weight',bindings.formatted_weight,
                region.logical_shape,region.dtype,region.element_bytes,region.selection_id,None))
    views.update((view.view_id,view) for view in selected)
    header = ('fused_moe_packed_tiles',kernel.variant,query.view_id,weight.view_id,output.view_id,
        routing.view_id,bindings.formatted_weight,','.join(map(str,counts)))
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-packed-expert-dots')
    slices,family = _packed_tile_recipe(views,scope,bindings.query_ready,bindings.routing_ready,
        bindings.formatted_ready,bindings.visibility_domain,
        query_unmapped=unique_by(bindings.allocations,'allocation_id')[query.allocation_id].size_bytes is None)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Expert row segments and full-K output dots are logical tiles; native NZ addresses/capacity/alias remain unknown.',
            'Public ND descriptor is not a second steady-state operand read; formatted weight availability is an invocation-boundary fact.',
            'Group-list counts/cumsum mode and native scan multiplicity remain unbound; its complete INT64 interface is read once.',
            'Existing CANN tile/cache/store/scan price projections remain separate from logical dot rereads.',
            'Cold ND-to-NZ conversion, persistent weight lifetimes, EP/quantized paths and wrapper scratch remain partial.')),
        bindings.allocations,(*views.values(),*slices),(),bindings.events,(),(family,))
    packed_moe_tile_demands(program)
    return program


def _packed_tile_recipe(views,scope,query_ready,routing_ready,formatted_ready,visibility, *, query_unmapped=False):
    header = scope.op_ids
    query,descriptor,output,routing = (views[name] for name in header[2:6])
    counts = tuple(map(int,header[7].split(',')))
    reduction,columns = descriptor.shape[1:]
    evidence = ('CANN GroupedMatmul: prefix sums of actual expert counts select packed input/output rows.',
        'Each logical output dot closes the complete K input/weight selection; experts and output coordinates are independent.',
        'Preformatted NZ weight selections retain actual per-expert availability and unknown physical coordinates.')
    accesses = [TileAccessTemplate('group_list',routing.view_id,Operation.READ,(),(None,),
        (routing_ready,),(),evidence,unmapped=True)]
    slices,axes,offset = [],[],0
    availability = iter(formatted_ready)
    for expert,count in enumerate(counts):
        if not count:continue
        row,column = f'expert{expert}.row',f'expert{expert}.column'
        axes.extend((TileAxis(row,count,1),TileAxis(column,columns,1)))
        operands = []
        for role,parent,width in (('input',query,reduction),('output',output,columns)):
            if role=='input' and query_unmapped:
                view=LogicalSelectionView(f'{scope.scope_id}.{expert}.{role}',parent.allocation_id,
                    (count,width),parent.dtype,parent.element_bytes,f'{scope.scope_id}.{expert}.packed_input_prefix',parent.view_id,
                    (LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),offset),
                     LogicalCoordinateBinding(1,(LogicalCoordinateTerm(1,1),))))
            else:
                view = TensorView(f'{scope.scope_id}.{expert}.{role}',parent.allocation_id,
                    parent.offset_bytes+offset*width*parent.element_bytes,(count,width),parent.strides_bytes,
                    parent.dtype,parent.element_bytes)
            slices.append(view);operands.append(view)
        inp,out = operands
        weight = views[f'{scope.scope_id}.{expert}.weight']
        grouped = (TileDependency('group_list',EventKind.READ_READY,()),)
        accesses.extend((
            TileAccessTemplate(f'{expert}.input',inp.view_id,Operation.READ,(row,column),(row,None),
                (query_ready,),grouped,evidence,unmapped=query_unmapped),
            TileAccessTemplate(f'{expert}.weight',weight.view_id,Operation.READ,(row,column),(None,column),
                (next(availability),),grouped,evidence,unmapped=True),
            TileAccessTemplate(f'{expert}.output',out.view_id,Operation.WRITE,(row,column),(row,column),(),(
                TileDependency(f'{expert}.input',EventKind.READ_READY,()),
                TileDependency(f'{expert}.weight',EventKind.READ_READY,())),evidence)))
        offset += count
    return tuple(slices),TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),visibility,evidence)


def packed_moe_tile_demands(program):
    """Fail-closed packed expert selections and unchanged interface projection."""
    program.__post_init__()
    if len(program.tile_programs) != 1:raise ValueError('packed MoE requires one actual group-list/dot pipeline')
    family, = program.tile_programs
    header = family.scope.op_ids
    if len(header) != 8 or header[0] != 'fused_moe_packed_tiles' or header[1] not in ('w1','w2'):
        raise ValueError('packed grouped MoE requires its typed parent and count header')
    views,accesses = unique_by(program.views,'view_id'),unique_by(family.accesses,'access_id')
    query,descriptor,output,routing = (views[name] for name in header[2:6])
    counts = tuple(map(int,header[7].split(',')))
    experts,reduction,columns = descriptor.shape
    if len(counts) != experts or min(counts) < 0 or sum(counts) != query.shape[0] or not sum(counts):
        raise ValueError('packed grouped MoE expert rows must cover the packed parent once')
    if query.shape[1] != reduction or output.shape != (sum(counts),columns):
        raise ValueError('packed grouped MoE ND descriptor/input/output geometry mismatch')
    if len({query.allocation_id,descriptor.allocation_id,output.allocation_id,routing.allocation_id,header[6]}) != 5:
        raise ValueError('packed grouped MoE ND and formatted weight storage identities must be separate')
    for view in (query,descriptor,output):
        strides,size = [],view.element_bytes
        for extent in reversed(view.shape):strides.append(size);size *= extent
        if view.strides_bytes != tuple(reversed(strides)) or (view.dtype,view.element_bytes) != (query.dtype,query.element_bytes):
            raise ValueError('packed grouped MoE public tensor storage dtype/layout mismatch')
    if not isinstance(routing,LogicalSelectionView) or (routing.shape,routing.dtype,routing.element_bytes) != ((experts,),'int64',8):
        raise ValueError('packed grouped MoE group-list storage mismatch')
    weights,ready = [],[]
    for expert,count in enumerate(counts):
        if not count:continue
        view = views[f'{program.program_id}.{expert}.weight']
        if (not isinstance(view,LogicalSelectionView) or view.allocation_id != header[6] or view.parent_view_id is not None
                or (view.shape,view.dtype,view.element_bytes) != ((reduction,columns),query.dtype,query.element_bytes)):
            raise ValueError('packed grouped MoE formatted weight must retain unresolved NZ layout')
        weights.append(view)
        access = accesses[f'{expert}.weight']
        if len(access.input_ready) != 1:raise ValueError('packed grouped MoE formatted availability missing')
        ready.append(access.input_ready[0])
    first = next(expert for expert,count in enumerate(counts) if count)
    if len(accesses[f'{first}.input'].input_ready) != 1 or len(accesses['group_list'].input_ready) != 1:
        raise ValueError('packed grouped MoE requires actual input/group-list availability')
    slices,expected = _packed_tile_recipe(views,family.scope,accesses[f'{first}.input'].input_ready[0],
        accesses['group_list'].input_ready[0],tuple(ready),family.visibility_domain,
        query_unmapped=unique_by(program.allocations,'allocation_id')[query.allocation_id].size_bytes is None)
    if family != expected or any(views[view.view_id] != view for view in slices):
        raise ValueError('packed grouped MoE geometry or actual group/input/formatted-weight producer causality changed')
    if set(views) != {query.view_id,descriptor.view_id,output.view_id,routing.view_id,
            *(view.view_id for view in weights),*(view.view_id for view in slices)}:
        raise ValueError('packed grouped MoE contains an undeclared operand selection')
    width = query.element_bytes
    return {'projection':(header[1],query.shape[0],reduction,columns),'routing':(counts,counts),
        'read_bytes':experts*8+sum(count*reduction+reduction*columns for count in counts if count)*width,
        'write_bytes':sum(counts)*columns*width}


def has_moe_alignment_source(kernel,hardware):
    from src.kernel.launch_config.select_launch_config import moe_count_sort_source_program
    return (kernel.kind is KernelKind.VECTOR and kernel.variant in ('moe_align','moe_align_small_batch')
        and kernel.execution_support=='supported' and hardware.kernel_memory_model=='gpu_mshr'
        and kernel.launch.algorithm in ('moe_align','observed_moe_aux:moe_align','moe_align_small_batch','observed_moe_aux:moe_align_small_batch')
        and moe_count_sort_source_program(kernel.hardware_id)=='cuda_moe_count_sort_int32_unmasked'
        and ({'alignment_block_size','alignment_capacity','tokens_per_expert'}<=set(kernel.attributes)
            or kernel.variant=='moe_align_small_batch' and kernel.hardware_id in ('nvidia_a800','nvidia_h200')
            and not {'alignment_block_size','alignment_capacity','tokens_per_expert'}&set(kernel.attributes)
            and 0<kernel.shape['tokens']*kernel.shape['topk']<1024 and 0<kernel.shape['experts']<=64
            and kernel.launch.grid==(1,1,1) and (kernel.launch.block==(max(kernel.shape['experts'],32),1,1)
                or kernel.hardware_id=='nvidia_h200' and kernel.launch.block==(max(kernel.shape['experts'],32)+256,1,1))
            or kernel.variant=='moe_align' and kernel.hardware_id in ('nvidia_a800','nvidia_h200')
            and not {'alignment_block_size','alignment_capacity','tokens_per_expert'}&set(kernel.attributes)
            and 0<kernel.shape['experts']<1024 and kernel.shape['tokens']*kernel.shape['topk']>0
            and kernel.launch.block==(1024,1,1)
            and kernel.launch.grid in (((1,1,1),(2,1,1)) if kernel.hardware_id=='nvidia_h200' else ((1,1,1),))))


def standalone_moe_alignment_bindings(kernel, *, call_id,binding_id, device_id="logical-local-device", rank=0, stream="offline"):
    from src.kernel.memory_moe_alignment import small_alignment_bindings,regular_alignment_bindings
    if kernel.variant=='moe_align_small_batch':return small_alignment_bindings(kernel,call_id=call_id,binding_id=binding_id,device_id=device_id,rank=rank,stream=stream)
    return regular_alignment_bindings(kernel,call_id=call_id,binding_id=binding_id,device_id=device_id,rank=rank,stream=stream)


def build_moe_alignment_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_moe_alignment import build_small_alignment_source,build_regular_alignment_source
    if not has_moe_alignment_source(kernel,hardware):raise ValueError('Alignment source requires the declared CUDA algorithm and actual launch')
    if kernel.variant=='moe_align_small_batch':return build_small_alignment_source(kernel,bindings,granularity,program_id=program_id)
    return build_regular_alignment_source(kernel,bindings,granularity,program_id=program_id)


def moe_alignment_source_demands(program, *, expected_kernel):
    from src.kernel.memory_moe_alignment import small_alignment_source_demands,regular_alignment_source_demands
    if expected_kernel.variant=='moe_align_small_batch':return small_alignment_source_demands(program,expected_kernel=expected_kernel)
    return regular_alignment_source_demands(program,expected_kernel=expected_kernel)
