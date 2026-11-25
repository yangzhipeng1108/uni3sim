"""Dense FA2 external tensor interfaces; launch replay remains a projection."""

from src.kernel.prediction_cache import runtime_lru_cache
from dataclasses import replace
from math import prod,ceil
from contextlib import contextmanager
from contextvars import ContextVar,copy_context

from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity,
    MemoryAccess, MemoryAccessProgram, MemoryAllocation, Operation, Ordering,
    ProgramEvent, Semantics, TensorView, LogicalSelectionView, Unmapped, Strided, requested_bytes, unique_by,
    TileAxis,TileAccessTemplate,TileDependency,TileProgram,TileCausalPrefix,TileIndexBinding,TileIndexTerm,
    LogicalCoordinateBinding,LogicalCoordinateTerm,ConstantValue,
)

from src.kernel.launch_config.tiling_spec import (
    METAX_MLA_KV_BLOCK,METAX_MLA_SCHEDULER_PARTITIONS,derive_mla_scheduler_work,metax_mla_combine_block_m,
    flash_attention_packed_varlen_scheduled_workgroups,
)

_MLA_WORK = ContextVar('metax_mla_source_work')
_GROUPED_MLA_WORK = ContextVar('grouped_mla_source_work')


def has_metax_mla_metadata_source(kernel):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant=='flash_mla_metadata'
        and kernel.hardware_id=='metax_c500' and kernel.execution_support=='supported')


def fa3_decode_source_splits(q_lens,kv_lens,query_heads,static_splits,num_sms):
    """Use the existing FA3 source scheduler rule unchanged."""
    from src.kernel.predictors.flash_attn import fa3_decode_dynamic_splits
    return fa3_decode_dynamic_splits(q_lens,kv_lens,query_heads,static_splits,num_sms)


def has_metax_mla_source(kernel):
    if (kernel.kind is not KernelKind.FLASH_ATTN or kernel.hardware_id != 'metax_c500'
            or kernel.execution_support != 'supported' or kernel.dtype != 'bf16'
            or kernel.launch.algorithm not in ('metax_flash_attention_persistent_queue','metax_flash_mla_combine')):
        return False
    if (kernel.launch.algorithm == 'metax_flash_attention_persistent_queue' and kernel.variant != 'decode'):
        return False
    if kernel.launch.algorithm == 'metax_flash_mla_combine' and kernel.variant not in ('combine','split_combine'):
        return False
    if 'model_input_q_lens' in kernel.attributes:
        if (tuple(kernel.attributes['q_lens']),tuple(kernel.attributes['kv_lens'])) != (
                tuple(kernel.attributes['model_input_q_lens']),tuple(kernel.attributes['model_input_kv_lens'])):
            return False
    return True


def metax_mla_scheduler_work(kernel):
    if kernel.launch.algorithm not in ('metax_flash_attention_persistent_queue','metax_flash_mla_combine'):
        raise ValueError('MetaX FlashMLA scheduler requires its runtime family')
    shape = kernel.shape
    kv_lens,q_lens = tuple(kernel.attributes['kv_lens']),tuple(kernel.attributes['q_lens'])
    if ((shape['q_len'],shape['head_dim'],shape['num_query_heads'],shape['num_kv_heads']) != (1,512,64,1)
            or kernel.dtype != 'bf16' or kernel.attributes['kv_layout'] != 'mla'
            or kernel.attributes['causal'] is not True or len(q_lens) != shape['batch']
            or len(kv_lens) != shape['batch'] or any(length != 1 for length in q_lens)):
        raise ValueError('MetaX FlashMLA source requires its executed Q1/D512/HQ64/HKV1 BF16 ABI')
    work = derive_mla_scheduler_work(kv_lens)
    if kernel.launch.tiling.num_kv_splits != max(work.splits_per_request):
        raise ValueError('MetaX FlashMLA recorded split count disagrees with source scheduler')
    return work


def _mla_stage(kernel):
    return 'main' if kernel.launch.algorithm == 'metax_flash_attention_persistent_queue' else 'combine'


def _mla_layouts(kv_lens,work,stage):
    batch = len(kv_lens)
    layouts = (('query',(batch,64,576),'bf16',2),('cache',(sum(kv_lens),576),'bf16',2)) if stage == 'main' else ()
    layouts += (('output',(batch,64,512),'bf16',2),('lse',(batch,64),'fp32',4))
    segments = sum(count for count in work.splits_per_request if count > 1)
    if segments:
        layouts += (('partial_output',(segments,64,512),'fp32',4),('partial_lse',(segments,64),'fp32',4))
    from src.kernel.memory_metax_mla_controls import control_layouts
    return (*layouts,*control_layouts(kv_lens,stage))


def _mla_requests(work,stage):
    reads,writes = [],[]
    for request,count in enumerate(work.splits_per_request):
        if stage == 'main':
            reads.extend((('query',request,Operation.READ,count),('cache',request,Operation.READ,1)))
            roles = ('output','lse') if count == 1 else ('partial_output','partial_lse')
        elif count > 1:
            reads.extend((('partial_output',request,Operation.READ,1),('partial_lse',request,Operation.READ,1)))
            roles = ('output','lse')
        else:
            continue
        writes.extend((role,request,Operation.WRITE,1) for role in roles)
    return tuple(reads+writes)


def _mla_region(view,role,request,kv_lens,work):
    shapes = {'query':(1,64,576),'cache':(kv_lens[request],576),
        'output':(1,64,512),'lse':(1,64),
        'partial_output':(work.splits_per_request[request],64,512),
        'partial_lse':(work.splits_per_request[request],64)}
    selection = f'metax_mla.{role}.request{request}.kv{kv_lens[request]}.splits{work.splits_per_request[request]}'
    return Unmapped(shapes[role],view.dtype,view.element_bytes,selection)


def _metax_mla_tile_recipe(header,parents,input_ready,scope,visibility):
    from dataclasses import replace
    stage,kv_lens = header[1],tuple(map(int,header[2].split(',')))
    work = derive_mla_scheduler_work(kv_lens)
    requests = _mla_requests(work,stage)
    read_keys = tuple((role,request) for role,request,op,_ in requests if op is Operation.READ)
    if len(input_ready) != len(read_keys):raise ValueError('MetaX MLA tile source requires every actual request availability')
    ready = dict(zip(read_keys,input_ready,strict=True))
    roots,selected,families = {},[],[]
    for role,request,_,_ in requests:
        parent = parents[role];region = _mla_region(parent,role,request,kv_lens,work)
        view = LogicalSelectionView(f'{scope.scope_id}.{role}.{request}.selection',parent.allocation_id,
            region.logical_shape,parent.dtype,parent.element_bytes,region.selection_id,None)
        roots[(role,request)] = view;selected.append(view)
    evidence = ('Exact FlashMLA 104-partition scheduler segments retain request, first_kv_block, kv_blocks and split_index.',
        'KV16 panels partition each real segment with a masked token tail; its shared latent576 includes the leading512 value channels.',
        'Direct BF16 output and split FP32 output/LSE publish only their own request/segment closure; combine never reads a direct request.',
        'Logical head64 packs and per-request/segment selections do not resolve native page/scratch addresses or actor issue order.')

    def child(identity,role,request,shape,selection):
        root = roots[(role,request)]
        view = LogicalSelectionView(f'{identity}.{role}',root.allocation_id,shape,root.dtype,root.element_bytes,
            f'{root.selection_id}.{selection}',root.view_id)
        selected.append(view)
        return view

    if stage == 'main':
        for worker,segments in enumerate(work.segments_by_partition):
            for segment in segments:
                request = segment.request_index;split = segment.split_index
                start = segment.first_kv_block*METAX_MLA_KV_BLOCK
                stop = min(kv_lens[request],start+segment.kv_blocks*METAX_MLA_KV_BLOCK)
                identity = f'{scope.scope_id}.request{request}.split{split}'
                query = child(identity,'query',request,(64,576),'head_pack64')
                cache = child(identity,'cache',request,(stop-start,576),f'kv{start}:{stop}')
                direct = work.splits_per_request[request] == 1
                out_role,lse_role = ('output','lse') if direct else ('partial_output','partial_lse')
                output = child(identity,out_role,request,(64,512),f'segment{split}.head_pack64')
                lse = child(identity,lse_role,request,(64,),f'segment{split}.head_pack64')
                accesses = (
                    TileAccessTemplate('query',query.view_id,Operation.READ,('head',),('head',None),
                        (ready[('query',request)],),(),evidence,unmapped=True),
                    TileAccessTemplate('cache',cache.view_id,Operation.READ,('kv_panel',),('kv_panel',None),
                        (ready[('cache',request)],),(),evidence,unmapped=True),
                    TileAccessTemplate('output',output.view_id,Operation.WRITE,('head',),('head',None),(),(
                        TileDependency('query',EventKind.READ_READY,()),
                        TileDependency('cache',EventKind.READ_READY,('kv_panel',))),evidence,unmapped=True),
                    TileAccessTemplate('lse',lse.view_id,Operation.WRITE,('head',),('head',),(),(
                        TileDependency('query',EventKind.READ_READY,()),
                        TileDependency('cache',EventKind.READ_READY,('kv_panel',))),evidence,unmapped=True))
                family_scope = replace(scope,scope_id=identity,op_ids=(*header,'segment',str(worker),str(request),
                    str(split),str(segment.first_kv_block),str(segment.kv_blocks)),agent='logical-metax-mla-segments')
                families.append(TileProgram(identity,family_scope,(TileAxis('head',64,64),
                    TileAxis('kv_panel',stop-start,METAX_MLA_KV_BLOCK)),accesses,visibility,evidence))
    else:
        for request,count in enumerate(work.splits_per_request):
            if count == 1:continue
            identity = f'{scope.scope_id}.request{request}.combine'
            output = child(identity,'output',request,(64,512),'head_pack64')
            lse = child(identity,'lse',request,(64,),'head_pack64')
            accesses = (
                TileAccessTemplate('partial_output',roots[('partial_output',request)].view_id,Operation.READ,
                    ('split','head'),('split','head',None),(ready[('partial_output',request)],),(),evidence,unmapped=True),
                TileAccessTemplate('partial_lse',roots[('partial_lse',request)].view_id,Operation.READ,
                    ('split','head'),('split','head'),(ready[('partial_lse',request)],),(),evidence,unmapped=True),
                TileAccessTemplate('output',output.view_id,Operation.WRITE,('head',),('head',None),(),(
                    TileDependency('partial_output',EventKind.READ_READY,('split',)),
                    TileDependency('partial_lse',EventKind.READ_READY,('split',))),evidence,unmapped=True),
                TileAccessTemplate('lse',lse.view_id,Operation.WRITE,('head',),('head',),(),(
                    TileDependency('partial_lse',EventKind.READ_READY,('split',)),),evidence,unmapped=True))
            family_scope = replace(scope,scope_id=identity,op_ids=(*header,'combine_request',str(request)),
                agent='logical-metax-mla-reduction')
            families.append(TileProgram(identity,family_scope,(TileAxis('split',count,1),TileAxis('head',64,metax_mla_combine_block_m(len(kv_lens),1,64))),
                accesses,visibility,evidence))
    return tuple(selected),tuple(families)


def metax_mla_tile_demands(program):
    """Check exact scheduler/panel closures before deriving legacy projections."""
    program.__post_init__()
    if not program.scopes or program.scopes[0].op_ids[:1] != ('metax_mla',):
        raise ValueError('MetaX MLA requires its request-domain source header')
    header = program.scopes[0].op_ids
    if len(header) != 3 or header[1] not in ('main','combine'):
        raise ValueError('MetaX MLA source header mismatch')
    stage,kv_lens = header[1],tuple(map(int,header[2].split(',')))
    work = derive_mla_scheduler_work(kv_lens)
    layouts,requests = _mla_layouts(kv_lens,work,stage),_mla_requests(work,stage)
    views = unique_by(program.views,'view_id')
    parents = {role:views[role] for role,_,_,_ in layouts}
    if len({view.allocation_id for view in parents.values()}) != len(layouts):
        raise ValueError('MetaX MLA logical roles require distinct parents within one invocation')
    for role,shape,dtype,width in layouts:
        view = parents[role]
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes,view.offset_bytes) != (
                shape,dtype,width,dense_strides(shape,width),0):
            raise ValueError('MetaX MLA source storage dtype or semantic extent mismatch')
    from src.kernel.memory_metax_mla_controls import control_layouts,attach_controls
    controls=control_layouts(kv_lens,stage);ready = {};control_ready={}
    for family in program.tile_programs:
        if family.scope.op_ids[:3] != header:raise ValueError('MetaX MLA mixed request domain')
        for access in family.accesses:
            if access.operation is Operation.READ:
                if access.scope is not None and access.scope.agent=='logical-metax-mla-shared-state':continue
                if len(access.input_ready) != 1:raise ValueError('MetaX MLA actual request input readiness missing')
                if access.scope is None:
                    view=views[access.view_id]
                    while isinstance(view,LogicalSelectionView) and view.parent_view_id is not None:view=views[view.parent_view_id]
                    role=view.view_id
                    if role in control_ready and control_ready[role]!=access.input_ready[0]:raise ValueError('MetaX MLA control availability changed across readers')
                    control_ready[role]=access.input_ready[0];continue
                if access.scope.op_ids[:3]!=header:raise ValueError('MetaX MLA payload source domain differs')
                request = int(access.scope.op_ids[5] if stage == 'main' else access.scope.op_ids[4])
                role = access.access_id.rsplit('.',1)[1]
                key = (role,request)
                if key in ready and ready[key] != access.input_ready[0]:
                    raise ValueError('MetaX MLA panels must preserve actual per-request readiness')
                ready[key] = access.input_ready[0]
    selected,families = _metax_mla_tile_recipe(header,parents,tuple(ready[(role,request)]
        for role,request,op,_ in requests if op is Operation.READ),program.scopes[0],
        program.tile_programs[0].visibility_domain)
    if set(control_ready)!={role for role,*_ in controls}:raise ValueError('MetaX MLA omitted actual control reads')
    control_views,families=attach_controls(header,parents,tuple(control_ready[role] for role,*_ in controls),work,
        program.scopes[0],program.tile_programs[0].visibility_domain,families)
    selected=(*selected,*control_views)
    from src.kernel.memory_metax_mla_scratch import attach_main_scratch
    family,=families
    scratch_allocations,scratch_views,scratch_events,family=attach_main_scratch(header,program.scopes[0],family,(*parents.values(),*selected))
    families=(family,);selected=(*selected,*scratch_views)
    if (program.tile_programs != families or set(views) != {*parents,*(view.view_id for view in selected)}
            or any(views[view.view_id] != view for view in selected)
            or any(item not in program.allocations for item in scratch_allocations)
            or any(item not in program.events for item in scratch_events)):
        raise ValueError('MetaX MLA requires exact scheduler segments and actual request operand or partial READ_READY closure')
    from src.kernel.memory_access import _tile_template_requested_bytes
    roots = {f'{program.scopes[0].scope_id}.{role}.{request}.selection':(role,request)
        for role,request,_,_ in requests}
    totals = {(role,request):0 for role,request,_,_ in requests}
    for family in program.tile_programs:
        axes = unique_by(family.axes,'axis_id')
        for access in family.accesses:
            if access.scope is None or access.scope.agent=='logical-metax-mla-shared-state':continue
            view = views[access.view_id]
            while isinstance(view,LogicalSelectionView) and view.parent_view_id is not None:
                view = views[view.parent_view_id]
            totals[roots[view.view_id]] += _tile_template_requested_bytes(views,axes,family,access)
    # One complete query selection is reread by every actual segment. This
    # validated multiplicity is removed only for the legacy compulsory view;
    # all requested/read bytes still come from the sole tile source above.
    volumes = {}
    for role,request,_,repeat in requests:
        volume,remainder = divmod(totals[(role,request)],repeat)
        if remainder:raise ValueError('MetaX MLA request projection lost exact segment multiplicity')
        volumes[(role,request)] = volume
    result=_metax_mla_prediction_demand(stage,kv_lens,work,parents,requests,volumes)
    result['control_read_bytes']=sum(_tile_template_requested_bytes(views,unique_by(family.axes,'axis_id'),family,access)
        for family in program.tile_programs for access in family.accesses if access.scope is None)
    allocations=unique_by(program.allocations,'allocation_id')
    for domain,field in (('logical-shared','shared'),('logical-register-state','register_state')):
        for operation in (Operation.READ,Operation.WRITE):
            result[f'{field}_{operation.value}_bytes']=sum(_tile_template_requested_bytes(views,unique_by(family.axes,'axis_id'),family,access)
                for family in program.tile_programs for access in family.accesses
                if access.operation is operation and allocations[views[access.view_id].allocation_id].memory_domain.startswith(domain))
    result['logical_total_read_bytes']=result['read_bytes']+result['control_read_bytes']+result['shared_read_bytes']+result['register_state_read_bytes']
    result['logical_total_write_bytes']=result['write_bytes']+result['shared_write_bytes']+result['register_state_write_bytes']
    if stage=='main':
        issued_padding=0
        for family in program.tile_programs:
            axes=unique_by(family.axes,'axis_id')
            for access in family.accesses:
                if access.scope is None or access.scope.agent!='logical-metax-mla-shared-state':continue
                if not access.access_id.endswith(('.cache_shared_write','.key_shared_read','.value_shared_read')):continue
                axis,=access.iteration_axes;tile=axes[axis];view=views[access.view_id]
                issued_padding+=(tile.count*tile.block-tile.extent)*view.shape[-1]*view.element_bytes
        if result['smem_projection_bytes']!=result['shared_read_bytes']+result['shared_write_bytes']+issued_padding:
            raise ValueError('MetaX MLA analytical staging differs from its actual shared tile requests')
        result['shared_issued_padding_projection_bytes']=issued_padding
    return result


def _metax_mla_projected_demands(stage,kv_lens,work,views,requests,volumes):
    read = sum(volumes[(role,request)]*repeat for role,request,op,repeat in requests if op is Operation.READ)
    write = sum(volumes[(role,request)] for role,request,op,_ in requests if op is Operation.WRITE)
    if stage == 'main':
        compulsory = sum(volumes[(role,r)] for r in range(len(kv_lens)) for role in ('query','cache'))
        padded = work.kv_block_count*METAX_MLA_KV_BLOCK
        cache_extra = (padded-sum(kv_lens))*576*views['cache'].element_bytes
        l2,dram = read+cache_extra+write,compulsory+write
        # Filled from validated shared-stage reads/writes and their exact
        # masked panel padding in metax_mla_tile_demands, never rebuilt here.
        smem = 0
    else:
        compulsory,l2,dram,smem = read,read+write,read+write,0
    return {'stage':stage,'kv_lens':kv_lens,'work':work,'read_bytes':read,'write_bytes':write,
        'compulsory_input_bytes':compulsory,'l2_projection_bytes':l2,'dram_projection_bytes':min(l2,dram),
        'smem_projection_bytes':smem}


def _metax_mla_prediction_demand(stage,kv_lens,work,views,requests,volumes):
    result=_metax_mla_projected_demands(stage,kv_lens,work,views,requests,volumes)
    if stage=='main':
        from src.kernel.memory_metax_mla_scratch import main_smem_prediction_bytes
        query_bytes=sum(volumes[(role,request)]*repeat for role,request,op,repeat in requests
            if role=='query' and op is Operation.READ)
        output_bytes=sum(volumes[(role,request)] for role,request,op,_ in requests
            if role in ('output','partial_output') and op is Operation.WRITE)
        result['smem_projection_bytes']=main_smem_prediction_bytes(work,query_bytes,output_bytes)
    return result


def metax_mla_prediction_work(kernel,bindings):
    """Evaluate the same payload/staging projection at the public boundary."""
    from src.kernel.memory_access import covers,validate_storage_bindings
    work,kv_lens,stage,requests,controls,payload_reads,views=_metax_mla_public_operands(kernel,bindings)
    bindings.__post_init__()
    validate_storage_bindings(bindings.allocations,bindings.views,bindings.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual MetaX MLA public boundary; native segment/control/private state is exported separately.',)))
    required=tuple((role,_mla_region(views[role],role,request,kv_lens,work))
        for role,request,op,_ in requests if op is Operation.READ)
    required+=tuple((role,Unmapped(views[role].shape,views[role].dtype,views[role].element_bytes,
        f'metax_mla.{role}.control')) for role,*_ in controls)
    events=unique_by(bindings.events,'event_id')
    for (role,region),ref in zip(required,bindings.input_ready,strict=True):
        if ref.node_id not in events:raise ValueError('MetaX MLA input readiness is missing')
        event=events[ref.node_id];parent=views[role]
        logical=ref.kind is EventKind.LOGICAL_AVAILABLE and event.region==region
        if (event.kind is not ref.kind or ref.allow_skipped_read
                or ref.kind not in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE,EventKind.LOGICAL_AVAILABLE)
                or event.allocation_id!=parent.allocation_id or event.visibility_domain!=bindings.visibility_domain
                or event.view_id is not None and event.view_id!=parent.view_id
                or not (logical or event.region is not None and covers(event.region,parent.region()))):
            raise ValueError('MetaX MLA requires actual per-request/control readiness')
    volumes={(role,request):requested_bytes(_mla_region(views[role],role,request,kv_lens,work))
        for role,request,_,_ in requests}
    return _metax_mla_prediction_demand(stage,kv_lens,work,views,requests,volumes)


def standalone_metax_mla_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    work = metax_mla_scheduler_work(kernel)
    kv_lens,stage = tuple(kernel.attributes['kv_lens']),_mla_stage(kernel)
    layouts = _mla_layouts(kv_lens,work,stage)
    allocations,views,events,ready = [],[],[],[]
    for role,shape,dtype,width in layouts:
        identity = f'{binding_id}.{role}'
        allocation = MemoryAllocation(identity,'logical-local-device','logical-native-tensor',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = TensorView(role,identity,0,shape,dense_strides(shape,width),dtype,width)
        allocations.append(allocation); views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
    by_role = unique_by(views,'view_id')
    requests = _mla_requests(work,stage)
    for role,request,operation,_ in requests:
        if operation is not Operation.READ:
            continue
        view = by_role[role]
        allocation = next(item for item in allocations if item.allocation_id == view.allocation_id)
        event = ProgramEvent(f'{binding_id}.{role}.{request}.available',EventKind.LOGICAL_AVAILABLE,
            view.allocation_id,_mla_region(view,role,request,kv_lens,work),'mla-visible',
            (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
        events.append(event); ready.append(EventRef(event.event_id,event.kind))
    from src.kernel.memory_metax_mla_controls import control_layouts
    controls=control_layouts(kv_lens,stage)
    for role,*_ in controls:
        view=by_role[role];allocation=next(item for item in allocations if item.allocation_id==view.allocation_id)
        event=ProgramEvent(f'{binding_id}.{role}.available',EventKind.LOGICAL_AVAILABLE,view.allocation_id,
            Unmapped(view.shape,view.dtype,view.element_bytes,f'metax_mla.{role}.control'),'mla-visible',
            (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
        events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        (*tuple(role for role,_,op,_ in requests if op is Operation.READ),*(role for role,*_ in controls)),
        tuple(role for role,_,_,_ in layouts if any(r == role and op is Operation.WRITE for r,_,op,_ in requests)),
        tuple(ready),0,'offline','mla-visible')


def _metax_mla_public_operands(kernel,bindings):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_metax_mla_source(kernel):
        raise NotImplementedError('MetaX MLA source outside its retained nonpadded scheduler domain')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('MetaX MLA requires typed query/cache/output/LSE/partial bindings')
    work = metax_mla_scheduler_work(kernel)
    kv_lens,stage = tuple(kernel.attributes['kv_lens']),_mla_stage(kernel)
    if stage == 'main':
        expected_grid,block = (1,1,METAX_MLA_SCHEDULER_PARTITIONS),(512,1,1)
    else:
        block_m = metax_mla_combine_block_m(len(kv_lens),1,64)
        expected_grid,block = (-(-len(kv_lens)*64//block_m),1,1),(256,1,1)
    if kernel.launch.grid != expected_grid or kernel.launch.block != block:
        raise ValueError('MetaX FlashMLA source grid/block disagrees with source')
    layouts,requests = _mla_layouts(kv_lens,work,stage),_mla_requests(work,stage)
    from src.kernel.memory_metax_mla_controls import control_layouts
    controls=control_layouts(kv_lens,stage)
    payload_reads=tuple(role for role,_,op,_ in requests if op is Operation.READ)
    expected_reads = (*payload_reads,*(role for role,*_ in controls))
    expected_writes = tuple(role for role,_,_,_ in layouts if any(r == role and op is Operation.WRITE for r,_,op,_ in requests))
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (expected_reads,expected_writes,len(expected_reads)):
        raise ValueError('MetaX MLA must retain all query/cache or split-output/LSE roles')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    if set(views) != {role for role,_,_,_ in layouts} or len({v.allocation_id for v in views.values()}) != len(layouts):
        raise ValueError('MetaX MLA role alias requires an independently declared native mapping')
    for role,shape,dtype,width in layouts:
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes,view.offset_bytes) != (
                shape,dtype,width,dense_strides(shape,width),0):
            raise ValueError('MetaX MLA logical interface shape/dtype/selection descriptor mismatch')
    return work,kv_lens,stage,requests,controls,payload_reads,views


def build_metax_mla_source(kernel,bindings,granularity, *, program_id):
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise NotImplementedError('MetaX MLA exports logical segment/panel tiles; native addresses remain unknown')
    work,kv_lens,stage,requests,controls,payload_reads,views=_metax_mla_public_operands(kernel,bindings)
    from src.kernel.memory_metax_mla_controls import attach_controls
    header = ('metax_mla',stage,','.join(map(str,kv_lens)))
    scope = AccessScope(f'{program_id}.interface',header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-mla-interface')
    selected,families = _metax_mla_tile_recipe(header,views,bindings.input_ready[:len(payload_reads)],scope,bindings.visibility_domain)
    control_views,families=attach_controls(header,views,bindings.input_ready[len(payload_reads):],work,scope,bindings.visibility_domain,families)
    from src.kernel.memory_metax_mla_scratch import attach_main_scratch
    family,=families
    scratch_allocations,scratch_views,scratch_events,family=attach_main_scratch(header,scope,family,(*bindings.views,*selected,*control_views))
    families=(family,)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Exact scheduler segments and masked KV16 panels; native page mappings, capacities and scratch layout remain unknown.',
            'One shared latent576 panel includes the leading512 value channels; no independent cache allocation or physical transaction claim.',
            'Actual scheduler/length/split-prefix/page-table control reads, logical shared staging and register-only state carries are bound; native scratch addresses/issue order remain unresolved.',
            'Direct and split outputs retain disjoint logical request/segment selections; only split requests enter combine.',
            'Existing cache/SMEM/DRAM/compute price projections remain unchanged and do not assert measured residency.')),
        (*bindings.allocations,*scratch_allocations),(*bindings.views,*selected,*control_views,*scratch_views),(scope,),
        (*bindings.events,*scratch_events),(),families)
    metax_mla_source_demands(program,expected_kernel=kernel)
    return program


def metax_mla_source_demands(program, *, expected_kernel=None):
    if expected_kernel is not None:
        if (not has_metax_mla_source(expected_kernel) or program.hardware_id!=expected_kernel.hardware_id
                or len(program.scopes)!=1 or program.scopes[0].op_ids!=('metax_mla',_mla_stage(expected_kernel),
                    ','.join(map(str,expected_kernel.attributes['kv_lens'])))):
            raise ValueError('MetaX MLA source differs from its actual request/kernel binding')
    if not program.tile_programs:raise ValueError('MetaX MLA requires its actual compressed tile source, including direct-request prefix reads')
    return metax_mla_tile_demands(program)


@contextmanager
def metax_mla_source_work(kernel,work):
    token = _MLA_WORK.set((kernel,work))
    try:
        yield
    finally:
        _MLA_WORK.reset(token)


def current_metax_mla_source_work(kernel):
    context = copy_context()
    if _MLA_WORK in context:
        bound,work = context[_MLA_WORK]
        if bound != kernel:
            raise ValueError('MetaX MLA source crossed a kernel boundary')
        return work
    from src.kernel.memory_access_builder import _prediction_bindings
    bindings,_=_prediction_bindings(kernel,None)
    if bindings is None:bindings=standalone_metax_mla_bindings(kernel,call_id='mla',binding_id='mla')
    return metax_mla_prediction_work(kernel,bindings)


def has_dense_flash_attention_source(kernel):
    if kernel.kind is not KernelKind.FLASH_ATTN or kernel.execution_support != 'supported' or kernel.launch is None:
        return False
    # Both retained runners construct dense BSHD tensors for uniform Q1.
    # C500 varlen prefill uses a different packed LSE ABI and is not inferred.
    native = ((kernel.hardware_id == 'nvidia_a800' and (kernel.launch.algorithm=='flash_attn_dense' or _dense_paged_source(kernel)))
        or (kernel.hardware_id == 'metax_c500' and kernel.variant == 'decode' and kernel.shape['q_len'] == 1
            and kernel.shape['head_dim'] in (64,128,192) and kernel.dtype == 'bf16' and kernel.launch.algorithm in (
                'metax_flash_attention_trait_direct:decode','metax_flash_attention_trait_direct:decode_q64_full_tile')))
    return (kernel.kind is KernelKind.FLASH_ATTN and native
            and kernel.variant in ('prefill','decode') and kernel.execution_support == 'supported'
            and (kernel.attributes['kv_layout'] == 'contiguous' or _dense_paged_source(kernel))
            and kernel.launch.tiling.num_kv_splits == 1
            and 'model_input_q_lens' not in kernel.attributes and 'model_input_kv_lens' not in kernel.attributes
            and ((all(length == kernel.shape['q_len'] for length in kernel.attributes['q_lens'])
                and all(length == kernel.shape['kv_len'] for length in kernel.attributes['kv_lens']))
                or _dense_paged_source(kernel) and _fa2_ragged_admitted(kernel)))


def has_grouped_mla_source(kernel):
    if (kernel.kind is not KernelKind.FLASH_ATTN or kernel.hardware_id != 'nvidia_a800'
            or kernel.dtype != 'bf16' or kernel.execution_support != 'supported'
            or kernel.launch.algorithm not in ('triton_grouped_mla_stage1','triton_grouped_mla_stage2')
            or 'model_input_q_lens' in kernel.attributes):
        return False
    shape = kernel.shape
    if (shape['q_len']!=1 or shape['num_kv_heads']!=1 or kernel.attributes['kv_layout']!='mla'
            or not all(q==1 for q in kernel.attributes['q_lens'])):return False
    return (kernel.launch.algorithm=='triton_grouped_mla_stage2'
        or (shape['head_dim'],shape['num_query_heads'])==(512,64))


def grouped_mla_valid_split_count(kv,splits):
    if kv <= 0 or splits <= 0:
        raise ValueError('Grouped MLA request length and split count must be positive')
    width = -(-kv//splits)
    return -(-kv//width)


def _grouped_mla_header_geometry(header):
    if len(header)==5:return 64,512
    if len(header)!=7 or header[1]!='combine':raise ValueError('Grouped MLA requires its declared stage/head/value ABI')
    heads,width=map(int,header[5:])
    if min(heads,width)<=0:raise ValueError('Grouped MLA requires positive head/value geometry')
    return heads,width


def _grouped_mla_layouts(batch,kv_rows,splits,stage,heads=64,value_width=512):
    # torch.empty[B,H,S,D+1] packs the FP32 output and its last-column LSE.
    partial_strides = (heads*splits*(value_width+1)*4,splits*(value_width+1)*4,(value_width+1)*4,4)
    layouts = [('lengths','lengths',(batch,),'int32',4,(4,),0,batch*4)]
    if stage == 'main':
        layouts += [('query','query',(batch,64,576),'bf16',2,(64*576*2,576*2,2),0,batch*64*576*2),
            ('block_table','block_table',(kv_rows,),'int32',4,(4,),0,None),
            ('key','cache',(kv_rows,576),'bf16',2,(576*2,2),0,None),
            ('value','cache',(kv_rows,512),'bf16',2,(576*2,2),0,None)]
    capacity=batch*heads*splits*(value_width+1)*4
    layouts += [('partial_output','partial',(batch,heads,splits,value_width),'fp32',4,partial_strides,0,capacity),
        ('partial_lse','partial',(batch,heads,splits),'fp32',4,partial_strides[:3],value_width*4,capacity)]
    if stage == 'combine':
        layouts += [('output','output',(batch,heads,value_width),'bf16',2,(heads*value_width*2,value_width*2,2),0,batch*heads*value_width*2),
            ('lse','lse',(batch,heads),'bf16',2,(heads*2,2),0,batch*heads*2)]
    return tuple(layouts)


def _grouped_mla_requests(stage,splits):
    if stage == 'main':
        return (('lengths',Operation.READ,4*splits),('query',Operation.READ,splits),
            ('block_table',Operation.READ,4),('key',Operation.READ,4),('value',Operation.READ,4),
            ('partial_output',Operation.WRITE,1),('partial_lse',Operation.WRITE,1))
    return (('lengths',Operation.READ,64),('partial_output',Operation.READ,1),
        ('partial_lse',Operation.READ,1),('output',Operation.WRITE,1),('lse',Operation.WRITE,1))


def _grouped_mla_read_roles(stage,kv_lens,splits):
    counts=tuple(grouped_mla_valid_split_count(kv,splits) for kv in kv_lens)
    if stage=='combine' and len(set(counts))>1:
        return ('lengths',*(f'{role}.request{request}' for request in range(len(kv_lens))
            for role in ('partial_output','partial_lse')))
    return tuple(role for role,operation,_ in _grouped_mla_requests(stage,splits) if operation is Operation.READ)


def _grouped_mla_binding_layouts(kv_lens,splits,stage,heads=64,value_width=512):
    layouts=_grouped_mla_layouts(len(kv_lens),sum(kv_lens),splits,stage,heads,value_width)
    if stage!='combine' or len({grouped_mla_valid_split_count(kv,splits) for kv in kv_lens})==1:return layouts
    parents={row[0]:row for row in layouts};selected=[]
    for request,kv in enumerate(kv_lens):
        valid=grouped_mla_valid_split_count(kv,splits)
        for role in ('partial_output','partial_lse'):
            _,parent,shape,dtype,width,strides,offset,capacity=parents[role]
            selected.append((f'{role}.request{request}',parent,(heads,valid,*shape[3:]),dtype,width,strides[1:],
                offset+request*strides[0],capacity))
    return (*layouts,*selected)


def _grouped_mla_region(view,role,valid_splits):
    if role in ('key','value','block_table'):
        return Unmapped(view.shape,view.dtype,view.element_bytes,f'triton_grouped_mla.{role}.valid_kv_rows')
    if role in ('partial_output','partial_lse'):
        return Strided(view.offset_bytes,(*view.shape[:2],valid_splits,*view.shape[3:]),
            view.strides_bytes,view.element_bytes)
    return view.region()


def _grouped_mla_tile_recipe(header,parents,ready,scope,visibility):
    from dataclasses import replace
    stage,kv_lens,splits = header[1],tuple(map(int,header[2].split(','))),int(header[3])
    heads,value_width=_grouped_mla_header_geometry(header)
    selected,families = [],[]
    mixed=stage=='combine' and len({grouped_mla_valid_split_count(kv,splits) for kv in kv_lens})>1
    evidence = ('vLLM0.11 triton_decode_attention.py grouped stage1: H16, KV32, cdiv(length,splits), query before empty-split guard.',
        'Req_to_tokens INT32 is loaded per valid KV lane before K/V; selected page values may repeat and imply no distinct storage or page mapping.',
        'Packed FP32[batch,head,split,value_width+1] preserves actual output/LSE offsets; stage2 masks values by Lv and reduces only valid splits per head.')
    for request,kv in enumerate(kv_lens):
        identity = f'{scope.scope_id}.request{request}'
        lengths = parents['lengths']
        length = TensorView(f'{identity}.lengths',lengths.allocation_id,lengths.offset_bytes+request*4,(1,),(4,),'int32',4)
        selected.append(length)
        axes = [TileAxis('head',heads,16 if stage == 'main' else 1)]
        accesses = []
        if stage == 'main':
            axes.append(TileAxis('split',splits,1))
            query = parents['query']
            qview = TensorView(f'{identity}.query',query.allocation_id,query.offset_bytes+request*query.strides_bytes[0],
                (64,576),query.strides_bytes[1:],'bf16',2)
            selected.append(qview)
            accesses.extend((
                TileAccessTemplate('lengths',length.view_id,Operation.READ,('head','split'),(None,),
                    (ready['lengths'],),(),evidence),
                TileAccessTemplate('query',qview.view_id,Operation.READ,('head','split'),('head',None),
                    (ready['query'],),(TileDependency('lengths',EventKind.READ_READY,()),),evidence)))
            width = -(-kv//splits);valid = -(-kv//width)
            full_count,tail = divmod(kv,width)
            groups = ([] if full_count == 0 else [('full',full_count,width,0)])
            if tail:groups.append(('tail',1,tail,full_count))
            if full_count+(1 if tail else 0) != valid:raise ValueError('Grouped MLA split partition lost its exact tail')
            for label,count,tokens,first in groups:
                split_axis,token_axis = f'{label}.split',f'{label}.token'
                axes.extend((TileAxis(split_axis,count,1),TileAxis(token_axis,tokens,32)))
                binding = (TileIndexBinding('split',(TileIndexTerm(split_axis,1),),first),)
                length_dep = TileDependency('lengths',EventKind.READ_READY,(),binding)
                query_dep = TileDependency('query',EventKind.READ_READY,(),binding)
                group_views = {}
                for role in ('block_table','key','value'):
                    parent = parents[role]
                    shape = (count,tokens) if role == 'block_table' else (count,tokens,parent.shape[1])
                    view = LogicalSelectionView(f'{identity}.{label}.{role}',parent.allocation_id,shape,parent.dtype,
                        parent.element_bytes,f'triton_grouped_mla.{role}.request{request}.split{first}:{first+count}.width{width}.tokens{tokens}',parent.view_id)
                    selected.append(view);group_views[role] = view
                loop = ('head',split_axis,token_axis)
                page = group_views['block_table']
                accesses.append(TileAccessTemplate(f'{label}.block_table',page.view_id,Operation.READ,loop,
                    (split_axis,token_axis),(ready['block_table'],),(length_dep,),evidence,unmapped=True))
                for role in ('key','value'):
                    view = group_views[role]
                    accesses.append(TileAccessTemplate(f'{label}.{role}',view.view_id,Operation.READ,loop,
                        (split_axis,token_axis,None),(ready[role],),
                        (TileDependency(f'{label}.block_table',EventKind.READ_READY,()),),evidence,unmapped=True))
                output_deps = (query_dep,TileDependency(f'{label}.key',EventKind.READ_READY,(token_axis,)),
                    TileDependency(f'{label}.value',EventKind.READ_READY,(token_axis,)))
                for role in ('partial_output','partial_lse'):
                    parent = parents[role]
                    shape = (64,count,512) if role == 'partial_output' else (64,count)
                    view = TensorView(f'{identity}.{label}.{role}',parent.allocation_id,
                        parent.offset_bytes+request*parent.strides_bytes[0]+first*parent.strides_bytes[2],shape,
                        parent.strides_bytes[1:],parent.dtype,parent.element_bytes)
                    selected.append(view)
                    deps = output_deps if role == 'partial_output' else output_deps[:2]
                    mapping = ('head',split_axis,None) if role == 'partial_output' else ('head',split_axis)
                    accesses.append(TileAccessTemplate(f'{label}.{role}',view.view_id,Operation.WRITE,
                        ('head',split_axis),mapping,(),deps,evidence))
        else:
            valid = grouped_mla_valid_split_count(kv,splits)
            axes.append(TileAxis('split',valid,1))
            accesses.append(TileAccessTemplate('lengths',length.view_id,Operation.READ,('head',),(None,),
                (ready['lengths'],),(),evidence))
            for role in ('partial_output','partial_lse','output','lse'):
                parent = parents[role]
                partial = role.startswith('partial_')
                shape = ((heads,valid,value_width) if role == 'partial_output' else (heads,valid)) if partial else ((heads,value_width) if role == 'output' else (heads,))
                view = TensorView(f'{identity}.{role}',parent.allocation_id,parent.offset_bytes+request*parent.strides_bytes[0],
                    shape,parent.strides_bytes[1:],parent.dtype,parent.element_bytes)
                selected.append(view)
                if partial:
                    mapping = ('head','split',None) if role == 'partial_output' else ('head','split')
                    accesses.append(TileAccessTemplate(role,view.view_id,Operation.READ,('head','split'),mapping,
                        (ready[f'{role}.request{request}' if mixed else role],),(TileDependency('lengths',EventKind.READ_READY,()),),evidence))
                else:
                    dependencies = (TileDependency('partial_lse',EventKind.READ_READY,('split',)),)
                    if role == 'output':dependencies = (TileDependency('partial_output',EventKind.READ_READY,('split',)),*dependencies)
                    accesses.append(TileAccessTemplate(role,view.view_id,Operation.WRITE,('head',),
                        ('head',None) if role == 'output' else ('head',),(),dependencies,evidence))
        family_scope = replace(scope,scope_id=identity,op_ids=(*header,str(request)),agent='logical-grouped-mla-panels')
        families.append(TileProgram(identity,family_scope,tuple(axes),tuple(accesses),visibility,evidence))
    return tuple(selected),tuple(families)


def grouped_mla_tile_demands(program):
    program.__post_init__()
    header = program.scopes[0].op_ids
    if len(header) not in (5,7) or header[0] != 'triton_grouped_mla' or header[1] not in ('main','combine'):
        raise ValueError('Grouped MLA requires its actual request/split header')
    stage,kv_lens,splits,scratch = header[1],tuple(map(int,header[2].split(','))),int(header[3]),int(header[4])
    heads,value_width=_grouped_mla_header_geometry(header)
    counts = {grouped_mla_valid_split_count(kv,splits) for kv in kv_lens}
    if scratch not in (0,2,4):raise ValueError('Grouped MLA source split/projection domain mismatch')
    valid=next(iter(counts)) if len(counts)==1 else None
    layouts = _grouped_mla_binding_layouts(kv_lens,splits,stage,heads,value_width)
    views = unique_by(program.views,'view_id');parents = {row[0]:views[row[0]] for row in layouts};allocations = {}
    for role,parent,shape,dtype,width,strides,offset,_ in layouts:
        view = parents[role]
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes,view.offset_bytes) != (shape,dtype,width,strides,offset):
            raise ValueError('Grouped MLA source requires actual packed FP32 partial and BF16 final LSE storage')
        if parent in allocations and allocations[parent] != view.allocation_id:
            raise ValueError('Grouped MLA source lost its shared packed partial or cache parent')
        allocations[parent] = view.allocation_id
    if len(set(allocations.values())) != len(allocations):raise ValueError('Grouped MLA source aliases unrelated parents')
    ready = {}
    for family in program.tile_programs:
        for template in family.accesses:
            if template.operation is not Operation.READ:continue
            role = template.access_id.rsplit('.',1)[-1]
            if stage=='combine' and len(counts)>1 and role in ('partial_output','partial_lse'):
                role=f'{role}.request{family.scope.op_ids[-1]}'
            if len(template.input_ready) != 1:raise ValueError('Grouped MLA requires each actual input availability')
            if role in ready and ready[role] != template.input_ready[0]:raise ValueError('Grouped MLA lost actual invocation input readiness')
            ready[role] = template.input_ready[0]
    events = unique_by(program.events,'event_id')
    if stage=='combine' and len(counts)>1:
        for role in _grouped_mla_read_roles(stage,kv_lens,splits)[1:]:
            event=events[ready[role].node_id]
            if event.kind is not EventKind.INITIALIZED or event.region!=parents[role].region():
                raise ValueError('Grouped MLA requires exact request-local partial availability, without empty slots')
    for role in ('block_table','key','value') if stage == 'main' else ():
        ref = ready[role];event = events[ref.node_id]
        if ref.kind is not EventKind.LOGICAL_AVAILABLE or event.region != _grouped_mla_region(parents[role],role,valid):
            raise ValueError('Grouped MLA requires exact selected map/cache availability')
    selected,families = _grouped_mla_tile_recipe(header,parents,ready,program.scopes[0],program.tile_programs[0].visibility_domain)
    if (program.tile_programs != families or set(views) != {*parents,*(view.view_id for view in selected)}
            or any(views[view.view_id] != view for view in selected)):
        raise ValueError('Grouped MLA requires exact typed reads/stores/repeats and actual operand and partial READ_READY closure')
    from src.kernel.memory_access import _tile_template_requested_bytes
    values = {role:0 for role,_,_ in _grouped_mla_requests(stage,splits)}
    for family in program.tile_programs:
        axes = unique_by(family.axes,'axis_id')
        for access in family.accesses:
            role = access.access_id.rsplit('.',1)[-1]
            values[role] += _tile_template_requested_bytes(views,axes,family,access)
    analytical=_grouped_mla_prediction_values(stage,kv_lens,splits,heads,value_width,parents)
    if values!=analytical:raise ValueError('Grouped MLA analytical work differs from its actual request tiles')
    return _grouped_mla_projected_demands(stage,kv_lens,splits,valid,scratch,parents,analytical)


def _grouped_mla_prediction_values(stage,kv_lens,splits,heads,value_width,views):
    batch=len(kv_lens);active=sum(grouped_mla_valid_split_count(kv,splits) for kv in kv_lens)
    partial={'partial_output':active*heads*value_width*views['partial_output'].element_bytes,
        'partial_lse':active*heads*views['partial_lse'].element_bytes}
    if stage=='main':
        groups=-(-heads//16)
        return {'lengths':batch*groups*splits*views['lengths'].element_bytes,
            'query':prod(views['query'].shape)*views['query'].element_bytes*splits,
            'block_table':sum(kv_lens)*groups*views['block_table'].element_bytes,
            'key':sum(kv_lens)*groups*views['key'].shape[-1]*views['key'].element_bytes,
            'value':sum(kv_lens)*groups*views['value'].shape[-1]*views['value'].element_bytes,**partial}
    return {'lengths':batch*heads*views['lengths'].element_bytes,**partial,
        **{role:prod(views[role].shape)*views[role].element_bytes for role in ('output','lse')}}


def grouped_mla_prediction_work(kernel,bindings):
    """Validate packed/native operands and evaluate the same request extents."""
    from src.kernel.memory_access import covers,validate_storage_bindings
    stage,splits,heads,value_width,views=_grouped_mla_public_operands(kernel,bindings)
    bindings.__post_init__();kv_lens=tuple(kernel.attributes['kv_lens'])
    counts={grouped_mla_valid_split_count(kv,splits) for kv in kv_lens}
    valid=next(iter(counts)) if len(counts)==1 else None
    scratch=int(kernel.launch.parameters['split_scratch_dram_bytes_per_element'])
    if scratch not in (0,2,4):raise ValueError('Grouped MLA source split/projection domain mismatch')
    validate_storage_bindings(bindings.allocations,bindings.views,bindings.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual grouped MLA native operand boundary; per-request tile state is exported separately.',)))
    events=unique_by(bindings.events,'event_id')
    for role,ref in zip(bindings.read_views,bindings.input_ready,strict=True):
        if ref.node_id not in events:raise ValueError('Grouped MLA input readiness is missing')
        event=events[ref.node_id];parent=views[role];region=_grouped_mla_region(parent,role,valid)
        if stage=='main' and role in ('key','value','block_table'):
            ready=ref.kind is EventKind.LOGICAL_AVAILABLE and event.region==region
        elif stage=='combine' and len(counts)>1 and role!='lengths':
            ready=ref.kind is EventKind.INITIALIZED and event.region==region
        else:ready=event.region is not None and covers(event.region,region)
        if (not ready or event.kind is not ref.kind or ref.allow_skipped_read
                or ref.kind not in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE,EventKind.LOGICAL_AVAILABLE)
                or event.allocation_id!=parent.allocation_id or event.visibility_domain!=bindings.visibility_domain
                or event.view_id is not None and event.view_id!=parent.view_id):
            raise ValueError('Grouped MLA requires actual operand/request readiness')
    values=_grouped_mla_prediction_values(stage,kv_lens,splits,heads,value_width,views)
    return _grouped_mla_projected_demands(stage,kv_lens,splits,valid,scratch,views,values)


def _grouped_mla_projected_demands(stage,kv_lens,splits,valid_splits,scratch_width,views,values):
    requests = _grouped_mla_requests(stage,splits)
    read = sum(values[role] for role,op,_ in requests if op is Operation.READ)
    write = sum(values[role] for role,op,_ in requests if op is Operation.WRITE)
    full_partial = sum(prod(views[role].shape)*views[role].element_bytes for role in ('partial_output','partial_lse'))
    if stage == 'main':
        padding = sum(-(-kv//32)*32-kv for kv in kv_lens)*(576+512)*2*4
        l2 = values['query']+values['key']+values['value']+padding+full_partial
        compulsory = values['query']//splits+values['key']//4
        dram = compulsory+full_partial//4*scratch_width
        priced_write = full_partial
    else:
        partial = values['partial_output']+values['partial_lse']
        # Newly admitted mixed-count combines previously used the unbound
        # FP32-LSE price. Preserve that envelope while publishing real BF16
        # LSE stores; uniform sources keep their already corrected price.
        legacy_width=views['output'].shape[1:]!=(64,512)
        priced_lse=grouped_mla_lse_write_projection(values['lse'])*(2 if valid_splits is None or legacy_width else 1)
        priced_write = values['output']+priced_lse
        l2,dram = full_partial+priced_write,full_partial//4*scratch_width+priced_write
        compulsory = partial
    return {'stage':stage,'kv_lens':kv_lens,'splits':splits,'valid_splits':valid_splits,
        'valid_splits_by_request':tuple(grouped_mla_valid_split_count(kv,splits) for kv in kv_lens),'read_bytes':read,'write_bytes':write,
        'metadata_read_bytes':values['lengths']+(values['block_table'] if stage == 'main' else 0),'priced_write_bytes':priced_write,
        'compulsory_input_bytes':compulsory,'l2_projection_bytes':l2,'dram_projection_bytes':min(l2,dram),
        # The pre-existing Roofline template charges head_dim channels for
        # each Q/K/V role. Retain that nominal projection separately from
        # actual 576-wide Q/K and the shared leading 512-wide V parent.
        'legacy_nominal_compulsory_input_bytes':(
            prod(views['query'].shape)*views['query'].element_bytes
            *views['value'].shape[-1]//views['query'].shape[-1]
            +2*prod(views['value'].shape)*views['value'].element_bytes) if stage=='main' else partial}


def standalone_grouped_mla_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_grouped_mla_source(kernel):
        raise NotImplementedError('Grouped MLA source requires its declared Q1/head/capture ABI')
    batch,splits = kernel.shape['batch'],kernel.launch.tiling.num_kv_splits
    stage = 'main' if kernel.launch.algorithm == 'triton_grouped_mla_stage1' else 'combine'
    valid_splits = grouped_mla_valid_split_count(kernel.attributes['kv_lens'][0],splits)
    requests = _grouped_mla_requests(stage,splits)
    read_roles=_grouped_mla_read_roles(stage,kernel.attributes['kv_lens'],splits)
    allocations,views,events,ready = {},[],[],[]
    for role,parent,shape,dtype,width,strides,offset,capacity in _grouped_mla_binding_layouts(kernel.attributes['kv_lens'],splits,stage,
            kernel.shape['num_query_heads'],kernel.shape['head_dim']):
        identity = f'{binding_id}.{parent}'
        if parent not in allocations:
            allocation = MemoryAllocation(identity,'logical-local-device','logical-native-tensor',capacity,
                AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
            allocations[parent] = allocation
            events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        view = TensorView(role,identity,offset,shape,strides,dtype,width)
        views.append(view)
        if role in read_roles:
            region = _grouped_mla_region(view,role,valid_splits)
            kind = EventKind.LOGICAL_AVAILABLE if isinstance(region,Unmapped) else EventKind.INITIALIZED
            event = ProgramEvent(f'{binding_id}.{role}.available',kind,identity,region,'triton-mla-visible',
                (EventRef(allocations[parent].allocated_event,EventKind.ALLOCATED),))
            events.append(event); ready.append(EventRef(event.event_id,kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations.values()),tuple(views),tuple(events),
        read_roles,
        tuple(role for role,op,_ in requests if op is Operation.WRITE),tuple(ready),0,'offline','triton-mla-visible')


def _grouped_mla_public_operands(kernel,bindings):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_grouped_mla_source(kernel):
        raise NotImplementedError('Grouped MLA graph/other head ABI remains source-unbound')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('Grouped MLA requires typed query/cache/packed partial bindings')
    stage = 'main' if kernel.launch.algorithm == 'triton_grouped_mla_stage1' else 'combine'
    batch,splits = kernel.shape['batch'],kernel.launch.tiling.num_kv_splits
    heads,value_width=kernel.shape['num_query_heads'],kernel.shape['head_dim']
    valid_splits = grouped_mla_valid_split_count(kernel.attributes['kv_lens'][0],splits)
    if kernel.launch.block != (128,1,1) or kernel.launch.grid != ((batch,4,splits) if stage == 'main' else (batch,heads,1)):
        raise ValueError('Grouped MLA source grid/block disagrees with retained wrapper')
    if stage == 'main' and tuple(kernel.launch.tiling.tile_shape[key] for key in ('query_heads','kv','qk_head','value_head')) != (16,32,576,512):
        raise ValueError('Grouped MLA source requires retained H16/KV32/576/512 tiling')
    requests = _grouped_mla_requests(stage,splits)
    read_roles=_grouped_mla_read_roles(stage,kernel.attributes['kv_lens'],splits)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (
            read_roles,
            tuple(role for role,op,_ in requests if op is Operation.WRITE),len(read_roles)):
        raise ValueError('Grouped MLA source requires all actual metadata and payload/LSE roles')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    parents = {}
    layouts = _grouped_mla_binding_layouts(kernel.attributes['kv_lens'],splits,stage,heads,value_width)
    if set(views) != {layout[0] for layout in layouts}:
        raise ValueError('Grouped MLA typed roles mismatch')
    for role,parent,shape,dtype,width,strides,offset,_ in layouts:
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes,view.offset_bytes) != (shape,dtype,width,strides,offset):
            raise ValueError('Grouped MLA packed view shape/dtype/stride/offset mismatch')
        if parent in parents and parents[parent] != view.allocation_id:
            raise ValueError('Grouped MLA cache halves and partial payload/LSE must retain their actual shared parent')
        parents[parent] = view.allocation_id
    if len(set(parents.values())) != len(parents):
        raise ValueError('Grouped MLA unrelated native parents cannot alias')
    return stage,splits,heads,value_width,views


def build_grouped_mla_source(kernel,bindings,granularity, *, program_id):
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise NotImplementedError('Grouped MLA issued page/metadata mapping remains pending')
    stage,splits,heads,value_width,views=_grouped_mla_public_operands(kernel,bindings)
    header = ('triton_grouped_mla',stage,','.join(map(str,kernel.attributes['kv_lens'])),str(splits),
        str(int(kernel.launch.parameters['split_scratch_dram_bytes_per_element'])))
    if (heads,value_width)!=(64,512):header=(*header,str(heads),str(value_width))
    scope = AccessScope(f'{program_id}.interface',header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-triton-mla-interface')
    selected,families = _grouped_mla_tile_recipe(header,views,dict(zip(bindings.read_views,bindings.input_ready,strict=True)),scope,bindings.visibility_domain)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'H16/KV32 main panels and per-head valid-split combine retain known packed partial addresses and unknown page mappings.',
            'Query/length reads include empty split CTAs; only valid splits read cache/map values or publish/read partial output/LSE.',
            'Selected INT32 page-table lane reads may repeat pages; their count is not distinct native capacity or traffic.',
            'Length and page-table service remain explicitly unpriced; existing cache/padding/scratch projections are preserved.',
            'Full native metadata allocation mapping, actor issue schedule, scratch and wrapper lifetimes remain partial.')),
        bindings.allocations,(*bindings.views,*selected),(scope,),bindings.events,(),families)
    grouped_mla_tile_demands(program)
    return program


def grouped_mla_lse_write_projection(native_bytes):
    """Price the actual caller-provided final-LSE storage width."""
    return native_bytes


def grouped_mla_source_demands(program):
    if not program.tile_programs:
        raise ValueError('Grouped MLA requires its actual compressed tile source')
    return grouped_mla_tile_demands(program)


@contextmanager
def grouped_mla_source_work(kernel,work):
    token = _GROUPED_MLA_WORK.set((kernel,work))
    try:
        yield
    finally:
        _GROUPED_MLA_WORK.reset(token)


def current_grouped_mla_source_work(kernel):
    context = copy_context()
    if _GROUPED_MLA_WORK in context:
        bound,work = context[_GROUPED_MLA_WORK]
        if bound != kernel:
            raise ValueError('Grouped MLA source crossed a kernel boundary')
        return work
    from src.kernel.memory_access_builder import _prediction_bindings
    binding,_=_prediction_bindings(kernel,None)
    if binding is None:binding=standalone_grouped_mla_bindings(kernel,call_id='mla',binding_id='mla')
    return grouped_mla_prediction_work(kernel,binding)


def flash_mla_metadata_operand_layouts(kernel):
    """Legacy get_mla_metadata INT32 sequence lengths and public outputs."""
    if kernel.dtype != 'int32' or kernel.variant != 'flash_mla_metadata':
        raise ValueError('FlashMLA metadata requires its legacy INT32 ABI')
    batch,partitions = kernel.shape['batch'],kernel.shape['scheduler_partitions']
    return ((((batch,),'int32',4),),
            (((partitions,8),'int32',4),((batch+1,),'int32',4)))


def _flash_mla_metadata_fields(scheduler,scope):
    return TensorView(f'{scope.scope_id}.scheduler_fields',scheduler.allocation_id,scheduler.offset_bytes,
        (scheduler.shape[0],5),scheduler.strides_bytes,scheduler.dtype,scheduler.element_bytes)


def _flash_mla_metadata_tiles(views,scope,ready,visibility,evidence, *, native_fields):
    lengths,scheduler,splits = (views[role] for role in ('input0','output0','output1'))
    dependency = (TileDependency('length',EventKind.READ_READY,('batch',)),)
    return TileProgram(scope.scope_id,scope,(TileAxis('batch',lengths.shape[0],1),
        TileAxis('partition',scheduler.shape[0],1),TileAxis('split_slot',splits.shape[0],1)),(
        TileAccessTemplate('length',lengths.view_id,Operation.READ,('batch',),('batch',),(ready,),(),evidence),
        TileAccessTemplate('scheduler',_flash_mla_metadata_fields(scheduler,scope).view_id if native_fields else scheduler.view_id,Operation.WRITE,('partition',),('partition',None),(),dependency,evidence),
        TileAccessTemplate('split',splits.view_id,Operation.WRITE,('split_slot',),('split_slot',),(),dependency,evidence)),
        visibility,evidence)


def build_flash_mla_metadata_tiles(kernel,bindings,scope,evidence, *, program_id):
    if kernel.hardware_id=='metax_c500':
        from src.kernel.memory_metax_mla_metadata import build_metax_metadata_source
        return build_metax_metadata_source(kernel,bindings,scope,evidence,program_id=program_id)
    from dataclasses import replace
    scope = replace(scope,scope_id=program_id,agent='logical-flash-mla-metadata')
    native_fields=kernel.hardware_id=='metax_c500'
    family = _flash_mla_metadata_tiles(unique_by(bindings.views,'view_id'),scope,
        bindings.input_ready[0],bindings.visibility_domain,evidence,native_fields=native_fields)
    selected=(_flash_mla_metadata_fields(unique_by(bindings.views,'view_id')['output0'],scope),) if native_fields else ()
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,True,(
            ('Retained XCORE1000 metadata bitcode stores five INT32 scheduler fields in each eight-INT32 storage record; padding lanes5..7 are unwritten.' if native_fields else
             'Public scheduler interface writes eight INT32 fields per record; C500-only native field evidence is not applied to other hardware.'),
            'Scheduler values and precise input-to-output membership are unknown; public outputs retain the complete input closure without cross-output ordering.',
            'Native scheduler scratch, padding, issue multiplicity and consumer aliases remain unmodeled.')),
        bindings.allocations,(*bindings.views,*selected),(),bindings.events,(),(family,))
    flash_mla_metadata_source_demands(program)
    return program


def flash_mla_metadata_tile_demands(program):
    if program.hardware_id=='metax_c500':
        from src.kernel.memory_metax_mla_metadata import metax_metadata_source_demands
        return metax_metadata_source_demands(program)
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs) != 1:
        raise ValueError('FlashMLA metadata requires its unique compressed INT32 tile source')
    family, = program.tile_programs
    views = unique_by(program.views,'view_id')
    native_fields=program.hardware_id=='metax_c500'
    fields=_flash_mla_metadata_fields(views['output0'],family.scope) if native_fields else views['output0']
    expected_views={'input0','output0','output1',fields.view_id}
    if set(views) != expected_views or views[fields.view_id]!=fields:
        raise ValueError('FlashMLA metadata complete INT32 tensor ABI requires its real five-field writer view')
    lengths,scheduler,splits = (views[role] for role in ('input0','output0','output1'))
    if len(lengths.shape) != 1 or len(scheduler.shape) != 2:
        raise ValueError('FlashMLA metadata input/scheduler rank mismatch')
    batch,partitions = lengths.shape[0],scheduler.shape[0]
    expected = (((batch,),(4,)),((partitions,8),(32,4)),((batch+1,),(4,)))
    for view,(shape,strides) in zip((lengths,scheduler,splits),expected,strict=True):
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes) != (shape,strides,'int32',4):
            raise ValueError('FlashMLA metadata must conserve its complete INT32 tensor ABI')
    if len({view.allocation_id for view in (lengths,scheduler,splits)}) != 3:
        raise ValueError('FlashMLA metadata buffers must have distinct parents')
    templates = unique_by(family.accesses,'access_id')
    if ('length' not in templates or len(templates['length'].input_ready) != 1
            or family.scope.op_ids != ('flash_mla_metadata',) or family.scope.agent != 'logical-flash-mla-metadata'):
        raise ValueError('FlashMLA metadata requires actual sequence-length READ_READY')
    expected = _flash_mla_metadata_tiles(views,family.scope,templates['length'].input_ready[0],
        family.visibility_domain,family.evidence_refs,native_fields=native_fields)
    if family != expected:raise ValueError('FlashMLA metadata outputs require actual sequence-length READ_READY closure')
    return {'batch':batch,'partitions':partitions,'read_bytes':requested_bytes(lengths.region()),
        'write_bytes':requested_bytes(fields.region())+requested_bytes(splits.region()),
        'priced_write_bytes':requested_bytes(scheduler.region())+requested_bytes(splits.region()),
        'scheduler_written_fields':5 if native_fields else 8,'scheduler_storage_fields':8}


def flash_mla_metadata_source_demands(program):
    """Consume the unique typed tile source without inventing scheduler values."""
    return flash_mla_metadata_tile_demands(program)


def dense_attention_operand_shapes(kernel):
    batch, query, kv, dimension, heads, kv_heads = (kernel.shape[name] for name in
        ("batch", "q_len", "kv_len", "head_dim", "num_query_heads", "num_kv_heads"))
    return ((batch, query, heads, dimension), (batch, kv, kv_heads, dimension),
            (batch, kv, kv_heads, dimension), (batch, query, heads, dimension))


def dense_strides(shape, width):
    strides, size = [], width
    for dimension in reversed(shape):
        strides.append(size)
        size *= dimension
    return tuple(reversed(strides))


def dense_attention_operand_layouts(kernel):
    names = ('query','key','value','output')
    layouts = tuple((name,shape,kernel.dtype,kernel.shape['dtype_bytes'])
        for name,shape in zip(names,dense_attention_operand_shapes(kernel),strict=True))
    return (*layouts,('lse',(kernel.shape['batch'],kernel.shape['num_query_heads'],kernel.shape['q_len']),'fp32',4),
        *(_dense_control_layouts(kernel) if _dense_paged_source(kernel) else ()))


def _metax_kvcache_source(kernel):
    if (kernel.hardware_id!='metax_c500' or kernel.variant!='decode' or kernel.shape['q_len']!=1
            or kernel.launch.algorithm!='metax_flash_attention_trait_direct:decode'
            or kernel.attributes['kv_layout'] not in ('paged','paged_varlen')
            or kernel.launch.tiling.num_kv_splits!=1 or kernel.shape['head_dim']!=128):return False
    tile=kernel.launch.tiling.tile_shape
    return (tile['q']==16 and tile['kv']==16 and tile['head']==128
        and 'query_heads_per_workgroup' in tile and tile['query_heads_per_workgroup']==4
        and all(q==1 for q in kernel.attributes['q_lens']))


def _dense_control_layouts(kernel):
    controls=_fa2_paged_control_layouts(kernel.shape['batch'],kernel.shape['kv_len'])
    # with_kvcache receives lengths/table directly, without cumulative Q.
    return controls[:-1] if _metax_kvcache_source(kernel) else controls


def _dense_paged_source(kernel):
    return _metax_kvcache_source(kernel) or (kernel.hardware_id=='nvidia_a800' and kernel.variant in ('prefill','decode')
        and kernel.launch.algorithm=='vllm_flash_attn_splitkv' and kernel.launch.tiling.num_kv_splits==1
        and kernel.attributes['kv_layout'] in ('paged','paged_varlen','chunked_prefill'))


def _fa2_paged_control_layouts(batch,keys):
    """Public Varlen inputs: token-to-page choices, used K lengths and Q endpoints."""
    return (('block_table',(batch,keys),'int32',4),('cache_lengths',(batch,),'int32',4),('cu_query',(batch+1,),'int32',4))


def standalone_dense_attention_bindings(kernel, *, call_id, binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    allocations, views, events, ready = [], [], [], []
    names = ("query", "key", "value", "output", "lse")
    paged=_dense_paged_source(kernel)
    for index, (name, shape, dtype, width) in enumerate(dense_attention_operand_layouts(kernel)):
        identity = f"{binding_id}.{name}"
        allocation = MemoryAllocation(identity, "logical-local-device", "logical-cache-selection" if paged and name in ('key','value','block_table') else "logical-storage", None if paged and name in ('key','value','block_table') else prod(shape)*width,
            AddressLevel.OBJECT, None, None, f"{identity}.allocated", None)
        view = (LogicalSelectionView(name,identity,shape,dtype,width,'fa2.paged.block_table.valid_token_lookup',None)
            if paged and name=='block_table' else TensorView(name, identity, 0, shape, dense_strides(shape, width), dtype, width))
        allocations.append(allocation)
        views.append(view)
        events.append(ProgramEvent(allocation.allocated_event, EventKind.ALLOCATED, identity, None, None, ()))
        if index < 3 or name in ('block_table','cache_lengths','cu_query'):
            event_id = f"{identity}.initialized"
            unresolved=paged and name in ('key','value','block_table')
            kind=EventKind.LOGICAL_AVAILABLE if unresolved else EventKind.INITIALIZED
            region=(view.region() if name=='block_table' or not unresolved else Unmapped(shape,dtype,width,f'fa2.paged.{name}.declared_input'))
            events.append(ProgramEvent(event_id, kind, identity, region, "attention-visible",
                (EventRef(allocation.allocated_event, EventKind.ALLOCATED),)))
            ready.append(EventRef(event_id, kind))
    return ElementwiseBindings(binding_id, call_id, tuple(allocations), tuple(views), tuple(events),
        (*names[:3],*(tuple(role for role,*_ in _dense_control_layouts(kernel)) if paged else ())), names[3:], tuple(ready), 0, "offline", "attention-visible")


def _dense_attention_tile_recipe(views,input_ready,scope,causal,paged,visibility):
    program_id=scope.scope_id
    batch,query,heads,dimension=views['query'].shape
    keys,kv_heads=views['key'].shape[1:3];group=heads//kv_heads
    controls=tuple(role for role in ('cache_lengths','cu_query') if role in views) if paged else ()
    grouped=[];grouped_ids={}
    for role in ('query','output','lse',*(('cu_query',) if 'cu_query' in controls else ())):
        parent=views[role]
        if role in ('query','output'):
            shape=(batch,query,kv_heads,group,dimension)
            coordinates=(((0,1),),((1,1),),((2,group),(3,1)),((4,1),))
            if isinstance(parent,TensorView):
                bs,qs,hs,ds=parent.strides_bytes;strides=(bs,qs,hs*group,hs,ds)
        elif role=='lse':
            shape=(batch,kv_heads,group,query)
            coordinates=(((0,1),),((1,group),(2,1)),((3,1),))
            if isinstance(parent,TensorView):
                bs,hs,qs=parent.strides_bytes;strides=(bs,hs*group,hs,qs)
        else:
            shape=(batch,2);coordinates=(((0,1),(1,1)),);strides=(4,4)
        label=f'{role}.pairs' if role=='cu_query' else f'{role}.groups'
        if isinstance(parent,LogicalSelectionView):
            identity=f'{program_id}.{label}'
            mapping=tuple(LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(child,coefficient)
                for child,coefficient in terms)) for axis,terms in enumerate(coordinates))
            grouped.append(LogicalSelectionView(identity,parent.allocation_id,shape,parent.dtype,parent.element_bytes,
                f'{parent.selection_id}.{label}',parent.view_id,mapping))
        else:
            identity=label
            grouped.append(TensorView(identity,parent.allocation_id,parent.offset_bytes,shape,strides,parent.dtype,parent.element_bytes))
        grouped_ids[role]=identity
    evidence=('FA2 online softmax: an independent query row/GQA head group streams matching KV-head tokens and retains reduction state on chip.',
        'End-aligned causal predicate key < KV-Q+query+1 masks actual logical K/V reads; native launch tiles/issued padding/cache traffic remain separate.',
        'FP32 LSE is logsumexp(QK) and depends on Q/K reduction; output additionally consumes V. No fabricated global score/probability scratch or all-query barrier.')
    if paged:evidence+=('FA2 paged KV address selection reads an INT32 block-table value before the selected K/V payload. Logical lookup per valid token is not a native table issue count or unique entry extent; page size/values and physical cache addresses remain unknown.',)
    output_axes=('batch','kv_head','query');stream_axes=(*output_axes,'key')
    def scoped(role):
        return AccessScope(f'{program_id}.{role}',('flash_attention',role),scope.call_id,(),scope.rank,scope.stream,'dense-logical-query-kv')
    control_deps=tuple(TileDependency(role,EventKind.READ_READY,()) for role in controls)
    accesses=[TileAccessTemplate('query',grouped_ids['query'],Operation.READ,output_axes,('batch','query','kv_head',None,None),
        (input_ready[0],),control_deps,evidence,scope=scoped('query'))]
    if paged:
        accesses.append(TileAccessTemplate('block_table',views['block_table'].view_id,Operation.READ,stream_axes,('batch','key'),
            (input_ready[3],),control_deps,evidence,unmapped=True,scope=scoped('block_table')))
        accesses.append(TileAccessTemplate('cache_lengths',views['cache_lengths'].view_id,Operation.READ,output_axes,('batch',),(input_ready[4],),(),evidence,scope=scoped('cache_lengths')))
        if 'cu_query' in controls:
            accesses.append(TileAccessTemplate('cu_query',grouped_ids['cu_query'],Operation.READ,output_axes,('batch',None),(input_ready[5],),(),evidence,scope=scoped('cu_query')))
    for index,role in enumerate(('key','value'),1):
        accesses.append(TileAccessTemplate(role,views[role].view_id,Operation.READ,stream_axes,('batch','key','kv_head',None),
            (input_ready[index],),(TileDependency('block_table',EventKind.READ_READY,()),) if paged else (),evidence,unmapped=paged,scope=scoped(role)))
    qk=(TileDependency('query',EventKind.READ_READY,()),TileDependency('key',EventKind.READ_READY,('key',)))
    accesses.append(TileAccessTemplate('output',grouped_ids['output'],Operation.WRITE,output_axes,('batch','query','kv_head',None,None),(),
        (*qk,TileDependency('value',EventKind.READ_READY,('key',))),evidence,scope=scoped('output')))
    accesses.append(TileAccessTemplate('lse',grouped_ids['lse'],Operation.WRITE,output_axes,('batch','kv_head',None,'query'),(),
        qk,evidence,scope=scoped('lse')))
    if isinstance(views['query'],LogicalSelectionView):
        accesses=[replace(access,unmapped=True) for access in accesses]
    family=TileProgram(program_id,scope,(TileAxis('batch',batch,1),TileAxis('kv_head',kv_heads,1),
        TileAxis('query',query,1),TileAxis('key',keys,1)),tuple(accesses),visibility,evidence,
        causal_prefixes=(TileCausalPrefix('key','query'),) if causal else ())
    return tuple(grouped),family


def build_dense_attention_interface(kernel, hardware, bindings, granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings, ElementwiseBindings):
        raise TypeError("dense attention requires typed tensor operand bindings")
    paged=_dense_paged_source(kernel)
    read_roles=('query','key','value',*(tuple(role for role,*_ in _dense_control_layouts(kernel)) if paged else ()))
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (read_roles,('output','lse'),len(read_roles)):
        raise ValueError("dense attention requires Q/K/V input readiness and O/LSE outputs")
    if granularity not in (Granularity.OP, Granularity.KERNEL, Granularity.TILE):
        raise ValueError("unknown dense attention logical granularity")
    views = unique_by(bindings.views, "view_id")
    layouts=dense_attention_operand_layouts(kernel)
    operands = tuple(views[name] for name,_,_,_ in layouts)
    if set(views) != {name for name,_,_,_ in layouts}:
        raise ValueError('dense attention requires exactly its Q/K/V/O/LSE roles')
    for view, (role,shape,dtype,width) in zip(operands,layouts,strict=True):
        if (view.shape, view.dtype, view.element_bytes) != (shape, dtype, width):
            raise ValueError("dense attention external operand shape/storage mismatch")
        if role=='block_table':
            if not isinstance(view,LogicalSelectionView) or view.parent_view_id is not None or view.coordinate_map:
                raise ValueError('Paged FA2 lookup must retain unresolved table entries and capacity')
        elif not isinstance(view,TensorView) or view.strides_bytes != dense_strides(shape, width):
            raise NotImplementedError("dense attention replay projection requires BSHD tensor strides")
    if len({view.allocation_id for view in operands}) != len(layouts):
        raise NotImplementedError("dense attention alias needs an independently declared replay/lifetime contract")
    scope=AccessScope(program_id,('flash_attention','causal' if kernel.attributes['causal'] else 'noncausal',*(('paged_lookup','legacy_output_only') if paged else ())),bindings.call_id,(),bindings.rank,bindings.stream,'dense-online-softmax')
    if paged and (_fa2_is_ragged(kernel) or _metax_kvcache_source(kernel)):
        header=_dense_ragged_header(kernel)
        grouped,families=_dense_ragged_families(header,views,bindings.input_ready,replace(scope,op_ids=header),bindings.visibility_domain)
    else:
        grouped,family=_dense_attention_tile_recipe(views,bindings.input_ready,scope,kernel.attributes['causal'],paged,bindings.visibility_domain)
        families=(family,)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,not paged,(
            'Compressed query/GQA-group and KV-token semantic tiles; no native CTA or instruction issue trace claim.',
            'Causal masks preserve exact logical stream requests; compulsory footprint, native masked tile work and L2/DRAM projections remain distinct.',
            'FP32 BHQ LSE is explicit; RNG/internal implementation side effects, other attention ABIs and native scratch remain pending.')),
        bindings.allocations,(*bindings.views,*grouped),(),bindings.events,(),families)
    from src.kernel.memory_path_solver import dense_attention_interface_demands
    dense_attention_interface_demands(program)
    return program


def _dense_ragged_header(kernel):
    return ('metax_kvcache' if _metax_kvcache_source(kernel) else 'fa2_ragged_dense','causal' if kernel.attributes['causal'] else 'noncausal',
        *(str(kernel.shape[name]) for name in ('batch','q_len','kv_len','head_dim','num_query_heads','num_kv_heads')),
        kernel.dtype,str(kernel.shape['dtype_bytes']),','.join(map(str,kernel.attributes['q_lens'])),
        ','.join(map(str,kernel.attributes['kv_lens'])),'paged_lookup','legacy_output_only')


def _dense_ragged_geometry(header):
    if (len(header)!=14 or header[0] not in ('fa2_ragged_dense','metax_kvcache') or header[1] not in ('causal','noncausal')
            or header[-2:]!=('paged_lookup','legacy_output_only')):
        raise ValueError('Ragged dense attention requires exact request geometry')
    b,q,kv,d,h,hkv=map(int,header[2:8]);width=int(header[9])
    qs=tuple(map(int,header[10].split(',')));ks=tuple(map(int,header[11].split(',')))
    if (min(b,q,kv,d,h,hkv,width)<=0 or h%hkv or len(qs)!=b or len(ks)!=b
            or any(not 0<x<=q for x in qs) or any(not x<=y<=kv for x,y in zip(qs,ks,strict=True))):
        raise ValueError('Ragged dense request lengths exceed actual logical parents')
    if header[0]=='metax_kvcache' and (q!=1 or any(x!=1 for x in qs) or d!=128):
        raise ValueError('MetaX kvcache source requires its Q1/D128 logical ABI')
    return b,q,kv,d,h,hkv,width,qs,ks


def _dense_ragged_layouts(header, *, request=None):
    b,q,kv,d,h,hkv,width,qs,ks=_dense_ragged_geometry(header)
    if request is not None:b,q,kv=1,qs[request],ks[request]
    return (('query',(b,q,h,d),header[8],width),('key',(b,kv,hkv,d),header[8],width),
        ('value',(b,kv,hkv,d),header[8],width),('output',(b,q,h,d),header[8],width),
        ('lse',(b,h,q),'fp32',4),*(_fa2_paged_control_layouts(b,kv)[:-1] if header[0]=='metax_kvcache' else _fa2_paged_control_layouts(b,kv)))


def _dense_ragged_families(header,parents,input_ready,scope,visibility):
    b,*_=_dense_ragged_geometry(header);selected=[];families=[]
    for index in range(b):
        local_scope=replace(scope,scope_id=f'{scope.scope_id}.request{index}',op_ids=(*header[:-2],str(index),*header[-2:]))
        request_parents,local=_fa2_request_parents(parents,_dense_ragged_layouts(header,request=index),index,local_scope.scope_id)
        nested,family=_dense_attention_tile_recipe(local,input_ready,local_scope,header[1]=='causal',True,visibility)
        selected.extend((*request_parents,*nested));families.append(family)
    return tuple(selected),tuple(families)


def has_dense_ragged_analytical_work(kernel):
    return (has_dense_flash_attention_source(kernel) and _dense_paged_source(kernel)
        and (_fa2_is_ragged(kernel) or _metax_kvcache_source(kernel)))


def _dense_ragged_prediction_demand(header):
    """Sum the same request/GQA stream extents without materializing tiles."""
    b,q,kv,d,h,hkv,width,qs,ks=_dense_ragged_geometry(header)
    query=sum(qs)*h*d*width;key=sum(ks)*hkv*d*width
    pairs=sum(x*(y-x)+x*(x+1)//2 if header[1]=='causal' else x*y
        for x,y in zip(qs,ks,strict=True))
    return {'query':query,'key':key,'value':key,'output':query,'lse':sum(qs)*h*4,
        'block_table_logical_read_bytes':pairs*hkv*4,
        'control_metadata_logical_read_bytes':sum(qs)*hkv*(4 if header[0]=='metax_kvcache' else 12)}


def dense_ragged_prediction_work(kernel,hardware,bindings):
    """Check actual public storage/readiness, then evaluate the shared demand."""
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_access import covers,validate_storage_bindings
    if kernel.hardware_id!=hardware.hardware_id or not has_dense_ragged_analytical_work(kernel):
        raise ValueError('Ragged dense analytical work requires its existing source ABI')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('Ragged dense attention requires typed public operands')
    bindings.__post_init__()
    layouts=dense_attention_operand_layouts(kernel)
    reads=('query','key','value',*(role for role,*_ in _dense_control_layouts(kernel)))
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(reads,('output','lse'),len(reads)):
        raise ValueError('Ragged dense actual operand/readiness roles differ')
    views=unique_by(bindings.views,'view_id')
    if set(views)!={role for role,*_ in layouts} or len({v.allocation_id for v in views.values()})!=len(layouts):
        raise ValueError('Ragged dense parents must remain exact and distinct')
    for role,shape,dtype,width in layouts:
        view=views[role]
        if (view.shape,view.dtype,view.element_bytes)!=(shape,dtype,width):
            raise ValueError('Ragged dense parent storage mismatch')
        if role=='block_table':
            if not isinstance(view,LogicalSelectionView) or view.parent_view_id is not None or view.coordinate_map:
                raise ValueError('Ragged dense table must retain unknown storage')
        elif not isinstance(view,TensorView) or view.strides_bytes!=dense_strides(shape,width):
            raise ValueError('Ragged dense requires its declared logical parents')
    validate_storage_bindings(bindings.allocations,bindings.views,bindings.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual ragged attention public boundary; request tiles are exported separately.',)))
    events=unique_by(bindings.events,'event_id')
    for role,ref in zip(reads,bindings.input_ready,strict=True):
        if ref.node_id not in events:raise ValueError('Ragged dense input readiness is missing')
        event=events[ref.node_id];parent=views[role]
        logical=(role in ('key','value') and ref.kind is EventKind.LOGICAL_AVAILABLE
            and isinstance(event.region,Unmapped) and
            (event.region.logical_shape,event.region.dtype,event.region.element_bytes)==
            (parent.shape,parent.dtype,parent.element_bytes))
        if (event.kind is not ref.kind or ref.allow_skipped_read
                or ref.kind not in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE,EventKind.LOGICAL_AVAILABLE)
                or event.allocation_id!=parent.allocation_id or event.visibility_domain!=bindings.visibility_domain
                or event.view_id is not None and event.view_id!=parent.view_id
                or not (logical or event.region is not None and covers(event.region,parent.region()))):
            raise ValueError('Ragged dense requires its actual parent readiness')
    return _dense_ragged_prediction_demand(_dense_ragged_header(kernel))


def dense_ragged_tile_demands(program):
    program.__post_init__();first=program.tile_programs[0];ids=first.scope.op_ids
    header=(*ids[:12],*ids[-2:]);_dense_ragged_geometry(header)
    expected_hardware='metax_c500' if header[0]=='metax_kvcache' else 'nvidia_a800'
    if program.hardware_id!=expected_hardware or not first.scope.scope_id.endswith('.request0'):
        raise ValueError('Ragged dense source hardware/request identity differs')
    views=unique_by(program.views,'view_id');layouts=_dense_ragged_layouts(header)
    parents={role:views[role] for role,_,_,_ in layouts}
    if len({v.allocation_id for v in parents.values()})!=len(layouts):raise ValueError('Ragged dense parents must remain distinct')
    for role,shape,dtype,width in layouts:
        view=parents[role]
        if (view.shape,view.dtype,view.element_bytes)!=(shape,dtype,width):raise ValueError('Ragged dense parent storage mismatch')
        if role=='block_table':
            if not isinstance(view,LogicalSelectionView) or view.parent_view_id is not None or view.coordinate_map:
                raise ValueError('Ragged dense table must retain unknown storage')
        elif not isinstance(view,TensorView) or view.strides_bytes!=dense_strides(shape,width):
            raise ValueError('Ragged dense requires its declared logical parents')
    reads=('query','key','value','block_table','cache_lengths',*(('cu_query',) if header[0]!='metax_kvcache' else ()));ready={}
    for family in program.tile_programs:
        for access in family.accesses:
            if access.operation is not Operation.READ:continue
            role=access.access_id
            if role not in reads or len(access.input_ready)!=1 or role in ready and ready[role]!=access.input_ready[0]:
                raise ValueError('Ragged dense actual input availability differs')
            ready[role]=access.input_ready[0]
    scope=replace(first.scope,scope_id=first.scope.scope_id[:-9],op_ids=header)
    selected,expected=_dense_ragged_families(header,parents,tuple(ready[role] for role in reads),scope,first.visibility_domain)
    if (program.tile_programs!=expected or set(views)!={*parents,*(view.view_id for view in selected)}
            or any(views[v.view_id]!=v for v in selected)):
        raise ValueError('Ragged dense requires exact request maps and every actual READ_READY')
    events=unique_by(program.events,'event_id')
    for role,ref in ready.items():
        if events[ref.node_id].allocation_id!=parents[role].allocation_id:
            raise ValueError('Ragged dense readiness belongs to a foreign parent')
    return _dense_ragged_prediction_demand(header)


_FA2_SPLIT_WORK = ContextVar('fa2_split_source_work')


def has_fa2_split_source(kernel):
    if kernel.kind is not KernelKind.FLASH_ATTN or kernel.execution_support != 'supported' or kernel.launch is None:
        return False
    native = ((kernel.hardware_id == 'nvidia_a800' and (
            (kernel.launch.algorithm in ('flash_attn_splitkv','vllm_flash_attn_splitkv') and kernel.variant == 'decode')
            or (kernel.launch.algorithm == 'flash_attn_splitkv_combine' and kernel.variant in ('combine','split_combine'))))
        or (kernel.hardware_id == 'metax_c500' and (
            (kernel.launch.algorithm == 'metax_flash_attention_trait_direct:decode' and kernel.variant == 'decode')
            or (kernel.launch.algorithm in ('metax_flash_attn_lut','metax_flash_attention_generic')
                and kernel.variant in ('combine','split_combine')))))
    return (kernel.kind is KernelKind.FLASH_ATTN and native
        and kernel.execution_support == 'supported' and kernel.dtype == 'bf16'
        and (kernel.attributes['kv_layout'] == 'contiguous' or kernel.hardware_id=='nvidia_a800'
            and kernel.attributes['kv_layout'] in ('paged','paged_varlen','chunked_prefill'))
        and (kernel.shape['q_len']==1 or kernel.hardware_id=='nvidia_a800' and kernel.variant in ('combine','split_combine'))
        and kernel.shape['head_dim'] in ((64,128,192) if kernel.hardware_id == 'metax_c500' else (64,128,256))
        and kernel.launch.tiling is not None and kernel.launch.tiling.num_kv_splits is not None
        and 1 < kernel.launch.tiling.num_kv_splits <= 128
        and 'model_input_q_lens' not in kernel.attributes and 'model_input_kv_lens' not in kernel.attributes
        and ((all(length == kernel.shape['q_len'] for length in kernel.attributes['q_lens'])
            and all(length == kernel.shape['kv_len'] for length in kernel.attributes['kv_lens']))
            or _fa2_ragged_admitted(kernel)))


def _fa2_is_ragged(kernel):
    return (any(length!=kernel.shape['q_len'] for length in kernel.attributes['q_lens'])
        or any(length!=kernel.shape['kv_len'] for length in kernel.attributes['kv_lens']))


def _fa2_ragged_admitted(kernel):
    # A800 Varlen supplies exact logical request lengths. Captured/model-input
    # geometry remains a different domain and is rejected by the caller guard.
    q_lens,kv_lens=kernel.attributes['q_lens'],kernel.attributes['kv_lens']
    return ((kernel.hardware_id=='nvidia_a800' or _metax_kvcache_source(kernel)) and kernel.attributes['kv_layout'] in ('paged','paged_varlen','chunked_prefill')
        and len(q_lens)==len(kv_lens)==kernel.shape['batch']
        and all(type(q) is int and 0<q<=kernel.shape['q_len'] for q in q_lens)
        and all(type(kv) is int and q<=kv<=kernel.shape['kv_len'] for q,kv in zip(q_lens,kv_lens,strict=True)))


def _fa2_ragged_lengths(header):
    if len(header)!=15 or header[11:13]!=('nvidia_a800_paged','ragged'):
        raise ValueError('FA2 ragged source requires its actual Varlen geometry')
    qs=tuple(map(int,header[13].split(',')));ks=tuple(map(int,header[14].split(',')))
    b,q,kv=map(int,header[2:5])
    if (len(qs)!=b or len(ks)!=b or any(not 0<x<=q for x in qs)
            or any(not x<=y<=kv for x,y in zip(qs,ks,strict=True))):
        raise ValueError('FA2 ragged request lengths exceed the logical parents')
    return qs,ks


def _fa2_request_parents(parents,layouts,index,scope):
    """Select request membership; no packed offset or physical stride claim."""
    selected=[];local={}
    for role,shape,dtype,width in layouts:
        parent=parents[role]
        batch_axis=1 if role in ('partial_output','partial_lse') else 0
        coordinates=tuple(LogicalCoordinateBinding(axis,() if axis==batch_axis else (LogicalCoordinateTerm(axis,1),),
            index if axis==batch_axis else 0) for axis in range(len(shape)))
        # cu_query is B+1 endpoint storage: child B=1 still contains its two
        # consecutive endpoints, so its coordinate retains the child index.
        if role=='cu_query':
            coordinates=(LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),index),)
        view=LogicalSelectionView(f'{scope}.{role}.request',parent.allocation_id,shape,dtype,width,
            f'fa2.request.{index}.{role}',parent.view_id,coordinates)
        selected.append(view);local[role]=view
    return tuple(selected),local


def _fa2_split_families(header,parents,ready,scope,visibility):
    return _cached_fa2_split_families(header,tuple(parents.items()),
        tuple(sorted(ready.items())),scope,visibility)


@runtime_lru_cache(maxsize=8)
def _cached_fa2_split_families(header,parent_items,ready_items,scope,visibility):
    # Cache the complete actual invocation, rather than individual requests.
    # A ragged batch can exceed a small per-request cache before its first
    # consumer finishes. All parent geometry/identities and readiness remain
    # in this key; each public program still validates its actual bindings.
    parents,ready=dict(parent_items),dict(ready_items)
    if len(header)!=15:
        selected,family=_fa2_split_tile_recipe(header,parents,ready,scope,visibility)
        return selected,(family,)
    qs,ks=_fa2_ragged_lengths(header);selected=[];families=[]
    for index,(q,kv) in enumerate(zip(qs,ks,strict=True)):
        local_header=(*header[:2],'1',str(q),str(kv),*header[5:12])
        local_scope=replace(scope,scope_id=f'{scope.scope_id}.request{index}',op_ids=(*header,str(index)))
        request_parents,local=_fa2_request_parents(parents,_fa2_split_layouts(local_header),index,local_scope.scope_id)
        nested,family=_fa2_split_tile_recipe(local_header,local,ready,local_scope,visibility)
        selected.extend((*request_parents,*nested));families.append(family)
    return tuple(selected),tuple(families)


def _fa2_ragged_demands(program):
    from src.kernel.memory_access import _tile_template_requested_bytes
    header=program.tile_programs[0].scope.op_ids[:-1];qs,ks=_fa2_ragged_lengths(header)
    if program.hardware_id!='nvidia_a800':raise ValueError('FA2 ragged hardware differs')
    views=unique_by(program.views,'view_id');parents={role:views[role] for role,_,_,_ in _fa2_split_layouts(header)}
    _validate_fa2_split_views(header,parents)
    ready={}
    for family in program.tile_programs:
        for access in family.accesses:
            if access.operation is not Operation.READ:continue
            role=access.access_id.rsplit('.',1)[-1]
            if len(access.input_ready)!=1 or role in ready and ready[role]!=access.input_ready[0]:
                raise ValueError('FA2 ragged requires every actual input availability')
            ready[role]=access.input_ready[0]
    first=program.tile_programs[0].scope
    if not first.scope_id.endswith('.request0'):raise ValueError('FA2 ragged request source identity differs')
    scope=replace(first,scope_id=first.scope_id[:-9],op_ids=header)
    selected,expected=_fa2_split_families(header,parents,ready,scope,program.tile_programs[0].visibility_domain)
    if (program.tile_programs!=expected or set(views)!={*parents,*(view.view_id for view in selected)}
            or any(views[view.view_id]!=view for view in selected)):
        raise ValueError('FA2 ragged requires exact request maps and every actual payload/partial READ_READY')
    events=unique_by(program.events,'event_id')
    for role,ref in ready.items():
        if ref.kind is not EventKind.LOGICAL_AVAILABLE or events[ref.node_id].region!=_fa2_split_region(parents[role]):
            raise ValueError('FA2 ragged input availability must describe its actual parent')
    totals=[]
    for index,(family,q,kv) in enumerate(zip(program.tile_programs,qs,ks,strict=True)):
        values={role:0 for role in parents};axes=unique_by(family.axes,'axis_id')
        for access in family.accesses:
            view=views[access.view_id]
            while view.view_id not in parents:view=views[view.parent_view_id]
            values[view.view_id]+=_tile_template_requested_bytes(views,axes,family,access)
        local_header=(*header[:2],'1',str(q),str(kv),*header[5:12])
        if values!=_fa2_split_prediction_values(local_header):
            raise ValueError('FA2 ragged tile bytes differ from analytical request demand')
        totals.append(_fa2_split_projected_values(local_header,values))
    return _fa2_ragged_projected_values(header,totals)


def _fa2_ragged_projected_values(header,totals):
    result=dict(totals[0]);result['geometry']=tuple(map(int,header[2:8]))
    for key in result:
        if key in ('stage','geometry','splits','scratch_dram_width'):continue
        result[key]=None if totals[0][key] is None else sum(row[key] for row in totals)
    return result


def _fa2_split_header(kernel):
    stage = 'main' if kernel.variant == 'decode' else 'combine'
    shape = kernel.shape
    tile = kernel.launch.tiling.tile_shape
    header = ('fa2_split',stage,*(str(shape[key]) for key in ('batch','q_len','kv_len','head_dim','num_query_heads','num_kv_heads')),
        str(kernel.launch.tiling.num_kv_splits),str(tile['kv'] if 'kv' in tile else 1),
        str(kernel.launch.parameters['split_scratch_dram_bytes_per_element']))
    if kernel.hardware_id=='metax_c500':return (*header,'metax_c500')
    if kernel.attributes['kv_layout'] in ('paged','paged_varlen','chunked_prefill'):
        header=(*header,'nvidia_a800_paged')
        if _fa2_is_ragged(kernel):
            return (*header,'ragged',','.join(map(str,kernel.attributes['q_lens'])),','.join(map(str,kernel.attributes['kv_lens'])))
        return header
    return header


def _fa2_split_layouts(header):
    if len(header)==15:
        _fa2_ragged_lengths(header)
        header=header[:12]
    if (len(header) not in (11,12) or (len(header) == 12 and header[11] not in ('metax_c500','nvidia_a800_paged'))
            or header[0] != 'fa2_split' or header[1] not in ('main','combine')):
        raise ValueError('FA2 split requires its typed stage/geometry header')
    b,q,kv,d,h,hkv,s,tile = map(int,header[2:10])
    head_domain = (64,128,192) if header[-1]=='metax_c500' else (64,128,256)
    if (min(b,q,kv,d,h,hkv,s,tile) <= 0 or (q!=1 and (header[1]=='main' or header[-1]=='metax_c500'))
            or d not in head_domain or not 1 < s <= 128 or h%hkv):
        raise ValueError('FA2 split source outside its uniform Q1 BF16 domain')
    if float(header[10]) not in (0,2,4):
        raise ValueError('FA2 split scratch projection requires an explicit retained element width')
    partials = (('partial_output',(s,b,h,q,d),'fp32',4),('partial_lse',(s,b,h,q),'fp32',4))
    if header[1] == 'main':
        return (('query',(b,q,h,d),'bf16',2),('key',(b,kv,hkv,d),'bf16',2),
            ('value',(b,kv,hkv,d),'bf16',2),*partials,
            *(_fa2_paged_control_layouts(b,kv) if header[-1]=='nvidia_a800_paged' else ()))
    return (*partials,('output',(b,q,h,d),'bf16',2),('lse',(b,h,q),'fp32',4))


def _fa2_split_requests(header):
    if len(header)==15:
        _fa2_ragged_lengths(header)
        header=header[:12]
    b,q,kv,d,h,hkv,s,tile = map(int,header[2:10])
    if header[1] == 'main':
        tiles = -(-kv//tile)
        valid = -(-tiles//-(-tiles//s))
        # Empty splits branch before Q/K/V loads but still write 0/-inf
        # partial payloads. This is not the old Triton MLA no-write branch.
        return (('query',Operation.READ,valid),('key',Operation.READ,1),('value',Operation.READ,1),
            ('partial_output',Operation.WRITE,1),('partial_lse',Operation.WRITE,1),
            *((('block_table',Operation.READ,1),('cache_lengths',Operation.READ,1),('cu_query',Operation.READ,1))
              if header[-1]=='nvidia_a800_paged' else ()))
    # Native reducer publishes final LSE after the LSE reduction, before
    # reading/accumulating partial O. It does not depend on O payload.
    return (('partial_lse',Operation.READ,1),('lse',Operation.WRITE,1),
        ('partial_output',Operation.READ,1),('output',Operation.WRITE,1))


def _fa2_split_region(view):
    return Unmapped(view.shape,view.dtype,view.element_bytes,f'fa2_split.{view.view_id}.uniform_Q1_payload')


def _fa2_split_tile_recipe(header,parents,ready,scope,visibility):
    return _cached_fa2_split_tile_recipe(header,tuple(parents.items()),
        tuple(ready.items()),scope,visibility)


@runtime_lru_cache(maxsize=8)
def _cached_fa2_split_tile_recipe(header,parent_items,ready_items,scope,visibility):
    # A source is checked by several consumers within one prediction. Reuse
    # that exact immutable recipe, including every actual storage, actor and
    # readiness reference. Preserve parent ordering in the exported program.
    parents,ready=dict(parent_items),dict(ready_items)
    b,q,kv,d,h,hkv,s,tile = map(int,header[2:10]);gqa = h//hkv
    selected,roots = [],{}
    for role,parent in parents.items():
        region = _fa2_split_region(parent)
        identity = tuple(LogicalCoordinateBinding(axis,(LogicalCoordinateTerm(axis,1),)) for axis in range(len(parent.shape)))
        view = LogicalSelectionView(f'{scope.scope_id}.{role}.selection',parent.allocation_id,parent.shape,parent.dtype,
            parent.element_bytes,region.selection_id,parent.view_id,identity)
        selected.append(view);roots[role] = view
    evidence = ('Common FA2 split/combine source: KV blocks per split=ceil(ceil(KV/N)/S); valid tails are masked.',
        'A logical GQA work group uses one KV head and its associated query heads; native head replay/CTA issue scheduling is a separate projection.',
        'Empty main splits branch before Q/K/V loads but publish FP32 zero O and negative-infinity LSE.',
        'Combine publishes final LSE after LSE reduction and before partial-O reads; it cannot require O for LSE publication.',
        'Typed parent coordinate maps preserve known logical membership without claiming native addresses or capacities.')

    def mapping(*entries):
        return tuple(LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(child,coefficient)
            for child,coefficient in terms),offset) for axis,(terms,offset) in enumerate(entries))

    def child(label,role,shape,coordinates):
        root = roots[role]
        view = LogicalSelectionView(f'{scope.scope_id}.{label}.{role}',root.allocation_id,shape,root.dtype,root.element_bytes,
            f'{root.selection_id}.{label}',root.view_id,coordinates)
        selected.append(view);return view

    if header[1] == 'main':
        blocks = -(-kv//tile);width = -(-blocks//s)*tile
        full,tail = divmod(kv,width);valid = full+(1 if tail else 0)
        groups = ([] if not full else [('full',full,width,0)])
        if tail:groups.append(('tail',1,tail,full))
        if valid < s:groups.append(('empty',s-valid,0,valid))
        axes = [TileAxis('batch',b,1),TileAxis('kv_head',hkv,1)];accesses = []
        paged=header[-1]=='nvidia_a800_paged'
        if paged:
            axes.append(TileAxis('control_split',s,1))
            length=child('control','cache_lengths',(b,),mapping((((0,1),),0)))
            query_ends=child('control','cu_query',(b,2),mapping((((0,1),(1,1)),0)))
            control_loop=('control_split','batch','kv_head')
            accesses.extend((
                TileAccessTemplate('cache_lengths',length.view_id,Operation.READ,control_loop,('batch',),(ready['cache_lengths'],),(),evidence,unmapped=True),
                TileAccessTemplate('cu_query',query_ends.view_id,Operation.READ,control_loop,('batch',None),(ready['cu_query'],),(),evidence,unmapped=True)))
        for label,count,tokens,first in groups:
            split_axis = f'{label}.split';axes.append(TileAxis(split_axis,count,1))
            out_map = mapping((((0,1),),first),(((1,1),),0),(((2,gqa),(3,1)),0),((),0),(((4,1),),0))
            lse_map = out_map[:-1]
            output = child(label,'partial_output',(count,b,hkv,gqa,d),out_map)
            lse = child(label,'partial_lse',(count,b,hkv,gqa),lse_map)
            loop = (split_axis,'batch','kv_head')
            controls=(tuple(TileDependency(role,EventKind.READ_READY,(),(TileIndexBinding('control_split',
                (TileIndexTerm(split_axis,1),),offset=first),)) for role in ('cache_lengths','cu_query')) if paged else ())
            if label == 'empty':
                accesses.extend((
                    TileAccessTemplate('empty.partial_output',output.view_id,Operation.WRITE,loop,
                        (split_axis,'batch','kv_head',None,None),(),controls,evidence,unmapped=True,constant_value=ConstantValue.ZERO),
                    TileAccessTemplate('empty.partial_lse',lse.view_id,Operation.WRITE,loop,
                        (split_axis,'batch','kv_head',None),(),controls,evidence,unmapped=True,constant_value=ConstantValue.NEGATIVE_INFINITY)))
                continue
            token_axis = f'{label}.kv_panel';axes.append(TileAxis(token_axis,tokens,tile))
            query = child(label,'query',(b,hkv,gqa,d),mapping((((0,1),),0),((),0),(((1,gqa),(2,1)),0),(((3,1),),0)))
            accesses.append(TileAccessTemplate(f'{label}.query',query.view_id,Operation.READ,loop,
                ('batch','kv_head',None,None),(ready['query'],),controls,evidence,unmapped=True))
            if header[-1]=='nvidia_a800_paged':
                table=child(label,'block_table',(count,b,tokens),mapping((((1,1),),0),(((0,width),(2,1)),first*width)))
                accesses.append(TileAccessTemplate(f'{label}.block_table',table.view_id,Operation.READ,(*loop,token_axis),
                    (split_axis,'batch',token_axis),(ready['block_table'],),controls,evidence,unmapped=True))
            for role in ('key','value'):
                view = child(label,role,(count,b,hkv,tokens,d),mapping((((1,1),),0),(((0,width),(3,1)),first*width),
                    (((2,1),),0),(((4,1),),0)))
                accesses.append(TileAccessTemplate(f'{label}.{role}',view.view_id,Operation.READ,(*loop,token_axis),
                    (split_axis,'batch','kv_head',token_axis,None),(ready[role],),
                    (TileDependency(f'{label}.block_table',EventKind.READ_READY,()),) if header[-1]=='nvidia_a800_paged' else (),evidence,unmapped=True))
            deps = (TileDependency(f'{label}.query',EventKind.READ_READY,()),
                TileDependency(f'{label}.key',EventKind.READ_READY,(token_axis,)),
                TileDependency(f'{label}.value',EventKind.READ_READY,(token_axis,)))
            accesses.extend((
                TileAccessTemplate(f'{label}.partial_output',output.view_id,Operation.WRITE,loop,
                    (split_axis,'batch','kv_head',None,None),(),deps,evidence,unmapped=True),
                TileAccessTemplate(f'{label}.partial_lse',lse.view_id,Operation.WRITE,loop,
                    (split_axis,'batch','kv_head',None),(),deps,evidence,unmapped=True)))
    elif q!=1:
        axes=(TileAxis('split',s,1),TileAxis('batch',b,1),TileAxis('head',h,1),TileAxis('query',q,1))
        partial_lse=child('reduce','partial_lse',(s,b,h,q),mapping((((0,1),),0),(((1,1),),0),(((2,1),),0),(((3,1),),0)))
        partial_o=child('reduce','partial_output',(s,b,h,q,d),mapping((((0,1),),0),(((1,1),),0),(((2,1),),0),(((3,1),),0),(((4,1),),0)))
        lse=child('reduce','lse',(b,h,q),mapping((((0,1),),0),(((1,1),),0),(((2,1),),0)))
        output=child('reduce','output',(b,h,q,d),mapping((((0,1),),0),(((2,1),),0),(((1,1),),0),(((3,1),),0)))
        reduced=('batch','head','query');loaded=('split',*reduced)
        accesses=(
            TileAccessTemplate('partial_lse',partial_lse.view_id,Operation.READ,loaded,loaded,(ready['partial_lse'],),(),evidence,unmapped=True),
            TileAccessTemplate('lse',lse.view_id,Operation.WRITE,reduced,reduced,(),(TileDependency('partial_lse',EventKind.READ_READY,('split',)),),evidence,unmapped=True),
            TileAccessTemplate('partial_output',partial_o.view_id,Operation.READ,loaded,(*loaded,None),(ready['partial_output'],),
                (TileDependency('lse',EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True),
            TileAccessTemplate('output',output.view_id,Operation.WRITE,reduced,(*reduced,None),(),
                (TileDependency('partial_output',EventKind.READ_READY,('split',)),TileDependency('partial_lse',EventKind.READ_READY,('split',))),evidence,unmapped=True))
    else:
        axes = (TileAxis('split',s,1),TileAxis('batch',b,1),TileAxis('head',h,1))
        partial_lse = child('reduce','partial_lse',(s,b,h),mapping((((0,1),),0),(((1,1),),0),(((2,1),),0),((),0)))
        partial_o = child('reduce','partial_output',(s,b,h,d),mapping((((0,1),),0),(((1,1),),0),(((2,1),),0),((),0),(((3,1),),0)))
        lse = child('reduce','lse',(b,h),mapping((((0,1),),0),(((1,1),),0),((),0)))
        output = child('reduce','output',(b,h,d),mapping((((0,1),),0),((),0),(((1,1),),0),(((2,1),),0)))
        accesses = (
            TileAccessTemplate('partial_lse',partial_lse.view_id,Operation.READ,('split','batch','head'),('split','batch','head'),
                (ready['partial_lse'],),(),evidence,unmapped=True),
            TileAccessTemplate('lse',lse.view_id,Operation.WRITE,('batch','head'),('batch','head'),(),
                (TileDependency('partial_lse',EventKind.READ_READY,('split',)),),evidence,unmapped=True),
            TileAccessTemplate('partial_output',partial_o.view_id,Operation.READ,('split','batch','head'),('split','batch','head',None),
                (ready['partial_output'],),(TileDependency('lse',EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True),
            TileAccessTemplate('output',output.view_id,Operation.WRITE,('batch','head'),('batch','head',None),(),(
                TileDependency('partial_output',EventKind.READ_READY,('split',)),
                TileDependency('partial_lse',EventKind.READ_READY,('split',))),evidence,unmapped=True))
    return tuple(selected),TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),visibility,evidence)


def fa2_split_tile_demands(program):
    from src.kernel.memory_access import _tile_template_requested_bytes
    program.__post_init__()
    if program.tile_programs and len(program.tile_programs[0].scope.op_ids)==16:
        return _fa2_ragged_demands(program)
    if len(program.tile_programs) != 1:raise ValueError('FA2 split requires its actual compressed tile source')
    family, = program.tile_programs;header = family.scope.op_ids
    layouts = _fa2_split_layouts(header)
    if ((header[-1]=='nvidia_a800_paged' and program.hardware_id!='nvidia_a800')
            or (header[-1]=='metax_c500' and program.hardware_id!='metax_c500')):
        raise ValueError('FA2 split source hardware program identity differs')
    views = unique_by(program.views,'view_id')
    parents = {role:views[role] for role,_,_,_ in layouts}
    _validate_fa2_split_views(header,parents)
    ready = {}
    for access in family.accesses:
        if access.operation is not Operation.READ:continue
        role = access.access_id.rsplit('.',1)[-1]
        if len(access.input_ready) != 1:raise ValueError('FA2 split requires actual input/partial availability')
        if role in ready and ready[role] != access.input_ready[0]:raise ValueError('FA2 split must retain actual input availability')
        ready[role] = access.input_ready[0]
    selected,expected = _fa2_split_tile_recipe(header,parents,ready,family.scope,family.visibility_domain)
    if (family != expected or set(views) != {*parents,*(view.view_id for view in selected)}
            or any(views[view.view_id] != view for view in selected)):
        raise ValueError('FA2 split must preserve exact split maps, constant empty writes and every actual payload/partial READ_READY')
    events = unique_by(program.events,'event_id')
    for role,ref in ready.items():
        if ref.kind is not EventKind.LOGICAL_AVAILABLE or events[ref.node_id].region != _fa2_split_region(parents[role]):
            raise ValueError('FA2 split requires exact actual payload/partial selection availability')
    roots = {f'{family.scope.scope_id}.{role}.selection':role for role in parents}
    values = {role:0 for role in parents};axes = unique_by(family.axes,'axis_id')
    for access in family.accesses:
        view = views[access.view_id]
        while view.parent_view_id not in parents:view = views[view.parent_view_id]
        values[roots[view.view_id]] += _tile_template_requested_bytes(views,axes,family,access)
    if values!=_fa2_split_prediction_values(header):
        raise ValueError('FA2 split tile bytes differ from analytical request demand')
    return _fa2_split_projected_values(header,values)


def _fa2_split_prediction_values(header):
    """Exact public split demand, including stores by empty split actors."""
    layouts=_fa2_split_layouts(header)
    values={role:prod(shape)*width for role,shape,_,width in layouts}
    if header[1]=='main':
        b,q,kv,d,h,hkv,s,tile=map(int,header[2:10])
        values['query']*=_fa2_split_requests(header)[0][2]
        if header[-1]=='nvidia_a800_paged':
            values['block_table']=b*kv*hkv*4
            values['cache_lengths']=s*b*hkv*4
            values['cu_query']=s*b*hkv*2*4
    return values


def _fa2_split_prediction_demand(header):
    if len(header)!=15:
        return _fa2_split_projected_values(header,_fa2_split_prediction_values(header))
    qs,ks=_fa2_ragged_lengths(header)
    totals=[]
    for q,kv in zip(qs,ks,strict=True):
        local=(*header[:2],'1',str(q),str(kv),*header[5:12])
        totals.append(_fa2_split_projected_values(local,_fa2_split_prediction_values(local)))
    return _fa2_ragged_projected_values(header,totals)


def _fa2_split_projected_values(header,values):
    b,q,kv,d,h,hkv,s,tile = map(int,header[2:10])
    requests = _fa2_split_requests(header)
    read = sum(values[role] for role,op,_ in requests if op is Operation.READ)
    write = sum(values[role] for role,op,_ in requests if op is Operation.WRITE)
    if header[1] == 'main':
        valid = requests[0][2];query_unique,remainder = divmod(values['query'],valid)
        if remainder:raise ValueError('FA2 query replay cannot be projected without exact split multiplicity')
        kv_unique = values['key']+values['value'];partial = values['partial_output']+values['partial_lse']
    else:
        partial = values['partial_output']+values['partial_lse']
        # Retained ABI footprint fields for shared legacy price helpers. These
        # are not Q/K/V accesses in the combine's actual source.
        query_unique = b*q*h*d*2;kv_unique = b*kv*hkv*d*4
    return {'stage':header[1],'geometry':(b,q,kv,d,h,hkv),'splits':s,'read_bytes':read,'write_bytes':write,'priced_write_bytes':write,
        'query_unique_bytes':query_unique,'kv_unique_bytes':kv_unique,'compulsory_input_bytes':query_unique+kv_unique if header[1] == 'main' else partial,
        'partial_bytes':partial,'block_table_logical_read_bytes':values['block_table'] if 'block_table' in values else 0,
        'control_metadata_logical_read_bytes':sum(values[role] for role in ('cache_lengths','cu_query')) if 'cache_lengths' in values else 0,
        'scratch_dram_width':float(header[10]),'combine_l2_bytes':partial+write if header[1] == 'combine' else None,
        'combine_dram_bytes':partial/4*float(header[10])+write if header[1] == 'combine' else None}


def standalone_fa2_split_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_fa2_split_source(kernel):
        raise NotImplementedError('FA2 split source requires its uniform contiguous Q1 ABI')
    header = _fa2_split_header(kernel)
    allocations,views,events,reads,ready = [],[],[],[],[]
    requests = _fa2_split_requests(header)
    for role,shape,dtype,width in _fa2_split_layouts(header):
        identity = f'{binding_id}.{role}'
        allocation = MemoryAllocation(identity,'logical-local-device','logical-native-tensor',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = (LogicalSelectionView(role,identity,shape,dtype,width,f'fa2_split.{role}.uniform_Q1_payload',None)
            if role=='block_table' else TensorView(role,identity,0,shape,dense_strides(shape,width),dtype,width))
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
    by_role = unique_by(views,'view_id')
    for role,op,_ in requests:
        if op is not Operation.READ:
            continue
        view = by_role[role];identity = view.allocation_id
        event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
            _fa2_split_region(view),'fa2-visible',(EventRef(f'{identity}.allocated',EventKind.ALLOCATED),))
        events.append(event);reads.append(role);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),tuple(reads),
        tuple(role for role,op,_ in requests if op is Operation.WRITE),tuple(ready),0,'offline','fa2-visible')


def _fa2_split_public_operands(kernel,bindings):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_fa2_split_source(kernel):
        raise NotImplementedError('FA2 split native ABI/page/graph mapping remains unbound')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('FA2 split requires actual typed payload/partial bindings')
    header = _fa2_split_header(kernel)
    requests,layouts = _fa2_split_requests(header),_fa2_split_layouts(header)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (
            tuple(role for role,op,_ in requests if op is Operation.READ),
            tuple(role for role,op,_ in requests if op is Operation.WRITE),
            sum(op is Operation.READ for _,op,_ in requests)):
        raise ValueError('FA2 split requires every actual payload/partial input and stage output')
    views = unique_by(bindings.views,'view_id')
    _validate_fa2_split_views(header,views)
    return header,views


def fa2_split_prediction_work(kernel,bindings):
    from src.kernel.memory_access import validate_storage_bindings
    header,views=_fa2_split_public_operands(kernel,bindings)
    bindings.__post_init__()
    validate_storage_bindings(bindings.allocations,bindings.views,bindings.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual FA2 split public storage and readiness; request tile maps are explicit exports.',)))
    events=unique_by(bindings.events,'event_id')
    for role,ref in zip(bindings.read_views,bindings.input_ready,strict=True):
        if ref.node_id not in events:raise ValueError('FA2 split input readiness is missing')
        event=events[ref.node_id];parent=views[role]
        if (ref.kind is not EventKind.LOGICAL_AVAILABLE or event.kind is not ref.kind or ref.allow_skipped_read
                or event.allocation_id!=parent.allocation_id or event.visibility_domain!=bindings.visibility_domain
                or event.view_id is not None and event.view_id!=parent.view_id
                or event.region!=_fa2_split_region(parent)):
            raise ValueError('FA2 split requires actual payload/partial selection availability')
    return _fa2_split_prediction_demand(header)


def build_fa2_split_source(kernel,bindings,granularity, *, program_id):
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise NotImplementedError('FA2 split does not expose native issued/address order')
    header,views=_fa2_split_public_operands(kernel,bindings)
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-fa2-split-panels')
    selected,families = _fa2_split_families(header,views,dict(zip(bindings.read_views,bindings.input_ready,strict=True)),scope,bindings.visibility_domain)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Exact logical split membership, GQA groups and token tails are typed coordinate relations; native addresses/capacities remain unknown.',
            'Empty splits publish explicit FP32 zero/-inf without Q/K/V reads; combine publishes LSE before partial-O reads.',
            'Native allocation padding, metadata/RNG/local scratch and retained-binary byte identity remain partial; common source is a semantic reference.',
            'Existing head replay/cache/SMEM/compute projections remain separate from this logical source.')),
        bindings.allocations,(*bindings.views,*selected),(),bindings.events,(),families)
    fa2_split_tile_demands(program)
    return program


def _validate_fa2_split_views(header,views):
    layouts = _fa2_split_layouts(header)
    if set(views) != {row[0] for row in layouts} or len({v.allocation_id for v in views.values()}) != len(layouts):
        raise ValueError('FA2 split requires distinct actual payload/partial/final parents')
    for role,shape,dtype,width in layouts:
        v = views[role]
        if role=='block_table':
            if (not isinstance(v,LogicalSelectionView) or (v.shape,v.dtype,v.element_bytes)!=(shape,dtype,width)
                    or v.parent_view_id is not None or v.coordinate_map or v.region()!=_fa2_split_region(v)):
                raise ValueError('FA2 paged table requires exact unaddressed logical token selections')
            continue
        if (v.shape,v.dtype,v.element_bytes,v.strides_bytes,v.offset_bytes) != (shape,dtype,width,dense_strides(shape,width),0):
            raise ValueError('FA2 split storage dtype/shape/logical stride mismatch')


def fa2_split_source_demands(program):
    if not program.tile_programs:raise ValueError('FA2 split requires its actual compressed tile source')
    return fa2_split_tile_demands(program)


@contextmanager
def fa2_split_source_work(kernel,work):
    token = _FA2_SPLIT_WORK.set((kernel,work))
    try:
        yield
    finally:
        _FA2_SPLIT_WORK.reset(token)


def current_fa2_split_source_work(kernel):
    context = copy_context()
    if _FA2_SPLIT_WORK in context:
        current,work = context[_FA2_SPLIT_WORK]
        if current != kernel:
            raise ValueError('FA2 split source crossed its kernel boundary')
        return work
    from src.kernel.memory_access_builder import fa2_split_prediction_work as evaluate
    return evaluate(kernel)


_CANN_ATTENTION_WORK = ContextVar('cann_attention_source_work')


def has_cann_attention_source(kernel):
    return (kernel.kind is KernelKind.FLASH_ATTN and kernel.hardware_id == 'ascend_910b'
        and kernel.execution_support == 'supported' and kernel.dtype == 'bf16' and kernel.variant == 'decode'
        and kernel.launch.algorithm.startswith('cann_flash_attention_score:')
        and kernel.attributes['kv_layout'] in ('contiguous','paged','paged_varlen') and kernel.attributes['causal'] is False
        and kernel.shape['q_len'] == 1 and kernel.launch.tiling is not None
        and kernel.launch.tiling.num_kv_splits == 1
        and ('model_input_q_lens' not in kernel.attributes
            or (tuple(kernel.attributes['q_lens']),tuple(kernel.attributes['kv_lens']))
                == (tuple(kernel.attributes['model_input_q_lens']),tuple(kernel.attributes['model_input_kv_lens'])))
        and all(length == 1 for length in kernel.attributes['q_lens'])
        and len(kernel.attributes['q_lens']) == kernel.shape['batch']
        and len(kernel.attributes['kv_lens']) == kernel.shape['batch']
        and all(0 < length <= kernel.shape['kv_len'] for length in kernel.attributes['kv_lens'])
        and (kernel.attributes['kv_layout'] != 'contiguous'
            or all(length == kernel.shape['kv_len'] for length in kernel.attributes['kv_lens'])))


def _cann_attention_header(kernel,hardware):
    mechanism = hardware.fia_tiling_data_materialization_service
    payload,aiv_ratio,transfer = (('unbound',)*3 if mechanism is None else
        tuple(str(mechanism[key]) for key in ('payload_bytes','aiv_per_group','aiv_transfer_bytes')))
    header = ('cann_bsh_decode',*(str(kernel.shape[key]) for key in ('batch','kv_len','head_dim','num_query_heads','num_kv_heads')),
        payload,str(kernel.launch.grid_size),aiv_ratio,transfer)
    if kernel.attributes['kv_layout'] == 'contiguous':
        return header
    from src.kernel.launch_config.ascend_910b.flash_attn_launch import _CANN_FIA_PAGE_SIZE
    return (*header,kernel.attributes['kv_layout'],str(_CANN_FIA_PAGE_SIZE),
        ','.join(map(str,kernel.attributes['kv_lens'])))


def _cann_attention_paged_geometry(header):
    if len(header) == 10:
        return None
    if len(header) != 13 or header[10] not in ('paged','paged_varlen'):
        raise ValueError('CANN paged source requires explicit layout/page/request geometry')
    page = int(header[11])
    lengths = tuple(map(int,header[12].split(',')))
    if page <= 0 or len(lengths) != int(header[1]) or any(not 0 < length <= int(header[2]) for length in lengths):
        raise ValueError('CANN paged source has invalid request token selections')
    pages = tuple((length+page-1)//page for length in lengths)
    return lengths,pages


def _cann_attention_layouts(header):
    if len(header) not in (10,13) or header[0] != 'cann_bsh_decode':
        raise ValueError('CANN BSH decode requires public payload and argument geometry')
    batch,kv,dimension,heads,kv_heads = map(int,header[1:6])
    if min(batch,kv,dimension,heads,kv_heads,int(header[7])) <= 0 or heads%kv_heads:
        raise ValueError('CANN BSH decode requires positive integral GQA/actor geometry')
    arguments = ()
    if header[6] == 'unbound':
        if header[8:10] != ('unbound','unbound'):
            raise ValueError('CANN unknown argument cannot have invented actor-copy metadata')
    else:
        payload,aiv_ratio,transfer = (int(header[index]) for index in (6,8,9))
        if min(payload,aiv_ratio) <= 0 or transfer < payload:
            raise ValueError('CANN argument requires positive source payload/actor/copy geometry')
        arguments = (('tiling',(payload,)),)
    paged = _cann_attention_paged_geometry(header)
    cache_shape = (batch,kv,kv_heads*dimension) if paged is None else (sum(paged[0]),kv_heads,dimension)
    metadata = () if paged is None else (('block_table',(batch,max(paged[1]))),)
    return (*arguments,*metadata,('query',(batch,1,heads*dimension)),('key',cache_shape),
        ('value',cache_shape),('output',(batch,1,heads*dimension)))


def _cann_attention_requests(header):
    arguments = (() if header[6] == 'unbound' else
        (('tiling','tiling_aic',int(header[7])),('tiling','tiling_aiv',int(header[7])*int(header[8]))))
    metadata = () if _cann_attention_paged_geometry(header) is None else (('block_table','block_table',1),)
    return (*arguments,*metadata,*((role,role,1) for role in ('query','key','value','output')))


def _cann_attention_storage_type(role):
    if role == 'tiling':
        return 1,'uint8'
    if role == 'block_table':
        return 4,'int32'
    if role in ('query','key','value','output'):
        return 2,'bf16'
    raise ValueError('Unknown CANN attention storage role')


def _cann_attention_region(view,header):
    paged = _cann_attention_paged_geometry(header)
    if paged is not None and view.view_id in ('block_table','key','value'):
        _,pages = paged
        shape = (sum(pages),) if view.view_id == 'block_table' else view.shape
        selection = f'cann_fia.{view.view_id}.page{header[11]}.kv{header[12]}.row_width{max(pages)}.valid_only'
        return Unmapped(shape,view.dtype,view.element_bytes,selection)
    return Unmapped(view.shape,view.dtype,view.element_bytes,f'cann_fia.{view.view_id}.public_payload')


def standalone_cann_attention_bindings(kernel,hardware, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_attention_source(kernel):
        raise NotImplementedError('CANN attention public BSH decode source ABI is unbound')
    allocations,views,events,ready,read_roles = [],[],[],[],[]
    header = _cann_attention_header(kernel,hardware)
    layouts=_cann_attention_layouts(header)
    if header[6]!='unbound' and _has_cann_paged_gqa_source(kernel):
        layouts+=tuple((role,(kernel.shape['batch'],)) for role in ('actual_query','actual_key'))
    for role,shape in layouts:
        identity = f'{binding_id}.{role}'
        width,dtype = (8,'int64') if role in ('actual_query','actual_key') else _cann_attention_storage_type(role)
        allocation = MemoryAllocation(identity,'logical-local-device',
            'logical-native-launch-arguments' if role == 'tiling' else 'logical-native-tensor',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = TensorView(role,identity,0,shape,dense_strides(shape,width),dtype,width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role != 'output':
            availability=_cann_attention_region(view,header)
            if role=='block_table' and header[6]!='unbound' and _has_cann_paged_gqa_source(kernel):
                # Availability exposes the opaque input object, not initialized
                # values or requests for its padded cells. Tile selections
                # below retain only the valid per-request page coordinates.
                availability=Unmapped(view.shape,view.dtype,view.element_bytes,f'cann_fia.{role}.input_object')
            event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
                availability,'cann-attention-visible',(EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind));read_roles.append(role)
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        tuple(read_roles),('output',),tuple(ready),0,'offline','cann-attention-visible')


def _has_cann_paged_gqa_source(kernel):
    from src.kernel.launch_config import FiaCrossDomainLaunchProgram
    return (has_cann_attention_source(kernel) and kernel.attributes['kv_layout'] in ('paged','paged_varlen')
        and kernel.launch.algorithm in ('cann_flash_attention_score:103000000010222221','cann_flash_attention_score:103000000010622221')
        and kernel.shape['head_dim'] in (64,128) and isinstance(kernel.launch.source_program,FiaCrossDomainLaunchProgram)
        and kernel.shape['num_query_heads']//kernel.shape['num_kv_heads']<=128)


def _cann_bsh_native_ranges(kernel):
    """Admit exact ordinary BSH GQA head ranges from the selected source program."""
    from src.kernel.launch_config import FiaCrossDomainLaunchProgram
    if (not has_cann_attention_source(kernel) or kernel.attributes['kv_layout']!='contiguous'
            or 'model_input_q_lens' in kernel.attributes or 'model_input_kv_lens' in kernel.attributes
            or kernel.launch.algorithm not in ('cann_flash_attention_score:103000000010022221',
                'cann_flash_attention_score:103000000010422221')
            or kernel.shape['kv_len']%32 or kernel.shape['head_dim']%32
            or not isinstance(kernel.launch.source_program,FiaCrossDomainLaunchProgram)):
        return None
    group_rows=kernel.shape['num_query_heads']//kernel.shape['num_kv_heads']
    if group_rows>128 or group_rows>kernel.launch.parameters['cann_source_m_base_size']:return None
    block=min(kernel.shape['kv_len'],int(kernel.launch.parameters['cann_source_s2_base_size']))
    if block<=0 or block%32:raise ValueError('CANN BSH source requires its selected aligned KV block')
    stacks=(kernel.shape['kv_len']+block-1)//block;ranges=[];next_head=0
    for group in kernel.launch.source_program.aic_groups:
        count,remainder=divmod(len(group.tasks),stacks)
        if remainder or count==0:return None
        for ordinal,task in enumerate(group.tasks):
            stack=ordinal%stacks
            if (task.b_n2!=next_head+ordinal//stacks or task.g_s1!=0 or task.actual_m!=group_rows
                    or task.s2!=stack or task.actual_s2!=min(block,kernel.shape['kv_len']-stack*block)
                    or task.is_head_first!=(stack==0) or task.is_head_last!=(stack==stacks-1)
                    or task.is_s2_split_core):return None
        ranges.append((next_head,count));next_head+=count
    if next_head!=kernel.shape['batch']*kernel.shape['num_kv_heads']:return None
    return block,tuple(ranges)


def _cann_bsh_native_recipe(header,parents,ready,scope,visibility, *, _fd_segment=None):
    """Compressed native QK/Vec1/PV/Vec2 stages, with independent AIV row halves."""
    if len(header)!=14 or header[0]!='cann_bsh_native_tiles':raise ValueError('CANN BSH native header mismatch')
    public=header[1:11];batch,kv,dimension,heads,kv_heads=map(int,public[1:6]);rows=heads//kv_heads
    block=int(header[11]);ranges=tuple(tuple(map(int,item.split(':'))) for item in header[12].split(','))
    lanes=int(header[13])
    if _fd_segment is None and (len(public)!=10 or kv%32 or dimension%32 or block<=0 or block%32 or block>kv or not 0<rows<=128
            or not ranges or len(ranges)!=int(public[7]) or public[6]=='unbound' or int(public[8])!=2 or lanes not in (1,8)
            or sum(count for _,count in ranges)!=batch*kv_heads
            or any(count<=0 or start!=sum(value for _,value in ranges[:index]) for index,(start,count) in enumerate(ranges))):
        raise ValueError('CANN BSH native task/actor geometry mismatch')
    if _fd_segment is not None:
        _,first,count,start,stacks,_=_fd_segment
        ranges=((first,count),)
        kv=min(kv,(start+stacks)*block)-start*block
    allocations,selected,events,families=[],[],[],[]
    evidence=('CANN8.5.1 fused_infer_attention_score_v3.cpp:1507/1539 selects BF16 ordinary non-FlashDecode BSH GQA.',
        'fia_block_cube_nonquant_gqa.h:950-985 reuses the complete Q L1 snapshot across the KV stacks of one head task.',
        'fia_block_vec_nonquant.h:433/446/612/623/669: score/probability/partial-O GM handoffs; previous accumulator reload only after first stack and GM store only before last.',
        'SetMSplitInfo: M<=16 runs entirely on AIV0; larger M partitions at a 16-row boundary. Both AIV argument reads remain independent.',
        'InitBuffers initializes 512 FP32 default max/sum lanes on each AIV; SoftmaxFlashV2 consumes them on the first stack. BRC uses eight stored FP32 lanes per row.',
        'Logical versions do not resolve native ring2 addresses, Fixpipe padding, framework buffers or physical capacity.')
    def coord(axis,terms=(),offset=0,divisor=1,modulus=None):
        return LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(index,value) for index,value in terms),offset,divisor,modulus)
    def child(label,role,shape,mapping):
        parent=parents[role]
        if _fd_segment is not None and role in ('key','value'):
            mapping=tuple(replace(binding,offset=binding.offset+_fd_segment[3]*block) if binding.axis==1 else binding for binding in mapping)
        view=LogicalSelectionView(f'{scope.scope_id}.{label}',parent.allocation_id,shape,parent.dtype,parent.element_bytes,
            f'cann_bsh.{label}',parent.view_id if _fd_segment is not None else role,mapping);selected.append(view);return view
    def scratch(label,shape,dtype,width,domain):
        identity=f'{scope.scope_id}.{label}'
        allocation=MemoryAllocation(identity,'logical-local-device',domain,None,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=LogicalSelectionView(identity,identity,shape,dtype,width,f'cann_bsh.{label}',None,())
        allocations.append(allocation);selected.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()));return view
    stacks=(kv+block-1)//block;middle_count=max(0,stacks-2)
    phases=(('first',0,1,min(block,kv)),*((('middle',1,middle_count,block),) if middle_count else ()),
        *((('last',stacks-1,1,kv-(stacks-1)*block),) if stacks>1 else ()))
    for core,(first,count) in enumerate(ranges):
        accesses=[];axes=[TileAxis('task',count,1)]
        if middle_count:axes.append(TileAxis('middle',middle_count,1))
        prefix=f'core{core}'
        def add(name,view,operation,iteration,view_axes,deps=(),inputs=()):
            accesses.append(TileAccessTemplate(name,view.view_id,operation,iteration,view_axes,inputs,deps,evidence,unmapped=True))
        def dep(name,kind=EventKind.READ_READY,bindings=()):return TileDependency(name,kind,(),bindings)
        # ClearWorkspaceImpl's per-AIC KFC message area. Constants are bound
        # by fia_workspace_ring_geometry_cann_8_5_1.json, independently of
        # operator workload timings. This is a separate framework primitive.
        clear=scratch(f'{prefix}.kfc_clear',(2*64*128*2+2*64,),'uint8',1,'logical-framework-gm')
        accesses.append(TileAccessTemplate('framework_clear',clear.view_id,Operation.WRITE,(),(None,),(),(),evidence,
            unmapped=True,constant_value=ConstantValue.ZERO))
        tiling=parents['tiling']
        add('argument_aic',tiling,Operation.READ,(),(None,),inputs=(ready['tiling'],))
        for part in (0,1):add(f'argument_aiv{part}',tiling,Operation.READ,(),(None,),inputs=(ready['tiling'],))
        defaults={}
        for part in (0,1):
            for state in ('max','sum'):
                identity=f'{scope.scope_id}.{prefix}.v{part}.default_{state}'
                allocation=MemoryAllocation(identity,'logical-local-device','logical-ub-state',2048,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
                view=TensorView(identity,identity,0,(512,),(4,),'fp32',4)
                allocations.append(allocation);selected.append(view)
                events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
                initializer_scope=replace(scope,scope_id=f'{identity}.initializer',op_ids=('Duplicate','SOFTMAX_MIN_NUM' if state=='max' else '0.0'),
                    agent=f'logical-cann-bsh-aiv{part}-initializer')
                accesses.append(TileAccessTemplate(f'v{part}.default_{state}',view.view_id,Operation.WRITE,(),(None,),(),
                    (dep(f'argument_aiv{part}'),),evidence,scope=initializer_scope,constant_value=ConstantValue.ZERO if state=='sum' else None))
                defaults[part,state]=view
        query=child(f'{prefix}.query','query',(count,rows,dimension),(
            coord(0,((0,1),),first,kv_heads),coord(1),
            coord(2,((0,rows*dimension),(1,dimension),(2,1)),first*rows*dimension,modulus=heads*dimension)))
        add('query',query,Operation.READ,('task',),('task',None,None),(dep('argument_aic'),),(ready['query'],))
        shared={}
        for phase,stack,length,columns in phases:
            middle=phase=='middle';iteration=('task','middle') if middle else ('task',)
            mapping=(coord(0,((0,1),),first,kv_heads),
                coord(1,((1,block),(2,1)) if middle else ((1,1),),stack*block),
                coord(2,((0,dimension),(3,1)) if middle else ((0,dimension),(2,1)),first*dimension,modulus=kv_heads*dimension))
            shape=(count,length,columns,dimension) if middle else (count,columns,dimension)
            for role in ('key','value'):
                view=child(f'{prefix}.{phase}.{role}',role,shape,mapping)
                add(f'{phase}.{role}',view,Operation.READ,iteration,('task','middle',None,None) if middle else ('task',None,None),
                    (dep('argument_aic'),),(ready[role],))
                shared[phase,role]=dep(f'{phase}.{role}')
        first_rows=rows if rows<=16 else ((rows+15)//16+1)//2*16
        for part,(start,part_rows) in enumerate(((0,first_rows),(first_rows,rows-first_rows))):
            if not part_rows:continue
            label=f'{prefix}.v{part}';pfx=f'v{part}'
            axes.append(TileAxis(f'row{part}',part_rows,1));row=f'row{part}';rv=('task',row)
            output=child(f'{label}.output','output',(count,part_rows,dimension),(
                coord(0,((0,1),),first,kv_heads),coord(1),
                coord(2,((0,rows*dimension),(1,dimension),(2,1)),first*rows*dimension+start*dimension,modulus=heads*dimension)))
            maximum=scratch(f'{label}.max',(count,part_rows,lanes),'fp32',4,'logical-ub-state')
            total=scratch(f'{label}.sum',(count,part_rows,lanes),'fp32',4,'logical-ub-state');state_rv=(*rv,None)
            accumulator=scratch(f'{label}.accumulator',(count,part_rows,dimension),'fp32',4,'logical-gm-workspace') if stacks>1 else None
            previous=None
            for phase,stack,length,columns in phases:
                middle=phase=='middle';last=phase==('first' if stacks==1 else 'last');p=f'{pfx}.{phase}'
                iteration=('task','middle',row) if middle else ('task',row)
                pv=('task','middle',row,None) if middle else (*rv,None)
                shape=(count,length,part_rows,columns) if middle else (count,part_rows,columns)
                scores=scratch(f'{label}.{phase}.scores',shape,'fp32',4,'logical-gm-workspace')
                probability=scratch(f'{label}.{phase}.probability',shape,'bf16',2,'logical-gm-workspace')
                add(f'{p}.scores_write',scores,Operation.WRITE,iteration,pv,
                    (dep('query'),shared[phase,'key'],dep('framework_clear',EventKind.WRITE_VISIBLE)))
                add(f'{p}.scores_read',scores,Operation.READ,iteration,pv,
                    (dep(f'{p}.scores_write',EventKind.WRITE_VISIBLE),dep(f'argument_aiv{part}')))
                binding=(TileIndexBinding('middle',(),middle_count-1),) if previous=='middle' else ()
                for name,view in (('max',maximum),('sum',total)):
                    if phase=='first':
                        parent=defaults[part,name]
                        subview=TensorView(f'{parent.view_id}.active',parent.allocation_id,0,(part_rows,lanes),(lanes*4,4),'fp32',4)
                        selected.append(subview)
                        accesses.append(TileAccessTemplate(f'{p}.{name}_read',subview.view_id,Operation.READ,('task',row),(row,None),(),
                            (dep(f'{pfx}.default_{name}',EventKind.WRITE_VISIBLE),),evidence))
                    else:
                        carry=(TileDependency(f'{pfx}.middle.{name}_write',EventKind.WRITE_VISIBLE,(),
                            carry_axis='middle',initial_access_id=f'{pfx}.first.{name}_write'),) if middle else (
                                dep(f'{pfx}.{previous}.{name}_write',EventKind.WRITE_VISIBLE,binding),)
                        add(f'{p}.{name}_read',view,Operation.READ,iteration,state_rv,carry)
                score_deps=(dep(f'{p}.scores_read'),)
                add(f'{p}.max_write',maximum,Operation.WRITE,iteration,state_rv,(*score_deps,dep(f'{p}.max_read')))
                dmdeps=()
                dm=scratch(f'{label}.{phase}.dm',(count,length,part_rows,lanes) if middle else (count,part_rows,lanes),'fp32',4,'logical-ub-state')
                dmv=('task','middle',row,None) if middle else state_rv
                add(f'{p}.dm_write',dm,Operation.WRITE,iteration,dmv,(dep(f'{p}.max_read'),dep(f'{p}.max_write',EventKind.WRITE_VISIBLE)))
                if phase!='first':
                    add(f'{p}.dm_read',dm,Operation.READ,iteration,dmv,(dep(f'{p}.dm_write',EventKind.WRITE_VISIBLE),));dmdeps=(dep(f'{p}.dm_read'),)
                add(f'{p}.probability_write',probability,Operation.WRITE,iteration,pv,(*score_deps,dep(f'{p}.max_write',EventKind.WRITE_VISIBLE)))
                add(f'{p}.probability_read',probability,Operation.READ,iteration,pv,
                    (dep(f'{p}.probability_write',EventKind.WRITE_VISIBLE),dep('argument_aic')))
                add(f'{p}.sum_write',total,Operation.WRITE,iteration,state_rv,
                    (*score_deps,dep(f'{p}.max_write',EventKind.WRITE_VISIBLE),*dmdeps,dep(f'{p}.sum_read')))
                tmp=scratch(f'{label}.{phase}.partial',(count,length,part_rows,dimension) if middle else (count,part_rows,dimension),
                    'fp32',4,'logical-gm-workspace')
                add(f'{p}.partial_write',tmp,Operation.WRITE,iteration,pv,(dep(f'{p}.probability_read'),shared[phase,'value']))
                add(f'{p}.partial_read',tmp,Operation.READ,iteration,pv,(dep(f'{p}.partial_write',EventKind.WRITE_VISIBLE),dep(f'argument_aiv{part}')))
                accdeps=(dep(f'{p}.partial_read'),*dmdeps)
                if phase!='first':
                    carry=(TileDependency(f'{pfx}.middle.accumulator_write',EventKind.WRITE_VISIBLE,(),
                        carry_axis='middle',initial_access_id=f'{pfx}.first.accumulator_write'),) if middle else (
                            dep(f'{pfx}.{previous}.accumulator_write',EventKind.WRITE_VISIBLE,binding),)
                    add(f'{p}.accumulator_read',accumulator,Operation.READ,iteration,(*rv,None),carry);accdeps+=(dep(f'{p}.accumulator_read'),)
                if last:
                    add(f'{p}.normalization_sum_read',total,Operation.READ,iteration,state_rv,(dep(f'{p}.sum_write',EventKind.WRITE_VISIBLE),))
                    add(f'{pfx}.output',output,Operation.WRITE,('task',row),(*rv,None),(*accdeps,dep(f'{p}.normalization_sum_read')))
                else:add(f'{p}.accumulator_write',accumulator,Operation.WRITE,iteration,(*rv,None),accdeps)
                previous=phase
        family_scope=replace(scope,scope_id=f'{scope.scope_id}.{prefix}',agent='logical-cann-bsh-stack-stream')
        families.append(TileProgram(family_scope.scope_id,family_scope,tuple(axes),tuple(accesses),visibility,evidence))
    return tuple(allocations),tuple(selected),tuple(events),tuple(families)


def _build_cann_bsh_native(kernel,hardware,bindings,header,parents,ready, *, program_id):
    native=_cann_bsh_native_ranges(kernel)
    if native is None:raise ValueError('CANN BSH native source requires exact ordinary task ranges')
    block,ranges=native
    native_header=('cann_bsh_native_tiles',*header,str(block),','.join(f'{start}:{count}' for start,count in ranges),
        '8' if kernel.launch.algorithm.endswith('103000000010422221') else '1')
    scope=AccessScope(program_id,native_header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-bsh-stack-stream')
    allocations,selected,events,families=_cann_bsh_native_recipe(native_header,parents,ready,scope,bindings.visibility_domain)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Native ordinary BSH GQA logical stages and actual AIC/AIV argument actors; GM and UB state domains are distinct.',
            'Native ring2 placement, Fixpipe issued padding/transactions and framework allocation/lifetimes remain unresolved.',
            'Original lower-bound/cache/pipe prices remain explicit projections; logical GM replays are not silently charged as DRAM.')),
        (*bindings.allocations,*allocations),(*bindings.views,*selected),(scope,),(*bindings.events,*events),(),families)
    _cann_bsh_native_demands(program,expected_kernel=kernel)
    return program


def _cann_bsh_native_demands(program, *, expected_kernel=None):
    from src.kernel.memory_access import tile_template_requested_bytes
    program.__post_init__()
    if len(program.scopes)!=1 or program.accesses or not program.tile_programs:raise ValueError('CANN BSH native source requires its unique compressed stream')
    scope,=program.scopes;header=scope.op_ids;public=header[1:11]
    views=unique_by(program.views,'view_id');roles=tuple(role for role,_ in _cann_attention_layouts(public));parents={role:views[role] for role in roles}
    _validate_cann_attention_views(public,parents)
    if expected_kernel is not None:
        native=_cann_bsh_native_ranges(expected_kernel)
        if (native is None or program.hardware_id!=expected_kernel.hardware_id
                or tuple(map(int,public[1:6]))!=tuple(expected_kernel.shape[name] for name in ('batch','kv_len','head_dim','num_query_heads','num_kv_heads'))
                or header[11:]!=(str(native[0]),','.join(f'{start}:{count}' for start,count in native[1]),
                    '8' if expected_kernel.launch.algorithm.endswith('103000000010422221') else '1')):
            raise ValueError('CANN BSH native source differs from its actual kernel')
    ready={}
    for access in program.tile_programs[0].accesses:
        if access.input_ready:
            view=views[access.view_id]
            while isinstance(view,LogicalSelectionView) and view.parent_view_id is not None:view=views[view.parent_view_id]
            ready[view.view_id]=access.input_ready[0]
    allocations,selected,events,expected=_cann_bsh_native_recipe(header,parents,ready,scope,program.tile_programs[0].visibility_domain)
    if (program.tile_programs!=expected or set(views)!={*roles,*(view.view_id for view in selected)}
            or any(views[view.view_id]!=view for view in selected) or any(item not in program.allocations for item in allocations)
            or any(item not in program.events for item in events)):
        raise ValueError('CANN BSH stream lost actual stage/argument/state readiness or parent mapping')
    totals={};gm_read=gm_write=ub_read=ub_write=0;domains={item.allocation_id:item.memory_domain for item in program.allocations}
    for family in program.tile_programs:
        for access in family.accesses:
            amount=tile_template_requested_bytes(program,family,access);name=access.access_id
            totals[name]=amount+(totals[name] if name in totals else 0);domain=domains[views[access.view_id].allocation_id]
            if domain=='logical-gm-workspace':
                if access.operation is Operation.READ:gm_read+=amount
                else:gm_write+=amount
            elif domain=='logical-ub-state':
                if access.operation is Operation.READ:ub_read+=amount
                else:ub_write+=amount
    values={role:prod(parents[role].shape)*parents[role].element_bytes for role in roles}
    payload=sum(values[role] for role in ('query','key','value'));argument=sum(totals[name] for name in ('argument_aic','argument_aiv0','argument_aiv1'))
    scores=sum(amount for name,amount in totals.items() if name.endswith('.scores_write'))//4
    return {'geometry':tuple(map(int,public[1:6])),'query_unique_bytes':values['query'],
        'kv_unique_bytes':values['key']+values['value'],'kv_interface_bytes':values['key']+values['value'],
        'metadata_read_bytes':0,'payload_read_bytes':payload,'read_bytes':payload+argument,'write_bytes':values['output'],
        'argument_payload_bytes':int(public[6]),'argument_aiv_transfer_bytes':int(public[9]),
        'argument_aic_materializations':int(public[7]),'argument_aiv_materializations':int(public[7])*int(public[8]),'argument_read_bytes':argument,
        'native_workspace_stream_bound':True,'gm_workspace_read_bytes':gm_read,'gm_workspace_write_bytes':gm_write,
        'ub_state_read_bytes':ub_read,'ub_state_write_bytes':ub_write,'totals':totals,
        'priced_score_elements':scores,'priced_output_elements':values['output']//parents['output'].element_bytes,
        # Retained aggregate pricing counted one Q row per MM2 partial. Keep
        # that explicit projection separate from the true reused Q requests.
        'priced_task_query_l2_bytes':sum(amount for name,amount in totals.items() if name.endswith('.partial_write'))//2,
        'key_requested_bytes':sum(amount for name,amount in totals.items() if name.endswith('.key')),
        'value_requested_bytes':sum(amount for name,amount in totals.items() if name.endswith('.value')),
        'framework_clear_write_bytes':totals['framework_clear'],
        'native_ring_alias_lifetime_bound':False}


def _cann_bsh_fd_geometry(kernel):
    """Exact mixed full-head/partial-head intervals in the selected main program."""
    from src.kernel.launch_config import FiaCrossDomainLaunchProgram
    if (not has_cann_attention_source(kernel) or kernel.attributes['kv_layout']!='contiguous'
            or kernel.launch.algorithm!='cann_flash_attention_score:103000000010522221'
            or 'model_input_q_lens' in kernel.attributes or 'model_input_kv_lens' in kernel.attributes
            or kernel.shape['kv_len']%32 or kernel.shape['head_dim']%32
            or not isinstance(kernel.launch.source_program,FiaCrossDomainLaunchProgram)):
        return None
    rows=kernel.shape['num_query_heads']//kernel.shape['num_kv_heads']
    if not 0<rows<=128 or rows>kernel.launch.parameters['cann_source_m_base_size']:return None
    block=int(kernel.launch.parameters['cann_source_s2_base_size']);kv=kernel.shape['kv_len']
    if block<=0 or block%32 or block>kv:return None
    stacks=(kv+block-1)//block;segments=[];cursor=0
    for group in kernel.launch.source_program.aic_groups:
        tasks=group.tasks;offset=0;local=[]
        while offset<len(tasks):
            first=tasks[offset];head=first.b_n2;start=first.s2;count=0;split=first.is_s2_split_core
            while offset<len(tasks) and tasks[offset].b_n2==head:
                task=tasks[offset];stack=cursor%stacks
                if (task.b_n2!=cursor//stacks or task.s2!=stack or task.g_s1!=0 or task.actual_m!=rows
                        or task.actual_s2!=min(block,kv-stack*block) or task.is_s2_split_core!=split
                        or task.is_head_first!=(count==0)
                        or task.is_head_last!=(offset==len(tasks)-1 or tasks[offset+1].b_n2!=head)):
                    return None
                count+=1;offset+=1;cursor+=1
            if not split and (start!=0 or count!=stacks):return None
            segment=(group.aic_group,head,1,start,count,int(split))
            if (local and not split and local[-1][5]==0 and local[-1][1]+local[-1][2]==head):
                previous=local[-1];local[-1]=(*previous[:2],previous[2]+1,*previous[3:])
            else:local.append(segment)
        segments.extend(local)
    if cursor!=kernel.shape['batch']*kernel.shape['num_kv_heads']*stacks:return None
    partial={}
    for core,head,count,start,length,split in segments:
        if split:
            if count!=1:return None
            if head not in partial:partial[head]=[]
            partial[head].append((start,length))
    if not partial:return None
    for parts in partial.values():
        position=0
        if len(parts)<2:return None
        for start,length in sorted(parts):
            if start!=position:return None
            position+=length
        if position!=stacks:return None
    return block,tuple(segments)


def _cann_bsh_fd_recipe(header,parents,ready,scope,visibility):
    """Reuse the native main stages, then publish real FD sum/max/O consumers."""
    if len(header)!=13 or header[0]!='cann_bsh_flashdecode_tiles':raise ValueError('CANN BSH FD header mismatch')
    public=header[1:11];batch,kv,dimension,heads,kv_heads=map(int,public[1:6]);rows=heads//kv_heads;block=int(header[11])
    segments=tuple(tuple(map(int,item.split(':'))) for item in header[12].split(';'))
    if not 0<rows<=128 or kv%32 or dimension%32 or block<=0 or block%32 or block>kv or int(public[8])!=2:
        raise ValueError('CANN BSH FD geometry mismatch')
    evidence=('CANN8.5.1 v3 key103000000010522221: BF16 BSH GQA with BRC and FlashDecode; mixed full and split heads are selected by actual source task flags.',
        'VecNonQuant:362/695/823 writes packed FP32 sum/max and individually normalized FP32 accumOut for split heads; full heads write BF16 O directly.',
        'FiaKernelNonQuant::FlashDecode:706 has a real SyncAll before terminal consumption; independent main head tasks remain unordered.',
        'VecFlashDecode:266/283/285 reads every actual partial O and eight FP32 sum/max lanes per row; :474 accumulates weighted partials without a fabricated zero initializer.',
        'Native rings, partial-pool offsets/capacity and terminal worker assignment remain unaddressed; no physical trace or additional final-LSE output is asserted.')
    allocations=[];views=[];events=[];axes={};accesses={};split_heads={};parts={}
    def retain(items,values,key):
        known={getattr(item,key):item for item in items}
        for item in values:
            identity=getattr(item,key)
            if identity in known:
                if known[identity]!=item:raise ValueError('CANN FD shared actor identity disagrees')
            else:items.append(item);known[identity]=item
    def scratch(label,shape,dtype='fp32',width=4,domain='logical-ub-state'):
        identity=f'{scope.scope_id}.{label}'
        allocation=MemoryAllocation(identity,'logical-local-device',domain,None,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=LogicalSelectionView(identity,identity,shape,dtype,width,f'cann_fd.{label}',None,())
        allocations.append(allocation);views.append(view);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()));return view
    common={'framework_clear','argument_aic','argument_aiv0','argument_aiv1','v0.default_max','v0.default_sum','v1.default_max','v1.default_sum'}
    cursor=0;core_seen=set();stacks=(kv+block-1)//block
    for index,segment in enumerate(segments):
        if len(segment)!=6:raise ValueError('CANN FD segment arity mismatch')
        core,head,count,start,length,split=segment
        if (not 0<=core<int(public[7]) or count<=0 or length<=0 or start<0 or start+length>stacks or split not in (0,1)
                or head*stacks+start!=cursor or split and count!=1 or not split and (start!=0 or length!=stacks)):
            raise ValueError('CANN FD head/KV coverage mismatch')
        cursor+=count*length;core_seen.add(core)
        private=dict(parents)
        if split:
            root=scratch(f's{index}.partial_parent',parents['output'].shape,domain='logical-gm-workspace')
            private['output']=root
            if head not in split_heads:split_heads[head]=[]
            split_heads[head].append(index)
        inner=('cann_bsh_native_tiles',*public,str(block),f'{head}:{count}','8')
        inner_scope=replace(scope,scope_id=f'{scope.scope_id}.s{index}',op_ids=inner)
        a,v,e,f=_cann_bsh_native_recipe(inner,private,ready,inner_scope,visibility,_fd_segment=segment)
        family,=f
        def axis(name):
            if split and name in ('task','row0','row1'):return f'fd{head}.{name}'
            return f's{index}.{name}'
        def access_id(name):return f'core{core}.{name}' if name in common else f's{index}.{name}'
        roots={access.view_id:f'{scope.scope_id}.core{core}.{access.access_id}' for access in family.accesses
            if access.access_id in common and access.operation is Operation.WRITE}
        allocation_map={view.allocation_id:roots[view.view_id] for view in v if view.view_id in roots}
        event_map={event.event_id:f'{allocation_map[event.allocation_id]}.allocated' for event in e if event.allocation_id in allocation_map}
        new_a=tuple(replace(item,allocation_id=allocation_map[item.allocation_id],allocated_event=event_map[item.allocated_event])
            if item.allocation_id in allocation_map else item for item in a)
        new_v=tuple(replace(item,view_id=roots[item.view_id] if item.view_id in roots else item.view_id,
            allocation_id=allocation_map[item.allocation_id] if item.allocation_id in allocation_map else item.allocation_id) for item in v)
        new_e=tuple(replace(item,event_id=event_map[item.event_id],allocation_id=allocation_map[item.allocation_id]) if item.event_id in event_map else item for item in e)
        retain(allocations,new_a,'allocation_id');retain(views,new_v,'view_id');retain(events,new_e,'event_id')
        for item in family.axes:
            changed=replace(item,axis_id=axis(item.axis_id))
            if changed.axis_id in axes and axes[changed.axis_id]!=changed:raise ValueError('CANN FD shared head row extent disagrees')
            axes[changed.axis_id]=changed
        lifted={}
        for access in family.accesses:
            dependencies=tuple(replace(dep,access_id=access_id(dep.access_id),reduction_axes=tuple(axis(name) for name in dep.reduction_axes),
                coordinate_bindings=tuple(replace(binding,axis_id=axis(binding.axis_id),terms=tuple(replace(term,axis_id=axis(term.axis_id))
                    for term in binding.terms)) for binding in dep.coordinate_bindings),carry_axis=None if dep.carry_axis is None else axis(dep.carry_axis),
                initial_access_id=None if dep.initial_access_id is None else access_id(dep.initial_access_id)) for dep in access.dependencies)
            item_scope=replace(scope,scope_id=f'{scope.scope_id}.{access_id(access.access_id)}.scope',
                op_ids=access.scope.op_ids if access.scope is not None else ('cann_fd_actor',str(core),access.access_id) if access.access_id in common else ('cann_fd_segment',*map(str,segment)),
                agent=f'logical-cann-fd-core{core}')
            changed=replace(access,access_id=access_id(access.access_id),view_id=roots[access.view_id] if access.view_id in roots else access.view_id,
                iteration_axes=tuple(axis(name) for name in access.iteration_axes),view_axes=tuple(None if name is None else axis(name) for name in access.view_axes),
                dependencies=dependencies,evidence_refs=evidence,scope=item_scope)
            if changed.access_id in accesses and accesses[changed.access_id]!=changed:raise ValueError('CANN FD actor materialization duplicated inconsistently')
            accesses[changed.access_id]=changed;lifted[access.access_id]=changed
        if split:
            last='first' if length==1 else 'last';task_axis=axis('task')
            first_rows=rows if rows<=16 else ((rows+15)//16+1)//2*16
            for part,(row_start,part_rows) in enumerate(((0,first_rows),(first_rows,rows-first_rows))):
                if not part_rows:continue
                row_axis=axis(f'row{part}');iteration=(task_axis,row_axis);view_axes=(task_axis,row_axis,None)
                part_output=next(item for item in new_v if item.view_id==lifted[f'v{part}.output'].view_id)
                part_state={}
                for state in ('sum','max'):
                    writer=lifted[f'v{part}.{last}.{state}_write'];ub=next(item for item in new_v if item.view_id==writer.view_id)
                    read_id=f's{index}.v{part}.fd_{state}_read';write_id=f's{index}.v{part}.fd_{state}_write'
                    gm=scratch(f's{index}.v{part}.fd_{state}',(1,part_rows,8),domain='logical-gm-workspace')
                    accesses[read_id]=TileAccessTemplate(read_id,ub.view_id,Operation.READ,iteration,view_axes,(),
                        (TileDependency(writer.access_id,EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True)
                    accesses[write_id]=TileAccessTemplate(write_id,gm.view_id,Operation.WRITE,iteration,view_axes,(),
                        (TileDependency(read_id,EventKind.READ_READY,()),),evidence,unmapped=True)
                    part_state[state]=(gm,write_id)
                parts[index,part]=(part_output,lifted[f'v{part}.output'].access_id,part_state,row_start,part_rows)
    if cursor!=batch*kv_heads*stacks or core_seen!=set(range(int(public[7]))) or not split_heads:
        raise ValueError('CANN FD source does not cover its actual head/actor domain')
    # The source has an actual kernel-wide SyncAll. Keep its producer set;
    # this barrier is not a scheduling policy invented for independent heads.
    barrier=tuple(TileDependency(access.access_id,EventKind.WRITE_VISIBLE if access.operation is Operation.WRITE else EventKind.READ_READY,
        access.iteration_axes) for access in accesses.values() if access.access_id.endswith(('.output','fd_sum_write','fd_max_write'))
            or '.argument_' in access.access_id or '.default_' in access.access_id and access.operation is Operation.WRITE)
    def terminal(head,indexes,part):
        if len(indexes)<2:raise ValueError('CANN FD split head needs multiple actual producers')
        if (indexes[0],part) not in parts:return
        rows=parts[indexes[0],part][4];row_start=parts[indexes[0],part][3]
        task_axis=f'fd{head}.task';row_axis=f'fd{head}.row{part}';iteration=(task_axis,row_axis);rv=(task_axis,row_axis,None);prefix=f'fd{head}.v{part}'
        def dep(name,kind=EventKind.READ_READY):return TileDependency(name,kind,())
        def add(name,view,operation,deps=(),value=None,symbol=None):
            access_scope=None if symbol is None else replace(scope,scope_id=f'{scope.scope_id}.{name}.scope',op_ids=('Duplicate',symbol),agent='logical-cann-fd-terminal')
            accesses[name]=TileAccessTemplate(name,view.view_id,operation,iteration,rv,(),deps,evidence,unmapped=True,scope=access_scope,constant_value=value)
        matrix_reads=[];lse_reads={state:[] for state in ('sum','max')}
        for index in indexes:
            output,writer,state,start,extent=parts[index,part];read=f'{prefix}.p{index}.output_read'
            if (start,extent)!=(row_start,rows):raise ValueError('CANN FD partial row ownership disagrees')
            add(read,output,Operation.READ,(dep(writer,EventKind.WRITE_VISIBLE),*(item for item in barrier if item.access_id!=writer)));matrix_reads.append(read)
            for name,(view,producer) in state.items():
                read=f'{prefix}.p{index}.{name}_read'
                add(read,view,Operation.READ,(dep(producer,EventKind.WRITE_VISIBLE),*(item for item in barrier if item.access_id!=producer)));lse_reads[name].append(read)
        maximum=scratch(f'{prefix}.maximum',(1,rows,8));total=scratch(f'{prefix}.sum',(1,rows,8))
        add(f'{prefix}.max_init',maximum,Operation.WRITE,barrier,symbol='-ConstInfo.FLOAT_MAX')
        add(f'{prefix}.max_init_read',maximum,Operation.READ,(dep(f'{prefix}.max_init',EventKind.WRITE_VISIBLE),))
        add(f'{prefix}.max_write',maximum,Operation.WRITE,(dep(f'{prefix}.max_init_read'),*(dep(name) for name in lse_reads['max'])))
        add(f'{prefix}.max_read',maximum,Operation.READ,(dep(f'{prefix}.max_write',EventKind.WRITE_VISIBLE),))
        add(f'{prefix}.sum_init',total,Operation.WRITE,barrier,value=ConstantValue.ZERO,symbol='0.0')
        add(f'{prefix}.sum_init_read',total,Operation.READ,(dep(f'{prefix}.sum_init',EventKind.WRITE_VISIBLE),))
        weights=[];weight_reads=[]
        for index,sum_read,max_read in zip(indexes,lse_reads['sum'],lse_reads['max'],strict=True):
            weight=scratch(f'{prefix}.p{index}.weight',(1,rows,8));weights.append(weight)
            name=f'{prefix}.p{index}.weight'
            add(f'{name}_write',weight,Operation.WRITE,(dep(sum_read),dep(max_read),dep(f'{prefix}.max_read')))
            add(f'{name}_read',weight,Operation.READ,(dep(f'{name}_write',EventKind.WRITE_VISIBLE),));weight_reads.append(f'{name}_read')
        add(f'{prefix}.sum_write',total,Operation.WRITE,(dep(f'{prefix}.sum_init_read'),*(dep(name) for name in weight_reads)))
        add(f'{prefix}.sum_read',total,Operation.READ,(dep(f'{prefix}.sum_write',EventKind.WRITE_VISIBLE),))
        accumulator=scratch(f'{prefix}.accumulator',(1,rows,dimension));previous=None
        for index,weight,weight_read,matrix_read in zip(indexes,weights,weight_reads,matrix_reads,strict=True):
            name=f'{prefix}.p{index}';normalized=scratch(f'{name}.normalized_weight',(1,rows,8))
            add(f'{name}.scale_write',normalized,Operation.WRITE,(dep(weight_read),dep(f'{prefix}.sum_read')))
            add(f'{name}.scale_read',normalized,Operation.READ,(dep(f'{name}.scale_write',EventKind.WRITE_VISIBLE),))
            dependencies=(dep(matrix_read),dep(f'{name}.scale_read'))
            if previous is not None:
                weighted=scratch(f'{name}.weighted',(1,rows,dimension))
                add(f'{name}.weighted_write',weighted,Operation.WRITE,dependencies)
                add(f'{name}.weighted_read',weighted,Operation.READ,(dep(f'{name}.weighted_write',EventKind.WRITE_VISIBLE),))
                add(f'{name}.acc_read',accumulator,Operation.READ,(dep(previous,EventKind.WRITE_VISIBLE),))
                dependencies=(dep(f'{name}.weighted_read'),dep(f'{name}.acc_read'))
            previous=f'{name}.acc_write';add(previous,accumulator,Operation.WRITE,dependencies)
        add(f'{prefix}.acc_read',accumulator,Operation.READ,(dep(previous,EventKind.WRITE_VISIBLE),))
        parent=parents['output'];mapping=(LogicalCoordinateBinding(0,(),head//kv_heads),LogicalCoordinateBinding(1,()),
            LogicalCoordinateBinding(2,(LogicalCoordinateTerm(1,dimension),LogicalCoordinateTerm(2,1)),(head%kv_heads*(heads//kv_heads)+row_start)*dimension))
        output=LogicalSelectionView(f'{scope.scope_id}.{prefix}.output',parent.allocation_id,(1,rows,dimension),parent.dtype,parent.element_bytes,
            f'cann_fd.head{head}.output',parent.view_id,mapping);views.append(output)
        add(f'{prefix}.output',output,Operation.WRITE,(dep(f'{prefix}.acc_read'),))
    for head,indexes in sorted(split_heads.items()):
        for part in (0,1):terminal(head,indexes,part)
    family=TileProgram(scope.scope_id,scope,tuple(axes.values()),tuple(accesses.values()),visibility,evidence)
    return tuple(allocations),tuple(views),tuple(events),(family,)


def _build_cann_bsh_fd(kernel,hardware,bindings,header,parents,ready, *, program_id):
    geometry=_cann_bsh_fd_geometry(kernel)
    if geometry is None:raise ValueError('CANN BSH FD requires its exact mixed source intervals')
    block,segments=geometry
    native=('cann_bsh_flashdecode_tiles',*header,str(block),';'.join(':'.join(map(str,item)) for item in segments))
    scope=AccessScope(program_id,native,bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-bsh-flashdecode')
    allocations,views,events,families=_cann_bsh_fd_recipe(native,parents,ready,scope,bindings.visibility_domain)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Actual full/partial head source intervals, normalized FP32 partials, packed FP32 sum/max and true pre-terminal SyncAll.',
            'No resolved native ring/pool addresses or terminal-worker placement. Disabled final LSE does not remove internal FD row states.',
            'Requested logical streams are separate from retained source-issued/cache/DRAM price projections.')),
        (*bindings.allocations,*allocations),(*bindings.views,*views),(scope,),(*bindings.events,*events),(),families)
    _cann_bsh_fd_demands(program,expected_kernel=kernel);return program


def _cann_bsh_fd_demands(program, *, expected_kernel=None):
    from src.kernel.memory_access import tile_template_requested_bytes
    program.__post_init__()
    if len(program.scopes)!=1 or program.accesses or len(program.tile_programs)!=1:raise ValueError('CANN BSH FD needs its unique compressed producer/terminal source')
    scope,=program.scopes;header=scope.op_ids;public=header[1:11]
    all_views=unique_by(program.views,'view_id');parents={role:all_views[role] for role,_ in _cann_attention_layouts(public)}
    _validate_cann_attention_views(public,parents)
    if expected_kernel is not None:
        geometry=_cann_bsh_fd_geometry(expected_kernel)
        if (geometry is None or program.hardware_id!=expected_kernel.hardware_id
                or tuple(map(int,public[1:6]))!=tuple(expected_kernel.shape[name] for name in ('batch','kv_len','head_dim','num_query_heads','num_kv_heads'))
                or header[11:]!=(str(geometry[0]),';'.join(':'.join(map(str,item)) for item in geometry[1]))):
            raise ValueError('CANN BSH FD source differs from its actual kernel')
    family,=program.tile_programs;ready={}
    for access in family.accesses:
        if access.input_ready:
            view=all_views[access.view_id]
            while isinstance(view,LogicalSelectionView) and view.parent_view_id is not None:view=all_views[view.parent_view_id]
            ready[view.view_id]=access.input_ready[0]
    allocations,views,events,expected=_cann_bsh_fd_recipe(header,parents,ready,scope,family.visibility_domain)
    if (program.tile_programs!=expected or set(all_views)!={*parents,*(view.view_id for view in views)}
            or any(all_views[view.view_id]!=view for view in views) or any(item not in program.allocations for item in allocations)
            or any(item not in program.events for item in events)):
        raise ValueError('CANN BSH FD lost normalized partial/sum/max producers, real barrier or terminal dependencies')
    totals={};gm_read=gm_write=ub_read=ub_write=0;domains={item.allocation_id:item.memory_domain for item in program.allocations}
    for access in family.accesses:
        amount=tile_template_requested_bytes(program,family,access);name=access.access_id;totals[name]=amount
        domain=domains[all_views[access.view_id].allocation_id]
        if domain=='logical-gm-workspace':
            if access.operation is Operation.READ:gm_read+=amount
            else:gm_write+=amount
        elif domain=='logical-ub-state':
            if access.operation is Operation.READ:ub_read+=amount
            else:ub_write+=amount
    values={role:prod(view.shape)*view.element_bytes for role,view in parents.items()};payload=sum(values[role] for role in ('query','key','value'))
    argument=sum(amount for name,amount in totals.items() if '.argument_' in name)
    scores=sum(amount for name,amount in totals.items() if name.endswith('.scores_write'))//4
    return {'geometry':tuple(map(int,public[1:6])),'query_unique_bytes':values['query'],'kv_unique_bytes':values['key']+values['value'],
        'kv_interface_bytes':values['key']+values['value'],'metadata_read_bytes':0,'payload_read_bytes':payload,'read_bytes':payload+argument,'write_bytes':values['output'],
        'argument_payload_bytes':int(public[6]),'argument_aiv_transfer_bytes':int(public[9]),'argument_aic_materializations':int(public[7]),
        'argument_aiv_materializations':int(public[7])*2,'argument_read_bytes':argument,
        'native_workspace_stream_bound':True,'flashdecode_stream_bound':True,'gm_workspace_read_bytes':gm_read,'gm_workspace_write_bytes':gm_write,
        'ub_state_read_bytes':ub_read,'ub_state_write_bytes':ub_write,'totals':totals,'native_ring_alias_lifetime_bound':False,
        'priced_score_elements':scores,'priced_output_elements':values['output']//2,
        'priced_task_query_l2_bytes':sum(amount for name,amount in totals.items() if name.endswith('.partial_write'))//2,
        'key_requested_bytes':sum(amount for name,amount in totals.items() if name.endswith('.key')),
        'value_requested_bytes':sum(amount for name,amount in totals.items() if name.endswith('.value')),
        'framework_clear_write_bytes':sum(amount for name,amount in totals.items() if name.endswith('.framework_clear'))}


def bound_cann_bsh_workspace_work(kernel):
    """Source-derived elements for the retained lower-bound price projection."""
    if (_cann_bsh_native_ranges(kernel) is None and _cann_bsh_fd_geometry(kernel) is None
            and not _has_cann_paged_gqa_source(kernel)):return None
    context=copy_context()
    if _CANN_ATTENTION_WORK not in context:raise ValueError('CANN BSH workspace pricing requires its bound production source')
    current,work=context[_CANN_ATTENTION_WORK]
    if current!=kernel:raise ValueError('CANN BSH workspace source binding mismatch')
    if not work['native_workspace_stream_bound']:return None
    return work


def build_cann_attention_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_attention_source(kernel):
        raise NotImplementedError('CANN attention MLA/TND/graph source ABI is unbound')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('CANN BSH decode requires typed Q/K/V/O and declared argument bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise NotImplementedError('CANN attention does not expose native issued addresses')
    header = _cann_attention_header(kernel,hardware)
    if header[6]!='unbound' and _has_cann_paged_gqa_source(kernel):
        from src.kernel.memory_cann_paged_attention import build_cann_paged_attention_source
        return build_cann_paged_attention_source(kernel,hardware,bindings,program_id=program_id)
    layouts = _cann_attention_layouts(header)
    read_roles = tuple(role for role,_ in layouts if role != 'output')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (read_roles,('output',),len(read_roles)):
        raise ValueError('CANN BSH decode requires actual payload and argument readiness')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    _validate_cann_attention_views(header,views)
    if header[6]!='unbound' and _cann_bsh_fd_geometry(kernel) is not None:
        return _build_cann_bsh_fd(kernel,hardware,bindings,header,views,dict(zip(read_roles,bindings.input_ready,strict=True)),program_id=program_id)
    if header[6]!='unbound' and _cann_bsh_native_ranges(kernel) is not None:
        return _build_cann_bsh_native(kernel,hardware,bindings,header,views,dict(zip(read_roles,bindings.input_ready,strict=True)),program_id=program_id)
    if granularity is Granularity.TILE:raise NotImplementedError('CANN BSH native tile stream remains unbound for this ABI')
    scopes,accesses,reads,argument_reads = [],[],[],{}
    metadata_read = None
    availability = dict(zip(read_roles,bindings.input_ready,strict=True))
    for role,action,repeat in _cann_attention_requests(header):
        view=views[role];identity=f'{program_id}.{action}'
        scopes.append(AccessScope(identity,(*header,action),bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-bsh-interface'))
        if role == 'tiling':
            operation,completion = Operation.READ,EventKind.READ_READY
            dependencies = (availability[role],)
            argument_reads[action] = EventRef(identity,completion)
        elif role != 'output':
            operation,completion = Operation.READ,EventKind.READ_READY
            dependencies = (availability[role],)+(() if not argument_reads else (argument_reads['tiling_aic'],))
            if role == 'block_table':
                metadata_read = EventRef(identity,completion)
            elif role in ('key','value') and metadata_read is not None:
                dependencies += (metadata_read,)
            reads.append(EventRef(identity,completion))
        else:
            operation,completion = Operation.WRITE,EventKind.WRITE_VISIBLE
            dependencies = (*reads,EventRef(allocations[view.allocation_id].allocated_event,EventKind.ALLOCATED))
            if argument_reads:dependencies += (argument_reads['tiling_aiv'],)
        accesses.append(MemoryAccess(identity,identity,view.allocation_id,None,operation,_cann_attention_region(view,header),
            Semantics.LOGICAL,None,None,None,None,bindings.visibility_domain,dependencies,completion,repeat,(
                'Public BSH BF16 payload; CANN source base disables final LSE. Actual Q/K/V reads precede output.',
                'Source-mandatory AIC and AIV argument materializations are independent actor-local reads; no native global barrier or address/issue order is asserted.',)))
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(granularity,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Known public BSH payload, logical selected cache rows and typed opaque argument; cache-view dense strides describe logical coordinates only, never native pool layout/capacity. Native addresses and internal workspace/lifetimes remain pending.',
            ('Argument source facts unbound: only the public payload is represented; missing mechanism is never observed zero.' if header[6] == 'unbound' else
             'AIC/AIV read logical struct bytes; aligned AIV GM-to-UB transfer is a separate source-copy projection, never extra logical fields.'),
            'Host actual-length arguments and disabled final-LSE output are not fabricated tensor reads/stores.',
            'Original replay/line sharing/cache/workspace/prologue/MTE/AIV/compute/sync services are retained; argument demand is not charged again as QKV traffic.',
            'Paged cache selections and valid INT32 block-table reads have unknown page values, physical mapping/stride/capacity/alias and native metadata materialization; interface bytes are not unique physical pages.',
            'TND causal mask/MLA/graph and non-BF16 ABIs require complete source bindings.')),
        bindings.allocations,bindings.views,tuple(scopes),bindings.events,tuple(accesses))


def _validate_cann_attention_views(header,views):
    layouts = _cann_attention_layouts(header)
    if set(views) != {role for role,_ in layouts} or len({v.allocation_id for v in views.values()}) != len(layouts):
        raise ValueError('CANN BSH payload/argument native parent aliases are unbound')
    for role,shape in layouts:
        view = views[role]
        width,dtype = _cann_attention_storage_type(role)
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes,view.offset_bytes) != (shape,dtype,width,dense_strides(shape,width),0):
            raise ValueError('CANN attention public payload/argument storage/shape mismatch')


def cann_attention_source_demands(program, *, expected_kernel=None):
    if program.scopes and program.scopes[0].op_ids[0]=='cann_paged_gqa':
        from src.kernel.memory_cann_paged_attention import cann_paged_attention_source_demands
        return cann_paged_attention_source_demands(program,expected_kernel=expected_kernel)
    if program.tile_programs and program.scopes and program.scopes[0].op_ids[0]=='cann_bsh_flashdecode_tiles':
        return _cann_bsh_fd_demands(program,expected_kernel=expected_kernel)
    if program.tile_programs:return _cann_bsh_native_demands(program,expected_kernel=expected_kernel)
    header = program.scopes[0].op_ids[:-1]
    views,scopes = unique_by(program.views,'view_id'),unique_by(program.scopes,'scope_id')
    _validate_cann_attention_views(header,views)
    requests = _cann_attention_requests(header)
    if len(program.accesses) != len(requests):
        raise ValueError('CANN BSH decode lost an actual payload/argument access')
    reads,arguments,values = set(),{},{}
    metadata_read = None
    for access,(role,action,repeat) in zip(program.accesses,requests,strict=True):
        view = views[role]
        if (scopes[access.scope_id].op_ids != (*header,action) or access.region != _cann_attention_region(view,header)
                or access.allocation_id != view.allocation_id or access.view_id is not None or access.repeat != repeat
                or access.operation is not (Operation.WRITE if role == 'output' else Operation.READ)):
            raise ValueError('CANN BSH decode must conserve typed payload and actor argument accesses')
        if role == 'tiling':
            arguments[action] = EventRef(access.access_id,EventKind.READ_READY)
        elif role != 'output':
            if arguments and arguments['tiling_aic'] not in access.dependencies:
                raise ValueError('CANN operand requires actual AIC argument READ_READY')
            if role == 'block_table':
                metadata_read = EventRef(access.access_id,EventKind.READ_READY)
            elif role in ('key','value') and metadata_read is not None and metadata_read not in access.dependencies:
                raise ValueError('CANN paged K/V requires actual block-table READ_READY')
            reads.add(EventRef(access.access_id,EventKind.READ_READY))
        else:
            if not reads <= set(access.dependencies):
                raise ValueError('CANN attention output requires every actual Q/K/V READ_READY')
            if arguments and arguments['tiling_aiv'] not in access.dependencies:
                raise ValueError('CANN output requires actual AIV argument READ_READY')
        values[action] = requested_bytes(access.region)*repeat
    bound = header[6] != 'unbound'
    payload_read = sum(values[role] for role in ('query','key','value'))
    argument_read = sum(values[key] for key in arguments)
    paged = _cann_attention_paged_geometry(header)
    metadata_read_bytes = 0 if paged is None else values['block_table']
    kv_interface_bytes = values['key']+values['value']
    return {'geometry':tuple(map(int,header[1:6])),'query_unique_bytes':values['query'],
        'kv_unique_bytes':kv_interface_bytes if paged is None else None,
        'kv_interface_bytes':kv_interface_bytes,'metadata_read_bytes':metadata_read_bytes,
        'payload_read_bytes':payload_read,
        'read_bytes':payload_read+argument_read+metadata_read_bytes,'write_bytes':values['output'],
        'argument_payload_bytes':int(header[6]) if bound else None,
        'argument_aiv_transfer_bytes':int(header[9]) if bound else None,
        'argument_aic_materializations':int(header[7]) if bound else None,
        'argument_aiv_materializations':int(header[7])*int(header[8]) if bound else None,
        'argument_read_bytes':argument_read if bound else None,'native_workspace_stream_bound':False}


@contextmanager
def cann_attention_source_work(kernel,work):
    token = _CANN_ATTENTION_WORK.set((kernel,work))
    try:
        yield
    finally:
        _CANN_ATTENTION_WORK.reset(token)


def current_cann_attention_source_work(kernel,hardware):
    context = copy_context()
    if _CANN_ATTENTION_WORK in context:
        current,work = context[_CANN_ATTENTION_WORK]
        if current != kernel:
            raise ValueError('CANN attention source crossed its kernel boundary')
        return work
    bindings = standalone_cann_attention_bindings(kernel,hardware,call_id='cann.attention',binding_id='cann.attention')
    return cann_attention_source_demands(build_cann_attention_source(kernel,hardware,bindings,Granularity.KERNEL,program_id='cann.attention'))


_METAX_VARLEN_WORK = ContextVar('metax_varlen_source_work')


def has_metax_varlen_source(kernel):
    if (kernel.kind is not KernelKind.FLASH_ATTN or kernel.hardware_id != 'metax_c500'
            or kernel.execution_support != 'supported' or kernel.dtype != 'bf16' or kernel.variant != 'prefill'
            or kernel.launch.tiling.num_kv_splits != 1 or kernel.attributes['causal'] is not True
            or kernel.shape['q_len'] <= 1 or 'model_input_q_lens' in kernel.attributes
            or 'model_input_kv_lens' in kernel.attributes):return False
    q,kv = tuple(kernel.attributes['q_lens']),tuple(kernel.attributes['kv_lens'])
    algorithm,layout = kernel.launch.algorithm,kernel.attributes['kv_layout']
    if layout == 'contiguous':
        return (algorithm in ('metax_flash_attention_trait_direct:prefill_q128_full_tile',
            'metax_flash_attention_trait_direct:prefill_q128_split_full_tile')
            and all(a <= b and a%128 == 0 and b%64 == 0 for a,b in zip(q,kv,strict=True)))
    if layout not in ('paged','paged_varlen','chunked_prefill') or any(not 0 < a <= b for a,b in zip(q,kv,strict=True)):
        return False
    tile = kernel.launch.tiling.tile_shape
    if not {'q', 'kv'} <= set(tile):return False
    if tile['kv'] != 64:return False
    direct_grid = (kernel.shape['num_query_heads'],-(-kernel.shape['q_len']//tile['q']),kernel.shape['batch'])
    if algorithm == 'metax_flash_attention_trait_direct:prefill_q128_split_full_tile':
        return (tile['q'] == 128 and kernel.launch.grid == direct_grid
            and all(a%128 == 0 and b%64 == 0 for a,b in zip(q,kv,strict=True)))
    if (tile['q'] != 64 or kernel.shape['head_dim'] not in (64,128,192)
            or kernel.launch.block != (256,1,1)):return False
    if algorithm == 'metax_flash_attention_trait_packed:prefill_q64_full_tile':
        return (layout == 'chunked_prefill' and 'query_heads_per_workgroup' in tile
            and tile['query_heads_per_workgroup'] == 1)
    return (kernel.launch.grid == direct_grid and layout in ('paged','paged_varlen') and algorithm in (
        'metax_flash_attention_trait_direct:prefill_q64_full_tile','metax_flash_attention_trait_direct:prefill'))


def _metax_varlen_prices_lse(kernel):
    # Preserve the pre-source service envelope; a logical LSE write is known
    # even where the legacy native-pipeline price only accounted for O.
    tile = kernel.launch.tiling.tile_shape
    if tile['q'] == 128:return True
    if kernel.shape['head_dim'] not in (64,128) or max(kernel.attributes['q_lens']) != kernel.shape['q_len']:
        return False
    direct_grid = (kernel.shape['num_query_heads'],-(-kernel.shape['q_len']//64),kernel.shape['batch'])
    if kernel.launch.grid != direct_grid:return False
    if kernel.launch.algorithm == 'metax_flash_attention_trait_packed:prefill_q64_full_tile':
        return kernel.launch.grid_size == flash_attention_packed_varlen_scheduled_workgroups(
            batch=kernel.shape['batch'],q_len=kernel.shape['q_len'],tile_q=64,
            query_heads=kernel.shape['num_query_heads'],
            query_heads_per_workgroup=tile['query_heads_per_workgroup'],splits=1)
    return True


def _metax_varlen_paged(header):
    if len(header) == 6:return False
    if (len(header) not in (8,9) or header[6] not in ('paged','paged_varlen','chunked_prefill')
            or header[7] not in ('64','128')):
        raise ValueError('C500 varlen requires explicit logical paged layout and native query block')
    if len(header) == 9 and (header[8] != 'legacy_output_only' or header[7] != '64'):
        raise ValueError('C500 varlen requires an explicit Q64 output-only price projection')
    return True


def _metax_varlen_header(kernel):
    header = ('metax_varlen_prefill',*(str(kernel.shape[key]) for key in ('head_dim','num_query_heads','num_kv_heads')),
        ','.join(map(str,kernel.attributes['q_lens'])),','.join(map(str,kernel.attributes['kv_lens'])))
    if kernel.attributes['kv_layout'] == 'contiguous':return header
    paged = (*header,kernel.attributes['kv_layout'],str(kernel.launch.tiling.tile_shape['q']))
    return paged if _metax_varlen_prices_lse(kernel) else (*paged,'legacy_output_only')


def _metax_varlen_layouts(header):
    paged = _metax_varlen_paged(header)
    if header[0] != 'metax_varlen_prefill':
        raise ValueError('C500 varlen source requires its exact length/header domain')
    d,h,hkv = map(int,header[1:4])
    q,kv = (tuple(map(int,text.split(','))) for text in header[4:6])
    if (not q or len(q) != len(kv) or min(d,h,hkv,*q,*kv) <= 0 or h%hkv
            or d not in (64,128,192) or (paged and header[7] == '64' and d == 192 and len(header) != 9)
            or any(query > cache or ((not paged or header[7] == '128') and (query%128 or cache%64))
                for query,cache in zip(q,kv,strict=True))):
        raise ValueError('C500 varlen source requires its BF16 GQA request and native query-block domain')
    return (('cu_query',(len(q)+1,),'int32',4),('cu_kv',(len(kv)+1,),'int32',4),
        ('query',(sum(q),h,d),'bf16',2),('key',(sum(kv),hkv,d),'bf16',2),
        ('value',(sum(kv),hkv,d),'bf16',2),
        *((('block_table',(sum(kv),),'int32',4),) if paged else ()),
        ('output',(sum(q),h,d),'bf16',2),('lse',(h,sum(q)),'fp32',4))


def _metax_varlen_region(view,header):
    return Unmapped(view.shape,view.dtype,view.element_bytes,
        f'metax_varlen.{view.view_id}.q{header[4]}.kv{header[5]}')


def standalone_metax_varlen_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_metax_varlen_source(kernel):
        raise NotImplementedError('C500 varlen source requires its admitted prefill ABI')
    header = _metax_varlen_header(kernel)
    allocations,views,events,ready = [],[],[],[]
    for role,shape,dtype,width in _metax_varlen_layouts(header):
        identity = f'{binding_id}.{role}'
        allocation = MemoryAllocation(identity,'logical-local-device','logical-native-tensor',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        if _metax_varlen_paged(header) and role in ('key','value','block_table'):
            view = LogicalSelectionView(role,identity,shape,dtype,width,
                f'metax_varlen.{role}.q{header[4]}.kv{header[5]}',None)
        else:view = TensorView(role,identity,0,shape,dense_strides(shape,width),dtype,width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role not in ('output','lse'):
            event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
                _metax_varlen_region(view,header),'metax-varlen-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        tuple(role for role,_,_,_ in _metax_varlen_layouts(header) if role not in ('output','lse')),
        ('output','lse'),tuple(ready),0,'offline','metax-varlen-visible')


def _validate_metax_varlen_views(header,views):
    layouts = _metax_varlen_layouts(header);paged = _metax_varlen_paged(header)
    if set(views) != {row[0] for row in layouts} or len({view.allocation_id for view in views.values()}) != len(layouts):
        raise ValueError('C500 varlen native operand aliases are unbound')
    for role,shape,dtype,width in layouts:
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes) != (shape,dtype,width):
            raise ValueError('C500 varlen TND/INT32/FP32 storage mismatch')
        if paged and role in ('key','value','block_table'):
            if (not isinstance(view,LogicalSelectionView) or view.parent_view_id is not None or view.coordinate_map
                    or view.selection_id != _metax_varlen_region(view,header).selection_id):
                raise ValueError('C500 paged source requires opaque logical cache/page-lookup parents')
        elif (not isinstance(view,TensorView) or view.strides_bytes != dense_strides(shape,width) or view.offset_bytes != 0):
            raise ValueError('C500 varlen TND/INT32 cumulative/FP32 LSE storage mismatch')


def _metax_varlen_tile_recipe(header,parents,ready,scope,visibility):
    """Packed request membership is known; native addresses and issue replay are not."""
    d,h,hkv = map(int,header[1:4]);group = h//hkv;paged = _metax_varlen_paged(header)
    lengths,cache_lengths = (tuple(map(int,text.split(','))) for text in header[4:6])
    evidence = ('FA2 causal online softmax streams the end-aligned causal KV prefix for each independent query/GQA group.',
        'Packed TND request boundaries come from the actual cumulative input arrays; each request reads its two endpoint fields.',
        'Logical endpoint requests and scalar query/KV tiles do not assert native CTA replay, addresses, transactions or cache hits.',
        'FP32 LSE requires Q/K and cumulative inputs; output additionally requires V, with no cross-request barrier.')
    if paged:evidence += (
        'Paged K/V and INT32 lookup streams retain ordered logical request/token selections, not a fabricated contiguous cache or native table shape.',
        'Each logical token lookup is shared by its query/head consumers and precedes matching K/V; native page size, entry deduplication and issue replay remain unknown.')
    selected,roots,families = [],{},[]
    for role,parent in parents.items():
        coordinates = tuple(LogicalCoordinateBinding(axis,(LogicalCoordinateTerm(axis,1),)) for axis in range(len(parent.shape)))
        root = LogicalSelectionView(f'{scope.scope_id}.{role}.selection',parent.allocation_id,parent.shape,parent.dtype,
            parent.element_bytes,_metax_varlen_region(parent,header).selection_id,parent.view_id,coordinates)
        roots[role] = root;selected.append(root)
    offset = cache_offset = 0
    for request,(length,cache_length) in enumerate(zip(lengths,cache_lengths,strict=True)):
        children = {}
        def child(role,shape,entries):
            root = roots[role]
            coordinates = tuple(LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(child_axis,coefficient)
                for child_axis,coefficient in terms),shift) for axis,(terms,shift) in enumerate(entries))
            view = LogicalSelectionView(f'{scope.scope_id}.r{request}.{role}',root.allocation_id,shape,root.dtype,
                root.element_bytes,f'{root.selection_id}.r{request}',root.view_id,coordinates)
            children[role] = view;selected.append(view)
        for role in ('cu_query','cu_kv'):
            child(role,(2,),((((0,1),),request),))
        for role in ('query','output'):
            child(role,(length,hkv,group,d),((((0,1),),offset),(((1,group),(2,1)),0),(((3,1),),0)))
        for role in ('key','value'):
            child(role,(cache_length,hkv,d),((((0,1),),cache_offset),(((1,1),),0),(((2,1),),0)))
        child('lse',(hkv,group,length),((((0,group),(1,1)),0),(((2,1),),offset)))
        if paged:child('block_table',(cache_length,),((((0,1),),cache_offset),))
        output_axes = ('kv_head','query');stream_axes = (*output_axes,'key')
        accesses = [TileAccessTemplate(role,children[role].view_id,Operation.READ,(),(None,),
            (ready[role],),(),evidence,unmapped=True) for role in ('cu_query','cu_kv')]
        metadata = tuple(TileDependency(role,EventKind.READ_READY,()) for role in ('cu_query','cu_kv'))
        if paged:
            accesses.append(TileAccessTemplate('block_table',children['block_table'].view_id,Operation.READ,
                ('lookup_token',),('lookup_token',),(ready['block_table'],),metadata,evidence,unmapped=True))
        accesses.append(TileAccessTemplate('query',children['query'].view_id,Operation.READ,output_axes,
            ('query','kv_head',None,None),(ready['query'],),metadata,evidence,unmapped=True))
        for role in ('key','value'):
            accesses.append(TileAccessTemplate(role,children[role].view_id,Operation.READ,stream_axes,
                ('key','kv_head',None),(ready[role],),(*metadata,*(
                    (TileDependency('block_table',EventKind.READ_READY,(),(TileIndexBinding('lookup_token',(TileIndexTerm('key',1),)),)),)
                    if paged else ())),evidence,unmapped=True))
        qk = (TileDependency('query',EventKind.READ_READY,()),TileDependency('key',EventKind.READ_READY,('key',)))
        accesses.append(TileAccessTemplate('output',children['output'].view_id,Operation.WRITE,output_axes,
            ('query','kv_head',None,None),(),(*qk,TileDependency('value',EventKind.READ_READY,('key',))),evidence,unmapped=True))
        accesses.append(TileAccessTemplate('lse',children['lse'].view_id,Operation.WRITE,output_axes,
            ('kv_head',None,'query'),(),qk,evidence,unmapped=True))
        family_scope = replace(scope,scope_id=f'{scope.scope_id}.r{request}',op_ids=(*header,str(request)),
            agent='logical-metax-varlen-queries')
        families.append(TileProgram(family_scope.scope_id,family_scope,(TileAxis('kv_head',hkv,1),
            TileAxis('query',length,1),TileAxis('key',cache_length,1),*(
                (TileAxis('lookup_token',cache_length,1),) if paged else ())),tuple(accesses),visibility,evidence,
            causal_prefixes=(TileCausalPrefix('key','query'),)))
        offset += length;cache_offset += cache_length
    return tuple(selected),tuple(families)


def _metax_varlen_public_operands(kernel,bindings):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_metax_varlen_source(kernel):
        raise NotImplementedError('C500 varlen source admission or captured execution ABI is unbound')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('C500 varlen source requires actual typed tensor and cumulative inputs')
    header = _metax_varlen_header(kernel);layouts = _metax_varlen_layouts(header)
    roles = tuple(row[0] for row in layouts)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (roles[:-2],roles[-2:],len(roles)-2):
        raise ValueError('C500 varlen requires both cumulative arrays and Q/K/V/O/LSE')
    views = unique_by(bindings.views,'view_id');_validate_metax_varlen_views(header,views)
    return header,roles,views


def _metax_varlen_prediction_values(header):
    """Request partitions cover each public stream once; endpoint fields overlap."""
    return {role:prod(shape)*width for role,shape,_,width in _metax_varlen_layouts(header)}


def metax_varlen_prediction_work(kernel,bindings):
    from src.kernel.memory_access import validate_storage_bindings
    header,roles,views=_metax_varlen_public_operands(kernel,bindings)
    bindings.__post_init__()
    validate_storage_bindings(bindings.allocations,bindings.views,bindings.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual C500 varlen public storage and readiness; detailed request tiles are explicit exports.',)))
    events=unique_by(bindings.events,'event_id')
    for role,ref in zip(roles[:-2],bindings.input_ready,strict=True):
        if ref.node_id not in events:raise ValueError('C500 varlen actual cumulative/payload readiness is missing')
        event=events[ref.node_id];parent=views[role]
        if (ref.kind is not EventKind.LOGICAL_AVAILABLE or event.kind is not ref.kind or ref.allow_skipped_read
                or event.allocation_id!=parent.allocation_id or event.visibility_domain!=bindings.visibility_domain
                or event.view_id is not None and event.view_id!=parent.view_id
                or event.region!=_metax_varlen_region(parent,header)):
            raise ValueError('C500 varlen requires actual cumulative/payload selection availability')
    return _metax_varlen_projected_values(header,_metax_varlen_prediction_values(header))


def build_metax_varlen_source(kernel,bindings,granularity, *, program_id):
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('Unknown C500 varlen logical granularity')
    header,roles,views=_metax_varlen_public_operands(kernel,bindings)
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-metax-varlen-queries')
    selected,families = _metax_varlen_tile_recipe(header,views,dict(zip(roles[:-2],bindings.input_ready,strict=True)),
        scope,bindings.visibility_domain)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Packed request/query/GQA and causal KV logical tiles preserve exact TND membership and FP32 LSE.',
            'Native addresses/capacities, workspace/RNG and CTA-issued replay remain unknown.',
            'Cumulative endpoint reads repeat at shared boundaries; the existing service projection uses the unique public input footprint.',
            'Metadata logical reads remain separate from unchanged tensor/cache/SMEM/compute/metadata-service pricing.',
            *(('Logical paged cache and lookup-stream extents are not native allocation/table dimensions; page size, entry sharing and physical KV uniqueness are unknown.',)
                if _metax_varlen_paged(header) else ()))),
        bindings.allocations,(*bindings.views,*selected),(scope,),bindings.events,(),families)
    metax_varlen_source_demands(program)
    return program


def metax_varlen_source_demands(program):
    from src.kernel.memory_access import _tile_template_requested_bytes
    program.__post_init__()
    if not program.tile_programs or program.accesses or len(program.scopes) != 1:
        raise ValueError('C500 varlen requires its unique packed tile source')
    scope, = program.scopes;header = scope.op_ids;layouts = _metax_varlen_layouts(header)
    views = unique_by(program.views,'view_id');parents = {role:views[role] for role,*_ in layouts}
    _validate_metax_varlen_views(header,parents)
    first = unique_by(program.tile_programs[0].accesses,'access_id')
    if any(len(first[role].input_ready) != 1 for role,*_ in layouts[:-2]):
        raise ValueError('C500 varlen requires actual cumulative/payload availability')
    ready = {role:first[role].input_ready[0] for role,*_ in layouts[:-2]}
    events = unique_by(program.events,'event_id')
    for role,ref in ready.items():
        if ref.kind is not EventKind.LOGICAL_AVAILABLE or events[ref.node_id].region != _metax_varlen_region(parents[role],header):
            raise ValueError('C500 varlen requires actual cumulative/payload selection availability')
    selected,families = _metax_varlen_tile_recipe(header,parents,ready,scope,program.tile_programs[0].visibility_domain)
    if (program.tile_programs != families or set(views) != {*parents,*(view.view_id for view in selected)}
            or any(views[view.view_id] != view for view in selected)):
        raise ValueError('C500 varlen tile membership or actual cumulative READ_READY causality mismatch')
    # Request selections partition logical interface streams. Physical page
    # aliases are unknown; only cumulative endpoint overlap is resolved here.
    values = {role:0 for role,*_ in layouts};endpoints = {role:set() for role in ('cu_query','cu_kv')}
    for family in program.tile_programs:
        axes = unique_by(family.axes,'axis_id');accesses = unique_by(family.accesses,'access_id')
        for role,access in accesses.items():
            view = views[access.view_id]
            if role in endpoints:
                binding, = view.coordinate_map
                endpoints[role].update((binding.offset,binding.offset+1))
                continue
            count = _tile_template_requested_bytes(views,axes,family,access)
            if role in ('key','value'):
                # The end-aligned stream visits K-Q+q+1 tokens for each
                # query q. Its final query covers the complete selected KV.
                queries,keys = axes['query'].extent,axes['key'].extent
                pairs = queries*(keys-queries)+queries*(queries+1)//2
                count,remainder = divmod(count*keys,pairs)
                if remainder:raise ValueError('C500 varlen causal replay cannot be projected exactly')
            values[role] += count
    for role in endpoints:values[role] = len(endpoints[role])*views[role].element_bytes
    if values!=_metax_varlen_prediction_values(header):
        raise ValueError('C500 varlen request tile bytes differ from public stream projection')
    return _metax_varlen_projected_values(header,values)


def _metax_varlen_projected_values(header,values):
    payload = sum(values[role] for role in ('query','key','value'));metadata = values['cu_query']+values['cu_kv']
    page_reads = values['block_table'] if _metax_varlen_paged(header) else 0
    return {'query_unique_bytes':values['query'],'kv_interface_bytes':values['key']+values['value'],
        'payload_read_bytes':payload,'metadata_read_bytes':metadata+page_reads,'read_bytes':payload+metadata+page_reads,
        'write_bytes':values['output']+values['lse'],
        'priced_write_bytes':values['output']+(values['lse'] if len(header) != 9 else 0),
        'unpriced_lse_write_bytes':values['lse'] if len(header) == 9 else 0,
        **({'page_lookup_read_bytes':page_reads,
            'page_table_unique_bytes':None,'kv_unique_bytes':None} if _metax_varlen_paged(header) else {})}


@contextmanager
def metax_varlen_source_work(kernel,work):
    token = _METAX_VARLEN_WORK.set((kernel,work))
    try:
        yield
    finally:
        _METAX_VARLEN_WORK.reset(token)


def current_metax_varlen_source_work(kernel,hardware):
    context = copy_context()
    if _METAX_VARLEN_WORK in context:
        current,work = context[_METAX_VARLEN_WORK]
        if current != kernel:
            raise ValueError('C500 varlen source crossed its kernel boundary')
        return work
    from src.kernel.memory_access_builder import metax_varlen_prediction_work as evaluate
    return evaluate(kernel,hardware)


def cann_split_fuse_stack_geometry(
        packed_rows: int, attended_kv: int, query_rows: int,
        kv_stack_ceiling: int):
    """Aggregate one SplitFuse task's regular 512-column stack loop.

    Vec1 visits the full stacks with identical geometry; causal masking can
    only begin near the tail because a source Q tile is at most 128 rows.
    Representing the full, masked-full and final stack classes preserves the
    exact source count without materializing every 512-column stack.
    """

    if (packed_rows <= 0 or attended_kv <= 0 or query_rows <= 0
            or kv_stack_ceiling <= 0):
        raise ValueError("CANN SplitFuse stack dimensions must be positive")
    stack_count = ceil(attended_kv / kv_stack_ceiling)
    final_stack_elements = (
        attended_kv - (stack_count - 1) * kv_stack_ceiling)
    if final_stack_elements <= 0:
        raise AssertionError("CANN SplitFuse final stack is empty")
    full_stack_count = stack_count - 1
    if query_rows == 1:
        first_masked_stack = stack_count
    else:
        first_masked_stack = (
            ceil(
                (attended_kv - query_rows + 2) / kv_stack_ceiling) - 1)
    if not 0 <= first_masked_stack <= stack_count:
        raise AssertionError("CANN SplitFuse causal stack boundary is invalid")
    unmasked_full_count = min(first_masked_stack, full_stack_count)
    masked_full_count = full_stack_count - unmasked_full_count
    final_stack_masked = first_masked_stack <= full_stack_count

    def row_tile(column_round: int) -> int:
        value = min((8192 // column_round // 8) * 8, 64)
        if value <= 0:
            raise ValueError(
                "CANN SplitFuse Vec1 row tile is empty for the KV stack")
        return value

    full_column_round = kv_stack_ceiling
    full_row_tile = row_tile(full_column_round)
    final_alignment = 32 if final_stack_masked else 16
    final_column_round = (
        ceil(final_stack_elements / final_alignment) * final_alignment)
    final_row_tile = row_tile(final_column_round)
    mask_regions = ()
    if masked_full_count:
        mask_regions += (((packed_rows,full_column_round),masked_full_count),)
    if final_stack_masked:
        mask_regions += (((packed_rows,final_column_round),1),)
    full_exp_invocations = tuple(
        full_stack_count * ceil(subblock_rows / full_row_tile)
        for subblock_rows in (packed_rows // 2, packed_rows - packed_rows // 2))
    exp_invocations = tuple(
        invocations + ceil(subblock_rows / final_row_tile)
        for invocations, subblock_rows in zip(
            full_exp_invocations,
            (packed_rows // 2, packed_rows - packed_rows // 2)))
    return mask_regions, exp_invocations



_CANN_TND_WORK = ContextVar('cann_tnd_source_work')


def has_cann_tnd_source(kernel):
    return (kernel.kind is KernelKind.FLASH_ATTN and kernel.hardware_id == 'ascend_910b'
        and kernel.execution_support == 'supported' and kernel.dtype == 'bf16' and kernel.variant == 'prefill'
        and kernel.launch is not None and (
            (kernel.attributes['kv_layout'] == 'contiguous'
                and kernel.launch.algorithm == 'cann_flash_attention_score:5000000000000200203')
            or (kernel.attributes['kv_layout'] in ('paged','paged_varlen','chunked_prefill')
                and kernel.launch.algorithm == 'cann_flash_attention_score:5000000000010200203'))
        and kernel.launch.tiling is not None and kernel.launch.tiling.num_kv_splits == 1
        and kernel.attributes['causal'] is True
        and ('model_input_q_lens' not in kernel.attributes
            or (tuple(kernel.attributes['q_lens']),tuple(kernel.attributes['kv_lens']))
                == (tuple(kernel.attributes['model_input_q_lens']),tuple(kernel.attributes['model_input_kv_lens']))))


def _cann_tnd_header(kernel):
    header = ('cann_tnd_prefill',*(str(kernel.shape[key]) for key in ('head_dim','num_query_heads','num_kv_heads')),
        ','.join(map(str,kernel.attributes['q_lens'])),','.join(map(str,kernel.attributes['kv_lens'])))
    if kernel.attributes['kv_layout'] == 'contiguous':
        return header
    from src.kernel.launch_config.ascend_910b.flash_attn_launch import _CANN_FIA_PAGE_SIZE
    return (*header,kernel.attributes['kv_layout'],str(_CANN_FIA_PAGE_SIZE))


def _cann_tnd_pages(header):
    if len(header) == 6:
        return None
    if (len(header) != 8 or header[6] not in ('paged','paged_varlen','chunked_prefill')
            or header[7] != '128'):
        raise ValueError('CANN TND requires its source128 page and explicit cache layout')
    return tuple((int(length)+127)//128 for length in header[5].split(','))


def _cann_tnd_native_stream(header):
    """Valid logical elements follow the same source stacks at every positive tail.

    Epilogue alignment adds issued padding, which remains separate from the
    valid-element stream. GetQSBlockTile is128 for every KV length; neither a
    query tail nor a KV tail changes the logical QK/online-state/PV recurrence.
    """
    return (len(header) in (6,8) and bool(header[4]) and bool(header[5])
        and all(int(value)>0 for field in header[4:6] for value in field.split(',')))


def _cann_tnd_layouts(header):
    if len(header) not in (6,8) or header[0] != 'cann_tnd_prefill':
        raise ValueError('CANN TND requires its exact public length header')
    d,h,hkv = map(int,header[1:4])
    q,kv = (tuple(map(int,value.split(','))) for value in header[4:6])
    if (not q or len(q) != len(kv) or min(d,h,hkv,*q,*kv) <= 0 or h%hkv or h == hkv
            or d > 256 or d%16 or any(x > y for x,y in zip(q,kv,strict=True))):
        raise ValueError('CANN TND requires causal BF16 GQA lengths and 16-aligned D<=256')
    pages = _cann_tnd_pages(header)
    metadata = () if pages is None else (('block_table',(len(q),max(pages)),'int32',4),)
    if _cann_tnd_native_stream(header):
        metadata += (('actual_query',(len(q),),'int64',8),('actual_key',(len(q),),'int64',8))
    return (*metadata,('query',(sum(q),h,d),'bf16',2),('key',(sum(kv),hkv,d),'bf16',2),
        ('value',(sum(kv),hkv,d),'bf16',2),('mask',(2048,2048),'int8',1),('output',(sum(q),h,d),'bf16',2))


def standalone_cann_tnd_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_tnd_source(kernel):
        raise NotImplementedError('CANN TND source requires its exact contiguous or paged prefill ABI')
    header = _cann_tnd_header(kernel)
    allocations,views,events,ready = [],[],[],[]
    for role,shape,dtype,width in _cann_tnd_layouts(header):
        identity = f'{binding_id}.{role}'
        allocation = MemoryAllocation(identity,'logical-local-device','logical-native-tensor',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = TensorView(role,identity,0,shape,dense_strides(shape,width),dtype,width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role != 'output':
            event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
                Unmapped(shape,dtype,width,f'cann_tnd.{role}.declared_input'),'cann-tnd-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        tuple(role for role,_,_,_ in _cann_tnd_layouts(header) if role != 'output'),
        ('output',),tuple(ready),0,'offline','cann-tnd-visible')


def _validate_cann_tnd_views(header,views):
    layouts = _cann_tnd_layouts(header)
    if set(views) != {row[0] for row in layouts} or len({view.allocation_id for view in views.values()}) != len(layouts):
        raise ValueError('CANN TND native parent aliases remain unbound')
    for role,shape,dtype,width in layouts:
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes,view.offset_bytes) != (
                shape,dtype,width,dense_strides(shape,width),0):
            raise ValueError('CANN TND BF16 payload/INT8 mask storage mismatch')


def _cann_tnd_query_classes(q,kv):
    """Q128 classes with a512 token stride; bounded in the query length."""
    if q <= 128:return ((0,q,1,kv),)
    classes=[];full,tail=divmod(q,128)
    for residue in range(min(4,full)):
        start=residue*128;groups=(full-1-residue)//4+1;attended=kv-q+start+128
        # Separate the finite first/last-state transitions. The repeated class
        # then has at least one middle stack for every query coordinate.
        while groups:
            stacks=(attended+511)//512;columns=attended-(stacks-1)*512
            middle=stacks-2-int(stacks>2 and columns<127)
            if middle >= 1:break
            classes.append((start,128,1,attended));start+=512;attended+=512;groups-=1
        if groups:classes.append((start,128,groups,attended))
    if tail:classes.append((full*128,tail,1,kv))
    return tuple(classes)


def _cann_tnd_priced_mask_bytes(header):
    h,hkv=map(int,header[2:4]);total=0
    q_lens,kv_lens=(tuple(map(int,text.split(','))) for text in header[4:6])
    for q,kv in zip(q_lens,kv_lens,strict=True):
        bundle=max(1,min(h//hkv,2*((128//q)//2)));full,tail=divmod(h//hkv,bundle)
        for _,rows,groups,attended in _cann_tnd_query_classes(q,kv):
            for heads,count in ((bundle,full),(tail,int(tail>0))):
                if not count:continue
                regions,_=cann_split_fuse_stack_geometry(rows*heads,attended,rows,512)
                total+=groups*count*hkv*sum(prod(shape)*repeat for shape,repeat in regions)
    return total


def _cann_tnd_long_recipe(header,parents,ready,scope,visibility):
    """Lift the same native stack recipe across independent query work items."""
    allocations,selected,events,families=[],[],[],[];qo=ko=0
    q_lens,kv_lens=(tuple(map(int,text.split(','))) for text in header[4:6])
    public=set(parents)
    for request,(q,kv) in enumerate(zip(q_lens,kv_lens,strict=True)):
        for index,(start,rows,groups,attended) in enumerate(_cann_tnd_query_classes(q,kv)):
            local=replace(scope,scope_id=f'{scope.scope_id}.request{request}.query_class{index}')
            virtual=(*header[:4],str(rows),str(attended),*header[6:])
            a,v,e,f=_cann_tnd_native_recipe(virtual,parents,ready,local,visibility,force_one_head=q>128)
            for family in f:
                group='query_group';accesses=[];dimensions={}
                base_middle=next((axis.extent for axis in family.axes if axis.axis_id == 'middle'),0)
                if groups>1 and not base_middle:raise AssertionError('Repeated CANN query class requires its actual middle state')
                for access in family.accesses:
                    for axis,name in enumerate(access.view_axes):
                        if name == 'middle':dimensions[access.view_id]=(axis,base_middle+groups-1)
                local_ids={view.view_id for view in v}
                local_views={view.view_id:view for view in v}
                needed={access.view_id for access in family.accesses};pending=list(needed)
                while pending:
                    parent=local_views[pending.pop()].parent_view_id
                    if parent in local_ids and parent not in needed:needed.add(parent);pending.append(parent)
                for view in v:
                    if view.view_id not in needed:continue
                    shape=list(view.shape)
                    if view.view_id in dimensions:
                        axis,size=dimensions[view.view_id];shape[axis]=size
                    parent=view.parent_view_id
                    mapping=[]
                    if parent in local_ids:
                        mapping.append(LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),)))
                        mapping.extend(LogicalCoordinateBinding(binding.axis+1,
                            tuple(LogicalCoordinateTerm(term.axis+1,term.coefficient) for term in binding.terms),
                            binding.offset,binding.divisor,binding.modulus) for binding in view.coordinate_map)
                    elif parent in public:
                        for binding in view.coordinate_map:
                            terms=tuple(LogicalCoordinateTerm(term.axis+1,term.coefficient) for term in binding.terms)
                            offset=binding.offset
                            if binding.axis == 0:
                                if parent in ('query','output'):
                                    offset+=qo+start;terms+=(LogicalCoordinateTerm(0,512),)
                                elif parent in ('key','value'):
                                    offset+=ko
                                    if view.view_id.endswith(('.last','.penultimate')):terms+=(LogicalCoordinateTerm(0,512),)
                                elif parent in ('actual_query','actual_key'):
                                    cumulative=parent == 'actual_query' or len(header) == 6
                                    offset=max(0,request-1) if cumulative else request
                                    shape[0]=2 if cumulative and request>0 else 1
                            if parent == 'block_table':
                                if binding.axis == 0:offset=request
                                elif binding.axis == 1 and view.view_id.endswith(('.last','.penultimate')):
                                    terms+=(LogicalCoordinateTerm(0,4),)
                            mapping.append(LogicalCoordinateBinding(binding.axis,terms,offset,binding.divisor,binding.modulus))
                    elif parent is not None:raise ValueError('CANN query lift lost its actual parent')
                    selected.append(replace(view,shape=(groups,*shape),coordinate_map=tuple(mapping)))
                for access in family.accesses:
                    deps=[]
                    for dependency in access.dependencies:
                        bindings=tuple(replace(binding,terms=(*binding.terms,TileIndexTerm(group,1)))
                            if binding.axis_id == 'middle' else binding for binding in dependency.coordinate_bindings)
                        deps.append(replace(dependency,coordinate_bindings=bindings))
                    accesses.append(replace(access,iteration_axes=(group,*access.iteration_axes),
                        view_axes=(group,*access.view_axes),dependencies=tuple(deps)))
                axes=(TileAxis(group,groups,1),*(replace(axis,extent=base_middle+groups-1)
                    if axis.axis_id == 'middle' else axis for axis in family.axes))
                families.append(replace(family,scope=replace(family.scope,op_ids=(*header,str(request),str(index))),
                    axes=axes,accesses=tuple(accesses),causal_prefixes=(TileCausalPrefix('middle',group,base_middle,1,1),) if base_middle else ()))
            allocations.extend(a);events.extend(e)
        qo+=q;ko+=kv
    return tuple(allocations),tuple(selected),tuple(events),tuple(families)


def _cann_tnd_native_recipe(header,parents,ready,scope,visibility, *, force_one_head=False):
    # Source construction and consumption use the same complete immutable
    # recipe. Bindings, readiness, scope and the selected head mode all remain
    # in the key; a different actual caller must compile its own recipe.
    return _cached_cann_tnd_native_recipe(header,tuple(sorted(parents.items())),
        tuple(sorted(ready.items())),scope,visibility,force_one_head)


@runtime_lru_cache(maxsize=8)
def _cached_cann_tnd_native_recipe(header,parent_items,ready_items,scope,visibility,force_one_head):
    """CANN8.5.1 regular BF16/NOLSEOUT: QK -> Vec1 -> PV -> Vec2.

    Workspace versions are logical per-task selections, not an invented native
    worker assignment or ring allocation extent. The source's three-slot reuse
    is recorded as evidence; physical alias/lifetime replay remains unbound.
    """
    parents=dict(parent_items);ready=dict(ready_items)
    if not force_one_head and any(int(q)>128 for q in header[4].split(',')):
        return _cann_tnd_long_recipe(header,parents,ready,scope,visibility)
    d,h,hkv = map(int,header[1:4]);gqa = h//hkv
    paged=_cann_tnd_pages(header) is not None
    q_lens,kv_lens = (tuple(map(int,text.split(','))) for text in header[4:6])
    selected,allocations,events,families = [],[],[],[]
    evidence = (
        'CANN8.5.1 flash_attention_regular.h:315-493; exact BF16/FP32 epilogues selected by flash_attention_interface.cpp:75/90.',
        'Head-major packed rows; Q once per task; KV512 stacks; gS FP32, gP BF16, gOTmp FP32 are GM logical versions.',
        'Online row max/sum/dm are FP32 UB; first state derives from actual scores, never a synthetic initializer.',
        'block_epilogue_rescale_o.hpp:369-398: GM accumulator only for subblockRows > roundDown(8192/D,8).',
        'Paged128 QK/PV each require their actual INT32 page values (block_mmad_qk.hpp:147-180; block_mmad_pv.hpp:135-168); lookup envelope is not native issue multiplicity.',
        'Native ring3 worker placement, Fixpipe padding/issued transactions and framework services remain unbound; these are requested logical elements.')
    def coord(axis,terms=(),offset=0,divisor=1,modulus=None):
        return LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(index,value) for index,value in terms),offset,divisor,modulus)
    def child(label,role,shape,mapping):
        parent = parents[role]
        view = LogicalSelectionView(f'{scope.scope_id}.{label}',parent.allocation_id,shape,parent.dtype,parent.element_bytes,
            f'cann_tnd.{label}',role,mapping)
        selected.append(view);return view
    def scratch(label,shape,dtype,width,domain):
        identity = f'{scope.scope_id}.{label}'
        allocation = MemoryAllocation(identity,'logical-local-device',domain,None,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = LogicalSelectionView(identity,identity,shape,dtype,width,f'cann_tnd.{label}',None,())
        allocations.append(allocation);selected.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        return view
    qo,ko = 0,0
    for request,(q,kv) in enumerate(zip(q_lens,kv_lens,strict=True)):
        bundle = 1 if force_one_head else max(1,min(gqa,2*((128//q)//2)))
        full,tail = divmod(gqa,bundle)
        for label,heads,count,head_start in (('full',bundle,full,0),('tail',tail,int(tail>0),full*bundle)):
            if not count:continue
            identity = f'{scope.scope_id}.r{request}.{label}'
            tasks,stacks = hkv*count,(kv+511)//512
            final_columns = kv-(stacks-1)*512
            # Q<=128 can mask at most two stacks. Isolate a masked full
            # penultimate stack so the repeated middle is entirely unmasked.
            masked_penultimate = stacks>2 and final_columns<q-1
            middle_count = max(0,stacks-2-int(masked_penultimate))
            axes = [TileAxis('task',tasks,1)]
            if middle_count:axes.append(TileAxis('middle',middle_count,1))
            accesses = []
            def add(name,view,operation,iteration,view_axes,deps=(),inputs=()):
                accesses.append(TileAccessTemplate(name,view.view_id,operation,iteration,view_axes,inputs,deps,evidence,unmapped=True))
            def dep(name,kind=EventKind.READ_READY,bindings=()):
                return TileDependency(name,kind,(),bindings)
            # Semantic cumulative endpoints: B INT64 values, no leading zero.
            # This binds the required control values, not native per-core rereads.
            for role in ('actual_query','actual_key'):
                cumulative=role == 'actual_query' or not paged
                endpoints = 2 if cumulative and request>0 else 1
                view = child(f'r{request}.{label}.{role}',role,(endpoints,),
                    (coord(0,((0,1),),max(0,request-1) if cumulative else request),))
                add(role,view,Operation.READ,('task',),(None,),inputs=(ready[role],))
            metadata = tuple(dep(role) for role in ('actual_query','actual_key'))
            phases = (('first',0,1,min(512,kv)),
                *( (('middle',1,middle_count,512),) if middle_count else ()),
                *( (('penultimate',stacks-2,1,512),) if masked_penultimate else ()),
                *( (('last',stacks-1,1,final_columns),) if stacks>1 else ()))
            phase_dependencies = {}
            for phase,start,length,columns in phases:
                iteration = ('task','middle') if phase == 'middle' else ('task',)
                kvdeps = []
                for role in ('key','value'):
                    lookupdeps=()
                    if paged:
                        pages=(columns+127)//128;name=f'{role}_table.{phase}'
                        if phase == 'middle':
                            table=child(f'r{request}.{label}.{name}','block_table',(tasks,length,pages),(
                                coord(0,(),request),coord(1,((1,4),(2,1)),start*4)))
                            table_axes=('task','middle',None)
                        else:
                            table=child(f'r{request}.{label}.{name}','block_table',(tasks,pages),(
                                coord(0,(),request),coord(1,((1,1),),start*4)))
                            table_axes=('task',None)
                        add(name,table,Operation.READ,iteration,table_axes,metadata,(ready['block_table'],))
                        lookupdeps=(dep(name),)
                    view = child(f'r{request}.{label}.{role}.{phase}',role,(tasks,length*columns,d),(
                        coord(0,((1,1),),ko+start*512),coord(1,((0,1),),divisor=count),coord(2,((2,1),))))
                    # Each middle iteration is exactly one512-column stack.
                    view_axes = ('task','middle',None) if phase == 'middle' else ('task',None,None)
                    if phase == 'middle':
                        # Flattened key positions are partitioned by a512 block axis.
                        view = replace(view,shape=(tasks,length,512,d),coordinate_map=(
                            coord(0,((1,512),(2,1)),ko+start*512),coord(1,((0,1),),divisor=count),coord(2,((3,1),))))
                        selected[-1] = view;view_axes = ('task','middle',None,None)
                    add(f'{role}.{phase}',view,Operation.READ,iteration,view_axes,(*metadata,*lookupdeps),(ready[role],))
                    kvdeps.append(dep(f'{role}.{phase}'))
                phase_dependencies[phase] = tuple(kvdeps)
            # Both vector subblocks share the task's K/V; their Q rows are disjoint.
            split = q//2 if heads == 1 else q*(heads//2)
            for sub,start,rows in ((0,0,split),(1,split,q*heads-split)):
                if not rows:continue
                row = f'row{sub}';axes.append(TileAxis(row,rows,1));prefix = f'sub{sub}'
                # A task index identifies KVhead and the bundle within its GQA group.
                # tasks = kvhead*count + bundleIndex; bundle widths can have a tail.
                # Use a two-level selected parent to retain both quotient and remainder.
                for role in ('query','output'):
                    # Parent coordinates use [KVhead,bundle,head,token,D].
                    root = child(f'r{request}.{label}.{prefix}.{role}.packed',role,(hkv,count,heads,q,d),(
                        coord(0,((3,1),),qo),coord(1,((0,gqa),(1,heads),(2,1)),head_start),coord(2,((4,1),))))
                    view = LogicalSelectionView(f'{identity}.{prefix}.{role}',root.allocation_id,(tasks,rows,d),'bf16',2,
                        f'cann_tnd.{identity}.{prefix}.{role}',root.view_id,(
                            coord(0,((0,1),),divisor=count),coord(1,((0,1),),modulus=count),
                            coord(2,((1,1),),start,divisor=q),coord(3,((1,1),),start,modulus=q),coord(4,((2,1),))))
                    selected.append(view)
                    if role == 'query':
                        add(f'{prefix}.query',view,Operation.READ,('task',row),('task',row,None),metadata,(ready['query'],))
                    else:output = view
                max_state = scratch(f'r{request}.{label}.{prefix}.max',(tasks,rows),'fp32',4,'logical-ub-state')
                sum_state = scratch(f'r{request}.{label}.{prefix}.sum',(tasks,rows),'fp32',4,'logical-ub-state')
                spill = rows > (8192//d//8)*8
                accumulator = scratch(f'r{request}.{label}.{prefix}.accumulator',(tasks,rows,d),'fp32',4,
                    'logical-gm-workspace' if spill else 'logical-ub-state') if stacks>1 else None
                previous_phase = None
                for phase,stack,length,columns in phases:
                    middle = phase == 'middle';last = phase == ('first' if stacks == 1 else 'last')
                    iteration = ('task','middle',row) if middle else ('task',row)
                    rv = ('task',row)
                    pv = ('task','middle',row,None) if middle else ('task',row,None)
                    shape = (tasks,length,rows,columns) if middle else (tasks,rows,columns)
                    p = f'{prefix}.{phase}'
                    s = scratch(f'r{request}.{label}.{p}.scores',shape,'fp32',4,'logical-gm-workspace')
                    probability = scratch(f'r{request}.{label}.{p}.probability',shape,'bf16',2,'logical-gm-workspace')
                    add(f'{p}.scores_write',s,Operation.WRITE,iteration,pv,(
                        dep(f'{prefix}.query'),phase_dependencies[phase][0]))
                    add(f'{p}.scores_read',s,Operation.READ,iteration,pv,(dep(f'{p}.scores_write',EventKind.WRITE_VISIBLE),))
                    score_deps = (dep(f'{p}.scores_read'),)
                    tri = kv-q;stack_start=stack*512;stack_end=stack_start+columns
                    if not middle and tri < stack_end-1:
                        if tri >= stack_start:
                            mask_row,mask_column = tri%32,0
                            maskcols = stack_end-(tri//32*32)
                        else:
                            mask_row,mask_column,maskcols = 0,stack_start-tri,columns
                        mask = child(f'r{request}.{label}.{prefix}.mask','mask',(tasks,rows,maskcols),(
                            coord(0,((1,1),),start,modulus=q),
                            coord(1,((2,1),))))
                        # Row token offset and mask-origin offset are separate maps.
                        token = mask
                        root = child(f'r{request}.{label}.{prefix}.mask_origin','mask',(q,maskcols),(
                            coord(0,((0,1),),mask_row),coord(1,((1,1),),mask_column)))
                        # Different masked stacks need distinct selected views.
                        root = replace(root,view_id=f'{root.view_id}.{phase}',selection_id=f'{root.selection_id}.{phase}')
                        selected[-1] = root
                        selected.remove(token)
                        mask = replace(token,view_id=f'{token.view_id}.{phase}',selection_id=f'{token.selection_id}.{phase}',parent_view_id=root.view_id)
                        selected.append(mask)
                        add(f'{p}.mask',mask,Operation.READ,('task',row),('task',row,None),inputs=(ready['mask'],))
                        score_deps += (dep(f'{p}.mask'),)
                    previous = previous_phase
                    binding = (TileIndexBinding('middle',(),middle_count-1),) if previous == 'middle' else ()
                    for state,view in (('max',max_state),('sum',sum_state)):
                        if phase != 'first':
                            carry = (TileDependency(f'{prefix}.middle.{state}_write',EventKind.WRITE_VISIBLE,(),
                                carry_axis='middle',initial_access_id=f'{prefix}.first.{state}_write'),) if middle else (
                                    dep(f'{prefix}.{previous}.{state}_write',EventKind.WRITE_VISIBLE,binding),)
                            add(f'{p}.{state}_read',view,Operation.READ,iteration,rv,carry)
                    maxdeps = score_deps+(() if phase == 'first' else (dep(f'{p}.max_read'),))
                    add(f'{p}.max_write',max_state,Operation.WRITE,iteration,rv,maxdeps)
                    dmdeps = ()
                    if phase != 'first':
                        dmshape = (tasks,length,rows) if middle else (tasks,rows)
                        dm = scratch(f'r{request}.{label}.{p}.dm',dmshape,'fp32',4,'logical-ub-state')
                        dmv = ('task','middle',row) if middle else rv
                        add(f'{p}.dm_write',dm,Operation.WRITE,iteration,dmv,(dep(f'{p}.max_read'),dep(f'{p}.max_write',EventKind.WRITE_VISIBLE)))
                        add(f'{p}.dm_read',dm,Operation.READ,iteration,dmv,(dep(f'{p}.dm_write',EventKind.WRITE_VISIBLE),))
                        dmdeps = (dep(f'{p}.dm_read'),)
                    add(f'{p}.probability_write',probability,Operation.WRITE,iteration,pv,
                        (*score_deps,dep(f'{p}.max_write',EventKind.WRITE_VISIBLE)))
                    add(f'{p}.probability_read',probability,Operation.READ,iteration,pv,(dep(f'{p}.probability_write',EventKind.WRITE_VISIBLE),))
                    add(f'{p}.sum_write',sum_state,Operation.WRITE,iteration,rv,
                        (*score_deps,dep(f'{p}.max_write',EventKind.WRITE_VISIBLE),*dmdeps,
                            *(() if phase == 'first' else (dep(f'{p}.sum_read'),))))
                    tmpshape = (tasks,length,rows,d) if middle else (tasks,rows,d)
                    tmp = scratch(f'r{request}.{label}.{p}.temporary',tmpshape,'fp32',4,'logical-gm-workspace')
                    add(f'{p}.temporary_write',tmp,Operation.WRITE,iteration,pv,(dep(f'{p}.probability_read'),phase_dependencies[phase][1]))
                    add(f'{p}.temporary_read',tmp,Operation.READ,iteration,pv,(dep(f'{p}.temporary_write',EventKind.WRITE_VISIBLE),))
                    accdeps = (dep(f'{p}.temporary_read'),*dmdeps)
                    if phase != 'first':
                        carry = (TileDependency(f'{prefix}.middle.accumulator_write',EventKind.WRITE_VISIBLE,(),
                            carry_axis='middle',initial_access_id=f'{prefix}.first.accumulator_write'),) if middle else (
                                dep(f'{prefix}.{previous}.accumulator_write',EventKind.WRITE_VISIBLE,binding),)
                        add(f'{p}.accumulator_read',accumulator,Operation.READ,iteration,(*rv,None),carry)
                        accdeps += (dep(f'{p}.accumulator_read'),)
                    if last:
                        add(f'{prefix}.output',output,Operation.WRITE,('task',row),('task',row,None),
                            (*accdeps,dep(f'{p}.sum_write',EventKind.WRITE_VISIBLE)))
                    else:
                        add(f'{p}.accumulator_write',accumulator,Operation.WRITE,iteration,(*rv,None),accdeps)
                    previous_phase = phase
            families.append(TileProgram(identity,replace(scope,scope_id=identity,op_ids=(*header,str(request),label)),
                tuple(axes),tuple(accesses),visibility,evidence))
        qo+=q;ko+=kv
    return tuple(allocations),tuple(selected),tuple(events),tuple(families)


def _build_cann_tnd_native_stream(kernel,bindings,granularity, *, program_id):
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('Unknown CANN native logical granularity')
    header = _cann_tnd_header(kernel);parents = unique_by(bindings.views,'view_id')
    _validate_cann_tnd_views(header,parents)
    roles = tuple(role for role,_,_,_ in _cann_tnd_layouts(header) if role != 'output')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (roles,('output',),len(roles)):
        raise ValueError('CANN native stream requires real INT64 lengths/payload/mask parents')
    ready = dict(zip(roles,bindings.input_ready,strict=True))
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-tnd-stack-stream')
    allocations,selected,events,families = _cann_tnd_native_recipe(header,parents,ready,scope,bindings.visibility_domain)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Native stack logical dataflow with FP32 online state and conditional GM O updates; no native address/issued-padding trace.',
            'GM workspace versions retain ring3 provenance without inventing worker placement, physical capacity or complete alias lifetime.',
            'Known INT64 control endpoints and UB state are separate from DRAM; legacy payload/mask/service price projections remain unchanged.',
            'Paged128 parents retain valid-token/table selection envelopes. Per-task K/PV page-value requirements are explicit; native lookup issue multiplicity, page values, physical mapping and persistent-cache lifetime remain unbound.')),
        (*bindings.allocations,*allocations),(*bindings.views,*selected),(scope,),(*bindings.events,*events),(),families)
    _cann_tnd_native_demands(program)
    return program


def _cann_tnd_native_demands(program):
    work=_validated_cann_tnd_native_demands(program)
    return {**work,'totals':dict(work['totals'])}


@runtime_lru_cache(maxsize=8)
def _validated_cann_tnd_native_demands(program):
    from src.kernel.memory_access import _tile_template_requested_bytes
    program.__post_init__()
    if len(program.scopes) != 1 or not program.tile_programs or program.accesses:
        raise ValueError('CANN native stream requires its actual tile source')
    scope, = program.scopes;header = scope.op_ids
    if not _cann_tnd_native_stream(header):raise ValueError('CANN native stack domain mismatch')
    views = unique_by(program.views,'view_id');roles = tuple(row[0] for row in _cann_tnd_layouts(header))
    parents = {role:views[role] for role in roles};_validate_cann_tnd_views(header,parents)
    first = program.tile_programs[0];ready = {}
    for access in first.accesses:
        if access.input_ready:
            view = views[access.view_id]
            while isinstance(view,LogicalSelectionView) and view.parent_view_id is not None:view = views[view.parent_view_id]
            ready[view.view_id] = access.input_ready[0]
    # Q1 has no actual mask request, but its parent availability remains part
    # of the public ABI and is reconstructed from the exact bound event.
    if 'mask' not in ready:
        candidates = tuple(event for event in program.events if event.kind in (EventKind.LOGICAL_AVAILABLE,EventKind.INITIALIZED,EventKind.WRITE_VISIBLE)
            and event.allocation_id == parents['mask'].allocation_id and event.visibility_domain == first.visibility_domain)
        if len(candidates) != 1:raise ValueError('CANN native mask parent readiness is ambiguous')
        ready['mask'] = EventRef(candidates[0].event_id,candidates[0].kind)
    allocations,selected,events,expected = _cann_tnd_native_recipe(header,parents,ready,scope,first.visibility_domain)
    actual_allocations=unique_by(program.allocations,'allocation_id')
    actual_events=unique_by(program.events,'event_id')
    if (program.tile_programs != expected or set(views) != {*parents,*(view.view_id for view in selected)}
            or any(views[view.view_id] != view for view in selected)
            or any(allocation.allocation_id not in actual_allocations
                or actual_allocations[allocation.allocation_id] != allocation for allocation in allocations)
            or any(event.event_id not in actual_events or actual_events[event.event_id] != event for event in events)):
        raise ValueError('CANN native stream lost actual stack/state/mask READ_READY or parent mapping')
    totals = {};gm_read=gm_write=ub_read=ub_write=requested_read=0
    domains = {allocation.allocation_id:allocation.memory_domain for allocation in program.allocations}
    for family in program.tile_programs:
        axes=unique_by(family.axes,'axis_id')
        for access in family.accesses:
            amount = _tile_template_requested_bytes(views,axes,family,access)
            if access.operation is Operation.READ:requested_read+=amount
            totals[access.access_id] = amount+(totals[access.access_id] if access.access_id in totals else 0)
            domain = domains[views[access.view_id].allocation_id]
            if domain == 'logical-gm-workspace':
                if access.operation is Operation.READ:gm_read+=amount
                else:gm_write+=amount
            if domain == 'logical-ub-state':
                if access.operation is Operation.READ:ub_read+=amount
                else:ub_write+=amount
    size = lambda role:prod(parents[role].shape)*parents[role].element_bytes
    payload = sum(size(role) for role in ('query','key','value'))
    mask = sum(amount for name,amount in totals.items() if name.endswith('.mask'))
    lengths = sum(amount for name,amount in totals.items() if name in ('actual_query','actual_key'))
    table_reads=sum(amount for name,amount in totals.items() if name.startswith(('key_table.','value_table.')))
    priced_mask = _cann_tnd_priced_mask_bytes(header)
    return {'totals':totals,'query_unique_bytes':size('query'),'kv_interface_bytes':size('key')+size('value'),
        'kv_unique_bytes':None if _cann_tnd_pages(header) is not None else size('key')+size('value'),
        'payload_read_bytes':payload,'mask_read_bytes':mask,
        'priced_mask_read_bytes':priced_mask,'metadata_read_bytes':lengths+table_reads,'read_bytes':requested_read,
        'write_bytes':size('output'),'native_length_arrays_dtype':'int64','native_length_read_bytes':lengths,
        'page_lookup_read_bytes':table_reads,
        'native_workspace_stream_bound':True,'gm_workspace_read_bytes':gm_read,'gm_workspace_write_bytes':gm_write,
        'ub_state_read_bytes':ub_read,'ub_state_write_bytes':ub_write,
        'native_ring_alias_lifetime_bound':False}


def build_cann_tnd_source(kernel,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_tnd_source(kernel):
        raise NotImplementedError('CANN TND graph/other ABI remains unbound')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('CANN TND requires actual typed payload and mask bindings')
    return _build_cann_tnd_native_stream(kernel,bindings,granularity,program_id=program_id)


def cann_tnd_source_demands(program):
    """Current CANN TND consumers require the single tile/state source."""
    if not program.tile_programs or program.accesses:
        raise ValueError('CANN TND projection requires its current tile/state source')
    return _cann_tnd_native_demands(program)


@contextmanager
def cann_tnd_source_work(kernel,work):
    token = _CANN_TND_WORK.set((kernel,work))
    try:
        yield
    finally:
        _CANN_TND_WORK.reset(token)


def current_cann_tnd_source_work(kernel,hardware):
    context = copy_context()
    if _CANN_TND_WORK in context:
        current,work = context[_CANN_TND_WORK]
        if current != kernel:
            raise ValueError('CANN TND source crossed its kernel boundary')
        return work
    bindings = standalone_cann_tnd_bindings(kernel,call_id='cann.tnd',binding_id='cann.tnd')
    return cann_tnd_source_demands(build_cann_tnd_source(kernel,bindings,Granularity.KERNEL,program_id='cann.tnd'))


def has_fa3_prefill_source(kernel):
    return (kernel.kind is KernelKind.FLASH_ATTN and kernel.hardware_id == 'nvidia_h200'
        and kernel.variant == 'mla_prefill' and kernel.execution_support == 'supported'
        and kernel.dtype in ('bf16','fp16') and kernel.launch.algorithm == 'vllm_fa3_sm90_mla_prefill'
        and kernel.launch.tiling.num_kv_splits == 1
        and (kernel.shape['qk_head_dim'],kernel.shape['v_head_dim']) == (192,128)
        and all(length > 0 for length in (*kernel.attributes['q_lens'],*kernel.attributes['kv_lens'])))


def _fa3_prefill_header(kernel):
    # A supplied context/noncausal descriptor retains its original numerical
    # interface without claiming that the native vLLM wrapper emits it.
    return ('fa3_varlen_prefill',','.join(map(str,kernel.attributes['q_lens'])),str(kernel.shape['num_query_heads']),
        str(kernel.launch.grid_size),kernel.dtype,','.join(map(str,kernel.attributes['kv_lens'])),
        'causal' if kernel.attributes['causal'] else 'noncausal')


def _fa3_prefill_layouts(header):
    if len(header) != 7 or header[0] != 'fa3_varlen_prefill' or int(header[3])<=0 or header[4] not in ('bf16','fp16') or header[6] not in ('causal','noncausal'):raise ValueError('FA3 prefill requires its actual lengths and 16-bit dtype')
    lengths = tuple(map(int,header[1].split(',')));kv_lengths=tuple(map(int,header[5].split(',')));heads = int(header[2])
    if not lengths or len(lengths)!=len(kv_lengths) or min(*lengths,*kv_lengths,heads) <= 0:raise ValueError('FA3 prefill requires positive paired lengths/heads')
    tokens,batch = sum(lengths),len(lengths)
    return (('query',(tokens,heads,192),header[4],2),('key',(sum(kv_lengths),heads,192),header[4],2),
        ('value',(sum(kv_lengths),heads,128),header[4],2),('cu_query',(batch+1,),'int32',4),('cu_key',(batch+1,),'int32',4),
        # Pinned flash_api: prepared Q lengths + causal head-swizzle vector;
        # unsplit has no dynamic-split vector and Sort is explicitly false.
        ('scheduler_metadata',(1+(1+int(header[6]=='causal' and max(lengths)>1))*(-(-batch//4)*4),),'int32',4),
        ('output',(tokens,heads,128),header[4],2),('lse',(heads,tokens),'fp32',4))


def fa3_prefill_storage_layouts(kernel):
    """Existing native extents for actual model allocation owners."""
    if not has_fa3_prefill_source(kernel):
        raise ValueError('FA3 public layouts require the existing unsplit source')
    return _fa3_prefill_layouts(_fa3_prefill_header(kernel))


def standalone_fa3_prefill_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_fa3_prefill_source(kernel):raise NotImplementedError('FA3 fresh unsplit prefill ABI is unbound')
    allocations,views,events,ready = [],[],[],[]
    layouts = _fa3_prefill_layouts(_fa3_prefill_header(kernel))
    for index,(role,shape,dtype,width) in enumerate(layouts):
        identity = f'{binding_id}.{role}'
        allocation = MemoryAllocation(identity,'logical-local-device','logical-fa3-prefill',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = TensorView(role,identity,0,shape,dense_strides(shape,width),dtype,width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if index < 5:
            event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
                Unmapped(shape,dtype,width,f'{binding_id}.{role}.declared_input'),'fa3-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    roles = tuple(role for role,_,_,_ in layouts)
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        roles[:5],roles[6:],tuple(ready),0,'offline','fa3-visible')


def _fa3_prefill_controls(header,parents,ready,scope,visibility,allocation):
    """Pinned prepare -> typed fields -> native persistent queue messages."""
    from src.kernel.memory_access import TileWriteSource,TileAtomicAdd,SymbolicExtent
    lengths=tuple(map(int,header[1].split(',')));heads=int(header[2]);workers=int(header[3])
    batch=len(lengths);rounded=-(-batch//4)*4;swizzled=header[6]=='causal' and max(lengths)>1;vectors=1+int(swizzled)
    meta=parents['scheduler_metadata'];selected=[];events=[];families=[];fields={};available={}
    evidence=('Pinned FA3 launch: VarlenDynamicPersistent, Prepared=true, Sort=false; LPT is the normalized causal flag.',
        'prepare reads each logical request endpoint pair (including values shared by native warp shuffles), then writes prepared-Q and optional head-swizzle; these are logical dependencies, not issued-thread scalar-load multiplicities.',
        'Static unsplit prepare clears one shared total-block scalar and the global semaphore per helper CTA; the split-only shared reduction and compile-time disabled sort have no reads.',
        'mainloop load calls scheduler_prefetch exactly once per valid work tile, including the early empty-K return; atomic order and worker assignment remain unresolved.',
        'One logical int4 work message is published and consumed per valid tile and terminal worker. Views are logical versions, not simultaneous physical shared memory or issued warp-broadcast duplication.')
    for role,offset,shape in (('prepared_query',0,(batch,)),*(
            (('head_swizzle',rounded*4,(batch,)),) if swizzled else ()),('semaphore',vectors*rounded*4,(1,))):
        view=TensorView(f'{scope.scope_id}.{role}',meta.allocation_id,meta.offset_bytes+offset,shape,(4,),'int32',4)
        fields[role]=view;selected.append(view)
    prefix=allocation.allocation_id
    helper_id=f'{scope.scope_id}.prepare';helper_scope=replace(scope,scope_id=helper_id,op_ids=(*header,'prepare_varlen'),agent='logical-fa3-prepare-requests')
    ctas=-(-batch//992)
    shared=MemoryAllocation(f'{prefix}.prepare_shared',allocation.device_id,'logical-fa3-prepare-shared',ctas*4,
        AddressLevel.OBJECT,None,None,f'{prefix}.prepare_shared.allocated',None)
    shared_view=TensorView(f'{scope.scope_id}.prepare_shared',shared.allocation_id,0,(ctas,),(4,),'int32',4)
    selected.append(shared_view);events.append(ProgramEvent(shared.allocated_event,EventKind.ALLOCATED,shared.allocation_id,None,None,()))
    accesses=[]
    for role,index in (('cu_query',3),('cu_key',4)):
        parent=parents[role]
        view=LogicalSelectionView(f'{helper_id}.{role}',parent.allocation_id,(batch,2),'int32',4,
            f'fa3.prepare.{role}.request_endpoint_pair',role,(
                LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),LogicalCoordinateTerm(1,1))),))
        selected.append(view)
        accesses.append(TileAccessTemplate(role,view.view_id,Operation.READ,('request',),('request',None),
            (ready[index],),(),evidence,unmapped=True))
    for role,dependency in (('prepared_query','cu_query'),*(
            (('head_swizzle','cu_key'),) if swizzled else ())):
        accesses.append(TileAccessTemplate(role+'.write',fields[role].view_id,Operation.WRITE,('request',),('request',),(),
            (TileDependency(dependency,EventKind.READ_READY,()),),evidence,unmapped=True))
    accesses.extend((TileAccessTemplate('semaphore.zero',fields['semaphore'].view_id,Operation.WRITE,('prepare_cta',),(None,),(),(),
        evidence,unmapped=True,constant_value=ConstantValue.ZERO),
        TileAccessTemplate('shared.zero',shared_view.view_id,Operation.WRITE,('prepare_cta',),('prepare_cta',),(),(),
            evidence,constant_value=ConstantValue.ZERO)))
    families.append(TileProgram(helper_id,helper_scope,(TileAxis('request',batch,1),TileAxis('prepare_cta',ctas,1)),
        tuple(accesses),visibility,evidence))
    for role,view in fields.items():
        write=role+'.write' if role!='semaphore' else 'semaphore.zero'
        event=ProgramEvent(f'{prefix}.{role}.prepared',EventKind.LOGICAL_AVAILABLE,meta.allocation_id,
            Unmapped(view.shape,'int32',4,f'{prefix}.{role}.actual_prepared_field'),visibility,
            (EventRef(allocation.allocated_event,EventKind.ALLOCATED),),view_id=view.view_id,
            tile_sources=(TileWriteSource(helper_id,write),))
        events.append(event);available[role]=EventRef(event.event_id,event.kind)
    work=sum(-(-length//128)*heads for length in lengths)
    counter_id=f'{scope.scope_id}.scheduler_counter';counter_scope=replace(scope,scope_id=counter_id,op_ids=(*header,'scheduler_counter'),agent='logical-fa3-scheduler-rmw')
    counter=(TileAccessTemplate('read',fields['semaphore'].view_id,Operation.READ,('work',),(None,),
        (available['semaphore'],),(),evidence,unmapped=True),
        TileAccessTemplate('increment',fields['semaphore'].view_id,Operation.WRITE,('work',),(None,),(),
            (TileDependency('read',EventKind.READ_READY,()),),evidence,unmapped=True,atomic_add=TileAtomicAdd('read',1)))
    families.append(TileProgram(counter_id,counter_scope,(TileAxis('work',work,1),),counter,visibility,evidence))
    # Shared work-info is private to an unknown assigned worker. Preserve
    # logical versions and publication rather than assigning a physical CTA.
    info=MemoryAllocation(f'{prefix}.work_info_versions',allocation.device_id,'logical-fa3-work-info-versions',None,
        AddressLevel.OBJECT,None,None,f'{prefix}.work_info_versions.allocated',None)
    events.append(ProgramEvent(info.allocated_event,EventKind.ALLOCATED,info.allocation_id,None,None,()))
    terminal_id=f'{scope.scope_id}.terminal';terminal_scope=replace(scope,scope_id=terminal_id,op_ids=(*header,'terminal_worker_message'),agent='logical-fa3-terminal-messages')
    pq=fields['prepared_query']
    terminal_q=LogicalSelectionView(terminal_id+'.query',pq.allocation_id,(workers,),'int32',4,
        'fa3.scheduler.terminal_actual_previous_batch_query',pq.view_id)
    terminal_info=LogicalSelectionView(terminal_id+'.info',info.allocation_id,(workers,4),'int32',4,
        'fa3.scheduler.terminal_int4_versions',None)
    selected.extend((terminal_q,terminal_info))
    terminal=(TileAccessTemplate('prepared_query',terminal_q.view_id,Operation.READ,('worker',),('worker',),
        (available['prepared_query'],),(),evidence,unmapped=True),
        TileAccessTemplate('publish',terminal_info.view_id,Operation.WRITE,('worker',),('worker',None),(),
            (TileDependency('prepared_query',EventKind.READ_READY,()),),evidence,unmapped=True),
        TileAccessTemplate('consume',terminal_info.view_id,Operation.READ,('worker',),('worker',None),(),
            (TileDependency('publish',EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True))
    families.append(TileProgram(terminal_id,terminal_scope,(TileAxis('worker',workers,1),),terminal,visibility,evidence))
    if batch>1:
        extra=SymbolicExtent(f'{prefix}.extra_prepared_query_reads',workers*(min(batch,31)-1))
        identity=f'{scope.scope_id}.scheduler_group_lookahead';family_scope=replace(scope,scope_id=identity,op_ids=(*header,'extra_group_lookahead'),agent='logical-fa3-query-lookahead')
        view=LogicalSelectionView(identity+'.query',pq.allocation_id,(extra,),'int32',4,
            'fa3.scheduler.extra_actual_group_query_entries',pq.view_id)
        selected.append(view)
        families.append(TileProgram(identity,family_scope,(TileAxis('entry',extra,1),),(
            TileAccessTemplate('prepared_query',view.view_id,Operation.READ,('entry',),('entry',),
                (available['prepared_query'],),(),evidence,unmapped=True),),visibility,evidence))
    return (shared,info),tuple(selected),tuple(events),tuple(families),fields,available,info


def _fa3_prefill_tile_recipe(header,parents,ready,scope,visibility,fields,available,info):
    lengths = tuple(map(int,header[1].split(',')));kv_lengths=tuple(map(int,header[5].split(',')));heads = int(header[2]);offset = kv_offset = 0
    if len(ready) != 5:raise ValueError('FA3 prefill requires all known actual data/cumulative inputs')
    selected,families = [],[]
    evidence = ('Pinned H200 FA3 unsplit varlen template: 16-bit Q/K192, V/O128, declared128-row query blocks, FP32 final LSE. Public supplied descriptors do not prove native wrapper emission.',
        'A query-block/head work item retains its own KV prefix and reduction; last query rows and KV tokens are masked.',
        'Logical key-token panels are compressed under the native128-row query block; no instruction/TMA padding or cache claim.',
        'INT32 endpoints, actual prepared-Q/head-swizzle reads and int4 producer/consumer messages retain their true field and prepare dependencies; worker assignment stays unresolved.')
    def child(label,role,shape,mapping=()):
        parent = parents[role] if role in parents else fields[role]
        view = LogicalSelectionView(f'{scope.scope_id}.{label}.{role}',parent.allocation_id,shape,parent.dtype,parent.element_bytes,
            f'fa3_prefill.{label}.{role}',parent.view_id,mapping)
        selected.append(view);return view
    def coordinate(axis,*terms,offset=0):
        return LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(index,coefficient) for index,coefficient in terms),offset)
    for request,(length,kv_length) in enumerate(zip(lengths,kv_lengths,strict=True)):
        full,tail = divmod(length,128)
        for phase,groups,rows,start in (('full',full,128,0),('tail',int(tail>0),tail,full*128)):
            if not groups:continue
            identity = f'{scope.scope_id}.r{request}.{phase}'
            operands = {}
            for role,width in (('query',192),('output',128)):
                operands[role] = child(identity,role,(groups,heads,rows,width),(
                    coordinate(0,(0,128),(2,1),offset=offset+start),coordinate(1,(1,1)),coordinate(2,(3,1))))
            operands['lse'] = child(identity,'lse',(groups,heads,rows),(
                coordinate(0,(1,1)),coordinate(1,(0,128),(2,1),offset=offset+start)))
            for role,width in (('key',192),('value',128)):
                operands[role] = child(identity,role,(kv_length,heads,width),(
                    coordinate(0,(0,1),offset=kv_offset),coordinate(1,(1,1)),coordinate(2,(2,1))))
            for role in ('cu_query','cu_key'):
                operands[role] = child(identity,role,(2,),(coordinate(0,(0,1),offset=request),))
            control_roles=('prepared_query',*(('head_swizzle',) if 'head_swizzle' in fields else ()))
            for role in control_roles:
                operands[role]=child(identity,role,(1,),(coordinate(0,offset=request),))
            work_info=LogicalSelectionView(identity+'.work_info',info.allocation_id,(groups,heads,4),'int32',4,
                f'fa3.scheduler.valid_work_message.r{request}.{phase}',None)
            selected.append(work_info)
            accesses = []
            for index,role in enumerate(('cu_query','cu_key'),3):
                view = operands[role]
                accesses.append(TileAccessTemplate(role,view.view_id,Operation.READ,('query_block','head'),
                    (None,),(ready[index],),(),evidence,unmapped=True))
            for role in control_roles:
                accesses.append(TileAccessTemplate(role,operands[role].view_id,Operation.READ,('query_block','head'),
                    (None,),(available[role],),(),evidence,unmapped=True))
            accesses.append(TileAccessTemplate('work_info.publish',work_info.view_id,Operation.WRITE,('query_block','head'),
                ('query_block','head',None),(),tuple(TileDependency(role,EventKind.READ_READY,()) for role in control_roles),evidence,unmapped=True))
            accesses.append(TileAccessTemplate('work_info.consume',work_info.view_id,Operation.READ,('query_block','head'),
                ('query_block','head',None),(),(TileDependency('work_info.publish',EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True))
            metadata = tuple(TileDependency(role,EventKind.READ_READY,()) for role in ('cu_query','cu_key','work_info.consume'))
            accesses.append(TileAccessTemplate('query',operands['query'].view_id,Operation.READ,('query_block','head'),
                ('query_block','head',None,None),(ready[0],),metadata,evidence,unmapped=True))
            for index,role in enumerate(('key','value'),1):
                accesses.append(TileAccessTemplate(role,operands[role].view_id,Operation.READ,('query_block','head','key'),
                    ('key','head',None),(ready[index],),metadata,evidence,unmapped=True))
            qk = (TileDependency('query',EventKind.READ_READY,()),TileDependency('key',EventKind.READ_READY,('key',)))
            accesses.append(TileAccessTemplate('lse',operands['lse'].view_id,Operation.WRITE,('query_block','head'),
                ('query_block','head',None),(),qk,evidence,unmapped=True))
            accesses.append(TileAccessTemplate('output',operands['output'].view_id,Operation.WRITE,('query_block','head'),
                ('query_block','head',None,None),(),(*qk,TileDependency('value',EventKind.READ_READY,('key',))),evidence,unmapped=True))
            family_scope = replace(scope,scope_id=identity,op_ids=(*header,str(request),phase))
            limits=(TileCausalPrefix('key','query_block',128+kv_length-length if phase == 'full' else kv_length,128,1),) if header[6]=='causal' else ()
            families.append(TileProgram(identity,family_scope,(TileAxis('query_block',groups,1),TileAxis('head',heads,1),
                TileAxis('key',kv_length,1)),tuple(accesses),visibility,evidence,limits))
        offset += length
        kv_offset += kv_length
    return tuple(selected),tuple(families)


def build_fa3_prefill_source(kernel,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_fa3_prefill_source(kernel) or not isinstance(bindings,ElementwiseBindings):
        raise TypeError('FA3 prefill requires its actual unsplit typed bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown FA3 logical granularity')
    if dict(kernel.launch.tiling.tile_shape) != {'q':128,'kv':128,'qk_head':192,'value_head':128}:
        raise ValueError('FA3 prefill requires its actual128x128/192/128 source tile')
    header = _fa3_prefill_header(kernel);layouts = _fa3_prefill_layouts(header);roles = tuple(row[0] for row in layouts)
    if ((bindings.read_views,bindings.write_views) != (roles[:5],roles[6:])
            or len(bindings.input_ready) != len(roles[:5])):raise ValueError('FA3 prefill data/metadata roles or readiness arity mismatch')
    views = unique_by(bindings.views,'view_id')
    _validate_fa3_prefill_parents(header,views)
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-fa3-prefill-query-blocks')
    public_allocations=unique_by(bindings.allocations,'allocation_id')
    controls=_fa3_prefill_controls(header,views,bindings.input_ready,scope,bindings.visibility_domain,
        public_allocations[views['scheduler_metadata'].allocation_id])
    internal_allocations,control_views,control_events,control_families,fields,available,info=controls
    selected,families = _fa3_prefill_tile_recipe(header,views,bindings.input_ready,scope,bindings.visibility_domain,fields,available,info)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Actual public payload and query-block causal union; no native issued-byte/address/cache trace.',
            'Prepare field writers, known queue atomics and shared int4 versions are explicit and unpriced; issued load/broadcast multiplicity is not inferred.',
            'Multi-request group-lookahead cardinality and worker assignment remain explicitly unbound; exact complete bytes/ordinary DAG export require an appropriate value-aware consumer.',
            'Legacy unique-input/output-only price projection is retained; FP32 LSE service is separately unpriced.')),
        (*bindings.allocations,*internal_allocations),(*bindings.views,*control_views,*selected),(scope,),
        (*bindings.events,*control_events),(),(*control_families,*families))
    fa3_prefill_source_demands(program,expected_kernel=kernel,expected_bindings=bindings)
    return program


def _validate_fa3_prefill_parents(header,views):
    layouts = _fa3_prefill_layouts(header)
    if set(views) != {role for role,_,_,_ in layouts}:raise ValueError('FA3 prefill requires every public/metadata parent')
    aliases={}
    for role,view in views.items():
        if view.allocation_id not in aliases:aliases[view.allocation_id]=[]
        aliases[view.allocation_id].append(role)
    for roles in aliases.values():
        if len(roles)==1:continue
        q,k=views['cu_query'],views['cu_key']
        if (set(roles)!={'cu_query','cu_key'} or header[1]!=header[5]
                or (q.offset_bytes,q.shape,q.strides_bytes)!=(k.offset_bytes,k.shape,k.strides_bytes)):
            raise ValueError('FA3 prefill native alias ABI is unbound')
    for role,shape,dtype,width in layouts:
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes) != (shape,dtype,width):raise ValueError('FA3 prefill storage shape/dtype mismatch')
        if (not isinstance(view,TensorView)
                or role in ('query','key','value','output') and view.strides_bytes[-1]!=width
                or role in ('cu_query','cu_key') and view.strides_bytes!=(4,)):
            raise ValueError('FA3 prefill requires source-contiguous channels and length arrays')
        if role=='scheduler_metadata' and (not isinstance(view,TensorView) or view.strides_bytes!=(4,)):
            raise ValueError('FA3 prefill scheduler requires its actual contiguous typed parent')


def fa3_prefill_source_demands(program,*,expected_kernel=None,expected_bindings=None):
    from src.kernel.memory_access import _tile_template_requested_bytes,SymbolicExtent
    program.__post_init__()
    if program.hardware_id!='nvidia_h200' or len(program.scopes)!=1 or not program.tile_programs or program.accesses:
        raise ValueError('FA3 prefill requires its actual compressed source')
    scope,=program.scopes;header=scope.op_ids;layouts=_fa3_prefill_layouts(header)
    if expected_kernel is not None and (not has_fa3_prefill_source(expected_kernel) or header!=_fa3_prefill_header(expected_kernel)):
        raise ValueError('FA3 prefill source belongs to a foreign kernel')
    all_views=unique_by(program.views,'view_id');parents={role:all_views[role] for role,_,_,_ in layouts}
    _validate_fa3_prefill_parents(header,parents)
    data=tuple(f for f in program.tile_programs if f.scope.op_ids[-1] in ('full','tail'))
    if not data:raise ValueError('FA3 prefill lost its actual payload families')
    first=unique_by(data[0].accesses,'access_id');ready=tuple(first[role].input_ready[0] for role in
        ('query','key','value','cu_query','cu_key'))
    allocations=unique_by(program.allocations,'allocation_id')
    internal,control_views,control_events,control_families,fields,available,info=_fa3_prefill_controls(
        header,parents,ready,scope,data[0].visibility_domain,allocations[parents['scheduler_metadata'].allocation_id])
    selected,expected=_fa3_prefill_tile_recipe(header,parents,ready,scope,data[0].visibility_domain,fields,available,info)
    if (program.tile_programs!=(*control_families,*expected)
            or set(all_views)!={*parents,*(v.view_id for v in (*control_views,*selected))}
            or any(all_views[v.view_id]!=v for v in (*control_views,*selected))
            or program.allocations[-len(internal):]!=internal or program.events[-len(control_events):]!=control_events):
        raise ValueError('FA3 prefill lost actual causal query-block/parent/metadata/QK/V reduction or prepare/control producer')
    if expected_bindings is not None:
        b=expected_bindings
        if (program.binding_id!=b.binding_id or (scope.call_id,scope.rank,scope.stream)!=(b.call_id,b.rank,b.stream)
                or program.allocations[:len(b.allocations)]!=b.allocations or program.views[:len(b.views)]!=b.views
                or program.events[:len(b.events)]!=b.events or ready!=b.input_ready
                or data[0].visibility_domain!=b.visibility_domain):
            raise ValueError('FA3 prefill source does not consume actual caller storage/readiness')
    roles=tuple(role for role,_,_,_ in layouts)
    totals={role:0 for role in (*roles[:5],*roles[6:])};control_totals={}
    known_read=known_write=0;unknown_read=False;scheduler_read=0;scheduler_unknown=False
    data_ids={family.tile_program_id for family in data}
    for family in program.tile_programs:
        axes=unique_by(family.axes,'axis_id')
        for access in family.accesses:
            view=all_views[access.view_id]
            unknown=any(isinstance(value,SymbolicExtent) for value in view.shape)
            count=None if unknown else _tile_template_requested_bytes(all_views,axes,family,access)
            if family.tile_program_id in data_ids and access.access_id in totals:totals[access.access_id]+=count
            else:
                label=f'{family.scope.op_ids[-1]}/{access.access_id}'
                if label not in control_totals:control_totals[label]=count
                elif count is None or control_totals[label] is None:control_totals[label]=None
                else:control_totals[label]+=count
            if access.operation is Operation.READ:
                unknown_read|=unknown
                if count is not None:known_read+=count
                if view.allocation_id==parents['scheduler_metadata'].allocation_id:
                    scheduler_unknown|=unknown
                    if count is not None:scheduler_read+=count
            else:
                if count is None:raise ValueError('FA3 prefill has no symbolic store cardinality')
                known_write+=count
    return {'totals':totals,'control_totals':control_totals,'known_read_bytes':known_read,
        'read_bytes':None if unknown_read else known_read,'write_bytes':known_write,
        'projected_read_bytes':sum(prod(parents[role].shape)*parents[role].element_bytes for role in ('query','key','value')),
        'projected_write_bytes':totals['output'],'unpriced_lse_write_bytes':totals['lse'],
        'native_kv_panel_bytes':128*sum(parents[role].shape[-1]*parents[role].element_bytes for role in ('key','value')),
        'known_cumulative_read_bytes':totals['cu_query']+totals['cu_key']+sum(
            control_totals[f'prepare_varlen/{role}'] for role in ('cu_query','cu_key')),
        'unpriced_metadata_read_bytes':None if scheduler_unknown else scheduler_read+totals['cu_query']+totals['cu_key'],
        'scheduler_metadata_read_bytes':None if scheduler_unknown else scheduler_read}


def fa3_prefill_prediction_work(kernel,hardware,bindings):
    """Exact priced source payloads without scheduler/tile graph expansion."""
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_access import validate_storage_bindings,validate_input_readiness,Strided
    if kernel.hardware_id!=hardware.hardware_id or not has_fa3_prefill_source(kernel):
        raise ValueError('FA3 prefill requires its existing unsplit source ABI')
    if dict(kernel.launch.tiling.tile_shape)!={'q':128,'kv':128,'qk_head':192,'value_head':128}:
        raise ValueError('FA3 prefill requires its actual128x128/192/128 source tile')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('FA3 prefill requires actual typed caller operands')
    bindings.__post_init__()
    header=_fa3_prefill_header(kernel);layouts=_fa3_prefill_layouts(header)
    roles=tuple(role for role,*_ in layouts)
    if ((bindings.read_views,bindings.write_views)!=(roles[:5],roles[6:])
            or len(bindings.input_ready)!=5):
        raise ValueError('FA3 prefill data/metadata roles or readiness arity mismatch')
    parents=unique_by(bindings.views,'view_id');_validate_fa3_prefill_parents(header,parents)
    for view in parents.values():
        if (not isinstance(view,TensorView)
                or view.region()!=Strided(view.offset_bytes,view.shape,view.strides_bytes,view.element_bytes)):
            raise ValueError('FA3 prefill actual parent geometry was corrupted')
    validate_storage_bindings(bindings.allocations,bindings.views,bindings.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual FA3 prefill parents/readiness; detailed control program remains an explicit export.',)))
    events=unique_by(bindings.events,'event_id')
    for role,ready in zip(bindings.read_views,bindings.input_ready,strict=True):
        validate_input_readiness(parents[role],ready,unmapped=True,views=parents,events=events,
            visibility_domain=bindings.visibility_domain)
    q,kv=kernel.attributes['q_lens'],kernel.attributes['kv_lens'];heads=kernel.shape['num_query_heads']
    tokens=sum(q);visits=0
    for length,keys in zip(q,kv,strict=True):
        # Same per-M causal key prefix as _fa3_prefill_tile_recipe, counting
        # typed values rather than instantiating its tile/queue objects.
        full,tail=divmod(length,128)
        if kernel.attributes['causal']:
            empty=max(0,min(full,(length-keys)//128))
            visits+=128*(full*(full+1)-empty*(empty+1))//2+(keys-length)*(full-empty)+int(tail>0)*keys
        else:
            visits+=(full+int(tail>0))*keys
    totals={'query':tokens*heads*192*2,'key':visits*heads*192*2,
        'value':visits*heads*128*2,'output':tokens*heads*128*2,'lse':tokens*heads*4}
    return {'totals':totals,
        'projected_read_bytes':sum(prod(parents[role].shape)*parents[role].element_bytes
            for role in ('query','key','value')),
        'projected_write_bytes':totals['output'],'unpriced_lse_write_bytes':totals['lse'],
        'native_kv_panel_bytes':128*(192+128)*2}


_FA3_PREFILL_WORK = ContextVar('fa3_prefill_source_work')


@contextmanager
def fa3_prefill_source_work(kernel,work):
    token = _FA3_PREFILL_WORK.set((kernel,work))
    try:yield
    finally:_FA3_PREFILL_WORK.reset(token)


def current_fa3_prefill_source_work(kernel):
    context = copy_context()
    if _FA3_PREFILL_WORK not in context:return None
    current,work = context[_FA3_PREFILL_WORK]
    if current != kernel:raise ValueError('FA3 prefill source crossed its kernel boundary')
    return work


def has_fa3_combine_source(kernel):
    return (kernel.kind is KernelKind.FLASH_ATTN and kernel.hardware_id == 'nvidia_h200'
        and kernel.variant in ('combine','split_combine') and kernel.execution_support == 'supported'
        and kernel.dtype in ('bf16','fp16','float16') and kernel.launch.algorithm == 'vllm_fa3_sm90_combine')


def _fa3_combine_header(kernel,hardware):
    return ('fa3_split_combine',','.join(map(str,kernel.attributes['q_lens'])),
        ','.join(map(str,kernel.attributes['kv_lens'])),str(kernel.shape['num_query_heads']),
        str(kernel.launch.tiling.num_kv_splits),str(int(hardware.num_compute_units)),
        'causal' if kernel.attributes['causal'] else 'noncausal',
        str(kernel.attributes['graph_model_input_tokens'] if 'graph_model_input_tokens' in kernel.attributes
            else sum(kernel.attributes['model_input_q_lens']) if 'model_input_q_lens' in kernel.attributes
            else sum(kernel.attributes['q_lens'])),kernel.dtype)


def _fa3_combine_geometry(header):
    from src.kernel.predictors.flash_attn import fa3_decode_dynamic_splits
    if len(header)!=9 or header[0]!='fa3_split_combine' or header[6] not in ('causal','noncausal') or header[8] not in ('bf16','fp16','float16'):raise ValueError('FA3 combine source header mismatch')
    q,kv=tuple(map(int,header[1].split(','))),tuple(map(int,header[2].split(',')))
    heads,static,sms=map(int,header[3:6])
    if len(set(q))!=1 or not 1<=q[0]<=512 or heads<=0 or static<=1:
        raise ValueError('FA3 combine requires its owned uniform packed-query domain')
    if int(header[7])<sum(q):raise ValueError('FA3 combine allocation cannot shrink logical rows')
    splits=fa3_decode_dynamic_splits(q,kv,heads,static,sms)
    return q,kv,heads,static,splits


def _fa3_combine_layouts(header):
    q,kv,heads,static,splits=_fa3_combine_geometry(header)
    tokens=int(header[7]);batch=len(q) if tokens==sum(q) else tokens
    rounded=-(-batch//4)*4
    # Q1 disables causal in flash_api. Speculative Q>1 adds head swizzle.
    vectors=2+int(q[0]>1 and header[6]=='causal')
    layouts=[('cu_query',(batch+1,),'int32',4),
        ('scheduler_metadata',(vectors*rounded+1,),'int32',4),
        ('dynamic_splits',(batch,),'int32',4),('semaphore',(1,),'int32',4),
        ('partial_output',(static,heads,tokens,512),'fp32',4),
        ('partial_lse',(static,heads,tokens),'fp32',4),
        ('output',(tokens,heads,512),header[8],2),('lse',(heads,tokens),'fp32',4)]
    for request,(length,count) in enumerate(zip(q,splits,strict=True)):
        if count>1:
            layouts.extend(((f'partial_output.r{request}',(count,heads,length,512),'fp32',4),
                (f'partial_lse.r{request}',(count,heads,length),'fp32',4)))
    return tuple(layouts)


def standalone_fa3_combine_bindings(kernel,hardware,*,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_fa3_combine_source(kernel):raise NotImplementedError('FA3 combine native ABI is unbound')
    return _standalone_fa3_combine_bindings(_fa3_combine_header(kernel,hardware),call_id,binding_id)


def _standalone_fa3_combine_bindings(header,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    q,kv,heads,static,splits=_fa3_combine_geometry(header)
    layouts=_fa3_combine_layouts(header);allocations=[];views=[];events=[];ready=[]
    roots={};tokens=int(header[7]);batch=len(q) if tokens==sum(q) else tokens
    rounded=-(-batch//4)*4;vectors=2+int(q[0]>1 and header[6]=='causal')
    for role,shape,dtype,width in layouts:
        if role in ('dynamic_splits','semaphore'):
            identity=roots['scheduler_metadata'];offset=(rounded if role=='dynamic_splits' else vectors*rounded)*4
            strides=(4,)
        elif '.r' in role:
            base,request_text=role.split('.r');request=int(request_text);identity=roots[base]
            offset=sum(q[:request])*(512 if base=='partial_output' else 1)*4
            strides=((heads*tokens*512*4,tokens*512*4,512*4,4) if base=='partial_output'
                else (heads*tokens*4,tokens*4,4))
        else:
            identity=f'{binding_id}.{role}';roots[role]=identity;offset=0;strides=dense_strides(shape,width)
            allocation=MemoryAllocation(identity,'logical-local-device','logical-fa3-native',None,
                AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
            allocations.append(allocation)
            events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        views.append(TensorView(role,identity,offset,shape,strides,dtype,width))
    read_roles=('cu_query','dynamic_splits',*(role for role,_,_,_ in layouts if '.r' in role))
    by_role=unique_by(views,'view_id');by_allocation=unique_by(allocations,'allocation_id')
    for role in read_roles:
        view=by_role[role];allocation=by_allocation[view.allocation_id]
        # Only valid dynamic-split/request subviews are available. Reserved
        # static slots and bypassed request partials are never initialized.
        event=ProgramEvent(f'{binding_id}.{role}.available',EventKind.LOGICAL_AVAILABLE,view.allocation_id,
            Unmapped(view.shape,view.dtype,view.element_bytes,f'{binding_id}.{role}.declared_native_input'),
            'fa3-visible',(EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
        events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        read_roles,('output','lse','semaphore'),tuple(ready),0,'offline','fa3-visible')


def _fa3_combine_recipe(header,parents,ready,scope,visibility,storage_prefix,device):
    return _cached_fa3_combine_recipe(header,tuple(sorted(parents.items())),tuple(sorted(ready.items())),scope,visibility,storage_prefix,device)


@runtime_lru_cache(maxsize=8)
def _cached_fa3_combine_recipe(header,parent_records,ready_records,scope,visibility,storage_prefix,device):
    parents,ready=dict(parent_records),dict(ready_records)
    q,kv,heads,static,splits=_fa3_combine_geometry(header)
    selected=[];allocations=[];events=[];families=[];offset=0
    evidence=('Pinned flash_fwd_combine_kernel.h: rows8/value128/stages4; head-major packed rows.',
        'LSE is loaded independently by each value-column CTA; final LSE is stored only by k_block==0.',
        'First min(3,dynamic_splits) partial-O loads are unconditional; tail loads and scale reads wait on the actual INT32 max-valid-split READ.',
        'Max-valid-split is derived from non-minus-infinity LSE values, may be -1, and remains unbound without actual READ values.',
        'Scratch views name logical versions of staged values, not simultaneous native SRAM capacity or physical ring addresses.',
        'Epilogue store_zero writes minus-infinity LSE and may leave partial-O indeterminate; boundary availability permits native loads, not an assertion that all O slots were written or numerically valid.',
        'Only positive normalized LSE scales contribute to register accumulation; indeterminate O from empty splits is never assigned a value.')
    def dep(name,kind=EventKind.READ_READY,axes=()):return TileDependency(name,kind,axes)
    def coordinate(axis,*terms,offset=0):
        return LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(i,c) for i,c in terms),offset)
    def child(identity,role,shape,mapping):
        parent=parents[role]
        view=LogicalSelectionView(identity,parent.allocation_id,shape,parent.dtype,parent.element_bytes,
            f'fa3_combine.{identity}',role,mapping)
        selected.append(view);return view
    def scratch(identity,shape,dtype,width):
        allocation_id=f'{storage_prefix}.{identity}'
        allocation=MemoryAllocation(allocation_id,device,'logical-fa3-versioned-state',
            prod(shape)*width,AddressLevel.OBJECT,None,None,f'{allocation_id}.allocated',None)
        view=TensorView(identity,allocation_id,0,shape,dense_strides(shape,width),dtype,width)
        allocations.append(allocation);selected.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,allocation_id,None,None,()))
        return view
    for request,(length,count) in enumerate(zip(q,splits,strict=True)):
        if count<=1:
            identity=f'{scope.scope_id}.r{request}.bypass'
            controls=[]
            for role,shape,mapping in (
                    ('cu_query',(2,),(coordinate(0,(0,1),offset=request),)),
                    ('dynamic_splits',(1,),(coordinate(0,offset=request),))):
                view=child(identity+'.'+role,role,shape,mapping)
                controls.append(TileAccessTemplate(role,view.view_id,Operation.READ,('row_block','column'),
                    (None,),(ready[role],),(),evidence,unmapped=True))
            family_scope=replace(scope,scope_id=identity,op_ids=(*header,str(request),'bypass'))
            families.append(TileProgram(identity,family_scope,(TileAxis('row_block',length*heads,8),TileAxis('column',4,1)),
                tuple(controls),visibility,evidence))
            offset+=length;continue
        pre=min(3,count)
        for packed in range(length*heads):
            query,head=packed%length,packed//length
            for column in range(4):
                identity=f'{scope.scope_id}.r{request}.row{packed}.col{column}'
                accesses=[];axes=[TileAxis('split',count,1),TileAxis('preload',pre,1),
                    TileAxis('compute_prefix',pre,1,'last_valid',0,True,count-1)]
                def access(name,view,operation,loops,view_axes,dependencies=(),input_refs=(),unmapped=False,constant=None):
                    accesses.append(TileAccessTemplate(name,view.view_id,operation,loops,view_axes,input_refs,
                        dependencies,evidence,unmapped=unmapped,constant_value=constant))
                cu=child(identity+'.cu_query','cu_query',(2,),(coordinate(0,(0,1),offset=request),))
                ds=child(identity+'.dynamic_splits','dynamic_splits',(1,),(coordinate(0,offset=request),))
                access('cu_query',cu,Operation.READ,(),(None,),input_refs=(ready['cu_query'],),unmapped=True)
                access('dynamic_splits',ds,Operation.READ,(),(None,),input_refs=(ready['dynamic_splits'],),unmapped=True)
                controls=(dep('cu_query'),dep('dynamic_splits'))
                lse=child(identity+'.partial_lse',f'partial_lse.r{request}',(count,),(
                    coordinate(0,(0,1)),coordinate(1,offset=head),coordinate(2,offset=query)))
                access('partial_lse',lse,Operation.READ,('split',),('split',),controls,
                    (ready[f'partial_lse.r{request}'],),True)
                scale=scratch(identity+'.scale',(count,),'fp32',4)
                last=scratch(identity+'.max_valid',(1,),'int32',4)
                reduced=(dep('partial_lse',axes=('split',)),)
                access('scale_write',scale,Operation.WRITE,(),(None,),reduced)
                access('max_valid_write',last,Operation.WRITE,(),(None,),reduced)
                access('last_valid',last,Operation.READ,(),(None,),(dep('max_valid_write',EventKind.WRITE_VISIBLE),))
                first=child(identity+'.O_prefix',f'partial_output.r{request}',(pre,128),(
                    coordinate(0,(0,1)),coordinate(1,offset=head),coordinate(2,offset=query),
                    coordinate(3,(1,1),offset=column*128)))
                access('O_preload',first,Operation.READ,('preload',),('preload',None),controls,
                    (ready[f'partial_output.r{request}'],),True)
                prefix=scratch(identity+'.staged_prefix',(pre,128),'fp32',4)
                access('prefix_stage',prefix,Operation.WRITE,(),(None,None),(dep('O_preload',axes=('preload',)),))
                access('prefix_compute',prefix,Operation.READ,('compute_prefix',),('compute_prefix',None),
                    (dep('last_valid'),dep('prefix_stage',EventKind.WRITE_VISIBLE)))
                # A strided contained view keeps the complete scale producer;
                # no guessed scale sign or max-valid value is inserted.
                scale_prefix=TensorView(identity+'.scale_prefix',scale.allocation_id,0,(pre,),(4,),'fp32',4)
                selected.append(scale_prefix)
                access('scale_prefix',scale_prefix,Operation.READ,('compute_prefix',),('compute_prefix',),
                    (dep('last_valid'),dep('scale_write',EventKind.WRITE_VISIBLE)))
                output_deps=[*controls,dep('last_valid'),dep('prefix_compute',axes=('compute_prefix',)),dep('scale_prefix',axes=('compute_prefix',))]
                if count>pre:
                    tail=count-pre;axes.append(TileAxis('tail',tail,1,'last_valid',pre,True,count-1))
                    rest=child(identity+'.O_tail',f'partial_output.r{request}',(tail,128),(
                        coordinate(0,(0,1),offset=pre),coordinate(1,offset=head),coordinate(2,offset=query),
                        coordinate(3,(1,1),offset=column*128)))
                    tail_state=scratch(identity+'.staged_tail',(tail,128),'fp32',4)
                    access('O_tail',rest,Operation.READ,('tail',),('tail',None),(*controls,dep('last_valid')),
                        (ready[f'partial_output.r{request}'],),True)
                    access('tail_stage',tail_state,Operation.WRITE,('tail',),('tail',None),
                        (dep('last_valid'),dep('O_tail')))
                    access('tail_compute',tail_state,Operation.READ,('tail',),('tail',None),
                        (dep('last_valid'),dep('tail_stage',EventKind.WRITE_VISIBLE)))
                    scale_tail=TensorView(identity+'.scale_tail',scale.allocation_id,pre*4,(tail,),(4,),'fp32',4)
                    selected.append(scale_tail)
                    access('scale_tail',scale_tail,Operation.READ,('tail',),('tail',),
                        (dep('last_valid'),dep('scale_write',EventKind.WRITE_VISIBLE)))
                    output_deps.extend((dep('tail_compute',axes=('tail',)),dep('scale_tail',axes=('tail',))))
                output=child(identity+'.output','output',(128,),(
                    coordinate(0,offset=offset+query),coordinate(1,offset=head),coordinate(2,(0,1),offset=column*128)))
                access('output',output,Operation.WRITE,(),(None,),tuple(output_deps),unmapped=True)
                if column==0:
                    final_lse=child(identity+'.lse','lse',(1,),(
                        coordinate(0,offset=head),coordinate(1,offset=offset+query)))
                    access('lse',final_lse,Operation.WRITE,(),(None,),reduced,unmapped=True)
                family_scope=replace(scope,scope_id=identity,op_ids=(*header,str(request),str(packed),str(column)))
                families.append(TileProgram(identity,family_scope,tuple(axes),tuple(accesses),visibility,evidence))
        offset+=length
    # The last grid CTA resets the semaphore before any virtual-batch, row,
    # or split early return, including calls in which every request bypasses.
    identity=f'{scope.scope_id}.reset';family_scope=replace(scope,scope_id=identity,op_ids=(*header,'reset'))
    reset=TileAccessTemplate('semaphore_reset','semaphore',Operation.WRITE,(),(None,),(),(),evidence,
        unmapped=True,constant_value=ConstantValue.ZERO)
    families.append(TileProgram(identity,family_scope,(TileAxis('reset',1,1),),(reset,),visibility,evidence))
    return tuple(allocations),tuple(selected),tuple(events),tuple(families)


def build_fa3_combine_source(kernel,hardware,bindings,granularity,*,program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_fa3_combine_source(kernel) or not isinstance(bindings,ElementwiseBindings):
        raise TypeError('FA3 combine requires its native typed bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown FA3 granularity')
    header=_fa3_combine_header(kernel,hardware)
    expected=standalone_fa3_combine_bindings(kernel,hardware,call_id=bindings.call_id,binding_id=bindings.binding_id)
    if (bindings.read_views,bindings.write_views)!=(expected.read_views,expected.write_views):
        raise ValueError('FA3 combine native roles mismatch')
    parents=unique_by(bindings.views,'view_id')
    _validate_fa3_combine_parents(header,parents)
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-fa3-combine')
    ready=dict(zip(bindings.read_views,bindings.input_ready,strict=True))
    public_allocations=unique_by(bindings.allocations,'allocation_id')
    device=public_allocations[parents['output'].allocation_id].device_id
    allocations,selected,events,families=_fa3_combine_recipe(header,parents,ready,scope,bindings.visibility_domain,bindings.binding_id,device)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Native global payload selections and versioned logical scratch; no physical SRAM footprint/ring addresses or issued-thread duplication.',
            'Actual max-valid-split READ values remain unknown; conditional traffic is an envelope, never an exact full-split count.',
            'Legacy full-split FP32 input budget and all-request BF16 output budget are retained only as explicit unmodified price projections.',
            'Scheduler/prepare production is an external interface at this source boundary.')),
        (*bindings.allocations,*allocations),(*bindings.views,*selected),(scope,),(*bindings.events,*events),(),families)


def _validate_fa3_combine_parents(header,parents):
    reference=_standalone_fa3_combine_bindings(header,'geometry','geometry')
    expected=unique_by(reference.views,'view_id')
    if set(parents)!=set(expected):raise ValueError('FA3 combine native parents mismatch')
    actual_aliases={};expected_aliases={}
    for role,view in parents.items():
        other=expected[role]
        if (not isinstance(view,TensorView) or
                (view.shape,view.dtype,view.element_bytes,view.offset_bytes,view.strides_bytes)!=
                (other.shape,other.dtype,other.element_bytes,other.offset_bytes,other.strides_bytes)):
            raise ValueError('FA3 combine native shape/encoding/stride/offset mismatch')
        if view.allocation_id not in actual_aliases:actual_aliases[view.allocation_id]=[]
        if other.allocation_id not in expected_aliases:expected_aliases[other.allocation_id]=[]
        actual_aliases[view.allocation_id].append(role);expected_aliases[other.allocation_id].append(role)
    if sorted(actual_aliases.values())!=sorted(expected_aliases.values()):
        raise ValueError('FA3 combine requires actual native parent/subview aliases')


def _fa3_combine_prediction_demand(header):
    """Evaluate the existing request/row/value-column budget analytically."""
    q,kv,heads,static,splits=_fa3_combine_geometry(header)
    totals={name:0 for name in ('partial_lse','O_preload','O_tail','output','lse',
        'semaphore_reset','cu_query','dynamic_splits')}
    totals['semaphore_reset']=4;active=0
    for length,count in zip(q,splits,strict=True):
        rows=length*heads;ctas=-(-rows//8)*4
        if count<=1:
            totals['cu_query']+=ctas*8;totals['dynamic_splits']+=ctas*4
            continue
        active+=ctas;pre=min(3,count)
        totals['partial_lse']+=rows*4*count*4
        totals['O_preload']+=rows*4*pre*128*4
        totals['O_tail']+=rows*4*(count-pre)*128*4
        totals['output']+=rows*512*2;totals['lse']+=rows*4
        totals['cu_query']+=rows*4*8;totals['dynamic_splits']+=rows*4*4
    return {'totals':totals,'projected_read_bytes':totals['O_preload']+totals['O_tail']+totals['partial_lse']//4,
        'projected_write_bytes':sum(q)*heads*512*2,'active_ctas':active,'max_splits':max(splits),
        'partial_output_exact_bytes':None if any(count>3 for count in splits) else totals['O_preload']}


def fa3_combine_prediction_work(kernel,hardware,bindings):
    """Validate actual native parents without constructing private row graphs."""
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_access import covers,validate_storage_bindings
    if kernel.hardware_id!=hardware.hardware_id or not has_fa3_combine_source(kernel):
        raise ValueError('FA3 analytical combine requires its existing source ABI')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('FA3 combine requires typed public operands')
    bindings.__post_init__();header=_fa3_combine_header(kernel,hardware)
    layouts=_fa3_combine_layouts(header)
    reads=('cu_query','dynamic_splits',*(role for role,*_ in layouts if '.r' in role))
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(reads,('output','lse','semaphore'),len(reads)):
        raise ValueError('FA3 combine native operand/readiness roles mismatch')
    parents=unique_by(bindings.views,'view_id');_validate_fa3_combine_parents(header,parents)
    validate_storage_bindings(bindings.allocations,bindings.views,bindings.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual FA3 combine public boundary; private row/split state remains an explicit export.',)))
    events=unique_by(bindings.events,'event_id')
    for role,ref in zip(reads,bindings.input_ready,strict=True):
        if ref.node_id not in events:raise ValueError('FA3 combine input readiness is missing')
        event=events[ref.node_id];parent=parents[role]
        logical=(ref.kind is EventKind.LOGICAL_AVAILABLE and isinstance(event.region,Unmapped)
            and (event.region.logical_shape,event.region.dtype,event.region.element_bytes)==
            (parent.shape,parent.dtype,parent.element_bytes))
        if (event.kind is not ref.kind or ref.allow_skipped_read
                or ref.kind not in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE,EventKind.LOGICAL_AVAILABLE)
                or event.allocation_id!=parent.allocation_id or event.visibility_domain!=bindings.visibility_domain
                or event.view_id is not None and event.view_id!=parent.view_id
                or not (logical or event.region is not None and covers(event.region,parent.region()))):
            raise ValueError('FA3 combine requires actual partial/control parent readiness')
    return _fa3_combine_prediction_demand(header)


def fa3_combine_source_demands(program,*,expected_kernel=None,hardware=None,expected_bindings=None):
    program.__post_init__()
    if program.hardware_id!='nvidia_h200' or len(program.scopes)!=1 or program.accesses or not program.tile_programs:
        raise ValueError('FA3 combine requires its actual compressed production source')
    scope,=program.scopes;header=scope.op_ids;q,kv,heads,static,splits=_fa3_combine_geometry(header)
    if expected_kernel is not None:
        if hardware is None or header!=_fa3_combine_header(expected_kernel,hardware):
            raise ValueError('FA3 combine source belongs to a foreign kernel')
    all_views=unique_by(program.views,'view_id');parents={role:all_views[role] for role,_,_,_ in _fa3_combine_layouts(header)}
    _validate_fa3_combine_parents(header,parents)
    ready={}
    for family in program.tile_programs:
        for access in family.accesses:
            for ref in access.input_ready:
                role=('partial_lse.r'+family.scope.op_ids[len(header)] if access.access_id=='partial_lse' else
                    'partial_output.r'+family.scope.op_ids[len(header)] if access.access_id in ('O_preload','O_tail') else access.access_id)
                if role in ready and ready[role]!=ref:raise ValueError('FA3 combine input readiness changed across contexts')
                ready[role]=ref
    public_allocations=unique_by(program.allocations,'allocation_id')
    device=public_allocations[parents['output'].allocation_id].device_id
    allocations,selected,events,families=_fa3_combine_recipe(header,parents,ready,scope,program.tile_programs[0].visibility_domain,program.binding_id,device)
    if (program.tile_programs!=families or set(all_views)!={*parents,*(view.view_id for view in selected)}
            or any(all_views[v.view_id]!=v for v in selected)
            or allocations and program.allocations[-len(allocations):]!=allocations
            or events and program.events[-len(events):]!=events):
        raise ValueError('FA3 combine lost native selections, staging, LSE or max-valid dependencies')
    if expected_bindings is not None:
        nviews=len(expected_bindings.views);nalloc=len(expected_bindings.allocations);nevents=len(expected_bindings.events)
        if (program.binding_id!=expected_bindings.binding_id or scope.call_id!=expected_bindings.call_id
                or scope.rank!=expected_bindings.rank or scope.stream!=expected_bindings.stream
                or program.views[:nviews]!=expected_bindings.views or program.allocations[:nalloc]!=expected_bindings.allocations
                or program.events[:nevents]!=expected_bindings.events
                or any(ready[role]!=ref for role,ref in zip(expected_bindings.read_views,expected_bindings.input_ready,strict=True) if role in ready)):
            raise ValueError('FA3 combine source does not consume actual caller storage/readiness')
    return _fa3_combine_prediction_demand(header)


_FA3_COMBINE_WORK=ContextVar('fa3_combine_source_work')


@contextmanager
def fa3_combine_source_work(kernel,work):
    token=_FA3_COMBINE_WORK.set((kernel,work))
    try:yield
    finally:_FA3_COMBINE_WORK.reset(token)


def current_fa3_combine_source_work(kernel):
    context=copy_context()
    if _FA3_COMBINE_WORK not in context:return None
    current,work=context[_FA3_COMBINE_WORK]
    if current!=kernel:raise ValueError('FA3 combine source crossed its kernel boundary')
    return work


def logical_mla_request_extents(program):
    query=tuple(view for view in program.views if view.view_id=='query' or view.view_id.startswith('query.request'))
    caches=tuple(view for view in program.views if view.view_id=='compressed_kv' or view.view_id.startswith('compressed_kv.request'))
    if not query or len(query)!=len(caches):raise ValueError('MLA source must retain every actual request query/cache extent')
    heads=query[0].shape[1]
    if any(view.shape[1]!=heads for view in query):raise ValueError('MLA request query heads differ')
    return tuple(view.shape[0] for view in query),tuple(view.shape[0] for view in caches),heads


def flashinfer_mla_source_plan(program,*,expected_kernel,num_sms):
    """flashinfer scheduler.cuh MLAPlan (causal): per-cluster KV tile work.

    Packed query rows (q_len x heads) are cut into cluster tiles of
    cluster_size x 64 rows; each tile's causal KV extent is split into
    chunks of at most kv_len_limit tokens handed to the least-cost cluster
    (cost_function = 2 x rows + kv).  Returns, per cluster, the KV tiles its
    CTAs iterate and the number of work items.
    """

    import heapq,math
    from src.kernel.memory_path_solver import logical_interface_source_bytes
    logical_interface_source_bytes(program,expected_kernel=expected_kernel)
    kernel=expected_kernel
    q_lens,kv_lens,heads=logical_mla_request_extents(program)
    tile_q = kernel.launch.tiling.tile_shape["q"]
    tile_kv = kernel.launch.tiling.tile_shape["kv"]
    packed = [q * heads for q in q_lens]
    cluster_size = 2 if sum(packed) // len(packed) > 64 else 1
    clusters = num_sms // cluster_size
    cluster_tile_q = cluster_size * tile_q

    def causal_end(q_len, kv_len, index, tiles):
        if index + 1 == tiles:
            return kv_len
        return max(min(kv_len - q_len + math.ceil(
            (index + 1) * cluster_tile_q / heads), kv_len), 0)

    work = []
    for q_len, kv_len, rows in zip(q_lens, kv_lens, packed):
        tiles = math.ceil(rows / cluster_tile_q)
        for index in range(tiles - 1, -1, -1):
            work.append(causal_end(q_len, kv_len, index, tiles))
    average = max(math.ceil(sum(work) / clusters), 1)
    if average <= 8:
        limit = 32
    elif average <= 16:
        limit = 64
    elif average <= 32:
        limit = 128
    elif average <= 64:
        limit = 192
    else:
        limit = math.ceil(average / 256) * 256
    heap = [(0.0, index) for index in range(clusters)]
    kv_tiles = [0] * clusters
    items = [0] * clusters
    # DevicePersistentMergeStates (mla.cuh): each merge CTA owns row_chunk
    # packed rows of one split query tile, 4 rows per pass (256 threads / 64
    # threads per 512-wide row), and walks every KV chunk of the tile once per
    # pass.
    merge_iterations = []
    rows_per_pass = 256 // (
        kernel.launch.tiling.tile_shape["value_head"] // 8)
    tile_rows = []
    for q_len, kv_len, rows in zip(q_lens, kv_lens, packed):
        tiles = math.ceil(rows / cluster_tile_q)
        for index in range(tiles - 1, -1, -1):
            tile_rows.append(min(cluster_tile_q, rows - index * cluster_tile_q))
    for remaining, row_tile in zip(work, tile_rows):
        if remaining > limit:
            qo_chunks = max(remaining * cluster_size // limit, 1)
            row_chunk = math.ceil(row_tile / qo_chunks)
            kv_chunks = math.ceil(remaining / limit)
            merge_iterations.append(
                math.ceil(row_chunk / rows_per_pass) * kv_chunks)
    for remaining in work:
        start = 0
        while True:
            cost, index = heapq.heappop(heap)
            length = min(remaining, limit)
            heapq.heappush(
                heap, (cost + 2 * cluster_tile_q + length, index))
            kv_tiles[index] += (
                math.ceil((start + length) / tile_kv) - start // tile_kv
                if length else 0)
            items[index] += 1
            remaining -= length
            start += length
            if remaining <= 0:
                break
    return kv_tiles, items, cluster_size, tuple(merge_iterations)



def fa3_mla_decode_partition_blocks(q_lens,kv_lens,heads,static_splits,num_sms,block_n):
    splits=fa3_decode_source_splits(q_lens,kv_lens,heads,static_splits,num_sms)
    tiles=[]
    for kv_len,split_count in zip(kv_lens,splits):
        count=ceil(kv_len/block_n);per_split=ceil(count/split_count)
        tiles.extend(max(0,min(count,(index+1)*per_split)-index*per_split) for index in range(split_count))
    return tiles


def fa3_mla_decode_source_tiles(program,*,expected_kernel,num_sms):
    from src.kernel.memory_path_solver import logical_interface_source_bytes
    logical_interface_source_bytes(program,expected_kernel=expected_kernel)
    q_lens,kv_lens,heads=logical_mla_request_extents(program)
    return fa3_mla_decode_partition_blocks(q_lens,kv_lens,heads,expected_kernel.launch.tiling.num_kv_splits,
        num_sms,expected_kernel.launch.tiling.tile_shape['kv'])


def fa3_split_prefill_tensor_parent_plan(main, combine, hardware, *, binding_id):
    """Plan the pinned BF16192/128 split call's public tensor parents.

    This is a storage/producer plan, not a MemoryAccessProgram or input-ready
    declaration. Actual main/combine access events must establish readiness.
    Reserved partial O may be prefetched without being numerically valid.
    Scheduler/control buffers and their prepare/reset edges are separate.
    """
    from src.kernel.predictors.flash_attn import (
        _fa3_split_prefill_work, _fa3_gqa_observed_work)
    from src.kernel.kernel_spec import KernelKind, KernelSpec

    if not isinstance(main, KernelSpec) or not isinstance(combine, KernelSpec):
        raise TypeError('Split-prefill parent planning requires normalized KernelSpecs')
    if (not binding_id or main.kind is not KernelKind.FLASH_ATTN
            or combine.kind is not KernelKind.FLASH_ATTN
            or main.variant != 'mla_prefill' or combine.variant != 'combine'
            or main.execution_support != 'supported' or combine.execution_support != 'supported'
            or main.hardware_id != hardware.hardware_id
            or combine.hardware_id != hardware.hardware_id
            or main.launch.algorithm != 'vllm_fa3_sm90_mla_prefill_split_partial'
            or combine.launch.algorithm != 'vllm_fa3_sm90_mla_prefill_combine_partial'
            or main.dtype != combine.dtype or main.dtype != 'bf16'
            or main.launch.tiling.num_kv_splits != combine.launch.tiling.num_kv_splits
            or main.shape['num_query_heads'] != combine.shape['num_query_heads']
            or tuple(main.attributes['q_lens']) != tuple(combine.attributes['q_lens'])
            or tuple(main.attributes['kv_lens']) != tuple(combine.attributes['kv_lens'])):
        raise ValueError('Split-prefill parents require one matching source call')
    for kernel in (main, combine):
        if (kernel.attributes['causal'] is not False
                or any(name in kernel.attributes for name in (
                    'graph_logical_tokens', 'graph_model_input_tokens', 'graph_runtime_mode',
                    'model_input_shape', 'model_input_q_lens', 'model_input_kv_lens'))):
            raise ValueError('Split-prefill parents require unpadded source requests')
    if (main.attributes['full_cuda_graphs'] is not False
            or main.attributes['max_cudagraph_capture_size'] != 0):
        raise ValueError('Split-prefill parents require an eager source call')
    mw = _fa3_split_prefill_work(main, hardware)
    cw = _fa3_gqa_observed_work(combine, hardware)
    if mw['splits'] != cw['splits']:
        raise ValueError('Split-prefill main and combine have different dynamic splits')
    q_lens = tuple(main.attributes['q_lens'])
    kv_lens = tuple(main.attributes['kv_lens'])
    tokens, keys, heads = sum(q_lens), sum(kv_lens), main.shape['num_query_heads']
    static = main.launch.tiling.num_kv_splits
    # flash_api.cpp varlen allocations: O is token-major; LSE is head-major.
    layouts = (('query', (tokens, heads, 192), 'bf16', 2),
        ('key', (keys, heads, 192), 'bf16', 2),
        ('value', (keys, heads, 128), 'bf16', 2),
        ('partial_output', (static, heads, tokens, 128), 'fp32', 4),
        ('partial_lse', (static, heads, tokens), 'fp32', 4),
        ('output', (tokens, heads, 128), 'bf16', 2),
        ('lse', (heads, tokens), 'fp32', 4))
    parents = {role: TensorView(role, f'{binding_id}.{role}', 0, shape,
        dense_strides(shape, width), dtype, width) for role, shape, dtype, width in layouts}
    views = list(parents.values())
    main_writes, combine_reads, combine_writes, final_producers = [], [], [], []

    def subview(role, parent_role, offset, shape, strides):
        parent = parents[parent_role]
        view = TensorView(role, parent.allocation_id, offset, shape, strides,
            parent.dtype, parent.element_bytes)
        views.append(view)
        return role

    prefix = 0
    for request, (q, kv, count) in enumerate(zip(q_lens, kv_lens, mw['splits'], strict=True)):
        output = subview(f'output.r{request}', 'output', prefix * heads * 128 * 2,
            (q, heads, 128), parents['output'].strides_bytes)
        lse = subview(f'lse.r{request}', 'lse', prefix * 4,
            (heads, q), parents['lse'].strides_bytes)
        producer = 'main' if count == 1 else 'combine'
        final_producers.extend(((output, producer), (lse, producer)))
        if count == 1:
            main_writes.extend((output, lse))
        else:
            # Noncausal ranges are consecutive; all M tiles share the same
            # finite-LSE split prefix, including the last partial Q tile.
            panels = -(-kv // 128)
            per_split = -(-panels // count)
            valid = -(-panels // per_split)
            strides_o = parents['partial_output'].strides_bytes
            strides_lse = parents['partial_lse'].strides_bytes
            partial_lse = subview(f'partial_lse.r{request}', 'partial_lse', prefix * 4,
                (count, heads, q), strides_lse)
            partial_o = subview(f'partial_output.valid.r{request}', 'partial_output', prefix * 128 * 4,
                (valid, heads, q, 128), strides_o)
            main_writes.extend((partial_o, partial_lse))
            combine_reads.append(partial_lse)
            # First three prefetches are unconditional; a prefetch view is
            # not evidence of a main WRITE or finite-LSE validity.
            prefetch = min(3, count)
            combine_reads.append(subview(f'partial_output.prefetch.r{request}', 'partial_output',
                prefix * 128 * 4, (prefetch, heads, q, 128), strides_o))
            if valid > prefetch:
                combine_reads.append(subview(f'partial_output.tail.r{request}', 'partial_output',
                    prefix * 128 * 4 + prefetch * strides_o[0],
                    (valid - prefetch, heads, q, 128), strides_o))
            combine_writes.extend((output, lse))
        prefix += q
    return {'views': tuple(views), 'main_reads': ('query', 'key', 'value'),
        'main_writes': tuple(main_writes), 'combine_reads': tuple(combine_reads),
        'combine_writes': tuple(combine_writes), 'final_producers': tuple(final_producers),
        'dynamic_splits': mw['splits'], 'readiness_bound': False}


def _fa3_split_prefill_combine_geometry(header):
    from src.kernel.predictors.flash_attn import _fa3_request_tile_splits
    if len(header)!=9 or header[0]!='fa3_split_prefill_combine' or header[6]!='noncausal' or header[8]!='bf16':
        raise ValueError('FA3 split-prefill combine requires its own source header')
    q,kv=tuple(map(int,header[1].split(','))),tuple(map(int,header[2].split(',')))
    heads,static,sms=map(int,header[3:6])
    if (not q or len(q)!=len(kv) or len(q)>992 or min(*q,*kv,heads,sms)<=0 or static<=1 or static>256
            or any(a>b for a,b in zip(q,kv,strict=True))
            or int(header[7])!=sum(q)):
        raise ValueError('FA3 split-prefill combine requires actual unpadded paired requests')
    splits=_fa3_request_tile_splits(q,kv,static,sms,block_m=128,block_n=128,ratio=1,kv_heads=heads)
    return q,kv,heads,static,splits


def _fa3_split_prefill_combine_header(main,combine,hardware):
    fa3_split_prefill_tensor_parent_plan(main,combine,hardware,binding_id='source_admission')
    return ('fa3_split_prefill_combine',','.join(map(str,main.attributes['q_lens'])),
        ','.join(map(str,main.attributes['kv_lens'])),str(main.shape['num_query_heads']),
        str(main.launch.tiling.num_kv_splits),str(int(hardware.num_compute_units)),
        'noncausal',str(sum(main.attributes['q_lens'])),'bf16')


def _fa3_split_prefill_combine_layouts(header):
    q,kv,heads,static,splits=_fa3_split_prefill_combine_geometry(header)
    tokens=int(header[7]);batch=len(q) if tokens==sum(q) else tokens
    rounded=-(-batch//4)*4
    # Q1 disables causal in flash_api. Speculative Q>1 adds head swizzle.
    vectors=2+int(q[0]>1 and header[6]=='causal')
    layouts=[('cu_query',(batch+1,),'int32',4),
        ('scheduler_metadata',(vectors*rounded+1,),'int32',4),
        ('dynamic_splits',(batch,),'int32',4),('semaphore',(1,),'int32',4),
        ('partial_output',(static,heads,tokens,128),'fp32',4),
        ('partial_lse',(static,heads,tokens),'fp32',4),
        ('output',(tokens,heads,128),header[8],2),('lse',(heads,tokens),'fp32',4)]
    for request,(length,count) in enumerate(zip(q,splits,strict=True)):
        if count>1:
            layouts.extend(((f'partial_output.r{request}',(count,heads,length,128),'fp32',4),
                (f'partial_lse.r{request}',(count,heads,length),'fp32',4)))
    return tuple(layouts)


def standalone_fa3_split_prefill_combine_bindings(main,combine,hardware,*,call_id,binding_id):
    return _standalone_fa3_split_prefill_combine_bindings(
        _fa3_split_prefill_combine_header(main,combine,hardware),call_id,binding_id)


def _standalone_fa3_split_prefill_combine_bindings(header,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    q,kv,heads,static,splits=_fa3_split_prefill_combine_geometry(header)
    layouts=_fa3_split_prefill_combine_layouts(header);allocations=[];views=[];events=[];ready=[]
    roots={};tokens=int(header[7]);batch=len(q) if tokens==sum(q) else tokens
    rounded=-(-batch//4)*4;vectors=2+int(q[0]>1 and header[6]=='causal')
    for role,shape,dtype,width in layouts:
        if role in ('dynamic_splits','semaphore'):
            identity=roots['scheduler_metadata'];offset=(rounded if role=='dynamic_splits' else vectors*rounded)*4
            strides=(4,)
        elif '.r' in role:
            base,request_text=role.split('.r');request=int(request_text);identity=roots[base]
            offset=sum(q[:request])*(128 if base=='partial_output' else 1)*4
            strides=((heads*tokens*128*4,tokens*128*4,128*4,4) if base=='partial_output'
                else (heads*tokens*4,tokens*4,4))
        else:
            identity=f'{binding_id}.{role}';roots[role]=identity;offset=0;strides=dense_strides(shape,width)
            allocation=MemoryAllocation(identity,'logical-local-device','logical-fa3-native',None,
                AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
            allocations.append(allocation)
            events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        views.append(TensorView(role,identity,offset,shape,strides,dtype,width))
    read_roles=('cu_query','dynamic_splits',*(role for role,_,_,_ in layouts if '.r' in role))
    by_role=unique_by(views,'view_id');by_allocation=unique_by(allocations,'allocation_id')
    for role in read_roles:
        view=by_role[role];allocation=by_allocation[view.allocation_id]
        # Only valid dynamic-split/request subviews are available. Reserved
        # static slots and bypassed request partials are never initialized.
        event=ProgramEvent(f'{binding_id}.{role}.available',EventKind.LOGICAL_AVAILABLE,view.allocation_id,
            Unmapped(view.shape,view.dtype,view.element_bytes,f'{binding_id}.{role}.declared_native_input'),
            'fa3-visible',(EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
        events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        read_roles,('output','lse','semaphore'),tuple(ready),0,'offline','fa3-visible')


def _validate_fa3_split_prefill_parent_geometry(role,view,expected):
    """Preserve caller row/head pitches accepted by flash_api.cpp.

    Native scratch and LSE remain fresh dense allocations. Public tensors
    may be channel slices of a larger projection; no materializing copy is
    implied by their non-dense outer dimensions.
    """
    from src.kernel.memory_access import Strided
    if (not isinstance(view,TensorView)
            or (view.shape,view.dtype,view.element_bytes)!=(expected.shape,expected.dtype,expected.element_bytes)
            or view.region()!=Strided(view.offset_bytes,view.shape,view.strides_bytes,view.element_bytes)):
        raise ValueError('Split-prefill prediction parent shape/encoding/stride differs')
    if role in ('query','key','value','output'):
        rows,heads,channels=view.shape
        row_stride,head_stride,inner_stride=view.strides_bytes
        if (inner_stride!=view.element_bytes or view.offset_bytes%view.element_bytes
                or head_stride<channels*view.element_bytes
                or row_stride<(heads-1)*head_stride+channels*view.element_bytes):
            raise ValueError('Split-prefill public parent requires nonoverlapping rows and contiguous channels')
    elif role in ('cu_query','cu_key'):
        if view.strides_bytes!=(4,) or view.offset_bytes%4:
            raise ValueError('Split-prefill cumulative lengths require contiguous INT32 storage')
    elif (view.offset_bytes,view.strides_bytes)!=(expected.offset_bytes,expected.strides_bytes):
        raise ValueError('Split-prefill native scratch shape/encoding/stride/offset differs')


def _validate_fa3_split_prefill_combine_parents(header,parents):
    reference=_standalone_fa3_split_prefill_combine_bindings(header,'geometry','geometry')
    expected=unique_by(reference.views,'view_id')
    if set(parents)!=set(expected):raise ValueError('FA3 combine native parents mismatch')
    actual_aliases={};expected_aliases={}
    for role,view in parents.items():
        other=expected[role]
        _validate_fa3_split_prefill_parent_geometry(role,view,other)
        if view.allocation_id not in actual_aliases:actual_aliases[view.allocation_id]=[]
        if other.allocation_id not in expected_aliases:expected_aliases[other.allocation_id]=[]
        actual_aliases[view.allocation_id].append(role);expected_aliases[other.allocation_id].append(role)
    if sorted(actual_aliases.values())!=sorted(expected_aliases.values()):
        raise ValueError('FA3 combine requires actual native parent/subview aliases')


def _fa3_split_prefill_combine_recipe(header,parent_records,ready_records,scope,visibility,storage_prefix,device):
    parents,ready=dict(parent_records),dict(ready_records)
    q,kv,heads,static,splits=_fa3_split_prefill_combine_geometry(header)
    selected=[];allocations=[];events=[];families=[];offset=0
    evidence=('Pinned flash_fwd_combine_kernel.h: rows8/value128/stages4; head-major packed rows.',
        'LSE is loaded independently by each value-column CTA; final LSE is stored only by k_block==0.',
        'First min(3,dynamic_splits) partial-O loads are unconditional; tail loads and scale reads wait on the actual INT32 max-valid-split READ.',
        'Max-valid-split is derived from non-minus-infinity LSE values, may be -1, and remains unbound without actual READ values.',
        'Scratch views name logical versions of staged values, not simultaneous native SRAM capacity or physical ring addresses.',
        'Epilogue store_zero writes minus-infinity LSE and may leave partial-O indeterminate; boundary availability permits native loads, not an assertion that all O slots were written or numerically valid.',
        'Only positive normalized LSE scales contribute to register accumulation; indeterminate O from empty splits is never assigned a value.')
    def dep(name,kind=EventKind.READ_READY,axes=()):return TileDependency(name,kind,axes)
    def coordinate(axis,*terms,offset=0):
        return LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(i,c) for i,c in terms),offset)
    def child(identity,role,shape,mapping):
        parent=parents[role]
        view=LogicalSelectionView(identity,parent.allocation_id,shape,parent.dtype,parent.element_bytes,
            f'fa3_combine.{identity}',role,mapping)
        selected.append(view);return view
    def scratch(identity,shape,dtype,width):
        allocation_id=f'{storage_prefix}.{identity}'
        allocation=MemoryAllocation(allocation_id,device,'logical-fa3-versioned-state',
            prod(shape)*width,AddressLevel.OBJECT,None,None,f'{allocation_id}.allocated',None)
        view=TensorView(identity,allocation_id,0,shape,dense_strides(shape,width),dtype,width)
        allocations.append(allocation);selected.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,allocation_id,None,None,()))
        return view
    for request,(length,count) in enumerate(zip(q,splits,strict=True)):
        if count<=1:
            identity=f'{scope.scope_id}.r{request}.bypass'
            controls=[]
            for role,shape,mapping in (
                    ('cu_query',(2,),(coordinate(0,(0,1),offset=request),)),
                    ('dynamic_splits',(1,),(coordinate(0,offset=request),))):
                view=child(identity+'.'+role,role,shape,mapping)
                controls.append(TileAccessTemplate(role,view.view_id,Operation.READ,('row_block','column'),
                    (None,),(ready[role],),(),evidence,unmapped=True))
            family_scope=replace(scope,scope_id=identity,op_ids=(*header,str(request),'bypass'))
            families.append(TileProgram(identity,family_scope,(TileAxis('row_block',length*heads,8),TileAxis('column',1,1)),
                tuple(controls),visibility,evidence))
            offset+=length;continue
        pre=min(3,count)
        for packed in range(length*heads):
            query,head=packed%length,packed//length
            for column in range(1):
                identity=f'{scope.scope_id}.r{request}.row{packed}.col{column}'
                accesses=[];axes=[TileAxis('split',count,1),TileAxis('preload',pre,1),
                    TileAxis('compute_prefix',pre,1,'last_valid',0,True,count-1)]
                def access(name,view,operation,loops,view_axes,dependencies=(),input_refs=(),unmapped=False,constant=None):
                    accesses.append(TileAccessTemplate(name,view.view_id,operation,loops,view_axes,input_refs,
                        dependencies,evidence,unmapped=unmapped,constant_value=constant))
                cu=child(identity+'.cu_query','cu_query',(2,),(coordinate(0,(0,1),offset=request),))
                ds=child(identity+'.dynamic_splits','dynamic_splits',(1,),(coordinate(0,offset=request),))
                access('cu_query',cu,Operation.READ,(),(None,),input_refs=(ready['cu_query'],),unmapped=True)
                access('dynamic_splits',ds,Operation.READ,(),(None,),input_refs=(ready['dynamic_splits'],),unmapped=True)
                controls=(dep('cu_query'),dep('dynamic_splits'))
                lse=child(identity+'.partial_lse',f'partial_lse.r{request}',(count,),(
                    coordinate(0,(0,1)),coordinate(1,offset=head),coordinate(2,offset=query)))
                access('partial_lse',lse,Operation.READ,('split',),('split',),controls,
                    (ready[f'partial_lse.r{request}'],),True)
                scale=scratch(identity+'.scale',(count,),'fp32',4)
                last=scratch(identity+'.max_valid',(1,),'int32',4)
                reduced=(dep('partial_lse',axes=('split',)),)
                access('scale_write',scale,Operation.WRITE,(),(None,),reduced)
                access('max_valid_write',last,Operation.WRITE,(),(None,),reduced)
                access('last_valid',last,Operation.READ,(),(None,),(dep('max_valid_write',EventKind.WRITE_VISIBLE),))
                first=child(identity+'.O_prefix',f'partial_output.r{request}',(pre,128),(
                    coordinate(0,(0,1)),coordinate(1,offset=head),coordinate(2,offset=query),
                    coordinate(3,(1,1),offset=column*128)))
                access('O_preload',first,Operation.READ,('preload',),('preload',None),controls,
                    (ready[f'partial_output.r{request}'],),True)
                prefix=scratch(identity+'.staged_prefix',(pre,128),'fp32',4)
                access('prefix_stage',prefix,Operation.WRITE,(),(None,None),(dep('O_preload',axes=('preload',)),))
                access('prefix_compute',prefix,Operation.READ,('compute_prefix',),('compute_prefix',None),
                    (dep('last_valid'),dep('prefix_stage',EventKind.WRITE_VISIBLE)))
                # A strided contained view keeps the complete scale producer;
                # no guessed scale sign or max-valid value is inserted.
                scale_prefix=TensorView(identity+'.scale_prefix',scale.allocation_id,0,(pre,),(4,),'fp32',4)
                selected.append(scale_prefix)
                access('scale_prefix',scale_prefix,Operation.READ,('compute_prefix',),('compute_prefix',),
                    (dep('last_valid'),dep('scale_write',EventKind.WRITE_VISIBLE)))
                output_deps=[*controls,dep('last_valid'),dep('prefix_compute',axes=('compute_prefix',)),dep('scale_prefix',axes=('compute_prefix',))]
                if count>pre:
                    tail=count-pre;axes.append(TileAxis('tail',tail,1,'last_valid',pre,True,count-1))
                    rest=child(identity+'.O_tail',f'partial_output.r{request}',(tail,128),(
                        coordinate(0,(0,1),offset=pre),coordinate(1,offset=head),coordinate(2,offset=query),
                        coordinate(3,(1,1),offset=column*128)))
                    tail_state=scratch(identity+'.staged_tail',(tail,128),'fp32',4)
                    access('O_tail',rest,Operation.READ,('tail',),('tail',None),(*controls,dep('last_valid')),
                        (ready[f'partial_output.r{request}'],),True)
                    access('tail_stage',tail_state,Operation.WRITE,('tail',),('tail',None),
                        (dep('last_valid'),dep('O_tail')))
                    access('tail_compute',tail_state,Operation.READ,('tail',),('tail',None),
                        (dep('last_valid'),dep('tail_stage',EventKind.WRITE_VISIBLE)))
                    scale_tail=TensorView(identity+'.scale_tail',scale.allocation_id,pre*4,(tail,),(4,),'fp32',4)
                    selected.append(scale_tail)
                    access('scale_tail',scale_tail,Operation.READ,('tail',),('tail',),
                        (dep('last_valid'),dep('scale_write',EventKind.WRITE_VISIBLE)))
                    output_deps.extend((dep('tail_compute',axes=('tail',)),dep('scale_tail',axes=('tail',))))
                parent=parents['output']
                output=TensorView(identity+'.output',parent.allocation_id,
                    parent.offset_bytes+(offset+query)*parent.strides_bytes[0]+head*parent.strides_bytes[1],
                    (128,),(parent.strides_bytes[2],),parent.dtype,parent.element_bytes)
                selected.append(output)
                access('output',output,Operation.WRITE,(),(None,),tuple(output_deps),unmapped=True)
                if column==0:
                    parent=parents['lse']
                    final_lse=TensorView(identity+'.lse',parent.allocation_id,
                        parent.offset_bytes+head*parent.strides_bytes[0]+(offset+query)*parent.strides_bytes[1],
                        (1,),(parent.strides_bytes[1],),parent.dtype,parent.element_bytes)
                    selected.append(final_lse)
                    access('lse',final_lse,Operation.WRITE,(),(None,),reduced,unmapped=True)
                family_scope=replace(scope,scope_id=identity,op_ids=(*header,str(request),str(packed),str(column)))
                families.append(TileProgram(identity,family_scope,tuple(axes),tuple(accesses),visibility,evidence))
        offset+=length
    # The last grid CTA resets the semaphore before any virtual-batch, row,
    # or split early return, including calls in which every request bypasses.
    identity=f'{scope.scope_id}.reset';family_scope=replace(scope,scope_id=identity,op_ids=(*header,'reset'))
    reset=TileAccessTemplate('semaphore_reset','semaphore',Operation.WRITE,(),(None,),(),(),evidence,
        unmapped=True,constant_value=ConstantValue.ZERO)
    families.append(TileProgram(identity,family_scope,(TileAxis('reset',1,1),),(reset,),visibility,evidence))
    return tuple(allocations),tuple(selected),tuple(events),tuple(families)


def fa3_split_prefill_call_storage_layouts(main,combine,hardware):
    """Source-owned public/workspace extents, without allocations or readiness."""
    header=_fa3_split_prefill_combine_header(main,combine,hardware)
    plan=fa3_split_prefill_tensor_parent_plan(main,combine,hardware,binding_id='layout_only')
    data=tuple((view.view_id,view.shape,view.dtype,view.element_bytes)
        for view in plan['views'] if view.view_id in ('query','key','value'))
    return (*data,('cu_key',(main.shape['batch']+1,),'int32',4),
        *_fa3_split_prefill_combine_layouts(header))


def standalone_fa3_split_prefill_call_bindings(main,combine,hardware,*,call_id,binding_id,scheduler_metadata_precomputed):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if type(scheduler_metadata_precomputed) is not bool:
        raise TypeError('Split-prefill metadata lifetime must be an actual caller boolean')
    base=standalone_fa3_split_prefill_combine_bindings(main,combine,hardware,call_id=call_id,binding_id=binding_id)
    plan=fa3_split_prefill_tensor_parent_plan(main,combine,hardware,binding_id=binding_id)
    allocations=list(base.allocations);views=list(base.views)
    # Only real external Q/K/V and cumulative-length inputs are available.
    # Scheduler fields and partials below acquire actual prepare/main writers.
    events=[event for event in base.events if event.kind is EventKind.ALLOCATED
        or event.event_id==f'{binding_id}.cu_query.available']
    for view in plan['views']:
        if view.view_id not in ('query','key','value'):
            continue
        allocations.append(MemoryAllocation(view.allocation_id,'logical-local-device','logical-fa3-call',None,
            AddressLevel.OBJECT,None,None,view.allocation_id+'.allocated',None))
        views.append(view)
    identity=binding_id+'.cu_key'
    allocations.append(MemoryAllocation(identity,'logical-local-device','logical-fa3-call',None,
        AddressLevel.OBJECT,None,None,identity+'.allocated',None))
    views.append(TensorView('cu_key',identity,0,(main.shape['batch']+1,),(4,),'int32',4))
    allocated={event.event_id for event in events}
    for allocation in allocations:
        if allocation.allocated_event not in allocated:
            events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,allocation.allocation_id,None,None,()))
    by_role=unique_by(views,'view_id')
    for role in ('query','key','value','cu_key'):
        view=by_role[role]
        events.append(ProgramEvent(f'{binding_id}.{role}.available',EventKind.LOGICAL_AVAILABLE,view.allocation_id,
            Unmapped(view.shape,view.dtype,view.element_bytes,f'{binding_id}.{role}.external_input'),
            base.visibility_domain,(EventRef(view.allocation_id+'.allocated',EventKind.ALLOCATED),)))
    reads=('query','key','value','cu_query','cu_key')
    if scheduler_metadata_precomputed:
        meta=by_role['scheduler_metadata']
        view=TensorView('prepared_query',meta.allocation_id,meta.offset_bytes,(main.shape['batch'],),(4,),'int32',4)
        views.append(view);by_role['prepared_query']=view
        reads+=('prepared_query','dynamic_splits','semaphore')
        for role in reads[-3:]:
            view=by_role[role]
            events.append(ProgramEvent(f'{binding_id}.{role}.available',EventKind.LOGICAL_AVAILABLE,view.allocation_id,
                Unmapped(view.shape,view.dtype,view.element_bytes,f'{binding_id}.{role}.caller_precomputed_input'),
                base.visibility_domain,(EventRef(meta.allocation_id+'.allocated',EventKind.ALLOCATED),),view.view_id))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),reads,
        ('output','lse','semaphore'),tuple(EventRef(f'{binding_id}.{role}.available',EventKind.LOGICAL_AVAILABLE)
            for role in reads),base.rank,base.stream,base.visibility_domain)


def build_fa3_split_prefill_call_source(main,combine,hardware,bindings,granularity,*,program_id,scheduler_metadata_precomputed):
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_access import TileWriteSource
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('Split-prefill whole call requires actual typed caller operands')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('Unknown split-prefill call granularity')
    header=_fa3_split_prefill_combine_header(main,combine,hardware)
    q,kv,heads,static,splits=_fa3_split_prefill_combine_geometry(header)
    expected=standalone_fa3_split_prefill_call_bindings(main,combine,hardware,
        call_id=bindings.call_id,binding_id=bindings.binding_id,scheduler_metadata_precomputed=scheduler_metadata_precomputed)
    if ((bindings.read_views,bindings.write_views)!=(expected.read_views,expected.write_views)
            or len(bindings.input_ready)!=len(bindings.read_views)):
        raise ValueError('Split-prefill whole-call operand/readiness roles mismatch')
    parents=unique_by(bindings.views,'view_id');reference=unique_by(expected.views,'view_id')
    if set(parents)!=set(reference):
        raise ValueError('Split-prefill whole call lost an actual public/control parent')
    for role,view in parents.items():
        other=reference[role]
        _validate_fa3_split_prefill_parent_geometry(role,view,other)
    # Preserve all native alias classes while allowing real caller identities.
    actual_aliases={};reference_aliases={}
    for role,view in parents.items():
        if view.allocation_id not in actual_aliases:actual_aliases[view.allocation_id]=[]
        if reference[role].allocation_id not in reference_aliases:reference_aliases[reference[role].allocation_id]=[]
        actual_aliases[view.allocation_id].append(role)
        reference_aliases[reference[role].allocation_id].append(role)
    if sorted(actual_aliases.values())!=sorted(reference_aliases.values()):
        raise ValueError('Split-prefill whole-call native parent aliases differ')
    allocations=list(bindings.allocations);selected=[];events=list(bindings.events);families=[]
    allocation_map=unique_by(allocations,'allocation_id')
    ready=dict(zip(bindings.read_views,bindings.input_ready,strict=True))
    visibility=bindings.visibility_domain
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-fa3-split-call')
    evidence=('Pinned source_803020 noncausal192/128 varlen split library call; no duration inputs.',
        'Prepare dynamic-split output depends on all request Q/K endpoint reads; numeric values are not fabricated.',
        'Main emits FP32 LSE for every dynamic task, including empty minus-infinity writes; partial O only for nonempty tasks.',
        'Native scheduler worker assignment, queue/barrier service and issued-thread multiplicity remain unbound.')

    def publish(view,family,access_id):
        event=ProgramEvent(family.tile_program_id+'.'+access_id+'.written',EventKind.LOGICAL_AVAILABLE,view.allocation_id,
            Unmapped(view.shape,view.dtype,view.element_bytes,view.view_id+'.actual_source_write'),visibility,
            (EventRef(allocation_map[view.allocation_id].allocated_event,EventKind.ALLOCATED),),view.view_id,
            (TileWriteSource(family.tile_program_id,access_id),))
        events.append(event)
        return EventRef(event.event_id,event.kind)

    def child(identity,role,shape,mapping):
        parent=parents[role]
        view=LogicalSelectionView(identity,parent.allocation_id,shape,parent.dtype,parent.element_bytes,
            'fa3_split_call.'+identity,role,mapping)
        selected.append(view)
        return view

    def coord(axis,*terms,offset=0):
        return LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(index,scale) for index,scale in terms),offset)

    meta=parents['scheduler_metadata'];batch=len(q)
    prepared=(parents['prepared_query'] if scheduler_metadata_precomputed else
        TensorView(program_id+'.prepared_query',meta.allocation_id,meta.offset_bytes,(batch,),(4,),'int32',4))
    if not scheduler_metadata_precomputed:selected.append(prepared)
    fields={'prepared_query':prepared,'dynamic_splits':parents['dynamic_splits'],'semaphore':parents['semaphore']}
    prep_id=program_id+'.prepare';prep_accesses=[]
    for role in ('cu_query','cu_key'):
        view=child(prep_id+'.'+role,role,(batch,2),(coord(0,(0,1),(1,1)),))
        prep_accesses.append(TileAccessTemplate(role,view.view_id,Operation.READ,('request',),('request',None),
            (ready[role],),(),evidence,unmapped=True))
    reductions=tuple(TileDependency(role,EventKind.READ_READY,('request',)) for role in ('cu_query','cu_key'))
    prep_accesses.extend((
        TileAccessTemplate('prepared_query.write',prepared.view_id,Operation.WRITE,(),(None,),(),
            (TileDependency('cu_query',EventKind.READ_READY,('request',)),),evidence,unmapped=True),
        TileAccessTemplate('dynamic_splits.write',fields['dynamic_splits'].view_id,Operation.WRITE,(),(None,),(),
            reductions,evidence,unmapped=True),
        TileAccessTemplate('semaphore.zero',fields['semaphore'].view_id,Operation.WRITE,(),(None,),(),(),
            evidence,unmapped=True,constant_value=ConstantValue.ZERO)))
    prep=TileProgram(prep_id,replace(scope,scope_id=prep_id,op_ids=(*header,'prepare')),
        (TileAxis('request',batch,1),),tuple(prep_accesses),visibility,evidence)
    if scheduler_metadata_precomputed:
        # The caller supplies prepared metadata; this call does not create
        # another prepare or assert a producer inside the current library call.
        # Discard the unsubmitted prepare-only selections, never its inputs.
        selected[:]=[view for view in selected if not view.view_id.startswith(prep_id+'.')]
        field_ready={role:ready[role] for role in fields}
    else:
        families.append(prep)
        field_ready={role:publish(view,prep,role+'.write' if role!='semaphore' else 'semaphore.zero')
            for role,view in fields.items()}
    source_writes={request:[] for request in range(batch)}
    source_lse={request:[] for request in range(batch)}
    result_refs=[];q_prefix=kv_prefix=0
    for request,(length,keys,count) in enumerate(zip(q,kv,splits,strict=True)):
        panels=-(-keys//128);per_split=-(-panels//count)
        for split in range(count):
            lo=split*per_split*128;hi=min(keys,(split+1)*per_split*128)
            for head in range(heads):
                for first in range(0,length,128):
                    rows=min(128,length-first)
                    identity=f'{program_id}.main.r{request}.s{split}.h{head}.q{first}'
                    accesses=[]
                    for role,field in fields.items():
                        if role=='semaphore':continue
                        parent_role=prepared.view_id if role=='prepared_query' else 'dynamic_splits'
                        if role=='prepared_query':parents[parent_role]=prepared
                        view=child(identity+'.'+role,parent_role,(1,),(coord(0,offset=request),))
                        accesses.append(TileAccessTemplate(role,view.view_id,Operation.READ,(),(None,),
                            (field_ready[role],),(),evidence,unmapped=True))
                    deps=[TileDependency(role,EventKind.READ_READY,()) for role in ('prepared_query','dynamic_splits')]
                    for role in ('cu_query','cu_key'):
                        view=child(identity+'.'+role,role,(2,),(coord(0,(0,1),offset=request),))
                        accesses.append(TileAccessTemplate(role,view.view_id,Operation.READ,(),(None,),
                            (ready[role],),tuple(deps),evidence,unmapped=True))
                    deps.extend(TileDependency(role,EventKind.READ_READY,()) for role in ('cu_query','cu_key'))
                    if hi>lo:
                        for role,width,begin,size in (('query',192,q_prefix+first,rows),
                                ('key',192,kv_prefix+lo,hi-lo),('value',128,kv_prefix+lo,hi-lo)):
                            view=child(identity+'.'+role,role,(size,width),(
                                coord(0,(0,1),offset=begin),coord(1,offset=head),coord(2,(1,1))))
                            accesses.append(TileAccessTemplate(role,view.view_id,Operation.READ,(),(None,None),
                                (ready[role],),tuple(deps),evidence,unmapped=True))
                        deps.extend(TileDependency(role,EventKind.READ_READY,()) for role in ('query','key','value'))
                    if count==1:
                        output=parents['output'];lse=parents['lse']
                        o_offset=output.offset_bytes+(q_prefix+first)*output.strides_bytes[0]+head*output.strides_bytes[1]
                        lse_offset=lse.offset_bytes+head*lse.strides_bytes[0]+(q_prefix+first)*lse.strides_bytes[1]
                        o_strides=(output.strides_bytes[0],output.strides_bytes[2]);lse_stride=lse.strides_bytes[1]
                    else:
                        output=parents['partial_output'];lse=parents['partial_lse']
                        o_offset=output.offset_bytes+split*output.strides_bytes[0]+head*output.strides_bytes[1]+(q_prefix+first)*output.strides_bytes[2]
                        lse_offset=lse.offset_bytes+split*lse.strides_bytes[0]+head*lse.strides_bytes[1]+(q_prefix+first)*lse.strides_bytes[2]
                        o_strides=(output.strides_bytes[2],output.strides_bytes[3]);lse_stride=lse.strides_bytes[2]
                    lse_view=TensorView(identity+'.lse',lse.allocation_id,lse_offset,(rows,),(lse_stride,),'fp32',4)
                    selected.append(lse_view)
                    lse_deps=tuple(dep for dep in deps if dep.access_id!='value')
                    accesses.append(TileAccessTemplate('lse',lse_view.view_id,Operation.WRITE,(),(None,),(),lse_deps,
                        evidence,unmapped=True,constant_value=ConstantValue.NEGATIVE_INFINITY if hi<=lo else None))
                    if hi>lo:
                        o_view=TensorView(identity+'.output',output.allocation_id,o_offset,(rows,128),o_strides,output.dtype,output.element_bytes)
                        selected.append(o_view)
                        accesses.append(TileAccessTemplate('output',o_view.view_id,Operation.WRITE,(),(None,None),(),tuple(deps),evidence,unmapped=True))
                    family=TileProgram(identity,replace(scope,scope_id=identity,op_ids=(*header,'main',str(request),str(split),str(head),str(first))),
                        (TileAxis('epilogue',1,1),),tuple(accesses),visibility,evidence)
                    families.append(family)
                    lse_ref=publish(lse_view,family,'lse');source_lse[request].append(lse_ref)
                    if count==1:result_refs.append((lse_view.view_id,lse_ref))
                    if hi>lo:
                        o_ref=publish(o_view,family,'output');source_writes[request].append(o_ref)
                        if count==1:result_refs.append((o_view.view_id,o_ref))
        q_prefix+=length;kv_prefix+=keys
    # Aggregate availability is justified by this recipe's complete set of
    # per-M/head/split writer fields. O accessibility includes reserved empty
    # slots, explicitly without claiming their initialization or validity.
    # flash_api launches combine after main on the call's stream. Retain
    # that source-stage boundary even for the bypass/reset-only path.
    main_done=ProgramEvent(program_id+'.main.tensor_outputs_complete',EventKind.COMPUTE_DONE,None,None,None,
        tuple(ref for request in range(batch) for ref in (*source_lse[request],*source_writes[request])))
    events.append(main_done)
    after_main={}
    for role in ('dynamic_splits','semaphore'):
        view=fields[role]
        event=ProgramEvent(f'{program_id}.{role}.after_main',EventKind.LOGICAL_AVAILABLE,view.allocation_id,
            Unmapped(view.shape,view.dtype,view.element_bytes,f'{program_id}.{role}.source_stage_order'),visibility,
            (field_ready[role],EventRef(main_done.event_id,main_done.kind)),view.view_id)
        events.append(event);after_main[role]=EventRef(event.event_id,event.kind)
    combine_ready={'cu_query':ready['cu_query'],'dynamic_splits':after_main['dynamic_splits']}
    for request,count in enumerate(splits):
        if count<=1:continue
        for base,refs in (('partial_lse',tuple(source_lse[request])),
                ('partial_output',tuple((*source_writes[request],*source_lse[request])))):
            role=f'{base}.r{request}';view=parents[role]
            event=ProgramEvent(f'{program_id}.{role}.produced',EventKind.LOGICAL_AVAILABLE,view.allocation_id,
                Unmapped(view.shape,view.dtype,view.element_bytes,f'{program_id}.{role}.actual_main_epilogues'),
                visibility,(EventRef(allocation_map[view.allocation_id].allocated_event,EventKind.ALLOCATED),*refs),role)
            events.append(event);combine_ready[role]=EventRef(event.event_id,event.kind)
    combine_parents={role:parents[role] for role,*_ in _fa3_split_prefill_combine_layouts(header)}
    _validate_fa3_split_prefill_combine_parents(header,combine_parents)
    combine_scope=replace(scope,scope_id=program_id+'.combine',op_ids=header)
    internal,children,combine_events,combine_families=_fa3_split_prefill_combine_recipe(header,
        tuple(sorted(combine_parents.items())),tuple(sorted(combine_ready.items())),combine_scope,visibility,
        bindings.binding_id+'.combine',allocation_map[parents['output'].allocation_id].device_id)
    combine_families=tuple(replace(family,accesses=tuple(replace(access,input_ready=(after_main['semaphore'],))
        if access.access_id=='semaphore_reset' else access for access in family.accesses))
        if family.scope.op_ids[-1]=='reset' else family for family in combine_families)
    allocations.extend(internal);selected.extend(children);events.extend(combine_events);families.extend(combine_families)
    child_views=unique_by(children,'view_id')
    for family in combine_families:
        for access in family.accesses:
            if access.access_id in ('output','lse'):
                view=child_views[access.view_id]
                result_refs.append((view.view_id,publish(view,family,access.access_id)))
            elif access.access_id=='semaphore_reset':
                reset_ref=publish(parents['semaphore'],family,access.access_id)
    complete=ProgramEvent(program_id+'.tensor_and_reset_complete',EventKind.COMPUTE_DONE,None,None,None,
        (*tuple(ref for role,ref in result_refs),reset_ref))
    events.append(complete)
    # Every final element is covered by the actual writer fields above.
    # Publish the two full parent results after their union and reset, not
    # one global-fence event per row. No copy or extra WRITE is introduced.
    for role in ('output','lse'):
        view=parents[role]
        events.append(ProgramEvent(program_id+'.'+role+'.library_result',EventKind.LOGICAL_AVAILABLE,
            view.allocation_id,Unmapped(view.shape,view.dtype,view.element_bytes,
                program_id+'.'+role+'.complete_source_writer_union'),visibility,
            (EventRef(allocation_map[view.allocation_id].allocated_event,EventKind.ALLOCATED),
                EventRef(complete.event_id,complete.kind)),role))
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,main.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            *evidence,'Prepare/main/128-wide combine tensor producer/readiness graph, not an issued-memory trace.',
            'O full dynamic views are accessible after valid writes, not numerically initialized in empty slots.',
            'Conditional combine O tails retain real LSE/max-valid READ bounds.',
            'Native queue worker assignment, synchronization service, physical cache reuse and caller/dispatch integration remain open.')),
        tuple(allocations),(*bindings.views,*selected),(scope,),tuple(events),(),tuple(families))


def fa3_split_prefill_call_source_work(program,*,main,combine,hardware,bindings,scheduler_metadata_precomputed):
    from src.kernel.memory_access import _tile_template_requested_bytes
    program.__post_init__()
    expected=build_fa3_split_prefill_call_source(main,combine,hardware,bindings,Granularity.TILE,
        program_id=program.program_id,scheduler_metadata_precomputed=scheduler_metadata_precomputed)
    if program!=expected:
        raise ValueError('Split-prefill whole call lost actual source writers, aliases, preparation or readiness')
    views=unique_by(program.views,'view_id')
    totals={role:0 for role in ('query','key','value','output','lse')}
    tensor_parents={role:0 for role in ('partial_output','partial_lse','output','lse')}
    public_parents=unique_by(bindings.views,'view_id')
    for family in program.tile_programs:
        if '.main.r' not in family.tile_program_id:continue
        axes=unique_by(family.axes,'axis_id')
        for access in family.accesses:
            if access.access_id in totals:
                totals[access.access_id]+=_tile_template_requested_bytes(views,axes,family,access)
            if access.operation is Operation.WRITE:
                view=views[access.view_id]
                for role in tensor_parents:
                    if view.allocation_id==public_parents[role].allocation_id:
                        tensor_parents[role]+=_tile_template_requested_bytes(views,axes,family,access)
    return {'main_tensor_totals':totals,'main_tensor_bytes_by_parent':tensor_parents,
        'scheduler_metadata_precomputed':scheduler_metadata_precomputed,'main_control_service_bound':False,
        'source_specific_tensor_readiness_bound':True,'dispatch_integrated':False}


def fa3_split_prefill_call_result_bindings(program,*,main,combine,hardware,bindings,call_id,binding_id,scheduler_metadata_precomputed):
    from src.kernel.memory_access_builder import ElementwiseBindings
    fa3_split_prefill_call_source_work(program,main=main,combine=combine,hardware=hardware,bindings=bindings,
        scheduler_metadata_precomputed=scheduler_metadata_precomputed)
    parent_ids={view.allocation_id for view in bindings.views if view.view_id in ('output','lse')}
    results=tuple(event for event in program.events if event.event_id.endswith('.library_result')
        and event.allocation_id in parent_ids and event.view_id is not None)
    return ElementwiseBindings(binding_id,call_id,program.allocations,program.views,program.events,
        tuple(event.view_id for event in results),(),tuple(EventRef(event.event_id,event.kind) for event in results),
        bindings.rank,bindings.stream,bindings.visibility_domain)


"""Unapplied public-predict bridge; ordinary pricing builds no detailed graph."""

_FA3_SPLIT_PREFILL_PREDICTION = ContextVar('fa3_split_prefill_prediction')


def fa3_split_prefill_call_prediction_work(main,combine,hardware,bindings,*,scheduler_metadata_precomputed):
    """Validate real call operands and source quantities without private DAGs."""
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_access import validate_storage_bindings,validate_input_readiness
    from src.kernel.predictors.flash_attn import _fa3_split_prefill_work,_fa3_gqa_observed_work
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('Split-prefill prediction requires actual typed caller bindings')
    _fa3_split_prefill_combine_header(main,combine,hardware)
    expected=standalone_fa3_split_prefill_call_bindings(main,combine,hardware,
        call_id=bindings.call_id,binding_id=bindings.binding_id,scheduler_metadata_precomputed=scheduler_metadata_precomputed)
    bindings.__post_init__()
    if ((bindings.read_views,bindings.write_views)!=(expected.read_views,expected.write_views)
            or len(bindings.input_ready)!=len(bindings.read_views)):
        raise ValueError('Split-prefill prediction operand/readiness roles mismatch')
    actual=unique_by(bindings.views,'view_id');reference=unique_by(expected.views,'view_id')
    if set(actual)!=set(reference):raise ValueError('Split-prefill prediction parent roles differ')
    actual_aliases={};expected_aliases={}
    for role,view in actual.items():
        other=reference[role]
        _validate_fa3_split_prefill_parent_geometry(role,view,other)
        if view.allocation_id not in actual_aliases:actual_aliases[view.allocation_id]=[]
        if other.allocation_id not in expected_aliases:expected_aliases[other.allocation_id]=[]
        actual_aliases[view.allocation_id].append(role);expected_aliases[other.allocation_id].append(role)
    if sorted(actual_aliases.values())!=sorted(expected_aliases.values()):
        raise ValueError('Split-prefill prediction lost actual native parent aliases')
    validate_storage_bindings(bindings.allocations,bindings.views,bindings.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual split-prefill caller boundary; detailed source recipes remain explicit export.',)))
    events=unique_by(bindings.events,'event_id')
    for role,ref in zip(bindings.read_views,bindings.input_ready,strict=True):
        validate_input_readiness(actual[role],ref,unmapped=True,views=actual,events=events,
            visibility_domain=bindings.visibility_domain)
    mw=_fa3_split_prefill_work(main,hardware);cw=_fa3_gqa_observed_work(combine,hardware)
    output=prod(actual['output'].shape)*2;lse=prod(actual['lse'].shape)*4
    if mw['splits']!=cw['splits'] or mw['direct_write_bytes']+cw['write_bytes']!=output+lse:
        raise ValueError('Split-prefill source result bytes or dynamic splits differ')
    return {'main_write_bytes':mw['write_bytes'],'main_l2_read_bytes':mw['logical_l2_read_bytes'],
        'main_direct_write_bytes':mw['direct_write_bytes'],'partial_output_write_bytes':mw['partial_output_bytes'],
        'partial_lse_write_bytes':mw['partial_lse_bytes'],'combine_write_bytes':cw['write_bytes'],
        'output_bytes':output,'lse_bytes':lse,'splits':mw['splits'],
        'metadata_precomputed':scheduler_metadata_precomputed,'physical_service_bound':False}


@contextmanager
def fa3_split_prefill_call_prediction_scope(main,combine,hardware,bindings,*,scheduler_metadata_precomputed):
    """One real library call; no guessed sibling or metadata lifetime facts."""
    from copy import deepcopy
    work=fa3_split_prefill_call_prediction_work(main,combine,hardware,bindings,
        scheduler_metadata_precomputed=scheduler_metadata_precomputed)
    seal=deepcopy(bindings);kernels=repr((main,combine))
    token=_FA3_SPLIT_PREFILL_PREDICTION.set((main,combine,hardware,bindings,scheduler_metadata_precomputed,work,seal,kernels))
    try:
        yield work
    finally:
        _FA3_SPLIT_PREFILL_PREDICTION.reset(token)


def current_fa3_split_prefill_prediction_work(kernel,hardware):
    from src.kernel.memory_access_builder import _prediction_bindings
    context=copy_context()
    if _FA3_SPLIT_PREFILL_PREDICTION not in context:return None
    main,combine,actual_hardware,bindings,precomputed,work,seal,kernels=context[_FA3_SPLIT_PREFILL_PREDICTION]
    if hardware is not actual_hardware or kernel not in (main,combine) or repr((main,combine))!=kernels:
        raise ValueError('Split-prefill prediction crossed or changed its actual source call')
    actual,_=_prediction_bindings(kernel,None)
    if actual is not bindings or actual!=seal:
        raise ValueError('Split-prefill prediction requires the unchanged actual caller bindings')
    observed=fa3_split_prefill_call_prediction_work(main,combine,hardware,actual,scheduler_metadata_precomputed=precomputed)
    if observed!=work:raise ValueError('Split-prefill prediction source quantities changed')
    return observed


def _validate_fa3_sparse_mla_input_boundary(kernel,hardware,bindings,selected_rows):
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_access import Span
    from src.kernel.predictors.flash_attn import _fa3_sparse_mla_work
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('Sparse FA3 input export requires actual typed caller bindings')
    work=_fa3_sparse_mla_work(kernel,hardware)
    if kernel.variant!='mla_decode':
        raise ValueError('Sparse FA3 input export requires its actual main stage')
    batch,heads=kernel.shape['batch'],kernel.shape['num_query_heads']
    used=tuple(kernel.attributes['kv_lens']);capacity=kernel.launch.tiling.tile_shape['kv_capacity']
    roles=('query_rope','query_latent','key_cache','value_cache','block_table','used_key','cu_query')
    if (bindings.read_views!=roles or bindings.write_views or len(bindings.input_ready)!=len(roles)):
        raise ValueError('Sparse FA3 input operand roles/readiness arity mismatch')
    views=unique_by(bindings.views,'view_id')
    if set(views)!=set((*roles,'cache_rows')) or any(type(view) is not TensorView for view in views.values()):
        raise ValueError('Sparse FA3 requires actual resolved tensor views including the cache row parent')
    for view in views.values():
        canonical=TensorView(view.view_id,view.allocation_id,view.offset_bytes,view.shape,
            view.strides_bytes,view.dtype,view.element_bytes)
        if view.region()!=canonical.region():
            raise ValueError('Sparse caller tensor derived region differs from its actual fields')
    if (type(selected_rows) is not tuple or len(selected_rows)!=batch
            or any(type(row) is not tuple or len(row)!=length
                for row,length in zip(selected_rows,used,strict=True))):
        raise ValueError('Sparse FA3 requires the actual live index prefix, separate from capacity')
    root=views['cache_rows'];pages=root.shape[0]
    if (root.shape!=(pages,576) or (root.dtype,root.element_bytes)!=('bf16',2)
            or root.strides_bytes[1]!=2 or root.strides_bytes[0]<1152):
        raise ValueError('Sparse FA3 requires its actual flat BF16 latent512/RoPE64 cache')
    if any(type(row) is not int or not 0<=row<pages for rows in selected_rows for row in rows):
        raise ValueError('Sparse FA3 live physical indices must fit the actual cache row parent')
    stride=root.strides_bytes[0]
    for role,width,offset in (('key_cache',64,1024),('value_cache',512,0)):
        view=views[role]
        if (view.shape,view.dtype,view.element_bytes,view.allocation_id,view.offset_bytes,view.strides_bytes)!=(
                (pages,1,1,width),'bf16',2,root.allocation_id,root.offset_bytes+offset,
                (stride,width*2,width*2,2)):
            raise ValueError('Sparse FA3 K/V must retain their real latent/RoPE cache aliases')
    for role,width in (('query_rope',64),('query_latent',512)):
        view=views[role]
        if ((view.shape,view.dtype,view.element_bytes)!=((batch,heads,width),'bf16',2)
                or view.strides_bytes[2]!=2 or view.strides_bytes[1]<width*2
                or view.strides_bytes[0]<(heads-1)*view.strides_bytes[1]+width*2):
            raise ValueError('Sparse FA3 query shape or actual head/token stride mismatch')
    for role,shape in (('block_table',(batch,capacity)),('used_key',(batch,)),('cu_query',(batch+1,))):
        view=views[role]
        if (view.shape,view.dtype,view.element_bytes)!=(shape,'int32',4):
            raise ValueError('Sparse FA3 metadata storage must preserve capacity and real live lengths')
        if view.strides_bytes[-1]!=4 or (role=='block_table' and view.strides_bytes[0]<capacity*4):
            raise ValueError('Sparse FA3 metadata must retain its source contiguous scalar layout')
    metadata={views[role].allocation_id for role in ('block_table','used_key','cu_query')}
    payload={views[role].allocation_id for role in ('query_rope','query_latent','cache_rows')}
    if len(metadata)!=3 or metadata&payload or root.allocation_id in {
            views['query_rope'].allocation_id,views['query_latent'].allocation_id}:
        raise ValueError('Sparse FA3 fresh metadata/query/cache aliases differ from the source ABI')
    allocations=unique_by(bindings.allocations,'allocation_id')
    if set(allocations)!={view.allocation_id for view in views.values()}:
        raise ValueError('Sparse FA3 input export requires exactly its actual operand allocations')
    from src.kernel.memory_access import validate_storage_bindings,_union_covers
    _,_,_,events,_,ancestors=validate_storage_bindings(bindings.allocations,bindings.views,bindings.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual sparse caller snapshot boundary;no detailed source graph.',)))
    visible={}
    for role,ref in zip(bindings.read_views,bindings.input_ready,strict=True):
        if ref.allow_skipped_read or ref.node_id not in events or events[ref.node_id].kind is not ref.kind:
            raise ValueError('Sparse source readiness must refer to actual caller events')
        reachable=(*ancestors[ref.node_id],ref)
        visible[role]=tuple(events[item.node_id].region for item in reachable
            if item.node_id in events and events[item.node_id].kind in (EventKind.INITIALIZED,EventKind.LOGICAL_AVAILABLE)
            and events[item.node_id].allocation_id==views[role].allocation_id
            and events[item.node_id].visibility_domain==bindings.visibility_domain)
    def available(role,region):
        if not _union_covers(visible[role],region):
            raise ValueError('Sparse caller selected input lacks initialized/visible readiness')
    for role in ('used_key','cu_query'):available(role,views[role].region())
    for request,length in enumerate(used):
        if not length:continue
        for role,width in (('query_rope',64),('query_latent',512)):
            view=views[role]
            available(role,Strided(view.offset_bytes+request*view.strides_bytes[0],(heads,),
                (view.strides_bytes[1],),width*2))
        view=views['block_table']
        available('block_table',Span(view.offset_bytes+request*view.strides_bytes[0],length*4))
    for row in sorted({row for request in selected_rows for row in request}):
        for role,width in (('key_cache',64),('value_cache',512)):
            view=views[role]
            available(role,Span(view.offset_bytes+row*stride,width*2))
    return work,views,allocations


def _validate_fa3_sparse_mla_call_boundary(parts,hardware,inputs,storage,selected_rows):
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_access import Span,TileWriteSource,covers
    from src.kernel.kernel_spec import KernelSpec
    from src.kernel.predictors.flash_attn import _fa3_sparse_mla_work
    if type(parts) is not tuple or len(parts) not in (2,3):
        raise ValueError('Sparse source requires its original whole library tuple')
    if (any(not isinstance(part,KernelSpec) for part in parts)
            or parts[0].kind is not KernelKind.VECTOR or parts[0].variant!='fa3_prepare_varlen'
            or parts[1].kind is not KernelKind.FLASH_ATTN or parts[1].variant!='mla_decode'):
        raise ValueError('Sparse source requires its original prepare and main stages')
    main=parts[1];work=_fa3_sparse_mla_work(main,hardware)
    expected=hardware.fa3_sparse_mla_forward_specs(main.shape,main.dtype,main.attributes,
        topk_capacity=main.launch.tiling.tile_shape['kv_capacity'])
    if parts!=expected:raise ValueError('Sparse source stages differ from their actual whole library call')
    if not isinstance(inputs,ElementwiseBindings) or not isinstance(storage,ElementwiseBindings):
        raise TypeError('Sparse source requires actual typed input and fresh storage bindings')
    if (storage.call_id,storage.rank,storage.stream,storage.visibility_domain)!=(
            inputs.call_id,inputs.rank,inputs.stream,inputs.visibility_domain):
        raise ValueError('Sparse source storage must retain its actual caller and lifetime domain')
    if storage.read_views or storage.input_ready or storage.write_views!=('output','lse','semaphore'):
        raise ValueError('Sparse fresh storage cannot declare precomputed or initialized scratch inputs')
    if any(event.kind not in (EventKind.ALLOCATED,EventKind.FREED) for event in storage.events):
        raise ValueError('Sparse fresh scheduler/partials/results require their actual source writers')
    input_work,input_views,input_allocations=_validate_fa3_sparse_mla_input_boundary(main,hardware,inputs,selected_rows)
    parents=unique_by(storage.views,'view_id');layouts=fa3_sparse_mla_call_storage_layouts(main,hardware)
    if set(parents)!={role for role,*_ in layouts}:raise ValueError('Sparse source storage roles differ from source ABI')
    b,h=main.shape['batch'],main.shape['num_query_heads'];s=main.launch.tiling.num_kv_splits
    rounded=ceil(b/4)*4;vectors=1+int(s>1);meta=parents['scheduler_metadata']
    for role,shape,dtype,width in layouts:
        view=parents[role]
        offset=(rounded*4 if role=='dynamic_splits' else vectors*rounded*4 if role=='semaphore' else 0)
        if (type(view) is not TensorView or (view.shape,view.dtype,view.element_bytes,view.offset_bytes,view.strides_bytes)!=(
                shape,dtype,width,offset,dense_strides(shape,width))):
            raise ValueError('Sparse source fresh storage shape/offset/strides differ')
        canonical=TensorView(view.view_id,view.allocation_id,view.offset_bytes,view.shape,view.strides_bytes,view.dtype,view.element_bytes)
        if view.region()!=canonical.region():raise ValueError('Sparse fresh storage derived region differs')
        if role in ('prepared_query','dynamic_splits','semaphore') and view.allocation_id!=meta.allocation_id:
            raise ValueError('Sparse scheduler fields must alias the actual metadata parent')
    roots=('scheduler_metadata','output','lse',*(('partial_output','partial_lse') if s>1 else ()))
    if len({parents[role].allocation_id for role in roots})!=len(roots):
        raise ValueError('Sparse source fresh tensor parents may not alias')
    allocations=(*inputs.allocations,*storage.allocations);allocation_map=unique_by(allocations,'allocation_id')
    if any(allocation.freed_event is not None for allocation in allocations):
        raise NotImplementedError('Sparse complete tensor export requires live call allocations; retirement binding remains unbound')
    if {item.allocation_id for item in storage.allocations}!={parents[role].allocation_id for role in roots}:
        raise ValueError('Sparse source storage requires exactly its actual native parent allocations')
    from src.kernel.memory_access import validate_storage_bindings
    validate_storage_bindings(allocations,(*inputs.views,*storage.views),(*inputs.events,*storage.events),
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual fresh sparse library storage;source writers remain in explicit export.',)))
    return {'main':main,'main_work':work,'input_views':input_views,'parents':parents,
        'allocations':allocations,'allocation_map':allocation_map}


def build_fa3_sparse_mla_input_source(kernel,hardware,bindings,granularity,*,program_id,selected_rows):
    """Explicit sparse input export, not a prediction hot path or complete call DAG.

    selected_rows is a caller-declared snapshot of each live block-table prefix,
    paired with that binding's actual input-ready event. Unknown values must not
    be replaced by synthetic rows. It proves selected storage/readiness under
    those caller facts, not device values, index conversion or emitted traffic.
    Bindings carry no tensor contents: equality with the device table/counts is
    a caller responsibility, not a numeric check performed by this exporter.
    """
    from src.kernel.memory_access import Span
    if granularity not in (Granularity.OP,Granularity.KERNEL):
        raise ValueError('Sparse FA3 resolved input export requires OP or KERNEL granularity')
    work,views,allocations=_validate_fa3_sparse_mla_input_boundary(kernel,hardware,bindings,selected_rows)
    batch,heads=kernel.shape['batch'],kernel.shape['num_query_heads']
    used=tuple(kernel.attributes['kv_lens']);capacity=kernel.launch.tiling.tile_shape['kv_capacity']
    stride=views['cache_rows'].strides_bytes[0];roles=bindings.read_views
    scope=AccessScope(program_id,('fa3_sparse_mla_input',str(batch),str(heads),str(capacity),
        ','.join(map(str,used))),bindings.call_id,(),bindings.rank,bindings.stream,'logical-fa3-sparse-selected-inputs')
    ready=dict(zip(roles,bindings.input_ready,strict=True));accesses=[]
    evidence=('Pinned vLLM0.30 flashattn_mla_sparse._run_mqa_kernel: q_rope64/q_nope512, page1 and shared latent512/RoPE64 rows.',
        'Live physical row values are explicit caller snapshot facts tied to the supplied block-table availability; no invented indices or capacity-as-used.',
        'Logical task payload matches the existing source demand. Metadata reads are boundary exposure, not native issue multiplicity.')
    def read(identity,role,region,dependencies=()):
        view=views[role];allocation=allocations[view.allocation_id]
        refs=tuple(dict.fromkeys((EventRef(allocation.allocated_event,EventKind.ALLOCATED),ready[role],*dependencies)))
        access=MemoryAccess(identity,scope.scope_id,view.allocation_id,view.view_id,Operation.READ,region,
            Semantics.LOGICAL,None,None,None,None,bindings.visibility_domain,refs,EventKind.READ_READY,1,evidence)
        accesses.append(access)
        return EventRef(identity,EventKind.READ_READY)
    for request,(length,count,rows) in enumerate(zip(used,work['splits'],selected_rows,strict=True)):
        label=f'{program_id}.r{request}'
        v=views['used_key'];length_ref=read(label+'.used_key','used_key',Span(v.offset_bytes+request*4,4))
        v=views['cu_query'];query_ref=read(label+'.cu_query','cu_query',Span(v.offset_bytes+request*4,8))
        panels=ceil(length/64);per_split=ceil(panels/count)
        for split in range(count):
            first=split*per_split*64;stop=min(length,(split+1)*per_split*64)
            if first>=stop:continue
            for m in range(ceil(heads/64)):
                task=f'{label}.s{split}.m{m}';live_heads=min(64,heads-m*64)
                control=(length_ref,query_ref)
                for role,width in (('query_rope',64),('query_latent',512)):
                    view=views[role]
                    region=Strided(view.offset_bytes+request*view.strides_bytes[0]+m*64*view.strides_bytes[1],
                        (live_heads,),(view.strides_bytes[1],),width*2)
                    read(task+'.'+role,role,region,control)
                view=views['block_table']
                index_ref=read(task+'.indices','block_table',Span(
                    view.offset_bytes+request*view.strides_bytes[0]+first*4,(stop-first)*4),control)
                for selected in range(first,stop):
                    for role,width in (('key_cache',64),('value_cache',512)):
                        view=views[role]
                        read(f'{task}.i{selected}.{role}',role,Span(
                            view.offset_bytes+rows[selected]*stride,width*2),(length_ref,index_ref))
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(granularity,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Explicit selected-input source boundary only; prepare/scratch/main-output/combine are not represented by this export.',
            'Caller snapshots are not GPU observations. Unknown global indices remain unsupported; duplicate selected rows are retained.',
            'No native page1 service, issued transaction, cache-hit, index-conversion cost or complete-call timing claim.')),
        bindings.allocations,bindings.views,(scope,),bindings.events,tuple(accesses))


def fa3_sparse_mla_input_source_work(program,*,kernel,hardware,bindings,selected_rows):
    """Consume exactly the exported source, including aliases and real readiness."""
    expected=build_fa3_sparse_mla_input_source(kernel,hardware,bindings,program.coverage.granularity,
        program_id=program.program_id,selected_rows=selected_rows)
    if program!=expected:raise ValueError('Sparse FA3 input source differs from the actual caller snapshot')
    by_view={role:0 for role in bindings.read_views}
    for access in program.accesses:by_view[access.view_id]+=requested_bytes(access.region)
    return {'read_bytes_by_view':by_view,'selected_rows':selected_rows,
        'physical_service_bound':False,'coverage':'selected_input_boundary_only'}


def fa3_sparse_mla_call_storage_layouts(main,hardware):
    """Source-owned fresh storage geometry; identities/readiness come from caller."""
    from src.kernel.predictors.flash_attn import _fa3_sparse_mla_work
    _fa3_sparse_mla_work(main,hardware)
    if main.variant!='mla_decode':raise ValueError('Sparse call storage requires its main spec')
    b,h=main.shape['batch'],main.shape['num_query_heads'];s=main.launch.tiling.num_kv_splits
    rounded=ceil(b/4)*4;vectors=1+int(s>1)
    layouts=[('scheduler_metadata',(vectors*rounded+1,),'int32',4),
        ('prepared_query',(b,),'int32',4),('semaphore',(1,),'int32',4),
        ('output',(b,h,512),'bf16',2),('lse',(h,b),'fp32',4)]
    if s>1:layouts.extend((('dynamic_splits',(b,),'int32',4),
        ('partial_output',(s,h,b,512),'fp32',4),('partial_lse',(s,h,b),'fp32',4)))
    return tuple(layouts)


def build_fa3_sparse_mla_call_source(parts,hardware,inputs,storage,granularity,*,program_id,selected_rows):
    """Explicit complete library tensor source; no physical-service or hot-path claim.

    Caller supplies the original prepare/main[/combine] tuple, actual input and
    fresh output/scratch bindings and a live physical-index snapshot. Source
    conversion, GPU content truth, queue atomics and page1 service remain open.
    """
    from src.kernel.memory_access import Span,TileWriteSource,covers
    from src.kernel.predictors.flash_attn import _fa3_sparse_mla_work
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('Unknown sparse source granularity')
    boundary=_validate_fa3_sparse_mla_call_boundary(parts,hardware,inputs,storage,selected_rows)
    main,work,parents=boundary['main'],boundary['main_work'],boundary['parents']
    allocations,allocation_map=boundary['allocations'],boundary['allocation_map']
    b,h=main.shape['batch'],main.shape['num_query_heads'];s=main.launch.tiling.num_kv_splits
    input_program=build_fa3_sparse_mla_input_source(main,hardware,inputs,Granularity.KERNEL,
        program_id=program_id+'.inputs',selected_rows=selected_rows)
    all_views=[*inputs.views,*storage.views];events=[*inputs.events,*storage.events];families=[]
    input_views=unique_by(inputs.views,'view_id');input_events=unique_by(inputs.events,'event_id')
    input_ready=dict(zip(inputs.read_views,inputs.input_ready,strict=True));visibility=inputs.visibility_domain
    header=('fa3_sparse_mla_call',str(b),str(h),str(s),','.join(map(str,main.attributes['kv_lens'])))
    scope=AccessScope(program_id,header,inputs.call_id,(),inputs.rank,inputs.stream,'logical-fa3-sparse-library-call')
    evidence=('Pinned source_506341 sparse BF16/Q1/page1 call. Actual external input snapshots; no production durations.',
        'Fresh prepare fields and main epilogues are real producers. Empty split partial O remains unwritten.',
        '512-wide combine source is reused unchanged; LSE-driven max-valid READ and conditional tails remain unknown.',
        'Queue worker assignment/atomics,issued traffic,physical service and actual runtime source binding remain partial.')
    def publish(view,family,name):
        event=ProgramEvent(family.tile_program_id+'.'+name+'.produced',EventKind.LOGICAL_AVAILABLE,
            view.allocation_id,Unmapped(view.shape,view.dtype,view.element_bytes,view.view_id+'.source_writer'),visibility,
            (EventRef(allocation_map[view.allocation_id].allocated_event,EventKind.ALLOCATED),),view.view_id,
            (TileWriteSource(family.tile_program_id,name),))
        events.append(event);return EventRef(event.event_id,event.kind)
    def child(identity,view,region):
        if isinstance(region,Span):
            shape=(region.length_bytes//view.element_bytes,);strides=(view.element_bytes,);offset=region.offset_bytes
        else:
            shape=(*region.counts,region.element_span_bytes//view.element_bytes)
            strides=(*region.strides_bytes,view.element_bytes);offset=region.base_offset_bytes
        result=TensorView(identity,view.allocation_id,offset,shape,strides,view.dtype,view.element_bytes)
        all_views.append(result);return result
    def actual_ready(role,view):
        # Select an actual available ancestor of this operand's supplied snapshot;
        # never manufacture a more convenient initializer for a selected row.
        reachable=set();pending=[input_ready[role]]
        while pending:
            ref=pending.pop()
            if ref.node_id in reachable:continue
            reachable.add(ref.node_id);pending.extend(input_events[ref.node_id].dependencies)
        for identity in sorted(reachable):
            event=input_events[identity]
            if (event.kind in (EventKind.INITIALIZED,EventKind.LOGICAL_AVAILABLE)
                    and event.allocation_id==view.allocation_id and event.visibility_domain==visibility
                    and event.region is not None and covers(event.region,view.region())):
                return EventRef(identity,event.kind)
        raise NotImplementedError('Sparse TILE export requires an actual available region covering each resolved field')
    prep_id=program_id+'.prepare';prep_access=[]
    for role in ('cu_query','used_key'):
        view=input_views[role]
        prep_access.append(TileAccessTemplate(role,view.view_id,Operation.READ,(),(None,)*len(view.shape),
            (actual_ready(role,view),),(),evidence))
    prepared=parents['prepared_query'];semaphore=parents['semaphore']
    prep_access.extend((
        TileAccessTemplate('prepared_query',prepared.view_id,Operation.WRITE,(),(None,),(),
            (TileDependency('cu_query',EventKind.READ_READY,()),),evidence,unmapped=True),
        TileAccessTemplate('semaphore',semaphore.view_id,Operation.WRITE,(),(None,),(),(),evidence,
            unmapped=True,constant_value=ConstantValue.ZERO)))
    if s>1:
        prep_access.append(TileAccessTemplate('dynamic_splits','dynamic_splits',Operation.WRITE,(),(None,),(),
            (TileDependency('cu_query',EventKind.READ_READY,()),TileDependency('used_key',EventKind.READ_READY,())),
            evidence,unmapped=True))
    prep=TileProgram(prep_id,replace(scope,scope_id=prep_id,op_ids=(*header,'prepare')),
        (TileAxis('prepare',1,1),),tuple(prep_access),visibility,evidence);families.append(prep)
    fields=('prepared_query','semaphore',*(('dynamic_splits',) if s>1 else ()))
    field_ready={role:publish(parents[role],prep,role) for role in fields}
    prep_done=ProgramEvent(program_id+'.prepare.tensor_outputs_complete',EventKind.COMPUTE_DONE,None,None,None,
        tuple(field_ready.values()));events.append(prep_done)
    # The real prepare->main grid/stream boundary includes semaphore zero;
    # retain it without inventing queue atomic counts or a timed global fence.
    event=ProgramEvent(program_id+'.prepared_query.after_prepare',EventKind.LOGICAL_AVAILABLE,
        prepared.allocation_id,Unmapped(prepared.shape,prepared.dtype,prepared.element_bytes,
            program_id+'.prepared_query.source_stage_order'),visibility,
        (field_ready['prepared_query'],EventRef(prep_done.event_id,prep_done.kind)),prepared.view_id)
    events.append(event);field_ready['prepared_query']=EventRef(event.event_id,event.kind)
    flat={access.access_id:access for access in input_program.accesses};writes={r:[] for r in range(b)}
    lse_writes={r:[] for r in range(b)};results=[];all_main=[]
    for request,(length,count) in enumerate(zip(main.attributes['kv_lens'],work['splits'],strict=True)):
        per_split=ceil(ceil(length/64)/count)
        for split in range(count):
            live=min(length,(split+1)*per_split*64)>split*per_split*64
            for m in range(ceil(h/64)):
                identity=f'{program_id}.main.r{request}.s{split}.m{m}';accesses=[];rows=min(64,h-m*64)
                prefix=f'{program_id}.inputs.r{request}';task=f'{prefix}.s{split}.m{m}'
                selected=[flat[prefix+'.'+role] for role in ('used_key','cu_query')]
                selected.extend(access for name,access in flat.items() if name.startswith(task+'.'))
                names={access.access_id:access.access_id[len(task)+1:] if access.access_id.startswith(task+'.')
                    else access.access_id[len(prefix)+1:] for access in selected}
                for role in ('prepared_query',*(('dynamic_splits',) if s>1 else ())):
                    parent=parents[role]
                    view=LogicalSelectionView(identity+'.'+role,parent.allocation_id,(1,),'int32',4,identity+'.'+role,
                        role,(LogicalCoordinateBinding(0,(),request),))
                    all_views.append(view)
                    accesses.append(TileAccessTemplate(role,view.view_id,Operation.READ,(),(None,),
                        (field_ready[role],),(),evidence,unmapped=True))
                control=tuple(TileDependency(role,EventKind.READ_READY,()) for role in
                    ('prepared_query',*(('dynamic_splits',) if s>1 else ())))
                for access in selected:
                    role=access.view_id;view=child(identity+'.'+names[access.access_id],input_views[role],access.region)
                    dependencies=tuple(TileDependency(names[ref.node_id],EventKind.READ_READY,())
                        for ref in access.dependencies if ref.node_id in names)
                    accesses.append(TileAccessTemplate(names[access.access_id],view.view_id,Operation.READ,(),
                        (None,)*len(view.shape),(actual_ready(role,view),),(*control,*dependencies),evidence))
                deps=tuple(TileDependency(access.access_id,EventKind.READ_READY,()) for access in accesses)
                output=parents['output' if count==1 else 'partial_output'];lse=parents['lse' if count==1 else 'partial_lse']
                if count==1:
                    o_offset=request*h*1024+m*64*1024;o_stride=1024
                    l_offset=(m*64*b+request)*4;l_stride=b*4
                else:
                    o_offset=(split*h*b+m*64*b+request)*512*4;o_stride=b*512*4
                    l_offset=(split*h*b+m*64*b+request)*4;l_stride=b*4
                l_view=TensorView(identity+'.lse',lse.allocation_id,l_offset,(rows,),(l_stride,),'fp32',4);all_views.append(l_view)
                accesses.append(TileAccessTemplate('lse',l_view.view_id,Operation.WRITE,(),(None,),(),deps,evidence,
                    unmapped=True,constant_value=ConstantValue.NEGATIVE_INFINITY if not live else None))
                if live or count==1:
                    o_view=TensorView(identity+'.output',output.allocation_id,o_offset,(rows,512),
                        (o_stride,output.element_bytes),output.dtype,output.element_bytes);all_views.append(o_view)
                    accesses.append(TileAccessTemplate('output',o_view.view_id,Operation.WRITE,(),(None,None),(),deps,
                        evidence,unmapped=True,constant_value=ConstantValue.ZERO if not live else None))
                family=TileProgram(identity,replace(scope,scope_id=identity,op_ids=(*header,'main',str(request),str(split),str(m))),
                    (TileAxis('epilogue',1,1),),tuple(accesses),visibility,evidence);families.append(family)
                ref=publish(l_view,family,'lse');lse_writes[request].append(ref);all_main.append(ref)
                if count==1:results.append(ref)
                if live or count==1:
                    ref=publish(o_view,family,'output');writes[request].append(ref);all_main.append(ref)
                    if count==1:results.append(ref)
    main_done=ProgramEvent(program_id+'.main.complete',EventKind.COMPUTE_DONE,None,None,None,tuple(all_main));events.append(main_done)
    if s>1:
        combine=parts[2];_fa3_sparse_mla_work(combine,hardware)
        combine_header=('fa3_split_combine',','.join('1' for _ in range(b)),','.join(map(str,main.attributes['kv_lens'])),
            str(h),str(s),str(int(hardware.num_compute_units)),'causal',str(b),'bf16')
        if _fa3_combine_geometry(combine_header)[-1]!=work['splits']:
            raise ValueError('Sparse complete-call combine must keep the actual source dynamic splits')
        combine_parents={role:parents[role] for role in ('scheduler_metadata','dynamic_splits','semaphore',
            'partial_output','partial_lse','output','lse')};combine_parents['cu_query']=input_views['cu_query']
        after={}
        for role in ('dynamic_splits','semaphore'):
            view=parents[role];event=ProgramEvent(program_id+'.'+role+'.after_main',EventKind.LOGICAL_AVAILABLE,
                view.allocation_id,Unmapped(view.shape,view.dtype,view.element_bytes,program_id+'.'+role+'.stage_order'),visibility,
                (field_ready[role],EventRef(main_done.event_id,main_done.kind)),view.view_id)
            events.append(event);after[role]=EventRef(event.event_id,event.kind)
        combine_ready={'cu_query':input_ready['cu_query'],'dynamic_splits':after['dynamic_splits']}
        for request,count in enumerate(work['splits']):
            if count<=1:continue
            for role in ('partial_output','partial_lse'):
                root=parents[role];shape=(count,h,1,512) if role=='partial_output' else (count,h,1)
                view=TensorView(f'{role}.r{request}',root.allocation_id,request*root.strides_bytes[2],shape,
                    root.strides_bytes,'fp32',4);all_views.append(view);combine_parents[view.view_id]=view
                refs=tuple(lse_writes[request]) if role=='partial_lse' else tuple((*writes[request],*lse_writes[request]))
                event=ProgramEvent(program_id+'.'+view.view_id+'.accessible_after_main',EventKind.LOGICAL_AVAILABLE,
                    view.allocation_id,Unmapped(view.shape,view.dtype,view.element_bytes,
                        program_id+'.'+view.view_id+'.source_accessibility_not_empty_O_initialization'),visibility,
                    (EventRef(allocation_map[view.allocation_id].allocated_event,EventKind.ALLOCATED),*refs),view.view_id)
                events.append(event);combine_ready[view.view_id]=EventRef(event.event_id,event.kind)
        internal,children,combine_events,combine_families=_fa3_combine_recipe(combine_header,combine_parents,combine_ready,
            replace(scope,scope_id=program_id+'.combine',op_ids=combine_header),visibility,
            storage.binding_id+'.combine',allocation_map[parents['output'].allocation_id].device_id)
        allocation_map.update(unique_by(internal,'allocation_id'));children=list(children)
        # Rebind the existing source's final logical stores to their exact typed
        # parent regions; no copy, new writer or guessed availability is added.
        child_map=unique_by(children,'view_id');combine_families=list(combine_families)
        for index,family in enumerate(combine_families):
            for access in family.accesses:
                if access.access_id not in ('output','lse'):continue
                old=child_map[access.view_id];parent=combine_parents[old.parent_view_id]
                starts=tuple(binding.offset for binding in old.coordinate_map)
                offset=parent.offset_bytes+sum(start*stride for start,stride in zip(starts,parent.strides_bytes,strict=True))
                view=TensorView(old.view_id,parent.allocation_id,offset,old.shape,(parent.element_bytes,),parent.dtype,parent.element_bytes)
                children[children.index(old)]=view;child_map[view.view_id]=view
                results.append(publish(view,family,access.access_id))
            if family.scope.op_ids[-1]=='reset':
                family=replace(family,accesses=tuple(replace(access,input_ready=(after['semaphore'],))
                    if access.access_id=='semaphore_reset' else access for access in family.accesses))
                combine_families[index]=family;results.append(publish(parents['semaphore'],family,'semaphore_reset'))
        allocations=(*allocations,*internal);all_views.extend(children);events.extend(combine_events);families.extend(combine_families)
    complete=ProgramEvent(program_id+'.results_and_optional_reset.complete',EventKind.COMPUTE_DONE,None,None,None,tuple(results));events.append(complete)
    for role in ('output','lse'):
        view=parents[role]
        events.append(ProgramEvent(program_id+'.'+role+'.library_result',EventKind.LOGICAL_AVAILABLE,view.allocation_id,
            Unmapped(view.shape,view.dtype,view.element_bytes,program_id+'.'+role+'.actual_writer_union'),visibility,
            (EventRef(allocation_map[view.allocation_id].allocated_event,EventKind.ALLOCATED),EventRef(complete.event_id,complete.kind)),role))
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,storage.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,evidence),tuple(allocations),tuple(all_views),
        (scope,),tuple(events),(),tuple(families))


def fa3_sparse_mla_call_source_work(program,*,parts,hardware,inputs,storage,selected_rows):
    from src.kernel.memory_access import _tile_template_requested_bytes
    expected=build_fa3_sparse_mla_call_source(parts,hardware,inputs,storage,Granularity.TILE,
        program_id=program.program_id,selected_rows=selected_rows)
    if program!=expected:raise ValueError('Sparse call lost actual source writers,aliases,readiness or caller snapshot')
    views=unique_by(program.views,'view_id');parents=unique_by(storage.views,'view_id')
    totals={stage:{role:0 for role in ('output','lse','partial_output','partial_lse','semaphore')}
        for stage in ('prepare','main','combine')}
    for family in program.tile_programs:
        stage='prepare' if family.tile_program_id.endswith('.prepare') else 'main' if '.main.r' in family.tile_program_id else 'combine'
        axes=unique_by(family.axes,'axis_id')
        for access in family.accesses:
            if access.operation is not Operation.WRITE:continue
            view=views[access.view_id]
            for role in totals[stage]:
                if role in parents and view.allocation_id==parents[role].allocation_id:
                    # Scheduler slices share a parent; only the actual reset/zero is charged here.
                    if role=='semaphore' and access.access_id not in ('semaphore','semaphore_reset'):continue
                    totals[stage][role]+=_tile_template_requested_bytes(views,axes,family,access)
    return {'write_bytes_by_stage_and_parent':totals,'physical_service_bound':False,
        'runtime_integrated':False,'coverage':'complete_library_tensor_source_only'}


def fa3_sparse_mla_call_result_bindings(program,*,parts,hardware,inputs,storage,selected_rows,call_id,binding_id):
    """Expose actual final O/LSE parents after their source writers/optional reset."""
    from src.kernel.memory_access_builder import ElementwiseBindings
    fa3_sparse_mla_call_source_work(program,parts=parts,hardware=hardware,inputs=inputs,storage=storage,selected_rows=selected_rows)
    refs=tuple(EventRef(program.program_id+'.'+role+'.library_result',EventKind.LOGICAL_AVAILABLE)
        for role in ('output','lse'))
    return ElementwiseBindings(binding_id,call_id,program.allocations,program.views,program.events,
        ('output','lse'),(),refs,inputs.rank,inputs.stream,inputs.visibility_domain)


_FA3_SPARSE_MLA_PREDICTION = ContextVar('fa3_sparse_mla_prediction')


def fa3_sparse_mla_call_prediction_work(parts,hardware,inputs,storage,*,selected_rows):
    """Validate real whole-call facts without constructing any detailed source graph."""
    from src.kernel.predictors.flash_attn import _fa3_sparse_mla_work
    boundary=_validate_fa3_sparse_mla_call_boundary(parts,hardware,inputs,storage,selected_rows)
    main,mw,parents=boundary['main'],boundary['main_work'],boundary['parents']
    cw=_fa3_sparse_mla_work(parts[2],hardware) if len(parts)==3 else None
    output=prod(parents['output'].shape)*2;lse=prod(parents['lse'].shape)*4
    combine_write=0 if cw is None else cw['write_bytes']
    reset=0 if cw is None else cw['semaphore_reset_bytes']
    if mw['direct_write_bytes']+combine_write-reset!=output+lse:
        raise ValueError('Sparse library source result bytes differ from actual output parents')
    if cw is not None and cw['splits']!=mw['splits']:
        raise ValueError('Sparse paired source dynamic splits differ')
    return {'main_write_bytes':mw['write_bytes'],'main_l2_read_bytes':mw['logical_l2_read_bytes'],
        'main_direct_write_bytes':mw['direct_write_bytes'],'partial_output_write_bytes':mw['partial_output_bytes'],
        'partial_lse_write_bytes':mw['partial_lse_bytes'],'combine_write_bytes':combine_write,
        'output_bytes':output,'lse_bytes':lse,'splits':mw['splits'],'semaphore_reset_bytes':reset,
        'physical_service_bound':False,'device_snapshot_truth_verified':False}


@contextmanager
def fa3_sparse_mla_call_prediction_scope(parts,hardware,inputs,storage,*,selected_rows):
    """Keep the actual sparse call and declared index snapshot through pricing."""
    from copy import deepcopy
    work=fa3_sparse_mla_call_prediction_work(parts,hardware,inputs,storage,selected_rows=selected_rows)
    seal=(deepcopy(inputs),deepcopy(storage));kernels=repr(parts)
    token=_FA3_SPARSE_MLA_PREDICTION.set((parts,hardware,inputs,storage,selected_rows,work,seal,kernels))
    try:
        yield work
    finally:
        _FA3_SPARSE_MLA_PREDICTION.reset(token)


def current_fa3_sparse_mla_prediction_work(kernel,hardware):
    from src.kernel.memory_access_builder import _prediction_bindings
    context=copy_context()
    if _FA3_SPARSE_MLA_PREDICTION not in context:return None
    parts,actual_hardware,inputs,storage,rows,work,seal,kernels=context[_FA3_SPARSE_MLA_PREDICTION]
    if hardware is not actual_hardware or not any(kernel is part for part in parts[1:]) or repr(parts)!=kernels:
        raise ValueError('Sparse prediction crossed or changed its actual source call')
    actual,_=_prediction_bindings(kernel,None)
    expected=inputs if kernel is parts[1] else storage
    if actual is not expected or (inputs,storage)!=seal:
        raise ValueError('Sparse prediction requires unchanged actual caller bindings')
    observed=fa3_sparse_mla_call_prediction_work(parts,hardware,inputs,storage,selected_rows=rows)
    if observed!=work:raise ValueError('Sparse prediction source quantities changed')
    return observed
