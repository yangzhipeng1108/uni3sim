"""Typed CANN routing interfaces and source-owned metadata work."""
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,MemoryAccess,
    MemoryAccessProgram,MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,
    TensorView,LogicalSelectionView,Unmapped,requested_bytes,unique_by,TileAxis,TileAccessTemplate,TileDependency,TileProgram,
)


def has_cann_moe_gating_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant == 'moe_gating_topk'
        and kernel.execution_support == 'supported' and hardware.kernel_memory_model == 'npu_tile')


def has_cann_moe_expert_cast_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant == 'moe_cast'
        and kernel.attributes['cast_role'] == 'expert_tokens' and kernel.execution_support == 'supported'
        and hardware.kernel_memory_model == 'npu_tile')


def has_cann_moe_unbound_cast_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant=='moe_cast'
        and kernel.attributes['cast_role']=='topk_ids' and kernel.execution_support=='supported'
        and hardware.kernel_memory_model=='npu_tile')


def standalone_cann_unbound_cast_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    shape=tuple(kernel.shape[name] for name in ('tokens','topk'));allocations=[];views=[];events=[]
    for role in ('input','output'):
        identity=f'{binding_id}.{role}'
        allocation=MemoryAllocation(identity,'logical-local-device','logical-storage',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=LogicalSelectionView(role,identity,shape,None,None,f'{binding_id}.{role}.logical_ids',None)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role=='input':
            event=ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,view.region(),'cast-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready=EventRef(event.event_id,event.kind)
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('input',),('output',),(ready,),0,'offline','cast-visible')


def build_cann_unbound_cast_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_moe_unbound_cast_source(kernel,hardware):raise ValueError('Unknown-storage cast requires its declared topk_ids role')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('Unknown-storage cast requires actual operand bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown-storage cast exposes logical scalar tiles')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(('input',),('output',),1):
        raise ValueError('Unknown-storage cast requires its input/output roles')
    evidence=('KernelSpec VECTOR/moe_cast topk_ids declares one pointwise conversion for each routed ID.',
        'The legacy descriptor does not supply input/output storage encodings: no INT32/INT64 conversion or byte-copy ABI is inferred.',
        'The existing 4-byte input plus4-byte output price is retained as a named legacy projection, never claimed as requested bytes.')
    tokens,topk=(kernel.shape[name] for name in ('tokens','topk'))
    scope=AccessScope(program_id,('cann_unbound_cast','topk_ids',str(tokens),str(topk)),bindings.call_id,(),
        bindings.rank,bindings.stream,'logical-unbound-cast')
    read=TileAccessTemplate('input','input',Operation.READ,('token','route'),('token','route'),bindings.input_ready,(),evidence,unmapped=True)
    write=TileAccessTemplate('output','output',Operation.WRITE,('token','route'),('token','route'),(),
        (TileDependency('input',EventKind.READ_READY,()),),evidence,unmapped=True)
    family=TileProgram(program_id,scope,(TileAxis('token',tokens,1),TileAxis('route',topk,1)),(read,write),bindings.visibility_domain,evidence)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Logical element correspondence only: dtype/width, byte counts, parent coordinates, native layout and issue are unbound.',
            'Consumers needing exact bytes or executable storage encoding must reject the unknown fields.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    cann_unbound_cast_source_demands(program,expected_kernel=kernel)
    return program


def cann_unbound_cast_source_demands(program, *, expected_kernel):
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:raise ValueError('Unknown-storage cast requires its unique scalar tile source')
    family,=program.tile_programs;views=unique_by(program.views,'view_id');kernel=expected_kernel
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant!='moe_cast' or kernel.attributes['cast_role']!='topk_ids'
            or kernel.execution_support!='supported' or kernel.hardware_id!=program.hardware_id
            or program.hardware_id!='ascend_910b'):
        raise ValueError('Unknown-storage cast differs from its actual kernel')
    shape=tuple(kernel.shape[name] for name in ('tokens','topk'))
    if (family.scope.op_ids!=('cann_unbound_cast','topk_ids',*map(str,shape)) or family.scope.agent!='logical-unbound-cast'
            or family.axes!=(TileAxis('token',shape[0],1),TileAxis('route',shape[1],1))
            or family.partitions or family.causal_prefixes or set(views)!={'input','output'}
            or views['input'].allocation_id==views['output'].allocation_id):
        raise ValueError('Unknown-storage cast must retain each actual routed-ID coordinate')
    for role in ('input','output'):
        view=views[role]
        if (not isinstance(view,LogicalSelectionView) or view.shape!=shape or view.dtype is not None
                or view.element_bytes is not None or view.parent_view_id is not None or view.coordinate_map):
            raise ValueError('Unknown-storage cast cannot invent input/output storage encoding or parent mapping')
    if len(family.accesses)!=2:raise ValueError('Unknown-storage cast must retain exactly its input/output')
    read,write=family.accesses
    if (read.access_id!='input' or write.access_id!='output' or len(read.input_ready)!=1
            or read!=TileAccessTemplate('input','input',Operation.READ,('token','route'),('token','route'),read.input_ready,(),family.evidence_refs,unmapped=True)
            or write!=TileAccessTemplate('output','output',Operation.WRITE,('token','route'),('token','route'),(),
                (TileDependency('input',EventKind.READ_READY,()),),family.evidence_refs,unmapped=True)):
        raise ValueError('Unknown-storage cast output requires its actual per-element READ_READY')
    elements=shape[0]*shape[1]
    return {'elements':elements,'requested_read_bytes':None,'requested_write_bytes':None,
        'legacy_read_price_bytes':elements*4,'legacy_write_price_bytes':elements*4,'storage_encoding_bound':False}


def has_cann_moe_init_source(kernel,hardware):
    supported=(kernel.kind is KernelKind.VECTOR and kernel.variant in ('moe_init_routing','moe_init_routing_v2','moe_init_routing_v3')
        and kernel.dtype in ('bf16','fp16') and kernel.execution_support == 'supported'
        and hardware.kernel_memory_model == 'npu_tile')
    if not supported or kernel.variant=='moe_init_routing':return supported
    return has_cann_native_init_domain(kernel)


def has_cann_native_init_domain(kernel):
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant not in ('moe_init_routing_v2','moe_init_routing_v3')
            or kernel.execution_support!='supported' or kernel.dtype not in ('bf16','fp16')):return False
    if 'model_input_shape' in kernel.attributes:return False
    return ('active_expert_range' not in kernel.attributes
        or kernel.attributes['active_expert_range']==(0,kernel.shape['experts']))


def has_cann_moe_transpose_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant == 'moe_transpose'
        and kernel.execution_support == 'supported' and hardware.kernel_memory_model == 'npu_tile'
        and kernel.shape['tokens'] > 0 and kernel.shape['topk'] > 0)


def cann_transpose_execution_kernel(kernel):
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant!='moe_transpose'
            or 'model_input_shape' not in kernel.attributes):return kernel
    if kernel.attributes['graph_runtime_mode'] not in ('FULL','PIECEWISE'):return kernel
    from dataclasses import replace
    physical=kernel.attributes['model_input_shape']
    if any(name not in physical for name in ('tokens','topk')):
        raise ValueError('Captured transpose requires actual token and TOPK extents')
    if all(kernel.shape[name]==physical[name] for name in ('tokens','topk')):return kernel
    shape={**kernel.shape,**physical};attributes=dict(kernel.attributes)
    attributes['graph_logical_tokens']=attributes['graph_model_input_tokens']
    return replace(kernel,shape=shape,attributes=attributes)


def is_cann_transpose_alias(kernel):
    kernel=cann_transpose_execution_kernel(kernel)
    return (kernel.kind is KernelKind.VECTOR and kernel.variant=='moe_transpose'
        and min(kernel.shape['tokens'],kernel.shape['topk'])==1)


def standalone_cann_transpose_alias_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    kernel=cann_transpose_execution_kernel(kernel)
    if not is_cann_transpose_alias(kernel):raise ValueError('Transpose alias requires a singleton dimension')
    tokens,topk=(kernel.shape[key] for key in ('tokens','topk'));identity=f'{binding_id}.indices'
    allocation=MemoryAllocation(identity,'logical-local-device','logical-storage',tokens*topk*4,
        AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
    source=TensorView('input0',identity,0,(tokens,topk),(4,tokens*4),'int32',4)
    output=TensorView('output0',identity,0,(tokens,topk),(topk*4,4),'int32',4)
    allocated=ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,())
    available=ProgramEvent(f'{identity}.available',EventKind.INITIALIZED,identity,source.region(),
        'transpose-visible',(EventRef(allocated.event_id,allocated.kind),))
    return ElementwiseBindings(binding_id,call_id,(allocation,),(source,output),(allocated,available),
        ('input0',),('output0',),(EventRef(available.event_id,available.kind),),0,'offline','transpose-visible')


def build_cann_transpose_alias_source(kernel,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    kernel=cann_transpose_execution_kernel(kernel)
    if not is_cann_transpose_alias(kernel) or granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('Transpose alias requires its declared singleton view ABI')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('Transpose alias requires actual index bindings')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(('input0',),('output0',),1):
        raise ValueError('Transpose alias requires its actual input availability and output view')
    tokens,topk=(kernel.shape[key] for key in ('tokens','topk'))
    scope=AccessScope(program_id,('moe_transpose_alias',str(tokens),str(topk)),bindings.call_id,(),
        bindings.rank,bindings.stream,'logical-zero-memory-view')
    output=next(view for view in bindings.views if view.view_id=='output0')
    published=ProgramEvent(f'{program_id}.view_available',EventKind.COMPUTE_DONE,None,None,None,bindings.input_ready)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'arange(T*K).view(K,T).transpose(0,1) is already contiguous when T=1 or K=1.',
            'contiguous() returns the same storage; view publication has no tensor reads, writes, arithmetic or device launch.')),
        bindings.allocations,bindings.views,(scope,),(*bindings.events,published),())
    cann_transpose_alias_demands(program);return program


def cann_transpose_alias_demands(program):
    program.__post_init__()
    if program.accesses or program.tile_programs or len(program.scopes)!=1 or len(program.views)!=2 or len(program.allocations)!=1:
        raise ValueError('Transpose view alias cannot invent memory operations or a second object')
    scope,=program.scopes
    if len(scope.op_ids)!=3 or scope.op_ids[0]!='moe_transpose_alias':raise ValueError('Unknown transpose alias declaration')
    tokens,topk=map(int,scope.op_ids[1:]);views=unique_by(program.views,'view_id')
    if min(tokens,topk)!=1 or set(views)!={'input0','output0'}:raise ValueError('Only singleton transpose is a contiguous alias')
    allocation,=program.allocations
    for role,strides in (('input0',(4,tokens*4)),('output0',(topk*4,4))):
        view=views[role]
        if (view.allocation_id,view.offset_bytes,view.shape,view.strides_bytes,view.dtype,view.element_bytes)!=(
                allocation.allocation_id,0,(tokens,topk),strides,'int32',4):raise ValueError('Transpose alias lost its actual INT32 shared storage/reshape')
    published=tuple(event for event in program.events if event.event_id==f'{program.program_id}.view_available')
    if (len(published)!=1 or published[0].kind is not EventKind.COMPUTE_DONE or len(published[0].dependencies)!=1
            or published[0].allocation_id is not None or published[0].region is not None):
        raise ValueError('Transpose alias must forward actual input availability without a tensor write')
    ready=unique_by(program.events,'event_id')[published[0].dependencies[0].node_id]
    if ready.kind not in (EventKind.INITIALIZED,EventKind.LOGICAL_AVAILABLE) or ready.allocation_id!=allocation.allocation_id:
        raise ValueError('Transpose alias lost its actual input availability')
    return {'tokens':tokens,'topk':topk,'read_bytes':0,'write_bytes':0}


def cann_moe_transpose_source_demands(program):
    """arange(T*K).view(K,T).transpose(0,1).contiguous() INT32 views."""
    if program.scopes and program.scopes[0].op_ids[:1]==('moe_transpose_alias',):return cann_transpose_alias_demands(program)
    if program.tile_programs:
        from src.kernel.memory_path_solver import scalar_tile_source_work
        work = scalar_tile_source_work(program)
        if work['scope'].op_ids != ('moe_transpose',) or len(work['reads']) != 1 or len(work['writes']) != 1 or len(work['shape']) != 2:
            raise ValueError('Legacy routing transpose requires one input/output')
        source, = work['reads']; destination, = work['writes']; tokens,topk = work['shape']
        if (min(tokens,topk) <= 1 or source.allocation_id == destination.allocation_id
                or any((view.dtype,view.element_bytes,view.strides_bytes) != ('int32',4,strides)
                    for view,strides in ((source,(4,tokens*4)),(destination,(topk*4,4))))):
            raise ValueError('Legacy routing transpose requires actual INT32 transposed-input/contiguous-output strides')
        return {'tokens':tokens,'topk':topk,'read_bytes':work['read_bytes'],'write_bytes':work['write_bytes']}
    raise ValueError('Transpose projection requires its current tile source or explicit view alias')


def _init_layouts(tokens,topk,hidden,dtype):
    return (('hidden',(tokens,hidden),dtype,2),('row_idx',(tokens,topk),'int32',4),
        ('expert_idx',(tokens,topk),'int32',4),('expanded_x',(tokens*topk,hidden),dtype,2),
        ('expanded_row_idx',(tokens*topk,),'int32',4),('expanded_expert_idx',(tokens*topk,),'int32',4))


def _init_region(view,binding_id):
    return Unmapped(view.shape,view.dtype,view.element_bytes,f'{binding_id}.{view.view_id}.native_init_routing')


def standalone_cann_moe_init_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if kernel.variant!='moe_init_routing':
        return _standalone_native_init_bindings(kernel,call_id=call_id,binding_id=binding_id)
    allocations,views,events,ready = [],[],[],[]
    layouts = _init_layouts(*(kernel.shape[key] for key in ('tokens','topk','hidden')),kernel.dtype)
    for index,(role,shape,dtype,width) in enumerate(layouts):
        identity = f'{binding_id}.{role}'
        strides,size = [],width
        for axis in reversed(shape):
            strides.append(size); size *= axis
        allocation = MemoryAllocation(identity,'logical-local-device','logical-native-tensor',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width)
        allocations.append(allocation); views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if index < 3:
            event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
                _init_region(view,binding_id),'cann-init-visible',(EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event); ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('hidden','row_idx','expert_idx'),('expanded_x','expanded_row_idx','expanded_expert_idx'),
        tuple(ready),0,'offline','cann-init-visible')


def build_cann_moe_init_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if kernel.variant!='moe_init_routing':
        return _build_native_init_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('Legacy CANN InitRouting requires three typed inputs and outputs')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('Legacy InitRouting exports logical token/route selections')
    roles = ('hidden','row_idx','expert_idx','expanded_x','expanded_row_idx','expanded_expert_idx')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (roles[:3],roles[3:],3):
        raise ValueError('Legacy InitRouting must retain all public tensor operands')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    layouts = _init_layouts(*(kernel.shape[key] for key in ('tokens','topk','hidden')),kernel.dtype)
    if set(views) != set(roles) or len({view.allocation_id for view in views.values()}) != 6:
        raise ValueError('Legacy InitRouting input/output native alias is unbound')
    for role,shape,dtype,width in layouts:
        if (views[role].shape,views[role].dtype,views[role].element_bytes) != (shape,dtype,width):
            raise ValueError('Legacy InitRouting shape/storage dtype mismatch')
    tokens,topk,hidden = (kernel.shape[key] for key in ('tokens','topk','hidden'))
    evidence = ('Ascend fused_moe_runner._routing: hidden, INT32 row_idx/expert_idx -> expanded_x and two INT32 expanded indices',
        'row_idx is arange(T*K).view(K,T).transpose(0,1).contiguous; routing values/native permutation remain unbound',
        'The later compute_expert_tokens call is separate; V2/V3 metadata/scales are different ABIs')
    evidence += ('One logical input-token tile feeds its TOPK route outputs. Output positions retain the unknown sorting map, not invented contiguous offsets.',
        'Global expert sorting requires all row/expert metadata reads; independent output tokens do not wait on unrelated hidden payload reads.')
    reads = tuple(TileAccessTemplate(role,role,Operation.READ,('token',),('token',None),
        (ready,),(),evidence,unmapped=True) for role,ready in zip(roles[:3],bindings.input_ready,strict=True))
    mapping_ready = tuple(TileDependency(role,EventKind.READ_READY,('token',)) for role in roles[1:3])
    writes = []
    for role in roles[3:]:
        parent = views[role]
        selected = LogicalSelectionView(f'{program_id}.{role}.by_input_route',parent.allocation_id,
            (tokens,topk,hidden) if role == 'expanded_x' else (tokens,topk),parent.dtype,parent.element_bytes,
            f'{bindings.binding_id}.{role}.input_token_route_sort_selection',role)
        views[selected.view_id] = selected
        dependencies = ((TileDependency('hidden',EventKind.READ_READY,()),) if role == 'expanded_x' else ()) + mapping_ready
        writes.append(TileAccessTemplate(role,selected.view_id,Operation.WRITE,('token','route'),
            ('token','route',None) if role == 'expanded_x' else ('token','route'),(),dependencies,evidence,unmapped=True))
    scope = AccessScope(program_id,('cann_moe_init',),bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-route-expansion')
    family = TileProgram(program_id,scope,(TileAxis('token',tokens,1),TileAxis('route',topk,1)),
        (*reads,*writes),bindings.visibility_domain,evidence)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Legacy V1 three-input/three-output logical interface, not an internal sorting trace.',
             'Native capacity/layout/permutation, scratch/queues and cross-call producer identities remain unbound.')),
        bindings.allocations,tuple(views.values()),(),bindings.events,(),(family,))
    cann_moe_init_source_demands(program)
    return program


def cann_moe_init_source_demands(program):
    if (any(family.scope.op_ids and family.scope.op_ids[0]=='cann_native_init' for family in program.tile_programs)
            or any(scope.op_ids and scope.op_ids[0]=='cann_native_init' for scope in program.scopes)):
        return _native_init_demands(program)
    if program.tile_programs:
        return cann_moe_init_tile_demands(program)
    views,scopes = unique_by(program.views,'view_id'),unique_by(program.scopes,'scope_id')
    roles = ('hidden','row_idx','expert_idx','expanded_x','expanded_row_idx','expanded_expert_idx')
    if set(views) != set(roles) or len(program.accesses) != 6:
        raise ValueError('Legacy InitRouting requires all six public operands')
    if len(views['hidden'].shape) != 2 or len(views['row_idx'].shape) != 2:
        raise ValueError('Legacy InitRouting requires explicit token/hidden/TOPK dimensions')
    tokens,hidden = views['hidden'].shape
    topk = views['row_idx'].shape[1]
    dtype = views['hidden'].dtype
    if dtype not in ('bf16','fp16') or len({view.allocation_id for view in views.values()}) != 6:
        raise ValueError('Legacy InitRouting floating storage/native alias mismatch')
    for index,(access,(role,shape,storage,width)) in enumerate(zip(program.accesses,_init_layouts(tokens,topk,hidden,dtype),strict=True)):
        view = views[role]
        if ((view.shape,view.dtype,view.element_bytes) != (shape,storage,width)
                or scopes[access.scope_id].op_ids != ('cann_moe_init',role)
                or access.allocation_id != view.allocation_id or access.view_id is not None
                or access.operation is not (Operation.READ if index < 3 else Operation.WRITE)
                or access.region != _init_region(view,program.binding_id) or access.repeat != 1):
            raise ValueError('Legacy InitRouting must conserve actual BF16/FP16 and INT32 requests')
    nodes = {event.event_id:event for event in program.events}
    nodes.update({access.access_id:access for access in program.accesses})
    required = {EventRef(access.access_id,EventKind.READ_READY) for access in program.accesses[:3]}
    for output in program.accesses[3:]:
        pending,ancestors = list(output.dependencies),set()
        while pending:
            ref = pending.pop()
            if ref not in ancestors:
                ancestors.add(ref); pending.extend(nodes[ref.node_id].dependencies)
        if not required <= ancestors:
            raise ValueError('Legacy InitRouting outputs require actual hidden/row_idx/expert_idx READ_READY')
    return {'tokens':tokens,'topk':topk,'hidden':hidden,
        'read_bytes':sum(requested_bytes(access.region) for access in program.accesses[:3]),
        'write_bytes':sum(requested_bytes(access.region) for access in program.accesses[3:])}


def cann_moe_init_tile_demands(program):
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    if len(program.tile_programs) != 1:
        raise ValueError('Legacy InitRouting requires one token/route expansion')
    family, = program.tile_programs
    views = unique_by(program.views,'view_id')
    accesses = unique_by(family.accesses,'access_id')
    roles = ('hidden','row_idx','expert_idx','expanded_x','expanded_row_idx','expanded_expert_idx')
    if (family.scope.op_ids != ('cann_moe_init',) or family.scope.agent != 'logical-cann-route-expansion'
            or set(accesses) != set(roles) or len(views) != 9):
        raise ValueError('Legacy InitRouting requires all six public operands and actual sorted output selections')
    tokens,hidden = views['hidden'].shape
    topk = views['row_idx'].shape[1]
    dtype = views['hidden'].dtype
    if dtype not in ('bf16','fp16') or len({views[role].allocation_id for role in roles}) != 6:
        raise ValueError('Legacy InitRouting floating storage/native alias mismatch')
    if tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != (('token',tokens,1),('route',topk,1)):
        raise ValueError('Legacy InitRouting must preserve independent input tokens and actual TOPK routes')
    mapping_ready = tuple(TileDependency(role,EventKind.READ_READY,('token',)) for role in roles[1:3])
    for index,(role,shape,storage,width) in enumerate(_init_layouts(tokens,topk,hidden,dtype)):
        parent = views[role]; access = accesses[role]; view = views[access.view_id]
        if (parent.shape,parent.dtype,parent.element_bytes) != (shape,storage,width):
            raise ValueError('Legacy InitRouting must conserve actual BF16/FP16 and INT32 requests')
        if index < 3:
            expected = (Operation.READ,('token',),('token',None),())
            if access.view_id != role:
                raise ValueError('Legacy InitRouting reads must retain actual caller operands')
        else:
            selection_shape = (tokens,topk,hidden) if role == 'expanded_x' else (tokens,topk)
            if (not isinstance(view,LogicalSelectionView) or view.parent_view_id != role or view.shape != selection_shape):
                raise ValueError('Legacy InitRouting sorted positions must retain actual public output parents')
            dependencies = ((TileDependency('hidden',EventKind.READ_READY,()),) if role == 'expanded_x' else ()) + mapping_ready
            expected = (Operation.WRITE,('token','route'),('token','route',None) if role == 'expanded_x' else ('token','route'),dependencies)
        if (access.operation,access.iteration_axes,access.view_axes,access.dependencies) != expected:
            raise ValueError('Legacy InitRouting outputs require actual hidden/row_idx/expert_idx READ_READY with the declared token/sort selections')
        if (not access.unmapped or access.view_axis_divisors or access.view_axis_padding or access.indirect_target is not None):
            raise ValueError('Legacy InitRouting native permutation/addresses must stay unresolved')
    return {'tokens':tokens,'topk':topk,'hidden':hidden,
        'read_bytes':logical_requested_bytes(program,Operation.READ),
        'write_bytes':logical_requested_bytes(program,Operation.WRITE)}


def _layouts(tokens,experts,topk):
    return (('logits',(tokens,experts),'fp32'),('weights',(tokens,topk),'fp32'),
            ('expert_ids',(tokens,topk),'int32'),('router_output',(tokens,experts),'fp32'))


def _region(view,binding_id):
    return Unmapped(view.shape,view.dtype,4,f'{binding_id}.{view.view_id}.native_gating_payload')


def standalone_cann_moe_gating_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    tokens,experts,topk = (kernel.shape[key] for key in ('tokens','experts','topk'))
    allocations,views,events,ready = [],[],[],[]
    for role,shape,dtype in _layouts(tokens,experts,topk):
        identity = f'{binding_id}.{role}'
        allocation = MemoryAllocation(identity,'logical-local-device','logical-native-tensor',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = TensorView(role,identity,0,shape,(shape[1]*4,4),dtype,4)
        allocations.append(allocation); views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role == 'logits':
            event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
                _region(view,binding_id),'cann-gating-visible',(EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event); ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('logits',),('weights','expert_ids','router_output'),tuple(ready),0,'offline','cann-gating-visible')


def build_cann_moe_gating_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('CANN gating top-k requires its typed public tensor ABI')
    if granularity not in (Granularity.OP, Granularity.KERNEL, Granularity.TILE):
        raise ValueError('CANN gating exports logical row tiles only')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (
            ('logits',),('weights','expert_ids','router_output'),1):
        raise ValueError('CANN gating top-k must retain all three public outputs')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    layouts = _layouts(*(kernel.shape[key] for key in ('tokens','experts','topk')))
    if set(views) != {role for role,_,_ in layouts} or len({view.allocation_id for view in views.values()}) != 4:
        raise ValueError('CANN gating top-k native operand alias is unbound')
    for role,shape,dtype in layouts:
        if (views[role].shape,views[role].dtype,views[role].element_bytes) != (shape,dtype,4):
            raise ValueError('CANN gating top-k shape/storage dtype mismatch')
    evidence = ('CANN MoeGatingTopK public ABI: FP32 logits, FP32 weights, INT32 expert IDs, full FP32 router output',
        'Independent token rows: normalization and top-k use the complete expert dimension of their own row',
        'Top-k values, logit conversion producer, native layout/address/capacity and scratch remain unbound')
    accesses = [TileAccessTemplate('logits', 'logits', Operation.READ, ('row',), ('row', None),
        bindings.input_ready, (), evidence, unmapped=True)]
    for role in bindings.write_views:
        accesses.append(TileAccessTemplate(role, role, Operation.WRITE, ('row',), ('row', None), (),
            (TileDependency('logits', EventKind.READ_READY, ()),), evidence, unmapped=True))
    scope = AccessScope(program_id, ('cann_moe_gating', 'row_topk'), bindings.call_id, (),
        bindings.rank, bindings.stream, 'logical-cann-gating-rows')
    family = TileProgram(program_id, scope, (TileAxis('row', kernel.shape['tokens'], 1),),
        tuple(accesses), bindings.visibility_domain, evidence)
    program = MemoryAccessProgram('midlayer.memory-access.v1', program_id, hardware.hardware_id, bindings.binding_id,
        Coverage(Granularity.TILE, Semantics.LOGICAL, Ordering.PARTIAL, False, (
            'All public output rows, including full router output, are materialized; values and physical mappings remain unknown.',
            'Native scratch, issued order and full logit/cast/routing producer chain remain partial.')),
        bindings.allocations, bindings.views, (), bindings.events, (), (family,))
    cann_moe_gating_source_demands(program)
    return program


def cann_moe_gating_tile_demands(program):
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    if len(program.tile_programs) != 1:
        raise ValueError('CANN gating requires one independent row loop')
    family, = program.tile_programs
    views = unique_by(program.views, 'view_id')
    accesses = unique_by(family.accesses, 'access_id')
    roles = ('logits', 'weights', 'expert_ids', 'router_output')
    if (set(views) != set(roles) or set(accesses) != set(roles)
            or family.scope.agent != 'logical-cann-gating-rows'
            or family.scope.op_ids != ('cann_moe_gating', 'row_topk')):
        raise ValueError('CANN gating top-k requires complete logits and all public outputs')
    tokens, experts = views['logits'].shape
    topk = views['weights'].shape[1]
    if topk > experts or len({view.allocation_id for view in views.values()}) != 4:
        raise ValueError('CANN gating top-k geometry/native alias is invalid')
    if tuple((axis.axis_id, axis.extent, axis.block) for axis in family.axes) != (('row', tokens, 1),):
        raise ValueError('CANN gating must preserve independent complete token rows')
    for role, shape, dtype in _layouts(tokens, experts, topk):
        view, access = views[role], accesses[role]
        if ((view.shape, view.dtype, view.element_bytes) != (shape, dtype, 4)
                or access.view_id != role or access.operation is not (Operation.READ if role == 'logits' else Operation.WRITE)
                or access.iteration_axes != ('row',) or access.view_axes != ('row', None)
                or access.view_axis_divisors or access.indirect_target is not None or not access.unmapped):
            raise ValueError('CANN gating top-k must conserve actual FP32/INT32 operand requests')
        if role != 'logits' and access.dependencies != (TileDependency('logits', EventKind.READ_READY, ()),):
            raise ValueError('CANN gating top-k outputs require actual logits READ_READY for their row')
    return {'tokens': tokens, 'experts': experts, 'topk': topk,
        'read_bytes': logical_requested_bytes(program, Operation.READ),
        'write_bytes': logical_requested_bytes(program, Operation.WRITE)}


def cann_moe_gating_source_demands(program):
    if not program.tile_programs or program.accesses:
        raise ValueError('CANN gating requires its unique row tile source')
    return cann_moe_gating_tile_demands(program)


def cann_moe_expert_cast_source_demands(program, *, expected_kernel=None):
    """Public INT32 expert-token metadata is widened to GroupedMatmulV5 INT64."""
    if not program.tile_programs or program.accesses:
        raise ValueError('Expert-token Cast requires its unique scalar tile source')
    from src.kernel.memory_path_solver import scalar_tile_source_work
    work = scalar_tile_source_work(program)
    if work['scope'].op_ids != ('moe_cast','expert_tokens') or len(work['reads']) != 1 or len(work['writes']) != 1 or len(work['shape']) != 1:
        raise ValueError('Expert-token Cast requires exactly one input/output')
    source, = work['reads']; destination, = work['writes']
    if (source.allocation_id == destination.allocation_id or any(
            (view.dtype,view.element_bytes,view.strides_bytes) != (dtype,width,(width,))
            for view,dtype,width in ((source,'int32',4),(destination,'int64',8)))):
        raise ValueError('Expert-token Cast must conserve actual INT32 input and INT64 output')
    if expected_kernel is not None and (expected_kernel.kind is not KernelKind.VECTOR
            or expected_kernel.variant!='moe_cast' or expected_kernel.attributes['cast_role']!='expert_tokens'
            or expected_kernel.execution_support!='supported' or expected_kernel.hardware_id!=program.hardware_id
            or work['shape']!=(expected_kernel.shape['experts'],)):
        raise ValueError('Expert-token Cast source differs from its actual expert-token kernel')
    return {'elements':work['shape'][0],'read_bytes':work['read_bytes'],'write_bytes':work['write_bytes']}


def has_cann_expert_tokens_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant == 'moe_compute_expert_tokens'
        and kernel.execution_support == 'supported' and hardware.kernel_memory_model == 'npu_tile')


def _expert_worker_geometry(items,experts,cores,key):
    if min(items,experts,cores) <= 0 or key not in (1001,1002,1003):
        raise ValueError('Expert-token source requires positive geometry and a known program key')
    normal = -(-items//cores)
    if key == 1003:
        normal = -(-normal//320)*320
    used = min(cores,-(-items//normal))
    return normal,used,(-(-experts//256) if key == 1003 else 1)


def _expert_layouts(items,experts,cores,key):
    layouts = (('sorted_experts',(items,)),('expert_tokens',(experts,)))
    return layouts+((('workspace',(experts,cores)),) if key != 1001 else ())


def _expert_region(view,binding_id,selection,used):
    shape = (view.shape[0],used) if selection == 'active_workers' else view.shape
    return Unmapped(shape,'int32',4,f'{binding_id}.{view.view_id}.{selection}')


def _expert_requests(items,experts,cores,key):
    _,used,scans = _expert_worker_geometry(items,experts,cores,key)
    plan = [('input','sorted_experts',Operation.READ,'full',scans,())]
    if key == 1001:
        plan.append(('counts_init','expert_tokens',Operation.WRITE,'full',1,()))
        previous = 'counts_init'
        # A legal untimed linearization of commutative atomic updates. This
        # is not an assertion about native worker issue or serialization.
        for worker in range(used):
            read,write = f'counts_read{worker}',f'counts_write{worker}'
            plan.append((read,'expert_tokens',Operation.READ,'full',1,(previous,)))
            plan.append((write,'expert_tokens',Operation.WRITE,'full',1,('input',read)))
            previous = write
    else:
        plan.extend((
            ('workspace_init','workspace',Operation.WRITE,'full',1,()),
            ('workspace_update','workspace',Operation.WRITE,'active_workers',1,('input','workspace_init')),
            ('workspace_read','workspace',Operation.READ,'active_workers',1,('workspace_update',)),
            ('counts_output','expert_tokens',Operation.WRITE,'full',1,('workspace_read',))))
    return tuple(plan)


def standalone_cann_expert_tokens_bindings(kernel,hardware, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    items = kernel.shape['tokens']*kernel.shape['topk']
    experts = kernel.shape['experts']
    cores = int(hardware.resource_by_role('compute_vector').total_units)
    key = int(kernel.launch.parameters['cann_tiling_key'])
    if kernel.launch.grid != (cores,1,1) or kernel.launch.block != (1,1,1):
        raise ValueError('CANN MoeComputeExpertTokens requires the source-defined full-AIV launch')
    _expert_worker_geometry(items,experts,cores,key)
    allocations,views,events,ready = [],[],[],[]
    for role,shape in _expert_layouts(items,experts,cores,key):
        identity = f'{binding_id}.{role}'
        strides,size = [],4
        for extent in reversed(shape):
            strides.append(size); size *= extent
        allocation = MemoryAllocation(identity,'logical-local-device','logical-native-tensor',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = TensorView(role,identity,0,shape,tuple(reversed(strides)),'int32',4)
        allocations.append(allocation); views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role == 'sorted_experts':
            event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
                _expert_region(view,binding_id,'full',0),'expert-tokens-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event); ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('sorted_experts',),('expert_tokens',),tuple(ready),0,'offline','expert-tokens-visible')


def build_cann_expert_tokens_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('ComputeExpertTokens requires typed sorted indices/counts/workspace')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('ComputeExpertTokens exports logical worker/scan/count tiles')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (('sorted_experts',),('expert_tokens',),1):
        raise ValueError('ComputeExpertTokens input/output roles mismatch')
    items,experts = kernel.shape['tokens']*kernel.shape['topk'],kernel.shape['experts']
    cores = int(hardware.resource_by_role('compute_vector').total_units)
    key = int(kernel.launch.parameters['cann_tiling_key'])
    if kernel.launch.grid != (cores,1,1) or kernel.launch.block != (1,1,1):
        raise ValueError('CANN MoeComputeExpertTokens requires the source-defined full-AIV launch')
    _,used,_ = _expert_worker_geometry(items,experts,cores,key)
    layouts = _expert_layouts(items,experts,cores,key)
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    if set(views) != {role for role,_ in layouts} or len({view.allocation_id for view in views.values()}) != len(layouts):
        raise ValueError('ComputeExpertTokens requires distinct input/output/workspace parents')
    for role,shape in layouts:
        if (views[role].shape,views[role].dtype,views[role].element_bytes) != (shape,'int32',4):
            raise ValueError('ComputeExpertTokens requires its complete INT32 tensor ABI')
    header = ('cann_expert_tokens',str(items),str(experts),str(cores),str(key))
    evidence = ('microbench/ascend_910b/references/moe_compute_expert_tokens_public_source.json: versioned Int32SS/M/L InitOutput, scans, atomic counts or workspace reduction',
        'Public-source binary binding remains pending; native scratch layout/addresses and SyncAll lowering are not fabricated',
        'Logical requests exclude 32B GM copy projection padding; small-program RMW linearization is untimed, not native issue order')
    axes,selected,accesses = _expert_tile_requests(items,experts,cores,key,views,bindings.input_ready,
        bindings.binding_id,program_id,evidence)
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-expert-token-tiles')
    family = TileProgram(program_id,scope,axes,accesses,bindings.visibility_domain,evidence)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Public-source logical scans, initialization, counts and workspace dependency account; retained binary binding remains pending.',
             'Workspace axes/active workers are semantic selections, not resolved native strides or addresses.',
             'Small RMW is a valid logical linearization; atomic read/cache service remains within the inherited unverified write projection.',
             'Physical 32B copy projection and existing compute/synchronization services are separate from logical requests.')),
        bindings.allocations,(*bindings.views,*selected),(),bindings.events,(),(family,))
    cann_expert_tokens_source_demands(program)
    return program


def _expert_tile_requests(items,experts,cores,key,views,input_ready,binding_id,program_id,evidence):
    normal,used,scans = _expert_worker_geometry(items,experts,cores,key)
    if len(input_ready) != 1:raise ValueError('Expert-token workers require their actual sorted input availability')
    if key == 1001:
        axes = (TileAxis('expert',experts,1),)
        accesses = [TileAccessTemplate('counts_init','expert_tokens',Operation.WRITE,('expert',),('expert',),(),(),evidence,unmapped=True)]
        selected = []
        previous = 'counts_init'
        for worker in range(used):
            size = normal if worker < used-1 else items-(used-1)*normal
            parent = views['sorted_experts']
            view = LogicalSelectionView(f'{program_id}.worker{worker}.input',parent.allocation_id,(size,),'int32',4,
                f'{binding_id}.sorted_experts.worker{worker}.logical_items{worker*normal}:{worker*normal+size}','sorted_experts')
            selected.append(view)
            read,update = f'counts_read{worker}',f'counts_write{worker}'
            accesses.extend((
                TileAccessTemplate(f'input{worker}',view.view_id,Operation.READ,(),(None,),input_ready,(),evidence,unmapped=True),
                TileAccessTemplate(read,'expert_tokens',Operation.READ,('expert',),('expert',),(),
                    (TileDependency(previous,EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True),
                TileAccessTemplate(update,'expert_tokens',Operation.WRITE,('expert',),('expert',),(),
                    (TileDependency(f'input{worker}',EventKind.READ_READY,()),TileDependency(read,EventKind.READ_READY,())),evidence,unmapped=True)))
            previous = update
        return axes,tuple(selected),tuple(accesses)
    group = 256 if key == 1003 else experts
    axes = (TileAxis('scan',experts,group),TileAxis('worker',items,normal))
    parent = views['workspace']
    selected = LogicalSelectionView(f'{program_id}.active_workspace',parent.allocation_id,(experts,used),'int32',4,
        f'{binding_id}.workspace.expert_active_worker_columns','workspace')
    cell = ('scan','worker')
    accesses = (
        TileAccessTemplate('input','sorted_experts',Operation.READ,cell,('worker',),input_ready,(),evidence,unmapped=True),
        TileAccessTemplate('workspace_init','workspace',Operation.WRITE,('scan',),('scan',None),(),(),evidence,unmapped=True),
        TileAccessTemplate('workspace_update',selected.view_id,Operation.WRITE,cell,cell,(),
            (TileDependency('input',EventKind.READ_READY,()),TileDependency('workspace_init',EventKind.WRITE_VISIBLE,())),
            evidence,unmapped=True,view_axis_divisors=(1,normal)),
        TileAccessTemplate('workspace_read',selected.view_id,Operation.READ,cell,cell,(),
            (TileDependency('workspace_update',EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True,view_axis_divisors=(1,normal)),
        TileAccessTemplate('counts_output','expert_tokens',Operation.WRITE,('scan',),('scan',),(),
            (TileDependency('workspace_read',EventKind.READ_READY,('worker',)),),evidence,unmapped=True))
    return axes,(selected,),accesses


def _expert_copy_demands(items,experts,cores,key,read,write):
    normal,used,scans = _expert_worker_geometry(items,experts,cores,key)
    # Retain source-owned 320-element/32B copy alignment separately from
    # logical selections. No copy padding becomes a fictitious tensor read.
    limit = items if key == 1001 else 320
    def copied(worker_items):
        full,tail = divmod(worker_items,limit)
        return full*limit*4+(-(-tail*4//32)*32 if tail else 0)
    input_transfer = ((used-1)*copied(normal)+copied(items-(used-1)*normal))*scans
    workspace_transfer = 0 if key == 1001 else experts*(-(-used*4//32)*32)
    return {'items':items,'experts':experts,'cores':cores,'used_cores':used,'key':key,'scans':scans,
        'read_bytes':read,'write_bytes':write,'gm_read_projection_bytes':input_transfer+workspace_transfer,
        'gm_write_projection_bytes':write}


def cann_expert_tokens_tile_demands(program):
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    if len(program.tile_programs) != 1:raise ValueError('ComputeExpertTokens requires one actual worker/scan program')
    family, = program.tile_programs
    header = family.scope.op_ids
    if len(header) != 5 or header[0] != 'cann_expert_tokens' or family.scope.agent != 'logical-cann-expert-token-tiles':
        raise ValueError('ComputeExpertTokens source geometry is missing')
    items,experts,cores,key = (int(value) for value in header[1:])
    layouts = _expert_layouts(items,experts,cores,key)
    views = unique_by(program.views,'view_id')
    if len({views[role].allocation_id for role,_ in layouts}) != len(layouts):
        raise ValueError('ComputeExpertTokens logical storage roles must remain distinct')
    for role,shape in layouts:
        if (views[role].shape,views[role].dtype,views[role].element_bytes) != (shape,'int32',4):
            raise ValueError('ComputeExpertTokens logical tensor shape/dtype mismatch')
    inputs = tuple(access for access in family.accesses if access.access_id.startswith('input'))
    if not inputs:raise ValueError('ComputeExpertTokens requires actual sorted input reads')
    axes,selected,accesses = _expert_tile_requests(items,experts,cores,key,views,inputs[0].input_ready,
        program.binding_id,family.tile_program_id,family.evidence_refs)
    if (family.axes != axes or family.accesses != accesses or set(views) != {role for role,_ in layouts}|{view.view_id for view in selected}
            or any(views[view.view_id] != view for view in selected)):
        raise ValueError('ComputeExpertTokens requires actual input/update/read completion and exact worker/scan/count selections')
    return _expert_copy_demands(items,experts,cores,key,logical_requested_bytes(program,Operation.READ),logical_requested_bytes(program,Operation.WRITE))


def cann_expert_tokens_source_demands(program):
    if program.tile_programs:return cann_expert_tokens_tile_demands(program)
    views,scopes = unique_by(program.views,'view_id'),unique_by(program.scopes,'scope_id')
    if not program.accesses:
        raise ValueError('ComputeExpertTokens must preserve its input and generated state')
    header = scopes[program.accesses[0].scope_id].op_ids[:5]
    if len(header) != 5 or header[0] != 'cann_expert_tokens':
        raise ValueError('ComputeExpertTokens source geometry is missing')
    items,experts,cores,key = (int(value) for value in header[1:])
    normal,used,scans = _expert_worker_geometry(items,experts,cores,key)
    layouts = _expert_layouts(items,experts,cores,key)
    if set(views) != {role for role,_ in layouts} or len({view.allocation_id for view in views.values()}) != len(layouts):
        raise ValueError('ComputeExpertTokens logical storage roles must remain distinct')
    for role,shape in layouts:
        if (views[role].shape,views[role].dtype,views[role].element_bytes) != (shape,'int32',4):
            raise ValueError('ComputeExpertTokens logical tensor shape/dtype mismatch')
    plan = _expert_requests(items,experts,cores,key)
    if len(program.accesses) != len(plan):
        raise ValueError('ComputeExpertTokens must retain every scan/init/update/reduction request')
    nodes = {event.event_id:event for event in program.events}
    nodes.update({access.access_id:access for access in program.accesses})
    refs = {}
    for access,(role,view_role,operation,selection,repeat,required) in zip(program.accesses,plan,strict=True):
        view = views[view_role]
        if (scopes[access.scope_id].op_ids != (*header,role) or access.allocation_id != view.allocation_id
                or access.view_id is not None or access.operation is not operation or access.repeat != repeat
                or access.region != _expert_region(view,program.binding_id,selection,used)):
            raise ValueError('ComputeExpertTokens must conserve logical input/counts/workspace requests')
        pending,ancestors = list(access.dependencies),set()
        while pending:
            ref = pending.pop()
            if ref not in ancestors:
                ancestors.add(ref); pending.extend(nodes[ref.node_id].dependencies)
        if not {refs[name] for name in required} <= ancestors:
            raise ValueError('ComputeExpertTokens requires actual input/update/read completion, not initialization alone')
        refs[role] = EventRef(access.access_id,access.completion)

    read = sum(requested_bytes(access.region)*access.repeat for access in program.accesses if access.operation is Operation.READ)
    write = sum(requested_bytes(access.region)*access.repeat for access in program.accesses if access.operation is Operation.WRITE)
    return _expert_copy_demands(items,experts,cores,key,read,write)


def has_cann_moe_range_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant == 'moe_range'
        and kernel.execution_support == 'supported' and hardware.kernel_memory_model == 'npu_tile'
        and kernel.hardware_id == 'ascend_910b' and kernel.launch.algorithm == 'aclnn_moe_range')


def standalone_cann_moe_range_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    elements = kernel.shape['tokens']*kernel.shape['topk']
    identity = f'{binding_id}.row_idx'
    allocation = MemoryAllocation(identity,'logical-local-device','logical-native-tensor',None,
        AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
    view = TensorView('row_idx',identity,0,(elements,),(4,),'int32',4)
    event = ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,())
    return ElementwiseBindings(binding_id,call_id,(allocation,),(view,),(event,),(),('row_idx',),(),
        0,'offline','cann-range-visible')


def _cann_moe_range_region(elements):
    return Unmapped((elements,),'int32',4,'cann_moe_range.row_idx.arange_TK')


def build_cann_moe_range_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_moe_range_source(kernel,hardware):
        raise NotImplementedError('CANN Range requires its public routing arange ABI')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('CANN Range requires its actual typed output binding')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('unknown CANN range logical granularity')
    if (bindings.read_views,bindings.write_views,bindings.input_ready) != ((),('row_idx',),()):
        raise ValueError('CANN Range host scalar arguments are not tensor inputs')
    if len(bindings.views) != 1 or len(bindings.allocations) != 1:
        raise ValueError('CANN Range requires exactly one INT32 output parent')
    tokens,topk,experts = (kernel.shape[key] for key in ('tokens','topk','experts'))
    elements = tokens*topk
    view,allocation = bindings.views[0],bindings.allocations[0]
    if (view.view_id,view.allocation_id,view.shape,view.strides_bytes,view.dtype,view.element_bytes,view.offset_bytes) != (
            'row_idx',allocation.allocation_id,(elements,),(4,),'int32',4,0):
        raise ValueError('CANN Range must retain its actual INT32 TK output storage')
    scope = AccessScope(program_id,('cann_moe_range',str(tokens),str(topk),str(experts)),
        bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-arange')
    evidence = ('Public routing torch.arange(T*TOPK,dtype=int32) creates one independent output coordinate per semantic tile.',
        'aclnnArange start/end/step are host scalars, not payload tensor reads; native parent coordinates remain unresolved.')
    access = TileAccessTemplate('range',view.view_id,Operation.WRITE,('element',),('element',),(),(),evidence,unmapped=True)
    family = TileProgram(program_id,scope,(TileAxis('element',elements,1),),(access,),bindings.visibility_domain,evidence)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Actual INT32[TK] output coordinates and nullary scalar work only; no fabricated expert-domain tensor read or output.',
            'Native materialization/workspace/address/capacity and downstream parent mapping remain unknown.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))


def cann_moe_range_source_demands(program):
    if program.tile_programs:
        program.__post_init__()
        if len(program.tile_programs) != 1 or len(program.views) != 1 or len(program.allocations) != 1:
            raise ValueError('CANN Range requires exactly its actual single output request')
        family, = program.tile_programs; header = family.scope.op_ids
        if len(header) != 4 or header[0] != 'cann_moe_range' or len(family.accesses) != 1:
            raise ValueError('CANN Range requires its exact public TK geometry')
        tokens,topk,experts = map(int,header[1:]); elements=tokens*topk
        view,allocation,access = program.views[0],program.allocations[0],family.accesses[0]
        if (min(tokens,topk,experts) <= 0
                or (view.view_id,view.allocation_id,view.shape,view.strides_bytes,view.dtype,view.element_bytes,view.offset_bytes)
                    != ('row_idx',allocation.allocation_id,(elements,),(4,),'int32',4,0)
                or tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != (('element',elements,1),)
                or (access.view_id,access.operation,access.iteration_axes,access.view_axes,access.dependencies,access.input_ready,
                    access.unmapped,access.indirect_target,access.view_axis_divisors) != ('row_idx',Operation.WRITE,('element',),('element',),(),(),True,None,())):
            raise ValueError('CANN Range must conserve its actual nullary INT32 TK output')
        from src.kernel.memory_path_solver import logical_requested_bytes
        written = logical_requested_bytes(program,Operation.WRITE)
        return {'tokens':tokens,'topk':topk,'experts':experts,'elements':elements,'read_bytes':0,'write_bytes':written,
            'priced_read_bytes':0,'priced_write_bytes':written,'legacy_alu_projection_elements':experts+elements}
    if len(program.accesses) != 1 or len(program.views) != 1 or len(program.allocations) != 1 or len(program.scopes) != 1:
        raise ValueError('CANN Range requires exactly its actual single output request')
    view,allocation,scope,access = program.views[0],program.allocations[0],program.scopes[0],program.accesses[0]
    header = scope.op_ids
    if len(header) != 4 or header[0] != 'cann_moe_range':
        raise ValueError('CANN Range requires its exact public TK geometry')
    tokens,topk,experts = map(int,header[1:])
    elements = tokens*topk
    if (min(tokens,topk,experts) <= 0
            or (view.view_id,view.allocation_id,view.shape,view.strides_bytes,view.dtype,view.element_bytes,view.offset_bytes)
                != ('row_idx',allocation.allocation_id,(elements,),(4,),'int32',4,0)
            or (access.scope_id,access.allocation_id,access.view_id,access.operation,access.region,access.repeat)
                != (scope.scope_id,allocation.allocation_id,None,Operation.WRITE,_cann_moe_range_region(elements),1)
            or EventRef(allocation.allocated_event,EventKind.ALLOCATED) not in access.dependencies):
        raise ValueError('CANN Range must conserve its actual nullary INT32 TK output')
    return {'tokens':tokens,'topk':topk,'experts':experts,'elements':elements,'read_bytes':0,
        'write_bytes':requested_bytes(access.region),
        'priced_read_bytes':0,'priced_write_bytes':requested_bytes(access.region),
        'legacy_alu_projection_elements':experts+elements}


def _native_init_layouts(kernel):
    tokens,topk,hidden,experts=(kernel.shape[key] for key in ('tokens','topk','hidden','experts'))
    count_dtype,count_width=('int32',4) if kernel.variant=='moe_init_routing_v2' else ('int64',8)
    return (('hidden',(tokens,hidden),kernel.dtype,2),('expert_idx',(tokens,topk),'int32',4),
        ('expanded_x',(tokens*topk,hidden),kernel.dtype,2),('expanded_row_idx',(tokens*topk,),'int32',4),
        ('expert_tokens',(experts,),count_dtype,count_width))


def _standalone_native_init_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    allocations,views,events,ready=[],[],[],[]
    for index,(role,shape,dtype,width) in enumerate(_native_init_layouts(kernel)):
        identity=f'{binding_id}.{role}';strides=[];stride=width
        for extent in reversed(shape):strides.insert(0,stride);stride*=extent
        allocation=MemoryAllocation(identity,'logical-local-device','logical-native-tensor',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=TensorView(role,identity,0,shape,tuple(strides),dtype,width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if index<2:
            event=ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
                Unmapped(shape,dtype,width,f'{identity}.native_interface'),'cann-init-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('hidden','expert_idx'),('expanded_x','expanded_row_idx','expert_tokens'),tuple(ready),0,'offline','cann-init-visible')


def _native_init_recipe(hidden_ready,expert_ready,evidence):
    """Logical expert sort states and the actual index-before-gather relation."""
    all_ids=(TileDependency('expert_idx',EventKind.READ_READY,('token',)),)
    one_row=('routed_row',)
    return (
        TileAccessTemplate('expert_idx','expert_idx',Operation.READ,('token',),('token',None),(expert_ready,),(),evidence,unmapped=True),
        TileAccessTemplate('sort_keys_write','sort_keys',Operation.WRITE,one_row,one_row,(),all_ids,evidence,unmapped=True),
        TileAccessTemplate('sorted_rows_write','sorted_rows',Operation.WRITE,one_row,one_row,(),all_ids,evidence,unmapped=True),
        TileAccessTemplate('sort_keys_read','sort_keys',Operation.READ,one_row,one_row,(),
            (TileDependency('sort_keys_write',EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True),
        TileAccessTemplate('sorted_rows_read','sorted_rows',Operation.READ,one_row,one_row,(),
            (TileDependency('sorted_rows_write',EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True),
        TileAccessTemplate('hidden','hidden.selected',Operation.READ,one_row,('routed_row',None),(hidden_ready,),
            (TileDependency('sorted_rows_read',EventKind.READ_READY,()),),evidence,unmapped=True),
        TileAccessTemplate('expanded_x','expanded_x',Operation.WRITE,one_row,('routed_row',None),(),
            (TileDependency('hidden',EventKind.READ_READY,()),),evidence,unmapped=True),
        TileAccessTemplate('expanded_row_idx','expanded_row_idx.selected',Operation.WRITE,one_row,one_row,(),
            (TileDependency('sorted_rows_read',EventKind.READ_READY,()),),evidence,unmapped=True),
        TileAccessTemplate('expert_tokens','expert_tokens',Operation.WRITE,('expert',),('expert',),(),
            (TileDependency('sort_keys_read',EventKind.READ_READY,('routed_row',)),),evidence,unmapped=True))


def _build_native_init_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_moe_init_source(kernel,hardware) or kernel.variant not in ('moe_init_routing_v2','moe_init_routing_v3'):
        raise NotImplementedError('Native InitRouting source requires a fresh unquantized full expert range')
    if not isinstance(bindings,ElementwiseBindings) or granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise TypeError('Native InitRouting requires typed logical interface bindings')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(
            ('hidden','expert_idx'),('expanded_x','expanded_row_idx','expert_tokens'),2):
        raise ValueError('Native InitRouting public inputs/outputs are incomplete')
    views=unique_by(bindings.views,'view_id');layouts=_native_init_layouts(kernel)
    if set(views)!={role for role,*_ in layouts} or len({view.allocation_id for view in views.values()})!=5:
        raise ValueError('Native InitRouting requires five distinct public tensor parents')
    for role,shape,dtype,width in layouts:
        stride=width;strides=[]
        for extent in reversed(shape):strides.insert(0,stride);stride*=extent
        view=views[role]
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes)!=(shape,tuple(strides),dtype,width):
            raise ValueError('Native InitRouting public shape/dtype/strides mismatch')
    tokens,topk,hidden,experts=(kernel.shape[key] for key in ('tokens','topk','hidden','experts'));rows=tokens*topk
    allocations=list(bindings.allocations);events=list(bindings.events)
    for role,dtype in (('sort_keys','fp32'),('sorted_rows','int32')):
        identity=f'{bindings.binding_id}.{role}'
        allocation=MemoryAllocation(identity,bindings.allocations[0].device_id,'logical-internal-state',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        views[role]=TensorView(role,identity,0,(rows,),(4,),dtype,4)
    # Values of the sorted permutation, and scatter/gather output orientation,
    # are unknown. These selections never imply an identity permutation.
    views['hidden.selected']=LogicalSelectionView('hidden.selected',views['hidden'].allocation_id,(rows,hidden),kernel.dtype,2,
        f'{bindings.binding_id}.sorted_rows_select_hidden','hidden')
    views['expanded_row_idx.selected']=LogicalSelectionView('expanded_row_idx.selected',views['expanded_row_idx'].allocation_id,
        (rows,),'int32',4,f'{bindings.binding_id}.native_index_orientation','expanded_row_idx')
    evidence=('CANN8.5.1 V2/V3 unquantized dropless full-range routing: FP32 negated expert sort keys and INT32 sorted row state.',
        'Logical sort states are produced from all expert IDs; payload selection waits for its actual sorted-row read.',
        'FullLoad may keep these states and payload panels in UB; multi-core sort/merge uses GM workspace. Placement, padding and issue multiplicity remain unbound.',
        'Row indices are INT32; counts V2 INT32/V3 INT64. No no-scale expandedScale or dropless before-capacity output is fabricated.')
    accesses=_native_init_recipe(bindings.input_ready[0],bindings.input_ready[1],evidence)
    scope=AccessScope(program_id,('cann_native_init',kernel.variant),bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-native-route')
    family=TileProgram(program_id,scope,(TileAxis('token',tokens,1),TileAxis('routed_row',rows,1),TileAxis('expert',experts,1)),
        accesses,bindings.visibility_domain,evidence)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Logical sort-state and public routing dataflow; permutation values, native placement/capacity and workspace transaction multiplicities remain unknown.',
             'Routed payload reads count logical operand uses: FullLoad can reuse row panels in UB; GatherOut may reload per route. No native GM issue count is inferred.',
             'V2 before-capacity and V3 scale are absent in the full-range dropless/no-scale ABI; old priced envelopes remain separate.')),
        tuple(allocations),tuple(views.values()),(),tuple(events),(),(family,))
    _native_init_demands(program)
    return program


def _native_init_demands(program):
    from src.kernel.memory_path_solver import logical_view_requested_bytes
    if not program.tile_programs or program.accesses or len(program.tile_programs)!=1:
        raise ValueError('Native InitRouting requires its unique public route tile source')
    program.__post_init__();family,=program.tile_programs;views=unique_by(program.views,'view_id')
    if family.scope.op_ids not in (('cann_native_init','moe_init_routing_v2'),('cann_native_init','moe_init_routing_v3')):
        raise ValueError('Native InitRouting source variant is unknown')
    tokens,topk=views['expert_idx'].shape;_,hidden=views['hidden'].shape;experts,=views['expert_tokens'].shape;rows=tokens*topk
    dtype=views['hidden'].dtype;count_dtype,count_width=('int32',4) if family.scope.op_ids[1]=='moe_init_routing_v2' else ('int64',8)
    layouts=(('hidden',(tokens,hidden),dtype,2),('expert_idx',(tokens,topk),'int32',4),
        ('expanded_x',(rows,hidden),dtype,2),('expanded_row_idx',(rows,),'int32',4),
        ('expert_tokens',(experts,),count_dtype,count_width),('sort_keys',(rows,),'fp32',4),('sorted_rows',(rows,),'int32',4))
    if dtype not in ('bf16','fp16') or len({views[role].allocation_id for role,*_ in layouts})!=7:
        raise ValueError('Native InitRouting requires distinct public tensors and logical sort states')
    if tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes)!=(('token',tokens,1),('routed_row',rows,1),('expert',experts,1)):
        raise ValueError('Native InitRouting route/expert partition changed')
    if set(views)!={role for role,*_ in layouts}|{'hidden.selected','expanded_row_idx.selected'} or len(family.accesses)!=9:
        raise ValueError('Native InitRouting source operands are incomplete')
    for role,shape,storage,width in layouts:
        view=views[role];strides=[];stride=width
        for extent in reversed(shape):strides.insert(0,stride);stride*=extent
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes)!=(shape,tuple(strides),storage,width):
            raise ValueError('Native InitRouting must preserve payload and metadata storage')
    accesses=unique_by(family.accesses,'access_id')
    if len(accesses['hidden'].input_ready)!=1 or len(accesses['expert_idx'].input_ready)!=1:
        raise ValueError('Native InitRouting public input readiness is incomplete')
    expected=_native_init_recipe(accesses['hidden'].input_ready[0],accesses['expert_idx'].input_ready[0],family.accesses[0].evidence_refs)
    if family.accesses!=expected:raise ValueError('Native InitRouting source lost sort/payload metadata READ_READY or state publication')
    for role,parent_role,shape in (('hidden.selected','hidden',(rows,hidden)),('expanded_row_idx.selected','expanded_row_idx',(rows,))):
        selected=views[role];parent=views[parent_role]
        if (not isinstance(selected,LogicalSelectionView) or selected.parent_view_id!=parent_role
                or (selected.allocation_id,selected.dtype,selected.element_bytes)!=(parent.allocation_id,parent.dtype,parent.element_bytes)
                or selected.shape!=shape or selected.coordinate_map):
            raise ValueError('Native InitRouting must retain unknown permutation values and actual selection parents')
    amount=lambda role,operation:logical_view_requested_bytes(program,role,operation)
    payload=amount('hidden.selected',Operation.READ);metadata=amount('expert_idx',Operation.READ)
    payload_write=amount('expanded_x',Operation.WRITE);row_write=amount('expanded_row_idx.selected',Operation.WRITE)
    count_write=amount('expert_tokens',Operation.WRITE)
    internal_read=sum(amount(role,Operation.READ) for role in ('sort_keys','sorted_rows'))
    internal_write=sum(amount(role,Operation.WRITE) for role in ('sort_keys','sorted_rows'))
    priced_write=payload_write+(row_write+2*count_write if count_width==4 else 2*row_write+count_write)
    return {'tokens':tokens,'topk':topk,'hidden':hidden,'experts':experts,
        'read_bytes':payload+metadata+internal_read,'write_bytes':payload_write+row_write+count_write+internal_write,
        'logical_internal_read_bytes':internal_read,'logical_internal_write_bytes':internal_write,
        'priced_read_bytes':payload//topk+metadata,'priced_write_bytes':priced_write,
        'logical_sort_state_bound':True,'native_sort_scratch_bound':False,'native_payload_loads_bound':False,
        'active_expert_range':(0,experts),'expanded_scale_present':False}
