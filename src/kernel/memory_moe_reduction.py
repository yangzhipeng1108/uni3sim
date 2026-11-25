"""MoE top-k two-byte floating output reduction with explicit slice regions."""
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,
    MemoryAccessProgram,MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,
    TensorView,LogicalSelectionView,Unmapped,unique_by,TileAxis,TileAccessTemplate,TileDependency,TileProgram,
)


def has_route_weight_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR
        and kernel.variant in ('moe_routing_weight_norm','moe_routing_weight_scale')
        and kernel.execution_support=='supported' and hardware.kernel_memory_model=='gpu_mshr')


def _route_weight_layouts(variant,tokens,topk):
    if variant=='moe_routing_weight_norm':return (('weights',(tokens,topk)),('output',(tokens,)))
    if variant=='moe_routing_weight_scale':return (('weights',(tokens,topk)),('row_sum',(tokens,)),('output',(tokens,topk)))
    raise ValueError('Unknown public route-weight operation')


def standalone_route_weight_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    allocations,views,events,ready=[],[],[],[]
    layouts=_route_weight_layouts(kernel.variant,kernel.shape['tokens'],kernel.shape['topk'])
    for role,shape in layouts:
        identity=f'{binding_id}.{role}';strides=(shape[1]*4,4) if len(shape)==2 else (4,)
        size=4
        for extent in shape:size*=extent
        allocation=MemoryAllocation(identity,'logical-local-device','logical-storage',size,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=TensorView(role,identity,0,shape,strides,'fp32',4)
        allocations.append(allocation);views.append(view);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role!='output':
            event=ProgramEvent(f'{identity}.initialized',EventKind.INITIALIZED,identity,view.region(),'route-weight-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),tuple(role for role,_ in layouts[:-1]),
        ('output',),tuple(ready),0,'offline','route-weight-visible')


def _route_weight_tiles(variant,views,ready,scope,visibility,evidence):
    tokens,topk=views['weights'].shape
    accesses=[TileAccessTemplate('weights','weights',Operation.READ,('token',),('token',None),(ready['weights'],),(),evidence)]
    dependencies=[TileDependency('weights',EventKind.READ_READY,())]
    if variant=='moe_routing_weight_scale':
        accesses.append(TileAccessTemplate('row_sum','row_sum',Operation.READ,('token',),('token',),(ready['row_sum'],),(),evidence))
        dependencies.append(TileDependency('row_sum',EventKind.READ_READY,()))
    mapping=('token',) if variant=='moe_routing_weight_norm' else ('token',None)
    accesses.append(TileAccessTemplate('output','output',Operation.WRITE,('token',),mapping,(),tuple(dependencies),evidence))
    return TileProgram(scope.scope_id,scope,(TileAxis('token',tokens,1),),tuple(accesses),visibility,evidence)


def build_route_weight_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_route_weight_source(kernel,hardware):raise ValueError('Public route-weight source is not admitted')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('Public route weights require typed FP32 row bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Public route weights use logical row tiles')
    layouts=_route_weight_layouts(kernel.variant,kernel.shape['tokens'],kernel.shape['topk'])
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(tuple(role for role,_ in layouts[:-1]),('output',),len(layouts)-1):
        raise ValueError('Public route-weight operand roles differ')
    views=unique_by(bindings.views,'view_id')
    if set(views)!={role for role,_ in layouts}:raise ValueError('Public route-weight views differ')
    if any(views[role].shape!=shape for role,shape in layouts):raise ValueError('Public route-weight views differ from actual kernel shape')
    evidence=('vLLM fused_moe.py:fused_topk and router/grouped_topk_router.py: topk_weights / topk_weights.sum(dim=-1,keepdim=True).',
        'Router weights and row sums are FP32 independently of the nominal model activation dtype.',
        'Each token is an independent complete TOPK reduction or broadcast-divide block; no native CUDA/HIP issue order or physical transaction claim.')
    scope=AccessScope(program_id,('moe_route_weights',kernel.variant),bindings.call_id,(),bindings.rank,bindings.stream,'logical-route-weight-rows')
    ready=dict(zip(bindings.read_views,bindings.input_ready,strict=True))
    family=_route_weight_tiles(kernel.variant,views,ready,scope,bindings.visibility_domain,evidence)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,True,(
            'Public out-of-place FP32 sum/divide interface only; native reduction scratch and physical issue remain unbound.',
            'The existing aggregate-read-only auxiliary price is an explicit separate projection, not the logical read/write stream.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    route_weight_source_demands(program,expected_kernel=kernel);return program


def route_weight_source_demands(program, *, expected_kernel=None):
    program.__post_init__()
    if program.scopes or program.accesses or len(program.tile_programs)!=1:raise ValueError('Route weights require their unique compressed row source')
    family,=program.tile_programs;header=family.scope.op_ids
    if len(header)!=2 or header[0]!='moe_route_weights' or family.scope.agent!='logical-route-weight-rows':raise ValueError('Route-weight row source identity differs')
    variant=header[1];views=unique_by(program.views,'view_id');tokens,topk=views['weights'].shape
    if expected_kernel is not None and (expected_kernel.kind is not KernelKind.VECTOR or program.hardware_id!=expected_kernel.hardware_id
            or expected_kernel.variant!=variant or (tokens,topk)!=(expected_kernel.shape['tokens'],expected_kernel.shape['topk'])):
        raise ValueError('Route-weight source differs from its actual kernel')
    layouts=_route_weight_layouts(variant,tokens,topk)
    if set(views)!={role for role,_ in layouts} or len({view.allocation_id for view in views.values()})!=len(views):
        raise ValueError('Public route weights require distinct out-of-place parents')
    for role,shape in layouts:
        view=views[role];strides=(shape[1]*4,4) if len(shape)==2 else (4,)
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes)!=(shape,strides,'fp32',4):raise ValueError('Route-weight FP32 tensor ABI differs')
    templates=unique_by(family.accesses,'access_id');ready={}
    for role,_ in layouts[:-1]:
        access=templates[role]
        if len(access.input_ready)!=1:raise ValueError('Route-weight input readiness differs')
        ready[role]=access.input_ready[0]
    expected=_route_weight_tiles(variant,views,ready,family.scope,family.visibility_domain,family.evidence_refs)
    if family!=expected:raise ValueError('Route-weight output requires its own complete input READ_READY')
    from src.kernel.memory_path_solver import logical_requested_bytes
    read=logical_requested_bytes(program,Operation.READ);write=logical_requested_bytes(program,Operation.WRITE)
    return {'tokens':tokens,'topk':topk,'routing_elements':tokens*topk,'read_bytes':read,'write_bytes':write,
        'legacy_aggregate_read_price_bytes':read+write}


def has_moe_reduction_source(kernel,hardware):
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant not in ('moe_sum','moe_reduce')
            or kernel.execution_support != 'supported' or kernel.dtype not in ('bf16','fp16')
            or hardware.kernel_memory_model != 'gpu_mshr'):
        return False
    selected = kernel.shape['tokens']*kernel.shape['hidden']*kernel.attributes['work_fraction']
    return selected > 0 and selected.is_integer()


def _geometry(kernel):
    tokens,topk,hidden = (kernel.shape[name] for name in ('tokens','topk','hidden'))
    selected = int(tokens*hidden*kernel.attributes['work_fraction'])
    if selected <= 0 or selected != tokens*hidden*kernel.attributes['work_fraction']:
        raise ValueError('MoE reduction requires a positive integral output slice')
    return tokens,topk,hidden,selected


def standalone_moe_reduction_bindings(kernel, *, call_id,binding_id, device_id="logical-local-device", rank=0, stream="offline"):
    from src.kernel.memory_access_builder import ElementwiseBindings
    tokens,topk,hidden,selected = _geometry(kernel)
    allocations,views,events,ready = [],[],[],[]
    for role,shape,strides in (('routed',(tokens,topk,hidden),(topk*hidden*2,hidden*2,2)),
                               ('output',(tokens,hidden),(hidden*2,2))):
        identity = f'{binding_id}.{role}'
        size = tokens*hidden*2*(topk if role == 'routed' else 1)
        allocation = MemoryAllocation(identity,device_id,'logical-storage',size,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = TensorView(role,identity,0,shape,strides,kernel.dtype,2)
        allocations.append(allocation); views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role == 'routed':
            # The invocation supplies the complete routed tensor. Its unknown
            # partial output selection is a child, not a new initialized root.
            region = view.region()
            kind = EventKind.INITIALIZED
            event = ProgramEvent(f'{identity}.available',kind,identity,region,'moe-reduction-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event); ready.append(EventRef(event.event_id,kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('routed',),('output',),tuple(ready),rank,stream,'moe-reduction-visible')


def build_moe_reduction_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('MoE reduction requires typed routed/output floating tensors')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('unknown MoE reduction logical granularity')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (('routed',),('output',),1):
        raise ValueError('MoE reduction source roles mismatch')
    tokens,topk,hidden,selected = _geometry(kernel)
    dtype = kernel.dtype
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    if set(views) != {'routed','output'} or views['routed'].allocation_id == views['output'].allocation_id:
        raise ValueError('MoE reduction requires distinct complete input/output roles')
    expected = {'routed':((tokens,topk,hidden),(topk*hidden*2,hidden*2,2)),
        'output':((tokens,hidden),(hidden*2,2))}
    for role,view in views.items():
        shape,strides = expected[role]
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes) != (shape,strides,dtype,2):
            raise ValueError('MoE reduction routed/output layout mismatch')
    evidence = ('MetaX fused_moe_runner intermediate_cache3[T,TOPK,H].sum(dim=1) preserves floating dtype',
        'PyTorch Reduce.cuh/TensorIterator work_fraction selects output elements; each output consumes all TOPK inputs',
        'Unknown output-slice offsets are not flattened into fabricated physical coordinates')
    full = selected == tokens*hidden
    if full:
        selected_views = (); read_view,write_view = views['routed'],views['output']
        axes = (TileAxis('token',tokens,1),TileAxis('hidden',hidden,1))
        iterations = ('token','hidden'); read_mapping = ('token',None,'hidden'); write_mapping = iterations
        write_divisors = ()
    else:
        # Logical ordinals of selected outputs, not fabricated parent offsets.
        # Every ordinal consumes TOPK input values; mapping into T/H is unknown.
        read_view = LogicalSelectionView('routed.selection',views['routed'].allocation_id,(selected*topk,),dtype,2,
            f'{bindings.binding_id}.routed.tensor_iterator_output_slice','routed')
        write_view = LogicalSelectionView('output.selection',views['output'].allocation_id,(selected,),dtype,2,
            f'{bindings.binding_id}.output.tensor_iterator_output_slice','output')
        selected_views = (read_view,write_view)
        axes = (TileAxis('selected_input',selected*topk,topk),)
        iterations = read_mapping = write_mapping = ('selected_input',); write_divisors = (topk,)
    evidence += ('Each semantic output-coordinate tile reduces all actual TOPK contributions; scalar ordinals of partial slices retain unknown parent coordinates.',)
    read = TileAccessTemplate('routed',read_view.view_id,Operation.READ,iterations,read_mapping,
        bindings.input_ready,(),evidence,unmapped=not full)
    write = TileAccessTemplate('output',write_view.view_id,Operation.WRITE,iterations,write_mapping,(),
        (TileDependency('routed',EventKind.READ_READY,()),),evidence,unmapped=not full,view_axis_divisors=write_divisors)
    scope = AccessScope(program_id,('moe_reduction',kernel.variant),bindings.call_id,(),bindings.rank,bindings.stream,'logical-moe-reduction')
    family = TileProgram(program_id,scope,axes,(read,write),bindings.visibility_domain,evidence)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Two-byte floating routed tensor reduces TOPK for each independent selected output coordinate.',
             'Partial ordinal groups retain selected counts and unknown parent offsets; separate invocations are not assumed disjoint.',
             'On-chip reduction state is not fabricated GM scratch; native partial reloads/issue order remain separate.')),
        bindings.allocations,(*bindings.views,*selected_views),(),bindings.events,(),(family,))
    moe_reduction_source_demands(program,expected_kernel=kernel)
    return program


def moe_reduction_source_demands(program, *, expected_kernel=None):
    if not program.tile_programs or program.accesses or program.scopes:
        raise ValueError('MoE reduction requires its unique tile source')
    from math import prod
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    if len(program.tile_programs) != 1:
        raise ValueError('MoE reduction requires one independent output-coordinate loop')
    family, = program.tile_programs
    if family.causal_prefixes or family.partitions:
        raise ValueError('MoE reduction requires its exact independent output-coordinate mapping')
    views = unique_by(program.views,'view_id')
    if family.scope.op_ids not in (('moe_reduction','moe_sum'),('moe_reduction','moe_reduce')) or len(family.accesses) != 2:
        raise ValueError('MoE reduction source variant/roles mismatch')
    read,write = family.accesses; tokens,topk,hidden = views['routed'].shape
    dtype = views['routed'].dtype
    if dtype not in ('bf16','fp16') or views['routed'].allocation_id == views['output'].allocation_id:
        raise ValueError('MoE reduction requires distinct two-byte floating parents')
    selected = prod(views[write.view_id].shape); full = selected == tokens*hidden
    expected_roles={'routed','output'} if full else {'routed','output','routed.selection','output.selection'}
    if set(views)!=expected_roles or family.scope.agent!='logical-moe-reduction':
        raise ValueError('MoE reduction requires its exact public parents and selected roles')
    if expected_kernel is not None and (expected_kernel.kind is not KernelKind.VECTOR
            or expected_kernel.variant not in ('moe_sum','moe_reduce') or expected_kernel.execution_support!='supported'
            or expected_kernel.hardware_id!=program.hardware_id or expected_kernel.dtype!=dtype
            or family.scope.op_ids!=('moe_reduction',expected_kernel.variant)
            or (tokens,topk,hidden,selected)!=_geometry(expected_kernel)):
        raise ValueError('MoE reduction source differs from its actual kernel')
    if not 0 < selected <= tokens*hidden:
        raise ValueError('MoE reduction selected output exceeds parent')
    expected_axes = (('token',tokens,1),('hidden',hidden,1)) if full else (('selected_input',selected*topk,topk),)
    if tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != expected_axes:
        raise ValueError('MoE reduction must group actual TOPK inputs per selected output')
    iterations = ('token','hidden') if full else ('selected_input',)
    for role,access,operation in (('routed',read,Operation.READ),('output',write,Operation.WRITE)):
        parent = views[role]; active = views[access.view_id]
        shape = (tokens,topk,hidden) if role == 'routed' else (tokens,hidden)
        strides = (topk*hidden*2,hidden*2,2) if role == 'routed' else (hidden*2,2)
        mapping = (('token',None,'hidden') if role == 'routed' else iterations) if full else iterations
        divisors = (topk,) if not full and role == 'output' else ()
        selected_shape = ((selected*topk if role == 'routed' else selected),)
        if (not full and (not isinstance(active,LogicalSelectionView)
                or access.view_id!=f'{role}.selection' or active.parent_view_id!=role
                or active.coordinate_map
                or active.selection_id!=f'{program.binding_id}.{role}.tensor_iterator_output_slice')):
            raise ValueError('MoE reduction selected ordinals require their actual unaddressed parent')
        if ((parent.shape,parent.strides_bytes,parent.dtype,parent.element_bytes) != (shape,strides,dtype,2)
                or (active.allocation_id,active.dtype,active.element_bytes) != (parent.allocation_id,dtype,2)
                or active.shape != (shape if full else selected_shape)
                or access.operation is not operation or access.iteration_axes != iterations or access.view_axes != mapping
                or access.unmapped != (not full) or access.view_axis_divisors != divisors or access.indirect_target is not None
                or access.constant_value is not None or access.view_axis_padding or access.scope is not None
                or access.value_index is not None or access.atomic_add is not None or access.predicate is not None):
            raise ValueError('MoE reduction source must conserve actual floating TOPK requests')
    if read.dependencies or len(read.input_ready)!=1 or write.input_ready:
        raise ValueError('MoE reduction requires actual routed readiness and its independent output publication')
    if write.dependencies != (TileDependency(read.access_id,EventKind.READ_READY,()),):
        raise ValueError('MoE reduction output requires actual routed input READ_READY')
    return {'tokens':tokens,'topk':topk,'hidden':hidden,'output_elements':selected,'full_invocation':full,
        'read_bytes':logical_requested_bytes(program,Operation.READ),'write_bytes':logical_requested_bytes(program,Operation.WRITE)}
