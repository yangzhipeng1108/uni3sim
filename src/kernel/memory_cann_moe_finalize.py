"""CANN no-bias finalization and torch_npu unpermutation logical tensor ABIs."""
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,MemoryAccess,
    MemoryAccessProgram,MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,
    TensorView,Unmapped,requested_bytes,unique_by,LogicalSelectionView,TileAxis,TileAccessTemplate,TileDependency,TileProgram,
)


def has_cann_moe_finalize_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant in ('moe_finalize_routing','moe_token_unpermute')
        and kernel.dtype == 'bf16' and kernel.execution_support == 'supported'
        and hardware.kernel_memory_model == 'npu_tile'
        and (kernel.variant == 'moe_token_unpermute' or ('cann_tiling_key' in kernel.launch.parameters
             and kernel.launch.parameters['cann_tiling_key'] == 20036.0)))


def _layouts(variant,tokens,topk,hidden):
    if variant == 'moe_finalize_routing':
        scale_dtype,scale_width,map_role = 'bf16',2,'expanded_src_to_dst_row'
    elif variant == 'moe_token_unpermute':
        scale_dtype,scale_width,map_role = 'fp32',4,'sorted_indices'
    else:
        raise ValueError('CANN finalization requires its explicit source variant')
    return (('down',(tokens*topk,hidden),'bf16',2),('scales',(tokens,topk),scale_dtype,scale_width),
            (map_role,(tokens*topk,),'int32',4),('output',(tokens,hidden),'bf16',2))


def _region(view,binding_id):
    return Unmapped(view.shape,view.dtype,view.element_bytes,f'{binding_id}.{view.view_id}.native_finalize_payload')


def standalone_cann_moe_finalize_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    tokens,topk,hidden = (kernel.shape[key] for key in ('tokens','topk','hidden'))
    allocations,views,events,ready = [],[],[],[]
    for role,shape,dtype,width in _layouts(kernel.variant,tokens,topk,hidden):
        identity = f'{binding_id}.{role}'
        strides,size = [],width
        for axis in reversed(shape):
            strides.append(size); size *= axis
        allocation = MemoryAllocation(identity,'logical-local-device','logical-native-tensor',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width)
        allocations.append(allocation); views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role != 'output':
            event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
                _region(view,binding_id),'cann-finalize-visible',(EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event); ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        tuple(role for role,_,_,_ in _layouts(kernel.variant,tokens,topk,hidden)[:-1]),('output',),tuple(ready),0,'offline','cann-finalize-visible')


def build_cann_moe_finalize_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('CANN finalization requires its typed no-bias tensor ABI')
    if granularity not in (Granularity.OP, Granularity.KERNEL, Granularity.TILE):
        raise ValueError('CANN finalization exports logical token reductions only')
    tokens,topk,hidden = (kernel.shape[key] for key in ('tokens','topk','hidden'))
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    layouts = _layouts(kernel.variant,tokens,topk,hidden)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (
            tuple(role for role,_,_,_ in layouts[:-1]),('output',),3):
        raise ValueError('CANN finalization no-bias roles mismatch')
    if set(views) != {role for role,_,_,_ in layouts} or len({view.allocation_id for view in views.values()}) != 4:
        raise ValueError('CANN finalization native role overlap is unbound')
    for role,shape,dtype,width in layouts:
        if (views[role].shape,views[role].dtype,views[role].element_bytes) != (shape,dtype,width):
            raise ValueError('CANN finalization shape/storage dtype mismatch')
    evidence = (('CANN8.5.1 MoeFinalizeRoutingV2 BF16 no-bias key20036: source token/route map selects permuted down rows'
        if kernel.variant == 'moe_finalize_routing' else
        'torch_npu.npu_moe_token_unpermute: sorted indices associate permuted input rows with original token/topk slots'),
        'Each logical output token reduces its actual top-k routed rows and probability/scales; no fabricated permutation values or native offsets',
        'Conditional bias/expertForSourceRow arguments are not read by this specialization')
    map_role = layouts[2][0]
    selected_map = LogicalSelectionView(f'{program_id}.token_map', views[map_role].allocation_id,
        (tokens,topk), 'int32', 4, f'{bindings.binding_id}.{kernel.variant}.output_token_route_map', map_role)
    selected_down = LogicalSelectionView(f'{program_id}.token_down', views['down'].allocation_id,
        (tokens,topk,hidden), 'bf16', 2, f'{bindings.binding_id}.{kernel.variant}.output_token_selected_down', 'down')
    views[selected_map.view_id] = selected_map
    views[selected_down.view_id] = selected_down
    down_ready, scale_ready, map_ready = bindings.input_ready
    accesses = (
        TileAccessTemplate('row_map',selected_map.view_id,Operation.READ,('token',),('token',None),
            (map_ready,),(),evidence,unmapped=True),
        TileAccessTemplate('down',selected_down.view_id,Operation.READ,('token',),('token',None,None),
            (down_ready,),(TileDependency('row_map',EventKind.READ_READY,()),),evidence,unmapped=True),
        TileAccessTemplate('scales','scales',Operation.READ,('token',),('token',None),
            (scale_ready,),(),evidence,unmapped=True),
        TileAccessTemplate('output','output',Operation.WRITE,('token',),('token',None),(),tuple(
            TileDependency(role,EventKind.READ_READY,()) for role in ('down','scales','row_map')),evidence,unmapped=True))
    scope = AccessScope(program_id,('cann_moe_finalize',kernel.variant,map_role),bindings.call_id,(),
        bindings.rank,bindings.stream,'logical-cann-routed-reduction')
    family = TileProgram(program_id,scope,(TileAxis('token',tokens,1),),accesses,bindings.visibility_domain,evidence)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Each token reduces its actual top-k selected rows; map values, selected physical row coordinates, native formats and capacities remain unknown.',
            'No-bias specialization retains exact BF16/FP32 scales ABI; native scratch, issue order and complete sort/cast producer chain remain partial.')),
        bindings.allocations,tuple(views.values()),(),bindings.events,(),(family,))
    cann_moe_finalize_source_demands(program)
    return program


def cann_moe_finalize_tile_demands(program):
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    if len(program.tile_programs) != 1:
        raise ValueError('CANN finalization requires one independent token reduction loop')
    family, = program.tile_programs
    header = family.scope.op_ids
    if (len(header) != 3 or header[0] != 'cann_moe_finalize'
            or family.scope.agent != 'logical-cann-routed-reduction'):
        raise ValueError('CANN finalization requires its explicit source variant')
    variant, map_role = header[1:]
    views = unique_by(program.views,'view_id')
    accesses = unique_by(family.accesses,'access_id')
    if set(accesses) != {'row_map','down','scales','output'}:
        raise ValueError('CANN finalization requires its complete no-bias source')
    tokens, hidden = views['output'].shape
    topk = views['scales'].shape[1]
    layouts = _layouts(variant,tokens,topk,hidden)
    if map_role != layouts[2][0] or len({views[role].allocation_id for role,_,_,_ in layouts}) != 4:
        raise ValueError('CANN finalization map variant/native alias mismatch')
    for role,shape,dtype,width in layouts:
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes) != (shape,dtype,width):
            raise ValueError('CANN finalization must conserve actual heterogeneous operand requests')
    if tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != (('token',tokens,1),):
        raise ValueError('CANN finalization requires independent complete output token rows')
    for role,parent,shape,mapping in (('row_map',map_role,(tokens,topk),('token',None)),
            ('down','down',(tokens,topk,hidden),('token',None,None))):
        view = views[accesses[role].view_id]
        if (not isinstance(view,LogicalSelectionView) or view.parent_view_id != parent or view.shape != shape):
            raise ValueError('CANN finalization selected map/down must retain actual routed parents')
    for role,mapping in (('row_map',('token',None)),('down',('token',None,None)),
            ('scales',('token',None)),('output',('token',None))):
        access = accesses[role]
        if (access.operation is not (Operation.WRITE if role == 'output' else Operation.READ)
                or access.iteration_axes != ('token',) or access.view_axes != mapping
                or access.view_axis_divisors or access.indirect_target is not None or not access.unmapped
                or (role in ('scales','output') and access.view_id != role)):
            raise ValueError('CANN finalization must conserve actual token/topk operand selections')
    if (accesses['down'].dependencies != (TileDependency('row_map',EventKind.READ_READY,()),)
            or accesses['output'].dependencies != tuple(TileDependency(role,EventKind.READ_READY,()) for role in ('down','scales','row_map'))):
        raise ValueError('CANN finalization output requires actual down/scales/row-map READ_READY for its token')
    return {'tokens':tokens,'topk':topk,'hidden':hidden,
        'read_bytes':logical_requested_bytes(program,Operation.READ),'write_bytes':logical_requested_bytes(program,Operation.WRITE)}


def cann_moe_finalize_source_demands(program):
    if not program.tile_programs or program.accesses:
        raise ValueError('CANN finalization requires its unique token reduction tile source')
    return cann_moe_finalize_tile_demands(program)
