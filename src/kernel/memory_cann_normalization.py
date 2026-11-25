"""CANN normalization's out-of-place outputs and logical GM interfaces."""

from src.kernel.prediction_cache import runtime_lru_cache
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity, MemoryAccess,
    MemoryAccessProgram, MemoryAllocation, Operation, Ordering, ProgramEvent, Semantics,
    TensorView, unique_by,TileAxis,TileAccessTemplate,TileDependency,TileProgram,TileIndexBinding,TileIndexTerm,
)

_WORK = ContextVar('cann_normalization_source_work')


def has_cann_normalization_source(kernel, hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant in ('rmsnorm','add_rmsnorm')
            and kernel.execution_support == 'supported' and hardware.kernel_memory_model == 'npu_tile'
            and 'rstd_output_bytes' in kernel.launch.parameters)


@contextmanager
def cann_normalization_source_work(kernel, work):
    token = _WORK.set((kernel,work))
    try:
        yield
    finally:
        _WORK.reset(token)


def bound_cann_normalization_work(kernel):
    context = copy_context()
    if _WORK not in context:
        return None
    current,work = context[_WORK]
    if current != kernel:
        raise ValueError('CANN normalization source crossed its kernel boundary')
    return work


def _layouts(kernel):
    rows,hidden = kernel.shape['rows'],kernel.shape['hidden']
    fused = kernel.variant == 'add_rmsnorm'
    inputs = ('input','residual','weight','tiling') if fused else ('input','weight','tiling')
    outputs = ('output','residual_output','rstd') if fused else ('output','rstd')
    payload = int(kernel.launch.parameters['cann_tiling_payload_bytes'])
    if payload != (80 if fused else 28) or payload != kernel.launch.parameters['cann_tiling_payload_bytes']:
        raise ValueError('CANN normalization requires its source entry typed tiling payload')
    return tuple((role,(payload,) if role == 'tiling' else (hidden,) if role == 'weight' else (rows,) if role == 'rstd' else (rows,hidden),
                  'uint8' if role == 'tiling' else 'fp32' if role == 'rstd' else kernel.dtype,
                  1 if role == 'tiling' else 4 if role == 'rstd' else kernel.shape['dtype_bytes'])
                 for role in (*inputs,*outputs)),inputs,outputs


def standalone_cann_normalization_bindings(kernel, *, call_id, binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    layouts,inputs,outputs = _layouts(kernel)
    allocations,views,events,ready = [],[],[],[]
    for role,shape,dtype,width in layouts:
        identity = f'{binding_id}.{role}'
        strides,size = [],width
        for axis in reversed(shape):
            strides.append(size)
            size *= axis
        view = TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width)
        allocation = MemoryAllocation(identity,'logical-local-device',
            'logical-native-launch-arguments' if role == 'tiling' else 'logical-storage',size,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation)
        views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role in inputs:
            event = ProgramEvent(f'{identity}.initialized',EventKind.INITIALIZED,identity,view.region(),'cann-norm-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event)
            ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        inputs,outputs,tuple(ready),0,'offline','cann-norm-visible')


def build_cann_normalization_source(kernel, hardware, bindings, granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('CANN normalization requires typed input/gamma/y/rstd/residual bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('unknown CANN normalization logical granularity')
    layouts,inputs,outputs = _layouts(kernel)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (inputs,outputs,len(inputs)):
        raise ValueError('CANN normalization source operand roles mismatch')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    if set(views) != {role for role,*_ in layouts} or len({view.allocation_id for view in views.values()}) != len(views):
        raise ValueError('CANN normalization outputs are distinct native tensors, not GPU in-place updates')
    for role,shape,dtype,width in layouts:
        view = views[role]
        strides,size = [],width
        for axis in reversed(shape):
            strides.append(size)
            size *= axis
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes) != (shape,tuple(reversed(strides)),dtype,width):
            raise ValueError('CANN normalization source matrix/rstd storage mismatch')
    rows = kernel.shape['rows']
    if kernel.launch.parameters['rstd_output_bytes'] != rows*4:
        raise ValueError('CANN normalization launch rstd disagrees with its FP32 row output')
    copies = int(kernel.launch.parameters['norm_weight_l2_copies'])
    if copies <= 0 or copies != kernel.launch.parameters['norm_weight_l2_copies']:
        raise ValueError('CANN gamma copy count must be an explicit positive discrete fact')
    replay = 0
    if 'input_reread_count' in kernel.launch.parameters:
        replay = int(kernel.launch.parameters['input_reread_count'])
        if replay != 1 or replay != kernel.launch.parameters['input_reread_count']:
            raise ValueError('CANN normalization source supports its declared zero/one Split-D reread')
    return build_cann_normalization_tiles(kernel,bindings,views,inputs,outputs,copies,replay,program_id=program_id)


def cann_normalization_source_demands(program):
    from src.kernel.memory_path_solver import logical_requested_bytes
    from src.kernel.memory_access import requested_bytes
    if program.tile_programs:
        return cann_normalization_tile_demands(program)
    logical_requested_bytes(program,Operation.READ)
    scopes,views = unique_by(program.scopes,'scope_id'),unique_by(program.views,'view_id')
    operands,variant = {},None
    for access in program.accesses:
        roles = scopes[access.scope_id].op_ids
        if len(roles) != 4 or roles[0] != 'cann_normalization' or roles[2] in operands:
            raise ValueError('CANN normalization source roles must be explicit and unique')
        if variant is not None and variant != roles[1]:
            raise ValueError('CANN normalization source mixes variants')
        variant = roles[1]
        if access.view_id != roles[3] or access.region != views[roles[3]].region():
            raise ValueError('CANN normalization source must conserve actual tensor regions')
        operands[roles[2]] = access
    fused = variant == 'add_rmsnorm'
    expected = {'input','weight','output','rstd','tiling'} | ({'residual','residual_output'} if fused else set())
    replay = 'residual_reread' in operands if fused else 'input_reread' in operands
    if replay:
        expected |= {'residual_reread' if fused else 'input_reread'}
    if variant not in ('rmsnorm','add_rmsnorm') or set(operands) != expected:
        raise ValueError('CANN normalization source operand/replay roles incomplete')
    rows,hidden = views[operands['input'].view_id].shape
    width = views[operands['input'].view_id].element_bytes
    if len({view.allocation_id for view in views.values()}) != len(views):
        raise ValueError('CANN normalization cannot replace distinct native outputs with GPU aliases')
    for role,access in operands.items():
        if access.operation is not (Operation.WRITE if role in ('output','residual_output','rstd') else Operation.READ):
            raise ValueError('CANN normalization source operation mismatch')
        view = views[access.view_id]
        expected_view = ('residual_output' if fused else 'input') if role.endswith('_reread') else role
        if access.view_id != expected_view:
            raise ValueError('CANN normalization source role must use its actual tensor view')
        shape = ((80 if fused else 28),) if role == 'tiling' else (hidden,) if role == 'weight' else (rows,) if role == 'rstd' else (rows,hidden)
        dtype,element_width = ('uint8',1) if role == 'tiling' else ('fp32',4) if role == 'rstd' else (views['input'].dtype,width)
        strides,size = [],element_width
        for axis in reversed(shape):
            strides.append(size)
            size *= axis
        if ((view.shape,view.strides_bytes,view.dtype,view.element_bytes) != (shape,tuple(reversed(strides)),dtype,element_width)
                or (role not in ('weight','tiling') and access.repeat != 1)):
            raise ValueError('CANN normalization source row/dtype/repeat mismatch')
    nodes = {event.event_id:event for event in program.events}
    nodes.update({access.access_id:access for access in program.accesses})

    def requires(role,prior):
        pending,ancestors = list(operands[role].dependencies),set()
        while pending:
            ref = pending.pop()
            if ref not in ancestors:
                ancestors.add(ref)
                pending.extend(nodes[ref.node_id].dependencies)
        if any(EventRef(operands[item].access_id,EventKind.READ_READY) not in ancestors for item in prior):
            raise ValueError('CANN normalization lacks actual input/gamma/reread causality')

    initial = ('input','residual') if fused else ('input',)
    for role in (*initial,'weight'):
        requires(role,('tiling',))
    requires('rstd',initial)
    normalization_inputs = (*initial,'weight',*tuple(role for role in ('input_reread','residual_reread') if role in operands))
    requires('output',normalization_inputs)
    if fused:
        requires('residual_output',initial)
    for role in ('input_reread','residual_reread'):
        if role in operands:
            requires(role,initial)
            view_role = 'residual_output' if fused else 'input'
            if operands[role].view_id != view_role:
                raise ValueError('CANN normalization must reread its actual source tensor')
            if fused:
                pending,ancestors = list(operands[role].dependencies),set()
                while pending:
                    ref = pending.pop()
                    if ref not in ancestors:
                        ancestors.add(ref)
                        pending.extend(nodes[ref.node_id].dependencies)
                visible = EventRef(operands['residual_output'].access_id,EventKind.WRITE_VISIBLE)
                if visible not in ancestors:
                    raise ValueError('CANN Split-D reread requires the actual residual output WRITE_VISIBLE')
    volumes = {role:requested_bytes(access.region) for role,access in operands.items()}
    return {'geometry':(rows,hidden,width),'input_bytes':volumes['input'],'weight_bytes':volumes['weight'],
        'weight_copies':operands['weight'].repeat,'initial_read_bytes':sum(volumes[role] for role in initial),
        'output_bytes':volumes['output'],'residual_write_bytes':volumes['residual_output'] if fused else 0,
        'rstd_bytes':volumes['rstd'],'actual_reread_bytes':sum(volumes[role] for role in ('input_reread','residual_reread') if role in volumes),
        'reread_bytes':volumes['residual_reread' if fused else 'input_reread'] if replay else 0,
        'tiling_payload_bytes':volumes['tiling'],'tiling_materializations':operands['tiling'].repeat,
        'tiling_read_bytes':volumes['tiling']*operands['tiling'].repeat,
        'interface_unique_bytes':sum(requested_bytes(view.region()) for view in views.values()),
        'requested_read_bytes':logical_requested_bytes(program,Operation.READ),'requested_write_bytes':logical_requested_bytes(program,Operation.WRITE)}


def _worker_fragments(rows,block_factor,workers):
    if min(rows,block_factor,workers) <= 0 or -(-rows//block_factor) != workers:
        raise ValueError('CANN NORM block factor disagrees with rows and blockDim')
    tail=rows-(workers-1)*block_factor
    if tail == block_factor:return ((0,workers,block_factor),)
    return (((0,workers-1,block_factor),) if workers > 1 else ())+((workers-1,1,tail),)


def build_cann_normalization_tiles(kernel,bindings,views,inputs,outputs,copies,replay, *, program_id):
    rows,hidden=views['input'].shape;width=views['input'].element_bytes
    block_factor=int(kernel.launch.parameters['norm_block_factor']);workers=kernel.launch.grid_size
    if block_factor != kernel.launch.parameters['norm_block_factor']:
        raise ValueError('CANN blockFactor must be an actual integer row partition')
    fragments=_worker_fragments(rows,block_factor,workers)
    actual_copies=sum(count*(-(-local_rows//64) if replay else 1) for _,count,local_rows in fragments)
    if actual_copies != copies:
        raise ValueError('CANN gamma copies must match actual AIV/64-row-group ownership')
    ready=dict(zip(inputs,bindings.input_ready,strict=True));families=[];selected=[]
    evidence=('CANN8.5.1 norm_block_factor assigns contiguous rows per AIV; only the final AIV may have a shorter row extent.',
        'One typed GET_TILING_DATA read per AIV; gamma is retained per AIV or fetched per at-most64-row Split-D group.',
        'Independent full-hidden row reductions preserve distinct y/rstd/residual_sum outputs and actual residual WRITE_VISIBLE before reload.',
        'Row tiles are semantic reductions; UB queues, aligned rstd transfer payload and native column issue remain existing separate projections.')
    for first,count,local_rows in fragments:
        identity=f'{program_id}.aiv{first}';data={}
        for role in (*tuple(name for name in inputs if name in ('input','residual')), *outputs):
            view=views[role];shape=(count,local_rows) if role == 'rstd' else (count,local_rows,hidden)
            strides=(local_rows*view.element_bytes,view.element_bytes) if role == 'rstd' else (
                local_rows*hidden*view.element_bytes,hidden*view.element_bytes,view.element_bytes)
            offset=view.offset_bytes+first*block_factor*(1 if role == 'rstd' else hidden)*view.element_bytes
            alias=TensorView(f'{role}.aiv{first}',view.allocation_id,offset,shape,strides,view.dtype,view.element_bytes)
            selected.append(alias);data[role]=alias
        axes=[TileAxis('aiv',count,1),TileAxis('row',local_rows,1)]
        if replay:axes.append(TileAxis('gamma_group',-(-local_rows//64),1))
        accesses=[TileAccessTemplate('tiling','tiling',Operation.READ,('aiv',),(None,),(ready['tiling'],),(),evidence)]
        tiling=(TileDependency('tiling',EventKind.READ_READY,()),)
        gamma_axes=('aiv','gamma_group') if replay else ('aiv',)
        accesses.append(TileAccessTemplate('weight','weight',Operation.READ,gamma_axes,(None,),(ready['weight'],),tiling,evidence))
        initial=[]
        for role in ('input','residual') if kernel.variant == 'add_rmsnorm' else ('input',):
            accesses.append(TileAccessTemplate(role,data[role].view_id,Operation.READ,('aiv','row'),('aiv','row',None),
                (ready[role],),tiling,evidence));initial.append(TileDependency(role,EventKind.READ_READY,()))
        initial=tuple(initial)
        if kernel.variant == 'add_rmsnorm':
            accesses.append(TileAccessTemplate('residual_output',data['residual_output'].view_id,Operation.WRITE,
                ('aiv','row'),('aiv','row',None),(),initial,evidence))
        dependencies=initial
        if replay:
            fused=kernel.variant == 'add_rmsnorm';role='residual_reread' if fused else 'input_reread';parent='residual_output' if fused else 'input'
            prior=(TileDependency('residual_output',EventKind.WRITE_VISIBLE,()),) if fused else initial
            accesses.append(TileAccessTemplate(role,data[parent].view_id,Operation.READ,('aiv','row'),('aiv','row',None),
                () if fused else (ready['input'],),prior,evidence))
            dependencies+= (TileDependency(role,EventKind.READ_READY,()),)
        gamma_binding=(TileIndexBinding('gamma_group',(TileIndexTerm('row',1),),divisor=64),) if replay else ()
        dependencies+= (TileDependency('weight',EventKind.READ_READY,(),gamma_binding),)
        accesses.append(TileAccessTemplate('output',data['output'].view_id,Operation.WRITE,('aiv','row'),('aiv','row',None),(),dependencies,evidence))
        accesses.append(TileAccessTemplate('rstd',data['rstd'].view_id,Operation.WRITE,('aiv','row'),('aiv','row'),(),initial,evidence))
        scope=AccessScope(identity,('cann_normalization_tiles',kernel.variant,str(first),str(block_factor),str(workers),str(replay)),
            bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-row-reduction')
        families.append(TileProgram(identity,scope,tuple(axes),tuple(accesses),bindings.visibility_domain,evidence))
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Compressed row reductions and AIV/gamma ownership; at most two regular/tail fragments, independent of production row count.',
            'Distinct y/rstd/residual_sum, exact logical ND extents and all argument reads remain explicit.',
            'On-chip UB state is not fabricated GM scratch; native column passes and aligned DMA payload remain separate.')),
        bindings.allocations,(*bindings.views,*selected),(),bindings.events,(),tuple(families))
    cann_normalization_tile_demands(program)
    return program


def cann_normalization_tile_demands(program):
    # Reuse only a complete immutable source, including its actual bindings.
    # Own the returned mapping so consumers cannot poison cached projections.
    return dict(_cann_normalization_tile_demands(program))


@runtime_lru_cache(maxsize=8)
def _cann_normalization_tile_demands(program):
    from src.kernel.memory_access import requested_bytes,tile_template_requested_bytes
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__();views=unique_by(program.views,'view_id')
    if not program.tile_programs:raise ValueError('CANN normalization row tile source missing')
    rows,hidden=views['input'].shape;width=views['input'].element_bytes
    first_header=program.tile_programs[0].scope.op_ids
    if len(first_header) != 6 or first_header[0] != 'cann_normalization_tiles' or first_header[1] not in ('rmsnorm','add_rmsnorm'):
        raise ValueError('CANN normalization row ownership header missing')
    variant=first_header[1];fused=variant == 'add_rmsnorm';block_factor,workers=map(int,first_header[3:5])
    replay_flag=int(first_header[5])
    if replay_flag not in (0,1):raise ValueError('CANN normalization requires an explicit zero/one reread fact')
    fragments=_worker_fragments(rows,block_factor,workers)
    if len(program.tile_programs) != len(fragments):raise ValueError('CANN normalization AIV fragments incomplete')
    parent_roles=('input','weight','tiling','output','rstd',*(('residual','residual_output') if fused else ()))
    if len({views[role].allocation_id for role in parent_roles}) != len(parent_roles):
        raise ValueError('CANN normalization cannot replace distinct native outputs with GPU aliases')
    volumes={};replay_mode=None
    for family,(first,count,local_rows) in zip(program.tile_programs,fragments,strict=True):
        if family.scope.op_ids != ('cann_normalization_tiles',variant,str(first),str(block_factor),str(workers),str(replay_flag)):
            raise ValueError('CANN normalization must retain actual AIV ownership')
        accesses=unique_by(family.accesses,'access_id')
        replay='residual_reread' in accesses if fused else 'input_reread' in accesses
        if replay != bool(replay_flag) or (replay_mode is not None and replay_mode != replay):
            raise ValueError('CANN normalization cannot omit its declared fragment reread')
        replay_mode=replay
        expected={'tiling','weight','input','output','rstd'}|({'residual','residual_output'} if fused else set())
        if replay:expected|={'residual_reread' if fused else 'input_reread'}
        if set(accesses) != expected:raise ValueError('CANN normalization source operand/replay roles incomplete')
        axes=(('aiv',count,1),('row',local_rows,1),*((('gamma_group',-(-local_rows//64),1),) if replay else ()))
        if tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != axes or family.causal_prefixes:
            raise ValueError('CANN normalization must retain exact AIV/row/gamma groups')
        for role,access in accesses.items():
            parent_role=('residual_output' if fused else 'input') if role.endswith('_reread') else role
            parent=views[parent_role];view=views[access.view_id]
            metadata=role in ('tiling','weight');is_rstd=role == 'rstd'
            parent_shape=((80 if fused else 28),) if role == 'tiling' else (hidden,) if role == 'weight' else (rows,) if is_rstd else (rows,hidden)
            dtype,element_width=('uint8',1) if role == 'tiling' else ('fp32',4) if is_rstd else (views['input'].dtype,width)
            from src.kernel.memory_flash_attention import dense_strides
            if (parent.shape,parent.strides_bytes,parent.dtype,parent.element_bytes) != (parent_shape,dense_strides(parent_shape,element_width),dtype,element_width):
                raise ValueError('CANN normalization source matrix/rstd/metadata storage mismatch')
            shape=parent_shape if metadata else (count,local_rows) if is_rstd else (count,local_rows,hidden)
            strides=parent.strides_bytes if metadata else dense_strides(shape,element_width)
            offset=parent.offset_bytes+(0 if metadata else first*block_factor*(1 if is_rstd else hidden)*element_width)
            loop=(('aiv','gamma_group') if replay else ('aiv',)) if role == 'weight' else ('aiv',) if role == 'tiling' else ('aiv','row')
            mapping=(None,) if metadata else ('aiv','row') if is_rstd else ('aiv','row',None)
            operation=Operation.WRITE if role in ('output','rstd','residual_output') else Operation.READ
            if ((view.allocation_id,view.shape,view.strides_bytes,view.dtype,view.element_bytes,view.offset_bytes)
                    != (parent.allocation_id,shape,strides,dtype,element_width,offset)
                    or access.operation is not operation or access.iteration_axes != loop or access.view_axes != mapping
                    or access.unmapped or access.indirect_target is not None or access.view_axis_divisors):
                raise ValueError('CANN normalization tiles must conserve actual ND row and argument selections')
            if role not in volumes:volumes[role]=0
            volumes[role]+=tile_template_requested_bytes(program,family,access)
        initial=tuple(TileDependency(role,EventKind.READ_READY,()) for role in (('input','residual') if fused else ('input',)))
        tiling=(TileDependency('tiling',EventKind.READ_READY,()),)
        for role in ('input','weight',*(('residual',) if fused else ())):
            if accesses[role].dependencies != tiling:raise ValueError('CANN normalization lacks actual input/gamma/reread causality')
        required=initial
        if fused and accesses['residual_output'].dependencies != initial:
            raise ValueError('CANN normalization lacks actual input/gamma/reread causality')
        if replay:
            name='residual_reread' if fused else 'input_reread'
            expected_prior=(TileDependency('residual_output',EventKind.WRITE_VISIBLE,()),) if fused else initial
            if accesses[name].dependencies != expected_prior:
                raise ValueError('CANN Split-D reread requires the actual residual output WRITE_VISIBLE' if fused else 'CANN normalization lacks actual input/gamma/reread causality')
            required+=(TileDependency(name,EventKind.READ_READY,()),)
        binding=(TileIndexBinding('gamma_group',(TileIndexTerm('row',1),),divisor=64),) if replay else ()
        required+=(TileDependency('weight',EventKind.READ_READY,(),binding),)
        if accesses['output'].dependencies != required or accesses['rstd'].dependencies != initial:
            raise ValueError('CANN normalization lacks actual input/gamma/reread causality')
    gamma=requested_bytes(views['weight'].region());payload=requested_bytes(views['tiling'].region())
    initial_read=volumes['input']+(volumes['residual'] if fused else 0)
    actual_reread=volumes['residual_reread' if fused else 'input_reread'] if replay_mode else 0
    return {'geometry':(rows,hidden,width),'input_bytes':volumes['input'],'weight_bytes':gamma,
        'weight_copies':volumes['weight']//gamma,'initial_read_bytes':initial_read,'output_bytes':volumes['output'],
        'residual_write_bytes':volumes['residual_output'] if fused else 0,'rstd_bytes':volumes['rstd'],
        'actual_reread_bytes':actual_reread,'reread_bytes':actual_reread,'tiling_payload_bytes':payload,
        'tiling_materializations':volumes['tiling']//payload,'tiling_read_bytes':volumes['tiling'],
        'interface_unique_bytes':sum(requested_bytes(views[role].region()) for role in parent_roles),
        'requested_read_bytes':logical_requested_bytes(program,Operation.READ),'requested_write_bytes':logical_requested_bytes(program,Operation.WRITE)}
