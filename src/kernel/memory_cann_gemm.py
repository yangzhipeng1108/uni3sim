"""CANN non-split matrix payloads, with native format mapping unbound."""

from src.kernel.prediction_cache import runtime_lru_cache, RuntimeCache
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from collections import OrderedDict
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity, MemoryAccess,
    MemoryAccessProgram, MemoryAllocation, Operation, Ordering, ProgramEvent, Semantics,
    TensorView, LogicalSelectionView, Unmapped, unique_by, requested_bytes, TileAxis, TileAccessTemplate, TileDependency, TileProgram, TileIndexBinding, LogicalCoordinateBinding, LogicalCoordinateTerm,
)

_WORK = ContextVar('cann_gemm_source_work')
_EXECUTION_KERNELS = RuntimeCache()


def has_cann_gemm_source(kernel):
    return (kernel.kind is KernelKind.GEMM and kernel.hardware_id == 'ascend_910b'
            and kernel.variant == 'gemm' and kernel.dtype == 'bf16' and kernel.execution_support == 'supported'
            and kernel.attributes['output_dtype'] in (None,'bf16')
            and kernel.launch.tiling is not None and kernel.launch.tiling.split_k == 1
            and kernel.launch.algorithm in ('ascend_cann_matmul','ascend_cann_matmul_v3',
                'ascend_cann_matmul_v3_asw_65536','ascend_cann_injected_matmul'))


def cann_split_execution_kernel(kernel):
    """Reuse only the pure captured-shape transform of an immutable spec.

    Source admission, binding and pricing can ask about the same original
    KernelSpec several times. Retain the exact object to prevent ID reuse;
    caller sources and their storage/readiness are never cached here.
    """
    if kernel.kind is not KernelKind.GEMM or kernel.hardware_id != 'ascend_910b':
        return kernel
    identity = id(kernel)
    if identity in _EXECUTION_KERNELS:
        prior, execution = _EXECUTION_KERNELS[identity]
        if prior is not kernel:
            raise RuntimeError('CANN execution KernelSpec identity collision')
        _EXECUTION_KERNELS.move_to_end(identity)
        return execution
    execution = _cann_split_execution_kernel(kernel)
    _EXECUTION_KERNELS[identity] = (kernel, execution)
    if len(_EXECUTION_KERNELS) > 256:
        _EXECUTION_KERNELS.popitem(last=False)
    return execution


def _cann_split_execution_kernel(kernel):
    """Use captured operands for bound CANN matrix programs, keeping the launch.

    The complete model-input descriptor identifies the executed operands;
    logical request geometry is retained on the caller. Padding values,
    logical-to-captured element aliases and native addresses remain unknown.
    """
    if (kernel.kind is not KernelKind.GEMM or kernel.hardware_id!='ascend_910b'
            or kernel.variant!='gemm' or 'model_input_shape' not in kernel.attributes
            or kernel.attributes['graph_runtime_mode'] not in ('FULL','PIECEWISE')
            or not (has_cann_gemm_source(kernel) or kernel.launch.algorithm in (
                'cann_matmul_v3_single_core_split_k','cann_matmul_v3_deterministic_split_k'))):
        return kernel
    physical=kernel.attributes['model_input_shape']
    if not {'b','m','n','k'}<=set(physical):
        return kernel
    if all(physical[name]==kernel.shape[name] for name in ('b','m','n','k')):return kernel
    if has_cann_gemm_source(kernel):
        # The ordinary matrix public ABI binds row padding only. Do not
        # interpret a changed operand width/batch as a row projection.
        if any(physical[name]!=kernel.shape[name] for name in ('b','n','k')):
            return kernel
        if (physical['m']<kernel.shape['m'] or physical['m']*kernel.attributes['graph_logical_tokens']
                !=kernel.shape['m']*kernel.attributes['graph_model_input_tokens']):
            raise ValueError('CANN matrix replay rows differ from the declared graph mapping')
    from dataclasses import replace
    attributes=dict(kernel.attributes);attributes['graph_logical_tokens']=attributes['graph_model_input_tokens']
    execution=replace(kernel,shape=dict(physical),attributes=attributes)
    if has_cann_gemm_source(execution) or has_cann_single_core_gemm_source(execution) or has_cann_deterministic_gemm_source(execution):return execution
    return kernel


def has_cann_single_core_gemm_source(kernel):
    from src.kernel.launch_config.launch_program import CannGemmTilePipelineProgram
    if (kernel.kind is not KernelKind.GEMM or kernel.hardware_id != 'ascend_910b'
            or kernel.variant != 'gemm' or kernel.dtype != 'bf16' or kernel.execution_support != 'supported'
            or kernel.attributes['output_dtype'] not in (None,'bf16')
            or ('model_input_shape' in kernel.attributes and any(
                kernel.attributes['model_input_shape'][name]!=kernel.shape[name] for name in ('b','m','n','k')))
            or kernel.launch.algorithm != 'cann_matmul_v3_single_core_split_k'):
        return False
    program = kernel.launch.source_program
    return (isinstance(program,CannGemmTilePipelineProgram)
        and program.program_kind == 'cann_gemm_single_core_split_k_v1'
        and kernel.shape['n'] == 8192 and kernel.shape['k'] == 28672
        and program.workspace_buffer_count == 1 and program.partial_element_bytes == 4)


def _single_core_header(kernel):
    source = kernel.launch.source_program
    return ('cann_single_core_split_k',*(str(kernel.shape[name]) for name in ('b','m','n','k')),
        *(str(value) for value in (source.aic_workers,source.single_core_m,source.single_core_n,
            source.single_core_k,source.base_m,source.step_m)))


def _single_core_requests(header):
    if len(header) != 11 or header[0] != 'cann_single_core_split_k':
        raise ValueError('CANN single-core source requires its worker/K-chunk geometry')
    b,m,n,k,workers,core_m,core_n,core_k,base_m,step_m = map(int,header[1:])
    if min(b,m,n,k,workers,core_m,core_n,core_k,base_m,step_m) <= 0 or (n,k) != (8192,28672):
        raise ValueError('CANN single-core source outside its captured matrix domain')
    rows = b*m
    if -(-rows//core_m)*-(-n//core_n) != workers or core_k % 128:
        raise ValueError('CANN single-core source must give one block to each bounded AIC')
    # Every AIC owns a different full output block. Independent blocks can
    # be linearized by K pass without changing any accumulator's order.
    # This is a logical reference order, not a native cross-AIC barrier.
    full_m_groups,tail = divmod(rows,core_m)
    inner_m = base_m*step_m
    stripes = full_m_groups*-(-core_m//inner_m)+(0 if tail == 0 else -(-tail//inner_m))
    n_groups = -(-n//core_n)
    state = f'all_worker_blocks.rows0:{rows}.cols0:{n}'
    for chunk,start in enumerate(range(0,k,core_k)):
        inner = min(core_k,k-start)
        yield ('A',0,chunk,'input_a',(rows,inner),n_groups,f'rows0:{rows}.K{start}:{start+inner}')
        yield ('B',0,chunk,'input_b',(n,inner),stripes,f'cols0:{n}.K{start}:{start+inner}')
        if chunk:
            yield ('workspace',0,chunk,'rmw_read',(rows,n),1,state)
        yield ('workspace',0,chunk,'partial_first' if chunk == 0 else 'partial_update',(rows,n),1,state)
    yield ('workspace',0,chunk,'aiv_read',(rows,n),1,state)
    yield ('C',0,chunk,'output',(rows,n),1,state)


def _single_core_region(view,role,shape,selection):
    return Unmapped(shape,view.dtype,view.element_bytes,f'cann_single_core.{role}.{selection}')


def standalone_cann_single_core_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_single_core_gemm_source(kernel):
        raise NotImplementedError('CANN single-core split-K source ABI is not bound')
    program = kernel.launch.source_program
    layouts = (*((role,shape,'bf16',2) for role,shape in _layouts(kernel)),
        ('workspace',(program.aic_workers,program.single_core_m,program.single_core_n),'fp32',4))
    allocations,views,events,read_views,ready = [],[],[],[],[]
    for role,shape,dtype,width in layouts:
        identity = f'{binding_id}.{role}'
        strides,size = [],width
        for extent in reversed(shape):
            strides.append(size);size *= extent
        allocation = MemoryAllocation(identity,'logical-local-device','logical-native-matrix',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation)
        views.append(TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width))
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
    by_role = unique_by(views,'view_id')
    for role,worker,chunk,action,shape,_,selection in _single_core_requests(_single_core_header(kernel)):
        if action not in ('input_a','input_b'):
            continue
        view = by_role[role]
        event = ProgramEvent(f'{binding_id}.{role}.{worker}.{chunk}.available',EventKind.LOGICAL_AVAILABLE,
            view.allocation_id,_single_core_region(view,role,shape,selection),'cann-visible',
            (EventRef(f'{view.allocation_id}.allocated',EventKind.ALLOCATED),))
        events.append(event);read_views.append(role);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        tuple(read_views),('workspace','C'),tuple(ready),0,'offline','cann-visible')


def build_cann_single_core_source(kernel,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_single_core_gemm_source(kernel):
        raise NotImplementedError('CANN single-core source requires its retained BF16 worker program')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('CANN single-core source requires actual A/B/C/workspace bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('CANN single-core exports logical stripe/column accumulator tiles')
    source,header = kernel.launch.source_program,_single_core_header(kernel)
    if kernel.launch.grid != (source.aic_workers,1,1) or kernel.launch.block != (1,1,1):
        raise ValueError('CANN single-core source worker grid mismatch')
    requests = tuple(_single_core_requests(header))
    expected_reads = tuple(role for role,_,_,action,_,_,_ in requests if action in ('input_a','input_b'))
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (expected_reads,('workspace','C'),len(expected_reads)):
        raise ValueError('CANN single-core source requires each actual input selection and internal workspace')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    layouts = (*((role,shape,'bf16',2) for role,shape in _layouts(kernel)),
        ('workspace',(source.aic_workers,source.single_core_m,source.single_core_n),'fp32',4))
    if set(views) != {'A','B','C','workspace'} or len({v.allocation_id for v in views.values()}) != 4:
        raise ValueError('CANN single-core native operand/workspace alias is unbound')
    for role,shape,dtype,width in layouts:
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes,view.offset_bytes) != (shape,dtype,width,0):
            raise ValueError('CANN single-core matrix/workspace storage dtype or extent mismatch')
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-single-core-states')
    selected,families = _single_core_tile_recipe(header,views,bindings.input_ready,scope,bindings.visibility_domain)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Native ND/NZ/local tile/GM workspace addresses/capacity remain unknown; no issued trace.',
            'K passes linearize independent output-block states without changing per-block accumulation order; no native cross-AIC barrier/order or atomic cache transaction claim.',
            'Logical RMW reads are separate from retained AIV-read pricing; existing workspace DRAM/cache projection is unverified legacy policy.',
            'Captured K28672/N8192 and bounded AIC count allow source steps without per-element or arbitrary tile-grid expansion.')),
        bindings.allocations,(*bindings.views,*selected),(),bindings.events,(),families)
    cann_single_core_source_demands(program)
    return program


def _single_core_tile_recipe(header,views,input_ready,scope,visibility):
    from dataclasses import replace
    requests = tuple(_single_core_requests(header))
    b,m,n,k,workers,cm,cn,ck,bm,sm = map(int,header[1:])
    rows,inner_m = b*m,bm*sm
    chunks = -(-k//ck)
    if len(input_ready) != chunks*2:raise ValueError('CANN single-core requires every actual K-slice availability')
    evidence = ('CANN8.5.1 key65632: retained single_core_m/n/k and base_m*step_m input stripes define logical work blocks.',
        'Each output stripe/column block retains its own FP32 accumulator chain and paired-AIV BF16 publication; no cross-output barrier.',
        'K-slice availability is exact logical selection evidence; all native ND/NZ/workspace offsets and capacities remain unknown.')
    selected,roots,ready = [],{},iter(input_ready)
    for role,_,chunk,action,shape,_,selection in requests:
        if action not in ('input_a','input_b'):continue
        region = _single_core_region(views[role],role,shape,selection)
        view = LogicalSelectionView(f'{scope.scope_id}.{role}.K{chunk}',views[role].allocation_id,
            shape,views[role].dtype,views[role].element_bytes,region.selection_id,None)
        selected.append(view);roots[(role,chunk)] = (view,next(ready))
    families = []
    for start in range(0,rows,cm):
        count = min(cm,rows-start)
        identity = f'{scope.scope_id}.rows{start}'
        workspace = LogicalSelectionView(f'{identity}.state',views['workspace'].allocation_id,(count,n),'fp32',4,
            f'cann_single_core.workspace.rows{start}:{start+count}.cols0:{n}','workspace')
        output = LogicalSelectionView(f'{identity}.output',views['C'].allocation_id,(count,n),'bf16',2,
            f'cann_single_core.C.rows{start}:{start+count}.cols0:{n}','C')
        selected.extend((workspace,output));accesses = []
        previous = None
        for chunk in range(chunks):
            a,a_ready = roots[('A',chunk)];bv,b_ready = roots[('B',chunk)]
            stripe = LogicalSelectionView(f'{identity}.A.K{chunk}',a.allocation_id,(count,a.shape[1]),'bf16',2,
                f'{a.selection_id}.rows{start}:{start+count}',a.view_id)
            selected.append(stripe)
            a_id,b_id = f'input_a{chunk}',f'input_b{chunk}'
            accesses.extend((
                TileAccessTemplate(a_id,stripe.view_id,Operation.READ,('row','column'),('row',None),
                    (a_ready,),(),evidence,unmapped=True),
                TileAccessTemplate(b_id,bv.view_id,Operation.READ,('row','column'),('column',None),
                    (b_ready,),(),evidence,unmapped=True)))
            dependencies = (TileDependency(a_id,EventKind.READ_READY,()),TileDependency(b_id,EventKind.READ_READY,()))
            if previous is not None:
                read = f'rmw_read{chunk}'
                accesses.append(TileAccessTemplate(read,workspace.view_id,Operation.READ,('row','column'),('row','column'),(),
                    (TileDependency(previous,EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True))
                dependencies += (TileDependency(read,EventKind.READ_READY,()),)
            previous = f'partial_first{chunk}' if chunk == 0 else f'partial_update{chunk}'
            accesses.append(TileAccessTemplate(previous,workspace.view_id,Operation.WRITE,('row','column'),('row','column'),(),
                dependencies,evidence,unmapped=True))
        accesses.extend((
            TileAccessTemplate('aiv_read',workspace.view_id,Operation.READ,('row','column'),('row','column'),(),
                (TileDependency(previous,EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True),
            TileAccessTemplate('output',output.view_id,Operation.WRITE,('row','column'),('row','column'),(),
                (TileDependency('aiv_read',EventKind.READ_READY,()),),evidence,unmapped=True)))
        family_scope = replace(scope,scope_id=identity,op_ids=(*header,str(start),str(count)))
        families.append(TileProgram(identity,family_scope,(TileAxis('row',count,inner_m),TileAxis('column',n,cn)),
            tuple(accesses),visibility,evidence))
    return tuple(selected),tuple(families)


def cann_single_core_tile_demands(program):
    work=_cann_single_core_tile_demands(program)
    return {**work,'totals':dict(work['totals'])}


@runtime_lru_cache(maxsize=8)
def _cann_single_core_tile_demands(program):
    from src.kernel.memory_access import tile_template_requested_bytes
    from dataclasses import replace
    program.__post_init__()
    if not program.tile_programs:raise ValueError('CANN single-core requires its actual stripe state loops')
    family = program.tile_programs[0]
    header = family.scope.op_ids[:11]
    if len(family.scope.op_ids) != 13 or family.scope.agent != 'logical-cann-single-core-states':
        raise ValueError('CANN single-core stripe fragment is missing')
    b,m,n,k,workers,cm,cn,ck,bm,sm = map(int,header[1:])
    views = unique_by(program.views,'view_id')
    expected = {'A':((b,m,k),'bf16',2),'B':((n,k),'bf16',2),'C':((b,m,n),'bf16',2),
        'workspace':((workers,cm,cn),'fp32',4)}
    if len({views[role].allocation_id for role in expected}) != 4:
        raise ValueError('CANN single-core source requires independent actual native parents')
    for role,(shape,dtype,width) in expected.items():
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes,view.offset_bytes) != (shape,dtype,width,0):
            raise ValueError('CANN single-core source storage/geometry mismatch')
    inputs = tuple(access for access in family.accesses if access.access_id.startswith(('input_a','input_b')))
    if any(len(access.input_ready) != 1 for access in inputs):raise ValueError('CANN single-core K slices require actual readiness')
    scope = replace(family.scope,scope_id=program.program_id,op_ids=header)
    selected,families = _single_core_tile_recipe(header,views,tuple(access.input_ready[0] for access in inputs),scope,family.visibility_domain)
    if set(views) != set(expected)|{view.view_id for view in selected} or any(views[view.view_id] != view for view in selected):
        raise ValueError('CANN single-core source must conserve actual K-range and partial/output access')
    if program.tile_programs != families:
        actual = tuple(access for group in program.tile_programs for access in group.accesses)
        reference = tuple(access for group in families for access in group.accesses)
        for access,expected_access in zip(actual,reference):
            if access != expected_access:
                if access.access_id.startswith(('rmw_read','aiv_read')):
                    raise ValueError('CANN accumulator/reduction requires actual previous WRITE_VISIBLE, never initializer')
                if access.access_id == 'output':raise ValueError('CANN final BF16 output requires actual paired-AIV workspace READ_READY')
                break
        raise ValueError('CANN partial requires actual A/B and accumulator READ_READY in each stripe state')
    totals = {name:0 for name in ('input_a','input_b','rmw_read','partial_first','partial_update','aiv_read','output')}
    for group in program.tile_programs:
        for access in group.accesses:
            role = next(name for name in totals if access.access_id.startswith(name))
            totals[role] += tile_template_requested_bytes(program,group,access)
    return {'geometry':(b,m,n,k),'totals':totals,'operand_unique_bytes':(b*m*k+n*k)*2,
        'read_bytes':sum(totals[role] for role in ('input_a','input_b','rmw_read','aiv_read')),
        'write_bytes':sum(totals[role] for role in ('partial_first','partial_update','output'))}


def cann_single_core_source_demands(program):
    if program.tile_programs:return cann_single_core_tile_demands(program)
    header = program.scopes[0].op_ids
    requests = tuple(_single_core_requests(header))
    views,scopes = unique_by(program.views,'view_id'),unique_by(program.scopes,'scope_id')
    b,m,n,k,workers,core_m,core_n,_,_,_ = map(int,header[1:])
    expected = {'A':((b,m,k),'bf16',2),'B':((n,k),'bf16',2),'C':((b,m,n),'bf16',2),
        'workspace':((workers,core_m,core_n),'fp32',4)}
    if set(views) != set(expected) or len({v.allocation_id for v in views.values()}) != 4:
        raise ValueError('CANN single-core source requires independent actual native parents')
    for role,(shape,dtype,width) in expected.items():
        v=views[role]
        if (v.shape,v.dtype,v.element_bytes,v.offset_bytes) != (shape,dtype,width,0):
            raise ValueError('CANN single-core source storage/geometry mismatch')
    if len(program.accesses) != len(requests):
        raise ValueError('CANN single-core source lost an input/update/reduction step')
    totals = {name:0 for name in ('input_a','input_b','rmw_read','partial_first','partial_update','aiv_read','output')}
    latest,inputs,updates,reduced = {},{},{},{}
    for access,(role,worker,chunk,action,shape,repeat,selection) in zip(program.accesses,requests,strict=True):
        reading = action in ('input_a','input_b','rmw_read','aiv_read')
        if (scopes[access.scope_id].op_ids != (*header,role,str(worker),str(chunk),action)
                or access.region != _single_core_region(views[role],role,shape,selection)
                or access.allocation_id != views[role].allocation_id or access.view_id is not None
                or access.operation is not (Operation.READ if reading else Operation.WRITE) or access.repeat != repeat):
            raise ValueError('CANN single-core source must conserve actual K-range and partial/output access')
        deps = set(access.dependencies)
        if action in ('input_a','input_b'):
            if action == 'input_a':inputs[worker]=set()
            inputs[worker].add(EventRef(access.access_id,EventKind.READ_READY))
        elif action in ('rmw_read','aiv_read'):
            if latest[worker] not in deps:
                raise ValueError('CANN accumulator/reduction requires actual previous WRITE_VISIBLE, never initializer')
            if action == 'rmw_read':updates[worker]=EventRef(access.access_id,EventKind.READ_READY)
            else:reduced[worker]=EventRef(access.access_id,EventKind.READ_READY)
        elif action in ('partial_first','partial_update'):
            if not inputs[worker] <= deps or (action == 'partial_update' and updates[worker] not in deps):
                raise ValueError('CANN partial requires actual A/B and accumulator READ_READY')
            latest[worker]=EventRef(access.access_id,EventKind.WRITE_VISIBLE)
        elif action == 'output':
            if reduced[worker] not in deps:
                raise ValueError('CANN final BF16 output requires actual paired-AIV workspace READ_READY')
        totals[action] += requested_bytes(access.region)*repeat
    return {'geometry':(b,m,n,k),'totals':totals,'operand_unique_bytes':(b*m*k+n*k)*2,
        'read_bytes':sum(totals[x] for x in ('input_a','input_b','rmw_read','aiv_read')),
        'write_bytes':sum(totals[x] for x in ('partial_first','partial_update','output'))}


def cann_single_core_traffic_projection(work,l2_replacement_capacity_bytes):
    t=work['totals'];workspace=t['partial_first']+t['partial_update']
    if l2_replacement_capacity_bytes is not None and (type(l2_replacement_capacity_bytes) is not int or l2_replacement_capacity_bytes <= 0):
        raise ValueError('CANN single-core split-K L2 capacity must be positive')
    return {'operand_l2_read_bytes':float(t['input_a']+t['input_b']),
        'operand_dram_read_bytes':float(work['operand_unique_bytes']),
        'workspace_l2_write_bytes':float(workspace),
        'workspace_dram_write_bytes':float(workspace if l2_replacement_capacity_bytes is not None and workspace > l2_replacement_capacity_bytes else 0),
        'workspace_l2_read_bytes':float(t['aiv_read']),'output_l2_write_bytes':float(t['output']),
        'output_dram_write_bytes':float(t['output'])}


@contextmanager
def cann_gemm_source_work(kernel, program, work):
    token = _WORK.set((kernel,program,work))
    try:
        yield
    finally:
        _WORK.reset(token)


def bound_cann_gemm_source(kernel):
    context = copy_context()
    if _WORK not in context:
        return None
    current,program,work = context[_WORK]
    if current != kernel:
        raise ValueError('CANN GEMM source crossed its kernel boundary')
    return program,work


def _layouts(kernel):
    b,m,n,k = (kernel.shape[name] for name in ('b','m','n','k'))
    return (('A',(b,m,k)),('B',(n,k)),('C',(b,m,n)))


def _region(view, binding_id):
    return Unmapped(view.shape,view.dtype,view.element_bytes,f'{binding_id}.{view.view_id}.native_matrix_payload')


def standalone_cann_gemm_bindings(kernel, *, call_id, binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    allocations,views,events,ready = [],[],[],[]
    for role,shape in _layouts(kernel):
        identity = f'{binding_id}.{role}'
        strides,size = [],2
        for axis in reversed(shape):
            strides.append(size)
            size *= axis
        # Shape/strides describe the public logical matrix only; accesses
        # never resolve these descriptor strides to native ND/NZ addresses.
        view = TensorView(role,identity,0,shape,tuple(reversed(strides)),'bf16',2)
        allocation = MemoryAllocation(identity,'logical-local-device','logical-native-matrix',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation)
        views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role != 'C':
            event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
                _region(view,binding_id),'cann-visible',(EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event)
            ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('A','B'),('C',),tuple(ready),0,'offline','cann-visible')


def build_cann_gemm_source(kernel, hardware, bindings, granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('CANN matrix source requires typed native operand availability')
    if granularity not in (Granularity.OP, Granularity.KERNEL, Granularity.TILE):
        raise ValueError('CANN source exports logical matrix tiles only')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (('A','B'),('C',),2):
        raise ValueError('CANN matrix source operand roles mismatch')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    if set(views) != {'A','B','C'} or len({view.allocation_id for view in views.values()}) != 3:
        raise NotImplementedError('CANN matrix alias requires explicit native storage mapping')
    for role,shape in _layouts(kernel):
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes) != (shape,'bf16',2):
            raise ValueError('CANN logical matrix shape/dtype mismatch')
    evidence = ('ascend_910b/gemm_runner.py:torch Linear(K,N), input(B,SL,K), output(B,SL,N)',
        'One output coordinate consumes its full mathematical K dot; this semantic partition is independent of native AIC scheduling',
        'Logical B descriptor [N,K] does not resolve native ND/NZ addresses, transpose format, capacity or conversion state')
    b, m, n, k = (kernel.shape[name] for name in ('b', 'm', 'n', 'k'))
    axes = (TileAxis('batch', b, 1), TileAxis('row', m, 1), TileAxis('column', n, 1))
    iterations = ('batch', 'row', 'column')
    accesses = (
        TileAccessTemplate('A', 'A', Operation.READ, iterations, ('batch', 'row', None),
            (bindings.input_ready[0],), (), evidence, unmapped=True),
        TileAccessTemplate('B', 'B', Operation.READ, iterations, ('column', None),
            (bindings.input_ready[1],), (), evidence, unmapped=True),
        TileAccessTemplate('C', 'C', Operation.WRITE, iterations, iterations, (), (
            TileDependency('A', EventKind.READ_READY, ()), TileDependency('B', EventKind.READ_READY, ())),
            evidence, unmapped=True))
    scope = AccessScope(program_id, ('cann_gemm', 'non_split_dot'), bindings.call_id, (),
        bindings.rank, bindings.stream, 'logical-cann-matrix-dots')
    family = TileProgram(program_id, scope, axes, accesses, bindings.visibility_domain, evidence)
    program = MemoryAccessProgram('midlayer.memory-access.v1', program_id, hardware.hardware_id, bindings.binding_id,
        Coverage(Granularity.TILE, Semantics.LOGICAL, Ordering.PARTIAL, False, (
            'Public logical descriptors do not resolve native ND/NZ addresses, transpose mappings, capacity or physical footprint.',
            'Compressed independent output dots retain logical operand rereads; retained native tile/cache/padding projections consume the validated interface extents.',
            'Internal conversion/local buffers, warm producer identity and native scheduling remain partial.')),
        bindings.allocations, bindings.views, (), bindings.events, (), (family,))
    cann_gemm_source_demands(program)
    return program


def cann_gemm_tile_demands(program):
    """Validate logical output dots; project their compulsory interface volume."""
    return dict(_cann_gemm_tile_demands(program))


@runtime_lru_cache(maxsize=8)
def _cann_gemm_tile_demands(program):
    from math import prod
    program.__post_init__()
    if len(program.tile_programs) != 1:
        raise ValueError('CANN matrix requires one dot-product tile loop')
    family, = program.tile_programs
    views = unique_by(program.views, 'view_id')
    accesses = unique_by(family.accesses, 'access_id')
    if (set(views) != {'A', 'B', 'C'} or set(accesses) != {'A', 'B', 'C'}
            or family.scope.agent != 'logical-cann-matrix-dots'
            or family.scope.op_ids != ('cann_gemm', 'non_split_dot')):
        raise ValueError('CANN matrix source roles must be unique')
    a, weight, output = (views[name] for name in ('A', 'B', 'C'))
    if len(a.shape) != 3 or len(weight.shape) != 2:
        raise ValueError('CANN matrix source contraction/output geometry mismatch')
    b, m, k = a.shape
    n, inner = weight.shape
    if inner != k or output.shape != (b, m, n):
        raise ValueError('CANN matrix source contraction/output geometry mismatch')
    if len({view.allocation_id for view in views.values()}) != 3:
        raise ValueError('CANN matrix source requires independent operand parents')
    if tuple((axis.axis_id, axis.extent, axis.block) for axis in family.axes) != (
            ('batch', b, 1), ('row', m, 1), ('column', n, 1)):
        raise ValueError('CANN matrix must preserve independent exact output dots')
    iterations = ('batch', 'row', 'column')
    for name, mapping in (('A', ('batch', 'row', None)), ('B', ('column', None)), ('C', iterations)):
        access, view = accesses[name], views[name]
        if ((view.dtype, view.element_bytes) != ('bf16', 2) or access.view_id != name
                or access.operation is not (Operation.WRITE if name == 'C' else Operation.READ)
                or access.iteration_axes != iterations or access.view_axes != mapping
                or access.view_axis_divisors or access.indirect_target is not None or not access.unmapped):
            raise ValueError('CANN matrix must preserve unresolved exact logical dot operands')
    required = tuple(TileDependency(name, EventKind.READ_READY, ()) for name in ('A', 'B'))
    if accesses['C'].dependencies != required:
        raise ValueError('CANN matrix output lacks input READ_READY causality for its own dot')
    # Interface extents are mathematical logical payloads, not native unique
    # footprint or replayed requests. This preserves the existing pricing ABI.
    operands = tuple(prod(views[name].shape) * views[name].element_bytes for name in ('A', 'B', 'C'))
    return {'geometry': (b, m, n, k), 'read_bytes': sum(operands[:2]),
        'write_bytes': operands[2], 'operand_bytes': operands}


def cann_gemm_source_demands(program):
    if program.tile_programs:
        return cann_gemm_tile_demands(program)
    from src.kernel.memory_path_solver import logical_requested_bytes
    from src.kernel.memory_access import requested_bytes
    logical_requested_bytes(program,Operation.READ)
    scopes,views = unique_by(program.scopes,'scope_id'),unique_by(program.views,'view_id')
    operands = {}
    for access in program.accesses:
        roles = scopes[access.scope_id].op_ids
        if len(roles) != 2 or roles[0] != 'cann_gemm' or roles[1] not in ('A','B','C') or roles[1] in operands:
            raise ValueError('CANN matrix source roles must be unique')
        role = roles[1]
        view = views[role]
        if (access.repeat != 1 or access.view_id is not None or not isinstance(access.region,Unmapped)
                or access.allocation_id != view.allocation_id or access.region.logical_shape != view.shape
                or (access.region.dtype,access.region.element_bytes,view.dtype,view.element_bytes) != ('bf16',2,'bf16',2)
                or access.operation is not (Operation.WRITE if role == 'C' else Operation.READ)):
            raise ValueError('CANN matrix source storage/operation mismatch')
        operands[role] = access
    if set(operands) != {'A','B','C'}:
        raise ValueError('CANN matrix source operand missing')
    if len({access.allocation_id for access in operands.values()}) != 3:
        raise ValueError('CANN matrix source requires independent operand parents')
    b,m,k = operands['A'].region.logical_shape
    n,inner = operands['B'].region.logical_shape
    if inner != k or operands['C'].region.logical_shape != (b,m,n):
        raise ValueError('CANN matrix source contraction/output geometry mismatch')
    nodes = {event.event_id:event for event in program.events}
    nodes.update({access.access_id:access for access in program.accesses})
    pending,ancestors = list(operands['C'].dependencies),set()
    while pending:
        ref = pending.pop()
        if ref not in ancestors:
            ancestors.add(ref)
            pending.extend(nodes[ref.node_id].dependencies)
    if any(EventRef(operands[role].access_id,EventKind.READ_READY) not in ancestors for role in ('A','B')):
        raise ValueError('CANN matrix output lacks input READ_READY causality')
    return {'geometry':(b,m,n,k),'read_bytes':logical_requested_bytes(program,Operation.READ),
            'write_bytes':logical_requested_bytes(program,Operation.WRITE),
            'operand_bytes':tuple(requested_bytes(operands[role].region) for role in ('A','B','C'))}


def has_cann_deterministic_gemm_source(kernel):
    from src.kernel.launch_config.launch_program import CannGemmTilePipelineProgram
    if (kernel.kind is not KernelKind.GEMM or kernel.hardware_id != 'ascend_910b'
            or kernel.variant != 'gemm' or kernel.dtype != 'bf16'
            or kernel.execution_support != 'supported'
            or kernel.attributes['output_dtype'] not in (None,'bf16')
            or ('model_input_shape' in kernel.attributes and any(
                kernel.attributes['model_input_shape'][name]!=kernel.shape[name] for name in ('b','m','n','k')))
            or kernel.launch.algorithm != 'cann_matmul_v3_deterministic_split_k'):
        return False
    source = kernel.launch.source_program
    return (isinstance(source,CannGemmTilePipelineProgram)
        and source.program_kind == 'cann_gemm_deterministic_split_k_v1'
        and source.workspace_buffer_count == 2 and source.partial_element_bytes == 4
        and source.scheduler_policy in ('n_buffer_33','l2_split_matmul'))


def _deterministic_header(kernel):
    source = kernel.launch.source_program
    return ('cann_deterministic_split_k',*(str(kernel.shape[key]) for key in ('b','m','n','k')),
        *(str(value) for value in (source.aic_workers,source.single_core_m,source.single_core_n,source.single_core_k)))


def _deterministic_layouts(header):
    if len(header) != 9 or header[0] != 'cann_deterministic_split_k':
        raise ValueError('CANN deterministic source requires its typed slab/K geometry')
    b,m,n,k,workers,cm,cn,ck = map(int,header[1:])
    if min(b,m,n,k,workers,cm,cn,ck) <= 0 or -(-k//ck) < workers:
        raise ValueError('CANN deterministic source must give every logical K worker a chunk')
    # These views name logical descriptors, never native reserved strides.
    # Separate parents represent the two workspace buffers; their physical
    # addresses, padding, capacities and allocation coalescing are unknown.
    return (('A',(b,m,k),'bf16',2),('B',(n,k),'bf16',2),('C',(b,m,n),'bf16',2),
        ('workspace0',(workers,cm,cn),'fp32',4),('workspace1',(workers,cm,cn),'fp32',4))


def _deterministic_region(view,selection,shape):
    return Unmapped(shape,view.dtype,view.element_bytes,f'cann_deterministic.{view.view_id}.{selection}')


def standalone_cann_deterministic_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_deterministic_gemm_source(kernel):
        raise NotImplementedError('CANN deterministic Split-K source ABI is unbound')
    header = _deterministic_header(kernel)
    allocations,views,events,reads,ready = [],[],[],[],[]
    for role,shape,dtype,width in _deterministic_layouts(header):
        identity = f'{binding_id}.{role}'
        strides,size = [],width
        for extent in reversed(shape):
            strides.append(size);size *= extent
        allocation = MemoryAllocation(identity,'logical-local-device','logical-native-matrix',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation)
        views.append(TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width))
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
    for role in ('A','B'):
        view = next(view for view in views if view.view_id == role)
        event = ProgramEvent(f'{binding_id}.{role}.available',EventKind.LOGICAL_AVAILABLE,view.allocation_id,
            _deterministic_region(view,'complete_input',view.shape),'cann-visible',
            (EventRef(f'{view.allocation_id}.allocated',EventKind.ALLOCATED),))
        events.append(event);reads.append(role);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        tuple(reads),('workspace0','workspace1','C'),tuple(ready),0,'offline','cann-visible')


def build_cann_deterministic_source(kernel,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_deterministic_gemm_source(kernel):
        raise NotImplementedError('CANN deterministic Split-K source ABI is unbound')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('CANN deterministic source requires A/B/C and two workspace parents')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('Unknown CANN deterministic logical granularity')
    source,header = kernel.launch.source_program,_deterministic_header(kernel)
    if kernel.launch.grid != (source.aic_workers,1,1) or kernel.launch.block != (1,1,1):
        raise ValueError('CANN deterministic source AIC grid mismatch')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (('A','B'),('workspace0','workspace1','C'),2):
        raise ValueError('CANN deterministic source requires both actual inputs and both internal buffers')
    views = unique_by(bindings.views,'view_id');_validate_deterministic_views(header,views)
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-deterministic-dots')
    selected,family = _deterministic_tile_recipe(header,views,bindings.input_ready,scope,bindings.visibility_domain)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Scalar output/K-owner semantic tiles retain literal single_core_K chunks and balanced logical K owners; no native CTA/issue claim.',
            'Workspace0/1 are distinct logical parents with unknown native strides/capacities; each scalar RMW reloads its actual previous K-pass write.',
            'All worker AIV reads precede their own BF16 output; the source collective release precedes the next macro slab.',
            'Native lane mapping, ND/NZ/UB addresses, atomic transactions and physical cache residency remain unbound; RMW read service remains unpriced.')),
        bindings.allocations,(*bindings.views,*selected),(scope,),bindings.events,(),(family,))
    cann_deterministic_source_demands(program)
    return program


def _validate_deterministic_views(header,views):
    layouts = _deterministic_layouts(header)
    if set(views) != {row[0] for row in layouts} or len({v.allocation_id for v in views.values()}) != len(layouts):
        raise ValueError('CANN deterministic native parent aliases are unbound')
    for role,shape,dtype,width in layouts:
        v = views[role]
        strides,size = [],width
        for extent in reversed(shape):
            strides.append(size);size *= extent
        if (v.shape,v.dtype,v.element_bytes,v.offset_bytes,v.strides_bytes) != (shape,dtype,width,0,tuple(reversed(strides))):
            raise ValueError('CANN deterministic logical storage dtype/descriptor mismatch')


def cann_deterministic_source_demands(program):
    work=_cann_deterministic_source_demands(program)
    return {**work,'totals':dict(work['totals'])}


@runtime_lru_cache(maxsize=8)
def _cann_deterministic_source_demands(program):
    from src.kernel.memory_access import tile_template_requested_bytes
    program.__post_init__()
    if program.accesses or len(program.scopes) != 1 or len(program.tile_programs) != 1:
        raise ValueError('CANN deterministic requires its actual scalar K-owner tile source')
    scope, = program.scopes;header = scope.op_ids
    all_views = unique_by(program.views,'view_id')
    views = {role:all_views[role] for role,_,_,_ in _deterministic_layouts(header)}
    _validate_deterministic_views(header,views)
    family, = program.tile_programs;ready = {}
    for access in family.accesses:
        if access.access_id.endswith(('input_a_first','input_b_first','input_a_update','input_b_update')):
            role = 'A' if 'input_a_' in access.access_id else 'B'
            if len(access.input_ready) != 1 or role in ready and ready[role] != access.input_ready[0]:
                raise ValueError('CANN deterministic actual operand availability missing or inconsistent')
            ready[role] = access.input_ready[0]
    if set(ready) != {'A','B'}:raise ValueError('CANN deterministic requires both actual matrix inputs')
    selected,expected = _deterministic_tile_recipe(header,views,(ready['A'],ready['B']),scope,family.visibility_domain)
    if (family != expected or set(all_views) != {*views,*(view.view_id for view in selected)}
            or any(all_views[view.view_id] != view for view in selected)):
        raise ValueError('CANN deterministic lost actual previous WRITE_VISIBLE, A/B/accumulator READ_READY, all-worker AIV reduction or slab release')
    totals = {name:0 for name in ('input_a','input_b','rmw_read','partial_first','partial_update','aiv_read','output')}
    for access in family.accesses:
        if access.access_id.endswith(('input_a_first','input_a_update')):role = 'input_a'
        elif access.access_id.endswith(('input_b_first','input_b_update')):role = 'input_b'
        else:role = access.access_id.rsplit('.',1)[-1]
        totals[role] += tile_template_requested_bytes(program,family,access)
    from math import prod
    return {'geometry':tuple(map(int,header[1:5])),'totals':totals,
        'operand_unique_bytes':sum(prod(views[role].shape)*views[role].element_bytes for role in ('A','B')),
        'read_bytes':sum(totals[role] for role in ('input_a','input_b','rmw_read','aiv_read')),
        'write_bytes':sum(totals[role] for role in ('partial_first','partial_update','output'))}


def cann_deterministic_traffic_projection(work):
    totals = work['totals']
    workspace = totals['partial_first']+totals['partial_update']
    return {'operand_l2_read_bytes':float(totals['input_a']+totals['input_b']),
        'operand_dram_read_bytes':float(work['operand_unique_bytes']),
        'workspace_l2_write_bytes':float(workspace),'workspace_dram_write_bytes':float(workspace),
        'workspace_l2_read_bytes':float(totals['aiv_read']),
        'output_l2_write_bytes':float(totals['output']),'output_dram_write_bytes':float(totals['output'])}


def _deterministic_workers(header):
    _deterministic_layouts(header)
    k,workers,ck = int(header[4]),int(header[5]),int(header[8])
    chunks = -(-k//ck);regular,extra = divmod(chunks,workers)
    for worker in range(workers):
        first = worker*regular+min(worker,extra);count = regular+int(worker < extra)
        yield worker,first*ck,min(k,(first+count)*ck)-first*ck


def _deterministic_tile_recipe(header,views,input_ready,scope,visibility):
    from dataclasses import replace
    b,m,n,k,workers,cm,cn,ck = map(int,header[1:]);rows = b*m
    if len(input_ready) != 2:raise ValueError('CANN deterministic requires complete A/B logical availability')
    evidence = ('CANN8.5.1 deterministic source: contiguous balanced logical K owners, literal single_core_K chunks and two FP32 workspace parents.',
        'Each output coordinate is a semantic K-chunk dot tile: A rows broadcast across columns, B columns across rows; these are logical operand requests, not native DMA/CTA transactions.',
        'A scalar partial reloads its previous actual WRITE_VISIBLE on later K passes; first write is the explicit carry initializer, never a memory initializer.',
        'The last actual partial of every worker feeds the AIV reduction for that same coordinate; collective output release gates the next slab.',
        'Logical input/state/output coordinates do not resolve native ND/NZ/workspace strides, addresses, capacities or cache residency.')
    selected,accesses,axes = [],[],[];roots = {};owner_rows = tuple(_deterministic_workers(header))
    for worker,start,count in owner_rows:
        for role,parent,shape,mapping in (
                ('A',views['A'],(rows,count),(
                    LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),divisor=m),
                    LogicalCoordinateBinding(1,(LogicalCoordinateTerm(0,1),),modulus=m),
                    LogicalCoordinateBinding(2,(LogicalCoordinateTerm(1,1),),start))),
                ('B',views['B'],(n,count),(
                    LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),)),
                    LogicalCoordinateBinding(1,(LogicalCoordinateTerm(1,1),),start)))):
            root = LogicalSelectionView(f'{scope.scope_id}.w{worker}.{role}',parent.allocation_id,shape,'bf16',2,
                f'cann_deterministic.{role}.owner{worker}.K{start}:{start+count}',role,mapping)
            roots[(worker,role)] = root;selected.append(root)
        if count > ck:axes.append(TileAxis(f'k_w{worker}',count-ck,ck))
    previous = None;previous_axes = ();group = 0
    for ns in range(0,n,cn):
        ne = min(cn,n-ns)
        for ms in range(0,rows,cm):
            me = min(cm,rows-ms);row,column = f'row_g{group}',f'column_g{group}'
            axes.extend((TileAxis(row,me,1),TileAxis(column,ne,1)))
            release = () if previous is None else (TileDependency(previous,EventKind.WRITE_VISIBLE,previous_axes),)
            reductions = []
            for worker,start,count in owner_rows:
                prefix = f'g{group}.w{worker}';state_parent = views[f'workspace{group%2}']
                state = LogicalSelectionView(f'{scope.scope_id}.{prefix}.state',state_parent.allocation_id,(me,ne),'fp32',4,
                    f'cann_deterministic.group{group}.worker{worker}.rows{ms}:{ms+me}.cols{ns}:{ns+ne}',state_parent.view_id,(
                        LogicalCoordinateBinding(0,(),worker),LogicalCoordinateBinding(1,(LogicalCoordinateTerm(0,1),)),
                        LogicalCoordinateBinding(2,(LogicalCoordinateTerm(1,1),))))
                selected.append(state)
                phases = (('first',0,min(ck,count)),*( (('update',ck,count-ck),) if count > ck else ()))
                for phase,offset,length in phases:
                    update = phase == 'update';axis = f'k_w{worker}'
                    for role,local_start,extent,spatial,ready in (('A',ms,me,row,input_ready[0]),('B',ns,ne,column,input_ready[1])):
                        parent = roots[(worker,role)]
                        view = LogicalSelectionView(f'{scope.scope_id}.{prefix}.{role}.{phase}',parent.allocation_id,(extent,length),'bf16',2,
                            f'{parent.selection_id}.rows{local_start}:{local_start+extent}.localK{offset}:{offset+length}',parent.view_id,(
                                LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),local_start),
                                LogicalCoordinateBinding(1,(LogicalCoordinateTerm(1,1),),offset)))
                        selected.append(view)
                        accesses.append(TileAccessTemplate(f'{prefix}.input_{role.lower()}_{phase}',view.view_id,Operation.READ,
                            (spatial,axis) if update else (spatial,),(spatial,axis) if update else (spatial,None),
                            (ready,),release,evidence,unmapped=True))
                    dependencies = tuple(TileDependency(f'{prefix}.input_{role}_{phase}',EventKind.READ_READY,()) for role in ('a','b'))
                    if update:
                        reload = f'{prefix}.rmw_read'
                        accesses.append(TileAccessTemplate(reload,state.view_id,Operation.READ,(row,column,axis),(row,column),(),
                            (TileDependency(f'{prefix}.partial_update',EventKind.WRITE_VISIBLE,(),carry_axis=axis,
                                initial_access_id=f'{prefix}.partial_first'),),evidence,unmapped=True))
                        dependencies += (TileDependency(reload,EventKind.READ_READY,()),)
                    accesses.append(TileAccessTemplate(f'{prefix}.partial_update' if update else f'{prefix}.partial_first',
                        state.view_id,Operation.WRITE,(row,column,axis) if update else (row,column),(row,column),(),
                        (*dependencies,*release),evidence,unmapped=True))
                if count > ck:
                    last = -(-(count-ck)//ck)-1
                    final = TileDependency(f'{prefix}.partial_update',EventKind.WRITE_VISIBLE,(),(TileIndexBinding(f'k_w{worker}',(),last),))
                else:final = TileDependency(f'{prefix}.partial_first',EventKind.WRITE_VISIBLE,())
                accesses.append(TileAccessTemplate(f'{prefix}.aiv_read',state.view_id,Operation.READ,(row,column),(row,column),(),(final,),evidence,unmapped=True))
                reductions.append(TileDependency(f'{prefix}.aiv_read',EventKind.READ_READY,()))
            parent = views['C']
            output = LogicalSelectionView(f'{scope.scope_id}.g{group}.output',parent.allocation_id,(me,ne),'bf16',2,
                f'cann_deterministic.C.rows{ms}:{ms+me}.cols{ns}:{ns+ne}','C',(
                    LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),ms,divisor=m),
                    LogicalCoordinateBinding(1,(LogicalCoordinateTerm(0,1),),ms,modulus=m),
                    LogicalCoordinateBinding(2,(LogicalCoordinateTerm(1,1),),ns)))
            selected.append(output);previous = f'g{group}.output';previous_axes = (row,column)
            accesses.append(TileAccessTemplate(previous,output.view_id,Operation.WRITE,(row,column),(row,column),(),tuple(reductions),evidence,unmapped=True))
            group += 1
    return tuple(selected),TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),visibility,evidence)
