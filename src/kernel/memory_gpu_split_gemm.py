"""Declared split-K interfaces: external partials or logical in-place accumulation."""
from contextlib import contextmanager
from contextvars import ContextVar,copy_context
from math import prod

from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,
    MemoryAccessProgram,MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,
    TensorView,LogicalSelectionView,Unmapped,unique_by,
    TileAxis,TileAccessTemplate,TileDependency,TileProgram,TilePartition,TileIndexBinding,
)

_WORK = ContextVar('gpu_split_gemm_source_work')


def has_gpu_split_gemm_source(kernel,hardware):
    return hardware.kernel_memory_model == 'gpu_mshr' and has_gpu_split_gemm_abi(kernel)


def has_gpu_split_gemm_abi(kernel):
    return (kernel.kind is KernelKind.GEMM and kernel.variant in ('gemm','batched_gemm')
        and kernel.execution_support == 'supported'
        and kernel.dtype in ('bf16','fp16','fp32')
        and (kernel.variant != 'batched_gemm' or kernel.attributes['weight_dtype'] == 'bf16')
        and kernel.attributes['output_dtype'] in (None,kernel.dtype)
        and (kernel.launch.algorithm in ('nvidia_cublaslt_gemm','cublas_analytical_gemm',
            'metax_mcblas_gemm','mcblas_analytical_gemm','mcblas_injected_gemm:gemm',
            'cublas_analytical_gemm:batched_gemm','nvidia_cublaslt_gemm:batched_gemm')
            or kernel.hardware_id=='metax_c500' and kernel.launch.algorithm=='observed_gemm:gemm')
        and kernel.launch.tiling is not None and kernel.launch.tiling.split_k is not None
        and kernel.launch.tiling.split_k > 1
        and kernel.launch.parameters['split_k_partial_bytes_per_element'] >= 0
        and kernel.launch.parameters['split_k_partial_bytes_per_element']
            == int(kernel.launch.parameters['split_k_partial_bytes_per_element']))


@contextmanager
def gpu_split_gemm_source_work(kernel,program,work):
    token = _WORK.set((kernel,program,work))
    try:
        yield
    finally:
        _WORK.reset(token)


def bound_gpu_split_gemm_source(kernel):
    context = copy_context()
    if _WORK not in context:
        return None
    current,program,work = context[_WORK]
    if current != kernel:
        raise ValueError('split-K source crossed a kernel boundary')
    return program,work


def _partial_layout(dtype,width,splits,elements):
    storage_width = {'bf16':2,'fp16':2,'fp32':4}[dtype]
    if width == storage_width:
        return (splits,elements),dtype,width
    if width == 4:
        return (splits,elements),'fp32',4
    return (splits,elements,width),'uint8',1


def _layouts(kernel):
    b,m,n,k = (kernel.shape[key] for key in ('b','m','n','k'))
    width = kernel.shape['dtype_bytes']
    batched = kernel.variant == 'batched_gemm'
    layouts = [('A',(b,m,k),kernel.dtype,width),
        ('B',(b,k,n) if batched else (k,n),kernel.dtype,width)]
    partial = int(kernel.launch.parameters['split_k_partial_bytes_per_element'])
    if partial:
        shape,dtype,storage_width = _partial_layout(kernel.dtype,partial,kernel.launch.tiling.split_k,b*m*n)
        layouts.append(('partials',shape,dtype,storage_width))
    else:
        layouts.append(('C',(b,m,n),kernel.dtype,width))
    return tuple(layouts)


def _region(view,binding_id):
    if view.view_id in ('A','B'):
        return Unmapped(view.shape,view.dtype,view.element_bytes,f'{binding_id}.{view.view_id}.logical_blas_operand')
    return view.region()


def standalone_gpu_split_gemm_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    allocations,views,events,ready = [],[],[],[]
    for role,shape,dtype,width in _layouts(kernel):
        identity = f'{binding_id}.{role}'
        strides,size = [],width
        for axis in reversed(shape):
            strides.append(size); size *= axis
        allocation = MemoryAllocation(identity,'logical-local-device','logical-storage',
            None if role in ('A','B') else size,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width)
        allocations.append(allocation); views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role in ('A','B'):
            event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,
                _region(view,binding_id),'split-visible',(EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event); ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('A','B'),(views[-1].view_id,),tuple(ready),0,'offline','split-visible')


def build_gpu_split_gemm_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('split-K requires typed operands and actual partial/accumulator bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('split-K exports logical output-tile groups')
    layouts = _layouts(kernel)
    output = layouts[-1][0]
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (('A','B'),(output,),2):
        raise ValueError('split-K operand/output roles mismatch')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    if set(views) != {role for role,_,_,_ in layouts} or len({view.allocation_id for view in views.values()}) != 3:
        raise ValueError('split-K inputs must not alias partial/accumulator output')
    for role,shape,dtype,width in layouts:
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes) != (shape,dtype,width):
            raise ValueError('split-K tensor shape/storage ABI mismatch')
        if role not in ('A','B'):
            strides,size = [],width
            for axis in reversed(shape):
                strides.append(size); size *= axis
            if view.strides_bytes != tuple(reversed(strides)):
                raise ValueError('split-K partial/accumulator output requires its declared dense ABI')
    splits = kernel.launch.tiling.split_k
    geometry = tuple(str(kernel.shape[key]) for key in ('b','m','n','k'))
    mode = 'external' if output == 'partials' else 'in_place'
    tm,tn = (kernel.launch.tiling.tile_shape[axis] for axis in ('m','n'))
    header = ('gpu_split_gemm',kernel.variant,*geometry,str(splits),mode,str(tm),str(tn))
    evidence = ('Retained grid-z K partition and declared partial width select the existing logical external-partial or destination-accumulation account, not a native trace',
        'External gemm_reduce consumes declared [split,elements] partial storage; final C is not written by that main kernel',
        'In-place logical initialization then read-modify-write dependencies do not claim hardware slice execution order',
        'Declared split count may be a retained lower bound; no unobserved split count or native operand layout is fabricated',
        'A logical output group uses the retained M/N blocks and the full-K dot closure across all retained contributions; exact per-split K membership and within-group partial readiness remain unresolved.')
    b,m,n,k = (kernel.shape[axis] for axis in ('b','m','n','k'))
    batched = kernel.variant == 'batched_gemm'
    axes = ((TileAxis('batch',b,1),) if batched else ()) + (
        TileAxis('row',m if batched else b*m,tm),TileAxis('column',n,tn))
    selected = _split_tile_views(views,b,m,n,k,batched,splits,mode,bindings.binding_id,program_id)
    partitions=()
    if mode=='external':
        names=('batch','row','column') if batched else ('row','column')
        axes=(*axes,TileAxis('k',k,1),TileAxis('split',splits,1))
        partitions=(TilePartition('k',None,class_axis_id='split',context_axes=names),)
        evidence += ('Every logical K coordinate belongs to exactly one declared partial class for its output tile; assignment can differ across batch/M/N tiles and remains unbound.',
            'K is a mathematical reduction unit, not an issued MMA tile or an evenly divided native K interval. No implicit K/split rounding or measured duration supplies class membership.')
        accesses=_split_class_accesses(views,selected,batched,bindings.input_ready,evidence)
        agent='logical-split-k-classes'
    else:
        names=('batch','row','column') if batched else ('row','column')
        axes=(*axes,TileAxis('k',k,1),TileAxis('split',splits,1))
        partitions=(TilePartition('k',None,class_axis_id='split',context_axes=names),)
        accesses=_split_tile_accesses(views,selected,batched,splits,mode,bindings.input_ready,evidence)
        agent='logical-split-k-classes'
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,agent)
    family = TileProgram(program_id,scope,axes,accesses,bindings.visibility_domain,evidence,partitions=partitions)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('BLAS operand descriptors do not supply native layouts, addresses or unique physical footprint.',
             'Typed partial/accumulator extents are logical interface capacity, not measured allocator padding.',
             'Logical dependency graph is untimed; vendor issue/atomic/slice order, scratch/semaphores and physical cache replay remain separate.',
             'In-place C expresses the existing logical accumulation algorithm; native internal scratch and epilogue storage aliases remain unbound.',
             'Existing raster/cache/residency/DRAM partial projections are retained; requested bytes are not observed traffic.')),
        bindings.allocations,(*bindings.views,*selected),(),bindings.events,(),(family,))
    gpu_split_gemm_source_demands(program,expected_kernel=kernel)
    return program


def _split_tile_views(views,b,m,n,k,batched,splits,mode,binding_id,program_id):
    selected = []
    if not batched:
        a = views['A']
        selected.append(LogicalSelectionView(f'{program_id}.A.folded_rows',a.allocation_id,(b*m,k),a.dtype,a.element_bytes,
            f'{binding_id}.A.folded_logical_b_m_rows','A'))
    output = views['partials' if mode == 'external' else 'C']
    shape = ((b,m,n) if batched else (b*m,n))
    if mode == 'external':shape = (splits,*shape,*output.shape[2:])
    strides,size = [],output.element_bytes
    for extent in reversed(shape):
        strides.append(size);size *= extent
    selected.append(TensorView(f'{program_id}.output_tiles',output.allocation_id,output.offset_bytes,shape,
        tuple(reversed(strides)),output.dtype,output.element_bytes))
    return tuple(selected)


def _split_tile_accesses(views,selected,batched,splits,mode,input_ready,evidence):
    names = ('batch','row','column') if batched else ('row','column')
    a_ready,b_ready = input_ready
    a = views['A'] if batched else selected[0]
    output = selected[-1]
    reads = (
        TileAccessTemplate('A',a.view_id,Operation.READ,(*names,'k'),('batch','row','k') if batched else ('row','k'),
            (a_ready,),(),evidence,unmapped=True),
        TileAccessTemplate('B','B',Operation.READ,(*names,'k'),('batch','k','column') if batched else ('k','column'),
            (b_ready,),(),evidence,unmapped=True))
    if mode!='in_place':raise ValueError('external partials require explicit K classes')
    accesses = list(reads)
    for step in range(splits):
        dependencies = tuple(TileDependency(access.access_id,EventKind.READ_READY,('k',),
            (TileIndexBinding('split',(),step),),partition_axis='k') for access in reads)
        if step:
            read = f'accumulator_read{step}'
            accesses.append(TileAccessTemplate(read,output.view_id,Operation.READ,names,names,(),
                (TileDependency(f'C_write{step-1}',EventKind.WRITE_VISIBLE,()),),evidence))
            dependencies += (TileDependency(read,EventKind.READ_READY,()),)
        accesses.append(TileAccessTemplate(f'C_write{step}',output.view_id,Operation.WRITE,names,names,(),dependencies,evidence))
    return tuple(accesses)


def _split_class_accesses(views,selected,batched,input_ready,evidence):
    names=('batch','row','column') if batched else ('row','column')
    a=views['A'] if batched else selected[0];output=selected[-1]
    reads=(
        TileAccessTemplate('A',a.view_id,Operation.READ,(*names,'k'),('batch','row','k') if batched else ('row','k'),
            (input_ready[0],),(),evidence,unmapped=True),
        TileAccessTemplate('B','B',Operation.READ,(*names,'k'),('batch','k','column') if batched else ('k','column'),
            (input_ready[1],),(),evidence,unmapped=True))
    required=tuple(TileDependency(access.access_id,EventKind.READ_READY,('k',),partition_axis='k') for access in reads)
    mapping=('split',*names,*((None,)*(len(output.shape)-len(names)-1)))
    return (*reads,TileAccessTemplate('partials_write',output.view_id,Operation.WRITE,('split',*names),mapping,(),required,evidence))


def gpu_split_gemm_tile_demands(program):
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs) != 1:raise ValueError('split-K requires its unique compressed source')
    family, = program.tile_programs
    header = family.scope.op_ids
    if len(header) != 10 or header[0] != 'gpu_split_gemm' or header[1] not in ('gemm','batched_gemm') or family.scope.agent not in ('logical-split-output-groups','logical-split-k-classes'):
        raise ValueError('split-K tile geometry/header mismatch')
    _,variant,b,m,n,k,splits,mode,tm,tn = header
    b,m,n,k,splits,tm,tn = (int(value) for value in (b,m,n,k,splits,tm,tn))
    if min(b,m,n,k,tm,tn) <= 0 or splits <= 1 or mode not in ('external','in_place'):
        raise ValueError('split-K tile geometry must preserve actual outputs and multiple contributions')
    if family.scope.agent!='logical-split-k-classes':
        raise ValueError('split-K requires actual K member classes rather than whole-K partial barriers')
    views = unique_by(program.views,'view_id')
    a,b_view,c = (views[role] for role in ('A','B','partials' if mode == 'external' else 'C'))
    width = {'bf16':2,'fp16':2,'fp32':4}[a.dtype]
    batched = variant == 'batched_gemm'
    if (len({view.allocation_id for view in (a,b_view,c)}) != 3
            or (a.shape,a.element_bytes) != ((b,m,k),width)
            or (b_view.shape,b_view.dtype,b_view.element_bytes) != ((b,k,n) if batched else (k,n),a.dtype,width)):
        raise ValueError('split-K tile operand geometry/storage mismatch')
    partial_width = prod(c.shape[2:])*c.element_bytes if mode == 'external' else 0
    expected = (_partial_layout(a.dtype,partial_width,splits,b*m*n) if mode == 'external'
        else ((b,m,n),a.dtype,width))
    if (c.shape,c.dtype,c.element_bytes) != expected:
        raise ValueError('split-K tile output must preserve every declared partial/accumulator element')
    strides,size = [],c.element_bytes
    for extent in reversed(c.shape):
        strides.append(size);size *= extent
    if c.strides_bytes != tuple(reversed(strides)):
        raise ValueError('split-K tile output must retain its actual dense ABI')
    axes = ((TileAxis('batch',b,1),) if batched else ()) + (TileAxis('row',m if batched else b*m,tm),TileAxis('column',n,tn))
    if mode in ('external','in_place'):
        names=('batch','row','column') if batched else ('row','column')
        axes=(*axes,TileAxis('k',k,1),TileAxis('split',splits,1))
        if (len(family.partitions)!=1 or family.partitions[0] != TilePartition('k',family.partitions[0].members,
                class_axis_id='split',context_axes=names)):
            raise ValueError('split-K requires its exact output-context K partition')
    if family.causal_prefixes:raise ValueError('GEMM K classes cannot omit a causal subset of the reduction')
    selected = _split_tile_views(views,b,m,n,k,batched,splits,mode,program.binding_id,family.tile_program_id)
    if (family.axes != axes or set(views) != {'A','B','partials' if mode == 'external' else 'C'}|{view.view_id for view in selected}
            or any(views[view.view_id] != view for view in selected)):
        raise ValueError('split-K tile views must retain actual output grouping and caller parents')
    if len(family.accesses) < 3 or tuple(access.access_id for access in family.accesses[:2]) != ('A','B'):
        raise ValueError('split-K requires both actual operand reads')
    ready = tuple(access.input_ready[0] for access in family.accesses[:2] if len(access.input_ready) == 1)
    if len(ready) != 2:raise ValueError('split-K requires exact operand availability')
    expected_accesses = (_split_class_accesses(views,selected,batched,ready,family.evidence_refs) if mode=='external'
        else _split_tile_accesses(views,selected,batched,splits,mode,ready,family.evidence_refs))
    if family.accesses != expected_accesses:
        if any(access.access_id.startswith('accumulator_read') and access.dependencies != expected.dependencies
            for access,expected in zip(family.accesses,expected_accesses)):
            raise ValueError('split-K accumulator read requires actual prior output WRITE_VISIBLE')
        raise ValueError('split-K output requires actual operand and accumulator READ_READY for its output group')
    return {'geometry':(b,m,n,k),'operand_geometry':(b,m,n,k,width,width,width),'mode':mode,'split_k':splits,
        'partial_element_bytes':partial_width,'output_bytes':b*m*n*width,
        'operand_read_bytes':prod(a.shape)*a.element_bytes+prod(b_view.shape)*b_view.element_bytes,
        'read_bytes':logical_requested_bytes(program,Operation.READ),'write_bytes':logical_requested_bytes(program,Operation.WRITE)}


def gpu_split_gemm_source_demands(program, *, expected_kernel=None):
    """Current producer only; never reconstruct an old whole-call fallback."""
    work=gpu_split_gemm_tile_demands(program)
    if expected_kernel is not None:
        kernel=expected_kernel
        if not has_gpu_split_gemm_abi(kernel) or kernel.hardware_id!=program.hardware_id:
            raise ValueError('Split-K source differs from its actual invocation')
        header=('gpu_split_gemm',kernel.variant,*(str(kernel.shape[key]) for key in ('b','m','n','k')),
            str(kernel.launch.tiling.split_k),'external' if kernel.launch.parameters['split_k_partial_bytes_per_element'] else 'in_place',
            str(kernel.launch.tiling.tile_shape['m']),str(kernel.launch.tiling.tile_shape['n']))
        views=unique_by(program.views,'view_id')
        if (program.tile_programs[0].scope.op_ids!=header
                or any((views[role].shape,views[role].dtype,views[role].element_bytes)!=(shape,dtype,width)
                    for role,shape,dtype,width in _layouts(kernel))):
            raise ValueError('Split-K source differs from its actual invocation')
    return work
