"""Block-FP8 payload/scale tensor interfaces and wrapper quantizer dataflow."""
from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config.block_fp8_gemm import (
    BLOCK_FP8_GEMM_ALGORITHMS, TRTLLM_SWAP_AB_ALGORITHM, DEEPGEMM_FP8_EINSUM_BATCHED_ALGORITHM,
)
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity, MemoryAccess,
    MemoryAccessProgram, MemoryAllocation, Operation, Ordering, ProgramEvent, Semantics,
    TensorView, region_end, unique_by,TileAxis,TileAccessTemplate,TileDependency,TileProgram,
)


def has_block_fp8_source(kernel):
    return (kernel.kind is KernelKind.GEMM and kernel.execution_support == 'supported'
            and kernel.launch.algorithm in BLOCK_FP8_GEMM_ALGORITHMS)


def block_fp8_layouts(kernel):
    b, m, n, k = (kernel.shape[name] for name in ('b', 'm', 'n', 'k'))
    batched = kernel.launch.algorithm == DEEPGEMM_FP8_EINSUM_BATCHED_ALGORITHM
    if not batched and b != 1:
        raise ValueError('ordinary block-FP8 source requires one matrix problem')
    return block_fp8_geometry_layouts(b,m,n,k,batched_output=batched,
        internal_quantizer=kernel.launch.algorithm == TRTLLM_SWAP_AB_ALGORITHM)


def block_fp8_geometry_layouts(b,m,n,k, *, batched_output,internal_quantizer):
    """One storage recipe shared by source generation and its projection."""
    kb, nb, ma = -(-k//128), -(-n//128), -(-m//4)*4
    # A is group-major even though o_proj exposes its transposed BHR view.
    a = ((b,m,k), (m*k,k,1), 'fp8', 1)
    a_scale = ((b,m,kb), (ma*kb*4,4,ma*4), 'fp32', 4)
    weight = ((b,n,k), (n*k,k,1), 'fp8', 1)
    weight_scale = ((b,nb,kb), (nb*kb*4,kb*4,4), 'fp32', 4)
    output_strides = ((n*2,b*n*2,2) if batched_output else (m*n*2,n*2,2))
    output = ((b,m,n), output_strides, 'bf16', 2)
    if internal_quantizer:
        if b != 1:raise ValueError('internal block-FP8 quantizer requires one matrix problem')
        raw = ((b,m,k), (m*k*2,k*2,2), 'bf16', 2)
        return (('input',raw), ('weight',weight), ('weight_scale',weight_scale),
                ('quantized',a), ('activation_scale',a_scale), ('output',output))
    return (('quantized',a), ('activation_scale',a_scale), ('weight',weight),
            ('weight_scale',weight_scale), ('output',output))


def standalone_block_fp8_bindings(kernel, *, call_id, binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    internal = kernel.launch.algorithm == TRTLLM_SWAP_AB_ALGORITHM
    layouts = block_fp8_layouts(kernel)
    read_count = 3 if internal else 4
    allocations, views, events, ready = [], [], [], []
    for index, (name, (shape, strides, dtype, width)) in enumerate(layouts):
        identity = f'{binding_id}.{name}'
        view = TensorView(name, identity, 0, shape, strides, dtype, width)
        # Required view extent, not CUDA allocator capacity. Alignment holes
        # are deliberately not initialized by a logical input declaration.
        allocation = MemoryAllocation(identity, 'logical-local-device', 'logical-storage', region_end(view.region()),
            AddressLevel.OBJECT, None, None, f'{identity}.allocated', None)
        allocations.append(allocation)
        views.append(view)
        events.append(ProgramEvent(allocation.allocated_event, EventKind.ALLOCATED, identity, None, None, ()))
        if index < read_count:
            initialized = f'{identity}.initialized'
            events.append(ProgramEvent(initialized, EventKind.INITIALIZED, identity, view.region(), 'fp8-visible',
                (EventRef(allocation.allocated_event, EventKind.ALLOCATED),)))
            ready.append(EventRef(initialized, EventKind.INITIALIZED))
    names = tuple(name for name,_ in layouts)
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        names[:read_count],names[read_count:],tuple(ready),0,'offline','fp8-visible')


def build_block_fp8_source(kernel, hardware, bindings, granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings, ElementwiseBindings):
        raise TypeError('block-FP8 source requires payload, scale and output tensor bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('unknown block-FP8 logical granularity')
    internal = kernel.launch.algorithm == TRTLLM_SWAP_AB_ALGORITHM
    layouts = block_fp8_layouts(kernel)
    read_count = 3 if internal else 4
    if (len(bindings.read_views),len(bindings.write_views),len(bindings.input_ready)) != (read_count,len(layouts)-read_count,read_count):
        raise ValueError('block-FP8 payload/scale/output operand count mismatch')
    views = unique_by(bindings.views,'view_id')
    operands = tuple(views[name] for name in (*bindings.read_views,*bindings.write_views))
    roles = {}
    for view,(role,layout) in zip(operands,layouts):
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes) != layout:
            raise ValueError('block-FP8 operand shape/storage/scale layout mismatch')
        roles[role] = view
    if len({view.allocation_id for view in operands}) != len(operands):
        raise NotImplementedError('block-FP8 payload/scale alias needs an explicit implementation lifecycle')
    allocations = unique_by(bindings.allocations,'allocation_id')
    scopes, accesses, events = [], [], list(bindings.events)
    evidence = ('block_fp8_gemm static 128x128 weight and 1x128 activation recipe',
        'FlashInfer scale_1x128_kernel: scales[kblock*align4(M)+row], FP8 row-major payload',
        'vLLM per_token_group_quant_fp8/fused_inv_rope_fp8_quant: FP32 MN-major aligned scales; o_proj BHR/HDR/BHD ABI')

    b,m,n,k = (kernel.shape[name] for name in ('b','m','n','k'))
    # These are algorithmic block-scale units, not an inferred native CTA:
    # each activation row has one scale per K128; weights share N128/K128.
    axes = (TileAxis('batch',b,1),TileAxis('m',m,1),TileAxis('n',n,128),TileAxis('k',k,128))
    evidence += ('Logical block-scale recipe partitions independent row/N128 outputs and K128 reduction panels; native TMA/CTA scheduling remains separate.',)
    accesses = []
    incoming = {role:ready for (role,_),ready in zip(layouts[:read_count],bindings.input_ready,strict=True)}
    quant_axes = ('batch','m','k'); main_axes = ('batch','m','n','k'); output_axes = ('batch','m','n')
    mappings = {'input':(('batch','m','k'),()),'quantized':(('batch','m','k'),()),
        'activation_scale':(('batch','m','k'),(1,1,128)),
        'weight':(('batch','n','k'),()),'weight_scale':(('batch','n','k'),(1,128,128)),
        'output':(('batch','m','n'),())}
    def template(phase,role,operation,iterations,ready=(),dependencies=()):
        mapping,divisors = mappings[role]
        return TileAccessTemplate(f'{phase}.{role}',roles[role].view_id,operation,iterations,mapping,
            ready,dependencies,evidence,view_axis_divisors=divisors)
    if internal:
        accesses.append(template('quantize','input',Operation.READ,quant_axes,(incoming['input'],)))
        for role in ('quantized','activation_scale'):
            accesses.append(template('quantize',role,Operation.WRITE,quant_axes,dependencies=(
                TileDependency('quantize.input',EventKind.READ_READY,()),)))
    for role in ('quantized','activation_scale','weight','weight_scale'):
        produced = internal and role in ('quantized','activation_scale')
        accesses.append(template('gemm',role,Operation.READ,main_axes,
            ready=() if produced else (incoming[role],),
            dependencies=(TileDependency(f'quantize.{role}',EventKind.WRITE_VISIBLE,()),) if produced else ()))
    accesses.append(template('gemm','output',Operation.WRITE,output_axes,dependencies=tuple(
        TileDependency(f'gemm.{role}',EventKind.READ_READY,('k',))
        for role in ('quantized','activation_scale','weight','weight_scale'))))
    scope = AccessScope(program_id,('block_fp8',),bindings.call_id,(),bindings.rank,bindings.stream,'block-scaled-logical-tiles')
    family = TileProgram(program_id,scope,axes,tuple(accesses),bindings.visibility_domain,evidence)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,True,
            ('FP8 e4m3 payload, FP32 block scales and BF16 output retain exact valid storage regions, including scale-alignment holes.',
             'Row/N128/K128 blocks are recipe units; native CTA/TMA shape, physical transactions and cache sharing remain separate.',
             'TRT quantizer writes feed the actual matching payload/scale read tile; no full-call quantization barrier.',
             'Roofline main-stage projection still excludes TRT quantizer service; compulsory bytes are distinct from tile replay demand.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    from src.kernel.memory_path_solver import block_fp8_source_demands
    block_fp8_source_demands(program)
    return program


def has_fp8_group_quant_source(kernel):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant == 'fp8_group_quant'
            and kernel.execution_support == 'supported')


def fused_silu_fp8_group_quant(kernel):
    return kernel.launch.algorithm.startswith(
        "vllm_0_30_sm90_silu_mul_group128_fp8_quant_")


def fp8_group_quant_layouts(kernel):
    # The current native contract supplies a flat group count. It does not
    # bind scale strides or a row/column tensor ABI.
    group_size = kernel.attributes['group_size']
    groups = kernel.shape['element_count'] // group_size
    if fused_silu_fp8_group_quant(kernel):
        return (('input', ((groups, 2 * group_size),
                    (2 * group_size * kernel.shape['dtype_bytes'], kernel.shape['dtype_bytes']),
                    kernel.dtype, kernel.shape['dtype_bytes'])),
                ('quantized', ((groups, group_size), (group_size, 1), 'fp8', 1)),
                ('activation_scale', ((groups,), None, 'fp32', 4)))
    return fp8_group_quant_geometry_layouts(None, kernel.shape['element_count'],
        group_size, kernel.dtype, kernel.shape['dtype_bytes'], None)


def fp8_group_quant_geometry_layouts(rows,columns,group_size,dtype,element_bytes,layout):
    """Declared contiguous payload and FP32 scale ABI, shared with projection."""
    groups = columns//group_size
    if rows is None:
        if layout is not None:raise ValueError('Unknown quantizer rows cannot declare a scale layout')
        return (('input',((groups,group_size),(group_size*element_bytes,element_bytes),dtype,element_bytes)),
                ('quantized',((groups,group_size),(group_size,1),'fp8',1)),
                ('activation_scale',((groups,),None,'fp32',4)))
    if layout == 'row_major':
        scale_strides = (groups*4,4)
    elif layout == 'column_major':
        scale_strides = (4,rows*4)
    elif layout == 'column_major_align4':
        scale_strides = (4,-(-rows//4)*16)
    else:
        raise NotImplementedError('FP8 group quantizer needs its declared scale storage layout')
    return (('input',((rows,columns),(columns*element_bytes,element_bytes),dtype,element_bytes)),
            ('quantized',((rows,columns),(columns,1),'fp8',1)),
            ('activation_scale',((rows,groups),scale_strides,'fp32',4)))


def standalone_fp8_group_quant_bindings(kernel, *, call_id, binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_access import LogicalSelectionView
    allocations, views, events = [], [], []
    for name,(shape,strides,dtype,width) in fp8_group_quant_layouts(kernel):
        identity = f'{binding_id}.{name}'
        view = (LogicalSelectionView(name,identity,shape,dtype,width,f'{identity}.groups',None)
            if strides is None else TensorView(name,identity,0,shape,strides,dtype,width))
        allocation = MemoryAllocation(identity,'logical-local-device','logical-storage',
            None if strides is None else region_end(view.region()),
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation)
        views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if name == 'input':
            initialized = f'{identity}.initialized'
            events.append(ProgramEvent(initialized,EventKind.INITIALIZED,identity,view.region(),'fp8-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),)))
            ready = EventRef(initialized,EventKind.INITIALIZED)
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('input',),('quantized','activation_scale'),(ready,),0,'offline','fp8-visible')


def build_fp8_group_quant_source(kernel, hardware, bindings, granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_access import LogicalSelectionView
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('FP8 group quantizer requires payload and FP32 scale bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('unknown FP8 group logical granularity')
    if (len(bindings.read_views),len(bindings.write_views),len(bindings.input_ready)) != (1,2,1):
        raise ValueError('FP8 group quantizer operand/readiness count mismatch')
    views = unique_by(bindings.views,'view_id')
    operands = tuple(views[name] for name in (*bindings.read_views,*bindings.write_views))
    for view,(_,layout) in zip(operands,fp8_group_quant_layouts(kernel)):
        shape,strides,dtype,width=layout
        if strides is None:
            matches=(isinstance(view,LogicalSelectionView) and view.shape==shape
                and (view.dtype,view.element_bytes)==(dtype,width) and view.parent_view_id is None
                and not view.coordinate_map)
        else:matches=(isinstance(view,TensorView) and (view.shape,view.strides_bytes,view.dtype,view.element_bytes)==layout)
        if not matches:
            raise ValueError('FP8 group quantizer operand storage mismatch')
    if len({view.allocation_id for view in operands}) != 3:
        raise NotImplementedError('FP8 quantizer input/payload/scale overlap has no declared in-place algorithm')
    known_rows = False
    operation = 'fused_silu_fp8_group_quant' if fused_silu_fp8_group_quant(kernel) else 'fp8_group_quant'
    scope = AccessScope(f'{program_id}.scope',(operation,'unknown_rows'),bindings.call_id,(),
        bindings.rank,bindings.stream,'per-token-group-quantizer')
    evidence = ('vLLM per_token_group_quant_fp8: contiguous input/output payload and explicit FP32 scale strides',
                'per_token_group_quant_8bit: per-group absmax, scaling and e4m3 conversion')
    group_size = kernel.attributes['group_size']
    tiled_views = []
    if known_rows:
        rows,columns=operands[0].shape;groups=columns//group_size
        for view in operands[:2]:
            tiled_views.append(TensorView(f'{view.view_id}.groups',view.allocation_id,view.offset_bytes,
                (rows,groups,group_size),(view.strides_bytes[0],group_size*view.strides_bytes[1],view.strides_bytes[1]),
                view.dtype,view.element_bytes))
        axes=(TileAxis('row',rows,1),TileAxis('group',groups,1))
        iterations=('row','group');payload_mapping=('row','group',None);scale_mapping=('row','group')
    else:
        groups=kernel.shape['element_count']//group_size
        tiled_views.extend(operands[:2]);axes=(TileAxis('group',groups,1),)
        iterations=('group',);payload_mapping=('group',None);scale_mapping=('group',)
    evidence += ('Each row/group absmax reduction consumes exactly group_size input elements and publishes its FP8 group and one FP32 scale.',
                 'Semantic quantization groups are independent; source ABI has whole groups, without invented padded columns or native issue order.')
    read = TileAccessTemplate('input',tiled_views[0].view_id,Operation.READ,iterations,payload_mapping,
        bindings.input_ready,(),evidence)
    dependencies = (TileDependency('input',EventKind.READ_READY,()),)
    payload = TileAccessTemplate('payload',tiled_views[1].view_id,Operation.WRITE,iterations,payload_mapping,
        (),dependencies,evidence)
    scale = TileAccessTemplate('scale',operands[2].view_id,Operation.WRITE,iterations,scale_mapping,
        (),dependencies,evidence,unmapped=not known_rows)
    family = TileProgram(program_id,scope,axes,
        (read,payload,scale),bindings.visibility_domain,evidence)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,known_rows,
            ('FP32 scales only; UE8M0 packed scales require a separate declared ABI.',
             'Logical scale writes omit alignment holes; allocation capacity and issued/cache traffic are not inferred.',
             'Whole contiguous groups remain known without row facts; scale selections then have unknown storage addresses and capacity.')),
        bindings.allocations,(*bindings.views,*tiled_views) if known_rows else bindings.views,(),bindings.events,(),(family,))
    from src.kernel.memory_path_solver import fp8_group_quant_source_demands
    fp8_group_quant_source_demands(program)
    return program
