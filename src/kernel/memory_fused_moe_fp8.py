"""FP8 grouped projection operands, separate from the caller's quantizer."""
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity,
    MemoryAccess, MemoryAccessProgram, MemoryAllocation, Operation, Ordering,
    ProgramEvent, Semantics, TensorView, Unmapped, unique_by,
    LogicalSelectionView,LogicalCoordinateBinding,LogicalCoordinateTerm,
    TileAxis,TileAccessTemplate,TileDependency,TileProgram,
)
from src.kernel.memory_fused_moe import fused_moe_projection, fused_moe_routing


def has_fp8_fused_moe_source(kernel):
    return (kernel.kind is KernelKind.FUSED_MOE and kernel.variant in ('w1','w2')
            and kernel.execution_support == 'supported' and kernel.attributes['expert_parallel_size'] == 1
            and kernel.launch.algorithm.startswith(('vllm_triton_fused_moe_fp8_block128_',
                                                    'sglang_triton_fused_moe_fp8_block128_')))


def fp8_fused_kernel_counts(kernel):
    _,counts = fused_moe_routing(kernel)
    if kernel.launch.algorithm.startswith('sglang_'):
        if 'fused_shared_experts' not in kernel.attributes or kernel.attributes['fused_shared_experts'] != 1:
            raise ValueError('SGLang supported shared-expert source requires its explicit one-expert fact')
        return (*counts,kernel.shape['tokens']),kernel.shape['topk']+1
    return counts,kernel.shape['topk']


def fp8_fused_tma(kernel):
    return kernel.launch.algorithm == f'sglang_triton_fused_moe_fp8_block128_aligned_tma_{kernel.variant}'


def fp8_fused_naive(kernel):
    return kernel.launch.algorithm == f'vllm_triton_fused_moe_fp8_block128_naive_{kernel.variant}'


def fp8_fused_aligned(kernel):
    return kernel.launch.algorithm == f'vllm_triton_fused_moe_fp8_block128_aligned_{kernel.variant}'


def fp8_fused_sglang_aligned(kernel):
    return kernel.launch.algorithm in ('sglang_triton_fused_moe_fp8_block128_aligned_w1',
        'sglang_triton_fused_moe_fp8_block128_aligned_tma_w2')


def fp8_fused_quant_rows(kernel):
    if kernel.variant == 'w1':
        return kernel.shape['tokens']
    counts,topk = fp8_fused_kernel_counts(kernel)
    rows = kernel.shape['tokens']*topk
    if fp8_fused_tma(kernel):
        # SGLang allocates/quantizes total_tokens, not just valid routes.
        # The launch records sorted-token workspace length from the UP
        # config; DOWN may have a different BLOCK_SIZE_M.
        rows = int(kernel.launch.parameters['padded_rows'])
    return rows


def fp8_fused_layouts(kernel):
    variant,rows,reduction,columns = fused_moe_projection(kernel)
    tokens = kernel.shape['tokens']
    counts,topk = fp8_fused_kernel_counts(kernel)
    experts,rows = len(counts),fp8_fused_quant_rows(kernel)
    layouts=(('query',(rows,reduction),'fp8',1),('query_scale',(rows,reduction//128),'fp32',4),
            ('weight',(experts,columns,reduction),'fp8',1),
            ('weight_scale',(experts,columns//128,reduction//128),'fp32',4),
            ('routing',(tokens*topk,),'int32',4),
            ('router_weight',(tokens,topk),'fp32',4),
            ('output',(tokens,topk,columns),kernel.dtype,kernel.shape['dtype_bytes']))
    if fp8_fused_naive(kernel):
        return (*layouts[:-1],('num_tokens_post_padded',(1,),'int32',4),layouts[-1])
    if fp8_fused_aligned(kernel) or fp8_fused_sglang_aligned(kernel):
        capacity=int(kernel.launch.parameters['padded_rows'])
        bm=kernel.launch.tiling.tile_shape['m']
        return (*layouts[:4],layouts[5],('sorted_token_ids',(capacity,),'int32',4),
            ('expert_ids',(-(-capacity//bm),),'int32',4),
            ('num_tokens_post_padded',(1,),'int32',4),layouts[-1])
    return layouts


def standalone_fp8_fused_bindings(kernel, *, call_id, binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    allocations,views,events,ready = [],[],[],[]
    roles = []
    for role,shape,dtype,width in fp8_fused_layouts(kernel):
        identity = f'{binding_id}.{role}'
        strides,size = [],width
        for dimension in reversed(shape):
            strides.append(size)
            size *= dimension
        storage = 'logical-storage'
        if role == 'output' and kernel.launch.algorithm.startswith('sglang_') and kernel.variant == 'w1':
            # Valid-output descriptor does not assert sorted row addresses.
            # The wrapper actually allocates the padded up-config workspace.
            allocation_rows = int(kernel.launch.parameters['padded_rows'])
            size = allocation_rows*shape[-1]*width
            storage = 'logical-valid-output-descriptor'
        view = TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width)
        allocation = MemoryAllocation(identity,'logical-local-device',storage,size,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation)
        views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role != 'output':
            event = ProgramEvent(f'{identity}.initialized',EventKind.INITIALIZED,identity,view.region(),'moe-fp8-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event)
            ready.append(EventRef(event.event_id,event.kind))
            roles.append(role)
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        tuple(roles),('output',),tuple(ready),0,'offline','moe-fp8-visible')


def build_fp8_fused_source(kernel, hardware, bindings, granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('FP8 grouped projection requires payload/scale/routing/output bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('FP8 grouped source exports logical tile work only')
    layouts = fp8_fused_layouts(kernel)
    inputs=len(layouts)-1
    if (len(bindings.read_views),len(bindings.input_ready),len(bindings.write_views)) != (inputs,inputs,1):
        raise ValueError('FP8 grouped projection operand/readiness count mismatch')
    views = unique_by(bindings.views,'view_id')
    operands = {role:views[name] for (role,*_),name in zip(layouts,(*bindings.read_views,*bindings.write_views),strict=True)}
    for role,shape,dtype,width in layouts:
        view = operands[role]
        strides,size = [],width
        for dimension in reversed(shape):
            strides.append(size)
            size *= dimension
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes) != (shape,tuple(reversed(strides)),dtype,width):
            raise ValueError('FP8 grouped projection operand shape/storage mismatch')
    if len({view.allocation_id for view in operands.values()}) != len(operands):
        raise NotImplementedError('FP8 payload/scale/route/output overlap has no native lifecycle binding')
    readiness = dict(zip(tuple(role for role,*_ in layouts[:-1]),bindings.input_ready,strict=True))
    events_by_id = unique_by(bindings.events,'event_id')
    selected_parents = ('query','query_scale','router_weight') if kernel.variant == 'w2' else ('query','query_scale')
    for role in selected_parents:
        view,ref = operands[role],readiness[role]
        if ref.kind not in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE):
            raise ValueError('FP8 selected payload/scale requires complete parent readiness')
        event = events_by_id[ref.node_id]
        if event.allocation_id != view.allocation_id or event.region != view.region():
            raise ValueError('FP8 selected payload/scale requires complete parent readiness')
    if fp8_fused_aligned(kernel) or fp8_fused_sglang_aligned(kernel):
        return build_fp8_aligned_tiles(kernel,hardware,bindings,operands,readiness,program_id=program_id)
    return build_fp8_grouped_tiles(kernel,hardware,bindings,operands,views,readiness,program_id=program_id)


def _fp8_aligned_recipe(scope,parents,ready,visibility):
    """Native M/N controls with masked payloads and complete-K dot blocks."""
    if scope.op_ids[0]=='fused_moe_fp8_sglang_aligned':
        return _fp8_sglang_aligned_recipe(scope,parents,ready,visibility)
    from dataclasses import replace
    header=scope.op_ids
    variant,bm,bn,capacity=header[1],int(header[2]),int(header[3]),int(header[4])
    counts=tuple(int(value) for value in header[5].split(','))
    columns,reduction=parents['weight'].shape[1:]
    if bm<=0 or bn!=128 or capacity<=0 or any(count<0 for count in counts):
        raise ValueError('FP8 aligned source requires actual BM/native N128 and nonnegative routes')
    evidence=('vLLM0.24 fused_moe.py:396-415: every native M/N program loads padded-count before its independent sorted-ID and expert-ID reads.',
        'Sorted IDs include the complete BM padded slots; input/scales/output retain valid-route masks. Expert placement is a logical coordinate, not a reconstructed token address.',
        'Full-K matrix and scale reads precede W2 router-weight multiplication and its own output store. No scratch is materialized for register accumulation.')
    selected=[];families=[];moffset=0
    dep=lambda role:TileDependency(role,EventKind.READ_READY,())
    for expert,count in enumerate(counts):
        if not count:continue
        identity=f'{scope.scope_id}.e{expert}'
        local=replace(scope,scope_id=identity,agent='fp8-aligned-native-blocks',
            op_ids=(*header,str(expert)))
        slices={}
        shapes=(('sorted_token_ids',(-(-count//bm)*bm,)),('expert_ids',(-(-count//bm),)),
            ('query',(count,reduction)),('query_scale',(count,reduction//128)),('output',(count,columns)))
        if variant=='w2':shapes+=(('router_weight',(count,)),)
        for role,shape in shapes:
            parent=parents[role]
            coordinates=()
            if role in ('sorted_token_ids','expert_ids'):
                coordinates=(LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),moffset*(bm if role=='sorted_token_ids' else 1)),)
            child=LogicalSelectionView(f'{identity}.{role}',parent.allocation_id,shape,parent.dtype,
                parent.element_bytes,f'{identity}.{role}.valid_routes',parent.view_id,coordinates)
            selected.append(child);slices[role]=child
        for role,shape in (('weight',(columns,reduction)),('weight_scale',(columns//128,reduction//128))):
            parent=parents[role];size=shape[0]*shape[1]*parent.element_bytes
            child=TensorView(f'{identity}.{role}',parent.allocation_id,parent.offset_bytes+expert*size,
                shape,parent.strides_bytes[1:],parent.dtype,parent.element_bytes)
            selected.append(child);slices[role]=child
        actor=('row','column')
        accesses=[TileAccessTemplate('num_tokens_post_padded',parents['num_tokens_post_padded'].view_id,
            Operation.READ,actor,(None,),(ready['num_tokens_post_padded'],),(),evidence)]
        for role in ('sorted_token_ids','expert_ids'):
            accesses.append(TileAccessTemplate(role,slices[role].view_id,Operation.READ,actor,('row',),
                (ready[role],),(dep('num_tokens_post_padded'),),evidence,unmapped=True,
                view_axis_divisors=(bm,) if role=='expert_ids' else (),
                view_axis_padding=(True,) if role=='sorted_token_ids' else ()))
        routed=(dep('sorted_token_ids'),dep('expert_ids'))
        for role,mapping in (('query',('row',None)),('query_scale',('row',None)),
                ('weight',('column',None)),('weight_scale',('column',None))):
            accesses.append(TileAccessTemplate(role,slices[role].view_id,Operation.READ,actor,mapping,
                (ready[role],),routed,evidence,unmapped=role in ('query','query_scale'),
                view_axis_divisors=(128,1) if role=='weight_scale' else ()))
        matrix=tuple(dep(role) for role in ('query','query_scale','weight','weight_scale'))
        output_deps=matrix
        if variant=='w2':
            accesses.append(TileAccessTemplate('router_weight',slices['router_weight'].view_id,
                Operation.READ,actor,('row',),(ready['router_weight'],),matrix,evidence,unmapped=True))
            output_deps+=(dep('router_weight'),)
        accesses.append(TileAccessTemplate('output',slices['output'].view_id,Operation.WRITE,actor,
            ('row','column'),(),output_deps,evidence,unmapped=True))
        families.append(TileProgram(identity,local,(TileAxis('row',count,bm),TileAxis('column',columns,bn)),
            tuple(accesses),visibility,evidence))
        moffset+=-(-count//bm)
    inactive=-(-capacity//bm)-moffset
    if inactive<0:raise ValueError('FP8 aligned routes exceed the native grid')
    if inactive:
        identity=f'{scope.scope_id}.inactive'
        local=replace(scope,scope_id=identity,agent='fp8-aligned-count-early-return',op_ids=(*header,'inactive'))
        families.append(TileProgram(identity,local,(TileAxis('row',inactive,1),TileAxis('column',columns,bn)),
            (TileAccessTemplate('num_tokens_post_padded',parents['num_tokens_post_padded'].view_id,
                Operation.READ,('row','column'),(None,),(ready['num_tokens_post_padded'],),(),evidence),),visibility,evidence))
    return tuple(selected),tuple(families)


def _fp8_sglang_aligned_recipe(scope,parents,ready,visibility):
    """Native N programs retain scale membership even when BN is below 128."""
    from dataclasses import replace
    header=scope.op_ids;variant=header[1]
    bm,bn,capacity=(int(value) for value in header[2:5])
    counts=tuple(int(value) for value in header[5].split(','));tma=header[-1]=='1'
    columns,reduction=parents['weight'].shape[1:]
    if bm<=0 or bn not in (64,128,256) or columns%bn or capacity<=0 or any(count<0 for count in counts):
        raise ValueError('SGLang aligned source requires its actual positive native BM/N programs')
    nprograms=columns//bn
    evidence=('SGLang0.5.12 fused_moe_triton_kernels.py:431-440: each native M/N program loads padded-count, then independently loads full BM sorted IDs and one expert ID.',
        'BN64 programs independently reload the shared N128 FP32 scale group; BN256 programs consume two groups. Logical scale membership is explicit and never treated as a physical address.',
        'TMA descriptor loads the complete BM A box; activation scales, post-K router weights and output stores retain valid-token masks. Quantizer and sorted intermediate allocation are distinct caller facts.')
    selected=[];families=[];moffset=0
    dep=lambda role:TileDependency(role,EventKind.READ_READY,())
    for expert,count in enumerate(counts):
        if not count:continue
        identity=f'{scope.scope_id}.e{expert}'
        local=replace(scope,scope_id=identity,agent='sglang-fp8-native-blocks',op_ids=(*header,str(expert)))
        payload_rows=-(-count//bm)*bm if tma else count
        shapes=(('sorted_token_ids',(-(-count//bm)*bm,)),('expert_ids',(-(-count//bm),)),
            ('query',(payload_rows,reduction)),('query_scale',(count,reduction//128)),
            ('output',(count,nprograms,bn)))
        if variant=='w2':shapes+=(('router_weight',(count,)),)
        slices={}
        for role,shape in shapes:
            parent=parents[role];coordinates=()
            if role in ('sorted_token_ids','expert_ids'):
                coordinates=(LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),moffset*(bm if role=='sorted_token_ids' else 1)),)
            if role=='query' and tma:
                coordinates=(LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),moffset*bm),
                    LogicalCoordinateBinding(1,(LogicalCoordinateTerm(1,1),)))
            child=LogicalSelectionView(f'{identity}.{role}',parent.allocation_id,shape,parent.dtype,
                parent.element_bytes,f'{identity}.{role}.selected',parent.view_id,coordinates)
            selected.append(child);slices[role]=child
        weight=parents['weight']
        child=TensorView(f'{identity}.weight',weight.allocation_id,weight.offset_bytes+expert*columns*reduction,
            (nprograms,bn,reduction),(bn*reduction,reduction,1),weight.dtype,weight.element_bytes)
        selected.append(child);slices['weight']=child
        scale=parents['weight_scale']
        child=LogicalSelectionView(f'{identity}.weight_scale',scale.allocation_id,
            (nprograms,max(1,bn//128),reduction//128),scale.dtype,scale.element_bytes,
            f'{identity}.native_N128_scale_membership',scale.view_id,(
                LogicalCoordinateBinding(0,(),expert),
                LogicalCoordinateBinding(1,(LogicalCoordinateTerm(0,bn),LogicalCoordinateTerm(1,128)),divisor=128),
                LogicalCoordinateBinding(2,(LogicalCoordinateTerm(2,1),))))
        selected.append(child);slices['weight_scale']=child
        actor=('row','column')
        accesses=[TileAccessTemplate('num_tokens_post_padded',parents['num_tokens_post_padded'].view_id,
            Operation.READ,actor,(None,),(ready['num_tokens_post_padded'],),(),evidence)]
        for role in ('sorted_token_ids','expert_ids'):
            accesses.append(TileAccessTemplate(role,slices[role].view_id,Operation.READ,actor,('row',),
                (ready[role],),(dep('num_tokens_post_padded'),),evidence,unmapped=True,
                view_axis_divisors=(bm,) if role=='expert_ids' else (),
                view_axis_padding=(True,) if role=='sorted_token_ids' else ()))
        routed=(dep('sorted_token_ids'),dep('expert_ids'))
        for role,mapping in (('query',('row',None)),('query_scale',('row',None)),
                ('weight',('column',None,None)),('weight_scale',('column',None,None))):
            accesses.append(TileAccessTemplate(role,slices[role].view_id,Operation.READ,actor,mapping,
                (ready[role],),routed,evidence,unmapped=role!='weight',
                view_axis_padding=(True,False) if role=='query' and tma else ()))
        matrix=tuple(dep(role) for role in ('query','query_scale','weight','weight_scale'));output_deps=matrix
        if variant=='w2':
            accesses.append(TileAccessTemplate('router_weight',slices['router_weight'].view_id,
                Operation.READ,actor,('row',),(ready['router_weight'],),matrix,evidence,unmapped=True))
            output_deps+=(dep('router_weight'),)
        accesses.append(TileAccessTemplate('output',slices['output'].view_id,Operation.WRITE,actor,
            ('row','column',None),(),output_deps,evidence,unmapped=True))
        families.append(TileProgram(identity,local,(TileAxis('row',count,bm),TileAxis('column',nprograms,1)),
            tuple(accesses),visibility,evidence))
        moffset+=-(-count//bm)
    inactive=-(-capacity//bm)-moffset
    if inactive<0:raise ValueError('SGLang native padded routes exceed the actual grid')
    if inactive:
        identity=f'{scope.scope_id}.inactive'
        local=replace(scope,scope_id=identity,agent='sglang-fp8-count-early-return',op_ids=(*header,'inactive'))
        families.append(TileProgram(identity,local,(TileAxis('row',inactive,1),TileAxis('column',nprograms,1)),
            (TileAccessTemplate('num_tokens_post_padded',parents['num_tokens_post_padded'].view_id,
                Operation.READ,('row','column'),(None,),(ready['num_tokens_post_padded'],),(),evidence),),visibility,evidence))
    return tuple(selected),tuple(families)


def build_fp8_aligned_tiles(kernel,hardware,bindings,parents,ready, *, program_id):
    counts,_=fp8_fused_kernel_counts(kernel)
    bm,bn=(kernel.launch.tiling.tile_shape[name] for name in ('m','n'))
    header=('fused_moe_fp8_aligned',kernel.variant,str(bm),str(bn),
        str(int(kernel.launch.parameters['padded_rows'])),','.join(str(count) for count in counts),
        *(parents[role].view_id for role,*_ in fp8_fused_layouts(kernel)))
    if fp8_fused_sglang_aligned(kernel):
        header=('fused_moe_fp8_sglang_aligned',*header[1:],str(int(fp8_fused_tma(kernel))))
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'fp8-aligned-interface')
    selected,families=_fp8_aligned_recipe(scope,parents,ready,bindings.visibility_domain)
    partition=('Native BM/N64-N256 programs retain explicit N128 scale-group membership, padded sorted IDs, expert IDs and count-only inactive programs.'
        if fp8_fused_sglang_aligned(kernel) else 'Native BM/N128 control/payload blocks include padded sorted IDs, expert IDs and count-only inactive programs.')
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            partition,
            'Quantizer is a separate producer. Runtime sorted values and physical token addresses are unknown; independent tile programs have no imposed total order.',
            'Public callee ABI; alignment producer and full-model parameter lifetime require caller evidence. Logical requests are separate from legacy DRAM/cache prices.')),
        bindings.allocations,(*parents.values(),*selected),(scope,),bindings.events,(),families)
    fp8_fused_source_demands(program,expected_kernel=kernel)
    return program


def fp8_aligned_tile_demands(program, *, expected_kernel=None):
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    scope,=program.scopes;header=scope.op_ids
    shared=header[0]=='fused_moe_fp8_sglang_aligned'
    if (len(header)!=(16 if shared else 15) or header[0] not in ('fused_moe_fp8_aligned','fused_moe_fp8_sglang_aligned')
            or header[1] not in ('w1','w2') or shared and header[-1] not in ('0','1')):
        raise ValueError('FP8 aligned source requires the complete native control interface')
    tma=shared and header[-1]=='1'
    variant=header[1];bm,bn,capacity=(int(value) for value in header[2:5])
    counts=tuple(int(value) for value in header[5].split(','))
    if bm<=0 or bn not in ((64,128,256) if shared else (128,)) or capacity<=0 or any(count<0 for count in counts):
        raise ValueError('FP8 aligned source requires actual BM/native N128 and nonnegative routes')
    roles=('query','query_scale','weight','weight_scale','router_weight','sorted_token_ids','expert_ids','num_tokens_post_padded','output')
    views=unique_by(program.views,'view_id');parents=dict(zip(roles,(views[name] for name in header[6:15]),strict=True))
    experts,columns,reduction=parents['weight'].shape
    tokens,topk,n=parents['output'].shape;rows=capacity if tma else tokens if variant=='w1' else tokens*topk
    layouts=(((rows,reduction),'fp8',1),((rows,reduction//128),'fp32',4),
        ((experts,columns,reduction),'fp8',1),((experts,columns//128,reduction//128),'fp32',4),
        ((tokens,topk),'fp32',4),((capacity,),'int32',4),((-(-capacity//bm),),'int32',4),((1,),'int32',4),((tokens,topk,columns),'bf16',2))
    if n!=columns or len(counts)!=experts or sum(counts)!=tokens*topk or reduction%128 or columns%128:
        raise ValueError('FP8 aligned route and scale geometry differs')
    if shared and (experts!=257 or topk!=9 or counts[-1]!=tokens or tma!=(variant=='w2')):
        raise ValueError('SGLang shared expert and TMA projection facts differ')
    if len({parent.allocation_id for parent in parents.values()})!=len(parents):
        raise ValueError('FP8 aligned operands must retain distinct parents')
    for parent,(shape,dtype,width) in zip(parents.values(),layouts,strict=True):
        strides=[];size=width
        for extent in reversed(shape):strides.append(size);size*=extent
        if (parent.shape,parent.dtype,parent.element_bytes,parent.strides_bytes)!=(shape,dtype,width,tuple(reversed(strides))):
            raise ValueError('FP8 aligned typed parent layout differs')
    ready={}
    for family in program.tile_programs:
        for access in family.accesses:
            if access.operation is Operation.READ:
                if len(access.input_ready)!=1 or access.access_id in ready and ready[access.access_id]!=access.input_ready[0]:
                    raise ValueError('FP8 aligned actual parent readiness differs')
                ready[access.access_id]=access.input_ready[0]
    required=set(roles)-{'output'}-({'router_weight'} if variant=='w1' else set())
    if set(ready)!=required:raise ValueError('FP8 aligned source lost an actual control/payload read')
    if not program.tile_programs:raise ValueError('FP8 aligned source requires native tile programs')
    selected,families=_fp8_aligned_recipe(scope,parents,ready,program.tile_programs[0].visibility_domain)
    if (program.tile_programs!=families or set(views)!={*(parent.view_id for parent in parents.values()),*(view.view_id for view in selected)}
            or any(views[view.view_id]!=view for view in selected)):
        raise ValueError('FP8 aligned source lost native metadata, masks or READ_READY causality')
    events=unique_by(program.events,'event_id')
    for role,ref in ready.items():
        event=events[ref.node_id];parent=parents[role]
        if (ref.kind not in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE)
                or event.allocation_id!=parent.allocation_id or event.region!=parent.region()):
            raise ValueError('FP8 aligned source requires complete actual parent readiness')
    if expected_kernel is not None:
        expected_counts,_=fp8_fused_kernel_counts(expected_kernel)
        expected_bm,expected_bn=(expected_kernel.launch.tiling.tile_shape[name] for name in ('m','n'))
        expected_variant,_,expected_k,expected_n=fused_moe_projection(expected_kernel)
        if (not (fp8_fused_sglang_aligned(expected_kernel) if shared else fp8_fused_aligned(expected_kernel))
                or program.hardware_id!=expected_kernel.hardware_id
                or (variant,rows,reduction,columns)!=(expected_variant,fp8_fused_quant_rows(expected_kernel),expected_k,expected_n)
                or counts!=expected_counts or (bm,bn)!=(expected_bm,expected_bn)
                or capacity!=int(expected_kernel.launch.parameters['padded_rows'])
                or -(-capacity//bm)*-(-columns//bn)!=expected_kernel.launch.grid[0]):
            raise ValueError('FP8 aligned source differs from its actual native kernel')
        if shared:
            # The source's sorted array is allocated with the UP BM. Its
            # strictly increasing extent in BM proves the supplied native
            # BM matches that allocation; a future unequal DOWN BM fails.
            routed=tokens*topk
            expected_capacity=routed*bm if routed<experts else routed+(experts+1)*(bm-1)
            if capacity!=expected_capacity:raise ValueError('SGLang native BM differs from the UP alignment allocation')
    routes=sum(counts);active=sum(count>0 for count in counts)
    payload_rows=sum(-(-count//bm)*bm for count in counts) if tma else routes
    legacy=(routes*4+payload_rows*reduction+routes*(reduction//128)*4+active*columns*reduction
        +active*(columns//128)*(reduction//128)*4+(routes*4 if variant=='w2' else 0))
    base_counts=counts[:-1] if shared else counts
    return {'projection':(variant,rows,reduction,columns),'routing':(base_counts,base_counts),
        'read_bytes':legacy,'legacy_read_price_bytes':legacy,
        'logical_read_requested_bytes':logical_requested_bytes(program,Operation.READ),
        'write_bytes':logical_requested_bytes(program,Operation.WRITE)}


def build_fp8_grouped_tiles(kernel,hardware,bindings,operands,views,readiness, *, program_id):
    _,_,reduction,columns = fused_moe_projection(kernel)
    counts,_ = fp8_fused_kernel_counts(kernel)
    tma = fp8_fused_tma(kernel)
    shared = kernel.launch.algorithm.startswith('sglang_')
    naive=fp8_fused_naive(kernel)
    row_block = 1 if naive else kernel.launch.tiling.tile_shape['m']
    evidence = ('vLLM/SGLang FP8 grouped source: actual launch BM row groups and 128-column scale-recipe blocks consume complete K dots',
        'FP8 payload and FP32 per128 activation/weight scales share actual typed parents; caller quantizer is a distinct producer',
        'TMA A boxes include full BM rows while activation scales, router weights and outputs retain valid-token masks',
        'Expert route ordinals are logical selections; actual sorted indices, physical offsets and allocator footprint remain unknown')
    if naive:evidence+=('vLLM0.24 fused_moe.py naive assignment: each CTA has one valid route, reads the INT32 post-padding scalar before routing, and repeats both metadata reads for each N tile.',)
    families = []
    for expert,count in enumerate(counts):
        if not count:
            continue
        identity = f'{program_id}.{expert}'
        payload_rows = -(-count//row_block)*row_block if tma else count
        roles = (('routing',(count,)),('query',(payload_rows,reduction)),('query_scale',(count,reduction//128)),
                 ('output',(count,columns)))
        if kernel.variant == 'w2':
            roles += (('router_weight',(count,)),)
        selected = {}
        for role,shape in roles:
            parent = operands[role]
            view = LogicalSelectionView(f'{identity}.{role}_selection',parent.allocation_id,shape,
                parent.dtype,parent.element_bytes,f'{bindings.binding_id}.{kernel.variant}.expert{expert}.{role}.valid_route_ordinal',parent.view_id)
            views[view.view_id] = view
            selected[role] = view
        for role,shape in (('weight',(columns,reduction)),('weight_scale',(columns//128,reduction//128))):
            parent = operands[role]
            size = shape[0]*shape[1]*parent.element_bytes
            view = TensorView(f'{identity}.{role}',parent.allocation_id,parent.offset_bytes+expert*size,
                shape,parent.strides_bytes[1:],parent.dtype,parent.element_bytes)
            views[view.view_id] = view
            selected[role] = view
        accesses=[]
        if naive:
            accesses.append(TileAccessTemplate('num_tokens_post_padded',operands['num_tokens_post_padded'].view_id,Operation.READ,
                ('row','column'),(None,),(readiness['num_tokens_post_padded'],),(),evidence))
        routing_dependencies=(TileDependency('num_tokens_post_padded',EventKind.READ_READY,()),) if naive else ()
        accesses.append(TileAccessTemplate('routing',selected['routing'].view_id,Operation.READ,('row','column') if naive else ('row',),('row',),
            (readiness['routing'],),routing_dependencies,evidence,unmapped=True))
        routed = (TileDependency('routing',EventKind.READ_READY,()),)
        for role,mapping in (('query',('row',None)),('query_scale',('row',None)),
                ('weight',('column',None)),('weight_scale',('column',None))):
            divisors = (128,1) if role == 'weight_scale' else ()
            padding = (True,False) if role == 'query' and tma else ()
            accesses.append(TileAccessTemplate(role,selected[role].view_id,Operation.READ,('row','column'),mapping,
                (readiness[role],),routed,evidence,unmapped=role in ('query','query_scale'),
                view_axis_divisors=divisors,view_axis_padding=padding))
        if kernel.variant == 'w2':
            accesses.append(TileAccessTemplate('router_weight',selected['router_weight'].view_id,Operation.READ,('row','column') if naive else ('row',),('row',),
                (readiness['router_weight'],),routed,evidence,unmapped=True))
        required = tuple(TileDependency(access.access_id,EventKind.READ_READY,()) for access in accesses if access.access_id != 'routing')
        accesses.append(TileAccessTemplate('output',selected['output'].view_id,Operation.WRITE,('row','column'),('row','column'),
            (),required,evidence,unmapped=True))
        header = ('fused_moe_fp8_tiles',kernel.variant,str(expert),str(int(shared)),str(int(tma)),str(row_block),
            *(operands[role].view_id for role,*_ in fp8_fused_layouts(kernel)))
        scope = AccessScope(identity,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-fp8-routed-blocks')
        families.append(TileProgram(identity,scope,(TileAxis('row',count,row_block),TileAxis('column',columns,128)),
            tuple(accesses),bindings.visibility_domain,evidence))
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Compressed BM route groups and scale-recipe N128 blocks; full-K local accumulation has no fabricated GM scratch.',
            'TMA payload padding and valid scale/output masks stay distinct; sorted-token offsets, metadata side effects and physical footprint remain unknown.',
            'Separate caller quantizers and existing naive/aligned/TMA cache/traffic projections are preserved; logical replay is not observed DRAM.',
            'Native metadata, compiler local storage, EP and complete wrapper lifetimes remain partial.')),
        bindings.allocations,tuple(views.values()),(),bindings.events,(),tuple(families))
    fp8_fused_source_demands(program,expected_kernel=kernel)
    return program


def fp8_grouped_tile_demands(program):
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    views = unique_by(program.views,'view_id')
    descriptor = None
    counts_by_expert = {}
    payload_rows_total = 0
    for family in program.tile_programs:
        header = family.scope.op_ids
        if (len(header) not in (13,14) or header[0] != 'fused_moe_fp8_tiles'
                or header[1] not in ('w1','w2') or header[3] not in ('0','1') or header[4] not in ('0','1')
                or family.scope.agent != 'logical-fp8-routed-blocks' or family.causal_prefixes):
            raise ValueError('FP8 grouped tile source requires explicit variant and payload-box facts')
        current = (*header[:2],*header[3:])
        if descriptor is None:
            descriptor = current
        elif descriptor != current:
            raise ValueError('FP8 grouped tile source must retain shared operand parents and box facts')
        variant,expert,shared,tma,row_block = header[1],int(header[2]),bool(int(header[3])),bool(int(header[4])),int(header[5])
        naive=len(header)==14
        if naive:
            query,qscale,weight,wscale,routing,router,counter,output=(views[name] for name in header[6:])
            if shared or tma or row_block!=1 or (counter.shape,counter.dtype,counter.element_bytes,counter.strides_bytes)!=((1,),'int32',4,(4,)):
                raise ValueError('FP8 naive assignment requires its actual INT32 scalar and one valid row per CTA')
        else:query,qscale,weight,wscale,routing,router,output = (views[name] for name in header[6:])
        experts,columns,reduction = weight.shape
        tokens,topk,out_columns = output.shape
        rows = query.shape[0]
        if (reduction%128 or columns%128 or out_columns != columns or row_block <= 0 or (tma and not shared)
                or (variant == 'w1' and rows != tokens) or (variant == 'w2' and not tma and rows != tokens*topk)):
            raise ValueError('FP8 grouped tile parent geometry/box facts mismatch')
        expected = (((rows,reduction),'fp8',1),((rows,reduction//128),'fp32',4),
            ((experts,columns,reduction),'fp8',1),((experts,columns//128,reduction//128),'fp32',4),
            ((tokens*topk,),'int32',4),((tokens,topk),'fp32',4))
        for view,(shape,dtype,width) in zip((query,qscale,weight,wscale,routing,router),expected,strict=True):
            if (view.shape,view.dtype,view.element_bytes) != (shape,dtype,width):
                raise ValueError('FP8 grouped tile payload/scale parent storage mismatch')
        for view in (query,qscale,weight,wscale,routing,router,output):
            strides, size = [], view.element_bytes
            for dimension in reversed(view.shape):
                strides.append(size)
                size *= dimension
            if view.strides_bytes != tuple(reversed(strides)):
                raise ValueError('FP8 grouped tile must preserve contiguous payload/scale parents')
        parents=(query,qscale,weight,wscale,routing,router,output,*((counter,) if naive else ()))
        if len({view.allocation_id for view in parents}) != len(parents):
            raise ValueError('FP8 grouped tile parent alias needs a source lifecycle')
        axes = unique_by(family.axes,'axis_id')
        if set(axes) != {'row','column'} or axes['row'].block != row_block or (axes['column'].extent,axes['column'].block) != (columns,128):
            raise ValueError('FP8 grouped tiles must preserve BM work and N128 scale groups')
        count = axes['row'].extent
        if not 0 <= expert < experts or expert in counts_by_expert:
            raise ValueError('FP8 grouped tile source duplicates or misplaces an expert')
        counts_by_expert[expert] = count
        payload_rows = -(-count//row_block)*row_block if tma else count
        payload_rows_total += payload_rows
        accesses = unique_by(family.accesses,'access_id')
        roles = {'routing','query','query_scale','weight','weight_scale','output'} | ({'router_weight'} if variant == 'w2' else set())
        if naive:roles.add('num_tokens_post_padded')
        if set(accesses) != roles:
            raise ValueError('FP8 grouped tile source must retain every actual payload/scale/router operand')
        selected_roles = [('routing',routing,(count,),('row',)),('query',query,(payload_rows,reduction),('row',None)),
            ('query_scale',qscale,(count,reduction//128),('row',None)),('output',output,(count,columns),('row','column'))]
        if variant == 'w2':selected_roles.append(('router_weight',router,(count,),('row',)))
        for role,parent,shape,mapping in selected_roles:
            access = accesses[role]; selected = views[access.view_id]
            if (not isinstance(selected,LogicalSelectionView) or selected.parent_view_id != parent.view_id
                    or selected.shape != shape or access.view_axes != mapping or not access.unmapped):
                raise ValueError('FP8 grouped selected payload/scale/output geometry mismatch')
        for role,parent,shape in (('weight',weight,(columns,reduction)),('weight_scale',wscale,(columns//128,reduction//128))):
            selected = views[accesses[role].view_id]
            size = shape[0]*shape[1]*parent.element_bytes
            if ((selected.shape,selected.dtype,selected.element_bytes,selected.offset_bytes,selected.strides_bytes,selected.allocation_id)
                    != (shape,parent.dtype,parent.element_bytes,parent.offset_bytes+expert*size,parent.strides_bytes[1:],parent.allocation_id)):
                raise ValueError('FP8 grouped expert weight payload/scale slice mismatch')
        for role,access in accesses.items():
            iterations = ('row',) if role in ('router_weight','routing') and not naive else ('row','column')
            mapping = ((None,) if role=='num_tokens_post_padded' else ('column',None) if role in ('weight','weight_scale') else (('row','column') if role == 'output' else (('row',) if role in ('routing','router_weight') else ('row',None))))
            if (access.operation is not (Operation.WRITE if role == 'output' else Operation.READ)
                    or access.iteration_axes != iterations or access.view_axes != mapping or access.indirect_target is not None
                    or access.view_axis_divisors != ((128,1) if role == 'weight_scale' else ())
                    or access.view_axis_padding != ((True,False) if role == 'query' and tma else ())
                    or access.unmapped != (role not in ('weight','weight_scale','num_tokens_post_padded'))):
                raise ValueError('FP8 grouped tile must preserve actual BM/mask/scale coordinates')
            if role not in ('routing','output','num_tokens_post_padded') and access.dependencies != (TileDependency('routing',EventKind.READ_READY,()),):
                raise ValueError('FP8 grouped lacks actual scale/payload/routing producer causality')
        if naive:
            scalar=accesses['num_tokens_post_padded']
            if (scalar.view_id!=counter.view_id or scalar.dependencies or len(scalar.input_ready)!=1
                    or accesses['routing'].dependencies!=(TileDependency('num_tokens_post_padded',EventKind.READ_READY,()),)):
                raise ValueError('FP8 naive routing requires its actual counter READ_READY')
        required = {TileDependency(role,EventKind.READ_READY,()) for role in roles-{'routing','output'}}
        if len(accesses['output'].dependencies) != len(required) or set(accesses['output'].dependencies) != required:
            raise ValueError('FP8 grouped lacks actual scale/payload/routing producer causality')
    if descriptor is None:
        raise ValueError('FP8 grouped tile source requires active expert work')
    counts = tuple(counts_by_expert[index] if index in counts_by_expert else 0 for index in range(experts))
    if sum(counts) != tokens*topk or (shared and (counts[-1] != tokens or topk < 2)):
        raise ValueError('FP8 grouped tile routes/shared expert must conserve token placement')
    base_counts = counts[:-1] if shared else counts
    active = len(counts_by_expert); routes = sum(counts)
    read_bytes = (routes*4 + payload_rows_total*reduction + routes*(reduction//128)*4
        + active*columns*reduction + active*(columns//128)*(reduction//128)*4
        + (routes*4 if variant == 'w2' else 0))
    return {'projection':(variant,rows,reduction,columns),'routing':(base_counts,base_counts),
        'read_bytes':read_bytes,'write_bytes':logical_requested_bytes(program,Operation.WRITE),
        'logical_read_requested_bytes':logical_requested_bytes(program,Operation.READ),
        'legacy_read_price_bytes':read_bytes}


def fp8_fused_source_demands(program, *, expected_kernel=None):
    if any(scope.op_ids and scope.op_ids[0] in ('fused_moe_fp8_aligned','fused_moe_fp8_sglang_aligned') for scope in program.scopes):
        return fp8_aligned_tile_demands(program,expected_kernel=expected_kernel)
    if expected_kernel is not None:
        if (not has_fp8_fused_moe_source(expected_kernel) or program.hardware_id!=expected_kernel.hardware_id
                or not program.tile_programs or any(len(f.scope.op_ids)!=6+len(fp8_fused_layouts(expected_kernel))
                    for f in program.tile_programs)):
            raise ValueError('FP8 grouped source differs from its actual kernel and counter ABI')
    if program.tile_programs:
        result=fp8_grouped_tile_demands(program)
        if expected_kernel is not None:
            variant,_,reduction,columns=fused_moe_projection(expected_kernel)
            counts,_=fused_moe_routing(expected_kernel)
            if (result['projection']!=(variant,fp8_fused_quant_rows(expected_kernel),reduction,columns)
                    or result['routing']!=(counts,counts)):
                raise ValueError('FP8 grouped source differs from its actual kernel geometry/routing')
        return result
    from src.kernel.memory_access import requested_bytes
    scopes,views = unique_by(program.scopes,'scope_id'),unique_by(program.views,'view_id')
    families = ('fused_moe_fp8','fused_moe_fp8_sglang','fused_moe_fp8_sglang_tma')
    headers = [scope for scope in program.scopes if len(scope.op_ids) in (11,12)
               and scope.op_ids[0] in families and scope.op_ids[2:4] == ('shared','routing')]
    if len(headers) != 1:
        raise ValueError('FP8 grouped source requires one explicit operand descriptor')
    header = headers[0]
    variant = header.op_ids[1]
    family = header.op_ids[0]
    sglang,tma = family != 'fused_moe_fp8',family == 'fused_moe_fp8_sglang_tma'
    if len(header.op_ids) != (12 if tma else 11):
        raise ValueError('FP8 TMA descriptor requires its actual row-box tile fact')
    tile_m = int(header.op_ids[11]) if tma else None
    if tma and tile_m <= 0:
        raise ValueError('FP8 TMA row-box tile must be positive')
    query,qscale,weight,wscale,routing,router,output = (views[name] for name in header.op_ids[4:11])
    experts,columns,reduction = weight.shape
    tokens,topk,out_columns = output.shape
    rows = query.shape[0]
    if (variant == 'w1' and rows != tokens) or (variant == 'w2' and not tma and rows != tokens*topk):
        raise ValueError('FP8 grouped query row extent mismatch')
    expected = (((rows,reduction),'fp8',1),((rows,reduction//128),'fp32',4),
                ((experts,columns,reduction),'fp8',1),((experts,columns//128,reduction//128),'fp32',4),
                ((tokens*topk,),'int32',4),((tokens,topk),'fp32',4))
    if variant not in ('w1','w2') or out_columns != columns or reduction%128 or columns%128:
        raise ValueError('FP8 grouped projection requires complete128-wide payload/scale geometry')
    for view,(shape,dtype,width) in zip((query,qscale,weight,wscale,routing,router),expected,strict=True):
        if (view.shape,view.dtype,view.element_bytes) != (shape,dtype,width):
            raise ValueError('FP8 grouped operand storage/scale dtype mismatch')
    group,by_expert = None,{}
    for access in program.accesses:
        scope = scopes[access.scope_id]
        if scope.scope_id == header.scope_id:
            if group is not None or access.operation is not Operation.READ:
                raise ValueError('FP8 grouped routing source must be unique')
            group = access
            continue
        roles = scope.op_ids
        if len(roles) != 4 or roles[:2] != (family,variant) or access.repeat != 1:
            raise ValueError('FP8 grouped expert/role source mismatch')
        expert = int(roles[2])
        if expert not in by_expert:
            by_expert[expert] = {}
        if roles[3] in by_expert[expert]:
            raise ValueError('FP8 grouped duplicate expert role')
        by_expert[expert][roles[3]] = access
    if group is None or not by_expert:
        raise ValueError('FP8 grouped actual routing/operands missing')
    if (not isinstance(group.region,Unmapped) or group.allocation_id != routing.allocation_id
            or (group.region.logical_shape,group.region.dtype,group.region.element_bytes) != ((tokens*topk,),'int32',4)):
        raise ValueError('FP8 grouped routing payload storage mismatch')
    nodes = {event.event_id:event for event in program.events}
    nodes.update({access.access_id:access for access in program.accesses})

    def requires(access,refs):
        pending,ancestors = list(access.dependencies),set()
        while pending:
            ref = pending.pop()
            if ref not in ancestors:
                ancestors.add(ref)
                pending.extend(nodes[ref.node_id].dependencies)
        if not set(refs) <= ancestors:
            raise ValueError('FP8 grouped lacks actual scale/payload/routing producer causality')

    counts = [0]*experts
    for expert,operands in by_expert.items():
        required = {'query','query_scale','weight','weight_scale','output'}
        if variant == 'w2':
            required.add('router_weight')
        if not 0 <= expert < experts or set(operands) != required:
            raise ValueError('FP8 grouped expert operand count mismatch')
        count = operands['query_scale'].region.logical_shape[0]
        payload_rows = operands['query'].region.logical_shape[0]
        if payload_rows != (-(-count//tile_m)*tile_m if tma else count):
            raise ValueError('FP8 grouped payload box must contain its valid scale rows')
        if variant == 'w2':
            access = operands['router_weight']
            if (not isinstance(access.region,Unmapped) or access.allocation_id != router.allocation_id
                    or (access.region.logical_shape,access.region.dtype,access.region.element_bytes) != ((count,),'fp32',4)):
                raise ValueError('FP8 grouped router-weight storage mismatch')
        for role,parent,shape in (('query',query,(payload_rows,reduction)),('query_scale',qscale,(count,reduction//128)),
                                  ('output',output,(count,columns))):
            access = operands[role]
            if (not isinstance(access.region,Unmapped) or access.region.logical_shape != shape
                    or access.allocation_id != parent.allocation_id or access.region.dtype != parent.dtype
                    or access.region.element_bytes != parent.element_bytes):
                raise ValueError('FP8 grouped selected payload/scale/output geometry mismatch')
        for role,parent,shape in (('weight',weight,(columns,reduction)),('weight_scale',wscale,(columns//128,reduction//128))):
            access = operands[role]
            view = views[access.view_id]
            size = shape[0]*shape[1]*parent.element_bytes
            if (view.shape != shape or view.offset_bytes != parent.offset_bytes+expert*size
                    or view.strides_bytes != parent.strides_bytes[1:] or access.region != view.region()
                    or access.allocation_id != parent.allocation_id or (view.dtype,view.element_bytes) != (parent.dtype,parent.element_bytes)):
                raise ValueError('FP8 grouped weight payload/scale expert slice mismatch')
        reads = [access for role,access in operands.items() if role != 'output']
        for access in reads:
            if access.operation is not Operation.READ:
                raise ValueError('FP8 grouped operand must be read')
            requires(access,(EventRef(group.access_id,EventKind.READ_READY),))
        written = operands['output']
        if written.operation is not Operation.WRITE:
            raise ValueError('FP8 grouped output must be written')
        requires(written,tuple(EventRef(access.access_id,EventKind.READ_READY) for access in reads))
        counts[expert] = count
    if sum(counts) != tokens*topk:
        raise ValueError('FP8 grouped routed rows must conserve tokens*topk')
    if sglang and (counts[-1] != tokens or topk < 2):
        raise ValueError('FP8 fused shared expert must receive each token once')
    base_counts = tuple(counts[:-1]) if sglang else tuple(counts)
    return {'projection':(variant,rows,reduction,columns),'routing':(base_counts,base_counts),
            'read_bytes':sum(requested_bytes(access.region)*access.repeat for access in program.accesses if access.operation is Operation.READ),
            'write_bytes':sum(requested_bytes(access.region)*access.repeat for access in program.accesses if access.operation is Operation.WRITE)}
