"""DSV3 SIMT router: reused BF16 vectors and the real four-warp shared join."""
from math import prod

from src.kernel.kernel_spec import KernelKind, DTYPE_BYTES
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity,
    MemoryAccessProgram, MemoryAllocation, Operation, Ordering, ProgramEvent,
    Semantics, TensorView, TileAccessTemplate, TileAxis, TileDependency,
    TileProgram,TilePartition,SymbolicExtent,LogicalSelectionView, unique_by,
)


def has_public_router_abi(kernel):
    """The existing caller-declared mathematical router contract.

    This includes synthetic descriptors accepted by the numerical template;
    it does not admit them as observed or executable vendor implementations.
    Proven native interfaces keep their more specific sources.
    """
    from src.kernel.memory_access_builder import has_gpu_nonsplit_gemm_abi
    return (kernel.kind is KernelKind.GEMM and kernel.variant=='moe_router'
        and kernel.execution_support=='supported'
        and not has_observed_router_abi(kernel) and not has_simt_router_abi(kernel)
        and not has_gpu_nonsplit_gemm_abi(kernel)
        and (kernel.launch.tiling is not None and kernel.launch.tiling.split_k is not None
            and kernel.launch.tiling.split_k>=1 or kernel.launch.algorithm=='vllm_dsv3_router_gemm_float_out'
            or kernel.launch.tiling is None and kernel.launch.algorithm=='cublaslt_observed_grid:moe_router'
                and 'split_k_count_observed' in kernel.launch.parameters
                and kernel.launch.parameters['split_k_count_observed']==0.0
                and kernel.launch.parameters['split_k_partial_bytes_per_element']>0))


def _public_router_layouts(kernel,*,count=None):
    _,rows=_token_extents(kernel);n,k=kernel.shape['n'],kernel.shape['k']
    if 'model_input_shape' in kernel.attributes:
        physical=kernel.attributes['model_input_shape'];rows=physical['b']*physical['m'];n,k=physical['n'],physical['k']
    output=kernel.dtype if kernel.attributes['output_dtype'] is None else kernel.attributes['output_dtype']
    splits=1 if kernel.launch.tiling is None else kernel.launch.tiling.split_k
    if kernel.launch.tiling is None and kernel.launch.algorithm=='cublaslt_observed_grid:moe_router':
        splits=SymbolicExtent('public-router.unobserved-split-count',1) if count is None else count
    partial=kernel.launch.parameters['split_k_partial_bytes_per_element']
    if not partial:result=('C',(rows,n),output,DTYPE_BYTES[output])
    elif partial==DTYPE_BYTES[output]:result=('partials',(splits,rows,n),output,int(partial))
    elif partial==4 and output in ('bf16','fp16','fp32'):result=('partials',(splits,rows,n),'fp32',4)
    else:result=('partials',(splits,rows,n,int(partial)),'uint8',1)
    return rows,n,k,splits,(('A',(rows,k),'bf16',2),('B',(n,k),'bf16',2),result)


def standalone_public_router_bindings(kernel,*,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_public_router_abi(kernel):raise ValueError('Public router requires its declared mathematical interface')
    count=(SymbolicExtent(binding_id+'.split_count',1) if kernel.launch.tiling is None
        and kernel.launch.algorithm=='cublaslt_observed_grid:moe_router' else None)
    _,_,_,_,layouts=_public_router_layouts(kernel,count=count)
    allocations=[];views=[];events=[];ready=[]
    for role,shape,dtype,width in layouts:
        identity=binding_id+'.'+role
        allocation=MemoryAllocation(identity,'logical-local-device','logical-storage',None if role=='partials' else prod(shape)*width,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        view=(LogicalSelectionView(role,identity,shape,dtype,width,identity+'.unbound_partial_layout',None) if role=='partials'
            else TensorView(role,identity,0,shape,_strides(shape,width),dtype,width))
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role in ('A','B'):
            event=ProgramEvent(identity+'.ready',EventKind.INITIALIZED,identity,view.region(),'router-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),('A','B'),(layouts[-1][0],),tuple(ready),0,'offline','router-visible')


def build_public_router_source(kernel,hardware,bindings,granularity,*,program_id):
    if not has_public_router_abi(kernel) or kernel.hardware_id!=hardware.hardware_id:
        raise ValueError('Public router source differs from its declared kernel/hardware')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown public router granularity')
    views=unique_by(bindings.views,'view_id');count=None
    if kernel.launch.tiling is None and kernel.launch.algorithm=='cublaslt_observed_grid:moe_router':
        if 'partials' not in views or not isinstance(views['partials'].shape[0],SymbolicExtent) or views['partials'].shape[0].minimum!=1:
            raise ValueError('Public observed-grid descriptor must retain its genuinely unknown positive split count')
        count=views['partials'].shape[0]
    rows,n,k,splits,layouts=_public_router_layouts(kernel,count=count);output=layouts[-1][0]
    if ((bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(('A','B'),(output,),2)
            or set(views)!={role for role,_,_,_ in layouts} or len({v.allocation_id for v in views.values()})!=3
            or any(not isinstance(views[role],LogicalSelectionView if role=='partials' else TensorView)
                or (views[role].shape,views[role].dtype,views[role].element_bytes)!=(shape,dtype,width)
                    for role,shape,dtype,width in layouts)):
        raise ValueError('Public router requires its actual distinct BF16 operands and declared typed result')
    evidence=('Public C[row,expert] = cast(sum_k A[row,k]*B[expert,k]) scalar dot-product interface.',
        'Each independent output consumes its own K reduction; operand reuse remains explicit across outputs.',
        'Caller-supplied numerical descriptors do not establish a native kernel, compiled launch, issued traffic, or private workspace. This source is the public mathematical interface only.')
    accesses=(TileAccessTemplate('activation','A',Operation.READ,('row','expert','k'),('row','k'),(bindings.input_ready[0],),(),evidence),
        TileAccessTemplate('weight','B',Operation.READ,('row','expert','k'),('expert','k'),(bindings.input_ready[1],),(),evidence),
        TileAccessTemplate('output','C',Operation.WRITE,('row','expert'),('row','expert'),(),
            tuple(TileDependency(name,EventKind.READ_READY,('k',)) for name in ('activation','weight')),evidence))
    axes=(TileAxis('row',rows,1),TileAxis('expert',n,1),TileAxis('k',k,1));partitions=()
    if output=='partials' or splits>1:
        from src.kernel.memory_access import TileIndexBinding,TileIndexTerm
        axes=(*axes,TileAxis('split',splits,1))
        partitions=(TilePartition('k',None,class_axis_id='split',context_axes=('row','expert')),)
        evidence=(*evidence,'The declared K-class cardinality is retained, with unbound per-output K membership. No uniform K/split ranges or native slice order are inferred.')
        from dataclasses import replace
        reads=tuple(replace(access,evidence_refs=evidence) for access in accesses[:2])
        if output=='partials':
            mapping=('split','row','expert',*((None,)*(len(views[output].shape)-3)))
            accesses=(*reads,TileAccessTemplate('output','partials',Operation.WRITE,('split','row','expert'),mapping,(),
                tuple(TileDependency(name,EventKind.READ_READY,('k',),partition_axis='k') for name in ('activation','weight')),evidence,unmapped=True))
        else:
            axes=(*axes,TileAxis('update',splits-1,1))
            first=tuple(TileDependency(name,EventKind.READ_READY,('k',),(TileIndexBinding('split',(),0),),partition_axis='k')
                for name in ('activation','weight'))
            following=tuple(TileDependency(name,EventKind.READ_READY,('k',),
                (TileIndexBinding('split',(TileIndexTerm('update',1),),1),),partition_axis='k') for name in ('activation','weight'))
            accesses=(*reads,TileAccessTemplate('initial','C',Operation.WRITE,('row','expert'),('row','expert'),(),first,evidence),
                TileAccessTemplate('accumulator','C',Operation.READ,('row','expert','update'),('row','expert'),(),
                    (TileDependency('update',EventKind.WRITE_VISIBLE,(),carry_axis='update',initial_access_id='initial'),),evidence),
                TileAccessTemplate('update','C',Operation.WRITE,('row','expert','update'),('row','expert'),(),
                    (*following,TileDependency('accumulator',EventKind.READ_READY,())),evidence))
    header=('moe_router','public_scalar_dot_product')
    if not has_router_replay_mapping(kernel):
        header=(*header,f"logical_rows={kernel.shape['b']*kernel.shape['m']}",f"logical_experts={kernel.shape['n']}",f"logical_hidden={kernel.shape['k']}")
        evidence=(*evidence,'Full captured mathematical operands are retained. The legacy logical-work charge is explicit metadata; a source-to-logical coordinate/prefix mapping is not claimed.')
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-router-public-interface')
    family=TileProgram(program_id,scope,axes,accesses,bindings.visibility_domain,evidence,partitions=partitions)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,evidence),bindings.allocations,bindings.views,(),bindings.events,(),(family,))


def public_router_source_work(program,*,expected_kernel,hardware,expected_bindings=None):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if len(program.tile_programs)!=1 or program.accesses:raise ValueError('Public router requires its unique compressed source')
    family,=program.tile_programs;reads=unique_by(family.accesses,'access_id')
    if expected_bindings is None:
        expected_bindings=ElementwiseBindings(program.binding_id,family.scope.call_id,program.allocations,program.views,program.events,
            ('A','B'),(_public_router_layouts(expected_kernel)[-1][-1][0],),(reads['activation'].input_ready[0],reads['weight'].input_ready[0]),
            family.scope.rank,family.scope.stream,family.visibility_domain)
    expected=build_public_router_source(expected_kernel,hardware,expected_bindings,Granularity.TILE,program_id=program.program_id)
    if expected!=program:raise ValueError('Public router source differs from actual caller bindings or complete K dependencies')
    logical,execution=_token_extents(expected_kernel);views=unique_by(program.views,'view_id')
    experts,hidden=views['B'].shape[0],views['A'].shape[1]
    if len(family.scope.op_ids)==5:
        logical,experts,hidden=(int(value.split('=',1)[1]) for value in family.scope.op_ids[2:])
    widths=(views['A'].element_bytes,views['B'].element_bytes,
        DTYPE_BYTES[expected_kernel.dtype if expected_kernel.attributes['output_dtype'] is None else expected_kernel.attributes['output_dtype']])
    # The scalar-dot interface shares B across rows and A across experts.
    # Keep compulsory storage and issued logical requests distinct. Captured
    # descriptors retain the explicit logical charge, not a guessed prefix.
    activation,weight,output=logical*hidden*widths[0],experts*hidden*widths[1],logical*experts*widths[2]
    nominal=expected_kernel.shape['dtype_bytes']
    partial_class=None
    if 'partials' in views:
        from math import prod
        partial=views['partials']
        partial_class=logical*experts*partial.element_bytes*prod(partial.shape[3:])
    return {'logical_rows':logical,'execution_rows':execution,'experts':experts,'hidden':hidden,
        'operand_widths':widths,'nominal_pricing_width':nominal,
        'simt_l2_bytes':weight+experts*activation+output,
        'simt_dram_bytes':weight+activation+output,
        'input_bytes':activation+weight,'partial_bytes_per_class':partial_class,
        'nominal_operand_bytes':(logical*hidden*nominal,experts*hidden*nominal,logical*experts*nominal)}


def has_observed_router_abi(kernel):
    return (kernel.kind is KernelKind.GEMM and kernel.variant=='moe_router'
        and kernel.hardware_id=='nvidia_h200' and kernel.execution_support=='supported'
        and kernel.launch.algorithm=='cublaslt_observed_grid:moe_router'
        and kernel.launch.tiling is None and kernel.launch.parameters['split_k_count_observed']==0.0
        and kernel.launch.parameters['split_k_partial_bytes_per_element']==4.0
        and has_router_replay_mapping(kernel)
        and kernel.attributes['weight_dtype']=='bf16' and kernel.attributes['weight_group_size']==0)


def standalone_observed_router_bindings(kernel,*,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_observed_router_abi(kernel):raise ValueError('Observed router requires its grid-only external FP32-partial ABI')
    _,rows=_token_extents(kernel);experts,hidden=kernel.shape['n'],kernel.shape['k']
    count=SymbolicExtent(binding_id+'.split_count',1)
    layouts=(('A',(rows,hidden),'bf16',2),('B',(experts,hidden),'bf16',2),('partials',(count,rows,experts),'fp32',4))
    allocations=[];views=[];events=[];ready=[]
    for role,shape,dtype,width in layouts:
        identity=binding_id+'.'+role
        allocation=MemoryAllocation(identity,'logical-local-device','logical-storage',None if role=='partials' else prod(shape)*width,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        view=(LogicalSelectionView(role,identity,shape,dtype,width,identity+'.unobserved_split_layout',None) if role=='partials'
            else TensorView(role,identity,0,shape,_strides(shape,width),dtype,width))
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role!='partials':
            event=ProgramEvent(identity+'.ready',EventKind.INITIALIZED,identity,view.region(),'router-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),('A','B'),('partials',),tuple(ready),0,'offline','router-visible')


def build_observed_router_source(kernel,hardware,bindings,granularity,*,program_id):
    if not has_observed_router_abi(kernel) or hardware.hardware_id!=kernel.hardware_id:
        raise ValueError('Observed router source differs from actual hardware/launch')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown observed router granularity')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(('A','B'),('partials',),2):
        raise ValueError('Observed router requires both actual operands and external partials')
    views=unique_by(bindings.views,'view_id');_,rows=_token_extents(kernel);n,k=kernel.shape['n'],kernel.shape['k']
    count=views['partials'].shape[0]
    if (set(views)!={'A','B','partials'} or not isinstance(count,SymbolicExtent) or count.minimum!=1
            or not isinstance(views['partials'],LogicalSelectionView) or views['partials'].shape!=(count,rows,n)
            or (views['partials'].dtype,views['partials'].element_bytes)!=('fp32',4)
            or any((views[role].shape,views[role].dtype,views[role].element_bytes)!=(shape,'bf16',2)
                for role,shape in (('A',(rows,k)),('B',(n,k))))):
        raise ValueError('Observed router must retain the unobserved split cardinality and actual BF16 operands')
    if len({v.allocation_id for v in views.values()})!=3:raise ValueError('Observed router operands and partials must retain distinct public parents')
    evidence=('H200 observed cuBLAS router: grid/block and separate FP32-partial reduction are retained; tile geometry and split count were not observed.',
        'Mathematical row/expert/K coordinates partition across an unbound number of external partial classes; neither uniform K slices nor a native CTA count is inferred.',
        'Each partial is FP32. Its positive class cardinality, physical layout, capacity and exact output traffic remain unknown; the historical two-class price projection does not establish an actual minimum of two.')
    accesses=(TileAccessTemplate('activation','A',Operation.READ,('row','expert','k'),('row','k'),(bindings.input_ready[0],),(),evidence),
        TileAccessTemplate('weight','B',Operation.READ,('row','expert','k'),('expert','k'),(bindings.input_ready[1],),(),evidence),
        TileAccessTemplate('partials','partials',Operation.WRITE,('split','row','expert'),('split','row','expert'),(),
            tuple(TileDependency(name,EventKind.READ_READY,('k',),partition_axis='k') for name in ('activation','weight')),evidence,unmapped=True))
    scope=AccessScope(program_id,('moe_router','observed_grid_symbolic_split'),bindings.call_id,(),bindings.rank,bindings.stream,'logical-router-external-partials')
    family=TileProgram(program_id,scope,(TileAxis('row',rows,1),TileAxis('expert',n,1),TileAxis('k',k,1),TileAxis('split',count,1)),
        accesses,bindings.visibility_domain,evidence,partitions=(TilePartition('k',None,class_axis_id='split',context_axes=('row','expert')),))
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,evidence),bindings.allocations,bindings.views,(),bindings.events,(),(family,))


def observed_router_source_work(program,*,expected_kernel,hardware,expected_bindings=None,legacy_split_lower_bound=None):
    from src.kernel.memory_access_builder import ElementwiseBindings
    program.__post_init__()
    if len(program.tile_programs)!=1 or program.accesses:raise ValueError('Observed router requires its unique compressed source')
    family,=program.tile_programs;views=unique_by(program.views,'view_id');reads=unique_by(family.accesses,'access_id')
    if expected_bindings is None:
        expected_bindings=ElementwiseBindings(program.binding_id,family.scope.call_id,program.allocations,program.views,program.events,
            ('A','B'),('partials',),(reads['activation'].input_ready[0],reads['weight'].input_ready[0]),
            family.scope.rank,family.scope.stream,family.visibility_domain)
    expected=build_observed_router_source(expected_kernel,hardware,expected_bindings,Granularity.TILE,program_id=program.program_id)
    if expected!=program:raise ValueError('Observed router source differs from actual caller bindings or symbolic split dataflow')
    logical,execution=_token_extents(expected_kernel);count=views['partials'].shape[0]
    a,b=views['A'],views['B'];partial=views['partials']
    if legacy_split_lower_bound is not None and (type(legacy_split_lower_bound) is not int or legacy_split_lower_bound<1):
        raise ValueError('Legacy split price projection requires its explicit positive integer policy')
    return {'logical_rows':logical,'execution_rows':execution,'experts':b.shape[0],'hidden':a.shape[1],
        'minimum_split_count':count.minimum,'actual_split_count':None,'legacy_pricing_split_lower_bound':legacy_split_lower_bound,
        'legacy_operand_read_bytes':logical*a.shape[1]*a.element_bytes+prod(b.shape)*b.element_bytes,
        'legacy_partial_write_lower_bound_bytes':None if legacy_split_lower_bound is None else legacy_split_lower_bound*logical*b.shape[0]*partial.element_bytes,
        'actual_partial_write_bytes':None}


def has_observed_router_reducer_abi(kernel):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant=='gemm_reduce'
        and kernel.execution_support=='supported' and kernel.launch.algorithm=='observed_gemm_aux:gemm_reduce'
        and 'split_k_count_observed' in kernel.launch.parameters
        and kernel.launch.parameters['split_k_count_observed']==0.0
        and kernel.launch.parameters['split_k_partial_bytes_per_element']>=1.0)


def _unobserved_reducer_layouts(kernel,count):
    from src.kernel.memory_access_builder import _gemm_elementwise_extents
    _,elements=_gemm_elementwise_extents(kernel);partial=kernel.launch.parameters['split_k_partial_bytes_per_element']
    native=kernel.hardware_id=='nvidia_h200' and partial==4.0
    output='fp32' if native else kernel.dtype;width=DTYPE_BYTES[output]
    if native:source=((count,elements),'fp32',4)
    elif partial!=int(partial):source=((count,elements),None,None)
    elif partial==kernel.shape['dtype_bytes']:source=((count,elements),kernel.dtype,int(partial))
    elif partial==4 and kernel.dtype in ('bf16','fp16','fp32'):source=((count,elements),'fp32',4)
    else:source=((count,elements,int(partial)),'uint8',1)
    return source,((elements,),output,width),native


def standalone_observed_router_reducer_bindings(kernel,*,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings,_gemm_elementwise_extents
    if not has_observed_router_reducer_abi(kernel):raise ValueError('Observed reducer requires its explicitly unobserved positive count')
    # The observed cuBLASLt launch explicitly binds an FP32 result even
    # when the enclosing logical router descriptor carries nominal BF16.
    _,elements=_gemm_elementwise_extents(kernel);count=SymbolicExtent(binding_id+'.split_count',1)
    (shape,dtype,input_width),(_,output_dtype,width),_=_unobserved_reducer_layouts(kernel,count)
    source=MemoryAllocation(binding_id+'.input0','logical-local-device','logical-storage',None,AddressLevel.OBJECT,None,None,binding_id+'.input0.allocated',None)
    output=MemoryAllocation(binding_id+'.output0','logical-local-device','logical-storage',elements*width,AddressLevel.OBJECT,None,None,binding_id+'.output0.allocated',None)
    selected=LogicalSelectionView('input0',source.allocation_id,shape,dtype,input_width,binding_id+'.unobserved_partial_layout',None)
    result=TensorView('output0',output.allocation_id,0,(elements,),(width,),output_dtype,width)
    ready=ProgramEvent(binding_id+'.partials.available',EventKind.LOGICAL_AVAILABLE,source.allocation_id,selected.region(),'router-visible',
        (EventRef(source.allocated_event,EventKind.ALLOCATED),))
    events=(ProgramEvent(source.allocated_event,EventKind.ALLOCATED,source.allocation_id,None,None,()),
        ProgramEvent(output.allocated_event,EventKind.ALLOCATED,output.allocation_id,None,None,()),ready)
    return ElementwiseBindings(binding_id,call_id,(source,output),(selected,result),events,('input0',),('output0',),
        (EventRef(ready.event_id,ready.kind),),0,'offline','router-visible')


def build_observed_router_reducer_source(kernel,hardware,bindings,granularity,*,program_id):
    from src.kernel.memory_access_builder import _gemm_elementwise_extents
    if not has_observed_router_reducer_abi(kernel) or hardware.hardware_id!=kernel.hardware_id:
        raise ValueError('Observed router reducer differs from actual hardware/launch')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown observed reducer granularity')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(('input0',),('output0',),1):
        raise ValueError('Observed reducer requires actual partial and output bindings')
    views=unique_by(bindings.views,'view_id');logical,elements=_gemm_elementwise_extents(kernel)
    source,result=views['input0'],views['output0'];count=source.shape[0]
    input_layout,output_layout,native=_unobserved_reducer_layouts(kernel,count)
    if (set(views)!={'input0','output0'} or not isinstance(source,LogicalSelectionView)
            or isinstance(count,SymbolicExtent) and count.minimum!=1 or type(count) is int and count<1
            or (source.shape,source.dtype,source.element_bytes)!=input_layout
            or not isinstance(result,TensorView) or (result.shape,result.dtype,result.element_bytes,result.strides_bytes)
                !=(*output_layout,(output_layout[2],))
            or source.allocation_id==result.allocation_id):
        raise ValueError('Observed reducer must retain unknown positive partial count and actual output geometry')
    evidence=('Observed H200 splitKreduce_kernel consumes external FP32 partial classes; its actual class count and layout were not observed.',
        'Each public output element waits for all corresponding logical partial classes; independent elements have no cross-output barrier.',
        'The canonical observed router selection requires FP32 output; nominal KernelSpec dtype is retained only in the legacy pricing projection.',
        'KernelSpec split_k is the retained analytical price projection, not the symbolic actual count.')
    if not native:
        evidence=('Caller-declared reducer with explicitly unobserved positive split count; public output and partial encoding retain their declared scalar/byte ABI.',
            'Each output waits for every corresponding logical partial class, without a cross-output barrier or native implementation claim.',
            'KernelSpec split_k and nominal byte widths remain the original numerical pricing projection, not an observed cardinality or physical layout.')
    accesses=(TileAccessTemplate('partials','input0',Operation.READ,('split','element'),('split','element',*((None,)*(len(source.shape)-2))),bindings.input_ready,(),evidence,unmapped=True),
        TileAccessTemplate('output','output0',Operation.WRITE,('element',),('element',),(),
            (TileDependency('partials',EventKind.READ_READY,('split',)),),evidence))
    scope=AccessScope(program_id,('observed_router_reducer',f'logical_elements={logical}',f'execution_elements={elements}'),
        bindings.call_id,(),bindings.rank,bindings.stream,'logical-router-partial-reduction')
    family=TileProgram(program_id,scope,(TileAxis('split',count,1),TileAxis('element',elements,1)),accesses,bindings.visibility_domain,evidence)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,evidence),bindings.allocations,bindings.views,(),bindings.events,(),(family,))


def observed_router_reducer_source_work(program,*,expected_kernel,hardware,expected_bindings=None):
    from src.kernel.memory_access_builder import ElementwiseBindings,_gemm_elementwise_extents
    program.__post_init__()
    if len(program.tile_programs)!=1 or program.accesses:raise ValueError('Observed reducer requires its unique compressed source')
    family,=program.tile_programs;reads=unique_by(family.accesses,'access_id');views=unique_by(program.views,'view_id')
    if expected_bindings is None:
        expected_bindings=ElementwiseBindings(program.binding_id,family.scope.call_id,program.allocations,program.views,program.events,
            ('input0',),('output0',),reads['partials'].input_ready,family.scope.rank,family.scope.stream,family.visibility_domain)
    expected=build_observed_router_reducer_source(expected_kernel,hardware,expected_bindings,Granularity.TILE,program_id=program.program_id)
    if expected!=program:raise ValueError('Observed reducer source differs from actual caller bindings or partial dependencies')
    logical,elements=_gemm_elementwise_extents(expected_kernel);policy=expected_kernel.attributes['split_k']
    count=views['input0'].shape[0];width=expected_kernel.launch.parameters['split_k_partial_bytes_per_element']
    native=expected_kernel.hardware_id=='nvidia_h200' and width==4.0
    legacy_width=4 if native else int(width)
    return {'logical_elements':logical,'execution_elements':elements,'actual_split_count':None if isinstance(count,SymbolicExtent) else count,
        'legacy_pricing_split_count':policy,'legacy_read_bytes':policy*logical*legacy_width,
        'legacy_write_bytes':logical*expected_kernel.shape['dtype_bytes'],
        'actual_read_bytes':None if isinstance(count,SymbolicExtent) or views['input0'].element_bytes is None
            else count*prod(views['input0'].shape[1:])*views['input0'].element_bytes,
        'actual_write_bytes':elements*views['output0'].element_bytes}


def has_simt_router_source(kernel, hardware):
    return hardware.hardware_id == 'nvidia_h200' and has_simt_router_abi(kernel)


def has_declared_router_extents(kernel):
    if 'model_input_shape' not in kernel.attributes:
        return True
    physical=kernel.attributes['model_input_shape']
    return {'b','m','n','k'} <= set(physical) and all(physical[axis]==kernel.shape[axis] for axis in ('b','m','n','k'))


def has_router_replay_mapping(kernel):
    if has_declared_router_extents(kernel):
        return True
    physical=kernel.attributes['model_input_shape']
    return ({'b','m','n','k'} <= set(physical)
        and kernel.attributes['graph_runtime_mode'] in ('FULL','PIECEWISE')
        and all(physical[axis]==kernel.shape[axis] for axis in ('b','n','k'))
        and physical['m'] >= kernel.shape['m']
        and physical['m']*kernel.attributes['graph_logical_tokens']
            == kernel.shape['m']*kernel.attributes['graph_model_input_tokens'])


def _token_extents(kernel):
    logical=kernel.shape['b']*kernel.shape['m']
    if 'model_input_shape' not in kernel.attributes:
        return logical,logical
    physical=kernel.attributes['model_input_shape']
    return logical,physical['b']*physical['m']


def nonsplit_router_logical_price_program(program,*,expected_kernel,hardware):
    """Select a uniform logical-row charge from the complete replay source.

    This restricted program is a price projection, never the exported or
    caller-bound execution source. Parent capacities/readiness remain those
    of the real replay. The existing traffic/cache formula consumes this
    prefix rather than independently reconstructing its operand geometry.
    """
    from dataclasses import replace
    from src.kernel.memory_access_builder import has_gpu_nonsplit_gemm_source
    from src.kernel.memory_path_solver import logical_gemm_operand_geometry
    kernel=expected_kernel
    if kernel.variant!='moe_router' or not has_gpu_nonsplit_gemm_source(kernel,hardware):
        raise ValueError('Logical router price requires its supported nonsplit source')
    logical,execution=_token_extents(kernel)
    width=2 if kernel.dtype=='bf16' else 4
    expected=(execution,kernel.shape['n'],kernel.shape['k'],2,2,width)
    if logical_gemm_operand_geometry(program)!=expected:
        raise ValueError('Router price source does not cover its complete actual replay operands')
    family,=program.tile_programs;views=unique_by(program.views,'view_id')
    a,b,c=(views[access.view_id] for access in family.accesses)
    if ((a.shape,b.shape,c.shape)!=((execution,expected[2]),(expected[1],expected[2]),(execution,expected[1]))
            or (a.dtype,b.dtype,c.dtype)!=('bf16','bf16',kernel.dtype)):
        raise ValueError('Router replay price requires its actual typed complete matrices')
    if logical==execution:return program
    replacements={a.view_id:replace(a,shape=(logical,expected[2])),c.view_id:replace(c,shape=(logical,expected[1]))}
    result=replace(program,views=tuple(replacements[v.view_id] if v.view_id in replacements else v for v in program.views),
        tile_programs=(replace(family,axes=tuple(replace(axis,extent=logical) if axis.axis_id=='m' else axis for axis in family.axes)),),
        coverage=replace(program.coverage,limitations=(*program.coverage.limitations,
            'Uniform logical-row pricing projection only; full replay source and storage remain separate. No padding values inferred.')))
    logical_gemm_operand_geometry(result)
    return result


def has_simt_router_abi(kernel):
    return (kernel.kind is KernelKind.GEMM and kernel.variant == 'moe_router'
        and kernel.execution_support == 'supported' and kernel.hardware_id == 'nvidia_h200'
        and has_router_replay_mapping(kernel)
        and kernel.dtype in ('bf16', 'fp32')
        and kernel.attributes['output_dtype'] in (None, kernel.dtype)
        and (kernel.shape['k'], kernel.shape['n']) in ((6144,256),(7168,256),(7168,384))
        and 1 <= _token_extents(kernel)[1] <= 16
        and kernel.launch.algorithm == 'vllm_dsv3_router_gemm_float_out'
        and kernel.launch.tiling is None and kernel.launch.block == (128,1,1)
        and kernel.launch.grid == (kernel.shape['n'],1,1)
        and kernel.attributes['weight_dtype'] == 'bf16' and kernel.attributes['weight_group_size'] == 0)


def _strides(shape, width):
    return tuple(prod(shape[index+1:]) * width for index in range(len(shape)))


def _layouts(kernel):
    _,t=_token_extents(kernel);n,k=kernel.shape['n'],kernel.shape['k']
    # The public native B is [expert,K], not the row-major [K,expert] of a
    # generic GEMM view. Each CTA streams its own contiguous expert row once.
    return (('A',(t,k),'bf16'),('B',(n,k),'bf16'),('C',(t,n),kernel.dtype))


def standalone_simt_router_bindings(kernel, *, call_id, binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    allocations,views,events,ready = [],[],[],[]
    for role,shape,dtype in _layouts(kernel):
        identity=f'{binding_id}.{role}';width=DTYPE_BYTES[dtype]
        allocation=MemoryAllocation(identity,'logical-local-device','logical-storage',prod(shape)*width,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        view=TensorView(role,identity,0,shape,_strides(shape,width),dtype,width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role != 'C':
            event=ProgramEvent(identity+'.ready',EventKind.INITIALIZED,identity,view.region(),'router-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('A','B'),('C',),tuple(ready),0,'offline','router-visible')


def build_simt_router_source(kernel, hardware, bindings, granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_simt_router_source(kernel,hardware):
        raise NotImplementedError('SIMT router source requires the declared native algorithm and launch')
    if not isinstance(bindings,ElementwiseBindings) or granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise TypeError('SIMT router needs typed projection bindings and logical granularity')
    views=unique_by(bindings.views,'view_id');allocations=unique_by(bindings.allocations,'allocation_id')
    if set(views) != {'A','B','C'} or bindings.read_views != ('A','B') or bindings.write_views != ('C',) or len(bindings.input_ready)!=2:
        raise ValueError('SIMT router must preserve both actual operands and its typed result')
    for role,shape,dtype in _layouts(kernel):
        view=views[role];width=DTYPE_BYTES[dtype]
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes) != (shape,dtype,width,_strides(shape,width)):
            raise ValueError('SIMT router operand dtype/layout differs from its actual kernel')
    if len({v.allocation_id for v in views.values()}) != 3:
        raise ValueError('SIMT router projection requires independent input/output parents')
    if len({allocations[v.allocation_id].device_id for v in views.values()}) != 1:
        raise ValueError('SIMT router projection operands must share the actual compute device')
    logical,t=_token_extents(kernel);n,k=kernel.shape['n'],kernel.shape['k'];iterations=k//1024
    a,b,c=(views[role] for role in ('A','B','C'))
    vectors=[]
    for view,count in ((a,t),(b,n)):
        shape=(count,iterations,4,256)
        vectors.append(TensorView(view.view_id+'_vectors',view.allocation_id,view.offset_bytes,shape,
            _strides(shape,2),'bf16',2))
    identity=f'{bindings.binding_id}.router_warp_sums';shape=(n,t,4)
    scratch=MemoryAllocation(identity,allocations[c.allocation_id].device_id,'router-cta-shared',prod(shape)*4,
        AddressLevel.OBJECT,None,None,identity+'.allocated',None)
    sums=TensorView('warp_sums',identity,0,shape,_strides(shape,4),'fp32',4)
    allocated=ProgramEvent(scratch.allocated_event,EventKind.ALLOCATED,identity,None,None,())
    evidence=('vLLM 0.24 DSV3 router native BF16/FP32 output templates: one 128-thread CTA per expert.',
        'Each warp reads 32*8 BF16 elements per K iteration; B is reused across all tokens in its expert CTA.',
        'Four FP32 warp sums per token are written before the CTA barrier and read by thread zero; no cross-expert barrier.',
        f'Logical token prefix {logical}; actual execution token extent {t}. Replay padding values are unknown, not zero work.')
    axes=(TileAxis('expert',n,1),TileAxis('token',t,1),TileAxis('ki',iterations,1),TileAxis('warp',4,1))
    accesses=(
        TileAccessTemplate('weight','B_vectors',Operation.READ,('expert','ki','warp'),('expert','ki','warp',None),
            (bindings.input_ready[1],),(),evidence),
        TileAccessTemplate('activation','A_vectors',Operation.READ,('expert','token','ki','warp'),('token','ki','warp',None),
            (bindings.input_ready[0],),(),evidence),
        TileAccessTemplate('warp_store','warp_sums',Operation.WRITE,('expert','token','warp'),('expert','token','warp'),(),
            (TileDependency('weight',EventKind.READ_READY,('ki',)),TileDependency('activation',EventKind.READ_READY,('ki',))),evidence),
        TileAccessTemplate('warp_join','warp_sums',Operation.READ,('expert','token','warp'),('expert','token','warp'),(),
            (TileDependency('warp_store',EventKind.WRITE_VISIBLE,('token','warp')),),evidence),
        TileAccessTemplate('output','C',Operation.WRITE,('expert','token'),('token','expert'),(),
            (TileDependency('warp_join',EventKind.READ_READY,('warp',)),),evidence))
    scope=AccessScope(program_id,('moe_router','simt'),bindings.call_id,(),bindings.rank,bindings.stream,'logical-router-cta')
    family=TileProgram(program_id,scope,axes,accesses,bindings.visibility_domain,evidence)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'K vectors and shared reduction are source-bound logical requests; compiler register/spill and physical issue are unbound.',
            'Shared storage enumerates separate expert CTA objects; its sum is not simultaneously resident device capacity.',
            'Source-package algorithm reference does not assert byte identity with the installed binary.')),
        (*bindings.allocations,scratch),(*bindings.views,*vectors,sums),(),(*bindings.events,allocated),(),(family,))


def simt_router_source_work(program, *, expected_kernel, hardware, expected_bindings=None):
    """Revalidate the actual source; keep shared requests out of retained pricing."""
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_path_solver import logical_view_requested_bytes,logical_view_unique_bytes
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:
        raise ValueError('SIMT router requires its actual compressed CTA source')
    family,=program.tile_programs;views=unique_by(program.views,'view_id')
    if expected_bindings is None:
        roots=tuple(views[role] for role in ('A','B','C'));ids={view.allocation_id for view in roots}
        allocations=tuple(a for a in program.allocations if a.allocation_id in ids)
        events=tuple(e for e in program.events if e.allocation_id in ids)
        reads=unique_by(family.accesses,'access_id')
        ready=(reads['activation'].input_ready[0],reads['weight'].input_ready[0])
        expected_bindings=ElementwiseBindings(program.binding_id,family.scope.call_id,allocations,roots,events,
            ('A','B'),('C',),ready,family.scope.rank,family.scope.stream,family.visibility_domain)
    expected=build_simt_router_source(expected_kernel,hardware,expected_bindings,Granularity.TILE,program_id=program.program_id)
    if expected!=program:
        raise ValueError('SIMT router source lost its actual operands, warp reduction, CTA barrier or caller binding')
    read=lambda role:logical_view_requested_bytes(program,role,Operation.READ)
    write=lambda role:logical_view_requested_bytes(program,role,Operation.WRITE)
    work={'activation_read_bytes':read('A_vectors'),'weight_read_bytes':read('B_vectors'),
        'activation_unique_bytes':logical_view_unique_bytes(program,'A_vectors',Operation.READ),
        'output_write_bytes':write('C'),'shared_read_bytes':read('warp_sums'),'shared_write_bytes':write('warp_sums')}
    logical,execution=_token_extents(expected_kernel)
    # Existing lowering prices logical work even when replay uses a padded
    # input. Each native source token has the same complete vector/output
    # requests, so the declared logical prefix is an exact subset projection.
    # This projection does not claim that replay omits the other requests.
    projected=('activation_read_bytes','activation_unique_bytes','output_write_bytes')
    if any(work[role]%execution for role in projected):
        raise ValueError('SIMT router logical prefix requires complete uniform source token rows')
    work['legacy_logical_projection']={role:work[role]//execution*logical for role in projected}
    work['legacy_logical_projection']['weight_read_bytes']=work['weight_read_bytes']
    work['logical_tokens']=logical;work['execution_tokens']=execution
    return work
