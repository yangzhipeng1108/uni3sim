"""Marlin routed wrapper payloads and shared intermediate-cache lifetime."""
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity,
    MemoryAccessProgram, MemoryAllocation, Operation, Ordering, ProgramEvent, Semantics,
    TensorView, Unmapped, unique_by,
)


def has_routed_moe_abi(kernel):
    return (kernel.kind is KernelKind.FUSED_MOE and kernel.variant == 'routed_experts'
            and kernel.execution_support == 'supported'
            and kernel.attributes['expert_parallel_size'] == 1
            and kernel.launch.algorithm in ('vllm_marlin_moe_w4a16_aggregate','vllm_marlin_moe_mxfp4_aggregate'))


def has_routed_moe_source(kernel, hardware):
    return has_routed_moe_abi(kernel) and hardware.kernel_memory_model == 'gpu_mshr'


def routed_moe_layouts(kernel):
    tokens,experts,topk,hidden,intermediate = (kernel.shape[name] for name in ('tokens','experts','topk','hidden','intermediate'))
    scale_dtype,width = ('bf16',2) if kernel.attributes['routed_weight_dtype'] == 'int4' else ('uint8',1)
    # Packed and permuted weight/scale parents are byte/element descriptors;
    # they do not assert a row-major nibble or Marlin permutation mapping.
    return _routed_layouts(tokens,experts,topk,hidden,intermediate,scale_dtype,width)


def _routed_layouts(tokens,experts,topk,hidden,intermediate,scale_dtype,width):
    rows=tokens*topk
    return (('query',(tokens,hidden),'bf16',2),('routing',(rows,),'int32',4),
            ('router_weight',(tokens,topk),'fp32',4),
            ('w13_weight',(experts,hidden*intermediate),'uint8',1),
            ('w13_scale',(experts,2*intermediate*(hidden//32)),scale_dtype,width),
            ('w2_weight',(experts,intermediate*hidden//2),'uint8',1),
            ('w2_scale',(experts,hidden*(intermediate//32)),scale_dtype,width),
            ('w13',(rows,2*intermediate),'bf16',2),('activation',(rows,intermediate),'bf16',2),
            ('down',(rows,hidden),'bf16',2),('output',(tokens,hidden),'bf16',2))


def standalone_routed_moe_bindings(kernel, *, call_id, binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    layouts = routed_moe_layouts(kernel)
    allocations,views,events,reads,ready = [],[],[],[],[]
    cache_rows = kernel.shape['tokens']*kernel.shape['topk']
    cache_bytes = cache_rows*max(2*kernel.shape['intermediate'],kernel.shape['hidden'])*2
    for index,(role,shape,dtype,width) in enumerate(layouts):
        identity = f'{binding_id}.cache13' if role in ('w13','down') else f'{binding_id}.{role}'
        strides,size = [],width
        for axis in reversed(shape):
            strides.append(size)
            size *= axis
        views.append(TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width))
        if role != 'down':
            allocation = MemoryAllocation(identity,'logical-local-device','logical-storage',
                cache_bytes if role == 'w13' else size,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
            allocations.append(allocation)
            events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if index < 7:
            view = views[-1]
            event = ProgramEvent(f'{identity}.initialized',EventKind.INITIALIZED,identity,view.region(),'marlin-visible',
                (EventRef(f'{identity}.allocated',EventKind.ALLOCATED),))
            events.append(event)
            reads.append(role)
            ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),tuple(reads),
        ('w13','activation','down','output'),tuple(ready),0,'offline','marlin-visible')


def build_routed_moe_source(kernel, hardware, bindings, granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('routed wrapper requires typed payload/scale/intermediate bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise NotImplementedError('Marlin wrapper exposes logical M tiles, not a native issued trace')
    expected = standalone_routed_moe_bindings(kernel,call_id=bindings.call_id,binding_id=bindings.binding_id)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (
            expected.read_views,expected.write_views,len(expected.input_ready)):
        raise ValueError('routed wrapper operand roles mismatch')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    if set(views) != {view.view_id for view in expected.views}:
        raise ValueError('routed wrapper operand descriptor missing')
    for ref in expected.views:
        view = views[ref.view_id]
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes) != (ref.shape,ref.strides_bytes,ref.dtype,ref.element_bytes):
            raise ValueError('routed wrapper payload/scale storage mismatch')
    if views['w13'].allocation_id != views['down'].allocation_id or len({view.allocation_id for view in views.values()}) != len(views)-1:
        raise ValueError('Marlin W13/down must reuse cache13 with other parents distinct')
    readiness = dict(zip(bindings.read_views,bindings.input_ready,strict=True))
    source_events = unique_by(bindings.events,'event_id')
    for role in bindings.read_views:
        event = source_events[readiness[role].node_id]
        view = views[role]
        resolved=(event.kind in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE) and event.region==view.region())
        logical=(event.kind is EventKind.LOGICAL_AVAILABLE and isinstance(event.region,Unmapped)
            and (event.region.logical_shape,event.region.dtype,event.region.element_bytes)==(view.shape,view.dtype,view.element_bytes))
        if event.allocation_id != view.allocation_id or not (resolved or logical):
            raise ValueError('routed selected input requires complete parent readiness')
    return _build_routed_moe_tiles(kernel,hardware,bindings,views,readiness,program_id)


def _routed_moe_tiles(views,ready,scope,visibility,evidence,unknown_allocations,metadata_ready=()):
    """One logical vector dot per original (token, top-k) routed row."""
    from src.kernel.memory_access import (LogicalSelectionView,LogicalCoordinateBinding,LogicalCoordinateTerm,
        TileAxis,TileAccessTemplate,TileDependency,TileProgram)
    tokens,hidden=views['query'].shape;rows,intermediate=views['activation'].shape;topk=rows//tokens
    axes=(TileAxis('token',tokens,1),TileAxis('topk',topk,1));selected=[];accesses=[]
    routed=('token','topk');dep=lambda name,kind,reduction=():TileDependency(name,kind,reduction)
    def reshape(role,name,shape):
        parent=views[role]
        coordinates=(LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,topk),LogicalCoordinateTerm(1,1))),)
        if len(parent.shape)==2:
            coordinates+= (LogicalCoordinateBinding(1,(LogicalCoordinateTerm(2,1),)),)
        selected.append(LogicalSelectionView(name,parent.allocation_id,shape,parent.dtype,parent.element_bytes,
            f'{scope.scope_id}.{name}.original_token_topk_row',role,coordinates))
    reshape('routing','routing_rows',(tokens,topk))
    reshape('w13','w13_rows',(tokens,topk,2*intermediate))
    reshape('activation','activation_rows',(tokens,topk,intermediate))
    reshape('down','down_rows',(tokens,topk,hidden))
    if not metadata_ready:
        accesses.append(TileAccessTemplate('routing','routing_rows',Operation.READ,routed,routed,(ready['routing'],),(),evidence,unmapped=True))
    for role in ('w13_weight','w13_scale','w2_weight','w2_scale'):
        parent=views[role]
        selected.append(LogicalSelectionView(f'{role}.selected',parent.allocation_id,(tokens,topk,parent.shape[1]),
            parent.dtype,parent.element_bytes,f'{scope.scope_id}.{role}.expert_from_actual_routed_id',role))
    def read(stage,role,view_id,mapping,dependencies,unmapped=False):
        identity=f'{stage}.{role}'
        unresolved=unmapped or view_id.endswith('_rows') or views[role].allocation_id in unknown_allocations
        accesses.append(TileAccessTemplate(identity,view_id,Operation.READ,routed,mapping,
            (ready[role],) if role in ready else (),tuple(dependencies),evidence,unmapped=unresolved))
        return dep(identity,EventKind.READ_READY)
    def write(stage,role,dependencies):
        identity=f'{stage}.{role}'
        accesses.append(TileAccessTemplate(identity,f'{role}_rows',Operation.WRITE,routed,(*routed,None),(),tuple(dependencies),evidence,unmapped=True))
        return identity
    routing=metadata_ready[0] if metadata_ready else (dep('routing',EventKind.READ_READY),)
    inputs=[read('w13','query','query',('token',None),routing)]
    inputs.extend(read('w13',role,f'{role}.selected',(*routed,None),routing,True) for role in ('w13_weight','w13_scale'))
    write('w13','w13',inputs)
    w13_barrier=(dep('w13.w13',EventKind.WRITE_VISIBLE,routed),)
    loaded=read('activation','w13','w13_rows',(*routed,None),w13_barrier)
    write('activation','activation',(loaded,))
    activation_barrier=(dep('activation.activation',EventKind.WRITE_VISIBLE,routed),)
    w2_routing=metadata_ready[1] if metadata_ready else routing
    inputs=[read('w2','activation','activation_rows',(*routed,None),(*activation_barrier,*w2_routing)),
        read('w2','router_weight','router_weight',routed,(*w2_routing,*activation_barrier))]
    inputs.extend(read('w2',role,f'{role}.selected',(*routed,None),(*w2_routing,*activation_barrier),True)
        for role in ('w2_weight','w2_scale'))
    write('w2','down',inputs)
    accesses.append(TileAccessTemplate('sum.read','down_rows',Operation.READ,routed,(*routed,None),(),
        (dep('w2.down',EventKind.WRITE_VISIBLE,routed),),evidence,unmapped=True))
    accesses.append(TileAccessTemplate('sum.output','output',Operation.WRITE,('token',),('token',None),(),
        (dep('sum.read',EventKind.READ_READY,('topk',)),),evidence,unmapped=views['output'].allocation_id in unknown_allocations))
    return tuple(selected),TileProgram(scope.scope_id,scope,axes,tuple(accesses),visibility,evidence)


def _routed_alignment_components(views,allocations,ready,scope,visibility,binding_id,block,counts):
    """Inline the no-map alignment producer and its public reads.

    Both vLLM 0.30 branches share the same public control interface. Native
    persistent-stripe replay is separate from these logical interface reads.
    """
    from dataclasses import replace
    from src.kernel.memory_access import LogicalSelectionView,TileAxis,TileAccessTemplate,TileDependency
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_moe_alignment import _small_components,_cached_regular_components
    rows=views['routing'].shape[0];experts=len(counts)
    small=rows<1024 and experts<=64
    if -(-experts//32)*32>=1024:return (),(),(),(),(),()
    capacity=rows+experts*(block-1)
    if rows<experts:capacity=min(capacity,rows*block)
    prefix=f'{scope.scope_id}.alignment';public=[];events=[]
    parents=[replace(views['routing'],view_id='topk_ids')]
    device=allocations[views['routing'].allocation_id].device_id
    layouts=(('sorted_token_ids',capacity),('expert_ids',-(-capacity//block)),('num_tokens_post_padded',1))
    if not small:layouts=(*layouts,('cumsum',experts+1))
    for name,count in layouts:
        identity=f'{binding_id}.alignment.{name}'
        allocation=MemoryAllocation(identity,device,'logical-marlin-alignment',count*4,AddressLevel.OBJECT,None,None,
            identity+'.allocated',None)
        public.append(allocation);parents.append(TensorView(name,identity,0,(count,),(4,),'int32',4))
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
    binding=ElementwiseBindings(binding_id,scope.call_id,(allocations[views['routing'].allocation_id],*public),tuple(parents),(),
        ('topk_ids',),('sorted_token_ids','expert_ids','num_tokens_post_padded'),(ready['routing'],),scope.rank,scope.stream,visibility)
    if small:
        producer_scope=replace(scope,scope_id=prefix,agent='logical-marlin-small-alignment')
        reference=('vLLM 0.30 moe_align_block_size.py public capacity and csrc/libtorch_stable/moe/moe_align_sum_kernels.cu no-map small branch; 256 fill workers plus max(E,32) counting workers.',
            'Private expert counters, prefix and second-ID pass reuse the shared CUDA small-alignment source; no native Marlin stripe issue-count claim.')
        scratch,selected,internal,producer=_small_components((rows,experts,max(experts,32),block,capacity,counts),binding,producer_scope,reference)
    else:
        producer_scope=replace(scope,scope_id=prefix,agent='logical-marlin-regular-alignment')
        reference=('vLLM 0.30 no-map ordinary alignment: independent fill and 1024-worker histogram/BlockScan, followed by a separate atomic count-and-sort kernel.',
            'Versioned CUB raking dataflow is a logical algorithm reference, not a native binary layout or Marlin stripe issue-count claim.')
        scratch,selected,internal,producer=_cached_regular_components((rows,experts,block,capacity,counts),binding,producer_scope,reference)
        extra_views,sort_accesses=_routed_regular_sort_components(parents,ready,producer,scope,visibility,reference,
            unresolved_input=allocations[views['routing'].allocation_id].size_bytes is None)
        selected=(*selected,*extra_views)
        producer=replace(producer,accesses=(*producer.accesses,*sort_accesses))
    public_views=unique_by(parents,'view_id');control_views=[];controls=[]
    padded=sum(-(-n//block)*block for n in counts);blocks=padded//block
    axes=(*producer.axes,TileAxis('marlin_block',blocks,1))
    def completion(writer):return TileDependency(writer.access_id,EventKind.WRITE_VISIBLE,writer.iteration_axes)
    # Native helpers expose a completed tensor partition. Reads keep that
    # partition's actual producers; no initialized event is invented for it.
    sorted_writers=tuple(completion(a) for a in producer.accesses if a.access_id.startswith('sorted') or a.access_id=='scatter')
    owner_writers=tuple(completion(a) for a in producer.accesses
        if (a.access_id.startswith('owner') and a.operation is Operation.WRITE) or a.access_id=='inactive_blocks')
    for name,shape,parent,writers in (
            ('sorted',(blocks,block),'sorted_token_ids',sorted_writers),
            ('expert',(blocks,),'expert_ids',owner_writers)):
        source=public_views[parent]
        selected_view=LogicalSelectionView(prefix+'.'+name,source.allocation_id,shape,'int32',4,
            prefix+'.'+name+'.active_prefix',parent)
        control_views.append(selected_view)
        for stage in ('w13','w2'):
            boundary=(TileDependency('activation.activation',EventKind.WRITE_VISIBLE,('token','topk')),) if stage=='w2' else ()
            controls.append(TileAccessTemplate(stage+'.metadata.'+name,selected_view.view_id,Operation.READ,('marlin_block',),
                ('marlin_block',None) if name=='sorted' else ('marlin_block',),(),(*writers,*boundary),reference,unmapped=True))
    for stage in ('w13','w2'):
        boundary=(TileDependency('activation.activation',EventKind.WRITE_VISIBLE,('token','topk')),) if stage=='w2' else ()
        controls.append(TileAccessTemplate(stage+'.metadata.padded_count','num_tokens_post_padded',Operation.READ,(),(None,),(),
            (TileDependency('padded_count',EventKind.WRITE_VISIBLE,()),*boundary),reference))
    dependencies=tuple(tuple(TileDependency(a.access_id,EventKind.READ_READY,(),
        partition_axis='topk' if a.iteration_axes else None) for a in controls if a.access_id.startswith(stage+'.'))
        for stage in ('w13','w2'))
    return (*public,*scratch),(*parents,*selected,*control_views),(*events,*internal),axes,(*producer.accesses,*controls),dependencies


def _routed_regular_sort_components(parents,ready,producer,scope,visibility,reference,*,unresolved_input):
    from dataclasses import replace
    from src.kernel.memory_access import TileDependency
    from src.kernel.memory_fused_moe import _moe_count_sort_tiles
    public=unique_by(parents,'view_id')
    def completion(a):
        return TileDependency(a.access_id,EventKind.READ_READY if a.operation is Operation.READ else EventKind.WRITE_VISIBLE,a.iteration_axes)
    prefix=tuple(completion(a) for a in producer.accesses if a.access_id.startswith('prefix') and a.operation is Operation.WRITE)
    boundary=tuple(completion(a) for a in producer.accesses if a.access_id in ('sorted_initialize','inactive_start','inactive_blocks','padded_count')
        or a.access_id.startswith('owner'))
    selected,sort=_moe_count_sort_tiles(public,{'topk_ids':ready['routing']},scope,visibility,reference,
        cumsum_dependencies=prefix,topk_unmapped=unresolved_input)
    # A second kernel starts after both first-stage CTAs, including readers of
    # cumsum which atomic updates would otherwise race with. No actor order
    # is imposed among the second kernel's equal-key atomic operations.
    accesses=tuple(replace(a,dependencies=(*a.dependencies,*boundary),
        scope=replace(scope,scope_id=scope.scope_id+'.sort.ids',agent='logical-marlin-sort-global')) if a.access_id=='expert'
        else replace(a,scope=replace(scope,scope_id=scope.scope_id+'.sort.'+a.access_id,agent='logical-marlin-sort-global'))
        for a in sort.accesses)
    return selected,accesses


def _routed_native_config(output_columns,reduction_columns,routed_rows,block,sms,max_shared_bytes):
    """Prove the pinned selector's choice without supplying compiler registers.

    Every candidate has thread_n >=64. When its workload occupancy cap is
    below two, the selector clamps every valid candidate to one block/SM,
    regardless of numRegs. Strict greater-than priority then keeps the first
    valid compiled configuration. Other workloads remain explicitly unknown.
    """
    if 4*(output_columns//64)*routed_rows>=2*sms:return None
    thread_k,thread_n,threads=(128,128,256) if block<=16 else (64,256,256)
    if output_columns%thread_n or reduction_columns%thread_k:return None
    stages=4;physical_m=-(-block//16)*16
    sh_b=stages*thread_k*thread_n//2
    sh_red=physical_m*(thread_n+8)*2
    temporary=max(sh_b,sh_red,min(sh_b,sh_red)+thread_n*2)
    cache_bytes=(temporary+stages*physical_m*thread_k*2
        +stages*(-(-thread_k//32))*thread_n*2+physical_m*16)
    if cache_bytes>max_shared_bytes-512:return None
    return (thread_k,thread_n,threads,stages,1)


def _routed_native_config_facts(views,block,sms,max_shared_bytes):
    tokens,hidden=views['query'].shape;rows,intermediate=views['activation'].shape
    return {stage:_routed_native_config(n,k,rows,block,sms,max_shared_bytes)
        for stage,n,k in (('w13',2*intermediate,hidden),('w2',hidden,intermediate))}


def _routed_single_panel_scratch(views,scope,block,counts,sms,max_shared_bytes):
    """Actual first-to-last FP32 fragment versions in the two-slice domain.

    No configuration is guessed: the selector must be register-independent,
    and every active CTA owns one K panel. BM8 stores all eight scratch rows,
    including inactive output rows, as required by global_reduce_fp32.
    """
    from dataclasses import replace
    from src.kernel.memory_access import LogicalSelectionView,TileAxis,TileAccessTemplate,TileDependency
    configs=_routed_native_config_facts(views,block,sms,max_shared_bytes)
    blocks=sum(-(-value//block) for value in counts)
    selected=[];axes=[];accesses=[];covered={}
    for stage,n,k in (('w13',views['w13'].shape[1],views['query'].shape[1]),
            ('w2',views['down'].shape[1],views['activation'].shape[1])):
        config=configs[stage]
        valid=(block==8 and config is not None and k//config[0]<=2 and blocks*(n//config[1])*(k//config[0])<=sms)
        covered[stage]=valid
        if not valid or k==config[0]:continue
        tk,tn,threads,stages,bpsm=config
        column=stage+'.native_column';axes.append(TileAxis(column,n//tn,1))
        role=stage+'.c_tmp';parent=views[role];identity=stage+'.native_partial_versions'
        view=LogicalSelectionView(identity,parent.allocation_id,(blocks,n//tn,8,tn),'fp32',4,
            'marlin.single_panel.first_to_last_fragments.'+stage,role)
        selected.append(view)
        evidence=('Pinned default non-atomic FP32 reduce, BM8: first contributor stores even int4 fragment groups; the last contributor reads those exact logical versions. Physical fragment permutation remains unknown.',
            'The workload-capped selector proves (thread_k,thread_n,threads,stages,blocks_per_sm); grid=SM and K-panels*MN-tiles<=SM make each active CTA own one K panel.',
            'This supplements the logical operand interface; complete native replay and acquire polling remain separately pending.')
        prefix='runtime.'+stage+'.scratch';context=('marlin_block',column)
        operands=(stage+'.activation' if stage=='w2' else stage+'.query',stage+'.'+stage+'_weight',stage+'.'+stage+'_scale')
        dependencies=tuple(TileDependency(name,EventKind.READ_READY,('token','topk'),partition_axis='topk') for name in operands)
        actor=replace(scope,scope_id=scope.scope_id+'.'+prefix,agent='logical-marlin-native-fp32-fragments')
        accesses.append(TileAccessTemplate(prefix+'.write',identity,Operation.WRITE,context,(*context,None,None),(),dependencies,evidence,unmapped=True,scope=actor))
        accesses.append(TileAccessTemplate(prefix+'.read',identity,Operation.READ,context,(*context,None,None),(),
            (TileDependency(prefix+'.write',EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True,scope=actor))
    return tuple(selected),tuple(axes),tuple(accesses),covered


def _routed_runtime_components(views,allocations,scope,binding_id,block,sms,max_shared_bytes):
    """Known default-wrapper storage and zeroing, without inventing traffic.

    C_tmp and subsequent lock traffic remain unbound. Allocation capacity is
    not a request count or evidence that any global reduction actually ran.
    """
    from dataclasses import replace
    from src.kernel.memory_access import ConstantValue,TileAxis,TileAccessTemplate
    device=allocations[views['routing'].allocation_id].device_id
    capacity=views['sorted_token_ids'].shape[0]
    max_blocks_per_sm=4;max_thread_n=256
    output_sizes=(('w13',views['w13'].shape[1]),('w2',views['down'].shape[1]))
    layouts=(('marlin.workspace',sms*max_blocks_per_sm,'int32','logical-marlin-lock-workspace'),
        *((f'{stage}.c_tmp',min(n*capacity,sms*max_blocks_per_sm*block*max_thread_n)*(2 if block==8 else 1),'fp32','logical-marlin-fp32-scratch')
            for stage,n in output_sizes))
    created=[];selected=[];events=[]
    for role,count,dtype,domain in layouts:
        identity=f'{binding_id}.{role}'
        allocation=MemoryAllocation(identity,device,domain,count*4,AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        created.append(allocation);selected.append(TensorView(role,identity,0,(count,),(4,),dtype,4))
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
    source=('vLLM 0.30 MarlinExperts.apply passes no external workspace; _fused_marlin_moe creates marlin_make_workspace_new(device,4), whose default path is torch.zeros(INT32).',
        'ops.cu allocates separate FP32 C_tmp per GEMM using min(N*sorted_capacity,SM*4*BM*max_thread_n), doubled for BM8; marlin.cuh defines max_thread_n=256.',
        'Workspace initialization is explicit. Compiler-independent BM8 one-panel/two-contributor C_tmp fragments are generated separately; acquire polling and other native configurations remain unbound, not zero.')
    configurations=_routed_native_config_facts(views,block,sms,max_shared_bytes)
    facts=('marlin_default_workspace',str(sms),str(max_shared_bytes),*(
        ','.join(map(str,configurations[stage])) if configurations[stage] is not None else 'unbound' for stage in ('w13','w2')))
    actor=replace(scope,scope_id=scope.scope_id+'.workspace.zero',op_ids=facts,agent='logical-marlin-runtime')
    zero=TileAccessTemplate('runtime.workspace.zero','marlin.workspace',Operation.WRITE,('marlin_workspace_word',),
        ('marlin_workspace_word',),(),(),source,scope=actor,constant_value=ConstantValue.ZERO)
    return tuple(created),tuple(selected),tuple(events),(TileAxis('marlin_workspace_word',sms*max_blocks_per_sm,1),),(zero,)


def _routed_recipe(views,allocations,ready,scope,visibility,binding_id,block,counts,evidence,*,num_compute_units,max_shared_bytes):
    from dataclasses import replace
    from src.kernel.memory_access import TilePartition
    unknown=frozenset(a.allocation_id for a in allocations.values() if a.size_bytes is None)
    scratch,metadata_views,events,axes,accesses,dependencies=_routed_alignment_components(
        views,allocations,ready,scope,visibility,binding_id,block,counts)
    if dependencies:
        runtime_allocations,runtime_views,runtime_events,runtime_axes,runtime_accesses=_routed_runtime_components(
            {**views,**unique_by(metadata_views,'view_id')},allocations,scope,binding_id,block,num_compute_units,max_shared_bytes)
        from src.kernel.memory_access import TileDependency
        initialized=TileDependency('runtime.workspace.zero',EventKind.WRITE_VISIBLE,('marlin_workspace_word',))
        accesses=tuple(replace(a,dependencies=(*a.dependencies,initialized)) if a.access_id.startswith('w13.metadata.') else a for a in accesses)
        scratch=(*scratch,*runtime_allocations);metadata_views=(*metadata_views,*runtime_views)
        events=(*events,*runtime_events);axes=(*axes,*runtime_axes);accesses=(*accesses,*runtime_accesses)
    selected,family=_routed_moe_tiles(views,ready,scope,visibility,evidence,unknown,dependencies)
    if dependencies:
        native_views,native_axes,native_accesses,covered=_routed_single_panel_scratch(
            {**views,**unique_by(metadata_views,'view_id')},scope,block,counts,num_compute_units,max_shared_bytes)
        from src.kernel.memory_access import TileDependency
        completion={stage:TileDependency('runtime.'+stage+'.scratch.read',EventKind.READ_READY,(stage+'.native_column',),partition_axis='topk')
            for stage in ('w13','w2') if any(a.access_id=='runtime.'+stage+'.scratch.read' for a in native_accesses)}
        family=replace(family,accesses=tuple(replace(a,dependencies=(*a.dependencies,completion[stage]))
            if (stage:=a.access_id.split('.',1)[0]) in completion and a.access_id in ('w13.w13','w2.down') else a for a in family.accesses))
        selected=(*selected,*native_views);axes=(*axes,*native_axes);accesses=(*accesses,*native_accesses)
    partitions=(TilePartition('topk',class_axis_id='marlin_block',context_axes=('token',)),) if dependencies else ()
    return scratch,(*selected,*metadata_views),events,replace(family,axes=(*family.axes,*axes),
        accesses=(*family.accesses,*accesses),partitions=partitions)


def _build_routed_moe_tiles(kernel,hardware,bindings,views,ready,program_id):
    block=kernel.launch.tiling.tile_shape['m'];counts=kernel.attributes['local_tokens_per_expert']
    header=('routed_moe_rows',str(block),*(str(count) for count in counts))
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-marlin-wrapper')
    evidence=('vLLM0.30 experts/marlin_moe.py:_fused_marlin_moe W13→activation→W2→sum with shared cache13.',
        'Each original (token,top-k) row is a full-N/full-K mathematical vector dot. Native block-M and its padded rows remain separate facts.',
        'Packed/permuted weight and scale selections remain opaque; each matrix block requests its complete mathematical operands.',
        'All W13 results precede activation; all activation reads precede W2 cache13 overwrite; final token reductions are independent.',
        'No-map alignment metadata has actual producers and logical block consumers within the CUDA expert-range admission; native persistent-stripe replay, FP32 reduction scratch and locks remain unbound.')
    scratch,selected,events,family=_routed_recipe(views,unique_by(bindings.allocations,'allocation_id'),ready,scope,
        bindings.visibility_domain,bindings.binding_id,block,counts,evidence,num_compute_units=int(hardware.num_compute_units),
        max_shared_bytes=hardware.max_smem_per_block_bytes)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Logical routed-row matrix/activation/reduction tiles; no packed permutation, native worker or physical address claim.',
            'W13/down share cache13; actual whole-launch boundaries protect every expert activation read.',
            'Both no-map alignment branches are generated for padded expert count below 1024; original row→sorted-block membership stays unknown. Other expert domains, native stripe replay, FP32 scratch, locks and EP remain partial.',
            'Default workspace zeroing and separate C_tmp capacities are represented. The proven BM8 one-K-panel-per-CTA domain includes actual FP32 partial versions, including padded scratch rows; other C_tmp and all native lock/poll streams remain unbound. Statistics count only represented requests.',
            'Legacy compulsory weight projection retains the declared active-expert operands once; it asserts no cache residency.')),
        (*bindings.allocations,*scratch),(*bindings.views,*selected),(),(*bindings.events,*events),(),(family,))
    routed_moe_source_demands(program,expected_kernel=kernel,expected_hardware=hardware)
    return program


def _routed_source_events_present(actual, expected):
    """Compare complete event values once, using their unique native IDs."""
    if all(type(event) is ProgramEvent for event in (*actual, *expected)):
        events = unique_by(actual, 'event_id')
        return all(event.event_id in events and events[event.event_id] == event for event in expected)
    # Subclasses can supply different equality semantics. Preserve the
    # original membership checks for those contracts without inferring IDs.
    return all(event in actual for event in expected)


def routed_moe_source_demands(program, *, expected_kernel=None,expected_hardware=None,expected_bindings=None):
    """Validate the unique tile pipeline before projecting retained pricing."""
    from src.kernel.memory_access import _tile_template_requested_bytes
    program.__post_init__()
    if program.scopes or program.accesses or len(program.tile_programs)!=1:
        raise ValueError('routed wrapper requires its unique compressed row pipeline')
    family,=program.tile_programs;header=family.scope.op_ids;views=unique_by(program.views,'view_id')
    if (len(header)<3 or header[0]!='routed_moe_rows' or family.scope.agent!='logical-marlin-wrapper'):
        raise ValueError('routed wrapper tile identity mismatch')
    block=int(header[1]);counts=tuple(map(int,header[2:]));query=views['query'];tokens,hidden=query.shape
    rows,intermediate=views['activation'].shape;topk,remainder=divmod(rows,tokens)
    if (block not in (8,16,32,48,64) or remainder or not topk or len(counts)!=views['w13_weight'].shape[0]
            or min(counts)<0 or sum(counts)!=rows or not any(counts)
            or views['output'].shape!=query.shape
            or (views['w13'].allocation_id,views['w13'].offset_bytes)!=(views['down'].allocation_id,views['down'].offset_bytes)):
        raise ValueError('routed wrapper row conservation/shared workspace mismatch')
    if expected_kernel is not None:
        kernel=expected_kernel
        if (kernel.kind is not KernelKind.FUSED_MOE or kernel.variant!='routed_experts'
                or kernel.hardware_id!=program.hardware_id or kernel.execution_support!='supported'
                or kernel.launch.algorithm not in ('vllm_marlin_moe_w4a16_aggregate','vllm_marlin_moe_mxfp4_aggregate')
                or kernel.attributes['expert_parallel_size']!=1
                or counts!=kernel.attributes['local_tokens_per_expert']
                or block!=kernel.launch.tiling.tile_shape['m']
                or _routed_layouts_from_views(views)!=routed_moe_layouts(kernel)):
            raise ValueError('routed wrapper source differs from its actual kernel')
    layouts=_routed_layouts_from_views(views)
    if len({views[role].allocation_id for role,_,_,_ in layouts})!=len(layouts)-1:
        raise ValueError('routed wrapper payload parents require only the declared cache13 alias')
    for role,shape,dtype,width in layouts:
        view=views[role];strides=[];size=width
        for extent in reversed(shape):strides.append(size);size*=extent
        if not isinstance(view,TensorView) or (view.shape,view.dtype,view.element_bytes,view.strides_bytes)!=(shape,dtype,width,tuple(reversed(strides))):
            raise ValueError('routed wrapper operand storage geometry mismatch')
    templates=unique_by(family.accesses,'access_id');ready={}
    metadata_bound=-(-len(counts)//32)*32<1024
    sms=max_shared_bytes=None
    if metadata_bound:
        if 'runtime.workspace.zero' not in templates or templates['runtime.workspace.zero'].scope is None:
            raise ValueError('routed wrapper runtime workspace source is missing')
        facts=templates['runtime.workspace.zero'].scope.op_ids
        if len(facts)!=5 or facts[0]!='marlin_default_workspace':raise ValueError('routed wrapper runtime hardware facts differ')
        sms=int(facts[1])
        max_shared_bytes=int(facts[2])
        if sms<=0 or str(sms)!=facts[1] or max_shared_bytes<=512 or str(max_shared_bytes)!=facts[2]:raise ValueError('routed wrapper runtime hardware capacity is invalid')
    if expected_hardware is not None:
        if (expected_hardware.hardware_id!=program.hardware_id
                or metadata_bound and (sms!=expected_hardware.num_compute_units or max_shared_bytes!=expected_hardware.max_smem_per_block_bytes)):
            raise ValueError('routed wrapper runtime hardware facts differ from actual HWSpec')
    routing_source=('count_id0' if rows<1024 and len(counts)<=64 else 'expert_read') if metadata_bound else 'routing'
    if routing_source not in templates or len(templates[routing_source].input_ready)!=1:
        raise ValueError('routed wrapper source input readiness missing')
    ready['routing']=templates[routing_source].input_ready[0]
    for stage,roles in (('w13',('query','w13_weight','w13_scale')),('w2',('router_weight','w2_weight','w2_scale'))):
        for role in roles:
            template=templates[f'{stage}.{role}']
            if len(template.input_ready)!=1:raise ValueError('routed wrapper source input readiness missing')
            ready[role]=template.input_ready[0]
    if expected_bindings is not None:
        from src.kernel.memory_access_builder import ElementwiseBindings
        binding=expected_bindings
        if (not isinstance(binding,ElementwiseBindings) or program.binding_id!=binding.binding_id
                or (family.scope.call_id,family.scope.rank,family.scope.stream,family.visibility_domain)
                    !=(binding.call_id,binding.rank,binding.stream,binding.visibility_domain)
                or binding.read_views!=('query','routing','router_weight','w13_weight','w13_scale','w2_weight','w2_scale')
                or binding.write_views!=('w13','activation','down','output')
                or program.views[:len(binding.views)]!=binding.views
                or program.allocations[:len(binding.allocations)]!=binding.allocations
                or program.events[:len(binding.events)]!=binding.events
                or tuple(ready[role] for role in binding.read_views)!=binding.input_ready):
            raise ValueError('Marlin source differs from its actual caller bindings and ordered input readiness')
    allocations=unique_by(program.allocations,'allocation_id')
    selected_allocations,selected,internal_events,expected=_routed_recipe(views,allocations,ready,family.scope,
        family.visibility_domain,program.binding_id,block,counts,family.evidence_refs,num_compute_units=sms,max_shared_bytes=max_shared_bytes)
    parent_roles={role for role,_,_,_ in layouts}
    if (family!=expected or set(views)!=parent_roles|{view.view_id for view in selected}
            or any(views[view.view_id]!=view for view in selected)
            or any(allocations[a.allocation_id]!=a for a in selected_allocations)
            or not _routed_source_events_present(program.events, internal_events)):
        raise ValueError('routed wrapper lost actual stage/scale/alias producer causality')
    axes=unique_by(family.axes,'axis_id')
    activation_read=output_write=packed_read=routing_read=0
    for access in family.accesses:
        if access.access_id.startswith('count_id') or access.access_id=='expert_read':
            # Retain the former single logical routing-pass price using the
            # actual first ID pass, without retaining a phantom wrapper read.
            requested=_tile_template_requested_bytes(views,axes,family,access)
            activation_read+=requested;routing_read+=requested
            continue
        if access.access_id not in ('routing','w13.query','w13.w13_weight','w13.w13_scale','w13.w13',
                'activation.w13','activation.activation','w2.activation','w2.router_weight','w2.w2_weight','w2.w2_scale',
                'w2.down','sum.read','sum.output'):continue
        amount=_tile_template_requested_bytes(views,axes,family,access)
        if access.operation is Operation.WRITE:output_write+=amount
        elif access.access_id in ('w13.w13_weight','w13.w13_scale','w2.w2_weight','w2.w2_scale'):
            # Compulsory pricing retains each actual expert operand once.
            packed_read+=sum(count>0 for count in counts)*(amount//rows)
        else:activation_read+=amount
    active=sum(count>0 for count in counts)
    if packed_read%active:raise ValueError('routed wrapper expert weight payload mismatch')
    metadata_reads=metadata_writes=0
    for access in family.accesses:
        is_global=(access.scope is not None and access.scope.agent in (
            'logical-small-alignment-global','logical-regular-alignment-global','logical-marlin-sort-global')
            or '.metadata.' in access.access_id)
        if not is_global:continue
        amount=_tile_template_requested_bytes(views,axes,family,access)
        if access.operation is Operation.READ:metadata_reads+=amount
        else:metadata_writes+=amount
    native_covered=(_routed_single_panel_scratch(views,family.scope,block,counts,sms,max_shared_bytes)[3] if metadata_bound else None)
    scratch_complete=native_covered is not None and all(native_covered.values())
    scratch_totals={operation:sum(_tile_template_requested_bytes(views,axes,family,a) for a in family.accesses
        if a.access_id.startswith('runtime.') and '.scratch.' in a.access_id and a.operation is operation) for operation in (Operation.READ,Operation.WRITE)}
    return {'routing':(counts,counts),'routed_traffic':{'activation_read':activation_read,'output_write':output_write,
        'weight_compulsory':packed_read,'weight_per_expert':packed_read//active},'geometry':(tokens,topk,hidden,intermediate),
        'alignment_metadata':{'logical_global_read_bytes':metadata_reads if metadata_bound else None,
            'logical_global_write_bytes':metadata_writes if metadata_bound else None,
            'retained_routing_pass_read_bytes':routing_read if metadata_bound else None,
            'native_stripe_replay_bound':False,'additional_metadata_in_retained_price':False},
        'runtime_storage':{'known_workspace_zero_write_bytes':_tile_template_requested_bytes(views,axes,family,templates['runtime.workspace.zero']) if metadata_bound else None,
            'compiler_independent_configurations':_routed_native_config_facts(views,block,sms,max_shared_bytes) if metadata_bound else None,
            'single_panel_scratch_covered':native_covered,
            'c_tmp_capacity_bytes':{stage:allocations[views[stage+'.c_tmp'].allocation_id].size_bytes for stage in ('w13','w2')} if metadata_bound else None,
            'native_workspace_read_bytes':None,'native_workspace_update_write_bytes':None,
            'native_c_tmp_read_bytes':scratch_totals[Operation.READ] if scratch_complete else None,
            'native_c_tmp_write_bytes':scratch_totals[Operation.WRITE] if scratch_complete else None,
            'additional_runtime_in_retained_price':False}}


def _routed_layouts_from_views(views):
    """Canonical descriptor geometry from the source's typed public parents."""
    tokens,hidden=views['query'].shape;rows,intermediate=views['activation'].shape;experts=views['w13_weight'].shape[0]
    topk,remainder=divmod(rows,tokens)
    if remainder:raise ValueError('routed wrapper requires a complete top-k row axis')
    scale=views['w13_scale']
    if (scale.dtype,scale.element_bytes) not in (('bf16',2),('uint8',1)):
        raise ValueError('routed wrapper packed scale encoding mismatch')
    return _routed_layouts(tokens,experts,topk,hidden,intermediate,scale.dtype,scale.element_bytes)
