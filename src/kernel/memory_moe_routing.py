"""Source-declared GPU top-k rows and their actual public metadata outputs."""
from dataclasses import replace
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,
    MemoryAccessProgram,MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,
    TensorView,TileAxis,TileAccessTemplate,TileDependency,TileProgram,unique_by,
)


_METADATA_VARIANTS=('moe_router_logit_cast','moe_routed_id_widen','moe_routed_id_extent','moe_expert_map_lookup')


def has_moe_metadata_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant in _METADATA_VARIANTS
        and kernel.execution_support=='supported' and hardware.kernel_memory_model=='gpu_mshr')


def _moe_row_extents(kernel):
    logical=kernel.shape['tokens']
    if 'model_input_shape' not in kernel.attributes:return logical,logical
    physical=kernel.attributes['model_input_shape'];attributes=kernel.attributes
    execution=physical['tokens']
    if (any(physical[name]!=kernel.shape[name] for name in physical if name not in ('tokens','element_count'))
            or execution<logical
            or (execution!=logical and (attributes['graph_runtime_mode'] not in ('FULL','PIECEWISE')
                or execution*attributes['graph_logical_tokens']!=logical*attributes['graph_model_input_tokens']))
            or ('element_count' in physical and physical['element_count']!=kernel.shape['element_count']
                and physical['element_count']*attributes['graph_logical_tokens']
                    !=kernel.shape['element_count']*attributes['graph_model_input_tokens'])):
        raise ValueError('MoE metadata requires unchanged non-row geometry and its actual replay row mapping')
    return logical,execution


def _moe_replay_header(kernel,base):
    logical,execution=_moe_row_extents(kernel)
    return base if logical==execution else (*base,f'logical_tokens={logical}',f'execution_tokens={execution}')


def _metadata_layouts(kernel,*,execution=True):
    tokens,experts,topk=(kernel.shape[name] for name in ('tokens','experts','topk'))
    if execution:tokens=_moe_row_extents(kernel)[1]
    variant=kernel.variant
    if variant=='moe_router_logit_cast':return {'input':((tokens,experts),'bf16',2),'output':((tokens,experts),'fp32',4)}
    if variant=='moe_routed_id_widen':return {'input':((tokens,topk),'int32',4),'output':((tokens,topk),'int64',8)}
    if variant=='moe_routed_id_extent':return {'input':((tokens,topk),'int64',8),'output':((1,),'int64',8)}
    if variant=='moe_expert_map_lookup':
        block=16 if tokens<=experts else 64
        blocks=-(-(tokens*topk+experts*(block-1))//block)
        return {'expert_ids':((blocks,),'int32',4),'expert_map':((experts,),'int32',4),'output':((blocks,),'int32',4)}
    raise ValueError('Unknown MoE metadata tensor ABI')


def standalone_moe_metadata_bindings(kernel,*,call_id,binding_id, device_id="logical-local-device", rank=0, stream="offline"):
    from math import prod
    from src.kernel.memory_access_builder import ElementwiseBindings
    layouts=_metadata_layouts(kernel);reads=tuple(role for role in layouts if role!='output')
    allocations=[];views=[];events=[];ready=[]
    for role,(shape,dtype,width) in layouts.items():
        identity=f'{binding_id}.{role}';strides=(shape[1]*width,width) if len(shape)==2 else (width,)
        allocation=MemoryAllocation(identity,device_id,'logical-moe-metadata',prod(shape)*width,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        view=TensorView(role,identity,0,shape,strides,dtype,width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role in reads:
            event=ProgramEvent(identity+'.ready',EventKind.INITIALIZED,identity,view.region(),'moe-metadata-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),reads,('output',),
        tuple(ready),rank,stream,'moe-metadata-visible')


def _metadata_tiles(kernel,views,ready,scope,visibility):
    from src.kernel.memory_access import LogicalSelectionView,TileValueIndex
    variant=kernel.variant
    evidence=('vLLM legacy fused_topk converts BF16 gating_output to FP32; routed-ID conversion is INT32 to INT64 before PyTorch bincount range reductions.',
        'Each extent invocation reads the INT64 routed tensor once and writes one INT64 scalar; the two recipe invocations remain distinct.',
        'moe_align_block_size returns INT32 expert_ids and applies expert_map[expert_ids] over the full padded-block tensor.',
        'Scalar semantic tiles retain hidden tail counts; no native scheduling, reduction scratch, address values or cache-hit behavior is inferred.')
    if variant=='moe_expert_map_lookup':
        blocks=views['expert_ids'].shape[0];axes=(TileAxis('block',blocks,1),)
        selection=LogicalSelectionView('map_selection',views['expert_map'].allocation_id,(blocks,),'int32',4,
            scope.scope_id+'.map_by_expert_id','expert_map')
        accesses=(TileAccessTemplate('expert_ids','expert_ids',Operation.READ,('block',),('block',),(ready['expert_ids'],),(),evidence),
            TileAccessTemplate('map_selection','map_selection',Operation.READ,('block',),('block',),(ready['expert_map'],),
                (TileDependency('expert_ids',EventKind.READ_READY,()),),evidence,unmapped=True,
                value_index=TileValueIndex('expert_ids','loaded_scalar')),
            TileAccessTemplate('output','output',Operation.WRITE,('block',),('block',),(),
                (TileDependency('map_selection',EventKind.READ_READY,()),),evidence))
        return TileProgram(scope.scope_id,scope,axes,accesses,visibility,evidence),(selection,)
    rows,columns=views['input'].shape;axes=(TileAxis('token',rows,1),TileAxis('column',columns,1))
    read=TileAccessTemplate('input','input',Operation.READ,('token','column'),('token','column'),(ready['input'],),(),evidence)
    reduced=variant=='moe_routed_id_extent'
    write=TileAccessTemplate('output','output',Operation.WRITE,() if reduced else ('token','column'),
        (None,) if reduced else ('token','column'),(),
        (TileDependency('input',EventKind.READ_READY,('token','column') if reduced else ()),),evidence)
    return TileProgram(scope.scope_id,scope,axes,(read,write),visibility,evidence),()


def metax_extent_single_wave_geometry(kernel,hardware):
    """Prove the no-workspace all-to-one INT64 MACA 3.7.2 branch.

    The installed ReduceConfigMaca vectorizes a contiguous INT64 input
    above 128 values, up to a 16-byte load, retaining exact divisibility.
    An actual one-wave block with one output needs neither block-shared
    reduction nor global partials/semaphores. For one to four elements,
    ReduceNoStruct uses its CUDA configuration and the same no-shared
    lane/register pattern. Other branches stay unbound.
    """
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant!='moe_routed_id_extent'
            or kernel.hardware_id!='metax_c500' or hardware.hardware_id!=kernel.hardware_id
            or hardware.kernel_memory_model!='gpu_mshr'
            or kernel.execution_support!='supported'
            or kernel.launch.algorithm!='torch_metax_routed_id_extent_reduce'):
        return None
    elements=_moe_row_extents(kernel)[1]*kernel.shape['topk']
    vector=2 if elements>128 and elements%2==0 else 1
    lanes=min(512,1<<((elements//vector).bit_length()-1))
    if (lanes>hardware.parallel_width or kernel.launch.block!=(lanes,1,1)
            or kernel.launch.grid!=(1,1,1)):
        return None
    return {'input_elements':elements,'input_vector_elements':vector,'lanes':lanes,
        'shared_bytes':0,'global_partial_bytes':0,'semaphore_bytes':0,
        'state':('MACA3.7.2 ReduceNoStruct: INT64 lane registers and warp shuffle; one final INT64 output' if elements<=4 else
            'INT64 lane registers and warp shuffle; one final INT64 output')}


def metax_extent_state_geometry(kernel,hardware):
    """Static MACA3.7.2 scalar INT64 all-to-one configuration, not pricing launch."""
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant!='moe_routed_id_extent'
            or kernel.hardware_id!='metax_c500' or hardware.hardware_id!=kernel.hardware_id
            or hardware.kernel_memory_model!='gpu_mshr' or kernel.execution_support!='supported'
            or kernel.launch.algorithm not in ('torch_metax_routed_id_extent_reduce','moe_routed_id_extent')):
        return None
    elements=_moe_row_extents(kernel)[1]*kernel.shape['topk']
    # TensorIterator indexes element starts: numel and 1+max byte offset
    # must each be below 2**31. A contiguous 2-GiB INT64 tensor still fits.
    if elements<=4 or elements>=2**31 or 1+(elements-1)*8>=2**31:
        return None  # ReduceNoStruct / TensorIterator 32-bit partitioning are separate.
    vector=2 if elements>128 and elements%2==0 else 1
    lanes=min(512,1<<((elements//vector).bit_length()-1))
    values=-(-elements//lanes)
    ctas=max(min(hardware.num_compute_units*4,-(-values//16)),-(-values//256)) if values>=256 else 1
    analytic=(min(1<<(elements.bit_length()-1),hardware.parallel_width),1,1)
    generic=(kernel.launch.algorithm=='moe_routed_id_extent' and kernel.launch.block==(256,1,1)
        and kernel.launch.grid==(-(-elements//256),1,1))
    if not generic and (kernel.launch.block,kernel.launch.grid) not in ((analytic,(1,1,1)),((lanes,1,1),(1,ctas,1))):
        return None
    return {'input_elements':elements,'vector':vector,'lanes':lanes,'ctas':ctas,
        'shared_bytes_per_cta':lanes*8+8 if lanes>hardware.parallel_width else 0,
        'global_partial_capacity':ctas*8+8 if ctas>1 else 0,'semaphore_bytes':4 if ctas>1 else 0}


def cuda_extent_state_geometry(kernel,hardware):
    """Installed CUDA2.8 scalar INT64 configuration, separate from pricing."""
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant!='moe_routed_id_extent'
            or kernel.hardware_id not in ('nvidia_a800','nvidia_h200') or hardware.hardware_id!=kernel.hardware_id
            or hardware.kernel_memory_model!='gpu_mshr' or kernel.execution_support!='supported'
            or kernel.launch.algorithm!='moe_routed_id_extent'):
        return None
    elements=_moe_row_extents(kernel)[1]*kernel.shape['topk']
    if elements>=2**31 or 1+(elements-1)*8>=2**31:return None
    if kernel.launch.block!=(256,1,1) or kernel.launch.grid!=(-(-elements//256),1,1):return None
    vector=4 if elements>128 else 1
    lanes=min(512,1<<((elements//vector).bit_length()-1))
    values=-(-elements//lanes);block_y=values>=16
    target=hardware.num_compute_units*(hardware.max_threads_per_unit//lanes)
    ctas=max(min(target,-(-values//16)),-(-values//256)) if block_y and values>=256 else 1
    return {'input_elements':elements,'vector':vector,'lanes':lanes,'ctas':ctas,
        'block_y_reduce':block_y,'shared_bytes_per_cta':lanes*8 if lanes>hardware.parallel_width or block_y else 0,
        'global_partial_capacity':ctas*8 if ctas>1 else 0,'semaphore_bytes':4 if ctas>1 else 0}


def _build_metax_extent_state(kernel,hardware,bindings,*,program_id):
    return _build_extent_state(kernel,hardware,bindings,program_id=program_id,profile='maca')


def cuda_extent_partition_geometry(kernel,hardware):
    """TensorIterator launches, each retaining compressed native CTA tiles.

    Installed TensorIterator splits the reduced dimension at floor(n/2),
    yields the left half first, and accumulates the right half in output.
    A child is split again only when its own contiguous indexing exceeds
    32 bits. This enumerates native launches, never elements or CTA tiles.
    """
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant!='moe_routed_id_extent' or kernel.hardware_id not in ('nvidia_a800','nvidia_h200')
            or hardware.hardware_id!=kernel.hardware_id or hardware.kernel_memory_model!='gpu_mshr'
            or kernel.execution_support!='supported' or kernel.launch.algorithm!='moe_routed_id_extent'):
        return None
    n=_moe_row_extents(kernel)[1]*kernel.shape['topk']
    if n<=2**28 or kernel.launch.block!=(256,1,1) or kernel.launch.grid!=(-(-n//256),1,1):return None
    return _extent_partition_slices(n)


def _extent_partition_slices(elements):
    """Depth-first left-first TensorIterator::split on its reduced dimension."""
    pending=[(0,elements)];pieces=[]
    while pending:
        offset,size=pending.pop()
        if size<=2**28:
            pieces.append((offset,size))
        else:
            left=size//2
            pending.extend(((offset+left,size-left),(offset,left)))
    return tuple(pieces)


def metax_extent_partition_geometry(kernel,hardware):
    """All-to-one children retain the ordinary MACA continuous dispatch."""
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant!='moe_routed_id_extent'
            or kernel.hardware_id!='metax_c500' or hardware.hardware_id!=kernel.hardware_id
            or hardware.kernel_memory_model!='gpu_mshr' or kernel.execution_support!='supported'):
        return None
    n=_moe_row_extents(kernel)[1]*kernel.shape['topk']
    if n<=2**28:return None
    generic=(kernel.launch.algorithm=='moe_routed_id_extent' and kernel.launch.block==(256,1,1)
        and kernel.launch.grid==(-(-n//256),1,1))
    analytical=(kernel.launch.algorithm=='torch_metax_routed_id_extent_reduce'
        and kernel.launch.block==(hardware.parallel_width,1,1) and kernel.launch.grid==(1,1,1))
    if not (generic or analytical):return None
    return _extent_partition_slices(n)


def _build_cuda_partitioned_extent(kernel,hardware,bindings,*,program_id):
    return _build_partitioned_extent(kernel,hardware,bindings,program_id=program_id,profile='cuda')


def _build_partitioned_extent(kernel,hardware,bindings,*,program_id,profile):
    """Compose actual launches and their scalar output RAW dependencies."""
    from src.kernel.memory_access import LogicalSelectionView
    if profile not in ('cuda','maca'):raise ValueError('Unknown partitioned extent profile')
    cuda=profile=='cuda'
    pieces=cuda_extent_partition_geometry(kernel,hardware) if cuda else metax_extent_partition_geometry(kernel,hardware)
    if pieces is None:raise ValueError('GPU extent partition geometry is unbound')
    original=unique_by(bindings.views,'view_id');input_view=original['input'];output=original['output']
    expected=_metadata_layouts(kernel)['input'][0]
    if (input_view.shape,input_view.strides_bytes)!=(expected,(expected[1]*8,8)):
        raise ValueError('Partitioned extent requires its actual contiguous input')
    allocations=list(bindings.allocations);events=list(bindings.events);views=list(bindings.views);axes=[];accesses=[]
    evidence=('Installed TensorIterator split(int) halves the reduced dimension, left first; right should_accumulate=true.',
        'Installed CUDA Reduce.cuh uses the public INT64 output as accumulator, without an extra accumulation buffer.',
        'Each 32-bit leaf is one native kernel invocation with compressed CTA ownership, shared tree and global joins; input tiles are never expanded.')
    if not cuda:
        evidence=('Installed MACA3.7.2 TensorIterator split halves the reduced dimension, left first, with right output accumulation.',
            'Reduce.cuh passes non-null acc_buf_ptr into each child; that disables transpose only. get_reduce_kernel_type still selects ordinary continuous all-to-one MACA reduction.',
            'INT64 can accumulate in the public output; AccumulationBuffer has no device payload. Each 32-bit leaf retains compressed MACA CTA ownership and state.')
    previous=None
    for index,(offset,size) in enumerate(pieces):
        prefix=f'p{index}.'
        shape={**kernel.shape,'tokens':size,'topk':1}
        for name in ('element_count','elements'):
            if name in shape:shape[name]=size
        child_launch=(kernel.launch if not cuda and kernel.launch.algorithm=='torch_metax_routed_id_extent_reduce'
            else replace(kernel.launch,grid=(-(-size//256),1,1)))
        child_kernel=replace(kernel,shape=shape,attributes={},launch=child_launch)
        child_binding=replace(bindings,binding_id=bindings.binding_id+'.p'+str(index),views=(
            replace(input_view,offset_bytes=input_view.offset_bytes+offset*8,shape=(size,1),strides_bytes=(8,8)),output))
        child=_build_extent_state(child_kernel,hardware,child_binding,program_id=program_id+'.p'+str(index),profile=profile)
        allocations.extend(a for a in child.allocations if a.allocation_id not in {p.allocation_id for p in bindings.allocations})
        events.extend(e for e in child.events if e.event_id not in {p.event_id for p in bindings.events})
        names={v.view_id:('output' if index==len(pieces)-1 and v.view_id=='output' else prefix+v.view_id) for v in child.views}
        for view in child.views:
            if names[view.view_id]=='output':continue
            if isinstance(view,LogicalSelectionView):
                view=replace(view,parent_view_id=None if view.parent_view_id is None else names[view.parent_view_id])
            views.append(replace(view,view_id=names[view.view_id]))
        family,=child.tile_programs
        axes.extend(replace(axis,axis_id=prefix+axis.axis_id) for axis in family.axes)
        output_accesses=tuple(a for a in family.accesses if a.operation is Operation.WRITE and a.view_id=='output')
        if len(output_accesses)!=1:raise ValueError('Partitioned extent requires one scalar result per child')
        final,=output_accesses
        for access in family.accesses:
            deps=tuple(replace(d,access_id=prefix+d.access_id,reduction_axes=tuple(prefix+a for a in d.reduction_axes)) for d in access.dependencies)
            if access.access_id==final.access_id and index:
                read=prefix+'accumulate_output'
                accesses.append(TileAccessTemplate(read,f'p{index-1}.output',Operation.READ,(),(None,),(),
                    (TileDependency(previous,EventKind.WRITE_VISIBLE,()),*deps),evidence))
                deps=(*deps,TileDependency(read,EventKind.READ_READY,()))
            accesses.append(replace(access,access_id=prefix+access.access_id,view_id=names[access.view_id],
                iteration_axes=tuple(prefix+a for a in access.iteration_axes),
                view_axes=tuple(None if a is None else prefix+a for a in access.view_axes),dependencies=deps,
                atomic_add=None if access.atomic_add is None else replace(access.atomic_add,read_access_id=prefix+access.atomic_add.read_access_id)))
        previous=prefix+final.access_id
    scope=AccessScope(program_id,('cuda_extent_partitioned' if cuda else 'metax_extent_partitioned',str(sum(size for _,size in pieces))),bindings.call_id,(),
        bindings.rank,bindings.stream,'logical-cuda-extent-state' if cuda else 'logical-metax-extent-state')
    family=TileProgram(program_id,scope,tuple(axes),tuple(accesses),bindings.visibility_domain,evidence)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,evidence),tuple(allocations),tuple(views),(),tuple(events),(),(family,))


def _build_extent_state(kernel,hardware,bindings,*,program_id,profile):
    """At most three CTA ownership groups; never enumerate lanes or input tiles."""
    from src.kernel.memory_access import ConstantValue,TileAtomicAdd,LogicalSelectionView,LogicalCoordinateBinding,LogicalCoordinateTerm
    if profile not in ('maca','cuda'):raise ValueError('Unknown native extent source profile')
    cuda=profile=='cuda'
    geometry=cuda_extent_state_geometry(kernel,hardware) if cuda else metax_extent_state_geometry(kernel,hardware)
    if geometry is None:raise ValueError('MetaX extent native state lacks its static dispatch binding')
    views=unique_by(bindings.views,'view_id');parents=unique_by(bindings.allocations,'allocation_id')
    source,output=views['input'],views['output'];rows,topk=_metadata_layouts(kernel)['input'][0]
    if ((source.shape,source.dtype,source.element_bytes,source.strides_bytes)!=((rows,topk),'int64',8,(topk*8,8))
            or (output.shape,output.dtype,output.element_bytes,output.strides_bytes)!=((1,),'int64',8,(8,))
            or source.allocation_id==output.allocation_id):
        raise ValueError('MetaX extent needs its actual distinct contiguous INT64 public tensors')
    device=parents[source.allocation_id].device_id
    if parents[output.allocation_id].device_id!=device:raise ValueError('MetaX extent public operands differ in device')
    allocations=list(bindings.allocations);events=list(bindings.events);selected=list(bindings.views);all_accesses=[];all_axes=[]
    n,w,c,v=(geometry[k] for k in ('input_elements','lanes','ctas','vector'));unit=w*v
    rounds,remainder=divmod(n,c*unit);full_tail,tail=divmod(remainder,unit)
    evidence=('Installed MACA3.7.2 ReduceConfigMaca and scalar InputPerOutputContinuousReduceKernel INT64 branch.',
        'Native static config is separate from the retained analytical pricing launch; no duration or observed launch is inferred.',
        'CTA j reads vector blocks j, j+C, j+2C; exact contiguous vectors and tails preserve each input byte once.',
        'Source __syncthreads joins each CTA input before shared writes. Block-x tree reads the upper half and rewrites only the lower half; the final wave stays in registers.',
        'Global partials publish before atomicAdd; the symbolic last finisher reads all partials once. No winning CTA or atomic order is invented.')
    if cuda:
        evidence=('Installed PyTorch2.8 CUDA Reduce.cuh scalar INT64 ReduceConfig/ReduceOp; generic pricing launch is retained.',
            'Exact logical contiguous input/vector tails are projected onto at most three ownership groups; native pointer alignment/issued transactions remain unbound.',
            'Source-text block-y stores precede block-x shared tree; native DCE/CSE is not asserted.',
            'The symbolic last CTA joins all partials/tickets and reuses its own shared arena; no winning CTA/address/order is invented.')
    def allocate(role,size,domain):
        identity=bindings.binding_id+'.'+role
        allocation=MemoryAllocation(identity,device,domain,size,AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        return identity
    partial=allocate('cta_partials',geometry['global_partial_capacity'],'logical-device-workspace') if c>1 else None
    counter=allocate('completion_counter',4,'logical-device-workspace') if c>1 else None
    flag=allocate('last_cta_flag',c,'logical-cta-shared') if c>1 else None
    shared=allocate('cta_shared',c*geometry['shared_bytes_per_cta'],'logical-cta-shared') if w>hardware.parallel_width else None
    if c>1:
        selected.extend((TensorView('cta_partials',partial,0,(c,),(8,),'int64',8),
            TensorView('completion_counter',counter,0,(1,),(4,),'int32',4),
            TensorView('last_cta_flag',flag,0,(c,),(1,),'bool',1)))
        init=TileAccessTemplate('counter_zero','completion_counter',Operation.WRITE,(),(None,),(),(),evidence,constant_value=ConstantValue.ZERO)
        all_accesses.append(init)
    partial_ready=[];ticket_ready=[];partial_groups=[]
    groups=[]
    if full_tail:groups.append((0,full_tail,unit))
    if tail:groups.append((full_tail,1,tail))
    consumed=full_tail+(1 if tail else 0)
    if consumed<c:groups.append((consumed,c-consumed,0))
    for group,(start,count,extra) in enumerate(groups):
        prefix=f'g{group}'
        accesses=[];cta_axis=prefix+'_cta';thread_axis=prefix+'_thread'
        def view(role,allocation,offset,shape,strides,dtype='int64',width=8):
            name=prefix+'.'+role;selected.append(TensorView(name,allocation,offset,shape,strides,dtype,width));return name
        def access(name,role,operation,mapping,dependencies=(),ready=(),loops=('cta',),atomic=None):
            identity=prefix+'.'+name
            rename=lambda a:cta_axis if a=='cta' else thread_axis if a=='thread' else a
            accesses.append(TileAccessTemplate(identity,role,operation,tuple(rename(a) for a in loops),tuple(rename(a) for a in mapping),ready,dependencies,evidence,atomic_add=atomic))
            return identity
        def dependency(identity,kind=EventKind.READ_READY,reduced=()):return TileDependency(identity,kind,tuple(cta_axis if a=='cta' else a for a in reduced))
        joined=[]
        if rounds:
            name=view('input_rounds',source.allocation_id,source.offset_bytes+start*unit*8,(count,rounds,w,v),
                (unit*8,c*unit*8,v*8,8))
            joined.append(dependency(access('input_rounds',name,Operation.READ,('cta',None,None,None),ready=bindings.input_ready)))
        if extra:
            name=view('input_tail',source.allocation_id,source.offset_bytes+(rounds*c+start)*unit*8,
                (count,extra),(unit*8,8))
            joined.append(dependency(access('input_tail',name,Operation.READ,('cta',None),ready=bindings.input_ready)))
        if not joined:raise ValueError('Native extent configured an idle CTA without input ownership')
        if shared is not None:
            stride=geometry['shared_bytes_per_cta'];name=view('shared_seed',shared,start*stride,(count,w),(stride,8))
            if cuda and geometry['block_y_reduce']:
                y_seed=access('shared_y_seed',name,Operation.WRITE,('cta',None),tuple(joined))
                seed_ready=(*joined,dependency(y_seed,EventKind.WRITE_VISIBLE))
            else:seed_ready=tuple(joined)
            previous=access('shared_seed',name,Operation.WRITE,('cta',None),seed_ready)
            offset=w//2
            while offset>=hardware.parallel_width:
                upper=view('shared_upper_'+str(offset),shared,start*stride+offset*8,(count,offset),(stride,8))
                lower=view('shared_lower_'+str(offset),shared,start*stride,(count,offset),(stride,8))
                read=access('shared_read_'+str(offset),upper,Operation.READ,('cta',None),
                    (dependency(previous,EventKind.WRITE_VISIBLE),))
                previous=access('shared_write_'+str(offset),lower,Operation.WRITE,('cta',None),
                    (dependency(read),dependency(previous,EventKind.WRITE_VISIBLE)))
                offset//=2
            joined=[dependency(previous,EventKind.WRITE_VISIBLE)]
        if c==1:
            access('output','output',Operation.WRITE,(None,),tuple(joined),loops=('cta',))
        else:
            name=view('partial',partial,start*8,(count,),(8,))
            published=access('partial',name,Operation.WRITE,('cta',),tuple(joined));partial_ready.append(dependency(published,EventKind.WRITE_VISIBLE,('cta',)))
            partial_groups.append((prefix,name,cta_axis))
            read=access('ticket_read','completion_counter',Operation.READ,(None,),(
                TileDependency('counter_zero',EventKind.WRITE_VISIBLE,()),dependency(published,EventKind.WRITE_VISIBLE)))
            ticket=access('ticket_write','completion_counter',Operation.WRITE,(None,),
                (dependency(read),),atomic=TileAtomicAdd(read,1));ticket_ready.append(dependency(ticket,EventKind.WRITE_VISIBLE,('cta',)))
            name=view('flag',flag,start,(count,),(1,),'bool',1)
            marked=access('flag_write',name,Operation.WRITE,('cta',),(dependency(ticket,EventKind.WRITE_VISIBLE),))
            access('flag_read',name,Operation.READ,('cta',),(dependency(marked,EventKind.WRITE_VISIBLE),),loops=('cta','thread'))
        all_axes.extend((TileAxis(cta_axis,count,1),*((TileAxis(thread_axis,w,1),) if c>1 else ())))
        all_accesses.extend(accesses)
    if c>1:
        final_ready=[]
        for prefix,name,axis in partial_groups:
            identity='final_partials_'+prefix
            all_accesses.append(TileAccessTemplate(identity,name,Operation.READ,(axis,),(axis,),(),tuple((*partial_ready,*ticket_ready)),evidence))
            final_ready.append(TileDependency(identity,EventKind.READ_READY,(axis,)))
        if cuda:
            # The shared array belongs to the unknown last finisher. Split
            # each source store into disjoint logical chunks, so every later
            # upper-half read has its exact same-view writer, without guessing
            # the selected CTA's physical offset or manufacturing readiness.
            root='finisher_shared';selection=bindings.binding_id+'.last_finisher_shared'
            selected.append(LogicalSelectionView(root,shared,(w,),'int64',8,selection,None))
            chunks=[];offset=w//2
            while offset>=hardware.parallel_width:
                chunks.append((offset,offset));offset//=2
            chunks.append((0,hardware.parallel_width));ready={};names={}
            for start,size in chunks:
                name='finisher_chunk_'+str(start);names[start]=name
                selected.append(LogicalSelectionView(name,shared,(size,),'int64',8,selection+'.'+str(start),root,
                    (LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),offset=start),)))
                y='finisher_y_seed_'+str(start);x='finisher_x_seed_'+str(start)
                all_accesses.append(TileAccessTemplate(y,name,Operation.WRITE,(),(None,),(),tuple(final_ready),evidence,unmapped=True))
                all_accesses.append(TileAccessTemplate(x,name,Operation.WRITE,(),(None,),(),
                    (*final_ready,TileDependency(y,EventKind.WRITE_VISIBLE,())),evidence,unmapped=True))
                ready[start]=x
            offset=w//2
            while offset>=hardware.parallel_width:
                read='finisher_upper_'+str(offset)
                all_accesses.append(TileAccessTemplate(read,names[offset],Operation.READ,(),(None,),(),
                    (TileDependency(ready[offset],EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True))
                for start,size in chunks:
                    if start>=offset:continue
                    write='finisher_lower_'+str(offset)+'_'+str(start)
                    all_accesses.append(TileAccessTemplate(write,names[start],Operation.WRITE,(),(None,),(),
                        (TileDependency(read,EventKind.READ_READY,()),TileDependency(ready[start],EventKind.WRITE_VISIBLE,())),evidence,unmapped=True))
                    ready[start]=write
                offset//=2
            final_ready=[TileDependency(ready[0],EventKind.WRITE_VISIBLE,())]
        all_accesses.append(TileAccessTemplate('final_output','output',Operation.WRITE,(),(None,),(),tuple(final_ready),evidence))
    tag='cuda_extent_state' if cuda else 'metax_extent_state'
    scope=AccessScope(program_id,(tag,str(n),str(w),str(c),str(v)),bindings.call_id,(),bindings.rank,bindings.stream,
        'logical-cuda-extent-state' if cuda else 'logical-metax-extent-state')
    family=TileProgram(program_id,scope,tuple(all_axes),tuple(all_accesses),bindings.visibility_domain,evidence)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'CUDA2.8 scalar INT64 logical state; symbolic finisher/shared selection, no physical transaction/observed launch claim.' if cuda else
                'MACA3.7.2 contiguous INT64 native logical state; no physical transaction/cache or observed CTA launch claim.',
            'Analytical launch and legacy aggregate-read price retained. Tiny/other backend native implementations remain separate.',
            'One shared last-finisher flag per CTA; atomic previous values and winning CTA remain symbolic.')),
        tuple(allocations),tuple(selected),(),tuple(events),(),(family,))


def _extent_state_work(program,kernel,hardware):
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_path_solver import logical_requested_bytes,logical_view_requested_bytes
    if hardware is None:raise ValueError('MetaX native extent source validation requires its actual HWSpec')
    views=unique_by(program.views,'view_id');family=program.tile_programs[0]
    ids={views[role].allocation_id for role in ('input','output')}
    inputs=tuple(a for f in program.tile_programs for a in f.accesses if a.input_ready)
    readiness={r for a in inputs for r in a.input_ready}
    if len(readiness)!=1:raise ValueError('Native extent lost its actual input readiness')
    binding=ElementwiseBindings(program.binding_id,family.scope.call_id,
        tuple(a for a in program.allocations if a.allocation_id in ids),tuple(views[role] for role in ('input','output')),
        tuple(e for e in program.events if e.allocation_id in ids),('input',),('output',),tuple(readiness),
        family.scope.rank,family.scope.stream,family.visibility_domain)
    profile='cuda' if family.scope.op_ids[0] in ('cuda_extent_state','cuda_extent_partitioned') else 'maca'
    fresh=(_build_partitioned_extent(kernel,hardware,binding,program_id=program.program_id,profile=profile)
        if family.scope.op_ids[0] in ('cuda_extent_partitioned','metax_extent_partitioned') else
        _build_extent_state(kernel,hardware,binding,program_id=program.program_id,profile=profile))
    if fresh!=program:raise ValueError('MetaX extent native ownership/state/READ_READY causality differs from source')
    read,write=(logical_requested_bytes(program,operation) for operation in (Operation.READ,Operation.WRITE))
    logical,execution=_moe_row_extents(kernel)
    input_roles={a.view_id for f in program.tile_programs for a in f.accesses
        if a.operation is Operation.READ and views[a.view_id].allocation_id==views['input'].allocation_id}
    public_read=sum(logical_view_requested_bytes(program,role,Operation.READ) for role in input_roles)*logical//execution
    public_write=logical_view_requested_bytes(program,'output',Operation.WRITE)
    elements=public_read//views['input'].element_bytes
    return {'read_bytes':read,'write_bytes':write,'legacy_aggregate_read_only_bytes':public_read+public_write,
        'legacy_read_bytes':public_read,'legacy_write_bytes':public_write,'execution_tokens':execution,
        'selected_map_unique_bytes':0,'tokens':logical,'topk':kernel.shape['topk'],'elements':elements}


def build_moe_metadata_source(kernel,hardware,bindings,granularity,*,program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_moe_metadata_source(kernel,hardware) or not isinstance(bindings,ElementwiseBindings):
        raise TypeError('MoE metadata requires supported typed GPU operands')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown MoE metadata granularity')
    reads=tuple(role for role in _metadata_layouts(kernel) if role!='output')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(reads,('output',),len(reads)):
        raise ValueError('MoE metadata actual read/output roles differ')
    if cuda_extent_partition_geometry(kernel,hardware) is not None:
        program=_build_cuda_partitioned_extent(kernel,hardware,bindings,program_id=program_id)
        moe_metadata_source_work(program,expected_kernel=kernel,hardware=hardware)
        return program
    if metax_extent_partition_geometry(kernel,hardware) is not None:
        program=_build_partitioned_extent(kernel,hardware,bindings,program_id=program_id,profile='maca')
        moe_metadata_source_work(program,expected_kernel=kernel,hardware=hardware)
        return program
    native=metax_extent_state_geometry(kernel,hardware)
    if native is not None and (native['lanes']>hardware.parallel_width or native['ctas']>1):
        program=_build_metax_extent_state(kernel,hardware,bindings,program_id=program_id)
        moe_metadata_source_work(program,expected_kernel=kernel,hardware=hardware)
        return program
    native=cuda_extent_state_geometry(kernel,hardware)
    if native is not None and (native['shared_bytes_per_cta'] or native['ctas']>1):
        program=_build_extent_state(kernel,hardware,bindings,program_id=program_id,profile='cuda')
        moe_metadata_source_work(program,expected_kernel=kernel,hardware=hardware)
        return program
    scope=AccessScope(program_id,_moe_replay_header(kernel,
        ('moe_metadata',kernel.variant,*(str(kernel.shape[name]) for name in ('tokens','experts','topk')))),
        bindings.call_id,(),bindings.rank,bindings.stream,'logical-moe-metadata')
    family,extra=_metadata_tiles(kernel,unique_by(bindings.views,'view_id'),dict(zip(reads,bindings.input_ready,strict=True)),
        scope,bindings.visibility_domain)
    native=metax_extent_single_wave_geometry(kernel,hardware)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Typed public casts, one-invocation scalar extent reduction or unknown expert-map selections.',
            'The legacy aggregate read-only price is derived from logical read+write requests, not asserted as actual traffic direction.',
            ('MACA 3.7.2 source-bound one-wave INT64 extent: register/shuffle state only; no shared/global partial or semaphore allocation. Native issue/cache services remain unbound.' if native is not None else
                'Native reduction spill/scratch, concrete selected expert values, repeated-address footprint and physical cache services are unbound.'))),
        bindings.allocations,(*bindings.views,*extra),(),bindings.events,(),(family,))
    moe_metadata_source_work(program,expected_kernel=kernel,hardware=hardware)
    return program


def validate_moe_metadata_bindings(program,*,expected_kernel,expected_bindings):
    """Check caller-owned roots separately from source-private native state."""
    from src.kernel.memory_access_builder import ElementwiseBindings
    binding=expected_bindings
    if not isinstance(binding,ElementwiseBindings):raise TypeError('MoE metadata requires typed actual caller bindings')
    roles=set(_metadata_layouts(expected_kernel));views=unique_by(program.views,'view_id')
    expected=unique_by(binding.views,'view_id');ids={a.allocation_id for a in binding.allocations}
    expected_events=unique_by(binding.events,'event_id')
    if (set(expected)!=roles or not roles<=set(views) or program.binding_id!=binding.binding_id
            or any(views[role]!=expected[role] for role in roles)
            or {a.allocation_id:a for a in program.allocations if a.allocation_id in ids}
                !=unique_by(binding.allocations,'allocation_id')
            or {e.event_id:e for e in program.events if e.allocation_id in ids or e.event_id in expected_events}!=expected_events
            or len(program.tile_programs)!=1):
        raise ValueError('MoE metadata source differs from actual caller public parents/readiness')
    family,=program.tile_programs
    if ((family.scope.call_id,family.scope.rank,family.scope.stream,family.visibility_domain)
            !=(binding.call_id,binding.rank,binding.stream,binding.visibility_domain)
            or {r for access in family.accesses for r in access.input_ready}!=set(binding.input_ready)):
        raise ValueError('MoE metadata source differs from actual caller context/readiness')


def moe_metadata_source_work(program,*,expected_kernel,hardware=None,expected_bindings=None):
    from src.kernel.memory_access import LogicalSelectionView
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__();kernel=expected_kernel
    if expected_bindings is not None:
        validate_moe_metadata_bindings(program,expected_kernel=kernel,expected_bindings=expected_bindings)
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant not in _METADATA_VARIANTS
            or kernel.execution_support!='supported' or kernel.hardware_id!=program.hardware_id):
        raise ValueError('MoE metadata differs from actual kernel')
    if hardware is not None and (hardware.hardware_id!=kernel.hardware_id or hardware.kernel_memory_model!='gpu_mshr'):
        raise ValueError('MoE metadata requires its matching actual HWSpec GPU execution path')
    if kernel.hardware_id=='metax_c500' and kernel.variant=='moe_routed_id_extent':
        if hardware is None:raise ValueError('MetaX extent source validation requires its actual HWSpec')
        if metax_extent_partition_geometry(kernel,hardware) is not None:
            if not program.tile_programs or program.tile_programs[0].scope.op_ids[:1]!=('metax_extent_partitioned',):
                raise ValueError('MetaX extent source requires complete native state for its partitioned kernel')
        native=metax_extent_state_geometry(kernel,hardware)
        if native is not None and (native['lanes']>hardware.parallel_width or native['ctas']>1):
            if not program.tile_programs or program.tile_programs[0].scope.op_ids[:1]!=('metax_extent_state',):
                raise ValueError('MetaX extent source requires complete native state for its actual kernel')
    if kernel.hardware_id in ('nvidia_a800','nvidia_h200') and kernel.variant=='moe_routed_id_extent':
        if hardware is None:raise ValueError('CUDA extent source validation requires its actual HWSpec')
        if cuda_extent_partition_geometry(kernel,hardware) is not None:
            if not program.tile_programs or program.tile_programs[0].scope.op_ids[:1]!=('cuda_extent_partitioned',):
                raise ValueError('CUDA extent source requires complete native state for its partitioned kernel')
        native=cuda_extent_state_geometry(kernel,hardware)
        if native is not None and (native['shared_bytes_per_cta'] or native['ctas']>1):
            if not program.tile_programs or program.tile_programs[0].scope.op_ids[:1]!=('cuda_extent_state',):
                raise ValueError('CUDA extent source requires complete native state for its actual kernel')
    if program.tile_programs and program.tile_programs[0].scope.op_ids[:1] in (('metax_extent_state',),('cuda_extent_state',),('cuda_extent_partitioned',),('metax_extent_partitioned',)):
        return _extent_state_work(program,kernel,hardware)
    if program.scopes or program.accesses or len(program.tile_programs)!=1:raise ValueError('MoE metadata requires its unique compressed source')
    family,=program.tile_programs;layouts=_metadata_layouts(kernel);views=unique_by(program.views,'view_id')
    lookup=kernel.variant=='moe_expert_map_lookup';expected_roles={*layouts,* (('map_selection',) if lookup else ())}
    header=_moe_replay_header(kernel,('moe_metadata',kernel.variant,*(str(kernel.shape[name]) for name in ('tokens','experts','topk'))))
    if (set(views)!=expected_roles or family.scope.op_ids!=header or family.scope.agent!='logical-moe-metadata'):
        raise ValueError('MoE metadata tensor/source roles differ')
    parents=set()
    for role,(shape,dtype,width) in layouts.items():
        view=views[role];strides=(shape[1]*width,width) if len(shape)==2 else (width,)
        if not isinstance(view,TensorView) or (view.shape,view.dtype,view.element_bytes,view.strides_bytes)!=(shape,dtype,width,strides):
            raise ValueError('MoE metadata dtype/shape/stride differs')
        parents.add(view.allocation_id)
    if len(parents)!=len(layouts):raise ValueError('MoE metadata input/output parents must be distinct')
    if lookup and not isinstance(views['map_selection'],LogicalSelectionView):raise ValueError('Expert-map values require an unresolved selected view')
    templates=unique_by(family.accesses,'access_id');reads=tuple(role for role in layouts if role!='output')
    ready={}
    for role in reads:
        name='map_selection' if role=='expert_map' else role
        if name not in templates or len(templates[name].input_ready)!=1:raise ValueError('MoE metadata actual input readiness missing')
        ready[role]=templates[name].input_ready[0]
    expected,extra=_metadata_tiles(kernel,views,ready,family.scope,family.visibility_domain)
    if family!=expected or lookup and views['map_selection']!=extra[0]:
        raise ValueError('MoE metadata lost actual cast/reduction/index READ_READY causality')
    read,write=(logical_requested_bytes(program,operation) for operation in (Operation.READ,Operation.WRITE))
    from math import prod
    logical_layouts=_metadata_layouts(kernel,execution=False)
    elements=logical_layouts['output'][0][0] if lookup else prod(logical_layouts['input'][0])
    if lookup:
        # The old price selected once per logical expert-id slot, rather
        # than reading the complete expert-map allocation.
        legacy_read=elements*(views['expert_ids'].element_bytes+views['expert_map'].element_bytes)
    else:legacy_read=elements*views['input'].element_bytes
    legacy_write=prod(logical_layouts['output'][0])*views['output'].element_bytes
    return {'read_bytes':read,'write_bytes':write,'legacy_aggregate_read_only_bytes':legacy_read+legacy_write,
        'legacy_read_bytes':legacy_read,'legacy_write_bytes':legacy_write,'execution_tokens':_moe_row_extents(kernel)[1],
        'selected_map_unique_bytes':None if lookup else 0,'tokens':kernel.shape['tokens'],'topk':kernel.shape['topk'],
        'elements':elements}


def _a800_topk_source(kernel):
    from src.kernel.launch_config.nvidia_a800.fused_moe_launch import uses_specialized_topk_softmax
    return (kernel.hardware_id=='nvidia_a800' and kernel.launch.algorithm=='moe_topk'
        and kernel.launch.block==(32,4,1) and uses_specialized_topk_softmax(kernel.shape['experts']))


def _h200_topk_source(kernel):
    # vLLM 0.24 fused_topk passes its declared gating tensor directly to
    # topk_softmax, without the older A800 FP32 input conversion.
    from src.kernel.launch_config.nvidia_a800.fused_moe_launch import uses_specialized_topk_softmax
    return (kernel.hardware_id=='nvidia_h200' and kernel.launch.algorithm=='moe_topk'
        and kernel.launch.block==(32,4,1) and uses_specialized_topk_softmax(kernel.shape['experts']))


def _c500_topk_source(kernel):
    experts=kernel.shape['experts'] if kernel.variant=='moe_topk' else 0
    return (kernel.hardware_id=='metax_c500' and 0<experts<=256 and experts&(experts-1)==0
        and ((kernel.launch.algorithm=='vllm_0_24_metax_topk_gating_softmax' and kernel.launch.block==(32,4,1))
            or (kernel.launch.algorithm=='moe_topk' and kernel.launch.block==(256,1,1))))


def has_gpu_topk_source_abi(kernel):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant=='moe_topk' and kernel.dtype=='bf16'
        and kernel.execution_support=='supported'
        and (kernel.launch.algorithm=='vllm_topk_softmax_bf16_int32' or _a800_topk_source(kernel)
            or _h200_topk_source(kernel) or _c500_topk_source(kernel)))


def has_gpu_topk_source(kernel,hardware):
    return hardware.kernel_memory_model=='gpu_mshr' and has_gpu_topk_source_abi(kernel)


def _gpu_topk_input_type(kernel,view=None):
    if _a800_topk_source(kernel) or _c500_topk_source(kernel):expected=('fp32',4)
    elif kernel.launch.algorithm=='vllm_topk_softmax_bf16_int32' or _h200_topk_source(kernel):expected=('bf16',2)
    else:raise ValueError('Unknown GPU top-k source input encoding')
    if view is None:return expected
    actual=(view.dtype,view.element_bytes)
    # H200's retained public fused_topk wrapper forwards gating_output
    # unchanged, independently of hidden_states.dtype. The explicitly named
    # BF16 native ABI and the A800/MetaX FP32 conversion remain fixed.
    if actual!=expected and not (_h200_topk_source(kernel) and actual==('fp32',4)):
        raise ValueError('GPU top-k logits differ from its declared input encoding')
    return actual


def bind_gpu_topk_logits_type(kernel,bindings,dtype,width):
    """Retain the actual raw gating tensor; never insert a numeric cast."""
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_gpu_topk_source_abi(kernel) or not isinstance(bindings,ElementwiseBindings):
        raise ValueError('GPU top-k logits require their actual public source ABI')
    from src.kernel.kernel_spec import DTYPE_BYTES
    if dtype not in DTYPE_BYTES or width!=DTYPE_BYTES[dtype]:
        raise ValueError('GPU top-k logits require their exact scalar encoding width')
    views=unique_by(bindings.views,'view_id');source=views['logits']
    if not isinstance(source,TensorView):raise ValueError('GPU top-k logits require their actual typed row input')
    selected=replace(source,dtype=dtype,element_bytes=width,strides_bytes=(source.shape[1]*width,width))
    _gpu_topk_input_type(kernel,selected)
    if len(bindings.input_ready)!=1:raise ValueError('GPU top-k requires one actual logits readiness')
    ready,=bindings.input_ready
    return replace(bindings,
        allocations=tuple(replace(a,size_bytes=source.shape[0]*source.shape[1]*width)
            if a.allocation_id==source.allocation_id else a for a in bindings.allocations),
        views=tuple(selected if v.view_id=='logits' else v for v in bindings.views),
        events=tuple(replace(e,region=selected.region()) if e.event_id==ready.node_id else e for e in bindings.events))


def _layouts(tokens,experts,topk,input_dtype='bf16',input_width=2):
    return (('logits',(tokens,experts),input_dtype,input_width),('weights',(tokens,topk),'fp32',4),
            ('expert_ids',(tokens,topk),'int32',4),('source_rows',(tokens,topk),'int32',4))


def bind_gpu_topk_gating_input(kernel,bindings,dtype,width):
    """Bind the A800 wrapper's raw gating tensor before its explicit float cast."""
    if not (_a800_topk_source(kernel) and (dtype,width)==('bf16',2)):
        return bind_gpu_topk_logits_type(kernel,bindings,dtype,width)
    # vLLM 0.11 fused_topk explicitly calls gating_output.float(). The
    # selected TopK launch and its legacy price stay intact; this records
    # its wrapper tensor dataflow without inventing another priced launch.
    from src.kernel.memory_access_builder import ElementwiseBindings
    if (not isinstance(bindings,ElementwiseBindings) or bindings.read_views!=('logits',)
            or len(bindings.input_ready)!=1):
        raise ValueError('A800 raw gating requires its unique actual wrapper input')
    views=unique_by(bindings.views,'view_id');original=views['logits']
    _gpu_topk_input_type(kernel,original)
    raw=replace(original,view_id='gating_input',dtype=dtype,element_bytes=width,
        strides_bytes=(original.shape[1]*width,width))
    identity=bindings.binding_id+'.fp32_logits'
    parents=unique_by(bindings.allocations,'allocation_id');parent=parents[original.allocation_id]
    cast=replace(parent,allocation_id=identity,memory_domain='logical-a800-topk-fp32-cast',
        allocated_event=identity+'.allocated')
    logits=replace(original,allocation_id=identity)
    ready,=bindings.input_ready
    return replace(bindings,read_views=('gating_input',),
        allocations=tuple(replace(a,size_bytes=original.shape[0]*original.shape[1]*width)
            if a.allocation_id==original.allocation_id else a for a in bindings.allocations)+(cast,),
        views=tuple(raw if v.view_id=='logits' else v for v in bindings.views)+(logits,),
        events=tuple(replace(e,region=raw.region()) if e.event_id==ready.node_id else e for e in bindings.events)+(
            ProgramEvent(cast.allocated_event,EventKind.ALLOCATED,identity,None,None,()),))


def standalone_gpu_topk_bindings(kernel, *, call_id,binding_id, device_id="logical-local-device", rank=0, stream="offline"):
    from src.kernel.memory_access_builder import ElementwiseBindings
    allocations=[];views=[];events=[]
    tokens=_moe_row_extents(kernel)[1]
    for role,shape,dtype,width in _layouts(tokens,kernel.shape['experts'],kernel.shape['topk'],*_gpu_topk_input_type(kernel)):
        identity=f'{binding_id}.{role}'
        allocation=MemoryAllocation(identity,device_id,'logical-storage',shape[0]*shape[1]*width,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=TensorView(role,identity,0,shape,(shape[1]*width,width),dtype,width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role=='logits':
            event=ProgramEvent(f'{identity}.initialized',EventKind.INITIALIZED,identity,view.region(),'topk-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready=EventRef(event.event_id,event.kind)
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('logits',),('weights','expert_ids','source_rows'),(ready,),rank,stream,'topk-visible')


def _tiles(views,ready,scope,visibility,evidence):
    tokens=views['logits'].shape[0]
    if 'gating_input' in views:
        accesses=[TileAccessTemplate('gating_input','gating_input',Operation.READ,('token',),('token',None),(ready,),(),evidence),
            TileAccessTemplate('float_cast','logits',Operation.WRITE,('token',),('token',None),(),
                (TileDependency('gating_input',EventKind.READ_READY,()),),evidence),
            TileAccessTemplate('logits','logits',Operation.READ,('token',),('token',None),(),
                (TileDependency('float_cast',EventKind.WRITE_VISIBLE,()),),evidence)]
    else:
        accesses=[TileAccessTemplate('logits','logits',Operation.READ,('token',),('token',None),(ready,),(),evidence)]
    for role in ('weights','expert_ids','source_rows'):
        accesses.append(TileAccessTemplate(role,role,Operation.WRITE,('token',),('token',None),(),
            (TileDependency('logits',EventKind.READ_READY,()),),evidence))
    return TileProgram(scope.scope_id,scope,(TileAxis('token',tokens,1),),tuple(accesses),visibility,evidence)


def build_gpu_topk_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_gpu_topk_source(kernel,hardware):raise ValueError('GPU top-k requires its declared source ABI')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('GPU top-k requires typed row operands')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('GPU top-k exposes logical row tiles')
    views=unique_by(bindings.views,'view_id');raw='gating_input' in views
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=((('gating_input',) if raw else ('logits',)),('weights','expert_ids','source_rows'),1):
        raise ValueError('GPU top-k must retain all three actual metadata outputs')
    evidence=('Source-declared topkGating: A800 vLLM0.11 and the retained MetaX FP32/32-lane template consume FP32 logits; H200 vLLM0.24 forwards its actual BF16/FP32 gating tensor without a cast. The separately named BF16 native ABI stays fixed. Each row reads its complete logits and writes FP32 weights, INT32 expert ids and INT32 source rows.',
        'The specialized power-of-two expert family through E256 needs no GM softmax workspace; independent rows require no global ordering.',
        'Native wrapper dispatch, not the nominal auxiliary price, determines dtype and complete output roles.')
    if raw:evidence+=('vLLM 0.11 fused_topk: gating_output.float() writes a distinct FP32 tensor before the native TopK reads it. BF16 gating remains its actual producer dtype; the cast is not a relabel or a new priced launch.',)
    scope=AccessScope(program_id,_moe_replay_header(kernel,('gpu_topk_rows',kernel.launch.algorithm)),
        bindings.call_id,(),bindings.rank,bindings.stream,'logical-gpu-topk-rows')
    family=_tiles(views,bindings.input_ready[0],scope,bindings.visibility_domain,evidence)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,True,(
            'Complete public specialized fused row interface; no native instruction or physical transaction claim.',
            'Legacy aggregate-read and FMA price remain separate from actual logical metadata writes.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    gpu_topk_source_demands(program,expected_kernel=kernel);return program


def gpu_topk_source_demands(program, *, expected_kernel):
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    if program.scopes or program.accesses or len(program.tile_programs)!=1:
        raise ValueError('GPU top-k requires its unique compressed row source')
    family,=program.tile_programs;views=unique_by(program.views,'view_id')
    if (expected_kernel.kind is not KernelKind.VECTOR or expected_kernel.variant!='moe_topk'
            or expected_kernel.hardware_id!=program.hardware_id or expected_kernel.dtype!='bf16'):
        raise ValueError('GPU top-k source differs from its actual typed kernel')
    tokens,experts,topk=(expected_kernel.shape[name] for name in ('tokens','experts','topk'))
    logical_tokens,tokens=_moe_row_extents(expected_kernel)
    if _c500_topk_source(expected_kernel) and expected_kernel.launch.algorithm=='moe_topk':
        # The existing analytical launch is retained as selected. Its generic
        # grid is not a statement about the native row-kernel CTA schedule.
        if expected_kernel.launch.grid!=(-(-(tokens*experts)//256),1,1) or expected_kernel.launch.parameters:
            raise ValueError('GPU top-k source must retain its actual analytical launch')
    elif _a800_topk_source(expected_kernel) or _h200_topk_source(expected_kernel) or _c500_topk_source(expected_kernel):
        # Preserve the existing selected row launch. The projected H200
        # descriptor forwards its actual public input rather than an A800 cast.
        load=min(16,experts*4)//4;vectors=max(1,experts//(load*32));per_thread=vectors*load
        threads=experts//per_thread;rows=32//threads
        if (expected_kernel.launch.grid!=(-(-(-(-tokens//rows))//4),1,1)
                or expected_kernel.launch.parameters):
            raise ValueError('GPU top-k source must retain its actual source-declared row launch')
    elif (expected_kernel.launch.algorithm!='vllm_topk_softmax_bf16_int32' or experts!=256 or topk!=8
            or not 0<tokens<=8 or expected_kernel.launch.grid!=(-(-tokens//4),1,1)
            or expected_kernel.launch.block!=(32,4,1) or expected_kernel.launch.parameters):
        raise ValueError('GPU top-k source must retain its declared native launch and type facts')
    raw='gating_input' in views
    if (family.scope.op_ids!=_moe_replay_header(expected_kernel,('gpu_topk_rows',expected_kernel.launch.algorithm))
            or family.scope.agent!='logical-gpu-topk-rows' or set(views)!={'logits','weights','expert_ids','source_rows',*(('gating_input',) if raw else ())}
            or len({view.allocation_id for view in views.values()})!=(5 if raw else 4)):
        raise ValueError('GPU top-k row source must preserve its four distinct tensor parents')
    for role,shape,dtype,width in _layouts(tokens,experts,topk,*_gpu_topk_input_type(expected_kernel,views['logits'])):
        view=views[role]
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes)!=(shape,dtype,width,(shape[1]*width,width)):
            raise ValueError('GPU top-k tensor must retain its actual row/type layout')
    if raw:
        view=views['gating_input'];allocations=unique_by(program.allocations,'allocation_id')
        cast=allocations[views['logits'].allocation_id]
        if (not _a800_topk_source(expected_kernel) or not isinstance(view,TensorView)
                or (view.shape,view.dtype,view.element_bytes,view.strides_bytes)!=((tokens,experts),'bf16',2,(experts*2,2))
                or cast.memory_domain!='logical-a800-topk-fp32-cast' or cast.size_bytes!=tokens*experts*4
                or cast.device_id!=allocations[view.allocation_id].device_id):
            raise ValueError('A800 wrapper float cast differs from its actual typed storage')
    read=unique_by(family.accesses,'access_id')['gating_input' if raw else 'logits']
    if len(read.input_ready)!=1:raise ValueError('GPU top-k requires actual logits readiness')
    expected=_tiles(views,read.input_ready[0],family.scope,family.visibility_domain,family.evidence_refs)
    if family!=expected:raise ValueError('GPU top-k metadata must depend on its complete logits READ_READY')
    read_bytes=logical_requested_bytes(program,Operation.READ)
    # Preserve the existing aggregate price. Raw-logit encoding is a public
    # storage fact, not permission to double the legacy BF16 read-only price.
    from src.kernel.memory_path_solver import logical_view_unique_bytes
    native_read=logical_view_unique_bytes(program,'logits',Operation.READ)
    legacy_read=native_read//views['logits'].element_bytes*_gpu_topk_input_type(expected_kernel)[1]//tokens*logical_tokens
    return {'tokens':logical_tokens,'execution_tokens':tokens,'experts':experts,'topk':topk,
        'read_bytes':read_bytes,
        'write_bytes':logical_requested_bytes(program,Operation.WRITE),
        'legacy_aggregate_read_price_bytes':legacy_read*
            (1 if _a800_topk_source(expected_kernel) or _c500_topk_source(expected_kernel) else 2),
        'legacy_fma_operations':logical_tokens*experts}
