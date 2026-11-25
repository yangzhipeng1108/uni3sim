"""CUDA alignment's histogram, prefix scratch and sorted metadata dataflow.

The installed wrapper and actual W1 supply block/capacity/routing context;
bare auxiliary inputs retain the native scalar arguments as unbound values.
Versioned CUDA/CUB references own private rows, logical raking scratch and
atomic count dataflow. These requests do not assert native issued traffic.
"""

from src.kernel.prediction_cache import runtime_lru_cache
from dataclasses import replace

from src.kernel.memory_access import (
    AccessScope,AddressLevel,ConstantValue,Coverage,EventKind,EventRef,Granularity,
    LogicalSelectionView,MemoryAccessProgram,MemoryAllocation,Operation,Ordering,
    ProgramEvent,Semantics,TensorView,TileAccessTemplate,TileAxis,TileDependency,
    TileIndexBinding,TileIndexTerm,TileProgram,TileValueIndex,unique_by,_tile_template_requested_bytes,
    _replace_source_identity,
)


def _geometry(kernel):
    from src.kernel.launch_config.select_launch_config import moe_alignment_capacity,moe_count_sort_source_program
    s=kernel.shape;a=kernel.attributes;r=s['tokens']*s['topk'];e=s['experts'];threads=max(e,32)
    source_block=(kernel.launch.block==(threads,1,1)
        or kernel.hardware_id=='nvidia_h200' and kernel.launch.block==(threads+256,1,1))
    if (kernel.kind.value!='VECTOR' or kernel.variant!='moe_align_small_batch' or kernel.execution_support!='supported'
            or kernel.launch.algorithm not in ('moe_align_small_batch','observed_moe_aux:moe_align_small_batch')
            or kernel.hardware_id not in ('nvidia_a800','nvidia_h200')
            or moe_count_sort_source_program(kernel.hardware_id)!='cuda_moe_count_sort_int32_unmasked'
            or not 0<r<1024 or not 0<e<=64 or kernel.launch.grid!=(1,1,1) or not source_block):
        raise ValueError('Small-batch alignment requires actual one-CTA CUDA source')
    context={'alignment_block_size','alignment_capacity','tokens_per_expert'}
    if not context&set(a):return r,e,threads,None,None,None
    if not context<=set(a):raise ValueError('Small-batch alignment requires complete actual W1 context')
    block,capacity,counts=(a[name] for name in ('alignment_block_size','alignment_capacity','tokens_per_expert'))
    if (capacity!=moe_alignment_capacity(kernel.hardware_id,s['tokens'],s['topk'],e,block)
            or len(counts)!=e or sum(counts)!=r or sum(-(-n//block)*block for n in counts)>capacity):
        raise ValueError('Small-batch alignment disagrees with actual public allocation/routing facts')
    return r,e,threads,block,capacity,counts


def small_alignment_bindings(kernel,*,call_id,binding_id, device_id="logical-local-device", rank=0, stream="offline"):
    from src.kernel.memory_access_builder import ElementwiseBindings
    r,e,t,block,capacity,counts=_geometry(kernel)
    symbolic=block is None;envelope=2**31-1
    layouts=(('topk_ids',r),('sorted_token_ids',envelope if symbolic else capacity),
        ('expert_ids',envelope if symbolic else -(-capacity//block)),('num_tokens_post_padded',1))
    if symbolic:layouts=(*layouts,('block_size_arg',1),('capacity_arg',1))
    allocations=[];views=[];events=[];ready=[]
    for role,count in layouts:
        identity=f'{binding_id}.{role}'
        unknown=symbolic and role in ('sorted_token_ids','expert_ids')
        allocation=MemoryAllocation(identity,device_id,
            'logical-kernel-argument' if role.endswith('_arg') else 'logical-alignment-interface',None if unknown else count*4,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        view=(LogicalSelectionView(role,identity,(count,),'int32',4,identity+'.unknown_capacity',None)
            if unknown else TensorView(role,identity,0,(count,),(4,),'int32',4))
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role=='topk_ids' or role.endswith('_arg'):
            initialized=ProgramEvent(identity+'.initialized',EventKind.INITIALIZED,identity,view.region(),
                'moe-alignment-visible',(EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(initialized);ready.append(EventRef(initialized.event_id,initialized.kind))
    reads=('topk_ids','block_size_arg','capacity_arg') if symbolic else ('topk_ids',)
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),reads,
        ('sorted_token_ids','expert_ids','num_tokens_post_padded'),tuple(ready),rank,stream,'moe-alignment-visible')


def _components(kernel,bindings,scope,evidence):
    return _small_components(_geometry(kernel),bindings,scope,evidence)


def _small_components(geometry,bindings,scope,evidence):
    r,e,t,block,capacity,counts=geometry;parents=unique_by(bindings.views,'view_id')
    symbolic=block is None;envelope=2**31-1
    source_allocations=unique_by(bindings.allocations,'allocation_id');device=source_allocations[parents['topk_ids'].allocation_id].device_id
    views=[];allocations=[];events=[];accesses=[];axes=[TileAxis('expert',e,1),
        TileAxis('padded',envelope if symbolic else capacity,1,value_limit_access_id='capacity_arg' if symbolic else None),TileAxis('worker',t,1)]
    for role,count in (('topk_ids',r),('sorted_token_ids',envelope if symbolic else capacity),
            ('expert_ids',envelope if symbolic else -(-capacity//block)),('num_tokens_post_padded',1)):
        view=parents[role]
        if symbolic and role in ('sorted_token_ids','expert_ids'):
            if (not isinstance(view,LogicalSelectionView) or (view.shape,view.dtype,view.element_bytes)!=((count,),'int32',4)
                    or source_allocations[view.allocation_id].size_bytes is not None or view.parent_view_id is not None):
                raise ValueError('Unbound alignment output requires its unknown-capacity INT32 selection')
        elif (not isinstance(view,TensorView) or (view.shape,view.strides_bytes,view.dtype,view.element_bytes)!=((count,),(4,),'int32',4)):
            raise ValueError('Small alignment requires its actual INT32 public tensors')
    if len({v.allocation_id for v in parents.values()})!=len(parents):raise ValueError('Small alignment public objects must be distinct')
    def allocate(name,domain,size):
        identity=f'{bindings.binding_id}.{scope.scope_id}.{name}'
        allocation=MemoryAllocation(identity,device,domain,size,AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        return identity
    def tensor(name,allocation,offset,count,stride=4):
        result=TensorView(f'{scope.scope_id}.{name}',allocation,offset,(count,),(stride,),'int32',4)
        views.append(result);return result
    def selection(name,parent,count):
        result=LogicalSelectionView(f'{scope.scope_id}.{name}',parent.allocation_id,(count,),'int32',4,
            f'cuda.small_alignment.{name}',parent.view_id)
        views.append(result);return result
    def input_ids(thread,n):
        parent=parents['topk_ids']
        if source_allocations[parent.allocation_id].size_bytes is not None:
            return tensor(f'ids_thread{thread}',parent.allocation_id,parent.offset_bytes+4*thread,n,4*t)
        from src.kernel.memory_access import LogicalCoordinateBinding,LogicalCoordinateTerm
        result=LogicalSelectionView(f'{scope.scope_id}.ids_thread{thread}',parent.allocation_id,(n,),'int32',4,
            f'cuda.small_alignment.ids_thread{thread}',parent.view_id,
            (LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,t),),thread),))
        views.append(result);return result
    def dep(name,kind=EventKind.READ_READY,reduction=(),bindings=(),carry=None,initial=None):
        return TileDependency(name,kind,reduction,bindings,carry,initial)
    def add(name,view,operation,loops,mapping,deps=(),*,ready=(),unmapped=False,value=None,index=None,domain='shared'):
        actor=replace(scope,scope_id=f'{scope.scope_id}.{name}.scope',agent=f'logical-small-alignment-{domain}')
        accesses.append(TileAccessTemplate(name,view.view_id,operation,loops,mapping,ready,deps,evidence,
            unmapped=unmapped,constant_value=value,value_index=index,scope=actor))
    if symbolic:
        for index,role in enumerate(('block_size_arg','capacity_arg'),1):
            view=parents[role]
            if not isinstance(view,TensorView) or (view.shape,view.dtype,view.element_bytes)!=((1,),'int32',4):
                raise ValueError('Unbound alignment requires actual signed scalar kernel arguments')
            add(role,view,Operation.READ,(),(None,),ready=(bindings.input_ready[index],),domain='argument')
    shared=allocate('shared','logical-shared-arena',4*((t+1)*e+e+1))
    prefix=tensor('cumsum',shared,0,e+1)
    rows=[tensor(f'counts_row{thread}',shared,4*(e+1+thread*e),e) for thread in range(t+1)]
    for thread,row in enumerate(rows):
        add(f'counts_init{thread}',row,Operation.WRITE,(),(None,),value=ConstantValue.ZERO)
    add('sorted_initialize',parents['sorted_token_ids'],Operation.WRITE,('padded',),('padded',),
        (dep('capacity_arg'),) if symbolic else (),unmapped=symbolic,domain='global')
    active=[]
    for thread in range(min(r,t)):
        n=-(-(r-thread)//t);loop=f'thread{thread}.step';axes.append(TileAxis(loop,n,1));active.append((thread,n,loop))
        ids=input_ids(thread,n)
        key=selection(f'count_key{thread}',rows[thread+1],n)
        read=f'count_id{thread}';modify=f'count_write{thread}';initial=f'counts_init{thread+1}'
        add(read,ids,Operation.READ,(loop,),(loop,),ready=(bindings.input_ready[0],),
            unmapped=isinstance(ids,LogicalSelectionView),domain='global')
        index=TileValueIndex(read,'loaded_scalar')
        add(f'count_read{thread}',key,Operation.READ,(loop,),(loop,),(
            dep(initial,EventKind.WRITE_VISIBLE),dep(read),dep(modify,EventKind.WRITE_VISIBLE,carry=loop,initial=initial)),unmapped=True,index=index)
        add(modify,key,Operation.WRITE,(loop,),(loop,),(dep(f'count_read{thread}'),dep(read)),unmapped=True,index=index)
    # Each expert thread scans worker rows independently. No expert-to-expert
    # barrier is invented between these native source loop iterations.
    for thread in range(t):
        before=f'counts_init{thread}' if thread==0 else f'scan_write{thread-1}'
        add(f'scan_previous{thread}',rows[thread],Operation.READ,('expert',),('expert',),(dep(before,EventKind.WRITE_VISIBLE),))
        own=(dep(f'counts_init{thread+1}',EventKind.WRITE_VISIBLE),)
        if thread<min(r,t):own=(*own,dep(f'count_write{thread}',EventKind.WRITE_VISIBLE,(f'thread{thread}.step',)))
        add(f'scan_current{thread}',rows[thread+1],Operation.READ,('expert',),('expert',),own)
        add(f'scan_write{thread}',rows[thread+1],Operation.WRITE,('expert',),('expert',),
            (dep(f'scan_previous{thread}'),dep(f'scan_current{thread}')))
    prefix_writers=[]
    zero=tensor('prefix0',shared,0,1)
    add('prefix0',zero,Operation.WRITE,(),(None,),value=ConstantValue.ZERO);prefix_writers.append(dep('prefix0',EventKind.WRITE_VISIBLE))
    for expert in range(e):
        total=tensor(f'total{expert}',shared,rows[-1].offset_bytes+4*expert,1)
        previous=tensor(f'prefix_previous{expert}',shared,4*expert,1)
        output=tensor(f'prefix_output{expert}',shared,4*(expert+1),1)
        binding=(TileIndexBinding('expert',(),expert),)
        add(f'total_read{expert}',total,Operation.READ,(),(None,),(
            dep(f'scan_write{t-1}',EventKind.WRITE_VISIBLE,bindings=binding),))
        add(f'prefix_read{expert}',previous,Operation.READ,(),(None,),(
            dep('prefix0' if expert==0 else f'prefix_write{expert-1}',EventKind.WRITE_VISIBLE),))
        add(f'prefix_write{expert}',output,Operation.WRITE,(),(None,),
            (dep(f'total_read{expert}'),dep(f'prefix_read{expert}'),*((dep('block_size_arg'),) if symbolic else ())))
        prefix_writers.append(dep(f'prefix_write{expert}',EventKind.WRITE_VISIBLE))
    end=tensor('prefix_end',shared,4*e,1)
    add('padded_read',end,Operation.READ,(),(None,),(dep(f'prefix_write{e-1}',EventKind.WRITE_VISIBLE),))
    add('padded_count',parents['num_tokens_post_padded'],Operation.WRITE,(),(None,),(dep('padded_read'),),domain='global')
    add('inactive_start',end,Operation.READ,('worker',),(None,),(dep(f'prefix_write{e-1}',EventKind.WRITE_VISIBLE),))
    # Actual final CTA join protects prefix fields and all private count rows
    # before any worker performs the second ID pass and in-place rank updates.
    prefix_join=tuple(prefix_writers)
    scan_join=tuple(dep(f'scan_write{thread}',EventKind.WRITE_VISIBLE,('expert',)) for thread in range(t))
    offset=0
    for expert,count in enumerate((None,)*e if symbolic else counts):
        pairs=tensor(f'prefix_pair{expert}',shared,4*expert,2)
        add(f'owner_bounds{expert}',pairs,Operation.READ,(),(None,),prefix_join)
        if symbolic:
            name=f'owner{expert}.block_count';identity=allocate(name,'logical-register-state',None)
            value=LogicalSelectionView(f'{scope.scope_id}.{name}',identity,(1,),'int32',4,
                f'cuda.small_alignment.{name}.from_prefix_difference_div_block_size',None);views.append(value)
            add(name+'.write',value,Operation.WRITE,(),(None,),
                (dep(f'owner_bounds{expert}'),dep('block_size_arg')),unmapped=True,domain='register')
            add(name,value,Operation.READ,(),(None,),(dep(name+'.write',EventKind.WRITE_VISIBLE),),unmapped=True,domain='register')
            loop=f'owner{expert}.block';axes.append(TileAxis(loop,envelope,1,value_limit_access_id=name))
            output=selection(f'owner{expert}',parents['expert_ids'],envelope)
            add(f'owner{expert}',output,Operation.WRITE,(loop,),(loop,),(dep(name),dep(f'owner_bounds{expert}')),
                unmapped=True,domain='global')
            continue
        blocks=-(-count//block)
        if blocks:
            loop=f'owner{expert}.block';axes.append(TileAxis(loop,blocks,1))
            output=tensor(f'owner{expert}',parents['expert_ids'].allocation_id,parents['expert_ids'].offset_bytes+4*offset,blocks)
            add(f'owner{expert}',output,Operation.WRITE,(loop,),(loop,),(dep(f'owner_bounds{expert}'),),domain='global');offset+=blocks
    if symbolic:
        name='inactive_block_count';identity=allocate(name,'logical-register-state',None)
        value=LogicalSelectionView(f'{scope.scope_id}.{name}',identity,(1,),'int32',4,
            'cuda.small_alignment.ceil_capacity_div_block_minus_prefix_end_div_block',None);views.append(value)
        add(name+'.write',value,Operation.WRITE,(),(None,),
            (dep('inactive_start',bindings=(TileIndexBinding('worker',(),0),)),dep('capacity_arg'),dep('block_size_arg')),
            unmapped=True,domain='register')
        add(name,value,Operation.READ,(),(None,),(dep(name+'.write',EventKind.WRITE_VISIBLE),),unmapped=True,domain='register')
        axes.append(TileAxis('inactive_block',envelope,1,value_limit_access_id=name))
        output=selection('inactive_blocks',parents['expert_ids'],envelope)
        add('inactive_blocks',output,Operation.WRITE,('inactive_block',),('inactive_block',),(dep(name),
            dep('inactive_start',bindings=(TileIndexBinding('worker',(TileIndexTerm('inactive_block',1),),modulus=t),))),
            unmapped=True,domain='global')
    spare=None if symbolic else parents['expert_ids'].shape[0]-offset
    if spare is not None and spare:
        loop='inactive_block';axes.append(TileAxis(loop,spare,1))
        output=tensor('inactive_blocks',parents['expert_ids'].allocation_id,parents['expert_ids'].offset_bytes+4*offset,spare)
        add('inactive_blocks',output,Operation.WRITE,(loop,),(loop,),(
            dep('inactive_start',bindings=(TileIndexBinding('worker',(TileIndexTerm(loop,1),),modulus=t),)),),domain='global')
    for thread,n,loop in active:
        ids=next(v for v in views if v.view_id==f'{scope.scope_id}.ids_thread{thread}')
        read=f'sort_id{thread}';add(read,ids,Operation.READ,(loop,),(loop,),ready=(bindings.input_ready[0],),
            unmapped=isinstance(ids,LogicalSelectionView),domain='global')
        key=selection(f'rank_count{thread}',rows[thread],n);prefix_key=selection(f'rank_prefix{thread}',prefix,n)
        index=TileValueIndex(read,'loaded_scalar');initial=f'counts_init{thread}'
        add(f'rank_count_read{thread}',key,Operation.READ,(loop,),(loop,),(
            dep(initial,EventKind.WRITE_VISIBLE),dep(read),*scan_join,*prefix_join,
            dep(f'rank_increment{thread}',EventKind.WRITE_VISIBLE,carry=loop,initial=initial)),unmapped=True,index=index)
        add(f'rank_prefix_read{thread}',prefix_key,Operation.READ,(loop,),(loop,),(*prefix_join,dep(read)),unmapped=True,index=index)
        identity=allocate(f'rank_ssa{thread}','logical-register-state',None)
        rank=LogicalSelectionView(f'{scope.scope_id}.rank_ssa{thread}',identity,(n,),'int32',4,f'cuda.small_alignment.rank_ssa{thread}',None)
        views.append(rank)
        add(f'rank_value{thread}',rank,Operation.WRITE,(loop,),(loop,),
            (dep(f'rank_count_read{thread}'),dep(f'rank_prefix_read{thread}'),dep(read)),unmapped=True,domain='register')
        add(f'rank_load{thread}',rank,Operation.READ,(loop,),(loop,),
            (dep(f'rank_value{thread}',EventKind.WRITE_VISIBLE),),unmapped=True,domain='register')
        output=selection(f'sorted_rank{thread}',parents['sorted_token_ids'],n)
        add(f'sorted{thread}',output,Operation.WRITE,(loop,),(loop,),(
            dep(f'rank_load{thread}'),dep('sorted_initialize',EventKind.WRITE_VISIBLE,('padded',))),unmapped=True,
            index=TileValueIndex(f'rank_load{thread}','loaded_scalar'),domain='global')
        add(f'rank_increment{thread}',key,Operation.WRITE,(loop,),(loop,),
            (dep(f'rank_count_read{thread}'),dep(read)),unmapped=True,index=index)
    family=TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),bindings.visibility_domain,evidence)
    return tuple(allocations),tuple(views),tuple(events),family


def build_small_alignment_source(kernel,bindings,granularity,*,program_id):
    r,e,t,block,capacity,counts=_geometry(kernel)
    symbolic=block is None
    if (granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE)
            or bindings.read_views!=(('topk_ids','block_size_arg','capacity_arg') if symbolic else ('topk_ids',))
            or bindings.write_views!=('sorted_token_ids','expert_ids','num_tokens_post_padded')
            or len(bindings.input_ready)!=(3 if symbolic else 1)):raise ValueError('Small alignment public call boundary differs')
    header=('cuda_small_alignment',str(r),str(e),str(t),str(block),str(capacity),
        *(('unbound_native_arguments',) if symbolic else tuple(str(n) for n in counts)))
    scope=AccessScope(program_id,header,
        bindings.call_id,(),bindings.rank,bindings.stream,'logical-small-alignment')
    evidence=('Versioned CUDA one-CTA small-batch algorithm reference and installed no-map INT32 wrapper; actual selected W1 gives block size/capacity and declared routing counts.',
        'Worker row t+1 counts routed ordinals t+j*threads; each expert scans worker rows, thread0 publishes the block-rounded prefix, and worker t sorts from row t.',
        'Two real ID passes; shared private counters use ordered non-atomic RMWs. The actual final CTA join precedes sorting. No byte identity with the separate algorithm reference is claimed.',
        'Shared-relative offsets are logical arena offsets; native addresses, issued transactions/register allocation and concrete expert/rank values remain unknown.',
        'Without actual W1 context, block_size/max_num_tokens_padded remain scalar kernel arguments; owner/spare limits are scalar register results of prefix differences and block_size. INT32 envelopes are not allocation sizes or exact traffic.')
    allocations,views,events,family=_components(kernel,bindings,scope,evidence)
    result=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Unresolved actual expert/rank values require a value-aware consumer; no invented route membership.',
            'Logical register SSA is unpriced; default aggregate auxiliary pricing is retained explicitly.')),
        (*bindings.allocations,*allocations),(*bindings.views,*views),(),(*bindings.events,*events),(),(family,))
    small_alignment_source_demands(result,expected_kernel=kernel);return result


def small_alignment_source_demands(program,*,expected_kernel):
    from src.kernel.memory_access_builder import ElementwiseBindings
    kernel=expected_kernel;r,e,t,block,capacity,counts=_geometry(kernel);symbolic=block is None;program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:raise ValueError('Small alignment requires one unique tile source')
    family,=program.tile_programs;views=unique_by(program.views,'view_id');accesses=unique_by(family.accesses,'access_id')
    roles=('topk_ids','sorted_token_ids','expert_ids','num_tokens_post_padded')
    if symbolic:roles=(*roles,'block_size_arg','capacity_arg')
    parents=tuple(views[role] for role in roles)
    header=('cuda_small_alignment',str(r),str(e),str(t),str(block),str(capacity),
        *(('unbound_native_arguments',) if symbolic else tuple(str(n) for n in counts)))
    if program.hardware_id!=kernel.hardware_id or family.scope.op_ids!=header or family.scope.agent!='logical-small-alignment':
        raise ValueError('Small alignment source crossed its actual kernel')
    allocations=unique_by(program.allocations,'allocation_id')
    public=tuple(allocations[view.allocation_id] for view in parents);identities={a.allocation_id for a in public}
    events=tuple(event for event in program.events if event.allocation_id in identities)
    if 'count_id0' not in accesses or len(accesses['count_id0'].input_ready)!=1:raise ValueError('Small alignment lost its actual routed ID read')
    ready=accesses['count_id0'].input_ready
    if symbolic:ready=(*ready,*accesses['block_size_arg'].input_ready,*accesses['capacity_arg'].input_ready)
    binding=ElementwiseBindings(program.binding_id,family.scope.call_id,public,parents,events,
        ('topk_ids','block_size_arg','capacity_arg') if symbolic else ('topk_ids',),
        ('sorted_token_ids','expert_ids','num_tokens_post_padded'),ready,family.scope.rank,family.scope.stream,family.visibility_domain)
    scratch,selected,internal_events,expected=_components(kernel,binding,family.scope,family.evidence_refs)
    if (family!=expected or set(views)!=set(roles)|{v.view_id for v in selected}
            or any(views[v.view_id]!=v for v in selected)
            or any(allocations[a.allocation_id]!=a for a in scratch)
            or any(event not in program.events for event in internal_events)):
        raise ValueError('Small alignment lost actual private counter/prefix/second-pass rank dependencies')
    axes=unique_by(family.axes,'axis_id')
    def amount(operation,domain):
        if any(a.operation is operation and a.scope.agent==f'logical-small-alignment-{domain}'
                and any(axes[name].value_limit_access_id is not None for name in a.iteration_axes) for a in family.accesses):
            return None
        return sum(_tile_template_requested_bytes(views,axes,family,a) for a in family.accesses
            if a.operation is operation and a.scope.agent==f'logical-small-alignment-{domain}')
    return {'routed_items':r,'experts':e,'threads':t,'capacity':capacity,'read_bytes':amount(Operation.READ,'global'),
        'write_bytes':amount(Operation.WRITE,'global'),'shared_read_bytes':amount(Operation.READ,'shared'),
        'shared_write_bytes':amount(Operation.WRITE,'shared'),'register_read_bytes':amount(Operation.READ,'register'),
        'register_write_bytes':amount(Operation.WRITE,'register'),'legacy_aggregate_read_price_bytes':e*8+r*16}


def _regular_geometry(kernel):
    from src.kernel.launch_config.select_launch_config import moe_alignment_capacity,moe_count_sort_source_program
    s=kernel.shape;a=kernel.attributes;r=s['tokens']*s['topk'];e=s['experts']
    if (kernel.kind.value!='VECTOR' or kernel.variant!='moe_align' or kernel.execution_support!='supported'
            or kernel.hardware_id not in ('nvidia_a800','nvidia_h200') or not 0<e<1024 or r<=0
            or kernel.launch.algorithm not in ('moe_align','observed_moe_aux:moe_align')
            or kernel.launch.block!=(1024,1,1)
            or kernel.launch.grid not in (((1,1,1),(2,1,1)) if kernel.hardware_id=='nvidia_h200' else ((1,1,1),))
            or moe_count_sort_source_program(kernel.hardware_id)!='cuda_moe_count_sort_int32_unmasked'):
        raise ValueError('Ordinary alignment requires its actual CUDA 1024-worker count/scan source')
    context={'alignment_block_size','alignment_capacity','tokens_per_expert'}
    if not context&set(a):return r,e,None,None,None
    if not context<=set(a):raise ValueError('Ordinary alignment requires complete actual W1 context')
    block,capacity,counts=(a[name] for name in ('alignment_block_size','alignment_capacity','tokens_per_expert'))
    if (capacity!=moe_alignment_capacity(kernel.hardware_id,s['tokens'],s['topk'],e,block)
            or len(counts)!=e or sum(counts)!=r or sum(-(-n//block)*block for n in counts)>capacity):
        raise ValueError('Ordinary alignment differs from actual wrapper allocation/routing facts')
    return r,e,block,capacity,counts


def _regular_prediction_demand(geometry):
    """The retained analytical price, independent of private scan execution.

    Four INT32 route operations and two INT32 expert-counter operations are
    the existing aggregate price. Private raking/register requests are an
    inspection projection, not additional DRAM work in this model.
    """
    r,e,block,capacity,counts=geometry
    return {'routed_items':r,'experts':e,'capacity':capacity,
        'legacy_aggregate_read_price_bytes':e*8+r*16}


def regular_alignment_prediction_work(kernel,binding):
    """Validate the actual public boundary and directly evaluate its demand.

    The private histogram/warp-scan graph is only needed for explicit source
    inspection. Normal prediction uses the same retained demand expression
    as that graph's consumer, with the actual caller layout and readiness.
    """
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_access import covers,Unmapped,validate_storage_bindings
    if not isinstance(binding,ElementwiseBindings):
        raise TypeError('Ordinary alignment requires typed elementwise bindings')
    binding.__post_init__()
    geometry=_regular_geometry(kernel);symbolic=geometry[2] is None
    reads=('topk_ids','block_size_arg','capacity_arg') if symbolic else ('topk_ids',)
    if (binding.read_views!=reads or len(binding.input_ready)!=len(reads)
            or binding.write_views!=('cumsum','sorted_token_ids','expert_ids','num_tokens_post_padded')):
        raise ValueError('Ordinary alignment requires its actual INT32 input/output call boundary')
    parents,allocations=_regular_public_bindings(geometry,binding)
    # Only the actual public storage boundary is checked. No access program,
    # private state, tile or scan graph is constructed.
    validate_storage_bindings(binding.allocations,binding.views,binding.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Actual ordinary-alignment public storage; private scan execution is not materialized.',)))
    events=unique_by(binding.events,'event_id')
    for role,ref in zip(reads,binding.input_ready,strict=True):
        view=parents[role]
        if ref.node_id not in events:
            raise ValueError('Ordinary alignment input readiness event is missing')
        event=events[ref.node_id]
        if (ref.kind is not event.kind or ref.kind not in
                (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE,EventKind.LOGICAL_AVAILABLE)
                or ref.allow_skipped_read or event.allocation_id!=view.allocation_id
                or event.visibility_domain!=binding.visibility_domain
                or event.view_id is not None and event.view_id!=view.view_id):
            raise ValueError('Ordinary alignment input readiness allocation/visibility mismatch')
        if isinstance(event.region,Unmapped):
            if ((event.region.logical_shape,event.region.dtype,event.region.element_bytes)
                    !=(view.shape,view.dtype,view.element_bytes)):
                raise ValueError('Ordinary alignment requires its complete typed input readiness')
        elif event.region is None or not covers(event.region,view.region()):
            raise ValueError('Ordinary alignment requires its complete input readiness')
    return _regular_prediction_demand(geometry)


def regular_alignment_bindings(kernel,*,call_id,binding_id, device_id="logical-local-device", rank=0, stream="offline"):
    from src.kernel.memory_access_builder import ElementwiseBindings
    r,e,block,capacity,counts=_regular_geometry(kernel);symbolic=block is None;upper=2**31-1
    layouts=(('topk_ids',r),('cumsum',e+1),('sorted_token_ids',upper if symbolic else capacity),
        ('expert_ids',upper if symbolic else -(-capacity//block)),('num_tokens_post_padded',1))
    if symbolic:layouts=(*layouts,('block_size_arg',1),('capacity_arg',1))
    allocations=[];views=[];events=[];ready=[]
    for role,n in layouts:
        identity=f'{binding_id}.{role}';unknown=symbolic and role in ('sorted_token_ids','expert_ids')
        allocation=MemoryAllocation(identity,device_id,'logical-alignment-interface',None if unknown else n*4,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        view=(LogicalSelectionView(role,identity,(n,),'int32',4,identity+'.unknown_capacity',None) if unknown
            else TensorView(role,identity,0,(n,),(4,),'int32',4))
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role=='topk_ids' or role.endswith('_arg'):
            event=ProgramEvent(identity+'.initialized',EventKind.INITIALIZED,identity,view.region(),'moe-alignment-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),));events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('topk_ids','block_size_arg','capacity_arg') if symbolic else ('topk_ids',),
        ('cumsum','sorted_token_ids','expert_ids','num_tokens_post_padded'),tuple(ready),rank,stream,'moe-alignment-visible')


def _regular_public_bindings(geometry,binding):
    r,e,block,capacity,counts=geometry;symbolic=block is None;upper=2**31-1
    parents=unique_by(binding.views,'view_id');public=unique_by(binding.allocations,'allocation_id')
    layouts=(('topk_ids',r),('cumsum',e+1),('sorted_token_ids',upper if symbolic else capacity),
        ('expert_ids',upper if symbolic else -(-capacity//block)),('num_tokens_post_padded',1))
    if symbolic:layouts=(*layouts,('block_size_arg',1),('capacity_arg',1))
    if set(parents)!={role for role,_ in layouts} or len({v.allocation_id for v in parents.values()})!=len(parents):
        raise ValueError('Ordinary alignment public storage roles must be exact and distinct')
    for role,n in layouts:
        view=parents[role]
        if symbolic and role in ('sorted_token_ids','expert_ids'):
            if (not isinstance(view,LogicalSelectionView) or (view.shape,view.dtype,view.element_bytes)!=((n,),'int32',4)
                    or public[view.allocation_id].size_bytes is not None or view.parent_view_id is not None):
                raise ValueError('Unbound ordinary alignment output must preserve its unknown-capacity INT32 selection')
        elif not isinstance(view,TensorView) or (view.shape,view.strides_bytes,view.dtype,view.element_bytes)!=((n,),(4,),'int32',4):
            raise ValueError('Ordinary alignment public operands must retain their actual INT32 layout')
    return parents,public


@runtime_lru_cache(maxsize=16)
def _cached_regular_components(geometry,binding,scope,evidence):
    """Share immutable raking structure, retaining each actual invocation.

    Only the bare ABI's route extent is a graph parameter. Its two selected
    views and route axis are rebuilt with their actual geometry below; all
    public storage, readiness, private identities and actor context are bound
    anew. The resulting complete program still passes ordinary validation.
    """
    _regular_public_bindings(geometry,binding)
    r,e,block,capacity,counts=geometry
    if block is not None:
        # A bound ABI already has its exact geometry. Build directly in its
        # actual namespace; the common scan prefix is shared independently of
        # counts, so copying and renaming that same prefix buys no reuse.
        return _regular_template_components(geometry,binding,scope,evidence)
    canonical_geometry=(1,e,block,capacity,counts) if block is None else geometry
    canonical_ids={a.allocation_id:f'regular.parent.{i}' for i,a in enumerate(binding.allocations)}
    input_allocation=next(v.allocation_id for v in binding.views if v.view_id=='topk_ids')
    canonical_allocations=tuple(replace(a,allocation_id=canonical_ids[a.allocation_id],
        allocated_event=canonical_ids[a.allocation_id]+'.allocated',
        freed_event=None if a.freed_event is None else canonical_ids[a.allocation_id]+'.freed',
        size_bytes=(None if a.size_bytes is None else 4) if block is None and a.allocation_id==input_allocation else a.size_bytes)
        for a in binding.allocations)
    canonical_views=[]
    for view in binding.views:
        changes={'allocation_id':canonical_ids[view.allocation_id]}
        if isinstance(view,LogicalSelectionView):changes['selection_id']='regular.parent.'+view.view_id
        if block is None and view.view_id=='topk_ids':changes['shape']=(1,)
        canonical_views.append(replace(view,**changes))
    canonical_ready=tuple(EventRef(f'regular.ready.{i}',ref.kind) for i,ref in enumerate(binding.input_ready))
    canonical_binding=replace(binding,binding_id='regular.recipe',call_id='regular.call',
        allocations=canonical_allocations,views=tuple(canonical_views),events=(),input_ready=canonical_ready,
        rank=0,stream='regular.stream',visibility_domain='regular.visibility')
    canonical_scope=replace(scope,scope_id='regular.alignment.recipe',op_ids=_regular_geometry_header(canonical_geometry),
        call_id='regular.call',rank=0,stream='regular.stream')
    allocations,views,events,family=_regular_template_components(
        canonical_geometry,canonical_binding,canonical_scope,evidence)
    public_ids={canonical_ids[a.allocation_id]:a.allocation_id for a in binding.allocations}
    prefix=f'regular.recipe.{canonical_scope.scope_id}.'
    actual_prefix=f'{binding.binding_id}.{scope.scope_id}.'
    def actual_identity(identity):
        if identity in public_ids:return public_ids[identity]
        if not identity.startswith(prefix):raise ValueError('Alignment template lost its actual storage namespace')
        return actual_prefix+identity[len(prefix):]
    public_view_ids=frozenset(v.view_id for v in binding.views)
    def actual_view(identity):
        if identity is None or identity in public_view_ids:return identity
        prefix=canonical_scope.scope_id+'.'
        if not identity.startswith(prefix):raise ValueError('Alignment template lost its actual view namespace')
        return scope.scope_id+identity[len(canonical_scope.scope_id):]
    allocations=tuple(_replace_source_identity(a,allocation_id=actual_identity(a.allocation_id),
        allocated_event=actual_identity(a.allocated_event)) for a in allocations)
    selected=[]
    for view in views:
        changes={'view_id':actual_view(view.view_id),'allocation_id':actual_identity(view.allocation_id)}
        if isinstance(view,LogicalSelectionView):changes['parent_view_id']=actual_view(view.parent_view_id)
        actual=_replace_source_identity(view,**changes)
        if block is None and r!=1 and actual.view_id in (
                f'{scope.scope_id}.histogram_keys',f'{scope.scope_id}.source_ids'):
            actual=replace(actual,shape=(r,))
        selected.append(actual)
    events=tuple(_replace_source_identity(event,event_id=actual_identity(event.event_id),
        allocation_id=actual_identity(event.allocation_id)) for event in events)
    ready=dict(zip((ref.node_id for ref in canonical_ready),binding.input_ready,strict=True))
    accesses=tuple(_replace_source_identity(access,view_id=actual_view(access.view_id),
        scope=_replace_source_identity(access.scope,scope_id=actual_view(access.scope.scope_id),
            call_id=scope.call_id,rank=scope.rank,stream=scope.stream),
        input_ready=tuple(ready[ref.node_id] for ref in access.input_ready)) for access in family.accesses)
    axes=tuple(replace(axis,extent=r) if block is None and axis.axis_id=='route' else axis for axis in family.axes)
    rebound=_replace_source_identity(family,tile_program_id=scope.scope_id,
        scope=scope,visibility_domain=binding.visibility_domain,accesses=accesses)
    if axes!=family.axes:rebound=replace(rebound,axes=axes)
    return allocations,tuple(selected),events,rebound


@runtime_lru_cache(maxsize=8)
def _regular_template_components(geometry,binding,scope,evidence,_prefix_only=False):
    """Share the complete fixed scan, then append actual expert-owner loops.

    Prefix actors declare their own operation IDs, so the outer count header
    is absent from that cache key. The returned full program retains its
    original scope/header, current owner extents and every scan dependency.
    """
    from src.kernel.memory_access import TileAtomicAdd
    r,e,block,capacity,counts=geometry;symbolic=block is None;upper=2**31-1
    parents,public=_regular_public_bindings(geometry,binding)
    device=public[parents['topk_ids'].allocation_id].device_id
    allocations=[];views=[];events=[];accesses=[]
    axes=[TileAxis('route',r,1),TileAxis('worker',1024,1),
        TileAxis('padded',upper if symbolic else capacity,1,value_limit_access_id='capacity_arg' if symbolic else None)]
    def allocate(name,n,domain):
        identity=f'{binding.binding_id}.{scope.scope_id}.{name}'
        allocation=MemoryAllocation(identity,device,domain,None if n is None else n*4,AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        return identity
    def vector(name,allocation,n,offset=0):
        view=TensorView(f'{scope.scope_id}.{name}',allocation,offset,(n,),(4,),'int32',4);views.append(view);return view
    def selected(name,parent,n):
        view=LogicalSelectionView(f'{scope.scope_id}.{name}',parent.allocation_id,(n,),'int32',4,
            f'cuda.regular_alignment.{name}',parent.view_id);views.append(view);return view
    def dep(name,kind=EventKind.READ_READY,reduction=(),bindings=()):return TileDependency(name,kind,reduction,bindings)
    def add(name,view,operation,loops=(),mapping=(None,),deps=(),*,ready=(),unmapped=False,value=None,index=None,atomic=None,domain='shared'):
        actor=replace(scope,scope_id=f'{scope.scope_id}.{name}.scope',op_ids=('cuda_regular_alignment',name),
            agent=f'logical-regular-alignment-{domain}')
        accesses.append(TileAccessTemplate(name,view.view_id,operation,loops,mapping,ready,deps,evidence,
            unmapped=unmapped,constant_value=value,value_index=index,atomic_add=atomic,scope=actor))
    def register(name,n,deps,*,value=None):
        identity=allocate(name,None,'logical-register-state')
        view=LogicalSelectionView(f'{scope.scope_id}.{name}',identity,(n,),'int32',4,f'cuda.regular_alignment.{name}',None);views.append(view)
        add(name+'.write',view,Operation.WRITE,deps=deps,unmapped=True,value=value,domain='register')
        add(name,view,Operation.READ,deps=(dep(name+'.write',EventKind.WRITE_VISIBLE),),unmapped=True,domain='register')
        return name
    if _prefix_only:
        if symbolic:
            for index,role in enumerate(('block_size_arg','capacity_arg'),1):
                add(role,parents[role],Operation.READ,ready=(binding.input_ready[index],),domain='argument')
        histogram=vector('histogram',allocate('histogram',-(-e//32)*32,'logical-shared-histogram'),-(-e//32)*32)
        keys=selected('histogram_keys',histogram,r)
        add('histogram_init',histogram,Operation.WRITE,value=ConstantValue.ZERO)
        source_ids=parents['topk_ids']
        unresolved=public[source_ids.allocation_id].size_bytes is None
        if unresolved:source_ids=selected('source_ids',source_ids,r)
        add('expert_read',source_ids,Operation.READ,('route',),('route',),ready=(binding.input_ready[0],),
            unmapped=unresolved,domain='global')
        key=TileValueIndex('expert_read','loaded_scalar')
        add('histogram_read',keys,Operation.READ,('route',),('route',),(dep('histogram_init',EventKind.WRITE_VISIBLE),dep('expert_read')),unmapped=True,index=key)
        add('histogram_increment',keys,Operation.WRITE,('route',),('route',),(dep('histogram_read'),dep('expert_read')),
            unmapped=True,index=key,atomic=TileAtomicAdd('histogram_read',1))
        add('sorted_initialize',parents['sorted_token_ids'],Operation.WRITE,('padded',),('padded',),
            (dep('capacity_arg'),) if symbolic else (),unmapped=symbolic,domain='global')
        # A logical 32x32 raking arena records scalar payload slots. Native
        # alignment/padding and addresses are not inferred from this object.
        grid=allocate('raking_grid',1024,'logical-shared-raking-grid')
        rows=[vector(f'raking_row{lane}',grid,32,lane*32*4) for lane in range(32)]
        for lane,row in enumerate(rows):
            first=lane*32;n=min(32,max(0,e-first));inputs=()
            if n:
                bins=vector(f'histogram_group{lane}',histogram.allocation_id,n,first*4)
                add(f'histogram_group{lane}',bins,Operation.READ,deps=(dep('histogram_init',EventKind.WRITE_VISIBLE),
                    dep('histogram_increment',EventKind.WRITE_VISIBLE,('route',))))
                inputs=(dep(f'histogram_group{lane}'),*((dep('block_size_arg'),) if symbolic else ()))
            add(f'placement{lane}',row,Operation.WRITE,deps=inputs,value=None if n else ConstantValue.ZERO)
        placement_join=tuple(dep(f'placement{lane}',EventKind.WRITE_VISIBLE) for lane in range(32))
        partials=[]
        for lane,row in enumerate(rows):
            add(f'upsweep_load{lane}',row,Operation.READ,deps=placement_join)
            cache=register(f'upsweep_cache{lane}',32,(dep(f'upsweep_load{lane}'),))
            partials.append(register(f'upsweep_sum{lane}',1,(dep(cache),)))
        # WarpScan<int32,32> shuffle scan: each lane keeps its own previous
        # value and only lanes >= distance consume that previous peer value.
        for stage,distance in enumerate((1,2,4,8,16)):
            previous=partials;partials=[]
            for lane in range(32):
                inputs=(dep(previous[lane]),*((dep(previous[lane-distance]),) if lane>=distance else ()))
                partials.append(register(f'warp_stage{stage}.lane{lane}',1,inputs))
        downsweep=[]
        for lane,row in enumerate(rows):
            prefix=register(f'exclusive_prefix{lane}',1,(dep(partials[lane-1]),) if lane else (dep(partials[0]),),
                value=ConstantValue.ZERO if lane==0 else None)
            add(f'downsweep_load{lane}',row,Operation.READ,deps=(dep(f'placement{lane}',EventKind.WRITE_VISIBLE),dep(f'upsweep_load{lane}'),dep(prefix)))
            cache=register(f'downsweep_cache{lane}',32,(dep(f'downsweep_load{lane}'),))
            scanned=register(f'downsweep_scanned{lane}',32,(dep(cache),dep(prefix)))
            add(f'downsweep_store{lane}',row,Operation.WRITE,deps=(dep(scanned),))
            downsweep.append(dep(f'downsweep_store{lane}',EventKind.WRITE_VISIBLE))
        for lane,row in enumerate(rows):add(f'copyback{lane}',row,Operation.READ,deps=tuple(downsweep))
        prefix_writes=[]
        for expert in range(e+1):
            output=vector(f'prefix{expert}',parents['cumsum'].allocation_id,1,parents['cumsum'].offset_bytes+expert*4)
            add(f'prefix{expert}',output,Operation.WRITE,deps=(dep(f'copyback{expert//32}'),),domain='global')
            prefix_writes.append(dep(f'prefix{expert}',EventKind.WRITE_VISIBLE))
        end=vector('prefix_end',parents['cumsum'].allocation_id,1,parents['cumsum'].offset_bytes+e*4)
        add('padded_count',parents['num_tokens_post_padded'],Operation.WRITE,deps=(dep(f'copyback{e//32}'),),domain='global')
        add('inactive_start',end,Operation.READ,('worker',),(None,),(prefix_writes[-1],),domain='global')
        return tuple(allocations),tuple(views),tuple(events),tuple(axes),tuple(accesses),tuple(prefix_writes)
    prefix_geometry=(r,e,block,capacity,None)
    prefix_scope=replace(scope,op_ids=_regular_geometry_header(prefix_geometry))
    common=_regular_template_components(prefix_geometry,binding,prefix_scope,evidence,True)
    allocations,views,events,axes,accesses=(list(values) for values in common[:5])
    prefix_writes=common[5]
    offset=0
    for expert in range(e):
        pair=vector(f'owner_bounds{expert}',parents['cumsum'].allocation_id,2,parents['cumsum'].offset_bytes+expert*4)
        add(f'owner_bounds{expert}',pair,Operation.READ,deps=(prefix_writes[expert],prefix_writes[expert+1]),domain='global')
        if symbolic:
            name=register(f'owner{expert}.block_count',1,(dep(f'owner_bounds{expert}'),dep('block_size_arg')))
            loop=f'owner{expert}.block';axes.append(TileAxis(loop,upper,1,value_limit_access_id=name))
            output=selected(f'owner{expert}',parents['expert_ids'],upper)
            add(f'owner{expert}',output,Operation.WRITE,(loop,),(loop,),(dep(name),),unmapped=True,domain='global')
        else:
            n=-(-counts[expert]//block)
            if n:
                loop=f'owner{expert}.block';axes.append(TileAxis(loop,n,1))
                output=vector(f'owner{expert}',parents['expert_ids'].allocation_id,n,parents['expert_ids'].offset_bytes+offset*4)
                add(f'owner{expert}',output,Operation.WRITE,(loop,),(loop,),(dep(f'owner_bounds{expert}'),),domain='global');offset+=n
    if symbolic:
        name=register('inactive_block_count',1,(dep('inactive_start',bindings=(TileIndexBinding('worker',(),0),)),dep('block_size_arg'),dep('capacity_arg')))
        axes.append(TileAxis('inactive_block',upper,1,value_limit_access_id=name));output=selected('inactive_blocks',parents['expert_ids'],upper)
        add('inactive_blocks',output,Operation.WRITE,('inactive_block',),('inactive_block',),(dep(name),),unmapped=True,domain='global')
    else:
        spare=parents['expert_ids'].shape[0]-offset
        if spare:
            axes.append(TileAxis('inactive_block',spare,1))
            output=vector('inactive_blocks',parents['expert_ids'].allocation_id,spare,parents['expert_ids'].offset_bytes+offset*4)
            add('inactive_blocks',output,Operation.WRITE,('inactive_block',),('inactive_block',),
                (dep('inactive_start',bindings=(TileIndexBinding('worker',(TileIndexTerm('inactive_block',1),),modulus=1024),)),),domain='global')
    return tuple(allocations),tuple(views),tuple(events),TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),binding.visibility_domain,evidence)


def _regular_geometry_header(geometry):
    r,e,block,capacity,counts=geometry
    return ('cuda_regular_alignment',str(r),str(e),str(block),str(capacity),
        *(('unbound_native_arguments',) if counts is None else tuple(str(n) for n in counts)))


def build_regular_alignment_source(kernel,binding,granularity,*,program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(binding,ElementwiseBindings):raise TypeError('Ordinary alignment requires typed elementwise bindings')
    geometry=_regular_geometry(kernel);r,e,block,capacity,counts=geometry;symbolic=block is None
    if (granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE)
            or binding.read_views!=(('topk_ids','block_size_arg','capacity_arg') if symbolic else ('topk_ids',))
            or binding.write_views!=('cumsum','sorted_token_ids','expert_ids','num_tokens_post_padded')
            or len(binding.input_ready)!=(3 if symbolic else 1)):
        raise ValueError('Ordinary alignment requires its actual INT32 input/output call boundary')
    result=_cached_regular_source(geometry,kernel.hardware_id,binding,program_id)
    regular_alignment_source_demands(result,expected_kernel=kernel);return result


@runtime_lru_cache(maxsize=16)
def _cached_regular_source(geometry,hardware_id,binding,program_id):
    scope=AccessScope(program_id,_regular_geometry_header(geometry),binding.call_id,(),binding.rank,binding.stream,'logical-regular-alignment')
    evidence=('Versioned CUDA ordinary alignment and CUB 2.2 BlockScan<int32,1024> default non-memoized raking algorithm references; no installed binary byte identity claim.',
        'The 32x32 logical shared payload grid, up/down cached register segments, warp shuffle prefixes, copyback and public prefix/owner controls retain source dependencies. Native padding/issue/register allocation remain unknown.',
        'BM/capacity and owner/fill loop values remain explicit scalar inputs/results when the standalone auxiliary context is absent; logical requests do not replace the explicit retained aggregate price.')
    allocations,views,events,family=_cached_regular_components(geometry,binding,scope,evidence)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware_id,binding.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Logical versioned raking reference; physical scratch offsets/padding, issued loads and concrete expert/index/loop values are not inferred.',)),
        (*binding.allocations,*allocations),(*binding.views,*views),(),(*binding.events,*events),(),(family,))


@runtime_lru_cache(maxsize=16)
def _validate_regular_program(program):
    if not isinstance(program,MemoryAccessProgram):raise TypeError('Ordinary alignment requires a typed memory access program')
    program.__post_init__()


def regular_alignment_source_demands(program,*,expected_kernel):
    return dict(_regular_alignment_source_demands(program,expected_kernel.hardware_id,
        _regular_geometry(expected_kernel)))


@runtime_lru_cache(maxsize=16)
def _regular_alignment_source_demands(program,hardware_id,geometry):
    from src.kernel.memory_access_builder import ElementwiseBindings
    r,e,block,capacity,counts=geometry;symbolic=block is None;_validate_regular_program(program)
    if program.accesses or program.scopes or len(program.tile_programs)!=1:raise ValueError('Ordinary alignment requires its unique compressed count/raking source')
    family,=program.tile_programs;views=unique_by(program.views,'view_id');accesses=unique_by(family.accesses,'access_id')
    roles=('topk_ids','cumsum','sorted_token_ids','expert_ids','num_tokens_post_padded')
    if symbolic:roles=(*roles,'block_size_arg','capacity_arg')
    parents=tuple(views[role] for role in roles);allocations=unique_by(program.allocations,'allocation_id')
    public=tuple(allocations[view.allocation_id] for view in parents);identities={a.allocation_id for a in public}
    ready=accesses['expert_read'].input_ready
    if symbolic:ready=(*ready,*accesses['block_size_arg'].input_ready,*accesses['capacity_arg'].input_ready)
    binding=ElementwiseBindings(program.binding_id,family.scope.call_id,public,parents,
        tuple(event for event in program.events if event.allocation_id in identities),
        ('topk_ids','block_size_arg','capacity_arg') if symbolic else ('topk_ids',),
        ('cumsum','sorted_token_ids','expert_ids','num_tokens_post_padded'),ready,family.scope.rank,family.scope.stream,family.visibility_domain)
    scratch,selected,internal,expected=_cached_regular_components(geometry,binding,family.scope,family.evidence_refs)
    events=unique_by(program.events,'event_id')
    if (program.hardware_id!=hardware_id or family.scope.op_ids!=_regular_geometry_header(geometry)
            or family.scope.agent!='logical-regular-alignment' or family!=expected
            or set(views)!=set(roles)|{v.view_id for v in selected} or any(views[v.view_id]!=v for v in selected)
            or any(allocations[a.allocation_id]!=a for a in scratch)
            or any(event.event_id not in events or events[event.event_id]!=event for event in internal)):
        raise ValueError('Ordinary alignment lost count/raking/cache/prefix dependencies')
    axes=unique_by(family.axes,'axis_id')
    def amount(operation,domain):
        templates=tuple(a for a in family.accesses if a.operation is operation and a.scope.agent==f'logical-regular-alignment-{domain}')
        if any(axes[name].value_limit_access_id is not None for a in templates for name in a.iteration_axes):return None
        return sum(_tile_template_requested_bytes(views,axes,family,a) for a in templates)
    return {**_regular_prediction_demand(geometry),
        **{f'{domain+"_" if domain!="global" else ""}{operation.value}_bytes':amount(operation,domain)
            for domain in ('global','shared','register') for operation in (Operation.READ,Operation.WRITE)},
        }
