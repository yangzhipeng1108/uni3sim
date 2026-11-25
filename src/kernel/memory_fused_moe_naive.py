"""BF16 grouped callee controls for the source-known token-wise assignment."""
from dataclasses import replace
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,MemoryAccessProgram,
    MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,TensorView,Unmapped,
    LogicalSelectionView,TileAxis,TileAccessTemplate,TileDependency,TileProgram,unique_by,
)


def has_h200_bf16_controls(kernel):
    return (kernel.kind is KernelKind.FUSED_MOE and kernel.hardware_id=='nvidia_h200'
        and kernel.execution_support=='supported' and kernel.dtype=='bf16' and kernel.variant in ('w1','w2')
        and ('routed_weight_dtype' not in kernel.attributes or kernel.attributes['routed_weight_dtype']=='bf16')
        and (kernel.launch.algorithm in ('triton_grouped_gemm:w1','triton_grouped_gemm:w2','triton_fused_moe_w1','triton_fused_moe_w2')
            or (kernel.attributes['expert_parallel_size']==1
                and kernel.launch.algorithm==f'vllm_0_11_sm90_unquantized_{kernel.variant}')))


def h200_bf16_naive(kernel):
    # The0.11 callee always consumes sorted IDs; its small-M launch cap is
    # independent of the0.24 token-wise assignment optimization.
    if kernel.launch.algorithm in ('vllm_0_11_sm90_unquantized_w1', 'vllm_0_11_sm90_unquantized_w2'):
        return False
    return kernel.attributes['expert_parallel_size']==1 and kernel.shape['tokens']*kernel.shape['topk']*4<=kernel.shape['experts']


def _header(kernel):
    from src.kernel.memory_fused_moe import _native_moe_header
    header=_native_moe_header(kernel)
    if h200_bf16_naive(kernel):
        header=(*header[:11],str(kernel.shape['tokens']*kernel.shape['topk']),*header[12:],'naive')
    return header


def _control_layouts(header):
    from src.kernel.memory_fused_moe import _native_moe_control_layouts
    if header[-1]!='naive':return _native_moe_control_layouts(header)
    t,e,topk,k,n,bm,bn,bk,grid_m,routes=map(int,header[2:12])
    return (('expert_ids',(routes,),'int32',4),('num_tokens_post_padded',(1,),'int32',4),
        *((('router_weight',(t,topk),'fp32',4),) if header[1]=='w2' else ()))


def standalone_h200_bf16_bindings(kernel,hardware, *, call_id,binding_id):
    from src.kernel.memory_fused_moe import FusedMoEBindings,_layouts
    header=_header(kernel);primary='expert_ids' if header[-1]=='naive' else 'sorted_token_ids'
    layouts=tuple((role,shape,'bf16',2) for role,shape in _layouts(kernel,hardware))+_control_layouts(header)
    allocations=[];views=[];events=[];ready={}
    for role,shape,dtype,width in layouts:
        identity=f'{binding_id}.routing' if role==primary else f'{binding_id}.{role}'
        unknown=role in ('expert_ids','sorted_token_ids');size=width;strides=[]
        for extent in reversed(shape):strides.append(size);size*=extent
        allocation=MemoryAllocation(identity,'logical-local-device','logical-native-moe-control' if unknown else 'logical-storage',
            None if unknown else size,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role!='output':
            kind=EventKind.LOGICAL_AVAILABLE if unknown else EventKind.INITIALIZED
            region=Unmapped(shape,dtype,width,f'{identity}.required_logical_prefix') if unknown else view.region()
            event=ProgramEvent(f'{identity}.available',kind,identity,region,'moe-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready[role]=EventRef(event.event_id,event.kind)
    return FusedMoEBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),'query','weight','output',
        f'{binding_id}.routing',ready['query'],ready['weight'],ready[primary],0,'offline','moe-visible')


def _naive_recipe(scope,parents,ready,visibility):
    header=scope.op_ids
    if len(header)!=15 or header[0]!='native_grouped_moe' or header[-1]!='naive' or header[1] not in ('w1','w2'):
        raise ValueError('BF16 naive source requires its explicit token-wise assignment')
    t,e,topk,k,n,bm,bn,bk,grid_m,routes=map(int,header[2:12]);counts=tuple(map(int,header[12].split(',')))
    if (min(t,e,topk,k,n,bm,bn,bk,grid_m,routes)<=0 or len(counts)!=e or min(counts)<0
            or sum(counts)!=routes or routes!=t*topk or routes*4>e or grid_m<routes
            or header[13]!=str(int(header[1]=='w2'))):raise ValueError('BF16 naive routing/native grid differs')
    evidence=('vLLM0.24 _prepare_expert_assignment: no expert_map and routes*4<=experts selects one valid token per M program, no sorted_token_ids pointer.',
        'fused_moe.py:396-415 reads padded-count, then expert_ids for each active native M/N program. Other BM lanes are masked; expert-local selections do not resolve original token order.',
        'Full-K matrix reads precede W2 FP32 router read and its output store. Existing launch/cache prices remain a separate projection, including surplus count-only programs.')
    selected=[];families=[];dep=lambda role:TileDependency(role,EventKind.READ_READY,())
    for expert,count in enumerate(counts):
        if not count:continue
        identity=f'{scope.scope_id}.e{expert}';slices={}
        roles=(('expert_ids',(count,)),('query',(count,k)),('output',(count,n)))
        if header[1]=='w2':roles+=(('router_weight',(count,)),)
        for role,shape in roles:
            parent=parents[role]
            child=LogicalSelectionView(f'{identity}.{role}',parent.allocation_id,shape,parent.dtype,parent.element_bytes,
                f'{identity}.{role}.unsorted_route_ordinal',parent.view_id)
            selected.append(child);slices[role]=child
        parent=parents['weight'];child=TensorView(f'{identity}.weight',parent.allocation_id,parent.offset_bytes+expert*n*k*2,
            (n,k),(k*2,2),'bf16',2);selected.append(child);slices['weight']=child
        actor=('row','column')
        accesses=[TileAccessTemplate('num_tokens_post_padded',parents['num_tokens_post_padded'].view_id,Operation.READ,
            actor,(None,),(ready['num_tokens_post_padded'],),(),evidence),
            TileAccessTemplate('expert_ids',slices['expert_ids'].view_id,Operation.READ,actor,('row',),
                (ready['expert_ids'],),(dep('num_tokens_post_padded'),),evidence,unmapped=True)]
        for role,mapping in (('query',('row',None)),('weight',('column',None))):
            accesses.append(TileAccessTemplate(role,slices[role].view_id,Operation.READ,actor,mapping,(ready[role],),
                (dep('expert_ids'),),evidence,unmapped=role=='query'))
        output_deps=(dep('query'),dep('weight'))
        if header[1]=='w2':
            accesses.append(TileAccessTemplate('router_weight',slices['router_weight'].view_id,Operation.READ,actor,('row',),
                (ready['router_weight'],),output_deps,evidence,unmapped=True));output_deps+=(dep('router_weight'),)
        accesses.append(TileAccessTemplate('output',slices['output'].view_id,Operation.WRITE,actor,('row','column'),(),output_deps,evidence,unmapped=True))
        local=replace(scope,scope_id=identity,op_ids=(*header,str(expert)),agent='native-naive-bf16-blocks')
        families.append(TileProgram(identity,local,(TileAxis('row',count,1),TileAxis('column',n,bn)),tuple(accesses),visibility,evidence))
    if grid_m>routes:
        identity=f'{scope.scope_id}.inactive';local=replace(scope,scope_id=identity,agent='native-naive-bf16-early-return')
        families.append(TileProgram(identity,local,(TileAxis('row',grid_m-routes,1),TileAxis('column',n,bn)),
            (TileAccessTemplate('num_tokens_post_padded',parents['num_tokens_post_padded'].view_id,Operation.READ,
                ('row','column'),(None,),(ready['num_tokens_post_padded'],),(),evidence),),visibility,evidence))
    return tuple(selected),tuple(families)


def build_h200_bf16_source(kernel,hardware,bindings,views, *, program_id):
    from src.kernel.memory_fused_moe import build_native_grouped_moe_source
    if not h200_bf16_naive(kernel):return build_native_grouped_moe_source(kernel,hardware,bindings,views,program_id=program_id)
    header=_header(kernel);parents={role:views[role] for role in ('query','weight','output',*(r for r,*_ in _control_layouts(header)))}
    ready={'query':bindings.query_ready,'weight':bindings.weight_ready,'expert_ids':bindings.routing_ready}
    for role in ('num_tokens_post_padded',*(('router_weight',) if header[1]=='w2' else ())):
        event,=(event for event in bindings.events if event.allocation_id==parents[role].allocation_id
            and event.kind in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE,EventKind.LOGICAL_AVAILABLE))
        ready[role]=EventRef(event.event_id,event.kind)
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'native-naive-bf16-interface')
    selected,families=_naive_recipe(scope,parents,ready,bindings.visibility_domain)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Source-known token-wise assignment has one valid row per native M program and no sorted-ID reads.',
            'Expert IDs and post-padding scalar are typed invocation-boundary inputs; native offsets, capacity and full auxiliary producers remain unknown.',
            'Register full-K accumulation has no fabricated GM scratch. Logical requests do not change inherited launch/cache/DRAM prices.')),
        bindings.allocations,(*bindings.views,*selected),(scope,),bindings.events,(),families)
    h200_naive_demands(program);return program


def h200_naive_demands(program):
    from src.kernel.memory_access import _tile_template_requested_bytes
    program.__post_init__();scope,=program.scopes;header=scope.op_ids
    if program.hardware_id!='nvidia_h200':raise ValueError('BF16 naive source hardware ABI differs')
    t,e,topk,k,n,bm,bn,bk,grid_m,routes=map(int,header[2:12]);counts=tuple(map(int,header[12].split(',')))
    views=unique_by(program.views,'view_id')
    layouts=(('query',((t if header[1]=='w1' else routes),k),'bf16',2),('weight',(e,n,k),'bf16',2),
        ('output',(t,topk,n),'bf16',2),*_control_layouts(header))
    parents={role:views[role] for role,*_ in layouts}
    if len({v.allocation_id for v in parents.values()})!=len(parents):raise ValueError('BF16 naive parents alias')
    for role,shape,dtype,width in layouts:
        view=parents[role];strides=[];size=width
        for extent in reversed(shape):strides.append(size);size*=extent
        if (view.shape,view.strides_bytes,view.offset_bytes,view.dtype,view.element_bytes)!=(shape,tuple(reversed(strides)),0,dtype,width):
            raise ValueError('BF16 naive typed parent layout differs')
    ready={}
    for family in program.tile_programs:
        for access in family.accesses:
            if access.operation is not Operation.READ:continue
            if len(access.input_ready)!=1 or access.access_id in ready and ready[access.access_id]!=access.input_ready[0]:
                raise ValueError('BF16 naive input readiness differs')
            ready[access.access_id]=access.input_ready[0]
    if set(ready)!=set(parents)-{'output'} or not program.tile_programs:raise ValueError('BF16 naive input read missing')
    selected,families=_naive_recipe(scope,parents,ready,program.tile_programs[0].visibility_domain)
    if (families!=program.tile_programs or set(views)!={*parents,*(v.view_id for v in selected)}
            or any(views[v.view_id]!=v for v in selected)):raise ValueError('BF16 naive metadata/mask/READ_READY causality differs')
    events=unique_by(program.events,'event_id')
    for role,ref in ready.items():
        event=events[ref.node_id];parent=parents[role]
        logical=(ref.kind is EventKind.LOGICAL_AVAILABLE and isinstance(event.region,Unmapped)
            and (event.region.logical_shape,event.region.dtype,event.region.element_bytes)==(parent.shape,parent.dtype,parent.element_bytes))
        if (ref.kind not in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE,EventKind.LOGICAL_AVAILABLE)
                or event.allocation_id!=parent.allocation_id or not (logical or event.region==parent.region())):
            raise ValueError('BF16 naive complete parent readiness differs')
    totals={role:0 for role in (*ready,'output')}
    for family in families:
        axes=unique_by(family.axes,'axis_id')
        for access in family.accesses:totals[access.access_id]+=_tile_template_requested_bytes(views,axes,family,access)
    return {'projection':(header[1],t if header[1]=='w1' else routes,k,n),'routing':(counts,counts),
        'read_bytes':routes*4+routes*k*2+sum(c>0 for c in counts)*n*k*2,'write_bytes':routes*n*2,
        'logical_requested_read_bytes':sum(value for role,value in totals.items() if role!='output'),
        'logical_requested_write_bytes':totals['output'],'control_read_bytes':totals['expert_ids']+totals['num_tokens_post_padded'],
        'router_weight_ingress_bytes':totals['router_weight'] if 'router_weight' in totals else 0,
        'router_weight_compulsory_bytes':routes*4 if header[1]=='w2' else 0}
