"""GPU RoPE's in-place Q/K regions and position-selected cache requests."""
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,
    MemoryAccessProgram,MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,
    TensorView,Unmapped,requested_bytes,unique_by,TileAxis,TileAccessTemplate,TileDependency,TileProgram,TileIndexBinding,TileIndexTerm,
)


def has_gpu_rotary_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant in ('rope','rotary')
        and kernel.execution_support == 'supported' and hardware.kernel_memory_model == 'gpu_mshr'
        and kernel.dtype in ('bf16','fp16','fp32')
        and kernel.launch.block[1:] == (1,1) and kernel.launch.grid[1:] == (1,1)
        and {'seq_len','num_query_heads','num_kv_heads','head_dim'} <= set(kernel.shape)
        and kernel.launch.grid_size == kernel.shape['seq_len']
        and (kernel.shape['head_dim'] if kernel.variant == 'rotary' else kernel.shape['rotary_dim']) % 2 == 0)


def has_tensor_only_rotary_source(kernel,hardware):
    """Compatibility adapters explicitly expose Q/K traffic without serving metadata."""
    return (kernel.kind is KernelKind.VECTOR and kernel.variant in ('rope','rotary')
        and kernel.execution_support=='supported'
        and hardware.kernel_memory_model in ('pim_bank','systolic_scratchpad')
        and kernel.dtype in ('bf16','fp16','fp32')
        and {'seq_len','num_query_heads','num_kv_heads','head_dim'}<=set(kernel.shape)
        and (kernel.shape['head_dim'] if kernel.variant=='rotary' else kernel.shape['rotary_dim'])%2==0)


def standalone_tensor_rotary_bindings(kernel, *, call_id,binding_id):
    # Reuse the active Q/K storage recipe, retaining precisely the tensor-only
    # ABI. No position/cache object or availability enters the returned source.
    from dataclasses import replace
    binding=standalone_rotary_bindings(kernel,call_id=call_id,binding_id=binding_id)
    views=tuple(view for view in binding.views if view.view_id in binding.write_views)
    identities={view.allocation_id for view in views}
    return replace(binding,allocations=tuple(item for item in binding.allocations if item.allocation_id in identities),
        views=views,events=tuple(event for event in binding.events if event.allocation_id in identities),
        read_views=binding.write_views,input_ready=binding.input_ready[:len(views)])


def _tensor_rotary_tiles(views,ready,scope,visibility,evidence):
    query=views['query'];tokens=query.shape[0]
    axes=[TileAxis('token',tokens,1)];accesses=[]
    roles=('query','key') if 'key' in views else ('query',)
    for role in roles:
        view=views[role];axis=f'{role}_head';axes.append(TileAxis(axis,view.shape[1],1))
        accesses.extend((
            TileAccessTemplate(f'{role}_read',role,Operation.READ,('token',axis),('token',axis,None),
                (ready[role],),(),evidence),
            TileAccessTemplate(f'{role}_write',role,Operation.WRITE,('token',axis),('token',axis,None),(),
                (TileDependency(f'{role}_read',EventKind.READ_READY,()),),evidence)))
    return TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),visibility,evidence)


def build_tensor_rotary_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_tensor_only_rotary_source(kernel,hardware):raise ValueError('Tensor-only rotary requires its compatibility ABI')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('Tensor-only rotary requires typed in-place Q/K bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown rotary granularity')
    tokens,queries,keys,head,rotary,width=_geometry(kernel);roles=('query','key') if keys else ('query',)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(roles,roles,len(roles)):
        raise ValueError('Tensor-only rotary must retain exactly its active Q/K input/output roles')
    views=unique_by(bindings.views,'view_id')
    if set(views)!=set(roles):raise ValueError('Tensor-only rotary cannot invent position or cache operands')
    for role,heads in (('query',queries),('key',keys)):
        if not heads:continue
        view=views[role]
        if ((view.shape,view.dtype,view.element_bytes,view.strides_bytes[1:])!=
                ((tokens,heads,rotary),kernel.dtype,width,(head*width,width))
                or view.strides_bytes[0]<heads*head*width):
            raise ValueError('Tensor-only rotary Q/K active region or head stride mismatch')
    scope=AccessScope(program_id,('tensor_only_rotary',str(head)),bindings.call_id,(),bindings.rank,bindings.stream,
        'logical-rotary-tensors')
    evidence=('Existing pim_bank/systolic_scratchpad VECTOR rotary compatibility ABI exposes only active Q/K reads and in-place writes.',
        'One complete active head region is independent of channel pairing; no GPU/NPU metadata, native issue or physical residency claim.')
    family=_tensor_rotary_tiles(views,dict(zip(roles,bindings.input_ready,strict=True)),scope,bindings.visibility_domain,evidence)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,True,(
            'Tensor-only compatibility interface: positions/cache are not part of this model, rather than observed zero traffic.',
            'Logical full-head partitions do not describe native PIM banks, systolic blocks or an implemented serving runner.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    tensor_rotary_source_demands(program)
    return program


def tensor_rotary_source_demands(program):
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:
        raise ValueError('Tensor-only rotary requires its unique compressed head source')
    family,=program.tile_programs;header=family.scope.op_ids
    views=unique_by(program.views,'view_id');templates=unique_by(family.accesses,'access_id')
    if (len(header)!=2 or header[0]!='tensor_only_rotary' or family.scope.agent!='logical-rotary-tensors'
            or set(views) not in ({'query'},{'query','key'})):
        raise ValueError('Tensor-only rotary requires its declared head/tensor ABI')
    head=int(header[1]);query=views['query'];tokens,queries,rotary=query.shape;width=query.element_bytes
    keys=views['key'].shape[1] if 'key' in views else 0;roles=('query','key') if keys else ('query',)
    if (head<rotary or rotary%2 or query.dtype not in ('bf16','fp16','fp32')
            or len({view.allocation_id for view in views.values()})!=len(roles)):
        raise ValueError('Tensor-only rotary aliases need an independently declared Q/K mapping')
    ready={}
    for role in roles:
        view=views[role];identity=f'{role}_read'
        if ((view.shape,view.dtype,view.element_bytes,view.strides_bytes[1:])!=
                ((tokens,view.shape[1],rotary),query.dtype,width,(head*width,width))
                or view.strides_bytes[0]<view.shape[1]*head*width or identity not in templates
                or len(templates[identity].input_ready)!=1):
            raise ValueError('Tensor-only rotary must conserve its active in-place head selections')
        ready[role]=templates[identity].input_ready[0]
    expected=_tensor_rotary_tiles(views,ready,family.scope,family.visibility_domain,family.evidence_refs)
    if family!=expected:raise ValueError('Tensor-only rotary output requires its own complete input READ_READY')
    from src.kernel.memory_path_solver import logical_requested_bytes
    read=logical_requested_bytes(program,Operation.READ);write=logical_requested_bytes(program,Operation.WRITE)
    return {'tokens':tokens,'rotary_dim':rotary,'query_heads':queries,'key_heads':keys,'dtype_bytes':width,
        'active_elements':read//width,'tensor_read_bytes':read,'tensor_write_bytes':write,
        'per_token_position_bytes':0,'per_token_cache_row_bytes':0,
        'logical_position_requests_bytes':0,'logical_cache_requests_bytes':0}


def _geometry(kernel):
    shape = kernel.shape
    tokens,queries,keys,head = (shape[name] for name in ('seq_len','num_query_heads','num_kv_heads','head_dim'))
    rotary = head if kernel.variant == 'rotary' else shape['rotary_dim']
    if kernel.variant == 'rope' and not kernel.attributes['key_present']:
        keys = 0
    if rotary % 2 or shape['element_count'] != tokens*(queries+keys)*rotary:
        raise ValueError('GPU RoPE requires complete active Q/K rotation pairs')
    return tokens,queries,keys,head,rotary,shape['dtype_bytes']


def _cache_region(view,binding_id):
    return Unmapped(view.shape,view.dtype,view.element_bytes,f'{binding_id}.position_selected_cos_sin')


def _validate_rotary_aliases(views,keys):
    tensors=('query','key') if keys else ('query',)
    if views['cache_selection'].allocation_id in {views[role].allocation_id for role in (*tensors,'positions')}:
        raise NotImplementedError('RoPE cache alias needs a bound native cache mapping')
    if views['positions'].allocation_id in {views[role].allocation_id for role in tensors}:
        raise NotImplementedError('RoPE position/tensor aliases need a source lifetime schedule')
    if keys and views['query'].allocation_id==views['key'].allocation_id:
        query,key=views['query'],views['key']
        if query.strides_bytes[0]!=key.strides_bytes[0]:
            raise NotImplementedError('RoPE shared Q/K parent requires a common declared row stride')
        stride=query.strides_bytes[0]
        if stride==0:raise ValueError('RoPE shared Q/K parent requires a positive row stride')
        starts=(query.offset_bytes%stride,key.offset_bytes%stride)
        ends=tuple(start+(view.shape[1]-1)*view.strides_bytes[1]+view.shape[2]*view.element_bytes
            for start,view in zip(starts,(query,key),strict=True))
        if max(ends)>stride or max(starts)<min(ends):
            raise NotImplementedError('RoPE Q/K overlapping row envelopes need an independently proven alias mapping')


def standalone_rotary_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    tokens,queries,keys,head,rotary,width = _geometry(kernel)
    layouts = [('query',(tokens,queries,rotary),kernel.dtype,width)]
    if keys:
        layouts.append(('key',(tokens,keys,rotary),kernel.dtype,width))
    tensors = tuple(role for role,*_ in layouts)
    layouts += [('positions',(tokens,),'int64',8),('cache_selection',(tokens,rotary),kernel.dtype,width)]
    allocations,views,events,ready = [],[],[],[]
    for role,shape,dtype,element_width in layouts:
        identity = f'{binding_id}.{role}'
        if role in tensors:
            strides = (shape[1]*head*width,head*width,width)
            size = tokens*shape[1]*head*width
        else:
            strides = (rotary*width,width) if role == 'cache_selection' else (8,)
            size = None if role == 'cache_selection' else tokens*8
        allocation = MemoryAllocation(identity,'logical-local-device','logical-storage',size,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view = TensorView(role,identity,0,shape,strides,dtype,element_width)
        allocations.append(allocation); views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        kind = EventKind.LOGICAL_AVAILABLE if role == 'cache_selection' else EventKind.INITIALIZED
        region = _cache_region(view,binding_id) if role == 'cache_selection' else view.region()
        event = ProgramEvent(f'{identity}.available',kind,identity,region,'rotary-visible',
            (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
        events.append(event); ready.append(EventRef(event.event_id,kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        (*tensors,'positions','cache_selection'),tensors,tuple(ready),0,'offline','rotary-visible')


def build_gpu_rotary_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('GPU RoPE requires typed Q/K/position/cache bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('unknown GPU RoPE logical granularity')
    tokens,queries,keys,head,rotary,width = _geometry(kernel)
    tensors = ('query','key') if keys else ('query',)
    roles = (*tensors,'positions','cache_selection')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (roles,tensors,len(roles)):
        raise ValueError('GPU RoPE input/output roles mismatch')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    if set(views) != set(roles):
        raise ValueError('GPU RoPE tensor/cache views must be explicit')
    for role,heads in (('query',queries),('key',keys)):
        if not heads:
            continue
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes[1:]) != (
                (tokens,heads,rotary),kernel.dtype,width,(head*width,width)) or view.strides_bytes[0] < heads*head*width:
            raise ValueError('GPU RoPE active Q/K region/head stride mismatch')
    position,cache = views['positions'],views['cache_selection']
    if (position.shape,position.dtype,position.element_bytes,position.strides_bytes) != ((tokens,),'int64',8,(8,)):
        raise ValueError('GPU RoPE requires its actual contiguous INT64 positions')
    if (cache.shape,cache.dtype,cache.element_bytes) != ((tokens,rotary),kernel.dtype,width):
        raise ValueError('GPU RoPE selected cache payload mismatch')
    _validate_rotary_aliases(views,keys)
    return _build_rotary_tiles(kernel,bindings,views,program_id=program_id,paired=kernel.variant=='rope')


def rotary_source_demands(program):
    if program.scopes and program.scopes[0].op_ids[:1]==('cann_rotary',):
        from src.kernel.memory_cann_rotary import cann_rotary_source_demands
        return cann_rotary_source_demands(program)
    if (program.accesses or program.scopes or len(program.tile_programs)!=1):
        raise ValueError('GPU rotary requires its unique head or declared-pair tile source')
    family,=program.tile_programs
    if family.scope.op_ids and family.scope.op_ids[0]=='tensor_only_rotary':
        return tensor_rotary_source_demands(program)
    if family.scope.op_ids and family.scope.op_ids[0]=='gpu_rotary_heads':
        return rotary_head_tile_demands(program)
    return rotary_pair_tile_demands(program)


def build_rotary_pair_tiles(kernel,bindings,views, *, program_id):
    if kernel.variant!='rope':raise ValueError('Pair tiles require the declared RoPE layout ABI')
    return _build_rotary_tiles(kernel,bindings,views,program_id=program_id,paired=True)


def _build_rotary_tiles(kernel,bindings,views, *, program_id,paired):
    tokens,queries,keys,head,rotary,width=_geometry(kernel);pairs=rotary//2;threads=kernel.launch.block_size
    neox=kernel.attributes['is_neox'] if paired else None;roles=('query','key') if keys else ('query',)
    evidence=('vLLM0.24 pos_encoding_kernels.cu:apply_rotary_embedding loops i=threadIdx.x; i<heads*(rotary_dim/2); i+=blockDim.x independently for Q and K.',
        ('Each pair reads and updates exactly two channels; neox halves or interleaved channels come from the declared caller ABI.' if paired else
         'Each full head consumes and updates its complete declared channel region; neox/interleaved pairing is not inferred.'),
        'Cache pair reads require that actual thread position READ_READY; positions values/cache capacity/physical selection remain unknown.')
    selected=[]
    if not paired:
        evidence+=('Legacy full-head rotary does not declare channel pairing. A full head is a self-contained rotation block for either neox or interleaved pairing; its final publication requires exactly its own complete input and cache pairs.',)
    for role in roles:
        if paired:
            view=views[role];ts,hs,ws=view.strides_bytes
            shape=(tokens,view.shape[1],2,pairs) if neox else (tokens,view.shape[1],pairs,2)
            strides=(ts,hs,pairs*ws,ws) if neox else (ts,hs,2*ws,ws)
            selected.append(TensorView(f'{role}.pairs',view.allocation_id,view.offset_bytes,shape,strides,view.dtype,view.element_bytes))
    cache=views['cache_selection']
    cache_pairs=TensorView('cache.pairs',cache.allocation_id,0,(tokens,2,pairs),(rotary*width,pairs*width,width),cache.dtype,width)
    selected.append(cache_pairs)
    ready=dict(zip(bindings.read_views,bindings.input_ready,strict=True))
    available=ProgramEvent(f'{program_id}.cache_pairs.available',EventKind.LOGICAL_AVAILABLE,cache.allocation_id,
        Unmapped(cache_pairs.shape,cache_pairs.dtype,width,f'{program_id}.cache_pairs'),bindings.visibility_domain,(ready['cache_selection'],))
    accesses=[TileAccessTemplate('positions','positions',Operation.READ,('token','thread'),('token',),
        (ready['positions'],),(),evidence)]
    axes=[TileAxis('token',tokens,1),TileAxis('thread',threads,1),TileAxis('pair',pairs,1)]
    for role in roles:
        name=f'{role}_head';heads=views[role].shape[1];axes.append(TileAxis(name,heads,1))
        loop=('token',name,'pair') if paired else ('token',name)
        mapping=(('token',name,None,'pair') if neox else ('token',name,'pair',None)) if paired else ('token',name,None)
        operand=f'{role}.pairs' if paired else role
        binding=TileIndexBinding('thread',(TileIndexTerm(name,pairs),TileIndexTerm('pair',1)),modulus=threads)
        accesses.append(TileAccessTemplate(f'{role}_read',operand,Operation.READ,loop,mapping,(ready[role],),(),evidence))
        accesses.append(TileAccessTemplate(f'{role}_cache','cache.pairs',Operation.READ,('token',name,'pair'),('token',None,'pair'),
            (EventRef(available.event_id,available.kind),),(TileDependency('positions',EventKind.READ_READY,(),(binding,)),),evidence,unmapped=True))
        accesses.append(TileAccessTemplate(f'{role}_write',operand,Operation.WRITE,loop,mapping,(),
            (TileDependency(f'{role}_read',EventKind.READ_READY,()),TileDependency(f'{role}_cache',EventKind.READ_READY,() if paired else ('pair',))),evidence))
    header=('gpu_rotary_pairs','neox' if neox else 'interleaved',str(threads)) if paired else ('gpu_rotary_heads',str(threads))
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,
        'logical-rotary-pairs' if paired else 'logical-rotary-heads')
    family=TileProgram(program_id,scope,tuple(axes),tuple(accesses),bindings.visibility_domain,evidence)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            ('Pair coordinates preserve exact active in-place channels, untouched suffixes, source thread position multiplicity and per-head cache requests.' if paired else
             'Independent complete-head blocks preserve full in-place channels, actual per-head cache pair reads and their precise position-thread dependencies.'),
            'Positions and native cache coordinates/capacity remain unknown; logical selected pairs are not issued transactions or observed cache rows.',
            'Full-head blocks do not resolve unknown neox/interleaved channel pairing or per-pair early publication; fused/native RoPE ABIs remain separate.')),
        bindings.allocations,(*bindings.views,*selected),(),(*bindings.events,available),(),(family,))
    rotary_source_demands(program)
    return program


def rotary_head_tile_demands(program):
    """A full head has known memory work without choosing its channel pairing."""
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:
        raise ValueError('GPU rotary requires its unique full-head tile source')
    family,=program.tile_programs;header=family.scope.op_ids
    if len(header)!=2 or header[0]!='gpu_rotary_heads':raise ValueError('GPU rotary requires its full-head/thread ABI')
    views=unique_by(program.views,'view_id');accesses=unique_by(family.accesses,'access_id')
    query=views['query'];tokens,queries,rotary=query.shape;keys=views['key'].shape[1] if 'key' in views else 0
    threads=int(header[1]);pairs=rotary//2;width=query.element_bytes;tensors=('query','key') if keys else ('query',)
    if (rotary%2 or query.dtype not in ('bf16','fp16','fp32') or set(views)!={*tensors,'positions','cache_selection','cache.pairs'}
            or set(accesses)!={'positions',*(f'{role}_{part}' for role in tensors for part in ('read','cache','write'))}):
        raise ValueError('GPU rotary full-head roles or geometry mismatch')
    expected=(('token',tokens,1),('thread',threads,1),('pair',pairs,1),('query_head',queries,1))
    if keys:expected+=(('key_head',keys,1),)
    if tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes)!=expected or family.causal_prefixes:
        raise ValueError('GPU rotary full-head loop geometry mismatch')
    for access in family.accesses:
        if access.view_axis_divisors or access.view_axis_padding or access.constant_value is not None or access.indirect_target is not None:
            raise ValueError('GPU rotary must retain its declared complete-head selections')
    position=accesses['positions'];pos=views[position.view_id];cache_parent=views['cache_selection'];cache_view=views['cache.pairs']
    if ((pos.shape,pos.strides_bytes,pos.dtype,pos.element_bytes)!=((tokens,),(8,),'int64',8)
            or position.operation is not Operation.READ or position.iteration_axes!=('token','thread')
            or position.view_axes!=('token',) or position.unmapped):
        raise ValueError('GPU rotary source positions mismatch')
    if ((cache_parent.shape,cache_parent.dtype,cache_parent.element_bytes)!=( (tokens,rotary),query.dtype,width)
            or (cache_view.shape,cache_view.dtype,cache_view.element_bytes,cache_view.allocation_id)
                !=((tokens,2,pairs),query.dtype,width,cache_parent.allocation_id)):
        raise ValueError('GPU rotary full-head cache selection mismatch')
    _validate_rotary_aliases(views,keys)
    for role in tensors:
        view=views[role];heads=view.shape[1];name=f'{role}_head';read,cache,write=(accesses[f'{role}_{part}'] for part in ('read','cache','write'))
        if ((view.shape,view.dtype,view.element_bytes,view.strides_bytes[1:])!=((tokens,heads,rotary),query.dtype,width,(rotary*width,width))
                or view.strides_bytes[0]<heads*rotary*width or read.operation is not Operation.READ or write.operation is not Operation.WRITE
                or any(access.view_id!=role or access.iteration_axes!=('token',name) or access.view_axes!=('token',name,None)
                    or access.unmapped for access in (read,write))
                or cache.operation is not Operation.READ or cache.view_id!='cache.pairs' or not cache.unmapped
                or cache.iteration_axes!=('token',name,'pair') or cache.view_axes!=('token',None,'pair')):
            raise ValueError('GPU rotary must conserve its actual full-head in-place regions')
        binding=TileIndexBinding('thread',(TileIndexTerm(name,pairs),TileIndexTerm('pair',1)),modulus=threads)
        if (cache.dependencies!=(TileDependency('positions',EventKind.READ_READY,(),(binding,)),)
                or write.dependencies!=(TileDependency(f'{role}_read',EventKind.READ_READY,()),
                    TileDependency(f'{role}_cache',EventKind.READ_READY,('pair',)))):
            raise ValueError('GPU RoPE lacks actual Q/K/position/cache READ_READY causality; full-head publication requires its own actual inputs/cache pairs')
    return _rotary_tile_work(program,tokens,rotary,queries,keys,width,tensors)


def rotary_pair_tile_demands(program):
    program.__post_init__()
    if len(program.tile_programs) != 1:
        raise ValueError('GPU RoPE requires one actual pair loop')
    family,=program.tile_programs;views=unique_by(program.views,'view_id');accesses=unique_by(family.accesses,'access_id')
    header=family.scope.op_ids
    if len(header) != 3 or header[:2] not in (('gpu_rotary_pairs','neox'),('gpu_rotary_pairs','interleaved')):
        raise ValueError('GPU RoPE requires its declared pair/thread ABI')
    threads=int(header[2]);tokens,queries,rotary=views['query'].shape;pairs=rotary//2
    keys=views['key'].shape[1] if 'key' in views else 0;width=views['query'].element_bytes
    _validate_rotary_aliases(views,keys)
    tensors=('query','key') if keys else ('query',)
    if set(accesses) != {'positions',*(f'{role}_{part}' for role in tensors for part in ('read','cache','write'))}:
        raise ValueError('GPU RoPE pair source roles incomplete')
    expected=(('token',tokens,1),('thread',threads,1),('pair',pairs,1),('query_head',queries,1))
    if keys:expected+= (('key_head',keys,1),)
    if tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != expected or family.causal_prefixes:
        raise ValueError('GPU RoPE source loop geometry mismatch')
    position=accesses['positions'];pos_view=views[position.view_id]
    if ((pos_view.shape,pos_view.strides_bytes,pos_view.dtype,pos_view.element_bytes) != ((tokens,),(8,),'int64',8)
            or position.operation is not Operation.READ or position.iteration_axes != ('token','thread') or position.view_axes != ('token',)):
        raise ValueError('GPU RoPE source positions mismatch')
    for role in tensors:
        parent=views[role];read,cache,write=(accesses[f'{role}_{part}'] for part in ('read','cache','write'))
        active=views[read.view_id];cache_view=views[cache.view_id];name=f'{role}_head';loop=('token',name,'pair')
        ts,hs,ws=parent.strides_bytes
        neox=header[1] == 'neox';shape=(tokens,parent.shape[1],2,pairs) if neox else (tokens,parent.shape[1],pairs,2)
        strides=(ts,hs,pairs*ws,ws) if neox else (ts,hs,2*ws,ws)
        mapping=('token',name,None,'pair') if neox else ('token',name,'pair',None)
        if ((active.allocation_id,active.offset_bytes,active.shape,active.strides_bytes,active.dtype,active.element_bytes)
                != (parent.allocation_id,parent.offset_bytes,shape,strides,parent.dtype,width)
                or read.operation is not Operation.READ or write.operation is not Operation.WRITE or read.view_id != write.view_id
                or any(access.iteration_axes != loop or access.view_axes != mapping or access.unmapped or access.view_axis_divisors for access in (read,write))
                or cache.operation is not Operation.READ or cache.iteration_axes != loop or cache.view_axes != ('token',None,'pair')
                or not cache.unmapped or (cache_view.shape,cache_view.dtype,cache_view.element_bytes) != ((tokens,2,pairs),parent.dtype,width)
                or cache_view.allocation_id != views['cache_selection'].allocation_id):
            raise ValueError('GPU RoPE must conserve its actual in-place pairs and cache selection')
        binding=TileIndexBinding('thread',(TileIndexTerm(name,pairs),TileIndexTerm('pair',1)),modulus=threads)
        if (cache.dependencies != (TileDependency('positions',EventKind.READ_READY,(),(binding,)),)
                or write.dependencies != (TileDependency(f'{role}_read',EventKind.READ_READY,()),TileDependency(f'{role}_cache',EventKind.READ_READY,()))):
            raise ValueError('GPU RoPE lacks actual Q/K/position/cache READ_READY causality')
    return _rotary_tile_work(program,tokens,rotary,queries,keys,width,tuple(f'{role}.pairs' for role in tensors))


def _rotary_tile_work(program,tokens,rotary,queries,keys,width,tensor_views):
    from src.kernel.memory_path_solver import logical_requested_bytes,logical_view_requested_bytes
    views=unique_by(program.views,'view_id')
    read_bytes=sum(logical_view_requested_bytes(program,role,Operation.READ) for role in tensor_views)
    return {'tokens':tokens,'rotary_dim':rotary,'query_heads':queries,'key_heads':keys,'dtype_bytes':width,
        'active_elements':read_bytes//width,'tensor_read_bytes':read_bytes,
        'tensor_write_bytes':logical_requested_bytes(program,Operation.WRITE),'per_token_position_bytes':requested_bytes(views['positions'].region()),
        'per_token_cache_row_bytes':requested_bytes(views['cache_selection'].region()),
        'logical_position_requests_bytes':logical_view_requested_bytes(program,'positions',Operation.READ),
        'logical_cache_requests_bytes':logical_view_requested_bytes(program,'cache.pairs',Operation.READ)}
