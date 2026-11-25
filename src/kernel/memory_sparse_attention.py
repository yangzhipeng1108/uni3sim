"""Sparse attention selected inputs with honest unresolved cache coordinates."""
from dataclasses import dataclass,replace
from itertools import accumulate
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity, MemoryAccess,
    MemoryAccessProgram, MemoryAllocation, Operation, Ordering, ProgramEvent, Semantics,
    TensorView, Typed, Unmapped, unique_by,LogicalSelectionView,
    TileAxis,TileAccessTemplate,TileDependency,TileProgram,TileCausalPrefix,
    LogicalCoordinateBinding,LogicalCoordinateTerm,ConstantValue,
)


@dataclass(frozen=True)
class SparseAttentionBindings(Typed):
    binding_id: str
    call_id: str
    allocations: tuple[MemoryAllocation, ...]
    views: tuple[TensorView | LogicalSelectionView, ...]
    events: tuple[ProgramEvent, ...]
    query_view: str
    output_view: str
    indices: str | None
    cache: str
    query_ready: EventRef
    input_ready: tuple[EventRef, ...]
    rank: int
    stream: str
    visibility_domain: str
    resources: tuple[tuple[str, str], ...] = ()
    availability: tuple[tuple[str, EventRef], ...] = ()
    query_parts: tuple[str, ...] = ()
    query_row_ranges: tuple[tuple[int, int], ...] = ()
    kv_parts: tuple[str, ...] = ()


def has_sparse_attention_source(kernel):
    return (kernel.kind is KernelKind.SPARSE_ATTN and kernel.execution_support == 'supported'
            and kernel.variant in ('dsa_sparse_prefill','dsa_sparse_decode','dsv4_sparse_prefill','dsv4_sparse_decode'))


def sparse_attention_geometry(kernel):
    query_lengths,cache_lengths = kernel.attributes['q_lens'],kernel.attributes['kv_lens']
    capacity = kernel.attributes['topk_capacity']
    pairs = 0
    for q,kv in zip(query_lengths,cache_lengths,strict=True):
        start = kv-q+1;ramp = min(q,max(0,capacity-start+1))
        pairs += ramp*(2*start+ramp-1)//2+(q-ramp)*capacity
    return (sum(query_lengths),kernel.shape['num_query_heads'],kernel.shape['qk_head_dim'],
            kernel.shape['v_head_dim'],pairs)


def sparse_attention_input_regions(kernel, *, selection_id):
    _,heads,qk_dim,value_dim,pairs = sparse_attention_geometry(kernel)
    coordinate = f"{selection_id}.q{','.join(map(str,kernel.attributes['q_lens']))}.kv{','.join(map(str,kernel.attributes['kv_lens']))}.k{kernel.attributes['topk_capacity']}"
    # Native prefill receives expanded K/V roles from kv_b projection;
    # decode uses latent MQA cache roles shared by the query heads.
    repeat = heads if kernel.variant == 'dsa_sparse_decode' else 1
    key_shape = (pairs,qk_dim) if repeat == heads and kernel.variant.endswith('decode') else (pairs,heads,qk_dim)
    value_shape = (pairs,value_dim) if kernel.variant.endswith('decode') else (pairs,heads,value_dim)
    return ((Unmapped((pairs,),'int32',4,f'{coordinate}.indices'),heads),
            (Unmapped(key_shape,kernel.dtype,kernel.shape['dtype_bytes'],f'{coordinate}.key'),repeat),
            (Unmapped(value_shape,kernel.dtype,kernel.shape['dtype_bytes'],f'{coordinate}.value'),repeat))


def standalone_sparse_attention_bindings(kernel, *, call_id, binding_id):
    if kernel.variant.startswith("dsv4_"):
        return standalone_v4_sparse_bindings(kernel,call_id=call_id,binding_id=binding_id)
    tokens,heads,qk_dim,value_dim,_ = sparse_attention_geometry(kernel)
    width = kernel.shape['dtype_bytes']
    allocations,views,events = [],[],[]
    for name,dimension in (('query',qk_dim),('output',value_dim)):
        identity = f'{binding_id}.{name}'
        shape = (tokens,heads,dimension)
        view = TensorView(name,identity,0,shape,(heads*dimension*width,dimension*width,width),kernel.dtype,width)
        allocation = MemoryAllocation(identity,'logical-local-device','logical-storage',tokens*heads*dimension*width,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation)
        views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if name == 'query':
            initialized = f'{identity}.initialized'
            events.append(ProgramEvent(initialized,EventKind.INITIALIZED,identity,view.region(),'sparse-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),)))
            query_ready = EventRef(initialized,EventKind.INITIALIZED)
    for name in ('indices','cache'):
        identity = f'{binding_id}.{name}'
        allocations.append(MemoryAllocation(identity,'logical-local-device','logical-sparse-storage',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None))
        events.append(ProgramEvent(f'{identity}.allocated',EventKind.ALLOCATED,identity,None,None,()))
    indices,cache = allocations[-2].allocation_id,allocations[-1].allocation_id
    ready = []
    for role,allocation,(region,_) in zip(('indices','key','value'),(indices,cache,cache),
            sparse_attention_input_regions(kernel,selection_id=binding_id),strict=True):
        event_id = f'{binding_id}.{role}.available'
        allocated = next(item.allocated_event for item in allocations if item.allocation_id == allocation)
        events.append(ProgramEvent(event_id,EventKind.LOGICAL_AVAILABLE,allocation,region,'sparse-visible',
            (EventRef(allocated,EventKind.ALLOCATED),)))
        ready.append(EventRef(event_id,EventKind.LOGICAL_AVAILABLE))
    return SparseAttentionBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),'query','output',
        indices,cache,query_ready,tuple(ready),0,'offline','sparse-visible')


def build_sparse_attention_source(kernel, hardware, bindings, granularity, *, program_id):
    if kernel.variant.startswith("dsv4_"):
        return build_v4_sparse_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if not isinstance(bindings,SparseAttentionBindings):
        raise TypeError('sparse attention requires typed query/index/cache/output bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('unknown sparse attention logical granularity')
    tokens,heads,qk_dim,value_dim,_ = sparse_attention_geometry(kernel)
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    query,output = views[bindings.query_view],views[bindings.output_view]
    rope=None;cat_parts=None
    if bindings.query_parts:
        if (len(bindings.query_parts)!=2 or len(set(bindings.query_parts))!=2
                or not bindings.query_row_ranges or kernel.variant=='dsa_sparse_decode' and qk_dim<=value_dim):
            raise ValueError('Sparse decode tuple requires its actual NoPE/RoPE parts and request row ranges')
        parts=tuple(views[name] for name in bindings.query_parts)
        if kernel.variant=='dsa_sparse_decode' and bindings.query_parts[0]==bindings.query_view:
            rope=parts[1]
        elif kernel.variant=='dsa_sparse_prefill' and bindings.query_view not in bindings.query_parts:
            cat_parts=parts
        else:raise ValueError('Sparse query parts differ from actual decode tuple or prefill materialization')
        nope_width=value_dim if rope is not None else parts[0].shape[2]
        if (not 0<nope_width<qk_dim or parts[0].shape[1:]!=(heads,nope_width)
                or parts[1].shape[1:]!=(heads,qk_dim-nope_width)
                or any((v.dtype,v.element_bytes)!=(kernel.dtype,kernel.shape['dtype_bytes']) for v in parts)):
            raise ValueError('Sparse decode NoPE/RoPE storage differs from its actual field geometry')
    elif bindings.query_row_ranges:
        raise ValueError('Packed sparse query cannot invent a tuple row mapping')
    for view,dimension in (((query,qk_dim),) if rope is None and cat_parts is None else ()) + ((output,value_dim),):
        if (view.shape,view.dtype,view.element_bytes) != ((tokens,heads,dimension),kernel.dtype,kernel.shape['dtype_bytes']):
            raise ValueError('sparse attention query/output storage mismatch')
    if len(bindings.input_ready) != 3 or len({query.allocation_id,output.allocation_id,bindings.indices,bindings.cache}) != 4:
        raise ValueError('sparse attention operand identity/readiness mismatch')
    if rope is not None and rope.allocation_id in (query.allocation_id,output.allocation_id,bindings.indices,bindings.cache):
        raise ValueError('Sparse decode tuple requires its two actual independent query parents')
    if rope is not None and len({allocations[identity].device_id for identity in (
            query.allocation_id,rope.allocation_id,output.allocation_id,bindings.indices,bindings.cache)})!=1:
        raise ValueError('Sparse decode tuple operands must share their actual device')
    if cat_parts is not None:
        width=kernel.shape['dtype_bytes']
        if (not isinstance(query,TensorView) or query.shape[1:]!=(heads,qk_dim)
                or query.strides_bytes!=(heads*qk_dim*width,qk_dim*width,width)
                or any(p.shape[0]!=query.shape[0] for p in cat_parts)
                or len({query.allocation_id,output.allocation_id,bindings.indices,bindings.cache,
                    *(p.allocation_id for p in cat_parts)})!=6
                or len({allocations[v.allocation_id].device_id for v in (query,output,*cat_parts)})!=1):
            raise ValueError('Prefill cat requires full-row independent inputs and its actual new packed destination')
    kv_parts=tuple(views[name] for name in bindings.kv_parts) if bindings.kv_parts else None
    if kv_parts is not None:
        if cat_parts is None or len(kv_parts)!=4 or kernel.variant!='dsa_sparse_prefill':
            raise ValueError('Prefill KV materialization requires its actual query/copy source ABI')
        _validate_prefill_kv_parts(kv_parts,allocations,(sum(kernel.attributes['kv_lens']),heads,qk_dim,value_dim,kernel.shape['dtype_bytes'],kernel.dtype))
        if kv_parts[0].allocation_id!=bindings.cache:raise ValueError('Prefill K storage differs from its actual binding')
    header = ('dsa_sparse_tiles',kernel.variant,str(kernel.attributes['topk_capacity']),
        ','.join(map(str,kernel.attributes['q_lens'])),','.join(map(str,kernel.attributes['kv_lens'])),
        bindings.query_view,bindings.output_view)
    if rope is not None:
        header=('dsa_sparse_tuple_tiles',*header[1:],rope.view_id,
            ','.join(f'{start}:{count}' for start,count in bindings.query_row_ranges))
    elif cat_parts is not None:
        header=('dsa_sparse_prefill_cat_tiles',*header[1:],*(v.view_id for v in cat_parts),
            ','.join(f'{start}:{count}' for start,count in bindings.query_row_ranges))
        if kv_parts is not None:header=('dsa_sparse_prefill_cat_kv_tiles',*header[1:],*bindings.kv_parts)
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-selected-sparse-queries')
    roots = tuple(LogicalSelectionView(f'{program_id}.{role}.root',allocation,region.logical_shape,
        region.dtype,region.element_bytes,region.selection_id,
        (kv_parts[0].view_id if role=='key' else kv_parts[3].view_id if role=='value' else None) if kv_parts is not None else None)
        for role,allocation,(region,_) in zip(
            ('indices','key','value'),(bindings.indices,bindings.cache,bindings.cache if kv_parts is None else kv_parts[3].allocation_id),
            sparse_attention_input_regions(kernel,selection_id=bindings.binding_id),strict=True))
    ready=(bindings.query_ready,*bindings.input_ready)
    if rope is not None:
        availability=dict(bindings.availability)
        if (len(availability)!=len(bindings.availability) or set(availability)!={rope.view_id}
                or bindings.resources):
            raise ValueError('Sparse decode tuple requires its actual RoPE availability')
        ready=(*ready,availability[rope.view_id])
    elif cat_parts is not None:
        availability=dict(bindings.availability)
        wanted={cat_parts[1].view_id,*((kv_parts[2].view_id,) if kv_parts is not None else ())}
        if (len(availability)!=len(bindings.availability) or set(availability)!=wanted
                or bindings.resources):raise ValueError('Prefill cat requires its actual two input availabilities')
        ready=(*ready,availability[cat_parts[1].view_id])
        if kv_parts is not None:ready=(*ready,availability[kv_parts[2].view_id])
        if any(e.allocation_id==query.allocation_id and e.kind is not EventKind.ALLOCATED for e in bindings.events):
            raise ValueError('Prefill cat output cannot be initialized before its actual writes')
        if kv_parts is not None and any(e.allocation_id==bindings.cache and e.kind is not EventKind.ALLOCATED for e in bindings.events):
            raise ValueError('Prefill K concat cannot be initialized before its actual field writes')
    selected,families = _dsa_tile_recipe(header,query,output,roots,ready,scope,bindings.visibility_domain,
        rope=rope,cat_parts=cat_parts,kv_parts=kv_parts)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Query/head and selected-candidate logical tiles retain each actual index READ_READY before K/V; output only requires its own query/head reads.',
            'Selected index values, page/block mappings, cache capacity, aliases and distinct physical footprint remain unknown.',
            ('Prefill K is produced by actual field copies; V retains its packed projection parent.' if kv_parts is not None else
                'QK/value roles share their actual parent; prefill expanded heads and decode latent head requests remain distinct.'),
            'Metadata/LSE/split scratch and prefill helper side effects remain partial; no issued/cache-hit/physical address claim.')),
        bindings.allocations,(*bindings.views,*roots,*selected),(scope,),bindings.events,(),families)
    dsa_sparse_tile_demands(program)
    return program


def _validate_prefill_kv_parts(parts,allocations,geometry):
    """Actual K concat and the V slice of the real KV-b projection."""
    if any(not isinstance(p,TensorView) or len(p.shape)!=2 for p in parts):
        raise ValueError('Prefill KV parts require their actual two-dimensional typed parents')
    key,nope,rope,value=parts;rows,heads,qk,v,width,dtype=geometry;r=rope.shape[1];n=qk-r
    if (not 0<r<qk
            or (key.shape,nope.shape,rope.shape,value.shape)!=((rows*heads,qk),(rows*heads,n),(rows,r),(rows*heads,v))
            or any((p.dtype,p.element_bytes)!=(dtype,width) for p in parts)
            or key.strides_bytes!=(qk*width,width) or key.offset_bytes!=0
            or nope.strides_bytes!=value.strides_bytes or nope.strides_bytes!=((n+v)*width,width)
            or value.offset_bytes!=nope.offset_bytes+n*width or nope.allocation_id!=value.allocation_id
            or rope.strides_bytes[1]!=width or rope.strides_bytes[0]<r*width
            or len({key.allocation_id,nope.allocation_id,rope.allocation_id})!=3
            or len({allocations[p.allocation_id].device_id for p in parts})!=1):
        raise ValueError('Prefill KV parts differ from actual packed projection, broadcast RoPE or separate K storage')


def _dsa_tile_recipe(header,query,output,roots,ready,scope,visibility,*,rope=None,cat_parts=None,kv_parts=None):
    tuple_query=rope is not None;cat=cat_parts is not None
    if ((not tuple_query and not cat and (len(header)!=7 or header[0]!='dsa_sparse_tiles'))
            or (tuple_query and (len(header)!=9 or header[0]!='dsa_sparse_tuple_tiles' or header[1]!='dsa_sparse_decode'))
            or (cat and (len(header)!=(14 if kv_parts is not None else 10)
                or header[0]!=('dsa_sparse_prefill_cat_kv_tiles' if kv_parts is not None else 'dsa_sparse_prefill_cat_tiles')
                or header[1]!='dsa_sparse_prefill'))
            or header[1] not in ('dsa_sparse_prefill','dsa_sparse_decode')):
        raise ValueError('DSA query tile ABI missing')
    capacity = int(header[2]);qlens,kvlens = (tuple(int(text) for text in field.split(',')) for field in header[3:5])
    tokens=sum(qlens);heads=query.shape[1]
    ranges=(tuple(tuple(int(n) for n in pair.split(':')) for pair in header[9 if cat else 8].split(',')) if tuple_query or cat
        else tuple(zip((0,*accumulate(qlens[:-1])),qlens,strict=True)))
    if (len(ranges)!=len(qlens) or any(len(pair)!=2 for pair in ranges)
            or any(start<0 or count!=q for (start,count),q in zip(ranges,qlens,strict=True))
            or any(start+count>query.shape[0] or tuple_query and start+count>rope.shape[0] for start,count in ranges)
            or any(a+n>b for (a,n),(b,_) in zip(sorted(ranges),sorted(ranges)[1:]))):
        raise ValueError('Sparse query ranges must preserve distinct actual request rows')
    if (capacity <= 0 or len(qlens) != len(kvlens) or (not tuple_query and not cat and query.shape[0]!=tokens)
            or any(not 0 < q <= kv for q,kv in zip(qlens,kvlens,strict=True))
            or (query.view_id,output.view_id) != header[5:7]
            or output.shape[:2] != (tokens,heads) or (query.dtype,query.element_bytes) != (output.dtype,output.element_bytes)):
        raise ValueError('DSA query/output/request geometry mismatch')
    evidence = ('Sparse MLA attends min(topk_capacity,KV-Q+query+1) selected rows for each end-aligned query.',
        'Independent query/head work reads actual indices before its selected K/V; native selected values and cache addresses remain unknown.',
        ('Prefill K and V retain their independent concat and projection parents and existing selected request multiplicities.' if kv_parts is not None else
            'Prefill expanded K/V heads and decode shared latent cache use the same parent and their existing logical request multiplicities.'))
    def row_view(parent,identity,first,count):
        if isinstance(parent,TensorView):
            return TensorView(identity,parent.allocation_id,parent.offset_bytes+first*parent.strides_bytes[0],
                (count,*parent.shape[1:]),parent.strides_bytes,parent.dtype,parent.element_bytes)
        if not isinstance(parent,LogicalSelectionView):raise TypeError('Sparse query/output requires a typed actual view')
        return LogicalSelectionView(identity,parent.allocation_id,(count,*parent.shape[1:]),parent.dtype,parent.element_bytes,
            f'{parent.selection_id}.rows{first}:{count}',parent.view_id,
            (LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),offset=first),
                LogicalCoordinateBinding(1,(LogicalCoordinateTerm(1,1),)),LogicalCoordinateBinding(2,(LogicalCoordinateTerm(2,1),))))
    def materialize(identity,first,count):
        children=[];accesses=[];offset=0
        for role,parent,available in zip(('nope','rope'),cat_parts,(ready[0],ready[4]),strict=True):
            source=row_view(parent,f'{identity}.cat.{role}.input',first,count)
            target=TensorView(f'{identity}.cat.{role}.output',query.allocation_id,
                query.offset_bytes+first*query.strides_bytes[0]+offset*query.element_bytes,
                source.shape,query.strides_bytes,query.dtype,query.element_bytes)
            children.extend((source,target));offset+=parent.shape[2]
            evidence_cat=('vLLM 0.30 DeepseekV32Attention: torch.cat((q_nope,mqa_q),dim=-1) materializes every input row.',
                'Reads preserve producer layouts; writes share one new packed query allocation. Helper traffic remains explicitly unpriced.')
            accesses.extend((TileAccessTemplate(f'cat.{role}.read',source.view_id,Operation.READ,('query','head'),('query','head',None),
                (available,),(),evidence_cat,unmapped=isinstance(source,LogicalSelectionView)),
                TileAccessTemplate(f'cat.{role}.write',target.view_id,Operation.WRITE,('query','head'),('query','head',None),(),
                    (TileDependency(f'cat.{role}.read',EventKind.READ_READY,()),),evidence_cat)))
        return children,accesses
    def materialize_key(identity,first,keys):
        parent,nope,rope,value=kv_parts;rows=keys*heads;offset=first*heads
        full=replace(parent,view_id=f'{identity}.key.parent',offset_bytes=parent.offset_bytes+offset*parent.strides_bytes[0],shape=(rows,parent.shape[1]))
        source=replace(nope,view_id=f'{identity}.key_cat.nope.input',offset_bytes=nope.offset_bytes+offset*nope.strides_bytes[0],shape=(rows,nope.shape[1]))
        rotated=LogicalSelectionView(f'{identity}.key_cat.rope.input',rope.allocation_id,(rows,rope.shape[1]),
            rope.dtype,rope.element_bytes,f'{rope.view_id}.broadcast_request{first}:{keys}',rope.view_id,
            (LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),offset=offset,divisor=heads),
                LogicalCoordinateBinding(1,(LogicalCoordinateTerm(1,1),))))
        left=replace(full,view_id=f'{identity}.key_cat.nope.output',shape=(rows,nope.shape[1]))
        right=replace(full,view_id=f'{identity}.key_cat.rope.output',offset_bytes=full.offset_bytes+nope.shape[1]*full.element_bytes,
            shape=(rows,rope.shape[1]))
        evidence_kv=('vLLM 0.30 SparseMLACommonImpl._project_kv splits one packed KV-b result into NoPE and V.',
            'MLACommonBaseImpl._concat_k_nope_k_pe writes NoPE and broadcast RoPE into separate fields of a new K tensor.',
            'Field-copy requests are explicit and unpriced; selected indices and native addresses remain unresolved.')
        accesses=[]
        for role,src,dst,available in zip(('nope','rope'),(source,rotated),(left,right),(ready[2],ready[5]),strict=True):
            accesses.extend((TileAccessTemplate(f'key_cat.{role}.read',src.view_id,Operation.READ,('kvrow',),('kvrow',None),
                (available,),(),evidence_kv,unmapped=isinstance(src,LogicalSelectionView)),
                TileAccessTemplate(f'key_cat.{role}.write',dst.view_id,Operation.WRITE,('kvrow',),('kvrow',None),(),
                    (TileDependency(f'key_cat.{role}.read',EventKind.READ_READY,()),),evidence_kv)))
        return (full,source,rotated,left,right),accesses
    selected,families = [],[];offset = 0;kv_offset=0
    for request,(queries,keys) in enumerate(zip(qlens,kvlens,strict=True)):
        first = keys-queries+1;ramp = min(queries,max(0,capacity-first+1))
        groups=(((0,queries,min(keys,capacity),True),) if kv_parts is not None else
            ((0,ramp,first+ramp-1,True),(ramp,queries-ramp,capacity,False)))
        for start,count,maximum,causal in groups:
            if not count:continue
            identity = f'{scope.scope_id}.r{request}.q{start}';children = []
            for role,parent in (('query',query),('output',output)):
                row=ranges[request][0] if role=='query' else offset
                view=row_view(parent,f'{identity}.{role}',row+start,count)
                selected.append(view);children.append(view)
            q,out = children
            iterations = ('query','head','key')
            if cat:
                children,accesses=materialize(identity,ranges[request][0]+start,count);selected.extend(children)
                for role,part in zip(('query','query_rope'),(children[1],children[3]),strict=True):
                    dependency='cat.nope.write' if role=='query' else 'cat.rope.write'
                    accesses.append(TileAccessTemplate(role,part.view_id,Operation.READ,('query','head'),('query','head',None),(),
                        (TileDependency(dependency,EventKind.WRITE_VISIBLE,()),),evidence))
            else:
                accesses = [TileAccessTemplate('query',q.view_id,Operation.READ,('query','head'),('query','head',None),(ready[0],),(),evidence,
                    unmapped=isinstance(q,LogicalSelectionView))]
            if tuple_query:
                part=row_view(rope,f'{identity}.query_rope',ranges[request][0]+start,count)
                selected.append(part)
                accesses.append(TileAccessTemplate('query_rope',part.view_id,Operation.READ,('query','head'),
                    ('query','head',None),(ready[4],),(),evidence,unmapped=isinstance(part,LogicalSelectionView)))
            key_parent=None
            if kv_parts is not None:
                children,key_accesses=materialize_key(identity,kv_offset,keys);selected.extend(children)
                key_parent=children[0];accesses.extend(key_accesses)
            for index,(role,root) in enumerate(zip(('indices','key','value'),roots,strict=True)):
                view = LogicalSelectionView(f'{identity}.{role}',root.allocation_id,(count,maximum,*root.shape[1:]),
                    root.dtype,root.element_bytes,f'{root.selection_id}.request{request}.q{start}',
                    key_parent.view_id if role=='key' and key_parent is not None else root.view_id)
                selected.append(view)
                mapping = ('query','key') if role == 'indices' else (('query','key',None) if header[1].endswith('decode') else ('query','key','head',None))
                deps = () if role == 'indices' else (TileDependency('indices',EventKind.READ_READY,()),)
                if role=='key' and key_parent is not None:
                    deps=(*deps,*(TileDependency(f'key_cat.{part}.write',EventKind.WRITE_VISIBLE,('kvrow',)) for part in ('nope','rope')))
                accesses.append(TileAccessTemplate(role,view.view_id,Operation.READ,iterations,mapping,
                    () if role=='key' and key_parent is not None else (ready[index+1],),deps,evidence,unmapped=True))
            accesses.append(TileAccessTemplate('output',out.view_id,Operation.WRITE,('query','head'),('query','head',None),(),
                (TileDependency('query',EventKind.READ_READY,()),
                    *((TileDependency('query_rope',EventKind.READ_READY,()),) if tuple_query or cat else ()),
                    *(TileDependency(role,EventKind.READ_READY,('key',)) for role in ('indices','key','value'))),evidence,
                    unmapped=isinstance(out,LogicalSelectionView)))
            local = replace(scope,scope_id=identity,op_ids=(*header,str(request),str(start)))
            axes=(TileAxis('query',count,1),TileAxis('head',heads,1),TileAxis('key',maximum,1),
                *((TileAxis('kvrow',keys*heads,1),) if kv_parts is not None else ()))
            families.append(TileProgram(identity,local,axes,tuple(accesses),visibility,evidence,
                causal_prefixes=(TileCausalPrefix('key','query',offset=keys-queries+1 if kv_parts is not None else None),) if causal else ()))
        offset += queries
        kv_offset += keys
    if cat:
        # Include rows outside this attention phase, including captured pad
        # rows: torch.cat ran before the phase slice. Do not enumerate rows.
        cursor=0
        for first,count in (*sorted(ranges),(query.shape[0],0)):
            if first>cursor:
                identity=f'{scope.scope_id}.cat.unused{cursor}'
                children,accesses=materialize(identity,cursor,first-cursor);selected.extend(children)
                families.append(TileProgram(identity,replace(scope,scope_id=identity,op_ids=(*header,'unused',str(cursor))),
                    (TileAxis('query',first-cursor,1),TileAxis('head',heads,1)),tuple(accesses),visibility,evidence))
            cursor=first+count
    return tuple(selected),tuple(families)


def dsa_sparse_tile_demands(program, *, expected_kernel=None,expected_bindings=None):
    from src.kernel.memory_access import tile_template_requested_bytes
    program.__post_init__()
    if program.coverage.executable or program.accesses or not program.tile_programs or len(program.scopes) != 1:
        raise ValueError('Sparse DSA requires its unique nonexecutable query tile source')
    scope, = program.scopes;header = scope.op_ids;views = unique_by(program.views,'view_id')
    if not header:raise ValueError('DSA selected-query tile header is missing')
    tuple_query=header[0]=='dsa_sparse_tuple_tiles'
    has_kv=header[0]=='dsa_sparse_prefill_cat_kv_tiles'
    cat=header[0] in ('dsa_sparse_prefill_cat_tiles','dsa_sparse_prefill_cat_kv_tiles')
    if len(header)!=(14 if has_kv else 10 if cat else 9 if tuple_query else 7):raise ValueError('DSA selected-query tile header is missing')
    query,output = (views[name] for name in header[5:7])
    rope=views[header[7]] if tuple_query else None
    cat_parts=tuple(views[name] for name in header[7:9]) if cat else None
    kv_parts=tuple(views[name] for name in header[10:14]) if has_kv else None
    dimension=query.shape[2]+(rope.shape[2] if tuple_query else 0)
    if expected_bindings is not None:
        b=expected_bindings
        if (not isinstance(b,SparseAttentionBindings) or program.binding_id!=b.binding_id
                or (scope.call_id,scope.rank,scope.stream)!=(b.call_id,b.rank,b.stream)
                or program.allocations!=b.allocations or program.events!=b.events
                or any(v.view_id not in views or views[v.view_id]!=v for v in b.views)
                or header[5:7]!=(b.query_view,b.output_view)):
            raise ValueError('DSA source differs from its actual caller bindings')
        if tuple_query:
            if (b.query_parts!=(query.view_id,rope.view_id)
                    or header[8]!=','.join(f'{start}:{count}' for start,count in b.query_row_ranges)):
                raise ValueError('DSA source differs from its actual caller tuple rows')
        elif cat:
            if (b.query_parts!=header[7:9] or header[9]!=','.join(f'{start}:{count}' for start,count in b.query_row_ranges)):
                raise ValueError('DSA source differs from its actual caller prefill cat rows')
        elif b.query_parts or b.query_row_ranges:
            raise ValueError('DSA source dropped its actual caller query tuple')
        if b.kv_parts!=(header[10:14] if has_kv else ()):
            raise ValueError('DSA source differs from actual caller prefill KV parts')
    if expected_kernel is not None:
        kernel = expected_kernel
        expected_header = (header[0],kernel.variant,str(kernel.attributes['topk_capacity']),
            ','.join(map(str,kernel.attributes['q_lens'])),','.join(map(str,kernel.attributes['kv_lens'])))
        tokens = sum(kernel.attributes['q_lens'])
        if (kernel.kind is not KernelKind.SPARSE_ATTN or kernel.variant not in ('dsa_sparse_prefill','dsa_sparse_decode')
                or kernel.execution_support != 'supported' or kernel.hardware_id != program.hardware_id
                or header[:5] != expected_header or query.dtype != kernel.dtype or output.dtype != kernel.dtype
                or query.element_bytes != kernel.shape['dtype_bytes'] or output.element_bytes != kernel.shape['dtype_bytes']
                or dimension!=kernel.shape['qk_head_dim'] or query.shape[1]!=kernel.shape['num_query_heads']
                or (not tuple_query and not cat and query.shape[0]!=tokens)
                or (tuple_query and (query.shape[2]!=kernel.shape['v_head_dim']
                    or rope.shape[1]!=query.shape[1] or (rope.dtype,rope.element_bytes)!=(query.dtype,query.element_bytes)))
                or output.shape != (tokens,kernel.shape['num_query_heads'],kernel.shape['v_head_dim'])):
            raise ValueError('Source differs from its actual DSA attention invocation')
    roots = tuple(views[f'{scope.scope_id}.{role}.root'] for role in ('indices','key','value'))
    first = unique_by(program.tile_programs[0].accesses,'access_id')
    ready = tuple(first[role].input_ready[0] for role in ('cat.nope.read' if cat else 'query','indices',
        'key_cat.nope.read' if has_kv else 'key','value'))
    if tuple_query:ready=(*ready,first['query_rope'].input_ready[0])
    elif cat:ready=(*ready,first['cat.rope.read'].input_ready[0])
    if has_kv:ready=(*ready,first['key_cat.rope.read'].input_ready[0])
    if expected_bindings is not None:
        wanted=(b.query_ready,*b.input_ready)
        if tuple_query:wanted=(*wanted,dict(b.availability)[rope.view_id])
        elif cat:wanted=(*wanted,dict(b.availability)[cat_parts[1].view_id])
        if has_kv:wanted=(*wanted,dict(b.availability)[kv_parts[2].view_id])
        if ready!=wanted or any(f.visibility_domain!=b.visibility_domain for f in program.tile_programs):
            raise ValueError('DSA source differs from actual caller query/cache readiness')
    if cat:
        if (not isinstance(query,TensorView) or query.strides_bytes!=(query.shape[1]*query.shape[2]*query.element_bytes,
                query.shape[2]*query.element_bytes,query.element_bytes)
                or not 0<cat_parts[0].shape[2]<query.shape[2]
                or cat_parts[0].shape[1]!=query.shape[1]
                or cat_parts[1].shape[1:]!=(query.shape[1],query.shape[2]-cat_parts[0].shape[2])
                or any(p.shape[0]!=query.shape[0] or (p.dtype,p.element_bytes)!=(query.dtype,query.element_bytes) for p in cat_parts)
                or any(e.allocation_id==query.allocation_id and e.kind is not EventKind.ALLOCATED for e in program.events)):
            raise ValueError('Prefill cat lost actual full-row inputs or fabricated preinitialized output')
    if has_kv:
        _validate_prefill_kv_parts(kv_parts,unique_by(program.allocations,'allocation_id'),
            (sum(int(n) for n in header[4].split(',')),query.shape[1],dimension,output.shape[2],query.element_bytes,query.dtype))
        if any(e.allocation_id==kv_parts[0].allocation_id and e.kind is not EventKind.ALLOCATED for e in program.events):
            raise ValueError('Prefill K concat cannot invent preinitialized content')
    selected,expected = _dsa_tile_recipe(header,query,output,roots,ready,scope,program.tile_programs[0].visibility_domain,
        rope=rope,cat_parts=cat_parts,kv_parts=kv_parts)
    parents=(query,output,*((rope,) if tuple_query else ()),*(cat_parts if cat else ()),*(kv_parts if has_kv else ()),*roots,*selected)
    if (program.tile_programs != expected or set(views) != {view.view_id for view in parents}
            or any(views[view.view_id] != view for view in selected)):
        raise ValueError('Sparse attention lacks actual index/payload producer causality or exact query membership')
    tokens=sum(int(n) for n in header[3].split(','));heads=query.shape[1]
    totals = {Operation.READ:0,Operation.WRITE:0};helpers={Operation.READ:0,Operation.WRITE:0};key_helpers=dict(helpers);index_bytes = 0
    for family in program.tile_programs:
        for template in family.accesses:
            count = tile_template_requested_bytes(program,family,template)
            if template.access_id.startswith('key_cat.'):key_helpers[template.operation]+=count
            elif template.access_id.startswith('cat.'):helpers[template.operation]+=count
            else:totals[template.operation] += count
            if template.access_id == 'indices':index_bytes += count
    pairs,remainder = divmod(index_bytes,heads*4)
    if remainder:raise ValueError('DSA selected index replay cannot be projected exactly')
    selector = f'{program.binding_id}.q{header[3]}.kv{header[4]}.k{header[2]}'
    expected_parents=(None,kv_parts[0].view_id,kv_parts[3].view_id) if has_kv else (None,None,None)
    if any(root.parent_view_id!=parent or root.selection_id != f'{selector}.{role}'
            for role,root,parent in zip(('indices','key','value'),roots,expected_parents,strict=True)):
        raise ValueError('DSA requires actual selected-index/cache roots')
    for root,role,width in zip(roots,('indices','key','value'),(4,query.element_bytes,query.element_bytes),strict=True):
        shape = ((pairs,) if role == 'indices' else ((pairs,dimension if role == 'key' else output.shape[2])
            if header[1].endswith('decode') else (pairs,heads,dimension if role == 'key' else output.shape[2])))
        if (root.shape,root.dtype,root.element_bytes) != (shape,'int32' if role == 'indices' else query.dtype,width):
            raise ValueError('DSA typed selected index/cache recipe mismatch')
    if ((not has_kv and roots[1].allocation_id!=roots[2].allocation_id)
            or has_kv and (roots[1].allocation_id,roots[2].allocation_id)!=(kv_parts[0].allocation_id,kv_parts[3].allocation_id)
            or len({query.allocation_id,output.allocation_id,roots[0].allocation_id,roots[1].allocation_id})!=4):
        raise ValueError('DSA shared cache/independent operand identity mismatch')
    if tuple_query and rope.allocation_id in {query.allocation_id,output.allocation_id,roots[0].allocation_id,roots[1].allocation_id}:
        raise ValueError('DSA tuple query parts must retain independent actual storage')
    if cat and len({query.allocation_id,output.allocation_id,roots[0].allocation_id,roots[1].allocation_id,
            *(p.allocation_id for p in cat_parts)})!=6:
        raise ValueError('Prefill cat requires independent actual inputs and a new packed query parent')
    result={'geometry':(tokens,heads,dimension,output.shape[2],pairs),'read_bytes':totals[Operation.READ],'write_bytes':totals[Operation.WRITE]}
    if cat:result.update(unpriced_query_materialization_read_bytes=helpers[Operation.READ],
        unpriced_query_materialization_write_bytes=helpers[Operation.WRITE])
    if has_kv:result.update(unpriced_key_materialization_read_bytes=key_helpers[Operation.READ],
        unpriced_key_materialization_write_bytes=key_helpers[Operation.WRITE])
    return result



def _v4_prefix_sum(total,capacity,divisor):
    active=min(total,capacity*divisor-1);quotient,remainder=divmod(active,divisor)
    return divisor*quotient*(quotient-1)//2+quotient*(remainder+1)+(total-active)*capacity


def v4_sparse_geometry(kernel):
    q_lengths,kv_lengths = kernel.attributes['q_lens'],kernel.attributes['kv_lens']
    ratio,window = kernel.shape['compress_ratio'],kernel.shape['window_size']
    selected = kernel.attributes['selected_kv_lens']
    selected_pairs = sum(q*count for q,count in zip(q_lengths,selected,strict=True))
    prefix=_v4_prefix_sum
    if kernel.attributes['compressed_selection'] == 'all_completed_states':
        selected_pairs = sum(prefix(kv,kernel.attributes['topk_capacity'],ratio)-prefix(kv-q,kernel.attributes['topk_capacity'],ratio)
            for q,kv in zip(q_lengths,kv_lengths,strict=True))
    swa_pairs = sum(prefix(kv,window,1)-prefix(kv-q,window,1)
        for q,kv in zip(q_lengths,kv_lengths,strict=True))
    compressed_union = sum(kv//ratio for kv in kv_lengths) if ratio > 1 else 0
    swa_union = sum(q+min(kv-q,window-1) for q,kv in zip(q_lengths,kv_lengths,strict=True))
    split_rows = sum(q*s for q,s in zip(q_lengths,kernel.attributes['split_counts'],strict=True)) if kernel.variant.endswith('decode') else 0
    return sum(q_lengths),kernel.shape['num_query_heads'],kernel.shape['qk_head_dim'],kernel.shape['v_head_dim'],selected_pairs,swa_pairs,compressed_union,swa_union,split_rows


def v4_regions(kernel, *, selection_id):
    tokens,heads,qk_dim,value_dim,selected,swa,c_union,s_union,splits = v4_sparse_geometry(kernel)
    label = f"{selection_id}.q{','.join(map(str,kernel.attributes['q_lens']))}.kv{','.join(map(str,kernel.attributes['kv_lens']))}.s{','.join(map(str,kernel.attributes['selected_kv_lens']))}.c{kernel.shape['compress_ratio']}.window{kernel.shape['window_size']}.capacity{kernel.attributes['topk_capacity']}.{kernel.attributes['compressed_selection']}"
    prefill = kernel.variant.endswith('prefill')
    regions = {}
    if selected:
        regions['indices'] = Unmapped((selected,),'int32',4,f'{label}.indices')
    if not prefill and kernel.shape['compress_ratio']==4:
        capacity=kernel.attributes['topk_capacity']
        regions['indices']=Unmapped((tokens,capacity),'int32',4,f'{label}.local_indices')
        regions['global_indices']=Unmapped((tokens,capacity),'int32',4,f'{label}.global_indices')
        regions['compressed_pages']=Unmapped((tokens,capacity),'int32',4,f'{label}.conditional_page_lookups')
        regions['token_to_req']=Unmapped((tokens,),'int32',4,f'{label}.token_to_req')
        regions['is_valid_token']=Unmapped((tokens,),'bool',1,f'{label}.is_valid_token')
    compressed = c_union if prefill else selected
    if compressed:
        regions['compressed'] = Unmapped((compressed,kernel.shape['compressed_state_bytes']),'uint8',1,f'{label}.compressed')
    # One584B packed SWA record, represented by the retained two decode
    # role uses. This is not a proven two-fetch issued trace.
    regions['swa'] = Unmapped((s_union if prefill else swa,kernel.shape['compressed_state_bytes']),
                              'uint8',1,f'{label}.swa')
    if prefill:
        regions['workspace'] = Unmapped((c_union+s_union,qk_dim),'bf16',2,f'{label}.gather_union')
        regions['key_role'] = Unmapped((selected+swa,qk_dim),'bf16',2,f'{label}.selected_key_role')
        regions['value_role'] = Unmapped((selected+swa,value_dim),'bf16',2,f'{label}.selected_value_role')
    else:
        # The caller forwards decode_swa_indices [T,1,window] and its
        # INT32[T] lengths. Flatten only the singleton KV-head dimension;
        # slot values and cache addresses remain unknown.
        regions['swa_indices'] = Unmapped((tokens,kernel.shape['window_size']),'int32',4,f'{label}.swa_indices')
        regions['swa_lengths'] = Unmapped((tokens,),'int32',4,f'{label}.swa_lengths')
        if kernel.shape['compress_ratio'] > 1:
            regions['compressed_lengths'] = Unmapped((tokens,),'int32',4,f'{label}.compressed_lengths')
        partial = f"{label}.splits{','.join(map(str,kernel.attributes['split_counts']))}"
        regions['partial_output'] = Unmapped((splits,heads,value_dim),'fp32',4,f'{partial}.partial_output')
        regions['partial_lse'] = Unmapped((splits,heads),'fp32',4,f'{partial}.partial_lse')
    return regions


def standalone_v4_sparse_bindings(kernel, *, call_id, binding_id):
    tokens,heads,qk_dim,value_dim,*_ = v4_sparse_geometry(kernel)
    allocations,views,events = [],[],[]
    for role,dimension in (('query',qk_dim),('output',value_dim)):
        identity = f'{binding_id}.{role}'
        dtype = kernel.dtype if kernel.variant.endswith('prefill') else 'bf16'
        view = TensorView(role,identity,0,(tokens,heads,dimension),
            (heads*dimension*2,dimension*2,2),dtype,2)
        allocation = MemoryAllocation(identity,'logical-local-device','logical-storage',tokens*heads*dimension*2,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation)
        views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role == 'query':
            event = ProgramEvent(f'{identity}.initialized',EventKind.INITIALIZED,identity,view.region(),'sparse-visible',
                                 (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event)
            query_ready = EventRef(event.event_id,EventKind.INITIALIZED)
    resources,availability = [],[]
    for role,region in v4_regions(kernel,selection_id=binding_id).items():
        if role in ('key_role','value_role'):
            continue
        identity = f'{binding_id}.{role}'
        generated=kernel.variant.endswith('decode') and kernel.shape['compress_ratio']==4 and role in ('global_indices','compressed_lengths')
        from math import prod
        allocations.append(MemoryAllocation(identity,'logical-local-device','logical-v4-sparse-storage',
            prod(region.logical_shape)*region.element_bytes if generated else None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None))
        events.append(ProgramEvent(f'{identity}.allocated',EventKind.ALLOCATED,identity,None,None,()))
        resources.append((role,identity))
        if not generated and role in ('indices','compressed','swa','swa_indices','swa_lengths','compressed_lengths','compressed_pages','token_to_req','is_valid_token'):
            event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,region,'sparse-visible',
                                 (EventRef(f'{identity}.allocated',EventKind.ALLOCATED),))
            events.append(event)
            availability.append((role,EventRef(event.event_id,EventKind.LOGICAL_AVAILABLE)))
    objects = dict(resources)
    # The V4 caller always passes its learned, padded FP32 sink parameter.
    # Its values are external inputs, including padded heads; the constructor's
    # -inf initialization is not the value of a loaded model parameter.
    identity = f'{binding_id}.attn_sink'
    sink = TensorView('attn_sink',identity,0,(heads,),(4,),'fp32',4)
    allocation = MemoryAllocation(identity,'logical-local-device','logical-v4-parameter',heads*4,
        AddressLevel.OBJECT,None,None,identity+'.allocated',None)
    event = ProgramEvent(identity+'.available',EventKind.INITIALIZED,identity,sink.region(),'sparse-visible',
        (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
    allocations.append(allocation);views.append(sink)
    events.extend((ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()),event))
    binding = SparseAttentionBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        'query','output',objects['indices'] if 'indices' in objects else None,objects['swa'],query_ready,(EventRef(event.event_id,event.kind),),
        0,'offline','sparse-visible',tuple(resources),tuple(availability))
    return _v4_prefill_aux_bindings(kernel,binding) if kernel.variant.endswith('prefill') else binding


def build_v4_sparse_source(kernel, hardware, bindings, granularity, *, program_id):
    if not isinstance(bindings,SparseAttentionBindings):
        raise TypeError('V4 sparse attention requires typed query/cache/workspace/partial roles')
    parents = unique_by(bindings.views,'view_id')
    sink = parents['attn_sink']
    if (sink.shape,sink.dtype,sink.element_bytes,sink.strides_bytes) != ((kernel.shape['num_query_heads'],),'fp32',4,(4,)):
        raise ValueError('V4 attention requires its padded FP32 sink parameter')
    if len(bindings.input_ready) != 1:
        raise ValueError('V4 attention requires sink input availability')
    ref, = bindings.input_ready
    event = unique_by(bindings.events,'event_id')[ref.node_id]
    if (event.kind is not ref.kind or event.kind not in (EventKind.INITIALIZED,EventKind.WRITE_VISIBLE)
            or event.allocation_id != sink.allocation_id or event.region != sink.region()):
        raise ValueError('V4 sink availability must cover its actual parameter')
    if kernel.variant.endswith('prefill'):
        if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown V4 prefill granularity')
        return _build_v4_prefill_tiles(kernel,hardware,bindings,program_id=program_id)
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('Unknown V4 decode granularity')
    tokens,heads,qk_dim,value_dim,selected,swa,*_ = v4_sparse_geometry(kernel)
    prefill = kernel.variant.endswith('prefill')
    head_packs = heads//64 if prefill else 1
    if prefill and (heads%64 or kernel.launch.grid[0] != tokens*head_packs):
        raise ValueError('dsv4 sparse prefill launch disagrees with the64-head CTA pack count')
    regions = v4_regions(kernel,selection_id=bindings.binding_id)
    objects,available = dict(bindings.resources),dict(bindings.availability)
    if len(objects) != len(bindings.resources) or len(available) != len(bindings.availability):
        raise ValueError('duplicate V4 sparse resource/availability role')
    needed = set(regions)-{'key_role','value_role'}
    inputs=needed & {'indices','compressed','swa','swa_indices','swa_lengths','compressed_lengths','compressed_pages','token_to_req','is_valid_token'}
    if kernel.shape['compress_ratio']==4:inputs-= {'compressed_lengths'}
    if set(objects) != needed or set(available) != inputs:
        raise ValueError('V4 sparse cache/workspace/partial role or availability missing')
    if bindings.cache != objects['swa'] or bindings.indices != (objects['indices'] if 'indices' in objects else None):
        raise ValueError('V4 sparse index/cache parent identity mismatch')
    if len(set(objects.values())) != len(objects):
        raise NotImplementedError('V4 sparse role overlap requires an explicit implementation lifecycle')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    query,output = views[bindings.query_view],views[bindings.output_view]
    for view,dimension in ((query,qk_dim),(output,value_dim)):
        if (view.shape,view.dtype,view.element_bytes) != ((tokens,heads,dimension),'bf16',2):
            raise ValueError('V4 sparse query/output storage mismatch')
    if len({query.allocation_id,output.allocation_id,*objects.values()}) != len(objects)+2:
        raise ValueError('V4 query/output/cache/temporary parents must be distinct')
    return _build_v4_decode_tiles(kernel,hardware,bindings,program_id=program_id)


def _v4_decode_parameters(header):
    if len(header)!=12 or header[0]!='v4_sparse_decode_tiles':raise ValueError('V4 decode requires its own tile ABI')
    q,kv,selected,splits=(tuple(map(int,text.split(','))) for text in header[1:5])
    ratio,window,repeat,capacity=(int(header[i]) for i in (5,6,10,11))
    if (not q or len({len(q),len(kv),len(selected),len(splits)})!=1 or min((*q,*splits,window,capacity))<=0
            or any(k<n or not 0<=s<=capacity for n,k,s in zip(q,kv,selected,strict=True))
            or ratio not in (1,4,128) or repeat!=1 or header[7] not in ('candidate_upper_bound','all_completed_states','declared_causal_topk')):
        raise ValueError('V4 decode request/split/selection ABI mismatch')
    return q,kv,selected,splits,ratio,window


def _v4_c4_mapping_recipe(identity,q,offset,capacity,roots,ready,evidence):
    """Actual full-capacity C4 mapper, with conditional page requests.

    Each (query,candidate) is one logical lane of the source's 1024-lane
    panels; omitted panel padding adds no memory accesses. No page value or
    positivity of a local index is inferred from q/kv lengths.
    """
    from src.kernel.memory_access import ReadValuePredicate,TileValueIndex
    selected=[];accesses=[]
    def logical(role,shape):
        parent=roots[role]
        mapping=(LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),offset),)
        if len(shape)==2:mapping+= (LogicalCoordinateBinding(1,(LogicalCoordinateTerm(1,1),)),)
        # Page-table coordinates depend on the loaded request/local-index
        # values, never the candidate's ordinal within the source array.
        if role=='compressed_pages':mapping=()
        view=LogicalSelectionView(f'{identity}.map.{role}',parent.allocation_id,shape,parent.dtype,parent.element_bytes,
            f'{identity}.map.{role}.actual_input_values',parent.view_id,mapping)
        selected.append(view);return view
    for role,name in (('token_to_req','map.request'),('is_valid_token','map.valid')):
        view=logical(role,(q,))
        accesses.append(TileAccessTemplate(name,view.view_id,Operation.READ,('query',),('query',),
            (ready[role],),(),evidence,unmapped=True))
    local=logical('indices',(q,capacity));pages=logical('compressed_pages',(q,capacity))
    lanes=('query','candidate')
    accesses.append(TileAccessTemplate('map.local_indices',local.view_id,Operation.READ,lanes,lanes,
        (ready['indices'],),(),evidence,unmapped=True))
    accesses.append(TileAccessTemplate('map.pages',pages.view_id,Operation.READ,lanes,lanes,
        (ready['compressed_pages'],),(TileDependency('map.local_indices',EventKind.READ_READY,()),
            TileDependency('map.request',EventKind.READ_READY,())),evidence,unmapped=True,
        value_index=TileValueIndex('map.local_indices','loaded_scalar'),
        predicate=ReadValuePredicate('map.local_indices','nonnegative')))
    parent=roots['global_indices']
    output=TensorView(f'{identity}.map.global_indices',parent.allocation_id,parent.offset_bytes+offset*capacity*4,
        (q,capacity),(capacity*4,4),'int32',4)
    parent=roots['compressed_lengths']
    lengths=TensorView(f'{identity}.map.lengths',parent.allocation_id,parent.offset_bytes+offset*4,(q,),(4,),'int32',4)
    selected.extend((output,lengths))
    accesses.append(TileAccessTemplate('map.global_indices',output.view_id,Operation.WRITE,lanes,lanes,(),
        (TileDependency('map.local_indices',EventKind.READ_READY,()),
            TileDependency('map.pages',EventKind.READ_READY,(),allow_skipped_read=True)),evidence))
    accesses.append(TileAccessTemplate('map.lengths',lengths.view_id,Operation.WRITE,('query',),('query',),(),
        (TileDependency('map.local_indices',EventKind.READ_READY,('candidate',)),
            TileDependency('map.valid',EventKind.READ_READY,())),evidence))
    return tuple(selected),tuple(accesses),output,lengths


def _v4_decode_recipe(header,parents,roots,ready,scope,visibility):
    from src.kernel.memory_access import TilePartition
    qlens,kvlens,selected,splits,ratio,window=_v4_decode_parameters(header)
    heads,dim=parents[header[8]].shape[1:];value_dim=parents[header[9]].shape[2]
    views=[];families=[];query_offset=partial_offset=0
    evidence=('V4 sparse decode logical query and FP32 split/combine interfaces; split counts are explicit per request.',
        'The planner key-to-split membership is unbound. No even-share schedule or physical cache coordinate is inferred.',
        'Packed SWA key/value role uses retain the declared two-role projection, not two measured native fetches.',
        'C4 cache_utils.py compute_global_topk_indices_and_lens reads every local top-k entry in masked 1024-lane panels, conditionally loads its page for nonnegative IDs, writes all global slots including -1, and independently counts valid entries before applying is_valid_token. Conditional page bytes are an upper bound, not observed traffic.',
        'v0.30 sparse_swa.py supplies INT32[T,1,window] slot indices and INT32[T] lengths; the V4 CUDA caller forwards these plus compressed top-k lengths. Logical active slots wait for their own length read; cache roles wait for actual slot-index READ_READY. Native planner/issue replay is unbound.',
        'V4 attention.py declares an FP32 padded-head sink; nvidia/flashmla.py passes it to decode and prefill. The same-version CPU FlashMLA algorithm reference applies sink correction to final O; one logical per-query head-vector read, with no CPU/GPU issue or scratch equivalence claim.')
    def term(axis,*terms,offset=0):
        return LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(i,c) for i,c in terms),offset)
    for request,(q,kv,count,nsplit) in enumerate(zip(qlens,kvlens,selected,splits,strict=True)):
        identity=f'{scope.scope_id}.r{request}';local=replace(scope,scope_id=identity,op_ids=(*header,str(request)))
        accesses=[];partitions=[];prefixes=[]
        axes=[TileAxis('query',q,1),TileAxis('split',nsplit,1)]
        def view(role,shape,coordinates=(),root=False):
            parent=roots[role] if root else parents[header[8] if role=='query' else header[9]]
            child=LogicalSelectionView(f'{identity}.{role}',parent.allocation_id,shape,parent.dtype,parent.element_bytes,
                f'{identity}.{role}.logical',parent.view_id,coordinates)
            views.append(child);return child
        def add(name,parent,op,loops,mapping,deps=(),inputs=(),unmapped=True):
            accesses.append(TileAccessTemplate(name,parent.view_id,op,loops,mapping,inputs,deps,evidence,unmapped=unmapped))
        if ratio==4:
            capacity=int(header[11]);axes.append(TileAxis('candidate',capacity,1))
            mapped,producer,global_indices,generated_lengths=_v4_c4_mapping_recipe(
                identity,q,query_offset,capacity,roots,ready,evidence)
            views.extend(mapped);accesses.extend(producer)
        query=view('query',(q,heads,dim),(term(0,(0,1),offset=query_offset),term(1,(1,1)),term(2,(2,1))))
        output=view('output',(q,heads,value_dim),(term(0,(0,1),offset=query_offset),term(1,(1,1)),term(2,(2,1))))
        add('attn_sink',parents['attn_sink'],Operation.READ,('query',),(None,),inputs=(ready['attn_sink'],))
        add('query',query,Operation.READ,('query',),('query',None,None),inputs=(ready['query'],))
        dependencies=[TileDependency('query',EventKind.READ_READY,())]
        for role in ('swa_lengths',*(('compressed_lengths',) if ratio>1 else ())):
            if role=='compressed_lengths' and ratio==4:
                add(role,generated_lengths,Operation.READ,('query',),('query',),
                    (TileDependency('map.lengths',EventKind.WRITE_VISIBLE,()),),unmapped=False)
            else:
                length=view(role,(q,),(term(0,(0,1),offset=query_offset),),root=True)
                add(role,length,Operation.READ,('query',),('query',),inputs=(ready[role],))
            dependencies.append(TileDependency(role,EventKind.READ_READY,()))
        count=min(int(header[11]),kv//ratio) if header[7]=='all_completed_states' else count
        if count:
            axes.append(TileAxis('compressed_key',count,1));partitions.append(TilePartition('compressed_key',None,'split',('query',)))
            prefixes.append(TileCausalPrefix('compressed_key','query',kv-q+1,1,ratio))
            if ratio==4:
                indices=replace(global_indices,view_id=f'{identity}.indices',shape=(q,count))
                views.append(indices)
            else:indices=view('indices',(q,count),root=True)
            packed=view('compressed',(q,count,584),root=True)
            add('indices',indices,Operation.READ,('query','compressed_key'),('query','compressed_key'),
                (TileDependency('compressed_lengths',EventKind.READ_READY,()),
                    *((TileDependency('map.global_indices',EventKind.WRITE_VISIBLE,('candidate',)),) if ratio==4 else ())),
                inputs=() if ratio==4 else (ready['indices'],),unmapped=ratio!=4)
            add('compressed',packed,Operation.READ,('query','compressed_key'),('query','compressed_key',None),
                (TileDependency('indices',EventKind.READ_READY,()),),inputs=(ready['compressed'],))
            dependencies.append(TileDependency('compressed',EventKind.READ_READY,('compressed_key',),partition_axis='compressed_key'))
        swa_count=min(window,kv);axes.append(TileAxis('swa_key',swa_count,1))
        prefixes.append(TileCausalPrefix('swa_key','query',kv-q+1))
        partitions.append(TilePartition('swa_key',None,'split',('query',)))
        indices=view('swa_indices',(q,swa_count),(term(0,(0,1),offset=query_offset),term(1,(1,1))),root=True)
        add('swa_indices',indices,Operation.READ,('query','swa_key'),('query','swa_key'),
            (TileDependency('swa_lengths',EventKind.READ_READY,()),),inputs=(ready['swa_indices'],))
        swa=view('swa',(q,swa_count,584),root=True)
        for role in ('swa_key_use','swa_value_use'):
            add(role,swa,Operation.READ,('query','swa_key'),('query','swa_key',None),
                (TileDependency('swa_indices',EventKind.READ_READY,()),),inputs=(ready['swa'],))
            dependencies.append(TileDependency(role,EventKind.READ_READY,('swa_key',),partition_axis='swa_key'))
        final=[]
        for role,shape in (('partial_output',(q,nsplit,heads,value_dim)),('partial_lse',(q,nsplit,heads))):
            mapping=(term(0,(0,nsplit),(1,1),offset=partial_offset),term(1,(2,1)),
                *((term(2,(3,1)),) if role=='partial_output' else ()))
            temporary=view(role,shape,mapping,root=True);indices=('query','split',*((None,)*(len(shape)-2)))
            add(role+'.write',temporary,Operation.WRITE,('query','split'),indices,tuple(dependencies))
            add(role+'.read',temporary,Operation.READ,('query','split'),indices,
                (TileDependency(role+'.write',EventKind.WRITE_VISIBLE,()),))
            final.append(TileDependency(role+'.read',EventKind.READ_READY,('split',)))
        add('output',output,Operation.WRITE,('query',),('query',None,None),
            (*final,TileDependency('attn_sink',EventKind.READ_READY,())))
        families.append(TileProgram(identity,local,tuple(axes),tuple(accesses),visibility,evidence,tuple(prefixes),tuple(partitions)))
        query_offset+=q;partial_offset+=q*nsplit
    return tuple(views),tuple(families)


def _build_v4_decode_tiles(kernel,hardware,bindings,*,program_id):
    parents=unique_by(bindings.views,'view_id');regions=v4_regions(kernel,selection_id=bindings.binding_id)
    objects=dict(bindings.resources);available=dict(bindings.availability)
    roots={role:LogicalSelectionView(f'{program_id}.{role}.root',objects[role],region.logical_shape,
        region.dtype,region.element_bytes,region.selection_id,None) for role,region in regions.items()}
    if kernel.shape['compress_ratio']==4:
        tokens=sum(kernel.attributes['q_lens']);capacity=kernel.attributes['topk_capacity']
        for role,shape,strides in (('global_indices',(tokens,capacity),(capacity*4,4)),('compressed_lengths',(tokens,),(4,))):
            roots[role]=TensorView(f'{program_id}.{role}.root',objects[role],0,shape,strides,'int32',4)
    header=('v4_sparse_decode_tiles',*( ','.join(map(str,kernel.attributes[name])) for name in
        ('q_lens','kv_lens','selected_kv_lens','split_counts')),str(kernel.shape['compress_ratio']),str(kernel.shape['window_size']),
        kernel.attributes['compressed_selection'],bindings.query_view,bindings.output_view,
        str(kernel.shape['swa_token_bytes']//kernel.shape['compressed_state_bytes']),str(kernel.attributes['topk_capacity']))
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-v4-decode-query-splits')
    selected,families=_v4_decode_recipe(header,parents,roots,
        {'query':bindings.query_ready,'attn_sink':bindings.input_ready[0],**available},scope,bindings.visibility_domain)
    result=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Query-local split producers and combine readers share exact FP32 parents. Key-to-split memberships remain unbound.',
            'Causal selection and the existing factor-two partial pricing projection remain separate from native traffic.',
            'Actual SWA slot-index and per-query length inputs are explicit logical payloads, and remain invocation-boundary inputs without an invented metadata producer.',
            'The padded FP32 sink parameter gates final O; pricing counts its unique input payload. Native scheduler metadata, final-LSE and native issue details remain unbound; this is not full-domain acceptance.')),
        bindings.allocations,(*bindings.views,*roots.values(),*selected),(scope,),bindings.events,(),families)
    v4_decode_tile_demands(result);return result


def v4_decode_tile_demands(program, *, expected_kernel=None):
    from src.kernel.memory_access import tile_template_requested_bytes
    program.__post_init__()
    if program.accesses or len(program.scopes)!=1 or not program.tile_programs:
        raise ValueError('V4 decode requires its unique request/split tile source')
    scope,=program.scopes;header=scope.op_ids;qlens,kvlens,selected,splits,ratio,window=_v4_decode_parameters(header)
    if expected_kernel is not None:
        kernel = expected_kernel
        if (kernel.kind is not KernelKind.SPARSE_ATTN or kernel.variant != 'dsv4_sparse_decode'
                or kernel.hardware_id != program.hardware_id or kernel.execution_support != 'supported'
                or (qlens,kvlens,selected,splits,ratio,window) != (
                    tuple(kernel.attributes['q_lens']),tuple(kernel.attributes['kv_lens']),
                    tuple(kernel.attributes['selected_kv_lens']),tuple(kernel.attributes['split_counts']),
                    kernel.shape['compress_ratio'],kernel.shape['window_size'])
                or header[7] != kernel.attributes['compressed_selection']
                or int(header[11]) != kernel.attributes['topk_capacity']):
            raise ValueError('Source differs from its actual V4 decode invocation')
    views=unique_by(program.views,'view_id');parents={name:views[name] for name in (header[8],header[9],'attn_sink')}
    roots={role:views[f'{scope.scope_id}.{role}.root'] for role in ('indices','global_indices','compressed','swa','swa_indices','swa_lengths','compressed_lengths','compressed_pages','token_to_req','is_valid_token','partial_output','partial_lse')
        if f'{scope.scope_id}.{role}.root' in views}
    query,output=parents[header[8]],parents[header[9]];tokens,heads,dim=query.shape;value_dim=output.shape[2]
    if expected_kernel is not None and (heads,dim,value_dim) != (expected_kernel.shape['num_query_heads'],
            expected_kernel.shape['qk_head_dim'],expected_kernel.shape['v_head_dim']):
        raise ValueError('Source differs from its actual V4 attention dimensions')
    sink=parents['attn_sink']
    if (sink.shape,sink.dtype,sink.element_bytes,sink.strides_bytes)!=((heads,),'fp32',4,(4,)):
        raise ValueError('V4 decode requires its actual padded FP32 sink')
    if ((tokens,heads)!=(sum(qlens),output.shape[1]) or output.shape[0]!=tokens
            or query.dtype!=output.dtype or query.dtype!='bf16' or query.element_bytes!=output.element_bytes or query.element_bytes!=2):
        raise ValueError('V4 decode request query/output typed geometry mismatch')
    count=sum(q*n for q,n in zip(qlens,selected,strict=True)) if header[7]=='candidate_upper_bound' else sum(
        _v4_prefix_sum(k,int(header[11]),ratio)-_v4_prefix_sum(k-q,int(header[11]),ratio) for q,k in zip(qlens,kvlens,strict=True))
    swa=sum(_v4_prefix_sum(k,window,1)-_v4_prefix_sum(k-q,window,1) for q,k in zip(qlens,kvlens,strict=True))
    partials=sum(q*n for q,n in zip(qlens,splits,strict=True))
    expected_roots={'swa':((swa,584),'uint8',1),'swa_indices':((tokens,window),'int32',4),'swa_lengths':((tokens,),'int32',4),
        'partial_output':((partials,heads,value_dim),'fp32',4),
        'partial_lse':((partials,heads),'fp32',4)}
    if ratio>1:expected_roots['compressed_lengths']=((tokens,),'int32',4)
    if count:expected_roots.update(indices=((count,),'int32',4),compressed=((count,584),'uint8',1))
    if ratio==4:
        expected_roots.update(indices=((tokens,int(header[11])),'int32',4),global_indices=((tokens,int(header[11])),'int32',4),
            compressed_pages=((tokens,int(header[11])),'int32',4),token_to_req=((tokens,),'int32',4),is_valid_token=((tokens,),'bool',1))
    if set(roots)!=set(expected_roots):raise ValueError('V4 decode cache/partial roles differ from actual requests')
    for role,(shape,dtype,width) in expected_roots.items():
        view=roots[role]
        generated=ratio==4 and role in ('global_indices','compressed_lengths')
        valid_type=(isinstance(view,TensorView) and view.offset_bytes==0
            and view.strides_bytes==((shape[1]*4,4) if len(shape)==2 else (4,))) if generated else (
                isinstance(view,LogicalSelectionView) and view.parent_view_id is None)
        if not valid_type or (view.shape,view.dtype,view.element_bytes)!=(shape,dtype,width):
            raise ValueError('V4 decode cache/partial typed root geometry mismatch')
    if len({view.allocation_id for view in (*parents.values(),*roots.values())})!=len(parents)+len(roots):
        raise ValueError('V4 decode query/output/cache/partial parents must be distinct')
    ready={}
    helper_inputs={'map.local_indices':'indices','map.pages':'compressed_pages','map.request':'token_to_req','map.valid':'is_valid_token'}
    for family in program.tile_programs:
        for access in family.accesses:
            if ratio==4 and access.access_id in ('indices','compressed_lengths'):continue
            if access.access_id not in (*helper_inputs,'query','attn_sink','indices','compressed','swa_indices','swa_lengths','compressed_lengths','swa_key_use','swa_value_use'):continue
            role=helper_inputs[access.access_id] if access.access_id in helper_inputs else 'swa' if access.access_id in ('swa_key_use','swa_value_use') else access.access_id
            if len(access.input_ready)!=1 or role in ready and ready[role]!=access.input_ready[0]:
                raise ValueError('V4 decode input availability differs between requests')
            ready[role]=access.input_ready[0]
    children,expected=_v4_decode_recipe(header,parents,roots,ready,scope,program.tile_programs[0].visibility_domain)
    if (program.tile_programs!=expected or set(views)!={*parents,*(view.view_id for view in roots.values()),*(view.view_id for view in children)}
            or any(views[view.view_id]!=view for view in children)):
        raise ValueError('V4 decode lost request/split membership or actual index/partial producer causality')
    read=write=partial_read=partial_write=indices_bytes=swa_bytes=priced_indices_bytes=budget_extra=sink_bytes=0
    metadata_bytes={'swa_indices':0,'swa_lengths':0,'compressed_lengths':0}
    helper_read=helper_write=conditional_pages=0
    for family in program.tile_programs:
        envelope=replace(family,causal_prefixes=tuple(prefix for prefix in family.causal_prefixes
            if prefix.axis_id!='compressed_key')) if header[7]=='candidate_upper_bound' else family
        for access in family.accesses:
            size=tile_template_requested_bytes(program,family,access)
            if access.operation is Operation.READ:read+=size
            else:write+=size
            if access.access_id=='attn_sink':sink_bytes+=size
            if access.access_id in metadata_bytes:metadata_bytes[access.access_id]+=size
            if access.access_id.startswith('map.'):
                if access.operation is Operation.READ:helper_read+=size
                else:helper_write+=size
                if access.access_id=='map.pages':conditional_pages+=size
            if access.access_id.endswith('.read'):partial_read+=size
            if access.access_id.endswith('.write'):partial_write+=size
            if access.access_id in ('indices','compressed'):
                candidate=tile_template_requested_bytes(program,envelope,access)
                budget_extra+=candidate-size
                if access.access_id=='indices':indices_bytes+=size;priced_indices_bytes+=candidate
            if access.access_id in ('swa_key_use','swa_value_use'):swa_bytes+=size
    tokens,heads,dim=parents[header[8]].shape;value_dim=parents[header[9]].shape[2]
    # Source requests keep both logical K/V uses and conditional page reads.
    # Pricing uses one shared packed record and only proven page metadata;
    # neither duplicate mathematical roles nor candidate padding are traffic.
    exact_pages = indices_bytes if ratio == 4 and header[7] in ('all_completed_states','declared_causal_topk') else 0
    return {'geometry':(tokens,heads,dim,value_dim,indices_bytes//4+swa_bytes//(2*584)),
        'read_bytes':read-sink_bytes+heads*4-swa_bytes//2-conditional_pages+exact_pages+partial_read,
        'write_bytes':write+partial_write,
        'logical_read_bytes':read,'logical_write_bytes':write,'active_selected_pairs':indices_bytes//4,
        'candidate_selected_pairs':priced_indices_bytes//4,'candidate_budget_extra_read_bytes':0,
        'unique_sink_read_bytes':heads*4,'decode_metadata_read_bytes':metadata_bytes,
        'c4_helper_read_upper_bound_bytes':helper_read,'c4_helper_write_bytes':helper_write,
        'conditional_c4_page_read_upper_bound_bytes':conditional_pages,
        'exact_c4_page_read_bytes':exact_pages if ratio != 4 or header[7] in ('all_completed_states','declared_causal_topk') else None,
        'logical_read_is_upper_bound':ratio==4}



def _v4_prefill_aux_bindings(kernel,binding):
    """Caller-declared helper buffers; known logical service stays unpriced."""
    tokens,heads,_,_,_,_,compressed,swa,_ = v4_sparse_geometry(kernel)
    capacity = kernel.attributes['topk_capacity'] if kernel.shape['compress_ratio'] > 1 else 0
    combined = -(-(capacity+kernel.shape['window_size'])//128)*128
    allocations,views,events = list(binding.allocations),list(binding.views),list(binding.events)
    resources,available = list(binding.resources),list(binding.availability)
    inputs = (('seq_lengths',(kernel.shape['batch'],)),('gather_lengths',(kernel.shape['batch'],)),
        ('query_offsets',(kernel.shape['batch']+1,)))
    for role,shape,output in (*( (role,shape,False) for role,shape in inputs),
            ('combined_indices',(tokens,combined),True),('combined_lengths',(tokens,),True),
            *((('compressed_lengths',(kernel.shape['batch'],),True),) if kernel.shape['compress_ratio'] > 1 else ())):
        identity = f'{binding.binding_id}.{role}';size = 4
        strides = []
        for extent in reversed(shape):strides.append(size);size *= extent
        view = TensorView(role,identity,0,shape,tuple(reversed(strides)),'int32',4)
        allocation = MemoryAllocation(identity,'logical-local-device','logical-v4-prefill-helper',size,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation);views.append(view);resources.append((role,identity))
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if not output:
            event = ProgramEvent(f'{identity}.available',EventKind.INITIALIZED,identity,view.region(),binding.visibility_domain,
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);available.append((role,EventRef(event.event_id,event.kind)))
    for role,count in (('compressed_pages',compressed),('swa_pages',swa)):
        if not count:continue
        identity = f'{binding.binding_id}.{role}'
        allocation = MemoryAllocation(identity,'logical-local-device','logical-page-selections',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation);resources.append((role,identity))
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        region = Unmapped((count,),'int32',4,f'{binding.binding_id}.{role}')
        event = ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,region,binding.visibility_domain,
            (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
        events.append(event);available.append((role,EventRef(event.event_id,event.kind)))
    # The pinned FlashMLA sparse_prefill API allocates these two independent
    # results even when the vLLM caller only retains out. They are not views
    # of out, nor initialized inputs to attention.
    for role in ('max_logits','lse'):
        identity = f'{binding.binding_id}.{role}'
        view = TensorView(role,identity,0,(tokens,heads),(heads*4,4),'fp32',4)
        allocation = MemoryAllocation(identity,'logical-local-device','logical-v4-prefill-result',tokens*heads*4,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation);views.append(view);resources.append((role,identity))
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
    return replace(binding,allocations=tuple(allocations),views=tuple(views),events=tuple(events),
        resources=tuple(resources),availability=tuple(available))


def _v4_prefill_header(kernel,binding):
    header = ('v4_prefill_tiles',','.join(map(str,kernel.attributes['q_lens'])),','.join(map(str,kernel.attributes['kv_lens'])),
        ','.join(map(str,kernel.attributes['selected_kv_lens'])),str(kernel.shape['compress_ratio']),
        str(kernel.shape['window_size']),str(kernel.attributes['topk_capacity']),kernel.attributes['compressed_selection'],
        binding.query_view,binding.output_view)
    return (*header,','.join(f'{start}:{count}' for start,count in binding.query_row_ranges)) if binding.query_row_ranges else header


def _v4_prefill_parameters(header):
    if len(header) not in (10,11) or header[0] != 'v4_prefill_tiles':raise ValueError('V4 prefill tile ABI missing')
    qlens,kvlens,selected = (tuple(map(int,text.split(','))) for text in header[1:4])
    ratio,window,capacity = map(int,header[4:7])
    if (not len(qlens) == len(kvlens) == len(selected) or min(ratio,window,capacity) <= 0
            or ratio not in (1,4,128)
            or any(not 0 < q <= kv for q,kv in zip(qlens,kvlens,strict=True))
            or header[7] not in ('candidate_upper_bound','all_completed_states','declared_causal_topk')):
        raise ValueError('V4 prefill request/selection geometry invalid')
    expected = tuple(0 if ratio == 1 else min(capacity,kv//ratio) for kv in kvlens)
    if (selected != expected or ratio == 1 and header[7] == 'all_completed_states'
            or ratio == 128 and capacity%128
            or (ratio == 128 or header[7] == 'all_completed_states') and max(kvlens)//ratio > capacity):
        raise ValueError('V4 prefill candidate envelope does not match its declared selection ABI')
    return qlens,kvlens,selected,ratio,window,capacity


def _v4_prefill_query_ranges(header,rows):
    lengths = tuple(map(int,header[1].split(',')))
    ranges = (tuple(tuple(map(int,pair.split(':'))) for pair in header[10].split(',')) if len(header) == 11 else
        tuple(zip((0,*accumulate(lengths[:-1])),lengths,strict=True)))
    if (len(ranges) != len(lengths) or any(len(pair) != 2 for pair in ranges)
            or any(first < 0 or count != q or first+count > rows for (first,count),q in zip(ranges,lengths,strict=True))
            or any(first+count > next_first for (first,count),(next_first,_) in zip(sorted(ranges),sorted(ranges)[1:]))
            or len(header) == 10 and rows != sum(lengths)):
        raise ValueError('V4 prefill query rows differ from actual producer request slices')
    return ranges


def _v4_prefill_roots(bindings,program_id):
    available = dict(bindings.availability);events = unique_by(bindings.events,'event_id');objects = dict(bindings.resources)
    roots = []
    for role in ('indices','compressed','swa','compressed_pages','swa_pages'):
        if role not in objects:continue
        event = events[available[role].node_id]
        if not isinstance(event.region,Unmapped) or event.allocation_id != objects[role]:raise ValueError('V4 source selection availability mismatch')
        region = event.region
        roots.append(LogicalSelectionView(f'{program_id}.{role}.root',objects[role],region.logical_shape,
            region.dtype,region.element_bytes,region.selection_id,None))
    return tuple(roots)


def _v4_prefill_recipe(header,parents,roots,ready,scope,visibility):
    qlens,kvlens,selected,ratio,window,capacity = _v4_prefill_parameters(header)
    query,output = parents[header[8]],parents[header[9]];heads = query.shape[1];packs = heads//64
    ranges = _v4_prefill_query_ranges(header,query.shape[0])
    if (heads%64 or query.shape[1:] != (heads,512) or output.shape != (sum(qlens),heads,512)
            or query.dtype not in ('bf16','fp16') or output.dtype != query.dtype
            or (query.element_bytes,output.element_bytes) != (2,2)):
        raise ValueError('V4 prefill requires its declared 512D two-byte 64-head packs')
    selected_views,families = [],[];token_start = 0
    evidence = ('vLLM v0.30 deepseek_v4/nvidia/flashmla.py _forward_prefill: compressed/SWA row gathers publish one shared BF16 workspace before sparse attention.',
        'Per-request logical workspace rows have exact compressed/SWA partition membership; dynamic selected row values and native chunk offsets remain unknown.',
        'cache_utils.combine_topk_swa_indices uses floor((position+1)/ratio), causal SWA, an aligned INT32 buffer filled -1, active updates and per-query lengths.',
        'One packed584B gather row and one512D BF16 workspace row per semantic tile; attention reuses selected rows per64-head pack, not a cache-hit claim.',
        'Helper producer/consumer roles and unique sink payload are priced separately from exported native FP32 max-logit/LSE and unknown gather-page services.',
        'Pinned FlashMLA SM90 sparse phase1 stores max_logits and lse per query/64-head CTA; these are the QK reduction before sink correction and do not depend on V or sink.',
        'V4 attention.py declares the actual sink parameter and nvidia/flashmla.py passes it to sparse prefill. Pinned SM90 phase1 applies sink correction only to final O; no native issue or scratch-layout equivalence claim.')
    for request,(queries,keys) in enumerate(zip(qlens,kvlens,strict=True)):
        identity = f'{scope.scope_id}.r{request}';comp_rows = keys//ratio if ratio > 1 else 0
        swa_rows = queries+min(keys-queries,window-1);union = comp_rows+swa_rows
        comp_capacity = capacity if header[7] == 'all_completed_states' else selected[request]
        comp_max = min(comp_capacity,keys//ratio) if ratio > 1 else 0;swa_max = min(window,keys)
        local_views = {};accesses = [];axes = [TileAxis('query',queries,1),TileAxis('pack',packs,1)]
        prefixes = []
        def add_view(view):selected_views.append(view);local_views[view.view_id] = view;return view
        def access(role,view,operation,iteration,mapping,dependencies=(),input_role=None):
            accesses.append(TileAccessTemplate(role,view.view_id,operation,iteration,mapping,
                (ready[input_role],) if input_role is not None else (),dependencies,evidence,
                unmapped=isinstance(view,LogicalSelectionView)))
        def dep(role,axes=(),kind=EventKind.READ_READY):return TileDependency(role,kind,axes)
        for role,parent in (('query',query),('output',output)):
            first = ranges[request][0] if role == 'query' else token_start
            view = add_view(TensorView(f'{identity}.{role}',parent.allocation_id,parent.offset_bytes+first*parent.strides_bytes[0],
                (queries,packs,64,512),(parent.strides_bytes[0],64*parent.strides_bytes[1],*parent.strides_bytes[1:]),parent.dtype,2))
            if role == 'query':access('query',view,Operation.READ,('query','pack'),('query','pack',None,None),input_role='query')
            else:out = view
        parent=parents['attn_sink']
        sink=add_view(TensorView(f'{identity}.attn_sink',parent.allocation_id,parent.offset_bytes,
            (packs,64),(64*4,4),'fp32',4))
        access('attn_sink',sink,Operation.READ,('query','pack'),('pack',None),input_role='attn_sink')
        for role,count,start in (('seq_lengths',1,request),('gather_lengths',1,request),('query_offsets',2,request)):
            parent = parents[role]
            view = add_view(TensorView(f'{identity}.{role}',parent.allocation_id,parent.offset_bytes+start*parent.strides_bytes[0],
                (count,),parent.strides_bytes,'int32',4))
            access(role,view,Operation.READ,(),(None,),input_role=role)
        parent = parents['query_offsets']
        base = add_view(LogicalSelectionView(f'{identity}.query_base',parent.allocation_id,(1,),'int32',4,
            f'{identity}.native_chunk_query_base',parent.view_id))
        access('query_base',base,Operation.READ,(),(None,),input_role='query_offsets')
        if ratio > 1:
            parent = parents['compressed_lengths']
            length = add_view(TensorView(f'{identity}.compressed_length',parent.allocation_id,
                parent.offset_bytes+request*parent.strides_bytes[0],(1,),parent.strides_bytes,'int32',4))
            access('compressed_lengths',length,Operation.WRITE,(),(None,),(dep('seq_lengths'),))
            access('read_compressed_lengths',length,Operation.READ,(),(None,),(dep('compressed_lengths',kind=EventKind.WRITE_VISIBLE),))
        workspace = add_view(LogicalSelectionView(f'{identity}.workspace',roots['workspace'].allocation_id,(union,512),'bf16',2,
            f'{roots["workspace"].selection_id}.request{request}',None))
        producer_deps = []
        for role,count,start in (('compressed',comp_rows,0),('swa',swa_rows,comp_rows)):
            if not count:continue
            axis = f'gather_{role}';axes.append(TileAxis(axis,count,1))
            root = roots[role];pages = roots[f'{role}_pages']
            page = add_view(LogicalSelectionView(f'{identity}.{role}_pages',pages.allocation_id,(count,),'int32',4,
                f'{pages.selection_id}.request{request}',pages.view_id))
            access(f'{role}_pages',page,Operation.READ,(axis,),(axis,),(dep('read_compressed_lengths') if role == 'compressed' else dep('seq_lengths'),),f'{role}_pages')
            packed = add_view(LogicalSelectionView(f'{identity}.{role}',root.allocation_id,(count,584),'uint8',1,
                f'{root.selection_id}.request{request}',root.view_id))
            access(role,packed,Operation.READ,(axis,),(axis,None),(dep(f'{role}_pages'),),role)
            written = add_view(LogicalSelectionView(f'{identity}.gather_{role}',workspace.allocation_id,(count,512),'bf16',2,
                f'{workspace.selection_id}.{role}',workspace.view_id,(
                    LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),start),LogicalCoordinateBinding(1,(LogicalCoordinateTerm(1,1),)))))
            access(f'gather_{role}',written,Operation.WRITE,(axis,),(axis,None),(dep(role),))
            producer_deps.append(dep(f'gather_{role}',(axis,),EventKind.WRITE_VISIBLE))
        combined_parent = parents['combined_indices'];length_parent = parents['combined_lengths']
        combined = add_view(TensorView(f'{identity}.combined',combined_parent.allocation_id,
            combined_parent.offset_bytes+token_start*combined_parent.strides_bytes[0],(queries,combined_parent.shape[1]),combined_parent.strides_bytes,'int32',4))
        length = add_view(TensorView(f'{identity}.lengths',length_parent.allocation_id,
            length_parent.offset_bytes+token_start*length_parent.strides_bytes[0],(queries,),length_parent.strides_bytes,'int32',4))
        access('combined_fill',combined,Operation.WRITE,('query',),('query',None))
        accesses[-1] = replace(accesses[-1],constant_value=ConstantValue.NEGATIVE_ONE)
        length_deps = [dep(role) for role in ('seq_lengths','gather_lengths','query_offsets','query_base')]
        output_deps = [dep('query'),dep('attn_sink')]
        qk_deps = [dep('query'),dep('read_lengths')]
        for role,count,divisor in (('comp',comp_max,ratio),('swa_selected',swa_max,1)):
            if not count:continue
            axis = f'key_{role}';axes.append(TileAxis(axis,count,1))
            prefixes.append(TileCausalPrefix(axis,'query',keys-queries+1,1,divisor))
            loop = ('query',axis);packed_loop = ('query','pack',axis)
            deps = [dep('combined_fill',kind=EventKind.WRITE_VISIBLE),*length_deps[:4]]
            if role == 'comp':
                root = roots['indices']
                indices = add_view(LogicalSelectionView(f'{identity}.indices',root.allocation_id,(queries,count),'int32',4,
                    f'{root.selection_id}.request{request}.causal_floor{ratio}',root.view_id))
                access('indices_input',indices,Operation.READ,loop,('query',axis),input_role='indices')
                deps.append(dep('indices_input'));length_deps.append(dep('indices_input',(axis,)))
            updated = add_view(LogicalSelectionView(f'{identity}.combined_{role}',combined.allocation_id,(queries,count),'int32',4,
                f'{identity}.combined.{role}.causal',combined.view_id))
            access(f'combine_{role}',updated,Operation.WRITE,loop,('query',axis),tuple(deps))
            access(f'indices_{role}',updated,Operation.READ,packed_loop,('query',axis),
                (dep(f'combine_{role}',kind=EventKind.WRITE_VISIBLE),))
            for operand in ('key','value'):
                view = add_view(LogicalSelectionView(f'{identity}.{operand}_{role}',workspace.allocation_id,(queries,count,512),'bf16',2,
                    f'{workspace.selection_id}.selected.{role}',workspace.view_id))
                access(f'{operand}_{role}',view,Operation.READ,packed_loop,('query',axis,None),
                    (*producer_deps,dep(f'indices_{role}'),dep('read_lengths')))
                output_deps.append(dep(f'{operand}_{role}',(axis,)))
                if operand == 'key':qk_deps.append(dep(f'{operand}_{role}',(axis,)))
        access('combined_lengths',length,Operation.WRITE,('query',),('query',),tuple(length_deps))
        access('read_lengths',length,Operation.READ,('query','pack'),('query',),(dep('combined_lengths',kind=EventKind.WRITE_VISIBLE),))
        for role in ('max_logits','lse'):
            parent = parents[role]
            view = add_view(TensorView(f'{identity}.{role}',parent.allocation_id,
                parent.offset_bytes+token_start*parent.strides_bytes[0],(queries,packs,64),
                (parent.strides_bytes[0],64*parent.strides_bytes[1],parent.strides_bytes[1]),'fp32',4))
            access(role,view,Operation.WRITE,('query','pack'),('query','pack',None),tuple(qk_deps))
        access('output',out,Operation.WRITE,('query','pack'),('query','pack',None,None),(*output_deps,dep('read_lengths')))
        local = replace(scope,scope_id=identity,op_ids=(*header,str(request)))
        families.append(TileProgram(identity,local,tuple(axes),tuple(accesses),visibility,evidence,tuple(prefixes)))
        token_start += queries
    return tuple(selected_views),tuple(families)


def _build_v4_prefill_tiles(kernel,hardware,bindings, *, program_id):
    tokens,heads,*_ = v4_sparse_geometry(kernel)
    if heads%64 or kernel.launch.grid[0] != tokens*(heads//64):
        raise ValueError('V4 sparse prefill launch disagrees with its actual 64-head pack count')
    regions = v4_regions(kernel,selection_id=bindings.binding_id);objects,available = dict(bindings.resources),dict(bindings.availability)
    auxiliary = {'seq_lengths','gather_lengths','query_offsets','combined_indices','combined_lengths','swa_pages','max_logits','lse'}
    if 'compressed' in regions:auxiliary.add('compressed_pages')
    if kernel.shape['compress_ratio'] > 1:auxiliary.add('compressed_lengths')
    needed = (set(regions)-{'key_role','value_role'})|auxiliary
    inputs = (needed-{'workspace','combined_indices','combined_lengths','compressed_lengths','max_logits','lse'})
    if (len(objects) != len(bindings.resources) or len(available) != len(bindings.availability)
            or set(objects) != needed or set(available) != inputs or len(set(objects.values())) != len(objects)
            or bindings.cache != objects['swa'] or bindings.indices != (objects['indices'] if 'indices' in objects else None)):
        raise ValueError('V4 prefill requires exact cache/helper/temporary parents and availability')
    parents = unique_by(bindings.views,'view_id')
    if set(parents) != ({bindings.query_view,bindings.output_view,'attn_sink','seq_lengths','gather_lengths','query_offsets','combined_indices','combined_lengths','max_logits','lse'}
            | ({'compressed_lengths'} if kernel.shape['compress_ratio'] > 1 else set())):
        raise ValueError('V4 prefill actual query/output/helper tensor views missing')
    roots = _v4_prefill_roots(bindings,program_id)
    workspace = regions['workspace']
    roots = (*roots,LogicalSelectionView(f'{program_id}.workspace.root',objects['workspace'],workspace.logical_shape,
        workspace.dtype,workspace.element_bytes,workspace.selection_id,None))
    root_by_role = {view.view_id[len(program_id)+1:-5]:view for view in roots}
    header = _v4_prefill_header(kernel,bindings)
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-v4-prefill-rows')
    selected,families = _v4_prefill_recipe(header,parents,root_by_role,
        {'query':bindings.query_ready,'attn_sink':bindings.input_ready[0],**available},scope,bindings.visibility_domain)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Per-request gather row partitions publish the actual workspace parent; selected attention requires its complete gather producers and actual combined-index writes.',
            'Integer causal prefixes retain compression boundaries and SWA tails without per-query DAG materialization.',
            'Page/slot/index values, native prefill chunk offsets/capacities and exact physical selections remain unresolved.',
            'Combined-index fill, combine and consumer roles are priced; unknown gather-page services remain unpriced.',
            'Pricing uses active causal floor masks rather than final-request candidate padding; no observed DRAM or cache-residency claim.',
            'The actual FP32 sink parameter gates final O and contributes its unique input payload. Pinned FlashMLA sparse_prefill API and SM90 phase1 bind separate FP32 max-logit/LSE result writes, also unpriced.')),
        bindings.allocations,(*bindings.views,*roots,*selected),(scope,),bindings.events,(),families)
    v4_prefill_tile_demands(program)
    return program


def v4_prefill_tile_demands(program, *, expected_kernel=None,expected_bindings=None):
    from src.kernel.memory_access import tile_template_requested_bytes
    program.__post_init__()
    if program.accesses or len(program.scopes) != 1 or not program.tile_programs:
        raise ValueError('V4 prefill requires its unique row/query tile source')
    scope, = program.scopes;header = scope.op_ids
    qlens,kvlens,selected,ratio,window,capacity = _v4_prefill_parameters(header)
    views = unique_by(program.views,'view_id')
    if expected_kernel is not None:
        kernel = expected_kernel
        if (kernel.kind is not KernelKind.SPARSE_ATTN or kernel.variant != 'dsv4_sparse_prefill'
                or kernel.execution_support != 'supported' or kernel.hardware_id != program.hardware_id
                or kernel.dtype not in ('bf16','fp16') or views[header[8]].dtype != kernel.dtype
                or kernel.shape['qk_head_dim'] != 512 or kernel.shape['v_head_dim'] != 512
                or (qlens,kvlens,selected,ratio,window,capacity) != (
                    tuple(kernel.attributes['q_lens']),tuple(kernel.attributes['kv_lens']),
                    tuple(kernel.attributes['selected_kv_lens']),kernel.shape['compress_ratio'],
                    kernel.shape['window_size'],kernel.attributes['topk_capacity'])
                or header[7] != kernel.attributes['compressed_selection']
                or views[header[8]].shape[1:] != (kernel.shape['num_query_heads'],512)):
            raise ValueError('Source differs from its actual V4 prefill invocation')
    if expected_bindings is not None:
        b = expected_bindings
        if (not isinstance(b,SparseAttentionBindings) or program.binding_id != b.binding_id
                or (scope.call_id,scope.rank,scope.stream) != (b.call_id,b.rank,b.stream)
                or program.allocations != b.allocations or program.events != b.events
                or program.views[:len(b.views)] != b.views or header[8:10] != (b.query_view,b.output_view)):
            raise ValueError('Source differs from its actual V4 prefill caller bindings')
        if (header[10:] != ((','.join(f'{start}:{count}' for start,count in b.query_row_ranges),) if b.query_row_ranges else ())
                or any(f.visibility_domain != b.visibility_domain for f in program.tile_programs)):
            raise ValueError('Source differs from its actual V4 prefill caller bindings')
    parent_ids = (header[8],header[9],'attn_sink','seq_lengths','gather_lengths','query_offsets','combined_indices','combined_lengths','max_logits','lse',
        *(('compressed_lengths',) if ratio > 1 else ()))
    parents = {name:views[name] for name in parent_ids}
    roots = {role:views[f'{scope.scope_id}.{role}.root'] for role in ('indices','compressed','swa','workspace','compressed_pages','swa_pages')
        if f'{scope.scope_id}.{role}.root' in views}
    _,heads,dimension = parents[header[8]].shape;tokens = sum(qlens)
    sink=parents['attn_sink']
    if (sink.shape,sink.dtype,sink.element_bytes,sink.strides_bytes)!=((heads,),'fp32',4,(4,)):
        raise ValueError('V4 prefill requires its actual padded FP32 sink')
    for role in ('max_logits','lse'):
        parent = parents[role]
        if (not isinstance(parent,TensorView) or
                (parent.shape,parent.dtype,parent.element_bytes,parent.strides_bytes) != ((tokens,heads),'fp32',4,(heads*4,4))):
            raise ValueError('V4 prefill requires independent dense native FP32 auxiliary outputs')
    compressed = sum(kv//ratio for kv in kvlens) if ratio > 1 else 0
    swa = sum(q+min(kv-q,window-1) for q,kv in zip(qlens,kvlens,strict=True))
    combined = -(-((capacity if ratio > 1 else 0)+window)//128)*128
    for name,shape in (('seq_lengths',(len(qlens),)),('gather_lengths',(len(qlens),)),('query_offsets',(len(qlens)+1,)),
            ('combined_indices',(tokens,combined)),('combined_lengths',(tokens,)),
            *((('compressed_lengths',(len(qlens),)),) if ratio > 1 else ())):
        view = parents[name]
        if not isinstance(view,TensorView) or (view.shape,view.dtype,view.element_bytes) != (shape,'int32',4):
            raise ValueError('V4 prefill helper integer ABI mismatch')
        if (view.strides_bytes[-1] != 4 or name == 'combined_indices' and view.strides_bytes[0] < combined*4):
            raise ValueError('V4 prefill helper pointers require their native dense scalar indexing')
    expected_roots = {'swa':((swa,584),'uint8',1),'swa_pages':((swa,),'int32',4),
        'workspace':((compressed+swa,512),'bf16',2)}
    if compressed:expected_roots.update({'compressed':((compressed,584),'uint8',1),'compressed_pages':((compressed,),'int32',4)})
    if sum(q*s for q,s in zip(qlens,selected,strict=True)):
        expected_roots['indices'] = (roots['indices'].shape,'int32',4)
    if set(roots) != set(expected_roots):raise ValueError('V4 prefill cache/metadata/gather roles missing')
    if expected_bindings is not None:
        resources = dict(b.resources)
        if (len(resources) != len(b.resources) or any(role not in resources or view.allocation_id != resources[role]
                for role,view in roots.items()) or b.cache != resources['swa']
                or b.indices != (resources['indices'] if 'indices' in resources else None)):
            raise ValueError('Source differs from its actual V4 prefill caller resources')
    for role,(shape,dtype,width) in expected_roots.items():
        view = roots[role]
        if (not isinstance(view,LogicalSelectionView) or view.parent_view_id is not None
                or (view.shape,view.dtype,view.element_bytes) != (shape,dtype,width)):
            raise ValueError('V4 prefill typed cache/page/workspace root mismatch')
    all_parents = (*parents.values(),*roots.values())
    if len({view.allocation_id for view in all_parents}) != len(all_parents):
        raise ValueError('V4 query/helper/cache/workspace parents require distinct storage')
    ready = {}
    external = {'query','attn_sink','seq_lengths','gather_lengths','query_offsets','compressed','swa','compressed_pages','swa_pages','indices_input','query_base'}
    for family in program.tile_programs:
        for access in family.accesses:
            if access.access_id not in external:continue
            role = 'indices' if access.access_id == 'indices_input' else 'query_offsets' if access.access_id == 'query_base' else access.access_id
            if len(access.input_ready) != 1 or role in ready and ready[role] != access.input_ready[0]:
                raise ValueError('V4 prefill input availability differs between tiles')
            ready[role] = access.input_ready[0]
    if expected_bindings is not None:
        if len(b.input_ready) != 1:raise ValueError('Actual V4 prefill caller requires its sink readiness')
        actual_ready = {'query':b.query_ready,'attn_sink':b.input_ready[0],**dict(b.availability)}
        if any(role not in actual_ready or ref != actual_ready[role] for role,ref in ready.items()):
            raise ValueError('Source differs from its actual V4 prefill caller readiness')
    selected_views,expected = _v4_prefill_recipe(header,parents,roots,ready,scope,program.tile_programs[0].visibility_domain)
    if (program.tile_programs != expected or set(views) != {*parent_ids,*(view.view_id for view in roots.values()),*(view.view_id for view in selected_views)}
            or any(views[view.view_id] != view for view in selected_views)):
        raise ValueError('V4 prefill lost actual gather/index/query causality or request membership')
    read = write = logical_read = logical_write = actual_selected = swa_pairs = sink_bytes = aux_bytes = 0
    for family in program.tile_programs:
        for access in family.accesses:
            value = tile_template_requested_bytes(program,family,access)
            if access.operation is Operation.READ:logical_read += value
            else:logical_write += value
            if access.access_id == 'attn_sink':sink_bytes += value
            if access.access_id in ('max_logits','lse'):aux_bytes += value
            if access.access_id in ('query','compressed','swa','indices_input') or access.access_id.startswith(('key_','value_')):read += value
            if access.access_id in ('gather_compressed','gather_swa','output'):write += value
            if access.access_id == 'indices_input':actual_selected += value//4
            if access.access_id == 'key_swa_selected':swa_pairs += value//(512*2*(heads//64))
    # The validated source owns operand identity and causal membership.
    # Project its active roles at the current operator's stated granularity:
    # helper scalar uses repeat over 256 workers, consumer indices are one
    # logical input role, sink is one unique parameter, auxiliary outputs
    # and unobserved gather-page services remain separately unpriced.
    combined_capacity = -(-((capacity if ratio > 1 else 0)+window)//128)*128
    entries = actual_selected + swa_pairs
    helper_reads = len(qlens)*256*5*4 + 4*(entries+tokens)
    helper_writes = 4*(tokens*combined_capacity+entries+tokens)
    return {'geometry':(tokens,heads,dimension,dimension,entries),
        'read_bytes':read+helper_reads+heads*4,'write_bytes':write+helper_writes,
        'logical_read_bytes':logical_read,'logical_write_bytes':logical_write,
        'index_helper_read_bytes':helper_reads,'index_helper_write_bytes':helper_writes,
        'unique_sink_read_bytes':heads*4,'unpriced_aux_write_bytes':aux_bytes,
        'candidate_envelope_extra_read_bytes':0,'active_compressed_pairs':actual_selected,'swa_pairs':swa_pairs}
