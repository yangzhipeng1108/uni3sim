"""Sparse indexer inputs, opaque cache selections, and score/top-k dataflow."""
from dataclasses import dataclass,replace
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity, MemoryAccess,
    MemoryAccessProgram, MemoryAllocation, Operation, Ordering, ProgramEvent, Semantics,
    TensorView, Typed, Unmapped, unique_by, LogicalSelectionView,
    LogicalCoordinateBinding,LogicalCoordinateTerm,
    TileAxis,TileAccessTemplate,TileDependency,TileProgram,TileCausalPrefix,
)


@dataclass(frozen=True)
class SparseIndexerBindings(Typed):
    binding_id: str
    call_id: str
    allocations: tuple[MemoryAllocation, ...]
    views: tuple[TensorView | LogicalSelectionView, ...]
    events: tuple[ProgramEvent, ...]
    query_view: str | None
    weight_view: str | None
    indices_view: str | None
    key_cache: str | None
    logits: str | None
    query_ready: EventRef | None
    weight_ready: EventRef | None
    key_ready: tuple[EventRef, ...]
    rank: int
    stream: str
    visibility_domain: str


V4_INDEXER_SOURCE_GAP = (
    "V4 indexer memory source is partial: actual score backing/strides, clear-to-selection "
    "output alias, prefill gather storage and selector row metadata are unbound; "
    "the useful staged Roofline remains available without memory bindings")


def has_sparse_indexer_source(kernel):
    return (kernel.kind is KernelKind.SPARSE_ATTN_INDEXER and kernel.execution_support == 'supported'
            and kernel.variant in ('dsa_indexer_prefill','dsa_indexer_decode','dsa_indexer_mixed','dsa_index_reuse'))


def indexer_geometry(kernel):
    query_lengths,cache_lengths = kernel.attributes['q_lens'],kernel.attributes['kv_lens']
    ratio = kernel.shape['compress_ratio'] if kernel.variant.startswith('dsv4_') else 1
    # Sum completed states without materializing one integer per query.
    def prefix(total):
        quotient,remainder = divmod(total,ratio)
        return ratio*quotient*(quotient-1)//2+quotient*(remainder+1)
    pairs = sum(prefix(kv)-prefix(kv-q) for q,kv in zip(query_lengths,cache_lengths,strict=True))
    return sum(query_lengths),kernel.shape['num_query_heads'],kernel.shape['head_dim'],pairs,tuple(zip(query_lengths,cache_lengths,strict=True))


def cache_regions(kernel, *, selection_id):
    _,_,dimension,pairs,_ = indexer_geometry(kernel)
    if not pairs:
        return ()
    selector = f"{selection_id}.q{','.join(map(str,kernel.attributes['q_lens']))}.kv{','.join(map(str,kernel.attributes['kv_lens']))}.c{kernel.shape['compress_ratio'] if kernel.variant.startswith('dsv4_') else 1}"
    return (Unmapped((pairs,dimension),'fp8',1,f'{selector}.key_payload'),
            Unmapped((pairs,-(-dimension//128)),'fp32',4,f'{selector}.key_scale'))


def standalone_sparse_indexer_bindings(kernel, *, call_id, binding_id):
    if kernel.variant.startswith('dsv4_'):
        raise NotImplementedError(V4_INDEXER_SOURCE_GAP)
    if kernel.variant == 'dsa_index_reuse':
        return SparseIndexerBindings(binding_id,call_id,(),(),(),None,None,None,None,None,None,None,(),0,'offline','indexer-visible')
    tokens,heads,dimension,pairs,_ = indexer_geometry(kernel)
    capacity = kernel.attributes['topk_capacity']
    layouts = (('query',(tokens,heads,dimension),'fp8',1),('weight',(tokens,heads),'fp32',4),
               ('indices',(tokens,capacity),'int32',4))
    allocations,views,events,ready = [],[],[],[]
    for name,shape,dtype,width in layouts:
        identity = f'{binding_id}.{name}'
        strides,size = [],width
        for axis in reversed(shape):
            strides.append(size)
            size *= axis
        view = TensorView(name,identity,0,shape,tuple(reversed(strides)),dtype,width)
        allocation = MemoryAllocation(identity,'logical-local-device','logical-storage',size,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation)
        views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if name != 'indices':
            initialized = f'{identity}.initialized'
            events.append(ProgramEvent(initialized,EventKind.INITIALIZED,identity,view.region(),'indexer-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),)))
            ready.append(EventRef(initialized,EventKind.INITIALIZED))
    cache_id,logits_id,key_ready = None,None,[]
    if pairs:
        for name in ('cache','logits'):
            identity = f'{binding_id}.{name}'
            allocations.append(MemoryAllocation(identity,'logical-local-device','logical-indexer-storage',None,
                AddressLevel.OBJECT,None,None,f'{identity}.allocated',None))
            events.append(ProgramEvent(f'{identity}.allocated',EventKind.ALLOCATED,identity,None,None,()))
        cache_id,logits_id = allocations[-2].allocation_id,allocations[-1].allocation_id
        for index,region in enumerate(cache_regions(kernel,selection_id=binding_id)):
            event = ProgramEvent(f'{cache_id}.available{index}',EventKind.LOGICAL_AVAILABLE,cache_id,region,'indexer-visible',
                (EventRef(allocations[-2].allocated_event,EventKind.ALLOCATED),))
            events.append(event)
            key_ready.append(EventRef(event.event_id,EventKind.LOGICAL_AVAILABLE))
    return SparseIndexerBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        'query','weight','indices',cache_id,logits_id,*ready,tuple(key_ready),0,'offline','indexer-visible')


def build_sparse_indexer_source(kernel, hardware, bindings, granularity, *, program_id):
    if kernel.variant.startswith('dsv4_'):
        raise NotImplementedError(V4_INDEXER_SOURCE_GAP)
    if not isinstance(bindings,SparseIndexerBindings):
        raise TypeError('sparse indexer requires typed query/weight/cache/logit/indices bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('unknown sparse indexer logical granularity')
    alias = kernel.variant == 'dsa_index_reuse'
    input_roles = ('sparse_indexer','alias') if alias else ('sparse_indexer','inputs',bindings.query_view,bindings.weight_view)
    scopes = [AccessScope(f'{program_id}.scope',input_roles,bindings.call_id,(),
        bindings.rank,bindings.stream,'logical-sparse-indexer')]
    limits = ('Cache block/page transforms, unique footprint, actual allocator capacities and logits strides are unavailable.',
        'FP8 query scale is folded into FP32 head weights; FP8 keys have separately represented FP32-storage scales.',
        'DSA per-head key repeat is the retained logical work projection, not distinct key storage or proven physical read replication.',
        'Valid causal logits only; padded/invalid logit writes, top-k schedule, prefill gather/packing and cache insertion side effects remain unbound.',
        'Native output top-k capacity is stored including sentinel positions; effective top-k is not its allocation/write extent.',
        'Alias declares no memory operation; absent caller producer identity stays unknown rather than inventing a copied buffer.')
    if alias:
        if (bindings.query_view is not None or bindings.weight_view is not None or bindings.key_cache is not None
                or bindings.logits is not None or bindings.query_ready is not None or bindings.weight_ready is not None or bindings.key_ready):
            raise ValueError('index reuse must not declare a score/query/cache operation')
        if bindings.indices_view is not None:
            view = unique_by(bindings.views,'view_id')[bindings.indices_view]
            if view.dtype != 'int32' or view.element_bytes != 4 or len(view.shape) != 2 or view.shape[1] != kernel.attributes['topk_capacity']:
                raise ValueError('index reuse must alias its actual int32 producer buffer capacity')
        elif bindings.allocations or bindings.views or bindings.events:
            raise ValueError('unknown index reuse cannot invent storage or lifecycle events')
        return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
            Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.UNORDERED,False,limits),
            bindings.allocations,bindings.views,tuple(scopes),bindings.events,())
    tokens,heads,dimension,pairs,_ = indexer_geometry(kernel)
    views = unique_by(bindings.views,'view_id')
    query,weight,indices = (views[name] for name in (bindings.query_view,bindings.weight_view,bindings.indices_view))
    for view,shape,dtype,width in ((query,(tokens,heads,dimension),'fp8',1),(weight,(tokens,heads),'fp32',4),
                                  (indices,(tokens,kernel.attributes['topk_capacity']),'int32',4)):
        if (view.shape,view.dtype,view.element_bytes) != (shape,dtype,width):
            raise ValueError('sparse indexer operand shape/storage mismatch')
    if len({view.allocation_id for view in (query,weight,indices)}) != 3:
        raise NotImplementedError('sparse query/weights/indices alias requires an explicit lifecycle')
    if pairs:
        if bindings.key_cache is None or bindings.logits is None or len(bindings.key_ready) != 2:
            raise ValueError('sparse indexer cache/score availability missing')
        if {bindings.key_cache,bindings.logits} & {view.allocation_id for view in (query,weight,indices)} or bindings.key_cache == bindings.logits:
            raise NotImplementedError('sparse indexer cache/logits overlap needs a source lifecycle')
    elif bindings.key_cache is not None or bindings.logits is not None or bindings.key_ready:
        raise ValueError('empty compressed prefix must not fabricate cache/score work')
    ratio = kernel.shape['compress_ratio'] if kernel.variant.startswith('dsv4_') else 1
    header = ('sparse_indexer_tiles',kernel.variant,str(ratio),
        ','.join(map(str,kernel.attributes['q_lens'])),','.join(map(str,kernel.attributes['kv_lens'])),
        bindings.query_view,bindings.weight_view,bindings.indices_view)
    scope = replace(scopes[0],scope_id=program_id,op_ids=header,agent='logical-sparse-indexer-queries')
    roots = ()
    if pairs:
        roots = tuple(LogicalSelectionView(f'{program_id}.{role}.root',bindings.key_cache,region.logical_shape,
            region.dtype,region.element_bytes,region.selection_id,None) for role,region in zip(
                ('key','key_scale'),cache_regions(kernel,selection_id=bindings.binding_id),strict=True))
        roots += (LogicalSelectionView(f'{program_id}.logits.root',bindings.logits,(pairs,),'fp32',4,
            f'{bindings.binding_id}.valid_logits',None),)
    selected,families = _indexer_tile_recipe(header,(query,weight,indices),roots,
        (bindings.query_ready,bindings.weight_ready,*bindings.key_ready),scope,bindings.visibility_domain)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,limits),bindings.allocations,
        (*bindings.views,*roots,*selected),(scope,),bindings.events,(),families)
    sparse_indexer_tile_demands(program)
    return program


def _indexer_query_groups(queries,keys,ratio):
    """At most four rectangular groups share floor-prefix geometry."""
    full,tail = divmod(queries,ratio)
    split = ratio-(keys-queries+1)%ratio
    groups = []
    for first,count,width in ((0,full,ratio),(full,1,tail)):
        if not count or not width:continue
        middle = min(split,width)
        for start,end in ((0,middle),(middle,width)):
            if start < end:groups.append((first,count,start,end-start))
    return tuple(groups)


def _indexer_tile_recipe(header,parents,roots,ready,scope,visibility):
    if len(header) != 8 or header[0] != 'sparse_indexer_tiles':raise ValueError('Sparse indexer tile ABI missing')
    if header[1].startswith('dsv4_'):
        raise NotImplementedError(V4_INDEXER_SOURCE_GAP)
    variant,ratio = header[1],int(header[2]);q_lens,kv_lens = (tuple(map(int,text.split(','))) for text in header[3:5])
    if variant not in ('dsa_indexer_prefill','dsa_indexer_decode','dsa_indexer_mixed','dsv4_indexer_prefill','dsv4_indexer_decode'):
        raise ValueError('Unknown sparse indexer tile variant')
    if (not variant.startswith('dsv4_') and ratio != 1):raise ValueError('DSA indexer must retain uncompressed candidates')
    if ratio <= 0 or len(q_lens) != len(kv_lens) or any(not 0 < q <= kv for q,kv in zip(q_lens,kv_lens,strict=True)):
        raise ValueError('Sparse indexer causal request geometry invalid')
    query,weight,indices = parents;tokens,heads,dimension = query.shape
    if (tuple(view.view_id for view in parents) != header[5:] or tokens != sum(q_lens)
            or (query.dtype,query.element_bytes) != ('fp8',1)
            or (weight.shape,weight.dtype,weight.element_bytes) != ((tokens,heads),'fp32',4)
            or indices.shape[0] != tokens or (indices.dtype,indices.element_bytes) != ('int32',4)):
        raise ValueError('Sparse indexer query/head weight/output capacity ABI mismatch')
    evidence = ('FP8 weighted-head scorer materializes one FP32 logit per completed causal candidate; top-k reloads its own row.',
        'Completed compressed states=floor((KV-Q+query+1)/ratio); split query blocks and at most two lane groups per full/tail block give bounded ordinary causal streams.',
        'DSA head replay and V4 shared-key requests preserve the existing logical interface policy; neither asserts issued transactions or cache hits.',
        'Every query publishes the full INT32 top-k capacity, including sentinel positions; native cache and logits coordinates remain unknown.')
    selected,families = [],[];offset = 0
    for request,(queries,keys) in enumerate(zip(q_lens,kv_lens,strict=True)):
        for first,count,residue,lanes in _indexer_query_groups(queries,keys,ratio):
            maximum = (keys-queries+1+residue+(first+count-1)*ratio)//ratio
            identity = f'{scope.scope_id}.r{request}.b{first}.lane{residue}'
            children = []
            for role,parent in zip(('query','weight','indices'),parents,strict=True):
                if isinstance(parent,LogicalSelectionView):
                    if role!='indices':raise ValueError('Only a caller-bound top-k buffer may use an unresolved indexer output selection')
                    view=LogicalSelectionView(f'{identity}.{role}',parent.allocation_id,(count,lanes,*parent.shape[1:]),
                        parent.dtype,parent.element_bytes,f'{parent.selection_id}.r{request}.b{first}.lane{residue}',parent.view_id,
                        (LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,ratio),LogicalCoordinateTerm(1,1)),
                            offset=offset+first*ratio+residue),LogicalCoordinateBinding(1,(LogicalCoordinateTerm(2,1),))))
                else:
                    view = TensorView(f'{identity}.{role}',parent.allocation_id,
                        parent.offset_bytes+(offset+first*ratio+residue)*parent.strides_bytes[0],(count,lanes,*parent.shape[1:]),
                        (parent.strides_bytes[0]*ratio,*parent.strides_bytes),parent.dtype,parent.element_bytes)
                selected.append(view);children.append(view)
            q,w,out = children;accesses = []
            axes = (TileAxis('query',count,1),TileAxis('lane',lanes,1))
            # The retained ABI reads the complete Q/weight inputs whenever
            # any request has score work. Preserve this boundary projection.
            if roots:
                accesses.extend((TileAccessTemplate('query',q.view_id,Operation.READ,('query','lane'),('query','lane',None,None),
                    (ready[0],),(),evidence),TileAccessTemplate('weight',w.view_id,Operation.READ,('query','lane'),('query','lane',None),
                    (ready[1],),(),evidence)))
            prefixes = ()
            if maximum:
                axes += (TileAxis('key',maximum,1),TileAxis('head',heads,1))
                prefixes = (TileCausalPrefix('key','query'),)
                for index,(role,root) in enumerate(zip(('key','key_scale','logits'),roots,strict=True)):
                    shape = (count,lanes,maximum,*root.shape[1:])
                    view = LogicalSelectionView(f'{identity}.{role}',root.allocation_id,shape,root.dtype,root.element_bytes,
                        f'{root.selection_id}.request{request}.residue{residue}.ratio{ratio}',root.view_id)
                    selected.append(view)
                    if role == 'logits':logits = view;continue
                    loop = ('query','lane','key') if variant.startswith('dsv4_') else ('query','lane','key','head')
                    accesses.append(TileAccessTemplate(role,view.view_id,Operation.READ,loop,('query','lane','key',None),
                        (ready[index+2],),(),evidence,unmapped=True))
                head_reduction = () if variant.startswith('dsv4_') else ('head',)
                deps = (TileDependency('query',EventKind.READ_READY,()),TileDependency('weight',EventKind.READ_READY,()),
                    TileDependency('key',EventKind.READ_READY,head_reduction),TileDependency('key_scale',EventKind.READ_READY,head_reduction))
                accesses.append(TileAccessTemplate('score',logits.view_id,Operation.WRITE,('query','lane','key'),('query','lane','key'),(),deps,evidence,unmapped=True))
                accesses.append(TileAccessTemplate('topk',logits.view_id,Operation.READ,('query','lane','key'),('query','lane','key'),(),
                    (TileDependency('score',EventKind.WRITE_VISIBLE,()),),evidence,unmapped=True))
            deps = (TileDependency('topk',EventKind.READ_READY,('key',)),) if maximum else ()
            accesses.append(TileAccessTemplate('indices',out.view_id,Operation.WRITE,('query','lane'),('query','lane',None),(),deps,evidence,
                unmapped=isinstance(out,LogicalSelectionView)))
            local = replace(scope,scope_id=identity,op_ids=(*header,str(request),str(first),str(residue)))
            families.append(TileProgram(identity,local,axes,tuple(accesses),visibility,evidence,causal_prefixes=prefixes))
        offset += queries
    return tuple(selected),tuple(families)


def sparse_indexer_tile_demands(program):
    from src.kernel.memory_access import tile_template_requested_bytes
    program.__post_init__()
    if not program.tile_programs or program.accesses or len(program.scopes) != 1:
        raise ValueError('Sparse indexer requires its unique causal tile source')
    scope, = program.scopes;header = scope.op_ids;views = unique_by(program.views,'view_id')
    parents = tuple(views[name] for name in header[5:8])
    roots = tuple(view for view in program.views if view.view_id in tuple(f'{scope.scope_id}.{role}.root' for role in ('key','key_scale','logits')))
    if roots and len(roots) != 3:raise ValueError('Sparse indexer payload/scale/logits root missing')
    first = unique_by(program.tile_programs[0].accesses,'access_id')
    ready = ()
    if roots:
        ready = (first['query'].input_ready[0],first['weight'].input_ready[0])
        key_ready = {}
        for family in program.tile_programs:
            for access in family.accesses:
                if access.access_id not in ('key','key_scale'):continue
                key_ready[access.access_id] = access.input_ready[0]
        ready += (key_ready['key'],key_ready['key_scale'])
    selected,expected = _indexer_tile_recipe(header,parents,roots,ready,scope,program.tile_programs[0].visibility_domain)
    if (program.tile_programs != expected or set(views) != {view.view_id for view in (*parents,*roots,*selected)}
            or any(views[view.view_id] != view for view in selected)):
        raise ValueError('Sparse indexer lacks actual score/input producer causality or exact query membership')
    totals = {Operation.READ:0,Operation.WRITE:0};pairs = 0
    for family in program.tile_programs:
        for template in family.accesses:
            amount = tile_template_requested_bytes(program,family,template)
            totals[template.operation] += amount
            if template.access_id == 'score':pairs += amount//4
    tokens,heads,dimension = parents[0].shape;capacity = parents[2].shape[1]
    if roots:
        selector = f'{program.binding_id}.q{header[3]}.kv{header[4]}.c{header[2]}'
        if (tuple(root.selection_id for root in roots) != (f'{selector}.key_payload',f'{selector}.key_scale',f'{program.binding_id}.valid_logits')
                or any(root.parent_view_id is not None for root in roots)):
            raise ValueError('Sparse indexer requires its exact available selection roots')
        if (roots[0].shape,roots[1].shape,roots[2].shape) != ((pairs,dimension),(pairs,-(-dimension//128)),(pairs,)):
            raise ValueError('Sparse indexer completed states/payload/FP32 scale recipe mismatch')
        if (roots[0].dtype,roots[0].element_bytes,roots[1].dtype,roots[1].element_bytes,roots[2].dtype,roots[2].element_bytes) != ('fp8',1,'fp32',4,'fp32',4):
            raise ValueError('Sparse indexer typed payload/scale/logits ABI mismatch')
        if roots[0].allocation_id != roots[1].allocation_id or len({roots[0].allocation_id,roots[2].allocation_id,*(view.allocation_id for view in parents)}) != 5:
            raise ValueError('Sparse indexer cache/logits/operand identity mismatch')
    return {'alias':False,'geometry':(tokens,heads,dimension,pairs,capacity),
        'read_bytes':totals[Operation.READ],'write_bytes':totals[Operation.WRITE]}


def validate_indexer_invocation(program,kernel):
    """Equal byte totals cannot substitute a different request/candidate domain."""
    if (not has_sparse_indexer_source(kernel) or kernel.hardware_id != program.hardware_id
            or kernel.dtype != 'int32' or len(program.scopes) != 1):
        raise ValueError('Source differs from its actual indexer invocation')
    scope, = program.scopes
    if kernel.variant == 'dsa_index_reuse':
        if scope.op_ids != ('sparse_indexer','alias') or program.accesses or program.tile_programs:
            raise ValueError('Source differs from its actual indexer invocation')
        if program.views:
            if len(program.views) != 1 or len(program.allocations) != 1:
                raise ValueError('Index alias must retain only its actual producer storage')
            view, = program.views
            if (not isinstance(view,(TensorView,LogicalSelectionView)) or view.dtype != 'int32' or view.element_bytes != 4
                    or len(view.shape) != 2 or view.shape[1] != kernel.attributes['topk_capacity']):
                raise ValueError('Source differs from its actual indexer invocation')
        elif program.allocations or program.events:
            raise ValueError('Unknown index alias cannot invent storage')
        return
    header = scope.op_ids
    ratio = kernel.shape['compress_ratio'] if kernel.variant.startswith('dsv4_') else 1
    expected = ('sparse_indexer_tiles',kernel.variant,str(ratio),
        ','.join(map(str,kernel.attributes['q_lens'])),','.join(map(str,kernel.attributes['kv_lens'])))
    if len(header) != 8 or header[:5] != expected:
        raise ValueError('Source differs from its actual indexer invocation')
    views = unique_by(program.views,'view_id')
    query,weight,indices = (views[name] for name in header[5:])
    tokens = sum(kernel.attributes['q_lens'])
    if ((query.shape,weight.shape,indices.shape) != (
            (tokens,kernel.shape['num_query_heads'],kernel.shape['head_dim']),
            (tokens,kernel.shape['num_query_heads']),(tokens,kernel.attributes['topk_capacity']))):
        raise ValueError('Source differs from its actual indexer invocation')
