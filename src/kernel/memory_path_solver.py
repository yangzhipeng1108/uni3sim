"""Hierarchical byte/request capacity and finite-source memory solver."""

import math


def logical_byte_copy_payload(program) -> int:
    """Project byte-copy storage work to one link/DMA payload, not residency.

    The copy ABI owns one source read and one destination write. Its storage
    ledger counts both; a PCIe/NVLink payload is transferred once. No cache
    traffic or physical request geometry is inferred from logical addresses.
    """
    from src.kernel.memory_access import (
        EventKind, Granularity, MemoryAccessProgram, Operation, Semantics,
    )
    if not isinstance(program, MemoryAccessProgram):
        raise TypeError("byte-copy projection requires MemoryAccessProgram")
    if program.coverage.semantics is not Semantics.LOGICAL:
        raise ValueError("byte-copy payload projection requires logical semantics")
    program.__post_init__()
    if program.accesses:
        raise ValueError('byte-copy projection requires its unique independent-byte tile source')
    if program.tile_programs:
        from src.kernel.memory_access import TileDependency,unique_by
        if program.scopes:
            raise ValueError('byte-copy projection requires its unique independent-byte tile source')
        if len(program.tile_programs) != 1:
            raise ValueError('byte-copy projection requires one source read and destination write')
        family, = program.tile_programs
        if (len(family.accesses) != 2 or len(family.axes) != 1
                or family.scope.op_ids not in (('MEMCPY',),('KV_OFFLOAD',))):
            raise ValueError('byte-copy projection requires one source read and destination write')
        read,write = family.accesses; axis, = family.axes
        views = unique_by(program.views,'view_id')
        for access,operation in ((read,Operation.READ),(write,Operation.WRITE)):
            view = views[access.view_id]
            if (access.operation is not operation or access.iteration_axes != ('byte',) or access.view_axes != ('byte',)
                    or access.view_axis_divisors or access.view_axis_padding or access.constant_value is not None
                    or access.unmapped or access.indirect_target is not None
                    or (view.shape,view.strides_bytes,view.dtype,view.element_bytes) != ((axis.extent,),(1,),'uint8',1)
                    or axis.axis_id != 'byte' or axis.block != 1):
                raise ValueError('byte-copy source and destination must conserve independent byte selections')
        if write.dependencies != (TileDependency(read.access_id,EventKind.READ_READY,()),):
            raise ValueError('byte-copy destination must depend on source READ_READY')
        return axis.extent
    if (not program.allocations and not program.views and not program.events and len(program.scopes)==1
            and program.scopes[0].op_ids in (('MEMCPY',),('KV_OFFLOAD',))
            and program.coverage.granularity is Granularity.TILE):
        return 0
    raise ValueError('byte-copy projection requires its unique independent-byte tile source or explicit empty payload')


def eager_attention_interface_demands(program):
    """Volumes/geometry of the declared compatibility primitive interfaces."""
    from src.kernel.memory_access import EventKind, EventRef, MemoryAccessProgram, Operation, Semantics, requested_bytes, unique_by
    if not isinstance(program, MemoryAccessProgram) or program.coverage.semantics is not Semantics.LOGICAL:
        raise TypeError("Eager interface projection requires a logical source program")
    if not program.tile_programs or program.accesses:
        raise ValueError('Eager requires its unique primitive tile pipeline')
    return eager_attention_tile_demands(program)



def eager_attention_tile_demands(program):
    """Primitive compulsory projections from actual materialized tile roles."""
    from src.kernel.memory_access import EventKind,Operation,TileDependency,requested_bytes,unique_by
    program.__post_init__()
    if len(program.tile_programs) != 1:raise ValueError('Eager requires one actual primitive tile pipeline')
    family,=program.tile_programs;accesses=unique_by(family.accesses,'access_id');views=unique_by(program.views,'view_id')
    names=('qkt.query','qkt.key','qkt.score','softmax.score','softmax.probability','pv.probability','pv.value','pv.output')
    if set(accesses) != set(names) or family.scope.op_ids != ('eager_attention','decomposed'):
        raise ValueError('Eager primitive tile roles incomplete')
    q,key,score,p,svalue,out=(views[accesses[name].view_id] for name in ('qkt.query','qkt.key','qkt.score','softmax.probability','pv.value','pv.output'))
    rows,dimension=q.shape;kd,length=key.shape
    if kd != dimension or svalue.shape != (length,dimension) or score.shape != (rows,length) or p.shape != (rows,length) or out.shape != q.shape:
        raise ValueError('Eager primitive tile geometry mismatch')
    if len({view.allocation_id for view in (q,key,score,p,svalue,out)}) != 6:
        raise ValueError('Eager materialized intermediate identity was not preserved')
    if (accesses['softmax.score'].view_id != accesses['qkt.score'].view_id
            or accesses['pv.probability'].view_id != accesses['softmax.probability'].view_id):
        raise ValueError('Eager materialized intermediate identity was not preserved')
    axes=(('m',rows,1),('n',length,1),('d',dimension,1))
    if tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != axes or family.causal_prefixes:
        raise ValueError('Eager must retain exact dot/row reduction coordinates')
    mappings=(('m',None),(None,'n'),('m','n'),('m','n'),('m','n'),('m','n'),('n','d'),('m','d'))
    for name,mapping in zip(names,mappings,strict=True):
        access=accesses[name];view=views[access.view_id]
        expected_loop=('m','n','d') if name in ('pv.probability','pv.value') else ('m','d') if name == 'pv.output' else ('m','n')
        operation=Operation.WRITE if name in ('qkt.score','softmax.probability','pv.output') else Operation.READ
        if (access.operation is not operation or access.iteration_axes != expected_loop or access.view_axes != mapping
                or access.view_axis_divisors or access.unmapped or access.indirect_target is not None
                or (view.dtype,view.element_bytes) != (q.dtype,q.element_bytes)
                or view.strides_bytes != (view.shape[1]*view.element_bytes,view.element_bytes)):
            raise ValueError('Eager must conserve its actual dense scalar storage selections')
    dependencies={
        'qkt.score':(TileDependency('qkt.query',EventKind.READ_READY,()),TileDependency('qkt.key',EventKind.READ_READY,())),
        'softmax.score':(TileDependency('qkt.score',EventKind.WRITE_VISIBLE,()),),
        'softmax.probability':(TileDependency('softmax.score',EventKind.READ_READY,('n',)),),
        'pv.probability':(TileDependency('softmax.probability',EventKind.WRITE_VISIBLE,()),),
        'pv.output':(TileDependency('pv.probability',EventKind.READ_READY,('n',)),TileDependency('pv.value',EventKind.READ_READY,('n',)))}
    if any(accesses[name].dependencies != expected for name,expected in dependencies.items()):
        raise ValueError('Eager primitive interface lacks producer/input readiness causality')
    size=lambda view:requested_bytes(view.region())
    return size(q)+size(key)+size(score),size(score)+size(p),size(p)+size(svalue)+size(out),(rows,dimension,length)


def unfused_moe_primitive_demands(program):
    """Derive compatibility work from the typed primitive interface DAG."""
    from src.kernel.memory_access import EventKind, EventRef, MemoryAccessProgram, Operation, Semantics, requested_bytes, unique_by
    if not isinstance(program, MemoryAccessProgram) or program.coverage.semantics is not Semantics.LOGICAL:
        raise TypeError("unfused-MoE projection requires a logical primitive program")
    if program.tile_programs:
        from src.kernel.memory_unfused_moe import unfused_moe_tile_demands
        return unfused_moe_tile_demands(program)
    scopes,views = unique_by(program.scopes,"scope_id"),unique_by(program.views,"view_id")
    stages = {}
    for access in program.accesses:
        roles = scopes[access.scope_id].op_ids
        if len(roles) != 3 or roles[0] != "unfused_moe":
            raise ValueError("unfused-MoE requires explicit primitive roles")
        name,kind = roles[1:]
        if name not in stages:
            stages[name] = (kind,[])
        if stages[name][0] != kind or access.repeat != 1 or access.region != views[access.view_id].region():
            raise ValueError("unfused-MoE primitive region/kind is not conserved")
        stages[name][1].append(access)
    if not {"router","router_softmax","output_pointwise"} <= set(stages):
        raise ValueError("unfused-MoE router/software primitive missing")
    nodes = {event.event_id:event for event in program.events}
    nodes.update({access.access_id:access for access in program.accesses})

    def requires(node, refs):
        pending,ancestors = list(node.dependencies),set()
        while pending:
            reference = pending.pop()
            if reference not in ancestors:
                ancestors.add(reference)
                pending.extend(nodes[reference.node_id].dependencies)
        if not set(refs) <= ancestors:
            raise ValueError("unfused-MoE primitive lacks actual input/producer readiness causality")

    def stage(name,kind):
        actual,accesses = stages[name]
        read_count = 1 if kind == "softmax" else 2
        if (actual != kind or len(accesses) != read_count+1
                or any(access.operation is not Operation.READ for access in accesses[:-1])
                or accesses[-1].operation is not Operation.WRITE):
            raise ValueError("unfused-MoE primitive operand/direction count mismatch")
        reads,write = accesses[:-1],accesses[-1]
        tensors = tuple(views[access.view_id] for access in accesses)
        if any((view.dtype,view.element_bytes) != (tensors[0].dtype,tensors[0].element_bytes) for view in tensors):
            raise ValueError("unfused-MoE primitive storage width/dtype mismatch")
        if kind == "mm":
            left,right,out = tensors
            rows,reduction = left.shape
            inner,columns = right.shape
            if inner != reduction or out.shape != (rows,columns):
                raise ValueError("unfused-MoE matrix geometry mismatch")
            geometry = (rows,reduction,columns)
        else:
            if any(view.shape != tensors[0].shape for view in tensors):
                raise ValueError("unfused-MoE software primitive geometry mismatch")
            geometry = tensors[0].shape
        requires(write,tuple(EventRef(access.access_id,EventKind.READ_READY) for access in reads))
        return geometry,reads,write

    previous_write = None
    for name,(kind,_) in stages.items():
        _,reads,write = stage(name,kind)
        if previous_write is not None:
            prior = (EventRef(previous_write.access_id,EventKind.WRITE_VISIBLE),)
            requires(write,prior)
            for read in reads:
                requires(read,prior)
        previous_write = write
    router,_,router_write = stage("router","mm")
    tokens,hidden,experts = router
    softmax,softmax_reads,_ = stage("router_softmax","softmax")
    if softmax != (tokens,experts) or softmax_reads[0].allocation_id != router_write.allocation_id:
        raise ValueError("unfused-MoE router logits temporary identity mismatch")
    requires(softmax_reads[0],(EventRef(router_write.access_id,EventKind.WRITE_VISIBLE),))
    counts = [0]*experts
    expert_mm_bytes = 0
    intermediate = None
    for name in tuple(stages):
        if not name.endswith(".w1"):
            continue
        if not name.startswith("expert"):
            raise ValueError("unfused-MoE expert identity missing")
        index = int(name[6:-3])
        if not 0 <= index < experts or counts[index] != 0:
            raise ValueError("unfused-MoE expert identity outside router domain")
        first,first_reads,first_write = stage(name,"mm")
        count,dimension,gate_up = first
        if dimension != hidden or gate_up % 2:
            raise ValueError("unfused-MoE W1 geometry mismatch")
        width = gate_up//2
        if intermediate is not None and intermediate != width:
            raise ValueError("unfused-MoE expert intermediate dimension mismatch")
        intermediate = width
        counts[index] = count
        prefix = name[:-3]
        activation,_,activated = stage(f"{prefix}.pointwise","pointwise")
        second,second_reads,second_write = stage(f"{prefix}.w2","mm")
        if activation != (count,width) or second != (count,width,hidden):
            raise ValueError("unfused-MoE pointwise/W2 geometry mismatch")
        if second_reads[0].allocation_id != activated.allocation_id:
            raise ValueError("unfused-MoE pointwise result identity mismatch")
        requires(second_reads[0],(EventRef(activated.access_id,EventKind.WRITE_VISIBLE),))
        expert_mm_bytes += sum(requested_bytes(access.region) for access in (*first_reads,first_write,*second_reads,second_write))
    expected = {"router","router_softmax","output_pointwise"}
    for index,count in enumerate(counts):
        if count:
            expected.update(f"expert{index}.{kind}" for kind in ("w1","pointwise","w2"))
    if set(stages) != expected or intermediate is None:
        raise ValueError("unfused-MoE unexpected/missing expert primitive")
    output,_,_ = stage("output_pointwise","pointwise")
    if output != (sum(counts),hidden):
        raise ValueError("unfused-MoE final elewise must conserve routed-row work, not claim top-k reduction")
    return {"geometry":(tokens,experts,hidden,intermediate),"tokens_per_expert":tuple(counts),
            "expert_mm_bytes":expert_mm_bytes,
            "primitive_requested_bytes":sum(requested_bytes(access.region) for access in program.accesses)}


def sparse_indexer_source_demands(program, *, expected_kernel=None):
    """Projected tensor work and score/top-k producer causality from source."""
    from src.kernel.memory_access import EventKind, EventRef, MemoryAccessProgram, Operation, Semantics, Unmapped, requested_bytes, unique_by
    if not isinstance(program,MemoryAccessProgram) or program.coverage.semantics is not Semantics.LOGICAL:
        raise TypeError("sparse indexer demands require a typed logical program")
    if program.coverage.executable:
        raise ValueError("sparse opaque-address source cannot claim executable memory coverage")
    if expected_kernel is not None:
        from src.kernel.memory_sparse_indexer import validate_indexer_invocation
        validate_indexer_invocation(program,expected_kernel)
    scopes,views = unique_by(program.scopes,"scope_id"),unique_by(program.views,"view_id")
    if not program.accesses and not program.tile_programs:
        if len(program.scopes) != 1 or program.scopes[0].op_ids != ("sparse_indexer","alias"):
            raise ValueError("zero memory work requires explicit index alias semantics")
        return {"alias":True,"read_bytes":0,"write_bytes":0,"geometry":None}
    from src.kernel.memory_sparse_indexer import sparse_indexer_tile_demands
    return sparse_indexer_tile_demands(program)


def fused_moe_source_demands(program):
    """Valid routed matrix work and producer ancestry, without DRAM inference."""
    if program.scopes and program.scopes[0].op_ids[:1]==('native_grouped_moe',):
        from src.kernel.memory_fused_moe import native_grouped_moe_demands
        return native_grouped_moe_demands(program)
    if program.tile_programs:
        if program.tile_programs[0].scope.op_ids[0] == 'fused_moe_packed_tiles':
            from src.kernel.memory_fused_moe import packed_moe_tile_demands
            return packed_moe_tile_demands(program)
        return grouped_moe_tile_demands(program)
    from src.kernel.memory_access import EventKind, EventRef, Operation, Unmapped, unique_by
    logical_requested_bytes(program,Operation.READ)
    if not program.accesses:
        return _empty_ep_fused_moe_source_demands(program)
    if any(len(scope.op_ids) >= 3 and scope.op_ids[:1] == ('fused_moe',) and scope.op_ids[2] == 'packed'
           for scope in program.scopes):
        return _packed_fused_moe_source_demands(program)
    scopes,views = unique_by(program.scopes,'scope_id'),unique_by(program.views,'view_id')
    by_expert,routed = {},None
    variant = None
    for access in program.accesses:
        roles = scopes[access.scope_id].op_ids
        if len(roles) != 4 or roles[0] != 'fused_moe' or roles[1] not in ('w1','w2') or access.repeat != 1:
            raise ValueError('grouped MoE requires explicit projection/expert/role requests')
        if variant is not None and variant != roles[1]:
            raise ValueError('grouped MoE mixes projection variants')
        variant = roles[1]
        if roles[2:] == ('shared','routing'):
            if routed is not None or access.operation is not Operation.READ or not isinstance(access.region,Unmapped):
                raise ValueError('grouped MoE requires one opaque valid-routing interface')
            routed = access
            continue
        index = int(roles[2])
        if index not in by_expert:
            by_expert[index] = {}
        if roles[3] in by_expert[index]:
            raise ValueError('grouped MoE duplicate expert role')
        by_expert[index][roles[3]] = access
    if routed is None or not by_expert or routed.region.dtype != 'int32' or routed.region.element_bytes != 4:
        raise ValueError('grouped MoE routing/expert requests missing')
    example = next(iter(by_expert.values()))
    input_parent,weight_parent,output_parent = (example[role].allocation_id for role in ('input','weight','output'))
    parents = []
    for allocation in (input_parent,weight_parent,output_parent):
        candidates = [view for view in views.values() if view.allocation_id == allocation]
        if allocation == weight_parent:
            candidates = [view for view in candidates if len(view.shape) == 3]
        if len(candidates) != 1:
            raise ValueError('grouped MoE requires actual unique tensor parent views')
        parents.append(candidates[0])
    query,weight,output = parents
    for view in parents:
        strides,size = [],view.element_bytes
        for axis in reversed(view.shape):
            strides.append(size)
            size *= axis
        if view.strides_bytes != tuple(reversed(strides)):
            raise ValueError('grouped MoE parent tensor layout must be contiguous')
    experts,columns,reduction = weight.shape
    headers = [scope for scope in program.scopes if scope.op_ids[:1] == ('fused_moe_ep',)]
    if len(headers) > 1:
        raise ValueError('grouped MoE duplicate EP source descriptor')
    if headers:
        roles = headers[0].op_ids
        if len(roles) != 6 or roles[1] != variant:
            raise ValueError('grouped MoE EP source descriptor mismatch')
        ep,expert_rank = int(roles[2]),int(roles[3])
        global_counts = tuple(int(value) for value in roles[4].split(','))
        owned = tuple(int(value) for value in roles[5].split(','))
        if (ep <= 1 or not 0 <= expert_rank < ep or len(global_counts) != experts*ep
                or min(global_counts) < 0 or owned != tuple(range(expert_rank*experts,(expert_rank+1)*experts))):
            raise ValueError('grouped MoE global/local EP placement mismatch')
        experts = len(global_counts)
    else:
        owned = tuple(range(experts))
        global_counts = None
    tokens,topk,out_columns = output.shape
    source_rows = tokens if variant == 'w1' else tokens*topk
    if (query.shape != (source_rows,reduction) or out_columns != columns
            or any((view.dtype,view.element_bytes) != (query.dtype,query.element_bytes) for view in (weight,output))):
        raise ValueError('grouped MoE parent matrix geometry/storage mismatch')
    nodes = {event.event_id:event for event in program.events}
    nodes.update({access.access_id:access for access in program.accesses})

    def requires(access, refs):
        pending,ancestors = list(access.dependencies),set()
        while pending:
            ref = pending.pop()
            if ref not in ancestors:
                ancestors.add(ref)
                pending.extend(nodes[ref.node_id].dependencies)
        if not set(refs) <= ancestors:
            raise ValueError('grouped MoE lacks actual routing/input/weight producer causality')

    counts = [0]*experts
    for expert,roles in by_expert.items():
        if expert not in owned or set(roles) != {'input','weight','output'}:
            raise ValueError('grouped MoE expert identity/operand roles mismatch')
        read,loaded,written = (roles[name] for name in ('input','weight','output'))
        if (read.operation is not Operation.READ or loaded.operation is not Operation.READ or written.operation is not Operation.WRITE
                or not isinstance(read.region,Unmapped) or not isinstance(written.region,Unmapped)
                or (read.allocation_id,loaded.allocation_id,written.allocation_id) != (input_parent,weight_parent,output_parent)):
            raise ValueError('grouped MoE requests must conserve the shared tensor parents')
        rows,inner = read.region.logical_shape
        if (inner != reduction or written.region.logical_shape != (rows,columns)
                or any((access.region.dtype,access.region.element_bytes) != (query.dtype,query.element_bytes) for access in (read,written))):
            raise ValueError('grouped MoE expert row/dtype geometry mismatch')
        view = views[loaded.view_id]
        if (view.shape != (columns,reduction) or view.offset_bytes != weight.offset_bytes+owned.index(expert)*columns*reduction*weight.element_bytes
                or (view.dtype,view.element_bytes) != (weight.dtype,weight.element_bytes)
                or view.strides_bytes != (reduction*weight.element_bytes,weight.element_bytes) or loaded.region != view.region()):
            raise ValueError('grouped MoE expert weight slice storage mismatch')
        requires(read,(EventRef(routed.access_id,EventKind.READ_READY),))
        requires(loaded,(EventRef(routed.access_id,EventKind.READ_READY),))
        requires(written,(EventRef(read.access_id,EventKind.READ_READY),EventRef(loaded.access_id,EventKind.READ_READY)))
        counts[expert] = rows
    if global_counts is None:
        global_counts = tuple(counts)
    expected_local = tuple(count if expert in owned else 0 for expert,count in enumerate(global_counts))
    if sum(global_counts) != tokens*topk or tuple(counts) != expected_local or routed.region.logical_shape != (sum(counts),):
        raise ValueError('grouped MoE global/local routed rows must conserve placement and tokens*topk')
    return {'projection':(variant,source_rows,reduction,columns),'routing':(global_counts,tuple(counts)),
            'read_bytes':logical_requested_bytes(program,Operation.READ),'write_bytes':logical_requested_bytes(program,Operation.WRITE)}


def grouped_moe_tile_demands(program):
    """Validate routed dots; retain compulsory logical-interface projections."""
    from src.kernel.memory_access import LogicalSelectionView, EventKind, Operation, TileDependency, unique_by
    program.__post_init__()
    views = unique_by(program.views, 'view_id')
    parents = None
    counts_by_expert = {}
    for family in program.tile_programs:
        header = family.scope.op_ids
        if (len(header) != 7 or header[0] != 'fused_moe_tiles' or header[1] not in ('w1','w2')
                or family.scope.agent != 'logical-routed-expert-dots' or family.causal_prefixes):
            raise ValueError('grouped tile source requires explicit expert-local dot semantics')
        _, variant, expert_text, query_id, weight_id, output_id, routing_id = header
        current = (variant, query_id, weight_id, output_id, routing_id)
        if parents is None:
            parents = current
        elif parents != current:
            raise ValueError('grouped tiles must retain their actual shared parents')
        expert = int(expert_text)
        if expert in counts_by_expert:
            raise ValueError('grouped tile source duplicates an expert selection')
        axes = unique_by(family.axes, 'axis_id')
        if set(axes) != {'row','column'} or any(axis.block != 1 for axis in family.axes):
            raise ValueError('grouped tiles require independent scalar output dots')
        count = axes['row'].extent
        counts_by_expert[expert] = count
        query, weight, output, routing = (views[name] for name in (query_id,weight_id,output_id,routing_id))
        if (len(query.shape) != 2 or len(weight.shape) != 3 or len(output.shape) != 3
                or not isinstance(routing,LogicalSelectionView) or routing.parent_view_id is not None
                or (routing.dtype,routing.element_bytes) != ('int32',4)):
            raise ValueError('grouped tile source requires actual query/weight/output and logical routing parents')
        tokens, topk, columns = output.shape
        local_experts, wn, reduction = weight.shape
        source_rows = tokens if variant == 'w1' else tokens*topk
        if (query.shape != (source_rows,reduction) or wn != columns or axes['column'].extent != columns
                or any((view.dtype,view.element_bytes) != (query.dtype,query.element_bytes) for view in (weight,output))
                or len({view.allocation_id for view in (query,weight,output,routing)}) != 4):
            raise ValueError('grouped tile source parent geometry/storage mismatch')
        for view in (query,weight,output):
            strides, size = [], view.element_bytes
            for dimension in reversed(view.shape):
                strides.append(size)
                size *= dimension
            if view.strides_bytes != tuple(reversed(strides)):
                raise ValueError('grouped tile actual parent layout must remain contiguous')
        accesses = unique_by(family.accesses, 'access_id')
        if set(accesses) != {'route','input','weight','output'}:
            raise ValueError('grouped tile source must retain routing and all dot operands')
        for role,parent,shape,iterations,mapping in (
                ('route',routing,(count,),('row',),('row',)),
                ('input',query,(count,reduction),('row','column'),('row',None)),
                ('output',output,(count,columns),('row','column'),('row','column'))):
            access = accesses[role]
            selected = views[access.view_id]
            if (not isinstance(selected,LogicalSelectionView) or selected.parent_view_id != parent.view_id
                    or (selected.shape,selected.dtype,selected.element_bytes) != (shape,parent.dtype,parent.element_bytes)
                    or access.operation is not (Operation.WRITE if role == 'output' else Operation.READ)
                    or access.iteration_axes != iterations or access.view_axes != mapping or not access.unmapped
                    or access.view_axis_divisors or access.indirect_target is not None):
                raise ValueError('grouped tile selected route/input/output must retain its actual typed parent')
        for role in ('input','weight'):
            if accesses[role].dependencies != (TileDependency('route',EventKind.READ_READY,()),):
                raise ValueError('grouped tile lacks actual routing/input/weight producer causality')
        if accesses['output'].dependencies != tuple(TileDependency(role,EventKind.READ_READY,()) for role in ('input','weight')):
            raise ValueError('grouped tile lacks actual routing/input/weight producer causality')
    if parents is None:
        raise ValueError('grouped tile source requires active expert work')
    variant, query_id, weight_id, output_id, routing_id = parents
    query, weight, output, routing = (views[name] for name in (query_id,weight_id,output_id,routing_id))
    tokens, topk, columns = output.shape
    local_experts, _, reduction = weight.shape
    ep_headers = [scope.op_ids for scope in program.scopes if scope.op_ids[:1] == ('fused_moe_ep',)]
    if ep_headers:
        if len(ep_headers) != 1 or len(ep_headers[0]) != 6 or ep_headers[0][1] != variant:
            raise ValueError('grouped tile EP placement descriptor mismatch')
        _,_,ep_text,rank_text,counts_text,owned_text = ep_headers[0]
        ep, rank = int(ep_text),int(rank_text)
        global_counts = tuple(int(value) for value in counts_text.split(','))
        owned = tuple(int(value) for value in owned_text.split(','))
        if (ep <= 1 or not 0 <= rank < ep or len(global_counts) != local_experts*ep
                or min(global_counts) < 0 or owned != tuple(range(rank*local_experts,(rank+1)*local_experts))):
            raise ValueError('grouped tile global/local EP placement mismatch')
    else:
        owned = tuple(range(local_experts))
        global_counts = tuple(counts_by_expert[index] if index in counts_by_expert else 0 for index in owned)
    local_counts = tuple(counts_by_expert[index] if index in counts_by_expert else 0 for index in range(len(global_counts)))
    if (not set(counts_by_expert) <= set(owned) or sum(global_counts) != tokens*topk
            or local_counts != tuple(count if index in owned else 0 for index,count in enumerate(global_counts))
            or routing.shape != (sum(local_counts),)):
        raise ValueError('grouped tile global/local routed rows must conserve placement and tokens*topk')
    for family in program.tile_programs:
        expert = int(family.scope.op_ids[2])
        accesses = unique_by(family.accesses,'access_id')
        loaded = accesses['weight']; selected = views[loaded.view_id]
        if ((selected.shape,selected.dtype,selected.element_bytes,selected.offset_bytes,selected.strides_bytes)
                != ((columns,reduction),weight.dtype,weight.element_bytes,
                    weight.offset_bytes+owned.index(expert)*columns*reduction*weight.element_bytes,
                    (reduction*weight.element_bytes,weight.element_bytes))
                or selected.allocation_id != weight.allocation_id or loaded.operation is not Operation.READ
                or loaded.iteration_axes != ('row','column') or loaded.view_axes != ('column',None)
                or loaded.view_axis_divisors or loaded.unmapped or loaded.indirect_target is not None):
            raise ValueError('grouped tile expert weight must retain its compact local slot')
    # The existing prediction ABI consumes compulsory logical interfaces,
    # while exported requests retain per-output operand reuse. Unknown
    # selections never authorize a physical unique-footprint projection.
    routes = sum(local_counts); width = query.element_bytes
    return {'projection':(variant,query.shape[0],reduction,columns),'routing':(global_counts,local_counts),
        'read_bytes':routes*4 + routes*reduction*width + len(counts_by_expert)*columns*reduction*width,
        'write_bytes':routes*columns*width}


def _empty_ep_fused_moe_source_demands(program):
    """Empty local mathematical work; native helpers remain outside this scope."""
    from src.kernel.memory_access import unique_by
    views = unique_by(program.views,'view_id')
    headers = [scope for scope in program.scopes if scope.op_ids[:1] == ('fused_moe_ep',)]
    if len(headers) != 1 or len(headers[0].op_ids) != 9 or any(name not in views for name in headers[0].op_ids[6:]):
        raise ValueError('empty grouped MoE requires explicit EP placement and parent geometry')
    _,variant,ep_text,rank_text,counts_text,owned_text = headers[0].op_ids[:6]
    ep,rank = int(ep_text),int(rank_text)
    counts = tuple(int(value) for value in counts_text.split(','))
    owned = tuple(int(value) for value in owned_text.split(','))
    query,weight,output = (views[name] for name in headers[0].op_ids[6:])
    experts,columns,reduction = weight.shape
    tokens,topk,out_columns = output.shape
    source_rows = tokens if variant == 'w1' else tokens*topk
    if (variant not in ('w1','w2') or ep <= 1 or not 0 <= rank < ep or len(counts) != experts*ep
            or min(counts) < 0 or owned != tuple(range(rank*experts,(rank+1)*experts))
            or sum(counts) != tokens*topk or any(counts[index] for index in owned)
            or query.shape != (source_rows,reduction) or out_columns != columns
            or len({view.allocation_id for view in (query,weight,output)}) != 3
            or any((view.dtype,view.element_bytes) != (query.dtype,query.element_bytes) for view in (weight,output))):
        raise ValueError('empty grouped MoE cannot erase nonzero local work or alter EP placement')
    return {'projection':(variant,source_rows,reduction,columns),
        'routing':(counts,(0,)*len(counts)),'read_bytes':0,'write_bytes':0}


def _packed_fused_moe_source_demands(program):
    """Packed matrix segments and opaque NZ weights from one declared source."""
    from src.kernel.memory_access import EventKind, EventRef, Operation, Unmapped, unique_by
    scopes,views = unique_by(program.scopes,'scope_id'),unique_by(program.views,'view_id')
    headers = [scope for scope in program.scopes if len(scope.op_ids) == 8
               and scope.op_ids[0] == 'fused_moe' and scope.op_ids[2:5] == ('packed','shared','routing')]
    if len(headers) != 1:
        raise ValueError('packed grouped MoE requires one explicit parent/group-list descriptor')
    header = headers[0]
    variant = header.op_ids[1]
    query,descriptor,output = (views[name] for name in header.op_ids[5:])
    experts,reduction,columns = descriptor.shape
    rows,inner = query.shape
    if (variant not in ('w1','w2') or inner != reduction or output.shape != (rows,columns)
            or any((view.dtype,view.element_bytes) != (query.dtype,query.element_bytes) for view in (descriptor,output))):
        raise ValueError('packed grouped MoE ND descriptor/input/output geometry mismatch')
    for view in (query,descriptor,output):
        strides,size = [],view.element_bytes
        for axis in reversed(view.shape):
            strides.append(size)
            size *= axis
        if view.strides_bytes != tuple(reversed(strides)):
            raise ValueError('packed grouped MoE public tensor descriptor must be contiguous')
    by_expert,group = {},None
    formatted_parent = None
    for access in program.accesses:
        scope = scopes[access.scope_id]
        if access.repeat != 1:
            raise ValueError('packed grouped MoE interface does not invent issued repetitions')
        if scope.scope_id == header.scope_id:
            if group is not None:
                raise ValueError('packed grouped MoE duplicate group-list read')
            group = access
            if (access.operation is not Operation.READ or not isinstance(access.region,Unmapped)
                    or (access.region.logical_shape,access.region.dtype,access.region.element_bytes) != ((experts,),'int64',8)):
                raise ValueError('packed grouped MoE group-list storage mismatch')
            continue
        roles = scope.op_ids
        if len(roles) != 5 or roles[:3] != ('fused_moe',variant,'packed'):
            raise ValueError('packed grouped MoE expert/operand role mismatch')
        expert = int(roles[3])
        if expert not in by_expert:
            by_expert[expert] = {}
        if roles[4] in by_expert[expert]:
            raise ValueError('packed grouped MoE duplicate expert operand')
        by_expert[expert][roles[4]] = access
    if group is None or not by_expert:
        raise ValueError('packed grouped MoE group-list/expert source missing')
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
            raise ValueError('packed grouped MoE lacks actual group/input/formatted-weight producer causality')

    counts,offset = [0]*experts,0
    for expert,operands in sorted(by_expert.items()):
        if not 0 <= expert < experts or set(operands) != {'input','weight','output'}:
            raise ValueError('packed grouped MoE expert/operand identities mismatch')
        read,weight,written = (operands[role] for role in ('input','weight','output'))
        if (read.operation is not Operation.READ or weight.operation is not Operation.READ or written.operation is not Operation.WRITE
                or read.view_id is None or written.view_id is None):
            raise ValueError('packed grouped MoE requires resolved row segments and weight reads')
        count = views[read.view_id].shape[0]
        for access,parent,width in ((read,query,reduction),(written,output,columns)):
            view = views[access.view_id]
            if (access.allocation_id != parent.allocation_id or view.shape != (count,width)
                    or view.offset_bytes != parent.offset_bytes+offset*width*parent.element_bytes
                    or view.strides_bytes != parent.strides_bytes or access.region != view.region()
                    or (view.dtype,view.element_bytes) != (parent.dtype,parent.element_bytes)):
                raise ValueError('packed grouped MoE expert prefix row segment mismatch')
        if (not isinstance(weight.region,Unmapped) or weight.view_id is not None
                or (weight.region.logical_shape,weight.region.dtype,weight.region.element_bytes) != ((reduction,columns),query.dtype,query.element_bytes)):
            raise ValueError('packed grouped MoE formatted weight must retain unresolved NZ layout')
        if formatted_parent is None:
            formatted_parent = weight.allocation_id
        if weight.allocation_id != formatted_parent or formatted_parent in (query.allocation_id,descriptor.allocation_id,output.allocation_id,group.allocation_id):
            raise ValueError('packed grouped MoE ND and formatted weight storage identities must be separate')
        requires(read,(EventRef(group.access_id,EventKind.READ_READY),))
        requires(weight,(EventRef(group.access_id,EventKind.READ_READY),))
        requires(written,(EventRef(read.access_id,EventKind.READ_READY),EventRef(weight.access_id,EventKind.READ_READY)))
        counts[expert] = count
        offset += count
    if offset != rows:
        raise ValueError('packed grouped MoE expert rows must cover the packed parent once')
    return {'projection':(variant,rows,reduction,columns),'routing':(tuple(counts),tuple(counts)),
            'read_bytes':logical_requested_bytes(program,Operation.READ),'write_bytes':logical_requested_bytes(program,Operation.WRITE)}


def short_context_topk_source_demands(program, *, expected_kernel=None):
    """Only the actual independent row source can authorize the full capacity."""
    return short_context_topk_tile_demands(program,expected_kernel=expected_kernel)



def sparse_attention_source_demands(program, *, expected_kernel=None,expected_bindings=None):
    """Dsa selected cache requests and actual index/payload/output causality."""
    from src.kernel.memory_sparse_attention import dsa_sparse_tile_demands
    return dsa_sparse_tile_demands(program,expected_kernel=expected_kernel,expected_bindings=expected_bindings)


def v4_sparse_source_demands(program, *, expected_kernel=None,expected_bindings=None):
    """Source work and temporary producer chains, without a residency claim."""
    from src.kernel.memory_access import MemoryAccessProgram,Semantics
    if not isinstance(program,MemoryAccessProgram):raise TypeError('V4 sparse demands require a typed source program')
    if program.coverage.semantics is not Semantics.LOGICAL:raise ValueError('V4 sparse demands require logical source semantics')
    if (expected_kernel is not None and expected_kernel.variant == 'dsv4_sparse_prefill'
            and (len(program.scopes) != 1 or program.scopes[0].op_ids[:1] != ('v4_prefill_tiles',))):
        raise ValueError('Source differs from its actual V4 prefill invocation')
    if program.tile_programs:
        from src.kernel.memory_sparse_attention import v4_prefill_tile_demands,v4_decode_tile_demands
        if program.scopes and program.scopes[0].op_ids[:1]==('v4_sparse_decode_tiles',):
            return v4_decode_tile_demands(program,expected_kernel=expected_kernel)
        return v4_prefill_tile_demands(program,expected_kernel=expected_kernel,expected_bindings=expected_bindings)
    raise ValueError('V4 sparse attention requires its current tile source')


def dense_attention_interface_bytes(program):
    """Inherited Q/K/V/O pricing subset; explicit LSE remains unpriced here."""
    values = dense_attention_interface_demands(program)
    return tuple(values[role] for role in ("query", "key", "value", "output"))


def dense_attention_interface_demands(program):
    """Validate all known external tensor stores and their actual read ancestry."""
    from src.kernel.memory_access import EventKind, EventRef, MemoryAccessProgram, Operation, Semantics, requested_bytes, unique_by
    if not isinstance(program, MemoryAccessProgram):
        raise TypeError("dense attention projection requires a source program")
    if program.coverage.semantics is not Semantics.LOGICAL:
        raise ValueError("dense attention interface projection requires logical semantics")
    if program.tile_programs:
        return dense_attention_tile_demands(program)
    scopes = unique_by(program.scopes, "scope_id")
    views = unique_by(program.views, "view_id")
    values = {}
    input_ready = []
    outputs = []
    for access in program.accesses:
        roles = scopes[access.scope_id].op_ids
        if len(roles) != 2 or roles[0] != "flash_attention" or roles[1] in values:
            raise ValueError("dense attention interface requires explicit unique operand roles")
        role = roles[1]
        if role not in ("query", "key", "value", "output", "lse"):
            raise ValueError("unknown dense attention interface role")
        operation = Operation.WRITE if role in ("output","lse") else Operation.READ
        if access.operation is not operation or access.repeat != 1:
            raise ValueError("dense attention interface direction/repeat mismatch")
        if access.region != views[access.view_id].region():
            raise ValueError("dense attention interface must conserve its actual tensor region")
        if role in ("output","lse"):
            outputs.append(access)
        else:
            input_ready.append(EventRef(access.access_id, EventKind.READ_READY))
        values[role] = requested_bytes(access.region)
    if set(values) != {"query", "key", "value", "output", "lse"}:
        raise ValueError("dense attention external operand missing")
    batch,query,heads,_ = views['query'].shape
    lse = views['lse']
    if (lse.shape,lse.dtype,lse.element_bytes,lse.strides_bytes) != (
            (batch,heads,query),'fp32',4,(heads*query*4,query*4,4)):
        raise ValueError('dense attention requires its FP32[B,H,Q] LSE storage')
    nodes = {event.event_id: event for event in program.events}
    nodes.update({access.access_id: access for access in program.accesses})
    for output in outputs:
        pending = list(output.dependencies)
        ancestors = set()
        while pending:
            reference = pending.pop()
            if reference not in ancestors:
                ancestors.add(reference)
                pending.extend(nodes[reference.node_id].dependencies)
        if not set(input_ready) <= ancestors:
            raise ValueError("dense attention output/LSE requires all Q/K/V READ_READY dependencies")
    return values


def dense_attention_output_projection(program):
    """Both output and FP32 LSE stores use their actual source bytes."""
    work = dense_attention_interface_demands(program)
    if program.tile_programs and program.tile_programs[0].scope.op_ids[-2:]==('paged_lookup','legacy_output_only'):
        return work['output']
    return work['output']+work['lse']




def dense_attention_tile_demands(program):
    """Compulsory Q/K/V/O/LSE work from exact masked online-softmax tiles."""
    from src.kernel.memory_access import EventKind,Operation,TileDependency,TileCausalPrefix,requested_bytes,unique_by,LogicalSelectionView,tile_template_requested_bytes
    program.__post_init__()
    if program.tile_programs and program.tile_programs[0].scope.op_ids[0] in ('fa2_ragged_dense','metax_kvcache'):
        from src.kernel.memory_flash_attention import dense_ragged_tile_demands
        return dense_ragged_tile_demands(program)
    if len(program.tile_programs) != 1:
        raise ValueError('dense attention requires one actual query/KV program')
    family,=program.tile_programs;views=unique_by(program.views,'view_id');roles=unique_by(family.accesses,'access_id')
    paged=family.scope.op_ids[-2:]==('paged_lookup','legacy_output_only')
    if set(roles) != {'query','key','value','output','lse'}|({'block_table','cache_lengths','cu_query'} if paged else set()) or family.scope.op_ids not in (
            ('flash_attention','causal'),('flash_attention','noncausal'),
            ('flash_attention','causal','paged_lookup','legacy_output_only'),('flash_attention','noncausal','paged_lookup','legacy_output_only')):
        raise ValueError('dense attention external operand/mask missing')
    batch,query,heads,dimension=views['query'].shape;keys,kv_heads=views['key'].shape[1:3]
    if heads%kv_heads:raise ValueError('dense attention requires exact GQA groups')
    group=heads//kv_heads
    if (tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != (
            ('batch',batch,1),('kv_head',kv_heads,1),('query',query,1),('key',keys,1))
            or family.causal_prefixes != ((TileCausalPrefix('key','query'),) if family.scope.op_ids[1] == 'causal' else ())):
        raise ValueError('dense attention must preserve actual causal query/KV coordinates')
    from src.kernel.memory_flash_attention import dense_strides
    expected={'query':(batch,query,heads,dimension),'key':(batch,keys,kv_heads,dimension),
        'value':(batch,keys,kv_heads,dimension),'output':(batch,query,heads,dimension),'lse':(batch,heads,query)}
    for role,shape in expected.items():
        parent=views[role];access=roles[role];view=views[access.view_id]
        width=4 if role == 'lse' else views['query'].element_bytes
        dtype='fp32' if role == 'lse' else views['query'].dtype
        if (parent.shape,parent.strides_bytes,parent.dtype,parent.element_bytes) != (shape,dense_strides(shape,width),dtype,width):
            raise ValueError('dense attention requires its FP32[B,H,Q] LSE storage' if role == 'lse' else 'dense attention must retain actual BSHD typed parents')
        if role in ('query','output'):
            active=(batch,query,kv_heads,group,dimension)
            bs,qs,hs,ds=parent.strides_bytes;strides=(bs,qs,hs*group,hs,ds)
            mapping=('batch','query','kv_head',None,None)
        elif role == 'lse':
            active=(batch,kv_heads,group,query);bs,hs,qs=parent.strides_bytes;strides=(bs,hs*group,hs,qs)
            mapping=('batch','kv_head',None,'query')
        else:active=shape;strides=parent.strides_bytes;mapping=('batch','key','kv_head',None)
        iterations=('batch','kv_head','query','key') if role in ('key','value') else ('batch','kv_head','query')
        if ((view.allocation_id,view.offset_bytes,view.shape,view.strides_bytes,view.dtype,view.element_bytes)
                != (parent.allocation_id,parent.offset_bytes,active,strides,dtype,width)
                or access.operation is not (Operation.WRITE if role in ('output','lse') else Operation.READ)
                or access.iteration_axes != iterations or access.view_axes != mapping or access.view_axis_divisors
                or access.unmapped!=(paged and role in ('key','value')) or access.indirect_target is not None):
            raise ValueError('dense attention tile must conserve actual query/GQA/KV selections')
    qk=(TileDependency('query',EventKind.READ_READY,()),TileDependency('key',EventKind.READ_READY,('key',)))
    if roles['output'].dependencies != (*qk,TileDependency('value',EventKind.READ_READY,('key',))):
        raise ValueError('dense attention output requires all Q/K/V READ_READY dependencies')
    if roles['lse'].dependencies != qk:
        raise ValueError('dense attention LSE requires Q/K READ_READY dependencies')
    controls=tuple(TileDependency(role,EventKind.READ_READY,()) for role in ('cache_lengths','cu_query')) if paged else ()
    if roles['query'].dependencies!=controls:raise ValueError('Paged FA2 query requires its actual length/endpoint READ_READY')
    for role in ('key','value'):
        if roles[role].dependencies != ((TileDependency('block_table',EventKind.READ_READY,()),) if paged else ()):
            raise ValueError('Paged FA2 K/V requires its actual block-table READ_READY')
    result={role:requested_bytes(views[role].region()) for role in expected}
    if paged:
        if len({views[role].allocation_id for role in (*expected,'block_table','cache_lengths','cu_query')})!=8:
            raise ValueError('Paged FA2 public payload and metadata parents must remain distinct')
        table=views['block_table'];access=roles['block_table']
        if (program.hardware_id!='nvidia_a800' or not isinstance(table,LogicalSelectionView) or table.parent_view_id is not None or table.coordinate_map
                or (table.shape,table.dtype,table.element_bytes)!=((batch,keys),'int32',4)
                or table.allocation_id in {views[role].allocation_id for role in expected}
                or access.view_id!='block_table' or access.operation is not Operation.READ or not access.unmapped
                or access.iteration_axes!=('batch','kv_head','query','key') or access.view_axes!=('batch','key')
                or access.dependencies!=controls or access.value_index is not None or access.indirect_target is not None
                or access.view_axis_divisors or len(access.input_ready)!=1
                or access.input_ready[0].kind is not EventKind.LOGICAL_AVAILABLE):
            raise ValueError('Paged FA2 lookup requires exact logical token selections and boundary availability')
        result['block_table_logical_read_bytes']=tile_template_requested_bytes(program,family,access)
        for role,shape,mapping in (('cache_lengths',(batch,),('batch',)),('cu_query',(batch+1,),('batch',None))):
            parent=views[role];access=roles[role];selected=views[access.view_id]
            if ((parent.shape,parent.strides_bytes,parent.dtype,parent.element_bytes,parent.offset_bytes)!=(shape,(4,),'int32',4,0)
                    or access.view_id!=('cu_query.pairs' if role=='cu_query' else role)
                    or access.operation is not Operation.READ or access.iteration_axes!=('batch','kv_head','query')
                    or access.view_axes!=mapping or access.dependencies or access.unmapped or len(access.input_ready)!=1
                    or access.value_index is not None or access.indirect_target is not None or access.predicate is not None):
                raise ValueError('Paged FA2 control requires typed cache lengths and cumulative Q endpoints')
            if role=='cu_query' and (selected.allocation_id,selected.shape,selected.strides_bytes,selected.offset_bytes,selected.dtype,selected.element_bytes)!=(parent.allocation_id,(batch,2),(4,4),0,'int32',4):
                raise ValueError('Paged FA2 cumulative Q reads require the actual overlapping endpoint pairs')
        result['control_metadata_logical_read_bytes']=sum(tile_template_requested_bytes(program,family,roles[role]) for role in ('cache_lengths','cu_query'))
    return result


def normalization_source_bytes(program,*,expected_kernel=None):
    """Initial tensor, repeated gamma, second-pass and in-place output work."""
    from src.kernel.memory_access import EventKind, EventRef, MemoryAccessProgram, Operation, Semantics, requested_bytes, unique_by
    from src.kernel.memory_access_builder import MemoryAccessWork
    direct=isinstance(program,MemoryAccessWork)
    if not direct and not isinstance(program, MemoryAccessProgram):
        raise TypeError("normalization projection requires a typed source program")
    if not direct and program.coverage.semantics is not Semantics.LOGICAL:
        raise ValueError("normalization projection requires logical semantics")
    if direct or program.tile_programs:
        from src.kernel.memory_access import tile_template_requested_bytes,TileDependency
        if direct:family=program
        else:
            program.__post_init__()
            if len(program.tile_programs) != 1:raise ValueError('normalization requires one actual row tile loop')
            family,=program.tile_programs
        views=unique_by(program.views,'view_id');axes=unique_by(family.axes,'axis_id')
        roles=unique_by(family.accesses,'access_id');fused='residual' in roles
        expected={'input','weight','reread','output'}|({'residual','residual_update'} if fused else set())
        header=program.operation_ids if direct else family.scope.op_ids
        if (len(header) not in (1,3) or header[0]!='normalization' or set(roles) != expected
                or set(axes) != {'row'} or axes['row'].block != 1):
            raise ValueError('normalization tile roles/reduction partition incomplete')
        rows=axes['row'].extent
        logical_rows=rows
        if len(header)==3:
            logical_rows=int(header[1].removeprefix('logical_rows='));execution_rows=int(header[2].removeprefix('execution_rows='))
            if (header[1:]!=(f'logical_rows={logical_rows}',f'execution_rows={execution_rows}')
                    or not 0<logical_rows<execution_rows or execution_rows!=rows):
                raise ValueError('normalization replay declaration differs from actual row extent')
        if expected_kernel is not None:
            from src.kernel.kernel_spec import KernelKind
            from src.kernel.memory_normalization import _normalization_row_extents
            kernel=expected_kernel;view=views[roles['input'].view_id]
            if (kernel.kind is not KernelKind.VECTOR or kernel.variant!=('add_rmsnorm' if fused else 'rmsnorm')
                    or kernel.hardware_id!=program.hardware_id or (logical_rows,rows)!=_normalization_row_extents(kernel)
                    or (view.shape,view.dtype,view.element_bytes)!=((rows,kernel.shape['hidden']),kernel.dtype,kernel.shape['dtype_bytes'])):
                raise ValueError('normalization source differs from actual kernel/replay mapping')
        for role,access in roles.items():
            operation=Operation.WRITE if role in ('output','residual_update') else Operation.READ
            if access.operation is not operation or access.iteration_axes != ('row',) or access.view_axes != ((None,) if role=='weight' else ('row',None)):
                raise ValueError('normalization tile direction/row selection mismatch')
        required={'output':(TileDependency('reread',EventKind.READ_READY,()),TileDependency('weight',EventKind.READ_READY,())),
            'reread':(TileDependency('residual_update',EventKind.WRITE_VISIBLE,()) if fused else TileDependency('input',EventKind.READ_READY,()),)}
        if fused:required['residual_update']=(TileDependency('input',EventKind.READ_READY,()),TileDependency('residual',EventKind.READ_READY,()))
        for role,dependencies in required.items():
            if not set(dependencies) <= set(roles[role].dependencies):
                raise ValueError('normalization source lacks actual input/updated-residual readiness')
        if fused:
            if (roles['reread'].view_id != roles['residual_update'].view_id or roles['residual_update'].view_id != roles['residual'].view_id
                    or roles['output'].view_id != roles['input'].view_id):
                raise ValueError('fused normalization tile must retain its actual in-place identities')
        elif roles['reread'].view_id != roles['input'].view_id:
            raise ValueError('RMSNorm tile must reread its actual input')
        if direct:
            from src.kernel.memory_access import _tile_template_requested_bytes
            volumes={role:_tile_template_requested_bytes(views,axes,family,access) for role,access in roles.items()}
        else:volumes={role:tile_template_requested_bytes(program,family,access) for role,access in roles.items()}
        # The source records every captured row and pass. The existing
        # predictor's logical-prefix domain remains an explicit projection;
        # gamma's unique footprint is not scaled by the number of rows.
        if logical_rows!=rows:volumes={role:value//rows*logical_rows for role,value in volumes.items()}
        weight=views[roles['weight'].view_id].shape[0]*views[roles['weight'].view_id].element_bytes
        return {'input_bytes':volumes['input'],'initial_read_bytes':volumes['input']+(volumes['residual'] if fused else 0),
            'weight_bytes':weight,'logical_weight_read_bytes':volumes['weight'],'reread_bytes':volumes['reread'],
            'output_bytes':volumes['output'],'residual_write_bytes':volumes['residual_update'] if fused else 0,
            'resident_footprint_bytes':2*volumes['input']+weight}
    scopes, views = unique_by(program.scopes, "scope_id"), unique_by(program.views, "view_id")
    roles = {}
    for access in program.accesses:
        op_ids = scopes[access.scope_id].op_ids
        if len(op_ids) != 2 or op_ids[0] != "normalization" or op_ids[1] in roles:
            raise ValueError("normalization source requires explicit unique access roles")
        if access.region != views[access.view_id].region():
            raise ValueError("normalization source region must conserve its tensor view")
        roles[op_ids[1]] = access
    fused = "residual" in roles
    expected = {"input", "weight", "reread", "output"}
    if fused:
        expected |= {"residual", "residual_update"}
    if set(roles) != expected:
        raise ValueError("normalization source operand roles incomplete")
    rows, hidden = views[roles["input"].view_id].shape
    for role, access in roles.items():
        operation = Operation.WRITE if role in ("output", "residual_update") else Operation.READ
        if access.operation is not operation or access.repeat != (rows if role == "weight" else 1):
            raise ValueError("normalization source direction or repeat mismatch")
    nodes = {event.event_id: event for event in program.events}
    nodes.update({access.access_id: access for access in program.accesses})

    def requires(role, references):
        pending, ancestors = list(roles[role].dependencies), set()
        while pending:
            reference = pending.pop()
            if reference not in ancestors:
                ancestors.add(reference)
                pending.extend(nodes[reference.node_id].dependencies)
        if not set(references) <= ancestors:
            raise ValueError("normalization source lacks actual input/updated-residual readiness")

    requires("output", tuple(EventRef(roles[role].access_id, EventKind.READ_READY) for role in ("input", "reread", "weight")))
    if fused:
        requires("residual_update", tuple(EventRef(roles[role].access_id, EventKind.READ_READY) for role in ("input", "residual")))
        requires("reread", (EventRef(roles["residual_update"].access_id, EventKind.WRITE_VISIBLE),))
        if (roles["reread"].view_id != roles["residual_update"].view_id
                or roles["residual_update"].view_id != roles["residual"].view_id
                or roles["output"].view_id != roles["input"].view_id):
            raise ValueError("fused normalization requires actual in-place tensor identities")
    elif roles["reread"].view_id != roles["input"].view_id:
        raise ValueError("RMSNorm must reread the actual input tensor")
    volumes = {role: requested_bytes(access.region) for role, access in roles.items()}
    initial_read = volumes["input"] + (volumes["residual"] if fused else 0)
    residual_write = volumes["residual_update"] if fused else 0
    return {"input_bytes": volumes["input"], "initial_read_bytes": initial_read,
            "weight_bytes": volumes["weight"], "logical_weight_read_bytes": volumes["weight"]*rows,
            "reread_bytes": volumes["reread"], "output_bytes": volumes["output"],
            "residual_write_bytes": residual_write,
            "resident_footprint_bytes": 2*volumes["input"] + volumes["weight"]}


def collective_source_work(program):
    """Project the one current logical collective tile source."""
    from src.kernel.memory_access import MemoryAccessProgram,Semantics
    if not isinstance(program,MemoryAccessProgram) or program.coverage.semantics is not Semantics.LOGICAL:
        raise TypeError('collective projection requires a logical source program')
    if not program.tile_programs:
        raise ValueError('collective projection requires its current tile source')
    agent=program.tile_programs[0].scope.agent
    if agent=='variable-collective-segments':return variable_collective_source_work(program)
    if agent=='all-reduce-elements':return numeric_all_reduce_tile_source_work(program)
    if agent=='all-reduce-unbound-words':return untyped_all_reduce_tile_source_work(program)
    return collective_copy_tile_source_work(program)


def untyped_all_reduce_tile_source_work(program):
    """Conserve all participant bytes without inventing numeric word width."""
    from src.kernel.memory_access import EventKind,Operation,TileDependency,TensorView,unique_by
    program.__post_init__()
    if program.scopes or program.accesses or len(program.tile_programs)!=1:
        raise ValueError('untyped all-reduce requires its single compressed word-partition source')
    family,=program.tile_programs;header=family.scope.op_ids
    if (family.scope.agent!='all-reduce-unbound-words' or len(header)<5
            or header[:3]!=('communication','all_reduce','sum') or len(family.axes)!=1
            or len(family.partitions)!=1 or family.partitions[0].axis_id!='byte' or family.causal_prefixes):
        raise ValueError('untyped all-reduce must retain actual participants and its unknown word partition')
    ranks=tuple(int(value) for value in header[3:]);axis,=family.axes
    if len(set(ranks))!=len(ranks) or min(ranks)<0 or (axis.axis_id,axis.block)!=('byte',1):
        raise ValueError('all-reduce participant/byte coordinates differ')
    reads=tuple(access for access in family.accesses if access.operation is Operation.READ)
    writes=tuple(access for access in family.accesses if access.operation is Operation.WRITE)
    if ((len(reads),len(writes))!=(len(ranks),len(ranks))
            or {a.scope.rank for a in reads if a.scope is not None}!=set(ranks)
            or {a.scope.rank for a in writes if a.scope is not None}!=set(ranks)):
        raise ValueError('all-reduce requires every actual input/output participant')
    views=unique_by(program.views,'view_id')
    required={TileDependency(access.access_id,EventKind.READ_READY,('byte',),partition_axis='byte') for access in reads}
    for access in family.accesses:
        view=views[access.view_id];role='input' if access.operation is Operation.READ else 'output'
        if (not isinstance(view,TensorView) or (view.shape,view.dtype,view.element_bytes,view.strides_bytes)!=((axis.extent,),'uint8',1,(1,))
                or access.iteration_axes!=('byte',) or access.view_axes!=('byte',)
                or access.view_axis_divisors or access.view_axis_padding or access.indirect_target is not None or access.unmapped
                or access.scope is None or access.scope.op_ids!=('communication','all_reduce',role)):
            raise ValueError('untyped all-reduce must retain its actual complete byte views')
        if access.operation is Operation.WRITE:
            if len(access.dependencies)!=len(required) or set(access.dependencies)!=required:
                raise ValueError('all-reduce word requires all actual participant READ_READY inputs')
        elif access.dependencies:raise ValueError('all-reduce inputs cannot acquire unrelated barriers')
    if (len({views[a.view_id].allocation_id for a in reads})!=len(reads)
            or len({views[a.view_id].allocation_id for a in writes})!=len(writes)):
        raise ValueError('all-reduce peer aliases require actual participant storage mappings')
    for written in writes:
        output=views[written.view_id]
        for loaded in reads:
            source=views[loaded.view_id]
            if output.allocation_id==source.allocation_id and (written.scope.rank!=loaded.scope.rank or output.region()!=source.region()):
                raise ValueError('all-reduce in-place words must retain their matching participant geometry')
    return axis.extent,len(ranks),'all_reduce'


def numeric_all_reduce_tile_source_work(program):
    """Typed scalar sums conserve each participant's matching element."""
    from src.kernel.memory_access import EventKind,Operation,TileDependency,unique_by,TensorView,LogicalSelectionView
    from src.kernel.kernel_spec import DTYPE_BYTES
    program.__post_init__()
    if len(program.tile_programs) != 1:
        raise ValueError('all-reduce source requires one scalar coordinate loop')
    family, = program.tile_programs
    header = family.scope.op_ids
    if len(header) < 5 or header[:3] != ('communication','all_reduce','sum'):
        raise ValueError('all-reduce numeric reduction and participants must be explicit')
    ranks = tuple(int(value) for value in header[3:])
    if len(set(ranks)) != len(ranks) or min(ranks) < 0:
        raise ValueError('all-reduce participants must be unique')
    views = unique_by(program.views,'view_id')
    reads = tuple(access for access in family.accesses if access.operation is Operation.READ)
    writes = tuple(access for access in family.accesses if access.operation is Operation.WRITE)
    if (len(reads) != len(ranks) or len(writes) != len(ranks) or len(family.axes) != 1
            or {access.scope.rank for access in reads if access.scope is not None} != set(ranks)
            or {access.scope.rank for access in writes if access.scope is not None} != set(ranks)):
        raise ValueError('all-reduce requires every actual input and output participant')
    axis, = family.axes
    first = views[reads[0].view_id]
    if first.dtype not in ('bf16','fp16','fp32','int8','int32','int64') or (axis.axis_id,axis.block) != ('element',1):
        raise ValueError('all-reduce requires independently declared numeric scalar coordinates')
    width = DTYPE_BYTES[first.dtype]
    required = {TileDependency(access.access_id,EventKind.READ_READY,()) for access in reads}
    if (len({views[access.view_id].allocation_id for access in reads}) != len(reads)
            or len({views[access.view_id].allocation_id for access in writes}) != len(writes)):
        raise ValueError('all-reduce peer aliases need actual participant storage mappings')
    for written in writes:
        output = views[written.view_id]
        for loaded in reads:
            source = views[loaded.view_id]
            if output.allocation_id == source.allocation_id and (written.scope.rank != loaded.scope.rank or output.region() != source.region()):
                raise ValueError('all-reduce in-place words must retain their matching participant geometry')
    for access in family.accesses:
        view = views[access.view_id]
        role = 'input' if access.operation is Operation.READ else 'output'
        if ((view.shape,view.dtype,view.element_bytes) != ((axis.extent,),first.dtype,width)
                or (isinstance(view,TensorView) and view.strides_bytes != (width,))
                or access.iteration_axes != ('element',) or access.view_axes != ('element',)
                or access.view_axis_divisors or access.view_axis_padding or access.indirect_target is not None
                or access.unmapped != isinstance(view,LogicalSelectionView)
                or access.scope is None or access.scope.op_ids != ('communication','all_reduce',role)):
            raise ValueError('all-reduce must preserve matching numeric element coordinates and rank scopes')
        if access.operation is Operation.WRITE:
            if len(access.dependencies) != len(required) or set(access.dependencies) != required:
                raise ValueError('all-reduce output element requires all actual participant READ_READY inputs')
        elif access.dependencies:
            raise ValueError('independent all-reduce inputs cannot acquire unrelated tile barriers')
    return axis.extent*width,len(ranks),'all_reduce'


def collective_copy_tile_source_work(program):
    """Validate peer coordinate placement and direct contributor causality."""
    from src.kernel.memory_access import EventKind,Operation,TileDependency,unique_by
    program.__post_init__()
    if len(program.tile_programs) != 1:
        raise ValueError('collective copy source requires one coordinate loop')
    family, = program.tile_programs
    header=family.scope.op_ids
    if len(header) < 4 or header[0] != 'communication' or header[1] not in ('all_gather','all_to_all','send_recv'):
        raise ValueError('collective copy operation/participants must be explicit')
    operation=header[1];ranks=tuple(int(value) for value in header[2:])
    if len(set(ranks)) != len(ranks) or min(ranks) < 0:
        raise ValueError('collective copy participants must be unique')
    views=unique_by(program.views,'view_id');accesses=unique_by(family.accesses,'access_id')
    reads=tuple(access for access in family.accesses if access.operation is Operation.READ)
    writes=tuple(access for access in family.accesses if access.operation is Operation.WRITE)
    if len(family.axes) != 1 or family.axes[0].axis_id != 'byte' or family.axes[0].block != 1:
        raise ValueError('collective copy requires independent byte coordinates')
    chunk=family.axes[0].extent;payload=chunk*len(ranks) if operation == 'all_to_all' else chunk
    input_ranks={access.scope.rank for access in reads if access.scope is not None}
    output_ranks={access.scope.rank for access in writes if access.scope is not None}
    if operation == 'send_recv':
        if len(ranks) != 2 or len(input_ranks) != 1 or len(output_ranks) != 1 or input_ranks & output_ranks or input_ranks|output_ranks != set(ranks):
            raise ValueError('point-to-point requires distinct source and destination participants')
    elif input_ranks != set(ranks) or output_ranks != set(ranks):
        raise ValueError('collective copy requires all participant interfaces')
    if (len(reads),len(writes)) != ((1,1) if operation == 'send_recv' else
            (len(ranks),len(ranks)**2) if operation == 'all_gather' else (len(ranks)**2,len(ranks)**2)):
        raise ValueError('collective copy peer segment count mismatch')
    parents={}
    for access in family.accesses:
        view=views[access.view_id];scope=access.scope
        if (scope is None or scope.op_ids != ('communication',operation,'input' if access.operation is Operation.READ else 'output')
                or access.iteration_axes != ('byte',) or access.view_axes != ('byte',) or access.view_axis_divisors
                or access.unmapped or access.indirect_target is not None
                or (view.shape,view.strides_bytes,view.dtype,view.element_bytes) != ((chunk,),(1,),'uint8',1)):
            raise ValueError('collective copy must conserve actual peer byte selections')
        parent=max((item for item in program.views if item.allocation_id == view.allocation_id),key=lambda item:item.shape[0])
        expected=payload*len(ranks) if operation == 'all_gather' and access.operation is Operation.WRITE else payload
        if (parent.shape,parent.strides_bytes,parent.dtype,parent.element_bytes) != ((expected,),(1,),'uint8',1):
            raise ValueError('collective copy parent extent differs from payload semantics')
        parents[access.access_id]=parent
    pairs=set()
    for write in writes:
        if len(write.dependencies) != 1 or write.dependencies[0].access_id not in accesses:
            raise ValueError('collective output requires actual participant READ_READY dependencies')
        reference,=write.dependencies;read=accesses[reference.access_id]
        if read.operation is not Operation.READ or reference != TileDependency(read.access_id,EventKind.READ_READY,()):
            raise ValueError('collective output requires actual participant READ_READY dependencies')
        source,destination=read.scope.rank,write.scope.rank
        if (source,destination) in pairs:raise ValueError('collective copy peer pair repeated')
        pairs.add((source,destination))
        src,dst=views[read.view_id],views[write.view_id]
        source_index,destination_index=ranks.index(source),ranks.index(destination)
        expected_read=parents[read.access_id].offset_bytes+(destination_index*chunk if operation == 'all_to_all' else 0)
        expected_write=parents[write.access_id].offset_bytes+(source_index*(payload if operation == 'all_gather' else chunk) if operation != 'send_recv' else 0)
        if src.offset_bytes != expected_read or dst.offset_bytes != expected_write:
            raise ValueError('collective copy must preserve actual source/destination peer segment placement')
    if operation != 'send_recv' and pairs != {(source,destination) for source in ranks for destination in ranks}:
        raise ValueError('collective copy peer coverage incomplete')
    return payload,len(ranks),operation


def block_fp8_source_demands(program):
    """Read/write phase work from the unique tile source, without a new timing law."""
    from src.kernel.memory_access import MemoryAccessProgram, Semantics
    if not isinstance(program, MemoryAccessProgram) or program.coverage.semantics is not Semantics.LOGICAL:
        raise TypeError("block-FP8 projection requires a logical source program")
    return block_fp8_tile_source_demands(program)


def block_fp8_tile_source_demands(program):
    """Compulsory stage work from the actual block-scale tile selections."""
    from src.kernel.memory_access import EventKind,Operation,TileDependency,requested_bytes,unique_by
    program.__post_init__()
    if program.accesses or len(program.tile_programs) != 1:
        raise ValueError('block-FP8 requires one actual phase/coordinate program')
    family, = program.tile_programs
    accesses = unique_by(family.accesses,'access_id'); views = unique_by(program.views,'view_id')
    roles = ('quantized','activation_scale','weight','weight_scale','output')
    main = {role:accesses[f'gemm.{role}'] for role in roles}
    if family.scope.op_ids != ('block_fp8',):
        raise ValueError('block-FP8 phase/operand roles must be unique and explicit')
    raw = 'quantize.input' in accesses
    expected = {f'gemm.{role}' for role in roles}
    if raw:expected |= {'quantize.input','quantize.quantized','quantize.activation_scale'}
    if set(accesses) != expected:
        raise ValueError('block-FP8 source payload or scale operand missing')
    source,weight,output = (views[main[role].view_id] for role in ('quantized','weight','output'))
    b,m,k = source.shape; wb,n,wk = weight.shape
    if (wb,wk) != (b,k) or output.shape != (b,m,n):
        raise ValueError('block-FP8 matrix/scale geometry mismatch')
    if tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != (('batch',b,1),('m',m,1),('n',n,128),('k',k,128)):
        raise ValueError('block-FP8 tile loop must retain row/N128/K128 recipe blocks')
    mappings = {'quantized':(('batch','m','k'),(1,1,1)),
        'activation_scale':(('batch','m','k'),(1,1,128)),
        'weight':(('batch','n','k'),(1,1,1)),'weight_scale':(('batch','n','k'),(1,128,128)),
        'output':(('batch','m','n'),(1,1,1))}
    for role,access in main.items():
        view = views[access.view_id]; mapping,divisors = mappings[role]
        is_output = role == 'output'
        if (access.operation is not (Operation.WRITE if is_output else Operation.READ)
                or access.iteration_axes != (('batch','m','n') if is_output else ('batch','m','n','k'))
                or access.view_axes != mapping or (access.view_axis_divisors or (1,)*3) != divisors
                or access.unmapped or access.indirect_target is not None):
            raise ValueError('block-FP8 tile must conserve actual grouped payload/scale selections')
    for role in ('quantized','weight'):
        if (views[main[role].view_id].dtype,views[main[role].view_id].element_bytes) != ('fp8',1):
            raise ValueError('block-FP8 payload storage must preserve its one-byte recipe')
    for role,shape in (('activation_scale',(b,m,-(-k//128))),('weight_scale',(b,-(-n//128),-(-k//128)))):
        view = views[main[role].view_id]
        if (view.shape,view.dtype,view.element_bytes) != (shape,'fp32',4):
            raise ValueError('block-FP8 scale storage or grouping mismatch')
    if (output.dtype,output.element_bytes) != ('bf16',2):
        raise ValueError('block-FP8 output storage must preserve BF16')
    from src.kernel.memory_block_fp8 import block_fp8_geometry_layouts
    # With one batch, the two approved first-axis strides address the same
    # elements. Multiple batches require the actual BHD output layout.
    batched_output = b > 1 or output.strides_bytes == (n*2,b*n*2,2)
    layouts=block_fp8_geometry_layouts(b,m,n,k,batched_output=batched_output,internal_quantizer=raw)
    for role,layout in layouts:
        access=accesses['quantize.input'] if role=='input' else main[role]
        view=views[access.view_id]
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes) != layout:
            raise ValueError('block-FP8 tile storage must retain actual payload/scale/output strides')
    required = tuple(TileDependency(f'gemm.{role}',EventKind.READ_READY,('k',)) for role in roles[:-1])
    if main['output'].dependencies != required:
        raise ValueError('block-FP8 phase lacks actual payload/scale readiness')
    quantizer_bytes = 0
    if raw:
        incoming = accesses['quantize.input']; input_view = views[incoming.view_id]
        if ((input_view.shape,input_view.dtype,input_view.element_bytes) != ((b,m,k),'bf16',2)
                or incoming.operation is not Operation.READ or incoming.iteration_axes != ('batch','m','k')
                or incoming.view_axes != ('batch','m','k') or incoming.view_axis_divisors):
            raise ValueError('block-FP8 wrapper quantizer requires its actual BF16 group input')
        for role in ('quantized','activation_scale'):
            producer = accesses[f'quantize.{role}']
            if (producer.view_id != main[role].view_id or producer.operation is not Operation.WRITE
                    or producer.iteration_axes != ('batch','m','k') or producer.view_axes != main[role].view_axes
                    or (producer.view_axis_divisors or (1,)*3) != (main[role].view_axis_divisors or (1,)*3)
                    or producer.dependencies != (TileDependency('quantize.input',EventKind.READ_READY,()),)
                    or main[role].dependencies != (TileDependency(producer.access_id,EventKind.WRITE_VISIBLE,()),)):
                raise ValueError('block-FP8 phase lacks actual payload/scale readiness')
        quantizer_bytes = sum(requested_bytes(views[accesses[f'quantize.{role}'].view_id].region())
            for role in ('input','quantized','activation_scale'))
    # Each role partitions the complete declared view; replay across M/N
    # remains in logical_requested_bytes, never inflated into compulsory work.
    return {'geometry':(b,m,n,k),'weight_element_bytes':weight.element_bytes,'output_element_bytes':output.element_bytes,
        'main_read_bytes':sum(requested_bytes(views[main[role].view_id].region()) for role in roles[:-1]),
        'main_write_bytes':requested_bytes(output.region()),'quantizer_total_bytes':quantizer_bytes}


def fp8_group_quant_source_demands(program):
    """Keep the unique grouped source, its storage ABI and input causality."""
    from src.kernel.memory_access import EventKind, Operation, Semantics, TileDependency, TensorView, LogicalSelectionView, unique_by
    from src.kernel.memory_block_fp8 import fp8_group_quant_geometry_layouts
    program.__post_init__()
    if (program.coverage.semantics is not Semantics.LOGICAL or program.accesses
            or program.scopes or len(program.tile_programs) != 1):
        raise ValueError('FP8 group quantizer requires its unique actual row/group tile source')
    family, = program.tile_programs
    fused = family.scope.op_ids[0] == 'fused_silu_fp8_group_quant'
    if (len(family.scope.op_ids) != 2 or family.scope.op_ids[0] not in (
            'fp8_group_quant', 'fused_silu_fp8_group_quant') or len(family.accesses) != 3):
        raise ValueError('FP8 group quantizer requires declared scale layout, one input read and two output writes')
    read,payload,scale = family.accesses
    views = unique_by(program.views,'view_id')
    source,result,scales = (views[access.view_id] for access in family.accesses)
    known_rows=family.scope.op_ids[1]!='unknown_rows'
    if fused and known_rows:
        raise NotImplementedError('fused SiLU quantizer currently binds the flat grouped source')
    expected_result_shape = ((source.shape[0], source.shape[1] // 2)
        if fused and len(source.shape) == 2 else source.shape)
    if (not all(isinstance(view,TensorView) for view in (source,result))
            or len(source.shape) != (3 if known_rows else 2)
            or result.shape != expected_result_shape):
        raise ValueError('FP8 group quantizer must expose complete actual group payloads')
    if known_rows:
        rows,groups,width=source.shape;axes=(('row',rows,1),('group',groups,1))
        iterations=('row','group');payload_mapping=('row','group',None);scale_mapping=('row','group')
    else:
        groups,width=source.shape;rows=None;axes=(('group',groups,1),)
        iterations=('group',);payload_mapping=('group',None);scale_mapping=('group',)
    if tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != axes:
        raise ValueError('FP8 group quantizer grouping or storage width mismatch')
    layouts=fp8_group_quant_geometry_layouts(rows,groups*width,width,source.dtype,source.element_bytes,
        family.scope.op_ids[1] if known_rows else None)
    for view,(_,layout) in zip((source,result,scales),layouts,strict=True):
        shape,strides,dtype,element_bytes=layout
        if known_rows and view is not scales:
            shape=(rows,groups,width);strides=(strides[0],width*strides[1],strides[1])
        if fused and view is result:
            shape,strides = (groups,width // 2),(width // 2,1)
        if strides is None:
            allocation=unique_by(program.allocations,'allocation_id')[view.allocation_id]
            matches=(isinstance(view,LogicalSelectionView) and (view.shape,view.dtype,view.element_bytes)==(shape,dtype,element_bytes)
                and view.parent_view_id is None and not view.coordinate_map and allocation.size_bytes is None)
        else:matches=(isinstance(view,TensorView) and (view.shape,view.strides_bytes,view.dtype,view.element_bytes)==(shape,strides,dtype,element_bytes))
        if not matches:
            raise ValueError('FP8 group quantizer must retain actual payload/scale strides and storage')
    if len({view.allocation_id for view in (source,result,scales)})!=3:
        raise ValueError('FP8 group quantizer requires distinct input/payload/scale storage')
    for access,mapping,operation in ((read,payload_mapping,Operation.READ),
            (payload,payload_mapping,Operation.WRITE),(scale,scale_mapping,Operation.WRITE)):
        if (access.operation is not operation or access.iteration_axes != iterations
                or access.view_axes != mapping or access.unmapped != (access is scale and not known_rows) or access.indirect_target is not None
                or access.view_axis_divisors or access.view_axis_padding or access.constant_value is not None):
            raise ValueError('FP8 group quantizer must conserve actual group selections')
    required = (TileDependency(read.access_id,EventKind.READ_READY,()),)
    if payload.dependencies != required or scale.dependencies != required:
        raise ValueError('FP8 group quantizer write lacks actual input READ_READY causality')
    return logical_requested_bytes(program,Operation.READ),logical_requested_bytes(program,Operation.WRITE)


def logical_paged_kv_update_bytes(program):
    """Validate the declared token/slot copy ABI; destinations stay unaddressed."""
    from src.kernel.memory_access import (
        EventKind, Indirection, KVComponent, MemoryAccessProgram, Operation, Semantics,
        TensorView, TileDependency, unique_by, tile_template_requested_bytes,
    )
    if not isinstance(program,MemoryAccessProgram):
        raise TypeError('paged KV projection requires a typed source program')
    program.__post_init__()
    if (program.coverage.semantics is not Semantics.LOGICAL or program.coverage.executable
            or program.accesses or program.scopes or len(program.tile_programs)!=1):
        raise ValueError('paged KV requires its unique nonexecutable token-slot tile source')
    family,=program.tile_programs;header=family.scope.op_ids
    if len(header)!=10 or header[:2]!=('kv_cache','paged_update'):
        raise ValueError('paged KV source must retain its declared geometry, mapping and slot ABI')
    mapping=Indirection(header[2]);tokens,heads,dimension=map(int,header[3:6])
    dtype=header[6];width,index_width,passes=map(int,header[7:10])
    if (min(tokens,heads,dimension,width)<=0 or index_width not in (4,8) or passes not in (1,2)
            or mapping not in (Indirection.PAGED_KV_HND,Indirection.PAGED_KV_NHD)
            or tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes)!=(('token',tokens,1),)):
        raise ValueError('paged KV source requires its actual independent token-slot partition')
    accesses=unique_by(family.accesses,'access_id');views=unique_by(program.views,'view_id')
    slot_ids=tuple(f'slots{index}' for index in range(passes))
    if set(accesses)!={*slot_ids,'read_key','write_key','read_value','write_value'}:
        raise ValueError('paged KV tile lost a declared payload/slot/store access')
    for access in family.accesses:
        if (access.iteration_axes!=('token',) or access.unmapped or access.view_axis_divisors
                or access.view_axis_padding or access.constant_value is not None):
            raise ValueError('paged KV tiles must conserve actual token selections')
    for name in slot_ids:
        access=accesses[name];view=views[access.view_id]
        if (not isinstance(view,TensorView) or (view.shape,view.dtype,view.element_bytes)
                !=((tokens,),'int32' if index_width==4 else 'int64',index_width)
                or access.operation is not Operation.READ or access.view_axes!=('token',)
                or access.indirect_target is not None or access.dependencies or len(access.input_ready)!=1):
            raise ValueError('paged KV source lost its declared integer slot reads')
    if len({accesses[name].view_id for name in slot_ids})!=1:
        raise ValueError('paged KV repeated slot reads must retain the same actual integer input')
    for index,(role,component) in enumerate((('key',KVComponent.KEY),('value',KVComponent.VALUE))):
        read=accesses[f'read_{role}'];store=accesses[f'write_{role}'];view=views[read.view_id]
        slot_id=slot_ids[index if passes==2 else 0]
        if (not isinstance(view,TensorView) or (view.shape,view.dtype,view.element_bytes)
                !=((tokens,heads,dimension),dtype,width) or read.operation is not Operation.READ
                or read.view_axes!=('token',None,None) or read.indirect_target is not None
                or read.dependencies or len(read.input_ready)!=1):
            raise ValueError('paged KV source lost its actual payload shape/storage/read')
        target=store.indirect_target
        if target is None:raise ValueError('paged KV tile must retain its unknown slot transform')
        if target.value_source_access_ids:raise ValueError('paged KV copy cannot substitute computed value sources')
        if (store.operation is not Operation.WRITE or store.view_id!=read.view_id or store.view_axes!=read.view_axes
                or target.mapping is not mapping or target.component is not component or target.index_access_id!=slot_id):
            raise ValueError('paged KV stores must retain declared mapping, component and slot pass')
        if store.dependencies!=(TileDependency(read.access_id,EventKind.READ_READY,()),
                TileDependency(slot_id,EventKind.READ_READY,())):
            raise ValueError('paged KV tile requires its actual matching payload and slot READ_READY')
    tensor=sum(tile_template_requested_bytes(program,family,accesses[f'read_{role}']) for role in ('key','value'))
    slots=sum(tile_template_requested_bytes(program,family,accesses[name]) for name in slot_ids)
    write=sum(tile_template_requested_bytes(program,family,accesses[f'write_{role}']) for role in ('key','value'))
    if tensor!=write:raise ValueError('paged KV tile payload bytes do not conserve')
    return tensor,slots,write


def logical_gemm_operand_geometry(program, *, batched=False):
    """Project the unique C=A@B tile source to operand extents.

    This is logical geometry, not a physical address trace or cache result.
    Tile traversal, padding and replacement remain separately declared launch
    projections. Noncontiguous/aliased inputs need a different cache projection.
    """
    from src.kernel.memory_access import (
        MemoryAccessProgram, Operation, unique_by, EventKind,
    )
    from src.kernel.memory_access_builder import MemoryAccessWork
    if isinstance(program,MemoryAccessWork):
        family=program
        operations=program.operation_ids
    elif not isinstance(program, MemoryAccessProgram):
        raise TypeError("GEMM geometry requires a typed source program")
    else:
        if not program.tile_programs or program.accesses:
            raise ValueError('GEMM geometry requires its unique tile source')
        program.__post_init__()
        if len(program.tile_programs) != 1:
            raise ValueError('GEMM projection requires its single actual tile loop')
        family, = program.tile_programs
        operations=family.scope.op_ids
    views = unique_by(program.views,'view_id')
    if operations != ('gemm',) or len(family.accesses) != 3:
        raise ValueError('GEMM tile source requires two operands and its reduced output')
    a,b,c=family.accesses
    prefix=('batch',) if batched else ()
    if ((a.operation,b.operation,c.operation) != (Operation.READ,Operation.READ,Operation.WRITE)
            or a.view_axes!=(*prefix,'m','k') or c.view_axes!=(*prefix,'m','n')
            or b.view_axes not in ((*prefix,'k','n'),(*prefix,'n','k'))
            or a.iteration_axes != (*prefix,'m','n','k') or b.iteration_axes != a.iteration_axes
            or c.iteration_axes != (*prefix,'m','n')
            or {(dep.access_id,dep.kind,dep.reduction_axes) for dep in c.dependencies}
                != {(a.access_id,EventKind.READ_READY,('k',)),(b.access_id,EventKind.READ_READY,('k',))}):
        raise ValueError('GEMM projection lost actual tiled operands or K reduction readiness')
    operands=tuple(views[access.view_id] for access in (a,b,c))
    if len({view.allocation_id for view in operands}) != 3:
        raise NotImplementedError('GEMM cache projection requires independent operands')
    axes=unique_by(family.axes,'axis_id')
    if set(axes) != {*prefix,'m','n','k'}:
        raise ValueError('GEMM projection requires its exact tile axes')
    geometry=(axes['m'].extent,axes['n'].extent,axes['k'].extent,*(view.element_bytes for view in operands))
    return (axes['batch'].extent,*geometry) if batched else geometry


def moe_abs_source_work(program):
    """INT32 routing-index Abs preserves elements and its actual read edge."""
    if not program.tile_programs or program.accesses:
        raise ValueError('MoE Abs requires its unique tile source')
    work = scalar_tile_source_work(program)
    if work['scope'].op_ids != ('moe_abs',) or len(work['reads']) != 1 or len(work['writes']) != 1:
        raise ValueError('MoE Abs requires one unary input/output source')
    source, = work['reads']; destination, = work['writes']
    if (source.allocation_id == destination.allocation_id or any(
            (view.dtype,view.element_bytes,view.strides_bytes) != ('int32',4,(4,))
            for view in (source,destination)) or len(work['shape']) != 1):
        raise ValueError('MoE Abs requires distinct contiguous INT32 routing-row parents')
    return {'elements':work['shape'][0],'read_bytes':work['read_bytes'],'write_bytes':work['write_bytes']}


def short_context_topk_tile_demands(program, *, expected_kernel=None):
    from src.kernel.memory_access import EventKind, Operation, TileDependency, unique_by
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs) != 1:
        raise ValueError("short-context fill requires one independent row loop")
    family, = program.tile_programs
    if (family.scope.agent != "logical-short-context-fill" or len(family.scope.op_ids) != 2 or family.scope.op_ids[0] != "dsv4_short_context_topk"
            or len(family.accesses) != 2 or len(family.axes) != 1 or family.causal_prefixes or family.partitions):
        raise ValueError("short-context fill requires its row source scope")
    read, write = family.accesses
    if ((read.access_id,write.access_id)!=('position','indices') or len(read.input_ready)!=1
            or read.dependencies or write.input_ready):
        raise ValueError('short-context source must preserve its actual dynamic row')
    for access in (read,write):
        if (access.constant_value is not None or access.predicate is not None or access.value_index is not None
                or access.atomic_add is not None or access.scope is not None or access.view_axis_padding):
            raise ValueError('short-context source must preserve its actual dynamic row')
    views = unique_by(program.views, "view_id")
    if len(views)!=2:raise ValueError('short-context source has unrelated views')
    position, indices = views[read.view_id], views[write.view_id]
    axis, = family.axes
    ratio = int(family.scope.op_ids[1])
    if ratio!=4 or family.scope.op_ids[1]!='4':
        raise ValueError('short-context source requires its supported C4 compression ratio')
    if expected_kernel is not None:
        from src.kernel.kernel_spec import KernelKind
        kernel = expected_kernel
        if (kernel.kind is not KernelKind.VECTOR or kernel.variant!='dsv4_short_context_topk'
                or kernel.execution_support!='supported' or kernel.hardware_id!=program.hardware_id
                or kernel.dtype!='int32' or kernel.attributes['position_bytes']!=8
                or (kernel.shape['seq_len'],kernel.shape['topk_capacity'])!=indices.shape
                or kernel.shape['compress_ratio']!=ratio):
            raise ValueError('Source differs from its actual short-context invocation')
    if (len(position.shape) != 1 or len(indices.shape) != 2 or position.shape != indices.shape[:1]
            or position.allocation_id == indices.allocation_id
            or (position.dtype, position.element_bytes, position.strides_bytes) != ("int64", 8, (8,))
            or (indices.dtype, indices.element_bytes, indices.strides_bytes) != ("int32", 4, (indices.shape[1]*4,4))
            or (axis.axis_id, axis.extent, axis.block) != ("row", position.shape[0], 1)):
        raise ValueError("short-context fill must preserve actual positions and full INT32 capacity")
    for access, operation, mapping in ((read, Operation.READ, ("row",)), (write, Operation.WRITE, ("row", None))):
        if (access.operation is not operation or access.iteration_axes != ("row",) or access.view_axes != mapping
                or access.view_axis_divisors or access.indirect_target is not None):
            raise ValueError("short-context fill must preserve each independent position/output row")
    if write.dependencies != (TileDependency(read.access_id, EventKind.READ_READY, ()),):
        raise ValueError("short-context fill lacks actual position read causality for its row")
    return {"geometry": indices.shape, "read_bytes": logical_requested_bytes(program, Operation.READ),
        "write_bytes": logical_requested_bytes(program, Operation.WRITE)}


def softmax_source_work(program):
    """Known row reductions or explicitly unbound logical grouping, never coarse."""
    from src.kernel.memory_access import EventKind,Operation,TileDependency,unique_by
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:
        raise ValueError('softmax requires its unique compressed reduction source')
    family,=program.tile_programs
    if family.scope.agent=='logical-softmax-rows':return softmax_row_tile_work(program)
    if (family.scope.agent!='logical-softmax-partition' or family.scope.op_ids!=('softmax',)
            or len(family.axes)!=1 or len(family.accesses)!=2 or family.causal_prefixes
            or len(family.partitions)!=1):
        raise ValueError('flat softmax requires its explicit unbound partition interface')
    read,write=family.accesses;views=unique_by(program.views,'view_id')
    source,destination=views[read.view_id],views[write.view_id];axis,=family.axes;partition,=family.partitions
    if (len(source.shape)!=1 or source.shape!=destination.shape or source.dtype!=destination.dtype
            or source.element_bytes!=destination.element_bytes or axis.axis_id!='element'
            or (axis.extent,axis.block)!=(source.shape[0],1) or partition.axis_id!='element'
            or len(views)!=len({read.view_id,write.view_id})):
        raise ValueError('flat softmax must conserve its complete active input/output extent')
    if source.allocation_id==destination.allocation_id and source!=destination:
        # Distinct view identifiers may still name exactly the same in-place
        # region. Shifted/overlapping aliases need extra inter-group evidence.
        if source.region()!=destination.region():raise NotImplementedError('softmax cross-group aliases require their actual source schedule')
    for access,operation in ((read,Operation.READ),(write,Operation.WRITE)):
        if (access.operation is not operation or access.iteration_axes!=('element',) or access.view_axes!=('element',)
                or access.view_axis_divisors or access.view_axis_padding or access.constant_value is not None
                or access.indirect_target is not None or access.scope is not None):
            raise ValueError('flat softmax must retain one actual scalar selection per logical member')
    if (read.dependencies or len(read.input_ready)!=1 or write.input_ready
            or write.dependencies!=(TileDependency('input',EventKind.READ_READY,('element',),partition_axis='element'),)
            or read.access_id!='input' or write.access_id!='output'):
        raise ValueError('flat softmax output requires its own complete input equivalence class READ_READY')
    return {'rows':None,'columns':None,'element_count':source.shape[0],
        'read_bytes':logical_requested_bytes(program,Operation.READ),'write_bytes':logical_requested_bytes(program,Operation.WRITE)}


def softmax_row_tile_work(program):
    """Whole-row reduction interface, without native pass or DRAM assumptions."""
    from src.kernel.memory_access import EventKind, Operation, TileDependency, unique_by
    program.__post_init__()
    if len(program.tile_programs) != 1 or program.accesses:
        raise ValueError("softmax requires one row tile source")
    family, = program.tile_programs
    if (family.scope.agent != "logical-softmax-rows" or family.scope.op_ids != ("softmax",)
            or len(family.accesses) != 2 or len(family.axes) != 1):
        raise ValueError("softmax requires its explicit row reduction scope")
    read, write = family.accesses
    views = unique_by(program.views, "view_id")
    source, destination = views[read.view_id], views[write.view_id]
    axis, = family.axes
    if (len(source.shape) != 2 or source.shape != destination.shape
            or (source.dtype, source.element_bytes) != (destination.dtype, destination.element_bytes)
            or (axis.axis_id, axis.extent, axis.block) != ("row", source.shape[0], 1)):
        raise ValueError("softmax must preserve actual row/column geometry and dtype")
    for access, operation in ((read, Operation.READ), (write, Operation.WRITE)):
        view = views[access.view_id]
        if (access.operation is not operation or access.iteration_axes != ("row",)
                or access.view_axes != ("row", None) or access.view_axis_divisors
                or access.indirect_target is not None):
            raise ValueError("softmax tile must consume or publish a complete independent row")
        if view.strides_bytes[0] < (view.shape[1] - 1) * view.strides_bytes[1] + view.element_bytes:
            raise NotImplementedError("overlapping softmax rows require an explicit source schedule")
    if write.dependencies != (TileDependency(read.access_id, EventKind.READ_READY, ()),):
        raise ValueError("softmax output row requires its actual complete input READ_READY")
    return {"rows": source.shape[0], "columns": source.shape[1],
        "read_bytes": logical_requested_bytes(program, Operation.READ),
        "write_bytes": logical_requested_bytes(program, Operation.WRITE)}


def scalar_tile_source_work(program):
    """Validate independent source coordinates and actual read-to-write edges."""
    from src.kernel.memory_access import EventKind,Operation,TileDependency,unique_by
    from src.kernel.memory_access_builder import MemoryAccessWork
    direct=isinstance(program,MemoryAccessWork)
    if direct:family=program
    else:
        program.__post_init__()
        if program.accesses or program.scopes or len(program.tile_programs) != 1:
            raise ValueError("scalar source requires one independent coordinate loop")
        family,=program.tile_programs
    if not direct and (family.scope.agent != "logical-elementwise" or family.causal_prefixes or family.partitions):
        raise ValueError("scalar source requires its explicit algorithm scope")
    views = unique_by(program.views,"view_id")
    reads = tuple(access for access in family.accesses if access.operation is Operation.READ)
    writes = tuple(access for access in family.accesses if access.operation is Operation.WRITE)
    header=program.operation_ids if direct else family.scope.op_ids
    binary_variants=('activation','silu','moe_act_and_mul','moe_swiglu')
    unary_variants=('gemm_reduce','gemm_elementwise','moe_abs','moe_transpose','moe_cast')
    if (header in tuple((variant,) for variant in (*binary_variants,*unary_variants))
            or header[:1]==('gemm_elementwise',)):
        expected_reads=2 if header[0] in binary_variants else 1
        if len(reads)!=expected_reads or len(writes)!=1:
            raise ValueError('scalar source must retain every declared algorithm operand and output')
    shape = views[(writes or reads)[0].view_id].shape
    names = tuple(f"coordinate{index}" for index in range(len(shape)))
    if (tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes)
            != tuple((name,dimension,1) for name,dimension in zip(names,shape,strict=True))):
        raise ValueError("scalar source must retain independent exact output coordinates")
    required = tuple(TileDependency(access.access_id,EventKind.READ_READY,()) for access in reads)
    for access in family.accesses:
        view = views[access.view_id]
        reduction = header == ("gemm_reduce",) and access.operation is Operation.READ
        mapping = (None,*names,*((None,) if len(view.shape) == 3 else ())) if reduction else names
        if (access.iteration_axes != names or access.view_axes != mapping or access.indirect_target is not None
                or access.view_axis_divisors or access.view_axis_padding or access.constant_value is not None
                or access.scope is not None
                or (view.shape[1:2] != shape if reduction else view.shape != shape)):
            raise ValueError("scalar source operand coordinates differ from actual output work")
        if (access.operation is Operation.READ and (len(access.input_ready)!=(0 if direct else 1) or access.dependencies)
                or access.operation is Operation.WRITE and access.input_ready):
            raise ValueError('scalar source must retain actual operand readiness and independent publication')
        if access.operation is Operation.WRITE and access.dependencies != required:
            raise ValueError("scalar output requires actual input READ_READY for its own coordinate")
    return {'scope':None if direct else family.scope,'reads':tuple(views[access.view_id] for access in reads),
        'writes':tuple(views[access.view_id] for access in writes),'shape':shape,
        'read_bytes':logical_requested_bytes(program,Operation.READ),'write_bytes':logical_requested_bytes(program,Operation.WRITE)}


def logical_requested_bytes(program, operation, *, value_limits=None) -> int:
    """Exact logical work; unknown scalar predicates require a value binding."""
    from src.kernel.memory_access_builder import MemoryAccessWork,LogicalMemoryInvocation
    if isinstance(program,(MemoryAccessWork,LogicalMemoryInvocation)):
        if value_limits is not None:raise ValueError('Memory work has no declared value-limit operands')
        return program.requested_bytes(operation)
    from src.kernel.memory_access import _tile_value_limit_bindings,_tile_limited_requested_bytes,unique_by
    total=logical_candidate_requested_bytes(program,operation)
    needed=any(access.operation is operation and any(axis.value_limit_access_id is not None and axis.axis_id in access.iteration_axes
        for axis in family.axes) for family in program.tile_programs for access in family.accesses)
    limits=_tile_value_limit_bindings(program,value_limits,required=needed)
    if value_limits:
        views=unique_by(program.views,'view_id')
        for family in program.tile_programs:
            axes=unique_by(family.axes,'axis_id');bound=limits[family.tile_program_id]
            total+=sum(_tile_limited_requested_bytes(views,axes,family,access,bound)
                -_tile_limited_requested_bytes(views,axes,family,access,{})
                for access in family.accesses if access.operation is operation)
    if (any(access.operation is operation and access.predicate is not None for access in program.accesses)
            or any(access.operation is operation and access.predicate is not None
                for family in program.tile_programs for access in family.accesses)):
        raise ValueError('exact logical bytes require bound read-value predicates; candidate bytes are an explicit all-valid envelope')
    return total


def logical_candidate_requested_bytes(program, operation) -> int:
    """Untimed logical work by direction, without a cache/DRAM assumption."""
    from src.kernel.memory_access import (
        MemoryAccessProgram, Operation, Semantics, requested_bytes,
    )
    if not isinstance(program, MemoryAccessProgram) or not isinstance(operation, Operation):
        raise TypeError("logical byte projection requires a typed program and operation")
    if program.coverage.semantics is not Semantics.LOGICAL:
        raise ValueError("requested logical bytes cannot consume issued/transaction semantics")
    from src.kernel.memory_access import _tile_template_requested_bytes, unique_by
    views = unique_by(program.views,'view_id')
    total = sum(requested_bytes(access.region)*access.repeat for access in program.accesses if access.operation is operation)
    for family in program.tile_programs:
        axes = unique_by(family.axes,'axis_id')
        total += sum(_tile_template_requested_bytes(views,axes,family,access)
            for access in family.accesses if access.operation is operation)
    return total


def logical_unique_bytes(program,operation):
    """Distinct allocation-relative selections; never a DRAM/cache inference."""
    from src.kernel.memory_access_builder import MemoryAccessWork,LogicalMemoryInvocation
    if isinstance(program,(MemoryAccessWork,LogicalMemoryInvocation)):return program.unique_bytes(operation)
    from src.kernel.memory_access import unique_by,_unique_region_bytes,Unmapped
    logical_requested_bytes(program,operation)
    views=unique_by(program.views,'view_id')
    regions={allocation.allocation_id:[] for allocation in program.allocations}
    for access in program.accesses:
        if access.operation is operation:regions[access.allocation_id].append(access.region)
    for family in program.tile_programs:
        for access in family.accesses:
            if access.operation is not operation:continue
            view=views[access.view_id]
            target=view.allocation_id if access.indirect_target is None else access.indirect_target.allocation_id
            region=(Unmapped(view.shape,view.dtype,view.element_bytes,family.tile_program_id)
                if access.unmapped or access.indirect_target is not None else view.region())
            regions[target].append(region)
    values=tuple(_unique_region_bytes(items) for items in regions.values())
    if any(value is None for value in values):
        raise NotImplementedError('unique selection footprint is unavailable for unresolved tile mappings')
    return sum(values)


def logical_view_requested_bytes(program, view_id, operation) -> int:
    """Logical requests to one declared view, including explicit rereads."""
    from src.kernel.memory_access_builder import MemoryAccessWork
    if isinstance(program,MemoryAccessWork):return program.requested_bytes(operation,view_id)
    from src.kernel.memory_access import requested_bytes, unique_by
    # Reuse the public projection's typed/semantic boundary checks.
    logical_requested_bytes(program, operation)
    views = unique_by(program.views, "view_id")
    if view_id not in views:
        raise ValueError("logical requested-byte projection references an unknown view")
    accesses = tuple(access for access in program.accesses
                     if access.view_id == view_id and access.operation is operation)
    from src.kernel.memory_access import tile_template_requested_bytes
    templates=tuple((family,access) for family in program.tile_programs for access in family.accesses
        if access.view_id == view_id and access.operation is operation)
    if not accesses and not templates:
        raise ValueError("logical view has no requests in the selected direction")
    return (sum(requested_bytes(access.region)*access.repeat for access in accesses)
        +sum(tile_template_requested_bytes(program,family,access) for family,access in templates))


def logical_view_unique_bytes(program,view_id,operation):
    """Distinct selections for one operand; never a cache/DRAM projection."""
    from src.kernel.memory_access_builder import MemoryAccessWork
    if isinstance(program,MemoryAccessWork):return program.unique_bytes(operation,view_id)
    from src.kernel.memory_access import unique_by,_unique_region_bytes
    logical_view_requested_bytes(program,view_id,operation)
    view=unique_by(program.views,'view_id')[view_id]
    regions=[access.region for access in program.accesses if access.view_id == view_id and access.operation is operation]
    for family in program.tile_programs:
        for access in family.accesses:
            if access.view_id != view_id or access.operation is not operation:continue
            if access.unmapped or access.indirect_target is not None:
                raise NotImplementedError('unique operand footprint requires known selections')
            regions.append(view.region())
    result=_unique_region_bytes(regions)
    if result is None:raise NotImplementedError('unique operand footprint requires known selections')
    return result

from src.kernel.model_contract import (
    MemoryAccessClassDiagnostic,
    MemoryAccessClassService,
    MemoryArbitrationDiagnostic,
    MemoryDiagnostics,
    MemoryHierarchy,
    MemoryNodeDiagnostic,
    MemoryPathDemand,
    MemoryService,
    QueueDiagnostic,
)


def memory_hierarchy_for_path(
        hardware, demand: MemoryPathDemand, *, sm_freq_hz: float,
        dram_freq_hz: float, dram_capacity_multiplier: float) -> MemoryHierarchy:
    """Instantiate one path's hardware services from HWSpec capabilities."""

    if not isinstance(demand, MemoryPathDemand):
        raise TypeError(
            "memory_hierarchy_for_path demand must be MemoryPathDemand")
    for name, value in (
            ("sm_freq_hz", sm_freq_hz),
            ("dram_freq_hz", dram_freq_hz),
            ("dram_capacity_multiplier", dram_capacity_multiplier)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    memory_model = hardware.kernel_memory_model
    services = []
    if memory_model == "gpu_mshr":
        l2_latency_s = hardware.l2_latency_ns * 1e-9
        dram_increment_s = (
            hardware.dram_latency_ns - hardware.l2_latency_ns) * 1e-9
        if l2_latency_s <= 0 or dram_increment_s < 0:
            raise ValueError(
                "GPU memory hierarchy requires a non-decreasing latency path")
        for node in demand.nodes:
            resource = hardware.resource_by_role(node.resource_role)
            byte_capacity = resource.effective_capacity(
                demand.source.active_units,
                sm_freq_hz=sm_freq_hz,
                dram_freq_hz=dram_freq_hz)
            if node.resource_role == "dram":
                byte_capacity = hardware.dram_capacity_for_access(
                    demand.access_kind,
                    active_units=demand.source.active_units,
                    sm_freq_hz=sm_freq_hz,
                    dram_freq_hz=dram_freq_hz,
                )
            byte_capacity *= node.independent_service_streams
            if node.capacity_ceiling_resource_role is not None:
                ceiling_resource = hardware.resource_by_role(
                    node.capacity_ceiling_resource_role)
                ceiling_capacity = ceiling_resource.effective_capacity(
                    demand.source.active_units,
                    sm_freq_hz=sm_freq_hz,
                    dram_freq_hz=dram_freq_hz)
                byte_capacity = min(byte_capacity, ceiling_capacity)
            if node.resource_role == "l2_cache":
                configured_service_cycles = hardware.memory_subsystem[
                    "l2_partition_service_cycles_bw"]
                if configured_service_cycles is None:
                    # A byte-capacity-only service is the only defensible model
                    # when partition count or transaction issue interval is not
                    # independently known.  Do not manufacture an M/D/1 queue
                    # from an estimated partition count.
                    services.append(MemoryService(
                        resource_role=node.resource_role,
                        byte_capacity_per_s=byte_capacity,
                        request_capacity_per_s=None,
                        unloaded_latency_s=l2_latency_s,
                        queue_model="none",
                        service_time_s=None,
                    ))
                else:
                    service_cycles = float(configured_service_cycles)
                    partitions = int(resource.num_partitions)
                    if service_cycles <= 0 or partitions <= 0:
                        raise ValueError(
                            "GPU L2 service cycles and partitions must be "
                            "positive when request queueing is enabled")
                    request_capacity = (
                        partitions * sm_freq_hz / service_cycles)
                    service_capacity = min(
                        byte_capacity / node.transaction_bytes,
                        request_capacity,
                    )
                    services.append(MemoryService(
                        resource_role=node.resource_role,
                        byte_capacity_per_s=byte_capacity,
                        request_capacity_per_s=request_capacity,
                        unloaded_latency_s=l2_latency_s,
                        queue_model="md1",
                        service_time_s=partitions / service_capacity,
                    ))
            elif node.resource_role == "dram":
                services.append(MemoryService(
                    resource_role=node.resource_role,
                    byte_capacity_per_s=(
                        byte_capacity * dram_capacity_multiplier),
                    request_capacity_per_s=None,
                    unloaded_latency_s=dram_increment_s,
                    queue_model="none",
                    service_time_s=None,
                ))
            else:
                raise ValueError(
                    "gpu_mshr path supports only l2_cache and dram, got "
                    f"{node.resource_role!r}")
        configured_frontend_rate = hardware.memory_subsystem[
            "lsu_peak_issue_rate_per_sm"]
        frontend_rate = (
            None if configured_frontend_rate is None else
            float(configured_frontend_rate) * sm_freq_hz)
        source_slots = hardware.source_slots_per_unit
    elif memory_model == "npu_tile":
        if demand.rate_unit != "bytes":
            raise ValueError(
                "npu_tile memory requires byte-only demand until physical "
                "MTE transaction geometry is independently established")
        if demand.access_kind == "read":
            allowed_roles = {
                "mte2", "l0a_buffer", "l0b_buffer", "l2_cache", "dram"}
        elif demand.access_kind == "write":
            allowed_roles = {"mte3", "l2_cache", "dram"}
        else:
            raise AssertionError(
                f"unhandled NPU access kind: {demand.access_kind!r}")
        for node in demand.nodes:
            if node.resource_role not in allowed_roles:
                raise ValueError(
                    f"npu_tile {demand.access_kind} path does not support "
                    f"role {node.resource_role!r}")
            resource = hardware.resource_by_role(node.resource_role)
            byte_capacity = resource.effective_capacity(
                demand.source.active_units,
                sm_freq_hz=sm_freq_hz,
                dram_freq_hz=dram_freq_hz)
            if node.resource_role == "dram":
                byte_capacity = hardware.dram_capacity_for_access(
                    demand.access_kind,
                    active_units=demand.source.active_units,
                    sm_freq_hz=sm_freq_hz,
                    dram_freq_hz=dram_freq_hz,
                )
            if node.resource_role in ("mte2", "mte3"):
                if resource.total_units <= 0:
                    raise ValueError(
                        f"{node.resource_role.upper()} resource has no "
                        "physical units")
                byte_capacity *= (
                    min(demand.source.active_units, resource.total_units)
                    / resource.total_units)
            elif node.resource_role == "dram":
                byte_capacity *= dram_capacity_multiplier
            elif node.resource_role == "l2_cache":
                pass
            elif node.resource_role in ("l0a_buffer", "l0b_buffer"):
                pass
            else:
                raise AssertionError(
                    f"unhandled validated NPU role: {node.resource_role!r}")
            byte_capacity *= node.independent_service_streams
            if node.capacity_ceiling_resource_role is not None:
                ceiling_resource = hardware.resource_by_role(
                    node.capacity_ceiling_resource_role)
                ceiling_capacity = ceiling_resource.effective_capacity(
                    demand.source.active_units,
                    sm_freq_hz=sm_freq_hz,
                    dram_freq_hz=dram_freq_hz)
                byte_capacity = min(byte_capacity, ceiling_capacity)
            services.append(MemoryService(
                resource_role=node.resource_role,
                byte_capacity_per_s=byte_capacity,
                request_capacity_per_s=None,
                # Retained 910B evidence establishes only aggregate INI byte
                # capacities.  Unknown first-request latency must not be
                # silently reconstructed from an unproven transfer quantum.
                unloaded_latency_s=0.0,
                queue_model="none",
                service_time_s=None,
            ))
        frontend_rate = None
        # No independent MTE2/MTE3 descriptor or request-source capacity is
        # available for the byte-only aggregate path.
        source_slots = None
    elif memory_model in ("pim_bank", "systolic_scratchpad"):
        raise NotImplementedError(
            f"structured memory hierarchy is not defined for {memory_model!r}")
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")
    return MemoryHierarchy(
        services=tuple(services),
        frontend_rate_per_unit_s=frontend_rate,
        source_slots_per_unit=source_slots,
        queues=(),
    )


def _memory_state(demand, hierarchy, services, queue_capacities, initial_rate,
                  *, include_open_queue_wait):
    """Evaluate one candidate initial request rate.

    ``include_open_queue_wait=False`` is used only for the exact capacity
    sufficiency test.  It asks whether finite source/credit populations cover
    the unloaded bandwidth-delay product; it never manufactures a finite wait
    at rho=1.
    """

    node_state = {}
    for node in demand.nodes:
        service = services[node.resource_role]
        gamma = node.transactions / demand.initial_transactions
        request_capacity = service.request_capacity_per_s
        byte_request_capacity = (
            service.byte_capacity_per_s / node.transaction_bytes)
        if request_capacity is None:
            service_capacity = byte_request_capacity
        else:
            service_capacity = min(byte_request_capacity, request_capacity)
        arrival_rate = gamma * initial_rate
        rho = arrival_rate / service_capacity
        if rho > 1.0 and not math.isclose(
                rho, 1.0, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"candidate rate exceeds {node.resource_role!r} capacity")
        rho = min(rho, 1.0)
        if not include_open_queue_wait or service.queue_model == "none":
            queue_wait = 0.0
        elif rho >= 1.0:
            queue_wait = math.inf
        elif service.queue_model == "md1":
            queue_wait = (
                rho / (2.0 * (1.0 - rho)) * service.service_time_s)
        else:
            raise AssertionError(
                f"unhandled queue model: {service.queue_model!r}")
        node_state[node.resource_role] = {
            "gamma": gamma,
            "service_capacity": service_capacity,
            "arrival_rate": arrival_rate,
            "rho": rho,
            "queue_wait": queue_wait,
            "increment_latency": service.unloaded_latency_s + queue_wait,
        }

    source_lifetime = sum(
        state["gamma"] * state["increment_latency"]
        for state in node_state.values())
    workload_slots = demand.source.workload_slots_per_unit
    hardware_slots = hierarchy.source_slots_per_unit
    if workload_slots is None:
        source_slots = None
    elif hardware_slots is None:
        source_slots = workload_slots
    else:
        source_slots = min(workload_slots, hardware_slots)
    if source_slots is None:
        source_capacity = None
    elif source_lifetime == 0.0:
        source_capacity = math.inf
    else:
        source_capacity = (
            demand.source.active_units * source_slots / source_lifetime)

    frontend_candidates = []
    if demand.source.frontend_rate_per_unit_s is not None:
        frontend_candidates.append(
            demand.source.frontend_rate_per_unit_s
            * demand.source.active_units)
    if hierarchy.frontend_rate_per_unit_s is not None:
        frontend_candidates.append(
            hierarchy.frontend_rate_per_unit_s * demand.source.active_units)
    frontend_capacity = (
        min(frontend_candidates) if frontend_candidates else None)

    queue_state = {}
    for queue_use in demand.queues:
        capacity = queue_capacities[queue_use.name]
        lifetime = sum(
            weight * node_state[role]["increment_latency"]
            for role, weight in queue_use.lifetime_weights)
        allocation_ratio = queue_use.allocations_per_initial_request
        if lifetime == 0.0:
            initial_capacity = math.inf
        else:
            initial_capacity = (
                capacity.max_entries / (allocation_ratio * lifetime))
        allocation_rate = allocation_ratio * initial_rate
        queue_state[queue_use.name] = {
            "allocation_rate": allocation_rate,
            "lifetime": lifetime,
            "used_entries": allocation_rate * lifetime,
            "initial_capacity": initial_capacity,
        }

    limits = []
    if source_capacity is not None:
        limits.append(("source", source_capacity))
    if frontend_capacity is not None:
        limits.append(("frontend", frontend_capacity))
    limits.extend(
        (f"queue:{name}", state["initial_capacity"])
        for name, state in queue_state.items())
    if limits:
        bottleneck, phi = min(limits, key=lambda item: item[1])
    else:
        bottleneck, phi = None, math.inf
    return {
        "nodes": node_state,
        "workload_slots": workload_slots,
        "hardware_slots": hardware_slots,
        "source_slots": source_slots,
        "source_lifetime": source_lifetime,
        "source_capacity": source_capacity,
        "frontend_capacity": frontend_capacity,
        "queues": queue_state,
        "phi": phi,
        "bottleneck": bottleneck,
    }


def _solve_byte_only_path(
        demand: MemoryPathDemand, hierarchy: MemoryHierarchy, services
        ) -> MemoryDiagnostics:
    """Solve a path whose physical transaction geometry is unknown."""

    if demand.rate_unit != "bytes":
        raise ValueError("byte-only solver requires a byte-rate demand")
    if hierarchy.frontend_rate_per_unit_s is not None:
        raise ValueError(
            "byte-only paths cannot consume a request frontend ceiling")
    if hierarchy.source_slots_per_unit is not None:
        raise ValueError(
            "byte-only paths cannot consume hardware request-source slots")
    if hierarchy.queues:
        raise ValueError("byte-only paths cannot consume request queues")

    initial_bytes = demand.nodes[0].bytes
    initial_capacity_by_role = {}
    gamma_by_role = {}
    for node in demand.nodes:
        service = services[node.resource_role]
        if service.request_capacity_per_s is not None:
            raise ValueError(
                "byte-only paths cannot consume a request-rate capacity")
        if service.queue_model != "none" or service.service_time_s is not None:
            raise ValueError(
                "byte-only paths cannot consume a request queue model")
        if service.unloaded_latency_s != 0.0:
            raise ValueError(
                "byte-only paths cannot consume first-request latency")
        gamma = node.bytes / initial_bytes
        gamma_by_role[node.resource_role] = gamma
        initial_capacity_by_role[node.resource_role] = (
            service.byte_capacity_per_s / gamma)

    lambda_capacity = min(initial_capacity_by_role.values())
    if not math.isfinite(lambda_capacity) or lambda_capacity <= 0:
        raise ValueError("byte-only memory path produced a non-positive capacity")
    saturated_roles = {
        role for role, value in initial_capacity_by_role.items()
        if math.isclose(value, lambda_capacity, rel_tol=1e-12, abs_tol=0.0)
    }
    node_diagnostics = []
    for node in demand.nodes:
        role = node.resource_role
        service = services[role]
        arrival_rate = gamma_by_role[role] * lambda_capacity
        rho = arrival_rate / service.byte_capacity_per_s
        node_diagnostics.append(MemoryNodeDiagnostic(
            resource_role=role,
            transactions=None,
            bytes=node.bytes,
            byte_capacity_per_s=service.byte_capacity_per_s,
            request_capacity_per_s=None,
            service_capacity_per_s=service.byte_capacity_per_s,
            arrival_rate_per_s=arrival_rate,
            rho=min(rho, 1.0),
            queue_wait_s=0.0,
            capacity_time_s=node.bytes / service.byte_capacity_per_s,
            saturated=role in saturated_roles,
            unloaded_latency_s=service.unloaded_latency_s,
        ))
    bottleneck = "+".join(
        f"service:{role}" for role in sorted(saturated_roles))
    return MemoryDiagnostics(
        path_name=demand.name,
        access_kind=demand.access_kind,
        rate_unit="bytes",
        initial_transactions=None,
        lambda_initial_per_s=lambda_capacity,
        lambda_capacity_per_s=lambda_capacity,
        workload_slots_per_unit=None,
        hardware_source_slots_per_unit=None,
        source_slots_per_unit=None,
        source_lifetime_s=0.0,
        source_rate_capacity_per_s=None,
        frontend_rate_capacity_per_s=None,
        steady_time_s=initial_bytes / lambda_capacity,
        first_request_latency_s=None,
        bottleneck=bottleneck,
        capacity_bound=True,
        nodes=tuple(node_diagnostics),
        queues=(),
    )


def solve_memory_path(
        demand: MemoryPathDemand,
        hierarchy: MemoryHierarchy) -> MemoryDiagnostics:
    """Solve the bounded scalar initial-request rate for one memory path.

    The service-capacity result is exact: no ``rho=0.99`` clamp and no
    ``(1-epsilon) * capacity`` prediction are used.  Open M/D/1 waiting is
    reported only for a finite-source/credit root strictly below capacity.
    """

    if not isinstance(demand, MemoryPathDemand):
        raise TypeError("solve_memory_path demand must be MemoryPathDemand")
    if not isinstance(hierarchy, MemoryHierarchy):
        raise TypeError("solve_memory_path hierarchy must be MemoryHierarchy")
    services = {value.resource_role: value for value in hierarchy.services}
    demand_roles = tuple(node.resource_role for node in demand.nodes)
    missing_services = set(demand_roles) - set(services)
    extra_services = set(services) - set(demand_roles)
    if missing_services or extra_services:
        raise ValueError(
            "memory path/service schema mismatch: "
            f"missing={sorted(missing_services)}, "
            f"extra={sorted(extra_services)}")
    queue_capacities = {value.name: value for value in hierarchy.queues}
    demand_queues = {value.name for value in demand.queues}
    missing_queues = demand_queues - set(queue_capacities)
    extra_queues = set(queue_capacities) - demand_queues
    if missing_queues or extra_queues:
        raise ValueError(
            "memory queue schema mismatch: "
            f"missing={sorted(missing_queues)}, extra={sorted(extra_queues)}")

    if demand.rate_unit == "bytes":
        return _solve_byte_only_path(demand, hierarchy, services)
    if demand.rate_unit != "transactions":
        raise AssertionError(f"unknown memory-path rate unit: {demand.rate_unit}")

    service_capacity_by_role = {}
    initial_capacity_by_role = {}
    for node in demand.nodes:
        service = services[node.resource_role]
        byte_capacity = (
            service.byte_capacity_per_s / node.transaction_bytes)
        if service.request_capacity_per_s is None:
            service_capacity = byte_capacity
        else:
            service_capacity = min(
                byte_capacity, service.request_capacity_per_s)
        gamma = node.transactions / demand.initial_transactions
        service_capacity_by_role[node.resource_role] = service_capacity
        initial_capacity_by_role[node.resource_role] = service_capacity / gamma
    lambda_capacity = min(initial_capacity_by_role.values())
    if not math.isfinite(lambda_capacity) or lambda_capacity <= 0:
        raise ValueError("memory path produced a non-positive capacity")

    # A closed finite population reaches service saturation exactly when its
    # unloaded bandwidth-delay product fits.  This is the physical capacity
    # branch; evaluating open M/D/1 at rho=1 would instead diverge.
    capacity_state = _memory_state(
        demand, hierarchy, services, queue_capacities, lambda_capacity,
        include_open_queue_wait=False)
    if capacity_state["phi"] >= lambda_capacity:
        solved_rate = lambda_capacity
        capacity_bound = True
        solved_state = capacity_state
    else:
        capacity_bound = False
        low = 0.0
        high = math.nextafter(lambda_capacity, 0.0)
        low_state = _memory_state(
            demand, hierarchy, services, queue_capacities, low,
            include_open_queue_wait=True)
        high_state = _memory_state(
            demand, hierarchy, services, queue_capacities, high,
            include_open_queue_wait=True)
        low_value = low - low_state["phi"]
        high_value = high - high_state["phi"]
        if not low_value < 0.0:
            raise RuntimeError("memory fixed point has no lower bracket")
        if not high_value > 0.0:
            raise RuntimeError("memory fixed point has no upper bracket")
        for _ in range(256):
            midpoint = (low + high) / 2.0
            midpoint_state = _memory_state(
                demand, hierarchy, services, queue_capacities, midpoint,
                include_open_queue_wait=True)
            midpoint_value = midpoint - midpoint_state["phi"]
            if midpoint_value > 0.0:
                high = midpoint
                high_state = midpoint_state
            else:
                low = midpoint
                low_state = midpoint_state
            numerical_tolerance = max(
                math.ulp(lambda_capacity) * 8.0,
                lambda_capacity * 1e-12)
            if high - low <= numerical_tolerance:
                break
        else:
            raise RuntimeError("memory fixed-point bisection did not converge")
        solved_rate = (low + high) / 2.0
        solved_state = _memory_state(
            demand, hierarchy, services, queue_capacities, solved_rate,
            include_open_queue_wait=True)

    saturated_roles = {
        role for role, value in initial_capacity_by_role.items()
        if capacity_bound and math.isclose(
            value, lambda_capacity, rel_tol=1e-12, abs_tol=0.0)
    }
    node_diagnostics = []
    for node in demand.nodes:
        role = node.resource_role
        service = services[role]
        state = solved_state["nodes"][role]
        if role in saturated_roles:
            queue_wait = None
            rho = 1.0
        elif service.queue_model == "md1":
            # The exact-capacity sufficiency branch intentionally evaluates
            # finite populations with unloaded lifetimes.  That does not make
            # a different, sub-saturated M/D/1 node queue-free: its diagnostic
            # still follows the analytical queue law at the solved arrival
            # rate.  Only the saturated open queue has undefined wait.
            rho = state["rho"]
            queue_wait = (
                rho / (2.0 * (1.0 - rho)) * service.service_time_s)
        else:
            queue_wait = state["queue_wait"]
            rho = state["rho"]
        node_diagnostics.append(MemoryNodeDiagnostic(
            resource_role=role,
            transactions=node.transactions,
            bytes=node.bytes,
            byte_capacity_per_s=service.byte_capacity_per_s,
            request_capacity_per_s=service.request_capacity_per_s,
            service_capacity_per_s=service_capacity_by_role[role],
            arrival_rate_per_s=state["arrival_rate"],
            rho=rho,
            queue_wait_s=queue_wait,
            capacity_time_s=(
                node.transactions / service_capacity_by_role[role]),
            saturated=role in saturated_roles,
            unloaded_latency_s=service.unloaded_latency_s,
        ))
    queue_diagnostics = tuple(
        QueueDiagnostic(
            name=name,
            allocation_rate_per_s=state["allocation_rate"],
            lifetime_s=state["lifetime"],
            used_entries=state["used_entries"],
            max_entries=queue_capacities[name].max_entries,
            initial_request_rate_capacity_per_s=state["initial_capacity"],
        )
        for name, state in solved_state["queues"].items()
    )
    first_request_latency = sum(
        (node.transactions / demand.initial_transactions)
        * services[node.resource_role].unloaded_latency_s
        for node in demand.nodes)
    if capacity_bound:
        if saturated_roles:
            bottleneck = "+".join(
                f"service:{role}" for role in sorted(saturated_roles))
        else:
            raise AssertionError(
                "capacity-bound memory path has no saturated service")
    else:
        bottleneck = solved_state["bottleneck"]
    return MemoryDiagnostics(
        path_name=demand.name,
        access_kind=demand.access_kind,
        rate_unit="transactions",
        initial_transactions=demand.initial_transactions,
        lambda_initial_per_s=solved_rate,
        lambda_capacity_per_s=lambda_capacity,
        workload_slots_per_unit=solved_state["workload_slots"],
        hardware_source_slots_per_unit=solved_state["hardware_slots"],
        source_slots_per_unit=solved_state["source_slots"],
        source_lifetime_s=solved_state["source_lifetime"],
        source_rate_capacity_per_s=solved_state["source_capacity"],
        frontend_rate_capacity_per_s=solved_state["frontend_capacity"],
        steady_time_s=demand.initial_transactions / solved_rate,
        first_request_latency_s=first_request_latency,
        bottleneck=bottleneck,
        capacity_bound=capacity_bound,
        nodes=tuple(node_diagnostics),
        queues=queue_diagnostics,
    )


def _access_class_capacity(service, grid_size):
    """Return the measured capacity at or below this launch's CTA count.

    The table is measured, ascending in grid, and never extrapolated: a launch
    smaller than the first measured grid keeps the entry capacity, and one
    larger than the last keeps the last measured point rather than projecting
    past the sweep.
    """

    table = service.logical_byte_capacity_per_s_by_grid_ctas
    if table is None or grid_size is None:
        return service.logical_byte_capacity_per_s
    resolved = None
    for grid, capacity in table:
        if grid <= grid_size:
            resolved = capacity
    if resolved is None:
        return service.logical_byte_capacity_per_s
    return resolved


def solve_memory_access_classes(
        demands, services, *,
        grid_size=None) -> tuple[MemoryAccessClassDiagnostic, ...]:
    """Resolve exact full-chip typed access-class capacities.

    These logical-byte services are independent lower bounds.  They do not
    mutate physical hierarchy traffic, consume DRAM arbitration capacity, or
    scale with the active execution-domain fraction.
    """

    if not isinstance(demands, tuple):
        raise TypeError("memory access-class demands must be a tuple")
    if any(not isinstance(path, MemoryPathDemand) for path in demands):
        raise TypeError(
            "memory access-class demands must contain MemoryPathDemand")
    if not isinstance(services, tuple):
        raise TypeError("memory access-class services must be a tuple")
    if any(not isinstance(service, MemoryAccessClassService)
           for service in services):
        raise TypeError(
            "memory access-class services must contain "
            "MemoryAccessClassService")

    service_by_identity = {
        (
            service.service,
            service.access_kind,
            service.ingress_resource_role,
            service.signature,
        ): service
        for service in services
    }
    if len(service_by_identity) != len(services):
        raise ValueError(
            "memory access-class services repeat an exact signature")
    service_names = {service.service for service in services}
    diagnostics = []
    for path in demands:
        for demand in path.access_class_demands:
            if demand.service not in service_names:
                raise ValueError(
                    "memory path declares access-class service without a "
                    f"hardware capability: {demand.service!r}")
            identity = (
                demand.service,
                path.access_kind,
                demand.ingress_resource_role,
                demand.signature,
            )
            if identity not in service_by_identity:
                raise ValueError(
                    "memory access-class demand has no exact hardware "
                    "signature: "
                    f"service={demand.service!r}, "
                    f"access_kind={path.access_kind!r}, "
                    f"ingress={demand.ingress_resource_role!r}, "
                    f"signature={demand.signature!r}")
            service = service_by_identity[identity]
            capacity = _access_class_capacity(service, grid_size)
            service_time = demand.logical_bytes / capacity
            diagnostics.append(MemoryAccessClassDiagnostic(
                path_name=path.name,
                service=demand.service,
                access_kind=path.access_kind,
                ingress_resource_role=demand.ingress_resource_role,
                signature=demand.signature,
                scope=service.scope,
                logical_bytes=demand.logical_bytes,
                logical_byte_capacity_per_s=capacity,
                service_time_s=service_time,
                evidence_ref=service.evidence_ref,
            ))
    return tuple(diagnostics)


def solve_memory_arbitration(
        demands, diagnostics, *, resource_role, arbitration_mode,
        read_capacity_per_s, write_capacity_per_s):
    """Price directional paths once under the declared hardware contract.

    Shared modes are half duplex and therefore add directional service times.
    An independent-directional endpoint overlaps read and write service, so
    its lower bound is their maximum.  This resolver intentionally has no
    direction-switch input.
    """

    if not isinstance(demands, tuple):
        raise TypeError("memory arbitration demands must be a tuple")
    if not isinstance(diagnostics, tuple):
        raise TypeError("memory arbitration diagnostics must be a tuple")
    if any(not isinstance(value, MemoryPathDemand) for value in demands):
        raise TypeError(
            "memory arbitration demands must contain MemoryPathDemand")
    if any(not isinstance(value, MemoryDiagnostics)
           for value in diagnostics):
        raise TypeError(
            "memory arbitration diagnostics must contain MemoryDiagnostics")
    role = resource_role.strip() if isinstance(resource_role, str) else ""
    if not role:
        raise ValueError(
            "memory arbitration resource_role must be a non-empty string")
    if arbitration_mode not in (
            "shared_common_capacity", "shared_directional_capacity",
            "independent_directional_capacity"):
        raise ValueError(
            f"unknown memory arbitration mode: {arbitration_mode!r}")
    capacities = {
        "read": read_capacity_per_s,
        "write": write_capacity_per_s,
    }
    for access_kind, capacity in capacities.items():
        if (isinstance(capacity, bool)
                or not isinstance(capacity, (int, float))
                or not math.isfinite(capacity)
                or capacity <= 0):
            raise ValueError(
                f"{access_kind} arbitration capacity must be finite and "
                "positive")
        capacities[access_kind] = float(capacity)
    if (arbitration_mode == "shared_common_capacity"
            and not math.isclose(
                capacities["read"], capacities["write"],
                rel_tol=1e-12, abs_tol=0.0)):
        raise ValueError(
            "shared_common_capacity requires equal read/write capacity")

    demand_names = tuple(value.name for value in demands)
    diagnostic_names = tuple(value.path_name for value in diagnostics)
    if (len(demand_names) != len(set(demand_names))
            or len(diagnostic_names) != len(set(diagnostic_names))):
        raise ValueError("memory arbitration path names must be unique")
    if set(demand_names) != set(diagnostic_names):
        raise ValueError(
            "memory arbitration demand/diagnostic path names disagree")
    diagnostic_by_path = {
        value.path_name: value for value in diagnostics}
    bytes_by_access = {"read": 0.0, "write": 0.0}
    time_by_access = {"read": 0.0, "write": 0.0}
    found = False
    for demand in demands:
        diagnostic = diagnostic_by_path[demand.name]
        if diagnostic.access_kind != demand.access_kind:
            raise ValueError(
                f"memory path {demand.name!r} access kind changed during "
                "solving")
        demand_nodes = tuple(
            value for value in demand.nodes
            if value.resource_role == role)
        diagnostic_nodes = tuple(
            value for value in diagnostic.nodes
            if value.resource_role == role)
        if len(demand_nodes) != len(diagnostic_nodes):
            raise ValueError(
                f"memory path {demand.name!r} has inconsistent {role!r} "
                "demand and diagnostics")
        if not demand_nodes:
            continue
        if len(demand_nodes) != 1:
            raise ValueError(
                f"memory path {demand.name!r} must visit shared resource "
                f"{role!r} exactly once")
        found = True
        demand_node = demand_nodes[0]
        diagnostic_node = diagnostic_nodes[0]
        if not math.isclose(
                demand_node.bytes, diagnostic_node.bytes,
                rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                f"memory path {demand.name!r} changed {role!r} bytes during "
                "solving")
        expected_capacity = capacities[demand.access_kind]
        if not math.isclose(
                diagnostic_node.byte_capacity_per_s, expected_capacity,
                rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(
                f"memory path {demand.name!r} did not use the declared "
                f"{demand.access_kind} {role!r} capacity")
        expected_time = demand_node.bytes / expected_capacity
        if not math.isclose(
                diagnostic_node.capacity_time_s, expected_time,
                rel_tol=1e-12, abs_tol=1e-18):
            raise ValueError(
                f"memory path {demand.name!r} has inconsistent {role!r} "
                "capacity time")
        bytes_by_access[demand.access_kind] += demand_node.bytes
        time_by_access[demand.access_kind] += expected_time
    if not found:
        raise ValueError(
            f"memory arbitration has no demand for resource {role!r}")
    if arbitration_mode == "independent_directional_capacity":
        analytical_time = max(
            time_by_access["read"], time_by_access["write"])
    elif arbitration_mode in (
            "shared_common_capacity", "shared_directional_capacity"):
        analytical_time = (
            time_by_access["read"] + time_by_access["write"])
    else:
        raise AssertionError(
            f"unhandled memory arbitration mode: {arbitration_mode!r}")
    return MemoryArbitrationDiagnostic(
        resource_role=role,
        arbitration_mode=arbitration_mode,
        read_bytes=bytes_by_access["read"],
        write_bytes=bytes_by_access["write"],
        read_capacity_per_s=capacities["read"],
        write_capacity_per_s=capacities["write"],
        read_service_time_s=time_by_access["read"],
        write_service_time_s=time_by_access["write"],
        analytical_time_s=analytical_time,
        effective_time_s=analytical_time,
        switch_count=None,
        read_to_write_turnaround_s=None,
        write_to_read_turnaround_s=None,
        bound_kind="lower_bound",
        schedule_status="unavailable",
        overridden=False,
    )


def logical_interface_source_bytes(program, *, expected_kernel, hardware=None):
    """Project explicit public object accesses, with no per-operator byte formula."""
    from src.kernel.memory_access import EventKind, Operation, Semantics, unique_by
    from src.kernel.memory_access_builder import _logical_interface_layouts,_logical_interface_read_repeat
    from src.kernel.memory_access_builder import MemoryAccessWork,LogicalMemoryInvocation
    if isinstance(program,LogicalMemoryInvocation):
        from math import prod
        program.assert_logical_source(expected_kernel,hardware)
        reads,writes=program.layouts
        if any(width is None for _,_,_,width in (*reads,*writes)):
            raise NotImplementedError('Logical source encoding must be declared before byte pricing')
        return (sum(prod(shape)*width*_logical_interface_read_repeat(expected_kernel,role)
                    for role,shape,_,width in reads),
                sum(prod(shape)*width for _,shape,_,width in writes))
    if isinstance(program,MemoryAccessWork):
        invocation=program.invocation
        if (invocation.logical_hardware is None
                or (program.source_kernel if program.source_kernel is not None else invocation.kernel)!=expected_kernel
                or hardware is not None and invocation.logical_hardware is not hardware):
            raise ValueError('Logical work crossed its source boundary')
        invocation.assert_logical_source(expected_kernel,hardware);program.__post_init__()
        if any(view.element_bytes is None for view in program.views):
            raise NotImplementedError('Logical source encoding must be declared before byte pricing')
        return program.requested_bytes(Operation.READ),program.requested_bytes(Operation.WRITE)
    program.__post_init__()
    if program.hardware_id!=expected_kernel.hardware_id or len(program.scopes)!=1 or program.tile_programs:
        raise ValueError('Logical source does not match the expected kernel summary')
    scope,=program.scopes
    if scope.op_ids!=(expected_kernel.kind.value,expected_kernel.variant):
        raise ValueError('Logical source operation differs from prediction')
    reads,writes=_logical_interface_layouts(expected_kernel,hardware)
    views=unique_by(program.views,'view_id')
    if set(views)!={item[0] for item in (*reads,*writes)}:
        raise ValueError('Logical source operands differ from prediction')
    expected_ids=set()
    accesses=unique_by(program.accesses,'access_id')
    read_ids={program.program_id+'.read.'+role for role,_,_,_ in reads}
    for operation,layouts in ((Operation.READ,reads),(Operation.WRITE,writes)):
        for role,shape,dtype,width in layouts:
            view=views[role]
            if (view.shape,view.dtype,view.element_bytes)!=(shape,dtype,width):
                raise ValueError('Logical source shape or encoding differs from prediction')
            if width is None:
                raise NotImplementedError('Logical source encoding must be declared before byte pricing')
            identity=program.program_id+('.read.' if operation is Operation.READ else '.write.')+role
            expected_ids.add(identity)
            if identity not in accesses:raise ValueError('Logical source omits an operand access')
            access=accesses[identity]
            repeat=_logical_interface_read_repeat(expected_kernel,role) if operation is Operation.READ else 1
            if (access.operation,access.allocation_id,access.region,access.repeat)!=(operation,view.allocation_id,view.region(),repeat):
                raise ValueError('Logical source access differs from its declared operand')
            if access.scope_id!=scope.scope_id or access.semantics is not Semantics.LOGICAL:
                raise ValueError('Logical source access lost its operation scope or semantics')
            if operation is Operation.WRITE and {ref.node_id for ref in access.dependencies
                    if ref.kind is EventKind.READ_READY}!=read_ids:
                raise ValueError('Logical source output requires every input read dependency')
    if set(accesses)!=expected_ids:raise ValueError('Logical source has unexpected duplicate work')
    return logical_requested_bytes(program,Operation.READ),logical_requested_bytes(program,Operation.WRITE)


def public_attention_source_work(program,*,expected_kernel,hardware):
    from src.kernel.memory_access import requested_bytes,LogicalCoordinateBinding,LogicalCoordinateTerm,unique_by
    from src.kernel.memory_access_builder import has_public_attention_source,public_attention_dimensions
    if not has_public_attention_source(expected_kernel):raise ValueError('Not a public attention source domain')
    read,write=logical_interface_source_bytes(program,expected_kernel=expected_kernel,hardware=hardware)
    from src.kernel.memory_access_builder import LogicalMemoryInvocation
    mode,_,_=public_attention_dimensions(expected_kernel)
    if isinstance(program,LogicalMemoryInvocation):
        from math import prod
        layouts={role:(shape,width) for role,shape,_,width in program.layouts[0]}
        (q,qw),(k,kw),(v,vw)=(layouts[role] for role in ('query','key','value'))
        qb,kb,vb=(prod(shape)*width for shape,width in ((q,qw),(k,kw),(v,vw)))
    else:
        views=unique_by(program.views,'view_id');qv,kv,vv=(views[role] for role in ('query','key','value'))
        if mode=='grouped':
            mapping=tuple(LogicalCoordinateBinding(axis,(LogicalCoordinateTerm(axis,1),)) for axis in range(3))
            if (vv.allocation_id,vv.parent_view_id,vv.coordinate_map)!=(kv.allocation_id,'key',mapping):
                raise ValueError('Grouped MLA value must retain its actual leading-channel key alias')
        q,k,v=qv.shape,kv.shape,vv.shape;qw,kw,vw=qv.element_bytes,kv.element_bytes,vv.element_bytes
        qb,kb,vb=(requested_bytes(view.region()) for view in (qv,kv,vv))
    return {'read_bytes':read,'write_bytes':write,'query_bytes':qb,
        'nominal_compulsory_input_bytes':(qb*v[-1]//q[-1]+2*vb) if mode=='grouped' else read,
        'key_bytes':kb,'value_bytes':vb,'query_row_bytes':q[-2]*q[-1]*qw,
        'kv_head_row_bytes':k[-1]*kw+v[-1]*vw,'query_heads':q[-2],'kv_heads':k[-2]}


def public_gemm_source_work(program,*,expected_kernel,hardware):
    """Conditional nominal work from opaque mathematical GEMM selections.

    Unknown input encodings stay unknown in the public program. The old
    nominal scalar-width hypothesis is exposed separately, never promoted
    to native requested bytes, capacity, or an observed storage descriptor.
    """
    from math import prod
    from src.kernel.memory_access import EventKind,Operation,Semantics,unique_by
    from src.kernel.memory_access_builder import has_public_tensor_gemm_source,_logical_interface_layouts
    kernel=expected_kernel
    from src.kernel.memory_access_builder import MemoryAccessWork,LogicalMemoryInvocation
    if isinstance(program,LogicalMemoryInvocation):
        program.assert_logical_source(kernel,hardware)
        if not has_public_tensor_gemm_source(kernel,hardware):
            raise ValueError('Public GEMM work crossed its checked source')
        layouts={role:(shape,width) for role,shape,_,width in (*program.layouts[0],*program.layouts[1])}
    elif isinstance(program,MemoryAccessWork):
        invocation=program.invocation
        if (invocation.logical_hardware is not hardware
                or (program.source_kernel if program.source_kernel is not None else invocation.kernel)!=kernel
                or not has_public_tensor_gemm_source(kernel,hardware)):
            raise ValueError('Public GEMM work crossed its checked source')
        invocation.assert_logical_source(kernel,hardware);program.__post_init__()
        views=unique_by(program.views,'view_id')
    else:
        program.__post_init__()
        if (not has_public_tensor_gemm_source(kernel,hardware) or program.hardware_id!=hardware.hardware_id
                or kernel.hardware_id!=hardware.hardware_id or program.coverage.executable
                or len(program.scopes)!=1 or program.tile_programs):
            raise ValueError('Public GEMM requires its actual nonexecutable mathematical source')
        scope,=program.scopes
        if scope.op_ids!=(kernel.kind.value,kernel.variant):raise ValueError('Public GEMM scope differs from its kernel')
        reads,writes=_logical_interface_layouts(kernel,hardware)
        views=unique_by(program.views,'view_id');accesses=unique_by(program.accesses,'access_id')
        expected_ids=set();read_ids={program.program_id+'.read.'+role for role,_,_,_ in reads}
        if set(views)!={role for role,_,_,_ in (*reads,*writes)}:raise ValueError('Public GEMM operand roles differ')
        for operation,layouts in ((Operation.READ,reads),(Operation.WRITE,writes)):
            for role,shape,dtype,width in layouts:
                view=views[role];identity=program.program_id+('.read.' if operation is Operation.READ else '.write.')+role
                expected_ids.add(identity)
                if identity not in accesses:raise ValueError('Public GEMM omitted its operand access')
                access=accesses[identity]
                if ((view.shape,view.dtype,view.element_bytes)!=(shape,dtype,width)
                        or (access.operation,access.allocation_id,access.region,access.repeat,access.scope_id,access.semantics)
                        !=(operation,view.allocation_id,view.region(),1,scope.scope_id,Semantics.LOGICAL)):
                    raise ValueError('Public GEMM access differs from its mathematical operand')
                if operation is Operation.WRITE and {ref.node_id for ref in access.dependencies if ref.kind is EventKind.READ_READY}!=read_ids:
                    raise ValueError('Public GEMM result requires both actual operand reads')
        if set(accesses)!=expected_ids:raise ValueError('Public GEMM contains unexpected work')
    if not isinstance(program,LogicalMemoryInvocation):
        layouts={role:(view.shape,view.element_bytes) for role,view in views.items()}
    (a,_),(b,_),(c,cw)=(layouts[role] for role in ('A','B','logical_result'))
    batches,rows,k=a;n=c[-1]
    # This hypothesis already belongs to the retained analytic model. It is
    # not used to fill the unknown dtype/width of either public input view.
    nominal_width=kernel.shape['dtype_bytes']
    operands=(prod(a)*nominal_width,prod(b)*nominal_width,prod(c)*cw)
    partial_bytes=prod(layouts['partials'][0]) if 'partials' in layouts else 0
    return {'geometry':(batches,rows,n,k,nominal_width,nominal_width,cw),
        'nominal_operand_bytes':operands,'declared_partial_bytes':partial_bytes,
        'input_storage_known':False}


def logical_interface_source_traffic(program,*,expected_kernel,hardware):
    from src.kernel.memory_access import Operation,requested_bytes
    read_l2,write=logical_interface_source_bytes(program,expected_kernel=expected_kernel,hardware=hardware)
    # Compulsory role reads occur once; source-declared gamma replays are L2 work.
    from src.kernel.memory_access_builder import MemoryAccessWork,LogicalMemoryInvocation
    if isinstance(program,LogicalMemoryInvocation):
        from math import prod
        read_dram=sum(prod(shape)*width for _,shape,_,width in program.layouts[0])
    elif isinstance(program,MemoryAccessWork):
        views={view.view_id:view for view in program.views}
        read_dram=sum(requested_bytes(views[access.view_id].region()) for access in program.accesses if access.operation is Operation.READ)
    else:
        read_dram=sum(requested_bytes(access.region) for access in program.accesses if access.operation is Operation.READ)
    return read_l2,read_dram,write


def logical_mla_kv_step_bytes(program,*,expected_kernel,kv_rows):
    logical_interface_source_bytes(program,expected_kernel=expected_kernel)
    from src.kernel.memory_access_builder import LogicalMemoryInvocation
    layouts=({role:(shape,width) for role,shape,_,width in program.layouts[0]}
        if isinstance(program,LogicalMemoryInvocation)
        else {view.view_id:(view.shape,view.element_bytes) for view in program.views})
    names=('compressed_kv','key_rope') if 'compressed_kv' in layouts else ('compressed_kv.request0','key_rope.request0')
    if not all(name in layouts for name in names):raise ValueError('MLA panel requires compressed KV and RoPE source roles')
    return kv_rows*sum(layouts[name][0][-1]*layouts[name][1] for name in names)


def logical_combine_source_traffic(program,*,expected_kernel,hardware,known_output_only=False):
    read,write=logical_interface_source_bytes(program,expected_kernel=expected_kernel,hardware=hardware)
    if known_output_only:
        # Existing unbound MACA scratch policy: price its proven output only;
        # this is explicitly partial and does not claim zero actual reads.
        return float(write),float(write)
    width=expected_kernel.launch.parameters['split_scratch_dram_bytes_per_element']
    if width<0:raise ValueError('Negative declared scratch DRAM width')
    return float(read+write),float(read/4*width+write)


def variable_collective_source_work(program):
    """Validate every participant segment and contributor before pricing wire work."""
    from src.kernel.memory_access import Operation, EventKind, TileDependency, TensorView, unique_by
    from src.kernel.kernel_spec import DTYPE_BYTES
    program.__post_init__()
    if program.scopes or program.accesses or not program.tile_programs:
        raise ValueError("variable collective requires its compressed segment sources")
    first = program.tile_programs[0].scope.op_ids
    if len(first) < 9 or first[:1] != ("communication",) or first[2] != "partitioned":
        raise ValueError("variable collective source requires ordered participants and segment sizes")
    operation, count = first[1], int(first[3])
    if count < 2 or len(first) != 5 + 2 * count or operation not in ("all_gatherv", "reduce_scatterv"):
        raise ValueError("unknown variable collective coordinate contract")
    ranks = tuple(int(value) for value in first[4:4 + count])
    sizes = tuple(int(value) for value in first[4 + count:4 + 2 * count])
    if len(set(ranks)) != count or min(ranks) < 0 or min(sizes) < 0 or sum(sizes) <= 0:
        raise ValueError("variable collective ranks and payload extents must be valid")
    expected = {index for index, size in enumerate(sizes) if size}
    if len(program.tile_programs) != len(expected):
        raise ValueError("variable collective must cover every positive segment once")
    views = unique_by(program.views, "view_id")
    gather, seen = operation == "all_gatherv", set()
    for family in program.tile_programs:
        header = family.scope.op_ids
        if (family.scope.agent != "variable-collective-segments" or header[:-1] != first[:-1]
                or len(family.axes) != 1 or family.causal_prefixes):
            raise ValueError("variable collective segments disagree on their physical group")
        index = int(header[-1])
        if index not in expected or index in seen:
            raise ValueError("variable collective contains a missing or repeated segment")
        seen.add(index)
        reads = tuple(access for access in family.accesses if access.operation is Operation.READ)
        writes = tuple(access for access in family.accesses if access.operation is Operation.WRITE)
        sources, destinations = ((ranks[index],), ranks) if gather else (ranks, (ranks[index],))
        if (len(reads) != len(sources) or len(writes) != len(destinations)
                or any(access.scope is None for access in family.accesses)
                or {access.scope.rank for access in reads} != set(sources)
                or {access.scope.rank for access in writes} != set(destinations)):
            raise ValueError("variable segment must retain every required participant")
        dtype = views[reads[0].view_id].dtype
        if dtype not in ("uint8", "bf16", "fp16", "fp32", "int8", "int32", "int64"):
            raise ValueError("variable collective lacks an explicit scalar storage ABI")
        width = DTYPE_BYTES[dtype]
        axis = "byte" if dtype == "uint8" else "element"
        if (sizes[index] % width or any(size % width for size in sizes)
                or (family.axes[0].axis_id, family.axes[0].extent, family.axes[0].block) != (axis, sizes[index] // width, 1)):
            raise ValueError("variable collective coordinates must cover complete segment elements")
        unknown_sum = not gather and dtype == "uint8"
        if (len(family.partitions) != int(unknown_sum)
                or unknown_sum and family.partitions[0].axis_id != axis):
            raise ValueError("byte-only scatter sum must retain its unknown scalar partition")
        required = tuple(TileDependency(access.access_id, EventKind.READ_READY,
            (axis,) if unknown_sum else (), partition_axis=axis if unknown_sum else None) for access in reads)
        for access in family.accesses:
            read = access.operation is Operation.READ
            role = "input" if read else "output"
            view = views[access.view_id]
            if (not isinstance(view, TensorView) or view.shape != (sizes[index] // width,)
                    or (view.dtype, view.element_bytes, view.strides_bytes) != (dtype, width, (width,))
                    or access.iteration_axes != (axis,) or access.view_axes != (axis,)
                    or access.view_axis_divisors or access.view_axis_padding or access.unmapped
                    or access.indirect_target is not None or access.scope.op_ids != ("communication", operation, role)):
                raise ValueError("variable collective must retain exact contiguous participant segments")
            full_size = (sizes[index] if (gather and read or not gather and not read) else sum(sizes))
            parent_offsets = {parent.offset_bytes for parent in program.views if isinstance(parent, TensorView)
                and parent.allocation_id == view.allocation_id and parent.shape == (full_size // width,)
                and (parent.dtype, parent.element_bytes, parent.strides_bytes) == (dtype, width, (width,))}
            offset = 0 if (gather and read or not gather and not read) else sum(sizes[:index])
            if len(parent_offsets) != 1 or view.offset_bytes != next(iter(parent_offsets)) + offset:
                raise ValueError("variable collective segment must preserve its original participant offset")
            if read:
                if access.dependencies or len(access.input_ready) != 1:
                    raise ValueError("variable collective input requires its actual readiness event")
            elif len(access.dependencies) != len(required) or set(access.dependencies) != set(required):
                raise ValueError("variable collective output must depend on all contributing inputs")
    return sum(sizes), count, operation
