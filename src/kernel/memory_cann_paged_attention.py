"""CANN non-FlashDecode paged GQA: request/head stacks and real controls."""
from dataclasses import replace
from math import prod

from src.kernel.memory_access import (
    AccessScope,Coverage,EventKind,EventRef,Granularity,LogicalCoordinateBinding,
    LogicalCoordinateTerm,LogicalSelectionView,MemoryAccessProgram,Operation,
    Ordering,Semantics,TileAccessTemplate,TileDependency,TileProgram,unique_by,
    _tile_template_requested_bytes,
)


def paged_gqa_geometry(kernel):
    """Retain every actual head and full/tail stack of the selected program."""
    block=int(kernel.launch.parameters['cann_source_s2_base_size'])
    batch=kernel.shape['batch'];heads=kernel.shape['num_kv_heads']
    rows=kernel.shape['num_query_heads']//heads;lengths=tuple(kernel.attributes['kv_lens'])
    if block<=0 or block%128 or not 0<rows<=128 or len(lengths)!=batch:
        raise ValueError('Paged GQA requires its selected page-aligned stack and request lengths')
    segments=[];seen=set()
    for group in kernel.launch.source_program.aic_groups:
        tasks=group.tasks;position=0
        while position<len(tasks):
            head=tasks[position].b_n2;request,first=divmod(head,heads)
            if request>=batch or head in seen:raise ValueError('Paged GQA has an unknown or duplicate head')
            stacks=-(-lengths[request]//block);current=tasks[position:position+stacks]
            if (len(current)!=stacks or any(task.b_n2!=head or task.g_s1!=0 or task.s2!=index
                    or task.actual_m!=rows or task.actual_s2!=min(block,lengths[request]-index*block)
                    or task.is_s2_split_core for index,task in enumerate(current))):
                raise ValueError('Paged GQA source requires complete non-FlashDecode head stacks')
            seen.add(head);position+=stacks
            if segments and segments[-1][:2]==(group.aic_group,request) and segments[-1][2]+segments[-1][3]==first:
                core,b,start,count=segments[-1];segments[-1]=(core,b,start,count+1)
            else:segments.append((group.aic_group,request,first,1))
    if seen!=set(range(batch*heads)):raise ValueError('Paged GQA source omitted an actual head')
    return block,tuple(segments)


def _header(kernel,hardware):
    from src.kernel.memory_flash_attention import _cann_attention_header
    block,segments=paged_gqa_geometry(kernel)
    return ('cann_paged_gqa',*_cann_attention_header(kernel,hardware),str(block),
        '8' if kernel.launch.algorithm.endswith('103000000010622221') else '1',
        ';'.join(':'.join(map(str,segment)) for segment in segments))


def _parents(header,views):
    from src.kernel.memory_flash_attention import _cann_attention_layouts,_validate_cann_attention_views
    public=header[1:14]
    roles=tuple(role for role,_ in _cann_attention_layouts(public))
    _validate_cann_attention_views(public,{role:views[role] for role in roles})
    for role in ('actual_query','actual_key'):
        view=views[role]
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes)!=(
                (int(public[1]),),'int64',8,(8,)):
            raise ValueError('Paged GQA requires its actual per-request 64-bit length arrays')
    return {role:views[role] for role in (*roles,'actual_query','actual_key')}


def _recipe(header,parents,ready,scope,visibility):
    from src.kernel.memory_flash_attention import _cann_bsh_native_recipe
    if len(header)!=17 or header[0]!='cann_paged_gqa':raise ValueError('Unknown paged GQA source header')
    public=header[1:14];batch,maximum,dimension,queries,heads=map(int,public[1:6])
    lengths=tuple(map(int,public[12].split(',')));block,lanes=map(int,header[14:16])
    segments=tuple(tuple(map(int,item.split(':'))) for item in header[16].split(';'))
    if (len(lengths)!=batch or min(lengths)<=0 or max(lengths)>maximum or block<=0 or block%128
            or dimension not in (64,128) or queries%heads or not 0<queries//heads<=128
            or lanes not in (1,8) or int(public[8])!=2 or public[11]!='128'):
        raise ValueError('Paged GQA source geometry differs')
    if segments!=tuple(sorted(segments)) or len({(core,request) for core,request,_,_ in segments})!=len(segments):
        raise ValueError('Paged GQA requires one ordered contiguous head range per actor/request')
    seen=set();cores=set();allocations={};views={};events={};families={};offsets=[];offset=0
    evidence=('CANN8.5.1 arch32 FIA non-FlashDecode paged GQA: exact selected head/stack ownership and shared Q L1 snapshot.',
        'FiaKernelNonQuant::GetTaskDealMode reads per-request Q/KV lengths on each actor at its first task or batch transition.',
        'Native score/probability/partial-O GM and default/online UB max/sum dataflow reuses the same GQA block implementation as contiguous BSH.',
        'K and PV require their own logical INT32 page selections; lookup issue multiplicity, page values, physical addresses and ring placement are unresolved.')
    for length in lengths:offsets.append(offset);offset+=length
    def coordinate(axis,terms=(),offset=0,divisor=1,modulus=None):
        return LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(index,value) for index,value in terms),offset,divisor,modulus)
    def retain(target,identity,value):
        if identity in target and target[identity]!=value:raise ValueError('Paged GQA shared actor state differs')
        target[identity]=value
    for index,(core,request,first,count) in enumerate(segments):
        if (core>=int(public[7]) or request>=batch or count<=0 or first+count>heads
                or any((request,head) in seen for head in range(first,first+count))):
            raise ValueError('Paged GQA head ranges overlap or exceed their actual request')
        seen.update((request,head) for head in range(first,first+count));cores.add(core)
        segment_scope=replace(scope,scope_id=f'{scope.scope_id}.s{index}')
        local={};local_views=[]
        for role in ('query','key','value','output'):
            parent=parents[role];shape=(1,1,queries*dimension) if role in ('query','output') else (1,lengths[request],heads*dimension)
            mapping=((coordinate(0,offset=request),coordinate(1,((1,1),)),coordinate(2,((2,1),)))
                if role in ('query','output') else (coordinate(0,((1,1),),offsets[request]),
                    coordinate(1,((2,1),),divisor=dimension),coordinate(2,((2,1),),modulus=dimension)))
            view=LogicalSelectionView(f'{segment_scope.scope_id}.{role}.parent',parent.allocation_id,shape,
                parent.dtype,parent.element_bytes,f'cann_paged.{index}.{role}',parent.view_id,mapping)
            local[role]=view;local_views.append(view)
        local['tiling']=parents['tiling']
        local_public=('cann_bsh_decode','1',str(lengths[request]),str(dimension),str(queries),str(heads),
            public[6],'1',public[8],public[9])
        local_header=('cann_bsh_native_tiles',*local_public,str(block),f'{first}:{count}',str(lanes))
        a,v,e,f=_cann_bsh_native_recipe(local_header,local,ready,segment_scope,visibility,
            _fd_segment=(core,first,count,0,-(-lengths[request]//block),0))
        family,=f;axes={axis.axis_id:f's{index}.{axis.axis_id}' for axis in family.axes}
        shared={'framework_clear','argument_aic','argument_aiv0','argument_aiv1',
            'v0.default_max','v0.default_sum','v1.default_max','v1.default_sum'}
        identities={access.access_id:(f'aic{core}.{access.access_id}' if access.access_id in shared else f's{index}.{access.access_id}')
            for access in family.accesses}
        view_ids={view.view_id:view.view_id for view in (*local_views,*v)};allocation_ids={item.allocation_id:item.allocation_id for item in a}
        view_ids['tiling']='tiling'
        for view in v:
            for suffix in ('kfc_clear','v0.default_max','v0.default_sum','v1.default_max','v1.default_sum'):
                if view.view_id==f'{segment_scope.scope_id}.core0.{suffix}':
                    shared_id=f'{scope.scope_id}.aic{core}.{suffix}'
                    view_ids[view.view_id]=shared_id;allocation_ids[view.allocation_id]=shared_id
        event_ids={item.event_id:f'{allocation_ids[item.allocation_id]}.allocated' for item in e}
        for item in a:
            changed=replace(item,allocation_id=allocation_ids[item.allocation_id],allocated_event=event_ids[item.allocated_event])
            retain(allocations,changed.allocation_id,changed)
        for item in e:
            changed=replace(item,event_id=event_ids[item.event_id],allocation_id=allocation_ids[item.allocation_id])
            retain(events,changed.event_id,changed)
        for view in (*local_views,*v):
            changed=replace(view,view_id=view_ids[view.view_id],
                allocation_id=allocation_ids[view.allocation_id] if view.allocation_id in allocation_ids else view.allocation_id)
            if isinstance(view,LogicalSelectionView) and view.parent_view_id in view_ids:
                changed=replace(changed,parent_view_id=view_ids[view.parent_view_id])
            if isinstance(changed,LogicalSelectionView) and view_ids[view.view_id]!=view.view_id:
                changed=replace(changed,selection_id=f'cann_paged.shared.{changed.view_id}')
            retain(views,changed.view_id,changed)
        if core not in families:families[core]=[[],{}]
        core_axes,accesses=families[core]
        core_axes.extend(replace(axis,axis_id=axes[axis.axis_id]) for axis in family.axes)
        controls={}
        for actor in ('aic','aiv0','aiv1'):
            refs=[]
            for role in ('actual_query','actual_key'):
                identity=f'aic{core}.r{request}.{actor}.{role}';view_id=f'{scope.scope_id}.r{request}.{role}'
                parent=parents[role]
                view=LogicalSelectionView(view_id,parent.allocation_id,(1,),'int64',8,
                    f'cann_paged.{request}.{role}',role,(coordinate(0,((0,1),),request),))
                retain(views,view_id,view)
                access=TileAccessTemplate(identity,view_id,Operation.READ,(),(None,),(ready[role],),
                    (TileDependency(f'aic{core}.argument_{actor}',EventKind.READ_READY,()),),evidence,unmapped=True)
                retain(accesses,identity,access);refs.append(TileDependency(identity,EventKind.READ_READY,()))
            controls[actor]=tuple(refs)
        for access in family.accesses:
            dependencies=tuple(replace(dep,access_id=identities[dep.access_id],
                reduction_axes=tuple(axes[name] for name in dep.reduction_axes),
                coordinate_bindings=tuple(replace(binding,axis_id=axes[binding.axis_id],
                    terms=tuple(replace(term,axis_id=axes[term.axis_id]) for term in binding.terms)) for binding in dep.coordinate_bindings),
                carry_axis=axes[dep.carry_axis] if dep.carry_axis is not None else None,
                initial_access_id=identities[dep.initial_access_id] if dep.initial_access_id is not None else None)
                for dep in access.dependencies)
            actor='aiv0' if access.access_id.startswith('v0.') else 'aiv1' if access.access_id.startswith('v1.') else 'aic'
            if access.access_id not in shared:dependencies+=controls[actor]
            iteration=tuple(axes[name] for name in access.iteration_axes)
            if access.access_id.endswith(('.key','.value')):
                operand=views[view_ids[access.view_id]];middle='middle' in access.iteration_axes
                phase=access.access_id.split('.',1)[0];start=0 if phase=='first' else 1 if middle else -(-lengths[request]//block)-1
                if middle:shape=(count,operand.shape[1],block//128);mapping=(coordinate(0,offset=request),coordinate(1,((1,block//128),(2,1)),start*block//128))
                else:shape=(count,-(-operand.shape[1]//128));mapping=(coordinate(0,offset=request),coordinate(1,((1,1),),start*block//128))
                identity=f'{identities[access.access_id]}.page_table';view_id=f'{scope.scope_id}.{identity}'
                table=LogicalSelectionView(view_id,parents['block_table'].allocation_id,shape,'int32',4,
                    f'cann_paged.{identity}','block_table',mapping);retain(views,view_id,table)
                lookup=TileAccessTemplate(identity,view_id,Operation.READ,iteration,
                    (axes['task'],axes['middle'],None) if middle else (axes['task'],None),(ready['block_table'],),
                    (TileDependency(f'aic{core}.argument_aic',EventKind.READ_READY,()),*controls['aic']),evidence,unmapped=True)
                retain(accesses,identity,lookup);dependencies+=(TileDependency(identity,EventKind.READ_READY,()),)
            access_scope=access.scope
            if access_scope is not None:
                identity=view_ids[access.view_id]
                access_scope=replace(access_scope,scope_id=f'{identity}.initializer')
            changed=replace(access,access_id=identities[access.access_id],view_id=view_ids[access.view_id],
                iteration_axes=iteration,view_axes=tuple(axes[name] if name is not None else None for name in access.view_axes),
                dependencies=dependencies,evidence_refs=evidence,scope=access_scope)
            retain(accesses,changed.access_id,changed)
    if seen!={(request,head) for request in range(batch) for head in range(heads)} or cores!=set(range(int(public[7]))):
        raise ValueError('Paged GQA source has an incomplete head/actor domain')
    result=[]
    for core,(axes,accesses) in sorted(families.items()):
        core_scope=replace(scope,scope_id=f'{scope.scope_id}.aic{core}',agent='logical-cann-paged-gqa-stacks')
        result.append(TileProgram(core_scope.scope_id,core_scope,tuple(axes),tuple(accesses.values()),visibility,evidence))
    return tuple(allocations.values()),tuple(views.values()),tuple(events.values()),tuple(result)


def build_cann_paged_attention_source(kernel,hardware,bindings, *, program_id):
    header=_header(kernel,hardware);parents=_parents(header,unique_by(bindings.views,'view_id'))
    read_roles=tuple(role for role in parents if role!='output')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(read_roles,('output',),len(read_roles)):
        raise ValueError('Paged GQA bindings omit actual payload/argument/length operands')
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-paged-gqa-stacks')
    ready=dict(zip(read_roles,bindings.input_ready,strict=True))
    allocations,views,events,families=_recipe(header,parents,ready,scope,bindings.visibility_domain)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Exact logical head/stack, length/page control and GM/UB state dataflow; native ring addresses and page values remain unknown.',
            'Control/page selection requests are not native scalar issue counts or DRAM traffic; original explicit price projections remain separate.')),
        (*bindings.allocations,*allocations),(*bindings.views,*views),(scope,),(*bindings.events,*events),(),families)
    cann_paged_attention_source_demands(program,expected_kernel=kernel)
    return program


def cann_paged_attention_source_demands(program, *, expected_kernel=None):
    program.__post_init__()
    if (program.hardware_id!='ascend_910b' or len(program.scopes)!=1 or not program.tile_programs or program.accesses):
        raise ValueError('Paged GQA requires its unique CANN tile stream')
    scope,=program.scopes;header=scope.op_ids;public=header[1:14]
    all_views=unique_by(program.views,'view_id');parents=_parents(header,all_views)
    if expected_kernel is not None:
        # Header equality also checks the complete selected task ownership.
        from src.kernel.memory_flash_attention import _has_cann_paged_gqa_source
        if (not _has_cann_paged_gqa_source(expected_kernel) or program.hardware_id!=expected_kernel.hardware_id):
            raise ValueError('Paged GQA source differs from its actual kernel')
        # Argument fields are already bound by the caller/HWSpec header. Avoid
        # loading a second hardware configuration in a source projection.
        block,segments=paged_gqa_geometry(expected_kernel)
        if (tuple(map(int,public[1:6]))!=tuple(expected_kernel.shape[name] for name in ('batch','kv_len','head_dim','num_query_heads','num_kv_heads'))
                or public[12]!=','.join(map(str,expected_kernel.attributes['kv_lens']))
                or public[7]!=str(expected_kernel.launch.grid_size) or public[10]!=expected_kernel.attributes['kv_layout']
                or public[11]!='128'
                or header[14:]!=(str(block),'8' if expected_kernel.launch.algorithm.endswith('103000000010622221') else '1',
                    ';'.join(':'.join(map(str,segment)) for segment in segments))):
            raise ValueError('Paged GQA source differs from its selected request/stack program')
    ready={}
    for family in program.tile_programs:
        for access in family.accesses:
            if access.input_ready:
                view=all_views[access.view_id]
                while isinstance(view,LogicalSelectionView) and view.parent_view_id is not None:view=all_views[view.parent_view_id]
                ref,=access.input_ready
                if view.view_id in ready and ready[view.view_id]!=ref:raise ValueError('Paged GQA parent readiness is ambiguous')
                ready[view.view_id]=ref
    allocations,views,events,expected=_recipe(header,parents,ready,scope,program.tile_programs[0].visibility_domain)
    known_allocations=unique_by(program.allocations,'allocation_id');known_events=unique_by(program.events,'event_id')
    if (program.tile_programs!=expected or set(all_views)!={*parents,*(view.view_id for view in views)}
            or any(all_views[view.view_id]!=view for view in views)
            or any(item.allocation_id not in known_allocations or known_allocations[item.allocation_id]!=item for item in allocations)
            or any(item.event_id not in known_events or known_events[item.event_id]!=item for item in events)):
        raise ValueError('Paged GQA lost actual length/page/stack/state READ_READY or typed parent mapping')
    totals={};domains={item.allocation_id:item.memory_domain for item in program.allocations}
    traffic={(domain,operation):0 for domain in ('logical-gm-workspace','logical-ub-state') for operation in Operation}
    reads=metadata=0
    for family in program.tile_programs:
        axes=unique_by(family.axes,'axis_id')
        for access in family.accesses:
            amount=_tile_template_requested_bytes(all_views,axes,family,access)
            totals[access.access_id]=amount
            if access.operation is Operation.READ:
                reads+=amount
                if access.access_id.endswith(('.actual_query','.actual_key','.page_table')):metadata+=amount
            key=(domains[all_views[access.view_id].allocation_id],access.operation)
            if key in traffic:traffic[key]+=amount
    size=lambda role:prod(parents[role].shape)*parents[role].element_bytes
    argument=sum(amount for name,amount in totals.items() if '.argument_' in name)
    return {'geometry':tuple(map(int,public[1:6])),'query_unique_bytes':size('query'),'kv_unique_bytes':None,
        'kv_interface_bytes':size('key')+size('value'),'payload_read_bytes':sum(size(role) for role in ('query','key','value')),
        'write_bytes':size('output'),'read_bytes':reads,'metadata_read_bytes':metadata,
        'argument_payload_bytes':int(public[6]),'argument_aiv_transfer_bytes':int(public[9]),
        'argument_aic_materializations':int(public[7]),'argument_aiv_materializations':int(public[7])*int(public[8]),
        'argument_read_bytes':argument,'native_workspace_stream_bound':True,
        'gm_workspace_read_bytes':traffic['logical-gm-workspace',Operation.READ],
        'gm_workspace_write_bytes':traffic['logical-gm-workspace',Operation.WRITE],
        'ub_state_read_bytes':traffic['logical-ub-state',Operation.READ],
        'ub_state_write_bytes':traffic['logical-ub-state',Operation.WRITE],'totals':totals,
        'priced_score_elements':sum(amount for name,amount in totals.items() if name.endswith('.scores_write'))//4,
        'priced_output_elements':size('output')//2,
        'priced_task_query_l2_bytes':sum(amount for name,amount in totals.items() if name.endswith('.partial_write'))//2,
        'key_requested_bytes':sum(amount for name,amount in totals.items() if name.endswith('.key')),
        'value_requested_bytes':sum(amount for name,amount in totals.items() if name.endswith('.value')),
        'framework_clear_write_bytes':sum(amount for name,amount in totals.items() if name.endswith('.framework_clear'))}
