"""Two-pass GPU RMSNorm tensors and in-place residual update causality."""
from functools import lru_cache

from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity,
    MemoryAccess, MemoryAccessProgram, MemoryAllocation, Operation, Ordering,
    ProgramEvent, Semantics, TensorView, unique_by,TileAxis,TileAccessTemplate,TileDependency,TileProgram,
)


def has_layernorm_source(kernel,hardware):
    del hardware
    return (kernel.kind is KernelKind.VECTOR and kernel.variant=='layernorm'
        and kernel.execution_support=='supported')


def standalone_layernorm_bindings(kernel,*,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    _,rows=_normalization_row_extents(kernel);hidden=kernel.shape['hidden'];width=kernel.shape['dtype_bytes']
    inputs=('input','gamma','beta') if kernel.attributes['affine'] else ('input',)
    allocations=[];views=[];events=[];ready=[]
    for role in (*inputs,'output'):
        dtype=kernel.dtype
        role_width=width
        shape=(hidden,) if role in ('gamma','beta') else (rows,hidden)
        strides=(role_width,) if len(shape)==1 else (hidden*role_width,role_width)
        identity=f'{binding_id}.{role}';size=hidden*role_width*(1 if len(shape)==1 else rows)
        allocation=MemoryAllocation(identity,'logical-local-device','logical-layernorm-tensor',size,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        view=TensorView(role,identity,0,shape,strides,dtype,role_width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role in inputs:
            event=ProgramEvent(identity+'.ready',EventKind.INITIALIZED,identity,view.region(),'layernorm-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),inputs,('output',),
        tuple(ready),0,'offline','layernorm-visible')


def _layernorm_private_roles(wrapper,source):
    if not isinstance(source,TensorView) or len(source.shape)!=2:
        raise ValueError('LayerNorm requires its typed row tensor')
    if not wrapper:return ()
    if source.element_bytes!=4:return ('fp32_input','fp32_output')
    # PyTorch contiguous ignores the stride of a size-one dimension.
    return ('fp32_input',) if source.shape[0]>1 and source.strides_bytes[0]!=source.shape[1]*4 else ()


def _layernorm_row_tiles(views,ready,scope,visibility):
    rows,hidden=views['input'].shape
    evidence=('LayerNorm public row contract: normalize the entire hidden dimension using its mean/variance and optional gamma/beta.',
        'Each semantic row tile consumes its input values and affine vectors; internal reduction/value reuse does not assert a native global-load count.',
        'LayerNorm output is separate from its input. Only an explicit program supplies native public side-outputs; native multipass/private workspace remains unbound.')
    if 'mean' in views:
        evidence+=('PyTorch2.8 functional layer_norm returns native_layer_norm()[0]; native_layer_norm also returns per-row mean/rstd. FP32 wrapper input makes both statistics FP32. These public outputs do not prove native global rereads or issued stores.',)
    accesses=[]
    if 'fp32_input' in views:
        evidence+=('vLLM LayerNorm.forward: F.layer_norm(x.float(), FP32 weight/bias, eps).type_as(x). The native public reference first makes its input contiguous: a strided FP32 input has a dense copy even though float() is a no-op. Cast/contiguous tensors and native FP32 mean/rstd side-outputs are public logical device objects; native multipass and compiler private state remain unbound.',)
        accesses.extend((
            TileAccessTemplate('input','input',Operation.READ,('row',),('row',None),(ready['input'],),(),evidence),
            TileAccessTemplate('fp32_input_write','fp32_input',Operation.WRITE,('row',),('row',None),(),
                (TileDependency('input',EventKind.READ_READY,()),),evidence),
            TileAccessTemplate('fp32_input_read','fp32_input',Operation.READ,('row',),('row',None),(),
                (TileDependency('fp32_input_write',EventKind.WRITE_VISIBLE,()),),evidence)))
        for role in (('gamma','beta') if 'gamma' in views else ()):
            accesses.append(TileAccessTemplate(role,role,Operation.READ,('row',),(None,),(ready[role],),(),evidence))
        dependencies=tuple(TileDependency(name,EventKind.READ_READY,()) for name in
            ('fp32_input_read',*(('gamma','beta') if 'gamma' in views else ())))
        if 'fp32_output' in views:
            accesses.extend((
                TileAccessTemplate('fp32_output_write','fp32_output',Operation.WRITE,('row',),('row',None),(),dependencies,evidence),
                TileAccessTemplate('fp32_output_read','fp32_output',Operation.READ,('row',),('row',None),(),
                    (TileDependency('fp32_output_write',EventKind.WRITE_VISIBLE,()),),evidence)))
            dependencies=(TileDependency('fp32_output_read',EventKind.READ_READY,()),)
        accesses.append(TileAccessTemplate('output','output',Operation.WRITE,('row',),('row',None),(),dependencies,evidence))
        for role in ('mean','rstd'):
            accesses.append(TileAccessTemplate(role,role,Operation.WRITE,('row',),('row',None),(),
                (TileDependency('fp32_input_read',EventKind.READ_READY,()),),evidence))
        return TileProgram(scope.scope_id,scope,(TileAxis('row',rows,1),),tuple(accesses),visibility,evidence)
    for role in ('input',*(('gamma','beta') if 'gamma' in views else ())):
        accesses.append(TileAccessTemplate(role,role,Operation.READ,('row',),
            ('row',None) if role=='input' else (None,),(ready[role],),(),evidence))
    accesses.append(TileAccessTemplate('output','output',Operation.WRITE,('row',),('row',None),(),
        tuple(TileDependency(access.access_id,EventKind.READ_READY,()) for access in accesses),evidence))
    if 'mean' in views:
        for role in ('mean','rstd'):
            accesses.append(TileAccessTemplate(role,role,Operation.WRITE,('row',),('row',None),(),
                (TileDependency('input',EventKind.READ_READY,()),),evidence))
    return TileProgram(scope.scope_id,scope,(TileAxis('row',rows,1),),tuple(accesses),visibility,evidence)


def _layernorm_header(kernel):
    logical,execution=_normalization_row_extents(kernel)
    base=('layernorm_rows',str(int(kernel.attributes['affine'])))
    return base if logical==execution else (*base,f'logical_rows={logical}',f'execution_rows={execution}')


def build_layernorm_source(kernel,hardware,bindings,granularity,*,program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_layernorm_source(kernel,hardware) or not isinstance(bindings,ElementwiseBindings):
        raise TypeError('LayerNorm requires its supported typed row operands')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown LayerNorm granularity')
    expected=('input','gamma','beta') if kernel.attributes['affine'] else ('input',)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(expected,('output',),len(expected)):
        raise ValueError('LayerNorm affine operand/readiness roles differ')
    caller_bindings=bindings
    views=unique_by(bindings.views,'view_id')
    scope=AccessScope(program_id,_layernorm_header(kernel),bindings.call_id,(),
        bindings.rank,bindings.stream,'logical-layernorm-row-reduction')
    family=_layernorm_row_tiles(views,dict(zip(expected,bindings.input_ready,strict=True)),scope,bindings.visibility_domain)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Public mathematical row tiles, with affine input reuse and no cross-row ordering.',
            'Requested logical parameter reads are distinct from retained unique-object pricing.',
            'Declared FP32 wrapper includes public mean/rstd side-outputs. Native multipass/private workspace ABI remains unbound; generic unspecified programs do not infer side-output objects.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    layernorm_source_work(program,expected_kernel=kernel,expected_bindings=caller_bindings)
    return program


def layernorm_source_work(program,*,expected_kernel=None,expected_bindings=None):
    from src.kernel.memory_path_solver import logical_requested_bytes,logical_view_unique_bytes
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:
        raise ValueError('LayerNorm requires its unique compressed row source')
    family,=program.tile_programs;header=family.scope.op_ids
    wrapper=len(header)>2 and header[2]=='program=vllm_fp32_layernorm'
    mapping=header[3:] if wrapper else header[2:]
    if (len(mapping) not in (0,2) or header[:2] not in (('layernorm_rows','0'),('layernorm_rows','1'))
            or family.scope.agent!='logical-layernorm-row-reduction'):
        raise ValueError('LayerNorm row source declaration differs')
    affine=header[1]=='1';roles=('input','gamma','beta') if affine else ('input',)
    views=unique_by(program.views,'view_id')
    source=views['input']
    if not isinstance(source,TensorView) or len(source.shape)!=2:raise ValueError('LayerNorm requires its typed row tensor')
    rows,hidden=source.shape;width=source.element_bytes
    private=_layernorm_private_roles(wrapper,source)
    statistics=('mean','rstd') if wrapper else ()
    if set(views)!={*roles,'output',*private,*statistics}:raise ValueError('LayerNorm requires its exact affine input/output/cast parents')
    logical_rows=rows
    if mapping:
        if not mapping[0].startswith('logical_rows='):raise ValueError('LayerNorm logical replay mapping is missing')
        logical_rows=int(mapping[0].split('=',1)[1])
        if not 0<logical_rows<rows or mapping!=(f'logical_rows={logical_rows}',f'execution_rows={rows}'):
            raise ValueError('LayerNorm replay mapping differs from actual row storage')
    for role,view in views.items():
        shape=(hidden,) if role in ('gamma','beta') else (rows,1) if role in statistics else (rows,hidden)
        dtype,role_width=('fp32',4) if wrapper and role in ('gamma','beta',*private,*statistics) else (source.dtype,width)
        strides=(role_width,) if len(shape)==1 else (shape[1]*role_width,role_width)
        valid_input_stride=(role=='input' and isinstance(view,TensorView) and view.strides_bytes[1]==role_width
            and view.strides_bytes[0]>=hidden*role_width)
        if (not isinstance(view,TensorView) or (view.shape,view.dtype,view.element_bytes)!=(shape,dtype,role_width)
                or view.strides_bytes!=strides and not valid_input_stride):
            raise ValueError('LayerNorm row/affine tensor ABI differs')
    if len({v.allocation_id for v in views.values()})!=len(views):raise ValueError('LayerNorm output/parameter parents must be distinct')
    allocations=unique_by(program.allocations,'allocation_id')
    for role in private:
        allocation=allocations[views[role].allocation_id]
        if (allocation.memory_domain,allocation.size_bytes,allocation.device_id)!=('logical-layernorm-cast-buffer' if width!=4 else 'logical-layernorm-contiguous-buffer',rows*hidden*4,
                allocations[source.allocation_id].device_id):
            raise ValueError('LayerNorm explicit FP32 cast storage differs from its actual wrapper')
    for role in statistics:
        allocation=allocations[views[role].allocation_id]
        if (allocation.memory_domain,allocation.size_bytes,allocation.device_id)!=('logical-layernorm-statistic',rows*4,
                allocations[source.allocation_id].device_id):
            raise ValueError('LayerNorm FP32 statistic storage differs from its declared native ABI')
    accesses=unique_by(family.accesses,'access_id')
    expected_accesses={*roles,'output',*statistics,*(('fp32_input_write','fp32_input_read') if 'fp32_input' in private else ()),
        *(('fp32_output_write','fp32_output_read') if 'fp32_output' in private else ())}
    if set(accesses)!=expected_accesses:raise ValueError('LayerNorm row roles missing')
    if any(len(accesses[role].input_ready)!=1 for role in roles):raise ValueError('LayerNorm requires actual input availability')
    ready={role:accesses[role].input_ready[0] for role in roles}
    if expected_bindings is not None:
        from src.kernel.memory_access_builder import ElementwiseBindings
        binding=expected_bindings
        if (not isinstance(binding,ElementwiseBindings)
                or (binding.read_views,binding.write_views)!=(roles,('output',))
                or program.binding_id!=binding.binding_id
                or (family.scope.call_id,family.scope.rank,family.scope.stream,family.visibility_domain)
                    !=(binding.call_id,binding.rank,binding.stream,binding.visibility_domain)
                or {role:views[role] for role in (*roles,'output')}!=unique_by(binding.views,'view_id')
                or program.allocations[:len(binding.allocations)]!=binding.allocations
                or program.events[:len(binding.events)]!=binding.events
                or tuple(ready[role] for role in roles)!=binding.input_ready):
            raise ValueError('LayerNorm source differs from actual caller bindings')
    expected=_layernorm_row_tiles(views,ready,family.scope,family.visibility_domain)
    if family!=expected:raise ValueError('LayerNorm lost its actual row input/affine READ_READY causality')
    if expected_kernel is not None:
        k=expected_kernel
        if (k.kind is not KernelKind.VECTOR or k.variant!='layernorm' or k.hardware_id!=program.hardware_id
                or k.execution_support!='supported'
                or (logical_rows,rows)!=_normalization_row_extents(k) or header!=_layernorm_header(k)
                or (hidden,source.dtype,width,affine)!=(k.shape['hidden'],k.dtype,k.shape['dtype_bytes'],k.attributes['affine'])):
            raise ValueError('LayerNorm source differs from actual kernel')
    read=sum(logical_view_unique_bytes(program,role,Operation.READ)*(logical_rows if role=='input' else rows)//rows for role in roles)
    write=logical_view_unique_bytes(program,'output',Operation.WRITE)*logical_rows//rows
    legacy_read=sum(logical_view_unique_bytes(program,role,Operation.READ)//views[role].element_bytes*width
        *(logical_rows if role=='input' else rows)//rows for role in roles)
    return {'rows':logical_rows,'execution_rows':rows,'hidden':hidden,'affine':affine,'read_bytes':read,
        'write_bytes':write,'legacy_read_bytes':legacy_read,'legacy_write_bytes':write,
        'logical_read_bytes':logical_requested_bytes(program,Operation.READ),
        'logical_write_bytes':logical_requested_bytes(program,Operation.WRITE)}


_PAIR_NORMS=('dsv32_eh_norm','dsv32_embed_eh_norm','dsv4_q_kv_rmsnorm')


def has_pairnorm_source(kernel,hardware):
    del hardware
    return kernel.kind is KernelKind.VECTOR and kernel.variant in _PAIR_NORMS and kernel.execution_support=='supported'


def _normalization_row_extents(kernel):
    logical=kernel.shape['rows']
    if 'model_input_shape' not in kernel.attributes:return logical,logical
    physical=kernel.attributes['model_input_shape']
    declared={name:value for name,value in kernel.shape.items() if name!='dtype_bytes'}
    if kernel.variant in ('rmsnorm','add_rmsnorm'):
        # KernelSpec derives tok as a row alias; lowering's model-input
        # descriptor need not repeat that derived field or dtype width.
        if (('tok' in declared and declared['tok']!=logical)
                or 'tok' in physical and physical['tok']!=physical['rows']
                or 'dtype_bytes' in physical and physical['dtype_bytes']!=kernel.shape['dtype_bytes']):
            raise ValueError('Normalization row/dtype aliases contradict its replay geometry')
        declared={name:value for name,value in declared.items() if name!='tok'}
        physical={name:value for name,value in physical.items() if name not in ('tok','dtype_bytes')}
    if (set(physical)!=set(declared)
            or any(physical[name]!=declared[name] for name in physical if name!='rows')
            or physical['rows']<logical
            or (physical['rows']!=logical and (kernel.attributes['graph_runtime_mode'] not in ('FULL','PIECEWISE')
                or physical['rows']*kernel.attributes['graph_logical_tokens']
                    !=logical*kernel.attributes['graph_model_input_tokens']))):
        raise ValueError(f'Normalization requires unchanged non-row geometry and its actual replay row mapping: '
            f'logical={dict(kernel.shape)}, physical={dict(physical)}')
    return logical,physical['rows']


def _pairnorm_header(kernel):
    logical,execution=_normalization_row_extents(kernel)
    base=('paired_normalization',kernel.variant)
    return base if logical==execution else (*base,f'logical_rows={logical}',f'execution_rows={execution}')


def pairnorm_parameter_roles(variant):
    if variant=='dsv4_q_kv_rmsnorm':return ('q_weight','kv_weight')
    if variant in ('dsv32_eh_norm','dsv32_embed_eh_norm'):return ('e_weight','h_weight')
    raise ValueError('Unknown paired normalization variant')


def _pairnorm_layouts(variant,rows,first,second):
    # Both source kernels keep reductions in registers. The MTP pair stores
    # two disjoint column slices of one actual [rows,2H] output allocation.
    if variant=='dsv4_q_kv_rmsnorm':
        return {'q':((rows,first),'bf16','q',0,(first*2,2)),
            'kv':((rows,second),'bf16','kv',0,(second*2,2)),
            'q_weight':((first,),'bf16','q_weight',0,(2,)),
            'kv_weight':((second,),'bf16','kv_weight',0,(2,)),
            'q_out':((rows,first),'bf16','q_out',0,(first*2,2)),
            'kv_out':((rows,second),'bf16','kv_out',0,(second*2,2))}
    if variant not in ('dsv32_eh_norm','dsv32_embed_eh_norm') or first!=second:
        raise ValueError('MTP paired normalization requires equal hidden widths')
    layouts={'position':((rows,),'int64','position',0,(8,)),
        'embedding':((rows,first),'bf16','embedding',0,(first*2,2)),
        'previous':((rows,first),'bf16','previous',0,(first*2,2)),
        'e_weight':((first,),'bf16','e_weight',0,(2,)),
        'h_weight':((first,),'bf16','h_weight',0,(2,)),
        'e_out':((rows,first),'bf16','output',0,(first*4,2)),
        'h_out':((rows,first),'bf16','output',first*2,(first*4,2))}
    if variant=='dsv32_embed_eh_norm':layouts['input_ids']=((rows,),'int32','input_ids',0,(4,))
    return layouts


def _pairnorm_roles(variant):
    if variant=='dsv4_q_kv_rmsnorm':return ('q','kv','q_weight','kv_weight'),('q_out','kv_out')
    if variant in ('dsv32_eh_norm','dsv32_embed_eh_norm'):
        return (('position','input_ids','embedding','previous','e_weight','h_weight') if variant=='dsv32_embed_eh_norm'
            else ('position','embedding','previous','e_weight','h_weight')),('e_out','h_out')
    raise ValueError('Unknown paired normalization roles')


def standalone_pairnorm_bindings(kernel,*,call_id,binding_id):
    from src.kernel.kernel_spec import DTYPE_BYTES
    from src.kernel.memory_access import LogicalSelectionView
    from src.kernel.memory_access_builder import ElementwiseBindings
    variant=kernel.variant;_,rows=_normalization_row_extents(kernel)
    first,second=((kernel.shape['q_hidden'],kernel.shape['kv_hidden']) if variant=='dsv4_q_kv_rmsnorm'
        else (kernel.shape['hidden'],kernel.shape['hidden']))
    layouts=_pairnorm_layouts(variant,rows,first,second);reads,writes=_pairnorm_roles(variant)
    sizes={}
    for role,(shape,dtype,parent,offset,strides) in layouts.items():
        size=offset+sum((count-1)*stride for count,stride in zip(shape,strides,strict=True))+DTYPE_BYTES[dtype]
        if parent not in sizes or size>sizes[parent]:sizes[parent]=size
    allocations=[];events=[];views=[];ready=[]
    for parent,size in sizes.items():
        identity=f'{binding_id}.{parent}'
        allocation=MemoryAllocation(identity,'logical-local-device','logical-paired-normalization',
            None if variant=='dsv32_embed_eh_norm' and parent=='embedding' else size,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
    for role,(shape,dtype,parent,offset,strides) in layouts.items():
        identity=f'{binding_id}.{parent}';width=DTYPE_BYTES[dtype]
        view=(LogicalSelectionView(role,identity,shape,dtype,width,identity+'.selected_by_input_ids',None)
            if variant=='dsv32_embed_eh_norm' and role=='embedding' else TensorView(role,identity,offset,shape,strides,dtype,width))
        views.append(view)
        if role in reads:
            event=ProgramEvent(identity+'.ready',EventKind.LOGICAL_AVAILABLE if isinstance(view,LogicalSelectionView) else EventKind.INITIALIZED,
                identity,view.region(),'pairnorm-visible',
                (EventRef(identity+'.allocated',EventKind.ALLOCATED),))
            events.append(event)
    by_id=unique_by(events,'event_id')
    for role in reads:
        parent=layouts[role][2];event=by_id[f'{binding_id}.{parent}.ready'];ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),reads,writes,
        tuple(ready),0,'offline','pairnorm-visible')


def _pairnorm_tiles(variant,views,ready,scope,visibility):
    from src.kernel.memory_access import LogicalSelectionView,TileValueIndex
    reads,writes=_pairnorm_roles(variant)
    first='q' if variant=='dsv4_q_kv_rmsnorm' else 'embedding';rows=views[first].shape[0]
    evidence=('vLLM 0.30 fused_qk_rmsnorm.py and DeepSeek V32 common/kernels.py / fused_embed_norm.py: one token row per task, mask off>=hidden.',
        'Register reductions have no global scratch; the two output rows have independent input dependencies.',
        'Folded embedding always loads selected values before the position-zero select; unknown token IDs do not authorize concrete addresses.')
    accesses=[]
    for role in reads:
        view=views[role]
        mapped=(None,) if role.endswith('_weight') else (('row',) if len(view.shape)==1 else ('row',None))
        deps=(TileDependency('input_ids',EventKind.READ_READY,()),) if variant=='dsv32_embed_eh_norm' and role=='embedding' else ()
        accesses.append(TileAccessTemplate(role,role,Operation.READ,('row',),mapped,(ready[role],),deps,evidence,
            unmapped=isinstance(view,LogicalSelectionView),value_index=TileValueIndex('input_ids','loaded_scalar')
                if variant=='dsv32_embed_eh_norm' and role=='embedding' else None))
    dependencies=(('q','q_weight'),('kv','kv_weight')) if variant=='dsv4_q_kv_rmsnorm' else (
        ('position','embedding','e_weight'),('previous','h_weight'))
    for role,inputs in zip(writes,dependencies,strict=True):
        accesses.append(TileAccessTemplate(role,role,Operation.WRITE,('row',),('row',None),(),
            tuple(TileDependency(name,EventKind.READ_READY,()) for name in inputs),evidence))
    return TileProgram(scope.scope_id,scope,(TileAxis('row',rows,1),),tuple(accesses),visibility,evidence)


def build_pairnorm_source(kernel,hardware,bindings,granularity,*,program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_pairnorm_source(kernel,hardware) or not isinstance(bindings,ElementwiseBindings):
        raise TypeError('Paired normalization requires supported typed operands')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown paired normalization granularity')
    reads,writes=_pairnorm_roles(kernel.variant)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(reads,writes,len(reads)):
        raise ValueError('Paired normalization input/output/readiness roles differ')
    scope=AccessScope(program_id,_pairnorm_header(kernel),bindings.call_id,(),bindings.rank,bindings.stream,
        'source-paired-normalization')
    family=_pairnorm_tiles(kernel.variant,unique_by(bindings.views,'view_id'),dict(zip(reads,bindings.input_ready,strict=True)),
        scope,bindings.visibility_domain)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,kernel.variant!='dsv32_embed_eh_norm',(
            'Source row tasks and exact hidden tail masks; register reductions, no invented global scratch.',
            'Full declared replay rows execute; padding values and embedding IDs stay unknown. Logical-prefix prices remain separate from these requests.',
            'Parameter logical rereads are distinct from retained unique-parameter pricing.',
            'Folded embedding capacity/ID values and exact unique footprint remain unknown; selected-row requests are not a vocabulary capacity.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    pairnorm_source_work(program,expected_kernel=kernel)
    return program


def pairnorm_source_work(program,*,expected_kernel=None):
    from src.kernel.memory_access import LogicalSelectionView
    from src.kernel.memory_path_solver import logical_requested_bytes,logical_view_requested_bytes,logical_view_unique_bytes
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:raise ValueError('Paired normalization requires a unique compressed source')
    family,=program.tile_programs;header=family.scope.op_ids
    if len(header) not in (2,4) or header[0]!='paired_normalization' or header[1] not in _PAIR_NORMS or family.scope.agent!='source-paired-normalization':
        raise ValueError('Paired normalization declaration differs')
    variant=header[1];reads,writes=_pairnorm_roles(variant);views=unique_by(program.views,'view_id')
    if set(views)!={*reads,*writes}:raise ValueError('Paired normalization exact operand roles missing')
    first=views['q' if variant=='dsv4_q_kv_rmsnorm' else 'previous']
    if not isinstance(first,TensorView) or len(first.shape)!=2:raise ValueError('Paired normalization needs typed row geometry')
    rows,width=first.shape
    logical_rows=rows
    if len(header)==4:
        logical_rows=int(header[2].removeprefix('logical_rows='))
        execution_rows=int(header[3].removeprefix('execution_rows='))
        if (header[2:]!=(f'logical_rows={logical_rows}',f'execution_rows={execution_rows}')
                or not 0<logical_rows<execution_rows or execution_rows!=rows):
            raise ValueError('Paired normalization declared replay rows differ from actual source')
    second=views['kv'].shape[1] if variant=='dsv4_q_kv_rmsnorm' and len(views['kv'].shape)==2 else width
    layouts=_pairnorm_layouts(variant,rows,width,second);parents={}
    if variant=='dsv4_q_kv_rmsnorm' and views['q'].allocation_id==views['kv'].allocation_id:
        # The real fused QKV projection supplies adjacent Q/KV column slices
        # of one complete row-major result; source row loads keep that stride.
        stride=(width+second)*2
        layouts['q']=((rows,width),'bf16','qkv_input',0,(stride,2))
        layouts['kv']=((rows,second),'bf16','qkv_input',width*2,(stride,2))
    for role,(shape,dtype,parent,offset,strides) in layouts.items():
        view=views[role]
        if variant=='dsv32_embed_eh_norm' and role=='embedding':
            if (not isinstance(view,LogicalSelectionView) or (view.shape,view.dtype,view.element_bytes,view.parent_view_id,view.coordinate_map)
                    !=(shape,dtype,2,None,())):raise ValueError('Folded embedding requires an unresolved typed selected-row parent')
            allocation=unique_by(program.allocations,'allocation_id')[view.allocation_id]
            if allocation.size_bytes is not None:raise ValueError('Selected rows cannot invent vocabulary capacity')
        elif (not isinstance(view,TensorView) or (view.shape,view.dtype,view.offset_bytes,view.strides_bytes)!=(shape,dtype,offset,strides)):
            raise ValueError('Paired normalization tensor ABI differs')
        if parent in parents and parents[parent]!=view.allocation_id:raise ValueError('MTP concat outputs must retain their shared parent')
        parents[parent]=view.allocation_id
    if len(set(parents.values()))!=len(parents):raise ValueError('Paired normalization unrelated parents must not alias')
    accesses=unique_by(family.accesses,'access_id')
    if set(accesses)!={*reads,*writes} or any(len(accesses[role].input_ready)!=1 for role in reads):
        raise ValueError('Paired normalization actual input reads missing')
    ready={role:accesses[role].input_ready[0] for role in reads}
    if family!=_pairnorm_tiles(variant,views,ready,family.scope,family.visibility_domain):
        raise ValueError('Paired normalization lost same-row causal dependencies')
    if expected_kernel is not None:
        kernel=expected_kernel
        expected_widths=((kernel.shape['q_hidden'],kernel.shape['kv_hidden']) if kernel.variant=='dsv4_q_kv_rmsnorm'
            else (kernel.shape['hidden'],kernel.shape['hidden'])) if kernel.variant in _PAIR_NORMS else None
        if (kernel.kind is not KernelKind.VECTOR or kernel.variant!=variant or kernel.hardware_id!=program.hardware_id
                or kernel.execution_support!='supported' or kernel.dtype!='bf16'
                or (logical_rows,rows)!=_normalization_row_extents(kernel) or family.scope.op_ids!=_pairnorm_header(kernel)
                or (width,second)!=expected_widths):
            raise ValueError('Paired normalization source differs from actual kernel')
    parameters=pairnorm_parameter_roles(variant)
    read=sum(logical_view_unique_bytes(program,role,Operation.READ) if role in parameters
        else logical_view_requested_bytes(program,role,Operation.READ)//rows*logical_rows for role in reads)
    return {'variant':variant,'rows':logical_rows,'execution_rows':rows,'first_hidden':width,'second_hidden':second,'read_bytes':read,
        'write_bytes':logical_requested_bytes(program,Operation.WRITE)//rows*logical_rows,'logical_read_bytes':logical_requested_bytes(program,Operation.READ),
        'logical_write_bytes':logical_requested_bytes(program,Operation.WRITE),
        'selected_embedding_unique_bytes':None if variant=='dsv32_embed_eh_norm' else 0}


def has_gpu_normalization_source(kernel, hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant in ("rmsnorm", "add_rmsnorm")
            and kernel.execution_support == "supported" and hardware.kernel_memory_model == "gpu_mshr"
            and (("input_reread_count" in kernel.launch.parameters
                  and kernel.launch.parameters["input_reread_count"] == 1.0)
                 or ("input_reread_count" not in kernel.launch.parameters
                     and kernel.launch.algorithm == f"observed_vector:{kernel.variant}"))
            and "rstd_output_bytes" not in kernel.launch.parameters)


def has_pim_normalization_source(kernel, hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant in ('rmsnorm','add_rmsnorm')
        and kernel.execution_support=='supported' and hardware.kernel_memory_model=='pim_bank'
        and 'rstd_output_bytes' not in kernel.launch.parameters)


def build_pim_normalization(kernel, hardware, bindings, granularity, *, program_id):
    """Public row tensors; bank command multiplicity remains a physical projection."""
    from dataclasses import replace
    if not has_pim_normalization_source(kernel,hardware):raise ValueError('PIM normalization requires its declared tensor ABI')
    # Reuse common typed row operands, complete reductions and in-place alias
    # validation. The bank adapter has a distinct add -> normalize dataflow.
    base=build_gpu_normalization(kernel,hardware,bindings,granularity,program_id=program_id)
    family,=base.tile_programs
    evidence=('VECTOR RMSNorm tensor ABI: each output row consumes its full hidden input and gamma row.',
        'PIM bank_stream_model normalization_stream: reduction followed by rescaling the same row; protocol commands are not tensor read identities.',
        'PIM add-RMSNorm first materializes its residual sum, then normalizes that produced state. Independent rows have no global barrier.')
    accesses=[]
    for access in family.accesses:
        access=replace(access,evidence_refs=evidence)
        if kernel.variant=='add_rmsnorm' and access.access_id=='reread':
            accesses.append(replace(access,access_id='normalization_input'))
            access=replace(access,dependencies=(TileDependency('normalization_input',EventKind.READ_READY,()),))
        accesses.append(access)
    scope=replace(family.scope,op_ids=('pim_normalization',kernel.variant),agent='logical-pim-row-reduction')
    program=replace(base,coverage=Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
        'Complete mathematical row inputs/outputs and produced-residual reloads; no native PIM launch or bank-address trace is inferred.',
        'Reduction scalar state stays in the primitive accumulator; gamma is an explicit logical operand.',
        'Bank command counts, buffer refills, row cycles and scheduling remain the existing separate protocol projection.')),
        tile_programs=(replace(family,scope=scope,accesses=tuple(accesses),evidence_refs=evidence),))
    pim_normalization_source_work(program,expected_kernel=kernel)
    return program


def pim_normalization_source_work(program, *, expected_kernel=None):
    from src.kernel.memory_access import requested_bytes,tile_template_requested_bytes
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:
        raise ValueError('PIM normalization requires its unique compressed row source')
    family,=program.tile_programs;header=family.scope.op_ids
    if (family.scope.agent!='logical-pim-row-reduction' or header not in
            (('pim_normalization','rmsnorm'),('pim_normalization','add_rmsnorm'))
            or family.causal_prefixes or family.partitions or len(family.axes)!=1):
        raise ValueError('PIM normalization requires its declared independent row partition')
    fused=header[1]=='add_rmsnorm';roles=unique_by(family.accesses,'access_id');views=unique_by(program.views,'view_id')
    expected={'input','reread','weight','output'}|({'residual','residual_update','normalization_input'} if fused else set())
    if set(roles)!=expected:raise ValueError('PIM normalization must retain every tensor pass')
    source=views[roles['input'].view_id];gamma=views[roles['weight'].view_id];output=views[roles['output'].view_id]
    if len(source.shape)!=2:raise ValueError('PIM normalization requires row/hidden tensors')
    rows,hidden=source.shape;width=source.element_bytes;axis,=family.axes
    if expected_kernel is not None:
        if (expected_kernel.kind is not KernelKind.VECTOR or header[1]!=expected_kernel.variant
                or (program.hardware_id,rows,hidden,source.dtype,width)!=(expected_kernel.hardware_id,
                    expected_kernel.shape['rows'],expected_kernel.shape['hidden'],expected_kernel.dtype,expected_kernel.shape['dtype_bytes'])
                or 'rstd_output_bytes' in expected_kernel.launch.parameters):
            raise ValueError('PIM normalization source differs from its actual kernel')
    if (axis.axis_id,axis.extent,axis.block)!=('row',rows,1):raise ValueError('PIM normalization row extent changed')
    dependencies={'output':(TileDependency('reread',EventKind.READ_READY,()),TileDependency('weight',EventKind.READ_READY,())),
        'reread':(TileDependency('normalization_input',EventKind.READ_READY,()) if fused else TileDependency('input',EventKind.READ_READY,()),)}
    if fused:
        dependencies['residual_update']=(TileDependency('input',EventKind.READ_READY,()),TileDependency('residual',EventKind.READ_READY,()))
        dependencies['normalization_input']=(TileDependency('residual_update',EventKind.WRITE_VISIBLE,()),)
    for role,access in roles.items():
        view=views[access.view_id];weight=role=='weight';write=role in ('output','residual_update')
        layout=((hidden,),(width,),source.dtype,width) if weight else ((rows,hidden),(hidden*width,width),source.dtype,width)
        if (view.shape,view.strides_bytes,view.dtype,view.element_bytes)!=layout:
            raise ValueError('PIM normalization actual tensor shape/dtype/stride changed')
        if (access.operation is not (Operation.WRITE if write else Operation.READ) or access.iteration_axes!=('row',)
                or access.view_axes!=((None,) if weight else ('row',None)) or access.unmapped or access.indirect_target is not None
                or access.view_axis_divisors or access.view_axis_padding or access.constant_value is not None or access.scope is not None):
            raise ValueError('PIM normalization must conserve its actual row selections')
        expected_dependencies=dependencies[role] if role in dependencies else ()
        if access.dependencies!=expected_dependencies or (bool(access.input_ready) if write else len(access.input_ready)!=1):
            raise ValueError('PIM normalization lacks actual input or produced-residual readiness')
    if fused:
        residual=views[roles['residual'].view_id]
        if (output!=source or any(views[roles[role].view_id]!=residual
                for role in ('residual_update','normalization_input','reread'))
                or len({view.allocation_id for view in (source,residual,gamma)})!=3):
            raise ValueError('PIM add-RMSNorm must retain its actual in-place tensor identities')
    elif (views[roles['reread'].view_id]!=source or len({view.allocation_id for view in (source,gamma,output)})!=3):
        raise ValueError('PIM RMSNorm must reread its own input and publish its distinct output')
    volumes={role:tile_template_requested_bytes(program,family,access) for role,access in roles.items()}
    return {'rows':rows,'hidden':hidden,'output_elements':volumes['output']//width,'fused':fused,
        'input_bytes':volumes['input'],'initial_read_bytes':volumes['input']+(volumes['residual'] if fused else 0),
        'weight_bytes':requested_bytes(gamma.region()),'logical_weight_read_bytes':volumes['weight'],
        'reread_bytes':volumes['reread']+(volumes['normalization_input'] if fused else 0),
        'output_bytes':volumes['output'],'residual_write_bytes':volumes['residual_update'] if fused else 0,
        'resident_footprint_bytes':sum(requested_bytes(view.region()) for view in
            ((source,residual,gamma) if fused else (source,gamma,output)))}


def normalization_operand_records(kernel, *, binding_id, device_id):
    rows, hidden, width = (kernel.shape[name] for name in ("rows", "hidden", "dtype_bytes"))
    rows=_normalization_row_extents(kernel)[1]
    fused = kernel.variant == "add_rmsnorm"
    names = ("input", "residual", "weight") if fused else ("input", "weight", "output")
    allocations, views = [], []
    for name in names:
        shape = (hidden,) if name == "weight" else (rows, hidden)
        size = hidden*width if name == "weight" else rows*hidden*width
        identity = f"{binding_id}.{name}"
        allocation = MemoryAllocation(identity, device_id, "logical-storage", size,
            AddressLevel.OBJECT, None, None, f"{identity}.allocated", None)
        view = TensorView(name, identity, 0, shape, (width,) if name == "weight" else (hidden*width, width), kernel.dtype, width)
        allocations.append(allocation)
        views.append(view)
    read_names = ("input", "residual", "weight") if fused else ("input", "weight")
    write_names = ("input", "residual") if fused else ("output",)
    return tuple(allocations),tuple(views),read_names,write_names


def validate_normalization_operands(kernel,operands,read_roles,write_roles):
    fused=kernel.variant=='add_rmsnorm'
    views = unique_by(operands, "view_id")
    reads = tuple(views[name] for name in read_roles)
    writes = tuple(views[name] for name in write_roles)
    rows, hidden, width = (kernel.shape[name] for name in ("rows", "hidden", "dtype_bytes"))
    logical_rows=rows
    logical_rows,rows=_normalization_row_extents(kernel)
    for index, view in enumerate((*reads, *writes)):
        is_weight = index == len(reads)-1
        shape = (hidden,) if is_weight else (rows, hidden)
        strides = (width,) if is_weight else (hidden*width, width)
        if (view.shape, view.strides_bytes, view.dtype, view.element_bytes) != (shape, strides, kernel.dtype, width):
            raise ValueError("normalization operand tensor storage mismatch")
    if len({view.allocation_id for view in reads}) != len(reads):
        raise NotImplementedError("normalization input/weight alias needs a declared source lifecycle")
    if fused:
        if any(output != source for output, source in zip(writes, reads[:2])):
            raise ValueError("fused add-RMSNorm must update its actual input and residual views in place")
    elif writes[0].allocation_id in {view.allocation_id for view in reads}:
        raise NotImplementedError("RMSNorm restrict output cannot alias its input or weight")
    return views,reads,writes


@lru_cache(maxsize=2)
def normalization_access_patterns(fused):
    def template(role,view,operation,dependencies=()):
        return TileAccessTemplate(role,view,operation,('row',),(None,) if view=='weight' else ('row',None),(),dependencies,())
    accesses=[template('input','input',Operation.READ)]
    if fused:
        accesses.extend((template('residual','residual',Operation.READ),
            template('residual_update','residual',Operation.WRITE,(
                TileDependency('input',EventKind.READ_READY,()),TileDependency('residual',EventKind.READ_READY,())))))
        prior=TileDependency('residual_update',EventKind.WRITE_VISIBLE,())
    else:prior=TileDependency('input',EventKind.READ_READY,())
    accesses.extend((template('reread','residual' if fused else 'input',Operation.READ,(prior,)),
        template('weight','weight',Operation.READ),template('output','input' if fused else 'output',Operation.WRITE,(
            TileDependency('reread',EventKind.READ_READY,()),TileDependency('weight',EventKind.READ_READY,())))))
    return tuple(accesses)


def standalone_normalization_bindings(kernel, hardware, *, call_id, binding_id,
        device_id="logical-local-device",rank=0,stream="offline",ready_event_prefix=""):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if type(ready_event_prefix) is not str:raise TypeError('Normalization readiness prefix must be a string')
    allocations,views,read_names,write_names=normalization_operand_records(kernel,binding_id=binding_id,device_id=device_id)
    events=[];ready=[]
    for allocation,view in zip(allocations,views,strict=True):
        identity=allocation.allocation_id
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if view.view_id in read_names:
            initialized=f'{ready_event_prefix}{identity}.initialized'
            events.append(ProgramEvent(initialized,EventKind.INITIALIZED,identity,view.region(),'norm-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),)))
            ready.append(EventRef(initialized,EventKind.INITIALIZED))
    return ElementwiseBindings(binding_id,call_id,allocations,views,tuple(events),read_names,write_names,tuple(ready),rank,stream,'norm-visible')


def build_gpu_normalization(kernel, hardware, bindings, granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings, ElementwiseBindings):
        raise TypeError("normalization requires typed tensor operand bindings")
    if granularity not in (Granularity.OP, Granularity.KERNEL,Granularity.TILE):
        raise ValueError("unknown normalization logical granularity")
    fused = kernel.variant == "add_rmsnorm"
    if (len(bindings.read_views), len(bindings.write_views), len(bindings.input_ready)) != ((3, 2, 3) if fused else (2, 1, 2)):
        raise ValueError("normalization operand/readiness count mismatch")
    views,reads,writes=validate_normalization_operands(kernel,bindings.views,bindings.read_views,bindings.write_views)
    logical_rows,rows=_normalization_row_extents(kernel)
    evidence=("vLLM RMSNorm/fused-add RMSNorm source: each logical row owns its full hidden reduction and both data passes.",
        "One gamma selection per row; input/residual output aliases and actual first-pass/update readiness are retained.",
        "Row tiles are semantic reduction units, not an inferred native issue schedule or a global row barrier.")
    from dataclasses import replace
    ready=dict(zip(bindings.read_views,bindings.input_ready,strict=True))
    accesses=tuple(replace(access,input_ready=(ready[access.view_id],) if access.operation is Operation.READ else (),
        evidence_refs=evidence) for access in normalization_access_patterns(fused))
    header=('normalization',) if logical_rows==rows else ('normalization',f'logical_rows={logical_rows}',f'execution_rows={rows}')
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-row-reduction')
    family=TileProgram(program_id,scope,(TileAxis('row',rows,1),),tuple(accesses),bindings.visibility_domain,evidence)
    return MemoryAccessProgram("midlayer.memory-access.v1",program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,True,(
            "Compressed independent row reductions with actual two-pass reads and in-place residual publication.",
            "Reduction state stays on chip; no fabricated global scratch. Weight cache sharing and physical transactions remain separate projections.",
            "NPU rstd/transfer ABI and other multipass/register-retained domains remain separate source patterns.")),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))


def has_public_normalization_source(kernel,hardware):
    """Declared row tensors outside the more specific native source domains."""
    from src.kernel.memory_cann_normalization import has_cann_normalization_source
    return (kernel.kind is KernelKind.VECTOR and kernel.variant in ('rmsnorm','add_rmsnorm')
        and kernel.execution_support=='supported'
        and not has_gpu_normalization_source(kernel,hardware)
        and not has_pim_normalization_source(kernel,hardware)
        and not has_cann_normalization_source(kernel,hardware))


def _public_rstd_bytes(kernel):
    if 'rstd_output_bytes' not in kernel.launch.parameters:return 0
    count=kernel.launch.parameters['rstd_output_bytes']
    if count<0 or count!=int(count):raise ValueError('Declared rstd storage requires integral non-negative bytes')
    return int(count)


def standalone_public_normalization_bindings(kernel,hardware,*,call_id,binding_id):
    from dataclasses import replace
    binding=standalone_normalization_bindings(kernel,hardware,call_id=call_id,binding_id=binding_id)
    count=_public_rstd_bytes(kernel)
    if not count:return binding
    identity=binding_id+'.rstd'
    allocation=MemoryAllocation(identity,'logical-local-device','logical-storage',count,
        AddressLevel.OBJECT,None,None,identity+'.allocated',None)
    # The launch declares a byte extent, not an independently proven scalar
    # encoding for this otherwise unbound side output.
    view=TensorView('rstd',identity,0,(count,),(1,),'uint8',1)
    event=ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,())
    return replace(binding,allocations=(*binding.allocations,allocation),views=(*binding.views,view),
        events=(*binding.events,event),write_views=(*binding.write_views,'rstd'))


def build_public_normalization(kernel,hardware,bindings,granularity,*,program_id):
    from dataclasses import replace
    if not has_public_normalization_source(kernel,hardware):raise ValueError('Not a public normalization source domain')
    count=_public_rstd_bytes(kernel);core=bindings
    if count:
        views=unique_by(bindings.views,'view_id')
        if 'rstd' not in views or bindings.write_views[-1:]!=('rstd',):
            raise ValueError('Public normalization requires its declared rstd output')
        view=views['rstd']
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes)!=((count,),'uint8',1,(1,)):
            raise ValueError('Public normalization rstd byte extent differs from its declaration')
        if view.allocation_id in {v.allocation_id for v in bindings.views if v.view_id!='rstd'}:
            raise ValueError('Public rstd storage cannot silently alias a normalization operand')
        core=replace(bindings,allocations=tuple(a for a in bindings.allocations if a.allocation_id!=view.allocation_id),
            views=tuple(v for v in bindings.views if v.view_id!='rstd'),
            events=tuple(e for e in bindings.events if e.allocation_id!=view.allocation_id),
            write_views=bindings.write_views[:-1])
    if {v.view_id for v in core.views}!=set(core.read_views)|set(core.write_views):
        raise ValueError('Public normalization has undeclared tensor roles')
    base=build_gpu_normalization(kernel,hardware,core,granularity,program_id=program_id)
    family,=base.tile_programs
    accesses=family.accesses
    if count:
        dependencies=(TileDependency('input',EventKind.READ_READY,('row',)),)
        if kernel.variant=='add_rmsnorm':dependencies+= (TileDependency('residual',EventKind.READ_READY,('row',)),)
        accesses+= (TileAccessTemplate('rstd','rstd',Operation.WRITE,(),(None,),(),dependencies,
            ('Declared rstd byte buffer; scalar dtype, native issue and row-to-byte mapping remain unbound.',)),)
    limitations=(*base.coverage.limitations,
        'Public row-value dataflow only; no native multipass, register retention or physical traffic is inferred.',
        'Existing launch-declared input rereads and rstd transfer padding remain explicit physical price projections.')
    return replace(base,allocations=bindings.allocations,views=bindings.views,events=bindings.events,
        coverage=replace(base.coverage,executable=False,limitations=limitations),
        tile_programs=(replace(family,scope=replace(family.scope,agent='logical-public-normalization'),accesses=accesses),))


def public_normalization_source_work(program,*,expected_kernel,hardware):
    from dataclasses import replace
    from src.kernel.memory_access import tile_template_requested_bytes
    from src.kernel.memory_path_solver import normalization_source_bytes
    program.__post_init__()
    if (program.hardware_id!=hardware.hardware_id or hardware.hardware_id!=expected_kernel.hardware_id
            or not has_public_normalization_source(expected_kernel,hardware)
            or len(program.tile_programs)!=1 or program.accesses):
        raise ValueError('Public normalization source does not match its kernel/hardware')
    family,=program.tile_programs
    if family.scope.agent!='logical-public-normalization' or program.coverage.executable:
        raise ValueError('Public normalization cannot claim native/executable source fidelity')
    roles=unique_by(family.accesses,'access_id');count=_public_rstd_bytes(expected_kernel)
    if bool(count)!=('rstd' in roles):raise ValueError('Public normalization rstd source differs from its declaration')
    if count:
        access=roles['rstd'];view=unique_by(program.views,'view_id')[access.view_id]
        dependencies=(TileDependency('input',EventKind.READ_READY,('row',)),)
        if expected_kernel.variant=='add_rmsnorm':dependencies+=(TileDependency('residual',EventKind.READ_READY,('row',)),)
        if ((access.operation,access.iteration_axes,access.view_axes,access.input_ready,access.dependencies)
                !=(Operation.WRITE,(),(None,),(),dependencies)
                or (view.shape,view.dtype,view.element_bytes)!=((count,),'uint8',1)
                or tile_template_requested_bytes(program,family,access)!=count):
            raise ValueError('Public rstd output lost its declared bytes or input reduction dependencies')
    base=replace(program,tile_programs=(replace(family,accesses=tuple(a for a in family.accesses if a.access_id!='rstd')),))
    work=normalization_source_bytes(base,expected_kernel=expected_kernel)
    # Logical two-pass value consumption does not establish how often a
    # compiler reloads external storage. Keep that pre-existing launch fact
    # separate from the public tensor dataflow.
    repeat=expected_kernel.launch.parameters['input_reread_count'] if 'input_reread_count' in expected_kernel.launch.parameters else 0
    if repeat<0 or repeat!=int(repeat):raise ValueError('Normalization input reread count must be a non-negative integer')
    work['reread_bytes']=int(repeat)*work['input_bytes']
    work['rstd_bytes']=count
    work['resident_footprint_bytes']+=count
    return work
