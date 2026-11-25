"""Grouped reduction inputs and caller-declared or explicitly opaque results."""
from src.kernel.kernel_spec import KernelKind,DTYPE_BYTES
from math import prod
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,MemoryAllocation,
    MemoryAccessProgram,Operation,Ordering,ProgramEvent,Semantics,TensorView,LogicalSelectionView,
    LogicalPartialSelectionView,LogicalCoordinateBinding,LogicalCoordinateTerm,TileValueIndex,
    TileAxis,TileAccessTemplate,TileDependency,TileProgram,unique_by,
)

# Retained only for decoding the historical native source declaration when a
# caller explicitly supplies that declaration. Current KernelSpec no longer
# manufactures this field; current SGLang EAGLE top-k uses the typed
# ``vllm_mtp_eager`` operations instead.
_LEGACY_SGLANG_TOPK1_BLOCK = 8192


def has_generic_reduction_source(kernel,hardware):
    del hardware
    return (kernel.kind is KernelKind.VECTOR and kernel.variant=='reduce'
        and kernel.execution_support=='supported')


def _group_extents(kernel):
    logical=kernel.shape['element_count']
    if 'model_input_shape' not in kernel.attributes:return logical,logical
    physical=kernel.attributes['model_input_shape']
    if set(physical)!={'element_count'}:
        raise ValueError('Grouped reduction replay requires its exact output-group geometry')
    execution=physical['element_count']
    if (execution<logical or (execution!=logical and (
            execution*kernel.attributes['graph_logical_tokens']!=logical*kernel.attributes['graph_model_input_tokens']
            or kernel.attributes['graph_runtime_mode'] not in ('FULL','PIECEWISE')))):
        raise ValueError('Grouped reduction replay differs from its actual graph group mapping')
    return logical,execution


def _header(kernel):
    logical,execution=_group_extents(kernel)
    base=('grouped_reduction',str(kernel.attributes['reduction_factor']))
    return base if logical==execution else (*base,f'logical_groups={logical}',f'execution_groups={execution}')


def standalone_generic_reduction_bindings(kernel,*,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if 'reduction_program' in kernel.attributes and kernel.attributes['reduction_program'] is not None:
        return _native_bindings(kernel,call_id=call_id,binding_id=binding_id)
    _,count=_group_extents(kernel);factor=kernel.attributes['reduction_factor'];width=kernel.shape['dtype_bytes']
    source=MemoryAllocation(binding_id+'.input','logical-local-device','logical-reduction-input',count*factor*width,
        AddressLevel.OBJECT,None,None,binding_id+'.input.allocated',None)
    result_dtype=(kernel.attributes['result_dtype']
        if 'result_dtype' in kernel.attributes else None)
    result_width=None if result_dtype is None else DTYPE_BYTES[result_dtype]
    destination=MemoryAllocation(binding_id+'.output','logical-local-device','logical-reduction-output',
        None if result_width is None else count*result_width,
        AddressLevel.OBJECT,None,None,binding_id+'.output.allocated',None)
    input_view=TensorView('input',source.allocation_id,0,(count,factor),(factor*width,width),kernel.dtype,width)
    output=(LogicalSelectionView('output',destination.allocation_id,(count,),None,None,
        binding_id+'.output_encoding_unbound',None) if result_width is None else
        TensorView('output',destination.allocation_id,0,(count,),(result_width,),result_dtype,result_width))
    initialized=ProgramEvent(binding_id+'.input.ready',EventKind.INITIALIZED,source.allocation_id,input_view.region(),
        'reduction-visible',(EventRef(source.allocated_event,EventKind.ALLOCATED),))
    return ElementwiseBindings(binding_id,call_id,(source,destination),(input_view,output),(
        ProgramEvent(source.allocated_event,EventKind.ALLOCATED,source.allocation_id,None,None,()),
        ProgramEvent(destination.allocated_event,EventKind.ALLOCATED,destination.allocation_id,None,None,()),initialized),
        ('input',),('output',),(EventRef(initialized.event_id,initialized.kind),),0,'offline','reduction-visible')


def _tiles(views,ready,scope,visibility):
    count,factor=views['input'].shape
    evidence=('KernelSpec VECTOR/reduce: element_count output groups and reduction_factor inputs per group.',
        'Serving workload recipes use one output per sum/argmax group, e.g. shared-weight-sum and grouped vocabulary reductions.',
        'A typed result uses only the actual caller declaration; an unspecified descriptor remains opaque. Reduction operator and native partial scratch/grid algorithm are not guessed.')
    accesses=(TileAccessTemplate('input','input',Operation.READ,('row','reduce'),('row','reduce'),(ready,),(),evidence),
        TileAccessTemplate('output','output',Operation.WRITE,('row',),('row',),(),
            (TileDependency('input',EventKind.READ_READY,('reduce',)),),evidence,
            unmapped=isinstance(views['output'],LogicalSelectionView)))
    return TileProgram(scope.scope_id,scope,(TileAxis('row',count,1),TileAxis('reduce',factor,1)),accesses,visibility,evidence)


def build_generic_reduction_source(kernel,hardware,bindings,granularity,*,program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_generic_reduction_source(kernel,hardware) or not isinstance(bindings,ElementwiseBindings):
        raise TypeError('Generic reduction requires supported typed inputs and an explicit result role')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown generic reduction granularity')
    if 'reduction_program' in kernel.attributes and kernel.attributes['reduction_program'] is not None:
        return _native_source(kernel,hardware,bindings,program_id=program_id)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(('input',),('output',),1):
        raise ValueError('Generic reduction input/output/readiness roles differ')
    scope=AccessScope(program_id,_header(kernel),bindings.call_id,(),
        bindings.rank,bindings.stream,'logical-grouped-reduction')
    family=_tiles(unique_by(bindings.views,'view_id'),bindings.input_ready[0],scope,bindings.visibility_domain)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Each logical output group waits for all its own input elements; independent groups have no cross-row barrier.',
            'Output dtype/capacity/selection values are unbound unless a caller explicitly supplies a typed output. No sum or INT64 argmax is guessed.',
            'Legacy price uses input footprint as total traffic, with an input-width output projection; it is not asserted as actual read/write traffic.',
            'Native scratch, multi-CTA partial layout, register reduction and issued addresses remain unbound.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    generic_reduction_source_work(program,expected_kernel=kernel,expected_bindings=bindings);return program


def generic_reduction_source_work(program,*,expected_kernel,expected_bindings=None):
    if (expected_kernel.kind is not KernelKind.VECTOR or expected_kernel.variant!='reduce'
            or expected_kernel.execution_support!='supported'):
        raise ValueError('Generic reduction source requires its supported reduction kernel')
    if ('reduction_program' in expected_kernel.attributes
            and expected_kernel.attributes['reduction_program'] is not None):
        return _native_source_work(program,expected_kernel,expected_bindings)
    from src.kernel.memory_path_solver import logical_view_requested_bytes
    program.__post_init__();kernel=expected_kernel
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant!='reduce' or kernel.execution_support!='supported'
            or kernel.hardware_id!=program.hardware_id):raise ValueError('Generic reduction source differs from actual kernel')
    if program.accesses or program.scopes or len(program.tile_programs)!=1:raise ValueError('Generic reduction requires a unique compressed source')
    family,=program.tile_programs;views=unique_by(program.views,'view_id')
    if expected_bindings is not None:
        from src.kernel.memory_access_builder import ElementwiseBindings
        if (not isinstance(expected_bindings,ElementwiseBindings)
                or program.binding_id!=expected_bindings.binding_id
                or (family.scope.call_id,family.scope.rank,family.scope.stream,family.visibility_domain)
                    !=(expected_bindings.call_id,expected_bindings.rank,expected_bindings.stream,expected_bindings.visibility_domain)
                or views!=unique_by(expected_bindings.views,'view_id')
                or unique_by(program.allocations,'allocation_id')!=unique_by(expected_bindings.allocations,'allocation_id')
                or unique_by(program.events,'event_id')!=unique_by(expected_bindings.events,'event_id')):
            raise ValueError('Generic reduction source differs from actual caller input/result bindings')
    if (set(views)!={'input','output'} or family.scope.op_ids!=_header(kernel)
            or family.scope.agent!='logical-grouped-reduction'):raise ValueError('Generic reduction source declaration differs')
    source,output=views['input'],views['output'];logical,count=_group_extents(kernel);factor=kernel.attributes['reduction_factor']
    width=kernel.shape['dtype_bytes']
    if (not isinstance(source,TensorView) or (source.shape,source.dtype,source.element_bytes,source.strides_bytes[-1])
            !=((count,factor),kernel.dtype,width,width) or source.strides_bytes[0]<factor*width):
        raise ValueError('Generic reduction input group geometry or dtype differs')
    if isinstance(output,LogicalSelectionView):
        allocation=unique_by(program.allocations,'allocation_id')[output.allocation_id]
        if ((output.shape,output.dtype,output.element_bytes,output.parent_view_id,output.coordinate_map)
                !=((count,),None,None,None,()) or allocation.size_bytes is not None):
            raise ValueError('Opaque reduction result cannot invent dtype, capacity or parent mapping')
        actual_write=None
    elif isinstance(output,TensorView):
        if output.shape!=(count,) or output.strides_bytes!=(output.element_bytes,):raise ValueError('Typed reduction result extent or stride differs')
        actual_write=logical_view_requested_bytes(program,'output',Operation.WRITE)
    else:raise TypeError('Generic reduction result must be typed or explicitly opaque')
    result_dtype=(kernel.attributes['result_dtype']
        if 'result_dtype' in kernel.attributes else None)
    if result_dtype is not None and (not isinstance(output,TensorView)
            or (output.dtype,output.element_bytes)!=(result_dtype,DTYPE_BYTES[result_dtype])):
        raise ValueError('Reduction output differs from declared result dtype')
    if source.allocation_id==output.allocation_id:raise ValueError('Unspecified reduction result cannot silently alias its input')
    accesses=unique_by(family.accesses,'access_id')
    if set(accesses)!={'input','output'} or len(accesses['input'].input_ready)!=1:raise ValueError('Generic reduction actual input readiness missing')
    if expected_bindings is not None and accesses['input'].input_ready!=expected_bindings.input_ready:
        raise ValueError('Generic reduction input readiness differs from actual caller')
    if family!=_tiles(views,accesses['input'].input_ready[0],family.scope,family.visibility_domain):
        raise ValueError('Generic reduction lost its full own-group READ_READY dependency')
    read=logical_view_requested_bytes(program,'input',Operation.READ)
    # Preserve the historical N*F*input_width total and its direction split.
    # The nominal result-width assumption remains a named price projection,
    # never used to type the actual public result or claim requested bytes.
    legacy_total=read*logical//count;legacy_output=logical*source.element_bytes
    return {'input_elements':logical*factor,'output_elements':logical,'input_elements_per_output':factor,
        'execution_input_elements':count*factor,'execution_output_elements':count,
        'logical_read_bytes':read,'logical_write_bytes':actual_write,'output_encoding_known':actual_write is not None,
        'legacy_read_bytes':legacy_total-legacy_output,'legacy_write_bytes':legacy_output,'legacy_total_bytes':legacy_total}


def _native_geometry(kernel):
    declaration=kernel.attributes['reduction_program']
    if declaration is None:raise ValueError('Native topk1 requires its explicit source declaration')
    if kernel.hardware_id not in ('nvidia_a800','nvidia_h200'):
        raise ValueError('SGLang topk1 program requires its CUDA source path')
    partial=declaration['name']=='sglang_v0519_topk1_partial'
    splits=-(-declaration['vocab']//_LEGACY_SGLANG_TOPK1_BLOCK) if partial else declaration['splits']
    logical,count=_group_extents(kernel)
    if partial:
        legacy_groups=logical//declaration['rows']
        if not legacy_groups or logical%declaration['rows'] or count%legacy_groups:
            raise ValueError('Topk1 replay must preserve full per-request legacy groups')
        rows=count//legacy_groups
    else:rows=count
    return declaration,partial,rows,splits,logical,count


def _native_layouts(kernel):
    declaration,partial,rows,splits,_,_=_native_geometry(kernel)
    if partial:
        return {'input':((rows,declaration['vocab']),'float32')},{
            'output':((rows,splits),'float32'),'indices':((rows,splits),'int32')}
    inputs={'input':((rows,splits),'float32'),'indices':((rows,splits),'int32'),'positions':((rows,),'int64')}
    outputs={'output':((rows,1),'int64'),'probability':((rows,1),'float32'),'positions':((rows,),'int64')}
    if declaration['draft_columns']:outputs['draft_tokens']=((rows,),'int64')
    return inputs,outputs


def _native_strides(shape,width):
    return tuple(prod(shape[i+1:])*width for i in range(len(shape)))


def _native_bindings(kernel,*,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    declaration,_,rows,_,_,_=_native_geometry(kernel);inputs,outputs=_native_layouts(kernel)
    allocations=[];views=[];events=[];ready=[]
    for role,(shape,dtype) in {**inputs,**outputs}.items():
        width=DTYPE_BYTES[dtype];identity=f'{binding_id}.{role}'
        strides=_native_strides(shape,width);offset=0;size=prod(shape)*width
        if role=='draft_tokens':
            strides=(declaration['draft_columns']*width,);offset=declaration['draft_column']*width
            size=rows*declaration['draft_columns']*width
        allocation=MemoryAllocation(identity,'logical-local-device','logical-reduction-native',size,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        view=TensorView(role,identity,offset,shape,strides,dtype,width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role in inputs:
            event=ProgramEvent(identity+'.ready',EventKind.INITIALIZED,identity,view.region(),'reduction-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),tuple(inputs),tuple(outputs),
        tuple(ready),0,'offline','reduction-visible')


def _native_tiles(kernel,views,ready,scope,visibility):
    declaration,partial,rows,splits,_,_=_native_geometry(kernel)
    evidence=('SGLang v0.5.19 kernels/ops/speculative/topk1.py: BLOCK8192 valid vocabulary loads, FP32 partial values and INT32 global indices.',
        'Finalize reads all partial values, selects one INT32 index per row, writes INT64 index and FP32 probability, and updates positions in place.',
        'Optional draft-token write covers only the declared column; disabled dummy pointer has no access. Compiler-private state and selected values/addresses remain unbound.')
    dep=lambda name,axes=():TileDependency(name,EventKind.READ_READY,axes)
    extra=()
    if partial:
        axes=(TileAxis('row',rows,1),TileAxis('vocab',declaration['vocab'],_LEGACY_SGLANG_TOPK1_BLOCK))
        accesses=(TileAccessTemplate('input','input',Operation.READ,('row','vocab'),('row','vocab'),(ready['input'],),(),evidence),
            *(TileAccessTemplate(role,role,Operation.WRITE,('row','vocab'),('row','vocab'),(),(dep('input'),),evidence,
                view_axis_divisors=(1,_LEGACY_SGLANG_TOPK1_BLOCK)) for role in ('output','indices')))
    else:
        axes=(TileAxis('row',rows,1),TileAxis('split',splits,1))
        positions_unmapped=ready['positions'].kind is EventKind.LOGICAL_AVAILABLE
        selected=(TensorView('selected_indices',views['indices'].allocation_id,views['indices'].offset_bytes,
            (rows,),(views['indices'].strides_bytes[0],),'int32',4) if splits==1 else
            LogicalPartialSelectionView('selected_indices',views['indices'].allocation_id,(rows,),'int32',4,
                scope.scope_id+'.argmax_selected_partial_index','indices',
                (LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),)),),(1,)))
        extra=(selected,)
        accesses=(
            TileAccessTemplate('values','input',Operation.READ,('row','split'),('row','split'),(ready['input'],),(),evidence),
            TileAccessTemplate('index','selected_indices',Operation.READ,('row',),('row',),(ready['indices'],),
                (dep('values',('split',)),),evidence,unmapped=splits!=1,
                value_index=None if splits==1 else TileValueIndex('values','argmax_index')),
            TileAccessTemplate('positions','positions',Operation.READ,('row',),('row',),(ready['positions'],),(),evidence,unmapped=positions_unmapped),
            TileAccessTemplate('output','output',Operation.WRITE,('row',),('row',None),(),(dep('index'),),evidence),
            TileAccessTemplate('probability','probability',Operation.WRITE,('row',),('row',None),(),(),evidence),
            TileAccessTemplate('position_update','positions',Operation.WRITE,('row',),('row',),(),(dep('positions'),),evidence,unmapped=positions_unmapped))
        if declaration['draft_columns']:
            accesses=(*accesses,TileAccessTemplate('draft_tokens','draft_tokens',Operation.WRITE,('row',),('row',),(),(dep('index'),),evidence))
    return TileProgram(scope.scope_id,scope,axes,accesses,visibility,evidence),extra


def _native_header(kernel):
    declaration,partial,rows,splits,logical,count=_native_geometry(kernel)
    return (declaration['name'],str(rows),str(splits),str(logical),str(count),
        str(declaration['vocab']) if partial else str(declaration['draft_columns']),
        str(_LEGACY_SGLANG_TOPK1_BLOCK) if partial else str(declaration['draft_column']))


def _native_source(kernel,hardware,bindings,*,program_id):
    inputs,outputs=_native_layouts(kernel)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(tuple(inputs),tuple(outputs),len(inputs)):
        raise ValueError('Native topk1 actual input/output roles differ')
    scope=AccessScope(program_id,_native_header(kernel),bindings.call_id,(),bindings.rank,bindings.stream,'logical-sglang-topk1')
    family,extra=_native_tiles(kernel,unique_by(bindings.views,'view_id'),dict(zip(inputs,bindings.input_ready,strict=True)),scope,bindings.visibility_domain)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Exact public topk1 split/finalize storage and row-local dependencies; selected partial index remains symbolic.',
            'Legacy reduction_factor price remains an explicit projection, separate from native reads/writes and 8192-tail geometry.')),
        bindings.allocations,(*bindings.views,*extra),(),bindings.events,(),(family,))
    _native_source_work(program,kernel,bindings);return program


def _native_source_work(program,kernel,bindings):
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__();inputs,outputs=_native_layouts(kernel)
    if (program.hardware_id!=kernel.hardware_id or program.scopes or program.accesses or len(program.tile_programs)!=1
            or program.coverage.granularity is not Granularity.TILE or program.coverage.semantics is not Semantics.LOGICAL
            or program.coverage.ordering is not Ordering.PARTIAL or program.coverage.executable):
        raise ValueError('Native topk1 source differs from actual kernel')
    family,=program.tile_programs;views=unique_by(program.views,'view_id');public={**inputs,**outputs}
    declaration,partial,rows,splits,logical,count=_native_geometry(kernel)
    if family.scope.op_ids!=_native_header(kernel) or family.scope.agent!='logical-sglang-topk1':
        raise ValueError('Native topk1 source declaration differs')
    for role,(shape,dtype) in public.items():
        v=views[role];width=DTYPE_BYTES[dtype]
        strides=_native_strides(shape,width)
        if role=='draft_tokens':strides=(declaration['draft_columns']*width,)
        if (not isinstance(v,TensorView) or (v.shape,v.dtype,v.element_bytes)!=(shape,dtype,width)
                or (v.strides_bytes!=strides and not (partial and role=='input'
                    and v.strides_bytes[-1]==width and v.strides_bytes[0]>=shape[1]*width))
                or role=='draft_tokens' and v.offset_bytes!=declaration['draft_column']*width):
            raise ValueError('Native topk1 public view differs from its actual storage ABI')
    if len({views[role].allocation_id for role in public})!=len(public):
        raise ValueError('Native topk1 distinct operands cannot silently alias')
    accesses=unique_by(family.accesses,'access_id');ready={}
    for role in inputs:
        name='values' if not partial and role=='input' else 'index' if not partial and role=='indices' else role
        if name not in accesses or len(accesses[name].input_ready)!=1:raise ValueError('Native topk1 input availability missing')
        ready[role]=accesses[name].input_ready[0]
    expected,extra=_native_tiles(kernel,views,ready,family.scope,family.visibility_domain)
    if (family!=expected or set(views)!={*public,*(v.view_id for v in extra)}
            or any(views[v.view_id]!=v for v in extra)):
        raise ValueError('Native topk1 lost its actual split/selection/write causality')
    if bindings is not None and (program.binding_id!=bindings.binding_id
            or bindings.read_views!=tuple(inputs) or bindings.write_views!=tuple(outputs)
            or (family.scope.call_id,family.scope.rank,family.scope.stream,family.visibility_domain)
                !=(bindings.call_id,bindings.rank,bindings.stream,bindings.visibility_domain)
            or {role:views[role] for role in public}!=unique_by(bindings.views,'view_id')
            or tuple(ready.values())!=bindings.input_ready
            or program.allocations!=bindings.allocations or program.events!=bindings.events):
        raise ValueError('Native topk1 differs from actual caller bindings')
    factor=kernel.attributes['reduction_factor'];width=kernel.shape['dtype_bytes']
    legacy_total=logical*factor*width;legacy_output=logical*width
    per_output=splits
    if partial:
        per_output=(declaration['vocab'] if splits==1 else
            _LEGACY_SGLANG_TOPK1_BLOCK if declaration['vocab']%_LEGACY_SGLANG_TOPK1_BLOCK==0 else None)
    return {'input_elements':logical*factor,'output_elements':logical,'input_elements_per_output':per_output,
        'legacy_input_elements_per_output':factor,
        'execution_input_elements':rows*declaration['vocab'] if partial else rows*splits,
        'execution_output_elements':rows*splits if partial else rows,
        'native_output_elements':declaration['rows']*splits if partial else declaration['rows'],
        'logical_read_bytes':logical_requested_bytes(program,Operation.READ),
        'logical_write_bytes':logical_requested_bytes(program,Operation.WRITE),'output_encoding_known':True,
        'legacy_read_bytes':legacy_total-legacy_output,'legacy_write_bytes':legacy_output,'legacy_total_bytes':legacy_total}
