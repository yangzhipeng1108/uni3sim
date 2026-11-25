"""mHC logical row/hidden/projection tiles and actual shared intermediates."""
from math import gcd,prod

from src.kernel.kernel_spec import KernelKind,DTYPE_BYTES
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,MemoryAccessProgram,
    MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,TensorView,
    TileAxis,TileAccessTemplate,TileDependency,TileIndexBinding,TileProgram,TilePartition,unique_by,
)


MHC_SOURCE_VARIANTS=('mhc_post','hc_head','mhc_post_prenorm','mhc_pre_norm','mhc_pre_norm_broadcast')


def has_mhc_source(kernel,hardware):
    return not (kernel.kind is not KernelKind.VECTOR or kernel.variant not in MHC_SOURCE_VARIANTS
            or kernel.execution_support!='supported' or hardware.kernel_memory_model!='gpu_mshr')


def has_native_post_prenorm_geometry(kernel):
    """Native shared state needs complete source-bound 128-thread slices."""
    shape=kernel.shape;mix=shape['hc_mult']*(shape['hc_mult']+2)
    return (kernel.launch.block==(128,1,1) and shape['hidden']%(shape['n_splits']*128)==0
        and mix%shape['tile_n']==0)


def _row_extents(kernel):
    h,m=(kernel.shape[name] for name in ('hidden','hc_mult'))
    logical=kernel.shape['element_count']//(h*m)
    if 'model_input_shape' not in kernel.attributes:return logical,logical
    physical=kernel.attributes['model_input_shape']
    fields=set(kernel.shape)-{'dtype_bytes','elements'}
    if (set(physical)-{'dtype_bytes','elements'}!=fields
            or any(physical[name]!=kernel.shape[name] for name in fields-{'element_count'})
            or ('elements' in physical and physical['elements']!=physical['element_count'])
            or ('dtype_bytes' in physical and physical['dtype_bytes']!=kernel.shape['dtype_bytes'])
            or physical['element_count']%(h*m)):
        raise ValueError('mHC replay requires unchanged non-row geometry and complete rows')
    execution=physical['element_count']//(h*m)
    if (execution<logical or (execution!=logical and (
            kernel.attributes['graph_runtime_mode'] not in ('FULL','PIECEWISE')
            or execution*kernel.attributes['graph_logical_tokens']
                !=logical*kernel.attributes['graph_model_input_tokens']))):
        raise ValueError('mHC replay rows differ from their declared graph token mapping')
    return logical,execution


def _geometry(kernel):
    h,m=(kernel.shape[name] for name in ('hidden','hc_mult'))
    _,execution=_row_extents(kernel)
    return execution,h,m,m*(m+2)


def mhc_parameter_roles(variant):
    if variant=='mhc_post':return ()
    if variant=='hc_head':return ('fn','hc_scale','hc_base')
    if variant=='mhc_post_prenorm':return ('weight_t',)
    if variant in ('mhc_pre_norm','mhc_pre_norm_broadcast'):return ('hc_scale','hc_base','norm_weight')
    raise ValueError('Unknown mHC source variant')


def _layouts(kernel):
    t,h,m,mix=_geometry(kernel);variant=kernel.variant
    if variant in ('mhc_post','mhc_post_prenorm'):
        inputs={'comb_mix':((t,m,m),'fp32'),'residual':((t,m,h),'bf16'),
            'post_mix':((t,m),'fp32'),'x':((t,h),'bf16')}
        outputs={'residual_out':((t,m,h),'bf16')}
        if variant=='mhc_post_prenorm':
            s=kernel.shape['n_splits'];inputs['weight_t']=((mix,m,h),'fp32')
            outputs.update(projection_partial=((s,t,mix),'fp32'),sqrsum_partial=((s,t),'fp32'))
            if not has_native_post_prenorm_geometry(kernel):return inputs,outputs,{}
            n=kernel.shape['tile_n']
            return inputs,outputs,{'warp_projection':((t,s,mix//n,4,n),'fp32'),'warp_sqrsum':((t,s,4),'fp32'),
                'staged_post':((t,s,mix//n,m),'fp32'),'staged_comb':((t,s,mix//n,m,m),'fp32')}
        return inputs,outputs,{'staged_residual':((t,m,h),'bf16'),'staged_x':((t,h),'bf16')}
    if variant=='hc_head':
        return {'residual':((t,m,h),'bf16'),'fn':((m,m,h),'fp32'),
            'hc_scale':((1,),'fp32'),'hc_base':((m,),'fp32')},{'output':((t,h),'bf16')},{
                'gates':((t,m),'fp32'),'staged_residual':((t,m,h),'bf16')}
    if variant in ('mhc_pre_norm','mhc_pre_norm_broadcast'):
        s=kernel.shape['n_splits'];broadcast=variant=='mhc_pre_norm_broadcast'
        return {'projection_partial':((s,t,mix),'fp32'),'sqrsum_partial':((s,t),'fp32'),
            'hc_scale':((3,),'fp32'),'hc_base':((mix,),'fp32'),
            'residual':((t,h) if broadcast else (t,m,h),'bf16'),'norm_weight':((h,),'bf16')}, {
                'post_mix':((t,m),'fp32'),'comb_mix':((t,m,m),'fp32'),'layer_input':((t,h),'bf16'),
                **({'residual_out':((t,m,h),'bf16')} if broadcast else {})}, {
                    'mixes':((t,mix),'fp32'),'gates':((t,m),'fp32'),'weighted':((t,h),'bf16'),
                    'staged_residual':((t,h) if broadcast else (t,m,h),'bf16'),
                    'staged_norm_weight':((t,h),'bf16')}
    raise ValueError('Unknown mHC source tensor ABI')


def _strides(shape,width):
    return tuple(prod(shape[index+1:])*width for index in range(len(shape)))


def _private_fields(kernel,role,shape,dtype):
    if role in ('warp_projection','warp_sqrsum'):
        t,h,m,mix=_geometry(kernel);s,n=(kernel.shape[name] for name in ('n_splits','tile_n'))
        # Native 128 threads reduce four warps through one [4,tile_n+1]
        # shared parent per CTA. The final column is used only by tile zero.
        whole=(t,s,mix//n,4,n+1);strides=_strides(whole,4)
        return 'warp',0 if role=='warp_projection' else n*4,(
            strides if role=='warp_projection' else (strides[0],strides[1],strides[3])),prod(whole)*4
    width=DTYPE_BYTES[dtype]
    return role,0,_strides(shape,width),prod(shape)*width


def standalone_mhc_bindings(kernel,*,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    inputs,outputs,_=_layouts(kernel);allocations=[];views=[];events=[];ready=[]
    for role,(shape,dtype) in {**inputs,**outputs}.items():
        width=DTYPE_BYTES[dtype];identity=f'{binding_id}.{role}'
        allocation=MemoryAllocation(identity,'logical-local-device','mhc-global',prod(shape)*width,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        view=TensorView(role,identity,0,shape,_strides(shape,width),dtype,width)
        allocations.append(allocation);views.append(view);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role in inputs:
            event=ProgramEvent(identity+'.ready',EventKind.INITIALIZED,identity,view.region(),'mhc-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),tuple(inputs),tuple(outputs),
        tuple(ready),0,'offline','mhc-visible')


def _header(kernel):
    logical,execution=_row_extents(kernel)
    base=('mhc',kernel.variant,*(f'{name}={value}' for name,value in sorted(kernel.shape.items())))
    return base if logical==execution else (*base,f'execution_rows={execution}')


def _tiles(kernel,views,ready,scope,visibility):
    if kernel.variant=='mhc_post_prenorm' and not has_native_post_prenorm_geometry(kernel):
        return _abstract_post_prenorm_tiles(kernel,ready,scope,visibility)
    t,h,m,mix=_geometry(kernel);variant=kernel.variant;block=gcd(h,1024)
    axes=[TileAxis('token',t,1),TileAxis('hidden',h,block)];accesses=[]
    evidence=('vLLM 0.30 model_executor/kernels/mhc/tilelang_kernels.py public tensor ABI and source passes.',
        'mhc_post uses gcd(H,1024) hidden chunks. hc_head rereads residual after complete row projections; pre-norm retains its BF16 shared weighted row.',
        'Fused post/pre repeats its post mapping per projection tile and split, and only tile zero emits residual and squared-sum partials.',
        'Logical work blocks and actual shared intermediates; no compiler issue/physical transactions or cache residency inferred.',
        'Shared staging is represented as logical per-iteration snapshots: reused native tile buffers are renamed by hidden work coordinate. Snapshot capacity is not native live shared capacity; compiler pipeline bank counts remain unbound.')
    def dep(name,reduction=(),bindings=()):return TileDependency(name,EventKind.READ_READY,reduction,bindings)
    def visible(name,reduction=()):return TileDependency(name,EventKind.WRITE_VISIBLE,reduction)
    def read(name,role,loops,mapping,deps=(),divisors=()):
        accesses.append(TileAccessTemplate(name,role,Operation.READ,loops,mapping,(ready[role],) if role in ready else (),deps,evidence,
            view_axis_divisors=divisors))
    def write(name,role,loops,mapping,deps,divisors=()):
        accesses.append(TileAccessTemplate(name,role,Operation.WRITE,loops,mapping,(),deps,evidence,view_axis_divisors=divisors))
    if variant=='mhc_post':
        read('comb','comb_mix',('token',),('token',None,None));read('post','post_mix',('token',),('token',None))
        read('residual','residual',('token','hidden'),('token',None,'hidden'))
        read('x','x',('token','hidden'),('token','hidden'))
        for role,mapping in (('residual',('token',None,'hidden')),('x',('token','hidden'))):
            staged='staged_'+role
            write(staged,staged,('token','hidden'),mapping,(dep(role),))
            read(staged+'_read',staged,('token','hidden'),mapping,(visible(staged),))
        write('residual_out','residual_out',('token','hidden'),('token',None,'hidden'),
            tuple(dep(n) for n in ('comb','post','staged_residual_read','staged_x_read')))
    elif variant=='hc_head':
        read('project_residual','residual',('token','hidden'),('token',None,'hidden'))
        read('fn','fn',('token','hidden'),(None,None,'hidden'))
        read('scale','hc_scale',('token',),(None,));read('base','hc_base',('token',),(None,))
        write('gates','gates',('token',),('token',None),(dep('project_residual',('hidden',)),dep('fn',('hidden',)),dep('scale'),dep('base')))
        read('gate_read','gates',('token','hidden'),('token',None),(visible('gates'),))
        read('apply_residual','residual',('token','hidden'),('token',None,'hidden'),(visible('gates'),))
        write('staged_residual','staged_residual',('token','hidden'),('token',None,'hidden'),(dep('apply_residual'),))
        read('staged_residual_read','staged_residual',('token','hidden'),('token',None,'hidden'),(visible('staged_residual'),))
        write('output','output',('token','hidden'),('token','hidden'),(dep('gate_read'),dep('staged_residual_read')))
    elif variant in ('mhc_pre_norm','mhc_pre_norm_broadcast'):
        s=kernel.shape['n_splits'];axes.append(TileAxis('split',s,1));broadcast=variant=='mhc_pre_norm_broadcast'
        read('partials','projection_partial',('token','split'),('split','token',None))
        read('sqrsum','sqrsum_partial',('token','split'),('split','token'))
        read('scale','hc_scale',('token',),(None,));read('base','hc_base',('token',),(None,))
        write('mixes','mixes',('token',),('token',None),(dep('partials',('split',)),dep('sqrsum',('split',))))
        read('mix_read','mixes',('token',),('token',None),(visible('mixes'),))
        mix_deps=(dep('mix_read'),dep('scale'),dep('base'))
        for role in ('post_mix','comb_mix'):
            write(role,role,('token',),('token',None) if role=='post_mix' else ('token',None,None),mix_deps)
        write('gates','gates',('token',),('token',None),mix_deps)
        read('gate_read','gates',('token','hidden'),('token',None),(visible('gates'),))
        read('residual','residual',('token','hidden'),('token','hidden') if broadcast else ('token',None,'hidden'),(visible('gates'),))
        mapping=('token','hidden') if broadcast else ('token',None,'hidden')
        write('staged_residual','staged_residual',('token','hidden'),mapping,(dep('residual'),))
        read('staged_residual_read','staged_residual',('token','hidden'),mapping,(visible('staged_residual'),))
        write('weighted','weighted',('token','hidden'),('token','hidden'),(dep('staged_residual_read'),dep('gate_read')))
        if broadcast:
            write('residual_out','residual_out',('token','hidden'),('token',None,'hidden'),(dep('staged_residual_read'),))
        read('weighted_read','weighted',('token','hidden'),('token','hidden'),(visible('weighted',('hidden',)),))
        read('norm_weight','norm_weight',('token','hidden'),('hidden',))
        write('staged_norm_weight','staged_norm_weight',('token','hidden'),('token','hidden'),(dep('norm_weight'),))
        read('staged_norm_weight_read','staged_norm_weight',('token','hidden'),('token','hidden'),(visible('staged_norm_weight'),))
        write('layer_input','layer_input',('token','hidden'),('token','hidden'),(dep('weighted_read'),dep('staged_norm_weight_read')))
    elif variant=='mhc_post_prenorm':
        s,n=(kernel.shape[name] for name in ('n_splits','tile_n'));part=h//s
        axes=[TileAxis('token',t,1),TileAxis('hidden',h,part),TileAxis('projection',mix,n)]
        loops=('token','hidden','projection')
        read('comb','comb_mix',loops,('token',None,None));read('post','post_mix',loops,('token',None))
        for role,mapping in (('post',('token','hidden','projection',None)),('comb',('token','hidden','projection',None,None))):
            staged='staged_'+role;divisors=(1,part,n,*((1,)*(len(mapping)-3)))
            write(staged,staged,loops,mapping,(dep(role),),divisors)
            read(staged+'_read',staged,loops,mapping,(visible(staged),),divisors)
        read('residual','residual',loops,('token',None,'hidden'));read('x','x',loops,('token','hidden'))
        read('weight','weight_t',loops,('projection',None,'hidden'))
        inputs=('staged_comb_read','staged_post_read','residual','x')
        warp_axes=('token','hidden','projection',None,None);warp_divisors=(1,part,n,1,1)
        write('warp_projection','warp_projection',loops,warp_axes,tuple(dep(name) for name in (*inputs,'weight')),warp_divisors)
        read('warp_projection_read','warp_projection',loops,warp_axes,(visible('warp_projection'),),warp_divisors)
        write('projection_partial','projection_partial',loops,('hidden','token','projection'),
            (dep('warp_projection_read'),),(part,1,1))
        zero=(TileIndexBinding('projection',(),0),)
        deps=tuple(dep(name,bindings=zero) for name in inputs)
        write('residual_out','residual_out',('token','hidden'),('token',None,'hidden'),deps)
        write('warp_sqrsum','warp_sqrsum',('token','hidden'),('token','hidden',None),deps,(1,part,1))
        read('warp_sqrsum_read','warp_sqrsum',('token','hidden'),('token','hidden',None),(visible('warp_sqrsum'),),(1,part,1))
        write('sqrsum_partial','sqrsum_partial',('token','hidden'),('hidden','token'),(dep('warp_sqrsum_read'),),(part,1))
    else:raise ValueError('Unknown mHC tile stages')
    return TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),visibility,evidence)


def _abstract_post_prenorm_tiles(kernel,ready,scope,visibility):
    """Keep the retained mathematical contract without inventing native slices.

    A hidden coordinate contributes to one declared partial class. Membership
    is unresolved; in particular H/S is never rounded into an asserted slice.
    The projection vector is small public output geometry, not a native CTA.
    """
    t,h,_,_=_geometry(kernel);splits=kernel.shape['n_splits']
    evidence=('KernelSpec mhc_post_prenorm public post mapping, FP32 weight projection and squared-sum partials; existing Forward mathematical interface.',
        'Each hidden coordinate belongs to exactly one declared split class per token. Class membership, native tiling, shared scratch and issue order are unbound.',
        'The full projection vector is one mathematical vector tile per hidden coordinate; these operand references do not assert native loads or projection-tile reuse.')
    inputs=(('comb','comb_mix',('token',None,None)),('post','post_mix',('token',None)),
        ('residual','residual',('token',None,'hidden')),('x','x',('token','hidden')),
        ('weight','weight_t',(None,None,'hidden')))
    reads=tuple(TileAccessTemplate(name,role,Operation.READ,('token','hidden'),mapping,
        (ready[role],),(),evidence) for name,role,mapping in inputs)
    post=tuple(TileDependency(name,EventKind.READ_READY,()) for name,_,_ in inputs if name!='weight')
    partial=lambda names:tuple(TileDependency(name,EventKind.READ_READY,('hidden',),partition_axis='hidden') for name in names)
    accesses=(*reads,
        TileAccessTemplate('residual_out','residual_out',Operation.WRITE,('token','hidden'),('token',None,'hidden'),(),post,evidence),
        TileAccessTemplate('projection_partial','projection_partial',Operation.WRITE,('split','token'),('split','token',None),(),
            partial(tuple(name for name,_,_ in inputs)),evidence),
        TileAccessTemplate('sqrsum_partial','sqrsum_partial',Operation.WRITE,('split','token'),('split','token'),(),
            partial(tuple(name for name,_,_ in inputs if name!='weight')),evidence))
    return TileProgram(scope.scope_id,scope,(TileAxis('token',t,1),TileAxis('hidden',h,1),TileAxis('split',splits,1)),
        accesses,visibility,evidence,partitions=(TilePartition('hidden',None,class_axis_id='split',context_axes=('token',)),))


def build_mhc_source(kernel,hardware,bindings,granularity,*,program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_mhc_source(kernel,hardware) or not isinstance(bindings,ElementwiseBindings):raise TypeError('mHC requires its supported typed ABI')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown mHC source granularity')
    inputs,outputs,private=_layouts(kernel)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(tuple(inputs),tuple(outputs),len(inputs)):
        raise ValueError('mHC public tensor/readiness roles differ')
    allocations=list(bindings.allocations);views=list(bindings.views);events=list(bindings.events)
    device=allocations[0].device_id;created=set()
    for role,(shape,dtype) in private.items():
        parent,offset,strides,size=_private_fields(kernel,role,shape,dtype)
        width=DTYPE_BYTES[dtype];identity=f'{bindings.binding_id}.{program_id}.{parent}'
        if parent not in created:
            allocation=MemoryAllocation(identity,device,'mhc-shared',size,AddressLevel.OBJECT,None,None,identity+'.allocated',None)
            allocations.append(allocation);created.add(parent)
            events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        views.append(TensorView(role,identity,offset,shape,strides,dtype,width))
    scope=AccessScope(program_id,_header(kernel),bindings.call_id,(),bindings.rank,bindings.stream,'logical-mhc-source')
    family=_tiles(kernel,unique_by(tuple(views),'view_id'),dict(zip(inputs,bindings.input_ready,strict=True)),scope,bindings.visibility_domain)
    limitations=(
        'Source row/hidden/projection work and actual gates/weighted shared state; per-token shared objects are logical actor storage, not concurrent device capacity.',
        'Logical rereads, shared work and retained global unique-footprint pricing are separate.',
        'DeepGEMM prenorm, optional delayed/save-pre/aux-output branches, compiler spills and physical issue/cache behavior are not claimed.')
    if kernel.variant=='mhc_post_prenorm' and not has_native_post_prenorm_geometry(kernel):
        limitations=('Mathematical public tensor contract only; each hidden coordinate contributes to one partial class per token, with unresolved class membership.',
            'Operand references and public result bytes do not establish native load counts, layouts, compiler/private scratch or a runnable launch.',
            'Explicit native tile/shared geometry remains unbound; legacy unique-footprint pricing and diagnostics are preserved.')
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,limitations),
        tuple(allocations),tuple(views),(),tuple(events),(),(family,))
    mhc_source_work(program,expected_kernel=kernel,expected_bindings=bindings);return program


def mhc_source_work(program,*,expected_kernel,expected_bindings=None):
    from src.kernel.memory_path_solver import logical_view_requested_bytes,logical_view_unique_bytes
    program.__post_init__();kernel=expected_kernel
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant not in MHC_SOURCE_VARIANTS
            or kernel.hardware_id!=program.hardware_id or kernel.execution_support!='supported' or kernel.dtype!='bf16'):
        raise ValueError('mHC source differs from actual kernel')
    if program.accesses or program.scopes or len(program.tile_programs)!=1:raise ValueError('mHC requires its unique compressed tile source')
    family,=program.tile_programs;inputs,outputs,private=_layouts(kernel);views=unique_by(program.views,'view_id')
    if set(views)!={*inputs,*outputs,*private} or family.scope.op_ids!=_header(kernel) or family.scope.agent!='logical-mhc-source':
        raise ValueError('mHC source exact tensor/geometry roles differ')
    allocations=unique_by(program.allocations,'allocation_id');parents={}
    if expected_bindings is not None:
        public={role:views[role] for role in (*inputs,*outputs)}
        caller_allocations=unique_by(expected_bindings.allocations,'allocation_id')
        public_events=tuple(event for event in program.events if event.allocation_id in caller_allocations)
        if (program.binding_id!=expected_bindings.binding_id
                or (family.scope.call_id,family.scope.rank,family.scope.stream,family.visibility_domain)
                    !=(expected_bindings.call_id,expected_bindings.rank,expected_bindings.stream,expected_bindings.visibility_domain)
                or public!=unique_by(expected_bindings.views,'view_id')
                or any(allocations[name]!=value for name,value in caller_allocations.items())
                or public_events!=expected_bindings.events):
            raise ValueError('mHC source differs from actual caller bindings')
    for role,(shape,dtype) in {**inputs,**outputs,**private}.items():
        view=views[role];width=DTYPE_BYTES[dtype];parent,offset,strides,size=(
            _private_fields(kernel,role,shape,dtype) if role in private else (role,0,_strides(shape,width),prod(shape)*width))
        if (not isinstance(view,TensorView) or (view.shape,view.strides_bytes,view.dtype,view.element_bytes,view.offset_bytes)
                !=(shape,strides,dtype,width,offset)):
            raise ValueError('mHC typed tensor ABI differs')
        if role in private and (allocations[view.allocation_id].memory_domain,allocations[view.allocation_id].size_bytes)!=('mhc-shared',size):
            raise ValueError('mHC temporary must retain actual shared storage')
        if parent in parents and parents[parent]!=view.allocation_id:raise ValueError('mHC shared fields lost their actual parent')
        parents[parent]=view.allocation_id
    if len(set(parents.values()))!=len(parents):raise ValueError('mHC unrelated public/shared objects must not alias')
    accesses=unique_by(family.accesses,'access_id');ready={}
    for role in inputs:
        candidates=[a for a in family.accesses if a.view_id==role and a.operation is Operation.READ]
        if not candidates or any(len(a.input_ready)!=1 for a in candidates):raise ValueError('mHC actual public input reads missing')
        refs={a.input_ready[0] for a in candidates}
        if len(refs)!=1:raise ValueError('mHC repeated reads changed public input availability')
        ready[role]=next(iter(refs))
    if expected_bindings is not None and tuple(ready[role] for role in inputs)!=expected_bindings.input_ready:
        raise ValueError('mHC source readiness differs from actual caller bindings')
    if family!=_tiles(kernel,views,ready,family.scope,family.visibility_domain):raise ValueError('mHC source lost actual per-row/split/pass dependencies')
    logical,execution=_row_extents(kernel);parameters=mhc_parameter_roles(kernel.variant)
    # The actual source covers captured rows. Keep the existing logical-row
    # price using its uniform per-token public regions; parameters are not
    # token-scaled. No padding value or cache behavior is assumed.
    read=sum(logical_view_unique_bytes(program,role,Operation.READ)*(execution if role in parameters else logical)//execution
        for role in inputs)
    write=sum(logical_view_unique_bytes(program,role,Operation.WRITE)*logical//execution for role in outputs)
    return {'read_bytes':read,'write_bytes':write,
        'logical_rows':logical,'execution_rows':execution,
        'logical_global_read_bytes':sum(logical_view_requested_bytes(program,role,Operation.READ) for role in inputs),
        'logical_global_write_bytes':sum(logical_view_requested_bytes(program,role,Operation.WRITE) for role in outputs),
        'logical_shared_read_bytes':sum(logical_view_requested_bytes(program,role,Operation.READ) for role in private),
        'logical_shared_write_bytes':sum(logical_view_requested_bytes(program,role,Operation.WRITE) for role in private)}
