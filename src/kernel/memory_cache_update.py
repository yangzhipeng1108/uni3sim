"""Untimed paged KV copy source with honestly unresolved slot/page addresses."""
from dataclasses import dataclass, replace

from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity, Indirect, Indirection,
    MemoryAccess, MemoryAccessProgram, MemoryAllocation, Operation, Ordering, ProgramEvent, KVComponent,
    Semantics, Span, TensorView, Typed, unique_by,
    TileAxis,TileDependency,TileAccessTemplate,TileProgram,TileIndirectTarget,LogicalSelectionView,Unmapped,ConstantValue,
    TileIndexBinding,TileIndexTerm,ReadValuePredicate,
)
from src.kernel.kernel_spec import KernelKind


@dataclass(frozen=True)
class CacheUpdateBindings(Typed):
    binding_id: str
    call_id: str
    allocations: tuple[MemoryAllocation, ...]
    views: tuple[TensorView, ...]
    events: tuple[ProgramEvent, ...]
    key_view: str
    value_view: str
    slots_view: str
    key_cache: str
    value_cache: str
    key_ready: EventRef
    value_ready: EventRef
    slots_ready: EventRef
    rank: int
    stream: str
    visibility_domain: str


def has_paged_kv_source(kernel, hardware):
    return (kernel.kind is KernelKind.KV_CACHE and kernel.variant == "paged_kv_cache"
        and hardware.kernel_memory_model in ("gpu_mshr", "npu_tile")
        and kernel.shape["seq_len"] > 0)


def paged_kv_source_index_type(kernel):
    """Slot ABI proven by the selected cache program, without loading HWSpec.

    Unrecognized or launchless programs remain partial. This metadata-only
    entry is for caller binding admission, not a new hardware selector.
    """
    if (kernel.kind is not KernelKind.KV_CACHE or kernel.variant != "paged_kv_cache"
            or kernel.execution_support != "supported" or kernel.launch is None):
        return None
    if "cann_tiling_key" in kernel.launch.parameters:
        slot_read_passes(kernel)  # Validate the supported ReshapeAndCache key.
        return "int32", 4
    if kernel.launch.algorithm in ("kv_cache", "kvcache", "paged_kv_cache"):
        return "int64", 8
    if (kernel.hardware_id == "nvidia_h200" and kernel.launch.algorithm in (
            "vllm_0_27_sm90_source_kv_cache", "vllm_0_27_sm90_source_kvcache",
            "vllm_0_30_sm90_source_kv_cache", "vllm_0_30_sm90_source_kvcache")):
        return "int64", 8
    return None


def has_tensor_only_kv_source(kernel, hardware):
    """The PIM compatibility model declares payload copies, without a slot ABI."""
    return (kernel.kind is KernelKind.KV_CACHE and kernel.variant == 'paged_kv_cache'
        and kernel.execution_support == 'supported' and hardware.kernel_memory_model == 'pim_bank'
        and kernel.shape['seq_len'] > 0)


def standalone_tensor_kv_bindings(kernel, *, call_id, binding_id):
    from math import prod
    from src.kernel.memory_access_builder import ElementwiseBindings
    shape=tuple(kernel.shape[name] for name in ('seq_len','num_kv_heads','head_dim'))
    width=kernel.shape['dtype_bytes'];allocations=[];views=[];events=[];ready=[]
    for role in ('key','value','key_cache','value_cache'):
        output=role.endswith('_cache');identity=f'{binding_id}.{role}'
        allocation=MemoryAllocation(identity,'logical-local-device','logical-cache-selection' if output else 'logical-storage',
            None if output else prod(shape)*width,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        views.append(LogicalSelectionView(role,identity,shape,kernel.dtype,width,f'tensor_kv.{role}',None) if output else
            TensorView(role,identity,0,shape,(shape[1]*shape[2]*width,shape[2]*width,width),kernel.dtype,width))
        if not output:
            event=ProgramEvent(f'{identity}.initialized',EventKind.INITIALIZED,identity,views[-1].region(),'tensor-kv-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        ('key','value'),('key_cache','value_cache'),tuple(ready),0,'offline','tensor-kv-visible')


def _tensor_kv_tiles(views,ready,scope,visibility,evidence):
    templates=[]
    for role in ('key','value'):
        templates.extend((
            TileAccessTemplate(f'read_{role}',role,Operation.READ,('token',),('token',None,None),
                (ready[role],),(),evidence),
            TileAccessTemplate(f'write_{role}',f'{role}_cache',Operation.WRITE,('token',),('token',None,None),(),
                (TileDependency(f'read_{role}',EventKind.READ_READY,()),),evidence,unmapped=True)))
    return TileProgram(scope.scope_id,scope,(TileAxis('token',views['key'].shape[0],1),),tuple(templates),visibility,evidence)


def build_tensor_kv_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_tensor_only_kv_source(kernel,hardware):raise ValueError('Tensor-only KV requires the PIM payload ABI')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('Tensor-only KV requires typed payload bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown KV granularity')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(('key','value'),('key_cache','value_cache'),2):
        raise ValueError('Tensor-only KV must retain both copy operands and outputs')
    views=unique_by(bindings.views,'view_id')
    if set(views)!={'key','value','key_cache','value_cache'}:
        raise ValueError('Tensor-only KV requires exactly its two payloads and cache selections')
    scope=AccessScope(program_id,('tensor_only_kv',kernel.variant),bindings.call_id,(),bindings.rank,bindings.stream,'logical-tensor-kv-copy')
    evidence=('PIM paged_kv_cache compatibility ABI copies K and V payload once; it has no serving slot index ABI.',
        'Independent token copies retain own K/V READ_READY; cache selections do not imply addresses, capacity or native bank placement.')
    family=_tensor_kv_tiles(views,dict(zip(bindings.read_views,bindings.input_ready,strict=True)),scope,bindings.visibility_domain,evidence)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Tensor-only compatibility ABI; no observed-zero slot traffic or native PIM Serving claim.',
            'Cache selection order, page/slot mapping, physical capacity and bank addresses are unknown.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    tensor_kv_source_work(program,expected_kernel=kernel)
    return program


def tensor_kv_source_work(program, *, expected_kernel=None):
    from math import prod
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:
        raise ValueError('Tensor-only KV requires its unique compressed copy source')
    family,=program.tile_programs;views=unique_by(program.views,'view_id')
    if (family.scope.op_ids!=('tensor_only_kv','paged_kv_cache') or family.scope.agent!='logical-tensor-kv-copy'
            or set(views)!={'key','value','key_cache','value_cache'}):
        raise ValueError('Tensor-only KV requires exactly its two payloads and cache selections')
    key=views['key'];shape=key.shape;width=key.element_bytes
    if len(shape)!=3 or len({view.allocation_id for view in views.values()})!=4:
        raise ValueError('Tensor-only KV cache aliases require a declared mapping')
    for role in ('key','value','key_cache','value_cache'):
        view=views[role]
        if (view.shape,view.dtype,view.element_bytes)!=(shape,key.dtype,width):raise ValueError('Tensor-only KV typed copy shape mismatch')
        if role.endswith('_cache'):
            if (not isinstance(view,LogicalSelectionView) or view.selection_id!=f'tensor_kv.{role}'
                    or view.parent_view_id is not None or view.coordinate_map):
                raise ValueError('Tensor-only KV cache mapping must remain explicitly unresolved')
        elif not isinstance(view,TensorView):raise ValueError('Tensor-only KV input requires a tensor view')
    if expected_kernel is not None:
        expected_shape=tuple(expected_kernel.shape[name] for name in ('seq_len','num_kv_heads','head_dim'))
        if (expected_kernel.kind is not KernelKind.KV_CACHE or expected_kernel.variant!='paged_kv_cache'
                or program.hardware_id!=expected_kernel.hardware_id or shape!=expected_shape
                or key.dtype!=expected_kernel.dtype or width!=expected_kernel.shape['dtype_bytes']
                or prod(shape)!=expected_kernel.shape['element_count']):
            raise ValueError('Tensor-only KV source differs from its actual kernel')
    templates=unique_by(family.accesses,'access_id');ready={}
    for role in ('key','value'):
        if f'read_{role}' not in templates or len(templates[f'read_{role}'].input_ready)!=1:
            raise ValueError('Tensor-only KV requires both initialized input reads')
        ready[role]=templates[f'read_{role}'].input_ready[0]
    if family!=_tensor_kv_tiles(views,ready,family.scope,family.visibility_domain,family.evidence_refs):
        raise ValueError('Tensor-only KV copy requires its own complete payload READ_READY')
    read=logical_requested_bytes(program,Operation.READ);write=logical_requested_bytes(program,Operation.WRITE)
    if read!=write or read!=2*prod(shape)*width:raise ValueError('Tensor-only KV copy requests changed')
    return {'active_elements':2*prod(shape),'tensor_read_bytes':read,'tensor_write_bytes':write,'slot_read_bytes':0}


def slot_read_passes(kernel):
    from src.kernel.predictors.vector import _CANN_RESHAPE_CACHE_EQUAL_KEY, _CANN_RESHAPE_CACHE_FALLBACK_KEY
    if "cann_tiling_key" not in kernel.launch.parameters:
        return 1
    key = int(kernel.launch.parameters["cann_tiling_key"])
    if key == _CANN_RESHAPE_CACHE_EQUAL_KEY:
        return 1
    if key == _CANN_RESHAPE_CACHE_FALLBACK_KEY:
        return 2
    raise ValueError("paged KV source requires the declared ReshapeAndCache key")


def standalone_cache_update_bindings(kernel, hardware, *, call_id, binding_id,
                                     device_id='logical-local-device', rank=0, stream='offline',
                                     cache_parents=None, scoped=False):
    from src.kernel.predictors.vector import _serving_vector_index_bytes
    if type(scoped) is not bool:
        raise TypeError('KV source scope request must be a boolean')
    if cache_parents is not None:
        if type(cache_parents) is not tuple or len(cache_parents) != 2:
            raise TypeError('KV source requires its two actual cache parents')
        for parent in cache_parents:
            if not isinstance(parent, MemoryAllocation):
                raise TypeError('KV source cache parent requires a typed allocation')
            parent.__post_init__()
            if parent.device_id != device_id or parent.size_bytes is not None:
                raise ValueError('KV source cannot change cache device or infer pool capacity')
    width = _serving_vector_index_bytes(hardware)
    if width not in (4, 8):
        raise NotImplementedError("paged KV source needs a declared serving slot index ABI")
    shape = (kernel.shape["seq_len"], kernel.shape["num_kv_heads"], kernel.shape["head_dim"])
    tensors = (("key", shape, kernel.dtype, kernel.shape["dtype_bytes"]),
               ("value", shape, kernel.dtype, kernel.shape["dtype_bytes"]),
               ("slots", (shape[0],), "int32" if width == 4 else "int64", width))
    allocations, views, events = [], [], []
    for name, extent, dtype, element_width in tensors:
        allocation_id = f"{binding_id}.{name}"
        strides, size = [], element_width
        for dimension in reversed(extent):
            strides.append(size)
            size *= dimension
        allocation = MemoryAllocation(allocation_id, device_id, "logical-storage", size,
            AddressLevel.OBJECT, None, None, f"{allocation_id}.allocated", None)
        allocations.append(allocation)
        views.append(TensorView(name, allocation_id, 0, extent, tuple(reversed(strides)), dtype, element_width))
        events.extend((ProgramEvent(allocation.allocated_event, EventKind.ALLOCATED, allocation_id, None, None, ()),
            ProgramEvent(f"{binding_id + '.' if scoped else ''}{allocation_id}.initialized", EventKind.INITIALIZED, allocation_id, Span(0, size),
                "kv-input-visible", (EventRef(allocation.allocated_event, EventKind.ALLOCATED),))))
    for index, name in enumerate(("key_cache", "value_cache")):
        allocation_id = f"{binding_id}.{name}"
        parent = (MemoryAllocation(allocation_id, device_id, "paged-kv-cache", None,
            AddressLevel.OBJECT, None, None, f"{allocation_id}.allocated", None)
            if cache_parents is None else cache_parents[index])
        allocations.append(parent)
        allocation_id = parent.allocation_id
        events.append(ProgramEvent(allocations[-1].allocated_event, EventKind.ALLOCATED, allocation_id, None, None, ()))
    return CacheUpdateBindings(binding_id, call_id, tuple(allocations), tuple(views), tuple(events),
        "key", "value", "slots", allocations[-2].allocation_id, allocations[-1].allocation_id,
        *(EventRef(f"{binding_id + '.' if scoped else ''}{binding_id}.{name}.initialized", EventKind.INITIALIZED) for name in ("key", "value", "slots")),
        rank, stream, "kv-input-visible")


def build_paged_kv_update(kernel, hardware, bindings, granularity, *, program_id):
    if not isinstance(bindings, CacheUpdateBindings):
        raise TypeError("paged KV update requires typed cache/source/slot bindings")
    if granularity not in (Granularity.OP, Granularity.KERNEL,Granularity.TILE):
        raise ValueError("unknown logical KV granularity")
    views = unique_by(bindings.views, "view_id")
    allocations = unique_by(bindings.allocations, "allocation_id")
    shape = (kernel.shape["seq_len"], kernel.shape["num_kv_heads"], kernel.shape["head_dim"])
    key, value, slots = (views[name] for name in (bindings.key_view, bindings.value_view, bindings.slots_view))
    for view in (key, value):
        if (view.shape, view.dtype, view.element_bytes) != (shape, kernel.dtype, kernel.shape["dtype_bytes"]):
            raise ValueError("paged KV source tensor shape/dtype mismatch")
    from src.kernel.predictors.vector import _serving_vector_index_bytes
    width = _serving_vector_index_bytes(hardware)
    if (slots.shape != (shape[0],) or slots.element_bytes != width
            or slots.dtype != ("int32" if width == 4 else "int64")):
        raise ValueError("paged KV slot view disagrees with the serving index ABI")
    if any(
            cache in (key.allocation_id, value.allocation_id, slots.allocation_id)
            for cache in (bindings.key_cache, bindings.value_cache)):
        raise NotImplementedError("paged KV cache/source alias requires an explicit in-place algorithm")
    if kernel.attributes["cache_head_layout"] == "hnd":
        mapping = Indirection.PAGED_KV_HND
    elif kernel.attributes["cache_head_layout"] == "nhd":
        mapping = Indirection.PAGED_KV_NHD
    else:
        raise ValueError("unknown paged KV head layout")
    passes=slot_read_passes(kernel)
    scope = AccessScope(f"{program_id}.scope", ("kv_cache", "paged_update",mapping.value,
        *(str(value) for value in shape),kernel.dtype,str(kernel.shape['dtype_bytes']),str(width),str(passes)), bindings.call_id, (),
        bindings.rank, bindings.stream, "logical-token-slot-tile")
    evidence=("reshape_and_cache semantic tile: one token row and its actual integer slot selection; not a native CTA/issue claim.",
        "Every token tile carries both K/V payload reads, required slot-read passes and indirect stores.")
    templates=[]
    for index in range(passes):
        templates.append(TileAccessTemplate(f'slots{index}',slots.view_id,Operation.READ,('token',),('token',),
            (bindings.slots_ready,),(),evidence))
    for index,(role,view,ready,cache) in enumerate((('key',key,bindings.key_ready,bindings.key_cache),
            ('value',value,bindings.value_ready,bindings.value_cache))):
        read_id=f'read_{role}'
        templates.append(TileAccessTemplate(read_id,view.view_id,Operation.READ,('token',),('token',None,None),
            (ready,),(),evidence))
        slot=f'slots{index if passes == 2 else 0}'
        target=TileIndirectTarget(cache,slot,mapping,KVComponent.KEY if role == 'key' else KVComponent.VALUE)
        templates.append(TileAccessTemplate(f'write_{role}',view.view_id,Operation.WRITE,('token',),('token',None,None),(),
            (TileDependency(read_id,EventKind.READ_READY,()),TileDependency(slot,EventKind.READ_READY,())),evidence,
            indirect_target=target))
    family=TileProgram(program_id,scope,(TileAxis('token',shape[0],1),),tuple(templates),bindings.visibility_domain,evidence)
    program = MemoryAccessProgram("midlayer.memory-access.v1", program_id, hardware.hardware_id, bindings.binding_id,
        Coverage(Granularity.TILE, Semantics.LOGICAL, Ordering.PARTIAL, False,
            ("Actual slot values, block size, page mapping, physical cache capacity and alias footprint are unavailable.",
             "Each token/slot tile has logical K/V/index selections and dependencies; missing physical mapping does not prevent logical tiles.",
             "Logical payload and index multiplicity are separate from original issued/cache demand projections.")),
        bindings.allocations, bindings.views, (), bindings.events, (), (family,))
    from src.kernel.memory_path_solver import logical_paged_kv_update_bytes
    logical_paged_kv_update_bytes(program)
    return program


def has_target_cache_source(kernel):
    return (kernel.kind is KernelKind.KV_CACHE and kernel.execution_support == 'supported'
        and (kernel.variant in ('mla_kv_cache','dsa_fp8_index_cache','dsv4_compressor_state_cache','dsv4_qnorm_rope_kv_insert')
            or kernel.variant == 'dsv4_compressed_state_cache'))


def _target_cache_header(kernel):
    variant = kernel.variant;tokens = kernel.shape['seq_len']
    if variant == 'mla_kv_cache':
        return ('target_cache_tiles',variant,str(tokens),str(kernel.shape['compressed_dim']),'0',kernel.dtype,str(kernel.shape['dtype_bytes']))
    if variant == 'dsa_fp8_index_cache':
        return ('target_cache_tiles',variant,str(tokens),str(kernel.shape['key_dim']),str(kernel.shape['scale_bytes_per_token']),'bf16','2')
    if variant == 'dsv4_compressor_state_cache':
        return ('target_cache_tiles',variant,str(tokens),str(kernel.shape['state_bytes']//4),'0','fp32','4')
    if variant == 'dsv4_qnorm_rope_kv_insert':
        return ('target_cache_tiles',variant,str(tokens),'512','8','bf16','2',
            str(kernel.shape['num_query_heads']),str(kernel.shape['padded_query_heads']))
    raise ValueError('Unknown target cache input ABI')


def _target_cache_layouts(header):
    if len(header) == 8 and header[-1] == 'mla_split_512_64':
        if header[1] != 'mla_kv_cache' or header[3:7] != ('576', '0', 'bf16', '2'):
            raise ValueError('Split MLA requires its BF16 latent512 plus rope64 interface')
        tokens = int(header[2])
        return (('latent', (tokens, 1, 512), 'bf16', 2, False),
                ('rope', (tokens, 1, 64), 'bf16', 2, False),
                ('slots', (tokens,), 'int64', 8, False)), (
                ('cache_payload', (tokens, 1, 576), 'bf16', 2, True),)
    if len(header) not in (7,9) or header[0] != 'target_cache_tiles':raise ValueError('Target cache tile ABI missing')
    variant = header[1];tokens,dimension,scale_bytes = map(int,header[2:5]);dtype,width = header[5],int(header[6])
    slots = ('slots',(tokens,),'int64',8,False)
    if min(tokens,dimension,width) <= 0:raise ValueError('Target cache tile extents must be positive')
    if variant == 'dsv4_qnorm_rope_kv_insert':
        if len(header) != 9 or (dimension,scale_bytes,dtype,width) != (512,8,'bf16',2):
            raise ValueError('Fused insert requires its declared 512/64/584 storage ABI')
        heads,padded = map(int,header[7:])
        if padded not in (8,16,32,64,128) or not 0 < heads <= padded:raise ValueError('Invalid fused query head padding')
        return (('query',(tokens,heads,512),'bf16',2,False),('kv',(tokens,1,512),'bf16',2,False),
            ('positions',(tokens,),'int64',8,False),slots,('rope',(tokens,1,64),'fp32',4,True)), (
            ('query_live',(tokens,heads,512),'bf16',2,False),
            *((('query_pad',(tokens,padded-heads,512),'bf16',2,False),) if heads < padded else ()),
            ('cache_payload',(tokens,1,448),'fp8',1,True),('cache_rope',(tokens,1,64),'bf16',2,True),
            ('cache_scale',(tokens,1,8),'uint8',1,True))
    if len(header) != 7:raise ValueError('Unexpected target cache ABI fields')
    if variant == 'mla_kv_cache' and scale_bytes == 0:
        return (('latent',(tokens,1,dimension),dtype,width,True),slots), (('cache_payload',(tokens,1,dimension),dtype,width,True),)
    if variant == 'dsa_fp8_index_cache' and scale_bytes > 0 and (dtype,width) == ('bf16',2):
        scales = (scale_bytes//4,'fp32',4) if scale_bytes%4 == 0 else (scale_bytes,'uint8',1)
        return (('key',(tokens,1,dimension),'bf16',2,False),slots), (
            ('cache_payload',(tokens,1,dimension),'fp8',1,True),('cache_scale',(tokens,1,scales[0]),scales[1],scales[2],True))
    if (variant == 'dsv4_compressor_state_cache' and scale_bytes == 0 and (dtype,width) == ('fp32',4)
            and dimension in (512,1024,2048)):
        return (('projection',(tokens,1,dimension),'fp32',4,False),('positions',(tokens,),'int64',8,False),slots,
            ('ape',(tokens,1,dimension//2),'fp32',4,True)), (('cache_state',(tokens,1,dimension),'fp32',4,True),)
    raise ValueError('Target cache storage/quantization ABI mismatch')


def _target_cache_selection(header,role):
    return f'{header[0]}.{header[1]}.{role}.T{header[2]}.D{header[3]}.S{header[4]}' + (f'.H{header[7]}.P{header[8]}' if len(header) == 9 else '')


def standalone_target_cache_bindings(kernel, *, call_id,binding_id):
    from math import prod
    from src.kernel.memory_access_builder import ElementwiseBindings
    if kernel.variant == 'dsv4_compressed_state_cache':
        return _standalone_c4_finish_bindings(kernel,call_id=call_id,binding_id=binding_id)
    header = _target_cache_header(kernel);reads,writes = _target_cache_layouts(header)
    allocations,views,events,ready = [],[],[],[]
    cache = f'{binding_id}.cache'
    for role,shape,dtype,width,opaque in (*reads,*writes):
        output = role in {row[0] for row in writes}
        query_output = role in ('query_live','query_pad')
        identity = f'{binding_id}.query_output' if query_output else cache if output else f'{binding_id}.{role}'
        if identity not in {item.allocation_id for item in allocations}:
            allocation = MemoryAllocation(identity,'logical-local-device','logical-token-slot-cache' if output else 'logical-storage',
                int(header[2])*int(header[8])*512*2 if query_output else None if opaque else prod(shape)*width,
                AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
            allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if opaque:
            view = LogicalSelectionView(role,identity,shape,dtype,width,_target_cache_selection(header,role),None)
        else:
            strides,size = [],width
            for extent in reversed(shape):strides.append(size);size *= extent
            view = (TensorView(role,identity,int(header[7])*1024 if role == 'query_pad' else 0,shape,
                (int(header[8])*1024,1024,2),dtype,width) if query_output else
                TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width))
        views.append(view)
        if not output:
            kind = EventKind.LOGICAL_AVAILABLE if opaque else EventKind.INITIALIZED
            event = ProgramEvent(f'{identity}.available',kind,identity,view.region(),'target-cache-visible',
                (EventRef(f'{identity}.allocated',EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        tuple(row[0] for row in reads),tuple(row[0] for row in writes),tuple(ready),0,'offline','target-cache-visible')


def _validate_target_cache_views(header,views):
    reads,writes = _target_cache_layouts(header)
    if set(views) != {row[0] for row in (*reads,*writes)}:raise ValueError('Target cache operand roles missing')
    for role,shape,dtype,width,opaque in (*reads,*writes):
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes) != (shape,dtype,width):raise ValueError('Target cache typed operand mismatch')
        if opaque:
            if not isinstance(view,LogicalSelectionView) or view.parent_view_id is not None or view.selection_id != _target_cache_selection(header,role):
                raise ValueError('Target cache requires its actual opaque selection identity')
        elif not isinstance(view,TensorView):raise ValueError('Target cache requires actual typed input tensor views')
    if header[1] == 'dsv4_qnorm_rope_kv_insert':
        query = views['query_live'];padded = int(header[8]);heads = int(header[7])
        if query.strides_bytes != (padded*1024,1024,2):raise ValueError('Fused query output requires actual padded token stride')
        if heads < padded:
            pad = views['query_pad']
            if (pad.allocation_id,pad.offset_bytes,pad.strides_bytes) != (query.allocation_id,query.offset_bytes+heads*1024,query.strides_bytes):
                raise ValueError('Fused query padding must cover the same output parent after the live heads')
        cache_roles = ('cache_payload','cache_rope','cache_scale')
        if query.allocation_id in {views[row[0]].allocation_id for row in reads}|{views[role].allocation_id for role in cache_roles}:
            raise ValueError('Fused query output may not alias inputs or the KV cache')
        writes = tuple(row for row in writes if row[0] in cache_roles)
    caches = {views[row[0]].allocation_id for row in writes};inputs = {views[row[0]].allocation_id for row in reads}
    if len(caches) != 1 or len(inputs) != len(reads) or caches&inputs:raise ValueError('Target cache input/cache aliases require a native lifetime contract')


def _target_cache_tile_recipe(header,views,ready,scope,visibility):
    reads,writes = _target_cache_layouts(header);variant = header[1]
    if variant == 'dsv4_qnorm_rope_kv_insert':
        return _fused_insert_tile_recipe(header,views,ready,scope,visibility)
    if variant=='dsa_fp8_index_cache' and int(header[3])%128==0 and int(header[4])==int(header[3])//128*4:
        return _dsa_warp_cache_tile_recipe(header,views,ready,scope,visibility)
    evidence = ('vLLM logical cache append: one token and its actual INT64 slot; slot values/pages/negative-slot masks remain unknown.',
        'FP8 key and FP32-storage scale stores consume the actual BF16 row reduction; no fabricated FP8 intermediate memory read.',
        'V4 save_partial_states consumes FP32 KV/score projection, position-selected FP32 APE, INT64 positions and slots.',
        'Candidate logical requests retain the existing all-valid-slot price projection; masked native traffic and distinct footprint are not inferred.')
    accesses = [TileAccessTemplate('slots','slots',Operation.READ,('token',),('token',),(ready['slots'],),(),evidence)]
    if variant == 'dsv4_compressor_state_cache':
        accesses.append(TileAccessTemplate('positions','positions',Operation.READ,('token',),('token',),(ready['positions'],),(),evidence))
    for role,shape,dtype,width,opaque in reads:
        if role in ('slots','positions'):continue
        deps = (TileDependency('slots',EventKind.READ_READY,()),)
        if role == 'ape':deps += (TileDependency('positions',EventKind.READ_READY,()),)
        accesses.append(TileAccessTemplate(role,role,Operation.READ,('token',),('token',None,None),
            (ready[role],),deps,evidence,unmapped=opaque,
            predicate=ReadValuePredicate('slots','nonnegative') if variant in ('mla_kv_cache','dsa_fp8_index_cache') else None))
    values = {'mla_kv_cache':('latent',),'dsa_fp8_index_cache':('key',),'dsv4_compressor_state_cache':('projection','ape')}[variant]
    if len(header) == 8 and header[-1] == 'mla_split_512_64':values = ('latent', 'rope')
    for role,shape,dtype,width,opaque in writes:
        component = (KVComponent.SCALE if role == 'cache_scale' else KVComponent.KEY if variant == 'dsa_fp8_index_cache'
            else KVComponent.STATE if variant == 'dsv4_compressor_state_cache' else KVComponent.LATENT)
        target = TileIndirectTarget(views[role].allocation_id,'slots',Indirection.TOKEN_SLOT,component,values)
        accesses.append(TileAccessTemplate(role,role,Operation.WRITE,('token',),('token',None,None),(),
            (TileDependency('slots',EventKind.READ_READY,()),*(TileDependency(value,EventKind.READ_READY,()) for value in values)),
            evidence,unmapped=True,indirect_target=target,
            predicate=ReadValuePredicate('slots','nonnegative') if variant in ('mla_kv_cache','dsa_fp8_index_cache') else None))
    return TileProgram(scope.scope_id,scope,(TileAxis('token',int(header[2]),1),),tuple(accesses),visibility,evidence)


def _dsa_warp_cache_tile_recipe(header,views,ready,scope,visibility):
    """Source-defined 32-lane by four-value independent quantization groups."""
    evidence=('vLLM0.27.1 cache_kernels.cu indexer_k_quant_and_cache_kernel uses VEC_SIZE4 and five warp XOR max rounds; each warp covers128 BF16 values.',
        'This source domain has quant_block_size128 and one FP32 scale per128 values; each output waits only for its own group.',
        'Slot metadata is a logical unique read within each CTA and is reread across CTAs; this is not native per-lane issue traffic or cache residency.',
        'The source body uses warp registers/shuffles for this reduction, without a shared-memory temporary buffer.',
        'Other legacy scale descriptors keep their explicit logical row ABI rather than inventing a native warp partition.')
    predicate=ReadValuePredicate('slots','nonnegative')
    metadata=TileDependency('slots',EventKind.READ_READY,(),(TileIndexBinding('cta',(TileIndexTerm('component',1),),divisor=4),))
    values=TileDependency('key',EventKind.READ_READY,());mapping=('token',None,'component');loop=('token','component')
    accesses=[TileAccessTemplate('slots','slots',Operation.READ,('token','cta'),('token',),(ready['slots'],),(),evidence),
        TileAccessTemplate('key','key',Operation.READ,loop,mapping,(ready['key'],),(metadata,),evidence,predicate=predicate)]
    for role,component,divisors in (('cache_payload',KVComponent.KEY,()),('cache_scale',KVComponent.SCALE,(1,1,128))):
        target=TileIndirectTarget(views[role].allocation_id,'slots',Indirection.TOKEN_SLOT,component,('key',))
        accesses.append(TileAccessTemplate(role,role,Operation.WRITE,loop,mapping,(),(metadata,values),evidence,
            unmapped=True,indirect_target=target,view_axis_divisors=divisors,predicate=predicate))
    return TileProgram(scope.scope_id,scope,(TileAxis('token',int(header[2]),1),TileAxis('component',int(header[3]),128),
        TileAxis('cta',-(-int(header[3])//512),1)),
        tuple(accesses),visibility,evidence)


def build_target_cache_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_target_cache_source(kernel):raise NotImplementedError('Target cache source ABI is not admitted')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown target cache logical granularity')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('Target cache requires typed operand and slot bindings')
    if kernel.variant == 'dsv4_compressed_state_cache':
        program=_build_compressed_finish_source(kernel,hardware,bindings,program_id=program_id)
        target_cache_source_demands(program,expected_kernel=kernel)
        return program
    header = _target_cache_header(kernel)
    if kernel.variant == 'mla_kv_cache' and bindings.read_views == ('latent', 'rope', 'slots'):
        header = (*header, 'mla_split_512_64')
    reads,writes = _target_cache_layouts(header)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (tuple(row[0] for row in reads),tuple(row[0] for row in writes),len(reads)):
        raise ValueError('Target cache input/output/readiness roles mismatch')
    views = unique_by(bindings.views,'view_id');_validate_target_cache_views(header,views)
    scope = AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-token-slot-transform')
    family = _target_cache_tile_recipe(header,views,dict(zip(bindings.read_views,bindings.input_ready,strict=True)),scope,bindings.visibility_domain)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Signed slot values, skipped-token masks, pages, native capacities and store address transforms remain unknown; no executable backend.',
            'Logical candidate requests retain the original all-valid slot service projection; they are not observed emitted bytes or residency.',
            'MLA latent input is a logical concatenated interface; native component addresses/packing and persistent producer aliases remain unbound.',
            'Non-FP32-aligned legacy scale-byte descriptors stay opaque bytes, not a certified native quantizer ABI.',
            'V4 APE row values/position mapping and state-cache lifetime remain unknown; save-partial work is not repeated finish work.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    target_cache_source_demands(program,expected_kernel=kernel)
    return program


def target_cache_source_demands(program, *, expected_kernel=None):
    from src.kernel.memory_access import tile_template_requested_bytes
    program.__post_init__()
    if expected_kernel is not None:
        if (not has_target_cache_source(expected_kernel) or program.hardware_id!=expected_kernel.hardware_id
                or not program.tile_programs):
            raise ValueError('Target cache source differs from its actual kernel')
        compressed=expected_kernel.variant=='dsv4_compressed_state_cache'
        header=_compressed_finish_header(expected_kernel) if compressed else _target_cache_header(expected_kernel)
        if expected_kernel.variant == 'mla_kv_cache' and program.tile_programs[0].scope.op_ids == (*header, 'mla_split_512_64'):
            header = (*header, 'mla_split_512_64')
        if any((family.scope.op_ids[:-2] if compressed else family.scope.op_ids)!=header for family in program.tile_programs):
            raise ValueError('Target cache source differs from its actual kernel')
    if program.tile_programs and program.tile_programs[0].scope.op_ids[0] in ('compressed_finish_c4','compressed_finish_c128'):
        return _compressed_finish_demands(program)
    if program.accesses or len(program.tile_programs) != 1:raise ValueError('Target cache requires its unique token/slot tile source')
    family, = program.tile_programs;header = family.scope.op_ids;views = unique_by(program.views,'view_id')
    reads,writes = _target_cache_layouts(header);_validate_target_cache_views(header,views)
    accesses = unique_by(family.accesses,'access_id');ready = {}
    for role,*_ in reads:
        if len(accesses[role].input_ready) != 1:raise ValueError('Target cache actual input availability missing')
        ready[role] = accesses[role].input_ready[0]
    if family != _target_cache_tile_recipe(header,views,ready,family.scope,family.visibility_domain):
        raise ValueError('Target cache lacks actual slot/value/position READ_READY causality')
    values = {access.access_id:tile_template_requested_bytes(program,family,access) for access in family.accesses}
    input_role = {'mla_kv_cache':'latent','dsa_fp8_index_cache':'key','dsv4_compressor_state_cache':'projection',
        'dsv4_qnorm_rope_kv_insert':'query'}[header[1]]
    return {'variant':header[1],'tokens':int(header[2]),'dimension':int(header[3]),
        'input_elements':values[input_role]//views[input_role].element_bytes + (values['rope']//2 if header[-1] == 'mla_split_512_64' else 0),
        'scale_bytes':values['cache_scale'] if 'cache_scale' in values else 0,
        'read_bytes':sum(values[row[0]] for row in reads),'write_bytes':sum(values[row[0]] for row in writes),
        'legacy_unique_slot_read_price_bytes':sum(values[row[0]] for row in reads)-values['slots']+views['slots'].shape[0]*8,
        'effective_masked_write_bytes':None}


def _fused_insert_tile_recipe(header,views,ready,scope,visibility):
    """Per-token metadata and independent full-row Q/KV semantics, not issue counts."""
    tokens,heads = int(header[2]),int(header[7])
    evidence = ('vLLM v0.30 fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu processDeepseekV4Slot: live head RMSNorm, interleaved last-64 RoPE, independent padding and KV branches.',
        'One semantic head-row tile; token position/cos-sin are shared logical unique selections, not native per-warp load counts or observed DRAM.',
        'KV record selections share one unknown cache parent: FP8[448], BF16[64], UE8M0[8] including zero scale pad; pages and values unresolved.')
    templates = []
    for role in ('positions','slots','rope','query','kv'):
        iteration = ('token','head') if role == 'query' else ('token',)
        coordinates = ('token','head',None) if role == 'query' else ('token',None,None) if role in ('kv','rope') else ('token',)
        deps = (TileDependency('positions',EventKind.READ_READY,()),) if role == 'rope' else ()
        templates.append(TileAccessTemplate(role,role,Operation.READ,iteration,coordinates,(ready[role],),deps,evidence,
            unmapped=role == 'rope'))
    templates.append(TileAccessTemplate('query_live','query_live',Operation.WRITE,('token','head'),('token','head',None),(),
        (TileDependency('query',EventKind.READ_READY,()),TileDependency('rope',EventKind.READ_READY,())),evidence))
    if 'query_pad' in views:
        templates.append(TileAccessTemplate('query_pad','query_pad',Operation.WRITE,('token',),('token',None,None),(),(),evidence,
            constant_value=ConstantValue.ZERO))
    for role,component in (('cache_payload',KVComponent.KEY),('cache_rope',KVComponent.LATENT),('cache_scale',KVComponent.SCALE)):
        values = ('kv','rope')
        target = TileIndirectTarget(views[role].allocation_id,'slots',Indirection.TOKEN_SLOT,component,values)
        templates.append(TileAccessTemplate(role,role,Operation.WRITE,('token',),('token',None,None),(),
            tuple(TileDependency(value,EventKind.READ_READY,()) for value in ('slots',*values)),evidence,
            unmapped=True,indirect_target=target))
    return TileProgram(scope.scope_id,scope,(TileAxis('token',tokens,1),TileAxis('head',heads,1)),tuple(templates),visibility,evidence)


def _c4_finish_header(kernel):
    return ('compressed_finish_c4',str(kernel.shape['state_bytes']),str(kernel.shape['seq_len']),
        str(len(kernel.attributes['state_old_lens'])),*(str(value) for pair in zip(
            kernel.attributes['state_old_lens'],kernel.attributes['state_new_lens'],strict=True) for value in pair))


def _c4_finish_layout(header):
    """O(requests) typed operands and at most two boundary groups per request."""
    if len(header) < 6 or header[0] != 'compressed_finish_c4':raise ValueError('C4 finish source header missing')
    record,tokens,requests = map(int,header[1:4])
    if record not in (132,584) or tokens <= 0 or requests <= 0 or len(header) != 4+2*requests:
        raise ValueError('Invalid C4 finish record or request header')
    dim = 128 if record == 132 else 512
    operands,groups,request_rows = [],[],[]
    def operand(role,shape,dtype,width,opaque,parent,output=False):
        operands.append((role,shape,dtype,width,opaque,parent,output))
    total = 0
    for request in range(requests):
        old,new = map(int,header[4+request*2:6+request*2]);count = new-old
        if old < 0 or new <= 0 or count < 0:raise ValueError('C4 request lengths must be nonnegative and monotonic')
        total += count
        if count == 0:continue
        prefix = f'r{request}'
        request_rows.append((request,count))
        for role in ('slots','positions',*(('cache_slots',) if record == 584 else ())):
            operand(f'{prefix}.{role}',(count,),'int64',8,False,f'{prefix}.{role}')
        states = new//4-old//4
        if not states:continue
        first_end = (old//4+1)*4
        # The only short C4 window is the initial boundary at position 3.
        partitions = ((0,1,4),(1,states-1,8)) if first_end == 4 else ((0,states,8),)
        for group,(first,size,window) in enumerate(partitions):
            if not size:continue
            label = f'{prefix}.g{group}';first_token = first_end+4*first-old-1
            groups.append((request,group,size,window,first_token))
            operand(f'{label}.request_index',(size,),'int32',4,True,'request_indices')
            operand(f'{label}.block_table',(size,window),'int32',4,True,'block_table')
            for role in ('kv','score'):
                operand(f'{label}.{role}',(size,window,dim),'fp32',4,True,'partial_state_cache')
            operand(f'{label}.rope',(size,1,64),'fp32',4,True,'rope_table')
            if record == 132:
                operand(f'{label}.cache_slots',(size,),'int64',8,True,'destination_slots')
                output = (('payload',128,'fp8',1),('scale',1,'fp32',4))
            else:
                output = (('payload',448,'fp8',1),('rope_output',64,'bf16',2),('scale',8,'uint8',1))
            for role,width,dtype,element_bytes in output:
                operand(f'{label}.{role}',(size,1,width),dtype,element_bytes,True,'output_cache',True)
    if total != tokens:raise ValueError('C4 token guards must cover every request delta exactly')
    if groups:operand('gamma',(dim,),'bf16',2,False,'norm_gamma')
    return tuple(operands),tuple(request_rows),tuple(groups)


def _c4_selection(header,role):
    # The name binds old/new lengths, request and boundary group; no slot value
    # or address is invented. Blocks and state rows remain typed selections.
    return '.'.join((*header,role))


def _standalone_c4_finish_bindings(kernel, *, call_id,binding_id):
    from math import prod
    from src.kernel.memory_access_builder import ElementwiseBindings
    header = _compressed_finish_header(kernel);operands,_,_ = _compressed_finish_layout(header)
    allocations,views,events,ready = [],[],[],[]
    allocation_ids = set()
    for role,shape,dtype,width,opaque,parent,output in operands:
        identity = f'{binding_id}.{parent}'
        if identity not in allocation_ids:
            allocation_ids.add(identity)
            allocation = MemoryAllocation(identity,'logical-local-device','logical-compressor-state',
                int(header[2])*512*4 if parent == 'scratch' else None if opaque else prod(shape)*width,
                AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
            allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if opaque:
            view = LogicalSelectionView(role,identity,shape,dtype,width,_c4_selection(header,role),None)
        else:
            strides,size = [],width
            for extent in reversed(shape):strides.append(size);size *= extent
            view = (TensorView(role,identity,_c128_scratch_offset(header,role),shape,(128*512*4,64*4,4),dtype,width)
                if parent == 'scratch' else TensorView(role,identity,0,shape,tuple(reversed(strides)),dtype,width))
        views.append(view)
        if not output:
            kind = EventKind.LOGICAL_AVAILABLE if opaque else EventKind.INITIALIZED
            event = ProgramEvent(f'{binding_id}.{role}.available',kind,identity,view.region(),'c4-finish-visible',
                (EventRef(f'{identity}.allocated',EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        tuple(row[0] for row in operands if not row[-1]),tuple(row[0] for row in operands if row[-1]),tuple(ready),
        0,'offline','c4-finish-visible')


def dsv4_c128_stage_bindings(aggregate_kernel, stage_kernel, bindings):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings, ElementwiseBindings):
        raise TypeError("C128 stage projection requires typed aggregate bindings")
    if (aggregate_kernel.variant != "dsv4_compressed_state_cache"
            or aggregate_kernel.attributes["finish_stage"] is not None
            or stage_kernel.variant != aggregate_kernel.variant
            or stage_kernel.attributes["finish_stage"] not in ("compress", "store")):
        raise ValueError("C128 stage bindings require an aggregate-to-stage source transition")
    aggregate_header = _compressed_finish_header(aggregate_kernel)
    aggregate_operands, _, _ = _compressed_finish_layout(aggregate_header)
    aggregate_reads = tuple(row[0] for row in aggregate_operands if not row[-1])
    aggregate_writes = tuple(row[0] for row in aggregate_operands if row[-1])
    if (bindings.read_views, bindings.write_views, len(bindings.input_ready)) != (
            aggregate_reads, aggregate_writes, len(aggregate_reads)):
        raise ValueError("C128 aggregate bindings do not match their actual kernel")
    aggregate_views = unique_by(bindings.views, "view_id")
    _validate_finish_views(aggregate_header, aggregate_views)
    expected = _standalone_c4_finish_bindings(stage_kernel,
        call_id=bindings.call_id, binding_id=bindings.binding_id)
    expected_views = unique_by(expected.views, "view_id")
    views = []
    for role, expected_view in expected_views.items():
        view = aggregate_views[role]
        if (view.shape, view.dtype, view.element_bytes) != (
                expected_view.shape, expected_view.dtype, expected_view.element_bytes):
            raise ValueError("C128 stage projection changed an actual typed view")
        if isinstance(expected_view, LogicalSelectionView):
            if not isinstance(view, LogicalSelectionView):
                raise TypeError("C128 logical stage selection lost its actual view type")
            view = replace(view, selection_id=expected_view.selection_id)
        views.append(view)
    stage_views = unique_by(views, "view_id")
    allocations = unique_by(bindings.allocations, "allocation_id")
    ready_by_role = dict(zip(bindings.read_views, bindings.input_ready, strict=True))
    events = list(bindings.events)
    input_ready = []
    for role in expected.read_views:
        if stage_kernel.attributes["finish_stage"] == "store" and role.endswith(".scratch"):
            view = stage_views[role]
            event_id = f"{bindings.binding_id}.{role}.compress_initialized"
            event = ProgramEvent(event_id, EventKind.INITIALIZED, view.allocation_id,
                view.region(), bindings.visibility_domain,
                (EventRef(allocations[view.allocation_id].allocated_event, EventKind.ALLOCATED),))
            events.append(event)
            input_ready.append(EventRef(event.event_id, event.kind))
        else:
            ready = ready_by_role[role]
            for index, existing in enumerate(events):
                if existing.event_id == ready.node_id:
                    events[index] = replace(existing, region=stage_views[role].region())
            input_ready.append(ready)
    return ElementwiseBindings(bindings.binding_id, bindings.call_id,
        bindings.allocations, tuple(views), tuple(events), expected.read_views,
        expected.write_views, tuple(input_ready), bindings.rank, bindings.stream,
        bindings.visibility_domain)


def _validate_finish_views(header,views):
    operands,_,_ = _compressed_finish_layout(header)
    if set(views) != {row[0] for row in operands}:raise ValueError('C4 finish operand roles incomplete')
    parents = {};identities = {}
    for role,shape,dtype,width,opaque,parent,output in operands:
        view = views[role]
        if (view.shape,view.dtype,view.element_bytes) != (shape,dtype,width):raise ValueError('C4 finish typed operand mismatch')
        if opaque:
            if (not isinstance(view,LogicalSelectionView) or view.parent_view_id is not None
                    or view.selection_id != _c4_selection(header,role)):
                raise ValueError('C4 finish requires exact request/window/slot selections')
        elif not isinstance(view,TensorView):raise ValueError('Compressed finish requires real typed metadata/gamma/scratch views')
        if parent == 'scratch' and (view.offset_bytes,view.strides_bytes) != (_c128_scratch_offset(header,role),(128*512*4,64*4,4)):
            raise ValueError('C128 scratch must preserve its actual token/column offsets and strides')
        if parent in parents and parents[parent] != view.allocation_id:raise ValueError('C4 source selections lost their shared parent')
        if view.allocation_id in identities and identities[view.allocation_id] != parent:
            raise ValueError('C4 independent inputs and output may not acquire an undeclared alias')
        parents[parent] = view.allocation_id;identities[view.allocation_id] = parent


def _c4_finish_families(header,views,ready,program_id,call_id,rank,stream,visibility):
    _,requests,groups = _c4_finish_layout(header);families = []
    record = int(header[1])
    request_groups = {request:[] for request,_ in requests}
    for group in groups:request_groups[group[0]].append(group)
    evidence = ('vLLM v0.30 C4 CuTe sparse_attn_compress_cutedsl and indexer fused_compress_quant_cache: one completed boundary consumes its valid 4/8-row KV/score window, then full-column RMSNorm and interleaved last-64 RoPE.',
        'Native all-token metadata guards are separate from boundary-only request/block-table/state/gamma/rope/output work; no save-partial producer repetition.',
        'Window selections bind exact request old/new lengths and overlap coefficient groups; state page values and cache destinations remain unknown, not addressable regions.',
        'Token/row logical traffic is the existing all-valid unique envelope, not per-lane native issue counts or observed DRAM.')
    for request,count in requests:
        prefix = f'r{request}';axes = [TileAxis('token',count,1)];templates = []
        for role in ('slots','positions',*(('cache_slots',) if record == 584 else ())):
            name = f'{prefix}.{role}'
            templates.append(TileAccessTemplate(name,name,Operation.READ,('token',),('token',),(ready[name],),(),evidence))
        for req,group,size,window,first_token in request_groups[request]:
            label = f'{prefix}.g{group}';axis = f'boundary{group}';axes.append(TileAxis(axis,size,1))
            mapping = (TileIndexBinding('token',(TileIndexTerm(axis,4),),first_token),)
            def guard(role):return TileDependency(f'{prefix}.{role}',EventKind.READ_READY,(),mapping)
            def dep(role):return TileDependency(f'{label}.{role}',EventKind.READ_READY,())
            for role in ('request_index','block_table','kv','score','rope',*(('cache_slots',) if record == 132 else ())):
                name = f'{label}.{role}';view = views[name]
                dependencies = (guard('slots'),guard('positions'))
                if role == 'block_table':dependencies += (dep('request_index'),)
                elif role in ('kv','score'):dependencies += (dep('block_table'),)
                templates.append(TileAccessTemplate(name,name,Operation.READ,(axis,),(axis,*(None for _ in view.shape[1:])),
                    (ready[name],),dependencies,evidence,unmapped=True))
            gamma = f'{label}.gamma'
            templates.append(TileAccessTemplate(gamma,'gamma',Operation.READ,(axis,),(None,),(ready['gamma'],),
                (guard('positions'),guard('slots')),evidence))
            dependencies = (dep('kv'),dep('score'),dep('rope'),TileDependency(gamma,EventKind.READ_READY,()),
                dep('cache_slots') if record == 132 else guard('cache_slots'))
            for role in ('payload','scale',*(('rope_output',) if record == 584 else ())):
                name = f'{label}.{role}'
                templates.append(TileAccessTemplate(name,name,Operation.WRITE,(axis,),(axis,None,None),(),dependencies,evidence,unmapped=True))
        scope = AccessScope(f'{program_id}.{prefix}',(*header,'request',str(request)),call_id,(),rank,stream,'logical-compression-boundary')
        families.append(TileProgram(scope.scope_id,scope,tuple(axes),tuple(templates),visibility,evidence))
    return tuple(families)


def _build_compressed_finish_source(kernel,hardware,bindings, *, program_id):
    header = _compressed_finish_header(kernel);operands,_,_ = _compressed_finish_layout(header)
    reads = tuple(row[0] for row in operands if not row[-1]);writes = tuple(row[0] for row in operands if row[-1])
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (reads,writes,len(reads)):
        raise ValueError('C4 finish requires its actual typed guard/window/value readiness bindings')
    views = unique_by(bindings.views,'view_id');_validate_finish_views(header,views)
    families = _compressed_finish_families(header,views,dict(zip(reads,bindings.input_ready,strict=True)),program_id,
        bindings.call_id,bindings.rank,bindings.stream,bindings.visibility_domain)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Completed-boundary tiles preserve C4 short/overlap windows and C128 full-window column splits.',
            'Signed slot masks, page values, actual producer aliases, positions/cos-sin values and native issued loads remain unresolved.',
            'C128 exposes full-token FP32 scratch with selected boundary rows; standalone aggregate is two logical launches, not one native kernel.')),
        bindings.allocations,bindings.views,(),bindings.events,(),families)
    _compressed_finish_demands(program)
    return program


def _compressed_finish_demands(program):
    from src.kernel.memory_access import tile_template_requested_bytes
    if program.accesses or not program.tile_programs:raise ValueError('C4 finish requires its actual boundary tile source')
    first = program.tile_programs[0];header = first.scope.op_ids[:-2]
    operands,_,_ = _compressed_finish_layout(header);views = unique_by(program.views,'view_id');_validate_finish_views(header,views)
    ready = {}
    for family in program.tile_programs:
        for access in family.accesses:
            if access.operation is Operation.READ:
                if (not access.input_ready and header[0] == 'compressed_finish_c128'
                        and _c128_finish_stage(header) == 'aggregate' and access.access_id.endswith('.scratch_read')):
                    continue
                if len(access.input_ready) != 1:raise ValueError('C4 finish input availability missing')
                if access.view_id in ready and ready[access.view_id] != access.input_ready[0]:raise ValueError('C4 input readiness differs across boundary groups')
                ready[access.view_id] = access.input_ready[0]
    if set(ready) != {row[0] for row in operands if not row[-1]}:raise ValueError('C4 finish actual reads incomplete')
    scope = first.scope
    expected = _compressed_finish_families(header,views,ready,program.program_id,scope.call_id,scope.rank,scope.stream,first.visibility_domain)
    if program.tile_programs != expected:raise ValueError('C4 finish lost actual request/window/guard/value READ_READY causality')
    read_bytes = write_bytes = rows = window_elements = states = tokens = 0
    for family in program.tile_programs:
        for access in family.accesses:
            value = tile_template_requested_bytes(program,family,access)
            if access.operation is Operation.READ:read_bytes += value
            else:write_bytes += value
            if access.access_id.endswith('.block_table'):rows += value//4
            if access.access_id.endswith('.kv'):window_elements += value//4
            if access.access_id.endswith('.payload'):states += value//views[access.view_id].shape[-1]
            if access.access_id.endswith('.scratch_write') and _c128_finish_stage(header) == 'compress':states += value//(512*4)
            if access.access_id.endswith('.slots'):tokens += value//8
    dim = 128 if header[1] == '132' else 512
    expected_tokens = int(header[2])*(2 if header[0] == 'compressed_finish_c128' and _c128_finish_stage(header) == 'aggregate' else 1)
    if tokens != expected_tokens:raise ValueError('Compressed finish token guards do not match the declared stages')
    return {'variant':'dsv4_compressed_state_cache','tokens':int(header[2]),'dimension':dim,'states':states,'rows':rows,
        'window_elements':window_elements,'state_elements':states*dim,'read_bytes':read_bytes,'write_bytes':write_bytes,
        'effective_masked_write_bytes':None}


def _compressed_finish_header(kernel):
    header = _c4_finish_header(kernel)
    if kernel.shape['compress_ratio'] == 4:return header
    if kernel.shape['compress_ratio'] != 128:raise ValueError('Unknown compressed finish ratio')
    stage = kernel.attributes['finish_stage']
    return ('compressed_finish_c128',*header[1:],'aggregate' if stage is None else stage)


def _c128_finish_stage(header):
    if header[0] != 'compressed_finish_c128' or header[-1] not in ('compress','store','aggregate'):
        raise ValueError('C128 requires its actual compress/store stage')
    return header[-1]


def _compressed_finish_layout(header):
    if header[0] == 'compressed_finish_c4':return _c4_finish_layout(header)
    if header[0] == 'compressed_finish_c128':return _c128_finish_layout(header)
    raise ValueError('Unknown compressed finish tile ABI')


def _compressed_finish_families(header,views,ready,program_id,call_id,rank,stream,visibility):
    if header[0] == 'compressed_finish_c4':
        return _c4_finish_families(header,views,ready,program_id,call_id,rank,stream,visibility)
    if header[0] == 'compressed_finish_c128':
        return _c128_finish_families(header,views,ready,program_id,call_id,rank,stream,visibility)
    raise ValueError('Unknown compressed finish tile ABI')


def _c128_finish_layout(header):
    stage = _c128_finish_stage(header);record,tokens,requests = map(int,header[1:4])
    if record != 584 or tokens <= 0 or requests <= 0 or len(header) != 5+2*requests:
        raise ValueError('C128 requires its 512D/584B request ABI')
    compress,store = stage != 'store',stage != 'compress'
    operands,groups,request_rows = [],[],[];total = 0
    def operand(role,shape,dtype,width,opaque,parent,output=False):
        operands.append((role,shape,dtype,width,opaque,parent,output))
    for request in range(requests):
        old,new = map(int,header[4+2*request:6+2*request]);count = new-old
        if old < 0 or new <= 0 or count < 0:raise ValueError('C128 request lengths must be monotonic')
        total += count
        if not count:continue
        request_rows.append((request,count));prefix = f'r{request}'
        for phase in ('compress','store'):
            if (phase == 'compress' and not compress) or (phase == 'store' and not store):continue
            for role,dtype,width in (('slots','int64',8),('positions','int64',8),
                    ('request_index','int32',4) if phase == 'compress' else ('cache_slots','int64',8)):
                operand(f'{prefix}.{phase}.{role}',(count,),dtype,width,False,f'{prefix}.{role}')
        states = new//128-old//128
        if not states:continue
        first_token = (old//128+1)*128-old-1
        groups.append((request,0,states,128,first_token));label = f'{prefix}.g0'
        if compress:
            # Eight 64-column CTAs, each requesting sixteen block-8 page ids.
            operand(f'{label}.block_table',(states,8,16),'int32',4,True,'block_table')
            for role in ('kv','score'):
                operand(f'{label}.{role}',(states,8,128,64),'fp32',4,True,'partial_state_cache')
        operand(f'{prefix}.scratch',(states,8,64),'fp32',4,False,'scratch',compress)
        if store:
            operand(f'{label}.rope',(states,1,64),'fp32',4,True,'rope_table')
            for role,width,dtype,element in (('payload',448,'fp8',1),('rope_output',64,'bf16',2),('scale',8,'uint8',1)):
                operand(f'{label}.{role}',(states,1,width),dtype,element,True,'output_cache',True)
    if total != tokens:raise ValueError('C128 guards must cover exact request deltas')
    if groups and store:operand('gamma',(512,),'bf16',2,False,'norm_gamma')
    return tuple(operands),tuple(request_rows),tuple(groups)


def _c128_scratch_offset(header,role):
    if _c128_finish_stage(header) not in ('compress','store','aggregate'):raise ValueError('Unknown scratch stage')
    request = int(role.split('.')[0][1:]);prefix_tokens = 0
    for index in range(request):
        old,new = map(int,header[4+2*index:6+2*index]);prefix_tokens += new-old
    old = int(header[4+2*request]);first_token = (old//128+1)*128-old-1
    return (prefix_tokens+first_token)*512*4


def _c128_finish_families(header,views,ready,program_id,call_id,rank,stream,visibility):
    stage = _c128_finish_stage(header);_,requests,groups = _c128_finish_layout(header)
    compress,store = stage != 'store',stage != 'compress'
    group_by_request = {row[0]:row for row in groups};families = []
    evidence = ('vLLM v0.30 SparseAttnCompressC128Block8Kernel: eight 64-column splits, 128 FP32 KV/score rows, sixteen block-8 page ids per split, FP32 scratch[token,512].',
        'SparseAttnNormRopeStoreKernel reads the completed FP32 scratch row, BF16 gamma, position-selected FP32 RoPE and slots before packed 584B cache insert.',
        'Scratch row is the original boundary token index, not a compact completed-state index; inactive token rows remain unwritten.',
        'Position/slot/request guards retain the existing logical unique envelope; not per-warp replicated issue counts or observed DRAM.',
        'Aggregate describes two source launches; actual Serving uses two calls and its explicit producer dependency, never a fabricated one-kernel ABI.')
    for request,count in requests:
        prefix = f'r{request}';axes = [TileAxis('token',count,1)];templates = []
        for phase in ('compress','store'):
            if (phase == 'compress' and not compress) or (phase == 'store' and not store):continue
            for role in ('slots','positions','request_index' if phase == 'compress' else 'cache_slots'):
                name = f'{prefix}.{phase}.{role}'
                templates.append(TileAccessTemplate(name,name,Operation.READ,('token',),('token',),(ready[name],),(),evidence))
        if request in group_by_request:
            _,_,size,_,first_token = group_by_request[request];label = f'{prefix}.g0'
            axes.extend((TileAxis('boundary',size,1),TileAxis('head',8,1)))
            mapping = (TileIndexBinding('token',(TileIndexTerm('boundary',128),),first_token),)
            def guard(phase,role):return TileDependency(f'{prefix}.{phase}.{role}',EventKind.READ_READY,(),mapping)
            def dep(role):return TileDependency(f'{label}.{role}',EventKind.READ_READY,())
            if compress:
                for role in ('block_table','kv','score'):
                    name = f'{label}.{role}';view = views[name]
                    dependencies = tuple(guard('compress',role) for role in ('slots','positions','request_index'))
                    if role != 'block_table':dependencies += (dep('block_table'),)
                    templates.append(TileAccessTemplate(name,name,Operation.READ,('boundary','head'),
                        ('boundary','head',*(None for _ in view.shape[2:])),(ready[name],),dependencies,evidence,unmapped=True))
                templates.append(TileAccessTemplate(f'{prefix}.scratch_write',f'{prefix}.scratch',Operation.WRITE,
                    ('boundary','head'),('boundary','head',None),(),(dep('kv'),dep('score')),evidence))
            if store:
                dependencies = tuple(guard('store',role) for role in ('slots','positions','cache_slots'))
                if compress:dependencies += (TileDependency(f'{prefix}.scratch_write',EventKind.WRITE_VISIBLE,()),)
                templates.append(TileAccessTemplate(f'{prefix}.scratch_read',f'{prefix}.scratch',Operation.READ,
                    ('boundary','head'),('boundary','head',None),() if compress else (ready[f'{prefix}.scratch'],),dependencies,evidence))
                for role in ('gamma','rope'):
                    name = f'{label}.{role}';view_role = 'gamma' if role == 'gamma' else name
                    view = views[view_role]
                    templates.append(TileAccessTemplate(name,view_role,Operation.READ,('boundary',),
                        (None,) if role == 'gamma' else ('boundary',None,None),(ready[view_role],),
                        tuple(guard('store',role) for role in ('positions','slots')),evidence,unmapped=role == 'rope'))
                output_deps = (TileDependency(f'{prefix}.scratch_read',EventKind.READ_READY,('head',)),
                    dep('gamma'),dep('rope'),guard('store','cache_slots'))
                for role in ('payload','rope_output','scale'):
                    name = f'{label}.{role}'
                    templates.append(TileAccessTemplate(name,name,Operation.WRITE,('boundary',),('boundary',None,None),(),output_deps,evidence,unmapped=True))
        scope = AccessScope(f'{program_id}.{prefix}',(*header,'request',str(request)),call_id,(),rank,stream,'logical-compression-boundary')
        families.append(TileProgram(scope.scope_id,scope,tuple(axes),tuple(templates),visibility,evidence))
    return tuple(families)
