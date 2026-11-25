"""Declared int32 nullary routing fills, without fabricated fill values."""
from dataclasses import dataclass, replace
from functools import cached_property
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,
    MemoryAccessProgram,MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,
    TensorView,TileAxis,TileAccessTemplate,TileProgram,
)


def has_moe_fill_source(kernel,hardware):
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant != 'moe_fill'
            or kernel.execution_support != 'supported' or hardware.kernel_memory_model != 'gpu_mshr'):
        return False
    if (kernel.hardware_id == 'metax_c500' and kernel.launch.algorithm in (
            'pytorch_2_4_metax_int32_fill','pytorch_2_4_metax_int32_fill_nullary_opt')):
        from src.kernel.launch_config.select_launch_config import select_launch_config
        expected=select_launch_config(hardware.hardware_id,KernelKind.VECTOR,_execution_shape(kernel),kernel.dtype,'moe_fill',
            {'fill_role':kernel.attributes['fill_role'],'launch_policy':'injected_launch'},hardware)
        if (kernel.launch.algorithm,kernel.launch.grid,kernel.launch.block) != (expected.algorithm,expected.grid,expected.block):
            raise ValueError('Source-bound nullary fill must preserve its launch-owner output partition')
        return True
    if kernel.launch.algorithm == 'pytorch_int32_scalar_fill':
        if kernel.hardware_id != 'nvidia_h200':
            raise ValueError('Native scalar fill requires its H200 source owner')
        from src.kernel.launch_config.select_launch_config import select_launch_config
        expected=select_launch_config(hardware.hardware_id,KernelKind.VECTOR,_execution_shape(kernel),kernel.dtype,'moe_fill',
            {'fill_role':kernel.attributes['fill_role']},hardware)
        if kernel.launch != expected:
            raise ValueError('Native scalar fill must preserve its source-owned launch')
        return True
    if kernel.launch.algorithm != 'observed_moe_aux:moe_fill':
        return False
    if kernel.hardware_id not in ('metax_c500','nvidia_h200'):
        return False
    role = kernel.attributes['fill_role']
    tokens,experts,topk = (_execution_shape(kernel)[key] for key in ('tokens','experts','topk'))
    _,elements = _geometry(tokens,experts,topk,role)
    # The observation boundary already validates the exact int32 FillFunctor
    # symbol/role. Recheck the declared geometry on every public caller and
    # model/memo hit rather than accepting arbitrary observed helper launches.
    if kernel.hardware_id == 'metax_c500':
        if kernel.launch.block not in ((256,1,1),(1024,1,1)):
            raise ValueError('Observed MetaX nullary fill requires its declared int32 vec4 block')
        threads = kernel.launch.block[0]
        expected_grid = (-(-elements//(threads*4)),1,1)
    else:
        logical_tokens,logical_experts,logical_topk=(kernel.shape[key] for key in ('tokens','experts','topk'))
        if role != 'num_tokens_post_padded' or logical_tokens*logical_topk*4 > logical_experts:
            raise ValueError('H200 naive routing fill requires scalar role and source sparsity domain')
        if kernel.launch.block != (128,1,1):
            raise ValueError('H200 naive routing fill requires its declared int32 vec4 block')
        expected_grid = (1,1,1)
    if kernel.launch.grid != expected_grid:
        raise ValueError('Observed nullary fill grid does not conserve its actual output extent')
    return True


def _execution_shape(kernel):
    if 'model_input_shape' not in kernel.attributes:return kernel.shape
    # KernelSpec validates the complete captured descriptor. Its per-operator
    # token field need not equal the application's graph token count (the
    # generic Vector caller and full-MoE caller retain different work views).
    return kernel.attributes['model_input_shape']


def _geometry(tokens,experts,topk,role):
    block_m = 16 if tokens <= experts else 64
    padded = tokens*topk+experts*(block_m-1)
    if role == 'sorted_token_ids':
        elements = padded
    elif role == 'expert_ids':
        elements = (padded+block_m-1)//block_m
    elif role == 'num_tokens_post_padded':
        elements = 1
    else:
        raise ValueError('MoE fill requires an explicit routing output role')
    return block_m,elements


@dataclass(frozen=True)
class FillMemoryInvocation:
    """Parameter-only invocation; canonical records are an export projection.

    Prototype restricted to the source-owned nullary INT32 fill ABI. There
    are no tensor inputs, aliases or input-readiness dependencies to discard.
    """
    kernel: object
    binding_id: str
    call_id: str
    device_id: str
    rank: int
    stream: str

    def __post_init__(self):
        from src.kernel.kernel_spec import KernelSpec
        from src.kernel.memory_access import check_type
        if not isinstance(self.kernel,KernelSpec) or self.kernel.variant!='moe_fill':
            raise TypeError('Fill invocation requires its actual nullary KernelSpec')
        for value in (self.binding_id,self.call_id,self.device_id,self.stream):
            check_type(value,str)
        check_type(self.rank,int)
        shape=_execution_shape(self.kernel)
        _,elements=_geometry(*(shape[key] for key in ('tokens','experts','topk')),
            self.kernel.attributes['fill_role'])
        check_type(elements*4,int)
        if elements<=0:raise ValueError('Fill invocation requires a positive output extent')

    def work(self,kernel,hardware):
        if kernel is not self.kernel and kernel!=self.kernel:
            raise ValueError('Fill invocation crossed its complete source contract')
        if kernel.hardware_id!=hardware.hardware_id or not has_moe_fill_source(kernel,hardware):
            raise ValueError('Fill invocation requires its supported source hardware/launch')
        shape=_execution_shape(kernel)
        tokens,experts,topk=(shape[key] for key in ('tokens','experts','topk'))
        role=kernel.attributes['fill_role']
        _,elements=_geometry(tokens,experts,topk,role)
        logical=tuple(kernel.shape[key] for key in ('tokens','experts','topk'))
        if min(logical)<=0 or any(a>b for a,b in zip(logical,(tokens,experts,topk),strict=True)):
            raise ValueError('Nullary fill logical pricing cannot exceed its declared captured dimensions')
        _,priced_elements=_geometry(*logical,role)
        return dict(tokens=tokens,experts=experts,topk=topk,fill_role=role,elements=elements,
            write_bytes=elements*4,priced_elements=priced_elements,priced_write_bytes=priced_elements*4)

    @cached_property
    def materialized(self):
        binding=standalone_moe_fill_bindings(self.kernel,call_id=self.call_id,binding_id=self.binding_id)
        return replace(binding,rank=self.rank,stream=self.stream,
            allocations=tuple(replace(a,device_id=self.device_id) for a in binding.allocations))

    @cached_property
    def output(self):
        return _output_records(self.kernel,self.binding_id,self.device_id)

    def __getattr__(self,name):
        if name in ('allocations','views','events','read_views','write_views','input_ready','visibility_domain'):
            return getattr(self.materialized,name)
        raise AttributeError(name)


def _output_records(kernel,binding_id,device_id):
    role = kernel.attributes['fill_role']
    _,elements = _geometry(*(_execution_shape(kernel)[key] for key in ('tokens','experts','topk')),role)
    identity = f'{binding_id}.{role}'
    allocation = MemoryAllocation(identity,device_id,'logical-storage',elements*4,
        AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
    view = TensorView(role,identity,0,(elements,),(4,),'int32',4)
    return allocation,view


def standalone_moe_fill_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    role=kernel.attributes['fill_role']
    allocation,view=_output_records(kernel,binding_id,'logical-local-device')
    identity=allocation.allocation_id
    event = ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,())
    return ElementwiseBindings(binding_id,call_id,(allocation,),(view,),(event,),(),(role,),(),
        0,'offline','moe-fill-visible')


def build_moe_fill_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_moe_fill_source(kernel,hardware):
        raise NotImplementedError('Nullary fill storage ABI is not source-bound')
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('MoE fill requires its typed int32 output binding')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('unknown nullary fill logical granularity')
    role = kernel.attributes['fill_role']
    tokens,experts,topk = (_execution_shape(kernel)[key] for key in ('tokens','experts','topk'))
    block_m,elements = _geometry(tokens,experts,topk,role)
    if (bindings.read_views,bindings.write_views,bindings.input_ready) != ((),(role,),()):
        raise ValueError('Nullary MoE fill cannot consume tensor inputs')
    if len(bindings.views) != 1 or len(bindings.allocations) != 1:
        raise ValueError('MoE fill requires exactly one output object')
    view,allocation = bindings.views[0],bindings.allocations[0]
    if ((view.view_id,view.allocation_id,view.shape,view.strides_bytes,view.dtype,view.element_bytes)
            != (role,allocation.allocation_id,(elements,),(4,),'int32',4)):
        raise ValueError('MoE fill output role/extent/int32 layout mismatch')
    logical=tuple(kernel.shape[key] for key in ('tokens','experts','topk'))
    projection=() if logical == (tokens,experts,topk) else ('legacy_logical',*(str(value) for value in logical))
    scope = AccessScope(program_id,('moe_fill',role,str(tokens),str(experts),str(topk),str(block_m),*projection),
        bindings.call_id,(),bindings.rank,bindings.stream,'logical-routing-initialization')
    evidence = ('Declared int32 FillFunctor initializes each independent scalar output coordinate.',
        'C500 role extents/BM16/BM64 are launch-owner facts; H200 naive routing assigns only one torch.full INT32 scalar.',
        'No invented fill value, native issue geometry or downstream sorting alias.')
    access = TileAccessTemplate('fill',role,Operation.WRITE,('element',),('element',),(),(),evidence)
    family = TileProgram(program_id,scope,(TileAxis('element',elements,1),),(access,),bindings.visibility_domain,evidence)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Independent nullary INT32 coordinates consume no tensor read; scalar assignment is not tensor ALU work.',
             'Fill value, alignment producer coordinates, native issue order and downstream metadata identity remain unknown.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))


def moe_fill_source_demands(program):
    if not program.tile_programs or program.accesses:
        raise ValueError('Nullary MoE fill requires its unique scalar tile source')
    program.__post_init__()
    if len(program.tile_programs) != 1 or len(program.views) != 1 or len(program.allocations) != 1:
        raise ValueError('MoE fill requires exactly one typed write-only output')
    family, = program.tile_programs
    header = family.scope.op_ids
    if len(header) not in (6,10) or header[0] != 'moe_fill' or len(family.accesses) != 1:
        raise ValueError('MoE fill source geometry header mismatch')
    _,role,tokens,experts,topk,block_m = header[:6]
    tokens,experts,topk = int(tokens),int(experts),int(topk)
    expected_block,elements = _geometry(tokens,experts,topk,role)
    view,allocation,access = program.views[0],program.allocations[0],family.accesses[0]
    if (min(tokens,experts,topk) <= 0 or int(block_m) != expected_block
            or (view.view_id,view.allocation_id,view.shape,view.strides_bytes,view.dtype,view.element_bytes)
                != (role,allocation.allocation_id,(elements,),(4,),'int32',4)
            or tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != (('element',elements,1),)
            or (access.view_id,access.operation,access.iteration_axes,access.view_axes,access.dependencies,access.input_ready,
                access.unmapped,access.indirect_target,access.view_axis_divisors) != (role,Operation.WRITE,('element',),('element',),(),(),False,None,())):
        raise ValueError('MoE fill must conserve its actual nullary int32 output request')
    from src.kernel.memory_path_solver import logical_requested_bytes
    logical=(tokens,experts,topk)
    if len(header) == 10:
        if header[6] != 'legacy_logical':raise ValueError('Unknown nullary fill price projection')
        logical=tuple(map(int,header[7:]))
        if min(logical)<=0 or any(a>b for a,b in zip(logical,(tokens,experts,topk),strict=True)):
            raise ValueError('Nullary fill logical pricing cannot exceed its declared captured dimensions')
    _,priced_elements=_geometry(*logical,role)
    return {'tokens':tokens,'experts':experts,'topk':topk,'fill_role':role,'elements':elements,
        'write_bytes':logical_requested_bytes(program,Operation.WRITE),
        'priced_elements':priced_elements,'priced_write_bytes':priced_elements*4}
