"""Source-bound native startup observation; never adds a counter reset or sync."""
import ast
from contextlib import contextmanager
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def ast_hash(source):
    return hashlib.sha256(ast.dump(ast.parse(source),include_attributes=False).encode()).hexdigest()


def method_hash(source,owner,name):
    body=ast.parse(source).body
    if owner is not None:
        cls,=(n for n in body if isinstance(n,ast.ClassDef)and n.name==owner)
        body=cls.body
    node,=(n for n in body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==name)
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def integer_bytes(value):
    if type(value)is not int or value<0:raise TypeError('Native byte count must be a nonnegative integer')
    return value


def number(value):
    if type(value)not in(int,float):raise TypeError('Native memory budget must preserve its original number type')
    return {'value':value,'float_hex':value.hex()if type(value)is float else None}


def snapshot(value):
    fields=('torch_peak','free_memory','total_memory','cuda_memory','torch_memory','non_torch_memory')
    values={field:getattr(value,field)for field in fields}
    if any(type(value)is not int for value in values.values()):raise TypeError('Original snapshot must preserve integer byte fields')
    return values


def kv_binding(config):
    return {'num_blocks':integer_bytes(config.num_blocks),
        'tensors':[{'size_bytes':integer_bytes(t.size),'shared_by':list(t.shared_by)}for t in config.kv_cache_tensors],
        'groups':[{'layer_names':list(g.layer_names),'spec_type':type(g.kv_cache_spec).__name__,
            'block_size_tokens':integer_bytes(g.kv_cache_spec.block_size),
            'page_size_bytes':integer_bytes(g.kv_cache_spec.page_size_bytes),
            'capacity_tokens':integer_bytes(config.num_blocks*g.kv_cache_spec.block_size)}for g in config.kv_cache_groups]}


class Observer:
    def __init__(self,directory,manifest,identity):
        self.directory,self.manifest,self.identity=directory,manifest,identity
        self.epoch=0
        self.sequence=0
        self.reset_installed=False

    def write(self,record):
        path=self.directory/f'memory.rank-process-{os.getpid()}.jsonl'
        with path.open('a',encoding='utf8')as stream:stream.write(json.dumps(record,sort_keys=True)+'\n')

    def native(self,device):
        npu=sys.modules['torch'].npu
        stats=npu.memory_stats(device)
        keys=('allocated_bytes.all.current','allocated_bytes.all.peak','reserved_bytes.all.current','reserved_bytes.all.peak')
        values={key:integer_bytes(stats[key])for key in keys}
        free,total=npu.mem_get_info(device)
        if not 0<=integer_bytes(free)<=integer_bytes(total):raise ValueError('Native free/total bytes are inconsistent')
        values.update(free_bytes=free,total_bytes=total)
        return values

    def emit(self,phase,device,**fields):
        self.sequence+=1
        self.write({'record_type':'native_startup_memory_bytes','phase':phase,'pid':os.getpid(),
            'sequence':self.sequence,'device':str(device),'physical_device_index':self.identity['device_index'],
            'model':self.identity['model'],'config_sha256':self.identity['config_sha256'],
            'source_bindings':self.manifest['identity'],'software_versions':self.manifest['software_versions'],
            'native_bytes':self.native(device),'counter_epoch':self.epoch,
            'peak_scope':'since_last_observed_original_runtime_reset'if self.epoch else 'unobserved_prior_reset_epoch',
            **fields})

    def validate(self,module_name,source):
        binding=self.manifest['modules'][module_name]
        if ast_hash(source)!=binding['module_ast_sha256']:raise RuntimeError('Bound module AST changed: '+module_name)
        for rule in binding['methods']:
            if method_hash(source,rule['owner'],rule['name'])!=rule['method_ast_sha256']:
                raise RuntimeError('Bound method AST changed: '+module_name+'.'+rule['name'])

    def install_reset_observer(self):
        if self.reset_installed:raise RuntimeError('Native reset observer installed twice')
        torch=sys.modules['torch'];original=torch.npu.reset_peak_memory_stats
        if torch.accelerator.reset_peak_memory_stats is not original:
            raise RuntimeError('Expected source-bound accelerator→NPU reset alias')
        @functools.wraps(original)
        def reset(*args,**kwargs):
            device=args[0]if args else kwargs['device']if 'device'in kwargs else None
            self.emit('runtime_peak_reset.before',device)
            result=original(*args,**kwargs)
            self.epoch+=1
            self.emit('runtime_peak_reset.after',device,reset_owner='original_runtime_call')
            return result
        torch.npu.reset_peak_memory_stats=reset
        torch.accelerator.reset_peak_memory_stats=reset
        self.reset_installed=True

    def patch(self,module,source):
        name=module.__name__;self.validate(name,source)
        self.write({'record_type':'installed_source_binding','pid':os.getpid(),'module':name,
            'source_file':module.__file__,'module_ast_sha256':ast_hash(source)})
        if name=='vllm_ascend.patch.platform.patch_torch_accelerator':
            self.install_reset_observer()
        elif name=='vllm.utils.mem_utils':
            original=module.memory_profiling
            @functools.wraps(original)
            @contextmanager
            def profiling(*args,**kwargs):
                with original(*args,**kwargs)as result:
                    if not self.reset_installed:raise RuntimeError('Runtime peak reset calls are not observed')
                    device=result.before_profile.device_
                    self.emit('profile.begin',device,before_create_snapshot=snapshot(result.before_create),
                        before_profile_snapshot=snapshot(result.before_profile),weights_memory_bytes=integer_bytes(result.weights_memory))
                    yield result
                self.emit('profile.end',device,after_profile_snapshot=snapshot(result.after_profile),
                    torch_peak_increase_bytes=number(result.torch_peak_increase),
                    non_torch_increase_bytes=number(result.non_torch_increase),
                    weights_memory_bytes=integer_bytes(result.weights_memory),non_KV_cache_memory_bytes=number(result.non_kv_cache_memory))
            module.memory_profiling=profiling
        elif name=='vllm_ascend.worker.model_runner_v1':
            cls=module.NPUModelRunner
            for method,phase in(('load_model','model_load'),('capture_model','all_capture')):
                setattr(cls,method,self.wrap(getattr(cls,method),phase,False))
        elif name=='vllm_ascend.worker.worker':
            cls=module.NPUWorker
            for method,phase in(('init_device','worker_init'),('determine_available_memory','available_KV'),('initialize_from_config','KV_allocate'),('compile_or_warm_up_model','startup_warmup')):
                setattr(cls,method,self.wrap(getattr(cls,method),phase,True))
        elif name not in('vllm_ascend.platform','vllm.v1.worker.gpu_model_runner','vllm_ascend.utils','vllm.v1.kv_cache_interface'):
            raise RuntimeError('Unknown source dependency '+name)

    def wrap(self,original,phase,worker):
        @functools.wraps(original)
        def observed(instance,*args,**kwargs):
            device=instance.init_snapshot.device_ if worker and phase!='worker_init'else None if phase=='worker_init'else instance.device
            fields={}
            if phase=='KV_allocate':
                config=args[0]if args else kwargs['kv_cache_config']
                fields['requested_KV_binding']=kv_binding(config)
            if phase!='worker_init':self.emit(phase+'.begin',device,**fields)
            result=original(instance,*args,**kwargs)
            if not worker and phase=='model_load':fields['model_memory_usage']=number(instance.model_memory_usage)
            if worker:
                device=instance.init_snapshot.device_
                fields.update(initial_snapshot=snapshot(instance.init_snapshot),requested_memory=number(instance.requested_memory),
                    resolved_cache_block_size_tokens=integer_bytes(instance.cache_config.block_size),
                    max_num_batched_tokens=integer_bytes(instance.vllm_config.scheduler_config.max_num_batched_tokens),
                    max_num_seqs=integer_bytes(instance.vllm_config.scheduler_config.max_num_seqs))
                if phase=='available_KV':fields.update(available_KV_native=number(instance.available_kv_cache_memory_bytes),available_KV_return_bytes=integer_bytes(result))
                if phase=='KV_allocate':
                    fields['actual_runner_KV_binding']=kv_binding(instance.model_runner.kv_cache_config)
                    fields['runner_block_size_tokens']=integer_bytes(instance.model_runner.block_size)
                    fields['runner_kernel_block_sizes']=instance.model_runner.kernel_block_sizes
            self.emit(phase+'.end',device,**fields)
            return result
        return observed


def install():
    directory=Path(os.environ['MIDLAYER_STARTUP_MEMORY_RAW']).resolve()
    if not directory.is_dir():raise FileNotFoundError(directory)
    manifest=json.loads(Path(os.environ['MIDLAYER_STARTUP_MEMORY_BINDINGS']).read_text())
    if list(sys.version_info[:2])!=manifest['binding_ast_python']:raise RuntimeError('Source binding AST Python version changed')
    for name,version in manifest['software_versions'].items():
        if importlib.metadata.version(name)!=version:raise RuntimeError('Software version changed: '+name)
    identity=json.loads(Path(os.environ['MIDLAYER_STARTUP_MEMORY_IDENTITY']).read_text())
    observer=Observer(directory,manifest,identity)
    class Loader(importlib.abc.Loader):
        def __init__(self,original):self.original=original
        def create_module(self,spec):return self.original.create_module(spec)
        def exec_module(self,module):
            source=self.original.get_source(module.__name__);observer.validate(module.__name__,source)
            self.original.exec_module(module);observer.patch(module,source)
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,name,path,target=None):
            if name not in manifest['modules']:return None
            spec=importlib.machinery.PathFinder.find_spec(name,path)
            if spec is None or spec.loader is None:raise ImportError(name)
            spec.loader=Loader(spec.loader);return spec
    if any(name in sys.modules for name in manifest['modules']):raise RuntimeError('Observer bootstrap installed after a bound module imported')
    sys.meta_path.insert(0,Finder())
