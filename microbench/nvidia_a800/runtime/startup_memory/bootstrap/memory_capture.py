"""Source-bound CUDA startup observation without added resets or syncs."""

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


def file_hash(source):
    return hashlib.sha256(source.encode()).hexdigest()


def ast_hash(source):
    tree = ast.dump(ast.parse(source), include_attributes=False)
    return hashlib.sha256(tree.encode()).hexdigest()


def method_hash(source, owner, name):
    body = ast.parse(source).body
    if owner is not None:
        cls, = (
            node
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == owner
        )
        body = cls.body
    method, = (
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    value = ast.dump(method, include_attributes=False)
    return hashlib.sha256(value.encode()).hexdigest()


def unsigned_bytes(value):
    if type(value) is not int or value < 0:
        raise TypeError("Native byte count must be a nonnegative integer")
    return value


def signed_bytes(value):
    if type(value) is not int:
        raise TypeError("Native byte delta must be an integer")
    return value


def exact_number(value):
    if type(value) not in (int, float):
        raise TypeError("Native memory budget must preserve its number type")
    return {
        "value": value,
        "float_hex": value.hex() if type(value) is float else None,
    }


def snapshot(value):
    fields = (
        "torch_peak",
        "free_memory",
        "total_memory",
        "cuda_memory",
        "torch_memory",
        "non_torch_memory",
    )
    result = {field: getattr(value, field) for field in fields}
    if any(type(item) is not int for item in result.values()):
        raise TypeError("MemorySnapshot byte fields must remain integers")
    return result


def kv_binding(config):
    return {
        "num_blocks": unsigned_bytes(config.num_blocks),
        "tensors": [
            {
                "size_bytes": unsigned_bytes(tensor.size),
                "shared_by": list(tensor.shared_by),
            }
            for tensor in config.kv_cache_tensors
        ],
        "groups": [
            {
                "layer_names": list(group.layer_names),
                "spec_type": type(group.kv_cache_spec).__name__,
                "block_size_tokens": unsigned_bytes(
                    group.kv_cache_spec.block_size
                ),
                "page_size_bytes": unsigned_bytes(
                    group.kv_cache_spec.page_size_bytes
                ),
                "capacity_tokens": unsigned_bytes(
                    config.num_blocks * group.kv_cache_spec.block_size
                ),
            }
            for group in config.kv_cache_groups
        ],
    }


def actual_storage_metadata(tensors):
    """Read tensor storage metadata only; never launch a device operation."""
    unique = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        address = storage.data_ptr()
        size = unsigned_bytes(storage.nbytes())
        if address in unique and unique[address] != size:
            raise RuntimeError("One KV storage address reported inconsistent sizes")
        unique[address] = size
    return {
        "runner_tensor_count": len(tensors),
        "unique_storage_count": len(unique),
        "unique_storage_bytes": sum(unique.values()),
        "measurement": "untyped_storage_metadata_no_device_operation",
    }


class Observer:
    def __init__(self, directory, manifest, identity):
        self.directory = directory
        self.manifest = manifest
        self.identity = identity
        self.epoch = 0
        self.sequence = 0
        self.reset_installed = False
        self.profile_wrapper = None
        self.device = None

    def write(self, record):
        path = self.directory / f"memory.rank-process-{os.getpid()}.jsonl"
        with path.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def native(self, device):
        cuda = sys.modules["torch"].cuda
        stats = cuda.memory_stats(device)
        keys = (
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
        )
        values = {key: unsigned_bytes(stats[key]) for key in keys}
        free, total = cuda.mem_get_info(device)
        if not 0 <= unsigned_bytes(free) <= unsigned_bytes(total):
            raise ValueError("Native CUDA free/total bytes are inconsistent")
        values.update(free_bytes=free, total_bytes=total)
        return values

    def emit(self, phase, device, **fields):
        self.sequence += 1
        self.write(
            {
                "record_type": "native_startup_memory_bytes",
                "phase": phase,
                "pid": os.getpid(),
                "sequence": self.sequence,
                "device": str(device),
                "physical_device_index": self.identity["device_index"],
                "model": self.identity["model"],
                "config_sha256": self.identity["config_sha256"],
                "capture_scope": self.identity["capture_scope"],
                "source_bindings": self.manifest["identity"],
                "software_versions": self.manifest["software_versions"],
                "native_bytes": self.native(device),
                "counter_epoch": self.epoch,
                "peak_scope": (
                    "since_last_observed_original_runtime_reset"
                    if self.epoch
                    else "unobserved_prior_reset_epoch"
                ),
                **fields,
            }
        )

    def validate(self, module_name, source, source_file=None):
        binding = self.manifest["modules"][module_name]
        if file_hash(source) != binding["file_sha256"]:
            raise RuntimeError("Bound module file changed: " + module_name)
        if ast_hash(source) != binding["module_ast_sha256"]:
            raise RuntimeError("Bound module AST changed: " + module_name)
        if source_file is not None:
            if Path(source_file).resolve() != Path(binding["source_file"]).resolve():
                raise RuntimeError("Bound module path changed: " + module_name)
        for rule in binding["methods"]:
            if method_hash(source, rule["owner"], rule["name"]) != rule[
                "method_ast_sha256"
            ]:
                raise RuntimeError(
                    "Bound method AST changed: "
                    + module_name
                    + "."
                    + rule["name"]
                )

    def install_reset_observer(self):
        if self.reset_installed:
            raise RuntimeError("Native CUDA reset observer installed twice")
        torch = sys.modules["torch"]
        original = torch.cuda.reset_peak_memory_stats

        @functools.wraps(original)
        def reset(*args, **kwargs):
            if args:
                device = args[0]
            elif "device" in kwargs:
                device = kwargs["device"]
            else:
                device = None
            self.emit("runtime_peak_reset.before", device)
            result = original(*args, **kwargs)
            self.epoch += 1
            self.emit(
                "runtime_peak_reset.after",
                device,
                reset_owner="original_runtime_call",
            )
            return result

        torch.cuda.reset_peak_memory_stats = reset
        self.reset_installed = True

    def patch(self, module, source):
        name = module.__name__
        self.validate(name, source, module.__file__)
        self.write(
            {
                "record_type": "installed_source_binding",
                "pid": os.getpid(),
                "module": name,
                "source_file": module.__file__,
                "file_sha256": file_hash(source),
                "module_ast_sha256": ast_hash(source),
            }
        )
        if name == "vllm.utils":
            self.install_reset_observer()
            original = module.memory_profiling

            @functools.wraps(original)
            @contextmanager
            def profiling(*args, **kwargs):
                with original(*args, **kwargs) as result:
                    if not self.reset_installed:
                        raise RuntimeError("Runtime peak resets are not observed")
                    if self.device is None:
                        raise RuntimeError("CUDA device was not observed at worker init")
                    device = self.device
                    self.emit(
                        "profile.begin",
                        device,
                        before_create=snapshot(result.before_create),
                        before_profile=snapshot(result.before_profile),
                        weights_memory_bytes=unsigned_bytes(
                            result.weights_memory
                        ),
                    )
                    yield result
                self.emit(
                    "profile.end",
                    device,
                    before_create=snapshot(result.before_create),
                    before_profile=snapshot(result.before_profile),
                    after_profile=snapshot(result.after_profile),
                    weights_memory_bytes=unsigned_bytes(result.weights_memory),
                    torch_peak_increase_bytes=signed_bytes(
                        result.torch_peak_increase
                    ),
                    non_torch_increase_bytes=signed_bytes(
                        result.non_torch_increase
                    ),
                    non_kv_cache_memory_bytes=signed_bytes(
                        result.non_kv_cache_memory
                    ),
                )

            module.memory_profiling = profiling
            self.profile_wrapper = profiling
        elif name == "vllm.v1.worker.gpu_model_runner":
            runner = module.GPUModelRunner
            runner.load_model = self.wrap_runner_load(runner.load_model)
            runner.capture_model = self.wrap_method(
                runner.capture_model, "all_capture", False
            )
            runner.initialize_kv_cache = self.wrap_runner_kv(
                runner.initialize_kv_cache
            )
        elif name == "vllm.v1.worker.gpu_worker":
            if module.memory_profiling is not self.profile_wrapper:
                raise RuntimeError("GPUWorker did not import observed profiling")
            worker = module.Worker
            worker.init_device = self.wrap_method(
                worker.init_device, "worker_init", True
            )
            worker.load_model = self.wrap_method(
                worker.load_model, "worker_model_load", True
            )
            worker.determine_available_memory = self.wrap_method(
                worker.determine_available_memory, "available_KV", True
            )
            worker.initialize_from_config = self.wrap_method(
                worker.initialize_from_config, "KV_allocate", True
            )
            worker.compile_or_warm_up_model = self.wrap_method(
                worker.compile_or_warm_up_model, "startup_warmup", True
            )
        else:
            raise RuntimeError("Unknown source dependency " + name)

    def worker_fields(self, instance):
        return {
            "initial_snapshot": snapshot(instance.init_snapshot),
            "requested_memory": exact_number(instance.requested_memory),
            "configured_block_size_tokens": unsigned_bytes(
                instance.cache_config.block_size
            ),
            "max_num_batched_tokens": unsigned_bytes(
                instance.vllm_config.scheduler_config.max_num_batched_tokens
            ),
            "max_num_seqs": unsigned_bytes(
                instance.vllm_config.scheduler_config.max_num_seqs
            ),
        }

    def wrap_method(self, original, phase, worker):
        @functools.wraps(original)
        def observed(instance, *args, **kwargs):
            is_init = phase == "worker_init"
            if is_init:
                device = None
            elif worker:
                device = instance.device
            else:
                device = instance.device
            fields = {}
            if phase == "KV_allocate":
                config = args[0] if args else kwargs["kv_cache_config"]
                fields["requested_kv_binding"] = kv_binding(config)
            if not is_init:
                self.emit(phase + ".begin", device, **fields)
            result = original(instance, *args, **kwargs)
            if worker:
                device = instance.device
                if is_init:
                    self.device = device
                fields.update(self.worker_fields(instance))
                if phase == "available_KV":
                    fields.update(
                        non_torch_memory_bytes=signed_bytes(
                            instance.non_torch_memory
                        ),
                        peak_activation_memory_bytes=signed_bytes(
                            instance.peak_activation_memory
                        ),
                        available_kv_cache_memory=exact_number(
                            instance.available_kv_cache_memory_bytes
                        ),
                        available_kv_return_bytes=unsigned_bytes(result),
                    )
                if phase == "KV_allocate":
                    fields["resolved_runner_kv_binding"] = kv_binding(
                        instance.model_runner.kv_cache_config
                    )
            self.emit(phase + ".end", device, **fields)
            return result

        return observed

    def wrap_runner_load(self, original):
        @functools.wraps(original)
        def observed(instance, *args, **kwargs):
            self.emit("model_load.begin", instance.device)
            result = original(instance, *args, **kwargs)
            self.emit(
                "model_load.end",
                instance.device,
                model_memory_usage_bytes=unsigned_bytes(
                    instance.model_memory_usage
                ),
            )
            return result

        return observed

    def wrap_runner_kv(self, original):
        @functools.wraps(original)
        def observed(instance, *args, **kwargs):
            config = args[0] if args else kwargs["kv_cache_config"]
            self.emit(
                "runner_KV_initialize.begin",
                instance.device,
                requested_kv_binding=kv_binding(config),
            )
            result = original(instance, *args, **kwargs)
            self.emit(
                "runner_KV_initialize.end",
                instance.device,
                resolved_runner_kv_binding=kv_binding(
                    instance.kv_cache_config
                ),
                actual_storage=actual_storage_metadata(instance.kv_caches),
            )
            return result

        return observed


def install():
    directory = Path(os.environ["MIDLAYER_STARTUP_MEMORY_RAW"]).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    manifest = json.loads(
        Path(os.environ["MIDLAYER_STARTUP_MEMORY_BINDINGS"]).read_text()
    )
    if list(sys.version_info[:2]) != manifest["binding_ast_python"]:
        raise RuntimeError("Source binding AST Python version changed")
    for name, version in manifest["software_versions"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("Software version changed: " + name)
    identity = json.loads(
        Path(os.environ["MIDLAYER_STARTUP_MEMORY_IDENTITY"]).read_text()
    )
    observer = Observer(directory, manifest, identity)

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            source = self.original.get_source(module.__name__)
            observer.validate(module.__name__, source, self.original.get_filename(module.__name__))
            self.original.exec_module(module)
            observer.patch(module, source)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in manifest["modules"]:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(name)
            spec.loader = Loader(spec.loader)
            return spec

    if any(name in sys.modules for name in manifest["modules"]):
        raise RuntimeError("Observer installed after a bound module imported")
    sys.meta_path.insert(0, Finder())
