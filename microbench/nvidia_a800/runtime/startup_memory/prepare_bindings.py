"""Bind the installed vLLM 0.11 CUDA startup sources without importing them."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "bootstrap"))
from memory_capture import ast_hash, file_hash, method_hash


TARGETS = {
    "vllm.utils": [
        (None, "memory_profiling"),
        ("MemorySnapshot", "measure"),
    ],
    "vllm.v1.worker.gpu_worker": [
        ("Worker", "init_device"),
        ("Worker", "load_model"),
        ("Worker", "determine_available_memory"),
        ("Worker", "initialize_from_config"),
        ("Worker", "compile_or_warm_up_model"),
    ],
    "vllm.v1.worker.gpu_model_runner": [
        ("GPUModelRunner", "load_model"),
        ("GPUModelRunner", "capture_model"),
        ("GPUModelRunner", "initialize_kv_cache"),
    ],
}


def source_file(site_packages, module_name):
    parts = module_name.split(".")
    candidate = site_packages.joinpath(*parts)
    package = candidate / "__init__.py"
    module = candidate.with_suffix(".py")
    if package.is_file():
        return package
    if module.is_file():
        return module
    raise FileNotFoundError(module_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "bindings.json")
    args = parser.parse_args()
    manifest = {
        "identity": "nvidia-a800-vllm-0.11.0-startup-native-bytes-reset-epochs-v1",
        "binding_ast_python": list(sys.version_info[:2]),
        "software_versions": {
            name: importlib.metadata.version(name) for name in ("vllm", "torch")
        },
        "modules": {},
    }
    if manifest["software_versions"]["vllm"] != "0.11.0":
        raise RuntimeError("Only the bound vLLM 0.11.0 source is supported")
    for name, methods in TARGETS.items():
        path = source_file(args.site_packages.resolve(), name)
        source = path.read_text()
        manifest["modules"][name] = {
            "source_file": str(path.resolve()),
            "file_sha256": file_hash(source),
            "module_ast_sha256": ast_hash(source),
            "methods": [
                {
                    "owner": owner,
                    "name": method,
                    "method_ast_sha256": method_hash(
                        source, owner, method
                    ),
                }
                for owner, method in methods
            ],
        }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "modules": len(TARGETS),
                "bound_methods": sum(map(len, TARGETS.values())),
                "device_imports": [
                    name for name in ("torch", "vllm") if name in sys.modules
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
