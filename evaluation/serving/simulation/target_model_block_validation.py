"""Source-pinned tiny GLM-5.2 and Kimi-K2.6 CUDA block validation."""

from __future__ import annotations

from evaluation.json_shape import file_sha256

import os
from pathlib import Path
import subprocess
import sys

import torch
import transformers
from transformers import (
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    GlmMoeDsaConfig,
    GlmMoeDsaForCausalLM,
)


_EXPECTED_TRANSFORMERS_COMMIT = (
    "0dad7b822255a0ae261ec45ae937371e859ffd1a")
_EXPECTED_VLLM_COMMIT = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
_EXPECTED_MODEL_COMMITS = {
    "glm_5_2": "b4734de4facf877f85769a911abafc5283eab3d9",
    "kimi_k2_6": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
}
_EXPECTED_SOURCE_HASHES = {
    "transformers": {
        "src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py": (
            "3d72daa66b8a1bd703833b9b85a2b76e14fbd5158fe1238866d36e40b60225cc"
        ),
        "src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py": (
            "0560406b772637b451cedfaccbb1d5a6c73fd002a0ae05807e95af89ced44da7"
        ),
        "src/transformers/models/deepseek_v3/configuration_deepseek_v3.py": (
            "f4a8b645b0742cf37f6f86e7339cd9bcadb923775ed0274b787bdd0b8f531451"
        ),
        "src/transformers/models/deepseek_v3/modeling_deepseek_v3.py": (
            "97c3686994c4e054e19f3cb2a88fc6929b51c7f52d93f0bffaaad8938d916462"
        ),
    },
    "vllm": {
        "vllm/model_executor/models/registry.py": (
            "f1dae30a951fda360151c8de9dd605d01c7138a69f70fe518286d57da69dc463"
        ),
        "vllm/model_executor/models/deepseek_v2.py": (
            "bcbd9cde689ffe3f5f271aff3a5e01351b05a68371192cac9d65cb5ffc6b4ba9"
        ),
        "vllm/model_executor/layers/mla.py": (
            "ccdbafa998b1a7f7db47527df85efdfa4bc3027f6ec769fd4c30b3aee963a81b"
        ),
        "vllm/model_executor/layers/attention/mla_attention.py": (
            "272c01ff03e52cdfaed6fc7bc2e3e5f641d35910ea0bf5365b928c9401c4df5f"
        ),
        "vllm/model_executor/layers/attention/sparse_mla_attention.py": (
            "1aa19b3dc2d74acd73d77a3d6b04c98b5e36585907308986098aff120a82d1cd"
        ),
        "vllm/model_executor/layers/sparse_attn_indexer.py": (
            "a6386326379dcf94439a0b020cd51a71ebfd0a36eaeb563417c821456b82a475"
        ),
        "vllm/v1/attention/backends/mla/flashmla_sparse.py": (
            "0544c431bbf311d7283fee7f34cd7ad8a2df7961380646f2c12db2e861608ee4"
        ),
    },
}
_EXPECTED_MODEL_CONFIG_HASHES = {
    "glm_5_2": "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a",
    "kimi_k2_6": "85825ca6e18cbe539eb83ee09eedfb3f4222265929f06e9f535a6d9364f55899",
}


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        text=True,
    ).strip()


def _verify_checkout(
    name: str,
    root: Path,
    expected_commit: str,
    expected_hashes: dict[str, str],
) -> dict[str, object]:
    commit = _git_commit(root)
    if commit != expected_commit:
        raise RuntimeError(
            f"{name} checkout commit changed: expected {expected_commit}, "
            f"got {commit}")
    files = []
    for relative_path, expected_hash in expected_hashes.items():
        actual_hash = _sha256(root / relative_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{name} source changed at {relative_path}: expected "
                f"{expected_hash}, got {actual_hash}")
        files.append({
            "relative_path": relative_path,
            "sha256": actual_hash,
        })
    return {
        "root": str(root.resolve()),
        "commit": commit,
        "files": files,
    }


def _verify_installed_transformers() -> dict[str, object]:
    package_root = Path(transformers.__file__).resolve().parent
    files = []
    prefix = "src/transformers/"
    for relative_path, expected_hash in _EXPECTED_SOURCE_HASHES[
            "transformers"].items():
        if not relative_path.startswith(prefix):
            raise AssertionError(
                f"unexpected Transformers source path: {relative_path}")
        installed_path = package_root / relative_path[len(prefix):]
        actual_hash = _sha256(installed_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"installed Transformers source changed at {installed_path}: "
                f"expected {expected_hash}, got {actual_hash}")
        files.append({
            "relative_path": relative_path,
            "sha256": actual_hash,
        })
    return {
        "version": transformers.__version__,
        "package_root": str(package_root),
        "files": files,
    }


def _verify_model_checkout(
    name: str,
    root: Path,
) -> dict[str, object]:
    commit = _git_commit(root)
    expected_commit = _EXPECTED_MODEL_COMMITS[name]
    if commit != expected_commit:
        raise RuntimeError(
            f"{name} checkout commit changed: expected {expected_commit}, "
            f"got {commit}")
    config_hash = _sha256(root / "config.json")
    expected_hash = _EXPECTED_MODEL_CONFIG_HASHES[name]
    if config_hash != expected_hash:
        raise RuntimeError(
            f"{name} config changed: expected {expected_hash}, "
            f"got {config_hash}")
    return {
        "root": str(root.resolve()),
        "commit": commit,
        "config_sha256": config_hash,
    }


def _tensor_summary(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (tuple, list)):
        return [_tensor_summary(item) for item in value]
    return None


def _execute(
    name: str,
    config: object,
    model_class: type,
    hook_names: tuple[str, ...],
) -> dict[str, object]:
    torch.manual_seed(20260828)
    model = model_class(config).to(
        device="cuda:0", dtype=torch.bfloat16).eval()
    events = []
    handles = []

    def hook(module_name: str):
        def record(module, args, output):
            events.append({
                "module": module_name,
                "class": module.__class__.__name__,
                "inputs": _tensor_summary(args),
                "output": _tensor_summary(output),
            })
        return record

    modules = dict(model.named_modules())
    for module_name in hook_names:
        if module_name not in modules:
            raise RuntimeError(f"missing hook module {name}:{module_name}")
        handles.append(
            modules[module_name].register_forward_hook(hook(module_name)))

    with torch.inference_mode():
        prefill = model(
            input_ids=torch.tensor([[1, 3, 5]], device="cuda:0"),
            use_cache=True,
        )
        prefill_event_count = len(events)
        decode = model(
            input_ids=torch.tensor([[7]], device="cuda:0"),
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
    for handle in handles:
        handle.remove()

    cache = decode.past_key_values
    result = {
        "model": name,
        "prefill_logits": _tensor_summary(prefill.logits),
        "decode_logits": _tensor_summary(decode.logits),
        "cache_class": cache.__class__.__name__,
        "cache_sequence_length": cache.get_seq_length(),
        "cache_layer_0_key_shape": list(cache.layers[0].keys.shape),
        "prefill_events": events[:prefill_event_count],
        "decode_events": events[prefill_event_count:],
    }
    if name == "glm_5_2":
        result["indexer_modes"] = [
            "shared" if layer.self_attn.skip_topk else "full"
            for layer in model.model.layers
        ]
        result["full_indexer_cache_shape"] = list(
            model.model.layers[0].self_attn.indexer._cached_keys.shape)
        result["shared_indexer_cache_is_empty"] = (
            model.model.layers[1].self_attn.indexer._cached_keys is None)
    del model, prefill, decode
    torch.cuda.empty_cache()
    return result


def _tiny_configs() -> tuple[GlmMoeDsaConfig, DeepseekV3Config]:
    common = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "n_shared_experts": 1,
        "n_routed_experts": 4,
        "routed_scaling_factor": 2.5,
        "kv_lora_rank": 8,
        "q_lora_rank": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "qk_nope_head_dim": 8,
        "n_group": 1,
        "topk_group": 1,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "dtype": "bfloat16",
        "use_cache": True,
    }
    glm = GlmMoeDsaConfig(
        **common,
        max_position_embeddings=128,
        mlp_layer_types=["dense", "sparse"],
        index_topk=4,
        index_head_dim=8,
        index_n_heads=2,
        indexer_types=["full", "shared"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000},
    )
    kimi = DeepseekV3Config(
        **common,
        max_position_embeddings=1024,
        first_k_dense_replace=1,
        rope_interleave=True,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 50000,
            "factor": 64,
            "original_max_position_embeddings": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1,
            "mscale_all_dim": 1,
        },
    )
    return glm, kimi


def run_target_model_block_validation(
    *,
    vllm_root: Path,
    transformers_root: Path,
    glm_root: Path,
    kimi_root: Path,
) -> dict[str, object]:
    """Validate fixed upstream sources and execute both tiny CUDA models."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if transformers.__version__ != "5.10.2":
        raise RuntimeError(
            "target block validation requires Transformers 5.10.2, got "
            f"{transformers.__version__}")
    repository_root = Path(__file__).resolve().parents[3]
    baseline = _git_commit(repository_root)
    glm_config, kimi_config = _tiny_configs()
    glm_hooks = (
        "model.layers.0.self_attn.q_a_proj",
        "model.layers.0.self_attn.q_b_proj",
        "model.layers.0.self_attn.kv_a_proj_with_mqa",
        "model.layers.0.self_attn.kv_b_proj",
        "model.layers.0.self_attn.indexer.wq_b",
        "model.layers.0.self_attn.indexer.wk",
        "model.layers.0.self_attn.indexer.weights_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp",
        "model.layers.1.self_attn",
        "model.layers.1.mlp.gate",
        "model.layers.1.mlp.experts",
        "model.layers.1.mlp.shared_experts",
    )
    kimi_hooks = (
        "model.layers.0.self_attn.q_a_proj",
        "model.layers.0.self_attn.q_b_proj",
        "model.layers.0.self_attn.kv_a_proj_with_mqa",
        "model.layers.0.self_attn.kv_b_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp",
        "model.layers.1.self_attn",
        "model.layers.1.mlp.gate",
        "model.layers.1.mlp.experts",
        "model.layers.1.mlp.shared_experts",
    )
    result = {
        "schema_version": 1,
        "pid": os.getpid(),
        "midlayer_baseline": baseline,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": _verify_installed_transformers(),
        "device": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "source_checkouts": {
            "transformers": _verify_checkout(
                "transformers",
                transformers_root,
                _EXPECTED_TRANSFORMERS_COMMIT,
                _EXPECTED_SOURCE_HASHES["transformers"],
            ),
            "vllm": _verify_checkout(
                "vllm",
                vllm_root,
                _EXPECTED_VLLM_COMMIT,
                _EXPECTED_SOURCE_HASHES["vllm"],
            ),
            "glm_5_2": _verify_model_checkout("glm_5_2", glm_root),
            "kimi_k2_6": _verify_model_checkout("kimi_k2_6", kimi_root),
        },
        "scope": {
            "cuda_execution": "tiny_transformers_control_flow",
            "production_shapes": "validated_separately_from_fixed_source",
            "vllm_execution": False,
            "vllm_blocker": (
                "vLLM 0.27.1 requires PyTorch 2.13.0; the isolated CUDA "
                "environment has PyTorch 2.7.1+cu126"
            ),
            "transformers_cache_difference": (
                "Transformers expands K/V before DynamicCache; fixed vLLM "
                "source defines compressed MLA and a separate GLM index cache"
            ),
        },
        "glm_5_2": _execute(
            "glm_5_2", glm_config, GlmMoeDsaForCausalLM, glm_hooks),
        "kimi_k2_6": _execute(
            "kimi_k2_6", kimi_config, DeepseekV3ForCausalLM, kimi_hooks),
    }
    return result
