"""Write a single-GPU DeepSeek-R1 model directory with the per-rank shapes of TP=N.

The proxy keeps every replicated dimension (hidden 7168, q_lora 1536, kv_lora
512, rope 64, router 7168->256, 256 routed experts, top-8, vocab) and divides
only the dimensions vLLM shards across tensor-parallel ranks: attention heads,
dense MLP intermediate and (EP1) routed/shared expert intermediate.  Run with
TP1 on one H200 it launches the same per-rank kernels as one rank of the TP=N
server, minus the TP collectives.  Evaluation input only (dummy weights).
Usage: nvidia_h200_deepseek_r1_rank_proxy_config.py <src model dir> <dst dir> <tp> <layers>
"""

import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    src, dst, tp, layers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    for key in ("num_attention_heads", "num_key_value_heads", "intermediate_size", "moe_intermediate_size"):
        if config[key] % tp:
            raise ValueError(f"{key}={config[key]} is not divisible by tp={tp}")
        config[key] //= tp
    if not 0 < layers <= config["num_hidden_layers"] or layers <= config["first_k_dense_replace"]:
        raise ValueError(f"layers={layers} must keep at least one MoE layer")
    config["num_hidden_layers"] = layers
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("configuration_deepseek.py", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(src / name, dst / name)
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(dst, {k: config[k] for k in ("num_hidden_layers", "num_attention_heads", "intermediate_size", "moe_intermediate_size")})


if __name__ == "__main__":
    main()
