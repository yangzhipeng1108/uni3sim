"""Generate v9-compatible workload artifacts for the Token Factory experiments.

The experiment_plan workload schema requires:

    version, workload{name,prompt_len,output_len}, request_ids, prompts,
    prompt_lengths, sampling, token_controls, num_requests,
    requested_num_requests, seed, hash

where ``hash`` is the canonical digest the plan recomputes.  This script emits
exactly that shape for a fixed synthetic length so the manifests load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

EXPDIR = os.path.dirname(os.path.abspath(__file__))


def canonical_hash(artifact_without_hash: dict) -> str:
    payload = json.dumps(
        artifact_without_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build(prompt_len: int, output_len: int, num_requests: int, seed: int,
          name: str) -> dict:
    prompts = ["tokenfactory-" + "x" * min(prompt_len, 16)
               for _ in range(num_requests)]
    artifact = {
        "version": 1,
        "workload": {
            "name": name,
            "prompt_len": prompt_len,
            "output_len": output_len,
        },
        "request_ids": [f"req-{index:05d}" for index in range(num_requests)],
        "prompts": prompts,
        "prompt_lengths": [prompt_len] * num_requests,
        "sampling": {
            "ignore_eos": True,
            "max_tokens": output_len,
            "min_tokens": output_len,
            "temperature": 0.0,
        },
        "token_controls": {
            "ignore_eos": True,
            "max_tokens": output_len,
            "min_tokens": output_len,
        },
        "num_requests": num_requests,
        "requested_num_requests": num_requests,
        "seed": seed,
    }
    artifact["hash"] = canonical_hash(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-len", type=int, default=77000)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="tf-synthetic")
    parser.add_argument("--out", default="nvidia_h20_pd_tf03_workload.json")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    artifact = build(args.prompt_len, args.output_len, args.num_requests,
                     args.seed, args.name)
    print(json.dumps(artifact, ensure_ascii=False, indent=2)[:800])
    if args.write:
        out = os.path.join(EXPDIR, args.out)
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print("\nwritten:", out)


if __name__ == "__main__":
    main()
