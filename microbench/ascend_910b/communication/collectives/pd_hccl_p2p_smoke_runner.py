#!/usr/bin/env python3
"""Verify one direct device-resident HCCL transfer used by 1P1D Serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset.ascend_p2p_hccl.p2p_hccl_engine import P2pHcclEngine


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be in [1, 65535]")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=("producer", "consumer"))
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--local-port", required=True, type=_port)
    parser.add_argument("--peer-ip", required=True)
    parser.add_argument("--peer-port", required=True, type=_port)
    parser.add_argument("--result", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.result.exists():
        raise FileExistsError(arguments.result)

    import torch
    import torch_npu
    from vllm.config.kv_transfer import KVTransferConfig

    torch.npu.set_device(0)
    kv_roles = {"producer": "kv_producer", "consumer": "kv_consumer"}
    kv_ranks = {"producer": 0, "consumer": 1}
    config = KVTransferConfig(
        kv_connector="P2pHcclConnector",
        kv_role=kv_roles[arguments.role],
        kv_rank=kv_ranks[arguments.role],
        kv_parallel_size=2,
        kv_ip=arguments.local_ip,
        kv_port=arguments.local_port,
        kv_connector_extra_config={"send_type": "PUT", "pg_timeout_s": 30},
    )
    engine = P2pHcclEngine(0, config, port_offset=0)
    tensor_id = "pd-hccl-smoke"
    peer = f"{arguments.peer_ip}:{arguments.peer_port}"
    if arguments.role == "producer":
        payload = torch.arange(1024, device="npu:0", dtype=torch.float32)
        engine.send_tensor(tensor_id, payload, peer)
        expected_sum = float(payload.sum().item())
    elif arguments.role == "consumer":
        payload = engine.recv_tensor(tensor_id, peer)
        expected_sum = float(torch.arange(1024, dtype=torch.float32).sum().item())
        actual_sum = float(payload.sum().item())
        if actual_sum != expected_sum:
            raise RuntimeError(
                f"HCCL payload mismatch: expected sum {expected_sum}, got {actual_sum}"
            )
    else:
        raise AssertionError("HCCL smoke role choices are exhaustive")
    torch.npu.synchronize()
    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    arguments.result.write_text(json.dumps({
        "role": arguments.role,
        "tensor_device": str(payload.device),
        "tensor_shape": list(payload.shape),
        "tensor_dtype": str(payload.dtype),
        "payload_sum": expected_sum,
        "validation": "passed",
    }, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
