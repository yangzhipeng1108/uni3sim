#!/usr/bin/env python3
"""Verify one Ascend Mooncake NPU-to-NPU KV-style transfer without host staging."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time
from typing import Any


def _host() -> str:
    value = socket.gethostbyname(socket.gethostname())
    if not value:
        raise RuntimeError("Mooncake probe resolved an empty host address")
    return value


def _worker(role: str, device: int, size: int, connection: Any) -> None:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    try:
        import torch
        import torch_npu  # noqa: F401
        from mooncake.engine import TransferEngine

        torch.npu.set_device(0)
        values = torch.arange(size, dtype=torch.uint8)
        if role == "sender":
            tensor = values.npu()
        elif role == "receiver":
            tensor = torch.zeros_like(values).npu()
        else:
            raise AssertionError("Mooncake transfer role is exhaustive")
        torch.npu.synchronize()
        if tensor.data_ptr() % (2 * 1024 * 1024) != 0:
            raise RuntimeError("Mooncake Ascend transport requires a 2 MiB-aligned NPU buffer")
        engine = TransferEngine()
        host = _host()
        status = engine.initialize(f"{host}:0:npu_{device}", "P2PHANDSHAKE", "rdma", "")
        if status != 0:
            raise RuntimeError(f"Mooncake initialize returned {status}")
        status = engine.register_memory(tensor.data_ptr(), tensor.nbytes)
        if status != 0:
            raise RuntimeError(f"Mooncake register_memory returned {status}")
        endpoint = f"{host}:{engine.get_rpc_port()}"
        connection.send({"kind": "ready", "role": role, "endpoint": endpoint, "address": tensor.data_ptr(), "bytes": tensor.nbytes})
        command = connection.recv()
        if role == "sender":
            if not isinstance(command, dict) or set(command) != {"kind", "endpoint", "address"}:
                raise RuntimeError("sender received an invalid transfer command")
            if command["kind"] != "transfer" or not isinstance(command["endpoint"], str) or not isinstance(command["address"], int):
                raise RuntimeError("sender received an invalid transfer command")
            started_ns = time.perf_counter_ns()
            status = engine.batch_transfer_sync_write(command["endpoint"], [tensor.data_ptr()], [command["address"]], [tensor.nbytes])
            connection.send({"kind": "transfer", "status": status, "elapsed_ns": time.perf_counter_ns() - started_ns})
        elif role == "receiver":
            if command != {"kind": "verify"}:
                raise RuntimeError("receiver received an invalid verification command")
            torch.npu.synchronize()
            expected = torch.arange(size, dtype=torch.uint8)
            actual = tensor.cpu()
            connection.send({"kind": "verify", "equal": bool(torch.equal(actual, expected)), "actual_sum": int(actual.sum().item()), "expected_sum": int(expected.sum().item())})
        else:
            raise AssertionError("role is exhaustive")
        status = engine.unregister_memory(tensor.data_ptr())
        if status != 0:
            raise RuntimeError(f"Mooncake unregister_memory returned {status}")
    except BaseException as error:
        connection.send({"kind": "error", "role": role, "type": type(error).__name__, "message": str(error)})
        raise
    finally:
        connection.close()


def _receive(connection: Any, label: str) -> dict[str, object]:
    if not connection.poll(90):
        raise TimeoutError(f"timed out waiting for {label}")
    result = connection.recv()
    if not isinstance(result, dict) or "kind" not in result or not isinstance(result["kind"], str):
        raise RuntimeError(f"{label} returned invalid data")
    if result["kind"] == "error":
        raise RuntimeError(f"{label} failed: {result}")
    return result


def run_probe(size: int) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    receiver_parent, receiver_child = context.Pipe()
    sender_parent, sender_child = context.Pipe()
    receiver = context.Process(target=_worker, args=("receiver", 1, size, receiver_child))
    sender = context.Process(target=_worker, args=("sender", 0, size, sender_child))
    receiver.start()
    sender.start()
    receiver_ready = _receive(receiver_parent, "receiver readiness")
    sender_ready = _receive(sender_parent, "sender readiness")
    sender_parent.send({"kind": "transfer", "endpoint": receiver_ready["endpoint"], "address": receiver_ready["address"]})
    transfer = _receive(sender_parent, "transfer")
    receiver_parent.send({"kind": "verify"})
    verification = _receive(receiver_parent, "verification")
    sender.join(30)
    receiver.join(30)
    if sender.is_alive() or receiver.is_alive():
        raise TimeoutError("Mooncake probe worker did not exit")
    if sender.exitcode != 0 or receiver.exitcode != 0:
        raise RuntimeError(f"Mooncake worker exits sender={sender.exitcode}, receiver={receiver.exitcode}")
    if transfer["status"] != 0 or verification["equal"] is not True:
        raise RuntimeError(f"Mooncake transfer did not verify: {transfer}, {verification}")
    return {"schema_version": 1, "kind": "ascend_910b_mooncake_device_transfer_probe", "memory_kind": "npu", "bytes": size, "sender": sender_ready, "receiver": receiver_ready, "transfer": transfer, "verification": verification}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bytes < 2 * 1024 * 1024 or args.bytes % (2 * 1024 * 1024) != 0:
        raise ValueError("--bytes must be a positive multiple of 2 MiB")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    result = run_probe(args.bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
