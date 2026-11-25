#!/usr/bin/env python3
"""Run one isolated same-host Mooncake transfer on two MetaX C500 devices."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time
from typing import Any


def _endpoint_host() -> str:
    host = socket.gethostbyname(socket.gethostname())
    if not host:
        raise RuntimeError("Mooncake probe resolved an empty host address")
    return host


def _buffer(torch: Any, memory_kind: str, size: int, source: bool) -> Any:
    values = torch.arange(size, dtype=torch.uint8) % 251 if source else torch.zeros(size, dtype=torch.uint8)
    if memory_kind == "device":
        return values.to("cuda:0")
    if memory_kind == "host":
        return values.pin_memory()
    raise AssertionError("memory kind was validated")


def _worker(
    role: str,
    physical_device: int,
    protocol: str,
    memory_kind: str,
    size: int,
    connection: Any,
) -> None:
    os.environ["MACA_VISIBLE_DEVICES"] = str(physical_device)
    try:
        import torch
        from mooncake.engine import TransferEngine

        if memory_kind == "device":
            torch.cuda.set_device(0)
        tensor = _buffer(torch, memory_kind, size, role == "sender")
        if memory_kind == "device":
            torch.cuda.synchronize()
        engine = TransferEngine()
        host = _endpoint_host()
        initialize_status = engine.initialize(host, "P2PHANDSHAKE", protocol, "")
        if initialize_status != 0:
            raise RuntimeError(f"Mooncake initialize returned {initialize_status}")
        register_status = engine.register_memory(tensor.data_ptr(), tensor.nbytes)
        if register_status != 0:
            raise RuntimeError(f"Mooncake register_memory returned {register_status}")
        connection.send({
            "kind": "ready",
            "role": role,
            "endpoint": f"{host}:{engine.get_rpc_port()}",
            "address": tensor.data_ptr(),
            "bytes": tensor.nbytes,
        })
        command = connection.recv()
        if not isinstance(command, dict) or "kind" not in command:
            raise RuntimeError("Mooncake probe received an invalid command")
        if role == "sender":
            if command["kind"] != "transfer":
                raise RuntimeError("sender did not receive transfer command")
            started_ns = time.perf_counter_ns()
            transfer_status = engine.batch_transfer_sync_write(
                command["endpoint"],
                [tensor.data_ptr()],
                [command["address"]],
                [tensor.nbytes],
            )
            elapsed_ns = time.perf_counter_ns() - started_ns
            connection.send({
                "kind": "transfer_result",
                "status": transfer_status,
                "elapsed_ns": elapsed_ns,
            })
        elif role == "receiver":
            if command["kind"] != "verify":
                raise RuntimeError("receiver did not receive verify command")
            if memory_kind == "device":
                torch.cuda.synchronize()
            expected = torch.arange(size, dtype=torch.uint8) % 251
            actual = tensor.cpu()
            connection.send({
                "kind": "verify_result",
                "equal": bool(torch.equal(actual, expected)),
                "actual_sum": int(actual.sum().item()),
                "expected_sum": int(expected.sum().item()),
            })
        else:
            raise AssertionError("role was validated")
        unregister_status = engine.unregister_memory(tensor.data_ptr())
        if unregister_status != 0:
            raise RuntimeError(f"Mooncake unregister_memory returned {unregister_status}")
    except BaseException as error:
        connection.send({
            "kind": "error",
            "role": role,
            "type": type(error).__name__,
            "message": str(error),
        })
        raise
    finally:
        connection.close()


def _receive(connection: Any, label: str, timeout_s: float = 60.0) -> dict[str, Any]:
    if not connection.poll(timeout_s):
        raise TimeoutError(f"timed out waiting for {label}")
    value = connection.recv()
    if not isinstance(value, dict) or "kind" not in value:
        raise RuntimeError(f"{label} returned an invalid message")
    if value["kind"] == "error":
        raise RuntimeError(f"{label} failed: {value}")
    return value


def run_probe(protocol: str, memory_kind: str, size: int) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    receiver_parent, receiver_child = context.Pipe()
    sender_parent, sender_child = context.Pipe()
    receiver = context.Process(
        target=_worker,
        args=("receiver", 1, protocol, memory_kind, size, receiver_child),
    )
    sender = context.Process(
        target=_worker,
        args=("sender", 0, protocol, memory_kind, size, sender_child),
    )
    receiver.start()
    sender.start()
    receiver_ready = _receive(receiver_parent, "receiver readiness")
    sender_ready = _receive(sender_parent, "sender readiness")
    sender_parent.send({
        "kind": "transfer",
        "endpoint": receiver_ready["endpoint"],
        "address": receiver_ready["address"],
    })
    transfer = _receive(sender_parent, "transfer")
    receiver_parent.send({"kind": "verify"})
    verification = _receive(receiver_parent, "verification")
    sender.join(timeout=30.0)
    receiver.join(timeout=30.0)
    if sender.is_alive() or receiver.is_alive():
        raise TimeoutError("Mooncake probe worker did not exit")
    if sender.exitcode != 0 or receiver.exitcode != 0:
        raise RuntimeError(
            f"Mooncake worker exits were sender={sender.exitcode}, receiver={receiver.exitcode}"
        )
    return {
        "schema_version": 1,
        "kind": "metax_c500_mooncake_transfer_probe",
        "protocol": protocol,
        "memory_kind": memory_kind,
        "bytes": size,
        "sender": sender_ready,
        "receiver": receiver_ready,
        "transfer": transfer,
        "verification": verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("maca", "rdma", "tcp"), required=True)
    parser.add_argument("--memory-kind", choices=("host", "device"), required=True)
    parser.add_argument("--bytes", type=int, default=1 << 20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bytes < 1:
        raise ValueError("--bytes must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace probe output: {args.output}")
    result = run_probe(args.protocol, args.memory_kind, args.bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
