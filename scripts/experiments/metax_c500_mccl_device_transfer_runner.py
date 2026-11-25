#!/usr/bin/env python3
"""Validate one direct MCCL transfer between two MetaX C500 devices."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any


def _worker(rank: int, physical_device: int, size: int, connection: Any) -> None:
    os.environ["MACA_VISIBLE_DEVICES"] = str(physical_device)
    try:
        import torch
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            buffer_type,
            cudaStream_t,
            ncclDataTypeEnum,
        )
        from vllm_metax.patch.plugin_enhancement.distributed.pynccl_wrapper import (
            MCCLLibrary,
        )

        torch.cuda.set_device(0)
        tensor = (
            torch.arange(size, dtype=torch.uint8, device="cuda:0") % 251
            if rank == 0
            else torch.zeros(size, dtype=torch.uint8, device="cuda:0")
        )
        stream = torch.cuda.Stream(device=0)
        library = MCCLLibrary()
        if rank == 0:
            unique_id = library.ncclGetUniqueId()
            connection.send({"kind": "unique_id", "value": bytes(unique_id.internal)})
            command = connection.recv()
            unique_id_bytes = command["value"]
        else:
            command = connection.recv()
            unique_id_bytes = command["value"]
        communicator = library.ncclCommInitRank(
            2, library.unique_id_from_bytes(unique_id_bytes), rank
        )
        connection.send({"kind": "ready", "rank": rank})
        command = connection.recv()
        if command["kind"] != "transfer":
            raise RuntimeError("MCCL worker did not receive transfer command")
        started_ns = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            if rank == 0:
                library.ncclSend(
                    buffer_type(tensor.data_ptr()),
                    tensor.numel(),
                    ncclDataTypeEnum.from_torch(tensor.dtype),
                    1,
                    communicator,
                    cudaStream_t(stream.cuda_stream),
                )
            else:
                library.ncclRecv(
                    buffer_type(tensor.data_ptr()),
                    tensor.numel(),
                    ncclDataTypeEnum.from_torch(tensor.dtype),
                    0,
                    communicator,
                    cudaStream_t(stream.cuda_stream),
                )
        stream.synchronize()
        elapsed_ns = time.perf_counter_ns() - started_ns
        result: dict[str, object] = {
            "kind": "result",
            "rank": rank,
            "elapsed_ns": elapsed_ns,
        }
        if rank == 1:
            expected = torch.arange(size, dtype=torch.uint8) % 251
            actual = tensor.cpu()
            result.update({
                "equal": bool(torch.equal(actual, expected)),
                "actual_sum": int(actual.sum().item()),
                "expected_sum": int(expected.sum().item()),
            })
        connection.send(result)
    except BaseException as error:
        connection.send({
            "kind": "error",
            "rank": rank,
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


def run_probe(size: int, physical_devices: tuple[int, int]) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    rank0_parent, rank0_child = context.Pipe()
    rank1_parent, rank1_child = context.Pipe()
    rank0 = context.Process(
        target=_worker, args=(0, physical_devices[0], size, rank0_child)
    )
    rank1 = context.Process(
        target=_worker, args=(1, physical_devices[1], size, rank1_child)
    )
    rank0.start()
    rank1.start()
    unique_id = _receive(rank0_parent, "rank 0 unique ID")
    rank0_parent.send(unique_id)
    rank1_parent.send(unique_id)
    rank0_ready = _receive(rank0_parent, "rank 0 readiness")
    rank1_ready = _receive(rank1_parent, "rank 1 readiness")
    rank0_parent.send({"kind": "transfer"})
    rank1_parent.send({"kind": "transfer"})
    rank0_result = _receive(rank0_parent, "rank 0 transfer")
    rank1_result = _receive(rank1_parent, "rank 1 transfer")
    rank0.join(timeout=30.0)
    rank1.join(timeout=30.0)
    if rank0.is_alive() or rank1.is_alive():
        raise TimeoutError("MCCL probe worker did not exit")
    if rank0.exitcode != 0 or rank1.exitcode != 0:
        raise RuntimeError(
            f"MCCL worker exits were rank0={rank0.exitcode}, rank1={rank1.exitcode}"
        )
    return {
        "schema_version": 1,
        "kind": "metax_c500_mccl_transfer_probe",
        "bytes": size,
        "physical_devices": list(physical_devices),
        "rank0": {**rank0_ready, **rank0_result},
        "rank1": {**rank1_ready, **rank1_result},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=1 << 20)
    parser.add_argument("--devices", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bytes < 1:
        raise ValueError("--bytes must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace probe output: {args.output}")
    physical_devices = tuple(args.devices)
    if physical_devices[0] == physical_devices[1]:
        raise ValueError("--devices must name two distinct devices")
    result = run_probe(args.bytes, physical_devices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
