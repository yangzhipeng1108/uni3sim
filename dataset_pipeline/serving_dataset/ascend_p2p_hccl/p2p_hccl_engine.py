"""Strict HCCL device P2P transport for a one-rank-per-role PD canary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from collections import deque
import logging
import socket
import threading
from typing import Any

import msgpack
import torch
import torch.distributed as dist
import torch_npu
import zmq

from vllm.config.kv_transfer import KVTransferConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor
    ready_event: Any | None


@dataclass(frozen=True)
class PendingReceive:
    tensor_id: str
    tensor: torch.Tensor
    group: Any


class P2pHcclEngine:
    """Transfer KV tensors through a private two-rank ProcessGroupHCCL.

    The ZMQ channel carries only tensor metadata and a per-connection TCPStore
    endpoint.  Tensor payloads are sent directly from one NPU allocation to
    the peer NPU allocation; no host staging path exists in this connector.
    """

    def __init__(self, local_rank: int, config: KVTransferConfig, *, port_offset: int) -> None:
        if config.kv_parallel_size != 2:
            raise ValueError("P2pHcclConnector requires kv_parallel_size=2")
        if local_rank != 0 or port_offset != 0:
            raise NotImplementedError("P2pHcclConnector currently supports TP1 only")
        extra = config.kv_connector_extra_config
        for field in ("send_type", "pg_timeout_s"):
            if field not in extra:
                raise KeyError(f"P2pHcclConnector requires {field}")
        if extra["send_type"] != "PUT":
            if extra["send_type"] != "PUT_ASYNC":
                raise ValueError("P2pHcclConnector requires PUT or PUT_ASYNC")
        timeout_s = extra["pg_timeout_s"]
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s < 1:
            raise ValueError("P2pHcclConnector pg_timeout_s must be a positive integer")
        if not isinstance(config.kv_ip, str) or not config.kv_ip:
            raise ValueError("P2pHcclConnector kv_ip must be non-empty")
        if isinstance(config.kv_port, bool) or not isinstance(config.kv_port, int):
            raise TypeError("P2pHcclConnector kv_port must be an integer")

        self.config = config
        self.device = torch.device(f"npu:{local_rank}")
        self.zmq_address = f"{config.kv_ip}:{config.kv_port + port_offset}"
        self.timeout = timedelta(seconds=timeout_s)
        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")
        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)
        self.socks: dict[str, Any] = {}
        self.groups: dict[str, Any] = {}
        self.recv_store: dict[str, torch.Tensor] = {}
        self.pending_receives: deque[PendingReceive] = deque()
        self.recv_store_cv = threading.Condition()
        self.send_queue: deque[SendQueueItem] = deque()
        self.send_queue_cv = threading.Condition()
        self._send_in_flight = 0
        self._send_error: BaseException | None = None
        self.send_type = extra["send_type"]
        self._send_thread = None
        if self.send_type == "PUT_ASYNC":
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._send_thread.start()
        self._listener_error: BaseException | None = None
        self._listener_thread = threading.Thread(target=self._listen, daemon=True)
        self._listener_thread.start()

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.config.kv_ip, 0))
            return int(sock.getsockname()[1])

    def _make_group(self, remote_address: str, *, rank: int, pg_port: int) -> Any:
        if remote_address in self.groups:
            return self.groups[remote_address]
        host, separator, _ = remote_address.partition(":")
        if not separator or not host:
            raise ValueError("P2pHcclConnector remote address is malformed")
        torch.npu.set_device(self.device)
        store = dist.TCPStore(host, pg_port, 2, rank == 0, self.timeout, wait_for_workers=True)
        group = torch_npu._C._distributed_c10d.ProcessGroupHCCL(
            store, rank, 2, self.timeout)
        self.groups[remote_address] = group
        return group

    def _connect(self, remote_address: str) -> Any:
        if remote_address in self.socks:
            return self.groups[remote_address]
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        sock.connect(f"tcp://{remote_address}")
        pg_port = self._free_port()
        sock.send(msgpack.dumps({"cmd": "NEW", "pg_port": pg_port}))
        self.socks[remote_address] = sock
        return self._make_group(remote_address, rank=0, pg_port=pg_port)

    def send_tensor(self, tensor_id: str, tensor: torch.Tensor, remote_address: str | None = None) -> bool:
        if remote_address is None:
            raise ValueError("P2pHcclConnector requires a remote address for every KV tensor")
        if tensor.device != self.device:
            raise ValueError(f"P2pHcclConnector expected tensor on {self.device}, got {tensor.device}")
        ready_event = None
        if self.send_type == "PUT_ASYNC":
            # ``save_kv_layer`` runs on the model stream, whereas this
            # engine performs HCCL from its sender thread.  ProcessGroupHCCL
            # does not inherit the model stream dependency across threads;
            # make the KV slice visible before the background thread submits
            # its direct device-to-device send.
            torch.npu.set_device(self.device)
            # Keep an independent NPU allocation alive after the producer
            # returns from its Graph replay.  The original indexed slice can
            # otherwise be reused by the paged KV cache before the sender
            # thread has submitted its HCCL work.
            tensor = tensor.clone()
            ready_event = torch.npu.Event()
            ready_event.record(torch.npu.current_stream())
        item = SendQueueItem(tensor_id, remote_address, tensor, ready_event)
        if self.send_type == "PUT":
            return self.send_sync(item)
        with self.send_queue_cv:
            self.send_queue.append(item)
            self.send_queue_cv.notify()
        return True

    def _send_loop(self) -> None:
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                self._send_in_flight += 1
            try:
                if item.ready_event is not None:
                    item.ready_event.synchronize()
                self.send_sync(item)
            except BaseException as error:
                with self.send_queue_cv:
                    if self._send_error is None:
                        self._send_error = error
                raise
            finally:
                with self.send_queue_cv:
                    self._send_in_flight -= 1
                    self.send_queue_cv.notify_all()

    def send_sync(self, item: SendQueueItem) -> bool:
        group = self._connect(item.remote_address)
        sock = self.socks[item.remote_address]
        sock.send(msgpack.dumps({
            "cmd": "PUT", "tensor_id": item.tensor_id,
            "shape": tuple(item.tensor.shape),
            "dtype": str(item.tensor.dtype).replace("torch.", ""),
        }))
        if sock.recv() != b"0":
            raise RuntimeError("HCCL peer rejected a KV tensor")
        group.send([item.tensor], 1, 0).wait()
        return True

    def recv_tensor(self, tensor_id: str, remote_address: str | None = None) -> torch.Tensor:
        if remote_address is None:
            raise ValueError("P2pHcclConnector requires a remote address for every KV tensor")
        while True:
            with self.recv_store_cv:
                if tensor_id in self.recv_store:
                    return self.recv_store.pop(tensor_id)
                while not self.pending_receives:
                    if self._listener_error is not None:
                        raise RuntimeError("HCCL listener failed") from self._listener_error
                    self.recv_store_cv.wait(timeout=1.0)
                pending = self.pending_receives.popleft()
            # A private ProcessGroupHCCL receive must run on the Decode model
            # thread.  Running it in the ZMQ listener races Decode's graph
            # replay during a mixed prefill/decode step.  Drain producer-order
            # metadata here, so HCCL's tag-zero ordering remains exact even
            # when this logical request is not the first one in the batch.
            torch.npu.set_device(self.device)
            pending.group.recv([pending.tensor], 0, 0).wait()
            with self.recv_store_cv:
                if pending.tensor_id in self.recv_store:
                    raise RuntimeError(
                        f"duplicate HCCL receive tensor ID: {pending.tensor_id}"
                    )
                self.recv_store[pending.tensor_id] = pending.tensor
                self.recv_store_cv.notify_all()

    def _listen(self) -> None:
        try:
            while True:
                ready = dict(self.poller.poll())
                if self.router_socket not in ready:
                    continue
                identity, message = self.router_socket.recv_multipart()
                remote_address = identity.decode()
                data = msgpack.loads(message)
                if data["cmd"] == "NEW":
                    pg_port = data["pg_port"]
                    if isinstance(pg_port, bool) or not isinstance(pg_port, int):
                        raise TypeError("HCCL private group port must be an integer")
                    self._make_group(remote_address, rank=1, pg_port=pg_port)
                    continue
                if data["cmd"] != "PUT":
                    raise ValueError("P2pHcclConnector received an unsupported control command")
                for field in ("tensor_id", "shape", "dtype"):
                    if field not in data:
                        raise KeyError(f"P2pHcclConnector PUT lacks {field}")
                tensor = torch.empty(tuple(data["shape"]), dtype=getattr(torch, data["dtype"]), device=self.device)
                group = self.groups[remote_address]
                with self.recv_store_cv:
                    if data["tensor_id"] in self.recv_store or any(
                        item.tensor_id == data["tensor_id"]
                        for item in self.pending_receives
                    ):
                        raise RuntimeError(
                            f"duplicate HCCL receive tensor ID: {data['tensor_id']}"
                        )
                    self.pending_receives.append(
                        PendingReceive(data["tensor_id"], tensor, group)
                    )
                    self.recv_store_cv.notify_all()
                self.router_socket.send_multipart([identity, b"0"])
        except BaseException as error:
            self._listener_error = error
            with self.recv_store_cv:
                self.recv_store_cv.notify_all()
            raise

    def wait_for_sent(self) -> None:
        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                # Decode can only enter its model thread after Prefill has
                # returned the handoff response.  The queued HCCL work must
                # therefore remain asynchronous here; waiting for the queue
                # would create a P→D control-plane deadlock before Decode can
                # issue the matching receives.
                if self._send_error is not None:
                    error = self._send_error
                    self._send_error = None
                    raise RuntimeError("asynchronous HCCL KV send failed") from error
        if self._listener_error is not None:
            raise RuntimeError("HCCL listener failed") from self._listener_error

    def get_finished(self, finished_req_ids: set[str], no_compile_layers: Any) -> tuple[None, None]:
        return None, None
