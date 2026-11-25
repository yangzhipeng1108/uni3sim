# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import msgpack
import torch
import zmq

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    buffer_type, cudaStream_t, ncclComm_t, ncclDataTypeEnum)
from vllm_metax.patch.plugin_enhancement.distributed.pynccl_wrapper import (
    MCCLLibrary)

logger = logging.getLogger(__name__)

@contextmanager
def set_p2p_mccl_context(num_channels: str):
    original_values: dict[str, tuple[bool, str]] = {}
    env_vars = [
        'NCCL_MAX_NCHANNELS',
        'NCCL_MIN_NCHANNELS',
        'NCCL_CUMEM_ENABLE',
        'NCCL_BUFFSIZE',
        'NCCL_PROTO',  # LL,LL128,SIMPLE
        'NCCL_ALGO',  # RING,TREE
    ]

    for var in env_vars:
        original_values[var] = (
            (True, os.environ[var]) if var in os.environ else (False, "")
        )

    logger.info("set_p2p_mccl_context, original_values: %s", original_values)

    try:
        os.environ['NCCL_MAX_NCHANNELS'] = num_channels
        os.environ['NCCL_MIN_NCHANNELS'] = num_channels
        os.environ['NCCL_CUMEM_ENABLE'] = '1'
        yield
    finally:
        for var in env_vars:
            existed, value = original_values[var]
            if existed:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]


@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor


class P2pMcclEngine:

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 hostname: str = "",
                 port_offset: int = 0,
                 library_path: Optional[str] = None) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        if library_path is not None:
            raise ValueError("P2pMcclEngine uses the MetaX plugin MCCL library path")
        self.mccl = MCCLLibrary()

        if not hostname:
            hostname = self.config.kv_ip
        if not isinstance(hostname, str) or not hostname:
            raise ValueError("P2pMcclConnector kv_ip must be non-empty")
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port

        # Each card corresponds to a ZMQ address.
        self.zmq_address = f"{self._hostname}:{self._port}"

        # The `http_port` must be consistent with the port of OpenAI.
        self.http_address = (
            f"{self._hostname}:"
            f"{self.config.kv_connector_extra_config['http_port']}")

        self.proxy_address = ""

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        extra_config = self.config.kv_connector_extra_config
        for field in ("send_type", "nccl_num_channels"):
            if field not in extra_config:
                raise KeyError(f"P2pMcclConnector requires {field}")
        # The connector intentionally uses direct device-to-device PUT. Host
        # staging and GET mode are not valid fallbacks for this capture path.
        self.send_type = extra_config["send_type"]
        if self.send_type not in ("PUT", "PUT_ASYNC"):
            raise ValueError("P2pMcclConnector requires PUT or PUT_ASYNC")
        self.send_queue: deque[SendQueueItem] = deque()
        self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self._send_in_flight = 0
        self._send_error: BaseException | None = None
        if self.send_type == "PUT_ASYNC":
            self._send_thread = threading.Thread(target=self.send_async,
                                                 daemon=True)
            self._send_thread.start()

        # tensor_id: torch.Tensor/(addr, dtype, shape)
        self.recv_store: dict[str, Any] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.socks: dict[str, Any] = {}  # remote_address: client socket
        self.comms: dict[str, Any] = {}  # remote_address: (ncclComm_t, rank)

        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)

        self.nccl_num_channels = str(extra_config["nccl_num_channels"])

        self._listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True)
        self._listener_thread.start()

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self.ping, daemon=True)
            self._ping_thread.start()

        logger.info(
            "P2pMcclEngine init, rank:%d, local_rank:%d, http_address:%s, "
            "zmq_address:%s, proxy_address:%s, send_type:%s, buffer_size_"
            "threshold:%.2f, nccl_num_channels:%s", self.rank, self.local_rank,
            self.http_address, self.zmq_address, self.proxy_address,
            self.send_type, self.buffer_size_threshold, self.nccl_num_channels)

    def create_connect(self, remote_address: str):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            if remote_address in self.comms:
                logger.info("👋comm exists, remote_address:%s, comms:%s",
                            remote_address, self.comms)
                return sock, self.comms[remote_address]

            unique_id = self.mccl.ncclGetUniqueId()
            data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
            sock.send(msgpack.dumps(data))

            with torch.cuda.device(self.device):
                rank = 0
                with set_p2p_mccl_context(self.nccl_num_channels):
                    comm: ncclComm_t = self.mccl.ncclCommInitRank(
                        2, unique_id, rank)
                self.comms[remote_address] = (comm, rank)
                logger.info("🤝ncclCommInitRank Success, %s👉%s, MyRank:%s",
                            self.zmq_address, remote_address, rank)

        return self.socks[remote_address], self.comms[remote_address]

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: str | None = None,
    ) -> bool:
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True

        item = SendQueueItem(tensor_id=tensor_id,
                             remote_address=remote_address,
                             tensor=tensor)

        if self.send_type == "PUT":
            return self.send_sync(item)

        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                self.send_queue.append(item)
                self.send_queue_cv.notify()
            return True

        raise AssertionError("P2pMcclConnector send type was validated")

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: str | None = None,
    ) -> torch.Tensor:
        with self.recv_store_cv:
            while tensor_id not in self.recv_store:
                self.recv_store_cv.wait()
            tensor = self.recv_store.pop(tensor_id)
        if tensor is None:
            raise RuntimeError(f"MCCL failed to receive tensor {tensor_id}")
        self.buffer_size -= tensor.element_size() * tensor.numel()
        return tensor

    def listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket not in socks:
                continue

            remote_address, message = self.router_socket.recv_multipart()
            data = msgpack.loads(message)
            if data["cmd"] == "NEW":
                unique_id = self.mccl.unique_id_from_bytes(
                    bytes(data["unique_id"]))
                with torch.cuda.device(self.device):
                    rank = 1
                    with set_p2p_mccl_context(self.nccl_num_channels):
                        comm: ncclComm_t = self.mccl.ncclCommInitRank(
                            2, unique_id, rank)
                    self.comms[remote_address.decode()] = (comm, rank)
                    logger.info("🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
                                self.zmq_address, remote_address.decode(),
                                rank)
            elif data["cmd"] == "PUT":
                tensor_id = data["tensor_id"]
                try:
                    dtype = getattr(torch, data["dtype"])
                    tensor_size = (
                        torch.empty((), dtype=dtype).element_size()
                        * int(torch.Size(data["shape"]).numel())
                    )
                    if self.buffer_size + tensor_size > self.buffer_size_threshold:
                        self.router_socket.send_multipart([remote_address, b"1"])
                        tensor = None
                    else:
                        with torch.cuda.stream(self.recv_stream):
                            tensor = torch.empty(data["shape"], dtype=dtype,
                                                 device=self.device)
                        self.router_socket.send_multipart([remote_address, b"0"])
                        comm, rank = self.comms[remote_address.decode()]
                        self.recv(comm, tensor, rank ^ 1, self.recv_stream)
                        self.buffer_size += tensor_size
                except torch.cuda.OutOfMemoryError:
                    self.router_socket.send_multipart([remote_address, b"1"])
                    tensor = None
                    logger.warning(
                        "🔴[PUT]Recv Tensor, Out Of Memory, %s👈%s, "
                        "data:%s", self.zmq_address, remote_address.decode(),
                        data)

                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.have_received_tensor_id(tensor_id)
                    self.recv_store_cv.notify()

            else:
                raise RuntimeError(
                    f"unexpected MCCL control command {data['cmd']!r}")

    def have_sent_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split('#')[0]
        if request_id not in self.send_request_id_to_tensor_ids:
            self.send_request_id_to_tensor_ids[request_id] = set()
        self.send_request_id_to_tensor_ids[request_id].add(tensor_id)

    def have_received_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split('#')[0]
        if request_id not in self.recv_request_id_to_tensor_ids:
            self.recv_request_id_to_tensor_ids[request_id] = set()
        self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)

    def send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                self._send_in_flight += 1
            try:
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

    def wait_for_sent(self):
        if self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.send_queue_cv:
                while self.send_queue or self._send_in_flight:
                    self.send_queue_cv.wait()
                if self._send_error is not None:
                    error = self._send_error
                    self._send_error = None
                    raise RuntimeError("asynchronous MCCL send failed") from error
            duration = time.time() - start_time
            logger.debug(
                "🚧[PUT_ASYNC]It took %.3fms to wait for the send_queue"
                " to be empty, rank:%d", duration * 1000, self.rank)

    def send_sync(self, item: SendQueueItem) -> bool:
        if item.remote_address is None:
            return False
        if item.remote_address not in self.socks:
            self.create_connect(item.remote_address)

        tensor = item.tensor

        sock = self.socks[item.remote_address]
        comm, rank = self.comms[item.remote_address]
        data = {
            "cmd": "PUT",
            "tensor_id": item.tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", "")
        }
        sock.send(msgpack.dumps(data))

        response = sock.recv()
        if response != b"0":
            raise RuntimeError(
                "MCCL peer rejected tensor "
                f"{item.tensor_id} with response {response.decode()!r}")

        self.send(comm, tensor.to(self.device), rank ^ 1, self.send_stream)

        if self.send_type == "PUT_ASYNC":
            self.have_sent_tensor_id(item.tensor_id)

        return True

    def get_finished(
            self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        # Clear the buffer upon request completion.
        for request_id in finished_req_ids:
            for layer_name in no_compile_layers:
                tensor_id = request_id + "#" + layer_name
                if tensor_id in self.recv_store:
                    with self.recv_store_cv:
                        tensor = self.recv_store.pop(tensor_id)
                    self.buffer_size -= tensor.element_size() * tensor.numel()
            if request_id in self.send_request_id_to_tensor_ids:
                del self.send_request_id_to_tensor_ids[request_id]
            if request_id in self.recv_request_id_to_tensor_ids:
                del self.recv_request_id_to_tensor_ids[request_id]

        # TODO:Retrieve requests that have already sent the KV cache.
        finished_sending: set[str] = set()

        # TODO:Retrieve requests that have already received the KV cache.
        finished_recving: set[str] = set()

        return finished_sending or None, finished_recving or None

    def ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.debug("ping start, zmq_address:%s", self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)

    def send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            self.mccl.ncclSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), dst,
                               comm, cudaStream_t(stream.cuda_stream))
        stream.synchronize()

    def recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            self.mccl.ncclRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), src,
                               comm, cudaStream_t(stream.cuda_stream))
        stream.synchronize()

    def close(self) -> None:
        self._listener_thread.join()
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join()
        if self._ping_thread is not None:
            self._ping_thread.join()
