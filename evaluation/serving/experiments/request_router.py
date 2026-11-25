#!/usr/bin/env python3
"""OpenAI-compatible, evidence-preserving request router for real vLLM PD."""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request
import re


_HANDOFF_TRANSPORTS = (
    "openai_kv_transfer_params",
    "p2p_request_id",
)
_CANONICAL_REQUEST_ID = re.compile(r"request-([0-9]+)")
_WARMUP_REQUEST_ID = re.compile(r"warmup-[a-z0-9-]+")


def main(argv: list[str] | None = None) -> int:
    """Run one PD router until its process is terminated."""

    args = _parse_args(argv)
    state = _router_state(args)
    ThreadingHTTPServer((args.host, args.port), _openai_handler(state)).serve_forever()
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route real vLLM PD requests")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prefill-base-url", required=True)
    parser.add_argument("--decode-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--handoff-transport", choices=_HANDOFF_TRANSPORTS, required=True)
    parser.add_argument("--prefill-kv-ip", required=True)
    parser.add_argument("--prefill-kv-port", type=int, required=True)
    parser.add_argument("--decode-kv-ip", required=True)
    parser.add_argument("--decode-kv-port", type=int, required=True)
    parser.add_argument("--handoff-log", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-origin-monotonic-ns", type=int, required=True)
    parser.add_argument("--host-boot-id", required=True)
    return parser.parse_args(argv)


def _router_state(args: argparse.Namespace) -> dict[str, object]:
    if args.run_origin_monotonic_ns < 0:
        raise ValueError("run origin monotonic timestamp must be non-negative")
    for label, value in (
        ("prefill KV port", args.prefill_kv_port),
        ("decode KV port", args.decode_kv_port),
    ):
        if value <= 0 or value > 65535:
            raise ValueError(f"{label} must be in [1, 65535]")
    path = args.handoff_log.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "model": args.model,
        "prefill_base_url": args.prefill_base_url.rstrip("/"),
        "decode_base_url": args.decode_base_url.rstrip("/"),
        "handoff_transport": args.handoff_transport,
        "prefill_kv_ip": args.prefill_kv_ip,
        "prefill_kv_port": args.prefill_kv_port,
        "decode_kv_ip": args.decode_kv_ip,
        "decode_kv_port": args.decode_kv_port,
        "handoff_log": path,
        "run_id": args.run_id,
        "run_origin_monotonic_ns": args.run_origin_monotonic_ns,
        "host_boot_id": args.host_boot_id,
        "request_sequence": 0,
        "lock": threading.Lock(),
    }


def _openai_handler(state: dict[str, object]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        response_started = False

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                try:
                    payload = _request_json(
                        f"{_state_text(state, 'decode_base_url')}/v1/models"
                    )
                except RuntimeError as exc:
                    self._write_json(503, {"error": str(exc)})
                    return
                self._write_json(200, payload)
                return
            self._write_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/completions":
                self._write_json(404, {"error": "not found"})
                return
            try:
                payload = self._read_json()
                _require_client_streaming(payload)
                if payload["model"] != _state_text(state, "model"):
                    raise RuntimeError(
                        f"request model {payload['model']!r} does not match "
                        f"router model {_state_text(state, 'model')!r}")
                _route_streaming_completion(self, state, payload)
            except RuntimeError as exc:
                if self.response_started:
                    # An SSE response has no valid JSON error trailer.  The
                    # handoff record already contains the failure evidence.
                    self.close_connection = True
                    return
                self._write_json(502, {"error": str(exc)})

        def _read_json(self) -> dict[str, Any]:
            header = self.headers["Content-Length"]
            if header is None:
                raise RuntimeError("Content-Length header is required")
            try:
                length = int(header)
            except ValueError as exc:
                raise RuntimeError("Content-Length must be an integer") from exc
            if length <= 0:
                raise RuntimeError("request body must be non-empty")
            try:
                payload = json.loads(self.rfile.read(length).decode())
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON request: {exc}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("request payload must be a JSON object")
            return payload

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _route_streaming_completion(
    handler: BaseHTTPRequestHandler,
    state: dict[str, object],
    payload: dict[str, Any],
) -> None:
    received_ns = time.perf_counter_ns()
    request_sequence = _next_request_sequence(state)
    external_request_id, request_idx = _external_request_identity(handler)
    prefill_request_id, decode_request_id = _role_request_ids(
        state, external_request_id)
    prefill_payload = dict(payload)
    prefill_payload["max_tokens"] = 1
    prefill_payload["stream"] = False
    # vLLM's benchmark client requests streaming usage metadata.  The Prefill
    # control request is deliberately non-streaming, where that option is
    # invalid and rejected by the OpenAI protocol validator.
    if "stream_options" in prefill_payload:
        del prefill_payload["stream_options"]
    if _state_text(state, "handoff_transport") == "openai_kv_transfer_params":
        # Mooncake's producer consumes this transfer identity.  The thin
        # producer adapter returns the live engine/bootstrap endpoint in its
        # response, and only that returned payload reaches Decode.
        prefill_payload["kv_transfer_params"] = {
            "do_remote_decode": True,
            "transfer_id": _mooncake_transfer_id(state, external_request_id),
        }
    prefill_response_ns: int | None = None
    kv_transfer_params: dict[str, object] | None = None
    decode_sent_ns: int | None = None
    response = None
    first_token_ns: int | None = None
    stream_completed_ns: int | None = None
    failure: str | None = None
    try:
        prefill_response = _request_json(
            f"{_state_text(state, 'prefill_base_url')}/v1/completions",
            prefill_payload,
            {"X-Request-Id": prefill_request_id},
        )
        prefill_response_ns = time.perf_counter_ns()
        kv_transfer_params = _validate_prefill_handoff(
            state, prefill_response, external_request_id)
        decode_payload = dict(payload)
        decode_payload["stream"] = True
        if kv_transfer_params is not None:
            decode_payload["kv_transfer_params"] = kv_transfer_params
        decode_sent_ns = time.perf_counter_ns()
        response = _open_stream(
            f"{_state_text(state, 'decode_base_url')}/v1/completions",
            decode_payload,
            {"X-Request-Id": decode_request_id},
        )
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.response_started = True
        while True:
            line = response.readline()
            if not line:
                break
            handler.wfile.write(line)
            handler.wfile.flush()
            if first_token_ns is None and _sse_line_has_visible_token(line):
                first_token_ns = time.perf_counter_ns()
        stream_completed_ns = time.perf_counter_ns()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (OSError, urllib_error.URLError)):
            raise RuntimeError(f"Decode SSE forwarding failed: {exc}") from exc
        raise
    finally:
        if response is not None:
            response.close()
        _append_handoff_record(
            state,
            {
                "schema_version": 1,
                "event_kind": "pd_router_handoff",
                "run_id": _state_text(state, "run_id"),
                "run_origin_monotonic_ns": state["run_origin_monotonic_ns"],
                "host_boot_id": _state_text(state, "host_boot_id"),
                "request_idx": request_idx,
                "router_sequence": request_sequence,
                "external_request_id": external_request_id,
                "producer_request_id": prefill_request_id,
                "consumer_request_id": decode_request_id,
                "handoff_transport": _state_text(state, "handoff_transport"),
                "producer_control_response": _relative_or_none(
                    state, prefill_response_ns),
                "consumer_request_sent": _relative_or_none(
                    state, decode_sent_ns),
                "consumer_first_visible_token": _relative_or_none(state, first_token_ns),
                "consumer_stream_completed": _relative_or_none(
                    state, stream_completed_ns),
                "router_received": _relative_ns(state, received_ns),
                "kv_transfer_params": kv_transfer_params,
                "kv_transfer_params_sha256": _json_sha256(kv_transfer_params),
                "transfer_start": {"status": "unobserved_by_router"},
                "transfer_end": {"status": "unobserved_by_router"},
                "consumer_kv_load": {"status": "requires_decode_instrumentation"},
                "router_status": "completed" if failure is None else "failed",
                "failure": failure,
            },
        )


def _require_client_streaming(payload: Mapping[str, object]) -> None:
    if "stream" not in payload:
        raise RuntimeError("PD router requires explicit stream=true")
    if payload["stream"] is not True:
        raise RuntimeError("PD router only supports stream=true client requests")


def _role_request_ids(
    state: Mapping[str, object], external_request_id: str
) -> tuple[str, str]:
    transport = _state_text(state, "handoff_transport")
    if transport == "openai_kv_transfer_params":
        # The producer request ID is returned inside Mooncake's transfer
        # parameters.  Keep the same canonical ID on both isolated engines so
        # the consumer can use that identity directly and Scheduler tracing
        # remains joinable to requests.csv.
        return external_request_id, external_request_id
    if transport == "p2p_request_id":
        decode_ip = _state_text(state, "decode_kv_ip")
        decode_port = _state_port(state, "decode_kv_port")
        prefill_ip = _state_text(state, "prefill_kv_ip")
        prefill_port = _state_port(state, "prefill_kv_port")
        return (
            f"{external_request_id}___decode_addr_{decode_ip}:{decode_port}",
            f"{external_request_id}___prefill_addr_{prefill_ip}:{prefill_port}___",
        )
    raise RuntimeError(f"unsupported PD handoff transport: {transport!r}")


def _mooncake_transfer_id(
    state: Mapping[str, object], external_request_id: str
) -> str:
    if _CANONICAL_REQUEST_ID.fullmatch(external_request_id) is None and (
        _WARMUP_REQUEST_ID.fullmatch(external_request_id) is None
    ):
        raise RuntimeError("Mooncake transfer identity requires a valid PD request ID")
    return f"{_state_text(state, 'run_id')}:mooncake:{external_request_id}"


def _validate_prefill_handoff(
    state: Mapping[str, object],
    response: Mapping[str, object],
    external_request_id: str,
) -> dict[str, object] | None:
    transport = _state_text(state, "handoff_transport")
    if "kv_transfer_params" not in response:
        raise RuntimeError("Prefill response omits kv_transfer_params")
    params = response["kv_transfer_params"]
    if transport == "openai_kv_transfer_params":
        if not isinstance(params, dict) or not params:
            raise RuntimeError(
                "Prefill response lacks non-empty kv_transfer_params for "
                f"external request {external_request_id!r}")
        required = {
            "do_remote_prefill", "remote_engine_id", "remote_bootstrap_addr",
            "transfer_id",
        }
        if set(params) != required:
            raise RuntimeError("Mooncake Prefill response has an invalid handoff field set")
        if params["do_remote_prefill"] is not True:
            raise RuntimeError("Mooncake Prefill response does not enable remote Prefill")
        for field in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id"):
            if not isinstance(params[field], str) or not params[field]:
                raise RuntimeError(f"Mooncake Prefill response {field} is invalid")
        if not params["remote_bootstrap_addr"].startswith("http://"):
            raise RuntimeError("Mooncake Prefill response bootstrap address is invalid")
        if params["transfer_id"] != _mooncake_transfer_id(state, external_request_id):
            raise RuntimeError("Mooncake Prefill response transfer identity disagrees with router")
        return dict(params)
    if transport == "p2p_request_id":
        if params is not None:
            raise RuntimeError(
                "P2pNcclConnector must use request-id routing, not "
                "OpenAI kv_transfer_params")
        return None
    raise RuntimeError(f"unsupported PD handoff transport: {transport!r}")


def _open_stream(
    url: str,
    payload: Mapping[str, object],
    headers: Mapping[str, str],
):
    request_headers = {"Content-Type": "application/json", **headers}
    try:
        return urllib_request.urlopen(
            urllib_request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers=request_headers,
            ),
            timeout=60.0,
        )
    except (OSError, urllib_error.URLError) as exc:
        raise RuntimeError(f"upstream streaming request failed for {url}: {exc}") from exc


def _request_json(
    url: str,
    payload: Mapping[str, object] | None = None,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request_headers = {} if headers is None else dict(headers)
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    try:
        with urllib_request.urlopen(
            urllib_request.Request(url, data=data, headers=request_headers),
            timeout=60.0,
        ) as response:
            body = response.read().decode()
    except (OSError, urllib_error.URLError) as exc:
        raise RuntimeError(f"upstream request failed for {url}: {exc}") from exc
    try:
        if not body:
            raise RuntimeError(f"upstream returned an empty response for {url}")
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"upstream returned invalid JSON for {url}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"upstream returned non-object JSON for {url}")
    return decoded


def _sse_line_has_visible_token(line: bytes) -> bool:
    prefix = b"data: "
    if not line.startswith(prefix):
        return False
    raw_payload = line[len(prefix):].strip()
    if raw_payload == b"[DONE]":
        return False
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Decode SSE payload is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Decode SSE payload must be a JSON object")
    if "choices" not in payload or not isinstance(payload["choices"], list):
        raise RuntimeError("Decode SSE payload lacks choices")
    for value in payload["choices"]:
        if not isinstance(value, dict):
            raise RuntimeError("Decode SSE choice must be a JSON object")
        if "text" not in value or not isinstance(value["text"], str):
            raise RuntimeError("Decode SSE choice lacks text")
        if value["text"]:
            return True
    return False


def _next_request_sequence(state: dict[str, object]) -> int:
    lock = state["lock"]
    with lock:
        sequence = state["request_sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise RuntimeError("router request sequence is invalid")
        state["request_sequence"] = sequence + 1
    return sequence


def _external_request_identity(
    handler: BaseHTTPRequestHandler,
) -> tuple[str, int | None]:
    incoming = handler.headers["X-Request-Id"]
    if incoming is None or not incoming:
        raise RuntimeError("PD router requires a non-empty X-Request-Id")
    canonical = _CANONICAL_REQUEST_ID.fullmatch(incoming)
    if canonical is not None:
        return incoming, int(canonical.group(1))
    if _WARMUP_REQUEST_ID.fullmatch(incoming) is not None:
        return incoming, None
    raise RuntimeError(
        "PD router X-Request-Id must be canonical request-<index> or a "
        "reserved warmup ID"
    )


def _append_handoff_record(state: Mapping[str, object], record: Mapping[str, object]) -> None:
    lock = state["lock"]
    path = state["handoff_log"]
    if not isinstance(path, Path):
        raise RuntimeError("router handoff log path is invalid")
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
            handle.flush()


def _state_text(state: Mapping[str, object], key: str) -> str:
    value = state[key]
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"router state {key!r} must be non-empty text")
    return value


def _state_port(state: Mapping[str, object], key: str) -> int:
    value = state[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"router state {key!r} must be an integer")
    if value <= 0 or value > 65535:
        raise RuntimeError(f"router state {key!r} is outside TCP port range")
    return value


def _relative_ns(state: Mapping[str, object], timestamp_ns: int) -> int:
    origin = state["run_origin_monotonic_ns"]
    if isinstance(origin, bool) or not isinstance(origin, int):
        raise RuntimeError("router run origin is invalid")
    if timestamp_ns < origin:
        raise RuntimeError("router event predates configured run origin")
    return timestamp_ns - origin


def _relative_or_none(
    state: Mapping[str, object], timestamp_ns: int | None
) -> int | None:
    if timestamp_ns is None:
        return None
    return _relative_ns(state, timestamp_ns)


def _json_sha256(value: object) -> str | None:
    if value is None:
        return None
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
