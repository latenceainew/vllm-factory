"""
Async HTTP reverse proxy for multi-instance vLLM serving (beta).

Transparently forwards every incoming HTTP request to one of N vLLM
backend workers, enforcing a per-backend concurrency cap via asyncio
semaphores and selecting backends with round-robin. An optional
request-affinity mode can keep similar JSON requests on the same backend
to preserve backend-local caches.

The dispatcher works with any endpoint (``/pooling``, ``/v1/embeddings``,
``/health``, etc.) and does not import any plugin code. When request
affinity is disabled it treats request bodies as opaque. When affinity is
enabled it may inspect JSON payload shape to derive a stable routing key.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any, Sequence

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

logger = logging.getLogger("vllm-factory.dispatcher")

_DEFAULT_MAX_BS = 32
_DEFAULT_AFFINITY_CACHE_SIZE = 2048
_CLIENT_TIMEOUT = ClientTimeout(total=300, connect=10)
_VOLATILE_JSON_FIELDS = frozenset({"input", "inputs", "prompt", "prompts", "text", "texts"})


def _text_length_bucket(value: Any) -> Any:
    if isinstance(value, str):
        length = len(value)
        if length <= 256:
            return "__len_le_256__"
        if length <= 1024:
            return "__len_le_1024__"
        if length <= 4096:
            return "__len_le_4096__"
        return "__len_gt_4096__"
    if isinstance(value, list):
        total_chars = sum(len(item) for item in value if isinstance(item, str))
        item_count = len(value)
        if total_chars <= 512:
            bucket = "__list_chars_le_512__"
        elif total_chars <= 2048:
            bucket = "__list_chars_le_2048__"
        elif total_chars <= 8192:
            bucket = "__list_chars_le_8192__"
        else:
            bucket = "__list_chars_gt_8192__"
        return {"bucket": bucket, "items": item_count}
    return "__affinity_text__"


def _normalize_affinity_json(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            item = value[key]
            if key in _VOLATILE_JSON_FIELDS:
                normalized[key] = _text_length_bucket(item)
                continue
            normalized[key] = _normalize_affinity_json(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_affinity_json(item) for item in value]
    return value


class Dispatcher:
    """Thin async reverse proxy with per-backend concurrency caps.

    Parameters
    ----------
    backend_urls:
        Base URLs of the vLLM backend servers (e.g. ``["http://127.0.0.1:9100", ...]``).
    max_bs:
        Maximum number of in-flight requests allowed per backend.
        Should match the backend's ``--max-num-seqs``.
    host:
        Host to bind the dispatcher to.
    port:
        Port to bind the dispatcher to.
    """

    def __init__(
        self,
        backend_urls: Sequence[str],
        max_bs: int = _DEFAULT_MAX_BS,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_request_affinity: bool = False,
        affinity_cache_size: int = _DEFAULT_AFFINITY_CACHE_SIZE,
    ):
        if not backend_urls:
            raise ValueError("At least one backend URL is required")

        self._backend_urls = list(backend_urls)
        self._max_bs = max(1, max_bs)
        self._host = host
        self._port = port
        self._enable_request_affinity = enable_request_affinity
        self._affinity_cache_size = max(1, affinity_cache_size)
        self._num_backends = len(self._backend_urls)

        self._semaphores: list[asyncio.Semaphore] = []
        self._rr_counter = 0
        self._affinity_map: OrderedDict[str, int] = OrderedDict()
        self._routing_stats = {
            "requests": 0,
            "backend_counts": [0 for _ in range(self._num_backends)],
            "affinity_hits": 0,
            "affinity_misses": 0,
            "affinity_fallbacks": 0,
        }
        self._session: ClientSession | None = None
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        """Start the dispatcher HTTP server."""
        self._semaphores = [asyncio.Semaphore(self._max_bs) for _ in range(self._num_backends)]
        self._rr_counter = 0
        self._affinity_map.clear()
        self._routing_stats = {
            "requests": 0,
            "backend_counts": [0 for _ in range(self._num_backends)],
            "affinity_hits": 0,
            "affinity_misses": 0,
            "affinity_fallbacks": 0,
        }

        connector = TCPConnector(
            limit=self._max_bs * self._num_backends * 2,
            keepalive_timeout=60,
        )
        self._session = ClientSession(connector=connector, timeout=_CLIENT_TIMEOUT)

        app = web.Application()
        app.router.add_route("*", "/health", self._handle_health)
        app.router.add_route("*", "/stats", self._handle_stats)
        app.router.add_route("*", "/{path:.*}", self._handle_proxy)

        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

        logger.info(
            f"Dispatcher listening on {self._host}:{self._port} "
            f"-> {self._num_backends} backends (max_bs={self._max_bs}, "
            f"request_affinity={'on' if self._enable_request_affinity else 'off'})"
        )

    async def stop(self) -> None:
        """Shut down the dispatcher."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _has_capacity(self, idx: int) -> bool:
        sem = self._semaphores[idx]
        return sem._value > 0  # noqa: SLF001 — fast capacity check

    def _remember_affinity(self, affinity_key: str, idx: int) -> None:
        self._affinity_map[affinity_key] = idx
        self._affinity_map.move_to_end(affinity_key)
        while len(self._affinity_map) > self._affinity_cache_size:
            self._affinity_map.popitem(last=False)

    def _forget_affinity(self, affinity_key: str | None, idx: int) -> None:
        if affinity_key is None:
            return
        pinned_idx = self._affinity_map.get(affinity_key)
        if pinned_idx == idx:
            self._affinity_map.pop(affinity_key, None)

    @staticmethod
    def _parse_json_body(body: bytes, content_type: str | None) -> Any | None:
        if not body or not content_type or "json" not in content_type.lower():
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _schema_request_size(schema: Any) -> int | None:
        if not isinstance(schema, dict):
            return None

        size = 0

        entities = schema.get("entities")
        if isinstance(entities, list):
            size += len(entities)
        elif isinstance(entities, dict):
            size += len(entities)

        classifications = schema.get("classifications")
        if isinstance(classifications, list):
            size += len(classifications)

        relations = schema.get("relations")
        if isinstance(relations, list):
            size += len(relations)
        elif isinstance(relations, dict):
            size += len(relations)

        structures = schema.get("structures") or schema.get("json_structures")
        if isinstance(structures, dict):
            size += len(structures)
            for structure in structures.values():
                if not isinstance(structure, dict):
                    continue
                fields = structure.get("fields")
                if isinstance(fields, list):
                    size += len(fields)

        return size

    def _record_routing_observability(self, idx: int, affinity_state: str) -> None:
        stats = self._routing_stats
        stats["requests"] += 1
        stats["backend_counts"][idx] += 1
        if affinity_state == "hit":
            stats["affinity_hits"] += 1
        elif affinity_state == "miss":
            stats["affinity_misses"] += 1
        elif affinity_state == "fallback":
            stats["affinity_fallbacks"] += 1

    def _snapshot_stats(self) -> dict[str, Any]:
        return {
            "requests": self._routing_stats["requests"],
            "backend_counts": list(self._routing_stats["backend_counts"]),
            "affinity_hits": self._routing_stats["affinity_hits"],
            "affinity_misses": self._routing_stats["affinity_misses"],
            "affinity_fallbacks": self._routing_stats["affinity_fallbacks"],
            "request_affinity_enabled": self._enable_request_affinity,
            "backend_urls": list(self._backend_urls),
            "max_batch_size": self._max_bs,
            "affinity_cache_entries": len(self._affinity_map),
            "affinity_cache_size": self._affinity_cache_size,
            "backend_capacity": [
                {
                    "backend_index": idx,
                    "available_slots": sem._value,  # noqa: SLF001 - mirrors fast-path check
                    "max_slots": self._max_bs,
                }
                for idx, sem in enumerate(self._semaphores)
            ],
        }

    @staticmethod
    def _build_route_headers(
        idx: int,
        affinity_state: str,
        schema_size: int | None,
    ) -> dict[str, str]:
        headers = {
            "X-VLLM-Factory-Backend-Index": str(idx),
            "X-VLLM-Factory-Affinity-State": affinity_state,
        }
        if schema_size is not None:
            headers["X-VLLM-Factory-Schema-Size"] = str(schema_size)
        return headers

    def _log_route_observability(
        self,
        *,
        method: str,
        path: str,
        idx: int,
        backend_url: str,
        affinity_state: str,
        schema_size: int | None,
        status: int,
    ) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return

        logger.info(
            "Dispatcher routed request method=%s path=%s backend_idx=%d backend_url=%s "
            "affinity_state=%s schema_size=%s status=%d",
            method,
            path,
            idx,
            backend_url,
            affinity_state,
            schema_size if schema_size is not None else "_unknown",
            status,
        )

    @staticmethod
    def _make_affinity_key(
        method: str,
        path: str,
        query_string: str,
        content_type: str | None,
        body: bytes,
        payload: Any | None = None,
    ) -> str | None:
        if not body:
            return f"{method}:{path}?{query_string}"

        normalized_body: str
        if content_type and "json" in content_type.lower():
            try:
                if payload is None:
                    payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                normalized_body = hashlib.sha1(body).hexdigest()
            else:
                normalized_body = json.dumps(
                    _normalize_affinity_json(payload),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
        else:
            normalized_body = hashlib.sha1(body).hexdigest()

        digest = hashlib.sha1(normalized_body.encode("utf-8")).hexdigest()
        return f"{method}:{path}?{query_string}:{digest}"

    def _pick_backend(self, affinity_key: str | None = None) -> int:
        """Select a backend index via round-robin, preferring one with capacity.

        Tries up to N backends starting from the round-robin cursor.
        Falls back to the cursor position if all are at capacity (the
        semaphore will queue the request).
        """
        if affinity_key is not None:
            pinned_idx = self._affinity_map.get(affinity_key)
            if pinned_idx is not None and self._has_capacity(pinned_idx):
                self._affinity_map.move_to_end(affinity_key)
                return pinned_idx

        start = self._rr_counter % self._num_backends
        self._rr_counter += 1

        for offset in range(self._num_backends):
            idx = (start + offset) % self._num_backends
            if self._has_capacity(idx):
                if affinity_key is not None:
                    self._remember_affinity(affinity_key, idx)
                return idx

        if affinity_key is not None:
            self._remember_affinity(affinity_key, start)
        return start

    async def _handle_proxy(self, request: web.Request) -> web.Response:
        """Forward a request to a backend and relay the response."""
        assert self._session is not None

        body = await request.read()
        content_type = request.headers.get("Content-Type")
        payload = self._parse_json_body(body, content_type)
        affinity_key = None
        if self._enable_request_affinity:
            affinity_key = self._make_affinity_key(
                request.method,
                request.path,
                request.query_string,
                content_type,
                body,
                payload=payload,
            )

        affinity_state = "disabled"
        if self._enable_request_affinity:
            pinned_idx = self._affinity_map.get(affinity_key) if affinity_key is not None else None
            if pinned_idx is None:
                affinity_state = "miss"
            elif self._has_capacity(pinned_idx):
                affinity_state = "hit"
            else:
                affinity_state = "fallback"

        idx = self._pick_backend(affinity_key=affinity_key)
        backend_url = self._backend_urls[idx]
        target = f"{backend_url}/{request.match_info['path']}"
        if request.query_string:
            target = f"{target}?{request.query_string}"

        schema_size = None
        if isinstance(payload, dict):
            data = payload.get("data")
            schema = None
            labels = None
            if isinstance(data, dict) and "schema" in data:
                schema = data.get("schema")
            elif isinstance(data, dict) and "labels" in data:
                labels = data.get("labels")
            elif "schema" in payload:
                schema = payload.get("schema")
            elif "labels" in payload:
                labels = payload.get("labels")
            schema_size = self._schema_request_size(schema)
            if schema_size is None and labels is not None:
                schema_size = self._schema_request_size({"entities": labels})

        self._record_routing_observability(idx, affinity_state)

        async with self._semaphores[idx]:
            try:
                async with self._session.request(
                    method=request.method,
                    url=target,
                    headers={
                        k: v
                        for k, v in request.headers.items()
                        if k.lower() not in ("host", "transfer-encoding")
                    },
                    data=body,
                ) as resp:
                    resp_body = await resp.read()
                    response = web.Response(
                        status=resp.status,
                        body=resp_body,
                        content_type=resp.content_type,
                    )
                    response.headers.update(
                        self._build_route_headers(idx, affinity_state, schema_size)
                    )
                    self._log_route_observability(
                        method=request.method,
                        path=request.path,
                        idx=idx,
                        backend_url=backend_url,
                        affinity_state=affinity_state,
                        schema_size=schema_size,
                        status=resp.status,
                    )
                    return response
            except Exception as exc:
                self._forget_affinity(affinity_key, idx)
                logger.error(f"Backend {idx} ({backend_url}) error: {exc}")
                response = web.Response(
                    status=502,
                    text=f'{{"error": "backend unavailable: {exc}"}}',
                    content_type="application/json",
                )
                response.headers.update(self._build_route_headers(idx, affinity_state, schema_size))
                self._log_route_observability(
                    method=request.method,
                    path=request.path,
                    idx=idx,
                    backend_url=backend_url,
                    affinity_state=affinity_state,
                    schema_size=schema_size,
                    status=502,
                )
                return response

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Aggregate health: healthy if at least one backend is reachable."""
        assert self._session is not None

        for url in self._backend_urls:
            try:
                async with self._session.get(
                    f"{url}/health", timeout=ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        return web.Response(text="OK", content_type="text/plain")
            except Exception:
                continue

        return web.Response(
            status=503,
            text='{"error": "no healthy backends"}',
            content_type="application/json",
        )

    async def _handle_stats(self, request: web.Request) -> web.Response:
        """Return dispatcher-local routing and capacity counters."""
        return web.json_response(self._snapshot_stats())
