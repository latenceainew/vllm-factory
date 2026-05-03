"""Benchmark orchestrator — runs sweep benchmarks across load patterns."""

from __future__ import annotations

import asyncio
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

from .registry import PluginEntry, get_entry
from .results import BenchResult, SweepPoint
from .vanilla_runners import get_runner

PORT = 9998
BASE_URL = f"http://localhost:{PORT}"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"
DEFAULT_SERVER_LOG_DIR = DEFAULT_OUTPUT_DIR / "server_logs"
DEFAULT_CONCURRENCY_LEVELS = [1, 4, 8, 16, 32, 64]
DEFAULT_MODES = ["saturate", "staggered"]
DEFAULT_STAGGERED_LOAD_FRACTION = 0.85


# ---------------------------------------------------------------------------
# Server lifecycle (adapted from scripts/serve_parity_test.py)
# ---------------------------------------------------------------------------

def start_server(
    entry: PluginEntry,
    log_dir: str | Path = DEFAULT_SERVER_LOG_DIR,
) -> tuple[subprocess.Popen, Path]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", entry.model_id,
        "--io-processor-plugin", entry.io_plugin,
        "--port", str(PORT),
        "--trust-remote-code",
    ]
    cmd += entry.serve_flags

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    log_path = log_dir / f"{entry.plugin_name}_{ts}.log"
    log_fh = open(log_path, "w")

    env = os.environ.copy()
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        env=env, text=True, start_new_session=True,
    )
    return proc, log_path


def wait_healthy(timeout: int = 240) -> bool:
    import requests as req
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = req.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                return True
        except req.ConnectionError:
            pass
        time.sleep(3)
    return False


def kill_server(proc: subprocess.Popen):
    if proc.poll() is None:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5)
    time.sleep(2)


def _read_log_tail(path: str | Path, max_chars: int = 4000) -> str:
    path = Path(path)
    if not path.exists():
        return "(no server log found)"
    text = path.read_text(errors="replace")
    return text[-max_chars:]


# ---------------------------------------------------------------------------
# Async HTTP load test
# ---------------------------------------------------------------------------

async def _send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    payload_data,
    payload_key: str = "data",
    request_task: str | None = None,
) -> float:
    body = {
        "model": model,
        payload_key: payload_data,
    }
    if request_task is not None:
        body["task"] = request_task
    start = time.perf_counter()
    async with session.post(url, json=body) as resp:
        raw = await resp.read()
        if resp.status != 200:
            raise RuntimeError(
                f"HTTP {resp.status} from {url}: {raw[:300]}"
            )
        if raw[:10].startswith(b'{"error"'):
            raise RuntimeError(
                f"Server returned error from {url}: {raw[:300]}"
            )
    return (time.perf_counter() - start) * 1000


def _latency_summary(latencies: list[float]) -> dict:
    latencies = sorted(latencies)
    n = len(latencies)
    if n == 0:
        return {
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "mean_ms": 0.0,
        }
    return {
        "p50_ms": latencies[min(n - 1, int(n * 0.50))],
        "p95_ms": latencies[min(n - 1, int(n * 0.95))],
        "p99_ms": latencies[min(n - 1, int(n * 0.99))],
        "mean_ms": statistics.mean(latencies),
    }


async def _warmup_vllm(
    entry: PluginEntry,
    dataset: list,
    concurrency: int,
    warmup: int,
) -> None:
    url = f"{BASE_URL}{entry.endpoint}"
    pk = entry.payload_key
    request_task = entry.request_task

    connector = aiohttp.TCPConnector(limit=max(8, concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"  [vLLM] Warming up ({warmup} requests)...")
        sem = asyncio.Semaphore(max(1, concurrency))

        async def bounded(i: int) -> float:
            async with sem:
                return await _send_request(
                    session,
                    url,
                    entry.model_id,
                    dataset[i % len(dataset)],
                    pk,
                    request_task=request_task,
                )

        warmup_tasks = [
            bounded(i)
            for i in range(warmup)
        ]
        await asyncio.gather(*warmup_tasks)


async def _run_vllm_saturate(
    entry: PluginEntry,
    dataset: list,
    num_requests: int,
    concurrency: int,
) -> dict:
    url = f"{BASE_URL}{entry.endpoint}"
    pk = entry.payload_key
    request_task = entry.request_task

    connector = aiohttp.TCPConnector(limit=max(8, concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)
        start = time.perf_counter()

        async def bounded(data) -> float:
            async with sem:
                return await _send_request(
                    session,
                    url,
                    entry.model_id,
                    data,
                    pk,
                    request_task=request_task,
                )

        latencies = await asyncio.gather(
            *[bounded(dataset[i % len(dataset)]) for i in range(num_requests)]
        )
        elapsed = time.perf_counter() - start

    return {
        "req_per_s": num_requests / elapsed if elapsed > 0 else 0.0,
        **_latency_summary(latencies),
    }


def _poisson_arrival_offsets(
    num_requests: int,
    arrival_rate_rps: float,
    seed: int = 42,
) -> list[float]:
    if num_requests <= 0:
        return []
    rng = random.Random(seed)
    offsets = [0.0]
    elapsed = 0.0
    for _ in range(1, num_requests):
        elapsed += rng.expovariate(arrival_rate_rps)
        offsets.append(elapsed)
    return offsets


async def _run_vllm_staggered(
    entry: PluginEntry,
    dataset: list,
    num_requests: int,
    concurrency: int,
    arrival_rate_rps: float,
) -> dict:
    url = f"{BASE_URL}{entry.endpoint}"
    pk = entry.payload_key
    request_task = entry.request_task

    connector = aiohttp.TCPConnector(limit=max(8, concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)
        offsets = _poisson_arrival_offsets(num_requests, arrival_rate_rps)
        start = time.perf_counter()

        async def scheduled_send(i: int, offset_s: float) -> float:
            await asyncio.sleep(offset_s)
            async with sem:
                return await _send_request(
                    session,
                    url,
                    entry.model_id,
                    dataset[i % len(dataset)],
                    pk,
                    request_task=request_task,
                )

        latencies = await asyncio.gather(
            *[scheduled_send(i, offset) for i, offset in enumerate(offsets)]
        )
        elapsed = time.perf_counter() - start

    return {
        "req_per_s": num_requests / elapsed if elapsed > 0 else 0.0,
        **_latency_summary(latencies),
        "target_arrival_rps": arrival_rate_rps,
    }


# ---------------------------------------------------------------------------
# Vanilla baseline measurement
# ---------------------------------------------------------------------------

_OOM_SENTINEL = "__OOM__"


def _is_cuda_oom(exc: BaseException) -> bool:
    """Detect CUDA out-of-memory errors across torch versions."""
    import torch
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _oom_result() -> dict:
    """Return a sentinel result for OOM batch sizes."""
    return {
        "req_per_s": 0.0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "p99_ms": 0.0,
        "mean_ms": 0.0,
        _OOM_SENTINEL: True,
    }


def _run_vanilla_baseline(
    entry: PluginEntry,
    dataset: list,
    num_requests: int,
    batch_size: int,
    n_warmup: int = 3,
    runner=None,
) -> dict:
    owns_runner = runner is None
    if owns_runner:
        vanilla_model_id = entry.vanilla_kwargs.get("hf_model_id", entry.model_id)
        extra_kwargs = {k: v for k, v in entry.vanilla_kwargs.items() if k != "hf_model_id"}
        runner = get_runner(entry.vanilla_family, vanilla_model_id, **extra_kwargs)

    try:
        batch_size = max(1, batch_size)
        n_batches = max(1, (num_requests + batch_size - 1) // batch_size)
        cursor = 0

        def next_batch() -> list:
            nonlocal cursor
            batch = []
            for _ in range(batch_size):
                batch.append(dataset[cursor % len(dataset)])
                cursor += 1
            return batch

        try:
            print(f"  [Vanilla] Warming up ({n_warmup} batches of {batch_size})...")
            for _ in range(n_warmup):
                runner.run(next_batch(), n_warmup=0, n_runs=1)

            print(f"  [Vanilla] Timed run ({n_batches} batches of {batch_size})...")
            latencies = []
            for _ in range(n_batches):
                batch = next_batch()
                t0 = time.perf_counter()
                runner.run(batch, n_warmup=0, n_runs=1)
                elapsed = (time.perf_counter() - t0) * 1000
                per_req = elapsed / batch_size
                latencies.extend([per_req] * batch_size)

            total_elapsed_s = sum(latencies) / 1000
            total_reqs = n_batches * batch_size

            return {
                "req_per_s": total_reqs / total_elapsed_s if total_elapsed_s > 0 else 0,
                **_latency_summary(latencies),
            }
        except Exception as exc:
            if _is_cuda_oom(exc):
                import torch
                torch.cuda.empty_cache()
                return _oom_result()
            raise
    finally:
        if owns_runner:
            runner.cleanup()


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_gpu() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    plugin_name: str,
    num_requests: int = 500,
    concurrency_levels: list[int] | None = None,
    modes: list[str] | None = None,
    warmup: int = 100,
    seq_len: int | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    staggered_load_fraction: float = DEFAULT_STAGGERED_LOAD_FRACTION,
) -> BenchResult:
    entry = get_entry(plugin_name)
    if seq_len is not None:
        entry.seq_len = seq_len

    concurrency_levels = concurrency_levels or DEFAULT_CONCURRENCY_LEVELS
    concurrency_levels = sorted({max(1, int(level)) for level in concurrency_levels})
    modes = [mode.lower() for mode in (modes or DEFAULT_MODES)]
    valid_modes = {"saturate", "staggered"}
    unknown_modes = [mode for mode in modes if mode not in valid_modes]
    if unknown_modes:
        raise ValueError(f"Unknown modes: {unknown_modes}. Expected one of {sorted(valid_modes)}")

    warmup = max(warmup, 3)

    dataset = entry.get_dataset()
    gpu = _detect_gpu()
    reference_model_id = entry.vanilla_kwargs.get("hf_model_id", entry.model_id)

    from vllm_factory.compat.vllm_capabilities import detect as _detect_caps
    caps = _detect_caps()
    vllm_version = caps.version or "unknown"
    compat_mode = "native"

    print(f"\n{'='*70}")
    print(f"  BENCHMARK: {plugin_name}")
    print(f"  Model:     {reference_model_id}")
    print(f"  Served:    {entry.model_id}")
    print(f"  GPU:       {gpu}")
    print(f"  vLLM:      {vllm_version}")
    print(f"  Compat:    {compat_mode}")
    print(f"  Requests:  {num_requests}")
    print(f"  Levels:    {concurrency_levels}")
    print(f"  Modes:     {modes}")
    print(f"  Seq len:   {entry.seq_len}")
    print(f"  Warmup:    {warmup}")
    print(f"{'='*70}\n")

    # Phase 0: Model preparation (if needed)
    if entry.prep_fn is not None:
        print("[Phase 0] Preparing model...")
        entry.prep_fn()

    # Phase 1: vLLM server benchmark
    print("[Phase 1] Starting vLLM server...")
    proc, server_log_path = start_server(entry)
    vllm_metrics: dict[tuple[str, int], dict] = {}
    try:
        if not wait_healthy():
            raise RuntimeError(
                "vLLM server failed to start. "
                f"Server log tail:\n{_read_log_tail(server_log_path, max_chars=2500)}"
            )
        print("  Server healthy.")

        asyncio.run(
            _warmup_vllm(
                entry,
                dataset,
                concurrency=max(concurrency_levels),
                warmup=warmup,
            )
        )

        for concurrency in concurrency_levels:
            print(f"\n  [vLLM] Concurrency {concurrency}")
            saturate_metrics = asyncio.run(
                _run_vllm_saturate(entry, dataset, num_requests, concurrency)
            )
            if "saturate" in modes:
                vllm_metrics[("saturate", concurrency)] = saturate_metrics
                print(
                    f"    saturate  req/s={saturate_metrics['req_per_s']:.1f}  "
                    f"p50={saturate_metrics['p50_ms']:.1f}ms  "
                    f"p99={saturate_metrics['p99_ms']:.1f}ms"
                )

            if "staggered" in modes:
                arrival_rate_rps = max(
                    0.1,
                    saturate_metrics["req_per_s"] * staggered_load_fraction,
                )
                staggered_metrics = asyncio.run(
                    _run_vllm_staggered(
                        entry,
                        dataset,
                        num_requests,
                        concurrency,
                        arrival_rate_rps,
                    )
                )
                vllm_metrics[("staggered", concurrency)] = staggered_metrics
                print(
                    f"    staggered req/s={staggered_metrics['req_per_s']:.1f}  "
                    f"p50={staggered_metrics['p50_ms']:.1f}ms  "
                    f"p99={staggered_metrics['p99_ms']:.1f}ms  "
                    f"target={arrival_rate_rps:.1f} rps"
                )
    finally:
        print("  Killing vLLM server...")
        kill_server(proc)

    # Phase 2: Vanilla baseline (reuse single runner for all batch sizes)
    print("\n[Phase 2] Running vanilla baseline...")
    vanilla_model_id = entry.vanilla_kwargs.get("hf_model_id", entry.model_id)
    extra_kwargs = {k: v for k, v in entry.vanilla_kwargs.items() if k != "hf_model_id"}
    shared_runner = get_runner(entry.vanilla_family, vanilla_model_id, **extra_kwargs)
    vanilla_metrics: dict[int, dict] = {}
    last_good: dict | None = None
    try:
        if entry.vanilla_batch_size is not None:
            metrics = _run_vanilla_baseline(
                entry,
                dataset,
                num_requests,
                batch_size=entry.vanilla_batch_size,
                runner=shared_runner,
            )
            for concurrency in concurrency_levels:
                vanilla_metrics[concurrency] = dict(metrics)
                if metrics.get(_OOM_SENTINEL):
                    print(
                        f"  [Vanilla@{concurrency}] OOM — batch size too large for GPU"
                    )
                else:
                    print(
                        f"  [Vanilla@{concurrency}] req/s={metrics['req_per_s']:.1f}  "
                        f"p50={metrics['p50_ms']:.1f}ms  "
                        f"p99={metrics['p99_ms']:.1f}ms  "
                        f"(fixed batch size {entry.vanilla_batch_size})"
                    )
        else:
            for concurrency in concurrency_levels:
                metrics = _run_vanilla_baseline(
                    entry,
                    dataset,
                    num_requests,
                    batch_size=concurrency,
                    runner=shared_runner,
                )
                vanilla_metrics[concurrency] = metrics
                if metrics.get(_OOM_SENTINEL):
                    print(
                        f"  [Vanilla@{concurrency}] OOM — batch size too large for GPU"
                    )
                    if last_good is not None:
                        vanilla_metrics[concurrency] = {
                            k: v for k, v in last_good.items() if k != _OOM_SENTINEL
                        }
                else:
                    last_good = metrics
                    print(
                        f"  [Vanilla@{concurrency}] req/s={metrics['req_per_s']:.1f}  "
                        f"p50={metrics['p50_ms']:.1f}ms  "
                        f"p99={metrics['p99_ms']:.1f}ms"
                    )
    finally:
        shared_runner.cleanup()

    # Phase 3: Compute factors
    parity_score = _known_parity(plugin_name)
    sweeps: list[SweepPoint] = []
    for concurrency in concurrency_levels:
        vanilla = vanilla_metrics[concurrency]
        for mode in modes:
            vllm = vllm_metrics.get((mode, concurrency))
            if vllm is None:
                continue

            throughput_factor = (
                vllm["req_per_s"] / vanilla["req_per_s"]
                if vanilla["req_per_s"] > 0 else 0.0
            )
            latency_factor = (
                vanilla["p50_ms"] / vllm["p50_ms"]
                if vllm["p50_ms"] > 0 else 0.0
            )
            sweeps.append(
                SweepPoint(
                    mode=mode,
                    concurrency=concurrency,
                    target_arrival_rps=(
                        round(vllm["target_arrival_rps"], 2)
                        if vllm.get("target_arrival_rps") is not None
                        else None
                    ),
                    vllm_req_per_s=round(vllm["req_per_s"], 1),
                    vllm_p50_ms=round(vllm["p50_ms"], 1),
                    vllm_p95_ms=round(vllm["p95_ms"], 1),
                    vllm_p99_ms=round(vllm["p99_ms"], 1),
                    vanilla_req_per_s=round(vanilla["req_per_s"], 1),
                    vanilla_p50_ms=round(vanilla["p50_ms"], 1),
                    vanilla_p95_ms=round(vanilla["p95_ms"], 1),
                    vanilla_p99_ms=round(vanilla["p99_ms"], 1),
                    throughput_factor=round(throughput_factor, 2),
                    latency_factor=round(latency_factor, 2),
                )
            )

    result = BenchResult(
        plugin=plugin_name,
        model_id=reference_model_id,
        served_model_id=entry.model_id,
        gpu=gpu,
        seq_len=entry.seq_len,
        num_requests=num_requests,
        concurrency_levels=concurrency_levels,
        modes=modes,
        sweeps=sweeps,
        parity_metric=entry.parity_metric,
        parity_score=parity_score,
        dataset_label=entry.dataset_label,
        vllm_version=vllm_version,
        compat_mode=compat_mode,
    )

    path = result.save(output_dir)
    print("\n[Phase 3] Summary")
    for mode in modes:
        best = result.best_sweep(mode)
        if best is None:
            continue
        print(
            f"  Best {mode:9s} throughput={best.throughput_factor:.1f}x  "
            f"latency={best.latency_factor:.1f}x  "
            f"at level={best.concurrency}"
        )
    print(f"  Parity ({entry.parity_metric}): {parity_score}")
    print(f"  Result saved to {path}")
    print(f"  Server log: {server_log_path}")

    return result


def _known_parity(plugin_name: str) -> float:
    """Return CI-validated parity scores from the README."""
    scores = {
        "embeddinggemma": 1.0000,
        "moderncolbert": 0.9700,
        "colbert_zero": 0.9700,
        "lfm2_colbert": 1.0000,
        "colqwen3": 0.9966,
        "collfm2": 0.9996,
        "nemotron_colembed": 0.9997,
        "mmbert_gliner": 1.000,
        "mt5_gliner": 1.000,
        "deberta_gliner2": 1.000,
        "deberta_gliner_linker": 1.0000,
        "modernbert_gliner_rerank": 1.0000,
        "lateon": 0.9700,
    }
    return scores.get(plugin_name, 0.0)
