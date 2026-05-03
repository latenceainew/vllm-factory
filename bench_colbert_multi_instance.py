"""
Multi-instance ColBERT throughput (SciFact docs), mirroring bench_gliner2_512.py.

Targets LateOn / ModernColBERT-style models served with ``moderncolbert_io``,
``POST /pooling``, and ``task=token_embed``. Spawns N independent vLLM
``api_server`` processes with round-robin clients (same pattern as the GLiNER2
multi-instance bench).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

_REPO_ROOT = Path(__file__).resolve().parent


def _colbert_serve_flags(gpu_memory_utilization: float) -> list[str]:
    return [
        "--enforce-eager",
        "--dtype", "bfloat16",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", "8192",
        "--max-num-batched-tokens", "8192",
        "--disable-log-stats",
        "--uvicorn-log-level", "warning",
    ]


def start_instance(
    model_id: str,
    io_plugin: str,
    port: int,
    gpu_util: float,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--io-processor-plugin",
        io_plugin,
        "--port",
        str(port),
        "--trust-remote-code",
        *_colbert_serve_flags(gpu_util),
    ]
    env = os.environ.copy()
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    log_path = Path(f"/tmp/colbert_multi_bench_{port}.log")
    log_fh = open(log_path, "w")
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        cwd=str(_REPO_ROOT),
        start_new_session=True,
    )


def wait_instance_healthy(port: int, timeout: int = 600) -> bool:
    import requests

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def kill_all(procs: list[subprocess.Popen]):
    for proc in procs:
        if proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=5)
                except Exception:
                    pass
    time.sleep(2)


async def send_request(
    session: aiohttp.ClientSession,
    model_id: str,
    port: int,
    data: dict,
) -> float:
    body = {"model": model_id, "data": data, "task": "token_embed"}
    url = f"http://localhost:{port}/pooling"
    t0 = time.perf_counter()
    async with session.post(url, json=body) as resp:
        raw = await resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} on :{port}: {raw[:300]!r}")
    return (time.perf_counter() - t0) * 1000


async def warmup_requests(
    dataset: list[dict],
    n: int,
    ports: list[int],
    model_id: str,
) -> None:
    conn = aiohttp.TCPConnector(limit=64)
    async with aiohttp.ClientSession(connector=conn) as session:
        sem = asyncio.Semaphore(32)

        async def bounded(i: int) -> float:
            async with sem:
                port = ports[i % len(ports)]
                return await send_request(
                    session, model_id, port, dataset[i % len(dataset)]
                )

        await asyncio.gather(*[bounded(i) for i in range(n)])


async def run_saturate(
    dataset: list[dict],
    concurrency: int,
    num_reqs: int,
    ports: list[int],
    model_id: str,
) -> dict:
    conn = aiohttp.TCPConnector(limit=max(16, concurrency * 2))
    async with aiohttp.ClientSession(connector=conn) as session:
        sem = asyncio.Semaphore(concurrency)
        t0 = time.perf_counter()

        async def bounded(i: int) -> float:
            async with sem:
                port = ports[i % len(ports)]
                return await send_request(
                    session, model_id, port, dataset[i % len(dataset)]
                )

        latencies = await asyncio.gather(*[bounded(i) for i in range(num_reqs)])
        elapsed = time.perf_counter() - t0

    lat = sorted(latencies)
    n = len(lat)
    return {
        "req_per_s": num_reqs / elapsed,
        "p50_ms": lat[int(n * 0.50)],
        "p95_ms": lat[min(n - 1, int(n * 0.95))],
        "p99_ms": lat[min(n - 1, int(n * 0.99))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ColBERT multi-instance saturate bench")
    parser.add_argument(
        "--model",
        default="lightonai/LateOn",
        help="HF model id (default: lightonai/LateOn)",
    )
    parser.add_argument(
        "--io-processor-plugin",
        default="moderncolbert_io",
        help="IO processor plugin (default: moderncolbert_io)",
    )
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--first-port", type=int, default=9100)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Per-instance GPU memory fraction (default: 0.70 / num_instances)",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--num-requests", type=int, default=512)
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default="4,8,16,32,64,128",
        help="Comma-separated concurrency sweep",
    )
    args = parser.parse_args()

    if args.num_instances < 1:
        parser.error("num-instances must be >= 1")

    ports = [args.first_port + i for i in range(args.num_instances)]
    per_inst_gpu = args.gpu_memory_utilization
    if per_inst_gpu is None:
        per_inst_gpu = max(0.12, round(0.70 / args.num_instances, 2))

    sys.path.insert(0, str(_REPO_ROOT))
    from bench.registry import dataset_scifact_colbert

    dataset = dataset_scifact_colbert(512)
    conc_levels = [int(x.strip()) for x in args.concurrency_levels.split(",") if x.strip()]

    print("=" * 65)
    print("  ColBERT multi-instance throughput (SciFact docs)")
    print(f"  model={args.model!r}  io={args.io_processor_plugin!r}")
    print(
        f"  {args.num_instances} instances  ports={ports[0]}..{ports[-1]}  "
        f"gpu_util/instance={per_inst_gpu}"
    )
    print("=" * 65)

    print("\n[1/4] Dataset ready:", len(dataset), "docs")
    print("\n[2/4] Starting vLLM instances...")
    procs: list[subprocess.Popen] = []
    for port in ports:
        print(f"  Launching :{port} ...")
        procs.append(
            start_instance(args.model, args.io_processor_plugin, port, per_inst_gpu)
        )

    try:
        for port in ports:
            print(f"  Waiting for :{port} ...", end=" ", flush=True)
            if not wait_instance_healthy(port):
                print("FAILED")
                logp = Path(f"/tmp/colbert_multi_bench_{port}.log")
                if logp.exists():
                    print(logp.read_text(errors="replace")[-2500:])
                raise RuntimeError(f"Instance on port {port} failed to start")
            print("healthy")

        print(f"\n[3/4] Warmup ({args.warmup} requests, round-robin)...")
        asyncio.run(warmup_requests(dataset, args.warmup, ports, args.model))
        print("  Done.")

        print(
            f"\n[4/4] Saturate sweep ({args.num_requests} req/level, "
            f"round-robin across {args.num_instances} instances)"
        )
        print(f"{'Concurrency':>12} {'req/s':>10} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10}")
        print("-" * 58)

        best_rps = 0.0
        best_c = 0
        for c in conc_levels:
            result = asyncio.run(
                run_saturate(dataset, c, args.num_requests, ports, args.model)
            )
            rps = result["req_per_s"]
            print(
                f"{c:>12} {rps:>10.1f} {result['p50_ms']:>10.1f} "
                f"{result['p95_ms']:>10.1f} {result['p99_ms']:>10.1f}"
            )
            if rps > best_rps:
                best_rps = rps
                best_c = c

        print("-" * 58)
        print(f"\n  PEAK: {best_rps:.1f} req/s at concurrency={best_c}")
    finally:
        print("\nShutting down instances...")
        kill_all(procs)
        print("Done.")


if __name__ == "__main__":
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    os.chdir(_REPO_ROOT)
    main()
