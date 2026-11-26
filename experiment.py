import asyncio
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import aiohttp
import pynvml as nv
from prometheus_client.parser import text_string_to_metric_families

REQUESTS = 48
STAGES = ("queue", "prefill", "decode")


def payload(i):
    text = (f"Request {i:04d}. Explain how an inference server handles multiple "
            "users. Write a detailed answer in English using complete sentences.")
    return dict(model="qwen-lab", messages=[dict(role="user", content=text)],
                temperature=0, max_tokens=64, min_tokens=64, ignore_eos=True,
                stream=True, return_token_ids=True, stream_options=dict(include_usage=True))


async def metrics(session):
    async with session.get("/metrics") as response:
        response.raise_for_status()
        families = text_string_to_metric_families(await response.text())
        return {sample.name: sample.value for family in families for sample in family.samples
                if sample.name.startswith("vllm:request_")
                and sample.name.endswith(("_sum", "_count"))
                and sample.labels.get("model_name") == "qwen-lab"}


async def settled(session, before, n):
    for _ in range(100):
        after = await metrics(session)
        counts = [f"vllm:request_{stage}_time_seconds_count" for stage in STAGES]
        if all(after.get(k, 0) - before.get(k, 0) == n for k in counts):
            return after
        await asyncio.sleep(0.1)
    raise RuntimeError("server metrics do not match the number of completed requests")


async def run_group(chat, session, gpu, concurrency):
    before = await metrics(session)
    await asyncio.gather(*(chat(session, i) for i in range(concurrency)))
    before = await settled(session, before, concurrency)
    samples = []
    start = perf_counter()

    def sample():
        used = nv.nvmlDeviceGetMemoryInfo(gpu).used / 2**20
        samples.append([round(perf_counter() - start, 4), round(used, 2)])

    async def monitor():
        while True:
            sample()
            await asyncio.sleep(0.2)

    ids = iter(range(REQUESTS))

    async def worker():
        return [await chat(session, i) for i in ids]

    monitor_task = asyncio.create_task(monitor())
    try:
        batches = await asyncio.gather(*(worker() for _ in range(concurrency)))
        duration = perf_counter() - start
        sample()
    finally:
        monitor_task.cancel()
        await asyncio.gather(monitor_task, return_exceptions=True)
    after = await settled(session, before, REQUESTS)
    stages = {}
    for stage in STAGES:
        key = f"vllm:request_{stage}_time_seconds_sum"
        stages[stage + "_ms"] = (after[key] - before.get(key, 0)) * 1000 / REQUESTS
    rows = sorted((r for batch in batches for r in batch), key=lambda r: r["id"])
    return dict(concurrency=concurrency, wall_seconds=duration, requests=rows,
                vram_samples_s_mib=samples, server_means=stages)


async def run(chat):
    phase, destination = sys.argv[1:]
    if phase not in ("baseline", "tuned", "awq"):
        raise ValueError("unknown phase")
    nv.nvmlInit()
    try:
        gpu = nv.nvmlDeviceGetHandleByIndex(0)
        data = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(),
                    gpu=nv.nvmlDeviceGetName(gpu), driver=nv.nvmlSystemGetDriverVersion(),
                    kernel=platform.release(), python=platform.python_version(),
                    vllm=version("vllm"), torch=version("torch"), groups=[])
        url = "http://127.0.0.1:" + os.getenv("PORT", "8017")
        async with aiohttp.ClientSession(url, timeout=aiohttp.ClientTimeout(total=120)) as session:
            for concurrency in (1, 4, 16):
                result = await run_group(chat, session, gpu, concurrency)
                data["groups"].append(result)
                print(f"{phase} concurrency={concurrency} completed {REQUESTS} requests", flush=True)
    finally:
        nv.nvmlShutdown()
    out = Path(destination)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(out)
