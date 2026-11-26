import asyncio
from time import perf_counter

from experiment import OUTPUT_TOKENS, payload, run, stream_events


async def chat(session, i):
    first = last = usage = None
    start = perf_counter()
    async with session.post("/v1/chat/completions", json=payload(i)) as response:
        async for event in stream_events(response):
            usage = event.get("usage") or usage
            if event["choices"] and event["choices"][0].get("token_ids"):
                last = perf_counter()
                first = last if first is None else first
    if first is None or not usage or usage["completion_tokens"] != OUTPUT_TOKENS:
        raise ValueError("missing token event or unexpected token count")
    return dict(id=i, prompt_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
                ttft_ms=(first - start) * 1000, tpot_ms=(last - first) * 1000 / (OUTPUT_TOKENS - 1))


async def batch(session, concurrency, requests):
    ids = iter(range(requests))

    async def worker():
        return [await chat(session, i) for i in ids]
    return await asyncio.gather(*(worker() for _ in range(concurrency)))

if __name__ == "__main__":
    asyncio.run(run(batch, (1, 4, 16)))
