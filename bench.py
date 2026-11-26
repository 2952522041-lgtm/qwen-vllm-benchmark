import asyncio
import json
from time import perf_counter

from experiment import OUTPUT_TOKENS, payload, run


async def chat(session, i):
    first = last = usage = None
    start = perf_counter()
    async with session.post("/v1/chat/completions", json=payload(i)) as response:
        response.raise_for_status()
        async for line in response.content:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise ValueError(event["error"])
            usage = event.get("usage") or usage
            if event["choices"] and event["choices"][0].get("token_ids"):
                last = perf_counter()
                first = last if first is None else first
        else:
            raise ValueError("incomplete SSE stream")
    if first is None or not usage or usage["completion_tokens"] != OUTPUT_TOKENS:
        raise ValueError("missing token event or unexpected token count")
    return dict(id=i, prompt_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
                ttft_ms=(first - start) * 1000, tpot_ms=(last - first) * 1000 / (OUTPUT_TOKENS - 1))


if __name__ == "__main__":
    asyncio.run(run(chat))
