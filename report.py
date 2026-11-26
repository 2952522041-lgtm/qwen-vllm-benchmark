import json
import math
import sys
from pathlib import Path
from statistics import mean


def percentile95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def table(title, result):
    print(f"### {title}\n")
    print("| 并发 | TTFT 均值 / P95 (ms) | TPOT 均值 (ms/token) | 输出吞吐 (tokens/s) | 整卡峰值 (MiB) | 引擎排队均值 (ms) |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for g in result["groups"]:
        rows = g["requests"]
        ttft = [r["ttft_ms"] for r in rows]
        tpot = mean(r["tpot_ms"] for r in rows)
        rate = sum(r["output_tokens"] for r in rows) / g["wall_seconds"]
        peak = max(mib for _, mib in g["vram_samples_s_mib"])
        print(f"| {g['concurrency']} | {mean(ttft):.1f} / {percentile95(ttft):.1f} | "
              f"{tpot:.2f} | {rate:.1f} | {peak:.0f} | {g['server_means']['queue_ms']:.1f} |")
    print()


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    for phase, title in (("baseline", "基线 · FP16，最多 4 条序列"),
                         ("tuned", "调参后 · FP16，最多 16 条序列"),
                         ("awq", "量化后 · AWQ 4-bit，最多 16 条序列")):
        table(title, json.loads((folder / f"{phase}.json").read_text()))
