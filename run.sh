#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PORT=${PORT:-8017}
out=${RESULTS_DIR:-.local/results}
mkdir -p "$out" .local
pid=""
stop_server() {
    if [[ -n "$pid" ]]; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        pid=""
    fi
}
trap stop_server EXIT
trap 'exit 130' INT TERM
if [[ $# -eq 0 ]]; then set -- baseline tuned awq; fi
for phase in "$@"; do
    case "$phase" in baseline|tuned|awq) ;; *) echo "unknown phase: $phase" >&2; exit 2 ;; esac
    if ss -Hltn "sport = :$PORT" | grep -q .; then
        echo "Port $PORT is already in use" >&2; exit 1
    fi
    log=".local/$phase.log"
    setsid bash serve.sh "$phase" >"$log" 2>&1 &
    pid=$!
    echo "Starting $phase, log in $log"
    for ((step=0; step<300; step++)); do
        if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null; then break; fi
        if ! kill -0 "$pid" 2>/dev/null; then tail -40 "$log" >&2; exit 1; fi
        sleep 2
    done
    bash smoke.sh > "$out/$phase-chat.json.tmp"
    python -c 'import json,sys; assert json.load(open(sys.argv[1]))["choices"][0]["message"]["content"].strip()' "$out/$phase-chat.json.tmp"
    mv "$out/$phase-chat.json.tmp" "$out/$phase-chat.json"
    python bench.py "$phase" "$out/$phase.json"
    grep -oE 'Model loading took [0-9.]+ GiB memory|Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens' \
        "$log" > "$out/$phase-engine.txt"
    stop_server
done
