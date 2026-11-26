#!/usr/bin/env bash
set -euo pipefail
curl --fail-with-body -sS --max-time 90 "http://127.0.0.1:${PORT:-8017}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen-lab","messages":[{"role":"user","content":"用一句中文解释什么是首 token 延迟。"}],"temperature":0,"max_tokens":80,"chat_template_kwargs":{"enable_thinking":false}}'
