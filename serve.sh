#!/usr/bin/env bash
set -euo pipefail
mode=${1:-baseline}
model=Qwen3-4B-FP8
revision=96b30dc13593a244a5e59e84687309f53c375cfa
seqs=4
case "$mode" in
    baseline) ;;
    tuned) seqs=16 ;;
    awq) seqs=16; model=Qwen3-4B-AWQ; revision=74d4bd2bd4bff9cafc9345221320bffb08b406a3 ;;
    *) echo "usage: bash serve.sh {baseline|tuned|awq}" >&2; exit 2 ;;
esac
source_model="Qwen/$model"
revision_args=(--revision "$revision")
if [[ -n "${MODEL_DIR:-}" ]]; then
    source_model="$MODEL_DIR/$model"
    revision_args=()
fi
export VLLM_NO_USAGE_STATS=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve "$source_model" "${revision_args[@]}" --served-model-name qwen-lab \
    --host 127.0.0.1 --port "${PORT:-8017}" --dtype half --enforce-eager \
    --max-num-seqs "$seqs" --gpu-memory-utilization 0.74 --max-model-len 512 \
    --max-num-batched-tokens 512 --no-enable-prefix-caching --generation-config vllm
