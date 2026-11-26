#!/usr/bin/env bash
set -euo pipefail
mode=${1:-baseline}
model=Qwen2.5-0.5B-Instruct
revision=7ae557604adf67be50417f59c2c2f167def9a775
seqs=4
case "$mode" in
    baseline) ;;
    tuned) seqs=16 ;;
    awq) seqs=16; model+=-AWQ; revision=db09cd27ead7fee40cdee309693cf83601b9c899 ;;
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
    --max-num-seqs "$seqs" --gpu-memory-utilization 0.55 --max-model-len 512 \
    --max-num-batched-tokens 2048 --no-enable-prefix-caching --generation-config vllm
