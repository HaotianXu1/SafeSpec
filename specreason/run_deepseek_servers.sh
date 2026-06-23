#!/bin/bash
# 起 DeepSeek 评测三件套 vLLM 服务（持久, nohup）：
#   base  70B  TP4  GPU0-3  :30007  (生成)
#   pool  70B  TP2  GPU4,5  :30009  deepseek70b-embed  (MEAN pooling, 给 safety head 出 hidden state)
#   draft 8B   TP1  GPU6    :30005  (draft)
set -u
DATA_DIR="/data/xuhaotian"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate python310
export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"; export NO_PROXY="$no_proxy"
export VLLM_CACHE_ROOT="$DATA_DIR/.cache/vllm" TORCHINDUCTOR_CACHE_DIR="$DATA_DIR/.cache/inductor" TRITON_CACHE_DIR="$DATA_DIR/.cache/triton" TMPDIR="$DATA_DIR/tmp"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR" run_logs
BASE=/data/LLM_models/DeepSeek-R1-Distill-Llama-70B
DRAFT=/data/LLM_models/DeepSeek-R1-Distill-Llama-8B
LOG=run_logs

health() { curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "http://localhost:$1/health" 2>/dev/null | grep -q 200; }

# GPU6/7 他人占用；布局: base 70B TP2(0,1) + draft 8B 独占 GPU3 + pool 70B TP2(4,5)
if ! health 30007; then
  echo "[base] launching 70B TP2 on GPU0,1 :30007"
  CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve "$BASE" --dtype auto --tensor-parallel-size 2 \
    --max-model-len 6144 --gpu-memory-utilization 0.95 --enable-prefix-caching --port 30007 \
    > "$LOG/ds_vllm_base.log" 2>&1 & echo "  pid=$!"
fi
if ! health 30005; then
  echo "[draft] launching 8B on GPU3 :30005"
  CUDA_VISIBLE_DEVICES=3 nohup vllm serve "$DRAFT" --dtype auto --tensor-parallel-size 1 \
    --max-model-len 8192 --gpu-memory-utilization 0.6 --enable-prefix-caching --port 30005 \
    > "$LOG/ds_vllm_draft.log" 2>&1 & echo "  pid=$!"
fi
if ! health 30009; then
  echo "[pool] launching 70B pooling TP2 on GPU4,5 :30009 (deepseek70b-embed)"
  CUDA_VISIBLE_DEVICES=4,5 nohup vllm serve "$BASE" --tensor-parallel-size 2 \
    --runner pooling --convert embed --pooler-config '{"pooling_type":"MEAN","normalize":false}' \
    --enforce-eager --max-model-len 2560 --gpu-memory-utilization 0.92 \
    --port 30009 --served-model-name deepseek70b-embed \
    > "$LOG/ds_vllm_pool.log" 2>&1 & echo "  pid=$!"
fi

echo "waiting for all three healthy..."
for i in $(seq 1 150); do
  ok=0
  for p in 30007 30005 30009; do health $p && ok=$((ok+1)); done
  echo "  [$i] healthy=$ok/3 ($(date '+%H:%M:%S'))"
  [ "$ok" -eq 3 ] && { echo "ALL READY"; break; }
  for p in 30007 30005 30009; do
    grep -qiE "No space left|out of memory|Engine core init|Failed core proc|ValueError|raise" "$LOG/ds_vllm_$([ $p = 30007 ] && echo base || ([ $p = 30005 ] && echo draft || echo pool)).log" 2>/dev/null && echo "  WARN err in port $p log"
  done
  sleep 6
done
