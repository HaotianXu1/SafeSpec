#!/bin/bash
# 官方配置(32768 budget)重跑 gsm8k/math 的 target_only + speculative, 框架+分片并行加速。
# base 70B TP4(GPU0-3,:30007) + draft 8B(GPU4,:30005)。GPU5=他人(liyu25)不碰, GPU6,7=GUI-Owl不碰。
# 4 个任务顺序跑(各分片10并发,N=100), 复用 run_spec_official_parallel.sh。
set -u
cd /data/xuhaotian/specreason-origin
export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"; export NO_PROXY="$no_proxy"
export VLLM_CACHE_ROOT=/data/xuhaotian/.cache/vllm TORCHINDUCTOR_CACHE_DIR=/data/xuhaotian/.cache/inductor
export TRITON_CACHE_DIR=/data/xuhaotian/.cache/triton TMPDIR=/data/xuhaotian/tmp
VLLM=/data/xuhaotian/anaconda3/envs/python310/bin/vllm
BASE=/data/LLM_models/DeepSeek-R1-Distill-Llama-70B
DRAFT=/data/LLM_models/DeepSeek-R1-Distill-Llama-8B
L=run_logs; mkdir -p "$L" results
MAIN=$L/math_gsm8k_official_main.txt; : > "$MAIN"
say(){ echo "$*" | tee -a "$MAIN"; }
health(){ curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "http://localhost:$1/health" 2>/dev/null | grep -q 200; }

# 起服务(若未起)
if ! health 30007; then
  say "[起] base 70B TP4 GPU0-3 :30007"
  CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$VLLM" serve "$BASE" --tensor-parallel-size 4 \
    --max-model-len 33792 --gpu-memory-utilization 0.90 --max-num-seqs 64 \
    --enable-prefix-caching --port 30007 > "$L/mg_base_tp4.log" 2>&1 &
fi
if ! health 30005; then
  say "[起] draft 8B GPU4 :30005"
  CUDA_VISIBLE_DEVICES=4 nohup "$VLLM" serve "$DRAFT" --tensor-parallel-size 1 \
    --max-model-len 33792 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --port 30005 > "$L/mg_draft.log" 2>&1 &
fi
say "[等] base+draft 就绪..."
for _ in $(seq 1 180); do health 30007 && health 30005 && break; sleep 5; done
say "[就绪] 开始 4 任务 $(date '+%H:%M:%S')"

for DS in gsm8k math; do
  for MODE in target_only speculative; do
    OUT=results/${DS}_${MODE}_official
    say ">>> [$(date '+%H:%M:%S')] $DS / $MODE -> $OUT"
    t0=$(date +%s)
    DS=$DS MODE=$MODE OUT=$OUT TAG=${DS}_${MODE} STATUS=$L/${DS}_${MODE}_status.txt \
      bash run_spec_official_parallel.sh > "$L/${DS}_${MODE}_main.log" 2>&1
    say "    完成 $DS/$MODE  ${SECONDS}s起算 用时$(($(date +%s)-t0))s  题数=$(ls -d "$OUT"/*/ 2>/dev/null|wc -l)"
  done
done
say "[math_gsm8k_official ALL DONE $(date '+%m-%d %H:%M:%S')]"
