#!/bin/bash
# DeepSeek 70B/8B benign benchmark @ N=100 (0-99), aug6 head, safety_threshold=0.6, force_base_answer=0.
# 模式顺序按 GPU 需求降序, 跑完一阶段就释放不再需要的卡(共享机器友好):
#   Phase1 spec_ppl  (base+draft+pool, 5卡) -> 释放 pool(GPU3,4)
#   Phase2 speculative(base+draft, 3卡)      -> 释放 draft(GPU5)
#   Phase3 target_only(base, 2卡)
# spec_ppl 全新 0-99; speculative/target_only 复用 bench_ds 旧 0-49(同 force_base=0) + 补跑 50-99。
# draft 独占 GPU5 -> gpu-memory-utilization 0.9(原 0.6), 更大 KV cache + prefix-caching -> 更快。
set -u
cd /data/xuhaotian/specreason-origin
export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"; export NO_PROXY="$no_proxy"
export VLLM_CACHE_ROOT=/data/xuhaotian/.cache/vllm TORCHINDUCTOR_CACHE_DIR=/data/xuhaotian/.cache/inductor
export TRITON_CACHE_DIR=/data/xuhaotian/.cache/triton TMPDIR=/data/xuhaotian/tmp
PY=/data/xuhaotian/anaconda3/envs/python310/bin/python
VLLM=/data/xuhaotian/anaconda3/envs/python310/bin/vllm
BASE=/data/LLM_models/DeepSeek-R1-Distill-Llama-70B
DRAFT=/data/LLM_models/DeepSeek-R1-Distill-Llama-8B
HEAD=/data/xuhaotian/Safety_head/safety_head_ckpt_deepseek70b_layerlast_mix_aug6/safety_head_best.pt
THR=9; TB="${TB:-6144}"; MS="${MS:-128}"; STHR="${STHR:-0.6}"
OUT=results/bench_ds_n100; SRC=results/bench_ds; L=run_logs
STATUS=$L/bench_ds_n100_status.txt; mkdir -p "$OUT" "$L"; : > "$STATUS"
say(){ echo "$*" | tee -a "$STATUS"; }

health(){ curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "http://localhost:$1/health" 2>/dev/null | grep -q 200; }
wait_health(){ local p=$1 n=${2:-160}; for _ in $(seq 1 $n); do health "$p" && return 0; sleep 5; done; return 1; }
ensure_base(){ health 30007 && return 0; say "    [launch] base 70B TP2 GPU1,2 :30007"
  CUDA_VISIBLE_DEVICES=1,2 nohup "$VLLM" serve "$BASE" --dtype auto --tensor-parallel-size 2 \
    --max-model-len 12288 --gpu-memory-utilization 0.95 --enable-prefix-caching --port 30007 > "$L/ds_n100_base.log" 2>&1 &
  wait_health 30007 || say "    [WARN] base not healthy"; }
ensure_draft(){ health 30005 && return 0; say "    [launch] draft 8B GPU5 :30005 util0.9"
  CUDA_VISIBLE_DEVICES=5 nohup "$VLLM" serve "$DRAFT" --dtype auto --tensor-parallel-size 1 \
    --max-model-len 12288 --gpu-memory-utilization 0.9 --enable-prefix-caching --port 30005 > "$L/ds_n100_draft.log" 2>&1 &
  wait_health 30005 || say "    [WARN] draft not healthy"; }
ensure_pool(){ health 30009 && return 0; say "    [launch] pool 70B TP2 GPU3,4 :30009"
  CUDA_VISIBLE_DEVICES=3,4 nohup "$VLLM" serve "$BASE" --tensor-parallel-size 2 \
    --runner pooling --convert embed --pooler-config '{"pooling_type":"MEAN","normalize":false}' \
    --enforce-eager --max-model-len 2560 --gpu-memory-utilization 0.92 \
    --port 30009 --served-model-name deepseek70b-embed > "$L/ds_n100_pool.log" 2>&1 &
  wait_health 30009 || say "    [WARN] pool not healthy"; }
# 释放某实例: 先按端口杀父进程, 再按 nvidia-smi 杀这些卡上我自己的残留 worker(共享机器只杀自己)
free_instance(){ local port=$1; shift; local me=$(id -un)
  pkill -u "$(id -u)" -f "vllm serve.*--port $port" 2>/dev/null; sleep 3
  for g in "$@"; do
    for pid in $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
      [ -z "$pid" ] && continue
      [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" = "$me" ] && { kill -9 "$pid" 2>/dev/null; say "    [free] GPU$g killed my pid $pid"; }
    done
  done; sleep 5
  for g in "$@"; do say "    [free] GPU$g now $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader)"; done; }

run_range(){ local ds=$1 mode=$2 s=$3 e=$4; local log="$L/bench_ds_n100_${ds}_${mode}.log"
  if [ "$mode" = "spec_ppl" ]; then
    $PY spec_reason_ppl.py --dataset_name "$ds" --start_id $s --end_id $((e+1)) --num_repeats 1 \
      --score_threshold $THR --score_method greedy --token_budget $TB --max_steps $MS \
      --run_mode spec_ppl --force_base_answer 0 --base_port 30007 --draft_port 30005 \
      --base_model_path "$BASE" --draft_model_path "$DRAFT" --safety_head_path "$HEAD" --safety_model_path "$BASE" \
      --safety_backend vllm --safety_pool_port 30009 --safety_pool_model deepseek70b-embed \
      --safety_threshold $STHR --output_dir "$OUT/${ds}_spec_ppl" >> "$log" 2>&1
  else
    $PY spec_reason.py --dataset_name "$ds" --start_id $s --end_id $e \
      --score_threshold $THR --token_budget $TB --max_steps $MS --force_base_answer 0 \
      --run_mode "$mode" --model_key deepseek --base_port 30007 --draft_port 30005 \
      --base_model_path "$BASE" --draft_model_path "$DRAFT" --output_dir "$OUT/${ds}_${mode}" >> "$log" 2>&1
  fi; }
do_cell(){ local ds=$1 mode=$2 s=$3 e=$4; local cell="${ds}_${mode}"
  mkdir -p "$OUT/$cell"
  [ "$s" -gt 0 ] && cp -rn "$SRC/$cell/"* "$OUT/$cell/" 2>/dev/null
  say ">>> [$(date '+%m-%d %H:%M:%S')] $cell  run $s-$e $( [ $s -gt 0 ] && echo '(+reuse 0-49)' || echo '(fresh)')"
  local t0=$(date +%s); run_range "$ds" "$mode" "$s" "$e"; local rc=$?
  say "    done $cell rc=$rc $(($(date +%s)-t0))s  total=$(ls -d "$OUT/$cell"/*/ 2>/dev/null|wc -l)"; }

say "[bench_ds_n100] start $(date '+%m-%d %H:%M:%S') thr=$THR tb=$TB ms=$MS sthr=$STHR head=aug6 force_base=0 draft_util=0.9"
# 全程 5 卡(spec_ppl 每个数据集都用 pool, 无法中途释放)。按数据集顺序跑, 每跑完一个数据集就出完整三模式表。
ensure_base; ensure_draft; ensure_pool
# gpqa spec_ppl 复用上次 aug6 thr0.5 的 25 条(已验证 max safety 0.0058、0 fire,thr0.5≡thr0.6,force_base=0/TB/MS/头全同)→ 只补跑 25-99
mkdir -p "$OUT/gpqa_spec_ppl"; cp -rn results/bench_ds_v3/gpqa_spec_ppl_aug6/* "$OUT/gpqa_spec_ppl/" 2>/dev/null
say "[reuse] gpqa_spec_ppl 复用 aug6 thr0.5 的 $(ls -d $OUT/gpqa_spec_ppl/*/ 2>/dev/null|wc -l) 条(0-24)"
for ds in gpqa gsm8k math; do
  say "==== dataset: $ds ===="
  do_cell "$ds" target_only 50 99    # 复用 0-49 + 补跑 50-99
  do_cell "$ds" speculative 50 99    # 复用 0-49 + 补跑 50-99
  if [ "$ds" = gpqa ]; then          # 复用 0-24(已 copy) + 补跑 25-99
    say ">>> [$(date '+%m-%d %H:%M:%S')] gpqa_spec_ppl  run 25-99 (+reuse 0-24)"
    t0=$(date +%s); run_range gpqa spec_ppl 25 99; rc=$?
    say "    done gpqa_spec_ppl rc=$rc $(($(date +%s)-t0))s  total=$(ls -d $OUT/gpqa_spec_ppl/*/ 2>/dev/null|wc -l)"
  else
    do_cell "$ds" spec_ppl 0 99      # gsm8k/math 全新(无可复用的同配置 spec_ppl)
  fi
  say "==== $ds complete ===="
done
say "[bench_ds_n100 done $(date '+%m-%d %H:%M:%S')]"
