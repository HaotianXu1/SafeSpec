#!/bin/bash
# speculative @ 官方配置(token_budget 32768, 不截断) + 分片并行加速。
# base 70B TP4(:30007) + draft 8B(:30005) 已由外部起好。
# 10 个 spec_reason.py 分片进程并发跑 gpqa 0-99(各10题), 共享同一组服务 -> vLLM 连续批处理提速。
set -u
cd /data/xuhaotian/specreason-origin
export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"; export NO_PROXY="$no_proxy"; export TMPDIR=/data/xuhaotian/tmp
PY=/data/xuhaotian/anaconda3/envs/python310/bin/python
BASE=/data/LLM_models/DeepSeek-R1-Distill-Llama-70B
DRAFT=/data/LLM_models/DeepSeek-R1-Distill-Llama-8B
OUT="${OUT:-results/gpqa_spec_official}"; mkdir -p "$OUT" run_logs
STATUS="${STATUS:-run_logs/spec_official_status.txt}"; : > "$STATUS"
TB="${TB:-32768}"; MS="${MS:-600}"; THR="${THR:-9}"; NSHARD=10; PERSHARD=10; TAG="${TAG:-official}"
DS="${DS:-gpqa}"; MODE="${MODE:-speculative}"; FB="${FB:-0}"
say(){ echo "$*" | tee -a "$STATUS"; }
health(){ curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "http://localhost:$1/health" 2>/dev/null | grep -q 200; }

say "[spec_official] 等 base(:30007)+draft(:30005) 就绪..."
until health 30007 && health 30005; do sleep 10; done
say "[spec_official] 服务就绪, 起 $NSHARD 分片并发 $(date '+%H:%M:%S')  DS=$DS MODE=$MODE TB=$TB MS=$MS thr=$THR force_base=$FB"

for i in $(seq 0 $((NSHARD-1))); do
  S=$((i*PERSHARD)); E=$((S+PERSHARD-1))
  $PY spec_reason.py --dataset_name "$DS" --start_id $S --end_id $E \
    --token_budget $TB --max_steps $MS --score_threshold $THR --force_base_answer $FB \
    --run_mode "$MODE" --model_key deepseek --base_port 30007 --draft_port 30005 \
    --base_model_path "$BASE" --draft_model_path "$DRAFT" \
    --output_dir "$OUT" > "run_logs/spec_${TAG}_shard_${i}.log" 2>&1 &
  say "  分片 $i: 题 $S-$E  pid $!"
done
say "[spec_official] 等所有分片完成..."
wait
say "[spec_official done $(date '+%m-%d %H:%M:%S')]  共完成 $(ls -d "$OUT"/*/ 2>/dev/null|wc -l) 题"
