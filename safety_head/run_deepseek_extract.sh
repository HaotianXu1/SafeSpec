#!/bin/bash
# 抽 DeepSeek-R1-Distill-Llama-70B(base) 最后一层 mask-mean-pool 特征，缓存到磁盘。
# 数据并行 2 shard × 3 GPU(device_map auto, 流水线切分 70B; 3卡留足激活+hidden states余量防OOM)。
# 用法: bash run_deepseek_extract.sh <shard_id> <gpus> <num_shards>
set -u
cd /data/xuhaotian/Safety_head
export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"; export NO_PROXY="$no_proxy"
export TMPDIR=/data/xuhaotian/tmp
export HF_HOME=/data/xuhaotian/.cache/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
PY=/data/xuhaotian/anaconda3/envs/python310/bin/python
SHARD=$1; GPUS=$2; NSHARDS=${3:-2}
$PY extract_features.py \
  --model_path /data/LLM_models/DeepSeek-R1-Distill-Llama-70B \
  --train_jsonl /data/xuhaotian/Safety_head/mix_data/train_mix_deepseek70b.jsonl \
  --ood_jsonl  /data/xuhaotian/Safety_head/mix_data/ood_eval_deepseek70b.jsonl \
  --out_path   /data/xuhaotian/Safety_head/mix_data/features_deepseek70b.pt \
  --extract_layers -1 --val_ratio 0.08 --seed 42 \
  --batch_size 2 --max_length 2560 --num_workers 2 \
  --gpu "$GPUS" --device_map auto --torch_dtype bfloat16 \
  --shard_id "$SHARD" --num_shards "$NSHARDS"
