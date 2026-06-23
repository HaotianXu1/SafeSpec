#!/usr/bin/env bash
cd /path/to/safety_head_ckpts

export PYTHONUNBUFFERED=1

PY=python

DATA_JSONL=/path/to/safety_head_ckpts/train_data_qwen_4b/train_qwen3_4b_with_benign1k.jsonl

nohup $PY train.py \
  --model_path Qwen/Qwen3-32B \
  --data_dir "$DATA_JSONL" \
  --save_dir safety_head_ckpt_qwen32b_layer8 \
  --hidden_layer_index 8 \
  --device_map auto \
  --gpu 4,5 \
  > nohup_train_layer8.log 2>&1 &

nohup $PY train.py \
  --model_path Qwen/Qwen3-32B \
  --data_dir "$DATA_JSONL" \
  --save_dir safety_head_ckpt_qwen32b_layer4 \
  --hidden_layer_index 4 \
  --device_map auto \
  --gpu 6,7 \
  > nohup_train_layer4.log 2>&1 &

# nohup $PY train.py \
#   --model_path Qwen/Qwen3-32B \
#   --data_dir "$DATA_JSONL" \
#   --save_dir safety_head_ckpt_qwen32b_layer48 \
#   --hidden_layer_index 48 \
#   --device_map auto \
#   --gpu 6,7 \
#   > nohup_train_layer48.log 2>&1 &