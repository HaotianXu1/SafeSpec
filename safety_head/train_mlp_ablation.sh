#!/usr/bin/env bash
# 并行跑 MLP ablation：
#   第一种：mlp_hidden_ratio=1.0  （d → d → 1，全宽隐层）
#   第三种：mlp_hidden_ratio=0.25 （d → d/4 → 1）
# 二者除隐层宽度外其它超参相同；各占一对 GPU，可按机器改 GPU / hidden_layer。

set -euo pipefail
cd /path/to/safety_head_ckpts

export PYTHONUNBUFFERED=1

PY=python

DATA_JSONL=/path/to/safety_head_ckpts/train_data_qwen_4b/train_qwen3_4b_with_benign1k.jsonl
MODEL_PATH=Qwen/Qwen3-32B

# 与 layer ablation 对齐时可改成 4、8、16 等；默认最后一层
HIDDEN_LAYER_INDEX=-1

# 两路任务各用一对卡，勿与机器上其它占卡任务冲突
GPU_FULL=2,3
GPU_QUARTER=0,1

# 第一种：d → d → 1
# nohup $PY train.py \
#   --model_path "$MODEL_PATH" \
#   --data_dir "$DATA_JSONL" \
#   --save_dir safety_head_ckpt_mlp_r1p0 \
#   --hidden_layer_index "$HIDDEN_LAYER_INDEX" \
#   --mlp_hidden_ratio 1.0 \
#   --mlp_num_hidden_layers 1 \
#   --device_map auto \
#   --gpu "$GPU_FULL" \
#   > nohup_train_mlp_r1p0.log 2>&1 &

# 第三种：d → d/4 → 1
nohup $PY train.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_JSONL" \
  --save_dir safety_head_ckpt_mlp_r0p25 \
  --hidden_layer_index "$HIDDEN_LAYER_INDEX" \
  --mlp_hidden_ratio 0.25 \
  --mlp_num_hidden_layers 1 \
  --device_map auto \
  --gpu "$GPU_QUARTER" \
  > nohup_train_mlp_r0p25.log 2>&1 &

# echo "Started two jobs: mlp_r1p0   d→d→1   (GPU $GPU_FULL)   -> nohup_train_mlp_r1p0.log"
echo "                  mlp_r0p25 d→d/4→1 (GPU $GPU_QUARTER) -> nohup_train_mlp_r0p25.log"
