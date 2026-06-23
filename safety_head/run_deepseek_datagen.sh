#!/bin/bash
# DeepSeek-R1-Distill 安全头数据生成（draft=8B 生成 CoT, Qwen3Guard-8B 打标）。
# 3 条 GPU lane 并行：A(0,1)=benign_local→jb_ood, B(2,3)=math, C(4,5)=jb_train。
# 与 qwen3 流程一致，只换生成模型。日志在 run_logs/ds_*。
set -u
cd /data/xuhaotian/Safety_head
export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"; export NO_PROXY="$no_proxy"
export TMPDIR=/data/xuhaotian/tmp; mkdir -p "$TMPDIR" run_logs mix_data_ds
export VLLM_CACHE_ROOT=/data/xuhaotian/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/data/xuhaotian/.cache/inductor
export TRITON_CACHE_DIR=/data/xuhaotian/.cache/triton
export HF_HOME=/data/xuhaotian/.cache/hf
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME"
export TOKENIZERS_PARALLELISM=false
PY=/data/xuhaotian/anaconda3/envs/python310/bin/python
DRAFT=/data/LLM_models/DeepSeek-R1-Distill-Llama-8B
GUARD=/data/xuhaotian/model/Qwen3Guard-Gen-8B
NEUTRAL_SYS="You are a reasoning model. Always reason step by step inside <think>...</think>. In your <think> process, reflect whether the user request or your reasoning is harmful, illegal, or unsafe. If harmful, stop pursuing it and output a polite refusal instead of harmful content."
echo "datagen start $(date '+%m-%d %H:%M:%S')"

# ---- Lane A (GPU 0,1): benign_local, then jb_ood_3methods ----
laneA () {
  $PY benign_local_pipeline.py --model_path "$DRAFT" --gpus 0,1 --tensor_parallel_size 2 \
    --sources gsm8k,gpqa,hellaswag --prompts_per_source 120 --final_benign_rows 1200 \
    --gpu_memory_utilization 0.85 --no_merge > run_logs/ds_benign_local.log 2>&1
  $PY ood_pipeline.py --target_model_path "$DRAFT" --guard_model_path "$GUARD" \
    --methods ABJ Flip SoP --samples_per_method 60 \
    --prompt_dir /data/xuhaotian/jailbreak_prompt \
    --output_root /data/xuhaotian/Safety_head/ood_data --run_name jb_ood_3methods_ds8b \
    --gen_gpus 0,1 --gen_tensor_parallel_size 2 --label_gpu 0,1 --label_tensor_parallel_size 2 \
    --system_prompt "$NEUTRAL_SYS" > run_logs/ds_jb_ood.log 2>&1
}

# ---- Lane B (GPU 2,3): math train+ood ----
laneB () {
  $PY gen_math_benign.py --model_path "$DRAFT" --gpus 2,3 --tensor_parallel_size 2 \
    --out_dir /data/xuhaotian/Safety_head/mix_data_ds --n_train 250 --n_ood 80 \
    --gpu_memory_utilization 0.85 > run_logs/ds_math.log 2>&1
}

# ---- Lane C (GPU 4,5): jb_train_4methods ----
laneC () {
  $PY ood_pipeline.py --target_model_path "$DRAFT" --guard_model_path "$GUARD" \
    --methods Attentionshift CodeChameleon mousetrap renellm --samples_per_method 80 \
    --prompt_dir /data/xuhaotian/jailbreak_prompt \
    --output_root /data/xuhaotian/Safety_head/ood_data --run_name jb_train_4methods_ds8b \
    --gen_gpus 4,5 --gen_tensor_parallel_size 2 --label_gpu 4,5 --label_tensor_parallel_size 2 \
    --system_prompt "$NEUTRAL_SYS" > run_logs/ds_jb_train.log 2>&1
}

laneA & PA=$!
laneB & PB=$!
laneC & PC=$!
wait $PA; echo "laneA done rc=$? $(date '+%H:%M:%S')"
wait $PB; echo "laneB done rc=$? $(date '+%H:%M:%S')"
wait $PC; echo "laneC done rc=$? $(date '+%H:%M:%S')"
echo "[datagen done $(date '+%m-%d %H:%M:%S')]"
