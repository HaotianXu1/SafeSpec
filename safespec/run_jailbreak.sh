#!/bin/bash
#
# 用法（无需改文件，全部用环境变量 / 末尾参数覆盖）：
#   METHODS=CodeChameleon bash run_jailbreak.sh                 # 换越狱方法（多个: METHODS="ABJ CodeChameleon"）
#   RUN_MODE=speculative bash run_jailbreak.sh                  # 换模式: spec_ppl(默认,带安全头) / speculative(原型) / target_only / draft_only
#   VLLM_BASE_GPUS=1,7 VLLM_SMALL_GPUS=3 SAFETY_GPUS=6 TP_SIZE_BASE=2 bash run_jailbreak.sh   # 指定GPU
#   bash run_jailbreak.sh --max_prompts_per_method 10          # 末尾参数原样透传给 run_jailbreak_pipeline.py（冒烟/限量）
#   SAFETY_THRESHOLD=0.6 bash run_jailbreak.sh                  # 调安全阈值
# 结果写到 ${JSON_OUTPUT_DIR}/${RUN_MODE}/<方法名>.json，每条含 judge_results.verdict(safe/jailbreak)
#
# 让本地 vLLM(base/draft) 调用绕过 http_proxy（机器上设了 127.0.0.1:7897 代理）；
# 外部 judge(gpt-5.1) 不在 no_proxy 列表里，仍走代理联网。
export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

# --- 显卡分配配置 ---
# 当前配置：target=Qwen3-32B(base)，draft=Qwen3-1.7B，safety=Qwen3-32B + 最后一层 head
VLLM_BASE_GPUS="${VLLM_BASE_GPUS:-0,1}"   # 32B base：需 TP>=2
VLLM_SMALL_GPUS="${VLLM_SMALL_GPUS:-2}"   # 1.7B draft
SAFETY_GPUS="${SAFETY_GPUS:-3}"           # safety head backbone(32B) 单卡推理

# 设置基础目录 (repo 内路径用脚本所在目录; 模型/数据路径可用环境变量覆盖)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/data/xuhaotian}"
# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-python310}"
conda activate "$CONDA_ENV"

# 配置：Qwen3-32B + Qwen3-1.7B
BASE_MODEL_NAME="${BASE_MODEL_NAME:-/data/xuhaotian/model/Qwen3-32B}"
SMALL_MODEL_NAME="${SMALL_MODEL_NAME:-/data/xuhaotian/model/Qwen3-1.7B}"

# 短标签：仅用于结果目录命名（与真实路径无关，可自改）
BASE_MODEL_TAG="${BASE_MODEL_TAG:-qwen3-32b}"
SMALL_MODEL_TAG="${SMALL_MODEL_TAG:-qwen3-1.7b}"

LOGFILE_DIR="${LOGFILE_DIR:-$SCRIPT_DIR/run_logs}"
TP_SIZE_BASE="${TP_SIZE_BASE:-2}"
TP_SIZE_SMALL="${TP_SIZE_SMALL:-1}"
PORT_BASE="${PORT_BASE:-30007}"
PORT_SMALL="${PORT_SMALL:-30005}"
BASE_MAX_MODEL_LEN="${BASE_MAX_MODEL_LEN:-8192}"
BASE_GPU_MEMORY_UTILIZATION="${BASE_GPU_MEMORY_UTILIZATION:-0.6}"
SMALL_MAX_MODEL_LEN="${SMALL_MAX_MODEL_LEN:-8192}"
SMALL_GPU_MEMORY_UTILIZATION="${SMALL_GPU_MEMORY_UTILIZATION:-0.25}"
# Safety head：默认用混合数据训练的 Qwen3-32B 最后一层 head
MODEL_KEY="${MODEL_KEY:-qwen3}"
SAFETY_MODEL_PATH="${SAFETY_MODEL_PATH:-/data/xuhaotian/model/Qwen3-32B}"
SAFETY_HEAD_PATH="${SAFETY_HEAD_PATH:-${DATA_DIR}/Safety_head/safety_head_ckpt_qwen32b_layerlast_mix_aug2/safety_head_best.pt}"
JSON_OUTPUT_DIR="${JSON_OUTPUT_DIR:-$SCRIPT_DIR/results/jailbreak/${BASE_MODEL_TAG}_lastlayer_mix_${SMALL_MODEL_TAG}}"
# 实际写入路径为：${JSON_OUTPUT_DIR}/${RUN_MODE}/<方法名>.json

# 新 head 接最后一层（ckpt 内 hidden_layer_index=-1）；绝不能传 --safety_hidden_layer 覆盖
SAFETY_HIDDEN_EXTRA=()
BASE_VLLM_EXTRA_ARGS=()
SMALL_VLLM_EXTRA_ARGS=()
# 运行参数
RUN_MODE="${RUN_MODE:-spec_ppl}"  # 可选: speculative, spec_ppl, draft_only, target_only, transformers_only, psr, safedecoding, secdecoding
THRESHOLD=9          # PPL/spec 接受阈值 (score_threshold)
SAFETY_THRESHOLD="${SAFETY_THRESHOLD:-0.5}" # Safety head：分数 >= 此值判为 unsafe（新 head 推荐 0.5）
METHODS_ARGS=(--methods ${METHODS:-ABJ})
SEC_ARGS=""
if [ "$RUN_MODE" == "secdecoding" ]; then
    SEC_ARGS="--sec_large_model_path $BASE_MODEL_NAME --sec_small_base_path $SMALL_MODEL_NAME --sec_small_expert_path $SMALL_MODEL_NAME --sec_small_expert_lora /data/xuhaotian/SafeDecoding-main/lora_modules/deepseek_8b --sec_alpha 1.2 --sec_first_n_tokens 100 --sec_signal_stride 1 --sec_unsafe_threshold 0.15"
fi
PIPELINE_GPUS="$SAFETY_GPUS"
if [ "$RUN_MODE" == "transformers_only" ]; then
    PIPELINE_GPUS="$VLLM_BASE_GPUS"
fi

# 确保日志目录存在
mkdir -p "$LOGFILE_DIR"

if [ "$RUN_MODE" == "spec_ppl" ] && [ ! -f "$SAFETY_HEAD_PATH" ]; then
    echo "spec_ppl 需要 safety head，但当前不存在：$SAFETY_HEAD_PATH"
    echo "请确认路径，或将 RUN_MODE 改为 speculative/target_only/draft_only 后再运行。"
    exit 1
fi

# 清理函数
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$VLLM_BASE_PID" ]; then kill $VLLM_BASE_PID; fi
    if [ ! -z "$VLLM_SMALL_PID" ]; then kill $VLLM_SMALL_PID; fi
    exit
}
trap cleanup INT TERM EXIT

# 等待服务启动的函数
wait_for_server() {
    local port=$1
    echo "Waiting for server on port $port..."
    while true; do
        if curl -s -f --noproxy '*' http://localhost:$port/health > /dev/null; then
            echo "Server on port $port is ready!"
            break
        else
            sleep 5
        fi
    done
}

if [ "$RUN_MODE" != "draft_only" ] && [ "$RUN_MODE" != "safedecoding" ] && [ "$RUN_MODE" != "secdecoding" ] && [ "$RUN_MODE" != "transformers_only" ]; then
    echo "Starting Base Model ($BASE_MODEL_NAME) on GPUs $VLLM_BASE_GPUS..."
    CUDA_VISIBLE_DEVICES=$VLLM_BASE_GPUS vllm serve "$BASE_MODEL_NAME" \
        --dtype auto \
        --tensor-parallel-size "$TP_SIZE_BASE" \
        --max-model-len "$BASE_MAX_MODEL_LEN" \
        --gpu-memory-utilization "$BASE_GPU_MEMORY_UTILIZATION" \
        --enable-prefix-caching \
        "${BASE_VLLM_EXTRA_ARGS[@]}" \
        --port $PORT_BASE \
        > "${LOGFILE_DIR}/vllm_base_jailbreak.log" 2>&1 &
    VLLM_BASE_PID=$!
    wait_for_server $PORT_BASE
fi

if [ "$RUN_MODE" != "target_only" ] && [ "$RUN_MODE" != "safedecoding" ] && [ "$RUN_MODE" != "secdecoding" ] && [ "$RUN_MODE" != "transformers_only" ]; then
    echo "Starting Small Model ($SMALL_MODEL_NAME) on GPU $VLLM_SMALL_GPUS..."
    CUDA_VISIBLE_DEVICES=$VLLM_SMALL_GPUS vllm serve "$SMALL_MODEL_NAME" \
        --dtype auto \
        --tensor-parallel-size "$TP_SIZE_SMALL" \
        --max-model-len "$SMALL_MAX_MODEL_LEN" \
        --gpu-memory-utilization "$SMALL_GPU_MEMORY_UTILIZATION" \
        --enable-prefix-caching \
        "${SMALL_VLLM_EXTRA_ARGS[@]}" \
        --port $PORT_SMALL \
        > "${LOGFILE_DIR}/vllm_small_jailbreak.log" 2>&1 &
    VLLM_SMALL_PID=$!
    wait_for_server $PORT_SMALL
fi

echo "Servers ready. Running jailbreak pipeline (run_mode=$RUN_MODE, json_out=${JSON_OUTPUT_DIR}/${RUN_MODE}, score_threshold=$THRESHOLD, safety_threshold=$SAFETY_THRESHOLD, safety_head=$(basename "$SAFETY_HEAD_PATH"))..."

# 运行 Python 流水线脚本
CUDA_VISIBLE_DEVICES=$PIPELINE_GPUS python "$SCRIPT_DIR/run_jailbreak_pipeline.py" \
    --run_mode "$RUN_MODE" \
    --json_output_dir "$JSON_OUTPUT_DIR" \
    --gpu_ids "$PIPELINE_GPUS" \
    --score_threshold "$THRESHOLD" \
    --safety_threshold "$SAFETY_THRESHOLD" \
    --token_budget 4096 \
    --model_key "$MODEL_KEY" \
    --base_model_path "$BASE_MODEL_NAME" \
    --draft_model_path "$SMALL_MODEL_NAME" \
    --safety_model_path "$SAFETY_MODEL_PATH" \
    --safety_head_path "$SAFETY_HEAD_PATH" \
    "${METHODS_ARGS[@]}" \
    "${SAFETY_HIDDEN_EXTRA[@]}" \
    --base_port $PORT_BASE \
    --draft_port $PORT_SMALL \
    $SEC_ARGS \
    "$@"

# 等待脚本执行完毕
wait $!
