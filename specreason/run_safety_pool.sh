#!/bin/bash
#
# 启动 safety 的 vLLM pooling 实例（替代旧的 HF transformers 32B 安全后端）。
# pooling 用 MEAN + normalize=false，与训练时 HF last-layer mask-mean-pool 数值等价
# （已奇偶校验 cos≈1.0、head sigmoid |Δ|<0.003）。供 spec_reason_ppl --safety_backend vllm 查询。
#
# 幂等：若 POOL_PORT 已健康则直接返回。缓存/临时全部指向 /data（根分区 / 已满）。
#
# 用法：
#   bash run_safety_pool.sh                       # 默认 GPU 7、端口 30009
#   POOL_GPU=5 POOL_PORT=30009 bash run_safety_pool.sh
#
set -u
DATA_DIR="/data/xuhaotian"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-python310}"

export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

POOL_PORT="${POOL_PORT:-30009}"
POOL_GPU="${POOL_GPU:-7}"
POOL_MODEL="${POOL_MODEL:-qwen3-32b-embed}"
SAFETY_MODEL_PATH="${SAFETY_MODEL_PATH:-${DATA_DIR}/model/Qwen3-32B}"
MAX_LEN="${POOL_MAX_LEN:-2048}"
LOG="${POOL_LOG:-${DATA_DIR}/specreason-origin/run_logs/vllm_pool.log}"

health() { curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "http://localhost:$POOL_PORT/health" 2>/dev/null | grep -q 200; }

if health; then
    echo "[pool] :$POOL_PORT 已在运行，跳过启动。"
    exit 0
fi

# 缓存/临时重定向到有空间的 /data（根分区 / 已满会导致 cudagraph 编译 ENOSPC）
mkdir -p "$DATA_DIR/.cache/vllm" "$DATA_DIR/.cache/inductor" "$DATA_DIR/.cache/triton" "$DATA_DIR/tmp" "$(dirname "$LOG")"
export VLLM_CACHE_ROOT="$DATA_DIR/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$DATA_DIR/.cache/inductor"
export TRITON_CACHE_DIR="$DATA_DIR/.cache/triton"
export TMPDIR="$DATA_DIR/tmp"

echo "[pool] 启动 pooling 实例: model=$SAFETY_MODEL_PATH gpu=$POOL_GPU port=$POOL_PORT served=$POOL_MODEL (--enforce-eager)"
CUDA_VISIBLE_DEVICES="$POOL_GPU" nohup vllm serve "$SAFETY_MODEL_PATH" \
    --runner pooling --convert embed \
    --pooler-config '{"pooling_type":"MEAN","normalize":false}' \
    --enforce-eager --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "${POOL_GPU_UTIL:-0.92}" \
    --port "$POOL_PORT" --served-model-name "$POOL_MODEL" \
    > "$LOG" 2>&1 &
POOL_PID=$!
echo "[pool] PID=$POOL_PID，等待就绪（日志 $LOG）..."

for i in $(seq 1 120); do
    if health; then echo "[pool] READY (PID=$POOL_PID)"; exit 0; fi
    if grep -qiE "No space left|out of memory|Engine core initialization failed|Failed core proc" "$LOG" 2>/dev/null; then
        echo "[pool] 启动失败："; grep -iE "No space left|out of memory|Engine core init" "$LOG" | tail -3; exit 1
    fi
    if ! kill -0 "$POOL_PID" 2>/dev/null; then echo "[pool] 进程退出，见 $LOG"; tail -3 "$LOG"; exit 1; fi
    sleep 4
done
echo "[pool] 超时未就绪，见 $LOG"; exit 1
