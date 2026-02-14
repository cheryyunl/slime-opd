#!/bin/bash

# On-Policy Distillation (OPD) for Vision-Language Models on GEO3K dataset
# Based on examples/geo3k_vlm/run_geo3k_vlm.sh with OPD support added
#
# This script:
# 1. Starts a teacher model server (Qwen3-VL-32B)
# 2. Trains a student model (Qwen3-VL-8B) using OPD
#
# Usage:
#   bash examples/opd_vlm/run_opd_vlm_geo3k.sh
#   SLIME_SCRIPT_TRAIN_BACKEND=fsdp bash examples/opd_vlm/run_opd_vlm_geo3k.sh
#   SLIME_SCRIPT_STUDENT_MODEL=Qwen3-VL-4B-Instruct bash examples/opd_vlm/run_opd_vlm_geo3k.sh

# ============================================================================
# Configuration
# ============================================================================

# Training backend
TRAIN_BACKEND=${SLIME_SCRIPT_TRAIN_BACKEND:-"megatron"}

# Student model (being trained)
STUDENT_MODEL=${SLIME_SCRIPT_STUDENT_MODEL:-"Qwen3-VL-4B-Thinking"}

# Teacher model (provides log-probs for distillation)
# Note: -Thinking models naturally output step-by-step reasoning (built-in CoT)
# This provides longer, more detailed responses - better for distillation!
TEACHER_MODEL=${SLIME_SCRIPT_TEACHER_MODEL:-"Qwen3-VL-30B-A3B-Thinking"}

# Dataset
DATASET_NAME=${SLIME_SCRIPT_DATASET_NAME:-"chenhegu/geo3k_imgurl"}
DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")

# GPU allocation
# For 16 GPUs: 8 for teacher, 8 for student
# For 8 GPUs: 4 for teacher, 4 for student
# Note: 8x H100 80GB is sufficient for 30B-A3B teacher + 4B student
NUM_GPUS_TOTAL=${SLIME_SCRIPT_NUM_GPUS:-8}
NUM_GPUS_TEACHER=${SLIME_SCRIPT_NUM_GPUS_TEACHER:-4}
NUM_GPUS_STUDENT=${SLIME_SCRIPT_NUM_GPUS_STUDENT:-4}

# OPD settings
OPD_KL_COEF=${SLIME_SCRIPT_OPD_KL_COEF:-1.0}  # KL penalty coefficient
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141

# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# ============================================================================
# Validate Configuration
# ============================================================================

VALID_MODELS="
  Qwen2.5-VL-3B-Instruct
  Qwen2.5-VL-7B-Instruct
  Qwen2.5-VL-32B-Instruct
  Qwen2.5-VL-72B-Instruct
  Qwen3-VL-2B-Instruct
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-30B-A3B-Instruct
  Qwen3-VL-235B-A22B-Instruct
  Qwen3-VL-2B-Thinking
  Qwen3-VL-4B-Thinking
  Qwen3-VL-8B-Thinking
  Qwen3-VL-30B-A3B-Thinking
  Qwen3-VL-235B-A22B-Thinking
"

if ! echo "$VALID_MODELS" | grep -qw "$STUDENT_MODEL"; then
   echo "Error: STUDENT_MODEL must be one of: $VALID_MODELS"
   exit 1
fi

if ! echo "$VALID_MODELS" | grep -qw "$TEACHER_MODEL"; then
   echo "Error: TEACHER_MODEL must be one of: $VALID_MODELS"
   exit 1
fi

# Validate GPU allocation
if [ $((NUM_GPUS_TEACHER + NUM_GPUS_STUDENT)) -gt $NUM_GPUS_TOTAL ]; then
   echo "Error: NUM_GPUS_TEACHER ($NUM_GPUS_TEACHER) + NUM_GPUS_STUDENT ($NUM_GPUS_STUDENT) exceeds NUM_GPUS_TOTAL ($NUM_GPUS_TOTAL)"
   exit 1
fi

echo "============================================"
echo "OPD VLM Configuration:"
echo "  Student Model: $STUDENT_MODEL"
echo "  Teacher Model: $TEACHER_MODEL"
echo "  Total GPUs: $NUM_GPUS_TOTAL"
echo "  Teacher GPUs: $NUM_GPUS_TEACHER"
echo "  Student GPUs: $NUM_GPUS_STUDENT"
echo "  OPD KL Coef: $OPD_KL_COEF"
echo "  Train Backend: $TRAIN_BACKEND"
echo "============================================"

# ============================================================================
# Cleanup
# ============================================================================

pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force
   pkill -9 ray
fi
pkill -9 slime
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray
fi
pkill -9 slime
pkill -9 redis

set -ex

export PYTHONBUFFERED=16

# ============================================================================
# Detect NVLink
# ============================================================================

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ============================================================================
# Download Models and Dataset
# ============================================================================

mkdir -p /root/models /root/datasets

# Download student model
if [ ! -d "/root/models/${STUDENT_MODEL}" ]; then
   echo "Downloading student model: ${STUDENT_MODEL}..."
   hf download Qwen/${STUDENT_MODEL} --local-dir /root/models/${STUDENT_MODEL}
fi

# Download teacher model
if [ ! -d "/root/models/${TEACHER_MODEL}" ]; then
   echo "Downloading teacher model: ${TEACHER_MODEL}..."
   hf download Qwen/${TEACHER_MODEL} --local-dir /root/models/${TEACHER_MODEL}
fi

# Download dataset
if [ ! -d "/root/datasets/${DATASET_LOCAL_NAME}" ]; then
   echo "Downloading dataset: ${DATASET_NAME}..."
   hf download --repo-type dataset ${DATASET_NAME} --local-dir /root/datasets/${DATASET_LOCAL_NAME}
fi

# ============================================================================
# Start Teacher Model Server
# ============================================================================

echo "Starting teacher model server (${TEACHER_MODEL}) on ${NUM_GPUS_TEACHER} GPUs..."

# Build GPU list for teacher (e.g., "0,1,2,3,4,5,6,7")
TEACHER_GPU_IDS=$(seq -s, 0 $((NUM_GPUS_TEACHER - 1)))

LOG_FILE="/tmp/teacher_sglang_$(date +%Y%m%d_%H%M%S).log"

# Launch teacher server in background
CUDA_VISIBLE_DEVICES=${TEACHER_GPU_IDS} python3 -m sglang.launch_server \
    --model-path /root/models/${TEACHER_MODEL} \
    --host 0.0.0.0 \
    --port ${TEACHER_PORT} \
    --tp ${NUM_GPUS_TEACHER} \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.6 \
    > "$LOG_FILE" 2>&1 &

TEACHER_PID=$!
echo "Teacher server started with PID: $TEACHER_PID, logging to: $LOG_FILE"

# Wait for teacher server to be ready
echo "Waiting for teacher model server to start..."
MAX_WAIT=300  # 5 minutes
WAITED=0
until curl -sf http://${TEACHER_IP}:${TEACHER_PORT}/health_generate > /dev/null; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Error: Teacher server failed to start within ${MAX_WAIT} seconds"
        echo "Last 20 lines of log:"
        tail -n 20 "$LOG_FILE"
        exit 1
    fi
    echo "Waiting... ($WAITED/$MAX_WAIT seconds)"
    tail -n 5 "$LOG_FILE"
    sleep 5
    WAITED=$((WAITED + 5))
done

curl http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info
echo "Teacher model server is ready at ${TEACHER_IP}:${TEACHER_PORT}"
sleep 5

# ============================================================================
# Training Arguments
# ============================================================================

# Checkpoint args
CKPT_ARGS=(
   --hf-checkpoint /root/models/${STUDENT_MODEL}
   --rotary-base 5000000  # Required for Qwen3-VL
)

# Rollout args
ROLLOUT_ARGS=(
   --prompt-data /root/datasets/${DATASET_LOCAL_NAME}/train.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 512
)

# OPD-specific args
OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-kl-coef ${OPD_KL_COEF}
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
   --custom-reward-function examples.opd_vlm.opd_vlm_reward.reward_func
   --custom-post-process-rewards examples.opd_vlm.opd_vlm_reward.post_process_rewards
)

# Note: We use custom reward functions instead of --rm-type math
# because OPD needs to call the teacher model for log-probs.
# If you want hybrid OPD + task rewards, modify post_process_rewards to include math scores.

# Multimodal keys (required for VLM)
MULTIMODAL_KEYS='{"image": "images"}'

# Evaluation args
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data ${DATASET_LOCAL_NAME} /root/datasets/${DATASET_LOCAL_NAME}/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
)

# GRPO args (advantage estimator)
GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# Optimizer args
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# SGLang args
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
)

# Wandb args
if [ -n "$WANDB_API_KEY" ]; then
   STUDENT_MODEL_LOWER=$(echo "$STUDENT_MODEL" | tr '[:upper:]' '[:lower:]')
   WANDB_ARGS=(
      --use-wandb
      --wandb-project slime-opd-vlm
      --wandb-group ${STUDENT_MODEL_LOWER}-opd-${TRAIN_BACKEND}
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

MISC_ARGS=(
   --colocate
)

# Backend-specific args
if [ "$TRAIN_BACKEND" = "fsdp" ]; then
   BACKEND_ARGS=(
      --train-backend fsdp
      --gradient-checkpointing
      --sglang-attention-backend fa3
      --attn-implementation flash_attention_3
      --update-weight-buffer-size 536870912
   )
   MODEL_ARGS=()
else
   # megatron backend (default)
   BACKEND_ARGS=(
      --train-backend megatron
      --load /root/models/${STUDENT_MODEL}
      --tensor-model-parallel-size 4
      --sequence-parallel
      --pipeline-model-parallel-size 1
      --context-parallel-size 1
      --expert-model-parallel-size 1
      --expert-tensor-parallel-size 1
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
      --use-dynamic-batch-size
      --max-tokens-per-gpu 4096
      --attention-dropout 0.0
      --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      --attention-backend flash
      --megatron-to-hf-mode bridge
   )

   # Get MODEL_ARGS from scripts/models
   SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
   MODEL_ARGS_FILE=$(echo "$STUDENT_MODEL" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
   MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"
fi

# ============================================================================
# Start Ray and Training
# ============================================================================

# Start Ray if not using external Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS_STUDENT} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

echo "Starting OPD training..."
echo "Teacher: ${TEACHER_MODEL} (${NUM_GPUS_TEACHER} GPUs)"
echo "Student: ${STUDENT_MODEL} (${NUM_GPUS_STUDENT} GPUs)"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS_STUDENT} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPD_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}

echo "Training completed!"
echo "Teacher server log: $LOG_FILE"
