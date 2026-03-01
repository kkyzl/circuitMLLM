#!/bin/bash
#SBATCH --job-name=analogseeker_sft      # Job name
#SBATCH --partition=dev                  # Partition (edit for your system)
#SBATCH --time=12:00:00                  # Wall time limit
#SBATCH --account=MST111121
#SBATCH --nodes=1                        # (-N) Number of nodes
#SBATCH --gpus-per-node=8               # GPUs per node
#SBATCH --cpus-per-task=8               # (-c) CPU cores per task
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -o analogseeker_sft%j.log        # stdout
#SBATCH -e analogseeker_sft%j.err        # stderr

# ============================================================================
# AnalogSeeker NSC-SFT Training — SLURM Submission Script
#
# Runtime:
#   RUNTIME=container — Singularity/Apptainer container (HPC clusters)
#   RUNTIME=bare      — pip-installed deps on remote server (auto-installs)
#
# Modes:
#   MODE=deepspeed  — DeepSpeed ZeRO-3 + FP16 (for 24GB GPUs like 4090/3090)
#   MODE=legacy     — torchrun/single GPU + BF16 (for 80GB GPUs like A100)
#
# Usage:
#   sbatch run_train.sh          # on SLURM cluster
#   bash  run_train.sh           # on remote server (bare mode)
# ============================================================================

# --- Configuration (edit these) ---
RUNTIME=bare     # "container" (Singularity/Apptainer) or "bare" (pip on remote server)
SIF=/path/to/analogseeker_train.sif
CODE_DIR=./
DATA_DIR=./analog_data
HF_CACHE=/scratch/hf_cache
OUTPUT_DIR=./output
NGPU=8
MODE=deepspeed   # "deepspeed" or "legacy"

# WandB: set your API key or use offline mode
# export WANDB_API_KEY=your_key_here
# For air-gapped nodes, uncomment:
# export WANDB_MODE=offline

# --- Common training args ---
COMMON_ARGS=" \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --data_path /data/sft.jsonl \
    --output_dir /output \
    --max_seq_length 8192 \
    --num_train_epochs 1 \
    --learning_rate 2e-6 \
    --lambda_kl 0.1 \
    --lora_rank 64 \
    --gradient_checkpointing \
    --report_to wandb \
    --wandb_project analogseeker_sft \
    --kl_chunk_size 512 \
"

# --- Mode-specific args ---
if [ "$MODE" = "deepspeed" ]; then
    # DeepSpeed ZeRO-3: batch=1 per GPU, FP16 for RTX 3090 compat
    TRAIN_ARGS="${COMMON_ARGS} \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps $((32 / NGPU)) \
        --fp16 \
        --deepspeed /code/ds_zero3_config.json \
    "
else
    # Legacy: batch=2, BF16 for A100/H100
    TRAIN_ARGS="${COMMON_ARGS} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps $((64 / NGPU / 2)) \
        --bf16 \
    "
fi

# --- Optional: resume from checkpoint ---
# TRAIN_ARGS="$TRAIN_ARGS --resume_from_checkpoint /output/checkpoints/epoch-1"
# TRAIN_ARGS="$TRAIN_ARGS --num_train_epochs 3"

# --- Optional: unattended multi-epoch with early stopping ---
# TRAIN_ARGS="$TRAIN_ARGS --num_train_epochs 3 --early_stop_patience 2"

# --- Optional: pilot run ---
# TRAIN_ARGS="$TRAIN_ARGS --pilot_steps 20"

# --- NCCL settings for heterogeneous GPUs ---
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# --- Runtime setup ---
if [ "$RUNTIME" = "container" ]; then
    module purge
    ml singularity

    CONTAINER_CMD="apptainer exec --nv --no-home \
        --env WANDB_API_KEY=${WANDB_API_KEY:-} \
        --env WANDB_MODE=${WANDB_MODE:-online} \
        --env NCCL_IB_DISABLE=${NCCL_IB_DISABLE} \
        --env NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
        -B ${CODE_DIR}:/code \
        -B ${DATA_DIR}:/data \
        -B ${HF_CACHE}:/hf_cache \
        -B ${OUTPUT_DIR}:/output \
        ${SIF}"
    SCRIPT=/code/train.py
    DS_CONFIG=/code/ds_zero3_config.json
else
    # Bare-metal: install dependencies if needed
    echo "=== Bare-metal mode: checking dependencies ==="
    if ! python3 -c "import transformers" 2>/dev/null; then
        echo "Installing dependencies..."
        pip install -r ${CODE_DIR}/requirements.txt
    fi
    CONTAINER_CMD=""
    SCRIPT=${CODE_DIR}/train.py
    DS_CONFIG=${CODE_DIR}/ds_zero3_config.json
fi

# --- Resolve paths for bare-metal (use local paths instead of container mounts) ---
if [ "$RUNTIME" = "bare" ]; then
    TRAIN_ARGS=$(echo "$TRAIN_ARGS" | sed "s|/data/|${DATA_DIR}/|g; s|/output|${OUTPUT_DIR}|g; s|/code/|${CODE_DIR}/|g")
fi

# --- Launch ---
if [ "$MODE" = "deepspeed" ]; then
    echo "=== DeepSpeed ZeRO-3 training (${NGPU} GPUs, FP16) ==="
    ${CONTAINER_CMD} deepspeed \
        --num_gpus=${NGPU} \
        ${SCRIPT} ${TRAIN_ARGS}
elif [ "$NGPU" -eq 1 ]; then
    echo "=== Single GPU training (legacy) ==="
    ${CONTAINER_CMD} python3 ${SCRIPT} ${TRAIN_ARGS}
else
    echo "=== Multi-GPU training (${NGPU} GPUs, torchrun, legacy) ==="
    ${CONTAINER_CMD} torchrun --nproc_per_node=${NGPU} ${SCRIPT} ${TRAIN_ARGS}
fi

echo "=== Training complete ==="
