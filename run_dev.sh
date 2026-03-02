#!/bin/bash
#SBATCH --job-name=asft_dev
#SBATCH --partition=dev
#SBATCH --account=MST111121
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH -o asft_dev_%j.out
#SBATCH -e asft_dev_%j.err
# 可選：寄信通知
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=YOUR_EMAIL_HERE

set -euo pipefail

module purge
ml singularity/3.7.1

# ----- Host paths -----
SIF="$HOME/analogseeker_train.sif"          # TODO: 改成你的 sif 實際路徑
CODE_DIR="$PWD"
DATA_DIR="$PWD/analog_data"
OUTPUT_DIR="$PWD/output"

# 建議 cache 放 /work（若你們沒有 /work，改成 "$HOME/hf_cache"）
HF_CACHE="/work/$USER/hf_cache"

mkdir -p "$OUTPUT_DIR" "$HF_CACHE"

# ----- Threading / tokenizers (避免 OpenBLAS thread 爆炸) -----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

# ----- NCCL (dev 先保守) -----
# 單節點通常關 IB 更穩；確認都 OK 再改回 0
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# ----- HF cache routing -----
export HF_HOME=/hf_cache
export TORCH_HOME=/hf_cache/torch
export XDG_CACHE_HOME=/hf_cache/xdg

# 可選：如果你已經把模型 cache 準備好，想強制離線避免 compute node 外網問題就打開：
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# ----- torchrun -----
NGPU_RAW="${SLURM_GPUS_ON_NODE:-4}"
NGPU="${NGPU_RAW##*:}"     # 防止有些系統回傳 gpu:4
MASTER_ADDR="127.0.0.1"    # 單節點用 loopback 最穩
MASTER_PORT=29500

echo "NGPU_RAW=${NGPU_RAW} -> NGPU=${NGPU}"
echo "HF_CACHE(host)=${HF_CACHE} -> /hf_cache(container)"

# ----- Training args (container paths) -----
ARGS=" \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --data_path /data/sft.jsonl \
  --output_dir /output \
  --max_seq_length 8192 \
  --num_train_epochs 1 \
  --learning_rate 2e-6 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lambda_kl 0.1 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --gradient_checkpointing \
  --attn_implementation eager \
  --kl_chunk_size 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --bf16 \
  --dataloader_num_workers 4 \
  --pilot_steps 30 \
  --evals_per_epoch 999999 \
  --format_check_samples 0 \
  --report_to none \
"

# ---- Run (不要用 srun 再包 torchrun 多進程) ----
singularity exec --nv --no-home \
  --env HF_HOME=/hf_cache \
  --env TORCH_HOME=/hf_cache/torch \
  --env XDG_CACHE_HOME=/hf_cache/xdg \
  --env HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} \
  --env TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0} \
  --env NCCL_DEBUG=${NCCL_DEBUG} \
  --env NCCL_IB_DISABLE=${NCCL_IB_DISABLE} \
  --env NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
  --env OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  --env MKL_NUM_THREADS=${MKL_NUM_THREADS} \
  --env OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} \
  --env NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS} \
  --env TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM} \
  -B "${CODE_DIR}:/code" \
  -B "${DATA_DIR}:/data" \
  -B "${OUTPUT_DIR}:/output" \
  -B "${HF_CACHE}:/hf_cache" \
  "${SIF}" \
  torchrun --nproc_per_node="${NGPU}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    /code/train.py ${ARGS}

echo "=== DEV PILOT DONE ==="