#!/bin/bash
#SBATCH --job-name=pixcell-tme-full
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/gpfs/projects/bagherilab/pohaoc2/logs/tme_full_training/slurm-%A_%a.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gpfs/projects/bagherilab/pohaoc2}
CONDA_ROOT=${CONDA_ROOT:-$PROJECT_ROOT/tools/miniconda3}
ENV_PREFIX=${ENV_PREFIX:-$PROJECT_ROOT/envs/pixcell}
REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
WORK_BASE=${WORK_BASE:-$PROJECT_ROOT/checkpoints/tme_full_training}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs/tme_full_training}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-80}
S3_URI=${S3_URI:-}
REQUIRE_S3=${REQUIRE_S3:-0}

VARIANTS=(concat additive grouped per_channel)
CONFIGS=(
    configs/config_controlnet_exp_a1_concat.py
    configs/config_controlnet_exp_a2_bypass.py
    configs/config_controlnet_exp.py
    configs/config_controlnet_exp_a1_per_channel.py
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
VARIANT=${VARIANTS[$TASK_ID]}
CONFIG=${CONFIGS[$TASK_ID]}

mkdir -p "$LOG_DIR" "$WORK_BASE"

RUN_ID="${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
WORK_DIR="$WORK_BASE/${VARIANT}_${RUN_ID}"
RUN_LOG="$LOG_DIR/${VARIANT}_${RUN_ID}.log"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_PREFIX"
set -u

cd "$REPO_ROOT"

{
    echo "job_id=${SLURM_JOB_ID:-manual}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID:-manual}"
    echo "array_task_id=$TASK_ID"
    echo "variant=$VARIANT"
    echo "config=$CONFIG"
    echo "host=$(hostname)"
    echo "repo_root=$REPO_ROOT"
    echo "work_dir=$WORK_DIR"
    echo "log_dir=$LOG_DIR"
    echo "seed=$SEED"
    echo "batch_size=$BATCH_SIZE"
    echo "s3_uri=$S3_URI"
    echo "require_s3=$REQUIRE_S3"
    echo "python=$(which python)"
    python --version
    echo
    nvidia-smi
    echo
    echo "=== depth-27 TME full training start ==="
} | tee "$RUN_LOG"

if [[ -n "$S3_URI" && "$REQUIRE_S3" == "1" ]]; then
    echo "=== S3 preflight ===" | tee -a "$RUN_LOG"
    env AWS_REGION="${AWS_REGION:-us-east-1}" AWS_EC2_METADATA_DISABLED="${AWS_EC2_METADATA_DISABLED:-true}" \
        timeout 60 aws s3 ls "$S3_URI" 2>&1 | tee -a "$RUN_LOG"
fi

python -u stage2_train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" 2>&1 | tee -a "$RUN_LOG"

if [[ -n "$S3_URI" ]]; then
    DEST="${S3_URI%/}/${VARIANT}_${RUN_ID}/"
    echo "=== uploading to $DEST ===" | tee -a "$RUN_LOG"
    env AWS_REGION="${AWS_REGION:-us-east-1}" AWS_EC2_METADATA_DISABLED="${AWS_EC2_METADATA_DISABLED:-true}" \
        aws s3 sync "$WORK_DIR" "$DEST" 2>&1 | tee -a "$RUN_LOG"
    env AWS_REGION="${AWS_REGION:-us-east-1}" AWS_EC2_METADATA_DISABLED="${AWS_EC2_METADATA_DISABLED:-true}" \
        aws s3 cp "$RUN_LOG" "$DEST" 2>&1 | tee -a "$RUN_LOG"
fi
