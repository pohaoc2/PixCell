#!/bin/bash
#SBATCH --job-name=pixcell-tme-upload-one
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/bagherilab/pohaoc2/logs/tme_full_training/upload-one-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gpfs/projects/bagherilab/pohaoc2}
CONDA_ROOT=${CONDA_ROOT:-$PROJECT_ROOT/tools/miniconda3}
ENV_PREFIX=${ENV_PREFIX:-$PROJECT_ROOT/envs/pixcell}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs/tme_full_training}
S3_URI=${S3_URI:?Set S3_URI to the destination prefix}
RUN_DIR=${RUN_DIR:?Set RUN_DIR to the completed training work directory}

mkdir -p "$LOG_DIR"
RUN_NAME=$(basename "$RUN_DIR")
RUN_LOG="$LOG_DIR/upload-${RUN_NAME}-${SLURM_JOB_ID:-manual}.log"
TRAIN_LOG="$LOG_DIR/${RUN_NAME}.log"
DEST="${S3_URI%/}/${RUN_NAME}/"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_PREFIX"
set -u

{
    echo "job_id=${SLURM_JOB_ID:-manual}"
    echo "host=$(hostname)"
    echo "run_dir=$RUN_DIR"
    echo "train_log=$TRAIN_LOG"
    echo "dest=$DEST"
    echo "python=$(which python)"
    aws --version
    echo
    echo "=== S3 preflight ==="
} | tee "$RUN_LOG"

env AWS_REGION="${AWS_REGION:-us-east-1}" AWS_EC2_METADATA_DISABLED="${AWS_EC2_METADATA_DISABLED:-true}" \
    aws s3 ls "$S3_URI" 2>&1 | tee -a "$RUN_LOG"

echo "=== uploading $RUN_DIR ===" | tee -a "$RUN_LOG"
env AWS_REGION="${AWS_REGION:-us-east-1}" AWS_EC2_METADATA_DISABLED="${AWS_EC2_METADATA_DISABLED:-true}" \
    aws s3 sync "$RUN_DIR" "$DEST" 2>&1 | tee -a "$RUN_LOG"

if [[ -f "$TRAIN_LOG" ]]; then
    echo "=== uploading $TRAIN_LOG ===" | tee -a "$RUN_LOG"
    env AWS_REGION="${AWS_REGION:-us-east-1}" AWS_EC2_METADATA_DISABLED="${AWS_EC2_METADATA_DISABLED:-true}" \
        aws s3 cp "$TRAIN_LOG" "$DEST" 2>&1 | tee -a "$RUN_LOG"
fi

echo "=== upload complete ===" | tee -a "$RUN_LOG"
