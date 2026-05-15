#!/bin/bash
#SBATCH --job-name=pixcell-tme-upload
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/bagherilab/pohaoc2/logs/tme_full_training/upload-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gpfs/projects/bagherilab/pohaoc2}
CONDA_ROOT=${CONDA_ROOT:-$PROJECT_ROOT/tools/miniconda3}
ENV_PREFIX=${ENV_PREFIX:-$PROJECT_ROOT/envs/pixcell}
WORK_BASE=${WORK_BASE:-$PROJECT_ROOT/checkpoints/tme_full_training}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs/tme_full_training}
S3_URI=${S3_URI:?Set S3_URI to the destination prefix}
TRAIN_ARRAY_JOB_ID=${TRAIN_ARRAY_JOB_ID:?Set TRAIN_ARRAY_JOB_ID, e.g. 95470}

mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/upload-${SLURM_JOB_ID:-manual}.log"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_PREFIX"
set -u

{
    echo "job_id=${SLURM_JOB_ID:-manual}"
    echo "host=$(hostname)"
    echo "work_base=$WORK_BASE"
    echo "log_dir=$LOG_DIR"
    echo "s3_uri=$S3_URI"
    echo "train_array_job_id=$TRAIN_ARRAY_JOB_ID"
    echo "python=$(which python)"
    aws --version
    echo
    echo "=== S3 preflight ==="
} | tee "$RUN_LOG"

env AWS_REGION="${AWS_REGION:-us-east-1}" AWS_EC2_METADATA_DISABLED="${AWS_EC2_METADATA_DISABLED:-true}" \
    aws s3 ls "$S3_URI" 2>&1 | tee -a "$RUN_LOG"

shopt -s nullglob
run_dirs=("$WORK_BASE"/*_"${TRAIN_ARRAY_JOB_ID}"_*)
if [[ ${#run_dirs[@]} -eq 0 ]]; then
    echo "No run dirs matched $WORK_BASE/*_${TRAIN_ARRAY_JOB_ID}_*" | tee -a "$RUN_LOG"
    exit 1
fi

for work_dir in "${run_dirs[@]}"; do
    run_name=$(basename "$work_dir")
    dest="${S3_URI%/}/${run_name}/"
    echo "=== uploading $work_dir -> $dest ===" | tee -a "$RUN_LOG"
    env AWS_REGION="${AWS_REGION:-us-east-1}" AWS_EC2_METADATA_DISABLED="${AWS_EC2_METADATA_DISABLED:-true}" \
        aws s3 sync "$work_dir" "$dest" 2>&1 | tee -a "$RUN_LOG"

    variant_log="$LOG_DIR/${run_name}.log"
    if [[ -f "$variant_log" ]]; then
        env AWS_REGION="${AWS_REGION:-us-east-1}" AWS_EC2_METADATA_DISABLED="${AWS_EC2_METADATA_DISABLED:-true}" \
            aws s3 cp "$variant_log" "$dest" 2>&1 | tee -a "$RUN_LOG"
    fi
done

echo "=== upload complete ===" | tee -a "$RUN_LOG"
