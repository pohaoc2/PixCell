#!/bin/bash
#SBATCH --job-name=pixcell-tme-mid
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=0-3
#SBATCH --output=/gpfs/projects/bagherilab/pohaoc2/logs/tme_midlength_depth27/slurm-%A_%a.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gpfs/projects/bagherilab/pohaoc2}
CONDA_ROOT=${CONDA_ROOT:-$PROJECT_ROOT/tools/miniconda3}
ENV_PREFIX=${ENV_PREFIX:-$PROJECT_ROOT/envs/pixcell}
REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
WORK_BASE=${WORK_BASE:-$PROJECT_ROOT/checkpoints/tme_midlength_depth27}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs/tme_midlength_depth27}
SEED=${SEED:-42}
PASS_THRESHOLD=${PASS_THRESHOLD:-10}
MIN_STEP=${MIN_STEP:-500}

VARIANTS=(concat additive grouped per_channel)
CONFIGS=(
    configs/config_controlnet_exp_smoke_depth27_bs2_concat_midlength.py
    configs/config_controlnet_exp_smoke_depth27_bs2_additive_midlength.py
    configs/config_controlnet_exp_smoke_depth27_bs2_grouped_midlength.py
    configs/config_controlnet_exp_smoke_depth27_bs2_per_channel_midlength.py
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
VARIANT=${VARIANTS[$TASK_ID]}
CONFIG=${CONFIGS[$TASK_ID]}

mkdir -p "$LOG_DIR" "$WORK_BASE"

RUN_ID="${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
WORK_DIR="$WORK_BASE/${VARIANT}_${RUN_ID}"
RUN_LOG="$LOG_DIR/${VARIANT}_${RUN_ID}.log"
SUMMARY_JSON="$LOG_DIR/${VARIANT}_${RUN_ID}.summary.json"

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
    echo "min_step=$MIN_STEP"
    echo "pass_threshold=$PASS_THRESHOLD"
    echo "python=$(which python)"
    python --version
    echo
    nvidia-smi
    echo
    echo "=== depth-27 TME mid-length smoke start ==="
} | tee "$RUN_LOG"

set +e
python -u stage2_train.py "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --seed "$SEED" 2>&1 | tee -a "$RUN_LOG"
train_exit=${PIPESTATUS[0]}
set -e

if [[ "$train_exit" -ne 0 ]]; then
    echo "training_status=FAIL exit_code=$train_exit" | tee -a "$RUN_LOG"
    exit "$train_exit"
fi

python -u tools/debug/check_tme_midlength.py \
    "$WORK_DIR/train_log.jsonl" \
    --threshold "$PASS_THRESHOLD" \
    --min-step "$MIN_STEP" \
    --summary-json "$SUMMARY_JSON" 2>&1 | tee -a "$RUN_LOG"
