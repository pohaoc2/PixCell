#!/bin/bash
#SBATCH --job-name=pixcell-smoke
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/projects/bagherilab/pohaoc2/logs/pixcell_smoketest/slurm-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gpfs/projects/bagherilab/pohaoc2}
CONDA_ROOT=${CONDA_ROOT:-$PROJECT_ROOT/tools/miniconda3}
ENV_PREFIX=${ENV_PREFIX:-$PROJECT_ROOT/envs/pixcell}
REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
WORK_BASE=${WORK_BASE:-$PROJECT_ROOT/checkpoints/pixcell_smoketest}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs/pixcell_smoketest}
SEED=${SEED:-7}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-240}

mkdir -p "$LOG_DIR" "$WORK_BASE"

RUN_LOG="$LOG_DIR/run-${SLURM_JOB_ID}.log"
STATUS_FILE="$LOG_DIR/run-${SLURM_JOB_ID}.status"
WORK_DIR="$WORK_BASE/job-${SLURM_JOB_ID}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_PREFIX"
set -u

cd "$REPO_ROOT"

{
    echo "job_id=$SLURM_JOB_ID"
    echo "host=$(hostname)"
    echo "repo_root=$REPO_ROOT"
    echo "work_dir=$WORK_DIR"
    echo "seed=$SEED"
    echo "timeout_seconds=$TIMEOUT_SECONDS"
    echo "python=$(which python)"
    python --version
    echo
    nvidia-smi
    echo
    echo "=== stage2 smoke test start ==="
} | tee "$RUN_LOG"

set +e
timeout --foreground --signal=TERM "$TIMEOUT_SECONDS" \
    python -u stage2_train.py configs/config_controlnet_exp.py \
    --work-dir "$WORK_DIR" \
    --seed "$SEED" \
    --debug 2>&1 | tee -a "$RUN_LOG"
cmd_exit=${PIPESTATUS[0]}
set -e

status="FAIL"
reason="stage2_train exited with error"

if [[ "$cmd_exit" -eq 0 ]]; then
    status="PASS"
    reason="stage2_train exited cleanly within smoke-test window"
elif [[ "$cmd_exit" -eq 124 ]]; then
    if grep -q "Traceback (most recent call last):" "$RUN_LOG"; then
        status="FAIL"
        reason="timed out, but traceback detected before timeout"
    else
        status="PASS_TIMEOUT"
        reason="stage2_train stayed alive until timeout without traceback"
    fi
elif grep -q "Traceback (most recent call last):" "$RUN_LOG"; then
    status="FAIL"
    reason="python traceback detected"
fi

{
    echo "status=$status"
    echo "reason=$reason"
    echo "exit_code=$cmd_exit"
    echo "run_log=$RUN_LOG"
    echo "work_dir=$WORK_DIR"
} | tee "$STATUS_FILE"

case "$status" in
    PASS|PASS_TIMEOUT)
        exit 0
        ;;
    *)
        exit 1
        ;;
esac
