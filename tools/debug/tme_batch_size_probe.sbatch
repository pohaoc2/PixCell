#!/bin/bash
#SBATCH --job-name=pixcell-tme-bs
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/projects/bagherilab/pohaoc2/logs/tme_batch_size_probe/slurm-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gpfs/projects/bagherilab/pohaoc2}
CONDA_ROOT=${CONDA_ROOT:-$PROJECT_ROOT/tools/miniconda3}
ENV_PREFIX=${ENV_PREFIX:-$PROJECT_ROOT/envs/pixcell}
REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs/tme_batch_size_probe}
SEED=${SEED:-42}
STEPS=${STEPS:-20}
BATCH_SIZES=${BATCH_SIZES:-"32 64 128 180 256 300"}
BASE_CONFIG=${BASE_CONFIG:-configs/config_controlnet_exp.py}

RUN_ID=${SLURM_JOB_ID:-manual}
OUTPUT_DIR="$LOG_DIR/job-${RUN_ID}"
mkdir -p "$OUTPUT_DIR"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_PREFIX"
set -u

cd "$REPO_ROOT"

{
    echo "job_id=${SLURM_JOB_ID:-manual}"
    echo "host=$(hostname)"
    echo "repo_root=$REPO_ROOT"
    echo "output_dir=$OUTPUT_DIR"
    echo "base_config=$BASE_CONFIG"
    echo "batch_sizes=$BATCH_SIZES"
    echo "steps=$STEPS"
    echo "seed=$SEED"
    echo "python=$(which python)"
    python --version
    echo
    nvidia-smi
    echo
    echo "=== depth-27 TME batch-size probe start ==="
} | tee "$OUTPUT_DIR/run.log"

python -u tools/debug/probe_tme_batch_size.py \
    --base-config "$BASE_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --batch-sizes $BATCH_SIZES 2>&1 | tee -a "$OUTPUT_DIR/run.log"
