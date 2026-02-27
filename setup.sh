#!/bin/bash
# PixCell EC2 Full Setup Script
# Environment: Python 3.12, CUDA 12.8, PyTorch 2.10 (matching Colab)
set -e

echo "=== [1/6] Creating conda environment ==="
conda create -n pixcell python=3.12 -y
eval "$(conda shell.bash hook)"
conda activate pixcell

echo "=== [2/6] Installing CUDA toolkit (for nvcc) ==="
conda install -c nvidia cuda-nvcc=12.8 cuda-toolkit=12.8 -y
nvcc --version

echo "=== [3/6] Installing PyTorch 2.10 + CUDA 12.8 ==="
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== [4/6] Installing setuptools (required for mmcv build) ==="
pip install setuptools==75.2.0 wheel

echo "=== [5/6] Installing mmcv 1.7.0 from source ==="
pip install mmcv==1.7.0 --no-build-isolation

echo "=== [6/6] Installing remaining requirements ==="
pip install -r requirements.txt

echo ""
echo "=== Verifying installation ==="
python -c "
import torch, mmcv, cv2, numpy as np
print(f'✓ torch={torch.__version__}')
print(f'✓ mmcv={mmcv.__version__}')
print(f'✓ opencv={cv2.__version__}')
print(f'✓ numpy={np.__version__}')
print(f'✓ CUDA available={torch.cuda.is_available()}')
print(f'✓ CUDA version={torch.version.cuda}')
"
