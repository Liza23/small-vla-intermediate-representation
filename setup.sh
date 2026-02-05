#!/bin/bash
# Setup script for VLA-Flow

set -e

echo "================================================"
echo "Setting up VLA-Flow environment"
echo "================================================"

# Create conda environment
echo ""
echo "[1/4] Creating conda environment..."
conda env create -f environment.yml

echo ""
echo "[2/4] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vla-flow

echo ""
echo "[3/4] Installing LIBERO..."
pip install libero

echo ""
echo "[4/4] Creating output directories..."
mkdir -p outputs
mkdir -p logs

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate vla-flow"
echo "2. Download LIBERO dataset or set data path in configs/single_task_overfit.yaml"
echo "3. Update task_name in config to your desired task"
echo "4. Start training: python train.py --config configs/single_task_overfit.yaml"
echo ""
echo "For multi-GPU training:"
echo "  torchrun --nproc_per_node=2 train.py --config configs/single_task_overfit.yaml"
echo ""
