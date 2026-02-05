# Quick Start Guide

Get started with VLA training in 5 minutes!

## 1. Setup Environment

```bash
# Clone or navigate to the project
cd vla-flow

# Run setup script (creates conda env and installs dependencies)
bash setup.sh

# Activate environment
conda activate vla-flow
```

## 2. Prepare Dataset

### Option A: Use Existing LIBERO Dataset

If you already have LIBERO dataset downloaded:

```bash
# Edit config file
nano configs/single_task_overfit.yaml

# Update these lines:
# data:
#   data_path: "/path/to/your/libero/dataset"
#   task_name: "LIVING_ROOM_SCENE0_put_the_black_bowl_on_top_of_the_cabinet"
```

### Option B: Download LIBERO Dataset

```bash
# Download using LIBERO tools
python -c "
from libero.libero import benchmark
import os

# Download LIBERO-Spatial (smallest suite, ~500MB)
benchmark_dict = benchmark.get_benchmark_dict()
spatial_suite = benchmark_dict['libero_spatial']()

print(f'Dataset downloaded to: {spatial_suite.get_task_names()}')
print(f'Available tasks: {len(spatial_suite.get_task_names())}')
"

# The dataset will be at ~/.libero/datasets/libero_spatial/
```

## 3. Verify Dataset

```bash
# Visualize dataset samples
python utils/visualize_data.py --config configs/single_task_overfit.yaml --num-samples 4

# This will:
# - Print dataset statistics
# - Show sample images with actions
# - Save visualization to dataset_visualization.png
```

## 4. Start Training

### Single GPU

```bash
python train.py --config configs/single_task_overfit.yaml
```

### Multi-GPU (2x Ada 6000)

```bash
torchrun --nproc_per_node=2 train.py --config configs/single_task_overfit.yaml
```

Training progress will be saved to `outputs/TASK_NAME_TIMESTAMP/`

## 5. Monitor Training

### Option A: TensorBoard

```bash
# In a separate terminal
tensorboard --logdir outputs/

# Open browser to http://localhost:6006
```

### Option B: Watch Logs

```bash
# Watch training output
tail -f outputs/*/logs/events.out.tfevents.*
```

## 6. Evaluate Model

```bash
# Evaluate on validation set
python evaluate.py \
    --checkpoint outputs/TASK_NAME/best_model.pth \
    --config configs/single_task_overfit.yaml \
    --num-flow-steps 50
```

## 7. Run Inference

```bash
# Predict action for a single observation
python inference.py \
    --checkpoint outputs/TASK_NAME/best_model.pth \
    --config configs/single_task_overfit.yaml \
    --image /path/to/observation.png \
    --instruction "put the black bowl on top of the cabinet" \
    --proprio "0.0,0.0,0.0,0.0,0.0,0.0,0.0" \
    --ee-pose "0.5,0.0,0.3,0.0,0.0,0.0,1.0"
```

## Expected Timeline

For single task overfitting on 2x Ada 6000:

- **Setup**: 5-10 minutes
- **Dataset download**: 5-15 minutes (depends on internet)
- **Training to convergence**: 1-3 hours (200-500 epochs)
- **Expected final loss**: < 0.01

## Tips

1. **Start small**: Use single task overfitting first to verify everything works
2. **Monitor memory**: Watch `nvidia-smi` to ensure you're not OOM
3. **Adjust batch size**: If OOM, reduce batch_size in config
4. **Visualize data**: Always verify dataset loading before long training runs
5. **Save often**: Set `save_interval` to save checkpoints frequently

## Troubleshooting

### "No module named 'libero'"
```bash
pip install libero
```

### "Dataset path not found"
- Check `data_path` in config points to correct location
- Verify LIBERO dataset is downloaded
- Try absolute path instead of relative path

### "CUDA out of memory"
- Reduce `batch_size` in config (try 32, 16, or 8)
- Use single GPU instead of multi-GPU
- Close other GPU processes

### "Loss is NaN"
- Reduce learning rate (try 1e-4 or 1e-5)
- Check dataset for invalid values
- Verify image normalization is correct

## Next Steps

After successful single task overfitting:

1. **Try different tasks**: Change `task_name` in config
2. **Multi-task training**: Train on multiple LIBERO tasks
3. **Add more modalities**: Incorporate depth images, force sensors
4. **Tune hyperparameters**: Experiment with learning rate, model size
5. **Deploy in simulation**: Test in LIBERO simulation environment

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Open an issue if you encounter problems
- Verify your setup matches the requirements (Python 3.10, PyTorch 2.1, CUDA 12.1)
