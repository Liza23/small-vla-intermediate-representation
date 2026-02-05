# VLA with Rectified Flow

Vision-Language-Action (VLA) model using Rectified Flow for robotic manipulation.

## Features

- **Rectified Flow**: Efficient flow matching for action prediction
- **Multi-modal inputs**: RGB images, language instructions, proprioception, end-effector pose
- **CLIP encoders**: Pretrained CLIP ViT-B/16 for vision and language
- **Multi-GPU training**: DistributedDataParallel support for 2+ GPUs
- **LIBERO dataset**: Single task overfitting for rapid prototyping

## Architecture

```
Inputs:
├── RGB Image (224x224)     -> CLIP ViT-B/16 -> [512]
├── Language Instruction    -> CLIP Text     -> [512]
├── Joint Positions (7D)    -> MLP           -> [512]
└── EE Pose (7D)            -> MLP           -> [512]
                                   ↓
                            Fusion Layer -> [512]
                                   ↓
                            Rectified Flow
                                   ↓
                            Actions (7D)
```

## Setup

### 1. Create Conda Environment

```bash
cd vla-flow
conda env create -f environment.yml
conda activate vla-flow
```

### 2. Download LIBERO Dataset

```bash
# Install LIBERO
pip install libero

# Download LIBERO-Spatial dataset
python -c "
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.envs as envs

# This will download the dataset to ~/.libero/
# Dataset structure:
# ~/.libero/datasets/libero_spatial/
"
```

Or manually download from: https://github.com/Lifelong-Robot-Learning/LIBERO

### 3. Update Config

Edit `configs/single_task_overfit.yaml`:
- Set `data.data_path` to your LIBERO dataset path
- Choose a task from LIBERO-Spatial (10 tasks available)
- Adjust batch size based on your GPU memory

Available LIBERO-Spatial tasks:
1. `LIVING_ROOM_SCENE0_put_the_black_bowl_on_top_of_the_cabinet`
2. `LIVING_ROOM_SCENE1_put_the_white_mug_on_the_left_plate`
3. `LIVING_ROOM_SCENE2_put_the_wine_bottle_on_the_rack`
4. `LIVING_ROOM_SCENE3_put_the_white_mug_on_the_plate`
5. `LIVING_ROOM_SCENE4_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`
6. `LIVING_ROOM_SCENE5_put_the_cream_cheese_in_the_bowl`
7. `LIVING_ROOM_SCENE6_put_the_butter_in_the_tray`
8. `KITCHEN_SCENE0_put_the_black_bowl_in_the_top_drawer_of_the_cabinet`
9. `KITCHEN_SCENE1_put_the_wine_bottle_on_top_of_the_cabinet`
10. `KITCHEN_SCENE2_put_the_white_mug_on_the_left_plate`

## Training

### Single GPU

```bash
python train.py --config configs/single_task_overfit.yaml
```

### Multi-GPU (2x Ada 6000)

```bash
# Using torchrun
torchrun --nproc_per_node=2 train.py --config configs/single_task_overfit.yaml

# Or using SLURM
srun --nodes=1 --gres=gpu:2 python train.py --config configs/single_task_overfit.yaml
```

### Resume Training

```bash
python train.py \
    --config configs/single_task_overfit.yaml \
    --resume outputs/TASK_NAME_TIMESTAMP/checkpoint_epoch_100.pth
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir outputs/TASK_NAME_TIMESTAMP/logs

# Watch loss
tail -f outputs/TASK_NAME_TIMESTAMP/logs/events.out.tfevents.*
```

## Inference

### Basic Usage

```bash
python inference.py \
    --checkpoint outputs/TASK_NAME/best_model.pth \
    --config configs/single_task_overfit.yaml \
    --image /path/to/image.png \
    --instruction "put the black bowl on top of the cabinet" \
    --proprio "0.0,0.0,0.0,0.0,0.0,0.0,0.0" \
    --ee-pose "0.5,0.0,0.3,0.0,0.0,0.0,1.0" \
    --num-flow-steps 50
```

### Python API

```python
import torch
import numpy as np
from models import VLAModel
from inference import load_model, preprocess_image, predict_action

# Load model
model, config = load_model(
    checkpoint_path='outputs/best_model.pth',
    config_path='configs/single_task_overfit.yaml',
    device='cuda'
)

# Prepare inputs
image = preprocess_image('observation.png')
instruction = "put the black bowl on top of the cabinet"
proprio = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ee_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])

# Predict action
action = predict_action(
    model=model,
    image=image,
    instruction=instruction,
    proprio=proprio,
    ee_pose=ee_pose,
    num_flow_steps=50,
    device='cuda'
)

print(f"Predicted action: {action}")
```

## Model Details

### Input Specifications

- **RGB Image**: (224, 224, 3), normalized with CLIP statistics
- **Language**: Text string, max 77 tokens
- **Proprioception**: (7,) - joint positions [rad]
- **End-effector Pose**: (7,) - [x, y, z, qx, qy, qz, qw]

### Output Specifications

- **Action**: (7,) - continuous action values
  - For LIBERO: [dx, dy, dz, droll, dpitch, dyaw, gripper]
  - Values are delta transformations

### Model Size

- **Total parameters**: ~100M
  - CLIP Vision: ~85M (frozen)
  - CLIP Text: ~60M (frozen)
  - Trainable: ~5M (encoders + flow network)

### Memory Requirements

- **Training**: ~20GB per GPU (batch_size=64)
- **Inference**: ~2GB

## Hyperparameter Tuning

Key hyperparameters to tune:

1. **Learning Rate**: Start with 3e-4, reduce if unstable
2. **Batch Size**: Increase for better gradient estimates
3. **Flow Steps**: More steps = better quality but slower (inference)
4. **Flow Layers**: Deeper network for complex action distributions
5. **Hidden Dim**: Larger for more expressive representations

## Expected Results (Single Task Overfitting)

- **Epoch 0**: Loss ~1.0
- **Epoch 100**: Loss ~0.1
- **Epoch 500**: Loss ~0.01 (near perfect overfitting)

Training should converge within 200-500 epochs for single task.

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Enable gradient checkpointing (add to model)
- Use mixed precision training

### Loss Not Decreasing
- Check data loading (visualize samples)
- Reduce learning rate
- Increase batch size
- Verify action normalization

### NaN Loss
- Gradient clipping is enabled (max_norm=1.0)
- Reduce learning rate
- Check for invalid data samples

## Project Structure

```
vla-flow/
├── configs/
│   └── single_task_overfit.yaml    # Training configuration
├── data/
│   ├── __init__.py
│   └── libero_dataset.py           # LIBERO dataset loader
├── models/
│   ├── __init__.py
│   └── vla_model.py                # VLA + Rectified Flow
├── outputs/                         # Training outputs (checkpoints, logs)
├── environment.yml                  # Conda environment
├── train.py                         # Training script
├── inference.py                     # Inference script
└── README.md                        # This file
```

## Citation

```bibtex
@article{liu2023rectified,
  title={Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
  author={Liu, Xingchao and Gong, Chengyue and Liu, Qiang},
  journal={arXiv preprint arXiv:2209.03003},
  year={2023}
}

@inproceedings{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  booktitle={NeurIPS},
  year={2023}
}
```

## License

MIT License

## Contributing

Feel free to open issues or submit PRs for improvements!
