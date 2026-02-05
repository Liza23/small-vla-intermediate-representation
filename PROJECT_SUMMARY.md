# VLA with Rectified Flow - Project Summary

## Overview

This project implements a Vision-Language-Action (VLA) model using **Rectified Flow** for robotic manipulation. The model is designed to train on **LIBERO** dataset and can be trained on **2x Ada 6000 GPUs (100GB total VRAM)**.

## Key Design Decisions

### 1. Rectified Flow vs Other Generative Models

**Why Rectified Flow?**
- **Straight path interpolation**: x_t = t * x_1 + (1-t) * x_0 (simplest possible)
- **Fast sampling**: Requires fewer steps than diffusion (50 vs 1000)
- **Training efficiency**: Single loss term, no complex noise schedules
- **Deterministic**: More predictable than score-based models
- **Good for robotics**: Fast inference is critical for real-time control

**Alternatives considered:**
- Diffusion Policy: Slower sampling, more complex training
- VAE: Mode collapse issues
- Direct regression: Doesn't handle multimodal action distributions

### 2. Model Architecture

```
Input Modalities:
┌─────────────────┐
│  RGB Image      │ ──> CLIP ViT-B/16 (frozen) ──> [512]
│  (224x224x3)    │
└─────────────────┘

┌─────────────────┐
│  Language       │ ──> CLIP Text (frozen) ───────> [512]
│  Instruction    │
└─────────────────┘

┌─────────────────┐
│  Joint Pos      │ ──> MLP (trainable) ──────────> [512]
│  (7D)           │
└─────────────────┘

┌─────────────────┐
│  EE Pose        │ ──> MLP (trainable) ──────────> [512]
│  (7D)           │
└─────────────────┘

              ↓
    ┌─────────────────┐
    │  Fusion MLP     │
    │  (trainable)    │
    └─────────────────┘
              ↓
         [512] condition
              ↓
    ┌─────────────────┐
    │ Rectified Flow  │
    │  v(x_t, t, c)   │
    │  (trainable)    │
    └─────────────────┘
              ↓
        Action (7D)
```

**Design rationale:**
- **Frozen CLIP**: Leverage pretrained vision-language alignment, reduce training time
- **Separate encoders**: Each modality gets dedicated processing
- **Fusion layer**: Learn cross-modal interactions
- **Flow network**: Predict velocity field for action generation

### 3. Training Strategy

**Single Task Overfitting First**
- Validate implementation quickly
- Debug data loading and model architecture
- Establish baseline performance
- Then scale to multi-task

**Hyperparameters (2x Ada 6000):**
- Batch size: 64 per GPU (128 total)
- Learning rate: 3e-4 (AdamW)
- Scheduler: Cosine annealing
- Gradient clipping: 1.0
- Expected memory: ~20GB per GPU

### 4. Dataset Integration (LIBERO)

**LIBERO-Spatial chosen because:**
- 10 spatial reasoning tasks
- Good for testing generalization
- Moderate complexity
- Well-documented

**Data loading strategy:**
- Load HDF5 files on-the-fly (memory efficient)
- Random train/val split
- Optional augmentation (brightness/contrast)
- Efficient batching with DataLoader

### 5. Action Representation

**Continuous actions (7-DOF):**
- [dx, dy, dz, droll, dpitch, dyaw, gripper]
- Delta transformations (not absolute poses)
- Easier to learn than discrete tokens
- Natural for robotic control

**Why not discretized actions?**
- More VLA-like but adds complexity
- Requires action binning/unbinning
- Can introduce quantization errors
- Continuous is simpler and works well for single task

## Implementation Details

### Rectified Flow Math

**Training:**
```
1. Sample timestep: t ~ Uniform(0, 1)
2. Sample noise: x_0 ~ N(0, I)
3. Interpolate: x_t = t * x_1 + (1-t) * x_0
4. Target velocity: v_target = x_1 - x_0
5. Predict velocity: v_pred = flow_net(x_t, t, condition)
6. Loss: L = ||v_pred - v_target||^2
```

**Inference (Euler integration):**
```
1. Start from noise: x_0 ~ N(0, I)
2. For t = 0 to 1 (50 steps):
   - Predict velocity: v = flow_net(x_t, t, condition)
   - Update: x_{t+dt} = x_t + v * dt
3. Return final x_1 (predicted action)
```

### Memory Optimization

**For 2x Ada 6000 (50GB each):**
- Freeze CLIP encoders (~145M params): Saves memory and backprop time
- Use gradient checkpointing if needed: Trade compute for memory
- Efficient data loading: Only load current batch
- Mixed precision (future): Can reduce memory by 2x

### Multi-GPU Strategy

**DistributedDataParallel (DDP):**
- Data parallel across GPUs
- Gradients averaged across ranks
- Efficient for large batch training
- Near-linear scaling to 2 GPUs

## File Structure

```
vla-flow/
├── models/
│   ├── vla_model.py         # Main VLA + Rectified Flow
│   └── __init__.py
├── data/
│   ├── libero_dataset.py    # LIBERO dataset loader
│   └── __init__.py
├── configs/
│   └── single_task_overfit.yaml  # Training config
├── utils/
│   ├── visualize_data.py    # Dataset visualization
│   └── __init__.py
├── train.py                 # Training script (DDP support)
├── inference.py             # Inference script
├── evaluate.py              # Evaluation script
├── environment.yml          # Conda environment
├── setup.sh                 # Setup script
├── README.md                # Full documentation
├── QUICKSTART.md            # Quick start guide
└── PROJECT_SUMMARY.md       # This file
```

## Model Size Breakdown

```
Component               Parameters    Memory      Frozen?
────────────────────────────────────────────────────────
CLIP Vision (ViT-B/16)  ~85M         ~340MB      Yes
CLIP Text               ~60M         ~240MB      Yes
Proprio Encoder         ~3.6K        ~14KB       No
EE Encoder              ~3.6K        ~14KB       No
Fusion Layer            ~1.3M        ~5MB        No
Rectified Flow          ~3M          ~12MB       No
────────────────────────────────────────────────────────
Total                   ~149M        ~600MB
Trainable               ~4.3M        ~17MB
```

**Training memory breakdown (batch_size=64, per GPU):**
- Model parameters: ~600MB
- Gradients: ~17MB (trainable only)
- Optimizer states: ~34MB (AdamW has 2 states)
- Activations: ~15GB (majority of memory)
- Input batch: ~3GB (64 × 224 × 224 × 3 images + embeddings)
- **Total**: ~20GB per GPU

## Performance Expectations

### Single Task Overfitting (LIBERO-Spatial)

**Training curves:**
```
Epoch    Loss      Action MSE    Action MAE
──────────────────────────────────────────
0        1.0       0.5          0.4
50       0.3       0.15         0.2
100      0.1       0.05         0.1
200      0.03      0.01         0.05
500      0.005     0.001        0.01
```

**Convergence criteria:**
- Loss < 0.01: Near perfect overfitting
- Action MSE < 0.001: High precision
- Training time: 1-3 hours on 2x Ada 6000

### Inference Speed

- Single action prediction: ~50ms (50 flow steps)
- Faster with fewer steps: ~10ms (10 steps)
- Real-time capable: ~20 Hz control frequency

## Future Extensions

### 1. Multi-Task Training
- Train on all 10 LIBERO-Spatial tasks
- Add task conditioning or prompt tuning
- Test generalization across tasks

### 2. Diffusion Policy Baseline
- Compare Rectified Flow vs Diffusion
- Measure training efficiency and sample quality

### 3. Larger Models
- Unfreeze CLIP and fine-tune
- Use CLIP ViT-L/14 for better features
- Add depth images, force sensors

### 4. Action Tokenization
- Discretize actions into tokens
- True VLA with language model head
- Compare continuous vs discrete

### 5. Real Robot Deployment
- Export to ONNX for faster inference
- Test on physical robot (e.g., Franka Emika)
- Fine-tune with real-world data

## References

**Rectified Flow:**
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2023)

**LIBERO:**
- Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning" (NeurIPS 2023)

**VLA:**
- Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (2023)

**Diffusion Policy:**
- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)

## Questions & Answers

**Q: Why Rectified Flow instead of standard diffusion?**
A: Faster sampling (50 steps vs 1000), simpler training (no noise schedule), and works well for robotics where speed matters.

**Q: Why freeze CLIP?**
A: Pretrained features are strong, freezing reduces memory and compute, and single task doesn't need fine-tuning.

**Q: Can this scale to multi-task?**
A: Yes! Just train on multiple tasks. May want to unfreeze CLIP for better cross-task generalization.

**Q: How to handle multimodal action distributions?**
A: Rectified Flow naturally handles multimodality through its probabilistic formulation. Can sample different actions by varying initial noise.

**Q: What if I have less GPU memory?**
A: Reduce batch_size, use gradient checkpointing, or use mixed precision training (fp16).

**Q: How to deploy on real robot?**
A: Export to ONNX, optimize inference (fewer flow steps), and integrate with robot control loop.

## Acknowledgments

This implementation is inspired by:
- OpenVLA (open-source VLA model)
- Diffusion Policy (action diffusion for robotics)
- RT-2 (vision-language-action models)
- LIBERO benchmark (task suite and dataset)
