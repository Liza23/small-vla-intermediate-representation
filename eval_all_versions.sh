#!/bin/bash
# Evaluate all VLA model versions (v0, v1, v2, v3) and generate rollout videos

set -e  # Exit on error

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla-flow

# Common parameters
NUM_EPISODES=1
NUM_FLOW_STEPS=1000
MAX_STEPS=1000  # Increased from 300 to 500
FPS=20
DEVICE=cuda

# Base directory
BASE_DIR="/home/ldahiya/max_vla/small-vla-intermediate-representation"
cd $BASE_DIR

echo "========================================"
echo "Evaluating All VLA Model Versions"
echo "========================================"
echo ""

# V0 (Base Model)
# echo "========== Evaluating V0 (Base Model) =========="
# python record_rollout_video.py \
#   --checkpoint /data/user_data/ldahiya/outputs/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_20260205_004747/best_model.pth \
#   --config configs/single_task_overfit.yaml \
#   --num-episodes $NUM_EPISODES \
#   --num-flow-steps $NUM_FLOW_STEPS \
#   --max-steps $MAX_STEPS \
#   --output-dir ./videos/eval_rollouts_v0 \
#   --fps $FPS \
#   --device $DEVICE \
#   --model-version v0

echo ""
echo "========== Evaluating V1 (Future Latent Prediction) =========="
python record_rollout_video.py \
  --checkpoint /data/user_data/ldahiya/outputs_v1/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_v1_20260206_023820/best_model.pth \
  --config configs/single_task_overfit_v1.yaml \
  --num-episodes $NUM_EPISODES \
  --num-flow-steps $NUM_FLOW_STEPS \
  --max-steps $MAX_STEPS \
  --output-dir ./videos/eval_rollouts_v1 \
  --fps $FPS \
  --device $DEVICE \
  --model-version v1

echo ""
echo "========== Evaluating V2 (Future EE Pose Prediction) =========="
python record_rollout_video.py \
  --checkpoint /data/user_data/ldahiya/outputs_v2/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_v2_20260206_002615/best_model.pth  \
  --config configs/single_task_overfit_v2.yaml \
  --num-episodes $NUM_EPISODES \
  --num-flow-steps $NUM_FLOW_STEPS \
  --max-steps $MAX_STEPS \
  --output-dir ./videos/eval_rollouts_v2 \
  --fps $FPS \
  --device $DEVICE \
  --model-version v2

echo ""
echo "========== Evaluating V3 (Future State Renderer) =========="
python record_rollout_video.py \
  --checkpoint /data/user_data/ldahiya/outputs_v3/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_v3_20260205_220525/best_model.pth \
  --config configs/single_task_overfit_v3.yaml \
  --num-episodes $NUM_EPISODES \
  --num-flow-steps $NUM_FLOW_STEPS \
  --max-steps $MAX_STEPS \
  --output-dir ./videos/eval_rollouts_v3 \
  --fps $FPS \
  --device $DEVICE \
  --model-version v3

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - ./videos/eval_rollouts_v0/"
echo "  - ./videos/eval_rollouts_v1/"
echo "  - ./videos/eval_rollouts_v2/"
echo "  - ./videos/eval_rollouts_v3/"
echo ""
