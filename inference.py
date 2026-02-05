"""
Inference script for VLA model
"""

import argparse
import yaml
import torch
import numpy as np
from PIL import Image
import cv2

from models import VLAModel


def load_model(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create model
    model = VLAModel(
        action_dim=config['model']['action_dim'],
        proprio_dim=config['model']['proprio_dim'],
        ee_dim=config['model']['ee_dim'],
        hidden_dim=config['model']['hidden_dim'],
        flow_hidden_dim=config['model']['flow_hidden_dim'],
        flow_num_layers=config['model']['flow_num_layers'],
        clip_model_name=config['model']['clip_model_name'],
        freeze_vision=config['model']['freeze_vision'],
        freeze_text=config['model']['freeze_text'],
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle DDP wrapped models
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

    return model, config


def preprocess_image(image_path: str, image_size=(224, 224)):
    """
    Load and preprocess image for model input.

    Args:
        image_path: Path to image file
        image_size: Target size (H, W)

    Returns:
        Preprocessed image tensor [1, 3, H, W]
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)

    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    image = (image - mean) / std

    # Convert to tensor
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = image.unsqueeze(0)  # Add batch dimension

    return image


@torch.no_grad()
def predict_action(
    model: VLAModel,
    image: torch.Tensor,
    instruction: str,
    proprio: np.ndarray,
    ee_pose: np.ndarray,
    num_flow_steps: int = 50,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Predict action given observation.

    Args:
        model: VLA model
        image: Preprocessed image [1, 3, H, W]
        instruction: Language instruction
        proprio: Robot proprioception [7]
        ee_pose: End-effector pose [7]
        num_flow_steps: Number of flow sampling steps
        device: Device

    Returns:
        Predicted action [7]
    """
    # Move to device
    image = image.to(device)

    # Tokenize instruction
    text_inputs = model.tokenizer(
        instruction,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs['input_ids'].to(device)
    attention_mask = text_inputs['attention_mask'].to(device)

    # Convert proprio and ee_pose to tensors
    proprio = torch.from_numpy(proprio).float().unsqueeze(0).to(device)
    ee_pose = torch.from_numpy(ee_pose).float().unsqueeze(0).to(device)

    # Predict action
    action = model.predict_action(
        images=image,
        input_ids=input_ids,
        attention_mask=attention_mask,
        proprio=proprio,
        ee_pose=ee_pose,
        num_flow_steps=num_flow_steps,
    )

    return action.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description='VLA inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--instruction', type=str, required=True, help='Language instruction')
    parser.add_argument('--proprio', type=str, required=True, help='Joint positions (comma-separated)')
    parser.add_argument('--ee-pose', type=str, required=True, help='EE pose (x,y,z,qx,qy,qz,qw)')
    parser.add_argument('--num-flow-steps', type=int, default=50, help='Number of flow steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    # Load model
    model, config = load_model(args.checkpoint, args.config, args.device)

    # Preprocess image
    image = preprocess_image(args.image)

    # Parse proprio and ee_pose
    proprio = np.array([float(x) for x in args.proprio.split(',')])
    ee_pose = np.array([float(x) for x in args.ee_pose.split(',')])

    assert len(proprio) == 7, "Proprio must have 7 values"
    assert len(ee_pose) == 7, "EE pose must have 7 values"

    # Predict action
    print("\nPredicting action...")
    print(f"Instruction: {args.instruction}")
    print(f"Proprio: {proprio}")
    print(f"EE pose: {ee_pose}")

    action = predict_action(
        model=model,
        image=image,
        instruction=args.instruction,
        proprio=proprio,
        ee_pose=ee_pose,
        num_flow_steps=args.num_flow_steps,
        device=args.device,
    )

    print(f"\nPredicted action: {action}")
    print(f"Action shape: {action.shape}")


if __name__ == '__main__':
    main()
