"""
Record video of VLA policy rollouts in LIBERO simulation
Similar to OpenVLA visualization
"""

import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import os
import imageio

from models import VLAModel, VLAModelV1, VLAModelV2, VLAModelV3
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def load_model(checkpoint_path: str, config_path: str, device: str = 'cuda', model_version: str = 'v2'):
    """Load trained VLA model"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Select model class based on version
    if model_version == 'v0':
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
    elif model_version == 'v1':
        model = VLAModelV1(
            action_dim=config['model']['action_dim'],
            proprio_dim=config['model']['proprio_dim'],
            ee_dim=config['model']['ee_dim'],
            hidden_dim=config['model']['hidden_dim'],
            flow_hidden_dim=config['model']['flow_hidden_dim'],
            flow_num_layers=config['model']['flow_num_layers'],
            clip_model_name=config['model']['clip_model_name'],
            freeze_vision=config['model']['freeze_vision'],
            freeze_text=config['model']['freeze_text'],
            future_hidden_dim=config['model']['future_hidden_dim'],
            future_num_layers=config['model']['future_num_layers'],
            decoder_output_size=config['model']['decoder_output_size'],
        )
    elif model_version == 'v2':
        model = VLAModelV2(
            action_dim=config['model']['action_dim'],
            proprio_dim=config['model']['proprio_dim'],
            ee_dim=config['model']['ee_dim'],
            hidden_dim=config['model']['hidden_dim'],
            flow_hidden_dim=config['model']['flow_hidden_dim'],
            flow_num_layers=config['model']['flow_num_layers'],
            clip_model_name=config['model']['clip_model_name'],
            freeze_vision=config['model']['freeze_vision'],
            freeze_text=config['model']['freeze_text'],
            pose_predictor_hidden_dim=config['model']['pose_predictor_hidden_dim'],
            pose_predictor_num_layers=config['model']['pose_predictor_num_layers'],
        )
    elif model_version == 'v3':
        model = VLAModelV3(
            action_dim=config['model']['action_dim'],
            proprio_dim=config['model']['proprio_dim'],
            ee_dim=config['model']['ee_dim'],
            hidden_dim=config['model']['hidden_dim'],
            flow_hidden_dim=config['model']['flow_hidden_dim'],
            flow_num_layers=config['model']['flow_num_layers'],
            clip_model_name=config['model']['clip_model_name'],
            freeze_vision=config['model']['freeze_vision'],
            freeze_text=config['model']['freeze_text'],
            future_hidden_dim=config['model']['future_hidden_dim'],
            future_num_layers=config['model']['future_num_layers'],
            decoder_output_size=config['model']['decoder_output_size'],
            pose_predictor_hidden_dim=config['model']['pose_predictor_hidden_dim'],
            pose_predictor_num_layers=config['model']['pose_predictor_num_layers'],
        )
    else:
        raise ValueError(f"Unknown model version: {model_version}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    return model, config


def preprocess_observation(obs, model, env):
    """Preprocess LIBERO observation for model input"""
    # Extract image (agentview)
    if 'agentview_image' in obs:
        image = obs['agentview_image']  # (H, W, 3)
    elif 'agentview_rgb' in obs:
        image = obs['agentview_rgb']
    else:
        # Fallback to eye in hand camera
        image = obs.get('robot0_eye_in_hand_image', np.zeros((256, 256, 3), dtype=np.uint8))

    # Resize to 224x224
    import cv2
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    image = (image - mean) / std

    # Convert to tensor (H, W, C) -> (1, C, H, W)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    # Extract proprio (joint positions)
    # LIBERO env returns observations in robot0_* format
    if 'joint_states' in obs:
        proprio = obs['joint_states']
        ee_states = obs['ee_states']
        gripper = obs['gripper_states'][0]
    else:
        # Alternative key names from robosuite
        proprio = obs.get('robot0_joint_pos', np.zeros(7))
        ee_pos = obs.get('robot0_eef_pos', np.zeros(3))
        ee_quat = obs.get('robot0_eef_quat', np.zeros(4))
        ee_states = np.concatenate([ee_pos, ee_quat[:3]])  # Use first 3 components
        gripper = obs.get('robot0_gripper_qpos', np.zeros(2))[0]

    proprio = torch.from_numpy(np.array(proprio)).unsqueeze(0).float()
    ee_pose = np.concatenate([ee_states, [gripper]])  # (7,)
    ee_pose = torch.from_numpy(ee_pose).unsqueeze(0).float()

    return image, proprio, ee_pose


def record_rollout(
    env,
    model,
    task_description: str,
    num_flow_steps: int = 50,
    max_steps: int = 300,
    device: str = 'cuda',
    video_path: str = 'rollout.mp4',
    fps: int = 20,
):
    """
    Record a video of policy rollout.

    Returns:
        success (bool): Whether the task was completed
        episode_length (int): Number of steps taken
        frames (list): List of RGB frames
    """
    obs = env.reset()
    done = False
    episode_length = 0
    frames = []

    # Tokenize task description
    text_inputs = model.tokenizer(
        task_description,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs['input_ids'].to(device)
    attention_mask = text_inputs['attention_mask'].to(device)

    print(f"Recording rollout: {task_description}")
    print(f"Max steps: {max_steps}")

    while not done and episode_length < max_steps:
        # Get frame from observation (agentview camera)
        if 'agentview_image' in obs:
            frame = obs['agentview_image']  # (H, W, 3)
        elif 'agentview_rgb' in obs:
            frame = obs['agentview_rgb']
        else:
            # Fallback: use robot0_eye_in_hand if agentview not available
            frame = obs.get('robot0_eye_in_hand_image', np.zeros((256, 256, 3), dtype=np.uint8))

        # Flip frame vertically (robosuite images are upside down)
        frame = np.flipud(frame)
        frames.append(frame)

        # Preprocess observation
        image, proprio, ee_pose = preprocess_observation(obs, model, env)
        image = image.to(device)
        proprio = proprio.to(device)
        ee_pose = ee_pose.to(device)

        # Predict action
        with torch.no_grad():
            action = model.predict_action(
                images=image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                proprio=proprio,
                ee_pose=ee_pose,
                num_flow_steps=num_flow_steps,
            )

        action = action.cpu().numpy()[0]  # (7,)

        # Step environment
        obs, reward, done, info = env.step(action)
        episode_length += 1

        # Print progress every 50 steps
        if episode_length % 50 == 0:
            print(f"  Step {episode_length}/{max_steps}")

    success = info.get('success', False)

    # Save video
    print(f"Saving video to: {video_path}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Success: {success}")
    print(f"  Episode length: {episode_length}")

    # Write video
    with imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"✓ Video saved: {video_path}")

    return success, episode_length


def main():
    parser = argparse.ArgumentParser(description='Record VLA policy rollout video')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num-episodes', type=int, default=3, help='Number of episodes to record')
    parser.add_argument('--num-flow-steps', type=int, default=50, help='Number of flow sampling steps')
    parser.add_argument('--max-steps', type=int, default=300, help='Max steps per episode')
    parser.add_argument('--output-dir', type=str, default='./videos', help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=20, help='Video FPS')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--model-version', type=str, default='v2', choices=['v0', 'v1', 'v2', 'v3'],
                        help='Model version to use (v0/v1/v2/v3)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("="*80)
    print(f"LOADING MODEL (Version: {args.model_version})")
    print("="*80)
    model, config = load_model(args.checkpoint, args.config, args.device, args.model_version)

    # Get task info from config
    task_name = config['data']['task_name']
    task_description = task_name.replace('_', ' ')

    # Get benchmark
    print("\n" + "="*80)
    print("SETTING UP LIBERO ENVIRONMENT")
    print("="*80)
    benchmark_dict = benchmark.get_benchmark_dict()
    libero_suite = benchmark_dict['libero_spatial']()

    # Find task index
    task_names = libero_suite.get_task_names()
    task_idx = None
    for i, name in enumerate(task_names):
        if task_name in name:
            task_idx = i
            break

    if task_idx is None:
        raise ValueError(f"Task not found: {task_name}")

    print(f"Task: {task_name}")
    print(f"Description: {task_description}")

    # Create environment with camera observations
    env_args = {
        "bddl_file_name": libero_suite.get_task_bddl_file_path(task_idx),
        "camera_heights": 256,  # Higher resolution for video
        "camera_widths": 256,
        "camera_names": ["agentview", "robot0_eye_in_hand"],  # Include camera views
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,  # Enable camera observations
        "reward_shaping": True,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(42)

    print(f"✓ Environment created")

    # Record episodes
    print("\n" + "="*80)
    print(f"RECORDING {args.num_episodes} EPISODES")
    print("="*80)

    successes = []
    episode_lengths = []

    for episode_idx in range(args.num_episodes):
        print(f"\n--- Episode {episode_idx + 1}/{args.num_episodes} ---")

        video_path = os.path.join(
            args.output_dir,
            f"episode_{episode_idx + 1}.mp4"
        )

        success, length = record_rollout(
            env=env,
            model=model,
            task_description=task_description,
            num_flow_steps=args.num_flow_steps,
            max_steps=args.max_steps,
            device=args.device,
            video_path=video_path,
            fps=args.fps,
        )

        successes.append(success)
        episode_lengths.append(length)

    env.close()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Success Rate: {np.mean(successes)*100:.1f}% ({sum(successes)}/{args.num_episodes})")
    print(f"Avg Episode Length: {np.mean(episode_lengths):.1f} steps")
    print(f"\nVideos saved to: {args.output_dir}/")
    for i in range(args.num_episodes):
        status = "✓ SUCCESS" if successes[i] else "✗ FAILED"
        print(f"  - episode_{i+1}.mp4 ({status}, {episode_lengths[i]} steps)")
    print("="*80)


if __name__ == '__main__':
    main()
