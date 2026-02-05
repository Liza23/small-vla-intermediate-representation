"""
Evaluate VLA policy in LIBERO simulation environment
Rollout the policy and measure success rate
"""

import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import os

from models import VLAModel
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def load_model(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    """Load trained VLA model"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    return model, config


def preprocess_observation(obs, model):
    """Preprocess LIBERO observation for model input"""
    # Extract image (agentview)
    image = obs['agentview_rgb']  # (H, W, 3)

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
    proprio = obs['joint_states']  # (7,)
    proprio = torch.from_numpy(proprio).unsqueeze(0).float()

    # Extract end-effector state
    ee_states = obs['ee_states']  # (6,) - [pos(3), ori(3)]
    gripper = obs['gripper_states'][0]  # First gripper value
    ee_pose = np.concatenate([ee_states, [gripper]])  # (7,)
    ee_pose = torch.from_numpy(ee_pose).unsqueeze(0).float()

    return image, proprio, ee_pose


def rollout_episode(
    env,
    model,
    task_description: str,
    num_flow_steps: int = 50,
    max_steps: int = 300,
    device: str = 'cuda',
    render: bool = False,
):
    """
    Rollout a single episode with the VLA policy.

    Returns:
        success (bool): Whether the task was completed successfully
        episode_length (int): Number of steps taken
        episode_return (float): Cumulative reward
    """
    obs = env.reset()
    done = False
    episode_length = 0
    episode_return = 0.0

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

    while not done and episode_length < max_steps:
        # Preprocess observation
        image, proprio, ee_pose = preprocess_observation(obs, model)
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
        episode_return += reward
        episode_length += 1

        if render:
            env.render()

    success = info.get('success', False)

    return success, episode_length, episode_return


def evaluate_policy(
    model,
    config,
    task_name: str,
    task_description: str,
    num_episodes: int = 10,
    num_flow_steps: int = 50,
    device: str = 'cuda',
    render: bool = False,
):
    """
    Evaluate policy on a LIBERO task.

    Returns:
        results (dict): Success rate, avg episode length, avg return
    """
    # Get benchmark
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

    # Get task
    task = libero_suite.get_task(task_idx)

    # Create environment
    env_args = {
        "bddl_file_name": libero_suite.get_task_bddl_file_path(task_idx),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    print(f"\nEvaluating on task: {task_name}")
    print(f"Task description: {task_description}")
    print(f"Running {num_episodes} episodes...")
    print("=" * 80)

    successes = []
    episode_lengths = []
    episode_returns = []

    for episode_idx in tqdm(range(num_episodes), desc="Rollout"):
        success, length, ret = rollout_episode(
            env=env,
            model=model,
            task_description=task_description,
            num_flow_steps=num_flow_steps,
            device=device,
            render=render,
        )

        successes.append(success)
        episode_lengths.append(length)
        episode_returns.append(ret)

        print(f"Episode {episode_idx + 1}: Success={success}, Length={length}, Return={ret:.2f}")

    env.close()

    # Compute metrics
    success_rate = np.mean(successes) * 100
    avg_length = np.mean(episode_lengths)
    avg_return = np.mean(episode_returns)

    results = {
        'success_rate': success_rate,
        'avg_episode_length': avg_length,
        'avg_episode_return': avg_return,
        'num_episodes': num_episodes,
    }

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Success Rate: {success_rate:.1f}% ({sum(successes)}/{num_episodes})")
    print(f"Avg Episode Length: {avg_length:.1f} steps")
    print(f"Avg Episode Return: {avg_return:.2f}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLA in LIBERO simulation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--num-flow-steps', type=int, default=50, help='Number of flow sampling steps')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, config = load_model(args.checkpoint, args.config, args.device)

    # Get task info from config
    task_name = config['data']['task_name']
    task_description = task_name.replace('_', ' ')

    # Evaluate
    results = evaluate_policy(
        model=model,
        config=config,
        task_name=task_name,
        task_description=task_description,
        num_episodes=args.num_episodes,
        num_flow_steps=args.num_flow_steps,
        device=args.device,
        render=args.render,
    )

    # Save results
    output_dir = os.path.dirname(args.checkpoint)
    results_path = os.path.join(output_dir, 'eval_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
