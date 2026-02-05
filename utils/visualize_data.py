"""
Utility script to visualize LIBERO dataset samples
"""

import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import LIBERODataset


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """
    Denormalize image for visualization.

    Args:
        image: [3, H, W] normalized tensor

    Returns:
        [H, W, 3] uint8 array
    """
    # CLIP normalization stats
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # Denormalize
    image = image * std + mean

    # Clip to [0, 1]
    image = torch.clamp(image, 0, 1)

    # Convert to numpy and transpose
    image = image.permute(1, 2, 0).numpy()

    # Convert to uint8
    image = (image * 255).astype(np.uint8)

    return image


def visualize_batch(dataset: LIBERODataset, num_samples: int = 4):
    """Visualize a few samples from the dataset"""

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 4, 8))

    for i in range(num_samples):
        # Get random sample
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]

        # Denormalize image
        image = denormalize_image(sample['image'])

        # Plot image
        if num_samples == 1:
            ax_img = axes[0]
            ax_text = axes[1]
        else:
            ax_img = axes[0, i]
            ax_text = axes[1, i]

        ax_img.imshow(image)
        ax_img.axis('off')
        ax_img.set_title(f"Sample {idx}")

        # Display information
        info_text = f"Task: {dataset.task_description}\n\n"
        info_text += f"Proprio: {sample['proprio'].numpy()}\n\n"
        info_text += f"EE Pose: {sample['ee_pose'].numpy()}\n\n"
        info_text += f"Action: {sample['action'].numpy()}"

        ax_text.text(0.1, 0.5, info_text, fontsize=8, verticalalignment='center',
                     family='monospace')
        ax_text.axis('off')

    plt.tight_layout()
    plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to dataset_visualization.png")
    plt.show()


def print_dataset_stats(dataset: LIBERODataset):
    """Print statistics about the dataset"""
    print("="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Task: {dataset.task_name}")
    print(f"Description: {dataset.task_description}")
    print(f"Number of trajectories: {len(dataset.trajectories)}")
    print(f"Total timesteps: {len(dataset)}")
    print()

    # Compute action statistics
    all_actions = []
    for traj in dataset.trajectories:
        all_actions.append(traj['actions'])

    all_actions = np.concatenate(all_actions, axis=0)

    print("Action statistics:")
    print(f"  Shape: {all_actions.shape}")
    print(f"  Mean: {all_actions.mean(axis=0)}")
    print(f"  Std: {all_actions.std(axis=0)}")
    print(f"  Min: {all_actions.min(axis=0)}")
    print(f"  Max: {all_actions.max(axis=0)}")
    print()

    # Compute proprio statistics
    all_proprio = []
    for traj in dataset.trajectories:
        all_proprio.append(traj['proprio'])

    all_proprio = np.concatenate(all_proprio, axis=0)

    print("Proprioception statistics:")
    print(f"  Shape: {all_proprio.shape}")
    print(f"  Mean: {all_proprio.mean(axis=0)}")
    print(f"  Std: {all_proprio.std(axis=0)}")
    print(f"  Min: {all_proprio.min(axis=0)}")
    print(f"  Max: {all_proprio.max(axis=0)}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Visualize LIBERO dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=4, help='Number of samples to visualize')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create dataset
    print("Loading dataset...")
    dataset = LIBERODataset(
        data_path=config['data']['data_path'],
        task_name=config['data']['task_name'],
    )

    # Print statistics
    print_dataset_stats(dataset)

    # Visualize samples
    print(f"\nVisualizing {args.num_samples} random samples...")
    visualize_batch(dataset, num_samples=args.num_samples)


if __name__ == '__main__':
    main()
