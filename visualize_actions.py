"""
Visualize VLA action predictions vs ground truth
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models import VLAModel
from data import LIBERODataset

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


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

    return model, config


def collect_predictions(
    model,
    dataset,
    num_samples: int = 500,
    num_flow_steps: int = 50,
    device: str = 'cuda',
):
    """
    Collect action predictions and ground truth.

    Returns:
        pred_actions: (N, 7) predicted actions
        gt_actions: (N, 7) ground truth actions
    """
    pred_actions = []
    gt_actions = []

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Collecting predictions"):
        sample = dataset[idx]

        # Move to device
        image = sample['image'].unsqueeze(0).to(device)
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        proprio = sample['proprio'].unsqueeze(0).to(device)
        ee_pose = sample['ee_pose'].unsqueeze(0).to(device)
        gt_action = sample['action'].numpy()

        # Predict
        with torch.no_grad():
            pred_action = model.predict_action(
                images=image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                proprio=proprio,
                ee_pose=ee_pose,
                num_flow_steps=num_flow_steps,
            )

        pred_actions.append(pred_action.cpu().numpy()[0])
        gt_actions.append(gt_action)

    pred_actions = np.array(pred_actions)
    gt_actions = np.array(gt_actions)

    return pred_actions, gt_actions


def plot_action_comparison(pred_actions, gt_actions, save_path='action_comparison.png'):
    """
    Plot predicted vs ground truth actions for each dimension.
    """
    action_names = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(7):
        ax = axes[i]

        # Scatter plot
        ax.scatter(gt_actions[:, i], pred_actions[:, i], alpha=0.3, s=10)

        # Perfect prediction line
        min_val = min(gt_actions[:, i].min(), pred_actions[:, i].min())
        max_val = max(gt_actions[:, i].max(), pred_actions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        # Labels
        ax.set_xlabel(f'Ground Truth {action_names[i]}', fontsize=12)
        ax.set_ylabel(f'Predicted {action_names[i]}', fontsize=12)
        ax.set_title(f'{action_names[i]} (MSE: {np.mean((pred_actions[:, i] - gt_actions[:, i])**2):.4f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[7])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_error_distribution(pred_actions, gt_actions, save_path='error_distribution.png'):
    """
    Plot error distributions for each action dimension.
    """
    action_names = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
    errors = pred_actions - gt_actions

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(7):
        ax = axes[i]

        # Histogram
        ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')

        # Stats
        mean_error = np.mean(errors[:, i])
        std_error = np.std(errors[:, i])

        ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}')
        ax.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero Error')

        ax.set_xlabel(f'Error ({action_names[i]})', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{action_names[i]} Error (μ={mean_error:.4f}, σ={std_error:.4f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[7])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_trajectory_comparison(pred_actions, gt_actions, traj_length=100, save_path='trajectory_comparison.png'):
    """
    Plot a sample trajectory showing predicted vs ground truth actions over time.
    """
    action_names = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']

    # Take first trajectory
    pred_traj = pred_actions[:traj_length]
    gt_traj = gt_actions[:traj_length]

    fig, axes = plt.subplots(7, 1, figsize=(14, 18))

    for i in range(7):
        ax = axes[i]

        timesteps = np.arange(traj_length)
        ax.plot(timesteps, gt_traj[:, i], 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax.plot(timesteps, pred_traj[:, i], 'r--', linewidth=2, label='Predicted', alpha=0.7)

        ax.set_ylabel(action_names[i], fontsize=12)
        ax.set_title(f'{action_names[i]} Over Time', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Timestep', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_correlation_matrix(pred_actions, gt_actions, save_path='correlation_matrix.png'):
    """
    Plot correlation between predicted and ground truth for each dimension.
    """
    action_names = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']

    correlations = []
    for i in range(7):
        corr = np.corrcoef(pred_actions[:, i], gt_actions[:, i])[0, 1]
        correlations.append(corr)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(action_names, correlations, color=plt.cm.RdYlGn([(c + 1) / 2 for c in correlations]))
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Perfect Correlation')
    ax.axhline(y=0.9, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good Correlation')

    ax.set_ylabel('Correlation Coefficient', fontsize=14)
    ax.set_xlabel('Action Dimension', fontsize=14)
    ax.set_title('Prediction Correlation by Action Dimension', fontsize=16)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def print_summary_stats(pred_actions, gt_actions):
    """Print summary statistics"""
    action_names = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
    errors = pred_actions - gt_actions

    print("\n" + "="*80)
    print("ACTION PREDICTION STATISTICS")
    print("="*80)
    print(f"{'Dimension':<10} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Correlation':<12}")
    print("-"*80)

    for i in range(7):
        mse = np.mean(errors[:, i]**2)
        mae = np.mean(np.abs(errors[:, i]))
        rmse = np.sqrt(mse)
        corr = np.corrcoef(pred_actions[:, i], gt_actions[:, i])[0, 1]

        print(f"{action_names[i]:<10} {mse:<12.6f} {mae:<12.6f} {rmse:<12.6f} {corr:<12.6f}")

    # Overall stats
    overall_mse = np.mean(errors**2)
    overall_mae = np.mean(np.abs(errors))
    overall_rmse = np.sqrt(overall_mse)

    print("-"*80)
    print(f"{'Overall':<10} {overall_mse:<12.6f} {overall_mae:<12.6f} {overall_rmse:<12.6f} {'-':<12}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize VLA action predictions')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of samples to visualize')
    parser.add_argument('--num-flow-steps', type=int, default=50, help='Number of flow steps')
    parser.add_argument('--output-dir', type=str, default='./visualizations', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("Loading model and dataset...")
    model, config = load_model(args.checkpoint, args.config, args.device)

    dataset = LIBERODataset(
        data_path=config['data']['data_path'],
        task_name=config['data']['task_name'],
    )

    print(f"Collecting predictions from {args.num_samples} samples...")
    pred_actions, gt_actions = collect_predictions(
        model=model,
        dataset=dataset,
        num_samples=args.num_samples,
        num_flow_steps=args.num_flow_steps,
        device=args.device,
    )

    # Print statistics
    print_summary_stats(pred_actions, gt_actions)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_action_comparison(
        pred_actions, gt_actions,
        save_path=os.path.join(args.output_dir, 'action_comparison.png')
    )

    plot_error_distribution(
        pred_actions, gt_actions,
        save_path=os.path.join(args.output_dir, 'error_distribution.png')
    )

    plot_trajectory_comparison(
        pred_actions, gt_actions,
        save_path=os.path.join(args.output_dir, 'trajectory_comparison.png')
    )

    plot_correlation_matrix(
        pred_actions, gt_actions,
        save_path=os.path.join(args.output_dir, 'correlation_matrix.png')
    )

    print(f"\n✓ All visualizations saved to: {args.output_dir}/")
    print("  - action_comparison.png       (Scatter plots)")
    print("  - error_distribution.png      (Error histograms)")
    print("  - trajectory_comparison.png   (Time series)")
    print("  - correlation_matrix.png      (Correlation bar chart)")


if __name__ == '__main__':
    main()
