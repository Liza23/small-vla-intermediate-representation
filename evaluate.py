"""
Evaluation script for VLA model
Computes prediction metrics on validation/test data
"""

import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from models import VLAModel
from data import create_dataloaders


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    num_flow_steps: int = 50,
) -> dict:
    """
    Evaluate model on a dataset.

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_losses = []
    all_mse = []
    all_mae = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move to device
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        proprio = batch['proprio'].to(device)
        ee_pose = batch['ee_pose'].to(device)
        actions_gt = batch['action'].to(device)

        # Compute loss
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            proprio=proprio,
            ee_pose=ee_pose,
            actions=actions_gt,
        )
        loss = outputs['loss']
        all_losses.append(loss.item())

        # Predict actions
        actions_pred = model.predict_action(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            proprio=proprio,
            ee_pose=ee_pose,
            num_flow_steps=num_flow_steps,
        )

        # Compute metrics
        mse = torch.mean((actions_pred - actions_gt) ** 2, dim=-1)
        mae = torch.mean(torch.abs(actions_pred - actions_gt), dim=-1)

        all_mse.extend(mse.cpu().numpy().tolist())
        all_mae.extend(mae.cpu().numpy().tolist())

    # Aggregate metrics
    metrics = {
        'loss': np.mean(all_losses),
        'mse': np.mean(all_mse),
        'mae': np.mean(all_mae),
        'rmse': np.sqrt(np.mean(all_mse)),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLA model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--num-flow-steps', type=int, default=50, help='Number of flow steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Evaluating model from: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")

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
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Handle DDP wrapped models
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Create dataloader
    train_loader, val_loader = create_dataloaders(
        data_path=config['data']['data_path'],
        task_name=config['data']['task_name'],
        batch_size=args.batch_size,
        num_workers=config['data']['num_workers'],
        train_split=config['data'].get('train_split', 0.9),
    )

    dataloader = train_loader if args.split == 'train' else val_loader

    if dataloader is None or len(dataloader) == 0:
        print(f"No data available for split: {args.split}")
        return

    # Evaluate
    print(f"\nEvaluating on {len(dataloader.dataset)} samples...")
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=args.device,
        num_flow_steps=args.num_flow_steps,
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Split: {args.split}")
    print(f"Number of samples: {len(dataloader.dataset)}")
    print(f"Flow steps: {args.num_flow_steps}")
    print()
    print(f"Loss (Flow Matching): {metrics['loss']:.6f}")
    print(f"MSE (Action):         {metrics['mse']:.6f}")
    print(f"MAE (Action):         {metrics['mae']:.6f}")
    print(f"RMSE (Action):        {metrics['rmse']:.6f}")
    print("="*60)


if __name__ == '__main__':
    main()
