"""
Training script for VLA v1.1 with Future Prediction
Supports multi-GPU training with DDP + WandB future visualization
"""

import os
import argparse
import yaml
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb

from models import VLAModelV1
from data.libero_dataset_v1 import create_dataloaders_v1


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def visualize_future_prediction(
    images: torch.Tensor,
    future_images_gt: torch.Tensor,
    decoded_future: torch.Tensor,
    step: int,
    num_samples: int = 4,
):
    """
    Create WandB visualization of future prediction.

    Args:
        images: [B, 3, H, W] current images (normalized)
        future_images_gt: [B, 3, H, W] GT next images (normalized)
        decoded_future: [B, 3, 64, 64] predicted next images (0-1 range)
        step: global step
        num_samples: number of samples to visualize
    """
    # Denormalize current and GT images
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)

    images_viz = images[:num_samples] * std + mean
    future_gt_viz = future_images_gt[:num_samples] * std + mean

    # Decoded images are already in [0, 1] range
    decoded_viz = decoded_future[:num_samples]

    # Resize decoded to match GT size for comparison
    decoded_viz_resized = torch.nn.functional.interpolate(
        decoded_viz,
        size=future_gt_viz.shape[2:],
        mode='bilinear',
        align_corners=False
    )

    # Clip to [0, 1]
    images_viz = torch.clamp(images_viz, 0, 1)
    future_gt_viz = torch.clamp(future_gt_viz, 0, 1)
    decoded_viz_resized = torch.clamp(decoded_viz_resized, 0, 1)

    # Create visualization grid (current, GT future, predicted future)
    wandb_images = []
    for i in range(num_samples):
        # Flip images vertically to correct orientation
        current_img = np.flip(images_viz[i].cpu().numpy().transpose(1, 2, 0), axis=0)
        gt_img = np.flip(future_gt_viz[i].cpu().numpy().transpose(1, 2, 0), axis=0)
        pred_img = np.flip(decoded_viz_resized[i].cpu().numpy().transpose(1, 2, 0), axis=0)

        wandb_images.append(wandb.Image(
            current_img,
            caption=f"Current (t)"
        ))
        wandb_images.append(wandb.Image(
            gt_img,
            caption=f"GT Future (t+1)"
        ))
        wandb_images.append(wandb.Image(
            pred_img,
            caption=f"Predicted Future (t+1)"
        ))

    wandb.log({
        "future_predictions": wandb_images,
    }, step=step)


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    rank: int = 0,
    log_interval: int = 10,
    vis_interval: int = 100,
    writer: SummaryWriter = None,
    global_step: int = 0,
) -> float:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_loss_future = 0.0
    total_loss_action = 0.0
    total_loss_decoder = 0.0
    num_batches = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        disable=(rank != 0)
    )

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(device)
        future_images = batch['image_next'].to(device)  # NEW: t+1 images
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        proprio = batch['proprio'].to(device)
        ee_pose = batch['ee_pose'].to(device)
        actions = batch['action'].to(device)

        # Forward pass
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            proprio=proprio,
            ee_pose=ee_pose,
            actions=actions,
            future_images=future_images,  # NEW: GT future for loss
        )

        loss = outputs['loss']
        loss_future = outputs['loss_future']
        loss_action = outputs['loss_action']
        loss_decoder = outputs['loss_decoder']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_loss_future += loss_future.item()
        total_loss_action += loss_action.item()
        total_loss_decoder += loss_decoder.item()
        num_batches += 1

        # Logging
        if rank == 0:
            pbar.set_postfix({
                'loss': loss.item(),
                'future': loss_future.item(),
                'action': loss_action.item(),
            })

            if writer is not None and (batch_idx % log_interval == 0):
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/loss_future', loss_future.item(), global_step)
                writer.add_scalar('train/loss_action', loss_action.item(), global_step)
                writer.add_scalar('train/loss_decoder', loss_decoder.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

                # Log to WandB
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_loss_future': loss_future.item(),
                    'batch_loss_action': loss_action.item(),
                    'batch_loss_decoder': loss_decoder.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'step': global_step,
                })

            # Visualize future predictions periodically
            if (batch_idx % vis_interval == 0) and (batch_idx > 0):
                print(f"\n🎨 Creating visualization at batch {batch_idx}, step {global_step}...")
                try:
                    with torch.no_grad():
                        visualize_future_prediction(
                            images=images,
                            future_images_gt=future_images,
                            decoded_future=outputs['decoded_future'],
                            step=global_step,
                            num_samples=4,
                        )
                    print(f"✓ Visualization logged to WandB")
                except Exception as e:
                    print(f"✗ Visualization failed: {e}")
                    import traceback
                    traceback.print_exc()

            global_step += 1

    avg_loss = total_loss / num_batches
    avg_loss_future = total_loss_future / num_batches
    avg_loss_action = total_loss_action / num_batches
    avg_loss_decoder = total_loss_decoder / num_batches

    return avg_loss, avg_loss_future, avg_loss_action, avg_loss_decoder, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    rank: int = 0,
) -> tuple:
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    total_loss_future = 0.0
    total_loss_action = 0.0
    total_loss_decoder = 0.0
    num_batches = 0

    pbar = tqdm(
        val_loader,
        desc="Validation",
        disable=(rank != 0)
    )

    for batch in pbar:
        # Move to device
        images = batch['image'].to(device)
        future_images = batch['image_next'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        proprio = batch['proprio'].to(device)
        ee_pose = batch['ee_pose'].to(device)
        actions = batch['action'].to(device)

        # Forward pass
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            proprio=proprio,
            ee_pose=ee_pose,
            actions=actions,
            future_images=future_images,
        )

        loss = outputs['loss']
        loss_future = outputs['loss_future']
        loss_action = outputs['loss_action']
        loss_decoder = outputs['loss_decoder']

        total_loss += loss.item()
        total_loss_future += loss_future.item()
        total_loss_action += loss_action.item()
        total_loss_decoder += loss_decoder.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    if num_batches == 0:
        return float('inf'), float('inf'), float('inf'), float('inf')

    avg_loss = total_loss / num_batches
    avg_loss_future = total_loss_future / num_batches
    avg_loss_action = total_loss_action / num_batches
    avg_loss_decoder = total_loss_decoder / num_batches

    return avg_loss, avg_loss_future, avg_loss_action, avg_loss_decoder


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    save_path: str,
    is_best: bool = False,
):
    """Save model checkpoint"""
    # Get model state dict (handle DDP wrapper)
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
        torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(description='Train VLA v1.1 model with future prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        print(f"Training VLA v1.1 with {world_size} GPUs")
        print(f"Config: {config}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(
        config['training']['output_dir'],
        f"{config['data']['task_name']}_v1_{timestamp}"  # v1 suffix
    )

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        # Save config
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    # Create dataloaders
    if world_size > 1:
        # Use DistributedSampler for multi-GPU
        from data.libero_dataset_v1 import LIBERODatasetV1, collate_fn_v1
        from torch.utils.data import DataLoader

        dataset = LIBERODatasetV1(
            data_path=config['data']['data_path'],
            task_name=config['data']['task_name'],
            augmentation=config['data'].get('augmentation', False),
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        train_loader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            sampler=sampler,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_fn_v1,
            pin_memory=True,
        )

        val_loader = None  # For single task overfitting

    else:
        train_loader, val_loader = create_dataloaders_v1(
            data_path=config['data']['data_path'],
            task_name=config['data']['task_name'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            train_split=config['data'].get('train_split', 1.0),
            augmentation=config['data'].get('augmentation', False),
        )

    # Create model
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
        future_hidden_dim=config['model'].get('future_hidden_dim', 1024),
        future_num_layers=config['model'].get('future_num_layers', 3),
        decoder_output_size=config['model'].get('decoder_output_size', 64),
    )

    model = model.to(device)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params:,}")

    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training'].get('min_lr', 1e-6),
    )

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if args.resume:
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")

        checkpoint = torch.load(args.resume, map_location=device)

        # Load model state dict (handle DDP wrapper)
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # TensorBoard writer
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

        # Initialize WandB
        wandb.init(
            project="babel-vla-mini-train",
            name=f"{config['data']['task_name'][:40]}_v1_{timestamp}",  # v1 suffix
            config=config,
            dir=output_dir,
            tags=["v1.1", "future_prediction"],
        )

    # Training loop
    if rank == 0:
        print("\n" + "="*50)
        print("Starting VLA v1.1 training")
        print("="*50)

    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Set epoch for distributed sampler
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_loss, train_loss_future, train_loss_action, train_loss_decoder, global_step = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            rank=rank,
            log_interval=config['training']['log_interval'],
            vis_interval=config['training'].get('vis_interval', 100),
            writer=writer,
            global_step=global_step,
        )

        # Validate (only for single GPU or rank 0)
        if val_loader is not None and rank == 0 and len(val_loader) > 0:
            val_loss, val_loss_future, val_loss_action, val_loss_decoder = validate(
                model=model,
                val_loader=val_loader,
                device=device,
                rank=rank,
            )
        else:
            val_loss = train_loss
            val_loss_future = train_loss_future
            val_loss_action = train_loss_action
            val_loss_decoder = train_loss_decoder

        # Log
        if rank == 0:
            print(f"Epoch {epoch}: train={train_loss:.4f} (future={train_loss_future:.4f}, action={train_loss_action:.4f}), val={val_loss:.4f}")

            if writer is not None:
                writer.add_scalar('epoch/train_loss', train_loss, epoch)
                writer.add_scalar('epoch/train_loss_future', train_loss_future, epoch)
                writer.add_scalar('epoch/train_loss_action', train_loss_action, epoch)
                writer.add_scalar('epoch/train_loss_decoder', train_loss_decoder, epoch)
                writer.add_scalar('epoch/val_loss', val_loss, epoch)

            # Log to WandB
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_loss_future': train_loss_future,
                'train_loss_action': train_loss_action,
                'train_loss_decoder': train_loss_decoder,
                'val_loss': val_loss,
                'val_loss_future': val_loss_future,
                'val_loss_action': val_loss_action,
                'val_loss_decoder': val_loss_decoder,
                'lr': optimizer.param_groups[0]['lr'],
            })

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            if (epoch + 1) % config['training']['save_interval'] == 0 or is_best:
                save_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    save_path=save_path,
                    is_best=is_best,
                )
                print(f"Saved checkpoint to {save_path}")

        # Step scheduler
        scheduler.step()

    # Cleanup
    if rank == 0:
        if writer is not None:
            writer.close()
        wandb.finish()

    cleanup_distributed()

    if rank == 0:
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Output directory: {output_dir}")
        print("="*50)


if __name__ == '__main__':
    main()
