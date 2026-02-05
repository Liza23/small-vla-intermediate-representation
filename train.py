"""
Training script for VLA with Rectified Flow
Supports multi-GPU training with DDP
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

from models import VLAModel
from data import create_dataloaders


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


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    rank: int = 0,
    log_interval: int = 10,
    writer: SummaryWriter = None,
    global_step: int = 0,
) -> float:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        disable=(rank != 0)
    )

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(device)
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
        )

        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Logging
        if rank == 0:
            pbar.set_postfix({'loss': loss.item()})

            if writer is not None and (batch_idx % log_interval == 0):
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

            global_step += 1

    avg_loss = total_loss / num_batches
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    rank: int = 0,
) -> float:
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        val_loader,
        desc="Validation",
        disable=(rank != 0)
    )

    for batch in pbar:
        # Move to device
        images = batch['image'].to(device)
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
        )

        loss = outputs['loss']

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


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
    parser = argparse.ArgumentParser(description='Train VLA model')
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
        print(f"Training with {world_size} GPUs")
        print(f"Config: {config}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(
        config['training']['output_dir'],
        f"{config['data']['task_name']}_{timestamp}"
    )

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        # Save config
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    # Create dataloaders
    if world_size > 1:
        # Use DistributedSampler for multi-GPU
        from data.libero_dataset import LIBERODataset
        from torch.utils.data import DataLoader

        dataset = LIBERODataset(
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
            pin_memory=True,
        )

        val_loader = None  # For single task overfitting, we don't need validation

    else:
        train_loader, val_loader = create_dataloaders(
            data_path=config['data']['data_path'],
            task_name=config['data']['task_name'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            train_split=config['data'].get('train_split', 1.0),  # Use all data for overfitting
            augmentation=config['data'].get('augmentation', False),
        )

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
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # TensorBoard writer
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

    # Training loop
    if rank == 0:
        print("\n" + "="*50)
        print("Starting training")
        print("="*50)

    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Set epoch for distributed sampler
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_loss, global_step = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            rank=rank,
            log_interval=config['training']['log_interval'],
            writer=writer,
            global_step=global_step,
        )

        # Validate (only for single GPU or rank 0)
        if val_loader is not None and rank == 0:
            val_loss = validate(
                model=model,
                val_loader=val_loader,
                device=device,
                rank=rank,
            )
        else:
            val_loss = train_loss  # Use train loss if no validation

        # Log
        if rank == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if writer is not None:
                writer.add_scalar('epoch/train_loss', train_loss, epoch)
                writer.add_scalar('epoch/val_loss', val_loss, epoch)

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
    if rank == 0 and writer is not None:
        writer.close()

    cleanup_distributed()

    if rank == 0:
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Output directory: {output_dir}")
        print("="*50)


if __name__ == '__main__':
    main()
