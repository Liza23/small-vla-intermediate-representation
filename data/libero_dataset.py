"""
LIBERO Dataset Loader for VLA Training
"""

import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from transformers import CLIPTokenizer


class LIBERODataset(Dataset):
    """
    PyTorch Dataset for LIBERO.

    Loads a single task for overfitting experiments.
    """
    def __init__(
        self,
        data_path: str,
        task_name: str,
        split: str = "train",
        tokenizer_name: str = "openai/clip-vit-base-patch16",
        image_size: Tuple[int, int] = (224, 224),
        max_length: int = 77,
        augmentation: bool = False,
    ):
        """
        Args:
            data_path: Path to LIBERO dataset root
            task_name: Name of the task (e.g., 'LIVING_ROOM_SCENE0_put_the_black_bowl_on_top_of_the_cabinet')
            split: 'train' or 'val'
            tokenizer_name: CLIP tokenizer name
            image_size: Target image size (H, W)
            max_length: Max sequence length for tokenization
            augmentation: Whether to apply data augmentation
        """
        super().__init__()

        self.data_path = data_path
        self.task_name = task_name
        self.split = split
        self.image_size = image_size
        self.max_length = max_length
        self.augmentation = augmentation

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)

        # Load dataset
        self.trajectories = []
        self.task_description = ""
        self._load_data()

        print(f"Loaded {len(self.trajectories)} trajectories for task: {task_name}")
        print(f"Task description: {self.task_description}")

    def _load_data(self):
        """Load LIBERO HDF5 dataset"""
        # LIBERO dataset structure:
        # data/
        #   libero_spatial/
        #     {task_name}_demo.hdf5

        # Find the demo file for this task
        demo_file = os.path.join(self.data_path, "libero_spatial", f"{self.task_name}_demo.hdf5")

        if not os.path.exists(demo_file):
            raise ValueError(f"Demo file not found: {demo_file}")

        # Load all demos from the file
        self._load_trajectory(demo_file)

    def _load_trajectory(self, hdf5_path: str):
        """Load all demonstrations from HDF5 file"""
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # LIBERO structure: data/demo_0, data/demo_1, ...
                data_group = f['data']
                num_demos = len(data_group.keys())

                print(f"Loading {num_demos} demonstrations from {os.path.basename(hdf5_path)}")

                # Extract task description from filename
                self.task_description = self.task_name.replace('_', ' ')

                # Load each demo
                for demo_idx in range(num_demos):
                    demo_key = f'demo_{demo_idx}'
                    if demo_key not in data_group:
                        continue

                    demo = data_group[demo_key]

                    # Extract observations
                    # LIBERO stores:
                    # - agentview_rgb: (T, H, W, 3)
                    # - joint_states: (T, 7)
                    # - ee_states: (T, 6) - [pos(3), ori(3)]
                    # - gripper_states: (T, 2)

                    images = np.array(demo['obs']['agentview_rgb'])  # (T, H, W, 3)
                    joint_states = np.array(demo['obs']['joint_states'])  # (T, 7)
                    ee_states = np.array(demo['obs']['ee_states'])  # (T, 6)
                    gripper_states = np.array(demo['obs']['gripper_states'])  # (T, 2)

                    # Actions
                    actions = np.array(demo['actions'])  # (T, 7)

                    # For ee_pose, we'll use ee_states (6D) and pad with a dummy value
                    # to make it 7D for consistency with model expectations
                    # Or we can adjust model to accept 6D
                    # For now, let's pad with gripper state
                    ee_pose = np.concatenate([ee_states, gripper_states[:, :1]], axis=-1)  # (T, 7)

                    # Store trajectory
                    traj = {
                        'images': images,
                        'proprio': joint_states,
                        'ee_pose': ee_pose,
                        'actions': actions,
                        'gripper': gripper_states,
                        'length': len(images),
                    }

                    self.trajectories.append(traj)

        except Exception as e:
            print(f"Error loading {hdf5_path}: {e}")
            import traceback
            traceback.print_exc()

    def __len__(self) -> int:
        """Total number of timesteps across all trajectories"""
        return sum(traj['length'] for traj in self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single transition.

        Returns:
            Dictionary with:
            - 'image': [3, H, W] normalized RGB image
            - 'input_ids': [max_length] tokenized text
            - 'attention_mask': [max_length] attention mask
            - 'proprio': [7] joint positions
            - 'ee_pose': [7] end-effector pose
            - 'action': [7] target action
        """
        # Find trajectory and timestep
        traj_idx = 0
        timestep = idx

        for i, traj in enumerate(self.trajectories):
            if timestep < traj['length']:
                traj_idx = i
                break
            timestep -= traj['length']

        traj = self.trajectories[traj_idx]

        # Get data
        image = traj['images'][timestep]  # (H, W, 3)
        proprio = traj['proprio'][timestep]  # (7,)
        ee_pose = traj['ee_pose'][timestep]  # (7,)
        action = traj['actions'][timestep]  # (7,)

        # Process image
        image = self._process_image(image)  # (3, H, W)

        # Tokenize text
        text_inputs = self.tokenizer(
            self.task_description,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            'image': image,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'proprio': torch.from_numpy(proprio).float(),
            'ee_pose': torch.from_numpy(ee_pose).float(),
            'action': torch.from_numpy(action).float(),
        }

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Process image: resize, normalize, and convert to tensor.

        Args:
            image: (H, W, 3) uint8 array

        Returns:
            (3, H, W) normalized tensor
        """
        # Resize
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

        # Data augmentation (optional)
        if self.augmentation:
            # Random brightness/contrast
            if np.random.rand() < 0.5:
                alpha = np.random.uniform(0.8, 1.2)  # Contrast
                beta = np.random.uniform(-0.1, 0.1)  # Brightness
                image = np.clip(alpha * image + beta * 255, 0, 255).astype(np.uint8)

        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Normalize using CLIP stats
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        image = (image - mean) / std

        # Convert to tensor (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'proprio': torch.stack([item['proprio'] for item in batch]),
        'ee_pose': torch.stack([item['ee_pose'] for item in batch]),
        'action': torch.stack([item['action'] for item in batch]),
    }


def create_dataloaders(
    data_path: str,
    task_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.9,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_path: Path to LIBERO dataset
        task_name: Task name
        batch_size: Batch size
        num_workers: Number of dataloader workers
        train_split: Fraction of data for training

    Returns:
        (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = LIBERODataset(
        data_path=data_path,
        task_name=task_name,
        **kwargs
    )

    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
