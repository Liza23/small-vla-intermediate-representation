"""
Vision-Language-Action (VLA) Model with Rectified Flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from typing import Dict, Optional, Tuple
import numpy as np


class RectifiedFlow(nn.Module):
    """
    Rectified Flow for action prediction.
    Uses straight paths between noise and data for efficient sampling.
    """
    def __init__(
        self,
        action_dim: int,
        condition_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_timesteps: int = 1000,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Flow network: predicts velocity field v(x_t, t, condition)
        layers = []
        input_dim = action_dim + condition_dim + hidden_dim

        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ])

        layers.append(nn.Linear(hidden_dim, action_dim))
        self.flow_net = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,  # [B, action_dim]
        t: torch.Tensor,    # [B, 1]
        condition: torch.Tensor,  # [B, condition_dim]
    ) -> torch.Tensor:
        """Predict velocity field v(x_t, t, condition)"""
        # Time embedding
        t_embed = self.time_embed(t)  # [B, hidden_dim]

        # Concatenate inputs
        flow_input = torch.cat([x_t, condition, t_embed], dim=-1)

        # Predict velocity
        v = self.flow_net(flow_input)  # [B, action_dim]
        return v

    def compute_loss(
        self,
        x_1: torch.Tensor,  # Target actions [B, action_dim]
        condition: torch.Tensor,  # [B, condition_dim]
    ) -> torch.Tensor:
        """
        Rectified Flow loss: E[||v(x_t, t) - (x_1 - x_0)||^2]
        where x_t = t * x_1 + (1 - t) * x_0 (straight path)
        """
        batch_size = x_1.shape[0]

        # Sample random timesteps
        t = torch.rand(batch_size, 1, device=x_1.device)

        # Sample noise (x_0)
        x_0 = torch.randn_like(x_1)

        # Compute x_t on the straight path
        x_t = t * x_1 + (1 - t) * x_0

        # Target velocity is simply (x_1 - x_0)
        target_v = x_1 - x_0

        # Predict velocity
        pred_v = self.forward(x_t, t, condition)

        # MSE loss
        loss = F.mse_loss(pred_v, target_v)
        return loss

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,  # [B, condition_dim]
        num_steps: int = 50,
    ) -> torch.Tensor:
        """
        Sample actions using Euler integration.

        Args:
            condition: Conditioning information
            num_steps: Number of integration steps

        Returns:
            Sampled actions [B, action_dim]
        """
        batch_size = condition.shape[0]
        device = condition.device

        # Start from noise
        x_t = torch.randn(batch_size, self.action_dim, device=device)

        # Time steps
        dt = 1.0 / num_steps

        # Euler integration
        for step in range(num_steps):
            t = torch.ones(batch_size, 1, device=device) * (step * dt)

            # Predict velocity
            v = self.forward(x_t, t, condition)

            # Update
            x_t = x_t + v * dt

        return x_t


class VLAModel(nn.Module):
    """
    Vision-Language-Action Model with Rectified Flow.

    Architecture:
    1. CLIP Vision Encoder for RGB images
    2. CLIP Text Encoder for language instructions
    3. MLP for proprioception (joint positions, end-effector pose)
    4. Fusion layer to combine all modalities
    5. Rectified Flow for action prediction
    """
    def __init__(
        self,
        action_dim: int = 7,  # LIBERO: 7-DOF (x, y, z, quat, gripper)
        proprio_dim: int = 7,  # Joint positions
        ee_dim: int = 7,  # End-effector pose (x, y, z, quat)
        hidden_dim: int = 512,
        flow_hidden_dim: int = 512,
        flow_num_layers: int = 4,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        freeze_vision: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # CLIP Vision Encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, hidden_dim)

        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # CLIP Text Encoder
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        # Proprioception encoder (joint positions)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

        # End-effector encoder
        self.ee_encoder = nn.Sequential(
            nn.Linear(ee_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Rectified Flow for action prediction
        self.flow = RectifiedFlow(
            action_dim=action_dim,
            condition_dim=hidden_dim,
            hidden_dim=flow_hidden_dim,
            num_layers=flow_num_layers,
        )

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB images using CLIP vision encoder.

        Args:
            images: [B, C, H, W] RGB images (normalized)

        Returns:
            Vision embeddings [B, hidden_dim]
        """
        vision_outputs = self.vision_encoder(pixel_values=images)
        # Use pooled output (CLS token)
        vision_embeds = vision_outputs.pooler_output  # [B, 768]
        vision_embeds = self.vision_proj(vision_embeds)  # [B, hidden_dim]
        return vision_embeds

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode language instructions using CLIP text encoder.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]

        Returns:
            Text embeddings [B, hidden_dim]
        """
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use pooled output
        text_embeds = text_outputs.pooler_output  # [B, 512]
        text_embeds = self.text_proj(text_embeds)  # [B, hidden_dim]
        return text_embeds

    def encode_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        """Encode proprioception (joint positions)"""
        return self.proprio_encoder(proprio)

    def encode_ee(self, ee_pose: torch.Tensor) -> torch.Tensor:
        """Encode end-effector pose"""
        return self.ee_encoder(ee_pose)

    def fuse_modalities(
        self,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        proprio_embeds: torch.Tensor,
        ee_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse all modality embeddings"""
        # Concatenate all modalities
        fused = torch.cat([
            vision_embeds,
            text_embeds,
            proprio_embeds,
            ee_embeds,
        ], dim=-1)  # [B, hidden_dim * 4]

        # Fusion network
        condition = self.fusion(fused)  # [B, hidden_dim]
        return condition

    def forward(
        self,
        images: torch.Tensor,  # [B, C, H, W]
        input_ids: torch.Tensor,  # [B, seq_len]
        attention_mask: torch.Tensor,  # [B, seq_len]
        proprio: torch.Tensor,  # [B, proprio_dim]
        ee_pose: torch.Tensor,  # [B, ee_dim]
        actions: Optional[torch.Tensor] = None,  # [B, action_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Dictionary with:
            - 'loss': Flow matching loss (if actions provided)
            - 'condition': Fused condition vector
        """
        # Encode all modalities
        vision_embeds = self.encode_vision(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        proprio_embeds = self.encode_proprio(proprio)
        ee_embeds = self.encode_ee(ee_pose)

        # Fuse modalities
        condition = self.fuse_modalities(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds
        )

        output = {'condition': condition}

        # Compute loss if actions provided
        if actions is not None:
            loss = self.flow.compute_loss(actions, condition)
            output['loss'] = loss

        return output

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        proprio: torch.Tensor,
        ee_pose: torch.Tensor,
        num_flow_steps: int = 50,
    ) -> torch.Tensor:
        """
        Predict actions at inference time.

        Returns:
            Predicted actions [B, action_dim]
        """
        # Encode and fuse
        vision_embeds = self.encode_vision(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        proprio_embeds = self.encode_proprio(proprio)
        ee_embeds = self.encode_ee(ee_pose)

        condition = self.fuse_modalities(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds
        )

        # Sample actions using flow
        actions = self.flow.sample(condition, num_steps=num_flow_steps)

        return actions
