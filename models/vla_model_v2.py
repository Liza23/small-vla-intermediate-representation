"""
VLA v2: Vision-Language-Action Model with Future Gripper Pose Prediction
Simpler and more grounded than v1 - predicts where the end-effector will be
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from typing import Dict, Optional, Tuple
import numpy as np

# Import RectifiedFlow from v0 (reuse)
from .vla_model import RectifiedFlow


class FutureEEPosePredictor(nn.Module):
    """
    Predicts future end-effector pose from current observation.
    Much simpler than v1's visual prediction - only 7D output!
    """
    def __init__(
        self,
        input_dim: int = 512 * 4,  # Concatenated features
        output_dim: int = 7,       # ee_pose: (x, y, z, quat[3], gripper)
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.predictor = nn.Sequential(*layers)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Predict future ee_pose.

        Args:
            fused_features: [B, input_dim] concatenated features

        Returns:
            future_ee_pose: [B, 7] predicted future end-effector pose
        """
        return self.predictor(fused_features)


class VLAModelV2(nn.Module):
    """
    VLA v2: Vision-Language-Action Model with Future Gripper Pose Prediction.

    Architecture:
    1. Encode current observation (vision + text + proprio + ee_pose)
    2. Predict future ee_pose (t+1) - only 7D!
    3. Fuse current features + predicted future ee_pose
    4. Predict action with Rectified Flow

    Changes from v1:
    - Replaced 768D visual latent prediction with 7D gripper pose prediction
    - Simpler, faster, more interpretable
    - Direct geometric grounding
    """
    def __init__(
        self,
        action_dim: int = 7,
        proprio_dim: int = 7,
        ee_dim: int = 7,
        hidden_dim: int = 512,
        flow_hidden_dim: int = 512,
        flow_num_layers: int = 4,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        freeze_vision: bool = True,
        freeze_text: bool = True,
        pose_predictor_hidden_dim: int = 512,
        pose_predictor_num_layers: int = 3,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.ee_dim = ee_dim

        # ====== REUSED FROM V0 ======
        # CLIP Vision Encoder (same as v0)
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, hidden_dim)

        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # CLIP Text Encoder (same as v0)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        # Proprioception encoder (same as v0)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

        # End-effector encoder (same as v0)
        self.ee_encoder = nn.Sequential(
            nn.Linear(ee_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

        # ====== NEW IN V2 ======
        # Future EE pose predictor (much simpler than v1!)
        self.future_ee_predictor = FutureEEPosePredictor(
            input_dim=hidden_dim * 4,  # vision + text + proprio + ee
            output_dim=ee_dim,         # 7D pose
            hidden_dim=pose_predictor_hidden_dim,
            num_layers=pose_predictor_num_layers,
        )

        # Project predicted future ee_pose to hidden_dim for fusion
        self.future_ee_proj = nn.Linear(ee_dim, hidden_dim)

        # Modified fusion layer (includes predicted future ee_pose)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),  # +1 for future ee_pose
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ====== REUSED FROM V0 ======
        # Rectified Flow for action prediction (same as v0)
        self.flow = RectifiedFlow(
            action_dim=action_dim,
            condition_dim=hidden_dim,
            hidden_dim=flow_hidden_dim,
            num_layers=flow_num_layers,
        )

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode RGB images using CLIP vision encoder (same as v0)"""
        vision_outputs = self.vision_encoder(pixel_values=images)
        vision_embeds = vision_outputs.pooler_output  # [B, 768]
        vision_embeds = self.vision_proj(vision_embeds)  # [B, hidden_dim]
        return vision_embeds

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode language using CLIP text encoder (same as v0)"""
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = text_outputs.pooler_output  # [B, 512]
        text_embeds = self.text_proj(text_embeds)  # [B, hidden_dim]
        return text_embeds

    def encode_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        """Encode proprioception (same as v0)"""
        return self.proprio_encoder(proprio)

    def encode_ee(self, ee_pose: torch.Tensor) -> torch.Tensor:
        """Encode end-effector pose (same as v0)"""
        return self.ee_encoder(ee_pose)

    def predict_future_ee_pose(
        self,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        proprio_embeds: torch.Tensor,
        ee_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict future end-effector pose (NEW in v2).

        Args:
            vision_embeds: [B, hidden_dim]
            text_embeds: [B, hidden_dim]
            proprio_embeds: [B, hidden_dim]
            ee_embeds: [B, hidden_dim]

        Returns:
            future_ee_pose: [B, 7] predicted ee_pose for t+1
        """
        # Concatenate all modalities
        fused = torch.cat([
            vision_embeds,
            text_embeds,
            proprio_embeds,
            ee_embeds,
        ], dim=-1)  # [B, hidden_dim * 4]

        # Predict future ee_pose
        future_ee_pose = self.future_ee_predictor(fused)  # [B, 7]

        return future_ee_pose

    def fuse_with_future_ee(
        self,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        proprio_embeds: torch.Tensor,
        ee_embeds: torch.Tensor,
        future_ee_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse all modalities including predicted future ee_pose (MODIFIED from v0).

        Args:
            All embeds: [B, hidden_dim]
            future_ee_pose: [B, 7]

        Returns:
            condition: [B, hidden_dim]
        """
        # Project future ee_pose to hidden_dim
        future_ee_embeds = self.future_ee_proj(future_ee_pose)  # [B, hidden_dim]

        # Concatenate all modalities + future
        fused = torch.cat([
            vision_embeds,
            text_embeds,
            proprio_embeds,
            ee_embeds,
            future_ee_embeds,
        ], dim=-1)  # [B, hidden_dim * 5]

        # Fusion network
        condition = self.fusion(fused)  # [B, hidden_dim]
        return condition

    def forward(
        self,
        images: torch.Tensor,           # [B, C, H, W] current image (t)
        input_ids: torch.Tensor,        # [B, seq_len]
        attention_mask: torch.Tensor,   # [B, seq_len]
        proprio: torch.Tensor,          # [B, proprio_dim]
        ee_pose: torch.Tensor,          # [B, ee_dim]
        actions: Optional[torch.Tensor] = None,        # [B, action_dim]
        future_ee_pose: Optional[torch.Tensor] = None,  # [B, ee_dim] target ee_pose (t+1)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with future gripper pose prediction.

        Returns:
            Dictionary with:
            - 'predicted_future_ee_pose': Predicted future ee_pose [B, 7]
            - 'target_future_ee_pose': GT future ee_pose [B, 7] (if provided)
            - 'condition': Fused condition vector
            - 'loss_ee_pose': EE pose prediction loss (if future_ee_pose provided)
            - 'loss_action': Action prediction loss (if actions provided)
            - 'loss': Total loss (if both provided)
        """
        # Encode all modalities
        vision_embeds = self.encode_vision(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        proprio_embeds = self.encode_proprio(proprio)
        ee_embeds = self.encode_ee(ee_pose)

        # Predict future ee_pose
        predicted_future_ee_pose = self.predict_future_ee_pose(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds
        )

        output = {
            'predicted_future_ee_pose': predicted_future_ee_pose,
        }

        # Compute ee_pose prediction loss if GT provided
        if future_ee_pose is not None:
            output['target_future_ee_pose'] = future_ee_pose

            # MSE loss in ee_pose space
            loss_ee_pose = F.mse_loss(predicted_future_ee_pose, future_ee_pose)
            output['loss_ee_pose'] = loss_ee_pose

        # Fuse with predicted future for action prediction
        condition = self.fuse_with_future_ee(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds, predicted_future_ee_pose
        )
        output['condition'] = condition

        # Compute action loss if actions provided
        if actions is not None:
            loss_action = self.flow.compute_loss(actions, condition)
            output['loss_action'] = loss_action

        # Total loss
        if 'loss_ee_pose' in output and 'loss_action' in output:
            # Weighted combination
            output['loss'] = output['loss_ee_pose'] + output['loss_action']

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
        # Encode
        vision_embeds = self.encode_vision(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        proprio_embeds = self.encode_proprio(proprio)
        ee_embeds = self.encode_ee(ee_pose)

        # Predict future ee_pose
        predicted_future_ee_pose = self.predict_future_ee_pose(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds
        )

        # Fuse with future
        condition = self.fuse_with_future_ee(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds, predicted_future_ee_pose
        )

        # Sample actions using flow
        actions = self.flow.sample(condition, num_steps=num_flow_steps)

        return actions
