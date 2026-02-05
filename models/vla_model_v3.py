"""
VLA v3.0: Vision-Language-Action Model with Future Image + Gripper Pose Prediction
Combines V1 (future image) + V2 (future EE pose) → Rendered future state → Action

Architecture progression:
- V3.0: Feature concatenation (simple)
- V3.1: Cross-attention (future upgrade)
- V3.2: Spatial fusion (future upgrade)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from typing import Dict, Optional, Tuple
import numpy as np

# Import RectifiedFlow from v0 (reuse)
from .vla_model import RectifiedFlow


class FutureLatentPredictor(nn.Module):
    """
    Predicts future visual latent from current observation.
    (Reused from V1)
    """
    def __init__(
        self,
        input_dim: int = 512 * 4,  # Concatenated features
        output_dim: int = 768,     # CLIP visual latent dim
        hidden_dim: int = 1024,
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
        Predict future visual latent.

        Args:
            fused_features: [B, input_dim] concatenated features

        Returns:
            future_latent: [B, 768] CLIP visual latent
        """
        return self.predictor(fused_features)


class FutureLatentDecoder(nn.Module):
    """
    Decodes future visual latent to RGB image.
    (Reused from V1)
    """
    def __init__(
        self,
        latent_dim: int = 768,
        output_size: int = 64,
    ):
        super().__init__()

        self.output_size = output_size

        # Decoder: latent → image
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size * output_size * 3),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to RGB image.

        Args:
            latent: [B, 768]

        Returns:
            image: [B, 3, H, W] in [0, 1] range
        """
        B = latent.shape[0]

        # Decode
        flat = self.decoder(latent)  # [B, H*W*3]

        # Reshape to image
        image = flat.view(B, 3, self.output_size, self.output_size)

        # Sigmoid to [0, 1]
        image = torch.sigmoid(image)

        return image


class FutureEEPosePredictor(nn.Module):
    """
    Predicts future end-effector pose from current observation.
    (Reused from V2)
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


class FutureStateRenderer(nn.Module):
    """
    V3.0: Renders future state by concatenating visual latent + EE pose.

    Future upgrades:
    - V3.1: Cross-attention between visual and pose
    - V3.2: Spatial fusion with positional encoding
    """
    def __init__(
        self,
        visual_latent_dim: int = 768,
        ee_pose_dim: int = 7,
        output_dim: int = 512,
    ):
        super().__init__()

        # V3.0: Simple concatenation + MLP fusion
        self.fusion = nn.Sequential(
            nn.Linear(visual_latent_dim + ee_pose_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        visual_latent: torch.Tensor,  # [B, 768]
        ee_pose: torch.Tensor,         # [B, 7]
    ) -> torch.Tensor:
        """
        Render future state from visual latent + EE pose.

        Returns:
            rendered_state: [B, 512]
        """
        # V3.0: Concatenate and fuse
        combined = torch.cat([visual_latent, ee_pose], dim=-1)  # [B, 775]
        rendered = self.fusion(combined)  # [B, 512]

        return rendered


class VLAModelV3(nn.Module):
    """
    VLA v3.0: Vision-Language-Action Model with Future Image + Gripper Pose Prediction.

    Architecture:
    1. Encode current observation (vision + text + proprio + ee_pose)
    2. Predict future visual latent (t+1) - from V1
    3. Predict future ee_pose (t+1) - from V2
    4. Render future state (V3.0: concatenate visual + pose)
    5. Fuse current features + rendered future
    6. Predict action with Rectified Flow

    Key innovation: Model must "imagine" both what it will see AND where it will be
    before deciding on actions.
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
        # Future visual prediction (from V1)
        future_hidden_dim: int = 1024,
        future_num_layers: int = 3,
        decoder_output_size: int = 64,
        # Future EE pose prediction (from V2)
        pose_predictor_hidden_dim: int = 512,
        pose_predictor_num_layers: int = 3,
        # Loss weights (tunable)
        future_loss_weight: float = 0.5,
        action_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.ee_dim = ee_dim
        self.future_loss_weight = future_loss_weight
        self.action_loss_weight = action_loss_weight

        # ====== ENCODERS (same as V0/V1/V2) ======
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

        # Proprioception encoder
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

        # ====== FUTURE PREDICTION (from V1 + V2) ======
        # Future visual latent predictor (from V1)
        self.future_latent_predictor = FutureLatentPredictor(
            input_dim=hidden_dim * 4,
            output_dim=768,  # CLIP visual latent
            hidden_dim=future_hidden_dim,
            num_layers=future_num_layers,
        )

        # Future visual decoder (from V1)
        self.future_decoder = FutureLatentDecoder(
            latent_dim=768,
            output_size=decoder_output_size,
        )

        # Future EE pose predictor (from V2)
        self.future_ee_predictor = FutureEEPosePredictor(
            input_dim=hidden_dim * 4,
            output_dim=ee_dim,
            hidden_dim=pose_predictor_hidden_dim,
            num_layers=pose_predictor_num_layers,
        )

        # ====== NEW IN V3: FUTURE STATE RENDERER ======
        self.future_renderer = FutureStateRenderer(
            visual_latent_dim=768,
            ee_pose_dim=ee_dim,
            output_dim=hidden_dim,
        )

        # ====== FUSION LAYER (modified from V0) ======
        # Fuses current observation + rendered future state
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),  # vision + text + proprio + ee + rendered_future
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ====== ACTION PREDICTION (same as V0/V1/V2) ======
        self.flow = RectifiedFlow(
            action_dim=action_dim,
            condition_dim=hidden_dim,
            hidden_dim=flow_hidden_dim,
            num_layers=flow_num_layers,
        )

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode RGB images using CLIP vision encoder"""
        vision_outputs = self.vision_encoder(pixel_values=images)
        vision_embeds = vision_outputs.pooler_output  # [B, 768]
        vision_embeds = self.vision_proj(vision_embeds)  # [B, hidden_dim]
        return vision_embeds

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode language using CLIP text encoder"""
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = text_outputs.pooler_output  # [B, 512]
        text_embeds = self.text_proj(text_embeds)  # [B, hidden_dim]
        return text_embeds

    def encode_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        """Encode proprioception"""
        return self.proprio_encoder(proprio)

    def encode_ee(self, ee_pose: torch.Tensor) -> torch.Tensor:
        """Encode end-effector pose"""
        return self.ee_encoder(ee_pose)

    def predict_future(
        self,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        proprio_embeds: torch.Tensor,
        ee_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict future visual latent AND future EE pose.

        Returns:
            future_latent: [B, 768]
            decoded_future: [B, 3, 64, 64]
            future_ee_pose: [B, 7]
        """
        # Concatenate all current modalities
        fused = torch.cat([
            vision_embeds,
            text_embeds,
            proprio_embeds,
            ee_embeds,
        ], dim=-1)  # [B, hidden_dim * 4]

        # Predict future visual latent (from V1)
        future_latent = self.future_latent_predictor(fused)  # [B, 768]

        # Decode to image for visualization (from V1)
        decoded_future = self.future_decoder(future_latent)  # [B, 3, 64, 64]

        # Predict future EE pose (from V2)
        future_ee_pose = self.future_ee_predictor(fused)  # [B, 7]

        return future_latent, decoded_future, future_ee_pose

    def render_future_state(
        self,
        future_latent: torch.Tensor,
        future_ee_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        V3.0: Render future state by combining visual latent + EE pose.

        Returns:
            rendered_future: [B, 512]
        """
        return self.future_renderer(future_latent, future_ee_pose)

    def forward(
        self,
        images: torch.Tensor,           # [B, C, H, W] current image (t)
        input_ids: torch.Tensor,        # [B, seq_len]
        attention_mask: torch.Tensor,   # [B, seq_len]
        proprio: torch.Tensor,          # [B, proprio_dim]
        ee_pose: torch.Tensor,          # [B, ee_dim]
        actions: Optional[torch.Tensor] = None,           # [B, action_dim]
        future_images: Optional[torch.Tensor] = None,     # [B, C, H, W] GT future image
        future_ee_pose: Optional[torch.Tensor] = None,    # [B, ee_dim] GT future ee_pose
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with future image + gripper pose prediction.

        Returns:
            Dictionary with:
            - 'predicted_future_latent': Predicted future visual latent [B, 768]
            - 'decoded_future': Decoded future image [B, 3, 64, 64]
            - 'predicted_future_ee_pose': Predicted future EE pose [B, 7]
            - 'rendered_future': Rendered future state [B, 512]
            - 'condition': Final condition vector for action prediction
            - Losses (if GT provided)
        """
        # 1. Encode current observation
        vision_embeds = self.encode_vision(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        proprio_embeds = self.encode_proprio(proprio)
        ee_embeds = self.encode_ee(ee_pose)

        # 2. Predict future (visual latent + EE pose)
        future_latent, decoded_future, predicted_future_ee_pose = self.predict_future(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds
        )

        # 3. Render future state (V3.0: concatenate visual + pose)
        rendered_future = self.render_future_state(future_latent, predicted_future_ee_pose)

        output = {
            'predicted_future_latent': future_latent,
            'decoded_future': decoded_future,
            'predicted_future_ee_pose': predicted_future_ee_pose,
            'rendered_future': rendered_future,
        }

        # 4. Compute future prediction losses if GT provided
        if future_images is not None:
            # Encode GT future image to get GT latent
            gt_future_latent = self.vision_encoder(pixel_values=future_images).pooler_output  # [B, 768]

            # Loss in latent space (from V1)
            loss_future = F.mse_loss(future_latent, gt_future_latent)
            output['loss_future'] = loss_future

            # Loss in pixel space for decoder (from V1)
            # Resize GT to match decoder output
            gt_resized = F.interpolate(
                future_images,
                size=(decoded_future.shape[2], decoded_future.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            # Denormalize GT from CLIP stats to [0, 1]
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(gt_resized.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(gt_resized.device)
            gt_resized = gt_resized * std + mean
            gt_resized = torch.clamp(gt_resized, 0, 1)

            loss_decoder = F.mse_loss(decoded_future, gt_resized)
            output['loss_decoder'] = loss_decoder

        if future_ee_pose is not None:
            # Loss for EE pose prediction (from V2)
            loss_ee_pose = F.mse_loss(predicted_future_ee_pose, future_ee_pose)
            output['loss_ee_pose'] = loss_ee_pose
            output['target_future_ee_pose'] = future_ee_pose

        # 5. Fuse current observation + rendered future for action prediction
        condition = self.fusion(torch.cat([
            vision_embeds,
            text_embeds,
            proprio_embeds,
            ee_embeds,
            rendered_future,  # ← KEY: Action conditioned on rendered future!
        ], dim=-1))

        output['condition'] = condition

        # 6. Compute action loss if actions provided
        if actions is not None:
            loss_action = self.flow.compute_loss(actions, condition)
            output['loss_action'] = loss_action

        # 7. Total loss with configurable weights
        if 'loss_future' in output and 'loss_decoder' in output and 'loss_ee_pose' in output and 'loss_action' in output:
            # Future prediction gets weight 0.5 (split across 3 components)
            # Action prediction gets weight 1.0
            future_weight_per_component = self.future_loss_weight / 3.0

            output['loss'] = (
                future_weight_per_component * output['loss_future'] +
                future_weight_per_component * output['loss_decoder'] +
                future_weight_per_component * output['loss_ee_pose'] +
                self.action_loss_weight * output['loss_action']
            )

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
        # Encode current observation
        vision_embeds = self.encode_vision(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        proprio_embeds = self.encode_proprio(proprio)
        ee_embeds = self.encode_ee(ee_pose)

        # Predict future
        future_latent, _, predicted_future_ee_pose = self.predict_future(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds
        )

        # Render future state
        rendered_future = self.render_future_state(future_latent, predicted_future_ee_pose)

        # Fuse
        condition = self.fusion(torch.cat([
            vision_embeds,
            text_embeds,
            proprio_embeds,
            ee_embeds,
            rendered_future,
        ], dim=-1))

        # Sample actions using flow
        actions = self.flow.sample(condition, num_steps=num_flow_steps)

        return actions
