"""
VLA v1.1: Vision-Language-Action Model with Future State Prediction
Adds intermediate future visual representation prediction before action prediction
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
    Predicts future visual state latent from current observation.
    Uses current vision + text + state to predict what scene will look like at t+1.
    """
    def __init__(
        self,
        input_dim: int = 512 * 4,  # Concatenated features
        latent_dim: int = 768,     # CLIP ViT-B/16 output dim
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

        # Output layer to latent space
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.predictor = nn.Sequential(*layers)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Predict future latent.

        Args:
            fused_features: [B, input_dim] concatenated features

        Returns:
            future_latent: [B, latent_dim] predicted future visual latent
        """
        return self.predictor(fused_features)


class FutureLatentDecoder(nn.Module):
    """
    Decode CLIP latent back to 64x64 RGB image for visualization.
    Trained with supervised loss against downsampled GT images.
    """
    def __init__(
        self,
        latent_dim: int = 768,
        output_size: int = 64,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_size * output_size * 3),
            nn.Sigmoid()  # [0, 1] range
        )

        self.output_size = output_size

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.

        Args:
            latent: [B, latent_dim]

        Returns:
            images: [B, 3, output_size, output_size]
        """
        pixels = self.decoder(latent)  # [B, size*size*3]
        images = pixels.view(-1, 3, self.output_size, self.output_size)
        return images


class VLAModelV1(nn.Module):
    """
    VLA v1.1: Vision-Language-Action Model with Future Prediction.

    Architecture:
    1. Encode current observation (vision + text + proprio + ee_pose)
    2. Predict future visual latent (t+1)
    3. Fuse current features + predicted future
    4. Predict action with Rectified Flow

    Changes from v0:
    - Added FutureLatentPredictor
    - Added FutureLatentDecoder for visualization
    - Modified fusion to include predicted future
    - Dual loss: future prediction + action prediction
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
        future_hidden_dim: int = 1024,
        future_num_layers: int = 3,
        decoder_output_size: int = 64,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

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

        # ====== NEW IN V1 ======
        # Future latent predictor
        self.future_predictor = FutureLatentPredictor(
            input_dim=hidden_dim * 4,  # vision + text + proprio + ee
            latent_dim=self.vision_encoder.config.hidden_size,  # 768 for ViT-B/16
            hidden_dim=future_hidden_dim,
            num_layers=future_num_layers,
        )

        # Decoder for visualization
        self.future_decoder = FutureLatentDecoder(
            latent_dim=self.vision_encoder.config.hidden_size,
            output_size=decoder_output_size,
        )

        # Modified fusion layer (now includes predicted future)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),  # +1 for future (projected to hidden_dim)
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Project future latent to hidden_dim for fusion
        self.future_proj = nn.Linear(self.vision_encoder.config.hidden_size, hidden_dim)

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

    def encode_vision_raw(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to raw CLIP latent (no projection).
        Used for GT target extraction.
        """
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=images)
            return vision_outputs.pooler_output  # [B, 768]

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

    def predict_future_latent(
        self,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        proprio_embeds: torch.Tensor,
        ee_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict future visual state latent (NEW in v1).

        Args:
            vision_embeds: [B, hidden_dim]
            text_embeds: [B, hidden_dim]
            proprio_embeds: [B, hidden_dim]
            ee_embeds: [B, hidden_dim]

        Returns:
            future_latent: [B, 768] predicted CLIP latent for t+1
        """
        # Concatenate all modalities
        fused = torch.cat([
            vision_embeds,
            text_embeds,
            proprio_embeds,
            ee_embeds,
        ], dim=-1)  # [B, hidden_dim * 4]

        # Predict future latent
        future_latent = self.future_predictor(fused)  # [B, 768]

        return future_latent

    def decode_future(self, future_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode future latent to image for visualization (NEW in v1).

        Args:
            future_latent: [B, 768]

        Returns:
            decoded_image: [B, 3, 64, 64]
        """
        return self.future_decoder(future_latent)

    def fuse_with_future(
        self,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        proprio_embeds: torch.Tensor,
        ee_embeds: torch.Tensor,
        future_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse all modalities including predicted future (MODIFIED from v0).

        Args:
            All embeds: [B, hidden_dim]
            future_latent: [B, 768]

        Returns:
            condition: [B, hidden_dim]
        """
        # Project future to hidden_dim
        future_embeds = self.future_proj(future_latent)  # [B, hidden_dim]

        # Concatenate all modalities + future
        fused = torch.cat([
            vision_embeds,
            text_embeds,
            proprio_embeds,
            ee_embeds,
            future_embeds,
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
        future_images: Optional[torch.Tensor] = None,  # [B, C, H, W] target image (t+1)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with future prediction.

        Returns:
            Dictionary with:
            - 'predicted_future_latent': Predicted future latent [B, 768]
            - 'target_future_latent': GT future latent [B, 768] (if future_images provided)
            - 'decoded_future': Decoded future image [B, 3, 64, 64]
            - 'condition': Fused condition vector
            - 'loss_future': Future prediction loss (if future_images provided)
            - 'loss_action': Action prediction loss (if actions provided)
            - 'loss': Total loss (if both provided)
        """
        # Encode all modalities
        vision_embeds = self.encode_vision(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        proprio_embeds = self.encode_proprio(proprio)
        ee_embeds = self.encode_ee(ee_pose)

        # Predict future latent
        predicted_future_latent = self.predict_future_latent(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds
        )

        output = {
            'predicted_future_latent': predicted_future_latent,
        }

        # Decode for visualization
        decoded_future = self.decode_future(predicted_future_latent)
        output['decoded_future'] = decoded_future

        # Compute future prediction loss if GT provided
        if future_images is not None:
            # Encode GT future image to latent
            target_future_latent = self.encode_vision_raw(future_images)
            output['target_future_latent'] = target_future_latent

            # MSE loss in latent space
            loss_future = F.mse_loss(predicted_future_latent, target_future_latent)
            output['loss_future'] = loss_future

            # Optional: Decoder reconstruction loss
            future_images_small = F.interpolate(future_images, size=64, mode='bilinear')
            loss_decoder = F.mse_loss(decoded_future, future_images_small)
            output['loss_decoder'] = loss_decoder

        # Fuse with predicted future for action prediction
        condition = self.fuse_with_future(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds, predicted_future_latent
        )
        output['condition'] = condition

        # Compute action loss if actions provided
        if actions is not None:
            loss_action = self.flow.compute_loss(actions, condition)
            output['loss_action'] = loss_action

        # Total loss
        if 'loss_future' in output and 'loss_action' in output:
            # Weighted combination
            output['loss'] = output['loss_future'] + output['loss_action'] + 0.1 * output['loss_decoder']

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

        # Predict future
        predicted_future_latent = self.predict_future_latent(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds
        )

        # Fuse with future
        condition = self.fuse_with_future(
            vision_embeds, text_embeds, proprio_embeds, ee_embeds, predicted_future_latent
        )

        # Sample actions using flow
        actions = self.flow.sample(condition, num_steps=num_flow_steps)

        return actions
