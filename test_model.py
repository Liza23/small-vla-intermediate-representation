"""
Quick test script to verify model architecture works
"""

import torch
from models import VLAModel


def test_model():
    """Test model forward pass"""
    print("Testing VLA Model Architecture...")
    print("="*60)

    # Create model
    model = VLAModel(
        action_dim=7,
        proprio_dim=7,
        ee_dim=7,
        hidden_dim=512,
        flow_hidden_dim=512,
        flow_num_layers=4,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print()

    # Create dummy inputs
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 49408, (batch_size, 77))
    attention_mask = torch.ones(batch_size, 77)
    proprio = torch.randn(batch_size, 7)
    ee_pose = torch.randn(batch_size, 7)
    actions = torch.randn(batch_size, 7)

    print("Input shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Attention mask: {attention_mask.shape}")
    print(f"  Proprio: {proprio.shape}")
    print(f"  EE pose: {ee_pose.shape}")
    print(f"  Actions: {actions.shape}")
    print()

    # Test forward pass (training)
    print("Testing forward pass (training mode)...")
    model.train()
    outputs = model(
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        proprio=proprio,
        ee_pose=ee_pose,
        actions=actions,
    )

    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Condition shape: {outputs['condition'].shape}")
    print()

    # Test inference
    print("Testing inference (sampling actions)...")
    model.eval()
    with torch.no_grad():
        predicted_actions = model.predict_action(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            proprio=proprio,
            ee_pose=ee_pose,
            num_flow_steps=10,  # Use fewer steps for quick test
        )

    print(f"  Predicted actions shape: {predicted_actions.shape}")
    print(f"  Action range: [{predicted_actions.min():.2f}, {predicted_actions.max():.2f}]")
    print()

    # Test backward pass
    print("Testing backward pass...")
    model.train()
    outputs = model(
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        proprio=proprio,
        ee_pose=ee_pose,
        actions=actions,
    )
    loss = outputs['loss']
    loss.backward()

    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    print(f"  Parameters with gradients: {has_grad}")
    print()

    print("="*60)
    print("✓ All tests passed!")
    print("Model is ready for training.")
    print("="*60)


if __name__ == '__main__':
    test_model()
