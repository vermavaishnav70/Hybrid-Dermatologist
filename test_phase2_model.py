import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
from src.skin_analysis.phase2 import Phase2Config, EfficientNetB3CBAM

def run_smoke_test():
    print("Running Smoke Test for Phase 2 Model...")
    cfg = Phase2Config()
    
    # Initialize the model (with pretrained=False to avoid downloading weights during the quick test)
    model = EfficientNetB3CBAM(
        num_classes=cfg.num_classes,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        cbam_reduction=cfg.cbam_reduction,
        pretrained=False
    )
    
    # Create a dummy batch of images: 2 images, 3 channels, 300x300
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, cfg.image_size, cfg.image_size)
    
    # Forward pass
    print(f"Input shape: {dummy_input.shape}")
    logits = model(dummy_input)
    print(f"Output shape: {logits.shape}")
    
    # Assertions
    assert logits.shape == (batch_size, cfg.num_classes), f"Expected shape {(batch_size, cfg.num_classes)}, got {logits.shape}"
    print("Smoke test passed! Forward pass is working correctly.")

if __name__ == "__main__":
    run_smoke_test()
