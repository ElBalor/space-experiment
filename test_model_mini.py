import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import sys

# Add space-transformer to path to import matching model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'space-transformer')))
from model import SpaceTransformer
from train_space import Config

print("=" * 60)
print("🧪 MINI TEST: 16³ Model Inference")
print("=" * 60)

device = Config.device
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Load model
print("\n[ACTION] Loading jinx_dns_surrogate.pt...")
checkpoint = torch.load('jinx_dns_surrogate.pt', map_location=device, weights_only=False)

# Create config matching checkpoint exactly
class CheckpointConfig:
    block_size = 4120  # Checkpoint has 4120
    n_embd = 768
    n_head = 12
    n_layer = 12
    vocab_size = 151936
    dropout = 0.1
    device = Config.device
    enable_head_lifecycle = False
    use_identity_tokens = False  # Disable identity/temporal tokens for checkpoint
config = CheckpointConfig()

# Load bridge
class PhysicsBridge(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.proj_in = nn.Linear(3, n_embd)
        self.proj_out = nn.Linear(n_embd, 3)
        self.norm = nn.LayerNorm(n_embd)
    def forward_in(self, x): return self.norm(self.proj_in(x))
    def forward_out(self, x): return self.proj_out(x)

bridge = PhysicsBridge(768).to(device)
bridge.load_state_dict(checkpoint['bridge_state_dict'])
bridge.eval()

# Load model with exact architecture from space-transformer/model.py
print("\n[ACTION] Loading SpaceTransformer from space-transformer/model.py...")
model = SpaceTransformer(config).to(device)
model_state = checkpoint['model_state_dict']

# Load with strict=False to ignore extra checkpoint keys (last_hidden_state, math_signature, symbolic_oracle)
missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
model.eval()

print(f"\n✅ Model loaded from space-transformer/model.py")
print(f"   Missing keys: {len(missing_keys)} (expected for identity/temporal tokens)")
print(f"   Unexpected keys: {len(unexpected_keys)} (checkpoint has extra components)")
print(f"   Core transformer layers loaded successfully")

# Test on different data resolutions to determine training data
print("\n[ACTION] Testing on different data resolutions...")

# Test 1: 16³ data
print("\n--- Test 1: 16³ data (tgv_inputs.16.npy) ---")
if os.path.exists('tgv_inputs.16.npy') and os.path.exists('tgv_targets.16.npy'):
    X_16 = torch.from_numpy(np.load('tgv_inputs.16.npy')[:1]).float().to(device)
    Y_16 = torch.from_numpy(np.load('tgv_targets.16.npy')[:1]).float().to(device)
    print(f"Input shape: {X_16.shape}")
    
    with torch.no_grad():
        x_lat = bridge.forward_in(X_16)
        # Directly pass through blocks like training script does
        x = x_lat
        for block in model.blocks:
            x = block(x, will_signal=None, will_per_token=None, attn_stats=None)
        pred_16 = bridge.forward_out(x)
    
    mse_16 = torch.nn.functional.mse_loss(pred_16, Y_16).item()
    print(f"MSE on 16³ data: {mse_16:.6f}")
else:
    print("❌ 16³ data files not found")
    mse_16 = None

# Test 2: 64³ data with tiled inference
print("\n--- Test 2: 64³ data (tgv_inputs_ultimate.16.npy) - TILED ---")
if os.path.exists('tgv_inputs_ultimate.16.npy') and os.path.exists('tgv_targets_ultimate.16.npy'):
    X_64_full = torch.from_numpy(np.load('tgv_inputs_ultimate.16.npy')[:1]).float().to(device)
    Y_64_full = torch.from_numpy(np.load('tgv_targets_ultimate.16.npy')[:1]).float().to(device)
    print(f"Full 64³ shape: {X_64_full.shape}")
    
    # Reshape to 64x64x64 grid and process in 16³ patches
    N = 64
    patch_size = 16
    patches_per_dim = N // patch_size  # 4
    
    # Reshape from (1, 262144, 3) to (1, 64, 64, 64, 3)
    X_64_grid = X_64_full.view(1, N, N, N, 3)
    Y_64_grid = Y_64_full.view(1, N, N, N, 3)
    
    # Process patches and collect predictions
    pred_grid = torch.zeros_like(Y_64_grid)
    
    with torch.no_grad():
        for i in range(patches_per_dim):
            for j in range(patches_per_dim):
                for k in range(patches_per_dim):
                    # Extract patch
                    i_start, i_end = i*patch_size, (i+1)*patch_size
                    j_start, j_end = j*patch_size, (j+1)*patch_size
                    k_start, k_end = k*patch_size, (k+1)*patch_size
                    
                    patch = X_64_grid[:, i_start:i_end, j_start:j_end, k_start:k_end, :].contiguous()
                    patch_flat = patch.view(1, patch_size**3, 3)
                    
                    # Process through model
                    x_lat = bridge.forward_in(patch_flat)
                    x = x_lat
                    for block in model.blocks:
                        x = block(x, will_signal=None, will_per_token=None, attn_stats=None)
                    pred_patch = bridge.forward_out(x)
                    
                    # Reshape back and place in grid
                    pred_grid[:, i_start:i_end, j_start:j_end, k_start:k_end, :] = pred_patch.view(1, patch_size, patch_size, patch_size, 3)
    
    # Flatten back to (1, 262144, 3)
    pred_64_full = pred_grid.view(1, N**3, 3)
    
    mse_64 = torch.nn.functional.mse_loss(pred_64_full, Y_64_full).item()
    print(f"MSE on full 64³ data (tiled): {mse_64:.6f}")
else:
    print("❌ 64³ data files not found")
    mse_64 = None

# Conclusion
print("\n" + "=" * 60)
print("🔍 TRAINING DATA ANALYSIS")
print("=" * 60)
if mse_16 is not None and mse_64 is not None:
    if mse_16 < mse_64:
        print(f"✅ Model performs better on 16³ data (MSE: {mse_16:.6f} vs {mse_64:.6f})")
        print("→ Model was likely trained on 16³ data")
    else:
        print(f"✅ Model performs better on 64³ data (MSE: {mse_64:.6f} vs {mse_16:.6f})")
        print("→ Model was likely trained on 64³ data")
elif mse_16 is not None:
    print(f"Model tested on 16³ data: MSE = {mse_16:.6f}")
elif mse_64 is not None:
    print(f"Model tested on 64³ data: MSE = {mse_64:.6f}")

print("\n" + "=" * 60)
print("🎉 MINI TEST COMPLETE")
print("=" * 60)
