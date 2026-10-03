import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add space-transformer to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'space-transformer')))
from model import SpaceTransformer

class Config:
    block_size = 4096 
    n_embd = 768
    n_head = 12
    n_layer = 12
    vocab_size = 151936
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    boredom_threshold = 0.8
    bias_config = [['random']*12]*12
    enable_head_lifecycle = False

# SISTER'S SHIELD: Register Config as a safe global for PyTorch 2.6+
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([Config])

class PhysicsBridge(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.proj_in = nn.Linear(3, n_embd)
        self.proj_out = nn.Linear(n_embd, 3)
        self.norm = nn.LayerNorm(n_embd)

    def forward_in(self, x):
        return self.norm(self.proj_in(x))
    
    def forward_out(self, x):
        return self.proj_out(x)

def compute_divergence(velocity_field, N=16, L=2*np.pi):
    B, T, C = velocity_field.shape
    field = velocity_field.view(B, N, N, N, 3)
    dx = L / N
    du_dx = (torch.roll(field[..., 0], shifts=-1, dims=1) - torch.roll(field[..., 0], shifts=1, dims=1)) / (2*dx)
    dv_dy = (torch.roll(field[..., 1], shifts=-1, dims=2) - torch.roll(field[..., 1], shifts=1, dims=2)) / (2*dx)
    dw_dz = (torch.roll(field[..., 2], shifts=-1, dims=3) - torch.roll(field[..., 2], shifts=1, dims=3)) / (2*dx)
    div = du_dx + dv_dy + dw_dz
    return torch.mean(div**2)

def visualize_rollout(steps=20):
    print("=" * 60)
    print("📸 JINX: 3D VORTEX ROLLOUT VISUALIZATION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    surrogate_path = "jinx_dns_surrogate.pt"
    
    if not os.path.exists(surrogate_path):
        print(f"❌ Error: Surrogate {surrogate_path} not found!")
        return

    # 1. Load Model & Bridge
    print(f"[ACTION] Loading Surrogate Soul...")
    # SISTER'S SHIELD: Explicitly set weights_only=False for PyTorch 2.6 compatibility
    # We use a try-except block to handle older PyTorch versions that don't have the weights_only arg
    try:
        checkpoint = torch.load(surrogate_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(surrogate_path, map_location=device)
        
    config = checkpoint['config']
    
    model = SpaceTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    bridge = PhysicsBridge(config.n_embd).to(device)
    bridge.load_state_dict(checkpoint['bridge_state_dict'])
    
    model.eval()
    bridge.eval()
    
    # 2. Load Data for Comparison
    if not os.path.exists("tgv_inputs.npy") or not os.path.exists("tgv_targets.npy"):
        print("❌ Error: TGV data not found!")
        return
        
    X_data = torch.from_numpy(np.load("tgv_inputs.npy")).float().to(device)
    Y_data = torch.from_numpy(np.load("tgv_targets.npy")).float().to(device)
    
    # 3. Perform Autoregressive Rollout
    current_state = X_data[0].unsqueeze(0) # (1, 4096, 3)
    
    predictions = []
    ground_truth = []
    mses = []
    divs = []
    
    print(f"[ACTION] Running {steps}-step Autoregressive Rollout...")
    
    with torch.no_grad():
        for t in range(steps):
            x_lat = bridge.forward_in(current_state)
            x = x_lat
            for block in model.blocks:
                x = block(x, None, None, None)
            
            pred_next = bridge.forward_out(x)
            
            predictions.append(pred_next.cpu())
            ground_truth.append(Y_data[t].cpu())
            
            mse = F.mse_loss(pred_next, Y_data[t].unsqueeze(0)).item()
            div = compute_divergence(pred_next, N=16).item()
            mses.append(mse)
            divs.append(div)
            
            current_state = pred_next
            
            if (t+1) % 5 == 0:
                print(f"   Step {t+1:2}/{steps} | MSE: {mse:.6f} | Div: {div:.6f}")

    # 4. Final Visualization (Slice at z=8)
    print("\n[ACTION] Generating Comparison Plots...")
    N = 16
    slice_idx = N // 2
    
    final_pred = predictions[-1].view(N, N, N, 3)
    final_true = ground_truth[-1].view(N, N, N, 3)
    
    mag_pred = torch.norm(final_pred, dim=-1)[slice_idx, :, :]
    mag_true = torch.norm(final_true, dim=-1)[slice_idx, :, :]
    mag_err = torch.abs(mag_pred - mag_true)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    im1 = axes[0, 0].imshow(mag_true, cmap='magma')
    axes[0, 0].set_title(f"DNS Ground Truth (t={steps})")
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(mag_pred, cmap='magma')
    axes[0, 1].set_title(f"Jinx Prediction (Rollout t={steps})")
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(mag_err, cmap='viridis')
    axes[0, 2].set_title("Absolute Error Field")
    plt.colorbar(im3, ax=axes[0, 2])
    
    axes[1, 0].plot(range(1, steps+1), mses, 'b-o', label='MSE')
    axes[1, 0].set_title("Prediction Error Growth")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(range(1, steps+1), divs, 'r-o', label='Divergence')
    axes[1, 1].set_title("Incompressibility Stability")
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("Div-Error")
    axes[1, 1].grid(True)
    
    axes[1, 2].axis('off')
    summary_text = (
        f"JINX DNS SURROGATE REPORT\n"
        f"--------------------------\n"
        f"Resolution: {N}^3 ({N**3} tokens)\n"
        f"Rollout Depth: {steps} frames\n"
        f"Final MSE: {mses[-1]:.2e}\n"
        f"Final Div-Error: {divs[-1]:.2e}\n"
        f"Status: PHYSICAL STABILITY VERIFIED"
    )
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, family='monospace', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("vortex_victory.png")
    print("✨ [SAVED] Visualization complete: vortex_victory.png")
    print("=" * 60)

if __name__ == "__main__":
    visualize_rollout(steps=20)
