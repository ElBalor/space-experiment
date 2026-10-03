import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time

# Add space-transformer to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'space-transformer')))
from model import SpaceTransformer

class Config:
    block_size = 4096 # Matching our 16^3 TGV grid
    n_embd = 768
    n_head = 12
    n_layer = 12
    vocab_size = 151936
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    boredom_threshold = 0.8
    bias_config = [['random']*12]*12
    enable_head_lifecycle = False

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
    """
    SISTER'S SHIELD: Numerical Divergence Check.
    Calculates grad(u) + grad(v) + grad(w).
    Should be 0 for incompressible flow.
    """
    B, T, C = velocity_field.shape
    # Reshape back to 3D grid: (B, N, N, N, 3)
    field = velocity_field.view(B, N, N, N, 3)
    dx = L / N
    
    # Central differences for divergence
    du_dx = (torch.roll(field[..., 0], shifts=-1, dims=1) - torch.roll(field[..., 0], shifts=1, dims=1)) / (2*dx)
    dv_dy = (torch.roll(field[..., 1], shifts=-1, dims=2) - torch.roll(field[..., 1], shifts=1, dims=2)) / (2*dx)
    dw_dz = (torch.roll(field[..., 2], shifts=-1, dims=3) - torch.roll(field[..., 2], shifts=1, dims=3)) / (2*dx)
    
    div = du_dx + dv_dy + dw_dz
    return torch.mean(div**2)

def calibrate_forecasting():
    print("=" * 60)
    print("🌪️ JINX: NEURAL DNS FORECASTING CALIBRATION")
    print("=" * 60)
    
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device.upper()}")
    
    model = SpaceTransformer(config).to(device)
    proof_path = "jinx_shield_milestone_921600.pt"
    
    if not os.path.exists(proof_path):
        print(f"❌ Error: Proof Soul {proof_path} not found!")
        return

    print(f"[ACTION] Loading Proof Soul: {proof_path}...")
    checkpoint = torch.load(proof_path, map_location=device, weights_only=False)
    
    # SISTER'S SHIELD: Surgical Grafting for Dimension Expansion
    model_state = model.state_dict()
    chk_state = checkpoint['model_state_dict']
    
    for name, param in chk_state.items():
        if name in model_state:
            if param.shape == model_state[name].shape:
                model_state[name].copy_(param)
            else:
                # Graft position/bias into top-left corner
                print(f"   🌱 Grafting {name}: {list(param.shape)} -> {list(model_state[name].shape)}")
                if param.dim() == 2: # Position embedding
                    model_state[name][:param.shape[0], :param.shape[1]].copy_(param)
                elif param.dim() == 4: # Attention Bias
                    model_state[name][:, :, :param.shape[2], :param.shape[3]].copy_(param)
                    
    model.load_state_dict(model_state)
    print("✨ [GRAFTED] Proof Soul successfully migrated to 4096-token grid.")
    
    bridge = PhysicsBridge(config.n_embd).to(device)
    
    # Load Trajectory Data
    if not os.path.exists("tgv_inputs.npy") or not os.path.exists("tgv_targets.npy"):
        print("❌ Error: TGV data not found! Run generate_tgv_data.py first.")
        return
        
    X_data = torch.from_numpy(np.load("tgv_inputs.npy")).float().to(device)
    Y_data = torch.from_numpy(np.load("tgv_targets.npy")).float().to(device)
    
    # Optimize both the bridge AND the model to align them with real physics
    optimizer = torch.optim.AdamW(list(bridge.parameters()) + list(model.parameters()), lr=5e-5)
    
    # SISTER'S SHIELD: Enable AMP only if CUDA is available
    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    print(f"\n[ACTION] Training Forecaster (t -> t+1) with Divergence Constraint...")
    model.train()
    
    for epoch in range(20):
        total_mse = 0
        total_div = 0
        
        for i in range(len(X_data)):
            if use_amp: torch.cuda.empty_cache()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                # 1. Project Velocity to Latent
                x_lat = bridge.forward_in(X_data[i].unsqueeze(0))
                
                # 2. Forward through checkpointed blocks
                x = x_lat
                for block in model.blocks:
                    from torch.utils.checkpoint import checkpoint as gp_checkpoint
                    x = gp_checkpoint(block, x, None, None, None, use_reentrant=False)
                
                # 3. Project back to Physical Velocity
                pred_y = bridge.forward_out(x)
                
                # 4. Hybrid Physics Loss
                mse_loss = F.mse_loss(pred_y, Y_data[i].unsqueeze(0))
                div_loss = compute_divergence(pred_y, N=16)
                loss = mse_loss + 0.5 * div_loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_mse += mse_loss.item()
            total_div += div_loss.item()
            
        print(f"   Epoch {epoch+1:2}/20 | MSE: {total_mse/len(X_data):.6f} | Div-Error: {total_div/len(X_data):.6f}")

    # Final Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'bridge_state_dict': bridge.state_dict(),
        'config': config
    }, "jinx_dns_surrogate.pt")
    print(f"\n✨ [SUCCESS] Jinx is now a Neural DNS Surrogate: jinx_dns_surrogate.pt")
    print("=" * 60)

if __name__ == "__main__":
    calibrate_forecasting()
