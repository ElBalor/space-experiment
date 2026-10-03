import sys
import os

# SISTER'S SHIELD: Environment Sanitizer
sys.path = [str(p) for p in sys.path if isinstance(p, str) and p]
os.environ["TORCH_OPTIMIZER_COMPILED"] = "0"
os.environ["TRITON_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# API-Compatible Patch
try:
    import importlib.metadata
    class MockEntryPoints(list):
        def select(self, **kwargs): return []
    importlib.metadata.entry_points = lambda **kwargs: MockEntryPoints()
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from model import SpaceTransformer

class Config:
    block_size = 32768
    n_embd = 768
    n_head = 12
    n_layer = 12
    vocab_size = 151936
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    boredom_threshold = 0.8
    bias_config = [['random']*12]*12
    enable_head_lifecycle = False

try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([Config])
except: pass

class PhysicsBridge(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.proj_in = nn.Linear(3, n_embd)
        self.proj_out = nn.Linear(n_embd, 3)
        # SISTER'S SHIELD: Normalize the Bridge to prevent 32^3 grid shock
        self.norm = nn.LayerNorm(n_embd)
    def forward_in(self, x): 
        return self.norm(self.proj_in(x))
    def forward_out(self, x): 
        return self.proj_out(x)

def compute_divergence(velocity_field, N=32, L=2*np.pi):
    B, T, C = velocity_field.shape
    field = velocity_field.view(B, N, N, N, 3)
    dx = L / N
    du_dx = (torch.roll(field[..., 0], shifts=-1, dims=1) - torch.roll(field[..., 0], shifts=1, dims=1)) / (2*dx)
    dv_dy = (torch.roll(field[..., 1], shifts=-1, dims=2) - torch.roll(field[..., 1], shifts=1, dims=2)) / (2*dx)
    dw_dz = (torch.roll(field[..., 2], shifts=-1, dims=3) - torch.roll(field[..., 2], shifts=1, dims=3)) / (2*dx)
    div = du_dx + dv_dy + dw_dz
    return torch.mean(torch.tanh(div)**2)

def calibrate_ultimate():
    print("=" * 60)
    print("🌪️ JINX: ULTIMATE PHYSICS CALIBRATION (SURGICAL PRECISION)")
    print("=" * 60)
    
    config = Config()
    device = config.device
    
    # RUN IN PURE FLOAT32 FOR STABILITY (No AMP)
    model = SpaceTransformer(config).to(device).float()
    bridge = PhysicsBridge(config.n_embd).to(device).float()
    
    surrogate_path = "jinx_dns_surrogate.pt"
    print(f"[ACTION] Loading 16³ Surrogate...")
    checkpoint = torch.load(surrogate_path, map_location=device, weights_only=False)
    m_state = model.state_dict()
    chk_state = checkpoint['model_state_dict']
    for name, param in chk_state.items():
        if name in m_state:
            if param.shape == m_state[name].shape: m_state[name].copy_(param)
            else:
                if param.dim() == 2: # Pos Emb
                    m_state[name].zero_()
                    m_state[name][:param.shape[0], :param.shape[1]].copy_(param)
    model.load_state_dict(m_state)
    bridge.load_state_dict(checkpoint['bridge_state_dict'])

    # RESET FAST WEIGHTS
    with torch.no_grad():
        for b in model.blocks:
            b.attn.v_fast.zero_(); b.attn.q_fast.zero_()

    X_data = torch.from_numpy(np.load("tgv_inputs_ultimate.npy")).float().to(device)
    Y_data = torch.from_numpy(np.load("tgv_targets_ultimate.npy")).float().to(device)
    
    optimizer = torch.optim.AdamW(list(bridge.parameters()) + list(model.parameters()), lr=5e-6)
    model.train()
    
    print(f"\n[ACTION] Evolution starting in FULL FLOAT32 (No AMP)...")
    for epoch in range(5):
        epoch_mse = 0
        t0 = time.time()
        for i in range(len(X_data)):
            optimizer.zero_grad()
            
            x_lat = bridge.forward_in(X_data[i].unsqueeze(0))
            x_out, drive = model.forward_physics(x_lat)
            pred_y = bridge.forward_out(x_out)
            
            mse = F.mse_loss(pred_y, Y_data[i].unsqueeze(0))
            div = compute_divergence(pred_y, N=32)
            loss = mse + 0.05 * div
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.05)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
                optimizer.step()
                epoch_mse += mse.item()
            else:
                print(f"⚠️  Skip NaN Frame {i+1}")

            if (i + 1) % 10 == 0 or (i + 1) == 1:
                print(f"   [F {i+1:3}/500] Loss: {loss.item():.2e} | Will: {drive['will_power'].mean():.2f}")
            
        print(f"\n✅ EPOCH {epoch+1}/5 | Avg MSE: {epoch_mse/len(X_data):.6f}")
        torch.save(model.state_dict(), "yaka_invariant_32.pt")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    calibrate_ultimate()
