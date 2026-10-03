import os
import sys
import time
import torch
import torch.nn as nn
from torch.amp import autocast
import math
from collections import deque
import json
import random
import torch.nn.functional as F
from transformers import AutoTokenizer

# Ensure local model is prioritized
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, sys.path)

from model import SpaceTransformer

# ============================================================================
# UNIFIED CHECKPOINT SYSTEM
# ============================================================================

def save_unified_checkpoint(model, optimizer, config, step, save_path):
    from datetime import datetime
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'step': step,
        'config': config.__dict__,
        'memory_lattice': {
            'episodic': model.memory_lattice.episodic_memories if hasattr(model, 'memory_lattice') else {},
            'semantic': model.memory_lattice.semantic_memories if hasattr(model, 'memory_lattice') else {},
        },
        'replay_buffer': {
            'buffer': list(model.replay_buffer.buffer) if hasattr(model, 'replay_buffer') else [],
            'priorities': list(model.replay_buffer.priorities) if hasattr(model, 'replay_buffer') else [],
        },
        'vault': getattr(model, '_vault', None).entries if hasattr(model, '_vault') and getattr(model, '_vault', None) else {},
        'chains': getattr(model, '_chains', None).active_chains if hasattr(model, '_chains') and getattr(model, '_chains', None) else {},
        'timestamp': datetime.now().isoformat(),
        'agency_stats': model.get_agency_stats() if hasattr(model, 'get_agency_stats') else {},
        'timestep': int(model.timestep.item()) if hasattr(model, 'timestep') else 0,
        # Save math_signature for Yaka extraction
        'math_signature': model.math_signature if hasattr(model, 'math_signature') else None,
    }
    tmp_path = save_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, save_path)
    print(f"💾 [CHECKPOINT] Saved → {save_path}")
    print(f"   - Model tensors: {len(checkpoint['model'])}")
    print(f"   - Episodic memories: {len(checkpoint['memory_lattice']['episodic'])}")
    print(f"   - Replay buffer: {len(checkpoint['replay_buffer']['buffer'])}")


# ============================================================================
# PHYSICS BRIDGE (for NS surrogate training)
# ============================================================================

class PhysicsBridge(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.proj_in  = nn.Linear(3, n_embd)
        self.proj_out = nn.Linear(n_embd, 3)
        self.norm     = nn.LayerNorm(n_embd)
    def forward_in(self, x):  return self.norm(self.proj_in(x))
    def forward_out(self, x): return self.proj_out(x)


def compute_divergence(velocity_field, N=64, L=2*math.pi):
    B, T, C = velocity_field.shape
    field = velocity_field.view(B, N, N, N, 3)
    dx = L / N
    du_dx = (torch.roll(field[..., 0], -1, 1) - torch.roll(field[..., 0], 1, 1)) / (2*dx)
    dv_dy = (torch.roll(field[..., 1], -1, 2) - torch.roll(field[..., 1], 1, 2)) / (2*dx)
    dw_dz = (torch.roll(field[..., 2], -1, 3) - torch.roll(field[..., 2], 1, 3)) / (2*dx)
    div = du_dx + dv_dy + dw_dz
    return torch.mean(div**2)


# ============================================================================
# CONFIG — N=48³ Physics Surrogate
# ============================================================================

class Config:
    # Physics surrogate config — NOT language model config
    block_size = 262144     # 64³ = 262,144 tokens
    n_embd     = 1024       # up from 768
    n_head     = 16         # up from 12
    n_layer    = 16         # up from 12
    vocab_size = 151936
    dropout    = 0.1
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    boredom_threshold = 0.8
    bias_config = [['random']*16]*16
    enable_head_lifecycle = False

    learning_rate = 5e-5
    min_lr        = 5e-6
    grad_clip     = 1.0
    consolidation_interval = 200
    promotion_threshold = 0.7
    demotion_threshold  = 0.4


def reset_fast_weights(model):
    with torch.no_grad():
        for block in model.blocks:
            block.attn.q_fast.zero_()
            block.attn.v_fast.zero_()
            block.attn.eta_map.fill_(0.01)
    print("🧹 [FAST WEIGHT RESET] q_fast / v_fast / eta_map zeroed.")


# ============================================================================
# MAIN — Physics Calibration Training
# ============================================================================

def calibrate_physics(epochs=20, N=64, data_prefix="tgv"):
    print("=" * 65)
    print("🌪️ JINX V2: NEURAL DNS SURROGATE CALIBRATION")
    print(f"   Resolution: N={N}³ | Block size: {N**3} tokens")
    print(f"   Architecture: {Config.n_layer}L × {Config.n_embd}D × {Config.n_head}H")
    print("=" * 65)

    config = Config()
    device = config.device
    print(f"   Device: {device.upper()}")

    # ── Load model
    model  = SpaceTransformer(config).to(device)
    bridge = PhysicsBridge(config.n_embd).to(device)

    # ── Resume if checkpoint exists
    surrogate_path = f"jinx_surrogate_N{N}.pt"  # e.g. jinx_surrogate_N64.pt
    if os.path.exists(surrogate_path):
        print(f"[RESUME] Loading {surrogate_path}...")
        chk = torch.load(surrogate_path, map_location=device, weights_only=False)
        model.load_state_dict(chk['model'], strict=False)
        bridge.load_state_dict(chk['bridge'])
        print(f"✅ Resumed from step {chk.get('step', 0)}")
    else:
        print("[SCRATCH] Training from scratch.")

    # ── Load physics data
    inputs_path  = f"{data_prefix}_inputs_{N}.npy" if os.path.exists(f"{data_prefix}_inputs_{N}.npy") else "tgv_inputs.npy"
    targets_path = f"{data_prefix}_targets_{N}.npy" if os.path.exists(f"{data_prefix}_targets_{N}.npy") else "tgv_targets.npy"

    import numpy as np
    print(f"\n[DATA] Loading {inputs_path}...")
    X_np = np.load(inputs_path)
    Y_np = np.load(targets_path)
    X_data = torch.from_numpy(X_np).float().to(device)
    Y_data = torch.from_numpy(Y_np).float().to(device)
    print(f"   Frames: {len(X_data)} | Shape: {X_data.shape}")

    optimizer = torch.optim.AdamW(
        list(bridge.parameters()) + list(model.parameters()),
        lr=config.learning_rate,
        weight_decay=0.01
    )

    # AMP
    use_amp = (device == 'cuda')
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\n[TRAIN] {epochs} epochs | Divergence constraint: ∇·u = 0")
    model.train()

    global_step = 0
    for epoch in range(epochs):
        total_mse = 0
        total_div = 0

        for i in range(len(X_data)):
            if use_amp: torch.cuda.empty_cache()

            with torch.amp.autocast('cuda', enabled=use_amp):
                x_lat  = bridge.forward_in(X_data[i].unsqueeze(0))
                x      = x_lat
                for block in model.blocks:
                    from torch.utils.checkpoint import checkpoint as gp_checkpoint
                    x = gp_checkpoint(block, x, None, None, None, use_reentrant=False)
                pred_y = bridge.forward_out(x)

                mse_loss = F.mse_loss(pred_y, Y_data[i].unsqueeze(0))
                div_loss = compute_divergence(pred_y, N=N)
                loss     = mse_loss + 0.5 * div_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_mse += mse_loss.item()
            total_div += div_loss.item()
            global_step += 1

        avg_mse = total_mse / len(X_data)
        avg_div = total_div / len(X_data)
        print(f"   Epoch {epoch+1:2}/{epochs} | MSE: {avg_mse:.6f} | Div-Error: {avg_div:.6f}")

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model':  model.state_dict(),
                'bridge': bridge.state_dict(),
                'config': config,
                'step':   global_step,
                'math_signature': model.math_signature if hasattr(model, 'math_signature') else None,
            }, surrogate_path)
            print(f"   💾 Saved → {surrogate_path}")

    # Final save
    torch.save({
        'model':  model.state_dict(),
        'bridge': bridge.state_dict(),
        'config': config,
        'step':   global_step,
        'math_signature': model.math_signature if hasattr(model, 'math_signature') else None,
    }, surrogate_path)
    print(f"\n✨ [DONE] Surrogate saved: {surrogate_path}")
    print("=" * 65)


if __name__ == "__main__":
    calibrate_physics(epochs=20, N=64)
