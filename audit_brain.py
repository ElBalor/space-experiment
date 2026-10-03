import torch
import torch.nn.functional as F
from model import SpaceTransformer
import tiktoken
import os
import sys

# Import Config from train_space
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_space import Config

def audit_synapses():
    print("🧠 [DEEP SCAN] Starting Synaptic Autopsy...")
    print("="*60)

    # 1. Load the Soul
    ckpt_path = "jinx_final_soul.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint {ckpt_path} not found!")
        return

    config = Config()
    model = SpaceTransformer(config).to(config.device)
    checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # SISTER'S SHIELD: Surgical Loading
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            elif k == 'reservoir.state':
                # Handle 1D → 2D reservoir upgrade or batch dimension mismatch
                if v.dim() == 2 and model_state[k].dim() == 2:
                    # Both 2D, check if shapes match
                    if v.shape == model_state[k].shape:
                        filtered_state[k] = v
                    else:
                        print(f"   Skipping {k} (shape mismatch: {v.shape} vs {model_state[k].shape})")
                elif v.dim() == 2 and model_state[k].dim() == 3:
                    # Old 1D batched (B, d) → new 2D (d, d), take first sample
                    filtered_state[k] = v[0:1, :].unsqueeze(-1).expand(-1, -1, v.shape[-1]) if v.dim() == 3 else v
                    print(f"   Adapted {k}: {v.shape} -> {filtered_state[k].shape}")
                else:
                    filtered_state[k] = v
                    print(f"   Loaded {k}: {v.shape}")
            elif k == 'last_hidden_state' and v is not None and v.shape[0] > 1:
                # Take the first element of the batch
                # Also ensure it's on the correct device
                filtered_state[k] = v[0:1, :, :].to(config.device)
                print(f"   Adapted {k}: {v.shape} -> {filtered_state[k].shape}")
            else:
                print(f"   Skipping {k} (mismatch or not needed)")
        else:
            print(f"   Skipping {k} (not in model)")
        
    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    print("\n[PART 1] THE DENDRITIC PROBE (Logic Gates)")
    print("-" * 40)
    # Check if gates are blocking signal
    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            gate_w = block.mlp.dend2.weight
            gate_b = block.mlp.dend2.bias
            # Simulate mean activation
            gate_act = torch.sigmoid(gate_b).mean().item()
            print(f"   Layer {i:2} Gate Opening: {gate_act*100:.2f}% (Target: >40%)")

    print("\n[PART 2] SIGNAL PROPAGATION (The Manifold Flow)")
    print("-" * 40)
    # Check if thoughts survive the 12 layers
    test_input = torch.randint(0, config.vocab_size, (1, 64)).to(config.device)
    with torch.no_grad():
        x = model.token_embedding(test_input)
        print(f"   Input Signal Std:  {x.std().item():.4f}")
        for i, block in enumerate(model.blocks):
            x = block(x)
            if i in [0, 5, 11]:
                print(f"   Layer {i:2} Signal Std: {x.std().item():.4f}")
        
        final_std = x.std().item()
        status = "FLUID" if 0.1 < final_std < 5.0 else "EXPLODED" if final_std >= 5.0 else "COLLAPSED"
        print(f"   Final Manifold Status: {status}")

    print("\n[PART 3] POSITION COHERENCE (Vision Integrity)")
    print("-" * 40)
    # Neighboring positions should be similar but distinct
    pos_emb = model.position_embedding.weight
    sim = F.cosine_similarity(pos_emb[10:11], pos_emb[11:12]).item()
    print(f"   Neighboring Position Sim: {sim:.4f} (Target: 0.8 - 0.99)")
    
    # Check Identity vs Data overlap
    id_pos = pos_emb[0]
    data_pos = pos_emb[8] # Our current data start
    overlap = F.cosine_similarity(id_pos.unsqueeze(0), data_pos.unsqueeze(0)).item()
    print(f"   Identity vs Data Overlap: {overlap:.4f} (Target: < 0.5)")

    print("\n[PART 4] THE SOUL MIRROR (Identity Consistency)")
    print("-" * 40)
    ego = model.ego_engine.ego_vector
    harmony = F.cosine_similarity(ego, model.last_hidden_state.mean(dim=(0,1)).unsqueeze(0)).item()
    print(f"   Current Soul Alignment: {harmony:.4f} (Target: > 0.8)")

    print("\n" + "="*60)
    print("📊 [AUTOPSY COMPLETE]")
    print("="*60)

if __name__ == "__main__":
    audit_synapses()
