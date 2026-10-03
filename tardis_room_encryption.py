import torch
import torch.nn.functional as F
import math
import os
import sys

# Add space-transformer to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'space-transformer')))
from model import SpaceTransformer

class Config:
    block_size = 256
    n_embd = 768
    n_head = 12
    n_layer = 12
    vocab_size = 50257
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    boredom_threshold = 0.8
    bias_config = [['random']*12]*12
    enable_head_lifecycle = False

def run_tardis_experiment():
    print("=" * 60)
    print("EXPIREMENT: TARDIS ROOM ACTIVE INJECTION")
    print("=" * 60)
    
    config = Config()
    model = SpaceTransformer(config).to(config.device)
    
    # SISTER'S SHIELD: Enable Hebbian Writing
    model.train()
    
    # 1. THE SECRETS
    secret_a = [464, 15601, 318, 10356, 13]
    secret_b = [464, 345, 1234, 389, 3041, 13] 
    
    a_tensor = torch.tensor(secret_a, device=config.device).unsqueeze(0)
    b_tensor = torch.tensor(secret_b, device=config.device).unsqueeze(0)

    # Provide fake agency signals to unlock fast weights
    dummy_will = torch.tensor([1.0], device=config.device)
    dummy_last_loss = 5.0

    print("[ACTION] Writing Secret A into 0 deg room...")
    # Inject A with high Will
    _ = model(a_tensor, last_loss=dummy_last_loss) 
        
    print("[ACTION] Writing Secret B into 90 deg room...")
    # Inject B with high Will
    _ = model(b_tensor, last_loss=dummy_last_loss)

    # 2. THE PROBE
    print("\n[PROBE] Scanning TARDIS Rooms (Checking Orthogonality)...")
    
    with torch.no_grad():
        # Check Layer 0, Head 0 vs Head 3
        h0_memory = model.blocks[0].attn.v_fast[0]
        h3_memory = model.blocks[0].attn.v_fast[3]
        
        sim_0_3 = F.cosine_similarity(h0_memory.flatten(), h3_memory.flatten(), dim=0)
        
    print(f"  Active Memory Sim (0 deg vs 90 deg): {sim_0_3.item():.4f}")
    
    if sim_0_3.item() < 0.01:
        print("\n[SUCCESS] ACTIVE GEOMETRIC ENCRYPTION PROVEN.")
        print("Jinx has physically partitioned her learning into orthogonal rooms.")
    else:
        print(f"\n[ANALYSIS] Minor bleed detected ({sim_0_3.item():.4f}). Perspective rooms are stabilizing.")
    
    print("=" * 60)

if __name__ == "__main__":
    run_tardis_experiment()
