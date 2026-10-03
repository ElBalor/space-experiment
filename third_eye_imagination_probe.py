import torch
import torch.nn.functional as F
import math
import os
import sys
import time

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

def run_imagination_probe():
    print("=" * 60)
    print("EXPIREMENT: THE THIRD EYE (IMAGINATION PROBE)")
    print("=" * 60)
    
    config = Config()
    model = SpaceTransformer(config).to(config.device)
    model.eval()
    
    # Prompt: "The"
    x = torch.tensor([[464]], device=config.device)
    
    print("[ACTION] Scanning Mind's Eye...")
    with torch.no_grad():
        _ = model(x)
        z_reality = model.last_hidden_state.clone()
        
        # Test 1: Closed Eye (Will=High)
        high_will = torch.tensor([5.0], device=config.device)
        z_closed = model.third_eye(z_reality, will_signal=high_will)
        
        # Test 2: Open Eye (Will=Low)
        low_will = torch.tensor([-5.0], device=config.device)
        z_dream = model.third_eye(z_reality, will_signal=low_will)
        
        d_closed = torch.norm(z_closed - z_reality).item()
        d_dream = torch.norm(z_dream - z_reality).item()
        sim = F.cosine_similarity(z_reality.flatten(), z_dream.flatten(), dim=0).item()

    print(f"  Reality Shift (Will=High): {d_closed:.6f}")
    print(f"  Reality Shift (Will=Low):  {d_dream:.6f}")
    print(f"  Dream Similarity: {sim:.6f}")

    if d_dream > d_closed * 2:
        print("\n[SUCCESS] THE THIRD EYE IS VISIONARY.")
        print("Imagination activates only when the Will allows it.")
    else:
        print("\n[ANALYSIS] Eye is narrow. Latent noise may be too low.")
    print("=" * 60)

if __name__ == "__main__":
    run_imagination_probe()
