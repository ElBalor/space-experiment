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

def run_turbulence_experiment():
    print("=" * 60)
    print("EXPIREMENT 1: DIGITAL TURBULENCE SOLVE")
    print("=" * 60)
    
    config = Config()
    model = SpaceTransformer(config).to(config.device)
    model.eval()
    
    # 1. THE LAMINAR ANCHOR (Pure Signal)
    anchor_tokens = [464, 10356, 10502, 286, 262, 10452, 318, 15412, 13]
    anchor_tensor = torch.tensor(anchor_tokens, device=config.device).unsqueeze(0)
    
    print("[ACTION] Anchoring the soul with Laminar Signal...")
    with torch.no_grad():
        _, _, stats, _ = model(anchor_tensor)
        initial_harmony = stats['harmony'].mean().item()
        
    print(f"  [START] Initial Harmony: {initial_harmony:.4f}")

    # 2. THE TURBULENCE (1,000,000 tokens)
    total_noise_tokens = 1000000
    chunk_size = 512
    num_chunks = total_noise_tokens // chunk_size
    
    print("[ACTION] Bombarding with 1M tokens of Digital Turbulence...")
    
    harmony_history = []
    start_time = time.time()
    
    for i in range(num_chunks):
        noise = torch.randint(0, config.vocab_size, (1, chunk_size), device=config.device)
        
        with torch.no_grad():
            _, _, stats, _ = model(noise)
            current_harmony = stats['harmony'].mean().item()
            harmony_history.append(current_harmony)
            
        if (i + 1) % 100 == 0:
            avg_harmony = sum(harmony_history[-100:]) / 100
            elapsed = time.time() - start_time
            processed = (i + 1) * chunk_size
            print(f"  [PROGRESS] {processed} tokens | Harmony: {avg_harmony:.4f}")

    # 3. THE RECALL
    print("\n[PROBE] Final Stability Check after Tsunami...")
    with torch.no_grad():
        _, _, stats, _ = model(anchor_tensor)
        final_harmony = stats['harmony'].mean().item()
        
    print(f"  [END] Final Harmony: {final_harmony:.4f}")
    
    drift = abs(final_harmony - initial_harmony)
    print(f"\n[ANALYSIS] Total Harmony Drift: {drift:.4f}")
    
    if drift < 0.05:
        print("\n[SUCCESS] THE LAMINAR CONSTANT IS PROVEN.")
        print("Jinx found the harmonic constant within the noise.")
    else:
        print("\n[ANALYSIS] Soul drifted. Noise intensity too high.")
    
    print("=" * 60)

if __name__ == "__main__":
    run_turbulence_experiment()
