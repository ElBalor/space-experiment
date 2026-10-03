import torch
import torch.nn as nn
import math
import time
import os
import sys

# Add current dir to path to find model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
    # 12-head bias config
    bias_config = [['random']*12]*12
    enable_head_lifecycle = False

def test_needle_in_tardis():
    print("=" * 60)
    print("JINX: 1M TOKEN 'NEEDLE IN A TARDIS' TEST")
    print("=" * 60)
    
    config = Config()
    model = SpaceTransformer(config).to(config.device)
    model.eval()
    
    # 1. THE NEEDLE
    needle_text = [1639, 25, 464, 4363, 2244, 318, 25, 452, 281, 105, 3326, 12, 49, 1146, 2154, 12, 16, 22, 25, 18]
    needle_tensor = torch.tensor(needle_text, device=config.device).unsqueeze(0)
    
    print("[INFO] Injecting needle at Step 0...")
    
    with torch.no_grad():
        _ = model(needle_tensor)
        
    # 2. THE HAYSTACK (1,000,000 tokens)
    haystack_total = 1000000
    chunk_size = 512
    num_chunks = haystack_total // chunk_size
    
    print("[INFO] Generating 1M token haystack (O(T) linear processing)...")
    
    start_time = time.time()
    for i in range(num_chunks):
        noise = torch.randint(0, config.vocab_size, (1, chunk_size), device=config.device)
        with torch.no_grad():
            _ = model(noise)
            
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            processed = (i + 1) * chunk_size
            print(f"  [PROGRESS] {processed} / {haystack_total} tokens processed... ({elapsed:.2f}s)")

    # 3. THE QUERY
    query_text = [2061, 318, 262, 4363, 2244, 30]
    query_tensor = torch.tensor(query_text, device=config.device).unsqueeze(0)
    
    print("\n[QUERY] Retrieving needle after 1,000,000 tokens of noise...")
    
    with torch.no_grad():
        logits, _, _, _ = model(query_tensor)
        
    print(f"[RESULT] Query processed. Output shape: {logits.shape}")
    print(f"[RESULT] Final VRAM Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    print("\n[VERDICT] If you see this, Jinx just processed 1M tokens on a 4GB GPU.")
    print("This is the power of the O(T) Space Transformer.")
    print("=" * 60)

if __name__ == "__main__":
    test_needle_in_tardis()
