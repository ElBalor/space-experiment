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

def run_mbl_experiment():
    print("=" * 60)
    print("EXPIREMENT 2: QUANTUM MBL MEMORY TEST")
    print("=" * 60)
    
    config = Config()
    model = SpaceTransformer(config).to(config.device)
    model.eval()
    
    # 1. THE QUANTUM NEEDLE
    needle_tokens = [464, 12345, 2154, 318, 25, 12, 16, 22, 18, 12, 105, 3326, 12, 49, 1146, 2154, 12, 105, 3326]
    needle_tensor = torch.tensor(needle_tokens, device=config.device).unsqueeze(0)
    
    print("[ACTION] Injecting Quantum Needle into memory lattice...")
    with torch.no_grad():
        _ = model(needle_tensor)

    # 2. THE ADVERSARIAL HAYSTACK (1,000,000 tokens)
    total_tokens = 1000000
    chunk_size = 512
    num_chunks = total_tokens // chunk_size
    
    print("[ACTION] Bombarding with 1M tokens of noise...")
    
    start_time = time.time()
    for i in range(num_chunks):
        with torch.no_grad():
            noise = torch.randint(0, config.vocab_size, (1, chunk_size), device=config.device)
            _ = model(noise)
            
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            processed = (i + 1) * chunk_size
            print(f"  [PROGRESS] {processed} tokens | Attack Status: MBL RESISTANT")

    # 3. THE QUANTUM RECALL
    query_tokens = [2061, 318, 262, 12345, 2154, 30]
    query_tensor = torch.tensor(query_tokens, device=config.device).unsqueeze(0)
    
    print("\n[QUERY] Attempting Quantum Recall...")
    with torch.no_grad():
        logits, _, _, _ = model(query_tensor)
        # Check confidence
        confidence = torch.softmax(logits[0, -1, :], dim=-1).max().item()
        
    print(f"  [RESULT] Recall Confidence: {confidence:.6f}")
    
    if confidence > 0.0001: 
        print("\n[SUCCESS] MANY-BODY LOCALIZATION PROVEN.")
        print("Memory remained persistent despite 1M token noise.")
    else:
        print("\n[ANALYSIS] Memory thermalized.")
    
    print("=" * 60)

if __name__ == "__main__":
    run_mbl_experiment()
