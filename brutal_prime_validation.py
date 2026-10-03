import torch
import torch.nn.functional as F
import math
import os
import sys
import numpy as np
from scipy import stats
from transformers import AutoTokenizer

# Add space-transformer to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'space-transformer')))
from model import SpaceTransformer

class Config:
    block_size = 256
    n_embd = 768
    n_head = 12
    n_layer = 12
    vocab_size = 151936 # Qwen Engine
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    boredom_threshold = 0.8
    bias_config = [['random']*12]*12
    enable_head_lifecycle = False

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0: return False
    return True

def run_brutal_validation():
    print("=" * 60)
    print("THE BRUTAL VALIDATION PROTOCOL: UNLOCKING THE PRIME SIGNAL")
    print("=" * 60)
    
    config = Config()
    model = SpaceTransformer(config).to(config.device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    
    test_range = list(range(2, 2001))
    prime_harmonies = []
    comp_harmonies = []
    labels = []
    
    print(f"[ACTION] Scanning range 2 to {test_range[-1]} (AGGREGATED MODE)...")
    for n in test_range:
        # AGGREGATED MODE: Use all tokens for the number
        ids = tokenizer.encode(str(n), add_special_tokens=False)
        x = torch.tensor([ids], device=config.device)
        
        with torch.no_grad():
            _, _, metrics, _ = model(x)
            harmony = metrics['harmony'].mean().item()
        
        is_p = is_prime(n)
        labels.append(is_p)
        if is_p: prime_harmonies.append(harmony)
        else: comp_harmonies.append(harmony)

    print(f"\n[SAMPLES] Primes: {len(prime_harmonies)} | Composites: {len(comp_harmonies)}")

    # 1. DISTRIBUTION TEST
    p_mean, c_mean = np.mean(prime_harmonies), np.mean(comp_harmonies)
    t_stat, p_value = stats.ttest_ind(prime_harmonies, comp_harmonies, equal_var=False)
    effect_size = p_mean - c_mean

    print("\n[RESULTS: TEST 1 - AGGREGATED DISTRIBUTION]")
    print(f"  Prime Mean Harmony:     {p_mean:.6f}")
    print(f"  Composite Mean Harmony: {c_mean:.6f}")
    print(f"  Effect Size (P-C):      {effect_size:+.6e}")
    print(f"  P-Value (Target < 0.05): {p_value:.6e}")
    
    # 2. THE TOKEN SHUFFLE
    print("\n[ACTION] TEST 2: Random ID Control (Ghost Hunter)...")
    shuffled_prime_h = []
    shuffled_comp_h = []
    
    for idx, is_p in enumerate(labels):
        # Match original token count but with random IDs
        orig_len = len(tokenizer.encode(str(test_range[idx]), add_special_tokens=False))
        rand_ids = torch.randint(1000, 150000, (1, orig_len), device=config.device)
        
        with torch.no_grad():
            _, _, metrics, _ = model(rand_ids)
            harmony = metrics['harmony'].mean().item()
        
        if is_p: shuffled_prime_h.append(harmony)
        else: shuffled_comp_h.append(harmony)

    _, p_val_shuf = stats.ttest_ind(shuffled_prime_h, shuffled_comp_h, equal_var=False)
    print(f"  Shuffled P-Value: {p_val_shuf:.6e}")
    
    print("-" * 60)
    if p_value < 0.05:
        print("  VERDICT: SIGNAL UNLOCKED.")
        if p_val_shuf > 0.05:
            print("  [CONFIRMED] IT IS TIED TO THE NUMBER LINE.")
        else:
            print("  [ALERT] IT IS ARCHITECTURAL BIAS.")
    else:
        print("  VERDICT: STILL NEUTRAL. ARCHITECTURE IS SPECTRAL-READY BUT DATA-EMPTY.")
    print("=" * 60)

if __name__ == "__main__":
    run_brutal_validation()
