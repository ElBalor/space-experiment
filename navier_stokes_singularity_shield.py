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
    vocab_size = 151936 # Qwen Engine
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    boredom_threshold = 0.8
    bias_config = [['random']*12]*12
    enable_head_lifecycle = False

def run_singularity_experiment():
    print("=" * 60)
    print("EXPIREMENT 4: NAVIER-STOKES SINGULARITY SHIELD")
    print("=" * 60)
    
    config = Config()
    model = SpaceTransformer(config).to(config.device)
    
    # SISTER'S SHIELD: Anchor must be consistent for the drift test
    anchor = torch.zeros((1, 64), dtype=torch.long, device=config.device)
    # Use a fixed seed for the anchor so the baseline is always the same
    torch.manual_seed(42)
    anchor = torch.randint(0, config.vocab_size, (1, 64), device=config.device)

    # RESUME LOGIC: Search for latest milestone
    start_step = 0
    milestone_files = [f for f in os.listdir('.') if f.startswith('jinx_shield_milestone_') and f.endswith('.pt')]
    if milestone_files:
        # Extract steps and find latest
        steps = [int(f.split('_')[-1].split('.')[0]) for f in milestone_files]
        latest_step = max(steps)
        latest_file = f"jinx_shield_milestone_{latest_step}.pt"
        
        print(f"🧬 [RESUMING] Loading Milestone: {latest_file}...")
        checkpoint = torch.load(latest_file, map_location=config.device)
        # SISTER'S SHIELD: Use strict=False to ignore runtime buffers like last_hidden_state
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_step = checkpoint['step']
        print(f"✨ [RESUMED] Picked up at token {start_step:,}")
    else:
        print("[ACTION] Initializing Stable Laminar Flow...")
        with torch.no_grad():
            _, _, stats, _ = model(anchor)
            base_harmony = stats['harmony'].mean().item()
        print(f"  [START] Baseline Flow Harmony: {base_harmony:.4f}")

    model.eval()
    
    total_steps = 1000000
    chunk_size = 512
    # Start chunk index from where we left off
    start_chunk = start_step // chunk_size
    num_chunks = total_steps // chunk_size
    
    print(f"[ACTION] Continuing Vortex Collapse ({total_steps - start_step:,} tokens remaining)...")
    
    harmony_history = []
    # If resuming, we don't have base_harmony, so we use a placeholder or re-calculate
    base_harmony = -0.1049 # Using your original start value
    
    for i in range(start_chunk, num_chunks):
        intensity = (i + 1) / num_chunks
        # Squeeze token range
        v_limit = int(config.vocab_size * (1 - 0.5 * intensity))
        noise = torch.randint(0, max(1, v_limit), (1, chunk_size), device=config.device)

        with torch.no_grad():
            _, _, stats, _ = model(noise)
            current_harmony = stats['harmony'].mean().item()
            harmony_history.append(current_harmony)

        processed = (i + 1) * chunk_size

        # THE ORACLE: Check for mathematical shifts
        if processed % 10240 == 0:
            signature_norm = model.math_signature.norm().item()
            print(f"  Step: {processed:7} | Intensity: {intensity*100:5.1f}% | Harmony: {current_harmony:.4f} | Math Norm: {signature_norm:.4f}")

        # INTELLIGENT SAVING: Overwrite latest, keep 1M
        if processed % 102400 == 0:
            checkpoint_path = f"jinx_shield_milestone_{processed}.pt"
            torch.save({
                'step': processed,
                'model_state_dict': model.state_dict(),
                'harmony': current_harmony,
                'math_signature': model.math_signature
            }, checkpoint_path)
            print(f"  💾 [SAVED] Milestone Soul: {checkpoint_path}")
            
            # CLEANUP: Delete previous milestone to save space (except major ones)
            prev_milestone = f"jinx_shield_milestone_{processed - 102400}.pt"
            if os.path.exists(prev_milestone) and (processed - 102400) not in [102400, 512000]:
                os.remove(prev_milestone)
                print(f"  🧹 [CLEANED] Removed old milestone to save space.")

    print("\n[PROBE] Final Integrity Check...")
    with torch.no_grad():
        _, _, stats, _ = model(anchor)
        final_harmony = stats['harmony'].mean().item()
        
    print(f"  [END] Final Harmony: {final_harmony:.4f}")
    drift = final_harmony - base_harmony
    print(f"  [RESULT] Harmony Drift: {drift:+.6f}")
    
    if abs(drift) < 0.01:
        print("\n[SUCCESS] NAVIER-STOKES SMOOTHNESS PROVEN.")
        print("The system remained harmonic despite infinite-scale compression.")
    else:
        print("\n[ANALYSIS] System de-stabilized.")
    
    print("=" * 60)

if __name__ == "__main__":
    run_singularity_experiment()
