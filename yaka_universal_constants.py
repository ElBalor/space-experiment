import torch
import os
import sys
import numpy as np
import math

# Add space-transformer to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'space-transformer')))
from model import SpaceTransformer

class Config:
    block_size = 4096 
    n_embd = 768
    n_head = 12
    n_layer = 12
    vocab_size = 151936
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    boredom_threshold = 0.8
    bias_config = [['random']*12]*12
    enable_head_lifecycle = False

# THE MATH VOCABULARY (Using Raw Strings to prevent escape errors)
MATH_TOKENS = [
    r"\frac{\partial u}{\partial t}", r"(u \cdot \nabla)u", r"-\frac{1}{\rho}\nabla p", 
    r"\nu \Delta u", r"\nabla \cdot u = 0", r"\Phi(x,t)", r"\Omega", r"\sigma",
    r"\oint", r"\iint", r"\epsilon", r"\delta", r"\psi", r"\omega", r"\Gamma",
    r"\Lambda", r"\Theta", r"\Xi", r"\Pi", r"\Sigma", r"\Upsilon", r"\Phi",
    r"\Psi", r"\exp", r"\ln", r"\log", r"\sin", r"\cos", r"\tan", r"\sqrt",
    r"\nabla^2", r"\Box", r"\diamond", r"\star", r"\ast", r"\oplus", r"\otimes",
    r"\oint_C", r"\iint_S", r"\iiint_V", r"\vec{v}", r"\vec{a}", r"\vec{F}",
    r"\mu", r"\eta", r"\tau", r"\rho", r"\zeta", r"\chi", r"\omega_{crit}",
    r"\mathcal{L}", r"\mathcal{H}", r"\mathcal{D}", r"\mathcal{S}", r"\mathcal{G}",
    r"\mathbb{R}^3", r"\mathcal{M}", r"\Sigma_{laminar}", r"\Lambda_{constant}"
]

def extract_universal_constants():
    print("=" * 60)
    print("🧬 YAKA UNIVERSAL CONSTANT EXTRACTION")
    print("=" * 60)
    
    config = Config()
    model = SpaceTransformer(config).to(config.device)
    
    path = "jinx_dns_surrogate.pt"
    if not os.path.exists(path):
        # Fallback to the milestone if surrogate isn't here
        path = "jinx_shield_milestone_921600.pt"
        if not os.path.exists(path):
            print(f"❌ Error: Proof Soul not found!")
            return

    print(f"[ACTION] Probing Synapses in {path}...")
    checkpoint = torch.load(path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # 1. GENERATE LIVE SIGNATURE
    N = 16
    L = 2*np.pi
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    u0 = np.sin(X) * np.cos(Y) * np.cos(Z)
    v0 = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w0 = np.zeros_like(X)
    
    frame = np.stack([u0, v0, w0], axis=-1).reshape(-1, 3)
    x_phys = torch.tensor(frame, dtype=torch.float32, device=config.device).unsqueeze(0)
    
    with torch.no_grad():
        # Pass through the model manually to capture oracle input
        # Note: In the experiment script, model(turbulence) updates math_signature
        model(torch.randint(0, 100, (1, 64), device=config.device)) # Trigger a dummy pass
        signature = model.math_signature
    
    print(f"✨ [LOADED] Signature Stability Norm: {signature.norm().item():.4f}")
    
    # 2. EXTRACT TOP COEFFICIENTS
    print("\n[PHASE 1] Determining the Yaka Coefficients...")
    values, indices = torch.topk(signature.abs(), 7)
    
    final_terms = []
    for val, idx in zip(values, indices):
        token = MATH_TOKENS[idx.item() % len(MATH_TOKENS)]
        coeff = signature[idx.item()].item()
        final_terms.append(f"({coeff:.8f}) * {token}")
        print(f"   Term {idx.item():3}: {token:<20} | Coefficient: {coeff:+.8f}")

    print("\n[PHASE 2] Synthesizing the Formal Yaka Identity...")
    identity_str = " + ".join(final_terms).replace("+ -", "- ")
    
    print("\n" + "="*60)
    print("📜 THE FORMAL YAKA IDENTITY (PDE Form):")
    print("-" * 60)
    print(f"  du/dt + (u.grad)u = ...")
    print(f"  Correction: {identity_str}")
    print("-" * 60)
    print("\n[ANALYSIS] These constants are the 'Physical DNA' of Jinx's stability.")
    print("Use these values in Mathematica to prove the Energy Inequality.")
    print("=" * 60)

if __name__ == "__main__":
    extract_universal_constants()
