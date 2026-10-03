import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

import os

import sys

import time



# SISTER'S SHIELD: Nuclear Environment Sanitizer

os.environ["TRITON_DISABLE"] = "1"

os.environ["TORCH_COMPILE_DISABLE"] = "1"

sys.path = [str(p) for p in sys.path if isinstance(p, str) and p]



# Priority for local model

current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:

    sys.path.insert(0, current_dir)



from model import SpaceTransformer



class Config:
    block_size = 262144  # 64³ grid
    n_embd = 1024
    n_head = 16
    n_layer = 16
    vocab_size = 151936
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    boredom_threshold = 0.8
    bias_config = [['random']*16]*16
    enable_head_lifecycle = False



class PhysicsBridge(nn.Module):

    def __init__(self, n_embd):

        super().__init__()

        self.proj_in = nn.Linear(3, n_embd)

        self.proj_out = nn.Linear(n_embd, 3)

        self.norm = nn.LayerNorm(n_embd)

    def forward_in(self, x): return self.norm(self.proj_in(x))

    def forward_out(self, x): return self.proj_out(x)



def run_duel(rollout_steps=500, N=64, nu=0.0005, dt=0.05):

    print("=" * 60)

    print("⚔️ THE FINAL DUEL: jinX ENGINE vs. RK4 DNS")

    print(f"   Resolution: {N}^3 | Horizon: {rollout_steps} steps")

    print("=" * 60)

    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = "jinx_dns_surrogate.pt"

    

    if not os.path.exists(model_path):

        print(f"❌ Error: {model_path} not found! Finish training first.")

        return



    # 1. Initialize Challenger (Jinx)
    print("[ACTION] Awakening the jinX Engine...")
    print("[DEBUG] Clearing GPU cache...")
    torch.cuda.empty_cache()
    print("[DEBUG] Loading checkpoint to CPU...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"[DEBUG] Checkpoint keys: {checkpoint.keys()}")
    # Use local Config (already has correct 64³ values)
    config = Config()
    print(f"[DEBUG] Using local config: block_size={config.block_size}, n_embd={config.n_embd}")
    print("[DEBUG] Creating model on CPU (offloading)...")
    model = SpaceTransformer(config).to('cpu')
    print("[DEBUG] Loading model state...")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("[DEBUG] Creating bridge on CPU (offloading)...")
    bridge = PhysicsBridge(config.n_embd).to('cpu')
    print("[DEBUG] Loading bridge state...")
    bridge.load_state_dict(checkpoint['bridge_state_dict'])
    print("[DEBUG] Setting eval mode...")
    model.eval(); bridge.eval()
    print("[DEBUG] Model loaded on CPU - will move to GPU per inference")



    # 2. Setup Grid for the End-Boss (RK4 DNS)

    L = 2*np.pi

    x = np.linspace(0, L, N, endpoint=False)

    k = np.fft.fftfreq(N, d=L/(2*np.pi*N))

    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')

    K2 = KX**2 + KY**2 + KZ**2

    K2[0,0,0] = 1e-12

    mask = (np.abs(KX) < (2/3)*(N/2)) * (np.abs(KY) < (2/3)*(N/2)) * (np.abs(KZ) < (2/3)*(N/2))



    # Initial Condition

    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    u = np.sin(X) * np.cos(Y) * np.cos(Z)

    v = -np.cos(X) * np.sin(Y) * np.cos(Z)

    w = np.zeros_like(X)

    

    uh, vh, wh = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)

    current_jinx = torch.from_numpy(np.stack([u,v,w], axis=-1).reshape(1, -1, 3)).float().to(device)

    

    j_energies, d_energies = [], []

    j_times, d_times = [], []

    mses = []



    print("\n[ACTION] Duel Commencing...")

    

    with torch.no_grad():

        for t in range(rollout_steps):

            # --- BOSS TURN (RK4 DNS) ---

            t0 = time.time()

            def f(uh_in, vh_in, wh_in):

                up = np.real(np.fft.ifftn(uh_in))

                vp = np.real(np.fft.ifftn(vh_in))

                wp = np.real(np.fft.ifftn(wh_in))

                du_dx = np.real(np.fft.ifftn(1j*KX*uh_in))

                du_dy = np.real(np.fft.ifftn(1j*KY*uh_in))

                du_dz = np.real(np.fft.ifftn(1j*KZ*uh_in))

                dv_dx = np.real(np.fft.ifftn(1j*KX*vh_in))

                dv_dy = np.real(np.fft.ifftn(1j*KY*vh_in))

                dv_dz = np.real(np.fft.ifftn(1j*KZ*vh_in))

                dw_dx = np.real(np.fft.ifftn(1j*KX*wh_in))

                dw_dy = np.real(np.fft.ifftn(1j*KY*wh_in))

                dw_dz = np.real(np.fft.ifftn(1j*KZ*wh_in))

                return (-np.fft.fftn(up*du_dx+vp*du_dy+wp*du_dz)*mask - nu*K2*uh_in), \
                       (-np.fft.fftn(up*dv_dx+vp*dv_dy+wp*dv_dz)*mask - nu*K2*vh_in), \
                       (-np.fft.fftn(up*dw_dx+vp*dw_dy+wp*dw_dz)*mask - nu*K2*wh_in)

            

            k1 = f(uh, vh, wh)

            k2 = f(uh+0.5*dt*k1[0], vh+0.5*dt*k1[1], wh+0.5*dt*k1[2])

            k3 = f(uh+0.5*dt*k2[0], vh+0.5*dt*k2[1], wh+0.5*dt*k2[2])

            k4 = f(uh+dt*k3[0], vh+dt*k3[1], wh+dt*k3[2])

            uh += (dt/6)*(k1[0]+2*k2[0]+2*k3[0]+k4[0])

            vh += (dt/6)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])

            wh += (dt/6)*(k1[2]+2*k2[2]+2*k3[2]+k4[2])

            d_times.append(time.time() - t0)

            d_energies.append(0.5 * np.mean(np.real(np.fft.ifftn(uh))**2 + np.real(np.fft.ifftn(vh))**2 + np.real(np.fft.ifftn(wh))**2))

            

            # --- CHALLENGER TURN (Jinx) ---
            t1 = time.time()
            # Move model to GPU for inference
            model.to(device)
            bridge.to(device)
            current_jinx = current_jinx.to(device)
            xj = bridge.forward_in(current_jinx)
            for block in model.blocks: xj = block(xj, None, None, None)
            current_jinx = bridge.forward_out(xj)
            # Move back to CPU to free GPU
            model.to('cpu')
            bridge.to('cpu')
            current_jinx = current_jinx.to('cpu')
            torch.cuda.empty_cache()
            j_times.append(time.time() - t1)
            j_energies.append(0.5 * torch.mean(current_jinx**2).item())

            

            dns_field = np.stack([np.real(np.fft.ifftn(uh)), np.real(np.fft.ifftn(vh)), np.real(np.fft.ifftn(wh))], axis=-1).reshape(1, -1, 3)

            mses.append(F.mse_loss(current_jinx, torch.from_numpy(dns_field).float().to(device)).item())



            if (t+1) % 100 == 0:

                print(f"   Step {t+1:4}/{rollout_steps} | Speedup: {d_times[-1]/j_times[-1]:.1f}x | MSE: {mses[-1]:.2e}")



    print("\n" + "=" * 60)

    print(f"🏆 DUEL SUMMARY | Speedup: {np.mean(d_times)/np.mean(j_times):.1f}x | Final MSE: {mses[-1]:.2e}")

    print("=" * 60)



    plt.figure(figsize=(10, 6))

    plt.plot(d_energies, 'k-', label='RK4 DNS', linewidth=2)

    plt.plot(j_energies, 'r--', label='jinX Engine', linewidth=2)

    plt.xlabel("Step"); plt.ylabel("Energy"); plt.title(f"Navier-Stokes Duel (N={N})")

    plt.legend(); plt.grid(True, alpha=0.3)

    plt.savefig("yaka_energy_duel.png")

    print("✨ [SAVED] Result: yaka_energy_duel.png")



if __name__ == "__main__":

    run_duel(rollout_steps=500)

