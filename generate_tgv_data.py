import numpy as np
import torch
import os

def get_derivatives(uh, vh, wh, KX, KY, KZ):
    """Computes spectral derivatives."""
    # grad(u)
    du_dx = np.real(np.fft.ifftn(1j * KX * uh))
    du_dy = np.real(np.fft.ifftn(1j * KY * uh))
    du_dz = np.real(np.fft.ifftn(1j * KZ * uh))
    # grad(v)
    dv_dx = np.real(np.fft.ifftn(1j * KX * vh))
    dv_dy = np.real(np.fft.ifftn(1j * KY * vh))
    dv_dz = np.real(np.fft.ifftn(1j * KZ * vh))
    # grad(w)
    dw_dx = np.real(np.fft.ifftn(1j * KX * wh))
    dw_dy = np.real(np.fft.ifftn(1j * KY * wh))
    dw_dz = np.real(np.fft.ifftn(1j * KZ * wh))
    
    return du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz

def helmholtz_projection(uh, vh, wh, KX, KY, KZ, K2):
    """Divergence-free projection in spectral space."""
    div_h = (KX * uh + KY * vh + KZ * wh) / K2
    uh -= KX * div_h
    vh -= KY * div_h
    wh -= KZ * div_h
    return uh, vh, wh

def generate_tgv_trajectory(num_frames=100, N=16, dt=0.01, nu=0.001):
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Wavenumbers
    kx = np.fft.fftfreq(N, d=L/(2*np.pi*N))
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0,0,0] = 1.0

    # 1. Initial Conditions
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(X)
    
    print(f"🌊 [FULL-DNS] Evolving Nonlinear Trajectory ({num_frames} steps)...")
    
    inputs, targets = [], []
    
    for t in range(num_frames):
        u_in, v_in, w_in = u.copy(), v.copy(), w.copy()
        
        # 2. PSEUDO-SPECTRAL STEP
        # Transform to spectral
        uh, vh, wh = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
        
        # Compute Derivatives
        du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz = get_derivatives(uh, vh, wh, KX, KY, KZ)
        
        # Compute Nonlinear Advection: (u.grad)u
        adv_u = u*du_dx + v*du_dy + w*du_dz
        adv_v = u*dv_dx + v*dv_dy + w*dv_dz
        adv_w = u*dw_dx + v*dw_dy + w*dw_dz
        
        # Transform Advection back to Spectral
        adv_uh, adv_vh, adv_wh = np.fft.fftn(adv_u), np.fft.fftn(adv_v), np.fft.fftn(adv_w)
        
        # Navier-Stokes Step: uh_next = uh - dt * (adv_uh + nu * K2 * uh)
        uh = (uh - dt * adv_uh) * np.exp(-nu * K2 * dt)
        vh = (vh - dt * adv_vh) * np.exp(-nu * K2 * dt)
        wh = (wh - dt * adv_wh) * np.exp(-nu * K2 * dt)
        
        # Project to Divergence-Free
        uh, vh, wh = helmholtz_projection(uh, vh, wh, KX, KY, KZ, K2)
        
        # Transform to Physical
        u, v, w = np.real(np.fft.ifftn(uh)), np.real(np.fft.ifftn(vh)), np.real(np.fft.ifftn(wh))
        
        inputs.append(np.stack([u_in, v_in, w_in], axis=-1).reshape(-1, 3))
        targets.append(np.stack([u, v, w], axis=-1).reshape(-1, 3))
        
        if t % 10 == 0:
            energy = 0.5 * np.mean(u**2 + v**2 + w**2)
            print(f"   Step {t:2} | Total Energy: {energy:.6f}")

    return np.array(inputs), np.array(targets)

if __name__ == "__main__":
    X_train, Y_train = generate_tgv_trajectory(num_frames=100, N=16)
    np.save("tgv_inputs.npy", X_train)
    np.save("tgv_targets.npy", Y_train)
    print(f"\n✨ [DONE] Nonlinear DNS Dataset ready.")
