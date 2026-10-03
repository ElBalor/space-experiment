import numpy as np
import os
import time

def generate_tgv_ultimate(N=32, num_frames=500, dt=0.01, nu=0.0005, L=2*np.pi):
    """
    ULTIMATE PSEUDO-SPECTRAL DNS SOLVER (Navier-Stokes Heart)
    - Time Stepping: RK4 (Runge-Kutta 4th Order)
    - De-aliasing: 2/3 Rule (Spectral Filtering)
    - Precision: Double for generation, Float for training
    """
    print("=" * 60)
    print(f"🌊 GENERATING ULTIMATE DNS DATA: {N}^3 | {num_frames} Frames")
    print(f"   Viscosity: {nu} | Time-step: {dt}")
    print("=" * 60)

    # 1. Grid Setup
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Fourier wave numbers
    k = np.fft.fftfreq(N, d=L/(2*np.pi*N))
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1e-12 # Avoid div by zero

    # 2/3 De-aliasing Filter
    k_max = (2.0/3.0) * (N/2.0)
    mask = (np.abs(KX) < k_max) * (np.abs(KY) < k_max) * (np.abs(KZ) < k_max)

    def get_advection(uh, vh, wh):
        u = np.real(np.fft.ifftn(uh))
        v = np.real(np.fft.ifftn(vh))
        w = np.real(np.fft.ifftn(wh))

        du_dx = np.real(np.fft.ifftn(1j * KX * uh))
        du_dy = np.real(np.fft.ifftn(1j * KY * uh))
        du_dz = np.real(np.fft.ifftn(1j * KZ * uh))
        
        dv_dx = np.real(np.fft.ifftn(1j * KX * vh))
        dv_dy = np.real(np.fft.ifftn(1j * KY * vh))
        dv_dz = np.real(np.fft.ifftn(1j * KZ * vh))
        
        dw_dx = np.real(np.fft.ifftn(1j * KX * wh))
        dw_dy = np.real(np.fft.ifftn(1j * KY * wh))
        dw_dz = np.real(np.fft.ifftn(1j * KZ * wh))

        adv_u = u*du_dx + v*du_dy + w*du_dz
        adv_v = u*dv_dx + v*dv_dy + w*dv_dz
        adv_w = u*dw_dx + v*dw_dy + w*dw_dz

        adv_uh = np.fft.fftn(adv_u)
        adv_vh = np.fft.fftn(adv_v)
        adv_wh = np.fft.fftn(adv_w)

        return adv_uh * mask, adv_vh * mask, adv_wh * mask

    def helmholtz_projection(uh, vh, wh):
        div_h = (KX * uh + KY * vh + KZ * wh) / K2
        uh -= KX * div_h
        vh -= KY * div_h
        wh -= KZ * div_h
        return uh, vh, wh

    # 2. Initial Conditions
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(X)

    uh, vh, wh = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
    uh, vh, wh = helmholtz_projection(uh, vh, wh)

    inputs, targets = [], []
    start_time = time.time()

    # 3. Time Stepping: RK4
    for t_step in range(num_frames):
        u_in = np.real(np.fft.ifftn(uh)).astype(np.float32)
        v_in = np.real(np.fft.ifftn(vh)).astype(np.float32)
        w_in = np.real(np.fft.ifftn(wh)).astype(np.float32)
        inputs.append(np.stack([u_in, v_in, w_in], axis=-1).reshape(-1, 3))

        def f(u_h, v_h, w_h):
            adv_u, adv_v, adv_w = get_advection(u_h, v_h, w_h)
            du_dt = -adv_u - nu * K2 * u_h
            dv_dt = -adv_v - nu * K2 * v_h
            dw_dt = -adv_w - nu * K2 * w_h
            return du_dt, dv_dt, dw_dt

        k1_u, k1_v, k1_w = f(uh, vh, wh)
        k2_u, k2_v, k2_w = f(uh + 0.5*dt*k1_u, vh + 0.5*dt*k1_v, wh + 0.5*dt*k1_w)
        k3_u, k3_v, k3_w = f(uh + 0.5*dt*k2_u, vh + 0.5*dt*k2_v, wh + 0.5*dt*k2_w)
        k4_u, k4_v, k4_w = f(uh + dt*k3_u, vh + dt*k3_v, wh + dt*k3_w)

        uh += (dt/6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        vh += (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        wh += (dt/6.0) * (k1_w + 2*k2_w + 2*k3_w + k4_w)
        uh, vh, wh = helmholtz_projection(uh, vh, wh)

        u_out = np.real(np.fft.ifftn(uh)).astype(np.float32)
        v_out = np.real(np.fft.ifftn(vh)).astype(np.float32)
        w_out = np.real(np.fft.ifftn(wh)).astype(np.float32)
        targets.append(np.stack([u_out, v_out, w_out], axis=-1).reshape(-1, 3))

        if (t_step + 1) % 100 == 0:
            energy = 0.5 * np.mean(u_out**2 + v_out**2 + w_out**2)
            print(f"   Step {t_step+1:4}/{num_frames} | Energy: {energy:.6f} | Time: {time.time()-start_time:.2f}s")

    np.save("tgv_inputs_ultimate.npy", np.array(inputs))
    np.save("tgv_targets_ultimate.npy", np.array(targets))
    print(f"✨ [SUCCESS] Ultimate Dataset Saved: tgv_inputs_ultimate.npy")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=64)
    parser.add_argument('--frames', type=int, default=500)
    parser.add_argument('--nu', type=float, default=0.0005)
    args = parser.parse_args()
    
    data = generate_tgv_ultimate(N=args.N, num_frames=args.frames, nu=args.nu)
    
    # Save with N-specific names for multi-resolution support
    import numpy as np
    inputs  = np.array([d[0] for d in data]) if isinstance(data, list) else None
    targets = np.array([d[1] for d in data]) if isinstance(data, list) else None
    
    # The function already saves — just print confirmation
    print(f'\n✅ Dataset ready for N={args.N}³ training')
