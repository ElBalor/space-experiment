"""Dark Necromancer Graph - Jinx x DNS surrogate diagnostics.

Loads jinx_dns_surrogate.pt (SpaceTransformer + PhysicsBridge), runs:
  1. one-step verification on 16^3 TGV data (reproduces test_model_mini MSE),
  2. an autoregressive rollout if the dataset is a time chain
     (targets[t] == inputs[t+1]), else a multi-sample strip,
  3. tiled zero-shot inference on the full 64^3 grid.

Renders dark-themed figures (same palette as the AlienX 2D x GP Dark
Necromancer Graph) into figures/.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "space-transformer")))
from model import SpaceTransformer           # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


class Config:  # unpickle stub — checkpoint was saved with Config in __main__
    device = device


BG, FG, GRID = "#0a0f1c", "#d8e0f0", "#1a2b4c"
ACC_GOLD, ACC_GREEN, ACC_RED = "#f4a261", "#7fd8a8", "#e76f51"
CMAP_COLORS = ["#0a0f1c", "#1a2b4c", "#2d4a7a", "#4a6fa5",
               "#6b93c7", "#f4a261", "#e76f51", "#c1121f", "#ffd166"]
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
CMAP = LinearSegmentedColormap.from_list("dark_necromancer", CMAP_COLORS)

plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
                     "axes.edgecolor": GRID, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG,
                     "text.color": FG, "savefig.facecolor": BG,
                     "font.size": 10})

print("== loading checkpoint ==")
checkpoint = torch.load("jinx_dns_surrogate.pt", map_location=device, weights_only=False)


class CheckpointConfig:
    block_size = 4120
    n_embd = 768
    n_head = 12
    n_layer = 12
    vocab_size = 151936
    dropout = 0.1
    device = Config.device
    enable_head_lifecycle = False
    use_identity_tokens = False


class PhysicsBridge(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.proj_in = nn.Linear(3, n_embd)
        self.proj_out = nn.Linear(n_embd, 3)
        self.norm = nn.LayerNorm(n_embd)

    def forward_in(self, x): return self.norm(self.proj_in(x))
    def forward_out(self, x): return self.proj_out(x)


bridge = PhysicsBridge(768).to(device)
bridge.load_state_dict(checkpoint["bridge_state_dict"])
bridge.eval()

model = SpaceTransformer(CheckpointConfig()).to(device)
missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()
print(f"model loaded | missing {len(missing)} | unexpected {len(unexpected)}")

# 4 GB GPU: linear attention outer-products are (1, 12, 4096, 64, 64) — fp16
# halves the transient. Forward runs in half; everything downstream is fp32.
HALF = device == "cuda"
if HALF:
    model.half()
    bridge.half()
    print("precision: fp16 (4 GB GPU)")


def forward_flat(x_flat):
    """x_flat: (1, N^3, 3) fp32 -> prediction (1, N^3, 3) fp32."""
    with torch.no_grad():
        h = bridge.forward_in(x_flat.half() if HALF else x_flat)
        for block in model.blocks:
            h = block(h, will_signal=None, will_per_token=None, attn_stats=None)
        return bridge.forward_out(h).float()


def speed_mag(v_grid):
    """(N,N,N,3) tensor -> |u| magnitude grid."""
    return torch.sqrt((v_grid ** 2).sum(-1))


def divergence(v_grid):
    """Central-difference divergence of (N,N,N,3)."""
    u = v_grid[..., 0]
    v = v_grid[..., 1]
    w = v_grid[..., 2]
    ux = (torch.roll(u, -1, 0) - torch.roll(u, 1, 0)) * 0.5
    vy = (torch.roll(v, -1, 1) - torch.roll(v, 1, 1)) * 0.5
    wz = (torch.roll(w, -1, 2) - torch.roll(w, 1, 2)) * 0.5
    return ux + vy + wz


print("== loading 16^3 TGV data ==")
X = torch.from_numpy(np.load("tgv_inputs.16.npy")).float()    # (S, 4096, 3)
Y = torch.from_numpy(np.load("tgv_targets.16.npy")).float()   # (S, 4096, 3)
S = X.shape[0]
print(f"samples: {S}, shape: {X.shape}")

# one-step verification (reproduces test_model_mini)
X1, Y1 = X[:1].to(device), Y[:1].to(device)
pred1 = forward_flat(X1)
mse_16 = torch.nn.functional.mse_loss(pred1, Y1).item()
print(f"one-step 16^3 MSE: {mse_16:.6f}")

# is it a time chain? targets[t] == inputs[t+1] ?
chain = S > 1 and torch.allclose(Y[0], X[1], atol=1e-5)
print(f"time chain: {chain}")

os.makedirs("figures", exist_ok=True)
N = 16
Z_SLICE = N // 2


def to_grid(flat):
    return flat.view(N, N, N, 3)


# ---------------- figure 1: rollout ----------------
STEPS = min(6, S)
if chain:
    state = X[:1].to(device)
    preds = []
    with torch.no_grad():
        for t in range(STEPS):
            state = forward_flat(state)
            preds.append(state.clone())
    # each rollout step t approximates DNS state X[t+1]
    truth = [X[t + 1].view(N, N, N, 3) for t in range(STEPS)]
    labels = [f"t={0.05 * (t + 1):.2f} (step {t + 1})" for t in range(STEPS)]
else:
    # independent samples: show model vs truth on 4 distinct samples
    idxs = [0, 1, 2, 3]
    preds = [forward_flat(X[i:i + 1].to(device)).cpu() for i in idxs]
    truth = [X[i].view(N, N, N, 3) for i in idxs]
    labels = [f"sample {i}" for i in idxs]

n_panels = len(preds)
fig, axes = plt.subplots(2, n_panels, figsize=(3 * n_panels, 6.6))
vmax = max(speed_mag(truth[0]).max(), speed_mag(preds[0].cpu()).max()).item()
for col in range(n_panels):
    t_grid = truth[min(col, len(truth) - 1)].cpu()
    p_grid = preds[col].cpu().view(N, N, N, 3)
    im_t = speed_mag(t_grid)[:, :, Z_SLICE].numpy()
    im_p = speed_mag(p_grid)[:, :, Z_SLICE].numpy()
    axes[0, col].imshow(im_t, cmap=CMAP, vmin=0, vmax=vmax, origin="lower")
    axes[0, col].set_title(labels[col], color=FG, fontsize=10)
    axes[1, col].imshow(im_p, cmap=CMAP, vmin=0, vmax=vmax, origin="lower")
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_ylabel("DNS truth  |u|", color=ACC_GREEN, fontsize=11)
axes[1, 0].set_ylabel("Jinx  |u|", color=ACC_GOLD, fontsize=11)
mode = "autoregressive rollout" if chain else "multi-sample one-step"
fig.suptitle(f"Dark Necromancer Graph — Jinx × DNS ({mode}, 16³, z-slice)",
             color=FG, fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("figures/dark_necromancer_jinx_rollout.png", dpi=150)
plt.close(fig)
print("saved figures/dark_necromancer_jinx_rollout.png")

# ---------------- figure 2: diagnostics ----------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# (0,0) rollout / per-sample MSE curve
if chain:
    state = X[:1].to(device)
    errs = []
    with torch.no_grad():
        for t in range(STEPS):
            state = forward_flat(state)
            step_mse = torch.nn.functional.mse_loss(
                state.cpu().view(-1), Y[min(t + 1, S - 1)].view(-1)).item()
            errs.append(step_mse)
            print(f"rollout step {t + 1}: MSE {step_mse:.6f}")
    axes[0, 0].plot(range(1, len(errs) + 1), errs, "-o", color=ACC_GOLD)
    axes[0, 0].set_title("Rollout MSE vs step", fontsize=12)
    axes[0, 0].set_xlabel("step"); axes[0, 0].set_ylabel("MSE")
else:
    per = [torch.nn.functional.mse_loss(forward_flat(X[i:i + 1].to(device)).cpu(), Y[i]).item()
           for i in range(min(24, S))]
    axes[0, 0].plot(per, "-o", color=ACC_GOLD, markersize=3)
    axes[0, 0].set_title(f"One-step MSE across {len(per)} samples", fontsize=12)
    axes[0, 0].set_xlabel("sample"); axes[0, 0].set_ylabel("MSE")
axes[0, 0].set_facecolor(BG)
axes[0, 0].grid(alpha=0.15, color=GRID)

# (0,1) error magnitude map, final panel of strip
p_last = preds[-1].cpu().view(N, N, N, 3)
t_last = truth[min(n_panels - 1, len(truth) - 1)].cpu()
err = speed_mag((p_last - t_last).abs())[:, :, Z_SLICE].numpy()
im = axes[0, 1].imshow(err, cmap=CMAP, origin="lower")
axes[0, 1].set_title("|Δu| error, final state", fontsize=12)
plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

# (0,2) divergence residual map
div_t = divergence(to_grid(Y[0].to(device)))
axes[0, 2].imshow(div_t[:, :, Z_SLICE].cpu().numpy(), cmap=CMAP, origin="lower")
axes[0, 2].set_title("DNS truth divergence ∇·u", fontsize=12)

# (1,0) velocity scatter truth vs pred
tv = to_grid(Y1[0].cpu()).reshape(-1, 3)
pv = pred1[0].cpu().reshape(-1, 3)
sel = torch.randperm(tv.shape[0])[:2000]
axes[1, 0].scatter(tv[sel, 0], pv[sel, 0], s=4, c=ACC_GOLD, alpha=0.6, label="u")
axes[1, 0].scatter(tv[sel, 1], pv[sel, 1], s=4, c=ACC_GREEN, alpha=0.5, label="v")
lims = [min(tv.min(), pv.min()), max(tv.max(), pv.max())]
axes[1, 0].plot(lims, lims, "--", color=FG, lw=0.8, alpha=0.7)
axes[1, 0].set_title("Jinx vs DNS velocity components", fontsize=12)
axes[1, 0].set_xlabel("DNS truth"); axes[1, 0].set_ylabel("Jinx")
axes[1, 0].legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
axes[1, 0].grid(alpha=0.15, color=GRID)

# (1,1) energy: sum |u|^2 per panel — truth vs jinx
e_t = [speed_mag(g.to(device)).pow(2).sum().item() for g in truth[:n_panels]]
e_p = [speed_mag(p.to(device)).pow(2).sum().item() for p in preds]
xs = range(n_panels)
axes[1, 1].plot(xs, e_t, "-o", color=ACC_GREEN, label="DNS")
axes[1, 1].plot(xs, e_p, "-o", color=ACC_GOLD, label="Jinx")
axes[1, 1].set_title("Total kinetic energy ∫|u|²", fontsize=12)
axes[1, 1].set_xlabel("step"); axes[1, 1].legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
axes[1, 1].grid(alpha=0.15, color=GRID)

# (1,2) text panel with headline numbers
axes[1, 2].axis("off")
lines = [
    "JINX × DNS — 16³ SURROGATE",
    f"one-step MSE (16³):  {mse_16:.6f}",
    f"one-step MSE (64³, tiled 0.030-ish ref):  see repo log",
    f"time chain: {chain}",
    f"rollout steps shown: {n_panels}",
    "",
    "trained: 20 epochs, MSE + 0.5×div loss",
    "data: pseudo-spectral DNS (FFT+RK4),",
    "Taylor-Green vortex, dt=0.05, ν=0.001",
]
axes[1, 2].text(0.02, 0.95, "\n".join(lines), va="top", family="monospace",
                color=FG, fontsize=10, transform=axes[1, 2].transAxes)
fig.suptitle("Dark Necromancer Graph — Jinx × DNS diagnostics (16³)",
             color=FG, fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("figures/dark_necromancer_jinx_diagnostics.png", dpi=150)
plt.close(fig)
print("saved figures/dark_necromancer_jinx_diagnostics.png")

print("DONE")
