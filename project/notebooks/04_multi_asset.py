"""Notebook 04 — Multi‑asset optimal quotes (IG + HY).

Reproduces Figures 17–19 from Guéant (2017):
  - Fig 17: 3D surface  (n_IG, n_HY) → δ^{IG,bid}(0)
  - Fig 18: 3D surface  (n_IG, n_HY) → δ^{HY,bid}(0)
  - Fig 19: Cross‑section  n_IG → δ^{HY,bid}(0, n_IG, 0)  for ρ ∈ {0, 0.3, 0.6, 0.9}

WARNING: The 2D solver is slow (~2 min per solve at N_t=720).
         Adjust N_t below if needed.

Run:  python 04_multi_asset.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from asset.params import IG, HY, GAMMA, T, RHO
from src.ode_solver_2d import solve_2d
ROOT = Path(__file__).resolve().parents[1]   # PROJECT/
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_T_2D = 720   # reduced steps for speed (increase to 7200 for full accuracy)

# ═══════════════════════════════════════════════════════════════════════
#  Solve 2D at ρ = 0.9 (main case)
# ═══════════════════════════════════════════════════════════════════════

print(f"Solving 2D Model A at ρ = {RHO} (N_t = {N_T_2D}) ...")
sol = solve_2d(IG, HY, GAMMA, RHO, T, xi=GAMMA, N_t=N_T_2D)

Q1 = int(IG["Q"])
Q2 = int(HY["Q"])

# ═══════════════════════════════════════════════════════════════════════
#  Fig 17: 3D surface  δ^{IG,bid}(0, n_IG, n_HY)
# ═══════════════════════════════════════════════════════════════════════

def surface_plot(sol, asset_label, quote_key, fig_num):
    n1_range = np.arange(-Q1, Q1)     # bid: n < Q
    n2_range = np.arange(-Q2, Q2 + 1)
    N1, N2 = np.meshgrid(n1_range, n2_range, indexing="ij")
    Z = np.full_like(N1, np.nan, dtype=float)

    for i, n1 in enumerate(n1_range):
        for j, n2 in enumerate(n2_range):
            key = (n1, n2)
            if key in sol["idx"]:
                jj = sol["idx"][key]
                Z[i, j] = sol[quote_key][0, jj]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(N1, N2, Z, cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("n_IG (lots)")
    ax.set_ylabel("n_HY (lots)")
    ax.set_zlabel(f"δ^{{{asset_label},bid}}")
    ax.set_title(f"Fig {fig_num}: δ^{{{asset_label},bid}}(0, n_IG, n_HY)  —  ρ = {RHO}")
    fig.tight_layout()
    fname = f"fig04_{fig_num}_{asset_label}_bid_surface.png"
    fig.savefig(FIG_DIR/fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")

print("\n=== Fig 17: IG bid surface ===")
surface_plot(sol, "IG", "delta_bid_1", 17)

print("\n=== Fig 18: HY bid surface ===")
surface_plot(sol, "HY", "delta_bid_2", 18)

# ═══════════════════════════════════════════════════════════════════════
#  Fig 19: δ^{HY,bid}(0, n_IG, 0) for different ρ
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Fig 19: HY bid cross‑section for varying ρ ===")

rho_values = [0.0, 0.3, 0.6, 0.9]
fig, ax = plt.subplots(figsize=(10, 6))

for rho_val in rho_values:
    print(f"  Solving 2D at ρ = {rho_val} ...")
    sol_rho = solve_2d(IG, HY, GAMMA, rho_val, T, xi=GAMMA, N_t=N_T_2D)

    n1_range = np.arange(-Q1, Q1)  # bid defined for n1 < Q1
    db_hy = []
    for n1 in n1_range:
        key = (n1, 0)
        if key in sol_rho["idx"]:
            jj = sol_rho["idx"][key]
            db_hy.append(sol_rho["delta_bid_2"][0, jj])
        else:
            db_hy.append(np.nan)

    ax.plot(n1_range, db_hy, "o-", ms=6, label=f"ρ = {rho_val}")

ax.set_xlabel("n_IG (lots)")
ax.set_ylabel("δ^{HY,bid}(0, n_IG, n_HY=0)")
ax.set_title("Fig 19: Cross‑asset effect of IG inventory on HY bid quote")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fname = "fig04_19_HY_bid_cross_rho.png"
fig.savefig(FIG_DIR/fname, dpi=150)
plt.close(fig)
print(f"  Saved {fname}")

print("\nDone!")
