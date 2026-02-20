"""Notebook 03 — Model A vs Model B comparison.

Reproduces Figures 8–9 (IG) and 15–16 (HY) from Guéant (2017).
Shows that Model A (ξ = γ) and Model B (ξ = 0) produce very similar quotes,
validating Model B as a useful simplification.

Run:  python 03_model_a_vs_b.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from asset.params import IG, HY, GAMMA, T
from src.ode_solver_1d import solve_general
ROOT = Path(__file__).resolve().parents[1]   # PROJECT/
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
#  Solve both models for both assets
# ═══════════════════════════════════════════════════════════════════════

print("Solving IG Model A ...")
sol_ig_a = solve_general(IG, GAMMA, T, xi=GAMMA, N_t=7200)

print("Solving IG Model B ...")
sol_ig_b = solve_general(IG, GAMMA, T, xi=0.0, N_t=7200)

print("Solving HY Model A ...")
sol_hy_a = solve_general(HY, GAMMA, T, xi=GAMMA, N_t=7200)

print("Solving HY Model B ...")
sol_hy_b = solve_general(HY, GAMMA, T, xi=0.0, N_t=7200)

# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def compare_ab(sol_a, sol_b, params, label, fig_nums):
    """Compare quotes at t = 0 for Model A vs B."""

    lots = sol_a["lots"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Bid ──
    ax = axes[0]
    db_a = sol_a["delta_bid"][0, :]
    db_b = sol_b["delta_bid"][0, :]
    mask_a = np.isfinite(db_a)
    mask_b = np.isfinite(db_b)
    ax.plot(lots[mask_a], db_a[mask_a], "x", ms=9, label="Model A (ξ=γ)")
    ax.plot(lots[mask_b], db_b[mask_b], "o", ms=7, mfc="none", label="Model B (ξ=0)")
    ax.set_title(f"Fig {fig_nums[0]}: {label} — δ^bid(0, n)")
    ax.set_xlabel("n (lots)")
    ax.set_ylabel("δ^bid")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Ask ──
    ax = axes[1]
    da_a = sol_a["delta_ask"][0, :]
    da_b = sol_b["delta_ask"][0, :]
    mask_a = np.isfinite(da_a)
    mask_b = np.isfinite(da_b)
    ax.plot(lots[mask_a], da_a[mask_a], "x", ms=9, label="Model A (ξ=γ)")
    ax.plot(lots[mask_b], da_b[mask_b], "o", ms=7, mfc="none", label="Model B (ξ=0)")
    ax.set_title(f"Fig {fig_nums[1]}: {label} — δ^ask(0, n)")
    ax.set_xlabel("n (lots)")
    ax.set_ylabel("δ^ask")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Model A vs B — {label}", fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f"fig03_{label}_A_vs_B.png"
    fig.savefig(FIG_DIR/fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

    # ── Quantify difference ──
    mask = np.isfinite(db_a) & np.isfinite(db_b)
    max_diff_bid = np.max(np.abs(db_a[mask] - db_b[mask]))
    rel_diff_bid = max_diff_bid / np.mean(np.abs(db_a[mask]))
    print(f"  {label}: max |δ^b_A − δ^b_B| = {max_diff_bid:.4e}  "
          f"(relative: {rel_diff_bid:.2%})")


print("\n=== IG: Model A vs B (Figs 8–9) ===")
compare_ab(sol_ig_a, sol_ig_b, IG, "IG", fig_nums=(8, 9))

print("\n=== HY: Model A vs B (Figs 15–16) ===")
compare_ab(sol_hy_a, sol_hy_b, HY, "HY", fig_nums=(15, 16))

print("\nDone!")
