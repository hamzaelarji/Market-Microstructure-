"""Notebook 02 — Closed‑form approximations vs exact ODE solutions.

Reproduces Figures 2–7 of Guéant (2017):
  - Figs 2–5 : IG, σ normal  (approx overlaid on exact)
  - Figs 6–7 : IG, σ/2       (shows approx improves at low vol)
  - Figs 11–14: HY            (same overlay)

Run:  python 02_closed_form.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from asset.params import IG, HY, GAMMA, T
from src.ode_solver_1d import solve_general
ROOT = Path(__file__).resolve().parents[1]   
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
from src.closed_form import approx_quotes

# ═══════════════════════════════════════════════════════════════════════
#  Solve exact ODE
# ═══════════════════════════════════════════════════════════════════════

print("Solving IG Model A ...")
sol_ig = solve_general(IG, GAMMA, T, xi=GAMMA, N_t=7200)

print("Solving IG Model A (σ/2) ...")
IG_half = {**IG, "sigma": IG["sigma"] / 2}
sol_ig_half = solve_general(IG_half, GAMMA, T, xi=GAMMA, N_t=7200)

print("Solving HY Model A ...")
sol_hy = solve_general(HY, GAMMA, T, xi=GAMMA, N_t=7200)

# ═══════════════════════════════════════════════════════════════════════
#  Plotting helper
# ═══════════════════════════════════════════════════════════════════════

def overlay_plots(sol, params, label, sigma_label="σ"):
    """Plot bid, ask, spread, skew:  exact (crosses) + approx (line)."""

    lots = sol["lots"]
    Q = int(params["Q"])

    # Closed‑form approx
    n_arr = np.arange(-Q + 1, Q)
    db_cf, da_cf = approx_quotes(n_arr, params, GAMMA, xi=GAMMA)
    spread_cf = db_cf + da_cf
    skew_cf = db_cf - da_cf

    # Exact at t = 0
    db_exact = sol["delta_bid"][0, :]
    da_exact = sol["delta_ask"][0, :]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Bid ──
    ax = axes[0, 0]
    mask_b = np.isfinite(db_exact)
    ax.plot(lots[mask_b], db_exact[mask_b], "x", ms=8, label="Exact (ODE)")
    idx = np.array([int(n + Q) for n in n_arr])
    ax.plot(n_arr, db_cf, "-", lw=2, label="Approx (closed‑form)")
    ax.set_title(f"{label} ({sigma_label}) — δ^bid(0, n)")
    ax.set_xlabel("n (lots)")
    ax.set_ylabel("δ^bid")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Ask ──
    ax = axes[0, 1]
    mask_a = np.isfinite(da_exact)
    ax.plot(lots[mask_a], da_exact[mask_a], "o", ms=8, label="Exact (ODE)")
    ax.plot(n_arr, da_cf, "-", lw=2, label="Approx (closed‑form)")
    ax.set_title(f"{label} ({sigma_label}) — δ^ask(0, n)")
    ax.set_xlabel("n (lots)")
    ax.set_ylabel("δ^ask")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Spread ──
    ax = axes[1, 0]
    spread_exact = db_exact + da_exact
    mask_s = np.isfinite(spread_exact)
    ax.plot(lots[mask_s], spread_exact[mask_s], "s", ms=8, label="Exact")
    ax.plot(n_arr, spread_cf, "-", lw=2, label="Approx")
    ax.set_title(f"{label} ({sigma_label}) — Spread(0, n)")
    ax.set_xlabel("n (lots)")
    ax.set_ylabel("δ^b + δ^a")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Skew ──
    ax = axes[1, 1]
    skew_exact = db_exact - da_exact
    mask_k = np.isfinite(skew_exact)
    ax.plot(lots[mask_k], skew_exact[mask_k], "D", ms=8, label="Exact")
    ax.plot(n_arr, skew_cf, "-", lw=2, label="Approx")
    ax.set_title(f"{label} ({sigma_label}) — Skew(0, n)")
    ax.set_xlabel("n (lots)")
    ax.set_ylabel("δ^b − δ^a")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Closed‑form approx vs exact — {label} ({sigma_label})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f"fig02_{label}_{sigma_label.replace('/', '_')}.png"
    fig.savefig(FIG_DIR /fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
#  Generate figures
# ═══════════════════════════════════════════════════════════════════════

print("\n=== IG — σ normal (Figs 2–5) ===")
overlay_plots(sol_ig, IG, "IG", "σ")

print("\n=== IG — σ/2 (Figs 6–7) ===")
overlay_plots(sol_ig_half, IG_half, "IG", "σ÷2")

print("\n=== HY — σ normal (Figs 11–14) ===")
overlay_plots(sol_hy, HY, "HY", "σ")

print("\nDone!")
