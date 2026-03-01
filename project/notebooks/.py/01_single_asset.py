"""Notebook 01 — Single‑asset optimal quotes (IG & HY).

Reproduces Figures 1–5 (IG) and 10–14 (HY) from Guéant (2017).
Uses Model A (ξ = γ) with Newton solver (solve_general).

Run:  python 01_single_asset.py
Output:  fig01_*.png  in current directory
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
#  Solve
# ═══════════════════════════════════════════════════════════════════════

print("Solving IG Model A ...")
sol_ig = solve_general(IG, GAMMA, T, xi=GAMMA, N_t=7200)

print("Solving HY Model A ...")
sol_hy = solve_general(HY, GAMMA, T, xi=GAMMA, N_t=7200)

# ═══════════════════════════════════════════════════════════════════════
#  Helper
# ═══════════════════════════════════════════════════════════════════════

def plot_asset(sol, params, label, fig_offset):
    """Generate Figures 1–5 (or 10–14) for one asset."""

    lots = sol["lots"]
    times = sol["times"]
    Q = int(params["Q"])

    # ── Fig X+1 : t → δ^bid(t, n) for each n ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, 2 * Q - 1))
    for i, n in enumerate(range(-Q + 1, Q)):
        i_lot = int(n + Q)
        db = sol["delta_bid"][:, i_lot]
        if np.all(np.isnan(db)):
            continue
        ax.plot(times, db, color=colors[i], label=f"n={n:+d}", linewidth=0.8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("δ^bid ($/upfront)")
    ax.set_title(f"Fig {fig_offset+1}: {label} — δ^bid(t, n) over time")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR /f"fig01_{fig_offset+1}_{label}_bid_vs_time.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig01_{fig_offset+1}")

    # ── Fig X+2 : n → δ^bid(0, n) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = lots[1:]                                # skip n = -Q (no bid quote)
    db0 = sol["delta_bid"][0, 1:]
    mask = np.isfinite(db0)
    ax.plot(ns[mask], db0[mask], "x-", markersize=8, color="C0")
    ax.set_xlabel("Inventory n (lots)")
    ax.set_ylabel("δ^bid ($/upfront)")
    ax.set_title(f"Fig {fig_offset+2}: {label} — δ^bid(0, n)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR /f"fig01_{fig_offset+2}_{label}_bid_vs_n.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig01_{fig_offset+2}")

    # ── Fig X+3 : n → δ^ask(0, n) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = lots[:-1]                               # skip n = +Q (no ask quote)
    da0 = sol["delta_ask"][0, :-1]
    mask = np.isfinite(da0)
    ax.plot(ns[mask], da0[mask], "o-", markersize=8, color="C1")
    ax.set_xlabel("Inventory n (lots)")
    ax.set_ylabel("δ^ask ($/upfront)")
    ax.set_title(f"Fig {fig_offset+3}: {label} — δ^ask(0, n)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR /f"fig01_{fig_offset+3}_{label}_ask_vs_n.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig01_{fig_offset+3}")

    # ── Fig X+4 : n → spread(0, n) ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    spread = sol["delta_bid"][0, :] + sol["delta_ask"][0, :]
    mask = np.isfinite(spread)
    ax.plot(lots[mask], spread[mask], "s-", markersize=8, color="C2")
    ax.set_xlabel("Inventory n (lots)")
    ax.set_ylabel("Spread  δ^b + δ^a  ($/upfront)")
    ax.set_title(f"Fig {fig_offset+4}: {label} — Spread vs inventory")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR /f"fig01_{fig_offset+4}_{label}_spread_vs_n.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig01_{fig_offset+4}")

    # ── Fig X+5 : n → skew(0, n) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    skew = sol["delta_bid"][0, :] - sol["delta_ask"][0, :]
    mask = np.isfinite(skew)
    ax.plot(lots[mask], skew[mask], "D-", markersize=8, color="C3")
    ax.set_xlabel("Inventory n (lots)")
    ax.set_ylabel("Skew  δ^b − δ^a  ($/upfront)")
    ax.set_title(f"Fig {fig_offset+5}: {label} — Skew vs inventory")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR /f"fig01_{fig_offset+5}_{label}_skew_vs_n.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig01_{fig_offset+5}")


# ═══════════════════════════════════════════════════════════════════════
#  Generate all figures
# ═══════════════════════════════════════════════════════════════════════

print("\n=== IG Figures (1–5) ===")
plot_asset(sol_ig, IG, "IG", fig_offset=0)

print("\n=== HY Figures (10–14) ===")
plot_asset(sol_hy, HY, "HY", fig_offset=9)

print("\nDone!")
