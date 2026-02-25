"""Notebook 06 — Solver diagnostics plots.

Visual diagnostics for the recently added solver consistency checks:
1) 1D Model A: specialised solver vs general solver quote differences.
2) 2D solver: quote surfaces and finite/NaN boundary masks.

Run:  python 06_solver_diagnostics.py
"""

import sys
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from asset.params import GAMMA, HY, IG
from src.ode_solver_1d import solve_general, solve_model_a
from src.ode_solver_2d import solve_2d

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_1d_model_a_diagnostics():
    """Compare Model A specialised solver and general Newton solver."""
    T_test = 600.0
    N_t = 240

    print("Solving 1D Model A (specialised and general) ...")
    sol_a = solve_model_a(IG, GAMMA, T_test, N_t=N_t)
    sol_g = solve_general(IG, GAMMA, T_test, xi=GAMMA, N_t=N_t)

    lots = sol_a["lots"]
    times = sol_a["times"]

    db_diff = sol_a["delta_bid"] - sol_g["delta_bid"]
    da_diff = sol_a["delta_ask"] - sol_g["delta_ask"]

    # Plot 1: t=0 cross-section over inventory.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    mask = np.isfinite(sol_a["delta_bid"][0]) & np.isfinite(sol_g["delta_bid"][0])
    ax.plot(lots[mask], db_diff[0, mask], "o-", lw=1.2)
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_title("Model A bid diff at t=0")
    ax.set_xlabel("Inventory n (lots)")
    ax.set_ylabel("delta_bid_special - delta_bid_general")
    ax.grid(alpha=0.3)

    ax = axes[1]
    mask = np.isfinite(sol_a["delta_ask"][0]) & np.isfinite(sol_g["delta_ask"][0])
    ax.plot(lots[mask], da_diff[0, mask], "o-", lw=1.2, color="C1")
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_title("Model A ask diff at t=0")
    ax.set_xlabel("Inventory n (lots)")
    ax.set_ylabel("delta_ask_special - delta_ask_general")
    ax.grid(alpha=0.3)

    fig.suptitle("1D solver consistency at t=0", y=1.02)
    fig.tight_layout()
    f1 = FIG_DIR / "fig06_1_1d_model_a_t0_differences.png"
    fig.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {f1.name}")

    # Plot 2: absolute difference heatmaps (time x inventory).
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    bid_abs = np.where(np.isfinite(db_diff), np.abs(db_diff), np.nan)
    ask_abs = np.where(np.isfinite(da_diff), np.abs(da_diff), np.nan)

    im = axes[0].imshow(
        bid_abs,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[lots[0], lots[-1], times[0], times[-1]],
    )
    axes[0].set_title("abs bid diff")
    axes[0].set_xlabel("Inventory n (lots)")
    axes[0].set_ylabel("Time (s)")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    im = axes[1].imshow(
        ask_abs,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[lots[0], lots[-1], times[0], times[-1]],
    )
    axes[1].set_title("abs ask diff")
    axes[1].set_xlabel("Inventory n (lots)")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("1D solver absolute quote differences (specialised vs general)", y=1.02)
    fig.tight_layout()
    f2 = FIG_DIR / "fig06_2_1d_model_a_diff_heatmaps.png"
    fig.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {f2.name}")


def _grid_to_matrix(sol, key, n1_vals, n2_vals):
    z = np.full((len(n1_vals), len(n2_vals)), np.nan)
    for i, n1 in enumerate(n1_vals):
        for j, n2 in enumerate(n2_vals):
            idx = sol["idx"].get((int(n1), int(n2)))
            if idx is not None:
                z[i, j] = sol[key][0, idx]
    return z


def plot_2d_boundary_diagnostics():
    """Plot t=0 quote surfaces and finite masks to verify boundaries."""
    p1 = {**IG, "Q": 2}
    p2 = {**HY, "Q": 2}
    T_test = 300.0
    N_t = 60

    print("Solving 2D Model A (diagnostic grid) ...")
    sol = solve_2d(p1, p2, GAMMA, rho=0.3, T=T_test, xi=GAMMA, N_t=N_t)

    n1_vals = np.arange(-p1["Q"], p1["Q"] + 1)
    n2_vals = np.arange(-p2["Q"], p2["Q"] + 1)

    db1 = _grid_to_matrix(sol, "delta_bid_1", n1_vals, n2_vals)
    da1 = _grid_to_matrix(sol, "delta_ask_1", n1_vals, n2_vals)
    db2 = _grid_to_matrix(sol, "delta_bid_2", n1_vals, n2_vals)
    da2 = _grid_to_matrix(sol, "delta_ask_2", n1_vals, n2_vals)

    # Plot 3: quote values at t=0.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fields = [
        ("delta_bid_1", db1),
        ("delta_ask_1", da1),
        ("delta_bid_2", db2),
        ("delta_ask_2", da2),
    ]
    for ax, (title, z) in zip(axes.ravel(), fields):
        im = ax.imshow(
            z,
            origin="lower",
            interpolation="nearest",
            extent=[n2_vals[0], n2_vals[-1], n1_vals[0], n1_vals[-1]],
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("n2 (HY lots)")
        ax.set_ylabel("n1 (IG lots)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("2D quotes at t=0 (NaN on boundaries where quote is not admissible)", y=1.02)
    fig.tight_layout()
    f3 = FIG_DIR / "fig06_3_2d_quote_surfaces_t0.png"
    fig.savefig(f3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {f3.name}")

    # Plot 4: finite/NaN masks.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, (title, z) in zip(axes.ravel(), fields):
        mask = np.isfinite(z).astype(float)
        im = ax.imshow(
            mask,
            origin="lower",
            interpolation="nearest",
            extent=[n2_vals[0], n2_vals[-1], n1_vals[0], n1_vals[-1]],
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        ax.set_title(f"{title} finite mask")
        ax.set_xlabel("n2 (HY lots)")
        ax.set_ylabel("n1 (IG lots)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("2D finite masks at t=0 (1=finite quote, 0=NaN boundary)", y=1.02)
    fig.tight_layout()
    f4 = FIG_DIR / "fig06_4_2d_finite_masks_t0.png"
    fig.savefig(f4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {f4.name}")


if __name__ == "__main__":
    plot_1d_model_a_diagnostics()
    plot_2d_boundary_diagnostics()
    print("Done.")
