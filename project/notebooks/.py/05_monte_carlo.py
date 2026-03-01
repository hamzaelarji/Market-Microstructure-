"""Notebook 05 — Monte Carlo validation of the optimal strategy.

Not in the paper — this is an extension for additional insight.

Produces:
  - Fig A: Histogram of terminal P&L (optimal vs naive)
  - Fig B: Sample trajectory (price, quotes, inventory, MtM)
  - Fig C: E[|inventory|] and std[inventory] over time

Run:  python 05_monte_carlo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from asset.params import IG, GAMMA, T
from src.ode_solver_1d import solve_general
from asset.simulator import simulate_1d, simulate_naive
from src.closed_form import approx_quotes
ROOT = Path(__file__).resolve().parents[1]   # PROJECT/
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SIM = 2000   # increase for smoother histograms (slower)

# ═══════════════════════════════════════════════════════════════════════
#  Solve & simulate
# ═══════════════════════════════════════════════════════════════════════

print("Solving ODE (IG, Model A) ...")
sol = solve_general(IG, GAMMA, T, xi=GAMMA, N_t=7200)

print(f"Simulating {N_SIM} optimal trajectories ...")
res_opt = simulate_1d(sol, IG, GAMMA, T, N_sim=N_SIM, seed=42)

# Naive: use the optimal half‑spread at n = 0
half_spread_naive = sol["delta_bid"][0, int(IG["Q"])]  # n=0
print(f"Naive half_spread = {half_spread_naive:.4e}")
print(f"Simulating {N_SIM} naive trajectories ...")
res_naive = simulate_naive(IG, GAMMA, T, half_spread=half_spread_naive,
                           N_t=7200, N_sim=N_SIM, seed=42)

# ═══════════════════════════════════════════════════════════════════════
#  Fig A: P&L histogram
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Fig A: P&L histogram ===")
fig, ax = plt.subplots(figsize=(10, 6))

bins = np.linspace(
    min(res_opt["pnl"].min(), res_naive["pnl"].min()),
    max(res_opt["pnl"].max(), res_naive["pnl"].max()),
    80)

ax.hist(res_opt["pnl"], bins=bins, alpha=0.6, density=True, label="Optimal", color="C0")
ax.hist(res_naive["pnl"], bins=bins, alpha=0.4, density=True, label="Naive (fixed spread)", color="C1")
ax.axvline(np.mean(res_opt["pnl"]), color="C0", ls="--", lw=2,
           label=f"Optimal mean = {np.mean(res_opt['pnl']):.0f}")
ax.axvline(np.mean(res_naive["pnl"]), color="C1", ls="--", lw=2,
           label=f"Naive mean = {np.mean(res_naive['pnl']):.0f}")
ax.set_xlabel("Terminal P&L ($)")
ax.set_ylabel("Density")
ax.set_title("Fig A: P&L distribution — Optimal vs Naive strategy")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig05_A_pnl_histogram.png", dpi=150)
plt.close(fig)

# Print summary stats
for name, res in [("Optimal", res_opt), ("Naive", res_naive)]:
    pnl = res["pnl"]
    print(f"  {name:8s}: mean={np.mean(pnl):+.0f}  std={np.std(pnl):.0f}  "
          f"Sharpe={np.mean(pnl)/np.std(pnl):.3f}  "
          f"5%ile={np.percentile(pnl, 5):.0f}  95%ile={np.percentile(pnl, 95):.0f}")

# ═══════════════════════════════════════════════════════════════════════
#  Fig B: Sample trajectory
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Fig B: Sample trajectory ===")

# Pick a trajectory with moderate activity
idx_med = np.argsort(res_opt["n_bid_fills"] + res_opt["n_ask_fills"])[N_SIM // 2]

times = res_opt["times"]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Price
ax = axes[0]
ax.plot(times, res_opt["price"][idx_med], color="k", lw=0.8)
ax.set_ylabel("Mid price S")
ax.set_title(f"Fig B: Sample trajectory (traj #{idx_med})")
ax.grid(alpha=0.3)

# Inventory
ax = axes[1]
ax.step(times, res_opt["inventory"][idx_med], where="post", color="C0", lw=1.2)
ax.set_ylabel("Inventory (lots)")
ax.axhline(0, color="gray", ls=":", lw=0.8)
ax.grid(alpha=0.3)

# MtM
ax = axes[2]
ax.plot(times, res_opt["mtm"][idx_med], color="C2", lw=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("MtM ($)")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR/"fig05_B_sample_trajectory.png", dpi=150)
plt.close(fig)
print("  Saved fig05_B_sample_trajectory.png")

# ═══════════════════════════════════════════════════════════════════════
#  Fig C: Mean and std of inventory over time
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Fig C: Inventory statistics over time ===")

inv_opt = res_opt["inventory"].astype(float)
inv_naive = res_naive["inventory"].astype(float)

# Subsample for speed in plotting
step = max(1, len(times) // 500)
t_sub = times[::step]
mean_abs_opt = np.mean(np.abs(inv_opt[:, ::step]), axis=0)
std_opt = np.std(inv_opt[:, ::step], axis=0)
mean_abs_naive = np.mean(np.abs(inv_naive[:, ::step]), axis=0)
std_naive = np.std(inv_naive[:, ::step], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(t_sub, mean_abs_opt, label="Optimal", color="C0")
ax.plot(t_sub, mean_abs_naive, label="Naive", color="C1", ls="--")
ax.set_xlabel("Time (s)")
ax.set_ylabel("E[|inventory|] (lots)")
ax.set_title("Mean absolute inventory")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(t_sub, std_opt, label="Optimal", color="C0")
ax.plot(t_sub, std_naive, label="Naive", color="C1", ls="--")
ax.set_xlabel("Time (s)")
ax.set_ylabel("std(inventory) (lots)")
ax.set_title("Inventory volatility")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle("Fig C: Inventory control — Optimal vs Naive", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig05_C_inventory_stats.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved fig05_C_inventory_stats.png")

print("\nDone!")
