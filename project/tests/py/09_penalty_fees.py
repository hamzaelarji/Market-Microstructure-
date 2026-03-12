# %% [markdown]
# # Notebook 09 — Terminal Penalty & Transaction Costs
#
# **Paper**: Guéant (2017), §2–3 (terminal condition) + practical extension.
#
# This notebook explores two features the paper discusses but does not numerically examine:
# 1. **Terminal penalty ℓ(|q|)** — how θ(T, n) = −ℓ(|nΔ|) changes quotes near T.
# 2. **Maker fees** — impact on strategy profitability, break-even fee level.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parents[0]))

from market_making.core.solver_1d import solve_general
from market_making.core.intensity import C_coeff
from market_making.simulation.backtest import BacktestConfig, run_backtest

plt.style.use("seaborn-v0_8")

# %% [markdown]
# ## Part A — Terminal Penalty ℓ(|q|)
#
# The paper's terminal condition is θ(T, n) = −ℓ(|nΔ|).
# In all numerical experiments, the paper uses ℓ ≡ 0 (no penalty).
# Here we explore what happens with non-zero penalty.
#
# **Penalty types**:
# - Quadratic: ℓ(|q|) = c · q²  (aggressive unwinding at T)
# - Linear: ℓ(|q|) = c · |q|    (proportional penalty)

# %%
# Parameters (use CDX.NA.IG for consistency with the paper)
from market_making.params.assets import IG, HY, GAMMA, T

params = IG
gamma = GAMMA
T_val = 7200
N_t = 3600

# %%
# Solve with different penalties
def ell_quad(c):
    return lambda q_abs: c * q_abs ** 2

def ell_linear(c):
    return lambda q_abs: c * q_abs

sol_no_pen = solve_general(params, gamma, T_val, xi=gamma, N_t=N_t)
sol_quad = solve_general(params, gamma, T_val, xi=gamma, N_t=N_t,
                         ell_func=ell_quad(1e-4))
sol_lin = solve_general(params, gamma, T_val, xi=gamma, N_t=N_t,
                        ell_func=ell_linear(1e-4))

# %% [markdown]
# ### δ^bid(t, n=0) — convergence with and without penalty

# %%
Q = int(params["Q"])
times = sol_no_pen["times"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times, sol_no_pen["delta_bid"][:, Q], label="ℓ ≡ 0", linewidth=2)
ax.plot(times, sol_quad["delta_bid"][:, Q], label="ℓ = cq² (c=1e-4)", linewidth=2, linestyle="--")
ax.plot(times, sol_lin["delta_bid"][:, Q], label="ℓ = c|q| (c=1e-4)", linewidth=2, linestyle=":")
ax.set_xlabel("t (s)")
ax.set_ylabel("δ^bid(t, n=0)")
ax.set_title("Convergence: penalty effect vanishes far from T")
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ### Spread(0, n) comparison

# %%
lots = sol_no_pen["lots"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_label, get_y in [
    (axes[0], "Spread(0, n)", lambda s: s["delta_bid"][0, :] + s["delta_ask"][0, :]),
    (axes[1], "Skew(0, n)", lambda s: s["delta_bid"][0, :] - s["delta_ask"][0, :]),
]:
    for sol, label, ls in [(sol_no_pen, "ℓ ≡ 0", "-"), (sol_quad, "Quadratic", "--"), (sol_lin, "Linear", ":")]:
        y = get_y(sol)
        mask = np.isfinite(y)
        ax.plot(lots[mask], y[mask], ls, label=label, linewidth=2)
    ax.set_xlabel("n (lots)")
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(alpha=0.3)

fig.suptitle("At t=0, penalty barely changes quotes (asymptotic regime)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Penalty effect vs time horizon T

# %%
T_values = [600, 1200, 1800, 3600, 7200, 14400]
errors = []
for T_test in T_values:
    nt = max(300, T_test)
    s_no = solve_general(params, gamma, float(T_test), xi=gamma, N_t=nt)
    s_pen = solve_general(params, gamma, float(T_test), xi=gamma, N_t=nt,
                          ell_func=ell_quad(1e-4))
    db_no = s_no["delta_bid"][0, :]
    db_pen = s_pen["delta_bid"][0, :]
    m = np.isfinite(db_no) & np.isfinite(db_pen)
    errors.append(np.max(np.abs(db_no[m] - db_pen[m])) if m.any() else 0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(T_values, errors, "o-", linewidth=2, markersize=8)
ax.set_xlabel("T (s)")
ax.set_ylabel("max |δ(ℓ≠0) − δ(ℓ=0)| at t=0")
ax.set_title("Penalty effect vanishes exponentially with T")
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## Part B — Transaction Costs (Maker Fees)

# %%
# Fee sweep
fee_bps_range = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
from market_making.params.assets import IG
params_btc = {"sigma": 5.76, "A": 5.55, "k": 2.73, "Delta": 91.86, "Q": 4}  # fallback BTC

sharpes = []
mean_pnls = []
mean_fees = []

for fee_bps in fee_bps_range:
    cfg = BacktestConfig(params=params_btc, gamma=0.01, T=3600.0,
                         N_sim=500, seed=42, strategy="optimal",
                         maker_fee=fee_bps / 10000.0)
    res = run_backtest(cfg)
    sharpes.append(res.sharpe)
    mean_pnls.append(res.mean_pnl)
    mean_fees.append(float(np.mean(res.fees_paid)))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(fee_bps_range, sharpes, "o-", linewidth=2, markersize=8, color="teal")
axes[0].axhline(y=0, linestyle=":", color="gray", alpha=0.7)
axes[0].set_xlabel("Maker fee (bps)")
axes[0].set_ylabel("Sharpe ratio")
axes[0].set_title("Sharpe vs Maker Fee")
axes[0].grid(alpha=0.3)

gross = np.array(mean_pnls) + np.array(mean_fees)
axes[1].bar(range(len(fee_bps_range)), gross, label="Gross PnL", color="teal", alpha=0.7)
axes[1].bar(range(len(fee_bps_range)), -np.array(mean_fees), label="Fees", color="salmon", alpha=0.7)
axes[1].plot(range(len(fee_bps_range)), mean_pnls, "ko-", label="Net PnL", markersize=6)
axes[1].set_xticks(range(len(fee_bps_range)))
axes[1].set_xticklabels([f"{f:.1f}" for f in fee_bps_range])
axes[1].set_xlabel("Maker fee (bps)")
axes[1].set_ylabel("Mean PnL ($)")
axes[1].set_title("PnL Decomposition")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.show()

# %% [markdown]
# ### Key takeaways
#
# 1. **Terminal penalty** — at t = 0 (far from T), the penalty barely changes quotes.
#    The asymptotic regime dominates, justifying the paper's choice of ℓ ≡ 0.
#    The penalty only matters near T.
#
# 2. **Maker fees** — higher fees eat into spread profit. Break-even is typically
#    around 3–5 bps for BTC-scale parameters. Exchanges with negative maker fees
#    (rebates) would significantly boost performance.
