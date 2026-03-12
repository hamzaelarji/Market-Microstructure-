# %% [markdown]
# # Notebook 11 — Intraday Regime Adaptation
#
# **Extension**: Market conditions change intraday. Compare a **static** policy
# (single ODE solve with constant σ, A) vs a **regime-aware** policy that
# re-solves the ODE at each regime boundary.
#
# ## Regime structure
# We split the trading day into 3 periods:
# - **Open** (0–30%): high activity, high volatility
# - **Noon** (30–70%): low activity, low volatility
# - **Close** (70–100%): high activity, high volatility

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parents[0]))

from market_making.core.solver_1d import solve_general
from market_making.core.intensity import fill_prob

plt.style.use("seaborn-v0_8")

# %%
# Parameters — use BTC-like calibrated params
params = {"sigma": 5.76, "A": 5.55, "k": 2.73, "Delta": 91.86, "Q": 4}
gamma = 0.01
T = 3600.0
N_t = 3600
dt = T / N_t
Q = int(params["Q"])
n_states = 2 * Q + 1

# Regime multipliers
regimes = [
    {"name": "Open",  "A_mult": 1.45, "sigma_mult": 1.50},
    {"name": "Noon",  "A_mult": 0.60, "sigma_mult": 0.55},
    {"name": "Close", "A_mult": 1.35, "sigma_mult": 1.40},
]
breaks = [0, int(0.3 * N_t), int(0.7 * N_t), N_t]

# %% [markdown]
# ## Build time-varying A(t) and σ(t) paths

# %%
A_path = np.zeros(N_t)
sigma_path = np.zeros(N_t)
for ri, (s, e) in enumerate(zip(breaks[:-1], breaks[1:])):
    A_path[s:e] = params["A"] * regimes[ri]["A_mult"]
    sigma_path[s:e] = params["sigma"] * regimes[ri]["sigma_mult"]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
times_plot = np.linspace(0, T, N_t)
for ax, y, label in [(axes[0], A_path, "A(t)"), (axes[1], sigma_path, "σ(t)")]:
    ax.plot(times_plot, y, linewidth=2, color="teal")
    for b in breaks[1:-1]:
        ax.axvline(x=times_plot[b], linestyle=":", color="gray", alpha=0.7)
    ax.set_xlabel("t (s)"); ax.set_ylabel(label); ax.grid(alpha=0.3)
fig.suptitle("Intraday regime structure")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Solve: static vs regime-aware policies

# %%
# Static: single ODE solve with base params
print("Solving static policy...")
sol_static = solve_general(params, gamma, T, xi=gamma, N_t=N_t)
db_static = sol_static["delta_bid"]
da_static = sol_static["delta_ask"]

# Regime-aware: re-solve at each boundary
print("Solving regime-aware policy...")
db_regime = np.full((N_t + 1, n_states), np.nan)
da_regime = np.full((N_t + 1, n_states), np.nan)

for ri, (s, e) in enumerate(zip(breaks[:-1], breaks[1:])):
    rem = N_t - s
    rem_T = T * rem / N_t
    p_reg = {
        **params,
        "A": params["A"] * regimes[ri]["A_mult"],
        "sigma": params["sigma"] * regimes[ri]["sigma_mult"],
    }
    sol_reg = solve_general(p_reg, gamma, rem_T, xi=gamma, N_t=rem)
    seg = e - s
    db_regime[s:e, :] = sol_reg["delta_bid"][:seg, :]
    da_regime[s:e, :] = sol_reg["delta_ask"][:seg, :]

db_regime[-1, :] = db_regime[-2, :]
da_regime[-1, :] = da_regime[-2, :]
print("Done.")

# %% [markdown]
# ## Simulate both policies

# %%
def simulate(db_tbl, da_tbl, N_sim=1000, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((N_sim, N_t))
    u_b = rng.uniform(size=(N_sim, N_t))
    u_a = rng.uniform(size=(N_sim, N_t))

    pnl = np.zeros(N_sim)
    inv_all = np.zeros((N_sim, N_t + 1), dtype=int)
    mtm_all = np.zeros((N_sim, N_t + 1))

    for m in range(N_sim):
        S, X, n = 0.0, 0.0, 0
        for t in range(N_t):
            il = n + Q
            if not (0 <= il < n_states):
                inv_all[m, t+1] = n
                mtm_all[m, t+1] = X + n * params["Delta"] * S
                continue

            db_v = db_tbl[t, il]
            da_v = da_tbl[t, il]
            S += sigma_path[t] * np.sqrt(dt) * z[m, t]

            if n < Q and np.isfinite(db_v) and db_v > 0:
                lam = A_path[t] * np.exp(-params["k"] * db_v)
                if u_b[m, t] < fill_prob(lam, dt):
                    X -= (S - db_v) * params["Delta"]; n += 1
            if n > -Q and np.isfinite(da_v) and da_v > 0:
                lam = A_path[t] * np.exp(-params["k"] * da_v)
                if u_a[m, t] < fill_prob(lam, dt):
                    X += (S + da_v) * params["Delta"]; n -= 1

            inv_all[m, t+1] = n
            mtm_all[m, t+1] = X + n * params["Delta"] * S

        pnl[m] = X + n * params["Delta"] * S

    return pnl, inv_all, mtm_all

N_SIM = 1000
print(f"Simulating {N_SIM} paths...")
pnl_static, inv_static, mtm_static = simulate(db_static, da_static, N_SIM)
pnl_regime, inv_regime, mtm_regime = simulate(db_regime, da_regime, N_SIM)
print("Done.")

# %% [markdown]
# ## Results

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PnL histogram
axes[0].hist(pnl_static, bins=50, alpha=0.6, label=f"Static (Sharpe={np.mean(pnl_static)/max(np.std(pnl_static),1e-12):.3f})")
axes[0].hist(pnl_regime, bins=50, alpha=0.6, label=f"Regime (Sharpe={np.mean(pnl_regime)/max(np.std(pnl_regime),1e-12):.3f})")
axes[0].set_xlabel("PnL ($)")
axes[0].legend()
axes[0].set_title("PnL Distribution")
axes[0].grid(alpha=0.3)

# Spread path
times_arr = np.linspace(0, T, N_t + 1)
spread_static = db_static[:, Q] + da_static[:, Q]
spread_regime = db_regime[:, Q] + da_regime[:, Q]
axes[1].plot(times_arr, spread_static, label="Static", linewidth=1.5)
axes[1].plot(times_arr, spread_regime, label="Regime-aware", linewidth=1.5)
for b in breaks[1:-1]:
    axes[1].axvline(x=times_arr[b], linestyle=":", color="gray", alpha=0.7)
axes[1].set_xlabel("t (s)")
axes[1].set_ylabel("Spread at n=0")
axes[1].set_title("Spread Path")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Inventory control
axes[2].plot(times_arr, np.mean(np.abs(inv_static), axis=0), label="Static", linewidth=1.5)
axes[2].plot(times_arr, np.mean(np.abs(inv_regime), axis=0), label="Regime-aware", linewidth=1.5)
axes[2].set_xlabel("t (s)")
axes[2].set_ylabel("E[|inventory|]")
axes[2].set_title("Inventory Control")
axes[2].legend()
axes[2].grid(alpha=0.3)

fig.tight_layout()
plt.show()

# %%
print("=== SUMMARY ===")
for name, pnl in [("Static", pnl_static), ("Regime-aware", pnl_regime)]:
    s = np.std(pnl)
    print(f"{name:15s}  E[PnL]={np.mean(pnl):8.2f}  Std={s:8.2f}  "
          f"Sharpe={np.mean(pnl)/max(s,1e-12):6.3f}")

# %% [markdown]
# ### Key takeaways
#
# 1. The regime-aware policy adapts its spread to current market conditions:
#    wider during high-vol periods, tighter during quiet periods.
# 2. The improvement is modest when regimes are mild (±50% multipliers)
#    but becomes significant with more extreme regime shifts.
# 3. This motivates adaptive models (Hawkes, RL) that can continuously
#    adjust to market microstructure changes.
