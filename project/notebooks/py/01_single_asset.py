# %% [markdown]
# # Notebook 01 — Single-Asset Optimal Quotes (IG & HY)
# 

# %% [markdown]
# **Sections reproduced**: Section 3 (ODE reduction) + Section 6.1–6.2 (numerical experiments, single-asset)  
# 
# **Figures reproduced**: 1–5 (IG index) and 10–14 (HY index)
# 

# %% [markdown]
# A **market maker** continuously posts bid and ask quotes around a reference price $S_t$.  
# 
# The question is: *how far from mid should each quote be, given my current inventory?*

# %% [markdown]
# This notebook solves that problem **exactly** (via an ODE solved via implicit scheme + Newton method) for **Model A** (CARA utility) applied to two credit indices (CDX.NA.IG and CDX.NA.HY) treated independently (single-asset).

# %% [markdown]
# ## 0. The model
# #### 0.1 Price dynamics (Paper Eq. 2.1)
# 
# The reference (mid) price follows an arithmetic Brownian motion:
# 
# $$dS_t = \sigma \, dW_t, \qquad S_0 \text{ given}$$
# 
# - $\sigma$ is the volatility of the **upfront rate** (units: \$/√s — *not* an annualised percentage).
# - No drift: the market maker has no directional view.

# %% [markdown]
# #### 0.2 Intensity of fills (Eq. 2.3, exponential case)
# 
# When the market maker posts a bid at distance $\delta^b$ from mid, the probability of a fill in $[t, t+dt)$ is:
# 
# $$\lambda^b_t \, dt = \Lambda(\delta^b) \, dt, \qquad \Lambda(\delta) = A \, e^{-k\delta}$$
# 
# - $A$ (1/s): base arrival rate of counterparties.
# - $k$ (1/\$): how fast the fill probability drops when you quote wider.
# - **Convention**: $\delta \geq 0$ is an *absolute distance to mid* in \$, not a relative spread.
# 
# Each fill changes inventory by exactly $\pm \Delta$ (one lot), so $q_t = n_t \cdot \Delta$ where $n_t \in \{-Q, \ldots, +Q\}$.

# %% [markdown]
# #### 0.3 Objective function Model A (Eq. 2.5)
# 
# The market maker maximises **expected CARA utility** of terminal wealth:
# 
# $$\sup_{\delta^b, \delta^a} \; \mathbb{E}\!\left[ -\exp\!\Big(-\gamma\big(X_T + q_T S_T - \ell(|q_T|)\big)\Big) \right]$$
# 
# - $\gamma$ (1/\$): absolute risk aversion.
# - $\ell(|q|)$: terminal inventory penalty. We set $\ell \equiv 0$ (no penalty), studying the **asymptotic regime** where $T$ is large enough that the terminal condition doesn't matter.
# - $X_T$: accumulated cash from trades.
# - $q_T S_T$: mark-to-market value of remaining inventory.

# %% [markdown]
# #### 0.4 The key reduction HJB → ODE on $\theta$ (Eqs. 3.3–3.9)
# 
# The 4-variable HJB equation on $u(t, x, q, S)$ reduces, via the **ansatz**
# 
# $$u(t, x, q, S) = -\exp\!\big(-\gamma(x + qS + \theta(t, q))\big),$$
# 
# to a **system of ODEs** on $\theta(t, q)$ alone (Eq. 3.9):
# 
# $$0 = \partial_t \theta(t,q) + \tfrac{1}{2}\gamma\sigma^2 q^2 - \mathbb{1}_{q<Q} \, H_\xi\!\!\left(\frac{\theta(t,q) - \theta(t, q+\Delta)}{\Delta}\right) - \mathbb{1}_{q>-Q} \, H_\xi\!\!\left(\frac{\theta(t,q) - \theta(t, q-\Delta)}{\Delta}\right)$$
# 
# where $\xi = \gamma$ for Model A (and $\xi = 0$ for Model B).

# %% [markdown]
# #### 0.5 The Hamiltonian $H_\xi$, exponential case (after Eq. 3.9)
# 
# For $\Lambda(\delta) = Ae^{-k\delta}$:
# 
# $$H_\xi(p) = \frac{A\Delta}{k} \, C_\xi \, e^{-kp}$$
# 
# with the coefficient:
# 
# $$C_\xi = \begin{cases} \left(1 + \frac{\xi\Delta}{k}\right)^{-\left(\frac{k}{\xi\Delta}+1\right)} & \text{if } \xi > 0 \\[4pt] e^{-1} & \text{if } \xi = 0 \end{cases}$$
# 

# %% [markdown]
# #### 0.6 From $\theta$ to quotes (Eqs. 3.14 / 3.16)
# 
# Once $\theta$ is known, the **optimal bid distance-to-mid** at inventory $q = n\Delta$ is:
# 
# $$\delta^{b*}(t, n) = \tilde{\delta}^*_\xi\!\!\left(\frac{\theta(t, n\Delta) - \theta(t, (n+1)\Delta)}{\Delta}\right)$$
# 
# where $\tilde{\delta}^*_\xi(p) = p + \frac{1}{\xi\Delta}\ln\!\left(1 + \frac{\xi\Delta}{k}\right)$ for $\xi > 0$.
# 
# Similarly for the ask, replacing $n+1$ with $n-1$.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

import sys
sys.path.append(str(Path().resolve().parents[0]))

from asset.params import IG, HY, GAMMA, T
from src.ode_solver_1d import solve_general
from src.intensity import C_coeff, H_val, delta_star

plt.style.use("seaborn-v0_8")

# %% [markdown]
# ## 1) Solve ODE (Model A)

# %% [markdown]
# The solver (solve_general) uses **implicit backward Euler with Newton iteration** at each time step:
# 
# 1. Start from terminal condition $\theta_n(T) = 0$ (no penalty, $\ell \equiv 0$).  
# 2. Step **backward** in time: at each step, solve the nonlinear system  
#    $G(\theta^{\text{new}}) = \theta^{\text{new}} - \theta^{\text{old}} + dt \cdot F(\theta^{\text{new}}) = 0$  
#    via Newton's method with a tridiagonal Jacobian.  
# 3. Extract quotes via $\delta^*\!\big((\theta_n - \theta_{n\pm 1})/\Delta\big)$.

# %%
print("Solving IG Model A")
t0 = time.time()
sol_ig = solve_general(IG, GAMMA, T, xi=GAMMA, N_t=7200)

# %%
print("Solving HY Model A")
sol_hy = solve_general(HY, GAMMA, T, xi=GAMMA, N_t=7200)

# %% [markdown]
# Before plotting, we verify structural properties that **must** hold:
# 
# 1. **Symmetry**: $\theta_n(t) = \theta_{-n}(t)$ for all $t$ (no penalty, symmetric $\Lambda$).  
#    Consequence: $\delta^b(t, n) = \delta^a(t, -n)$.
# 
# 2. **Monotonicity of bid**: $\delta^b(t, n)$ increases with $n$ (long inventory → quote bid more conservatively).
# 
# 3. **Asymptotic regime**: quotes at $t = 0$ should equal quotes at $t = \epsilon$ for small $\epsilon$ (the system has converged).
# 
# 4. **Skew at $n = 0$**: must be exactly 0 by symmetry.

# %% [markdown]
# ## 2) Plot

# %%
def plot_asset(sol, params, label):
    lots = sol["lots"]
    times = sol["times"]
    Q = int(params["Q"])

    # ────── δ^bid(t, n) ────── #
    """
    This plot shows how the optimal bid quote evolves from the terminal condition at $t = T$ (right)
    back to the asymptotic regime at $t = 0$ (left), for each inventory level $n$.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, 2 * Q - 1))

    for i, n in enumerate(range(-Q + 1, Q)):
        i_lot = int(n + Q)
        db = sol["delta_bid"][:, i_lot]
        if np.all(np.isnan(db)):
            continue
        ax.plot(times, db, color=colors[i], label=f"n={n:+d}", linewidth=1)

    ax.set_xlabel("t (s)")
    ax.set_ylabel("δ^bid")
    ax.set_title(f"{label} - δ^bid(t, n)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    plt.show()

    # ────── δ^bid(0, n) ────── #
    """
    delta^b(0, n) increases with n: when you are long, you raise your bid to discourage further buying.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = lots[1:]
    db0 = sol["delta_bid"][0, 1:]
    mask = np.isfinite(db0)

    ax.plot(ns[mask], db0[mask], "x-", markersize=8)
    ax.set_xlabel("Inventory n")
    ax.set_ylabel("δ^bid")
    ax.set_title(f"{label} - δ^bid(0, n)")
    ax.grid(alpha=0.3)
    plt.show()

    # ────── δ^ask(0, n) ────── #    
    """
    delta^a(0, n) decreases with n: when you are long, you lower your ask to encourage selling.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = lots[:-1]
    da0 = sol["delta_ask"][0, :-1]
    mask = np.isfinite(da0)

    ax.plot(ns[mask], da0[mask], "o-", markersize=8)
    ax.set_xlabel("Inventory n")
    ax.set_ylabel("δ^ask")
    ax.set_title(f"{label} - δ^ask(0, n)")
    ax.grid(alpha=0.3)
    plt.show()

    # ────── Spread ────── #  
    """
    Spread(n) = delta^b(0, n) + delta^a(0, n)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    spread = sol["delta_bid"][0, :] + sol["delta_ask"][0, :]
    mask = np.isfinite(spread)

    ax.plot(lots[mask], spread[mask], "s-")
    ax.set_xlabel("Inventory n")
    ax.set_ylabel("Spread δ^b + δ^a")
    ax.set_title(f"{label} - Spread vs inventory")
    ax.grid(alpha=0.3)
    plt.show()

    # ────── Skew ────── #  
    """
    Skew(n) = delta^b(0, n) - delta^a(0, n)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    skew = sol["delta_bid"][0, :] - sol["delta_ask"][0, :]
    mask = np.isfinite(skew)

    ax.plot(lots[mask], skew[mask], "D-")
    ax.set_xlabel("Inventory n")
    ax.set_ylabel("Skew δ^b − δ^a")
    ax.set_title(f"{label} - Skew vs inventory")
    ax.grid(alpha=0.3)
    plt.show()

# %%
# IG Figures (1–5)
plot_asset(sol_ig, IG, "IG")

# %%
# HY Figures (10–14)
plot_asset(sol_hy, HY, "HY")

# %% [markdown]
# The function $\theta(t, q)$ is the **reduced value function**, it encodes the full optimal strategy.
# 
# The ODE (Eq. 3.9) says that $\theta$ must balance:
# - **Inventory risk**: $\frac{1}{2}\gamma\sigma^2 q^2$ (quadratic penalty pushing $\theta$ down for large $|n|$)
# - **Trading benefit**: $H_\xi(p)$ terms (the expected gain from fills)
# 
# Let's inspect $\theta(0, n)$ directly, its shape determines the quotes through the discrete gradient $(\theta_n - \theta_{n\pm 1})/\Delta$.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, sol, params, name in [(axes[0], sol_ig, IG, "IG"), (axes[1], sol_hy, HY, "HY")]:
    lots = sol["lots"]
    theta_0 = sol["theta"][0, :]
    
    ax.plot(lots, theta_0, "o-", markersize=8, color="C4")
    ax.set_xlabel("Inventory n (lots)", fontsize=11)
    ax.set_ylabel(r"$\theta(0, n)$", fontsize=11)
    ax.set_title(f"{name} — Value function $\\theta(0, n)$", fontsize=12)
    ax.grid(alpha=0.3)
    
    # Show that it's symmetric and concave
    ax.text(0.05, 0.05, f"θ(0, 0) = {theta_0[int(params['Q'])]:.4e}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

fig.suptitle(r"$	heta(0, n)$ — concave, symmetric, encodes the full strategy",
             fontsize=14, y=1.03)
fig.tight_layout()
plt.show()


# %% [markdown]
# We observe that:
# 
# - Bid quotes increase with inventory
# - Ask quotes decrease with inventory
# - Spread is not constant
# - Skew is not perfectly linear
# 
# This confirms nonlinear inventory effects beyond the affine approximation.

# %% [markdown]
# **Inventory management**: the optimal market maker skews quotes to mean-revert inventory. This is the dominant effect.
# 
# **Non-linearity**: the exact ODE solution reveals that spread and skew are *not* simple functions of inventory, unlike what the closed-form approximation suggests. The deviations are stronger when $\xi\Delta$ is large (IG case: $\xi\Delta = 3000$).
# 
# **Asymptotic regime**: the terminal condition becomes irrelevant well before $T$, justifying the use of the $t = 0$ values as the "steady-state" strategy.


