"""Monte Carlo simulator for optimal market making (Guéant 2017).

Simulates price paths + Poisson fills using pre‑computed optimal quotes
from the ODE solvers, and records P&L, inventory, and MtM trajectories.

Convention (matching ode_solver_1d / ode_solver_2d):
  δ = absolute distance to mid  (NOT relative)
  S^bid = S − δ^b ,   S^ask = S + δ^a
  Cash update on bid fill:   X ← X − (S − δ^b) · Δ
  Cash update on ask fill:   X ← X + (S + δ^a) · Δ
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.ode_solver_1d import solve_general
from src.intensity import Lambda
from asset.params import IG, GAMMA, T



# ═══════════════════════════════════════════════════════════════════════
#  Single‑asset simulator
# ═══════════════════════════════════════════════════════════════════════

def simulate_1d(sol, params, gamma, T, N_sim=10000, seed=42):
    """Run Monte Carlo trajectories for a single‑asset market maker.

    Parameters
    ----------
    sol : dict from ode_solver_1d (must contain delta_bid, delta_ask, lots, times)
    params : dict with sigma, A, k, Delta, Q
    gamma : float
    T : float — horizon
    N_sim : int — number of trajectories
    seed : int

    Returns
    -------
    dict with keys:
        pnl        : (N_sim,)       — terminal P&L per trajectory
        inventory  : (N_sim, N_t+1) — inventory in lots over time
        cash       : (N_sim, N_t+1) — cash process
        mtm        : (N_sim, N_t+1) — mark‑to‑market  X + n·Δ·S
        price      : (N_sim, N_t+1) — mid price
        n_bid_fills: (N_sim,)       — total bid fills
        n_ask_fills: (N_sim,)       — total ask fills
    """
    rng = np.random.default_rng(seed)

    sigma = params["sigma"]
    A     = params["A"]
    k     = params["k"]
    Delta = params["Delta"]
    Q     = params["Q"]

    times = sol["times"]
    N_t = len(times) - 1
    dt = T / N_t
    lots = sol["lots"]

    # Quote lookup:  delta_bid[t_idx, lot_idx], delta_ask[t_idx, lot_idx]
    db_table = sol["delta_bid"]   # (N_t+1, 2Q+1)
    da_table = sol["delta_ask"]

    # lot index offset: lot n → array index n + Q
    Q_int = int(Q)

    # ── allocate ──
    price     = np.zeros((N_sim, N_t + 1))
    cash      = np.zeros((N_sim, N_t + 1))
    inv_lots  = np.zeros((N_sim, N_t + 1), dtype=int)
    mtm       = np.zeros((N_sim, N_t + 1))
    n_bid_fills = np.zeros(N_sim, dtype=int)
    n_ask_fills = np.zeros(N_sim, dtype=int)

    # ── simulate ──
    dW = rng.standard_normal((N_sim, N_t))
    U_bid = rng.uniform(size=(N_sim, N_t))
    U_ask = rng.uniform(size=(N_sim, N_t))

    for m in range(N_sim):
        S = 0.0
        X = 0.0
        n = 0  # inventory in lots

        price[m, 0] = S
        cash[m, 0] = X
        inv_lots[m, 0] = n
        mtm[m, 0] = X + n * Delta * S

        for t_idx in range(N_t):
            # ── lookup quotes ──
            i_lot = n + Q_int
            db = db_table[t_idx, i_lot]
            da = da_table[t_idx, i_lot]

            can_bid = (n < Q_int) and np.isfinite(db)
            can_ask = (n > -Q_int) and np.isfinite(da)

            # ── price dynamics ──
            dS = sigma * np.sqrt(dt) * dW[m, t_idx] # dSt ​= σ.dWt
            S += dS

            # ── Poisson fills ──
            if can_bid:
                prob_bid = A * np.exp(-k * db) * dt # exp
                if U_bid[m, t_idx] < prob_bid:
                    n += 1
                    X -= (S - db) * Delta
                    n_bid_fills[m] += 1

            if can_ask:
                prob_ask = A * np.exp(-k * da) * dt
                if U_ask[m, t_idx] < prob_ask:
                    n -= 1
                    X += (S + da) * Delta
                    n_ask_fills[m] += 1

            price[m, t_idx + 1] = S
            cash[m, t_idx + 1] = X
            inv_lots[m, t_idx + 1] = n
            mtm[m, t_idx + 1] = X + n * Delta * S

    pnl = cash[:, -1] + inv_lots[:, -1] * Delta * price[:, -1]

    return dict(
        pnl=pnl, inventory=inv_lots, cash=cash,
        mtm=mtm, price=price,
        n_bid_fills=n_bid_fills, n_ask_fills=n_ask_fills,
        times=times, Delta=Delta)


# ═══════════════════════════════════════════════════════════════════════
#  Naive strategy (fixed symmetric spread)
# ═══════════════════════════════════════════════════════════════════════

def simulate_naive(params, gamma, T, half_spread, N_t=7200, N_sim=10000, seed=42):
    """Simulate a naive market maker with constant symmetric spread.

    Parameters
    ----------
    half_spread : float — constant δ (same bid and ask distance)
    """
    rng = np.random.default_rng(seed)

    sigma = params["sigma"]
    A     = params["A"]
    k     = params["k"]
    Delta = params["Delta"]
    Q     = params["Q"]

    dt = T / N_t
    Q_int = int(Q)

    price     = np.zeros((N_sim, N_t + 1))
    cash      = np.zeros((N_sim, N_t + 1))
    inv_lots  = np.zeros((N_sim, N_t + 1), dtype=int)
    mtm       = np.zeros((N_sim, N_t + 1))

    dW = rng.standard_normal((N_sim, N_t))
    U_bid = rng.uniform(size=(N_sim, N_t))
    U_ask = rng.uniform(size=(N_sim, N_t))

    prob_fill = A * np.exp(-k * half_spread) * dt

    for m in range(N_sim):
        S = 0.0
        X = 0.0
        n = 0

        price[m, 0] = S
        cash[m, 0] = X

        for t_idx in range(N_t):
            dS = sigma * np.sqrt(dt) * dW[m, t_idx]
            S += dS

            if n < Q_int and U_bid[m, t_idx] < prob_fill:
                n += 1
                X -= (S - half_spread) * Delta

            if n > -Q_int and U_ask[m, t_idx] < prob_fill:
                n -= 1
                X += (S + half_spread) * Delta

            price[m, t_idx + 1] = S
            cash[m, t_idx + 1] = X
            inv_lots[m, t_idx + 1] = n
            mtm[m, t_idx + 1] = X + n * Delta * S

    pnl = cash[:, -1] + inv_lots[:, -1] * Delta * price[:, -1]

    return dict(
        pnl=pnl, inventory=inv_lots, cash=cash,
        mtm=mtm, price=price, times=np.linspace(0, T, N_t + 1),
        Delta=Delta)


# ═══════════════════════════════════════════════════════════════════════
#  Quick smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    

    print("Solving ODE (Model A, IG) ...")
    sol = solve_general(IG, GAMMA, T, xi=GAMMA, N_t=7200)

    print("Simulating 1000 trajectories ...")
    res = simulate_1d(sol, IG, GAMMA, T, N_sim=1000, seed=0)

    print(f"  Mean P&L       = {np.mean(res['pnl']):.4e}")
    print(f"  Std  P&L       = {np.std(res['pnl']):.4e}")
    print(f"  Mean bid fills = {np.mean(res['n_bid_fills']):.1f}")
    print(f"  Mean ask fills = {np.mean(res['n_ask_fills']):.1f}")
    print(f"  Mean |inv| at T= {np.mean(np.abs(res['inventory'][:, -1])):.2f} lots")
