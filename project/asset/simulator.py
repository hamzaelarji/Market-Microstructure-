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

def simulate_1d(
    sol,
    params,
    gamma,
    T,
    N_sim=10000,
    seed=42,
    ell=None,                # ell: callable taking abs(q_dollars) -> penalty in dollars
    return_utility=True,     # compute CARA utility at terminal time
    model="A",               # "A" or "B" just for bookkeeping (utility same formula if you want)
):
    """
    Faithful Monte Carlo for Guéant (2017):
      - Poisson arrivals: N ~ Poisson(lambda * dt)
      - Trades executed at S_t (quotes posted around S_t on [t, t+dt])
      - Then price evolves: S_{t+dt} = S_t + sigma * sqrt(dt) * Z
      - Terminal liquidation penalty: ell(|q_T|) included in terminal wealth if provided
      - Utility: -exp(-gamma * (X_T + q_T S_T - ell(|q_T|))) if return_utility=True
    """

    rng = np.random.default_rng(seed)

    sigma = params["sigma"]
    A     = params["A"]
    k     = params["k"]
    Delta = params["Delta"]
    Q     = int(params["Q"])

    times = sol["times"]
    N_t = len(times) - 1
    dt = T / N_t

    db_table = sol["delta_bid"]  # (N_t+1, 2Q+1)
    da_table = sol["delta_ask"]  # (N_t+1, 2Q+1)

    # Storage
    price     = np.zeros((N_sim, N_t + 1))
    cash      = np.zeros((N_sim, N_t + 1))
    inv_lots  = np.zeros((N_sim, N_t + 1), dtype=int)
    mtm       = np.zeros((N_sim, N_t + 1))

    n_bid_fills = np.zeros(N_sim, dtype=int)
    n_ask_fills = np.zeros(N_sim, dtype=int)

    # Simulate Z for Brownian increments
    Z = rng.standard_normal((N_sim, N_t))

    # Default liquidation penalty: 0
    if ell is None:
        def ell(q_abs_dollars: float) -> float:
            return 0.0

    for m in range(N_sim):
        S = 0.0
        X = 0.0
        n = 0  # lots

        price[m, 0] = S
        cash[m, 0] = X
        inv_lots[m, 0] = n
        mtm[m, 0] = X + (n * Delta) * S

        for t_idx in range(N_t):
            # Quotes are posted based on current inventory n and current mid S_t
            i_lot = n + Q
            db = db_table[t_idx, i_lot]
            da = da_table[t_idx, i_lot]

            can_bid = (n < Q) and np.isfinite(db)
            can_ask = (n > -Q) and np.isfinite(da)

            # --- Poisson fills over [t, t+dt] using intensities at posted quotes ---
            # lambda = A * exp(-k * delta)
            # N ~ Poisson(lambda * dt)
            if can_bid:
                lam_bid = A * np.exp(-k * db)
                N_bid = rng.poisson(lam=lam_bid * dt)
                if N_bid > 0:
                    # Cap by inventory limit
                    N_exec = min(N_bid, Q - n)
                    # Execute buys at bid price based on S_t (NOT S_{t+dt})
                    X -= N_exec * (S - db) * Delta
                    n += N_exec
                    n_bid_fills[m] += N_exec

            if can_ask:
                lam_ask = A * np.exp(-k * da)
                N_ask = rng.poisson(lam=lam_ask * dt)
                if N_ask > 0:
                    N_exec = min(N_ask, n - (-Q))
                    # Execute sells at ask price based on S_t
                    X += N_exec * (S + da) * Delta
                    n -= N_exec
                    n_ask_fills[m] += N_exec

            # --- Price evolves AFTER trades (discretization consistent with quotes at S_t) ---
            S += sigma * np.sqrt(dt) * Z[m, t_idx]

            price[m, t_idx + 1] = S
            cash[m, t_idx + 1] = X
            inv_lots[m, t_idx + 1] = n
            mtm[m, t_idx + 1] = X + (n * Delta) * S

    # Terminal quantities
    qT_dollars = inv_lots[:, -1] * Delta
    ST = price[:, -1]
    XT = cash[:, -1]

    liquidation_penalty = np.array([ell(abs(q)) for q in qT_dollars], dtype=float)

    # "Paper-consistent" terminal wealth (MtM minus liquidation penalty)
    W_T = XT + qT_dollars * ST - liquidation_penalty

    # You can still report raw MtM PnL if you want
    pnl_mtm = XT + qT_dollars * ST
    pnl_paper = W_T

    out = dict(
        pnl_mtm=pnl_mtm,
        pnl_paper=pnl_paper,
        inventory=inv_lots,
        cash=cash,
        mtm=mtm,
        price=price,
        n_bid_fills=n_bid_fills,
        n_ask_fills=n_ask_fills,
        times=times,
        Delta=Delta,
        liquidation_penalty=liquidation_penalty,
        terminal_wealth=W_T,
    )

    if return_utility:
        # CARA utility at terminal time (same functional form as paper)
        out["utility"] = -np.exp(-gamma * W_T)

    return out


# ═══════════════════════════════════════════════════════════════════════
#  Naive strategy (fixed symmetric spread)
# ═══════════════════════════════════════════════════════════════════════

def simulate_naive(
    params,
    gamma,
    T,
    half_spread,
    N_t=7200,
    N_sim=10000,
    seed=42,
    ell=None,
    return_utility=True,
):
    """Simulate a naive market maker with constant symmetric spread.

    Returns dict with keys matching simulate_1d interface:
        pnl, inventory, cash, mtm, price, times, Delta,
        n_bid_fills, n_ask_fills
    Plus extras: pnl_paper, liquidation_penalty, terminal_wealth, utility.
    """
    rng = np.random.default_rng(seed)

    sigma = params["sigma"]
    A     = params["A"]
    k     = params["k"]
    Delta = params["Delta"]
    Q     = int(params["Q"])

    dt = T / N_t
    times = np.linspace(0, T, N_t + 1)

    price       = np.zeros((N_sim, N_t + 1))
    cash        = np.zeros((N_sim, N_t + 1))
    inv_lots    = np.zeros((N_sim, N_t + 1), dtype=int)
    mtm         = np.zeros((N_sim, N_t + 1))
    n_bid_fills = np.zeros(N_sim, dtype=int)
    n_ask_fills = np.zeros(N_sim, dtype=int)

    Z = rng.standard_normal((N_sim, N_t))

    if ell is None:
        def ell(q_abs_dollars: float) -> float:
            return 0.0

    lam = A * np.exp(-k * half_spread)  # same on both sides

    for m in range(N_sim):
        S = 0.0
        X = 0.0
        n = 0

        price[m, 0] = S
        cash[m, 0] = X
        inv_lots[m, 0] = n
        mtm[m, 0] = X + (n * Delta) * S

        for t_idx in range(N_t):
            # Bid arrivals
            N_bid = rng.poisson(lam=lam * dt) if n < Q else 0
            if N_bid > 0:
                N_exec = min(N_bid, Q - n)
                X -= N_exec * (S - half_spread) * Delta
                n += N_exec
                n_bid_fills[m] += N_exec

            # Ask arrivals
            N_ask = rng.poisson(lam=lam * dt) if n > -Q else 0
            if N_ask > 0:
                N_exec = min(N_ask, n - (-Q))
                X += N_exec * (S + half_spread) * Delta
                n -= N_exec
                n_ask_fills[m] += N_exec

            # Price evolves after trades
            S += sigma * np.sqrt(dt) * Z[m, t_idx]

            price[m, t_idx + 1] = S
            cash[m, t_idx + 1] = X
            inv_lots[m, t_idx + 1] = n
            mtm[m, t_idx + 1] = X + (n * Delta) * S

    # ── Terminal P&L ──
    qT_dollars = inv_lots[:, -1].astype(float) * Delta
    ST = price[:, -1]
    XT = cash[:, -1]
    pnl_mtm = XT + qT_dollars * ST

    liquidation_penalty = np.array([ell(abs(q)) for q in qT_dollars], dtype=float)
    W_T = pnl_mtm - liquidation_penalty

    out = dict(
        pnl=pnl_mtm,                
        n_bid_fills=n_bid_fills,     
        n_ask_fills=n_ask_fills,     
        inventory=inv_lots,
        cash=cash,
        mtm=mtm,
        price=price,
        times=times,
        Delta=Delta,
        pnl_mtm=pnl_mtm,
        pnl_paper=W_T,
        liquidation_penalty=liquidation_penalty,
        terminal_wealth=W_T,
    )

    if return_utility:
        out["utility"] = -np.exp(-gamma * W_T)

    return out


# ═══════════════════════════════════════════════════════════════════════
#  Quick smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("Solving ODE (Model A, IG) ...")
    sol = solve_general(IG, GAMMA, T, xi=GAMMA, N_t=7200) # * IG["Delta"]

    # Example liquidation penalty (choose and calibrate)
    #ℓ(∣q∣)=η∣q∣
    ETA = 1.0e-4
    ell = lambda q_abs: ETA * q_abs
    
    #ℓ(∣q∣)=η∣q∣^2
    #ETA = 1.0e-12  # exemple (choose and calibrate)
    #ell = lambda q_abs: ETA * (q_abs**2)

    print("Simulating 1000 trajectories (paper-faithful) ...")
    res = simulate_1d_paper(sol, IG, GAMMA, T, N_sim=1000, seed=0, ell=ell)

    print(f"  Mean MtM PnL        = {np.mean(res['pnl_mtm']):.4e}")
    print(f"  Mean paper PnL      = {np.mean(res['pnl_paper']):.4e}")
    print(f"  Mean liquidation    = {np.mean(res['liquidation_penalty']):.4e}")
    print(f"  Mean utility        = {np.mean(res['utility']):.4e}")
    print(f"  Mean bid fills      = {np.mean(res['n_bid_fills']):.1f} lots")
    print(f"  Mean ask fills      = {np.mean(res['n_ask_fills']):.1f} lots")
    print(f"  Mean |inv| at T     = {np.mean(np.abs(res['inventory'][:, -1])):.2f} lots")
