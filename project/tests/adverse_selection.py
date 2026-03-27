"""Adverse selection extension — market making with price drift μ.

dS = μdt + σdW.  The "informed" MM adjusts quotes by the
reservation-price shift μ/(γσ²); the "uninformed" MM ignores drift.
"""

import numpy as np
from market_making.core.solver_1d import solve_general
from market_making.core.closed_form import approx_quotes
from market_making.core.intensity import fill_prob


def skew_shift(mu, gamma, sigma):
    """Analytical skew correction for drift: Δr = μ / (γσ²)."""
    denom = gamma * sigma ** 2
    return mu / denom if abs(denom) > 1e-30 else 0.0


def run_drift_backtest(params, gamma, T, mu, N_sim=500, N_t=3600,
                       strategy="uninformed", seed=42, maker_fee=0.0):
    """MC backtest with drifted price.

    strategy: "informed" | "uninformed" | "naive"
    Returns BacktestResult-like dict.
    """
    xi = gamma
    sigma, A, k = params["sigma"], params["A"], params["k"]
    Delta, Q = params["Delta"], int(params["Q"])
    dt = T / N_t
    rng = np.random.default_rng(seed)

    # Quote tables
    sol = solve_general(params, gamma, T, xi=xi, N_t=N_t)
    db_table = sol["delta_bid"]
    da_table = sol["delta_ask"]

    # Informed: shift quotes by drift correction
    shift = skew_shift(mu, gamma, sigma) * Delta
    if strategy == "informed":
        db_table = db_table - shift
        da_table = da_table + shift

    if strategy == "naive":
        half_spread = db_table[0, Q]

    # Allocate
    pnl = np.zeros(N_sim)
    inv_all = np.zeros((N_sim, N_t + 1), dtype=int)
    mtm_all = np.zeros((N_sim, N_t + 1))
    price_all = np.zeros((N_sim, N_t + 1))
    cash_all = np.zeros((N_sim, N_t + 1))
    n_bid_fills = np.zeros(N_sim, dtype=int)
    n_ask_fills = np.zeros(N_sim, dtype=int)
    fees_paid = np.zeros(N_sim)

    dW = rng.standard_normal((N_sim, N_t))
    U_bid = rng.uniform(size=(N_sim, N_t))
    U_ask = rng.uniform(size=(N_sim, N_t))

    for m in range(N_sim):
        S, X, n = 0.0, 0.0, 0
        for t_idx in range(N_t):
            i_lot = n + Q
            if strategy == "naive":
                db = half_spread if n < Q else np.inf
                da = half_spread if n > -Q else np.inf
            else:
                db = db_table[t_idx, i_lot] if (n < Q and i_lot < db_table.shape[1]
                     and np.isfinite(db_table[t_idx, i_lot])) else np.inf
                da = da_table[t_idx, i_lot] if (n > -Q and i_lot >= 0
                     and np.isfinite(da_table[t_idx, i_lot])) else np.inf

            S += mu * dt + sigma * np.sqrt(dt) * dW[m, t_idx]

            if db < np.inf and db > 0:
                lam = A * np.exp(-k * db)
                if U_bid[m, t_idx] < fill_prob(lam, dt):
                    fee = maker_fee * abs(S - db) * Delta
                    X -= (S - db) * Delta + fee
                    n += 1
                    n_bid_fills[m] += 1
                    fees_paid[m] += fee

            if da < np.inf and da > 0:
                lam = A * np.exp(-k * da)
                if U_ask[m, t_idx] < fill_prob(lam, dt):
                    fee = maker_fee * abs(S + da) * Delta
                    X += (S + da) * Delta - fee
                    n -= 1
                    n_ask_fills[m] += 1
                    fees_paid[m] += fee

            price_all[m, t_idx + 1] = S
            cash_all[m, t_idx + 1] = X
            inv_all[m, t_idx + 1] = n
            mtm_all[m, t_idx + 1] = X + n * Delta * S

        pnl[m] = X + n * Delta * S

    times = np.linspace(0, T, N_t + 1)
    std = float(np.std(pnl))
    return {
        "pnl": pnl,
        "inventory": inv_all,
        "mtm": mtm_all,
        "price": price_all,
        "times": times,
        "n_bid_fills": n_bid_fills,
        "n_ask_fills": n_ask_fills,
        "fees_paid": fees_paid,
        "strategy": strategy,
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": std,
        "sharpe": float(np.mean(pnl) / max(std, 1e-12)),
        "mean_fills": float(np.mean(n_bid_fills + n_ask_fills)),
        "mean_abs_inv": float(np.mean(np.abs(inv_all[:, -1]))),
    }
