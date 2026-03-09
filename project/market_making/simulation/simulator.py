"""Monte Carlo simulator for optimal market making (Guéant 2017).

Simulates price paths + Poisson/Hawkes fills using pre‑computed optimal quotes
from the ODE solvers, and records P&L, inventory, and MtM trajectories.

Convention (matching solver_1d / solver_2d):
  δ = absolute distance to mid  (NOT relative)
  S^bid = S − δ^b ,   S^ask = S + δ^a
  Cash update on bid fill:   X ← X − (S − δ^b) · Δ
  Cash update on ask fill:   X ← X + (S + δ^a) · Δ
"""
from __future__ import annotations
import numpy as np
from market_making.core.intensity import Lambda, fill_prob
from market_making.core.hawkes import HawkesState


# ═══════════════════════════════════════════════════════════════════════════════
#  Single‑asset simulator (Poisson fills)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_1d(
    sol,
    params,
    gamma,
    T,
    N_sim=10000,
    seed=42,
    ell=None,
    return_utility=True,
):
    """Monte Carlo for Guéant (2017) with Poisson fills.

    Parameters
    ----------
    sol : dict — output of solve_general (delta_bid, delta_ask, times)
    params : dict — sigma, A, k, Delta, Q
    gamma : float — risk aversion (for CARA utility)
    T : float — trading horizon
    N_sim : int — number of Monte Carlo paths
    seed : int — RNG seed
    ell : callable(q_abs_dollars) → penalty, or None
    return_utility : bool — include CARA utility in output

    Returns
    -------
    dict with pnl_mtm, pnl_paper, inventory, cash, mtm, price,
         n_bid_fills, n_ask_fills, times, Delta,
         liquidation_penalty, terminal_wealth, [utility]
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

    db_table = sol["delta_bid"]
    da_table = sol["delta_ask"]

    price     = np.zeros((N_sim, N_t + 1))
    cash      = np.zeros((N_sim, N_t + 1))
    inv_lots  = np.zeros((N_sim, N_t + 1), dtype=int)
    mtm       = np.zeros((N_sim, N_t + 1))
    n_bid_fills = np.zeros(N_sim, dtype=int)
    n_ask_fills = np.zeros(N_sim, dtype=int)

    Z = rng.standard_normal((N_sim, N_t))

    if ell is None:
        ell = lambda q_abs_dollars: 0.0  # noqa: E731

    for m in range(N_sim):
        S = 0.0
        X = 0.0
        n = 0

        price[m, 0] = S
        cash[m, 0]  = X
        inv_lots[m, 0] = n
        mtm[m, 0] = X + (n * Delta) * S

        for t_idx in range(N_t):
            i_lot = n + Q
            db = db_table[t_idx, i_lot]
            da = da_table[t_idx, i_lot]

            can_bid = (n < Q) and np.isfinite(db)
            can_ask = (n > -Q) and np.isfinite(da)

            if can_bid:
                lam_bid = A * np.exp(-k * db)
                N_bid = rng.poisson(lam=lam_bid * dt)
                if N_bid > 0:
                    N_exec = min(N_bid, Q - n)
                    X -= N_exec * (S - db) * Delta
                    n += N_exec
                    n_bid_fills[m] += N_exec

            if can_ask:
                lam_ask = A * np.exp(-k * da)
                N_ask = rng.poisson(lam=lam_ask * dt)
                if N_ask > 0:
                    N_exec = min(N_ask, n - (-Q))
                    X += N_exec * (S + da) * Delta
                    n -= N_exec
                    n_ask_fills[m] += N_exec

            S += sigma * np.sqrt(dt) * Z[m, t_idx]

            price[m, t_idx + 1] = S
            cash[m, t_idx + 1]  = X
            inv_lots[m, t_idx + 1] = n
            mtm[m, t_idx + 1] = X + (n * Delta) * S

    qT_dollars = inv_lots[:, -1] * Delta
    ST = price[:, -1]
    XT = cash[:, -1]

    liquidation_penalty = np.array([ell(abs(q)) for q in qT_dollars], dtype=float)
    W_T = XT + qT_dollars * ST - liquidation_penalty
    pnl_mtm = XT + qT_dollars * ST

    out = dict(
        pnl_mtm=pnl_mtm,
        pnl_paper=W_T,
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
        out["utility"] = -np.exp(-gamma * W_T)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Naive strategy (fixed symmetric spread)
# ═══════════════════════════════════════════════════════════════════════════════

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

    Returns dict with keys matching simulate_1d interface.
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
        ell = lambda q_abs_dollars: 0.0  # noqa: E731

    lam = A * np.exp(-k * half_spread)

    for m in range(N_sim):
        S = 0.0
        X = 0.0
        n = 0

        price[m, 0] = S
        cash[m, 0]  = X
        inv_lots[m, 0] = n
        mtm[m, 0] = X + (n * Delta) * S

        for t_idx in range(N_t):
            N_bid = rng.poisson(lam=lam * dt) if n < Q else 0
            if N_bid > 0:
                N_exec = min(N_bid, Q - n)
                X -= N_exec * (S - half_spread) * Delta
                n += N_exec
                n_bid_fills[m] += N_exec

            N_ask = rng.poisson(lam=lam * dt) if n > -Q else 0
            if N_ask > 0:
                N_exec = min(N_ask, n - (-Q))
                X += N_exec * (S + half_spread) * Delta
                n -= N_exec
                n_ask_fills[m] += N_exec

            S += sigma * np.sqrt(dt) * Z[m, t_idx]

            price[m, t_idx + 1] = S
            cash[m, t_idx + 1]  = X
            inv_lots[m, t_idx + 1] = n
            mtm[m, t_idx + 1] = X + (n * Delta) * S

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


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline intensity models
# ═══════════════════════════════════════════════════════════════════════════════

def _base_intensity(delta, A, k, model_cfg):
    """Execution baseline intensity as a function of quote distance.

    Supported kinds: exponential (default), power_law, logistic.
    """
    model_cfg = model_cfg or {}
    kind = model_cfg.get("kind", "exponential")
    d = max(float(delta), 0.0)
    if kind == "exponential":
        return float(A) * np.exp(-float(k) * d)
    if kind == "power_law":
        delta0 = float(model_cfg.get("delta0", 1.0 / max(float(k), 1e-12)))
        alpha = float(model_cfg.get("alpha", 1.8))
        return float(A) * (1.0 + d / max(delta0, 1e-12)) ** (-alpha)
    if kind == "logistic":
        delta_mid = float(model_cfg.get("delta_mid", 1.0 / max(float(k), 1e-12)))
        slope = float(model_cfg.get("slope", 0.35 / max(float(k), 1e-12)))
        x = float(np.clip((d - delta_mid) / max(slope, 1e-12), -60.0, 60.0))
        return float(A) / (1.0 + np.exp(x))
    raise ValueError(f"Unknown intensity model kind: {kind!r}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Hawkes simulator (uses HawkesState — no manual state management)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_hawkes_1d(
    sol,
    params,
    T,
    hawkes_cfg,
    intensity_model=None,
    policy_name="gueant_static",
    policy_cfg=None,
    N_sim=1000,
    seed=42,
):
    """Run Hawkes-execution simulations with policy overlays.

    Parameters
    ----------
    sol : dict — output of solve_general (delta_bid, delta_ask, times)
    params : dict — sigma, A, k, Delta, Q
    T : float — horizon in seconds
    hawkes_cfg : dict — beta, alpha_self, alpha_cross, init_excitation
    intensity_model : dict or None — baseline shape (kind: exponential/power_law/logistic)
    policy_name : str — one of: gueant_static, inv_skew, hawkes_aware
    policy_cfg : dict or None — optional policy parameters (eta_inv, eta_exc, eta_wide)
    N_sim : int
    seed : int

    Returns
    -------
    dict with pnl, inventory, cash, mtm, price, times,
         n_bid_fills, n_ask_fills, turnover, avg_lambda_bid, avg_lambda_ask
    """
    policy_cfg = policy_cfg or {}
    rng = np.random.default_rng(seed)

    sigma = float(params["sigma"])
    A     = float(params["A"])
    k     = float(params["k"])
    Delta = float(params["Delta"])
    Q     = int(params["Q"])

    init_excitation = float(hawkes_cfg.get("init_excitation", 0.0))

    eta_inv  = float(policy_cfg.get("eta_inv",  0.25))
    eta_exc  = float(policy_cfg.get("eta_exc",  0.18))
    eta_wide = float(policy_cfg.get("eta_wide", 0.10))

    times = sol["times"]
    N_t = len(times) - 1
    dt = T / N_t

    db_table = sol["delta_bid"]
    da_table = sol["delta_ask"]

    price       = np.zeros((N_sim, N_t + 1))
    cash        = np.zeros((N_sim, N_t + 1))
    inv_lots    = np.zeros((N_sim, N_t + 1), dtype=int)
    mtm         = np.zeros((N_sim, N_t + 1))
    n_bid_fills = np.zeros(N_sim, dtype=int)
    n_ask_fills = np.zeros(N_sim, dtype=int)
    avg_lambda_bid = np.zeros(N_sim)
    avg_lambda_ask = np.zeros(N_sim)

    dW    = rng.standard_normal((N_sim, N_t))
    U_bid = rng.uniform(size=(N_sim, N_t))
    U_ask = rng.uniform(size=(N_sim, N_t))

    for m in range(N_sim):
        S = 0.0
        X = 0.0
        n = 0

        # Initialise HawkesState for this path
        state = HawkesState(hawkes_cfg, y_bid0=init_excitation, y_ask0=init_excitation)

        price[m, 0] = S
        cash[m, 0]  = X
        inv_lots[m, 0] = n
        mtm[m, 0] = X + n * Delta * S

        lam_bid_acc = 0.0
        lam_ask_acc = 0.0

        for t_idx in range(N_t):
            i_lot = n + Q
            db0 = db_table[t_idx, i_lot]
            da0 = da_table[t_idx, i_lot]
            can_bid = (n < Q) and np.isfinite(db0)
            can_ask = (n > -Q) and np.isfinite(da0)

            db = float(db0) if can_bid else np.nan
            da = float(da0) if can_ask else np.nan

            # Policy overlay (inventory skew + Hawkes-aware adjustments)
            if can_bid and can_ask:
                base_spread = max(db + da, 1e-12)
                inv_frac = n / max(Q, 1)
                mu_b = A * np.exp(-k * db)
                mu_a = A * np.exp(-k * da)
                norm = max(mu_b + mu_a, 1e-12)
                exc_imb = np.clip((state.y_bid - state.y_ask) / norm, -2.5, 2.5)
                exc_tot = np.clip((state.y_bid + state.y_ask) / norm, 0.0, 3.0)

                if policy_name in ("inv_skew", "hawkes_aware"):
                    skew = eta_inv * inv_frac * base_spread
                    db += skew
                    da -= skew

                if policy_name == "hawkes_aware":
                    skew_exc = eta_exc * exc_imb * base_spread
                    widen_exc = eta_wide * max(0.0, exc_tot - 0.5) * base_spread
                    db += skew_exc + widen_exc
                    da -= skew_exc + widen_exc

                db = max(db, 0.0)
                da = max(da, 0.0)

            # Price dynamics
            S += sigma * np.sqrt(dt) * dW[m, t_idx]

            fill_bid = 0
            fill_ask = 0

            if can_bid:
                mu_bid = _base_intensity(db, A, k, intensity_model)
                lam_bid = state.lambda_bid(mu_bid)
                if U_bid[m, t_idx] < fill_prob(lam_bid, dt):
                    fill_bid = 1
                    n += 1
                    X -= (S - db) * Delta
                    n_bid_fills[m] += 1
                lam_bid_acc += lam_bid

            if can_ask:
                mu_ask = _base_intensity(da, A, k, intensity_model)
                lam_ask = state.lambda_ask(mu_ask)
                if U_ask[m, t_idx] < fill_prob(lam_ask, dt):
                    fill_ask = 1
                    n -= 1
                    X += (S + da) * Delta
                    n_ask_fills[m] += 1
                lam_ask_acc += lam_ask

            # Hawkes state update delegated to HawkesState
            state.step(dt, fill_bid=bool(fill_bid), fill_ask=bool(fill_ask))

            price[m, t_idx + 1] = S
            cash[m, t_idx + 1]  = X
            inv_lots[m, t_idx + 1] = n
            mtm[m, t_idx + 1] = X + n * Delta * S

        avg_lambda_bid[m] = lam_bid_acc / max(N_t, 1)
        avg_lambda_ask[m] = lam_ask_acc / max(N_t, 1)

    pnl = cash[:, -1] + inv_lots[:, -1] * Delta * price[:, -1]
    turnover = (n_bid_fills + n_ask_fills) * Delta

    return dict(
        pnl=pnl,
        inventory=inv_lots,
        cash=cash,
        mtm=mtm,
        price=price,
        times=times,
        n_bid_fills=n_bid_fills,
        n_ask_fills=n_ask_fills,
        turnover=turnover,
        avg_lambda_bid=avg_lambda_bid,
        avg_lambda_ask=avg_lambda_ask,
        policy_name=policy_name,
    )
