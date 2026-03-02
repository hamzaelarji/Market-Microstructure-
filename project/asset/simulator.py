"""Monte Carlo simulator for optimal market making (Guéant 2017).

Simulates price paths + Poisson fills using pre‑computed optimal quotes
from the ODE solvers, and records P&L, inventory, and MtM trajectories.

Convention (matching ode_solver_1d / ode_solver_2d):
  δ = absolute distance to mid  (NOT relative)
  S^bid = S − δ^b ,   S^ask = S + δ^a
  Cash update on bid fill:   X ← X − (S − δ^b) · Δ
  Cash update on ask fill:   X ← X + (S + δ^a) · Δ
"""
from __future__ import annotations
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
    res = simulate_1d(sol, IG, GAMMA, T, N_sim=1000, seed=0, ell=ell)

    print(f"  Mean MtM PnL        = {np.mean(res['pnl_mtm']):.4e}")
    print(f"  Mean paper PnL      = {np.mean(res['pnl_paper']):.4e}")
    print(f"  Mean liquidation    = {np.mean(res['liquidation_penalty']):.4e}")
    print(f"  Mean utility        = {np.mean(res['utility']):.4e}")
    print(f"  Mean bid fills      = {np.mean(res['n_bid_fills']):.1f} lots")
    print(f"  Mean ask fills      = {np.mean(res['n_ask_fills']):.1f} lots")
    print(f"  Mean |inv| at T     = {np.mean(np.abs(res['inventory'][:, -1])):.2f} lots")


#"""Hawkes-type execution simulator for single-asset market making.

#This module replaces the independent Poisson fill model with self-/cross-exciting
#execution intensities while preserving the quote-control interface of the ODE policy.
#""


def _clip_prob_from_intensity(lam, dt):
    lam = max(float(lam), 0.0)
    # More stable than lam*dt when intensity spikes.
    return 1.0 - np.exp(-lam * dt)


def _base_intensity(delta, A, k, model_cfg):
    """Execution baseline intensity as a function of quote distance.

    Supported models:
    - exponential: A * exp(-k * delta)
    - power_law:  A * (1 + delta / delta0)^(-alpha)
    - logistic:   A / (1 + exp((delta - delta_mid) / slope))
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
        x = (d - delta_mid) / max(slope, 1e-12)
        x = float(np.clip(x, -60.0, 60.0))
        return float(A) / (1.0 + np.exp(x))
    raise ValueError(f"Unknown intensity model kind: {kind}")


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
    sol : dict
        Output of `solve_general`, containing delta_bid, delta_ask, times.
    params : dict
        Must contain sigma, A, k, Delta, Q.
    T : float
        Horizon in seconds.
    hawkes_cfg : dict
        Keys: beta, jump_self, jump_cross, init_excitation.
    intensity_model : dict or None
        Baseline shape for quote-dependent execution intensity.
        Examples:
          {"kind": "exponential"}
          {"kind": "power_law", "alpha": 1.8, "delta0": 5.6e-5}
          {"kind": "logistic", "delta_mid": 5.6e-5, "slope": 2.0e-5}
    policy_name : str
        One of: gueant_static, inv_skew, hawkes_aware.
    policy_cfg : dict or None
        Optional policy parameters.
    N_sim : int
    seed : int
    """
    policy_cfg = policy_cfg or {}
    rng = np.random.default_rng(seed)

    sigma = float(params["sigma"])
    A = float(params["A"])
    k = float(params["k"])
    Delta = float(params["Delta"])
    Q = int(params["Q"])

    beta = float(hawkes_cfg.get("beta", 4.0))
    jump_self = float(hawkes_cfg.get("jump_self", 2.0e-4))
    jump_cross = float(hawkes_cfg.get("jump_cross", 8.0e-5))
    init_excitation = float(hawkes_cfg.get("init_excitation", 0.0))

    eta_inv = float(policy_cfg.get("eta_inv", 0.25))
    eta_exc = float(policy_cfg.get("eta_exc", 0.18))
    eta_wide = float(policy_cfg.get("eta_wide", 0.10))

    times = sol["times"]
    N_t = len(times) - 1
    dt = T / N_t
    decay = np.exp(-beta * dt)

    db_table = sol["delta_bid"]
    da_table = sol["delta_ask"]

    price = np.zeros((N_sim, N_t + 1))
    cash = np.zeros((N_sim, N_t + 1))
    inv_lots = np.zeros((N_sim, N_t + 1), dtype=int)
    mtm = np.zeros((N_sim, N_t + 1))
    n_bid_fills = np.zeros(N_sim, dtype=int)
    n_ask_fills = np.zeros(N_sim, dtype=int)
    avg_lambda_bid = np.zeros(N_sim)
    avg_lambda_ask = np.zeros(N_sim)

    dW = rng.standard_normal((N_sim, N_t))
    U_bid = rng.uniform(size=(N_sim, N_t))
    U_ask = rng.uniform(size=(N_sim, N_t))

    for m in range(N_sim):
        S = 0.0
        X = 0.0
        n = 0
        yb = init_excitation
        ya = init_excitation

        price[m, 0] = S
        cash[m, 0] = X
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

            # Baseline Guéant policy.
            db = float(db0) if can_bid else np.nan
            da = float(da0) if can_ask else np.nan

            if can_bid and can_ask:
                base_spread = max(db + da, 1e-12)
                inv_frac = n / max(Q, 1)
                mu_b = A * np.exp(-k * db)
                mu_a = A * np.exp(-k * da)
                norm = max(mu_b + mu_a, 1e-12)
                exc_imb = np.clip((yb - ya) / norm, -2.5, 2.5)
                exc_tot = np.clip((yb + ya) / norm, 0.0, 3.0)

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
                lam_bid = mu_bid + yb
                p_bid = _clip_prob_from_intensity(lam_bid, dt)
                if U_bid[m, t_idx] < p_bid:
                    fill_bid = 1
                    n += 1
                    X -= (S - db) * Delta
                    n_bid_fills[m] += 1
                lam_bid_acc += lam_bid
            if can_ask:
                mu_ask = _base_intensity(da, A, k, intensity_model)
                lam_ask = mu_ask + ya
                p_ask = _clip_prob_from_intensity(lam_ask, dt)
                if U_ask[m, t_idx] < p_ask:
                    fill_ask = 1
                    n -= 1
                    X += (S + da) * Delta
                    n_ask_fills[m] += 1
                lam_ask_acc += lam_ask

            # Hawkes excitation update (discrete-time decay + jumps on events).
            yb = decay * yb + jump_self * fill_bid + jump_cross * fill_ask
            ya = decay * ya + jump_self * fill_ask + jump_cross * fill_bid

            price[m, t_idx + 1] = S
            cash[m, t_idx + 1] = X
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
