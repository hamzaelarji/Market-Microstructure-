"""Hawkes-type execution simulator for single-asset market making.

This module replaces the independent Poisson fill model with self-/cross-exciting
execution intensities while preserving the quote-control interface of the ODE policy.
"""

from __future__ import annotations

import numpy as np


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
