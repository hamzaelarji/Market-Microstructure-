"""Notebook 08 — Counterfactual Shadow-Policy (innovative extension).

Concept:
- Keep several candidate microstructure "worlds" (different A, k, sigma).
- Precompute an optimal quote table for each world with the same HJB/ODE framework.
- Online, infer which world is currently active from observed fills and price shocks.
- Quote using a stress-distorted Bayesian blend of world-specific optimal quotes.

This is not plain regime switching control. The agent does not know the regime and
does not simply pick one model; it performs online belief updates and robust blending.

Run:
  python 08_counterfactual_shadow_policy.py
"""

from pathlib import Path
import os
import sys
import argparse

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from asset.experiment_sets import get_scenario, scenario_names
from src.ode_solver_1d import solve_general

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_worlds(base):
    """Define candidate microstructure worlds."""
    return [
        dict(name="Calm", A=base["A"] * 0.80, k=base["k"] * 0.90, sigma=base["sigma"] * 0.75),
        dict(name="Base", A=base["A"] * 1.00, k=base["k"] * 1.00, sigma=base["sigma"] * 1.00),
        dict(name="Stressed", A=base["A"] * 1.35, k=base["k"] * 1.20, sigma=base["sigma"] * 1.45),
    ]


def build_true_path(n_t):
    """Unknown true world index path with one abrupt shock."""
    # Base -> Stressed around mid-session.
    switch = int(0.55 * n_t)
    idx = np.zeros(n_t, dtype=int)
    idx[:switch] = 1
    idx[switch:] = 2
    return idx


def precompute_tables(base_params, worlds, gamma, horizon, n_t):
    """Solve one policy table per world."""
    tables = []
    for w in worlds:
        p = {**base_params, "A": w["A"], "k": w["k"], "sigma": w["sigma"]}
        sol = solve_general(p, gamma, horizon, xi=gamma, N_t=n_t)
        tables.append(dict(delta_bid=sol["delta_bid"], delta_ask=sol["delta_ask"], params=p))
    return tables


def _safe_log_bernoulli(x, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return x * np.log(p) + (1 - x) * np.log(1 - p)


def simulate_one(
    mode,
    base_params,
    worlds,
    tables,
    true_world_idx_path,
    horizon,
    stress_eta=1.6,
    seed=0,
):
    """Simulate one trajectory for a chosen policy mode.

    Modes:
    - static: always use Base-world quote table
    - oracle: use true-world table at each step
    - shadow: belief-updated stress-distorted blending of all tables
    """
    rng = np.random.default_rng(seed)
    n_t = len(true_world_idx_path)
    dt = horizon / n_t
    times = np.linspace(0.0, horizon, n_t + 1)

    Delta = base_params["Delta"]
    Q = int(base_params["Q"])
    n_worlds = len(worlds)

    # State
    S = 0.0
    X = 0.0
    n = 0

    price = np.zeros(n_t + 1)
    cash = np.zeros(n_t + 1)
    inv = np.zeros(n_t + 1, dtype=int)
    db_used = np.zeros(n_t)
    da_used = np.zeros(n_t)
    n_bid_fills = 0
    n_ask_fills = 0

    # Belief over worlds (for shadow mode)
    posterior = np.ones(n_worlds) / n_worlds
    posterior_hist = np.zeros((n_t + 1, n_worlds))
    posterior_hist[0] = posterior

    for t_idx in range(n_t):
        i_lot = n + Q
        true_j = int(true_world_idx_path[t_idx])
        true_w = worlds[true_j]

        # Select quotes by mode
        if mode == "static":
            db = tables[1]["delta_bid"][t_idx, i_lot]
            da = tables[1]["delta_ask"][t_idx, i_lot]
        elif mode == "oracle":
            db = tables[true_j]["delta_bid"][t_idx, i_lot]
            da = tables[true_j]["delta_ask"][t_idx, i_lot]
        elif mode == "shadow":
            cand_db = np.array([tables[j]["delta_bid"][t_idx, i_lot] for j in range(n_worlds)])
            cand_da = np.array([tables[j]["delta_ask"][t_idx, i_lot] for j in range(n_worlds)])
            finite = np.isfinite(cand_db) & np.isfinite(cand_da)
            if not np.any(finite):
                db = np.nan
                da = np.nan
            else:
                # Stress distortion: overweight higher-vol worlds.
                sig = np.array([w["sigma"] for w in worlds])
                stress = sig / np.mean(sig)
                w = posterior * np.power(stress, stress_eta)
                w = np.where(finite, w, 0.0)
                if np.sum(w) <= 0:
                    w = finite.astype(float)
                w = w / np.sum(w)
                db = float(np.nansum(w * cand_db))
                da = float(np.nansum(w * cand_da))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        can_bid = (n < Q) and np.isfinite(db)
        can_ask = (n > -Q) and np.isfinite(da)
        db_used[t_idx] = db if np.isfinite(db) else np.nan
        da_used[t_idx] = da if np.isfinite(da) else np.nan

        # True market evolution
        dS = true_w["sigma"] * np.sqrt(dt) * rng.standard_normal()
        S = S + dS

        fill_bid = 0
        fill_ask = 0
        if can_bid:
            p_bid_true = np.clip(true_w["A"] * np.exp(-true_w["k"] * db) * dt, 0.0, 1.0)
            fill_bid = int(rng.uniform() < p_bid_true)
            if fill_bid:
                n += 1
                X -= (S - db) * Delta
                n_bid_fills += 1
        if can_ask:
            p_ask_true = np.clip(true_w["A"] * np.exp(-true_w["k"] * da) * dt, 0.0, 1.0)
            fill_ask = int(rng.uniform() < p_ask_true)
            if fill_ask:
                n -= 1
                X += (S + da) * Delta
                n_ask_fills += 1

        # Bayes update after observing (dS, fill_bid, fill_ask)
        if mode == "shadow":
            log_like = np.zeros(n_worlds)
            for j, w in enumerate(worlds):
                # Gaussian shock likelihood
                var = (w["sigma"] ** 2) * dt + 1e-18
                ll_price = -0.5 * (np.log(2 * np.pi * var) + (dS**2) / var)

                ll_fill = 0.0
                if can_bid:
                    p_bid_j = np.clip(w["A"] * np.exp(-w["k"] * db) * dt, 1e-12, 1 - 1e-12)
                    ll_fill += _safe_log_bernoulli(fill_bid, p_bid_j)
                if can_ask:
                    p_ask_j = np.clip(w["A"] * np.exp(-w["k"] * da) * dt, 1e-12, 1 - 1e-12)
                    ll_fill += _safe_log_bernoulli(fill_ask, p_ask_j)

                log_like[j] = ll_price + ll_fill

            log_post = np.log(np.clip(posterior, 1e-18, 1.0)) + log_like
            log_post -= np.max(log_post)
            posterior = np.exp(log_post)
            posterior /= np.sum(posterior)

        price[t_idx + 1] = S
        cash[t_idx + 1] = X
        inv[t_idx + 1] = n
        posterior_hist[t_idx + 1] = posterior

    pnl = cash[-1] + inv[-1] * Delta * price[-1]
    mtm = cash + inv * Delta * price
    return dict(
        pnl=pnl,
        mtm=mtm,
        inventory=inv,
        price=price,
        delta_bid=db_used,
        delta_ask=da_used,
        posterior=posterior_hist,
        times=times,
        n_bid_fills=n_bid_fills,
        n_ask_fills=n_ask_fills,
    )


def simulate_many(mode, n_sim, base_params, worlds, tables, true_world_idx_path, horizon, seed0):
    out = []
    for i in range(n_sim):
        out.append(
            simulate_one(
                mode,
                base_params,
                worlds,
                tables,
                true_world_idx_path,
                horizon,
                seed=seed0 + i,
            )
        )
    pnl = np.array([x["pnl"] for x in out])
    invT = np.array([x["inventory"][-1] for x in out])
    n_bid = np.array([x["n_bid_fills"] for x in out], dtype=float)
    n_ask = np.array([x["n_ask_fills"] for x in out], dtype=float)
    turnover = (n_bid + n_ask) * base_params["Delta"]
    return dict(pnl=pnl, invT=invT, n_bid=n_bid, n_ask=n_ask, turnover=turnover, sample=out[0])


def summarize(name, res, gamma):
    pnl = res["pnl"]
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl))
    sharpe = mu / max(sd, 1e-12)
    ce = -np.log(np.mean(np.exp(-gamma * pnl))) / gamma
    var5 = float(np.percentile(pnl, 5))
    cvar5 = float(np.mean(pnl[pnl <= var5]))
    mean_turnover = float(np.mean(res["turnover"]))
    mean_fills = float(np.mean(res["n_bid"] + res["n_ask"]))
    print(
        f"{name:7s} mean={mu:+.1f} std={sd:.1f} Sharpe={sharpe:.3f} "
        f"CE={ce:+.1f} VaR5={var5:+.1f} CVaR5={cvar5:+.1f} "
        f"fills={mean_fills:.2f} turnover={mean_turnover:.2e}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfactual shadow-policy experiment")
    parser.add_argument("--scenario", default="baseline", help="Scenario name from asset.experiment_sets")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios and exit")
    args = parser.parse_args()

    if args.list_scenarios:
        print("\n".join(scenario_names()))
        raise SystemExit(0)

    cfg = get_scenario(args.scenario)
    IG = cfg["IG"]
    GAMMA = cfg["GAMMA"]
    T = cfg["T"]
    scenario_tag = cfg["name"]

    N_T = 360
    N_SIM = 350

    print(f"Scenario: {scenario_tag}")
    print("Building candidate worlds and precomputing policies...")
    worlds = build_worlds(IG)
    tables = precompute_tables(IG, worlds, GAMMA, T, N_T)
    true_idx = build_true_path(N_T)

    print("Running Monte Carlo comparison...")
    res_static = simulate_many("static", N_SIM, IG, worlds, tables, true_idx, T, seed0=100)
    res_shadow = simulate_many("shadow", N_SIM, IG, worlds, tables, true_idx, T, seed0=200)
    res_oracle = simulate_many("oracle", N_SIM, IG, worlds, tables, true_idx, T, seed0=300)

    for name, res in [("Static", res_static), ("Shadow", res_shadow), ("Oracle", res_oracle)]:
        summarize(name, res, GAMMA)

    # --- Plot 1: posterior + true world (sample path)
    s = res_shadow["sample"]
    times = s["times"]
    t_mid = times[:-1]
    true_world = true_idx

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].step(t_mid, true_world, where="post", lw=1.5, color="k")
    axes[0].set_yticks([0, 1, 2], labels=[w["name"] for w in worlds])
    axes[0].set_title("True latent world (unknown to policy)")
    axes[0].grid(alpha=0.3)

    for j, w in enumerate(worlds):
        axes[1].plot(times, s["posterior"][:, j], lw=1.5, label=w["name"])
    axes[1].set_title("Shadow-policy posterior belief over worlds")
    axes[1].set_ylabel("Posterior probability")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    f1 = FIG_DIR / f"fig08_1_true_world_and_posterior_{scenario_tag}.png"
    fig.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f1.name}")

    # --- Plot 2: sample quotes and inventory comparison
    s_stat = res_static["sample"]
    s_shad = res_shadow["sample"]
    s_orac = res_oracle["sample"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    spread_stat = s_stat["delta_bid"] + s_stat["delta_ask"]
    spread_shad = s_shad["delta_bid"] + s_shad["delta_ask"]
    spread_orac = s_orac["delta_bid"] + s_orac["delta_ask"]
    axes[0].plot(t_mid, spread_stat, lw=1.2, label="Static")
    axes[0].plot(t_mid, spread_shad, lw=1.2, label="Shadow")
    axes[0].plot(t_mid, spread_orac, lw=1.2, label="Oracle")
    axes[0].set_ylabel("Spread")
    axes[0].set_title("Sample-path spread response")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].step(times, s_stat["inventory"], where="post", lw=1.2, label="Static")
    axes[1].step(times, s_shad["inventory"], where="post", lw=1.2, label="Shadow")
    axes[1].step(times, s_orac["inventory"], where="post", lw=1.2, label="Oracle")
    axes[1].set_ylabel("Inventory (lots)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Sample-path inventory")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    f2 = FIG_DIR / f"fig08_2_spread_and_inventory_sample_{scenario_tag}.png"
    fig.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f2.name}")

    # --- Plot 3: terminal PnL distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    all_min = min(res_static["pnl"].min(), res_shadow["pnl"].min(), res_oracle["pnl"].min())
    all_max = max(res_static["pnl"].max(), res_shadow["pnl"].max(), res_oracle["pnl"].max())
    bins = np.linspace(all_min, all_max, 80)

    ax.hist(res_static["pnl"], bins=bins, density=True, alpha=0.35, label="Static")
    ax.hist(res_shadow["pnl"], bins=bins, density=True, alpha=0.35, label="Shadow")
    ax.hist(res_oracle["pnl"], bins=bins, density=True, alpha=0.35, label="Oracle")
    ax.axvline(np.mean(res_static["pnl"]), color="C0", ls="--", lw=1.5)
    ax.axvline(np.mean(res_shadow["pnl"]), color="C1", ls="--", lw=1.5)
    ax.axvline(np.mean(res_oracle["pnl"]), color="C2", ls="--", lw=1.5)
    ax.set_title("Terminal P&L: Static vs Shadow vs Oracle")
    ax.set_xlabel("P&L ($)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    f3 = FIG_DIR / f"fig08_3_pnl_distributions_{scenario_tag}.png"
    fig.savefig(f3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f3.name}")

    print("Done.")
