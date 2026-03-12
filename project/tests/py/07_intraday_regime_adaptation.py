"""Notebook 07 — Intraday regime-aware market making (innovation extension).

Idea:
- Market conditions are not constant intraday.
- Simulate piecewise-constant liquidity/volatility regimes.
- Compare:
  1) Static policy: single solve with baseline params.
  2) Regime-aware policy: re-solve ODE at each regime boundary with updated params.

Run:
  python 07_intraday_regime_adaptation.py
"""

from pathlib import Path
import sys
import os
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


def build_intraday_paths(base_params, t_grid):
    """Create piecewise paths for A(t), sigma(t) with open/noon/close regimes."""
    n = len(t_grid) - 1
    A_path = np.zeros(n)
    sigma_path = np.zeros(n)

    # Fractions of day: [0, 0.3), [0.3, 0.7), [0.7, 1.0]
    # Open/close: higher vol + higher activity. Noon: calmer.
    breaks = [0, int(0.3 * n), int(0.7 * n), n]
    regimes = [
        dict(A_mult=1.45, sigma_mult=1.50, name="Open"),
        dict(A_mult=0.60, sigma_mult=0.55, name="Noon"),
        dict(A_mult=1.35, sigma_mult=1.40, name="Close"),
    ]

    for r, (s, e) in enumerate(zip(breaks[:-1], breaks[1:])):
        A_path[s:e] = base_params["A"] * regimes[r]["A_mult"]
        sigma_path[s:e] = base_params["sigma"] * regimes[r]["sigma_mult"]

    return A_path, sigma_path, breaks, regimes


def solve_regime_aware_tables(base_params, gamma, horizon, n_t, breaks, regimes):
    """Build quote tables by re-solving at each regime boundary."""
    Q = int(base_params["Q"])
    n_states = 2 * Q + 1
    db = np.full((n_t + 1, n_states), np.nan)
    da = np.full((n_t + 1, n_states), np.nan)

    for r, (s, e) in enumerate(zip(breaks[:-1], breaks[1:])):
        rem_steps = n_t - s
        rem_horizon = horizon * rem_steps / n_t

        p_reg = {
            **base_params,
            "A": base_params["A"] * regimes[r]["A_mult"],
            "sigma": base_params["sigma"] * regimes[r]["sigma_mult"],
        }
        sol = solve_general(p_reg, gamma, rem_horizon, xi=gamma, N_t=rem_steps)
        seg_len = e - s

        # Use the next seg_len rows from the local (remaining-horizon) solve.
        db[s:e, :] = sol["delta_bid"][:seg_len, :]
        da[s:e, :] = sol["delta_ask"][:seg_len, :]

    # Use last available row for terminal slot.
    db[-1, :] = db[-2, :]
    da[-1, :] = da[-2, :]
    return db, da


def simulate_with_paths(db_table, da_table, base_params, A_path, sigma_path, horizon, n_sim=2000, seed=7):
    """Simulate P&L under time-varying A(t), sigma(t) using supplied quote tables."""
    rng = np.random.default_rng(seed)

    k = base_params["k"]
    Delta = base_params["Delta"]
    Q = int(base_params["Q"])
    n_t = len(A_path)
    dt = horizon / n_t

    price = np.zeros((n_sim, n_t + 1))
    cash = np.zeros((n_sim, n_t + 1))
    inv = np.zeros((n_sim, n_t + 1), dtype=int)
    mtm = np.zeros((n_sim, n_t + 1))

    z = rng.standard_normal((n_sim, n_t))
    u_b = rng.uniform(size=(n_sim, n_t))
    u_a = rng.uniform(size=(n_sim, n_t))

    for m in range(n_sim):
        S = 0.0
        X = 0.0
        n = 0
        price[m, 0] = S
        cash[m, 0] = X
        inv[m, 0] = n
        mtm[m, 0] = X + n * Delta * S

        for t_idx in range(n_t):
            i_lot = n + Q
            db = db_table[t_idx, i_lot]
            da = da_table[t_idx, i_lot]
            can_bid = (n < Q) and np.isfinite(db)
            can_ask = (n > -Q) and np.isfinite(da)

            S += sigma_path[t_idx] * np.sqrt(dt) * z[m, t_idx]

            if can_bid:
                p_bid = A_path[t_idx] * np.exp(-k * db) * dt
                if u_b[m, t_idx] < p_bid:
                    n += 1
                    X -= (S - db) * Delta
            if can_ask:
                p_ask = A_path[t_idx] * np.exp(-k * da) * dt
                if u_a[m, t_idx] < p_ask:
                    n -= 1
                    X += (S + da) * Delta

            price[m, t_idx + 1] = S
            cash[m, t_idx + 1] = X
            inv[m, t_idx + 1] = n
            mtm[m, t_idx + 1] = X + n * Delta * S

    pnl = cash[:, -1] + inv[:, -1] * Delta * price[:, -1]
    return dict(pnl=pnl, inventory=inv, mtm=mtm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intraday regime-aware experiment")
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

    N_T = 480
    N_SIM = 800
    times = np.linspace(0.0, T, N_T + 1)

    print(f"Scenario: {scenario_tag}")

    print("Building intraday paths...")
    A_path, sigma_path, breaks, regimes = build_intraday_paths(IG, times)

    print("Solving static policy...")
    sol_static = solve_general(IG, GAMMA, T, xi=GAMMA, N_t=N_T)
    db_static = sol_static["delta_bid"]
    da_static = sol_static["delta_ask"]

    print("Solving regime-aware policy...")
    db_adapt, da_adapt = solve_regime_aware_tables(IG, GAMMA, T, N_T, breaks, regimes)

    print(f"Simulating {N_SIM} trajectories (static policy)...")
    res_static = simulate_with_paths(db_static, da_static, IG, A_path, sigma_path, T, n_sim=N_SIM, seed=10)

    print(f"Simulating {N_SIM} trajectories (regime-aware policy)...")
    res_adapt = simulate_with_paths(db_adapt, da_adapt, IG, A_path, sigma_path, T, n_sim=N_SIM, seed=10)

    pnl_s = res_static["pnl"]
    pnl_a = res_adapt["pnl"]
    sharpe_s = np.mean(pnl_s) / max(np.std(pnl_s), 1e-12)
    sharpe_a = np.mean(pnl_a) / max(np.std(pnl_a), 1e-12)

    print("\nSummary:")
    print(f"  Static      mean={np.mean(pnl_s):+.1f}  std={np.std(pnl_s):.1f}  Sharpe={sharpe_s:.3f}")
    print(f"  RegimeAware mean={np.mean(pnl_a):+.1f}  std={np.std(pnl_a):.1f}  Sharpe={sharpe_a:.3f}")
    print(f"  Delta mean PnL (adapt-static) = {np.mean(pnl_a) - np.mean(pnl_s):+.1f}")

    # Plot 1: regime multipliers + n=0 spread and delta.
    Q = int(IG["Q"])
    spread0_static = db_static[:, Q] + da_static[:, Q]
    spread0_adapt = db_adapt[:, Q] + da_adapt[:, Q]
    spread0_diff = spread0_adapt - spread0_static

    t_step = times[:-1]
    A_mult = A_path / IG["A"]
    sigma_mult = sigma_path / IG["sigma"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].step(t_step, A_mult, where="post", lw=1.5)
    axes[0].set_ylabel("A(t)/A0")
    axes[0].set_title("Intraday regime multipliers")
    axes[0].grid(alpha=0.3)

    axes[1].step(t_step, sigma_mult, where="post", lw=1.5, color="C1")
    axes[1].set_ylabel("sigma(t)/sigma0")
    axes[1].grid(alpha=0.3)

    axes[2].plot(times, spread0_static, label="Static policy", lw=1.5)
    axes[2].plot(times, spread0_adapt, label="Regime-aware policy", lw=1.5)
    axes[2].set_ylabel("Spread at n=0")
    axes[2].set_title("Quoted spread response")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(times, spread0_diff, lw=1.5, color="C3")
    axes[3].axhline(0.0, color="k", ls="--", lw=0.8)
    axes[3].set_ylabel("Adapt - Static")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(alpha=0.3)

    # Mark regime boundaries on all panels.
    for b in breaks[1:-1]:
        tb = times[b]
        for ax in axes:
            ax.axvline(tb, color="gray", ls=":", lw=0.9)

    fig.tight_layout()
    f1 = FIG_DIR / f"fig07_1_v2_intraday_paths_and_spreads_{scenario_tag}.png"
    fig.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {f1.name}")

    # Plot 2: PnL histogram comparison.
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(min(pnl_s.min(), pnl_a.min()), max(pnl_s.max(), pnl_a.max()), 80)
    ax.hist(pnl_s, bins=bins, density=True, alpha=0.45, label="Static", color="C0")
    ax.hist(pnl_a, bins=bins, density=True, alpha=0.45, label="Regime-aware", color="C3")
    ax.axvline(np.mean(pnl_s), color="C0", ls="--", lw=2)
    ax.axvline(np.mean(pnl_a), color="C3", ls="--", lw=2)
    ax.set_title("Terminal P&L under intraday regime shifts")
    ax.set_xlabel("P&L ($)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = FIG_DIR / f"fig07_2_pnl_hist_static_vs_regime_aware_{scenario_tag}.png"
    fig.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {f2.name}")

    # Plot 3: Inventory control over time.
    inv_s = res_static["inventory"].astype(float)
    inv_a = res_adapt["inventory"].astype(float)
    mean_abs_s = np.mean(np.abs(inv_s), axis=0)
    mean_abs_a = np.mean(np.abs(inv_a), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, mean_abs_s, lw=1.7, label="Static")
    ax.plot(times, mean_abs_a, lw=1.7, label="Regime-aware")
    ax.set_title("Inventory control under regime changes")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("E[|inventory|] (lots)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f3 = FIG_DIR / f"fig07_3_inventory_control_static_vs_regime_aware_{scenario_tag}.png"
    fig.savefig(f3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {f3.name}")

    print("Done.")
