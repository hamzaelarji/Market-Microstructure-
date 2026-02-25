"""Notebook 09 — Hawkes execution policy comparison pipeline.

Implements:
  A) Hawkes-type execution simulator (via asset.hawkes_simulator)
  B) Guéant static policy + adaptive variants
  C) Controlled comparison with metrics, plots, and statistical tests

Run examples:
  python 09_hawkes_policy_comparison.py
  python 09_hawkes_policy_comparison.py --scenario high_volatility
  python 09_hawkes_policy_comparison.py --list-scenarios
"""

from pathlib import Path
import os
import sys
import argparse
import csv

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))
from asset.experiment_sets import get_scenario, scenario_names
from asset.hawkes_simulator import simulate_hawkes_1d
from src.ode_solver_1d import solve_general

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def summarize_metrics(name, res, gamma):
    pnl = res["pnl"]
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl))
    sharpe = mu / max(sd, 1e-12)
    ce = -np.log(np.mean(np.exp(-gamma * pnl))) / gamma
    var5 = float(np.percentile(pnl, 5))
    cvar5 = float(np.mean(pnl[pnl <= var5]))
    mean_abs_invT = float(np.mean(np.abs(res["inventory"][:, -1])))
    mean_fills = float(np.mean(res["n_bid_fills"] + res["n_ask_fills"]))
    mean_turnover = float(np.mean(res["turnover"]))
    avg_lam = float(np.mean(res["avg_lambda_bid"] + res["avg_lambda_ask"]))
    return dict(
        policy=name,
        mean_pnl=mu,
        std_pnl=sd,
        sharpe=sharpe,
        certainty_equivalent=ce,
        var5=var5,
        cvar5=cvar5,
        mean_abs_invT=mean_abs_invT,
        mean_fills=mean_fills,
        mean_turnover=mean_turnover,
        avg_total_lambda=avg_lam,
    )


def paired_tests(base, challenger):
    """Paired tests on trajectory-wise P&L differences."""
    d = challenger - base
    t_stat, t_p = stats.ttest_rel(challenger, base)
    try:
        w_stat, w_p = stats.wilcoxon(d)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    return dict(
        mean_diff=float(np.mean(d)),
        median_diff=float(np.median(d)),
        ttest_p=float(t_p),
        wilcoxon_p=float(w_p) if np.isfinite(w_p) else np.nan,
    )


def print_metric_table(rows):
    headers = [
        "policy",
        "mean_pnl",
        "std_pnl",
        "sharpe",
        "certainty_equivalent",
        "var5",
        "cvar5",
        "mean_abs_invT",
        "mean_fills",
        "mean_turnover",
        "avg_total_lambda",
    ]
    print("\nMETRICS")
    print("-" * 120)
    print(" | ".join(f"{h:>18s}" for h in headers))
    print("-" * 120)
    for r in rows:
        vals = []
        for h in headers:
            v = r[h]
            if isinstance(v, str):
                vals.append(f"{v:>18s}")
            else:
                vals.append(f"{v:18.4g}")
        print(" | ".join(vals))
    print("-" * 120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hawkes execution comparison experiment")
    parser.add_argument("--scenario", default="baseline", help="Scenario from asset.experiment_sets")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--n-sim", type=int, default=500)
    parser.add_argument("--n-t", type=int, default=600)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--intensity-model", choices=["exponential", "power_law", "logistic"], default="exponential")
    parser.add_argument("--pl-alpha", type=float, default=1.8, help="Power-law tail exponent")
    parser.add_argument("--pl-delta0", type=float, default=None, help="Power-law scale in delta units")
    parser.add_argument("--log-mid", type=float, default=None, help="Logistic midpoint in delta units")
    parser.add_argument("--log-slope", type=float, default=None, help="Logistic slope in delta units")
    parser.add_argument(
        "--compare-intensity-models",
        action="store_true",
        help="Also run static-policy comparison across exponential/power_law/logistic in one figure/table.",
    )
    args = parser.parse_args()

    if args.list_scenarios:
        print("\n".join(scenario_names()))
        raise SystemExit(0)

    cfg = get_scenario(args.scenario)
    IG = cfg["IG"]
    GAMMA = cfg["GAMMA"]
    T = cfg["T"]
    tag = cfg["name"]

    print(f"Scenario: {tag}")
    print("Solving Guéant policy table...")
    sol = solve_general(IG, GAMMA, T, xi=GAMMA, N_t=args.n_t)

    # Tuned defaults from quick grid search to avoid weak adaptive effects.
    hawkes_cfg = dict(
        beta=4.0,
        jump_self=5.0e-4,
        jump_cross=2.0e-4,
        init_excitation=1.5e-4,
    )
    print("Hawkes config:", hawkes_cfg)
    def build_intensity_model(kind):
        if kind == "power_law":
            delta0 = args.pl_delta0 if args.pl_delta0 is not None else (1.0 / max(IG["k"], 1e-12))
            return dict(kind="power_law", alpha=args.pl_alpha, delta0=delta0)
        if kind == "logistic":
            dmid = args.log_mid if args.log_mid is not None else (1.0 / max(IG["k"], 1e-12))
            slope = args.log_slope if args.log_slope is not None else (0.35 / max(IG["k"], 1e-12))
            return dict(kind="logistic", delta_mid=dmid, slope=slope)
        return dict(kind="exponential")

    if args.intensity_model == "power_law":
        delta0 = args.pl_delta0 if args.pl_delta0 is not None else (1.0 / max(IG["k"], 1e-12))
        intensity_model = dict(kind="power_law", alpha=args.pl_alpha, delta0=delta0)
    elif args.intensity_model == "logistic":
        dmid = args.log_mid if args.log_mid is not None else (1.0 / max(IG["k"], 1e-12))
        slope = args.log_slope if args.log_slope is not None else (0.35 / max(IG["k"], 1e-12))
        intensity_model = dict(kind="logistic", delta_mid=dmid, slope=slope)
    else:
        intensity_model = dict(kind="exponential")
    print("Intensity model:", intensity_model)
    if intensity_model["kind"] == "power_law":
        model_desc = f"power_law (alpha={intensity_model['alpha']:.3g}, delta0={intensity_model['delta0']:.3e})"
    elif intensity_model["kind"] == "logistic":
        model_desc = (
            f"logistic (delta_mid={intensity_model['delta_mid']:.3e}, "
            f"slope={intensity_model['slope']:.3e})"
        )
    else:
        model_desc = "exponential"

    policy_specs = [
        ("gueant_static", {}),
        ("inv_skew", {"eta_inv": 0.40}),
        ("hawkes_aware", {"eta_inv": 0.40, "eta_exc": 1.20, "eta_wide": 0.45}),
    ]

    # Controlled comparison: same random seed per policy for paired tests.
    results = {}
    for name, pcfg in policy_specs:
        print(f"Simulating policy: {name}")
        results[name] = simulate_hawkes_1d(
            sol=sol,
            params=IG,
            T=T,
            hawkes_cfg=hawkes_cfg,
            intensity_model=intensity_model,
            policy_name=name,
            policy_cfg=pcfg,
            N_sim=args.n_sim,
            seed=args.seed,
        )

    metric_rows = [summarize_metrics(name, results[name], GAMMA) for name, _ in policy_specs]
    print_metric_table(metric_rows)

    # Paired statistical tests vs Guéant baseline.
    baseline_pnl = results["gueant_static"]["pnl"]
    test_rows = []
    for alt in ["inv_skew", "hawkes_aware"]:
        trow = paired_tests(baseline_pnl, results[alt]["pnl"])
        trow["vs_policy"] = alt
        test_rows.append(trow)

    print("\nPAIRED TESTS vs gueant_static")
    for r in test_rows:
        print(
            f"{r['vs_policy']:>12s}: mean_diff={r['mean_diff']:+.2f}, median_diff={r['median_diff']:+.2f}, "
            f"ttest_p={r['ttest_p']:.3g}, wilcoxon_p={r['wilcoxon_p']:.3g}"
        )

    # Save metrics CSV
    model_tag = intensity_model["kind"]
    metrics_csv = FIG_DIR / f"table09_metrics_{tag}_{model_tag}.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Saved {metrics_csv.name}")

    tests_csv = FIG_DIR / f"table09_tests_{tag}_{model_tag}.csv"
    with tests_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["vs_policy", "mean_diff", "median_diff", "ttest_p", "wilcoxon_p"])
        writer.writeheader()
        writer.writerows(test_rows)
    print(f"Saved {tests_csv.name}")

    # Plot 1: PnL distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    pmins = [results[k]["pnl"].min() for k in results]
    pmaxs = [results[k]["pnl"].max() for k in results]
    bins = np.linspace(min(pmins), max(pmaxs), 80)
    for name, color in [("gueant_static", "C0"), ("inv_skew", "C1"), ("hawkes_aware", "C3")]:
        ax.hist(results[name]["pnl"], bins=bins, density=True, alpha=0.35, label=name, color=color)
        ax.axvline(np.mean(results[name]["pnl"]), color=color, ls="--", lw=1.4)
    ax.set_title(f"Terminal P&L under Hawkes execution [{model_desc}]")
    ax.set_xlabel("P&L ($)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = FIG_DIR / f"fig09_1_pnl_distributions_{tag}_{model_tag}.png"
    fig.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f1.name}")

    # Plot 2: inventory risk profile over time + deltas vs static
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    times = results["gueant_static"]["times"]
    mean_abs_map = {}
    for name, color in [("gueant_static", "C0"), ("inv_skew", "C1"), ("hawkes_aware", "C3")]:
        mean_abs_inv = np.mean(np.abs(results[name]["inventory"]), axis=0)
        mean_abs_map[name] = mean_abs_inv
        axes[0].plot(times, mean_abs_inv, lw=1.6, label=name, color=color)
    axes[0].set_title(f"Mean absolute inventory over time [{model_desc}]")
    axes[0].set_ylabel("E[|inventory|] (lots)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    base_inv = mean_abs_map["gueant_static"]
    axes[1].plot(times, mean_abs_map["inv_skew"] - base_inv, lw=1.6, label="inv_skew - static", color="C1")
    axes[1].plot(times, mean_abs_map["hawkes_aware"] - base_inv, lw=1.6, label="hawkes_aware - static", color="C3")
    axes[1].axhline(0.0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Delta E[|inventory|]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    f2 = FIG_DIR / f"fig09_2_inventory_profiles_{tag}_{model_tag}.png"
    fig.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f2.name}")

    # Plot 3: paired PnL difference violin+jitter vs baseline (more informative than flat boxplots)
    d1 = results["inv_skew"]["pnl"] - baseline_pnl
    d2 = results["hawkes_aware"]["pnl"] - baseline_pnl
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot([d1, d2], positions=[1, 2], showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.3)
    # jittered sample points (subsample for readability)
    rng = np.random.default_rng(123)
    idx1 = rng.choice(len(d1), size=min(200, len(d1)), replace=False)
    idx2 = rng.choice(len(d2), size=min(200, len(d2)), replace=False)
    x1 = 1.0 + 0.06 * rng.standard_normal(len(idx1))
    x2 = 2.0 + 0.06 * rng.standard_normal(len(idx2))
    ax.scatter(x1, d1[idx1], s=8, alpha=0.2, color="C1")
    ax.scatter(x2, d2[idx2], s=8, alpha=0.2, color="C3")
    model_tick = intensity_model["kind"]
    ax.set_xticks(
        [1, 2],
        [f"inv_skew - static\n[{model_tick}]", f"hawkes_aware - static\n[{model_tick}]"],
    )
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_title(f"Paired trajectory P&L differences [{model_desc}]")
    ax.set_ylabel("Delta P&L ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f3 = FIG_DIR / f"fig09_3_paired_differences_{tag}_{model_tag}.png"
    fig.savefig(f3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f3.name}")

    # Print non-zero difference proportions for interpretability.
    nz1 = float(np.mean(np.abs(d1) > 1e-12))
    nz2 = float(np.mean(np.abs(d2) > 1e-12))
    print(f"Non-zero paired differences: inv_skew={100*nz1:.1f}%  hawkes_aware={100*nz2:.1f}%")

    print("Done.")

    # Optional: isolate intensity-model effect for the static policy.
    if args.compare_intensity_models:
        print("\nRunning static policy comparison across intensity models...")
        model_order = ["exponential", "power_law", "logistic"]
        static_rows = []
        static_res = {}
        for mk in model_order:
            im = build_intensity_model(mk)
            r = simulate_hawkes_1d(
                sol=sol,
                params=IG,
                T=T,
                hawkes_cfg=hawkes_cfg,
                intensity_model=im,
                policy_name="gueant_static",
                policy_cfg={},
                N_sim=args.n_sim,
                seed=args.seed,
            )
            static_res[mk] = r
            row = summarize_metrics(f"static_{mk}", r, GAMMA)
            row["intensity_model"] = mk
            static_rows.append(row)

        static_csv = FIG_DIR / f"table09_static_intensity_comparison_{tag}.csv"
        with static_csv.open("w", newline="") as f:
            fields = ["intensity_model"] + [k for k in static_rows[0].keys() if k not in ("policy",)]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in static_rows:
                out = {"intensity_model": row["intensity_model"]}
                for k in fields[1:]:
                    out[k] = row.get(k, "")
                writer.writerow(out)
        print(f"Saved {static_csv.name}")

        # Plot static PnL distributions across models.
        fig, ax = plt.subplots(figsize=(10, 6))
        pmins = [static_res[k]["pnl"].min() for k in model_order]
        pmaxs = [static_res[k]["pnl"].max() for k in model_order]
        bins = np.linspace(min(pmins), max(pmaxs), 80)
        colors = {"exponential": "C0", "power_law": "C2", "logistic": "C4"}
        for mk in model_order:
            p = static_res[mk]["pnl"]
            ax.hist(p, bins=bins, density=True, alpha=0.35, label=f"static [{mk}]", color=colors[mk])
            ax.axvline(np.mean(p), color=colors[mk], ls="--", lw=1.5)
        ax.set_title("Static policy: P&L across intensity models")
        ax.set_xlabel("P&L ($)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        f4 = FIG_DIR / f"fig09_4_static_across_intensity_models_{tag}.png"
        fig.savefig(f4, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {f4.name}")
