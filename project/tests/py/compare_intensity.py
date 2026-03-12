"""compare_intensity.py
======================
Side-by-side comparison of:
  • Original exponential Poisson intensity  (intensity.py)
  • New Hawkes-based intensity               (intensity_hawkes.py)

No external project imports needed — params are defined inline.
Run with:
    python compare_intensity.py
Produces:  project/figures/comparison_plots.png
"""

import sys
from pathlib import Path

# Ensure script runs from any cwd: resolve paths relative to this file
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = PROJECT_DIR / "figures"
sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "asset"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── project modules ────────────────────────────────────────────────────────
from intensity import Lambda, H_val, delta_star, C_coeff
from intensity_hawkes import (
    softplus, lambda_hawkes, lambda_hawkes_linear,
    fill_prob_from_intensity, HawkesState, DEFAULT_HAWKES_CFG
)

# ── parameters (replicate asset.params inline) ────────────────────────────
GAMMA = 0.1

IG = dict(sigma=0.3/100, A=140.0, k=1.5, Delta=0.01, Q=4)
HY = dict(sigma=1.5/100, A=40.0,  k=0.5, Delta=0.05, Q=4)

HAWKES_CFG = dict(beta=10.0, alpha_self=2.0, alpha_cross=0.5)

# ── colour palette ─────────────────────────────────────────────────────────
C_POISSON = "#2563EB"   # blue  — original
C_HAWKES  = "#DC2626"   # red   — Hawkes (no excitation)
C_EXCITED = "#16A34A"   # green — Hawkes (with excitation)
C_LINEAR  = "#9333EA"   # purple — linear Hawkes (legacy)

# ══════════════════════════════════════════════════════════════════════════
# 1. ΛΛΛΛ  Intensity vs δ
# ══════════════════════════════════════════════════════════════════════════
def plot_intensity_vs_delta(ax, params, title):
    delta = np.linspace(0, 3.0, 300)
    A, k  = params["A"], params["k"]

    lam_poisson   = Lambda(delta, A, k)                       # A·exp(-kδ)
    lam_h0        = lambda_hawkes(lam_poisson, 0.0)           # softplus(Λ)
    lam_h_excited = lambda_hawkes(lam_poisson, 2.0)           # softplus(Λ + y)
    lam_h_inhib   = lambda_hawkes(lam_poisson, -0.5)          # softplus(Λ - y)  inhibition

    ax.semilogy(delta, lam_poisson,   color=C_POISSON, lw=2,   label="Poisson: Λ(δ)")
    ax.semilogy(delta, lam_h0,        color=C_HAWKES,  lw=2,   ls="--",
                label="Hawkes: softplus(Λ, y=0)")
    ax.semilogy(delta, lam_h_excited, color=C_EXCITED, lw=2,   ls="-.",
                label="Hawkes: softplus(Λ, y=+2)  excitation")
    ax.semilogy(delta, lam_h_inhib,   color=C_LINEAR,  lw=1.5, ls=":",
                label="Hawkes: softplus(Λ, y=−0.5) inhibition")

    ax.set_xlabel("Quote distance δ")
    ax.set_ylabel("Intensity λ(δ)  [log scale]")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.set_xlim(0, 3)
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════
# 2. Fill probability vs δ
# ══════════════════════════════════════════════════════════════════════════
def plot_fill_prob(ax, params, title, dt=1.0):
    delta = np.linspace(0, 3.0, 300)
    A, k  = params["A"], params["k"]

    lam    = Lambda(delta, A, k)
    p_old  = np.minimum(lam * dt, 1.0)                  # first-order approx (legacy)
    p_new  = fill_prob_from_intensity(lam, dt)           # exact Poisson
    p_hawk = fill_prob_from_intensity(
                 lambda_hawkes(lam, 2.0), dt)            # Hawkes + excitation

    ax.plot(delta, p_old,  color=C_POISSON, lw=2,   label=f"Poisson: λ·dt  (dt={dt}s)")
    ax.plot(delta, p_new,  color=C_HAWKES,  lw=2,   ls="--",
            label="Poisson: 1−exp(−λ·dt)  (exact)")
    ax.plot(delta, p_hawk, color=C_EXCITED, lw=2,   ls="-.",
            label="Hawkes exact, y=+2")

    ax.set_xlabel("Quote distance δ")
    ax.set_ylabel("P(fill per step)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════
# 3. Excitation state dynamics over time
# ══════════════════════════════════════════════════════════════════════════
def plot_excitation_dynamics(ax):
    dt   = 0.01
    T    = 5.0
    N    = int(T / dt)
    time = np.arange(N) * dt

    cfg   = HAWKES_CFG
    state = HawkesState(cfg)
    mu    = 1.0  # fixed baseline (flat quotes)

    y_bid_hist  = np.zeros(N)
    y_ask_hist  = np.zeros(N)
    lam_bid_hist= np.zeros(N)
    lam_ask_hist= np.zeros(N)

    rng = np.random.default_rng(42)
    # Inject a few synthetic fills at fixed times
    fill_bid_times = {0.5, 1.0, 1.01, 1.02, 3.0}
    fill_ask_times = {2.0, 2.01}

    for i in range(N):
        t = time[i]
        y_bid_hist[i]   = state.y_bid
        y_ask_hist[i]   = state.y_ask
        lam_bid_hist[i] = state.lambda_bid(mu)
        lam_ask_hist[i] = state.lambda_ask(mu)
        fb = any(abs(t - s) < dt/2 for s in fill_bid_times)
        fa = any(abs(t - s) < dt/2 for s in fill_ask_times)
        state.step(dt, fb, fa)

    ax.plot(time, lam_bid_hist, color=C_POISSON, lw=1.5, label="λ_bid(t)")
    ax.plot(time, lam_ask_hist, color=C_HAWKES,  lw=1.5, label="λ_ask(t)", ls="--")
    ax.axhline(softplus(mu), color="gray", lw=1, ls=":", label="softplus(baseline)")

    for s in sorted(fill_bid_times):
        ax.axvline(s, color=C_POISSON, alpha=0.3, lw=1)
    for s in sorted(fill_ask_times):
        ax.axvline(s, color=C_HAWKES,  alpha=0.3, lw=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity λ(t)")
    ax.set_title("Excitation dynamics  (β=10, α_self=2, α_cross=0.5)\n"
                 "blue ticks = bid fills,  red ticks = ask fills")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════
# 4. Softplus vs Linear: behaviour in inhibition regime
# ══════════════════════════════════════════════════════════════════════════
def plot_softplus_vs_linear(ax):
    y_vals = np.linspace(-5, 5, 400)
    mu     = 0.5

    sp_vals  = lambda_hawkes(mu, y_vals)
    lin_vals = lambda_hawkes_linear(mu, y_vals)
    base     = np.full_like(y_vals, mu)

    ax.plot(y_vals, sp_vals,  color=C_HAWKES,  lw=2, label="Softplus: φ(Λ + y)")
    ax.plot(y_vals, lin_vals, color=C_LINEAR,  lw=2, ls="--",
            label="Linear clipped: max(Λ + y, 0)")
    ax.plot(y_vals, base,     color=C_POISSON, lw=1, ls=":", label="Λ (baseline)")

    ax.axvline(0, color="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.fill_between(y_vals, sp_vals, lin_vals,
                    where=(y_vals < -mu), alpha=0.15, color=C_EXCITED,
                    label="Inhibition gap (Softplus > 0 where linear=0)")

    ax.set_xlabel("Excitation state y(t)")
    ax.set_ylabel("Intensity λ")
    ax.set_title(f"Softplus vs Linear Hawkes  (Λ={mu})\n"
                  "Softplus models inhibition without hitting zero")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════
# 5. Simulated fill counts: Poisson vs Hawkes (Monte Carlo)
# ══════════════════════════════════════════════════════════════════════════
def simulate_fills(params, cfg, T=60.0, dt=0.001, n_paths=500,
                   delta_bid=1.0, delta_ask=1.0, xi=0.0):
    """Run n_paths simulations and collect fill counts per path.
    Vectorised over paths for speed.
    """
    A, k   = params["A"], params["k"]
    N      = int(T / dt)
    beta   = cfg["beta"]
    a_self = cfg["alpha_self"]
    a_cros = cfg["alpha_cross"]
    decay  = np.exp(-beta * dt)

    mu_bid = Lambda(delta_bid, A, k)
    mu_ask = Lambda(delta_ask, A, k)

    rng = np.random.default_rng(0)

    # ── Poisson (vectorised, no loop needed) ──
    pb = fill_prob_from_intensity(mu_bid, dt)
    pa = fill_prob_from_intensity(mu_ask, dt)
    fills_poisson = np.column_stack([
        rng.binomial(N, pb, n_paths).astype(float),
        rng.binomial(N, pa, n_paths).astype(float),
    ])

    # ── Hawkes (vectorised over paths, loop over time) ──
    y_bid = np.zeros(n_paths)
    y_ask = np.zeros(n_paths)
    fb_count = np.zeros(n_paths)
    fa_count = np.zeros(n_paths)

    for _ in range(N):
        lb = softplus(mu_bid + y_bid)
        la = softplus(mu_ask + y_ask)
        fb = rng.random(n_paths) < fill_prob_from_intensity(lb, dt)
        fa = rng.random(n_paths) < fill_prob_from_intensity(la, dt)
        fb_count += fb
        fa_count += fa
        # decay then jump
        y_bid = y_bid * decay + fb * a_self  + fa * a_cros
        y_ask = y_ask * decay + fa * a_self  + fb * a_cros

    fills_hawkes = np.column_stack([fb_count, fa_count])
    return fills_poisson, fills_hawkes


def plot_fill_distribution(ax, params, cfg, title):
    fp, fh = simulate_fills(params, cfg)

    bid_p, ask_p = fp[:, 0], fp[:, 1]
    bid_h, ask_h = fh[:, 0], fh[:, 1]

    bins = np.linspace(
        min(bid_p.min(), bid_h.min()) * 0.9,
        max(bid_p.max(), bid_h.max()) * 1.1,
        30
    )

    ax.hist(bid_p, bins=bins, color=C_POISSON, alpha=0.5,
            label=f"Poisson bid fills  μ={bid_p.mean():.0f}")
    ax.hist(bid_h, bins=bins, color=C_EXCITED, alpha=0.5,
            label=f"Hawkes bid fills   μ={bid_h.mean():.0f}")

    ax.axvline(bid_p.mean(), color=C_POISSON, lw=2, ls="--")
    ax.axvline(bid_h.mean(), color=C_EXCITED, lw=2, ls="--")

    ax.set_xlabel("Total bid fills per path  (T=600s, dt=1s)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════
# 6. Summary table  (printed to stdout + rendered as text axes)
# ══════════════════════════════════════════════════════════════════════════
def make_summary_table(ax):
    fp_ig, fh_ig = simulate_fills(IG, HAWKES_CFG)
    fp_hy, fh_hy = simulate_fills(HY, HAWKES_CFG)

    rows = [
        ["Metric", "Poisson IG", "Hawkes IG", "Poisson HY", "Hawkes HY"],
        ["Mean bid fills / path",
         f"{fp_ig[:,0].mean():.1f}", f"{fh_ig[:,0].mean():.1f}",
         f"{fp_hy[:,0].mean():.1f}", f"{fh_hy[:,0].mean():.1f}"],
        ["Std bid fills / path",
         f"{fp_ig[:,0].std():.1f}",  f"{fh_ig[:,0].std():.1f}",
         f"{fp_hy[:,0].std():.1f}",  f"{fh_hy[:,0].std():.1f}"],
        ["Mean ask fills / path",
         f"{fp_ig[:,1].mean():.1f}", f"{fh_ig[:,1].mean():.1f}",
         f"{fp_hy[:,1].mean():.1f}", f"{fh_hy[:,1].mean():.1f}"],
        ["Hawkes / Poisson ratio bid", "1.00",
         f"{fh_ig[:,0].mean()/fp_ig[:,0].mean():.2f}",
         "1.00",
         f"{fh_hy[:,0].mean()/fp_hy[:,0].mean():.2f}"],
    ]

    ax.axis("off")
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.6)
    # Header row style
    for j in range(5):
        tbl[(0, j)].set_facecolor("#1E3A5F")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Alternate row shading
    for i in range(1, len(rows)):
        col = "#F0F4FF" if i % 2 == 0 else "white"
        for j in range(5):
            tbl[(i, j)].set_facecolor(col)

    ax.set_title("Monte Carlo summary: Poisson vs Hawkes intensity\n"
                 "(500 paths × 60s, δ^b=δ^a=1.0, dt=0.001s)", pad=10)

    # Print to console too
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    for r in rows:
        print("  " + "  |  ".join(f"{c:>22}" for c in r))


# ══════════════════════════════════════════════════════════════════════════
# Assemble figure
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("Running comparison...  (Monte Carlo may take ~20s)")

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(
        "Exponential Poisson Intensity  vs  Hawkes-Based Intensity\n"
        "Lalor & Swishchuk (2025) — Guéant (2017) framework",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.42, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.94, bottom=0.04)

    # Row 0: intensity shape (IG + HY) + excitation dynamics
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])

    plot_intensity_vs_delta(ax00, IG, "Intensity Λ(δ) — IG (A=140, k=1.5)")
    plot_intensity_vs_delta(ax01, HY, "Intensity Λ(δ) — HY (A=40, k=0.5)")
    plot_excitation_dynamics(ax02)

    # Row 1: fill probability + softplus/linear + fill distribution IG
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    plot_fill_prob(ax10, IG, "Fill probability vs δ — IG  (dt=0.001s)", dt=0.001)
    plot_softplus_vs_linear(ax11)
    plot_fill_distribution(ax12, IG, HAWKES_CFG,
                           "Fill count distribution — IG\n(500 paths, T=60s, dt=0.001s)")

    # Row 2: fill distribution HY + summary table (spans 2 cols)
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1:])

    plot_fill_distribution(ax20, HY, HAWKES_CFG,
                           "Fill count distribution — HY\n(500 paths, T=60s, dt=0.001s)")
    make_summary_table(ax21)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "comparison_plots.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved → {out}")
    return out


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "=" * 60 + "\nERROR\n" + "=" * 60)
        traceback.print_exc()
        sys.exit(1)
