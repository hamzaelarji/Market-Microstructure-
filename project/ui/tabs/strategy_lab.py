"""Page 2 — Strategy Lab: backtest & compare market making strategies.

Combines Quote Lab + Strategy Battle into a single backtest page.
Includes adverse selection (drift μ) as a strategy option.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general
from market_making.core.closed_form import approx_quotes
from market_making.core.intensity import C_coeff
from market_making.simulation.backtest import BacktestConfig, run_backtest

from extensions.adverse_selection import run_drift_backtest, skew_shift

from ui.styles import PALETTE, PLOT_KW, show, hero_banner, section_header
from ui.components import add_calibrated_presets, insight
from ui.loaders import load_calibrated_params, load_mid_prices

_PARAMS, _META, _HAS_REAL = load_calibrated_params()
_ALL_PRESETS = add_calibrated_presets(_PARAMS)

STRATEGY_COLORS = {
    "Optimal": PALETTE[0],
    "Naive": PALETTE[1],
    "Closed-Form": PALETTE[3],
    "Informed (drift)": PALETTE[4],
    "Uninformed (drift)": PALETTE[5],
}


def render():
    hero_banner("⚔️", "Strategy Lab",
                "Backtest & compare market making strategies — including adverse selection")

    # ══════════════════════════════════════════════════════════
    #  CONFIG PANEL
    # ══════════════════════════════════════════════════════════
    section_header("CONFIGURATION")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        preset = st.selectbox("Asset", list(_ALL_PRESETS.keys()), key="lab_pre")
    params = _ALL_PRESETS[preset]
    with c2:
        gamma = st.number_input("γ", value=0.01, format="%.4g", key="lab_g")
    with c3:
        T_val = st.number_input("T (s)", value=3600.0, step=600.0, key="lab_T")
    with c4:
        N_sim = st.slider("N_sim", 100, 3000, 500, 100, key="lab_nsim")
    with c5:
        maker_fee_bps = st.number_input("Fee (bps)", value=1.0, step=0.5, key="lab_fee")

    maker_fee = maker_fee_bps / 10_000.0

    # ══════════════════════════════════════════════════════════
    #  STRATEGY SELECTOR
    # ══════════════════════════════════════════════════════════
    section_header("STRATEGIES")

    sc1, sc2 = st.columns([3, 2])
    with sc1:
        strats = st.multiselect(
            "Select strategies to compare",
            ["Optimal", "Naive", "Closed-Form", "Informed (drift)", "Uninformed (drift)"],
            default=["Optimal", "Naive"],
            key="lab_strats",
        )
    with sc2:
        mu = 0.0
        has_drift = any("drift" in s for s in strats)
        if has_drift:
            sigma = params["sigma"]
            mu = st.slider("μ (drift $/s)", min_value=-3.0 * sigma,
                           max_value=3.0 * sigma, value=sigma * 0.5,
                           step=sigma * 0.1, format="%.6g", key="lab_mu")
            st.caption(f"Skew shift = μ/(γσ²) · Δ = {skew_shift(mu, gamma, sigma) * params['Delta']:.4g}")

    if not strats:
        st.warning("Select at least one strategy.")
        return

    # ══════════════════════════════════════════════════════════
    #  QUOTE EXPLORER (pre-compute, no MC needed)
    # ══════════════════════════════════════════════════════════
    show_quotes = st.checkbox("Show quote explorer", value=True, key="lab_quotes")

    if show_quotes:
        _render_quotes(params, gamma, T_val, mu, strats)

    # ══════════════════════════════════════════════════════════
    #  RUN BACKTESTS
    # ══════════════════════════════════════════════════════════
    run_btn = st.button("▶ Run Backtest", type="primary", key="lab_run",
                        use_container_width=True)

    if run_btn:
        _run_backtests(params, gamma, T_val, N_sim, maker_fee, mu, strats)


# ──────────────────────────────────────────────────────────────
#  Quote Explorer (no MC, just ODE/CF quotes)
# ──────────────────────────────────────────────────────────────

@st.fragment
def _render_quotes(params, gamma, T_val, mu, strats):
    section_header("QUOTE EXPLORER")

    Q = int(params["Q"])
    lots = np.arange(-Q, Q + 1)
    n_arr = np.arange(-Q + 1, Q)
    sigma = params["sigma"]
    k = params["k"]
    Delta = params["Delta"]

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=["δ^bid(0, n)", "Spread(0, n)", "Skew(0, n)"])

    xi = gamma
    N_t = max(300, int(T_val))

    for strat in strats:
        color = STRATEGY_COLORS.get(strat, PALETTE[6])

        if strat == "Optimal":
            sol = solve_general(params, gamma, T_val, xi=xi, N_t=N_t)
            db, da = sol["delta_bid"][0, :], sol["delta_ask"][0, :]
            x = lots
        elif strat == "Closed-Form":
            db, da = approx_quotes(n_arr, params, gamma, xi=xi)
            x = n_arr
        elif strat == "Naive":
            sol = solve_general(params, gamma, T_val, xi=xi, N_t=N_t)
            half = sol["delta_bid"][0, Q]
            db = np.full_like(lots, half, dtype=float)
            da = np.full_like(lots, half, dtype=float)
            x = lots
        elif strat == "Informed (drift)":
            sol = solve_general(params, gamma, T_val, xi=xi, N_t=N_t)
            shift = skew_shift(mu, gamma, sigma) * Delta
            db = sol["delta_bid"][0, :] - shift
            da = sol["delta_ask"][0, :] + shift
            x = lots
        elif strat == "Uninformed (drift)":
            sol = solve_general(params, gamma, T_val, xi=xi, N_t=N_t)
            db, da = sol["delta_bid"][0, :], sol["delta_ask"][0, :]
            x = lots
        else:
            continue

        m = np.isfinite(db) if hasattr(db, '__len__') else np.ones(len(x), dtype=bool)
        fig.add_trace(go.Scatter(x=x[m], y=db[m], mode="lines+markers", name=strat,
            line=dict(color=color, width=2), marker=dict(size=5, color=color),
            legendgroup=strat), row=1, col=1)

        spread = db + da
        m_s = np.isfinite(spread)
        fig.add_trace(go.Scatter(x=x[m_s], y=spread[m_s], mode="lines+markers",
            name=strat, line=dict(color=color, width=2), marker=dict(size=5, color=color),
            legendgroup=strat, showlegend=False), row=1, col=2)

        skew = db - da
        m_k = np.isfinite(skew)
        fig.add_trace(go.Scatter(x=x[m_k], y=skew[m_k], mode="lines+markers",
            name=strat, line=dict(color=color, width=2), marker=dict(size=5, color=color),
            legendgroup=strat, showlegend=False), row=1, col=3)

    for c in (1, 2, 3):
        fig.update_xaxes(title_text="n (lots)", row=1, col=c)
    fig.update_layout(**PLOT_KW, height=350,
                      legend=dict(orientation="h", yanchor="bottom", y=1.05))
    show(fig)


# ──────────────────────────────────────────────────────────────
#  Backtest Runner
# ──────────────────────────────────────────────────────────────

def _run_backtests(params, gamma, T_val, N_sim, maker_fee, mu, strats):
    section_header("BACKTEST RESULTS")

    results = {}
    N_t = max(300, int(T_val))

    with st.spinner("Running backtests..."):
        for strat in strats:
            if strat in ("Optimal", "Naive", "Closed-Form"):
                strat_key = {"Optimal": "optimal", "Naive": "naive",
                             "Closed-Form": "closed_form"}[strat]
                cfg = BacktestConfig(params=params, gamma=gamma, T=T_val,
                                     N_t=N_t, N_sim=N_sim, seed=42,
                                     strategy=strat_key, maker_fee=maker_fee)
                res = run_backtest(cfg)
                results[strat] = {
                    "pnl": res.pnl,
                    "inventory": res.inventory,
                    "mtm": res.mtm,
                    "price": res.price,
                    "times": res.times,
                    "n_bid_fills": res.n_bid_fills,
                    "n_ask_fills": res.n_ask_fills,
                    "fees_paid": res.fees_paid,
                    "mean_pnl": res.mean_pnl,
                    "std_pnl": res.std_pnl,
                    "sharpe": res.sharpe,
                    "mean_fills": res.mean_fills,
                    "mean_abs_inv": res.mean_abs_inventory,
                    "strategy": strat,
                }

            elif strat == "Informed (drift)":
                r = run_drift_backtest(params, gamma, T_val, mu, N_sim=N_sim,
                                       N_t=N_t, strategy="informed", seed=42,
                                       maker_fee=maker_fee)
                r["strategy"] = strat
                results[strat] = r

            elif strat == "Uninformed (drift)":
                r = run_drift_backtest(params, gamma, T_val, mu, N_sim=N_sim,
                                       N_t=N_t, strategy="uninformed", seed=42,
                                       maker_fee=maker_fee)
                r["strategy"] = strat
                results[strat] = r

    if not results:
        st.warning("No results.")
        return

    # ── PnL Histogram ──
    fig_pnl = go.Figure()
    for name, data in results.items():
        fig_pnl.add_trace(go.Histogram(
            x=data["pnl"], name=name, opacity=0.6,
            marker_color=STRATEGY_COLORS.get(name, PALETTE[6]), nbinsx=50))
    fig_pnl.update_layout(**PLOT_KW, height=380, barmode="overlay",
                          xaxis_title="Terminal PnL ($)", yaxis_title="Count",
                          title="PnL Distribution")
    show(fig_pnl)

    # ── Cumulative MtM ──
    fig_mtm = go.Figure()
    for name, data in results.items():
        if "mtm" in data and data["mtm"].ndim == 2:
            mean_mtm = np.mean(data["mtm"], axis=0)
            times = data.get("times", np.linspace(0, T_val, len(mean_mtm)))
            fig_mtm.add_trace(go.Scatter(
                x=times, y=mean_mtm, mode="lines", name=name,
                line=dict(color=STRATEGY_COLORS.get(name, PALETTE[6]), width=2)))
    fig_mtm.update_layout(**PLOT_KW, height=320,
                          xaxis_title="t (s)", yaxis_title="Mean MtM ($)",
                          title="Cumulative Mark-to-Market")
    show(fig_mtm)

    # ── Inventory comparison ──
    fig_inv = go.Figure()
    for name, data in results.items():
        if "inventory" in data and data["inventory"].ndim == 2:
            mean_abs = np.mean(np.abs(data["inventory"]), axis=0)
            times = data.get("times", np.linspace(0, T_val, len(mean_abs)))
            fig_inv.add_trace(go.Scatter(
                x=times, y=mean_abs, mode="lines", name=name,
                line=dict(color=STRATEGY_COLORS.get(name, PALETTE[6]), width=2)))
    fig_inv.update_layout(**PLOT_KW, height=280,
                          xaxis_title="t (s)", yaxis_title="E[|inventory|]",
                          title="Inventory Control")
    show(fig_inv)

    # ── Metrics Table ──
    section_header("METRICS")
    rows = []
    for name, data in results.items():
        std = data.get("std_pnl", float(np.std(data["pnl"])))
        mean = data.get("mean_pnl", float(np.mean(data["pnl"])))
        sharpe = data.get("sharpe", mean / max(std, 1e-12))
        fills = data.get("mean_fills", 0)
        abs_inv = data.get("mean_abs_inv", 0)
        fees = float(np.mean(data.get("fees_paid", np.zeros(1))))

        rows.append({
            "Strategy": name,
            "E[PnL]": f"{mean:.2f}",
            "Std[PnL]": f"{std:.2f}",
            "Sharpe": f"{sharpe:.3f}",
            "Fills": f"{fills:.0f}",
            "E[|inv_T|]": f"{abs_inv:.2f}",
            "Fees": f"{fees:.2f}",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── Insights ──
    if "Informed (drift)" in results and "Uninformed (drift)" in results:
        s_inf = results["Informed (drift)"]["sharpe"]
        s_unf = results["Uninformed (drift)"]["sharpe"]
        insight(f"Informed Sharpe: {s_inf:.3f} vs Uninformed: {s_unf:.3f}. "
                "The drift-aware MM avoids adverse selection by adjusting its skew.", "🎯")

    if "Optimal" in results and "Naive" in results:
        insight("Optimal has higher Sharpe/CE despite lower mean PnL — "
                "it maximises CARA utility, not expected wealth.", "💡")
