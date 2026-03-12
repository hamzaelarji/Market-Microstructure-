"""Page 5 — Adverse Selection: What if prices have drift?

Extension: dS = μdt + σdW.  Compares informed vs uninformed market makers.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general
from market_making.core.closed_form import approx_quotes
from extensions.adverse_selection import (
    skew_shift, run_drift_backtest, sweep_drift
)
from ui.styles import PALETTE, PLOT_KW, show, hero_banner, section_header
from ui.components import param_sidebar, metrics_row, insight_box

DEFAULT_GAMMA = 0.01


def render(PARAMS, META, HAS_REAL_DATA):
    hero_banner("🎯", "Adverse Selection — Price Drift",
                "What happens when dS = μdt + σdW? Informed vs uninformed market makers.")

    with st.sidebar:
        section_header("PARAMETERS")
        cfg = param_sidebar(PARAMS, key_prefix="p5")

    params = cfg["params"]
    gamma, T_val = cfg["gamma"], cfg["T"]
    sigma = params["sigma"]

    # ── Theory ────────────────────────────────────────────────
    with st.expander("📖 Theory", expanded=False):
        st.markdown("Adding drift μ to the price process:")
        st.latex(r"dS_t = \mu \, dt + \sigma \, dW_t")
        st.markdown("The ODE gains an asymmetric term:")
        st.latex(r"\partial_t \theta_n + \tfrac{1}{2}\gamma\sigma^2 (n\Delta)^2 "
                 r"+ \gamma \mu (n\Delta) - H_\xi(\text{bid}) - H_\xi(\text{ask}) = 0")
        st.markdown("The **informed** MM adjusts quotes by the reservation-price shift:")
        st.latex(r"\text{skew shift} = \frac{\mu}{\gamma \sigma^2}")
        st.markdown("The **uninformed** MM uses the standard Guéant policy (μ=0) "
                    "and suffers **adverse selection** — buying before price drops, "
                    "selling before price rises.")

    # ── Controls ──────────────────────────────────────────────
    c1, c2 = st.columns([1, 3])
    with c1:
        mu = st.slider("μ (drift $/s)", min_value=-3.0 * sigma, max_value=3.0 * sigma,
                        value=0.0, step=sigma * 0.1, format="%.6f", key="p5_mu")
        n_sim = st.slider("N_sim", 100, 2000, 500, 100, key="p5_nsim")
        run_btn = st.button("▶ Run comparison", type="primary", key="p5_run")

    with c2:
        # ── Quote comparison ──────────────────────────────────
        section_header("QUOTE SHIFT FROM DRIFT")

        Q = int(params["Q"])
        n_arr = np.arange(-Q + 1, Q)
        db_0, da_0 = approx_quotes(n_arr, params, gamma, xi=gamma)

        shift = skew_shift(mu, gamma, sigma) * params["Delta"]
        db_mu = db_0 - shift
        da_mu = da_0 + shift

        fig_q = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                              subplot_titles=["δ^bid(0, n)", "δ^ask(0, n)"])

        fig_q.add_trace(go.Scatter(x=n_arr, y=db_0, mode="lines", name="μ = 0",
            line=dict(dash="dash", color=PALETTE[6], width=2)), row=1, col=1)
        fig_q.add_trace(go.Scatter(x=n_arr, y=db_mu, mode="lines+markers", name=f"μ = {mu:.2e}",
            line=dict(color=PALETTE[0], width=2), marker=dict(size=6, color=PALETTE[0])), row=1, col=1)

        fig_q.add_trace(go.Scatter(x=n_arr, y=da_0, mode="lines", name="μ = 0",
            line=dict(dash="dash", color=PALETTE[6], width=2), showlegend=False), row=1, col=2)
        fig_q.add_trace(go.Scatter(x=n_arr, y=da_mu, mode="lines+markers", name=f"μ = {mu:.2e}",
            line=dict(color=PALETTE[1], width=2), marker=dict(size=6, color=PALETTE[1]),
            showlegend=False), row=1, col=2)

        fig_q.update_layout(**PLOT_KW, height=320,
                            legend=dict(orientation="h", yanchor="bottom", y=1.05))
        show(fig_q)

        # ── Skew fan ─────────────────────────────────────────
        section_header("SKEW vs μ")
        mu_vals = np.linspace(-2 * sigma, 2 * sigma, 5)
        fig_sf = go.Figure()
        colors_fan = [PALETTE[1], PALETTE[5], PALETTE[6], PALETTE[4], PALETTE[0]]
        for mv, clr in zip(mu_vals, colors_fan):
            sh = skew_shift(mv, gamma, sigma) * params["Delta"]
            skew = (db_0 - sh) - (da_0 + sh)
            fig_sf.add_trace(go.Scatter(
                x=n_arr, y=skew, mode="lines", name=f"μ={mv:.2e}",
                line=dict(color=clr, width=2)))

        fig_sf.update_layout(**PLOT_KW, height=300,
                             xaxis_title="n (lots)", yaxis_title="Skew = δ^b − δ^a")
        show(fig_sf)

        # ── Monte Carlo comparison ────────────────────────────
        if run_btn:
            section_header("PnL COMPARISON: INFORMED vs UNINFORMED")

            with st.spinner("Running 3 backtests..."):
                res_inf = run_drift_backtest(params, gamma, T_val, mu, N_sim=n_sim,
                                             strategy="informed", seed=42)
                res_unf = run_drift_backtest(params, gamma, T_val, mu, N_sim=n_sim,
                                             strategy="uninformed", seed=42)
                res_naive = run_drift_backtest(params, gamma, T_val, mu, N_sim=n_sim,
                                               strategy="naive", seed=42)

            fig_pnl = go.Figure()
            for res, name, color in [
                (res_inf, "Informed", PALETTE[0]),
                (res_unf, "Uninformed", PALETTE[5]),
                (res_naive, "Naive", PALETTE[1]),
            ]:
                fig_pnl.add_trace(go.Histogram(
                    x=res["pnl"], name=name, opacity=0.6,
                    marker_color=color, nbinsx=40))

            fig_pnl.update_layout(**PLOT_KW, height=350, barmode="overlay",
                                  xaxis_title="Terminal PnL ($)", yaxis_title="Count")
            show(fig_pnl)

            metrics_row({
                "Informed Sharpe": res_inf["sharpe"],
                "Uninformed Sharpe": res_unf["sharpe"],
                "Naive Sharpe": res_naive["sharpe"],
                "Skew shift (Δ·μ/γσ²)": shift,
            })

            insight_box(f"With μ = {mu:.2e}, the uninformed MM faces adverse selection: "
                        "it systematically buys before price drops. "
                        "Even small drift (μ ~ 0.5σ) creates significant cost.")

            # ── Sharpe vs |μ| sweep ──
            section_header("SHARPE vs DRIFT MAGNITUDE")
            with st.spinner("Sweeping μ values..."):
                sweep = sweep_drift(params, gamma, T=T_val, N_sim=min(n_sim, 200), seed=42)

            fig_sw = go.Figure()
            fig_sw.add_trace(go.Scatter(
                x=sweep["mu_vals"], y=sweep["sharpe_informed"],
                mode="lines+markers", name="Informed",
                line=dict(color=PALETTE[0], width=2), marker=dict(size=6)))
            fig_sw.add_trace(go.Scatter(
                x=sweep["mu_vals"], y=sweep["sharpe_uninformed"],
                mode="lines+markers", name="Uninformed",
                line=dict(color=PALETTE[5], width=2), marker=dict(size=6)))
            fig_sw.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_sw.update_layout(**PLOT_KW, height=350,
                                 xaxis_title="μ (drift $/s)", yaxis_title="Sharpe ratio")
            show(fig_sw)
