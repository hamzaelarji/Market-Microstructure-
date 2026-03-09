"""ui/tabs/sensitivity.py — Tab 2: γ Sensitivity sweep."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general


def render_sensitivity(PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
                       load_mid_prices, run_quick_mc):
    """Render Tab 2 — γ Sensitivity: sweep γ → spread, skew, fill rate."""
    SYMBOLS = list(PARAMS.keys())

    st.caption(
        "How does risk aversion γ control the market maker? "
        "Higher γ → wider spreads, steeper skew, fewer fills."
    )

    c_s1, c_s2 = st.columns([1, 3])

    with c_s1:
        g1_sym = st.selectbox("Symbol", SYMBOLS, key="g1_sym")
        g1_params = PARAMS[g1_sym]
        g1_T = st.number_input("T (s)", value=3600.0, step=600.0, key="g1_T")
        g1_nsim = st.slider("MC paths", 100, 2000, 400, step=100, key="g1_nsim")
        g1_lo = st.number_input("γ min", value=0.001, format="%.4f", key="g1_lo")
        g1_hi = st.number_input("γ max", value=0.10, format="%.4f", key="g1_hi")
        g1_n = st.slider("# points", 5, 15, 10, key="g1_n")

        run_g1 = st.button("Run γ sweep", type="primary", key="g1_run")

    with c_s2:
        if run_g1:
            gammas = np.geomspace(g1_lo, g1_hi, g1_n)
            N_t_g1 = max(300, int(g1_T))
            results_g1 = []
            prog = st.progress(0, text="Sweeping γ …")

            mid_real_g1 = load_mid_prices(g1_sym)
            mid_arr_g1 = None
            if mid_real_g1 is not None and len(mid_real_g1) > N_t_g1:
                mid_arr_g1 = mid_real_g1.values[:N_t_g1 + 1]

            for idx, g in enumerate(gammas):
                prog.progress((idx + 1) / len(gammas),
                              text=f"γ = {g:.4f}  ({idx+1}/{len(gammas)})")
                try:
                    sol = solve_general(g1_params, g, g1_T, xi=g, N_t=N_t_g1)
                    db0 = sol["delta_bid"][0, :]
                    da0 = sol["delta_ask"][0, :]

                    sp = db0 + da0
                    finite_sp = sp[np.isfinite(sp)]
                    mean_spread = float(np.mean(finite_sp)) if len(finite_sp) > 0 else np.nan

                    sk = db0 - da0
                    lots_s = sol["lots"]
                    mask = np.isfinite(sk)
                    if mask.sum() >= 2:
                        skew_slope = np.polyfit(lots_s[mask], sk[mask], 1)[0]
                    else:
                        skew_slope = np.nan

                    mc = run_quick_mc(g1_params, g, g1_T, N_sim=g1_nsim,
                                      seed=42, mid_prices=mid_arr_g1)

                    results_g1.append({
                        "γ": g, "mean_spread": mean_spread,
                        "skew_slope": skew_slope, **mc,
                    })
                except Exception as e:
                    st.warning(f"γ={g:.4f} failed: {e}")

            prog.empty()
            st.session_state["g1_results"] = pd.DataFrame(results_g1)

        if "g1_results" in st.session_state:
            df = st.session_state["g1_results"]

            fig = make_subplots(
                rows=2, cols=2, vertical_spacing=0.14,
                subplot_titles=[
                    "Mean Spread vs γ", "Skew Slope vs γ",
                    "Fill Rate (fills/h) vs γ", "Sharpe Ratio vs γ",
                ],
            )
            traces = [
                (1, 1, "mean_spread", PALETTE[0]),
                (1, 2, "skew_slope", PALETTE[1]),
                (2, 1, "fill_rate_per_h", PALETTE[2]),
                (2, 2, "sharpe", "#a29bfe"),
            ]
            for r, c, col, color in traces:
                fig.add_trace(go.Scatter(
                    x=df["γ"], y=df[col],
                    mode="lines+markers",
                    marker=dict(size=8, color=color),
                    line=dict(color=color, width=2),
                    showlegend=False,
                ), row=r, col=c)

            for r in (1, 2):
                for c in (1, 2):
                    fig.update_xaxes(title_text="γ", type="log", row=r, col=c)
            fig.update_layout(**PLOT_KW, height=520)
            show(fig)

            st.dataframe(
                df[["γ", "mean_spread", "skew_slope", "fill_rate_per_h",
                    "mean_pnl", "sharpe", "ce"]].style.format({
                    "γ": "{:.4f}", "mean_spread": "{:.4f}",
                    "skew_slope": "{:.4f}", "fill_rate_per_h": "{:.1f}",
                    "mean_pnl": "{:+,.2f}", "sharpe": "{:.3f}",
                    "ce": "{:+,.2f}",
                }),
                use_container_width=True,
            )

            st.info(
                "**Interpretation:** As γ ↑ the market maker widens its spread "
                "(self-protection), steepens the skew (aggressive inventory rebalancing) "
                "and the fill rate drops. The Sharpe typically rises then plateaus — "
                "the model maximises CARA utility, not raw P&L."
            )
