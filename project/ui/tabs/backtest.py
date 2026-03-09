"""ui/tabs/backtest.py — Tab 5: MC Backtest and strategy comparison."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.simulation.backtest import BacktestConfig, run_backtest

DEFAULT_GAMMA = 0.01


def render_backtest(PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
                    load_mid_prices, run_quick_mc):
    """Render Tab 5 — Backtest: MC backtest with strategy comparison."""
    SYMBOLS = list(PARAMS.keys())

    c_set, c_res = st.columns([1, 3])

    with c_set:
        bt_sym = st.selectbox("Symbol", SYMBOLS, key="bt_sym")
        bt_params = st.session_state.get("calibrated_params", PARAMS[bt_sym]).copy()

        bt_gamma = st.number_input("γ", value=DEFAULT_GAMMA, format="%.4f", key="bt_g")
        bt_T = st.number_input("T (s)", value=3600.0, key="bt_t")
        bt_N = st.slider("Simulations", 100, 5000, 500, step=100)
        bt_fee = st.number_input("Maker fee (bps)", value=1.0, step=0.5) / 10000

        mid_real = load_mid_prices(bt_sym)
        use_real = False
        if mid_real is not None:
            use_real = st.checkbox("Use real prices", value=True)

        bt_strats = st.multiselect("Strategies",
            ["optimal", "naive", "closed_form"], default=["optimal", "naive"])

        run_bt = st.button("Run", type="primary", key="bt_run")

    with c_res:
        if run_bt:
            results = {}
            prog = st.progress(0)

            mid_prices_arr = None
            if use_real and mid_real is not None:
                N_t = int(bt_T)
                if len(mid_real) > N_t + 1:
                    idx0 = np.random.randint(0, len(mid_real) - N_t - 1)
                    mid_prices_arr = mid_real.values[idx0:idx0 + N_t + 1]

            for i, strat in enumerate(bt_strats):
                cfg = BacktestConfig(
                    params=bt_params, gamma=bt_gamma, T=bt_T,
                    N_sim=bt_N, maker_fee=bt_fee, strategy=strat, seed=42,
                    mid_prices=mid_prices_arr,
                )
                results[strat] = run_backtest(cfg)
                prog.progress((i + 1) / len(bt_strats))

            st.session_state["bt_results"] = results
            st.session_state["bt_N"] = bt_N
            prog.empty()

        if "bt_results" in st.session_state:
            results = st.session_state["bt_results"]
            bt_N_val = st.session_state.get("bt_N", 500)

            rdf = pd.DataFrame([r.summary() for r in results.values()]).set_index("strategy")
            st.dataframe(rdf.style.format({
                "mean_pnl": "{:+,.2f}", "std_pnl": "{:,.2f}", "sharpe": "{:.3f}",
                "CE": "{:+,.2f}", "mean_fills": "{:.1f}", "mean_abs_inv_T": "{:.2f}",
                "max_drawdown": "{:,.2f}", "pct_flat": "{:.1%}", "mean_fees": "{:,.4f}",
            }), use_container_width=True)

            fig = go.Figure()
            for i, (nm, r) in enumerate(results.items()):
                c = PALETTE[i % 3]
                fig.add_trace(go.Histogram(x=r.pnl, name=nm, opacity=0.5,
                    nbinsx=80, marker_color=c))
                fig.add_vline(x=r.mean_pnl, line_dash="dash", line_color=c,
                    annotation_text=f"μ={r.mean_pnl:+,.2f}")
            fig.update_layout(**PLOT_KW, height=320, xaxis_title="P&L ($)",
                              barmode="overlay")
            show(fig)

            fig2 = make_subplots(rows=1, cols=2,
                subplot_titles=["E[|inventory|]", "Terminal inventory"])
            for i, (nm, r) in enumerate(results.items()):
                c = PALETTE[i % 3]
                step = max(1, r.inventory.shape[1] // 500)
                ma = np.mean(np.abs(r.inventory[:, ::step].astype(float)), axis=0)
                fig2.add_trace(go.Scatter(x=r.times[::step], y=ma, name=nm,
                    line=dict(color=c)), row=1, col=1)
                fig2.add_trace(go.Histogram(x=r.inventory[:, -1], name=nm,
                    opacity=0.5, showlegend=False, marker_color=c), row=1, col=2)
            fig2.update_layout(**PLOT_KW, height=320)
            show(fig2)

            traj = st.slider("Trajectory #", 0, bt_N_val - 1, bt_N_val // 2)
            fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                subplot_titles=["Price", "Inventory", "MtM"],
                vertical_spacing=0.06)
            for i, (nm, r) in enumerate(results.items()):
                c = PALETTE[i % 3]
                fig3.add_trace(go.Scatter(x=r.times, y=r.price[traj],
                    showlegend=False, line=dict(color="#666", width=0.8)),
                    row=1, col=1)
                fig3.add_trace(go.Scatter(x=r.times, y=r.inventory[traj],
                    name=nm, line=dict(color=c, width=1.5)), row=2, col=1)
                fig3.add_trace(go.Scatter(x=r.times, y=r.mtm[traj],
                    showlegend=False, line=dict(color=c, width=1.5)),
                    row=3, col=1)
            fig3.update_layout(**PLOT_KW, height=550)
            show(fig3)
