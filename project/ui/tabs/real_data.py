"""Page 4 — Real Data: Calibration & Backtest.

Shows calibrated BTC/ETH parameters, data exploration, and
strategy battle on real mid-prices.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.simulation.backtest import compare_strategies
from market_making.params.assets import IG, HY
from ui.styles import PALETTE, PLOT_KW, show, hero_banner, section_header
from ui.loaders import load_calibrated_params, load_mid_prices
from ui.components import metrics_row, insight_box

DEFAULT_GAMMA = 0.01


def render(PARAMS, META, HAS_REAL_DATA):
    hero_banner("📡", "Real Data — Calibration & Backtest",
                "Binance BTC/ETH calibration + strategy replay on real mid-prices")

    if not HAS_REAL_DATA:
        st.warning("No calibrated data found. Using fallback parameters. "
                   "Run `real_data_pipeline.ipynb` to calibrate from Binance aggTrades.")

    tab_exp, tab_bt = st.tabs(["🔍 Data Explorer", "⚔️ Backtest"])

    SYMBOLS = list(PARAMS.keys())

    # ═══════════════════════════════════════════════════════════
    #  TAB A: Data Explorer
    # ═══════════════════════════════════════════════════════════
    with tab_exp:
        sym = st.selectbox("Symbol", SYMBOLS, key="p4_sym")
        p = PARAMS[sym]
        m = META.get(sym, {})

        # ── Parameter comparison table ──
        section_header("PARAMETER COMPARISON")

        table_data = {
            "": ["σ ($/√s)", "A (1/s)", "k (1/$)", "Δ ($)", "ξΔ (at γ=0.01)"],
            "CDX.NA.IG": [f"{IG['sigma']:.2e}", f"{IG['A']:.2e}", f"{IG['k']:.2e}",
                          f"{IG['Delta']:.0e}", f"{0.01 * IG['Delta']:.2e}"],
            "CDX.NA.HY": [f"{HY['sigma']:.2e}", f"{HY['A']:.2e}", f"{HY['k']:.2e}",
                          f"{HY['Delta']:.0e}", f"{0.01 * HY['Delta']:.2e}"],
        }
        for s in SYMBOLS:
            pp = PARAMS[s]
            table_data[s] = [f"{pp['sigma']:.4f}", f"{pp['A']:.4f}", f"{pp['k']:.4f}",
                             f"{pp['Delta']:.2f}", f"{0.01 * pp['Delta']:.4f}"]

        st.dataframe(table_data, hide_index=True, use_container_width=True)

        insight_box("CDX parameters differ from crypto by orders of magnitude "
                    "(σ ~ 1e-6 vs ~5, Δ ~ 50M vs ~100). "
                    "Meaningful comparison requires dimensionless quantities like ξΔ.")

        # ── Mid-price series ──
        mid = load_mid_prices(sym)
        if mid is not None:
            section_header("MID-PRICE TIME SERIES")
            fig_mid = go.Figure()
            fig_mid.add_trace(go.Scatter(
                x=mid.index if hasattr(mid, 'index') else np.arange(len(mid)),
                y=mid.values, mode="lines",
                line=dict(color=PALETTE[0], width=1)))
            fig_mid.update_layout(**PLOT_KW, height=300,
                                  xaxis_title="Time", yaxis_title="Mid price ($)")
            show(fig_mid)

            # ── Returns distribution ──
            section_header("RETURNS DISTRIBUTION")
            returns = np.diff(mid.values)
            returns = returns[np.isfinite(returns)]

            fig_ret = go.Figure()
            fig_ret.add_trace(go.Histogram(
                x=returns, nbinsx=100, marker_color=PALETTE[0], opacity=0.7, name="Returns"))

            # Normal overlay
            from scipy.stats import norm
            x_norm = np.linspace(returns.min(), returns.max(), 200)
            pdf = norm.pdf(x_norm, loc=np.mean(returns), scale=np.std(returns))
            pdf_scaled = pdf * len(returns) * (returns.max() - returns.min()) / 100
            fig_ret.add_trace(go.Scatter(
                x=x_norm, y=pdf_scaled, mode="lines", name="Normal",
                line=dict(color=PALETTE[1], width=2, dash="dash")))

            fig_ret.update_layout(**PLOT_KW, height=300,
                                  xaxis_title="ΔS ($)", yaxis_title="Count")
            show(fig_ret)

            from scipy.stats import skew, kurtosis
            metrics_row({
                "Mean": float(np.mean(returns)),
                "Std": float(np.std(returns)),
                "Skew": float(skew(returns)),
                "Kurtosis": float(kurtosis(returns)),
                "N points": len(mid),
            })

            if float(kurtosis(returns)) > 1:
                insight_box("Returns show excess kurtosis (fat tails) — the GBM assumption "
                            "of the Guéant model under-estimates extreme moves.")
        else:
            st.info("No mid-price data available for this symbol.")

        # ── Calibration metadata ──
        if m:
            section_header("CALIBRATION METADATA")
            st.json(m)

    # ═══════════════════════════════════════════════════════════
    #  TAB B: Backtest
    # ═══════════════════════════════════════════════════════════
    with tab_bt:
        st.caption("Replay the optimal strategy on real BTC/ETH mid-prices.")

        c1, c2 = st.columns([1, 3])
        with c1:
            bt_sym = st.selectbox("Symbol", SYMBOLS, key="p4_bt_sym")
            bt_gamma = st.number_input("γ", value=DEFAULT_GAMMA, format="%.4f", key="p4_bt_g")
            bt_nsim = st.slider("N_sim", 100, 2000, 500, 100, key="p4_bt_nsim")
            bt_fee = st.number_input("Maker fee (bps)", value=1.0, step=0.5, key="p4_bt_fee")
            run_bt = st.button("▶ Run Backtest", type="primary", key="p4_run_bt")

        with c2:
            if run_bt:
                bt_params = PARAMS[bt_sym]
                mid = load_mid_prices(bt_sym)

                mid_arr = None
                if mid is not None:
                    mid_arr = mid.values

                with st.spinner("Running 3-strategy backtest..."):
                    try:
                        results = compare_strategies(
                            bt_params, gamma=bt_gamma, T=3600.0,
                            N_sim=bt_nsim, seed=42, mid_prices=mid_arr,
                            strategies=["optimal", "naive", "closed_form"])
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
                        return

                # ── Cumulative PnL ──
                section_header("CUMULATIVE PnL (mean across simulations)")
                fig_cum = go.Figure()
                for name, color in [("optimal", PALETTE[0]), ("naive", PALETTE[1]),
                                    ("closed_form", PALETTE[3])]:
                    if name in results:
                        r = results[name]
                        mean_mtm = np.mean(r.mtm, axis=0)
                        fig_cum.add_trace(go.Scatter(
                            x=r.times, y=mean_mtm, mode="lines",
                            name=name.replace("_", " ").title(),
                            line=dict(color=color, width=2)))

                fig_cum.update_layout(**PLOT_KW, height=350,
                                      xaxis_title="t (s)", yaxis_title="Mean MtM ($)")
                show(fig_cum)

                # ── PnL distributions ──
                section_header("PnL DISTRIBUTIONS")
                fig_dist = go.Figure()
                for name, color in [("optimal", PALETTE[0]), ("naive", PALETTE[1]),
                                    ("closed_form", PALETTE[3])]:
                    if name in results:
                        fig_dist.add_trace(go.Histogram(
                            x=results[name].pnl, name=name.replace("_", " ").title(),
                            opacity=0.6, marker_color=color, nbinsx=40))

                fig_dist.update_layout(**PLOT_KW, height=300, barmode="overlay",
                                       xaxis_title="Terminal PnL ($)")
                show(fig_dist)

                # ── Summary ──
                section_header("SUMMARY")
                rows = []
                for name, r in results.items():
                    s = r.summary()
                    rows.append({
                        "Strategy": name.replace("_", " ").title(),
                        "E[PnL]": f"{s['mean_pnl']:.2f}",
                        "Std": f"{s['std_pnl']:.2f}",
                        "Sharpe": f"{s['sharpe']:.3f}",
                        "CE": f"{s['CE']:.2f}",
                        "Fills": f"{s['mean_fills']:.0f}",
                        "MaxDD": f"{s['max_drawdown']:.2f}",
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

                if mid_arr is not None:
                    insight_box("Backtest uses real mid-prices (Binance). "
                                "Fills are still simulated via Poisson — a simplification. "
                                "True LOB replay would require full order book data.")
            else:
                st.info("Configure parameters and press **▶ Run Backtest**.")
