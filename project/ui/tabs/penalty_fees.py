"""Page 6 — Terminal Penalty ℓ(|q|) & Transaction Cost Analysis.

Part A: explores ell_func in solve_general (never used in the paper's numerics).
Part B: maker fee impact — break-even analysis.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general
from extensions.penalty import solve_with_penalty, penalty_convergence_sweep
from extensions.fee_analysis import fee_sweep
from ui.styles import PALETTE, PLOT_KW, show, hero_banner, section_header
from ui.components import param_sidebar, metrics_row, insight_box

DEFAULT_GAMMA = 0.01


def render(PARAMS, META, HAS_REAL_DATA):
    hero_banner("⚖️", "Penalty & Fees",
                "Terminal inventory penalty ℓ(|q|) + maker fee impact analysis")

    with st.sidebar:
        section_header("PARAMETERS")
        cfg = param_sidebar(PARAMS, key_prefix="p6")

    params = cfg["params"]
    gamma, T_val = cfg["gamma"], cfg["T"]
    Q = int(params["Q"])

    tab_pen, tab_fee = st.tabs(["📋 Terminal Penalty", "💰 Transaction Costs"])

    # ═══════════════════════════════════════════════════════════
    #  TAB A: Terminal Penalty
    # ═══════════════════════════════════════════════════════════
    with tab_pen:
        st.caption("The paper discusses ℓ(|q|) but sets it to 0 in all numerical experiments. "
                   "What happens when we turn it on?")

        c1, c2 = st.columns([1, 3])
        with c1:
            pen_type = st.radio("Penalty type", ["None (ℓ ≡ 0)", "Linear ℓ = c|q|", "Quadratic ℓ = cq²"],
                                key="p6_ptype")
            pen_c = 0.0
            if "Linear" in pen_type:
                pen_c = st.number_input("c (linear)", value=1e-4, format="%.6f", step=1e-5, key="p6_cl")
                ptype = "linear"
            elif "Quadratic" in pen_type:
                pen_c = st.number_input("c (quadratic)", value=1e-6, format="%.8f", step=1e-7, key="p6_cq")
                ptype = "quadratic"
            else:
                ptype = "none"

            show_conv = st.checkbox("Show penalty vs T convergence", value=False, key="p6_conv")

        with c2:
            try:
                N_t = max(300, int(T_val))
                sol_no = solve_general(params, gamma, T_val, xi=gamma, N_t=N_t)
                if ptype != "none":
                    sol_pen = solve_with_penalty(params, gamma, T_val,
                                                 penalty_type=ptype, penalty_c=pen_c, N_t=N_t)
                else:
                    sol_pen = sol_no
            except Exception as e:
                st.error(f"Solve failed: {e}")
                return

            lots = sol_no["lots"]

            # ── δ^bid(t, n) showing terminal effect ──
            section_header("CONVERGENCE: δ^bid(t, n) WITH PENALTY")

            fig_conv = go.Figure()
            times = sol_pen["times"]
            n_show = 0
            i_lot = n_show + Q

            fig_conv.add_trace(go.Scatter(
                x=times, y=sol_no["delta_bid"][:, i_lot],
                mode="lines", name="ℓ ≡ 0",
                line=dict(color=PALETTE[6], width=2, dash="dash")))

            if ptype != "none":
                fig_conv.add_trace(go.Scatter(
                    x=times, y=sol_pen["delta_bid"][:, i_lot],
                    mode="lines", name=f"ℓ = {ptype} (c={pen_c:.2e})",
                    line=dict(color=PALETTE[0], width=2)))

            fig_conv.update_layout(**PLOT_KW, height=300,
                                   xaxis_title="t (s)", yaxis_title=f"δ^bid(t, n={n_show})")
            show(fig_conv)

            if ptype != "none":
                st.caption("Near T (right side), the penalty forces the MM to "
                           "aggressively unwind inventory — quotes deviate from steady state. "
                           "At t ≈ 0 (left), the effect vanishes.")

            # ── Comparison at t=0 ──
            section_header("QUOTES AT t = 0: WITH vs WITHOUT PENALTY")

            fig_cmp = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                                    subplot_titles=["Spread(0, n)", "Skew(0, n)"])

            spread_no = sol_no["delta_bid"][0, :] + sol_no["delta_ask"][0, :]
            spread_pen = sol_pen["delta_bid"][0, :] + sol_pen["delta_ask"][0, :]
            skew_no = sol_no["delta_bid"][0, :] - sol_no["delta_ask"][0, :]
            skew_pen = sol_pen["delta_bid"][0, :] - sol_pen["delta_ask"][0, :]

            for y, name, color, dash in [
                (spread_no, "ℓ ≡ 0", PALETTE[6], "dash"),
                (spread_pen, "With penalty", PALETTE[0], "solid"),
            ]:
                m = np.isfinite(y)
                fig_cmp.add_trace(go.Scatter(
                    x=lots[m], y=y[m], mode="lines+markers", name=name,
                    line=dict(color=color, width=2, dash=dash),
                    marker=dict(size=5)), row=1, col=1)

            for y, name, color, dash in [
                (skew_no, "ℓ ≡ 0", PALETTE[6], "dash"),
                (skew_pen, "With penalty", PALETTE[0], "solid"),
            ]:
                m = np.isfinite(y)
                fig_cmp.add_trace(go.Scatter(
                    x=lots[m], y=y[m], mode="lines+markers", name=name,
                    line=dict(color=color, width=2, dash=dash),
                    marker=dict(size=5), showlegend=False), row=1, col=2)

            fig_cmp.update_layout(**PLOT_KW, height=320,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.05))
            show(fig_cmp)

            insight_box("At t = 0 (far from T), the penalty barely changes quotes — "
                        "the asymptotic regime dominates. The penalty only matters near T, "
                        "justifying the paper's choice of ℓ ≡ 0.")

            # ── Penalty convergence sweep ──
            if show_conv and ptype != "none":
                section_header("PENALTY EFFECT vs TIME HORIZON T")
                with st.spinner("Sweeping T values..."):
                    conv = penalty_convergence_sweep(params, gamma, penalty_c=pen_c)

                fig_tc = go.Figure()
                fig_tc.add_trace(go.Scatter(
                    x=conv["T_values"], y=conv["max_error"],
                    mode="lines+markers", marker=dict(size=8, color=PALETTE[0]),
                    line=dict(color=PALETTE[0], width=2)))
                fig_tc.update_layout(**PLOT_KW, height=300,
                                     xaxis_title="T (s)",
                                     yaxis_title="max |δ(ℓ≠0) − δ(ℓ=0)| at t=0")
                show(fig_tc)

    # ═══════════════════════════════════════════════════════════
    #  TAB B: Transaction Costs
    # ═══════════════════════════════════════════════════════════
    with tab_fee:
        st.caption("Impact of maker fees on strategy profitability. "
                   "Binance default: 1 bps maker fee.")

        c1, c2 = st.columns([1, 3])
        with c1:
            max_fee = st.slider("Max fee (bps)", 1, 10, 5, key="p6_maxfee")
            n_sim_fee = st.slider("N_sim", 100, 1000, 300, 50, key="p6_nsim_fee")
            run_fee = st.button("▶ Run fee sweep", type="primary", key="p6_run_fee")

        with c2:
            if run_fee:
                fee_range = np.linspace(0, max_fee, min(max_fee * 2 + 1, 11))

                with st.spinner("Sweeping fee levels..."):
                    res = fee_sweep(params, gamma, fee_bps_range=fee_range,
                                    T=T_val, N_sim=n_sim_fee, seed=42)

                # ── Sharpe vs fee ──
                section_header("SHARPE RATIO vs MAKER FEE")
                fig_sh = go.Figure()
                fig_sh.add_trace(go.Scatter(
                    x=res["fee_bps"], y=res["sharpe"],
                    mode="lines+markers", marker=dict(size=8, color=PALETTE[0]),
                    line=dict(color=PALETTE[0], width=2)))
                fig_sh.add_hline(y=0, line_dash="dot", line_color="gray",
                                 annotation_text="Break-even")
                fig_sh.update_layout(**PLOT_KW, height=350,
                                     xaxis_title="Maker fee (bps)",
                                     yaxis_title="Sharpe ratio")
                show(fig_sh)

                # Find break-even
                sharpes = res["sharpe"]
                be_fee = None
                for i in range(len(sharpes) - 1):
                    if sharpes[i] >= 0 and sharpes[i + 1] < 0:
                        f = sharpes[i] / max(sharpes[i] - sharpes[i + 1], 1e-12)
                        be_fee = res["fee_bps"][i] + f * (res["fee_bps"][i + 1] - res["fee_bps"][i])
                        break

                # ── PnL decomposition ──
                section_header("PnL DECOMPOSITION")
                fig_decomp = go.Figure()
                fig_decomp.add_trace(go.Bar(
                    x=[f"{f:.1f} bps" for f in res["fee_bps"]],
                    y=res["gross_pnl"], name="Gross PnL",
                    marker_color=PALETTE[0]))
                fig_decomp.add_trace(go.Bar(
                    x=[f"{f:.1f} bps" for f in res["fee_bps"]],
                    y=-res["total_fees"], name="Fees (negative)",
                    marker_color=PALETTE[1]))
                fig_decomp.add_trace(go.Scatter(
                    x=[f"{f:.1f} bps" for f in res["fee_bps"]],
                    y=res["net_pnl"], mode="lines+markers", name="Net PnL",
                    line=dict(color=PALETTE[4], width=2),
                    marker=dict(size=8, color=PALETTE[4])))

                fig_decomp.update_layout(**PLOT_KW, height=350, barmode="relative",
                                         xaxis_title="Maker fee",
                                         yaxis_title="Mean PnL ($)")
                show(fig_decomp)

                be_text = f"Break-even fee: ~{be_fee:.1f} bps" if be_fee else "Strategy profitable across all fee levels tested"
                insight_box(f"{be_text}. "
                            "Higher fees eat into spread profit. "
                            "Crypto exchanges with negative maker fees (rebates) "
                            "would significantly boost performance.")
            else:
                st.info("Press **▶ Run fee sweep** to analyse fee impact.")
