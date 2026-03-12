"""Page 3 — Multi-Asset Market Making & Monte Carlo Validation.

Multi-asset: Paper Figures 17–19 (2D IG+HY surfaces, ρ effect).
Monte Carlo: Extension — PnL histogram, trajectories, inventory control.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general
from market_making.core.solver_2d import solve_2d
from market_making.simulation.backtest import BacktestConfig, run_backtest, compare_strategies
from market_making.params.assets import IG, HY, GAMMA as PAPER_GAMMA, RHO
from ui.styles import PALETTE, PLOT_KW, show, hero_banner, section_header
from ui.components import metrics_row, insight_box

DEFAULT_GAMMA = 0.01


def render(PARAMS, META, HAS_REAL_DATA):
    hero_banner("🌐", "Multi-Asset & Monte Carlo",
                "Paper §5 (correlated assets) + simulation validation")

    tab_ma, tab_mc = st.tabs(["🗺️ Multi-Asset (2D)", "🎲 Monte Carlo"])

    # ═══════════════════════════════════════════════════════════
    #  TAB A: Multi-Asset
    # ═══════════════════════════════════════════════════════════
    with tab_ma:
        st.caption("Two correlated assets (IG + HY). The optimal bid quote for each asset "
                   "depends on inventory in **both** assets.")

        c1, c2 = st.columns([1, 3])
        with c1:
            rho = st.slider("ρ (correlation)", 0.0, 1.0, 0.9, 0.05, key="p3_rho")
            gamma_2d = st.number_input("γ", value=PAPER_GAMMA, format="%.6f", key="p3_g2d")
            T_2d = st.number_input("T (s)", value=3600.0, step=600.0, key="p3_T2d")
            N_t_2d = st.slider("N_t (time steps)", 100, 1000, 300, 50, key="p3_Nt2d")
            use_paper = st.checkbox("Use paper params (CDX)", value=True, key="p3_paper")

        with c2:
            if use_paper:
                p1, p2 = IG, HY
                xi_2d = gamma_2d
            else:
                symbols = list(PARAMS.keys())
                if len(symbols) >= 2:
                    p1, p2 = PARAMS[symbols[0]], PARAMS[symbols[1]]
                    xi_2d = gamma_2d
                else:
                    st.warning("Need ≥ 2 calibrated assets for multi-asset mode.")
                    return

            try:
                with st.spinner("Solving 2D ODE (this may take ~30s)..."):
                    sol2d = solve_2d(p1, p2, gamma_2d, rho, T_2d,
                                    xi=xi_2d, N_t=N_t_2d)
            except Exception as e:
                st.error(f"2D solver failed: {e}")
                return

            grid = sol2d["grid"]
            idx = sol2d["idx"]
            Q1, Q2 = int(p1["Q"]), int(p2["Q"])
            lots1 = np.arange(-Q1, Q1 + 1)
            lots2 = np.arange(-Q2, Q2 + 1)

            # Build 2D matrix for δ^bid_1(n1, n2) at t=0
            db1_mat = np.full((len(lots1), len(lots2)), np.nan)
            for i, n1 in enumerate(lots1):
                for j, n2 in enumerate(lots2):
                    if (n1, n2) in idx:
                        db1_mat[i, j] = sol2d["delta_bid_1"][0, idx[(n1, n2)]]

            # Fig 17: 3D surface δ^bid_1
            fig17 = go.Figure(data=[go.Surface(
                z=db1_mat, x=lots2, y=lots1, colorscale="Viridis",
                colorbar=dict(title="δ^bid₁"))])
            fig17.update_layout(**PLOT_KW, height=450,
                                title="δ^bid Asset 1 (IG) — Fig 17",
                                scene=dict(xaxis_title="n₂ (HY lots)",
                                           yaxis_title="n₁ (IG lots)",
                                           zaxis_title="δ^bid₁"))
            show(fig17)

            # Build db2 matrix
            db2_mat = np.full((len(lots1), len(lots2)), np.nan)
            for i, n1 in enumerate(lots1):
                for j, n2 in enumerate(lots2):
                    if (n1, n2) in idx:
                        db2_mat[i, j] = sol2d["delta_bid_2"][0, idx[(n1, n2)]]

            fig18 = go.Figure(data=[go.Surface(
                z=db2_mat, x=lots2, y=lots1, colorscale="Plasma",
                colorbar=dict(title="δ^bid₂"))])
            fig18.update_layout(**PLOT_KW, height=450,
                                title="δ^bid Asset 2 (HY) — Fig 18",
                                scene=dict(xaxis_title="n₂ (HY lots)",
                                           yaxis_title="n₁ (IG lots)",
                                           zaxis_title="δ^bid₂"))
            show(fig18)

            # Fig 19: Cross-section — fix n2, sweep n1, vary ρ
            section_header("CROSS-ASSET EFFECT (Fig 19)")
            st.caption("Fix n₂, show how δ^bid₁ depends on n₁ for different ρ values.")

            rho_vals = [0.0, 0.3, 0.6, 0.9]
            fig19 = go.Figure()
            colors_rho = [PALETTE[0], PALETTE[4], PALETTE[5], PALETTE[1]]

            for rv, clr in zip(rho_vals, colors_rho):
                try:
                    sol_rho = solve_2d(p1, p2, gamma_2d, rv, T_2d,
                                       xi=xi_2d, N_t=min(N_t_2d, 200))
                    db1_n2_0 = []
                    for n1 in lots1:
                        if (n1, 0) in sol_rho["idx"]:
                            db1_n2_0.append(sol_rho["delta_bid_1"][0, sol_rho["idx"][(n1, 0)]])
                        else:
                            db1_n2_0.append(np.nan)
                    fig19.add_trace(go.Scatter(
                        x=lots1, y=db1_n2_0, mode="lines+markers",
                        name=f"ρ = {rv}", line=dict(color=clr, width=2),
                        marker=dict(size=6, color=clr)))
                except Exception:
                    pass

            fig19.update_layout(**PLOT_KW, height=350,
                                xaxis_title="n₁ (IG lots)", yaxis_title="δ^bid₁ at n₂=0")
            show(fig19)

            insight_box("Higher correlation → stronger cross-asset effect: "
                        "holding inventory in asset 2 shifts optimal quotes in asset 1. "
                        "At ρ=0, the assets decouple and 2D reduces to two independent 1D problems.")

    # ═══════════════════════════════════════════════════════════
    #  TAB B: Monte Carlo
    # ═══════════════════════════════════════════════════════════
    with tab_mc:
        st.caption("Validate the optimal strategy by simulation: does it actually work?")

        c_mc1, c_mc2 = st.columns([1, 3])
        with c_mc1:
            symbols = list(PARAMS.keys())
            mc_sym = st.selectbox("Asset", symbols, key="p3_mc_sym")
            mc_params = PARAMS[mc_sym]
            mc_gamma = st.number_input("γ", value=DEFAULT_GAMMA, format="%.4f", key="p3_mc_g")
            mc_T = st.number_input("T (s)", value=3600.0, step=600.0, key="p3_mc_T")
            mc_nsim = st.slider("N_sim", 100, 3000, 500, 100, key="p3_mc_nsim")
            run_mc = st.button("▶ Run Monte Carlo", type="primary", key="p3_run_mc")

        with c_mc2:
            if run_mc:
                with st.spinner("Running optimal + naive backtests..."):
                    try:
                        results = compare_strategies(mc_params, gamma=mc_gamma,
                                                     T=mc_T, N_sim=mc_nsim, seed=42,
                                                     strategies=["optimal", "naive"])
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
                        return

                res_opt = results["optimal"]
                res_naive = results["naive"]

                # ── PnL histogram ──
                section_header("PnL DISTRIBUTION")
                fig_pnl = go.Figure()
                for res, name, color in [
                    (res_opt, "Optimal", PALETTE[0]),
                    (res_naive, "Naive", PALETTE[1]),
                ]:
                    fig_pnl.add_trace(go.Histogram(
                        x=res.pnl, name=name, opacity=0.6,
                        marker_color=color, nbinsx=50))
                    fig_pnl.add_vline(x=res.mean_pnl, line_dash="dash",
                                      line_color=color, annotation_text=f"{name} mean")

                fig_pnl.update_layout(**PLOT_KW, height=380, barmode="overlay",
                                      xaxis_title="Terminal PnL ($)",
                                      yaxis_title="Count")
                show(fig_pnl)

                # ── Sample trajectory ──
                section_header("SAMPLE TRAJECTORY")
                fig_traj = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                         vertical_spacing=0.06,
                                         subplot_titles=["Mid Price", "Inventory (lots)", "MtM ($)"])

                times = res_opt.times
                fig_traj.add_trace(go.Scatter(x=times, y=res_opt.price[0],
                    mode="lines", line=dict(color=PALETTE[6], width=1), showlegend=False), row=1, col=1)
                fig_traj.add_trace(go.Scatter(x=times, y=res_opt.inventory[0],
                    mode="lines", line=dict(color=PALETTE[0], width=1.5), name="Optimal"), row=2, col=1)
                fig_traj.add_trace(go.Scatter(x=times, y=res_naive.inventory[0],
                    mode="lines", line=dict(color=PALETTE[1], width=1.5), name="Naive"), row=2, col=1)
                fig_traj.add_trace(go.Scatter(x=times, y=res_opt.mtm[0],
                    mode="lines", line=dict(color=PALETTE[0], width=1.5), showlegend=False), row=3, col=1)
                fig_traj.add_trace(go.Scatter(x=times, y=res_naive.mtm[0],
                    mode="lines", line=dict(color=PALETTE[1], width=1.5), showlegend=False), row=3, col=1)

                fig_traj.update_layout(**PLOT_KW, height=500)
                fig_traj.update_xaxes(title_text="t (s)", row=3, col=1)
                show(fig_traj)

                # ── Inventory control ──
                section_header("INVENTORY CONTROL")
                mean_abs_opt = np.mean(np.abs(res_opt.inventory), axis=0)
                mean_abs_naive = np.mean(np.abs(res_naive.inventory), axis=0)

                fig_inv = go.Figure()
                fig_inv.add_trace(go.Scatter(x=times, y=mean_abs_opt,
                    mode="lines", name="Optimal E[|n|]", line=dict(color=PALETTE[0], width=2)))
                fig_inv.add_trace(go.Scatter(x=times, y=mean_abs_naive,
                    mode="lines", name="Naive E[|n|]", line=dict(color=PALETTE[1], width=2)))
                fig_inv.update_layout(**PLOT_KW, height=300,
                                      xaxis_title="t (s)", yaxis_title="E[|inventory|]")
                show(fig_inv)

                # ── Summary table ──
                section_header("METRICS")
                metrics = {s: r.summary() for s, r in results.items()}
                st.dataframe(metrics, use_container_width=True)

                insight_box("The optimal strategy has LOWER mean PnL but HIGHER Sharpe and "
                            "Certainty Equivalent. It maximises CARA utility, not expected wealth. "
                            "The correct performance metric is CE, not mean PnL.")
            else:
                st.info("Press **▶ Run Monte Carlo** to simulate. This compares the ODE-optimal "
                        "strategy against a naive (symmetric, no skewing) baseline.")
