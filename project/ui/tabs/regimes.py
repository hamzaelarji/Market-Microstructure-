"""ui/tabs/regimes.py — Tab 6: Intraday Regimes (static vs regime-aware policy)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general

DEFAULT_GAMMA = 0.01


def render_regimes(PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
                   load_mid_prices, run_quick_mc):
    """Render Tab 6 — Intraday Regimes: static vs regime-aware policy comparison."""
    SYMBOLS = list(PARAMS.keys())

    st.caption(
        "Market conditions change intraday. Compare a **static** policy "
        "(single ODE solve) vs a **regime-aware** one that re-solves at each boundary."
    )

    c_r1, c_r2 = st.columns([1, 3])

    with c_r1:
        rg_sym = st.selectbox("Symbol", SYMBOLS, key="rg_sym")
        rg_params = PARAMS[rg_sym]
        rg_gamma = st.number_input("γ", value=DEFAULT_GAMMA, format="%.4f", key="rg_g")
        rg_T = st.number_input("T (s)", value=3600.0, step=600.0, key="rg_T")
        rg_nsim = st.slider("MC paths", 200, 3000, 600, step=100, key="rg_nsim")

        st.markdown("**Regime multipliers**")
        open_A = st.slider("Open — A ×", 0.5, 2.5, 1.45, 0.05, key="rg_oA")
        open_s = st.slider("Open — σ ×", 0.5, 2.5, 1.50, 0.05, key="rg_os")
        noon_A = st.slider("Noon — A ×", 0.2, 1.5, 0.60, 0.05, key="rg_nA")
        noon_s = st.slider("Noon — σ ×", 0.2, 1.5, 0.55, 0.05, key="rg_ns")
        close_A = st.slider("Close — A ×", 0.5, 2.5, 1.35, 0.05, key="rg_cA")
        close_s = st.slider("Close — σ ×", 0.5, 2.5, 1.40, 0.05, key="rg_cs")

        run_rg = st.button("Run", type="primary", key="rg_run")

    with c_r2:
        if run_rg:
            N_t_rg = max(300, int(rg_T))
            Q_rg = int(rg_params["Q"])
            n_states_rg = 2 * Q_rg + 1
            k_rg = rg_params["k"]
            Delta_rg = rg_params["Delta"]
            dt_rg = rg_T / N_t_rg

            regimes = [
                dict(A_mult=open_A, sigma_mult=open_s, name="Open"),
                dict(A_mult=noon_A, sigma_mult=noon_s, name="Noon"),
                dict(A_mult=close_A, sigma_mult=close_s, name="Close"),
            ]
            breaks = [0, int(0.3 * N_t_rg), int(0.7 * N_t_rg), N_t_rg]

            A_path = np.zeros(N_t_rg)
            sigma_path = np.zeros(N_t_rg)
            for ri, (s, e) in enumerate(zip(breaks[:-1], breaks[1:])):
                A_path[s:e] = rg_params["A"] * regimes[ri]["A_mult"]
                sigma_path[s:e] = rg_params["sigma"] * regimes[ri]["sigma_mult"]

            with st.spinner("Static policy …"):
                sol_static = solve_general(rg_params, rg_gamma, rg_T,
                                           xi=rg_gamma, N_t=N_t_rg)
                db_static = sol_static["delta_bid"]
                da_static = sol_static["delta_ask"]

            with st.spinner("Regime-aware policy …"):
                db_regime = np.full((N_t_rg + 1, n_states_rg), np.nan)
                da_regime = np.full((N_t_rg + 1, n_states_rg), np.nan)
                for ri, (s, e) in enumerate(zip(breaks[:-1], breaks[1:])):
                    rem = N_t_rg - s
                    rem_T = rg_T * rem / N_t_rg
                    p_reg = {
                        **rg_params,
                        "A": rg_params["A"] * regimes[ri]["A_mult"],
                        "sigma": rg_params["sigma"] * regimes[ri]["sigma_mult"],
                    }
                    sol_reg = solve_general(p_reg, rg_gamma, rem_T,
                                            xi=rg_gamma, N_t=rem)
                    seg = e - s
                    db_regime[s:e, :] = sol_reg["delta_bid"][:seg, :]
                    da_regime[s:e, :] = sol_reg["delta_ask"][:seg, :]
                db_regime[-1, :] = db_regime[-2, :]
                da_regime[-1, :] = da_regime[-2, :]

            def _sim_regime(db_tbl, da_tbl):
                rng = np.random.default_rng(42)
                z = rng.standard_normal((rg_nsim, N_t_rg))
                u_b = rng.uniform(size=(rg_nsim, N_t_rg))
                u_a = rng.uniform(size=(rg_nsim, N_t_rg))
                pnl = np.zeros(rg_nsim)
                inv_all = np.zeros((rg_nsim, N_t_rg + 1), dtype=int)
                mtm_all = np.zeros((rg_nsim, N_t_rg + 1))
                fills = np.zeros(rg_nsim, dtype=int)

                for m in range(rg_nsim):
                    S, X, n = 0.0, 0.0, 0
                    for t in range(N_t_rg):
                        il = n + Q_rg
                        if not (0 <= il < n_states_rg):
                            inv_all[m, t + 1] = n
                            mtm_all[m, t + 1] = X + n * Delta_rg * S
                            continue
                        db_v = db_tbl[t, il]
                        da_v = da_tbl[t, il]
                        S += sigma_path[t] * np.sqrt(dt_rg) * z[m, t]
                        if n < Q_rg and np.isfinite(db_v):
                            if u_b[m, t] < A_path[t] * np.exp(-k_rg * db_v) * dt_rg:
                                X -= (S - db_v) * Delta_rg; n += 1; fills[m] += 1
                        if n > -Q_rg and np.isfinite(da_v):
                            if u_a[m, t] < A_path[t] * np.exp(-k_rg * da_v) * dt_rg:
                                X += (S + da_v) * Delta_rg; n -= 1; fills[m] += 1
                        inv_all[m, t + 1] = n
                        mtm_all[m, t + 1] = X + n * Delta_rg * S
                    pnl[m] = X + n * Delta_rg * S
                return pnl, inv_all, mtm_all, fills

            with st.spinner("Simulating static …"):
                pnl_s, inv_s, mtm_s, fills_s = _sim_regime(db_static, da_static)
            with st.spinner("Simulating regime-aware …"):
                pnl_r, inv_r, mtm_r, fills_r = _sim_regime(db_regime, da_regime)

            st.session_state["rg_data"] = dict(
                pnl_s=pnl_s, pnl_r=pnl_r, inv_s=inv_s, inv_r=inv_r,
                mtm_s=mtm_s, mtm_r=mtm_r, fills_s=fills_s, fills_r=fills_r,
                A_path=A_path, sigma_path=sigma_path,
                T=rg_T, N_t=N_t_rg,
            )

        if "rg_data" in st.session_state:
            rd = st.session_state["rg_data"]
            t_ax = np.linspace(0, rd["T"], rd["N_t"] + 1)
            t_ax_inner = np.linspace(0, rd["T"], rd["N_t"])
            C_S, C_R = "#a29bfe", PALETTE[0]

            fig0 = make_subplots(rows=1, cols=2,
                subplot_titles=["A(t)", "σ(t)"])
            fig0.add_trace(go.Scatter(x=t_ax_inner, y=rd["A_path"],
                line=dict(color=C_R, width=2), showlegend=False), row=1, col=1)
            fig0.add_trace(go.Scatter(x=t_ax_inner, y=rd["sigma_path"],
                line=dict(color=C_R, width=2), showlegend=False), row=1, col=2)
            fig0.update_layout(**PLOT_KW, height=220)
            show(fig0)

            def _rg_sum(pnl, fills, label):
                return {
                    "Policy": label,
                    "E[P&L]": np.mean(pnl), "Std[P&L]": np.std(pnl),
                    "Sharpe": np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else 0,
                    "VaR 5%": np.percentile(pnl, 5),
                    "Mean fills": np.mean(fills),
                }
            st.dataframe(
                pd.DataFrame([
                    _rg_sum(rd["pnl_s"], rd["fills_s"], "Static"),
                    _rg_sum(rd["pnl_r"], rd["fills_r"], "Regime-aware"),
                ]).set_index("Policy").style.format({
                    "E[P&L]": "{:+,.2f}", "Std[P&L]": "{:,.2f}",
                    "Sharpe": "{:.3f}", "VaR 5%": "{:+,.2f}",
                    "Mean fills": "{:.1f}",
                }),
                use_container_width=True,
            )

            fig1 = make_subplots(rows=1, cols=2,
                subplot_titles=["P&L distribution", "Average MtM"])
            fig1.add_trace(go.Histogram(x=rd["pnl_s"], name="Static",
                opacity=0.5, marker_color=C_S, nbinsx=60), row=1, col=1)
            fig1.add_trace(go.Histogram(x=rd["pnl_r"], name="Regime",
                opacity=0.5, marker_color=C_R, nbinsx=60), row=1, col=1)
            fig1.add_trace(go.Scatter(x=t_ax, y=np.mean(rd["mtm_s"], axis=0),
                name="Static", line=dict(color=C_S, width=2)), row=1, col=2)
            fig1.add_trace(go.Scatter(x=t_ax, y=np.mean(rd["mtm_r"], axis=0),
                name="Regime", line=dict(color=C_R, width=2)), row=1, col=2)
            fig1.update_layout(**PLOT_KW, height=360, barmode="overlay")
            show(fig1)

            step_r = max(1, rd["inv_s"].shape[1] // 500)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=t_ax[::step_r],
                y=np.mean(np.abs(rd["inv_s"][:, ::step_r].astype(float)), axis=0),
                name="Static", line=dict(color=C_S, width=2)))
            fig2.add_trace(go.Scatter(
                x=t_ax[::step_r],
                y=np.mean(np.abs(rd["inv_r"][:, ::step_r].astype(float)), axis=0),
                name="Regime", line=dict(color=C_R, width=2)))
            fig2.update_layout(**PLOT_KW, height=300,
                xaxis_title="t (s)", yaxis_title="E[|inventory|]")
            show(fig2)

            st.info(
                "**Interpretation:** The regime-aware policy adapts quotes to changing "
                "conditions — wider spreads when volatile (open/close), tighter at noon. "
                "This typically yields better Sharpe than the static policy."
            )
