"""ui/tabs/param_lab.py — Tab 1: Parameter Lab."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general
from market_making.core.closed_form import approx_quotes
from market_making.core.intensity import C_coeff
from ui.styles import hero_banner, section_header

DEFAULT_GAMMA = 0.01


def render_param_lab(PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
                     load_mid_prices, run_quick_mc):
    """Render Tab 1 — Quote Lab: ODE quotes, 3D surface, Model A vs B."""
    hero_banner(
        "🔬",
        "Quote Lab",
        "Explore ODE-optimal quotes vs closed-form approximation. Compare Model A (ξ=γ) and Model B (ξ=0).",
    )
    SYMBOLS = list(PARAMS.keys())

    c_ctrl, c_viz = st.columns([1, 3])

    with c_ctrl:
        preset = st.selectbox("Preset", SYMBOLS + ["Custom"])
        if preset in PARAMS:
            d = {**PARAMS[preset], "gamma": DEFAULT_GAMMA}
        else:
            d = {"sigma": 5.0, "A": 5.0, "k": 3.0,
                 "Delta": 100.0, "Q": 4, "gamma": 0.01}

        sigma = st.number_input("σ ($/√s)", value=d["sigma"], format="%.4f",
                                step=d["sigma"] * 0.1)
        A = st.number_input("A (1/s)", value=d["A"], format="%.4f",
                            step=d["A"] * 0.1)
        k = st.number_input("k (1/$)", value=d["k"], format="%.4f",
                            step=d["k"] * 0.1)
        Delta = st.number_input("Δ ($)", value=d["Delta"], format="%.2f",
                                step=d["Delta"] * 0.1)
        Q = st.slider("Q", 1, 10, d["Q"])
        gamma = st.number_input("γ", value=d["gamma"], format="%.4f", step=0.001)
        T_lab = st.number_input("T (s)", value=3600.0, step=600.0)
        model = st.radio("Model", ["A (ξ=γ)", "B (ξ=0)"], horizontal=True)
        xi = gamma if "A" in model else 0.0
        params = {"sigma": sigma, "A": A, "k": k, "Delta": Delta, "Q": Q}

        show_both = st.checkbox("Compare A vs B side-by-side", value=False)
        show_3d = st.checkbox("Show 3D δ(t,n) surface", value=False)

    with c_viz:
        try:
            N_t_lab = max(300, int(T_lab))
            sol = solve_general(params, gamma, T_lab, xi=xi, N_t=N_t_lab)
            lots = sol["lots"]
            n_arr = np.arange(-Q + 1, Q)
            db_cf, da_cf = approx_quotes(n_arr, params, gamma, xi=xi)

            db = sol["delta_bid"][0, :]
            da = sol["delta_ask"][0, :]

            if show_both:
                xi_b = 0.0 if "A" in model else gamma
                sol_b = solve_general(params, gamma, T_lab, xi=xi_b, N_t=N_t_lab)
                db_b = sol_b["delta_bid"][0, :]
                da_b = sol_b["delta_ask"][0, :]

            fig = make_subplots(rows=2, cols=2, vertical_spacing=0.12,
                horizontal_spacing=0.08,
                subplot_titles=["δ bid(0,n)", "δ ask(0,n)",
                                "Spread(0,n)", "Skew(0,n)"])

            def _add(row, col, x_ode, y_ode, y_cf, legend=False):
                m = np.isfinite(y_ode)
                fig.add_trace(go.Scatter(
                    x=x_ode[m], y=y_ode[m], mode="markers",
                    name="ODE", marker=dict(size=8, color=PALETTE[0]),
                    showlegend=legend), row=row, col=col)
                fig.add_trace(go.Scatter(
                    x=n_arr, y=y_cf, mode="lines",
                    name="Closed-form", line=dict(dash="dash", color=PALETTE[1]),
                    showlegend=legend), row=row, col=col)

            def _add_b(row, col, x_ode, y_ode):
                m = np.isfinite(y_ode)
                fig.add_trace(go.Scatter(
                    x=x_ode[m], y=y_ode[m], mode="markers",
                    name="Other model", marker=dict(size=6, color=PALETTE[2],
                    symbol="circle-open"),
                    showlegend=(row == 1 and col == 1)), row=row, col=col)

            _add(1, 1, lots, db, db_cf, legend=True)
            _add(1, 2, lots, da, da_cf)
            _add(2, 1, lots, db + da, db_cf + da_cf)
            _add(2, 2, lots, db - da, db_cf - da_cf)

            if show_both:
                _add_b(1, 1, lots, db_b)
                _add_b(1, 2, lots, da_b)
                _add_b(2, 1, lots, db_b + da_b)
                _add_b(2, 2, lots, db_b - da_b)

            fig.update_layout(**PLOT_KW, height=520,
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            for r in (1, 2):
                for c in (1, 2):
                    fig.update_xaxes(title_text="n", row=r, col=c)
            show(fig)

            xi_D = xi * Delta
            d_s = (1.0 / xi_D) * np.log(1 + xi_D / k) if abs(xi_D) > 1e-12 else 1.0 / k
            C = C_coeff(xi_D, k)
            omega = np.sqrt(gamma * sigma**2 / (2.0 * A * Delta * k * C))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("δ_static", f"{d_s:.4f}")
            m2.metric("ω", f"{omega:.4e}")
            m3.metric("Spread(0)", f"{2 * d_s + omega * Delta:.4f}")
            m4.metric("ξΔ", f"{xi_D:.6f}")

            if show_3d:
                times_3d = sol["times"]
                db_3d = sol["delta_bid"]
                step_t = max(1, len(times_3d) // 80)
                t_sub = times_3d[::step_t]
                db_sub = db_3d[::step_t, :]
                db_plot = np.where(np.isfinite(db_sub), db_sub, np.nan)

                fig3d = go.Figure(data=[go.Surface(
                    z=db_plot, x=lots, y=t_sub,
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="δ^bid"),
                )])
                fig3d.update_layout(
                    **PLOT_KW, height=500,
                    scene=dict(
                        xaxis_title="n (lots)",
                        yaxis_title="t (s)",
                        zaxis_title="δ^bid",
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )
                show(fig3d)

        except Exception as e:
            st.error(f"ODE solve failed: {e}")
