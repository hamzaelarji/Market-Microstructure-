"""Page 1 — The Model: HJB → ODE reduction, single-asset optimal quotes.

Reproduces Paper Figures 1–5 (IG) and 10–14 (HY) interactively.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general
from market_making.core.intensity import C_coeff
from ui.styles import PALETTE, PLOT_KW, show, hero_banner, section_header
from ui.components import param_sidebar, metrics_row, insight_box


def render(PARAMS, META, HAS_REAL_DATA):
    hero_banner("📐", "The Model",
                "Guéant (2017) §2–3 — HJB → ODE reduction for optimal market making")

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        section_header("PARAMETERS")
        cfg = param_sidebar(PARAMS, key_prefix="p1", show_model=True)

    params = cfg["params"]
    gamma, T_val, xi = cfg["gamma"], cfg["T"], cfg["xi"]
    sigma, A, k, Delta, Q = params["sigma"], params["A"], params["k"], params["Delta"], int(params["Q"])

    # ── Theory card ───────────────────────────────────────────
    with st.expander("📖 Theory — click to expand", expanded=False):
        st.latex(r"dS_t = \sigma \, dW_t")
        st.markdown("**Fill intensity** (exponential):")
        st.latex(r"\Lambda(\delta) = A \, e^{-k\delta}")
        st.markdown("**CARA objective** (Model A):")
        st.latex(r"\sup_{\delta^b, \delta^a} \; \mathbb{E}\!\left[ -\exp\!\left(-\gamma(X_T + q_T S_T)\right) \right]")
        st.markdown("**ODE reduction** — the ansatz $u = -e^{-\\gamma(x+qS+\\theta)}$ yields:")
        st.latex(r"\partial_t \theta_n + \tfrac{1}{2}\gamma\sigma^2 (n\Delta)^2 "
                 r"- \mathbb{1}_{n<Q}\,H_\xi\!\left(\frac{\theta_n - \theta_{n+1}}{\Delta}\right) "
                 r"- \mathbb{1}_{n>-Q}\,H_\xi\!\left(\frac{\theta_n - \theta_{n-1}}{\Delta}\right) = 0")

        c1, c2 = st.columns(2)
        c1.markdown("**IG parameters** (CDX.NA.IG)")
        c1.code("σ = 5.83e-6  $/√s\nA = 9.10e-4  1/s\nk = 1.79e+4  1/$\nΔ = 50M  $\nQ = 4 lots")
        c2.markdown("**HY parameters** (CDX.NA.HY)")
        c2.code("σ = 2.15e-5  $/√s\nA = 1.06e-3  1/s\nk = 5.47e+3  1/$\nΔ = 10M  $\nQ = 4 lots")

    # ── Solve ODE ─────────────────────────────────────────────
    try:
        N_t = max(300, int(T_val))
        sol = solve_general(params, gamma, T_val, xi=xi, N_t=N_t)
    except Exception as e:
        st.error(f"ODE solve failed: {e}")
        return

    lots = sol["lots"]
    db = sol["delta_bid"][0, :]
    da = sol["delta_ask"][0, :]
    spread = db + da
    skew = db - da

    # ── 2×2 Quote Panel (Figs 1–5 / 10–14) ───────────────────
    section_header("OPTIMAL QUOTES AT t = 0")

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.10,
                        subplot_titles=["δ^bid(0, n)", "δ^ask(0, n)",
                                        "Spread(0, n)", "Skew(0, n)"])

    def _add(row, col, y, color, name, legend=False):
        m = np.isfinite(y)
        fig.add_trace(go.Scatter(
            x=lots[m], y=y[m], mode="lines+markers",
            name=name, marker=dict(size=7, color=color),
            line=dict(color=color, width=2),
            showlegend=legend,
        ), row=row, col=col)

    _add(1, 1, db, PALETTE[0], "bid", True)
    _add(1, 2, da, PALETTE[1], "ask", True)
    _add(2, 1, spread, PALETTE[4], "spread", True)
    _add(2, 2, skew, PALETTE[3], "skew", True)

    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(title_text="n (lots)", row=r, col=c)

    fig.update_layout(**PLOT_KW, height=520,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    show(fig)

    # ── Metrics ───────────────────────────────────────────────
    xi_D = xi * Delta
    if abs(xi_D) > 1e-12:
        d_s = (1.0 / xi_D) * np.log(1 + xi_D / k)
    else:
        d_s = 1.0 / k
    C = C_coeff(xi_D, k)
    omega = np.sqrt(gamma * sigma ** 2 / (2.0 * A * Delta * k * C))

    metrics_row({
        "δ_static": d_s,
        "ω (slope)": omega,
        "ξΔ": xi_D,
        "Spread(0,0)": float(spread[Q]) if np.isfinite(spread[Q]) else 0,
    })

    insight_box("The optimal MM skews quotes to mean-revert inventory: "
                "long → raise bid (buy less), short → lower ask (sell less). "
                "Skew(0,0) = 0 by symmetry.")

    # ── θ(0, n) value function ────────────────────────────────
    section_header("VALUE FUNCTION θ(0, n)")
    theta_0 = sol["theta"][0, :]

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=lots, y=theta_0, mode="lines+markers",
        marker=dict(size=8, color=PALETTE[3]),
        line=dict(color=PALETTE[3], width=2),
        name="θ(0, n)"))
    fig_t.add_annotation(x=0, y=theta_0[Q],
                         text=f"θ(0,0) = {theta_0[Q]:.4e}",
                         showarrow=True, arrowhead=2)
    fig_t.update_layout(**PLOT_KW, height=350,
                        xaxis_title="n (lots)", yaxis_title="θ(0, n)")
    show(fig_t)

    st.caption("θ(0,n) is concave and symmetric — its discrete gradient determines the quotes.")

    # ── Convergence / 3D ──────────────────────────────────────
    col_a, col_b = st.columns(2)
    show_conv = col_a.checkbox("Show δ^bid(t,n) convergence", value=False, key="p1_conv")
    show_3d = col_b.checkbox("Show 3D surface", value=False, key="p1_3d")

    if show_conv:
        section_header("CONVERGENCE δ^bid(t, n)")
        fig_c = go.Figure()
        times = sol["times"]
        import matplotlib.cm as cm
        colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                  for r, g, b, _ in cm.coolwarm(np.linspace(0, 1, 2 * Q - 1))]

        for i, n in enumerate(range(-Q + 1, Q)):
            i_lot = int(n + Q)
            db_t = sol["delta_bid"][:, i_lot]
            if np.all(np.isnan(db_t)):
                continue
            fig_c.add_trace(go.Scatter(
                x=times, y=db_t, mode="lines", name=f"n={n:+d}",
                line=dict(color=colors[i], width=1.5)))

        fig_c.update_layout(**PLOT_KW, height=400,
                            xaxis_title="t (s)", yaxis_title="δ^bid",
                            legend=dict(font=dict(size=9)))
        show(fig_c)
        st.caption("Quotes converge to the asymptotic regime well before t = 0. "
                   "The terminal condition at T is irrelevant.")

    if show_3d:
        section_header("3D SURFACE δ^bid(t, n)")
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
        fig3d.update_layout(**PLOT_KW, height=500,
                            scene=dict(xaxis_title="n (lots)",
                                       yaxis_title="t (s)",
                                       zaxis_title="δ^bid",
                                       bgcolor="rgba(0,0,0,0)"))
        show(fig3d)
