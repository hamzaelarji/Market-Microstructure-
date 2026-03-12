"""Page 2 — Closed-Form Approximations & Model A vs B.

Reproduces Paper Figures 6–9 (CF at σ, σ/2) and §3.4 (Model A vs B).
"""

import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general
from market_making.core.closed_form import approx_quotes
from market_making.core.intensity import C_coeff
from ui.styles import PALETTE, PLOT_KW, show, hero_banner, section_header
from ui.components import param_sidebar, metrics_row, insight_box


def render(PARAMS, META, HAS_REAL_DATA):
    hero_banner("📊", "Closed-Form Approximations & Model Comparison",
                "Guéant (2017) §4 — Guéant–Lehalle–Fernandez-Tapia formulas vs exact ODE")

    with st.sidebar:
        section_header("PARAMETERS")
        cfg = param_sidebar(PARAMS, key_prefix="p2", show_model=True)
        st.divider()
        sigma_mult = st.select_slider("σ multiplier", options=[0.25, 0.5, 1.0, 1.5, 2.0],
                                      value=1.0, key="p2_smult")
        show_avb = st.checkbox("Compare Model A vs B", value=False, key="p2_avb")
        show_sweep = st.checkbox("Show error vs σ sweep", value=False, key="p2_sweep")

    params = cfg["params"].copy()
    params["sigma"] = params["sigma"] * sigma_mult
    gamma, T_val, xi = cfg["gamma"], cfg["T"], cfg["xi"]
    Q = int(params["Q"])

    # ── Section 1: ODE vs CF ──────────────────────────────────
    section_header("ODE EXACT vs CLOSED-FORM APPROXIMATION")

    if sigma_mult != 1.0:
        st.caption(f"Showing results at σ × {sigma_mult} = {params['sigma']:.6g}")

    try:
        N_t = max(300, int(T_val))

        t0 = time.time()
        sol = solve_general(params, gamma, T_val, xi=xi, N_t=N_t)
        ode_time = time.time() - t0

        t0 = time.time()
        n_arr = np.arange(-Q + 1, Q)
        db_cf, da_cf = approx_quotes(n_arr, params, gamma, xi=xi)
        cf_time = time.time() - t0
    except Exception as e:
        st.error(f"Solve failed: {e}")
        return

    lots = sol["lots"]
    db_ode = sol["delta_bid"][0, :]
    da_ode = sol["delta_ask"][0, :]

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.10,
                        subplot_titles=["δ^bid(0, n)", "δ^ask(0, n)",
                                        "Spread(0, n)", "Skew(0, n)"])

    def _add_pair(row, col, x_ode, y_ode, x_cf, y_cf, legend=False):
        m = np.isfinite(y_ode)
        fig.add_trace(go.Scatter(
            x=x_ode[m], y=y_ode[m], mode="markers", name="ODE exact",
            marker=dict(size=8, color=PALETTE[0], symbol="x"),
            showlegend=legend), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x_cf, y=y_cf, mode="lines", name="Closed-form",
            line=dict(dash="dash", color=PALETTE[1], width=2),
            showlegend=legend), row=row, col=col)

    spread_ode = db_ode + da_ode
    skew_ode = db_ode - da_ode
    spread_cf = db_cf + da_cf
    skew_cf = db_cf - da_cf

    _add_pair(1, 1, lots, db_ode, n_arr, db_cf, legend=True)
    _add_pair(1, 2, lots, da_ode, n_arr, da_cf)
    _add_pair(2, 1, lots, spread_ode, n_arr, spread_cf)
    _add_pair(2, 2, lots, skew_ode, n_arr, skew_cf)

    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(title_text="n", row=r, col=c)

    fig.update_layout(**PLOT_KW, height=520,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    show(fig)

    # ── Error metrics ─────────────────────────────────────────
    # Compare on common inventory range
    db_ode_sub = sol["delta_bid"][0, 1:-1]  # n = -Q+1 to Q-1
    da_ode_sub = sol["delta_ask"][0, 1:-1]
    mask_b = np.isfinite(db_ode_sub) & (len(db_cf) == len(db_ode_sub))
    if len(db_cf) == len(db_ode_sub):
        err_bid = np.abs(db_ode_sub - db_cf)
        err_ask = np.abs(da_ode_sub - da_cf)
        max_err = float(np.nanmax(np.concatenate([err_bid, err_ask])))
        spread_err = np.abs((db_ode_sub + da_ode_sub) - (db_cf + da_cf))
        mean_spread = float(np.nanmean(db_cf + da_cf))
        rel_spread_err = float(np.nanmax(spread_err) / mean_spread) if mean_spread > 0 else 0
    else:
        max_err, rel_spread_err = 0, 0

    metrics_row({
        "max |δ_ODE − δ_CF|": max_err,
        "Rel. spread error": f"{rel_spread_err:.2%}",
        "ODE time": f"{ode_time:.3f}s",
        "CF time": f"{cf_time * 1000:.2f}ms",
        "Speedup": f"{ode_time / max(cf_time, 1e-9):.0f}×",
    })

    insight_box(f"The closed-form approximation is {ode_time/max(cf_time,1e-9):.0f}× faster. "
                "Quality depends on ωΔ: smaller ωΔ → better approximation. "
                "Halving σ dramatically improves CF accuracy.")

    # ── Error vs σ sweep ──────────────────────────────────────
    if show_sweep:
        section_header("ERROR vs σ MULTIPLIER")
        mults = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        max_errs = []
        for m in mults:
            p_tmp = cfg["params"].copy()
            p_tmp["sigma"] = p_tmp["sigma"] * m
            try:
                sol_tmp = solve_general(p_tmp, gamma, T_val, xi=xi, N_t=N_t)
                db_tmp = sol_tmp["delta_bid"][0, 1:-1]
                db_cf_tmp, _ = approx_quotes(n_arr, p_tmp, gamma, xi=xi)
                if len(db_cf_tmp) == len(db_tmp):
                    max_errs.append(float(np.nanmax(np.abs(db_tmp - db_cf_tmp))))
                else:
                    max_errs.append(0)
            except Exception:
                max_errs.append(0)

        fig_sw = go.Figure()
        fig_sw.add_trace(go.Scatter(
            x=mults, y=max_errs, mode="lines+markers",
            marker=dict(size=8, color=PALETTE[0]),
            line=dict(color=PALETTE[0], width=2)))
        fig_sw.update_layout(**PLOT_KW, height=350,
                             xaxis_title="σ multiplier",
                             yaxis_title="max |δ_ODE − δ_CF|")
        show(fig_sw)

    # ── Section 2: Model A vs B ───────────────────────────────
    if show_avb:
        section_header("MODEL A (ξ = γ) vs MODEL B (ξ = 0)")

        try:
            sol_a = solve_general(params, gamma, T_val, xi=gamma, N_t=N_t)
            sol_b = solve_general(params, gamma, T_val, xi=0.0, N_t=N_t)
        except Exception as e:
            st.error(f"Comparison failed: {e}")
            return

        fig_ab = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                               subplot_titles=["Spread(0, n)", "Skew(0, n)"])

        spread_a = sol_a["delta_bid"][0, :] + sol_a["delta_ask"][0, :]
        spread_b = sol_b["delta_bid"][0, :] + sol_b["delta_ask"][0, :]
        skew_a = sol_a["delta_bid"][0, :] - sol_a["delta_ask"][0, :]
        skew_b = sol_b["delta_bid"][0, :] - sol_b["delta_ask"][0, :]

        for y, name, color, show_l in [
            (spread_a, "Model A (ξ=γ)", PALETTE[0], True),
            (spread_b, "Model B (ξ=0)", PALETTE[1], True),
        ]:
            m = np.isfinite(y)
            fig_ab.add_trace(go.Scatter(
                x=lots[m], y=y[m], mode="lines+markers", name=name,
                marker=dict(size=6, color=color),
                line=dict(color=color, width=2),
                showlegend=show_l), row=1, col=1)

        for y, name, color in [
            (skew_a, "Model A", PALETTE[0]),
            (skew_b, "Model B", PALETTE[1]),
        ]:
            m = np.isfinite(y)
            fig_ab.add_trace(go.Scatter(
                x=lots[m], y=y[m], mode="lines+markers", name=name,
                marker=dict(size=6, color=color),
                line=dict(color=color, width=2),
                showlegend=False), row=1, col=2)

        fig_ab.update_xaxes(title_text="n (lots)", row=1, col=1)
        fig_ab.update_xaxes(title_text="n (lots)", row=1, col=2)
        fig_ab.update_layout(**PLOT_KW, height=380,
                             legend=dict(orientation="h", yanchor="bottom", y=1.05))
        show(fig_ab)

        # Comparative statics table
        Delta = params["Delta"]
        k = params["k"]
        xi_a, xi_b = gamma, 0.0
        d_s_a = (1.0 / (gamma * Delta)) * np.log(1 + gamma * Delta / k) if abs(gamma * Delta) > 1e-12 else 1.0 / k
        d_s_b = 1.0 / k
        C_a = C_coeff(gamma * Delta, k)
        C_b = C_coeff(0, k)
        omega_a = np.sqrt(gamma * params["sigma"] ** 2 / (2.0 * params["A"] * Delta * k * C_a))
        omega_b = np.sqrt(gamma * params["sigma"] ** 2 / (2.0 * params["A"] * Delta * k * C_b))

        st.markdown("**Comparative statics**")
        st.dataframe({
            "": ["δ_static", "ω (slope)", "Spread(0,0)", "ξΔ"],
            "Model A (ξ=γ)": [f"{d_s_a:.6g}", f"{omega_a:.6g}",
                              f"{2*d_s_a + omega_a*Delta:.6g}", f"{gamma*Delta:.6g}"],
            "Model B (ξ=0)": [f"{d_s_b:.6g}", f"{omega_b:.6g}",
                              f"{2*d_s_b + omega_b*Delta:.6g}", "0"],
        }, hide_index=True)

        insight_box("ξ = γ (Model A) penalises non-execution risk in addition to price risk. "
                    "ξ = 0 (Model B) only penalises inventory. "
                    "Model A quotes tighter but skews more aggressively.")
