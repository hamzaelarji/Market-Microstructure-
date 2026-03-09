"""ui/tabs/hawkes.py — Tab 7: Hawkes vs Poisson fill dynamics."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general

try:
    from market_making.core.hawkes import (
        HawkesState, lambda_hawkes, softplus,
        fill_prob_from_intensity, DEFAULT_HAWKES_CFG,
    )
    HAWKES_AVAILABLE = True
except ImportError:
    HAWKES_AVAILABLE = False

DEFAULT_GAMMA = 0.01


def render_hawkes(PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
                  load_mid_prices, run_quick_mc):
    """Render Tab 7 — Hawkes vs Poisson: compare fill dynamics."""
    SYMBOLS = list(PARAMS.keys())

    if not HAWKES_AVAILABLE:
        st.warning("Hawkes module not found — ensure `market_making/core/hawkes.py` exists.")
        return

    st.caption(
        "Same ODE-optimal quotes, two fill engines. "
        "Hawkes adds self- and cross-excitation → fill clustering, "
        "higher variance, and breaks the Guéant dimensional reduction."
    )

    c_h1, c_h2 = st.columns([1, 3])

    with c_h1:
        hk_sym = st.selectbox("Symbol", SYMBOLS, key="hk_sym")
        hk_params = PARAMS[hk_sym]
        hk_gamma = st.number_input("γ", value=DEFAULT_GAMMA, format="%.4f", key="hk_g")
        hk_T = st.number_input("T (s)", value=3600.0, step=600.0, key="hk_T")
        hk_nsim = st.slider("MC paths", 200, 3000, 600, step=100, key="hk_nsim")

        st.markdown("**Hawkes parameters**")
        hk_beta = st.number_input("β (decay)", value=10.0, step=1.0, key="hk_b")
        hk_as = st.number_input("α_self", value=2.0, step=0.5, key="hk_as")
        hk_ac = st.number_input("α_cross", value=0.5, step=0.1, key="hk_ac")

        run_hk = st.button("Run", type="primary", key="hk_run")

    with c_h2:
        if run_hk:
            N_t_hk = max(300, int(hk_T))
            dt_hk = hk_T / N_t_hk
            Q_hk = int(hk_params["Q"])
            ns_hk = 2 * Q_hk + 1
            A_hk = hk_params["A"]
            k_hk = hk_params["k"]
            Delta_hk = hk_params["Delta"]
            sigma_hk = hk_params["sigma"]
            hawkes_cfg = {"beta": hk_beta, "alpha_self": hk_as, "alpha_cross": hk_ac}

            with st.spinner("Solving ODE …"):
                sol_hk = solve_general(hk_params, hk_gamma, hk_T,
                                       xi=hk_gamma, N_t=N_t_hk)
                db_hk = sol_hk["delta_bid"]
                da_hk = sol_hk["delta_ask"]

            seed_hk = 42
            rng_p = np.random.default_rng(seed_hk)
            z_hk = rng_p.standard_normal((hk_nsim, N_t_hk))
            ub_hk = rng_p.uniform(size=(hk_nsim, N_t_hk))
            ua_hk = rng_p.uniform(size=(hk_nsim, N_t_hk))

            def _get_quotes(t, n):
                il = n + Q_hk
                if not (0 <= il < ns_hk):
                    return np.inf, np.inf
                d_b = db_hk[t, il] if (n < Q_hk and np.isfinite(db_hk[t, il])) else np.inf
                d_a = da_hk[t, il] if (n > -Q_hk and np.isfinite(da_hk[t, il])) else np.inf
                return d_b, d_a

            with st.spinner("Simulating Poisson …"):
                pnl_P = np.zeros(hk_nsim)
                fills_P = np.zeros(hk_nsim, dtype=int)
                inv_P = np.zeros((hk_nsim, N_t_hk + 1), dtype=int)

                for m in range(hk_nsim):
                    S, X, n = 0.0, 0.0, 0
                    for t in range(N_t_hk):
                        d_b, d_a = _get_quotes(t, n)
                        S += sigma_hk * np.sqrt(dt_hk) * z_hk[m, t]
                        if d_b < np.inf:
                            lam = A_hk * np.exp(-k_hk * max(d_b, 0))
                            if ub_hk[m, t] < 1 - np.exp(-lam * dt_hk):
                                X -= (S - d_b) * Delta_hk; n += 1; fills_P[m] += 1
                        if d_a < np.inf:
                            lam = A_hk * np.exp(-k_hk * max(d_a, 0))
                            if ua_hk[m, t] < 1 - np.exp(-lam * dt_hk):
                                X += (S + d_a) * Delta_hk; n -= 1; fills_P[m] += 1
                        inv_P[m, t + 1] = n
                    pnl_P[m] = X + n * Delta_hk * S

            with st.spinner("Simulating Hawkes …"):
                rng_h = np.random.default_rng(seed_hk + 1)
                ub_H = rng_h.uniform(size=(hk_nsim, N_t_hk))
                ua_H = rng_h.uniform(size=(hk_nsim, N_t_hk))

                pnl_H = np.zeros(hk_nsim)
                fills_H = np.zeros(hk_nsim, dtype=int)
                inv_H = np.zeros((hk_nsim, N_t_hk + 1), dtype=int)
                exc_hist = np.zeros((hk_nsim, N_t_hk))

                for m in range(hk_nsim):
                    S, X, n = 0.0, 0.0, 0
                    hs = HawkesState(hawkes_cfg)
                    for t in range(N_t_hk):
                        d_b, d_a = _get_quotes(t, n)
                        S += sigma_hk * np.sqrt(dt_hk) * z_hk[m, t]

                        mu_b = A_hk * np.exp(-k_hk * max(d_b, 0)) if d_b < np.inf else 0.0
                        mu_a = A_hk * np.exp(-k_hk * max(d_a, 0)) if d_a < np.inf else 0.0

                        lam_b = hs.lambda_bid(mu_b)
                        lam_a = hs.lambda_ask(mu_a)

                        fb = (ub_H[m, t] < fill_prob_from_intensity(lam_b, dt_hk)) if d_b < np.inf else False
                        fa = (ua_H[m, t] < fill_prob_from_intensity(lam_a, dt_hk)) if d_a < np.inf else False

                        if fb:
                            X -= (S - d_b) * Delta_hk; n += 1; fills_H[m] += 1
                        if fa:
                            X += (S + d_a) * Delta_hk; n -= 1; fills_H[m] += 1

                        hs.step(dt_hk, fb, fa)
                        exc_hist[m, t] = hs.y_bid + hs.y_ask
                        inv_H[m, t + 1] = n
                    pnl_H[m] = X + n * Delta_hk * S

            st.session_state["hk_data"] = dict(
                pnl_P=pnl_P, pnl_H=pnl_H,
                fills_P=fills_P, fills_H=fills_H,
                inv_P=inv_P, inv_H=inv_H,
                exc_hist=exc_hist,
                T=hk_T, N_t=N_t_hk,
            )

        if "hk_data" in st.session_state:
            hd = st.session_state["hk_data"]
            t_ax_h = np.linspace(0, hd["T"], hd["N_t"] + 1)
            C_P, C_H = "#74b9ff", "#ff7675"

            def _hk_sum(pnl, fills, label):
                return {
                    "Model": label,
                    "E[P&L]": np.mean(pnl), "Std[P&L]": np.std(pnl),
                    "Sharpe": np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else 0,
                    "Mean fills": np.mean(fills),
                    "Fills/h": np.mean(fills) / max(hd["T"] / 3600, 1e-6),
                }
            st.dataframe(
                pd.DataFrame([
                    _hk_sum(hd["pnl_P"], hd["fills_P"], "Poisson"),
                    _hk_sum(hd["pnl_H"], hd["fills_H"], "Hawkes"),
                ]).set_index("Model").style.format({
                    "E[P&L]": "{:+,.2f}", "Std[P&L]": "{:,.2f}",
                    "Sharpe": "{:.3f}", "Mean fills": "{:.1f}",
                    "Fills/h": "{:.1f}",
                }),
                use_container_width=True,
            )

            fig = make_subplots(
                rows=2, cols=2, vertical_spacing=0.14,
                subplot_titles=[
                    "P&L distribution", "Fill count distribution",
                    "E[|inventory|]", "Mean excitation state",
                ],
            )
            fig.add_trace(go.Histogram(x=hd["pnl_P"], name="Poisson",
                opacity=0.5, marker_color=C_P, nbinsx=60), row=1, col=1)
            fig.add_trace(go.Histogram(x=hd["pnl_H"], name="Hawkes",
                opacity=0.5, marker_color=C_H, nbinsx=60), row=1, col=1)

            fig.add_trace(go.Histogram(x=hd["fills_P"], name="Poisson",
                opacity=0.5, marker_color=C_P, showlegend=False), row=1, col=2)
            fig.add_trace(go.Histogram(x=hd["fills_H"], name="Hawkes",
                opacity=0.5, marker_color=C_H, showlegend=False), row=1, col=2)

            step_h = max(1, hd["inv_P"].shape[1] // 500)
            fig.add_trace(go.Scatter(
                x=t_ax_h[::step_h],
                y=np.mean(np.abs(hd["inv_P"][:, ::step_h].astype(float)), axis=0),
                name="Poisson", line=dict(color=C_P, width=2), showlegend=False,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=t_ax_h[::step_h],
                y=np.mean(np.abs(hd["inv_H"][:, ::step_h].astype(float)), axis=0),
                name="Hawkes", line=dict(color=C_H, width=2), showlegend=False,
            ), row=2, col=1)

            step_e = max(1, hd["exc_hist"].shape[1] // 500)
            mean_exc = np.mean(hd["exc_hist"][:, ::step_e], axis=0)
            fig.add_trace(go.Scatter(
                x=np.linspace(0, hd["T"], len(mean_exc)),
                y=mean_exc, line=dict(color=C_H, width=2),
                showlegend=False,
            ), row=2, col=2)

            fig.update_layout(**PLOT_KW, height=560, barmode="overlay")
            show(fig)

            st.info(
                "**Key finding:** Hawkes fills introduce clustering — bursts of fills "
                "then quiet periods. This increases P&L variance. Crucially, the Hawkes "
                "dynamics **destroy the dimensional reduction** central to the Guéant "
                "framework: the Hamiltonian is no longer expressible in closed form, "
                "so ODE-optimal quotes are no longer truly optimal under Hawkes."
            )
