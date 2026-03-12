"""Page 7 — Hawkes Self-Excitation: When Fills Cluster.

Compares Poisson vs Hawkes fill dynamics and shows that the Guéant
policy degrades when fills cluster.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general
from market_making.core.intensity import fill_prob, Lambda
from market_making.core.hawkes import (
    HawkesState, DEFAULT_HAWKES_CFG, softplus, lambda_hawkes
)
from ui.styles import PALETTE, PLOT_KW, show, hero_banner, section_header
from ui.components import param_sidebar, metrics_row, insight_box

DEFAULT_GAMMA = 0.01


def _simulate_path(params, gamma, T, N_t, fill_mode, hawkes_cfg, seed):
    """Simulate one path with Poisson or Hawkes fills, return traces."""
    xi = gamma
    sol = solve_general(params, gamma, T, xi=xi, N_t=N_t)
    sigma, A, k = params["sigma"], params["A"], params["k"]
    Delta, Q = params["Delta"], int(params["Q"])
    dt = T / N_t
    rng = np.random.default_rng(seed)

    times = np.linspace(0, T, N_t + 1)
    price = np.zeros(N_t + 1)
    inv = np.zeros(N_t + 1, dtype=int)
    mtm = np.zeros(N_t + 1)
    lambda_bid_arr = np.zeros(N_t)
    lambda_ask_arr = np.zeros(N_t)
    y_bid_arr = np.zeros(N_t)
    y_ask_arr = np.zeros(N_t)
    fill_times_bid = []
    fill_times_ask = []

    S, X, n = 0.0, 0.0, 0
    hs = HawkesState(hawkes_cfg) if fill_mode == "hawkes" else None

    for t_idx in range(N_t):
        i_lot = n + Q
        db = sol["delta_bid"][t_idx, i_lot] if (n < Q and np.isfinite(sol["delta_bid"][t_idx, i_lot])) else np.inf
        da = sol["delta_ask"][t_idx, i_lot] if (n > -Q and np.isfinite(sol["delta_ask"][t_idx, i_lot])) else np.inf

        S += sigma * np.sqrt(dt) * rng.standard_normal()

        # Intensities
        mu_b = A * np.exp(-k * max(db, 0)) if db < np.inf else 0
        mu_a = A * np.exp(-k * max(da, 0)) if da < np.inf else 0

        if fill_mode == "hawkes" and hs is not None:
            lam_b = hs.lambda_bid(mu_b)
            lam_a = hs.lambda_ask(mu_a)
            y_bid_arr[t_idx] = hs.y_bid
            y_ask_arr[t_idx] = hs.y_ask
        else:
            lam_b = mu_b
            lam_a = mu_a

        lambda_bid_arr[t_idx] = lam_b
        lambda_ask_arr[t_idx] = lam_a

        bid_fill = rng.random() < fill_prob(lam_b, dt) if db < np.inf else False
        ask_fill = rng.random() < fill_prob(lam_a, dt) if da < np.inf else False

        if bid_fill and n < Q:
            X -= (S - db) * Delta
            n += 1
            fill_times_bid.append(times[t_idx])

        if ask_fill and n > -Q:
            X += (S + da) * Delta
            n -= 1
            fill_times_ask.append(times[t_idx])

        if fill_mode == "hawkes" and hs is not None:
            hs.step(dt, bid_fill, ask_fill)

        price[t_idx + 1] = S
        inv[t_idx + 1] = n
        mtm[t_idx + 1] = X + n * Delta * S

    pnl = X + n * Delta * S
    return {
        "times": times, "price": price, "inv": inv, "mtm": mtm,
        "pnl": pnl, "lambda_bid": lambda_bid_arr, "lambda_ask": lambda_ask_arr,
        "y_bid": y_bid_arr, "y_ask": y_ask_arr,
        "fill_times_bid": np.array(fill_times_bid),
        "fill_times_ask": np.array(fill_times_ask),
    }


def _run_mc(params, gamma, T, N_t, N_sim, fill_mode, hawkes_cfg, seed):
    """Run N_sim paths, return PnL array."""
    rng = np.random.default_rng(seed)
    pnls = []
    for i in range(N_sim):
        res = _simulate_path(params, gamma, T, N_t, fill_mode, hawkes_cfg,
                             seed=int(rng.integers(1e9)))
        pnls.append(res["pnl"])
    return np.array(pnls)


def render(PARAMS, META, HAS_REAL_DATA):
    hero_banner("⚡", "Hawkes Self-Excitation",
                "When fills cluster: Poisson vs Hawkes dynamics")

    with st.sidebar:
        section_header("PARAMETERS")
        cfg = param_sidebar(PARAMS, key_prefix="p7")
        st.divider()
        section_header("HAWKES CONFIG")
        alpha_self = st.slider("α_self (self-excitation)", 0.0, 5.0, 2.0, 0.5, key="p7_as")
        alpha_cross = st.slider("α_cross (cross-excitation)", 0.0, 3.0, 0.5, 0.25, key="p7_ac")
        beta = st.slider("β (decay rate)", 1.0, 30.0, 10.0, 1.0, key="p7_beta")

    params = cfg["params"]
    gamma, T_val = cfg["gamma"], cfg["T"]
    hawkes_cfg = {"beta": beta, "alpha_self": alpha_self, "alpha_cross": alpha_cross}
    N_t = max(300, int(T_val))

    # ── Theory ────────────────────────────────────────────────
    with st.expander("📖 Theory", expanded=False):
        st.markdown("**Hawkes intensity** (Lalor & Swishchuk 2025):")
        st.latex(r"\lambda_{\text{bid}}(t) = \text{softplus}\!\left(\Lambda(\delta^b) + y_{\text{bid}}(t)\right)")
        st.markdown("Excitation dynamics:")
        st.latex(r"y(t+dt) = y(t) \cdot e^{-\beta \, dt} + \alpha_{\text{self}} \cdot \mathbb{1}_{\text{fill}}")
        st.markdown("Fills cluster in bursts: each fill increases the probability of the next. "
                    "This adds 2 state variables (y_bid, y_ask) to the system, "
                    "breaking the 2D → 4D dimensional reduction that makes Guéant's ODE tractable.")

    # ── Single-path comparison ────────────────────────────────
    section_header("SINGLE PATH: POISSON vs HAWKES")

    seed = st.number_input("Random seed", value=42, key="p7_seed")

    path_p = _simulate_path(params, gamma, T_val, N_t, "poisson", hawkes_cfg, seed=int(seed))
    path_h = _simulate_path(params, gamma, T_val, N_t, "hawkes", hawkes_cfg, seed=int(seed))

    t_arr = np.linspace(0, T_val, N_t)

    # Intensity traces
    fig_lam = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                            subplot_titles=["Poisson λ_bid(t)", "Hawkes λ_bid(t)"])

    fig_lam.add_trace(go.Scatter(x=t_arr, y=path_p["lambda_bid"],
        mode="lines", line=dict(color=PALETTE[6], width=1), showlegend=False), row=1, col=1)
    fig_lam.add_trace(go.Scatter(x=t_arr, y=path_h["lambda_bid"],
        mode="lines", line=dict(color=PALETTE[0], width=1), showlegend=False), row=1, col=2)

    fig_lam.update_layout(**PLOT_KW, height=280)
    show(fig_lam)

    # Excitation state
    section_header("EXCITATION DYNAMICS y(t)")
    fig_y = go.Figure()
    fig_y.add_trace(go.Scatter(x=t_arr, y=path_h["y_bid"], mode="lines",
        name="y_bid", line=dict(color=PALETTE[0], width=1.5)))
    fig_y.add_trace(go.Scatter(x=t_arr, y=path_h["y_ask"], mode="lines",
        name="y_ask", line=dict(color=PALETTE[1], width=1.5)))
    fig_y.update_layout(**PLOT_KW, height=250,
                        xaxis_title="t (s)", yaxis_title="Excitation y(t)")
    show(fig_y)

    # Inter-fill times
    section_header("FILL CLUSTERING")
    all_fills_p = np.sort(np.concatenate([path_p["fill_times_bid"], path_p["fill_times_ask"]]))
    all_fills_h = np.sort(np.concatenate([path_h["fill_times_bid"], path_h["fill_times_ask"]]))

    if len(all_fills_p) > 2 and len(all_fills_h) > 2:
        ift_p = np.diff(all_fills_p)
        ift_h = np.diff(all_fills_h)

        fig_ift = go.Figure()
        fig_ift.add_trace(go.Histogram(x=ift_p, name="Poisson", opacity=0.6,
            marker_color=PALETTE[6], nbinsx=40))
        fig_ift.add_trace(go.Histogram(x=ift_h, name="Hawkes", opacity=0.6,
            marker_color=PALETTE[0], nbinsx=40))
        fig_ift.update_layout(**PLOT_KW, height=300, barmode="overlay",
                              xaxis_title="Inter-fill time (s)", yaxis_title="Count")
        show(fig_ift)
        st.caption("Hawkes inter-fill times have a heavier left tail — fills come in bursts.")
    else:
        st.caption("Not enough fills for inter-fill time histogram.")

    # ── MC: policy mismatch ──
    section_header("POLICY MISMATCH: GUÉANT POLICY IN HAWKES WORLD")

    n_sim = st.slider("N_sim for MC", 100, 2000, 500, 100, key="p7_nsim")
    if st.button("▶ Run MC comparison", type="primary", key="p7_run"):
        with st.spinner("Running Poisson + Hawkes MC..."):
            pnl_p = _run_mc(params, gamma, T_val, N_t, n_sim, "poisson", hawkes_cfg, seed=42)
            pnl_h = _run_mc(params, gamma, T_val, N_t, n_sim, "hawkes", hawkes_cfg, seed=42)

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=pnl_p, name="Poisson world", opacity=0.6,
            marker_color=PALETTE[6], nbinsx=40))
        fig_mc.add_trace(go.Histogram(x=pnl_h, name="Hawkes world", opacity=0.6,
            marker_color=PALETTE[0], nbinsx=40))
        fig_mc.update_layout(**PLOT_KW, height=350, barmode="overlay",
                             xaxis_title="Terminal PnL ($)", yaxis_title="Count")
        show(fig_mc)

        sp = float(np.mean(pnl_p) / max(np.std(pnl_p), 1e-12))
        sh = float(np.mean(pnl_h) / max(np.std(pnl_h), 1e-12))
        metrics_row({
            "Sharpe (Poisson)": sp,
            "Sharpe (Hawkes)": sh,
            "Degradation": f"{(1 - sh / max(sp, 1e-12)) * 100:.1f}%",
            "Poisson fills": f"{len(all_fills_p)}",
            "Hawkes fills": f"{len(all_fills_h)}",
        })

        insight_box("The same Guéant policy underperforms in the Hawkes world because "
                    "it ignores fill clustering. Hawkes adds 2 state variables (y_b, y_a), "
                    "expanding the state space from 2D to 4D and breaking analytical tractability. "
                    "This motivates the RL approach on the next page.")
