"""ui/tabs/live_sim.py — Animated live market maker simulation theater.

Pre-computes one Monte Carlo path and replays it as an animated Plotly chart
with Play / Pause / scrub controls — all client-side via Plotly frames.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general
from market_making.simulation.simulator import simulate_1d
from ui.styles import TEAL, RED, YELLOW, PURPLE, GRAY, hero_banner, section_header


# ─── Path computation ─────────────────────────────────────────────────────────

def _compute_live_path(params: dict, gamma: float, T: float, N_t_anim: int, seed: int) -> dict:
    """Compute one simulation path and extract all data needed for animation.

    Simulates at fine resolution (≥1 step/s to keep Poisson lambdas valid)
    then subsamples down to N_t_anim points for the animation.
    """
    # Physics resolution: at least 1 step per second
    N_t_sim = max(N_t_anim * 4, int(T))
    sol = solve_general(params, gamma, T, xi=gamma, N_t=N_t_sim)

    sim = simulate_1d(sol, params, gamma, T, N_sim=1, seed=seed, return_utility=False)

    times_f = sim["times"]        # (N_t_sim+1,)
    price_f = sim["price"][0]     # (N_t_sim+1,)
    inv_f   = sim["inventory"][0] # (N_t_sim+1,) integer lots
    mtm_f   = sim["mtm"][0]       # (N_t_sim+1,)

    Q     = int(params["Q"])
    Delta = params["Delta"]

    # ── Reconstruct bid/ask quote prices at each fine timestep ─────────────
    db_table = sol["delta_bid"]   # (N_t_sim, 2Q+1)
    da_table = sol["delta_ask"]   # (N_t_sim, 2Q+1)

    bq_f = np.full(N_t_sim + 1, np.nan)
    aq_f = np.full(N_t_sim + 1, np.nan)

    for t in range(N_t_sim):
        n = int(np.clip(inv_f[t], -Q, Q))
        i_lot = n + Q
        db = db_table[t, i_lot]
        da = da_table[t, i_lot]
        if np.isfinite(db):
            bq_f[t] = price_f[t] - db
        if np.isfinite(da):
            aq_f[t] = price_f[t] + da
    # Fill last point
    for arr in (bq_f, aq_f):
        if not np.isfinite(arr[-1]) and np.isfinite(arr[-2]):
            arr[-1] = arr[-2]

    # ── Detect fill events from inventory changes (fine grid) ──────────────
    bid_fill_t_f, bid_fill_p_f = [], []
    ask_fill_t_f, ask_fill_p_f = [], []

    for t in range(1, N_t_sim + 1):
        di = int(inv_f[t]) - int(inv_f[t - 1])
        if di > 0:
            bid_fill_t_f.append(times_f[t])
            bid_fill_p_f.append(bq_f[t] if np.isfinite(bq_f[t]) else price_f[t])
        elif di < 0:
            ask_fill_t_f.append(times_f[t])
            ask_fill_p_f.append(aq_f[t] if np.isfinite(aq_f[t]) else price_f[t])

    # ── Subsample to N_t_anim for the animation ─────────────────────────────
    idx = np.linspace(0, N_t_sim, N_t_anim + 1, dtype=int)
    times     = times_f[idx]
    price     = price_f[idx]
    inv       = inv_f[idx]
    mtm       = mtm_f[idx]
    bid_quote = bq_f[idx]
    ask_quote = aq_f[idx]
    N_t       = N_t_anim

    bid_fill_t = np.array(bid_fill_t_f)
    bid_fill_p = np.array(bid_fill_p_f)
    ask_fill_t = np.array(ask_fill_t_f)
    ask_fill_p = np.array(ask_fill_p_f)

    return {
        "times":       times,
        "price":       price,
        "bid_quote":   bid_quote,
        "ask_quote":   ask_quote,
        "inventory":   inv,
        "mtm":         mtm,
        "bid_fill_t":  np.array(bid_fill_t),
        "bid_fill_p":  np.array(bid_fill_p),
        "ask_fill_t":  np.array(ask_fill_t),
        "ask_fill_p":  np.array(ask_fill_p),
        "Q":           Q,
        "Delta":       Delta,
        "N_t":         N_t,
    }


# ─── Inventory bar color ──────────────────────────────────────────────────────

def _inv_colors(inv_arr: np.ndarray, Q: int) -> list[str]:
    colors = []
    for n in inv_arr:
        frac = abs(n) / max(Q, 1)
        if n > 0:
            r = int(255 * frac)
            g = int(107 + (212 - 107) * (1 - frac))
            b = int(107 + (170 - 107) * (1 - frac))
        elif n < 0:
            r = 255
            g = int(107 * (1 - frac))
            b = int(107 * (1 - frac))
        else:
            r, g, b = 0, 212, 170
        colors.append(f"rgb({r},{g},{b})")
    return colors


# ─── Build Plotly animation ───────────────────────────────────────────────────

def _build_animation(path: dict, PLOT_KW: dict) -> go.Figure:
    """Build the Plotly figure with animation frames (3-panel layout)."""
    times     = path["times"]
    price     = path["price"]
    bq        = path["bid_quote"]
    aq        = path["ask_quote"]
    inv       = path["inventory"]
    mtm       = path["mtm"]
    Q         = path["Q"]
    N_t       = path["N_t"]

    bf_t = path["bid_fill_t"]
    bf_p = path["bid_fill_p"]
    af_t = path["ask_fill_t"]
    af_p = path["ask_fill_p"]

    # Subsample to at most 100 animation frames for performance
    n_frames = min(100, N_t)
    frame_indices = np.linspace(2, N_t, n_frames, dtype=int)

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.52, 0.24, 0.24],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=["", "", ""],
    )

    # ── Initial traces (frame 0) ─────────────────────────────────────────────
    i0 = int(frame_indices[0])

    # Row 1 — price line
    fig.add_trace(go.Scatter(
        x=times[:i0], y=price[:i0],
        name="Mid Price", mode="lines",
        line=dict(color="#666e78", width=1.5),
    ), row=1, col=1)

    # Row 1 — bid quote
    fig.add_trace(go.Scatter(
        x=times[:i0], y=bq[:i0],
        name="Bid Quote", mode="lines",
        line=dict(color=RED, width=1.8, dash="dot"),
    ), row=1, col=1)

    # Row 1 — ask quote
    fig.add_trace(go.Scatter(
        x=times[:i0], y=aq[:i0],
        name="Ask Quote", mode="lines",
        line=dict(color=TEAL, width=1.8, dash="dot"),
    ), row=1, col=1)

    # Row 1 — bid fill quote fills area between bid and ask
    fig.add_trace(go.Scatter(
        x=times[:i0], y=aq[:i0],
        fill="tonexty",
        fillcolor="rgba(0,212,170,0.05)",
        mode="none",
        showlegend=False,
        name="_spread_fill",
    ), row=1, col=1)

    # Row 1 — bid fills scatter
    m0b = bf_t <= times[i0] if len(bf_t) else np.array([], dtype=bool)
    fig.add_trace(go.Scatter(
        x=bf_t[m0b], y=bf_p[m0b],
        name="Bid Fill ★", mode="markers",
        marker=dict(symbol="star", size=13, color=RED,
                    line=dict(color="white", width=0.5)),
    ), row=1, col=1)

    # Row 1 — ask fills scatter
    m0a = af_t <= times[i0] if len(af_t) else np.array([], dtype=bool)
    fig.add_trace(go.Scatter(
        x=af_t[m0a], y=af_p[m0a],
        name="Ask Fill ★", mode="markers",
        marker=dict(symbol="star", size=13, color=TEAL,
                    line=dict(color="white", width=0.5)),
    ), row=1, col=1)

    # Row 2 — inventory bars
    ic0 = _inv_colors(inv[:i0], Q)
    fig.add_trace(go.Bar(
        x=times[:i0], y=inv[:i0].astype(float),
        name="Inventory", showlegend=False,
        marker=dict(color=ic0, line=dict(width=0)),
    ), row=2, col=1)

    # Row 3 — P&L
    fig.add_trace(go.Scatter(
        x=times[:i0], y=mtm[:i0],
        name="MtM P&L", mode="lines", showlegend=False,
        line=dict(color=YELLOW, width=2),
        fill="tozeroy",
        fillcolor="rgba(255,217,61,0.07)",
    ), row=3, col=1)

    # ── Animation frames ─────────────────────────────────────────────────────
    frames = []
    for fi, i in enumerate(frame_indices):
        mb = bf_t <= times[i] if len(bf_t) else np.array([], dtype=bool)
        ma = af_t <= times[i] if len(af_t) else np.array([], dtype=bool)
        ic = _inv_colors(inv[:i], Q)

        frame_data = [
            go.Scatter(x=times[:i], y=price[:i]),
            go.Scatter(x=times[:i], y=bq[:i]),
            go.Scatter(x=times[:i], y=aq[:i]),
            go.Scatter(x=times[:i], y=aq[:i]),        # spread fill
            go.Scatter(x=bf_t[mb],  y=bf_p[mb]),
            go.Scatter(x=af_t[ma],  y=af_p[ma]),
            go.Bar(x=times[:i], y=inv[:i].astype(float),
                   marker=dict(color=ic, line=dict(width=0))),
            go.Scatter(x=times[:i], y=mtm[:i]),
        ]
        frames.append(go.Frame(data=frame_data, name=str(fi)))

    fig.frames = frames

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        **PLOT_KW,
        height=680,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(size=10),
        ),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.12,
            "y": 1.065,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "#12151c",
            "bordercolor": "#1e2330",
            "font": dict(size=12, color="#c8cdd5"),
            "buttons": [
                {
                    "label": "▶  Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 90, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "⏸  Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {
                "prefix": "t = ",
                "suffix": " s",
                "visible": True,
                "font": dict(size=11, color="#7a8290"),
            },
            "pad": {"t": 50, "b": 10},
            "bgcolor": "#12151c",
            "bordercolor": "#1e2330",
            "tickcolor": "#1e2330",
            "font": dict(size=9, color="#555e6e"),
            "steps": [
                {
                    "args": [
                        [str(fi)],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": f"{times[i]:.0f}",
                    "method": "animate",
                }
                for fi, i in enumerate(frame_indices)
            ],
        }],
    )

    # Axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1,
                     titlefont=dict(size=10), tickfont=dict(size=9))
    fig.update_yaxes(title_text="Lots", row=2, col=1,
                     range=[-Q - 0.6, Q + 0.6],
                     titlefont=dict(size=10), tickfont=dict(size=9))
    fig.update_yaxes(title_text="P&L ($)", row=3, col=1,
                     titlefont=dict(size=10), tickfont=dict(size=9))
    fig.update_xaxes(title_text="Time (s)", row=3, col=1,
                     titlefont=dict(size=10), tickfont=dict(size=9))

    # Annotations for subplot titles
    fig.update_annotations(font_size=11, font_color="#7a8290")

    return fig


# ─── Main render function ─────────────────────────────────────────────────────

def render_live_sim(PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
                    load_mid_prices, run_quick_mc):
    """Render the animated live simulation tab."""

    hero_banner(
        "🎬",
        "Live Market Maker Simulation",
        "Watch the market maker quote in real time — bid/ask quotes skew automatically as inventory builds up.",
    )

    SYMBOLS = list(PARAMS.keys())

    col_ctrl, col_viz = st.columns([1, 3], gap="large")

    with col_ctrl:
        section_header("Parameters")

        sym = st.selectbox("Symbol", SYMBOLS, key="ls_sym")
        p = PARAMS[sym].copy()

        # Compute a safe default gamma: ensure ½γσ²(QΔ)² << revenue
        # Safe condition: γ < 0.1 / (σ² * (Q*Δ)²)
        sigma2 = p["sigma"] ** 2
        qDelta2 = (p["Q"] * p["Delta"]) ** 2
        gamma_safe_max = 0.1 / max(sigma2 * qDelta2, 1e-20)
        gamma_default = min(1e-2, max(1e-6, gamma_safe_max * 0.01))

        gamma = st.number_input(
            "γ  (risk aversion)", value=float(f"{gamma_default:.2e}"),
            min_value=1e-8, max_value=float(gamma_safe_max),
            format="%.2e", step=float(gamma_default),
            key="ls_gamma",
            help=f"Safe range for these params: γ ≤ {gamma_safe_max:.1e}",
        )
        if gamma > gamma_safe_max * 0.5:
            st.warning(f"γ is large for these params — spreads may go negative. Suggested: γ ≤ {gamma_safe_max:.1e}")

        T = st.number_input(
            "T  (horizon, seconds)", value=1800.0, min_value=60.0,
            step=300.0, key="ls_T",
        )
        N_t = st.slider(
            "Animation frames", 80, 200, 120, step=20, key="ls_Nt",
            help="Number of animation frames (physics runs at higher resolution)",
        )
        seed = st.number_input("Random seed", value=42, step=1, key="ls_seed")

        st.divider()
        section_header("How it works")
        st.caption(
            "**Red ★** = bid fill (market maker bought)  \n"
            "**Teal ★** = ask fill (market maker sold)  \n\n"
            "**Dotted lines** = actual bid/ask quotes  \n"
            "**Bar color** = inventory (teal = long, red = short)  \n\n"
            "Notice how the bid quote *rises* and ask quote *falls* when inventory "
            "is negative (short) — the model skews to attract fills that reduce risk."
        )

        st.divider()
        gen_btn = st.button("⚡ Generate & Animate", type="primary", key="ls_gen",
                            use_container_width=True)

    with col_viz:
        if gen_btn:
            with st.spinner("Solving ODE & simulating path …"):
                try:
                    path = _compute_live_path(p, gamma, T, N_t, int(seed))
                    st.session_state["ls_path"] = path
                    st.session_state["ls_plot_kw"] = PLOT_KW
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
                    return

        if "ls_path" in st.session_state:
            path    = st.session_state["ls_path"]
            plot_kw = st.session_state.get("ls_plot_kw", PLOT_KW)

            fig = _build_animation(path, plot_kw)
            show(fig, key="ls_anim_chart")

            # ── Summary strip ──────────────────────────────────────────────
            st.divider()
            c1, c2, c3, c4, c5 = st.columns(5)

            final_inv = int(path["inventory"][-1])
            final_pnl = float(path["mtm"][-1])
            n_bid     = len(path["bid_fill_t"])
            n_ask     = len(path["ask_fill_t"])
            inv_sign  = ("Long" if final_inv > 0 else ("Short" if final_inv < 0 else "Flat"))

            c1.metric("Final Inventory", f"{final_inv:+d} lots", delta=inv_sign)
            c2.metric("Final P&L (MtM)",
                      f"${final_pnl:+,.2f}",
                      delta="↑" if final_pnl >= 0 else "↓")
            c3.metric("Bid Fills", str(n_bid))
            c4.metric("Ask Fills", str(n_ask))
            c5.metric("Total Fills", str(n_bid + n_ask))

        else:
            # Placeholder before first run
            st.markdown(
                """
                <div style="
                    display:flex; flex-direction:column; align-items:center;
                    justify-content:center; height:480px;
                    background:#0c0f16; border:1px dashed #1e2330; border-radius:12px;
                    color:#3a4050; font-size:0.9rem;
                ">
                    <div style="font-size:3rem; margin-bottom:12px">🎬</div>
                    <div>Configure parameters and click <b style="color:#00d4aa">⚡ Generate & Animate</b></div>
                    <div style="margin-top:6px; font-size:0.75rem; color:#2a3040">
                        The animation runs entirely in your browser — no page reloads
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
