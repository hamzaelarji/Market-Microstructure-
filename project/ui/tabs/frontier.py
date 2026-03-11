"""ui/tabs/frontier.py — Tab 3: Efficient Frontier."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui.styles import hero_banner, section_header


def render_frontier(PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
                    load_mid_prices, run_quick_mc):
    """Render Tab 3 — Efficient Frontier: MC E[P&L] vs Std[P&L] for multiple γ values."""
    hero_banner(
        "📊",
        "Efficient Frontier",
        "Risk-return trade-off: sweep γ to trace the market maker's efficient frontier.",
    )

    SYMBOLS = list(PARAMS.keys())

    st.caption(
        "Each point = optimal market maker at a given γ. "
        "This is the Markowitz-style efficient frontier: "
        "**optimal ≠ highest E[P&L]** — it maximises Certainty Equivalent."
    )

    c_f1, c_f2 = st.columns([1, 3])

    with c_f1:
        ef_sym = st.selectbox("Symbol", SYMBOLS, key="ef_sym")
        ef_params = PARAMS[ef_sym]
        ef_T = st.number_input("T (s)", value=3600.0, step=600.0, key="ef_T")
        ef_nsim = st.slider("MC paths per γ", 200, 3000, 600, step=100, key="ef_nsim")
        ef_lo = st.number_input("γ min", value=0.001, format="%.4f", key="ef_lo")
        ef_hi = st.number_input("γ max", value=0.10, format="%.4f", key="ef_hi")
        ef_n = st.slider("# γ points", 5, 15, 12, key="ef_n")
        run_ef = st.button("Build frontier", type="primary", key="ef_run")

    with c_f2:
        if run_ef:
            gammas_ef = np.geomspace(ef_lo, ef_hi, ef_n)
            N_t_ef = max(300, int(ef_T))

            mid_real_ef = load_mid_prices(ef_sym)
            mid_arr_ef = None
            if mid_real_ef is not None and len(mid_real_ef) > N_t_ef:
                mid_arr_ef = mid_real_ef.values[:N_t_ef + 1]

            frontier = []
            prog = st.progress(0, text="Building efficient frontier …")
            for idx, g in enumerate(gammas_ef):
                prog.progress((idx + 1) / len(gammas_ef),
                              text=f"γ = {g:.4f} — MC …")
                try:
                    mc = run_quick_mc(ef_params, g, ef_T, N_sim=ef_nsim,
                                      seed=42, mid_prices=mid_arr_ef)
                    frontier.append({"γ": g, **mc})
                except Exception as e:
                    st.warning(f"γ={g:.4f} failed: {e}")
            prog.empty()
            st.session_state["ef_results"] = pd.DataFrame(frontier)

        if "ef_results" in st.session_state:
            df = st.session_state["ef_results"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["std_pnl"], y=df["mean_pnl"],
                mode="lines+markers",
                marker=dict(
                    size=14,
                    color=df["sharpe"],
                    colorscale="RdYlGn",
                    colorbar=dict(title="Sharpe", len=0.6),
                    showscale=True,
                    line=dict(width=1, color="rgba(255,255,255,0.3)"),
                ),
                line=dict(width=2, color="rgba(255,255,255,0.15)"),
                text=[
                    f"γ={r['γ']:.4f}<br>"
                    f"Sharpe={r['sharpe']:.3f}<br>"
                    f"CE={r['ce']:+,.2f}<br>"
                    f"VaR 5%={r['var5']:+,.2f}<br>"
                    f"Fills/h={r['fill_rate_per_h']:.1f}"
                    for _, r in df.iterrows()
                ],
                hoverinfo="text",
                name="Frontier",
            ))
            for _, r in df.iterrows():
                fig.add_annotation(
                    x=r["std_pnl"], y=r["mean_pnl"],
                    text=f"γ={r['γ']:.3f}", showarrow=False, yshift=14,
                    font=dict(size=9, color="rgba(255,255,255,0.45)"),
                )
            fig.update_layout(
                **PLOT_KW, height=500,
                xaxis_title="Std[P&L]  (risk)",
                yaxis_title="E[P&L]  (return)",
            )
            show(fig)

            fig2 = make_subplots(
                rows=1, cols=3, horizontal_spacing=0.08,
                subplot_titles=["CE vs γ", "VaR 5% vs γ", "Max Drawdown vs γ"],
            )
            fig2.add_trace(go.Scatter(
                x=df["γ"], y=df["ce"], mode="lines+markers",
                marker=dict(color=PALETTE[0], size=7),
                line=dict(color=PALETTE[0]), showlegend=False,
            ), row=1, col=1)
            fig2.add_trace(go.Scatter(
                x=df["γ"], y=df["var5"], mode="lines+markers",
                marker=dict(color=PALETTE[1], size=7),
                line=dict(color=PALETTE[1]), showlegend=False,
            ), row=1, col=2)
            fig2.add_trace(go.Scatter(
                x=df["γ"], y=df["max_dd"], mode="lines+markers",
                marker=dict(color=PALETTE[2], size=7),
                line=dict(color=PALETTE[2]), showlegend=False,
            ), row=1, col=3)
            for c in (1, 2, 3):
                fig2.update_xaxes(title_text="γ", type="log", row=1, col=c)
            fig2.update_layout(**PLOT_KW, height=300)
            show(fig2)

            st.dataframe(
                df[["γ", "mean_pnl", "std_pnl", "sharpe", "ce",
                    "var5", "max_dd", "fill_rate_per_h", "mean_abs_inv"]].style.format({
                    "γ": "{:.4f}", "mean_pnl": "{:+,.2f}",
                    "std_pnl": "{:,.2f}", "sharpe": "{:.3f}",
                    "ce": "{:+,.2f}", "var5": "{:+,.2f}",
                    "max_dd": "{:,.2f}", "fill_rate_per_h": "{:.1f}",
                    "mean_abs_inv": "{:.2f}",
                }),
                use_container_width=True,
            )

            st.info(
                "**Key insight:** The point with highest E[P&L] is NOT the optimal — "
                "the model maximises **Certainty Equivalent**, which penalises variance. "
                "A naive strategy may beat on raw P&L but lose on risk-adjusted metrics."
            )
