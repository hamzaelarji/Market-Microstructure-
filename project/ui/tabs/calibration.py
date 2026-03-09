"""ui/tabs/calibration.py — Tab 4: Calibration viewer and re-calibration."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_making.data.calibrate import calibrate, estimate_intensity


def render_calibration(PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
                       load_mid_prices, run_quick_mc):
    """Render Tab 4 — Calibration: show calibrated params and re-calibrate from CSV."""
    SYMBOLS = list(PARAMS.keys())

    if HAS_REAL_DATA:
        rows = []
        for sym in SYMBOLS:
            p = PARAMS[sym]
            m = META.get(sym, {})
            rows.append({
                "Symbol": sym,
                "σ": f"{p['sigma']:.4f}",
                "A": f"{p['A']:.4f}",
                "k": f"{p['k']:.4f}",
                "Δ": f"{p['Delta']:.2f}",
                "R²": f"{m.get('r_squared', '—')}",
                "Trades": f"{m.get('n_trades', 0):,}",
                "Period": m.get("period", "—"),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Symbol"),
                     use_container_width=True)

        rho = META.get("rho")
        if rho:
            st.caption(f"ρ(BTC, ETH) = {rho:.4f}")

    st.divider()
    st.markdown("##### Re-calibrate from CSV")
    uploaded = st.file_uploader(
        "Upload trades CSV (timestamp, price, quantity)", type=["csv"])
    if uploaded:
        trades = pd.read_csv(uploaded)
        if not all(c in trades.columns for c in ["timestamp", "price", "quantity"]):
            st.error("CSV needs columns: timestamp, price, quantity")
        else:
            trades["timestamp"] = pd.to_datetime(trades["timestamp"])
            freq = st.selectbox("Sampling freq", ["1s", "5s", "10s"])
            if st.button("Calibrate", type="primary"):
                with st.spinner("Fitting..."):
                    cal = calibrate(trades, freq=freq)

                c1, c2, c3 = st.columns(3)
                c1.metric("σ", f"{cal.sigma:.4f}")
                c2.metric("A", f"{cal.A:.4f}")
                c3.metric("k", f"{cal.k:.4f}")

                st.session_state["calibrated_params"] = cal.to_dict()
                st.session_state["calibrated_params"]["Q"] = 4

                A_fit, k_fit, r_sq, bins, lam_obs = estimate_intensity(trades, freq)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=bins, y=lam_obs, name="Observed",
                                     opacity=0.5, marker_color=PALETTE[0]))
                xf = np.linspace(0, bins[-1], 200)
                fig.add_trace(go.Scatter(x=xf, y=A_fit * np.exp(-k_fit * xf),
                    mode="lines", name=f"Fit (R²={r_sq:.3f})",
                    line=dict(color=PALETTE[1], width=2)))
                fig.update_layout(**PLOT_KW, height=360,
                    xaxis_title="δ ($)", yaxis_title="Λ(δ)")
                show(fig)
