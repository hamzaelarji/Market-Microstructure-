"""Optimal Market Making Dashboard.

Implements Guéant–Lehalle–Fernandez-Tapia (2017) framework.

Run:  streamlit run app.py
"""

import streamlit as st

from ui.styles import PALETTE, PLOT_KW, apply_styles, show
from ui.loaders import load_calibrated_params, load_mid_prices, run_quick_mc

from ui.tabs.param_lab     import render_param_lab
from ui.tabs.sensitivity   import render_sensitivity
from ui.tabs.frontier      import render_frontier
from ui.tabs.calibration   import render_calibration
from ui.tabs.backtest      import render_backtest
from ui.tabs.regimes       import render_regimes
from ui.tabs.hawkes        import render_hawkes
from ui.tabs.paper_trading import render_paper_trading
from ui.tabs.live_sim      import render_live_sim

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Optimal Market Making",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_styles()

# ─── Load data ────────────────────────────────────────────────────────────────
PARAMS, META, HAS_REAL_DATA = load_calibrated_params()
SYMBOLS = list(PARAMS.keys())


# ─── Header ───────────────────────────────────────────────────────────────────
h1, h2 = st.columns([4, 1])
h1.markdown("### ◈ Optimal Market Making")
if HAS_REAL_DATA and SYMBOLS:
    m = META.get(SYMBOLS[0], {})
    h2.caption(
        f"Calibrated {m.get('calibration_date','—')} · {m.get('n_days','—')}d · "
        f"R²={m.get('r_squared','—')}"
    )
else:
    h2.caption("⚠ Fallback params — run notebook 11")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🔬 Quote Lab",
    "🎬 Live Simulation",
    "⚔️ Strategy Battle",
    "📊 Efficient Frontier",
    "🌡️ Sensitivity Atlas",
    "⚡ Hawkes vs Poisson",
    "🔄 Intraday Regimes",
    "📡 Calibration",
    "📈 Paper Trading",
])

_ctx = (PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
        load_mid_prices, run_quick_mc)

with tab1:
    render_param_lab(*_ctx)

with tab2:
    render_live_sim(*_ctx)

with tab3:
    render_backtest(*_ctx)

with tab4:
    render_frontier(*_ctx)

with tab5:
    render_sensitivity(*_ctx)

with tab6:
    render_hawkes(*_ctx)

with tab7:
    render_regimes(*_ctx)

with tab8:
    render_calibration(*_ctx)

with tab9:
    render_paper_trading(*_ctx)
