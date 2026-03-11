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
    initial_sidebar_state="expanded",
)
apply_styles()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">◈ Optimal MM</div>',
        unsafe_allow_html=True,
    )

    _ctx_data = load_calibrated_params()
    PARAMS, META, HAS_REAL_DATA = _ctx_data
    SYMBOLS = list(PARAMS.keys())

    if HAS_REAL_DATA and SYMBOLS:
        m0 = META.get(SYMBOLS[0], {})
        st.caption(
            f"✓ Calibrated · {m0.get('calibration_date','—')}  \n"
            f"R² = {m0.get('r_squared','—')} · {m0.get('n_days','—')} days"
        )
    else:
        st.warning("Fallback params — run notebook 11", icon="⚠")

    st.divider()

    NAV = st.radio(
        "Navigation",
        [
            "🔬  Quote Lab",
            "🎬  Live Simulation",
            "⚔️  Strategy Battle",
            "📊  Efficient Frontier",
            "🌡️  Sensitivity Atlas",
            "⚡  Hawkes vs Poisson",
            "🔄  Intraday Regimes",
            "📡  Calibration",
            "📈  Paper Trading",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-cite">'
        'Guéant, Lehalle & Fernandez-Tapia (2017)<br>'
        '<em>Optimal Market Making</em><br>'
        'Applied Mathematical Finance 24(2)'
        '</div>',
        unsafe_allow_html=True,
    )

# ─── Context bundle ───────────────────────────────────────────────────────────
_ctx = (PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
        load_mid_prices, run_quick_mc)

# ─── Router ───────────────────────────────────────────────────────────────────
if "Quote Lab" in NAV:
    render_param_lab(*_ctx)
elif "Live Simulation" in NAV:
    render_live_sim(*_ctx)
elif "Strategy Battle" in NAV:
    render_backtest(*_ctx)
elif "Efficient Frontier" in NAV:
    render_frontier(*_ctx)
elif "Sensitivity Atlas" in NAV:
    render_sensitivity(*_ctx)
elif "Hawkes vs Poisson" in NAV:
    render_hawkes(*_ctx)
elif "Intraday Regimes" in NAV:
    render_regimes(*_ctx)
elif "Calibration" in NAV:
    render_calibration(*_ctx)
elif "Paper Trading" in NAV:
    render_paper_trading(*_ctx)
