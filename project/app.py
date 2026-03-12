"""Optimal Market Making — Guéant (2017) Dashboard.

2 pages:
  1. Paper Replication  — interactive reproduction of all paper figures
  2. Strategy Lab        — backtest & compare market making strategies

Run:  streamlit run app.py
"""

import streamlit as st
from ui.styles import apply_styles

st.set_page_config(
    page_title="Optimal Market Making",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_styles()

# ── Hide sidebar completely ───────────────────────────────────
st.markdown("""<style>[data-testid="stSidebar"]{display:none}</style>""",
            unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("### ◈ Optimal Market Making — Guéant (2017)")

# ── Horizontal tabs ───────────────────────────────────────────
tab1, tab2 = st.tabs(["📘 Paper Replication", "⚔️ Strategy Lab"])

with tab1:
    from ui.tabs.paper_replication import render as render_paper
    render_paper()

with tab2:
    from ui.tabs.strategy_lab import render as render_lab
    render_lab()
