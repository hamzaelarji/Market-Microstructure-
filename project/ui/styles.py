"""ui/styles.py — Design system: colors, CSS, Plotly template, helper widgets."""

import streamlit as st

# ─── Color Palette ────────────────────────────────────────────────────────────
TEAL   = "#00a882"
RED    = "#e05050"
YELLOW = "#d4a017"
PURPLE = "#8b5cf6"
BLUE   = "#2d7dd2"
ORANGE = "#d97706"
GRAY   = "#6b7280"

PALETTE = [TEAL, RED, YELLOW, PURPLE, BLUE, ORANGE, GRAY]

# ─── Plotly config ────────────────────────────────────────────────────────────
PLOT_KW = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11, color="#444444"),
    margin=dict(l=50, r=20, t=44, b=36),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.07)", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.07)", zeroline=False),
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
_CSS = """
<style>

/* ── Base & layout ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 3rem; padding-bottom: 1rem; max-width: 1400px; }
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Tabs (top-level) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #dde3ec;
    background: transparent;
    padding: 0 0.2rem;
}
.stTabs [data-baseweb="tab"] {
    height: 36px;
    border-radius: 6px 6px 0 0;
    padding: 0 14px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #8a96a8;
    background: transparent;
    border: none;
    transition: color 0.15s, background 0.15s;
}
.stTabs [aria-selected="true"] {
    color: #00a882 !important;
    background: rgba(0,168,130,0.07) !important;
    border-bottom: 2px solid #00a882 !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #333; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #f7f9fc;
    border: 1px solid #dde3ec;
    border-radius: 10px;
    padding: 14px 18px;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: #b0bcd0; }
[data-testid="stMetricLabel"] { font-size: 0.72rem; opacity: 0.55; letter-spacing: 0.5px; }
[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }
[data-testid="stMetricDelta"] { font-size: 0.75rem; }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4aa, #00a882);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.15s, transform 0.1s;
}
.stButton > button[kind="primary"]:hover { opacity: 0.9; transform: translateY(-1px); }
.stButton > button[kind="primary"]:active { transform: translateY(0); }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 8px; border-left-width: 3px; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #eef6f4 0%, #f0f5ff 100%);
    border: 1px solid #cce4de;
    border-left: 3px solid #00a882;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.hero-icon { font-size: 1.8rem; }
.hero-title {
    font-size: 1.15rem; font-weight: 700;
    background: linear-gradient(90deg, #00a882, #2d7dd2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-subtitle { font-size: 0.78rem; color: #7a8a9a; margin: 2px 0 0 0; }

/* ── Section header ── */
.section-header {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: #8a96a8;
    border-bottom: 1px solid #dde3ec;
    padding-bottom: 6px; margin: 18px 0 12px 0;
}

/* ── Dividers ── */
hr { border-color: #dde3ec; opacity: 1; }

/* ── Tables / dataframes ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Progress bars ── */
.stProgress > div > div { background: linear-gradient(90deg, #00d4aa, #2d7dd2); }

/* ── Pnl colors ── */
.pnl-positive { color: #00a882; }
.pnl-negative { color: #e05050; }

</style>
"""


def apply_styles():
    """Inject CSS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def show(fig, **kw):
    """Render a Plotly figure with full width."""
    st.plotly_chart(fig, use_container_width=True, theme=None, **kw)


def hero_banner(icon: str, title: str, subtitle: str = ""):
    """Render a top-of-tab hero banner."""
    st.markdown(
        f"""<div class="hero-banner">
          <div class="hero-icon">{icon}</div>
          <div>
            <p class="hero-title">{title}</p>
            <p class="hero-subtitle">{subtitle}</p>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


def section_header(text: str):
    """Render a small section header with bottom border."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def pnl_number(value: float, prefix: str = "$") -> str:
    """Format a P&L value with HTML color class."""
    cls = "pnl-positive" if value >= 0 else "pnl-negative"
    sign = "+" if value >= 0 else ""
    return f'<span class="{cls}">{prefix}{sign}{value:,.2f}</span>'


def strategy_color(name: str) -> str:
    """Map strategy name to a consistent color."""
    _MAP = {
        "optimal":       TEAL,
        "naive":         RED,
        "closed_form":   PURPLE,
        "inv_skew_only": ORANGE,
    }
    return _MAP.get(name, GRAY)
