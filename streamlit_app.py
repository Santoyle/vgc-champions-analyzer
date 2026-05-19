from __future__ import annotations

import streamlit as st
from src.app.utils.session import init_session
from src.app.components.sidebar import render_regulation_selector

st.set_page_config(
    page_title="VGC Champions Analyzer",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1) Inicializar estado global de sesión
# Idempotente — solo inicializa si no existe
init_session()

st.markdown(
    """
<style>
    .main { padding: 1rem 2rem; }
    .metric-card {
        background: #1E2130;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .metric-card h3 {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        color: #F8FAFC;
        font-size: 2rem;
        font-weight: 700;
    }
    .section-header {
        border-left: 4px solid #7C3AED;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    .pokemon-tag {
        display: inline-block;
        background: #2D3748;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: #E2E8F0;
    }
    .empty-state {
        background: #1E2130;
        border: 1px dashed #2D3748;
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        color: #94A3B8;
    }
    .empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
    .empty-state h3 { color: #CBD5E1; margin-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        padding: 0.5rem 1rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""",
    unsafe_allow_html=True,
)

# 2) Selector de regulación en sidebar
# Visible en TODAS las páginas gracias al entrypoint-as-layout
render_regulation_selector()

# 3) Router de páginas
pages = [
    st.Page(
        "src/app/pages/01_Meta_Overview.py",
        title="Meta Overview",
        icon="🏠",
        default=True,
    ),
    st.Page(
        "src/app/pages/02_Team_Builder.py",
        title="Team Builder",
        icon="🛠️",
    ),
    st.Page(
        "src/app/pages/03_Counter_Analyzer.py",
        title="Counter Analyzer",
        icon="🎯",
    ),
    st.Page(
        "src/app/pages/04_Predictions.py",
        title="Predictions",
        icon="🧬",
    ),
    st.Page(
        "src/app/pages/05_Analytics.py",
        title="Analytics",
        icon="📊",
    ),
]

pg = st.navigation(pages)
pg.run()
