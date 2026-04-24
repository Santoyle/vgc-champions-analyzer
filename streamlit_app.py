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
