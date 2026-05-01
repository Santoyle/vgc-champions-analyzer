from __future__ import annotations

import logging

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.modules.metrics import (
    compute_ei_from_duckdb,
    compute_stc_from_duckdb,
    compute_tpi_from_duckdb,
    get_top_ei_items,
)
from src.app.utils.db import get_duckdb
from src.app.utils.session import init_session

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers cacheados
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def load_ei_data(reg_id: str, _con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Carga y cachea datos EI para la regulación."""
    return compute_ei_from_duckdb(_con, reg_id)


@st.cache_data(ttl=3600, show_spinner=False)
def load_stc_data(reg_id: str, _con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Carga y cachea datos STC para la regulación."""
    return compute_stc_from_duckdb(_con, reg_id)


@st.cache_data(ttl=3600, show_spinner=False)
def load_tpi_data(reg_id: str, _con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Carga y cachea datos TPI para la regulación."""
    return compute_tpi_from_duckdb(_con, reg_id)


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

init_session()
reg_id: str = st.session_state["selected_reg_id"]
reg_config = st.session_state["regulation_config"]
active_state: str = st.session_state.get("active_state", "active")

st.title("📐 Analytics")
st.caption(
    f"Regulación: **{reg_id}** · "
    "Métricas competitivas avanzadas"
)

if active_state != "active":
    st.info(
        f"📁 Mostrando métricas históricas de **{reg_id}**."
    )

st.divider()

con = get_duckdb()

with st.spinner(f"Cargando métricas para {reg_id}..."):
    df_ei = load_ei_data(reg_id, con)
    df_stc = load_stc_data(reg_id, con)
    df_tpi = load_tpi_data(reg_id, con)

if df_ei.empty and df_stc.empty and df_tpi.empty:
    st.warning(
        f"⚠️ Sin datos para **{reg_id}**. "
        "El pipeline de ingesta necesita datos "
        "de uso e ítems para calcular las métricas.\n\n"
        f"Ejecuta: `python -m src.app.data.pipelines."
        f"prior_ingest --regs {reg_id}`"
    )
    st.stop()

m1, m2, m3 = st.columns(3)
with m1:
    n_ei = len(df_ei) if not df_ei.empty else 0
    st.metric("Pares EI calculados", n_ei)
with m2:
    n_stc = len(df_stc) if not df_stc.empty else 0
    st.metric("Pokémon en STC ranking", n_stc)
with m3:
    n_tpi = len(df_tpi) if not df_tpi.empty else 0
    st.metric("Pokémon en TPI ranking", n_tpi)

st.divider()

tab_ei, tab_stc, tab_tpi = st.tabs([
    "⚡ EI — Efficiency Item",
    "🏃 STC — Speed Tier Control",
    "🎯 TPI — Threat Pressure Index",
])

# ── TAB 1 — EI ───────────────────────────────────────────────────────────────

with tab_ei:
    st.subheader("EI — Efficiency Item")

    with st.expander("📖 ¿Qué mide EI?", expanded=False):
        st.markdown(
            """
**EI (Efficiency Item)** mide si un ítem mejora el
rendimiento de un Pokémon más allá de lo esperado
por popularidad.

- **EI > 0**: el ítem es más eficiente que el baseline
  (ítem más popular del Pokémon)
- **EI = 0**: es el propio baseline
- **EI < 0**: menos eficiente que el baseline

La métrica usa **shrinkage beta(10,10)** para
regularizar estimaciones con pocos meses de datos —
ítems con pocas observaciones se anclan al prior
neutro (50%) para evitar conclusiones precipitadas.
        """
        )

    if df_ei.empty:
        st.info(
            f"Sin datos de ítems disponibles para {reg_id}."
        )
    else:
        ei_col1, ei_col2 = st.columns(2)
        with ei_col1:
            min_ei = st.slider(
                "EI mínimo",
                min_value=-50.0,
                max_value=50.0,
                value=0.0,
                step=0.5,
                key="ei_min_score",
            )
        with ei_col2:
            top_ei_n = st.slider(
                "Top N pares",
                min_value=5,
                max_value=50,
                value=20,
                key="ei_top_n",
            )

        df_top_ei = get_top_ei_items(
            df_ei,
            top_n=top_ei_n,
            min_ei_score=min_ei,
        )

        if df_top_ei.empty:
            st.info(
                "Sin pares con EI suficiente. "
                "Prueba reducir el EI mínimo."
            )
        else:
            fig_ei = px.bar(
                df_top_ei.head(20),
                x="ei_score",
                y="pokemon",
                color="ei_score",
                orientation="h",
                color_continuous_scale="RdYlGn",
                hover_data={
                    "item": True,
                    "raw_pct": ":.1f",
                    "shrunk_pct": ":.1f",
                    "baseline_pct": ":.1f",
                    "n_months": True,
                },
                labels={
                    "ei_score": "EI Score",
                    "pokemon": "Pokémon",
                },
                title=f"Top ítems por EI — {reg_id}",
            )
            fig_ei.update_layout(
                height=max(300, len(df_top_ei) * 25),
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_ei, use_container_width=True)

            st.caption(
                f"Mostrando {len(df_top_ei)} pares "
                f"(Pokémon, Ítem) con EI ≥ {min_ei}"
            )
            st.dataframe(
                df_top_ei[[
                    "pokemon",
                    "item",
                    "ei_score",
                    "raw_pct",
                    "shrunk_pct",
                    "baseline_pct",
                    "n_months",
                ]].rename(columns={
                    "raw_pct": "Uso crudo (%)",
                    "shrunk_pct": "Uso ajustado (%)",
                    "baseline_pct": "Baseline (%)",
                    "ei_score": "EI Score",
                    "n_months": "Meses",
                }),
                use_container_width=True,
                hide_index=True,
            )


# ── TAB 2 — STC ───────────────────────────────────────────────────────────────

with tab_stc:
    st.subheader("STC — Speed Tier Control")

    with st.expander("📖 ¿Qué mide STC?", expanded=False):
        st.markdown(
            """
**STC (Speed Tier Control)** mide qué porcentaje
del meta un Pokémon puede superar en velocidad base.

- **STC = 1.0**: más rápido que todo el meta
- **STC = 0.5**: velocidad media del meta
- **STC = 0.0**: más lento que todo el meta

Un equipo con buen STC puede actuar primero con
frecuencia — crítico para Fake Out, Tailwind, etc.
        """
        )

    if df_stc.empty:
        st.info(
            f"Sin datos de velocidad disponibles para {reg_id}."
        )
    else:
        fig_stc = px.scatter(
            df_stc,
            x="base_speed",
            y="stc_score",
            hover_name="pokemon",
            color="stc_score",
            color_continuous_scale="Blues",
            size_max=12,
            labels={
                "base_speed": "Velocidad base",
                "stc_score": "STC Score",
            },
            title=f"Speed Tier Control — {reg_id}",
        )
        fig_stc.update_layout(
            height=400,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_stc, use_container_width=True)

        stc_col1, stc_col2 = st.columns(2)
        with stc_col1:
            st.caption("🏃 **Top 10 más rápidos**")
            st.dataframe(
                df_stc.head(10)[[
                    "pokemon",
                    "base_speed",
                    "stc_score",
                    "n_faster_than",
                ]].rename(columns={
                    "base_speed": "Vel. base",
                    "stc_score": "STC",
                    "n_faster_than": "Supera a",
                }),
                use_container_width=True,
                hide_index=True,
            )
        with stc_col2:
            st.caption("🐢 **Top 10 más lentos**")
            st.dataframe(
                df_stc.tail(10)[[
                    "pokemon",
                    "base_speed",
                    "stc_score",
                    "n_faster_than",
                ]].rename(columns={
                    "base_speed": "Vel. base",
                    "stc_score": "STC",
                    "n_faster_than": "Supera a",
                }),
                use_container_width=True,
                hide_index=True,
            )

        fig_hist = px.histogram(
            df_stc,
            x="base_speed",
            nbins=20,
            title=f"Distribución de velocidades base — {reg_id}",
            labels={"base_speed": "Velocidad base"},
            color_discrete_sequence=["#636EFA"],
        )
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)


# ── TAB 3 — TPI ───────────────────────────────────────────────────────────────

with tab_tpi:
    st.subheader("TPI — Threat Pressure Index")

    with st.expander("📖 ¿Qué mide TPI?", expanded=False):
        st.markdown(
            """
**TPI (Threat Pressure Index)** mide cuánta presión
ofensiva ejerce un Pokémon — qué tan amenazante es
para el meta actual.

- **TPI alto**: co-aparece frecuentemente con los
  Pokémon más dominantes del meta
- **TPI bajo**: aparece principalmente con Pokémon
  de nicho o poco usados

La métrica combina **correlación de co-uso** con
el **peso de uso del compañero** para identificar
qué Pokémon son realmente centrales en el meta.
        """
        )

    if df_tpi.empty:
        st.info(
            f"Sin datos de co-uso disponibles para {reg_id}. TPI requiere datos "
            "de teammates de Smogon."
        )
    else:
        _tpi_n = len(df_tpi)
        _tpi_slider_max = min(30, _tpi_n)
        _tpi_slider_min = min(5, _tpi_slider_max)
        top_tpi_n = st.slider(
            "Top N Pokémon",
            min_value=max(1, _tpi_slider_min),
            max_value=_tpi_slider_max,
            value=min(15, _tpi_slider_max),
            key="tpi_top_n",
        )

        df_tpi_top = df_tpi.head(top_tpi_n)

        fig_tpi = px.bar(
            df_tpi_top,
            x="tpi_score",
            y="pokemon",
            color="tpi_score",
            orientation="h",
            color_continuous_scale="Reds",
            hover_data={
                "top_teammate": True,
                "avg_co_usage": ":.1f",
                "n_teammates": True,
            },
            labels={
                "tpi_score": "TPI Score",
                "pokemon": "Pokémon",
            },
            title=f"Threat Pressure Index — {reg_id}",
        )
        fig_tpi.update_layout(
            height=max(300, top_tpi_n * 25),
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_tpi, use_container_width=True)

        st.dataframe(
            df_tpi_top[[
                "pokemon",
                "tpi_score",
                "top_teammate",
                "avg_co_usage",
                "n_teammates",
            ]].rename(columns={
                "tpi_score": "TPI Score",
                "top_teammate": "Top compañero",
                "avg_co_usage": "Co-uso prom (%)",
                "n_teammates": "Compañeros",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig_scatter = px.scatter(
            df_tpi,
            x="avg_co_usage",
            y="tpi_score",
            hover_name="pokemon",
            color="tpi_score",
            color_continuous_scale="Reds",
            labels={
                "avg_co_usage": "Co-uso promedio (%)",
                "tpi_score": "TPI Score",
            },
            title=f"TPI vs Co-uso promedio — {reg_id}",
        )
        fig_scatter.update_layout(
            height=350,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
