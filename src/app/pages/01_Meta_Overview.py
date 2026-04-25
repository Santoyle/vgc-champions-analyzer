from __future__ import annotations

import logging

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.core.regulation_active import get_active_regulation
from src.app.data.sql.views import (
    create_teammates_by_pkm,
    create_usage_by_reg,
    register_raw_view,
)
from src.app.utils.db import get_duckdb
from src.app.utils.session import init_session

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers cacheados
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def load_usage_data(
    reg_id: str,
    _con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Carga y cachea los datos de uso para la regulación.

    El parámetro _con tiene prefijo _ para que st.cache_data lo ignore
    al calcular la clave de cache (las conexiones DuckDB no son hasheables).

    Returns:
        DataFrame con columnas: regulation_id, pokemon, avg_usage_pct,
        total_raw_count, n_months, max_usage_pct, min_usage_pct.
        DataFrame vacío si no hay datos.
    """
    try:
        register_raw_view(_con)
        df = create_usage_by_reg(_con, reg_id)
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning("Error cargando datos de uso para %s: %s", reg_id, exc)
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_teammates_data(
    reg_id: str,
    _con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Carga y cachea datos de compañeros."""
    try:
        register_raw_view(_con)
        df = create_teammates_by_pkm(_con, reg_id)
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning("Error cargando teammates para %s: %s", reg_id, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Inicialización de sesión
# ---------------------------------------------------------------------------

init_session()
reg_id: str = st.session_state["selected_reg_id"]
reg_config = st.session_state["regulation_config"]
active_state: str = st.session_state.get("active_state", "active")

# ---------------------------------------------------------------------------
# Título y contexto
# ---------------------------------------------------------------------------

st.title("🏠 Meta Overview")
st.caption(
    f"Regulación: **{reg_id}** · "
    f"{reg_config.date_start} → {reg_config.date_end}"
)

if active_state != "active":
    st.info(
        f"📁 Estás viendo datos históricos de **{reg_id}**. "
        f"Esta regulación no está activa actualmente."
    )

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

con = get_duckdb()
with st.spinner(f"Cargando datos de {reg_id}..."):
    df_usage = load_usage_data(reg_id, con)
    df_teammates = load_teammates_data(reg_id, con)

if df_usage.empty:
    st.warning(
        f"⚠️ Sin datos de uso para **{reg_id}**. "
        f"El pipeline de ingesta puede no haber procesado esta regulación todavía.\n\n"
        f"Ejecuta: `python -m src.app.data.pipelines.prior_ingest --regs {reg_id}` "
        f"o espera al próximo run del daily-ingest."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Métricas summary
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Pokémon en meta", len(df_usage))
with col2:
    top1 = df_usage.iloc[0]["pokemon"] if len(df_usage) > 0 else "—"
    st.metric("Más usado", top1)
with col3:
    top1_pct = (
        f"{df_usage.iloc[0]['avg_usage_pct']:.1f}%"
        if len(df_usage) > 0
        else "—"
    )
    st.metric("Uso top-1", top1_pct)
with col4:
    n_months = (
        df_usage["n_months"].max() if "n_months" in df_usage.columns else 0
    )
    st.metric("Meses de datos", int(n_months))

st.divider()

# ---------------------------------------------------------------------------
# 4 Tabs
# ---------------------------------------------------------------------------

tab_quad, tab_tree, tab_trend, tab_clusters = st.tabs(
    [
        "📊 Cuadrante",
        "🗺️ Treemap",
        "📈 Tendencia",
        "🔵 Clusters",
    ]
)

# ── Tab 1: Cuadrante ──────────────────────────────────────────────────────

with tab_quad:
    st.subheader("Cuadrante de uso")
    st.caption(
        "Eje X: uso promedio. Eje Y: máximo de uso (proxy de relevancia). "
        "Tamaño: total de apariciones. Los datos vienen de Smogon ladder — "
        "no representan winrate real."
    )

    top_n = st.slider(
        "Mostrar top N Pokémon",
        min_value=10,
        max_value=min(100, len(df_usage)),
        value=min(30, len(df_usage)),
        step=5,
        key="quad_top_n",
    )
    df_plot = df_usage.head(top_n).copy()

    median_usage = df_plot["avg_usage_pct"].median()
    median_max = df_plot["max_usage_pct"].median()

    def get_zone(row: pd.Series) -> str:  # type: ignore[type-arg]
        high_usage = row["avg_usage_pct"] >= median_usage
        high_max = row["max_usage_pct"] >= median_max
        if high_usage and high_max:
            return "⭐ Staples"
        elif not high_usage and high_max:
            return "🔍 Sleepers"
        elif high_usage and not high_max:
            return "⚠️ Populares"
        else:
            return "🔵 Nicho"

    df_plot["zona"] = df_plot.apply(get_zone, axis=1)

    # Limpiar NaN en columnas numéricas antes del scatter
    df_plot["avg_usage_pct"] = df_plot["avg_usage_pct"].fillna(0)
    df_plot["max_usage_pct"] = df_plot["max_usage_pct"].fillna(0)
    df_plot["total_raw_count"] = (
        df_plot["total_raw_count"].fillna(1).clip(lower=1)
    )

    fig_quad = px.scatter(
        df_plot,
        x="avg_usage_pct",
        y="max_usage_pct",
        size="total_raw_count",
        color="zona",
        hover_name="pokemon",
        hover_data={
            "avg_usage_pct": ":.2f",
            "max_usage_pct": ":.2f",
            "n_months": True,
            "zona": False,
        },
        color_discrete_map={
            "⭐ Staples": "#00CC96",
            "🔍 Sleepers": "#636EFA",
            "⚠️ Populares": "#FFA15A",
            "🔵 Nicho": "#AB63FA",
        },
        title=f"Cuadrante de uso — {reg_id}",
        labels={
            "avg_usage_pct": "Uso promedio (%)",
            "max_usage_pct": "Pico máximo (%)",
        },
        size_max=40,
    )
    fig_quad.add_hline(
        y=median_max, line_dash="dash", line_color="gray", opacity=0.5
    )
    fig_quad.add_vline(
        x=median_usage, line_dash="dash", line_color="gray", opacity=0.5
    )
    fig_quad.update_layout(height=500, legend_title="Zona")
    st.plotly_chart(fig_quad, use_container_width=True)

    zone_summary = (
        df_plot.groupby("zona")
        .agg(
            n_pokemon=("pokemon", "count"),
            avg_uso=("avg_usage_pct", "mean"),
        )
        .round(2)
        .reset_index()
    )
    st.dataframe(zone_summary, use_container_width=True, hide_index=True)

# ── Tab 2: Treemap ────────────────────────────────────────────────────────

with tab_tree:
    st.subheader("Treemap de uso")
    st.caption(
        "El tamaño de cada celda es proporcional al uso promedio en el período."
    )

    df_tree = df_usage[df_usage["avg_usage_pct"] > 0].copy()
    df_tree = df_tree.nlargest(50, "avg_usage_pct")

    if df_tree.empty:
        st.info("Sin datos para el treemap.")
    else:
        fig_tree = go.Figure(
            go.Treemap(
                labels=df_tree["pokemon"].tolist(),
                parents=[""] * len(df_tree),
                values=df_tree["avg_usage_pct"].tolist(),
                textinfo="label+value",
                hovertemplate=(
                    "<b>%{label}</b><br>Uso: %{value:.1f}%<extra></extra>"
                ),
            )
        )
        fig_tree.update_layout(
            title=f"Treemap de uso — Reg {reg_id}",
            height=600,
            margin={"t": 50, "l": 10, "r": 10, "b": 10},
        )
        st.plotly_chart(fig_tree, use_container_width=True)

# ── Tab 3: Tendencia ──────────────────────────────────────────────────────

with tab_trend:
    st.subheader("Tendencia de uso por mes")

    if "n_months" not in df_usage.columns or df_usage["n_months"].max() <= 1:
        st.info(
            "📊 Solo hay datos de 1 mes disponibles. "
            "La tendencia temporal requiere al menos 2 meses de datos."
        )
    else:
        st.caption(
            "Evolución del uso de los Pokémon más relevantes a lo largo del período."
        )
        try:
            register_raw_view(con)
            df_monthly = con.execute(f"""
                SELECT
                    year_month,
                    pokemon,
                    AVG(usage_pct) AS usage_pct
                FROM raw_usage
                WHERE regulation_id = '{reg_id}'
                GROUP BY year_month, pokemon
                ORDER BY year_month, usage_pct DESC
            """).df()

            if df_monthly.empty:
                st.info("Sin datos mensuales disponibles.")
            else:
                top10 = (
                    df_monthly.groupby("pokemon")["usage_pct"]
                    .mean()
                    .nlargest(10)
                    .index.tolist()
                )
                df_trend = df_monthly[df_monthly["pokemon"].isin(top10)]

                fig_trend = px.line(
                    df_trend,
                    x="year_month",
                    y="usage_pct",
                    color="pokemon",
                    markers=True,
                    title=f"Evolución de uso top-10 — {reg_id}",
                    labels={
                        "year_month": "Mes",
                        "usage_pct": "Uso (%)",
                    },
                )
                fig_trend.update_layout(height=450)
                st.plotly_chart(fig_trend, use_container_width=True)

        except Exception as exc:  # noqa: BLE001
            log.warning("Error cargando tendencia para %s: %s", reg_id, exc)
            st.warning("No se pudo cargar la tendencia temporal.")

# ── Tab 4: Clusters ───────────────────────────────────────────────────────

with tab_clusters:
    st.subheader("Clusters de uso")
    st.caption("Agrupación de Pokémon por perfil de uso. Requiere scikit-learn.")

    if len(df_usage) < 10:
        st.info(
            "Se necesitan al menos 10 Pokémon con datos para calcular clusters."
        )
    else:
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            feature_cols = ["avg_usage_pct", "max_usage_pct", "min_usage_pct"]
            available_cols = [c for c in feature_cols if c in df_usage.columns]

            if len(available_cols) < 2:
                st.info("Columnas insuficientes para clustering.")
            else:
                df_cluster = df_usage.head(50).copy()
                X = df_cluster[available_cols].fillna(0).values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                n_clusters = st.slider(
                    "Número de clusters",
                    min_value=2,
                    max_value=min(8, len(df_cluster) - 1),
                    value=4,
                    key="cluster_n",
                )

                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10,
                )
                df_cluster["cluster"] = kmeans.fit_predict(X_scaled)
                df_cluster["cluster_label"] = (
                    "Cluster " + df_cluster["cluster"].astype(str)
                )

                fig_cluster = px.scatter(
                    df_cluster,
                    x="avg_usage_pct",
                    y="max_usage_pct",
                    color="cluster_label",
                    hover_name="pokemon",
                    title=f"KMeans clusters ({n_clusters}) — {reg_id}",
                    labels={
                        "avg_usage_pct": "Uso promedio (%)",
                        "max_usage_pct": "Pico uso (%)",
                    },
                )
                fig_cluster.update_layout(height=450)
                st.plotly_chart(fig_cluster, use_container_width=True)

                cluster_summary = (
                    df_cluster.groupby("cluster_label")
                    .agg(
                        n_pokemon=("pokemon", "count"),
                        avg_uso=("avg_usage_pct", "mean"),
                        ejemplos=(
                            "pokemon",
                            lambda x: ", ".join(x.head(3).tolist()),
                        ),
                    )
                    .round(2)
                    .reset_index()
                )
                st.dataframe(
                    cluster_summary,
                    use_container_width=True,
                    hide_index=True,
                )

        except ImportError:
            st.warning(
                "scikit-learn no disponible. "
                "Instala con: pip install scikit-learn"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Error en clustering para %s: %s", reg_id, exc)
            st.warning(f"Error al calcular clusters. Detalle: {exc}")
