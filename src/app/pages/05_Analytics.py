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


def _parse_paste_to_slugs(paste: str) -> list[str]:
    slugs = []
    for line in paste.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        # Primera palabra = nombre del Pokémon
        name = line.split("@")[0].split("(")[0]
        name = name.strip()
        if name:
            slugs.append(
                name.lower().replace(" ", "-")
            )
        if len(slugs) >= 6:
            break
    return slugs


_NATURES_25: tuple[str, ...] = (
    "Adamant",
    "Bashful",
    "Bold",
    "Brave",
    "Calm",
    "Careful",
    "Docile",
    "Gentle",
    "Hardy",
    "Hasty",
    "Impish",
    "Jolly",
    "Lax",
    "Lonely",
    "Mild",
    "Modest",
    "Naive",
    "Naughty",
    "Quiet",
    "Quirky",
    "Rash",
    "Relaxed",
    "Sassy",
    "Serious",
    "Timid",
)

_MOVE_TYPES_18: tuple[str, ...] = (
    "Normal",
    "Fire",
    "Water",
    "Electric",
    "Grass",
    "Ice",
    "Fighting",
    "Poison",
    "Ground",
    "Flying",
    "Psychic",
    "Bug",
    "Rock",
    "Ghost",
    "Dragon",
    "Dark",
    "Steel",
    "Fairy",
)


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

(
    tab_ei,
    tab_stc,
    tab_tpi,
    tab_mlwr,
    tab_spdo,
    tab_lre,
    tab_meti_meit,
    tab_tai,
    tab_shapley,
) = st.tabs([
    "⚡ EI — Efficiency Item",
    "🏃 STC — Speed Tier Control",
    "🎯 TPI — Threat Pressure Index",
    "⚡ MLWR",
    "📐 SPDO",
    "🎯 LRE",
    "⏱️ METI/MEIT",
    "🎲 TAI",
    "🔢 Shapley SV",
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


# ── Métricas V2 — Bloque 15 ─────────────

with tab_mlwr:
    # ── Métricas V2 — Bloque 15 ─────────────
    st.subheader("⚡ MLWR — Move-Level Win Rate")
    try:
        from src.app.metrics.mlwr import (
            compute_mlwr_level1,
        )
        import plotly.graph_objects as go

        reg = st.session_state["regulation_config"]
        reg_id_v2 = reg.regulation_id
        con_v2 = get_duckdb()

        try:
            top_pkm = con_v2.execute(
                """
                SELECT pokemon
                FROM read_parquet(
                  'data/raw/reg=*/source=pikalytics/*.parquet',
                  hive_partitioning=true
                )
                WHERE regulation_id = ?
                GROUP BY pokemon
                ORDER BY AVG(usage_pct) DESC
                LIMIT 20
                """,
                [reg_id_v2],
            ).df()["pokemon"].tolist()
        except Exception:
            top_pkm = [
                "garchomp",
                "incineroar",
                "primarina",
            ]

        pkm_mlwr = st.selectbox(
            "Pokémon (top meta)",
            options=top_pkm or ["garchomp"],
            key="v2_mlwr_pokemon",
        )

        if st.button("Calcular MLWR", key="v2_mlwr_calc"):
            df_mlwr = compute_mlwr_level1(
                reg_id_v2,
                con_v2,
                min_n=3,
            )
            df_f = df_mlwr[
                df_mlwr["pokemon_slug"] == pkm_mlwr
            ].copy()
            if df_f.empty:
                st.info(
                    "Sin datos suficientes "
                    "(min_n=3 usos por move)"
                )
            else:
                df_f = df_f.sort_values(
                    "mlwr", ascending=True
                )
                colors = [
                    "#2ecc71" if v > 0 else "#e74c3c"
                    for v in df_f["mlwr"]
                ]
                err_hi = (
                    df_f["wilson_high"] - df_f["mlwr"]
                ).tolist()
                err_lo = (
                    df_f["mlwr"] - df_f["wilson_low"]
                ).tolist()
                fig_m = go.Figure(
                    data=[
                        go.Bar(
                            y=df_f["move_slug"],
                            x=df_f["mlwr"],
                            orientation="h",
                            error_x=dict(
                                type="data",
                                symmetric=False,
                                array=err_hi,
                                arrayminus=err_lo,
                            ),
                            marker_color=colors,
                        )
                    ]
                )
                fig_m.update_layout(
                    xaxis_title="mlwr "
                    "(Wilson low / high)",
                    yaxis_title="move_slug",
                    title=(
                        f"MLWR — {pkm_mlwr} "
                        f"({reg_id_v2})"
                    ),
                    height=max(320, len(df_f) * 28),
                )
                st.plotly_chart(
                    fig_m, use_container_width=True
                )
    except Exception as exc:
        st.error(f"MLWR: {exc}")


with tab_spdo:
    # ── Métricas V2 — Bloque 15 ─────────────
    st.subheader("📐 SPDO")
    reg = st.session_state["regulation_config"]
    reg_id_v2 = reg.regulation_id

    with st.expander("Speed Tier Table", expanded=False):
        try:
            spd_pkm = st.text_input(
                "Pokémon (slug)",
                value="garchomp",
                key="v2_spd_tier_pkm",
            )
            spd_nat = st.selectbox(
                "Naturaleza",
                options=_NATURES_25,
                index=_NATURES_25.index("Jolly"),
                key="v2_spd_tier_nat",
            )
            if st.button(
                "Generar tabla",
                key="v2_spd_tier_btn",
            ):
                from src.app.metrics.spdo import (
                    build_speed_tier_table,
                )

                df_spd = build_speed_tier_table(
                    pokemon_slug=spd_pkm.strip(),
                    reg_id=reg_id_v2,
                    top_n=20,
                    my_nature=spd_nat,
                )
                cols_show = [
                    "target_pokemon",
                    "target_speed",
                    "spe_sp_to_outspeed",
                    "my_speed_at_that_sp",
                    "spe_sp_under_tr",
                ]
                avail = [
                    c for c in cols_show
                    if c in df_spd.columns
                ]
                if df_spd.empty or not avail:
                    st.info(
                        "Sin datos de speed tiers "
                        "para ese Pokémon."
                    )
                else:
                    st.dataframe(
                        df_spd[avail],
                        use_container_width=True,
                        hide_index=True,
                    )
        except Exception as exc:
            st.error(f"Speed Tier Table: {exc}")

    with st.expander("Defensive Optimizer", expanded=False):
        try:
            from src.app.core.schema import SPSpread

            df_pkm_i = st.text_input(
                "Pokémon defensor (slug)",
                value="incineroar",
                key="v2_spdo_def_slug",
            )
            atk_slug_i = st.text_input(
                "Atacante (slug)",
                value="garchomp",
                key="v2_spdo_atk_slug",
            )
            pwr_i = st.number_input(
                "Potencia del movimiento",
                min_value=0,
                max_value=250,
                value=100,
                key="v2_spdo_pwr",
            )
            type_i = st.selectbox(
                "Tipo del movimiento",
                options=_MOVE_TYPES_18,
                index=_MOVE_TYPES_18.index("Ground"),
                key="v2_spdo_movetype",
            )
            cat_i = st.selectbox(
                "Categoría",
                options=["Physical", "Special"],
                key="v2_spdo_cat",
            )
            surv_i = st.slider(
                "Supervivencia objetivo %",
                min_value=50,
                max_value=100,
                value=95,
                key="v2_spdo_surv",
            )
            fix_off_i = st.slider(
                "fixed_offensive_sp",
                min_value=0,
                max_value=32,
                value=0,
                key="v2_spdo_fixoff",
            )
            if st.button(
                "Optimizar",
                key="v2_spdo_go",
            ):
                from src.app.metrics.spdo import (
                    optimize_defensive_sp,
                )

                atk_spread = SPSpread(
                    hp=2,
                    atk=32,
                    **{"def": 0},
                    spa=0,
                    spd=0,
                    spe=32,
                )
                move_d = {
                    "power": int(pwr_i),
                    "type": type_i,
                    "category": cat_i,
                    "name": "Move",
                }
                opt_r = optimize_defensive_sp(
                    reg_id=reg_id_v2,
                    pokemon_slug=df_pkm_i.strip(),
                    attacker_slug=atk_slug_i.strip(),
                    attacker_move=move_d,
                    attacker_spread=atk_spread,
                    attacker_nature="Jolly",
                    survival_target_pct=surv_i / 100.0,
                    fixed_offensive_sp=int(fix_off_i),
                )
                if opt_r is None:
                    st.warning(
                        "No se pudo optimizar "
                        "(Pokémon o datos)."
                    )
                else:
                    sp = opt_r.spread
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("HP SP", sp.hp)
                    with c2:
                        st.metric("Def SP", sp.def_)
                    with c3:
                        st.metric(
                            "Costo total SP",
                            sp.total(),
                        )
                    st.metric(
                        "Prob. supervivencia",
                        f"{opt_r.survival_prob:.3f}",
                    )
                    st.caption(
                        f"Target {opt_r.target_prob:.2f} · "
                        f"found={opt_r.found}"
                    )
        except Exception as exc:
            st.error(f"Defensive Optimizer: {exc}")


with tab_lre:
    # ── Métricas V2 — Bloque 15 ─────────────
    st.subheader("🎯 LRE — Lead Recommendation")
    try:
        from pathlib import Path

        from src.app.metrics.lre import (
            lead_recommendation,
            populate_lead_pair_stats,
        )

        reg = st.session_state["regulation_config"]
        reg_id_v2 = reg.regulation_id
        con_v2 = get_duckdb()

        paste_my = st.text_area(
            "Equipo propio (Showdown)",
            height=120,
            key="v2_lre_my",
        )
        paste_opp = st.text_area(
            "Equipo rival (Showdown)",
            height=120,
            key="v2_lre_opp",
        )

        if st.button(
            "Recomendar Leads",
            key="v2_lre_rec",
        ):
            my_s = _parse_paste_to_slugs(paste_my)
            op_s = _parse_paste_to_slugs(paste_opp)
            if len(my_s) < 2 or len(op_s) < 2:
                st.warning(
                    "Pega al menos 2 Pokémon "
                    "por equipo."
                )
            else:
                df_lr = lead_recommendation(
                    reg_id_v2,
                    tuple(my_s[:6]),
                    tuple(op_s[:6]),
                    con_v2,
                )
                try:
                    tot_n = int(
                        con_v2.execute(
                            """
                            SELECT COALESCE(
                              SUM(n_matches), 0
                            )
                            FROM lead_pair_stats
                            WHERE regulation_id = ?
                            """,
                            [reg_id_v2],
                        ).fetchone()[0]
                    )
                except Exception:
                    tot_n = 0
                if tot_n < 50:
                    st.warning(
                        "⚠️ Datos insuficientes — "
                        "usando type matchup "
                        "heurístico"
                    )
                st.dataframe(
                    df_lr.head(5),
                    use_container_width=True,
                    hide_index=True,
                )

        if st.button(
            "Actualizar datos",
            key="v2_lre_pop",
        ):
            _root = Path(
                __file__
            ).resolve().parent.parent.parent.parent
            _dbp = _root / "data" / "vgc.duckdb"
            populate_lead_pair_stats(
                reg_id_v2,
                str(_dbp),
            )
            st.success(
                "lead_pair_stats actualizada."
            )
    except Exception as exc:
        st.error(f"LRE: {exc}")


with tab_meti_meit:
    # ── Métricas V2 — Bloque 15 ─────────────
    st.subheader("⏱️ METI / MEIT")
    try:
        from src.app.metrics.advanced_metrics import (
            compute_meit,
        )
        from src.app.metrics.mlwr import compute_meti

        reg = st.session_state["regulation_config"]
        reg_id_v2 = reg.regulation_id
        con_v2 = get_duckdb()

        mega_on = (
            reg.mechanics.mega_enabled
            if hasattr(reg, "mechanics")
            else True
        )

        if not mega_on:
            st.info(
                "Mega Evoluciones no disponibles "
                "en esta regulación"
            )
        else:
            try:
                top_pkm_m = con_v2.execute(
                    """
                    SELECT pokemon
                    FROM read_parquet(
                      'data/raw/reg=*/source=pikalytics/*.parquet',
                      hive_partitioning=true
                    )
                    WHERE regulation_id = ?
                    GROUP BY pokemon
                    ORDER BY AVG(usage_pct) DESC
                    LIMIT 20
                    """,
                    [reg_id_v2],
                ).df()["pokemon"].tolist()
            except Exception:
                top_pkm_m = [
                    "garchomp",
                    "incineroar",
                    "primarina",
                ]

            pkm_m = st.selectbox(
                "Pokémon",
                options=top_pkm_m or ["garchomp"],
                key="v2_meti_pkm",
            )

            if st.button(
                "Analizar Mega Timing",
                key="v2_meti_btn",
            ):
                df_m = compute_meti(
                    reg_id_v2,
                    con_v2,
                    mega_enabled=mega_on,
                )
                df_p = df_m[
                    df_m["pokemon_slug"] == pkm_m
                ]
                if df_p.empty:
                    st.info("Sin filas METI para "
                            "ese Pokémon.")
                else:
                    base_ref = float(
                        df_p["baseline_wr"].iloc[0]
                    )
                    fig_b = px.bar(
                        df_p,
                        x="turn_bucket",
                        y="win_rate",
                        title=(
                            f"WR por bucket — {pkm_m}"
                        ),
                        labels={
                            "turn_bucket": "Bucket",
                            "win_rate": "Win rate",
                        },
                    )
                    fig_b.add_hline(
                        y=base_ref,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="baseline_wr",
                    )
                    st.plotly_chart(
                        fig_b, use_container_width=True
                    )

                res_meit = compute_meit(
                    reg_id_v2,
                    pkm_m,
                    con_v2,
                    mega_enabled=mega_on,
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(
                        "activation_rate",
                        f"{res_meit['activation_rate']:.3f}",
                    )
                with c2:
                    st.metric(
                        "sustained_damage_ratio",
                        f"{res_meit['sustained_damage_ratio']:.3f}",
                    )
                with c3:
                    st.metric(
                        "opportunity_cost",
                        f"{res_meit['opportunity_cost']:.3f}",
                    )
                st.metric(
                    "Score MEIT",
                    f"{res_meit['meit_score']:.4f}",
                )
    except Exception as exc:
        st.error(f"METI/MEIT: {exc}")


with tab_tai:
    # ── Métricas V2 — Bloque 15 ─────────────
    st.subheader("🎲 TAI — Teambuilder Adaptability")
    try:
        from src.app.metrics.advanced_metrics import (
            compute_tai,
        )

        reg = st.session_state["regulation_config"]
        reg_id_v2 = reg.regulation_id
        con_v2 = get_duckdb()

        paste_tai = st.text_area(
            "Equipo (Showdown)",
            height=140,
            key="v2_tai_paste",
        )
        lam = st.slider(
            "λ penalización varianza",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            key="v2_tai_lambda",
        )
        if st.button("Calcular TAI", key="v2_tai_go"):
            sl = _parse_paste_to_slugs(paste_tai)
            if len(sl) < 2:
                st.warning(
                    "Se necesitan al menos 2 Pokémon."
                )
            else:
                out = compute_tai(
                    reg_id_v2,
                    tuple(sl[:6]),
                    con_v2,
                    lambda_penalty=lam,
                )
                if out["found"]:
                    st.metric(
                        "TAI score",
                        f"{out['tai_score']:.4f}",
                    )
                    br = pd.DataFrame(
                        out["archetype_breakdown"]
                    )
                    if not br.empty and set(
                        [
                            "archetype_id",
                            "usage_pct",
                            "estimated_wr",
                        ]
                    ).issubset(br.columns):
                        st.dataframe(
                            br[
                                [
                                    "archetype_id",
                                    "usage_pct",
                                    "estimated_wr",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.dataframe(
                            br,
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    st.warning(
                        "Sin clusters disponibles — "
                        "ejecuta el meta clustering"
                    )
    except Exception as exc:
        st.error(f"TAI: {exc}")


with tab_shapley:
    # ── Métricas V2 — Bloque 15 ─────────────
    st.subheader("🔢 Shapley Slot Value")
    try:
        from src.app.metrics.advanced_metrics import (
            shapley_slot_value,
        )

        reg = st.session_state["regulation_config"]
        reg_id_v2 = reg.regulation_id
        con_v2 = get_duckdb()

        paste_sv = st.text_area(
            "Equipo (Showdown)",
            height=140,
            key="v2_sv_paste",
        )
        st.caption(
            "Sin modelo WP activo — usando "
            "distribución uniforme"
        )
        if st.button(
            "Calcular Shapley Values",
            key="v2_sv_go",
        ):
            sl = _parse_paste_to_slugs(paste_sv)
            if len(sl) < 2:
                st.warning(
                    "Se necesitan al menos 2 Pokémon."
                )
            else:
                wp_model = None
                sv = shapley_slot_value(
                    reg_id_v2,
                    tuple(sl[:6]),
                    con_v2,
                    wp_model=wp_model,
                )
                if not sv:
                    st.info(
                        "Sin resultado (equipo "
                        "demasiado corto)."
                    )
                else:
                    import plotly.graph_objects as go

                    labels = list(sv.keys())
                    vals = list(sv.values())
                    colors = [
                        "#2ecc71" if v > 0 else "#e74c3c"
                        for v in vals
                    ]
                    fig_s = go.Figure(
                        data=[
                            go.Bar(
                                x=labels,
                                y=vals,
                                marker_color=colors,
                            )
                        ]
                    )
                    fig_s.update_layout(
                        title=(
                            "Contribución por slot "
                            "(Σ = WR - 0.5)"
                        ),
                        xaxis_title="pokemon_slug",
                        yaxis_title="shapley_value",
                        height=400,
                    )
                    st.plotly_chart(
                        fig_s, use_container_width=True
                    )
    except Exception as exc:
        st.error(f"Shapley SV: {exc}")
