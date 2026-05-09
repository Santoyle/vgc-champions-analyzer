from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import streamlit as st

from src.app.core.schema import RegulationConfig
from src.app.data.parsers.ps_paste import parse_paste
from src.app.modules.counter import (
    CounterResult,
    TeamMatchupResult,
    WPCounterResult,
    _load_pokemon_data,
    heuristic_counter,
    top_teams_vs_rival,
    wp_counter,
)
from src.app.utils.db import get_duckdb
from src.app.utils.session import init_session

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers de UI (no en counter.py — separación dominio/UI)
# ---------------------------------------------------------------------------


def _display_heuristic_results(
    counters: list[CounterResult],
    reg_id: str,
) -> None:
    """
    Renderiza los counters heurísticos en la UI.
    Extraída para reutilización entre modo heurístico y fallback del WP.
    """
    if not counters:
        st.info(
            "Sin datos suficientes para calcular counters. "
            "Verifica que pokemon_master.json esté disponible."
        )
        return

    st.caption(
        f"Top {len(counters)} counters heurísticos (Regulación: {reg_id}):"
    )
    for idx, counter in enumerate(counters[:10], start=1):
        bar_len = max(1, int(counter.score * 20))
        bar = "█" * bar_len
        types_str = " · ".join(counter.types) if counter.types else "?"
        threatens = (
            ", ".join(counter.counters_directly[:3])
            if counter.counters_directly
            else "—"
        )
        with st.container(border=True):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(
                    f"**{idx}. {counter.species}**  `{bar}` {counter.score:.2f}"
                )
                st.caption(f"Tipo: {types_str} · Amenaza: {threatens}")
            with c2:
                st.caption(
                    f"Tipo: {counter.type_advantage_score:.2f} · "
                    f"Spe: {counter.speed_tier_score:.2f}"
                )
    st.caption(
        "Heuristico v1 — basado en ventaja de tipo y speed tier."
    )


# ---------------------------------------------------------------------------
# Helpers cacheados
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_legal_roster(
    reg_id: str,
    _reg_config: RegulationConfig,
) -> list[str]:
    """
    Carga la lista de nombres legales para el roster desde pokemon_master.json
    filtrado por los dex_ids de la regulación.
    """
    master_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "pokemon_master.json"
    )
    try:
        raw = json.loads(master_path.read_text(encoding="utf-8"))
        pokemon_data = raw.get("pokemon", {})
        legal_ids = set(_reg_config.pokemon_legales)
        return sorted(
            [
                entry["name"].capitalize()
                for entry in pokemon_data.values()
                if isinstance(entry, dict)
                and int(entry.get("dex_id", 0)) in legal_ids
            ]
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("Error cargando roster legal para %s: %s", reg_id, exc)
        return []


# ---------------------------------------------------------------------------
# Inicialización de sesión
# ---------------------------------------------------------------------------

init_session()
reg_id: str = st.session_state["selected_reg_id"]
reg_config: RegulationConfig = st.session_state["regulation_config"]
active_state: str = st.session_state.get("active_state", "active")

# ---------------------------------------------------------------------------
# Título
# ---------------------------------------------------------------------------

st.title("🎯 Counter Analyzer")
st.caption(
    f"Regulación: **{reg_id}** · "
    "Pega el paste Showdown del rival para obtener los mejores counters."
)

if active_state != "active":
    st.info(
        f"📁 Analizando contra datos históricos de **{reg_id}**. "
        f"El meta actual puede haber cambiado."
    )

# ---------------------------------------------------------------------------
# Layout: paste rival (izquierda) | counters (derecha)
# ---------------------------------------------------------------------------

col_paste, col_counters = st.columns([2, 3])

with col_paste:
    st.subheader("Equipo rival")
    rival_paste = st.text_area(
        "Paste Showdown del rival:",
        height=320,
        placeholder=(
            "Incineroar @ Sitrus Berry\n"
            "Ability: Intimidate\n"
            "Level: 50\n"
            "- Fake Out\n"
            "- Parting Shot\n"
            "...\n\n"
            "Garchomp @ Choice Scarf\n"
            "..."
        ),
        key="rival_paste_input",
        label_visibility="collapsed",
    )

    parse_clicked = st.button(
        "🔍 Analizar rival",
        type="primary",
        key="btn_analyze_rival",
        use_container_width=True,
    )

    # Parsear equipo rival
    rival_team = None
    if rival_paste.strip():
        rival_team = parse_paste(rival_paste)
        if rival_team.slots:
            st.success(
                f"✅ {len(rival_team.slots)} Pokémon del rival detectados."
            )
            st.write("**Equipo rival parseado:**")
            for slot in rival_team.slots:
                item_str = f" @ {slot.item}" if slot.item else ""
                st.write(f"• **{slot.species}**{item_str}")
            if rival_team.parse_warnings:
                for w in rival_team.parse_warnings:
                    st.caption(f"⚠️ {w}")
        else:
            st.warning(
                "No se pudo parsear ningún Pokémon. "
                "Verifica el formato del paste."
            )
            for w in rival_team.parse_warnings:
                st.caption(f"⚠️ {w}")

with col_counters:
    if rival_team is None or not rival_team.slots:
        st.info(
            "👈 Pega el paste Showdown del rival y haz click en "
            "**Analizar rival** para ver los counters sugeridos."
        )
    else:
        st.subheader("Counters sugeridos")

        # Selector de modo
        counter_mode = st.radio(
            "Modo de análisis:",
            options=[
                "Heuristico v1 (tipo + velocidad)",
                "WP Model (requiere modelo entrenado)",
            ],
            horizontal=True,
            key="counter_mode_selector",
        )

        # Toggle de roster
        roster_mode = st.radio(
            "Roster de counters:",
            options=[
                "Legal en regulación activa",
                "Todos los Pokémon (Dex completo)",
            ],
            horizontal=True,
            key="counter_roster_mode",
        )

        with st.spinner("Calculando counters..."):
            if roster_mode.startswith("Legal"):
                roster = load_legal_roster(reg_id, reg_config)
            else:
                pkm_data = _load_pokemon_data()
                roster = [
                    v["name"].capitalize()
                    for v in pkm_data.values()
                    if isinstance(v, dict) and "name" in v
                ]

            use_wp = counter_mode.startswith("WP")

            if use_wp:
                wp_results = wp_counter(
                    rival_team,
                    roster=roster,
                    regulation_id=reg_id,
                    top_n=15,
                )
            else:
                wp_results = []

            heuristic_results = heuristic_counter(
                rival_team,
                roster=roster,
                top_n=15,
            )

        if use_wp:
            if not wp_results:
                st.warning(
                    f"Modelo WP no disponible para **{reg_id}**. "
                    "Entrena el modelo primero:\n"
                    "```\n"
                    "python scripts/train_wp.py --synthetic-data\n"
                    "```\n"
                    "Mostrando heuristico v1 como fallback:"
                )
                _display_heuristic_results(heuristic_results, reg_id)
            else:
                st.caption(
                    f"Top {len(wp_results)} counters via WP Model — "
                    f"Regulación: **{reg_id}**"
                )
                for idx, counter in enumerate(wp_results[:10], start=1):
                    bar_len = max(1, int(counter.wp_score * 20))
                    bar = "█" * bar_len
                    types_str = (
                        " · ".join(counter.types) if counter.types else "?"
                    )
                    threatens = (
                        ", ".join(counter.counters_directly[:3])
                        if counter.counters_directly
                        else "—"
                    )
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 2])
                        with c1:
                            st.markdown(
                                f"**{idx}. {counter.species}**"
                                f"  `{bar}` {counter.wp_score:.2f}"
                            )
                            st.caption(
                                f"Tipo: {types_str} · Amenaza: {threatens}"
                            )
                        with c2:
                            st.caption(
                                f"WP: {counter.wp_score:.3f} · "
                                f"Heur: {counter.heuristic_score:.2f}"
                            )
                st.caption(
                    "WP Model v1 — probabilidad de victoria estimada "
                    "con XGBoost calibrado."
                )
        else:
            _display_heuristic_results(heuristic_results, reg_id)

        # ── Top-5 equipos anti-rival ──────────────────────────────────────

        st.divider()
        st.subheader("Top equipos anti-rival")
        st.caption(
            "Equipos completos de replays recientes evaluados contra el equipo rival."
        )

        with st.spinner("Buscando mejores equipos..."):
            top_teams = top_teams_vs_rival(
                rival_team,
                regulation_id=reg_id,
                max_candidate_teams=200,
                top_n=5,
            )

        if not top_teams:
            st.info(
                "Sin equipos candidatos disponibles. "
                f"El pipeline LIVE necesita replays para **{reg_id}**."
            )
        else:
            use_wp_scores = any(t.wp_score is not None for t in top_teams)
            score_label = "WP Score" if use_wp_scores else "Score heuristico"
            st.caption(
                f"Scoring: **{score_label}** · {len(top_teams)} equipos encontrados"
            )

            for rank, team_result in enumerate(top_teams, start=1):
                score = (
                    team_result.wp_score
                    if team_result.wp_score is not None
                    else team_result.heuristic_score
                )
                score_str = f"{score:.3f}"
                mode_label = "WP" if use_wp_scores else "Heur"

                with st.expander(
                    f"#{rank} — Score: {score_str} ({mode_label})",
                    expanded=(rank == 1),
                ):
                    n_cols = min(6, len(team_result.team))
                    team_cols = st.columns(n_cols)
                    for i, pkm in enumerate(team_result.team[:6]):
                        with team_cols[i % n_cols]:
                            st.markdown(f"**{pkm}**")

                    meta_parts = [f"Regulación: {team_result.regulation_id}"]
                    if team_result.source_replay_id:
                        meta_parts.append(
                            f"Replay: {team_result.source_replay_id[:12]}..."
                        )
                    if team_result.wp_score is not None:
                        meta_parts.append(f"WP: {team_result.wp_score:.3f}")
                    meta_parts.append(
                        f"Heur: {team_result.heuristic_score:.3f}"
                    )
                    st.caption(" · ".join(meta_parts))
