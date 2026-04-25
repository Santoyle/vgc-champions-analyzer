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
    _load_pokemon_data,
    heuristic_counter,
)
from src.app.utils.db import get_duckdb
from src.app.utils.session import init_session

log = logging.getLogger(__name__)

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
        st.subheader("🎯 Counters sugeridos")

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

            counters = heuristic_counter(
                rival_team,
                roster=roster,
                top_n=15,
            )

        if not counters:
            st.info(
                "Sin datos suficientes para calcular counters. "
                "Verifica que pokemon_master.json esté disponible."
            )
        else:
            st.caption(
                f"Top {len(counters)} counters heurísticos para el equipo rival "
                f"(Regulación: {reg_id}):"
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
                "⚠️ Heurístico v1 — basado en ventaja de tipo y speed tier. "
                "Sin datos de replays todavía."
            )
