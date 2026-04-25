from __future__ import annotations

import json
import logging
from typing import Any

import duckdb
import pandas as pd
import streamlit as st

from src.app.core.schema import RegulationConfig
from src.app.data.parsers.ps_paste import parse_paste, team_to_records
from src.app.data.sql.views import (
    create_teammates_by_pkm,
    create_usage_by_reg,
    register_raw_view,
)
from src.app.utils.db import get_duckdb, get_sqlite
from src.app.utils.session import init_session

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers cacheados
# ---------------------------------------------------------------------------

_NATURES = [
    "Hardy", "Lonely", "Brave", "Adamant", "Naughty",
    "Bold", "Docile", "Relaxed", "Impish", "Lax",
    "Timid", "Hasty", "Serious", "Jolly", "Naive",
    "Modest", "Mild", "Quiet", "Bashful", "Rash",
    "Calm", "Gentle", "Sassy", "Careful", "Quirky",
]

_TERA_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]

_EMPTY_SLOT: dict[str, Any] = {
    "species": "",
    "item": "",
    "ability": "",
    "tera_type": "",
    "nature": "Hardy",
    "moves": ["", "", "", ""],
    "mega_capable": False,
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_legal_pokemon(
    reg_id: str,
    _reg_config: RegulationConfig,
) -> list[str]:
    """
    Retorna lista de nombres de Pokémon legales para el selector de la UI.

    En MVP retorna los primeros 200 por nombre ordenados alfabéticamente
    desde pokemon_legales del RegulationConfig. Los nombres reales vienen
    de pokemon_master.json en una tarea posterior — por ahora usa IDs
    convertidos a string como placeholder.

    Args:
        reg_id: Para clave de cache.
        _reg_config: RegulationConfig (prefijo _ para que cache_data
                     lo ignore al hashear).

    Returns:
        Lista de strings con nombres de Pokémon.
    """
    dex_ids = _reg_config.pokemon_legales[:200]
    return [f"Pokemon #{dex_id}" for dex_id in sorted(dex_ids)]


@st.cache_data(ttl=3600, show_spinner=False)
def load_teammates_suggestions(
    reg_id: str,
    _con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Carga sugerencias de compañeros (PMI co-usage).

    Args:
        reg_id: ID de la regulación.
        _con: Conexión DuckDB (ignorada por cache).

    Returns:
        DataFrame con columnas pokemon, teammate, avg_correlation.
        Vacío si no hay datos.
    """
    try:
        register_raw_view(_con)
        df = create_teammates_by_pkm(_con, reg_id)
        return df
    except Exception as exc:  # noqa: BLE001
        log.debug("Sin datos de teammates para %s: %s", reg_id, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Lógica de validación y exportación
# ---------------------------------------------------------------------------


def validate_team_ui(
    slots: list[dict[str, Any]],
    reg_config: RegulationConfig,
) -> list[str]:
    """
    Valida el equipo actual contra las reglas de la regulación seleccionada.

    Checks implementados:
    1. Species clause: no repetir especie.
    2. Item clause (si activa): no repetir ítem.
    3. Max 1 Mega-capable (si mega_enabled).
    4. Retorna lista vacía para equipo vacío — no validar lo que no existe.

    Args:
        slots: Lista de dicts con keys "species", "item", "mega_capable".
        reg_config: Configuración de la regulación.

    Returns:
        Lista de strings de error. Vacía si el equipo es válido o vacío.
    """
    errors: list[str] = []
    filled = [s for s in slots if s.get("species")]

    if not filled:
        return errors

    # Species clause
    species_list = [s["species"] for s in filled]
    if len(set(species_list)) != len(species_list):
        duplicates = [s for s in set(species_list) if species_list.count(s) > 1]
        errors.append(f"❌ Species clause: {duplicates} aparece más de una vez.")

    # Item clause
    if reg_config.clauses.item_clause:
        items = [s.get("item", "") for s in filled if s.get("item")]
        if len(set(items)) != len(items):
            dup_items = [i for i in set(items) if items.count(i) > 1]
            errors.append(
                f"❌ Item clause: {dup_items} aparece en más de un slot."
            )

    # Mega clause
    if reg_config.mechanics.mega_enabled:
        mega_count = sum(1 for s in filled if s.get("mega_capable", False))
        if mega_count > 1:
            errors.append(
                f"❌ Solo puede haber 1 Pokémon con Mega Stone. "
                f"Tienes {mega_count}."
            )

    return errors


def generate_showdown_paste(slots: list[dict[str, Any]]) -> str:
    """
    Genera un paste de Showdown desde los slots del equipo actual.

    Args:
        slots: Lista de dicts con datos del equipo.

    Returns:
        String en formato paste de Showdown. Vacío si no hay slots.
    """
    lines: list[str] = []
    for slot in slots:
        if not slot.get("species"):
            continue

        species = slot.get("species", "")
        item = slot.get("item", "")
        ability = slot.get("ability", "")
        tera = slot.get("tera_type", "")
        moves: list[str] = slot.get("moves", [])
        nature = slot.get("nature", "Hardy")

        lines.append(f"{species} @ {item}" if item else species)
        if ability:
            lines.append(f"Ability: {ability}")
        lines.append("Level: 50")
        if tera:
            lines.append(f"Tera Type: {tera}")
        lines.append(f"{nature} Nature")
        for move in moves[:4]:
            if move:
                lines.append(f"- {move}")
        lines.append("")

    return "\n".join(lines).strip()


def _slot_from_parsed(slot: Any) -> dict[str, Any]:
    """Convierte un ParsedSlot a dict de session_state."""
    return {
        "species": slot.species,
        "item": slot.item or "",
        "ability": slot.ability or "",
        "tera_type": slot.tera_type or "",
        "nature": slot.nature or "Hardy",
        "moves": (slot.moves + ["", "", "", ""])[:4],
        "mega_capable": slot.mega_capable,
    }


# ---------------------------------------------------------------------------
# Inicialización de sesión
# ---------------------------------------------------------------------------

init_session()
reg_id: str = st.session_state["selected_reg_id"]
reg_config: RegulationConfig = st.session_state["regulation_config"]
active_state: str = st.session_state.get("active_state", "active")

# ---------------------------------------------------------------------------
# Título y contexto
# ---------------------------------------------------------------------------

st.title("🛠️ Team Builder")
st.caption(
    f"Regulación: **{reg_id}** · "
    f"Mega: {'✅' if reg_config.mechanics.mega_enabled else '❌'} · "
    f"Tera: {'✅' if reg_config.mechanics.tera_enabled else '❌'}"
)

if active_state != "active":
    st.info(
        f"📁 Construyendo equipo para regulación histórica **{reg_id}**."
    )

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

con = get_duckdb()
legal_pokemon = load_legal_pokemon(reg_id, reg_config)
df_teammates = load_teammates_suggestions(reg_id, con)

# ---------------------------------------------------------------------------
# Inicializar team_slots en session_state (solo si no existe)
# ---------------------------------------------------------------------------

if "team_slots" not in st.session_state:
    st.session_state["team_slots"] = [
        dict(_EMPTY_SLOT) for _ in range(6)
    ]

# ---------------------------------------------------------------------------
# Layout principal: builder (izquierda) + info (derecha)
# ---------------------------------------------------------------------------

col_builder, col_info = st.columns([3, 2])

with col_builder:
    st.subheader("Equipo (6 slots)")

    for i in range(6):
        slot_label = (
            st.session_state["team_slots"][i].get("species")
            or f"Slot {i + 1} (vacío)"
        )
        with st.expander(f"**Slot {i + 1}:** {slot_label}", expanded=(i == 0)):
            s_col1, s_col2 = st.columns(2)

            with s_col1:
                species = st.selectbox(
                    "Pokémon",
                    options=[""] + legal_pokemon,
                    index=0,
                    key=f"slot_{i}_species",
                )
                st.session_state["team_slots"][i]["species"] = species

                item = st.text_input(
                    "Ítem",
                    value=st.session_state["team_slots"][i].get("item", ""),
                    key=f"slot_{i}_item",
                    placeholder="ej: Sitrus Berry",
                )
                st.session_state["team_slots"][i]["item"] = item

                ability = st.text_input(
                    "Habilidad",
                    value=st.session_state["team_slots"][i].get("ability", ""),
                    key=f"slot_{i}_ability",
                    placeholder="ej: Intimidate",
                )
                st.session_state["team_slots"][i]["ability"] = ability

            with s_col2:
                tera_options = (
                    _TERA_TYPES
                    if reg_config.mechanics.tera_enabled
                    else ["—"]
                )
                tera = st.selectbox(
                    "Tera Type",
                    options=[""] + tera_options,
                    key=f"slot_{i}_tera",
                    disabled=not reg_config.mechanics.tera_enabled,
                )
                st.session_state["team_slots"][i]["tera_type"] = tera

                nature = st.selectbox(
                    "Naturaleza",
                    options=_NATURES,
                    key=f"slot_{i}_nature",
                )
                st.session_state["team_slots"][i]["nature"] = nature

                mega_capable = st.checkbox(
                    "🔴 Mega Stone",
                    key=f"slot_{i}_mega",
                    disabled=not reg_config.mechanics.mega_enabled,
                )
                st.session_state["team_slots"][i]["mega_capable"] = mega_capable

            st.write("**Movimientos:**")
            m_cols = st.columns(2)
            moves: list[str] = []
            current_moves: list[str] = st.session_state["team_slots"][i].get(
                "moves", ["", "", "", ""]
            )
            for m_idx in range(4):
                with m_cols[m_idx % 2]:
                    move = st.text_input(
                        f"Move {m_idx + 1}",
                        value=current_moves[m_idx] if m_idx < len(current_moves) else "",
                        key=f"slot_{i}_move_{m_idx}",
                        placeholder="ej: Fake Out",
                        label_visibility="collapsed",
                    )
                    moves.append(move)
            st.session_state["team_slots"][i]["moves"] = moves

with col_info:
    st.subheader("Validación en vivo")

    current_slots: list[dict[str, Any]] = st.session_state["team_slots"]
    validation_errors = validate_team_ui(current_slots, reg_config)

    filled_count = sum(1 for s in current_slots if s.get("species"))
    st.metric("Slots completados", f"{filled_count}/6")

    if validation_errors:
        for err in validation_errors:
            st.error(err)
    elif filled_count > 0:
        st.success("✅ Equipo válido")
    else:
        st.info("Completa al menos 1 slot para validar.")

    st.divider()

    st.subheader("Sugerencias de compañeros")
    if df_teammates.empty:
        st.caption(f"Sin datos de co-uso disponibles para {reg_id}.")
    else:
        selected_pokemon = next(
            (s["species"] for s in current_slots if s.get("species")),
            None,
        )
        if selected_pokemon:
            suggestions = df_teammates[
                df_teammates["pokemon"] == selected_pokemon
            ].head(5)
            if not suggestions.empty:
                st.caption(f"Mejores compañeros para **{selected_pokemon}**:")
                for _, row in suggestions.iterrows():
                    st.write(
                        f"• {row['teammate']} ({row['avg_correlation']:.1f}%)"
                    )
            else:
                st.caption(
                    f"Sin sugerencias para {selected_pokemon} en {reg_id}."
                )
        else:
            st.caption("Selecciona un Pokémon para ver sugerencias.")

st.divider()

# ---------------------------------------------------------------------------
# Tabs inferiores: Import / Export / Guardados
# ---------------------------------------------------------------------------

tab_import, tab_export, tab_saved = st.tabs([
    "📥 Importar paste",
    "📤 Exportar paste",
    "💾 Equipos guardados",
])

# ── Tab: Importar ────────────────────────────────────────────────────────

with tab_import:
    st.subheader("Importar desde paste Showdown")
    paste_input = st.text_area(
        "Pega tu paste aquí:",
        height=250,
        placeholder=(
            "Incineroar @ Sitrus Berry\n"
            "Ability: Intimidate\n"
            "Level: 50\n"
            "Tera Type: Fire\n"
            "EVs: 252 HP / 4 Atk / 252 Def\n"
            "Impish Nature\n"
            "- Fake Out\n"
            "- Parting Shot\n"
            "..."
        ),
        key="paste_import_input",
    )

    if st.button("📥 Importar equipo", type="primary", key="btn_import"):
        if paste_input.strip():
            parsed = parse_paste(paste_input)
            if parsed.slots:
                new_slots = [_slot_from_parsed(s) for s in parsed.slots[:6]]
                while len(new_slots) < 6:
                    new_slots.append(dict(_EMPTY_SLOT))
                st.session_state["team_slots"] = new_slots
                st.success(
                    f"✅ {len(parsed.slots)} Pokémon importados correctamente."
                )
                for w in parsed.parse_warnings:
                    st.warning(f"⚠️ {w}")
                st.rerun()
            else:
                st.error("❌ No se pudo parsear ningún Pokémon del paste.")
        else:
            st.warning("El paste está vacío.")

# ── Tab: Exportar ────────────────────────────────────────────────────────

with tab_export:
    st.subheader("Exportar a paste Showdown")
    paste_output = generate_showdown_paste(st.session_state["team_slots"])

    if paste_output:
        st.code(paste_output, language="text")
        st.caption("Copia este paste y pégalo en Pokémon Showdown.")
    else:
        st.info("Completa al menos 1 slot para generar el paste.")

# ── Tab: Guardados ───────────────────────────────────────────────────────

with tab_saved:
    st.subheader("Equipos guardados")
    sqlite_con = get_sqlite()

    col_save, col_load = st.columns(2)

    with col_save:
        team_name = st.text_input(
            "Nombre del equipo",
            placeholder="ej: Mi equipo Sun",
            key="team_save_name",
        )
        if st.button("💾 Guardar equipo", key="btn_save"):
            paste = generate_showdown_paste(st.session_state["team_slots"])
            if not paste:
                st.warning("El equipo está vacío — agrega al menos 1 Pokémon.")
            elif not team_name.strip():
                st.warning("Ingresa un nombre para el equipo.")
            else:
                import uuid
                from datetime import datetime

                team_id = str(uuid.uuid4())[:8]
                sqlite_con.execute(
                    "INSERT INTO saved_teams "
                    "(team_id, name, regulation_id, paste_showdown, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        team_id,
                        team_name.strip(),
                        reg_id,
                        paste,
                        datetime.now().isoformat(),
                    ),
                )
                sqlite_con.commit()
                st.success(
                    f"✅ Equipo '{team_name}' guardado (ID: {team_id})"
                )

    with col_load:
        saved = sqlite_con.execute(
            "SELECT team_id, name, regulation_id, created_at "
            "FROM saved_teams "
            "WHERE regulation_id = ? "
            "ORDER BY created_at DESC LIMIT 20",
            (reg_id,),
        ).fetchall()

        if saved:
            st.caption(
                f"{len(saved)} equipo(s) guardado(s) para {reg_id}:"
            )
            for tid, name, rid, created in saved:
                t_col1, t_col2 = st.columns([3, 1])
                with t_col1:
                    st.write(f"**{name}** _{created[:10]}_")
                with t_col2:
                    if st.button(
                        "Cargar",
                        key=f"load_{tid}",
                        use_container_width=True,
                    ):
                        row = sqlite_con.execute(
                            "SELECT paste_showdown FROM saved_teams "
                            "WHERE team_id = ?",
                            (tid,),
                        ).fetchone()
                        if row:
                            parsed_saved = parse_paste(row[0])
                            if parsed_saved.slots:
                                loaded_slots = [
                                    _slot_from_parsed(s)
                                    for s in parsed_saved.slots[:6]
                                ]
                                while len(loaded_slots) < 6:
                                    loaded_slots.append(dict(_EMPTY_SLOT))
                                st.session_state["team_slots"] = loaded_slots
                                st.success(f"✅ '{name}' cargado")
                                st.rerun()
        else:
            st.caption(f"Sin equipos guardados para {reg_id}.")
