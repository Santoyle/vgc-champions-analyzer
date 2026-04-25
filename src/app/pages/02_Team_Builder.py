from __future__ import annotations

import json
import logging
from pathlib import Path
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
from src.app.modules.pmi import compute_pmi_from_teammates, get_top_teammates
from src.app.modules.validate import (
    format_errors_for_ui,
    validate_team as validate_team_module,
)
from src.app.utils.db import get_duckdb, get_sqlite
from src.app.utils.session import init_session

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_POKEMON_MASTER_PATH = _PROJECT_ROOT / "data" / "pokemon_master.json"

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


@st.cache_data(show_spinner=False)
def load_pokemon_master() -> dict[int, str]:
    """
    Carga el mapeo dex_id → nombre desde data/pokemon_master.json.

    Los nombres se capitalizan para display: "bulbasaur" → "Bulbasaur".
    Usa .capitalize() (no .title()) para preservar guiones:
    "mr-mime" → "Mr-mime" en lugar de "Mr-Mime".

    Returns:
        Dict {dex_id: nombre_capitalizado}.
        Dict vacío si el archivo no existe o hay error.
    """
    try:
        if not _POKEMON_MASTER_PATH.exists():
            return {}
        raw = json.loads(_POKEMON_MASTER_PATH.read_text(encoding="utf-8"))
        pokemon_data = raw.get("pokemon", {})
        return {
            int(dex_id): entry["name"].capitalize()
            for dex_id, entry in pokemon_data.items()
            if isinstance(entry, dict) and "name" in entry
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Error cargando pokemon_master.json: %s", exc)
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_legal_pokemon(
    reg_id: str,
    _reg_config: RegulationConfig,
) -> list[str]:
    """
    Retorna lista de nombres de Pokémon legales para el selector de la UI,
    usando nombres canónicos desde pokemon_master.json.

    Si un dex_id no tiene nombre en pokemon_master, usa "Pokemon #{dex_id}"
    como fallback.

    Args:
        reg_id: Para clave de cache.
        _reg_config: RegulationConfig (prefijo _ para que cache_data
                     lo ignore al hashear).

    Returns:
        Lista de strings con nombres capitalizados, ordenados alfabéticamente.
    """
    master = load_pokemon_master()
    names: list[str] = [
        master.get(dex_id, f"Pokemon #{dex_id}")
        for dex_id in _reg_config.pokemon_legales
    ]
    return sorted(names)


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


@st.cache_data(ttl=3600, show_spinner=False)
def load_pmi_data(
    reg_id: str,
    _con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Carga y cachea el DataFrame de PMI calculado.
    Combina teammates + usage para el cálculo.
    """
    try:
        register_raw_view(_con)
        df_tm = create_teammates_by_pkm(_con, reg_id)
        df_usage = create_usage_by_reg(_con, reg_id)
        if df_tm.empty or df_usage.empty:
            return pd.DataFrame()
        return compute_pmi_from_teammates(df_tm, df_usage)
    except Exception as exc:  # noqa: BLE001
        log.debug("Sin datos PMI para %s: %s", reg_id, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Lógica de validación y exportación
# ---------------------------------------------------------------------------


def validate_team_ui(
    slots: list[dict[str, Any]],
    reg_config: RegulationConfig,
) -> list[str]:
    """
    Wrapper UI de validate_team del módulo validate.
    Convierte ValidationError a strings para display en Streamlit.
    La lógica real vive en src/app/modules/validate.py — testeable
    sin levantar la app.
    """
    errors = validate_team_module(slots, reg_config)
    return format_errors_for_ui(errors)


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
    state_messages = {
        "transition": (
            f"⚠️ **Transición de regulación** — "
            f"Estás viendo **{reg_id}** durante la ventana de transición. "
            f"Los datos pueden estar incompletos."
        ),
        "no_active": (
            f"⏸️ **Sin regulación activa** — "
            f"Estás viendo datos históricos de **{reg_id}** "
            f"({reg_config.date_start} → {reg_config.date_end}). "
            f"Selecciona otra regulación en el sidebar si hay una más reciente."
        ),
    }
    msg = state_messages.get(
        active_state,
        f"📁 Regulación histórica **{reg_id}** — "
        f"los equipos válidos en esta regulación pueden no serlo en la activa.",
    )
    st.info(msg)

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

    df_pmi = load_pmi_data(reg_id, con)

    if df_pmi.empty:
        st.caption(f"Sin datos de co-uso disponibles para {reg_id}.")
    else:
        team_pokemon = [s["species"] for s in current_slots if s.get("species")]
        selected_pokemon = team_pokemon[0] if team_pokemon else None

        if selected_pokemon:
            suggestions = get_top_teammates(
                df_pmi,
                pokemon=selected_pokemon,
                top_n=10,
                min_ppmi=0.0,
                exclude=team_pokemon,
            )
            if suggestions:
                st.caption(
                    f"Mejores compañeros para **{selected_pokemon}** (por PPMI):"
                )
                for pair in suggestions[:8]:
                    bar_width = min(int(pair.ppmi * 20), 20)
                    bar = "█" * bar_width
                    st.write(
                        f"• **{pair.teammate}** `{bar}` {pair.co_usage_pct:.1f}%"
                    )
            else:
                st.caption(
                    f"Sin sugerencias PMI para {selected_pokemon} en {reg_id}."
                )
        else:
            st.caption("Selecciona un Pokémon para ver sugerencias PMI.")

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

    if not paste_output:
        st.info("Completa al menos 1 slot para generar el paste.")
    else:
        st.code(paste_output, language="text")
        st.caption("Copia este paste y pégalo en Pokémon Showdown o compártelo.")

        st.divider()

        filled_pokemon = [
            s["species"]
            for s in st.session_state["team_slots"]
            if s.get("species")
        ]

        if filled_pokemon:
            st.subheader("Contexto histórico del equipo")

            with st.spinner(f"Analizando equipo en {reg_id}..."):
                try:
                    register_raw_view(con)
                    df_usage = create_usage_by_reg(con, reg_id)
                except Exception:  # noqa: BLE001
                    df_usage = pd.DataFrame()

            if df_usage.empty:
                st.caption(f"Sin datos de uso disponibles para {reg_id}.")
            else:
                st.caption(
                    f"Uso de tus Pokémon en el meta de **{reg_id}** "
                    f"({reg_config.date_start} → {reg_config.date_end}):"
                )
                for pkm in filled_pokemon:
                    match = df_usage[
                        df_usage["pokemon"].str.lower() == pkm.lower()
                    ]
                    if not match.empty:
                        usage = float(match.iloc[0]["avg_usage_pct"])
                        n_months = int(match.iloc[0].get("n_months", 1))
                        bar_len = min(int(usage / 2), 30)
                        bar = "█" * bar_len
                        tier = (
                            "🔥 Top meta"
                            if usage >= 30
                            else "📊 Presente"
                            if usage >= 10
                            else "💡 Nicho"
                        )
                        st.write(
                            f"• **{pkm}**: {usage:.1f}% `{bar}` {tier} "
                            f"_(dato: {n_months} mes(es))_"
                        )
                    else:
                        st.write(
                            f"• **{pkm}**: sin datos en {reg_id} "
                            f"_(puede ser nuevo en Champions)_"
                        )

                if active_state != "active":
                    st.info(
                        f"📁 Estos datos corresponden a **{reg_id}** "
                        f"({reg_config.date_start} → {reg_config.date_end}), "
                        f"una regulación histórica. "
                        f"El meta actual puede ser muy diferente."
                    )

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
