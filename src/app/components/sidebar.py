"""
Componente del selector global de regulación para el sidebar de Streamlit.

render_regulation_selector() debe llamarse UNA SOLA VEZ desde el entrypoint
streamlit_app.py, después de init_session() y antes de pg.run(). Al estar
en el entrypoint, el selector aparece automáticamente en todas las páginas
sin que cada página tenga que incluirlo.

No llamar este componente desde páginas individuales.
"""

from __future__ import annotations

import logging
from datetime import date

import streamlit as st

from src.app.core.regulation_active import (
    RegulationState,
    get_active_regulation,
)
from src.app.core.schema import RegulationConfig
from src.app.utils.session import (
    REG_DIR,
    get_available_regulations,
    load_regulation,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _get_regulation_state(
    reg_id: str,
    active_reg_id: str,
    active_state: RegulationState,
    today: date | None = None,  # noqa: ARG001
) -> RegulationState:
    """
    Determina el estado visual de una regulación para el selector.

    Para la regulación activa retorna active_state. Para todas las demás
    retorna "no_active" — el label se construye comparando fechas
    directamente en _build_label sin necesidad de un estado más granular.

    Args:
        reg_id: ID de la regulación a evaluar.
        active_reg_id: ID de la regulación activa en sesión.
        active_state: Estado de la regulación activa.
        today: Fecha de referencia. Default: date.today().

    Returns:
        RegulationState para determinar el label visual.
    """
    if reg_id == active_reg_id:
        return active_state
    return "no_active"


def _build_label(
    reg_id: str,
    config: RegulationConfig,
    active_reg_id: str,
    active_state: RegulationState,
    today: date | None = None,
) -> str:
    """
    Construye el label visible en el selectbox según el estado temporal.

    Formatos:
        Activa:       "{reg_id}  ✅  (activa)"
        Transición:   "{reg_id}  ⚠️  (transición)"
        Sin activa:   "{reg_id}  ⏸️  (sin reg activa)"
        Futura:       "{reg_id}  🔜  (desde {DD/MM})"
        Histórica:    "{reg_id}  📁  ({mes AAAA}–{mes AAAA})"

    Args:
        reg_id: ID de la regulación.
        config: RegulationConfig de la regulación.
        active_reg_id: ID de la reg actualmente activa en sesión.
        active_state: Estado de la reg activa.
        today: Fecha de referencia. Default: date.today().

    Returns:
        String formateado para mostrar en el selectbox.
    """
    today = today or date.today()

    if reg_id == active_reg_id:
        if active_state == "active":
            return f"{reg_id}  ✅  (activa)"
        if active_state == "transition":
            return f"{reg_id}  ⚠️  (transición)"
        return f"{reg_id}  ⏸️  (sin reg activa)"

    if config.date_start > today:
        start_fmt = config.date_start.strftime("%d/%m")
        return f"{reg_id}  🔜  (desde {start_fmt})"

    start_fmt = config.date_start.strftime("%b %Y")
    end_fmt = config.date_end.strftime("%b %Y")
    return f"{reg_id}  📁  ({start_fmt}–{end_fmt})"


def _render_state_banner(
    active_state: RegulationState,
    selected_reg_id: str,
) -> None:
    """
    Muestra un caption en el sidebar con el estado de la regulación.

    Si el usuario seleccionó una regulación distinta a la detectada
    automáticamente, muestra el banner de modo histórico. Si está
    en la reg auto-detectada, muestra el estado correspondiente.

    Args:
        active_state: Estado de la reg seleccionada en session_state.
        selected_reg_id: ID de la regulación actualmente seleccionada.
    """
    try:
        auto_active = get_active_regulation(REG_DIR)
        if selected_reg_id != auto_active.regulation_id:
            st.sidebar.caption(
                "📁 Modo histórico — esta regulación no está activa"
            )
            return
    except Exception:  # noqa: BLE001
        pass

    state_badges: dict[RegulationState, str] = {
        "active": "✅ Regulación activa",
        "transition": "⚠️ Ventana de transición",
        "no_active": "⏸️ Sin regulación activa",
    }
    badge = state_badges.get(active_state, "")
    if badge:
        st.sidebar.caption(badge)


# ---------------------------------------------------------------------------
# Componente público
# ---------------------------------------------------------------------------


def render_regulation_selector() -> None:
    """
    Renderiza el selector de regulación en el sidebar de Streamlit.

    Debe llamarse UNA SOLA VEZ desde streamlit_app.py, después de
    init_session() y antes de pg.run(). No llamar desde páginas
    individuales — al estar en el entrypoint aparece en todas.

    Comportamiento:
      1. Obtiene las regulaciones disponibles en regulations/.
      2. Construye labels con íconos según estado temporal de cada reg.
      3. Muestra st.sidebar.selectbox con la regulación activa por defecto.
      4. Si el usuario cambia la selección, actualiza session_state y
         llama st.rerun() para recargar con la nueva reg.
      5. Muestra un banner contextual bajo el selector.
      6. Si no hay regulaciones: muestra st.sidebar.error y retorna.

    Side effects:
        Modifica st.session_state["selected_reg_id"] y
        st.session_state["regulation_config"] si el usuario cambia
        la selección. Llama st.rerun() solo cuando hay cambio real.
    """
    reg_ids = get_available_regulations()

    if not reg_ids:
        st.sidebar.error(
            "❌ No hay regulaciones disponibles.\n"
            "Agrega al menos un JSON en regulations/"
        )
        return

    current_reg_id: str = st.session_state.get("selected_reg_id", reg_ids[0])
    active_reg_id: str = st.session_state.get("selected_reg_id", reg_ids[0])
    active_state: RegulationState = st.session_state.get("active_state", "active")

    # Construir labels con íconos para cada regulación
    labels: dict[str, str] = {}
    for rid in reg_ids:
        try:
            config = load_regulation(rid)
            labels[rid] = _build_label(rid, config, active_reg_id, active_state)
        except Exception:  # noqa: BLE001
            labels[rid] = f"{rid}  ⚠️  (error al cargar)"

    current_idx = reg_ids.index(current_reg_id) if current_reg_id in reg_ids else 0

    selected: str = st.sidebar.selectbox(  # type: ignore[assignment]
        "Regulación",
        options=reg_ids,
        index=current_idx,
        format_func=lambda rid: labels.get(rid, rid),
        key="_reg_selector_widget",
        help=(
            "La regulación activa se detecta automáticamente por fecha. "
            "Puedes seleccionar regulaciones históricas para análisis comparativo."
        ),
    )

    # st.rerun() SOLO si la selección cambió — nunca incondicionalmente
    if selected != st.session_state.get("selected_reg_id"):
        try:
            new_config = load_regulation(selected)
            st.session_state["selected_reg_id"] = selected
            st.session_state["regulation_config"] = new_config
            log.info("Regulación cambiada a: %s", selected)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"❌ Error al cargar {selected}: {exc}")
            return

    _render_state_banner(active_state, current_reg_id)


__all__ = [
    "render_regulation_selector",
]
