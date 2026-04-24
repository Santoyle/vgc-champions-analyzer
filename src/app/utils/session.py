"""
Gestión del estado de sesión global de Streamlit.

init_session() debe llamarse como primera línea en cada página de
Streamlit, inmediatamente después de los imports. Es idempotente:
si la sesión ya fue inicializada (por otra página visitada antes),
retorna sin hacer nada y preserva la selección del usuario.

Este módulo también expone las APIs para cargar regulaciones desde
disco con invalidación automática de cache basada en mtime del archivo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src.app.core.regulation_active import (
    ActiveRegulation,
    RegulationState,
    get_active_regulation,
)
from src.app.core.schema import RegulationConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

# session.py está en src/app/utils/ — 4 niveles abajo de la raíz del proyecto:
# raíz ← src/ ← src/app/ ← src/app/utils/ ← src/app/utils/session.py
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
REG_DIR = _PROJECT_ROOT / "regulations"


# ---------------------------------------------------------------------------
# Carga de regulaciones con cache invalidable por mtime
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_regulation_cached(
    reg_id: str,
    mtime_ns: int,  # noqa: ARG001 — solo actúa como clave de cache
) -> RegulationConfig:
    """
    Carga y valida un JSON de regulación desde disco.

    Cacheada por (reg_id, mtime_ns). Cuando el archivo cambia en disco,
    mtime_ns cambia y Streamlit re-ejecuta automáticamente. El parámetro
    mtime_ns no se usa en el cuerpo — solo actúa como clave de
    invalidación del cache.

    Args:
        reg_id: Identificador de la regulación.
        mtime_ns: Timestamp de modificación del archivo en nanosegundos.
                  Solo para cache key.

    Returns:
        RegulationConfig validado.

    Raises:
        FileNotFoundError: Si el JSON no existe.
        ValidationError: Si el JSON no es válido según el schema.
    """
    import json

    path = REG_DIR / f"{reg_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"regulations/{reg_id}.json no existe. "
            f"Verifica que el archivo está en {REG_DIR}."
        )
    raw: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return RegulationConfig.model_validate(raw)


def load_regulation(reg_id: str) -> RegulationConfig:
    """
    API pública para cargar una regulación.

    Obtiene el mtime_ns del archivo automáticamente y llama a la versión
    cacheada. El caller nunca necesita gestionar el mtime_ns manualmente.

    Args:
        reg_id: Identificador de la regulación.

    Returns:
        RegulationConfig validado y cacheado.

    Raises:
        FileNotFoundError: Si el JSON no existe.
    """
    path = REG_DIR / f"{reg_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"regulations/{reg_id}.json no existe."
        )
    mtime_ns = path.stat().st_mtime_ns
    return _load_regulation_cached(reg_id, mtime_ns)


# ---------------------------------------------------------------------------
# Listado de regulaciones disponibles con cache invalidable por mtime
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def list_available_regulations(
    dir_mtime_ns: int,  # noqa: ARG001 — solo actúa como clave de cache
) -> list[str]:
    """
    Lista los regulation_ids disponibles en regulations/.

    Cacheada por dir_mtime_ns del directorio. Cuando se agrega o elimina
    un JSON, el mtime del directorio cambia y el cache se invalida
    automáticamente.

    Args:
        dir_mtime_ns: Timestamp de modificación del directorio
                      regulations/ en nanosegundos. Solo para cache key.

    Returns:
        Lista de stems de archivos JSON ordenada alfabéticamente.
        Ej: ["H", "I", "M-A"]
    """
    if not REG_DIR.exists():
        return []
    return sorted(p.stem for p in REG_DIR.glob("*.json"))


def get_available_regulations() -> list[str]:
    """
    API pública para listar regulaciones disponibles.

    Returns:
        Lista de regulation_ids disponibles, ordenada alfabéticamente.
        Lista vacía si regulations/ no existe o está vacío.
    """
    if not REG_DIR.exists():
        return []
    dir_mtime_ns = REG_DIR.stat().st_mtime_ns
    return list_available_regulations(dir_mtime_ns)


# ---------------------------------------------------------------------------
# Inicialización de sesión
# ---------------------------------------------------------------------------


def init_session() -> None:
    """
    Inicializa el estado de sesión global de Streamlit.

    IDEMPOTENTE: verifica en la primera línea si "selected_reg_id" ya
    existe en st.session_state. Si existe, retorna inmediatamente sin
    tocar nada — preserva la selección del usuario al navegar entre
    páginas.

    Claves inicializadas en st.session_state:
        selected_reg_id (str): ID de la reg activa.
        regulation_config (RegulationConfig): Config completo.
        active_state (RegulationState): Estado de elección.
        active_reason (str): Mensaje humano-legible del motivo.

    Side effects:
        Llama st.error() + st.stop() si no hay regulaciones
        disponibles — la app no puede funcionar sin al menos un
        JSON de regulación válido.

    Raises:
        No levanta excepciones — todos los errores se manejan
        internamente con st.error/st.stop.
    """
    if "selected_reg_id" in st.session_state:
        return

    try:
        active: ActiveRegulation = get_active_regulation(reg_dir=REG_DIR)
    except FileNotFoundError as exc:
        st.error(
            f"❌ No se encontraron regulaciones. {exc}\n\n"
            f"Asegúrate de que regulations/*.json existe "
            f"antes de iniciar la app."
        )
        st.stop()
        return  # inalcanzable — st.stop() detiene la ejecución; necesario para mypy

    st.session_state["selected_reg_id"] = active.regulation_id
    st.session_state["regulation_config"] = active.config
    st.session_state["active_state"] = active.state
    st.session_state["active_reason"] = active.reason

    log.info(
        "Sesión inicializada: reg=%s state=%s",
        active.regulation_id,
        active.state,
    )


__all__ = [
    "REG_DIR",
    "load_regulation",
    "get_available_regulations",
    "init_session",
]
