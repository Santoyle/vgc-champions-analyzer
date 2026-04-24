"""
Lógica de selección de la regulación activa.

Implementa get_active_regulation() con 3 escenarios en orden de
prioridad estricta:

  A  — Regulación única vigente hoy (date_start <= today <= date_end).
  B  — Ventana de transición: ninguna reg cubre today, pero una está
       a punto de empezar o acaba de terminar en los últimos 7 días.
  C  — Sin regulación activa: retorna la más reciente por date_end
       como fallback seguro.

El parámetro `today` es inyectable (default None → date.today()) para
que los tests puedan simular fechas arbitrarias de forma determinista
sin depender del reloj del sistema.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from src.app.core.schema import RegulationConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipos públicos
# ---------------------------------------------------------------------------

RegulationState = Literal[
    "active",      # Hay exactamente una reg vigente hoy
    "transition",  # Ventana de transición entre regs
    "no_active",   # Sin regulación vigente (hiato)
]


@dataclass(frozen=True)
class ActiveRegulation:
    """
    Resultado de get_active_regulation().
    Encapsula qué regulación está activa y por qué.

    Attributes:
        regulation_id: ID de la regulación elegida.
        config: RegulationConfig completo de la reg.
        state: Estado que motivó la elección.
        reason: Mensaje humano-legible para logs y UI.
    """

    regulation_id: str
    config: RegulationConfig
    state: RegulationState
    reason: str


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _load_all_regulations(reg_dir: Path) -> list[RegulationConfig]:
    """
    Carga y valida todos los archivos JSON en reg_dir.

    Ignora archivos corruptos o inválidos con log.warning —
    la app sigue funcionando con las regulaciones válidas disponibles.

    Args:
        reg_dir: Directorio que contiene los JSON de regulación
                 (ej: Path("regulations")).

    Returns:
        Lista de RegulationConfig válidos, ordenados por
        date_start ascendente.
    """
    if not reg_dir.exists():
        return []

    configs: list[RegulationConfig] = []
    for path in sorted(reg_dir.glob("*.json")):
        try:
            raw: dict[str, object] = json.loads(
                path.read_text(encoding="utf-8")
            )
            configs.append(RegulationConfig.model_validate(raw))
        except (json.JSONDecodeError, ValidationError, OSError) as exc:
            log.warning(
                "No se pudo cargar la regulación '%s': %s",
                path.name,
                exc,
            )

    configs.sort(key=lambda c: c.date_start)
    return configs


def _count_recent_tournaments(reg_id: str, days: int = 7) -> int:
    """
    Cuenta torneos recientes de una regulación en Limitless VGC
    (últimos N días).

    Usado en el Escenario B para desempatar entre regulaciones en
    ventana de transición.

    En MVP retorna 0 siempre — se implementa completamente en la
    Tarea 47 cuando exista el scraper de Limitless.

    Args:
        reg_id: ID de la regulación a consultar.
        days: Ventana de días hacia atrás.

    Returns:
        Número de torneos encontrados. 0 si no hay datos disponibles
        o hay error.
    """
    try:
        from src.app.data.scrapers import limitless  # type: ignore[import]

        return int(limitless.count_recent_tournaments(reg_id, days=days))
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "Limitless scraper no disponible para '%s' (days=%d): %s",
            reg_id,
            days,
            exc,
        )
        return 0


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------


def get_active_regulation(
    reg_dir: Path = Path("regulations"),
    today: date | None = None,
) -> ActiveRegulation:
    """
    Determina la regulación activa aplicando 3 escenarios en orden
    de prioridad estricta.

    ESCENARIO A — Regulación única vigente:
        Exactamente una reg con date_start <= today <= date_end.
        Retorna esa reg con state="active".

    ESCENARIO A' — Solapamiento (dato corrupto):
        Múltiples regs vigentes simultáneamente.
        Elige la de date_start más reciente.
        Loggea warning — CI debería haberlo bloqueado.

    ESCENARIO B — Ventana de transición:
        Ninguna reg cubre today, pero hay regs cuya ventana de
        transición alcanza today. Ventana: [date_start -
        transition_window_days, date_start). También incluye regs
        recién terminadas en los últimos 7 días.
        Desempate: reg con más torneos en Limitless en los
        últimos 7 días.
        Retorna con state="transition".

    ESCENARIO C — Sin regulación activa:
        Ningún escenario anterior aplica.
        Retorna la reg más reciente por date_end con
        state="no_active".

    Args:
        reg_dir: Directorio de regulaciones. Inyectable para testing.
        today: Fecha de referencia. Si None, usa date.today().
               Inyectable para testing determinista.

    Returns:
        ActiveRegulation con la reg elegida y el motivo.

    Raises:
        FileNotFoundError: Si reg_dir no existe o está vacío después
                           de filtrar inválidos.
    """
    today = today or date.today()
    regs = _load_all_regulations(reg_dir)

    if not regs:
        raise FileNotFoundError(
            f"No se encontraron regulaciones válidas en "
            f"{reg_dir.resolve()}. Crea al menos un JSON "
            f"antes de iniciar la app."
        )

    # ------------------------------------------------------------------
    # ESCENARIO A: exactamente una reg vigente
    # ------------------------------------------------------------------
    active = [r for r in regs if r.date_start <= today <= r.date_end]

    if len(active) == 1:
        r = active[0]
        return ActiveRegulation(
            regulation_id=r.regulation_id,
            config=r,
            state="active",
            reason=(
                f"Regulación '{r.regulation_id}' vigente "
                f"({r.date_start} → {r.date_end})."
            ),
        )

    # ------------------------------------------------------------------
    # ESCENARIO A': solapamiento (dato corrupto)
    # ------------------------------------------------------------------
    if len(active) > 1:
        r = max(active, key=lambda x: x.date_start)
        ids = [x.regulation_id for x in active]
        log.warning(
            "Solapamiento de fechas detectado entre %s. "
            "El CI debería haber bloqueado esto. "
            "Eligiendo la más reciente: %s.",
            ids,
            r.regulation_id,
        )
        return ActiveRegulation(
            regulation_id=r.regulation_id,
            config=r,
            state="active",
            reason=(
                f"Solapamiento entre {ids}. "
                f"Elegida por date_start más reciente."
            ),
        )

    # ------------------------------------------------------------------
    # ESCENARIO B: ventana de transición
    # ------------------------------------------------------------------
    candidates: list[RegulationConfig] = []

    for r in regs:
        window_start = r.date_start - timedelta(days=r.transition_window_days)
        if window_start <= today < r.date_start:
            candidates.append(r)

    just_ended = [
        r for r in regs
        if r.date_end < today <= r.date_end + timedelta(days=7)
    ]
    candidates.extend(just_ended)

    # Deduplicar preservando orden de aparición
    seen: set[str] = set()
    unique_candidates: list[RegulationConfig] = []
    for r in candidates:
        if r.regulation_id not in seen:
            seen.add(r.regulation_id)
            unique_candidates.append(r)

    if unique_candidates:
        ranked = sorted(
            unique_candidates,
            key=lambda r: (
                _count_recent_tournaments(r.regulation_id),
                r.date_start,
            ),
            reverse=True,
        )
        winner = ranked[0]
        candidate_ids = [r.regulation_id for r in unique_candidates]
        return ActiveRegulation(
            regulation_id=winner.regulation_id,
            config=winner,
            state="transition",
            reason=(
                f"Ventana de transición. Candidatas: "
                f"{candidate_ids}. Elegida por actividad "
                f"reciente en Limitless: "
                f"'{winner.regulation_id}'."
            ),
        )

    # ------------------------------------------------------------------
    # ESCENARIO C: sin regulación activa
    # ------------------------------------------------------------------
    most_recent = max(regs, key=lambda r: r.date_end)
    return ActiveRegulation(
        regulation_id=most_recent.regulation_id,
        config=most_recent,
        state="no_active",
        reason=(
            f"Sin regulación vigente el {today}. "
            f"Default: '{most_recent.regulation_id}' "
            f"(más reciente por date_end)."
        ),
    )


__all__ = [
    "ActiveRegulation",
    "RegulationState",
    "get_active_regulation",
]
