"""
Extracción de features para el modelo de Win Probability (WP) en VGC.

El modelo WP predice la probabilidad de victoria del jugador p1 dado el
estado actual de la batalla. Para el MVP (v1) se usan dos niveles de features:

  Nivel 1 — Features de equipo (pre-batalla):
    Composición de tipos, roles clave (Fake Out, Trick Room, Intimidate,
    redirección, Tailwind, control de velocidad), tamaño del equipo.

  Nivel 3 — Features de metadatos:
    Rating normalizado de la batalla, índice de mes desde el inicio de
    la regulación (para capturar evolución del meta).

El Nivel 2 (estado turno a turno: HP%, KOs, Mega, Tera) se añadirá en V2
cuando haya suficientes replays completos disponibles para un modelo más rico.

La función principal es extract_features(), que acepta cualquier objeto con
atributos de ParsedReplay (duck-typing para testabilidad sin importar el
scraper). features_to_dataframe() convierte la lista de ReplayFeatures a un
DataFrame listo para entrenar XGBoost.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

ALL_TYPES: list[str] = [
    "Normal", "Fire", "Water", "Electric", "Grass",
    "Ice", "Fighting", "Poison", "Ground", "Flying",
    "Psychic", "Bug", "Rock", "Ghost", "Dragon",
    "Dark", "Steel", "Fairy",
]

KEY_ROLES: list[str] = [
    "fake_out",
    "trick_room",
    "intimidate",
    "follow_me",
    "rage_powder",
    "tailwind",
    "icy_wind",
    "electroweb",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReplayFeatures:
    """
    Features extraídas de un replay para el modelo WP.

    Todos los valores son numéricos (float) para compatibilidad directa con
    XGBoost. Los campos de type_coverage tienen longitud 18 (un float por tipo).

    Attributes:
        replay_id: ID del replay de origen.
        regulation_id: Regulación del replay.
        label: 1.0 si p1 ganó, 0.0 si p2 ganó, None si no determinado.
        rating_norm: Rating normalizado al rango [0, 1] (base 1500-2000).
        month_idx: Meses desde el inicio de la regulación (0-based).
        p1_type_coverage: 18 floats con fracción de tipos en equipo p1.
        p2_type_coverage: 18 floats con fracción de tipos en equipo p2.
        p1_has_fake_out: 1.0 si p1 usó Fake Out en el replay.
        p2_has_fake_out: 1.0 si p2 usó Fake Out en el replay.
        p1_has_trick_room: 1.0 si p1 usó Trick Room.
        p2_has_trick_room: 1.0 si p2 usó Trick Room.
        p1_has_intimidate: 1.0 si p1 activó Intimidate.
        p2_has_intimidate: 1.0 si p2 activó Intimidate.
        p1_has_redirection: 1.0 si p1 usó Follow Me o Rage Powder.
        p2_has_redirection: 1.0 si p2 usó redirección.
        p1_has_tailwind: 1.0 si p1 usó Tailwind.
        p2_has_tailwind: 1.0 si p2 usó Tailwind.
        p1_speed_control: 1.0 si p1 usó Icy Wind o Electroweb.
        p2_speed_control: 1.0 si p2 usó control de velocidad.
        p1_team_size: Número de Pokémon en equipo p1 (float).
        p2_team_size: Número de Pokémon en equipo p2 (float).
    """

    replay_id: str
    regulation_id: str
    label: float | None
    rating_norm: float
    month_idx: int
    p1_type_coverage: list[float]
    p2_type_coverage: list[float]
    p1_has_fake_out: float
    p2_has_fake_out: float
    p1_has_trick_room: float
    p2_has_trick_room: float
    p1_has_intimidate: float
    p2_has_intimidate: float
    p1_has_redirection: float
    p2_has_redirection: float
    p1_has_tailwind: float
    p2_has_tailwind: float
    p1_speed_control: float
    p2_speed_control: float
    p1_team_size: float
    p2_team_size: float


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _normalize_rating(
    rating: int,
    min_rating: float = 1500.0,
    max_rating: float = 2000.0,
) -> float:
    """
    Normaliza el rating a [0, 1]. Valores fuera del rango se clampean.

    Args:
        rating: Rating entero de la batalla.
        min_rating: Rating mínimo esperado (default 1500).
        max_rating: Rating máximo esperado (default 2000).

    Returns:
        Float en [0.0, 1.0]. 1500 → 0.0, 2000 → 1.0.
    """
    clamped = max(min_rating, min(float(rating), max_rating))
    return (clamped - min_rating) / (max_rating - min_rating)


def _extract_roles_from_log(
    raw_log: str,
    player: str,
) -> dict[str, float]:
    """
    Extrae roles clave detectados en el battle log para un jugador.

    Busca patrones case-insensitive en las líneas del log que indican uso
    de moves o activación de habilidades clave. La extracción es best-effort:
    si el formato del log cambia, los roles quedan en 0.0 sin crash.

    Args:
        raw_log: Log completo del replay en formato Showdown.
        player: "p1" o "p2".

    Returns:
        Dict {role_name: float} con 1.0 para roles detectados, 0.0 para el resto.
    """
    roles: dict[str, float] = {role: 0.0 for role in KEY_ROLES}

    if not raw_log:
        return roles

    log_lower = raw_log.lower()
    player_lower = player.lower()
    move_prefix = f"|move|{player_lower}"

    # Fake Out
    if "fake out" in log_lower:
        for line in raw_log.splitlines():
            if "fake out" in line.lower() and move_prefix in line.lower():
                roles["fake_out"] = 1.0
                break

    # Trick Room
    if "trick room" in log_lower:
        for line in raw_log.splitlines():
            if "trick room" in line.lower() and move_prefix in line.lower():
                roles["trick_room"] = 1.0
                break

    # Intimidate (habilidad — aparece como |-ability|p1a: ...)
    if "intimidate" in log_lower:
        for line in raw_log.splitlines():
            if "intimidate" in line.lower() and f"|{player_lower}" in line.lower():
                roles["intimidate"] = 1.0
                break

    # Follow Me
    if "follow me" in log_lower:
        for line in raw_log.splitlines():
            if "follow me" in line.lower() and move_prefix in line.lower():
                roles["follow_me"] = 1.0
                break

    # Rage Powder
    if "rage powder" in log_lower:
        for line in raw_log.splitlines():
            if "rage powder" in line.lower() and move_prefix in line.lower():
                roles["rage_powder"] = 1.0
                break

    # Tailwind
    if "tailwind" in log_lower:
        for line in raw_log.splitlines():
            if "tailwind" in line.lower() and move_prefix in line.lower():
                roles["tailwind"] = 1.0
                break

    # Icy Wind
    if "icy wind" in log_lower:
        for line in raw_log.splitlines():
            if "icy wind" in line.lower() and move_prefix in line.lower():
                roles["icy_wind"] = 1.0
                break

    # Electroweb
    if "electroweb" in log_lower:
        for line in raw_log.splitlines():
            if "electroweb" in line.lower() and move_prefix in line.lower():
                roles["electroweb"] = 1.0
                break

    return roles


def _extract_type_coverage(
    team: list[str],
    pokemon_data: dict[str, dict[str, Any]],
) -> list[float]:
    """
    Calcula la cobertura de tipos de un equipo como fracción normalizada.

    Para cada uno de los 18 tipos, calcula la fracción de Pokémon del equipo
    que tiene ese tipo. Normalizar por tamaño del equipo hace que equipos de
    distintos tamaños sean directamente comparables.

    Args:
        team: Lista de nombres de Pokémon (capitalización libre).
        pokemon_data: Dict {name_lower: {types: [str, ...]}} de datos maestros.
                      Puede ser vacío.

    Returns:
        Lista de 18 floats en [0.0, 1.0]. Todos ceros si team está vacío
        o pokemon_data está vacío.
    """
    coverage = [0.0] * 18
    type_idx = {t: i for i, t in enumerate(ALL_TYPES)}

    if not team or not pokemon_data:
        return coverage

    for species in team:
        data = pokemon_data.get(species.lower(), {})
        for t in data.get("types", []):
            idx = type_idx.get(str(t))
            if idx is not None:
                coverage[idx] += 1.0

    n = len(team)
    if n > 0:
        coverage = [c / n for c in coverage]

    return coverage


def _month_idx_from_upload_time(
    upload_time: int,
    regulation_start_month: str = "2026-04",
) -> int:
    """
    Calcula el índice de mes desde el inicio de la regulación.

    Args:
        upload_time: Timestamp Unix del replay.
        regulation_start_month: Primer mes de la regulación (YYYY-MM).

    Returns:
        Índice entero >= 0. 0 = primer mes de la regulación.
        Retorna 0 si el cálculo falla.
    """
    try:
        dt = datetime.fromtimestamp(upload_time)
        start_year, start_month = map(int, regulation_start_month.split("-"))
        month_diff = (dt.year - start_year) * 12 + (dt.month - start_month)
        return max(0, month_diff)
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------


def extract_features(
    replay: Any,
    pokemon_data: dict[str, dict[str, Any]] | None = None,
    regulation_start_month: str = "2026-04",
) -> ReplayFeatures | None:
    """
    Extrae features de un replay para el modelo WP.

    Acepta cualquier objeto con los atributos de ParsedReplay (duck-typing).
    La extracción es best-effort: campos ausentes se tratan como valores
    por defecto seguros.

    Args:
        replay: Objeto con atributos: replay_id, regulation_id, p1, p2,
                winner, rating, upload_time, team_p1, team_p2, raw_log.
        pokemon_data: Dict de datos maestros para extracción de tipos.
                      None = type_coverage de ceros.
        regulation_start_month: Primer mes de la regulación (YYYY-MM).

    Returns:
        ReplayFeatures con todas las features numéricas.
        None si p1 o p2 están vacíos (replay inválido).
    """
    try:
        replay_id = str(getattr(replay, "replay_id", ""))
        regulation_id = str(getattr(replay, "regulation_id", ""))
        p1 = str(getattr(replay, "p1", ""))
        p2 = str(getattr(replay, "p2", ""))
        winner = getattr(replay, "winner", None)
        rating = int(getattr(replay, "rating", 1500) or 1500)
        upload_time = int(getattr(replay, "upload_time", 0) or 0)
        team_p1: list[str] = list(getattr(replay, "team_p1", []) or [])
        team_p2: list[str] = list(getattr(replay, "team_p2", []) or [])
        raw_log = str(getattr(replay, "raw_log", "") or "")
    except Exception as exc:  # noqa: BLE001
        log.debug("Error leyendo atributos de replay: %s", exc)
        return None

    if not p1 or not p2:
        log.debug("Replay %s sin p1/p2 — saltando", replay_id)
        return None

    # Determinar label
    label: float | None = None
    if winner is not None:
        winner_str = str(winner)
        if winner_str == p1:
            label = 1.0
        elif winner_str == p2:
            label = 0.0

    # Features de metadatos
    rating_norm = _normalize_rating(rating)
    month_idx = _month_idx_from_upload_time(upload_time, regulation_start_month)

    # Features de tipo
    pk_data: dict[str, dict[str, Any]] = pokemon_data or {}
    p1_types = _extract_type_coverage(team_p1, pk_data)
    p2_types = _extract_type_coverage(team_p2, pk_data)

    # Features de roles desde el log
    p1_roles = _extract_roles_from_log(raw_log, "p1")
    p2_roles = _extract_roles_from_log(raw_log, "p2")

    return ReplayFeatures(
        replay_id=replay_id,
        regulation_id=regulation_id,
        label=label,
        rating_norm=rating_norm,
        month_idx=month_idx,
        p1_type_coverage=p1_types,
        p2_type_coverage=p2_types,
        p1_has_fake_out=p1_roles["fake_out"],
        p2_has_fake_out=p2_roles["fake_out"],
        p1_has_trick_room=p1_roles["trick_room"],
        p2_has_trick_room=p2_roles["trick_room"],
        p1_has_intimidate=p1_roles["intimidate"],
        p2_has_intimidate=p2_roles["intimidate"],
        p1_has_redirection=max(p1_roles["follow_me"], p1_roles["rage_powder"]),
        p2_has_redirection=max(p2_roles["follow_me"], p2_roles["rage_powder"]),
        p1_has_tailwind=p1_roles["tailwind"],
        p2_has_tailwind=p2_roles["tailwind"],
        p1_speed_control=max(p1_roles["icy_wind"], p1_roles["electroweb"]),
        p2_speed_control=max(p2_roles["icy_wind"], p2_roles["electroweb"]),
        p1_team_size=float(len(team_p1)),
        p2_team_size=float(len(team_p2)),
    )


def features_to_dataframe(
    features_list: list[ReplayFeatures],
) -> pd.DataFrame:
    """
    Convierte lista de ReplayFeatures a DataFrame listo para XGBoost.

    Expande p1_type_coverage y p2_type_coverage a columnas individuales
    (p1_type_Normal, p1_type_Fire, ..., p2_type_Fairy). Las filas con
    label=None se incluyen — son útiles para predicción en producción.

    Args:
        features_list: Lista de ReplayFeatures a convertir.

    Returns:
        DataFrame con una fila por replay. Columnas: replay_id, regulation_id,
        label, rating_norm, month_idx, p1_type_{tipo} × 18, p2_type_{tipo} × 18,
        p1_has_fake_out, ..., p1_team_size, p2_team_size.
        DataFrame vacío si features_list está vacía.
    """
    if not features_list:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for feat in features_list:
        row: dict[str, Any] = {
            "replay_id": feat.replay_id,
            "regulation_id": feat.regulation_id,
            "label": feat.label,
            "rating_norm": feat.rating_norm,
            "month_idx": feat.month_idx,
            "p1_has_fake_out": feat.p1_has_fake_out,
            "p2_has_fake_out": feat.p2_has_fake_out,
            "p1_has_trick_room": feat.p1_has_trick_room,
            "p2_has_trick_room": feat.p2_has_trick_room,
            "p1_has_intimidate": feat.p1_has_intimidate,
            "p2_has_intimidate": feat.p2_has_intimidate,
            "p1_has_redirection": feat.p1_has_redirection,
            "p2_has_redirection": feat.p2_has_redirection,
            "p1_has_tailwind": feat.p1_has_tailwind,
            "p2_has_tailwind": feat.p2_has_tailwind,
            "p1_speed_control": feat.p1_speed_control,
            "p2_speed_control": feat.p2_speed_control,
            "p1_team_size": feat.p1_team_size,
            "p2_team_size": feat.p2_team_size,
        }
        for i, type_name in enumerate(ALL_TYPES):
            row[f"p1_type_{type_name}"] = (
                feat.p1_type_coverage[i] if i < len(feat.p1_type_coverage) else 0.0
            )
            row[f"p2_type_{type_name}"] = (
                feat.p2_type_coverage[i] if i < len(feat.p2_type_coverage) else 0.0
            )
        rows.append(row)

    return pd.DataFrame(rows)


def get_feature_names() -> list[str]:
    """
    Retorna la lista ordenada de nombres de features del modelo WP.

    Excluye replay_id, regulation_id y label (no son features de entrada).
    El orden es el mismo que las columnas de features en features_to_dataframe(),
    lo que garantiza consistencia entre entrenamiento y predicción en XGBoost.

    Returns:
        Lista de strings: 16 features base + 36 features de tipo (18 × 2).
        Total: 52 features.
    """
    base: list[str] = [
        "rating_norm",
        "month_idx",
        "p1_has_fake_out",
        "p2_has_fake_out",
        "p1_has_trick_room",
        "p2_has_trick_room",
        "p1_has_intimidate",
        "p2_has_intimidate",
        "p1_has_redirection",
        "p2_has_redirection",
        "p1_has_tailwind",
        "p2_has_tailwind",
        "p1_speed_control",
        "p2_speed_control",
        "p1_team_size",
        "p2_team_size",
    ]
    type_features: list[str] = [
        f"{side}_type_{t}" for side in ("p1", "p2") for t in ALL_TYPES
    ]
    return base + type_features


__all__ = [
    "ReplayFeatures",
    "extract_features",
    "features_to_dataframe",
    "get_feature_names",
    "ALL_TYPES",
    "KEY_ROLES",
]
