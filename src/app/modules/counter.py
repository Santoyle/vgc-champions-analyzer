"""
Heurístico v1 de counters para VGC Champions.

Calcula un score de "efectividad como counter" para cada Pokémon del roster
contra un equipo rival, combinando tres componentes:

  TYPE_ADVANTAGE (peso 0.5):
    Cuántos Pokémon rivales son débiles a los tipos del counter.
    +1.0 por rival débil, -0.5 por rival resistente, -1.0 por inmune.

  SPEED_TIER (peso 0.3):
    Proporción del equipo rival al que el counter supera en velocidad base.
    score = (faster - slower) / total_rival

  ITEM_SYNERGY (peso 0.2):
    Bonus estático por ítems que señalan roles counter reconocibles.

El score total se normaliza al rango [0, 1] dividiendo por el máximo absoluto.

Este módulo es el heurístico v1 — sin ML, basado solo en datos estáticos.
Será reemplazado por WP predict (Counter v2, Tarea 87) cuando haya
suficientes datos de replays. TYPE_CHART contiene la tabla de tipos
completa para los 18 tipos de Pokémon.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.app.modules.wp import (
    ALL_TYPES,
    ReplayFeatures,
    extract_features,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_POKEMON_MASTER_PATH = _PROJECT_ROOT / "data" / "pokemon_master.json"

# ---------------------------------------------------------------------------
# Tablas estáticas
# ---------------------------------------------------------------------------

TYPE_CHART: dict[str, dict[str, float]] = {
    "Normal":   {"Rock": 0.5, "Ghost": 0.0, "Steel": 0.5},
    "Fire":     {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 2.0,
                 "Bug": 2.0, "Rock": 0.5, "Dragon": 0.5, "Steel": 2.0},
    "Water":    {"Fire": 2.0, "Water": 0.5, "Grass": 0.5, "Ground": 2.0,
                 "Rock": 2.0, "Dragon": 0.5},
    "Electric": {"Water": 2.0, "Electric": 0.5, "Grass": 0.5, "Ground": 0.0,
                 "Flying": 2.0, "Dragon": 0.5},
    "Grass":    {"Fire": 0.5, "Water": 2.0, "Grass": 0.5, "Poison": 0.5,
                 "Ground": 2.0, "Flying": 0.5, "Bug": 0.5, "Rock": 2.0,
                 "Dragon": 0.5, "Steel": 0.5},
    "Ice":      {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 0.5,
                 "Ground": 2.0, "Flying": 2.0, "Dragon": 2.0, "Steel": 0.5},
    "Fighting": {"Normal": 2.0, "Ice": 2.0, "Poison": 0.5, "Flying": 0.5,
                 "Psychic": 0.5, "Bug": 0.5, "Rock": 2.0, "Ghost": 0.0,
                 "Dark": 2.0, "Steel": 2.0, "Fairy": 0.5},
    "Poison":   {"Grass": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5,
                 "Ghost": 0.5, "Steel": 0.0, "Fairy": 2.0},
    "Ground":   {"Fire": 2.0, "Electric": 2.0, "Grass": 0.5, "Poison": 2.0,
                 "Flying": 0.0, "Bug": 0.5, "Rock": 2.0, "Steel": 2.0},
    "Flying":   {"Electric": 0.5, "Grass": 2.0, "Fighting": 2.0,
                 "Bug": 2.0, "Rock": 0.5, "Steel": 0.5},
    "Psychic":  {"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5,
                 "Dark": 0.0, "Steel": 0.5},
    "Bug":      {"Fire": 0.5, "Grass": 2.0, "Fighting": 0.5, "Flying": 0.5,
                 "Psychic": 2.0, "Ghost": 0.5, "Dark": 2.0, "Steel": 0.5,
                 "Fairy": 0.5},
    "Rock":     {"Fire": 2.0, "Ice": 2.0, "Fighting": 0.5, "Ground": 0.5,
                 "Flying": 2.0, "Bug": 2.0, "Steel": 0.5},
    "Ghost":    {"Normal": 0.0, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5},
    "Dragon":   {"Dragon": 2.0, "Steel": 0.5, "Fairy": 0.0},
    "Dark":     {"Fighting": 0.5, "Psychic": 2.0, "Ghost": 2.0,
                 "Dark": 0.5, "Fairy": 0.5},
    "Steel":    {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2.0,
                 "Rock": 2.0, "Steel": 0.5, "Fairy": 2.0},
    "Fairy":    {"Fire": 0.5, "Fighting": 2.0, "Poison": 0.5,
                 "Dragon": 2.0, "Dark": 2.0, "Steel": 0.5},
}

ITEM_SYNERGY_SCORES: dict[str, float] = {
    "choice scarf":  0.3,
    "choice band":   0.2,
    "choice specs":  0.2,
    "focus sash":    0.1,
    "life orb":      0.2,
    "assault vest":  0.1,
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class CounterResult:
    """
    Resultado de un Pokémon candidato como counter.

    Attributes:
        species: Nombre de la especie counter.
        score: Score normalizado total (0.0-1.0).
        type_advantage_score: Componente de ventaja de tipo (sin normalizar).
        speed_tier_score: Componente de speed tier.
        item_synergy_score: Componente de ítem.
        types: Tipos del counter.
        counters_directly: Especies rivales que este counter amenaza.
    """

    species: str
    score: float
    type_advantage_score: float
    speed_tier_score: float
    item_synergy_score: float
    types: list[str] = field(default_factory=list)
    counters_directly: list[str] = field(default_factory=list)


@dataclass
class WPCounterResult:
    """
    Resultado de un counter calculado via WP model.

    A diferencia de CounterResult (heurístico), WPCounterResult usa el modelo
    de Win Probability para estimar la ventaja del counter contra el rival.

    Attributes:
        species: Nombre de la especie counter.
        wp_score: Win Probability estimada del counter (p1) contra el rival (p2).
                  Rango [0, 1]. >0.5 = counter favorito.
        heuristic_score: Score heurístico de respaldo (del heurístico v1).
        types: Tipos del counter.
        counters_directly: Rivales amenazados (del heurístico v1).
    """

    species: str
    wp_score: float
    heuristic_score: float
    types: list[str] = field(default_factory=list)
    counters_directly: list[str] = field(default_factory=list)


@dataclass
class TeamMatchupResult:
    """
    Resultado de un equipo completo evaluado como anti-rival.

    Attributes:
        team: Lista de Pokémon del equipo.
        wp_score: Win Probability estimada del equipo contra el rival.
                  None si no hay modelo WP disponible.
        heuristic_score: Score heurístico promedio de los slots del equipo.
        source_replay_id: ID del replay de origen si viene de replays reales.
                          None si es generado.
        regulation_id: Regulación del equipo.
    """

    team: list[str]
    wp_score: float | None
    heuristic_score: float
    source_replay_id: str | None
    regulation_id: str


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _load_pokemon_data() -> dict[str, dict[str, Any]]:
    """
    Carga datos de Pokémon desde pokemon_master.json.

    Returns:
        Dict {nombre_lower: {types, base_stats, ...}}.
        Dict vacío si el archivo no existe o hay error.
    """
    try:
        if not _POKEMON_MASTER_PATH.exists():
            log.warning(
                "pokemon_master.json no encontrado en %s", _POKEMON_MASTER_PATH
            )
            return {}
        raw = json.loads(_POKEMON_MASTER_PATH.read_text(encoding="utf-8"))
        pokemon_data = raw.get("pokemon", {})
        result: dict[str, dict[str, Any]] = {}
        for entry in pokemon_data.values():
            if isinstance(entry, dict):
                name = str(entry.get("name", "")).lower()
                if name:
                    result[name] = entry
        return result
    except Exception as exc:  # noqa: BLE001
        log.warning("Error cargando pokemon_master.json: %s", exc)
        return {}


def _load_recent_teams(
    regulation_id: str,
    max_teams: int = 200,
) -> list[dict[str, Any]]:
    """
    Carga equipos recientes desde los Parquets de Showdown en
    data/raw/reg={id}/source=showdown/.

    Retorna equipos únicos deduplicados por composición de Pokémon.

    Args:
        regulation_id: ID de la regulación.
        max_teams: Máximo de equipos únicos a cargar.

    Returns:
        Lista de dicts con keys: team (list[str]), replay_id (str),
        rating (int). Lista vacía si no hay datos.
    """
    import glob as glob_module
    import json as _json

    import pandas as pd

    raw_pattern = str(
        _PROJECT_ROOT
        / "data"
        / "raw"
        / f"reg={regulation_id}"
        / "source=showdown"
        / "*.parquet"
    )

    parquet_files = glob_module.glob(raw_pattern)
    if not parquet_files:
        log.debug("Sin Parquets de Showdown para %s", regulation_id)
        return []

    teams: list[dict[str, Any]] = []
    seen_compositions: set[str] = set()

    for pq_file in sorted(parquet_files, reverse=True):
        if len(teams) >= max_teams:
            break
        try:
            df = pd.read_parquet(pq_file)
            for _, row in df.iterrows():
                if len(teams) >= max_teams:
                    break
                for team_col in ("team_p1_json", "team_p2_json"):
                    try:
                        team: list[str] = _json.loads(
                            str(row.get(team_col, "[]"))
                        )
                        if not team or len(team) < 3:
                            continue
                        # Deduplicar por composición (set ordenado de nombres)
                        comp_key = "|".join(
                            sorted(str(p).lower() for p in team)
                        )
                        if comp_key in seen_compositions:
                            continue
                        seen_compositions.add(comp_key)
                        teams.append({
                            "team": [str(p) for p in team],
                            "replay_id": str(row.get("replay_id", "")),
                            "rating": int(row.get("rating", 1500) or 1500),
                        })
                    except Exception:  # noqa: BLE001
                        continue
        except Exception as exc:  # noqa: BLE001
            log.debug("Error leyendo %s: %s", pq_file, exc)
            continue

    log.info("Cargados %d equipos únicos para %s", len(teams), regulation_id)
    return teams


def _type_advantage_score(
    attacker_types: list[str],
    rival_slots: list[Any],
    pokemon_data: dict[str, dict[str, Any]],
) -> tuple[float, list[str]]:
    """
    Calcula el score de ventaja de tipo del counter contra el equipo rival.

    Args:
        attacker_types: Tipos del counter.
        rival_slots: Slots del equipo rival (objetos con .species).
        pokemon_data: Datos de Pokémon master.

    Returns:
        Tupla (score_raw, lista_rivales_amenazados).
    """
    score = 0.0
    threatened: list[str] = []

    for slot in rival_slots:
        species = str(getattr(slot, "species", "")).lower()
        rival_data = pokemon_data.get(species, {})
        rival_types = [str(t) for t in rival_data.get("types", [])]

        if not rival_types:
            continue

        best_multiplier = 1.0
        for atk_type in attacker_types:
            multiplier = 1.0
            type_row = TYPE_CHART.get(atk_type, {})
            for def_type in rival_types:
                multiplier *= type_row.get(def_type, 1.0)
            best_multiplier = max(best_multiplier, multiplier)

        if best_multiplier >= 2.0:
            score += 1.0
            threatened.append(str(getattr(slot, "species", "")))
        elif best_multiplier == 0.0:
            score -= 1.0
        elif best_multiplier <= 0.5:
            score -= 0.5

    return score, threatened


def _speed_tier_score(
    counter_spe: int,
    rival_slots: list[Any],
    pokemon_data: dict[str, dict[str, Any]],
) -> float:
    """
    Calcula el score de speed tier comparando el counter contra el rival.

    Args:
        counter_spe: Base speed del counter.
        rival_slots: Slots del equipo rival.
        pokemon_data: Datos de Pokémon master.

    Returns:
        Score entre -1.0 y 1.0.
    """
    if counter_spe == 0 or not rival_slots:
        return 0.0

    faster = 0
    slower = 0

    for slot in rival_slots:
        species = str(getattr(slot, "species", "")).lower()
        rival_data = pokemon_data.get(species, {})
        rival_spe = int(rival_data.get("base_stats", {}).get("speed", 0))
        if rival_spe == 0:
            continue
        if counter_spe > rival_spe:
            faster += 1
        elif counter_spe < rival_spe:
            slower += 1

    total = faster + slower
    if total == 0:
        return 0.0
    return (faster - slower) / total


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------


def heuristic_counter(
    rival_team: Any,
    roster: list[str],
    pokemon_data: dict[str, dict[str, Any]] | None = None,
    top_n: int = 20,
) -> list[CounterResult]:
    """
    Calcula los mejores counters heurísticos para un equipo rival.

    Para cada Pokémon del roster calcula un score combinado de:
      0.5 · type_advantage + 0.3 · speed_tier + 0.2 · item_synergy

    Los scores se normalizan al rango [0, 1] dividiendo por el máximo
    absoluto. Si pokemon_master.json no existe, retorna lista vacía sin crash.

    Args:
        rival_team: ParsedTeam del equipo rival. Debe tener atributo .slots.
        roster: Lista de nombres de Pokémon candidatos a counter.
        pokemon_data: Dict de datos maestros. Si None, lo carga desde disco.
        top_n: Número máximo de counters a retornar.

    Returns:
        Lista de CounterResult ordenada por score descendente.
        Lista vacía si el roster está vacío, el rival no tiene slots,
        o pokemon_master.json no está disponible.
    """
    if not roster:
        log.debug("Roster vacío — sin counters")
        return []

    rival_slots = getattr(rival_team, "slots", [])
    if not rival_slots:
        log.debug("Equipo rival sin slots — sin counters")
        return []

    if pokemon_data is None:
        pokemon_data = _load_pokemon_data()

    if not pokemon_data:
        log.warning(
            "pokemon_master.json vacío o no cargado — no se pueden calcular counters"
        )
        return []

    results: list[CounterResult] = []

    for species_name in roster:
        species_lower = species_name.lower()
        data = pokemon_data.get(species_lower, {})

        types = [str(t) for t in data.get("types", [])]
        base_spe = int(data.get("base_stats", {}).get("speed", 0))

        type_score, threatened = _type_advantage_score(
            types, rival_slots, pokemon_data
        )
        speed_score = _speed_tier_score(base_spe, rival_slots, pokemon_data)
        item_score = 0.0  # v1: sin ítem del roster individual

        raw_score = 0.5 * type_score + 0.3 * speed_score + 0.2 * item_score

        results.append(
            CounterResult(
                species=species_name,
                score=raw_score,
                type_advantage_score=type_score,
                speed_tier_score=speed_score,
                item_synergy_score=item_score,
                types=types,
                counters_directly=threatened,
            )
        )

    if not results:
        return []

    # Normalizar scores al rango [0, 1]
    max_score = max(abs(r.score) for r in results)
    if max_score > 0:
        for r in results:
            r.score = round(r.score / max_score, 4)

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_n]


def wp_counter(
    rival_team: Any,
    roster: list[str],
    regulation_id: str,
    pokemon_data: dict[str, dict[str, Any]] | None = None,
    top_n: int = 15,
) -> list[WPCounterResult]:
    """
    Calcula counters usando el modelo WP para scoring.

    Para cada Pokémon del roster, construye un ReplayFeatures simulado donde
    ese Pokémon es el equipo de p1 (solo ese slot) y el equipo rival es p2.
    Predice la WP con el modelo actual.

    Si el modelo WP no está disponible para la regulación, retorna lista vacía.

    Args:
        rival_team: ParsedTeam del equipo rival.
        roster: Lista de nombres de Pokémon candidatos a counter.
        regulation_id: ID de la regulación activa.
        pokemon_data: Datos maestros de Pokémon. Si None, se carga desde disco.
        top_n: Máximo de counters a retornar.

    Returns:
        Lista de WPCounterResult ordenada por wp_score descendente.
        Lista vacía si el modelo no está disponible.
    """
    try:
        from src.app.modules.wp_train import predict_proba
    except ImportError:
        log.warning("wp_train no disponible — WP counter no soportado")
        return []

    if not roster:
        return []

    rival_slots = getattr(rival_team, "slots", [])
    if not rival_slots:
        return []

    pk_data = pokemon_data or _load_pokemon_data()

    rival_species = [
        str(getattr(slot, "species", ""))
        for slot in rival_slots
        if getattr(slot, "species", "")
    ]
    if not rival_species:
        return []

    # Construir ReplayFeatures para cada counter candidato vs el equipo rival.
    # _MockReplayForWP es local para no contaminar el namespace del módulo.
    features_list: list[ReplayFeatures] = []
    valid_roster: list[str] = []

    for species_name in roster:

        class _MockReplayForWP:
            p1 = "counter_candidate"
            p2 = "rival"
            winner: str | None = None
            rating = 1750
            upload_time = 0
            raw_log = ""

            def __init__(self, counter: str, rival: list[str]) -> None:
                self.replay_id = f"wp-counter-{counter}"
                self.regulation_id = regulation_id
                self.team_p1: list[str] = [counter]
                self.team_p2: list[str] = rival

        mock_replay = _MockReplayForWP(species_name, rival_species)
        feat = extract_features(mock_replay, pokemon_data=pk_data)
        if feat is not None:
            features_list.append(feat)
            valid_roster.append(species_name)

    if not features_list:
        log.debug("Sin features válidas para WP counter")
        return []

    probs = predict_proba(features_list, regulation_id)
    if probs is None:
        log.info(
            "Modelo WP no disponible para %s — sin WP counters", regulation_id
        )
        return []

    # Score heurístico de respaldo para cada counter
    heuristic_results = heuristic_counter(
        rival_team,
        roster=valid_roster,
        pokemon_data=pk_data,
        top_n=len(valid_roster),
    )
    heuristic_map = {r.species: r.score for r in heuristic_results}
    types_map = {r.species: r.types for r in heuristic_results}
    threatens_map = {r.species: r.counters_directly for r in heuristic_results}

    results: list[WPCounterResult] = []
    for i, species_name in enumerate(valid_roster):
        if i >= len(probs):
            break
        results.append(
            WPCounterResult(
                species=species_name,
                wp_score=round(float(probs[i]), 4),
                heuristic_score=heuristic_map.get(species_name, 0.0),
                types=types_map.get(species_name, []),
                counters_directly=threatens_map.get(species_name, []),
            )
        )

    results.sort(key=lambda x: x.wp_score, reverse=True)
    return results[:top_n]


def top_teams_vs_rival(
    rival_team: Any,
    regulation_id: str,
    pokemon_data: dict[str, dict[str, Any]] | None = None,
    max_candidate_teams: int = 200,
    top_n: int = 5,
) -> list[TeamMatchupResult]:
    """
    Busca los mejores equipos completos anti-rival desde replays guardados.

    Estrategia:
      1. Carga equipos únicos de replays recientes de Showdown.
      2. Para cada equipo, calcula score:
         - Con WP model: predict_proba del equipo completo vs rival.
         - Sin WP model: promedio del heurístico v1 por slot.
      3. Retorna top_n ordenados por score.

    Args:
        rival_team: ParsedTeam del equipo rival.
        regulation_id: ID de la regulación activa.
        pokemon_data: Datos maestros de Pokémon. Si None, carga desde disco.
        max_candidate_teams: Máximo de equipos candidatos a evaluar.
        top_n: Número de equipos a retornar.

    Returns:
        Lista de TeamMatchupResult ordenada por score descendente.
        Lista vacía si no hay equipos candidatos o falla cualquier paso.
    """
    rival_slots = getattr(rival_team, "slots", [])
    if not rival_slots:
        return []

    rival_species = [
        str(getattr(slot, "species", ""))
        for slot in rival_slots
        if getattr(slot, "species", "")
    ]
    if not rival_species:
        return []

    pk_data = pokemon_data or _load_pokemon_data()
    try:
        candidate_teams = _load_recent_teams(regulation_id, max_candidate_teams)
    except Exception as exc:  # noqa: BLE001
        log.warning("Error cargando equipos candidatos: %s", exc)
        return []

    if not candidate_teams:
        log.info(
            "Sin equipos candidatos para %s — top_teams_vs_rival retorna vacío",
            regulation_id,
        )
        return []

    # ── Helpers internos para mock de replay/slot (locales, no contaminan módulo) ──

    def _heuristic_score_for_team(
        team_species: list[str],
    ) -> float:
        """Promedio de score heurístico por slot del equipo candidato."""

        class _Slot:
            def __init__(self, sp: str) -> None:
                self.species = sp

        class _Team:
            def __init__(self, sp: str) -> None:
                self.slots = [_Slot(sp)]

        scores: list[float] = []
        for sp in team_species:
            h = heuristic_counter(
                _Team(sp),
                roster=[sp],
                pokemon_data=pk_data,
                top_n=1,
            )
            if h:
                scores.append(h[0].score)
        return sum(scores) / len(scores) if scores else 0.0

    # ── Intentar usar WP model ──

    wp_available = False
    try:
        from src.app.modules.wp_train import load_model, predict_proba

        if load_model(regulation_id) is not None:
            wp_available = True
    except ImportError:
        pass

    results: list[TeamMatchupResult] = []

    if wp_available:
        from src.app.modules.wp import extract_features as _extract

        features_list: list[ReplayFeatures] = []
        valid_indices: list[int] = []

        for i, cand in enumerate(candidate_teams):

            class _MockTeamReplay:
                p1 = "candidate"
                p2 = "rival"
                winner: str | None = None
                upload_time = 0
                raw_log = ""

                def __init__(
                    self, idx: int, team: list[str], rival: list[str], rat: int
                ) -> None:
                    self.replay_id = f"team-{idx}"
                    self.regulation_id = regulation_id
                    self.rating = rat
                    self.team_p1 = team
                    self.team_p2 = rival

            mock = _MockTeamReplay(i, cand["team"], rival_species, cand["rating"])
            feat = _extract(mock, pokemon_data=pk_data)
            if feat is not None:
                features_list.append(feat)
                valid_indices.append(i)

        if features_list:
            from src.app.modules.wp_train import predict_proba as _predict

            probs = _predict(features_list, regulation_id)
            if probs is not None:
                for j, idx in enumerate(valid_indices):
                    if j >= len(probs):
                        break
                    cand = candidate_teams[idx]
                    results.append(
                        TeamMatchupResult(
                            team=cand["team"],
                            wp_score=round(float(probs[j]), 4),
                            heuristic_score=_heuristic_score_for_team(cand["team"]),
                            source_replay_id=cand.get("replay_id") or None,
                            regulation_id=regulation_id,
                        )
                    )
                results.sort(key=lambda x: x.wp_score or 0.0, reverse=True)
                return results[:top_n]

    # ── Fallback: heurístico promedio por slot ──

    log.info("WP no disponible — usando heurístico para top_teams_vs_rival")
    for cand in candidate_teams:
        results.append(
            TeamMatchupResult(
                team=cand["team"],
                wp_score=None,
                heuristic_score=_heuristic_score_for_team(cand["team"]),
                source_replay_id=cand.get("replay_id") or None,
                regulation_id=regulation_id,
            )
        )
    results.sort(key=lambda x: x.heuristic_score, reverse=True)
    return results[:top_n]


__all__ = [
    "CounterResult",
    "WPCounterResult",
    "TeamMatchupResult",
    "heuristic_counter",
    "wp_counter",
    "top_teams_vs_rival",
    "TYPE_CHART",
    "ITEM_SYNERGY_SCORES",
]
