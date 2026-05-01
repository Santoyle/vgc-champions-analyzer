"""
Detector de alerta temprana para nuevas regulaciones de formato Showdown.

Este módulo identifica format slugs nuevos que no están cubiertos por los
JSONs existentes en `regulations/`, y los confirma cruzando múltiples fuentes:
Showdown (primaria), Limitless y RK9.

La política de confirmación usa MIN_SOURCES_FOR_DETECTION=2 para reducir falsos
positivos: un slug candidato debe aparecer en al menos dos fuentes distintas
antes de reportarse como una nueva regulación.

La salida de este detector activa el flujo de la Tarea 107 (PR bot) para crear
automáticamente el borrador de nueva regulación.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_REGULATIONS_DIR = _PROJECT_ROOT / "regulations"

# Prefijos conocidos de formato gen9/gen10 Champions/VGC
KNOWN_FORMAT_PREFIXES: list[str] = [
    "gen9vgc",
    "gen9champions",
    "gen10vgc",
    "gen10champions",
]

# Mínimo de fuentes para confirmar detección
MIN_SOURCES_FOR_DETECTION = 2


@dataclass
class NewRegDetection:
    """
    Resultado de una detección de nueva regulación.

    Attributes:
        format_slug: Format slug detectado
                     (ej: "gen9championsbssregmb").
        sources: Lista de fuentes donde se detectó
                 (ej: ["showdown", "limitless"]).
        confidence: Número de fuentes donde apareció.
                    >= MIN_SOURCES_FOR_DETECTION para
                    considerarse válido.
        sample_data: Dict con datos de muestra de
                     cada fuente para referencia.
    """

    format_slug: str
    sources: list[str] = field(default_factory=list)
    confidence: int = 0
    sample_data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_confirmed(self) -> bool:
        """True si el formato apareció en suficientes fuentes."""
        return self.confidence >= MIN_SOURCES_FOR_DETECTION


def _get_known_format_slugs() -> set[str]:
    """
    Lee todos los JSONs de regulations/ y extrae slugs conocidos.

    Returns:
        Set de format slugs conocidos.
        Set vacío si no hay JSONs o hay error.
    """
    known: set[str] = set()

    if not _REGULATIONS_DIR.exists():
        log.warning("Directorio regulations/ no encontrado")
        return known

    for json_file in _REGULATIONS_DIR.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            slug = (
                data.get("battle_format", {}).get("format_slug", "")
                or data.get("format_slug", "")
                or data.get("battle_format_slug", "")
            )
            if slug:
                known.add(str(slug).lower())
        except Exception as exc:  # noqa: BLE001
            log.debug("Error leyendo %s: %s", json_file.name, exc)
            continue

    try:
        from src.app.data.scrapers.smogon_chaos import REGULATION_TO_FORMAT

        for slug in REGULATION_TO_FORMAT.values():
            known.add(str(slug).lower())
    except Exception:  # noqa: BLE001
        pass

    log.debug("Format slugs conocidos: %s", known)
    return known


def _generate_format_candidates(known_slugs: set[str]) -> list[str]:
    """
    Genera candidatos de format slug para probar en Showdown.

    Basado en patrones previos:
    - gen9vgc2025regi -> gen9vgc2025regj
    - gen9championsbssregma -> gen9championsbssregmb
    """
    candidates: list[str] = []

    for slug in known_slugs:
        # Patrón suffix letra al final (ej. ...regma -> ...regmb)
        match = re.search(r"(reg[a-z]+)([a-z])$", slug)
        if match:
            prefix = slug[: match.start(2)]
            last_char = match.group(2)
            next_char = chr(ord(last_char) + 1)
            if next_char <= "z":
                cand = prefix + next_char
                if cand not in known_slugs and any(
                    cand.startswith(pref) for pref in KNOWN_FORMAT_PREFIXES
                ):
                    candidates.append(cand)

        # Patrón con año (ej. ...2025regi -> ...2025regj)
        match2 = re.search(r"(\d{4}reg[a-z]+)([a-z])$", slug)
        if match2:
            prefix2 = slug[: match2.start(2)]
            last2 = match2.group(2)
            next2 = chr(ord(last2) + 1)
            if next2 <= "z":
                cand2 = prefix2 + next2
                if cand2 not in known_slugs and any(
                    cand2.startswith(pref) for pref in KNOWN_FORMAT_PREFIXES
                ):
                    candidates.append(cand2)

    return sorted(set(candidates))


def _check_showdown(known_slugs: set[str]) -> dict[str, NewRegDetection]:
    """
    Busca nuevos formatos en Pokémon Showdown.

    Args:
        known_slugs: Format slugs ya conocidos.

    Returns:
        Dict {format_slug: NewRegDetection} con candidatos detectados.
    """
    import httpx

    detections: dict[str, NewRegDetection] = {}
    base_url = "https://replay.pokemonshowdown.com/search.json"
    candidates = _generate_format_candidates(known_slugs)

    for candidate in candidates:
        if candidate.lower() in known_slugs:
            continue
        try:
            r = httpx.get(
                base_url,
                params={"format": candidate, "page": 1},
                timeout=10.0,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            battles = data if isinstance(data, list) else data.get("battles", [])
            if battles:
                log.info(
                    "Showdown: formato nuevo detectado '%s' (%d batallas)",
                    candidate,
                    len(battles),
                )
                detections[candidate] = NewRegDetection(
                    format_slug=candidate,
                    sources=["showdown"],
                    confidence=1,
                    sample_data={"showdown_battles": len(battles)},
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("Error chequeando Showdown %s: %s", candidate, exc)
            continue

    return detections


def _check_limitless(
    known_slugs: set[str],
    detections: dict[str, NewRegDetection],
) -> None:
    """
    Verifica torneos recientes en Limitless para confirmar candidatos.

    Modifica `detections` in-place añadiendo source "limitless" cuando aplica.
    """
    _ = known_slugs
    try:
        from src.app.data.scrapers.limitless import fetch_recent_tournaments

        tournaments = fetch_recent_tournaments("DETECT", days=14)

        for t in tournaments:
            name_lower = t.name.lower()
            keywords = [
                "regulation",
                "reg m-b",
                "reg mb",
                "champions 2",
                "season 2",
                "new season",
                "nueva temporada",
            ]
            if any(kw in name_lower for kw in keywords):
                log.info(
                    "Limitless: torneo con posible nueva reg detectado: '%s'",
                    t.name,
                )
                for det in detections.values():
                    if "limitless" not in det.sources:
                        det.sources.append("limitless")
                        det.confidence += 1
                        det.sample_data["limitless_tournament"] = t.name
                        break
    except Exception as exc:  # noqa: BLE001
        log.debug("Error chequeando Limitless: %s", exc)


def _check_rk9(
    known_slugs: set[str],
    detections: dict[str, NewRegDetection],
) -> None:
    """
    Verifica eventos RK9/pokedata para confirmar candidatos.

    Modifica `detections` in-place.
    """
    _ = known_slugs
    try:
        from src.app.data.scrapers.rk9_pokedata import fetch_official_events

        events = fetch_official_events("DETECT")

        for event in events:
            name_lower = event.name.lower()
            keywords = [
                "2026",
                "2027",
                "new reg",
                "regulation b",
                "reg b",
                "season 2",
            ]
            if any(kw in name_lower for kw in keywords):
                log.info("RK9: evento con posible nueva reg: '%s'", event.name)
                for det in detections.values():
                    if "rk9" not in det.sources:
                        det.sources.append("rk9")
                        det.confidence += 1
                        det.sample_data["rk9_event"] = event.name
                        break
    except Exception as exc:  # noqa: BLE001
        log.debug("Error chequeando RK9: %s", exc)


def detect_new_regulation() -> list[NewRegDetection]:
    """
    Ejecuta la detección completa de nuevas regulaciones.

    Flujo:
    1. Cargar slugs conocidos de regulations/
    2. Chequear Showdown (primaria)
    3. Confirmar con Limitless y RK9
    4. Retornar solo detecciones confirmadas

    Returns:
        Lista de detecciones confirmadas.
        Lista vacía si no hay nuevas regulaciones.
    """
    try:
        log.info("Iniciando detección de nueva regulación")
        known_slugs = _get_known_format_slugs()
        log.info("Slugs conocidos: %d — %s", len(known_slugs), known_slugs)

        detections = _check_showdown(known_slugs)
        if detections:
            _check_limitless(known_slugs, detections)
            _check_rk9(known_slugs, detections)

        confirmed = [d for d in detections.values() if d.is_confirmed]
        if confirmed:
            log.info(
                "🆕 Nueva(s) regulación(es) detectada(s): %s",
                [d.format_slug for d in confirmed],
            )
        else:
            log.info("No se detectaron nuevas regulaciones.")
        return confirmed
    except Exception as exc:  # noqa: BLE001
        log.warning("Error global en detector de regulación: %s", exc)
        return []


def format_detection_for_discord(detection: NewRegDetection) -> str:
    """
    Formatea una detección como mensaje de Discord.
    """
    sources_str = " + ".join(detection.sources)
    return (
        "🆕 **Nueva regulación detectada**\n"
        f"• Format: `{detection.format_slug}`\n"
        f"• Fuentes: {sources_str} "
        f"({detection.confidence}/{MIN_SOURCES_FOR_DETECTION})\n"
        f"• Acción: crear `regulations/{detection.format_slug}.json` "
        "y completar `pokemon_legales` y fechas"
    )


__all__ = [
    "NewRegDetection",
    "detect_new_regulation",
    "format_detection_for_discord",
    "MIN_SOURCES_FOR_DETECTION",
]
