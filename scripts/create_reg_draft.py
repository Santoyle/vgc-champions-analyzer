"""
Genera un JSON draft para una nueva regulación detectada por el bot.

Este script lo invoca el workflow `.github/workflows/detect-new-reg.yml` para
crear `regulations/{reg_id}.json` con placeholders que un humano debe completar
antes de mergear (`pokemon_legales`, fechas, ítems/moves, checksum vía
`scripts/rehash.py`, etc.).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_REGULATIONS_DIR = _PROJECT_ROOT / "regulations"


def _get_next_regulation_id(format_slug: str) -> str:
    """
    Deriva el regulation_id del format_slug.

    Ejemplos:
    - "gen9championsbssregmb" → "M-B"
    - "gen9vgc2026regn" → "N"
    - "gen9championsbssregmc" → "M-C"

    Busca el patrón "reg" seguido de letras al final del slug y lo convierte
    a mayúsculas con guión si tiene más de una letra.

    Args:
        format_slug: Format slug de Showdown.

    Returns:
        regulation_id derivado.
        "UNKNOWN" si no puede derivarse.
    """
    import re

    match = re.search(r"reg([a-z]+)$", format_slug)
    if not match:
        return "UNKNOWN"

    letters = match.group(1).upper()

    if len(letters) == 1:
        return letters
    if len(letters) == 2:
        return f"{letters[0]}-{letters[1]}"
    return letters


def generate_draft_json(
    format_slug: str,
    regulation_id: str | None = None,
) -> dict:
    """
    Genera un JSON draft para una nueva regulación.

    El draft tiene toda la estructura requerida por RegulationConfig pero con
    valores placeholder que el humano debe completar.

    Args:
        format_slug: Format slug detectado.
        regulation_id: Si None, se deriva del slug.

    Returns:
        Dict con el JSON draft.
    """
    if regulation_id is None:
        regulation_id = _get_next_regulation_id(format_slug)

    today = date.today().isoformat()

    return {
        "regulation_id": regulation_id,
        "game": "pokemon_champions",
        "date_start": "YYYY-MM-DD",
        "date_end": "YYYY-MM-DD",
        "battle_format": {
            "format_slug": format_slug,
            "team_size": 6,
            "bring": 6,
            "pick": 4,
            "level_cap": 50,
            "best_of_swiss": 1,
            "best_of_topcut": 3,
            "team_preview_sec": 90,
            "turn_sec": 45,
            "player_timer_sec": 420,
            "game_timer_sec": 1200,
        },
        "mechanics": {
            "mega_enabled": True,
            "mega_max_per_battle": 1,
            "tera_enabled": True,
            "z_moves_enabled": False,
            "dynamax_enabled": False,
            "stat_points_system": True,
            "stat_points_total": 66,
            "stat_points_cap_per_stat": 32,
            "iv_system": False,
        },
        "clauses": {
            "species_clause": True,
            "item_clause": True,
            "legendary_ban": True,
            "restricted_ban": False,
            "open_team_list": True,
        },
        "pokemon_legales": [],
        "mega_evolutions_disponibles": [],
        "items_legales": [],
        "moves_legales": [],
        "checksum_sha256": "",
        "last_verified": today,
        "schema_version": "1.0.0",
        "source_urls": {
            "official_site": "",
            "smogon_thread": "",
        },
        "transition_window_days": 7,
        "_draft_notes": (
            f"AUTO-GENERATED DRAFT — {today}\n"
            f"Format slug detectado: {format_slug}\n"
            f"TODO: completar pokemon_legales, "
            f"date_start, date_end, items_legales\n"
            f"Luego ejecutar: "
            f"python scripts/rehash.py "
            f"regulations/{regulation_id}.json"
        ),
    }


def main() -> int:
    """
    Entry point del script.

    Uso:
        python scripts/create_reg_draft.py --slug gen9championsbssregmb
            [--reg-id M-B] [--output regulations/M-B.json]
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Genera un JSON draft para una nueva "
            "regulación detectada automáticamente"
        )
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Format slug de Showdown detectado",
    )
    parser.add_argument(
        "--reg-id",
        default=None,
        dest="reg_id",
        help=(
            "regulation_id explícito. Default: derivado del slug."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path de salida. Default: regulations/{reg_id}.json"
        ),
    )
    args = parser.parse_args()

    reg_id = args.reg_id or _get_next_regulation_id(args.slug)

    output_path = Path(args.output) if args.output else (
        _REGULATIONS_DIR / f"{reg_id}.json"
    )

    if output_path.exists():
        log.error(
            "El archivo %s ya existe. No se sobreescribirá.",
            output_path,
        )
        return 1

    draft = generate_draft_json(args.slug, reg_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(draft, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    log.info(
        "Draft creado: %s\n"
        "regulation_id: %s\n"
        "format_slug:   %s\n"
        "Próximos pasos:\n"
        "  1. Editar %s con pokemon_legales y fechas\n"
        "  2. python scripts/rehash.py %s\n"
        "  3. git add %s && git commit",
        output_path,
        reg_id,
        args.slug,
        output_path,
        output_path,
        output_path,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
