"""
Script one-shot para generar data/pokemon_master.json.

Descarga datos de todos los Pokémon de Gen 1-9 (dex IDs 1-1025) desde
PokeAPI y construye el catálogo estático que usa el CI para validar que
los dex IDs en regulations/*.json son válidos.

Cuándo volver a ejecutarlo:
  - Cuando salga una nueva generación de Pokémon con nuevos dex IDs.
  - Cuando PokeAPI actualice datos de stats o abilities relevantes.

Limitaciones conocidas:
  - NO cubre Megas de Legends: Z-A (PokeAPI aún no las tiene).
    Esas formas se validan por nombre directamente en el JSON de regulación.
  - Cubre únicamente formas base (dex IDs 1-1025).

Uso:
  python scripts/fetch_pokemon_master.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))

POKEAPI_BASE = "https://pokeapi.co/api/v2"
OUTPUT_PATH = Path("data/pokemon_master.json")
MAX_DEX_ID = 1025
REQUEST_DELAY_SEC = 0.3

CANONICAL_STATS = {
    "hp",
    "attack",
    "defense",
    "special-attack",
    "special-defense",
    "speed",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def fetch_pokemon(client: httpx.Client, dex_id: int) -> dict[str, object]:
    """
    Descarga datos de un Pokémon desde PokeAPI.
    Reintentos automáticos con backoff exponencial (hasta 3 intentos).

    Args:
        client: Cliente httpx reutilizable con timeout y User-Agent.
        dex_id: Número de Pokédex nacional (1-1025).

    Returns:
        Diccionario con dex_id, name, types, base_stats y abilities.
    """
    response = client.get(f"{POKEAPI_BASE}/pokemon/{dex_id}")
    response.raise_for_status()
    data = response.json()

    types: list[str] = [t["type"]["name"] for t in data["types"]]

    base_stats: dict[str, int] = {
        s["stat"]["name"]: s["base_stat"]
        for s in data["stats"]
        if s["stat"]["name"] in CANONICAL_STATS
    }

    abilities: list[str] = [a["ability"]["name"] for a in data["abilities"]]

    return {
        "dex_id": dex_id,
        "name": data["name"],
        "types": types,
        "base_stats": base_stats,
        "abilities": abilities,
    }


def main() -> int:
    """
    Descarga datos de todos los Pokémon Gen 1-9 (dex IDs 1 a MAX_DEX_ID
    inclusive) y genera data/pokemon_master.json.

    Returns:
        0 si todos los Pokémon se descargaron correctamente,
        1 si alguno fue omitido por error de red.
    """
    pokemon_data: dict[str, object] = {}
    failures: list[int] = []

    headers = {"User-Agent": "vgc-champions-analyzer/0.1 pokemon-master-fetcher"}

    with httpx.Client(timeout=10.0, headers=headers) as client:
        for dex_id in range(1, MAX_DEX_ID + 1):
            if dex_id % 50 == 0:
                print(f"Descargando... {dex_id}/{MAX_DEX_ID}")

            try:
                entry = fetch_pokemon(client, dex_id)
                pokemon_data[str(dex_id)] = entry
                time.sleep(REQUEST_DELAY_SEC)
            except Exception:  # noqa: BLE001
                print(f"⚠ dex_id {dex_id} falló, se omite")
                failures.append(dex_id)

    output: dict[str, object] = {
        "metadata": {
            "total": len(pokemon_data),
            "max_dex_id": MAX_DEX_ID,
            "generated_at": date.today().isoformat(),
            "source": "https://pokeapi.co",
        },
        "pokemon": pokemon_data,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"✓ {len(pokemon_data)} Pokémon guardados en {OUTPUT_PATH}")
    if failures:
        print(f"⚠ {len(failures)} omitidos por error: {failures}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
