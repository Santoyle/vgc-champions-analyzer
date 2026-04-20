"""
6 checks del CI para validar todos los JSON en regulations/.

Este archivo es ejecutado por el workflow
.github/workflows/validate-regulations.yml en cada Pull Request
que toca regulations/, data/pokemon_master.json o los módulos
de schema y checksum.

Los tests se parametrizan automáticamente sobre todos los archivos
*.json encontrados en regulations/ mediante regulation_params().
Al agregar un nuevo archivo (ej: M-B.json), los 6 checks lo
validarán en el siguiente CI sin modificar este archivo.

Checks implementados:
  1. Pydantic schema válido
  2. Checksum SHA-256 coincide con el contenido
  3. regulation_id == stem del nombre de archivo
  4. Sin solapamiento de fechas entre regulaciones
  5. pokemon_legales ⊆ data/pokemon_master.json
  6. Sanity semántico (items no vacíos, last_verified, mega stones)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.app.core.checksum import verify_checksum
from src.app.core.schema import RegulationConfig

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

REGULATIONS_DIR = Path("regulations")
POKEMON_MASTER_PATH = Path("data/pokemon_master.json")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def all_regulation_paths() -> list[Path]:
    """Todos los archivos JSON en regulations/."""
    paths = sorted(REGULATIONS_DIR.glob("*.json"))
    if not paths:
        pytest.skip("No hay archivos en regulations/")
    return paths


@pytest.fixture(scope="session")
def all_regulations_raw(
    all_regulation_paths: list[Path],
) -> dict[str, dict[str, object]]:
    """
    Diccionario {stem: raw_dict} para todos los JSONs.
    Clave = stem del archivo (ej: "M-A").
    Cargado una sola vez por run de pytest (scope=session).
    """
    result: dict[str, dict[str, object]] = {}
    for path in all_regulation_paths:
        raw: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        result[path.stem] = raw
    return result


@pytest.fixture(scope="session")
def pokemon_master_ids() -> frozenset[int]:
    """
    Set de dex IDs válidos desde pokemon_master.json.
    Cargado una sola vez por run de pytest (scope=session).
    """
    raw: dict[str, object] = json.loads(POKEMON_MASTER_PATH.read_text(encoding="utf-8"))
    pokemon_dict = raw["pokemon"]
    assert isinstance(pokemon_dict, dict)
    return frozenset(int(k) for k in pokemon_dict.keys())


# ---------------------------------------------------------------------------
# Función helper de parametrización
# ---------------------------------------------------------------------------


def regulation_params() -> list[pytest.param]:
    """
    Genera parámetros pytest para cada JSON en regulations/.
    Permite que los 6 checks sean parametrizados con IDs
    legibles (ej: "M-A").

    Si regulations/ está vacío, retorna lista vacía y los
    tests parametrizados se omiten automáticamente.
    """
    paths = sorted(REGULATIONS_DIR.glob("*.json"))
    return [pytest.param(path, id=path.stem) for path in paths]


# ---------------------------------------------------------------------------
# CHECK 1 — Pydantic schema válido
# ---------------------------------------------------------------------------


class TestCheck1PydanticSchema:
    """
    CHECK 1: Cada JSON en regulations/ carga sin errores
    en RegulationConfig.model_validate().
    """

    @pytest.mark.parametrize("reg_path", regulation_params())
    def test_regulation_loads_without_error(self, reg_path: Path) -> None:
        """El JSON carga y pasa validación Pydantic."""
        raw: dict[str, object] = json.loads(reg_path.read_text(encoding="utf-8"))
        config = RegulationConfig.model_validate(raw)
        assert config.regulation_id is not None


# ---------------------------------------------------------------------------
# CHECK 2 — Checksum SHA-256 válido
# ---------------------------------------------------------------------------


class TestCheck2Checksum:
    """
    CHECK 2: El checksum_sha256 almacenado en cada JSON
    coincide con el checksum calculado del contenido.
    Detecta modificaciones manuales sin rehash.
    """

    @pytest.mark.parametrize("reg_path", regulation_params())
    def test_checksum_matches_content(self, reg_path: Path) -> None:
        """El checksum almacenado es válido para el
        contenido actual del archivo."""
        raw: dict[str, object] = json.loads(reg_path.read_text(encoding="utf-8"))
        stored = str(raw.get("checksum_sha256", ""))
        assert stored != "", (
            f"{reg_path.name}: checksum_sha256 está vacío. "
            f"Ejecuta: python scripts/rehash.py {reg_path}"
        )
        assert verify_checksum(raw, stored), (
            f"{reg_path.name}: checksum inválido. "
            f"Ejecuta: python scripts/rehash.py {reg_path}"
        )


# ---------------------------------------------------------------------------
# CHECK 3 — regulation_id coincide con nombre de archivo
# ---------------------------------------------------------------------------


class TestCheck3RegulationIdMatchesFilename:
    """
    CHECK 3: El campo regulation_id dentro del JSON debe
    coincidir exactamente con el stem del archivo.
    Ejemplo: M-A.json debe tener regulation_id == "M-A".
    """

    @pytest.mark.parametrize("reg_path", regulation_params())
    def test_regulation_id_matches_filename(self, reg_path: Path) -> None:
        """regulation_id == stem del archivo JSON."""
        raw: dict[str, object] = json.loads(reg_path.read_text(encoding="utf-8"))
        stored_id = str(raw.get("regulation_id", ""))
        assert stored_id == reg_path.stem, (
            f"{reg_path.name}: regulation_id='{stored_id}' "
            f"no coincide con filename stem='{reg_path.stem}'"
        )


# ---------------------------------------------------------------------------
# CHECK 4 — Sin solapamiento de fechas
# ---------------------------------------------------------------------------


class TestCheck4NoDateOverlap:
    """
    CHECK 4: Ningún par de regulaciones tiene fechas que se
    solapan. Opera sobre el conjunto completo de regulaciones
    — no está parametrizado por archivo individual.
    """

    def test_no_date_overlap_between_regulations(
        self,
        all_regulations_raw: dict[str, dict[str, object]],
    ) -> None:
        """No hay solapamiento de fechas entre ningún par
        de regulaciones en regulations/."""
        configs: list[RegulationConfig] = []
        for raw in all_regulations_raw.values():
            try:
                configs.append(RegulationConfig.model_validate(raw))
            except ValidationError:
                # Si no valida Pydantic, CHECK 1 ya lo reportará
                continue

        configs.sort(key=lambda c: c.date_start)

        conflicts: list[str] = []
        for i in range(len(configs) - 1):
            a = configs[i]
            b = configs[i + 1]
            if b.date_start <= a.date_end:
                conflicts.append(
                    f"{a.regulation_id} "
                    f"({a.date_start}→{a.date_end}) "
                    f"solapa con "
                    f"{b.regulation_id} "
                    f"({b.date_start}→{b.date_end})"
                )

        assert not conflicts, "Solapamiento de fechas detectado:\n" + "\n".join(
            conflicts
        )


# ---------------------------------------------------------------------------
# CHECK 5 — pokemon_legales ⊆ pokemon_master
# ---------------------------------------------------------------------------


class TestCheck5PokemonLegalesInMaster:
    """
    CHECK 5: Todos los dex IDs en pokemon_legales de cada
    regulación existen en data/pokemon_master.json.
    Evita referencias a Pokémon inexistentes o IDs incorrectos.
    """

    @pytest.mark.parametrize("reg_path", regulation_params())
    def test_all_pokemon_ids_exist_in_master(
        self,
        reg_path: Path,
        pokemon_master_ids: frozenset[int],
    ) -> None:
        """Todos los dex IDs de pokemon_legales están
        en pokemon_master.json."""
        raw: dict[str, object] = json.loads(reg_path.read_text(encoding="utf-8"))
        pokemon_legales = raw.get("pokemon_legales", [])
        assert isinstance(pokemon_legales, list)

        unknown_ids = [
            dex_id
            for dex_id in pokemon_legales
            if int(dex_id) not in pokemon_master_ids  # type: ignore[arg-type]
        ]
        assert not unknown_ids, (
            f"{reg_path.name}: dex IDs no encontrados "
            f"en pokemon_master.json: {unknown_ids[:10]}"
            f"{'...' if len(unknown_ids) > 10 else ''}"
        )


# ---------------------------------------------------------------------------
# CHECK 6 — Sanity semántico
# ---------------------------------------------------------------------------


class TestCheck6SemanticSanity:
    """
    CHECK 6: Validaciones semánticas que van más allá del
    schema Pydantic:
      - items_legales no vacío
      - last_verified no es una fecha futura
      - Cada mega_item de mega_evolutions_disponibles aparece
        en items_legales
    """

    @pytest.mark.parametrize("reg_path", regulation_params())
    def test_items_legales_not_empty(self, reg_path: Path) -> None:
        """items_legales tiene al menos 1 ítem."""
        raw: dict[str, object] = json.loads(reg_path.read_text(encoding="utf-8"))
        items = raw.get("items_legales", [])
        assert isinstance(items, list)
        assert len(items) > 0, f"{reg_path.name}: items_legales está vacío"

    @pytest.mark.parametrize("reg_path", regulation_params())
    def test_last_verified_not_future(self, reg_path: Path) -> None:
        """last_verified no es una fecha futura."""
        raw: dict[str, object] = json.loads(reg_path.read_text(encoding="utf-8"))
        last_verified_str = str(raw.get("last_verified", "1970-01-01"))
        last_verified = date.fromisoformat(last_verified_str)
        assert last_verified <= date.today(), (
            f"{reg_path.name}: last_verified " f"({last_verified}) es una fecha futura"
        )
