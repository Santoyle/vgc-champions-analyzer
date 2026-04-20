"""
Tests de RegulationConfig y el sistema de checksum SHA-256.

Cubre tres áreas:
  - TestLoadMA: carga y validación del JSON de Reg M-A.
  - TestChecksum: integridad, determinismo y manipulación del hash.
  - TestRegulationConfigValidators: validators de Pydantic y reglas
    de negocio definidas en RegulationConfig.
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.app.core.checksum import (
    EXCLUDED_FROM_HASH,
    compute_checksum,
    rehash_dict,
    verify_checksum,
)
from src.app.core.schema import (
    RegulationConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ma_json_path() -> Path:
    """Path al JSON de Reg M-A en regulations/."""
    return Path("regulations/M-A.json")


@pytest.fixture(scope="session")
def ma_raw(ma_json_path: Path) -> dict[str, object]:
    """Diccionario crudo cargado desde M-A.json."""
    return json.loads(ma_json_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def ma_config(ma_raw: dict[str, object]) -> RegulationConfig:
    """RegulationConfig validado desde M-A.json."""
    return RegulationConfig.model_validate(ma_raw)


# ---------------------------------------------------------------------------
# Grupo 1 — Carga y validación de M-A.json
# ---------------------------------------------------------------------------


class TestLoadMA:
    def test_load_ma_json_succeeds(self, ma_config: RegulationConfig) -> None:
        """M-A.json carga sin errores de validación."""
        assert ma_config.regulation_id == "M-A"
        assert ma_config.game == "pokemon_champions"

    def test_pokemon_legales_count(self, ma_config: RegulationConfig) -> None:
        """Reg M-A tiene exactamente 186 Pokémon legales."""
        assert len(ma_config.pokemon_legales) == 186

    def test_pokemon_legales_no_duplicates(self, ma_config: RegulationConfig) -> None:
        """No hay dex IDs duplicados en pokemon_legales."""
        assert len(set(ma_config.pokemon_legales)) == 186

    def test_mega_evolutions_count(self, ma_config: RegulationConfig) -> None:
        """Reg M-A tiene 60 Mega Evoluciones disponibles."""
        assert len(ma_config.mega_evolutions_disponibles) == 60

    def test_items_legales_count(self, ma_config: RegulationConfig) -> None:
        """Reg M-A tiene exactamente 117 ítems legales."""
        assert len(ma_config.items_legales) == 117

    def test_mega_enabled_true(self, ma_config: RegulationConfig) -> None:
        """Reg M-A tiene Mega Evolución habilitada."""
        assert ma_config.mechanics.mega_enabled is True

    def test_tera_enabled_true(self, ma_config: RegulationConfig) -> None:
        """Reg M-A tiene Tera habilitado."""
        assert ma_config.mechanics.tera_enabled is True

    def test_stat_points_system_true(self, ma_config: RegulationConfig) -> None:
        """Reg M-A usa el sistema de Stat Points."""
        assert ma_config.mechanics.stat_points_system is True

    def test_stat_points_total(self, ma_config: RegulationConfig) -> None:
        """Reg M-A tiene 66 Stat Points totales."""
        assert ma_config.mechanics.stat_points_total == 66

    def test_iv_system_false(self, ma_config: RegulationConfig) -> None:
        """Reg M-A no usa el sistema de IVs."""
        assert ma_config.mechanics.iv_system is False

    def test_date_range(self, ma_config: RegulationConfig) -> None:
        """Reg M-A corre del 8 abr al 17 jun 2026."""
        assert ma_config.date_start == date(2026, 4, 8)
        assert ma_config.date_end == date(2026, 6, 17)

    def test_transition_window_days(self, ma_config: RegulationConfig) -> None:
        """transition_window_days es 7 (default sensato)."""
        assert ma_config.transition_window_days == 7

    def test_frozen_config_immutable(self, ma_config: RegulationConfig) -> None:
        """RegulationConfig es inmutable (frozen=True)."""
        with pytest.raises((TypeError, ValidationError)):
            ma_config.regulation_id = "MUTATED"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Grupo 2 — Checksum
# ---------------------------------------------------------------------------


class TestChecksum:
    def test_checksum_verify_true(self, ma_raw: dict[str, object]) -> None:
        """El checksum almacenado en M-A.json es válido."""
        stored = str(ma_raw["checksum_sha256"])
        assert verify_checksum(ma_raw, stored) is True

    def test_checksum_known_value(self, ma_raw: dict[str, object]) -> None:
        """El checksum de M-A.json coincide con el valor
        calculado en la Tarea 17."""
        expected = "13f33fc4a157aa85f9bb7bf786341ea780f6e1f5" "feecb71aa99ecea3e9ae4c54"
        assert str(ma_raw["checksum_sha256"]) == expected

    def test_checksum_tampering_detected(self, ma_raw: dict[str, object]) -> None:
        """Modificar pokemon_legales invalida el checksum."""
        tampered = copy.deepcopy(ma_raw)
        pokemon = list(tampered["pokemon_legales"])  # type: ignore[arg-type]
        pokemon.append(9999)
        tampered["pokemon_legales"] = pokemon
        stored = str(ma_raw["checksum_sha256"])
        assert verify_checksum(tampered, stored) is False

    def test_checksum_excluded_fields_dont_affect_hash(
        self, ma_raw: dict[str, object]
    ) -> None:
        """Cambiar last_verified NO cambia el checksum
        (es un campo excluido del hash)."""
        modified = copy.deepcopy(ma_raw)
        modified["last_verified"] = "2099-01-01"
        original_hash = compute_checksum(ma_raw)
        modified_hash = compute_checksum(modified)
        assert original_hash == modified_hash

    def test_excluded_from_hash_fields(self) -> None:
        """EXCLUDED_FROM_HASH contiene exactamente los
        campos esperados."""
        assert "checksum_sha256" in EXCLUDED_FROM_HASH
        assert "last_verified" in EXCLUDED_FROM_HASH

    def test_rehash_dict_does_not_mutate_original(
        self, ma_raw: dict[str, object]
    ) -> None:
        """rehash_dict retorna copia — no muta el original."""
        original_checksum = str(ma_raw["checksum_sha256"])
        original_copy = copy.deepcopy(ma_raw)
        _ = rehash_dict(ma_raw)
        assert ma_raw == original_copy
        assert str(ma_raw["checksum_sha256"]) == original_checksum

    def test_rehash_dict_produces_valid_checksum(
        self, ma_raw: dict[str, object]
    ) -> None:
        """rehash_dict produce un checksum que pasa verify."""
        rehashed = rehash_dict(ma_raw)
        new_checksum = str(rehashed["checksum_sha256"])
        assert verify_checksum(rehashed, new_checksum) is True

    def test_compute_checksum_is_deterministic(self, ma_raw: dict[str, object]) -> None:
        """compute_checksum produce el mismo hash en
        múltiples llamadas con el mismo input."""
        hash1 = compute_checksum(ma_raw)
        hash2 = compute_checksum(ma_raw)
        assert hash1 == hash2

    def test_checksum_is_64_char_hex(self, ma_raw: dict[str, object]) -> None:
        """El checksum tiene exactamente 64 caracteres
        hexadecimales en minúscula."""
        checksum = compute_checksum(ma_raw)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)


# ---------------------------------------------------------------------------
# Grupo 3 — Validators de RegulationConfig
# ---------------------------------------------------------------------------


class TestRegulationConfigValidators:
    def test_dates_inverted_raises_validation_error(
        self, ma_raw: dict[str, object]
    ) -> None:
        """date_end anterior a date_start levanta
        ValidationError."""
        bad = copy.deepcopy(ma_raw)
        bad["date_start"] = "2026-12-31"
        bad["date_end"] = "2026-01-01"
        with pytest.raises(ValidationError):
            RegulationConfig.model_validate(bad)

    def test_date_end_equal_to_start_raises(self, ma_raw: dict[str, object]) -> None:
        """date_end igual a date_start levanta
        ValidationError."""
        bad = copy.deepcopy(ma_raw)
        bad["date_start"] = "2026-04-08"
        bad["date_end"] = "2026-04-08"
        with pytest.raises(ValidationError):
            RegulationConfig.model_validate(bad)

    def test_pokemon_legales_empty_raises(self, ma_raw: dict[str, object]) -> None:
        """pokemon_legales vacío levanta ValidationError."""
        bad = copy.deepcopy(ma_raw)
        bad["pokemon_legales"] = []
        with pytest.raises(ValidationError):
            RegulationConfig.model_validate(bad)

    def test_pokemon_legales_duplicates_raises(self, ma_raw: dict[str, object]) -> None:
        """pokemon_legales con dex IDs duplicados levanta
        ValidationError."""
        bad = copy.deepcopy(ma_raw)
        bad["pokemon_legales"] = [1, 2, 3, 1]
        with pytest.raises(ValidationError):
            RegulationConfig.model_validate(bad)

    def test_mega_enabled_without_megas_raises(self, ma_raw: dict[str, object]) -> None:
        """mega_enabled=True con lista vacía de megas
        levanta ValidationError."""
        bad = copy.deepcopy(ma_raw)
        bad["mega_evolutions_disponibles"] = []
        with pytest.raises(ValidationError):
            RegulationConfig.model_validate(bad)

    def test_extra_fields_forbidden(self, ma_raw: dict[str, object]) -> None:
        """Campos extra no definidos en el schema levantan
        ValidationError (extra='forbid')."""
        bad = copy.deepcopy(ma_raw)
        bad["campo_inventado"] = "valor"
        with pytest.raises(ValidationError):
            RegulationConfig.model_validate(bad)

    def test_invalid_game_value_raises(self, ma_raw: dict[str, object]) -> None:
        """game con valor no permitido levanta
        ValidationError."""
        bad = copy.deepcopy(ma_raw)
        bad["game"] = "juego_inexistente"
        with pytest.raises(ValidationError):
            RegulationConfig.model_validate(bad)

    def test_checksum_wrong_format_raises(self, ma_raw: dict[str, object]) -> None:
        """checksum_sha256 con formato incorrecto levanta
        ValidationError (debe ser 64 chars hex)."""
        bad = copy.deepcopy(ma_raw)
        bad["checksum_sha256"] = "not-a-valid-sha256"
        with pytest.raises(ValidationError):
            RegulationConfig.model_validate(bad)
