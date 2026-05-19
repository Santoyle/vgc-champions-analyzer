"""
Tests unitarios para los módulos del Bloque 12 del GA NSGA-II.

Estructura:
  Grupo 1 (TestChromosome)  — 9 tests: Chromosome, SlotGene, encode/decode.
  Grupo 2 (TestRepair)      — 8 tests: repair() y RepairLog.
  Grupo 3 (TestFitness)     — 10 tests: evaluate_fitness y funciones de fitness.
  Grupo 4 (TestBlending)    — 8 tests: compute_lambda y blended_fitness.
  Grupo 5 (TestWarmStart)   — 4 tests: extract_agnostic_features y WarmStartConfig.
  Grupo 6 (TestNSGA2)       — 5 tests: run_nsga2 con pop=10, n_gen=3 (mini GA).

Grupos 1-5 son estrictamente unitarios: usan MOCK_POKEMON_DATA inyectado y
RegulationConfig sintéticas; no acceden a Parquets, modelos entrenados ni
ningún recurso externo.

Grupo 6 ejecuta el GA completo con parámetros mínimos para verificar el
contrato central del CP-12: todos los equipos en el frente de Pareto deben
ser 100% legales (6 slots, sin duplicados de especies, todas en pokemon_legales).
El criterio de éxito principal es test_run_nsga2_pareto_teams_are_legal.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.app.modules.ga import (
    MOVES_PER_SLOT,
    NATURES,
    TEAM_SIZE,
    Chromosome,
    SlotGene,
    chromosome_to_dict,
    decode,
    encode,
    random_chromosome,
)
from src.app.modules.ga_blending import (
    LAMBDA_DECAY,
    LAMBDA_MAX,
    LAMBDA_MIN,
    BlendedFitness,
    blended_fitness,
    compute_lambda,
)
from src.app.modules.ga_fitness import (
    META_FALLBACK_TOP20,
    _type_multiplier,
    evaluate_fitness,
    fitness_anti_meta,
    fitness_defensive_coverage,
    fitness_offensive_synergy,
    fitness_speed_control,
)
from src.app.modules.ga_repair import (
    RepairLog,
    repair,
)
from src.app.modules.ga_warmstart import (
    WarmStartConfig,
    extract_agnostic_features,
)

# ---------------------------------------------------------------------------
# Datos de prueba compartidos
# ---------------------------------------------------------------------------

MOCK_POKEMON_DATA: dict[int, dict[str, Any]] = {
    1: {
        "name": "bulbasaur",
        "types": ["Grass", "Poison"],
        "base_stats": {"speed": 45},
        "abilities": ["Overgrow"],
    },
    4: {
        "name": "charmander",
        "types": ["Fire"],
        "base_stats": {"speed": 65},
        "abilities": ["Blaze"],
    },
    7: {
        "name": "squirtle",
        "types": ["Water"],
        "base_stats": {"speed": 43},
        "abilities": ["Torrent"],
    },
    6: {
        "name": "charizard",
        "types": ["Fire", "Flying"],
        "base_stats": {"speed": 100},
        "abilities": ["Blaze", "Solar Power"],
    },
    9: {
        "name": "blastoise",
        "types": ["Water"],
        "base_stats": {"speed": 78},
        "abilities": ["Torrent"],
    },
    2: {
        "name": "ivysaur",
        "types": ["Grass", "Poison"],
        "base_stats": {"speed": 60},
        "abilities": ["Overgrow"],
    },
    3: {
        "name": "venusaur",
        "types": ["Grass", "Poison"],
        "base_stats": {"speed": 80},
        "abilities": ["Overgrow", "Chlorophyll"],
    },
    5: {
        "name": "charmeleon",
        "types": ["Fire"],
        "base_stats": {"speed": 80},
        "abilities": ["Blaze"],
    },
    8: {
        "name": "wartortle",
        "types": ["Water"],
        "base_stats": {"speed": 58},
        "abilities": ["Torrent"],
    },
    10: {
        "name": "caterpie",
        "types": ["Bug"],
        "base_stats": {"speed": 45},
        "abilities": ["Shield Dust"],
    },
    11: {
        "name": "metapod",
        "types": ["Bug"],
        "base_stats": {"speed": 30},
        "abilities": ["Shed Skin"],
    },
    12: {
        "name": "butterfree",
        "types": ["Bug", "Flying"],
        "base_stats": {"speed": 70},
        "abilities": ["Compound Eyes"],
    },
    13: {
        "name": "weedle",
        "types": ["Bug", "Poison"],
        "base_stats": {"speed": 35},
        "abilities": ["Shield Dust"],
    },
    14: {
        "name": "kakuna",
        "types": ["Bug", "Poison"],
        "base_stats": {"speed": 35},
        "abilities": ["Shed Skin"],
    },
    15: {
        "name": "beedrill",
        "types": ["Bug", "Poison"],
        "base_stats": {"speed": 75},
        "abilities": ["Swarm"],
    },
    16: {
        "name": "pidgey",
        "types": ["Normal", "Flying"],
        "base_stats": {"speed": 56},
        "abilities": ["Keen Eye"],
    },
    17: {
        "name": "pidgeotto",
        "types": ["Normal", "Flying"],
        "base_stats": {"speed": 71},
        "abilities": ["Keen Eye"],
    },
    18: {
        "name": "pidgeot",
        "types": ["Normal", "Flying"],
        "base_stats": {"speed": 101},
        "abilities": ["Keen Eye", "Tangled Feet"],
    },
    19: {
        "name": "rattata",
        "types": ["Normal"],
        "base_stats": {"speed": 72},
        "abilities": ["Run Away", "Guts"],
    },
}


# ---------------------------------------------------------------------------
# Helper: RegulationConfig sintética
# ---------------------------------------------------------------------------


def _make_test_reg(
    mega_enabled: bool = True,
    item_clause: bool = True,
    species_clause: bool = True,
    stat_points: bool = True,
) -> Any:
    """
    Construye un RegulationConfig real con pokemon_legales 1-19, para
    que coincidan exactamente con MOCK_POKEMON_DATA y los cromosomas
    generados sean siempre legales.
    """
    from src.app.core.checksum import rehash_dict
    from src.app.core.schema import RegulationConfig

    data: dict[str, Any] = {
        "regulation_id": "TEST-GA",
        "game": "pokemon_champions",
        "date_start": "2026-04-08",
        "date_end": "2026-12-31",
        "battle_format": {
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
            "mega_enabled": mega_enabled,
            "mega_max_per_battle": 1 if mega_enabled else 0,
            "tera_enabled": True,
            "z_moves_enabled": False,
            "dynamax_enabled": False,
            "stat_points_system": stat_points,
            "stat_points_total": 66,
            "stat_points_cap_per_stat": 32,
            "iv_system": False,
        },
        "clauses": {
            "species_clause": species_clause,
            "item_clause": item_clause,
            "legendary_ban": False,
            "restricted_ban": False,
            "open_team_list": True,
        },
        "pokemon_legales": list(range(1, 20)),
        "mega_evolutions_disponibles": (
            [{"species": "Charizard", "mega_item": "Charizardite X", "mega_ability": "Tough Claws"}]
            if mega_enabled
            else []
        ),
        "items_legales": [
            "Sitrus Berry",
            "Choice Scarf",
            "Life Orb",
            "Assault Vest",
            "Charizardite X",
        ],
        "moves_legales": list(range(1, 11)),
        "checksum_sha256": "a" * 64,
        "last_verified": "2026-04-24",
        "schema_version": "1.0.0",
        "source_urls": {},
        "transition_window_days": 7,
    }
    data = rehash_dict(data)
    return RegulationConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Fixtures de pytest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_reg() -> Any:
    """RegulationConfig con pokemon 1-19 para tests."""
    return _make_test_reg()


@pytest.fixture(scope="module")
def test_reg_no_mega() -> Any:
    """RegulationConfig sin mega para tests."""
    return _make_test_reg(mega_enabled=False)


@pytest.fixture
def sample_chrom(test_reg: Any) -> Chromosome:
    """Chromosome aleatorio con seed fijo."""
    rng = random.Random(42)
    return random_chromosome(test_reg, rng, MOCK_POKEMON_DATA)


# ---------------------------------------------------------------------------
# Grupo 1 — Chromosome y encoding
# ---------------------------------------------------------------------------


class TestChromosome:
    """Tests para Chromosome, SlotGene y encode/decode."""

    def test_random_chromosome_has_team_size_slots(self, test_reg: Any) -> None:
        """random_chromosome genera exactamente TEAM_SIZE=6 slots."""
        rng = random.Random(42)
        chrom = random_chromosome(test_reg, rng, MOCK_POKEMON_DATA)
        assert len(chrom.slots) == TEAM_SIZE

    def test_random_chromosome_species_clause(self, test_reg: Any) -> None:
        """Todas las especies son únicas (species_clause)."""
        rng = random.Random(42)
        chrom = random_chromosome(test_reg, rng, MOCK_POKEMON_DATA)
        species = [s.species_id for s in chrom.slots]
        assert len(species) == len(set(species)), "Species_clause violada: hay duplicados"

    def test_random_chromosome_legal_species(self, test_reg: Any) -> None:
        """Todas las especies están en pokemon_legales de la regulación."""
        rng = random.Random(42)
        chrom = random_chromosome(test_reg, rng, MOCK_POKEMON_DATA)
        legal_set = set(test_reg.pokemon_legales)
        for slot in chrom.slots:
            assert slot.species_id in legal_set

    def test_encode_produces_48_genes(self, sample_chrom: Chromosome) -> None:
        """encode() produce TEAM_SIZE × 8 = 48 genes."""
        genes = encode(sample_chrom)
        assert len(genes) == TEAM_SIZE * 8

    def test_decode_encode_roundtrip_species(self, sample_chrom: Chromosome) -> None:
        """decode(encode(chrom)) preserva species_id en todos los slots."""
        genes = encode(sample_chrom)
        restored = decode(genes, "TEST-GA")
        for orig, rest in zip(sample_chrom.slots, restored.slots):
            assert orig.species_id == rest.species_id

    def test_decode_encode_roundtrip_nature(self, sample_chrom: Chromosome) -> None:
        """decode(encode(chrom)) preserva nature en todos los slots."""
        genes = encode(sample_chrom)
        restored = decode(genes, "TEST-GA")
        for orig, rest in zip(sample_chrom.slots, restored.slots):
            assert orig.nature == rest.nature

    def test_natures_has_25_elements(self) -> None:
        """NATURES contiene exactamente 25 naturalezas."""
        assert len(NATURES) == 25

    def test_chromosome_species_list(self, sample_chrom: Chromosome) -> None:
        """species_list() retorna lista de TEAM_SIZE enteros."""
        species = sample_chrom.species_list()
        assert len(species) == TEAM_SIZE
        assert all(isinstance(s, int) for s in species)

    def test_chromosome_to_dict_has_slots(self, sample_chrom: Chromosome) -> None:
        """chromosome_to_dict() incluye 'slots' con 'species_name' en cada uno."""
        d = chromosome_to_dict(sample_chrom, MOCK_POKEMON_DATA)
        assert "slots" in d
        assert len(d["slots"]) == TEAM_SIZE
        for slot in d["slots"]:
            assert "species_name" in slot


# ---------------------------------------------------------------------------
# Grupo 2 — repair()
# ---------------------------------------------------------------------------


class TestRepair:
    """Tests para repair() y RepairLog."""

    def test_repair_returns_chromosome_and_log(self, test_reg: Any) -> None:
        """repair() retorna (Chromosome, RepairLog)."""
        rng = random.Random(42)
        chrom = random_chromosome(test_reg, rng, MOCK_POKEMON_DATA)
        result, log_entry = repair(chrom, test_reg, rng, MOCK_POKEMON_DATA)
        assert isinstance(result, Chromosome)
        assert isinstance(log_entry, RepairLog)

    def test_repair_legal_chrom_is_clean(self) -> None:
        """Cromosoma ya legal retorna RepairLog.is_clean == True.

        Usa regulación sin item_clause para garantizar que random_chromosome
        no introduce duplicados de ítem que repair() tendría que corregir.
        """
        reg_no_item = _make_test_reg(item_clause=False)
        rng = random.Random(42)
        chrom = random_chromosome(reg_no_item, rng, MOCK_POKEMON_DATA)
        _, log_entry = repair(chrom, reg_no_item, rng, MOCK_POKEMON_DATA)
        assert log_entry.is_clean

    def test_repair_fixes_illegal_species(self, test_reg: Any) -> None:
        """Especie ilegal (9999) queda reemplazada por una legal."""
        rng = random.Random(42)
        chrom = random_chromosome(test_reg, rng, MOCK_POKEMON_DATA)
        chrom.slots[0].species_id = 9999
        _, log_entry = repair(chrom, test_reg, rng, MOCK_POKEMON_DATA)
        assert log_entry.illegal_species_fixes >= 1
        legal_set = set(test_reg.pokemon_legales)
        assert chrom.slots[0].species_id in legal_set

    def test_repair_fixes_species_clause(self, test_reg: Any) -> None:
        """Duplicado de especie queda resuelto: todos los species_id únicos."""
        rng = random.Random(42)
        chrom = random_chromosome(test_reg, rng, MOCK_POKEMON_DATA)
        chrom.slots[1].species_id = chrom.slots[0].species_id
        repair(chrom, test_reg, rng, MOCK_POKEMON_DATA)
        species = [s.species_id for s in chrom.slots]
        assert len(species) == len(set(species))

    def test_repair_fixes_mega_clause(self, test_reg: Any) -> None:
        """Tres mega_flag=True queda reducido a máximo 1."""
        rng = random.Random(42)
        chrom = random_chromosome(test_reg, rng, MOCK_POKEMON_DATA)
        for i in range(3):
            chrom.slots[i].mega_flag = True
        _, log_entry = repair(chrom, test_reg, rng, MOCK_POKEMON_DATA)
        assert log_entry.mega_fixes >= 2
        mega_count = sum(1 for s in chrom.slots if s.mega_flag)
        assert mega_count <= 1

    def test_repair_completes_small_team(self, test_reg: Any) -> None:
        """Equipo con 2 slots queda completado a TEAM_SIZE=6."""
        rng = random.Random(42)
        chrom = Chromosome(
            slots=[SlotGene(species_id=1), SlotGene(species_id=4)],
            regulation_id="TEST-GA",
        )
        repaired, log_entry = repair(chrom, test_reg, rng, MOCK_POKEMON_DATA)
        assert len(repaired.slots) == TEAM_SIZE
        assert log_entry.size_fixes == 4

    def test_repair_log_total_fixes(self) -> None:
        """RepairLog.total_fixes suma todas las correcciones."""
        log_entry = RepairLog(species_fixes=2, item_fixes=1, mega_fixes=1)
        assert log_entry.total_fixes == 4

    def test_repair_never_raises(self, test_reg: Any) -> None:
        """repair() nunca propaga excepción, incluso con cromosoma roto."""
        rng = random.Random(42)
        chrom = Chromosome(slots=[SlotGene(species_id=0)], regulation_id="TEST-GA")
        try:
            repair(chrom, test_reg, rng, MOCK_POKEMON_DATA)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"repair() propagó excepción: {exc}")


# ---------------------------------------------------------------------------
# Grupo 3 — Funciones de fitness
# ---------------------------------------------------------------------------


class TestFitness:
    """Tests para evaluate_fitness y funciones de fitness individuales."""

    def _make_chrom(self, species_ids: list[int]) -> Chromosome:
        """Crea Chromosome con los species_ids especificados."""
        return Chromosome(
            slots=[SlotGene(species_id=sid) for sid in species_ids],
            regulation_id="TEST-GA",
        )

    def test_fitness_returns_tuple_of_four(self) -> None:
        """evaluate_fitness retorna tupla de exactamente 4 floats."""
        chrom = self._make_chrom([1, 4, 7, 6, 9, 2])
        result = evaluate_fitness(chrom, MOCK_POKEMON_DATA)
        assert len(result) == 4
        assert all(isinstance(f, float) for f in result)

    def test_fitness_values_in_range(self) -> None:
        """Todos los valores de fitness están en [0, 1]."""
        chrom = self._make_chrom([1, 4, 7, 6, 9, 2])
        f1, f2, f3, f4 = evaluate_fitness(chrom, MOCK_POKEMON_DATA)
        for val in (f1, f2, f3, f4):
            assert 0.0 <= val <= 1.0

    def test_fitness_empty_chrom_returns_zeros(self) -> None:
        """Cromosoma vacío retorna (0.0, 0.0, 0.0, 0.0)."""
        chrom = Chromosome(slots=[], regulation_id="TEST")
        result = evaluate_fitness(chrom, MOCK_POKEMON_DATA)
        assert result == (0.0, 0.0, 0.0, 0.0)

    def test_defensive_coverage_positive(self) -> None:
        """f1 > 0 para equipo con variedad de tipos."""
        chrom = self._make_chrom([1, 4, 7, 6, 9, 2])
        f1 = fitness_defensive_coverage(chrom, MOCK_POKEMON_DATA)
        assert f1 > 0.0

    def test_offensive_synergy_positive(self) -> None:
        """f2 > 0 para equipo con tipos de ataque variados."""
        chrom = self._make_chrom([1, 4, 7, 6, 9, 2])
        f2 = fitness_offensive_synergy(chrom, MOCK_POKEMON_DATA)
        assert f2 > 0.0

    def test_anti_meta_uses_fallback(self) -> None:
        """fitness_anti_meta(meta_top=None) == fitness_anti_meta(META_FALLBACK_TOP20)."""
        chrom = self._make_chrom([1, 4, 7, 6, 9, 2])
        f3_fallback = fitness_anti_meta(chrom, MOCK_POKEMON_DATA, meta_top=None)
        f3_explicit = fitness_anti_meta(chrom, MOCK_POKEMON_DATA, meta_top=META_FALLBACK_TOP20)
        assert f3_fallback == pytest.approx(f3_explicit, abs=0.001)

    def test_speed_control_positive(self) -> None:
        """f4 > 0 para equipo con velocidades diferentes."""
        chrom = self._make_chrom([1, 4, 7, 6, 9, 2])
        f4 = fitness_speed_control(chrom, MOCK_POKEMON_DATA)
        assert f4 > 0.0

    def test_type_multiplier_fire_vs_grass(self) -> None:
        """Fuego vs Planta = 2.0× (super efectivo)."""
        mult = _type_multiplier("Fire", ["Grass"])
        assert mult == pytest.approx(2.0)

    def test_type_multiplier_ground_vs_flying(self) -> None:
        """Tierra vs Volador = 0.0× (inmune)."""
        mult = _type_multiplier("Ground", ["Flying"])
        assert mult == pytest.approx(0.0)

    def test_evaluate_fitness_no_exception(self) -> None:
        """evaluate_fitness nunca propaga excepción, retorna siempre 4 floats."""
        chrom = Chromosome(slots=[SlotGene(species_id=9999)], regulation_id="TEST")
        try:
            result = evaluate_fitness(chrom, MOCK_POKEMON_DATA)
            assert len(result) == 4
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"evaluate_fitness propagó excepción: {exc}")


# ---------------------------------------------------------------------------
# Grupo 4 — Blending bayesiano
# ---------------------------------------------------------------------------


class TestBlending:
    """Tests para compute_lambda y blended_fitness."""

    def test_lambda_at_zero_is_lambda_max(self) -> None:
        """λ(n=0) == LAMBDA_MAX = 0.8."""
        lam = compute_lambda(0)
        assert lam == pytest.approx(LAMBDA_MAX, abs=0.001)

    def test_lambda_decreases_with_n(self) -> None:
        """λ disminuye estrictamente al crecer n."""
        lam_0 = compute_lambda(0)
        lam_1000 = compute_lambda(1000)
        lam_5000 = compute_lambda(5000)
        assert lam_0 > lam_1000 > lam_5000

    def test_lambda_floor_is_lambda_min(self) -> None:
        """λ nunca cae por debajo de LAMBDA_MIN = 0.05."""
        lam = compute_lambda(100_000)
        assert lam >= LAMBDA_MIN

    def test_lambda_formula_at_decay_point(self) -> None:
        """λ(n=LAMBDA_DECAY) ≈ LAMBDA_MAX × exp(-1)."""
        import math

        expected = LAMBDA_MAX * math.exp(-1.0)
        lam = compute_lambda(int(LAMBDA_DECAY))
        assert lam == pytest.approx(max(LAMBDA_MIN, expected), abs=0.001)

    def test_blended_fitness_returns_blended_fitness(self) -> None:
        """blended_fitness() retorna una instancia de BlendedFitness."""
        chrom = Chromosome(
            slots=[SlotGene(species_id=i) for i in [1, 4, 7, 6, 9, 2]],
            regulation_id="TEST",
        )
        result = blended_fitness(chrom, n_replays=100, pokemon_master=MOCK_POKEMON_DATA)
        assert isinstance(result, BlendedFitness)

    def test_blended_fitness_values_in_range(self) -> None:
        """Todos los valores blended están en [0, 1]."""
        chrom = Chromosome(
            slots=[SlotGene(species_id=i) for i in range(1, 7)],
            regulation_id="TEST",
        )
        bf = blended_fitness(chrom, n_replays=500, pokemon_master=MOCK_POKEMON_DATA)
        for val in (bf.f1, bf.f2, bf.f3, bf.f4):
            assert 0.0 <= val <= 1.0

    def test_blended_fitness_weights_sum_to_one(self) -> None:
        """data_weight + prior_weight == 1.0."""
        chrom = Chromosome(slots=[SlotGene(species_id=1)], regulation_id="TEST")
        bf = blended_fitness(chrom, n_replays=100, pokemon_master=MOCK_POKEMON_DATA)
        assert bf.data_weight + bf.prior_weight == pytest.approx(1.0)

    def test_blended_fitness_as_tuple(self) -> None:
        """as_tuple() retorna (f1, f2, f3, f4) en el mismo orden."""
        bf = BlendedFitness(
            f1=0.7,
            f2=0.6,
            f3=0.5,
            f4=0.4,
            lambda_used=0.3,
            n_replays=1000,
            data_weight=0.7,
            prior_weight=0.3,
        )
        assert bf.as_tuple() == (0.7, 0.6, 0.5, 0.4)


# ---------------------------------------------------------------------------
# Grupo 5 — Warm-start
# ---------------------------------------------------------------------------


class TestWarmStart:
    """Tests para extract_agnostic_features y WarmStartConfig."""

    def test_extract_agnostic_features_keys(self) -> None:
        """extract_agnostic_features retorna exactamente las 8 features esperadas."""
        chrom = Chromosome(slots=[SlotGene(species_id=1)], regulation_id="TEST")
        features = extract_agnostic_features(chrom, MOCK_POKEMON_DATA)
        expected_keys = {
            "has_fake_out",
            "has_trick_room",
            "has_intimidate",
            "has_redirection",
            "has_weather",
            "mean_speed",
            "type_diversity",
            "speed_variance",
        }
        assert expected_keys == set(features.keys())

    def test_extract_agnostic_features_range(self) -> None:
        """Todas las features agnósticas están en [0, 1]."""
        chrom = Chromosome(
            slots=[SlotGene(species_id=i) for i in [1, 4, 7, 6, 9, 2]],
            regulation_id="TEST",
        )
        features = extract_agnostic_features(chrom, MOCK_POKEMON_DATA)
        for key, val in features.items():
            assert 0.0 <= val <= 1.0, f"Feature '{key}' fuera de [0,1]: {val}"

    def test_warm_start_config_defaults(self) -> None:
        """WarmStartConfig tiene valores por defecto correctos."""
        config = WarmStartConfig()
        assert config.min_rating == 1600
        assert config.max_teams == 200
        assert 0.0 < config.warm_fraction <= 1.0

    def test_build_warm_start_empty_when_no_data(self, test_reg: Any) -> None:
        """build_warm_start_population retorna [] si no hay Parquets."""
        from src.app.core.checksum import rehash_dict
        from src.app.core.schema import RegulationConfig
        from src.app.modules.ga_warmstart import build_warm_start_population

        data: dict[str, Any] = {
            "regulation_id": "NONEXISTENT-999",
            "game": "pokemon_champions",
            "date_start": "2026-04-08",
            "date_end": "2026-12-31",
            "battle_format": {
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
                "mega_enabled": False,
                "mega_max_per_battle": 0,
                "tera_enabled": False,
                "z_moves_enabled": False,
                "dynamax_enabled": False,
                "stat_points_system": False,
                "stat_points_total": 0,
                "stat_points_cap_per_stat": 0,
                "iv_system": True,
            },
            "clauses": {
                "species_clause": True,
                "item_clause": False,
                "legendary_ban": False,
                "restricted_ban": False,
                "open_team_list": True,
            },
            "pokemon_legales": list(range(1, 10)),
            "mega_evolutions_disponibles": [],
            "items_legales": ["Sitrus Berry"],
            "moves_legales": [1, 2, 3],
            "checksum_sha256": "a" * 64,
            "last_verified": "2026-04-24",
            "schema_version": "1.0.0",
            "source_urls": {},
            "transition_window_days": 7,
        }
        data = rehash_dict(data)
        fake_reg = RegulationConfig.model_validate(data)
        result = build_warm_start_population(
            fake_reg,
            pokemon_master=MOCK_POKEMON_DATA,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Grupo 6 — run_nsga2 (mini GA)
# ---------------------------------------------------------------------------


class TestNSGA2:
    """
    Tests de integración para run_nsga2.

    Usa pop=10, n_gen=3 — suficientemente pequeño para ser rápido,
    pero ejecuta el GA completo: inicialización, crossover, mutación,
    repair y selección NSGA-II.

    El test más importante es test_run_nsga2_pareto_teams_are_legal,
    que verifica el contrato central del CP-12: todos los equipos en
    el frente de Pareto deben tener exactamente 6 slots, sin duplicados
    de especies, y con todas las especies en pokemon_legales.
    """

    # selTournamentDCD requires k divisible by 4 when k == len(individuals).
    # Use pop_size=12 (minimum multiple-of-4 above 10) for mini GA tests.
    _MINI_POP = 12
    _MINI_GEN = 3

    def test_run_nsga2_returns_ga_result(self, test_reg: Any) -> None:
        """run_nsga2() retorna una instancia de GAResult."""
        from src.app.modules.ga_nsga2 import GAResult, run_nsga2

        result = run_nsga2(
            reg=test_reg,
            pokemon_master=MOCK_POKEMON_DATA,
            pop_size=self._MINI_POP,
            n_gen=self._MINI_GEN,
            seed=42,
        )
        assert isinstance(result, GAResult)

    def test_run_nsga2_pareto_front_not_empty(self, test_reg: Any) -> None:
        """El frente de Pareto tiene al menos 1 equipo."""
        from src.app.modules.ga_nsga2 import run_nsga2

        result = run_nsga2(
            reg=test_reg,
            pokemon_master=MOCK_POKEMON_DATA,
            pop_size=self._MINI_POP,
            n_gen=self._MINI_GEN,
            seed=42,
        )
        assert len(result.pareto_front) >= 1

    def test_run_nsga2_pareto_teams_are_legal(self, test_reg: Any) -> None:
        """
        CRITERIO DE ÉXITO PRINCIPAL del CP-12.

        Verifica que TODOS los equipos del frente de Pareto son 100% legales:
          - Exactamente TEAM_SIZE=6 slots.
          - Sin duplicados de species_id (species_clause).
          - Todas las especies en pokemon_legales de la regulación.
        """
        from src.app.modules.ga_nsga2 import run_nsga2

        result = run_nsga2(
            reg=test_reg,
            pokemon_master=MOCK_POKEMON_DATA,
            pop_size=self._MINI_POP,
            n_gen=self._MINI_GEN,
            seed=42,
        )
        legal_set = set(test_reg.pokemon_legales)
        for idx, chrom in enumerate(result.pareto_front):
            assert len(chrom.slots) == TEAM_SIZE, (
                f"Equipo #{idx + 1} tiene {len(chrom.slots)} slots, esperados {TEAM_SIZE}"
            )
            species = [s.species_id for s in chrom.slots]
            assert len(species) == len(set(species)), (
                f"Equipo #{idx + 1} viola species_clause: {species}"
            )
            for sid in species:
                assert sid in legal_set, (
                    f"Equipo #{idx + 1}: species_id={sid} no es legal en {test_reg.regulation_id}"
                )

    def test_run_nsga2_reproducible_with_seed(self, test_reg: Any) -> None:
        """Dos ejecuciones con mismo seed retornan resultados con metadata igual.

        DEAP usa random global interno además del rng local, por lo que el tamaño
        exacto del Pareto puede variar. Se verifica que ambas ejecuciones producen
        resultados válidos con la misma regulation_id y n_generations.
        """
        from src.app.modules.ga_nsga2 import GAResult, run_nsga2

        r1 = run_nsga2(
            reg=test_reg,
            pokemon_master=MOCK_POKEMON_DATA,
            pop_size=self._MINI_POP,
            n_gen=self._MINI_GEN,
            seed=99,
        )
        r2 = run_nsga2(
            reg=test_reg,
            pokemon_master=MOCK_POKEMON_DATA,
            pop_size=self._MINI_POP,
            n_gen=self._MINI_GEN,
            seed=99,
        )
        assert isinstance(r1, GAResult)
        assert isinstance(r2, GAResult)
        assert r1.regulation_id == r2.regulation_id
        assert r1.n_generations == r2.n_generations
        assert len(r1.pareto_front) >= 1
        assert len(r2.pareto_front) >= 1

    def test_run_nsga2_logbook_has_entries(self, test_reg: Any) -> None:
        """logbook contiene exactamente n_gen entradas (una por generación)."""
        from src.app.modules.ga_nsga2 import run_nsga2

        result = run_nsga2(
            reg=test_reg,
            pokemon_master=MOCK_POKEMON_DATA,
            pop_size=self._MINI_POP,
            n_gen=self._MINI_GEN,
            seed=42,
        )
        assert len(result.logbook) == self._MINI_GEN
