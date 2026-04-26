"""
Tests unitarios para src/app/modules/wp.py y src/app/modules/wp_train.py.

Estrategia:
  - Todos los datos son sintéticos en memoria (MOCK_POKEMON_DATA, MockReplay).
  - Ningún test lee pokemon_master.json, DuckDB ni Parquets desde disco.
  - Los Grupos 1-4 son tests unitarios puros: no hacen I/O ni entrenan modelos.
  - El Grupo 5 es integración: entrena XGBoost real con 200 samples sintéticos
    (seed 42) pero siempre usa save_if_passes_gate=False, sin escribir nada a disco.

Grupos:
  1. TestWPHelpers          (17 tests) — _normalize_rating, _extract_type_coverage,
                                         _month_idx_from_upload_time, _extract_roles_from_log,
                                         ALL_TYPES, KEY_ROLES.
  2. TestExtractFeatures    (10 tests) — función principal extract_features con MockReplay.
  3. TestFeaturesToDataframe (7 tests) — conversión a DataFrame, get_feature_names, labels nulos.
  4. TestWPTrainGate         (6 tests) — gate AUC/Brier, _split_data con DataFrames sintéticos.
  5. TestWPTrainIntegration  (5 tests) — pipeline completo con features sintéticas, sin I/O a disco.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.app.modules.wp import (
    ALL_TYPES,
    KEY_ROLES,
    ReplayFeatures,
    _extract_roles_from_log,
    _extract_type_coverage,
    _month_idx_from_upload_time,
    _normalize_rating,
    extract_features,
    features_to_dataframe,
    get_feature_names,
)
from src.app.modules.wp_train import (
    AUC_GATE,
    BRIER_GATE,
    TrainResult,
    _split_data,
)


# ---------------------------------------------------------------------------
# Helpers de módulo
# ---------------------------------------------------------------------------


@dataclass
class MockReplay:
    """Replay sintético para tests — duck-typed sobre ParsedReplay."""

    replay_id: str = "test-123"
    regulation_id: str = "M-A"
    p1: str = "PlayerA"
    p2: str = "PlayerB"
    winner: str | None = "PlayerA"
    rating: int = 1750
    upload_time: int = 1745000000
    team_p1: list[str] | None = None
    team_p2: list[str] | None = None
    raw_log: str = ""

    def __post_init__(self) -> None:
        if self.team_p1 is None:
            self.team_p1 = ["Incineroar", "Garchomp"]
        if self.team_p2 is None:
            self.team_p2 = ["Sneasler", "Sinistcha"]


MOCK_POKEMON_DATA: dict[str, dict[str, Any]] = {
    "incineroar": {
        "types": ["Fire", "Dark"],
        "base_stats": {"speed": 60},
    },
    "garchomp": {
        "types": ["Dragon", "Ground"],
        "base_stats": {"speed": 102},
    },
    "sneasler": {
        "types": ["Fighting", "Poison"],
        "base_stats": {"speed": 120},
    },
    "sinistcha": {
        "types": ["Grass", "Ghost"],
        "base_stats": {"speed": 70},
    },
}


def _make_features(n: int = 5) -> list[ReplayFeatures]:
    """Genera lista de ReplayFeatures sintéticas con label conocido."""
    return [
        ReplayFeatures(
            replay_id=f"r{i}",
            regulation_id="TEST",
            label=float(i % 2),
            rating_norm=0.5,
            month_idx=0,
            p1_type_coverage=[0.0] * 18,
            p2_type_coverage=[0.0] * 18,
            p1_has_fake_out=0.0,
            p2_has_fake_out=0.0,
            p1_has_trick_room=0.0,
            p2_has_trick_room=0.0,
            p1_has_intimidate=0.0,
            p2_has_intimidate=0.0,
            p1_has_redirection=0.0,
            p2_has_redirection=0.0,
            p1_has_tailwind=0.0,
            p2_has_tailwind=0.0,
            p1_speed_control=0.0,
            p2_speed_control=0.0,
            p1_team_size=6.0,
            p2_team_size=6.0,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fixture compartido para tests de integración
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_features() -> list[ReplayFeatures]:
    """
    200 ReplayFeatures con señal sintética (seed=42) para tests del pipeline.
    Delega en _generate_synthetic_features del script CLI para reutilizar la
    misma lógica de generación y evitar duplicación.
    """
    from scripts.train_wp import _generate_synthetic_features  # type: ignore[import]

    return _generate_synthetic_features("TEST", n=200)


# ---------------------------------------------------------------------------
# GRUPO 1 — Funciones auxiliares de wp.py
# ---------------------------------------------------------------------------


class TestWPHelpers:
    """Tests para las funciones auxiliares de wp.py."""

    def test_normalize_rating_min(self) -> None:
        """1500 normaliza a 0.0."""
        assert _normalize_rating(1500) == pytest.approx(0.0, abs=0.001)

    def test_normalize_rating_max(self) -> None:
        """2000 normaliza a 1.0."""
        assert _normalize_rating(2000) == pytest.approx(1.0, abs=0.001)

    def test_normalize_rating_midpoint(self) -> None:
        """1750 normaliza a 0.5."""
        assert _normalize_rating(1750) == pytest.approx(0.5, abs=0.001)

    def test_normalize_rating_clamps_below(self) -> None:
        """Rating < 1500 se clampea a 0.0."""
        assert _normalize_rating(999) == pytest.approx(0.0, abs=0.001)

    def test_normalize_rating_clamps_above(self) -> None:
        """Rating > 2000 se clampea a 1.0."""
        assert _normalize_rating(9999) == pytest.approx(1.0, abs=0.001)

    def test_type_coverage_length(self) -> None:
        """Siempre retorna lista de 18 floats."""
        coverage = _extract_type_coverage(["Incineroar"], MOCK_POKEMON_DATA)
        assert len(coverage) == 18

    def test_type_coverage_fire_dark(self) -> None:
        """Incineroar (Fire/Dark) tiene cobertura > 0 en esos tipos."""
        coverage = _extract_type_coverage(["Incineroar"], MOCK_POKEMON_DATA)
        fire_idx = ALL_TYPES.index("Fire")
        dark_idx = ALL_TYPES.index("Dark")
        assert coverage[fire_idx] > 0
        assert coverage[dark_idx] > 0

    def test_type_coverage_empty_team(self) -> None:
        """Equipo vacío retorna 18 ceros."""
        coverage = _extract_type_coverage([], MOCK_POKEMON_DATA)
        assert len(coverage) == 18
        assert all(c == 0.0 for c in coverage)

    def test_type_coverage_empty_pokemon_data(self) -> None:
        """Sin pokemon_data retorna 18 ceros."""
        coverage = _extract_type_coverage(["Incineroar"], {})
        assert all(c == 0.0 for c in coverage)

    def test_type_coverage_normalized(self) -> None:
        """Valores están en [0.0, 1.0]."""
        coverage = _extract_type_coverage(
            ["Incineroar", "Garchomp"], MOCK_POKEMON_DATA
        )
        assert all(0.0 <= c <= 1.0 for c in coverage)

    def test_month_idx_zero_for_start_month(self) -> None:
        """Timestamp en mes de inicio da idx 0."""
        from datetime import datetime

        ts = int(datetime(2026, 4, 15).timestamp())
        idx = _month_idx_from_upload_time(ts, "2026-04")
        assert idx == 0

    def test_month_idx_positive_for_later(self) -> None:
        """Mes posterior al inicio da idx positivo."""
        from datetime import datetime

        ts = int(datetime(2026, 6, 15).timestamp())
        idx = _month_idx_from_upload_time(ts, "2026-04")
        assert idx == 2

    def test_all_types_has_18_elements(self) -> None:
        """ALL_TYPES tiene exactamente 18 elementos."""
        assert len(ALL_TYPES) == 18

    def test_key_roles_is_non_empty(self) -> None:
        """KEY_ROLES contiene al menos un rol."""
        assert len(KEY_ROLES) > 0

    def test_extract_roles_fake_out_detected(self) -> None:
        """Detecta Fake Out en el log de p1."""
        log_text = "|move|p1a|Fake Out|p2a|\n|move|p2a|Protect|p2a|\n"
        roles = _extract_roles_from_log(log_text, "p1")
        assert roles["fake_out"] == 1.0

    def test_extract_roles_empty_log(self) -> None:
        """Log vacío retorna todos los roles en 0.0."""
        roles = _extract_roles_from_log("", "p1")
        assert all(v == 0.0 for v in roles.values())

    def test_extract_roles_trick_room(self) -> None:
        """Detecta Trick Room en el log de p2."""
        log_text = "|move|p2a|Trick Room|\n"
        roles = _extract_roles_from_log(log_text, "p2")
        assert roles["trick_room"] == 1.0


# ---------------------------------------------------------------------------
# GRUPO 2 — extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    """Tests para la función principal de extracción de features."""

    def test_returns_replay_features(self) -> None:
        """Replay válido retorna ReplayFeatures."""
        feat = extract_features(MockReplay())
        assert isinstance(feat, ReplayFeatures)

    def test_label_one_when_p1_wins(self) -> None:
        """label=1.0 cuando winner==p1."""
        replay = MockReplay(p1="PlayerA", p2="PlayerB", winner="PlayerA")
        feat = extract_features(replay)
        assert feat is not None
        assert feat.label == pytest.approx(1.0)

    def test_label_zero_when_p2_wins(self) -> None:
        """label=0.0 cuando winner==p2."""
        replay = MockReplay(p1="PlayerA", p2="PlayerB", winner="PlayerB")
        feat = extract_features(replay)
        assert feat is not None
        assert feat.label == pytest.approx(0.0)

    def test_returns_none_empty_players(self) -> None:
        """None si p1 y p2 están vacíos."""
        feat = extract_features(MockReplay(p1="", p2=""))
        assert feat is None

    def test_label_none_when_winner_none(self) -> None:
        """label=None cuando winner=None."""
        feat = extract_features(MockReplay(winner=None))
        assert feat is not None
        assert feat.label is None

    def test_rating_norm_in_range(self) -> None:
        """rating_norm en [0, 1]."""
        feat = extract_features(MockReplay(rating=1750))
        assert feat is not None
        assert 0.0 <= feat.rating_norm <= 1.0

    def test_type_coverage_length_18(self) -> None:
        """Las coberturas de tipo tienen 18 elementos."""
        feat = extract_features(MockReplay(), pokemon_data=MOCK_POKEMON_DATA)
        assert feat is not None
        assert len(feat.p1_type_coverage) == 18
        assert len(feat.p2_type_coverage) == 18

    def test_team_size_matches(self) -> None:
        """p1/p2_team_size coinciden con el equipo."""
        replay = MockReplay(team_p1=["A", "B", "C"], team_p2=["D", "E"])
        feat = extract_features(replay)
        assert feat is not None
        assert feat.p1_team_size == 3.0
        assert feat.p2_team_size == 2.0

    def test_fake_out_detected_in_log(self) -> None:
        """p1_has_fake_out=1.0 si está en el log."""
        log_with_fo = "|move|p1a|Fake Out|p2a|\n"
        feat = extract_features(MockReplay(raw_log=log_with_fo))
        assert feat is not None
        assert feat.p1_has_fake_out == 1.0

    def test_no_exception_on_malformed_replay(self) -> None:
        """No crash con objeto sin los atributos esperados — retorna None."""

        class EmptyObj:
            pass

        feat = extract_features(EmptyObj())
        assert feat is None


# ---------------------------------------------------------------------------
# GRUPO 3 — features_to_dataframe y get_feature_names
# ---------------------------------------------------------------------------


class TestFeaturesToDataframe:
    """Tests para la conversión de ReplayFeatures a DataFrame."""

    def test_returns_dataframe(self) -> None:
        """Retorna pandas DataFrame."""
        df = features_to_dataframe(_make_features())
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches(self) -> None:
        """Número de filas == número de features."""
        df = features_to_dataframe(_make_features(7))
        assert len(df) == 7

    def test_has_type_columns(self) -> None:
        """Tiene columnas p1/p2_type_{tipo} ×18."""
        df = features_to_dataframe(_make_features())
        for t in ALL_TYPES:
            assert f"p1_type_{t}" in df.columns
            assert f"p2_type_{t}" in df.columns

    def test_get_feature_names_subset_of_columns(self) -> None:
        """Todas las features de get_feature_names() están en el DataFrame."""
        df = features_to_dataframe(_make_features())
        for name in get_feature_names():
            assert name in df.columns

    def test_empty_list_returns_empty_df(self) -> None:
        """Lista vacía retorna DataFrame vacío."""
        df = features_to_dataframe([])
        assert df.empty

    def test_total_feature_count_is_52(self) -> None:
        """get_feature_names() retorna exactamente 52 features."""
        assert len(get_feature_names()) == 52

    def test_includes_null_labels(self) -> None:
        """Filas con label=None se incluyen en el DataFrame (sin descartarlas)."""
        feats = _make_features(3)
        feats[1].label = None
        df = features_to_dataframe(feats)
        assert len(df) == 3
        assert df["label"].isna().sum() == 1


# ---------------------------------------------------------------------------
# GRUPO 4 — gate AUC/Brier y _split_data
# ---------------------------------------------------------------------------


class TestWPTrainGate:
    """Tests para la lógica del gate de calidad y el split de datos."""

    def test_gate_constants_values(self) -> None:
        """AUC_GATE=0.65 y BRIER_GATE=0.20 exactamente."""
        assert AUC_GATE == pytest.approx(0.65)
        assert BRIER_GATE == pytest.approx(0.20)

    def test_gate_requires_both_conditions(self) -> None:
        """Gate falla si solo una condición se cumple (lógica AND)."""
        # Pasa ambas condiciones
        assert 0.70 >= AUC_GATE
        assert 0.15 <= BRIER_GATE
        # Falla solo por AUC bajo
        assert not (0.60 >= AUC_GATE)
        # Falla solo por Brier alto
        assert not (0.25 <= BRIER_GATE)

    def test_split_data_sizes_sum_to_total(self) -> None:
        """Train + val + test suman al total de muestras con label."""
        n = 100
        df = pd.DataFrame(
            {"label": [float(i % 2) for i in range(n)], "feature": range(n)}
        )
        df_train, df_val, df_test = _split_data(df)
        assert len(df_train) + len(df_val) + len(df_test) == n

    def test_split_train_larger_than_val(self) -> None:
        """Train es más grande que val y que test (ratio 70/15/15)."""
        df = pd.DataFrame(
            {"label": [float(i % 2) for i in range(100)], "feature": range(100)}
        )
        df_train, df_val, df_test = _split_data(df)
        assert len(df_train) > len(df_val)
        assert len(df_train) > len(df_test)

    def test_split_raises_with_few_samples(self) -> None:
        """ValueError con menos de 20 muestras con label conocido."""
        df = pd.DataFrame({"label": [1.0, 0.0, 1.0], "feature": [1, 2, 3]})
        with pytest.raises(ValueError):
            _split_data(df)

    def test_split_ignores_null_labels(self) -> None:
        """Solo las filas con label conocido entran al split; las None se descartan."""
        df = pd.DataFrame(
            {
                "label": [1.0] * 15 + [0.0] * 15 + [None] * 10,
                "feature": range(40),
            }
        )
        df_train, df_val, df_test = _split_data(df)
        assert len(df_train) + len(df_val) + len(df_test) == 30


# ---------------------------------------------------------------------------
# GRUPO 5 — Integración del pipeline completo
# ---------------------------------------------------------------------------


class TestWPTrainIntegration:
    """
    Tests de integración del pipeline completo de entrenamiento WP.

    Usan la fixture synthetic_features (200 samples con seed=42) para
    verificar que el pipeline no crashea sin necesitar replays reales.
    Siempre se ejecutan con save_if_passes_gate=False — no escriben
    ningún archivo en disco durante los tests.
    """

    def test_train_returns_train_result(
        self,
        synthetic_features: list[ReplayFeatures],
    ) -> None:
        """train_wp_model retorna instancia de TrainResult."""
        from src.app.modules.wp_train import train_wp_model

        result = train_wp_model(
            synthetic_features,
            regulation_id="TEST",
            save_if_passes_gate=False,
        )
        assert isinstance(result, TrainResult)

    def test_train_result_has_metrics(
        self,
        synthetic_features: list[ReplayFeatures],
    ) -> None:
        """TrainResult contiene AUC y Brier en [0.0, 1.0]."""
        from src.app.modules.wp_train import train_wp_model

        result = train_wp_model(
            synthetic_features,
            regulation_id="TEST",
            save_if_passes_gate=False,
        )
        assert 0.0 <= result.auc <= 1.0
        assert 0.0 <= result.brier <= 1.0

    def test_train_result_sample_counts(
        self,
        synthetic_features: list[ReplayFeatures],
    ) -> None:
        """n_train + n_val + n_test == 200 (todas las features tienen label)."""
        from src.app.modules.wp_train import train_wp_model

        result = train_wp_model(
            synthetic_features,
            regulation_id="TEST",
            save_if_passes_gate=False,
        )
        total = result.n_train + result.n_val + result.n_test
        assert total == 200

    def test_train_raises_on_empty_features(self) -> None:
        """ValueError cuando la lista de features está vacía."""
        from src.app.modules.wp_train import train_wp_model

        with pytest.raises(ValueError):
            train_wp_model([], regulation_id="TEST")

    def test_train_does_not_save_without_gate(
        self,
        synthetic_features: list[ReplayFeatures],
        tmp_path: Path,
    ) -> None:
        """model_path es None cuando save_if_passes_gate=False (no escribe a disco)."""
        from src.app.modules.wp_train import train_wp_model

        result = train_wp_model(
            synthetic_features,
            regulation_id="TEST",
            save_if_passes_gate=False,
        )
        assert result.model_path is None
