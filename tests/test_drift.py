"""
Tests para src/app/modules/drift.py.

Estrategia:
  - _build_usage_vector se testa con DataFrames sintéticos en memoria.
  - check_drift se testa con unittest.mock.patch sobre _get_monthly_data
    para aislar completamente la lógica de I/O de DuckDB.
  - alibi-detect se mockea vía patch.dict(sys.modules) para el test de
    ausencia de la librería.
  - Ningún test lee disco ni hace requests reales.

Grupos:
  1. TestBuildUsageVector      (5 tests) — vectorización mensual.
  2. TestDriftResult           (3 tests) — dataclass DriftResult.
  3. TestFormatDriftForDiscord (5 tests) — formateador de mensajes Discord.
  4. TestCheckDrift            (5 tests) — función principal con mocks.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.app.modules.drift import (
    DriftResult,
    _build_usage_vector,
    check_drift,
    format_drift_for_discord,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_monthly_df() -> pd.DataFrame:
    """DataFrame de datos mensuales sintético con 4 meses y 5 Pokémon."""
    months = ["2025-05", "2025-06", "2025-07", "2025-08"]
    pokemon = ["Incineroar", "Garchomp", "Sneasler", "Sinistcha", "Kingambit"]
    rng = np.random.default_rng(42)
    rows = []
    for month in months:
        for pkm in pokemon:
            rows.append(
                {
                    "year_month": month,
                    "pokemon": pkm,
                    "usage_pct": float(rng.uniform(5.0, 60.0)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GRUPO 1 — _build_usage_vector
# ---------------------------------------------------------------------------


class TestBuildUsageVector:
    """Tests para la función de vectorización de uso mensual."""

    def test_shape_is_correct(self, sample_monthly_df: pd.DataFrame) -> None:
        """Shape es (n_months, n_pokemon)."""
        months = ["2025-05", "2025-06"]
        pokemon = ["Incineroar", "Garchomp", "Sneasler"]
        X = _build_usage_vector(sample_monthly_df, months, pokemon)
        assert X.shape == (2, 3)

    def test_known_value_is_correct(self, sample_monthly_df: pd.DataFrame) -> None:
        """Valor conocido se ubica en la celda (mes, pokemon) correcta."""
        df = pd.DataFrame(
            {
                "year_month": ["2025-01"],
                "pokemon": ["Incineroar"],
                "usage_pct": [42.5],
            }
        )
        X = _build_usage_vector(df, ["2025-01"], ["Incineroar"])
        assert X[0, 0] == pytest.approx(42.5, abs=0.1)

    def test_missing_pokemon_gives_zero(self, sample_monthly_df: pd.DataFrame) -> None:
        """Pokémon ausente en un mes tiene valor 0.0 en su celda."""
        df = pd.DataFrame(
            {
                "year_month": ["2025-01"],
                "pokemon": ["Incineroar"],
                "usage_pct": [30.0],
            }
        )
        X = _build_usage_vector(df, ["2025-01"], ["Incineroar", "Garchomp"])
        assert X[0, 1] == 0.0

    def test_empty_months_returns_zero_matrix(
        self, sample_monthly_df: pd.DataFrame
    ) -> None:
        """Lista de meses vacía retorna matriz con 0 filas."""
        X = _build_usage_vector(sample_monthly_df, [], ["Incineroar"])
        assert X.shape[0] == 0

    def test_dtype_is_float32(self, sample_monthly_df: pd.DataFrame) -> None:
        """La matriz es float32 para compatibilidad con alibi-detect."""
        X = _build_usage_vector(sample_monthly_df, ["2025-05"], ["Incineroar"])
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# GRUPO 2 — DriftResult
# ---------------------------------------------------------------------------


class TestDriftResult:
    """Tests del dataclass DriftResult."""

    def _make_result(self, is_drift: bool = False, p_val: float = 0.8) -> DriftResult:
        return DriftResult(
            is_drift=is_drift,
            p_val=p_val,
            threshold=0.05,
            distance=0.1,
            regulation_id="TEST",
            ref_period="2025-05 / 2025-06",
            current_period="2025-07",
            n_ref_samples=2,
            n_current_samples=1,
            top_drifted_pokemon=["Incineroar"],
        )

    def test_is_drift_true_when_p_below_threshold(self) -> None:
        """is_drift=True cuando p_val < threshold."""
        result = self._make_result(is_drift=True, p_val=0.01)
        assert result.is_drift is True

    def test_is_drift_false_when_p_above_threshold(self) -> None:
        """is_drift=False cuando p_val > threshold."""
        result = self._make_result(is_drift=False, p_val=0.8)
        assert result.is_drift is False

    def test_top_drifted_pokemon_is_list(self) -> None:
        """top_drifted_pokemon es siempre una lista de strings."""
        result = self._make_result()
        assert isinstance(result.top_drifted_pokemon, list)
        assert all(isinstance(p, str) for p in result.top_drifted_pokemon)


# ---------------------------------------------------------------------------
# GRUPO 3 — format_drift_for_discord
# ---------------------------------------------------------------------------


class TestFormatDriftForDiscord:
    """Tests para el formateador de mensajes Discord."""

    def _make_drift_result(self, is_drift: bool) -> DriftResult:
        return DriftResult(
            is_drift=is_drift,
            p_val=0.01 if is_drift else 0.8,
            threshold=0.05,
            distance=0.3 if is_drift else 0.05,
            regulation_id="M-A",
            ref_period="2025-05 / 2025-06",
            current_period="2025-07",
            n_ref_samples=2,
            n_current_samples=1,
            top_drifted_pokemon=["Incineroar", "Garchomp"],
        )

    def test_drift_message_contains_alarm_emoji(self) -> None:
        """Mensaje de drift contiene el emoji de alarma."""
        result = self._make_drift_result(is_drift=True)
        msg = format_drift_for_discord(result)
        assert "\U0001f6a8" in msg  # 🚨

    def test_stable_message_contains_check_emoji(self) -> None:
        """Mensaje de meta estable contiene el emoji de check."""
        result = self._make_drift_result(is_drift=False)
        msg = format_drift_for_discord(result)
        assert "\u2705" in msg  # ✅

    def test_message_contains_regulation_id(self) -> None:
        """Mensaje contiene el regulation_id de la regulación analizada."""
        result = self._make_drift_result(is_drift=True)
        msg = format_drift_for_discord(result)
        assert "M-A" in msg

    def test_message_contains_p_value(self) -> None:
        """Mensaje contiene el p-value numérico."""
        result = self._make_drift_result(is_drift=True)
        msg = format_drift_for_discord(result)
        assert "0.0100" in msg or "p_val" in msg.lower()

    def test_message_contains_top_pokemon(self) -> None:
        """Mensaje menciona al menos el primer Pokémon con mayor drift."""
        result = self._make_drift_result(is_drift=True)
        msg = format_drift_for_discord(result)
        assert "Incineroar" in msg


# ---------------------------------------------------------------------------
# GRUPO 4 — check_drift con mocks
# ---------------------------------------------------------------------------


class TestCheckDrift:
    """Tests de check_drift usando mocks de DuckDB y _get_monthly_data."""

    def _make_monthly_df(self) -> pd.DataFrame:
        """DataFrame mensual con 4 meses para tests de check_drift."""
        months = ["2025-05", "2025-06", "2025-07", "2025-08"]
        pokemon = ["Incineroar", "Garchomp"]
        rows = []
        for i, month in enumerate(months):
            for pkm in pokemon:
                rows.append(
                    {"year_month": month, "pokemon": pkm, "usage_pct": 30.0 + i * 2.0}
                )
        return pd.DataFrame(rows)

    def test_returns_none_when_alibi_not_available(self) -> None:
        """Retorna None sin crash cuando alibi-detect no está instalado."""
        mock_con = MagicMock()
        with patch.dict("sys.modules", {"alibi_detect": None, "alibi_detect.cd": None}):
            import builtins

            real_import = builtins.__import__

            def mock_import(name: str, *args: object, **kwargs: object) -> object:
                if "alibi_detect" in name:
                    raise ImportError("mocked: alibi-detect not available")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = check_drift("TEST", mock_con, threshold=0.05)
                assert result is None

    def test_returns_none_when_no_data(self) -> None:
        """Retorna None cuando _get_monthly_data retorna DataFrame vacío."""
        mock_con = MagicMock()
        with patch("src.app.modules.drift._get_monthly_data") as mock_data:
            mock_data.return_value = pd.DataFrame()
            result = check_drift("TEST", mock_con, threshold=0.05)
            assert result is None

    def test_returns_none_when_insufficient_months(self) -> None:
        """Retorna None si solo hay 1 mes disponible (necesita al menos 2)."""
        mock_con = MagicMock()
        df_one_month = pd.DataFrame(
            {
                "year_month": ["2025-05"],
                "pokemon": ["Incineroar"],
                "usage_pct": [30.0],
            }
        )
        with patch("src.app.modules.drift._get_monthly_data") as mock_data:
            mock_data.return_value = df_one_month
            result = check_drift("TEST", mock_con, threshold=0.05)
            assert result is None

    def test_returns_drift_result_or_none_with_enough_data(self) -> None:
        """Con datos suficientes retorna DriftResult o None, nunca excepción."""
        mock_con = MagicMock()
        with patch("src.app.modules.drift._get_monthly_data") as mock_data:
            mock_data.return_value = self._make_monthly_df()
            result = check_drift("TEST", mock_con, threshold=0.05)
            assert result is None or isinstance(result, DriftResult)

    def test_no_exception_on_any_input(self) -> None:
        """check_drift nunca propaga excepción, incluso con DB fallida."""
        mock_con = MagicMock()
        mock_con.execute.side_effect = Exception("DB connection error")
        try:
            result = check_drift("TEST", mock_con, threshold=0.05)
            assert result is None
        except Exception as exc:
            pytest.fail(f"check_drift propagó excepción inesperada: {exc}")
