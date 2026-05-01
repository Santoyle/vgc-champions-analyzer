"""
Tests unitarios para métricas MVP del módulo `src.app.modules.metrics`.

Grupos:
1) Shrinkage beta matemática (`_apply_beta_shrinkage`).
2) EI (Efficiency Item).
3) STC (Speed Tier Control).
4) TPI (Threat Pressure Index).

Todos los tests usan DataFrames sintéticos en memoria (sin Parquets ni DuckDB).
La única lectura opcional de disco es `test_load_speed_map_incineroar_speed`,
que usa `pytest.skip` si no existe `pokemon_master.json`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app.modules.metrics import (
    BETA_ALPHA,
    BETA_BETA,
    PRIOR_MEAN,
    EIResult,
    STCResult,
    TPIResult,
    _apply_beta_shrinkage,
    _load_speed_map,
    compute_ei,
    compute_stc_meta,
    compute_stc_pokemon,
    compute_tpi,
    get_top_ei_items,
)


@pytest.fixture
def sample_items_df() -> pd.DataFrame:
    """
    DataFrame de ítems sintético que simula el output de create_items_by_pkm().
    3 Pokémon con 3 ítems cada uno.
    """
    return pd.DataFrame(
        {
            "regulation_id": ["TEST"] * 9,
            "pokemon": [
                "Incineroar",
                "Incineroar",
                "Incineroar",
                "Garchomp",
                "Garchomp",
                "Garchomp",
                "Sneasler",
                "Sneasler",
                "Sneasler",
            ],
            "item": [
                "Sitrus Berry",
                "Assault Vest",
                "Choice Scarf",
                "Choice Scarf",
                "Life Orb",
                "Focus Sash",
                "Focus Sash",
                "Choice Scarf",
                "Life Orb",
            ],
            "avg_pct": [45.0, 30.0, 15.0, 40.0, 35.0, 20.0, 50.0, 25.0, 20.0],
            "n_months_seen": [4, 4, 3, 4, 4, 2, 4, 3, 3],
        }
    )


@pytest.fixture
def sample_usage_df() -> pd.DataFrame:
    """
    DataFrame de uso sintético que simula el output de create_usage_by_reg().
    """
    return pd.DataFrame(
        {
            "regulation_id": ["TEST"] * 5,
            "pokemon": ["Incineroar", "Sneasler", "Garchomp", "Sinistcha", "Kingambit"],
            "avg_usage_pct": [54.0, 45.0, 37.0, 34.0, 27.0],
            "n_months": [4, 4, 4, 4, 4],
        }
    )


@pytest.fixture
def sample_teammates_df() -> pd.DataFrame:
    """
    DataFrame de teammates sintético que simula el output de
    create_teammates_by_pkm().
    """
    return pd.DataFrame(
        {
            "regulation_id": ["TEST"] * 6,
            "pokemon": ["Incineroar", "Incineroar", "Garchomp", "Garchomp", "Sneasler", "Sneasler"],
            "teammate": ["Garchomp", "Sneasler", "Incineroar", "Sinistcha", "Incineroar", "Garchomp"],
            "avg_correlation": [45.0, 35.0, 45.0, 20.0, 35.0, 30.0],
            "n_months_seen": [4, 4, 4, 4, 4, 4],
        }
    )


@pytest.fixture
def speed_map() -> dict[str, int]:
    """Mapa de velocidades sintético para tests."""
    return {
        "incineroar": 60,
        "garchomp": 102,
        "sneasler": 120,
        "sinistcha": 70,
        "kingambit": 50,
    }


class TestBetaShrinkage:
    """Tests matemáticos de la shrinkage beta."""

    def test_zero_months_returns_prior_mean(self) -> None:
        """n=0 retorna PRIOR_MEAN * 100 = 50.0."""
        result = _apply_beta_shrinkage(80.0, 0)
        assert result == pytest.approx(PRIOR_MEAN * 100.0, abs=0.01)

    def test_shrinkage_pulls_toward_prior(self) -> None:
        """
        Con pocos meses, el resultado se aleja de observed y se regulariza
        fuertemente (según la fórmula implementada).
        """
        observed = 80.0
        shrunk = _apply_beta_shrinkage(observed, 1)
        assert 0.0 <= shrunk < observed

    def test_many_months_approaches_observed(self) -> None:
        """Con muchos meses, shrunk ≈ observed."""
        observed = 80.0
        shrunk = _apply_beta_shrinkage(observed, 1000)
        assert shrunk == pytest.approx(observed, abs=2.0)

    def test_output_in_valid_range(self) -> None:
        """Output siempre en [0, 100]."""
        for obs in [0.0, 50.0, 100.0]:
            for n in [0, 1, 5, 20, 100]:
                result = _apply_beta_shrinkage(obs, n)
                assert 0.0 <= result <= 100.0

    def test_formula_correctness(self) -> None:
        """Verifica fórmula manual para n=4."""
        obs = 80.0
        n = 4
        obs_norm = obs / 100.0
        expected = (BETA_ALPHA * PRIOR_MEAN + n * obs_norm) / (BETA_ALPHA + BETA_BETA + n) * 100.0
        result = _apply_beta_shrinkage(obs, n)
        assert result == pytest.approx(expected, abs=0.01)

    def test_shrinkage_preserves_direction(self) -> None:
        high = _apply_beta_shrinkage(80.0, 4)
        low = _apply_beta_shrinkage(20.0, 4)
        assert high > low


class TestComputeEI:
    """Tests para la métrica EI."""

    def test_returns_dataframe(self, sample_items_df: pd.DataFrame) -> None:
        """compute_ei retorna DataFrame."""
        df = compute_ei(sample_items_df, "TEST")
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, sample_items_df: pd.DataFrame) -> None:
        """DataFrame tiene las columnas esperadas."""
        df = compute_ei(sample_items_df, "TEST")
        required = {
            "pokemon",
            "item",
            "regulation_id",
            "raw_pct",
            "shrunk_pct",
            "baseline_pct",
            "ei_score",
            "n_months",
            "is_baseline",
        }
        assert required.issubset(set(df.columns))

    def test_baseline_has_zero_ei(self, sample_items_df: pd.DataFrame) -> None:
        """El ítem baseline de cada Pokémon tiene EI = 0.0."""
        df = compute_ei(sample_items_df, "TEST")
        baselines = df[df["is_baseline"] == True]
        for _, row in baselines.iterrows():
            assert row["ei_score"] == pytest.approx(0.0, abs=0.001)

    def test_baseline_is_most_used_item(self, sample_items_df: pd.DataFrame) -> None:
        """El baseline de Incineroar es Sitrus Berry (45% — el más usado)."""
        df = compute_ei(sample_items_df, "TEST")
        inc_baseline = df[(df["pokemon"] == "Incineroar") & (df["is_baseline"] == True)]
        assert len(inc_baseline) == 1
        assert inc_baseline.iloc[0]["item"] == "Sitrus Berry"

    def test_empty_df_returns_empty(self) -> None:
        """DataFrame vacío retorna DataFrame vacío."""
        df = compute_ei(pd.DataFrame(), "TEST")
        assert df.empty

    def test_wrong_regulation_returns_empty(self, sample_items_df: pd.DataFrame) -> None:
        """Regulación inexistente retorna vacío."""
        df = compute_ei(sample_items_df, "NONEXISTENT")
        assert df.empty

    def test_ei_score_is_float(self, sample_items_df: pd.DataFrame) -> None:
        """ei_score es float en todos los registros."""
        df = compute_ei(sample_items_df, "TEST")
        assert np.issubdtype(df["ei_score"].dtype, np.floating)

    def test_get_top_ei_items_excludes_baseline(self, sample_items_df: pd.DataFrame) -> None:
        """get_top_ei_items no incluye baselines."""
        df_ei = compute_ei(sample_items_df, "TEST")
        df_top = get_top_ei_items(df_ei, top_n=20)
        if not df_top.empty:
            assert not df_top["is_baseline"].any()

    def test_get_top_ei_items_respects_min_score(self, sample_items_df: pd.DataFrame) -> None:
        """Filtra correctamente por min_ei_score."""
        df_ei = compute_ei(sample_items_df, "TEST")
        min_score = 5.0
        df_top = get_top_ei_items(df_ei, top_n=20, min_ei_score=min_score)
        if not df_top.empty:
            assert (df_top["ei_score"] >= min_score).all()

    def test_get_top_ei_items_respects_top_n(self, sample_items_df: pd.DataFrame) -> None:
        """Respeta el límite top_n."""
        df_ei = compute_ei(sample_items_df, "TEST")
        df_top = get_top_ei_items(df_ei, top_n=2)
        assert len(df_top) <= 2

    def test_eiresult_dataclass_instantiation(self) -> None:
        """EIResult puede instanciarse con tipos correctos."""
        r = EIResult("Pkm", "Item", "TEST", 10.0, 9.0, 9.0, 0.0, 2, True)
        assert isinstance(r, EIResult)


class TestComputeSTC:
    """Tests para la métrica STC."""

    def test_compute_stc_meta_returns_dataframe(
        self,
        sample_usage_df: pd.DataFrame,
        speed_map: dict[str, int],
    ) -> None:
        """compute_stc_meta retorna DataFrame."""
        df = compute_stc_meta(sample_usage_df, "TEST", speed_map)
        assert isinstance(df, pd.DataFrame)

    def test_stc_has_required_columns(
        self,
        sample_usage_df: pd.DataFrame,
        speed_map: dict[str, int],
    ) -> None:
        """DataFrame tiene columnas esperadas."""
        df = compute_stc_meta(sample_usage_df, "TEST", speed_map)
        required = {"pokemon", "base_speed", "stc_score", "n_faster_than"}
        if not df.empty:
            assert required.issubset(set(df.columns))

    def test_stc_scores_in_range(
        self,
        sample_usage_df: pd.DataFrame,
        speed_map: dict[str, int],
    ) -> None:
        """stc_score en [0, 1]."""
        df = compute_stc_meta(sample_usage_df, "TEST", speed_map)
        if not df.empty:
            assert (df["stc_score"] >= 0.0).all()
            assert (df["stc_score"] <= 1.0).all()

    def test_faster_pokemon_has_higher_stc(
        self,
        sample_usage_df: pd.DataFrame,
        speed_map: dict[str, int],
    ) -> None:
        """Pokémon más rápido tiene STC mayor."""
        df = compute_stc_meta(sample_usage_df, "TEST", speed_map)
        if df.empty:
            pytest.skip("Sin datos STC")
        sneasler = df[df["pokemon"] == "Sneasler"]
        incineroar = df[df["pokemon"] == "Incineroar"]
        if not sneasler.empty and not incineroar.empty:
            assert sneasler.iloc[0]["stc_score"] > incineroar.iloc[0]["stc_score"]

    def test_ordered_by_stc_descending(
        self,
        sample_usage_df: pd.DataFrame,
        speed_map: dict[str, int],
    ) -> None:
        """Resultado ordenado por stc_score desc."""
        df = compute_stc_meta(sample_usage_df, "TEST", speed_map)
        if len(df) >= 2:
            scores = df["stc_score"].tolist()
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_empty_usage_returns_empty(self, speed_map: dict[str, int]) -> None:
        """DataFrame de uso vacío retorna vacío."""
        df = compute_stc_meta(pd.DataFrame(), "TEST", speed_map)
        assert df.empty

    def test_compute_stc_pokemon_returns_result(
        self,
        sample_usage_df: pd.DataFrame,
        speed_map: dict[str, int],
    ) -> None:
        """compute_stc_pokemon retorna STCResult."""
        result = compute_stc_pokemon("Sneasler", "TEST", sample_usage_df, speed_map)
        assert result is not None
        assert isinstance(result, STCResult)

    def test_compute_stc_pokemon_unknown_returns_none(
        self,
        sample_usage_df: pd.DataFrame,
        speed_map: dict[str, int],
    ) -> None:
        """Pokémon desconocido retorna None."""
        result = compute_stc_pokemon("UnknownPkm", "TEST", sample_usage_df, speed_map)
        assert result is None

    def test_load_speed_map_returns_dict(self) -> None:
        """_load_speed_map retorna dict (o vacío si no existe archivo)."""
        speed = _load_speed_map()
        assert isinstance(speed, dict)

    def test_load_speed_map_incineroar_speed(self) -> None:
        """Incineroar tiene velocidad 60 en pokemon_master real."""
        speed = _load_speed_map()
        if not speed:
            pytest.skip("pokemon_master.json no disponible")
        assert speed.get("incineroar", 0) == 60


class TestComputeTPI:
    """Tests para la métrica TPI."""

    def test_compute_tpi_returns_dataframe(
        self,
        sample_teammates_df: pd.DataFrame,
        sample_usage_df: pd.DataFrame,
    ) -> None:
        """compute_tpi retorna DataFrame."""
        df = compute_tpi(sample_teammates_df, sample_usage_df, "TEST")
        assert isinstance(df, pd.DataFrame)

    def test_tpi_has_required_columns(
        self,
        sample_teammates_df: pd.DataFrame,
        sample_usage_df: pd.DataFrame,
    ) -> None:
        """DataFrame tiene columnas esperadas."""
        df = compute_tpi(sample_teammates_df, sample_usage_df, "TEST")
        required = {
            "pokemon",
            "regulation_id",
            "tpi_score",
            "n_teammates",
            "top_teammate",
            "avg_co_usage",
        }
        if not df.empty:
            assert required.issubset(set(df.columns))

    def test_tpi_scores_in_range(
        self,
        sample_teammates_df: pd.DataFrame,
        sample_usage_df: pd.DataFrame,
    ) -> None:
        """tpi_score en [0, 1]."""
        df = compute_tpi(sample_teammates_df, sample_usage_df, "TEST")
        if not df.empty:
            assert (df["tpi_score"] >= 0.0).all()
            assert (df["tpi_score"] <= 1.0).all()

    def test_tpi_ordered_descending(
        self,
        sample_teammates_df: pd.DataFrame,
        sample_usage_df: pd.DataFrame,
    ) -> None:
        """Resultado ordenado por tpi_score desc."""
        df = compute_tpi(sample_teammates_df, sample_usage_df, "TEST")
        if len(df) >= 2:
            scores = df["tpi_score"].tolist()
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_tpi_empty_teammates_returns_empty(self, sample_usage_df: pd.DataFrame) -> None:
        """teammates vacío retorna DataFrame vacío."""
        df = compute_tpi(pd.DataFrame(), sample_usage_df, "TEST")
        assert df.empty

    def test_tpi_empty_usage_returns_empty(self, sample_teammates_df: pd.DataFrame) -> None:
        """usage vacío retorna DataFrame vacío."""
        df = compute_tpi(sample_teammates_df, pd.DataFrame(), "TEST")
        assert df.empty

    def test_tpi_wrong_regulation_returns_empty(
        self,
        sample_teammates_df: pd.DataFrame,
        sample_usage_df: pd.DataFrame,
    ) -> None:
        """Regulación inexistente retorna vacío."""
        df = compute_tpi(sample_teammates_df, sample_usage_df, "NONEXISTENT")
        assert df.empty

    def test_tpi_n_teammates_correct(
        self,
        sample_teammates_df: pd.DataFrame,
        sample_usage_df: pd.DataFrame,
    ) -> None:
        """n_teammates refleja el número de compañeros del DataFrame."""
        df = compute_tpi(sample_teammates_df, sample_usage_df, "TEST")
        if not df.empty:
            inc = df[df["pokemon"] == "Incineroar"]
            if not inc.empty:
                assert inc.iloc[0]["n_teammates"] == 2

    def test_tpi_top_teammate_is_string(
        self,
        sample_teammates_df: pd.DataFrame,
        sample_usage_df: pd.DataFrame,
    ) -> None:
        """top_teammate es string no vacío."""
        df = compute_tpi(sample_teammates_df, sample_usage_df, "TEST")
        if not df.empty:
            assert df["top_teammate"].dtype == object
            assert (df["top_teammate"].str.len() > 0).all()

    def test_tpiresult_dataclass_instantiation(self) -> None:
        """TPIResult puede instanciarse con tipos correctos."""
        r = TPIResult("Pkm", "TEST", 0.5, 2, "Mate", 30.0, 0.5)
        assert isinstance(r, TPIResult)
