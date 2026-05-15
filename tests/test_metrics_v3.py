"""
Tests unitarios para normalize_metric_cross_reg (T-138, Bloque 16).

Función pura: no usa DuckDB, solo pd.Series y pd.DataFrame.
Cubre todos los casos del contrato público según la agenda T-138.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app.metrics.advanced_metrics import normalize_metric_cross_reg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(values: list[float], reg_id: str) -> pd.Series:
    """Crea una Series con índice de pokemon_slugs."""
    slugs = [f"pkm_{i}" for i in range(len(values))]
    return pd.Series(values, index=slugs, name=reg_id)


# ---------------------------------------------------------------------------
# Casos de entrada inválida
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_dict_returns_empty_dataframe(self) -> None:
        result = normalize_metric_cross_reg({})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_dict_has_correct_columns(self) -> None:
        result = normalize_metric_cross_reg({})
        expected = {
            "pokemon_slug",
            "regulation_id",
            "raw_value",
            "normalized_value",
        }
        assert expected.issubset(set(result.columns))

    def test_invalid_method_raises_value_error(self) -> None:
        series = _series([1.0, 2.0, 3.0], "REG_A")
        with pytest.raises(ValueError, match="method"):
            normalize_metric_cross_reg({"REG_A": series}, method="invalid")

    @pytest.mark.parametrize(
        "bad_method",
        [
            "ZSCORE",
            "MinMax",
            "l2",
            "normalize",
            "",
        ],
    )
    def test_various_invalid_methods_raise(self, bad_method: str) -> None:
        series = _series([1.0, 2.0], "REG_A")
        with pytest.raises(ValueError):
            normalize_metric_cross_reg({"REG_A": series}, method=bad_method)


# ---------------------------------------------------------------------------
# Método zscore
# ---------------------------------------------------------------------------


class TestZscore:
    def test_mean_is_zero(self) -> None:
        """Media de normalized_value por regulación ≈ 0."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        series = _series(values, "REG_X")
        result = normalize_metric_cross_reg({"REG_X": series}, method="zscore")
        mean_norm = result["normalized_value"].mean()
        assert abs(mean_norm) < 1e-9

    def test_std_is_one(self) -> None:
        """Desviación estándar de normalized_value ≈ 1."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        series = _series(values, "REG_X")
        result = normalize_metric_cross_reg({"REG_X": series}, method="zscore")
        std_norm = result["normalized_value"].std(ddof=0)
        assert abs(std_norm - 1.0) < 1e-6

    def test_constant_series_returns_zeros(self) -> None:
        """Serie constante (std=0) → normalized_value todos 0."""
        values = [5.0, 5.0, 5.0, 5.0]
        series = _series(values, "REG_X")
        result = normalize_metric_cross_reg({"REG_X": series}, method="zscore")
        assert (result["normalized_value"] == 0.0).all()

    def test_zscore_is_default_method(self) -> None:
        """method='zscore' es el default."""
        values = [1.0, 2.0, 3.0]
        series = _series(values, "REG_X")
        result_default = normalize_metric_cross_reg({"REG_X": series})
        result_explicit = normalize_metric_cross_reg({"REG_X": series}, method="zscore")
        pd.testing.assert_frame_equal(
            result_default.reset_index(drop=True),
            result_explicit.reset_index(drop=True),
        )

    @pytest.mark.parametrize("reg_id", ["REG_A", "REG_B", "REG_C"])
    def test_zscore_works_for_any_reg_id(self, reg_id: str) -> None:
        """La función acepta cualquier regulation_id."""
        series = _series([1.0, 2.0, 3.0], reg_id)
        result = normalize_metric_cross_reg({reg_id: series}, method="zscore")
        assert (result["regulation_id"] == reg_id).all()


# ---------------------------------------------------------------------------
# Método minmax
# ---------------------------------------------------------------------------


class TestMinmax:
    def test_min_is_zero(self) -> None:
        """Mínimo de normalized_value por regulación = 0."""
        values = [3.0, 7.0, 1.0, 9.0, 5.0]
        series = _series(values, "REG_X")
        result = normalize_metric_cross_reg({"REG_X": series}, method="minmax")
        assert result["normalized_value"].min() == pytest.approx(0.0)

    def test_max_is_one(self) -> None:
        """Máximo de normalized_value por regulación = 1."""
        values = [3.0, 7.0, 1.0, 9.0, 5.0]
        series = _series(values, "REG_X")
        result = normalize_metric_cross_reg({"REG_X": series}, method="minmax")
        assert result["normalized_value"].max() == pytest.approx(1.0)

    def test_all_values_in_range(self) -> None:
        """Todos los valores normalizados en [0, 1]."""
        values = [-100.0, 0.0, 50.0, 200.0]
        series = _series(values, "REG_X")
        result = normalize_metric_cross_reg({"REG_X": series}, method="minmax")
        assert (result["normalized_value"] >= 0.0).all()
        assert (result["normalized_value"] <= 1.0).all()

    def test_constant_series_returns_zeros(self) -> None:
        """Serie constante (rango=0) → normalized_value todos 0."""
        values = [7.0, 7.0, 7.0]
        series = _series(values, "REG_X")
        result = normalize_metric_cross_reg({"REG_X": series}, method="minmax")
        assert (result["normalized_value"] == 0.0).all()

    def test_two_values_extremes(self) -> None:
        """Dos valores: el menor = 0, el mayor = 1."""
        series = pd.Series([10.0, 30.0], index=["pkm_a", "pkm_b"])
        result = normalize_metric_cross_reg({"REG_X": series}, method="minmax")
        norm_by_slug = result.set_index("pokemon_slug")["normalized_value"]
        assert norm_by_slug["pkm_a"] == pytest.approx(0.0)
        assert norm_by_slug["pkm_b"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Preservación de raw_value
# ---------------------------------------------------------------------------


class TestRawValuePreserved:
    @pytest.mark.parametrize("method", ["zscore", "minmax"])
    def test_raw_value_matches_input(self, method: str) -> None:
        """raw_value debe coincidir exactamente con el valor original."""
        values = [1.5, 3.0, 4.5, 6.0]
        series = _series(values, "REG_X")
        result = normalize_metric_cross_reg({"REG_X": series}, method=method)
        result_sorted = result.set_index("pokemon_slug").sort_index()
        expected = series.sort_index()
        np.testing.assert_allclose(
            result_sorted["raw_value"].values,
            expected.values,
            rtol=1e-9,
        )

    @pytest.mark.parametrize("method", ["zscore", "minmax"])
    def test_raw_value_unchanged_after_normalization(self, method: str) -> None:
        """Normalizar no muta raw_value."""
        values = [100.0, 200.0, 300.0]
        series = _series(values, "REG_X")
        result = normalize_metric_cross_reg({"REG_X": series}, method=method)
        raw_sorted = sorted(result["raw_value"].tolist())
        assert raw_sorted == pytest.approx(sorted(values))


# ---------------------------------------------------------------------------
# Soporte de cualquier nombre de métrica
# ---------------------------------------------------------------------------


class TestAnyMetricName:
    @pytest.mark.parametrize(
        "metric_name",
        [
            "mlwr",
            "spdo",
            "lre",
            "tai",
            "meit",
            "win_rate",
            "usage_pct",
            "custom_score_2026",
        ],
    )
    def test_any_series_name_is_accepted(self, metric_name: str) -> None:
        """La función acepta Series con cualquier nombre."""
        series = pd.Series(
            [0.1, 0.5, 0.9],
            index=["pkm_a", "pkm_b", "pkm_c"],
            name=metric_name,
        )
        result = normalize_metric_cross_reg({"REG_X": series}, method="zscore")
        assert not result.empty
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Comportamiento con múltiples regulaciones
# ---------------------------------------------------------------------------


class TestMultipleRegulations:
    def test_each_reg_normalized_independently(self) -> None:
        """Zscore se calcula por regulación, no globalmente."""
        # REG_A: valores altos, REG_B: valores bajos
        # Pero dentro de cada reg la media debe ser 0
        series_a = _series([100.0, 200.0, 300.0], "REG_A")
        series_b = _series([1.0, 2.0, 3.0], "REG_B")
        result = normalize_metric_cross_reg(
            {"REG_A": series_a, "REG_B": series_b},
            method="zscore",
        )
        for reg in ["REG_A", "REG_B"]:
            subset = result[result["regulation_id"] == reg]
            mean_norm = subset["normalized_value"].mean()
            assert abs(mean_norm) < 1e-9, f"Media de {reg} no es 0: {mean_norm}"

    def test_output_contains_all_regulations(self) -> None:
        """Todas las regulaciones aparecen en el resultado."""
        regs = ["REG_A", "REG_B", "REG_C"]
        metric = {reg: _series([1.0, 2.0, 3.0], reg) for reg in regs}
        result = normalize_metric_cross_reg(metric, method="minmax")
        assert set(result["regulation_id"].unique()) == set(regs)

    def test_empty_series_skipped(self) -> None:
        """Series vacía dentro del dict se ignora."""
        series_valid = _series([1.0, 2.0, 3.0], "REG_A")
        series_empty = pd.Series([], index=[], dtype=float)
        result = normalize_metric_cross_reg(
            {"REG_A": series_valid, "REG_B": series_empty},
            method="zscore",
        )
        assert "REG_B" not in result["regulation_id"].values
        assert "REG_A" in result["regulation_id"].values

    def test_row_count_equals_total_pokemon(self) -> None:
        """Número de filas = suma de pokémon por regulación."""
        metric = {
            "REG_A": _series([1.0, 2.0], "REG_A"),
            "REG_B": _series([3.0, 4.0, 5.0], "REG_B"),
        }
        result = normalize_metric_cross_reg(metric, method="minmax")
        assert len(result) == 5

    def test_sorted_by_regulation_then_normalized_value_desc(
        self,
    ) -> None:
        """Resultado ordenado por regulation_id asc, normalized_value desc."""
        metric = {
            "REG_B": _series([10.0, 30.0, 20.0], "REG_B"),
            "REG_A": _series([5.0, 1.0, 3.0], "REG_A"),
        }
        result = normalize_metric_cross_reg(metric, method="minmax")
        # regulation_id debe estar en orden ascendente
        reg_ids = result["regulation_id"].tolist()
        assert reg_ids == sorted(reg_ids)

        # Dentro de cada reg, normalized_value desc
        for reg in result["regulation_id"].unique():
            subset = result[result["regulation_id"] == reg]["normalized_value"].tolist()
            assert subset == sorted(
                subset, reverse=True
            ), f"REG {reg} no está ordenada desc"


# ---------------------------------------------------------------------------
# Test de no-hardcoding (principio rector del proyecto)
# ---------------------------------------------------------------------------


class TestNoHardcodedRegulation:
    def test_no_literal_regulation_id_in_source(self) -> None:
        """normalize_metric_cross_reg no hardcodea ningún reg_id."""
        import inspect

        from src.app.metrics import advanced_metrics

        source = inspect.getsource(advanced_metrics.normalize_metric_cross_reg)
        # Verificar que no hay literales de regulación conocidos
        for literal in ('"M-A"', "'M-A'", '"M-B"', "'M-B'"):
            assert literal not in source, (
                f"Literal {literal} hardcodeado en " f"normalize_metric_cross_reg"
            )
