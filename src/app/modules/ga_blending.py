"""
Blending bayesiano λ para el GA NSGA-II de VGC.

Combina el fitness calculado desde datos reales de la regulación activa
(data_score) con un prior histórico basado en features agnósticas de
regulación (prior_score) mediante la fórmula:

    score_final = (1 - λ) * data_score + λ * prior_score

El coeficiente λ se calcula como:

    λ = max(LAMBDA_MIN, LAMBDA_MAX * exp(-n / LAMBDA_DECAY))
      = max(0.05, 0.8 * exp(-n / 2000))

donde n es el número de replays disponibles para la regulación activa.

Comportamiento de λ:
  - λ alto (→ 0.80): pocos datos → confiar en el prior histórico.
  - λ bajo (→ 0.05): muchos replays → confiar en los datos actuales.
  - LAMBDA_MIN = 0.05 garantiza que el prior nunca desaparece completamente,
    preservando conocimiento táctico incluso con grandes volúmenes de datos.

Esta transición suave permite que el GA use bien el conocimiento histórico
cuando la regulación es nueva y los datos son escasos, sin forzar un corte
abrupto cuando los datos crecen.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from src.app.modules.ga import (
    Chromosome,
    _load_pokemon_master,
)
from src.app.modules.ga_fitness import (
    META_FALLBACK_TOP20,
    evaluate_fitness,
)
from src.app.modules.ga_warmstart import (
    extract_agnostic_features,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes del blending
# ---------------------------------------------------------------------------

LAMBDA_MAX: float = 0.8      # λ máximo (n=0, sin datos)
LAMBDA_MIN: float = 0.05     # λ mínimo (floor; prior nunca desaparece)
LAMBDA_DECAY: float = 2000.0  # n en el que λ ≈ 0.15


# ---------------------------------------------------------------------------
# Cálculo de λ
# ---------------------------------------------------------------------------


def compute_lambda(n_replays: int) -> float:
    """
    Calcula el coeficiente de blending λ según el número de replays
    disponibles para la regulación activa.

    Fórmula: λ = max(LAMBDA_MIN, LAMBDA_MAX * exp(-n / LAMBDA_DECAY))

    Args:
        n_replays: Número de replays de la regulación activa. Clamp a 0 si < 0.

    Returns:
        λ como float en [LAMBDA_MIN, LAMBDA_MAX].
    """
    if n_replays < 0:
        n_replays = 0
    lam = LAMBDA_MAX * math.exp(-n_replays / LAMBDA_DECAY)
    return float(max(LAMBDA_MIN, lam))


# ---------------------------------------------------------------------------
# Resultado del blending
# ---------------------------------------------------------------------------


@dataclass
class BlendedFitness:
    """
    Fitness blended de un Chromosome, combinación de data_score y prior_score.

    Attributes:
        f1: Cobertura defensiva blended [0, 1].
        f2: Sinergia ofensiva blended [0, 1].
        f3: Anti-meta blended [0, 1].
        f4: Control de velocidad blended [0, 1].
        lambda_used: Valor de λ efectivamente usado.
        n_replays: n usado para calcular λ.
        data_weight: 1 - λ (peso de los datos actuales).
        prior_weight: λ (peso del prior histórico).
    """

    f1: float
    f2: float
    f3: float
    f4: float
    lambda_used: float
    n_replays: int
    data_weight: float
    prior_weight: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        """
        Retorna los 4 fitness como tupla compatible con DEAP.

        El orden coincide con evaluate_fitness(): (f1, f2, f3, f4).
        """
        return self.f1, self.f2, self.f3, self.f4


# ---------------------------------------------------------------------------
# Prior score
# ---------------------------------------------------------------------------


def _compute_prior_score(
    chrom: Chromosome,
    pokemon_master: dict[int, dict[str, Any]],
) -> tuple[float, float, float, float]:
    """
    Calcula el prior score de un Chromosome basado únicamente en sus
    features agnósticas de regulación.

    No usa datos de uso de ninguna regulación específica, por lo que es
    válido incluso cuando la regulación activa tiene 0 replays.

    Mapeo de features agnósticas a los 4 objetivos:
      f1_prior = (has_redirection + type_diversity) / 2
      f2_prior = (has_fake_out   + type_diversity) / 2
      f3_prior = (has_intimidate + has_weather)    / 2
      f4_prior = (mean_speed     + speed_variance) / 2

    Args:
        chrom: Chromosome a evaluar.
        pokemon_master: Datos maestros (dict[dex_id, data]).

    Returns:
        Tupla (f1, f2, f3, f4) en [0, 1].
    """
    features = extract_agnostic_features(chrom, pokemon_master)

    f1_prior = float(
        (features.get("has_redirection", 0.0) + features.get("type_diversity", 0.0))
        / 2.0
    )
    f2_prior = float(
        (features.get("has_fake_out", 0.0) + features.get("type_diversity", 0.0))
        / 2.0
    )
    f3_prior = float(
        (features.get("has_intimidate", 0.0) + features.get("has_weather", 0.0))
        / 2.0
    )
    f4_prior = float(
        (features.get("mean_speed", 0.0) + features.get("speed_variance", 0.0))
        / 2.0
    )

    return f1_prior, f2_prior, f3_prior, f4_prior


# ---------------------------------------------------------------------------
# Función principal de blending
# ---------------------------------------------------------------------------


def blended_fitness(
    chrom: Chromosome,
    n_replays: int,
    pokemon_master: dict[int, dict[str, Any]] | None = None,
    meta_top: list[str] | None = None,
) -> BlendedFitness:
    """
    Calcula el fitness blended de un Chromosome combinando data_score y
    prior_score con el coeficiente λ.

        score_final_i = (1 - λ) * data_i + λ * prior_i

    Nunca lanza excepción: si evaluate_fitness falla, usa (0, 0, 0, 0)
    como data_score y el resultado final queda dominado por el prior.

    Args:
        chrom: Chromosome a evaluar.
        n_replays: Número de replays de la regulación activa (para λ).
        pokemon_master: Datos maestros. Carga desde disco si None.
        meta_top: Top Pokémon del meta para f3. None = META_FALLBACK_TOP20.

    Returns:
        BlendedFitness con los 4 valores blended y metadatos del blending.
    """
    pm: dict[int, dict[str, Any]] = (
        _load_pokemon_master() if pokemon_master is None else pokemon_master  # type: ignore[assignment]
    )
    if meta_top is None:
        meta_top = META_FALLBACK_TOP20

    lam = compute_lambda(n_replays)
    data_w = 1.0 - lam

    # Data score (regulación activa)
    try:
        d1, d2, d3, d4 = evaluate_fitness(chrom, pm, meta_top)
    except Exception as exc:  # noqa: BLE001
        log.debug("Error en data fitness: %s", exc)
        d1 = d2 = d3 = d4 = 0.0

    # Prior score (features agnósticas, sin datos de regulación)
    try:
        p1, p2, p3, p4 = _compute_prior_score(chrom, pm)
    except Exception as exc:  # noqa: BLE001
        log.debug("Error en prior fitness: %s", exc)
        p1 = p2 = p3 = p4 = 0.0

    # Blending
    f1 = data_w * d1 + lam * p1
    f2 = data_w * d2 + lam * p2
    f3 = data_w * d3 + lam * p3
    f4 = data_w * d4 + lam * p4

    log.debug(
        "Blended fitness (λ=%.3f, n=%d): f1=%.3f f2=%.3f f3=%.3f f4=%.3f",
        lam,
        n_replays,
        f1,
        f2,
        f3,
        f4,
    )

    return BlendedFitness(
        f1=round(f1, 4),
        f2=round(f2, 4),
        f3=round(f3, 4),
        f4=round(f4, 4),
        lambda_used=lam,
        n_replays=n_replays,
        data_weight=data_w,
        prior_weight=lam,
    )


# ---------------------------------------------------------------------------
# Utilidad: contar replays
# ---------------------------------------------------------------------------


def count_replays_for_regulation(regulation_id: str) -> int:
    """
    Cuenta el número de replays (filas) disponibles en
    data/raw/reg={id}/source=showdown/ para determinar n_replays.

    Importa pandas de forma lazy para no añadir overhead al import del módulo.

    Args:
        regulation_id: ID de la regulación activa (ej: "M-A").

    Returns:
        Total de filas en todos los Parquets de Showdown para esa regulación.
        0 si no hay datos o la carpeta no existe.
    """
    import pandas as pd
    from pathlib import Path

    raw_dir = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "raw"
        / f"reg={regulation_id}"
        / "source=showdown"
    )

    if not raw_dir.exists():
        return 0

    total = 0
    for pq_file in raw_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq_file)
            total += len(df)
        except Exception:  # noqa: BLE001
            continue

    log.debug(
        "count_replays_for_regulation(%s) = %d",
        regulation_id,
        total,
    )
    return total


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "compute_lambda",
    "blended_fitness",
    "BlendedFitness",
    "count_replays_for_regulation",
    "LAMBDA_MAX",
    "LAMBDA_MIN",
    "LAMBDA_DECAY",
]
