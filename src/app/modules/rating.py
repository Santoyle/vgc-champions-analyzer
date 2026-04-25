"""
Implementación de Glicko-2 en stdlib pura (sin numpy ni scipy).

Glicko-2 es el sistema de rating usado por Smogon y plataformas
competitivas. Mejora sobre ELO añadiendo un parámetro de volatilidad
(σ) que captura cuán predecible es el rendimiento del jugador.

Los ratings se usan para ponderar datos de ladder por calidad del
jugador: una partida entre dos jugadores con rating alto (~1800+)
tiene más peso analítico que una partida de bajo nivel.

DEFAULT_RATING=1500 sigue la escala ELO estándar. Internamente,
Glicko-2 trabaja en una escala reducida centrada en 0 usando
SCALE=173.7178 como factor de conversión.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes del sistema
# ---------------------------------------------------------------------------

GLICKO2_TAU: float = 0.5
GLICKO2_EPSILON: float = 1e-6
DEFAULT_RATING: float = 1500.0
DEFAULT_RD: float = 350.0
DEFAULT_VOLATILITY: float = 0.06
SCALE: float = 173.7178


# ---------------------------------------------------------------------------
# Dataclass principal
# ---------------------------------------------------------------------------


@dataclass
class PlayerRating:
    """
    Rating Glicko-2 de un jugador.

    Los valores mu y phi se almacenan en escala Glicko-2 interna.
    Los métodos rating y rd los convierten a escala ELO para display.

    Attributes:
        player_name: Nombre del jugador en Showdown.
        regulation_id: Regulación para la que aplica.
        mu: Rating en escala Glicko-2 (0 = 1500 ELO).
        phi: Desviación en escala Glicko-2.
        sigma: Volatilidad.
        n_games: Número de partidas jugadas.
        last_updated: Fecha de última actualización.
    """

    player_name: str
    regulation_id: str
    mu: float = 0.0
    phi: float = 2.014
    sigma: float = DEFAULT_VOLATILITY
    n_games: int = 0
    last_updated: date = field(default_factory=date.today)

    @property
    def rating(self) -> float:
        """Rating en escala ELO (centrado en 1500)."""
        return self.mu * SCALE + DEFAULT_RATING

    @property
    def rd(self) -> float:
        """Rating deviation en escala ELO."""
        return self.phi * SCALE

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Intervalo de confianza 95% en escala ELO."""
        return (self.rating - 2 * self.rd, self.rating + 2 * self.rd)


# ---------------------------------------------------------------------------
# Funciones auxiliares del algoritmo Glicko-2
# ---------------------------------------------------------------------------


def _g(phi: float) -> float:
    """Función g(φ) del algoritmo Glicko-2."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """
    Función E(μ, μⱼ, φⱼ) — probabilidad esperada de victoria
    contra el oponente j.
    """
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _f(
    x: float,
    delta: float,
    phi: float,
    v: float,
    a: float,
) -> float:
    """
    Función f(x) para el algoritmo iterativo de actualización de σ.
    """
    ex = math.exp(x)
    num1 = ex * (delta**2 - phi**2 - v - ex)
    den1 = 2.0 * (phi**2 + v + ex) ** 2
    num2 = x - a
    den2 = GLICKO2_TAU**2
    return num1 / den1 - num2 / den2


# ---------------------------------------------------------------------------
# Algoritmo principal
# ---------------------------------------------------------------------------


def update_rating(
    player: PlayerRating,
    opponents: list[PlayerRating],
    scores: list[float],
) -> PlayerRating:
    """
    Actualiza el rating de un jugador según sus resultados contra
    una lista de oponentes.

    Implementa el Algoritmo Glicko-2 completo (Glickman 2012),
    pasos 1-7.

    Args:
        player: PlayerRating actual del jugador.
        opponents: Lista de PlayerRating de oponentes.
        scores: Lista de scores (1.0=victoria, 0.5=empate,
                0.0=derrota). Debe tener el mismo largo que opponents.

    Returns:
        Nuevo PlayerRating actualizado.

    Raises:
        ValueError: Si opponents y scores tienen distinto largo
                    o están vacíos.
    """
    if not opponents:
        raise ValueError("opponents no puede estar vacío")
    if len(opponents) != len(scores):
        raise ValueError(
            f"opponents ({len(opponents)}) y scores ({len(scores)}) "
            "deben tener el mismo largo"
        )

    mu = player.mu
    phi = player.phi
    sigma = player.sigma

    # Paso 2: Calcular v (varianza estimada)
    v_inv = sum(
        _g(opp.phi) ** 2
        * _E(mu, opp.mu, opp.phi)
        * (1 - _E(mu, opp.mu, opp.phi))
        for opp in opponents
    )
    v = float("inf") if v_inv == 0 else 1.0 / v_inv

    # Paso 3: Calcular delta (mejora estimada)
    delta = v * sum(
        _g(opp.phi) * (score - _E(mu, opp.mu, opp.phi))
        for opp, score in zip(opponents, scores)
    )

    # Paso 4: Actualizar σ con algoritmo iterativo (Illinois method)
    a = math.log(sigma**2)
    A = a

    if delta**2 > phi**2 + v:
        B = math.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while _f(a - k * abs(GLICKO2_TAU), delta, phi, v, a) < 0:
            k += 1
        B = a - k * abs(GLICKO2_TAU)

    fA = _f(A, delta, phi, v, a)
    fB = _f(B, delta, phi, v, a)

    while abs(B - A) > GLICKO2_EPSILON:
        C = A + (A - B) * fA / (fB - fA)
        fC = _f(C, delta, phi, v, a)
        if fC * fB <= 0:
            A = B
            fA = fB
        else:
            fA /= 2.0
        B = C
        fB = fC

    sigma_new = math.exp(A / 2.0)

    # Paso 5: Actualizar φ pre-rating
    phi_star = math.sqrt(phi**2 + sigma_new**2)

    # Paso 6: Actualizar μ y φ
    phi_new = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
    mu_new = mu + phi_new**2 * sum(
        _g(opp.phi) * (score - _E(mu, opp.mu, opp.phi))
        for opp, score in zip(opponents, scores)
    )

    # Paso 7: Retornar rating actualizado
    return PlayerRating(
        player_name=player.player_name,
        regulation_id=player.regulation_id,
        mu=mu_new,
        phi=phi_new,
        sigma=sigma_new,
        n_games=player.n_games + len(opponents),
        last_updated=date.today(),
    )


# ---------------------------------------------------------------------------
# Procesamiento por lotes
# ---------------------------------------------------------------------------


def update_ratings_from_replays(
    replays: list[Any],
    regulation_id: str,
    existing_ratings: dict[str, PlayerRating] | None = None,
) -> dict[str, PlayerRating]:
    """
    Actualiza ratings de todos los jugadores a partir de una lista
    de replays parseados.

    Los replays deben tener atributos p1, p2, winner. El winner se
    usa para determinar scores:
      - winner == p1 → p1 gana (1.0), p2 pierde (0.0)
      - winner == p2 → p2 gana (1.0), p1 pierde (0.0)
      - winner es None → empate (0.5 cada uno)

    Args:
        replays: Lista de objetos con atributos p1, p2, winner.
        regulation_id: ID de la regulación.
        existing_ratings: Ratings previos para inicializar jugadores
                          ya vistos. None = partir de cero.

    Returns:
        Dict {player_name: PlayerRating} actualizado.
    """
    ratings: dict[str, PlayerRating] = (
        existing_ratings.copy() if existing_ratings else {}
    )

    def _get_or_create(name: str) -> PlayerRating:
        if name not in ratings:
            ratings[name] = PlayerRating(
                player_name=name,
                regulation_id=regulation_id,
            )
        return ratings[name]

    for replay in replays:
        p1_name = str(getattr(replay, "p1", ""))
        p2_name = str(getattr(replay, "p2", ""))
        winner = getattr(replay, "winner", None)

        if not p1_name or not p2_name:
            continue

        p1 = _get_or_create(p1_name)
        p2 = _get_or_create(p2_name)

        if winner == p1_name:
            score_p1, score_p2 = 1.0, 0.0
        elif winner == p2_name:
            score_p1, score_p2 = 0.0, 1.0
        else:
            score_p1, score_p2 = 0.5, 0.5

        try:
            ratings[p1_name] = update_rating(p1, [p2], [score_p1])
            ratings[p2_name] = update_rating(p2, [p1], [score_p2])
        except (ValueError, ZeroDivisionError) as exc:
            log.debug(
                "Error actualizando rating %s vs %s: %s",
                p1_name,
                p2_name,
                exc,
            )
            continue

    return ratings


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


def persist_ratings(
    ratings: dict[str, PlayerRating],
    con: sqlite3.Connection,
) -> int:
    """
    Persiste los ratings en la tabla player_ratings de SQLite
    (creada si no existe).

    Operación idempotente: usa ON CONFLICT DO UPDATE (upsert).

    Args:
        ratings: Dict de ratings a persistir.
        con: Conexión SQLite activa.

    Returns:
        Número de ratings insertados/actualizados.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS player_ratings (
            player_name    TEXT NOT NULL,
            regulation_id  TEXT NOT NULL,
            mu             REAL NOT NULL,
            phi            REAL NOT NULL,
            sigma          REAL NOT NULL,
            n_games        INTEGER NOT NULL,
            rating_elo     REAL NOT NULL,
            rd_elo         REAL NOT NULL,
            last_updated   TEXT NOT NULL,
            PRIMARY KEY (player_name, regulation_id)
        )
    """)

    rows = [
        (
            r.player_name,
            r.regulation_id,
            r.mu,
            r.phi,
            r.sigma,
            r.n_games,
            r.rating,
            r.rd,
            r.last_updated.isoformat(),
        )
        for r in ratings.values()
    ]

    con.executemany(
        """
        INSERT INTO player_ratings
            (player_name, regulation_id, mu, phi,
             sigma, n_games, rating_elo, rd_elo, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_name, regulation_id) DO UPDATE SET
            mu           = excluded.mu,
            phi          = excluded.phi,
            sigma        = excluded.sigma,
            n_games      = excluded.n_games,
            rating_elo   = excluded.rating_elo,
            rd_elo       = excluded.rd_elo,
            last_updated = excluded.last_updated
        """,
        rows,
    )

    con.commit()
    log.info(
        "Persistidos %d ratings para regulaciones: %s",
        len(rows),
        list({r.regulation_id for r in ratings.values()}),
    )
    return len(rows)


__all__ = [
    "PlayerRating",
    "update_rating",
    "update_ratings_from_replays",
    "persist_ratings",
    "DEFAULT_RATING",
    "DEFAULT_RD",
]
