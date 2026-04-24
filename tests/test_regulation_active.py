"""
Tests para get_active_regulation() cubriendo los 4 escenarios
(A, A', B, C) y sus edge cases.

Principios de aislamiento:
  - today es siempre inyectado explícitamente — ningún test llama
    date.today() directamente, garantizando determinismo total.
  - Las regulaciones se crean en tmp_path (directorio temporal de pytest)
    en lugar de usar regulations/M-A.json real, evitando acoplamiento
    al contenido del repositorio.
  - _write_reg_json calcula el checksum con rehash_dict() antes de
    escribir, satisfaciendo el validator de RegulationConfig.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.app.core.checksum import rehash_dict
from src.app.core.regulation_active import (
    ActiveRegulation,
    get_active_regulation,
)


# ---------------------------------------------------------------------------
# Helper privado
# ---------------------------------------------------------------------------


def _write_reg_json(
    tmp_path: Path,
    reg_id: str,
    date_start: str,
    date_end: str,
    transition_window_days: int = 7,
) -> Path:
    """
    Escribe un JSON de regulación mínimo y válido en
    tmp_path/{reg_id}.json con checksum calculado.

    Usa mega_enabled=False para evitar la validación que exige
    mega_evolutions_disponibles no vacío.

    Returns:
        Path del archivo creado.
    """
    data: dict[str, object] = {
        "regulation_id": reg_id,
        "game": "pokemon_champions",
        "date_start": date_start,
        "date_end": date_end,
        "schema_version": "1.0.0",
        "last_verified": date.today().isoformat(),
        "checksum_sha256": "",
        "transition_window_days": transition_window_days,
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
            "item_clause": True,
            "legendary_ban": False,
            "restricted_ban": False,
            "open_team_list": False,
        },
        "pokemon_legales": [1, 2, 3],
        "mega_evolutions_disponibles": [],
        "items_legales": ["Sitrus Berry"],
        "moves_legales": [],
        "abilities_legales": [],
        "banned_moves": [],
        "banned_abilities": [],
        "source_urls": {},
    }

    data_with_checksum = rehash_dict(data)
    path = tmp_path / f"{reg_id}.json"
    path.write_text(
        json.dumps(data_with_checksum, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Grupo 1 — Escenario A (regulación activa única)
# ---------------------------------------------------------------------------


class TestEscenarioA:
    """
    ESCENARIO A: Exactamente una regulación vigente.
    today está dentro de [date_start, date_end].
    """

    def test_returns_active_state(self, tmp_path: Path) -> None:
        """Con una reg vigente hoy, state es 'active'."""
        _write_reg_json(
            tmp_path, "TEST-A",
            date_start="2026-01-01",
            date_end="2026-12-31",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 15),
        )
        assert result.state == "active"
        assert result.regulation_id == "TEST-A"

    def test_returns_correct_regulation_id(self, tmp_path: Path) -> None:
        """La reg retornada tiene el regulation_id correcto."""
        _write_reg_json(
            tmp_path, "TEST-A",
            date_start="2026-01-01",
            date_end="2026-12-31",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 15),
        )
        assert result.regulation_id == "TEST-A"

    def test_today_equals_date_start(self, tmp_path: Path) -> None:
        """today == date_start es vigente (borde izquierdo)."""
        _write_reg_json(
            tmp_path, "TEST-A",
            date_start="2026-04-08",
            date_end="2026-06-17",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 4, 8),
        )
        assert result.state == "active"

    def test_today_equals_date_end(self, tmp_path: Path) -> None:
        """today == date_end es vigente (borde derecho)."""
        _write_reg_json(
            tmp_path, "TEST-A",
            date_start="2026-04-08",
            date_end="2026-06-17",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 17),
        )
        assert result.state == "active"

    def test_with_two_regs_one_active(self, tmp_path: Path) -> None:
        """Con dos regs sin solapamiento, retorna la vigente."""
        _write_reg_json(
            tmp_path, "OLD-REG",
            date_start="2025-01-01",
            date_end="2025-12-31",
        )
        _write_reg_json(
            tmp_path, "NEW-REG",
            date_start="2026-01-01",
            date_end="2026-12-31",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 15),
        )
        assert result.regulation_id == "NEW-REG"
        assert result.state == "active"

    def test_result_is_active_regulation_instance(self, tmp_path: Path) -> None:
        """El resultado es una instancia de ActiveRegulation."""
        _write_reg_json(
            tmp_path, "TEST-A",
            date_start="2026-01-01",
            date_end="2026-12-31",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 15),
        )
        assert isinstance(result, ActiveRegulation)


# ---------------------------------------------------------------------------
# Grupo 2 — Escenario A' (solapamiento)
# ---------------------------------------------------------------------------


class TestEscenarioAPrime:
    """
    ESCENARIO A': Múltiples regulaciones vigentes simultáneamente
    (dato corrupto). Elige la de date_start más reciente.
    """

    def test_overlap_returns_most_recent(self, tmp_path: Path) -> None:
        """Con solapamiento, retorna la de date_start más reciente."""
        _write_reg_json(
            tmp_path, "OLDER",
            date_start="2026-01-01",
            date_end="2026-12-31",
        )
        _write_reg_json(
            tmp_path, "NEWER",
            date_start="2026-06-01",
            date_end="2026-12-31",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 7, 1),
        )
        assert result.regulation_id == "NEWER"

    def test_overlap_state_is_active(self, tmp_path: Path) -> None:
        """El state sigue siendo 'active' aunque haya solapamiento."""
        _write_reg_json(
            tmp_path, "OLDER",
            date_start="2026-01-01",
            date_end="2026-12-31",
        )
        _write_reg_json(
            tmp_path, "NEWER",
            date_start="2026-06-01",
            date_end="2026-12-31",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 7, 1),
        )
        assert result.state == "active"


# ---------------------------------------------------------------------------
# Grupo 3 — Escenario B (transición)
# ---------------------------------------------------------------------------


class TestEscenarioB:
    """
    ESCENARIO B: Ninguna reg vigente hoy, pero hay una reg cuya
    ventana de transición alcanza today.
    Ventana: [date_start - transition_window_days, date_start)
    """

    def test_day_before_start_is_transition(self, tmp_path: Path) -> None:
        """El día antes del date_start está en ventana de transición."""
        _write_reg_json(
            tmp_path, "UPCOMING",
            date_start="2026-06-18",
            date_end="2026-09-30",
            transition_window_days=7,
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 17),
        )
        assert result.state == "transition"
        assert result.regulation_id == "UPCOMING"

    def test_exactly_at_transition_window_start(self, tmp_path: Path) -> None:
        """El día exacto al inicio de la ventana (date_start - 7) es transición."""
        _write_reg_json(
            tmp_path, "UPCOMING",
            date_start="2026-06-25",
            date_end="2026-09-30",
            transition_window_days=7,
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 18),
        )
        assert result.state == "transition"

    def test_before_transition_window_is_no_active(self, tmp_path: Path) -> None:
        """Antes de la ventana de transición el estado es 'no_active'."""
        _write_reg_json(
            tmp_path, "UPCOMING",
            date_start="2026-06-25",
            date_end="2026-09-30",
            transition_window_days=7,
        )
        # 8 días antes del start — fuera de la ventana de 7 días
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 17),
        )
        assert result.state == "no_active"

    def test_just_ended_reg_is_transition(self, tmp_path: Path) -> None:
        """Una reg recién terminada (dentro de 7 días) entra en transición."""
        _write_reg_json(
            tmp_path, "JUST-ENDED",
            date_start="2026-01-01",
            date_end="2026-06-17",
            transition_window_days=7,
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 20),
        )
        assert result.state == "transition"


# ---------------------------------------------------------------------------
# Grupo 4 — Escenario C (sin regulación activa)
# ---------------------------------------------------------------------------


class TestEscenarioC:
    """
    ESCENARIO C: Ningún escenario anterior aplica.
    Retorna la regulación más reciente por date_end con state="no_active".
    """

    def test_future_only_reg_is_no_active(self, tmp_path: Path) -> None:
        """Una reg completamente futura sin ventana de transición es 'no_active'."""
        _write_reg_json(
            tmp_path, "FUTURE",
            date_start="2027-01-01",
            date_end="2027-12-31",
            transition_window_days=7,
        )
        # today está muy lejos del start — fuera de la ventana de 7 días
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 1, 1),
        )
        assert result.state == "no_active"

    def test_long_past_reg_is_no_active(self, tmp_path: Path) -> None:
        """Una reg terminada hace mucho retorna 'no_active'."""
        _write_reg_json(
            tmp_path, "OLD",
            date_start="2024-01-01",
            date_end="2024-06-30",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 1, 1),
        )
        assert result.state == "no_active"
        assert result.regulation_id == "OLD"

    def test_no_active_returns_most_recent_by_date_end(
        self, tmp_path: Path
    ) -> None:
        """Con múltiples regs pasadas, retorna la de date_end más reciente."""
        _write_reg_json(
            tmp_path, "OLDER",
            date_start="2023-01-01",
            date_end="2023-06-30",
        )
        _write_reg_json(
            tmp_path, "LESS-OLD",
            date_start="2024-01-01",
            date_end="2024-12-31",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 1),
        )
        assert result.regulation_id == "LESS-OLD"


# ---------------------------------------------------------------------------
# Grupo 5 — Edge cases y errores
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """
    Edge cases: directorio vacío, archivos corruptos, today inyectable.
    """

    def test_empty_directory_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        """Directorio vacío levanta FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_active_regulation(
                reg_dir=tmp_path,
                today=date(2026, 6, 15),
            )

    def test_nonexistent_directory_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        """Directorio inexistente levanta FileNotFoundError."""
        nonexistent = tmp_path / "no_existe"
        with pytest.raises(FileNotFoundError):
            get_active_regulation(
                reg_dir=nonexistent,
                today=date(2026, 6, 15),
            )

    def test_corrupted_json_is_skipped(self, tmp_path: Path) -> None:
        """Un JSON corrupto se ignora sin romper la función si hay
        al menos un JSON válido."""
        (tmp_path / "CORRUPT.json").write_text(
            "{ invalid json }", encoding="utf-8"
        )
        _write_reg_json(
            tmp_path, "VALID",
            date_start="2026-01-01",
            date_end="2026-12-31",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 15),
        )
        assert result.regulation_id == "VALID"

    def test_today_parameter_is_injectable(self, tmp_path: Path) -> None:
        """today es inyectable — el resultado cambia según la fecha pasada."""
        _write_reg_json(
            tmp_path, "REG-2026",
            date_start="2026-01-01",
            date_end="2026-12-31",
        )
        _write_reg_json(
            tmp_path, "REG-2025",
            date_start="2025-01-01",
            date_end="2025-12-31",
        )
        result_2026 = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 15),
        )
        result_2025 = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2025, 6, 15),
        )
        assert result_2026.regulation_id == "REG-2026"
        assert result_2025.regulation_id == "REG-2025"

    def test_reason_is_non_empty_string(self, tmp_path: Path) -> None:
        """El campo reason nunca es vacío — siempre tiene un mensaje explicativo."""
        _write_reg_json(
            tmp_path, "TEST",
            date_start="2026-01-01",
            date_end="2026-12-31",
        )
        result = get_active_regulation(
            reg_dir=tmp_path,
            today=date(2026, 6, 15),
        )
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0
