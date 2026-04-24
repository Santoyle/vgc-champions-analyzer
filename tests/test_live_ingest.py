"""
Tests para los scrapers LIVE y el pipeline de ingesta diaria.

Estrategia:
- Los tests de scrapers que hacen HTTP usan unittest.mock.patch
  para interceptar _fetch_url o las funciones de ingesta —
  NUNCA se hacen requests reales a internet.
- Los tests del parser ps_paste.py son de lógica pura:
  trabajan solo con strings y no necesitan mocks.
- Los tests de run_live_ingest mockean las funciones
  ingest_* directamente, sin tocar httpx.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.app.data.scrapers.pikalytics_champions import (
    PikalyticsEntry,
    PikalyticsSnapshot,
    _parse_markdown_table,
    fetch_pikalytics_snapshot,
    snapshot_to_records,
)
from src.app.data.scrapers.showdown_replays import (
    ParsedReplay,
    _parse_battle_log,
    replays_to_records,
)
from src.app.data.scrapers.limitless import (
    TournamentSummary,
    count_recent_tournaments,
    summaries_to_records,
)
from src.app.data.scrapers.rk9_pokedata import (
    OfficialEvent,
    _classify_event_type,
    events_to_records,
)
from src.app.data.parsers.ps_paste import (
    ParsedSlot,
    ParsedTeam,
    parse_paste,
    parse_slot,
    team_to_records,
)
from src.app.data.pipelines.live_ingest import (
    LIVE_SOURCES,
    run_live_ingest,
)

# ---------------------------------------------------------------------------
# Constantes de datos sintéticos
# ---------------------------------------------------------------------------

MARKDOWN_TABLE_SAMPLE = """
| Pokémon | Usage % | Win % |
|---------|---------|-------|
| Incineroar | 54.4% | 51.2% |
| Sneasler | 45.1% | 52.1% |
| Garchomp | 37.1% | 50.8% |
"""

SHOWDOWN_PASTE_INCINEROAR = """
Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
Tera Type: Fire
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Fake Out
- Parting Shot
- Flare Blitz
- Darkest Lariat
"""

SHOWDOWN_PASTE_FULL_TEAM = """
Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
Tera Type: Fire
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Fake Out
- Parting Shot
- Flare Blitz
- Darkest Lariat

Garchomp @ Choice Scarf
Ability: Rough Skin
Level: 50
Tera Type: Steel
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Earthquake
- Dragon Claw
- Stone Edge
- Protect

Sneasler @ Focus Sash
Ability: Unburden
Level: 50
Tera Type: Poison
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Close Combat
- Dire Claw
- Fake Out
- Protect
"""

SHOWDOWN_LOG_SAMPLE = """|
|player|p1|PlayerA|...
|player|p2|PlayerB|...
|poke|p1|Incineroar, L50, M
|poke|p1|Garchomp, L50
|poke|p1|Sneasler, L50
|poke|p2|Sinistcha, L50
|poke|p2|Kingambit, L50
|poke|p2|Basculegion, L50, F
|turn|1
|win|PlayerA
"""


# ---------------------------------------------------------------------------
# Grupo 1 — Tests del parser Pikalytics
# ---------------------------------------------------------------------------


class TestPikalyticsParser:
    """Tests para el parser de markdown de Pikalytics."""

    def test_parse_markdown_table_extracts_pokemon(self) -> None:
        """Parsea correctamente tabla markdown."""
        entries = _parse_markdown_table(MARKDOWN_TABLE_SAMPLE, "M-A")
        assert len(entries) == 3
        pokemon_names = [e.pokemon for e in entries]
        assert "Incineroar" in pokemon_names
        assert "Sneasler" in pokemon_names
        assert "Garchomp" in pokemon_names

    def test_parse_markdown_table_usage_pct(self) -> None:
        """Los porcentajes de uso se parsean correctamente."""
        entries = _parse_markdown_table(MARKDOWN_TABLE_SAMPLE, "M-A")
        incineroar = next(e for e in entries if e.pokemon == "Incineroar")
        assert incineroar.usage_pct == pytest.approx(54.4, abs=0.1)

    def test_parse_markdown_table_empty_text(self) -> None:
        """Texto vacío retorna lista vacía."""
        entries = _parse_markdown_table("", "M-A")
        assert entries == []

    def test_parse_markdown_table_no_table(self) -> None:
        """Texto sin tabla retorna lista vacía."""
        entries = _parse_markdown_table("Sin tabla aquí", "M-A")
        assert entries == []

    def test_snapshot_to_records_structure(self) -> None:
        """snapshot_to_records retorna dicts con las columnas esperadas."""
        snapshot = PikalyticsSnapshot(
            regulation_id="M-A",
            snapshot_date=date(2026, 4, 24),
            source_url="https://pikalytics.com",
            entries=[
                PikalyticsEntry(
                    pokemon="Incineroar",
                    usage_pct=54.4,
                    win_pct=51.2,
                    count=8234,
                    rank=1,
                )
            ],
            parse_method="markdown_ai",
        )
        records = snapshot_to_records(snapshot)
        assert len(records) == 1
        record = records[0]
        assert record["pokemon"] == "Incineroar"
        assert record["usage_pct"] == pytest.approx(54.4, abs=0.1)
        assert record["regulation_id"] == "M-A"
        assert "snapshot_date" in record
        assert "parse_method" in record

    def test_fetch_pikalytics_returns_snapshot_on_error(self) -> None:
        """fetch_pikalytics_snapshot retorna snapshot vacío (no levanta
        excepción) si falla la red."""
        with patch(
            "src.app.data.scrapers.pikalytics_champions._fetch_url"
        ) as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            snapshot = fetch_pikalytics_snapshot("M-A")
        assert isinstance(snapshot, PikalyticsSnapshot)
        assert snapshot.entries == []
        assert snapshot.parse_method == "fallback"


# ---------------------------------------------------------------------------
# Grupo 2 — Tests del parser de replays
# ---------------------------------------------------------------------------


class TestShowdownReplayParser:
    """Tests para el parser de battle logs."""

    def test_parse_battle_log_extracts_teams(self) -> None:
        """Extrae equipos de p1 y p2 correctamente."""
        team_p1, team_p2, winner = _parse_battle_log(
            SHOWDOWN_LOG_SAMPLE, "M-A", "test-123"
        )
        assert "Incineroar" in team_p1
        assert "Garchomp" in team_p1
        assert "Sneasler" in team_p1
        assert "Sinistcha" in team_p2
        assert "Kingambit" in team_p2
        assert "Basculegion" in team_p2

    def test_parse_battle_log_winner(self) -> None:
        """Extrae el ganador correctamente."""
        _, _, winner = _parse_battle_log(
            SHOWDOWN_LOG_SAMPLE, "M-A", "test-123"
        )
        assert winner == "PlayerA"

    def test_parse_battle_log_removes_gender_suffix(self) -> None:
        """Los géneros (M, F) no aparecen en el nombre del Pokémon —
        la coma separa el nombre."""
        team_p1, _, _ = _parse_battle_log(
            SHOWDOWN_LOG_SAMPLE, "M-A", "test-123"
        )
        # "Incineroar, L50, M" debe dar "Incineroar"
        assert all("," not in pkm for pkm in team_p1)

    def test_parse_battle_log_empty_log(self) -> None:
        """Log vacío retorna listas vacías y None."""
        team_p1, team_p2, winner = _parse_battle_log(
            "", "M-A", "test-empty"
        )
        assert team_p1 == []
        assert team_p2 == []
        assert winner is None

    def test_replays_to_records_no_raw_log(self) -> None:
        """replays_to_records NO incluye raw_log en los records
        (mantiene Parquet compacto)."""
        replay = ParsedReplay(
            replay_id="test-123",
            regulation_id="M-A",
            format_slug="gen9championsbssregma",
            p1="PlayerA",
            p2="PlayerB",
            rating=1800,
            upload_time=1745000000,
            team_p1=["Incineroar", "Garchomp"],
            team_p2=["Sneasler", "Sinistcha"],
            winner="PlayerA",
            raw_log="log content here",
        )
        records = replays_to_records([replay])
        assert len(records) == 1
        assert "raw_log" not in records[0]
        assert "team_p1_json" in records[0]
        assert "team_p2_json" in records[0]

    def test_replays_to_records_teams_as_json(self) -> None:
        """Los equipos se serializan como JSON strings."""
        replay = ParsedReplay(
            replay_id="test-456",
            regulation_id="M-A",
            format_slug="gen9championsbssregma",
            p1="A",
            p2="B",
            rating=1700,
            upload_time=0,
            team_p1=["Incineroar", "Garchomp"],
            team_p2=[],
            winner=None,
        )
        records = replays_to_records([replay])
        team_p1 = json.loads(records[0]["team_p1_json"])
        assert team_p1 == ["Incineroar", "Garchomp"]


# ---------------------------------------------------------------------------
# Grupo 3 — Tests del parser de paste Showdown
# ---------------------------------------------------------------------------


class TestPasteParser:
    """Tests para parse_paste y parse_slot (lógica pura, sin mocks)."""

    def test_parse_slot_basic(self) -> None:
        """Parsea un slot básico correctamente."""
        slot = parse_slot(SHOWDOWN_PASTE_INCINEROAR.strip())
        assert slot is not None
        assert slot.species == "Incineroar"
        assert slot.item == "Sitrus Berry"
        assert slot.ability == "Intimidate"
        assert slot.tera_type == "Fire"
        assert slot.nature == "Impish"
        assert slot.level == 50

    def test_parse_slot_evs(self) -> None:
        """Los EVs se parsean correctamente."""
        slot = parse_slot(SHOWDOWN_PASTE_INCINEROAR.strip())
        assert slot is not None
        assert slot.evs["hp"] == 252
        assert slot.evs["def"] == 252
        assert slot.evs["atk"] == 4

    def test_parse_slot_moves(self) -> None:
        """Los 4 moves se parsean correctamente."""
        slot = parse_slot(SHOWDOWN_PASTE_INCINEROAR.strip())
        assert slot is not None
        assert len(slot.moves) == 4
        assert "Fake Out" in slot.moves
        assert "Parting Shot" in slot.moves

    def test_parse_slot_mega_capable_true(self) -> None:
        """mega_capable es True si el item es Mega Stone."""
        slot = parse_slot(
            "Charizard @ Charizardite X\n"
            "Ability: Blaze\n"
            "- Flamethrower\n"
        )
        assert slot is not None
        assert slot.mega_capable is True

    def test_parse_slot_mega_capable_false(self) -> None:
        """mega_capable es False para items normales."""
        slot = parse_slot(SHOWDOWN_PASTE_INCINEROAR.strip())
        assert slot is not None
        assert slot.mega_capable is False

    def test_parse_paste_full_team(self) -> None:
        """Parsea un equipo de 3 Pokémon."""
        team = parse_paste(SHOWDOWN_PASTE_FULL_TEAM)
        assert len(team.slots) == 3
        species = [s.species for s in team.slots]
        assert "Incineroar" in species
        assert "Garchomp" in species
        assert "Sneasler" in species

    def test_parse_paste_empty(self) -> None:
        """Paste vacío retorna team vacío con warning."""
        team = parse_paste("")
        assert isinstance(team, ParsedTeam)
        assert len(team.slots) == 0
        assert len(team.parse_warnings) > 0

    def test_parse_paste_invalid_block(self) -> None:
        """Bloque inválido genera warning sin crash."""
        team = parse_paste(
            "Pokémon válido @ Sitrus Berry\n"
            "Ability: Overgrow\n"
            "- Tackle\n"
            "\n"
            "esto no es un bloque válido\n"
        )
        assert isinstance(team.parse_warnings, list)

    def test_parse_paste_with_nickname(self) -> None:
        """Nickname se separa de especie correctamente."""
        team = parse_paste(
            "Cindy (Incineroar) @ Sitrus Berry\n"
            "Ability: Intimidate\n"
            "- Fake Out\n"
        )
        assert len(team.slots) == 1
        assert team.slots[0].species == "Incineroar"
        assert team.slots[0].nickname == "Cindy"

    def test_team_to_records_structure(self) -> None:
        """team_to_records produce records con las columnas esperadas."""
        team = parse_paste(SHOWDOWN_PASTE_INCINEROAR.strip())
        records = team_to_records(team, "M-A")
        assert len(records) == 1
        record = records[0]
        expected_keys = {
            "regulation_id",
            "species",
            "item",
            "ability",
            "moves_json",
            "evs_json",
            "tera_type",
            "mega_capable",
        }
        assert expected_keys.issubset(set(record.keys()))


# ---------------------------------------------------------------------------
# Grupo 4 — Tests de Limitless y RK9
# ---------------------------------------------------------------------------


class TestLimitlessAndRK9:
    """Tests para scrapers Limitless y pokedata.ovh."""

    def test_count_recent_tournaments_returns_int(self) -> None:
        """count_recent_tournaments SIEMPRE retorna int, incluso si hay error."""
        with patch(
            "src.app.data.scrapers.limitless.fetch_recent_tournaments"
        ) as mock_fetch:
            mock_fetch.side_effect = Exception("network error")
            result = count_recent_tournaments("M-A", days=7)
        assert isinstance(result, int)
        assert result == 0

    def test_count_recent_tournaments_with_data(self) -> None:
        """count_recent_tournaments retorna el número de torneos cuando
        hay datos."""
        mock_tournaments = [
            TournamentSummary(
                tournament_id=f"t{i}",
                name=f"Tournament {i}",
                format_name="VGC Champions",
                regulation_id="M-A",
                date=date(2026, 4, 20),
                num_players=64,
                url=f"https://limitlesstcg.com/t{i}",
            )
            for i in range(3)
        ]
        with patch(
            "src.app.data.scrapers.limitless.fetch_recent_tournaments"
        ) as mock_fetch:
            mock_fetch.return_value = mock_tournaments
            result = count_recent_tournaments("M-A", days=7)
        assert result == 3

    def test_classify_event_type_regional(self) -> None:
        """Clasifica eventos Regionals correctamente."""
        assert _classify_event_type("Indianapolis Regionals") == "regional"

    def test_classify_event_type_international(self) -> None:
        """Clasifica NAIC correctamente."""
        assert _classify_event_type("NAIC 2026") == "international"

    def test_classify_event_type_unknown(self) -> None:
        """Retorna 'unknown' para tipos no reconocidos."""
        assert _classify_event_type("Random Tournament") == "unknown"

    def test_summaries_to_records_structure(self) -> None:
        """summaries_to_records produce records con las columnas esperadas."""
        tournaments = [
            TournamentSummary(
                tournament_id="indy2026",
                name="Indianapolis Regionals",
                format_name="VGC Champions",
                regulation_id="M-A",
                date=date(2026, 5, 29),
                num_players=512,
                url="https://limitlesstcg.com/indy",
            )
        ]
        records = summaries_to_records(tournaments)
        assert len(records) == 1
        assert records[0]["tournament_id"] == "indy2026"
        assert records[0]["regulation_id"] == "M-A"
        assert "date" in records[0]

    def test_events_to_records_none_date(self) -> None:
        """events_to_records maneja event_date=None."""
        events = [
            OfficialEvent(
                event_id="test-evt",
                name="Test Event",
                event_type="regional",
                regulation_id="M-A",
                event_date=None,
                location="",
                num_players=0,
                has_results=False,
                url="https://pokedata.ovh/events/test",
            )
        ]
        records = events_to_records(events)
        assert len(records) == 1
        assert records[0]["event_date"] is None


# ---------------------------------------------------------------------------
# Grupo 5 — Tests del pipeline run_live_ingest
# ---------------------------------------------------------------------------


class TestLiveIngestPipeline:
    """Tests para run_live_ingest con scrapers mockeados."""

    def test_live_sources_contains_four_sources(self) -> None:
        """LIVE_SOURCES tiene exactamente 4 fuentes."""
        assert len(LIVE_SOURCES) == 4
        assert "pikalytics" in LIVE_SOURCES
        assert "showdown" in LIVE_SOURCES
        assert "limitless" in LIVE_SOURCES
        assert "rk9" in LIVE_SOURCES

    def test_run_live_ingest_with_explicit_reg_id(
        self, tmp_path: Path
    ) -> None:
        """run_live_ingest usa reg_id explícito sin llamar a
        get_active_regulation."""
        with (
            patch(
                "src.app.data.pipelines.live_ingest.ingest_pikalytics"
            ) as mock_pika,
            patch(
                "src.app.data.pipelines.live_ingest.ingest_showdown"
            ) as mock_sd,
            patch(
                "src.app.data.pipelines.live_ingest.ingest_limitless"
            ) as mock_lim,
            patch(
                "src.app.data.pipelines.live_ingest.ingest_rk9"
            ) as mock_rk9,
        ):
            mock_pika.return_value = True
            mock_sd.return_value = True
            mock_lim.return_value = False
            mock_rk9.return_value = False

            results = run_live_ingest(
                reg_id="M-A",
                fecha="2026-04-24",
            )

        assert results["pikalytics"] is True
        assert results["showdown"] is True
        assert results["limitless"] is False
        assert results["rk9"] is False

    def test_run_live_ingest_partial_failure(self) -> None:
        """Fallo parcial no cancela otras fuentes."""
        with (
            patch(
                "src.app.data.pipelines.live_ingest.ingest_pikalytics"
            ) as mock_pika,
            patch(
                "src.app.data.pipelines.live_ingest.ingest_showdown"
            ) as mock_sd,
            patch(
                "src.app.data.pipelines.live_ingest.ingest_limitless"
            ) as mock_lim,
            patch(
                "src.app.data.pipelines.live_ingest.ingest_rk9"
            ) as mock_rk9,
        ):
            mock_pika.side_effect = Exception("Pikalytics down")
            mock_sd.return_value = True
            mock_lim.return_value = True
            mock_rk9.return_value = True

            try:
                results = run_live_ingest(
                    reg_id="M-A",
                    fecha="2026-04-24",
                )
                assert results.get("showdown") is True
            except Exception:
                pytest.fail(
                    "run_live_ingest propagó excepción de un scraper individual"
                )

    def test_run_live_ingest_sources_filter(self) -> None:
        """sources filter ejecuta solo las fuentes especificadas."""
        with (
            patch(
                "src.app.data.pipelines.live_ingest.ingest_pikalytics"
            ) as mock_pika,
            patch(
                "src.app.data.pipelines.live_ingest.ingest_showdown"
            ) as mock_sd,
            patch(
                "src.app.data.pipelines.live_ingest.ingest_limitless"
            ) as mock_lim,
            patch(
                "src.app.data.pipelines.live_ingest.ingest_rk9"
            ) as mock_rk9,
        ):
            mock_pika.return_value = True
            mock_sd.return_value = True
            mock_lim.return_value = True
            mock_rk9.return_value = True

            run_live_ingest(
                reg_id="M-A",
                fecha="2026-04-24",
                sources=["pikalytics"],
            )

            assert mock_pika.called
            assert not mock_sd.called
            assert not mock_lim.called
            assert not mock_rk9.called
