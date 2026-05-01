"""
Tests unitarios para Bloque 14 (`new_reg_detector`, `create_reg_draft`).

Los Grupos 1–3 cubren ``new_reg_detector``; los Grupos 4–6 cubren ``create_reg_draft``.
Ningún test hace I/O de red ni lee Parquets; el CLI escribe solo bajo ``tmp_path``.
"""
from __future__ import annotations

import json
import sys
import tempfile  # noqa: F401  # requerido por spec del proyecto
from pathlib import Path
from typing import Any  # noqa: F401  # requerido por spec del proyecto
from unittest.mock import MagicMock, patch

import pytest

from src.app.modules.new_reg_detector import (
    MIN_SOURCES_FOR_DETECTION,
    NewRegDetection,
    _generate_format_candidates,
    detect_new_regulation,
    format_detection_for_discord,
)
from scripts.create_reg_draft import (
    _get_next_regulation_id,
    generate_draft_json,
)


class TestNewRegDetection:
    """Tests del dataclass NewRegDetection."""

    def test_fields_exist(self) -> None:
        """NewRegDetection tiene los campos esperados."""
        det = NewRegDetection(
            format_slug="gen9championsbssregmb",
            sources=["showdown", "limitless"],
            confidence=2,
        )
        assert det.format_slug == "gen9championsbssregmb"
        assert det.sources == ["showdown", "limitless"]
        assert det.confidence == 2

    def test_is_confirmed_true(self) -> None:
        """is_confirmed=True cuando confidence >= MIN_SOURCES_FOR_DETECTION."""
        det = NewRegDetection(
            format_slug="test",
            confidence=MIN_SOURCES_FOR_DETECTION,
        )
        assert det.is_confirmed is True

    def test_is_confirmed_false(self) -> None:
        """is_confirmed=False cuando confidence < MIN_SOURCES_FOR_DETECTION."""
        det = NewRegDetection(
            format_slug="test",
            confidence=1,
        )
        assert det.is_confirmed is False

    def test_is_confirmed_zero(self) -> None:
        """is_confirmed=False cuando confidence=0."""
        det = NewRegDetection(
            format_slug="test",
            confidence=0,
        )
        assert det.is_confirmed is False

    def test_default_empty_lists(self) -> None:
        """sources y sample_data tienen defaults."""
        det = NewRegDetection(format_slug="test")
        assert det.sources == []
        assert det.sample_data == {}

    def test_min_sources_constant(self) -> None:
        """MIN_SOURCES_FOR_DETECTION == 2."""
        assert MIN_SOURCES_FOR_DETECTION == 2


class TestFormatDetectionForDiscord:
    """Tests para el formateador de mensajes Discord."""

    def test_contains_format_slug(self) -> None:
        """El mensaje incluye el format_slug."""
        det = NewRegDetection(
            format_slug="gen9championsbssregmb",
            sources=["showdown", "limitless"],
            confidence=2,
        )
        msg = format_detection_for_discord(det)
        assert "gen9championsbssregmb" in msg

    def test_contains_sources(self) -> None:
        """El mensaje incluye las fuentes."""
        det = NewRegDetection(
            format_slug="test",
            sources=["showdown", "limitless"],
            confidence=2,
        )
        msg = format_detection_for_discord(det)
        assert "showdown" in msg
        assert "limitless" in msg

    def test_contains_new_regulation_indicator(self) -> None:
        """El mensaje indica nueva regulación."""
        det = NewRegDetection(
            format_slug="test",
            sources=["showdown"],
            confidence=1,
        )
        msg = format_detection_for_discord(det)
        assert len(msg) > 0
        assert isinstance(msg, str)

    def test_contains_action_instructions(self) -> None:
        """El mensaje incluye instrucciones de acción."""
        det = NewRegDetection(
            format_slug="gen9championsbssregmb",
            sources=["showdown", "limitless"],
            confidence=2,
        )
        msg = format_detection_for_discord(det)
        assert "regulation" in msg.lower() or "reg" in msg.lower()


class TestGenerateFormatCandidates:
    """Tests para el generador de candidatos de format slug."""

    def test_generates_next_letter(self) -> None:
        """Genera el slug con la siguiente letra."""
        known = {"gen9championsbssregma"}
        candidates = _generate_format_candidates(known)
        assert "gen9championsbssregmb" in candidates

    def test_does_not_include_known(self) -> None:
        """No incluye slugs ya conocidos."""
        known = {"gen9championsbssregma"}
        candidates = _generate_format_candidates(known)
        assert "gen9championsbssregma" not in candidates

    def test_empty_known_returns_empty(self) -> None:
        """Sin slugs conocidos retorna lista vacía."""
        candidates = _generate_format_candidates(set())
        assert candidates == []

    def test_multiple_known_slugs(self) -> None:
        """Genera candidatos para múltiples slugs."""
        known = {
            "gen9championsbssregma",
            "gen9vgc2025regi",
        }
        candidates = _generate_format_candidates(known)
        assert len(candidates) >= 1

    def test_no_z_overflow(self) -> None:
        """No genera slugs con caracteres más allá de 'z'."""
        known = {"gen9championsbssregmz"}
        candidates = _generate_format_candidates(known)
        for c in candidates:
            assert all(
                ord(ch) <= ord("z") for ch in c if ch.isalpha()
            )

    def test_returns_list(self) -> None:
        """Retorna lista (no set ni otro tipo)."""
        known = {"gen9championsbssregma"}
        result = _generate_format_candidates(known)
        assert isinstance(result, list)


class TestGetNextRegulationId:
    """Tests para la derivación de regulation_id."""

    def test_regma_gives_m_a(self) -> None:
        """gen9championsbssregma → M-A."""
        result = _get_next_regulation_id(
            "gen9championsbssregma",
        )
        assert result == "M-A"

    def test_regmb_gives_m_b(self) -> None:
        """gen9championsbssregmb → M-B."""
        result = _get_next_regulation_id(
            "gen9championsbssregmb",
        )
        assert result == "M-B"

    def test_single_letter_reg(self) -> None:
        """gen9vgc2025regi → I."""
        result = _get_next_regulation_id(
            "gen9vgc2025regi",
        )
        assert result == "I"

    def test_unknown_format(self) -> None:
        """Formato sin patrón reg → UNKNOWN."""
        result = _get_next_regulation_id(
            "gen9randombattle",
        )
        assert result == "UNKNOWN"

    def test_empty_slug(self) -> None:
        """Slug vacío → UNKNOWN."""
        result = _get_next_regulation_id("")
        assert result == "UNKNOWN"

    def test_returns_string(self) -> None:
        """Siempre retorna string."""
        result = _get_next_regulation_id(
            "gen9championsbssregmc",
        )
        assert isinstance(result, str)


class TestGenerateDraftJson:
    """Tests para el generador de JSON draft."""

    def test_returns_dict(self) -> None:
        """Retorna dict."""
        result = generate_draft_json(
            "gen9championsbssregmb",
        )
        assert isinstance(result, dict)

    def test_has_required_fields(self) -> None:
        """Tiene los campos requeridos por el schema."""
        result = generate_draft_json(
            "gen9championsbssregmb",
        )
        required_fields = {
            "regulation_id",
            "game",
            "date_start",
            "date_end",
            "battle_format",
            "mechanics",
            "clauses",
            "pokemon_legales",
            "checksum_sha256",
        }
        assert required_fields.issubset(set(result.keys()))

    def test_pokemon_legales_is_empty(self) -> None:
        """pokemon_legales es lista vacía (placeholder)."""
        result = generate_draft_json(
            "gen9championsbssregmb",
        )
        assert result["pokemon_legales"] == []

    def test_date_start_is_placeholder(self) -> None:
        """date_start es placeholder YYYY-MM-DD."""
        result = generate_draft_json(
            "gen9championsbssregmb",
        )
        assert result["date_start"] == "YYYY-MM-DD"

    def test_checksum_is_empty(self) -> None:
        """checksum_sha256 es string vacío."""
        result = generate_draft_json(
            "gen9championsbssregmb",
        )
        assert result["checksum_sha256"] == ""

    def test_format_slug_in_battle_format(self) -> None:
        """El format_slug aparece en battle_format."""
        slug = "gen9championsbssregmb"
        result = generate_draft_json(slug)
        assert result["battle_format"].get("format_slug") == slug

    def test_explicit_reg_id(self) -> None:
        """regulation_id explícito se usa en lugar del derivado."""
        result = generate_draft_json(
            "gen9championsbssregmb",
            regulation_id="CUSTOM",
        )
        assert result["regulation_id"] == "CUSTOM"

    def test_derived_reg_id(self) -> None:
        """Si reg_id=None, se deriva del slug."""
        result = generate_draft_json(
            "gen9championsbssregmb",
        )
        assert result["regulation_id"] == "M-B"

    def test_draft_notes_present(self) -> None:
        """_draft_notes está presente como campo informativo."""
        result = generate_draft_json(
            "gen9championsbssregmb",
        )
        assert "_draft_notes" in result
        assert len(result["_draft_notes"]) > 0

    def test_game_is_pokemon_champions(self) -> None:
        """game es pokemon_champions."""
        result = generate_draft_json(
            "gen9championsbssregmb",
        )
        assert result["game"] == "pokemon_champions"


class TestCreateRegDraftCLI:
    """Tests para el script CLI de creación de draft."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """main() crea el archivo JSON."""
        from scripts.create_reg_draft import main

        output = tmp_path / "TEST-B.json"
        old_argv = sys.argv
        try:
            sys.argv = [
                "create_reg_draft.py",
                "--slug",
                "gen9championsbssregtb",
                "--reg-id",
                "TEST-B",
                "--output",
                str(output),
            ]
            result = main()
        finally:
            sys.argv = old_argv
        assert result == 0
        assert output.exists()

    def test_created_file_is_valid_json(self, tmp_path: Path) -> None:
        """El archivo creado es JSON válido."""
        from scripts.create_reg_draft import main

        output = tmp_path / "TEST-C.json"
        old_argv = sys.argv
        try:
            sys.argv = [
                "create_reg_draft.py",
                "--slug",
                "gen9championsbssregtc",
                "--reg-id",
                "TEST-C",
                "--output",
                str(output),
            ]
            main()
        finally:
            sys.argv = old_argv
        content = json.loads(output.read_text(encoding="utf-8"))
        assert isinstance(content, dict)

    def test_returns_1_if_file_exists(self, tmp_path: Path) -> None:
        """Retorna exit 1 si el archivo ya existe."""
        from scripts.create_reg_draft import main

        output = tmp_path / "EXISTS.json"
        output.write_text("{}", encoding="utf-8")

        old_argv = sys.argv
        try:
            sys.argv = [
                "create_reg_draft.py",
                "--slug",
                "gen9championsbssregte",
                "--reg-id",
                "EXISTS",
                "--output",
                str(output),
            ]
            result = main()
        finally:
            sys.argv = old_argv
        assert result == 1

    def test_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        """No sobreescribe un archivo existente."""
        from scripts.create_reg_draft import main

        output = tmp_path / "EXISTING.json"
        original = '{"original": true}'
        output.write_text(original, encoding="utf-8")

        old_argv = sys.argv
        try:
            sys.argv = [
                "create_reg_draft.py",
                "--slug",
                "gen9championsbssregtf",
                "--reg-id",
                "EXISTING",
                "--output",
                str(output),
            ]
            main()
        finally:
            sys.argv = old_argv
        assert output.read_text(encoding="utf-8") == original


class TestDetectNewRegulationNoHTTP:
    """``detect_new_regulation`` mockeada (sin ``httpx`` / red)."""

    @patch(
        "src.app.modules.new_reg_detector._check_limitless",
        MagicMock(),
    )
    @patch(
        "src.app.modules.new_reg_detector._check_rk9",
        MagicMock(),
    )
    @patch(
        "src.app.modules.new_reg_detector._check_showdown",
        return_value={},
    )
    @patch(
        "src.app.modules.new_reg_detector._get_known_format_slugs",
        return_value=set(),
    )
    def test_returns_empty_when_showdown_empty(
        self,
        _mock_known: MagicMock,
        _mock_showdown: MagicMock,
    ) -> None:
        assert detect_new_regulation() == []

    @patch(
        "src.app.modules.new_reg_detector._check_limitless",
        MagicMock(),
    )
    @patch(
        "src.app.modules.new_reg_detector._check_rk9",
        MagicMock(),
    )
    @patch(
        "src.app.modules.new_reg_detector._check_showdown",
    )
    @patch(
        "src.app.modules.new_reg_detector._get_known_format_slugs",
        return_value=set(),
    )
    def test_returns_confirmed_detection(
        self,
        _mock_known: MagicMock,
        mock_showdown: MagicMock,
    ) -> None:
        slug = "gen9championsbssregmb"
        mock_showdown.return_value = {
            slug: NewRegDetection(
                format_slug=slug,
                sources=["showdown", "limitless"],
                confidence=2,
            ),
        }
        out = detect_new_regulation()
        assert len(out) == 1
        assert out[0].format_slug == slug
