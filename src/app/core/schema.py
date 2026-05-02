"""
Schema central del proyecto vgc-champions-analyzer.

RegulationConfig es la fuente de verdad de cada regulación activa.
Todos los módulos que necesiten información de la regulación deben
recibirla como parámetro de tipo RegulationConfig; nunca deben
resolver ni hardcodear el regulation_id internamente.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BattleFormat(BaseModel):
    model_config = ConfigDict(frozen=True)

    team_size: int = Field(ge=1, le=6)
    bring: int = Field(ge=1, le=6)
    pick: int = Field(ge=1, le=6)
    level_cap: int = Field(ge=1, le=100)
    best_of_swiss: int = Field(ge=1, le=5)
    best_of_topcut: int = Field(ge=1, le=5)
    team_preview_sec: int = Field(default=90, ge=0)
    turn_sec: int = Field(default=45, ge=0)
    player_timer_sec: int = Field(default=420, ge=0)
    game_timer_sec: int = Field(default=1200, ge=0)


class Mechanics(BaseModel):
    model_config = ConfigDict(frozen=True)

    mega_enabled: bool
    mega_max_per_battle: int = Field(default=1, ge=0, le=6)
    tera_enabled: bool
    z_moves_enabled: bool
    dynamax_enabled: bool
    stat_points_system: bool
    stat_points_total: int = Field(default=0, ge=0)
    stat_points_cap_per_stat: int = Field(default=0, ge=0)
    iv_system: bool = True


class Clauses(BaseModel):
    model_config = ConfigDict(frozen=True)

    species_clause: bool
    item_clause: bool
    legendary_ban: bool
    restricted_ban: bool
    open_team_list: bool


class RegulationConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    regulation_id: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Identificador único. Ej: 'M-A', 'I', 'H'",
    )
    game: Literal["pokemon_champions", "scarlet_violet"]
    date_start: date
    date_end: date
    battle_format: BattleFormat
    mechanics: Mechanics
    clauses: Clauses
    pokemon_legales: list[int] = Field(
        ...,
        min_length=1,
        description="Lista de dex IDs legales en esta reg",
    )
    mega_evolutions_disponibles: list[dict[str, str]] = Field(
        default_factory=list,
        description="Lista de {species, mega_item, mega_ability}",
    )
    items_legales: list[str] = Field(..., min_length=1)
    moves_legales: list[int] = Field(default_factory=list)
    abilities_legales: list[str] = Field(default_factory=list)
    banned_moves: list[int] = Field(default_factory=list)
    banned_abilities: list[str] = Field(default_factory=list)
    source_urls: dict[str, str] = Field(
        default_factory=dict,
        description="URLs de fuentes para trazabilidad",
    )
    checksum_sha256: str = Field(
        ...,
        pattern=r"^[a-f0-9]{64}$",
        description="SHA256 del payload canónico sin este campo",
    )
    last_verified: date
    schema_version: str = Field(default="1.0.0")
    transition_window_days: int = Field(
        default=7,
        ge=0,
        le=60,
        description=(
            "Días de ventana de transición antes de date_start. "
            "Default 7 cubre el 90% de transiciones históricas VGC."
        ),
    )

    @field_validator("pokemon_legales")
    @classmethod
    def no_duplicate_dex_ids(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("pokemon_legales contiene dex IDs duplicados")
        return v

    @field_validator("mega_evolutions_disponibles")
    @classmethod
    def mega_items_have_required_keys(
        cls, v: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        required = {"species", "mega_item", "mega_ability"}
        for entry in v:
            missing = required - set(entry.keys())
            if missing:
                raise ValueError(
                    f"mega_evolutions_disponibles entry falta campos: {missing}"
                )
        return v

    @model_validator(mode="after")
    def dates_coherent(self) -> RegulationConfig:
        if self.date_end <= self.date_start:
            raise ValueError(
                f"date_end ({self.date_end}) debe ser "
                f"posterior a date_start ({self.date_start})"
            )
        return self

    @model_validator(mode="after")
    def mega_consistency(self) -> RegulationConfig:
        if self.mechanics.mega_enabled and not self.mega_evolutions_disponibles:
            raise ValueError(
                "mega_enabled=True pero mega_evolutions_disponibles está vacío"
            )
        return self

    def compute_checksum(self) -> str:
        payload = self.model_dump(
            exclude={"checksum_sha256"},
            mode="json",
        )
        blob = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def verify_checksum(self) -> bool:
        return self.checksum_sha256 == self.compute_checksum()


class SPSpread(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
    )

    hp: int = Field(ge=0, le=32, default=0)
    atk: int = Field(ge=0, le=32, default=0)
    def_: int = Field(
        ge=0,
        le=32,
        default=0,
        alias="def",
    )
    spa: int = Field(ge=0, le=32, default=0)
    spd: int = Field(ge=0, le=32, default=0)
    spe: int = Field(ge=0, le=32, default=0)

    @model_validator(mode="after")
    def _check_total(self) -> SPSpread:
        total = (
            self.hp + self.atk + self.def_
            + self.spa + self.spd + self.spe
        )
        if total > 66:
            raise ValueError(
                f"SP total {total} excede el "
                f"cap de 66"
            )
        return self

    def to_evs(
        self,
    ) -> tuple[int, int, int, int, int, int]:
        """
        Convierte SP a EVs equivalentes.
        1 SP = 8 EVs, cap 252 (no 256) por
        compatibilidad con fórmula gen 9.
        """
        return (
            min(self.hp * 8, 252),
            min(self.atk * 8, 252),
            min(self.def_ * 8, 252),
            min(self.spa * 8, 252),
            min(self.spd * 8, 252),
            min(self.spe * 8, 252),
        )

    def total(self) -> int:
        """Retorna la suma total de SP."""
        return (
            self.hp + self.atk + self.def_
            + self.spa + self.spd + self.spe
        )

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> SPSpread:
        """
        Construye SPSpread desde dict con keys
        hp, atk, def, spa, spd, spe.
        Acepta tanto 'def' como 'def_' como key.
        """
        return cls.model_validate(
            {
                "hp": d.get("hp", 0),
                "def": d.get("def", d.get("def_", 0)),
                "atk": d.get("atk", 0),
                "spa": d.get("spa", 0),
                "spd": d.get("spd", 0),
                "spe": d.get("spe", 0),
            }
        )


__all__ = [
    "RegulationConfig",
    "BattleFormat",
    "Mechanics",
    "Clauses",
    "SPSpread",
]
