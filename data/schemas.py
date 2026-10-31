"""Структуры данных, общие для источников."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Fixture:
    source: str
    source_id: str
    league: str
    season: int
    kickoff_utc: datetime
    home_raw: str
    away_raw: str
    fetched_at: str
    home_team_id: int | None = None
    away_team_id: int | None = None
    status: str = "NS"
    referee: str | None = None
    venue_city: str | None = None
    round: str | None = None
    home: str | None = None      # каноническое имя (после сопоставления)
    away: str | None = None


@dataclass
class Injury:
    team_id: int
    team: str
    player: str
    player_id: int | None
    kind: str          # "Missing Fixture" / "Questionable"
    reason: str
    fetched_at: str


@dataclass
class Lineup:
    team_id: int
    team: str
    starters: list[str]
    starter_ids: list[int]
    formation: str | None
    fetched_at: str


@dataclass
class OddsQuote:
    bookmaker: str
    market: str        # "h2h" | "totals"
    outcome: str       # "home" | "draw" | "away" | "over" | "under"
    price: float
    point: float | None = None
    last_update: str | None = None


@dataclass
class MatchOdds:
    event_id: str
    home_raw: str
    away_raw: str
    commence_time: datetime
    fetched_at: str
    quotes: list[OddsQuote] = field(default_factory=list)
    closing: dict[str, float] = field(default_factory=dict)  # последний снимок до начала матча
