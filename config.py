"""Глобальная конфигурация: лиги, пути, ключи API, параметры моделей и ставок."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class League:
    """Описание лиги и её идентификаторов в разных источниках."""

    code: str                      # наш код (CLI)
    name: str
    country: str
    hist_division: str | None      # код дивизиона football-data.co.uk (исторический датасет)
    api_football_id: int | None    # league id в API-Football (проверяется через /leagues)
    odds_api_key: str | None       # sport key в The Odds API (проверяется через /v4/sports)
    football_data_org: str | None  # код турнира football-data.org
    understat: str | None          # код лиги Understat (xG)
    calendar: str = "aug-may"      # "aug-may" или "calendar" (сезон = календарный год)
    priority: bool = True


LEAGUES: dict[str, League] = {
    lg.code: lg
    for lg in [
        League("EPL", "Premier League", "England", "E0", 39, "soccer_epl", "PL", "EPL"),
        League("LaLiga", "La Liga", "Spain", "SP1", 140, "soccer_spain_la_liga", "PD", "La_liga"),
        League("SerieA", "Serie A", "Italy", "I1", 135, "soccer_italy_serie_a", "SA", "Serie_A"),
        League("Bundesliga", "Bundesliga", "Germany", "D1", 78, "soccer_germany_bundesliga", "BL1", "Bundesliga"),
        League("Ligue1", "Ligue 1", "France", "F1", 61, "soccer_france_ligue_one", "FL1", "Ligue_1"),
        League("RPL", "Russian Premier League", "Russia", "RUS", 235, "soccer_russia_premier_league", None, "RFPL"),
        League("KPL", "Kazakhstan Premier League", "Kazakhstan", None, 389, None, None, None, calendar="calendar"),
        # --- остальные лиги включаются флагом --all-leagues или явно через --leagues ---
        League("Championship", "Championship", "England", "E1", 40, "soccer_efl_champ", "ELC", None, priority=False),
        League("Eredivisie", "Eredivisie", "Netherlands", "N1", 88, "soccer_netherlands_eredivisie", "DED", None, priority=False),
        League("Primeira", "Primeira Liga", "Portugal", "P1", 94, "soccer_portugal_primeira_liga", "PPL", None, priority=False),
        League("BelgiumPro", "Jupiler Pro League", "Belgium", "B1", 144, "soccer_belgium_first_div", None, None, priority=False),
        League("SuperLig", "Super Lig", "Turkey", "T1", 203, "soccer_turkey_super_league", None, None, priority=False),
        League("Scotland", "Premiership", "Scotland", "SC0", 179, "soccer_spl", None, None, priority=False),
    ]
}

PRIORITY_LEAGUES = [c for c, lg in LEAGUES.items() if lg.priority]


@dataclass(frozen=True)
class Settings:
    api_football_key: str = field(default_factory=lambda: os.getenv("API_FOOTBALL_KEY", ""))
    football_data_org_key: str = field(default_factory=lambda: os.getenv("FOOTBALL_DATA_ORG_KEY", ""))
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    cache_db: Path = field(default_factory=lambda: ROOT / os.getenv("CACHE_DB", "cache/football.sqlite"))
    report_dir: Path = field(default_factory=lambda: ROOT / os.getenv("REPORT_DIR", "output"))
    artifacts_dir: Path = ROOT / "artifacts"
    timezone: str = field(default_factory=lambda: os.getenv("TIMEZONE", "Asia/Almaty"))
    bankroll: float = field(default_factory=lambda: float(os.getenv("BANKROLL", "1000")))
    # Исторический датасет: football-data.co.uk + ClubElo (зеркало на GitHub, обновляется периодически)
    history_urls: tuple[str, ...] = (
        "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv",
    )
    clubelo_history_url: str = (
        "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/EloRatings.csv"
    )


@dataclass(frozen=True)
class ModelParams:
    dc_xi: float = 0.0019            # затухание Dixon-Coles в сутки (полураспад ≈ 1 год)
    dc_window_days: int = 3 * 365    # окно истории для подгонки DC
    dc_refit_days: int = 7           # переобучение DC в бэктесте раз в неделю
    dc_l2: float = 2.0               # ridge-регуляризация к априорным значениям
    elo_k: float = 20.0
    elo_home_adv: float = 60.0
    elo_season_regress: float = 0.2
    form_halflife: float = 4.0       # полураспад (в матчах) для экспоненциального затухания формы
    max_goals: int = 10


@dataclass(frozen=True)
class BettingParams:
    min_value: float = 0.03          # value > 3%
    kelly_fraction: float = 0.25     # 1/4 Kelly
    max_stake: float = 0.02          # потолок 2% банка
    max_best_bets: int = 5
    min_odds: float = 1.30
    max_odds: float = 8.0


SETTINGS = Settings()
MODEL = ModelParams()
BETTING = BettingParams()
