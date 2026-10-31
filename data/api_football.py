"""Клиент API-Football v3 (api-sports.io): матчи дня, составы, травмы, таблицы, недавние результаты.

Бесплатный тариф: 100 запросов/сутки, ~10 запросов/мин. Все ответы кэшируются в SQLite.
Документация: https://www.api-football.com/documentation-v3
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from data.cache import Cache
from data.http import ApiError, CachedHttp
from data.schemas import Fixture, Injury, Lineup

log = logging.getLogger(__name__)

BASE = "https://v3.football.api-sports.io"
FINISHED = {"FT", "AET", "PEN"}


class ApiFootball:
    source = "API-Football"

    def __init__(self, cache: Cache, api_key: str) -> None:
        if not api_key:
            raise ApiError("API_FOOTBALL_KEY не задан в .env")
        self.http = CachedHttp(cache, self.source, min_interval_s=6.5,  # ≤10 req/min на free
                               headers={"x-apisports-key": api_key})

    def _get(self, path: str, params: dict[str, Any], ttl_s: float | None, refresh: bool = False) -> tuple[list[Any], str]:
        payload, fetched_at = self.http.get_json(f"{BASE}{path}", params, ttl_s=ttl_s, refresh=refresh)
        errors = payload.get("errors")
        if errors:
            raise ApiError(f"{self.source} {path}: {errors}")
        remaining = self.http.last_response_headers.get("x-ratelimit-requests-remaining")
        if remaining is not None:
            log.info("API-Football: осталось запросов на сегодня: %s", remaining)
        return payload.get("response", []), fetched_at

    # ---------------- fixtures ----------------
    def fixtures_for_local_date(self, day: date, tz: str, league_ids: dict[int, str], refresh: bool = False) -> list[Fixture]:
        """Матчи, начинающиеся в локальные сутки `day` (tz). Запрашиваем 2 UTC-даты и фильтруем."""
        from zoneinfo import ZoneInfo
        zone = ZoneInfo(tz)
        start = datetime.combine(day, datetime.min.time(), zone).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        utc_days = sorted({start.date(), (end - timedelta(seconds=1)).date()})
        out: list[Fixture] = []
        for d in utc_days:
            rows, fetched_at = self._get("/fixtures", {"date": d.isoformat(), "timezone": "UTC"},
                                         ttl_s=1800, refresh=refresh)
            for r in rows:
                lid = r["league"]["id"]
                if lid not in league_ids:
                    continue
                ko = datetime.fromtimestamp(r["fixture"]["timestamp"], tz=timezone.utc)
                if not (start <= ko < end):
                    continue
                venue = r["fixture"].get("venue") or {}
                out.append(Fixture(
                    source=self.source, source_id=str(r["fixture"]["id"]), league=league_ids[lid],
                    season=int(r["league"]["season"]), kickoff_utc=ko,
                    home_raw=r["teams"]["home"]["name"], away_raw=r["teams"]["away"]["name"],
                    home_team_id=r["teams"]["home"]["id"], away_team_id=r["teams"]["away"]["id"],
                    status=r["fixture"]["status"]["short"], referee=r["fixture"].get("referee"),
                    venue_city=venue.get("city"), round=r["league"].get("round"), fetched_at=fetched_at,
                ))
        log.info("API-Football: %d матчей на %s (%s)", len(out), day, tz)
        return out

    def finished_results(self, league_id: int, season: int, date_from: date, date_to: date,
                         league_code: str, refresh: bool = False) -> tuple[pd.DataFrame, str]:
        """Завершённые матчи лиги за период — дополняют историю после даты обновления датасета."""
        rows, fetched_at = self._get("/fixtures", {"league": league_id, "season": season,
                                                   "from": date_from.isoformat(), "to": date_to.isoformat()},
                                     ttl_s=6 * 3600, refresh=refresh)
        recs = []
        for r in rows:
            if r["fixture"]["status"]["short"] not in FINISHED:
                continue
            ko = datetime.fromtimestamp(r["fixture"]["timestamp"], tz=timezone.utc)
            recs.append({
                "league": league_code, "season": season, "date": pd.Timestamp(ko.date()),
                "time": ko.strftime("%H:%M"), "home": r["teams"]["home"]["name"], "away": r["teams"]["away"]["name"],
                "hg": r["goals"]["home"], "ag": r["goals"]["away"], "src": f"API-Football@{fetched_at}",
            })
        return pd.DataFrame(recs), fetched_at

    def team_recent_fixtures(self, team_id: int, last: int = 6, refresh: bool = False) -> tuple[list[dict[str, Any]], str]:
        """Последние матчи команды во ВСЕХ турнирах (еврокубки/кубки) — для плотности календаря."""
        rows, fetched_at = self._get("/fixtures", {"team": team_id, "last": last}, ttl_s=6 * 3600, refresh=refresh)
        out = [{"date": datetime.fromtimestamp(r["fixture"]["timestamp"], tz=timezone.utc),
                "competition": r["league"]["name"], "type": r["league"].get("type")} for r in rows]
        return out, fetched_at

    # ---------------- lineups / injuries / standings ----------------
    def lineups(self, fixture_id: str, refresh: bool = True) -> tuple[list[Lineup], str]:
        rows, fetched_at = self._get("/fixtures/lineups", {"fixture": fixture_id}, ttl_s=600, refresh=refresh)
        out = [Lineup(team_id=r["team"]["id"], team=r["team"]["name"],
                      starters=[p["player"]["name"] for p in r.get("startXI", [])],
                      starter_ids=[p["player"]["id"] for p in r.get("startXI", [])],
                      formation=r.get("formation"), fetched_at=fetched_at) for r in rows]
        return out, fetched_at

    def injuries(self, fixture_id: str, refresh: bool = False) -> tuple[list[Injury], str]:
        rows, fetched_at = self._get("/injuries", {"fixture": fixture_id}, ttl_s=3 * 3600, refresh=refresh)
        out = [Injury(team_id=r["team"]["id"], team=r["team"]["name"], player=r["player"]["name"],
                      player_id=r["player"].get("id"), kind=r["player"].get("type") or "",
                      reason=r["player"].get("reason") or "", fetched_at=fetched_at) for r in rows]
        return out, fetched_at

    def standings(self, league_id: int, season: int, refresh: bool = False) -> tuple[pd.DataFrame, str]:
        rows, fetched_at = self._get("/standings", {"league": league_id, "season": season}, ttl_s=6 * 3600, refresh=refresh)
        recs = []
        for blk in rows:
            for group in blk["league"]["standings"]:
                for t in group:
                    recs.append({"rank": t["rank"], "team": t["team"]["name"], "team_id": t["team"]["id"],
                                 "points": t["points"], "played": t["all"]["played"], "gd": t["goalsDiff"]})
        return pd.DataFrame(recs), fetched_at

    def resolve_league_id(self, country: str, name: str, fallback: int | None) -> int | None:
        """Проверка/поиск league id по стране и названию (кэш 30 дней)."""
        try:
            rows, _ = self._get("/leagues", {"country": country, "type": "league"}, ttl_s=30 * 86400)
        except ApiError as exc:
            log.warning("Не удалось проверить id лиги %s: %s", name, exc)
            return fallback
        for r in rows:
            if r["league"]["name"].lower() == name.lower():
                return int(r["league"]["id"])
        return fallback
