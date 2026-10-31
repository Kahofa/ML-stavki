"""Understat: xG по матчам и игрокам (EPL, La Liga, Serie A, Bundesliga, Ligue 1, РПЛ).

Официального API нет: используется JSON-эндпоинт сайта (getLeagueData), при неудаче —
разбор встроенного в страницу JSON. Обязательно кэширование и паузы между запросами (≥3 с).
"""
from __future__ import annotations

import codecs
import json
import logging
import re
from typing import Any

import pandas as pd

from data.cache import Cache
from data.http import ApiError, CachedHttp

log = logging.getLogger(__name__)
BASE = "https://understat.com"


def _extract_embedded(html: str, var: str) -> Any:
    m = re.search(rf"{var}\s*=\s*JSON\.parse\('(.*?)'\)", html, re.S)
    if not m:
        raise ApiError(f"Understat: переменная {var} не найдена на странице")
    return json.loads(codecs.decode(m.group(1), "unicode_escape"))


class Understat:
    source = "Understat"

    def __init__(self, cache: Cache) -> None:
        self.http = CachedHttp(cache, self.source, min_interval_s=3.0,
                               headers={"User-Agent": "Mozilla/5.0 (research; cached)",
                                        "X-Requested-With": "XMLHttpRequest"})

    def league_season(self, league: str, season: int, refresh: bool = False, finished: bool = False) -> tuple[dict[str, Any], str]:
        ttl = None if finished else 12 * 3600
        try:
            data, fetched_at = self.http.get_json(f"{BASE}/getLeagueData/{league}/{season}", ttl_s=ttl, refresh=refresh)
            if isinstance(data, dict) and "dates" in data:
                return data, fetched_at
        except ApiError as exc:
            log.info("Understat JSON-эндпоинт недоступен (%s), пробуем HTML", exc)
        body, fetched_at, _ = self.http.get_raw(f"{BASE}/league/{league}/{season}", ttl_s=ttl, refresh=refresh)
        html = body.decode("utf-8", "ignore")
        return {"dates": _extract_embedded(html, "datesData"),
                "players": _extract_embedded(html, "playersData")}, fetched_at

    def matches_xg(self, league_code: str, understat_league: str, season: int, refresh: bool = False,
                   finished: bool = False) -> tuple[pd.DataFrame, str]:
        data, fetched_at = self.league_season(understat_league, season, refresh, finished)
        recs = []
        for m in data["dates"]:
            if not m.get("isResult"):
                continue
            recs.append({"league": league_code, "date": pd.Timestamp(m["datetime"][:10]),
                         "home_raw": m["h"]["title"], "away_raw": m["a"]["title"],
                         "hxg": float(m["xG"]["h"]), "axg": float(m["xG"]["a"])})
        return pd.DataFrame(recs), fetched_at

    def players(self, understat_league: str, season: int, refresh: bool = False) -> tuple[pd.DataFrame, str]:
        """Игроки: минуты, xG, xA — для взвешивания травм по вкладу в xG."""
        data, fetched_at = self.league_season(understat_league, season, refresh)
        df = pd.DataFrame(data.get("players", []))
        if df.empty:
            return df, fetched_at
        for c in ("time", "games", "xG", "xA", "goals", "assists"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.rename(columns={"player_name": "player", "team_title": "team", "time": "minutes"}), fetched_at
