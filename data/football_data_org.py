"""Клиент football-data.org v4 — резервный источник расписания (бесплатно: 10 запросов/мин, 12 турниров)."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from data.cache import Cache
from data.http import ApiError, CachedHttp
from data.schemas import Fixture

log = logging.getLogger(__name__)
BASE = "https://api.football-data.org/v4"


class FootballDataOrg:
    source = "football-data.org"

    def __init__(self, cache: Cache, api_key: str) -> None:
        if not api_key:
            raise ApiError("FOOTBALL_DATA_ORG_KEY не задан в .env")
        self.http = CachedHttp(cache, self.source, min_interval_s=6.5, headers={"X-Auth-Token": api_key})

    def fixtures_for_local_date(self, day: date, tz: str, comp_codes: dict[str, str],
                                refresh: bool = False) -> list[Fixture]:
        from zoneinfo import ZoneInfo
        start = datetime.combine(day, datetime.min.time(), ZoneInfo(tz)).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        params = {"dateFrom": start.date().isoformat(), "dateTo": end.date().isoformat(),
                  "competitions": ",".join(comp_codes)}
        data, fetched_at = self.http.get_json(f"{BASE}/matches", params, ttl_s=1800, refresh=refresh)
        out: list[Fixture] = []
        for m in data.get("matches", []):
            ko = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            if not (start <= ko < end):
                continue
            code = m["competition"]["code"]
            if code not in comp_codes:
                continue
            refs = m.get("referees") or []
            season_start = (m.get("season") or {}).get("startDate", "")
            out.append(Fixture(
                source=self.source, source_id=str(m["id"]), league=comp_codes[code],
                season=int(season_start[:4]) if season_start else ko.year,
                kickoff_utc=ko, home_raw=m["homeTeam"]["name"], away_raw=m["awayTeam"]["name"],
                status=m.get("status", ""), referee=refs[0]["name"] if refs else None, fetched_at=fetched_at))
        log.info("football-data.org: %d матчей на %s", len(out), day)
        return out
