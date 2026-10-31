"""Клиент The Odds API v4: коэффициенты 1X2 и тоталов у нескольких букмекеров.

Бесплатный тариф: 500 кредитов/месяц. Запрос /odds стоит (кол-во рынков × кол-во регионов) кредитов,
/sports — бесплатно. Каждый снимок сохраняется в SQLite (odds_snapshots) — последний снимок
до начала матча используется как закрывающая линия (CLV). Исторические снимки API — платные.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from data.cache import Cache
from data.http import ApiError, CachedHttp
from data.schemas import MatchOdds, OddsQuote

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4"
SHARP_BOOKS = ("pinnacle", "betfair_ex_eu", "matchbook")


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class OddsApi:
    source = "The Odds API"

    def __init__(self, cache: Cache, api_key: str) -> None:
        if not api_key:
            raise ApiError("ODDS_API_KEY не задан в .env")
        self.cache = cache
        self.key = api_key
        self.http = CachedHttp(cache, self.source, min_interval_s=1.0)

    def active_sports(self) -> set[str]:
        data, _ = self.http.get_json(f"{BASE}/sports/", {"apiKey": self.key}, ttl_s=86400, cache_params={})
        return {s["key"] for s in data if s.get("active")}

    def odds(self, sport_key: str, start: datetime, end: datetime, regions: str = "eu,uk",
             markets: str = "h2h,totals", refresh: bool = False) -> list[MatchOdds]:
        params = {"apiKey": self.key, "regions": regions, "markets": markets, "oddsFormat": "decimal",
                  "dateFormat": "iso",
                  "commenceTimeFrom": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "commenceTimeTo": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        cache_params = {k: v for k, v in params.items() if k != "apiKey"} | {"sport": sport_key}
        data, fetched_at = self.http.get_json(f"{BASE}/sports/{sport_key}/odds", params, ttl_s=1800,
                                              refresh=refresh, cache_params=cache_params)
        rem = self.http.last_response_headers.get("x-requests-remaining")
        if rem is not None:
            log.info("The Odds API: осталось кредитов: %s", rem)
        return self.parse(data, sport_key, fetched_at)

    def parse(self, data: list[dict[str, Any]], sport_key: str, fetched_at: str) -> list[MatchOdds]:
        out: list[MatchOdds] = []
        snap_rows: list[tuple[Any, ...]] = []
        for ev in data:
            mo = MatchOdds(event_id=ev["id"], home_raw=ev["home_team"], away_raw=ev["away_team"],
                           commence_time=_parse_ts(ev["commence_time"]), fetched_at=fetched_at)
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    for oc in mk.get("outcomes", []):
                        if mk["key"] == "h2h":
                            name = ("home" if oc["name"] == ev["home_team"] else
                                    "away" if oc["name"] == ev["away_team"] else "draw")
                            point = None
                        elif mk["key"] == "totals":
                            name = oc["name"].lower()
                            point = float(oc.get("point")) if oc.get("point") is not None else None
                        else:
                            continue
                        q = OddsQuote(bookmaker=bk["key"], market=mk["key"], outcome=name,
                                      price=float(oc["price"]), point=point,
                                      last_update=mk.get("last_update") or bk.get("last_update"))
                        mo.quotes.append(q)
                        snap_rows.append((ev["id"], sport_key, ev["commence_time"], ev["home_team"], ev["away_team"],
                                          q.bookmaker, q.market, q.outcome, q.point if q.point is not None else -1.0,
                                          q.price, q.last_update, fetched_at))
            out.append(mo)
        self.cache.put_odds(snap_rows)
        for mo in out:
            mo.closing = self.closing_line(mo.event_id, mo.commence_time)
        return out

    def closing_line(self, event_id: str, commence: datetime) -> dict[str, float]:
        """Последний сохранённый снимок до начала матча: средняя цена (sharp-букмекер, если есть)."""
        rows = [r for r in self.cache.odds_rows(event_id)
                if r["market"] == "h2h" and _parse_ts(r["fetched_at"]) <= commence]
        if not rows:
            return {}
        last_ts = max(r["fetched_at"] for r in rows)
        last = [r for r in rows if r["fetched_at"] == last_ts]
        sharp = [r for r in last if r["bookmaker"] in SHARP_BOOKS]
        use = sharp or last
        res: dict[str, float] = {}
        for oc in ("home", "draw", "away"):
            prices = [r["price"] for r in use if r["outcome"] == oc]
            if prices:
                res[oc] = float(sum(prices) / len(prices))
        res["_fetched_at"] = last_ts  # type: ignore[assignment]
        res["_minutes_before_ko"] = (commence - _parse_ts(last_ts)).total_seconds() / 60  # type: ignore[assignment]
        return res


def best_prices(mo: MatchOdds, total_line: float = 2.5) -> dict[str, tuple[float, str]]:
    """Лучшая цена по каждому исходу: {outcome: (price, bookmaker)}."""
    best: dict[str, tuple[float, str]] = {}
    for q in mo.quotes:
        if q.market == "totals" and (q.point is None or abs(q.point - total_line) > 1e-9):
            continue
        key = q.outcome if q.market == "h2h" else f"{q.outcome}25"
        if key not in best or q.price > best[key][0]:
            best[key] = (q.price, q.bookmaker)
    return best


def consensus_book(mo: MatchOdds) -> dict[str, dict[str, float]]:
    """Коэффициенты по букмекерам: {bookmaker: {"home":..,"draw":..,"away":..,"over25":..,"under25":..}}."""
    books: dict[str, dict[str, float]] = {}
    for q in mo.quotes:
        if q.market == "totals" and (q.point is None or abs(q.point - 2.5) > 1e-9):
            continue
        key = q.outcome if q.market == "h2h" else f"{q.outcome}25"
        books.setdefault(q.bookmaker, {})[key] = q.price
    return books
