"""Тесты разбора ответов API на зафиксированных примерах формата (без сети)."""
import json
from datetime import date, datetime, timezone

import pytest

from data.api_football import ApiFootball
from data.cache import Cache
from data.http import ApiError, CachedHttp
from data.odds_api import OddsApi, best_prices, consensus_book
from features.injuries import absence_impact, missing_from_lineup


class FakeResp:
    def __init__(self, payload, status=200, headers=None):
        self.content = json.dumps(payload).encode()
        self.status_code = status
        self.headers = headers or {}
        self.text = self.content.decode()


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


@pytest.fixture
def cache():
    return Cache(":memory:")


def test_http_cache_and_source_log(cache):
    sess = FakeSession([FakeResp({"a": 1})])
    http = CachedHttp(cache, "test", min_interval_s=0, session=sess)
    d1, ts1 = http.get_json("https://x/y", {"q": 1})
    d2, ts2 = http.get_json("https://x/y", {"q": 1})
    assert d1 == d2 == {"a": 1} and ts1 == ts2
    assert len(sess.calls) == 1                      # второй раз — из кэша
    log = cache.run_log()
    assert [e["from_cache"] for e in log] == [False, True]
    assert all(e["fetched_at"] for e in log)


def test_http_auth_error(cache):
    http = CachedHttp(cache, "test", min_interval_s=0, session=FakeSession([FakeResp({}, 401)]))
    with pytest.raises(ApiError):
        http.get_json("https://x/z")


def test_api_football_fixtures_filtered_by_local_day(cache):
    ts_in = int(datetime(2026, 10, 31, 14, 0, tzinfo=timezone.utc).timestamp())    # 19:00 Алматы
    ts_out = int(datetime(2026, 10, 31, 20, 0, tzinfo=timezone.utc).timestamp())   # 01:00 следующего дня
    def fx(fid, ts, lid=39):
        return {"fixture": {"id": fid, "timestamp": ts, "referee": "M. Oliver", "status": {"short": "NS"},
                            "venue": {"city": "London"}},
                "league": {"id": lid, "season": 2026, "round": "R10"},
                "teams": {"home": {"id": 1, "name": "Arsenal"}, "away": {"id": 2, "name": "Chelsea"}},
                "goals": {"home": None, "away": None}}
    payload = {"errors": [], "response": [fx(1, ts_in), fx(2, ts_out), fx(3, ts_in, lid=999)]}
    af = ApiFootball(cache, "k")
    af.http.session = FakeSession([FakeResp(payload), FakeResp({"errors": [], "response": []})])
    af.http.min_interval_s = 0
    out = af.fixtures_for_local_date(date(2026, 10, 31), "Asia/Almaty", {39: "EPL"})
    assert [f.source_id for f in out] == ["1"]
    assert out[0].league == "EPL" and out[0].referee == "M. Oliver" and out[0].fetched_at


def test_odds_api_parse_and_best_prices(cache):
    ev = {"id": "e1", "sport_key": "soccer_epl", "commence_time": "2026-10-31T15:00:00Z",
          "home_team": "Arsenal", "away_team": "Chelsea",
          "bookmakers": [
              {"key": "pinnacle", "last_update": "2026-10-31T10:00:00Z", "markets": [
                  {"key": "h2h", "outcomes": [{"name": "Arsenal", "price": 2.0}, {"name": "Chelsea", "price": 3.9},
                                              {"name": "Draw", "price": 3.6}]},
                  {"key": "totals", "outcomes": [{"name": "Over", "price": 1.95, "point": 2.5},
                                                 {"name": "Under", "price": 1.92, "point": 2.5}]}]},
              {"key": "bet365", "markets": [
                  {"key": "h2h", "outcomes": [{"name": "Arsenal", "price": 2.05}, {"name": "Chelsea", "price": 3.7},
                                              {"name": "Draw", "price": 3.5}]}]}]}
    api = OddsApi(cache, "k")
    out = api.parse([ev], "soccer_epl", "2026-10-31T11:00:00+00:00")
    mo = out[0]
    best = best_prices(mo)
    assert best["home"] == (2.05, "bet365") and best["away"] == (3.9, "pinnacle")
    assert best["over25"][0] == 1.95
    books = consensus_book(mo)
    assert books["pinnacle"]["draw"] == 3.6
    assert mo.closing["home"] == 2.0                     # последний снимок до начала, sharp-букмекер
    assert len(cache.odds_rows("e1")) == 8


def test_absence_impact():
    import pandas as pd
    tp = pd.DataFrame({"player": ["Mohamed Salah", "Virgil van Dijk", "Bench Guy"],
                       "minutes": [2500, 2700, 100], "xG": [15.0, 2.0, 0.1], "xA": [8.0, 0.5, 0.0]})
    imp = absence_impact(["M. Salah"], tp)
    assert imp.players and imp.players[0][0] == "Mohamed Salah"
    assert 0.85 <= imp.own_goal_mult < 1.0
    assert imp.opp_goal_mult >= 1.0
    assert missing_from_lineup(["Virgil van Dijk"], tp, top_n=2) == ["Mohamed Salah"]
    assert absence_impact(["X"], pd.DataFrame()).data_available is False


def test_circuit_breaker_on_connection_error(cache):
    import requests

    class Boom:
        calls = 0

        def get(self, *a, **k):
            Boom.calls += 1
            raise requests.exceptions.ProxyError("403 tunnel")

    CachedHttp._down_hosts.clear()
    http = CachedHttp(cache, "t", min_interval_s=0, session=Boom())
    for _ in range(3):
        with pytest.raises(ApiError):
            http.get_json("https://blocked.example/a")
    assert Boom.calls == 1          # после первого отказа хост больше не дёргаем
    CachedHttp._down_hosts.clear()
