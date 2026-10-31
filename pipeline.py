"""Оркестрация: данные -> признаки -> бэктест (обязательно) -> прогнозы на дату.

Режимы:
  live    — матчи, составы, травмы и коэффициенты из API (нужны ключи в .env и доступ к сети);
  replay  — ретро-прогноз на прошедшую дату по реальному историческому датасету (матчи и коэффициенты
            той даты, все результаты начиная с этой даты скрыты от моделей) — для проверки и демонстрации.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtest.walkforward import BacktestResult, run_backtest
from config import BETTING, LEAGUES, MODEL, SETTINGS, BettingParams
from data.cache import Cache
from data.history import CANON_COLS, append_matches, load_history, merge_xg, season_of
from data.http import ApiError
from data.teams import TeamMatcher
from features.builder import adjust_dc, dc_walkforward, prepare_features
from features.injuries import absence_impact, missing_from_lineup
from models.decision import MAX_PRICE_RATIO, bet_decision, fair_odds, sane_price
from models.dixon_coles import rescale_matrix_1x2, score_matrix, top_scores
from models.market import devig

log = logging.getLogger(__name__)

HIST_START = pd.Timestamp("2006-07-01")
DC_START = pd.Timestamp("2011-07-01")
GBM_TRAIN_START = 2011
FIRST_TEST_SEASON = 2016


@dataclass
class RunResult:
    day: date
    mode: str
    leagues: list[str]
    matches: pd.DataFrame = field(default_factory=pd.DataFrame)
    details: list[dict[str, Any]] = field(default_factory=list)
    backtest: BacktestResult | None = None
    warnings: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""


# ============================================================================ helpers
def _ref_and_best(books: dict[str, dict[str, float]], keys: tuple[str, ...]) -> tuple[dict[str, float], dict[str, tuple[float, str]]]:
    """Референсная линия (Pinnacle, иначе медиана по букмекерам) и лучшие цены с названием букмекера."""
    full = {b: o for b, o in books.items() if all(k in o for k in keys)}
    ref: dict[str, float] = {}
    if "pinnacle" in full:
        ref = {k: full["pinnacle"][k] for k in keys}
    elif full:
        ref = {k: float(np.median([o[k] for o in full.values()])) for k in keys}
    best: dict[str, tuple[float, str]] = {}
    for b, o in books.items():
        for k in keys:
            if k not in o:
                continue
            if k in ref and o[k] > ref[k] * MAX_PRICE_RATIO:  # выброс/устаревшая котировка
                continue
            if k not in best or o[k] > best[k][0]:
                best[k] = (o[k], b)
    return ref, best


def _season_for(league: str, d: date) -> int:
    return int(season_of(pd.Timestamp(d), LEAGUES[league].calendar))


# ============================================================================ live data
def fetch_live_fixtures(cache: Cache, day: date, leagues: list[str], refresh: bool, warnings: list[str]):
    from data.api_football import ApiFootball
    from data.football_data_org import FootballDataOrg

    if SETTINGS.api_football_key:
        try:
            af = ApiFootball(cache, SETTINGS.api_football_key)
            ids = {LEAGUES[c].api_football_id: c for c in leagues if LEAGUES[c].api_football_id}
            return af.fixtures_for_local_date(day, SETTINGS.timezone, ids, refresh), af
        except ApiError as exc:
            warnings.append(f"API-Football недоступен: {exc}")
    else:
        warnings.append("API_FOOTBALL_KEY не задан — составы и травмы недоступны")
    if SETTINGS.football_data_org_key:
        try:
            fd = FootballDataOrg(cache, SETTINGS.football_data_org_key)
            codes = {LEAGUES[c].football_data_org: c for c in leagues if LEAGUES[c].football_data_org}
            return fd.fixtures_for_local_date(day, SETTINGS.timezone, codes, refresh), None
        except ApiError as exc:
            warnings.append(f"football-data.org недоступен: {exc}")
    else:
        warnings.append("FOOTBALL_DATA_ORG_KEY не задан")
    return None, None


def extend_history_live(cache: Cache, af, hist: pd.DataFrame, leagues: list[str], day: date,
                        refresh: bool, warnings: list[str]) -> pd.DataFrame:
    """Дополняет историю результатами из API-Football после даты обновления датасета (и для КПЛ — целиком)."""
    if af is None:
        return hist
    frames = []
    for c in leagues:
        lg = LEAGUES[c]
        if not lg.api_football_id:
            continue
        last = hist.loc[hist["league"] == c, "date"].max()
        cur = _season_for(c, day)
        seasons = list(range(_season_for(c, last.date()), cur + 1)) if pd.notna(last) else [cur - 3, cur - 2, cur - 1, cur]
        for s in seasons:
            season_start = date(s, 1, 1) if lg.calendar == "calendar" else date(s, 7, 1)
            start = max(season_start, (last + pd.Timedelta(days=1)).date()) if pd.notna(last) else season_start
            if start >= day:
                continue
            try:
                df, _ = af.finished_results(lg.api_football_id, s, start, day - timedelta(days=1), c, refresh)
                frames.append(df)
            except ApiError as exc:
                warnings.append(f"[{c}] не удалось дополнить историю {s}: {exc}")
    if not frames:
        return hist
    extra = pd.concat(frames, ignore_index=True)
    if extra.empty:
        return hist
    # имена API-Football -> канонические
    known = {c: sorted(set(hist.loc[hist["league"] == c, "home"])) for c in leagues}
    matcher = TeamMatcher(known)
    for col in ("home", "away"):
        extra[col] = [matcher.match(lg, n) or n for lg, n in zip(extra["league"], extra[col])]
    extra["season"] = [_season_for(lg, d.date()) for lg, d in zip(extra["league"], extra["date"])]
    log.info("История дополнена %d матчами из API-Football", len(extra))
    return append_matches(hist, extra)


def fetch_understat(cache: Cache, hist: pd.DataFrame, leagues: list[str], day: date, refresh: bool,
                    warnings: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    from data.understat import Understat
    us = Understat(cache)
    players: dict[str, pd.DataFrame] = {}
    frames = []
    known = {c: sorted(set(hist.loc[hist["league"] == c, "home"])) for c in leagues}
    matcher = TeamMatcher(known)
    for c in leagues:
        code = LEAGUES[c].understat
        if not code:
            continue
        cur = _season_for(c, day)
        for s in range(max(2014, cur - 4), cur + 1):
            try:
                xg, _ = us.matches_xg(c, code, s, refresh=refresh and s == cur, finished=s < cur)
                if not xg.empty:
                    xg["home"] = [matcher.match(c, n) or n for n in xg["home_raw"]]
                    xg["away"] = [matcher.match(c, n) or n for n in xg["away_raw"]]
                    frames.append(xg)
                if s == cur:
                    pl, _ = us.players(code, s)
                    if not pl.empty:
                        pl["team_canon"] = [matcher.match(c, n.split(",")[0]) or n for n in pl["team"]]
                        players[c] = pl
            except ApiError as exc:
                warnings.append(f"Understat недоступен ({exc}) — вместо xG используется xG-прокси по ударам")
                hist = merge_xg(hist, pd.concat(frames, ignore_index=True)) if frames else hist
                return hist, players
    hist = merge_xg(hist, pd.concat(frames, ignore_index=True)) if frames else hist
    return hist, players


def fetch_odds(cache: Cache, leagues: list[str], start: datetime, end: datetime, refresh: bool,
               warnings: list[str]) -> dict[str, list]:
    from data.odds_api import OddsApi
    if not SETTINGS.odds_api_key:
        warnings.append("ODDS_API_KEY не задан — нет рыночных коэффициентов (уверенность понижена)")
        return {}
    try:
        api = OddsApi(cache, SETTINGS.odds_api_key)
        active = api.active_sports()
    except ApiError as exc:
        warnings.append(f"The Odds API недоступен: {exc}")
        return {}
    out = {}
    for c in leagues:
        key = LEAGUES[c].odds_api_key
        if not key or key not in active:
            warnings.append(f"[{c}] нет линии в The Odds API (sport key {key!r} неактивен/отсутствует)")
            continue
        try:
            out[c] = api.odds(key, start - timedelta(hours=3), end + timedelta(hours=3), refresh=refresh)
        except ApiError as exc:
            warnings.append(f"[{c}] коэффициенты недоступны: {exc}")
    return out


def live_fixture_rows(cache: Cache, day: date, leagues: list[str], hist: pd.DataFrame, fixtures, af,
                      refresh: bool, warnings: list[str], players: dict[str, pd.DataFrame],
                      injuries_budget: int = 40) -> pd.DataFrame:
    from data.odds_api import consensus_book

    zone = ZoneInfo(SETTINGS.timezone)
    start = datetime.combine(day, datetime.min.time(), zone)
    odds = fetch_odds(cache, leagues, start, start + timedelta(days=1), refresh, warnings)
    known = {c: sorted(set(hist.loc[(hist["league"] == c) & (hist["season"] >= hist["season"].max() - 2), "home"])) for c in leagues}
    matcher = TeamMatcher(known)
    now = datetime.now(timezone.utc)
    clubelo_map: dict[str, float] = {}
    try:
        from data.clubelo import ClubElo
        ce, _ = ClubElo(cache).ratings(day, refresh)
        clubelo_map = dict(zip(ce["club"], ce["elo"]))
    except (ApiError, ValueError, KeyError) as exc:
        warnings.append(f"ClubElo недоступен ({exc}) — сверка Elo без внешнего рейтинга")
    ce_matcher = TeamMatcher({"*": list(clubelo_map)}, cutoff=0.8) if clubelo_map else None
    rows = []
    budget = injuries_budget
    for fx in fixtures:
        low: list[str] = []
        home = matcher.match(fx.league, fx.home_raw)
        away = matcher.match(fx.league, fx.away_raw)
        if home is None or away is None:
            low.append("команда не найдена в истории")
        r: dict[str, Any] = {c: np.nan for c in CANON_COLS}
        r.update({"league": fx.league, "season": _season_for(fx.league, day), "date": pd.Timestamp(fx.kickoff_utc.date()),
                  "time": fx.kickoff_utc.astimezone(zone).strftime("%H:%M"), "home": home or fx.home_raw,
                  "away": away or fx.away_raw, "hg": np.nan, "ag": np.nan, "src": fx.source,
                  "fixture_id": fx.source_id, "kickoff_utc": fx.kickoff_utc, "fixture_fetched_at": fx.fetched_at,
                  "referee": fx.referee, "is_fixture": True})
        # ---- коэффициенты
        ev = None
        for mo in odds.get(fx.league, []):
            mh, ma = matcher.match(fx.league, mo.home_raw), matcher.match(fx.league, mo.away_raw)
            if mh == home and ma == away and abs((mo.commence_time - fx.kickoff_utc).total_seconds()) < 3 * 3600:
                ev = mo
                break
        if ev is not None:
            books = consensus_book(ev)
            ref, best = _ref_and_best(books, ("home", "draw", "away"))
            ref_ou, best_ou = _ref_and_best(books, ("over25", "under25"))
            if ref:
                r.update({"odds_h": ref["home"], "odds_d": ref["draw"], "odds_a": ref["away"]})
                r.update({"max_h": best["home"][0], "max_d": best["draw"][0], "max_a": best["away"][0],
                          "book_h": best["home"][1], "book_d": best["draw"][1], "book_a": best["away"][1]})
            if ref_ou:
                r.update({"odds_o25": ref_ou["over25"], "odds_u25": ref_ou["under25"],
                          "max_o25": best_ou["over25"][0], "max_u25": best_ou["under25"][0],
                          "book_o25": best_ou["over25"][1], "book_u25": best_ou["under25"][1]})
            r["n_books"] = len(books)
            r["odds_fetched_at"] = ev.fetched_at
            if ev.closing:
                r.update({"close_h": ev.closing.get("home"), "close_d": ev.closing.get("draw"),
                          "close_a": ev.closing.get("away")})
        else:
            low.append("нет коэффициентов")
        # ---- травмы / составы (API-Football)
        r["lineups_known"] = 0
        r["lam_mult"], r["mu_mult"] = 1.0, 1.0
        r["absent_h"], r["absent_a"] = "", ""
        if af is not None and fx.source == "API-Football" and budget > 0:
            try:
                tp = players.get(fx.league, pd.DataFrame())
                tph = tp[tp["team_canon"] == home] if not tp.empty else tp
                tpa = tp[tp["team_canon"] == away] if not tp.empty else tp
                miss_h: list[str] = []
                miss_a: list[str] = []
                mins_to_ko = (fx.kickoff_utc - now).total_seconds() / 60
                if -15 < mins_to_ko < 60:
                    lineups, _ = af.lineups(fx.source_id)
                    budget -= 1
                    for lu in lineups:
                        if lu.team_id == fx.home_team_id:
                            miss_h += missing_from_lineup(lu.starters, tph)
                        elif lu.team_id == fx.away_team_id:
                            miss_a += missing_from_lineup(lu.starters, tpa)
                    r["lineups_known"] = 2 if lineups else 0
                injuries, _ = af.injuries(fx.source_id)
                budget -= 1
                r["lineups_known"] = max(r["lineups_known"], 1)
                for inj in injuries:
                    if inj.kind.lower().startswith("missing"):
                        (miss_h if inj.team_id == fx.home_team_id else miss_a).append(inj.player)
                ih, ia = absence_impact(miss_h, tph), absence_impact(miss_a, tpa)
                # атака хозяев слабеет от их потерь, оборона хозяев слабеет -> гости забивают больше
                r["lam_mult"] = ih.own_goal_mult * ia.opp_goal_mult
                r["mu_mult"] = ia.own_goal_mult * ih.opp_goal_mult
                r["absent_h"] = ", ".join(f"{p} ({s:.0%})" for p, s in ih.players[:4]) or ", ".join(miss_h[:4])
                r["absent_a"] = ", ".join(f"{p} ({s:.0%})" for p, s in ia.players[:4]) or ", ".join(miss_a[:4])
                if (miss_h or miss_a) and not (ih.data_available and ia.data_available):
                    warnings.append(f"{home}–{away}: травмы есть, но нет данных о вкладе игроков (Understat) — без корректировки")
                    r["lam_mult"], r["mu_mult"] = 1.0, 1.0
            except ApiError as exc:
                warnings.append(f"{home}–{away}: травмы/составы недоступны: {exc}")
        if ce_matcher is not None:
            for side, col in ((r["home"], "clubelo_h"), (r["away"], "clubelo_a")):
                nm = ce_matcher.match("*", side)
                r[col] = clubelo_map.get(nm, np.nan) if nm else np.nan
        r["low_reasons"] = "; ".join(low)
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================ replay
def replay_rows(hist: pd.DataFrame, day: date, leagues: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Матчи даты из исторического датасета. Возвращает (строки без результата, факт)."""
    d = pd.Timestamp(day)
    fx = hist[(hist["date"] == d) & hist["league"].isin(leagues)].copy()
    actual = fx[["league", "home", "away", "hg", "ag"]].copy()
    fx["hg"] = np.nan
    fx["ag"] = np.nan
    fx["is_fixture"] = True
    fx["fixture_id"] = [f"hist-{i}" for i in range(len(fx))]
    fx["low_reasons"] = ""
    for k in ("h", "d", "a"):
        fx[f"book_{k}"] = "max (≈17 букмекеров)"
    fx["book_o25"] = fx["book_u25"] = "max (≈17 букмекеров)"
    fx["lam_mult"], fx["mu_mult"] = 1.0, 1.0
    fx["absent_h"] = fx["absent_a"] = ""
    return fx, actual


# ============================================================================ main entry
def run(day: date, leagues: list[str], mode: str = "live", refresh: bool = False, use_understat: bool = True,
        bankroll: float = SETTINGS.bankroll, betting: BettingParams = BETTING, cache: Cache | None = None) -> RunResult:
    cache = cache or Cache(SETTINGS.cache_db)
    res = RunResult(day=day, mode=mode, leagues=leagues)
    res.generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    hist, hist_ts = load_history(cache, leagues, refresh)
    hist = hist[hist["date"] >= HIST_START]
    fixtures_df = pd.DataFrame()
    actual = pd.DataFrame()
    af = None
    if mode == "replay":
        fixtures_df, actual = replay_rows(hist, day, leagues)
        hist = hist[hist["date"] < pd.Timestamp(day)]  # всё начиная с этой даты скрыто
        if fixtures_df.empty:
            res.warnings.append(f"В историческом датасете нет матчей выбранных лиг на {day}")
    else:
        fixtures, af = fetch_live_fixtures(cache, day, leagues, refresh, res.warnings)
        hist = extend_history_live(cache, af, hist, leagues, day, refresh, res.warnings)
        players: dict[str, pd.DataFrame] = {}
        if use_understat and fixtures:
            hist, players = fetch_understat(cache, hist, leagues, day, refresh, res.warnings)
        if fixtures is None:
            res.warnings.append("Список матчей получить не удалось — прогнозы на дату не строятся "
                                "(данные не выдумываются). Ниже — только результаты бэктеста.")
        elif not fixtures:
            res.warnings.append(f"На {day} ({SETTINGS.timezone}) матчей выбранных лиг нет")
        else:
            fixtures_df = live_fixture_rows(cache, day, leagues, hist, fixtures, af, refresh, res.warnings, players)
    last_hist = hist["date"].max()
    if pd.notna(last_hist) and (pd.Timestamp(day) - last_hist).days > 10 and mode == "live":
        res.warnings.append(f"История результатов заканчивается {last_hist.date()} — форма/Elo могут быть устаревшими; "
                            "затронутые матчи помечаются LOW DATA")

    # ---------------- единый путь: история + матчи дня -> признаки -> DC -> бэктест
    all_rows = pd.concat([hist.assign(is_fixture=False), fixtures_df], ignore_index=True) if not fixtures_df.empty \
        else hist.assign(is_fixture=False)
    all_rows["is_fixture"] = all_rows["is_fixture"].fillna(False).astype(bool)
    feat = prepare_features(all_rows)
    extra = [pd.Timestamp(day)] if not fixtures_df.empty else []
    feat = dc_walkforward(feat, DC_START, MODEL, extra_refit_dates=extra)
    fx_ix = feat.index[feat["is_fixture"]]
    for ix in fx_ix:  # корректировка DC на отсутствующих игроков (до GBM и ансамбля)
        lm, mm = float(feat.at[ix, "lam_mult"]), float(feat.at[ix, "mu_mult"])
        if abs(lm - 1) > 1e-9 or abs(mm - 1) > 1e-9:
            feat = adjust_dc(feat, ix, lm, mm)
    # отметка LOW DATA
    feat["low_data"] = False
    feat["low_reasons"] = feat["low_reasons"].astype(object) if "low_reasons" in feat else ""
    if len(fx_ix):
        stale = []
        for ix in fx_ix:
            r = feat.loc[ix]
            reasons = [x for x in str(r.get("low_reasons", "") or "").split("; ") if x]
            for side in ("home", "away"):
                prev = hist[(hist["league"] == r["league"]) & ((hist["home"] == r[side]) | (hist["away"] == r[side]))]
                if prev.empty:
                    reasons.append(f"нет истории: {r[side]}")
                    continue
                last = prev.iloc[prev["date"].argmax()]
                if (r["date"] - last["date"]).days > 45 and int(last["season"]) == int(r["season"]):
                    reasons.append(f"устаревшая история {r[side]} (последний матч {last['date'].date()})")
            stale.append("; ".join(dict.fromkeys(reasons)))
        feat.loc[fx_ix, "low_reasons"] = stale
        feat.loc[fx_ix, "low_data"] = [bool(s) for s in stale]

    max_season = int(feat["season"].max())
    test_seasons = list(range(FIRST_TEST_SEASON, max_season + 1))
    bt = run_backtest(feat, test_seasons, GBM_TRAIN_START, betting)
    res.backtest = bt
    if len(fx_ix):
        preds = bt.preds[bt.preds["is_fixture"]].copy()
        res.matches, res.details = build_match_outputs(preds, bt, bankroll, betting, actual)
    res.sources = summarize_sources(cache, hist_ts)
    return res


def summarize_sources(cache: Cache, hist_ts: str) -> list[dict[str, Any]]:
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for e in cache.run_log():
        key = (e["source"], e["status"])
        a = agg.setdefault(key, {"source": e["source"], "status": e["status"], "requests": 0, "from_cache": 0,
                                 "first_fetched_at": e["fetched_at"], "last_fetched_at": e["fetched_at"], "detail": ""})
        a["requests"] += 1
        a["from_cache"] += int(e["from_cache"])
        a["first_fetched_at"] = min(a["first_fetched_at"], e["fetched_at"])
        a["last_fetched_at"] = max(a["last_fetched_at"], e["fetched_at"])
        if e["detail"]:
            a["detail"] = e["detail"][:180]
    return list(agg.values())


# ============================================================================ outputs
OUTCOMES = [("П1", "ens_ph", "max_h", "odds_h", "book_h", "close_h"), ("X", "ens_pd", "max_d", "odds_d", "book_d", "close_d"),
            ("П2", "ens_pa", "max_a", "odds_a", "book_a", "close_a")]
CONF_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def explain(r: pd.Series) -> list[str]:
    lines = []
    ed = r["elo_diff"]
    fav = r["home"] if ed > 0 else r["away"]
    lines.append(f"Elo: перевес {fav} {abs(ed):.0f} п. (с учётом поля); ожидаемые голы Dixon-Coles "
                 f"{r['dc_lam']:.2f}–{r['dc_mu']:.2f}.")
    xg = []
    if pd.notna(r.get("h_xgf_5")) and pd.notna(r.get("a_xgf_5")):
        src = "xG" if str(r.get("xg_src")) == "understat" else "xG-прокси"
        xg.append(f"{src} за 5 матчей (за/против): {r['home']} {r['h_xgf_5']:.2f}/{r['h_xga_5']:.2f}, "
                  f"{r['away']} {r['a_xgf_5']:.2f}/{r['a_xga_5']:.2f}")
    if pd.notna(r.get("h_rest_days")) and pd.notna(r.get("a_rest_days")) and abs(r["h_rest_days"] - r["a_rest_days"]) >= 3:
        xg.append(f"отдых {r['h_rest_days']:.0f} vs {r['a_rest_days']:.0f} дн.")
    if r.get("derby", 0) == 1:
        xg.append("дерби")
    for side, nm in (("h", r["home"]), ("a", r["away"])):
        if r.get(f"releg_battle_{side}", 0) == 1:
            xg.append(f"{nm} борется за выживание")
        if r.get(f"title_race_{side}", 0) == 1:
            xg.append(f"{nm} в борьбе за титул")
    if xg:
        lines.append("; ".join(xg) + ".")
    mk = r[["mk_ph", "mk_pd", "mk_pa"]].to_numpy(dtype=float)
    mod = r[["mod_ph", "mod_pd", "mod_pa"]].to_numpy(dtype=float)
    tail = []
    if np.all(np.isfinite(mk)):
        k = int(np.argmax(np.abs(mod - mk)))
        tail.append(f"модели без рынка: {['П1', 'X', 'П2'][k]} {mod[k]:.0%} против рынка {mk[k]:.0%}")
    else:
        tail.append("рыночной линии нет — прогноз только по моделям")
    if r.get("absent_h") or r.get("absent_a"):
        tail.append(f"отсутствуют: {r['home']}: {r.get('absent_h') or '—'}; {r['away']}: {r.get('absent_a') or '—'}")
    lines.append("; ".join(tail) + ".")
    return lines[:3]


def build_match_outputs(preds: pd.DataFrame, bt: BacktestResult, bankroll: float, betting: BettingParams,
                        actual: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows, details = [], []
    act = {(a.league, a.home, a.away): (a.hg, a.ag) for a in actual.itertuples(index=False)} if not actual.empty else {}
    for _, r in preds.iterrows():
        p1x2 = r[["ens_ph", "ens_pd", "ens_pa"]].to_numpy(dtype=float)
        m = score_matrix(float(r["dc_lam"]), float(r["dc_mu"]), float(r["dc_rho"])) if pd.notna(r["dc_lam"]) else None
        top = top_scores(rescale_matrix_1x2(m, p1x2), 3) if m is not None and np.all(np.isfinite(p1x2)) else []
        options = []
        for name, pc, oc, rc, bc, cc in OUTCOMES:
            options.append((name, float(r[pc]), r.get(oc), r.get(rc), r.get(bc), r.get(cc)))
        po = float(r["ens_po25"]) if pd.notna(r.get("ens_po25")) else np.nan
        options.append(("ТБ 2.5", po, r.get("max_o25"), r.get("odds_o25"), r.get("book_o25"), np.nan))
        options.append(("ТМ 2.5", 1 - po if np.isfinite(po) else np.nan, r.get("max_u25"), r.get("odds_u25"),
                        r.get("book_u25"), np.nan))
        markets = []
        for name, p, odds, ref, book, close in options:
            fl = lambda x: float(x) if x is not None and pd.notna(x) else np.nan  # noqa: E731
            best_raw, ref = fl(odds), fl(ref)
            odds = sane_price(best_raw, ref)
            if np.isfinite(best_raw) and odds != best_raw:
                book = "референс (лучшая цена отброшена как выброс)"
            ok, v, stake = bet_decision(p, odds, r["confidence"], betting) if np.isfinite(odds) else (False, np.nan, 0.0)
            markets.append({"market": name, "p": p, "fair_odds": fair_odds(p) if np.isfinite(p) else np.nan,
                            "best_odds": odds, "bookmaker": book if isinstance(book, str) else "",
                            "value": v, "bet": ok, "stake_frac": stake, "stake": stake * bankroll,
                            "closing_odds": float(close) if close is not None and pd.notna(close) else np.nan})
        bets = [mk for mk in markets if mk["bet"]]
        best_bet = max(bets, key=lambda x: x["value"]) if bets else None
        key = (r["league"], r["home"], r["away"])
        fact = act.get(key)
        row = {
            "league": r["league"], "date": r["date"].date().isoformat(), "time_local": r.get("time"),
            "home": r["home"], "away": r["away"], "p_home": p1x2[0], "p_draw": p1x2[1], "p_away": p1x2[2],
            "p_over25": po, "p_under25": 1 - po if np.isfinite(po) else np.nan, "p_btts": float(r["ens_btts"]),
            "top_scores": ", ".join(f"{s} ({p:.0%})" for s, p in top),
            "mk_home": r["mk_ph"], "mk_draw": r["mk_pd"], "mk_away": r["mk_pa"],
            "dc_home": r["dc_ph"], "elo_home": r["elo_ph"], "gbm_home": r["gbm_ph"],
            "confidence": r["confidence"], "conf_score": r["conf_score"], "completeness": r["completeness"],
            "disagreement": r["disagreement"], "market_gap": r["market_gap"],
            "low_data": bool(r.get("low_data", False)), "low_reasons": r.get("low_reasons", ""),
            "conf_reasons": r["conf_reasons"],
            "best_bet": f"{best_bet['market']} @ {best_bet['best_odds']:.2f}" if best_bet else "",
            "best_bet_market": best_bet["market"] if best_bet else "",
            "best_bet_value": best_bet["value"] if best_bet else np.nan,
            "best_bet_stake_frac": best_bet["stake_frac"] if best_bet else 0.0,
            "best_bet_stake": best_bet["stake"] if best_bet else 0.0,
            "best_bet_bookmaker": best_bet["bookmaker"] if best_bet else "",
            "explanation": " ".join(explain(r)),
            "actual": f"{int(fact[0])}-{int(fact[1])}" if fact else "",
            "referee": r.get("referee") if isinstance(r.get("referee"), str) else "",
            "odds_fetched_at": r.get("odds_fetched_at", ""), "fixture_fetched_at": r.get("fixture_fetched_at", ""),
        }
        if row["low_data"] and row["confidence"] != "LOW":
            row["confidence"] = "LOW"
        rows.append(row)
        details.append({**row, "markets": markets, "explain_lines": explain(r)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_o"] = out["confidence"].map(CONF_ORDER)
        out = out.sort_values(["_o", "conf_score"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
        order = {(d["league"], d["home"], d["away"]): i for i, d in enumerate(out.to_dict("records"))}
        details.sort(key=lambda d: order[(d["league"], d["home"], d["away"])])
    return out, details
