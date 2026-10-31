"""Исторические матчи: football-data.co.uk (результаты, удары, коэффициенты) + ClubElo.

Источник по умолчанию — открытое зеркало на GitHub (xgabora/Club-Football-Match-Data),
собранное из football-data.co.uk и clubelo.com. Файл кэшируется на диске, время загрузки
пишется в журнал источников.
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import LEAGUES, SETTINGS, League
from data.cache import Cache
from data.http import CachedHttp

log = logging.getLogger(__name__)

CANON_COLS = [
    "league", "season", "date", "time", "home", "away", "hg", "ag",
    "hsh", "ash", "hst", "ast", "hxg", "axg",
    "odds_h", "odds_d", "odds_a", "max_h", "max_d", "max_a",
    "odds_o25", "odds_u25", "max_o25", "max_u25",
    "close_h", "close_d", "close_a",
    "clubelo_h", "clubelo_a", "src",
]

_RENAME = {
    "MatchDate": "date", "MatchTime": "time", "HomeTeam": "home", "AwayTeam": "away",
    "FTHome": "hg", "FTAway": "ag", "HomeShots": "hsh", "AwayShots": "ash",
    "HomeTarget": "hst", "AwayTarget": "ast",
    "OddHome": "odds_h", "OddDraw": "odds_d", "OddAway": "odds_a",
    "MaxHome": "max_h", "MaxDraw": "max_d", "MaxAway": "max_a",
    "Over25": "odds_o25", "Under25": "odds_u25", "MaxOver25": "max_o25", "MaxUnder25": "max_u25",
    "HomeElo": "clubelo_h", "AwayElo": "clubelo_a",
}


def season_of(date: pd.Series | pd.Timestamp, calendar: str = "aug-may"):
    """Сезон = год начала (для aug-may: июль..июнь), либо календарный год (правило по месяцу)."""
    if isinstance(date, pd.Timestamp):
        return date.year if calendar == "calendar" or date.month >= 7 else date.year - 1
    d = pd.to_datetime(date)
    if calendar == "calendar":
        return d.dt.year
    return np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)


def infer_seasons(dates: pd.Series, calendar: str = "aug-may", min_gap_days: int = 21) -> np.ndarray:
    """Сезон по летним паузам в календаре лиги (устойчиво к сезону 2019/20, доигранному в июле-августе 2020).

    Новый сезон начинается после паузы >= min_gap_days, если следующая игра — в июле..октябре.
    """
    rule = np.asarray(season_of(pd.Series(pd.to_datetime(dates)), calendar))
    if calendar == "calendar":
        return rule
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    uniq = np.sort(d.unique())
    seasons = {}
    current = None
    prev = None
    for u in uniq:
        ts = pd.Timestamp(u)
        if current is None:
            current = ts.year if ts.month >= 7 else ts.year - 1
        elif ((ts - prev).days >= min_gap_days and 7 <= ts.month <= 10) or \
                ((ts - prev).days >= 10 and 7 <= ts.month <= 8 and prev.month in (5, 6, 7)):
            current = ts.year
        seasons[u] = current
        prev = ts
    return d.map(lambda x: seasons[np.datetime64(x)]).to_numpy(dtype=int)


def download_history(cache: Cache, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Скачивает (или берёт из кэша) исторический CSV. Возвращает (raw_df, fetched_at)."""
    http = CachedHttp(cache, "football-data.co.uk (GitHub mirror)", min_interval_s=1.0, timeout_s=180)
    frames = []
    fetched = ""
    for url in SETTINGS.history_urls:
        body, fetched, _ = http.get_raw(url, ttl_s=7 * 86400, refresh=refresh)
        frames.append(pd.read_csv(io.BytesIO(body), low_memory=False))
    return pd.concat(frames, ignore_index=True), fetched


def to_canonical(raw: pd.DataFrame, leagues: list[League]) -> pd.DataFrame:
    div_map = {lg.hist_division: lg for lg in leagues if lg.hist_division}
    df = raw[raw["Division"].isin(div_map)].rename(columns=_RENAME).copy()
    df["league"] = df["Division"].map({d: lg.code for d, lg in div_map.items()})
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = 0
    for d, lg in div_map.items():
        m = df["Division"] == d
        df.loc[m, "season"] = infer_seasons(df.loc[m, "date"], lg.calendar)
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["src"] = "football-data.co.uk"
    df = unify_team_names(df)
    df = df.dropna(subset=["hg", "ag"])
    df = drop_playoff_matches(df)
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df["season"] = df["season"].astype(int)
    # некорректные коэффициенты (<=1) считаем отсутствующими
    for c in ["odds_h", "odds_d", "odds_a", "max_h", "max_d", "max_a", "odds_o25", "odds_u25", "max_o25", "max_u25"]:
        df.loc[df[c] <= 1.0, c] = np.nan
    df = df[CANON_COLS].sort_values(["date", "league", "home"]).reset_index(drop=True)
    return df


def unify_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Склеивает варианты написания одной команды внутри лиги ("Nott'm Forest"/"Nottm Forest", пробелы)."""
    out = df.copy()
    out["home"] = out["home"].astype(str).str.strip()
    out["away"] = out["away"].astype(str).str.strip()
    for lg, g in out.groupby("league"):
        counts = pd.concat([g["home"], g["away"]]).value_counts()
        canon: dict[str, str] = {}
        for name in counts.index:  # от частых к редким
            canon.setdefault(re.sub(r"[^a-z]", "", name.lower()), name)
        mapping = {n: canon[re.sub(r"[^a-z]", "", n.lower())] for n in counts.index}
        mapping = {k: v for k, v in mapping.items() if k != v}
        if mapping:
            log.info("[%s] объединены написания команд: %s", lg, mapping)
            m = out["league"] == lg
            out.loc[m, "home"] = out.loc[m, "home"].replace(mapping)
            out.loc[m, "away"] = out.loc[m, "away"].replace(mapping)
    return out


def drop_playoff_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает стыковые матчи (команды с единичными играми в сезоне), чтобы не искажать таблицу и Elo."""
    keep = np.ones(len(df), dtype=bool)
    d = df.reset_index(drop=True)
    for (_, _), g in d.groupby(["league", "season"]):
        cnt = pd.concat([g["home"], g["away"]]).value_counts()
        med = cnt.median()
        if med < 20:   # незавершённый текущий сезон — не трогаем
            continue
        rare = set(cnt[cnt < 0.25 * med].index)
        if rare:
            bad = g["home"].isin(rare) | g["away"].isin(rare)
            keep[g.index[bad]] = False
    if (~keep).sum():
        log.info("Исключено стыковых/кубковых матчей: %d", int((~keep).sum()))
    return d[keep].reset_index(drop=True)


def load_history(cache: Cache, league_codes: list[str], refresh: bool = False) -> tuple[pd.DataFrame, str]:
    raw, fetched = download_history(cache, refresh)
    df = to_canonical(raw, [LEAGUES[c] for c in league_codes])
    log.info("История: %d матчей, %s — %s (загружено %s UTC)", len(df),
             df["date"].min().date() if len(df) else "-", df["date"].max().date() if len(df) else "-", fetched)
    return df, fetched


def append_matches(hist: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Добавляет недавние матчи (например, из API-Football) без дублей по (league, date, home, away)."""
    if extra is None or extra.empty:
        return hist
    extra = extra.copy()
    for c in CANON_COLS:
        if c not in extra.columns:
            extra[c] = np.nan
    both = pd.concat([hist, extra[CANON_COLS]], ignore_index=True)
    both["_k"] = both["league"] + "|" + both["date"].dt.strftime("%Y-%m-%d") + "|" + both["home"] + "|" + both["away"]
    both = both.drop_duplicates("_k", keep="first").drop(columns="_k")
    return both.sort_values(["date", "league", "home"]).reset_index(drop=True)


def merge_xg(hist: pd.DataFrame, xg: pd.DataFrame) -> pd.DataFrame:
    """Подмешивает реальный xG (Understat) по (league, date, home, away); названия уже нормализованы."""
    if xg is None or xg.empty:
        return hist
    key = ["league", "date", "home", "away"]
    m = hist.drop(columns=["hxg", "axg"]).merge(xg[key + ["hxg", "axg"]], on=key, how="left")
    log.info("xG Understat сопоставлен для %d из %d матчей", int(m["hxg"].notna().sum()), len(m))
    return m[CANON_COLS]


def save_parquet_like(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
