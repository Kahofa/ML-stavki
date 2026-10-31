"""Признаки формы команд: xG/xGA (реальный xG или прокси по ударам), голы, очки, отдых, плотность календаря.

Все признаки матча считаются строго по матчам ДО его даты (shift), с экспоненциальным затуханием по матчам.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def fit_xg_proxy(df: pd.DataFrame) -> tuple[float, float]:
    """Голы ≈ a * удары_в_створ + b * удары_мимо (МНК без свободного члена) на переданных данных."""
    d = df.dropna(subset=["hsh", "ash", "hst", "ast"])
    if len(d) < 200:
        return 0.30, 0.03
    sot = np.concatenate([d["hst"], d["ast"]]).astype(float)
    off = np.clip(np.concatenate([d["hsh"] - d["hst"], d["ash"] - d["ast"]]).astype(float), 0, None)
    g = np.concatenate([d["hg"], d["ag"]]).astype(float)
    X = np.column_stack([sot, off])
    coef, *_ = np.linalg.lstsq(X, g, rcond=None)
    a, b = float(coef[0]), float(max(coef[1], 0.0))
    return a, b


def rolling_xg_proxy(df: pd.DataFrame) -> dict[tuple[str, int], tuple[float, float]]:
    """Коэффициенты xG-прокси для каждой (лига, сезон) по двум ПРЕДЫДУЩИМ сезонам этой лиги.

    Нужен потому, что определение «удара в створ» у football-data.co.uk менялось (например, АПЛ до 2013/14).
    """
    coefs: dict[tuple[str, int], tuple[float, float]] = {}
    done = df[df["hg"].notna()]
    for (lg, s) in df[["league", "season"]].drop_duplicates().itertuples(index=False):
        prev = done[(done["league"] == lg) & (done["season"].between(s - 2, s - 1))]
        coefs[(lg, int(s))] = fit_xg_proxy(prev) if len(prev.dropna(subset=["hst"])) >= 200 else (np.nan, np.nan)
    return coefs


def add_xg_columns(df: pd.DataFrame, coefs: dict[tuple[str, int], tuple[float, float]]) -> pd.DataFrame:
    """xg_h/xg_a: реальный xG, если есть (Understat), иначе прокси по ударам; источник в xg_src."""
    out = df.copy()
    ab = np.array([coefs.get((lg, int(s)), (np.nan, np.nan)) for lg, s in zip(out["league"], out["season"])])
    a, b = ab[:, 0], ab[:, 1]
    ph = a * out["hst"] + b * (out["hsh"] - out["hst"]).clip(lower=0)
    pa = a * out["ast"] + b * (out["ash"] - out["ast"]).clip(lower=0)
    real = out["hxg"].notna() & out["axg"].notna()
    out["xg_h"] = np.where(real, out["hxg"], ph)
    out["xg_a"] = np.where(real, out["axg"], pa)
    out["xg_src"] = np.where(real, "understat", np.where(np.isfinite(ph), "shots-proxy", "none"))
    return out


def _decayed_prior_mean(values: np.ndarray, n: int, halflife: float) -> np.ndarray:
    """Для каждой позиции t — взвешенное среднее values[t-n..t-1] (веса 0.5^(k/halflife), k=0 — последний)."""
    if len(values) == 0:
        return np.empty(0)
    w = 0.5 ** (np.arange(n) / halflife)
    ok = np.isfinite(values).astype(float)
    v = np.nan_to_num(values)
    num = np.convolve(v, w)[: len(v)]
    den = np.convolve(ok, w)[: len(v)]
    # сдвиг на 1: только прошлые матчи
    num = np.concatenate([[0.0], num[:-1]])
    den = np.concatenate([[0.0], den[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        res = np.where(den > 1e-9, num / den, np.nan)
    return res


def team_long(df: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на (матч, команда)."""
    base = df.reset_index().rename(columns={"index": "mid"})
    h = pd.DataFrame({"mid": base["mid"], "league": base["league"], "season": base["season"], "date": base["date"],
                      "team": base["home"], "opp": base["away"], "is_home": 1,
                      "gf": base["hg"], "ga": base["ag"], "xgf": base["xg_h"], "xga": base["xg_a"]})
    a = pd.DataFrame({"mid": base["mid"], "league": base["league"], "season": base["season"], "date": base["date"],
                      "team": base["away"], "opp": base["home"], "is_home": 0,
                      "gf": base["ag"], "ga": base["hg"], "xgf": base["xg_a"], "xga": base["xg_h"]})
    t = pd.concat([h, a], ignore_index=True)
    t["pts"] = np.where(t["gf"] > t["ga"], 3.0, np.where(t["gf"] == t["ga"], 1.0, 0.0))
    t.loc[t["gf"].isna(), "pts"] = np.nan
    return t.sort_values(["league", "team", "date", "mid"]).reset_index(drop=True)


FORM_STATS = ("xgf", "xga", "gf", "ga", "pts")


def compute_form(df: pd.DataFrame, halflife: float = 4.0) -> pd.DataFrame:
    """Возвращает df с признаками формы для home (h_*) и away (a_*) команд."""
    t = team_long(df)
    feats: dict[str, np.ndarray] = {}
    cols_overall = {f"{s}_{n}": np.full(len(t), np.nan) for s in FORM_STATS for n in (5, 10)}
    cols_venue = {f"{s}_venue5": np.full(len(t), np.nan) for s in ("xgf", "xga", "gf", "ga", "pts")}
    rest = np.full(len(t), np.nan)
    last14 = np.zeros(len(t))
    nprior = np.zeros(len(t))
    for (_, _), g in t.groupby(["league", "team"], sort=False):
        ix = g.index.to_numpy()
        for s in FORM_STATS:
            vals = g[s].to_numpy(dtype=float)
            for n in (5, 10):
                cols_overall[f"{s}_{n}"][ix] = _decayed_prior_mean(vals, n, halflife)
        dates = g["date"].to_numpy()
        dd = np.diff(dates).astype("timedelta64[D]").astype(float)
        rest[ix] = np.concatenate([[np.nan], dd])
        nprior[ix] = np.arange(len(ix))
        dnum = dates.astype("datetime64[D]").astype(np.int64)
        left = np.searchsorted(dnum, dnum - 14, side="left")
        last14[ix] = np.arange(len(ix)) - left
        for venue in (0, 1):
            sub = g[g["is_home"] == venue]
            six = sub.index.to_numpy()
            for s in ("xgf", "xga", "gf", "ga", "pts"):
                cols_venue[f"{s}_venue5"][six] = _decayed_prior_mean(sub[s].to_numpy(dtype=float), 5, halflife)
    for k, v in {**cols_overall, **cols_venue}.items():
        t[k] = v
    t["rest_days"] = np.clip(rest, 0, 30)
    t["games_14d"] = last14
    t["n_prior"] = nprior
    keep = list(cols_overall) + list(cols_venue) + ["rest_days", "games_14d", "n_prior"]
    home = t[t["is_home"] == 1].set_index("mid")[keep].add_prefix("h_")
    away = t[t["is_home"] == 0].set_index("mid")[keep].add_prefix("a_")
    out = df.join(home).join(away)
    return out


# ---------------------------------------------------------------------------------- таблица / мотивация
def compute_standings(df: pd.DataFrame, relegation_spots: int = 3) -> pd.DataFrame:
    """Положение в таблице ДО даты матча (по матчам того же сезона)."""
    res = {c: np.full(len(df), np.nan) for c in
           ("pos_h", "pos_a", "pts_h", "pts_a", "releg_gap_h", "releg_gap_a", "top_gap_h", "top_gap_a", "progress")}
    d = df.reset_index(drop=True)
    for (_, _), g in d.groupby(["league", "season"], sort=False):
        teams = sorted(set(g["home"]) | set(g["away"]))
        nt = len(teams)
        pts = dict.fromkeys(teams, 0.0)
        gd = dict.fromkeys(teams, 0.0)
        gf = dict.fromkeys(teams, 0.0)
        played = dict.fromkeys(teams, 0)
        for _, day in g.groupby("date", sort=True):
            order = sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t], t))
            pos = {t: i + 1 for i, t in enumerate(order)}
            releg_line = pts[order[nt - relegation_spots]] if nt > relegation_spots else 0.0
            leader = pts[order[0]]
            prog = np.mean(list(played.values())) / max(1, 2 * (nt - 1))
            ix = day.index.to_numpy()
            res["pos_h"][ix] = day["home"].map(pos)
            res["pos_a"][ix] = day["away"].map(pos)
            res["pts_h"][ix] = day["home"].map(pts)
            res["pts_a"][ix] = day["away"].map(pts)
            res["releg_gap_h"][ix] = day["home"].map(pts) - releg_line
            res["releg_gap_a"][ix] = day["away"].map(pts) - releg_line
            res["top_gap_h"][ix] = leader - day["home"].map(pts)
            res["top_gap_a"][ix] = leader - day["away"].map(pts)
            res["progress"][ix] = prog
            for r in day.itertuples(index=False):
                if pd.isna(r.hg):
                    continue
                hg, ag = int(r.hg), int(r.ag)
                played[r.home] += 1
                played[r.away] += 1
                gd[r.home] += hg - ag
                gd[r.away] += ag - hg
                gf[r.home] += hg
                gf[r.away] += ag
                pts[r.home] += 3 if hg > ag else 1 if hg == ag else 0
                pts[r.away] += 3 if ag > hg else 1 if hg == ag else 0
    out = df.copy()
    for k, v in res.items():
        out[k] = v
    return out


DERBIES: set[frozenset[str]] = {frozenset(p) for p in [
    ("Arsenal", "Tottenham"), ("Liverpool", "Everton"), ("Man United", "Man City"), ("Man United", "Liverpool"),
    ("Chelsea", "Arsenal"), ("Chelsea", "Tottenham"), ("Newcastle", "Sunderland"), ("Aston Villa", "Birmingham"),
    ("Real Madrid", "Barcelona"), ("Real Madrid", "Ath Madrid"), ("Sevilla", "Betis"), ("Barcelona", "Espanol"),
    ("Ath Bilbao", "Sociedad"), ("Valencia", "Levante"),
    ("Inter", "Milan"), ("Roma", "Lazio"), ("Juventus", "Torino"), ("Genoa", "Sampdoria"), ("Juventus", "Inter"),
    ("Dortmund", "Schalke 04"), ("Bayern Munich", "Dortmund"), ("Hamburg", "Werder Bremen"), ("FC Koln", "M'gladbach"),
    ("Paris SG", "Marseille"), ("Lyon", "St Etienne"), ("Nice", "Monaco"), ("Lens", "Lille"),
    ("Spartak Moscow", "CSKA Moscow"), ("Spartak Moscow", "Zenit"), ("CSKA Moscow", "Lokomotiv Moscow"),
    ("Spartak Moscow", "Lokomotiv Moscow"), ("Dynamo Moscow", "Spartak Moscow"),
    ("Kairat", "Astana"), ("Kairat", "Ordabasy"), ("Tobol", "Astana"),
]}


def add_motivation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    late = out["progress"] > 0.5
    out["releg_battle_h"] = (late & (out["releg_gap_h"].abs() <= 4)).astype(float)
    out["releg_battle_a"] = (late & (out["releg_gap_a"].abs() <= 4)).astype(float)
    out["title_race_h"] = (late & (out["top_gap_h"] <= 4)).astype(float)
    out["title_race_a"] = (late & (out["top_gap_a"] <= 4)).astype(float)
    out["derby"] = [float(frozenset((h, a)) in DERBIES) for h, a in zip(out["home"], out["away"])]
    return out
