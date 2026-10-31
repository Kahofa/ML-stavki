"""Сборка признаков для всех матчей (исторических и будущих) единым кодом — без утечки будущего.

Будущие матчи добавляются в тот же DataFrame с пустым результатом: все признаки считаются
только по матчам до их даты, поэтому бэктест и прогноз на сегодня проходят один и тот же путь.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from config import MODEL, ModelParams
from features.elo import EloSystem
from features.team_form import add_motivation, add_xg_columns, compute_form, compute_standings, rolling_xg_proxy
from models.dixon_coles import DixonColesModel
from models.market import devig

log = logging.getLogger(__name__)

OUTCOME_MAP = {"H": 0, "D": 1, "A": 2}


def outcome(df: pd.DataFrame) -> np.ndarray:
    y = np.where(df["hg"] > df["ag"], 0, np.where(df["hg"] == df["ag"], 1, 2))
    return np.where(df["hg"].isna(), -1, y)


def add_market(df: pd.DataFrame, method_1x2: str = "shin", method_2way: str = "power") -> pd.DataFrame:
    out = df.copy()
    o = out[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    p = np.vstack([devig(r, method_1x2) for r in o]) if len(o) else np.empty((0, 3))
    out["mk_ph"], out["mk_pd"], out["mk_pa"] = p[:, 0], p[:, 1], p[:, 2]
    ou = out[["odds_o25", "odds_u25"]].to_numpy(dtype=float)
    q = np.vstack([devig(r, method_2way) for r in ou]) if len(ou) else np.empty((0, 2))
    out["mk_po25"] = q[:, 0]
    out["overround_1x2"] = (1 / o).sum(axis=1) - 1
    return out


def prepare_features(df: pd.DataFrame, params: ModelParams = MODEL) -> pd.DataFrame:
    """Elo, xG-прокси, форма, таблица, мотивация, рыночные вероятности."""
    t0 = time.time()
    d = df.sort_values(["date", "league", "home"]).reset_index(drop=True)
    proxy = rolling_xg_proxy(d)
    d = add_xg_columns(d, proxy)
    d = EloSystem(k=params.elo_k, home_adv=params.elo_home_adv, season_regress=params.elo_season_regress).run(d)
    d["clubelo_diff"] = d["clubelo_h"] - d["clubelo_a"]
    d = compute_form(d, halflife=params.form_halflife)
    d = compute_standings(d)
    d = add_motivation(d)
    d = add_market(d)
    d["y"] = outcome(d)
    log.info("Признаки построены: %d матчей за %.1f c", len(d), time.time() - t0)
    return d


def dc_walkforward(df: pd.DataFrame, start: pd.Timestamp, params: ModelParams = MODEL,
                   extra_refit_dates: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    """Out-of-sample прогнозы Dixon-Coles: переобучение каждые `dc_refit_days` дней, по каждой лиге,
    только на матчах строго раньше даты переобучения (окно dc_window_days)."""
    t0 = time.time()
    out = df.copy()
    cols = ["dc_lam", "dc_mu", "dc_rho", "dc_ph", "dc_pd", "dc_pa", "dc_po25", "dc_btts", "dc_known"]
    for c in cols:
        out[c] = np.nan
    n_fits = 0
    for lg, g in out[out["date"] >= start].groupby("league"):
        hist_all = out[(out["league"] == lg) & out["hg"].notna()]
        refits = pd.date_range(start, g["date"].max() + pd.Timedelta(days=1), freq=f"{params.dc_refit_days}D")
        refits = sorted(set(refits) | {d for d in (extra_refit_dates or []) if d >= start})
        model: DixonColesModel | None = None
        for i, rd in enumerate(refits):
            nxt = refits[i + 1] if i + 1 < len(refits) else pd.Timestamp.max
            block = g[(g["date"] >= rd) & (g["date"] < nxt)]
            if block.empty:
                continue
            train = hist_all[(hist_all["date"] < rd) & (hist_all["date"] >= rd - pd.Timedelta(days=params.dc_window_days))]
            if len(train) < 100:
                continue
            try:
                model = DixonColesModel(xi=params.dc_xi, l2=params.dc_l2, max_goals=params.max_goals).fit(train, rd, warm=model)
                n_fits += 1
            except ValueError as exc:
                log.warning("DC %s %s: %s", lg, rd.date(), exc)
                continue
            for ix, r in block.iterrows():
                pr = model.predict(r["home"], r["away"])
                out.loc[ix, cols] = [pr["lam"], pr["mu"], model.rho, pr["p_home"], pr["p_draw"], pr["p_away"],
                                     pr["p_over25"], pr["p_btts"], float(pr["known"])]
    log.info("Dixon-Coles walk-forward: %d подгонок за %.1f c", n_fits, time.time() - t0)
    return out


def adjust_dc(df: pd.DataFrame, ix, lam_mult: float, mu_mult: float) -> pd.DataFrame:
    """Пересчитывает DC-рынки строки после корректировки ожидаемых голов (травмы/составы)."""
    from models.dixon_coles import markets_from_matrix, score_matrix
    r = df.loc[ix]
    if pd.isna(r["dc_lam"]):
        return df
    lam, mu = float(r["dc_lam"]) * lam_mult, float(r["dc_mu"]) * mu_mult
    mk = markets_from_matrix(score_matrix(lam, mu, float(r["dc_rho"])))
    df.loc[ix, ["dc_lam", "dc_mu", "dc_ph", "dc_pd", "dc_pa", "dc_po25", "dc_btts"]] = [
        lam, mu, mk["p_home"], mk["p_draw"], mk["p_away"], mk["p_over25"], mk["p_btts"]]
    return df


def data_completeness(df: pd.DataFrame) -> pd.Series:
    """Доля доступных ключевых входов (0..1) по каждому матчу."""
    checks = pd.DataFrame({
        "odds": df[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1),
        "xg_home": df["h_xgf_5"].notna(),
        "xg_away": df["a_xgf_5"].notna(),
        "hist_home": df["h_n_prior"] >= 5,
        "hist_away": df["a_n_prior"] >= 5,
        "dc_known": df["dc_known"] == 1,
        "clubelo": df["clubelo_diff"].notna(),
    })
    checks = checks.astype(float)
    if "lineups_known" in df:  # только для матчей дня (в истории данных о составах нет)
        checks["lineups_or_injuries"] = np.where(df["lineups_known"].isna(), np.nan, (df["lineups_known"] > 0).astype(float))
    return checks.mean(axis=1, skipna=True)
