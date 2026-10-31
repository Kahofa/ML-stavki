"""Walk-forward бэктест по сезонам без заглядывания в будущее.

Для каждого сезона S:
  * Elo-ordinal обучается на сезонах [S-5, S-1];
  * LightGBM — на всех сезонах признаков < S (ранняя остановка по S-1, затем дообучение);
  * Dixon-Coles — уже out-of-sample (еженедельное переобучение, см. features.builder);
  * веса ансамбля и калибратор подбираются на OOS-прогнозах сезонов S-2 и S-1, применяются к S.
Сравнение с бейзлайном «рыночные вероятности» (Bet365, маржа удалена методом Шина).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtest.metrics import betting_metrics, bootstrap_roi_ci, prob_metrics
from config import BETTING, BettingParams
from features.builder import data_completeness
from models.calibration import Calibrator, choose_calibrator
from models.decision import bet_decision, confidence, sane_price
from models.elo_model import EloOrdinalModel
from models.ensemble import LinearPool
from models.gbm import GBMModel

log = logging.getLogger(__name__)

VALID_SEASONS = 2  # сколько предыдущих сезонов OOS-прогнозов используется для весов ансамбля и калибровки

P1X2 = {"dc": ["dc_ph", "dc_pd", "dc_pa"], "elo": ["elo_ph", "elo_pd", "elo_pa"],
        "gbm": ["gbm_ph", "gbm_pd", "gbm_pa"], "market": ["mk_ph", "mk_pd", "mk_pa"]}


@dataclass
class SeasonArtifacts:
    season: int
    pool_full: LinearPool
    pool_model: LinearPool
    cal_full: Calibrator
    cal_model: Calibrator
    pool_ou: LinearPool | None
    cal_ou: Calibrator | None
    cal_btts: Calibrator
    cal_scores: dict[str, float]
    gbm: GBMModel
    elo: EloOrdinalModel


@dataclass
class BacktestResult:
    preds: pd.DataFrame
    artifacts: dict[int, SeasonArtifacts] = field(default_factory=dict)
    bets: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)


def _probs(df: pd.DataFrame, key: str) -> np.ndarray:
    return df[P1X2[key]].to_numpy(dtype=float)


def oos_elo(df: pd.DataFrame, seasons: list[int]) -> tuple[pd.DataFrame, dict[int, EloOrdinalModel]]:
    out = df.copy()
    for c in P1X2["elo"]:
        out[c] = np.nan
    models: dict[int, EloOrdinalModel] = {}
    for s in seasons:
        tr = out[(out["season"] >= s - 5) & (out["season"] < s) & (out["y"] >= 0)]
        m = EloOrdinalModel().fit(tr["elo_diff"].to_numpy(), tr["y"].to_numpy())
        models[s] = m
        ix = out["season"] == s
        out.loc[ix, P1X2["elo"]] = m.predict_proba(out.loc[ix, "elo_diff"].to_numpy())
    return out, models


def oos_gbm(df: pd.DataFrame, seasons: list[int], train_start: int) -> tuple[pd.DataFrame, dict[int, GBMModel]]:
    out = df.copy()
    for c in P1X2["gbm"]:
        out[c] = np.nan
    models: dict[int, GBMModel] = {}
    for s in seasons:
        tr = out[(out["season"] >= train_start) & (out["season"] < s - 1) & (out["y"] >= 0) & out["dc_ph"].notna()]
        va = out[(out["season"] == s - 1) & (out["y"] >= 0) & out["dc_ph"].notna()]
        m = GBMModel().fit(tr, tr["y"].to_numpy(), va, va["y"].to_numpy())
        models[s] = m
        ix = (out["season"] == s) & out["dc_ph"].notna()
        out.loc[ix, P1X2["gbm"]] = m.predict_proba(out.loc[ix])
        log.info("GBM сезон %d: train=%d valid=%d, итераций=%d", s, len(tr), len(va), m.best_iter)
    return out, models


def _fit_season_artifacts(val: pd.DataFrame, season: int, gbm: GBMModel, elo: EloOrdinalModel) -> SeasonArtifacts:
    v = val[(val["y"] >= 0)].dropna(subset=P1X2["dc"] + P1X2["elo"] + P1X2["gbm"] + P1X2["market"])
    y = v["y"].to_numpy()
    pool_full = LinearPool().fit({k: _probs(v, k) for k in ("dc", "elo", "gbm", "market")}, y)
    pool_model = LinearPool().fit({k: _probs(v, k) for k in ("dc", "elo", "gbm")}, y)
    cal_full, scores = choose_calibrator(pool_full.predict({k: _probs(v, k) for k in pool_full.names}), y)
    cal_model, _ = choose_calibrator(pool_model.predict({k: _probs(v, k) for k in pool_model.names}), y)
    vo = val[(val["y"] >= 0)].dropna(subset=["dc_po25", "mk_po25"])
    pool_ou = cal_ou = None
    if len(vo) > 300:
        yo = ((vo["hg"] + vo["ag"]) > 2.5).astype(int).to_numpy()
        two = lambda p: np.column_stack([1 - p, p])  # noqa: E731
        pool_ou = LinearPool().fit({"dc": two(vo["dc_po25"].to_numpy()), "market": two(vo["mk_po25"].to_numpy())}, yo)
        pou = pool_ou.predict({"dc": two(vo["dc_po25"].to_numpy()), "market": two(vo["mk_po25"].to_numpy())})[:, 1]
        cal_ou, _ = choose_calibrator(pou, yo)
    vb = val[(val["y"] >= 0)].dropna(subset=["dc_btts"])
    yb = ((vb["hg"] > 0) & (vb["ag"] > 0)).astype(int).to_numpy()
    cal_btts, _ = choose_calibrator(vb["dc_btts"].to_numpy(), yb)
    return SeasonArtifacts(season, pool_full, pool_model, cal_full, cal_model, pool_ou, cal_ou, cal_btts, scores, gbm, elo)


def apply_artifacts(df: pd.DataFrame, art: SeasonArtifacts) -> pd.DataFrame:
    out = df.copy()
    full = art.pool_full.predict({k: _probs(out, k) for k in art.pool_full.names})
    model = art.pool_model.predict({k: _probs(out, k) for k in art.pool_model.names})
    full_c = art.cal_full.transform(full)
    model_c = art.cal_model.transform(model)
    # если рынка нет — используем модельный ансамбль (веса без рынка)
    no_mk = ~np.isfinite(_probs(out, "market")).all(axis=1)
    full_c[no_mk] = model_c[no_mk]
    out[["ens_ph", "ens_pd", "ens_pa"]] = full_c
    out[["mod_ph", "mod_pd", "mod_pa"]] = model_c
    if art.pool_ou is not None and art.cal_ou is not None:
        two = lambda p: np.column_stack([1 - p, p])  # noqa: E731
        pou = art.pool_ou.predict({"dc": two(out["dc_po25"].to_numpy()), "market": two(out["mk_po25"].to_numpy())})[:, 1]
        out["ens_po25"] = art.cal_ou.transform(pou)
    else:
        out["ens_po25"] = out["dc_po25"]
    out["ens_btts"] = art.cal_btts.transform(out["dc_btts"].to_numpy())
    return out


def add_confidence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    comp = data_completeness(out)
    levels, scores, dis, gaps, reasons = [], [], [], [], []
    for i, r in enumerate(out.itertuples(index=False)):
        models = [np.array([r.dc_ph, r.dc_pd, r.dc_pa]), np.array([r.elo_ph, r.elo_pd, r.elo_pa]),
                  np.array([r.gbm_ph, r.gbm_pd, r.gbm_pa])]
        mk = np.array([r.mk_ph, r.mk_pd, r.mk_pa])
        final = np.array([r.mod_ph, r.mod_pd, r.mod_pa])  # разрыв меряем для модели без рынка
        low = bool(getattr(r, "low_data", False)) or r.h_n_prior < 3 or r.a_n_prior < 3
        c = confidence(float(comp.iloc[i]), models, final, mk, low_data=low)
        levels.append(c.level)
        scores.append(c.score)
        dis.append(c.disagreement)
        gaps.append(c.market_gap)
        reasons.append("; ".join(c.reasons))
    out["completeness"] = comp.to_numpy()
    out["confidence"] = levels
    out["conf_score"] = scores
    out["disagreement"] = dis
    out["market_gap"] = gaps
    out["conf_reasons"] = reasons
    return out


def simulate_bets(df: pd.DataFrame, params: BettingParams = BETTING, price: str = "best") -> pd.DataFrame:
    """Одна ставка на рынок (1X2, тотал 2.5) на матч — исход с максимальным value."""
    rows = []
    for r in df.itertuples(index=False):
        if r.y < 0:
            continue
        pick = (lambda best, ref: sane_price(best, ref)) if price == "best" else (lambda best, ref: ref)  # noqa: E731
        markets = {
            "1X2": [("H", r.ens_ph, pick(r.max_h, r.odds_h), r.odds_h, r.y == 0),
                    ("D", r.ens_pd, pick(r.max_d, r.odds_d), r.odds_d, r.y == 1),
                    ("A", r.ens_pa, pick(r.max_a, r.odds_a), r.odds_a, r.y == 2)],
        }
        if np.isfinite(r.ens_po25):
            over = (r.hg + r.ag) > 2.5
            markets["OU2.5"] = [("Over", r.ens_po25, pick(r.max_o25, r.odds_o25), r.odds_o25, over),
                                ("Under", 1 - r.ens_po25, pick(r.max_u25, r.odds_u25), r.odds_u25, not over)]
        for mkt, opts in markets.items():
            best = None
            for name, p, odds, ref_odds, won in opts:
                if not (np.isfinite(odds) if odds is not None else False):
                    continue
                ok, v, stake = bet_decision(float(p), float(odds), r.confidence, params)
                if ok and (best is None or v > best["value"]):
                    best = {"date": r.date, "league": r.league, "season": r.season, "home": r.home, "away": r.away,
                            "market": mkt, "pick": name, "p": float(p), "odds": float(odds), "value": v,
                            "stake_frac": stake, "won": int(won), "confidence": r.confidence,
                            "closing_odds": np.nan}
            if best:
                rows.append(best)
    return pd.DataFrame(rows)


def run_backtest(feat: pd.DataFrame, test_seasons: list[int], gbm_train_start: int,
                 params: BettingParams = BETTING) -> BacktestResult:
    first = min(test_seasons) - VALID_SEASONS
    seasons = list(range(first, max(test_seasons) + 1))
    d, elo_models = oos_elo(feat, seasons)
    d, gbm_models = oos_gbm(d, seasons, gbm_train_start)
    artifacts: dict[int, SeasonArtifacts] = {}
    parts = []
    for s in test_seasons:
        art = _fit_season_artifacts(d[d["season"].between(s - VALID_SEASONS, s - 1)], s, gbm_models[s], elo_models[s])
        artifacts[s] = art
        part = apply_artifacts(d[d["season"] == s], art)
        parts.append(part)
        log.info("Сезон %d: веса %s, калибровка=%s", s, art.pool_full.as_dict(), art.cal_full.method)
    preds = add_confidence(pd.concat(parts))
    bets = simulate_bets(preds, params)
    res = BacktestResult(preds=preds, artifacts=artifacts, bets=bets)
    res.summary = summarize(preds, bets, artifacts, params)
    return res


def summarize(preds: pd.DataFrame, bets: pd.DataFrame, artifacts: dict[int, SeasonArtifacts],
              params: BettingParams = BETTING) -> dict[str, Any]:
    done = preds[preds["y"] >= 0]
    ev = done.dropna(subset=sum(P1X2.values(), []) + ["ens_ph"])
    y = ev["y"].to_numpy()
    models = {
        "Dixon-Coles": _probs(ev, "dc"), "Elo": _probs(ev, "elo"), "LightGBM": _probs(ev, "gbm"),
        "Ансамбль без рынка (калибр.)": ev[["mod_ph", "mod_pd", "mod_pa"]].to_numpy(),
        "Рынок (Bet365, Шин) — бейзлайн": _probs(ev, "market"),
        "Ансамбль + рынок (калибр.)": ev[["ens_ph", "ens_pd", "ens_pa"]].to_numpy(),
    }
    overall = {k: prob_metrics(v, y) for k, v in models.items()}
    by_season = {}
    for s, g in ev.groupby("season"):
        yy = g["y"].to_numpy()
        by_season[int(s)] = {
            "n": len(g),
            "market": prob_metrics(_probs(g, "market"), yy)["log_loss"],
            "model_only": prob_metrics(g[["mod_ph", "mod_pd", "mod_pa"]].to_numpy(), yy)["log_loss"],
            "ensemble": prob_metrics(g[["ens_ph", "ens_pd", "ens_pa"]].to_numpy(), yy)["log_loss"],
            "dc": prob_metrics(_probs(g, "dc"), yy)["log_loss"],
            "weights": artifacts[int(s)].pool_full.as_dict() if int(s) in artifacts else {},
            "calibration": artifacts[int(s)].cal_full.method if int(s) in artifacts else "",
        }
    by_league = {}
    for lg, g in ev.groupby("league"):
        yy = g["y"].to_numpy()
        by_league[lg] = {"n": len(g), "market": prob_metrics(_probs(g, "market"), yy)["log_loss"],
                         "model_only": prob_metrics(g[["mod_ph", "mod_pd", "mod_pa"]].to_numpy(), yy)["log_loss"],
                         "ensemble": prob_metrics(g[["ens_ph", "ens_pd", "ens_pa"]].to_numpy(), yy)["log_loss"]}
    ou = done.dropna(subset=["dc_po25", "mk_po25", "ens_po25"])
    yo = ((ou["hg"] + ou["ag"]) > 2.5).astype(int).to_numpy()
    ou_metrics = {"Dixon-Coles": prob_metrics(ou["dc_po25"].to_numpy(), yo),
                  "Рынок": prob_metrics(ou["mk_po25"].to_numpy(), yo),
                  "Ансамбль (калибр.)": prob_metrics(ou["ens_po25"].to_numpy(), yo)}
    bt = done.dropna(subset=["dc_btts"])
    yb = ((bt["hg"] > 0) & (bt["ag"] > 0)).astype(int).to_numpy()
    btts_metrics = {"Dixon-Coles": prob_metrics(bt["dc_btts"].to_numpy(), yb),
                    "Dixon-Coles (калибр.)": prob_metrics(bt["ens_btts"].to_numpy(), yb),
                    "Бейзлайн: частота прошлого сезона": prob_metrics(np.full(len(yb), yb.mean()), yb)}
    conf = {}
    for lvl, g in ev.groupby("confidence"):
        conf[lvl] = {"n": len(g), "share": len(g) / len(ev),
                     "ens_log_loss": prob_metrics(g[["ens_ph", "ens_pd", "ens_pa"]].to_numpy(), g["y"].to_numpy())["log_loss"],
                     "market_log_loss": prob_metrics(_probs(g, "market"), g["y"].to_numpy())["log_loss"],
                     "accuracy": prob_metrics(g[["ens_ph", "ens_pd", "ens_pa"]].to_numpy(), g["y"].to_numpy())["accuracy"]}
    bet_sum = betting_metrics(bets)
    bet_sum["roi_ci95"] = bootstrap_roi_ci(bets)
    bets_b365 = simulate_bets(done, params, price="bet365")
    bet_b365 = betting_metrics(bets_b365)
    bet_b365["roi_ci95"] = bootstrap_roi_ci(bets_b365)
    by_market = {m: betting_metrics(g) for m, g in bets.groupby("market")} if not bets.empty else {}
    by_season_bets = {int(s): betting_metrics(g) for s, g in bets.groupby("season")} if not bets.empty else {}
    mk_ll = overall["Рынок (Bet365, Шин) — бейзлайн"]["log_loss"]
    verdict = {
        "model_only_beats_market": overall["Ансамбль без рынка (калибр.)"]["log_loss"] < mk_ll,
        "ensemble_beats_market": overall["Ансамбль + рынок (калибр.)"]["log_loss"] < mk_ll,
        "delta_model_only": overall["Ансамбль без рынка (калибр.)"]["log_loss"] - mk_ll,
        "delta_ensemble": overall["Ансамбль + рынок (калибр.)"]["log_loss"] - mk_ll,
        "roi_positive_significant": bool(bet_sum["bets"] and bet_sum["roi_ci95"][0] > 0),
    }
    ce = ev.dropna(subset=["clubelo_diff"])
    elo_corr = float(np.corrcoef(ce["elo_h"] - ce["elo_a"], ce["clubelo_diff"])[0, 1]) if len(ce) > 100 else np.nan
    return {"overall_1x2": overall, "elo_vs_clubelo_corr": elo_corr, "by_season": by_season, "by_league": by_league, "ou25": ou_metrics,
            "btts": btts_metrics, "confidence": conf, "betting_best_price": bet_sum, "betting_bet365": bet_b365,
            "betting_by_market": by_market, "betting_by_season": by_season_bets, "verdict": verdict,
            "n_matches": len(ev), "seasons": sorted(int(s) for s in ev["season"].unique()),
            "clv_note": ("CLV не рассчитан: в открытом историческом датасете нет закрывающих коэффициентов. "
                         "В живом режиме закрывающая линия = последний снимок The Odds API до начала матча.")}
