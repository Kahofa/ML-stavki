"""Модель Dixon-Coles (1997): двумерный Пуассон с поправкой tau на низкие счета и затуханием по времени.

lambda_home = exp(att_h + def_a + home), mu_away = exp(att_a + def_h).
Веса матчей w = exp(-xi * days_ago). Ridge-регуляризация тянет параметры к априорным значениям
(0 для старожилов лиги, «уровень новичка» для команд с малой историей) — стабилизирует оценки
для промоутированных клубов.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

log = logging.getLogger(__name__)

PROMOTED_ATT_PRIOR = -0.20
PROMOTED_DEF_PRIOR = 0.20


@dataclass
class DixonColesModel:
    xi: float = 0.0019
    l2: float = 2.0
    max_goals: int = 10
    teams: list[str] = field(default_factory=list)
    att: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dfn: np.ndarray = field(default_factory=lambda: np.zeros(0))
    home_adv: float = 0.25
    rho: float = -0.05
    team_weight: dict[str, float] = field(default_factory=dict)  # эффективное число матчей (с весами)
    fitted: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp, warm: "DixonColesModel | None" = None) -> "DixonColesModel":
        d = df[df["date"] < ref_date]
        if len(d) < 50:
            raise ValueError("Слишком мало матчей для Dixon-Coles")
        teams = sorted(set(d["home"]) | set(d["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = d["home"].map(idx).to_numpy()
        ai = d["away"].map(idx).to_numpy()
        x = d["hg"].to_numpy(dtype=float)
        y = d["ag"].to_numpy(dtype=float)
        days = (ref_date - d["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * days)
        tw = np.bincount(hi, w, n) + np.bincount(ai, w, n)
        # априор: команды с малой (взвешенной) историей в лиге — уровень «новичка»
        median_tw = np.median(tw)
        newcomer = tw < 0.35 * median_tw
        prior_att = np.where(newcomer, PROMOTED_ATT_PRIOR, 0.0)
        prior_def = np.where(newcomer, PROMOTED_DEF_PRIOR, 0.0)

        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        l2 = self.l2

        def unpack(p: np.ndarray):
            return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]

        def nll(p: np.ndarray) -> tuple[float, np.ndarray]:
            att, dfn, home, rho = unpack(p)
            ll_ = att[hi] + dfn[ai] + home
            lm_ = att[ai] + dfn[hi]
            lam = np.exp(ll_)
            mu = np.exp(lm_)
            tau = np.ones_like(lam)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            tau = np.maximum(tau, 1e-10)
            ll = w * (np.log(tau) + x * ll_ - lam + y * lm_ - mu)
            # градиенты по log-lambda и log-mu
            g_l = x - lam
            g_m = y - mu
            dtl = np.zeros_like(lam)
            dtm = np.zeros_like(lam)
            dtr = np.zeros_like(lam)
            dtl[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
            dtm[m00] = dtl[m00]
            dtr[m00] = -lam[m00] * mu[m00] / tau[m00]
            dtl[m01] = lam[m01] * rho / tau[m01]
            dtr[m01] = lam[m01] / tau[m01]
            dtm[m10] = mu[m10] * rho / tau[m10]
            dtr[m10] = mu[m10] / tau[m10]
            dtr[m11] = -1 / tau[m11]
            gl = w * (g_l + dtl)
            gm = w * (g_m + dtm)
            g_att = np.bincount(hi, gl, n) + np.bincount(ai, gm, n)
            g_def = np.bincount(ai, gl, n) + np.bincount(hi, gm, n)
            g_home = gl.sum()
            g_rho = (w * dtr).sum()
            pen = l2 * (np.sum((att - prior_att) ** 2) + np.sum((dfn - prior_def) ** 2)) + 100.0 * att.sum() ** 2
            g_att_pen = 2 * l2 * (att - prior_att) + 200.0 * att.sum()
            g_def_pen = 2 * l2 * (dfn - prior_def)
            f = -ll.sum() + pen
            grad = np.concatenate([-g_att + g_att_pen, -g_def + g_def_pen, [-g_home, -g_rho]])
            return float(f), grad

        p0 = np.concatenate([prior_att, prior_def, [0.25, -0.05]])
        if warm is not None and warm.fitted:
            widx = {t: i for i, t in enumerate(warm.teams)}
            for t, i in idx.items():
                if t in widx:
                    p0[i] = warm.att[widx[t]]
                    p0[n + i] = warm.dfn[widx[t]]
            p0[2 * n], p0[2 * n + 1] = warm.home_adv, warm.rho
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1.5), (-0.2, 0.2)]
        res = minimize(nll, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        att, dfn, home, rho = unpack(res.x)
        self.teams, self.att, self.dfn = teams, att, dfn
        self.home_adv, self.rho = float(home), float(rho)
        self.team_weight = dict(zip(teams, tw))
        self.fitted = True
        return self

    # ------------------------------------------------------------------ predict
    def rates(self, home: str, away: str) -> tuple[float, float, bool]:
        """(lambda, mu, known): известны ли обе команды модели."""
        idx = {t: i for i, t in enumerate(self.teams)}
        known = home in idx and away in idx
        ah, dh = (self.att[idx[home]], self.dfn[idx[home]]) if home in idx else (PROMOTED_ATT_PRIOR, PROMOTED_DEF_PRIOR)
        aa, da = (self.att[idx[away]], self.dfn[idx[away]]) if away in idx else (PROMOTED_ATT_PRIOR, PROMOTED_DEF_PRIOR)
        return float(np.exp(ah + da + self.home_adv)), float(np.exp(aa + dh)), known

    def score_matrix(self, lam: float, mu: float) -> np.ndarray:
        return score_matrix(lam, mu, self.rho, self.max_goals)

    def predict(self, home: str, away: str, lam_mult: float = 1.0, mu_mult: float = 1.0) -> dict[str, float | np.ndarray]:
        lam, mu, known = self.rates(home, away)
        lam *= lam_mult
        mu *= mu_mult
        m = self.score_matrix(lam, mu)
        return {"lam": lam, "mu": mu, "known": known, "matrix": m, **markets_from_matrix(m)}


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def markets_from_matrix(m: np.ndarray) -> dict[str, float]:
    i, j = np.indices(m.shape)
    return {
        "p_home": float(m[i > j].sum()), "p_draw": float(np.trace(m)), "p_away": float(m[i < j].sum()),
        "p_over25": float(m[(i + j) > 2.5].sum()), "p_btts": float(m[(i > 0) & (j > 0)].sum()),
    }


def rescale_matrix_1x2(m: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Масштабирует области H/D/A матрицы счетов под итоговые вероятности ансамбля 1X2."""
    i, j = np.indices(m.shape)
    out = m.copy()
    for mask, t in ((i > j, target[0]), (i == j, target[1]), (i < j, target[2])):
        s = m[mask].sum()
        if s > 0:
            out[mask] *= t / s
    return out / out.sum()


def top_scores(m: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    flat = np.argsort(m, axis=None)[::-1][:k]
    return [(f"{a}-{b}", float(m[a, b])) for a, b in (np.unravel_index(f, m.shape) for f in flat)]
