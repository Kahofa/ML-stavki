import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from models.calibration import Calibrator, choose_calibrator
from models.decision import bet_decision, confidence, kelly_stake, sane_price, value
from models.dixon_coles import (DixonColesModel, markets_from_matrix, rescale_matrix_1x2, score_matrix,
                                top_scores)
from models.elo_model import EloOrdinalModel
from models.ensemble import LinearPool, log_loss
from models.market import devig, overround, power, shin


# ------------------------------------------------------------------ market
@pytest.mark.parametrize("method", ["shin", "power", "multiplicative"])
def test_devig_sums_to_one(method):
    p = devig(np.array([1.50, 4.20, 6.50]), method)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(p > 0)


def test_shin_and_power_shift_margin_to_longshot():
    odds = np.array([1.30, 5.50, 11.0])
    pi = 1 / odds
    mult = pi / pi.sum()
    for p in (shin(odds), power(odds)):
        # favourite-longshot bias: фаворит получает больше, чем при пропорциональном удалении маржи
        assert p[0] > mult[0]
        assert p[2] < mult[2]


def test_devig_invalid_returns_nan():
    assert np.isnan(devig(np.array([1.0, 3.0, 4.0]))).all()
    assert np.isnan(devig(np.array([np.nan, 3.0, 4.0]))).all()


def test_overround():
    assert overround(np.array([2.0, 2.0])) == pytest.approx(0.0)
    assert overround(np.array([1.9, 1.9])) > 0


# ------------------------------------------------------------------ Dixon-Coles
def _simulate(n_rounds=6, seed=1):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(10)]
    att = np.linspace(-0.4, 0.4, 10)
    dfn = -np.linspace(-0.3, 0.3, 10)
    rows = []
    day = pd.Timestamp("2020-01-01")
    for r in range(n_rounds):
        for i, h in enumerate(teams):
            for j, a in enumerate(teams):
                if i == j:
                    continue
                lam = np.exp(att[i] + dfn[j] + 0.3)
                mu = np.exp(att[j] + dfn[i])
                rows.append({"date": day, "home": h, "away": a, "hg": rng.poisson(lam), "ag": rng.poisson(mu)})
                day += pd.Timedelta(hours=6)
    return pd.DataFrame(rows), att


def test_dixon_coles_recovers_strengths():
    df, att = _simulate()
    m = DixonColesModel(xi=0.0, l2=0.1).fit(df, df["date"].max() + pd.Timedelta(days=1))
    est = np.array([m.att[m.teams.index(f"T{i}")] for i in range(10)])
    assert np.corrcoef(est, att)[0, 1] > 0.9
    assert 0.15 < m.home_adv < 0.45
    p = m.predict("T9", "T0")
    assert p["p_home"] > p["p_away"]
    assert p["p_home"] + p["p_draw"] + p["p_away"] == pytest.approx(1.0, abs=1e-9)


def test_dixon_coles_gradient_matches_numeric():
    df, _ = _simulate(n_rounds=2, seed=3)
    ref = df["date"].max() + pd.Timedelta(days=1)
    captured = {}
    import scipy.optimize as so

    orig = so.minimize

    def spy(fun, x0, jac=None, **kw):
        captured["f"], captured["x0"] = fun, x0
        return orig(fun, x0, jac=jac, **kw)

    import models.dixon_coles as dcm
    dcm.minimize = spy
    try:
        DixonColesModel(xi=0.001, l2=1.0).fit(df, ref)
    finally:
        dcm.minimize = orig
    f = captured["f"]
    x = captured["x0"] + np.random.default_rng(0).normal(0, 0.1, len(captured["x0"]))
    x[-1] = -0.08
    err = check_grad(lambda z: f(z)[0], lambda z: f(z)[1], x)
    assert err / (np.linalg.norm(f(x)[1]) + 1e-9) < 1e-4


def test_score_matrix_and_markets():
    m = score_matrix(1.6, 1.1, -0.08)
    assert m.sum() == pytest.approx(1.0)
    mk = markets_from_matrix(m)
    assert mk["p_home"] + mk["p_draw"] + mk["p_away"] == pytest.approx(1.0)
    assert 0 < mk["p_over25"] < 1 and 0 < mk["p_btts"] < 1
    # rho<0 повышает вероятность 0-0 и 1-1 относительно независимого Пуассона
    indep = score_matrix(1.6, 1.1, 0.0)
    assert m[0, 0] > indep[0, 0] and m[1, 1] > indep[1, 1]


def test_rescale_matrix_hits_target():
    m = score_matrix(1.4, 1.2, -0.05)
    target = np.array([0.5, 0.3, 0.2])
    r = rescale_matrix_1x2(m, target)
    mk = markets_from_matrix(r)
    assert [mk["p_home"], mk["p_draw"], mk["p_away"]] == pytest.approx(list(target), abs=1e-9)
    assert len(top_scores(r, 3)) == 3


# ------------------------------------------------------------------ Elo ordinal
def test_elo_ordinal_monotonic_and_fit():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 150, 5000)
    true = EloOrdinalModel(beta=0.006, c1=-0.9, c2=0.3)
    p = true.predict_proba(diff)
    y = np.array([rng.choice(3, p=row) for row in p])
    m = EloOrdinalModel().fit(diff, y)
    assert m.beta == pytest.approx(0.006, rel=0.15)
    ph = m.predict_proba(np.array([-200.0, 0.0, 200.0]))[:, 0]
    assert ph[0] < ph[1] < ph[2]


# ------------------------------------------------------------------ ensemble / calibration
def test_linear_pool_prefers_informative_model():
    rng = np.random.default_rng(1)
    n = 3000
    true = rng.dirichlet([3, 2, 2], n)
    y = np.array([rng.choice(3, p=r) for r in true])
    noise = rng.dirichlet([1, 1, 1], n)
    pool = LinearPool().fit({"good": true, "noise": noise}, y)
    assert pool.weights[0] > 0.8
    out = pool.predict({"good": true, "noise": np.full_like(noise, np.nan)})
    assert np.allclose(out, true)  # нет модели -> вес перераспределяется


def test_calibrator_fixes_overconfidence():
    rng = np.random.default_rng(2)
    p_true = rng.uniform(0.1, 0.9, 6000)
    y = (rng.uniform(size=6000) < p_true).astype(int)
    over = np.clip(0.5 + (p_true - 0.5) * 1.8, 0.01, 0.99)  # пере-уверенная модель
    cal, scores = choose_calibrator(over, y)
    assert cal.method in ("platt", "isotonic")
    assert log_loss(cal.transform(over), y) < log_loss(over, y)
    multi = rng.dirichlet([2, 2, 2], 500)
    c3 = Calibrator("isotonic").fit(multi, rng.choice(3, 500))
    assert np.allclose(c3.transform(multi).sum(axis=1), 1.0)


# ------------------------------------------------------------------ decisions
def test_value_and_kelly():
    assert value(0.5, 2.2) == pytest.approx(0.1)
    # полный Келли = 0.1/1.2 = 0.0833; 1/4 -> 0.0208 -> потолок 2%
    assert kelly_stake(0.5, 2.2) == pytest.approx(0.02)
    assert kelly_stake(0.4, 2.2) == 0.0
    assert kelly_stake(0.52, 2.0) == pytest.approx(0.25 * 0.04 / 1.0)


def test_bet_decision_rules():
    assert bet_decision(0.5, 2.2, "MEDIUM")[0]
    assert not bet_decision(0.5, 2.2, "LOW")[0]          # уверенность ниже MEDIUM
    assert not bet_decision(0.5, 2.05, "HIGH")[0]        # value 2.5% < 3%


def test_sane_price():
    assert sane_price(2.25, 1.50) == 1.50
    assert sane_price(1.60, 1.50) == 1.60
    assert sane_price(np.nan, 1.50) == 1.50


def test_confidence_levels():
    mk = np.array([0.5, 0.27, 0.23])
    agree = [mk + 0.01 * np.array([1, -0.5, -0.5])] * 3
    assert confidence(1.0, agree, mk, mk).level == "HIGH"
    split = [np.array([0.7, 0.2, 0.1]), np.array([0.4, 0.3, 0.3]), np.array([0.5, 0.3, 0.2])]
    assert confidence(1.0, split, np.array([0.53, 0.27, 0.2]), mk).level == "LOW"
    assert confidence(1.0, agree, mk, mk, low_data=True).level == "LOW"
    assert confidence(1.0, agree, mk, None).level == "LOW"   # нет рынка
    assert confidence(0.7, agree, mk + np.array([0.04, -0.02, -0.02]), mk).level == "MEDIUM"
