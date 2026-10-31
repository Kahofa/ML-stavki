import numpy as np
import pandas as pd
import pytest

from backtest.metrics import betting_metrics, brier
from data.history import infer_seasons, unify_team_names
from data.teams import TeamMatcher
from features.elo import EloSystem, goal_multiplier
from features.team_form import _decayed_prior_mean, add_xg_columns, compute_form, compute_standings


def _league(n_days=12):
    teams = ["A", "B", "C", "D"]
    rows = []
    d = pd.Timestamp("2021-08-10")
    rng = np.random.default_rng(0)
    for k in range(n_days):
        pairs = [(teams[(k + i) % 4], teams[(k + i + 1 + k % 2) % 4]) for i in (0, 2)]
        for h, a in pairs:
            if h == a:
                continue
            rows.append({"league": "L", "season": 2021, "date": d, "home": h, "away": a,
                         "hg": int(rng.integers(0, 4)), "ag": int(rng.integers(0, 3)),
                         "hsh": 12, "ash": 9, "hst": 5, "ast": 3, "hxg": np.nan, "axg": np.nan})
        d += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_decayed_mean_uses_only_past():
    v = np.array([1.0, 2.0, 3.0, 100.0])
    out = _decayed_prior_mean(v, 5, 4.0)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(1.0)
    assert out[3] < 3.0  # значение 100 текущего матча не участвует


def test_form_has_no_leakage():
    df = _league()
    df = add_xg_columns(df, {("L", 2021): (0.3, 0.03)})
    f1 = compute_form(df)
    df2 = df.copy()
    last = df2.index[-1]
    df2.loc[last, ["hg", "ag"]] = [9, 0]   # меняем результат последнего матча
    f2 = compute_form(add_xg_columns(df2, {("L", 2021): (0.3, 0.03)}))
    cols = [c for c in f1.columns if c.startswith(("h_", "a_"))]
    pd.testing.assert_frame_equal(f1.loc[[last], cols], f2.loc[[last], cols])


def test_standings_before_match_date():
    df = _league(3)
    st = compute_standings(df)
    first_day = st[st["date"] == st["date"].min()]
    assert (first_day["pts_h"] == 0).all() and (first_day["pts_a"] == 0).all()


def test_elo_zero_sum_and_home_advantage():
    e = EloSystem(k=20, home_adv=60)
    e.update("L", "A", "B", 2, 0)
    assert e.ratings[("L", "A")] + e.ratings[("L", "B")] == pytest.approx(3000)
    assert e.expected(1500, 1500) > 0.5
    assert goal_multiplier(1) == 1 and goal_multiplier(3) == pytest.approx(14 / 8)


def test_infer_seasons_covid_restart():
    # сезон 2019/20 прерван в марте, доигран в июне-июле 2020 (еженедельные туры), 2020/21 стартовал в сентябре
    restart = pd.date_range("2020-06-17", "2020-07-26", freq="7D").tolist()
    dates = pd.to_datetime(["2019-08-10", "2020-03-08", *restart, "2020-09-12", "2021-05-23", "2021-08-14"])
    s = list(infer_seasons(pd.Series(dates)))
    assert s == [2019, 2019] + [2019] * len(restart) + [2020, 2020, 2021]


def test_unify_team_names():
    df = pd.DataFrame({"league": ["E"] * 3, "home": ["Nott'm Forest", "Nottm Forest", "Ajax "],
                       "away": ["X", "Nottm Forest", "Nottm Forest"]})
    out = unify_team_names(df)
    assert set(out["home"]) == {"Nottm Forest", "Ajax"}


def test_team_matcher():
    m = TeamMatcher({"EPL": ["Man United", "Man City", "Nottm Forest", "Wolves"],
                     "LaLiga": ["Ath Madrid", "Real Madrid"]})
    assert m.match("EPL", "Manchester United") == "Man United"
    assert m.match("EPL", "Wolverhampton Wanderers") == "Wolves"
    assert m.match("EPL", "Nottingham Forest") == "Nottm Forest"   # алиас "Nott'm Forest" ~ "Nottm Forest"
    assert m.match("LaLiga", "Atlético de Madrid") == "Ath Madrid"
    assert m.match("LaLiga", "Totally Unknown FC") is None


def test_brier_and_betting_metrics():
    p = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert brier(p, np.array([0, 1])) == 0
    bets = pd.DataFrame({"date": pd.to_datetime(["2021-01-01", "2021-01-02"]), "odds": [2.0, 3.0],
                         "won": [1, 0], "stake_frac": [0.02, 0.02], "closing_odds": [1.9, np.nan]})
    m = betting_metrics(bets)
    assert m["roi_flat"] == pytest.approx(0.0)
    assert m["clv_n"] == 1 and m["clv_mean"] == pytest.approx(2.0 / 1.9 - 1)
