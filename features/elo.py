"""Собственный Elo: учёт преимущества поля, разницы мячей (как World Football Elo) и регрессии к среднему
между сезонами. Рейтинги до матча считаются строго по прошлым матчам (без заглядывания в будущее)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def goal_multiplier(diff: int) -> float:
    d = abs(diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11 + d) / 8


@dataclass
class EloSystem:
    k: float = 20.0
    home_adv: float = 60.0
    season_regress: float = 0.2
    init: float = 1500.0
    ratings: dict[tuple[str, str], float] = field(default_factory=dict)   # (league, team) -> elo
    last_season: dict[str, int] = field(default_factory=dict)

    def _new_team_rating(self, league: str) -> float:
        vals = [v for (lg, _), v in self.ratings.items() if lg == league]
        if len(vals) < 4:
            return self.init
        return float(np.mean(sorted(vals)[: max(3, len(vals) // 5)]))  # уровень нижней пятой части лиги

    def get(self, league: str, team: str) -> float:
        key = (league, team)
        if key not in self.ratings:
            self.ratings[key] = self._new_team_rating(league)
        return self.ratings[key]

    def start_season(self, league: str, season: int, active_teams: set[str] | None = None) -> None:
        if self.last_season.get(league) == season:
            return
        keys = [k for k in self.ratings if k[0] == league]
        if keys:
            mean = float(np.mean([self.ratings[k] for k in keys]))
            for k in keys:
                self.ratings[k] = mean + (1 - self.season_regress) * (self.ratings[k] - mean)
        self.last_season[league] = season

    def expected(self, rh: float, ra: float) -> float:
        return 1.0 / (1.0 + 10 ** (-(rh + self.home_adv - ra) / 400.0))

    def update(self, league: str, home: str, away: str, hg: int, ag: int) -> None:
        rh, ra = self.get(league, home), self.get(league, away)
        e = self.expected(rh, ra)
        s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = self.k * goal_multiplier(hg - ag) * (s - e)
        self.ratings[(league, home)] = rh + delta
        self.ratings[(league, away)] = ra - delta

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Проходит матчи хронологически, возвращает df с колонками elo_h, elo_a (до матча).

        Матчи без результата (hg NaN) получают рейтинг, но не обновляют его.
        """
        out_h = np.empty(len(df))
        out_a = np.empty(len(df))
        d = df.reset_index(drop=True)
        for i, r in enumerate(d.itertuples(index=False)):
            self.start_season(r.league, r.season)
            out_h[i] = self.get(r.league, r.home)
            out_a[i] = self.get(r.league, r.away)
            if not (pd.isna(r.hg) or pd.isna(r.ag)):
                self.update(r.league, r.home, r.away, int(r.hg), int(r.ag))
        res = df.copy()
        res["elo_h"] = out_h
        res["elo_a"] = out_a
        res["elo_diff"] = out_h + self.home_adv - out_a
        return res
