"""Влияние отсутствующих игроков (травмы, дисквалификации, не вошедшие в подтверждённый состав).

Вес игрока = его доля в (xG + xA) команды (атака) и доля минут (общая роль/оборона).
Корректировка ожидаемых голов Dixon-Coles — эвристика с ограничением ±15%: исторических данных
о травмах в бэктесте нет, поэтому этот блок НЕ проверен бэктестом и в отчёте помечается отдельно.
"""
from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

ATT_REPLACEMENT_LOSS = 0.5   # замена восполняет ~половину вклада в атаке
DEF_MINUTES_EFFECT = 0.12    # потеря полной «единицы» основы (11 игроков) -> +12% к голам соперника
CAP = 0.15


@dataclass
class AbsenceImpact:
    att_share: float = 0.0           # доля (xG+xA) команды у отсутствующих
    minutes_share: float = 0.0       # доля минут команды у отсутствующих (×11 = «стартовые слоты»)
    players: list[tuple[str, float]] = field(default_factory=list)  # (имя, доля xG+xA)
    unmatched: list[str] = field(default_factory=list)
    data_available: bool = False

    @property
    def own_goal_mult(self) -> float:
        return max(1 - CAP, 1 - ATT_REPLACEMENT_LOSS * self.att_share)

    @property
    def opp_goal_mult(self) -> float:
        return min(1 + CAP, 1 + DEF_MINUTES_EFFECT * self.minutes_share)


def _find(name: str, names: list[str]) -> str | None:
    if name in names:
        return name
    # API-Football часто даёт "M. Salah", Understat — "Mohamed Salah"
    last = name.split()[-1].lower() if name.split() else name.lower()
    cands = [n for n in names if n.split() and n.split()[-1].lower() == last]
    if len(cands) == 1:
        return cands[0]
    best = difflib.get_close_matches(name, names, n=1, cutoff=0.8)
    return best[0] if best else None


def absence_impact(missing: list[str], team_players: pd.DataFrame) -> AbsenceImpact:
    """team_players: колонки player, minutes, xG, xA (Understat, текущий сезон)."""
    imp = AbsenceImpact()
    if team_players is None or team_players.empty:
        imp.unmatched = list(missing)
        return imp
    imp.data_available = True
    tp = team_players.copy()
    tp["contrib"] = tp["xG"].fillna(0) + tp["xA"].fillna(0)
    tot_c = float(tp["contrib"].sum()) or 1.0
    tot_m = float(tp["minutes"].sum()) or 1.0
    names = tp["player"].tolist()
    seen: set[str] = set()
    for p in missing:
        m = _find(p, names)
        if m is None:
            imp.unmatched.append(p)
            continue
        if m in seen:
            continue
        seen.add(m)
        row = tp[tp["player"] == m].iloc[0]
        share = float(row["contrib"]) / tot_c
        imp.att_share += share
        imp.minutes_share += float(row["minutes"]) / tot_m
        imp.players.append((m, share))
    imp.players.sort(key=lambda x: -x[1])
    return imp


def missing_from_lineup(starters: list[str], team_players: pd.DataFrame, top_n: int = 11) -> list[str]:
    """Игроки из топ-N по минутам, не попавшие в подтверждённый старт."""
    if team_players is None or team_players.empty or not starters:
        return []
    core = team_players.sort_values("minutes", ascending=False).head(top_n)["player"].tolist()
    return [p for p in core if _find(p, starters) is None]
