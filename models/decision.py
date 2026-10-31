"""Решения по ставкам: value, дробный Келли, уровень уверенности."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import BETTING, BettingParams


def value(p: float, odds: float) -> float:
    """value = p * odds - 1 (ожидаемая доходность на единицу ставки)."""
    return p * odds - 1.0


def fair_odds(p: float) -> float:
    return float("inf") if p <= 0 else 1.0 / p


def kelly_stake(p: float, odds: float, params: BettingParams = BETTING) -> float:
    """Доля банка: fraction * (p*o - 1)/(o - 1), с потолком max_stake."""
    if odds <= 1.0:
        return 0.0
    f = (p * odds - 1.0) / (odds - 1.0)
    return float(np.clip(params.kelly_fraction * f, 0.0, params.max_stake))


@dataclass
class Confidence:
    level: str            # HIGH / MEDIUM / LOW
    score: float          # 0..1
    completeness: float
    disagreement: float
    market_gap: float
    reasons: list[str]


def confidence(completeness: float, model_probs: list[np.ndarray], final: np.ndarray,
               market: np.ndarray | None, low_data: bool = False) -> Confidence:
    """Уверенность = f(полнота данных, согласие моделей, разрыв с рынком).

    disagreement — максимальное по исходам стандартное отклонение вероятностей независимых моделей
    (DC, Elo, GBM); market_gap — max |итог − рынок| по исходам.
    """
    reasons: list[str] = []
    mp = [m for m in model_probs if m is not None and np.all(np.isfinite(m))]
    dis = float(np.max(np.std(np.vstack(mp), axis=0))) if len(mp) >= 2 else 0.15
    gap = float(np.max(np.abs(final - market))) if market is not None and np.all(np.isfinite(market)) else np.nan
    if low_data:
        reasons.append("LOW DATA")
    if completeness < 0.6:
        reasons.append(f"неполные данные ({completeness:.0%})")
    if dis > 0.10:
        reasons.append(f"модели расходятся (σ={dis:.2f})")
    if np.isnan(gap):
        reasons.append("нет рыночной линии")
    elif gap > 0.12:
        reasons.append(f"сильный разрыв с рынком ({gap:.0%})")
    score = completeness * (1 - min(1.0, dis / 0.2)) * (1 - min(1.0, (0.1 if np.isnan(gap) else gap) / 0.25))
    if low_data or completeness < 0.6 or dis > 0.10 or np.isnan(gap) or gap > 0.12:
        level = "LOW"
    elif completeness >= 0.85 and dis <= 0.05 and gap <= 0.06:
        level = "HIGH"
    else:
        level = "MEDIUM"
    return Confidence(level, float(score), float(completeness), dis, float(gap) if not np.isnan(gap) else np.nan, reasons)


MAX_PRICE_RATIO = 1.15


def sane_price(best: float, ref: float, max_ratio: float = MAX_PRICE_RATIO) -> float:
    """Лучшая цена, если она правдоподобна; иначе референсная.

    Цены, превышающие референсную линию более чем на 15%, почти всегда — устаревшая/ошибочная котировка
    (в датасете встречаются «max» 2.25 при рыночных 1.50). Ставить по ним в бэктесте — самообман.
    """
    if not np.isfinite(best):
        return ref
    if np.isfinite(ref) and best > ref * max_ratio:
        return ref
    return best


def bet_decision(p: float, odds: float, conf_level: str, params: BettingParams = BETTING) -> tuple[bool, float, float]:
    """(ставить?, value, доля банка). Только value > min_value и уверенность >= MEDIUM."""
    if not np.isfinite(odds) or not np.isfinite(p):
        return False, float("nan"), 0.0
    v = value(p, odds)
    ok = (v > params.min_value and conf_level in ("HIGH", "MEDIUM")
          and params.min_odds <= odds <= params.max_odds)
    return ok, v, kelly_stake(p, odds, params) if ok else 0.0
