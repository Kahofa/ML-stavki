"""Метрики качества прогнозов и ставок."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.ensemble import log_loss


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Мультиклассовый Brier: среднее по матчам sum_k (p_k - 1[y=k])^2. Для бинарного — обычный."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    if p.ndim == 1:
        return float(np.mean((p - y) ** 2))
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p)
    pred = (p > 0.5).astype(int) if p.ndim == 1 else p.argmax(axis=1)
    return float(np.mean(pred == y))


def prob_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"n": int(len(y)), "log_loss": log_loss(p, y), "brier": brier(p, y), "accuracy": accuracy(p, y)}


def betting_metrics(bets: pd.DataFrame, bankroll0: float = 1.0) -> dict[str, float]:
    """bets: колонки odds, won (0/1), stake_frac, flat (1), closing_odds (может быть NaN), date."""
    if bets.empty:
        return {"bets": 0, "roi_flat": np.nan, "profit_flat": 0.0, "roi_kelly": np.nan, "bankroll_growth": 0.0,
                "max_drawdown": 0.0, "hit_rate": np.nan, "avg_odds": np.nan, "clv_mean": np.nan, "clv_n": 0,
                "clv_beat_rate": np.nan}
    b = bets.sort_values("date")
    pnl_flat = np.where(b["won"] == 1, b["odds"] - 1, -1.0)
    # Келли: ставки дня считаются от банка на начало дня
    bank = bankroll0
    peak = bank
    mdd = 0.0
    staked = 0.0
    profit_k = 0.0
    for _, day in b.groupby("date", sort=True):
        start = bank
        for r in day.itertuples(index=False):
            s = r.stake_frac * start
            staked += s
            pr = s * (r.odds - 1) if r.won == 1 else -s
            profit_k += pr
            bank += pr
        peak = max(peak, bank)
        mdd = max(mdd, (peak - bank) / peak)
    clv = (b["odds"] / b["closing_odds"] - 1).dropna() if "closing_odds" in b else pd.Series(dtype=float)
    return {
        "bets": int(len(b)), "roi_flat": float(pnl_flat.mean()), "profit_flat": float(pnl_flat.sum()),
        "roi_kelly": float(profit_k / staked) if staked else np.nan, "bankroll_growth": float(bank / bankroll0 - 1),
        "max_drawdown": float(mdd), "hit_rate": float(b["won"].mean()), "avg_odds": float(b["odds"].mean()),
        "clv_mean": float(clv.mean()) if len(clv) else np.nan, "clv_n": int(len(clv)),
        "clv_beat_rate": float((clv > 0).mean()) if len(clv) else np.nan,
    }


def bootstrap_roi_ci(bets: pd.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if bets.empty:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    pnl = np.where(bets["won"].to_numpy() == 1, bets["odds"].to_numpy() - 1, -1.0)
    means = [rng.choice(pnl, len(pnl)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
