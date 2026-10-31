"""Рыночные вероятности: удаление маржи букмекера методами Шина, power и мультипликативным."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def implied(odds: np.ndarray) -> np.ndarray:
    return 1.0 / np.asarray(odds, dtype=float)


def multiplicative(odds: np.ndarray) -> np.ndarray:
    pi = implied(odds)
    return pi / pi.sum()


def power(odds: np.ndarray) -> np.ndarray:
    """p_i = pi_i^k, где k подобран так, что sum p_i = 1 (k > 1 при положительной марже)."""
    pi = implied(odds)
    if abs(pi.sum() - 1.0) < 1e-12:
        return pi
    f = lambda k: float(np.sum(pi ** k) - 1.0)  # noqa: E731
    try:
        k = brentq(f, 0.2, 5.0, xtol=1e-12)
    except ValueError:
        return multiplicative(odds)
    p = pi ** k
    return p / p.sum()


def shin(odds: np.ndarray) -> np.ndarray:
    """Метод Шина (Shin 1993; формула Jullien & Salanié): учитывает долю инсайдерских денег z."""
    pi = implied(odds)
    s = pi.sum()
    if s <= 1.0:
        return pi / s

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / s) - z) / (2 * (1 - z))

    f = lambda z: float(probs(z).sum() - 1.0)  # noqa: E731
    try:
        z = brentq(f, 0.0, 0.4, xtol=1e-12)
    except ValueError:
        return power(odds)
    p = probs(z)
    return p / p.sum()


METHODS = {"shin": shin, "power": power, "multiplicative": multiplicative}


def devig(odds: np.ndarray, method: str = "shin") -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        return np.full(odds.shape, np.nan)
    return METHODS[method](odds)


def devig_frame(odds_matrix: np.ndarray, method: str = "shin") -> np.ndarray:
    """Построчное удаление маржи для массива (n, k). Строки с пропусками -> NaN."""
    return np.vstack([devig(row, method) for row in np.asarray(odds_matrix, dtype=float)]) if len(odds_matrix) else \
        np.empty((0, np.asarray(odds_matrix).shape[1] if np.asarray(odds_matrix).ndim == 2 else 0))


def overround(odds: np.ndarray) -> float:
    return float(implied(odds).sum() - 1.0)
