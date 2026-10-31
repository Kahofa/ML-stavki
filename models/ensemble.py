"""Ансамбль: линейный пул вероятностей с весами, подобранными по минимуму log-loss на валидации."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

EPS = 1e-12


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Мультиклассовый (или бинарный при p.ndim==1) log-loss."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    if p.ndim == 1:
        p = np.column_stack([1 - p, p])
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))))


@dataclass
class LinearPool:
    names: list[str] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, probs: dict[str, np.ndarray], y: np.ndarray) -> "LinearPool":
        """probs: {model: (n, k)} без NaN. Веса >= 0, сумма = 1 (softmax-параметризация)."""
        self.names = list(probs)
        stack = np.stack([probs[n] for n in self.names])  # (m, n, k)

        def f(z: np.ndarray) -> float:
            w = softmax(z)
            return log_loss(np.tensordot(w, stack, axes=1), y)

        best = None
        for z0 in (np.zeros(len(self.names)), *np.eye(len(self.names)) * 2):
            r = minimize(f, z0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-10})
            if best is None or r.fun < best.fun:
                best = r
        self.weights = softmax(best.x)
        return self

    def predict(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        """Если у матча нет какой-то модели (NaN), её вес перераспределяется на остальные."""
        stack = np.stack([probs[n] for n in self.names])  # (m, n, k)
        avail = np.isfinite(stack).all(axis=2)              # (m, n)
        w = self.weights[:, None] * avail
        wsum = w.sum(axis=0)
        w = np.where(wsum > 0, w / np.where(wsum > 0, wsum, 1), 0)
        out = np.einsum("mn,mnk->nk", w, np.nan_to_num(stack))
        out[wsum == 0] = np.nan
        return out

    def as_dict(self) -> dict[str, float]:
        return {n: round(float(w), 4) for n, w in zip(self.names, self.weights)}
