"""Elo -> вероятности 1X2 через порядковую логистическую регрессию (ordered logit) по разнице рейтингов."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class EloOrdinalModel:
    beta: float = 0.005
    c1: float = -0.6   # порог A | D
    c2: float = 0.6    # порог D | H
    fitted: bool = False

    def predict_proba(self, elo_diff: np.ndarray) -> np.ndarray:
        """Возвращает (n, 3) вероятностей [H, D, A]."""
        z = self.beta * np.asarray(elo_diff, dtype=float)
        p_a = expit(self.c1 - z)
        p_ad = expit(self.c2 - z)
        p_d = np.clip(p_ad - p_a, 1e-6, None)
        p_h = np.clip(1 - p_ad, 1e-6, None)
        p = np.column_stack([p_h, p_d, p_a])
        return p / p.sum(axis=1, keepdims=True)

    def fit(self, elo_diff: np.ndarray, y: np.ndarray) -> "EloOrdinalModel":
        """y: 0=H, 1=D, 2=A."""
        elo_diff = np.asarray(elo_diff, dtype=float)
        y = np.asarray(y, dtype=int)

        def nll(p: np.ndarray) -> float:
            self.beta, self.c1 = p[0], p[1]
            self.c2 = p[1] + np.exp(p[2])
            pr = self.predict_proba(elo_diff)
            return -float(np.log(pr[np.arange(len(y)), y]).mean())

        res = minimize(nll, np.array([0.004, -0.6, np.log(1.2)]), method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-9})
        self.beta, self.c1, self.c2 = res.x[0], res.x[1], res.x[1] + np.exp(res.x[2])
        self.fitted = True
        return self
