"""Калибровка вероятностей (isotonic / Platt, one-vs-rest с перенормировкой) и reliability diagram."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from models.ensemble import log_loss

log = logging.getLogger(__name__)
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    method: str = "isotonic"   # "isotonic" | "platt" | "none"
    models: list = field(default_factory=list)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        p = np.atleast_2d(p) if p.ndim > 1 else np.column_stack([1 - p, p])
        k = p.shape[1]
        self.models = []
        if self.method == "none":
            return self
        for c in range(k):
            yc = (y == c).astype(int)
            if self.method == "isotonic":
                m = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995).fit(p[:, c], yc)
            else:
                m = LogisticRegression(C=1e4).fit(_logit(p[:, c]).reshape(-1, 1), yc)
            self.models.append(m)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        binary = p.ndim == 1
        P = np.column_stack([1 - p, p]) if binary else p
        if self.method == "none" or not self.models:
            return p
        out = np.full_like(P, np.nan, dtype=float)
        ok = np.isfinite(P).all(axis=1)
        for c, m in enumerate(self.models):
            if self.method == "isotonic":
                out[ok, c] = m.predict(P[ok, c])
            else:
                out[ok, c] = m.predict_proba(_logit(P[ok, c]).reshape(-1, 1))[:, 1]
        out[ok] = np.clip(out[ok], EPS, None)
        out[ok] /= out[ok].sum(axis=1, keepdims=True)
        return out[:, 1] if binary else out


def choose_calibrator(p: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0,
                      min_gain: float = 5e-4) -> tuple[Calibrator, dict[str, float]]:
    """Выбор метода калибровки по CV log-loss на валидационном наборе.

    Калибровка принимается, только если улучшает CV log-loss относительно «без калибровки» больше чем
    на min_gain — иначе шумовые улучшения на валидации превращаются в систематические сдвиги на тесте.
    """
    scores: dict[str, float] = {}
    kf = KFold(folds, shuffle=True, random_state=seed)
    for method in ("none", "platt", "isotonic"):
        losses = []
        for tr, te in kf.split(p):
            cal = Calibrator(method).fit(p[tr], y[tr])
            losses.append(log_loss(cal.transform(p[te]), y[te]))
        scores[method] = float(np.mean(losses))
    best = min(scores, key=scores.get)
    if scores["none"] - scores[best] < min_gain:
        best = "none"
    log.info("Калибровка: CV log-loss %s -> выбран %s", {k: round(v, 5) for k, v in scores.items()}, best)
    return Calibrator(best).fit(p, y), scores


def reliability_table(p: np.ndarray, y_bin: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()), "freq": float(y_bin[m].mean())})
    return rows


def plot_reliability(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> Path:
    """curves: {label: (p, y_bin)} — сводная reliability diagram (все исходы 1X2 one-vs-rest)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1, label="идеальная калибровка")
    for label, (p, yb) in curves.items():
        rows = reliability_table(p, yb, bins=12)
        ax.plot([r["mean_pred"] for r in rows], [r["freq"] for r in rows], marker="o", ms=4, lw=1.6, label=label)
    ax.set_xlabel("Предсказанная вероятность")
    ax.set_ylabel("Наблюдаемая частота")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
