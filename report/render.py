"""Экспорт результатов: CSV (матчи, ставки, бэктест) и автономный HTML-отчёт."""
from __future__ import annotations

import base64
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import BETTING
from models.calibration import plot_reliability

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent / "templates"


def _fmt_pct(x: Any, digits: int = 0) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(x: Any, digits: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "—"
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def reliability_png(preds: pd.DataFrame, path: Path) -> Path | None:
    d = preds[preds["y"] >= 0].dropna(subset=["ens_ph", "mk_ph", "mod_ph"])
    if d.empty:
        return None
    y = d["y"].to_numpy()
    onehot = np.eye(3)[y].ravel()
    curves = {
        "Ансамбль + рынок (калибр.)": (d[["ens_ph", "ens_pd", "ens_pa"]].to_numpy().ravel(), onehot),
        "Ансамбль без рынка (калибр.)": (d[["mod_ph", "mod_pd", "mod_pa"]].to_numpy().ravel(), onehot),
        "Рынок (Bet365, Шин)": (d[["mk_ph", "mk_pd", "mk_pa"]].to_numpy().ravel(), onehot),
    }
    return plot_reliability(curves, path, "Reliability diagram: 1X2, все исходы (walk-forward)")


def verdict_text(summary: dict[str, Any]) -> list[str]:
    v = summary["verdict"]
    o = summary["overall_1x2"]
    mk = o["Рынок (Bet365, Шин) — бейзлайн"]["log_loss"]
    mo = o["Ансамбль без рынка (калибр.)"]["log_loss"]
    en = o["Ансамбль + рынок (калибр.)"]["log_loss"]
    lines = []
    if v["model_only_beats_market"]:
        lines.append(f"Модели без рынка лучше рынка по log-loss ({mo:.4f} против {mk:.4f}).")
    else:
        lines.append(f"Честно: модели без рынка НЕ лучше рынка — log-loss {mo:.4f} против {mk:.4f} у рыночных "
                     f"вероятностей (хуже на {mo - mk:+.4f}).")
    if v["ensemble_beats_market"]:
        lines.append(f"Ансамбль с рынком немного лучше чистого рынка: {en:.4f} против {mk:.4f}.")
    else:
        lines.append(f"Ансамбль с рынком не лучше рынка ({en:.4f} против {mk:.4f}): оптимизатор отдаёт рынку "
                     "почти весь вес — значит, независимые модели добавляют мало новой информации.")
    b = summary["betting_best_price"]
    if b["bets"]:
        lo, hi = b["roi_ci95"]
        sig = ("статистически значимо положителен" if lo > 0 else "статистически значимо отрицателен" if hi < 0
               else "статистически не отличается от нуля (95% ДИ включает 0)")
        lines.append(f"Ставки по правилам (value > {BETTING.min_value:.0%}, уверенность ≥ MEDIUM, лучшая цена): "
                     f"{b['bets']} ставок, ROI {b['roi_flat']:+.2%} (95% ДИ {lo:+.2%} … {hi:+.2%}) — {sig}.")
        b2 = summary["betting_bet365"]
        if b2["bets"]:
            lines.append(f"По ценам одного букмекера (Bet365) ROI {b2['roi_flat']:+.2%} на {b2['bets']} ставках — "
                         "без поиска лучшей цены преимущества нет.")
    lines.append(summary["clv_note"])
    return lines


def write_outputs(result, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{result.day.isoformat()}{'_replay' if result.mode == 'replay' else ''}"
    paths: dict[str, Path] = {}
    bt = result.backtest
    summary = bt.summary if bt else {}

    # ---------------- CSV
    if not result.matches.empty:
        p = out_dir / f"predictions_{tag}.csv"
        result.matches.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.4f")
        paths["predictions_csv"] = p
        mk_rows = [{"league": d["league"], "home": d["home"], "away": d["away"], "confidence": d["confidence"], **m}
                   for d in result.details for m in d["markets"]]
        p = out_dir / f"markets_{tag}.csv"
        pd.DataFrame(mk_rows).to_csv(p, index=False, encoding="utf-8-sig", float_format="%.4f")
        paths["markets_csv"] = p
    if bt is not None:
        rows = [{"season": s, **{k: v for k, v in d.items() if k != "weights"}, "weights": json.dumps(d["weights"])}
                for s, d in summary["by_season"].items()]
        p = out_dir / "backtest_by_season.csv"
        pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig", float_format="%.5f")
        paths["backtest_csv"] = p
        if not bt.bets.empty:
            p = out_dir / "backtest_bets.csv"
            bt.bets.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.4f")
            paths["backtest_bets_csv"] = p
        p = out_dir / "backtest_summary.json"
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
        paths["backtest_json"] = p

    # ---------------- HTML
    img_b64 = ""
    if bt is not None:
        png = reliability_png(bt.preds, out_dir / "reliability_diagram.png")
        if png:
            paths["reliability_png"] = png
            img_b64 = base64.b64encode(png.read_bytes()).decode()
    best = []
    skip = []
    if result.details:
        cands = [d for d in result.details if d["best_bet"]]
        best = sorted(cands, key=lambda d: -d["best_bet_value"])[:BETTING.max_best_bets]
        skip = [d for d in result.details if d["confidence"] == "LOW"]
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    env.filters["pct"] = _fmt_pct
    env.filters["num"] = _fmt_num
    html = env.get_template("report.html.j2").render(
        r=result, summary=summary, verdict=verdict_text(summary) if summary else [],
        best=best, skip=skip, details=result.details, img_b64=img_b64, betting=BETTING,
    )
    p = out_dir / f"report_{tag}.html"
    p.write_text(html, encoding="utf-8")
    paths["html"] = p
    return paths


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def console_summary(result) -> str:
    """Короткая текстовая сводка для терминала."""
    lines = []
    s = result.backtest.summary if result.backtest else None
    if s:
        lines.append(f"=== БЭКТЕСТ (walk-forward, сезоны {s['seasons'][0]}–{s['seasons'][-1]}, {s['n_matches']} матчей) ===")
        lines.append(f"{'Модель':40s} {'log-loss':>9s} {'Brier':>7s} {'Acc':>6s}")
        for k, v in s["overall_1x2"].items():
            lines.append(f"{k:40s} {v['log_loss']:9.4f} {v['brier']:7.4f} {v['accuracy']:6.1%}")
        lines.append("Тотал 2.5: " + ", ".join(f"{k} LL={v['log_loss']:.4f}" for k, v in s["ou25"].items()))
        lines.append("Обе забьют: " + ", ".join(f"{k} LL={v['log_loss']:.4f}" for k, v in s["btts"].items()))
        b = s["betting_best_price"]
        lines.append(f"Ставки: {b['bets']}, ROI(flat) {b['roi_flat']:+.2%} ДИ95 [{b['roi_ci95'][0]:+.2%}; {b['roi_ci95'][1]:+.2%}],"
                     f" ROI(Kelly) {b['roi_kelly']:+.2%}, банк {b['bankroll_growth']:+.1%}, макс. просадка {b['max_drawdown']:.1%}")
        lines.append(f"Сверка Elo: корреляция разницы собственного Elo и ClubElo = {s['elo_vs_clubelo_corr']:.3f}")
        lines.append("CLV: н/д (" + s["clv_note"] + ")")
        lines.extend("• " + t for t in verdict_text(s))
    if result.warnings:
        lines.append("=== ПРЕДУПРЕЖДЕНИЯ ===")
        lines.extend("! " + w for w in dict.fromkeys(result.warnings))
    if not result.matches.empty:
        lines.append(f"=== ПРОГНОЗЫ {result.day} ({'ретро-прогноз по истории' if result.mode == 'replay' else 'live'}) ===")
        for r in result.matches.itertuples(index=False):
            act = f"  факт {r.actual}" if r.actual else ""
            lines.append(f"[{r.confidence:6s}] {r.league:10s} {r.home} – {r.away}: П1 {r.p_home:.0%} X {r.p_draw:.0%} "
                         f"П2 {r.p_away:.0%} | ТБ2.5 {r.p_over25:.0%} | ОЗ {r.p_btts:.0%} | {r.top_scores}"
                         f"{' | СТАВКА ' + r.best_bet + f' value {r.best_bet_value:+.1%}, {r.best_bet_stake_frac:.2%} банка' if r.best_bet else ''}{act}")
    return "\n".join(lines)
