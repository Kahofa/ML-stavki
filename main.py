"""CLI: прогнозы на все матчи даты.

Примеры:
    python main.py                                   # сегодня (Asia/Almaty), приоритетные лиги
    python main.py --date 2026-10-31 --leagues EPL,LaLiga --refresh
    python main.py --all-leagues
    python main.py --date 2026-08-29 --replay        # ретро-прогноз по историческим данным
    python main.py --backtest-only
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

from config import LEAGUES, PRIORITY_LEAGUES, SETTINGS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Прогнозы на футбольные матчи дня (Dixon-Coles + Elo + LightGBM + рынок)")
    p.add_argument("--date", type=date.fromisoformat, default=None,
                   help=f"дата YYYY-MM-DD (по умолчанию сегодня в {SETTINGS.timezone})")
    p.add_argument("--leagues", type=str, default=None,
                   help=f"коды лиг через запятую. Доступно: {','.join(LEAGUES)}. По умолчанию: {','.join(PRIORITY_LEAGUES)}")
    p.add_argument("--all-leagues", action="store_true", help="включить все лиги из конфигурации")
    p.add_argument("--refresh", action="store_true", help="игнорировать кэш и перезагрузить данные из API")
    p.add_argument("--replay", action="store_true",
                   help="ретро-прогноз на прошедшую дату по историческому датасету (результаты с этой даты скрыты)")
    p.add_argument("--backtest-only", action="store_true", help="только бэктест и отчёт по нему")
    p.add_argument("--no-understat", action="store_true", help="не загружать xG с Understat")
    p.add_argument("--bankroll", type=float, default=SETTINGS.bankroll, help="банк для расчёта размера ставок")
    p.add_argument("--out", type=str, default=str(SETTINGS.report_dir), help="папка для отчётов")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("main")

    day = args.date or datetime.now(ZoneInfo(SETTINGS.timezone)).date()
    if args.all_leagues:
        leagues = list(LEAGUES)
    elif args.leagues:
        leagues = [c.strip() for c in args.leagues.split(",") if c.strip()]
    else:
        leagues = list(PRIORITY_LEAGUES)
    unknown = [c for c in leagues if c not in LEAGUES]
    if unknown:
        log.error("Неизвестные лиги: %s. Доступно: %s", unknown, ", ".join(LEAGUES))
        return 2

    from pathlib import Path

    from pipeline import run
    from report.render import console_summary, write_outputs

    mode = "replay" if args.replay else "live"
    if args.backtest_only:
        mode = "replay"
        day = date(1900, 1, 1)  # нет матчей: только бэктест
    log.info("Запуск: дата=%s, лиги=%s, режим=%s%s", day, ",".join(leagues), mode, ", refresh" if args.refresh else "")
    try:
        result = run(day, leagues, mode=mode, refresh=args.refresh, use_understat=not args.no_understat,
                     bankroll=args.bankroll)
    except Exception as exc:  # верхний уровень: понятное сообщение вместо трейсбека
        log.exception("Сбой выполнения: %s", exc)
        return 1
    if args.backtest_only:
        result.warnings = [w for w in result.warnings if "нет матчей" not in w]
        result.day = datetime.now(ZoneInfo(SETTINGS.timezone)).date()
        result.mode = "backtest"
    paths = write_outputs(result, Path(args.out))
    print(console_summary(result))
    print("\nФайлы:")
    for k, p in paths.items():
        print(f"  {k:20s} {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
