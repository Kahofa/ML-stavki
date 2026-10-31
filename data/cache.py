"""SQLite-кэш: HTTP-ответы, журнал источников (время загрузки), снимки коэффициентов."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    params TEXT,
    fetched_at TEXT NOT NULL,
    status INTEGER,
    body BLOB
);
CREATE TABLE IF NOT EXISTS source_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    params TEXT,
    fetched_at TEXT NOT NULL,
    from_cache INTEGER NOT NULL,
    status TEXT NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS odds_snapshots (
    event_id TEXT NOT NULL,
    sport_key TEXT NOT NULL,
    commence_time TEXT NOT NULL,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,
    outcome TEXT NOT NULL,
    point REAL,
    price REAL NOT NULL,
    last_update TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (event_id, bookmaker, market, outcome, point, fetched_at)
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Cache:
    """Тонкая обёртка над SQLite. Потокобезопасна в пределах процесса."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._run_log: list[dict[str, Any]] = []
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ---------- HTTP cache ----------
    @staticmethod
    def make_key(source: str, url: str, params: dict[str, Any] | None) -> str:
        raw = json.dumps([source, url, sorted((params or {}).items())], default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_http(self, key: str, max_age_s: float | None) -> tuple[bytes, str] | None:
        with self.tx() as c:
            row = c.execute("SELECT body, fetched_at FROM http_cache WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        body, fetched_at = row
        if max_age_s is not None:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)).total_seconds()
            if age > max_age_s:
                return None
        return body, fetched_at

    def put_http(self, key: str, source: str, url: str, params: dict[str, Any] | None,
                 status: int, body: bytes) -> str:
        ts = utcnow_iso()
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO http_cache VALUES (?,?,?,?,?,?,?)",
                (key, source, url, json.dumps(params or {}, default=str), ts, status, body),
            )
        return ts

    # ---------- source log ----------
    def log_source(self, source: str, endpoint: str, params: dict[str, Any] | None, fetched_at: str,
                   from_cache: bool, status: str, detail: str = "") -> None:
        self._run_log.append({"source": source, "endpoint": endpoint, "params": params or {},
                              "fetched_at": fetched_at, "from_cache": from_cache, "status": status,
                              "detail": detail})
        with self.tx() as c:
            c.execute(
                "INSERT INTO source_log (source, endpoint, params, fetched_at, from_cache, status, detail)"
                " VALUES (?,?,?,?,?,?,?)",
                (source, endpoint, json.dumps(params or {}, default=str), fetched_at, int(from_cache), status, detail),
            )

    def run_log(self) -> list[dict[str, Any]]:
        """Записи журнала источников за текущий запуск (с исходным временем загрузки данных)."""
        return list(self._run_log)

    # ---------- odds snapshots ----------
    def put_odds(self, rows: list[tuple[Any, ...]]) -> None:
        if not rows:
            return
        with self.tx() as c:
            c.executemany("INSERT OR IGNORE INTO odds_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def odds_rows(self, event_id: str) -> list[dict[str, Any]]:
        cols = ["event_id", "sport_key", "commence_time", "home", "away", "bookmaker", "market",
                "outcome", "point", "price", "last_update", "fetched_at"]
        with self.tx() as c:
            rows = c.execute("SELECT * FROM odds_snapshots WHERE event_id=? ORDER BY fetched_at", (event_id,)).fetchall()
        return [dict(zip(cols, r)) for r in rows]

    # ---------- kv ----------
    def kv_get(self, key: str) -> str | None:
        with self.tx() as c:
            row = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def kv_set(self, key: str, value: str) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?)", (key, value, utcnow_iso()))
