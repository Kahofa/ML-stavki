"""HTTP-клиент с кэшем в SQLite, паузами между запросами, ретраями и журналом источников."""
from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.parse import urlparse

import requests

from data.cache import Cache

log = logging.getLogger(__name__)


class ApiError(RuntimeError):
    """Ошибка внешнего API (сеть, лимит, авторизация, неверный ответ)."""


class RateLimitError(ApiError):
    pass


class CachedHttp:
    """GET с кэшированием. Каждый ответ сохраняется с временем загрузки (fetched_at, UTC).

    Предохранитель: если хост недоступен (отказ прокси/соединения), он помечается как недоступный
    до конца запуска — без долгих повторных попыток. Таймауты и 5xx повторяются с экспоненциальной паузой.
    """

    _down_hosts: dict[str, str] = {}

    def __init__(self, cache: Cache, source: str, min_interval_s: float = 1.0,
                 headers: dict[str, str] | None = None, timeout_s: float = 30.0,
                 max_retries: int = 4, session: requests.Session | None = None) -> None:
        self.cache = cache
        self.source = source
        self.min_interval_s = min_interval_s
        self.headers = headers or {}
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self._last_call = 0.0
        self.last_response_headers: dict[str, str] = {}

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def get_raw(self, url: str, params: dict[str, Any] | None = None, ttl_s: float | None = 3600,
                refresh: bool = False, cache_params: dict[str, Any] | None = None) -> tuple[bytes, str, bool]:
        """Возвращает (body, fetched_at_iso, from_cache).

        ttl_s=None — кэш бессрочный (для завершённых матчей). refresh=True — игнорировать кэш.
        cache_params — параметры для ключа кэша (без секретов), по умолчанию = params.
        """
        key_params = cache_params if cache_params is not None else params
        key = self.cache.make_key(self.source, url, key_params)
        if not refresh:
            hit = self.cache.get_http(key, ttl_s)
            if hit is not None:
                body, fetched_at = hit
                self.cache.log_source(self.source, url, key_params, fetched_at, True, "ok")
                return body, fetched_at, True

        host = urlparse(url).netloc
        if host in self._down_hosts:
            err = ApiError(f"{self.source}: хост {host} недоступен в этом запуске ({self._down_hosts[host]})")
            self.cache.log_source(self.source, url, key_params, _now(), False, "error", str(err))
            raise err
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, headers=self.headers, timeout=self.timeout_s)
            except (requests.exceptions.ProxyError, requests.exceptions.SSLError,
                    requests.exceptions.ConnectionError) as exc:  # отказ прокси / DNS / соединения
                reason = type(exc).__name__
                self._down_hosts[host] = reason
                err = ApiError(f"{self.source}: нет соединения с {host} ({reason}) — проверьте сеть/allowlist")
                log.warning("%s", err)
                self.cache.log_source(self.source, url, key_params, _now(), False, "error", str(err))
                raise err from exc
            except requests.RequestException as exc:  # таймаут и прочее — повторяем
                last_err = exc
                log.warning("%s: сетевая ошибка %s (попытка %d)", self.source, exc, attempt + 1)
                time.sleep(min(2 ** attempt, 16))
                continue
            self.last_response_headers = dict(resp.headers)
            if resp.status_code == 429:
                last_err = RateLimitError(f"{self.source}: превышен лимит (429)")
                retry_after = float(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                log.warning("%s: 429, ждём %.0f c", self.source, retry_after)
                time.sleep(min(retry_after, 60))
                continue
            if resp.status_code >= 500:
                last_err = ApiError(f"{self.source}: HTTP {resp.status_code}")
                time.sleep(min(2 ** attempt, 16))
                continue
            if resp.status_code in (401, 403):
                err = ApiError(f"{self.source}: доступ запрещён (HTTP {resp.status_code}) — проверьте ключ/тариф/сеть")
                self.cache.log_source(self.source, url, key_params, _now(), False, "error", str(err))
                raise err
            if resp.status_code >= 400:
                err = ApiError(f"{self.source}: HTTP {resp.status_code}: {resp.text[:200]}")
                self.cache.log_source(self.source, url, key_params, _now(), False, "error", str(err))
                raise err
            fetched_at = self.cache.put_http(key, self.source, url, key_params, resp.status_code, resp.content)
            self.cache.log_source(self.source, url, key_params, fetched_at, False, "ok")
            return resp.content, fetched_at, False

        self.cache.log_source(self.source, url, key_params, _now(), False, "error", str(last_err))
        raise ApiError(f"{self.source}: не удалось получить {url}: {last_err}")

    def get_json(self, url: str, params: dict[str, Any] | None = None, **kw: Any) -> tuple[Any, str]:
        body, fetched_at, _ = self.get_raw(url, params, **kw)
        try:
            return json.loads(body), fetched_at
        except json.JSONDecodeError as exc:
            raise ApiError(f"{self.source}: невалидный JSON от {url}") from exc


def _now() -> str:
    from data.cache import utcnow_iso
    return utcnow_iso()
