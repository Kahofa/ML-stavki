"""ClubElo (api.clubelo.com): рейтинги всех клубов на дату — для сверки с собственным Elo."""
from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd

from data.cache import Cache
from data.http import CachedHttp

log = logging.getLogger(__name__)

# ClubElo -> имена датасета (датасет уже переименовал часть клубов)
CLUBELO_RENAME = {"Bayern": "Bayern Munich", "Man United": "Man United", "Forest": "Nott'm Forest",
                  "Atletico": "Ath Madrid", "Bilbao": "Ath Bilbao", "Gladbach": "M'gladbach",
                  "Frankfurt": "Ein Frankfurt", "Koeln": "FC Koln", "Paris SG": "Paris SG",
                  "Saint-Etienne": "St Etienne", "Sociedad": "Sociedad", "Celta": "Celta",
                  "Espanyol": "Espanol", "Rayo Vallecano": "Vallecano", "Leipzig": "RB Leipzig"}


class ClubElo:
    source = "clubelo.com"

    def __init__(self, cache: Cache) -> None:
        self.http = CachedHttp(cache, self.source, min_interval_s=2.0)

    def ratings(self, day: date, refresh: bool = False) -> tuple[pd.DataFrame, str]:
        body, fetched_at, _ = self.http.get_raw(f"http://api.clubelo.com/{day.isoformat()}",
                                                ttl_s=12 * 3600, refresh=refresh)
        df = pd.read_csv(io.BytesIO(body))
        df["club"] = df["Club"].replace(CLUBELO_RENAME)
        return df[["club", "Country", "Level", "Elo"]].rename(columns={"Elo": "elo"}), fetched_at
