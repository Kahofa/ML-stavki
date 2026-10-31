"""Сопоставление названий команд между источниками.

Канонические имена — как в историческом датасете football-data.co.uk ("Man United").
Сначала словарь алиасов, затем нечёткое сравнение (difflib) среди команд лиги.
Несопоставленные имена логируются и НЕ подменяются догадкой ниже порога сходства.
"""
from __future__ import annotations

import difflib
import logging
import re
import unicodedata

log = logging.getLogger(__name__)

ALIASES: dict[str, str] = {
    # England
    "manchester united": "Man United", "manchester utd": "Man United", "man utd": "Man United",
    "manchester city": "Man City", "newcastle united": "Newcastle", "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham", "wolverhampton wanderers": "Wolves", "wolverhampton": "Wolves",
    "brighton and hove albion": "Brighton", "brighton & hove albion": "Brighton",
    "west ham united": "West Ham", "nottingham forest": "Nott'm Forest", "nottm forest": "Nott'm Forest",
    "leicester city": "Leicester", "leeds united": "Leeds", "afc bournemouth": "Bournemouth",
    "sheffield united": "Sheffield United", "sheffield utd": "Sheffield United", "luton town": "Luton",
    "ipswich town": "Ipswich", "norwich city": "Norwich", "burnley fc": "Burnley",
    # Spain
    "atletico madrid": "Ath Madrid", "atletico de madrid": "Ath Madrid", "club atletico de madrid": "Ath Madrid",
    "athletic club": "Ath Bilbao", "athletic bilbao": "Ath Bilbao", "real betis": "Betis",
    "real betis balompie": "Betis", "celta vigo": "Celta", "rc celta": "Celta", "espanyol": "Espanol",
    "rcd espanyol": "Espanol", "real sociedad": "Sociedad", "rayo vallecano": "Vallecano",
    "deportivo alaves": "Alaves", "fc barcelona": "Barcelona", "real valladolid": "Valladolid",
    "ud las palmas": "Las Palmas", "rcd mallorca": "Mallorca", "ca osasuna": "Osasuna",
    "girona fc": "Girona", "sevilla fc": "Sevilla", "villarreal cf": "Villarreal", "getafe cf": "Getafe",
    "real oviedo": "Oviedo", "levante ud": "Levante", "elche cf": "Elche",
    # Italy
    "ac milan": "Milan", "inter milan": "Inter", "internazionale": "Inter", "fc internazionale milano": "Inter",
    "as roma": "Roma", "ss lazio": "Lazio", "ssc napoli": "Napoli", "hellas verona": "Verona",
    "atalanta bc": "Atalanta", "juventus fc": "Juventus", "acf fiorentina": "Fiorentina",
    "torino fc": "Torino", "us lecce": "Lecce", "udinese calcio": "Udinese", "genoa cfc": "Genoa",
    "parma calcio 1913": "Parma", "como 1907": "Como", "cagliari calcio": "Cagliari", "us sassuolo": "Sassuolo",
    "pisa sc": "Pisa", "us cremonese": "Cremonese",
    # Germany
    "bayern munich": "Bayern Munich", "fc bayern munchen": "Bayern Munich", "bayern munchen": "Bayern Munich",
    "borussia dortmund": "Dortmund", "bayer leverkusen": "Leverkusen", "bayer 04 leverkusen": "Leverkusen",
    "borussia monchengladbach": "M'gladbach", "borussia mgladbach": "M'gladbach", "monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt", "rb leipzig": "RB Leipzig", "vfb stuttgart": "Stuttgart",
    "vfl wolfsburg": "Wolfsburg", "sc freiburg": "Freiburg", "1899 hoffenheim": "Hoffenheim",
    "tsg hoffenheim": "Hoffenheim", "tsg 1899 hoffenheim": "Hoffenheim", "1. fc union berlin": "Union Berlin",
    "union berlin": "Union Berlin", "fc augsburg": "Augsburg", "werder bremen": "Werder Bremen",
    "sv werder bremen": "Werder Bremen", "1. fsv mainz 05": "Mainz", "mainz 05": "Mainz", "fsv mainz 05": "Mainz",
    "1. fc heidenheim": "Heidenheim", "fc heidenheim": "Heidenheim", "fc st. pauli": "St Pauli",
    "st. pauli": "St Pauli", "1. fc koln": "FC Koln", "fc koln": "FC Koln", "hamburger sv": "Hamburg",
    "vfl bochum": "Bochum", "holstein kiel": "Holstein Kiel",
    # France
    "paris saint germain": "Paris SG", "paris saint-germain": "Paris SG", "psg": "Paris SG",
    "olympique marseille": "Marseille", "olympique de marseille": "Marseille", "olympique lyonnais": "Lyon",
    "as monaco": "Monaco", "losc lille": "Lille", "stade rennais": "Rennes", "stade de reims": "Reims",
    "ogc nice": "Nice", "rc lens": "Lens", "rc strasbourg": "Strasbourg", "stade brestois": "Brest",
    "stade brestois 29": "Brest", "fc nantes": "Nantes", "montpellier hsc": "Montpellier",
    "toulouse fc": "Toulouse", "aj auxerre": "Auxerre", "angers sco": "Angers", "le havre ac": "Le Havre",
    "as saint-etienne": "St Etienne", "saint-etienne": "St Etienne", "fc lorient": "Lorient",
    "paris fc": "Paris FC", "fc metz": "Metz",
    # Russia
    "zenit st petersburg": "Zenit", "zenit saint petersburg": "Zenit", "spartak moscow": "Spartak Moscow",
    "cska moscow": "CSKA Moscow", "lokomotiv moscow": "Lokomotiv Moscow", "dynamo moscow": "Dynamo Moscow",
    "fk krasnodar": "Krasnodar", "rubin kazan": "Rubin Kazan", "akhmat grozny": "Akhmat Grozny",
}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", _norm(s))


def _strip_affixes(s: str) -> str:
    return re.sub(r"\b(fc|cf|afc|sc|ac|as|ss|ssc|us|cd|ud|rc|rcd|sv|vfl|vfb|tsg|fk|1\.)\b", "", s).strip()


class TeamMatcher:
    def __init__(self, known_by_league: dict[str, list[str]], cutoff: float = 0.72) -> None:
        self.known = {lg: sorted(set(names)) for lg, names in known_by_league.items()}
        self.cutoff = cutoff
        self._memo: dict[tuple[str, str], str | None] = {}
        self.unmatched: set[tuple[str, str]] = set()

    def match(self, league: str, name: str) -> str | None:
        key = (league, name)
        if key in self._memo:
            return self._memo[key]
        res = self._match(league, name)
        self._memo[key] = res
        if res is None:
            self.unmatched.add(key)
            log.warning("Команда не сопоставлена: [%s] %r — матч будет помечен LOW DATA", league, name)
        return res

    def _match(self, league: str, name: str) -> str | None:
        known = self.known.get(league, [])
        if name in known:
            return name
        n = _norm(name)
        if n in ALIASES:
            target = _letters(ALIASES[n])
            hit = [k for k in known if _letters(k) == target]
            if hit:
                return hit[0]
            if not known:
                return ALIASES[n]
        norm_known = {_norm(k): k for k in known}
        if n in norm_known:
            return norm_known[n]
        stripped = {_strip_affixes(k): v for k, v in norm_known.items()}
        sn = _strip_affixes(n)
        if sn in stripped:
            return stripped[sn]
        best = difflib.get_close_matches(sn, list(stripped), n=1, cutoff=self.cutoff)
        if best:
            return stripped[best[0]]
        # подстрока ("Lens" in "racing club de lens")
        cands = [v for k, v in stripped.items() if len(k) >= 4 and (k in sn or sn in k)]
        if len(cands) == 1:
            return cands[0]
        return None
