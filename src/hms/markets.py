"""Market selection and hourly YES price history (pre-registration: "Markets")."""

import json
import re
import time
from datetime import datetime, timezone

import requests

from hms.db import connect

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"

WINDOW_START = datetime(2026, 8, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 10, 29, 23, 59, 59, tzinfo=timezone.utc)
N_MARKETS = 150
MIN_HISTORY_DAYS = 7.0
PRICE_START = int(datetime(2026, 7, 25, tzinfo=timezone.utc).timestamp())

EXCLUDED_TAGS = {"sports", "esports", "crypto-prices", "up-or-down", "nba", "nfl", "mlb",
                 "nhl", "soccer", "tennis", "golf", "ufc", "f1", "cricket", "games"}
EXCLUDED_QUESTION = re.compile(
    r"\bup or down\b|\b(above|below|reach|hit|dip to)\s+\$[\d,.]+|\bvs\.?\s|\bO/U\b|\bspread\b",
    re.IGNORECASE,
)


def is_excluded(question: str, tag_slugs: list[str]) -> bool:
    if set(tag_slugs) & EXCLUDED_TAGS:
        return True
    return bool(EXCLUDED_QUESTION.search(question))


def fetch_candidates(max_pages: int = 12) -> list[dict]:
    """Binary markets open at some point in the window, by lifetime volume (open and closed)."""
    out: dict[str, dict] = {}
    for closed in ("false", "true"):
        for page in range(max_pages):
            r = requests.get(GAMMA, timeout=60, params={
                "limit": 100, "offset": page * 100, "order": "volumeNum", "ascending": "false",
                "closed": closed, "include_tag": "true",
                "end_date_min": WINDOW_START.isoformat(),
                "start_date_max": WINDOW_END.isoformat(),
            })
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            for m in batch:
                out[m["id"]] = m
            time.sleep(0.3)
    return sorted(out.values(), key=lambda m: -(m.get("volumeNum") or 0))


def fetch_prices(token_id: str) -> list[tuple[int, float]]:
    r = requests.get(CLOB_HISTORY, timeout=60,
                     params={"market": token_id, "startTs": PRICE_START, "fidelity": 60})
    r.raise_for_status()
    return [(int(h["t"]), float(h["p"])) for h in r.json().get("history", [])]


def select_markets() -> int:
    conn = connect()
    if conn.execute("SELECT COUNT(*) FROM markets WHERE selected = 1").fetchone()[0] >= N_MARKETS:
        return N_MARKETS
    lo, hi = int(WINDOW_START.timestamp()), int(WINDOW_END.timestamp())
    rank, n_selected = 0, 0
    for m in fetch_candidates():
        if json.loads(m.get("outcomes") or "[]") != ["Yes", "No"]:
            continue
        tags = [t["slug"] for t in (m.get("tags") or [])]
        if is_excluded(m["question"], tags):
            continue
        rank += 1
        token_yes = json.loads(m["clobTokenIds"])[0]
        all_prices = fetch_prices(token_yes)
        prices = [(t, p) for t, p in all_prices if lo <= t <= hi]
        days = (prices[-1][0] - prices[0][0]) / 86400 if len(prices) > 1 else 0.0
        selected = int(days >= MIN_HISTORY_DAYS)
        conn.execute(
            "INSERT OR REPLACE INTO markets VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (m["id"], m["question"], m.get("slug"), token_yes, m.get("startDate"),
             m.get("endDate"), m.get("volumeNum"), ",".join(tags), rank, days, selected),
        )
        if selected:
            conn.executemany("INSERT OR IGNORE INTO prices VALUES (?,?,?)",
                             [(token_yes, t, p) for t, p in all_prices])
            n_selected += 1
        conn.commit()
        time.sleep(0.2)
        if n_selected >= N_MARKETS:
            break
    return n_selected


if __name__ == "__main__":
    print("selected", select_markets())
