"""Guardian Content API headline collection (amendment A2 replaced GDELT)."""

import os
import time
from datetime import datetime

import requests

from hms.db import connect
from hms.llm import load_key
from hms.markets import WINDOW_END, WINDOW_START

API = "https://content.guardianapis.com/search"
PAGE_SIZE, PAUSE_S = 200, 1.1  # developer key: 1 call/s, 500 calls/day
# One page of the 200 most relevant results per split period, so every period gets coverage.
PERIODS = [("2026-08-01", "2026-09-21"), ("2026-09-22", "2026-10-08"), ("2026-10-09", "2026-10-29")]


def parse_ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def fetch_page(query: str, start: str, end: str) -> dict:
    for attempt in range(4):
        time.sleep(PAUSE_S * (1 + 2 * attempt))
        r = requests.get(API, timeout=60, params={
            "q": query, "from-date": start, "to-date": end, "page-size": PAGE_SIZE,
            "order-by": "relevance", "use-date": "published",
            "api-key": os.environ["GUARDIAN_API_KEY"]})
        if r.status_code == 200:
            return r.json()["response"]
    r.raise_for_status()
    return {}


def collect() -> None:
    load_key()
    conn = connect()
    conn.execute("CREATE TABLE IF NOT EXISTS headlines_raw_gdelt AS SELECT * FROM headlines_raw "
                 "WHERE 0")
    if conn.execute("SELECT COUNT(*) FROM headlines_raw_gdelt").fetchone()[0] == 0:
        conn.execute("INSERT INTO headlines_raw_gdelt SELECT * FROM headlines_raw "
                     "WHERE url NOT LIKE 'https://www.theguardian.com/%'")
        conn.execute("DELETE FROM headlines_raw WHERE url NOT LIKE 'https://www.theguardian.com/%'")
    conn.execute("CREATE TABLE IF NOT EXISTS guardian_log (market_id TEXT PRIMARY KEY, "
                 "pages INTEGER, total INTEGER, n_stored INTEGER)")
    done = {r[0] for r in conn.execute("SELECT market_id FROM guardian_log")}
    todo = conn.execute(
        "SELECT q.market_id, q.query FROM queries q JOIN markets m USING (market_id) "
        "WHERE m.selected = 1 ORDER BY m.candidate_rank").fetchall()
    lo, hi = WINDOW_START.timestamp(), WINDOW_END.timestamp()
    for i, (market_id, query) in enumerate(todo):
        if market_id in done:
            continue
        n, pages, total = 0, 0, 0
        for start, end in PERIODS:
            resp = fetch_page(query, start, end)
            pages, total = pages + 1, total + resp.get("total", 0)
            rows = [(market_id, x["webUrl"], x["webTitle"], parse_ts(x["webPublicationDate"]),
                     "theguardian.com") for x in resp.get("results", [])]
            rows = [r for r in rows if lo <= r[3] <= hi]  # keep only the requested window
            conn.executemany("INSERT OR IGNORE INTO headlines_raw VALUES (?,?,?,?,?)", rows)
            n += len(rows)
        conn.execute("INSERT OR REPLACE INTO guardian_log VALUES (?,?,?,?)",
                     (market_id, pages, total, n))
        conn.commit()
        print(f"{i + 1}/{len(todo)} {market_id} total={total} stored={n}", flush=True)


if __name__ == "__main__":
    collect()
