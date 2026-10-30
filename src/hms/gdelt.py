"""GDELT DOC 2.0 headline collection, one stored query per market (prereg: "Headlines")."""

import time
from datetime import datetime, timedelta, timezone

import requests

from hms.db import connect
from hms.markets import WINDOW_END, WINDOW_START

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250
PAUSE_S = 5.5  # GDELT asks for at most one request every 5 s


def _fmt(d: datetime) -> str:
    return d.strftime("%Y%m%d%H%M%S")


def parse_seendate(s: str) -> int:
    return int(datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp())


def fetch(query: str, start: datetime, end: datetime) -> list[dict] | None:
    params = {"query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
              "maxrecords": MAX_RECORDS, "sort": "datedesc",
              "startdatetime": _fmt(start), "enddatetime": _fmt(end)}
    for attempt in range(4):
        time.sleep(PAUSE_S * (1 + attempt))
        try:
            r = requests.get(DOC_API, params=params, timeout=60)
        except requests.RequestException:
            continue
        if r.status_code == 200 and r.text.lstrip().startswith("{"):
            return r.json().get("articles", [])
        if r.status_code == 200 and not r.text.strip():
            return []
    return None


def chunks() -> list[tuple[datetime, datetime]]:
    edges = [WINDOW_START, datetime(2026, 9, 1, tzinfo=timezone.utc),
             datetime(2026, 10, 1, tzinfo=timezone.utc), WINDOW_END]
    return list(zip(edges[:-1], edges[1:]))


def split10(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    out, s = [], start
    while s < end:
        e = min(s + timedelta(days=10), end)
        out.append((s, e))
        s = e
    return out


def collect() -> None:
    conn = connect()
    todo = conn.execute(
        "SELECT q.market_id, q.query FROM queries q JOIN markets m USING (market_id) "
        "WHERE m.selected = 1 ORDER BY m.candidate_rank").fetchall()
    done = set(conn.execute("SELECT market_id, start FROM fetch_log WHERE status = 'ok'"))
    for i, (market_id, query) in enumerate(todo):
        for start, end in chunks():
            if (market_id, start.isoformat()) in done:
                continue
            arts = fetch(query, start, end)
            pieces = [(start, end, arts)]
            if arts is not None and len(arts) >= MAX_RECORDS:  # saturated: split into 10-day pieces
                pieces = [(s, e, fetch(query, s, e)) for s, e in split10(start, end)]
            for s, e, a in pieces:
                if a is None:
                    continue
                conn.executemany(
                    "INSERT OR IGNORE INTO headlines_raw VALUES (?,?,?,?,?)",
                    [(market_id, x["url"], x.get("title") or "", parse_seendate(x["seendate"]),
                      x.get("domain")) for x in a if x.get("seendate")])
            status = "ok" if all(a is not None for _, _, a in pieces) else "failed"
            conn.execute("INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?)",
                         (market_id, start.isoformat(), end.isoformat(),
                          sum(len(a or []) for _, _, a in pieces), status))
            conn.commit()
        print(f"{i + 1}/{len(todo)} {market_id}", flush=True)


if __name__ == "__main__":
    collect()
