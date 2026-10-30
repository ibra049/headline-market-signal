"""Resumable GDELT collection with conservative timestamps and a three-hour probe."""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from hms.db import connect
from hms.markets import WINDOW_END, WINDOW_START

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250
PAUSE_S = 30.0
MAX_ATTEMPTS = 6
PROBE_SECONDS = 3 * 3600
PROGRESS = Path("data/collection_progress.json")
LOG = logging.getLogger(__name__)
_last_request: float | None = None


class CollectionBlocked(RuntimeError):
    """The owner's three-hour availability check failed."""


def _fmt(d: datetime) -> str:
    return d.strftime("%Y%m%d%H%M%S")


def parse_seendate(s: str) -> int:
    return int(datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp())


def _wait(seconds, check):
    end = time.monotonic() + seconds
    while (remaining := end - time.monotonic()) > 0:
        check()
        time.sleep(min(remaining, 30))
    check()


def fetch(query: str, start: datetime, end: datetime, check=lambda: None) -> list[dict] | None:
    global _last_request
    params = {"query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
              "maxrecords": MAX_RECORDS, "sort": "datedesc",
              "startdatetime": _fmt(start), "enddatetime": _fmt(end)}
    backoff = PAUSE_S
    for attempt in range(1, MAX_ATTEMPTS + 1):
        check()
        if _last_request is not None:
            _wait(max(0, backoff - (time.monotonic() - _last_request)), check)
        _last_request = time.monotonic()
        LOG.info("attempt=%s/%s start=%s end=%s query=%r", attempt, MAX_ATTEMPTS,
                 start.isoformat(), end.isoformat(), query)
        try:
            r = requests.get(DOC_API, params=params, timeout=60)
            LOG.info("attempt=%s status=%s bytes=%s", attempt, r.status_code, len(r.content))
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                articles = r.json().get("articles")
                if not isinstance(articles, list):
                    LOG.warning("invalid articles payload; retrying")
                else:
                    return articles
            # Blank responses are ambiguous, so never mark these windows complete.
            backoff = min(120 * 2 ** (attempt - 1), 480) if r.status_code == 429 else PAUSE_S
        except (requests.RequestException, ValueError) as exc:
            LOG.warning("attempt=%s error=%s", attempt, type(exc).__name__)
            backoff = PAUSE_S
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


def in_window(articles, start, end):
    """Drop missing, malformed and out-of-window crawl times before insertion."""
    rows = []
    for article in articles:
        try:
            seen = parse_seendate(article["seendate"])
            if start.timestamp() <= seen < end.timestamp() and article.get("url"):
                rows.append((article["url"], article.get("title") or "", seen,
                             article.get("domain")))
        except (KeyError, TypeError, ValueError):
            continue
    LOG.info("window=%s/%s received=%s kept=%s dropped=%s", start.isoformat(), end.isoformat(),
             len(articles), len(rows), len(articles) - len(rows))
    return rows


def progress(conn, started, total, state="running"):
    ok = conn.execute("SELECT COUNT(*) FROM fetch_log f JOIN markets m USING (market_id) "
                      "WHERE m.selected=1 AND f.status='ok'").fetchone()[0]
    out = {"started_at": started, "updated_at": time.time(), "elapsed_s": time.time() - started,
           "ok_chunks": ok, "total_chunks": total, "fraction_ok": ok / total if total else 0,
           "raw_headlines": conn.execute("SELECT COUNT(*) FROM headlines_raw").fetchone()[0],
           "state": state}
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(out, indent=2) + "\n")
    return out


def record_blocker(out):
    path = Path("docs/STATUS.md")
    text = path.read_text() if path.exists() else "# Status\n"
    marker = "\n## GDELT collection probe\n"
    text = text.split(marker)[0]
    stamp = datetime.now(timezone.utc).isoformat()
    text += (f"{marker}\nStopped {stamp}: {out['ok_chunks']}/{out['total_chunks']} chunks ok "
             f"({out['fraction_ok']:.2%}) after {out['elapsed_s'] / 3600:.3f} hours; "
             f"{out['raw_headlines']} raw headlines. Fewer than 20% completed in three hours. "
             "Q2 collection is blocked; no partial-data model results were computed. "
             "Owner + Claude must choose an alternative timestamped source. Google News RSS "
             "remains rejected because date-only timestamps can leak the move.\n")
    path.write_text(text)


def collect() -> None:
    conn = connect()
    todo = conn.execute(
        "SELECT q.market_id, q.query FROM queries q JOIN markets m USING (market_id) "
        "WHERE m.selected = 1 ORDER BY m.candidate_rank").fetchall()
    total = len(todo) * len(chunks())
    if not total:
        raise RuntimeError("No selected market queries; run hms.markets and hms.queries first")
    previous = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    started = previous.get("started_at", time.time())

    def check():
        if time.time() - started >= PROBE_SECONDS:
            out = progress(conn, started, total)
            if out["fraction_ok"] < 0.2:
                out = progress(conn, started, total, "blocked")
                record_blocker(out)
                raise CollectionBlocked(f"{out['ok_chunks']}/{total} chunks ok after three hours")

    try:
        progress(conn, started, total)
        expected = {(mid, s.isoformat(), e.isoformat())
                    for mid, _ in todo for s, e in chunks()}
        while True:
            done = set(conn.execute(
                "SELECT market_id, start, end FROM fetch_log WHERE status='ok'"))
            if expected <= done:
                progress(conn, started, total, "complete")
                return
            for market_id, query in todo:
                for start, end in chunks():
                    if (market_id, start.isoformat(), end.isoformat()) in done:
                        continue
                    check()
                    arts = fetch(query, start, end, check)
                    pieces = [(start, end, arts)]
                    if arts is not None and len(arts) >= MAX_RECORDS:
                        pieces = [(s, e, fetch(query, s, e, check)) for s, e in split10(start, end)]
                    count, complete = 0, True
                    for s, e, articles in pieces:
                        if articles is None or len(articles) >= MAX_RECORDS:
                            # Saturated subwindows cannot establish complete coverage.
                            complete = False
                        if articles is None:
                            continue
                        rows = in_window(articles, s, e)
                        conn.executemany("INSERT OR IGNORE INTO headlines_raw VALUES (?,?,?,?,?)",
                                         [(market_id, *r) for r in rows])
                        count += len(rows)
                    status = "ok" if complete else "failed"
                    conn.execute("INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?)",
                                 (market_id, start.isoformat(), end.isoformat(), count, status))
                    conn.commit()
                    out = progress(conn, started, total)
                    LOG.info("market=%s chunk=%s status=%s progress=%s/%s", market_id,
                             start.isoformat(), status, out["ok_chunks"], total)
            _wait(PAUSE_S, check)
    finally:
        conn.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(message)s")
    logging.Formatter.converter = time.gmtime
    try:
        collect()
    except CollectionBlocked as exc:
        LOG.error("STOP: %s", exc)
        raise SystemExit(3) from exc


if __name__ == "__main__":
    main()
