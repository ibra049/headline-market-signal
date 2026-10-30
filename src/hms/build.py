"""Clean, dedupe and align headlines to prices; assign labels and time splits (all in SQL)."""

import re
from datetime import datetime, timezone

from hms.db import connect

LAG_S = 15 * 60
HORIZON_S = 6 * 3600
STALE_S = 2 * 3600
FLAT = 0.005
DEDUPE_S = 24 * 3600
MIN_WORDS = 5


def ts(y: int, m: int, d: int) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())


TRAIN_START, VAL_START, TEST_START = ts(2026, 8, 1), ts(2026, 9, 22), ts(2026, 10, 9)
WINDOW_END = int(datetime(2026, 10, 29, 23, 59, 59, tzinfo=timezone.utc).timestamp())

_SOURCE_SUFFIX = re.compile(r"\s+[-|–—]\s+[^-|–—]{2,40}$")
_NON_WORD = re.compile(r"[^\w\s]")


def normalize(title: str) -> str:
    t = _SOURCE_SUFFIX.sub("", title.strip())
    t = _NON_WORD.sub(" ", t.lower())
    return " ".join(t.split())


def dedupe(rows: list[tuple]) -> list[tuple]:
    """rows: (market_id, title, seendate, domain), any order. Keeps the earliest copy of a
    normalized title per market; a repeat within 24 h of the last kept copy is dropped."""
    kept, last_seen = [], {}
    for market_id, title, seen, domain in sorted(rows, key=lambda r: (r[0], r[2])):
        norm = normalize(title)
        if len(norm.split()) < MIN_WORDS:
            continue
        key = (market_id, norm)
        if key in last_seen and seen - last_seen[key] < DEDUPE_S:
            continue
        last_seen[key] = seen
        kept.append((market_id, title.strip(), norm, seen, domain))
    return kept


ALIGN_SQL = f"""
DELETE FROM events;
INSERT INTO events (headline_id, market_id, t0, p0, p1)
SELECT h.headline_id, h.market_id, h.seendate + {LAG_S},
  (SELECT pr.p FROM prices pr WHERE pr.token_id = m.token_yes
     AND pr.t <= h.seendate + {LAG_S} AND pr.t > h.seendate + {LAG_S - STALE_S}
     ORDER BY pr.t DESC LIMIT 1),
  (SELECT pr.p FROM prices pr WHERE pr.token_id = m.token_yes
     AND pr.t <= h.seendate + {LAG_S + HORIZON_S}
     AND pr.t > h.seendate + {LAG_S + HORIZON_S - STALE_S}
     ORDER BY pr.t DESC LIMIT 1)
FROM headlines h JOIN markets m USING (market_id);
UPDATE events SET dp = p1 - p0;
UPDATE events SET label = CASE WHEN dp IS NULL THEN NULL
  WHEN dp >= {FLAT} THEN 'up' WHEN dp <= -{FLAT} THEN 'down' ELSE 'flat' END;
UPDATE events SET split = CASE
  WHEN dp IS NULL THEN 'no_price'
  WHEN t0 < {TRAIN_START} OR t0 + {HORIZON_S} > {WINDOW_END} THEN 'out'
  WHEN t0 < {VAL_START} THEN
    CASE WHEN t0 + {HORIZON_S} >= {VAL_START} THEN 'purged' ELSE 'train' END
  WHEN t0 < {TEST_START} THEN
    CASE WHEN t0 + {HORIZON_S} >= {TEST_START} THEN 'purged' ELSE 'val' END
  ELSE 'test' END;
"""


def build() -> None:
    conn = connect()
    raw = conn.execute(
        "SELECT r.market_id, r.title, r.seendate, r.domain FROM headlines_raw r "
        "JOIN markets m USING (market_id) WHERE m.selected = 1").fetchall()
    conn.execute("DELETE FROM headlines")
    conn.executemany(
        "INSERT INTO headlines (market_id, title, norm_title, seendate, domain) VALUES (?,?,?,?,?)",
        dedupe(raw))
    conn.executescript(ALIGN_SQL)
    conn.commit()
    for row in conn.execute(
            "SELECT split, label, COUNT(*), COUNT(DISTINCT market_id) FROM events "
            "GROUP BY 1, 2 ORDER BY 1, 2"):
        print(row)


if __name__ == "__main__":
    build()
