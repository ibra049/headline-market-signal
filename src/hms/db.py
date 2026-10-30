"""SQLite store for markets, prices, headlines and aligned events."""

import os
import sqlite3
from pathlib import Path

# Outside the repo by default: cloud-synced folders (e.g. iCloud Documents) corrupt live SQLite.
DB_PATH = Path(os.environ.get("HMS_DB", Path.home() / ".hms" / "hms.sqlite"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    market_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    slug TEXT,
    token_yes TEXT NOT NULL,
    start_date TEXT,
    end_date TEXT,
    volume REAL,
    tags TEXT,
    candidate_rank INTEGER,
    history_days REAL,
    selected INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS prices (
    token_id TEXT NOT NULL,
    t INTEGER NOT NULL,          -- unix seconds, UTC
    p REAL NOT NULL,
    PRIMARY KEY (token_id, t)
);
CREATE TABLE IF NOT EXISTS queries (
    market_id TEXT PRIMARY KEY REFERENCES markets(market_id),
    query TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fetch_log (
    market_id TEXT NOT NULL,
    start TEXT NOT NULL,
    end TEXT NOT NULL,
    n_returned INTEGER,
    status TEXT,
    PRIMARY KEY (market_id, start, end)
);
CREATE TABLE IF NOT EXISTS headlines_raw (
    market_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    seendate INTEGER NOT NULL,   -- unix seconds, UTC (GDELT crawl time)
    domain TEXT,
    PRIMARY KEY (market_id, url)
);
CREATE TABLE IF NOT EXISTS headlines (
    headline_id INTEGER PRIMARY KEY,
    market_id TEXT NOT NULL,
    title TEXT NOT NULL,
    norm_title TEXT NOT NULL,
    seendate INTEGER NOT NULL,
    domain TEXT
);
CREATE TABLE IF NOT EXISTS events (
    headline_id INTEGER PRIMARY KEY REFERENCES headlines(headline_id),
    market_id TEXT NOT NULL,
    t0 INTEGER NOT NULL,
    p0 REAL,
    p1 REAL,
    dp REAL,
    label TEXT,                  -- up / down / flat
    split TEXT                   -- train / val / test / purged / out
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn
