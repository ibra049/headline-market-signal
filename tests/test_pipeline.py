from hms import build
from hms.db import connect
from hms.gdelt import parse_seendate
from hms.markets import is_excluded

H = 3600


def test_normalize_strips_source_and_punctuation():
    got = build.normalize("Fed Holds Rates Steady, Again! - Reuters")
    assert got == "fed holds rates steady again"


def test_dedupe_keeps_earliest_and_allows_repeat_after_24h():
    title = "Senate passes the spending bill tonight"
    rows = [("m", title, 10 * H, "a"), ("m", title, 2 * H, "b"), ("m", title, 30 * H, "c"),
            ("m", "too short title", 0, "d")]
    kept = build.dedupe(rows)
    assert [(r[3], r[4]) for r in kept] == [(2 * H, "b"), (30 * H, "c")]


def test_exclusion_rules():
    assert is_excluded("Lakers vs. Celtics", [])
    assert is_excluded("Bitcoin above $120,000 on October 31?", [])
    assert is_excluded("Will X happen?", ["sports"])
    assert not is_excluded("Will the Fed cut rates in December?", ["economy"])


def test_parse_seendate():
    assert parse_seendate("20261009T000000Z") == build.TEST_START


def _db(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    conn.execute("INSERT INTO markets (market_id, question, token_yes, selected) "
                 "VALUES ('m', 'q', 'tok', 1)")
    return conn


def test_alignment_uses_only_past_prices_and_labels(tmp_path):
    conn = _db(tmp_path)
    seen = build.TRAIN_START + 10 * H
    t0 = seen + build.LAG_S
    prices = [(t0 - 30 * 60, 0.40),   # last price known at t0 -> p0
              (t0 + 10 * 60, 0.90),   # after t0: must not be used as p0
              (t0 + build.HORIZON_S - 5 * 60, 0.43),  # last price at or before t0 + 6 h -> p1
              (t0 + build.HORIZON_S + 60, 0.99)]      # after horizon: ignored
    conn.executemany("INSERT INTO prices VALUES ('tok', ?, ?)", prices)
    conn.execute("INSERT INTO headlines (market_id, title, norm_title, seendate) "
                 "VALUES ('m', 't', 't', ?)", (seen,))
    conn.executescript(build.ALIGN_SQL)
    p0, p1, label, split = conn.execute("SELECT p0, p1, label, split FROM events").fetchone()
    assert (p0, p1, label, split) == (0.40, 0.43, "up", "train")


def test_stale_price_drops_event_and_purge(tmp_path):
    conn = _db(tmp_path)
    seen = build.VAL_START - 3 * H  # label window crosses into validation -> purged
    t0 = seen + build.LAG_S
    conn.executemany("INSERT INTO prices VALUES ('tok', ?, ?)",
                     [(t0 - 60, 0.5), (t0 + build.HORIZON_S - 60, 0.5)])
    conn.execute("INSERT INTO headlines (market_id, title, norm_title, seendate) "
                 "VALUES ('m', 'a', 'a', ?)", (seen,))
    conn.execute("INSERT INTO headlines (market_id, title, norm_title, seendate) "
                 "VALUES ('m', 'b', 'b', ?)", (build.TRAIN_START + 100 * H,))  # no prices nearby
    conn.executescript(build.ALIGN_SQL)
    rows = dict(conn.execute("SELECT headline_id, split FROM events").fetchall())
    assert rows == {1: "purged", 2: "no_price"}
    assert conn.execute("SELECT label FROM events WHERE headline_id = 1").fetchone()[0] == "flat"
