"""Guardian publication timestamps and collection bounds, without network calls."""

import pytest

from hms import guardian
from hms.db import connect


@pytest.mark.parametrize("value", [
    "2026-08-01T00:00:00Z",
    "2026-08-01T00:00:00+00:00",
    "2026-07-31T20:00:00-04:00",
    "2026-08-01T00:00:00.999Z",
])
def test_parse_ts_uses_utc_seconds(value):
    assert guardian.parse_ts(value) == int(guardian.WINDOW_START.timestamp())


def test_parse_ts_rejects_invalid_timestamp():
    with pytest.raises(ValueError):
        guardian.parse_ts("invalid")


def test_collection_filters_window_and_keeps_both_boundaries(tmp_path, monkeypatch):
    path = tmp_path / "guardian.sqlite"
    with connect(path) as conn:
        conn.execute("INSERT INTO markets (market_id, question, token_yes, selected) "
                     "VALUES ('m', 'Market question?', 'tok', 1)")
        conn.execute("INSERT INTO queries VALUES ('m', 'fixed query', 'fake', '2026-10-30')")

    dates = ["2026-07-31T23:59:59Z", "2026-08-01T00:00:00Z",
             "2026-10-29T23:59:59Z", "2026-10-30T00:00:00Z"]
    articles = [{"webUrl": f"https://www.theguardian.com/{i}",
                 "webTitle": "A headline with at least five words", "webPublicationDate": date}
                for i, date in enumerate(dates)]
    calls = []

    def fetch(query, start, end):
        calls.append((query, start, end))
        return {"total": len(articles), "results": articles}

    monkeypatch.setattr(guardian, "connect", lambda: connect(path))
    monkeypatch.setattr(guardian, "load_key", lambda: None)
    monkeypatch.setattr(guardian, "PERIODS", [("2026-08-01", "2026-10-29")])
    monkeypatch.setattr(guardian, "fetch_page", fetch)
    guardian.collect()
    guardian.collect()  # completed markets resume without refetching

    with connect(path) as conn:
        rows = conn.execute("SELECT url, seendate FROM headlines_raw ORDER BY seendate").fetchall()
        assert rows == [(articles[i]["webUrl"], guardian.parse_ts(dates[i])) for i in (1, 2)]
        assert conn.execute("SELECT n_stored FROM guardian_log").fetchone()[0] == 2
    assert calls == [("fixed query", "2026-08-01", "2026-10-29")]
