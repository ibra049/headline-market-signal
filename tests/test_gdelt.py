"""Collection regression tests: no real network, sleeps, labels or price moves."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from hms import gdelt
from hms.db import connect


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(gdelt.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(gdelt.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(gdelt, "_last_request", None)
    return now


def response(status=200, articles=None, body=None):
    text = body if body is not None else json.dumps({"articles": articles or []})
    return SimpleNamespace(status_code=status, text=text, content=text.encode(),
                           json=lambda: json.loads(text))


def test_rate_limit_backoff_six_attempts(monkeypatch, clock, caplog):
    times = []

    def get(*args, **kwargs):
        times.append(clock[0])
        return response(429)

    monkeypatch.setattr(gdelt.requests, "get", get)
    start, end = gdelt.chunks()[0]
    with caplog.at_level("INFO"):
        assert gdelt.fetch("unchanged query", start, end) is None
    assert times == [0, 120, 360, 840, 1320, 1800]
    assert sum("status=429" in r.message for r in caplog.records) == 6


def test_successes_still_paced_and_empty_json_is_valid(monkeypatch, clock):
    times = []

    def get(*args, **kwargs):
        times.append(clock[0])
        return response()

    monkeypatch.setattr(gdelt.requests, "get", get)
    for _ in range(3):
        assert gdelt.fetch("q", *gdelt.chunks()[0]) == []
    assert times == [0, 30, 60]


def test_blank_or_invalid_payload_not_marked_complete(monkeypatch, clock):
    responses = iter([response(body=""), response(body='{"error":"bad query"}'), response()])
    monkeypatch.setattr(gdelt.requests, "get", lambda *a, **k: next(responses))
    assert gdelt.fetch("q", *gdelt.chunks()[0]) == []
    assert clock[0] == 60


def test_seendate_window_is_half_open():
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    articles = [{"url": str(i), "title": "A news title with five words", "seendate": seen}
                for i, seen in enumerate(["20260731T235959Z", "20260801T000000Z",
                                          "20260801T235959Z", "20260802T000000Z",
                                          "20260803T000000Z", "bad", None])]
    assert [r[0] for r in gdelt.in_window(articles, start, end)] == ["1", "2"]


def test_collection_resumes_without_refetching_ok_chunks(tmp_path, monkeypatch):
    db = tmp_path / "test.sqlite"
    conn = connect(db)
    conn.execute("INSERT INTO markets (market_id,question,token_yes,selected) "
                 "VALUES ('m','q','tok',1)")
    conn.execute("INSERT INTO queries VALUES ('m','fixed query','model','created')")
    start, end = gdelt.chunks()[0]
    conn.execute("INSERT INTO fetch_log VALUES (?,?,?,0,'ok')",
                 ("m", start.isoformat(), end.isoformat()))
    conn.commit()
    conn.close()
    monkeypatch.setattr(gdelt, "connect", lambda: connect(db))
    monkeypatch.setattr(gdelt, "PROGRESS", tmp_path / "progress.json")
    monkeypatch.setattr(gdelt, "_wait", lambda *args: None)
    calls = []

    def fetch(query, start, end, check):
        calls.append((query, start, end))
        return []

    monkeypatch.setattr(gdelt, "fetch", fetch)
    gdelt.collect()
    assert len(calls) == 2 and all(c[0] == "fixed query" for c in calls)
    progress = json.loads(gdelt.PROGRESS.read_text())
    assert (progress["ok_chunks"], progress["total_chunks"], progress["state"]) == (
        3, 3, "complete")


def test_three_hour_probe_stops_before_another_request(tmp_path, monkeypatch):
    db = tmp_path / "test.sqlite"
    conn = connect(db)
    conn.execute("INSERT INTO markets (market_id,question,token_yes,selected) "
                 "VALUES ('m','q','tok',1)")
    conn.execute("INSERT INTO queries VALUES ('m','fixed query','model','created')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(gdelt, "connect", lambda: connect(db))
    monkeypatch.setattr(gdelt, "PROGRESS", tmp_path / "progress.json")
    gdelt.PROGRESS.write_text(json.dumps({"started_at": 0}))
    monkeypatch.setattr(gdelt.time, "time", lambda: 3 * 3600)
    blockers = []
    monkeypatch.setattr(gdelt, "record_blocker", blockers.append)
    monkeypatch.setattr(gdelt, "fetch", lambda *a: pytest.fail("requested after probe failed"))
    with pytest.raises(gdelt.CollectionBlocked):
        gdelt.collect()
    assert blockers[0]["ok_chunks"] == 0
    assert json.loads(gdelt.PROGRESS.read_text())["state"] == "blocked"
