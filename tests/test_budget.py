import json

import pytest

from hms import budget


def test_spend_includes_both_legacy_and_new_cache_formats(tmp_path):
    (tmp_path / 'q1.jsonl').write_text(json.dumps({
        'input_tokens': 1000, 'output_tokens': 100, 'cache_read': 1000, 'cache_write': 1000,
    }) + '\n')
    (tmp_path / 'q2.jsonl').write_text(json.dumps({'cost_usd': 0.02}) + '\n')
    assert budget.cached_spend(tmp_path) == pytest.approx(0.020285)


def test_budget_reserves_for_concurrent_calls_and_releases(monkeypatch):
    monkeypatch.setattr(budget, 'cached_spend', lambda: 4.97)
    guard = budget.SpendGuard()
    with guard.request() as billed:
        with pytest.raises(budget.SpendLimitExceeded):
            with guard.request():
                pass
        billed[0] = 0.001
    assert guard.spent == pytest.approx(4.971)
    assert guard.reserved == 0


def test_already_over_budget_stops_before_calls(monkeypatch):
    monkeypatch.setattr(budget, 'cached_spend', lambda: 5.01)
    with pytest.raises(budget.SpendLimitExceeded):
        budget.SpendGuard()
