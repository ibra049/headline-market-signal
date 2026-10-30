import json

import numpy as np
import pandas as pd
import pytest

from hms import q2


def events():
    return pd.DataFrame({"headline_id": [1, 2, 3], "market_id": ["a", "b", "b"],
                         "split": ["test"] * 3, "label": ["up", "down", "flat"],
                         "y": [1, 0, 0], "dp": [0.01, -0.01, 0]})


def test_empty_report_is_data_blocked_and_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(q2, "load_events", lambda: events().iloc[:0])
    monkeypatch.setattr(q2, "SCORES", tmp_path / "scores")
    monkeypatch.setattr(q2, "RESULTS", tmp_path)
    monkeypatch.setattr(q2, "calibration_plot", lambda df: None)
    q2.report()
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["decision"] == "DATA-BLOCKED"
    assert result["test_up_share"] is None
    assert result["n_test_nonflat"] == 0
    assert all(value is None for value in result["models"].values())
    assert "NaN" not in (tmp_path / "results.json").read_text()


@pytest.mark.parametrize("scores", [[0.9], [0.9, np.nan, 0.5], [0.9, 1.1, 0.5]])
def test_incomplete_or_invalid_scores_cannot_be_reported(tmp_path, monkeypatch, scores):
    monkeypatch.setattr(q2, "SCORES", tmp_path)
    pd.DataFrame({"headline_id": list(range(1, len(scores) + 1)), "score": scores}).to_csv(
        tmp_path / "baseline_qh.csv", index=False)
    with pytest.raises(ValueError):
        q2.evaluate(events(), "baseline_qh")


def test_reliability_curve_uses_nonflat_events(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(q2, "SCORES", tmp_path)
    monkeypatch.setattr(q2, "RESULTS", tmp_path)
    pd.DataFrame({"headline_id": [1, 2, 3], "score": [0.9, 0.1, 0.5]}).to_csv(
        tmp_path / "baseline_qh.csv", index=False)
    q2.calibration_plot(events())
    assert (tmp_path / "calibration.png").stat().st_size > 1000
