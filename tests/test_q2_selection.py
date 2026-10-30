"""Exercise selection and test-scoring order with fake models, never real training."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("mlflow")
pytest.importorskip("transformers")
from hms import q2_finetune as ft  # noqa: E402


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    df = pd.DataFrame({"headline_id": [1, 2, 3], "split": ["train", "val", "test"],
                       "title": ["train title", "val title", "test title"]})
    runs = tmp_path / "ft_runs"
    runs.mkdir()
    selection = runs / "selection.json"
    monkeypatch.setattr(ft, "RUNS", runs)
    monkeypatch.setattr(ft, "SCORES", tmp_path)
    monkeypatch.setattr(ft, "SELECTION", selection)
    monkeypatch.setattr(ft, "load_events", lambda: df)
    for name, auc in zip(ft.GRID, [0.60, 0.55]):
        (runs / f"{name}_qh").mkdir()
        (runs / f"{name}_qh.json").write_text(json.dumps(
            {"val_auc": auc, "data_sha256": ft.fingerprint(df)}))
    calls = []

    def load(path):
        assert selection.exists(), "model loaded before validation selection was frozen"
        calls.append(path)
        return SimpleNamespace(to=lambda dev: object())

    monkeypatch.setattr(ft.AutoModelForSequenceClassification, "from_pretrained", load)
    monkeypatch.setattr(ft.AutoTokenizer, "from_pretrained", lambda p: object())
    monkeypatch.setattr(ft, "device", lambda: SimpleNamespace(type="cpu"))
    monkeypatch.setattr(ft, "predict", lambda *a: np.array([0.4, 0.5, 0.6]))
    monkeypatch.setattr(ft, "_write_cost", lambda *a: None)

    def save(name, df, scores):
        pd.DataFrame({"headline_id": df.headline_id, "split": df.split, "score": scores}).to_csv(
            tmp_path / f"{name}.csv", index=False)

    monkeypatch.setattr(ft, "save", save)
    return runs, selection, calls


def test_freeze_selection_before_test_scoring_and_score_once(sweep):
    runs, selection, calls = sweep
    ft.select()
    assert json.loads(selection.read_text())["selected"] == ft.GRID[0]
    assert len(calls) == 1
    ft.select()
    assert len(calls) == 1  # a rerun cannot score frozen test again
    (runs / f"{ft.GRID[0]}_h").mkdir()
    ft.select()
    assert len(calls) == 2  # only the new ablation is scored


def test_incomplete_sweep_cannot_score_test(sweep):
    runs, selection, calls = sweep
    (runs / f"{ft.GRID[1]}_qh.json").unlink()
    with pytest.raises(FileNotFoundError):
        ft.select()
    assert not selection.exists() and not calls


def test_changed_validation_winner_cannot_rescore_test(sweep):
    runs, selection, calls = sweep
    ft.select()
    path = runs / f"{ft.GRID[1]}_qh.json"
    info = json.loads(path.read_text())
    info["val_auc"] = 0.9
    path.write_text(json.dumps(info))
    with pytest.raises(RuntimeError, match="frozen selection changed"):
        ft.select()
    assert len(calls) == 1
