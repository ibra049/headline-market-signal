from types import SimpleNamespace

import pytest

from hms import runtime


def test_low_memory_prevents_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="System-wide memory free percentage: 19%"))
    with pytest.raises(RuntimeError, match=">= 20%"):
        with runtime.training_guard():
            pytest.fail("entered training with low memory")


def test_only_one_training_process_and_lock_released(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="System-wide memory free percentage: 20%"))
    with runtime.training_guard():
        with pytest.raises(RuntimeError, match="Another training process"):
            with runtime.training_guard():
                pytest.fail("concurrent training admitted")
    with runtime.training_guard():
        pass
