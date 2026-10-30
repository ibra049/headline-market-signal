import hashlib
import json
import sys

import pytest

from scripts import eval_gate


@pytest.mark.parametrize(
    ("candidate", "champion", "minimum", "expected"),
    [
        (0.9, 0.8, 0.005, True),
        (0.8, 0.8, 0.005, False),
        (0.7, 0.8, 0.005, False),
        (0.8 + 0.005, 0.8, 0.005, False),
        (0.805001, 0.8, 0.005, True),
        (0.804999, 0.8, 0.005, False),
        (0.8, 0.8, 0.0, False),
        (0.0, None, 0.005, True),
    ],
)
def test_decide(candidate, champion, minimum, expected):
    assert eval_gate.decide(candidate, champion, minimum) is expected


@pytest.mark.parametrize(
    ("candidate", "champion", "minimum"),
    [(float("nan"), 0.8, 0.005), (0.8, float("inf"), 0.005), (1.1, None, 0.005),
     (0.8, 0.7, -0.005), (0.8, 0.7, float("nan"))],
)
def test_invalid_scores(candidate, champion, minimum):
    with pytest.raises(ValueError):
        eval_gate.decide(candidate, champion, minimum)


@pytest.fixture
def frozen_csv(tmp_path):
    path = tmp_path / "q1_test.csv"
    path.write_text("text,label\nloss,negative\nflat,neutral\ngain,positive\n", encoding="utf-8")
    path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    return path


@pytest.mark.parametrize("with_filename", [False, True])
def test_sha_accepts_valid_digest(frozen_csv, with_filename):
    if with_filename:
        sha_path = frozen_csv.with_suffix(".sha256")
        sha_path.write_text(sha_path.read_text().strip().upper() + "  q1_test.csv\n")
    eval_gate.verify_sha256(frozen_csv)


def test_sha_rejects_modified_data(frozen_csv):
    frozen_csv.write_text(frozen_csv.read_text() + "extra,positive\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        eval_gate.verify_sha256(frozen_csv)


@pytest.mark.parametrize("digest", ["", "not-a-sha", "g" * 64])
def test_sha_rejects_invalid_digest(frozen_csv, digest):
    frozen_csv.with_suffix(".sha256").write_text(digest)
    with pytest.raises(ValueError, match="invalid SHA-256"):
        eval_gate.verify_sha256(frozen_csv)


def test_sha_requires_recorded_hash(frozen_csv):
    frozen_csv.with_suffix(".sha256").unlink()
    with pytest.raises(FileNotFoundError):
        eval_gate.verify_sha256(frozen_csv)


class FakeClassifier:
    def __init__(self, _):
        pass

    def predict(self, texts):
        labels = {"loss": "negative", "flat": "neutral", "gain": "positive"}
        return [{"label": labels[text]} for text in texts]


def test_evaluate_uses_champion_metadata(monkeypatch, tmp_path, frozen_csv):
    monkeypatch.setattr(eval_gate, "OnnxClassifier", FakeClassifier)
    champion = tmp_path / "champion"
    champion.mkdir()
    (champion / "metadata.json").write_text(json.dumps({"macro_f1": 0.9}))
    result = eval_gate.evaluate(tmp_path / "candidate", champion, frozen_csv, 0.005)
    assert result["candidate_f1"] == 1.0
    assert result["champion_f1"] == 0.9
    assert result["delta"] == pytest.approx(0.1)
    assert result["promote"] is True


def test_evaluate_without_champion(monkeypatch, tmp_path, frozen_csv):
    monkeypatch.setattr(eval_gate, "OnnxClassifier", FakeClassifier)
    result = eval_gate.evaluate(tmp_path / "candidate", tmp_path / "absent", frozen_csv, 0.005)
    assert result == {
        "candidate_f1": 1.0, "champion_f1": None, "delta": None, "promote": True,
    }


def test_existing_champion_requires_metadata(monkeypatch, tmp_path, frozen_csv):
    monkeypatch.setattr(eval_gate, "OnnxClassifier", FakeClassifier)
    champion = tmp_path / "champion"
    champion.mkdir()
    (champion / "model.onnx").touch()
    with pytest.raises(ValueError, match="metadata.json is missing"):
        eval_gate.evaluate(tmp_path / "candidate", champion, frozen_csv, 0.005)


def test_macro_f1_includes_all_three_labels(monkeypatch, tmp_path, frozen_csv):
    monkeypatch.setattr(eval_gate, "OnnxClassifier", FakeClassifier)
    frozen_csv.write_text("text,label\nloss,negative\n")
    frozen_csv.with_suffix(".sha256").write_text(hashlib.sha256(frozen_csv.read_bytes()).hexdigest())
    result = eval_gate.evaluate(tmp_path, tmp_path / "absent", frozen_csv, 0.005)
    assert result["candidate_f1"] == pytest.approx(1 / 3)


def test_checksum_failure_prevents_inference(monkeypatch, tmp_path, frozen_csv):
    def forbidden(_):
        pytest.fail("classifier loaded before verifying checksum")

    monkeypatch.setattr(eval_gate, "OnnxClassifier", forbidden)
    frozen_csv.write_text("tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        eval_gate.evaluate(tmp_path, tmp_path, frozen_csv, 0.005)


@pytest.mark.parametrize(
    "content", ["wrong,columns\nx,y\n", "text,label\n", "text,label\nx,unknown\n"]
)
def test_invalid_test_csv(tmp_path, frozen_csv, content):
    frozen_csv.write_text(content)
    frozen_csv.with_suffix(".sha256").write_text(hashlib.sha256(frozen_csv.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="test CSV"):
        eval_gate.evaluate(tmp_path, tmp_path, frozen_csv, 0.005)


def test_cli_prints_json_and_exits_zero(monkeypatch, tmp_path, frozen_csv, capsys):
    monkeypatch.setattr(eval_gate, "OnnxClassifier", FakeClassifier)
    monkeypatch.setattr(sys, "argv", [
        "eval_gate.py", "--candidate", str(tmp_path / "candidate"),
        "--champion", str(tmp_path / "absent"), "--test", str(frozen_csv),
        "--min-delta", "0.005",
    ])
    assert eval_gate.main() == 0
    assert json.loads(capsys.readouterr().out)["promote"] is True


def test_cli_checksum_failure_is_nonzero(monkeypatch, tmp_path, frozen_csv, capsys):
    frozen_csv.write_text("changed")
    monkeypatch.setattr(sys, "argv", [
        "eval_gate.py", "--candidate", str(tmp_path), "--champion", str(tmp_path),
        "--test", str(frozen_csv),
    ])
    with pytest.raises(SystemExit) as exc:
        eval_gate.main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SHA-256 mismatch" in captured.err
