"""Evaluate a candidate against the frozen Q1 test set and print a JSON decision."""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from sklearn.metrics import f1_score

# Support both `python scripts/eval_gate.py` and imports from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hms.onnx_predict import LABELS, OnnxClassifier  # noqa: E402


def decide(candidate_f1: float, champion_f1: float | None, min_delta: float) -> bool:
    for score in (candidate_f1, champion_f1):
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("F1 must be finite and between 0 and 1")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative")
    return champion_f1 is None or candidate_f1 > champion_f1 + min_delta


def verify_sha256(test_path: str | Path, sha_path: str | Path | None = None) -> None:
    test_path = Path(test_path)
    sha_path = Path(sha_path) if sha_path is not None else test_path.with_suffix(".sha256")
    # Accept both a bare digest and the conventional `sha256sum` output format.
    fields = sha_path.read_text().split()
    if not fields or re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]) is None:
        raise ValueError(f"invalid SHA-256 file: {sha_path}")
    with test_path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != fields[0].lower():
        raise ValueError(f"SHA-256 mismatch for {test_path}")


def evaluate(candidate: Path, champion: Path, test: Path, min_delta: float) -> dict:
    verify_sha256(test)
    with test.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"text", "label"} <= set(reader.fieldnames):
            raise ValueError("test CSV must have text,label columns")
        rows = list(reader)
    if not rows:
        raise ValueError("test CSV must not be empty")
    if any(row["label"] not in LABELS or row["text"] is None for row in rows):
        raise ValueError("test CSV contains an invalid text or label")

    classifier = OnnxClassifier(candidate)
    predicted = classifier.predict([row["text"] for row in rows])
    candidate_f1 = float(
        f1_score(
            [row["label"] for row in rows],
            [prediction["label"] for prediction in predicted],
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
    )
    metadata_path = champion / "metadata.json"
    if (champion / "model.onnx").exists() and not metadata_path.exists():
        raise ValueError("champion model exists but metadata.json is missing")
    champion_f1 = (
        float(json.loads(metadata_path.read_text())["macro_f1"])
        if metadata_path.exists()
        else None
    )
    return {
        "candidate_f1": candidate_f1,
        "champion_f1": champion_f1,
        "delta": candidate_f1 - champion_f1 if champion_f1 is not None else None,
        "promote": decide(candidate_f1, champion_f1, min_delta),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=Path("data/frozen/q1_test.csv"))
    parser.add_argument("--min-delta", type=float, default=0.005)
    args = parser.parse_args()
    try:
        decision = evaluate(args.candidate, args.champion, args.test, args.min_delta)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"eval gate failed: {exc}\n")
    print(json.dumps(decision, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
