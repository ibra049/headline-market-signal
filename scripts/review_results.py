"""One cached Claude interpretation review of final machine-generated study results."""

import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hms.llm import Haiku  # noqa: E402


class Review(BaseModel):
    interpretation: str
    concerns: list[str]


def main():
    paths = [Path(p) for p in ("results/q1/results.json", "results/q2/results.json",
                              "docs/STATUS.md", "PREREGISTRATION.md")]
    user = "\n\n".join(f"FILE {p}\n{p.read_text()}" for p in paths)
    system = (
        "Independently review the interpretation of this pre-registered research study. "
        "Give one short paragraph and at most three concerns. Judge the supplied final numbers "
        "against the frozen protocol; do not change thresholds or propose trading/profit claims. "
        "Missing Q2 data means data blocked, never evidence for a null result. A Q1 bootstrap "
        "interval including zero does not establish statistical equivalence. This is an "
        "interpretation review, not an independent code audit. Distinguish cached-rate spend "
        "estimates from account billing. Do not invent unavailable metrics."
    )
    row = Haiku("final_review").map(system, [user], Review, workers=1)[0]
    out = row["out"]
    text = ("# Claude interpretation review\n\n"
            "One cached Claude Haiku 5.5 API review of the final recorded results and protocol. "
            "This is not a code audit.\n\n" + out["interpretation"] + "\n")
    if out["concerns"]:
        text += "\n" + "\n".join(f"- {c}" for c in out["concerns"]) + "\n"
    Path("docs/claude_review.md").write_text(text)
    print(f"Review cached; recorded request cost ${row['cost_usd']:.8f}")


if __name__ == "__main__":
    main()
