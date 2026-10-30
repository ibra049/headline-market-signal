"""Q1 data: download Financial PhraseBank (66% agreement) and write the frozen split."""

from __future__ import annotations

import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download

from hms.q1_common import FROZEN, ROOT, sha256_file, split_70_15_15, update_results

RAW = ROOT / "data" / "raw" / "phrasebank"
MEMBER = "FinancialPhraseBank-v1.0/Sentences_66Agree.txt"


def load_phrasebank() -> pd.DataFrame:
    zpath = hf_hub_download(
        "takala/financial_phrasebank",
        "data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
        local_dir=RAW,
    )
    with zipfile.ZipFile(zpath) as z:
        raw = z.read(MEMBER).decode("latin-1")
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        text, _, label = line.rpartition("@")
        rows.append({"text": text.strip(), "label": label.strip()})
    df = pd.DataFrame(rows)
    assert set(df["label"]) == {"negative", "neutral", "positive"}, set(df["label"])
    return df


def main() -> None:
    df = load_phrasebank()
    n_raw = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    train, val, test = split_70_15_15(df)
    FROZEN.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(FROZEN / f"q1_{name}.csv", index=False)
    digest = sha256_file(FROZEN / "q1_test.csv")
    (FROZEN / "q1_test.sha256").write_text(digest + "\n")
    info = {
        "source": "takala/financial_phrasebank Sentences_66Agree (CC BY-NC-SA 3.0)",
        "n_raw": n_raw,
        "n_after_exact_dedupe": len(df),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "label_counts_test": test["label"].value_counts().to_dict(),
        "test_sha256": digest,
    }
    update_results("data", info)
    print(info)


if __name__ == "__main__":
    main()
