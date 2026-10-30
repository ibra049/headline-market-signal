"""Q2 fine-tuned DistilRoBERTa on (market question, headline) -> P(up). One process per run.

python -m hms.q2_finetune --run lr2e-05_ep2 [--headline-only]
python -m hms.q2_finetune --select
"""

import argparse
import hashlib
import json
import time

import mlflow
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hms.q2 import SCORES, _write_cost, load_events, save
from hms.runtime import training_guard

BASE = "distilroberta-base"
GRID = ["lr2e-05_ep2", "lr5e-05_ep2"]  # amendment A1
BATCH, MAX_LEN, SEED = 8, 128, 42
RUNS = SCORES / "ft_runs"
SELECTION = RUNS / "selection.json"


def fingerprint(df):
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def parse(name: str) -> tuple[float, int]:
    lr, ep = name.split("_")
    return float(lr[2:]), int(ep[2:])


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode(tok, df, headline_only: bool):
    if headline_only:
        return tok(list(df["title"]), truncation=True, max_length=MAX_LEN, padding=True,
                   return_tensors="pt")
    return tok(list(df["question"]), list(df["title"]), truncation=True, max_length=MAX_LEN,
               padding=True, return_tensors="pt")


@torch.no_grad()
def predict(model, tok, df, dev, headline_only: bool, batch: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(df), batch):
        enc = {k: v.to(dev) for k, v in encode(tok, df.iloc[s:s + batch], headline_only).items()}
        out.append(torch.softmax(model(**enc).logits, -1)[:, 1].float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train(name: str, headline_only: bool) -> None:
    if name not in GRID:
        raise ValueError(f"Run must be one of {GRID}")
    lr, epochs = parse(name)
    tag = name + ("_h" if headline_only else "_qh")
    torch.manual_seed(SEED)
    df = load_events()
    data_sha = fingerprint(df)
    metadata = RUNS / f"{tag}.json"
    checkpoint = RUNS / tag
    if metadata.exists() and checkpoint.exists():
        if json.loads(metadata.read_text())["data_sha256"] != data_sha:
            raise RuntimeError("Events changed since training; do not reuse this run")
        print(f"skip completed {tag}", flush=True)
        return
    if SELECTION.exists():
        chosen = json.loads(SELECTION.read_text())["selected"]
        if not headline_only or name != chosen:
            raise RuntimeError("Selection is frozen; only its headline-only ablation may train")
    tr = df[(df.split == "train") & (df.label != "flat")].reset_index(drop=True)
    va = df[(df.split == "val") & (df.label != "flat")]
    if tr.y.nunique() < 2 or va.y.nunique() < 2:
        raise RuntimeError("Training and validation each require both direction classes")
    dev = device()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, attn_implementation="eager").to(dev)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("q2-direction")
    rng = np.random.default_rng(SEED)
    with mlflow.start_run(run_name=tag):
        mlflow.log_params({"base": BASE, "lr": lr, "epochs": epochs, "batch": BATCH,
                           "max_len": MAX_LEN, "headline_only": headline_only, "n_train": len(tr)})
        for ep in range(epochs):
            model.train()
            order = rng.permutation(len(tr))
            for s in range(0, len(order), BATCH):
                rows = tr.iloc[order[s:s + BATCH]]
                enc = {k: v.to(dev) for k, v in encode(tok, rows, headline_only).items()}
                loss = model(**enc, labels=torch.tensor(rows["y"].values, device=dev)).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
            val_auc = roc_auc_score(va["y"], predict(model, tok, va, dev, headline_only))
            mlflow.log_metric("val_auc", val_auc, step=ep + 1)
            print(f"{tag} epoch {ep + 1} val_auc {val_auc:.4f}", flush=True)
        t = time.perf_counter()
        # No sweep checkpoint scores the frozen test set before selection.
        development = df[df.split != "test"]
        score = predict(model, tok, development, dev, headline_only)
        per_1k = (time.perf_counter() - t) / len(development) * 1000
        model.save_pretrained(checkpoint)
        tok.save_pretrained(checkpoint)
    RUNS.mkdir(parents=True, exist_ok=True)
    save(f"ft_runs/{tag}", development, score)
    metadata.write_text(json.dumps({"val_auc": val_auc, "latency_s_per_1k": per_1k,
                                   "device": str(dev), "data_sha256": data_sha}))


def select() -> None:
    """Pick the grid run with the best validation AUC; never looks at test."""
    df = load_events()
    data_sha = fingerprint(df)
    info = {n: json.loads((RUNS / f"{n}_qh.json").read_text()) for n in GRID}
    if any(r["data_sha256"] != data_sha for r in info.values()):
        raise RuntimeError("Events changed since training; selection is invalid")
    vals = {n: r["val_auc"] for n, r in info.items()}
    best = max(vals, key=vals.get)
    selection = {"selected": best, "val_auc": vals, "data_sha256": data_sha}
    if SELECTION.exists() and json.loads(SELECTION.read_text()) != selection:
        raise RuntimeError("The frozen selection changed; refusing to rescore test")
    # Commit the choice to disk BEFORE loading or scoring any test inputs.
    SELECTION.write_text(json.dumps(selection, indent=2) + "\n")
    for suffix in ("qh", "h"):
        checkpoint = RUNS / f"{best}_{suffix}"
        target = SCORES / f"finetuned_{suffix}.csv"
        if not checkpoint.exists() or target.exists():
            continue
        dev = device()
        tok = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(dev)
        t = time.perf_counter()
        score = predict(model, tok, df, dev, suffix == "h")
        per_1k = (time.perf_counter() - t) / len(df) * 1000
        save(f"finetuned_{suffix}", df, score)
        _write_cost(f"finetuned_{suffix}", {"latency_s_per_1k": per_1k,
                                           "device": str(dev), "cost_usd_per_1k": 0.0})
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()
    print(json.dumps({"val_auc": vals, "selected": best}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", choices=GRID)
    ap.add_argument("--headline-only", action="store_true")
    group.add_argument("--select", action="store_true")
    a = ap.parse_args()
    if a.select:
        select()
    else:
        with training_guard():
            train(a.run, a.headline_only)
