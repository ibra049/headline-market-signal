"""Q2 fine-tuned DistilRoBERTa on (market question, headline) -> P(up). One process per run.

python -m hms.q2_finetune --run lr2e-05_ep2 [--headline-only]
python -m hms.q2_finetune --select
"""

import argparse
import json
import time

import mlflow
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hms.q2 import SCORES, _write_cost, load_events, save

BASE = "distilroberta-base"
GRID = ["lr2e-05_ep2", "lr5e-05_ep2"]  # amendment A1
BATCH, MAX_LEN, SEED = 8, 128, 42
RUNS = SCORES / "ft_runs"


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
    lr, epochs = parse(name)
    tag = name + ("_h" if headline_only else "_qh")
    torch.manual_seed(SEED)
    df = load_events()
    tr = df[(df.split == "train") & (df.label != "flat")].reset_index(drop=True)
    va = df[(df.split == "val") & (df.label != "flat")]
    dev = device()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2).to(dev)
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
        score = predict(model, tok, df, dev, headline_only)
        per_1k = (time.perf_counter() - t) / len(df) * 1000
    RUNS.mkdir(parents=True, exist_ok=True)
    save(f"ft_runs/{tag}", df, score)
    (RUNS / f"{tag}.json").write_text(json.dumps({"val_auc": val_auc, "latency_s_per_1k": per_1k,
                                                  "device": str(dev)}))


def select() -> None:
    """Pick the grid run with the best validation AUC; never looks at test."""
    vals = {n: json.loads((RUNS / f"{n}_qh.json").read_text())["val_auc"] for n in GRID
            if (RUNS / f"{n}_qh.json").exists()}
    best = max(vals, key=vals.get)
    for suffix in ("qh", "h"):
        src = RUNS / f"{best}_{suffix}.csv"
        if src.exists():
            (SCORES / f"finetuned_{suffix}.csv").write_text(src.read_text())
    info = json.loads((RUNS / f"{best}_qh.json").read_text())
    _write_cost("finetuned_qh", {"latency_s_per_1k": info["latency_s_per_1k"],
                                 "device": info["device"], "cost_usd_per_1k": 0.0})
    print(json.dumps({"val_auc": vals, "selected": best}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--headline-only", action="store_true")
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args()
    select() if a.select else train(a.run, a.headline_only)
