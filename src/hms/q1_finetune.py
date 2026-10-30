"""Q1 fine-tune: distilroberta-base sweep (lr x epochs), MLflow tracking and registry."""

from __future__ import annotations

import json
import os
import shutil

import mlflow
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from hms.q1_common import (
    ARTIFACTS,
    LABEL2ID,
    LABELS,
    RESULTS,
    ROOT,
    SEED,
    classification_metrics,
    load_split,
    macro_f1,
    time_per_1k,
    update_results,
)

BASE = "distilroberta-base"
MAX_LEN = 128
BATCH = 16
GRID = [(2e-5, 2), (2e-5, 3), (5e-5, 2), (5e-5, 3)]
SWEEP_DIR = ARTIFACTS / "q1_sweep"
SELECTED_DIR = ARTIFACTS / "q1_selected_torch"


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def batches(tok, texts, labels=None, batch=BATCH, shuffle=False, seed=SEED):
    idx = list(range(len(texts)))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for s in range(0, len(idx), batch):
        chunk = idx[s : s + batch]
        enc = tok([texts[i] for i in chunk], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor([labels[i] for i in chunk])
        yield enc


@torch.no_grad()
def predict_proba(model, tok, texts, dev, batch=32) -> np.ndarray:
    model.eval()
    out = []
    for enc in batches(tok, texts, batch=batch):
        enc = {k: v.to(dev) for k, v in enc.items()}
        out.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def train_one(lr, epochs, train, val, dev):
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID
    ).to(dev)
    texts, labels = train["text"].tolist(), train["label"].map(LABEL2ID).tolist()
    steps = epochs * ((len(texts) + BATCH - 1) // BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    yv = val["label"].map(LABEL2ID).to_numpy()
    for ep in range(epochs):
        model.train()
        total = 0.0
        for enc in batches(tok, texts, labels, shuffle=True, seed=SEED + ep):
            enc = {k: v.to(dev) for k, v in enc.items()}
            loss = model(**enc).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        pv = predict_proba(model, tok, val["text"].tolist(), dev)
        val_f1 = macro_f1(yv, pv.argmax(1))
        mlflow.log_metrics({"train_loss": total, "val_macro_f1": val_f1}, step=ep + 1)
    return model, tok, val_f1


def main() -> None:
    os.chdir(ROOT)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("q1-sentiment")
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    dev = device()
    sweep = []
    for lr, epochs in GRID:
        name = f"lr{lr:g}_ep{epochs}"
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({"base": BASE, "lr": lr, "epochs": epochs, "batch": BATCH,
                               "max_len": MAX_LEN, "seed": SEED, "device": str(dev)})
            model, tok, val_f1 = train_one(lr, epochs, train, val, dev)
            out_dir = SWEEP_DIR / name
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            mlflow.log_metric("final_val_macro_f1", val_f1)
            sweep.append({"run": name, "run_id": run.info.run_id, "lr": lr, "epochs": epochs,
                          "val_macro_f1": val_f1})
            print(sweep[-1], flush=True)
            del model
            if dev.type == "mps":
                torch.mps.empty_cache()

    best = max(sweep, key=lambda r: r["val_macro_f1"])
    if SELECTED_DIR.exists():
        shutil.rmtree(SELECTED_DIR)
    shutil.copytree(SWEEP_DIR / best["run"], SELECTED_DIR)

    tok = AutoTokenizer.from_pretrained(SELECTED_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(SELECTED_DIR).to(dev)
    texts = test["text"].tolist()
    y = test["label"].map(LABEL2ID).to_numpy()
    probs = predict_proba(model, tok, texts, dev)
    out = classification_metrics(probs, y)
    out["selected"] = best
    out["latency_s_per_1k_mps"] = time_per_1k(lambda t: predict_proba(model, tok, t, dev), texts, 2)
    cpu_model = AutoModelForSequenceClassification.from_pretrained(SELECTED_DIR)
    cpu = torch.device("cpu")
    out["latency_s_per_1k_cpu_torch"] = time_per_1k(
        lambda t: predict_proba(cpu_model, tok, t, cpu), texts, 1)
    out["cost_usd_per_1k"] = 0.0
    np.save(RESULTS / "probs_finetuned.npy", probs)

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({"test_macro_f1": out["macro_f1"], "test_accuracy": out["accuracy"]})
        mlflow.pytorch.log_model(model.cpu(), name="model")
        model_uri = f"runs:/{best['run_id']}/model"
    try:
        reg = mlflow.register_model(model_uri, "q1-sentiment")
        out["mlflow_registered"] = {"name": "q1-sentiment", "version": reg.version}
    except Exception as e:  # registry problems are reported, not fatal
        out["mlflow_registered"] = {"error": str(e)[:300]}
    update_results("finetuned_distilroberta", out)
    update_results("sweep", sweep)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
