"""Q1 fine-tune: distilroberta-base sweep (lr x epochs), MLflow tracking and registry."""

from __future__ import annotations

import argparse
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
from hms.runtime import training_guard

BASE = "distilroberta-base"
MAX_LEN = 128
BATCH = 8  # run 1 (lr2e-05_ep2) was trained with 16 before the out-of-memory kill
GRAD_CHECKPOINTING = True
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
        BASE, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
        # MPS SDPA kernel has no dropout support under gradient checkpointing; eager is the
        # same computation.
        attn_implementation="eager" if GRAD_CHECKPOINTING else None,
    ).to(dev)
    if GRAD_CHECKPOINTING:
        model.gradient_checkpointing_enable()
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


def run_name(lr, epochs) -> str:
    return f"lr{lr:g}_ep{epochs}"


def finished_run(client, exp_id, name):
    runs = client.search_runs(
        [exp_id], f"attributes.run_name = '{name}' and attributes.status = 'FINISHED'"
    )
    return runs[0] if runs else None


def setup():
    os.chdir(ROOT)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    exp = mlflow.set_experiment("q1-sentiment")
    return mlflow.MlflowClient(), exp.experiment_id


def train_setting(name: str) -> None:
    """Train one grid setting in this process; skip if its checkpoint and MLflow run exist."""
    client, exp_id = setup()
    lr, epochs = next((lr, ep) for lr, ep in GRID if run_name(lr, ep) == name)
    out_dir = SWEEP_DIR / name
    done = finished_run(client, exp_id, name)
    if out_dir.exists() and done is not None:
        print(f"skip {name}: checkpoint and FINISHED run {done.info.run_id} exist", flush=True)
        return
    train, val = load_split("train"), load_split("val")
    dev = device()
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({"base": BASE, "lr": lr, "epochs": epochs, "batch": BATCH,
                           "grad_checkpointing": GRAD_CHECKPOINTING, "max_len": MAX_LEN,
                           "seed": SEED, "device": str(dev)})
        model, tok, val_f1 = train_one(lr, epochs, train, val, dev)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        mlflow.log_metric("final_val_macro_f1", val_f1)
        print({"run": name, "run_id": run.info.run_id, "val_macro_f1": val_f1}, flush=True)


def select() -> None:
    """Pick the best finished run on validation macro-F1, score it once on the test set."""
    client, exp_id = setup()
    sweep = []
    for lr, epochs in GRID:
        name = run_name(lr, epochs)
        r = finished_run(client, exp_id, name)
        if r is None or not (SWEEP_DIR / name).exists():
            raise SystemExit(f"run {name} is not finished; train it first")
        sweep.append({"run": name, "run_id": r.info.run_id, "lr": lr, "epochs": epochs,
                      "batch": int(r.data.params.get("batch", 16)),
                      "grad_checkpointing": r.data.params.get("grad_checkpointing", "False"),
                      "val_macro_f1": r.data.metrics["final_val_macro_f1"]})
    best = max(sweep, key=lambda r: r["val_macro_f1"])
    if SELECTED_DIR.exists():
        shutil.rmtree(SELECTED_DIR)
    shutil.copytree(SWEEP_DIR / best["run"], SELECTED_DIR)

    test = load_split("test")
    dev = device()
    tok = AutoTokenizer.from_pretrained(SELECTED_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(SELECTED_DIR).to(dev)
    texts = test["text"].tolist()
    y = test["label"].map(LABEL2ID).to_numpy()
    probs = predict_proba(model, tok, texts, dev)
    out = classification_metrics(probs, y)
    out["selected"] = best
    out["latency_s_per_1k_mps"] = time_per_1k(lambda t: predict_proba(model, tok, t, dev), texts, 2)
    model = model.cpu()
    if dev.type == "mps":
        torch.mps.empty_cache()
    cpu = torch.device("cpu")
    out["latency_s_per_1k_cpu_torch"] = time_per_1k(
        lambda t: predict_proba(model, tok, t, cpu), texts, 1)
    out["cost_usd_per_1k"] = 0.0
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS / "probs_finetuned.npy", probs)

    # Persist the scientific result before optional artifact/registry operations.
    update_results("finetuned_distilroberta", out)
    update_results("sweep", sweep)
    client.log_metric(best["run_id"], "test_macro_f1", out["macro_f1"])
    client.log_metric(best["run_id"], "test_accuracy", out["accuracy"])
    try:
        # Export failures must not change the completed training run's status.
        with mlflow.start_run(run_name="selected-export"):
            info = mlflow.pytorch.log_model(
                model, name="model", pip_requirements=[f"torch=={torch.__version__}"],
                serialization_format="pickle",
            )
        reg = mlflow.register_model(info.model_uri, "q1-sentiment")
        out["mlflow_registered"] = {"name": "q1-sentiment", "version": reg.version}
    except Exception as e:  # registry problems are reported, not fatal
        out["mlflow_registered"] = {"error": str(e)[:300]}
    update_results("finetuned_distilroberta", out)
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", choices=[run_name(lr, ep) for lr, ep in GRID])
    g.add_argument("--select", action="store_true")
    args = ap.parse_args()
    if args.run:
        with training_guard():
            train_setting(args.run)
    else:
        select()


if __name__ == "__main__":
    main()
