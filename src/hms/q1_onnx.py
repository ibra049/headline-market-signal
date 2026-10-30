"""Q1 optimization: export the selected model to ONNX, int8 dynamic quantization, measure."""

from __future__ import annotations

import shutil

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hms.q1_common import (
    ARTIFACTS,
    FROZEN,
    LABEL2ID,
    LABELS,
    RESULTS,
    classification_metrics,
    load_split,
    time_per_1k,
    update_results,
    write_json,
)
from hms.q1_finetune import MAX_LEN, SELECTED_DIR

ONNX_DIR = ARTIFACTS / "q1_onnx"
VERSION = "1.0.0"


def export_fp32(path) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(SELECTED_DIR).eval()
    tok = AutoTokenizer.from_pretrained(SELECTED_DIR)
    enc = tok(["an example sentence"], return_tensors="pt")
    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


class OnnxRunner:
    """Same tokenization path as serving: `tokenizers` + onnxruntime, no torch."""

    def __init__(self, model_path, tokenizer_path):
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.enable_truncation(MAX_LEN)
        pad_id = self.tok.token_to_id("<pad>")
        self.tok.enable_padding(pad_id=pad_id, pad_token="<pad>")

    def predict_proba(self, texts, batch=32) -> np.ndarray:
        out = []
        for s in range(0, len(texts), batch):
            encs = self.tok.encode_batch(texts[s : s + batch])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            logits = self.sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            out.append(e / e.sum(axis=1, keepdims=True))
        return np.concatenate(out)


def main() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    fp32, int8 = ONNX_DIR / "model_fp32.onnx", ONNX_DIR / "model_int8.onnx"
    export_fp32(fp32)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    test = load_split("test")
    texts = test["text"].tolist()
    y = test["label"].map(LABEL2ID).to_numpy()
    tok_json = SELECTED_DIR / "tokenizer.json"
    res = {}
    for name, path in [("onnx_fp32", fp32), ("onnx_int8", int8)]:
        runner = OnnxRunner(path, tok_json)
        probs = runner.predict_proba(texts)
        m = classification_metrics(probs, y)
        m["latency_s_per_1k_cpu"] = time_per_1k(runner.predict_proba, texts, repeats=2)
        m["size_mb"] = round(path.stat().st_size / 1e6, 1)
        np.save(RESULTS / f"probs_{name}.npy", probs)
        res[name] = m
        print(name, m, flush=True)
    drop = res["onnx_fp32"]["macro_f1"] - res["onnx_int8"]["macro_f1"]
    res["int8_macro_f1_drop_vs_fp32"] = drop
    res["int8_accepted"] = bool(drop <= 0.01)
    update_results("optimization", res)

    chosen = "onnx_int8" if res["int8_accepted"] else "onnx_fp32"
    chosen_path = int8 if res["int8_accepted"] else fp32
    meta = {"version": VERSION, "macro_f1": res[chosen]["macro_f1"], "labels": LABELS,
            "base": "distilroberta-base", "quantization": chosen,
            "test_sha256": (FROZEN / "q1_test.sha256").read_text().strip()}
    for target in ["champion", "candidate"]:
        d = ARTIFACTS / target
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        shutil.copy(chosen_path, d / "model.onnx")
        for f in SELECTED_DIR.iterdir():
            if f.name.startswith(("tokenizer", "special_tokens", "vocab", "merges", "config")):
                shutil.copy(f, d / f.name)
        write_json(d / "metadata.json", meta)


if __name__ == "__main__":
    main()
