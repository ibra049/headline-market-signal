"""CPU sentiment inference without torch or transformers."""

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

LABELS = ("negative", "neutral", "positive")
MAX_LENGTH = 128
BATCH_SIZE = 32


class OnnxClassifier:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        if self.metadata.get("labels") != list(LABELS):
            raise ValueError(f"metadata labels must be {list(LABELS)} in that order")
        self.version = self.metadata["version"]
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("metadata version must be a nonempty string")
        score = self.metadata["macro_f1"]
        if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("metadata macro_f1 must be finite and between 0 and 1")

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        config_path = model_dir / "tokenizer_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        padding = self.tokenizer.padding or {}
        pad_token = config.get("pad_token") or padding.get("pad_token")
        if isinstance(pad_token, dict):
            pad_token = pad_token["content"]
        if pad_token is None:
            pad_token = next(
                (t for t in ("<pad>", "[PAD]") if self.tokenizer.token_to_id(t) is not None),
                None,
            )
        pad_id = self.tokenizer.token_to_id(pad_token) if pad_token is not None else None
        if pad_id is None:
            raise ValueError("tokenizer must define a padding token")
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.tokenizer.enable_padding(
            direction=config.get("padding_side", padding.get("direction", "right")),
            pad_id=pad_id,
            pad_token=pad_token,
            pad_type_id=padding.get("pad_type_id", 0),
            length=None,
            pad_to_multiple_of=None,
        )
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.inputs = self.session.get_inputs()
        for item in self.inputs:
            if item.name not in {"input_ids", "attention_mask", "token_type_ids"}:
                raise ValueError(f"unsupported ONNX input: {item.name}")
            if item.type not in {"tensor(int64)", "tensor(int32)"}:
                raise ValueError(f"unsupported ONNX input type: {item.type}")
        outputs = [item.name for item in self.session.get_outputs()]
        self.output_name = "logits" if "logits" in outputs else outputs[0]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(LABELS)), dtype=np.float32)
        probabilities = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            encodings = self.tokenizer.encode_batch(batch)
            values = {
                "input_ids": [e.ids for e in encodings],
                "attention_mask": [e.attention_mask for e in encodings],
                "token_type_ids": [e.type_ids for e in encodings],
            }
            feed = {
                item.name: np.asarray(
                    values[item.name],
                    dtype=np.int64 if item.type == "tensor(int64)" else np.int32,
                )
                for item in self.inputs
            }
            logits = np.asarray(
                self.session.run([self.output_name], feed)[0], dtype=np.float64
            )
            if logits.shape != (len(batch), len(LABELS)) or not np.isfinite(logits).all():
                raise ValueError("ONNX output must contain finite logits with shape (batch, 3)")
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities.append((exp / exp.sum(axis=1, keepdims=True)).astype(np.float32))
        return np.concatenate(probabilities, axis=0)

    def predict(self, texts: list[str]) -> list[dict]:
        return [
            {
                "label": LABELS[int(row.argmax())],
                "probs": {label: float(prob) for label, prob in zip(LABELS, row, strict=True)},
            }
            for row in self.predict_proba(texts)
        ]
