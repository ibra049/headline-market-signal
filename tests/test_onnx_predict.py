import json
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from src.hms import onnx_predict


@pytest.fixture
def classifier_factory(tmp_path, monkeypatch):
    tokenizer = Tokenizer(models.WordLevel(
        {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "bad": 4, "okay": 5, "good": 6},
        unk_token="<unk>",
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 2), ("</s>", 3)],
    )
    # Inference must replace serialized fixed padding with dynamic padding.
    tokenizer.enable_padding(pad_id=1, pad_token="<pad>", length=256, pad_to_multiple_of=64)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "pad_token": {"content": "<pad>"},
    }))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "version": "unit-test", "macro_f1": 0.8, "labels": list(onnx_predict.LABELS),
    }))

    class FakeSession:
        def __init__(self, path, providers):
            assert path == str(tmp_path / "model.onnx")
            assert providers == ["CPUExecutionProvider"]
            self.feeds = []

        def get_inputs(self):
            return inputs

        def get_outputs(self):
            return [SimpleNamespace(name="logits")]

        def run(self, outputs, feed):
            assert outputs == ["logits"]
            self.feeds.append(feed)
            ids = feed["input_ids"]
            logits = np.full((len(ids), 3), -1000.0, dtype=np.float32)
            logits[np.arange(len(ids)), np.clip(ids[:, 1] - 4, 0, 2)] = 1000.0
            return [logits]

    inputs = []
    monkeypatch.setattr(onnx_predict.ort, "InferenceSession", FakeSession)

    def factory(names=("input_ids", "attention_mask"), dtype="tensor(int64)"):
        inputs[:] = [SimpleNamespace(name=name, type=dtype) for name in names]
        return onnx_predict.OnnxClassifier(tmp_path)

    return factory, tmp_path


def test_probabilities_labels_order_and_stable_softmax(classifier_factory):
    factory, _ = classifier_factory
    classifier = factory()
    assert classifier.version == "unit-test"
    probabilities = classifier.predict_proba(["bad", "okay", "good"])
    assert probabilities.shape == (3, 3)
    np.testing.assert_allclose(probabilities, np.eye(3))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    predictions = classifier.predict(["bad", "okay", "good"])
    assert [p["label"] for p in predictions] == list(onnx_predict.LABELS)
    assert predictions[0]["probs"] == {"negative": 1.0, "neutral": 0.0, "positive": 0.0}
    # All values are ordinary JSON numbers.
    json.dumps(predictions, allow_nan=False)


def test_batching_padding_truncation_and_input_filtering(classifier_factory):
    factory, _ = classifier_factory
    classifier = factory()
    texts = ["bad", "good " * 200] + ["okay"] * 30 + ["good"] * 32 + ["bad"]
    probabilities = classifier.predict_proba(texts)
    assert probabilities.shape == (65, 3)
    feeds = classifier.session.feeds
    assert [feed["input_ids"].shape for feed in feeds] == [(32, 128), (32, 3), (1, 3)]
    assert set(feeds[0]) == {"input_ids", "attention_mask"}
    ids, mask = feeds[0]["input_ids"], feeds[0]["attention_mask"]
    assert ids.dtype == np.int64
    assert ids[0, :3].tolist() == [2, 4, 3]
    assert ids[0, 3:].tolist() == [1] * 125
    assert mask[0, :3].tolist() == [1, 1, 1]
    assert mask[0, 3:].tolist() == [0] * 125
    assert ids[1, -1] == 3  # Preserve the end special token during truncation.
    np.testing.assert_array_equal(probabilities.argmax(axis=1), [0, 2] + [1] * 30 + [2] * 32 + [0])


def test_optional_token_types_and_int32(classifier_factory):
    factory, _ = classifier_factory
    classifier = factory(("input_ids", "attention_mask", "token_type_ids"), "tensor(int32)")
    classifier.predict_proba(["good"])
    feed = classifier.session.feeds[0]
    assert set(feed) == {"input_ids", "attention_mask", "token_type_ids"}
    assert all(value.dtype == np.int32 for value in feed.values())
    np.testing.assert_array_equal(feed["token_type_ids"], [[0, 0, 0]])


def test_empty_input_does_not_run_model(classifier_factory):
    factory, _ = classifier_factory
    classifier = factory()
    assert classifier.predict_proba([]).shape == (0, 3)
    assert classifier.predict([]) == []
    assert classifier.session.feeds == []


def test_fallback_padding_token(classifier_factory):
    factory, path = classifier_factory
    (path / "tokenizer_config.json").unlink()
    classifier = factory()
    assert classifier.tokenizer.padding["pad_id"] == 1


def test_incorrect_label_order_is_rejected(classifier_factory):
    factory, path = classifier_factory
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["labels"] = ["positive", "neutral", "negative"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata labels"):
        factory()


@pytest.mark.parametrize("field,value", [("version", ""), ("macro_f1", float("nan"))])
def test_invalid_metadata(classifier_factory, field, value):
    factory, path = classifier_factory
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=field):
        factory()


@pytest.mark.parametrize("logits", [np.zeros((1, 2)), np.array([[float("nan"), 0, 1]])])
def test_invalid_model_output(classifier_factory, monkeypatch, logits):
    factory, _ = classifier_factory
    classifier = factory()
    monkeypatch.setattr(classifier.session, "run", lambda *_: [logits])
    with pytest.raises(ValueError, match="finite logits"):
        classifier.predict_proba(["bad"])


def test_unsupported_model_input(classifier_factory):
    factory, _ = classifier_factory
    with pytest.raises(ValueError, match="unsupported ONNX input"):
        factory(("input_ids", "unknown"))
