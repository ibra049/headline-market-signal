import pytest
from fastapi.testclient import TestClient

from serve.app import create_app


class FakeClassifier:
    version = "test-v1"

    def predict(self, texts):
        return [
            {"label": "neutral", "probs": {"negative": 0.1, "neutral": 0.8, "positive": 0.1}}
            for _ in texts
        ]


@pytest.fixture
def client():
    with TestClient(create_app(lambda _: FakeClassifier())) as client:
        yield client


def test_health_and_predictions(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test-v1"}
    response = client.post("/predict", json={"texts": ["Revenue is steady", "Profit is flat"]})
    assert response.status_code == 200
    assert response.json() == {
        "model_version": "test-v1",
        "predictions": FakeClassifier().predict(["Revenue is steady", "Profit is flat"]),
    }


@pytest.mark.parametrize(
    "body",
    [
        {"texts": []},
        {"texts": ["x"] * 257},
        {"texts": ["x" * 1001]},
        {"texts": [123]},
        {"texts": [None]},
        {"texts": "not a list"},
        {},
    ],
)
def test_invalid_requests(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_request_boundaries(client):
    response = client.post("/predict", json={"texts": ["é" * 1000] * 256})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 256


@pytest.mark.parametrize("model_dir", [None, "/tmp/custom-model"])
def test_load_once_at_startup(monkeypatch, model_dir):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    if model_dir is not None:
        monkeypatch.setenv("MODEL_DIR", model_dir)
    loaded = []

    def factory(path):
        loaded.append(path)
        return FakeClassifier()

    application = create_app(factory)
    assert loaded == []
    with TestClient(application) as client:
        client.get("/health")
        client.get("/health")
        client.post("/predict", json={"texts": ["test"]})
        assert loaded == [model_dir or "artifacts/champion"]


def test_startup_failure_is_not_healthy():
    def factory(_):
        raise FileNotFoundError("missing model")

    with pytest.raises(FileNotFoundError, match="missing model"):
        with TestClient(create_app(factory)):
            pass
