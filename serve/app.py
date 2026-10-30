"""Run with uvicorn serve.app:app --host 0.0.0.0 --port 7860."""

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, StrictStr, StringConstraints

from src.hms.onnx_predict import OnnxClassifier

Text = Annotated[StrictStr, StringConstraints(max_length=1000)]


class PredictRequest(BaseModel):
    texts: Annotated[list[Text], Field(min_length=1, max_length=256)]


def create_app(classifier_factory: Callable = OnnxClassifier) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier_factory(os.environ.get("MODEL_DIR", "artifacts/champion"))
        yield
        del app.state.classifier

    application = FastAPI(lifespan=lifespan)

    @application.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_version": request.app.state.classifier.version}

    @application.post("/predict")
    def predict(body: PredictRequest, request: Request):
        classifier = request.app.state.classifier
        return {"model_version": classifier.version, "predictions": classifier.predict(body.texts)}

    return application


app = create_app()
