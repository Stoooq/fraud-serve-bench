import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from app.schema import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from fastapi import FastAPI, HTTPException

DEFAULT = Path(__file__).parent.parent.parent / "artifacts"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", DEFAULT))

sess = ort.InferenceSession(str(ARTIFACTS_DIR / "model.onnx"))
meta = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
threshold = meta["best_threshold"]
feature_names = meta["feature_names"]
n_features = meta["n_features"]


def run_inference(x: np.ndarray) -> np.ndarray:
    start = time.perf_counter()
    proba = sess.run(None, {"float_input": x})[1][:, 1]
    predict_time = time.perf_counter() - start
    is_fraud = proba >= threshold

    return is_fraud, proba, predict_time


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "world"}


@app.post("/predict")
def predict(body: PredictRequest) -> PredictResponse:
    missing = set(feature_names) - set(body.features)
    extra = set(body.features) - set(feature_names)
    if missing or extra:
        raise HTTPException(
            422, f"Missing: {sorted(missing)}, unexpected: {sorted(extra)}"
        )

    x = np.array([[body.features[name] for name in feature_names]], dtype=np.float32)
    is_fraud, proba, predict_time = run_inference(x)

    return PredictResponse(
        is_fraud=bool(is_fraud[0]), proba=float(proba[0]), predict_time=predict_time
    )


@app.post("/predict/batch")
def predict_batch(body: BatchPredictRequest) -> BatchPredictResponse:
    if not body.rows:
        raise HTTPException(422, "Rows cannot be empty")
    if any(len(r) != n_features for r in body.rows):
        raise HTTPException(422, f"Each row must have {n_features} features")

    x = np.array(body.rows, dtype=np.float32)
    is_fraud, proba, predict_time = run_inference(x)

    return BatchPredictResponse(
        is_fraud=is_fraud.tolist(), proba=proba.tolist(), predict_time=predict_time
    )


@app.get("/schema")
def schema():
    return {
        "feature_names": feature_names,
        "n_features": n_features,
        "threshold": threshold,
        "model_type": meta["model_type"],
    }
