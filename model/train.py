import json
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
from config import load_config
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train():
    cfg = load_config()
    df = pd.read_csv(cfg.paths.data_path)
    X, y = df.drop(columns=["Class"]), df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        stratify=y,
        random_state=cfg.training.random_state,
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    max_depth=cfg.training.max_depth,
                    n_estimators=cfg.training.n_estimators,
                    class_weight="balanced_subsample",
                    random_state=cfg.training.random_state,
                ),
            ),
        ]
    )
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    print("Train time:", train_time)

    y_proba = pipe.predict_proba(X_test)
    auc_pr = average_precision_score(y_test, y_proba[:, 1])
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba[:, 1])
    print("Auc pr:", auc_pr)

    mask = recall[:-1] >= cfg.training.recall
    if not mask.any():
        raise ValueError("Threshold not found")

    valid_idx = np.flatnonzero(mask)
    best_pos = np.argmax(precision[:-1][mask])
    best_idx = valid_idx[best_pos]
    best_threshold = thresholds[best_idx]
    print("Best threshold:", best_threshold)

    y_pred = (y_proba[:, 1] >= best_threshold).astype(int)
    con_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:", con_matrix)
    print("Precision:", precision[best_idx])
    print("Recall:", recall[best_idx])

    initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = to_onnx(
        pipe,
        initial_types=initial_type,
        target_opset=17,
        options={id(pipe): {"zipmap": False}},
    )

    onnx_str = onnx_model.SerializeToString()
    sess = ort.InferenceSession(onnx_str)
    X_test32 = X_test.to_numpy(dtype=np.float32)
    onnx_proba = sess.run(None, {"float_input": X_test32})[1]

    sk_proba = pipe.predict_proba(X_test)[:, 1]
    max_diff = np.abs(sk_proba - onnx_proba[:, 1]).max()
    print("Max diff:", max_diff)

    clf = pipe.named_steps["rf"]
    model_type = type(clf).__name__
    model_params = clf.get_params()

    metadata = dict(
        model_type=model_type,
        model_params=model_params,
        test_size=cfg.training.test_size,
        train_time=train_time,
        auc_pr=float(auc_pr),
        best_threshold=float(best_threshold),
        confusion_matrix=con_matrix.tolist(),
        precision=float(precision[best_idx]),
        recall=float(recall[best_idx]),
        feature_names=list(X_train.columns),
        n_features=len(X_train.columns),
        random_state=cfg.training.random_state,
    )

    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.artifacts_dir / "model.onnx").write_bytes(onnx_str)
    (cfg.paths.artifacts_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


if __name__ == "__main__":
    train()
