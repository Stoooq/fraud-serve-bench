use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use ort::session::Session;
use ort::value::TensorRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Deserialize)]
struct PredictRequest {
    features: HashMap<String, f32>,
}

#[derive(Serialize)]
struct PredictResponse {
    is_fraud: bool,
    proba: f32,
    predict_time: f64,
}

#[derive(Clone)]
struct AppState {
    session: Arc<Mutex<Session>>,
    threshold: f32,
    feature_names: Arc<Vec<String>>,
}

async fn health() -> &'static str {
    "OK"
}

async fn predict(
    State(state): State<AppState>,
    Json(body): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, String)> {
    let mut values = Vec::with_capacity(state.feature_names.len());
    for name in state.feature_names.iter() {
        match body.features.get(name) {
            Some(v) => values.push(*v),
            None => {
                return Err((
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("Missing feature: {name}"),
                ));
            }
        }
    }

    let input = ndarray::Array2::<f32>::from_shape_vec((1, state.feature_names.len()), values)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let start = Instant::now();
    let proba = {
        let mut sess = state.session.lock().unwrap();
        let outputs = sess
            .run(ort::inputs!["float_input" => TensorRef::from_array_view(&input).unwrap()])
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let probs = outputs["probabilities"]
            .try_extract_array::<f32>()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        probs[[0, 1]]
    };
    let predict_time = start.elapsed().as_secs_f64();

    Ok(Json(PredictResponse {
        is_fraud: proba >= state.threshold,
        proba,
        predict_time,
    }))
}

#[tokio::main]
async fn main() {
    let session = Session::builder()
        .unwrap()
        .commit_from_file("../artifacts/model.onnx")
        .unwrap();

    let raw = std::fs::read_to_string("../artifacts/metadata.json").unwrap();
    let meta: serde_json::Value = serde_json::from_str(&raw).unwrap();

    let threshold = meta["best_threshold"].as_f64().unwrap() as f32;
    let feature_names: Vec<String> = serde_json::from_value(meta["feature_names"].clone()).unwrap();

    println!(
        "Model loaded, threshold = {threshold}, features = {}",
        feature_names.len()
    );

    let state = AppState {
        session: Arc::new(Mutex::new(session)),
        threshold,
        feature_names: Arc::new(feature_names),
    };

    let app = Router::new()
        .route("/", get(health))
        .route("/predict", post(predict))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8001")
        .await
        .unwrap();
    println!("serve-rust http://127.0.0.1:8001");
    axum::serve(listener, app).await.unwrap();
}
