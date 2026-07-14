from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: dict[str, float]

class PredictResponse(BaseModel):
    is_fraud: bool
    proba: float
    predict_time: float

class BatchPredictRequest(BaseModel):
    rows: list[list[float]]

class BatchPredictResponse(BaseModel):
    is_fraud: list[bool]
    proba: list[float]
    predict_time: float