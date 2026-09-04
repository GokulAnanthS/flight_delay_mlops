"""On-demand flight delay scoring API.

The HTTP counterpart to src/inference/predict.py's batch CLI. Both call the
exact same `predict_module.predict()` function on the same loaded model and
lookup tables, so serving-time behavior can never drift from batch-scoring
behavior — there is only one place the feature-building/prediction logic
lives.

Run locally with:
    uvicorn src.api.app:app --reload
"""

import argparse
import logging
from contextlib import asynccontextmanager

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field, field_validator

from src.inference import predict as predict_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class FlightRequest(BaseModel):
    FL_DATE: str = Field(..., description="Scheduled flight date, e.g. '2026-09-01'")
    OP_UNIQUE_CARRIER: str
    ORIGIN: str
    DEST: str
    ORIGIN_STATE_ABR: str
    DEST_STATE_ABR: str
    DISTANCE: float = Field(..., gt=0)

    @field_validator("FL_DATE")
    @classmethod
    def _parseable_date(cls, v: str) -> str:
        try:
            pd.to_datetime(v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"FL_DATE must be a parseable date, got {v!r}") from exc
        return v


class PredictRequest(BaseModel):
    flights: list[FlightRequest]


class FlightPrediction(BaseModel):
    delay_probability: float
    predicted_delay: int


class PredictResponse(BaseModel):
    predictions: list[FlightPrediction]


class ModelArtifacts:
    """Holds the model + lookup tables loaded once at startup.

    Loading these (especially the model) per-request would make every
    prediction pay the deserialization cost; loading them once and reusing
    across requests is the entire point of a long-running service versus the
    batch CLI.
    """

    def __init__(self):
        self.model = None
        self.latest = None
        self.overall_fallback = None
        self.categorical_dtypes = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        self.model = predict_module.load_model()
        tables, self.overall_fallback = predict_module.load_lookup_tables()
        self.latest = predict_module.latest_rates(tables)
        self.categorical_dtypes = predict_module.load_categorical_dtypes()


artifacts = ModelArtifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        artifacts.load()
        logger.info("Model artifacts loaded")
    except (FileNotFoundError, MlflowException) as exc:
        # Covers a missing local lookup/train file as well as MLflow registry
        # failures (no registered model yet, or nothing aliased 'production').
        # Let the app come up anyway so /health reports the problem instead of
        # the process crash-looping with no diagnosable reason in a container.
        logger.error("Failed to load model artifacts at startup: %s", exc)
    yield


app = FastAPI(title="Flight Delay Prediction API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if artifacts.loaded else "unavailable", "model_loaded": artifacts.loaded}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    if not artifacts.loaded:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")
    if not request.flights:
        raise HTTPException(status_code=400, detail="flights list must not be empty")

    df_raw = pd.DataFrame([f.model_dump() for f in request.flights])
    result = predict_module.predict(
        df_raw, artifacts.model, artifacts.latest, artifacts.overall_fallback, artifacts.categorical_dtypes
    )

    return PredictResponse(
        predictions=[
            FlightPrediction(delay_probability=float(row.delay_probability), predicted_delay=int(row.predicted_delay))
            for row in result.itertuples()
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()