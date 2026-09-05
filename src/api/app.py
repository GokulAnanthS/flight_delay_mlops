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
import secrets
from contextlib import asynccontextmanager

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field, field_validator

from src import config, registry
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
        self.version = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        self.model = predict_module.load_model()
        tables, self.overall_fallback = predict_module.load_lookup_tables()
        self.latest = predict_module.latest_rates(tables)
        self.categorical_dtypes = predict_module.load_categorical_dtypes()
        self.version = _current_production_version()


def _current_production_version() -> str | None:
    """Which registered version the 'production' alias currently points to.

    Reads it from the registered model's alias dict (not from a
    ModelVersion object returned by search) -- MLflow doesn't populate
    `.aliases` on search results, same gotcha src/registry.py works around.
    """
    client = registry.get_client()
    registered_model = client.get_registered_model(config.MLFLOW_MODEL_NAME)
    return registered_model.aliases.get(config.MLFLOW_PRODUCTION_ALIAS)


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
    return {
        "status": "ok" if artifacts.loaded else "unavailable",
        "model_loaded": artifacts.loaded,
        "production_version": artifacts.version,
    }


def _verify_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    """Require a shared-secret `X-Admin-Token` header matching config.ADMIN_RELOAD_TOKEN.

    Fails closed: if the server has no token configured (unset env var),
    the endpoint refuses every request rather than silently accepting all
    of them. `secrets.compare_digest` avoids leaking the token's value
    through response-time differences on a byte-by-byte string comparison.
    """
    if not config.ADMIN_RELOAD_TOKEN:
        raise HTTPException(status_code=503, detail="ADMIN_RELOAD_TOKEN is not configured on the server")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, config.ADMIN_RELOAD_TOKEN):
        raise HTTPException(status_code=403, detail="Missing or invalid X-Admin-Token header")


@app.post("/admin/reload", dependencies=[Depends(_verify_admin_token)])
def reload_endpoint():
    """Re-run the exact same load the startup `lifespan` handler does.

    This is Phase 10 (CD)'s entire mechanism: the API only ever reads the
    'production' alias at process startup, so a promotion (src/registry.py)
    is invisible to an already-running process until something makes it
    reload. Calling this after a promotion is that "something" -- no image
    rebuild or restart needed, just re-reading the registry.
    """
    try:
        artifacts.load()
    except (FileNotFoundError, MlflowException) as exc:
        logger.error("Failed to reload model artifacts: %s", exc)
        raise HTTPException(status_code=503, detail=f"Reload failed: {exc}") from exc

    logger.info("Model artifacts reloaded (now serving production v%s)", artifacts.version)
    return {"status": "reloaded", "model_loaded": artifacts.loaded, "production_version": artifacts.version}


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
