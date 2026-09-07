FROM python:3.10-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API needs at runtime -- not notebooks/ or tests/.
COPY src/ src/

# Scoring needs the historical-rate lookup tables and train.parquet's
# categorical dtype metadata (see src/inference/predict.py); everything
# else under data/ is gitignored and too large to ship in the image.
COPY data/processed/lookup_tables/ data/processed/lookup_tables/
COPY data/processed/train.parquet data/processed/train.parquet

EXPOSE 8000

# MLFLOW_TRACKING_URI and ADMIN_RELOAD_TOKEN are expected to be injected at
# deploy time (ECS task definition env vars) -- see config.py, which falls
# back to local sqlite / "unconfigured" if they're absent.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
