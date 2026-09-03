"""
Central configuration for the flight-delay pipeline: paths, columns, and model settings.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_DIR = PROJECT_ROOT / "data" / "raw" / "csv"
RAW_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOOKUP_DIR = PROCESSED_DIR / "lookup_tables"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"

# --- Target --------------------------------------------------------------

TARGET_COL = "ARR_DEL15"

# --- Columns dropped during feature engineering ---------------------------

# Only known after departure/arrival, including these would leak the outcome
# straight into the features.
LEAKAGE_COLS = [
    "DEP_DELAY", "DEP_DELAY_NEW", "DEP_DEL15",
    "ARR_DELAY", "ARR_DELAY_NEW",
    "ACTUAL_ELAPSED_TIME",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]
# Always 0 once rows without a known ARR_DEL15 are dropped, zero variance.
CONSTANT_COLS = ["CANCELLED", "DIVERTED"]
# High cardinality with no stable signal at this modeling grain.
LOW_VALUE_COLS = ["TAIL_NUM", "OP_CARRIER_FL_NUM"]

# --- Historical delay-rate features ---------------------------------------

# Each entry: (name prefix, columns to group by). "" (empty tuple) means a
# single global rate with no grouping.
HISTORICAL_RATE_GROUPS = {
    "carrier": ["OP_UNIQUE_CARRIER"],
    "route": ["route"],
    "origin": ["ORIGIN"],
    "global": [],
}
# Fill order when a grouped rate is missing (e.g. a carrier's first month):
# fall back to the global rate, then to the overall training-period mean.
HISTORICAL_RATE_FALLBACK_COL = "global_hist_delay_rate"

# --- Model feature set -----------------------------------------------------

CATEGORICAL_COLS = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "ORIGIN_STATE_ABR", "DEST_STATE_ABR"]

FEATURE_COLS = [
    "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
    "ORIGIN_STATE_ABR", "DEST_STATE_ABR",
    "DISTANCE", "DAY_OF_WEEK", "MONTH_NUM", "DAY_OF_MONTH", "IS_WEEKEND",
    "carrier_hist_delay_rate", "route_hist_delay_rate", "origin_hist_delay_rate",
]

# --- Train/val/test split ---

TRAIN_END = "2021-01-01"   # train: [start, TRAIN_END)
VAL_END = "2021-07-01"     # val:   [TRAIN_END, VAL_END), test: [VAL_END, end]

# --- Training -----------------------------------------------------------

# 2020 (COVID) is excluded from training: flight volume collapsed and delay
# dynamics were driven by pandemic disruption rather than the normal
# weather/congestion/scheduling patterns the model needs to generalize on.
# Measured, not assumed, see notebooks/03_modeling.ipynb: excluding 2020
# raised val ROC-AUC from 0.5627 to 0.6001 with otherwise identical settings.
EXCLUDE_YEARS_FROM_TRAINING = [2020]

# Best params from the RandomizedSearchCV run in notebooks/03_modeling.ipynb
# (15-iteration search, scored on val ROC-AUC with 2020 excluded from training).
XGB_TUNED_PARAMS = {
    "colsample_bytree": 0.8430179407605753,
    "learning_rate": 0.06774675463244163,
    "max_depth": 5,
    "min_child_weight": 4,
    "n_estimators": 238,
    "reg_alpha": 0.9656320330745594,
    "reg_lambda": 2.1167946962329225,
    "subsample": 0.7218455076693483,
}

RANDOM_STATE = 42

# MLflow experiment and tracking configuration.
# Stores run metadata in a local SQLite database shared across all entry points.
MLFLOW_EXPERIMENT_NAME = "flight-delay-prediction"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT.as_posix()}/mlflow.db"

# --- Model registry ---------------------------------------------------------

# MLflow Model Registry configuration for managing model versions and deployment aliases.
# Staging is assigned automatically; production requires an explicit promotion step.
MLFLOW_MODEL_NAME = "flight-delay-xgb"
MLFLOW_STAGING_ALIAS = "staging"
MLFLOW_PRODUCTION_ALIAS = "production"


# Local copy of the trained XGBoost model for quick inspection or backup.
# Prediction and API inference load the model from the MLflow production registry instead.
LOCAL_MODEL_PATH = MODELS_DIR / "xgb_model.json"

# --- Batch inference -------------------------------------------------------

# Minimum columns a new-flight record needs for scoring. DAY_OF_WEEK,
# MONTH_NUM, DAY_OF_MONTH, and route are all derived from FL_DATE/ORIGIN/DEST
# at scoring time rather than required as input.
INFERENCE_REQUIRED_COLS = [
    "FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
    "ORIGIN_STATE_ABR", "DEST_STATE_ABR", "DISTANCE",
]

PREDICTION_THRESHOLD = 0.5