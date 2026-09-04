"""Batch scoring for new flights: Phase 2 of the production pipeline.

Loads the model trained by src/training/train.py and the historical-rate
lookup tables persisted by src/data/features.py, builds the same feature set
used in training for a batch of new (undeparted) flights, and writes out a
delay probability + prediction per row.

Two things a training-time pipeline doesn't have to deal with, that scoring
does:

1. Historical delay-rate features assumed a KNOWN (YEAR, MONTH_NUM) to join
   against. A flight being scored hasn't happened yet, so instead of an exact
   month match we use each entity's most recent known rate (see
   `latest_rates`), the best information actually available at scoring time.
2. XGBoost's `enable_categorical` encodes categories as integer codes with no
   memory of what the training-time category-to-code mapping was. If new data
   is cast to "category" independently, codes can silently misalign with what
   the model was trained on and produce wrong (not even error-raising)
   predictions. `load_categorical_dtypes` reads the exact training-time
   category sets out of train.parquet (parquet preserves the categorical
   dtype's category list, in order) so scoring casts new data the same way.

The model itself is loaded from the MLflow model registry's 'production'
alias (see src/registry.py), not a hardcoded local file, whichever version
was last explicitly promoted is what scoring uses.
"""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_uri: str | None = None) -> xgb.XGBClassifier:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model_uri = model_uri or f"models:/{config.MLFLOW_MODEL_NAME}@{config.MLFLOW_PRODUCTION_ALIAS}"
    return mlflow.xgboost.load_model(model_uri)


def load_lookup_tables(lookup_dir: Path = config.LOOKUP_DIR) -> tuple[dict[str, pd.DataFrame], float]:
    tables = {name: pd.read_parquet(Path(lookup_dir) / f"{name}_hist.parquet") for name in config.HISTORICAL_RATE_GROUPS}
    overall_fallback = pd.read_parquet(Path(lookup_dir) / "overall_fallback.parquet")["overall_fallback"].iloc[0]
    return tables, overall_fallback


def load_categorical_dtypes(train_path: Path = config.TRAIN_PATH) -> dict[str, pd.CategoricalDtype]:
    sample = pd.read_parquet(train_path, columns=config.CATEGORICAL_COLS)
    return {col: sample[col].dtype for col in config.CATEGORICAL_COLS}


def latest_rates(tables: dict[str, pd.DataFrame]) -> dict[str, pd.Series | float]:
    """Collapse each lookup table to the single most recent rate per entity.

    A new flight has no (YEAR, MONTH_NUM) with known history yet, so scoring
    uses "the latest rate we've observed for this carrier/route/origin" rather
    than an exact-month join. The global table has no grouping, so it
    collapses to one scalar: the latest global rate.
    """
    latest = {}
    for name, group_cols in config.HISTORICAL_RATE_GROUPS.items():
        rate_col = f"{name}_hist_delay_rate"
        table = tables[name].sort_values(["YEAR", "MONTH_NUM"])
        if group_cols:
            latest[name] = table.groupby(group_cols, observed=True)[rate_col].last()
        else:
            latest[name] = table[rate_col].iloc[-1]
    return latest


def add_calendar_features_for_scoring(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    df["MONTH_NUM"] = df["FL_DATE"].dt.month
    df["DAY_OF_MONTH"] = df["FL_DATE"].dt.day
    # BTS convention is Monday=1 ... Sunday=7; pandas dayofweek is Monday=0 ... Sunday=6.
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek + 1
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)
    df["route"] = df["ORIGIN"].astype(str) + "-" + df["DEST"].astype(str)
    return df


def attach_historical_rates(df: pd.DataFrame, latest: dict[str, pd.Series | float], overall_fallback: float) -> pd.DataFrame:
    df = df.copy()
    for name, group_cols in config.HISTORICAL_RATE_GROUPS.items():
        rate_col = f"{name}_hist_delay_rate"
        if not group_cols:
            df[rate_col] = latest[name]
        else:
            df[rate_col] = df[group_cols[0]].map(latest[name])

    grouped_rate_cols = [f"{name}_hist_delay_rate" for name in config.HISTORICAL_RATE_GROUPS if name != "global"]
    for col in grouped_rate_cols:
        df[col] = df[col].fillna(df[config.HISTORICAL_RATE_FALLBACK_COL])
    for name in config.HISTORICAL_RATE_GROUPS:
        col = f"{name}_hist_delay_rate"
        df[col] = df[col].fillna(overall_fallback)
    return df


def apply_categorical_dtypes(df: pd.DataFrame, dtypes: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
    """Cast to the exact training-time category sets. A carrier/airport unseen
    during training becomes NaN (missing) rather than a new, unmapped code —
    XGBoost handles missing categorical values natively."""
    df = df.copy()
    for col, dtype in dtypes.items():
        df[col] = df[col].astype(dtype)
    return df


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in config.INFERENCE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def build_features(
    df_raw: pd.DataFrame,
    latest: dict[str, pd.Series | float],
    overall_fallback: float,
    categorical_dtypes: dict[str, pd.CategoricalDtype],
) -> pd.DataFrame:
    validate_input(df_raw)
    df = add_calendar_features_for_scoring(df_raw)
    df = attach_historical_rates(df, latest, overall_fallback)
    df = apply_categorical_dtypes(df, categorical_dtypes)
    return df[config.FEATURE_COLS]


def predict(
    df_raw: pd.DataFrame,
    model: xgb.XGBClassifier,
    latest: dict[str, pd.Series | float],
    overall_fallback: float,
    categorical_dtypes: dict[str, pd.CategoricalDtype],
    threshold: float = config.PREDICTION_THRESHOLD,
) -> pd.DataFrame:
    X = build_features(df_raw, latest, overall_fallback, categorical_dtypes)
    proba = model.predict_proba(X)[:, 1]

    result = df_raw.copy()
    result["delay_probability"] = proba
    result["predicted_delay"] = (proba >= threshold).astype(int)
    return result


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def run_batch_prediction(
    input_path: Path,
    output_path: Path,
    model_uri: str | None = None,
    lookup_dir: Path = config.LOOKUP_DIR,
    train_path: Path = config.TRAIN_PATH,
    threshold: float = config.PREDICTION_THRESHOLD,
) -> pd.DataFrame:
    logger.info("Loading model from %s", model_uri or f"models:/{config.MLFLOW_MODEL_NAME}@{config.MLFLOW_PRODUCTION_ALIAS}")
    model = load_model(model_uri)
    tables, overall_fallback = load_lookup_tables(lookup_dir)
    latest = latest_rates(tables)
    categorical_dtypes = load_categorical_dtypes(train_path)

    df_raw = _read_table(input_path)
    logger.info("Scoring %s flights from %s", f"{len(df_raw):,}", input_path)

    result = predict(df_raw, model, latest, overall_fallback, categorical_dtypes, threshold)
    _write_table(result, output_path)
    logger.info(
        "Wrote %s predictions to %s (%.1f%% predicted delayed)",
        f"{len(result):,}", output_path, 100 * result["predicted_delay"].mean(),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV or Parquet file of new flights to score")
    parser.add_argument("--output", type=Path, required=True, help="Where to write predictions (CSV or Parquet)")
    parser.add_argument("--model-uri", default=None, help="MLflow model URI; defaults to the registry's 'production' alias")
    parser.add_argument("--lookup-dir", type=Path, default=config.LOOKUP_DIR)
    parser.add_argument("--train-path", type=Path, default=config.TRAIN_PATH)
    parser.add_argument("--threshold", type=float, default=config.PREDICTION_THRESHOLD)
    args = parser.parse_args()
    run_batch_prediction(args.input, args.output, args.model_uri, args.lookup_dir, args.train_path, args.threshold)


if __name__ == "__main__":
    main()