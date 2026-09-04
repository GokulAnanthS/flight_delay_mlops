"""Train the production XGBoost model and track the run with MLflow.

Ports the winning configuration from notebooks/03_modeling.ipynb: tuned
XGBoost hyperparameters (from the RandomizedSearchCV run), trained with
2020 excluded (see config.EXCLUDE_YEARS_FROM_TRAINING for why) and
scale_pos_weight to correct for the ~5:1 class imbalance.
"""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from mlflow import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_training_data(
    train_path: Path = config.TRAIN_PATH, val_path: Path = config.VAL_PATH
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)

    train = train[~train["FL_DATE"].dt.year.isin(config.EXCLUDE_YEARS_FROM_TRAINING)]

    X_train = train[config.FEATURE_COLS]
    y_train = train[config.TARGET_COL]
    X_val = val[config.FEATURE_COLS]
    y_val = val[config.TARGET_COL]
    return X_train, y_train, X_val, y_val


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info("scale_pos_weight: %.3f", scale_pos_weight)

    model = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        random_state=config.RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        **config.XGB_TUNED_PARAMS,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: xgb.XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_val, proba),
        "pr_auc": average_precision_score(y_val, proba),
        "recall": recall_score(y_val, pred),
        "precision": precision_score(y_val, pred),
    }


def run_training(mlflow_tracking_uri: str = config.MLFLOW_TRACKING_URI) -> dict:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    X_train, y_train, X_val, y_val = load_training_data()
    logger.info("Training rows (2020 excluded): %s, val rows: %s", f"{len(X_train):,}", f"{len(X_val):,}")

    with mlflow.start_run():
        mlflow.log_params(config.XGB_TUNED_PARAMS)
        mlflow.log_param("exclude_years_from_training", config.EXCLUDE_YEARS_FROM_TRAINING)
        mlflow.log_param("feature_cols", config.FEATURE_COLS)

        model = train_model(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)
        logger.info("Val metrics: %s", metrics)
        mlflow.log_metrics(metrics)

        model_info = mlflow.xgboost.log_model(model, name="model", registered_model_name=config.MLFLOW_MODEL_NAME)
        version = model_info.registered_model_version
        MlflowClient().set_registered_model_alias(config.MLFLOW_MODEL_NAME, config.MLFLOW_STAGING_ALIAS, version)
        logger.info(
            "Registered %s v%s, aliased '%s'. Promote with: python -m src.registry promote %s",
            config.MLFLOW_MODEL_NAME, version, config.MLFLOW_STAGING_ALIAS, version,
        )

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.get_booster().save_model(str(config.LOCAL_MODEL_PATH))
        mlflow.log_artifact(str(config.LOCAL_MODEL_PATH))
        logger.info("Saved local model copy to %s", config.LOCAL_MODEL_PATH)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlflow-tracking-uri", default=config.MLFLOW_TRACKING_URI)
    args = parser.parse_args()
    run_training(args.mlflow_tracking_uri)


if __name__ == "__main__":
    main()