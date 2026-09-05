"""Feature engineering pipeline for flight delay prediction.

Ports notebooks/02_feature_engineering.ipynb into reusable, testable stages:

    load_raw -> clean_target -> drop_unused_columns -> add_calendar_features
             -> compute_historical_rate_tables -> merge_historical_features
             -> fill_historical_fallbacks -> select_model_frame -> split_train_val_test

`run_pipeline()` chains all of these and persists both the model ready
train/val/test splits and the historical rate lookup tables. The lookup
tables are saved separately (not just baked into the training split) because
a later inference/serving phase needs the same "delay rate as of month X"
values to score new flights, they can't be recomputed from a single new
row the way the calendar features can.
"""
 
import argparse
import logging
from pathlib import Path

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_raw(data_dir: Path = config.RAW_PARQUET_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir)


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no known outcome (cancelled/diverted flights) and cast the target to int."""
    df = df[df[config.TARGET_COL].notnull()].copy()
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = config.LEAKAGE_COLS + config.CONSTANT_COLS + config.LOW_VALUE_COLS
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MONTH_NUM"] = df["FL_DATE"].dt.month
    df["DAY_OF_MONTH"] = df["FL_DATE"].dt.day
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)
    df["YEAR"] = df["YEAR"].astype(int)
    return df


def build_expanding_rate(df_source: pd.DataFrame, group_cols: list[str], value_col: str = config.TARGET_COL) -> pd.DataFrame:
    """Delay rate for each group using only months strictly before (YEAR, MONTH_NUM).

    Never includes the current or future months, so it's safe to use as a
    feature for every row in that group's current month. `group_cols=[]`
    computes a single global (ungrouped) expanding rate.
    """
    all_cols = group_cols + ["YEAR", "MONTH_NUM"]
    monthly_group = (
        df_source.groupby(all_cols, observed=True)
        .agg(flights=(value_col, "size"), delayed=(value_col, "sum"))
        .reset_index()
        .sort_values(all_cols)
    )

    if group_cols:
        cum_flights_incl = monthly_group.groupby(group_cols, observed=True)["flights"].cumsum()
        cum_delayed_incl = monthly_group.groupby(group_cols, observed=True)["delayed"].cumsum()
    else:
        cum_flights_incl = monthly_group["flights"].cumsum()
        cum_delayed_incl = monthly_group["delayed"].cumsum()

    monthly_group["cum_flights"] = cum_flights_incl - monthly_group["flights"]
    monthly_group["cum_delayed"] = cum_delayed_incl - monthly_group["delayed"]
    monthly_group["expanding_rate"] = monthly_group["cum_delayed"] / monthly_group["cum_flights"]

    return monthly_group[group_cols + ["YEAR", "MONTH_NUM", "expanding_rate", "cum_flights"]]


def compute_historical_rate_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build the per entity expanding delay-rate lookup tables.

    Returns the (possibly augmented with a `route` column) dataframe alongside
    a {name: lookup_table} dict, one entry per group in config.HISTORICAL_RATE_GROUPS.
    """
    df = df.copy()
    if "route" not in df.columns:
        df["route"] = df["ORIGIN"].astype(str) + "-" + df["DEST"].astype(str)

    tables = {}
    for name, group_cols in config.HISTORICAL_RATE_GROUPS.items():
        rate_col = f"{name}_hist_delay_rate"
        flights_col = f"{name}_hist_flights"
        table = build_expanding_rate(df, group_cols).rename(
            columns={"expanding_rate": rate_col, "cum_flights": flights_col}
        )
        tables[name] = table

    return df, tables


def merge_historical_features(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = df.copy()
    for name, group_cols in config.HISTORICAL_RATE_GROUPS.items():
        merge_keys = group_cols + ["YEAR", "MONTH_NUM"]
        df = df.merge(tables[name], on=merge_keys, how="left")
    return df


def fill_historical_fallbacks(df: pd.DataFrame, overall_fallback: float) -> pd.DataFrame:
    """Fill missing historical rates: grouped rate -> global rate -> overall training mean.

    Rows land here when an entity (carrier/route/origin) has no prior months
    at all, most commonly the very first month in the dataset, where nothing
    has any history yet, including the global rate.
    """
    df = df.copy()
    rate_cols = [f"{name}_hist_delay_rate" for name in config.HISTORICAL_RATE_GROUPS]
    grouped_rate_cols = [c for c in rate_cols if c != config.HISTORICAL_RATE_FALLBACK_COL]

    for col in grouped_rate_cols:
        df[col] = df[col].fillna(df[config.HISTORICAL_RATE_FALLBACK_COL])
    for col in rate_cols:
        df[col] = df[col].fillna(overall_fallback)
    return df


def select_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in config.CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df[config.FEATURE_COLS + [config.TARGET_COL, "FL_DATE"]]


def split_train_val_test(
    df: pd.DataFrame, train_end: str = config.TRAIN_END, val_end: str = config.VAL_END
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["FL_DATE"] < train_end]
    val = df[(df["FL_DATE"] >= train_end) & (df["FL_DATE"] < val_end)]
    test = df[df["FL_DATE"] >= val_end]
    return train, val, test


def save_processed(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, out_dir: Path = config.PROCESSED_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet")
    val.to_parquet(out_dir / "val.parquet")
    test.to_parquet(out_dir / "test.parquet")
    logger.info("Saved train/val/test to %s", out_dir)


def save_lookup_tables(
    tables: dict[str, pd.DataFrame], overall_fallback: float, out_dir: Path = config.LOOKUP_DIR
) -> None:
    """Persist the historical rate lookup tables an inference stage will need later.

    A future scoring request has to look up "the most recent known delay rate
    for this carrier/route/origin" the same way training did, these tables
    are that record, plus the scalar fallback for entities with no history.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}_hist.parquet")
    pd.DataFrame([{"overall_fallback": overall_fallback}]).to_parquet(out_dir / "overall_fallback.parquet")
    logger.info("Saved %d lookup tables to %s", len(tables), out_dir)


def run_pipeline(
    raw_dir: Path = config.RAW_PARQUET_DIR,
    out_dir: Path = config.PROCESSED_DIR,
    train_end: str = config.TRAIN_END,
    val_end: str = config.VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading raw data from %s", raw_dir)
    df = load_raw(raw_dir)
    logger.info("%s rows loaded", f"{len(df):,}")

    df = clean_target(df)
    df = drop_unused_columns(df)
    df = add_calendar_features(df)
    df, tables = compute_historical_rate_tables(df)
    df = merge_historical_features(df, tables)

    # Computed from the training period only: a production system can't use
    # test-period outcomes to fill in training-row features. Uses the
    # caller's train_end (not config.TRAIN_END) so a CT cycle with walked-
    # forward cutoffs computes the fallback from ITS training window.
    overall_fallback = df.loc[df["FL_DATE"] < train_end, config.TARGET_COL].mean()
    df = fill_historical_fallbacks(df, overall_fallback)

    model_df = select_model_frame(df)
    train, val, test = split_train_val_test(model_df, train_end, val_end)
    logger.info(
        "Train: %s (%s -> %s)", f"{len(train):,}", train["FL_DATE"].min(), train["FL_DATE"].max()
    )
    logger.info("Val:   %s (%s -> %s)", f"{len(val):,}", val["FL_DATE"].min(), val["FL_DATE"].max())
    logger.info("Test:  %s (%s -> %s)", f"{len(test):,}", test["FL_DATE"].min(), test["FL_DATE"].max())

    save_processed(train, val, test, out_dir)
    save_lookup_tables(tables, overall_fallback)
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_PARQUET_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--train-end", default=config.TRAIN_END)
    parser.add_argument("--val-end", default=config.VAL_END)
    args = parser.parse_args()
    run_pipeline(args.raw_dir, args.out_dir, args.train_end, args.val_end)


if __name__ == "__main__":
    main()