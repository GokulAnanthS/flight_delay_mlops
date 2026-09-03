import numpy as np
import pandas as pd
import pytest

from src.data import features


def test_build_expanding_rate_uses_only_strictly_prior_months():
    df = pd.DataFrame({
        "g": ["A", "A", "A", "A", "B", "B"],
        "YEAR": [2018] * 6,
        "MONTH_NUM": [1, 1, 2, 3, 1, 2],
        "ARR_DEL15": [1, 0, 1, 0, 0, 1],
    })

    result = features.build_expanding_rate(df, ["g"]).set_index(["g", "MONTH_NUM"])

    # First month for each group has no prior history at all -> NaN, not 0.
    assert np.isnan(result.loc[("A", 1), "expanding_rate"])
    assert np.isnan(result.loc[("B", 1), "expanding_rate"])

    # A's month 2 rate reflects only month 1 (1 delayed / 2 flights).
    assert result.loc[("A", 2), "expanding_rate"] == pytest.approx(0.5)
    assert result.loc[("A", 2), "cum_flights"] == 2

    # A's month 3 rate reflects months 1+2 combined (2 delayed / 3 flights),
    # and must NOT include month 3's own outcome.
    assert result.loc[("A", 3), "expanding_rate"] == pytest.approx(2 / 3)
    assert result.loc[("A", 3), "cum_flights"] == 3

    # B's month 2 rate reflects only month 1 (0 delayed / 1 flight).
    assert result.loc[("B", 2), "expanding_rate"] == pytest.approx(0.0)


def test_build_expanding_rate_global_ignores_grouping():
    df = pd.DataFrame({
        "YEAR": [2018] * 4,
        "MONTH_NUM": [1, 1, 2, 2],
        "ARR_DEL15": [1, 1, 0, 0],
    })
    result = features.build_expanding_rate(df, []).set_index("MONTH_NUM")
    assert np.isnan(result.loc[1, "expanding_rate"])
    assert result.loc[2, "expanding_rate"] == pytest.approx(1.0)  # 2/2 delayed in month 1


def test_fill_historical_fallbacks_chains_grouped_then_global_then_overall():
    df = pd.DataFrame({
        "carrier_hist_delay_rate": [0.1, np.nan, np.nan],
        "route_hist_delay_rate": [0.2, 0.3, np.nan],
        "origin_hist_delay_rate": [0.4, np.nan, np.nan],
        "global_hist_delay_rate": [0.5, 0.6, np.nan],
    })
    filled = features.fill_historical_fallbacks(df, overall_fallback=0.99)

    assert filled["carrier_hist_delay_rate"].tolist() == [0.1, 0.6, 0.99]
    assert filled["route_hist_delay_rate"].tolist() == [0.2, 0.3, 0.99]
    assert filled["origin_hist_delay_rate"].tolist() == [0.4, 0.6, 0.99]
    assert filled["global_hist_delay_rate"].tolist() == [0.5, 0.6, 0.99]
    assert not filled.isnull().any().any()


def test_drop_unused_columns_removes_leakage_constant_and_low_value_cols():
    df = pd.DataFrame({col: [0] for col in features.config.LEAKAGE_COLS + features.config.CONSTANT_COLS + features.config.LOW_VALUE_COLS})
    df["DISTANCE"] = [500]

    result = features.drop_unused_columns(df)

    assert list(result.columns) == ["DISTANCE"]


def test_clean_target_drops_nulls_and_casts_to_int():
    df = pd.DataFrame({"ARR_DEL15": [1.0, 0.0, np.nan], "other": [1, 2, 3]})
    result = features.clean_target(df)

    assert len(result) == 2
    assert result["ARR_DEL15"].dtype == int
    assert result["ARR_DEL15"].tolist() == [1, 0]


def test_add_calendar_features():
    df = pd.DataFrame({
        "FL_DATE": pd.to_datetime(["2018-01-06", "2018-01-08"]),  # Sat, Mon
        "DAY_OF_WEEK": [6, 1],
        "YEAR": ["2018", "2018"],
    })
    result = features.add_calendar_features(df)

    assert result["MONTH_NUM"].tolist() == [1, 1]
    assert result["DAY_OF_MONTH"].tolist() == [6, 8]
    assert result["IS_WEEKEND"].tolist() == [1, 0]
    assert result["YEAR"].dtype == int


def test_split_train_val_test_boundaries_are_exclusive_of_next_split():
    df = pd.DataFrame({
        "FL_DATE": pd.to_datetime(["2020-12-31", "2021-01-01", "2021-06-30", "2021-07-01"]),
    })
    train, val, test = features.split_train_val_test(df, train_end="2021-01-01", val_end="2021-07-01")

    assert train["FL_DATE"].tolist() == [pd.Timestamp("2020-12-31")]
    assert val["FL_DATE"].tolist() == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-06-30")]
    assert test["FL_DATE"].tolist() == [pd.Timestamp("2021-07-01")]