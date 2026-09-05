"""Walk-forward split cutoffs for CT.

Starting from the baseline split (config.TRAIN_END / config.VAL_END -- the
fixed 2021-01-01 / 2021-07-01 cutoffs the model was originally trained and
evaluated against), advancing by `n_periods` months means:

  - train_end moves forward n months -> train grows to swallow what used to
    be the front of val (cumulative training, not a rolling window).
  - val_end moves forward n months by the same amount -> val's width stays
    constant, but it now covers what used to be the front of test.
  - whatever is newer than the new val_end is the new test slice -- the
    freshly-arrived month(s), not yet seen by training or validation.

Shifting both cutoffs by the same amount gives the "previous test rolls
into val, previous val rolls into train" behavior from a single integer,
without hand-tracking three separate moving windows.
"""

import pandas as pd

from src import config


def compute_cutoffs(
    n_periods: int,
    base_train_end: str = config.TRAIN_END,
    base_val_end: str = config.VAL_END,
) -> tuple[str, str]:
    """Return (train_end, val_end) walked forward by `n_periods` months.

    n_periods=0 returns the baseline cutoffs unchanged -- the state before
    any simulated month has arrived.
    """
    if n_periods < 0:
        raise ValueError(f"n_periods must be >= 0, got {n_periods}")

    offset = pd.DateOffset(months=n_periods)
    train_end = (pd.Timestamp(base_train_end) + offset).strftime("%Y-%m-%d")
    val_end = (pd.Timestamp(base_val_end) + offset).strftime("%Y-%m-%d")
    return train_end, val_end