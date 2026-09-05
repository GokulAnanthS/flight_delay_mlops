"""One CT (continuous training) cycle: release the next simulated month of
data, walk the train/val split forward to match, and retrain.

Each call to `run_cycle()`:
  1. Releases the earliest not-yet-released CSV from data/incoming/ into
     data/raw/csv/ (src.ct.arrivals) -- this is the "new data arrives" event.
  2. Converts it to Parquet (src.data.ingest) -- already skips previously
     converted files via its own manifest, so this is safe to call every
     cycle.
  3. Rebuilds the full feature pipeline (src.data.features) with the split
     cutoffs walked forward by however many months have arrived so far
     (src.ct.walkforward), so the new month becomes test, the old test
     rolls into val, and the old val rolls into train.
  4. Retrains (src.training.train) on the new cumulative train/val split --
     registers a new MLflow version aliased 'staging'. Promotion to
     'production' stays a separate, deliberate step (src/registry.py); CT
     never promotes on its own.

Run one cycle:
    python -m src.ct.cycle

Run until data/incoming/ is exhausted:
    python -m src.ct.cycle --until-exhausted
"""

import argparse
import logging

from src import config
from src.ct import arrivals
from src.ct.walkforward import compute_cutoffs
from src.data import features, ingest
from src.training import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_cycle(
    incoming_dir=config.INCOMING_DIR,
    raw_csv_dir=config.RAW_CSV_DIR,
    raw_parquet_dir=config.RAW_PARQUET_DIR,
    processed_dir=config.PROCESSED_DIR,
) -> dict | None:
    """Run a single release -> ingest -> feature-rebuild -> retrain cycle.

    Returns a summary dict (cycle number, released filename, split cutoffs,
    val metrics), or None if data/incoming/ had nothing left to release.
    """
    released = arrivals.release_next(incoming_dir, raw_csv_dir)
    if released is None:
        logger.info("No pending arrivals in %s -- nothing to do", incoming_dir)
        return None

    n_periods = arrivals.count_released(incoming_dir)
    train_end, val_end = compute_cutoffs(n_periods)
    logger.info(
        "Cycle %d: released %s -> train_end=%s, val_end=%s",
        n_periods, released.name, train_end, val_end,
    )

    ingest.convert_csvs_to_parquet(raw_csv_dir, raw_parquet_dir)
    features.run_pipeline(raw_parquet_dir, processed_dir, train_end, val_end)
    metrics = train.run_training()

    logger.info("Cycle %d complete. Val metrics: %s", n_periods, metrics)
    return {
        "cycle": n_periods,
        "released_file": released.name,
        "train_end": train_end,
        "val_end": val_end,
        **metrics,
    }

def run_bulk_cycle(
    incoming_dir=config.INCOMING_DIR,
    raw_csv_dir=config.RAW_CSV_DIR,
    raw_parquet_dir=config.RAW_PARQUET_DIR,
    processed_dir=config.PROCESSED_DIR,
) -> dict | None:
    """Release every pending file in data/incoming/ at once, then rebuild
    features and retrain a single time against the full resulting dataset --
    instead of one release+retrain cycle per file.

    Returns a summary dict, or None if data/incoming/ had nothing pending.
    """
    released = arrivals.release_all_pending(incoming_dir, raw_csv_dir)
    if not released:
        logger.info("No pending arrivals in %s -- nothing to do", incoming_dir)
        return None

    n_periods = arrivals.count_released(incoming_dir)
    train_end, val_end = compute_cutoffs(n_periods)
    logger.info(
        "Bulk cycle: released %d file(s) (%s) -> train_end=%s, val_end=%s",
        len(released), [f.name for f in released], train_end, val_end,
    )

    ingest.convert_csvs_to_parquet(raw_csv_dir, raw_parquet_dir)
    features.run_pipeline(raw_parquet_dir, processed_dir, train_end, val_end)
    metrics = train.run_training()

    logger.info("Bulk cycle complete. Val metrics: %s", metrics)
    return {
        "cycle": n_periods,
        "released_files": [f.name for f in released],
        "train_end": train_end,
        "val_end": val_end,
        **metrics,
    }

def run_until_exhausted(**kwargs) -> list[dict]:
    """Keep running cycles until data/incoming/ has no pending files left."""
    results = []
    while True:
        result = run_cycle(**kwargs)
        if result is None:
            break
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--until-exhausted", action="store_true",
        help="Keep releasing + retraining until data/incoming/ has nothing left",
    )
    parser.add_argument(
    "--all-at-once", action="store_true",
    help="Release every pending file in data/incoming/ and retrain once, instead of one cycle per file",
    )
    args = parser.parse_args()
    if args.until_exhausted:
        results = run_until_exhausted()
        logger.info("Ran %d cycle(s)", len(results))
    elif args.all_at_once:
        run_bulk_cycle()
    else:
        run_cycle()


if __name__ == "__main__":
    main()