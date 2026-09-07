"""Simulated data-arrival mechanism for CT.

data/incoming/ holds the user-uploaded post-2021 monthly BTS CSVs,
untouched. release_next() copies the earliest not-yet-released file into
data/raw/csv/ -- one call is one simulated "a month of new data arrives"
event. A manifest (data/incoming/_released_files.json) tracks release order
and count, independent of ingest.py's own conversion manifest -- that one
tracks CSV->Parquet conversion, a separate concern from "has this month been
simulated as arrived yet".
"""

import argparse
import logging
from pathlib import Path

from src import config
from src.data import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "_released_files.json"


def _load_manifest(incoming_dir) -> list[str]:
    manifest = storage.read_json(storage.join(incoming_dir, MANIFEST_FILENAME))
    return manifest["released"] if manifest else []


def _save_manifest(incoming_dir, released: list[str]) -> None:
    storage.write_json(storage.join(incoming_dir, MANIFEST_FILENAME), {"released": released})


def list_pending_arrivals(incoming_dir=config.INCOMING_DIR) -> list[str]:
    """Files in `incoming_dir` not yet released, oldest-by-name first."""
    released = set(_load_manifest(incoming_dir))
    all_files = storage.list_files(incoming_dir, "*.csv")
    return [f for f in all_files if storage.basename(f) not in released]


def count_released(incoming_dir=config.INCOMING_DIR) -> int:
    return len(_load_manifest(incoming_dir))


def release_all_pending(incoming_dir=config.INCOMING_DIR, raw_csv_dir=config.RAW_CSV_DIR) -> list[str]:
    """Copy every not-yet-released CSV into `raw_csv_dir` and record them all as released.

    Same idempotency rules as `release_next`: a file already in the manifest
    is skipped even if it's still sitting in `incoming_dir`. Returns the list
    of files released by this call (empty if nothing was pending).
    """
    pending = list_pending_arrivals(incoming_dir)
    if not pending:
        return []

    storage.ensure_dir(raw_csv_dir)
    released = _load_manifest(incoming_dir)
    for file in pending:
        storage.copy_file(file, storage.join(raw_csv_dir, storage.basename(file)))
        released.append(storage.basename(file))

    _save_manifest(incoming_dir, released)
    logger.info("Released %d file(s) into %s (%d released so far)", len(pending), raw_csv_dir, len(released))
    return pending


def release_next(incoming_dir=config.INCOMING_DIR, raw_csv_dir=config.RAW_CSV_DIR) -> str | None:
    """Copy the earliest not-yet-released CSV into `raw_csv_dir` and record it as released.

    Returns the source path that was released, or None if nothing is pending.
    Idempotent per file: a file already recorded in the manifest is never
    released again, even though it's still sitting untouched in `incoming_dir`
    (files are copied, not moved, so re-running ingest/features against
    raw_csv_dir stays reproducible).
    """
    pending = list_pending_arrivals(incoming_dir)
    if not pending:
        return None

    next_file = pending[0]
    storage.ensure_dir(raw_csv_dir)
    storage.copy_file(next_file, storage.join(raw_csv_dir, storage.basename(next_file)))

    released = _load_manifest(incoming_dir)
    released.append(storage.basename(next_file))
    _save_manifest(incoming_dir, released)

    logger.info("Released %s into %s (%d released so far)", storage.basename(next_file), raw_csv_dir, len(released))
    return next_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming-dir", type=Path, default=config.INCOMING_DIR)
    parser.add_argument("--raw-csv-dir", type=Path, default=config.RAW_CSV_DIR)
    args = parser.parse_args()
    result = release_next(args.incoming_dir, args.raw_csv_dir)
    if result is None:
        logger.info("Nothing pending in %s", args.incoming_dir)


if __name__ == "__main__":
    main()
