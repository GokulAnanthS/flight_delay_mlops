"""Simulated data-arrival mechanism for CT.

data/incoming/ holds the user-downloaded post-2021 monthly BTS CSVs,
untouched. release_next() copies the earliest not-yet-released file into
data/raw/csv/ -- one call is one simulated "a month of new data arrives"
event. A manifest (data/incoming/_released_files.json) tracks release order
and count, independent of ingest.py's own conversion manifest -- that one
tracks CSV->Parquet conversion, a separate concern from "has this month been
simulated as arrived yet".
"""

import argparse
import glob
import json
import logging
import shutil
from pathlib import Path

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "_released_files.json"


def _load_manifest(incoming_dir: Path) -> list[str]:
    manifest_path = Path(incoming_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        return json.load(f)["released"]


def _save_manifest(incoming_dir: Path, released: list[str]) -> None:
    manifest_path = Path(incoming_dir) / MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        json.dump({"released": released}, f, indent=2)


def list_pending_arrivals(incoming_dir: Path = config.INCOMING_DIR) -> list[Path]:
    """Files in `incoming_dir` not yet released, oldest-by-name first."""
    released = set(_load_manifest(incoming_dir))
    all_files = sorted(glob.glob(str(Path(incoming_dir) / "*.csv")))
    return [Path(f) for f in all_files if Path(f).name not in released]


def count_released(incoming_dir: Path = config.INCOMING_DIR) -> int:
    return len(_load_manifest(incoming_dir))
def release_all_pending(
    incoming_dir: Path = config.INCOMING_DIR, raw_csv_dir: Path = config.RAW_CSV_DIR
) -> list[Path]:
    """Copy every not-yet-released CSV into `raw_csv_dir` and record them all as released.

    Same idempotency rules as `release_next`: a file already in the manifest
    is skipped even if it's still sitting in `incoming_dir`. Returns the list
    of files released by this call (empty if nothing was pending).
    """
    pending = list_pending_arrivals(incoming_dir)
    if not pending:
        return []

    Path(raw_csv_dir).mkdir(parents=True, exist_ok=True)
    released = _load_manifest(incoming_dir)
    for file in pending:
        shutil.copy2(file, Path(raw_csv_dir) / file.name)
        released.append(file.name)

    _save_manifest(incoming_dir, released)
    logger.info("Released %d file(s) into %s (%d released so far)", len(pending), raw_csv_dir, len(released))
    return pending

def release_next(
    incoming_dir: Path = config.INCOMING_DIR, raw_csv_dir: Path = config.RAW_CSV_DIR
) -> Path | None:
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
    Path(raw_csv_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(next_file, Path(raw_csv_dir) / next_file.name)

    released = _load_manifest(incoming_dir)
    released.append(next_file.name)
    _save_manifest(incoming_dir, released)

    logger.info("Released %s into %s (%d released so far)", next_file.name, raw_csv_dir, len(released))
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