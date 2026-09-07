"""Convert raw BTS CSVs into a partitioned Parquet dataset.

Processes one CSV at a time (rather than concatenating in memory first) since
the full set of files is tens of GB combined.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import config
from src.data import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Tracks which source CSVs have already been converted, since the Parquet
# output is partitioned by YEAR/MONTH and has no other record of which
# source file each partition came from.
MANIFEST_FILENAME = "_converted_files.json"


def _load_manifest(output_dir) -> dict:
    return storage.read_json(storage.join(output_dir, MANIFEST_FILENAME)) or {}


def _save_manifest(output_dir, manifest: dict) -> None:
    storage.write_json(storage.join(output_dir, MANIFEST_FILENAME), manifest)


def convert_csvs_to_parquet(input_dir=config.RAW_CSV_DIR, output_dir=config.RAW_PARQUET_DIR) -> int:
    """Convert every not yet converted CSV in `input_dir` into `output_dir`, partitioned by YEAR/MONTH.
    A file is considered already converted if its name and size match an entry
    recorded in `output_dir/_converted_files.json` from a previous run.
    Returns the number of files converted (skipped files don't count).
    """
    storage.ensure_dir(output_dir)
    manifest = _load_manifest(output_dir)

    all_files = storage.list_files(input_dir, "*.csv")
    files = []
    for file in all_files:
        size = storage.get_size(file)
        entry = manifest.get(storage.basename(file))
        if entry is not None and entry.get("size") == size:
            logger.info("Skipping already-converted file: %s", file)
        else:
            files.append(file)

    logger.info("Found %d CSV files in %s (%d new, %d already converted)",
                len(all_files), input_dir, len(files), len(all_files) - len(files))

    for i, file in enumerate(files, start=1):
        logger.info("Processing %d/%d: %s", i, len(files), file)

        df = pd.read_csv(file, low_memory=False)
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")
        df["YEAR"] = df["FL_DATE"].dt.year
        df["MONTH"] = df["FL_DATE"].dt.month

        df.to_parquet(
            output_dir,
            engine="pyarrow",
            partition_cols=["YEAR", "MONTH"],
            index=False,
            existing_data_behavior="overwrite_or_ignore",
        )

        manifest[storage.basename(file)] = {"size": storage.get_size(file)}
        _save_manifest(output_dir, manifest)
        logger.info("Completed %d/%d", i, len(files))

    logger.info("%d new file(s) converted successfully", len(files))
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=config.RAW_CSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.RAW_PARQUET_DIR)
    args = parser.parse_args()
    convert_csvs_to_parquet(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
