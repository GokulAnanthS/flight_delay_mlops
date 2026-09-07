"""Local-filesystem/S3 dispatch for the handful of file operations the
pipeline needs that pandas/pyarrow don't cover natively (pandas/pyarrow
already read/write s3:// paths directly for the actual CSV/Parquet IO).

Every function here takes a path that's either a local path or an
"s3://bucket/key" string and does the right thing. Callers should never
branch on DATA_BACKEND themselves -- add a case here instead.
"""

import fnmatch
import json
import logging
import os
import posixpath
import shutil
from functools import lru_cache
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)


def is_s3_path(path) -> bool:
    return str(path).startswith("s3://")


def _parse_s3(path: str) -> tuple[str, str]:
    without_scheme = str(path)[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


@lru_cache(maxsize=1)
def _client():
    return boto3.client("s3")


def basename(path) -> str:
    """Path(...).name is unreliable on an s3:// string (worse on Windows,
    where Path is WindowsPath and treats ':' as drive-letter syntax), so
    dispatch explicitly instead of routing s3 URIs through pathlib.
    """
    if is_s3_path(path):
        return posixpath.basename(str(path).rstrip("/"))
    return Path(path).name


def join(base, *parts: str) -> str:
    """Join path segments, staying a string for s3:// paths and a Path otherwise."""
    if is_s3_path(base):
        return posixpath.join(str(base).rstrip("/"), *parts)
    return str(Path(base, *parts))


def ensure_dir(dir_) -> None:
    """No-op for S3 -- there are no real directories, just key prefixes."""
    if is_s3_path(dir_):
        return
    os.makedirs(dir_, exist_ok=True)


def list_files(dir_, pattern: str = "*.csv") -> list[str]:
    """Sorted, non-recursive glob of `pattern` directly under `dir_`.

    Matches glob.glob(dir_/pattern) semantics for local paths. For S3,
    list_objects_v2 has no glob support, so we list everything under the
    prefix and filter to immediate children (no further "/") whose
    basename matches `pattern` via fnmatch.
    """
    if is_s3_path(dir_):
        bucket, prefix = _parse_s3(dir_)
        prefix = prefix.rstrip("/") + "/"
        matches = []
        paginator = _client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                remainder = key[len(prefix):]
                if "/" in remainder:
                    continue
                if fnmatch.fnmatch(remainder, pattern):
                    matches.append(f"s3://{bucket}/{key}")
        return sorted(matches)

    import glob
    return sorted(glob.glob(str(Path(dir_) / pattern)))


def get_size(path) -> int:
    if is_s3_path(path):
        bucket, key = _parse_s3(path)
        return _client().head_object(Bucket=bucket, Key=key)["ContentLength"]
    return os.path.getsize(path)


def read_json(path):
    """Returns None if the object/file doesn't exist yet, parsed JSON otherwise."""
    if is_s3_path(path):
        bucket, key = _parse_s3(path)
        try:
            body = _client().get_object(Bucket=bucket, Key=key)["Body"].read()
        except _client().exceptions.NoSuchKey:
            return None
        return json.loads(body)

    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def write_json(path, data) -> None:
    payload = json.dumps(data, indent=2, sort_keys=True)
    if is_s3_path(path):
        bucket, key = _parse_s3(path)
        _client().put_object(Bucket=bucket, Key=key, Body=payload.encode("utf-8"))
        return
    with open(path, "w") as f:
        f.write(payload)


def copy_file(src, dest) -> None:
    src_is_s3, dest_is_s3 = is_s3_path(src), is_s3_path(dest)

    if not src_is_s3 and not dest_is_s3:
        shutil.copy2(src, dest)
        return

    if src_is_s3 and dest_is_s3:
        src_bucket, src_key = _parse_s3(src)
        dest_bucket, dest_key = _parse_s3(dest)
        _client().copy_object(
            Bucket=dest_bucket, Key=dest_key, CopySource={"Bucket": src_bucket, "Key": src_key}
        )
        return

    if dest_is_s3:
        bucket, key = _parse_s3(dest)
        _client().upload_file(str(src), bucket, key)
        return

    bucket, key = _parse_s3(src)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(bucket, key, str(dest))
