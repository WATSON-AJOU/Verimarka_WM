from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from app.config import (
    AWS_REGION,
    DOWNLOAD_RETRIES,
    HTTP_DOWNLOAD_TIMEOUT_SEC,
    MAX_INPUT_MB,
    S3_ENDPOINT_URL,
)


def is_http_url(source: str) -> bool:
    scheme = urlparse(source).scheme.lower()
    return scheme in {"http", "https"}


def is_s3_uri(source: str) -> bool:
    return source.lower().startswith("s3://")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme.lower() != "s3":
        raise ValueError(f"not an s3 uri: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"invalid s3 uri: {uri}")
    return bucket, key


def normalize_source(source: str, default_s3_bucket: str = "") -> str:
    source = (source or "").strip()
    if not source:
        raise ValueError("empty source")

    if is_http_url(source) or is_s3_uri(source):
        return source

    # local file path
    p = Path(source).expanduser()
    if p.exists():
        return str(p.resolve())

    # bare S3 key -> normalize with default bucket
    if default_s3_bucket:
        return f"s3://{default_s3_bucket}/{source.lstrip('/')}"

    raise ValueError(
        "source must be one of: http(s) url, s3:// uri, existing local path, "
        "or bare s3 key with S3_DEFAULT_BUCKET configured"
    )


def _suffix_from_source(source: str) -> str:
    parsed = urlparse(source)
    suffix = Path(parsed.path).suffix.lower()
    return suffix or ".bin"


def _cache_path(source: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_from_source(source)
    filename = hashlib.sha1(source.encode("utf-8")).hexdigest() + suffix
    return out_dir / filename


def _enforce_max_input_size(path: Path) -> None:
    max_bytes = max(1, int(MAX_INPUT_MB)) * 1024 * 1024
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"input file too large: {size} bytes (limit={max_bytes} bytes, MAX_INPUT_MB={MAX_INPUT_MB})"
        )


def _download_http(source: str, out_path: Path) -> Path:
    with urlopen(source, timeout=HTTP_DOWNLOAD_TIMEOUT_SEC) as r, out_path.open("wb") as f:
        shutil.copyfileobj(r, f)
    _enforce_max_input_size(out_path)
    return out_path


def _download_s3(source: str, out_path: Path) -> Path:
    try:
        import boto3
        from botocore.config import Config
    except Exception as exc:
        raise RuntimeError("boto3 is required for s3:// input support") from exc

    bucket, key = parse_s3_uri(source)
    cfg = Config(retries={"max_attempts": max(1, DOWNLOAD_RETRIES + 1), "mode": "standard"})
    client = boto3.client(
        "s3",
        region_name=AWS_REGION or None,
        endpoint_url=S3_ENDPOINT_URL or None,
        config=cfg,
    )
    with out_path.open("wb") as f:
        client.download_fileobj(bucket, key, f)
    _enforce_max_input_size(out_path)
    return out_path


def resolve_source_to_local(
    source: str,
    out_dir: Path,
    default_s3_bucket: str = "",
) -> Path:
    normalized = normalize_source(source, default_s3_bucket=default_s3_bucket)

    # normalized local path
    if not is_http_url(normalized) and not is_s3_uri(normalized):
        path = Path(normalized)
        if not path.exists():
            raise FileNotFoundError(f"source path not found: {normalized}")
        _enforce_max_input_size(path)
        return path.resolve()

    out_path = _cache_path(normalized, out_dir)
    if out_path.exists():
        return out_path.resolve()

    last_exc: Exception | None = None
    for attempt in range(DOWNLOAD_RETRIES + 1):
        try:
            if is_http_url(normalized):
                return _download_http(normalized, out_path).resolve()
            return _download_s3(normalized, out_path).resolve()
        except Exception as exc:
            last_exc = exc
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(0.3 * (attempt + 1))

    assert last_exc is not None
    raise RuntimeError(f"failed to fetch source: {normalized}") from last_exc
