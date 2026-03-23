from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import imagehash
import numpy as np

from app.config import (
    AWS_REGION,
    S3_DEFAULT_BUCKET,
    S3_ENDPOINT_URL,
    S3_PREFIX_REGISTER_REQUEST,
    S3_PREFIX_REJECTED,
    S3_PREFIX_VERIFY_REQUEST,
    S3_PREFIX_WM_REQUEST_ORIGINAL,
    S3_PREFIX_WM_RESULT,
    TMP_DIR,
    VECTOR_DSN,
    VECTOR_EMBED_COL,
    VECTOR_FILE_COL,
    VECTOR_ID_COL,
    VECTOR_KEY_COL,
    VECTOR_PHASH_COL,
    VECTOR_TABLE,
    VECTOR_URL_COL,
    runtime_signature,
)
from app.contracts_v1 import (
    ArchiveImageRequestV1,
    ArchiveImageResponseV1,
    InputItemV1,
    VectorUpsertRequestV1,
    VectorUpsertResponseV1,
)
from app.embedder import ClipEmbedder
from app.preprocess import load_image_fixed
from app.source_io import parse_s3_uri, resolve_source_to_local

_CACHE: dict[str, Any] = {}

KIND_TO_PREFIX = {
    "watermark_request_original": S3_PREFIX_WM_REQUEST_ORIGINAL,
    "watermark_result": S3_PREFIX_WM_RESULT,
    "verify_request": S3_PREFIX_VERIFY_REQUEST,
    "register_request": S3_PREFIX_REGISTER_REQUEST,
    "rejected_request": S3_PREFIX_REJECTED,
}


def _safe_part(v: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (v or "").strip())
    return s.strip("._-") or "unknown"


def _safe_filename(name: str) -> str:
    base = Path((name or "").strip()).name
    cleaned = _safe_part(base)
    if "." not in cleaned:
        return f"{cleaned}.bin"
    return cleaned


def _resolve_input_source(item: InputItemV1) -> str:
    source = item.local_path or item.url or item.s3_uri or item.s3_key
    if not source:
        raise ValueError("input requires one of: local_path, url, s3_uri, s3_key")
    return source


def _build_s3_client():
    try:
        import boto3
    except Exception as exc:
        raise RuntimeError("boto3 is required for S3 persistence") from exc

    return boto3.client(
        "s3",
        region_name=AWS_REGION or None,
        endpoint_url=S3_ENDPOINT_URL or None,
    )


def _source_filename(source: str, fallback: str = "asset.bin") -> str:
    p = Path(source)
    if p.exists():
        return p.name
    parsed = urlparse(source)
    name = Path(parsed.path).name
    return name or fallback


def _build_archive_key(req: ArchiveImageRequestV1, local_path: Path, source: str) -> str:
    prefix = KIND_TO_PREFIX[req.kind].strip("/")
    user_id = _safe_part(str(req.meta.get("user_id", "anon")))
    content_id = _safe_part(str(req.meta.get("content_id", "none")))
    job_id = _safe_part(req.job_id)

    raw_name = req.output_filename or req.input.filename or local_path.name or _source_filename(source)
    file_name = _safe_filename(raw_name)
    token = uuid.uuid4().hex[:12]
    return f"{prefix}/{user_id}/{content_id}/{job_id}/{token}_{file_name}"


def archive_image_v1(req: ArchiveImageRequestV1 | dict[str, Any]) -> ArchiveImageResponseV1:
    parsed = req if isinstance(req, ArchiveImageRequestV1) else ArchiveImageRequestV1.model_validate(req)

    bucket = (parsed.bucket or S3_DEFAULT_BUCKET).strip()
    if not bucket:
        return ArchiveImageResponseV1(
            job_id=parsed.job_id,
            kind=parsed.kind,
            success=False,
            reason="S3 bucket is empty. Set request.bucket or S3_DEFAULT_BUCKET.",
        )

    try:
        source = _resolve_input_source(parsed.input)
        local_path = resolve_source_to_local(source, TMP_DIR, default_s3_bucket=S3_DEFAULT_BUCKET)
        key = _build_archive_key(parsed, local_path, source)

        s3 = _build_s3_client()
        s3.upload_file(str(local_path), bucket, key)

        return ArchiveImageResponseV1(
            job_id=parsed.job_id,
            kind=parsed.kind,
            success=True,
            reason=None,
            bucket=bucket,
            s3_key=key,
            s3_uri=f"s3://{bucket}/{key}",
            file_name=Path(key).name,
        )
    except Exception as exc:
        return ArchiveImageResponseV1(
            job_id=parsed.job_id,
            kind=parsed.kind,
            success=False,
            reason=str(exc),
        )


def _get_embedder() -> ClipEmbedder:
    signature = runtime_signature()
    if _CACHE.get("embed_signature") != signature:
        _CACHE["embedder"] = ClipEmbedder()
        _CACHE["embed_signature"] = signature
    return _CACHE["embedder"]


def _phash_to_pg_bigint(path: str) -> int:
    ph_u64 = int(str(imagehash.phash(load_image_fixed(path))), 16)
    return ph_u64 if ph_u64 < (1 << 63) else ph_u64 - (1 << 64)


def _to_vec_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec.astype(np.float32).reshape(-1).tolist()) + "]"


def _infer_s3_key(item: InputItemV1) -> str | None:
    if item.s3_key:
        return item.s3_key
    if item.s3_uri:
        _, key = parse_s3_uri(item.s3_uri)
        return key
    return None


def _connect_pg():
    if not VECTOR_DSN:
        raise RuntimeError("VECTOR_DSN is empty. set VECTOR_DSN (or DB_*) before vector upsert.")
    try:
        import psycopg
    except Exception as exc:
        raise RuntimeError("psycopg is required for pgvector upsert.") from exc
    return psycopg.connect(VECTOR_DSN)


def upsert_vector_embedding_v1(req: VectorUpsertRequestV1 | dict[str, Any]) -> VectorUpsertResponseV1:
    parsed = req if isinstance(req, VectorUpsertRequestV1) else VectorUpsertRequestV1.model_validate(req)

    try:
        if not VECTOR_TABLE:
            raise RuntimeError("VECTOR_TABLE is empty.")
        if not VECTOR_EMBED_COL:
            raise RuntimeError("VECTOR_EMBED_COL is empty.")
        if not VECTOR_FILE_COL:
            raise RuntimeError("VECTOR_FILE_COL is empty.")
        if not VECTOR_ID_COL:
            raise RuntimeError("VECTOR_ID_COL is empty.")
        if not VECTOR_DSN:
            raise RuntimeError("VECTOR_DSN is empty. set VECTOR_DSN (or DB_*) before vector upsert.")

        source = _resolve_input_source(parsed.input)
        local_path = str(resolve_source_to_local(source, TMP_DIR, default_s3_bucket=S3_DEFAULT_BUCKET))

        embedder = _get_embedder()
        vec = embedder.embed_paths([local_path], batch_size=1)[0]
        vec_literal = _to_vec_literal(vec)
        phash_pg = _phash_to_pg_bigint(local_path)

        file_name = parsed.file_name or parsed.input.filename or Path(local_path).name
        s3_key = parsed.s3_key or _infer_s3_key(parsed.input)
        asset_url = parsed.asset_url or parsed.input.url

        cols = [VECTOR_FILE_COL]
        vals: list[Any] = [file_name]
        phs = ["%s"]

        if VECTOR_KEY_COL:
            cols.append(VECTOR_KEY_COL)
            vals.append(s3_key)
            phs.append("%s")

        if VECTOR_URL_COL:
            cols.append(VECTOR_URL_COL)
            vals.append(asset_url)
            phs.append("%s")

        if VECTOR_PHASH_COL:
            cols.append(VECTOR_PHASH_COL)
            vals.append(phash_pg)
            phs.append("%s")

        cols.append(VECTOR_EMBED_COL)
        vals.append(vec_literal)
        phs.append("%s::vector")

        col_sql = ", ".join(cols)
        ph_sql = ", ".join(phs)

        with _connect_pg() as conn:
            with conn.cursor() as cur:
                if s3_key and VECTOR_KEY_COL:
                    updates = [f"{VECTOR_FILE_COL}=EXCLUDED.{VECTOR_FILE_COL}", f"{VECTOR_EMBED_COL}=EXCLUDED.{VECTOR_EMBED_COL}"]
                    if VECTOR_URL_COL:
                        updates.append(f"{VECTOR_URL_COL}=EXCLUDED.{VECTOR_URL_COL}")
                    if VECTOR_PHASH_COL:
                        updates.append(f"{VECTOR_PHASH_COL}=EXCLUDED.{VECTOR_PHASH_COL}")
                    update_sql = ", ".join(updates)
                    sql = f"""
                        INSERT INTO {VECTOR_TABLE} ({col_sql})
                        VALUES ({ph_sql})
                        ON CONFLICT ({VECTOR_KEY_COL}) WHERE {VECTOR_KEY_COL} IS NOT NULL
                        DO UPDATE SET {update_sql}
                        RETURNING {VECTOR_ID_COL}
                    """
                    cur.execute(sql, vals)
                else:
                    sql = f"""
                        INSERT INTO {VECTOR_TABLE} ({col_sql})
                        VALUES ({ph_sql})
                        RETURNING {VECTOR_ID_COL}
                    """
                    cur.execute(sql, vals)

                row = cur.fetchone()
            conn.commit()

        record_id = None if row is None else int(row[0])
        return VectorUpsertResponseV1(
            job_id=parsed.job_id,
            success=True,
            reason=None,
            table=VECTOR_TABLE,
            record_id=record_id,
            file_name=file_name,
            s3_key=s3_key,
            phash=phash_pg,
        )

    except Exception as exc:
        return VectorUpsertResponseV1(
            job_id=parsed.job_id,
            success=False,
            reason=f"{type(exc).__name__}: {exc}",
            table=VECTOR_TABLE or None,
            record_id=None,
            file_name=parsed.file_name or parsed.input.filename,
            s3_key=parsed.s3_key or _infer_s3_key(parsed.input),
            phash=None,
        )
