#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import (  # noqa: E402
    ANN_BACKEND,
    AWS_REGION,
    EMBED_DIM,
    EMBED_MODEL,
    S3_PREFIX_REGISTER_REQUEST,
    S3_PREFIX_REJECTED,
    S3_PREFIX_VERIFY_REQUEST,
    S3_PREFIX_WM_REQUEST_ORIGINAL,
    S3_PREFIX_WM_RESULT,
    S3_DEFAULT_BUCKET,
    VECTOR_DSN,
    VECTOR_KEY_COL,
    VECTOR_PHASH_COL,
    VECTOR_TABLE,
)


def check_workflow_env() -> tuple[bool, str]:
    required = {
        "S3_DEFAULT_BUCKET": bool(S3_DEFAULT_BUCKET),
        "S3_PREFIX_REGISTER_REQUEST": bool(S3_PREFIX_REGISTER_REQUEST),
        "S3_PREFIX_VERIFY_REQUEST": bool(S3_PREFIX_VERIFY_REQUEST),
        "S3_PREFIX_REJECTED": bool(S3_PREFIX_REJECTED),
        "S3_PREFIX_WM_REQUEST_ORIGINAL": bool(S3_PREFIX_WM_REQUEST_ORIGINAL),
        "S3_PREFIX_WM_RESULT": bool(S3_PREFIX_WM_RESULT),
        "VECTOR_KEY_COL": bool(VECTOR_KEY_COL),
        "VECTOR_PHASH_COL": bool(VECTOR_PHASH_COL),
    }
    missing = [k for k, ok in required.items() if not ok]
    if missing:
        return False, "missing keys: " + ", ".join(missing)
    return True, "workflow env keys present"


def check_s3() -> tuple[bool, str]:
    if not S3_DEFAULT_BUCKET:
        return False, "S3_DEFAULT_BUCKET empty"
    try:
        import boto3
    except Exception as exc:
        return False, f"boto3 import failed: {exc}"
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION or None)
        s3.head_bucket(Bucket=S3_DEFAULT_BUCKET)
        return True, f"bucket ok: {S3_DEFAULT_BUCKET}"
    except Exception as exc:
        return False, f"s3 access failed: {exc}"


def check_pgvector() -> tuple[bool, str]:
    if ANN_BACKEND != "pgvector":
        return True, f"skip (ANN_BACKEND={ANN_BACKEND})"
    if not VECTOR_DSN:
        return False, "VECTOR_DSN empty"
    try:
        import psycopg
    except Exception as exc:
        return False, f"psycopg import failed: {exc}"
    try:
        with psycopg.connect(VECTOR_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("select current_database()")
                dbname = cur.fetchone()[0]
                cur.execute("select exists (select 1 from pg_extension where extname='vector')")
                has_vector = bool(cur.fetchone()[0])
                cur.execute(
                    """
                    select exists(
                        select 1
                        from information_schema.tables
                        where table_schema='public' and table_name=%s
                    )
                    """,
                    (VECTOR_TABLE,),
                )
                has_table = bool(cur.fetchone()[0])
        if not has_vector:
            return False, f"db={dbname}, vector extension missing"
        if not has_table:
            return False, f"db={dbname}, table missing: {VECTOR_TABLE}"
        return True, f"db={dbname}, vector ok, table ok: {VECTOR_TABLE}"
    except Exception as exc:
        return False, f"pgvector check failed: {exc}"


def main() -> int:
    print("=== IMG_GUARD PREFLIGHT ===")
    print(f"EMBED_MODEL={EMBED_MODEL} (dim={EMBED_DIM})")
    print(f"ANN_BACKEND={ANN_BACKEND}")
    print(f"S3_DEFAULT_BUCKET={S3_DEFAULT_BUCKET or '(empty)'}")
    print(f"VECTOR_TABLE={VECTOR_TABLE}")

    checks = [
        ("workflow-env", check_workflow_env()),
        ("s3", check_s3()),
        ("pgvector", check_pgvector()),
    ]

    failed = 0
    for name, (ok, msg) in checks:
        state = "OK" if ok else "FAIL"
        print(f"[{state}] {name}: {msg}")
        if not ok:
            failed += 1

    if failed:
        print(f"PREFLIGHT_FAIL count={failed}")
        return 1
    print("PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
