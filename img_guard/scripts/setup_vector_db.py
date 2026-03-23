#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import VECTOR_DSN  # noqa: E402


def main() -> int:
    if not VECTOR_DSN:
        print("ERROR: VECTOR_DSN is empty. Set VECTOR_DSN or DB_* env first.")
        return 1

    sql_path = ROOT / "sql" / "bootstrap_pgvector.sql"
    sql_text = sql_path.read_text(encoding="utf-8")

    with psycopg.connect(VECTOR_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_text)
            cur.execute("select exists (select 1 from pg_extension where extname='vector')")
            has_vector = cur.fetchone()[0]
            cur.execute(
                """
                select table_name
                from information_schema.tables
                where table_schema='public'
                  and table_name in (
                      'image_embeddings_clip_b32',
                      'image_embeddings_openclip_h14',
                      'image_embeddings_siglip2_so400m'
                  )
                order by table_name
                """
            )
            tables = [r[0] for r in cur.fetchall()]

    print("PGVECTOR_EXT:", has_vector)
    print("TABLES:", ", ".join(tables) if tables else "(none)")
    print("BOOTSTRAP_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
