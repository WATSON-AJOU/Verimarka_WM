#!/usr/bin/env python3
from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import AWS_REGION, S3_DEFAULT_BUCKET, S3_ENDPOINT_URL  # noqa: E402
from app.persist_service import upsert_vector_embedding_v1  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _iter_images(src_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [p for p in src_dir.glob(pattern) if p.is_file() and p.suffix.lower() in IMG_EXT]
    files.sort()
    return files


def _build_s3_client():
    try:
        import boto3
    except Exception as exc:
        raise RuntimeError("boto3 is required for --upload-s3-prefix mode") from exc
    return boto3.client(
        "s3",
        region_name=AWS_REGION or None,
        endpoint_url=S3_ENDPOINT_URL or None,
    )


def _to_posix_rel(path: Path, base: Path) -> str:
    return str(path.relative_to(base)).replace("\\", "/")


def _safe_key_part(v: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-/") else "_" for ch in v)


def main() -> int:
    ap = argparse.ArgumentParser(description="Preload local images into vector DB (optional S3 upload).")
    ap.add_argument("--src-dir", required=True, help="source directory containing images")
    ap.add_argument("--recursive", action="store_true", help="scan directory recursively")
    ap.add_argument("--limit", type=int, default=0, help="max images to process (0 = all)")
    ap.add_argument("--upload-s3-prefix", default="", help="if set, upload each image to s3://<bucket>/<prefix>/...")
    ap.add_argument("--bucket", default="", help="S3 bucket (default: S3_DEFAULT_BUCKET env)")
    ap.add_argument("--dry-run", action="store_true", help="print planned actions only")
    args = ap.parse_args()

    src_dir = Path(args.src_dir).expanduser().resolve()
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"ERROR: src-dir not found or not directory: {src_dir}")
        return 1

    images = _iter_images(src_dir, recursive=args.recursive)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print("ERROR: no images found")
        return 1

    bucket = (args.bucket or S3_DEFAULT_BUCKET).strip()
    use_s3 = bool(args.upload_s3_prefix.strip())
    s3 = None
    if use_s3:
        if not bucket:
            print("ERROR: bucket is empty. set --bucket or S3_DEFAULT_BUCKET.")
            return 1
        s3 = _build_s3_client()

    print(f"START preload | src={src_dir} | count={len(images)} | use_s3={use_s3}")
    ok = 0
    fail = 0

    for idx, p in enumerate(images, start=1):
        rel = _to_posix_rel(p, src_dir)
        file_name = p.name
        mime = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        s3_key = None

        try:
            if use_s3:
                prefix = args.upload_s3_prefix.strip("/ ")
                key = f"{prefix}/{_safe_key_part(rel)}"
                if args.dry_run:
                    print(f"[DRY] [{idx}/{len(images)}] upload {p} -> s3://{bucket}/{key}")
                else:
                    assert s3 is not None
                    extra = {"ContentType": mime}
                    s3.upload_file(str(p), bucket, key, ExtraArgs=extra)
                s3_key = key

            req = {
                "job_id": f"preload-{uuid.uuid4().hex[:12]}",
                "input": {
                    "local_path": str(p),
                    "filename": file_name,
                    "mime_type": mime,
                },
                "s3_key": s3_key,
                "file_name": file_name,
            }

            if args.dry_run:
                print(f"[DRY] [{idx}/{len(images)}] upsert file={file_name} s3_key={s3_key}")
                ok += 1
                continue

            resp = upsert_vector_embedding_v1(req)
            if resp.success:
                ok += 1
                print(f"[OK ] [{idx}/{len(images)}] id={resp.record_id} file={resp.file_name} s3_key={resp.s3_key}")
            else:
                fail += 1
                print(f"[ERR] [{idx}/{len(images)}] file={file_name} reason={resp.reason}")
        except Exception as exc:
            fail += 1
            print(f"[ERR] [{idx}/{len(images)}] file={file_name} exc={type(exc).__name__}: {exc}")

    print(f"DONE preload | ok={ok} fail={fail} total={len(images)}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

