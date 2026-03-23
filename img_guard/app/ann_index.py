# app/ann_index.py
"""
ann_index.py

로컬(HNSW)과 서비스(pgvector) 백엔드를 모두 지원하도록 분리.

LocalHNSWIndex:
- DB 이미지 목록 수집 + "순서 고정(manifest)" 저장
- DB 변경 감지(signature) → 필요 시 자동 rebuild
- 임베딩 생성/저장(embeddings.npy)
- HNSW 인덱스 생성/저장(hnsw.index)
- 쿼리 벡터로 Top-K 검색

PgVectorIndex (서비스):
- PostgreSQL(pgvector)에 쿼리 벡터로 Top-K 검색 요청
- 결과의 db_key/S3 URL을 기반으로 pHash용 이미지 다운로드 지원
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import hnswlib

from app.config import (
    ANN_BACKEND,
    DB_IMAGES_DIR,
    EMBED_DIM,
    EMBED_MODEL,
    EMBEDDINGS_PATH,
    HNSW_INDEX_PATH,
    TOP_K,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    # config.py에 추가한 값들
    DB_MANIFEST_PATH,
    DB_SIGNATURE_MODE,
    VECTOR_DSN,
    VECTOR_TABLE,
    VECTOR_EMBED_COL,
    VECTOR_ID_COL,
    VECTOR_FILE_COL,
    VECTOR_KEY_COL,
    VECTOR_URL_COL,
    VECTOR_PHASH_COL,
    VECTOR_S3_BUCKET,
    TMP_DIR,
)
from app.embedder import ClipEmbedder
from app.source_io import resolve_source_to_local
from app.types import ANNResult

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------
# Utils: DB scan / signature
# ---------------------------

def _iter_db_files(root: Path) -> List[Path]:
    if not root.exists():
        raise RuntimeError(f"DB_IMAGES_DIR not found: {root}")

    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            files.append(p)

    # 정렬 기준: "relative path string" (순서 안정성)
    files.sort(key=lambda x: str(x.relative_to(root)).lower())
    return files


def _make_db_id(root: Path, path: Path) -> str:
    """basename 금지. 상대경로를 ID로 사용."""
    return str(path.relative_to(root)).replace("\\", "/")


def _file_sig_mtime_size(p: Path) -> str:
    st = p.stat()
    # 문자열로 저장 (json-friendly)
    return f"{int(st.st_mtime)}:{st.st_size}"


def _file_sig_sha1(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_db_signature(root: Path, files: List[Path], mode: str = "mtime_size") -> str:
    """
    DB 전체 시그니처 생성.
    - mode="mtime_size": 빠름(권장). 운영에서 대부분 충분.
    - mode="sha1": 느리지만 가장 정확.
    """
    h = hashlib.sha1()
    h.update(mode.encode("utf-8"))

    for p in files:
        db_id = _make_db_id(root, p)
        if mode == "sha1":
            sig = _file_sig_sha1(p)
        else:
            sig = _file_sig_mtime_size(p)

        # 경로+sig 결합 → 전체 해시
        h.update(db_id.encode("utf-8"))
        h.update(b"\0")
        h.update(sig.encode("utf-8"))
        h.update(b"\n")

    return h.hexdigest()


# ---------------------------
# Manifest IO
# ---------------------------

@dataclass
class DBManifest:
    root: str                 # DB_IMAGES_DIR absolute
    signature_mode: str
    signature: str
    db_ids: List[str]         # relative ids in fixed order
    embed_model: str
    embed_dim: int

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "signature_mode": self.signature_mode,
            "signature": self.signature,
            "db_ids": self.db_ids,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
        }

    @staticmethod
    def from_dict(d: dict) -> "DBManifest":
        return DBManifest(
            root=d["root"],
            signature_mode=d["signature_mode"],
            signature=d["signature"],
            db_ids=list(d["db_ids"]),
            embed_model=d.get("embed_model", "clip_vit_b32_openai"),
            embed_dim=int(d.get("embed_dim", 512)),
        )


def save_manifest(manifest: DBManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)


def load_manifest(path: Path) -> DBManifest:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return DBManifest.from_dict(data)


# ---------------------------
# Runtime State
# ---------------------------

@dataclass
class IndexState:
    db_ids: List[str]                 # stable order
    db_paths: List[str]               # full paths aligned with db_ids
    path_map: Dict[str, str]          # db_id -> full path (O(1))
    db_vecs: np.ndarray               # (N, D) float32
    index: hnswlib.Index              # cosine space


class LocalHNSWIndex:
    """
    HNSW 기반 ANN 검색 인덱스.

    build(force=False):
      - 현재 DB 스캔 → manifest 생성 → embeddings.npy/hnsw.index 저장

    load():
      - manifest + embeddings.npy + hnsw.index 로드
      - DB가 바뀌었으면(시그니처 mismatch) 예외 대신 rebuild 유도 가능

    ensure_ready():
      - (파일 존재 + signature match)면 load
      - 아니면 build
    """

    def __init__(self):
        self.state: Optional[IndexState] = None

    # -----------
    # DB snapshot
    # -----------

    def _snapshot_db(self) -> Tuple[List[str], List[str], Dict[str, str], str]:
        """
        현재 DB를 스캔해서:
        - db_ids (relative, stable order)
        - db_paths (full, aligned)
        - path_map
        - db_signature
        를 반환
        """
        files = _iter_db_files(DB_IMAGES_DIR)
        if len(files) == 0:
            raise RuntimeError(f"DB 이미지가 비어있음: {DB_IMAGES_DIR}")

        db_ids: List[str] = []
        db_paths: List[str] = []
        path_map: Dict[str, str] = {}

        for p in files:
            db_id = _make_db_id(DB_IMAGES_DIR, p)
            full = str(p)
            db_ids.append(db_id)
            db_paths.append(full)
            path_map[db_id] = full

        sig = compute_db_signature(DB_IMAGES_DIR, files, mode=DB_SIGNATURE_MODE)
        return db_ids, db_paths, path_map, sig

    def _is_manifest_compatible(self, manifest: DBManifest) -> bool:
        # root가 다르면 다른 DB로 간주
        if Path(manifest.root).resolve() != DB_IMAGES_DIR.resolve():
            return False
        # signature_mode가 다르면 다시 빌드 권장
        if manifest.signature_mode != DB_SIGNATURE_MODE:
            return False
        # 모델/차원이 다르면 다른 인덱스로 간주
        if manifest.embed_model != EMBED_MODEL:
            return False
        if int(manifest.embed_dim) != int(EMBED_DIM):
            return False
        return True

    # -----
    # Build
    # -----

    def build(self, force: bool = False) -> None:
        """
        DB → embeddings.npy + hnsw.index + db_manifest.json 생성.
        force=False면 "manifest+files 존재 + signature match"면 load로 대체.
        """
        db_ids, db_paths, path_map, sig = self._snapshot_db()

        # 이미 빌드된 결과가 있고, DB가 안 바뀌었으면 load로
        if (not force) and DB_MANIFEST_PATH.exists() and EMBEDDINGS_PATH.exists() and HNSW_INDEX_PATH.exists():
            try:
                manifest = load_manifest(DB_MANIFEST_PATH)
                if self._is_manifest_compatible(manifest) and manifest.signature == sig:
                    self.load()
                    return
            except Exception:
                # manifest 손상 등 → 그냥 rebuild
                pass

        # 임베딩 생성
        embedder = ClipEmbedder()
        db_vecs = embedder.embed_paths(db_paths, batch_size=32)
        if db_vecs.dtype != np.float32:
            db_vecs = db_vecs.astype(np.float32)

        N, D = db_vecs.shape

        # HNSW index build
        index = hnswlib.Index(space="cosine", dim=D)
        index.init_index(max_elements=N, ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)
        index.add_items(db_vecs, np.arange(N))
        index.set_ef(HNSW_EF_SEARCH)

        # 저장
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(EMBEDDINGS_PATH), db_vecs)
        index.save_index(str(HNSW_INDEX_PATH))

        manifest = DBManifest(
            root=str(DB_IMAGES_DIR.resolve()),
            signature_mode=DB_SIGNATURE_MODE,
            signature=sig,
            db_ids=db_ids,
            embed_model=EMBED_MODEL,
            embed_dim=D,
        )
        save_manifest(manifest, DB_MANIFEST_PATH)

        self.state = IndexState(
            db_ids=db_ids,
            db_paths=db_paths,
            path_map=path_map,
            db_vecs=db_vecs,
            index=index,
        )

    # ----
    # Load
    # ----

    def load(self, strict: bool = True) -> None:
        """
        저장된 manifest + embeddings.npy + hnsw.index 로드.

        strict=True:
          - signature mismatch면 RuntimeError (운영에서 안전)
        strict=False:
          - mismatch면 내부적으로 rebuild 수행
        """
        if not DB_MANIFEST_PATH.exists() or not EMBEDDINGS_PATH.exists() or not HNSW_INDEX_PATH.exists():
            raise RuntimeError("인덱스 파일이 없음. build() 먼저 수행하세요.")

        manifest = load_manifest(DB_MANIFEST_PATH)
        if not self._is_manifest_compatible(manifest):
            if strict:
                raise RuntimeError("DB_MANIFEST가 현재 DB와 호환되지 않음. rebuild 필요.")
            self.build(force=True)
            return

        # 현재 DB 시그니처 확인
        files = _iter_db_files(DB_IMAGES_DIR)
        current_sig = compute_db_signature(DB_IMAGES_DIR, files, mode=DB_SIGNATURE_MODE)

        if manifest.signature != current_sig:
            if strict:
                raise RuntimeError("DB가 변경됨(signature mismatch). rebuild 필요.")
            self.build(force=True)
            return

        # manifest의 db_ids 순서를 신뢰하고, 그 순서로 full path 재구성
        path_map: Dict[str, str] = {}
        current_map: Dict[str, str] = {}
        for p in files:
            db_id = _make_db_id(DB_IMAGES_DIR, p)
            current_map[db_id] = str(p)

        db_paths: List[str] = []
        for db_id in manifest.db_ids:
            full = current_map.get(db_id)
            if full is None:
                # signature match인데 이게 None이면 이상한 상황 → rebuild 권장
                if strict:
                    raise RuntimeError(f"Manifest contains missing db_id: {db_id}")
                self.build(force=True)
                return
            db_paths.append(full)
            path_map[db_id] = full

        db_vecs = np.load(str(EMBEDDINGS_PATH)).astype(np.float32)
        N, D = db_vecs.shape

        if N != len(manifest.db_ids):
            if strict:
                raise RuntimeError("embeddings.npy rows != manifest db_ids length. rebuild 필요.")
            self.build(force=True)
            return

        index = hnswlib.Index(space="cosine", dim=D)
        index.load_index(str(HNSW_INDEX_PATH), max_elements=N)
        index.set_ef(HNSW_EF_SEARCH)

        self.state = IndexState(
            db_ids=manifest.db_ids,
            db_paths=db_paths,
            path_map=path_map,
            db_vecs=db_vecs,
            index=index,
        )

    def ensure_ready(self) -> None:
        """
        1) 파일들 존재 + signature match면 load
        2) 아니면 build
        """
        if self.state is not None:
            return

        if DB_MANIFEST_PATH.exists() and EMBEDDINGS_PATH.exists() and HNSW_INDEX_PATH.exists():
            try:
                self.load(strict=True)
                return
            except Exception:
                # 깨졌거나 DB 바뀜 → rebuild
                self.build(force=True)
                return

        self.build(force=True)

    # ------
    # Search
    # ------

    def search(self, query_vec: np.ndarray, k: int = TOP_K) -> List[ANNResult]:
        """
        query_vec: (D,) float32 normalized vector
        반환: ANNResult 리스트
          - db_file에는 "db_id(relative path)"를 넣음 (충돌 방지)
          - cosine만 채움 (pHash는 다음 단계에서 enrich)
        """
        self.ensure_ready()
        assert self.state is not None

        q = query_vec.astype(np.float32).reshape(1, -1)
        k = min(k, len(self.state.db_ids))

        labels, dists = self.state.index.knn_query(q, k=k)
        labels = labels[0].tolist()
        dists = dists[0].tolist()

        out: List[ANNResult] = []
        for idx, dist in zip(labels, dists):
            cos = 1.0 - float(dist)  # cosine similarity
            db_id = self.state.db_ids[idx]  # 상대경로 ID
            out.append(ANNResult(db_file=db_id, cosine=cos))
        return out

    def get_full_path(self, db_file: str) -> str | None:
        """
        db_file == db_id(relative path) 기준으로 O(1) 매핑
        """
        self.ensure_ready()
        assert self.state is not None
        return self.state.path_map.get(db_file)


# ---------------------------
# Service backend: pgvector
# ---------------------------

class PgVectorIndex:
    """
    PostgreSQL(pgvector) 기반 ANN 검색.
    - VECTOR_DSN / VECTOR_TABLE 등은 config.py에서 관리
    - search()는 Top-K 결과를 반환
    - get_full_path()는 후보의 URL/키를 기반으로 로컬 경로를 반환
    """

    def __init__(self):
        self._path_map: Dict[str, str] = {}
        TMP_DIR.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        if not VECTOR_DSN:
            raise RuntimeError("VECTOR_DSN is not set for pgvector backend.")
        try:
            import psycopg
        except Exception as exc:
            raise RuntimeError("psycopg is required for pgvector backend.") from exc
        return psycopg.connect(VECTOR_DSN)

    @staticmethod
    def _vec_to_str(vec: np.ndarray) -> str:
        v = vec.astype(np.float32).reshape(-1)
        return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"

    def search(self, query_vec: np.ndarray, k: int = TOP_K) -> List[ANNResult]:
        vec_str = self._vec_to_str(query_vec)

        # NOTE: table/column names are trusted config values (not user input).
        select_cols = [
            VECTOR_ID_COL,
            VECTOR_FILE_COL,
            VECTOR_KEY_COL,
            VECTOR_URL_COL if VECTOR_URL_COL else "NULL",
            VECTOR_PHASH_COL if VECTOR_PHASH_COL else "NULL",
        ]
        select_sql = ", ".join(select_cols)

        sql = f"""
            SELECT {select_sql},
                   1 - ({VECTOR_EMBED_COL} <=> %s::vector) AS cosine
            FROM {VECTOR_TABLE}
            ORDER BY {VECTOR_EMBED_COL} <=> %s::vector
            LIMIT %s
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_str, vec_str, k))
                rows = cur.fetchall()

        out: List[ANNResult] = []
        for row in rows:
            db_id = row[0]
            file_name = row[1]
            db_key = row[2]
            asset_url = row[3]
            db_phash = row[4]
            cosine = float(row[5])

            db_file = file_name or (Path(db_key).name if db_key else str(db_id))
            result = ANNResult(db_file=db_file, cosine=cosine, db_key=db_key or None, db_phash=db_phash)

            source = asset_url or db_key or file_name
            if source:
                # map by key and file name for lookup
                if db_key:
                    self._path_map[db_key] = source
                if file_name:
                    self._path_map[file_name] = source
                if db_file and db_file not in self._path_map:
                    self._path_map[db_file] = source

            out.append(result)

        return out

    def get_full_path(self, db_file: str) -> str | None:
        """
        db_file(or db_key) -> 로컬 경로
        - URL이면 다운로드 후 경로 반환
        - 로컬 경로면 그대로 반환
        - 그 외는 None
        """
        source = self._path_map.get(db_file)
        if not source:
            return None
        try:
            return str(
                resolve_source_to_local(
                    source=source,
                    out_dir=TMP_DIR,
                    default_s3_bucket=VECTOR_S3_BUCKET,
                )
            )
        except Exception:
            return None


# ---------------------------
# Facade
# ---------------------------

class ANNIndex:
    """
    백엔드 선택 래퍼.
    ANN_BACKEND=local -> LocalHNSWIndex
    ANN_BACKEND=pgvector -> PgVectorIndex
    """

    def __init__(self, backend: str | None = None):
        backend_name = (backend or ANN_BACKEND).lower()
        if backend_name == "local":
            self._impl = LocalHNSWIndex()
        elif backend_name in ("pgvector", "vector", "postgres", "postgresql"):
            self._impl = PgVectorIndex()
        else:
            raise ValueError(f"Unknown ANN_BACKEND: {backend_name}")

    def build(self, force: bool = False) -> None:
        if hasattr(self._impl, "build"):
            self._impl.build(force=force)

    def load(self, strict: bool = True) -> None:
        if hasattr(self._impl, "load"):
            self._impl.load(strict=strict)

    def ensure_ready(self) -> None:
        if hasattr(self._impl, "ensure_ready"):
            self._impl.ensure_ready()

    def search(self, query_vec: np.ndarray, k: int = TOP_K) -> List[ANNResult]:
        return self._impl.search(query_vec, k=k)

    def get_full_path(self, db_file: str) -> str | None:
        return self._impl.get_full_path(db_file)
