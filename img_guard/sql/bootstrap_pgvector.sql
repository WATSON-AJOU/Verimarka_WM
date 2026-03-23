-- bootstrap_pgvector.sql
-- pgvector + 모델별 임베딩 테이블 초기 세팅

CREATE EXTENSION IF NOT EXISTS vector;

-- CLIP ViT-B/32 (512)
CREATE TABLE IF NOT EXISTS image_embeddings_clip_b32 (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    s3_key TEXT,
    asset_url TEXT,
    phash BIGINT,
    embedding VECTOR(512) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- OpenCLIP ViT-H/14 (1024)
CREATE TABLE IF NOT EXISTS image_embeddings_openclip_h14 (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    s3_key TEXT,
    asset_url TEXT,
    phash BIGINT,
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- SigLIP2 SO400M 384 (1152)
CREATE TABLE IF NOT EXISTS image_embeddings_siglip2_so400m (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    s3_key TEXT,
    asset_url TEXT,
    phash BIGINT,
    embedding VECTOR(1152) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 보조 인덱스
CREATE INDEX IF NOT EXISTS idx_clip_b32_file_name ON image_embeddings_clip_b32 (file_name);
CREATE INDEX IF NOT EXISTS idx_openclip_h14_file_name ON image_embeddings_openclip_h14 (file_name);
CREATE INDEX IF NOT EXISTS idx_siglip2_file_name ON image_embeddings_siglip2_so400m (file_name);

CREATE INDEX IF NOT EXISTS idx_clip_b32_s3_key ON image_embeddings_clip_b32 (s3_key);
CREATE INDEX IF NOT EXISTS idx_openclip_h14_s3_key ON image_embeddings_openclip_h14 (s3_key);
CREATE INDEX IF NOT EXISTS idx_siglip2_s3_key ON image_embeddings_siglip2_so400m (s3_key);

-- UPSERT conflict target (s3_key)용 unique partial index
CREATE UNIQUE INDEX IF NOT EXISTS uq_clip_b32_s3_key
ON image_embeddings_clip_b32 (s3_key)
WHERE s3_key IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_openclip_h14_s3_key
ON image_embeddings_openclip_h14 (s3_key)
WHERE s3_key IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_siglip2_s3_key
ON image_embeddings_siglip2_so400m (s3_key)
WHERE s3_key IS NOT NULL;

-- 벡터 HNSW 인덱스 (코사인)
CREATE INDEX IF NOT EXISTS idx_clip_b32_hnsw
ON image_embeddings_clip_b32
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS idx_openclip_h14_hnsw
ON image_embeddings_openclip_h14
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS idx_siglip2_hnsw
ON image_embeddings_siglip2_so400m
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);
