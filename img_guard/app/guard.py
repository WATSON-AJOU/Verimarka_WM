# app/guard.py
"""
guard.py
- img_guard 파이프라인 관리자
- 입력: query 이미지 경로
- 출력: GuardResult (ALLOW/REVIEW/BLOCK + 근거)

구성:
1) embedder: query 임베딩
2) ann_index: Top-K 후보 검색
3) phash: 후보에 pHash 거리 채우기
4) policy: 3상태 판정
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import TOP_K
from app.embedder import ClipEmbedder
from app.ann_index import ANNIndex
from app.phash import PHashComparator
from app.policy import PolicyEngine
from app.types import GuardResult, Decision


@dataclass
class GuardEngine:
    """
    엔진 구성요소를 하나로 묶어 재사용 가능하게 함.
    - 서비스에서는 보통 서버 시작 시 1회 생성 후 계속 재사용
    """

    embedder: ClipEmbedder
    ann: ANNIndex
    phash: PHashComparator
    policy: PolicyEngine

    @classmethod
    def create(cls) -> "GuardEngine":
        """기본 구성으로 엔진 생성 (lazy하게 ann은 ensure_ready에서 준비됨)"""
        return cls(
            embedder=ClipEmbedder(),
            ann=ANNIndex(),
            phash=PHashComparator(),
            policy=PolicyEngine(),
        )

    def run(self, query_path: str, k: int = TOP_K) -> GuardResult:
        """
        query_path: 사용자가 등록하려는 이미지 경로
        k: ANN 후보 개수 
        """
        # 1) query 임베딩
        q_vec = self.embedder.embed_paths([query_path], batch_size=1)[0]

        # 2) ANN 검색 (cosine 기반 Top-K)
        candidates = self.ann.search(q_vec, k=k)

        # 3) 후보에 pHash 거리 채우기 (Top-PHASH까지만)
        candidates = self.phash.enrich(
            query_path=query_path,
            candidates=candidates,
            resolve_path_fn=self.ann.get_full_path,  # 주입(injection)
        )

        # 4) 정책 판정 (3상태)
        result = self.policy.decide(candidates)
        return result