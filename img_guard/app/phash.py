# app/phash.py
"""
phash.py
- ANN 검색 결과 후보에 대해 pHash 거리 계산
- '같음'에 대한 강한 증거를 제공하는 정밀 비교 단계
- 후보들 중 픽셀 구조가 거의 같은지를 확인

설계 원칙
1) ANN Top-K 중에서 TOP_PHASH개만 계산
2) ANNResult 를 in-place로 보강(파이프라인 연결 쉬움)
"""

from __future__ import annotations
from typing import Dict
import imagehash
from PIL import Image

from app.preprocess import load_image_fixed
from app.types import ANNResult
from app.config import TOP_PHASH


class PHashComparator:
    """
    pHash 기반 정밀 비교기
    - DB 이미지 pHash를 캐싱해서 반복 계산 방지
    """

    def __init__(self):
        self._cache: Dict[str, imagehash.ImageHash] = {}

    def _get_phash(self, path: str) -> imagehash.ImageHash:
        if path not in self._cache:
            img = load_image_fixed(path)
            self._cache[path] = imagehash.phash(img)
        return self._cache[path]

    def enrich(
        self,
        query_path: str,
        candidates: list[ANNResult],
        resolve_path_fn,
        top_n: int | None = None,
   ) -> list[ANNResult]:
        """
        query_path: 쿼리 이미지 경로
        candidates: ANN Top-K 결과
        resolve_path_fn: db_file -> 실제 경로 변환 함수
        """

        q_img = load_image_fixed(query_path)
        q_ph = imagehash.phash(q_img)

        limit = top_n if top_n is not None else TOP_PHASH
        for c in candidates[:limit]:
            lookup_key = c.db_key or c.db_file
            db_path = resolve_path_fn(lookup_key)
            if db_path is None:
                continue
            db_ph = self._get_phash(db_path)
            c.phash_dist = int(q_ph - db_ph)

        return candidates
