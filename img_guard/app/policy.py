# app/policy.py
"""
policy.py
- ANN + pHash 결과를 기반으로 3상태 정책 판정 수행
- ALLOW / REVIEW / BLOCK
"""

from __future__ import annotations
from typing import List

from app.config import (
    COS_BLOCK,
    PHASH_BLOCK,
    COS_ALLOW_A,
    PHASH_ALLOW_A,
    COS_ALLOW_B,
)
from app.types import ANNResult, GuardResult, Decision


class PolicyEngine:
    def decide(self, candidates: List[ANNResult]) -> GuardResult:
        """
        candidates: ANN + pHash 정보가 채워진 후보 리스트
                    (cosine 내림차순 정렬 상태 가정)
        """

        # 후보가 없다는 건 "비슷한 이미지 자체가 없음"
        if not candidates:
            return GuardResult(
                decision=Decision.ALLOW,
                reason="No similar images found in database",
                top_match=None,
                candidates=[],
            )

        top = candidates[0]

        # 1) BLOCK 조건
        if (
            top.cosine >= COS_BLOCK
            and top.phash_dist is not None
            and top.phash_dist <= PHASH_BLOCK
        ):
            return GuardResult(
                decision=Decision.BLOCK,
                reason=(
                    f"Highly similar image detected "
                    f"(cosine={top.cosine:.3f}, phash={top.phash_dist})"
                ),
                top_match=top,
                candidates=candidates,
            )

        # 2) ALLOW 조건
        if (
            # 케이스 A: 의미적으로도 낮고, 픽셀도 다름
            (
                top.cosine < COS_ALLOW_A
                and top.phash_dist is not None
                and top.phash_dist > PHASH_ALLOW_A
            )
            # 케이스 B: 의미적으로 확실히 다름
            or top.cosine < COS_ALLOW_B
        ):
            return GuardResult(
                decision=Decision.ALLOW,
                reason=f"No meaningful similarity (top cosine={top.cosine:.3f})",
                top_match=top,
                candidates=candidates,
            )

        # 3) 나머지는 REVIEW
        return GuardResult(
            decision=Decision.REVIEW,
            reason=(
                f"Ambiguous similarity requires review "
                f"(cosine={top.cosine:.3f}, phash={top.phash_dist})"
            ),
            top_match=top,
            candidates=candidates,
        )