from __future__ import annotations

import hashlib
import json
from typing import Any


def make_payload_bits(meta: dict[str, Any] | None, nbits: int) -> str:
    """Create deterministic payload bits from request metadata."""
    if nbits <= 0:
        raise ValueError("nbits must be > 0")

    raw = json.dumps(meta or {}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    bits = bin(int(digest, 16))[2:].zfill(256)

    if nbits > len(bits):
        # Repeat bits for oversized payloads; deterministic and simple.
        rep = (nbits // len(bits)) + 1
        bits = (bits * rep)[:nbits]
    else:
        bits = bits[:nbits]

    return bits


def make_payload_id(payload_bits: str) -> str:
    return hashlib.sha1(payload_bits.encode("utf-8")).hexdigest()[:16]
