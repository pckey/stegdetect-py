"""MD5 hash implementation.

Simple wrapper around hashlib.
"""

from __future__ import annotations

import hashlib


def md5_hash(data: bytes) -> bytes:
    """Return 16-byte MD5 digest.

    Args:
        data: Input data to hash.

    Returns:
        16-byte MD5 digest.
    """
    return hashlib.md5(data).digest()
