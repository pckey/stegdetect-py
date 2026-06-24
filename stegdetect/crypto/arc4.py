"""ARC4 (RC4) stream cipher implementation.

Used for JSteg and Outguess password cracking.
"""

from __future__ import annotations


class ARC4:
    """RC4 stream cipher."""

    def __init__(self, key: bytes) -> None:
        """Initialize with key (raw bytes, NOT hashed).

        Args:
            key: The cipher key as raw bytes.
        """
        self._key = key
        self._s = list(range(256))
        self._i = 0
        self._j = 0
        self._ksa()

    def _ksa(self) -> None:
        """Key Scheduling Algorithm - initialize S-box."""
        j = 0
        key_len = len(self._key)
        for i in range(256):
            j = (j + self._s[i] + self._key[i % key_len]) & 0xFF
            self._s[i], self._s[j] = self._s[j], self._s[i]

    def get_byte(self) -> int:
        """Generate one byte via PRGA.

        Returns:
            A single keystream byte (0-255).
        """
        self._i = (self._i + 1) & 0xFF
        self._j = (self._j + self._s[self._i]) & 0xFF
        self._s[self._i], self._s[self._j] = self._s[self._j], self._s[self._i]
        return self._s[(self._s[self._i] + self._s[self._j]) & 0xFF]

    def get_word(self) -> int:
        """Generate 4 bytes as big-endian uint32.

        Returns:
            A 32-bit unsigned integer from the keystream.
        """
        b0 = self.get_byte()
        b1 = self.get_byte()
        b2 = self.get_byte()
        b3 = self.get_byte()
        return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3

    def skip_bytes(self, n: int) -> None:
        """Skip n bytes efficiently.

        Args:
            n: Number of keystream bytes to skip.
        """
        for _ in range(n):
            self.get_byte()

    def encrypt(self, data: bytes) -> bytes:
        """XOR data with keystream.

        Args:
            data: Data to encrypt/decrypt.

        Returns:
            Encrypted/decrypted data.
        """
        return bytes(b ^ self.get_byte() for b in data)


def arc4_initkey(key: bytes) -> ARC4:
    """Initialize ARC4 with MD5-hashed key.

    MD5 the key first, then use the 16-byte digest as the ARC4 key.

    Args:
        key: The password/key as raw bytes.

    Returns:
        An ARC4 instance initialized with the MD5-hashed key.
    """
    from stegdetect.crypto.md5 import md5_hash

    digest = md5_hash(key)
    return ARC4(digest)


def arc4_fixedkey(key: bytes) -> ARC4:
    """JSteg-specific key setup.

    XOR key bytes into a 5-byte buffer (cycling), then use as ARC4 key.

    Args:
        key: The password/key as raw bytes.

    Returns:
        An ARC4 instance initialized with the JSteg-derived key.
    """
    buf = bytearray(5)
    key_len = len(key)
    for i in range(5):
        buf[i] = key[i % key_len]
    for i in range(5, key_len):
        buf[i % 5] ^= key[i]
    return ARC4(bytes(buf))
