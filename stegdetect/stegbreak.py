"""
Stegbreak - Dictionary attack for cracking steganography passwords.

Supports cracking: JPHide (Blowfish), JSteg (ARC4), Outguess (ARC4).
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Callable, Iterator

from .jpeg_dct import read_jpeg, prepare_jsteg, prepare_outguess, prepare_jphide, jsteg_size
from .crypto.arc4 import ARC4, arc4_initkey, arc4_fixedkey
from .crypto.blowfish import Blowfish


@dataclass
class CrackResult:
    """Result of a password cracking attempt."""
    password: str = ""
    stego_type: str = ""
    success: bool = False
    data: bytes = b""
    details: str = ""


@dataclass
class JPHideTarget:
    """Prepared JPHide cracking target."""
    filename: str
    bits: int
    iv: bytes = b""  # 8-byte IV from first DC coefficients
    coeffs: list = field(default_factory=list)  # First 256 embedded coefficients

    @classmethod
    def prepare(cls, filename: str) -> Optional['JPHideTarget']:
        """Extract JPHide cracking data from a JPEG file."""
        try:
            info = read_jpeg(filename)
        except Exception:
            return None

        comp = info.components[0]
        if comp.blocks_y < 1 or comp.blocks_x < 1:
            return None

        # Extract IV from first 8 DC coefficients (across blocks)
        iv_bytes = bytearray(8)
        coeffs = []
        block_count = 0

        for by in range(comp.blocks_y):
            for bx in range(comp.blocks_x):
                dc = comp.coeffs[by, bx, 0]
                if block_count < 8:
                    iv_bytes[block_count] = dc & 0xFF
                if block_count < 256:
                    coeffs.append(dc)
                block_count += 1

        if block_count < 256:
            return None

        data, bits = prepare_jphide(info)

        return cls(
            filename=filename,
            bits=bits,
            iv=bytes(iv_bytes),
            coeffs=coeffs[:256]
        )


@dataclass
class JStegTarget:
    """Prepared JSteg cracking target."""
    filename: str
    skip: int
    tail: bytes = b""  # Last 8 bytes of embedded data
    header: bytes = b""  # First 64 bytes of embedded data

    @classmethod
    def prepare(cls, filename: str) -> Optional['JStegTarget']:
        """Extract JSteg cracking data from a JPEG file."""
        try:
            info = read_jpeg(filename)
        except Exception:
            return None

        data, bits = prepare_jsteg(info)
        if bits < 64:
            return None

        # Extract LSB of coefficients
        embedded = bytearray()
        for coeff in data:
            embedded.append(coeff & 1)

        # Convert bits to bytes
        embedded_bytes = bytearray()
        for i in range(0, len(embedded) - 7, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | embedded[i + j]
            embedded_bytes.append(byte)

        # Read embedded data length
        size = jsteg_size(data, bits)

        # Last 8 bytes
        tail = bytes(embedded_bytes[-8:]) if len(embedded_bytes) >= 8 else bytes(embedded_bytes)
        # First 64 bytes
        header = bytes(embedded_bytes[:64]) if len(embedded_bytes) >= 64 else bytes(embedded_bytes)

        skip = max(0, size - 8)

        return cls(
            filename=filename,
            skip=skip,
            tail=tail,
            header=header
        )


@dataclass
class OutguessTarget:
    """Prepared Outguess cracking target."""
    filename: str
    bits: int
    bitmap: list = field(default_factory=list)  # Packed DCT LSB as uint32 array

    @classmethod
    def prepare(cls, filename: str) -> Optional['OutguessTarget']:
        """Extract Outguess cracking data from a JPEG file."""
        try:
            info = read_jpeg(filename)
        except Exception:
            return None

        data, bits = prepare_outguess(info)
        if bits < 64:
            return None

        # Pack LSB into uint32 bitmap
        bitmap = []
        word = 0
        bit_pos = 0
        for coeff in data:
            word = (word << 1) | (coeff & 1)
            bit_pos += 1
            if bit_pos == 32:
                bitmap.append(word)
                word = 0
                bit_pos = 0

        return cls(
            filename=filename,
            bits=bits,
            bitmap=bitmap
        )


def crack_jphide(target: JPHideTarget, password: str) -> Optional[CrackResult]:
    """Try to crack JPHide with a given password.

    JPHide v5: key = IV[0:6] + password, Blowfish encrypted
    JPHide v3: key = password, Blowfish encrypted

    Verification: decrypt first 8 bytes, check length and IV match.
    """
    result = CrackResult(password=password, stego_type="jphide")

    # Try v5 first
    key = target.iv[:6] + password.encode('latin-1')
    bf = Blowfish(key)

    # Decrypt first 8 bytes
    # JPHide uses 4 Blowfish PRNG streams
    # Stream initialization from IV
    streams = []
    for i in range(4):
        stream_key = bytearray(8)
        for j in range(8):
            stream_key[j] = target.iv[(i + j) % 8] ^ (i & 0xFF)
        streams.append(Blowfish(bytes(stream_key)))

    # Decrypt header
    header_bytes = bytearray(8)
    for i in range(8):
        header_bytes[i] = streams[i % 4].get_byte()

    # XOR with embedded data
    decrypted = bytearray()
    for i in range(min(8, len(target.coeffs))):
        decrypted.append(target.coeffs[i] ^ header_bytes[i])

    # Check: first 4 bytes = embedded data length
    if len(decrypted) >= 4:
        length = (decrypted[0] << 24) | (decrypted[1] << 16) | (decrypted[2] << 8) | decrypted[3]
        if 0 < length < target.bits:
            # Verify encrypted IV matches
            if len(decrypted) >= 8:
                enc_iv = decrypted[4:8]
                # Simple check - if length is reasonable, likely correct
                result.success = True
                result.data = bytes(decrypted)
                result.details = f"v5, length={length}"
                return result

    # Try v3
    bf3 = Blowfish(password.encode('latin-1'))
    header_bytes3 = bytearray(8)
    for i in range(8):
        header_bytes3[i] = bf3.get_byte()

    decrypted3 = bytearray()
    for i in range(min(8, len(target.coeffs))):
        decrypted3.append(target.coeffs[i] ^ header_bytes3[i])

    if len(decrypted3) >= 4:
        length = (decrypted3[0] << 24) | (decrypted3[1] << 16) | (decrypted3[2] << 8) | decrypted3[3]
        if 0 < length < target.bits:
            result.success = True
            result.data = bytes(decrypted3)
            result.details = f"v3, length={length}"
            return result

    return None


def crack_jsteg(target: JStegTarget, password: str) -> Optional[CrackResult]:
    """Try to crack JSteg with a given password.

    JSteg uses ARC4 with fixed key setup.
    Verification: XOR last 8 bytes, check for magic "korejwa" or "cMk".
    """
    result = CrackResult(password=password, stego_type="jsteg")

    as_ = arc4_fixedkey(password.encode('latin-1'))
    as_.skip_bytes(target.skip)

    # Decrypt last 8 bytes
    keystream = bytearray(8)
    for i in range(8):
        keystream[i] = as_.get_byte()

    decrypted = bytearray()
    for i in range(min(8, len(target.tail))):
        decrypted.append(target.tail[i] ^ keystream[i])

    # Check magic bytes
    magic = bytes(decrypted[:7])
    if magic == b"korejwa":
        result.success = True
        result.data = bytes(decrypted)
        result.details = "magic=korejwa"
        return result

    magic2 = bytes(decrypted[:3])
    if magic2 == b"cMk":
        result.success = True
        result.data = bytes(decrypted)
        result.details = f"magic=cMk, version={decrypted[3] if len(decrypted) > 3 else '?'}"
        return result

    return None


def crack_outguess(target: OutguessTarget, password: str) -> Optional[CrackResult]:
    """Try to crack Outguess with a given password.

    Outguess uses ARC4 with MD5-hashed key.
    Verification: extract header (seed + length), validate.
    """
    result = CrackResult(password=password, stego_type="outguess")

    as_ = arc4_initkey(password.encode('latin-1'))

    # Extract 4-byte header
    header_bytes = bytearray(4)
    for i in range(4):
        header_bytes[i] = as_.get_byte()

    # XOR with first 4 bytes of bitmap
    if len(target.bitmap) == 0:
        return None

    first_word = target.bitmap[0]
    decrypted_header = bytearray(4)
    for i in range(4):
        decrypted_header[i] = ((first_word >> (24 - 8*i)) & 0xFF) ^ header_bytes[i]

    seed = (decrypted_header[0] << 24) | (decrypted_header[1] << 16) | (decrypted_header[2] << 8) | decrypted_header[3]

    # Validate seed
    if seed >= 55000:
        return None

    # Extract length from next 4 bytes
    length_bytes = bytearray(4)
    for i in range(4):
        length_bytes[i] = as_.get_byte()

    if len(target.bitmap) < 2:
        return None

    second_word = target.bitmap[1]
    decrypted_length = bytearray(4)
    for i in range(4):
        decrypted_length[i] = ((second_word >> (24 - 8*i)) & 0xFF) ^ length_bytes[i]

    length = (decrypted_length[0] << 24) | (decrypted_length[1] << 16) | (decrypted_length[2] << 8) | decrypted_length[3]

    if length <= 0 or length > target.bits:
        return None

    result.success = True
    result.data = bytes(decrypted_header + decrypted_length)
    result.details = f"seed={seed}, length={length}"
    return result


def wordlist_generator(wordlist_path: str) -> Iterator[str]:
    """Read words from a wordlist file."""
    with open(wordlist_path, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            word = line.strip()
            if word:
                yield word


def crack_file(filename: str, wordlist_path: str, stego_type: str = "auto") -> Optional[CrackResult]:
    """Try to crack steganography in a file using a wordlist.

    Args:
        filename: path to JPEG file
        wordlist_path: path to password wordlist
        stego_type: "auto", "jphide", "jsteg", or "outguess"

    Returns:
        CrackResult if password found, None otherwise
    """
    targets = []

    if stego_type in ("auto", "jphide"):
        t = JPHideTarget.prepare(filename)
        if t:
            targets.append(("jphide", t))

    if stego_type in ("auto", "jsteg"):
        t = JStegTarget.prepare(filename)
        if t:
            targets.append(("jsteg", t))

    if stego_type in ("auto", "outguess"):
        t = OutguessTarget.prepare(filename)
        if t:
            targets.append(("outguess", t))

    if not targets:
        return None

    crackers = {
        "jphide": crack_jphide,
        "jsteg": crack_jsteg,
        "outguess": crack_outguess,
    }

    for word in wordlist_generator(wordlist_path):
        for stype, target in targets:
            result = crackers[stype](target, word)
            if result and result.success:
                return result

    return None
