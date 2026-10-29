#!/usr/bin/env python3
"""Regression tests for the corrected OutGuess 0.2 cracker in stegdetect-py.

These are *self-consistency round-trip* tests.  They build a realistic LSB
bitstream from a decoded JPEG, embed a known message with a known password
using a reference OutGuess 0.2 embedder (mirroring og_main.c steg_embed), and
then verify the fixed ``crack_outguess`` recovers it.  They also assert that a
wrong password and a clean image do NOT produce a false positive.

They guard against regressions of the two original defects that were fixed in
``stegbreak.py``:

  1. ARC4 key derivation must be MD5("Encryption"/"Seeding" + password),
     not MD5(password).
  2. The 4-byte header (seed:u16 LE, length:u16 LE) is read at the
     iterator-selected bit positions, not at the head of the LSB stream.

Notes on the OutGuess 0.2 algorithm that affect this test:

  * OutGuess spreads the embedded data across the WHOLE bitmap (the iterator
    ``adapt`` step size is proportional to ``bits / datalen``), so the final
    position lands near the end of the stream.  Because of the random walk
    there is a small chance the iterator runs past the end of the bitstream
    before all bytes are written.  We therefore search a few RNG seeds for one
    on which the embed completes; this keeps the test green across
    environments (different Pillow versions can change the exact DCT
    coefficients) while still exercising the real algorithm.
  * OutGuess' Golay(23,12) ECC packing stores the trailing padding count in the
    last byte.  When the message length is an exact multiple of 3 no padding is
    added, so the last byte is real data and the decoder rejects it.  We use a
    message length that is NOT a multiple of 3 for the ECC case.
"""

import io
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stegdetect.jpeg_dct import read_jpeg_bytes
from stegdetect.stegbreak import (
    _og_arc4,
    _OGIterator,
    _og_enc_len,
    _og_decode_data,
    _STEG_ERROR,
    _OG_ENC,
    _og_build_bitseq,
    _og_dezig,
    _ZZ,
    _og_extract,
    OutguessTarget,
    crack_outguess,
)

PASSWORD = b"correct_horse_battery_staple"
# 35 bytes -- deliberately NOT a multiple of 3 (see module note on ECC).
MESSAGE = b"flag{outguess_cracker_regression_o}"
assert len(MESSAGE) % 3 != 0
SEED_START = 12345
IMAGE_SIZE = 192


def _make_sample_jpeg_bytes(rng_seed: int) -> bytes:
    """Generate a deterministic baseline JPEG (no steganography)."""
    rng = np.random.default_rng(rng_seed)
    arr = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(
        buf, format="JPEG", quality=95, subsampling=0
    )
    return buf.getvalue()


def golay_encode(data_bytes: bytes) -> bytes:
    """Inverse of ``_og_decode_data``'s ECC branch (12 data bits -> 23 code bits).

    Padding matches OutGuess: append ``(3 - len%3) % 3`` bytes of ``[0, 1, ...]``
    so the decoder can recover the original length from the trailing count byte.
    """
    length = len(data_bytes)
    pad = (3 - (length % 3)) % 3
    db = bytearray(data_bytes) + bytes(range(pad))
    bits = []
    for byte in db:
        for i in range(8):
            bits.append((byte >> i) & 1)
    codebits = []
    for g in range(0, len(bits), 12):
        word = 0
        for i in range(12):
            word |= bits[g + i] << i
        cw = _OG_ENC[word]
        for i in range(23):
            codebits.append((cw >> i) & 1)
    while len(codebits) % 8:  # pad codebits to a whole number of bytes
        codebits.append(0)
    out = bytearray()
    for i in range(0, len(codebits), 8):
        byte = 0
        for j in range(8):
            byte |= codebits[i + j] << j
        out.append(byte)
    return bytes(out)


def _embed_into_bitseq(bitseq, password: bytes, message: bytes, flags: int, seed: int = 123):
    """Mirror og_main.c steg_embed onto an in-memory bitstream copy.

    Returns ``(bitseq, ok)`` where ``ok`` is False if the iterator ran past the
    end of the stream before all bytes were written (OutGuess' full-bitmap
    spread can overshoot on some images).
    """
    bitseq = list(bitseq)
    nbits = len(bitseq)
    it = _OGIterator(nbits, password)
    as_enc = _og_arc4("Encryption", password)

    if flags & _STEG_ERROR:
        body = golay_encode(message)
        admin = bytes([seed & 0xFF, (seed >> 8) & 0xFF, len(body) & 0xFF, (len(body) >> 8) & 0xFF])
        admin = golay_encode(admin)
    else:
        body = message
        admin = bytes([seed & 0xFF, (seed >> 8) & 0xFF, len(message) & 0xFF, (len(message) >> 8) & 0xFF])

    # The header bytes are encrypted with the Encryption ARC4 stream before
    # being written (OutGuess embeds the cipher bytes, not the plaintext).
    enc_admin = bytearray(admin)
    for j in range(len(enc_admin)):
        enc_admin[j] ^= as_enc.get_byte()
    for i in range(len(enc_admin)):
        for k in range(8):
            if it.off >= nbits:
                return bitseq, False
            bitseq[it.off] = (enc_admin[i] >> k) & 1
            it.next()

    it.seed(seed)
    as_enc2 = _og_arc4("Encryption", password)
    m = bytearray(body)
    for j in range(len(m)):
        m[j] ^= as_enc2.get_byte()
    remaining = len(body)
    for i in range(len(body)):
        it.adapt(remaining)
        for k in range(8):
            if it.off >= nbits:
                return bitseq, False
            bitseq[it.off] = (m[i] >> k) & 1
            it.next()
        remaining -= 1
    return bitseq, True


def _inject_bits_into_info(info, order: str, natural: bool, skip_minus1: bool, bitseq):
    """Write ``bitseq`` back into ``info``'s coefficients.

    Uses the EXACT same coefficient traversal as ``_og_build_bitseq`` so that
    rebuilding the bitstream from ``info`` reproduces ``bitseq`` exactly.
    """
    comps = info.components
    y = comps[0]
    n_mcu_x = (y.blocks_x + y.h_factor - 1) // y.h_factor
    n_mcu_y = (y.blocks_y + y.v_factor - 1) // y.v_factor

    def skip(v: int) -> bool:
        if skip_minus1:
            return v in (0, 1, -1)
        return (v & 1) == v  # v is 0 or 1  (-1's LSB is 1, so it is kept)

    pos = 0
    n = len(bitseq)

    def block_bits(c, by, bx):
        nonlocal pos
        vec = list(c.coeffs[by, bx, :].tolist())
        nat = _og_dezig(vec) if natural else vec
        for idx in range(64):
            if not skip(int(nat[idx])):
                if pos >= n:
                    return
                v = int(nat[idx])
                b = bitseq[pos]
                if b == 0:
                    if v & 1:
                        v -= 1
                else:
                    if not (v & 1):
                        v += 1
                nat[idx] = v
                pos += 1
        # write back to the stored (zigzag) layout
        if natural:
            new_vec = [0] * 64
            for z in range(64):
                new_vec[z] = nat[_ZZ[z]]
            vec = new_vec
        c.coeffs[by, bx, :] = vec

    if order == "component":
        for c in comps:
            for by in range(c.blocks_y):
                for bx in range(c.blocks_x):
                    block_bits(c, by, bx)
    else:  # mcu interleaved scan order
        for my in range(n_mcu_y):
            for mx in range(n_mcu_x):
                for c in comps:
                    for by in range(c.v_factor):
                        for bx in range(c.h_factor):
                            block_bits(c, my * c.v_factor + by, mx * c.h_factor + bx)


def _build_embedded_target(rng_seed: int, flags: int):
    """Decode a sample JPEG, embed MESSAGE, and return an OutguessTarget.

    Searches a few seeds until the OutGuess full-bitmap spread fits (see the
    module note).  ``flags`` selects ECC vs no-ECC embedding.
    """
    order, natural, skip_minus1 = "component", True, False
    for seed in range(SEED_START, SEED_START + 64):
        data = _make_sample_jpeg_bytes(seed)
        info = read_jpeg_bytes(data)
        bitseq = _og_build_bitseq(info, order, natural, skip_minus1)
        embedded, ok = _embed_into_bitseq(bitseq, PASSWORD, MESSAGE, flags, seed=123)
        if not ok:
            continue
        info2 = read_jpeg_bytes(data)
        _inject_bits_into_info(info2, order, natural, skip_minus1, embedded)
        return OutguessTarget(filename="sample.jpg", bits=len(embedded), info=info2)
    raise RuntimeError("could not find an embedding seed that fits the bitmap")


def _build_clean_target(rng_seed: int):
    """A target built from an unmodified decoded JPEG (no embedded data)."""
    data = _make_sample_jpeg_bytes(rng_seed)
    info = read_jpeg_bytes(data)
    bitseq = _og_build_bitseq(info, "component", True, False)
    return OutguessTarget(filename="sample.jpg", bits=len(bitseq), info=info)


# Build the shared fixtures once.
_NOS_SEED = SEED_START
for _s in range(SEED_START, SEED_START + 64):
    if _embed_into_bitseq(
        _og_build_bitseq(read_jpeg_bytes(_make_sample_jpeg_bytes(_s)), "component", True, False),
        PASSWORD, MESSAGE, 0, seed=123,
    )[1]:
        _NOS_SEED = _s
        break

TARGET_NO_ECC = _build_embedded_target(_NOS_SEED, 0)
TARGET_ECC = _build_embedded_target(_NOS_SEED, _STEG_ERROR)
TARGET_CLEAN = _build_clean_target(_NOS_SEED)


def test_no_ecc_roundtrip():
    """Correct password must recover the embedded no-ECC message."""
    res = crack_outguess(TARGET_NO_ECC, PASSWORD.decode())
    assert res is not None
    assert res.success is True
    assert res.data == MESSAGE


def test_ecc_roundtrip():
    """Correct password must recover the embedded ECC-protected message."""
    res = crack_outguess(TARGET_ECC, PASSWORD.decode())
    assert res is not None
    assert res.success is True
    assert res.data == MESSAGE


def test_ecc_decode_unit():
    """_og_extract must decode an ECC bitstream directly (unit check)."""
    bitseq = _og_build_bitseq(TARGET_ECC.info, "component", True, False)
    embedded, ok = _embed_into_bitseq(bitseq, PASSWORD, MESSAGE, _STEG_ERROR, seed=123)
    assert ok
    out, (seed, enclen) = _og_extract(embedded, PASSWORD, _STEG_ERROR)
    assert out == MESSAGE


def test_wrong_password_no_false_positive():
    """A wrong password must NOT produce a successful crack on the payload."""
    res = crack_outguess(TARGET_NO_ECC, "definitely_the_wrong_password")
    assert res is None or res.success is False


def test_clean_image_no_false_positive():
    """The clean (unmodified) image must not high-confidence match any password."""
    for pw in (PASSWORD.decode(), "another_guess", "123456"):
        res = crack_outguess(TARGET_CLEAN, pw)
        assert res is None or res.success is False


if __name__ == "__main__":
    tests = [
        test_no_ecc_roundtrip,
        test_ecc_roundtrip,
        test_ecc_decode_unit,
        test_wrong_password_no_false_positive,
        test_clean_image_no_false_positive,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
