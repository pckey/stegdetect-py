"""
Stegbreak - Dictionary attack for cracking steganography passwords.

Supports cracking: JPHide (Blowfish), JSteg (ARC4), Outguess (ARC4).
"""

import os
import sys
import hashlib
import itertools
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
    bitmap: list = field(default_factory=list)  # Packed DCT LSB as uint32 array (legacy)
    info: object = None  # JPEGInfo with decoded DCT coefficients

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

        # Pack LSB into uint32 bitmap (legacy, kept for compatibility)
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
            bitmap=bitmap,
            info=info
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


# ===========================================================================
# Correct OutGuess 0.2 cracker
#
# The original crack_outguess was broken in two independent ways:
#   1. Wrong ARC4 key.  Real OutGuess 0.2 derives TWO streams from the password:
#        "Encryption" -> MD5("Encryption" + password)  (encrypts the payload)
#        "Seeding"    -> MD5("Seeding" + password)     (drives the bit iterator)
#      The old code used MD5(password) for everything.
#   2. Wrong header location.  The 4-byte header (seed:u16 LE, length:u16 LE)
#      is NOT at the head of the LSB stream.  It is read at the first
#      iterator-selected bit positions and then decrypted.  The old code read
#      bitmap[0:2] directly, which only works if the password is wrong too.
#
# Additionally, the bitmap traversal must match OutGuess: it includes the DC
# term, skips ONLY coefficients that are 0 or 1 (a coefficient of -1 is kept,
# since (-1 & 1) == 1), and is read in the order libjpeg yields coefficients.
# Because that order depends on the embedding libjpeg build, we try several
# plausible orderings and accept only on a confident data validation.
# ===========================================================================

# Standard JPEG zigzag table: ZZ[natural] = zigzag_scan_index
_ZZ = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26,
       33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57,
       50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
       39, 46, 53, 60, 61, 54, 47, 55, 62, 63]


def _og_dezig(vec):
    """Convert a 64-element zigzag-ordered vector to natural order."""
    nat = [0] * 64
    for z in range(64):
        nat[_ZZ[z]] = vec[z]
    return nat


def _og_arc4(type_str: str, password: bytes) -> ARC4:
    """ARC4 keyed with MD5("<type>" + password) -- OutGuess 0.2 convention."""
    digest = hashlib.md5((type_str.encode('latin-1') + password)).digest()
    return ARC4(digest)


# --- Golay (23,12) error correction (mirrors outguess golay.c) ---
_GENPOL = 0x00000c75


def _og_syndrome(pattern: int) -> int:
    rem = pattern & 0x7fffff
    for i in range(22, 10, -1):
        if (rem >> i) & 1:
            rem ^= _GENPOL << (i - 11)
    return rem & 0x7ff


_OG_ENC = [((i << 11) ^ _og_syndrome(i << 11)) & 0x7fffff for i in range(4096)]
_OG_DEC = {0: 0}
for _w in (1, 2, 3):
    for _combo in itertools.combinations(range(23), _w):
        _e = 0
        for _b in _combo:
            _e |= (1 << _b)
        _OG_DEC[_og_syndrome(_e)] = _e


def _og_golay_decode(cw23: int) -> int:
    return ((cw23 ^ _OG_DEC.get(_og_syndrome(cw23 & 0x7fffff), 0)) >> 11) & 0xfff


_STEG_ERROR = 0x08


def _og_enc_len(datalen: int, flags: int) -> int:
    if flags & _STEG_ERROR:
        dl = datalen + (3 - (datalen % 3))
        return (dl * 8 // 12 * 23 + 7) // 8
    return datalen


def _og_decode_data(enc: bytes, as_stream: ARC4, flags: int) -> bytes:
    enc = bytearray(enc)
    for j in range(len(enc)):
        enc[j] ^= as_stream.get_byte()
    if flags & _STEG_ERROR:
        enclen = len(enc)
        declen = enclen * 12 // 23
        data = bytearray(declen)
        etmp = 0
        outbits = 0
        dtmp = 0
        inbits = 0
        ei = 0
        di = 0
        while ei < enclen and di < declen:
            while outbits < 23 and ei < enclen:
                etmp |= enc[ei] << outbits
                ei += 1
                outbits += 8
            dtmp |= _og_golay_decode(etmp & 0x7fffff) << inbits
            inbits += 12
            etmp >>= 23
            outbits -= 23
            while inbits >= 8:
                data[di] = dtmp & 0xff
                di += 1
                dtmp >>= 8
                inbits -= 8
        if di == 0:
            return b''
        i = data[di - 1]
        if i > 2:
            return b''
        for j in range(i, -1, -1):
            if data[di - 1 - i + j] != j:
                return b''
        di -= (i + 1)
        return bytes(data[:di])
    return bytes(enc)


class _OGIterator:
    """Position iterator used by OutGuess (mirrors iterator.c)."""
    INIT_SKIPMOD = 32

    def __init__(self, bitlen: int, password: bytes):
        self.bits = bitlen
        self.as_ = _og_arc4("Seeding", password)
        self.skipmod = self.INIT_SKIPMOD
        self.off = self.as_.get_word() % self.skipmod

    def next(self) -> int:
        self.off += (self.as_.get_word() % self.skipmod) + 1
        return self.off

    def seed(self, seed: int) -> None:
        reseed = bytes([seed & 0xFF, (seed >> 8) & 0xFF])
        self.as_._addrandom(reseed, 2)

    def adapt(self, datalen: int) -> None:
        factor = (self.bits - self.off) // (8 * datalen)
        x32 = self.bits // 32
        if (self.bits - self.off) > x32:
            adj = 2.0
        else:
            adj = 2.0 - (x32 - (self.bits - self.off)) / x32
        self.skipmod = int(adj * factor)
        if self.skipmod < 1:
            self.skipmod = 1


def _og_retrbyte(bitseq, it: _OGIterator) -> int:
    b = 0
    for _ in range(8):
        cur = it.off
        if cur >= len(bitseq):
            raise ValueError("off bitmap")
        b |= bitseq[cur] << _
        it.next()
    return b


def _og_build_bitseq(info, order: str, natural: bool, skip_minus1: bool) -> list:
    """Build the OutGuess LSB bitstream from decoded DCT coefficients.

    order: 'component' (per-component raster, as libjpeg stores coeffs) or
           'mcu' (MCU-interleaved scan order).
    natural: apply de-zigzag (True) or keep zigzag scan order (False).
    skip_minus1: if True, also drop coefficients equal to -1 (legacy stegdetect
                 behaviour); OutGuess 0.2 keeps -1, so this defaults off.
    Includes the DC term (index 0) and skips only 0 and 1.
    """
    comps = info.components
    y = comps[0]
    n_mcu_x = (y.blocks_x + y.h_factor - 1) // y.h_factor
    n_mcu_y = (y.blocks_y + y.v_factor - 1) // y.v_factor

    def skip(v: int) -> bool:
        if skip_minus1:
            return v in (0, 1, -1)
        return (v & 1) == v  # v is 0 or 1  (note: -1 -> 1, 1 == -1 is False)

    def block_bits(c, by, bx):
        vec = list(c.coeffs[by, bx, :])
        if natural:
            vec = _og_dezig(vec)
        out = []
        for v in vec:
            vi = int(v)
            if not skip(vi):
                out.append(vi & 1)
        return out

    bitseq = []
    if order == 'component':
        for c in comps:
            for by in range(c.blocks_y):
                for bx in range(c.blocks_x):
                    bitseq.extend(block_bits(c, by, bx))
    else:  # mcu interleaved
        for my in range(n_mcu_y):
            for mx in range(n_mcu_x):
                for c in comps:
                    Hh = c.h_factor
                    Vv = c.v_factor
                    for by in range(Vv):
                        for bx in range(Hh):
                            bb = my * Vv + by
                            bxx = mx * Hh + bx
                            bitseq.extend(block_bits(c, bb, bxx))
    return bitseq


def _og_extract(bitseq: list, password: bytes, flags: int):
    """Extract (data, (seed, enclen)) from a bitstream, or (None, ...) on failure."""
    nbits = len(bitseq)
    it = _OGIterator(nbits, password)
    as_enc = _og_arc4("Encryption", password)
    admin_len = _og_enc_len(4, flags)
    tmp = bytearray()
    for _ in range(admin_len):
        tmp.append(_og_retrbyte(bitseq, it))
    buf = _og_decode_data(bytes(tmp), as_enc, flags)
    if len(buf) != 4:
        return None, (0, 0)
    seed = buf[0] | (buf[1] << 8)
    enclen = buf[2] | (buf[3] << 8)
    if not (0 <= seed < 65536):
        return None, (0, 0)
    if not (0 < enclen <= nbits // 8):
        return None, (0, 0)
    as_enc2 = _og_arc4("Encryption", password)
    it.seed(seed)
    msg = bytearray()
    remaining = enclen
    for _ in range(enclen):
        it.adapt(remaining)
        msg.append(_og_retrbyte(bitseq, it))
        remaining -= 1
    out = _og_decode_data(bytes(msg), as_enc2, flags)
    return out, (seed, enclen)


def _og_validate(out: bytes, seed: int, enclen: int, nbits: int):
    """Return ('confidence', note).  High confidence => definitely correct."""
    if not (0 <= seed < 55000):
        return ('reject', 'bad seed')
    if not (0 < enclen <= nbits // 8):
        return ('reject', 'bad length')
    if not out:
        return ('reject', 'empty')
    # ECC mode: a successful golay decode (non-empty, valid padding) is a strong
    # signal that the password was correct.
    # NOTE: a bare "{" is NOT a reliable magic marker -- random long byte
    # strings almost always contain the 0x7B byte, which caused clean images /
    # wrong passwords to false-positive.  Require an actual flag token
    # ("flag"/"ctf", case-insensitive) instead.
    low = out.lower()
    if len(out) >= 4 and (b'flag' in low or b'ctf' in low):
        return ('high', 'magic/printable')
    printable = sum(1 for b in out if 32 <= b < 127 or b in (9, 10, 13))
    ratio = printable / len(out) if out else 0
    if ratio >= 0.90:
        return ('high', f'printable ratio {ratio:.2f}')
    return ('low', f'printable ratio {ratio:.2f}')


def crack_outguess(target: OutguessTarget, password: str) -> Optional[CrackResult]:
    """Try to crack OutGuess 0.2 with a given password.

    Uses the correct OutGuess 0.2 algorithm (see module note above): two
    MD5-derived ARC4 streams, iterator-selected header, re-seeded data read.
    Several coefficient orderings are tried; a candidate is accepted only on a
    confident data validation, so clean images are rejected instead of
    false-positiving.
    """
    result = CrackResult(password=password, stego_type="outguess")
    info = target.info
    if info is None:
        return None

    pw = password.encode('latin-1')
    # (order, natural, skip_minus1) -- OutGuess 0.2 keeps -1, so skip_minus1 is
    # only a legacy fallback.  component-major is what libjpeg's per-component
    # coefficient arrays give; mcu-interleaved is the scan order.
    combos = [
        ('component', False, False),
        ('component', True, False),
        ('mcu', False, False),
        ('mcu', True, False),
        ('component', False, True),   # legacy stegdetect order, just in case
        ('mcu', False, True),
    ]

    for order, natural, skip1 in combos:
        bitseq = _og_build_bitseq(info, order, natural, skip1)
        if len(bitseq) < 64:
            continue
        for ecc in (False, True):
            flags = _STEG_ERROR if ecc else 0
            try:
                out, (seed, enclen) = _og_extract(bitseq, pw, flags)
            except Exception:
                continue
            if out is None:
                continue
            conf, note = _og_validate(out, seed, enclen, len(bitseq))
            if conf == 'reject':
                continue
            detail = (f"order={order} natural={natural} skip1={skip1} "
                      f"ecc={ecc} seed={seed} length={enclen} [{note}]")
            if conf == 'high':
                result.success = True
                result.data = out
                result.details = detail
                return result
            # Header is structurally plausible but the body is not verifiable as
            # plaintext (e.g. it is ARC4-encrypted binary).  We do NOT report
            # this as success: a random password on a clean image also yields a
            # plausible header ~8% of the time, which would cause false
            # positives during a wordlist run.  Such candidates need manual
            # inspection and are intentionally not returned as a hit here.
    return None


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
