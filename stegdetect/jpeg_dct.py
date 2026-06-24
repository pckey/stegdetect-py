"""
jpeg_dct.py - JPEG DCT coefficient extraction module.

Extracts DCT coefficients from JPEG files by parsing the JPEG bitstream.
Supports baseline JPEG (SOF0) with Huffman coding.

This module implements:
- JPEG marker parsing (SOI, APP0, APP1, COM, DQT, SOF0, DHT, SOS)
- Huffman table decoding
- DCT coefficient decoding from entropy-coded segments
- Zigzag order handling
- MCU structure handling (interleaved vs non-interleaved scans)
- Coefficient preparation for various steganography detectors
"""

import struct
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# JPEG marker constants
# ---------------------------------------------------------------------------

# Marker codes
SOI = 0xD8  # Start of Image
APP0 = 0xE0  # JFIF
APP1 = 0xE1  # EXIF
COM = 0xFE  # Comment
DQT = 0xDB  # Define Quantization Table
SOF0 = 0xC0  # Start of Frame (Baseline DCT)
SOF1 = 0xC1  # Start of Frame (Extended Sequential DCT)
SOF2 = 0xC2  # Start of Frame (Progressive DCT)
DHT = 0xC4  # Define Huffman Table
SOS = 0xDA  # Start of Scan
DRI = 0xDD  # Define Restart Interval
RST0 = 0xD0  # Restart markers 0..7
RST7 = 0xD7
EOI = 0xD9  # End of Image

# Zigzag scan order for 8x8 block
# Maps zigzag index -> (row, col) in the 8x8 block
ZIGZAG_ORDER = [
    (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
    (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
    (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
    (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
    (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
    (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
    (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7),
]

# Inverse zigzag: maps (row, col) -> zigzag index
INV_ZIGZAG = [0] * 64
for _i, (_r, _c) in enumerate(ZIGZAG_ORDER):
    INV_ZIGZAG[_r * 8 + _c] = _i

# ---------------------------------------------------------------------------
# JPHide lookup table
# ---------------------------------------------------------------------------

# JPHide uses a lookup table (ltab) that maps embedding positions to
# (zigzag_index, skip_pattern) pairs. The table defines which coefficient
# positions in each 8x8 block are used for data embedding.
#
# The original JPHide C code defines:
#   TAIL1 = 12000 (first 12000 positions use the main table)
#   TAIL2 = 1200  (next 1200 positions)
#   TAIL3 = 120   (last 120 positions)
#
# The main table entries specify (zigzag_index, skip_mask) where skip_mask
# determines which subsequent blocks to skip.

# Main lookup table: list of (zigzag_index, skip_count)
# zigzag_index: position in the 8x8 block (in zigzag order, 0-63)
# skip_count: number of blocks to skip after using this position
# A skip_count of 0 means use the next block normally.
# A skip_count of 1 means skip 1 block, etc.
# The value 255 means "end of table entry list for this position cycle".
#
# This is a reconstruction based on the JPHide algorithm.
# The table cycles through coefficient positions, preferring lower-frequency
# AC coefficients (zigzag indices 1-10 primarily), with occasional skips.

_JPHIDE_LTAB = [
    # (zigzag_index, skip_count)
    # Primary positions - low frequency AC coefficients
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
    (9, 0), (10, 0),
    # Secondary positions
    (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0),
    (19, 0), (20, 0),
    # Higher frequency
    (21, 0), (22, 0), (23, 0), (24, 0), (25, 0), (26, 0), (27, 0), (28, 0),
    (29, 0), (30, 0),
    (31, 0), (32, 0), (33, 0), (34, 0), (35, 0), (36, 0), (37, 0), (38, 0),
    (39, 0), (40, 0),
    # With occasional skips for security
    (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
    (6, 1), (7, 1), (8, 1), (9, 1), (10, 1),
    (11, 1), (12, 1), (13, 1), (14, 1), (15, 1),
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
    (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
    (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
    (16, 0), (17, 0), (18, 0), (19, 0), (20, 0),
    (21, 0), (22, 0), (23, 0), (24, 0), (25, 0),
    (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),
    (6, 2), (7, 2), (8, 2), (9, 2), (10, 2),
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
    (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
    (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
    (16, 0), (17, 0), (18, 0), (19, 0), (20, 0),
    (21, 0), (22, 0), (23, 0), (24, 0), (25, 0),
    (26, 0), (27, 0), (28, 0), (29, 0), (30, 0),
    (31, 0), (32, 0), (33, 0), (34, 0), (35, 0),
    (1, 3), (2, 3), (3, 3), (4, 3), (5, 3),
    (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
    (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
    (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
    (16, 0), (17, 0), (18, 0), (19, 0), (20, 0),
    (21, 0), (22, 0), (23, 0), (24, 0), (25, 0),
    (26, 0), (27, 0), (28, 0), (29, 0), (30, 0),
    (31, 0), (32, 0), (33, 0), (34, 0), (35, 0),
    (36, 0), (37, 0), (38, 0), (39, 0), (40, 0),
    (41, 0), (42, 0), (43, 0), (44, 0), (45, 0),
    (1, 4), (2, 4), (3, 4), (4, 4), (5, 4),
    (6, 4), (7, 4), (8, 4), (9, 4), (10, 4),
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
    (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
    (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
    (16, 0), (17, 0), (18, 0), (19, 0), (20, 0),
    (21, 0), (22, 0), (23, 0), (24, 0), (25, 0),
    (26, 0), (27, 0), (28, 0), (29, 0), (30, 0),
    (31, 0), (32, 0), (33, 0), (34, 0), (35, 0),
    (36, 0), (37, 0), (38, 0), (39, 0), (40, 0),
    (41, 0), (42, 0), (43, 0), (44, 0), (45, 0),
    (46, 0), (47, 0), (48, 0), (49, 0), (50, 0),
    (51, 0), (52, 0), (53, 0), (54, 0), (55, 0),
    (1, 5), (2, 5), (3, 5), (4, 5), (5, 5),
    (6, 5), (7, 5), (8, 5), (9, 5), (10, 5),
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
    (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
    (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
    (16, 0), (17, 0), (18, 0), (19, 0), (20, 0),
    (21, 0), (22, 0), (23, 0), (24, 0), (25, 0),
    (26, 0), (27, 0), (28, 0), (29, 0), (30, 0),
    (31, 0), (32, 0), (33, 0), (34, 0), (35, 0),
    (36, 0), (37, 0), (38, 0), (39, 0), (40, 0),
    (41, 0), (42, 0), (43, 0), (44, 0), (45, 0),
    (46, 0), (47, 0), (48, 0), (49, 0), (50, 0),
    (51, 0), (52, 0), (53, 0), (54, 0), (55, 0),
    (56, 0), (57, 0), (58, 0), (59, 0), (60, 0),
    (61, 0), (62, 0), (63, 0),
]

# JPHide tail tables for different embedding density levels
_JPHIDE_TAIL1 = [
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
    (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
] * 1200  # 12000 entries

_JPHIDE_TAIL2 = [
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
    (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
] * 120  # 1200 entries

_JPHIDE_TAIL3 = [
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
    (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
] * 12  # 120 entries


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Component:
    """One color component (Y, Cb, Cr)."""

    def __init__(self):
        self.id: int = 0
        self.h_factor: int = 1  # horizontal sampling factor
        self.v_factor: int = 1  # vertical sampling factor
        self.quant_table_idx: int = 0
        self.blocks_x: int = 0  # number of 8x8 blocks horizontally
        self.blocks_y: int = 0  # number of 8x8 blocks vertically
        self.coeffs: np.ndarray = np.array([], dtype=np.int16)
        # shape: (blocks_y, blocks_x, 64)
        # coeffs[:,:,0] is DC, coeffs[:,:,1:64] are AC in zigzag order


class JPEGInfo:
    """Holds decoded JPEG DCT coefficients."""

    def __init__(self):
        self.width: int = 0
        self.height: int = 0
        self.num_components: int = 0
        self.quant_tables: list[np.ndarray] = []  # quantization tables (up to 4)
        self.components: list[Component] = []  # per-component data
        self.comment: bytes = b""  # JPEG comment


# ---------------------------------------------------------------------------
# Bitstream reader
# ---------------------------------------------------------------------------

class BitstreamReader:
    """Reads bits from a byte buffer, handling byte-stuffing (0xFF00)."""

    def __init__(self, data: bytes, offset: int = 0):
        self.data = data
        self.pos = offset
        self.bit_pos = 0  # bits consumed in current byte (0-7)
        self.current_byte = 0
        self._load_byte()

    def _load_byte(self):
        """Load the next byte, skipping stuffed 0x00 after 0xFF."""
        if self.pos >= len(self.data):
            self.current_byte = 0
            return
        self.current_byte = self.data[self.pos]
        self.pos += 1
        # Handle byte stuffing: if 0xFF followed by 0x00, skip the 0x00
        if self.current_byte == 0xFF and self.pos < len(self.data):
            next_byte = self.data[self.pos]
            if next_byte == 0x00:
                # Stuffed byte - consume the 0x00
                self.pos += 1
            elif next_byte >= RST0 and next_byte <= RST7:
                # Restart marker - not stuffed, but we should not consume it
                # here; the caller handles restart markers
                pass
            elif next_byte == EOI:
                # End of image
                pass

    def read_bit(self) -> int:
        """Read a single bit."""
        if self.bit_pos >= 8:
            self.bit_pos = 0
            self._load_byte()
        bit = (self.current_byte >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        return bit

    def read_bits(self, n: int) -> int:
        """Read n bits and return as an integer."""
        value = 0
        for _ in range(n):
            value = (value << 1) | self.read_bit()
        return value

    def align_to_byte(self):
        """Align to the next byte boundary."""
        if self.bit_pos > 0:
            self.bit_pos = 0
            self._load_byte()

    def get_position(self) -> int:
        """Get current byte position in the stream."""
        return self.pos


# ---------------------------------------------------------------------------
# Huffman tree
# ---------------------------------------------------------------------------

class HuffmanTable:
    """Huffman decoding table.

    Stores the Huffman codes for fast lookup. Uses a lookup table approach
    for efficient decoding.
    """

    def __init__(self):
        # min_code[len] = minimum code of length len
        # max_code[len] = maximum code of length len
        # val_ptr[len] = index into values[] for codes of length len
        self.min_code = [0] * 16
        self.max_code = [0] * 16
        self.val_ptr = [0] * 16
        self.values = []  # symbol values in code order
        self.max_length = 0

    def build(self, counts: list[int], values: list[int]):
        """Build Huffman table from code lengths and symbol values.

        Args:
            counts: list of 16 integers, counts[i] = number of codes of length i+1
            values: list of symbol values, ordered by code length then lexicographic
        """
        self.values = values[:]
        self.max_length = 16

        # Build the code tables per JPEG spec (ITU-T T.81, Annex C)
        code = 0
        si = 0  # current code length index
        for l in range(1, 17):
            self.min_code[l - 1] = code
            self.val_ptr[l - 1] = si
            num = counts[l - 1]
            if num > 0:
                self.max_code[l - 1] = code + num - 1
                si += num
            else:
                self.max_code[l - 1] = -1
            code = (code + num) << 1

    def decode(self, reader: BitstreamReader) -> int:
        """Decode one Huffman symbol from the bitstream.

        Args:
            reader: BitstreamReader positioned at the start of a Huffman code

        Returns:
            The decoded symbol value.
        """
        code = 0
        for l in range(1, 17):
            code = (code << 1) | reader.read_bit()
            if code <= self.max_code[l - 1] and code >= self.min_code[l - 1]:
                idx = self.val_ptr[l - 1] + (code - self.min_code[l - 1])
                return self.values[idx]
        raise ValueError(
            f"Invalid Huffman code: code={code:#x}, "
            f"min={self.min_code}, max={self.max_code}"
        )


# ---------------------------------------------------------------------------
# JPEG DCT coefficient value decoder
# ---------------------------------------------------------------------------

def _extend(v: int, t: int) -> int:
    """Extend a partial bit value to a signed integer.

    For magnitude t, the value v is interpreted as:
    - If v < 2^(t-1): v - 2^t + 1 (negative)
    - Otherwise: v (positive)

    This follows the JPEG spec for decoding DCT coefficient values.
    """
    if t == 0:
        return 0
    vt = 1 << (t - 1)
    if v < vt:
        return v - (1 << t) + 1
    else:
        return v


# ---------------------------------------------------------------------------
# JPEG parser
# ---------------------------------------------------------------------------

class _JPEGParser:
    """Internal JPEG parser that extracts DCT coefficients."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.info = JPEGInfo()

        # Huffman tables: dc_tables[component] and ac_tables[component]
        self.dc_tables: dict[int, HuffmanTable] = {}
        self.ac_tables: dict[int, HuffmanTable] = {}

        # Quantization tables
        self.quant_tables_raw: dict[int, np.ndarray] = {}

        # Restart interval
        self.restart_interval: int = 0

    def parse(self) -> JPEGInfo:
        """Parse the JPEG file and extract all DCT coefficients."""
        # Verify SOI marker
        marker = self._read_marker()
        if marker != SOI:
            raise ValueError(f"Expected SOI marker (0xFFD8), got 0xFF{marker:02X}")

        # Parse markers
        while self.pos < len(self.data):
            marker = self._read_marker()
            if marker is None:
                break

            if marker == EOI:
                break
            elif marker == COM:
                self._parse_comment()
            elif marker == DQT:
                self._parse_dqt()
            elif marker in (SOF0, SOF1):
                self._parse_sof()
            elif marker == DHT:
                self._parse_dht()
            elif marker == SOS:
                self._parse_sos()
                break  # After SOS, the rest is entropy-coded data
            elif marker == DRI:
                self._parse_dri()
            elif marker in (APP0, APP1):
                self._skip_marker()
            elif marker >= 0xE0 and marker <= 0xEF:
                # APPn markers
                self._skip_marker()
            else:
                # Unknown marker, try to skip
                self._skip_marker()

        return self.info

    def _read_marker(self) -> Optional[int]:
        """Read the next JPEG marker. Returns marker code or None."""
        # Skip to next 0xFF
        while self.pos < len(self.data) and self.data[self.pos] != 0xFF:
            self.pos += 1
        if self.pos >= len(self.data):
            return None

        # Skip padding 0xFF bytes
        while self.pos < len(self.data) and self.data[self.pos] == 0xFF:
            self.pos += 1

        if self.pos >= len(self.data):
            return None

        marker = self.data[self.pos]
        self.pos += 1

        # Skip 0x00 after 0xFF (not a marker)
        if marker == 0x00:
            return self._read_marker()

        return marker

    def _read_u16(self) -> int:
        """Read a 16-bit big-endian unsigned integer."""
        val = struct.unpack(">H", self.data[self.pos:self.pos + 2])[0]
        self.pos += 2
        return val

    def _read_u8(self) -> int:
        """Read an 8-bit unsigned integer."""
        val = self.data[self.pos]
        self.pos += 1
        return val

    def _skip_marker(self):
        """Skip a marker segment by reading its length and skipping."""
        if self.pos + 2 > len(self.data):
            return
        length = self._read_u16()
        self.pos += length - 2  # length includes itself

    def _parse_comment(self):
        """Parse a COM (Comment) marker."""
        length = self._read_u16()
        comment_data = self.data[self.pos:self.pos + length - 2]
        self.pos += length - 2
        self.info.comment = comment_data

    def _parse_dqt(self):
        """Parse a DQT (Define Quantization Table) marker."""
        length = self._read_u16()
        end_pos = self.pos + length - 2

        while self.pos < end_pos:
            precision_and_id = self._read_u8()
            precision = (precision_and_id >> 4) & 0x0F  # 0 = 8-bit, 1 = 16-bit
            table_id = precision_and_id & 0x0F

            if precision == 0:
                # 8-bit quantization values
                raw = self.data[self.pos:self.pos + 64]
                self.pos += 64
                qtable = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
            else:
                # 16-bit quantization values
                raw = self.data[self.pos:self.pos + 128]
                self.pos += 128
                qtable = np.frombuffer(raw, dtype=np.uint16).astype(np.int16)

            # De-zigzag the quantization table
            table = np.zeros(64, dtype=np.int16)
            for i in range(64):
                r, c = ZIGZAG_ORDER[i]
                table[r * 8 + c] = qtable[i]

            self.quant_tables_raw[table_id] = table

        # Store quantization tables in JPEGInfo
        self.info.quant_tables = [None] * 4
        for tid in range(4):
            if tid in self.quant_tables_raw:
                self.info.quant_tables[tid] = self.quant_tables_raw[tid]

    def _parse_sof(self):
        """Parse a SOF (Start of Frame) marker."""
        length = self._read_u16()
        precision = self._read_u8()  # sample precision (usually 8)
        self.info.height = self._read_u16()
        self.info.width = self._read_u16()
        self.info.num_components = self._read_u8()

        self.info.components = []
        for _ in range(self.info.num_components):
            comp = Component()
            comp.id = self._read_u8()
            factors = self._read_u8()
            comp.h_factor = (factors >> 4) & 0x0F
            comp.v_factor = factors & 0x0F
            comp.quant_table_idx = self._read_u8()
            self.info.components.append(comp)

        # Calculate block dimensions for each component
        # MCU size is based on max sampling factors
        max_h = max(c.h_factor for c in self.info.components)
        max_v = max(c.v_factor for c in self.info.components)
        mcu_width = max_h * 8
        mcu_height = max_v * 8

        for comp in self.info.components:
            # Number of MCUs
            mcus_x = (self.info.width + mcu_width - 1) // mcu_width
            mcus_y = (self.info.height + mcu_height - 1) // mcu_height
            # Blocks per MCU for this component
            comp.blocks_x = mcus_x * comp.h_factor
            comp.blocks_y = mcus_y * comp.v_factor
            # Allocate coefficient array
            comp.coeffs = np.zeros(
                (comp.blocks_y, comp.blocks_x, 64), dtype=np.int16
            )

    def _parse_dht(self):
        """Parse a DHT (Define Huffman Table) marker."""
        length = self._read_u16()
        end_pos = self.pos + length - 2

        while self.pos < end_pos:
            class_and_id = self._read_u8()
            table_class = (class_and_id >> 4) & 0x0F  # 0=DC, 1=AC
            table_id = class_and_id & 0x0F

            # Read code counts for each length (1-16)
            counts = [self._read_u8() for _ in range(16)]
            total_symbols = sum(counts)

            # Read symbol values
            values = [self._read_u8() for _ in range(total_symbols)]

            # Build Huffman table
            htable = HuffmanTable()
            htable.build(counts, values)

            if table_class == 0:
                self.dc_tables[table_id] = htable
            else:
                self.ac_tables[table_id] = htable

    def _parse_dri(self):
        """Parse a DRI (Define Restart Interval) marker."""
        length = self._read_u16()
        self.restart_interval = self._read_u16()

    def _parse_sos(self):
        """Parse a SOS (Start of Scan) marker and decode the entropy-coded data."""
        length = self._read_u16()
        num_components = self._read_u8()

        # Read component selectors
        comp_selectors = []
        for _ in range(num_components):
            comp_id = self._read_u8()
            table_ids = self._read_u8()
            dc_table_id = (table_ids >> 4) & 0x0F
            ac_table_id = table_ids & 0x0F
            comp_selectors.append((comp_id, dc_table_id, ac_table_id))

        # Skip spectral selection and successive approximation
        _spectral_start = self._read_u8()
        _spectral_end = self._read_u8()
        _successive_approx = self._read_u8()

        # Now decode the entropy-coded segment
        self._decode_entropy_coded(comp_selectors)

    def _decode_entropy_coded(self, comp_selectors: list):
        """Decode the entropy-coded segment following SOS.

        Args:
            comp_selectors: list of (component_id, dc_table_id, ac_table_id)
        """
        # Create a bitstream reader starting at current position
        reader = BitstreamReader(self.data, self.pos)

        # Map component IDs to their indices
        comp_id_to_idx = {}
        for i, comp in enumerate(self.info.components):
            comp_id_to_idx[comp.id] = i

        # Get max sampling factors for MCU calculation
        max_h = max(c.h_factor for c in self.info.components)
        max_v = max(c.v_factor for c in self.info.components)

        # Calculate number of MCUs
        mcu_width = max_h * 8
        mcu_height = max_v * 8
        mcus_x = (self.info.width + mcu_width - 1) // mcu_width
        mcus_y = (self.info.height + mcu_height - 1) // mcu_height
        total_mcus = mcus_x * mcus_y

        # DC predictors for each component
        dc_pred = [0] * self.info.num_components

        # Block counters for each component (to track position in coefficient arrays)
        block_counters = [0] * self.info.num_components

        # Restart handling
        mcu_count = 0
        rst_count = 0

        # Determine if this is an interleaved scan (multiple components)
        interleaved = len(comp_selectors) > 1

        if interleaved:
            # Interleaved scan: decode MCUs with all components
            for mcu_idx in range(total_mcus):
                # Check for restart markers
                if self.restart_interval > 0:
                    if rst_count >= self.restart_interval:
                        # Handle restart marker
                        reader.align_to_byte()
                        # Skip restart marker bytes
                        if reader.pos < len(reader.data):
                            if reader.data[reader.pos] == 0xFF:
                                reader.pos += 1
                                if reader.pos < len(reader.data):
                                    rst_marker = reader.data[reader.pos]
                                    reader.pos += 1
                                    if RST0 <= rst_marker <= RST7:
                                        # Reset DC predictors
                                        dc_pred = [0] * self.info.num_components
                                        rst_count = 0
                                        reader.bit_pos = 0
                                        reader._load_byte()

                mcu_row = mcu_idx // mcus_x
                mcu_col = mcu_idx % mcus_x

                for sel_idx, (comp_id, dc_tid, ac_tid) in enumerate(comp_selectors):
                    comp_idx = comp_id_to_idx[comp_id]
                    comp = self.info.components[comp_idx]
                    dc_table = self.dc_tables[dc_tid]
                    ac_table = self.ac_tables[ac_tid]

                    # Decode all blocks for this component in the MCU
                    for v_block in range(comp.v_factor):
                        for h_block in range(comp.h_factor):
                            # Calculate block position
                            bx = mcu_col * comp.h_factor + h_block
                            by = mcu_row * comp.v_factor + v_block

                            if bx < comp.blocks_x and by < comp.blocks_y:
                                self._decode_block(
                                    reader, comp, bx, by,
                                    dc_table, ac_table, dc_pred, comp_idx
                                )

                rst_count += 1
                mcu_count += 1
        else:
            # Non-interleaved scan: decode one component at a time
            comp_id, dc_tid, ac_tid = comp_selectors[0]
            comp_idx = comp_id_to_idx[comp_id]
            comp = self.info.components[comp_idx]
            dc_table = self.dc_tables[dc_tid]
            ac_table = self.ac_tables[ac_tid]

            for by in range(comp.blocks_y):
                for bx in range(comp.blocks_x):
                    # Check for restart markers
                    if self.restart_interval > 0:
                        block_num = by * comp.blocks_x + bx
                        if block_num > 0 and block_num % self.restart_interval == 0:
                            reader.align_to_byte()
                            if reader.pos < len(reader.data):
                                if reader.data[reader.pos] == 0xFF:
                                    reader.pos += 1
                                    if reader.pos < len(reader.data):
                                        rst_marker = reader.data[reader.pos]
                                        reader.pos += 1
                                        if RST0 <= rst_marker <= RST7:
                                            dc_pred[comp_idx] = 0
                                            reader.bit_pos = 0
                                            reader._load_byte()

                    self._decode_block(
                        reader, comp, bx, by,
                        dc_table, ac_table, dc_pred, comp_idx
                    )

    def _decode_block(
        self,
        reader: BitstreamReader,
        comp: Component,
        bx: int,
        by: int,
        dc_table: HuffmanTable,
        ac_table: HuffmanTable,
        dc_pred: list[int],
        comp_idx: int,
    ):
        """Decode one 8x8 block of DCT coefficients.

        Args:
            reader: Bitstream reader
            comp: Component being decoded
            bx: Block x index
            by: Block y index
            dc_table: DC Huffman table
            ac_table: AC Huffman table
            dc_pred: DC predictor values (modified in place)
            comp_idx: Component index
        """
        # Decode DC coefficient
        dc_symbol = dc_table.decode(reader)
        dc_diff = _extend(reader.read_bits(dc_symbol), dc_symbol) if dc_symbol > 0 else 0
        dc_val = dc_pred[comp_idx] + dc_diff
        dc_pred[comp_idx] = dc_val

        # Store DC coefficient at zigzag index 0
        comp.coeffs[by, bx, 0] = dc_val

        # Decode AC coefficients
        zigzag_idx = 1
        while zigzag_idx < 64:
            ac_symbol = ac_table.decode(reader)
            if ac_symbol == 0x00:
                # EOB - all remaining coefficients are zero
                break

            # ac_symbol encodes (run, size) where:
            #   run = upper 4 bits = number of zero coefficients before this one
            #   size = lower 4 bits = number of bits for the coefficient value
            run = (ac_symbol >> 4) & 0x0F
            size = ac_symbol & 0x0F

            # Skip zero coefficients
            zigzag_idx += run

            if zigzag_idx >= 64:
                break

            # Read the coefficient value
            if size > 0:
                raw_val = reader.read_bits(size)
                coeff_val = _extend(raw_val, size)
                comp.coeffs[by, bx, zigzag_idx] = coeff_val

            zigzag_idx += 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_jpeg(filename: str) -> JPEGInfo:
    """Read a JPEG file and extract all DCT coefficients.

    Args:
        filename: Path to the JPEG file.

    Returns:
        JPEGInfo object containing image dimensions, quantization tables,
        and DCT coefficients for each component.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid JPEG.
    """
    with open(filename, "rb") as f:
        data = f.read()

    if len(data) < 2 or data[0] != 0xFF or data[1] != SOI:
        raise ValueError("Not a valid JPEG file (missing SOI marker)")

    parser = _JPEGParser(data)
    return parser.parse()


def read_jpeg_bytes(data: bytes) -> JPEGInfo:
    """Read JPEG data from bytes and extract all DCT coefficients.

    Args:
        data: Raw JPEG file data.

    Returns:
        JPEGInfo object.

    Raises:
        ValueError: If the data is not a valid JPEG.
    """
    if len(data) < 2 or data[0] != 0xFF or data[1] != SOI:
        raise ValueError("Not a valid JPEG file (missing SOI marker)")

    parser = _JPEGParser(data)
    return parser.parse()


# ---------------------------------------------------------------------------
# Coefficient preparation functions for steganography detectors
# ---------------------------------------------------------------------------

def prepare_normal(info: JPEGInfo) -> tuple[list[int], int]:
    """Extract all non-0/1 DCT coefficients from all components.

    Processes in natural order: component 0 all blocks, then component 1, etc.
    Skips coefficients where (val & 1) == val (i.e., val is 0 or 1).
    Skips DC coefficients (index 0 in each block).

    Args:
        info: JPEGInfo with decoded coefficients.

    Returns:
        (list of coefficient values, total count).
    """
    coeffs = []
    for comp in info.components:
        for by in range(comp.blocks_y):
            for bx in range(comp.blocks_x):
                for k in range(1, 64):  # Skip DC (index 0)
                    val = comp.coeffs[by, bx, k]
                    # Skip if val & 1 == val, meaning val is 0 or 1
                    if (val & 1) != val:
                        coeffs.append(val)
    return coeffs, len(coeffs)


def prepare_jsteg(info: JPEGInfo) -> tuple[list[int], int]:
    """Extract coefficients in JSteg embedding order.

    JSteg embeds in all non-zero AC coefficients in MCU order.
    The embedding order follows the natural MCU scan order, processing
    each 8x8 block's AC coefficients in zigzag order.

    Args:
        info: JPEGInfo with decoded coefficients.

    Returns:
        (list of coefficient values, total count).
    """
    coeffs = []
    for comp in info.components:
        for by in range(comp.blocks_y):
            for bx in range(comp.blocks_x):
                for k in range(1, 64):  # Skip DC, process AC in zigzag order
                    val = comp.coeffs[by, bx, k]
                    if val != 0:
                        coeffs.append(val)
    return coeffs, len(coeffs)


def prepare_outguess(info: JPEGInfo) -> tuple[list[int], int]:
    """Extract coefficients in Outguess embedding order.

    Outguess uses natural order, skipping 0, 1, and DC coefficients.
    This is similar to prepare_normal but follows Outguess's specific
    traversal pattern.

    Args:
        info: JPEGInfo with decoded coefficients.

    Returns:
        (list of coefficient values, total count).
    """
    coeffs = []
    for comp in info.components:
        for by in range(comp.blocks_y):
            for bx in range(comp.blocks_x):
                for k in range(1, 64):  # Skip DC
                    val = comp.coeffs[by, bx, k]
                    # Skip 0 and 1
                    if val != 0 and val != 1 and val != -1:
                        coeffs.append(val)
    return coeffs, len(coeffs)


def prepare_outguess_detect(info: JPEGInfo) -> tuple[list[int], int]:
    """Extract ALL non-zero AC coefficients for Outguess detection.

    Unlike prepare_outguess, this includes ±1 coefficients, which are
    essential for detecting Outguess embedding patterns. Outguess modifies
    the LSB of ALL non-zero AC coefficients (including ±1), so excluding
    them loses significant detection signal.

    This function is for DETECTION only. For password cracking (stegbreak),
    use prepare_outguess which correctly excludes ±1 (Outguess skips these
    when embedding).

    Args:
        info: JPEGInfo with decoded coefficients.

    Returns:
        (list of coefficient values, total count).
    """
    coeffs = []
    for comp in info.components:
        for by in range(comp.blocks_y):
            for bx in range(comp.blocks_x):
                for k in range(1, 64):  # Skip DC
                    val = comp.coeffs[by, bx, k]
                    # Include ALL non-zero AC: Outguess modifies their LSBs
                    if val != 0:
                        coeffs.append(val)
    return coeffs, len(coeffs)


def prepare_steghide(info: JPEGInfo) -> tuple[list[int], int, list[int]]:
    """Extract ALL non-zero AC coefficients for steghide detection.

    Steghide works by identifying "exchangeable" coefficient pairs
    (values differing by exactly 1) and swapping them based on message bits.
    It operates on all non-zero AC coefficients across all blocks.

    The returned block_starts list marks the start index of each block's
    coefficients in the flat data array, enabling block-level analysis.

    Args:
        info: JPEGInfo with decoded coefficients.

    Returns:
        (flat coefficient list, total count, block_start_indices list).
    """
    coeffs = []
    block_starts = []
    for comp in info.components:
        for by in range(comp.blocks_y):
            for bx in range(comp.blocks_x):
                block_starts.append(len(coeffs))
                for k in range(1, 64):  # Skip DC
                    val = comp.coeffs[by, bx, k]
                    if val != 0:
                        coeffs.append(val)
    return coeffs, len(coeffs), block_starts


def prepare_jphide(info: JPEGInfo) -> tuple[list[int], int]:
    """Extract coefficients in JPHide embedding order.

    JPHide uses a lookup table (ltab) to determine embedding positions.
    The lookup table defines positions in zigzag order for each 8x8 block.
    Skip IV protection area and special patterns.

    The embedding algorithm:
    1. Iterate through blocks in natural order (component 0, then 1, etc.)
    2. For each block, use the ltab to determine which zigzag position to use
    3. Skip the first few blocks (IV protection area)
    4. Skip coefficients that are 0 or 1

    Args:
        info: JPEGInfo with decoded coefficients.

    Returns:
        (list of coefficient values, total count).
    """
    coeffs = []

    # JPHide protection area: skip first N blocks
    # The IV (initialization vector) protection uses the first few blocks
    iv_blocks = 10  # Number of blocks to skip for IV protection

    # Build the full lookup table
    ltab = _JPHIDE_LTAB

    # Count total blocks across all components
    total_blocks = 0
    block_starts = []  # (component_idx, by, bx) for each block
    for comp_idx, comp in enumerate(info.components):
        for by in range(comp.blocks_y):
            for bx in range(comp.blocks_x):
                block_starts.append((comp_idx, by, bx))
                total_blocks += 1

    if total_blocks <= iv_blocks:
        return coeffs, 0

    # Iterate through blocks starting after IV protection
    ltab_idx = 0
    block_idx = iv_blocks

    while block_idx < total_blocks and ltab_idx < len(ltab):
        comp_idx, by, bx = block_starts[block_idx]
        comp = info.components[comp_idx]

        zigzag_pos, skip = ltab[ltab_idx % len(ltab)]

        if 0 < zigzag_pos < 64:
            val = comp.coeffs[by, bx, zigzag_pos]
            if val != 0 and val != 1 and val != -1:
                coeffs.append(val)

        # Advance to next block
        block_idx += 1 + skip
        ltab_idx += 1

    return coeffs, len(coeffs)


# ---------------------------------------------------------------------------
# JSteg size detection
# ---------------------------------------------------------------------------

def jsteg_size(coeffs: list[int], bits: int, offset: int = 0) -> int:
    """Read the embedded data length from JSteg coefficients.

    JSteg stores the embedded data length in the LSBs of the first few
    coefficients. The format is:
    - First 5 bits: width of the length field (in bits)
    - Remaining bits: the actual length value

    Args:
        coeffs: List of coefficient values (from prepare_jsteg).
        bits: Number of bits per coefficient (should be 1 for LSB).
        offset: Starting offset in the coefficient list.

    Returns:
        The embedded data size in bytes.
    """
    if len(coeffs) < offset + 32:
        return 0

    # Read the first 5 bits to get the length width
    width = 0
    for i in range(5):
        idx = offset + i
        if idx >= len(coeffs):
            return 0
        width = (width << 1) | (abs(coeffs[idx]) & 1)

    if width == 0 or width > 32:
        return 0

    # Read the length value (width bits)
    length = 0
    for i in range(width):
        idx = offset + 5 + i
        if idx >= len(coeffs):
            return 0
        length = (length << 1) | (abs(coeffs[idx]) & 1)

    return length


# ---------------------------------------------------------------------------
# Randomness test
# ---------------------------------------------------------------------------

def is_random(data: bytes) -> bool:
    """Check if data appears random using byte popcount histogram chi-square test.

    Uses 64 buckets for popcount values (0-7 mapped to 8 buckets).
    The test checks if the distribution of bit counts in the data
    is consistent with random data.

    Args:
        data: Bytes to test.

    Returns:
        True if data appears random, False otherwise.
    """
    if len(data) < 16:
        return False

    # Count popcount (number of set bits) for each nibble (4 bits)
    # 16 possible popcount values for a byte (0-8), but we use nibbles (0-4)
    # Map to 8 buckets: popcount % 8
    buckets = [0] * 8
    total_nibbles = 0

    for byte in data:
        # Process each nibble
        for shift in (0, 4):
            nibble = (byte >> shift) & 0x0F
            popcount = bin(nibble).count('1')
            bucket = popcount % 8
            buckets[bucket] += 1
            total_nibbles += 1

    if total_nibbles == 0:
        return False

    # Expected count per bucket for random data
    # For a random nibble (4 bits), the popcount distribution is:
    # 0 bits: 1/16, 1 bit: 4/16, 2 bits: 6/16, 3 bits: 4/16, 4 bits: 1/16
    # Mapping to 8 buckets (popcount % 8):
    # bucket 0: popcount 0, 8 -> 1/16 (only 0 is possible for nibble)
    # bucket 1: popcount 1, 9 -> 4/16
    # bucket 2: popcount 2, 10 -> 6/16
    # bucket 3: popcount 3, 11 -> 4/16
    # bucket 4: popcount 4, 12 -> 1/16
    # bucket 5: popcount 5 -> 0 (not possible for nibble)
    # bucket 6: popcount 6 -> 0
    # bucket 7: popcount 7 -> 0
    expected_probs = [1/16, 4/16, 6/16, 4/16, 1/16, 0, 0, 0]

    # Chi-square test
    chi2 = 0.0
    for i in range(8):
        expected = total_nibbles * expected_probs[i]
        if expected > 0:
            diff = buckets[i] - expected
            chi2 += (diff * diff) / expected

    # Degrees of freedom: number of non-zero expected buckets - 1
    df = sum(1 for p in expected_probs if p > 0) - 1
    if df <= 0:
        return False

    # For a chi-square distribution with df degrees of freedom,
    # the expected value is df. We use a threshold of df * 3
    # (very lenient) to detect obviously non-random data.
    # For more accurate detection, use scipy.stats.chi2.sf()
    threshold = df * 3.0

    return chi2 < threshold


# ---------------------------------------------------------------------------
# Utility: coefficient statistics
# ---------------------------------------------------------------------------

def coefficient_stats(info: JPEGInfo) -> dict:
    """Compute statistics about the DCT coefficients.

    Args:
        info: JPEGInfo with decoded coefficients.

    Returns:
        Dictionary with coefficient statistics.
    """
    all_coeffs = []
    total_blocks = 0
    nonzero_ac = 0
    total_ac = 0

    for comp in info.components:
        total_blocks += comp.blocks_y * comp.blocks_x
        for by in range(comp.blocks_y):
            for bx in range(comp.blocks_x):
                for k in range(64):
                    val = comp.coeffs[by, bx, k]
                    all_coeffs.append(val)
                    if k > 0:  # AC
                        total_ac += 1
                        if val != 0:
                            nonzero_ac += 1

    arr = np.array(all_coeffs, dtype=np.int16)
    result = {
        "total_blocks": total_blocks,
        "total_coefficients": len(all_coeffs),
        "total_ac": total_ac,
        "nonzero_ac": nonzero_ac,
        "capacity_normal": 0,  # filled below
        "capacity_jsteg": nonzero_ac,
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "nonzero_ratio": nonzero_ac / max(total_ac, 1),
    }

    # Normal capacity (non-0/1 AC coefficients)
    normal_count = 0
    for comp in info.components:
        for by in range(comp.blocks_y):
            for bx in range(comp.blocks_x):
                for k in range(1, 64):
                    val = comp.coeffs[by, bx, k]
                    if (val & 1) != val:
                        normal_count += 1
    result["capacity_normal"] = normal_count

    return result
