"""
F5 steganography detection via recompression analysis.

F5 detection works by:
1. Estimating the original JPEG quality from quantization tables
2. Decompressing to spatial domain, then recompressing at the SAME quality
3. Comparing DCT coefficient histograms between original and recompressed
4. F5 embedding "shrinks" the histogram (fewer zeros, more small values),
   which the beta formula detects as a difference between original and
   recompressed histograms.

Key insight: recompressing at the WRONG quality creates massive DCT
differences that drown out the F5 signal. Must use estimated original quality.
"""

import numpy as np
from PIL import Image
import io
from .jpeg_dct import read_jpeg, read_jpeg_bytes, JPEGInfo, Component
from .dct import dct_to_matrix, dct_inverse, matrix_to_dct
from .detectors import DetectionResult


# Standard IJG luminance quantization table at quality 50
# (zigzag order as used internally)
_STD_LUM_Q50 = np.array([
    16, 11, 10, 16, 24,  40,  51,  61,
    12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,
    14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,
    24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
], dtype=np.float64)


def _estimate_jpeg_quality(qtable: np.ndarray) -> int:
    """Estimate JPEG quality from a quantization table.

    Uses the IJG standard quantization table: for a given quality q,
    the table entries are:
      qtable[i] = max(1, round(std[i] * scale / 100))
    where scale = 5000/q if q < 50 else 200 - 2*q

    Returns the quality value (1-100) that minimizes L1 error.
    Returns 75 as fallback if estimation fails.
    """
    if qtable is None or len(qtable) != 64:
        return 75

    best_quality = 75
    best_error = float('inf')
    qt = qtable.astype(np.float64)

    for q in range(1, 101):
        if q < 50:
            scale = 5000.0 / q
        else:
            scale = 200.0 - 2.0 * q
        expected = np.maximum(1, np.round(_STD_LUM_Q50 * scale / 100.0))
        error = np.sum(np.abs(qt - expected))
        if error < best_error:
            best_error = error
            best_quality = q

    return best_quality


def _dequantize_block(coeffs: np.ndarray, quant_table: np.ndarray) -> np.ndarray:
    """Dequantize a 64-element DCT block.
    out[j] = in[j] * quantval[j]
    """
    return coeffs.astype(np.float64) * quant_table.astype(np.float64)


def _quantize_block(block: np.ndarray, quant_table: np.ndarray) -> np.ndarray:
    """Quantize a 64-element DCT block.
    out[j] = round(in[j] / quantval[j])
    """
    return np.round(block / quant_table.astype(np.float64)).astype(np.int16)


def _build_luminance_image(info: JPEGInfo) -> np.ndarray:
    """Reconstruct luminance image from DCT coefficients.
    Returns 2D numpy array of pixel values (float64).
    """
    comp = info.components[0]  # Y component
    quant = info.quant_tables[comp.quant_table_idx]

    h = comp.blocks_y * 8
    w = comp.blocks_x * 8
    image = np.zeros((h, w), dtype=np.float64)

    for by in range(comp.blocks_y):
        for bx in range(comp.blocks_x):
            coeffs = comp.coeffs[by, bx]  # 64 elements, zigzag order
            # Dequantize
            deq = _dequantize_block(coeffs, quant)
            # Convert to 8x8 matrix
            mat = dct_to_matrix(deq)
            # IDCT
            pixels = dct_inverse(mat)
            # Add level shift (+128)
            pixels += 128.0
            # Clip
            pixels = np.clip(pixels, 0, 255)
            # Place in image
            y_start = by * 8
            x_start = bx * 8
            image[y_start:y_start+8, x_start:x_start+8] = pixels

    return image


def _crop_image(image: np.ndarray, margin: int = 4) -> np.ndarray:
    """Crop margin pixels from each edge."""
    return image[margin:-margin, margin:-margin]


def _blur_image(image: np.ndarray, d: float = 0.05) -> np.ndarray:
    """Apply 3x3 mean filter.
    d controls the center weight: center=1-8*d, neighbors=d
    """
    from scipy.ndimage import uniform_filter
    # 3x3 uniform filter gives each pixel the average of its 3x3 neighborhood
    # With d=0.05: center weight = 1 - 8*0.05 = 0.6, each neighbor = 0.05
    result = uniform_filter(image, size=3, mode='reflect')
    # Blend: result = d * (3x3 avg) + (1-9*d) * original
    # Actually for d=0.05: center=0.6, neighbors=0.05
    # uniform_filter gives equal weight 1/9 to all 9 pixels
    # We want: center * original + d * sum(neighbors)
    # = (1-8*d) * original + d * sum(neighbors)
    # = (1-8*d) * original + d * (9*avg - original)
    # = (1-9*d) * original + 9*d*avg
    result = (1 - 9*d) * image + 9*d * result
    return result


def _compress_to_jpeg(image: np.ndarray, quality: int) -> bytes:
    """Compress a numpy image array to JPEG bytes."""
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def _hkl(info: JPEGInfo, ik: int, il: int, val: int) -> int:
    """Count DCT coefficients at position (ik, il) with value val.
    ik, il are the 2D frequency indices (0-7 each).
    val is the coefficient value.
    """
    # Convert 2D position to zigzag index
    ZIGZAG = [
        (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
        (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
        (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
        (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
        (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
        (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
        (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
    ]
    zigzag_idx = None
    for idx, (r, c) in enumerate(ZIGZAG):
        if r == ik and c == il:
            zigzag_idx = idx
            break
    if zigzag_idx is None:
        return 0

    comp = info.components[0]
    count = 0
    for by in range(comp.blocks_y):
        for bx in range(comp.blocks_x):
            if comp.coeffs[by, bx, zigzag_idx] == val:
                count += 1
    return count


def _beta_kl(orig_info: JPEGInfo, est_info: JPEGInfo, k: int, l: int) -> float:
    """Estimate beta (embedding rate) for DCT position (k, l).

    beta = (h1*(orig0-est0) + (orig1-est1)*(est2-est1)) / (est1^2 + (est2-est1)^2)
    """
    # Get histogram values at position (k,l)
    est0 = _hkl(est_info, k, l, 0)
    est1 = _hkl(est_info, k, l, 1)
    est2 = _hkl(est_info, k, l, 2)
    orig0 = _hkl(orig_info, k, l, 0)
    orig1 = _hkl(orig_info, k, l, 1)

    h1 = est1 + est2

    denom = est1 * est1 + (est2 - est1) ** 2
    if denom == 0:
        return 0.0

    beta = (h1 * (orig0 - est0) + (orig1 - est1) * (est2 - est1)) / denom
    return max(0.0, beta)


def detect_f5_slow(info: JPEGInfo, quality_range: range = range(90, 99)) -> DetectionResult:
    """Slow F5 detection via recompression analysis.

    Uses the Fridrich et al. approach:
    1. Estimate original JPEG quality from quantization tables
    2. Recompress at the estimated quality (same DCT grid, no crop/blur)
    3. Compute beta (embedding rate estimate) at key DCT positions
    4. F5 shrinks the histogram (fewer zeros, more small magnitudes);
       beta measures this shrinkage relative to a clean recompression.

    The key positions (1,2), (2,1), (2,2) are chosen because F5
    preferentially embeds in low-frequency AC coefficients.

    Args:
        info: JPEGInfo of the original image
        quality_range: (unused, kept for API compatibility)

    Returns:
        DetectionResult with estimated embedding rate, or None
    """
    result = DetectionResult()
    result.stego_type = "f5"

    # Step 1: Estimate original JPEG quality
    if info.quant_tables and info.quant_tables[0] is not None:
        est_quality = _estimate_jpeg_quality(info.quant_tables[0])
    else:
        est_quality = 75  # fallback

    # Step 2: Build luminance image from DCT coefficients
    lum = _build_luminance_image(info)

    # Step 3: Recompress at estimated quality (same DCT grid, no crop/blur)
    # Using the SAME quality avoids quantization mismatch artifacts that
    # were the primary cause of false positives in the old implementation.
    try:
        jpeg_bytes = _compress_to_jpeg(lum, est_quality)
        est_info = read_jpeg_bytes(jpeg_bytes)
    except Exception:
        return None

    # Step 4: Compute beta at F5-relevant DCT positions
    # Position (1,2) is the primary F5 embedding position
    # Positions (2,1) and (2,2) provide confirmation
    target_positions = [(1, 2), (2, 1), (2, 2)]
    betas = {}
    for k, l in target_positions:
        b = _beta_kl(info, est_info, k, l)
        if not (np.isnan(b) or np.isinf(b)):
            betas[(k, l)] = b

    if not betas:
        return None

    # Step 5: Score computation
    # For a clean JPEG at the correct quality, beta ≈ 0 at all positions.
    # F5 embedding creates positive beta, strongest at (1,2).
    beta_12 = betas.get((1, 2), 0.0)
    beta_21 = betas.get((2, 1), 0.0)
    beta_22 = betas.get((2, 2), 0.0)

    # Primary score: (1,2) is the strongest F5 indicator
    # Threshold at 0.20: clean mmm.jpg has 0.00, F5 guess.jpg has 0.53
    if beta_12 > 0.20:
        score_primary = min(1.0, (beta_12 - 0.20) / 1.5)
    else:
        score_primary = 0.0

    # Secondary score: combined signal from all three positions
    mean_beta = np.mean(list(betas.values()))
    if mean_beta > 0.35:
        score_secondary = min(1.0, (mean_beta - 0.35) / 1.2)
    else:
        score_secondary = 0.0

    # Consensus: count positions with non-trivial beta signal
    active_positions = sum(1 for b in betas.values() if b > 0.25)

    # Weighted combination (primary position + overall mean)
    combined = 0.5 * score_primary + 0.5 * score_secondary

    # Consensus penalty: single-position signals are unreliable
    if active_positions < 2:
        combined *= 0.3
    elif active_positions == 2:
        combined *= 0.7

    result.confidence = min(combined, 1.0)
    result.estimated_bits = int(mean_beta * sum(
        c.blocks_y * c.blocks_x * 63 for c in info.components
    ))
    result.details = (
        f"Q={est_quality}, beta(1,2)={beta_12:.3f}, "
        f"beta(2,1)={beta_21:.3f}, "
        f"beta(2,2)={beta_22:.3f}, "
        f"active={active_positions}"
    )

    if result.confidence > 0.12:
        return result
    return None
