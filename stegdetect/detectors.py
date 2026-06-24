"""
Steganography detectors for JPEG images.

Implements detection for: jsteg, outguess, jphide, invisible secrets, f5.
Based on chi-square analysis of DCT coefficient histograms.
"""

import numpy as np
from typing import Optional
from .jpeg_dct import read_jpeg, prepare_normal, prepare_jsteg, prepare_outguess, prepare_outguess_detect, prepare_jphide, prepare_steghide
from .histogram import (
    build_dct_hist, unify_normal, unify_false_jsteg, unify_false_outguess,
    unify_outguess, unify_outguess_relaxed, unify_outguess_k2,
    unify_jphide, unify_false_jphide, unify_steghide, unify_steghide_block_level,
    NORM_OUTGUESS, JPHPOS
)
from .chi2 import chi2_test


# Detection result
class DetectionResult:
    """Result of steganography detection."""
    def __init__(self):
        self.stego_type: str = ""       # "jsteg", "outguess", "jphide", "invis", "f5"
        self.confidence: float = 0.0    # 0.0 to 1.0
        self.estimated_bits: int = 0   # estimated embedded data size in bits
        self.details: str = ""         # additional info

    def __repr__(self):
        return f"DetectionResult(type={self.stego_type}, confidence={self.confidence:.4f}, bits={self.estimated_bits})"


def _binsearch(data, bits, unify_fn, threshold=0.5):
    """Binary search for optimal analysis step size.

    Starts from maximum step size, accumulates chi-square values.
    If accumulated value exceeds threshold, terminates early.
    Otherwise, binary searches for smaller step sizes.

    Returns (percent, sum_value).
    """
    max_percent = 100
    min_percent = 1

    best_percent = max_percent
    best_sum = 0.0

    for percent in range(max_percent, min_percent - 1, -1):
        step = max(1, bits * percent // 100)
        sum_val = 0.0

        for start in range(0, bits, step):
            end = min(start + step, bits)
            hist = build_dct_hist(data, start, end)
            theo, obs, discard = unify_fn(hist)
            if len(theo) == 0:
                continue
            val = chi2_test(theo, obs, len(theo), discard)
            sum_val += val

        if sum_val > threshold:
            return percent, sum_val

        if sum_val > best_sum:
            best_sum = sum_val
            best_percent = percent

    return best_percent, best_sum


def detect_jsteg(data: list[int], bits: int, scale: float = 1.0) -> Optional[DetectionResult]:
    """Detect JSteg steganography using chi-square analysis.

    Algorithm:
    1. Binary search for optimal step size
    2. Calibrate with false_jsteg pairing
    3. Scan with normal pairing, accumulate chi-square
    4. Cross-validate with jsteg_size

    Args:
        data: DCT coefficients in JSteg order
        bits: number of coefficients
        scale: sensitivity scale factor

    Returns:
        DetectionResult if steganography detected, None otherwise
    """
    if bits < 100:
        return None

    # Binary search for optimal step
    percent, _ = _binsearch(data, bits, unify_false_jsteg)

    step = max(1, bits * percent // 100)
    sum_val = 0.0
    max_pos = 0
    max_val = 0.0
    decay = 0

    for start in range(0, bits, step):
        end = min(start + step, bits)
        hist = build_dct_hist(data, start, end)
        theo, obs, discard = unify_normal(hist)
        if len(theo) == 0:
            continue
        val = chi2_test(theo, obs, len(theo), discard)
        sum_val += val

        if val > max_val:
            max_val = val
            max_pos = start
        elif val < max_val * 0.5:
            decay += 1

    result = DetectionResult()
    result.stego_type = "jsteg"
    result.confidence = scale * sum_val / (2 * percent) if percent > 0 else 0

    # Estimate embedded bits
    if max_pos > 0:
        result.estimated_bits = max_pos
    else:
        result.estimated_bits = int(sum_val * bits / 100)

    if result.confidence > 0.1:
        return result
    return None


def _outguess_lsb_chi2(data: list[int], bits: int) -> float:
    """LSB-based chi-square analysis for outguess detection.

    Outguess embeds data by modifying LSBs of DCT coefficients.
    This test checks if the LSB distribution is consistent with
    random (steganographic) embedding vs natural JPEG statistics.

    Uses a sliding window approach with outguess pairing.
    """
    if bits < 50:
        return 0.0

    # Use multiple window sizes and take the best result
    best_score = 0.0

    for pct in [1, 2, 5, 10, 20]:
        step = max(1, bits * pct // 100)
        if step < 10:
            continue

        total_val = 0.0
        count = 0

        for start in range(0, bits, step):
            end = min(start + step, bits)
            hist = build_dct_hist(data, start, end)
            theo, obs, discard = unify_outguess_relaxed(hist)
            if len(theo) == 0:
                continue
            val = chi2_test(theo, obs, len(theo), discard)
            total_val += val
            count += 1

        if count > 0:
            score = total_val / count
            best_score = max(best_score, score)

    return best_score


def _outguess_pair_asymmetry(data: list[int], bits: int) -> float:
    """Pair asymmetry analysis for outguess detection.

    Outguess modifies coefficient values by +/-1, which creates asymmetry
    between adjacent histogram bins. This test measures the overall
    asymmetry of (2i-1, 2i) pairs used by outguess.

    Natural JPEG images tend to have symmetric pairs; steganographic
    embedding breaks this symmetry.
    """
    if bits < 50:
        return 0.0

    # Build full histogram
    hist = build_dct_hist(data, 0, bits)

    total_asymmetry = 0.0
    total_weight = 0.0

    for i in range(1, 128):
        if i in (64, 65):
            continue

        f = hist[2 * i - 1]
        fbar = hist[2 * i]

        # Weight by the sum of the pair (more data = more reliable)
        pair_sum = f + fbar
        if pair_sum < 1.0:
            continue

        # Asymmetry: how different are the two sides?
        # For natural images, f ≈ fbar (symmetric)
        # For stego images, embedding shifts values creating asymmetry
        asymmetry = abs(f - fbar) / pair_sum
        total_asymmetry += asymmetry * pair_sum
        total_weight += pair_sum

    if total_weight < 10.0:
        return 0.0

    avg_asymmetry = total_asymmetry / total_weight

    # Normalize: natural images typically have avg_asymmetry < 0.15
    # Stego images tend to have higher asymmetry (Outguess modifies LSBs)
    if avg_asymmetry > 0.15:
        return min(1.0, (avg_asymmetry - 0.15) / 0.4)
    return 0.0


def _outguess_rs_analysis(data: list[int], bits: int) -> float:
    """RS (Regular-Singular) analysis adapted for outguess detection.

    Measures the proportion of regular, singular, and unusable groups
    in the coefficient LSBs. Steganographic embedding increases the
    proportion of singular groups and decreases regular groups.

    Groups of 4 consecutive coefficients are classified as:
    - Regular (R): flipping LSBs of the middle two preserves smoothness
    - Singular (S): flipping LSBs of the middle two disrupts smoothness
    - Unusable (U): neither

    The difference R - S decreases with steganographic embedding.
    """
    if bits < 50:
        return 0.0

    def _smoothness(a, b, c, d):
        """Measure smoothness of a group of 4 values."""
        return abs(a - b) + abs(b - c) + abs(c - d)

    r_count = 0
    s_count = 0
    u_count = 0

    # Process groups of 4 consecutive coefficients
    for i in range(0, bits - 3, 4):
        a, b, c, d = data[i], data[i+1], data[i+2], data[i+3]

        # Original smoothness
        f_orig = _smoothness(a, b, c, d)

        # Smoothness after flipping LSBs of b and c
        b_flip = b ^ 1 if b >= 0 else -((-b) ^ 1)
        c_flip = c ^ 1 if c >= 0 else -((-c) ^ 1)
        f_flip = _smoothness(a, b_flip, c_flip, d)

        if f_flip > f_orig:
            r_count += 1  # Regular
        elif f_flip < f_orig:
            s_count += 1  # Singular
        else:
            u_count += 1  # Unusable

    total = r_count + s_count + u_count
    if total < 20:
        return 0.0

    # RS ratio: for natural images, R > S
    # For stego images, R approaches S
    r_ratio = r_count / total
    s_ratio = s_count / total

    # The difference R - S is the key indicator
    rs_diff = r_ratio - s_ratio

    # Natural images typically have rs_diff > 0.1
    # Stego images tend to have rs_diff close to 0 or negative
    if rs_diff < 0.05:
        return min(1.0, (0.05 - rs_diff) / 0.15)
    return 0.0


def _outguess_lsb_randomness(data: list[int], bits: int) -> float:
    """LSB randomness test for outguess detection.

    Measures the randomness of LSBs in the coefficient sequence.
    Outguess embedding introduces random-looking LSB patterns
    that differ from natural JPEG coefficient statistics.

    Uses runs test on LSB sequence: counts runs of identical bits.
    Random data has more runs; structured data has fewer.
    """
    if bits < 50:
        return 0.0

    # Extract LSBs
    lsbs = [abs(v) & 1 for v in data[:bits]]

    # Count runs
    if not lsbs:
        return 0.0

    runs = 1
    for i in range(1, len(lsbs)):
        if lsbs[i] != lsbs[i-1]:
            runs += 1

    n = len(lsbs)
    n0 = lsbs.count(0)
    n1 = lsbs.count(1)

    if n0 == 0 or n1 == 0:
        return 0.0

    # Expected runs for random sequence: E(R) = (2*n0*n1)/(n0+n1) + 1
    expected_runs = (2.0 * n0 * n1) / n + 1.0

    # Variance: Var(R) = 2*n0*n1*(2*n0*n1 - n) / (n^2 * (n-1))
    if n > 1:
        variance = (2.0 * n0 * n1 * (2.0 * n0 * n1 - n)) / (n * n * (n - 1))
        if variance > 0:
            std = variance ** 0.5
            z_score = (runs - expected_runs) / std
            # |z| < 1.96 means the sequence is consistent with random
            # Stego tends to produce more random-looking LSBs
            if abs(z_score) < 1.5:
                return min(1.0, (1.5 - abs(z_score)) / 1.5)

    return 0.0


def _outguess_extended_chi2(data: list[int], bits: int) -> float:
    """Extended chi2 (k=2) sliding window analysis for Outguess detection.

    This is the approach used by the original stegdetect C implementation:
    instead of standard k=1 pairing (two adjacent histogram bins), it uses
    k=2 grouping (four consecutive bins) with a sliding window.

    Outguess preserves k=1 chi-square by design (pair-wise LSB balance),
    but the extended k=2 grouping reveals embedding artifacts because
    shifting LSBs between adjacent pairs within a 4-bin group creates
    detectable statistical deviations that Outguess cannot control.

    Uses a sliding window approach similar to strategy 1, but with
    unify_outguess_k2 pairing and multiple window sizes.

    Returns a score in [0, 1]; higher = more likely Outguess.
    """
    if bits < 50:
        return 0.0

    best_score = 0.0

    # Try multiple window sizes: from small (finer granularity) to large
    for pct in [2, 5, 10, 20, 50]:
        step = max(1, bits * pct // 100)
        if step < 10:
            continue

        total_val = 0.0
        count = 0

        for start in range(0, bits, step):
            end = min(start + step, bits)
            hist = build_dct_hist(data, start, end)
            theo, obs, discard_ratio = unify_outguess_k2(hist)
            if len(theo) == 0:
                continue
            val = chi2_test(theo, obs, len(theo), discard_ratio)
            total_val += val
            count += 1

        if count > 0:
            score = total_val / count
            best_score = max(best_score, score)

    return best_score


def detect_outguess(data: list[int], bits: int, scale: float = 1.0) -> Optional[DetectionResult]:
    """Detect Outguess steganography using multi-strategy analysis.

    Uses multiple complementary detection strategies:
    1. Original chi-square sliding window (with relaxed pairing fallback)
    2. LSB-based chi-square analysis
    3. Pair asymmetry analysis
    4. RS (Regular-Singular) analysis
    5. LSB randomness test
    6. Extended chi2 (k=2) analysis — original stegdetect approach

    Each strategy produces an independent score. The final confidence
    is a weighted combination, with extended chi2 and asymmetry
    given highest weight as they are most effective against Outguess's
    chi2-preserving embedding.

    Args:
        data: DCT coefficients in Outguess order (ALL non-zero AC, including ±1)
        bits: number of coefficients
        scale: sensitivity scale factor

    Returns:
        DetectionResult if steganography detected, None otherwise
    """
    if bits < 50:
        return None

    result = DetectionResult()
    result.stego_type = "outguess"
    result.details = ""

    # Strategy 1: Original chi-square with outguess pairing
    chi2_score = 0.0
    percent, _ = _binsearch(data, bits, unify_false_outguess)
    window = max(1, bits * percent // 100)
    sum_val = 0.0

    for start in range(0, bits, window):
        end = min(start + window, bits)
        hist = build_dct_hist(data, start, end)
        theo, obs, discard_ratio = unify_outguess(hist)
        if len(theo) == 0:
            # Fallback to relaxed pairing for sparse histograms
            theo, obs, discard_ratio = unify_outguess_relaxed(hist)
            if len(theo) == 0:
                continue
        val = chi2_test(theo, obs, len(theo), discard_ratio)
        norm_idx = min(int(sum_val * 2), len(NORM_OUTGUESS) - 1)
        val *= NORM_OUTGUESS[norm_idx]
        sum_val += val

    chi2_score = sum_val

    # Strategy 2: LSB chi-square
    lsb_chi2_score = _outguess_lsb_chi2(data, bits)

    # Strategy 3: Pair asymmetry
    asymmetry_score = _outguess_pair_asymmetry(data, bits)

    # Strategy 4: RS analysis
    rs_score = _outguess_rs_analysis(data, bits)

    # Strategy 5: LSB randomness
    randomness_score = _outguess_lsb_randomness(data, bits)

    # Strategy 6: Extended chi2 (k=2 grouping)
    ext_chi2_score = _outguess_extended_chi2(data, bits)

    # Weighted combination
    # Extended chi2 (k=2) is the most reliable for Outguess detection
    # as it catches what k=1 preserves. Asymmetry and RS provide
    # independent complementary signals.
    if bits < 5000:
        weights = {
            'chi2': 0.10, 'lsb_chi2': 0.15, 'asymmetry': 0.20,
            'rs': 0.15, 'randomness': 0.10, 'ext_chi2': 0.30,
        }
    else:
        weights = {
            'chi2': 0.10, 'lsb_chi2': 0.15, 'asymmetry': 0.20,
            'rs': 0.15, 'randomness': 0.10, 'ext_chi2': 0.30,
        }

    combined = (
        weights['chi2'] * min(chi2_score, 1.0) +
        weights['lsb_chi2'] * lsb_chi2_score +
        weights['asymmetry'] * asymmetry_score +
        weights['rs'] * rs_score +
        weights['randomness'] * randomness_score +
        weights['ext_chi2'] * ext_chi2_score
    )

    result.confidence = scale * combined

    # Build details string
    details_parts = []
    if chi2_score > 0.01:
        details_parts.append(f"chi2={chi2_score:.3f}")
    if lsb_chi2_score > 0.01:
        details_parts.append(f"lsb_chi2={lsb_chi2_score:.3f}")
    if asymmetry_score > 0.01:
        details_parts.append(f"asym={asymmetry_score:.3f}")
    if rs_score > 0.01:
        details_parts.append(f"rs={rs_score:.3f}")
    if randomness_score > 0.01:
        details_parts.append(f"rand={randomness_score:.3f}")
    if ext_chi2_score > 0.01:
        details_parts.append(f"ext_chi2={ext_chi2_score:.3f}")
    result.details = ", ".join(details_parts)

    if result.confidence > 0.1:
        return result
    return None


# ---------------------------------------------------------------------------
# Steghide detection strategies
#
# Steghide preserves the global first-order histogram by SWAPPING values
# rather than modifying bits. Detection strategies focus on SECONDARY
# effects: local chi2 anomalies, spatial redistribution of exchangeable
# pairs, and window-level chi2 variance.
#
# The chi2_test with unify_steghide pairing returns values in [0,1] where
# HIGHER = histogram imbalance in a local region. Since steghide preserves
# the GLOBAL histogram, we look for:
# 1. High local/window chi2 (value redistribution creates local imbalances)
# 2. High variance of chi2 across windows (some windows normal, some anomalous)
# 3. Global chi2 lower than local average (preservation vs redistribution gap)
# ---------------------------------------------------------------------------


def _steghide_window_chi2_stats(data: list[int], bits: int) -> tuple[float, float, float]:
    """Collect chi2 statistics across multiple sliding windows.

    Runs chi2_test with steghide pairing on windows at 3 scales (2%, 5%, 10%).
    Returns (mean_chi2, spread, max_chi2).

    mean_chi2: average chi2 across all windows
    spread: max_chi2 - min_chi2, indicates uneven redistribution
    max_chi2: maximum chi2 observed
    """
    if bits < 100:
        return (0.0, 0.0, 0.0)

    all_vals = []
    for pct in [2, 5, 10]:
        step = max(1, bits * pct // 100)
        if step < 10:
            continue
        for start in range(0, bits, step):
            end = min(start + step, bits)
            if end - start < 50:
                continue
            hist = build_dct_hist(data, start, end)
            theo, obs, dr = unify_steghide(hist)
            if len(theo) == 0:
                continue
            all_vals.append(chi2_test(theo, obs, len(theo), dr))

    if len(all_vals) < 10:
        return (0.0, 0.0, 0.0)

    mean_val = sum(all_vals) / len(all_vals)
    min_val = min(all_vals)
    max_val = max(all_vals)
    spread = max_val - min_val

    return (mean_val, spread, max_val)


def _steghide_local_chunk_chi2(data: list[int], bits: int) -> tuple[float, float]:
    """Compute chi2 on fixed-size chunks spanning the coefficient sequence.

    Returns (avg_local_chi2, global_chi2).
    """
    if bits < 200:
        return (0.0, 0.0)

    # Global
    gh = build_dct_hist(data, 0, bits)
    gt, go, gd = unify_steghide(gh)
    global_chi2 = chi2_test(gt, go, len(gt), gd) if len(gt) > 0 else 0.0

    # Local chunks
    chunk_size = max(bits // 5, 200)
    if chunk_size < 100:
        return (0.0, global_chi2)

    chunk_scores = []
    for start in range(0, bits, chunk_size):
        end = min(start + chunk_size, bits)
        if end - start < 100:
            continue
        h = build_dct_hist(data, start, end)
        t, o, d = unify_steghide(h)
        if len(t) > 0:
            chunk_scores.append(chi2_test(t, o, len(t), d))

    if len(chunk_scores) < 3:
        return (0.0, global_chi2)

    avg_local = sum(chunk_scores) / len(chunk_scores)
    return (avg_local, global_chi2)


def detect_steghide(
    data: list[int], bits: int, block_starts: list[int], scale: float = 1.0
) -> Optional[DetectionResult]:
    """Detect steghide steganography.

    Steghide swaps exchangeable DCT coefficient pairs (|v1-v2|==1),
    preserving the global first-order histogram exactly. Detection uses
    SECONDARY effects of value redistribution:

    Strategy 1 (window_chi2_mean): Mean chi2 across small sliding windows.
        Steghide creates local histogram imbalances → high mean.
    Strategy 2 (window_chi2_spread): Spread of window chi2 values.
        Steghide creates uneven redistribution → high spread.
    Strategy 3 (local_vs_global): Comparison of chunk-level vs global chi2.
        Steghide preserves global but disrupts local → gap.

    These strategies produce chi2_test values in [0,1] where higher
    indicates more anomaly. For pure steghide images, global_chi2 ≈ 0
    but window/local chi2 is high.

    For images with mixed embedding (e.g., steghide after F5), global
    histogram may NOT be fully preserved, but local chi2 anomalies
    still provide a signal.

    Returns DetectionResult or None.
    """
    if bits < 100:
        return None

    # Strategy 1+2: Window-level chi2 statistics
    win_mean, win_spread, win_max = _steghide_window_chi2_stats(data, bits)

    # Strategy 3: Local vs global chi2
    local_chi2, global_chi2 = _steghide_local_chunk_chi2(data, bits)

    # ── Score computation ──
    # Score 1: Window mean chi2. Threshold at 0.12 (well above mmm.jpg's 0.06)
    if win_mean > 0.12:
        score_mean = min(1.0, (win_mean - 0.12) / 0.40)
    else:
        score_mean = 0.0

    # Score 2: Window chi2 spread. Threshold at 0.30 (mmm.jpg has 0.27)
    if win_spread > 0.30:
        score_spread = min(1.0, (win_spread - 0.30) / 0.35)
    else:
        score_spread = 0.0

    # Score 3: Local chi2 magnitude. Threshold at 0.10.
    if local_chi2 > 0.10:
        score_local = min(1.0, (local_chi2 - 0.10) / 0.35)
    else:
        score_local = 0.0

    # Score 4: Global-vs-local gap bonus
    # If global chi2 is low (< 0.05) but local chi2 is high, that's
    # a strong steghide signature (pure histogram-preserving embedding).
    # If only local is high (global also high), still a moderate signal.
    gap_bonus = 0.0
    if local_chi2 > 0.10:
        if global_chi2 < 0.05:
            gap_bonus = 0.15  # Strong pure-steghide bonus
        elif local_chi2 > global_chi2 * 1.5:
            gap_bonus = 0.05  # Moderate gap bonus

    # Weighted combination (weights sum to 1.0 + gap)
    base_score = (
        0.35 * score_mean +
        0.25 * score_spread +
        0.40 * score_local
    )
    combined = min(base_score + gap_bonus, 1.0)

    result = DetectionResult()
    result.stego_type = "steghide"
    result.confidence = scale * combined

    # Build details
    details_parts = []
    if score_mean > 0.01:
        details_parts.append(f"win_mean={win_mean:.3f}")
    if score_spread > 0.01:
        details_parts.append(f"spread={win_spread:.3f}")
    if score_local > 0.01:
        details_parts.append(f"local={local_chi2:.3f}")
    if gap_bonus > 0:
        details_parts.append(f"gap_bonus")
    if not details_parts:
        details_parts.append("negative")
    result.details = ", ".join(details_parts)

    if result.confidence > 0.1:
        return result
    return None


def detect_jphide(data: list[int], bits: int, scale: float = 1.0) -> Optional[DetectionResult]:
    """Detect JPHide steganography using multiple analysis techniques.

    Algorithm:
    1. Check image size requirements
    2. Chi-square test at specific positions (JPHPOS)
    3. Run-length analysis
    4. Zero/one analysis
    5. Empty pair analysis

    Args:
        data: DCT coefficients in JPHide order
        bits: number of coefficients
        scale: sensitivity scale factor

    Returns:
        DetectionResult if steganography detected, None otherwise
    """
    if bits < 500:
        return None

    result = DetectionResult()
    result.stego_type = "jphide"
    total_score = 0.0

    # Chi-square test at JPHide positions
    for pos in JPHPOS:
        if pos >= bits:
            break
        hist = build_dct_hist(data, 0, pos)
        theo, obs, discard = unify_jphide(hist)
        if len(theo) == 0:
            continue
        val = chi2_test(theo, obs, len(theo), discard)
        if val >= 0.9:
            total_score += val

    # Run-length analysis
    run_score = _jphide_runlength(data, bits)
    total_score += run_score

    # Zero/one analysis
    zo_score = _jphide_zero_one(data, bits)
    total_score += zo_score

    # Empty pair analysis
    ep_score = _jphide_empty_pair(data, bits)
    total_score += ep_score

    result.confidence = scale * total_score / 4.0

    if result.confidence > 0.1:
        return result
    return None


def _jphide_runlength(data: list[int], bits: int) -> float:
    """Run-length analysis for JPHide detection.
    Counts consecutive identical coefficient values.
    Steganography tends to break up natural run-lengths.
    """
    if bits < 10:
        return 0.0

    runs = []
    current_run = 1
    for i in range(1, bits):
        if data[i] == data[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)

    if not runs:
        return 0.0

    # Short runs indicate possible steganography
    short_runs = sum(1 for r in runs if r <= 2)
    ratio = short_runs / len(runs)

    return ratio if ratio > 0.5 else 0.0


def _jphide_zero_one(data: list[int], bits: int) -> float:
    """Zero/one analysis for JPHide detection.
    Checks distribution of -1, 0, +1 coefficients.
    Steganography tends to increase the count of these values.
    """
    count = 0
    for i in range(bits):
        if abs(data[i]) <= 1:
            count += 1

    ratio = count / bits if bits > 0 else 0
    # Normal images have ~10-20% of coefficients as -1, 0, 1
    # Stego images tend to have more
    if ratio > 0.3:
        return min(1.0, (ratio - 0.3) / 0.3)
    return 0.0


def _jphide_empty_pair(data: list[int], bits: int) -> float:
    """Empty pair analysis for JPHide detection.
    Checks for pairs where one side is zero.
    """
    empty = 0
    pairs = 0

    for i in range(0, bits - 1, 2):
        pairs += 1
        if data[i] == 0 or data[i+1] == 0:
            empty += 1

    if pairs == 0:
        return 0.0

    ratio = empty / pairs
    if ratio > 0.3:
        return min(1.0, (ratio - 0.3) / 0.3)
    return 0.0


def detect_invisible(data: list[int], bits: int, scale: float = 1.0) -> Optional[DetectionResult]:
    """Detect Invisible Secrets steganography.
    Similar to JSteg but uses different coefficient ordering.

    Args:
        data: DCT coefficients in normal order
        bits: number of coefficients
        scale: sensitivity scale factor

    Returns:
        DetectionResult if steganography detected, None otherwise
    """
    if bits < 100:
        return None

    percent, sum_val = _binsearch(data, bits, unify_normal)

    result = DetectionResult()
    result.stego_type = "invis"
    result.confidence = scale * sum_val / (2 * percent) if percent > 0 else 0

    if result.confidence > 0.1:
        return result
    return None


def detect_f5_simple(info) -> Optional[DetectionResult]:
    """Simple F5 detection - checks for F5 comment in JPEG header.

    Args:
        info: JPEGInfo object

    Returns:
        DetectionResult if F5 comment found, None otherwise
    """
    comment = info.comment.decode('latin-1', errors='ignore')
    if 'F5' in comment or 'f5' in comment.lower():
        result = DetectionResult()
        result.stego_type = "f5"
        result.confidence = 1.0
        result.details = "F5 signature found in JPEG comment"
        return result
    return None


# Detection flags
FLAG_JSTEG = 0x01
FLAG_OUTGUESS = 0x02
FLAG_JPHIDE = 0x04
FLAG_INVIS = 0x08
FLAG_F5 = 0x10
FLAG_F5_SLOW = 0x20
FLAG_STEGHIDE = 0x40
FLAG_ALL = FLAG_JSTEG | FLAG_OUTGUESS | FLAG_JPHIDE | FLAG_INVIS | FLAG_F5 | FLAG_F5_SLOW | FLAG_STEGHIDE


def detect(filename: str, flags: int = FLAG_ALL, scale: float = 1.0) -> list[DetectionResult]:
    """Main detection entry point.

    Detect steganography in a JPEG image using specified detectors.

    Args:
        filename: path to JPEG file
        flags: bitwise OR of FLAG_* constants
        scale: sensitivity scale factor (default 1.0, higher = more sensitive)

    Returns:
        List of DetectionResult objects, sorted by confidence descending
    """
    results = []

    try:
        info = read_jpeg(filename)
    except Exception as e:
        return results

    # Simple F5 detection (check comment)
    if flags & FLAG_F5:
        r = detect_f5_simple(info)
        if r:
            results.append(r)

    # Slow F5 detection (recompression analysis)
    if flags & FLAG_F5_SLOW:
        try:
            from .f5 import detect_f5_slow
            r = detect_f5_slow(info)
            if r:
                results.append(r)
        except ImportError:
            pass

    # JSteg detection
    if flags & FLAG_JSTEG:
        data, bits = prepare_jsteg(info)
        if bits > 0:
            r = detect_jsteg(data, bits, scale)
            if r:
                results.append(r)

    # Outguess detection
    if flags & FLAG_OUTGUESS:
        data, bits = prepare_outguess_detect(info)
        if bits > 0:
            r = detect_outguess(data, bits, scale)
            if r:
                results.append(r)

    # JPHide detection
    if flags & FLAG_JPHIDE:
        data, bits = prepare_jphide(info)
        if bits > 0:
            r = detect_jphide(data, bits, scale)
            if r:
                results.append(r)

    # Invisible Secrets detection
    if flags & FLAG_INVIS:
        data, bits = prepare_normal(info)
        if bits > 0:
            r = detect_invisible(data, bits, scale)
            if r:
                results.append(r)

    # Steghide detection
    if flags & FLAG_STEGHIDE:
        data, bits, block_starts = prepare_steghide(info)
        if bits > 0:
            r = detect_steghide(data, bits, block_starts, scale)
            if r:
                results.append(r)

    # Sort by confidence descending
    results.sort(key=lambda r: r.confidence, reverse=True)
    return results
