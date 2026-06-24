from typing import Callable

def build_dct_hist(data: list[int], start: int = 0, end: int | None = None) -> list[float]:
    """Build DCT coefficient histogram.
    Index 0 = value -128, index 256 = value +127.
    Range: [-128, 127].

    Args:
        data: DCT coefficient values
        start: start index in data
        end: end index in data (None = len(data))

    Returns:
        Histogram array of 257 floats.
    """
    hist = [0.0] * 257
    if end is None:
        end = len(data)
    for i in range(start, end):
        val = data[i]
        if -128 <= val <= 127:
            hist[val + 128] += 1.0
    return hist

def unify_normal(hist: list[float]) -> tuple[list[float], list[float], int]:
    """Normal LSB steganography pairing.
    For i from 0 to 127, skip i=64 (which is DCT value 0):
      theo[k] = (hist[2*i] + hist[2*i+1]) / 2
      obs[k] = hist[2*i+1]

    Returns (theo, obs, discard_count).
    """
    theo = []
    obs = []
    discard = 0
    for i in range(128):
        if i == 64:
            discard += 1
            continue
        t = (hist[2 * i] + hist[2 * i + 1]) / 2.0
        theo.append(t)
        obs.append(hist[2 * i + 1])
    return theo, obs, discard

def unify_false_jsteg(hist: list[float]) -> tuple[list[float], list[float], int]:
    """JSteg false positive calibration pairing.
    Similar to unify_normal but skips i=0, 64, 65.

    Returns (theo, obs, discard_count).
    """
    theo = []
    obs = []
    discard = 0
    for i in range(128):
        if i in (0, 64, 65):
            discard += 1
            continue
        t = (hist[2 * i] + hist[2 * i + 1]) / 2.0
        theo.append(t)
        obs.append(hist[2 * i + 1])
    return theo, obs, discard

def unify_false_outguess(hist: list[float]) -> tuple[list[float], list[float], int]:
    """Outguess false positive calibration pairing.
    Skips i=0, 64, 65.
    theo[k] = (hist[2*i-1] + hist[2*i]) / 2
    obs[k] = hist[2*i]

    Returns (theo, obs, discard_count).
    """
    theo = []
    obs = []
    discard = 0
    for i in range(128):
        if i in (0, 64, 65):
            discard += 1
            continue
        t = (hist[2 * i - 1] + hist[2 * i]) / 2.0
        theo.append(t)
        obs.append(hist[2 * i])
    return theo, obs, discard

def unify_outguess(hist: list[float]) -> tuple[list[float], list[float], float]:
    """Outguess-specific pairing with adaptive discard.
    For each pair, check if the smaller value > larger/4.
    Also check (f - f/3) - (fbar + f/3) > 0.
    Pairs that fail are discarded.

    Returns (theo, obs, discard_ratio).
    """
    theo = []
    obs = []
    discard = 0
    total = 0

    for i in range(1, 128):
        if i in (64, 65):
            continue
        total += 1

        f = hist[2 * i - 1]
        fbar = hist[2 * i]

        # Determine smaller and larger
        smaller = min(f, fbar)
        larger = max(f, fbar)

        if smaller <= larger / 4.0:
            discard += 1
            continue

        if (f - f / 3.0) - (fbar + f / 3.0) <= 0:
            discard += 1
            continue

        theo.append((f + fbar) / 2.0)
        obs.append(fbar)

    discard_ratio = discard / total if total > 0 else 0.0
    return theo, obs, discard_ratio


def unify_outguess_relaxed(hist: list[float]) -> tuple[list[float], list[float], float]:
    """Relaxed Outguess pairing for small images / sparse histograms.

    Only discards pairs where both values are zero (no information).
    This prevents excessive discard that causes empty theo/obs arrays
    on small images where the DCT histogram is very sparse.

    Returns (theo, obs, discard_ratio).
    """
    theo = []
    obs = []
    discard = 0
    total = 0

    for i in range(1, 128):
        if i in (64, 65):
            continue
        total += 1

        f = hist[2 * i - 1]
        fbar = hist[2 * i]

        # Only discard if both are zero — no information at all
        if f == 0.0 and fbar == 0.0:
            discard += 1
            continue

        theo.append((f + fbar) / 2.0)
        obs.append(fbar)

    discard_ratio = discard / total if total > 0 else 0.0
    return theo, obs, discard_ratio


def unify_outguess_k2(hist: list[float]) -> tuple[list[float], list[float], float]:
    """Extended chi2 (k=2) pairing for Outguess detection.

    Original stegdetect's approach: instead of pairing 2 adjacent bins
    (k=1), it groups 4 consecutive bins into pairs and tests for symmetry.

    Outguess preserves k=1 chi2 (pair-wise balance) by design, but the
    coarser k=2 grouping can detect embedding artifacts because the
    embedding shifts counts between adjacent pairs within the 4-bin window.

    For each group of 4 consecutive histogram bins:
      bin0 = hist[2i-1] + hist[2i]   (first pair)
      bin1 = hist[2i+1] + hist[2i+2] (second pair)
    Tests whether bin0 ≈ bin1 (symmetric within the 4-value window).

    Groups touching values -1, 0, 1 are skipped to avoid JPEG baseline
    artifacts at the zero boundary.

    Returns (theo, obs, discard_ratio).
    """
    theo = []
    obs = []
    discard = 0
    total = 0

    # Step by 2: each iteration covers 4 consecutive histogram bins
    for i in range(1, 127, 2):
        # Skip groups that include the sensitive bins around zero
        # i=63: bins 125,126,127,128 → values -3,-2,-1,0
        # i=64: bins 127,128,129,130 → values -1,0,1,2
        # i=65: bins 129,130,131,132 → values 1,2,3,4
        if i in (63, 64, 65):
            continue
        total += 1

        # First pair: odd + even
        count_0 = hist[2 * i - 1] + hist[2 * i]
        # Second pair: odd + even
        count_1 = hist[2 * i + 1] + hist[2 * i + 2]

        # Skip if both pairs are empty
        if count_0 == 0.0 and count_1 == 0.0:
            discard += 1
            continue

        # For extended chi2: expected = average of both pairs
        expected = (count_0 + count_1) / 2.0
        theo.append(expected)
        obs.append(count_1)

    discard_ratio = discard / total if total > 0 else 0.0
    return theo, obs, discard_ratio


def unify_jphide(hist: list[float]) -> tuple[list[float], list[float], int]:
    """JPHide-specific pairing.
    Excludes values -1, 0, +1 (histogram indices 127, 128, 129).
    For negative values: use odd indices
    For positive values: use even indices

    Returns (theo, obs, discard_count).
    """
    theo = []
    obs = []
    discard = 0

    # Process negative values (indices 0-126, representing -128 to -2)
    for i in range(0, 127, 2):
        if i + 1 >= 127:
            break
        t = (hist[i] + hist[i + 1]) / 2.0
        theo.append(t)
        obs.append(hist[i + 1])

    # Process positive values (indices 130-256, representing +2 to +127)
    for i in range(130, 257, 2):
        if i + 1 >= 257:
            break
        t = (hist[i] + hist[i + 1]) / 2.0
        theo.append(t)
        obs.append(hist[i + 1])

    return theo, obs, discard

def unify_false_jphide(hist: list[float]) -> tuple[list[float], list[float], int]:
    """JPHide false positive calibration.
    Similar to unify_jphide but with different pairing.

    Returns (theo, obs, discard_count).
    """
    theo = []
    obs = []
    discard = 0

    # For negative values
    for i in range(0, 127, 2):
        if i + 1 >= 127:
            break
        t = (hist[i] + hist[i + 1]) / 2.0
        theo.append(t)
        obs.append(hist[i])

    # For positive values
    for i in range(130, 257, 2):
        if i + 1 >= 257:
            break
        t = (hist[i] + hist[i + 1]) / 2.0
        theo.append(t)
        obs.append(hist[i])

    return theo, obs, discard

def unify_steghide(hist: list[float]) -> tuple[list[float], list[float], float]:
    """Steghide-specific adjacent-value pairing for chi-square analysis.

    Steghide embeds data by swapping DCT coefficient values that differ by
    exactly 1 (exchangeable pairs). This pairs adjacent histogram bins
    corresponding to these exchangeable value pairs.

    Pairing pattern:
    - Negative side: (-3,-2), (-5,-4), (-7,-6), ...  (odd values with next even)
    - Positive side: (1,2), (3,4), (5,6), ...        (odd values with next even)

    Pairs involving -1, 0, +1 are excluded:
    - (-1,0) and (1,0): 0 is not used by steghide
    - (-2,-1): -1 is a boundary case affected by JPEG baseline

    Uses relaxed discard strategy suitable for steghide's histogram-preserving
    embedding (global frequencies don't change, only spatial distribution).

    Returns (theo, obs, discard_ratio).
    """
    theo = []
    obs = []
    discard = 0
    total = 0

    for i in range(1, 128):
        # Skip histogram indices involving -1, 0, +1
        # hist[127]=-1, hist[128]=0, hist[129]=1, hist[130]=2
        if i in (63, 64, 65):
            continue
        total += 1

        f = hist[2 * i - 1]    # odd value
        fbar = hist[2 * i]     # even value (odd+1)

        # Only discard if both are zero — sparse coefficient histograms
        if f == 0.0 and fbar == 0.0:
            discard += 1
            continue

        theo.append((f + fbar) / 2.0)
        obs.append(fbar)

    discard_ratio = discard / total if total > 0 else 0.0
    return theo, obs, discard_ratio


def unify_steghide_block_level(block_coeffs: list[int]) -> tuple[list[float], list[float], float]:
    """Block-level steghide exchangeable pair analysis.

    Within a single 8x8 block, counts how many pairs of AC coefficients
    have values differing by exactly 1 (exchangeable for steghide).

    For each exchangeable pair (v1, v2) where v2 = v1 + 1, we test whether
    the proportion of such pairs within the block matches the expected
    proportion for natural JPEG images.

    Returns (theo, obs, discard_ratio) where theo = expected count and
    obs = actual count of exchangeable pairs.
    """
    n = len(block_coeffs)
    if n < 2:
        return [], [], 1.0

    # Count adjacent-value pairs in this block
    exchangeable_count = 0
    total_pairs = 0
    hist = build_dct_hist(block_coeffs)

    # Count how many adjacent histogram bins both have non-zero counts
    for v in range(-127, 127):
        if v == 0 or v == -1:
            continue  # skip pairs involving 0 and (-1,0)
        idx = v + 128
        idx_next = idx + 1
        if 0 <= idx < 257 and 0 <= idx_next < 257:
            f1 = hist[idx]
            f2 = hist[idx_next]
            if f1 > 0 and f2 > 0:
                exchangeable_count += min(f1, f2)

    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return [], [], 1.0

    # Expected exchangeable pairs in natural JPEG: typically very low
    # Steghide increases this proportion
    expected = total_pairs * 0.01  # ~1% baseline expectation

    return [expected], [float(exchangeable_count)], 0.0


# Normalization table for Outguess detection
# Original: norm_outguess[i] = sqrt(i/2) for i in 0..20
NORM_OUTGUESS = [0.0] + [(i / 2.0) ** 0.5 for i in range(1, 21)]

# JPHide position thresholds
JPHPOS = [500, 1000, 2000, 5000, 10000, 20000, 50000]
