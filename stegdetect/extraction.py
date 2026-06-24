"""
Feature extraction transforms for steganography classification.

Implements multiple transforms that extract statistical features from DCT
coefficients for use with the Fisher LDA classifier.
"""

import numpy as np
from typing import Optional
from .jpeg_dct import read_jpeg, JPEGInfo


def compute_stats(data: list[float], size: int) -> tuple[float, float, float, float]:
    """Compute mean, standard deviation, skewness, and excess kurtosis.

    Args:
        data: array of values
        size: number of values to use

    Returns:
        (mean, std, skew, kurtosis)
    """
    if size == 0:
        return 0.0, 0.0, 0.0, 0.0

    arr = np.array(data[:size], dtype=np.float64)
    n = len(arr)
    mean = np.mean(arr)

    if n < 2:
        return mean, 0.0, 0.0, 0.0

    std = np.std(arr, ddof=1)

    if std == 0:
        return mean, std, 0.0, 0.0

    # Skewness
    m3 = np.mean((arr - mean) ** 3)
    skew = m3 / (std ** 3)

    # Excess kurtosis
    m4 = np.mean((arr - mean) ** 4)
    kurt = m4 / (std ** 4) - 3.0

    return mean, std, skew, kurt


def build_histogram(values: list[int], nbins: int = 64) -> list[float]:
    """Build a histogram of integer values.

    Args:
        values: integer values
        nbins: number of bins

    Returns:
        Histogram array of nbins floats
    """
    if not values:
        return [0.0] * nbins

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        hist = [0.0] * nbins
        hist[nbins // 2] = float(len(values))
        return hist

    hist = [0.0] * nbins
    range_val = max_val - min_val

    for v in values:
        idx = int((v - min_val) * nbins / (range_val + 1))
        idx = min(idx, nbins - 1)
        hist[idx] += 1.0

    return hist


def spline_setup(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    """Set up natural cubic spline interpolation.

    Args:
        x: x coordinates (must be sorted)
        y: y coordinates

    Returns:
        (b, c) coefficients for the spline
    """
    n = len(x)
    if n < 2:
        return [], []

    h = [x[i+1] - x[i] for i in range(n-1)]

    if n == 2:
        b = [(y[1] - y[0]) / h[0]]
        c = [0.0]
        return b, c

    # Solve tridiagonal system for natural spline
    # c[0] = c[n-1] = 0
    alpha = [0.0] * n
    for i in range(1, n-1):
        alpha[i] = 3.0 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    # Forward elimination
    c_arr = [0.0] * n
    l = [0.0] * n
    mu = [0.0] * n
    z = [0.0] * n

    l[0] = 1.0
    for i in range(1, n-1):
        l[i] = 2.0 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        if l[i] == 0:
            l[i] = 1e-10
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    l[n-1] = 1.0

    # Back substitution
    for j in range(n-2, 0, -1):
        c_arr[j] = z[j] - mu[j] * c_arr[j+1]

    b = [(y[i+1] - y[i]) / h[i] - h[i] * (c_arr[i+1] + 2*c_arr[i]) / 3.0
         for i in range(n-1)]

    return b, c_arr


def spline_interpolate(x: list[float], y: list[float], xval: float) -> float:
    """Evaluate cubic spline at xval.

    Args:
        x: x coordinates
        y: y coordinates
        xval: point to evaluate

    Returns:
        Interpolated value
    """
    n = len(x)
    if n == 0:
        return 0.0
    if n == 1:
        return y[0]

    # Find interval
    idx = 0
    for i in range(n-1):
        if x[i] <= xval <= x[i+1]:
            idx = i
            break
    else:
        if xval < x[0]:
            return y[0]
        return y[-1]

    h = x[idx+1] - x[idx]
    if h == 0:
        return y[idx]

    b, c = spline_setup(x, y)
    if not b:
        return y[idx]

    t = (xval - x[idx]) / h
    # Linear interpolation as fallback
    return y[idx] + b[idx] * t


def esterror2(hist: list[float], size: int, x: float) -> float:
    """Estimate smoothing error at position x using spline interpolation.

    Args:
        hist: histogram values
        size: number of bins
        x: position to evaluate

    Returns:
        Estimated error
    """
    if size < 3:
        return 0.0

    xs = list(range(size))
    ys = hist[:size]

    # Get spline value
    spline_val = spline_interpolate(
        [float(v) for v in xs], ys, x
    )

    # Error = |actual - spline|
    ix = int(x)
    if 0 <= ix < size:
        actual = hist[ix]
        return abs(actual - spline_val)
    return 0.0


def distribution(info: JPEGInfo, freq_pos: int) -> Optional[tuple[float, float, float, float]]:
    """Extract statistical features for a specific DCT frequency position.

    Args:
        info: JPEGInfo object
        freq_pos: zigzag frequency position (1-63, skip DC)

    Returns:
        (mean, std, skew, kurtosis) or None
    """
    if not info.components:
        return None

    comp = info.components[0]
    values = []

    for by in range(comp.blocks_y):
        for bx in range(comp.blocks_x):
            val = comp.coeffs[by, bx, freq_pos]
            values.append(val)

    if not values:
        return None

    return compute_stats(values, len(values))


def spline_transform(info: JPEGInfo) -> tuple[np.ndarray, int]:
    """Spline transform: extract features from DCT coefficient histograms.

    For the first 18 DCT frequency positions, compute histogram,
    then use spline interpolation to estimate error distribution.
    Extract 4 statistics per position: mean, std, skew, kurtosis.
    Total: 18 * 4 = 72 features.

    Args:
        info: JPEGInfo object

    Returns:
        (feature_vector, num_points)
    """
    features = []

    for freq in range(1, 19):  # First 18 AC positions
        comp = info.components[0]
        values = []

        for by in range(comp.blocks_y):
            for bx in range(comp.blocks_x):
                val = comp.coeffs[by, bx, freq]
                values.append(val)

        if not values:
            features.extend([0.0] * 4)
            continue

        hist = build_histogram(values, nbins=64)

        # Compute spline error statistics
        errors = []
        for x in range(64):
            err = esterror2(hist, 64, float(x))
            errors.append(err)

        mean, std, skew, kurt = compute_stats(errors, len(errors))
        features.extend([mean, std, skew, kurt])

    return np.array(features, dtype=np.float64), 72


def gradient_transform(info: JPEGInfo) -> tuple[np.ndarray, int]:
    """Gradient transform: compute horizontal gradient then apply spline transform.

    Computes difference between adjacent blocks' DCT coefficients,
    then extracts spline features from the gradient.

    Args:
        info: JPEGInfo object

    Returns:
        (feature_vector, num_points)
    """
    if not info.components:
        return np.array([]), 0

    comp = info.components[0]
    if comp.blocks_x < 2:
        return np.array([]), 0

    # Compute horizontal gradient
    gradient_values = []

    for freq in range(1, 19):  # First 18 AC positions
        for by in range(comp.blocks_y):
            for bx in range(comp.blocks_x - 1):
                val = comp.coeffs[by, bx + 1, freq] - comp.coeffs[by, bx, freq]
                gradient_values.append(val)

    if not gradient_values:
        return np.array([]), 0

    hist = build_histogram(gradient_values, nbins=64)
    errors = []
    for x in range(64):
        err = esterror2(hist, 64, float(x))
        errors.append(err)

    mean, std, skew, kurt = compute_stats(errors, len(errors))
    features = [mean, std, skew, kurt]

    return np.array(features, dtype=np.float64), 4


def roughness_transform(info: JPEGInfo) -> tuple[np.ndarray, int]:
    """Roughness transform: compute frequency-weighted roughness.

    roughness = sqrt(sum((u^2+v^2)*val^2) / sum(u^2+v^2))
    where (u,v) are the 2D frequency indices.

    Extracts 4 statistics across all blocks.

    Args:
        info: JPEGInfo object

    Returns:
        (feature_vector, num_points)
    """
    if not info.components:
        return np.array([]), 0

    comp = info.components[0]

    # 2D frequency indices for zigzag positions
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

    roughness_values = []

    for by in range(comp.blocks_y):
        for bx in range(comp.blocks_x):
            num = 0.0
            den = 0.0
            for idx in range(1, 64):  # Skip DC
                u, v = ZIGZAG[idx]
                freq_sq = u * u + v * v
                val = comp.coeffs[by, bx, idx]
                num += freq_sq * val * val
                den += freq_sq
            if den > 0:
                roughness_values.append((num / den) ** 0.5)

    if not roughness_values:
        return np.array([]), 0

    mean, std, skew, kurt = compute_stats(roughness_values, len(roughness_values))
    features = [mean, std, skew, kurt]

    return np.array(features, dtype=np.float64), 4


def diffsquare_transform(info: JPEGInfo) -> tuple[np.ndarray, int]:
    """Difference-of-squares transform.

    For each row and column in the 8x8 block, compute the sum of
    squared differences between adjacent coefficients.
    Extracts 4 statistics per direction.
    Total: (8 rows + 8 columns) * 4 = 64 features.

    Args:
        info: JPEGInfo object

    Returns:
        (feature_vector, num_points)
    """
    if not info.components:
        return np.array([]), 0

    comp = info.components[0]

    features = []

    for by in range(comp.blocks_y):
        for bx in range(comp.blocks_x):
            # Row differences
            for row in range(8):
                diffs_sq = 0.0
                for col in range(7):
                    idx1 = row * 8 + col
                    idx2 = row * 8 + col + 1
                    diff = comp.coeffs[by, bx, idx1] - comp.coeffs[by, bx, idx2]
                    diffs_sq += diff * diff
                features.append(diffs_sq)

            # Column differences
            for col in range(8):
                diffs_sq = 0.0
                for row in range(7):
                    idx1 = row * 8 + col
                    idx2 = (row + 1) * 8 + col
                    diff = comp.coeffs[by, bx, idx1] - comp.coeffs[by, bx, idx2]
                    diffs_sq += diff * diff
                features.append(diffs_sq)

    if not features:
        return np.array([]), 0

    mean, std, skew, kurt = compute_stats(features, len(features))
    return np.array([mean, std, skew, kurt], dtype=np.float64), 4


# Transform registry
TRANSFORMS = {
    'spline': spline_transform,
    'gradient': gradient_transform,
    'roughness': roughness_transform,
    'diffsquare': diffsquare_transform,
}


def extract_features(filename: str, transform_name: str = 'spline') -> Optional[np.ndarray]:
    """Extract features from a JPEG file using the specified transform.

    Args:
        filename: path to JPEG file
        transform_name: name of the transform to use

    Returns:
        Feature vector, or None if extraction fails
    """
    try:
        info = read_jpeg(filename)
    except Exception:
        return None

    if transform_name not in TRANSFORMS:
        return None

    features, npoints = TRANSFORMS[transform_name](info)
    return features
