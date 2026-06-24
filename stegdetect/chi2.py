from scipy.stats import chi2 as chi2_dist

def chi2_cdf(chi: float, df: int) -> float:
    """Compute the CDF of the chi-square distribution.
    Uses scipy.stats.chi2.cdf.
    In the original C code, this used a lookup table chitable[256][1024].
    We use scipy directly for accuracy.

    Args:
        chi: chi-square statistic value
        df: degrees of freedom

    Returns:
        CDF value in [0, 1]
    """
    return chi2_dist.cdf(chi, df)

def chi2_test(theo: list[float], obs: list[float], size: int, discard: float) -> float:
    """Core chi-square test function.

    Accumulates theoretical and observed values.
    When accumulated observed >= 5, computes a chi-square term: (ymt - ytt)^2 / ytt
    Returns f * (1 - discard), where f = 1 - chi2cdf(sumchi, dgf-1)

    This corresponds to the original C function:
    float chi2(float *DCTtheo, float *DCTobs, int size, int discard)

    Args:
        theo: theoretical values array
        obs: observed values array
        size: number of pairs
        discard: discard ratio (0.0 to 1.0) or discard count (int)

    Returns:
        Detection value in [0, 1]. Higher = more likely steganography.
        Returns 0.0 if insufficient data for a valid test.
    """
    if size <= 0:
        return 0.0

    ymt = 0.0  # accumulated theoretical
    ytt = 0.0  # accumulated observed
    sumchi = 0.0
    dgf = 0  # degrees of freedom (number of chi-square terms computed)

    for i in range(size):
        ymt += theo[i]
        ytt += obs[i]
        if ytt >= 5.0:
            chi = (ymt - ytt) ** 2 / ytt
            sumchi += chi
            dgf += 1
            ymt = 0.0
            ytt = 0.0

    if dgf <= 0:
        return 0.0

    f = 1.0 - chi2_cdf(sumchi, dgf - 1)

    # Clamp discard to [0, 1] to prevent negative/nan results
    discard_clamped = max(0.0, min(1.0, float(discard)))
    result = f * (1.0 - discard_clamped)

    # Guard against nan/inf
    if result != result or result == float('inf'):
        return 0.0

    return result
