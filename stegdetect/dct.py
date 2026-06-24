import numpy as np

# Standard 8x8 DCT-II basis matrix, computed from the formula:
# D[i][j] = C(i) * C(j) * cos((2*j+1)*i*pi/16)
# where C(0) = 1/sqrt(8), C(k) = sqrt(2/8) for k > 0

def _build_dct_matrix():
    """Build the 8x8 orthonormal DCT-II basis matrix.
    D[i][j] = alpha(i) * cos((2*j+1)*i*pi/(2*N))
    where alpha(0) = sqrt(1/N), alpha(k) = sqrt(2/N) for k > 0, N=8
    """
    N = 8
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        alpha = np.sqrt(1.0 / N) if i == 0 else np.sqrt(2.0 / N)
        for j in range(N):
            D[i, j] = alpha * np.cos((2 * j + 1) * i * np.pi / (2 * N))
    return D

_D = _build_dct_matrix()
_DT = _D.T.copy()

def dct_forward(block: np.ndarray) -> np.ndarray:
    """Apply 2D DCT-II to an 8x8 block.
    out = D * in * D^T
    """
    return _D @ block @ _DT

def dct_inverse(block: np.ndarray) -> np.ndarray:
    """Apply 2D IDCT to an 8x8 block.
    out = D^T * in * D
    """
    return _DT @ block @ _D

def dct_to_matrix(dct_coeffs: np.ndarray) -> np.ndarray:
    """Convert 64-element 1D DCT array (zigzag order) to 8x8 matrix.
    Args:
        dct_coeffs: 64-element array in zigzag scan order
    Returns:
        8x8 matrix
    """
    # Zigzag scan order to matrix position mapping
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
    mat = np.zeros((8, 8), dtype=np.float64)
    for idx, (r, c) in enumerate(ZIGZAG):
        mat[r, c] = dct_coeffs[idx]
    return mat

def matrix_to_dct(mat: np.ndarray) -> np.ndarray:
    """Convert 8x8 matrix back to 64-element 1D zigzag order array."""
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
    coeffs = np.zeros(64, dtype=np.float64)
    for idx, (r, c) in enumerate(ZIGZAG):
        coeffs[idx] = mat[r, c]
    return coeffs
